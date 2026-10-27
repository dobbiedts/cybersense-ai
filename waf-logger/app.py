"""
CyberSense AI - WAF logger.

Reads the ModSecurity JSON audit log and records every request that
ModSecurity itself blocked into the shared event database, so the
Cyber Essentials loop sees attacks stopped by the signature layer as
well as those stopped by the AI gate.

    ModSecurity audit log (waf-logs volume) -> waf-logger -> /data/events.db

Rules:
  - Only requests ModSecurity blocked are recorded. CRS blocks through
    rule 949110 (inbound anomaly score exceeded). Requests the AI gate
    blocked also return 403 and appear in the audit log, but they have no
    949110 match and carry the X-CyberSense-Gate header, so they are skipped
    (the gate already records those itself).
  - Attack type comes from the CRS rule tags: attack-sqli (942xxx) and
    attack-xss (941xxx). Other attack classes are ignored for now.
  - One threat_events row per blocked request per attack type, with
    ml_verdict and dl_verdict set to "waf" and no model confidences.
  - Rule IDs and messages go into a waf_events table for the evidence report.
  - Event IDs are built from ModSecurity's unique_id, so re-reading the log
    after a restart never creates duplicates.
"""
import json
import os
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from datetime import datetime

AUDIT_LOG    = os.environ.get("AUDIT_LOG", "/waf-logs/audit.log")
DB_PATH      = os.environ.get("DB_PATH", "/data/events.db")
POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "1"))
CONFIG_URLS  = {
    "sqli": os.environ.get("SQLI_CONFIG_URL", "http://sql-api:5000/config"),
    "xss":  os.environ.get("XSS_CONFIG_URL",  "http://xss-api:5000/config"),
}

BLOCKING_RULES = {"949110"}            # CRS inbound anomaly score exceeded
TAG_TO_ATTACK  = {"attack-sqli": "sqli", "attack-xss": "xss"}
RULE_PREFIX_TO_ATTACK = {"942": "sqli", "941": "xss"}   # CRS rule ID ranges


def attacks_for(details):
    """Attack types for one CRS match: from its tags, or from its rule ID range."""
    found = {TAG_TO_ATTACK[t] for t in (details.get("tags") or []) if t in TAG_TO_ATTACK}
    prefix = str(details.get("ruleId", ""))[:3]
    if prefix in RULE_PREFIX_TO_ATTACK:
        found.add(RULE_PREFIX_TO_ATTACK[prefix])
    return found
SEVERITY_LEVELS = ["low", "medium", "high", "critical"]
MATCH_VALUE = re.compile(r"found within [A-Z_]+:[^:]*: (.*)$", re.DOTALL)


def log(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


# ---------------------------------------------------------------- database
def db():
    return sqlite3.connect(DB_PATH, timeout=10)


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = db()
    # Same schema the detection APIs create
    conn.execute('''
        CREATE TABLE IF NOT EXISTS threat_events (
            event_id      TEXT PRIMARY KEY,
            timestamp     TEXT NOT NULL,
            attack_type   TEXT NOT NULL,
            ml_verdict    TEXT NOT NULL,
            dl_verdict    TEXT NOT NULL,
            final_verdict TEXT NOT NULL,
            ml_confidence REAL,
            dl_confidence REAL,
            payload       TEXT,
            severity      TEXT
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS waf_events (
            event_id    TEXT PRIMARY KEY,
            timestamp   TEXT NOT NULL,
            attack_type TEXT NOT NULL,
            rule_ids    TEXT,
            messages    TEXT,
            method      TEXT,
            uri         TEXT,
            client_ip   TEXT,
            unique_id   TEXT
        )
    ''')
    conn.execute('CREATE TABLE IF NOT EXISTS waf_logger_state (key TEXT PRIMARY KEY, value TEXT)')
    conn.commit()
    conn.close()


def get_offset():
    conn = db()
    row = conn.execute("SELECT value FROM waf_logger_state WHERE key='offset'").fetchone()
    conn.close()
    return int(row[0]) if row else 0


def save_offset(offset):
    conn = db()
    conn.execute("INSERT OR REPLACE INTO waf_logger_state VALUES ('offset', ?)", (str(offset),))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------- risk config
def escalation_for(attack):
    """Use the same severity escalation the detection API is running with."""
    try:
        with urllib.request.urlopen(CONFIG_URLS[attack], timeout=3) as r:
            return max(0, int(json.loads(r.read()).get("severity_escalation", 0)))
    except Exception:
        return 0


def base_severity(crs_levels):
    """CRS severity: 0-2 emergency/alert/critical, 3 error, 4 warning, 5 notice."""
    worst = min(crs_levels) if crs_levels else 5
    if worst <= 2:
        return "high"
    if worst <= 4:
        return "medium"
    return "low"


def escalate(severity, steps):
    i = SEVERITY_LEVELS.index(severity) + steps
    return SEVERITY_LEVELS[min(i, len(SEVERITY_LEVELS) - 1)]


# ---------------------------------------------------------------- parsing
def parse_time(value):
    """ModSecurity time_stamp, e.g. 'Mon Oct 26 22:01:02 2026' (container clock is UTC)."""
    try:
        return datetime.strptime(value, "%a %b %d %H:%M:%S %Y").isoformat() + "Z"
    except Exception:
        return datetime.utcnow().isoformat() + "Z"


def header(headers, name):
    for k, v in (headers or {}).items():
        if k.lower() == name.lower():
            return v
    return None


def blocked_events(entry):
    """
    Return a list of events (one per attack type) for a request ModSecurity
    blocked, or [] if ModSecurity did not block it.
    """
    tx = entry.get("transaction", {})
    messages = tx.get("messages", []) or []
    rule_ids = {str(m.get("details", {}).get("ruleId", "")) for m in messages}

    if not rule_ids & BLOCKING_RULES:
        return []
    if header(tx.get("response", {}).get("headers"), "X-CyberSense-Gate"):
        return []       # the AI gate answered this request, not ModSecurity

    request = tx.get("request", {})
    uri = request.get("uri", "")
    by_attack = {}
    for m in messages:
        d = m.get("details", {})
        attacks = attacks_for(d)
        for attack in attacks:
            e = by_attack.setdefault(attack, {"rules": [], "messages": [], "levels": [], "value": None})
            e["rules"].append(str(d.get("ruleId", "")))
            e["messages"].append(m.get("message", ""))
            try:
                e["levels"].append(int(d.get("severity", 5)))
            except (TypeError, ValueError):
                pass
            match = MATCH_VALUE.search(d.get("data", "") or "")
            if match and not e["value"]:
                e["value"] = match.group(1)

    events = []
    for attack, e in by_attack.items():
        events.append({
            "attack_type": attack,
            "unique_id":   tx.get("unique_id") or f"{tx.get('time_stamp')}-{uri}",
            "timestamp":   parse_time(tx.get("time_stamp", "")),
            "rule_ids":    sorted(set(e["rules"])),
            "messages":    list(dict.fromkeys(e["messages"])),
            "crs_levels":  e["levels"],
            "payload":     e["value"] or urllib.parse.unquote_plus(uri),
            "method":      request.get("method", ""),
            "uri":         uri,
            "client_ip":   tx.get("client_ip", ""),
        })
    return events


def store(events):
    stored = 0
    conn = db()
    for e in events:
        event_id = f"waf-{e['unique_id']}-{e['attack_type']}"
        severity = escalate(base_severity(e["crs_levels"]), escalation_for(e["attack_type"]))
        cur = conn.execute(
            'INSERT OR IGNORE INTO threat_events VALUES (?,?,?,?,?,?,?,?,?,?)',
            (event_id, e["timestamp"], e["attack_type"], "waf", "waf", "threat",
             None, None, e["payload"][:300], severity)
        )
        conn.execute(
            'INSERT OR IGNORE INTO waf_events VALUES (?,?,?,?,?,?,?,?,?)',
            (event_id, e["timestamp"], e["attack_type"], ",".join(e["rule_ids"]),
             json.dumps(e["messages"])[:1000], e["method"], e["uri"][:500],
             e["client_ip"], e["unique_id"])
        )
        if cur.rowcount:
            stored += 1
            log("recorded", attack=e["attack_type"], rules=e["rule_ids"],
                severity=severity, uri=e["uri"][:120])
    conn.commit()
    conn.close()
    return stored


# ---------------------------------------------------------------- main loop
def process_new_lines(offset):
    size = os.path.getsize(AUDIT_LOG)
    if size < offset:
        log("log_truncated", previous_offset=offset, size=size)
        offset = 0
    if size == offset:
        return offset

    with open(AUDIT_LOG, "rb") as f:
        f.seek(offset)
        chunk = f.read()

    last_newline = chunk.rfind(b"\n")
    if last_newline == -1:
        return offset           # wait for the rest of the line
    complete = chunk[:last_newline + 1]

    for raw in complete.splitlines():
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            log("skipped_line", reason="not JSON", start=line[:80])
            continue
        events = blocked_events(entry)
        if events:
            store(events)

    offset += len(complete)
    save_offset(offset)
    return offset


def main():
    init_db()
    offset = get_offset()
    log("started", audit_log=AUDIT_LOG, db=DB_PATH, offset=offset)
    waiting = False
    while True:
        try:
            if not os.path.exists(AUDIT_LOG):
                if not waiting:
                    log("waiting_for_audit_log", path=AUDIT_LOG)
                    waiting = True
            else:
                waiting = False
                offset = process_new_lines(offset)
        except Exception as exc:
            log("error", error=str(exc))
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()