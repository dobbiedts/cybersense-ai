from flask import Flask, jsonify
import sqlite3
import requests
import os
import json
import threading
import time
from datetime import datetime, timedelta

app = Flask(__name__)

DB_PATH      = '/data/events.db'
CE_API_URL   = os.environ.get('CE_API_URL',   'http://compliance-engine:5002/api/assess-evidence')
SQLI_API_URL = os.environ.get('SQLI_API_URL', 'http://sql-api:5000')
XSS_API_URL  = os.environ.get('XSS_API_URL',  'http://xss-api:5000')

# --- Risk-adaptive feedback settings ---
PASS_THRESHOLD = 0.7   # CE score at or above this = control passing
FAIL_THRESHOLD = 0.5   # CE score below this = control failing

# Which CE controls each attack type implicates
ATTACK_CONTROLS = {
    "sqli": ["CE1_firewalls", "CE3_user_access"],
    "xss":  ["CE1_firewalls", "CE2_secure_config", "CE4_malware_protection"],
}

# How the APIs respond at each risk level
RISK_PROFILES = {
    "normal":   {"severity_escalation": 0, "watch_threshold": None},
    "elevated": {"severity_escalation": 1, "watch_threshold": 0.3},
    "critical": {"severity_escalation": 2, "watch_threshold": 0.2},
}

# Re-run the CE assessment after this many new threats, per risk level
REASSESS_AFTER = {"normal": 50, "elevated": 20, "critical": 5}
AUTO_CHECK_SECONDS = int(os.environ.get('AUTO_CHECK_SECONDS', '30'))
RISK_ORDER = ["normal", "elevated", "critical"]


def now_utc():
    return datetime.utcnow().isoformat() + "Z"


def db():
    return sqlite3.connect(DB_PATH)


def init_db():
    os.makedirs('/data', exist_ok=True)
    conn = db()
    conn.execute('CREATE TABLE IF NOT EXISTS ce_scores (timestamp TEXT, scores TEXT)')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS risk_log (
            timestamp       TEXT,
            trigger_source  TEXT,
            sqli_risk       TEXT,
            xss_risk        TEXT,
            sqli_min_score  REAL,
            xss_min_score   REAL,
            events_assessed INTEGER
        )
    ''')
    conn.commit()
    conn.close()


init_db()


# --- Read recent threat events ---
def get_recent_threats(hours=24):
    conn = db()
    since = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"
    try:
        rows = conn.execute('''
            SELECT attack_type, ml_verdict, dl_verdict, final_verdict,
                   ml_confidence, dl_confidence, payload, severity, timestamp
            FROM threat_events
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        ''', (since,)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()

    keys = ["attack_type", "ml_verdict", "dl_verdict", "final_verdict",
            "ml_confidence", "dl_confidence", "payload", "severity", "timestamp"]
    return [dict(zip(keys, r)) for r in rows]


# --- Build evidence document ---
def build_evidence_document(events):
    sqli = [e for e in events if e['attack_type'] == 'sqli']
    xss  = [e for e in events if e['attack_type'] == 'xss']
    high = [e for e in events if e['severity'] in ('high', 'critical')]

    doc = f"""AUTOMATED THREAT DETECTION EVIDENCE REPORT
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC
Source: CyberSense AI Detection Engine
Period: Last 24 hours

SUMMARY
Total threats detected: {len(events)}
SQL Injection attempts: {len(sqli)}
XSS attempts: {len(xss)}
High severity events: {len(high)}
"""

    if sqli:
        both = sum(1 for e in sqli if e['ml_verdict'] == 'threat' and e['dl_verdict'] == 'threat')
        doc += f"""
SQL INJECTION ATTACKS ({len(sqli)} detected)
Both ML and DL models confirmed: {both}
Highest ML confidence: {max(e['ml_confidence'] for e in sqli):.2%}
Highest DL confidence: {max(e['dl_confidence'] for e in sqli):.2%}

Cyber Essentials implications:
- CE Control 1 (Boundary Firewalls): SQL injection reaching the API
  layer indicates insufficient input filtering at the boundary. FAIL.
- CE Control 3 (User Access Control): SQLi attempts suggest database
  accounts may have excessive privileges. FAIL.
- CE Control 2 (Secure Configuration): Raw SQL queries accepted suggests
  application configuration gaps. WARN.
"""

    if xss:
        both = sum(1 for e in xss if e['ml_verdict'] == 'threat' and e['dl_verdict'] == 'threat')
        doc += f"""
CROSS-SITE SCRIPTING ATTACKS ({len(xss)} detected)
Both ML and DL models confirmed: {both}
Highest ML confidence: {max(e['ml_confidence'] for e in xss):.2%}
Highest DL confidence: {max(e['dl_confidence'] for e in xss):.2%}

Cyber Essentials implications:
- CE Control 1 (Boundary Firewalls): XSS payloads reaching the application
  suggest WAF or input sanitisation gaps. FAIL.
- CE Control 2 (Secure Configuration): XSS success indicates output
  encoding is not applied in application config. FAIL.
- CE Control 4 (Malware Protection): Script injection can be used to
  load malicious scripts in user browsers. WARN.
"""
    return doc


# --- Risk-adaptive feedback loop ---
def risk_level_for(min_score):
    if min_score < FAIL_THRESHOLD:
        return "critical"
    if min_score < PASS_THRESHOLD:
        return "elevated"
    return "normal"


def apply_risk_feedback(ce_scores, trigger, events_assessed):
    """
    Detection thresholds stay fixed. CE gaps change how the APIs respond:
    severity escalation and a near-miss watchlist for borderline payloads.
    """
    plan = {}
    for attack, controls in ATTACK_CONTROLS.items():
        min_score = min(float(ce_scores.get(c, 1.0)) for c in controls)
        level = risk_level_for(min_score)
        plan[attack] = {"risk_level": level, "min_ce_score": round(min_score, 3),
                        **RISK_PROFILES[level]}

    pushed = {}
    for attack, url in (("sqli", SQLI_API_URL), ("xss", XSS_API_URL)):
        body = {k: plan[attack][k] for k in ("risk_level", "severity_escalation", "watch_threshold")}
        try:
            r = requests.post(f"{url}/config/risk", json=body, timeout=5)
            pushed[attack] = "updated" if r.ok else f"error {r.status_code}"
        except Exception:
            pushed[attack] = "unreachable"

    conn = db()
    conn.execute(
        'INSERT INTO risk_log VALUES (?,?,?,?,?,?,?)',
        (now_utc(), trigger,
         plan["sqli"]["risk_level"], plan["xss"]["risk_level"],
         plan["sqli"]["min_ce_score"], plan["xss"]["min_ce_score"],
         events_assessed)
    )
    conn.commit()
    conn.close()

    return {"plan": plan, "apis": pushed}


# --- Full assessment cycle (manual or automatic) ---
def run_assessment(trigger="manual"):
    events = get_recent_threats(hours=24)
    if not events:
        return {"status": "no_threats", "events": 0}, 200

    doc = build_evidence_document(events)

    try:
        response = requests.post(CE_API_URL, json={"evidence_text": doc}, timeout=60)
        response.raise_for_status()
        ce_result = response.json()
    except Exception as exc:
        # Never change API behaviour on a failed assessment
        return {"status": "ce_engine_error", "error": str(exc)}, 502

    ce_scores = ce_result.get('ce_scores', {})
    conn = db()
    conn.execute('INSERT INTO ce_scores VALUES (?, ?)', (now_utc(), json.dumps(ce_scores)))
    conn.commit()
    conn.close()

    feedback = apply_risk_feedback(ce_scores, trigger, len(events))

    return {
        "status":           "assessed",
        "trigger":          trigger,
        "events_processed": len(events),
        "evidence_preview": doc[:300],
        "ce_result":        ce_result,
        "feedback_applied": feedback,
    }, 200


# --- Automatic reassessment: faster when risk is higher ---
def current_risk_and_last_assessment():
    conn = db()
    try:
        row = conn.execute(
            'SELECT timestamp, sqli_risk, xss_risk FROM risk_log ORDER BY timestamp DESC LIMIT 1'
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return "normal", None
    worst = max(row[1], row[2], key=RISK_ORDER.index)
    return worst, row[0]


def new_threats_since(timestamp):
    conn = db()
    try:
        if timestamp:
            n = conn.execute('SELECT COUNT(*) FROM threat_events WHERE timestamp > ?',
                             (timestamp,)).fetchone()[0]
        else:
            n = conn.execute('SELECT COUNT(*) FROM threat_events').fetchone()[0]
    except sqlite3.OperationalError:
        n = 0
    conn.close()
    return n


def auto_reassess_loop():
    while True:
        time.sleep(AUTO_CHECK_SECONDS)
        try:
            level, last = current_risk_and_last_assessment()
            new = new_threats_since(last)
            if new >= REASSESS_AFTER[level]:
                result, _ = run_assessment(trigger=f"auto ({new} new threats, risk {level})")
                print(f"[auto-reassess] {result.get('status')} after {new} new threats", flush=True)
        except Exception as exc:
            print(f"[auto-reassess] error: {exc}", flush=True)


threading.Thread(target=auto_reassess_loop, daemon=True).start()


# --- Routes ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200


@app.route('/api/push-evidence', methods=['POST'])
def push_evidence():
    body, status = run_assessment(trigger="manual")
    return jsonify(body), status


@app.route('/api/events', methods=['GET'])
def view_events():
    conn = db()
    try:
        rows = conn.execute('''
            SELECT timestamp, attack_type, ml_verdict, dl_verdict, final_verdict, severity
            FROM threat_events ORDER BY timestamp DESC LIMIT 50
        ''').fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    keys = ["timestamp", "attack_type", "ml_verdict", "dl_verdict", "final_verdict", "severity"]
    events = [dict(zip(keys, r)) for r in rows]
    return jsonify({"total": len(events), "events": events}), 200


@app.route('/api/ce-scores', methods=['GET'])
def view_ce_scores():
    conn = db()
    row = conn.execute(
        'SELECT timestamp, scores FROM ce_scores ORDER BY timestamp DESC LIMIT 1'
    ).fetchone()
    conn.close()
    if not row:
        return jsonify({"status": "no scores yet"}), 200
    return jsonify({"timestamp": row[0], "scores": json.loads(row[1])}), 200


@app.route('/api/feedback', methods=['GET'])
def view_feedback():
    conn = db()
    rows = conn.execute('''
        SELECT timestamp, trigger_source, sqli_risk, xss_risk, sqli_min_score, xss_min_score, events_assessed
        FROM risk_log ORDER BY timestamp DESC LIMIT 20
    ''').fetchall()
    conn.close()
    keys = ["timestamp", "trigger", "sqli_risk", "xss_risk",
            "sqli_min_ce_score", "xss_min_ce_score", "events_assessed"]
    return jsonify({"risk_log": [dict(zip(keys, r)) for r in rows]}), 200


@app.route('/api/watchlist', methods=['GET'])
def view_watchlist():
    conn = db()
    try:
        rows = conn.execute('''
            SELECT timestamp, attack_type, ml_score, dl_score, payload, risk_level
            FROM watchlist ORDER BY timestamp DESC LIMIT 50
        ''').fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    keys = ["timestamp", "attack_type", "ml_score", "dl_score", "payload", "risk_level"]
    items = [dict(zip(keys, r)) for r in rows]
    return jsonify({"total": len(items), "watchlist": items}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)