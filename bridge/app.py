from flask import Flask, request, jsonify
import sqlite3
import requests
import os
import json
from datetime import datetime, timedelta

app = Flask(__name__)

DB_PATH  = '/data/events.db'
CE_API_URL   = os.environ.get('CE_API_URL',  'http://compliance-engine:5002/api/assess-evidence')
SQLI_API_URL = os.environ.get('SQLI_API_URL', 'http://sql-api:5000')
XSS_API_URL  = os.environ.get('XSS_API_URL',  'http://xss-api:5000')

# --- Read recent threat events ---
def get_recent_threats(hours=24):
    conn = sqlite3.connect(DB_PATH)
    since = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"
    rows = conn.execute('''
        SELECT attack_type, ml_verdict, dl_verdict, final_verdict,
               ml_confidence, dl_confidence, payload, severity, timestamp
        FROM threat_events
        WHERE timestamp > ?
        ORDER BY timestamp DESC
    ''', (since,)).fetchall()
    conn.close()

    events = []
    for r in rows:
        events.append({
            "attack_type":   r[0],
            "ml_verdict":    r[1],
            "dl_verdict":    r[2],
            "final_verdict": r[3],
            "ml_confidence": r[4],
            "dl_confidence": r[5],
            "payload":       r[6],
            "severity":      r[7],
            "timestamp":     r[8]
        })
    return events

# --- Build evidence document ---
def build_evidence_document(events):
    sqli = [e for e in events if e['attack_type'] == 'sqli']
    xss  = [e for e in events if e['attack_type'] == 'xss']
    high = [e for e in events if e['severity'] == 'high']

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
        both_confirmed = sum(
            1 for e in sqli
            if e['ml_verdict'] == 'threat' and e['dl_verdict'] == 'threat'
        )
        max_ml = max(e['ml_confidence'] for e in sqli)
        max_dl = max(e['dl_confidence'] for e in sqli)

        doc += f"""
SQL INJECTION ATTACKS ({len(sqli)} detected)
Both ML and DL models confirmed: {both_confirmed}
Highest ML confidence: {max_ml:.2%}
Highest DL confidence: {max_dl:.2%}

Cyber Essentials implications:
- CE Control 1 (Boundary Firewalls): SQL injection reaching the API
  layer indicates insufficient input filtering at the boundary. FAIL.
- CE Control 3 (User Access Control): SQLi attempts suggest database
  accounts may have excessive privileges. FAIL.
- CE Control 2 (Secure Configuration): Raw SQL queries accepted suggests
  application configuration gaps. WARN.
"""

    if xss:
        both_confirmed = sum(
            1 for e in xss
            if e['ml_verdict'] == 'threat' and e['dl_verdict'] == 'threat'
        )
        max_ml = max(e['ml_confidence'] for e in xss)
        max_dl = max(e['dl_confidence'] for e in xss)

        doc += f"""
CROSS-SITE SCRIPTING ATTACKS ({len(xss)} detected)
Both ML and DL models confirmed: {both_confirmed}
Highest ML confidence: {max_ml:.2%}
Highest DL confidence: {max_dl:.2%}

Cyber Essentials implications:
- CE Control 1 (Boundary Firewalls): XSS payloads reaching the application
  suggest WAF or input sanitisation gaps. FAIL.
- CE Control 2 (Secure Configuration): XSS success indicates output
  encoding is not applied in application config. FAIL.
- CE Control 4 (Malware Protection): Script injection can be used to
  load malicious scripts in user browsers. WARN.
"""

    return doc

# --- Store CE scores ---
def store_ce_scores(scores):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS ce_scores (
            timestamp TEXT,
            scores    TEXT
        )
    ''')
    conn.execute(
        'INSERT INTO ce_scores VALUES (?, ?)',
        (datetime.utcnow().isoformat() + "Z", json.dumps(scores))
    )
    conn.commit()
    conn.close()

# --- FEEDBACK LOOP ---
def apply_feedback_loop(ce_scores):
    """
    CE scores are 0.0 to 1.0.
    Lower score = more failures = higher risk.
    Boost detection sensitivity where CE controls are failing.
    """
    ce1 = ce_scores.get('CE1_firewalls',          1.0)
    ce2 = ce_scores.get('CE2_secure_config',       1.0)
    ce3 = ce_scores.get('CE3_user_access',         1.0)
    ce4 = ce_scores.get('CE4_malware_protection',  1.0)

    # SQLi is implicated by CE1 and CE3
    sqli_risk  = 1 - min(ce1, ce3)
    sqli_boost = round(1.0 + (sqli_risk * 0.5), 3)  # max 1.5x sensitivity

    # XSS is implicated by CE1, CE2, and CE4
    xss_risk  = 1 - min(ce1, ce2, ce4)
    xss_boost = round(1.0 + (xss_risk * 0.5), 3)    # max 1.5x sensitivity

    results = {"sqli_boost": sqli_boost, "xss_boost": xss_boost}

    # Push threshold updates to both APIs
    try:
        requests.post(
            f"{SQLI_API_URL}/config/threshold",
            json={"threshold": 0.5, "sensitivity_boost": sqli_boost},
            timeout=5
        )
        results["sqli_api"] = "updated"
    except Exception:
        results["sqli_api"] = "unreachable"

    try:
        requests.post(
            f"{XSS_API_URL}/config/threshold",
            json={"threshold": 0.5, "sensitivity_boost": xss_boost},
            timeout=5
        )
        results["xss_api"] = "updated"
    except Exception:
        results["xss_api"] = "unreachable"

    # Store feedback values for dashboard and audit
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS feedback_log (
            timestamp   TEXT,
            sqli_boost  REAL,
            xss_boost   REAL,
            ce1_score   REAL,
            ce2_score   REAL,
            ce3_score   REAL
        )
    ''')
    conn.execute(
        'INSERT INTO feedback_log VALUES (?,?,?,?,?,?)',
        (
            datetime.utcnow().isoformat() + "Z",
            sqli_boost,
            xss_boost,
            ce1, ce2, ce3
        )
    )
    conn.commit()
    conn.close()

    return results

# --- Health check ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

# --- Push evidence and trigger feedback loop ---
@app.route('/api/push-evidence', methods=['POST'])
def push_evidence():
    events = get_recent_threats(hours=24)

    if not events:
        return jsonify({"status": "no_threats", "events": 0}), 200

    doc = build_evidence_document(events)

    # Call CE engine or fall back to mock
    try:
        response = requests.post(
            CE_API_URL,
            json={"evidence_text": doc},
            timeout=10
        )
        ce_result = response.json()
    except Exception:
        ce_result = {
            "ce_scores": {
                "CE1_firewalls":          0.2,
                "CE2_secure_config":      0.4,
                "CE3_user_access":        0.3,
                "CE4_malware_protection": 0.7,
                "CE5_patch_management":   0.6
            },
            "gaps": ["CE1", "CE2", "CE3"],
            "remediation": {
                "CE1": "Review firewall rules to block SQLi and XSS payloads.",
                "CE2": "Apply output encoding and input validation in application config.",
                "CE3": "Implement least-privilege access on all database accounts."
            },
            "note": "mock response - compliance engine not yet connected"
        }

    ce_scores = ce_result.get('ce_scores', {})
    store_ce_scores(ce_scores)

    # --- Run feedback loop ---
    feedback = apply_feedback_loop(ce_scores)

    return jsonify({
        "status":           "pushed",
        "events_processed": len(events),
        "evidence_preview": doc[:300],
        "ce_result":        ce_result,
        "feedback_applied": feedback
    }), 200

# --- View events ---
@app.route('/api/events', methods=['GET'])
def view_events():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute('''
        SELECT timestamp, attack_type, ml_verdict, dl_verdict,
               final_verdict, severity
        FROM threat_events
        ORDER BY timestamp DESC
        LIMIT 50
    ''').fetchall()
    conn.close()

    events = [
        {
            "timestamp":     r[0],
            "attack_type":   r[1],
            "ml_verdict":    r[2],
            "dl_verdict":    r[3],
            "final_verdict": r[4],
            "severity":      r[5]
        }
        for r in rows
    ]
    return jsonify({"total": len(events), "events": events}), 200

# --- View latest CE scores ---
@app.route('/api/ce-scores', methods=['GET'])
def view_ce_scores():
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            'SELECT timestamp, scores FROM ce_scores ORDER BY timestamp DESC LIMIT 1'
        ).fetchone()
    except Exception:
        row = None
    conn.close()

    if not row:
        return jsonify({"status": "no scores yet"}), 200

    return jsonify({
        "timestamp": row[0],
        "scores":    json.loads(row[1])
    }), 200

# --- View feedback log ---
@app.route('/api/feedback', methods=['GET'])
def view_feedback():
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute('''
            SELECT timestamp, sqli_boost, xss_boost, ce1_score, ce2_score, ce3_score
            FROM feedback_log
            ORDER BY timestamp DESC
            LIMIT 20
        ''').fetchall()
    except Exception:
        rows = []
    conn.close()

    log = [
        {
            "timestamp":  r[0],
            "sqli_boost": r[1],
            "xss_boost":  r[2],
            "ce1_score":  r[3],
            "ce2_score":  r[4],
            "ce3_score":  r[5]
        }
        for r in rows
    ]
    return jsonify({"feedback_log": log}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)