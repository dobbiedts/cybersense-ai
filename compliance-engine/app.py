import re
from flask import Flask, request, jsonify

app = Flask(__name__)

CONTROLS = [
    "CE1_firewalls",
    "CE2_secure_config",
    "CE3_user_access",
    "CE4_malware_protection",
    "CE5_patch_management",
]

# How strongly each attack type implicates each CE control (0 to 1)
IMPACT = {
    "sqli": {
        "CE1_firewalls": 0.35,
        "CE2_secure_config": 0.15,
        "CE3_user_access": 0.35,
        "CE4_malware_protection": 0.00,
        "CE5_patch_management": 0.10,
    },
    "xss": {
        "CE1_firewalls": 0.30,
        "CE2_secure_config": 0.35,
        "CE3_user_access": 0.05,
        "CE4_malware_protection": 0.25,
        "CE5_patch_management": 0.10,
    },
}

REMEDIATION = {
    "CE1": "Filter malicious input at the boundary: deploy a WAF in front of web apps and block SQLi/XSS payload patterns before they reach the application.",
    "CE2": "Harden application configuration: enforce input validation, output encoding, parameterised queries and a Content Security Policy.",
    "CE3": "Apply least privilege to database accounts: the web app account should have no admin rights and access only the tables it needs.",
    "CE4": "Strengthen malware protection: restrict script execution in browsers via CSP and keep endpoint protection active on servers and user devices.",
    "CE5": "Patch web frameworks, libraries and database software promptly; high-risk vulnerabilities must be fixed within 14 days under Cyber Essentials.",
}

PASS_THRESHOLD = 0.7
FAIL_THRESHOLD = 0.5


def extract_count(pattern, text):
    match = re.search(pattern, text, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def volume_factor(count):
    """More attacks = stronger evidence of a control gap, capped at 1.0."""
    if count == 0:
        return 0.0
    return min(1.0, 0.5 + 0.1 * count)


def score_evidence(text):
    sqli  = extract_count(r"SQL Injection attempts:\s*(\d+)", text)
    xss   = extract_count(r"XSS attempts:\s*(\d+)", text)
    total = extract_count(r"Total threats detected:\s*(\d+)", text)
    high  = extract_count(r"High severity events:\s*(\d+)", text)

    # High-severity share increases impact by up to 20%
    severity_factor = 1.0 + (0.2 * high / total) if total else 1.0

    scores = {c: 1.0 for c in CONTROLS}
    for attack, count in (("sqli", sqli), ("xss", xss)):
        factor = volume_factor(count) * severity_factor
        for control, weight in IMPACT[attack].items():
            scores[control] -= weight * factor

    scores = {c: round(max(0.0, min(1.0, s)), 3) for c, s in scores.items()}
    signals = {"sqli": sqli, "xss": xss, "total": total, "high_severity": high}
    return scores, signals


def status_of(score):
    if score >= PASS_THRESHOLD:
        return "pass"
    if score >= FAIL_THRESHOLD:
        return "warn"
    return "fail"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/api/assess-evidence", methods=["POST"])
def assess_evidence():
    data = request.get_json(silent=True) or {}
    text = data.get("evidence_text", "")

    if not text.strip():
        return jsonify({"error": "evidence_text is required"}), 400

    scores, signals = score_evidence(text)
    status = {c.split("_")[0]: status_of(s) for c, s in scores.items()}
    gaps = [c for c, st in status.items() if st != "pass"]

    return jsonify({
        "engine": "rules-v1",
        "ce_scores": scores,
        "status": status,
        "gaps": gaps,
        "remediation": {g: REMEDIATION[g] for g in gaps},
        "signals": signals,
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)