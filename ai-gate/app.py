"""
CyberSense AI - AI gate.

Sits between ModSecurity and the protected app:
    Client -> ModSecurity (signature rules) -> AI gate (ML + DL models) -> demo app

Every user-supplied value (query parameters, form fields, JSON values) is sent
to the SQLi and XSS detection APIs. Each API runs a Random Forest and a deep
learning model. If either API flags a value, the request is blocked with 403.
Otherwise it is forwarded unchanged to the app.

Before checking, each value is normalised (canonicalised) the way a WAF does:
repeated URL decoding, HTML entity decoding and removal of SQL inline comments,
so encoded or comment-split payloads can't slip past the models.

Attack-type attribution: both specialist APIs check every value, and each can
occasionally fire on the other's attack type. The gate asks with log=false,
decides the most likely attack type(s), and records only those via /record,
so the Cyber Essentials scores aren't skewed by mislabelled events.

Purely numeric values (e.g. 10, -3, 2.5) are allowed without asking the models:
they can't carry SQL injection or XSS, and the models are unreliable on very
short inputs (evaluation showed 10 of 301 numbers wrongly flagged).
"""
import html
import json
import os
import re
import time
import urllib.parse

import requests
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

APP_URL  = os.environ.get("APP_URL",  "http://demo-app:8000")
SQLI_URL = os.environ.get("SQLI_URL", "http://sql-api:5000/predict_sql")
XSS_URL  = os.environ.get("XSS_URL",  "http://xss-api:5000/predict")
SQLI_RECORD_URL = os.environ.get("SQLI_RECORD_URL", "http://sql-api:5000/record")
XSS_RECORD_URL  = os.environ.get("XSS_RECORD_URL",  "http://xss-api:5000/record")
RECORD_URLS = {"sqli": SQLI_RECORD_URL, "xss": XSS_RECORD_URL}
RECORD_TOKEN = os.environ.get("RECORD_TOKEN", "")
DETECTION_THRESHOLD = 0.5
# closed = block if the detection APIs can't be reached (safer), open = allow
FAIL_MODE = os.environ.get("GATE_FAIL_MODE", "closed").lower()
MAX_VALUE_LENGTH = int(os.environ.get("GATE_MAX_VALUE_LENGTH", "2000"))

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailers", "transfer-encoding", "upgrade", "content-length",
              "content-encoding", "host"}

session = requests.Session()
stats = {"checked": 0, "blocked": 0, "allowed": 0, "errors": 0,
         "recorded": 0, "suppressed_labels": 0, "record_errors": 0,
         "numeric_skipped": 0}

# A plain number: optional sign, digits, optional decimal part
NUMERIC = re.compile(r"[+-]?\d+(?:\.\d+)?")


# ---------------------------------------------------------------- inputs
def flatten(value, prefix=""):
    """Yield (name, string) pairs from nested JSON."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from flatten(v, f"{prefix}[{i}]")
    elif isinstance(value, str):
        yield prefix, value


def collect_inputs():
    inputs = []
    for name, values in request.args.lists():
        inputs += [(f"query:{name}", v) for v in values]
    for name, values in request.form.lists():
        inputs += [(f"form:{name}", v) for v in values]
    if request.is_json:
        body = request.get_json(silent=True)
        if body is not None:
            inputs += [(f"json:{n}", v) for n, v in flatten(body)]
    return [(n, v[:MAX_VALUE_LENGTH]) for n, v in inputs if v and v.strip()]


# ---------------------------------------------------------------- normalisation
SQL_INLINE_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
MAX_DECODE_ROUNDS = 3


def variants_of(value):
    """
    Return the raw value plus normalised forms that differ from it:
      decoded : URL-decoded repeatedly (catches double/triple encoding) and
                HTML-entity decoded
      no-comments : decoded form with SQL /* */ comments removed (catches
                UNI/**/ON SEL/**/ECT style keyword splitting)
    """
    decoded = value
    for _ in range(MAX_DECODE_ROUNDS):
        step = urllib.parse.unquote(decoded)
        if step == decoded:
            break
        decoded = step
    decoded = html.unescape(decoded)
    no_comments = SQL_INLINE_COMMENT.sub("", decoded)

    variants = [("raw", value)]
    for label, v in (("decoded", decoded), ("no-comments", no_comments)):
        if v.strip() and all(v != existing for _, existing in variants):
            variants.append((label, v))
    return variants


# ---------------------------------------------------------------- detection
def check_value(value):
    """Ask both detection APIs. Returns (verdicts, error)."""
    verdicts = {}
    for attack, url in (("sqli", SQLI_URL), ("xss", XSS_URL)):
        try:
            r = session.post(url, json={"payload": value, "log": False}, timeout=5)
            r.raise_for_status()
            verdicts[attack] = r.json()
        except Exception as exc:
            return verdicts, f"{attack} API error: {exc}"
    return verdicts, None


def strength(v):
    return float(v.get("ml_confidence", 0)) + float(v.get("dl_confidence", 0))


def attribute(hits):
    """
    hits: {attack: verdict} for attack types that flagged this value.
    Returns the set of attack types to record.
      - one type flagged: that type
      - both flagged: prefer the type(s) whose Random Forest also agrees
        (two models agreeing is stronger evidence than one); if neither or
        both RFs agree, the type with the stronger combined score, or both on a tie
    """
    if len(hits) <= 1:
        return set(hits)
    rf_agrees = {a for a, v in hits.items() if float(v.get("ml_confidence", 0)) > DETECTION_THRESHOLD}
    if len(rf_agrees) == 1:
        return rf_agrees
    scores = {a: strength(v) for a, v in hits.items()}
    best = max(scores.values())
    return {a for a, sc in scores.items() if abs(sc - best) < 1e-6}


def record(attack, payload, verdict, kind="threat"):
    headers = {"X-Record-Token": RECORD_TOKEN} if RECORD_TOKEN else {}
    body = {"payload": payload, "kind": kind,
            "ml_score": verdict.get("ml_confidence", 0),
            "dl_score": verdict.get("dl_confidence", 0)}
    try:
        r = session.post(RECORD_URLS[attack], json=body, headers=headers, timeout=5)
        r.raise_for_status()
        stats["recorded"] += 1
    except Exception as exc:
        stats["record_errors"] += 1
        print(json.dumps({"event": "record_error", "attack": attack, "error": str(exc)}), flush=True)


def inspect():
    findings = []
    to_record = []          # (attack, payload, verdict, kind)
    for name, value in collect_inputs():
        hits = {}           # attack -> (form, variant, verdict): strongest flagged variant
        watch = {}          # attack -> verdict, for values that pass but sit in the grey zone
        for form, variant in variants_of(value):
            if NUMERIC.fullmatch(variant.strip()):
                stats["numeric_skipped"] += 1
                continue
            verdicts, error = check_value(variant)
            if error:
                stats["errors"] += 1
                if FAIL_MODE == "closed":
                    return {"blocked": True, "reason": "detection unavailable", "detail": error}, []
                continue
            for attack, v in verdicts.items():
                if v.get("is_threat"):
                    if attack not in hits or strength(v) > strength(hits[attack][2]):
                        hits[attack] = (form, variant, v)
                elif v.get("watchlisted") and form == "raw":
                    watch[attack] = v

        primary = attribute({a: h[2] for a, h in hits.items()})
        stats["suppressed_labels"] += len(hits) - len(primary)
        for attack, (form, variant, v) in hits.items():
            findings.append({
                "input": name,
                "form": form,
                "attack_type": attack,
                "primary": attack in primary,
                "severity": v.get("severity"),
                "ml_confidence": v.get("ml_confidence"),
                "dl_confidence": round(float(v.get("dl_confidence", 0)), 4),
            })
            if attack in primary:
                to_record.append((attack, variant, v, "threat"))
        if not hits:
            to_record += [(a, value, v, "watch") for a, v in watch.items()]

    decision = {"blocked": bool(findings), "reason": "ai detection" if findings else None,
                "findings": findings}
    return decision, to_record


# ---------------------------------------------------------------- routes
@app.route("/gate/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/gate/stats")
def gate_stats():
    return jsonify({**stats, "fail_mode": FAIL_MODE}), 200


@app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
def gate(path):
    started = time.time()
    stats["checked"] += 1
    decision, to_record = inspect()
    elapsed_ms = round(1000 * (time.time() - started), 1)
    for attack, payload, verdict, kind in to_record:
        record(attack, payload, verdict, kind)

    if decision["blocked"]:
        stats["blocked"] += 1
        print(json.dumps({"event": "blocked", "path": "/" + path, "ms": elapsed_ms, **decision}),
              flush=True)
        status = 503 if decision["reason"] == "detection unavailable" else 403
        body = {"blocked_by": "CyberSense AI gate", **decision}
        resp = jsonify(body)
        resp.status_code = status
        resp.headers["X-CyberSense-Gate"] = "blocked"
        return resp

    stats["allowed"] += 1
    upstream = session.request(
        method=request.method,
        url=f"{APP_URL}/{path}",
        params=list(request.args.items(multi=True)),
        data=request.get_data(),
        headers={k: v for k, v in request.headers if k.lower() not in HOP_BY_HOP},
        allow_redirects=False,
        timeout=15,
    )
    headers = [(k, v) for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP]
    headers.append(("X-CyberSense-Gate", f"allowed; {elapsed_ms}ms"))
    return Response(upstream.content, status=upstream.status_code, headers=headers)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)