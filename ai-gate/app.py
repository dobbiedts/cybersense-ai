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

The detection APIs log every threat themselves, so AI gate blocks feed the
Cyber Essentials feedback loop automatically.
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
# closed = block if the detection APIs can't be reached (safer), open = allow
FAIL_MODE = os.environ.get("GATE_FAIL_MODE", "closed").lower()
MAX_VALUE_LENGTH = int(os.environ.get("GATE_MAX_VALUE_LENGTH", "2000"))

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailers", "transfer-encoding", "upgrade", "content-length",
              "content-encoding", "host"}

session = requests.Session()
stats = {"checked": 0, "blocked": 0, "allowed": 0, "errors": 0}


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
            r = session.post(url, json={"payload": value}, timeout=5)
            r.raise_for_status()
            verdicts[attack] = r.json()
        except Exception as exc:
            return verdicts, f"{attack} API error: {exc}"
    return verdicts, None


def inspect():
    findings = []
    for name, value in collect_inputs():
        seen = set()
        for form, variant in variants_of(value):
            verdicts, error = check_value(variant)
            if error:
                stats["errors"] += 1
                if FAIL_MODE == "closed":
                    return {"blocked": True, "reason": "detection unavailable", "detail": error}
                continue
            for attack, v in verdicts.items():
                if v.get("is_threat") and attack not in seen:
                    seen.add(attack)
                    findings.append({
                        "input": name,
                        "form": form,
                        "attack_type": attack,
                        "severity": v.get("severity"),
                        "ml_confidence": v.get("ml_confidence"),
                        "dl_confidence": round(float(v.get("dl_confidence", 0)), 4),
                    })
    return {"blocked": bool(findings), "reason": "ai detection" if findings else None,
            "findings": findings}


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
    decision = inspect()
    elapsed_ms = round(1000 * (time.time() - started), 1)

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