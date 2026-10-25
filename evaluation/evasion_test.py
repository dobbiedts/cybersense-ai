"""
CyberSense AI - evasion test harness (defensive evaluation).

Sends plain and obfuscated SQLi / XSS payloads, plus benign inputs, through
two paths and records which layer caught each one:

  Full stack : ModSecurity, then AI gate, then app  (default http://localhost:8080)
  AI only    : AI gate, then app, skips ModSecurity  (default http://localhost:8081)

On the full stack, a 403 can come from either layer. ModSecurity returns an
HTML error page, while the AI gate returns JSON containing "CyberSense AI gate",
so the response body tells us which layer blocked it.

Usage (on your machine, with the stack running):
    python evaluation/evasion_test.py --out evaluation/results/evasion_results.json
"""
import argparse
import json
import urllib.error
import urllib.parse
import urllib.request

GATE_MARKER = "CyberSense AI gate"


def send(base, path):
    """GET base+path. Returns (status, body_text)."""
    try:
        with urllib.request.urlopen(base + path, timeout=20) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return None, str(e)


def gate_detail(body):
    """Pull the model findings out of the AI gate's JSON block response."""
    try:
        data = json.loads(body)
    except Exception:
        return ""
    parts = []
    for f in data.get("findings", []):
        parts.append(f"{f.get('attack_type')}[{f.get('form')}](ml={f.get('ml_confidence')},"
                     f"dl={f.get('dl_confidence')})")
    return "; ".join(parts)


def classify(waf_blocked, ai_blocked):
    if waf_blocked and ai_blocked:
        return "both layers"
    if ai_blocked:
        return "AI models only"
    if waf_blocked:
        return "signatures only"
    return "not blocked"


def run(cases, label, stack_base, gate_base):
    print(f"\n{'=' * 104}\n{label}\n{'=' * 104}")
    print(f"{'#':>3}  {'name':<28}{'ModSec':>9}{'AI gate':>9}  {'caught by':<17}models")
    print("-" * 104)

    rows = []
    for i, (name, path) in enumerate(cases, 1):
        stack_status, stack_body = send(stack_base, path)
        gate_status, gate_body = send(gate_base, path)

        # ModSecurity blocked if the full stack returned 403 WITHOUT the gate's JSON
        waf_blocked = stack_status == 403 and GATE_MARKER not in stack_body
        ai_blocked = gate_status == 403
        caught = classify(waf_blocked, ai_blocked)
        models = gate_detail(gate_body)

        rows.append({"name": name, "path": path,
                     "stack_status": stack_status, "gate_status": gate_status,
                     "modsec_blocked": waf_blocked, "ai_blocked": ai_blocked,
                     "caught": caught, "models": models})
        print(f"{i:>3}  {name:<28}{('BLOCK' if waf_blocked else 'pass'):>9}"
              f"{('BLOCK' if ai_blocked else 'pass'):>9}  {caught:<17}{models}")
    return rows


def summarise(rows, expect_block):
    total = len(rows)
    waf = sum(r["modsec_blocked"] for r in rows)
    ai = sum(r["ai_blocked"] for r in rows)
    either = sum(r["modsec_blocked"] or r["ai_blocked"] for r in rows)

    print(f"\n{'-' * 60}")
    if expect_block:
        print(f"SUMMARY over {total} attacks")
        print(f"  Blocked by ModSecurity signatures : {waf}/{total} ({waf / total:.0%})")
        print(f"  Blocked by AI models (gate)       : {ai}/{total} ({ai / total:.0%})")
        print(f"  Blocked by at least one layer     : {either}/{total} ({either / total:.0%})")
        print(f"  Caught by AI models only          : {sum(r['caught'] == 'AI models only' for r in rows)}")
        print(f"  Caught by signatures only         : {sum(r['caught'] == 'signatures only' for r in rows)}")
        print(f"  MISSED BY BOTH                    : {sum(r['caught'] == 'not blocked' for r in rows)}")
    else:
        print(f"SUMMARY over {total} benign inputs")
        print(f"  Wrongly blocked by signatures     : {waf}/{total} ({waf / total:.0%})")
        print(f"  Wrongly blocked by AI models      : {ai}/{total} ({ai / total:.0%})")
        print(f"  Wrongly blocked by either         : {either}/{total} ({either / total:.0%})")


def search(v):
    return "/search?q=" + urllib.parse.quote(v, safe="")


def product(v):
    return "/product?id=" + urllib.parse.quote(v, safe="")


SQLI = [
    ("plain OR tautology",         product("1' OR '1'='1")),
    ("plain UNION SELECT",         product("1' UNION SELECT username,password FROM users--")),
    ("inline comment break",       product("1'/**/OR/**/'1'='1")),
    ("mixed case keywords",        product("1' uNiOn SeLeCt 1,2,3--")),
    ("double URL-encoded",         product("1%27%20OR%201%3D1")),
    ("tab/newline whitespace",     product("1'\tOR\n'1'='1")),
    ("comment-split keywords",     product("1' UNI/**/ON SEL/**/ECT 1--")),
    ("hex-encoded comparison",     product("1 OR 0x31=0x31")),
    ("time-based blind",           product("1'; WAITFOR DELAY '0:0:5'--")),
    ("CHAR() encoding",            product("1' OR CHAR(49)=CHAR(49)--")),
]

XSS = [
    ("plain script tag",           search("<script>alert(1)</script>")),
    ("img onerror",                search("<img src=x onerror=alert(1)>")),
    ("svg onload",                 search("<svg/onload=alert(1)>")),
    ("attribute breakout",         search('"><script>alert(1)</script>')),
    ("mixed case tag",             search("<ScRiPt>alert(1)</ScRiPt>")),
    ("HTML-entity encoded",        search("&lt;script&gt;alert(1)&lt;/script&gt;")),
    ("body onload no space",       search("<body/onload=alert(1)>")),
    ("javascript: URI",            search("javascript:alert(document.cookie)")),
    ("nested broken tag",          search("<scr<script>ipt>alert(1)</script>")),
    ("onclick no quotes",          search("<a href=x onclick=alert(1)>click</a>")),
]

BENIGN = [
    ("surname apostrophe",         search("O'Brien")),
    ("contraction",                search("it's a great laptop")),
    ("possessive",                 search("men's shoes")),
    ("ampersand",                  search("Fish & Chips")),
    ("less-than price",            search("laptops < £500")),
    ("email address",              search("sales@example.com")),
    ("normal product id",          product("2")),
    ("quoted phrase",              search('"best laptop 2026"')),
    ("maths expression",           search("3 * (4 + 5) = 27")),
    ("URL in query",               search("https://example.com/item?id=5")),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", default="http://localhost:8080")
    parser.add_argument("--gate", default="http://localhost:8081")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results = {}
    results["sqli"] = run(SQLI, "SQL INJECTION (plain + obfuscated), expected: blocked", args.stack, args.gate)
    summarise(results["sqli"], expect_block=True)
    results["xss"] = run(XSS, "CROSS-SITE SCRIPTING (plain + obfuscated), expected: blocked", args.stack, args.gate)
    summarise(results["xss"], expect_block=True)
    results["benign"] = run(BENIGN, "BENIGN INPUTS, expected: allowed", args.stack, args.gate)
    summarise(results["benign"], expect_block=False)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()