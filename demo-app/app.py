from flask import Flask, request, jsonify

app = Flask(__name__)

PRODUCTS = {
    "1": {"name": "Laptop", "price": 899},
    "2": {"name": "Headphones", "price": 129},
    "3": {"name": "Keyboard", "price": 59},
}


@app.route("/")
def home():
    return jsonify({"app": "CyberSense demo shop", "status": "ok"})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/search")
def search():
    q = request.args.get("q", "")
    results = [p for p in PRODUCTS.values() if q.lower() in p["name"].lower()]
    return jsonify({"query": q, "results": results})


@app.route("/product")
def product():
    pid = request.args.get("id", "")
    item = PRODUCTS.get(pid)
    if not item:
        return jsonify({"error": "not found"}), 404
    return jsonify(item)


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or request.form
    user = data.get("username", "")
    ok = user == "demo" and data.get("password", "") == "demo123"
    return jsonify({"user": user, "authenticated": ok}), (200 if ok else 401)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)