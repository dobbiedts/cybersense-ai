from flask import Flask, request, jsonify
import joblib
import pickle
import numpy as np
import os
import sqlite3
import uuid
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# --- Custom Attention Layer ---
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer="random_normal")
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[-1],),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        score             = tf.tanh(tf.matmul(inputs, self.W) + self.b)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector    = attention_weights * inputs
        context_vector    = tf.reduce_sum(context_vector, axis=1)
        return context_vector

app = Flask(__name__)

# --- Event DB setup ---
DB_PATH = '/data/events.db'

def init_db():
    os.makedirs('/data', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
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
        CREATE TABLE IF NOT EXISTS watchlist (
            event_id    TEXT PRIMARY KEY,
            timestamp   TEXT NOT NULL,
            attack_type TEXT NOT NULL,
            ml_score    REAL,
            dl_score    REAL,
            payload     TEXT,
            risk_level  TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- Detection and risk-adaptive response ---
# Detection threshold is fixed: evaluation showed 0.5 is near optimal and
# lowering it adds false alarms without catching meaningfully more attacks.
DETECTION_THRESHOLD = 0.5
SEVERITY_LEVELS = ["low", "medium", "high", "critical"]

# Set by the bridge from Cyber Essentials gap scores
risk_config = {"risk_level": "normal", "severity_escalation": 0, "watch_threshold": None}

@app.route('/config/risk', methods=['POST'])
def update_risk():
    data = request.get_json(silent=True) or {}
    level = data.get("risk_level", "normal")
    if level not in ("normal", "elevated", "critical"):
        return jsonify({"error": "risk_level must be normal, elevated or critical"}), 400
    watch = data.get("watch_threshold")
    risk_config["risk_level"] = level
    risk_config["severity_escalation"] = max(0, int(data.get("severity_escalation", 0)))
    risk_config["watch_threshold"] = float(watch) if watch is not None else None
    return jsonify({"status": "updated", "config": risk_config}), 200

@app.route('/config', methods=['GET'])
def get_config():
    return jsonify({"detection_threshold": DETECTION_THRESHOLD, **risk_config}), 200

def base_severity(ml_score, dl_score):
    ml_hit = ml_score > DETECTION_THRESHOLD
    dl_hit = dl_score > DETECTION_THRESHOLD
    if ml_hit and dl_hit and min(ml_score, dl_score) >= 0.8:
        return "high"
    if ml_hit and dl_hit:
        return "medium"
    return "low"

def escalate(severity):
    i = SEVERITY_LEVELS.index(severity) + risk_config["severity_escalation"]
    return SEVERITY_LEVELS[min(i, len(SEVERITY_LEVELS) - 1)]

def now_utc():
    return datetime.utcnow().isoformat() + "Z"

def log_event(attack_type, ml_score, dl_score, payload, severity):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'INSERT INTO threat_events VALUES (?,?,?,?,?,?,?,?,?,?)',
        (
            str(uuid.uuid4()),
            now_utc(),
            attack_type,
            "threat" if ml_score > DETECTION_THRESHOLD else "benign",
            "threat" if dl_score > DETECTION_THRESHOLD else "benign",
            "threat",
            float(ml_score),
            float(dl_score),
            payload[:300],
            severity
        )
    )
    conn.commit()
    conn.close()

def log_watchlist(attack_type, ml_score, dl_score, payload):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'INSERT INTO watchlist VALUES (?,?,?,?,?,?,?)',
        (
            str(uuid.uuid4()),
            now_utc(),
            attack_type,
            float(ml_score),
            float(dl_score),
            payload[:300],
            risk_config["risk_level"]
        )
    )
    conn.commit()
    conn.close()

def respond(attack_type, payload, ml_score, dl_score):
    """Fixed-threshold detection plus risk-adaptive response."""
    ml_score, dl_score = float(ml_score), float(dl_score)
    is_threat = ml_score > DETECTION_THRESHOLD or dl_score > DETECTION_THRESHOLD

    result = {
        "is_threat":     bool(is_threat),
        "ml_confidence": round(ml_score, 4),
        "dl_confidence": dl_score,
        "risk_level":    risk_config["risk_level"],
    }

    if is_threat:
        severity = escalate(base_severity(ml_score, dl_score))
        log_event(attack_type, ml_score, dl_score, payload, severity)
        result["severity"] = severity
    else:
        watch = risk_config["watch_threshold"]
        watchlisted = watch is not None and max(ml_score, dl_score) > watch
        if watchlisted:
            log_watchlist(attack_type, ml_score, dl_score, payload)
        result["watchlisted"] = bool(watchlisted)

    return jsonify(result)

# --- Health check ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

# --- Model paths ---
models_dir = '/models'

ml_model_path         = os.path.join(models_dir, 'random_forest.pkl')
tfidf_vectorizer_path = os.path.join(models_dir, 'xssvectorizer.pkl')
dnn_preprocessor_path = os.path.join(models_dir, 'xss_char_preprocessor_v2.pkl')
attention_model_path  = os.path.join(models_dir, 'xss_char_model_v2.keras')

ml_model         = joblib.load(ml_model_path)
tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
ml_attack_col    = list(ml_model.classes_).index(1)

with open(dnn_preprocessor_path, "rb") as f:
    dnn_preprocessor = pickle.load(f)

dnn_model = load_model(attention_model_path,
                       custom_objects={'AttentionLayer': AttentionLayer})

# --- Predict endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    data    = request.get_json(silent=True) or {}
    payload = data.get('payload', '')

    # Random Forest: probability of attack (share of trees voting attack)
    tfidf_features = tfidf_vectorizer.transform([payload]).toarray()
    ml_score       = ml_model.predict_proba(tfidf_features)[0][ml_attack_col]

    # Character-level attention model
    tokenizer = dnn_preprocessor['tokenizer']
    max_len   = dnn_preprocessor['max_len']
    sequence  = tokenizer.texts_to_sequences([payload])
    padded_sequence = pad_sequences(
        sequence,
        maxlen=max_len,
        padding=dnn_preprocessor.get('padding', 'post'),
        truncating=dnn_preprocessor.get('truncating', 'pre'),
    )
    dl_score = dnn_model.predict(padded_sequence, verbose=0)[0][0]

    return respond('xss', payload, ml_score, dl_score)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)