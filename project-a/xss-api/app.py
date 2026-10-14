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
    conn.commit()
    conn.close()

init_db()

# --- Detection threshold config ---
detection_config = {"threshold": 0.5, "sensitivity_boost": 1.0}

@app.route('/config/threshold', methods=['POST'])
def update_threshold():
    data = request.get_json()
    detection_config["threshold"]         = data.get("threshold", 0.5)
    detection_config["sensitivity_boost"] = data.get("sensitivity_boost", 1.0)
    return jsonify({"status": "updated", "config": detection_config}), 200

@app.route('/config', methods=['GET'])
def get_config():
    return jsonify(detection_config), 200

def log_event(attack_type, ml_conf, dl_conf, payload):
    severity = "high"   if ml_conf >= 1 and dl_conf > 0.8 else \
               "medium" if ml_conf >= 1 or  dl_conf > 0.5  else \
               "low"

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'INSERT INTO threat_events VALUES (?,?,?,?,?,?,?,?,?,?)',
        (
            str(uuid.uuid4()),
            datetime.utcnow().isoformat() + "Z",
            attack_type,
            "threat" if ml_conf >= 1  else "benign",
            "threat" if dl_conf > 0.5 else "benign",
            "threat",
            float(ml_conf),
            float(dl_conf),
            payload[:300],
            severity
        )
    )
    conn.commit()
    conn.close()

# --- Health check ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

# --- Model paths ---
models_dir = '/models'

ml_model_path         = os.path.join(models_dir, 'random_forest.pkl')
tfidf_vectorizer_path = os.path.join(models_dir, 'xssvectorizer.pkl')
dnn_preprocessor_path = os.path.join(models_dir, 'xss_char_preprocessor.pkl')
attention_model_path  = os.path.join(models_dir, 'xss_char_model.keras')

ml_model         = joblib.load(ml_model_path)
tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)

with open(dnn_preprocessor_path, "rb") as f:
    dnn_preprocessor = pickle.load(f)

dnn_model = load_model(attention_model_path,
                       custom_objects={'AttentionLayer': AttentionLayer})

# --- Predict endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    data    = request.json
    payload = data.get('payload', '')

    tfidf_features = tfidf_vectorizer.transform([payload]).toarray()
    ml_prediction  = ml_model.predict(tfidf_features)[0]

    tokenizer = dnn_preprocessor['tokenizer']
    max_len   = dnn_preprocessor['max_len']
    sequence  = tokenizer.texts_to_sequences([payload])
    padded_sequence = pad_sequences(
        sequence,
        maxlen=max_len,
        padding=dnn_preprocessor.get('padding', 'post'),
        truncating=dnn_preprocessor.get('truncating', 'pre'),
    )
    dl_prediction   = dnn_model.predict(padded_sequence)[0][0]

    threshold = detection_config["threshold"] / detection_config["sensitivity_boost"]
    is_threat = ml_prediction == 1 or dl_prediction > threshold

    if is_threat:
        log_event('xss', ml_prediction, dl_prediction, payload)

    return jsonify({
        "is_threat":     bool(is_threat),
        "ml_confidence": int(ml_prediction),
        "dl_confidence": float(dl_prediction)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)