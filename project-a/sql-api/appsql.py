from flask import Flask, request, jsonify
import joblib
import pickle
import numpy as np
import os
import sqlite3
import uuid
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

rf_model_path            = os.path.join(models_dir, 'RandomForest (1).pkl')
rf_vectorizer_path       = os.path.join(models_dir, 'sqlvectorizer (2).pkl')
bilstm_model_path        = os.path.join(models_dir, 'bilstm_model.h5')
bilstm_preprocessor_path = os.path.join(models_dir, 'bilstm_preprocessor.pkl')

RF_model      = joblib.load(rf_model_path)
RF_vectorizer = joblib.load(rf_vectorizer_path)

bilstm_model = load_model(bilstm_model_path)
with open(bilstm_preprocessor_path, "rb") as f:
    bilstm_preprocessor = pickle.load(f)

tokenizer = bilstm_preprocessor['tokenizer']
max_len   = bilstm_preprocessor['max_len']

# --- Predict endpoint ---
@app.route('/predict_sql', methods=['POST'])
def predict_sql():
    data    = request.json
    payload = data.get('payload', '')

    RF_features   = RF_vectorizer.transform([payload]).toarray()
    RF_prediction = RF_model.predict(RF_features)[0]

    sequence   = tokenizer.texts_to_sequences([payload])
    padded_seq = pad_sequences(sequence, maxlen=max_len, padding='post')
    lstm_pred  = bilstm_model.predict(padded_seq)[0][0]

    
    threshold = detection_config["threshold"] / detection_config["sensitivity_boost"]
    is_threat = RF_prediction == 1 or lstm_pred > threshold

    if is_threat:
        log_event('sqli', RF_prediction, lstm_pred, payload)

    return jsonify({
        "is_threat":     bool(is_threat),
        "ml_confidence": int(RF_prediction),
        "dl_confidence": float(lstm_pred)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)