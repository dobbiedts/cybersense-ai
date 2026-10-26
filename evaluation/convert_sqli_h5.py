"""
Convert the SQLi v2 BiLSTM from .h5 to the native .keras format.

The .h5 file stores a masking op ("NotEqual", from mask_zero=True) that
Keras 3 can't rebuild when loading. The trained weights are fine, so this
rebuilds the same architecture, loads the weights, checks the scores match
the training run, and saves a .keras file that loads cleanly.

Runs inside any container with the same TensorFlow version (e.g. xss-api).

Usage:
    python /tmp/convert_sqli_h5.py --h5 /tmp/sqli_bilstm_v2.h5 \
        --preprocessor /tmp/sqli_bilstm_preprocessor_v2.pkl \
        --out /tmp/sqli_bilstm_v2.keras
"""
import argparse
import pickle

import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences

parser = argparse.ArgumentParser()
parser.add_argument("--h5", required=True)
parser.add_argument("--preprocessor", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

with open(args.preprocessor, "rb") as f:
    prep = pickle.load(f)

tokenizer = prep["tokenizer"]
max_len = prep["max_len"]
vocab_size = len(tokenizer.word_index) + 1
print(f"vocab_size={vocab_size} max_len={max_len}")

# Same architecture as the training run
inputs = layers.Input(shape=(max_len,), name="char_sequence")
x = layers.Embedding(vocab_size, 64, mask_zero=True, name="char_embedding")(inputs)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.2), name="bilstm")(x)
x = layers.Dense(64, activation="relu", name="dense_features")(x)
x = layers.Dropout(0.4, name="dropout")(x)
outputs = layers.Dense(1, activation="sigmoid", name="sqli_probability")(x)
model = models.Model(inputs, outputs, name="sqli_character_bilstm_v2")

model.load_weights(args.h5)
print("weights loaded")

# Scores from the training run's spot checks, to confirm the conversion
expected = [
    ("' OR '1'='1", 1.0000),
    ("I can't find my order", 0.0012),
    ("') OR ('a'='a", 0.0289),
    ("1 UNION SELECT username, password FROM users", 0.2733),
    ("<a href=x onclick=alert(1)>click</a>", 0.0000),
]
padded = pad_sequences(
    tokenizer.texts_to_sequences([t for t, _ in expected]),
    maxlen=max_len,
    padding=prep.get("padding", "post"),
    truncating=prep.get("truncating", "pre"),
)
scores = model.predict(padded, verbose=0).reshape(-1)

ok = True
print("\nscore    training  match  payload")
for (text, want), got in zip(expected, scores):
    match = abs(float(got) - want) < 0.01
    ok = ok and match
    print(f"{got:.4f}   {want:.4f}    {'yes' if match else 'NO '}    {text}")

if not ok:
    raise SystemExit("\nScores don't match the training run. Not saving.")

model.save(args.out)
print(f"\nAll scores match. Saved {args.out}")