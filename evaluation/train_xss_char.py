"""
Retrain the XSS attention model with a character-level tokenizer that keeps
punctuation (< > " ' = ( ) / ; :), which the original word tokenizer stripped.

Runs INSIDE the xss-api container so the saved files match its TensorFlow version.

Usage:
    python /tmp/train_xss_char.py --data /tmp/xss.csv
"""
import argparse
import pickle
import sys

import numpy as np

sys.path.insert(0, "/app")
sys.path.insert(0, "/tmp")

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import app
from app import AttentionLayer
from evaluate import load_dataset, metrics

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--max-len", type=int, default=300)
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

tf.random.set_seed(args.seed)
np.random.seed(args.seed)

texts, y = load_dataset(args.data)
idx = np.arange(len(y))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=args.seed)
x_train = [texts[i] for i in train_idx]
x_test = [texts[i] for i in test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Same Tokenizer class the API already uses (works across Keras versions)
Tokenizer = type(app.dnn_preprocessor["tokenizer"])

# Character-level tokenizer, fitted on TRAINING data only
tokenizer = Tokenizer(char_level=True, lower=True, filters="", oov_token="\x00")
tokenizer.fit_on_texts(x_train)
vocab_size = len(tokenizer.word_index) + 1

def encode(batch):
    seqs = tokenizer.texts_to_sequences(batch)
    return pad_sequences(seqs, maxlen=args.max_len, padding="post", truncating="post")

X_train, X_test = encode(x_train), encode(x_test)

lengths = np.array([len(t) for t in x_train])
print(f"Vocabulary: {vocab_size} characters | max_len: {args.max_len} | "
      f"payloads longer than max_len: {(lengths > args.max_len).mean():.1%}")

# Same architecture as the original attention model
inputs = layers.Input(shape=(args.max_len,))
x = layers.Embedding(vocab_size, 64)(inputs)
x = layers.LSTM(128, return_sequences=True)(x)
x = AttentionLayer(name="attention_layer")(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = models.Model(inputs, outputs)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=args.epochs,
    batch_size=64,
    callbacks=[callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
    verbose=2,
)

scores = model.predict(X_test, batch_size=256, verbose=0).reshape(-1)
m = metrics(y_test, (scores > 0.5).astype(int))
print(f"\nTEST SET (20%, random_state={args.seed}): acc={m['accuracy']:.4f} "
      f"prec={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f} "
      f"fp={m['fp']} fn={m['fn']}")

check = ["<script>alert(document.cookie)</script>",
         "<img src=x onerror=alert(1)>",
         "<p>Hello world</p>",
         "hello world, my name is Ola"]
for text, s in zip(check, model.predict(encode(check), verbose=0).reshape(-1)):
    print(f"{s:.4f}  {text}")

model.save("/tmp/xss_char_model.keras")
with open("/tmp/xss_char_preprocessor.pkl", "wb") as f:
    pickle.dump({"tokenizer": tokenizer, "max_len": args.max_len,
                 "padding": "post", "truncating": "post"}, f)
print("\nSaved /tmp/xss_char_model.keras and /tmp/xss_char_preprocessor.pkl")