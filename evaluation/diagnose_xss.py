import sys
sys.path.insert(0, "/app")
sys.path.insert(0, "/tmp")

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import app as api
from evaluate import load_dataset, metrics

model = api.dnn_model
prep = api.dnn_preprocessor
tok = prep["tokenizer"]
max_len = prep["max_len"]

print("\n--- Preprocessor ---")
print("keys:", list(prep.keys()))
print("max_len:", max_len)
print("char_level:", getattr(tok, "char_level", None))
print("num_words:", tok.num_words)
print("word_index size:", len(tok.word_index))
print("lower:", getattr(tok, "lower", None))
print("oov_token:", getattr(tok, "oov_token", None))

print("\n--- Model ---")
print("input shape:", model.input_shape)
for layer in model.layers:
    if layer.__class__.__name__ == "Embedding":
        print("embedding input_dim:", layer.input_dim, "| output_dim:", layer.output_dim)
model.summary()

texts, y = load_dataset("/tmp/xss.csv")
_, idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
texts = [texts[i] for i in idx]
y = y[idx]
seqs = tok.texts_to_sequences(texts)

print("\n--- Padding / truncation test (test split) ---")
for padding in ("post", "pre"):
    for truncating in ("post", "pre"):
        X = pad_sequences(seqs, maxlen=max_len, padding=padding, truncating=truncating)
        pred = (model.predict(X, batch_size=256, verbose=0).reshape(-1) > 0.5).astype(int)
        m = metrics(y, pred)
        print(f"padding={padding:<5} truncating={truncating:<5} "
              f"acc={m['accuracy']:.4f} f1={m['f1']:.4f} fp={m['fp']} fn={m['fn']}")

lengths = np.array([len(s) for s in seqs])
print(f"\nSequence lengths: median={int(np.median(lengths))}, "
      f"95th pct={int(np.percentile(lengths, 95))}, "
      f"longer than max_len={int((lengths > max_len).sum())} of {len(lengths)}")
print(f"Empty sequences (no known tokens): {int((lengths == 0).sum())}")