"""
CyberSense AI - detection model evaluation.

Runs INSIDE the API container so it uses the exact same models, vectorizers
and preprocessing as the live API.

Usage (inside a container):
    python /tmp/evaluate.py --attack sqli --data /tmp/sqli.csv
    python /tmp/evaluate.py --attack xss  --data /tmp/xss.csv
"""
import argparse
import csv
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/app")
csv.field_size_limit(2**31 - 1)

TEXT_COLUMNS = ["Sentence", "Query", "Payload", "payload", "sentence", "query"]
LABEL_COLUMNS = ["Label", "label"]

# DL thresholds to sweep: 0.5 is the default; 0.403 and 0.357 are the
# effective thresholds the feedback loop produced (boost 1.241 and 1.4).
THRESHOLDS = [0.5, 0.45, 0.403, 0.357, 0.3, 0.25]


# ---------------------------------------------------------------- data
def load_dataset(path):
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        first = f.readline()
        f.seek(0)
        delimiter = "\t" if first.count("\t") > first.count(",") else ","
        quoting = csv.QUOTE_NONE if delimiter == "\t" else csv.QUOTE_MINIMAL
        reader = csv.reader(f, delimiter=delimiter, quoting=quoting)
        header = [h.strip() for h in next(reader)]

        text_idx = next((header.index(c) for c in TEXT_COLUMNS if c in header), None)
        label_idx = next((header.index(c) for c in LABEL_COLUMNS if c in header), None)
        if text_idx is None or label_idx is None:
            sys.exit(f"Could not find text/label columns in header: {header}")

        texts, labels, skipped = [], [], 0
        for row in reader:
            if len(row) <= max(text_idx, label_idx):
                skipped += 1
                continue
            label = row[label_idx].strip()
            if label not in ("0", "1"):
                skipped += 1
                continue
            texts.append(row[text_idx])
            labels.append(int(label))

    print(f"Loaded {len(texts)} rows from {path} "
          f"(columns: '{header[text_idx]}', '{header[label_idx]}', skipped {skipped})")
    return texts, np.array(labels)


def split_test_indices(n, test_size, seed):
    """Reproduce sklearn's train_test_split test rows for the same settings."""
    from sklearn.model_selection import train_test_split
    _, test_idx = train_test_split(np.arange(n), test_size=test_size, random_state=seed)
    return np.sort(test_idx)


# ---------------------------------------------------------------- models
def load_predictors(attack):
    """Import the API module itself so preprocessing is identical to production."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    padding, truncating = "post", "pre"
    if attack == "sqli":
        import appsql as api
        vectorizer, ml_model = api.RF_vectorizer, api.RF_model
        dl_model, tokenizer, max_len = api.bilstm_model, api.tokenizer, api.max_len
    else:
        import app as api
        vectorizer, ml_model = api.tfidf_vectorizer, api.ml_model
        dl_model = api.dnn_model
        tokenizer = api.dnn_preprocessor["tokenizer"]
        max_len = api.dnn_preprocessor["max_len"]
        padding = api.dnn_preprocessor.get("padding", padding)
        truncating = api.dnn_preprocessor.get("truncating", truncating)

    def ml_predict(texts, batch=2000):
        out = []
        for i in range(0, len(texts), batch):
            X = vectorizer.transform(texts[i:i + batch]).toarray()
            out.append(ml_model.predict(X))
        return np.concatenate(out).astype(int)

    def dl_scores(texts):
        seqs = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(seqs, maxlen=max_len, padding=padding, truncating=truncating)
        return dl_model.predict(padded, batch_size=256, verbose=0).reshape(-1)

    return ml_predict, dl_scores


# ---------------------------------------------------------------- metrics
def metrics(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": round((tp + tn) / len(y_true), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fp / (fp + tn) if fp + tn else 0.0, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def print_table(title, rows):
    print(f"\n{title}")
    print(f"{'Model':<26}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}{'FPR':>8}{'FP':>7}{'FN':>7}")
    print("-" * 80)
    for name, m in rows:
        print(f"{name:<26}{m['accuracy']:>8.4f}{m['precision']:>8.4f}{m['recall']:>8.4f}"
              f"{m['f1']:>8.4f}{m['fpr']:>8.4f}{m['fp']:>7}{m['fn']:>7}")


def evaluate(name, y, ml_pred, dl_score):
    dl_pred = (dl_score > 0.5).astype(int)
    detection = [
        ("ML only (Random Forest)", metrics(y, ml_pred)),
        ("DL only", metrics(y, dl_pred)),
        ("ML OR DL (deployed)", metrics(y, ((ml_pred == 1) | (dl_pred == 1)).astype(int))),
        ("ML AND DL (comparison)", metrics(y, ((ml_pred == 1) & (dl_pred == 1)).astype(int))),
    ]
    print_table(f"=== {name}: detection performance ===", detection)

    sweep = []
    for t in THRESHOLDS:
        combined = ((ml_pred == 1) | (dl_score > t)).astype(int)
        sweep.append((f"OR, DL threshold {t}", metrics(y, combined)))
    print_table(f"=== {name}: feedback loop threshold sweep ===", sweep)

    return {
        "rows": int(len(y)),
        "attacks": int(y.sum()),
        "benign": int(len(y) - y.sum()),
        "detection": dict(detection),
        "threshold_sweep": {t: m for t, (_, m) in zip(THRESHOLDS, sweep)},
    }


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", choices=["sqli", "xss"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    texts, y = load_dataset(args.data)
    ml_predict, dl_scores = load_predictors(args.attack)

    start = time.time()
    ml_pred = ml_predict(texts)
    dl_score = dl_scores(texts)
    elapsed = time.time() - start
    print(f"\nScored {len(texts)} payloads in {elapsed:.1f}s "
          f"({1000 * elapsed / len(texts):.2f} ms per payload, batched)")

    results = {
        "attack": args.attack,
        "dataset": args.data,
        "full_dataset": evaluate("FULL DATASET (includes training rows)", y, ml_pred, dl_score),
    }

    hold = split_test_indices(len(y), args.test_size, args.seed)
    results["test_split"] = evaluate(
        f"TEST SPLIT (test_size={args.test_size}, random_state={args.seed})",
        y[hold], ml_pred[hold], dl_score[hold],
    )

    misses = [
        {"payload": texts[i][:300], "label": int(y[i]),
         "ml": int(ml_pred[i]), "dl_score": round(float(dl_score[i]), 4)}
        for i in range(len(y))
        if int((ml_pred[i] == 1) or (dl_score[i] > 0.5)) != y[i]
    ]
    results["misclassified_sample"] = misses[:200]
    results["misclassified_total"] = len(misses)
    print(f"\nMisclassified by deployed OR logic: {len(misses)}")

    out = args.out or f"/tmp/eval_{args.attack}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full results to {out}")


if __name__ == "__main__":
    main()