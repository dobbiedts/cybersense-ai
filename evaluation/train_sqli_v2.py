#!/usr/bin/env python3
"""
SQLi v2 retraining script.

Trains:
  1. Random Forest + character n-gram TF-IDF
  2. Character-level BiLSTM

Adds SQLi-negative examples from:
  - Generated everyday text
  - Generated legitimate SQL statements
  - HTML and XSS examples from XSS_dataset.csv

Evaluates:
  - Original held-out SQLi dataset
  - Held-out everyday benign templates
  - Held-out legitimate SQL templates
  - Held-out HTML/XSS dataset samples
  - Diagnostic spot checks

Run inside the sql-api container:

  python /tmp/train_sqli_v2.py \
      --sqli /tmp/sqli.csv \
      --xss /tmp/xss.csv \
      --max-len 300

Output:
  /tmp/sqli_v2_artifacts/

The live API models are not overwritten.
"""

import argparse
import pickle
import random
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =====================================================================
# 1. ARGUMENTS AND RANDOM SEEDS
# =====================================================================

parser = argparse.ArgumentParser(
    description="Retrain SQLi Random Forest and BiLSTM models."
)

parser.add_argument(
    "--sqli",
    required=True,
    help="Path to the SQLi CSV, e.g. Modified_SQL_Dataset.csv",
)

parser.add_argument(
    "--xss",
    required=True,
    help="Path to the XSS CSV, e.g. XSS_dataset.csv",
)

parser.add_argument(
    "--preprocessor",
    default="/models/bilstm_preprocessor.pkl",
    help="Existing API preprocessor used to obtain the Tokenizer class",
)

parser.add_argument(
    "--max-len",
    type=int,
    default=300,
    help="Maximum number of characters processed by the BiLSTM",
)

parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument(
    "--synthetic",
    type=int,
    default=4000,
    help="Number of generated everyday benign training examples",
)

parser.add_argument(
    "--sql-negatives",
    type=int,
    default=4000,
    help="Number of generated legitimate SQL training examples",
)

parser.add_argument(
    "--heldout",
    type=int,
    default=1000,
    help="Number of held-out examples for each generated benign category",
)

parser.add_argument(
    "--xss-test-size",
    type=float,
    default=0.20,
    help="Fraction of eligible XSS samples reserved for evaluation",
)

parser.add_argument(
    "--rf-trees",
    type=int,
    default=300,
    help="Number of Random Forest trees",
)

parser.add_argument(
    "--max-features",
    type=int,
    default=50000,
    help="Maximum number of TF-IDF features",
)

parser.add_argument(
    "--output-dir",
    default="/tmp/sqli_v2_artifacts",
    help="Directory for the four new model artifacts",
)

args = parser.parse_args()

if args.max_len < 10:
    parser.error("--max-len must be at least 10")

if args.epochs < 1 or args.rf_trees < 1 or args.max_features < 1:
    parser.error("--epochs, --rf-trees and --max-features must be positive")

if min(args.synthetic, args.sql_negatives, args.heldout) < 0:
    parser.error("Sample counts cannot be negative")

if not 0 < args.xss_test_size < 1:
    parser.error("--xss-test-size must be greater than 0 and less than 1")

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

DETECTION_THRESHOLD = 0.5


# =====================================================================
# 2. LOAD THE EXISTING PREPROCESSOR'S TOKENIZER CLASS
# =====================================================================
#
# Do not import Tokenizer directly from
# tensorflow.keras.preprocessing.text.
#
# Instead, load the existing preprocessor and reuse its Tokenizer class.
# The existing max_len is deliberately NOT reused: this new model is
# character-level and therefore uses 300 characters by default.
#
# Loading the preprocessor directly avoids importing the full Flask API
# and loading the current production models into this training process.

with open(args.preprocessor, "rb") as file:
    existing_preprocessor = pickle.load(file)

if "tokenizer" not in existing_preprocessor:
    raise ValueError(
        f"No tokenizer found in {args.preprocessor}"
    )

TokenizerClass = type(existing_preprocessor["tokenizer"])

max_len = args.max_len

print(f"Tokenizer class: {TokenizerClass.__name__}")
print(f"New character-level max_len: {max_len}")
print("Existing preprocessor max_len is not being reused.")


# =====================================================================
# 3. DATASET HELPERS
# =====================================================================

def normalise_text(text):
    """Normalise whitespace and case for duplicate detection only."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def normalise_column_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_text_column(frame, excluded_columns=None):
    """Find a likely query/text column."""
    excluded_columns = set(excluded_columns or [])

    columns_by_name = {
        normalise_column_name(column): column
        for column in frame.columns
    }

    candidates = (
        "query",
        "sentence",
        "payload",
        "text",
        "sql",
        "input",
        "request",
        "content",
        "sample",
        "data",
    )

    for candidate in candidates:
        column = columns_by_name.get(candidate)

        if column is not None and column not in excluded_columns:
            return column

    remaining = [
        column
        for column in frame.columns
        if column not in excluded_columns
        and not str(column).lower().startswith("unnamed")
    ]

    if not remaining:
        raise ValueError(
            f"Could not identify a text column in {list(frame.columns)}"
        )

    return remaining[0]


def label_to_int(value):
    """Convert common binary label formats into 0 or 1."""
    if pd.isna(value):
        return None

    value_text = str(value).strip().lower()

    if value_text in {
        "0", "0.0", "false", "no", "benign",
        "normal", "legitimate", "negative", "non-sqli",
    }:
        return 0

    if value_text in {
        "1", "1.0", "true", "yes", "attack",
        "malicious", "sqli", "sql injection", "positive",
    }:
        return 1

    try:
        number = float(value_text)

        if number == 0:
            return 0

        if number == 1:
            return 1

    except (TypeError, ValueError):
        pass

    raise ValueError(
        f"Unsupported label value {value!r}. "
        "Expected binary labels 0 or 1."
    )


# =====================================================================
# 4. LOAD AND CLEAN THE ORIGINAL SQLi DATASET
# =====================================================================

def load_sqli_dataset(path):
    frame = pd.read_csv(
        path,
        low_memory=False,
        encoding_errors="replace",
    )

    if frame.empty:
        raise ValueError(f"SQLi dataset is empty: {path}")

    columns_by_name = {
        normalise_column_name(column): column
        for column in frame.columns
    }

    label_candidates = (
        "label",
        "class",
        "target",
        "issqli",
        "attack",
        "isattack",
        "category",
    )

    label_column = next(
        (
            columns_by_name[name]
            for name in label_candidates
            if name in columns_by_name
        ),
        None,
    )

    if label_column is None:
        candidate = frame.columns[-1]

        values = {
            str(value).strip().lower()
            for value in frame[candidate].dropna().unique()
        }

        if values and values.issubset({
            "0", "1", "0.0", "1.0",
            "true", "false", "yes", "no",
        }):
            label_column = candidate
        else:
            raise ValueError(
                "Could not identify the SQLi label column. "
                f"Available columns: {list(frame.columns)}"
            )

    text_column = find_text_column(
        frame,
        excluded_columns={label_column},
    )

    result = pd.DataFrame({
        "text": frame[text_column].fillna("").astype(str).str.strip(),
        "label": frame[label_column].map(label_to_int),
    })

    result = result[
        (result["text"] != "")
        & result["label"].notna()
    ].copy()

    result["label"] = result["label"].astype(int)
    result["_normalised"] = result["text"].map(normalise_text)

    # Remove identical payloads with contradictory labels.
    label_counts = result.groupby("_normalised")["label"].nunique()

    conflicting_keys = set(
        label_counts[label_counts > 1].index
    )

    if conflicting_keys:
        print(
            "Removing conflicting-label payloads:",
            len(conflicting_keys),
        )

        result = result[
            ~result["_normalised"].isin(conflicting_keys)
        ]

    before_deduplication = len(result)

    result = result.drop_duplicates(
        subset="_normalised",
        keep="first",
    ).reset_index(drop=True)

    print(
        "Duplicate SQLi rows removed:",
        before_deduplication - len(result),
    )

    if result.empty or result["label"].nunique() != 2:
        raise ValueError(
            "The SQLi dataset must contain both benign (0) "
            "and SQLi (1) examples."
        )

    texts = result["text"].tolist()
    labels = result["label"].to_numpy(dtype=np.int32)

    print(f"\nSQLi dataset: {len(texts):,} unique rows")
    print(f"Text column: {text_column}")
    print(f"Label column: {label_column}")
    print(f"Benign rows: {int((labels == 0).sum()):,}")
    print(f"SQLi rows: {int((labels == 1).sum()):,}")

    return texts, labels


# =====================================================================
# 5. LOAD TEXT FROM THE XSS DATASET
# =====================================================================
#
# The XSS labels are intentionally not used here.
#
# For a SQLi detector, an XSS payload is a negative example unless it
# also contains SQL injection. A separate XSS detector can still flag it.

def load_xss_texts(path):
    frame = pd.read_csv(
        path,
        low_memory=False,
        encoding_errors="replace",
    )

    if frame.empty:
        raise ValueError(f"XSS dataset is empty: {path}")

    text_column = find_text_column(frame)

    texts = (
        frame[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )

    texts = [
        text for text in texts
        if text and normalise_text(text)
    ]

    seen = set()
    unique_texts = []

    for text in texts:
        key = normalise_text(text)

        if key not in seen:
            seen.add(key)
            unique_texts.append(text)

    print(f"\nXSS text rows loaded: {len(texts):,}")
    print(f"XSS text column: {text_column}")
    print(f"Unique XSS text samples: {len(unique_texts):,}")

    return unique_texts


# =====================================================================
# 6. FILTER LIKELY SQL INJECTION FROM THE XSS NEGATIVE CORPUS
# =====================================================================
#
# These filters are conservative heuristics, not a complete SQL parser.
# Samples with known SQL-like patterns are excluded rather than labelled
# benign.
#
# Also exclude every exact normalised overlap with the original SQLi
# dataset. This avoids train/test leakage across the two source datasets.

SQL_PATTERNS = [
    re.compile(r"\bunion\s+(?:all\s+)?select\b", re.I),
    re.compile(r"\bselect\b[\s\S]{0,200}\bfrom\b", re.I),
    re.compile(r"\binsert\s+into\b", re.I),
    re.compile(r"\bupdate\s+[`\"\w]+\s+set\b", re.I),
    re.compile(r"\bdelete\s+from\b", re.I),
    re.compile(r"\bdrop\s+(?:table|database|schema)\b", re.I),
    re.compile(r"\b(?:alter|create|truncate)\s+table\b", re.I),
    re.compile(r"\binformation_schema\b", re.I),
    re.compile(r"\binto\s+outfile\b", re.I),
    re.compile(r"\bload_file\s*\(", re.I),
    re.compile(r"\b(?:sleep|benchmark)\s*\(", re.I),
    re.compile(
        r"""(?:['"]\s*)?\b(?:or|and)\b\s+['"]?\w+['"]?\s*=\s*['"]?\w+['"]?""",
        re.I,
    ),
]


def contains_sql_pattern(text):
    return any(pattern.search(text) for pattern in SQL_PATTERNS)


# =====================================================================
# 7. SYNTHETIC EVERYDAY BENIGN TEXT
# =====================================================================

EVERYDAY_VALUES_A = {
    "name": [
        "Sean", "Mary", "Liam", "Aoife", "Kevin",
        "Grace", "Ola", "Ada", "Tunde", "Niamh",
    ],
    "surname": [
        "O'Brien", "O'Connor", "D'Souza",
        "O'Neill", "D'Arcy", "O'Hara",
    ],
    "product": [
        "laptop", "headphones", "keyboard", "monitor",
        "shoes", "jacket", "phone case",
    ],
    "adjective": [
        "great", "cheap", "fast", "lovely",
        "broken", "brilliant", "too small",
    ],
    "brand": [
        "Dell", "Sony", "Apple", "Nike", "Adidas", "Samsung",
    ],
    "city": [
        "Aberdeen", "Glasgow", "Dundee", "Leeds", "Cork", "Lagos",
    ],
    "term": [
        "delivery status", "student account",
        "refund policy", "train ticket", "order history",
    ],
}

EVERYDAY_VALUES_B = {
    "name": [
        "James", "Fatima", "Chloe", "Daniel", "Zara",
        "Ewan", "Sophie", "Erin", "Kai", "Amara",
    ],
    "surname": [
        "O'Sullivan", "D'Angelo", "O'Donnell",
        "N'Diaye", "O'Reilly", "L'Estrange",
    ],
    "product": [
        "mouse", "tablet", "trainers", "backpack",
        "charger", "printer", "speaker",
    ],
    "adjective": [
        "amazing", "expensive", "slow", "decent",
        "faulty", "excellent", "a bit loud",
    ],
    "brand": [
        "Lenovo", "Bose", "Logitech", "Puma", "HP", "Canon",
    ],
    "city": [
        "Edinburgh", "Inverness", "Manchester",
        "Belfast", "Bristol", "Accra",
    ],
    "term": [
        "account settings", "parcel collection",
        "membership plan", "return label", "invoice status",
    ],
}

EVERYDAY_TEMPLATES_TRAIN = [
    "{surname}",
    "{name} {surname}",
    "it's a {adjective} {product}",
    "men's {product}",
    "women's {product}",
    "{brand}'s new {product}",
    "I can't find my {product}",
    "don't ship to {city} please",
    "what's the price of the {product}?",
    "price < {number}",
    "{product} under £{number}",
    "https://example.com/search?q={term}&page={number}",
    "{name}.{number}@example.com",
    "{number}% off {product}s",
    "please search for {term}",
    "customer {name} requested a refund",
    "the total is £{number} including delivery",
]

EVERYDAY_TEMPLATES_HELDOUT = [
    "{name}'s parcel should arrive tomorrow",
    "where's the nearest shop in {city}?",
    "the {brand} {product} arrived {adjective}",
    "please don't charge me twice for order {number}",
    "my contact is {name}.{number}@mail.example.org",
    "browse {term} at https://example.org/catalog/{number}",
    "the customer wrote: 'I didn't receive the {product}'",
    "£{number} was taken from my account yesterday",
    "is {product} available in {city} this week?",
    "rating for {brand} is {number} out of 5",
    "the account belongs to {name} {surname}",
    "I would like to exchange this {adjective} {product}",
    "reference={number}&source={term}",
    "delivery instructions: leave the {product} by the front door",
    "the {brand} product costs £{number} today",
]


# =====================================================================
# 8. LEGITIMATE SQL NEGATIVES
# =====================================================================
#
# These are examples of normal SQL statements, often parameterised.
# Parameterised queries remain necessary in the actual application.

SQL_VALUES_A = {
    "table": [
        "users", "orders", "products", "customers",
        "employees", "tickets", "payments",
    ],
    "column": [
        "id", "name", "email", "status",
        "price", "created_at", "category", "user_id",
    ],
    "column2": [
        "account_id", "region", "active",
        "quantity", "department", "order_id",
    ],
    "columns": [
        "id, name",
        "id, status",
        "name, price",
        "email, created_at",
    ],
}

SQL_VALUES_B = {
    "table": [
        "inventory", "audit_log", "subscriptions",
        "shipments", "invoices", "staff_records", "catalogue",
    ],
    "column": [
        "sku", "event_type", "available", "country",
        "description", "updated_at", "total",
    ],
    "column2": [
        "tenant_id", "owner_id", "is_active",
        "department_id", "reference_id",
    ],
    "columns": [
        "sku, description",
        "id, event_type",
        "reference, status",
        "owner_id, updated_at",
    ],
}

SQL_TEMPLATES_TRAIN = [
    "SELECT {column} FROM {table} WHERE {column} = ?",
    "SELECT {columns} FROM {table} WHERE {column} = ? AND {column2} = ?",
    "SELECT {columns} FROM {table} WHERE {column} LIKE ?",
    "SELECT COUNT(*) FROM {table} WHERE {column} = ?",
    "INSERT INTO {table} ({column}, {column2}) VALUES (?, ?)",
    "UPDATE {table} SET {column} = ? WHERE id = ?",
    "DELETE FROM {table} WHERE id = ?",
    "SELECT {columns} FROM {table} ORDER BY {column} DESC LIMIT ?",
    "SELECT {columns} FROM {table} WHERE {column} BETWEEN ? AND ?",
    "SELECT {columns} FROM {table} WHERE email = ?",
    "SELECT * FROM customers WHERE last_name = 'O''Brien'",
    "SELECT * FROM products WHERE name LIKE '%wireless%'",
    "SELECT id FROM orders WHERE status IN (?, ?, ?)",
    "SELECT * FROM users WHERE username = ? AND password_hash = ?",
]

SQL_TEMPLATES_HELDOUT = [
    "SELECT DISTINCT {column} FROM {table} WHERE status IN (?, ?)",
    "SELECT {columns} FROM {table} WHERE created_at >= ? ORDER BY created_at DESC LIMIT ?",
    "INSERT INTO audit_log (event_type, actor_id, note) VALUES (?, ?, ?)",
    "UPDATE {table} SET {column} = ? WHERE {column2} = ? AND is_active = TRUE",
    "SELECT {columns} FROM {table} WHERE COALESCE({column}, '') = ?",
    "SELECT {columns} FROM {table} WHERE {column} IS NULL",
    "SELECT SUM({column}) FROM {table} WHERE {column2} = ?",
    "SELECT {columns} FROM {table} WHERE id IN (?, ?, ?, ?)",
    "SELECT {columns} FROM {table} WHERE last_name = 'O''Reilly'",
    "SELECT {columns} FROM {table} WHERE price BETWEEN 10 AND 90",
    "SELECT {columns} FROM {table} WHERE LOWER(email) = LOWER(?)",
    "SELECT {columns} FROM {table} WHERE archived_at IS NULL ORDER BY id ASC",
    "INSERT INTO {table} ({column}, {column2}, status) VALUES (?, ?, 'pending')",
    "SELECT COALESCE(SUM({column}), 0) FROM {table} WHERE {column2} = ?",
]


def generate_templates(templates, values, count):
    result = []

    for _ in range(count):
        template = random.choice(templates)

        fields = {
            key: random.choice(items)
            for key, items in values.items()
        }

        fields["number"] = random.randint(1, 9999)

        result.append(template.format(**fields))

    return result


# =====================================================================
# 9. METRIC HELPERS
# =====================================================================

def report_metrics(name, y_true, predictions):
    y_true = np.asarray(y_true, dtype=np.int32)
    predictions = np.asarray(predictions, dtype=np.int32)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    print(
        f"{name}: "
        f"accuracy={accuracy_score(y_true, predictions):.4f} "
        f"precision={precision_score(y_true, predictions, zero_division=0):.4f} "
        f"recall={recall_score(y_true, predictions, zero_division=0):.4f} "
        f"f1={f1_score(y_true, predictions, zero_division=0):.4f} "
        f"FP={fp} FN={fn} TN={tn} TP={tp}"
    )


def report_false_positive_rate(name, predictions):
    predictions = np.asarray(predictions, dtype=np.int32)

    if len(predictions) == 0:
        print(f"{name}: no samples available")
        return

    flagged = int(predictions.sum())

    print(
        f"{name}: flagged {flagged}/{len(predictions)} "
        f"benign samples ({flagged / len(predictions):.2%})"
    )


# =====================================================================
# 10. LOAD AND SPLIT THE ORIGINAL SQLi DATA
# =====================================================================

texts, labels = load_sqli_dataset(args.sqli)

indices = np.arange(len(labels))

train_indices, test_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=args.seed,
    stratify=labels,
)

core_indices, validation_indices = train_test_split(
    train_indices,
    test_size=0.10,
    random_state=args.seed,
    stratify=labels[train_indices],
)

x_core = [texts[i] for i in core_indices]
y_core = labels[core_indices]

x_val = [texts[i] for i in validation_indices]
y_val = labels[validation_indices]

x_test = [texts[i] for i in test_indices]
y_test = labels[test_indices]

# All original SQLi payloads are used to identify cross-dataset overlaps.
all_original_sqli_texts = {
    normalise_text(text)
    for text in texts
}

# In addition, preserve the set of source payloads labelled SQLi.
known_sqli_positive = {
    normalise_text(text)
    for text, label in zip(texts, labels)
    if label == 1
}


# =====================================================================
# 11. FILTER AND SPLIT THE XSS CORPUS
# =====================================================================

xss_texts = load_xss_texts(args.xss)

xss_eligible = []
sql_pattern_skipped = 0
source_overlap_skipped = 0
positive_overlap_skipped = 0

for text in xss_texts:
    key = normalise_text(text)

    # Do not label SQL-like data as harmless for this detector.
    if contains_sql_pattern(text):
        sql_pattern_skipped += 1
        continue

    # Avoid duplicate payloads crossing between the SQLi and XSS datasets.
    if key in all_original_sqli_texts:
        source_overlap_skipped += 1
        continue

    # Explicit safety check for the known SQLi-positive set.
    if key in known_sqli_positive:
        positive_overlap_skipped += 1
        continue

    xss_eligible.append(text)

print("\nXSS filtering results:")
print(f"  Samples skipped for SQL-like syntax: {sql_pattern_skipped:,}")
print(f"  Samples overlapping original SQLi data: {source_overlap_skipped:,}")
print(f"  Samples overlapping known SQLi positives: {positive_overlap_skipped:,}")
print(f"  Eligible non-SQLi XSS/HTML samples: {len(xss_eligible):,}")

if len(xss_eligible) < 2:
    raise ValueError(
        "Not enough XSS samples remain after filtering. "
        "Check the XSS text column and the SQL-pattern filters."
    )

xss_train, xss_test = train_test_split(
    xss_eligible,
    test_size=args.xss_test_size,
    random_state=args.seed,
    shuffle=True,
)

print(f"  XSS/HTML training examples: {len(xss_train):,}")
print(f"  Held-out XSS/HTML examples: {len(xss_test):,}")


# =====================================================================
# 12. GENERATE EVERYDAY TEXT AND LEGITIMATE SQL
# =====================================================================

everyday_train = generate_templates(
    EVERYDAY_TEMPLATES_TRAIN,
    EVERYDAY_VALUES_A,
    args.synthetic,
)

legitimate_sql_train = generate_templates(
    SQL_TEMPLATES_TRAIN,
    SQL_VALUES_A,
    args.sql_negatives,
)

everyday_heldout = generate_templates(
    EVERYDAY_TEMPLATES_HELDOUT,
    EVERYDAY_VALUES_B,
    args.heldout,
)

legitimate_sql_heldout = generate_templates(
    SQL_TEMPLATES_HELDOUT,
    SQL_VALUES_B,
    args.heldout,
)


# =====================================================================
# 13. BUILD THE FINAL TRAINING SET
# =====================================================================
#
# Original SQLi validation and test rows remain separate.
# Every synthetic example and XSS/HTML sample added below is assigned
# label 0 for the SQLi classification task.

x_train = (
    x_core
    + everyday_train
    + legitimate_sql_train
    + xss_train
)

y_train = np.concatenate([
    y_core,
    np.zeros(
        len(everyday_train)
        + len(legitimate_sql_train)
        + len(xss_train),
        dtype=np.int32,
    ),
])

shuffle_indices = np.random.permutation(len(x_train))

x_train = [
    x_train[i]
    for i in shuffle_indices
]

y_train = y_train[shuffle_indices]

positive_rate = float(y_train.mean())

class_weight = {
    0: 0.5 / max(1e-8, 1.0 - positive_rate),
    1: 0.5 / max(1e-8, positive_rate),
}

print("\nTraining composition:")
print(f"  Original SQLi training samples: {len(x_core):,}")
print(f"  Generated everyday negatives: {len(everyday_train):,}")
print(f"  Generated legitimate-SQL negatives: {len(legitimate_sql_train):,}")
print(f"  XSS/HTML negatives: {len(xss_train):,}")
print(f"  Total training rows: {len(x_train):,}")
print(f"  SQLi positive training labels: {int(y_train.sum()):,}")
print(f"  Real-data validation rows: {len(x_val):,}")
print(f"  Original SQLi test rows: {len(x_test):,}")


# =====================================================================
# 14. TRAIN MODEL 1: RANDOM FOREST
# =====================================================================

print("\n" + "=" * 72)
print("TRAINING RANDOM FOREST + CHARACTER N-GRAM TF-IDF")
print("=" * 72)

rf_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    lowercase=True,
    max_features=args.max_features,
    min_df=2,
    sublinear_tf=True,
    dtype=np.float32,
)

X_train_rf = rf_vectorizer.fit_transform(x_train)
X_test_rf = rf_vectorizer.transform(x_test)

X_everyday_rf = rf_vectorizer.transform(everyday_heldout)
X_sql_rf = rf_vectorizer.transform(legitimate_sql_heldout)
X_xss_rf = rf_vectorizer.transform(xss_test)

print(
    f"TF-IDF training matrix: {X_train_rf.shape[0]:,} rows x "
    f"{X_train_rf.shape[1]:,} features"
)

rf_model = RandomForestClassifier(
    n_estimators=args.rf_trees,
    max_features="sqrt",
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=args.seed,
)

rf_model.fit(X_train_rf, y_train)

rf_attack_column = list(rf_model.classes_).index(1)


def rf_score(matrix):
    return rf_model.predict_proba(matrix)[:, rf_attack_column]


rf_test_scores = rf_score(X_test_rf)
rf_everyday_scores = rf_score(X_everyday_rf)
rf_sql_scores = rf_score(X_sql_rf)
rf_xss_scores = rf_score(X_xss_rf)


# =====================================================================
# 15. TRAIN MODEL 2: CHARACTER-LEVEL BiLSTM
# =====================================================================
#
# The existing API does:
#
#   tokenizer.texts_to_sequences([payload])
#   pad_sequences(..., maxlen=max_len, padding="post")
#
# Keras defaults to truncating="pre", so the script specifies it explicitly.
# The new tokenizer is character-level, and max_len defaults to 300.

print("\n" + "=" * 72)
print("TRAINING CHARACTER-LEVEL BiLSTM")
print("=" * 72)

tokenizer = TokenizerClass(
    char_level=True,
    lower=True,
    filters="",
    oov_token="\x00",
)

tokenizer.fit_on_texts(x_train)

vocab_size = len(tokenizer.word_index) + 1


def encode(batch):
    sequences = tokenizer.texts_to_sequences(batch)

    return pad_sequences(
        sequences,
        maxlen=max_len,
        padding="post",
        truncating="pre",
    )


X_train_dl = encode(x_train)
X_val_dl = encode(x_val)
X_test_dl = encode(x_test)

X_everyday_dl = encode(everyday_heldout)
X_sql_dl = encode(legitimate_sql_heldout)
X_xss_dl = encode(xss_test)

print(f"Character vocabulary: {vocab_size:,}")
print(f"Maximum sequence length: {max_len}")

inputs = layers.Input(
    shape=(max_len,),
    name="char_sequence",
)

x = layers.Embedding(
    input_dim=vocab_size,
    output_dim=64,
    mask_zero=True,
    name="char_embedding",
)(inputs)

x = layers.Bidirectional(
    layers.LSTM(
        64,
        return_sequences=False,
        dropout=0.2,
    ),
    name="bilstm",
)(x)

x = layers.Dense(
    64,
    activation="relu",
    name="dense_features",
)(x)

x = layers.Dropout(
    0.4,
    name="dropout",
)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="sqli_probability",
)(x)

dl_model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="sqli_character_bilstm_v2",
)

dl_model.compile(
    optimizer=optimizers.Adam(
        learning_rate=1e-3,
        clipnorm=1.0,
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

training_callbacks = [
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-5,
    ),
]

dl_model.fit(
    X_train_dl,
    y_train,
    validation_data=(X_val_dl, y_val),
    epochs=args.epochs,
    batch_size=64,
    class_weight=class_weight,
    callbacks=training_callbacks,
    verbose=2,
)


def dl_score(encoded):
    return dl_model.predict(
        encoded,
        batch_size=256,
        verbose=0,
    ).reshape(-1)


dl_test_scores = dl_score(X_test_dl)
dl_everyday_scores = dl_score(X_everyday_dl)
dl_sql_scores = dl_score(X_sql_dl)
dl_xss_scores = dl_score(X_xss_dl)


# =====================================================================
# 16. EVALUATE THE ORIGINAL SQLi TEST SET
# =====================================================================
#
# The live API declares a threat when either model exceeds 0.5.
# Therefore, measure both models and that combined OR verdict.

rf_test_prediction = (
    rf_test_scores > DETECTION_THRESHOLD
).astype(np.int32)

dl_test_prediction = (
    dl_test_scores > DETECTION_THRESHOLD
).astype(np.int32)

combined_test_prediction = (
    (rf_test_scores > DETECTION_THRESHOLD)
    | (dl_test_scores > DETECTION_THRESHOLD)
).astype(np.int32)

print("\n" + "=" * 72)
print("ORIGINAL SQLi TEST SPLIT")
print("=" * 72)

report_metrics(
    "Random Forest",
    y_test,
    rf_test_prediction,
)

report_metrics(
    "BiLSTM",
    y_test,
    dl_test_prediction,
)

report_metrics(
    "Combined API verdict (RF OR BiLSTM)",
    y_test,
    combined_test_prediction,
)


# =====================================================================
# 17. EVALUATE HELD-OUT EVERYDAY BENIGN TEXT
# =====================================================================

print("\n" + "=" * 72)
print("HELD-OUT EVERYDAY BENIGN TEXT")
print("=" * 72)

report_false_positive_rate(
    "Random Forest",
    (rf_everyday_scores > DETECTION_THRESHOLD).astype(np.int32),
)

report_false_positive_rate(
    "BiLSTM",
    (dl_everyday_scores > DETECTION_THRESHOLD).astype(np.int32),
)

everyday_combined = (
    (rf_everyday_scores > DETECTION_THRESHOLD)
    | (dl_everyday_scores > DETECTION_THRESHOLD)
).astype(np.int32)

report_false_positive_rate(
    "Combined API verdict",
    everyday_combined,
)


# =====================================================================
# 18. EVALUATE HELD-OUT LEGITIMATE SQL
# =====================================================================

print("\n" + "=" * 72)
print("HELD-OUT LEGITIMATE SQL")
print("=" * 72)

report_false_positive_rate(
    "Random Forest",
    (rf_sql_scores > DETECTION_THRESHOLD).astype(np.int32),
)

report_false_positive_rate(
    "BiLSTM",
    (dl_sql_scores > DETECTION_THRESHOLD).astype(np.int32),
)

sql_combined = (
    (rf_sql_scores > DETECTION_THRESHOLD)
    | (dl_sql_scores > DETECTION_THRESHOLD)
).astype(np.int32)

report_false_positive_rate(
    "Combined API verdict",
    sql_combined,
)


# =====================================================================
# 19. EVALUATE HELD-OUT XSS/HTML DATA
# =====================================================================
#
# These are SQLi false-positive measurements. They do not measure whether
# XSS is detected by your separate XSS model.

print("\n" + "=" * 72)
print("HELD-OUT XSS/HTML: SQLi FALSE-POSITIVE RATE")
print("=" * 72)

report_false_positive_rate(
    "Random Forest",
    (rf_xss_scores > DETECTION_THRESHOLD).astype(np.int32),
)

report_false_positive_rate(
    "BiLSTM",
    (dl_xss_scores > DETECTION_THRESHOLD).astype(np.int32),
)

xss_combined = (
    (rf_xss_scores > DETECTION_THRESHOLD)
    | (dl_xss_scores > DETECTION_THRESHOLD)
).astype(np.int32)

report_false_positive_rate(
    "Combined API verdict",
    xss_combined,
)


# =====================================================================
# 20. SPOT CHECKS
# =====================================================================
#
# XSS and ordinary HTML are expected to be benign FOR SQLi.
# Genuine SQL injection examples are expected to be labelled as SQLi.

checks = [
    # Everyday benign text
    ("O'Brien", 0),
    ("I can't find my order", 0),
    ("men's shoes", 0),
    ("https://shop.example/item?id=123&ref=home", 0),
    ("Fish & Chips < £10", 0),

    # Legitimate SQL
    ("SELECT * FROM users WHERE id = ?", 0),
    ("SELECT * FROM customers WHERE last_name = 'O''Brien'", 0),

    # Required HTML/XSS negative checks
    ("<a href=x onclick=alert(1)>click</a>", 0),
    ("<img src=x onerror=alert(1)>", 0),
    ("<b>bold</b> width=100", 0),

    # Additional non-SQLi examples
    ("<script>alert(document.cookie)</script>", 0),
    ("<svg/onload=alert('xss')>", 0),
    ("javascript:alert(1)", 0),

    # SQL injection examples
    ("' OR '1'='1", 1),
    ("' OR 1=1 --", 1),
    ("admin'--", 1),
    ("1 UNION SELECT username, password FROM users", 1),
    ("' UNION SELECT NULL,NULL--", 1),
    ("'; DROP TABLE users; --", 1),
    ("') OR ('a'='a", 1),
]

check_texts = [text for text, _ in checks]

check_expected = np.asarray(
    [label for _, label in checks],
    dtype=np.int32,
)

check_rf_scores = rf_score(
    rf_vectorizer.transform(check_texts)
)

check_dl_scores = dl_score(
    encode(check_texts)
)

check_combined_prediction = (
    (check_rf_scores > DETECTION_THRESHOLD)
    | (check_dl_scores > DETECTION_THRESHOLD)
).astype(np.int32)

print("\n" + "=" * 72)
print("SPOT CHECKS")
print("0 = benign for SQLi; 1 = SQL injection")
print("=" * 72)

for text, expected, rf_score_value, dl_score_value, combined in zip(
    check_texts,
    check_expected,
    check_rf_scores,
    check_dl_scores,
    check_combined_prediction,
):
    status = (
        "OK"
        if int(combined) == int(expected)
        else "REVIEW"
    )

    print(
        f"RF={rf_score_value:.4f} "
        f"BiLSTM={dl_score_value:.4f} "
        f"combined={int(combined)} "
        f"expected={int(expected)} "
        f"{status:<6} {text}"
    )


# =====================================================================
# 21. SAVE THE FOUR API-COMPATIBLE ARTIFACTS
# =====================================================================
#
# Filenames match the existing API's expectations.
# The staging directory keeps the existing live models untouched.

output_directory = Path(args.output_dir)
output_directory.mkdir(parents=True, exist_ok=True)

rf_model_path = output_directory / "RandomForest (1).pkl"
rf_vectorizer_path = output_directory / "sqlvectorizer (2).pkl"
dl_model_path = output_directory / "bilstm_model.h5"
dl_preprocessor_path = output_directory / "bilstm_preprocessor.pkl"

joblib.dump(rf_model, rf_model_path)
joblib.dump(rf_vectorizer, rf_vectorizer_path)

dl_model.save(str(dl_model_path))

with open(dl_preprocessor_path, "wb") as file:
    pickle.dump(
        {
            "tokenizer": tokenizer,
            "max_len": max_len,
            "padding": "post",
            "truncating": "pre",
            "char_level": True,
        },
        file,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

print("\n" + "=" * 72)
print("TRAINING COMPLETE")
print("=" * 72)

print("\nSaved artifacts:")
print(f"  {rf_model_path}")
print(f"  {rf_vectorizer_path}")
print(f"  {dl_model_path}")
print(f"  {dl_preprocessor_path}")

print("\nReview these results before replacing live models:")
print("  1. Original SQLi test precision, recall and F1")
print("  2. Combined API false-positive rate on held-out XSS/HTML")
print("  3. The three required HTML spot checks")
print("  4. Genuine SQLi spot checks and your existing evasion test")

print(
    "\nThe BiLSTM uses character-level tokenisation, "
    f"max_len={max_len}, post-padding and pre-truncation."
)

print("Live models have not been overwritten.")