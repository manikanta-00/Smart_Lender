"""
data_utils.py
-------------
Pure-Python (no pandas / numpy) utilities for:
  - loading the loan CSV
  - handling missing values (mean for numeric columns, mode for categorical)
  - encoding categorical variables into numbers
  - splitting into train / test sets

This mirrors the "4. Data Preprocessing & Feature Engineering" step from
the Smart Lender project brief.
"""

import csv
import random
import statistics
from collections import Counter

NUMERIC_COLS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History",
]
CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education",
                     "Self_Employed", "Property_Area"]

FEATURE_ORDER = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area",
]

TARGET_COL = "Loan_Status"


def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
    return rows


def _to_number(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_fill_values(rows):
    """Compute mean for numeric cols and mode for categorical cols."""
    fill = {}
    for col in NUMERIC_COLS:
        values = [_to_number(r.get(col)) for r in rows]
        values = [v for v in values if v is not None]
        fill[col] = round(statistics.mean(values), 2) if values else 0.0

    for col in CATEGORICAL_COLS:
        values = [r.get(col) for r in rows if r.get(col) not in (None, "")]
        fill[col] = Counter(values).most_common(1)[0][0] if values else "Unknown"

    return fill


def apply_fill(rows, fill_values):
    cleaned = []
    for r in rows:
        new_row = dict(r)
        for col in NUMERIC_COLS:
            num = _to_number(new_row.get(col))
            new_row[col] = num if num is not None else fill_values[col]
        for col in CATEGORICAL_COLS:
            if new_row.get(col) in (None, ""):
                new_row[col] = fill_values[col]
        cleaned.append(new_row)
    return cleaned


def build_encoders(rows):
    """Build a {column: {category: index}} mapping for each categorical column."""
    encoders = {}
    for col in CATEGORICAL_COLS:
        categories = sorted(set(r[col] for r in rows))
        encoders[col] = {cat: i for i, cat in enumerate(categories)}
    return encoders


def encode_row(row, encoders):
    """Turn a cleaned row (dict) into a numeric feature vector, in FEATURE_ORDER."""
    vector = []
    for col in FEATURE_ORDER:
        if col in CATEGORICAL_COLS:
            mapping = encoders[col]
            vector.append(float(mapping.get(row[col], 0)))
        else:
            vector.append(float(row[col]))
    return vector


def encode_dataset(rows, encoders):
    X = [encode_row(r, encoders) for r in rows]
    y = [1 if r[TARGET_COL] == "Y" else 0 for r in rows]
    return X, y


def normalize_features(X, mins=None, maxs=None):
    """Min-max normalize features (needed for KNN's distance calculation)."""
    n_features = len(X[0])
    if mins is None or maxs is None:
        mins = [min(row[i] for row in X) for i in range(n_features)]
        maxs = [max(row[i] for row in X) for i in range(n_features)]
    normalized = []
    for row in X:
        new_row = []
        for i in range(n_features):
            span = maxs[i] - mins[i]
            new_row.append((row[i] - mins[i]) / span if span > 0 else 0.0)
        normalized.append(new_row)
    return normalized, mins, maxs


def train_test_split(X, y, test_ratio=0.2, seed=42):
    rng = random.Random(seed)
    indices = list(range(len(X)))
    rng.shuffle(indices)
    n_test = int(len(X) * test_ratio)
    test_idx = set(indices[:n_test])
    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(len(X)):
        if i in test_idx:
            X_test.append(X[i]); y_test.append(y[i])
        else:
            X_train.append(X[i]); y_train.append(y[i])
    return X_train, y_train, X_test, y_test


def accuracy_score(y_true, y_pred):
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true) if y_true else 0.0
