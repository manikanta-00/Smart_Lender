"""
train.py
--------
End-to-end pipeline that mirrors the project brief's epics:

  Epic 1 - Data Collection            -> generate_dataset.generate_dataset
  Epic 2 - Visualizing & Analysing    -> eda.generate_eda_report
  Epic 3 - Data Pre-processing        -> data_utils (fill missing, encode)
  Epic 4 - Model Building             -> models.py (4 classifiers)
  Epic 5 - Application Building       -> app.py (served afterwards by run.py)

Run directly with:  python -m smart_lender.train
(or it is called automatically by run.py the first time you start the app)
"""

import os
import pickle
import time

from . import data_utils as du
from . import eda
from .generate_dataset import generate_dataset
from .models import (
    DecisionTreeClassifier,
    RandomForestClassifier,
    KNNClassifier,
    GradientBoostingClassifier,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "loan_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model_store", "model.pkl")
EDA_PATH = os.path.join(BASE_DIR, "static", "eda", "eda_report.html")


def run_pipeline(force_regenerate_data=False):
    t0 = time.time()

    # ---- Epic 1: Data Collection ----
    if force_regenerate_data or not os.path.exists(DATA_PATH):
        generate_dataset(DATA_PATH, n_rows=600)
    raw_rows = du.load_csv(DATA_PATH)
    print(f"[train] Loaded {len(raw_rows)} rows from {DATA_PATH}")

    # ---- Epic 2: Visualizing & Analysing the Data ----
    eda.generate_eda_report(raw_rows, EDA_PATH)

    # ---- Epic 3: Data Pre-processing ----
    fill_values = du.compute_fill_values(raw_rows)
    clean_rows = du.apply_fill(raw_rows, fill_values)
    encoders = du.build_encoders(clean_rows)
    X, y = du.encode_dataset(clean_rows, encoders)
    X_train, y_train, X_test, y_test = du.train_test_split(X, y, test_ratio=0.2, seed=42)
    print(f"[train] Train size: {len(X_train)}  Test size: {len(X_test)}")

    # KNN needs normalized features
    X_train_norm, mins, maxs = du.normalize_features(X_train)
    X_test_norm, _, _ = du.normalize_features(X_test, mins, maxs)

    # ---- Epic 4: Machine Learning Model Building ----
    results = {}

    print("[train] Training Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=6, min_samples_split=6, random_state=1)
    dt.fit(X_train, y_train)
    dt_train_acc = du.accuracy_score(y_train, dt.predict(X_train))
    dt_test_acc = du.accuracy_score(y_test, dt.predict(X_test))
    results["Decision Tree"] = {"model": dt, "needs_norm": False,
                                 "train_acc": dt_train_acc, "test_acc": dt_test_acc}

    print("[train] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=25, max_depth=7, min_samples_split=6, random_state=42)
    rf.fit(X_train, y_train)
    rf_train_acc = du.accuracy_score(y_train, rf.predict(X_train))
    rf_test_acc = du.accuracy_score(y_test, rf.predict(X_test))
    results["Random Forest"] = {"model": rf, "needs_norm": False,
                                 "train_acc": rf_train_acc, "test_acc": rf_test_acc}

    print("[train] Training KNN...")
    knn = KNNClassifier(k=9)
    knn.fit(X_train_norm, y_train)
    knn_train_acc = du.accuracy_score(y_train, knn.predict(X_train_norm))
    knn_test_acc = du.accuracy_score(y_test, knn.predict(X_test_norm))
    results["K-Nearest Neighbors"] = {"model": knn, "needs_norm": True,
                                       "train_acc": knn_train_acc, "test_acc": knn_test_acc}

    print("[train] Training Gradient Boosting (XGBoost-style)...")
    gb = GradientBoostingClassifier(n_estimators=60, learning_rate=0.15, max_depth=3)
    gb.fit(X_train, y_train)
    gb_train_acc = du.accuracy_score(y_train, gb.predict(X_train))
    gb_test_acc = du.accuracy_score(y_test, gb.predict(X_test))
    results["XGBoost"] = {"model": gb, "needs_norm": False,
                           "train_acc": gb_train_acc, "test_acc": gb_test_acc}

    print("\n[train] ==== Model Comparison ====")
    for name, r in results.items():
        print(f"  {name:22s} train_acc={r['train_acc']*100:5.1f}%   test_acc={r['test_acc']*100:5.1f}%")

    best_name = max(results, key=lambda n: results[n]["test_acc"])
    best = results[best_name]
    print(f"\n[train] Best model: {best_name} (test accuracy {best['test_acc']*100:.1f}%)")

    # ---- Save everything the web app needs to make predictions ----
    payload = {
        "model": best["model"],
        "model_name": best_name,
        "needs_norm": best["needs_norm"],
        "norm_mins": mins,
        "norm_maxs": maxs,
        "encoders": encoders,
        "fill_values": fill_values,
        "feature_order": du.FEATURE_ORDER,
        "all_results": {n: {"train_acc": r["train_acc"], "test_acc": r["test_acc"]} for n, r in results.items()},
    }
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"[train] Saved best model + encoders to {MODEL_PATH}")
    print(f"[train] Total pipeline time: {time.time() - t0:.1f}s\n")
    return payload


if __name__ == "__main__":
    run_pipeline(force_regenerate_data=False)
