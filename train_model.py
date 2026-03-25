"""
Fraud Detection Model Training Script
Uses csv + numpy (no pandas dependency)
Dataset: PaySim synthetic financial transactions
"""

import csv
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "PS_20174392719_1491204439457_log.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Only these types can be fraudulent in PaySim
VALID_TYPES = {"TRANSFER", "CASH_OUT"}


def load_data(path: str):
    print("Loading dataset (this may take a minute for 6M rows)...")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["type"] not in VALID_TYPES:
                continue
            rows.append([
                int(row["step"]),
                row["type"],
                float(row["amount"]),
                float(row["oldbalanceOrg"]),
                float(row["newbalanceOrig"]),
                float(row["oldbalanceDest"]),
                float(row["newbalanceDest"]),
                int(row["isFraud"]),
            ])

    print(f"Loaded {len(rows):,} TRANSFER/CASH_OUT rows")
    return rows


def build_features(rows, label_encoder=None, fit_encoder=True):
    """
    Returns X (numpy array), y (numpy array), fitted LabelEncoder
    """
    types = [r[1] for r in rows]

    if fit_encoder:
        le = LabelEncoder()
        type_enc = le.fit_transform(types)
    else:
        le = label_encoder
        type_enc = le.transform(types)

    X_list = []
    y_list = []

    for i, r in enumerate(rows):
        step, _, amount, old_orig, new_orig, old_dest, new_dest, label = r
        te = type_enc[i]

        balance_diff_orig      = old_orig - new_orig
        balance_diff_dest      = new_dest - old_dest
        amt_to_orig            = amount / old_orig if old_orig > 0 else 0.0
        orig_zero              = 1 if new_orig == 0 else 0
        dest_zero              = 1 if old_dest == 0 else 0
        err_orig               = old_orig - amount - new_orig
        err_dest               = old_dest + amount - new_dest

        X_list.append([
            step, te, amount,
            old_orig, new_orig,
            old_dest, new_dest,
            balance_diff_orig, balance_diff_dest,
            amt_to_orig, orig_zero, dest_zero,
            err_orig, err_dest
        ])
        y_list.append(label)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)
    return X, y, le


FEATURES = [
    "step", "type_enc", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "balance_diff_orig", "balance_diff_dest",
    "amount_to_orig_balance", "orig_balance_zero",
    "dest_balance_zero", "error_balance_orig", "error_balance_dest"
]


def train(X, y):
    fraud_rate = y.mean()
    print(f"Fraud rate in filtered data: {fraud_rate:.4%}")

    print("\nSplitting 80/20 stratified...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")

    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    print("\nEvaluating on test set...")
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"ROC-AUC:           {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Average Precision: {average_precision_score(y_test, y_proba):.4f}")
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))

    return model


def save_artifacts(model, le):
    joblib.dump(model, os.path.join(MODEL_DIR, "fraud_model.pkl"))
    joblib.dump(le,    os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(FEATURES, os.path.join(MODEL_DIR, "features.pkl"))
    print(f"\nArtifacts saved to ./{MODEL_DIR}/")


if __name__ == "__main__":
    rows = load_data(DATA_PATH)
    print("Building features...")
    X, y, le = build_features(rows)
    print(f"Feature matrix: {X.shape}")
    model = train(X, y)
    save_artifacts(model, le)
    print("Done.")
