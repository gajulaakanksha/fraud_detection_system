"""
Fraud Detection Model Training Script
Uses csv + numpy (no pandas dependency)
Dataset: PaySim synthetic financial transactions

S3 Config (set as environment variables on EC2):
  S3_DATA_BUCKET  =  valli-ai-poc-data    (bucket containing the CSV)
  S3_DATA_KEY     = PS_20174392719_1491204439457_log.csv
  S3_MODEL_BUCKET = valli-ai-models-224989089359-ap-south-1-an   (bucket to upload trained .pkl files)
  S3_MODEL_PREFIX = ""         (optional subfolder, e.g. "model/")

If S3_DATA_BUCKET is not set, falls back to local CSV file.
"""

import csv
import io
import numpy as np
import joblib
import os
import boto3
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import LabelEncoder

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
S3_DATA_BUCKET  = os.getenv("S3_DATA_BUCKET", "valli-ai-poc-data")
S3_DATA_KEY     = os.getenv("S3_DATA_KEY", "PS_20174392719_1491204439457_log.csv")
S3_MODEL_BUCKET = os.getenv("S3_MODEL_BUCKET", "valli-ai-models-224989089359-ap-south-1-an")
S3_MODEL_PREFIX = os.getenv("S3_MODEL_PREFIX", "")          # e.g. "" or "model/"

LOCAL_DATA_PATH = "PS_20174392719_1491204439457_log.csv"
MODEL_DIR       = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

VALID_TYPES = {"TRANSFER", "CASH_OUT"}


# ── Data loading ───────────────────────────────────────────────────────────────
def _parse_rows(reader) -> list:
    rows = []
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
    return rows


def load_data() -> list:
    if S3_DATA_BUCKET:
        print(f"Loading dataset from s3://{S3_DATA_BUCKET}/{S3_DATA_KEY} ...")
        s3 = boto3.client("s3")
        response = s3.get_object(Bucket=S3_DATA_BUCKET, Key=S3_DATA_KEY)
        body = response["Body"].read().decode("utf-8")
        rows = _parse_rows(csv.DictReader(io.StringIO(body)))
        print(f"Loaded {len(rows):,} TRANSFER/CASH_OUT rows from S3")
        return rows

    print(f"Loading dataset from local file: {LOCAL_DATA_PATH} ...")
    with open(LOCAL_DATA_PATH, newline="", encoding="utf-8") as f:
        rows = _parse_rows(csv.DictReader(f))
    print(f"Loaded {len(rows):,} TRANSFER/CASH_OUT rows from local")
    return rows


# ── Feature engineering ────────────────────────────────────────────────────────
def build_features(rows, label_encoder=None, fit_encoder=True):
    types = [r[1] for r in rows]
    if fit_encoder:
        le = LabelEncoder()
        type_enc = le.fit_transform(types)
    else:
        le = label_encoder
        type_enc = le.transform(types)

    X_list, y_list = [], []
    for i, r in enumerate(rows):
        step, _, amount, old_orig, new_orig, old_dest, new_dest, label = r
        te = type_enc[i]
        X_list.append([
            step, te, amount,
            old_orig, new_orig, old_dest, new_dest,
            old_orig - new_orig,                                        # balance_diff_orig
            new_dest - old_dest,                                        # balance_diff_dest
            amount / old_orig if old_orig > 0 else 0.0,                # amount_to_orig_balance
            1 if new_orig == 0 else 0,                                  # orig_balance_zero
            1 if old_dest == 0 else 0,                                  # dest_balance_zero
            old_orig - amount - new_orig,                               # error_balance_orig
            old_dest + amount - new_dest,                               # error_balance_dest
        ])
        y_list.append(label)

    return np.array(X_list, dtype=np.float64), np.array(y_list, dtype=np.int32), le


FEATURES = [
    "step", "type_enc", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "balance_diff_orig", "balance_diff_dest",
    "amount_to_orig_balance", "orig_balance_zero",
    "dest_balance_zero", "error_balance_orig", "error_balance_dest"
]


# ── Training ───────────────────────────────────────────────────────────────────
def train(X, y):
    print(f"Fraud rate: {y.mean():.4%}")
    print("Splitting 80/20 stratified...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"ROC-AUC:           {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Average Precision: {average_precision_score(y_test, y_proba):.4f}")
    print(confusion_matrix(y_test, y_pred))
    return model


# ── Save & upload ──────────────────────────────────────────────────────────────
def save_artifacts(model, le):
    artifacts = {
        "fraud_model.pkl":   model,
        "label_encoder.pkl": le,
        "features.pkl":      FEATURES,
    }

    # Save locally
    for filename, obj in artifacts.items():
        joblib.dump(obj, os.path.join(MODEL_DIR, filename))
        print(f"Saved locally: {MODEL_DIR}/{filename}")

    # Upload to valli-ai-models-224989089359-ap-south-1-an bucket
    s3 = boto3.client("s3")
    for filename in artifacts:
        local_path = os.path.join(MODEL_DIR, filename)
        s3_key = S3_MODEL_PREFIX + filename
        s3.upload_file(local_path, S3_MODEL_BUCKET, s3_key)
        print(f"Uploaded → s3://{S3_MODEL_BUCKET}/{s3_key}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rows = load_data()
    print("Building features...")
    X, y, le = build_features(rows)
    print(f"Feature matrix: {X.shape}")
    model = train(X, y)
    save_artifacts(model, le)
    print("Done.")
