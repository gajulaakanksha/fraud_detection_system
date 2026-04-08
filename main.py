"""
Fraud Detection API
FastAPI production service with real ML model inference

S3 Config (set as environment variables on EC2):
  S3_MODEL_BUCKET = valli-ai-models-224989089359-ap-south-1-an   (bucket containing the .pkl files)
  S3_MODEL_PREFIX = ""         (optional subfolder, e.g. "model/")

Falls back to local ./model/ folder if S3_MODEL_BUCKET is not set.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import numpy as np
import joblib
import io
import os
import boto3
from typing import Literal

# ── Config ─────────────────────────────────────────────────────────────────────
S3_MODEL_BUCKET = os.getenv("S3_MODEL_BUCKET", "valli-ai-models-224989089359-ap-south-1-an")
S3_MODEL_PREFIX = os.getenv("S3_MODEL_PREFIX", "")
MODEL_DIR       = "model"


# ── Model loading ──────────────────────────────────────────────────────────────
def _load_pkl_s3(s3, filename: str):
    key = S3_MODEL_PREFIX + filename
    print(f"Downloading s3://{S3_MODEL_BUCKET}/{key}")
    response = s3.get_object(Bucket=S3_MODEL_BUCKET, Key=key)
    return joblib.load(io.BytesIO(response["Body"].read()))


def load_artifacts():
    if S3_MODEL_BUCKET:
        try:
            s3 = boto3.client("s3")
            model    = _load_pkl_s3(s3, "fraud_model.pkl")
            features = _load_pkl_s3(s3, "features.pkl")
            encoder  = _load_pkl_s3(s3, "label_encoder.pkl")
            print("Model loaded from S3.")
            return model, features, encoder
        except Exception as e:
            print(f"S3 load failed: {e} — trying local fallback")

    # Local fallback
    paths = [
        os.path.join(MODEL_DIR, "fraud_model.pkl"),
        os.path.join(MODEL_DIR, "features.pkl"),
        os.path.join(MODEL_DIR, "label_encoder.pkl"),
    ]
    if not all(os.path.exists(p) for p in paths):
        print("Model artifacts not found.")
        return None, None, None

    model    = joblib.load(paths[0])
    features = joblib.load(paths[1])
    encoder  = joblib.load(paths[2])
    print("Model loaded from local disk.")
    return model, features, encoder


model, FEATURES, label_encoder = load_artifacts()


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time financial transaction fraud detection using ML",
    version="1.0.0"
)

# This MUST be defined before your routes to prevent CORS errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    step: int = Field(..., description="Hour of simulation (1–744)", example=1)
    type: Literal["TRANSFER", "CASH_OUT"] = Field(..., description="Transaction type")
    amount: float = Field(..., gt=0, description="Transaction amount", example=181.0)
    oldbalanceOrg: float = Field(..., ge=0, description="Sender balance before", example=181.0)
    newbalanceOrig: float = Field(..., ge=0, description="Sender balance after", example=0.0)
    oldbalanceDest: float = Field(..., ge=0, description="Receiver balance before", example=0.0)
    newbalanceDest: float = Field(..., ge=0, description="Receiver balance after", example=0.0)

    @validator("type")
    def validate_type(cls, v):
        if v not in ("TRANSFER", "CASH_OUT"):
            raise ValueError("type must be TRANSFER or CASH_OUT")
        return v


class PredictionResponse(BaseModel):
    prediction: Literal["fraud", "legitimate"]
    fraud_probability: float
    risk_level: Literal["low", "medium", "high", "critical"]
    confidence: float
    model_version: str = "1.0.0"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str = "1.0.0"


class BatchTransactionRequest(BaseModel):
    transactions: list[TransactionRequest]


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]
    total: int
    fraud_count: int


# ── Feature engineering ────────────────────────────────────────────────────────
def build_features(tx: TransactionRequest) -> np.ndarray:
    type_enc = int(label_encoder.transform([tx.type])[0])
    row = [
        tx.step, type_enc, tx.amount,
        tx.oldbalanceOrg, tx.newbalanceOrig,
        tx.oldbalanceDest, tx.newbalanceDest,
        tx.oldbalanceOrg - tx.newbalanceOrig,
        tx.newbalanceDest - tx.oldbalanceDest,
        tx.amount / tx.oldbalanceOrg if tx.oldbalanceOrg > 0 else 0.0,
        int(tx.newbalanceOrig == 0),
        int(tx.oldbalanceDest == 0),
        tx.oldbalanceOrg - tx.amount - tx.newbalanceOrig,
        tx.oldbalanceDest + tx.amount - tx.newbalanceDest,
    ]
    return np.array(row).reshape(1, -1)


def risk_label(prob: float) -> str:
    if prob < 0.25:   return "low"
    elif prob < 0.50: return "medium"
    elif prob < 0.75: return "high"
    return "critical"


def make_prediction(tx: TransactionRequest) -> PredictionResponse:
    X     = build_features(tx)
    proba = float(model.predict_proba(X)[0][1])
    pred  = "fraud" if proba >= 0.5 else "legitimate"
    conf  = proba if pred == "fraud" else 1 - proba
    return PredictionResponse(
        prediction=pred,
        fraud_probability=round(proba, 4),
        risk_level=risk_label(proba),
        confidence=round(conf, 4)
    )


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse(status="API is running successfully", model_loaded=model is not None)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(transaction: TransactionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return make_prediction(transaction)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchTransactionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if len(request.transactions) > 500:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 500.")
    results     = [make_prediction(tx) for tx in request.transactions]
    fraud_count = sum(1 for r in results if r.prediction == "fraud")
    return BatchPredictionResponse(results=results, total=len(results), fraud_count=fraud_count)


@app.get("/model/info", tags=["System"])
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "model_type": type(model).__name__,
        "features": FEATURES,
        "n_features": len(FEATURES),
        "supported_transaction_types": ["TRANSFER", "CASH_OUT"],
        "version": "1.0.0"
    }
