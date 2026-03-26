"""
VALLI API
FastAPI production service with real ML model inference
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import numpy as np
import joblib
import os
from typing import Literal

# ── Model loading ──────────────────────────────────────────────────────────────
MODEL_DIR = "model"

def load_artifacts():
    model_path    = os.path.join(MODEL_DIR, "fraud_model.pkl")
    features_path = os.path.join(MODEL_DIR, "features.pkl")
    encoder_path  = os.path.join(MODEL_DIR, "label_encoder.pkl")

    if not all(os.path.exists(p) for p in [model_path, features_path, encoder_path]):
        return None, None, None

    model    = joblib.load(model_path)
    features = joblib.load(features_path)
    encoder  = joblib.load(encoder_path)
    return model, features, encoder

model, FEATURES, label_encoder = load_artifacts()

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="VALLI API",
    description="Real-time financial transaction fraud detection using ML",
    version="1.0.0"
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

# ── Feature engineering (mirrors train_model.py) ───────────────────────────────
def build_features(tx: TransactionRequest) -> np.ndarray:
    type_enc = int(label_encoder.transform([tx.type])[0])

    balance_diff_orig       = tx.oldbalanceOrg - tx.newbalanceOrig
    balance_diff_dest       = tx.newbalanceDest - tx.oldbalanceDest
    amount_to_orig_balance  = tx.amount / tx.oldbalanceOrg if tx.oldbalanceOrg > 0 else 0.0
    orig_balance_zero       = int(tx.newbalanceOrig == 0)
    dest_balance_zero       = int(tx.oldbalanceDest == 0)
    error_balance_orig      = tx.oldbalanceOrg - tx.amount - tx.newbalanceOrig
    error_balance_dest      = tx.oldbalanceDest + tx.amount - tx.newbalanceDest

    row = [
        tx.step, type_enc, tx.amount,
        tx.oldbalanceOrg, tx.newbalanceOrig,
        tx.oldbalanceDest, tx.newbalanceDest,
        balance_diff_orig, balance_diff_dest,
        amount_to_orig_balance, orig_balance_zero,
        dest_balance_zero, error_balance_orig, error_balance_dest
    ]
    return np.array(row).reshape(1, -1)


def risk_label(prob: float) -> str:
    if prob < 0.25:
        return "low"
    elif prob < 0.50:
        return "medium"
    elif prob < 0.75:
        return "high"
    return "critical"


def make_prediction(tx: TransactionRequest) -> PredictionResponse:
    X = build_features(tx)
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
    """Check API and model status."""
    return HealthResponse(
        status="API is running successfully",
        model_loaded=model is not None
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(transaction: TransactionRequest):
    """
    Predict whether a single transaction is fraudulent.

    - Only TRANSFER and CASH_OUT types are supported (fraud only occurs in these).
    - Returns fraud probability, prediction label, and risk level.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train_model.py first."
        )
    return make_prediction(transaction)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchTransactionRequest):
    """
    Predict fraud for a batch of transactions (max 500).
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train_model.py first."
        )
    if len(request.transactions) > 500:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 500.")

    results = [make_prediction(tx) for tx in request.transactions]
    fraud_count = sum(1 for r in results if r.prediction == "fraud")

    return BatchPredictionResponse(
        results=results,
        total=len(results),
        fraud_count=fraud_count
    )


@app.get("/model/info", tags=["System"])
def model_info():
    """Return model metadata and feature list."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "model_type": type(model).__name__,
        "features": FEATURES,
        "n_features": len(FEATURES),
        "supported_transaction_types": ["TRANSFER", "CASH_OUT"],
        "version": "1.0.0"
    }
