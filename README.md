# Fraud Detection System

A production-ready ML-powered REST API for real-time financial transaction fraud detection, trained on the PaySim dataset (~6.3M transactions).

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```
This reads the CSV, engineers features, trains a RandomForest, and saves artifacts to `./model/`.

### 3. Start the API
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open interactive docs
```
http://localhost:8000/docs
```

---

## Project Structure

```
fraud_detection/
├── main.py                          # FastAPI application
├── train_model.py                   # Model training script
├── requirements.txt                 # Python dependencies
├── PS_20174392719_1491204439457_log.csv  # PaySim dataset
└── model/                           # Generated after training
    ├── fraud_model.pkl              # Trained RandomForest model
    ├── label_encoder.pkl            # Transaction type encoder
    └── features.pkl                 # Feature name list
```

---

## API Endpoints

### GET `/health`
Check if the API and model are running.

**Response**
```json
{
  "status": "API is running successfully",
  "model_loaded": true,
  "model_version": "1.0.0"
}
```

---

### POST `/predict`
Predict fraud for a single transaction.

**Request body**
```json
{
  "step": 1,
  "type": "TRANSFER",
  "amount": 181.0,
  "oldbalanceOrg": 181.0,
  "newbalanceOrig": 0.0,
  "oldbalanceDest": 0.0,
  "newbalanceDest": 0.0
}
```

**Response**
```json
{
  "prediction": "fraud",
  "fraud_probability": 0.9998,
  "risk_level": "critical",
  "confidence": 0.9998,
  "model_version": "1.0.0"
}
```

| Field | Description |
|---|---|
| `prediction` | `fraud` or `legitimate` |
| `fraud_probability` | Model's confidence score (0.0 – 1.0) |
| `risk_level` | `low` / `medium` / `high` / `critical` |
| `confidence` | How confident the model is in its prediction |

---

### POST `/predict/batch`
Score up to 500 transactions in one call.

**Request body**
```json
{
  "transactions": [
    { "step": 1, "type": "CASH_OUT", "amount": 5000.0, ... },
    { "step": 2, "type": "TRANSFER", "amount": 200.0, ... }
  ]
}
```

**Response**
```json
{
  "results": [ ... ],
  "total": 2,
  "fraud_count": 1
}
```

---

### GET `/model/info`
Returns model metadata and feature list.

**Response**
```json
{
  "model_type": "RandomForestClassifier",
  "features": ["step", "type_enc", "amount", "..."],
  "n_features": 14,
  "supported_transaction_types": ["TRANSFER", "CASH_OUT"],
  "version": "1.0.0"
}
```

---

## Input Field Reference

| Field | Type | Constraint | Description |
|---|---|---|---|
| `step` | int | required | Hour of simulation (1–744) |
| `type` | string | `TRANSFER` or `CASH_OUT` | Transaction type |
| `amount` | float | > 0 | Transaction amount |
| `oldbalanceOrg` | float | >= 0 | Sender balance before transaction |
| `newbalanceOrig` | float | >= 0 | Sender balance after transaction |
| `oldbalanceDest` | float | >= 0 | Receiver balance before transaction |
| `newbalanceDest` | float | >= 0 | Receiver balance after transaction |

> Only `TRANSFER` and `CASH_OUT` are supported — fraud never occurs in other transaction types in this dataset.

---

## Risk Level Thresholds

| Risk Level | Fraud Probability Range |
|---|---|
| low | 0.00 – 0.24 |
| medium | 0.25 – 0.49 |
| high | 0.50 – 0.74 |
| critical | 0.75 – 1.00 |

---

## Model Performance

Evaluated on 554,082 held-out test transactions:

| Metric | Score |
|---|---|
| ROC-AUC | 0.9991 |
| Average Precision | 0.9981 |
| Precision (Fraud) | 1.00 |
| Recall (Fraud) | 1.00 |
| False Negatives | 5 out of 1,643 fraud cases |

---

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `scikit-learn` | ML model (RandomForest) |
| `numpy` | Numerical computation |
| `joblib` | Model serialization |
| `pydantic` | Request/response validation |
