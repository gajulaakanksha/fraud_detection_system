# SOP – End-to-End Workflow Documentation
# Module: Fraud Detection System (ML Backend)

**Prepared by:** Akanksha
**Module:** Machine Learning Backend / API
**Version:** 1.0.0
**Date:** March 2026

---

## 1. Overview

### Purpose of the Module

This module is the intelligence layer of the fraud detection system. Its job is to look at a financial transaction — the amount, the account balances before and after, the type of transfer — and decide in real time whether it is fraudulent or legitimate.

Think of it like an automated bank security officer. Every time a transaction is submitted, this module analyses it in milliseconds and returns a verdict: fraud or legitimate, along with a risk score and risk level (low / medium / high / critical).

### Key Functionality

- Trains a machine learning model on 6.3 million real-world-style financial transactions
- Exposes that model as a live REST API that any frontend, mobile app, or other backend service can call
- Returns a fraud probability score, a clear prediction label, and a risk level for every transaction
- Supports both single transaction checks and bulk batch checks (up to 500 at once)
- Provides health check and model metadata endpoints for monitoring and integration

---

## 2. Architecture / Flow

### High-Level System Flow

```
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 1 — OFFLINE TRAINING (run once)                          │
│                                                                  │
│  Raw CSV Dataset (6.3M rows)                                     │
│       ↓                                                          │
│  Filter to TRANSFER & CASH_OUT only (2.77M rows)                 │
│       ↓                                                          │
│  Engineer 14 features from raw columns                           │
│       ↓                                                          │
│  Split: 80% Train / 20% Test                                     │
│       ↓                                                          │
│  Train RandomForest ML Model                                     │
│       ↓                                                          │
│  Evaluate (ROC-AUC: 0.9991)                                      │
│       ↓                                                          │
│  Save model + encoder + features → /model folder                 │
└──────────────────────────────────────────────────────────────────┘
                              ↓
                     model files on disk
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 2 — ONLINE INFERENCE (runs continuously)                 │
│                                                                  │
│  FastAPI Server starts → loads model into memory                 │
│       ↓                                                          │
│  Frontend / App sends transaction data via HTTP POST             │
│       ↓                                                          │
│  API validates the input (rejects bad data automatically)        │
│       ↓                                                          │
│  Same 14 features are computed from the input                    │
│       ↓                                                          │
│  Model scores the transaction → fraud probability (0.0 to 1.0)  │
│       ↓                                                          │
│  Risk level assigned (low / medium / high / critical)            │
│       ↓                                                          │
│  JSON response returned to caller in milliseconds                │
└──────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Training Flow (train_model.py)

| Step | What Happens | Why |
|---|---|---|
| 1 | Open CSV file row by row | Dataset is 470MB — read efficiently without loading all into memory at once |
| 2 | Skip rows where type is not TRANSFER or CASH_OUT | Fraud only occurs in these two types — other types add noise |
| 3 | Parse each row into numbers | ML models only understand numbers, not text |
| 4 | Compute 7 derived features per row | Raw columns alone don't capture fraud patterns well enough |
| 5 | Convert transaction type text to number (LabelEncoder) | "TRANSFER" → 1, "CASH_OUT" → 0 |
| 6 | Split into training set (80%) and test set (20%) | Test set is held back to measure real-world performance |
| 7 | Train RandomForest with 100 decision trees | Model learns patterns that distinguish fraud from legit |
| 8 | Evaluate on test set | Verify the model actually works before deploying |
| 9 | Save model, encoder, and feature list to disk | API loads these files to make predictions |

### Step-by-Step API Request Flow (main.py)

| Step | What Happens |
|---|---|
| 1 | Client sends a POST request to `/predict` with transaction JSON |
| 2 | Pydantic automatically validates all fields (type, amount, balances) |
| 3 | If validation fails → HTTP 422 error returned immediately |
| 4 | `build_features()` computes the same 14 features used during training |
| 5 | `model.predict_proba()` returns `[probability_legit, probability_fraud]` |
| 6 | If fraud probability ≥ 0.5 → prediction = "fraud", else "legitimate" |
| 7 | Risk level assigned based on probability range |
| 8 | Response JSON returned to client |

---

## 3. Implementation Details

### Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Core programming language |
| FastAPI | 0.135.2 | REST API framework |
| Uvicorn | 0.42.0 | Web server that runs FastAPI |
| scikit-learn | 1.8.0 | Machine learning library (RandomForest) |
| NumPy | 2.4.3 | Numerical array operations |
| joblib | 1.5.3 | Saving and loading the trained model |
| Pydantic | 2.12.5 | Input validation and data schemas |

### Key Files

| File | Role |
|---|---|
| `train_model.py` | Trains the ML model. Run once before starting the API. |
| `main.py` | The live API server. Handles all incoming prediction requests. |
| `requirements.txt` | Lists all Python packages needed to run the project. |
| `model/fraud_model.pkl` | The trained model saved to disk. |
| `model/label_encoder.pkl` | Converts transaction type text to numbers. |
| `model/features.pkl` | The ordered list of 14 feature names. |

### API Endpoints

| Method | Endpoint | Purpose | Who Calls It |
|---|---|---|---|
| GET | `/health` | Check if API and model are running | Monitoring systems, DevOps |
| POST | `/predict` | Predict fraud for one transaction | Frontend app, mobile app |
| POST | `/predict/batch` | Predict fraud for up to 500 transactions | Batch processing jobs |
| GET | `/model/info` | Get model metadata and feature list | Developers, debugging |

### The 14 Features the Model Uses

The model does not use raw columns directly. It uses a combination of original fields and derived calculations:

| # | Feature Name | What It Represents | Type |
|---|---|---|---|
| 1 | step | Hour of the transaction (1–744) | Original |
| 2 | type_enc | Transaction type as a number (0 or 1) | Derived |
| 3 | amount | Money transferred | Original |
| 4 | oldbalanceOrg | Sender's balance before | Original |
| 5 | newbalanceOrig | Sender's balance after | Original |
| 6 | oldbalanceDest | Receiver's balance before | Original |
| 7 | newbalanceDest | Receiver's balance after | Original |
| 8 | balance_diff_orig | How much the sender lost | Derived |
| 9 | balance_diff_dest | How much the receiver gained | Derived |
| 10 | amount_to_orig_balance | What % of sender's balance was moved | Derived |
| 11 | orig_balance_zero | Did sender's account go to zero? (1/0) | Derived |
| 12 | dest_balance_zero | Was receiver's account empty before? (1/0) | Derived |
| 13 | error_balance_orig | Accounting error on sender side | Derived |
| 14 | error_balance_dest | Accounting error on receiver side | Derived |

The most powerful features are `error_balance_orig` and `error_balance_dest`. In a legitimate transaction, money is perfectly conserved — these values are zero. Fraudsters often manipulate balances in ways that break this rule, making these features the strongest fraud signals.

### The ML Model — RandomForest (Plain English)

A Random Forest is like asking 100 different experts to independently look at a transaction and vote on whether it's fraud. Each expert (called a "decision tree") looks at different aspects of the transaction. The final answer is the majority vote.

Key settings:
- 100 trees (100 independent experts voting)
- `class_weight="balanced"` — since only 0.3% of transactions are fraud, the model is told to treat each fraud case as 337 times more important than a legit case, so it doesn't just ignore fraud
- Runs on all CPU cores simultaneously for speed

### Risk Level Logic

```
Fraud Probability 0.00 – 0.24  →  LOW      (safe to allow)
Fraud Probability 0.25 – 0.49  →  MEDIUM   (worth monitoring)
Fraud Probability 0.50 – 0.74  →  HIGH     (flag for review)
Fraud Probability 0.75 – 1.00  →  CRITICAL (block immediately)
```

---

## 4. Integration Points

### How This Module Connects with Other Modules

```
┌─────────────────┐         POST /predict          ┌──────────────────────┐
│  Frontend / UI  │  ──────────────────────────→   │  Fraud Detection API │
│  (React / Web)  │  ←──────────────────────────   │  (This Module)       │
└─────────────────┘    JSON response with           └──────────────────────┘
                       prediction + risk level
                       
┌─────────────────┐         POST /predict/batch     ┌──────────────────────┐
│  Backend Server │  ──────────────────────────→   │  Fraud Detection API │
│  (Node / Java)  │  ←──────────────────────────   │  (This Module)       │
└─────────────────┘    bulk results + fraud count   └──────────────────────┘

┌─────────────────┐         GET /health             ┌──────────────────────┐
│  AWS / DevOps   │  ──────────────────────────→   │  Fraud Detection API │
│  Monitoring     │  ←──────────────────────────   │  (This Module)       │
└─────────────────┘    status + model_loaded flag   └──────────────────────┘
```

### What This Module Needs from Other Modules

| Dependency | What It Needs |
|---|---|
| Frontend | To send transaction data in the correct JSON format (see Section 5) |
| Database / Backend | To provide the 7 transaction fields per request |
| DevOps / AWS | To host the API server and call `/health` for uptime monitoring |

### What Other Modules Get from This Module

| Consumer | What They Receive |
|---|---|
| Frontend | `prediction`, `fraud_probability`, `risk_level`, `confidence` |
| Backend | Same JSON — can store results in database or trigger alerts |
| Monitoring | `model_loaded: true/false`, `status` string |

### Request Format (what the frontend must send)

```json
{
  "step": 1,
  "type": "TRANSFER",
  "amount": 5000.00,
  "oldbalanceOrg": 10000.00,
  "newbalanceOrig": 5000.00,
  "oldbalanceDest": 0.00,
  "newbalanceDest": 5000.00
}
```

Rules the frontend must follow:
- `type` must be exactly `"TRANSFER"` or `"CASH_OUT"` — no other values accepted
- `amount` must be greater than 0
- All balance fields must be 0 or positive — no negative numbers

---

## 5. Setup & Execution

### Prerequisites

- Python 3.11 or higher installed
- The PaySim dataset CSV file placed in the project root
- Internet connection for first-time package installation

### Step 1 — Get the Dataset

Download from Kaggle: https://www.kaggle.com/datasets/ealaxi/paysim1

Place the file in the project folder. The filename must be exactly:
```
PS_20174392719_1491204439457_log.csv
```

### Step 2 — Set Up Python Environment

```bash
# Create a virtual environment (recommended)
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Activate it (Mac/Linux)
source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

Packages installed: fastapi, uvicorn, scikit-learn, numpy, joblib, pydantic

### Step 4 — Train the Model

```bash
python train_model.py
```

Expected output:
```
Loading dataset (this may take a minute for 6M rows)...
Loaded 2,770,409 TRANSFER/CASH_OUT rows
Building features...
Feature matrix: (2770409, 14)
Fraud rate in filtered data: 0.2965%

Splitting 80/20 stratified...
Train: 2,216,327  Test: 554,082

Training RandomForestClassifier...

Classification Report:
              precision    recall  f1-score
       Legit       1.00      1.00      1.00
       Fraud       1.00      1.00      1.00

ROC-AUC:           0.9991
Average Precision: 0.9981

Artifacts saved to ./model/
Done.
```

Training takes approximately 5–10 minutes depending on machine specs.

### Step 5 — Start the API Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Expected output:
```
INFO:     Started server process
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 6 — Test the API

Open a browser and go to:
```
http://localhost:8000/docs
```

This opens the interactive Swagger UI where you can test all endpoints without writing any code.

Or test via command line:

```bash
# Health check
curl http://localhost:8000/health

# Fraud prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"step":1,"type":"TRANSFER","amount":181.0,"oldbalanceOrg":181.0,"newbalanceOrig":0.0,"oldbalanceDest":0.0,"newbalanceDest":0.0}'
```

### Project Folder Structure After Setup

```
fraud_detection/
├── main.py                               ← API server
├── train_model.py                        ← Training script
├── requirements.txt                      ← Dependencies
├── .gitignore                            ← Git ignore rules
├── README.md                             ← Quick start guide
├── DESIGN.md                             ← Technical design
├── SOP_Fraud_Detection_Module.md         ← This document
├── PS_20174392719_1491204439457_log.csv  ← Dataset (not in Git)
└── model/                                ← Created after training
    ├── fraud_model.pkl
    ├── label_encoder.pkl
    └── features.pkl
```

---

## 6. Sample Output

### GET /health

```json
{
  "status": "API is running successfully",
  "model_loaded": true,
  "model_version": "1.0.0"
}
```

### POST /predict — Fraud Transaction

Input:
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

Output:
```json
{
  "prediction": "fraud",
  "fraud_probability": 0.9998,
  "risk_level": "critical",
  "confidence": 0.9998,
  "model_version": "1.0.0"
}
```

### POST /predict — Legitimate Transaction

Input:
```json
{
  "step": 1,
  "type": "CASH_OUT",
  "amount": 500.0,
  "oldbalanceOrg": 5000.0,
  "newbalanceOrig": 4500.0,
  "oldbalanceDest": 1000.0,
  "newbalanceDest": 1500.0
}
```

Output:
```json
{
  "prediction": "legitimate",
  "fraud_probability": 0.0021,
  "risk_level": "low",
  "confidence": 0.9979,
  "model_version": "1.0.0"
}
```

### POST /predict — Validation Error (wrong type)

Input:
```json
{
  "type": "PAYMENT",
  ...
}
```

Output (HTTP 422):
```json
{
  "detail": [
    {
      "type": "literal_error",
      "msg": "Input should be 'TRANSFER' or 'CASH_OUT'",
      "input": "PAYMENT"
    }
  ]
}
```

### GET /model/info

```json
{
  "model_type": "RandomForestClassifier",
  "features": [
    "step", "type_enc", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "balance_diff_orig", "balance_diff_dest",
    "amount_to_orig_balance", "orig_balance_zero",
    "dest_balance_zero", "error_balance_orig", "error_balance_dest"
  ],
  "n_features": 14,
  "supported_transaction_types": ["TRANSFER", "CASH_OUT"],
  "version": "1.0.0"
}
```

### Model Performance Summary

| Metric | Score | What It Means |
|---|---|---|
| ROC-AUC | 0.9991 | Near-perfect ability to rank fraud above legit |
| Average Precision | 0.9981 | Near-perfect precision across all thresholds |
| Fraud Recall | 99.7% | Catches 1,638 out of 1,643 fraud cases |
| False Negatives | 5 | Only 5 fraud cases missed in 554,082 test transactions |
| False Positives | 0 | Zero legitimate transactions wrongly flagged |

---

## 7. Challenges & Notes

### Challenges Faced

| Challenge | How It Was Solved |
|---|---|
| Dataset is 470MB — too large for GitHub | Added to `.gitignore`, linked to Kaggle download in README |
| pandas not installed in the environment | Rewrote data loading using Python's built-in `csv` module + NumPy |
| Severe class imbalance (0.3% fraud rate) | Used `class_weight="balanced"` in RandomForest |
| git push rejected due to large file already committed | Used `git rm --cached` + `git commit --amend` + force push to rewrite history |

### Known Limitations

| Limitation | Impact | Suggested Fix |
|---|---|---|
| Model is static — trained once, never updates | May miss new fraud patterns over time | Set up periodic retraining pipeline |
| No API authentication | Any caller can access the API | Add API key or OAuth2 token validation |
| No request logging | Cannot audit which transactions were checked | Add structured logging with transaction IDs |
| Dataset is synthetic (PaySim) | Model may behave differently on real bank data | Validate and retrain on real transaction data |
| Batch limit is 500 | Large batch jobs need multiple calls | Increase limit or add async job queue |
| Prediction threshold fixed at 0.5 | May not suit all risk appetites | Make threshold configurable per environment |

### Notes for Other Teams

- The API accepts only `TRANSFER` and `CASH_OUT` transaction types. The frontend must not send `PAYMENT`, `DEBIT`, or `CASH_IN` — these will be rejected with a validation error.
- The model must be trained before the API is started. If `model/` folder is missing, the API starts but returns HTTP 503 on all prediction endpoints.
- The `/health` endpoint is safe to call frequently — it does no ML computation and returns instantly.
- Interactive API documentation is always available at `http://<host>:8000/docs` — useful for frontend developers to test requests without writing code.
- The `model/` folder is excluded from Git. Each deployment environment must run `python train_model.py` once to generate the model files locally.
