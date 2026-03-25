# Design Document — Fraud Detection System

## 1. Overview

This system detects fraudulent financial transactions in real time using a machine learning model exposed via a REST API. It is designed for production use — the model is trained offline, serialized to disk, and loaded into memory at API startup for low-latency inference.

---

## 2. Dataset

**Source:** PaySim — a synthetic mobile money transaction simulator based on real financial logs.

| Property | Value |
|---|---|
| Total rows | ~6.3 million |
| Columns | 11 |
| Fraud rate (overall) | ~0.13% |
| Fraud rate (after filtering) | ~0.30% |
| Time span | 30 days (744 hours) |

### Raw Columns

| Column | Type | Description |
|---|---|---|
| step | int | Hour of simulation |
| type | string | PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN |
| amount | float | Transaction amount |
| nameOrig | string | Sender account ID |
| oldbalanceOrg | float | Sender balance before |
| newbalanceOrig | float | Sender balance after |
| nameDest | string | Receiver account ID |
| oldbalanceDest | float | Receiver balance before |
| newbalanceDest | float | Receiver balance after |
| isFraud | int | Ground truth label (1 = fraud) |
| isFlaggedFraud | int | Rule-based flag (not used) |

### Key Observation
Fraud exclusively occurs in `TRANSFER` and `CASH_OUT` transaction types. All other types (`PAYMENT`, `DEBIT`, `CASH_IN`) have zero fraud cases. Filtering to these two types reduces the dataset to 2.77M rows while retaining 100% of fraud signal.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   OFFLINE (Training)                │
│                                                     │
│   CSV Dataset → load_data() → build_features()      │
│       → RandomForestClassifier.fit()                │
│       → joblib.dump() → model/, encoder, features   │
└─────────────────────────────────────────────────────┘
                          │
                    model artifacts
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                  ONLINE (Inference)                 │
│                                                     │
│   FastAPI Server                                    │
│   ├── startup: load_artifacts() → memory            │
│   ├── POST /predict                                 │
│   │     → validate input (Pydantic)                 │
│   │     → build_features()                          │
│   │     → model.predict_proba()                     │
│   │     → return prediction + risk level            │
│   ├── POST /predict/batch (up to 500)               │
│   ├── GET  /health                                  │
│   └── GET  /model/info                              │
└─────────────────────────────────────────────────────┘
```

---

## 4. Feature Engineering

Raw transaction fields alone are insufficient. We derive 7 additional features that capture fraud-specific patterns:

| Feature | Formula | Rationale |
|---|---|---|
| `type_enc` | LabelEncoder(type) | Converts string to integer for ML |
| `balance_diff_orig` | oldbalanceOrg − newbalanceOrig | How much the sender lost |
| `balance_diff_dest` | newbalanceDest − oldbalanceDest | How much the receiver gained |
| `amount_to_orig_balance` | amount / oldbalanceOrg | What fraction of sender's balance was moved |
| `orig_balance_zero` | 1 if newbalanceOrig == 0 | Sender account drained to zero |
| `dest_balance_zero` | 1 if oldbalanceDest == 0 | Receiver account was empty (mule account signal) |
| `error_balance_orig` | oldbalanceOrg − amount − newbalanceOrig | Accounting error on sender side |
| `error_balance_dest` | oldbalanceDest + amount − newbalanceDest | Accounting error on receiver side |

### Why the error balance features matter

In a legitimate transaction, money is conserved:
```
sender_after  = sender_before  - amount  →  error_orig = 0
receiver_after = receiver_before + amount →  error_dest = 0
```

Fraudulent transactions frequently violate this conservation — the error terms become non-zero, making them the strongest discriminative features in the model.

### Final Feature Vector (14 features)

```
[step, type_enc, amount, oldbalanceOrg, newbalanceOrig,
 oldbalanceDest, newbalanceDest, balance_diff_orig,
 balance_diff_dest, amount_to_orig_balance, orig_balance_zero,
 dest_balance_zero, error_balance_orig, error_balance_dest]
```

---

## 5. Model

### Algorithm: RandomForestClassifier

A Random Forest builds many decision trees on random subsets of data and features, then aggregates their votes. It was chosen because:

- Handles severe class imbalance via `class_weight="balanced"`
- No feature scaling required (tree-based)
- Robust to outliers (large transaction amounts)
- Outputs calibrated probabilities via `predict_proba()`
- Parallelizable across CPU cores (`n_jobs=-1`)
- Interpretable feature importances

### Hyperparameters

| Parameter | Value | Reason |
|---|---|---|
| `n_estimators` | 100 | Enough trees for stable predictions |
| `max_depth` | 20 | Deep enough to capture complex patterns |
| `min_samples_leaf` | 10 | Prevents overfitting on rare fraud cases |
| `class_weight` | balanced | Compensates for 0.3% fraud rate |
| `n_jobs` | -1 | Uses all available CPU cores |
| `random_state` | 42 | Reproducible results |

### Class Imbalance Strategy

The dataset has ~0.3% fraud rate — 337 legit transactions for every 1 fraud. Without correction, a naive model achieves 99.7% accuracy by predicting "legit" for everything, which is useless.

`class_weight="balanced"` automatically computes per-class weights:
```
weight_fraud = total_samples / (2 × fraud_count)
weight_legit = total_samples / (2 × legit_count)
```
This makes the model penalize missing a fraud case much more heavily than a false alarm.

### Train / Test Split

| Split | Size | Rows |
|---|---|---|
| Train | 80% | 2,216,327 |
| Test | 20% | 554,082 |

`stratify=y` ensures both splits maintain the same ~0.3% fraud ratio.

---

## 6. Evaluation

### Results on Test Set

| Metric | Score |
|---|---|
| Accuracy | 1.00 |
| ROC-AUC | 0.9991 |
| Average Precision | 0.9981 |
| Precision (Fraud class) | 1.00 |
| Recall (Fraud class) | 1.00 |
| F1-Score (Fraud class) | 1.00 |

### Confusion Matrix

```
                Predicted Legit   Predicted Fraud
Actual Legit       552,439              0
Actual Fraud             5          1,638
```

5 false negatives (missed frauds) out of 1,643 total fraud cases = 99.7% recall.

### Why ROC-AUC over Accuracy

Accuracy is misleading on imbalanced data. A model that always predicts "legit" gets 99.7% accuracy but catches zero fraud. ROC-AUC measures the model's ability to rank fraud above legit across all decision thresholds — a much more honest metric.

---

## 7. API Design

### Framework: FastAPI

FastAPI was chosen for:
- Automatic request validation via Pydantic models
- Auto-generated OpenAPI/Swagger docs at `/docs`
- Async-capable for high throughput
- Type hints throughout for maintainability

### Request Validation

All inputs are validated by Pydantic before reaching business logic:
- `type` must be exactly `"TRANSFER"` or `"CASH_OUT"` (Literal type)
- `amount` must be > 0
- All balance fields must be >= 0
- Missing or wrong-type fields return HTTP 422 with a clear error message

### Model Loading Strategy

The model is loaded once at server startup into module-level variables:
```python
model, FEATURES, label_encoder = load_artifacts()
```
This means zero disk I/O per request — inference is purely in-memory. If model files are missing, the server still starts but returns HTTP 503 on prediction endpoints.

### Prediction Flow

```
POST /predict
    │
    ├─ Pydantic validates JSON body
    ├─ build_features() → 14-element numpy array (shape: 1×14)
    ├─ model.predict_proba() → [prob_legit, prob_fraud]
    ├─ threshold at 0.5 → "fraud" or "legitimate"
    ├─ risk_label() → "low" / "medium" / "high" / "critical"
    └─ return PredictionResponse
```

### Risk Level Thresholds

| Level | Probability Range | Suggested Action |
|---|---|---|
| low | 0.00 – 0.24 | Allow |
| medium | 0.25 – 0.49 | Monitor |
| high | 0.50 – 0.74 | Review |
| critical | 0.75 – 1.00 | Block |

---

## 8. Serialization

Three artifacts are saved with `joblib`:

| File | Contents | Used by |
|---|---|---|
| `model/fraud_model.pkl` | Trained RandomForestClassifier | Inference |
| `model/label_encoder.pkl` | Fitted LabelEncoder for type field | Feature engineering |
| `model/features.pkl` | Ordered list of 14 feature names | Metadata endpoint |

`joblib` is preferred over `pickle` for sklearn objects because it handles large numpy arrays more efficiently via memory-mapped files.

---

## 9. Limitations & Future Improvements

| Limitation | Improvement |
|---|---|
| Model is static — doesn't learn from new data | Add online learning or periodic retraining pipeline |
| No authentication on API endpoints | Add API key or OAuth2 middleware |
| Single-process inference | Add async workers or model serving (TorchServe / BentoML) |
| No request logging | Add structured logging with transaction IDs |
| Threshold fixed at 0.5 | Make threshold configurable per deployment environment |
| No drift detection | Monitor feature distributions over time to detect data drift |
| PaySim is synthetic | Validate on real transaction data before production deployment |
