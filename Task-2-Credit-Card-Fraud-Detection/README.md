# Task 2: Credit Card Fraud Detection

> **CodSoft Machine Learning Internship — Task 2**

## Overview

This project builds a supervised machine learning pipeline to detect fraudulent credit card transactions. Given a transaction's metadata (amount, merchant category, cardholder demographics, and temporal features), the model classifies it as **Legitimate** or **Fraudulent**.

Because fraud cases are extremely rare (~0.5% of transactions), special care is taken to handle **severe class imbalance** using `class_weight='balanced'` and evaluation metrics that go beyond simple accuracy (ROC-AUC, Precision-Recall AUC).

---

## Dataset

| Detail | Value |
|---|---|
| Source | [Kaggle — Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) |
| Author | Kartik Shenoy |
| Train file | `fraudTrain.csv` |
| Test file | `fraudTest.csv` |
| Total rows | ~1.8M (train) + ~550K (test) |
| Fraud rate | ~0.58% |

### Key Columns

| Column | Description |
|---|---|
| `trans_date_trans_time` | Timestamp of the transaction |
| `amt` | Transaction amount (USD) |
| `merchant` | Merchant name |
| `category` | Merchant category (e.g. `grocery_pos`, `shopping_net`) |
| `gender` | Card holder gender (M/F) |
| `dob` | Card holder date of birth |
| `city_pop` | Population of the card holder's city |
| `is_fraud` | **Target** — 1 = Fraudulent, 0 = Legitimate |

---

## Data Folder Setup

Download the dataset files and place them as follows:

```
codsoft-task2-fraud-detection/
├── credit_card_fraud_detection.py
├── requirements.txt
├── README.md
└── data/                   <-- create this folder
    ├── fraudTrain.csv      <-- place here
    └── fraudTest.csv       <-- place here
```

> If the files are missing, the script will print clear instructions and exit gracefully.

---

## Feature Engineering

| Feature | Source | Description |
|---|---|---|
| `amt` | Raw column | Transaction amount in USD |
| `category_enc` | `category` | Label-encoded merchant category |
| `gender_enc` | `gender` | Binary (F=0, M=1) |
| `city_pop` | Raw column | Population of card holder's city |
| `age` | `dob` + transaction date | Age of card holder in years |
| `hour` | `trans_date_trans_time` | Hour of the day (0–23) |
| `day_of_week` | `trans_date_trans_time` | Day of week (0=Mon … 6=Sun) |

---

## Models Used

| Model | Key Settings |
|---|---|
| **Logistic Regression** | `class_weight='balanced'`, `max_iter=1000`, `solver='lbfgs'` |
| **Decision Tree** | `class_weight='balanced'`, `max_depth=10` |
| **Random Forest** | `class_weight='balanced'`, `n_estimators=100`, `max_depth=15` |

All models are wrapped in an sklearn `Pipeline` with `StandardScaler` preprocessing.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place dataset files

```
data/fraudTrain.csv
data/fraudTest.csv
```

### 3. Run the script

```bash
python credit_card_fraud_detection.py
```

---

## Expected Output Files

All plots are saved to `./output/`:

| File | Description |
|---|---|
| `class_distribution.png` | Bar chart of Legitimate vs Fraudulent counts (train & test) |
| `confusion_matrix_logistic_regression.png` | Confusion matrix for Logistic Regression |
| `confusion_matrix_decision_tree.png` | Confusion matrix for Decision Tree |
| `confusion_matrix_random_forest.png` | Confusion matrix for Random Forest |
| `roc_curves.png` | Overlaid ROC and Precision-Recall curves for all models |
| `model_comparison.png` | Bar chart comparing ROC-AUC and Avg Precision across models |
| `feature_importance.png` | Random Forest feature importance scores |

---

## Expected Performance Metrics

Based on the Kaggle dataset, typical results are:

| Model | ROC-AUC | Avg Precision |
|---|---|---|
| Logistic Regression | ~0.90 | ~0.10 |
| Decision Tree | ~0.85 | ~0.25 |
| **Random Forest** | **~0.97** | **~0.65** |

> **Note:** Precision-Recall AUC (Average Precision) is the most meaningful metric here because the dataset is severely imbalanced. A high ROC-AUC can be misleading for rare-event detection.

---

## Class Imbalance Strategy

The dataset is highly imbalanced (~0.58% fraud). This project addresses it by:

1. **`class_weight='balanced'`** — Automatically adjusts loss weights so the minority (fraud) class has proportionally higher influence during training.
2. **Evaluation focus on PR-AUC** — Average Precision from the Precision-Recall curve is a better metric than accuracy or even ROC-AUC for imbalanced datasets.

---

## Project Structure

```
codsoft-task2-fraud-detection/
├── credit_card_fraud_detection.py   # Main ML script
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── data/
│   ├── fraudTrain.csv
│   └── fraudTest.csv
└── output/
    ├── class_distribution.png
    ├── confusion_matrix_*.png
    ├── roc_curves.png
    ├── model_comparison.png
    └── feature_importance.png
```
