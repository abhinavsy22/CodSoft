# Task 3: Customer Churn Prediction

> **CodSoft Machine Learning Internship — Task 3**

---

## Overview

Customer churn occurs when a customer stops doing business with a company. For banks, predicting which customers are likely to leave is critical to implementing proactive retention strategies.

This project builds a **binary classification** system that predicts whether a bank customer will churn (`Exited = 1`) or be retained (`Exited = 0`) using demographic, financial, and behavioural features.

Three machine learning models are trained and compared:
- **Logistic Regression** — interpretable linear baseline
- **Random Forest** — ensemble of decision trees (bagging)
- **Gradient Boosting** — sequential boosting ensemble

Class imbalance (~20% churn) is handled via `class_weight='balanced'`.

---

## Dataset

| Property | Details |
|---|---|
| **Name** | Bank Customer Churn Prediction |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction) |
| **File** | `Churn_Modelling.csv` |
| **Rows** | 10 000 customer records |
| **Target** | `Exited` (0 = Retained, 1 = Churned) |

### Column Reference

| Column | Type | Description |
|---|---|---|
| RowNumber | int | Row index (dropped) |
| CustomerId | int | Unique customer ID (dropped) |
| Surname | str | Customer surname (dropped) |
| CreditScore | int | Credit score |
| Geography | str | Country: France / Germany / Spain |
| Gender | str | Male / Female |
| Age | int | Customer age |
| Tenure | int | Years as a bank customer |
| Balance | float | Account balance |
| NumOfProducts | int | Number of bank products held |
| HasCrCard | int | Has credit card? (1/0) |
| IsActiveMember | int | Active account holder? (1/0) |
| EstimatedSalary | float | Estimated annual salary |
| **Exited** | **int** | **Target — 1 = Churned** |

---

## Project Structure

```
codsoft-task3-churn-prediction/
│
├── customer_churn_prediction.py   # Main ML script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── data/                          # <-- Place dataset here
│   └── Churn_Modelling.csv
│
└── output/                        # Generated automatically
    ├── churn_distribution.png
    ├── feature_correlation.png
    ├── model_comparison.png
    ├── roc_curves.png
    ├── confusion_matrix_logistic_regression.png
    ├── confusion_matrix_random_forest.png
    ├── confusion_matrix_gradient_boosting.png
    └── feature_importance.png
```

---

## Data Folder Setup

1. Visit the Kaggle dataset page:  
   👉 https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction

2. Download and extract the ZIP archive.

3. Create a `data/` folder in the project directory:
   ```
   codsoft-task3-churn-prediction/data/
   ```

4. Copy `Churn_Modelling.csv` into that folder.

If the file is missing, the script will print clear instructions and exit gracefully.

---

## Feature Engineering

| Step | Detail |
|---|---|
| **Drop** | `RowNumber`, `CustomerId`, `Surname` — carry no predictive value |
| **Label Encode** | `Gender`: Female → 0, Male → 1 |
| **One-Hot Encode** | `Geography`: creates `Geography_France`, `Geography_Germany`, `Geography_Spain` |
| **Scale** | All numeric features z-score normalised with `StandardScaler` inside each pipeline |
| **Imbalance** | `class_weight='balanced'` automatically up-weights minority (churn) class |

---

## Models Used

| Model | Library Class | Key Hyperparameters |
|---|---|---|
| **Logistic Regression** | `LogisticRegression` | `C=1.0`, `solver=lbfgs`, `max_iter=1000` |
| **Random Forest** | `RandomForestClassifier` | `n_estimators=200`, `max_depth=10` |
| **Gradient Boosting** | `GradientBoostingClassifier` | `n_estimators=200`, `learning_rate=0.1`, `max_depth=5` |

All models use `class_weight='balanced'` (where supported) and are wrapped in `sklearn.pipeline.Pipeline` with `StandardScaler`.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add the dataset

Follow the [Data Folder Setup](#data-folder-setup) instructions above.

### 3. Run the script

```bash
python customer_churn_prediction.py
```

The script will:
- Load and pre-process the data
- Generate EDA plots
- Train all three models
- Print evaluation metrics to the console
- Save all plots to `./output/`
- Run a demo prediction on a sample customer

---

## Expected Output Files

| File | Description |
|---|---|
| `churn_distribution.png` | Bar and pie chart of class balance |
| `feature_correlation.png` | Heatmap of numeric feature correlations |
| `roc_curves.png` | Overlaid ROC curves for all models |
| `model_comparison.png` | Side-by-side AUC bar chart |
| `confusion_matrix_logistic_regression.png` | Confusion matrix — LR |
| `confusion_matrix_random_forest.png` | Confusion matrix — RF |
| `confusion_matrix_gradient_boosting.png` | Confusion matrix — GB |
| `feature_importance.png` | Top features per model |

---

## Expected Performance Metrics

| Model | Test ROC-AUC | CV ROC-AUC (5-fold) |
|---|---|---|
| Logistic Regression | ~0.77 | ~0.77 |
| Random Forest | ~0.86 | ~0.86 |
| **Gradient Boosting** | **~0.87** | **~0.87** |

> Actual values may vary slightly due to random state and dataset version. Gradient Boosting typically achieves the best balance of precision and recall on this dataset.

---

## Demo Prediction

At the end of the run, the script uses the **best-performing model** to predict churn probability for a sample customer profile:

- **Country**: Germany  
- **Age**: 42  
- **Balance**: 0 (zero balance — high churn signal)  
- **Active Member**: No  
- **Products**: 2  

This customer is statistically likely to churn, demonstrating the model's ability to flag at-risk customers.

---

## License

This project was created as part of the **CodSoft Machine Learning Internship** and is intended for educational purposes.
