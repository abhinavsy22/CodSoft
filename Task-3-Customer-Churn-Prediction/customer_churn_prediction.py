"""
=============================================================================
CodSoft Internship - Task 3: Customer Churn Prediction
=============================================================================
Objective:
    Build a model to predict whether a bank customer will churn (leave the
    bank) using demographic and account features.

Dataset:
    Bank Customer Churn Prediction
    Source : https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction
    File   : Churn_Modelling.csv

Target Variable:
    Exited  -> 1 = customer churned, 0 = customer retained

Models Evaluated:
    1. Logistic Regression
    2. Random Forest Classifier
    3. Gradient Boosting Classifier

Author : CodSoft Intern
=============================================================================
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend - safe on all systems
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
DATA_FILE = os.path.join(DATA_DIR, "Churn_Modelling.csv")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "output")

# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="Set2")
COLORS = sns.color_palette("Set2")


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the churn dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to Churn_Modelling.csv.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all original columns.

    Raises
    ------
    SystemExit
        Prints actionable instructions and exits if the file is not found.
    """
    if not os.path.isfile(filepath):
        print("\n" + "=" * 65)
        print("  ERROR: Dataset file not found!")
        print("=" * 65)
        print(f"  Expected path: {filepath}")
        print()
        print("  How to fix:")
        print("  1. Download the dataset from Kaggle:")
        print("     https://www.kaggle.com/datasets/shantanudhakadd/"
              "bank-customer-churn-prediction")
        print("  2. Extract the ZIP and locate 'Churn_Modelling.csv'")
        print("  3. Create the 'data' folder next to this script:")
        print(f"     {DATA_DIR}")
        print("  4. Place 'Churn_Modelling.csv' inside that folder.")
        print("=" * 65 + "\n")
        sys.exit(1)

    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# =============================================================================
# 2. PRE-PROCESSING
# =============================================================================

def preprocess_data(df: pd.DataFrame):
    """
    Clean, encode, and split the raw churn dataset.

    Steps
    -----
    * Drop irrelevant identifier columns.
    * Label-encode Gender  (Female=0, Male=1).
    * One-hot encode Geography (France / Germany / Spain).
    * Separate features (X) from target (y).
    * Stratified train/test split (80/20).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe returned by load_data().

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
    feature_names                     : list[str]
    """
    # -- Drop identifier columns that carry no predictive signal -------------
    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    df = df.drop(columns=drop_cols)

    # -- Label-encode Gender -------------------------------------------------
    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])   # Female=0, Male=1
    print("[INFO] Gender encoded  -> Female=0, Male=1")

    # -- One-hot encode Geography --------------------------------------------
    df = pd.get_dummies(df, columns=["Geography"], drop_first=False)
    geo_cols = [c for c in df.columns if c.startswith("Geography_")]
    print(f"[INFO] Geography one-hot columns: {geo_cols}")

    # -- Ensure all three geography columns exist (robustness) ---------------
    for geo in ["Geography_France", "Geography_Germany", "Geography_Spain"]:
        if geo not in df.columns:
            df[geo] = 0

    # -- Define feature order ------------------------------------------------
    feature_names = [
        "CreditScore",
        "Geography_France",
        "Geography_Germany",
        "Geography_Spain",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    target = "Exited"

    X = df[feature_names].values
    y = df[target].values

    # -- Stratified split keeps class ratio identical in both sets -----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    churn_pct = y.mean() * 100
    print(f"[INFO] Churn rate in full dataset : {churn_pct:.2f}%")
    print(f"[INFO] Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, feature_names, df


# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

def plot_eda(df: pd.DataFrame, out_dir: str) -> None:
    """
    Generate and save EDA plots.

    Plots saved
    -----------
    * churn_distribution.png  - class balance bar/pie chart
    * feature_correlation.png - heatmap of numeric feature correlations
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- 3a. Churn distribution -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Customer Churn Distribution", fontsize=15, fontweight="bold")

    counts = df["Exited"].value_counts()
    labels = ["Retained (0)", "Churned (1)"]

    axes[0].bar(labels, counts.values, color=[COLORS[0], COLORS[1]], edgecolor="black")
    axes[0].set_title("Count of Retained vs Churned Customers")
    axes[0].set_ylabel("Number of Customers")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 40, str(v), ha="center", fontweight="bold")

    axes[1].pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=[COLORS[0], COLORS[1]],
        startangle=140,
        wedgeprops={"edgecolor": "white"},
    )
    axes[1].set_title("Churn Proportion")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "churn_distribution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved -> {out_path}")

    # ---- 3b. Feature correlation heatmap ----------------------------------
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "feature_correlation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved -> {out_path}")


# =============================================================================
# 4. BUILD PIPELINES
# =============================================================================

def build_pipelines() -> dict:
    """
    Construct sklearn Pipelines for each model.

    Each pipeline contains:
    1. StandardScaler  - zero-mean, unit-variance normalisation
    2. Classifier      - with class_weight='balanced' to handle imbalance

    Returns
    -------
    dict
        Keys are model names; values are sklearn Pipeline objects.
    """
    pipelines = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                solver="lbfgs",
                C=1.0,
            )),
        ]),

        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )),
        ]),

        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
            )),
        ]),
    }
    return pipelines


# =============================================================================
# 5. EVALUATE MODEL
# =============================================================================

def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Train a pipeline and compute evaluation metrics on the held-out test set.

    Metrics reported
    ----------------
    * Classification report (precision / recall / F1 per class)
    * ROC-AUC score
    * 5-fold cross-validated ROC-AUC on training data

    Parameters
    ----------
    name     : str            Human-readable model name.
    pipeline : Pipeline       Untrained sklearn pipeline.
    X_train  : np.ndarray     Training features.
    X_test   : np.ndarray     Test features.
    y_train  : np.ndarray     Training labels.
    y_test   : np.ndarray     Test labels.

    Returns
    -------
    dict  with keys: name, pipeline, y_pred, y_prob, roc_auc, cv_auc
    """
    print(f"\n{'=' * 60}")
    print(f"  Model: {name}")
    print(f"{'=' * 60}")

    # -- Train ---------------------------------------------------------------
    pipeline.fit(X_train, y_train)

    # -- Predict -------------------------------------------------------------
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # -- Metrics -------------------------------------------------------------
    roc_auc = roc_auc_score(y_test, y_prob)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train,
                                 cv=cv, scoring="roc_auc", n_jobs=-1)

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Retained", "Churned"]))
    print(f"  ROC-AUC (test)          : {roc_auc:.4f}")
    print(f"  CV ROC-AUC (5-fold)     : {cv_scores.mean():.4f}"
          f" +/- {cv_scores.std():.4f}")

    return {
        "name":    name,
        "pipeline": pipeline,
        "y_pred":  y_pred,
        "y_prob":  y_prob,
        "roc_auc": roc_auc,
        "cv_auc":  cv_scores.mean(),
    }


# =============================================================================
# 6. ROC CURVES PLOT
# =============================================================================

def plot_roc_curves(results: list, y_test: np.ndarray, out_dir: str) -> None:
    """
    Plot ROC curves for all models on a single figure.

    Parameters
    ----------
    results  : list of dicts returned by evaluate_model()
    y_test   : true labels
    out_dir  : directory to save the figure
    """
    plt.figure(figsize=(9, 7))

    for i, res in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        plt.plot(fpr, tpr,
                 label=f"{res['name']}  (AUC={res['roc_auc']:.3f})",
                 lw=2, color=COLORS[i])

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves - Model Comparison", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)

    out_path = os.path.join(out_dir, "roc_curves.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved -> {out_path}")


# =============================================================================
# 7. CONFUSION MATRIX PLOT
# =============================================================================

def plot_confusion_matrix(
    name: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    out_dir: str,
) -> None:
    """
    Save a labelled confusion matrix heatmap for a single model.

    Parameters
    ----------
    name    : model name (used in title and filename)
    y_test  : true labels
    y_pred  : predicted labels
    out_dir : output directory
    """
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Retained", "Churned"],
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {name}", fontsize=13, fontweight="bold")

    safe_name = name.lower().replace(" ", "_")
    out_path = os.path.join(out_dir, f"confusion_matrix_{safe_name}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved -> {out_path}")


# =============================================================================
# 8. MODEL COMPARISON BAR CHART
# =============================================================================

def plot_model_comparison(results: list, out_dir: str) -> None:
    """
    Side-by-side bar chart comparing ROC-AUC scores of all models.

    Parameters
    ----------
    results : list of dicts returned by evaluate_model()
    out_dir : output directory
    """
    names    = [r["name"]    for r in results]
    test_auc = [r["roc_auc"] for r in results]
    cv_auc   = [r["cv_auc"]  for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, test_auc, width,
                   label="Test ROC-AUC",  color=COLORS[0], edgecolor="black")
    bars2 = ax.bar(x + width / 2, cv_auc,   width,
                   label="CV ROC-AUC",    color=COLORS[1], edgecolor="black")

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("ROC-AUC Score", fontsize=12)
    ax.set_title("Model Comparison - ROC-AUC", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.4)

    out_path = os.path.join(out_dir, "model_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved -> {out_path}")


# =============================================================================
# 9. FEATURE IMPORTANCE PLOT
# =============================================================================

def plot_feature_importance(
    results: list,
    feature_names: list,
    out_dir: str,
) -> None:
    """
    Plot feature importances from tree-based models (RF and GB).
    For Logistic Regression, plot |coefficient| as a proxy.

    Parameters
    ----------
    results       : list of dicts returned by evaluate_model()
    feature_names : list of feature name strings
    out_dir       : output directory
    """
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 7))
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        clf = res["pipeline"].named_steps["clf"]
        name = res["name"]

        # Extract importance values depending on model type
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            importance_label = "Gini Importance"
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
            importance_label = "|Coefficient|"
        else:
            continue

        # Sort by importance descending
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_values   = importances[indices]

        bars = ax.barh(sorted_features[::-1], sorted_values[::-1],
                       color=COLORS[2], edgecolor="black")
        ax.set_title(f"{name}\nFeature Importance ({importance_label})",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel(importance_label, fontsize=10)
        ax.grid(axis="x", alpha=0.4)

    fig.suptitle("Feature Importance Across Models",
                 fontsize=14, fontweight="bold", y=1.02)
    out_path = os.path.join(out_dir, "feature_importance.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved -> {out_path}")


# =============================================================================
# 10. DEMO PREDICTION
# =============================================================================

def predict_customer(pipeline: Pipeline, feature_names: list) -> None:
    """
    Run a demo prediction on a hand-crafted sample customer record.

    The sample represents a middle-aged German customer with a
    low balance who has multiple products - a profile often
    associated with higher churn risk.

    Parameters
    ----------
    pipeline      : trained sklearn Pipeline (best model)
    feature_names : list of feature column names used during training
    """
    # Sample customer - adjust values freely to test different scenarios
    sample = {
        "CreditScore":       600,
        "Geography_France":  0,
        "Geography_Germany": 1,   # German customer
        "Geography_Spain":   0,
        "Gender":            0,   # Female
        "Age":               42,
        "Tenure":            3,
        "Balance":           0.0,
        "NumOfProducts":     2,
        "HasCrCard":         1,
        "IsActiveMember":    0,   # Inactive member - higher risk
        "EstimatedSalary":   134543.00,
    }

    sample_df = pd.DataFrame([sample])[feature_names]
    sample_arr = sample_df.values

    pred  = pipeline.predict(sample_arr)[0]
    prob  = pipeline.predict_proba(sample_arr)[0]

    print("\n" + "=" * 60)
    print("  DEMO: Prediction for a Sample Customer")
    print("=" * 60)
    print("  Input features:")
    for feat, val in sample.items():
        print(f"    {feat:<25}: {val}")
    print(f"\n  Prediction      : {'Churned (1)' if pred == 1 else 'Retained (0)'}")
    print(f"  Prob(Retained)  : {prob[0]:.4f}")
    print(f"  Prob(Churned)   : {prob[1]:.4f}")
    print("=" * 60)


# =============================================================================
# 11. MAIN
# =============================================================================

def main() -> None:
    """
    Orchestrate the full Customer Churn Prediction pipeline:
      1. Load data
      2. Pre-process data
      3. EDA plots
      4. Build model pipelines
      5. Train and evaluate each model
      6. Save visualisations
      7. Demo prediction
    """
    print("\n" + "=" * 60)
    print("  CodSoft Task 3 - Customer Churn Prediction")
    print("=" * 60)

    # -- Create output directory -------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)

    # -- Step 1: Load data --------------------------------------------------
    raw_df = load_data(DATA_FILE)

    # -- Step 2: Pre-process ------------------------------------------------
    X_train, X_test, y_train, y_test, feature_names, processed_df = (
        preprocess_data(raw_df)
    )

    # -- Step 3: EDA plots --------------------------------------------------
    print("\n[INFO] Generating EDA plots ...")
    plot_eda(processed_df, OUT_DIR)

    # -- Step 4: Build pipelines --------------------------------------------
    pipelines = build_pipelines()

    # -- Step 5: Train + evaluate all models --------------------------------
    print("\n[INFO] Training and evaluating models ...")
    results = []
    for model_name, pipe in pipelines.items():
        res = evaluate_model(
            model_name, pipe, X_train, X_test, y_train, y_test
        )
        results.append(res)

    # -- Step 6: Visualisations ---------------------------------------------
    print("\n[INFO] Saving evaluation plots ...")

    plot_roc_curves(results, y_test, OUT_DIR)

    for res in results:
        plot_confusion_matrix(res["name"], y_test, res["y_pred"], OUT_DIR)

    plot_model_comparison(results, OUT_DIR)
    plot_feature_importance(results, feature_names, OUT_DIR)

    # -- Summary table -------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Model Performance Summary")
    print("=" * 60)
    print(f"  {'Model':<25} {'Test AUC':>10} {'CV AUC':>10}")
    print("  " + "-" * 48)
    best_res = max(results, key=lambda r: r["roc_auc"])
    for res in results:
        marker = " <-- BEST" if res["name"] == best_res["name"] else ""
        print(f"  {res['name']:<25} {res['roc_auc']:>10.4f}"
              f" {res['cv_auc']:>10.4f}{marker}")
    print("=" * 60)

    # -- Step 7: Demo prediction using best model ---------------------------
    predict_customer(best_res["pipeline"], feature_names)

    print(f"\n[DONE] All output files saved to: {OUT_DIR}")
    print("       Run complete!\n")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    main()
