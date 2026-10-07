"""
==============================================================================
CodSoft Internship - Task 2: Credit Card Fraud Detection
==============================================================================
Objective:
    Build a machine learning model to detect fraudulent credit card
    transactions using supervised classification techniques.

Dataset:
    Kaggle - Credit Card Transactions Fraud Detection Dataset
    URL: https://www.kaggle.com/datasets/kartik2112/fraud-detection
    Files Required:
        - data/fraudTrain.csv
        - data/fraudTest.csv

Approach:
    1. Load and merge train/test CSV files
    2. Feature engineering (temporal, demographic, categorical features)
    3. Handle severe class imbalance via class_weight='balanced'
    4. Train Logistic Regression, Decision Tree, and Random Forest
    5. Evaluate with classification_report, confusion matrix, ROC-AUC,
       and precision-recall curves
    6. Save all plots to ./output/

Author  : CodSoft Intern
Task    : 2 - Credit Card Fraud Detection
==============================================================================
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend (safe for Windows)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
TRAIN_FILE = os.path.join(DATA_DIR, "fraudTrain.csv")
TEST_FILE  = os.path.join(DATA_DIR, "fraudTest.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Features used for model training
FEATURE_COLS = ["amt", "category_enc", "gender_enc",
                "city_pop", "age", "hour", "day_of_week"]
TARGET_COL   = "is_fraud"

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted")
PALETTE = {"Legitimate": "#4C72B0", "Fraudulent": "#DD8452"}


# ===========================================================================
# 1. DATA LOADING
# ===========================================================================

def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the training and test CSV files from disk.

    Parameters
    ----------
    train_path : str  Path to fraudTrain.csv
    test_path  : str  Path to fraudTest.csv

    Returns
    -------
    (train_df, test_df) as pandas DataFrames
    """
    # --- existence check ---------------------------------------------------
    missing = [p for p in (train_path, test_path) if not os.path.isfile(p)]
    if missing:
        print("\n" + "=" * 70)
        print("ERROR: Dataset file(s) not found:")
        for p in missing:
            print(f"  {p}")
        print()
        print("HOW TO FIX:")
        print("  1. Visit https://www.kaggle.com/datasets/kartik2112/fraud-detection")
        print("  2. Download fraudTrain.csv and fraudTest.csv")
        print("  3. Place them inside the 'data/' folder next to this script:")
        print(f"     {DATA_DIR}")
        print("  4. Re-run this script.")
        print("=" * 70 + "\n")
        sys.exit(1)

    print("[INFO] Loading training data  ...")
    train_df = pd.read_csv(train_path, index_col=0)
    print(f"       fraudTrain.csv  -> {train_df.shape[0]:,} rows, {train_df.shape[1]} cols")

    print("[INFO] Loading test data      ...")
    test_df = pd.read_csv(test_path, index_col=0)
    print(f"       fraudTest.csv   -> {test_df.shape[0]:,} rows, {test_df.shape[1]} cols")

    return train_df, test_df


# ===========================================================================
# 2. FEATURE ENGINEERING
# ===========================================================================

def engineer_features(df: pd.DataFrame, label_encoders: dict | None = None,
                       fit: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Create model-ready features from raw transaction columns.

    New columns added
    -----------------
    hour         : Hour of transaction (0-23)
    day_of_week  : Day of week (0=Monday .. 6=Sunday)
    age          : Age of card holder in years (derived from dob)
    category_enc : Label-encoded merchant category
    gender_enc   : Binary-encoded gender (F=0, M=1)

    Parameters
    ----------
    df             : Raw DataFrame
    label_encoders : Existing LabelEncoder dict (pass when fit=False)
    fit            : If True, fit new encoders; otherwise transform only

    Returns
    -------
    (engineered_df, label_encoders)
    """
    df = df.copy()

    # --- Temporal features -------------------------------------------------
    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"], format="mixed", dayfirst=False
    )
    df["hour"]        = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek

    # --- Age (years) --------------------------------------------------------
    df["dob"] = pd.to_datetime(df["dob"], format="mixed", dayfirst=False)
    reference_date = df["trans_date_trans_time"].max()
    df["age"] = (reference_date - df["dob"]).dt.days // 365

    # --- Label encode 'category' -------------------------------------------
    if label_encoders is None:
        label_encoders = {}

    if fit:
        le_cat = LabelEncoder()
        df["category_enc"] = le_cat.fit_transform(df["category"].astype(str))
        label_encoders["category"] = le_cat
    else:
        le_cat = label_encoders["category"]
        # Handle unseen labels gracefully
        known = set(le_cat.classes_)
        df["category"] = df["category"].apply(
            lambda x: x if x in known else le_cat.classes_[0]
        )
        df["category_enc"] = le_cat.transform(df["category"].astype(str))

    # --- Binary encode gender (F -> 0, M -> 1) -----------------------------
    df["gender_enc"] = (df["gender"].str.upper() == "M").astype(int)

    return df, label_encoders


# ===========================================================================
# 3. CLASS DISTRIBUTION PLOT
# ===========================================================================

def plot_class_distribution(y_train: pd.Series, y_test: pd.Series,
                             output_dir: str) -> None:
    """
    Bar chart showing fraudulent vs legitimate transaction counts
    for both train and test splits.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Distribution: Legitimate vs Fraudulent", fontsize=14, fontweight="bold")

    for ax, (y, title) in zip(axes, [(y_train, "Train Set"), (y_test, "Test Set")]):
        counts = y.value_counts().sort_index()
        labels = ["Legitimate", "Fraudulent"]
        colors = [PALETTE["Legitimate"], PALETTE["Fraudulent"]]
        bars   = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.2)

        for bar, count in zip(bars, counts.values):
            pct = count / counts.sum() * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.max() * 0.01,
                f"{count:,}\n({pct:.2f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Transaction Count")
        ax.set_ylim(0, counts.max() * 1.18)
        ax.tick_params(axis="x", labelsize=11)

    plt.tight_layout()
    path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {path}")


# ===========================================================================
# 4. BUILD PIPELINES
# ===========================================================================

def build_pipelines() -> dict:
    """
    Create sklearn Pipeline objects for each model.

    Each pipeline scales features with StandardScaler then applies
    the classifier with class_weight='balanced' to handle imbalance.

    Returns
    -------
    dict  {model_name: Pipeline}
    """
    pipelines = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    DecisionTreeClassifier(
                class_weight="balanced",
                max_depth=10,
                random_state=42,
            )),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                class_weight="balanced",
                n_estimators=100,
                max_depth=15,
                n_jobs=-1,
                random_state=42,
            )),
        ]),
    }
    return pipelines


# ===========================================================================
# 5. MODEL EVALUATION
# ===========================================================================

def evaluate_model(name: str, pipeline: Pipeline,
                   X_train: pd.DataFrame, y_train: pd.Series,
                   X_test:  pd.DataFrame, y_test:  pd.Series,
                   output_dir: str) -> dict:
    """
    Train a pipeline, print evaluation metrics, and save confusion matrix plot.

    Parameters
    ----------
    name      : Human-readable model name
    pipeline  : Untrained sklearn Pipeline
    X_train, y_train : Training features / labels
    X_test,  y_test  : Test features / labels
    output_dir : Directory for plots

    Returns
    -------
    dict with keys: name, roc_auc, avg_precision, fpr, tpr, precision, recall
    """
    print(f"\n{'=' * 60}")
    print(f"  Model: {name}")
    print(f"{'=' * 60}")

    # --- Training ----------------------------------------------------------
    print(f"  [INFO] Training {name} ...")
    pipeline.fit(X_train, y_train)

    # --- Predictions -------------------------------------------------------
    y_pred    = pipeline.predict(X_test)
    y_prob    = pipeline.predict_proba(X_test)[:, 1]

    # --- Classification Report --------------------------------------------
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Legitimate", "Fraudulent"]))

    # --- ROC-AUC -----------------------------------------------------------
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"  ROC-AUC Score     : {roc_auc:.4f}")

    # --- Average Precision (PR-AUC) ----------------------------------------
    avg_prec = average_precision_score(y_test, y_prob)
    print(f"  Avg Precision (PR): {avg_prec:.4f}")

    # --- Confusion Matrix plot ---------------------------------------------
    plot_confusion_matrix(name, y_test, y_pred, output_dir)

    # --- ROC / PR curve data -----------------------------------------------
    fpr, tpr, _   = roc_curve(y_test, y_prob)
    prec, rec, _  = precision_recall_curve(y_test, y_prob)

    return {
        "name":          name,
        "pipeline":      pipeline,
        "roc_auc":       roc_auc,
        "avg_precision": avg_prec,
        "fpr":           fpr,
        "tpr":           tpr,
        "precision":     prec,
        "recall":        rec,
    }


# ===========================================================================
# 6. CONFUSION MATRIX PLOT
# ===========================================================================

def plot_confusion_matrix(name: str, y_true: pd.Series,
                           y_pred: np.ndarray, output_dir: str) -> None:
    """
    Save a color-annotated confusion matrix heatmap for the given model.
    """
    cm     = confusion_matrix(y_true, y_pred)
    labels = ["Legitimate", "Fraudulent"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="white", ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix - {name}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    safe_name = name.lower().replace(" ", "_")
    path = os.path.join(output_dir, f"confusion_matrix_{safe_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Saved: {path}")


# ===========================================================================
# 7. ROC CURVES (all models on one chart)
# ===========================================================================

def plot_roc_curves(results: list[dict], output_dir: str) -> None:
    """
    Overlay ROC curves for all models on a single figure.
    """
    colors = ["#4C72B0", "#55A868", "#DD8452"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Model Comparison: ROC and Precision-Recall Curves",
                 fontsize=13, fontweight="bold")

    # --- ROC curve ---------------------------------------------------------
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    for res, color in zip(results, colors):
        ax.plot(res["fpr"], res["tpr"], color=color, lw=2,
                label=f"{res['name']} (AUC={res['roc_auc']:.3f})")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])

    # --- Precision-Recall curve --------------------------------------------
    ax = axes[1]
    for res, color in zip(results, colors):
        ax.plot(res["recall"], res["precision"], color=color, lw=2,
                label=f"{res['name']} (AP={res['avg_precision']:.3f})")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])

    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {path}")


# ===========================================================================
# 8. MODEL COMPARISON BAR CHART
# ===========================================================================

def plot_model_comparison(results: list[dict], output_dir: str) -> None:
    """
    Side-by-side bar chart comparing ROC-AUC and Average Precision
    across all trained models.
    """
    names     = [r["name"] for r in results]
    roc_aucs  = [r["roc_auc"] for r in results]
    avg_precs = [r["avg_precision"] for r in results]

    x     = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, roc_aucs,  width, label="ROC-AUC",
                   color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width / 2, avg_precs, width, label="Avg Precision (PR-AUC)",
                   color="#DD8452", edgecolor="white")

    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.005,
                f"{height:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0.0, 1.08)
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {path}")


# ===========================================================================
# 9. FEATURE IMPORTANCE (Random Forest only)
# ===========================================================================

def plot_feature_importance(rf_pipeline: Pipeline,
                             feature_names: list[str],
                             output_dir: str) -> None:
    """
    Horizontal bar chart of Random Forest feature importances.
    """
    rf_clf      = rf_pipeline.named_steps["clf"]
    importances = rf_clf.feature_importances_
    indices     = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color="#4C72B0", edgecolor="white",
    )
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("Random Forest - Feature Importances", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {path}")


# ===========================================================================
# 10. DEMO PREDICTION
# ===========================================================================

def demo_prediction(best_pipeline: Pipeline,
                    label_encoders: dict,
                    best_name: str) -> None:
    """
    Run a quick demo prediction on two hand-crafted transactions
    (one likely legitimate, one likely fraudulent).
    """
    print(f"\n{'=' * 60}")
    print("  DEMO PREDICTION  (using best model: {})".format(best_name))
    print(f"{'=' * 60}")

    # Build synthetic samples directly in feature space
    # Columns: amt, category_enc, gender_enc, city_pop, age, hour, day_of_week
    le_cat        = label_encoders["category"]
    shopping_enc  = le_cat.transform(["shopping_net"])[0]  \
                    if "shopping_net" in le_cat.classes_ else 0
    misc_enc      = le_cat.transform(["misc_net"])[0]      \
                    if "misc_net"      in le_cat.classes_ else 1

    samples = pd.DataFrame([
        # Likely legitimate: small grocery purchase, daytime, mid-aged cardholder
        {
            "amt":          12.50,
            "category_enc": shopping_enc,
            "gender_enc":   1,
            "city_pop":     45000,
            "age":          38,
            "hour":         14,
            "day_of_week":  2,
        },
        # Likely fraudulent: very large amount, late night, unusual category
        {
            "amt":          4200.00,
            "category_enc": misc_enc,
            "gender_enc":   0,
            "city_pop":     820,
            "age":          22,
            "hour":         3,
            "day_of_week":  6,
        },
    ])

    preds = best_pipeline.predict(samples)
    probs = best_pipeline.predict_proba(samples)[:, 1]

    for i, (pred, prob) in enumerate(zip(preds, probs), start=1):
        label = "FRAUDULENT" if pred == 1 else "LEGITIMATE"
        print(f"\n  Transaction {i}:")
        print(f"    Amount         : ${samples['amt'].iloc[i-1]:.2f}")
        print(f"    Hour of day    : {samples['hour'].iloc[i-1]}:00")
        print(f"    Prediction     : {label}")
        print(f"    Fraud Prob.    : {prob:.4f} ({prob*100:.2f}%)")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    """
    Orchestrate the full Credit Card Fraud Detection pipeline:
      load -> feature engineer -> plot distribution ->
      train models -> evaluate -> generate all plots -> demo predict
    """
    print("\n" + "=" * 70)
    print("  CodSoft Task 2: Credit Card Fraud Detection")
    print("=" * 70)

    # --- Output directory --------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Plots will be saved to: {OUTPUT_DIR}")

    # -----------------------------------------------------------------------
    # Step 1: Load data
    # -----------------------------------------------------------------------
    train_raw, test_raw = load_data(TRAIN_FILE, TEST_FILE)

    # -----------------------------------------------------------------------
    # Step 2: Feature engineering
    # -----------------------------------------------------------------------
    print("\n[INFO] Engineering features (train) ...")
    train_df, label_encoders = engineer_features(train_raw, fit=True)

    print("[INFO] Engineering features (test)  ...")
    test_df, _ = engineer_features(test_raw, label_encoders=label_encoders, fit=False)

    # Split features / target
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df[TARGET_COL]

    print(f"\n[INFO] Train size : {X_train.shape[0]:,} samples  "
          f"| Fraud rate: {y_train.mean()*100:.2f}%")
    print(f"[INFO] Test size  : {X_test.shape[0]:,} samples  "
          f"| Fraud rate: {y_test.mean()*100:.2f}%")

    # -----------------------------------------------------------------------
    # Step 3: Class distribution plot
    # -----------------------------------------------------------------------
    print("\n[INFO] Plotting class distribution ...")
    plot_class_distribution(y_train, y_test, OUTPUT_DIR)

    # -----------------------------------------------------------------------
    # Step 4: Build & evaluate models
    # -----------------------------------------------------------------------
    pipelines = build_pipelines()
    results   = []

    for model_name, pipeline in pipelines.items():
        res = evaluate_model(
            model_name, pipeline,
            X_train, y_train,
            X_test,  y_test,
            OUTPUT_DIR,
        )
        results.append(res)

    # -----------------------------------------------------------------------
    # Step 5: Aggregate plots
    # -----------------------------------------------------------------------
    print("\n[INFO] Generating aggregate plots ...")
    plot_roc_curves(results, OUTPUT_DIR)
    plot_model_comparison(results, OUTPUT_DIR)

    # Feature importance for Random Forest
    rf_result   = next(r for r in results if r["name"] == "Random Forest")
    plot_feature_importance(rf_result["pipeline"], FEATURE_COLS, OUTPUT_DIR)

    # -----------------------------------------------------------------------
    # Step 6: Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SUMMARY - Test Set Performance")
    print("=" * 60)
    print(f"  {'Model':<25} {'ROC-AUC':>10}  {'Avg Precision':>14}")
    print(f"  {'-'*25} {'-'*10}  {'-'*14}")
    for r in sorted(results, key=lambda x: x["roc_auc"], reverse=True):
        print(f"  {r['name']:<25} {r['roc_auc']:>10.4f}  {r['avg_precision']:>14.4f}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Step 7: Demo prediction with best model
    # -----------------------------------------------------------------------
    best = max(results, key=lambda x: x["roc_auc"])
    demo_prediction(best["pipeline"], label_encoders, best["name"])

    print("\n[DONE] All tasks completed successfully.")
    print(f"       Output files are in: {OUTPUT_DIR}\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
