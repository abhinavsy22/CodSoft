"""
=============================================================
 TASK 1 - Movie Genre Classification (IMDB Dataset)
=============================================================
 Uses TF-IDF vectorization with three classifiers:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Support Vector Machine (Linear SVC)

 Dataset: https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb

 Data files expected (place in ./data/ folder):
   - Genre Classification Dataset IMDb/train_data.txt
   - Genre Classification Dataset IMDb/test_data.txt
   - Genre Classification Dataset IMDb/test_data_solution.txt  (optional, for test evaluation)
=============================================================
"""

import os
import re
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_FILE = os.path.join(DATA_DIR, "train_data.txt")
TEST_FILE  = os.path.join(DATA_DIR, "test_data.txt")
TEST_SOL_FILE = os.path.join(DATA_DIR, "test_data_solution.txt")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  1. DATA LOADING
# ─────────────────────────────────────────────

def load_train_data(filepath: str) -> pd.DataFrame:
    """
    Train data format (space-separated, first 3 tokens are meta):
      <id> ::: <title> ::: <genre> ::: <plot description>
    """
    records = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ::: ")
            if len(parts) < 4:
                continue
            records.append({
                "id":          parts[0].strip(),
                "title":       parts[1].strip(),
                "genre":       parts[2].strip().lower(),
                "description": parts[3].strip(),
            })
    return pd.DataFrame(records)


def load_test_data(filepath: str, has_genre: bool = False) -> pd.DataFrame:
    """
    Test data format:
      <id> ::: <title> ::: <plot description>      (no genre)
    Test solution format:
      <id> ::: <title> ::: <genre> ::: <plot description>
    """
    records = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ::: ")
            if has_genre and len(parts) >= 4:
                records.append({
                    "id":          parts[0].strip(),
                    "title":       parts[1].strip(),
                    "genre":       parts[2].strip().lower(),
                    "description": parts[3].strip(),
                })
            elif not has_genre and len(parts) >= 3:
                records.append({
                    "id":          parts[0].strip(),
                    "title":       parts[1].strip(),
                    "description": parts[2].strip(),
                })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
#  2. TEXT PREPROCESSING
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Basic text cleaning for plot descriptions."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)               # keep only letters
    text = re.sub(r"\s+", " ", text).strip()             # collapse whitespace
    return text


# ─────────────────────────────────────────────
#  3. EDA HELPERS
# ─────────────────────────────────────────────

def plot_genre_distribution(df: pd.DataFrame, title: str = "Genre Distribution"):
    genre_counts = df["genre"].value_counts()
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(x=genre_counts.index, y=genre_counts.values, palette="viridis", ax=ax)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "genre_distribution.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [Saved] {out_path}")
    return genre_counts


def plot_description_length(df: pd.DataFrame):
    df = df.copy()
    df["desc_len"] = df["description"].apply(lambda x: len(x.split()))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["desc_len"], bins=60, color="steelblue", edgecolor="white")
    axes[0].set_title("Distribution of Plot Description Lengths")
    axes[0].set_xlabel("Number of Words")
    axes[0].set_ylabel("Count")

    genre_len = df.groupby("genre")["desc_len"].mean().sort_values(ascending=False)
    sns.barplot(x=genre_len.index, y=genre_len.values, palette="mako", ax=axes[1])
    axes[1].set_title("Avg Description Length by Genre")
    axes[1].set_xlabel("Genre")
    axes[1].set_ylabel("Avg Words")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "description_lengths.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [Saved] {out_path}")


# ─────────────────────────────────────────────
#  4. BUILD PIPELINES
# ─────────────────────────────────────────────

def build_pipelines() -> dict:
    """Returns a dict of named sklearn Pipelines."""
    tfidf_params = dict(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{2,}",  # words with 2+ chars
    )

    pipelines = {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf",   MultinomialNB(alpha=0.1)),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf",   LogisticRegression(
                C=5.0,
                max_iter=1000,
                solver="saga",
                n_jobs=-1,
                random_state=42,
            )),
        ]),
        "SVM (LinearSVC)": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf",   LinearSVC(
                C=1.0,
                max_iter=2000,
                random_state=42,
            )),
        ]),
    }
    return pipelines


# ─────────────────────────────────────────────
#  5. EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(name: str, pipeline, X_test, y_test, genres):
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_w   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  F1 Macro        : {f1_mac:.4f}")
    print(f"  F1 Weighted     : {f1_w:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=genres, zero_division=0))

    return {
        "model":       name,
        "accuracy":    acc,
        "f1_macro":    f1_mac,
        "f1_weighted": f1_w,
        "y_pred":      y_pred,
    }


def plot_confusion_matrix(name: str, y_true, y_pred, genres):
    cm = confusion_matrix(y_true, y_pred, labels=genres)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        xticklabels=genres,
        yticklabels=genres,
        cmap="Blues",
        ax=ax,
        linewidths=0.3,
    )
    ax.set_title(f"Confusion Matrix (Normalised) – {name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + "_confusion.png"
    out_path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [Saved] {out_path}")


def plot_model_comparison(results: list):
    df = pd.DataFrame(results)[["model", "accuracy", "f1_macro", "f1_weighted"]]
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, df["accuracy"],    width, label="Accuracy",    color="#4C72B0")
    ax.bar(x,          df["f1_macro"],   width, label="F1 Macro",    color="#DD8452")
    ax.bar(x + width,  df["f1_weighted"],width, label="F1 Weighted", color="#55A868")

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison – Accuracy & F1 Scores", fontsize=13, fontweight="bold")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [Saved] {out_path}")


# ─────────────────────────────────────────────
#  6. PREDICT CUSTOM INPUT
# ─────────────────────────────────────────────

def predict_genre(best_pipeline, text: str) -> str:
    cleaned = clean_text(text)
    genre   = best_pipeline.predict([cleaned])[0]
    return genre


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  TASK 1 – Movie Genre Classification (IMDB)")
    print("=" * 60)

    # ── Load data ──
    print("\n[1] Loading data ...")
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}\n"
            "Please download the dataset from:\n"
            "  https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb\n"
            "and place the files inside the ./data/ folder."
        )

    train_df = load_train_data(TRAIN_FILE)
    print(f"  Training samples : {len(train_df):,}")
    print(f"  Genres found     : {train_df['genre'].nunique()}")
    print(f"  Genre list       : {sorted(train_df['genre'].unique())}")

    # ── EDA ──
    print("\n[2] Exploratory Data Analysis ...")
    genre_counts = plot_genre_distribution(train_df)
    print(f"\n  Top-5 genres:\n{genre_counts.head().to_string()}")
    plot_description_length(train_df)

    # ── Preprocessing ──
    print("\n[3] Cleaning text ...")
    train_df["clean_desc"] = train_df["description"].apply(clean_text)

    # ── Prepare X, y ──
    X_train = train_df["clean_desc"].values
    y_train = train_df["genre"].values
    genres  = sorted(train_df["genre"].unique())

    # ── Load test data ──
    has_solution = os.path.exists(TEST_SOL_FILE)
    if has_solution:
        print(f"\n[4] Loading test solution file ...")
        test_df = load_test_data(TEST_SOL_FILE, has_genre=True)
        test_df["clean_desc"] = test_df["description"].apply(clean_text)
        X_test  = test_df["clean_desc"].values
        y_test  = test_df["genre"].values
        print(f"  Test samples : {len(test_df):,}")
    elif os.path.exists(TEST_FILE):
        print(f"\n[4] Loading test data (no labels – will use CV on train) ...")
        test_df = load_test_data(TEST_FILE, has_genre=False)
        test_df["clean_desc"] = test_df["description"].apply(clean_text)
        X_test  = None
        y_test  = None
    else:
        print("\n[4] No test file found – will use 5-fold CV on train data.")
        X_test, y_test = None, None

    # ── Train & Evaluate ──
    print("\n[5] Training classifiers ...")
    pipelines = build_pipelines()
    results   = []

    for name, pipe in pipelines.items():
        print(f"\n  ▶ {name}")
        t0 = time.time()
        pipe.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"    Trained in {elapsed:.1f}s")

        if y_test is not None:
            res = evaluate_model(name, pipe, X_test, y_test, genres)
            plot_confusion_matrix(name, y_test, res["y_pred"], genres)
            results.append(res)
        else:
            # 5-fold cross-validation on train
            cv_acc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
            print(f"    CV Accuracy  : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
            results.append({
                "model":       name,
                "accuracy":    cv_acc.mean(),
                "f1_macro":    0,
                "f1_weighted": 0,
            })

    # ── Summary ──
    print("\n[6] Model Comparison Summary")
    summary_df = pd.DataFrame(results)[["model", "accuracy", "f1_macro", "f1_weighted"]]
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "model_summary.csv"), index=False)

    if y_test is not None:
        plot_model_comparison(results)

    # ── Best model demo ──
    best = max(results, key=lambda r: r["accuracy"])
    best_pipeline = pipelines[best["model"]]
    print(f"\n  Best model: {best['model']}  (accuracy={best['accuracy']:.4f})")

    # ── Quick demo predictions ──
    print("\n[7] Sample Predictions with Best Model")
    demos = [
        "A young wizard discovers his magical heritage and joins a school for witches and wizards.",
        "Two detectives hunt a serial killer who uses the seven deadly sins as his motive.",
        "A young couple falls in love aboard a doomed ocean liner and struggle to survive.",
        "A group of soldiers on a secret mission behind enemy lines during World War II.",
        "A stand-up comedian reflects on his life and failed relationships with dark humour.",
    ]
    for desc in demos:
        genre = predict_genre(best_pipeline, desc)
        print(f"  '{desc[:70]}...'")
        print(f"   → Predicted genre: [{genre.upper()}]\n")

    print("=" * 60)
    print("  Done! Output files saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
