# Movie Genre Classification – IMDB Dataset

## Task Overview
**Task 1** from the ML Internship: Build a machine learning model that predicts a movie's genre based on its plot summary.

## Project Structure

```
movie-genre-classification/
├── movie_genre_classification.py   ← Main ML pipeline
├── requirements.txt
├── data/                           ← Place Kaggle dataset files here
│   ├── train_data.txt
│   ├── test_data.txt
│   └── test_data_solution.txt
└── output/                         ← Auto-generated charts & results
    ├── genre_distribution.png
    ├── description_lengths.png
    ├── model_comparison.png
    ├── naive_bayes_confusion.png
    ├── logistic_regression_confusion.png
    ├── svm_linearsvc_confusion.png
    └── model_summary.csv
```

## Setup

### 1. Download the Dataset
Go to: https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb

Download and extract the zip. Place the files inside `./data/`:
- `train_data.txt`
- `test_data.txt`
- `test_data_solution.txt`

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Model

```bash
python movie_genre_classification.py
```

## Dataset Format

The dataset uses ` ::: ` as a separator:

```
<id> ::: <title> ::: <genre> ::: <plot description>
```

**Example:**
```
1 ::: Oscar et la dame rose (2009) ::: drama ::: Listening in to a conversation between his doctor and parents...
```

## Approach

### Feature Extraction
**TF-IDF** (Term Frequency–Inverse Document Frequency) with:
- Up to **50,000 features** (unigrams + bigrams)
- Sublinear TF scaling
- Min document frequency = 2

### Classifiers

| Model | Notes |
|---|---|
| **Naive Bayes** | Fast, strong baseline for text |
| **Logistic Regression** | Probabilistic, handles multi-class well |
| **SVM (LinearSVC)** | Often best for high-dimensional text |

### Preprocessing
- Lowercase text
- Remove non-alphabetic characters
- Collapse whitespace

## Output Samples

The script generates:
- 📊 **Genre distribution** bar chart
- 📏 **Description length** histograms per genre
- 🔥 **Confusion matrices** for each model
- 📈 **Model comparison** bar chart
- 💾 **CSV summary** with accuracy, F1-macro, F1-weighted

## Expected Performance (approx.)

| Model | Accuracy | F1 Macro |
|---|---|---|
| Naive Bayes | ~0.52 | ~0.50 |
| Logistic Regression | ~0.63 | ~0.61 |
| SVM (LinearSVC) | ~0.65 | ~0.63 |

> Results may vary slightly based on dataset version.
