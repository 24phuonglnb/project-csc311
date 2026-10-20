# CSC311 — Sport Classification Project

## Overview

This project aims to classify survey responses into three sports: **Soccer, Baseball, and Basketball**, using numerical, categorical, and free-text features.

The dataset contains 870 observations from 290 respondents, with each respondent providing one response per sport.

## Repository Structure

```text
Project/
├── README.md
├── data_explored.py
├── csc311_project_training_202609.csv
└── processed_data/
    ├── train_clean.csv
    ├── val_clean.csv
    ├── test_clean.csv
    ├── train_scaled.csv
    ├── val_scaled.csv
    └── test_scaled.csv
```

## Data Preprocessing

The data is split by respondent ID into training (70%), validation (15%), and internal test (15%) sets to prevent respondent-level data leakage. All preprocessing parameters are learned exclusively from the training set.

The preprocessing pipeline includes:

- **Numerical features:** Numeric parsing, range validation, median imputation, and missingness indicators.
- **Likert features:** Centered ordinal encoding from −2 to +2, with median imputation.
- **Multi-select features:** Multi-hot encoding for playing locations, seasons, and music types.
- **Free-text features:** TF-IDF vectorization (up to 50 terms per question), with missing responses represented by empty strings.
- **Feature scaling:** StandardScaler fitted on the training set only.

The resulting datasets contain **234 numerical features and one label column**.

## Processed Datasets

Each CSV contains both model features and the target column (`label`).

| Files | Description |
|---|---|
| `*_clean.csv` | Cleaned and encoded features without standardization |
| `*_scaled.csv` | Same features, standardized using training-set statistics |

The `train` and `val` splits are intended for model development and selection. The internal `test` split should be used only for final evaluation after model selection.

Suggested usage:

- **Scaled:** Softmax Regression, kNN, MLP.
- **Clean:** Decision Trees, Random Forests.

## Loading the Data

```python
import pandas as pd

train = pd.read_csv("processed_data/train_scaled.csv")
val = pd.read_csv("processed_data/val_scaled.csv")

X_train = train.drop(columns=["label"])
y_train = train["label"]

X_val = val.drop(columns=["label"])
y_val = val["label"]
```

Use the same preprocessing variant (`clean` or `scaled`) consistently across splits.

## Reproducing Preprocessing

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Run:

```bash
python3 data_explored.py
```

The script performs preprocessing, prints training-only EDA statistics, displays exploratory plots, and exports six CSV files into `processed_data/`.

## Notes

- Random seed: `42`.
- Split sizes: 609 training, 129 validation, and 132 internal test observations.
- Respondent IDs are excluded from model features.
- The original labels (`Soccer`, `Baseball`, `Basketball`) are preserved in the exported CSV files.
- Unparseable ticket-price responses are imputed rather than assigned speculative numerical values.
