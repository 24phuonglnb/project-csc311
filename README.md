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

- **Scaled:** Softmax Regression, kNN.
- **Clean:** Decision Trees, Random Forests, MLP.

The MLP applies its own training-fitted scaling to the five continuous/ordinal
features while leaving binary and TF-IDF features at their natural scales.

For this implementation run, earlier model variants were also evaluated on the
same test split before the final preprocessing/model choice. The saved final
test score is therefore exploratory, not a pristine one-time estimate; use a
new untouched respondent-level holdout for an unbiased final claim.

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

## Training and NumPy-Only Inference

The MLP search uses the existing training split for fitting and the held-out
validation split for model and epoch selection. It does not perform
cross-validation or access the test split during search. After selection, final
training combines train and validation, evaluates the test split once, and
exports `mlp_outputs/mlp_model.npz`. Validation scores evaluate a five-seed
probability-averaged MLP ensemble to reduce dependence on a single initialization.
Only the five continuous/ordinal survey features are standardized; binary,
multi-hot, and TF-IDF features retain their natural scales.

Install the additional training dependencies and run:

```bash
pip install torch scikit-learn
python3 train_mlp.py search
python3 train_mlp.py final
```

`mlp_inference.py` loads the exported model and predicts from the
`*_clean.csv` feature representation using only NumPy and pandas at inference
time. It accepts a CSV with the same feature columns (the `label` column is
optional):

```bash
python3 mlp_inference.py processed_data/test_clean.csv --output predictions.csv
```

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
