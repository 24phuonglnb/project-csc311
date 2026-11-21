# CSC311 — Sport Classification Project

## Overview

This project aims to classify survey responses into three sports: **Soccer, Baseball, and Basketball**, using numerical, categorical, and free-text features.

The dataset contains 870 observations from 290 respondents, with each respondent providing one response per sport.

## Repository Structure

```text
Project/
├── README.md
├── MLP_MODEL.md                 # final MLP: pipeline, hyperparameters, results
├── data_explored.py
├── csc311_project_training_202609.csv
├── export_preprocessing.py      # writes mlp_preprocessing.json for pred.py
├── train_mlp.py                 # MLP search (validation) and final training
├── pred.py                      # submission: predict_all(filename)
├── processed_data/
│   ├── train_clean.csv
│   ├── val_clean.csv
│   ├── test_clean.csv
│   ├── train_scaled.csv
│   ├── val_scaled.csv
│   └── test_scaled.csv
└── mlp_outputs/
    ├── mlp_model.npz            # submitted model
    ├── mlp_preprocessing.json   # submitted preprocessing parameters
    ├── mlp_best_config.json
    ├── mlp_test_results.json
    ├── mlp_validation_results.csv
    ├── mlp_variant_summary.csv
    └── baseline_sgd/            # SGD model without PCA/k-means (88.64% test)
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

The MLP reads the `clean` files and fits its own standardization (all features)
on the rows it trains on.

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

The MLP is trained with plain mini-batch gradient descent (SGD, no momentum)
and an L2 penalty. Before the MLP, an optional feature pipeline is applied:
k-means cluster features from the TF-IDF
columns, then standardization, then optional PCA. All of it is fit on training
rows only. The search fits on the training split and selects the pipeline and
hyperparameters on the validation split (no cross-validation; the test split is
not read). Each configuration is scored by the mean validation macro-F1 of
single MLPs trained with five seeds. Final training refits the pipeline and one
MLP on train + validation, evaluates the test split, and exports
`mlp_outputs/mlp_model.npz`. See `MLP_MODEL.md` for the selected
configuration and results.

Install the additional training dependencies and run:

```bash
pip install torch scikit-learn
python3 export_preprocessing.py
python3 train_mlp.py search
python3 train_mlp.py final
```

## Submission (`pred.py`)

`pred.py` defines `predict_all(filename)`, which reads a raw survey CSV in the
same format as `csc311_project_training_202609.csv` (`id` and `label` optional)
and returns a list of predicted labels. It imports only the standard library,
NumPy, and pandas. It reapplies the `data_explored.py` preprocessing using the
training-fitted parameters in `mlp_outputs/mlp_preprocessing.json`, then runs
the MLP in `mlp_outputs/mlp_model.npz`.

Submit these three files together (about 270 KB):

- `pred.py`
- `mlp_outputs/mlp_model.npz`
- `mlp_outputs/mlp_preprocessing.json`

`pred.py` looks for the two data files next to itself first, then in
`mlp_outputs/`, so a flat MarkUs upload works.

```bash
python3 pred.py csc311_project_training_202609.csv
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
