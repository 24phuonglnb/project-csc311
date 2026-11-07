# Softmax Regression — CSC311 Sport Classification

Multinomial Softmax Regression implemented **from scratch with NumPy**.  
Classifies survey responses into **Baseball / Basketball / Soccer**.

---

## Directory layout

```
softmax_regression/
├── softmax_model.py        # NumPy SoftmaxRegression class
├── train_softmax.py        # Random-search hyperparameter tuning (50 configs)
├── evaluate.py             # Validation metrics and plots
├── evaluate_test.py        # Final test-set evaluation (run once)
├── predict.py              # Raw-survey → prediction inference pipeline
├── test_softmax.py         # 10 unit tests: gradients, sanity, split integrity
├── README.md
├── artifacts/              # Saved model and preprocessing parameters
│   ├── model_weights.npz              # W (234×3), b (3,), loss histories
│   ├── model_meta.json                # Hyperparameters and training summary
│   ├── feature_names.json             # Ordered feature names (234) + class order
│   └── preprocessing/
│       ├── scaler_params.npz          # StandardScaler mean and scale
│       ├── tfidf_params.json          # TF-IDF vocabulary and IDF weights
│       ├── multihot_vocab.json        # Multi-hot encoding vocabularies
│       ├── imputation_values.json     # Median imputation values
│       └── preprocessing_manifest.json
└── results/                # Generated outputs
    ├── best_config.json               # Best hyperparameter configuration
    ├── random_search_results.csv      # All 50 config results
    ├── random_search_summary.png      # F1 distribution across 50 configs
    ├── classification_report.txt      # Validation per-class metrics
    ├── confusion_matrix.png           # Validation confusion matrix
    ├── loss_curves.png                # Train + validation loss curves
    ├── test_classification_report.txt # Test per-class metrics
    ├── test_metrics.json              # Test metrics summary
    ├── test_confusion_matrix.png      # Test confusion matrix
    ├── test_predictions.csv           # Test predictions + probabilities
    └── overfitting_analysis.txt       # Train / val / test gap analysis
```

---

## Prerequisites

```bash
pip install numpy pandas matplotlib scikit-learn
```

> **Inference only** (`predict.py`) requires only `numpy` and `pandas` — no scikit-learn.

All commands below are run from the **project root** (the directory containing
`softmax_regression/` and `processed_data/`).

---

## Commands

### Run unit tests

```bash
python3 softmax_regression/test_softmax.py
```

Expected: `10/10 tests passed`

### Train — hyperparameter search (~4 seconds)

```bash
python3 softmax_regression/train_softmax.py
```

Samples 50 configurations (seed 42), trains each with mini-batch SGD and early
stopping, selects the best by validation macro-F1, and saves weights to
`softmax_regression/artifacts/`.

### Evaluate on validation set

```bash
python3 softmax_regression/evaluate.py
```

Saves loss curves, confusion matrix, and classification report to `results/`.

### Evaluate on test set (run once — final results)

```bash
python3 softmax_regression/evaluate_test.py
```

Loads the saved model without retraining and evaluates it on
`processed_data/test_scaled.csv`. Existing test results in `results/` are
already committed — re-running overwrites them.

### Predict on new raw survey data

```bash
python3 softmax_regression/predict.py --input path/to/new_responses.csv
# Optional: save predictions to file
python3 softmax_regression/predict.py --input path/to/new_responses.csv --output predictions.csv
```

The input CSV must follow the same schema as `csc311_project_training_202609.csv`.
No pre-processed features needed — `predict.py` applies the full pipeline from
raw survey responses using saved artifacts only (no scikit-learn required).

---

## Results

### Validation (model selection set)

| Metric | Value |
|---|---|
| Accuracy | **96.12%** |
| Macro-F1 | **0.9611** |
| Best epoch (0-indexed) | 2 |
| Actual stop epoch | 4 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Baseball | 0.9762 | 0.9535 | 0.9647 | 43 |
| Basketball | 0.9348 | 1.0000 | 0.9663 | 43 |
| Soccer | 0.9756 | 0.9302 | 0.9524 | 43 |

### Test (final evaluation)

| Metric | Value |
|---|---|
| Accuracy | **88.64%** |
| Macro-F1 | **0.8875** |
| Weighted-F1 | **0.8875** |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Baseball | 0.8125 | 0.8864 | 0.8478 | 44 |
| Basketball | 0.8889 | 0.9091 | 0.8989 | 44 |
| Soccer | 0.9744 | 0.8636 | 0.9157 | 44 |

---

## Best hyperparameter configuration

| Hyperparameter | Value |
|---|---|
| Learning rate | 0.2 |
| Batch size | 128 |
| Max epochs | 100 |
| L2 coefficient (λ) | 1.0 |
| Early-stopping patience | 2 |
| Random seed | 42 |

Selected by validation macro-F1 (primary) and validation cross-entropy
(tie-breaker) across 50 of 1,024 possible configurations.

---

## Implementation details

**Objective:**
```
L(W, b) = mean_cross_entropy + (λ / 2) × ‖W‖²
```
L2 regularisation is applied to **W only** (not bias `b`).

**Gradients** (analytically derived, verified against finite differences to < 1e-7):
```
dL/dW = (1/N) × X.T @ (P − Y_one_hot) + λ × W
dL/db = (1/N) × Σ(P − Y_one_hot)
```

**Training loop:**
1. Xavier-scaled weight initialisation (`scale = sqrt(2 / (D + K))`).
2. Each epoch: shuffle training data, run mini-batch SGD.
3. After each epoch: compute regularised train loss and plain val cross-entropy.
4. Early stopping: restore best weights when val CE does not improve for
   `patience` consecutive epochs.

---

## Data

- **Source CSVs:** `processed_data/train_scaled.csv`, `val_scaled.csv`, `test_scaled.csv`
  (produced by the team's `data_explored.py` preprocessing pipeline — not modified here)
- **Features:** 234 standardised features
- **Split (respondent-level, seed 42):** 609 train / 129 val / 132 test observations
- **Labels:** Baseball, Basketball, Soccer (balanced across all splits)

The test set is never used during training or model selection.

---

## Notes

- Random seed: `42` throughout (weight init, batch shuffling, hyperparameter sampling).
- No cross-validation (team decision: fixed 70/15/15 respondent-level split).
- Teammates' files (`data_explored.py`, raw CSV, processed CSVs) are not modified.
- Inference via `predict.py` requires only `numpy` and `pandas` at runtime.
