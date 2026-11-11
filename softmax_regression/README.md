# Softmax Regression — CSC311 Sport Classification

Multinomial Softmax Regression implemented **from scratch with NumPy**.  
Classifies survey responses into **Baseball / Basketball / Soccer**.

---

## Directory layout

```
softmax_regression/
├── softmax_model.py        # NumPy SoftmaxRegression class (fit, fit_fixed_epochs, save/load)
├── train_softmax.py        # Phase 1: random-search hyperparameter tuning (50 configs)
├── retrain_final.py        # Phase 2: final retraining on Train+Val; runs test evaluation
├── evaluate.py             # Validation metrics for the tuning checkpoint (artifacts/)
├── evaluate_test.py        # Test-set metrics for the final model (artifacts_final/)
├── predict.py              # Raw-survey → prediction inference pipeline
├── test_softmax.py         # 10 unit tests: gradients, sanity, split integrity
├── README.md
│
├── artifacts/              # Tuning checkpoint — trained on Train split only (preserved)
│   ├── model_weights.npz              # W (234×3), b (3,), loss histories
│   ├── model_meta.json                # Hyperparameters and training summary
│   ├── feature_names.json             # Ordered feature names (234) + class order
│   └── preprocessing/
│       ├── scaler_params.npz          # StandardScaler mean and scale (train-fitted)
│       ├── tfidf_params.json          # TF-IDF vocabulary and IDF weights (train-fitted)
│       ├── multihot_vocab.json        # Multi-hot encoding vocabularies (train-fitted)
│       ├── imputation_values.json     # Median imputation values (train-fitted)
│       └── preprocessing_manifest.json
│
├── artifacts_final/        # Final inference model — retrained on Train+Val (use this)
│   ├── model_weights.npz
│   ├── model_meta.json
│   ├── feature_names.json
│   └── preprocessing/      # Same preprocessing artifacts (copied from artifacts/)
│
└── results/                # Generated outputs
    ├── best_config.json               # Best hyperparameter configuration
    ├── random_search_results.csv      # All 50 config results
    ├── random_search_summary.png      # F1 distribution across 50 configs (not tracked)
    ├── classification_report.txt      # Validation per-class metrics (tuning checkpoint)
    ├── confusion_matrix.png           # Validation confusion matrix (not tracked)
    ├── loss_curves.png                # Train + validation loss curves (not tracked)
    ├── final_test_classification_report.txt  # Final model test per-class metrics
    ├── final_test_metrics.json               # Final model test metrics summary
    ├── final_test_confusion_matrix.png       # Final model test confusion matrix (not tracked)
    └── final_test_predictions.csv            # Final model test predictions (not tracked)
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

## Training workflow

The training procedure follows two phases, as specified in the project proposal.

### Phase 1 — Hyperparameter tuning (train split only)

Random search over 50 configurations. Each configuration is trained on the
609-sample training split with mini-batch SGD and **validation cross-entropy
early stopping** (patience from search space). The best configuration is
selected by **validation macro-F1** (tie-broken by validation cross-entropy).
Weights of the best configuration are saved to `artifacts/`.

### Phase 2 — Final retraining (Train + Val combined)

After selecting the best hyperparameters, the final model is trained **from
scratch** on the combined 738-sample Train+Val dataset for exactly
**3 epochs** (= `best_epoch_ + 1`, the number of epochs at which the tuning
run achieved its best validation checkpoint).  No validation split is used and
no early stopping is applied.  The final model is saved to `artifacts_final/`.
Preprocessing artifacts (scaler, TF-IDF, multi-hot, imputation) are **not
refitted** — they remain trained on the original training split.

The test set is never used during either tuning or retraining.

---

## Commands

### Run unit tests

```bash
python3 softmax_regression/test_softmax.py
```

Expected: `10/10 tests passed`

### Phase 1 — Hyperparameter search (~4 seconds)

```bash
python3 softmax_regression/train_softmax.py
```

Samples 50 configurations (seed 42), trains each with mini-batch SGD and early
stopping, selects the best by validation macro-F1, and saves the tuning
checkpoint to `softmax_regression/artifacts/`.

### Inspect tuning checkpoint — validation metrics

```bash
python3 softmax_regression/evaluate.py
```

Loads `artifacts/` and reports validation metrics for the tuning checkpoint.
Saves loss curves, confusion matrix, and classification report to `results/`.

### Phase 2 — Final retraining and test evaluation

```bash
python3 softmax_regression/retrain_final.py
```

Loads the best config from `results/best_config.json`, combines Train+Val (738
observations), trains from scratch for 3 fixed epochs, saves the final model to
`artifacts_final/`, and runs the **one-time test evaluation** saving results to
`results/final_test_*.`.

### Re-run test evaluation (final model only)

```bash
python3 softmax_regression/evaluate_test.py
```

Loads `artifacts_final/` and re-evaluates on `test_scaled.csv`. Overwrites
`results/final_test_*.` files.

### Predict on new raw survey data

```bash
python3 softmax_regression/predict.py --input path/to/new_responses.csv
# Optional: save predictions to file
python3 softmax_regression/predict.py --input path/to/new_responses.csv --output predictions.csv
```

Loads the **final model** from `artifacts_final/`. The input CSV must follow
the same schema as `csc311_project_training_202609.csv`. No pre-processed
features needed — `predict.py` applies the full pipeline from raw survey
responses using saved artifacts only (no scikit-learn required at inference
time).

---

## Results

### Hyperparameter selection — validation metrics
*(Tuning checkpoint trained on 609-sample Train split; used for model selection only)*

| Metric | Value |
|---|---|
| Accuracy | **96.12%** |
| Macro-F1 | **0.9611** |
| Best epoch (0-indexed) | 2 (epoch 3) |
| Early-stop epoch | 4 (epoch 5) |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Baseball | 0.9762 | 0.9535 | 0.9647 | 43 |
| Basketball | 0.9348 | 1.0000 | 0.9663 | 43 |
| Soccer | 0.9756 | 0.9302 | 0.9524 | 43 |

### Final model — test results
*(Retrained on 738-sample Train+Val; evaluated once on held-out test set)*

| Metric | Value |
|---|---|
| Training samples | 738 (Train + Val) |
| Training epochs | 3 (fixed) |
| Accuracy | **87.88%** |
| Macro-F1 | **0.8796** |
| Weighted-F1 | **0.8796** |
| Cross-entropy | **0.5640** |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Baseball | 0.8298 | 0.8864 | 0.8571 | 44 |
| Basketball | 0.8511 | 0.9091 | 0.8791 | 44 |
| Soccer | 0.9737 | 0.8409 | 0.9024 | 44 |

---

## Best hyperparameter configuration

| Hyperparameter | Value |
|---|---|
| Learning rate | 0.2 |
| Batch size | 128 |
| Max epochs (search space) | 100 |
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

**Tuning training loop (`fit`):**
1. Xavier-scaled weight initialisation (`scale = sqrt(2 / (D + K))`).
2. Each epoch: shuffle training data, run mini-batch SGD.
3. After each epoch: compute regularised train loss and plain val cross-entropy.
4. Early stopping: restore best weights when val CE does not improve for
   `patience` consecutive epochs.

**Final retraining loop (`fit_fixed_epochs`):**
1. Same Xavier-scaled initialisation from scratch (same seed).
2. Train on combined Train+Val data for exactly `n_epochs` passes.
3. No validation split, no early stopping, no weight restoration.
4. `best_epoch_` is set to `n_epochs − 1` (the last epoch's weights are kept).

---

## Data

- **Source CSVs:** `processed_data/train_scaled.csv`, `val_scaled.csv`, `test_scaled.csv`
  (produced by the team's `data_explored.py` preprocessing pipeline — not modified here)
- **Features:** 234 standardised features
- **Split (respondent-level, seed 42):** 609 train / 129 val / 132 test observations
- **Labels:** Baseball, Basketball, Soccer (balanced across all splits)

Preprocessing parameters (scaler mean/scale, TF-IDF vocabulary and IDF weights,
multi-hot vocabularies, imputation medians) are **fitted on the training split
only** and saved in `artifacts/preprocessing/`.  They are reused unchanged
during final retraining and inference.

---

## Notes

- Random seed: `42` throughout (weight init, batch shuffling, hyperparameter sampling).
- No cross-validation (team decision: fixed 70/15/15 respondent-level split).
- Teammates' files (`data_explored.py`, raw CSV, processed CSVs) are not modified.
- Inference via `predict.py` requires only `numpy` and `pandas` at runtime.
- `artifacts_final/` is the authoritative inference checkpoint; `artifacts/` is
  preserved for reproducibility of the tuning results.
