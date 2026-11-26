# CSC311 — Sport Classification from Survey Responses

## Project overview

Three-class classification of student survey responses into **Soccer**, **Baseball**, or **Basketball**. Each respondent answered questions about playing location, season, music, food, colour, song association, feelings, and two numerical fields (ease rating, ticket price). The models are compared on a held-out internal test set.

## Dataset

- **Source:** `csc311_project_training_202609.csv`
- **Size:** 870 observations, 290 unique respondents (3 observations per respondent — one per sport)
- **Features:** 12 survey question columns (free text, multi-select, Likert, and numerical); `id` and `label` are excluded from model inputs
- **Labels:** `Soccer`, `Baseball`, `Basketball` (balanced across respondents)

## Respondent-level train / validation / test split

The split is performed at the **respondent level** (seed 42) to prevent data leakage across rows from the same respondent:

| Split | Respondents | Observations |
|---|---|---|
| Train | 203 | 609 |
| Validation | 43 | 129 |
| Test | 44 | 132 |

## Preprocessing

All preprocessing parameters are fitted on the **training split only** and applied unchanged to validation and test.

| Feature type | Method |
|---|---|
| Numerical (`ease`, `price`) | String-parsed, range-validated, median-imputed, missingness indicator added |
| Likert (3 columns) | Ordinal encoding −2 to +2, median-imputed |
| Multi-select (location, season, music) | Multi-hot encoding with training-vocabulary |
| Free text (song, food, colour, feeling) | TF-IDF (up to 50 terms per column, training IDF weights) |
| Feature scaling | StandardScaler fitted on training rows only |

**Final (Train+Val) refit:** For the final submitted models, preprocessing is refitted on the combined 738-row Train+Val set, and models are retrained from scratch on the same data.

## Models compared

| Model | Accuracy | Macro-F1 | Notes |
|---|---|---|---|
| Decision Tree | **81.82%** | **0.8174** | scikit-learn; max_depth=5, min_samples_leaf=5 |
| Softmax Regression | **88.64%** | **0.8872** | NumPy from scratch; cross-entropy = 0.5624 |
| MLP | **89.39%** | **0.8937** | NumPy/PyTorch; 2 hidden layers (width 16, tanh) |

All results are on the held-out 132-observation test set. The test set was used **once** for each model, after model selection on the validation set.

## Hyperparameter selection

- **Decision Tree:** Grid search over `max_depth` × `min_samples_leaf` on validation macro-F1.
- **Softmax Regression:** Random search over 50 configurations (learning rate, batch size, L2, patience); selected by validation macro-F1.
- **MLP:** 20 pipeline variants (PCA × k-means) × 64 MLP configurations (random search); each scored by mean validation macro-F1 over 5 seeds.

## Final preprocessing refit protocol

After hyperparameter selection, all three models were retrained on the **combined Train+Val set (738 observations)**. For Softmax and MLP, preprocessing was also refitted on Train+Val before final training. No test data was used at any stage of model selection or training.

## Final test results (verified)

| Model | Test Accuracy | Test Macro-F1 |
|---|---|---|
| Decision Tree | 81.82% (108/132) | 0.8174 |
| Softmax Regression | 88.64% (117/132) | 0.8872 |
| **MLP** | **89.39% (118/132)** | **0.8937** |

Softmax cross-entropy on test: **0.5624**

## Running the prediction pipeline

`pred.py` is the submission file. It is entirely self-contained — MLP weights and preprocessing parameters are embedded as base64. No external files are needed.

```python
from pred import predict_all

predictions = predict_all("path/to/survey_responses.csv")
# Returns a list of strings: 'Soccer', 'Baseball', or 'Basketball'
```

The input CSV must have the same column schema as `csc311_project_training_202609.csv`. The `label` column is optional and ignored if present. The `id` column is also optional.

```bash
python3 pred.py csc311_project_training_202609.csv
```

**Prediction dependencies:** Python standard library + NumPy + pandas only. No scikit-learn, no PyTorch required at prediction time. 60 predictions run in under 0.1 seconds.

## Reproducing the data preprocessing

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
python3 data_explored.py
```

Outputs `processed_data/train_clean.csv`, `val_clean.csv`, `test_clean.csv`, `train_scaled.csv`, `val_scaled.csv`, `test_scaled.csv`.

## Reproducing the MLP

```bash
pip install torch scikit-learn numpy pandas
python3 export_preprocessing.py        # exports MLP preprocessing params
python3 mlp/train_mlp.py search        # validation search (~4 min)
python3 mlp/train_mlp.py final         # refit on Train+Val, one test evaluation
python3 mlp/embed_model.py             # embeds weights into pred.py
```

## Reproducing the Softmax Regression

```bash
pip install numpy pandas matplotlib scikit-learn
python3 softmax_regression/train_softmax.py      # Phase 1: hyperparameter search
python3 softmax_regression/retrain_final.py      # Phase 2: refit on Train+Val, test eval
```

## Reproducing the Decision Tree

```bash
pip install numpy pandas matplotlib scikit-learn joblib
python3 decision_tree/train_decision_tree.py     # grid search + refit + test eval
```

## Dependencies

**Prediction only** (`pred.py`): Python ≥ 3.10, NumPy, pandas
**Model training**: additionally scikit-learn, PyTorch (MLP only), matplotlib, seaborn

## Repository structure

```text
project-csc311/
├── pred.py                          # Submission: self-contained prediction pipeline
├── data_explored.py                 # EDA, respondent-level split, feature engineering
├── export_preprocessing.py          # Exports MLP preprocessing params to JSON
├── csc311_project_training_202609.csv  # Raw survey dataset (not committed if large)
├── processed_data/                  # Derived split CSVs (clean and scaled)
│   ├── train_clean.csv / train_scaled.csv
│   ├── val_clean.csv   / val_scaled.csv
│   └── test_clean.csv  / test_scaled.csv
├── decision_tree/
│   ├── train_decision_tree.py       # Grid search, final training, test evaluation
│   ├── artifacts/decision_tree.joblib  # Saved final tree
│   ├── results/                     # Metrics, confusion matrix, tree visualisation
│   └── README.md
├── softmax_regression/
│   ├── softmax_model.py             # NumPy SoftmaxRegression class
│   ├── train_softmax.py             # Phase 1: random hyperparameter search
│   ├── retrain_final.py             # Phase 2: refit on Train+Val + test evaluation
│   ├── evaluate.py                  # Validation metrics (tuning checkpoint)
│   ├── evaluate_test.py             # Test metrics (final model)
│   ├── predict.py                   # Standalone SM inference pipeline
│   ├── test_softmax.py              # Unit tests (gradient checks, split integrity)
│   ├── artifacts/                   # Tuning checkpoint (Train-only preprocessing)
│   ├── artifacts_final/             # Final model (Train+Val preprocessing)
│   ├── results/                     # Metrics, plots, search results
│   └── README.md
├── mlp/
│   ├── train_mlp.py                 # MLP hyperparameter search + final retraining
│   ├── embed_model.py               # Embeds model weights into pred.py
│   ├── mlp_outputs/                 # Trained weights, preprocessing JSON, search logs
│   └── MLP_MODEL.md
├── audits/                          # Error analysis and audit artifacts
│   └── final_error_analysis/
├── experiments/                     # Exploratory comparison scripts (preprocessing refit)
└── submission_ready/                # MarkUs upload files (do not modify)
    ├── pred.py
    ├── code.zip
    └── SUBMISSION_MANIFEST.md
```

## Notes

- Random seed: `42` for the respondent-level split, Softmax random search, and Decision Tree. MLP final retraining uses seed `311`.
- Respondent IDs are excluded from all model features.
- Original label strings (`Soccer`, `Baseball`, `Basketball`) are preserved verbatim in all outputs.
- The internal test split (44 respondents, 132 rows) is not used for any fitting or tuning decision.
- `submission_ready/` contains the verified MarkUs upload; do not regenerate or overwrite it.
