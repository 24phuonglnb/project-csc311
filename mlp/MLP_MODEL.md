# Final MLP model

## Pipeline

1. **Features:** the 234 features in `processed_data/*_clean.csv`.
2. **k-means cluster features:** each free-text question's 50 TF-IDF columns are
   clustered into k = 10 groups. Each row's cluster ID is added as a one-hot
   vector, giving 40 extra features.
3. **Standardization:** every feature is scaled to mean 0 and standard deviation 1.
4. **PCA:** keeps the components that explain 80% of the variance (105
   components after refitting on train + validation).
5. **MLP:** 105 → 16 → 16 → 3, with tanh activations and a softmax output.

| Hyperparameter | Value |
|---|---|
| Optimizer | Mini-batch SGD (no momentum) |
| Learning rate | 0.2 |
| Batch size | 64 |
| L2 coefficient | 0.01 |
| Epochs | 8 (median early-stopping epoch; patience 2, max 250) |
| Seed | 311 |

## How it was selected

- **Search:** 20 pipeline variants were compared: PCA {off, 80, 90, 95, 99%}
  × k-means k {off, 3, 5, 10}. Each got the same two-stage random search of 64
  MLP configurations over the project grid.
- **Scoring:** each configuration was trained on train and scored on
  validation, by mean macro-F1 over 5 seeds. No cross-validation was used.
- **Final fit:** the winning pipeline and MLP were refit on train + validation
  (738 rows).

Validation and test splits are grouped by respondent.

## Results

| | Accuracy | Macro-F1 |
|---|---|---|
| Validation (mean of 5 seeds, trained on train only) | 97.21% | 0.9721 |
| Test (132 rows) | **89.39%** (118/132) | 0.8937 |

| Test class | Precision | Recall | F1 |
|---|---|---|---|
| Baseball | 0.867 | 0.886 | 0.876 |
| Basketball | 0.875 | 0.955 | 0.913 |
| Soccer | 0.949 | 0.841 | 0.892 |

Test confusion matrix (rows = true, columns = predicted):

| | Baseball | Basketball | Soccer |
|---|---|---|---|
| **Baseball** | 39 | 3 | 2 |
| **Basketball** | 2 | 42 | 0 |
| **Soccer** | 4 | 3 | 37 |

For comparison, the same search without PCA or k-means scored 95.97% on
validation and 88.64% on test. The test split has been evaluated twice, once
for each of these two models. Both were chosen on validation first.

## Submission

`pred.py` defines `predict_all(filename)`. It reads a raw survey CSV and returns
a list of labels, using only the standard library, NumPy, and pandas. It needs
two files, which it looks for next to itself or in `mlp_outputs/`:

- `mlp_model.npz`: cluster centroids, scaler, PCA, and MLP weights
- `mlp_preprocessing.json`: training medians, multi-select options, and TF-IDF
  vocabularies and IDF weights

The three files total about 270 KB. Predicting 60 rows takes under 0.1 s.

## Reproduce

```bash
pip install torch scikit-learn numpy pandas
python3 export_preprocessing.py   # writes mlp_preprocessing.json; checks pred.py
                                  # reproduces processed_data/*_clean.csv
python3 train_mlp.py search       # validation-only search (~4 min)
python3 train_mlp.py final        # refit, single test evaluation, export
```
