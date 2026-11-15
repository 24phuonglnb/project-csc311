# Decision Tree - CSC311 Sport Classification

Decision tree classifier (scikit-learn) for Baseball / Basketball / Soccer.

## Files

```text
decision_tree/
├── train_decision_tree.py      # grid search, final training, test evaluation
├── artifacts/
│   └── decision_tree.joblib    # final tree + ordered feature names
└── results/
    ├── validation_grid.csv             # all 28 configurations
    ├── best_config.json                # selected configuration
    ├── test_metrics.json               # internal test metrics
    ├── test_classification_report.txt  # per-class precision / recall / F1
    ├── test_confusion_matrix.png
    └── tree_top_levels.png             # top 3 levels of the final tree
```

## How to run

From the project root, after `processed_data/` exists (`python3 data_explored.py`):

```bash
pip install numpy pandas matplotlib scikit-learn joblib
python3 decision_tree/train_decision_tree.py
```

The script uses the **un-scaled** `processed_data/*_clean.csv` files (trees are
insensitive to feature scaling).

## Method

1. **Grid search on the validation set** (trained on train only):
   `max_depth` in {2, 3, 4, 5, 7, 10, unlimited} x `min_samples_leaf` in {1, 2, 5, 10}.
   Criterion is Gini (scikit-learn default). Selection: highest validation
   macro-F1, ties broken by validation accuracy, then by fewer leaves.
2. **Retrain** the selected configuration on train + validation.
3. **Evaluate once** on the internal test set. The test set is not used for any
   choice.

## Results

Selected configuration: `criterion="gini"`, `max_depth=5`, `min_samples_leaf=5`, `random_state=42` (12 leaves when fitted on train).

| Setting | Accuracy | Macro-F1 |
|---|---:|---:|
| Validation (trained on train) | 90.70% | 0.905 |
| Internal test (trained on train + validation) | 81.82% | 0.817 |

Internal test, per class:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Baseball | 0.814 | 0.795 | 0.805 |
| Basketball | 0.737 | 0.955 | 0.832 |
| Soccer | 0.969 | 0.705 | 0.816 |

Observations:

- An unrestricted tree overfits (train 99.5% vs validation 86.8%); limiting depth
  and leaf size helps.
- The tree is rarely wrong when it predicts Soccer (precision 0.97) but misses
  about 30% of Soccer responses (recall 0.70), mostly predicting Basketball.
- Top splits use TF-IDF terms (`orange`, `waka`, `green`, `blue`) and
  `location_Indoor facility`, consistent with the exploratory analysis.
- Upweighting Soccer (class weight 2.0) found one more Soccer response on
  validation but lowered overall validation accuracy from 90.7% to 82.9%, so it
  was not used.

## Caveats

- The validation score is optimistic because 28 configurations were compared on
  it; the test score is the better estimate.
- The internal test set has 44 respondents (132 rows), so scores can vary by
  several points with a different split.
- Grouped cross-validation is **not** implemented: the exported CSVs do not
  contain respondent IDs.
- The saved joblib model requires scikit-learn and is not suitable for the final
  submission if scikit-learn is prohibited. If the tree is selected, export its
  structure and implement and verify inference using permitted dependencies.
