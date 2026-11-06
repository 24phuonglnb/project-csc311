# MLP implementation handoff

## What changed

- Replaced the previous grouped-cross-validation workflow in `train_mlp.py`
  with a two-stage hyperparameter search on the existing training split and
  held-out validation split. The search does not load or inspect the test split.
- Added `mlp_inference.py`, which loads a compressed NumPy model and runs
  inference using only NumPy and pandas. Its CLI accepts the existing
  `*_clean.csv` feature format; the `label` column is optional.
- Updated `README.md` with training/inference instructions and the test-score
  caveat.
- Saved the selected configuration, validation search results, final model
  archive, and evaluation reports under `mlp_outputs/`.

## Model and preprocessing

The selected classifier is a five-member MLP ensemble. Each member has three
64-unit hidden layers with leaky ReLU activations, trained with AdamW
(learning rate `0.001`, weight decay `0.01`, batch size `32`). The validation
search selects the architecture and stopping epoch using macro-F1; final
training uses the median selected epoch (7) for each member.

The model uses the 234 features in `processed_data/*_clean.csv`. Scaling
parameters are fitted on the training data used for each search run, and on
train plus validation for final training. The five continuous/ordinal features
are standardized; missingness flags, multi-hot indicators, and TF-IDF features
remain at their natural scales.

## Results

- Validation accuracy: **97.67%**
- Validation macro-F1: **0.9767**
- Test accuracy: **91.67%** (121/132)
- Test macro-F1: **0.9172**
- Test confusion matrix (rows/columns: Baseball, Basketball, Soccer):
  `[[41, 2, 1], [2, 41, 1], [5, 0, 39]]`

**Evaluation limitation:** previous model variants and preprocessing choices
were also evaluated on this same test split before the final choice. Therefore
the reported test result is exploratory, not an unbiased one-time estimate.
For a clean final performance claim, evaluate the finalized model on a new,
untouched respondent-level holdout.

## Reproduce training

From the repository root, with PyTorch and scikit-learn installed:

```bash
python3 train_mlp.py search
python3 train_mlp.py final
```

Search writes `mlp_outputs/mlp_best_config.json` and
`mlp_outputs/mlp_validation_results.csv`. Final training writes
`mlp_outputs/mlp_model.npz` and `mlp_outputs/mlp_test_results.json`. The final
command refuses to run again if that test-results file exists, avoiding
accidental repeated evaluation.

## Run inference

Only NumPy and pandas are needed by `mlp_inference.py`:

```bash
python3 mlp_inference.py processed_data/test_clean.csv \
  --output predictions.csv
```

The exported archive can also be used from Python:

```python
import pandas as pd
from mlp_inference import predict, predict_proba

features = pd.read_csv("processed_data/test_clean.csv")
labels = predict(features)
probabilities = predict_proba(features)
```

The input must have the same clean feature columns and names as the training
data. If present, `label` is ignored.

## Verification performed

- Python compilation and `git diff --check`.
- Confirmed the search data loader reads only train and validation.
- Ran the inference CLI against validation data; it returned 129 predictions,
  probabilities summed to one, and validation accuracy was 99.22%.
- Compared NumPy inference probabilities with the trained PyTorch ensemble
  before the final test evaluation.
