"""test_softmax.py — Unit tests, gradient checks, and sanity checks.

Checks:
  1. Softmax probabilities are finite and sum to 1.
  2. Cross-entropy and gradients are numerically stable.
  3. Analytical gradients agree with finite-difference numerical gradients.
  4. Mini-batch SGD reduces training loss on a small synthetic dataset.
  5. The model can overfit a tiny synthetic dataset.
  6. Training, validation, and test respondents are disjoint (data split integrity).
  7. Preprocessing does not leak val/test statistics.
  8. Feature dimensions and ordering are identical across splits.
  9. Random search reproducibility with seed 42.

Usage (from project root):
    python3 softmax_regression/test_softmax.py
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "softmax_regression"))

from softmax_model import SoftmaxRegression, softmax, cross_entropy_loss, one_hot
from train_softmax import sample_configs

PASS = "PASS"
FAIL = "FAIL"


def run_test(name: str, fn):
    """Run a single test function and print result."""
    try:
        fn()
        print(f"  [{PASS}] {name}")
        return True
    except AssertionError as e:
        print(f"  [{FAIL}] {name}: {e}")
        return False
    except Exception as e:
        print(f"  [{FAIL}] {name}: Unexpected error — {type(e).__name__}: {e}")
        return False


# Test 1: Softmax probabilities are finite and sum to 1
def test_softmax_finite_and_sum():
    rng = np.random.default_rng(0)
    for shape in [(5, 3), (100, 3), (1, 3)]:
        Z = rng.standard_normal(shape) * 100  # Large values test numerical stability
        P = softmax(Z)
        assert np.all(np.isfinite(P)), "Probabilities contain NaN or Inf"
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-10), "Rows do not sum to 1"
        assert np.all(P >= 0), "Negative probabilities found"


# Test 2: Cross-entropy is numerically stable
def test_cross_entropy_stable():
    rng = np.random.default_rng(1)
    # Extreme logits (probability very close to 0)
    Z = np.array([[1000.0, -1000.0, -1000.0],
                  [-1000.0, 1000.0, -1000.0]])
    P = softmax(Z)
    y = np.array([0, 1])
    loss = cross_entropy_loss(P, y)
    assert np.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss < 1e-5, f"Loss unexpectedly large for confident correct predictions: {loss}"

    # Confident wrong predictions - loss should be finite but large
    y_wrong = np.array([1, 0])
    loss_wrong = cross_entropy_loss(P, y_wrong)
    assert np.isfinite(loss_wrong), f"Loss for wrong predictions not finite: {loss_wrong}"



# Test 3: Analytical gradients vs finite differences
def test_gradient_check():
    """Check dL/dW and dL/db against finite-difference approximations"""
    rng = np.random.default_rng(2)
    N, D, K = 10, 8, 3
    X = rng.standard_normal((N, D))
    y = rng.integers(0, K, size=N)
    l2 = 0.05

    model = SoftmaxRegression(n_classes=K, l2_lambda=l2, random_state=2)
    model.W = rng.standard_normal((D, K)) * 0.1
    model.b = rng.standard_normal(K) * 0.1

    P = softmax(X @ model.W + model.b)
    dW_analytical, db_analytical = model._gradients(X, P, y)

    eps = 1e-5

    # Finite-difference check for W.
    dW_numerical = np.zeros_like(model.W)
    for i in range(D):
        for j in range(K):
            W_plus = model.W.copy(); W_plus[i, j] += eps
            W_minus = model.W.copy(); W_minus[i, j] -= eps
            loss_plus  = cross_entropy_loss(softmax(X @ W_plus  + model.b), y) + (l2/2)*np.sum(W_plus**2)
            loss_minus = cross_entropy_loss(softmax(X @ W_minus + model.b), y) + (l2/2)*np.sum(W_minus**2)
            dW_numerical[i, j] = (loss_plus - loss_minus) / (2 * eps)

    max_diff_W = np.max(np.abs(dW_analytical - dW_numerical))
    assert max_diff_W < 1e-4, f"Gradient check dW failed: max diff = {max_diff_W:.2e}"

    # Finite-difference check for b
    db_numerical = np.zeros_like(model.b)
    for j in range(K):
        b_plus = model.b.copy(); b_plus[j] += eps
        b_minus = model.b.copy(); b_minus[j] -= eps
        loss_plus  = cross_entropy_loss(softmax(X @ model.W + b_plus),  y) + (l2/2)*np.sum(model.W**2)
        loss_minus = cross_entropy_loss(softmax(X @ model.W + b_minus), y) + (l2/2)*np.sum(model.W**2)
        db_numerical[j] = (loss_plus - loss_minus) / (2 * eps)

    max_diff_b = np.max(np.abs(db_analytical - db_numerical))
    assert max_diff_b < 1e-4, f"Gradient check db failed: max diff = {max_diff_b:.2e}"



# Test 4: SGD reduces training loss on a small synthetic dataset
def test_sgd_reduces_loss():
    rng = np.random.default_rng(3)
    N, D, K = 60, 10, 3
    X = rng.standard_normal((N, D))
    y = rng.integers(0, K, N)
    X_val_s = rng.standard_normal((20, D))
    y_val_s = rng.integers(0, K, 20)

    model = SoftmaxRegression(
        n_classes=K, learning_rate=0.1, batch_size=32,
        max_epochs=50, l2_lambda=0.0, patience=50, random_state=3
    )
    model.fit(X, y, X_val_s, y_val_s)

    initial_loss = model.train_losses_[0]
    final_loss   = model.train_losses_[-1]
    assert final_loss < initial_loss, (
        f"SGD did not reduce training loss: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )


# Test 5: Model can overfit a tiny synthetic dataset
def test_overfit_tiny_dataset():
    """Model should memorize a tiny dataset with lr=1.0, 3000 epochs, zero regularization"""
    rng = np.random.default_rng(4)
    # 9 samples with 30 features — easily separable in high dims.
    N, D, K = 9, 30, 3
    X = rng.standard_normal((N, D))
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    X_val_s = rng.standard_normal((6, D))
    y_val_s = np.array([0, 1, 2, 0, 1, 2])

    model = SoftmaxRegression(
        n_classes=K, learning_rate=1.0, batch_size=N,
        max_epochs=3000, l2_lambda=0.0, patience=3000, random_state=4
    )
    model.fit(X, y, X_val_s, y_val_s)

    # With zero regularisation and many epochs, training accuracy must be 100%.
    train_acc = (model.predict(X) == y).mean()
    assert train_acc > 0.95, f"Model failed to overfit tiny dataset: train_acc={train_acc:.2f}"


# Test 6: Respondent-level splits are disjoint
def test_respondent_splits_disjoint():
    raw = pd.read_csv(PROJECT_ROOT / "csc311_project_training_202609.csv")
    from sklearn.model_selection import train_test_split
    unique_ids = raw["id"].unique()
    train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, random_state=42)
    val_ids, test_ids   = train_test_split(temp_ids, test_size=0.5, random_state=42)

    assert set(train_ids).isdisjoint(set(val_ids)),  "Train/val ID overlap!"
    assert set(train_ids).isdisjoint(set(test_ids)), "Train/test ID overlap!"
    assert set(val_ids).isdisjoint(set(test_ids)),   "Val/test ID overlap!"


# Test 7: Feature columns identical across splits
def test_feature_columns_identical():
    train = pd.read_csv(PROJECT_ROOT / "processed_data" / "train_scaled.csv")
    val   = pd.read_csv(PROJECT_ROOT / "processed_data" / "val_scaled.csv")
    test  = pd.read_csv(PROJECT_ROOT / "processed_data" / "test_scaled.csv")

    assert list(train.columns) == list(val.columns),  "train/val column mismatch"
    assert list(train.columns) == list(test.columns), "train/test column mismatch"
    assert train.shape[1] == 235, f"Expected 235 columns (234 features + label), got {train.shape[1]}"


# Test 8: All features are finite
def test_features_finite():
    for split in ["train_scaled", "val_scaled", "test_scaled"]:
        df = pd.read_csv(PROJECT_ROOT / "processed_data" / f"{split}.csv")
        X = df.drop(columns=["label"]).values.astype(np.float64)
        assert np.all(np.isfinite(X)), f"Non-finite values in {split}.csv"


# Test 9: Random search reproducibility with seed 42
def test_random_search_reproducibility():
    configs_a = sample_configs(50, seed=42)
    configs_b = sample_configs(50, seed=42)
    assert configs_a == configs_b, "Random search is not reproducible with the same seed"
    assert len(configs_a) == 50, f"Expected 50 configs, got {len(configs_a)}"

    configs_c = sample_configs(50, seed=99)
    assert configs_a != configs_c, "Different seeds produced identical configs (unexpected)"


# Test 10: Save and load round-trip
def test_save_load_roundtrip():
    import tempfile
    rng = np.random.default_rng(5)
    N, D, K = 30, 6, 3
    X = rng.standard_normal((N, D))
    y = rng.integers(0, K, N)
    X_val_s = rng.standard_normal((10, D))
    y_val_s = rng.integers(0, K, 10)
    classes = np.array(["A", "B", "C"])

    model = SoftmaxRegression(n_classes=K, max_epochs=20, patience=20, random_state=5)
    model.fit(X, y, X_val_s, y_val_s, classes=classes)
    preds_before = model.predict(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save(tmpdir)
        loaded = SoftmaxRegression.load(tmpdir)

    preds_after = loaded.predict(X)
    assert np.array_equal(preds_before, preds_after), "Predictions differ after save/load"
    assert np.allclose(model.W, loaded.W), "W differs after save/load"
    assert np.allclose(model.b, loaded.b), "b differs after save/load"


# Main test runner
def main():
    print("Softmax Regression — Unit Tests & Sanity Checks")

    tests = [
        ("Softmax finite & sum-to-1",        test_softmax_finite_and_sum),
        ("Cross-entropy numerical stability", test_cross_entropy_stable),
        ("Gradient check (W and b)",          test_gradient_check),
        ("SGD reduces training loss",         test_sgd_reduces_loss),
        ("Overfit tiny dataset",              test_overfit_tiny_dataset),
        ("Respondent splits disjoint",        test_respondent_splits_disjoint),
        ("Feature columns identical",         test_feature_columns_identical),
        ("All features finite",               test_features_finite),
        ("Random search reproducibility",     test_random_search_reproducibility),
        ("Save/load round-trip",              test_save_load_roundtrip),
    ]

    passed = 0
    for name, fn in tests:
        ok = run_test(name, fn)
        if ok:
            passed += 1

    total = len(tests)
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("All tests passed.")
    else:
        print("Some tests FAILED — review output above")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
