"""evaluate_test.py - Final test-set evaluation for the best Softmax model.

Usage (from project root):
    python3 softmax_regression/evaluate_test.py

Loads the saved best model from softmax_regression/artifacts/ and evaluates it
exactly once on processed_data/test_scaled.csv. Produces:
  - Test classification report (text)
  - Test metrics summary (JSON)
  - Test confusion matrix heatmap (PNG)
  - Test predictions with class probabilities (CSV)

Does NOT retrain, tune, or modify the model in any way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "softmax_regression"))

from softmax_model import SoftmaxRegression

DATA_DIR     = PROJECT_ROOT / "processed_data"
ARTIFACT_DIR = PROJECT_ROOT / "softmax_regression" / "artifacts"
RESULTS_DIR  = PROJECT_ROOT / "softmax_regression" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_test_data():
    """Return X_test, y_test (integer), y_test_labels (string), classes

    Feature columns are selected and ordered using artifacts/feature_names.json
    to guarantee exact alignment with the training feature matrix
    """
    with open(ARTIFACT_DIR / "feature_names.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_names"]
    classes = np.array(meta["classes"])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    test = pd.read_csv(DATA_DIR / "test_scaled.csv")

    missing = [c for c in feature_cols if c not in test.columns]
    if missing:
        raise ValueError(f"test_scaled.csv is missing {len(missing)} expected columns: {missing[:5]}…")

    X_test = test[feature_cols].values.astype(np.float64)
    y_test = test["label"].map(class_to_idx).values.astype(np.int64)
    y_test_labels = test["label"].values
    return X_test, y_test, y_test_labels, classes, feature_cols


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    save_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title("Softmax Regression - Confusion Matrix (Test)", fontsize=13)

    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved test confusion matrix: {save_path}")


def main():
    print("Softmax Regression - Final Test-Set Evaluation")

    # Load model (read-only - no retraining)
    model = SoftmaxRegression.load(ARTIFACT_DIR)
    print(f"\nModel loaded from: {ARTIFACT_DIR}")
    print(f"  Best epoch (0-indexed) : {model.best_epoch_}  (epoch {model.best_epoch_ + 1})")
    print(f"  Hyperparameters:")
    print(f"    learning_rate = {model.learning_rate}")
    print(f"    batch_size    = {model.batch_size}")
    print(f"    max_epochs    = {model.max_epochs}")
    print(f"    l2_lambda     = {model.l2_lambda}")
    print(f"    patience      = {model.patience}")

    # Load test data
    X_test, y_test, y_test_labels, classes, feature_cols = load_test_data()
    print(f"\nTest set: {X_test.shape[0]} observations × {X_test.shape[1]} features")
    print(f"Classes : {list(classes)}")

    # Predictions
    proba         = model.predict_proba(X_test)
    y_pred_int    = np.argmax(proba, axis=1)
    y_pred_labels = classes[y_pred_int]

    # Probability sanity check
    prob_sums = proba.sum(axis=1)
    max_dev = np.abs(prob_sums - 1).max()
    assert np.allclose(prob_sums, 1.0, atol=1e-6), (
        f"Probabilities don't sum to 1 (max deviation {max_dev:.2e})"
    )
    print(f"\nProbability sanity: all rows sum to 1 (max deviation {max_dev:.2e})")

    # Metrics
    test_acc       = accuracy_score(y_test_labels, y_pred_labels)
    test_macro_f1  = f1_score(y_test_labels, y_pred_labels, average="macro")
    test_weighted_f1 = f1_score(y_test_labels, y_pred_labels, average="weighted")
    report         = classification_report(y_test_labels, y_pred_labels, digits=4)

    print("TEST RESULTS")
    print(f"  Test accuracy    : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Test macro-F1    : {test_macro_f1:.4f}")
    print(f"  Test weighted-F1 : {test_weighted_f1:.4f}")
    print(f"\nPer-class report:")
    print(report)


    # Save: classification report (text)
    report_path = RESULTS_DIR / "test_classification_report.txt"
    with open(report_path, "w") as f:
        f.write("Softmax Regression - Final Test-Set Evaluation\n")
        f.write("\n\n")
        f.write("Model artifact : softmax_regression/artifacts\n")
        f.write("Test dataset   : processed_data/test_scaled.csv\n\n")
        f.write(f"Best epoch (0-indexed) : {model.best_epoch_}  (epoch {model.best_epoch_ + 1})\n\n")
        f.write(f"Test accuracy    : {test_acc:.4f}\n")
        f.write(f"Test macro-F1    : {test_macro_f1:.4f}\n")
        f.write(f"Test weighted-F1 : {test_weighted_f1:.4f}\n\n")
        f.write(report)
    print(f"Saved test report: {report_path}")

    # Save: metrics JSON
    metrics = {
        "dataset": "test_scaled.csv",
        "n_observations": int(X_test.shape[0]),
        "n_features": int(X_test.shape[1]),
        "model_artifact": "softmax_regression/artifacts",
        "best_epoch": int(model.best_epoch_),
        "hyperparameters": {
            "learning_rate": model.learning_rate,
            "batch_size": model.batch_size,
            "max_epochs": model.max_epochs,
            "l2_lambda": model.l2_lambda,
            "patience": model.patience,
            "random_state": model.random_state,
        },
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "test_weighted_f1": float(test_weighted_f1),
        "per_class": {},
    }
    for cls in classes:
        mask = y_test_labels == cls
        pred_mask = y_pred_labels == cls
        tp = int((mask & pred_mask).sum())
        support = int(mask.sum())
        precision = float(tp / pred_mask.sum()) if pred_mask.sum() > 0 else 0.0
        recall    = float(tp / support) if support > 0 else 0.0
        f1        = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        metrics["per_class"][cls] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   support,
        }

    metrics_path = RESULTS_DIR / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved test metrics: {metrics_path}")

    # Save: confusion matrix PNG
    plot_confusion_matrix(y_test_labels, y_pred_labels, classes,
                          RESULTS_DIR / "test_confusion_matrix.png")

    # Save: predictions CSV
    preds_df = pd.DataFrame({
        "true_label":      y_test_labels,
        "predicted_label": y_pred_labels,
        **{f"prob_{c}": proba[:, i] for i, c in enumerate(classes)},
    })
    preds_path = RESULTS_DIR / "test_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"Saved test predictions: {preds_path}")

    print("Evaluation complete.")
    return metrics


if __name__ == "__main__":
    main()
