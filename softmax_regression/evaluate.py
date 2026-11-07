"""evaluate.py — Validation metrics and visualisations for the best Softmax model.

Usage (from project root):
    python3 softmax_regression/evaluate.py

Loads the saved best model from softmax_regression/artifacts/ and produces:
  - Loss curves (train + val)
  - Confusion matrix heatmap
  - Per-class classification report
  - All plots saved to softmax_regression/results/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; saves figures to disk.
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


# Data loading
def load_val_data():
    """Return X_val, y_val (integer), y_val_labels (string), classes."""
    with open(ARTIFACT_DIR / "feature_names.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_names"]
    classes = np.array(meta["classes"])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    val = pd.read_csv(DATA_DIR / "val_scaled.csv")
    X_val = val[feature_cols].values.astype(np.float64)
    y_val = val["label"].map(class_to_idx).values.astype(np.int64)
    y_val_labels = val["label"].values
    return X_val, y_val, y_val_labels, classes


# Plotting helpers
def plot_loss_curves(model: SoftmaxRegression, save_path: Path) -> None:
    """Plot training and validation loss curves and save to disk."""
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = range(1, len(model.train_losses_) + 1)
    ax.plot(epochs, model.train_losses_, label="Train Loss (CE + L2)", linewidth=2)
    if model.train_ce_losses_:
        ax.plot(epochs, model.train_ce_losses_, label="Train Cross-Entropy (CE only)",
                linewidth=2, linestyle="--")
    ax.plot(epochs, model.val_losses_,   label="Validation Cross-Entropy (CE only)", linewidth=2)
    ax.axvline(model.best_epoch_ + 1, color="red", linestyle="--", linewidth=1.5,
               label=f"Best epoch: {model.best_epoch_ + 1}")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Softmax Regression — Training & Validation Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved loss curve    → {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    save_path: Path,
) -> None:
    """Plot and save a labelled confusion matrix heatmap"""
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
    ax.set_title("Softmax Regression — Confusion Matrix (Validation)", fontsize=13)

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
    print(f"Saved confusion mat  → {save_path}")


def plot_random_search_summary(results_csv: Path, save_path: Path) -> None:
    """Plot distribution of val macro-F1 scores across random search configs"""
    if not results_csv.exists():
        print("No random_search_results.csv found; skipping summary plot.")
        return
    df = pd.read_csv(results_csv).sort_values("config_idx")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(df["config_idx"], df["val_macro_f1"], color="steelblue", alpha=0.7)
    best_idx = df["val_macro_f1"].idxmax()
    ax.bar(df.loc[best_idx, "config_idx"], df.loc[best_idx, "val_macro_f1"],
           color="crimson", label=f"Best config #{int(df.loc[best_idx, 'config_idx'])}")
    ax.set_xlabel("Config index", fontsize=12)
    ax.set_ylabel("Val macro-F1", fontsize=12)
    ax.set_title("Random Search — Validation Macro-F1 per Configuration", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved search summary → {save_path}")


# Main evaluation
def main():
    print("Softmax Regression — Validation Evaluation")

    model = SoftmaxRegression.load(ARTIFACT_DIR)
    X_val, y_val, y_val_labels, classes = load_val_data()

    # Predictions
    y_pred_int    = model.predict(X_val)
    y_pred_labels = classes[y_pred_int]
    proba         = model.predict_proba(X_val)

    # Core metrics
    val_acc      = accuracy_score(y_val_labels, y_pred_labels)
    val_macro_f1 = f1_score(y_val_labels, y_pred_labels, average="macro")

    print(f"\n  Best epoch         : {model.best_epoch_ + 1}")
    print(f"  Actual stop epoch  : {len(model.train_losses_)}")
    print(f"  Validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Validation macro-F1: {val_macro_f1:.4f}")
    print(f"\n  Hyperparameters:")
    print(f"    learning_rate = {model.learning_rate}")
    print(f"    batch_size    = {model.batch_size}")
    print(f"    max_epochs    = {model.max_epochs}")
    print(f"    l2_lambda     = {model.l2_lambda}")
    print(f"    patience      = {model.patience}")

    print("\nPer-class report:")
    report = classification_report(y_val_labels, y_pred_labels, digits=4)
    print(report)

    # Save classification report to text
    report_path = RESULTS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Best epoch: {model.best_epoch_ + 1}\n")
        f.write(f"Actual stop epoch: {len(model.train_losses_)}\n")
        f.write(f"Validation accuracy: {val_acc:.4f}\n")
        f.write(f"Validation macro-F1: {val_macro_f1:.4f}\n\n")
        f.write(report)
    print(f"Saved report        → {report_path}")

    # Probabilities sanity check
    prob_sums = proba.sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-6), f"Probabilities don't sum to 1: {prob_sums}"
    print(f"\nProbability sanity: all rows sum to 1 (max deviation {np.abs(prob_sums - 1).max():.2e})")

    # Plots
    plot_loss_curves(model, RESULTS_DIR / "loss_curves.png")
    plot_confusion_matrix(y_val_labels, y_pred_labels, classes, RESULTS_DIR / "confusion_matrix.png")
    plot_random_search_summary(
        RESULTS_DIR / "random_search_results.csv",
        RESULTS_DIR / "random_search_summary.png",
    )

    print("\nEvaluation complete.")
    return {
        "val_accuracy": val_acc,
        "val_macro_f1": val_macro_f1,
        "best_epoch": model.best_epoch_ + 1,
        "stop_epoch": len(model.train_losses_),
    }


if __name__ == "__main__":
    main()
