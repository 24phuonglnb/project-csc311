"""retrain_final.py — Final retraining of the best Softmax model on Train+Val.

Usage (from project root):
    python3 softmax_regression/retrain_final.py

Procedure (follows the project proposal):
  1. Load the best hyperparameters from results/best_config.json.
     Do NOT re-run random search.
  2. Combine the already-preprocessed train_scaled.csv and val_scaled.csv
     into a single 738-observation dataset.
     Do NOT refit the scaler or any other preprocessing artifact.
  3. Train from scratch for exactly (best_epoch + 1) epochs = 3 epochs,
     using fit_fixed_epochs() — no early stopping, no validation split.

     Rationale for final training duration:
       The validation-based search found that the best weights were produced
       at epoch index 2 (epoch 3, 1-indexed).  Early stopping fired two
       epochs later (epoch 5), confirming convergence at epoch 3.  Retraining
       for the same 3 epochs on the larger Train+Val set is the standard
       practice: it replicates exactly the update schedule that produced the
       best validation checkpoint, without touching the test set.
  4. Save the final model to artifacts_final/ (distinct from the tuning
     checkpoint in artifacts/).
  5. Evaluate once on the internal held-out test set and save results to
     results/final_test_*.

The original tuning checkpoint (artifacts/) is preserved for reproducibility
of the validation-based hyperparameter selection results.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "softmax_regression"))

from softmax_model import SoftmaxRegression, cross_entropy_loss

DATA_DIR         = PROJECT_ROOT / "processed_data"
RESULTS_DIR      = PROJECT_ROOT / "softmax_regression" / "results"
TUNING_ART_DIR   = PROJECT_ROOT / "softmax_regression" / "artifacts"
FINAL_ART_DIR    = PROJECT_ROOT / "softmax_regression" / "artifacts_final"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FINAL_ART_DIR.mkdir(parents=True, exist_ok=True)


def load_trainval_data():
    """Combine train and val splits; return arrays and class metadata."""
    with open(TUNING_ART_DIR / "feature_names.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_names"]
    classes      = np.array(meta["classes"])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train = pd.read_csv(DATA_DIR / "train_scaled.csv")
    val   = pd.read_csv(DATA_DIR / "val_scaled.csv")
    combined = pd.concat([train, val], ignore_index=True)

    X = combined[feature_cols].values.astype(np.float64)
    y = combined["label"].map(class_to_idx).values.astype(np.int64)
    return X, y, classes, class_to_idx, feature_cols


def load_test_data(feature_cols, classes):
    """Load the held-out test set using the same feature ordering."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    test = pd.read_csv(DATA_DIR / "test_scaled.csv")

    missing = [c for c in feature_cols if c not in test.columns]
    if missing:
        raise ValueError(f"test_scaled.csv missing {len(missing)} columns: {missing[:5]}")

    X_test       = test[feature_cols].values.astype(np.float64)
    y_test       = test["label"].map(class_to_idx).values.astype(np.int64)
    y_test_labels = test["label"].values
    return X_test, y_test, y_test_labels


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
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
    ax.set_title("Softmax Regression — Confusion Matrix (Test, Final Model)", fontsize=12)
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
    print(f"Saved confusion matrix → {save_path}")


def main():
    print("=" * 70)
    print("Softmax Regression — Final Retraining on Train + Val")
    print("=" * 70)

    # Load best hyperparameters (do not re-run search)
    best_cfg_path = RESULTS_DIR / "best_config.json"
    with open(best_cfg_path) as f:
        best_cfg = json.load(f)

    # best_epoch is 0-indexed; we train for (best_epoch + 1) complete epochs
    tuning_best_epoch = int(best_cfg["best_epoch"])   # 2
    final_n_epochs    = tuning_best_epoch + 1          # 3

    print(f"\nBest config (from tuning):")
    print(f"  learning_rate = {best_cfg['learning_rate']}")
    print(f"  batch_size    = {best_cfg['batch_size']}")
    print(f"  l2_lambda     = {best_cfg['l2_lambda']}")
    print(f"  random_seed   = 42")
    print(f"  tuning best_epoch (0-indexed) = {tuning_best_epoch}  (epoch {tuning_best_epoch + 1})")
    print(f"  → final training epochs       = {final_n_epochs}")

    # Load Train + Val combined (do NOT refit any preprocessing artifacts)
    X_tv, y_tv, classes, _, feature_cols = load_trainval_data()
    print(f"\nTrain+Val combined: {X_tv.shape[0]} observations × {X_tv.shape[1]} features")
    print(f"Classes            : {list(classes)}")

    # Train final model from scratch for exactly final_n_epochs epochs
    model = SoftmaxRegression(
        n_classes=len(classes),
        learning_rate=best_cfg["learning_rate"],
        batch_size=int(best_cfg["batch_size"]),
        max_epochs=final_n_epochs,  # unused by fit_fixed_epochs, kept for metadata
        l2_lambda=best_cfg["l2_lambda"],
        patience=best_cfg["patience"],  # unused by fit_fixed_epochs, kept for metadata
        random_state=42,
    )

    t0 = time.perf_counter()
    model.fit_fixed_epochs(X_tv, y_tv, n_epochs=final_n_epochs, classes=classes)
    elapsed = time.perf_counter() - t0

    print(f"\nFinal training complete ({elapsed:.2f}s)")
    for ep, (tl, tc) in enumerate(
        zip(model.train_losses_, model.train_ce_losses_), start=1
    ):
        print(f"  Epoch {ep}: train_loss(CE+L2)={tl:.6f}  train_CE={tc:.6f}")

    # Save final model weights to artifacts_final/
    model.save(FINAL_ART_DIR)

    # Update model_meta.json with final-training context
    meta_path = FINAL_ART_DIR / "model_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta["training_mode"] = "final_trainval"
    meta["training_summary"]["n_epochs_fixed"]          = final_n_epochs
    meta["training_summary"]["tuning_best_epoch_0idx"]  = tuning_best_epoch
    meta["training_summary"]["train_val_n_observations"] = int(X_tv.shape[0])
    meta["training_summary"]["note"] = (
        "Retrained from scratch on Train+Val (738 obs) for exactly "
        f"{final_n_epochs} epochs (= tuning best_epoch+1). "
        "No early stopping. Preprocessing artifacts unchanged."
    )
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved final model → {FINAL_ART_DIR}")

    # Copy feature_names.json so artifacts_final/ is self-contained
    shutil.copy(TUNING_ART_DIR / "feature_names.json",
                FINAL_ART_DIR / "feature_names.json")
    # Copy preprocessing artifacts so predict.py works from artifacts_final/
    final_pre_dir   = FINAL_ART_DIR / "preprocessing"
    tuning_pre_dir  = TUNING_ART_DIR / "preprocessing"
    if tuning_pre_dir.exists() and not final_pre_dir.exists():
        shutil.copytree(tuning_pre_dir, final_pre_dir)

    # --- Test evaluation (run once, final model only) ---
    print("\n" + "=" * 70)
    print("Test-Set Evaluation — Final Model")
    print("=" * 70)

    X_test, y_test, y_test_labels = load_test_data(feature_cols, classes)
    print(f"Test set: {X_test.shape[0]} observations × {X_test.shape[1]} features")

    proba          = model.predict_proba(X_test)
    y_pred_int     = np.argmax(proba, axis=1)
    y_pred_labels  = classes[y_pred_int]

    prob_sums = proba.sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-6), (
        f"Probabilities don't sum to 1 (max dev {np.abs(prob_sums-1).max():.2e})"
    )

    test_ce           = cross_entropy_loss(proba, y_test)
    test_acc          = accuracy_score(y_test_labels, y_pred_labels)
    test_macro_f1     = f1_score(y_test_labels, y_pred_labels, average="macro")
    test_weighted_f1  = f1_score(y_test_labels, y_pred_labels, average="weighted")
    report            = classification_report(y_test_labels, y_pred_labels, digits=4)

    print(f"\n  Test accuracy    : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Test macro-F1    : {test_macro_f1:.4f}")
    print(f"  Test weighted-F1 : {test_weighted_f1:.4f}")
    print(f"  Test cross-entropy: {test_ce:.6f}")
    print(f"\nPer-class report:")
    print(report)

    # Save classification report
    rpt_path = RESULTS_DIR / "final_test_classification_report.txt"
    with open(rpt_path, "w") as f:
        f.write("Softmax Regression — Final Test-Set Evaluation (Retrained on Train+Val)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model artifact        : softmax_regression/artifacts_final\n")
        f.write(f"Test dataset          : processed_data/test_scaled.csv\n")
        f.write(f"Training data         : Train + Val ({X_tv.shape[0]} observations)\n")
        f.write(f"Final training epochs : {final_n_epochs} "
                f"(= tuning best_epoch {tuning_best_epoch+1}, 1-indexed)\n\n")
        f.write(f"Hyperparameters (unchanged from tuning):\n")
        f.write(f"  learning_rate = {best_cfg['learning_rate']}\n")
        f.write(f"  batch_size    = {int(best_cfg['batch_size'])}\n")
        f.write(f"  l2_lambda     = {best_cfg['l2_lambda']}\n")
        f.write(f"  random_seed   = 42\n\n")
        f.write(f"Test accuracy        : {test_acc:.4f}\n")
        f.write(f"Test macro-F1        : {test_macro_f1:.4f}\n")
        f.write(f"Test weighted-F1     : {test_weighted_f1:.4f}\n")
        f.write(f"Test cross-entropy   : {test_ce:.6f}\n\n")
        f.write(report)
    print(f"Saved final report → {rpt_path}")

    # Save metrics JSON
    per_class = {}
    for cls in classes:
        mask      = y_test_labels == cls
        pred_mask = y_pred_labels == cls
        tp        = int((mask & pred_mask).sum())
        support   = int(mask.sum())
        prec      = float(tp / pred_mask.sum()) if pred_mask.sum() > 0 else 0.0
        rec       = float(tp / support) if support > 0 else 0.0
        f1_cls    = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[cls] = {
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1_cls, 4),
            "support":   support,
        }

    final_metrics = {
        "model":                  "final (retrained on Train+Val)",
        "model_artifact":         "softmax_regression/artifacts_final",
        "dataset":                "test_scaled.csv",
        "n_observations_test":    int(X_test.shape[0]),
        "n_observations_trainval": int(X_tv.shape[0]),
        "n_features":             int(X_test.shape[1]),
        "final_training_epochs":  final_n_epochs,
        "tuning_best_epoch_0idx": tuning_best_epoch,
        "hyperparameters": {
            "learning_rate": best_cfg["learning_rate"],
            "batch_size":    int(best_cfg["batch_size"]),
            "l2_lambda":     best_cfg["l2_lambda"],
            "random_state":  42,
        },
        "test_cross_entropy":  float(test_ce),
        "test_accuracy":       float(test_acc),
        "test_macro_f1":       float(test_macro_f1),
        "test_weighted_f1":    float(test_weighted_f1),
        "per_class":           per_class,
        "reference_tuning_checkpoint": {
            "model_artifact":   "softmax_regression/artifacts",
            "val_accuracy":     best_cfg["val_accuracy"],
            "val_macro_f1":     best_cfg["val_macro_f1"],
            "note": (
                "Validation metrics for the tuning checkpoint trained on Train only. "
                "Test metrics for the tuning checkpoint are in results/test_metrics.json."
            ),
        },
    }
    metrics_path = RESULTS_DIR / "final_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Saved final metrics → {metrics_path}")

    # Save confusion matrix PNG
    plot_confusion_matrix(y_test_labels, y_pred_labels, classes,
                          RESULTS_DIR / "final_test_confusion_matrix.png")

    # Save predictions CSV
    preds_df = pd.DataFrame({
        "true_label":      y_test_labels,
        "predicted_label": y_pred_labels,
        **{f"prob_{c}": proba[:, i] for i, c in enumerate(classes)},
    })
    preds_path = RESULTS_DIR / "final_test_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"Saved final predictions → {preds_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Tuning checkpoint  (Train only, epoch {tuning_best_epoch+1}):")
    print(f"    Val accuracy : {best_cfg['val_accuracy']:.4f}")
    print(f"    Val macro-F1 : {best_cfg['val_macro_f1']:.4f}")
    print(f"  Final model  (Train+Val, {final_n_epochs} epochs):")
    print(f"    Test accuracy : {test_acc:.4f}")
    print(f"    Test macro-F1 : {test_macro_f1:.4f}")
    print("=" * 70)
    print("Done.  Final model saved to artifacts_final/.")
    print("Preprocessing artifacts unchanged — predict.py will load from artifacts_final/.")

    return final_metrics


if __name__ == "__main__":
    main()
