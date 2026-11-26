"""retrain_final.py — Final retraining of the best Softmax model on Train+Val.

Usage (from project root):
    python3 softmax_regression/retrain_final.py

Procedure (follows the project proposal):
  1. Load the best hyperparameters from results/best_config.json.
     Do NOT re-run random search.
  2. Refit ALL preprocessing on Train+Val (738 observations) from the raw CSV,
     then apply to both Train+Val and Test sets.
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
  5. Save all refit preprocessing artifacts to artifacts_final/preprocessing/.
  6. Evaluate once on the internal held-out test set and save results to
     results/final_test_*.

The original tuning checkpoint (artifacts/) is preserved for reproducibility
of the validation-based hyperparameter selection results.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

RAW_CSV = PROJECT_ROOT / "csc311_project_training_202609.csv"

EASE_COL = (
    "How easy do you think it would be for someone with no prior experience "
    "to start playing this sport recreationally? Rate it on a scale from 0 to 10, "
    "where 0 means extremely difficult and 10 means extremely easy."
)
PRICE_COL = (
    "How much would you be willing to pay for a decent seat "
    "at a professional competition of this sport?"
)
LIKERT_COLS = [
    "Success in this sport depends heavily on strategy and planning.",
    "Players must be comfortable with a high risk of injury.",
    "Individual brilliance can outweigh teamwork in this sport.",
]
MULTI_COLS = [
    ("Where might you play this sport?", "location"),
    ("During which seasons would you be most likely to play this sport recreationally in Toronto?", "season"),
    ("What types of music do you most associate with this sport?", "music"),
]
TEXT_COLS = [
    "Which song do you associate with this sport?",
    "What foods and drinks do you associate with watching this sport in the place or culture you know best?",
    "What colour do you associate with this sport?",
    "How do you feel when you watch this sport?",
]
MISSING_MARKERS = {"", "n/a", "na", "null", "none", "nan", "-", "unknown"}
LIKERT_MAP = {"Strongly disagree": -2, "Disagree": -1, "Neutral": 0, "Agree": 1, "Strongly agree": 2}


def _normalize_missing(df):
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        if col in {"id", "label"}:
            continue
        s = df[col].astype("string").str.strip()
        df[col] = s.mask(s.str.lower().isin(MISSING_MARKERS))
    return df


def _parse_price_val(value):
    if pd.isna(value):
        return float("nan")
    text = str(value).strip().lower()
    if text in {"free", "nothing", "zero", "unwilling", "0", "$0"}:
        return 0.0
    if re.search(r"\b(usd|eur|gbp|cny|rmb|yuan|euro|pound|vnd)\b|[€£¥₫]", text):
        return float("nan")
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    if len(nums) == 2 and re.search(r"\bto\b|[-–]", text):
        return sum(nums) / 2
    if len(nums) == 1:
        return nums[0]
    return float("nan")


def _fit_preprocessing(train_raw):
    """Fit all preprocessing parameters on train_raw (Train or Train+Val)."""
    df = _normalize_missing(train_raw.copy())
    df[EASE_COL] = pd.to_numeric(df[EASE_COL], errors="coerce").where(lambda s: s.between(0, 10))
    df[PRICE_COL] = df[PRICE_COL].apply(_parse_price_val).where(lambda s: s >= 0)
    params = {
        "imputation_numeric": {col: float(df[col].median()) for col in [EASE_COL, PRICE_COL]},
    }
    for col in TEXT_COLS:
        df[col] = df[col].fillna("")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col not in {"id", "label"}:
            df[col] = df[col].fillna("Unknown")
    params["imputation_likert"] = {col: float(df[col].map(LIKERT_MAP).median()) for col in LIKERT_COLS}
    params["multihot_vocab"] = {}
    for col, prefix in MULTI_COLS:
        choices = df[col].astype("string").str.replace(r"\s*,\s*", ",", regex=True)
        params["multihot_vocab"][prefix] = (
            choices.str.get_dummies(sep=",").drop(columns=["Unknown"], errors="ignore").columns.tolist()
        )
    params["tfidf_vectorizers"] = {}
    params["tfidf_col_prefixes"] = [f"text{i}" for i in range(len(TEXT_COLS))]
    for i, col in enumerate(TEXT_COLS):
        vec = TfidfVectorizer(max_features=50, stop_words="english")
        vec.fit(df[col].astype(str))
        params["tfidf_vectorizers"][col] = vec
    return params


def _apply_preprocessing(df_raw, params):
    """Apply fitted preprocessing params. Returns clean (unscaled) feature DataFrame + label."""
    df = _normalize_missing(df_raw.copy()).reset_index(drop=True)
    df[EASE_COL] = pd.to_numeric(df[EASE_COL], errors="coerce")
    df[EASE_COL] = df[EASE_COL].where(df[EASE_COL].between(0, 10))
    df[PRICE_COL] = df[PRICE_COL].apply(_parse_price_val)
    df[PRICE_COL] = df[PRICE_COL].where(df[PRICE_COL] >= 0)
    for col in [EASE_COL, PRICE_COL]:
        df[col + "_missing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(params["imputation_numeric"][col])
    for col in TEXT_COLS:
        df[col] = df[col].fillna("")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col not in {"id", "label"}:
            df[col] = df[col].fillna("Unknown")
    for col in LIKERT_COLS:
        df[col + "_missing"] = df[col].eq("Unknown").fillna(False).astype(int)
        df[col] = df[col].map(LIKERT_MAP).fillna(params["imputation_likert"][col])
    for col, prefix in MULTI_COLS:
        vocab = params["multihot_vocab"][prefix]
        choices = df[col].astype("string").str.replace(r"\s*,\s*", ",", regex=True)
        encoded = (
            choices.str.get_dummies(sep=",").drop(columns=["Unknown"], errors="ignore")
            .reindex(columns=vocab, fill_value=0).reset_index(drop=True)
        )
        df[prefix + "_missing"] = df[col].eq("Unknown").fillna(False).astype(int)
        df = pd.concat(
            [df.drop(columns=[col]).reset_index(drop=True), encoded.add_prefix(prefix + "_")],
            axis=1,
        )
    for i, col in enumerate(TEXT_COLS):
        vec = params["tfidf_vectorizers"][col]
        col_prefix = params["tfidf_col_prefixes"][i]
        names = [f"{col_prefix}_tfidf_{w}" for w in vec.get_feature_names_out()]
        matrix = vec.transform(df[col].astype(str)).toarray()
        feat_df = pd.DataFrame(matrix, columns=names).reset_index(drop=True)
        df = pd.concat([df.drop(columns=[col]).reset_index(drop=True), feat_df], axis=1)
    return df


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

    # --- Refit ALL preprocessing on Train+Val from raw CSV ---
    raw = pd.read_csv(RAW_CSV, dtype=str)
    unique_ids = raw["id"].unique()
    train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
    train_raw = raw[raw["id"].isin(train_ids)].reset_index(drop=True)
    val_raw   = raw[raw["id"].isin(val_ids)].reset_index(drop=True)
    test_raw  = raw[raw["id"].isin(test_ids)].reset_index(drop=True)
    tv_raw    = pd.concat([train_raw, val_raw], ignore_index=True)

    print(f"\nRefitting preprocessing on Train+Val ({len(tv_raw)} observations)...")
    pre_params = _fit_preprocessing(tv_raw)

    tv_df   = _apply_preprocessing(tv_raw, pre_params)
    test_df = _apply_preprocessing(test_raw, pre_params)

    feat_cols = [c for c in tv_df.columns if c not in {"id", "label"}]
    assert len(feat_cols) == 234, f"Expected 234 features, got {len(feat_cols)}"
    print(f"  Features: {len(feat_cols)}")

    # Fit StandardScaler on Train+Val clean features
    X_tv_clean   = tv_df[feat_cols].values.astype(np.float64)
    X_test_clean = test_df[feat_cols].values.astype(np.float64)

    scaler = StandardScaler()
    X_tv_scaled   = scaler.fit_transform(X_tv_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    # Load class metadata from tuning artifacts
    with open(TUNING_ART_DIR / "feature_names.json") as f:
        tuning_meta = json.load(f)
    classes = np.array(tuning_meta["classes"])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    y_tv   = tv_df["label"].map(class_to_idx).values.astype(np.int64)
    y_test_int    = test_df["label"].map(class_to_idx).values.astype(np.int64)
    y_test_labels = test_df["label"].values

    print(f"\nTrain+Val combined: {X_tv_scaled.shape[0]} observations × {X_tv_scaled.shape[1]} features")
    print(f"Classes            : {list(classes)}")

    # --- Save preprocessing artifacts to artifacts_final/preprocessing/ ---
    final_pre_dir = FINAL_ART_DIR / "preprocessing"
    final_pre_dir.mkdir(parents=True, exist_ok=True)

    # imputation_values.json
    imputation_values = {
        "numeric": {col: pre_params["imputation_numeric"][col] for col in [EASE_COL, PRICE_COL]},
        "likert":  pre_params["imputation_likert"],
    }
    with open(final_pre_dir / "imputation_values.json", "w") as f:
        json.dump(imputation_values, f, indent=2)

    # multihot_vocab.json
    with open(final_pre_dir / "multihot_vocab.json", "w") as f:
        json.dump(pre_params["multihot_vocab"], f, indent=2)

    # tfidf_params.json
    tfidf_params_out = {}
    for i, col in enumerate(TEXT_COLS):
        vec = pre_params["tfidf_vectorizers"][col]
        col_prefix = pre_params["tfidf_col_prefixes"][i]
        feature_names_out = [f"{col_prefix}_tfidf_{w}" for w in vec.get_feature_names_out()]
        vocab_dict = {word: int(idx) for idx, word in enumerate(vec.get_feature_names_out())}
        tfidf_params_out[col] = {
            "vocabulary":    vocab_dict,
            "idf_weights":   vec.idf_.tolist(),
            "feature_names": feature_names_out,
            "col_prefix":    col_prefix,
        }
    with open(final_pre_dir / "tfidf_params.json", "w") as f:
        json.dump(tfidf_params_out, f, indent=2)

    # scaler_params.npz
    np.savez(
        final_pre_dir / "scaler_params.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        var=scaler.var_,
        feature_names=np.array(feat_cols),
    )

    # preprocessing_manifest.json — update timestamp/note, keep structure
    manifest_path = final_pre_dir / "preprocessing_manifest.json"
    with open(TUNING_ART_DIR / "preprocessing" / "preprocessing_manifest.json") as f:
        manifest = json.load(f)
    manifest["fitted_on"] = "Train+Val (738 observations)"
    manifest["note"] = (
        "Preprocessing artifacts refit on Train+Val in retrain_final.py. "
        "TF-IDF uses index-based prefixes (text0–text3) to avoid duplicate column names."
    )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # feature_names.json — update with new TV-refit feature names
    feature_names_meta = {
        "feature_names": feat_cols,
        "classes": list(classes),
        "fitted_on": "Train+Val (738 observations)",
    }
    with open(FINAL_ART_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names_meta, f, indent=2)

    print(f"  Saved TV-refit preprocessing artifacts → {final_pre_dir}")

    # --- Train final model from scratch for exactly final_n_epochs epochs ---
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
    model.fit_fixed_epochs(X_tv_scaled, y_tv, n_epochs=final_n_epochs, classes=classes)
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
    meta["training_summary"]["n_epochs_fixed"]           = final_n_epochs
    meta["training_summary"]["tuning_best_epoch_0idx"]   = tuning_best_epoch
    meta["training_summary"]["train_val_n_observations"] = int(X_tv_scaled.shape[0])
    meta["training_summary"]["note"] = (
        "Retrained from scratch on Train+Val (738 obs) for exactly "
        f"{final_n_epochs} epochs (= tuning best_epoch+1). "
        "No early stopping. Preprocessing refit on Train+Val."
    )
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved final model → {FINAL_ART_DIR}")

    # --- Test evaluation (run once, final model only) ---
    print("\n" + "=" * 70)
    print("Test-Set Evaluation — Final Model")
    print("=" * 70)

    X_test = X_test_scaled
    y_test  = y_test_int
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
        f.write(f"Test dataset          : TV-refit test set ({X_test.shape[0]} obs)\n")
        f.write(f"Training data         : Train + Val ({X_tv_scaled.shape[0]} observations)\n")
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
        "dataset":                "TV-refit test set",
        "n_observations_test":    int(X_test.shape[0]),
        "n_observations_trainval": int(X_tv_scaled.shape[0]),
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
    print(f"  Final model  (Train+Val, {final_n_epochs} epochs, TV-refit preprocessing):")
    print(f"    Test accuracy : {test_acc:.4f}")
    print(f"    Test macro-F1 : {test_macro_f1:.4f}")
    print("=" * 70)
    print("Done.  Final model saved to artifacts_final/.")
    print("Preprocessing artifacts refit on Train+Val — predict.py will load from artifacts_final/.")

    return final_metrics


if __name__ == "__main__":
    main()
