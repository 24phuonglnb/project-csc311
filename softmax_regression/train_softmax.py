"""train_softmax.py — Random-search hyperparameter tuning for Softmax Regression.

Usage (from project root):
    python3 softmax_regression/train_softmax.py

Outputs (all in softmax_regression/results/ and softmax_regression/artifacts/):
    - random_search_results.csv : all 50 configurations with metrics
    - best_config.json          : best hyperparameter configuration
    - artifacts/model_weights.npz, artifacts/model_meta.json : best model weights
    - artifacts/feature_names.json : ordered list of feature column names
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Allow running from project root or from softmax_regression/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "softmax_regression"))

from softmax_model import SoftmaxRegression, cross_entropy_loss


# Configuration
SEED = 42
N_CONFIGS = 50
DATA_DIR = PROJECT_ROOT / "processed_data"
RESULTS_DIR = PROJECT_ROOT / "softmax_regression" / "results"
ARTIFACT_DIR = PROJECT_ROOT / "softmax_regression" / "artifacts"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# Hyperparameter search space (from project proposal)
SEARCH_SPACE = {
    "learning_rate": [0.001, 0.01, 0.1, 0.2],
    "batch_size":    [32, 64, 128, 256],
    "max_epochs":    [100, 250, 500, 1000],
    "l2_lambda":     [0.001, 0.01, 0.1, 1.0],
    "patience":      [1, 2, 5, 10],
}


# Data loading
def load_data():
    """Load train/val scaled CSVs and return numpy arrays + class info."""
    train = pd.read_csv(DATA_DIR / "train_scaled.csv")
    val   = pd.read_csv(DATA_DIR / "val_scaled.csv")

    feature_cols = [c for c in train.columns if c != "label"]
    classes = np.array(sorted(train["label"].unique()))  # Alphabetical: Baseball, Basketball, Soccer
    class_to_idx = {c: i for i, c in enumerate(classes)}

    X_train = train[feature_cols].values.astype(np.float64)
    y_train = train["label"].map(class_to_idx).values.astype(np.int64)

    X_val = val[feature_cols].values.astype(np.float64)
    y_val = val["label"].map(class_to_idx).values.astype(np.int64)

    # Save feature names for portability
    with open(ARTIFACT_DIR / "feature_names.json", "w") as f:
        json.dump({"feature_names": feature_cols, "classes": list(classes)}, f, indent=2)

    return X_train, y_train, X_val, y_val, classes, class_to_idx, feature_cols


# Random search
def sample_configs(n: int, seed: int) -> list[dict]:
    """Sample n distinct hyperparameter configurations using random search"""
    rng = np.random.default_rng(seed)
    configs = []
    seen = set()

    attempts = 0
    while len(configs) < n and attempts < n * 100:
        attempts += 1
        cfg = {
            k: rng.choice(v).item()
            for k, v in SEARCH_SPACE.items()
        }
        key = tuple(cfg[k] for k in sorted(cfg))
        if key not in seen:
            seen.add(key)
            configs.append(cfg)

    return configs


def evaluate_config(
    cfg: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    classes: np.ndarray,
    config_idx: int,
) -> dict:
    """Train one configuration and return a result dictionary."""
    model = SoftmaxRegression(
        n_classes=len(classes),
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        max_epochs=cfg["max_epochs"],
        l2_lambda=cfg["l2_lambda"],
        patience=cfg["patience"],
        random_state=SEED,
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train, X_val, y_val, classes=classes)
    elapsed = time.perf_counter() - t0

    # Validation metrics at best epoch
    y_pred_int = model.predict(X_val)
    y_pred_labels = classes[y_pred_int]
    y_val_labels  = classes[y_val]

    val_acc    = accuracy_score(y_val_labels, y_pred_labels)
    val_macro_f1 = f1_score(y_val_labels, y_pred_labels, average="macro")
    val_loss = min(model.val_losses_)

    result = {
        "config_idx":        config_idx,
        "learning_rate":     cfg["learning_rate"],
        "batch_size":        cfg["batch_size"],
        "max_epochs":        cfg["max_epochs"],
        "l2_lambda":         cfg["l2_lambda"],
        "patience":          cfg["patience"],
        "best_epoch":        model.best_epoch_,
        "actual_stop_epoch": len(model.train_losses_) - 1,
        "best_train_loss":   min(model.train_losses_),
        "best_val_loss":     val_loss,
        "val_accuracy":      val_acc,
        "val_macro_f1":      val_macro_f1,
        "runtime_s":         elapsed,
    }
    return result, model


def main():
    print("=" * 70)
    print("Softmax Regression — Random Hyperparameter Search")
    print("=" * 70)

    X_train, y_train, X_val, y_val, classes, class_to_idx, feature_cols = load_data()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Classes: {list(classes)}")
    print(f"n_features: {X_train.shape[1]}")

    configs = sample_configs(N_CONFIGS, seed=SEED)
    print(f"\nSampled {len(configs)} distinct configurations (seed={SEED}).\n")

    results = []
    best_macro_f1 = -1.0
    best_model = None
    best_result = None
    total_start = time.perf_counter()

    for i, cfg in enumerate(configs):
        print(
            f"[{i+1:02d}/{N_CONFIGS}] lr={cfg['learning_rate']:.3f} "
            f"bs={cfg['batch_size']} ep={cfg['max_epochs']} "
            f"l2={cfg['l2_lambda']:.3f} pat={cfg['patience']} ",
            end="",
            flush=True,
        )
        result, model = evaluate_config(cfg, X_train, y_train, X_val, y_val, classes, i + 1)
        results.append(result)

        print(
            f"val_f1={result['val_macro_f1']:.4f}  "
            f"val_acc={result['val_accuracy']:.4f}  "
            f"stop@ep{result['actual_stop_epoch']}  "
            f"({result['runtime_s']:.1f}s)"
        )

        # Select best by macro-F1, tie-break by val_loss (lower is better)
        if (
            result["val_macro_f1"] > best_macro_f1
            or (
                result["val_macro_f1"] == best_macro_f1
                and result["best_val_loss"] < (best_result["best_val_loss"] if best_result else np.inf)
            )
        ):
            best_macro_f1 = result["val_macro_f1"]
            best_model = model
            best_result = result

    total_elapsed = time.perf_counter() - total_start
    print(f"\nTotal search time: {total_elapsed:.1f}s")

    # Save results CSV
    results_df = pd.DataFrame(results).sort_values("val_macro_f1", ascending=False)
    csv_path = RESULTS_DIR / "random_search_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved search results → {csv_path}")

    # Save best config JSON
    best_cfg_path = RESULTS_DIR / "best_config.json"
    with open(best_cfg_path, "w") as f:
        json.dump(best_result, f, indent=2)
    print(f"Saved best config     → {best_cfg_path}")

    # Save best model weights and metadata
    best_model.save(ARTIFACT_DIR)
    print(f"Saved best model      → {ARTIFACT_DIR}")

    # Print summary
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    for k, v in best_result.items():
        print(f"  {k:<25}: {v}")

    return best_result, results_df


if __name__ == "__main__":
    main()
