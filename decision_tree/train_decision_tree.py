"""train_decision_tree.py - Decision tree training, tuning and final evaluation.

Usage (from the project root):
    python3 decision_tree/train_decision_tree.py

Reads the un-scaled processed data (trees do not need scaling):
    processed_data/train_clean.csv, val_clean.csv, test_clean.csv

Steps:
    1. Grid search over max_depth and min_samples_leaf on the validation set.
    2. Retrain the selected configuration on train + validation.
    3. Evaluate once on the internal test set.

Outputs:
    decision_tree/results/validation_grid.csv
    decision_tree/results/best_config.json
    decision_tree/results/test_metrics.json
    decision_tree/results/test_classification_report.txt
    decision_tree/results/test_confusion_matrix.png
    decision_tree/results/tree_top_levels.png
    decision_tree/artifacts/decision_tree.joblib
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "processed_data"
RESULTS_DIR = PROJECT_ROOT / "decision_tree" / "results"
ARTIFACT_DIR = PROJECT_ROOT / "decision_tree" / "artifacts"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_SPACE = {
    "max_depth": [2, 3, 4, 5, 7, 10, None],
    "min_samples_leaf": [1, 2, 5, 10],
}


def load_split(name: str):
    data = pd.read_csv(DATA_DIR / f"{name}_clean.csv")
    return data.drop(columns=["label"]), data["label"]


def short_name(column: str) -> str:
    """Shorten long survey column names for plotting."""
    if "How easy do you think" in column:
        return "Ease"
    if "How much would you be willing to pay" in column:
        return "Ticket price"
    if "Success in this sport" in column:
        return "Strategy"
    if "Players must be comfortable" in column:
        return "Injury risk"
    if "Individual brilliance" in column:
        return "Individual skill"
    if "_tfidf_" in column:
        return column.split("_tfidf_")[-1] + " (text)"
    return column[:30]


def main() -> None:
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    assert list(X_train.columns) == list(X_val.columns) == list(X_test.columns)

    # 1. Grid search on validation (the test set is not touched here).
    rows = []
    for depth, leaf in product(SEARCH_SPACE["max_depth"], SEARCH_SPACE["min_samples_leaf"]):
        tree = DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=leaf, random_state=SEED
        )
        tree.fit(X_train, y_train)
        val_pred = tree.predict(X_val)
        rows.append({
            "max_depth": "unlimited" if depth is None else depth,
            "min_samples_leaf": leaf,
            "n_leaves": tree.get_n_leaves(),
            "train_accuracy": accuracy_score(y_train, tree.predict(X_train)),
            "val_accuracy": accuracy_score(y_val, val_pred),
            "val_macro_f1": f1_score(y_val, val_pred, average="macro"),
        })
    grid = pd.DataFrame(rows).sort_values(
        ["val_macro_f1", "val_accuracy", "n_leaves"],
        ascending=[False, False, True],
    )
    grid.to_csv(RESULTS_DIR / "validation_grid.csv", index=False)
    print("\nTop 5 configurations by validation macro-F1:")
    print(grid.head(5).round(4).to_string(index=False))

    # Ties are broken by validation accuracy, then by the smaller tree.
    best = grid.iloc[0]
    best_depth = None if best["max_depth"] == "unlimited" else int(best["max_depth"])
    best_leaf = int(best["min_samples_leaf"])
    best_config = {
        "criterion": "gini",
        "max_depth": best_depth,
        "min_samples_leaf": best_leaf,
        "random_state": SEED,
        "val_accuracy": float(best["val_accuracy"]),
        "val_macro_f1": float(best["val_macro_f1"]),
    }
    with open(RESULTS_DIR / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\nSelected: max_depth={best_depth}, min_samples_leaf={best_leaf}")

    # 2. Retrain the selected configuration on train + validation.
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    final_tree = DecisionTreeClassifier(
        max_depth=best_depth, min_samples_leaf=best_leaf, random_state=SEED
    )
    final_tree.fit(X_trainval, y_trainval)
    joblib.dump(
        {"model": final_tree, "feature_names": list(X_train.columns)},
        ARTIFACT_DIR / "decision_tree.joblib",
    )

    # 3. Final evaluation on the internal test set (run once).
    test_pred = final_tree.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_macro_f1 = f1_score(y_test, test_pred, average="macro")
    report = classification_report(y_test, test_pred, digits=4, zero_division=0)

    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test macro-F1: {test_macro_f1:.4f}")
    print(report)

    with open(RESULTS_DIR / "test_metrics.json", "w") as f:
        json.dump(
            {
                "n_observations": int(len(y_test)),
                "test_accuracy": float(test_acc),
                "test_macro_f1": float(test_macro_f1),
                "config": best_config,
            },
            f,
            indent=2,
        )
    with open(RESULTS_DIR / "test_classification_report.txt", "w") as f:
        f.write("Decision Tree - Internal Test-Set Evaluation\n\n")
        f.write(f"Test accuracy : {test_acc:.4f}\n")
        f.write(f"Test macro-F1 : {test_macro_f1:.4f}\n\n")
        f.write(report)

    ConfusionMatrixDisplay.from_predictions(
        y_test, test_pred, labels=["Baseball", "Basketball", "Soccer"]
    )
    plt.title("Decision tree - internal test set")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "test_confusion_matrix.png", dpi=150)
    plt.close()

    plt.figure(figsize=(24, 12))
    plot_tree(
        final_tree,
        feature_names=[short_name(c) for c in X_train.columns],
        class_names=[str(c) for c in final_tree.classes_],
        max_depth=3,
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.savefig(RESULTS_DIR / "tree_top_levels.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved results to {RESULTS_DIR} and model to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
