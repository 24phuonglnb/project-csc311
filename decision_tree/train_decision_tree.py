"""train_decision_tree.py - Decision tree training, tuning and final evaluation.

Usage (from the project root):
    python3 decision_tree/train_decision_tree.py

Tuning phase: reads un-scaled processed data (train-only preprocessing):
    processed_data/train_clean.csv, val_clean.csv, test_clean.csv

Final retraining: refits ALL preprocessing on Train+Val (738 obs) from raw CSV,
then retrains the selected tree and evaluates on TV-refit test features.

Steps:
    1. Grid search over max_depth and min_samples_leaf on the validation set.
    2. Refit preprocessing on Train+Val, retrain the selected tree.
    3. Evaluate once on the internal test set (TV-refit features).

Outputs:
    decision_tree/results/validation_grid.csv
    decision_tree/results/best_config.json
    decision_tree/results/test_metrics.json
    decision_tree/results/test_classification_report.txt
    decision_tree/results/test_confusion_matrix.png
    decision_tree/results/tree_top_levels.png
    decision_tree/results/test_clean_final.csv
    decision_tree/artifacts/decision_tree.joblib
"""

from __future__ import annotations

import json
import re
from itertools import product
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
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
    """Apply fitted preprocessing params to a raw DataFrame. Returns clean feature DataFrame + label."""
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

    # 2. Refit preprocessing on Train + Validation, then retrain the selected tree.
    raw = pd.read_csv(RAW_CSV, dtype=str)
    unique_ids = raw["id"].unique()
    train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, random_state=SEED)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=SEED)
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

    X_trainval = tv_df[feat_cols].values
    y_trainval = tv_df["label"].values
    X_test_tv  = test_df[feat_cols].values
    y_test_tv  = test_df["label"].values

    final_tree = DecisionTreeClassifier(
        max_depth=best_depth, min_samples_leaf=best_leaf, random_state=SEED
    )
    final_tree.fit(X_trainval, y_trainval)
    joblib.dump(
        {"model": final_tree, "feature_names": feat_cols},
        ARTIFACT_DIR / "decision_tree.joblib",
    )

    # Save TV-refit test features for audit and reproducibility.
    test_out = test_df[feat_cols].copy()
    test_out["label"] = y_test_tv
    test_out.to_csv(RESULTS_DIR / "test_clean_final.csv", index=False)
    print(f"  Saved TV-refit test features → {RESULTS_DIR}/test_clean_final.csv")

    # 3. Final evaluation on the internal test set (run once).
    test_pred     = final_tree.predict(X_test_tv)
    test_acc      = accuracy_score(y_test_tv, test_pred)
    test_macro_f1 = f1_score(y_test_tv, test_pred, average="macro")
    report        = classification_report(y_test_tv, test_pred, digits=4, zero_division=0)

    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test macro-F1: {test_macro_f1:.4f}")
    print(report)

    with open(RESULTS_DIR / "test_metrics.json", "w") as f:
        json.dump(
            {
                "n_observations": int(len(y_test_tv)),
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
        y_test_tv, test_pred, labels=["Baseball", "Basketball", "Soccer"]
    )
    plt.title("Decision tree - internal test set")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "test_confusion_matrix.png", dpi=150)
    plt.close()

    plt.figure(figsize=(24, 12))
    plot_tree(
        final_tree,
        feature_names=[short_name(c) for c in feat_cols],
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
