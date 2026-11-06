"""Export the training-fitted preprocessing parameters used by pred.py.

Re-creates the respondent split from data_explored.py (seed 42), fits the same
medians, multi-hot vocabularies, and TF-IDF vocabularies/IDF weights on the
training rows only, writes mlp_outputs/mlp_preprocessing.json, and checks that
pred.build_features reproduces processed_data/{train,val,test}_clean.csv.

    python export_preprocessing.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import pred

ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "csc311_project_training_202609.csv"
DATA_DIR = ROOT / "processed_data"
OUTPUT = ROOT / "mlp_outputs" / pred.PREPROCESSING_FILE
SPLIT_SEED = 42  # data_explored.py SEED


def split_raw(raw):
    ids = raw["id"].unique()
    train_ids, temp_ids = train_test_split(ids, test_size=0.3, random_state=SPLIT_SEED)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=SPLIT_SEED)
    return {name: raw[raw["id"].isin(group)].reset_index(drop=True)
            for name, group in (("train", train_ids), ("val", val_ids),
                                ("test", test_ids))}


def fit_params(train, feature_names):
    norm = {column: pred.normalize_missing(train[column]) for column in train.columns}
    params = {
        "ease_median": float(pred.parse_ease(norm[pred.EASE_COL]).median()),
        "price_median": float(pred.parse_prices(norm[pred.PRICE_COL]).median()),
        "likert_medians": [float(pred.parse_likert(norm[column]).median())
                           for column in pred.LIKERT_COLS],
        "multi_options": {},
        "tfidf": [],
        "feature_names": feature_names,
    }
    for column, prefix in pred.MULTI_COLS:
        options = norm[column].dropna().str.replace(r'\s*,\s*', ',', regex=True)
        options = options.str.get_dummies(sep=',').columns
        params["multi_options"][prefix] = [o for o in options if o != 'Unknown']
    for column in pred.TEXT_COLS:
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        vectorizer.fit(norm[column].fillna('').astype(str))
        params["tfidf"].append({
            "column": column,
            "vocabulary": vectorizer.get_feature_names_out().tolist(),
            "idf": vectorizer.idf_.tolist(),
        })
    return params


def main():
    raw = pd.read_csv(RAW_FILE, dtype=str)
    splits = split_raw(raw)
    clean = {name: pd.read_csv(DATA_DIR / f"{name}_clean.csv") for name in splits}
    feature_names = [c for c in clean["train"].columns if c != "label"]
    params = fit_params(splits["train"], feature_names)

    for name, frame in splits.items():
        expected = clean[name].drop(columns="label").to_numpy(dtype=np.float64)
        produced = pred.build_features(frame, params)
        if produced.shape != expected.shape:
            raise AssertionError(f"{name}: shape {produced.shape} != {expected.shape}")
        max_error = float(np.abs(produced - expected).max())
        labels_match = (frame["label"].to_numpy() == clean[name]["label"].to_numpy()).all()
        if max_error > 1e-9 or not labels_match:
            raise AssertionError(f"{name}: max |diff| {max_error:.2e}, "
                                 f"labels match={labels_match}")
        print(f"{name}: {len(frame)} rows reproduced (max |diff| {max_error:.1e})")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(params, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
