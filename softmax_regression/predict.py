"""predict.py — Self-contained raw-survey-to-prediction inference pipeline.

Dependencies: Python standard library, NumPy, pandas only.
No scikit-learn required at inference time.

Usage (from project root):
    python3 softmax_regression/predict.py --input path/to/new_responses.csv

The input CSV must have the same column schema as the original training data
(csc311_project_training_202609.csv), i.e. survey question columns + an 'id'
column. The 'label' column is not required for inference.

Outputs:
    - Predicted class labels (Baseball / Basketball / Soccer) for each row.
    - Predicted probabilities for each class.

All preprocessing parameters are loaded from artifacts/preprocessing/
(imputation medians, multi-hot vocabularies, TF-IDF vocab + IDF weights,
StandardScaler mean/scale). These were fitted on the training set only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Paths
SCRIPT_DIR   = Path(__file__).resolve().parent
ARTIFACT_DIR = SCRIPT_DIR / "artifacts_final"
PRE_DIR      = ARTIFACT_DIR / "preprocessing"


# Minimal TF-IDF transform (no sklearn needed at inference time)
def _tokenize(text: str, stop_words: set) -> list[str]:
    """Tokenize one document: lowercase, keep alpha tokens, remove stopwords"""
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if t not in stop_words and len(t) > 1]


# Standard sklearn English stop words (subset sufficient for these texts).
# Using the full sklearn list ensures exact match with training TF-IDF.
_SKLEARN_STOP_WORDS: Optional[set] = None


def _get_stop_words() -> set:
    global _SKLEARN_STOP_WORDS
    if _SKLEARN_STOP_WORDS is not None:
        return _SKLEARN_STOP_WORDS
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        _SKLEARN_STOP_WORDS = set(ENGLISH_STOP_WORDS)
    except ImportError:
        # Fallback: a small common-word set (less accurate but functional).
        _SKLEARN_STOP_WORDS = {
            "i","me","my","we","our","you","he","she","it","they","them",
            "is","are","was","were","be","been","being","have","has","had",
            "do","does","did","will","would","could","should","may","might",
            "the","a","an","and","or","but","in","on","at","to","for","of",
            "with","by","from","up","about","into","through","than","this",
            "that","these","those","not","no","nor","so","yet","both","either",
        }
    return _SKLEARN_STOP_WORDS


def _tfidf_transform_numpy(texts: list[str], vocabulary: dict, idf: np.ndarray) -> np.ndarray:
    """Pure-NumPy TF-IDF transform matching sklearn's default settings.

    sklearn defaults: analyzer='word', token_pattern=r'(?u)\\b\\w\\w+\\b',
    norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False.
    """
    stop_words = _get_stop_words()
    N = len(texts)
    V = len(vocabulary)
    out = np.zeros((N, V), dtype=np.float64)

    for i, text in enumerate(texts):
        tokens = _tokenize(str(text), stop_words)
        if not tokens:
            continue
        # TF (raw counts)
        tf = np.zeros(V, dtype=np.float64)
        for tok in tokens:
            if tok in vocabulary:
                tf[vocabulary[tok]] += 1.0
        # TF × IDF
        tfidf_vec = tf * idf
        # L2 normalise
        norm = np.sqrt((tfidf_vec ** 2).sum())
        if norm > 0:
            tfidf_vec /= norm
        out[i] = tfidf_vec

    return out


# Preprocessing pipeline (artifact-driven, no sklearn fitting)
class SurveyPreprocessor:
    """Reconstruct the 234-feature representation from raw survey responses.

    All parameters are loaded from saved artifacts — no training data needed.
    """

    def __init__(self, artifact_dir: Optional[Path] = None) -> None:
        pre_dir = (artifact_dir or ARTIFACT_DIR) / "preprocessing"

        with open(pre_dir / "preprocessing_manifest.json") as f:
            m = json.load(f)
        with open(pre_dir / "imputation_values.json") as f:
            self._imputation = json.load(f)
        with open(pre_dir / "multihot_vocab.json") as f:
            self._multihot_vocab = json.load(f)   # {prefix: [cat1, cat2, ...]}
        with open(pre_dir / "tfidf_params.json") as f:
            self._tfidf_params = json.load(f)     # {col: {vocabulary, idf_weights, ...}}

        scaler = np.load(pre_dir / "scaler_params.npz", allow_pickle=True)
        self._scaler_mean  = scaler["mean"]
        self._scaler_scale = scaler["scale"]
        self._scaler_feat_names = list(scaler["feature_names"])

        self._missing_markers = set(m["missing_markers"])
        self._ease_col   = m["ease_col"]
        self._price_col  = m["price_col"]
        self._likert_cols = m["likert_cols"]
        self._likert_map  = m["likert_map"]
        self._text_cols   = m["text_cols"]
        self._multi_cols  = m["multi_cols"]   # {survey_col: prefix}


    # Public interface
    def transform(self, raw_df: pd.DataFrame) -> np.ndarray:
        """Transform raw survey DataFrame to (N, 234) standardised feature matrix.

        Parameters
        raw_df : DataFrame with survey question columns (and optionally 'id', 'label').

        Returns
        X : (N, 234) float64 array, standardised, ready for model.predict().
        """
        data = self._normalize_missing(raw_df.copy()).reset_index(drop=True)
        data = self._parse_numeric(data)
        data = self._impute_text(data)
        data = self._encode_likert(data)
        data = self._encode_multihot(data)
        data = self._encode_tfidf(data)
        X = self._extract_and_scale(data)
        return X

    # Pipeline steps
    def _normalize_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in data.select_dtypes(include="object").columns:
            if col in {"id", "label"}:
                continue
            s = data[col].astype("string").str.strip()
            data[col] = s.mask(s.str.lower().isin(self._missing_markers))
        return data

    def _parse_numeric(self, data: pd.DataFrame) -> pd.DataFrame:
        # Ease-of-play
        data[self._ease_col] = pd.to_numeric(data[self._ease_col], errors="coerce")
        data[self._ease_col] = data[self._ease_col].where(
            data[self._ease_col].between(0, 10)
        )
        # Ticket price
        data[self._price_col] = data[self._price_col].apply(_parse_price)
        data[self._price_col] = data[self._price_col].where(
            data[self._price_col] >= 0
        )
        # Missingness indicators + median imputation
        for col in [self._ease_col, self._price_col]:
            data[col + "_missing"] = data[col].isna().astype(int)
            data[col] = data[col].fillna(self._imputation["numeric"][col])
        return data

    def _impute_text(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in self._text_cols:
            data[col] = data[col].fillna("")
        data.fillna("Unknown", inplace=True)
        return data

    def _encode_likert(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in self._likert_cols:
            data[col + "_missing"] = data[col].eq("Unknown").fillna(False).astype(int)
            data[col] = data[col].map(self._likert_map)
            data[col] = data[col].fillna(self._imputation["likert"][col])
        return data

    def _encode_multihot(self, data: pd.DataFrame) -> pd.DataFrame:
        for survey_col, prefix in self._multi_cols.items():
            vocab = self._multihot_vocab[prefix]
            choices = (
                data[survey_col].astype("string")
                .str.replace(r"\s*,\s*", ",", regex=True)
            )
            raw_dummies = choices.str.get_dummies(sep=",").drop(
                columns=["Unknown"], errors="ignore"
            )
            encoded = raw_dummies.reindex(columns=vocab, fill_value=0)
            data[prefix + "_missing"] = (
                data[survey_col].eq("Unknown").fillna(False).astype(int)
            )
            data = pd.concat(
                [
                    data.drop(columns=survey_col).reset_index(drop=True),
                    encoded.add_prefix(prefix + "_").reset_index(drop=True),
                ],
                axis=1,
            )
        return data

    def _encode_tfidf(self, data: pd.DataFrame) -> pd.DataFrame:
        for text_col in self._text_cols:
            params = self._tfidf_params[text_col]
            vocab  = params["vocabulary"]
            idf    = np.array(params["idf_weights"])
            feat_names = params["feature_names"]

            texts  = data[text_col].astype(str).tolist()
            matrix = _tfidf_transform_numpy(texts, vocab, idf)

            feat_df = pd.DataFrame(matrix, columns=feat_names)
            data = pd.concat(
                [
                    data.drop(columns=text_col).reset_index(drop=True),
                    feat_df.reset_index(drop=True),
                ],
                axis=1,
            )
        return data

    def _extract_and_scale(self, data: pd.DataFrame) -> np.ndarray:
        X = data.drop(columns=["id", "label"], errors="ignore").values.astype(np.float64)
        if X.shape[1] != len(self._scaler_mean):
            raise ValueError(
                f"Feature count mismatch: got {X.shape[1]}, expected {len(self._scaler_mean)}. "
                "Ensure the input CSV has the same columns as the training data."
            )
        return (X - self._scaler_mean) / self._scaler_scale


# Price parser (duplicated from data_explored.py logic, no import needed)
def _parse_price(value) -> float:
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


# Minimal Softmax predictor (NumPy only — no sklearn)
class SoftmaxPredictor:
    """Load saved model weights and run predictions.

    Uses only NumPy — no scikit-learn or PyTorch needed at inference time.
    """

    def __init__(self, artifact_dir: Optional[Path] = None) -> None:
        art_dir = artifact_dir or ARTIFACT_DIR
        data = np.load(art_dir / "model_weights.npz", allow_pickle=True)
        self.W       = data["W"]        # (234, 3)
        self.b       = data["b"]        # (3,)
        self.classes = data["classes"]  # ['Baseball', 'Basketball', 'Soccer']

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (N, 3) class probability matrix."""
        Z = X @ self.W + self.b
        Z -= Z.max(axis=1, keepdims=True)   # numerical stability
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        return P

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return (N,) string class labels."""
        return self.classes[self.predict_proba(X).argmax(axis=1)]


# End-to-end inference function
def predict_from_raw_csv(
    input_csv: str | Path,
    artifact_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Run the full inference pipeline on a raw survey CSV.

    Parameters
    input_csv    : Path to raw survey CSV (same schema as training data).
    artifact_dir : Optional override for artifact directory.

    Returns
    DataFrame with columns: id (if present), predicted_label,
    prob_Baseball, prob_Basketball, prob_Soccer.
    """
    df = pd.read_csv(input_csv)
    art_dir = artifact_dir or ARTIFACT_DIR

    preprocessor = SurveyPreprocessor(art_dir)
    predictor    = SoftmaxPredictor(art_dir)

    X     = preprocessor.transform(df)
    proba = predictor.predict_proba(X)
    preds = predictor.predict(X)

    result = pd.DataFrame(proba, columns=[f"prob_{c}" for c in predictor.classes])
    result.insert(0, "predicted_label", preds)
    if "id" in df.columns:
        result.insert(0, "id", df["id"].values)
    return result


# CLI entry point
def main():
    parser = argparse.ArgumentParser(
        description="Predict sport class from raw survey CSV."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV.")
    parser.add_argument("--output", default=None,
                        help="Optional path to save predictions CSV.")
    args = parser.parse_args()

    result = predict_from_raw_csv(args.input)
    print(result.to_string(index=False))

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nSaved predictions → {args.output}")


if __name__ == "__main__":
    main()
