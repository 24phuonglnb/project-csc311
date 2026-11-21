"""CSC311 sport prediction: predict_all(filename) using only numpy and pandas.

Reproduces the preprocessing in data_explored.py (training-fitted medians,
multi-hot vocabularies, TF-IDF vocabularies and IDF weights) from
mlp_preprocessing.json, then runs the exported MLP from mlp_model.npz.
Both files are looked up next to this script, then in ./mlp_outputs/.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_FILE = "mlp_model.npz"
PREPROCESSING_FILE = "mlp_preprocessing.json"

EASE_COL = ('How easy do you think it would be for someone with no prior experience '
            'to start playing this sport recreationally? Rate it on a scale from 0 to 10, '
            'where 0 means extremely difficult and 10 means extremely easy.')
PRICE_COL = ('How much would you be willing to pay for a decent seat '
             'at a professional competition of this sport?')
LIKERT_COLS = [
    'Success in this sport depends heavily on strategy and planning.',
    'Players must be comfortable with a high risk of injury.',
    'Individual brilliance can outweigh teamwork in this sport.',
]
MULTI_COLS = [
    ('Where might you play this sport?', 'location'),
    ('During which seasons would you be most likely to play this sport '
     'recreationally in Toronto?', 'season'),
    ('What types of music do you most associate with this sport?', 'music'),
]
TEXT_COLS = [
    'Which song do you associate with this sport?',
    'What foods and drinks do you associate with watching this sport in the '
    'place or culture you know best?',
    'What colour do you associate with this sport?',
    'How do you feel when you watch this sport?',
]
LIKERT_MAP = {'Strongly disagree': -2, 'Disagree': -1, 'Neutral': 0,
              'Agree': 1, 'Strongly agree': 2}
MISSING_MARKERS = {'', 'n/a', 'na', 'null', 'none', 'nan', '-', 'unknown'}
TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")  # scikit-learn TfidfVectorizer default


def _find(name):
    for directory in (HERE, HERE / "mlp_outputs", HERE / "mlp" / "mlp_outputs"):
        if (directory / name).exists():
            return directory / name
    raise FileNotFoundError(f"{name} not found next to pred.py, in mlp_outputs/, or in mlp/mlp_outputs/.")

def normalize_missing(series):
    """Strip text and map heterogeneous missing markers to NaN."""
    def clean(value):
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        return np.nan if text.lower() in MISSING_MARKERS else text
    return series.map(clean)


def parse_price(value):
    """Parse simple monetary numbers/ranges; ambiguous answers become NaN."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {'free', 'nothing', 'zero', 'unwilling', '0', '$0'}:
        return 0.0
    if re.search(r'\b(usd|eur|gbp|cny|rmb|yuan|euro|pound|vnd)\b|[€£¥₫]', text):
        return np.nan
    text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)
    numbers = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', text)]
    if len(numbers) == 2 and re.search(r'\bto\b|[-–]', text):
        return sum(numbers) / 2
    if len(numbers) == 1:
        return numbers[0]
    return np.nan


def parse_ease(series):
    values = pd.to_numeric(series, errors='coerce').astype(float)
    return values.where(values.between(0, 10))


def parse_prices(series):
    values = series.map(parse_price).astype(float)
    return values.where(values >= 0)


def parse_likert(series):
    return series.map(LIKERT_MAP).astype(float)


def multi_hot(series, options):
    """Binary indicator per training option; matches str.get_dummies(sep=',')."""
    chosen = [
        set() if pd.isna(value) else set(re.sub(r'\s*,\s*', ',', value).split(','))
        for value in series
    ]
    return np.array([[float(option in row) for option in options] for row in chosen],
                    dtype=np.float64).reshape(len(series), len(options))


def tfidf(series, vocabulary, idf):
    """TF-IDF with a fixed vocabulary, raw counts, smooth IDF, and L2 norm."""
    index = {term: position for position, term in enumerate(vocabulary)}
    counts = np.zeros((len(series), len(vocabulary)), dtype=np.float64)
    for row, value in enumerate(series):
        if pd.isna(value):
            continue
        for token in TOKEN_PATTERN.findall(str(value).lower()):
            position = index.get(token)
            if position is not None:
                counts[row, position] += 1
    weighted = counts * np.asarray(idf, dtype=np.float64)
    norms = np.sqrt((weighted ** 2).sum(axis=1, keepdims=True))
    return np.divide(weighted, norms, out=np.zeros_like(weighted), where=norms > 0)


def build_features(frame, params):
    """Raw survey DataFrame -> (n_rows, 234) array in the *_clean.csv order."""
    frame = frame.rename(columns=lambda name: str(name).strip())
    required = [EASE_COL, PRICE_COL, *LIKERT_COLS,
                *(column for column, _ in MULTI_COLS), *TEXT_COLS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")
    data = {column: normalize_missing(frame[column]) for column in required}

    ease = parse_ease(data[EASE_COL])
    price = parse_prices(data[PRICE_COL])
    likert = [parse_likert(data[column]) for column in LIKERT_COLS]

    blocks = [
        ease.fillna(params["ease_median"]).to_numpy()[:, None],
        *(values.fillna(median).to_numpy()[:, None]
          for values, median in zip(likert, params["likert_medians"])),
        price.fillna(params["price_median"]).to_numpy()[:, None],
        ease.isna().to_numpy(dtype=np.float64)[:, None],
        price.isna().to_numpy(dtype=np.float64)[:, None],
        *(data[column].isna().to_numpy(dtype=np.float64)[:, None]
          for column in LIKERT_COLS),
    ]
    for column, prefix in MULTI_COLS:
        blocks.append(data[column].isna().to_numpy(dtype=np.float64)[:, None])
        blocks.append(multi_hot(data[column], params["multi_options"][prefix]))
    for column, text in zip(TEXT_COLS, params["tfidf"]):
        blocks.append(tfidf(data[column], text["vocabulary"], text["idf"]))

    features = np.hstack(blocks)
    if features.shape[1] != len(params["feature_names"]):
        raise ValueError("Preprocessing produced the wrong number of features.")
    return features


def _activate(values, name):
    if name == "relu":
        return np.maximum(values, 0.0)
    if name == "leaky_relu":
        return np.where(values >= 0, values, 0.01 * values)  # torch default slope
    if name == "tanh":
        return np.tanh(values)
    raise ValueError(f"Unsupported activation: {name!r}")


def add_cluster_features(values, params):
    """Append one-hot nearest-centroid k-means IDs for each TF-IDF block."""
    k = int(params["kmeans_k"])
    blocks = [np.asarray(values, dtype=np.float64)]
    for index in range(int(params["kmeans_n_blocks"]) if k else 0):
        block = blocks[0][:, params[f"kmeans_columns_{index}"]]
        centroids = params[f"kmeans_centroids_{index}"]
        distances = ((block[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        blocks.append(np.eye(k)[distances.argmin(axis=1)])
    return np.hstack(blocks)


def transform_features(values, params):
    """Clean features -> [k-means one-hots] -> standardize -> [PCA]."""
    values = add_cluster_features(values, params)
    if values.shape[1] != params["scaler_mean"].shape[0]:
        raise ValueError("Feature matrix has the wrong shape for this model.")
    values = (values - params["scaler_mean"]) / params["scaler_scale"]
    if int(params["pca_n_components"]):
        values = (values - params["pca_mean"]) @ params["pca_components"].T
    return values


def predict_proba_features(features, model_path=None):
    """Class probabilities from unscaled clean features (n_rows, 234)."""
    with np.load(model_path or _find(MODEL_FILE), allow_pickle=False) as model:
        values = transform_features(features, model)
        activation = str(model["activation"])
        n_layers = int(model["n_layers"])
        for layer in range(n_layers):
            values = values @ model[f"weight_{layer}"].T + model[f"bias_{layer}"]
            if layer < n_layers - 1:
                values = _activate(values, activation)
    values = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(values)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def predict_all(filename):
    """Return a list of predicted sport labels, one per row of the CSV."""
    frame = pd.read_csv(filename, dtype=str)
    if frame.empty:
        return []
    params = json.loads(_find(PREPROCESSING_FILE).read_text(encoding="utf-8"))
    probabilities = predict_proba_features(build_features(frame, params))
    with np.load(_find(MODEL_FILE), allow_pickle=False) as model:
        classes = model["classes"].astype(str)
    return [str(label) for label in classes[probabilities.argmax(axis=1)]]


if __name__ == "__main__":
    for prediction in predict_all(sys.argv[1]):
        print(prediction)
