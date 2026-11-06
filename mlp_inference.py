"""NumPy/pandas-only inference for the exported CSC311 MLP."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MODEL = Path(__file__).resolve().parent / "mlp_outputs" / "mlp_model.npz"


def _features_array(features, model):
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    if features.columns.has_duplicates:
        raise ValueError("Feature column names must be unique.")

    frame = features.drop(columns="label", errors="ignore")
    expected = model["feature_names"].tolist()
    missing = [name for name in expected if name not in frame.columns]
    extra = [name for name in frame.columns if name not in expected]
    if missing or extra:
        raise ValueError(f"Feature columns do not match model; missing={missing}, extra={extra}.")

    values = frame.loc[:, expected].to_numpy(dtype=np.float64)
    if not len(values):
        raise ValueError("At least one feature row is required.")
    if not np.isfinite(values).all():
        raise ValueError("Feature values must be finite.")
    mean = model["scaler_mean"]
    scale = model["scaler_scale"]
    if (mean.shape != (len(expected),) or scale.shape != mean.shape or
            not np.isfinite(mean).all() or not np.isfinite(scale).all() or
            (scale <= 0).any()):
        raise ValueError("Model contains invalid feature-scaling parameters.")
    return (values - mean) / scale


def _activate(values, name):
    if name == "relu":
        return np.maximum(values, 0)
    if name == "leaky_relu":
        return np.where(values >= 0, values, values * 0.01)
    if name == "tanh":
        return np.tanh(values)
    raise ValueError(f"Unsupported model activation: {name!r}.")


def predict_proba(features, model_path=DEFAULT_MODEL):
    """Return class probabilities for a feature DataFrame."""
    with np.load(model_path, allow_pickle=False) as model:
        scaled_values = _features_array(features, model)
        activation = str(model["activation"].item())
        n_layers = int(model["n_layers"])
        n_models = int(model["n_models"])
        if n_layers < 1 or n_models < 1:
            raise ValueError("Model must contain at least one network and linear layer.")
        ensemble = []
        for model_index in range(n_models):
            values = scaled_values
            for layer_index in range(n_layers):
                weight = model[f"weight_{model_index}_{layer_index}"]
                bias = model[f"bias_{model_index}_{layer_index}"]
                if (weight.ndim != 2 or bias.shape != (weight.shape[0],) or
                        not np.isfinite(weight).all() or not np.isfinite(bias).all()):
                    raise ValueError(
                        f"Model {model_index} layer {layer_index} has invalid parameters."
                    )
                if values.shape[1] != weight.shape[1]:
                    raise ValueError(
                        f"Model {model_index} layer {layer_index} has the wrong input width."
                    )
                values = values @ weight.T + bias
                if layer_index < n_layers - 1:
                    values = _activate(values, activation)
            shifted = values - values.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            ensemble.append(probabilities)
        probabilities = np.mean(ensemble, axis=0)
        if not np.isfinite(probabilities).all():
            raise ValueError("Model produced non-finite probabilities.")
        return probabilities


def predict(features, model_path=DEFAULT_MODEL):
    """Return predicted class labels indexed like the input DataFrame."""
    probabilities = predict_proba(features, model_path)
    with np.load(model_path, allow_pickle=False) as model:
        classes = model["classes"].astype(str)
    return pd.Series(classes[probabilities.argmax(axis=1)],
                     index=features.index, name="prediction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path,
                        help="CSV with the same clean features used for training.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    probabilities = predict_proba(frame, args.model)
    with np.load(args.model, allow_pickle=False) as model:
        classes = model["classes"].astype(str)
    output = pd.DataFrame({"prediction": classes[probabilities.argmax(axis=1)]})
    for index, name in enumerate(classes):
        output[f"probability_{name}"] = probabilities[:, index]
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} predictions to {args.output}")


if __name__ == "__main__":
    main()
