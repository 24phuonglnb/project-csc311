"""Train, evaluate, and export the CSC311 MLP.

The existing respondent-level train/validation/test splits are used directly:

    python train_mlp.py search
    python train_mlp.py final

Search fits preprocessing and models on train and uses validation for selection.
Final refits on train+validation, evaluates test once, and exports a NumPy model
that can be loaded by mlp_inference.py without PyTorch or scikit-learn.
"""

import argparse
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                            confusion_matrix, f1_score)
from sklearn.preprocessing import StandardScaler
from torch import nn

SEED = 311
ENSEMBLE_SEEDS = [1, 2, 3, 4, 5]
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "processed_data"
OUTPUT_DIR = ROOT / "mlp_outputs"
BROAD_CONFIGS = 48
NARROW_CONFIGS = 16
TOP_K_FOR_NARROWING = 5

SEARCH_SPACE = {
    "hidden_layers": [1, 2, 3, 5],
    "hidden_width": [16, 32, 48, 64],
    "activation": ["relu", "leaky_relu", "tanh"],
    "learning_rate": [0.001, 0.01, 0.1, 0.2],
    "batch_size": [32, 64, 128, 256],
    "max_epochs": [100, 250, 500, 1000],
    "l2": [0.001, 0.01, 0.1, 1.0],
    "patience": [1, 2, 5, 10],
}
ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
}


def load_split(data_dir, name):
    """Load a cleaned split and validate its feature matrix."""
    path = Path(data_dir) / f"{name}_clean.csv"
    frame = pd.read_csv(path)
    if "label" not in frame:
        raise ValueError(f"{path} does not contain a label column.")
    features = frame.drop(columns="label")
    if features.empty or features.columns.has_duplicates:
        raise ValueError(f"{path} must have non-empty, uniquely named features.")
    X = features.to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError(f"{path} contains missing or non-finite feature values.")
    labels = frame["label"].astype(str).to_numpy()
    if not len(labels) or not np.isfinite(X).all():
        raise ValueError(f"{path} contains no usable observations.")
    return X, labels, features.columns.to_list()


def load_data(data_dir, include_test=False):
    """Load development splits, opening test data only for final evaluation."""
    train_X, train_y, train_columns = load_split(data_dir, "train")
    val_X, val_y, val_columns = load_split(data_dir, "val")
    if train_columns != val_columns:
        raise ValueError("Train and validation feature columns must match in order.")
    classes = sorted(set(train_y))
    if not set(val_y).issubset(classes):
        raise ValueError("Validation contains labels not present in training.")
    test_X, test_y = None, None
    if include_test:
        test_X, test_y, test_columns = load_split(data_dir, "test")
        if train_columns != test_columns:
            raise ValueError("Test feature columns must match training in order.")
        if not set(test_y).issubset(classes):
            raise ValueError("Test contains labels not present in training.")
    return (train_X, train_y, val_X, val_y, test_X, test_y,
            train_columns, classes)


def to_tensors(X, y, classes):
    class_index = {name: index for index, name in enumerate(classes)}
    targets = np.asarray([class_index[label] for label in y], dtype=np.int64)
    return torch.from_numpy(np.asarray(X, dtype=np.float32)), torch.from_numpy(targets)


def fit_feature_scaler(X, columns):
    """Standardize the five continuous/ordinal survey features only.

    Keeping binary, multi-hot, and TF-IDF features at their original scale
    avoids inflating rare indicators by dividing them by a tiny standard
    deviation.
    """
    structured = [
        index for index, name in enumerate(columns)
        if not name.endswith("_missing")
        and not name.startswith(("location_", "season_", "music_"))
        and "_tfidf_" not in name
    ]
    if len(structured) != 5:
        raise ValueError(
            f"Expected five continuous/ordinal features, found {len(structured)}."
        )
    scaler = StandardScaler().fit(X[:, structured])
    means = np.zeros(X.shape[1], dtype=np.float64)
    scales = np.ones(X.shape[1], dtype=np.float64)
    means[structured] = scaler.mean_
    scales[structured] = scaler.scale_
    return means, scales


def scale_features(X, means, scales):
    return ((X - means) / scales).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, n_features, n_classes, config):
        super().__init__()
        layers = []
        width = n_features
        for _ in range(config["hidden_layers"]):
            layers.extend((
                nn.Linear(width, config["hidden_width"]),
                ACTIVATIONS[config["activation"]](),
            ))
            width = config["hidden_width"]
        layers.append(nn.Linear(width, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, values):
        return self.net(values)


def train_model(train, validation, n_classes, config, seed=SEED,
                fixed_epochs=None):
    """Fit with validation-loss early stopping, or for a fixed epoch count."""
    torch.manual_seed(seed)
    X_train, y_train = train
    model = MLP(X_train.shape[1], n_classes, config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"],
        weight_decay=config["l2"],
    )
    max_epochs = fixed_epochs or config["max_epochs"]
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    waiting = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for indices in torch.randperm(len(X_train)).split(config["batch_size"]):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(X_train[indices]), y_train[indices])
            if not torch.isfinite(loss):
                break
            loss.backward()
            optimizer.step()
        else:
            if validation is None:
                best_epoch = epoch
                continue
            model.eval()
            with torch.no_grad():
                val_loss = F.cross_entropy(model(validation[0]), validation[1]).item()
            if np.isfinite(val_loss) and val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }
                waiting = 0
            else:
                waiting += 1
                if waiting >= config["patience"]:
                    break
            continue
        break

    if validation is not None and best_state is None:
        raise FloatingPointError("Training diverged before a valid checkpoint was found.")
    if best_state is not None:
        model.load_state_dict(best_state)
    if best_epoch == 0:
        raise FloatingPointError("Training did not complete a finite epoch.")
    return model, best_epoch


def predict_probabilities(models, X):
    probabilities = []
    with torch.no_grad():
        for model in models:
            model.eval()
            probabilities.append(torch.softmax(model(X), dim=1).cpu().numpy())
    return np.mean(probabilities, axis=0)


def predict_indices(models, X):
    return predict_probabilities(models, X).argmax(axis=1)


def sample_configs(space, count, rng, seen):
    names = list(space)
    candidates = [
        dict(zip(names, values))
        for values in itertools.product(*(space[name] for name in names))
    ]
    candidates = [
        config for config in candidates
        if json.dumps(config, sort_keys=True) not in seen
    ]
    chosen = rng.sample(candidates, min(count, len(candidates)))
    seen.update(json.dumps(config, sort_keys=True) for config in chosen)
    return chosen


def evaluate_config(config, X_train, y_train, X_val, y_val, classes, columns):
    means, scales = fit_feature_scaler(X_train, columns)
    train = to_tensors(scale_features(X_train, means, scales), y_train, classes)
    validation = to_tensors(scale_features(X_val, means, scales), y_val, classes)
    models, epochs, individual_scores = [], [], []
    truth = validation[1].numpy()
    for seed in ENSEMBLE_SEEDS:
        model, best_epoch = train_model(
            train, validation, len(classes), config, seed=seed,
        )
        models.append(model)
        epochs.append(best_epoch)
        individual_scores.append(
            accuracy_score(truth, predict_indices([model], validation[0]))
        )
    predictions = predict_indices(models, validation[0])
    return {
        **config,
        "best_epoch": int(np.median(epochs)),
        "validation_accuracy": accuracy_score(truth, predictions),
        "validation_macro_f1": f1_score(
            truth, predictions, average="macro", zero_division=0
        ),
        "mean_seed_accuracy": float(np.mean(individual_scores)),
        "std_seed_accuracy": float(np.std(individual_scores)),
    }


def run_search(data_dir, output_dir, broad_count, narrow_count):
    X_train, y_train, X_val, y_val, _, _, columns, classes = load_data(data_dir)
    rng = random.Random(SEED)
    seen = set()
    started = time.monotonic()
    rows = []

    broad = sample_configs(SEARCH_SPACE, broad_count, rng, seen)
    for index, config in enumerate(broad, start=1):
        try:
            row = evaluate_config(
                config, X_train, y_train, X_val, y_val, classes, columns,
            )
            row["stage"] = "broad"
            rows.append(row)
            print(
                f"[broad {index}/{len(broad)}] "
                f"macro-F1={row['validation_macro_f1']:.4f} "
                f"accuracy={row['validation_accuracy']:.4f} "
                f"epoch={row['best_epoch']} ({time.monotonic() - started:.0f}s)",
                flush=True,
            )
        except FloatingPointError as error:
            print(f"[broad {index}/{len(broad)}] skipped: {error}", flush=True)

    if not rows:
        raise RuntimeError("Every broad-search configuration diverged.")
    top = sorted(
        rows,
        key=lambda row: (row["validation_macro_f1"],
                         row["validation_accuracy"]),
        reverse=True,
    )[:TOP_K_FOR_NARROWING]
    narrowed_space = {
        key: sorted({row[key] for row in top}, key=str)
        for key in SEARCH_SPACE
    }
    narrow = sample_configs(narrowed_space, narrow_count, rng, seen)
    for index, config in enumerate(narrow, start=1):
        try:
            row = evaluate_config(
                config, X_train, y_train, X_val, y_val, classes, columns,
            )
            row["stage"] = "narrow"
            rows.append(row)
            print(
                f"[narrow {index}/{len(narrow)}] "
                f"macro-F1={row['validation_macro_f1']:.4f} "
                f"accuracy={row['validation_accuracy']:.4f} "
                f"epoch={row['best_epoch']} ({time.monotonic() - started:.0f}s)",
                flush=True,
            )
        except FloatingPointError as error:
            print(f"[narrow {index}/{len(narrow)}] skipped: {error}", flush=True)

    results = pd.DataFrame(rows).sort_values(
        ["validation_macro_f1", "validation_accuracy", "hidden_layers",
         "hidden_width"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "mlp_validation_results.csv", index=False)
    best = results.iloc[0]
    best_config = {
        key: (best[key].item() if isinstance(best[key], np.generic) else best[key])
        for key in SEARCH_SPACE
    }
    best_config["final_epochs"] = int(best["best_epoch"])
    best_config["validation_accuracy"] = float(best["validation_accuracy"])
    best_config["validation_macro_f1"] = float(best["validation_macro_f1"])
    best_config["n_features"] = len(columns)
    best_config["classes"] = classes
    best_config["feature_scaling"] = "structured_only"
    (output_dir / "mlp_best_config.json").write_text(
        json.dumps(best_config, indent=2) + "\n"
    )
    print("\nSelected validation configuration:")
    print(json.dumps(best_config, indent=2))


def export_numpy_model(models, means, scales, columns, classes, activation, output_path):
    """Write all inference parameters in a non-pickled NumPy archive."""
    arrays = {
        "scaler_mean": means,
        "scaler_scale": scales,
        "feature_names": np.asarray(columns, dtype=np.str_),
        "classes": np.asarray(classes, dtype=np.str_),
        "activation": np.asarray(activation, dtype=np.str_),
    }
    arrays["n_models"] = np.asarray(len(models), dtype=np.int64)
    linear_layers = [
        [layer for layer in model.net if isinstance(layer, nn.Linear)]
        for model in models
    ]
    n_layers = len(linear_layers[0])
    if any(len(layers) != n_layers for layers in linear_layers):
        raise ValueError("Ensemble members must have matching architectures.")
    arrays["n_layers"] = np.asarray(n_layers, dtype=np.int64)
    for model_index, layers in enumerate(linear_layers):
        for layer_index, layer in enumerate(layers):
            arrays[f"weight_{model_index}_{layer_index}"] = (
                layer.weight.detach().cpu().numpy().copy()
            )
            arrays[f"bias_{model_index}_{layer_index}"] = (
                layer.bias.detach().cpu().numpy().copy()
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def run_final(data_dir, output_dir):
    config_path = output_dir / "mlp_best_config.json"
    results_path = output_dir / "mlp_test_results.json"
    if not config_path.exists():
        raise SystemExit("Run `python train_mlp.py search` before final training.")
    if results_path.exists():
        raise SystemExit(
            f"{results_path} already exists; refusing to evaluate the test set again."
        )

    config = json.loads(config_path.read_text())
    train_X, train_y, val_X, val_y, test_X, test_y, columns, classes = load_data(
        data_dir, include_test=True
    )
    if config["n_features"] != len(columns) or config["classes"] != classes:
        raise ValueError("The saved search configuration does not match the loaded data.")
    if config["feature_scaling"] != "structured_only":
        raise ValueError("The saved search configuration uses unsupported feature scaling.")

    development_X = np.vstack((train_X, val_X))
    development_y = np.concatenate((train_y, val_y))
    means, scales = fit_feature_scaler(development_X, columns)
    train = to_tensors(
        scale_features(development_X, means, scales), development_y, classes
    )
    models = [
        train_model(
            train, None, len(classes), config, seed=seed,
            fixed_epochs=int(config["final_epochs"]),
        )[0]
        for seed in ENSEMBLE_SEEDS
    ]

    model_path = output_dir / "mlp_model.npz"
    export_numpy_model(
        models, means, scales, columns, classes, config["activation"], model_path
    )

    # Confirm the standalone NumPy inference path matches the trained model
    # before the one-time test evaluation.
    from mlp_inference import predict_proba

    val_frame = pd.read_csv(Path(data_dir) / "val_clean.csv").drop(columns="label")
    sample = val_frame.iloc[:min(8, len(val_frame))]
    numpy_probabilities = predict_proba(sample, model_path)
    validation_sample = torch.from_numpy(
        scale_features(sample.to_numpy(dtype=np.float32), means, scales)
    )
    torch_probabilities = predict_probabilities(models, validation_sample)
    if not np.allclose(numpy_probabilities, torch_probabilities, atol=1e-6):
        raise RuntimeError("Exported NumPy inference does not match the trained model.")

    test_features = to_tensors(scale_features(test_X, means, scales), test_y, classes)[0]
    predictions = predict_indices(models, test_features)
    truth = np.asarray([classes.index(label) for label in test_y])
    accuracy = accuracy_score(truth, predictions)
    results = {
        "accuracy": float(accuracy),
        "macro_f1": float(f1_score(truth, predictions, average="macro",
                                  zero_division=0)),
        "weighted_f1": float(f1_score(truth, predictions, average="weighted",
                                     zero_division=0)),
        "classes": classes,
        "confusion_matrix": confusion_matrix(
            truth, predictions, labels=np.arange(len(classes))
        ).tolist(),
        "validation_accuracy": config["validation_accuracy"],
        "validation_macro_f1": config["validation_macro_f1"],
        "test_rows": len(test_y),
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "feature_scaling": config["feature_scaling"],
    }
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(classification_report(
        truth, predictions, labels=np.arange(len(classes)),
        target_names=classes, digits=4, zero_division=0,
    ))
    print(f"Test accuracy: {accuracy:.4%}")
    print(f"Test macro-F1: {results['macro_f1']:.4f}")
    print(f"Saved inference model: {model_path}")
    if accuracy <= 0.90:
        print("WARNING: the requested >90% test accuracy target was not reached.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="select a model using validation only")
    search.add_argument("--data-dir", type=Path, default=DATA_DIR)
    search.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    search.add_argument("--broad", type=int, default=BROAD_CONFIGS)
    search.add_argument("--narrow", type=int, default=NARROW_CONFIGS)
    search.set_defaults(run=lambda args: run_search(
        args.data_dir, args.output_dir, args.broad, args.narrow
    ))
    final = commands.add_parser("final", help="refit, test once, and export inference model")
    final.add_argument("--data-dir", type=Path, default=DATA_DIR)
    final.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    final.set_defaults(run=lambda args: run_final(args.data_dir, args.output_dir))
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.run(args)


if __name__ == "__main__":
    main()
