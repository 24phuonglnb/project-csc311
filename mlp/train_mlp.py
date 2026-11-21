"""Train, evaluate, and export the CSC311 MLP.

The existing respondent-level train/validation/test splits are used directly:

    python export_preprocessing.py   # writes TF-IDF block sizes used below
    python train_mlp.py search
    python train_mlp.py final

Feature pipeline variants (PCA and k-means fallbacks), each fit on training rows only:
    clean features [+ one-hot k-means cluster IDs per free-text question]
    -> standardize -> [PCA keeping a fraction of the variance] -> MLP

Search runs the same two-stage random search over the project MLP grid for
every pipeline variant, fitting on train and scoring on validation. Each
configuration is scored by the mean validation macro-F1 of single MLPs trained
with several seeds. Final refits the selected pipeline and a single MLP on
train+validation, evaluates the test split once, and exports a NumPy archive
loaded by pred.py without PyTorch or scikit-learn.
"""

import argparse
import itertools
import json
import os
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report,
                            confusion_matrix, f1_score)
from sklearn.preprocessing import StandardScaler
from torch import nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pred

SEED = 311
# Seeds used to score each configuration on validation (results are averaged).
SELECTION_SEEDS = [1, 2, 3, 4, 5]
# Seed for the single submitted model; fixed before any test evaluation.
FINAL_SEED = SEED
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "processed_data"
OUTPUT_DIR = ROOT / "mlp_outputs"
PREPROCESSING_PATH = OUTPUT_DIR / pred.PREPROCESSING_FILE
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
# 0 means "off". PCA keeps enough components for this fraction of variance;
# k-means adds k one-hot cluster indicators per free-text question.
PCA_VARIANCE = [0.0, 0.80, 0.90, 0.95, 0.99]
KMEANS_K = [0, 3, 5, 10]
VARIANTS = [{"pca_variance": p, "kmeans_k": k}
            for p in PCA_VARIANCE for k in KMEANS_K]
ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
}
# Ties prefer the plain pipeline, then the smaller network.
SORT_COLUMNS = ["mean_seed_macro_f1", "mean_seed_accuracy", "kmeans_k",
                "pca_variance", "hidden_layers", "hidden_width"]
SORT_ASCENDING = [False, False, True, True, True, True]


def load_split(data_dir, name):
    """Load a cleaned split and validate its feature matrix."""
    path = Path(data_dir) / f"{name}_clean.csv"
    frame = pd.read_csv(path)
    if "label" not in frame:
        raise ValueError(f"{path} does not contain a label column.")
    features = frame.drop(columns="label")
    if features.empty or features.columns.has_duplicates:
        raise ValueError(f"{path} must have non-empty, uniquely named features.")
    X = features.to_numpy(dtype=np.float64)
    if not len(X) or not np.isfinite(X).all():
        raise ValueError(f"{path} contains no rows or non-finite feature values.")
    labels = frame["label"].astype(str).to_numpy()
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


def text_blocks(columns):
    """Column indices of each free-text question's TF-IDF block, in order."""
    if not PREPROCESSING_PATH.exists():
        raise SystemExit("Run `python export_preprocessing.py` first.")
    sizes = [len(block["vocabulary"]) for block in
             json.loads(PREPROCESSING_PATH.read_text())["tfidf"]]
    start = len(columns) - sum(sizes)
    blocks = []
    for size in sizes:
        blocks.append(np.arange(start, start + size, dtype=np.int64))
        start += size
    if not all("_tfidf_" in columns[i] for block in blocks for i in block):
        raise ValueError("TF-IDF blocks are not the trailing feature columns.")
    return blocks


def fit_pipeline(X, columns, variant):
    """Fit k-means / scaler / PCA on X; return arrays in pred.py's format."""
    k = int(variant["kmeans_k"])
    pipeline = {"kmeans_k": np.asarray(k, dtype=np.int64),
                "kmeans_n_blocks": np.asarray(0, dtype=np.int64)}
    if k:
        blocks = text_blocks(columns)
        pipeline["kmeans_n_blocks"] = np.asarray(len(blocks), dtype=np.int64)
        for index, block in enumerate(blocks):
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X[:, block])
            pipeline[f"kmeans_columns_{index}"] = block
            pipeline[f"kmeans_centroids_{index}"] = kmeans.cluster_centers_
    expanded = pred.add_cluster_features(X, pipeline)
    scaler = StandardScaler().fit(expanded)
    pipeline["scaler_mean"] = scaler.mean_.astype(np.float64)
    pipeline["scaler_scale"] = scaler.scale_.astype(np.float64)
    pipeline["pca_n_components"] = np.asarray(0, dtype=np.int64)
    if variant["pca_variance"]:
        pca = PCA(n_components=float(variant["pca_variance"]), svd_solver="full")
        pca.fit((expanded - pipeline["scaler_mean"]) / pipeline["scaler_scale"])
        pipeline["pca_n_components"] = np.asarray(pca.n_components_, dtype=np.int64)
        pipeline["pca_mean"] = pca.mean_.astype(np.float64)
        pipeline["pca_components"] = pca.components_.astype(np.float64)
    return pipeline


def transform(X, pipeline):
    """Apply the pipeline with the exact NumPy code pred.py uses."""
    return pred.transform_features(X, pipeline).astype(np.float32)


def to_tensors(X, y, classes):
    class_index = {name: index for index, name in enumerate(classes)}
    targets = np.asarray([class_index[label] for label in y], dtype=np.int64)
    return torch.from_numpy(np.asarray(X, dtype=np.float32)), torch.from_numpy(targets)


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
    # Plain mini-batch gradient descent (no momentum). SGD's weight_decay adds
    # l2 * w to each gradient, i.e. an L2 penalty (l2 / 2) * ||w||^2 on the loss.
    optimizer = torch.optim.SGD(
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
        break  # a mini-batch loss was non-finite

    if validation is not None:
        if best_state is None:
            raise FloatingPointError("Training diverged before a valid checkpoint.")
        model.load_state_dict(best_state)
    elif best_epoch != max_epochs:
        raise FloatingPointError(
            f"Fixed-epoch training diverged after {best_epoch}/{max_epochs} epochs."
        )
    return model, best_epoch


def predict_probabilities(model, X):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(X), dim=1).cpu().numpy()


def predict_indices(model, X):
    return predict_probabilities(model, X).argmax(axis=1)


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


def evaluate_config(config, train, validation, n_classes):
    """Train one MLP per selection seed; report mean/std validation scores."""
    truth = validation[1].numpy()
    epochs, accuracies, macro_f1s = [], [], []
    for seed in SELECTION_SEEDS:
        model, best_epoch = train_model(train, validation, n_classes, config, seed=seed)
        predictions = predict_indices(model, validation[0])
        epochs.append(best_epoch)
        accuracies.append(accuracy_score(truth, predictions))
        macro_f1s.append(f1_score(truth, predictions, average="macro",
                                  zero_division=0))
    return {
        **config,
        "best_epoch": int(np.median(epochs)),
        "mean_seed_macro_f1": float(np.mean(macro_f1s)),
        "std_seed_macro_f1": float(np.std(macro_f1s)),
        "mean_seed_accuracy": float(np.mean(accuracies)),
        "std_seed_accuracy": float(np.std(accuracies)),
        "min_seed_accuracy": float(np.min(accuracies)),
        "max_seed_accuracy": float(np.max(accuracies)),
    }


def search_variant(args):
    """Two-stage random search for one pipeline variant (validation only)."""
    variant, data_dir, broad_count, narrow_count = args
    torch.set_num_threads(1)
    X_train, y_train, X_val, y_val, _, _, columns, classes = load_data(data_dir)
    pipeline = fit_pipeline(X_train, columns, variant)
    train = to_tensors(transform(X_train, pipeline), y_train, classes)
    validation = to_tensors(transform(X_val, pipeline), y_val, classes)
    rng = random.Random(SEED)  # same sampled configs for every variant
    seen = set()
    rows = []

    def run_stage(name, configs):
        for config in configs:
            try:
                row = evaluate_config(config, train, validation, len(classes))
            except FloatingPointError:
                continue
            rows.append({**variant, "n_inputs": train[0].shape[1], **row,
                         "stage": name})

    run_stage("broad", sample_configs(SEARCH_SPACE, broad_count, rng, seen))
    if not rows:
        return rows
    top = sorted(rows, key=lambda r: (r["mean_seed_macro_f1"],
                                      r["mean_seed_accuracy"]),
                 reverse=True)[:TOP_K_FOR_NARROWING]
    narrowed = {key: sorted({r[key] for r in top}, key=str) for key in SEARCH_SPACE}
    run_stage("narrow", sample_configs(narrowed, narrow_count, rng, seen))
    best = max(rows, key=lambda r: (r["mean_seed_macro_f1"], r["mean_seed_accuracy"]))
    print(f"variant pca={variant['pca_variance']} k={variant['kmeans_k']} "
          f"inputs={rows[0]['n_inputs']}: best macro-F1="
          f"{best['mean_seed_macro_f1']:.4f} accuracy={best['mean_seed_accuracy']:.4f}",
          flush=True)
    return rows


def run_search(data_dir, output_dir, broad_count, narrow_count, workers):
    _, _, _, _, _, _, columns, classes = load_data(data_dir)
    started = time.monotonic()
    tasks = [(variant, data_dir, broad_count, narrow_count) for variant in VARIANTS]
    with Pool(workers) as pool:
        rows = [row for result in pool.map(search_variant, tasks) for row in result]
    if not rows:
        raise RuntimeError("Every configuration diverged.")
    print(f"Search finished in {time.monotonic() - started:.0f}s")

    results = pd.DataFrame(rows).sort_values(
        SORT_COLUMNS, ascending=SORT_ASCENDING).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "mlp_validation_results.csv", index=False)
    summary = results.groupby(["pca_variance", "kmeans_k"], sort=False).head(1)
    summary = summary[["pca_variance", "kmeans_k", "n_inputs", "mean_seed_macro_f1",
                       "mean_seed_accuracy", "std_seed_accuracy", *SEARCH_SPACE,
                       "best_epoch"]].reset_index(drop=True)
    summary.to_csv(output_dir / "mlp_variant_summary.csv", index=False)
    print(summary.to_string())

    best = results.iloc[0]
    best_config = {
        key: (best[key].item() if isinstance(best[key], np.generic) else best[key])
        for key in [*SEARCH_SPACE, "pca_variance", "kmeans_k"]
    }
    best_config.update({
        "optimizer": "sgd",
        "final_epochs": int(best["best_epoch"]),
        "final_seed": FINAL_SEED,
        "selection_seeds": SELECTION_SEEDS,
        "validation_mean_seed_macro_f1": float(best["mean_seed_macro_f1"]),
        "validation_mean_seed_accuracy": float(best["mean_seed_accuracy"]),
        "validation_std_seed_accuracy": float(best["std_seed_accuracy"]),
        "n_features": len(columns),
        "classes": classes,
        "feature_scaling": "standard_all",
    })
    (output_dir / "mlp_best_config.json").write_text(
        json.dumps(best_config, indent=2) + "\n"
    )
    print("\nSelected validation configuration:")
    print(json.dumps(best_config, indent=2))


def export_numpy_model(model, pipeline, columns, classes, activation, output_path):
    """Write all inference parameters in a non-pickled NumPy archive."""
    linear_layers = [layer for layer in model.net if isinstance(layer, nn.Linear)]
    arrays = {
        **pipeline,
        "feature_names": np.asarray(columns, dtype=np.str_),
        "classes": np.asarray(classes, dtype=np.str_),
        "activation": np.asarray(activation, dtype=np.str_),
        "n_layers": np.asarray(len(linear_layers), dtype=np.int64),
    }
    for index, layer in enumerate(linear_layers):
        arrays[f"weight_{index}"] = layer.weight.detach().cpu().numpy().copy()
        arrays[f"bias_{index}"] = layer.bias.detach().cpu().numpy().copy()
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
    if "kmeans_k" not in config or config["optimizer"] != "sgd":
        raise ValueError("The saved search configuration is from an older procedure.")

    # Retrain on train+validation; the pipeline is refit on the same rows.
    development_X = np.vstack((train_X, val_X))
    development_y = np.concatenate((train_y, val_y))
    pipeline = fit_pipeline(development_X, columns, config)
    train = to_tensors(transform(development_X, pipeline), development_y, classes)
    model, _ = train_model(
        train, None, len(classes), config, seed=int(config["final_seed"]),
        fixed_epochs=int(config["final_epochs"]),
    )

    model_path = output_dir / "mlp_model.npz"
    export_numpy_model(model, pipeline, columns, classes, config["activation"],
                       model_path)

    # Confirm the NumPy forward pass in pred.py matches PyTorch before the
    # one-time test evaluation (uses validation features only).
    numpy_probabilities = pred.predict_proba_features(val_X, model_path)
    torch_probabilities = predict_probabilities(
        model, torch.from_numpy(transform(val_X, pipeline)))
    if not np.allclose(numpy_probabilities, torch_probabilities, atol=1e-5):
        raise RuntimeError("Exported NumPy inference does not match the trained model.")

    predictions = predict_indices(model, torch.from_numpy(transform(test_X, pipeline)))
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
        "classification_report": classification_report(
            truth, predictions, labels=np.arange(len(classes)),
            target_names=classes, output_dict=True, zero_division=0,
        ),
        "test_rows": len(test_y),
        "n_inputs": int(train[0].shape[1]),
        "config": config,
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
    search.add_argument("--workers", type=int,
                        default=max(1, min(len(VARIANTS), (os.cpu_count() or 2) - 1)))
    search.set_defaults(run=lambda args: run_search(
        args.data_dir, args.output_dir, args.broad, args.narrow, args.workers
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
