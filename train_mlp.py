"""CSC311 MLP: grouped cross-validated hyperparameter search and final retraining.

Reads train_clean.csv, val_clean.csv and test_clean.csv from the processed_data
folder next to this script. Run:

    python train_mlp.py search
    python train_mlp.py final
    python train_mlp.py final --use-test

`search` and `final` never touch the test set. Only add --use-test once the MLP
has been chosen as the final model across all three model families.

The CSVs have no ID column, so respondents are recovered from row order: each
consecutive block of three rows is one respondent. load_split() stops with an
error if the labels do not follow that pattern.

Uses the *_clean.csv files because the scaler has to be refit inside each fold.
The *_scaled.csv files were scaled with the whole training split.
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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn

SEED = 311
STABILITY_SEEDS = [1, 2, 3, 4, 5]
N_FOLDS = 5
EARLY_STOP_SIZE = 0.15     # share of each CV training fold used for early stopping
BROAD_CONFIGS = 60         # stage 1 random search
NARROW_CONFIGS = 30        # stage 2 random search
TOP_K_FOR_NARROWING = 5
MOMENTUM = 0.9
DATA_DIR = Path(__file__).resolve().parent / "processed_data"
OUTPUT_DIR = Path("mlp_outputs")

SEARCH_SPACE = {
    "hidden_layers": [1, 2, 3, 5],
    "hidden_width": [16, 32, 48, 64],
    "activation": ["relu", "leaky_relu", "tanh"],
    "learning_rate": [0.001, 0.01, 0.1, 0.2],
    "batch_size": [32, 64, 128, 256],
    "max_epochs": [100, 250, 500, 1000],
    "l2": [0.001, 0.01, 0.1, 1],
    "patience": [1, 2, 5, 10],
}
ACTIVATIONS = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_split(data_dir, name, group_offset=0):
    """Return a split's features, labels and respondent numbers."""
    df = pd.read_csv(Path(data_dir) / f"{name}_clean.csv")
    labels = df["label"].to_numpy()
    if len(df) % 3 or any(len(set(labels[i:i + 3])) != 3
                          for i in range(0, len(df), 3)):
        raise ValueError(f"{name}_clean.csv is not in blocks of three sports, so "
                         "respondents cannot be recovered from row order.")
    X = df.drop(columns="label").to_numpy(dtype=np.float32)
    groups = np.arange(len(df)) // 3 + group_offset
    return X, labels, groups


def load_development_data(data_dir):
    """Combine train and validation, the pool that cross-validation splits."""
    X_train, y_train, g_train = load_split(data_dir, "train")
    X_val, y_val, g_val = load_split(data_dir, "val", group_offset=g_train.max() + 1)
    return (np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]),
            np.concatenate([g_train, g_val]))


def to_tensors(X, y, classes):
    index = {name: i for i, name in enumerate(classes)}
    targets = np.array([index[label] for label in y])
    return torch.from_numpy(X.astype(np.float32)), torch.from_numpy(targets).long()


def build_folds(X, y, groups, classes):
    """Scale within each fold, using only that fold's fitting rows."""
    folds = []
    assignments = np.zeros(len(X), dtype=int)
    cv = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for number, (pool, val) in enumerate(cv.split(X, y, groups)):
        assignments[val] = number
        holdout = GroupShuffleSplit(n_splits=1, test_size=EARLY_STOP_SIZE,
                                    random_state=SEED)
        fit_pos, stop_pos = next(holdout.split(pool, groups=groups[pool]))
        fit, stop = pool[fit_pos], pool[stop_pos]
        scaler = StandardScaler().fit(X[fit])
        folds.append({
            "train": to_tensors(scaler.transform(X[fit]), y[fit], classes),
            "stop": to_tensors(scaler.transform(X[stop]), y[stop], classes),
            "val": to_tensors(scaler.transform(X[val]), y[val], classes),
        })
    return folds, assignments


# ---------------------------------------------------------------------------
# Model and training
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, n_features, n_classes, hidden_layers, hidden_width, activation):
        super().__init__()
        layers, size = [], n_features
        for _ in range(hidden_layers):
            layers += [nn.Linear(size, hidden_width), ACTIVATIONS[activation]()]
            size = hidden_width
        layers.append(nn.Linear(size, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_model(n_features, n_classes, config):
    return MLP(n_features, n_classes, config["hidden_layers"],
               config["hidden_width"], config["activation"])


def train_model(train, stop, n_classes, config, seed=SEED, fixed_epochs=None):
    """Train with early stopping, or for exactly fixed_epochs when stop is None."""
    torch.manual_seed(seed)
    X, y = train
    model = make_model(X.shape[1], n_classes, config)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"],
                                momentum=MOMENTUM, weight_decay=config["l2"])
    epochs = fixed_epochs or config["max_epochs"]
    best_loss, best_state, best_epoch, waited = np.inf, None, 0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in torch.randperm(len(X)).split(config["batch_size"]):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(X[batch]), y[batch])
            if not torch.isfinite(loss):  # a large learning rate diverged
                return finish(model, best_state, best_epoch)
            loss.backward()
            optimizer.step()

        if stop is None:
            best_epoch = epoch
            continue
        model.eval()
        with torch.no_grad():
            stop_loss = F.cross_entropy(model(stop[0]), stop[1]).item()
        if stop_loss < best_loss:
            best_loss, best_epoch, waited = stop_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= config["patience"]:
                break
    return finish(model, best_state, best_epoch)


def finish(model, best_state, best_epoch):
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, max(best_epoch, 1)


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.nan_to_num(model(X)).argmax(dim=1).numpy()


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------
def sample_configs(space, count, rng, seen):
    names = list(space)
    options = [dict(zip(names, values)) for values in itertools.product(*space.values())]
    unseen = [c for c in options if json.dumps(c, sort_keys=True) not in seen]
    chosen = rng.sample(unseen, min(count, len(unseen)))
    seen.update(json.dumps(c, sort_keys=True) for c in chosen)
    return chosen


def evaluate_config(config, folds, n_classes, seed=SEED):
    f1s, accuracies, epochs = [], [], []
    for fold in folds:
        model, best_epoch = train_model(fold["train"], fold["stop"], n_classes,
                                        config, seed=seed)
        predictions = predict(model, fold["val"][0])
        truth = fold["val"][1].numpy()
        f1s.append(f1_score(truth, predictions, average="macro", zero_division=0))
        accuracies.append(accuracy_score(truth, predictions))
        epochs.append(best_epoch)
    row = dict(config)
    row.update(mean_macro_f1=np.mean(f1s), std_macro_f1=np.std(f1s, ddof=1),
               mean_accuracy=np.mean(accuracies),
               std_accuracy=np.std(accuracies, ddof=1),
               median_best_epoch=int(np.median(epochs)))
    for i, (f1, accuracy) in enumerate(zip(f1s, accuracies)):
        row[f"fold{i}_macro_f1"] = f1
        row[f"fold{i}_accuracy"] = accuracy
    return row


def run_stage(stage, configs, folds, n_classes):
    rows = []
    start = time.time()
    for i, config in enumerate(configs, 1):
        row = evaluate_config(config, folds, n_classes)
        row["stage"] = stage
        rows.append(row)
        print(f"[{stage} {i}/{len(configs)}] macro-F1 {row['mean_macro_f1']:.4f} "
              f"± {row['std_macro_f1']:.4f} | {time.time() - start:.0f}s", flush=True)
    return rows


def narrowed_space(rows):
    best = sorted(rows, key=lambda r: r["mean_macro_f1"],
                  reverse=True)[:TOP_K_FOR_NARROWING]
    return {name: sorted({row[name] for row in best}, key=str)
            for name in SEARCH_SPACE}


def rank_results(rows, n_features, n_classes):
    table = pd.DataFrame(rows)
    table["parameters"] = [
        sum(p.numel() for p in make_model(n_features, n_classes, row).parameters())
        for row in rows
    ]
    # On a tie in macro-F1, prefer the smaller network.
    return table.sort_values(["mean_macro_f1", "parameters"],
                             ascending=[False, True]).reset_index(drop=True)


def warn_on_grid_edges(config):
    for name in ["hidden_layers", "hidden_width", "learning_rate", "l2", "max_epochs"]:
        values = SEARCH_SPACE[name]
        if config[name] in (values[0], values[-1]):
            print(f"Note: best {name}={config[name]} is at the edge of its range.")


def to_python(value):
    return value.item() if isinstance(value, np.generic) else value


def command_search(args):
    X, y, groups = load_development_data(args.data_dir)
    classes = sorted(set(y))
    folds, assignments = build_folds(X, y, groups, classes)
    print(f"Development pool: {len(X)} rows, {len(set(groups))} respondents, "
          f"{X.shape[1]} features, classes {classes}\n")

    OUTPUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"row": np.arange(len(X)), "respondent": groups,
                  "fold": assignments}).to_csv(OUTPUT_DIR / "cv_folds.csv", index=False)

    rng = random.Random(SEED)
    seen = set()
    rows = run_stage("broad", sample_configs(SEARCH_SPACE, args.broad, rng, seen),
                     folds, len(classes))
    narrow = sample_configs(narrowed_space(rows), args.narrow, rng, seen)
    rows += run_stage("narrow", narrow, folds, len(classes))

    table = rank_results(rows, X.shape[1], len(classes))
    table.to_csv(OUTPUT_DIR / "mlp_cv_results.csv", index=False)
    best = table.iloc[0]
    config = {name: to_python(best[name]) for name in SEARCH_SPACE}
    config["final_epochs"] = int(best["median_best_epoch"])
    (OUTPUT_DIR / "mlp_best_config.json").write_text(json.dumps(config, indent=2))

    print("\nBest configuration:", json.dumps(config, indent=2))
    print(f"CV macro-F1: {best['mean_macro_f1']:.4f} ± {best['std_macro_f1']:.4f}")
    print(f"CV accuracy: {best['mean_accuracy']:.4f} ± {best['std_accuracy']:.4f}")
    warn_on_grid_edges(config)

    stability = [evaluate_config(config, folds, len(classes), seed=seed)
                 for seed in STABILITY_SEEDS]
    stability = pd.DataFrame(stability)[["mean_macro_f1", "mean_accuracy"]]
    stability.insert(0, "seed", STABILITY_SEEDS)
    stability.to_csv(OUTPUT_DIR / "mlp_seed_stability.csv", index=False)
    print(f"\nBest config over {len(STABILITY_SEEDS)} new seeds: macro-F1 "
          f"{stability['mean_macro_f1'].mean():.4f} ± "
          f"{stability['mean_macro_f1'].std():.4f}")


# ---------------------------------------------------------------------------
# Final retraining
# ---------------------------------------------------------------------------
def command_final(args):
    config_path = OUTPUT_DIR / "mlp_best_config.json"
    if not config_path.exists():
        raise SystemExit("Run `python train_mlp.py search` first.")
    results_path = OUTPUT_DIR / "mlp_test_results.json"
    if args.use_test and results_path.exists():
        raise SystemExit("The test set has already been used once.")

    config = json.loads(config_path.read_text())
    X, y, _ = load_development_data(args.data_dir)
    classes = sorted(set(y))
    scaler = StandardScaler().fit(X)
    train = to_tensors(scaler.transform(X), y, classes)
    # No held-out rows remain, so train for the median epoch that CV selected.
    model, _ = train_model(train, None, len(classes), config,
                           fixed_epochs=config["final_epochs"])

    torch.save({"state_dict": model.state_dict(), "config": config,
                "classes": classes, "n_features": X.shape[1],
                "scaler_mean": torch.from_numpy(scaler.mean_),
                "scaler_scale": torch.from_numpy(scaler.scale_)},
               OUTPUT_DIR / "mlp_final.pt")
    print(f"Retrained on {len(X)} rows for {config['final_epochs']} epochs. "
          f"Saved {OUTPUT_DIR / 'mlp_final.pt'}.")

    if not args.use_test:
        print("The test set was not used.")
        return

    X_test, y_test, _ = load_split(args.data_dir, "test")
    X_test, y_test = to_tensors(scaler.transform(X_test), y_test, classes)
    predictions = predict(model, X_test)
    truth = y_test.numpy()
    results = {
        "accuracy": accuracy_score(truth, predictions),
        "macro_f1": f1_score(truth, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(truth, predictions, average="weighted", zero_division=0),
        "classes": classes,
        "confusion_matrix": confusion_matrix(truth, predictions).tolist(),
    }
    results_path.write_text(json.dumps(results, indent=2))
    print(classification_report(truth, predictions, target_names=classes, digits=4,
                                zero_division=0))
    print("Confusion matrix (rows are true classes):")
    print(np.array(results["confusion_matrix"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search")
    search.add_argument("--data-dir", default=DATA_DIR)
    search.add_argument("--broad", type=int, default=BROAD_CONFIGS)
    search.add_argument("--narrow", type=int, default=NARROW_CONFIGS)
    search.set_defaults(run=command_search)
    final = commands.add_parser("final")
    final.add_argument("--data-dir", default=DATA_DIR)
    final.add_argument("--use-test", action="store_true")
    final.set_defaults(run=command_final)
    args = parser.parse_args()

    torch.set_num_threads(1)  # faster than multithreading for a network this small
    args.run(args)


if __name__ == "__main__":
    main()