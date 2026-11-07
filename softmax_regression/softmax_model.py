"""softmax_model.py — Multinomial Softmax Regression, implemented from scratch with NumPy.

Objective (mean cross-entropy + L2 on weights):
    L(W, b) = mean_cross_entropy + (lambda / 2) * sum(W^2)

Gradient of the objective:
    dL/dW = (1/N) * X.T @ (P - Y_onehot) + lambda * W
    dL/db = (1/N) * sum(P - Y_onehot, axis=0)

No scikit-learn or PyTorch is used for training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


# Utility functions
def softmax(Z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis

    Parameters:
    Z : (N, K) logit matrix

    Returns:
    P : (N, K) probability matrix; each row sums to 1
    """
    Z_shift = Z - Z.max(axis=1, keepdims=True)
    exp_Z = np.exp(Z_shift)
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)


def cross_entropy_loss(P: np.ndarray, y_int: np.ndarray) -> float:
    """Mean cross-entropy loss.

    Parameters
    P      : (N, K) predicted probabilities (output of softmax).
    y_int  : (N,) integer class indices in {0, ..., K-1}.

    Returns
    Scalar mean cross-entropy loss.
    """
    N = len(y_int)
    log_p = np.log(np.clip(P[np.arange(N), y_int], 1e-15, 1.0))
    return -log_p.mean()


def one_hot(y_int: np.ndarray, K: int) -> np.ndarray:
    """Convert integer labels to one-hot matrix of shape (N, K)."""
    N = len(y_int)
    Y = np.zeros((N, K), dtype=np.float64)
    Y[np.arange(N), y_int] = 1.0
    return Y


# SoftmaxRegression class
class SoftmaxRegression:
    """Multinomial Softmax Regression trained with mini-batch SGD.

    Parameters:
    n_classes : int
        Number of output classes (K).
    learning_rate : float
        Step size for SGD.
    batch_size : int
        Mini-batch size.
    max_epochs : int
        Maximum number of training epochs.
    l2_lambda : float
        L2 regularisation coefficient applied to W (not b).
    patience : int
        Early-stopping patience: stop if validation loss does not improve for
        this many consecutive epochs.
    random_state : int
        Seed for reproducible weight initialisation and batch shuffling.
    """

    def __init__(
        self,
        n_classes: int = 3,
        learning_rate: float = 0.01,
        batch_size: int = 64,
        max_epochs: int = 500,
        l2_lambda: float = 0.01,
        patience: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.l2_lambda = l2_lambda
        self.patience = patience
        self.random_state = random_state

        # Learned parameters (set during fit)
        self.W: Optional[np.ndarray] = None   # (D, K)
        self.b: Optional[np.ndarray] = None   # (K,)
        self.classes_: Optional[np.ndarray] = None  # original class strings

        # Training history
        self.train_losses_: list = []
        self.train_ce_losses_: list = []
        self.val_losses_: list = []
        self.best_epoch_: int = 0


    # Core computations
    def _logits(self, X: np.ndarray) -> np.ndarray:
        """Compute linear logits Z = X @ W + b."""
        return X @ self.W + self.b

    def _predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities for input X."""
        return softmax(self._logits(X))

    def _regularized_loss(self, P: np.ndarray, y_int: np.ndarray) -> float:
        """Compute L = mean_CE + (lambda/2)*||W||^2."""
        ce = cross_entropy_loss(P, y_int)
        reg = (self.l2_lambda / 2.0) * np.sum(self.W ** 2)
        return ce + reg

    def _gradients(
        self, X: np.ndarray, P: np.ndarray, y_int: np.ndarray
    ):
        """Analytical gradients of the regularized objective.

        dL/dW = (1/N) * X.T @ (P - Y_onehot) + lambda * W
        dL/db = (1/N) * sum(P - Y_onehot, axis=0)
        """
        N = len(y_int)
        Y = one_hot(y_int, self.n_classes)
        delta = P - Y
        dW = (X.T @ delta) / N + self.l2_lambda * self.W
        db = delta.mean(axis=0)
        return dW, db


    # Training
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        classes: Optional[np.ndarray] = None,
    ) -> "SoftmaxRegression":
        """Train the model using mini-batch SGD with early stopping.

        Parameters
        X_train  : (N_train, D) feature matrix.
        y_train  : (N_train,) integer labels in {0, ..., K-1}.
        X_val    : (N_val, D) validation features.
        y_val    : (N_val,) validation integer labels.
        classes  : optional array of original class-name strings.

        Returns
        self
        """
        rng = np.random.default_rng(self.random_state)
        D = X_train.shape[1]
        K = self.n_classes

        # Xavier/Glorot-style initialisation
        scale = np.sqrt(2.0 / (D + K))
        self.W = rng.standard_normal((D, K)) * scale
        self.b = np.zeros(K)
        self.classes_ = classes

        best_val_loss = np.inf
        best_W = self.W.copy()
        best_b = self.b.copy()
        wait = 0

        self.train_losses_ = []
        self.train_ce_losses_ = []
        self.val_losses_ = []

        N = len(y_train)

        for epoch in range(self.max_epochs):
            # Shuffle training data each epoch
            perm = rng.permutation(N)
            X_shuf = X_train[perm]
            y_shuf = y_train[perm]

            # Mini-batch SGD updates
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                Xb = X_shuf[start:end]
                yb = y_shuf[start:end]
                Pb = softmax(Xb @ self.W + self.b)
                dW, db = self._gradients(Xb, Pb, yb)
                self.W -= self.learning_rate * dW
                self.b -= self.learning_rate * db

            # Epoch-level losses
            P_train = self._predict_proba_raw(X_train)
            train_ce = cross_entropy_loss(P_train, y_train)
            train_loss = train_ce + (self.l2_lambda / 2.0) * np.sum(self.W ** 2)
            self.train_losses_.append(train_loss)
            self.train_ce_losses_.append(train_ce)

            P_val = self._predict_proba_raw(X_val)
            val_loss = cross_entropy_loss(P_val, y_val)
            self.val_losses_.append(val_loss)

            # Early stopping on validation CE
            if val_loss < best_val_loss - 1e-9:
                best_val_loss = val_loss
                best_W = self.W.copy()
                best_b = self.b.copy()
                self.best_epoch_ = epoch
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        # Restore weights from the best epoch
        self.W = best_W
        self.b = best_b
        return self


    # Inference
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (N, K) class probability matrix"""
        self._check_fitted()
        return self._predict_proba_raw(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """return (N,) integer class predictions"""
        self._check_fitted()
        return np.argmax(self._predict_proba_raw(X), axis=1)

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """Return (N,) string class predictions using stored class names"""
        self._check_fitted()
        if self.classes_ is None:
            raise RuntimeError("classes_ not set; pass 'classes' to fit().")
        return self.classes_[self.predict(X)]


    # Serialisation
    def save(self, artifact_dir) -> None:
        """Save model parameters and metadata to artifact_dir"""
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            artifact_dir / "model_weights.npz",
            W=self.W,
            b=self.b,
            classes=np.array(self.classes_) if self.classes_ is not None else np.array([]),
            train_losses=np.array(self.train_losses_),
            train_ce_losses=np.array(self.train_ce_losses_),
            val_losses=np.array(self.val_losses_),
        )

        meta = {
            "hyperparameters": {
                "n_classes": self.n_classes,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "l2_lambda": self.l2_lambda,
                "patience": self.patience,
                "random_state": self.random_state,
            },
            "training_summary": {
                "best_epoch": int(self.best_epoch_),
                "actual_stop_epoch": len(self.train_losses_) - 1,
                "best_val_loss": float(min(self.val_losses_)) if self.val_losses_ else None,
            },
        }
        with open(artifact_dir / "model_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, artifact_dir) -> "SoftmaxRegression":
        """Load a saved model from artifact_dir."""
        artifact_dir = Path(artifact_dir)
        data = np.load(artifact_dir / "model_weights.npz", allow_pickle=True)
        with open(artifact_dir / "model_meta.json") as f:
            meta = json.load(f)
        hp = meta["hyperparameters"]

        model = cls(
            n_classes=hp["n_classes"],
            learning_rate=hp["learning_rate"],
            batch_size=hp["batch_size"],
            max_epochs=hp["max_epochs"],
            l2_lambda=hp["l2_lambda"],
            patience=hp["patience"],
            random_state=hp["random_state"],
        )
        model.W = data["W"]
        model.b = data["b"]
        classes_arr = data["classes"]
        model.classes_ = classes_arr if len(classes_arr) > 0 else None
        model.train_losses_ = data["train_losses"].tolist()
        model.train_ce_losses_ = (
            data["train_ce_losses"].tolist() if "train_ce_losses" in data else []
        )
        model.val_losses_ = data["val_losses"].tolist()
        model.best_epoch_ = meta["training_summary"]["best_epoch"]
        return model


    # Internal helpers
    def _check_fitted(self) -> None:
        if self.W is None or self.b is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
