"""
mlp.py
======

A small multi-layer perceptron classifier implemented with NumPy only.

Architecture
------------
The default network is:

    input -> dense(128) + ReLU -> dense(64) + ReLU -> dense(num_classes)
          -> softmax

Training uses mini-batch gradient descent with cross-entropy loss.
The implementation is intentionally explicit: every forward and
backward step is visible so the model remains educational.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class MLPConfig:
    """Hyperparameters for :class:`MLPClassifier`."""

    hidden_sizes: tuple[int, ...] = (128, 64)
    learning_rate: float = 0.01
    epochs: int = 50
    batch_size: int = 32
    l2: float = 0.0
    seed: int = 42


@dataclass
class MLPClassifier:
    """
    Multi-class MLP with ReLU hidden layers and softmax output.
    """

    config: MLPConfig = field(default_factory=MLPConfig)

    def __post_init__(self) -> None:
        cfg = self.config
        if cfg.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if cfg.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if cfg.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if any(size <= 0 for size in cfg.hidden_sizes):
            raise ValueError("hidden_sizes must contain only positive integers.")

        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []
        self.classes_: np.ndarray | None = None
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: Sequence[str] | np.ndarray) -> "MLPClassifier":
        """Train the network on a feature matrix and string labels."""
        X = _as_feature_matrix(X)
        y_arr = np.asarray(y).astype(str)
        if X.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows; got {X.shape[0]} and {y_arr.shape[0]}."
            )
        if X.shape[0] == 0:
            raise ValueError("MLPClassifier.fit requires at least one sample.")

        self.classes_, y_idx = np.unique(y_arr, return_inverse=True)
        self._initialize(input_dim=X.shape[1], output_dim=self.classes_.shape[0])

        rng = np.random.default_rng(self.config.seed)
        n = X.shape[0]
        for _ in range(self.config.epochs):
            order = rng.permutation(n)
            X_shuffled = X[order]
            y_shuffled = y_idx[order]

            for start in range(0, n, self.config.batch_size):
                end = min(n, start + self.config.batch_size)
                xb = X_shuffled[start:end]
                yb = y_shuffled[start:end]
                activations, pre_activations = self._forward(xb)
                self._backward(activations, pre_activations, yb)

            probs = self.predict_proba(X)
            self.loss_history_.append(_cross_entropy(probs, y_idx))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict one class label per feature vector."""
        proba = self.predict_proba(X)
        assert self.classes_ is not None
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities for every class."""
        self._require_fit()
        X = _as_feature_matrix(X)
        if X.shape[1] != self.weights_[0].shape[0]:
            raise ValueError(
                f"Expected {self.weights_[0].shape[0]} features, got {X.shape[1]}."
            )
        activations, _ = self._forward(X)
        return activations[-1]

    def _initialize(self, input_dim: int, output_dim: int) -> None:
        """Initialize dense layers with He-style scaling."""
        rng = np.random.default_rng(self.config.seed)
        dims = [input_dim, *self.config.hidden_sizes, output_dim]
        self.weights_ = []
        self.biases_ = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2.0 / fan_in)
            self.weights_.append((rng.normal(0.0, scale, (fan_in, fan_out))).astype(np.float32))
            self.biases_.append(np.zeros(fan_out, dtype=np.float32))

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return activations and pre-activations for every layer."""
        activations = [X.astype(np.float32)]
        pre_activations: list[np.ndarray] = []

        a = activations[0]
        for layer_idx, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            z = a @ W + b
            pre_activations.append(z)
            if layer_idx == len(self.weights_) - 1:
                a = _softmax(z)
            else:
                a = np.maximum(z, 0.0)
            activations.append(a.astype(np.float32))

        return activations, pre_activations

    def _backward(
        self,
        activations: list[np.ndarray],
        pre_activations: list[np.ndarray],
        y_idx: np.ndarray,
    ) -> None:
        """Backpropagate cross-entropy gradients and update weights."""
        batch_size = y_idx.shape[0]
        dz = activations[-1].copy()
        dz[np.arange(batch_size), y_idx] -= 1.0
        dz /= batch_size

        lr = self.config.learning_rate
        for layer_idx in reversed(range(len(self.weights_))):
            a_prev = activations[layer_idx]
            W = self.weights_[layer_idx]

            dW = a_prev.T @ dz + self.config.l2 * W
            db = np.sum(dz, axis=0)

            if layer_idx > 0:
                da_prev = dz @ W.T
                dz = da_prev * (pre_activations[layer_idx - 1] > 0)

            self.weights_[layer_idx] = (W - lr * dW).astype(np.float32)
            self.biases_[layer_idx] = (self.biases_[layer_idx] - lr * db).astype(np.float32)

    def _require_fit(self) -> None:
        if self.classes_ is None or not self.weights_ or not self.biases_:
            raise ValueError("Classifier has not been fitted yet.")


def _as_feature_matrix(X: np.ndarray) -> np.ndarray:
    """Normalize input to a 2-D float32 feature matrix."""
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 1-D or 2-D feature matrix; got shape {arr.shape}.")
    return arr


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return (exp_z / np.sum(exp_z, axis=1, keepdims=True)).astype(np.float32)


def _cross_entropy(probs: np.ndarray, y_idx: np.ndarray) -> float:
    """Mean negative log likelihood."""
    eps = 1e-8
    correct = probs[np.arange(y_idx.shape[0]), y_idx]
    return float(-np.mean(np.log(np.clip(correct, eps, 1.0))))
