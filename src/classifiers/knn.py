"""
knn.py
======

K-Nearest Neighbours baseline classifier.

KNN is a good first classifier for this project because it has almost
no training phase: it stores feature vectors and predicts by majority
vote among the nearest examples.  That makes it ideal for validating
the preprocessing/segmentation/feature pipeline before investing in
neural-network training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KNNClassifier:
    """Simple Euclidean-distance KNN classifier."""

    k: int = 3

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError(f"k must be positive; got {self.k}.")
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | list[str]) -> "KNNClassifier":
        """Store the training feature matrix and labels."""
        X = _as_feature_matrix(X)
        y_arr = np.asarray(y)
        if X.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows; got {X.shape[0]} and {y_arr.shape[0]}."
            )
        if X.shape[0] == 0:
            raise ValueError("KNNClassifier.fit requires at least one training sample.")

        self.X_train_ = X.astype(np.float32)
        self.y_train_ = y_arr.astype(str)
        self.classes_ = np.unique(self.y_train_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict one label per row of ``X``."""
        proba = self.predict_proba(X)
        assert self.classes_ is not None
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return vote fractions for every class.

        Ties are naturally represented as equal probabilities.  The
        caller can use the maximum probability as a confidence estimate.
        """
        self._require_fit()
        assert self.X_train_ is not None and self.y_train_ is not None and self.classes_ is not None

        X = _as_feature_matrix(X)
        if X.shape[1] != self.X_train_.shape[1]:
            raise ValueError(
                f"Expected {self.X_train_.shape[1]} features, got {X.shape[1]}."
            )

        k_eff = min(self.k, self.X_train_.shape[0])
        out = np.zeros((X.shape[0], self.classes_.shape[0]), dtype=np.float32)

        for i, row in enumerate(X):
            diff = self.X_train_ - row
            dist2 = np.sum(diff * diff, axis=1)
            nn_idx = np.argsort(dist2)[:k_eff]
            nn_labels = self.y_train_[nn_idx]
            for class_idx, cls in enumerate(self.classes_):
                out[i, class_idx] = np.mean(nn_labels == cls)

        return out

    def _require_fit(self) -> None:
        if self.X_train_ is None or self.y_train_ is None or self.classes_ is None:
            raise ValueError("Classifier has not been fitted yet.")


def _as_feature_matrix(X: np.ndarray) -> np.ndarray:
    """Normalize input to a 2-D float32 feature matrix."""
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 1-D or 2-D feature matrix; got shape {arr.shape}.")
    return arr
