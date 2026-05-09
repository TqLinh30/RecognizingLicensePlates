"""
tests/test_classifiers.py
=========================

Unit tests for Step 6 KNN and MLP classifiers.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.classifiers import KNNClassifier, MLPClassifier, MLPConfig


class TestKNNClassifier:

    def test_knn_predicts_nearest_label(self):
        X = np.array([[0, 0], [0, 1], [5, 5], [6, 5]], dtype=np.float32)
        y = np.array(["A", "A", "B", "B"])

        clf = KNNClassifier(k=1).fit(X, y)
        pred = clf.predict(np.array([[0.2, 0.1], [5.5, 5.1]], dtype=np.float32))

        assert pred.tolist() == ["A", "B"]

    def test_knn_probabilities_sum_to_one(self):
        X = np.array([[0, 0], [1, 0], [10, 0]], dtype=np.float32)
        y = np.array(["A", "A", "B"])
        clf = KNNClassifier(k=3).fit(X, y)

        proba = clf.predict_proba(np.array([[0, 0]], dtype=np.float32))

        assert proba.shape == (1, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_knn_rejects_unfitted_predict(self):
        with pytest.raises(ValueError):
            KNNClassifier().predict(np.zeros((1, 2), dtype=np.float32))


class TestMLPClassifier:

    def test_mlp_learns_simple_separable_clusters(self):
        X = np.array(
            [
                [-1.0, -1.0],
                [-1.2, -0.8],
                [1.0, 1.0],
                [1.2, 0.8],
            ],
            dtype=np.float32,
        )
        y = np.array(["A", "A", "B", "B"])
        cfg = MLPConfig(hidden_sizes=(8,), learning_rate=0.05, epochs=200, batch_size=4, seed=7)

        clf = MLPClassifier(cfg).fit(X, y)
        pred = clf.predict(X)

        assert pred.tolist() == y.tolist()
        assert clf.loss_history_[0] > clf.loss_history_[-1]

    def test_mlp_probabilities_sum_to_one(self):
        X = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
        y = np.array(["A", "B", "B"])
        clf = MLPClassifier(MLPConfig(hidden_sizes=(4,), epochs=10, batch_size=3)).fit(X, y)

        proba = clf.predict_proba(np.array([[1.5]], dtype=np.float32))

        assert proba.shape == (1, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_mlp_rejects_bad_hyperparameters(self):
        with pytest.raises(ValueError):
            MLPClassifier(MLPConfig(learning_rate=0.0))
