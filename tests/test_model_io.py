"""
tests/test_model_io.py
======================

Tests for saving/loading trained MLP models without pickle.
"""

from __future__ import annotations

import numpy as np

from src.classifiers import MLPClassifier, MLPConfig, load_mlp_model, save_mlp_model


def test_save_and_load_mlp_model_roundtrip(tmp_path):
    X = np.array(
        [
            [-1.0, -1.0],
            [-0.8, -1.2],
            [1.0, 1.0],
            [1.2, 0.8],
        ],
        dtype=np.float32,
    )
    y = np.array(["A", "A", "B", "B"])
    model = MLPClassifier(
        MLPConfig(hidden_sizes=(6,), learning_rate=0.05, epochs=80, batch_size=4, seed=5)
    ).fit(X, y)

    path = tmp_path / "model.npz"
    save_mlp_model(model, path, metadata={"feature_length": 2})
    loaded = load_mlp_model(path)

    assert loaded.classes_.tolist() == model.classes_.tolist()
    assert loaded.predict(X).tolist() == model.predict(X).tolist()
    assert np.allclose(loaded.predict_proba(X), model.predict_proba(X))
