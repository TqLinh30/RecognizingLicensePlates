"""
model_io.py
===========

Small serialization helpers for trained classifiers.

The project keeps model files in NumPy's ``.npz`` format instead of a
pickle.  That makes saved MLP models portable, inspectable, and safer to
load because the file contains arrays and JSON metadata rather than
arbitrary Python objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np

from src.classifiers.mlp import MLPClassifier, MLPConfig

PathLike = Union[str, Path]


def save_mlp_model(
    model: MLPClassifier,
    path: PathLike,
    metadata: dict[str, object] | None = None,
) -> None:
    """
    Save a fitted :class:`MLPClassifier` to ``path``.

    Parameters
    ----------
    model : MLPClassifier
        Fitted MLP model.
    path : str | Path
        Destination ``.npz`` file.
    metadata : dict, optional
        Extra JSON-serializable metadata such as dataset name, feature
        length, or training parameters.
    """
    if model.classes_ is None or not model.weights_ or not model.biases_:
        raise ValueError("Cannot save an unfitted MLPClassifier.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "model_type": "MLPClassifier",
        "hidden_sizes": list(model.config.hidden_sizes),
        "learning_rate": model.config.learning_rate,
        "epochs": model.config.epochs,
        "batch_size": model.config.batch_size,
        "l2": model.config.l2,
        "seed": model.config.seed,
        "num_layers": len(model.weights_),
    }
    if metadata:
        meta.update(metadata)

    arrays: dict[str, np.ndarray] = {
        "classes": model.classes_.astype(str),
        "loss_history": np.asarray(model.loss_history_, dtype=np.float32),
        "metadata": np.asarray(json.dumps(meta, sort_keys=True)),
    }
    for i, (weight, bias) in enumerate(zip(model.weights_, model.biases_)):
        arrays[f"weight_{i}"] = weight.astype(np.float32)
        arrays[f"bias_{i}"] = bias.astype(np.float32)

    np.savez_compressed(path, **arrays)


def load_mlp_model(path: PathLike) -> MLPClassifier:
    """
    Load a fitted :class:`MLPClassifier` from a ``.npz`` file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata.get("model_type") != "MLPClassifier":
            raise ValueError(f"Unsupported model type: {metadata.get('model_type')}")

        num_layers = int(metadata["num_layers"])
        cfg = MLPConfig(
            hidden_sizes=tuple(int(v) for v in metadata["hidden_sizes"]),
            learning_rate=float(metadata.get("learning_rate", 0.01)),
            epochs=max(1, int(metadata.get("epochs", 1))),
            batch_size=max(1, int(metadata.get("batch_size", 32))),
            l2=float(metadata.get("l2", 0.0)),
            seed=int(metadata.get("seed", 42)),
        )
        model = MLPClassifier(cfg)
        model.classes_ = data["classes"].astype(str)
        model.weights_ = [data[f"weight_{i}"].astype(np.float32) for i in range(num_layers)]
        model.biases_ = [data[f"bias_{i}"].astype(np.float32) for i in range(num_layers)]
        model.loss_history_ = data["loss_history"].astype(float).tolist()

    return model
