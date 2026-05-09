"""
zoning_template.py
==================

Prototype classifier based on fixed white-pixel regions.

This implements the idea of splitting each normalized character into a
fixed grid and comparing the percentage of foreground pixels in each
region.  It is interpretable, fast, and especially useful as a companion
to the MLP when characters are printed and high-contrast.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np

from src.features.zoning import zoning_features

PathLike = Union[str, Path]


@dataclass
class ZoningTemplateClassifier:
    """Nearest-prototype classifier over grid-density features."""

    grid: tuple[int, int] = (8, 8)
    temperature: float = 0.08
    max_templates_per_class: int = 60
    seed: int = 42

    def __post_init__(self) -> None:
        if self.grid[0] <= 0 or self.grid[1] <= 0:
            raise ValueError(f"grid dimensions must be positive; got {self.grid}.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if self.max_templates_per_class <= 0:
            raise ValueError("max_templates_per_class must be positive.")
        self.classes_: np.ndarray | None = None
        self.prototypes_: np.ndarray | None = None
        self.templates_: np.ndarray | None = None
        self.template_labels_: np.ndarray | None = None

    def fit(self, images: Sequence[np.ndarray] | np.ndarray, y: Sequence[str] | np.ndarray) -> "ZoningTemplateClassifier":
        """Build one mean zoning prototype per class."""
        y_arr = np.asarray(y).astype(str)
        X = self._features_from_images(images)
        if X.shape[0] != y_arr.shape[0]:
            raise ValueError(f"images and labels differ in length: {X.shape[0]} vs {y_arr.shape[0]}.")
        if X.shape[0] == 0:
            raise ValueError("fit requires at least one sample.")

        self.classes_ = np.unique(y_arr)
        rng = np.random.default_rng(self.seed)
        prototypes = []
        templates = []
        template_labels = []
        for cls in self.classes_:
            class_feats = X[y_arr == cls]
            proto = class_feats.mean(axis=0)
            prototypes.append(proto)
            if class_feats.shape[0] > self.max_templates_per_class:
                chosen = rng.choice(class_feats.shape[0], self.max_templates_per_class, replace=False)
                class_templates = class_feats[chosen]
            else:
                class_templates = class_feats
            templates.append(class_templates)
            template_labels.extend([cls] * class_templates.shape[0])
        self.prototypes_ = np.vstack(prototypes).astype(np.float32)
        self.templates_ = np.vstack(templates).astype(np.float32)
        self.template_labels_ = np.asarray(template_labels).astype(str)
        return self

    def predict(self, images: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        """Predict one label per character image."""
        proba = self.predict_proba(images)
        assert self.classes_ is not None
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, images: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        """Return softmax probabilities derived from prototype distance."""
        self._require_fit()
        assert self.prototypes_ is not None
        X = self._features_from_images(images)
        dist2 = self._class_distances(X)
        logits = -dist2 / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return (exp_logits / exp_logits.sum(axis=1, keepdims=True)).astype(np.float32)

    def _features_from_images(self, images: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        """Compute zoning vectors from one image or a batch of images."""
        arr = np.asarray(images)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        if arr.ndim != 3:
            raise ValueError(f"Expected character images shaped (N,H,W); got {arr.shape}.")
        return np.vstack([zoning_features(img, grid=self.grid) for img in arr]).astype(np.float32)

    def _require_fit(self) -> None:
        if self.classes_ is None or self.prototypes_ is None:
            raise ValueError("Classifier has not been fitted yet.")

    def _class_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Distance from every sample to every class.

        If stored templates are available, use the nearest template per
        class.  This handles multiple font styles better than one mean
        prototype.  Older saved models without templates fall back to
        the mean prototype.
        """
        assert self.classes_ is not None and self.prototypes_ is not None
        if self.templates_ is None or self.template_labels_ is None:
            diff = X[:, None, :] - self.prototypes_[None, :, :]
            return np.mean(diff * diff, axis=2)

        out = np.empty((X.shape[0], self.classes_.shape[0]), dtype=np.float32)
        for class_idx, cls in enumerate(self.classes_):
            class_templates = self.templates_[self.template_labels_ == cls]
            diff = X[:, None, :] - class_templates[None, :, :]
            out[:, class_idx] = np.mean(diff * diff, axis=2).min(axis=1)
        return out


def save_zoning_template_model(model: ZoningTemplateClassifier, path: PathLike) -> None:
    """Save a fitted zoning-template classifier to ``.npz``."""
    if model.classes_ is None or model.prototypes_ is None:
        raise ValueError("Cannot save an unfitted ZoningTemplateClassifier.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        classes=model.classes_.astype(str),
        prototypes=model.prototypes_.astype(np.float32),
        grid=np.asarray(model.grid, dtype=np.int32),
        temperature=np.asarray(model.temperature, dtype=np.float32),
        max_templates_per_class=np.asarray(model.max_templates_per_class, dtype=np.int32),
        seed=np.asarray(model.seed, dtype=np.int32),
        templates=np.asarray(model.templates_ if model.templates_ is not None else model.prototypes_, dtype=np.float32),
        template_labels=np.asarray(
            model.template_labels_ if model.template_labels_ is not None else model.classes_,
            dtype=str,
        ),
    )


def load_zoning_template_model(path: PathLike) -> ZoningTemplateClassifier:
    """Load a zoning-template classifier from ``.npz``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        grid_arr = data["grid"].astype(int)
        model = ZoningTemplateClassifier(
            grid=(int(grid_arr[0]), int(grid_arr[1])),
            temperature=float(data["temperature"]),
            max_templates_per_class=int(data["max_templates_per_class"]) if "max_templates_per_class" in data.files else 60,
            seed=int(data["seed"]) if "seed" in data.files else 42,
        )
        model.classes_ = data["classes"].astype(str)
        model.prototypes_ = data["prototypes"].astype(np.float32)
        if "templates" in data.files and "template_labels" in data.files:
            model.templates_ = data["templates"].astype(np.float32)
            model.template_labels_ = data["template_labels"].astype(str)
    return model
