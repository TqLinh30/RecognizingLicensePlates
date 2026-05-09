"""
pixel_template.py
=================

Raw-shape template classifier for normalized binary characters.

This is the most direct version of the user's fixed-region idea: after
segmentation every glyph is a 32x32 white-on-black image, so we compare
that binary shape against stored printed-character templates.  It does
not assume any license-plate format; it only answers "which character
does this crop look like?".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np

PathLike = Union[str, Path]


@dataclass
class PixelTemplateClassifier:
    """Nearest-template classifier over 32x32 binary glyph pixels."""

    max_templates_per_class: int = 120
    temperature: float = 0.035
    seed: int = 42

    def __post_init__(self) -> None:
        if self.max_templates_per_class <= 0:
            raise ValueError("max_templates_per_class must be positive.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.classes_: np.ndarray | None = None
        self.templates_: np.ndarray | None = None
        self.template_labels_: np.ndarray | None = None

    def fit(
        self,
        images: Sequence[np.ndarray] | np.ndarray,
        y: Sequence[str] | np.ndarray,
    ) -> "PixelTemplateClassifier":
        """Store a balanced subset of normalized templates per class."""
        X = _as_image_batch(images)
        y_arr = np.asarray(y).astype(str)
        if X.shape[0] != y_arr.shape[0]:
            raise ValueError(f"images and labels differ in length: {X.shape[0]} vs {y_arr.shape[0]}.")
        if X.shape[0] == 0:
            raise ValueError("fit requires at least one sample.")

        rng = np.random.default_rng(self.seed)
        self.classes_ = np.unique(y_arr)
        templates: list[np.ndarray] = []
        labels: list[str] = []
        for cls in self.classes_:
            class_images = X[y_arr == cls]
            if class_images.shape[0] > self.max_templates_per_class:
                chosen = rng.choice(class_images.shape[0], self.max_templates_per_class, replace=False)
                class_images = class_images[chosen]
            templates.append(class_images)
            labels.extend([cls] * class_images.shape[0])

        self.templates_ = np.vstack(templates).astype(np.float32)
        self.template_labels_ = np.asarray(labels).astype(str)
        return self

    def predict(self, images: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        """Predict one label per character image."""
        proba = self.predict_proba(images)
        assert self.classes_ is not None
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, images: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        """Return softmax probabilities from nearest-template distance."""
        self._require_fit()
        assert self.classes_ is not None
        X = _as_image_batch(images)
        dist = self._class_distances(X)
        logits = -dist / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return (exp_logits / np.maximum(exp_logits.sum(axis=1, keepdims=True), 1e-8)).astype(np.float32)

    def _class_distances(self, X: np.ndarray) -> np.ndarray:
        """Minimum mean squared distance to templates of each class."""
        assert self.classes_ is not None and self.templates_ is not None and self.template_labels_ is not None
        flat = X.reshape(X.shape[0], -1)
        template_flat = self.templates_.reshape(self.templates_.shape[0], -1)
        out = np.empty((flat.shape[0], self.classes_.shape[0]), dtype=np.float32)
        for class_idx, cls in enumerate(self.classes_):
            class_templates = template_flat[self.template_labels_ == cls]
            # Mean squared distance without materializing (N,T,D):
            # ||x-t||^2 = ||x||^2 + ||t||^2 - 2 x·t.
            x_norm = np.mean(flat * flat, axis=1, keepdims=True)
            t_norm = np.mean(class_templates * class_templates, axis=1, keepdims=True).T
            cross = (flat @ class_templates.T) / flat.shape[1]
            dist = x_norm + t_norm - 2.0 * cross
            out[:, class_idx] = dist.min(axis=1)
        return out

    def _require_fit(self) -> None:
        if self.classes_ is None or self.templates_ is None or self.template_labels_ is None:
            raise ValueError("Classifier has not been fitted yet.")


def save_pixel_template_model(model: PixelTemplateClassifier, path: PathLike) -> None:
    """Save a fitted pixel-template model to ``.npz``."""
    if model.classes_ is None or model.templates_ is None or model.template_labels_ is None:
        raise ValueError("Cannot save an unfitted PixelTemplateClassifier.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        classes=model.classes_.astype(str),
        templates=model.templates_.astype(np.float32),
        template_labels=model.template_labels_.astype(str),
        max_templates_per_class=np.asarray(model.max_templates_per_class, dtype=np.int32),
        temperature=np.asarray(model.temperature, dtype=np.float32),
        seed=np.asarray(model.seed, dtype=np.int32),
    )


def load_pixel_template_model(path: PathLike) -> PixelTemplateClassifier:
    """Load a pixel-template model from ``.npz``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        model = PixelTemplateClassifier(
            max_templates_per_class=int(data["max_templates_per_class"]),
            temperature=float(data["temperature"]),
            seed=int(data["seed"]),
        )
        model.classes_ = data["classes"].astype(str)
        model.templates_ = data["templates"].astype(np.float32)
        model.template_labels_ = data["template_labels"].astype(str)
    return model


def _as_image_batch(images: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Normalize one or more character images to float32 in [0, 1]."""
    arr = np.asarray(images)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Expected character images shaped (N,H,W); got {arr.shape}.")
    return (arr.astype(np.float32) / 255.0)
