"""
extractor.py
============

Feature extraction orchestration for Step 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from src.features.hog import HOGConfig, hog_descriptor, hog_length
from src.features.zoning import zoning_features


@dataclass
class FeatureConfig:
    """Configuration for the combined character feature vector."""

    hog: HOGConfig = field(default_factory=HOGConfig)
    zoning_grid: tuple[int, int] = (4, 4)
    include_hog: bool = True
    include_zoning: bool = True


def extract_character_features(
    image: np.ndarray,
    config: FeatureConfig | None = None,
) -> np.ndarray:
    """
    Extract the combined HOG + zoning descriptor for one character.
    """
    cfg = config or FeatureConfig()
    pieces: list[np.ndarray] = []
    if cfg.include_hog:
        pieces.append(hog_descriptor(image, cfg.hog))
    if cfg.include_zoning:
        pieces.append(zoning_features(image, cfg.zoning_grid))
    if not pieces:
        raise ValueError("At least one feature family must be enabled.")
    return np.concatenate(pieces).astype(np.float32)


def extract_batch_features(
    images: Iterable[np.ndarray],
    config: FeatureConfig | None = None,
) -> np.ndarray:
    """
    Extract features for a batch of character images.

    Returns an ``(N, D)`` float32 matrix.  Empty input returns a
    ``(0, 0)`` matrix.
    """
    vectors = [extract_character_features(img, config) for img in images]
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(vectors).astype(np.float32)


def feature_length(
    image_shape: tuple[int, int] = (32, 32),
    config: FeatureConfig | None = None,
) -> int:
    """Return the combined feature length for a character shape."""
    cfg = config or FeatureConfig()
    length = 0
    if cfg.include_hog:
        length += hog_length(image_shape, cfg.hog)
    if cfg.include_zoning:
        length += cfg.zoning_grid[0] * cfg.zoning_grid[1]
    if length == 0:
        raise ValueError("At least one feature family must be enabled.")
    return length
