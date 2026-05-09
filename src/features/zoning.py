"""
zoning.py
=========

Simple zoning features for binary character images.

The image is split into an evenly spaced grid and each zone stores the
foreground density.  Zoning is deliberately primitive, but it captures
coarse shape layout (top-heavy, left stroke, bottom loop...) and
complements HOG well.
"""

from __future__ import annotations

import numpy as np


def zoning_features(
    image: np.ndarray,
    grid: tuple[int, int] = (4, 4),
    foreground_threshold: int = 127,
) -> np.ndarray:
    """
    Compute foreground density in each grid cell.

    Returns cells in row-major order as a ``float32`` vector in
    ``[0, 1]``.
    """
    if image.ndim != 2:
        raise ValueError(f"zoning_features expects a 2-D image; got shape {image.shape}.")
    rows, cols = grid
    if rows <= 0 or cols <= 0:
        raise ValueError(f"grid dimensions must be positive; got {grid}.")

    H, W = image.shape
    ys = np.linspace(0, H, rows + 1, dtype=np.int32)
    xs = np.linspace(0, W, cols + 1, dtype=np.int32)
    fg = image > foreground_threshold

    feats: list[float] = []
    for r in range(rows):
        for c in range(cols):
            zone = fg[ys[r] : ys[r + 1], xs[c] : xs[c + 1]]
            feats.append(float(zone.mean()) if zone.size else 0.0)
    return np.asarray(feats, dtype=np.float32)
