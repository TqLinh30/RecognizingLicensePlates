"""
hog.py
======

Histogram of Oriented Gradients (HOG), implemented from scratch.

HOG describes a glyph by the distribution of local edge directions.
For license-plate characters this is a strong hand-crafted feature:
digits and letters differ mostly by stroke geometry, not by absolute
brightness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HOGConfig:
    """Configuration for :func:`hog_descriptor`."""

    cell_size: int = 8
    block_size: int = 2
    num_bins: int = 9
    epsilon: float = 1e-6


def hog_descriptor(
    image: np.ndarray,
    config: HOGConfig | None = None,
) -> np.ndarray:
    """
    Compute an unsigned-orientation HOG descriptor.

    Parameters
    ----------
    image : np.ndarray
        2-D character image.  Binary ``uint8`` images are ideal, but any
        numeric grayscale array is accepted.
    config : HOGConfig, optional
        Cell, block, and histogram settings.

    Returns
    -------
    np.ndarray
        1-D ``float32`` descriptor.
    """
    if image.ndim != 2:
        raise ValueError(f"hog_descriptor expects a 2-D image; got shape {image.shape}.")

    cfg = config or HOGConfig()
    if cfg.cell_size <= 0 or cfg.block_size <= 0 or cfg.num_bins <= 0:
        raise ValueError("cell_size, block_size, and num_bins must be positive.")

    img = image.astype(np.float32) / 255.0
    gx, gy = _gradient_xy(img)
    magnitude = np.sqrt(gx * gx + gy * gy)
    orientation = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0

    cell_hist = _cell_histograms(
        magnitude,
        orientation,
        cell_size=cfg.cell_size,
        num_bins=cfg.num_bins,
    )
    return _normalize_blocks(
        cell_hist,
        block_size=cfg.block_size,
        epsilon=cfg.epsilon,
    )


def hog_length(
    image_shape: tuple[int, int],
    config: HOGConfig | None = None,
) -> int:
    """Return descriptor length for an image shape and HOG config."""
    cfg = config or HOGConfig()
    h, w = image_shape
    cells_y = h // cfg.cell_size
    cells_x = w // cfg.cell_size
    if cells_y < cfg.block_size or cells_x < cfg.block_size:
        return cells_y * cells_x * cfg.num_bins
    blocks_y = cells_y - cfg.block_size + 1
    blocks_x = cells_x - cfg.block_size + 1
    return blocks_y * blocks_x * cfg.block_size * cfg.block_size * cfg.num_bins


def _gradient_xy(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference gradients with edge padding."""
    padded = np.pad(image, 1, mode="edge")
    gx = padded[1:-1, 2:] - padded[1:-1, :-2]
    gy = padded[2:, 1:-1] - padded[:-2, 1:-1]
    return gx.astype(np.float32), gy.astype(np.float32)


def _cell_histograms(
    magnitude: np.ndarray,
    orientation: np.ndarray,
    cell_size: int,
    num_bins: int,
) -> np.ndarray:
    """Build per-cell orientation histograms with bin interpolation."""
    H, W = magnitude.shape
    cells_y = H // cell_size
    cells_x = W // cell_size
    hist = np.zeros((cells_y, cells_x, num_bins), dtype=np.float32)
    if cells_y == 0 or cells_x == 0:
        return hist

    bin_width = 180.0 / num_bins
    for cy in range(cells_y):
        y0 = cy * cell_size
        y1 = y0 + cell_size
        for cx in range(cells_x):
            x0 = cx * cell_size
            x1 = x0 + cell_size
            mag = magnitude[y0:y1, x0:x1].ravel()
            ang = orientation[y0:y1, x0:x1].ravel()

            bin_pos = ang / bin_width
            lower = np.floor(bin_pos).astype(np.int32) % num_bins
            upper = (lower + 1) % num_bins
            upper_weight = bin_pos - np.floor(bin_pos)
            lower_weight = 1.0 - upper_weight

            np.add.at(hist[cy, cx], lower, mag * lower_weight)
            np.add.at(hist[cy, cx], upper, mag * upper_weight)

    return hist


def _normalize_blocks(
    cell_hist: np.ndarray,
    block_size: int,
    epsilon: float,
) -> np.ndarray:
    """
    Concatenate L2-normalized overlapping blocks.

    If the image is too small for one full block, fall back to a single
    normalized vector of all cell histograms.  This keeps the API usable
    for tiny debug images.
    """
    cells_y, cells_x, _ = cell_hist.shape
    if cells_y == 0 or cells_x == 0:
        return np.zeros(0, dtype=np.float32)

    blocks: list[np.ndarray] = []
    if cells_y < block_size or cells_x < block_size:
        flat = cell_hist.ravel()
        norm = np.sqrt(np.sum(flat * flat) + epsilon * epsilon)
        return (flat / norm).astype(np.float32)

    for by in range(cells_y - block_size + 1):
        for bx in range(cells_x - block_size + 1):
            block = cell_hist[by : by + block_size, bx : bx + block_size].ravel()
            norm = np.sqrt(np.sum(block * block) + epsilon * epsilon)
            blocks.append(block / norm)

    return np.concatenate(blocks).astype(np.float32)
