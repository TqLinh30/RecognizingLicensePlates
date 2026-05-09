"""
visualization.py
================

Debugging helpers for inspecting intermediate pipeline outputs.

These functions are *not* part of the recognition pipeline; they only
exist to make manual inspection during development easier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Union

import numpy as np

from src.utils.image_io import save_image

PathLike = Union[str, Path]


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Promote a grayscale array to a 3-channel RGB array (no copy when possible)."""
    if image.ndim == 2:
        # Stack the same single channel three times so concatenation works.
        return np.stack([image, image, image], axis=-1)
    return image


def save_side_by_side(
    images: Sequence[np.ndarray],
    path: PathLike,
    separator_width: int = 4,
    separator_value: int = 255,
) -> None:
    """
    Save a horizontal strip of images for visual comparison.

    All images are first promoted to RGB and padded vertically so they
    share the maximum height; mismatched widths are kept as-is.

    Parameters
    ----------
    images : sequence of np.ndarray
        Grayscale or RGB images.  Must be non-empty.
    path : str | Path
        Destination file.
    separator_width : int
        Width of the white gutter inserted between images.
    separator_value : int
        Pixel value of the separator (255 = white, 0 = black).
    """
    if len(images) == 0:
        raise ValueError("save_side_by_side requires at least one image.")

    # Normalize every image to 3-channel RGB so we can stack them together.
    rgb_images = [_to_rgb(img) for img in images]

    # Use the tallest image as the canvas height; pad shorter ones with
    # zeros at the bottom.  We pad with zeros (black) on purpose so that
    # the white separators stand out clearly.
    max_height = max(img.shape[0] for img in rgb_images)
    padded: list[np.ndarray] = []
    for img in rgb_images:
        h, w, _ = img.shape
        if h < max_height:
            pad = np.zeros((max_height - h, w, 3), dtype=img.dtype)
            img = np.vstack([img, pad])
        padded.append(img)

    # Build the separator column once and reuse it.
    separator = np.full(
        (max_height, separator_width, 3),
        fill_value=separator_value,
        dtype=np.uint8,
    )

    # Interleave images with separators: [img0, sep, img1, sep, ..., imgN].
    strip_pieces: list[np.ndarray] = []
    for i, img in enumerate(padded):
        if i > 0:
            strip_pieces.append(separator)
        strip_pieces.append(img)

    strip = np.hstack(strip_pieces)
    save_image(strip, path)
