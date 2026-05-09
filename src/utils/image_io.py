"""
image_io.py
===========

Thin wrappers around Pillow that convert between image files on disk and
NumPy arrays.  Pillow is used **only** for decoding/encoding image files
(PNG, JPEG, BMP, ...).  All pixel-level processing happens elsewhere on
plain NumPy arrays.

Conventions
-----------
* Color images are returned as ``uint8`` arrays of shape ``(H, W, 3)`` in
  RGB order.
* Grayscale images are returned as ``uint8`` arrays of shape ``(H, W)``.
* ``save_image`` accepts either layout and detects it from ``ndim``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

# Type alias for things that can name a file on disk.
PathLike = Union[str, Path]


def load_image(path: PathLike, as_gray: bool = False) -> np.ndarray:
    """
    Load an image file from disk into a NumPy array.

    Parameters
    ----------
    path : str | Path
        Path to the image file.
    as_gray : bool, default False
        If True, the image is decoded directly to single-channel grayscale
        by Pillow.  Most of the time we keep ``False`` and convert to
        grayscale ourselves with our own implementation in
        :mod:`src.preprocessing.grayscale` so the conversion is part of
        the pipeline rather than hidden inside Pillow.

    Returns
    -------
    np.ndarray
        ``uint8`` array of shape ``(H, W, 3)`` for RGB or ``(H, W)`` for
        grayscale.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    # Pillow handles all the format-specific decoding (libjpeg, zlib, ...).
    with Image.open(path) as img:
        # Some images come in palette ("P") or RGBA modes; normalize them
        # so downstream code only ever sees "RGB" or "L".
        if as_gray:
            img = img.convert("L")          # 8-bit luminance
        else:
            img = img.convert("RGB")        # 8-bit per channel, 3 channels

        # np.asarray copies into a contiguous uint8 buffer.
        array = np.asarray(img, dtype=np.uint8)

    return array


def save_image(array: np.ndarray, path: PathLike) -> None:
    """
    Persist a NumPy array as an image file.

    Parameters
    ----------
    array : np.ndarray
        Image data.  Accepted layouts:
        * ``(H, W)``      — grayscale
        * ``(H, W, 3)``   — RGB
        * ``(H, W, 4)``   — RGBA

        ``dtype`` should be ``uint8``.  Floating-point arrays are clipped
        to ``[0, 255]`` and cast for convenience during debugging.

    path : str | Path
        Destination file.  The format is inferred from the extension
        (``.png``, ``.jpg``, ``.bmp``, ...).

    Notes
    -----
    Parent directories are created automatically so callers don't have to
    worry about ``data/output/`` not existing yet.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Defensive cast: anything outside [0, 255] would be wrapped/garbled
    # otherwise.  This matches what most CV libraries do silently.
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    # Pick the Pillow mode from the array shape.
    if array.ndim == 2:
        mode = "L"
    elif array.ndim == 3 and array.shape[2] == 3:
        mode = "RGB"
    elif array.ndim == 3 and array.shape[2] == 4:
        mode = "RGBA"
    else:
        raise ValueError(
            f"Unsupported array shape for image saving: {array.shape}"
        )

    Image.fromarray(array, mode=mode).save(path)
