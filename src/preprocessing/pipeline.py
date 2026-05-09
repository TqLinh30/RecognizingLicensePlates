"""
pipeline.py
===========

Orchestrates the preprocessing sub-modules into a single function.

This is the canonical entry point used by the rest of the recognition
system.  Each call returns a dictionary of intermediate results so the
caller can either:
* take only the final binary image (``"binary"``), or
* inspect every stage when debugging.

Default pipeline
----------------
::

    RGB image
       │ rgb_to_grayscale
       ▼
    grayscale
       │ gaussian_blur (3×3, σ=1.0)
       ▼
    blurred
       │ clahe (8×8 tiles, clip=2.0)
       ▼
    enhanced
       │ otsu_threshold (invert=True)
       ▼
    binary  (characters white, background black)

The defaults reflect the analysis from the project plan and work well
for typical license-plate photos.  All parameters can be overridden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.preprocessing.grayscale import rgb_to_grayscale
from src.preprocessing.gaussian_blur import gaussian_blur
from src.preprocessing.clahe import clahe
from src.preprocessing.thresholding import otsu_threshold


# ---------------------------------------------------------------------------
# Configuration object
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    """
    Tunable parameters for the preprocessing pipeline.

    Attributes
    ----------
    blur_kernel_size, blur_sigma
        Gaussian-blur kernel size (odd) and sigma.
    clahe_grid
        Number of tiles (rows, cols) for CLAHE.
    clahe_clip_limit
        Clip multiplier for CLAHE.
    otsu_invert
        If True, the binary output has dark characters become 255
        (white).  Recommended for license plates so that subsequent
        morphological operations treat the characters as foreground.
    """
    blur_kernel_size: int = 3
    blur_sigma: float = 1.0
    clahe_grid: tuple[int, int] = (8, 8)
    clahe_clip_limit: float = 2.0
    otsu_invert: bool = True


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PreprocessResult:
    """All intermediate stages produced by :func:`preprocess`."""
    grayscale: np.ndarray
    blurred: np.ndarray
    enhanced: np.ndarray
    binary: np.ndarray
    otsu_threshold_value: int
    config: PreprocessConfig = field(default_factory=PreprocessConfig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(
    image: np.ndarray,
    config: Optional[PreprocessConfig] = None,
) -> PreprocessResult:
    """
    Run the full Step-1 preprocessing pipeline.

    Parameters
    ----------
    image : np.ndarray
        Input image, either RGB ``(H, W, 3)`` or grayscale ``(H, W)``,
        dtype ``uint8``.
    config : PreprocessConfig, optional
        Pipeline parameters.  Defaults are tuned for license-plate
        photos.

    Returns
    -------
    PreprocessResult
        Dataclass containing every intermediate image so callers can
        debug, visualize, or feed any one of them into Step 2.
    """
    cfg = config or PreprocessConfig()

    # 1. Grayscale --------------------------------------------------------
    gray = rgb_to_grayscale(image)

    # 2. Noise reduction --------------------------------------------------
    blurred = gaussian_blur(
        gray,
        kernel_size=cfg.blur_kernel_size,
        sigma=cfg.blur_sigma,
    )

    # 3. Adaptive contrast enhancement ------------------------------------
    enhanced = clahe(
        blurred,
        tile_grid_size=cfg.clahe_grid,
        clip_limit=cfg.clahe_clip_limit,
    )

    # 4. Binarization -----------------------------------------------------
    binary, t_value = otsu_threshold(enhanced, invert=cfg.otsu_invert)

    return PreprocessResult(
        grayscale=gray,
        blurred=blurred,
        enhanced=enhanced,
        binary=binary,
        otsu_threshold_value=t_value,
        config=cfg,
    )
