"""
plate_normalizer.py
===================

Step 3 of the ALPR pipeline:

    detected box -> crop with margin -> estimate skew -> rotate -> resize

The output is a canonical plate image, by default ``80x240`` pixels.
This stable geometry makes the segmentation and recognition stages much
simpler because character sizes and spacing become predictable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from src.detection.sobel import sobel
from src.normalization.geometric_transform import (
    crop_with_padding,
    resize_bilinear,
    rotate_image,
)
from src.normalization.hough_transform import estimate_skew_angle
from src.preprocessing.grayscale import rgb_to_grayscale


@dataclass
class NormalizationConfig:
    """Tunable parameters for plate cropping and deskewing."""

    target_shape: tuple[int, int] = (80, 240)  # (height, width)
    margin_ratio: float = 0.10
    fill_value: int = 255
    edge_threshold_ratio: float = 0.25
    hough_angle_limit: float = 15.0
    hough_theta_step: float = 0.5


@dataclass
class NormalizationResult:
    """All intermediate images produced by :func:`normalize_plate`."""

    cropped: np.ndarray
    edge_image: np.ndarray
    deskewed: np.ndarray
    normalized: np.ndarray
    angle_degrees: float
    box: tuple[int, int, int, int]
    config: NormalizationConfig = field(default_factory=NormalizationConfig)


def normalize_plate(
    image: np.ndarray,
    box_or_candidate: Sequence[int] | object,
    config: Optional[NormalizationConfig] = None,
) -> NormalizationResult:
    """
    Crop, deskew, and resize a detected plate region.

    Parameters
    ----------
    image : np.ndarray
        Full RGB or grayscale image.  RGB is converted to grayscale
        because character segmentation works on intensity.
    box_or_candidate : sequence or object
        Either ``(x, y, width, height)`` or an object exposing
        ``as_box()`` such as ``PlateCandidate`` from Step 2.
    config : NormalizationConfig, optional
        Geometry and Hough settings.

    Returns
    -------
    NormalizationResult
        Cropped, edge, deskewed, and final normalized plate images.
    """
    cfg = config or NormalizationConfig()
    box = _coerce_box(box_or_candidate)
    gray = rgb_to_grayscale(image)

    cropped = crop_with_padding(
        gray,
        box,
        margin_ratio=cfg.margin_ratio,
        fill_value=cfg.fill_value,
    )
    edge_image = _make_hough_edge_image(
        cropped,
        threshold_ratio=cfg.edge_threshold_ratio,
    )
    angle = estimate_skew_angle(
        edge_image,
        angle_limit=cfg.hough_angle_limit,
        theta_step=cfg.hough_theta_step,
    )
    deskewed = rotate_image(cropped, -angle, fill_value=cfg.fill_value)
    normalized = resize_bilinear(
        deskewed,
        target_shape=cfg.target_shape,
        fill_value=cfg.fill_value,
    )

    return NormalizationResult(
        cropped=cropped,
        edge_image=edge_image,
        deskewed=deskewed,
        normalized=normalized,
        angle_degrees=angle,
        box=box,
        config=cfg,
    )


def _make_hough_edge_image(
    gray: np.ndarray,
    threshold_ratio: float,
) -> np.ndarray:
    """
    Convert a crop into a sparse edge map for Hough voting.

    Full Sobel magnitude is used here, not only Sobel-X, because plate
    borders are mostly horizontal.  The detector used Sobel-X for
    locating plates; the normalizer cares about top/bottom borders.
    """
    if not 0.0 <= threshold_ratio <= 1.0:
        raise ValueError(
            f"threshold_ratio must be in [0, 1]; got {threshold_ratio}."
        )
    magnitude = sobel(gray).magnitude_uint8
    peak = int(magnitude.max())
    if peak == 0:
        return np.zeros_like(magnitude, dtype=np.uint8)
    threshold = int(round(threshold_ratio * peak))
    return ((magnitude > threshold).astype(np.uint8)) * 255


def _coerce_box(box_or_candidate: Sequence[int] | object) -> tuple[int, int, int, int]:
    """Accept either a tuple/list box or a Step-2 PlateCandidate."""
    if hasattr(box_or_candidate, "as_box"):
        box = box_or_candidate.as_box()  # type: ignore[attr-defined]
    else:
        box = box_or_candidate
    if len(box) != 4:  # type: ignore[arg-type]
        raise ValueError("box must contain exactly four values: x, y, width, height.")
    x, y, w, h = [int(v) for v in box]  # type: ignore[assignment]
    return (x, y, w, h)
