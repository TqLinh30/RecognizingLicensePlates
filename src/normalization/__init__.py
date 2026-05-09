"""
Normalization module (Pipeline Step 3)
======================================

Takes a detected license-plate candidate and produces a canonical,
deskewed plate crop for segmentation.
"""

from src.normalization.geometric_transform import (
    bilinear_sample,
    crop_with_padding,
    resize_bilinear,
    rotate_image,
)
from src.normalization.hough_transform import (
    HoughLine,
    HoughResult,
    estimate_skew_angle,
    hough_lines,
)
from src.normalization.plate_normalizer import (
    NormalizationConfig,
    NormalizationResult,
    normalize_plate,
)

__all__ = [
    "bilinear_sample",
    "crop_with_padding",
    "resize_bilinear",
    "rotate_image",
    "HoughLine",
    "HoughResult",
    "estimate_skew_angle",
    "hough_lines",
    "NormalizationConfig",
    "NormalizationResult",
    "normalize_plate",
]
