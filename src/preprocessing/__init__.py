"""
Preprocessing module (Pipeline Step 1)
======================================

Implements the first stage of the license-plate recognition pipeline:
turning a raw color photograph into a clean, contrast-enhanced
single-channel image suitable for edge-based detection in Step 2.

Sub-modules
-----------
grayscale       RGB → grayscale conversion (luminance formula).
gaussian_blur   Smoothing with a separable Gaussian kernel.
median_filter   Edge-preserving denoising for salt-and-pepper noise.
histogram       Histogram, CDF, and global histogram equalization.
clahe           Contrast Limited Adaptive Histogram Equalization.
thresholding    Binarization (fixed threshold and Otsu's method).

A convenience ``preprocess`` function wires these together with sensible
defaults and is the typical entry point.
"""

from src.preprocessing.grayscale import rgb_to_grayscale
from src.preprocessing.gaussian_blur import gaussian_blur
from src.preprocessing.median_filter import median_filter
from src.preprocessing.histogram import (
    compute_histogram,
    compute_cdf,
    histogram_equalization,
)
from src.preprocessing.clahe import clahe
from src.preprocessing.thresholding import otsu_threshold, fixed_threshold
from src.preprocessing.pipeline import PreprocessConfig, PreprocessResult, preprocess

__all__ = [
    "rgb_to_grayscale",
    "gaussian_blur",
    "median_filter",
    "compute_histogram",
    "compute_cdf",
    "histogram_equalization",
    "clahe",
    "otsu_threshold",
    "fixed_threshold",
    "PreprocessConfig",
    "PreprocessResult",
    "preprocess",
]
