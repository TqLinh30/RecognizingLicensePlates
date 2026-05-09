"""
tests/test_preprocessing.py
===========================

Unit tests for every sub-module of Step-1 preprocessing.

Run with::

    pytest tests/

The tests use only synthetic NumPy arrays so they don't require any
sample images on disk.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing import (
    rgb_to_grayscale,
    gaussian_blur,
    median_filter,
    compute_histogram,
    compute_cdf,
    histogram_equalization,
    clahe,
    otsu_threshold,
    fixed_threshold,
    preprocess,
)
from src.preprocessing.pipeline import PreprocessConfig


# ---------------------------------------------------------------------------
# rgb_to_grayscale
# ---------------------------------------------------------------------------

class TestGrayscale:

    def test_shape_changes_from_3d_to_2d(self):
        rgb = np.zeros((10, 20, 3), dtype=np.uint8)
        gray = rgb_to_grayscale(rgb)
        assert gray.shape == (10, 20)
        assert gray.dtype == np.uint8

    def test_grayscale_input_is_returned_unchanged(self):
        # Idempotent: calling on an already-2-D image must not error.
        gray = np.full((5, 5), 123, dtype=np.uint8)
        out = rgb_to_grayscale(gray)
        assert np.array_equal(out, gray)

    def test_pure_white_stays_white(self):
        rgb = np.full((4, 4, 3), 255, dtype=np.uint8)
        assert np.all(rgb_to_grayscale(rgb) == 255)

    def test_pure_black_stays_black(self):
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        assert np.all(rgb_to_grayscale(rgb) == 0)

    def test_luminance_weights(self):
        # A pure red image should map to round(0.299 * 255) = 76.
        red = np.zeros((1, 1, 3), dtype=np.uint8)
        red[0, 0] = (255, 0, 0)
        assert rgb_to_grayscale(red)[0, 0] == 76

        # A pure green image should map to round(0.587 * 255) = 150.
        green = np.zeros((1, 1, 3), dtype=np.uint8)
        green[0, 0] = (0, 255, 0)
        assert rgb_to_grayscale(green)[0, 0] == 150

        # A pure blue image should map to round(0.114 * 255) = 29.
        blue = np.zeros((1, 1, 3), dtype=np.uint8)
        blue[0, 0] = (0, 0, 255)
        assert rgb_to_grayscale(blue)[0, 0] == 29

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            rgb_to_grayscale(np.zeros((4, 4, 4), dtype=np.uint8))


# ---------------------------------------------------------------------------
# gaussian_blur
# ---------------------------------------------------------------------------

class TestGaussianBlur:

    def test_constant_image_unchanged(self):
        # A flat image must remain flat (unit DC gain).
        img = np.full((20, 20), 100, dtype=np.uint8)
        out = gaussian_blur(img, kernel_size=5, sigma=1.0)
        assert np.all(out == 100)

    def test_output_shape_preserved(self):
        img = np.random.randint(0, 256, (40, 60), dtype=np.uint8)
        out = gaussian_blur(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_blur_reduces_high_frequency(self):
        # Single bright spike on a black background.
        img = np.zeros((11, 11), dtype=np.uint8)
        img[5, 5] = 255
        out = gaussian_blur(img, kernel_size=5, sigma=1.0)
        # The peak should drop and the surrounding pixels should rise.
        assert out[5, 5] < 255
        assert out[5, 4] > 0
        # Energy is approximately conserved (within rounding).
        assert abs(int(out.sum()) - int(img.sum())) < img.size

    def test_invalid_kernel_size_raises(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError):
            gaussian_blur(img, kernel_size=4)
        with pytest.raises(ValueError):
            gaussian_blur(img, kernel_size=1)

    def test_3d_input_raises(self):
        with pytest.raises(ValueError):
            gaussian_blur(np.zeros((5, 5, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# median_filter
# ---------------------------------------------------------------------------

class TestMedianFilter:

    def test_removes_salt_and_pepper(self):
        # Uniform mid-gray image with a single white spike.
        img = np.full((9, 9), 100, dtype=np.uint8)
        img[4, 4] = 255
        out = median_filter(img, kernel_size=3)
        # The 3x3 neighbourhood is [100]*8 + [255], median = 100.
        assert out[4, 4] == 100

    def test_constant_image_unchanged(self):
        img = np.full((10, 10), 50, dtype=np.uint8)
        assert np.all(median_filter(img) == 50)

    def test_invalid_kernel_size_raises(self):
        with pytest.raises(ValueError):
            median_filter(np.zeros((5, 5), dtype=np.uint8), kernel_size=2)


# ---------------------------------------------------------------------------
# Histogram & equalization
# ---------------------------------------------------------------------------

class TestHistogram:

    def test_histogram_counts(self):
        img = np.array([[0, 1, 1], [2, 2, 2]], dtype=np.uint8)
        hist = compute_histogram(img)
        assert hist[0] == 1
        assert hist[1] == 2
        assert hist[2] == 3
        assert hist[3:].sum() == 0
        assert hist.shape == (256,)

    def test_cdf_is_monotonic(self):
        img = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        cdf = compute_cdf(compute_histogram(img))
        assert np.all(np.diff(cdf) >= 0)
        assert cdf[-1] == img.size

    def test_equalization_preserves_shape_and_dtype(self):
        img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        out = histogram_equalization(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_equalization_widens_dynamic_range(self):
        # Image whose values are squashed into the lower half.
        img = np.random.randint(50, 100, (64, 64), dtype=np.uint8)
        out = histogram_equalization(img)
        # After equalization the output should reach values much higher
        # than the original maximum of 99.
        assert out.max() > 200
        assert out.min() == 0

    def test_constant_image_unchanged(self):
        img = np.full((10, 10), 128, dtype=np.uint8)
        out = histogram_equalization(img)
        assert np.all(out == 128)


# ---------------------------------------------------------------------------
# CLAHE
# ---------------------------------------------------------------------------

class TestCLAHE:

    def test_output_shape_preserved(self):
        img = np.random.randint(0, 256, (40, 60), dtype=np.uint8)
        out = clahe(img, tile_grid_size=(4, 4), clip_limit=2.0)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_constant_image_maps_to_constant(self):
        # A flat image has no contrast to enhance.  Since the histogram
        # is a single spike, the CDF is 1.0 at that intensity and the
        # equalization formula maps it to 255 — every pixel gets the
        # same value, which is the only sensible behaviour: there is
        # nothing to *separate*.  We just check uniformity.
        img = np.full((32, 32), 128, dtype=np.uint8)
        out = clahe(img)
        assert len(np.unique(out)) == 1, (
            "Output of CLAHE on a constant input must be constant."
        )

    def test_low_contrast_image_is_brightened(self):
        # Gradient confined to mid-grays.
        img = np.tile(np.linspace(80, 120, 64, dtype=np.uint8), (64, 1))
        out = clahe(img, tile_grid_size=(2, 2), clip_limit=4.0)
        # CLAHE should expand the dynamic range.
        assert (out.max() - out.min()) > (img.max() - img.min())


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------

class TestThresholding:

    def test_fixed_threshold_basic(self):
        img = np.array([[0, 100], [128, 200]], dtype=np.uint8)
        out = fixed_threshold(img, threshold=128)
        # Strict greater-than: 128 is NOT above 128.
        expected = np.array([[0, 0], [0, 255]], dtype=np.uint8)
        assert np.array_equal(out, expected)

    def test_fixed_threshold_inverted(self):
        img = np.array([[0, 100], [128, 200]], dtype=np.uint8)
        out = fixed_threshold(img, threshold=128, invert=True)
        expected = np.array([[255, 255], [255, 0]], dtype=np.uint8)
        assert np.array_equal(out, expected)

    def test_otsu_finds_clear_separation(self):
        # Bimodal image: two well-separated clusters at 50 and 200.
        img = np.zeros((20, 20), dtype=np.uint8)
        img[:10] = 50
        img[10:] = 200
        binary, t = otsu_threshold(img)
        # For a perfectly bimodal histogram every threshold in
        # [50, 199] produces the same between-class variance, so the
        # exact tie-break depends on argmax.  All of those choices
        # correctly separate the two clusters.
        assert 50 <= t < 200
        # Half the pixels should be foreground, half background.
        assert binary[:10].sum() == 0
        assert binary[10:].sum() == 10 * 20 * 255

    def test_otsu_returns_threshold_value(self):
        img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        _, t = otsu_threshold(img)
        assert 0 <= t <= 255


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

class TestPipeline:

    def test_pipeline_runs_on_rgb(self):
        rgb = np.random.randint(0, 256, (60, 80, 3), dtype=np.uint8)
        result = preprocess(rgb)
        assert result.grayscale.shape == (60, 80)
        assert result.blurred.shape == (60, 80)
        assert result.enhanced.shape == (60, 80)
        assert result.binary.shape == (60, 80)
        # Binary output must contain only 0 or 255.
        unique_values = np.unique(result.binary)
        assert set(unique_values.tolist()).issubset({0, 255})

    def test_pipeline_runs_on_grayscale(self):
        gray = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        result = preprocess(gray)
        assert result.binary.shape == (50, 50)

    def test_custom_config_is_respected(self):
        rgb = np.zeros((30, 30, 3), dtype=np.uint8)
        cfg = PreprocessConfig(blur_kernel_size=5, blur_sigma=2.0,
                               clahe_grid=(4, 4), clahe_clip_limit=3.0,
                               otsu_invert=False)
        result = preprocess(rgb, cfg)
        assert result.config is cfg
