"""
tests/test_normalization.py
===========================

Unit tests for Step 3 plate cropping and normalization.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.normalization import (
    NormalizationConfig,
    crop_with_padding,
    estimate_skew_angle,
    hough_lines,
    normalize_plate,
    resize_bilinear,
    rotate_image,
)


class TestGeometricTransform:

    def test_crop_with_padding_clips_outside_image(self):
        img = np.full((10, 10), 20, dtype=np.uint8)
        img[0:4, 0:4] = 100

        crop = crop_with_padding(
            img,
            box=(0, 0, 4, 4),
            margin_ratio=0.5,
            fill_value=255,
        )

        assert crop.shape == (8, 8)
        assert crop[0, 0] == 255      # outside original image
        assert crop[2, 2] == 100      # original top-left pixel lands after margin

    def test_resize_bilinear_preserves_constant_image(self):
        img = np.full((8, 12), 77, dtype=np.uint8)
        out = resize_bilinear(img, (20, 30))
        assert out.shape == (20, 30)
        assert np.all(out == 77)

    def test_resize_bilinear_preserves_corner_values(self):
        img = np.array(
            [
                [0, 10],
                [20, 30],
            ],
            dtype=np.uint8,
        )
        out = resize_bilinear(img, (5, 5))
        assert out[0, 0] == 0
        assert out[0, -1] == 10
        assert out[-1, 0] == 20
        assert out[-1, -1] == 30

    def test_rotate_zero_degrees_returns_copy(self):
        img = np.random.randint(0, 256, (20, 30), dtype=np.uint8)
        out = rotate_image(img, 0.0)
        assert np.array_equal(out, img)
        assert out is not img


class TestHoughTransform:

    def test_hough_lines_finds_horizontal_line(self):
        edges = np.zeros((60, 120), dtype=np.uint8)
        edges[30, 10:110] = 255

        result = hough_lines(
            edges,
            theta_values=np.arange(85, 96, 1),
            min_votes=50,
            max_lines=1,
        )

        assert len(result.lines) == 1
        assert result.lines[0].theta_degrees == pytest.approx(90, abs=1)
        assert result.lines[0].votes >= 90

    def test_estimate_skew_angle_for_slanted_line(self):
        edges = np.zeros((80, 160), dtype=np.uint8)
        # y grows by 12 pixels over 120 columns -> atan(12/120) ~= 5.7 deg.
        for x in range(20, 140):
            y = int(round(30 + 0.10 * (x - 20)))
            edges[y, x] = 255

        angle = estimate_skew_angle(edges, angle_limit=10, theta_step=0.5, min_votes=20)
        assert angle == pytest.approx(5.7, abs=1.0)

    def test_empty_edge_image_returns_zero_angle(self):
        edges = np.zeros((50, 100), dtype=np.uint8)
        assert estimate_skew_angle(edges) == 0.0


class TestNormalizePlate:

    def test_normalize_plate_returns_target_shape(self):
        img = np.full((160, 300), 180, dtype=np.uint8)
        img[60:100, 80:220] = 240
        img[68:92, 100:112] = 30
        img[68:92, 135:147] = 30
        img[68:92, 170:182] = 30

        cfg = NormalizationConfig(target_shape=(80, 240), margin_ratio=0.05)
        res = normalize_plate(img, (80, 60, 140, 40), cfg)

        assert res.cropped.ndim == 2
        assert res.normalized.shape == (80, 240)
        assert res.normalized.dtype == np.uint8
        assert abs(res.angle_degrees) <= cfg.hough_angle_limit

    def test_invalid_target_shape_rejected(self):
        with pytest.raises(ValueError):
            resize_bilinear(np.zeros((4, 4), dtype=np.uint8), (0, 10))
