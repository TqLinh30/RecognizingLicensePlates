"""
tests/test_detection.py
=======================

Unit tests for every sub-module of Step-2 plate detection.

We avoid loading any image from disk: all tests use synthetic
NumPy arrays so they run anywhere with just NumPy + Pillow installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.detection import (
    sobel, sobel_x,
    rect, cross,
    dilate, erode, opening, closing,
    connected_components,
    detect_plate, DetectionConfig,
)


# ===========================================================================
# Sobel
# ===========================================================================

class TestSobel:

    def test_constant_image_zero_gradient(self):
        """Flat input produces zero gradient everywhere."""
        img = np.full((20, 20), 128, dtype=np.uint8)
        res = sobel(img)
        assert np.all(res.gx == 0)
        assert np.all(res.gy == 0)
        assert np.all(res.magnitude == 0)

    def test_vertical_edge_gives_strong_gx(self):
        """A pure vertical step edge → strong horizontal gradient at the step."""
        img = np.zeros((20, 20), dtype=np.uint8)
        img[:, 10:] = 255  # left half black, right half white
        res = sobel(img)
        # Horizontal gradient should be huge at column 10.  Vertical
        # gradient is zero away from the very top/bottom edges (where
        # smoothing introduces tiny boundary effects).
        assert res.gx[10, 10] > 500          # well above noise
        # Y-gradient should be very small in the interior.
        assert np.abs(res.gy[5:15, :]).max() < 1.0

    def test_horizontal_edge_gives_strong_gy(self):
        """A pure horizontal step edge → strong vertical gradient at the step."""
        img = np.zeros((20, 20), dtype=np.uint8)
        img[10:, :] = 255
        res = sobel(img)
        assert res.gy[10, 10] > 500
        # X-gradient should be near zero in the interior.
        assert np.abs(res.gx[:, 5:15]).max() < 1.0

    def test_sobel_x_returns_uint8(self):
        img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        out = sobel_x(img)
        assert out.dtype == np.uint8
        assert out.shape == img.shape

    def test_sobel_x_symmetric_response(self):
        """|∂I/∂x| should respond equally to a black-to-white and a white-to-black edge."""
        a = np.zeros((20, 20), dtype=np.uint8); a[:, 10:] = 255
        b = np.full((20, 20), 255, dtype=np.uint8); b[:, 10:] = 0
        ga = sobel_x(a)
        gb = sobel_x(b)
        # Pixel-wise the responses should match because we take abs().
        assert np.array_equal(ga, gb)


# ===========================================================================
# Morphology
# ===========================================================================

class TestMorphology:

    def test_rect_se_dimensions(self):
        se = rect(3, 5)
        assert se.shape == (3, 5)
        assert np.all(se == 1)

    def test_cross_se_pattern(self):
        se = cross(3)
        # Centre row and centre column are 1; corners are 0.
        assert se[1, 0] == 1 and se[1, 1] == 1 and se[1, 2] == 1
        assert se[0, 1] == 1 and se[2, 1] == 1
        assert se[0, 0] == 0 and se[2, 2] == 0

    def test_dilate_grows_single_pixel(self):
        """A single bright pixel dilated with a 3×3 SE becomes a 3×3 square."""
        img = np.zeros((10, 10), dtype=np.uint8)
        img[5, 5] = 255
        out = dilate(img, rect(3, 3))
        assert np.all(out[4:7, 4:7] == 255)
        # Outside that 3×3 neighbourhood, everything stays zero.
        mask = np.zeros_like(out, dtype=bool)
        mask[4:7, 4:7] = True
        assert out[~mask].sum() == 0

    def test_erode_removes_thin_lines(self):
        """A 1-pixel-wide line erodes to nothing under a 3×3 SE."""
        img = np.zeros((20, 20), dtype=np.uint8)
        img[10, :] = 255  # horizontal 1-pixel line
        out = erode(img, rect(3, 3))
        assert out.sum() == 0

    def test_opening_removes_small_specks(self):
        """A 1×1 noise pixel disappears; a large square survives."""
        img = np.zeros((20, 20), dtype=np.uint8)
        img[2, 2] = 255              # speck
        img[10:18, 10:18] = 255      # big square
        out = opening(img, rect(3, 3))
        assert out[2, 2] == 0
        # The big square remains roughly intact.
        assert out[12:16, 12:16].sum() > 0

    def test_closing_bridges_small_gap(self):
        """Two adjacent bright pixels with a one-pixel gap are merged by closing."""
        img = np.zeros((10, 10), dtype=np.uint8)
        img[5, 4] = 255
        img[5, 6] = 255
        out = closing(img, rect(1, 3))
        assert out[5, 5] == 255

    def test_dilate_erode_constant_image(self):
        """A fully foreground image survives both ops unchanged."""
        img = np.full((10, 10), 255, dtype=np.uint8)
        assert np.all(dilate(img, rect(3, 3)) == 255)
        # Erosion shrinks the foreground by SE_radius pixels at each
        # border, so the interior is still 255 even though the borders
        # become 0.  Centre pixels are always 255.
        eroded = erode(img, rect(3, 3))
        assert eroded[5, 5] == 255


# ===========================================================================
# Connected Components
# ===========================================================================

class TestConnectedComponents:

    def test_empty_image(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        res = connected_components(img)
        assert res.num_labels == 0
        assert res.stats == []

    def test_single_blob(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        img[3:7, 3:7] = 255
        res = connected_components(img)
        assert res.num_labels == 1
        s = res.stats[0]
        assert s.area == 16
        assert s.x == 3 and s.y == 3
        assert s.width == 4 and s.height == 4
        assert s.cx == pytest.approx(4.5)
        assert s.cy == pytest.approx(4.5)

    def test_two_disjoint_blobs(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        img[1:3, 1:3] = 255
        img[6:9, 6:9] = 255
        res = connected_components(img, connectivity=8)
        assert res.num_labels == 2
        # Order is row-major: the upper-left blob comes first.
        a, b = res.stats
        assert a.area == 4
        assert b.area == 9

    def test_4_vs_8_connectivity(self):
        """A diagonal pair is one blob in 8-conn, two blobs in 4-conn."""
        img = np.zeros((5, 5), dtype=np.uint8)
        img[1, 1] = 255
        img[2, 2] = 255
        assert connected_components(img, connectivity=4).num_labels == 2
        assert connected_components(img, connectivity=8).num_labels == 1

    def test_u_shape_unifies_under_8_conn(self):
        """A U-shape is one blob; tests the union-find path."""
        img = np.zeros((5, 7), dtype=np.uint8)
        # Bottom bar plus two vertical arms.
        img[4, :] = 255
        img[1:5, 0] = 255
        img[1:5, 6] = 255
        res = connected_components(img, connectivity=8)
        assert res.num_labels == 1

    def test_aspect_ratio_and_fill_ratio(self):
        img = np.zeros((10, 30), dtype=np.uint8)
        img[2:8, 5:25] = 255   # 6 high, 20 wide solid rectangle
        res = connected_components(img)
        s = res.stats[0]
        assert s.aspect_ratio == pytest.approx(20 / 6)
        assert s.fill_ratio == pytest.approx(1.0)


# ===========================================================================
# End-to-end plate detector
# ===========================================================================

def _synthetic_scene_with_plate(
    img_h: int = 200,
    img_w: int = 400,
    plate_x: int = 60,
    plate_y: int = 80,
    plate_w: int = 180,
    plate_h: int = 50,
) -> np.ndarray:
    """
    Build a fake "scene" with a single plate-shaped block of vertical bars
    against a smooth gray background.

    Returns an ``(H, W)`` uint8 image suitable for feeding to detect_plate.
    """
    img = np.full((img_h, img_w), 200, dtype=np.uint8)

    # Plate background: light gray rectangle.
    img[plate_y : plate_y + plate_h, plate_x : plate_x + plate_w] = 240

    # Add ~7 dark vertical bars inside the plate to mimic characters.
    n_bars = 7
    bar_w = 8
    spacing = (plate_w - n_bars * bar_w) // (n_bars + 1)
    for i in range(n_bars):
        bx = plate_x + spacing + i * (bar_w + spacing)
        img[plate_y + 10 : plate_y + plate_h - 10, bx : bx + bar_w] = 30

    return img


class TestDetectPlate:

    def test_finds_synthetic_plate(self):
        img = _synthetic_scene_with_plate()
        det = detect_plate(img)
        assert len(det.candidates) >= 1
        top = det.candidates[0]
        # The detected box should overlap heavily with the ground-truth
        # plate region (60, 80, 180, 50).
        assert 30 <= top.x <= 80
        assert 60 <= top.y <= 100
        # And the score should be reasonable.
        assert top.score > 0.05

    def test_no_plate_in_smooth_image(self):
        """A smooth gray field should produce zero candidates."""
        img = np.full((200, 400), 128, dtype=np.uint8)
        det = detect_plate(img)
        assert det.candidates == []

    def test_intermediate_maps_have_correct_shape(self):
        img = _synthetic_scene_with_plate()
        det = detect_plate(img)
        H, W = img.shape
        assert det.gradient.shape == (H, W)
        assert det.binary.shape == (H, W)
        assert det.closed.shape == (H, W)
        assert det.labels.shape == (H, W)

    def test_invalid_input_dtype_rejected(self):
        with pytest.raises(ValueError):
            detect_plate(np.zeros((50, 50), dtype=np.float32))

    def test_custom_config_respected(self):
        img = _synthetic_scene_with_plate()
        # Force aspect-ratio range to reject everything.
        cfg = DetectionConfig(min_aspect_ratio=10.0, max_aspect_ratio=20.0)
        det = detect_plate(img, cfg)
        assert det.candidates == []
