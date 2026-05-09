"""
tests/test_segmentation.py
==========================

Unit tests for Step 4 character segmentation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.detection import detect_plate
from src.normalization import normalize_plate
from src.preprocessing import preprocess
from src.segmentation import (
    SegmentationConfig,
    clean_character_crop,
    normalize_character,
    segment_characters,
)
from src.utils.image_io import load_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_plate(chars: int = 7) -> np.ndarray:
    """Build a clean normalized plate with dark rectangular glyphs."""
    plate = np.full((80, 240), 230, dtype=np.uint8)
    plate[8:72, 8:232] = 245

    char_w = 16
    char_h = 48
    spacing = (200 - chars * char_w) // (chars + 1)
    x = 20 + spacing
    for i in range(chars):
        top = 16 + (i % 2)  # tiny jitter so filters are not brittle
        plate[top : top + char_h, x : x + char_w] = 25
        x += char_w + spacing
    return plate


class TestCharacterSegmentation:

    def test_segment_characters_returns_left_to_right_components(self):
        plate = _synthetic_plate(chars=7)
        result = segment_characters(plate)

        assert len(result.characters) == 7
        xs = [char.x for char in result.characters]
        assert xs == sorted(xs)
        assert all(char.normalized.shape == (32, 32) for char in result.characters)
        assert all(char.normalized.dtype == np.uint8 for char in result.characters)

    def test_segment_characters_rejects_smooth_plate(self):
        plate = np.full((80, 240), 230, dtype=np.uint8)
        result = segment_characters(plate)
        assert result.characters == []

    def test_two_line_sorting_orders_top_row_then_bottom_row(self):
        plate = np.full((100, 180), 240, dtype=np.uint8)
        # Top row: three glyphs.
        for x in (20, 55, 90):
            plate[12:42, x : x + 14] = 20
        # Bottom row: four glyphs.
        for x in (18, 52, 86, 120):
            plate[60:92, x : x + 14] = 20

        cfg = SegmentationConfig(
            min_height_ratio=0.20,
            two_line_gap_ratio=0.12,
            reject_border_touching=False,
        )
        result = segment_characters(plate, cfg)

        assert len(result.characters) == 7
        assert [c.row_index for c in result.characters[:3]] == [0, 0, 0]
        assert [c.row_index for c in result.characters[3:]] == [1, 1, 1, 1]
        assert result.characters[0].x < result.characters[1].x < result.characters[2].x
        assert result.characters[3].x < result.characters[4].x < result.characters[5].x

    def test_normalize_character_preserves_foreground(self):
        char = np.zeros((20, 10), dtype=np.uint8)
        char[2:18, 3:7] = 255
        out = normalize_character(char, (32, 32))

        assert out.shape == (32, 32)
        assert out.max() == 255
        assert out.min() == 0
        assert out[:, 16].sum() > 0

    def test_clean_character_crop_removes_detached_border_fragments(self):
        char = np.zeros((40, 28), dtype=np.uint8)
        char[8:34, 12:16] = 255       # main vertical glyph
        char[0:2, 3:24] = 255         # detached top plate-border fragment
        char[38:40, 4:23] = 255       # detached bottom plate-border fragment

        cleaned = clean_character_crop(char)

        assert cleaned[0:2].sum() == 0
        assert cleaned[38:40].sum() == 0
        assert cleaned[8:34, 12:16].sum() > 0

    def test_slot_segmentation_keeps_detached_top_stroke(self):
        plate = np.full((80, 160), 230, dtype=np.uint8)
        # A broken "7": the top bar is separated from the diagonal body.
        # Pure connected-component cropping keeps only the tall body; the
        # slot crop should recover the detached bar before OCR.
        plate[20:24, 55:72] = 60
        for i, y in enumerate(range(25, 64)):
            x = 68 - int(i * 0.22)
            plate[y : y + 2, x : x + 5] = 25

        result = segment_characters(plate)

        assert len(result.characters) == 1
        char = result.characters[0]
        top_cols = np.flatnonzero(char.normalized[:10].sum(axis=0) > 0)
        assert char.width >= 18
        assert top_cols.size >= 4

    def test_sample_plate2_keeps_wide_seven_slots(self):
        image = load_image(PROJECT_ROOT / "data" / "samples" / "plate2.jpg")
        pre = preprocess(image)
        detection = detect_plate(pre.enhanced)
        normalization = normalize_plate(pre.enhanced, detection.candidates[0])

        result = segment_characters(normalization.normalized)

        assert len(result.characters) == 8
        seven_like_slots = result.characters[3:]
        assert all(char.width >= 18 for char in seven_like_slots)
        assert all(char.normalized[:10].sum() > 40 * 255 for char in seven_like_slots)

    def test_invalid_plate_dtype_rejected(self):
        with pytest.raises(ValueError):
            segment_characters(np.zeros((80, 240), dtype=np.float32))
