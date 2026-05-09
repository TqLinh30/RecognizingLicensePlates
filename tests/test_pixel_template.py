"""
tests/test_pixel_template.py
============================

Tests for raw-pixel template OCR.
"""

from __future__ import annotations

import numpy as np

from src.classifiers import (
    PixelTemplateClassifier,
    load_pixel_template_model,
    save_pixel_template_model,
)


def _glyph_left() -> np.ndarray:
    img = np.zeros((32, 32), dtype=np.uint8)
    img[5:27, 6:10] = 255
    return img


def _glyph_right() -> np.ndarray:
    img = np.zeros((32, 32), dtype=np.uint8)
    img[5:27, 22:26] = 255
    return img


def test_pixel_template_classifier_roundtrip(tmp_path):
    X = np.stack([_glyph_left(), _glyph_left(), _glyph_right(), _glyph_right()])
    y = np.array(["L", "L", "R", "R"])

    clf = PixelTemplateClassifier(max_templates_per_class=2).fit(X, y)
    assert clf.predict(np.stack([_glyph_left(), _glyph_right()])).tolist() == ["L", "R"]

    path = tmp_path / "pixel_templates.npz"
    save_pixel_template_model(clf, path)
    loaded = load_pixel_template_model(path)
    assert loaded.predict(np.stack([_glyph_left(), _glyph_right()])).tolist() == ["L", "R"]
