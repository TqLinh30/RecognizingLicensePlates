"""
tests/test_zoning_template.py
=============================

Tests for the fixed-grid zoning-template classifier.
"""

from __future__ import annotations

import numpy as np

from src.classifiers import (
    ZoningTemplateClassifier,
    load_zoning_template_model,
    save_zoning_template_model,
)


def _glyph_vertical() -> np.ndarray:
    img = np.zeros((32, 32), dtype=np.uint8)
    img[4:28, 14:18] = 255
    return img


def _glyph_horizontal() -> np.ndarray:
    img = np.zeros((32, 32), dtype=np.uint8)
    img[14:18, 4:28] = 255
    return img


def test_zoning_template_classifier_learns_simple_shapes(tmp_path):
    X = np.stack([_glyph_vertical(), _glyph_vertical(), _glyph_horizontal(), _glyph_horizontal()])
    y = np.array(["I", "I", "H", "H"])

    clf = ZoningTemplateClassifier(grid=(8, 8), max_templates_per_class=2).fit(X, y)
    pred = clf.predict(np.stack([_glyph_vertical(), _glyph_horizontal()]))

    assert pred.tolist() == ["I", "H"]

    path = tmp_path / "zoning.npz"
    save_zoning_template_model(clf, path)
    loaded = load_zoning_template_model(path)

    assert loaded.predict(np.stack([_glyph_vertical(), _glyph_horizontal()])).tolist() == ["I", "H"]
