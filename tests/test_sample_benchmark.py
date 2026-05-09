"""
tests/test_sample_benchmark.py
==============================

Regression benchmark for the real images bundled in ``data/samples``.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.gui.app import analyze_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = PROJECT_ROOT / "data" / "labels" / "sample_ocr_labels.json"


def test_labeled_sample_images_recognize_expected_text():
    with LABELS_PATH.open("r", encoding="utf-8") as fh:
        labels = json.load(fh)

    assert labels
    for filename, expected in labels.items():
        result = analyze_image(PROJECT_ROOT / "data" / "samples" / filename)

        assert f"Step 4: {len(expected)} character candidate(s)" in result.summary
        assert f"Character OCR result = {expected}" in result.summary
