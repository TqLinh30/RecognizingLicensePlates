"""
tests/test_synthetic_plate_chars.py
===================================

Smoke tests for the synthetic printed-character generator.
"""

from __future__ import annotations

from src.datasets.synthetic_plate_chars import generate_synthetic_plate_characters


def test_generate_synthetic_plate_characters_shape_and_labels():
    samples = generate_synthetic_plate_characters(chars="01A", samples_per_class=3, seed=123)

    assert samples.images.shape == (9, 32, 32)
    assert samples.images.dtype.name == "uint8"
    assert sorted(samples.labels.tolist()) == ["0", "0", "0", "1", "1", "1", "A", "A", "A"]
    assert len(samples.fonts) >= 1
    assert samples.images.max() == 255
