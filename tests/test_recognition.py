"""
tests/test_recognition.py
=========================

Unit tests for Step 7 post-processing and end-to-end orchestration.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.recognition import (
    RecognitionConfig,
    correct_character_for_slot,
    format_vietnam_plate,
    postprocess_predictions,
    recognize_license_plate,
)


class FakeClassifier:
    """Deterministic classifier used to test pipeline orchestration."""

    def __init__(self, labels: list[str]):
        self.labels = labels
        self.classes_ = np.array(sorted(set(labels + list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"))))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros((X.shape[0], self.classes_.shape[0]), dtype=np.float32)
        for i in range(X.shape[0]):
            label = self.labels[min(i, len(self.labels) - 1)]
            class_idx = int(np.where(self.classes_ == label)[0][0])
            out[i, class_idx] = 0.95
            out[i] += 0.05 / self.classes_.shape[0]
        return out


def _synthetic_scene() -> np.ndarray:
    """Build a simple scene containing a plate with seven dark glyphs."""
    img = np.full((200, 400, 3), 185, dtype=np.uint8)
    img[50:155, 35:365] = np.array([150, 150, 155], dtype=np.uint8)

    x, y, w, h = 90, 88, 190, 52
    img[y : y + h, x : x + w] = 245

    char_w = 14
    char_h = 34
    spacing = (w - 30 - 7 * char_w) // 8
    cx = x + 15 + spacing
    for i in range(7):
        img[y + 9 : y + 9 + char_h, cx : cx + char_w] = 25
        cx += char_w + spacing
    return img


class TestPostprocessing:

    def test_digit_and_letter_confusion_correction(self):
        assert correct_character_for_slot("O", "D") == "0"
        assert correct_character_for_slot("0", "L") == "O"
        assert correct_character_for_slot("S", "D") == "5"

    def test_postprocess_predictions_formats_plate(self):
        result = postprocess_predictions(
            list("3OA12S4"),
            confidences=[0.9, 0.9, 0.8, 0.7, 0.7, 0.4, 0.9],
            min_confidence=0.5,
        )

        assert result.raw_text == "3OA12S4"
        assert result.corrected_text == "30A1254"
        assert result.formatted_text == "30A-1254"
        assert result.low_confidence_indices == [5]

    def test_format_vietnam_plate_is_conservative(self):
        assert format_vietnam_plate("30A12345") == "30A-12345"
        assert format_vietnam_plate("30") == "30"


class TestRecognitionPipeline:

    def test_end_to_end_pipeline_with_fake_classifier(self):
        img = _synthetic_scene()
        clf = FakeClassifier(list("30A1234"))
        cfg = RecognitionConfig()

        result = recognize_license_plate(img, clf, cfg)

        assert result.found_plate
        assert result.found_characters
        assert len(result.segmentation.characters) == 7
        assert result.raw_text == "30A1234"
        assert result.text == "30A-1234"
        assert min(result.confidences) > 0.90

    def test_pipeline_rejects_classifier_without_proba(self):
        img = _synthetic_scene()
        with pytest.raises(ValueError):
            recognize_license_plate(img, object())
