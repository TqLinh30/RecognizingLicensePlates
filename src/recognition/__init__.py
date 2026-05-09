"""
Recognition module (Pipeline Step 7)
====================================

End-to-end orchestration and plate-text post-processing.
"""

from src.recognition.pipeline import (
    RecognitionConfig,
    RecognitionResult,
    recognize_license_plate,
)
from src.recognition.postprocessing import (
    PostprocessResult,
    correct_character_for_slot,
    format_vietnam_plate,
    postprocess_predictions,
    vietnam_plate_slots,
)

__all__ = [
    "RecognitionConfig",
    "RecognitionResult",
    "recognize_license_plate",
    "PostprocessResult",
    "correct_character_for_slot",
    "format_vietnam_plate",
    "postprocess_predictions",
    "vietnam_plate_slots",
]
