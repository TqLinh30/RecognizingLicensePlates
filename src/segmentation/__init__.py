"""
Segmentation module (Pipeline Step 4)
=====================================

Splits a normalized plate crop into ordered character candidates.
"""

from src.segmentation.char_segmentation import (
    CharacterCandidate,
    SegmentationConfig,
    SegmentationResult,
    draw_character_boxes,
    normalize_character,
    segment_characters,
)

__all__ = [
    "CharacterCandidate",
    "SegmentationConfig",
    "SegmentationResult",
    "draw_character_boxes",
    "normalize_character",
    "segment_characters",
]
