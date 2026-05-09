"""
postprocessing.py
=================

Step 7 post-processing for predicted character sequences.

Classifiers see characters one by one, so they do not know the legal
structure of a license plate.  Post-processing uses simple format
constraints to correct common OCR confusions and produce a readable
final string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


DIGITS = set("0123456789")
LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ALPHANUMERIC = DIGITS | LETTERS

DIGIT_CONFUSIONS = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
}

LETTER_CONFUSIONS = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}


@dataclass
class PostprocessResult:
    """Text output after validation/correction."""

    raw_text: str
    corrected_text: str
    formatted_text: str
    average_confidence: float
    low_confidence_indices: list[int] = field(default_factory=list)


def postprocess_predictions(
    labels: Sequence[str],
    confidences: Sequence[float] | None = None,
    min_confidence: float = 0.50,
    use_vietnam_format: bool = True,
) -> PostprocessResult:
    """
    Correct and format a sequence of predicted character labels.

    Labels are uppercased and stripped to alphanumeric characters.  If
    ``use_vietnam_format`` is true, the first two slots are treated as
    province digits, the third as a series letter, and the rest as
    digits.  This gives position-aware corrections such as ``O -> 0``
    in digit slots and ``0 -> O`` in the letter slot.
    """
    clean_labels = [_clean_label(label) for label in labels]
    clean_labels = [label for label in clean_labels if label]
    raw = "".join(clean_labels)

    if confidences is None:
        conf = np.ones(len(clean_labels), dtype=np.float32)
    else:
        conf = np.asarray(confidences, dtype=np.float32)[: len(clean_labels)]
        if conf.size < len(clean_labels):
            conf = np.pad(conf, (0, len(clean_labels) - conf.size), constant_values=0.0)

    slots = vietnam_plate_slots(len(raw)) if use_vietnam_format else ["A"] * len(raw)
    corrected = "".join(
        correct_character_for_slot(ch, slot)
        for ch, slot in zip(raw, slots)
    )
    formatted = format_vietnam_plate(corrected) if use_vietnam_format else corrected
    low = [i for i, value in enumerate(conf.tolist()) if value < min_confidence]

    return PostprocessResult(
        raw_text=raw,
        corrected_text=corrected,
        formatted_text=formatted,
        average_confidence=float(conf.mean()) if conf.size else 0.0,
        low_confidence_indices=low,
    )


def vietnam_plate_slots(length: int) -> list[str]:
    """
    Return expected slot types for a simplified Vietnamese plate format.

    Slot types:
    * ``D``: digit only,
    * ``L``: letter only,
    * ``A``: alphanumeric fallback.

    The common compact form is ``30A12345``: two province digits, one
    letter, then the serial digits.  This helper intentionally stays
    permissive so unusual local formats still produce output.
    """
    if length <= 0:
        return []
    slots = ["A"] * length
    if length >= 1:
        slots[0] = "D"
    if length >= 2:
        slots[1] = "D"
    if length >= 3:
        slots[2] = "L"
    for i in range(3, length):
        slots[i] = "D"
    return slots


def correct_character_for_slot(ch: str, slot: str) -> str:
    """Apply position-aware confusion correction."""
    ch = _clean_label(ch)
    if not ch:
        return ""
    if slot == "D":
        if ch in DIGITS:
            return ch
        return DIGIT_CONFUSIONS.get(ch, ch)
    if slot == "L":
        if ch in LETTERS:
            return ch
        return LETTER_CONFUSIONS.get(ch, ch)
    return ch if ch in ALPHANUMERIC else ""


def format_vietnam_plate(text: str) -> str:
    """
    Insert a readable separator into compact Vietnamese plate text.

    We keep formatting conservative: ``30A12345`` becomes
    ``30A-12345``.  More elaborate province-specific spacing can be
    layered on later without changing recognition internals.
    """
    compact = "".join(ch for ch in text.upper() if ch in ALPHANUMERIC)
    if len(compact) <= 3:
        return compact
    return f"{compact[:3]}-{compact[3:]}"


def _clean_label(label: str) -> str:
    """Uppercase one predicted label and keep only alphanumeric chars."""
    text = str(label).strip().upper()
    for ch in text:
        if ch in ALPHANUMERIC:
            return ch
    return ""
