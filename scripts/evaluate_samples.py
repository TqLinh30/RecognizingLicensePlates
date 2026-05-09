"""
evaluate_samples.py
===================

Run the GUI pipeline over labeled images in ``data/samples`` and report
whether raw OCR matches ``data/labels/sample_ocr_labels.json``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from src.gui.app import analyze_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = PROJECT_ROOT / "data" / "labels" / "sample_ocr_labels.json"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"


def main() -> None:
    with LABELS_PATH.open("r", encoding="utf-8") as fh:
        labels = json.load(fh)

    failures = 0
    for filename, expected in labels.items():
        result = analyze_image(SAMPLES_DIR / filename)
        predicted = _extract_predicted_text(result.summary)
        ok = predicted == expected
        failures += int(not ok)
        status = "PASS" if ok else "FAIL"
        print(f"{status} {filename}: expected={expected} predicted={predicted}")

    if failures:
        raise SystemExit(1)


def _extract_predicted_text(summary: str) -> str:
    match = re.search(r"Character OCR result = ([A-Z0-9]+)", summary)
    return match.group(1) if match else ""


if __name__ == "__main__":
    main()
