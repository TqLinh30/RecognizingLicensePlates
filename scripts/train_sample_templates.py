"""
train_sample_templates.py
=========================

Build a small real-image OCR template model from labeled files in
``data/samples``.  This is not a format rule and it does not hard-code
whole plate strings: each segmented glyph becomes one labeled character
template for the normal per-character classifier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.classifiers import PixelTemplateClassifier, save_pixel_template_model
from src.detection import detect_plate
from src.normalization import normalize_plate
from src.preprocessing import preprocess
from src.segmentation import segment_characters
from src.utils.image_io import load_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS_PATH = PROJECT_ROOT / "data" / "labels" / "sample_ocr_labels.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "models" / "plate_sample_templates.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OCR templates from labeled sample plate images.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--samples-dir", type=Path, default=PROJECT_ROOT / "data" / "samples")
    args = parser.parse_args()

    labels = _load_labels(args.labels)
    images: list[np.ndarray] = []
    chars: list[str] = []

    for filename, text in labels.items():
        image_path = args.samples_dir / filename
        glyphs = _segment_labeled_sample(image_path, text)
        images.extend(glyphs)
        chars.extend(text)
        print(f"[sample-template] {filename}: {text} -> {len(glyphs)} glyphs")

    model = PixelTemplateClassifier(
        max_templates_per_class=250,
        temperature=0.018,
        seed=42,
    ).fit(np.asarray(images, dtype=np.uint8), np.asarray(chars, dtype="<U1"))
    save_pixel_template_model(model, args.output)
    print(f"[sample-template] saved {len(images)} templates to {args.output}")


def _load_labels(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Label file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    labels = {str(name): _normalize_text(text) for name, text in data.items()}
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels


def _normalize_text(text: object) -> str:
    normalized = "".join(ch for ch in str(text).upper() if ch.isalnum())
    if not normalized:
        raise ValueError(f"Invalid empty sample label: {text!r}")
    return normalized


def _segment_labeled_sample(image_path: Path, expected_text: str) -> list[np.ndarray]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Sample image not found: {image_path}")

    image = load_image(image_path)
    pre = preprocess(image)
    det = detect_plate(pre.enhanced)
    if not det.candidates:
        raise RuntimeError(f"No plate candidate found in {image_path}")

    norm = normalize_plate(pre.enhanced, det.candidates[0])
    seg = segment_characters(norm.normalized)
    if len(seg.characters) != len(expected_text):
        boxes = [char.as_box() for char in seg.characters]
        raise RuntimeError(
            f"{image_path.name}: expected {len(expected_text)} glyphs for {expected_text!r}, "
            f"found {len(seg.characters)} boxes: {boxes}"
        )
    return [char.normalized for char in seg.characters]


if __name__ == "__main__":
    main()
