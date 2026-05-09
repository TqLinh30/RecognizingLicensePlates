"""
Train the raw-pixel template OCR classifier.

Usage:
    python -m scripts.train_pixel_template

This model compares each segmented 32x32 glyph directly against stored
synthetic printed-character templates.  It is format-free: no Vietnam
plate assumptions, only per-character recognition.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classifiers import PixelTemplateClassifier, save_pixel_template_model  # noqa: E402
from src.datasets.synthetic_plate_chars import (                              # noqa: E402
    DEFAULT_SYNTHETIC_CHARS,
    generate_synthetic_plate_characters,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train pixel-template OCR classifier.")
    parser.add_argument("--output", type=Path, default=Path("data/models/plate_pixel_templates.npz"))
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--templates-per-class", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--chars", type=str, default=DEFAULT_SYNTHETIC_CHARS)
    args = parser.parse_args()

    samples = generate_synthetic_plate_characters(
        chars=args.chars,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    model = PixelTemplateClassifier(
        max_templates_per_class=args.templates_per_class,
        temperature=args.temperature,
        seed=args.seed,
    ).fit(samples.images, samples.labels)

    pred = model.predict(samples.images)
    acc = float(np.mean(pred == samples.labels))
    print(f"[pixel-template] train accuracy on generated set: {acc:.4f}")

    save_pixel_template_model(model, args.output)
    print(f"[pixel-template] saved model: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
