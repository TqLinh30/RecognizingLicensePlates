"""
Train the zoning-template OCR classifier.

This script implements the fixed-grid white-region idea: every character
is split into an 8x8 grid, each cell stores foreground density, and each
class is represented by its average density vector.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classifiers import ZoningTemplateClassifier, save_zoning_template_model  # noqa: E402
from src.datasets.synthetic_plate_chars import (                                # noqa: E402
    DEFAULT_SYNTHETIC_CHARS,
    generate_synthetic_plate_characters,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train zoning-template OCR classifier.")
    parser.add_argument("--output", type=Path, default=Path("data/models/plate_zoning_templates.npz"))
    parser.add_argument("--samples-per-class", type=int, default=900)
    parser.add_argument("--grid", type=int, nargs=2, default=[8, 8])
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--templates-per-class", type=int, default=60)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--chars", type=str, default=DEFAULT_SYNTHETIC_CHARS)
    args = parser.parse_args()

    samples = generate_synthetic_plate_characters(
        chars=args.chars,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    model = ZoningTemplateClassifier(
        grid=(int(args.grid[0]), int(args.grid[1])),
        temperature=args.temperature,
        max_templates_per_class=args.templates_per_class,
        seed=args.seed,
    ).fit(samples.images, samples.labels)

    pred = model.predict(samples.images)
    train_acc = float(np.mean(pred == samples.labels))
    print(f"[zoning-template] train accuracy on generated set: {train_acc:.4f}")

    save_zoning_template_model(model, args.output)
    print(f"[zoning-template] saved model: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
