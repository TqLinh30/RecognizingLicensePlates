"""
Train the OCR MLP on synthetic printed license-plate characters.

Usage:
    python -m scripts.train_synthetic_mlp

This is usually a better starter model than EMNIST for plate OCR because
it trains on printed glyphs shaped like the segmented characters that the
pipeline actually produces.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classifiers import MLPClassifier, MLPConfig, save_mlp_model  # noqa: E402
from src.datasets.synthetic_plate_chars import (                      # noqa: E402
    DEFAULT_SYNTHETIC_CHARS,
    generate_synthetic_plate_characters,
)
from src.features import extract_character_features, feature_length    # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train synthetic printed-plate OCR model.")
    parser.add_argument("--output", type=Path, default=Path("data/models/plate_synthetic_mlp.npz"))
    parser.add_argument("--samples-per-class", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[160, 96])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--chars", type=str, default=DEFAULT_SYNTHETIC_CHARS)
    parser.add_argument(
        "--font",
        action="append",
        default=None,
        help="Optional font path. Can be provided multiple times.",
    )
    args = parser.parse_args()

    print("[synthetic] generating printed character dataset...")
    samples = generate_synthetic_plate_characters(
        chars=args.chars,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        font_paths=args.font,
    )
    print(
        f"[synthetic] samples: {samples.images.shape[0]} images, "
        f"classes={len(set(samples.labels))}, fonts={len(samples.fonts)}"
    )

    print("[synthetic] extracting HOG+zoning features...")
    X = np.empty((samples.images.shape[0], feature_length()), dtype=np.float32)
    for i, image in enumerate(samples.images):
        X[i] = extract_character_features(image)
        if (i + 1) % 2000 == 0 or i + 1 == samples.images.shape[0]:
            print(f"[synthetic] features {i + 1}/{samples.images.shape[0]}")

    if not 0.0 <= args.validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be in [0, 1).")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(X.shape[0])
    val_size = int(round(X.shape[0] * args.validation_ratio))
    val_idx = order[:val_size]
    train_idx = order[val_size:]
    X_train = X[train_idx]
    y_train = samples.labels[train_idx]
    X_val = X[val_idx] if val_size else np.zeros((0, X.shape[1]), dtype=np.float32)
    y_val = samples.labels[val_idx] if val_size else np.zeros(0, dtype=samples.labels.dtype)
    print(f"[synthetic] train/val split: {X_train.shape[0]} train, {X_val.shape[0]} val")

    cfg = MLPConfig(
        hidden_sizes=tuple(args.hidden_sizes),
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model = MLPClassifier(cfg)
    print("[synthetic] training MLP...")
    model.fit(X_train, y_train)
    print(f"[synthetic] final loss: {model.loss_history_[-1]:.4f}")

    train_pred = model.predict(X_train)
    train_acc = float(np.mean(train_pred == y_train))
    if X_val.shape[0]:
        val_pred = model.predict(X_val)
        val_acc = float(np.mean(val_pred == y_val))
    else:
        val_acc = 0.0
    print(f"[synthetic] train accuracy on generated train split: {train_acc:.4f}")
    if X_val.shape[0]:
        print(f"[synthetic] validation accuracy on generated holdout: {val_acc:.4f}")

    save_mlp_model(
        model,
        args.output,
        metadata={
            "dataset": "synthetic printed plate characters",
            "selected_chars": args.chars,
            "samples_per_class": args.samples_per_class,
            "feature_length": feature_length(),
            "train_accuracy_generated": train_acc,
            "validation_accuracy_generated": val_acc,
            "validation_ratio": args.validation_ratio,
            "font_count": len(samples.fonts),
            "fonts": samples.fonts,
        },
    )
    print(f"[synthetic] saved model: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
