"""
Train the OCR MLP on the official EMNIST dataset.

Usage:
    python -m scripts.train_emnist_mlp --download

The script downloads EMNIST from NIST, selects digits + uppercase
letters, extracts the same HOG+zoning features used by the GUI, trains
the NumPy MLP, and saves:

    data/models/emnist_mlp.npz
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
from src.datasets import EMNIST_GZIP_URLS, load_emnist_characters      # noqa: E402
from src.features import extract_character_features, feature_length    # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train EMNIST MLP OCR model.")
    parser.add_argument("--download", action="store_true", help="Download EMNIST if missing.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/emnist"))
    parser.add_argument("--output", type=Path, default=Path("data/models/emnist_mlp.npz"))
    parser.add_argument("--samples-per-class", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("[train] sources:")
    for url in EMNIST_GZIP_URLS:
        print("  -", url)
    print("[train] loading EMNIST samples...")
    samples = load_emnist_characters(
        root=args.raw_dir,
        split="byclass",
        train=True,
        max_per_class=args.samples_per_class,
        seed=args.seed,
        download=args.download,
    )
    print(f"[train] samples: {samples.images.shape[0]} images, classes={len(set(samples.labels))}")

    print("[train] extracting HOG+zoning features...")
    X = np.empty((samples.images.shape[0], feature_length()), dtype=np.float32)
    for i, image in enumerate(samples.images):
        X[i] = extract_character_features(image)
        if (i + 1) % 1000 == 0 or i + 1 == samples.images.shape[0]:
            print(f"[train] features {i + 1}/{samples.images.shape[0]}")

    cfg = MLPConfig(
        hidden_sizes=tuple(args.hidden_sizes),
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model = MLPClassifier(cfg)
    print("[train] training MLP...")
    model.fit(X, samples.labels)
    print(f"[train] final loss: {model.loss_history_[-1]:.4f}")

    train_pred = model.predict(X)
    train_acc = float(np.mean(train_pred == samples.labels))
    print(f"[train] train accuracy on sampled subset: {train_acc:.4f}")

    save_mlp_model(
        model,
        args.output,
        metadata={
            "dataset": "EMNIST ByClass",
            "dataset_urls": EMNIST_GZIP_URLS,
            "selected_chars": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "samples_per_class": args.samples_per_class,
            "feature_length": feature_length(),
            "train_accuracy_subset": train_acc,
        },
    )
    print(f"[train] saved model: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
