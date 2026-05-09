"""
tests/test_emnist_dataset.py
============================

Small parser test using a tiny fake EMNIST-style zip archive.
"""

from __future__ import annotations

import gzip
import struct
import zipfile

import numpy as np

from src.datasets.emnist import load_emnist_characters


def test_load_emnist_characters_from_tiny_fake_zip(tmp_path):
    root = tmp_path / "emnist"
    root.mkdir()
    zip_path = root / "gzip.zip"

    images = np.zeros((4, 28, 28), dtype=np.uint8)
    images[0, 5:20, 8:12] = 255
    images[1, 6:21, 10:14] = 255
    images[2, 8:12, 6:22] = 255
    images[3, 10:14, 8:24] = 255
    labels = np.array([0, 0, 1, 1], dtype=np.uint8)

    image_raw = struct.pack(">IIII", 2051, 4, 28, 28) + images.tobytes()
    label_raw = struct.pack(">II", 2049, 4) + labels.tobytes()

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("gzip/emnist-byclass-mapping.txt", "0 48\n1 65\n")
        zf.writestr(
            "gzip/emnist-byclass-train-images-idx3-ubyte.gz",
            gzip.compress(image_raw),
        )
        zf.writestr(
            "gzip/emnist-byclass-train-labels-idx1-ubyte.gz",
            gzip.compress(label_raw),
        )

    samples = load_emnist_characters(
        root=root,
        split="byclass",
        train=True,
        chars="0A",
        max_per_class=None,
        download=False,
    )

    assert samples.images.shape == (4, 32, 32)
    assert sorted(samples.labels.tolist()) == ["0", "0", "A", "A"]
    assert samples.images.dtype == np.uint8
