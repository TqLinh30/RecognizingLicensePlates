"""
emnist.py
=========

Downloader and IDX parser for the official EMNIST dataset.

EMNIST is distributed by NIST as a ``gzip.zip`` archive containing
MNIST-style IDX files.  We parse those binary files directly so the
project does not need torchvision, tensorflow, scipy, or scikit-learn.

Source:
    https://www.nist.gov/itl/products-and-services/emnist-dataset
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.segmentation import normalize_character


EMNIST_GZIP_URLS = [
    # Public mirror maintained by Western Sydney University / MARCS.
    # It hosts the same EMNIST gzip archive referenced by the NIST page.
    "https://rds.westernsydney.edu.au/Institutes/MARCS/BENS/EMNIST/emnist-gzip.zip",
    # Historical NIST URLs.  Some environments receive an HTML landing
    # page or a 403 from these, so the downloader validates the zip file.
    "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip",
    "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip",
]
EMNIST_GZIP_URL = EMNIST_GZIP_URLS[0]
DEFAULT_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class EMNISTSampleSet:
    """Prepared character samples loaded from EMNIST."""

    images: np.ndarray
    labels: np.ndarray
    source_zip: Path
    split: str
    train: bool


def load_emnist_characters(
    root: str | Path = "data/raw/emnist",
    split: str = "byclass",
    train: bool = True,
    chars: str = DEFAULT_CHARS,
    max_per_class: int | None = 300,
    seed: int = 42,
    download: bool = True,
) -> EMNISTSampleSet:
    """
    Load selected digit/uppercase samples from EMNIST.

    Parameters
    ----------
    root : str | Path
        Directory where ``gzip.zip`` is stored.
    split : str
        EMNIST split name. ``byclass`` is recommended because it keeps
        digits and uppercase letters as separate classes.
    train : bool
        Use train files if true, test files otherwise.
    chars : str
        Character labels to keep.
    max_per_class : int, optional
        Maximum examples per selected class.  ``None`` keeps every
        available sample, which can be large.
    seed : int
        Sampling seed.
    download : bool
        Download the official zip if it does not exist.
    """
    root = Path(root)
    zip_path = root / "gzip.zip"
    if download:
        download_emnist(zip_path)
    elif not zip_path.is_file():
        raise FileNotFoundError(
            f"EMNIST archive not found: {zip_path}. Re-run with download=True."
        )

    mapping = _read_mapping(zip_path, split)
    wanted = set(chars)
    label_to_char = {
        label_id: ch
        for label_id, ch in mapping.items()
        if ch in wanted
    }
    if not label_to_char:
        raise ValueError(f"No requested chars {chars!r} found in EMNIST mapping.")

    images = _read_images(zip_path, split, train=train)
    labels = _read_labels(zip_path, split, train=train)
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Image/label count mismatch: {images.shape[0]} vs {labels.shape[0]}."
        )

    selected_indices = _sample_indices(labels, label_to_char.keys(), max_per_class, seed)
    selected_images = images[selected_indices]
    selected_labels = labels[selected_indices]

    # EMNIST IDX images are transposed relative to normal display.  A
    # transpose is the documented correction used by TFDS and community
    # loaders.  We normalize each glyph to the same 32x32 canvas used by
    # the segmentation stage, so classifier train/inference match.
    prepared_images = np.empty((selected_images.shape[0], 32, 32), dtype=np.uint8)
    prepared_labels: list[str] = []
    for i, (img, label_id) in enumerate(zip(selected_images, selected_labels)):
        upright = img.T
        prepared_images[i] = normalize_character(upright, target_shape=(32, 32))
        prepared_labels.append(label_to_char[int(label_id)])

    order = np.random.default_rng(seed).permutation(prepared_images.shape[0])
    return EMNISTSampleSet(
        images=prepared_images[order],
        labels=np.asarray(prepared_labels, dtype="<U1")[order],
        source_zip=zip_path,
        split=split,
        train=train,
    )


def download_emnist(zip_path: str | Path) -> Path:
    """Download the official EMNIST zip archive if it is missing."""
    zip_path = Path(zip_path)
    if zip_path.is_file():
        return zip_path

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in EMNIST_GZIP_URLS:
        try:
            print(f"[emnist] downloading {url}")
            print(f"[emnist] destination: {zip_path}")
            _download_file(url, zip_path)
            if zipfile.is_zipfile(zip_path):
                return zip_path
            zip_path.unlink(missing_ok=True)
            raise ValueError("downloaded file is not a valid zip archive")
        except Exception as exc:
            last_error = exc
            print(f"[emnist] download failed from {url}: {exc}")

    raise RuntimeError("All EMNIST download URLs failed.") from last_error


def _download_file(url: str, path: Path) -> None:
    """Download a URL with a browser-like User-Agent and atomic replace."""
    tmp_path = path.with_suffix(path.suffix + ".part")
    tmp_path.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        with tmp_path.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    tmp_path.replace(path)


def _member_name(split: str, train: bool, kind: str) -> str:
    part = "train" if train else "test"
    suffix = "images-idx3-ubyte.gz" if kind == "images" else "labels-idx1-ubyte.gz"
    return f"gzip/emnist-{split}-{part}-{suffix}"


def _read_mapping(zip_path: Path, split: str) -> dict[int, str]:
    """Read ``emnist-<split>-mapping.txt`` from the zip archive."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(f"gzip/emnist-{split}-mapping.txt") as f:
            lines = f.read().decode("utf-8").splitlines()

    mapping: dict[int, str] = {}
    for line in lines:
        if not line.strip():
            continue
        label_text, ascii_text = line.split()
        mapping[int(label_text)] = chr(int(ascii_text))
    return mapping


def _read_images(zip_path: Path, split: str, train: bool) -> np.ndarray:
    """Read an EMNIST IDX image file from the zip archive."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(_member_name(split, train, "images")) as compressed:
            raw = gzip.decompress(compressed.read())

    magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise ValueError(f"Invalid IDX image magic number: {magic}")
    data = np.frombuffer(raw, dtype=np.uint8, offset=16)
    return data.reshape(count, rows, cols)


def _read_labels(zip_path: Path, split: str, train: bool) -> np.ndarray:
    """Read an EMNIST IDX label file from the zip archive."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(_member_name(split, train, "labels")) as compressed:
            raw = gzip.decompress(compressed.read())

    magic, count = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError(f"Invalid IDX label magic number: {magic}")
    data = np.frombuffer(raw, dtype=np.uint8, offset=8)
    if data.shape[0] != count:
        raise ValueError(f"IDX label count mismatch: expected {count}, got {data.shape[0]}.")
    return data


def _sample_indices(
    labels: np.ndarray,
    selected_label_ids: Iterable[int],
    max_per_class: int | None,
    seed: int,
) -> np.ndarray:
    """Sample a balanced subset of indices from selected labels."""
    rng = np.random.default_rng(seed)
    pieces: list[np.ndarray] = []
    for label_id in sorted(selected_label_ids):
        idx = np.flatnonzero(labels == label_id)
        if idx.size == 0:
            continue
        if max_per_class is not None and idx.size > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        pieces.append(idx)
    if not pieces:
        raise ValueError("No matching EMNIST samples were found.")
    return np.concatenate(pieces)
