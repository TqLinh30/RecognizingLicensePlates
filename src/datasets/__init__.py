"""
Dataset utilities
=================

Download and prepare public datasets used for training character OCR
models.
"""

from src.datasets.emnist import (
    EMNIST_GZIP_URL,
    EMNIST_GZIP_URLS,
    EMNISTSampleSet,
    load_emnist_characters,
)
from src.datasets.synthetic_plate_chars import (
    DEFAULT_SYNTHETIC_CHARS,
    SyntheticPlateCharSet,
    generate_synthetic_plate_characters,
)

__all__ = [
    "EMNIST_GZIP_URL",
    "EMNIST_GZIP_URLS",
    "EMNISTSampleSet",
    "load_emnist_characters",
    "DEFAULT_SYNTHETIC_CHARS",
    "SyntheticPlateCharSet",
    "generate_synthetic_plate_characters",
]
