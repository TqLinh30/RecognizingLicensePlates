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

__all__ = [
    "EMNIST_GZIP_URL",
    "EMNIST_GZIP_URLS",
    "EMNISTSampleSet",
    "load_emnist_characters",
]
