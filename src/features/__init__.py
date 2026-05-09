"""
Feature extraction module (Pipeline Step 5)
===========================================

Converts normalized character images into numeric feature vectors.
"""

from src.features.extractor import (
    FeatureConfig,
    extract_batch_features,
    extract_character_features,
    feature_length,
)
from src.features.hog import HOGConfig, hog_descriptor, hog_length
from src.features.zoning import zoning_features

__all__ = [
    "FeatureConfig",
    "extract_batch_features",
    "extract_character_features",
    "feature_length",
    "HOGConfig",
    "hog_descriptor",
    "hog_length",
    "zoning_features",
]
