"""
tests/test_features.py
======================

Unit tests for Step 5 HOG and zoning features.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features import (
    FeatureConfig,
    HOGConfig,
    extract_batch_features,
    extract_character_features,
    feature_length,
    hog_descriptor,
    hog_length,
    zoning_features,
)


class TestHOG:

    def test_blank_image_has_zero_hog(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        feat = hog_descriptor(img)
        assert feat.shape == (hog_length((32, 32)),)
        assert np.all(feat == 0)

    def test_vertical_bar_has_nonzero_hog(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[:, 14:18] = 255
        feat = hog_descriptor(img)
        assert feat.shape == (324,)
        assert feat.sum() > 0
        assert np.isfinite(feat).all()

    def test_invalid_hog_config_rejected(self):
        with pytest.raises(ValueError):
            hog_descriptor(np.zeros((32, 32), dtype=np.uint8), HOGConfig(cell_size=0))


class TestZoning:

    def test_zoning_reports_foreground_density(self):
        img = np.zeros((8, 8), dtype=np.uint8)
        img[:4, :4] = 255
        feat = zoning_features(img, grid=(2, 2))
        assert np.allclose(feat, [1.0, 0.0, 0.0, 0.0])

    def test_invalid_grid_rejected(self):
        with pytest.raises(ValueError):
            zoning_features(np.zeros((8, 8), dtype=np.uint8), grid=(0, 2))


class TestCombinedFeatures:

    def test_combined_feature_length(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        feat = extract_character_features(img)
        assert feat.shape == (340,)
        assert feature_length((32, 32)) == 340

    def test_batch_features_shape(self):
        imgs = [
            np.zeros((32, 32), dtype=np.uint8),
            np.full((32, 32), 255, dtype=np.uint8),
        ]
        batch = extract_batch_features(imgs)
        assert batch.shape == (2, 340)
        assert batch.dtype == np.float32

    def test_feature_families_can_be_toggled(self):
        cfg = FeatureConfig(include_hog=False, include_zoning=True, zoning_grid=(2, 3))
        img = np.zeros((32, 32), dtype=np.uint8)
        feat = extract_character_features(img, cfg)
        assert feat.shape == (6,)
