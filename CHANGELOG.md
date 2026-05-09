# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-09

### Added

- **Step 1 — Preprocessing module** with all algorithms implemented from scratch:
  - `rgb_to_grayscale` — BT.601 luminance conversion.
  - `gaussian_blur` — separable Gaussian filter with reflect padding.
  - `median_filter` — vectorized order-statistic filter for salt-and-pepper noise.
  - `compute_histogram`, `compute_cdf`, `histogram_equalization` — global histogram operations.
  - `clahe` — Contrast Limited Adaptive Histogram Equalization with bilinear inter-tile interpolation.
  - `fixed_threshold`, `otsu_threshold` — image binarization (manual and automatic).
  - `preprocess` — convenience pipeline wiring the defaults together.
- I/O utilities (`load_image`, `save_image`, `save_side_by_side`).
- CLI demo (`python -m src.preprocessing.demo`).
- Unit tests (`tests/test_preprocessing.py`).
- Project docs (`docs/step1_preprocessing.md`, `docs/gitflow.md`).
- Gitflow branching configuration and contribution guide.
