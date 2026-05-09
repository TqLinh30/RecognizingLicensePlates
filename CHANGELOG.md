# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-05-09

### Added

- Commit flow documentation in `docs/commitflow.md`, covering atomic commits,
  Conventional Commit scopes, verification, and release commit order.

## [0.3.0] - 2026-05-09

### Added

- **Step 3 - Plate normalization**:
  - `hough_lines` and `estimate_skew_angle` for near-horizontal line voting.
  - `crop_with_padding`, `rotate_image`, and `resize_bilinear` with bilinear interpolation.
  - `normalize_plate` to crop, deskew, and resize detected plates to a canonical shape.
- **Step 4 - Character segmentation**:
  - Otsu binarization plus connected-component character extraction.
  - Character filtering, row-aware sorting, and 32x32 character normalization.
- **Step 5 - Feature extraction**:
  - HOG descriptors with cell histograms and block normalization.
  - Zoning foreground-density descriptors.
  - Combined feature-vector extraction for single characters and batches.
- **Step 6 - Character classification**:
  - `KNNClassifier` baseline with probability-style vote fractions.
  - `MLPClassifier` with NumPy-only dense layers, ReLU, softmax, and backpropagation.
- **Step 7 - Recognition pipeline**:
  - `recognize_license_plate` end-to-end orchestration.
  - Vietnamese-plate postprocessing with position-aware confusion correction.
  - Unit tests for normalization, segmentation, features, classifiers, and recognition.

## [0.2.0] - 2026-05-09

### Added

- **Step 2 — License plate detection module** with all algorithms implemented from scratch:
  - `sobel`, `sobel_x` — separable Sobel gradient operators (1-D smoothing × 1-D differentiation).
  - `dilate`, `erode`, `opening`, `closing` — morphological operations on binary images via shifted-slice broadcasting.
  - `rect`, `cross` — structuring-element factories.
  - `connected_components` — two-pass connected-component labeling with path-compressed Union-Find. Returns per-component statistics (bounding box, area, centroid, aspect ratio, fill ratio).
  - `detect_plate` — end-to-end plate-candidate detector with composite scoring (aspect-ratio fit × fill ratio × gradient density).
  - `draw_candidates` — visualization helper that draws bounding boxes on the original image.
- CLI demo (`python -m src.detection.demo`).
- 22 unit tests in `tests/test_detection.py`, all passing.
- Documentation `docs/step2_detection.md`.

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
