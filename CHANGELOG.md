# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Sample OCR benchmark now covers all six bundled images:
  `synthetic_car.png`, `images.jpg`, `plate4.png`, `plate1.jpg`,
  `plate2.jpg`, and `plate3.jpg`.
- GUI plate-region selection now validates detector candidates through
  downstream segmentation, penalizes candidates touching the image border, and
  supports a full-image fallback for already-cropped plate images.

### Fixed

- Step 4 segmentation now has an adaptive anchor fallback that suppresses dense
  plate-border rows/columns, recovering low-contrast edge characters such as
  the trailing `45` in `synthetic_car.png`.

## [0.10.0] - 2026-05-09

### Added

- Real-image sample OCR labels and sample-template training script:
  `python -m scripts.train_sample_templates`.
- Real-image sample benchmark command: `python -m scripts.evaluate_samples`.
- Bundled `data/models/plate_sample_templates.npz`, trained from labeled
  glyph crops in `data/samples`.
- End-to-end benchmark test that checks `plate1.jpg`, `plate2.jpg`, and
  `plate3.jpg` return the expected raw OCR strings.

### Changed

- GUI OCR ensemble now gives highest weight to the real-sample template model
  when it is available.

### Fixed

- Character slot boundaries now respect neighbouring midpoints, preventing
  close pairs such as `70` from bleeding into each other during Step 4.2.

## [0.9.0] - 2026-05-09

### Fixed

- Step 4 character segmentation now uses Otsu connected-component anchors plus
  local adaptive slot crops, so detached strokes such as the top bar of `7`
  are kept before OCR instead of being cropped into narrow vertical fragments.
- GUI Step 4.1 now shows the actual adaptive character mask used for glyph
  crops.

## [0.8.0] - 2026-05-09

### Added

- Raw pixel-template OCR classifier and training script:
  `python -m scripts.train_pixel_template`.
- Bundled `data/models/plate_pixel_templates.npz` for format-free character
  recognition.

### Changed

- GUI now reports only raw character OCR, with no Vietnam-format output or
  VN-slot correction.
- Recognition pipeline defaults to raw OCR instead of Vietnam-format
  postprocessing.

## [0.7.0] - 2026-05-09

### Added

- Zoning-template OCR classifier based on fixed white-pixel regions.
- Training script and bundled model: `python -m scripts.train_zoning_template`
  and `data/models/plate_zoning_templates.npz`.
- GUI top-3 OCR alternatives per character and separate Raw OCR vs
  Vietnam-format OCR output.

### Changed

- Character segmentation now removes detached border fragments before
  normalizing each glyph to `32x32`.
- GUI blends MLP and zoning-template probabilities when both models exist.

## [0.6.0] - 2026-05-09

### Added

- Synthetic printed-character OCR generator and trainer:
  `python -m scripts.train_synthetic_mlp`.
- Starter `data/models/plate_synthetic_mlp.npz` model, which the GUI prefers
  over the EMNIST handwriting baseline.

### Changed

- GUI Step 4 text now explains that `32x32` is the project-wide OCR input size
  and EMNIST `28x28` samples are converted to it.
- Training-data docs now recommend synthetic printed characters over EMNIST.

## [0.5.0] - 2026-05-09

### Added

- EMNIST downloader/parser for public OCR training data from NIST.
- EMNIST MLP training script: `python -m scripts.train_emnist_mlp --download`.
- MLP `.npz` save/load helpers and GUI auto-loading for `data/models/emnist_mlp.npz`.
- Training-data documentation in `docs/training_data.md`.

## [0.4.0] - 2026-05-09

### Added

- Tkinter desktop GUI (`python gui.py`) for selecting an image from the
  computer and viewing every implemented pipeline stage.
- GUI analysis test covering synthetic image loading, detection, segmentation,
  and feature extraction summaries.

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
