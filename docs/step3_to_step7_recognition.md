# Steps 3-7: Normalization, Segmentation, Features, Classification, Recognition

This document summarizes the remaining ALPR stages added in release `v0.3.0`.
All algorithms are implemented from scratch with NumPy.

---

## Step 3: Plate Cropping & Normalization

Modules:

- `src/normalization/hough_transform.py`
- `src/normalization/geometric_transform.py`
- `src/normalization/plate_normalizer.py`

Pipeline:

1. Crop the detected `(x, y, width, height)` box with a configurable margin.
2. Compute a Sobel magnitude edge image inside the crop.
3. Run a Hough transform only around near-horizontal lines.
4. Estimate skew from vote-weighted Hough peaks.
5. Rotate by the negative skew angle using inverse mapping and bilinear interpolation.
6. Resize to a canonical shape, default `80x240`.

The Hough search is intentionally constrained around `90 +/- angle_limit` degrees
because plate borders are close to horizontal, while character strokes produce
many distracting vertical edges.

---

## Step 4: Character Segmentation

Module:

- `src/segmentation/char_segmentation.py`

Pipeline:

1. Apply Otsu thresholding with inversion so dark characters become white foreground.
2. Optionally clean small noise with morphological opening.
3. Run two-pass connected-component labeling.
4. Filter components by height, width, area, fill ratio, and border contact.
5. Sort components left-to-right, with a simple two-row split when vertical gaps are large.
6. Normalize each character to a `32x32` binary canvas while preserving aspect ratio.

---

## Step 5: Feature Extraction

Modules:

- `src/features/hog.py`
- `src/features/zoning.py`
- `src/features/extractor.py`

HOG defaults for a `32x32` character:

- cell size: `8`
- cells: `4x4`
- block size: `2x2`
- bins: `9`
- final HOG length: `324`

Zoning defaults:

- grid: `4x4`
- final zoning length: `16`

Combined default vector length:

```text
324 + 16 = 340
```

---

## Step 6: Character Classification

Modules:

- `src/classifiers/knn.py`
- `src/classifiers/mlp.py`

`KNNClassifier` stores all training vectors and predicts by Euclidean nearest
neighbours. It is useful as a baseline because it has no real training phase.

`MLPClassifier` implements:

- dense layers,
- ReLU hidden activations,
- softmax output,
- cross-entropy loss,
- mini-batch gradient descent,
- backpropagation.

The default architecture is:

```text
input -> 128 -> 64 -> num_classes
```

---

## Step 7: Postprocessing & Full Pipeline

Modules:

- `src/recognition/postprocessing.py`
- `src/recognition/pipeline.py`

The full pipeline is:

```python
from src.recognition import recognize_license_plate

result = recognize_license_plate(image_array, fitted_classifier)
print(result.text)
```

The classifier must expose:

- `predict_proba(X)`, returning an `(N, C)` probability matrix,
- `classes_`, an array of class labels.

Postprocessing applies a simplified Vietnamese plate format:

- first two slots: digits,
- third slot: letter,
- remaining slots: digits.

This enables position-aware corrections such as `O -> 0`, `S -> 5`, and
`0 -> O` in the appropriate slots.
