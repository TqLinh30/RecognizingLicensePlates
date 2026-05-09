# Step 2 — License-Plate Detection

This document describes the design, algorithms, and trade-offs of the
detection module, the second stage of the pipeline.

## Goal

Take the contrast-enhanced grayscale image from Step 1 and return a
ranked list of bounding boxes that may contain a license plate. The
top candidate is what Step 3 (cropping & normalization) will work on.

## Pipeline

```
   ┌─────────────────────────┐
   │  CLAHE-enhanced (Step1) │
   └────────────┬────────────┘
                │  sobel_x  (separable, abs value, scaled to uint8)
                ▼
   ┌─────────────────────────┐
   │   |∂I/∂x| gradient      │
   └────────────┬────────────┘
                │  threshold @ 20% of peak
                ▼
   ┌─────────────────────────┐
   │   binary edge map       │
   └────────────┬────────────┘
                │  closing with 3 × 25 horizontal SE
                ▼
   ┌─────────────────────────┐
   │   plate-shaped blobs    │
   └────────────┬────────────┘
                │  connected_components (8-connectivity)
                ▼
   ┌─────────────────────────┐
   │   labelled blobs + stats│
   └────────────┬────────────┘
                │  geometric + density filter, score, sort
                ▼
   ┌─────────────────────────┐
   │   PlateCandidate list   │
   └─────────────────────────┘
```

## Algorithms in detail

### 1. Sobel-X gradient (`sobel.py`)

Detects horizontal intensity changes (vertical strokes). Implemented
as **two 1-D passes** using separability:

> Sx = [1, 2, 1]ᵀ ⊗ [-1, 0, +1]

This is twice as fast as the equivalent 3×3 2-D convolution and gives
identical output. The full `sobel()` function returns both axes plus
the L2 magnitude; `sobel_x()` is a convenience that skips the Y-pass
because plate detection only cares about horizontal gradients.

### 2. Morphological closing (`morphology.py`)

The thresholded gradient is a forest of disconnected vertical bars
(one cluster per character). A horizontal closing with a long, narrow
SE merges them into a single solid blob.

Why a wide SE? The kernel must be **wider than the gap between
characters** but **narrower than the gap between plate and background
text** (e.g. dashboard text, road markings). A 3×25 default works well
for plates that occupy 5-25 % of the image width.

Implementation: dilation = max filter, erosion = min filter, both
vectorized via shifted-slice broadcasting. SE-active offsets are
extracted once and looped over.

### 3. Connected components (`connected_components.py`)

Two-pass labeling with union-find and path compression:

* **Pass 1**: scan in raster order, assign provisional labels by
  inheriting from already-visited neighbours, record equivalences
  in a Union-Find structure.
* **Pass 2**: replace every provisional label with its class root,
  compactify to 1..K.

Per-component statistics (bounding box, area, centroid, aspect ratio,
fill ratio) are computed in one vectorized pass using `np.add.at`,
`np.minimum.at`, and `np.maximum.at` scatters.

8-connectivity is used so that diagonal pixel contacts inside the
closed blob do not split the plate into multiple labels.

### 4. Candidate filtering & scoring (`plate_detector.py`)

For every component we apply **cheap geometric rejects first**:

| Filter                    | Default                 | Rationale                          |
|---------------------------|-------------------------|------------------------------------|
| `min_height_px`           | 15                      | Ignore tiny noise blobs            |
| `min_width_px`            | 40                      | Plates aren't pixel-wide           |
| `min_aspect_ratio`        | 1.5                     | Excludes very tall blobs           |
| `max_aspect_ratio`        | 6.5                     | Excludes very long thin streaks    |
| `min_area_ratio`          | 0.005                   | Plate is at least 0.5 % of image   |
| `max_area_ratio`          | 0.30                    | Plate is at most 30 % of image     |
| `min_fill_ratio`          | 0.30                    | Solid blobs preferred              |
| `min_gradient_density`    | 0.10                    | Plates have many edges inside      |

Survivors are scored by a composite metric:

> score = aspect_score · fill_score · gradient_score

* **aspect_score** is a Gaussian centred at 3.5 (Vietnamese single-line
  ratio) with σ=1.5, so square two-line plates still score reasonably.
* **fill_score** ramps linearly from `min_fill_ratio` to 1.0.
* **gradient_score** ramps linearly from `min_gradient_density` to 0.5.

Multiplication (not sum) makes any single failure heavily penalize the
candidate — a plate-shaped blob with no internal gradient is almost
certainly not a plate.

## Configuration

All knobs live on `DetectionConfig` and pass straight through to the
detector. The defaults are tuned for Vietnamese-style plates in
moderately good photographs.

## Output

```python
DetectionResult(
    gradient,    # uint8, |∂I/∂x| scaled to 0..255
    binary,      # uint8, thresholded gradient {0, 255}
    closed,      # uint8, after morphological closing
    labels,      # int32, connected-component labels
    candidates,  # list[PlateCandidate], sorted by score (best first)
    config,
)
```

Each `PlateCandidate` carries the bounding box (`x, y, width, height`),
the score, and the underlying `ComponentStats` for full traceability.

## What gets used downstream

Step 3 (plate cropping & normalization) consumes:
* `candidates[0].as_box()` — the bounding box to crop out of the
  original image.
* `gradient` — to find the plate's tilt angle via Hough transform
  without recomputing edges.

If the top candidate's score is very low (< 0.05 say), Step 3 should
flag the image as "no plate detected" and abort the pipeline.

## Performance

| Stage                      | Cost                      |
|----------------------------|---------------------------|
| Sobel-X (separable)        | O(H·W)                    |
| Threshold + edge mask      | O(H·W)                    |
| Closing (3×25 horizontal)  | O(H·W·\|SE\|) = O(H·W·28) |
| Connected components       | O(H·W·α(N))               |
| Filter + score             | O(K)                      |

End-to-end on a 1024×768 input: ~250 ms on a laptop CPU. The Pass-1
loop in CCL is the dominant cost (pure Python). For production we'd
JIT it with Numba; here we keep it didactic.
