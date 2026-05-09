# Step 1 — Preprocessing

This document describes the design, algorithms, and trade-offs of the
preprocessing module, which is the first stage of the license-plate
recognition pipeline.

## Goal

Turn a raw color photograph into a clean, contrast-enhanced binary
image where the characters of the plate stand out crisply against the
background. The downstream Step-2 detector relies on strong vertical
edges and well-defined connected components, both of which depend on
good preprocessing.

## Pipeline

```
   ┌────────────────────┐
   │     RGB image      │
   └─────────┬──────────┘
             │   rgb_to_grayscale  (BT.601 luminance)
             ▼
   ┌────────────────────┐
   │     Grayscale      │
   └─────────┬──────────┘
             │   gaussian_blur     (separable, 3×3, σ=1.0)
             ▼
   ┌────────────────────┐
   │      Blurred       │
   └─────────┬──────────┘
             │   clahe             (8×8 tiles, clip=2.0)
             ▼
   ┌────────────────────┐
   │  CLAHE-enhanced    │
   └─────────┬──────────┘
             │   otsu_threshold    (invert=True)
             ▼
   ┌────────────────────┐
   │      Binary        │
   └────────────────────┘
```

## Algorithms in detail

### 1. Grayscale conversion (`grayscale.py`)

Uses the **ITU-R BT.601** luminance formula:

> Y = 0.299 · R + 0.587 · G + 0.114 · B

Implementation note: vectorized as a single `np.dot` against a
3-element weight vector. This is significantly faster than the naive
expression `0.299*R + 0.587*G + 0.114*B` because NumPy can dispatch it
to a BLAS call.

### 2. Gaussian blur (`gaussian_blur.py`)

Removes high-frequency noise (sensor noise, JPEG artefacts) before
edge-sensitive operations.

The implementation exploits the **separability** of the 2-D Gaussian:

> G(x, y) = G(x) · G(y)

This reduces a 2-D convolution from O(k²) to O(2k) per pixel.
We perform two 1-D convolutions: first horizontal, then vertical.

Border handling: **reflect padding** (mirror at the boundary). This
avoids both the dark halo of zero-padding and the streaks of replicate
padding.

### 3. Median filter (`median_filter.py`)

Optional alternative to Gaussian blur, useful for salt-and-pepper noise.
Implemented as a vectorized stack of `k²` shifted copies followed by
`np.median` along the neighbour axis.

### 4. Histogram equalization (`histogram.py`)

The classical, **global** equalization:

> T(v) = round( (L − 1) · (CDF(v) − CDF_min) / (N − CDF_min) )

Useful when the entire image suffers from poor contrast. We rarely use
it directly because license-plate photos usually have **uneven lighting**
that benefits from CLAHE instead.

### 5. CLAHE (`clahe.py`)

Three-step adaptive equalization:

1. **Tile decomposition**: image is split into a grid (default 8×8).
2. **Per-tile equalization** with a contrast cap. Histogram bins above
   `clip_limit · tile_size / 256` are clipped and the excess is
   redistributed uniformly. This prevents contrast from being amplified
   beyond a controlled bound, which would otherwise blow up noise in
   flat regions.
3. **Bilinear interpolation between tile maps**: every pixel's output
   is the bilinear blend of the four nearest tile lookup tables. This
   eliminates the visible tile-boundary artefacts that would otherwise
   be present.

### 6. Otsu thresholding (`thresholding.py`)

Automatic threshold selection by maximising the **between-class variance**:

> σ²_b(t) = ω₀(t) · ω₁(t) · ( μ₀(t) − μ₁(t) )²

Implemented in the closed-form O(L) formulation (no inner loop).
Returns both the binary image and the chosen threshold so that callers
can log or visualize it.

The `invert=True` option makes dark characters become 255 (white) so
that later morphological operations naturally treat them as foreground.

## Configuration

All parameters are exposed via `PreprocessConfig`:

| Parameter           | Default     | Effect                                     |
|---------------------|-------------|--------------------------------------------|
| `blur_kernel_size`  | 3           | Larger → stronger smoothing                |
| `blur_sigma`        | 1.0         | Larger → stronger smoothing                |
| `clahe_grid`        | (8, 8)      | More tiles → more locally adaptive         |
| `clahe_clip_limit`  | 2.0         | Larger → more contrast (and more noise)    |
| `otsu_invert`       | True        | True for dark characters on light plates   |

## Performance

All operations are O(H·W·k) or better and fully vectorized in NumPy.
A 1024×768 input runs end-to-end in about 100 ms on a laptop CPU.

## What gets used downstream

Step 2 (license-plate detection) consumes the **`enhanced`** image
(grayscale, contrast-stretched) for Sobel-based edge detection. The
**`binary`** output is used by Step 4 (character segmentation) inside
the cropped plate.
