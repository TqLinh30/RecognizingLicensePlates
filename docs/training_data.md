# Training Data and OCR Models

The GUI can classify segmented characters when a trained model file exists. It
tries models in this order:

```text
data/models/plate_synthetic_mlp.npz
data/models/plate_zoning_templates.npz
data/models/emnist_mlp.npz
```

The GUI combines `plate_synthetic_mlp.npz` with
`plate_zoning_templates.npz` when both are present. EMNIST is kept as a fallback
baseline.

---

## Why 32x32?

EMNIST source images are `28x28`, but this project normalizes **all OCR inputs**
to `32x32`.

That is intentional:

- the segmentation stage outputs `32x32` binary glyph crops,
- HOG uses `8x8` cells, giving a clean `4x4` cell grid,
- training data and real segmented characters pass through the same
  `normalize_character(..., target_shape=(32, 32))` function.

So `28x28` is only the raw EMNIST format. The classifier always sees `32x32`.

---

## Recommended: Train Synthetic Printed Characters

From the project root:

```bash
python -m scripts.train_synthetic_mlp
```

This command:

1. renders digits `0-9` and uppercase letters `A-Z` with system fonts,
2. applies small rotation, shear, blur, threshold jitter, noise, and morphology,
3. normalizes each glyph to the same `32x32` canvas used by segmentation,
4. extracts HOG + zoning features,
5. trains the NumPy-only MLP classifier,
6. saves `data/models/plate_synthetic_mlp.npz`.

Train the zoning-template model:

```bash
python -m scripts.train_zoning_template
```

This implements fixed-region matching: each `32x32` glyph is split into an
`8x8` grid, each grid cell stores the percentage of white pixels, and inference
compares the character to stored synthetic templates.

The bundled starter model was trained this way with a generated holdout
accuracy of about 90%. It should be a better fit for printed plate characters
than EMNIST handwriting.

---

## Fallback: Train EMNIST

EMNIST is documented by NIST:

```text
https://www.nist.gov/itl/products-and-services/emnist-dataset
```

Train the fallback model:

```bash
python -m scripts.train_emnist_mlp --download
```

This downloads the public EMNIST ByClass archive and trains
`data/models/emnist_mlp.npz`.

The archive itself is large, so it is not committed to this repository. The
training script downloads it into `data/raw/emnist/`, which is ignored by Git.
The downloader tries a public Western Sydney/MARCS mirror first, then historical
NIST URLs, and validates that the downloaded file is a real zip archive.

After a model is saved, run:

```bash
python gui.py
```

The GUI will load the best available model automatically and show Step 6-7 OCR
output.

---

## Faster Or Stronger Training Options

Quick synthetic baseline:

```bash
python -m scripts.train_synthetic_mlp --samples-per-class 150 --epochs 30
```

Stronger synthetic model:

```bash
python -m scripts.train_synthetic_mlp --samples-per-class 1000 --epochs 140
```

EMNIST fallback:

```bash
python -m scripts.train_emnist_mlp --download --samples-per-class 120 --epochs 120 --learning-rate 0.03
```

---

## Important Limitation

The synthetic model is closer to real plates than EMNIST, but it is still not a
replacement for real plate-character data. Real plate accuracy will improve if
you later add:

- real segmented plate-character images,
- synthetic rendered plate fonts,
- augmentation for blur, tilt, noise, and thresholding artifacts.
