# Training Data and OCR Model

The GUI can classify segmented characters when this model file exists:

```text
data/models/emnist_mlp.npz
```

The model is trained from the public EMNIST ByClass dataset. EMNIST is
documented by NIST:

```text
https://www.nist.gov/itl/products-and-services/emnist-dataset
```

The archive itself is large, so it is not committed to this repository. The
training script downloads it into `data/raw/emnist/`, which is ignored by Git.
The downloader tries a public Western Sydney/MARCS mirror first, then historical
NIST URLs, and validates that the downloaded file is a real zip archive.

---

## Train The Starter Model

From the project root:

```bash
python -m scripts.train_emnist_mlp --download
```

This command:

1. downloads the official EMNIST `gzip.zip` archive if missing,
2. selects digits `0-9` and uppercase letters `A-Z`,
3. corrects EMNIST orientation with a transpose,
4. normalizes each glyph to `32x32`,
5. extracts the same HOG + zoning features used by the GUI,
6. trains the NumPy-only MLP classifier,
7. saves `data/models/emnist_mlp.npz`.

After the model is saved, run:

```bash
python gui.py
```

The GUI will load the model automatically and show Step 6-7 OCR output.

---

## Faster Or Stronger Training

Quick baseline:

```bash
python -m scripts.train_emnist_mlp --download --samples-per-class 100 --epochs 10
```

Reproduce the bundled starter model:

```bash
python -m scripts.train_emnist_mlp --download --samples-per-class 120 --epochs 120 --learning-rate 0.03
```

The first run spends most of its time downloading EMNIST. Later runs reuse the
local archive.

---

## Important Limitation

EMNIST is a handwriting dataset, while license plates use printed glyphs. This
is good enough for a starter OCR baseline and for wiring the full pipeline, but
real plate accuracy will improve if you later add:

- real segmented plate-character images,
- synthetic rendered plate fonts,
- augmentation for blur, tilt, noise, and thresholding artifacts.
