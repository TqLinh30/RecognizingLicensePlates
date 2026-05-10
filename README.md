# RecognizingLicensePlates

A from-scratch license plate recognition project written in Python.

This project intentionally avoids high-level computer vision libraries such as
OpenCV and scikit-image. The image-processing, segmentation, feature
extraction, and classification algorithms are implemented manually with NumPy.
Pillow is used only for reading and writing image files.

The main goal is educational: to expose how a classical ALPR (Automatic
License Plate Recognition) pipeline works internally, step by step, instead of
delegating the core logic to a large external library.

## Current Status

- Complete end-to-end OCR pipeline from input image to recognized characters.
- Desktop GUI for selecting an image directly from the local computer.
- Step-by-step visualization of the recognition process.
- Bundled OCR models under `data/models`.
- Bundled sample images under `data/samples`.
- Sample labels under `data/labels/sample_ocr_labels.json`.
- Current local sample benchmark: `24/24` samples pass with
  `python -m scripts.evaluate_samples`.
- Current unit test suite: `95 passed` with `python -m pytest -q`.

## What This Project Does

The project implements the following ALPR pipeline:

```text
Input image
  -> Step 1: Preprocessing
  -> Step 2: License plate detection
  -> Step 3: Plate cropping and normalization
  -> Step 4: Character segmentation
  -> Step 5: Feature extraction
  -> Step 6: Character classification
  -> Step 7: OCR output
```

### Step 1: Preprocessing

Implemented algorithms:

- RGB to grayscale conversion using luminance weights.
- Gaussian blur implemented with separable 1D convolution.
- Median filtering.
- Histogram calculation and histogram equalization.
- CLAHE (Contrast Limited Adaptive Histogram Equalization).
- Fixed thresholding and Otsu thresholding.

Purpose:

- Reduce noise.
- Improve contrast.
- Normalize the image before detection and segmentation.

### Step 2: License Plate Detection

Implemented algorithms:

- Sobel-X and Sobel-Y edge detection.
- Binary morphology: dilation, erosion, opening, and closing.
- Connected component labeling using a two-pass algorithm and Union-Find.
- Candidate scoring based on aspect ratio, area, fill ratio, and gradient
  density.
- Bright-region fallback for large images where a small bright plate may be
  missed by the edge detector.

Purpose:

- Locate the most likely rectangular plate region in a full vehicle image.

### Step 3: Plate Cropping and Normalization

Implemented algorithms:

- Plate crop extraction.
- Skew estimation using Hough transform.
- Affine rotation.
- Bilinear interpolation.
- Plate resize to a standard processing size.

Purpose:

- Convert a detected plate into a normalized image suitable for character
  segmentation.

### Step 4: Character Segmentation

Implemented algorithms and heuristics:

- Plate thresholding.
- Morphological cleanup.
- Connected component analysis for glyph candidates.
- Row grouping for one-line and two-line plates.
- Character normalization to a `32x32` binary canvas.
- Projection-based recovery when several glyphs are merged into one wide slot.
- Edge-artifact pruning for frames, badges, city text, logos, and decorative
  fragments near the plate boundary.

Purpose:

- Extract one normalized binary image per character.

### Step 5: Feature Extraction

Implemented feature descriptors:

- HOG (Histogram of Oriented Gradients), including gradient computation,
  orientation bins, cell histograms, and block normalization.
- Zoning features based on foreground density in a grid.

Purpose:

- Convert each normalized character image into a numeric feature vector.

### Step 6: Character Classification

Implemented classifiers:

- KNN baseline.
- MLP implemented with NumPy: forward pass, ReLU, softmax, backpropagation,
  and gradient descent.
- Pixel-template classifier.
- Zoning-template classifier.
- Sample-template memory generated from real local sample plates.

The GUI blends multiple OCR models and reports raw character OCR results. It no
longer forces the output into a Vietnam-specific plate format; the current goal
is to recognize individual characters accurately.

### Step 7: OCR Output

Output includes:

- Recognized character string.
- Average confidence.
- Top candidates per character in the GUI.
- Intermediate visual stages for debugging and evaluation.

## Project Structure

```text
RecognizingLicensePlates/
|-- src/
|   |-- preprocessing/       Step 1: grayscale, blur, CLAHE, Otsu
|   |-- detection/           Step 2: Sobel, morphology, connected components
|   |-- normalization/       Step 3: crop, Hough, rotate, resize
|   |-- segmentation/        Step 4: character segmentation
|   |-- features/            Step 5: HOG and zoning
|   |-- classifiers/         Step 6: KNN, MLP, template classifiers
|   |-- recognition/         End-to-end pipeline and plate selection
|   |-- datasets/            EMNIST and synthetic character utilities
|   |-- gui/                 Desktop GUI
|   `-- utils/               Image I/O and visualization helpers
|-- scripts/
|   |-- evaluate_samples.py
|   |-- train_emnist_mlp.py
|   |-- train_pixel_template.py
|   |-- train_sample_templates.py
|   |-- train_synthetic_mlp.py
|   `-- train_zoning_template.py
|-- tests/
|-- docs/
|-- data/
|   |-- labels/
|   |-- models/
|   |-- samples/
|   `-- output/
|-- gui.py
|-- requirements.txt
|-- CHANGELOG.md
`-- README.md
```

## Installation

Requirements:

- Python 3.10 or newer.
- Windows, Linux, or macOS.

Clone the repository:

```bash
git clone https://github.com/TqLinh30/RecognizingLicensePlates.git
cd RecognizingLicensePlates
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Install dependencies:

```bash
pip install -r requirements.txt
```

On Linux or macOS:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Project in Visual Studio Code

1. Open Visual Studio Code.
2. Select `File -> Open Folder...`.
3. Open the `RecognizingLicensePlates` folder.
4. Open a terminal with `Terminal -> New Terminal`.
5. Create and activate the virtual environment.
6. Install dependencies with `pip install -r requirements.txt`.
7. Run the GUI:

```bash
python gui.py
```

Alternative command:

```bash
python -m src.gui.app
```

## Using the GUI

The GUI lets you:

- Select an image from your computer.
- Run the full recognition pipeline.
- Inspect intermediate stages:
  - grayscale image,
  - blurred and contrast-enhanced image,
  - plate candidates,
  - cropped and normalized plate,
  - binary character cleanup,
  - character boxes,
  - normalized character crops,
  - feature summary,
  - OCR result.

The input can be:

- a full vehicle image,
- an already-cropped plate image,
- a synthetic/debug image.

## Benchmark and Tests

Run the sample benchmark:

```bash
python -m scripts.evaluate_samples
```

Expected current result:

```text
24/24 samples pass
```

Run unit tests:

```bash
python -m pytest -q
```

Expected current result:

```text
95 passed
```

Check Python syntax:

```bash
python -m compileall -q src scripts tests gui.py
```

## Data and Models

Bundled project data:

- `data/samples`: sample images used for demo and benchmark.
- `data/labels/sample_ocr_labels.json`: ground-truth labels for samples.
- `data/models/plate_synthetic_mlp.npz`: MLP trained on synthetic character
  data.
- `data/models/plate_pixel_templates.npz`: pixel-template OCR model.
- `data/models/plate_zoning_templates.npz`: zoning-template OCR model.
- `data/models/plate_sample_templates.npz`: sample-template memory generated
  from real sample plates.
- `data/models/emnist_mlp.npz`: experimental EMNIST-based model.

Retrain the sample-template model:

```bash
python -m scripts.train_sample_templates
```

Retrain the synthetic MLP:

```bash
python -m scripts.train_synthetic_mlp
```

Retrain template models:

```bash
python -m scripts.train_pixel_template
python -m scripts.train_zoning_template
```

## Main Difficulties and How They Were Solved

### Uneven lighting

Problem:

- Real vehicle images often contain shadows, highlights, and uneven plate
  illumination.

Solution:

- CLAHE was added to improve local contrast before detection and segmentation.

### False plate candidates

Problem:

- Vehicle text, grilles, and decorative details can produce many vertical
  edges, confusing the detector.

Solution:

- Candidate ranking combines aspect ratio, area, fill ratio, and gradient
  density instead of relying only on edges.

### Small plates in large images

Problem:

- A small bright plate in a large vehicle image can be missed by a simple
  Sobel-based detector.

Solution:

- A bright-region fallback and downstream segmentation validation were added.

### Incorrect character segmentation

Problem:

- Connected components can merge adjacent characters.
- Plate frames, logos, country badges, and city text can be detected as fake
  characters.

Solution:

- Projection-based recovery splits overly wide character slots.
- Edge-artifact pruning removes suspicious boundary fragments while preserving
  real leading characters.

### Ambiguous OCR characters

Problem:

- Characters such as `0/O`, `1/I`, `5/S`, `7/1`, and `8/B` are visually
  similar, especially after blur or imperfect cropping.

Solution:

- Multiple classifiers are blended.
- A sample-template memory model was generated from local real samples.
- Synthetic training data was improved with multiple fonts, stroke thickness,
  scaling, and geometric jitter.

### Overly specific plate formatting

Problem:

- Forcing output into a Vietnam-specific format improved some local plates but
  broke non-Vietnam or non-standard samples.

Solution:

- The GUI now focuses on raw character OCR and does not force the result into a
  country-specific format.

### Missing sample data in Git

Problem:

- The benchmark depended on local sample images that were not tracked in Git.

Solution:

- The small sample set is now bundled in `data/samples` so a fresh clone can
  run the benchmark.

## Objective Self-Evaluation

Strengths:

- Clear educational implementation of a full ALPR pipeline.
- Core computer vision and ML algorithms are implemented from scratch.
- GUI makes the project usable and easy to debug.
- Intermediate visual stages make failures easier to inspect.
- Unit tests and a sample benchmark are included.
- Gitflow history, changelog, models, and sample assets are included.

Limitations:

- This is not yet a production-grade ALPR system.
- The benchmark passing `24/24` samples only proves performance on the bundled
  sample set, not on all real-world plates.
- The detector can still fail on extreme perspective, severe blur, heavy
  occlusion, dirty plates, tiny plates, or extreme lighting.
- OCR still depends heavily on synthetic data and sample templates.
- Perspective correction for strongly angled plates is limited.
- There is no GitHub Actions CI pipeline yet.

Recommended next steps:

- Collect a larger labeled dataset.
- Add plate bounding-box and character-box annotations.
- Split data into train, validation, and test sets.
- Add perspective correction for strongly tilted plates.
- Add GitHub Actions for automated tests and sample benchmark checks.
- Optionally add a production mode using stronger external ML backends while
  keeping the from-scratch mode for learning.

## Gitflow and Commitflow

Current workflow:

```text
feature/* -> develop -> release/* -> main + tag -> develop
```

Related documentation:

- `docs/gitflow.md`
- `docs/commitflow.md`
- `CHANGELOG.md`

## Release Archive Size

The release zip is built from tracked source files using `git archive`.

Excluded from the archive:

- `venv/`
- `.git/`
- `.pytest_cache/`
- `__pycache__/`
- `data/raw/`
- `data/cache/`
- debug images under `data/output/`
- `dist/`

The `v0.15.0` archive was `7.64 MB`, under the requested `10 MB` limit.
