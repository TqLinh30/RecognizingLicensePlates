# Recognizing License Plates

A license plate recognition (LPR) system built **entirely from scratch** in
Python. The project intentionally avoids high-level computer vision libraries
such as OpenCV and scikit-image. Every image-processing and machine-learning
algorithm is implemented manually with NumPy; Pillow is used only for image
file I/O.

The goal is educational: expose the mathematics and engineering trade-offs
inside a classical ALPR pipeline.

---

## Pipeline Overview

```
Input Image
   |
   v
[1] Preprocessing
   |
   v
[2] License Plate Detection
   |
   v
[3] Plate Cropping & Normalization
   |
   v
[4] Character Segmentation
   |
   v
[5] Feature Extraction
   |
   v
[6] Character Classification
   |
   v
[7] Post-processing & Output
```

---

## Project Structure

```
RecognizingLicensePlates/
├── src/
│   ├── preprocessing/          # Step 1: grayscale, blur, CLAHE, Otsu
│   ├── detection/              # Step 2: Sobel, morphology, connected components
│   ├── normalization/          # Step 3: crop, Hough skew, rotate, resize
│   ├── segmentation/           # Step 4: connected-component character segmentation
│   ├── features/               # Step 5: HOG + zoning descriptors
│   ├── classifiers/            # Step 6: KNN baseline + NumPy MLP
│   ├── recognition/            # Step 7: end-to-end orchestration + postprocessing
│   ├── datasets/               # EMNIST downloader and IDX parser
│   ├── gui/                    # Desktop GUI for selecting and analyzing images
│   └── utils/                  # Image I/O and debug visualization helpers
├── tests/
│   ├── test_preprocessing.py
│   ├── test_detection.py
│   ├── test_normalization.py
│   ├── test_segmentation.py
│   ├── test_features.py
│   ├── test_classifiers.py
│   └── test_recognition.py
├── docs/
│   ├── step1_preprocessing.md
│   ├── step2_detection.md
│   ├── step3_to_step7_recognition.md
│   ├── training_data.md
│   ├── commitflow.md
│   └── gitflow.md
├── data/
│   ├── samples/
│   ├── output/
│   └── models/
├── scripts/
│   ├── train_synthetic_mlp.py
│   └── train_emnist_mlp.py
├── requirements.txt
├── gui.py
├── CHANGELOG.md
└── README.md
```

---

## Installation

```bash
git clone https://github.com/TqLinh30/RecognizingLicensePlates.git
cd RecognizingLicensePlates
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS, activate the environment with:

```bash
source venv/bin/activate
```

---

## Usage

Desktop GUI:

```bash
python gui.py
```

Or:

```bash
python -m src.gui.app
```

The GUI lets you choose an image from your computer and displays each
intermediate stage: preprocessing, plate detection, normalization, character
segmentation, feature extraction, and OCR if a trained model exists.

The repository includes a starter printed-character OCR model:

```text
data/models/plate_synthetic_mlp.npz
data/models/plate_pixel_templates.npz
data/models/plate_zoning_templates.npz
```

Train or refresh it:

```bash
python -m scripts.train_synthetic_mlp
python -m scripts.train_pixel_template
python -m scripts.train_zoning_template
```

The GUI loads these automatically and shows raw OCR plus top-3 alternatives per character.
EMNIST training is still available as a fallback with
`python -m scripts.train_emnist_mlp --download`. See
[`docs/training_data.md`](docs/training_data.md) for details and faster training
options.

Step 1 preprocessing demo:

```bash
python -m src.preprocessing.demo data/samples/plate.jpg
```

Step 2 detection demo:

```bash
python -m src.detection.demo data/samples/car.jpg
```

End-to-end recognition is exposed as a Python API. It expects a fitted
classifier, such as `KNNClassifier` or `MLPClassifier`:

```python
from src.recognition import recognize_license_plate

result = recognize_license_plate(image_array, fitted_classifier)
print(result.text)
```

---

## Roadmap

- [x] Step 1: Preprocessing (grayscale, blur, CLAHE, Otsu)
- [x] Step 2: License plate detection (Sobel + morphology + connected components)
- [x] Step 3: Plate cropping & normalization (Hough + affine transform)
- [x] Step 4: Character segmentation
- [x] Step 5: Feature extraction (HOG + zoning)
- [x] Step 6: Classification (KNN baseline + MLP)
- [x] Step 7: Post-processing & format validation

Next practical milestone: collect/prepare a labeled character dataset,
train `KNNClassifier` and `MLPClassifier`, then measure real plate
accuracy end to end.

---

## Gitflow Branching Model

This project follows Gitflow:

| Branch | Purpose |
|---|---|
| `main` | Production-ready tagged releases. |
| `develop` | Integration branch for completed features. |
| `feature/*` | New feature work. |
| `release/*` | Pre-release stabilization. |
| `hotfix/*` | Urgent fixes against `main`. |

Commit convention:

```text
<type>(<scope>): <subject>
```

Examples:

```text
feat(preprocessing): add Gaussian blur with separable kernel
fix(clahe): correct tile boundary interpolation
docs(step1): add Otsu derivation
```

For the detailed atomic commit workflow, see
[`docs/commitflow.md`](docs/commitflow.md).

---

## License

MIT
