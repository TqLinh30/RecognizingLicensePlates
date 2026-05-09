# Recognizing License Plates

A license plate recognition (LPR) system built **entirely from scratch** in Python.
This project intentionally avoids high-level computer vision libraries (OpenCV,
scikit-image, etc.) — every image-processing and pattern-recognition algorithm is
implemented manually using only NumPy, with Pillow used solely for file I/O.

The goal is **educational**: to deeply understand the underlying mathematics and
algorithms of classical computer vision and machine learning.

---

## Pipeline Overview

```
Input Image
   │
   ▼
[1] Preprocessing            ← (current step)
   │
   ▼
[2] License Plate Detection
   │
   ▼
[3] Plate Cropping & Normalization
   │
   ▼
[4] Character Segmentation
   │
   ▼
[5] Feature Extraction
   │
   ▼
[6] Character Classification
   │
   ▼
[7] Post-processing & Output
```

---

## Project Structure

```
RecognizingLicensePlates/
├── src/
│   ├── preprocessing/          # Step 1: image preprocessing
│   │   ├── __init__.py
│   │   ├── grayscale.py        # RGB → grayscale conversion
│   │   ├── gaussian_blur.py    # Gaussian smoothing filter
│   │   ├── median_filter.py    # Median filter for noise removal
│   │   ├── histogram.py        # Histogram & equalization
│   │   ├── clahe.py            # Contrast Limited Adaptive HE
│   │   └── thresholding.py     # Otsu thresholding
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_io.py         # Image read/write via Pillow
│   │   └── visualization.py    # Debug visualization helpers
│   └── __init__.py
├── tests/
│   └── test_preprocessing.py   # Unit tests for Step 1
├── data/
│   ├── samples/                # Sample input images
│   └── output/                 # Pipeline output
├── docs/
│   └── step1_preprocessing.md  # Detailed documentation
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Gitflow Branching Model

This project follows the **Gitflow** workflow:

| Branch        | Purpose                                                     |
|---------------|-------------------------------------------------------------|
| `main`        | Production-ready, tagged releases only.                     |
| `develop`     | Integration branch for completed features.                  |
| `feature/*`   | New features (e.g. `feature/step1-preprocessing`).          |
| `release/*`   | Pre-release stabilization (e.g. `release/v0.1.0`).          |
| `hotfix/*`    | Urgent fixes against `main`.                                |
| `bugfix/*`    | Non-urgent fixes against `develop`.                         |

### Standard workflow

```bash
# Start a new feature
git checkout develop
git pull origin develop
git checkout -b feature/step1-preprocessing

# ... commit work ...

# Finish a feature
git checkout develop
git merge --no-ff feature/step1-preprocessing
git branch -d feature/step1-preprocessing
git push origin develop

# Cut a release
git checkout -b release/v0.1.0 develop
# bump version, finalize docs
git checkout main
git merge --no-ff release/v0.1.0
git tag -a v0.1.0 -m "Release v0.1.0 - Preprocessing module"
git checkout develop
git merge --no-ff release/v0.1.0
git branch -d release/v0.1.0
```

### Commit message convention

```
<type>(<scope>): <subject>

Types: feat | fix | docs | style | refactor | test | chore
Examples:
  feat(preprocessing): add Gaussian blur with separable kernel
  fix(clahe): correct tile boundary interpolation
  docs(step1): add algorithm explanation for Otsu's method
```

---

## Installation

```bash
git clone https://github.com/TqLinh30/RecognizingLicensePlates.git
cd RecognizingLicensePlates
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage (Step 1 demo)

```bash
python -m src.preprocessing.demo data/samples/plate.jpg
```

Output is saved under `data/output/`.

---

## Roadmap

- [x] **Step 1**: Preprocessing (grayscale, blur, CLAHE, Otsu)
- [ ] Step 2: License plate detection (Sobel + morphology + connected components)
- [ ] Step 3: Plate cropping & normalization (Hough + affine transform)
- [ ] Step 4: Character segmentation
- [ ] Step 5: Feature extraction (HOG + zoning)
- [ ] Step 6: Classification (KNN baseline → MLP)
- [ ] Step 7: Post-processing & format validation

---

## License

MIT
