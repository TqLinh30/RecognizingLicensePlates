# Gitflow Workflow

This project follows the Gitflow branching model. This document is a
quick reference for contributors.

## Branch types

| Branch        | Base       | Merges to       | Lifetime  |
|---------------|------------|-----------------|-----------|
| `main`        | —          | —               | permanent |
| `develop`     | `main`     | —               | permanent |
| `feature/*`   | `develop`  | `develop`       | temporary |
| `release/*`   | `develop`  | `main`+`develop`| temporary |
| `hotfix/*`    | `main`     | `main`+`develop`| temporary |
| `bugfix/*`    | `develop`  | `develop`       | temporary |

`main` is always deployable and tagged with semantic versions. `develop`
is the integration branch where finished features land before being
batched into a release.

## Bootstrap (one time only)

```bash
# Init the repository
git init
git add .
git commit -m "chore: initial project skeleton"
git branch -M main

# Create the develop branch alongside main
git checkout -b develop
git push -u origin main
git push -u origin develop
```

## Working on a feature

```bash
# Always branch off develop
git checkout develop
git pull origin develop
git checkout -b feature/step1-preprocessing

# ... do work ...
git add .
git commit -m "feat(preprocessing): add CLAHE with bilinear interpolation"

# Push to remote periodically
git push -u origin feature/step1-preprocessing

# When done, open a Pull Request into develop.
# After review and merge, delete the feature branch:
git branch -d feature/step1-preprocessing
git push origin --delete feature/step1-preprocessing
```

## Cutting a release

```bash
git checkout develop
git pull origin develop
git checkout -b release/v0.1.0

# Bump version, finalise CHANGELOG, etc.
git commit -am "chore(release): bump version to v0.1.0"

# Merge to main and tag.
git checkout main
git merge --no-ff release/v0.1.0
git tag -a v0.1.0 -m "Release v0.1.0 — Preprocessing module"
git push origin main --tags

# Merge back to develop.
git checkout develop
git merge --no-ff release/v0.1.0
git push origin develop

# Delete the release branch.
git branch -d release/v0.1.0
git push origin --delete release/v0.1.0
```

## Hotfixing production

```bash
git checkout main
git checkout -b hotfix/fix-grayscale-overflow

# ... fix the bug ...
git commit -am "fix(grayscale): clip values before uint8 cast"

# Merge to main.
git checkout main
git merge --no-ff hotfix/fix-grayscale-overflow
git tag -a v0.1.1 -m "Hotfix v0.1.1"

# Merge to develop too.
git checkout develop
git merge --no-ff hotfix/fix-grayscale-overflow

git push origin main develop --tags
git branch -d hotfix/fix-grayscale-overflow
```

## Commit message convention

For the full atomic-commit workflow, see [`docs/commitflow.md`](commitflow.md).

```
<type>(<scope>): <subject>

<body>

<footer>
```

| Type      | When                                              |
|-----------|---------------------------------------------------|
| `feat`    | A new feature                                     |
| `fix`     | A bug fix                                         |
| `docs`    | Documentation only                                |
| `style`   | Whitespace, formatting (no logic change)          |
| `refactor`| Code change that neither fixes a bug nor adds feat|
| `perf`    | Performance improvement                           |
| `test`    | Adding or correcting tests                        |
| `chore`   | Build process, tooling, dependencies              |

### Examples

```
feat(preprocessing): add separable Gaussian blur
fix(clahe): correct tile boundary interpolation at corners
docs(step1): document Otsu's between-class variance derivation
refactor(thresholding): vectorize Otsu without inner loop
test(preprocessing): add edge cases for constant images
chore: add .gitignore for Python projects
```

## Per-step branch naming

Each pipeline step gets its own feature branch:

| Branch                              | Step                                |
|-------------------------------------|-------------------------------------|
| `feature/step1-preprocessing`       | This branch.                        |
| `feature/step2-plate-detection`     | Sobel + morphology + CC.            |
| `feature/step3-plate-normalization` | Hough + affine.                     |
| `feature/step4-char-segmentation`   | Connected-component segmentation.   |
| `feature/step5-feature-extraction`  | HOG + zoning.                       |
| `feature/step6-classification`      | KNN + MLP.                          |
| `feature/step7-postprocessing`      | Format validation + correction.     |

After each step's feature branch is merged into `develop`, we cut a
minor release (`v0.1.0`, `v0.2.0`, …).
