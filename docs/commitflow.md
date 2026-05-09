# Commit Flow

This project uses Gitflow for branches and a small, disciplined commit flow for
day-to-day changes. The goal is to keep the history readable enough that each
release tells a clear story.

---

## Core Rules

1. Commit one logical change at a time.
2. Keep commits buildable whenever practical.
3. Run the closest relevant verification before committing.
4. Stage intentionally; avoid `git add .` when unrelated files are dirty.
5. Use Conventional Commits so logs and changelogs stay consistent.

---

## Commit Message Format

```text
<type>(<scope>): <short subject>

<optional body>

<optional footer>
```

Examples:

```text
feat(normalization): add Hough-based skew estimation
test(segmentation): cover two-line plate ordering
docs(commitflow): document atomic commit workflow
fix(thresholding): avoid divide-by-zero warning in Otsu
```

Allowed types:

| Type | Use when |
|---|---|
| `feat` | Adding a user-facing feature or new algorithm |
| `fix` | Correcting a bug |
| `docs` | Documentation-only changes |
| `test` | Adding or correcting tests |
| `refactor` | Restructuring code without changing behavior |
| `perf` | Improving performance |
| `style` | Formatting-only changes |
| `chore` | Tooling, release prep, housekeeping |

Recommended scopes:

```text
preprocessing
detection
normalization
segmentation
features
classifiers
recognition
docs
release
```

---

## Daily Commit Workflow

Start from `develop`:

```bash
git switch develop
git pull origin develop
git switch -c feature/<short-task-name>
```

Work in small slices. Before each commit:

```bash
git status --short
git diff
git diff --staged
```

Stage only the files that belong to the logical change:

```bash
git add src/normalization/hough_transform.py tests/test_normalization.py
git commit -m "feat(normalization): add Hough line voting"
```

If unrelated files are dirty, leave them unstaged and mention them in the final
handoff. This prevents accidental commits of generated output, local data, or a
teammate's work.

---

## Suggested Commit Order For A Pipeline Step

For a new pipeline stage, prefer this sequence:

1. Algorithm implementation:

```bash
git commit -m "feat(features): add HOG descriptor"
```

2. Tests:

```bash
git commit -m "test(features): add HOG and zoning coverage"
```

3. Integration or orchestration:

```bash
git commit -m "feat(recognition): wire features into pipeline"
```

4. Documentation:

```bash
git commit -m "docs(step5): explain feature extraction choices"
```

5. Release prep:

```bash
git commit -m "chore(release): prepare v0.4.0"
```

Small projects can combine implementation and tests in one commit when the
change is tightly coupled, but the commit should still represent one idea.

---

## Verification Before Commit

Use the narrowest useful check while developing:

```bash
python -m compileall -q src tests
python -m pytest tests/test_features.py -q
```

Before merging a feature branch:

```bash
python -m pytest -q
```

If `pytest` is not installed in the environment, run a focused manual script and
record that limitation in the handoff.

---

## Finishing A Feature

Merge finished feature branches into `develop` with a merge commit:

```bash
git switch develop
git merge --no-ff feature/<short-task-name> -m "Merge feature/<short-task-name> into develop"
git branch -d feature/<short-task-name>
```

Then cut a release branch when the feature set is ready:

```bash
git switch -c release/vX.Y.Z
```

After release prep:

```bash
git switch main
git merge --no-ff release/vX.Y.Z -m "Merge release/vX.Y.Z into main"
git tag -a vX.Y.Z -m "Release vX.Y.Z - <summary>"

git switch develop
git merge --no-ff main -m "Merge release/vX.Y.Z into develop"
git branch -d release/vX.Y.Z
```

---

## What Not To Commit

Do not commit:

- virtual environments,
- cache directories,
- generated debug images,
- downloaded datasets,
- trained model binaries unless the release explicitly includes them,
- local credentials or tokens.

The `.gitignore` already excludes the most common generated files. Keep
placeholder `.gitkeep` files only when the empty folder itself is part of the
project structure.
