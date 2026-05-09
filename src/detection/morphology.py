"""
morphology.py
=============

Mathematical morphology — set-theoretic operations on binary images.

Background
----------
Treat a binary image as a *set* of foreground pixels (where the value is
255).  Morphological operators probe this set with another small set
called the **structuring element** (SE) and either grow, shrink, or
preserve the foreground depending on how the SE fits.

The four classical operations
-----------------------------
* **Dilation (⊕)** — every pixel covered by the SE when its origin is
  on a foreground pixel becomes foreground.  Effect: thickens shapes,
  fills small holes.
* **Erosion (⊖)** — a pixel stays foreground only if the *entire* SE
  fits inside the foreground.  Effect: shrinks shapes, removes thin
  protrusions and small isolated specks.
* **Opening (⊕ ∘ ⊖)** = erode then dilate.  Removes small bright
  artefacts smaller than the SE while preserving larger shapes.
* **Closing (⊖ ∘ ⊕)** = dilate then erode.  Fills small dark gaps
  inside the foreground while preserving larger shapes.

Why we need them in Step 2
--------------------------
After thresholding the X-gradient image, every character produces a
dense cluster of bright vertical strokes.  A single **horizontal closing**
with a long, narrow rectangular SE merges all of those strokes into one
solid blob — exactly what we want before running connected-component
analysis to find the plate's bounding box.

Implementation
--------------
For a binary image we exploit the fact that dilation is equivalent to a
*max filter* over the SE neighbourhood, and erosion is a *min filter*.
We vectorize both using shifted-slice broadcasting (the same trick we
used for the median filter), which is fast and easy to reason about.

Structuring-element conventions
-------------------------------
SEs are 2-D ``uint8`` arrays where ``1`` marks an active cell and ``0``
an inactive one.  Helpers are provided to build common shapes:
``rect(h, w)`` and ``cross(size)``.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Structuring-element factories
# ---------------------------------------------------------------------------

def rect(height: int, width: int) -> np.ndarray:
    """
    Build a solid rectangular structuring element of given height and width.

    Parameters
    ----------
    height, width : int
        Both must be positive.  Sizes can be even, but odd dimensions
        give a well-defined geometric centre — which matters because
        we treat the SE's centre as its origin.

    Returns
    -------
    np.ndarray
        ``uint8`` array of ones, shape ``(height, width)``.
    """
    if height < 1 or width < 1:
        raise ValueError(f"rect dimensions must be ≥ 1; got {height}x{width}.")
    return np.ones((height, width), dtype=np.uint8)


def cross(size: int) -> np.ndarray:
    """
    Build a 4-connectivity cross-shaped structuring element.

    Parameters
    ----------
    size : int
        Side length of the bounding square.  Must be odd and ≥ 3.

    Returns
    -------
    np.ndarray
        ``uint8`` array shaped like ``(size, size)`` with ones along
        the central row and column, zeros elsewhere.
    """
    if size < 3 or size % 2 == 0:
        raise ValueError(f"cross size must be odd and ≥ 3; got {size}.")
    se = np.zeros((size, size), dtype=np.uint8)
    centre = size // 2
    se[centre, :] = 1
    se[:, centre] = 1
    return se


# ---------------------------------------------------------------------------
# Internal: list of (dy, dx) offsets active in the SE
# ---------------------------------------------------------------------------

def _active_offsets(se: np.ndarray) -> list[tuple[int, int]]:
    """
    Return the (dy, dx) offsets of every active SE cell, *relative to the
    SE centre*.

    Example: a 3×1 horizontal SE centred at index 0 produces offsets
    ``[(0, -1), (0, 0), (0, 1)]``.
    """
    h, w = se.shape
    cy, cx = h // 2, w // 2
    ys, xs = np.where(se != 0)
    # `ys - cy` re-centres so the SE origin is at (0, 0).
    return list(zip((ys - cy).tolist(), (xs - cx).tolist()))


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def dilate(image: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Dilate a binary image with the given structuring element.

    Parameters
    ----------
    image : np.ndarray
        Binary image.  Any non-zero value is treated as foreground.
    se : np.ndarray
        Structuring element.

    Returns
    -------
    np.ndarray
        Dilated image with values in {0, 255}, dtype ``uint8``.

    Algorithm
    ---------
    Dilation = max filter over the SE neighbourhood.  We compute the
    pixel-wise *maximum* of the image shifted by every active SE
    offset.  Reflect padding ensures the borders behave sensibly.

    Cost: ``O(|SE| · H · W)`` — the |SE| factor is the count of active
    cells, *not* the bounding box, so a long thin SE is cheap.
    """
    if image.ndim != 2:
        raise ValueError(f"dilate expects a 2-D image; got shape {image.shape}.")

    offsets = _active_offsets(se)
    if not offsets:
        # Empty SE: leave the image untouched but normalise its values.
        return ((image > 0).astype(np.uint8)) * 255

    # Padding sized to the worst-case offset on each side.
    max_dy = max(abs(dy) for dy, _ in offsets)
    max_dx = max(abs(dx) for _, dx in offsets)
    padded = np.pad(image, ((max_dy, max_dy), (max_dx, max_dx)), mode="reflect")

    H, W = image.shape
    # Build the running maximum.  Starting from zeros means each
    # pixel ends up as the max over all shifted contributions.
    out = np.zeros((H, W), dtype=image.dtype)
    for dy, dx in offsets:
        # Slice padded[dy + max_dy : ..., dx + max_dx : ...] gives the
        # un-shifted region after compensating for the padding offset.
        y0 = dy + max_dy
        x0 = dx + max_dx
        shifted = padded[y0 : y0 + H, x0 : x0 + W]
        np.maximum(out, shifted, out=out)

    # Normalize to {0, 255}.
    return ((out > 0).astype(np.uint8)) * 255


def erode(image: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Erode a binary image with the given structuring element.

    Parameters
    ----------
    image : np.ndarray
        Binary image.
    se : np.ndarray
        Structuring element.

    Returns
    -------
    np.ndarray
        Eroded image, values in {0, 255}, dtype ``uint8``.

    Algorithm
    ---------
    Erosion = min filter over the SE neighbourhood.  A pixel survives
    only if **every** SE-shifted copy is foreground, which we enforce by
    taking the running minimum.

    We pad with zeros (constant 0) so the borders are correctly treated
    as background — anything within ``SE_radius`` of the edge cannot
    be a "fully fitting" SE position and must therefore be eroded away.
    """
    if image.ndim != 2:
        raise ValueError(f"erode expects a 2-D image; got shape {image.shape}.")

    offsets = _active_offsets(se)
    if not offsets:
        return ((image > 0).astype(np.uint8)) * 255

    max_dy = max(abs(dy) for dy, _ in offsets)
    max_dx = max(abs(dx) for _, dx in offsets)

    # Zero-padding at the borders so any SE that would hang over the
    # edge sees a 0 and (correctly) erodes that pixel.
    padded = np.pad(
        image, ((max_dy, max_dy), (max_dx, max_dx)),
        mode="constant", constant_values=0,
    )

    H, W = image.shape
    # Initialise with the largest possible value; running min will
    # bring it down.
    out = np.full((H, W), 255, dtype=image.dtype)
    for dy, dx in offsets:
        y0 = dy + max_dy
        x0 = dx + max_dx
        shifted = padded[y0 : y0 + H, x0 : x0 + W]
        np.minimum(out, shifted, out=out)

    return ((out > 0).astype(np.uint8)) * 255


def opening(image: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Morphological opening = erode followed by dilate.

    Removes bright noise specks smaller than the SE while leaving larger
    shapes essentially unchanged.
    """
    return dilate(erode(image, se), se)


def closing(image: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Morphological closing = dilate followed by erode.

    Bridges small dark gaps within the foreground.  This is the workhorse
    operation for our plate detector — applied with a long, narrow
    horizontal SE it merges the per-character vertical-stroke clusters
    into one solid plate-shaped blob.
    """
    return erode(dilate(image, se), se)
