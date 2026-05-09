"""
thresholding.py
===============

Image binarization.  Two methods:

* :func:`fixed_threshold` — pick a threshold by hand.  Useful when you
  already know the lighting conditions.
* :func:`otsu_threshold`   — choose the threshold automatically by
  maximising between-class variance.

Otsu's method (1979)
--------------------
Suppose every pixel is labelled as either "foreground" (intensity >
threshold ``t``) or "background" (intensity ≤ ``t``).  As ``t`` varies
between 0 and 255, the two classes have means ``μ0(t)``, ``μ1(t)`` and
proportions ``ω0(t)``, ``ω1(t)``.  Otsu defines the **between-class
variance**

    σ²_b(t) = ω0(t) · ω1(t) · ( μ0(t) − μ1(t) )²

and picks the ``t*`` that maximises it.  Intuitively, when ``t*`` is
the right cut, the two classes are well separated (large mean gap) and
neither is empty (both ω are far from zero).

Equivalent to: minimising the *within-class* variance — the choice
that makes each side of the threshold as homogeneous as possible.

Time complexity
---------------
A naive O(L²) loop tries every threshold and computes both class means
from scratch.  We use the O(L) recursive formulation: keep running sums
of the histogram and of ``i · h[i]`` and compute everything from those.
"""

from __future__ import annotations

import numpy as np

from src.preprocessing.histogram import compute_histogram


# ---------------------------------------------------------------------------
# Fixed threshold
# ---------------------------------------------------------------------------

def fixed_threshold(
    image: np.ndarray,
    threshold: int = 128,
    invert: bool = False,
) -> np.ndarray:
    """
    Binarize an image at a hand-picked threshold.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    threshold : int, default 128
        Cut-off value.  Pixels with ``intensity > threshold`` become 255,
        the rest 0.
    invert : bool, default False
        If True, swap foreground and background.  Useful for license
        plates where characters are dark on a light background — after
        inversion the characters are 255 (white) and the morphological
        operations of later stages naturally treat them as foreground.

    Returns
    -------
    np.ndarray
        Binary image, dtype ``uint8`` with values in {0, 255}.
    """
    if image.ndim != 2:
        raise ValueError(
            f"fixed_threshold expects a 2-D image; got shape {image.shape}."
        )
    if not 0 <= threshold <= 255:
        raise ValueError(
            f"threshold must be in [0, 255]; got {threshold}."
        )

    # Comparison operators on uint8 arrays return boolean arrays.
    # Multiplying by 255 (uint8) gives the canonical {0, 255} output.
    mask = image > threshold if not invert else image <= threshold
    return (mask.astype(np.uint8)) * 255


# ---------------------------------------------------------------------------
# Otsu's method
# ---------------------------------------------------------------------------

def otsu_threshold(image: np.ndarray, invert: bool = False) -> tuple[np.ndarray, int]:
    """
    Binarize an image using Otsu's automatic threshold selection.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, dtype ``uint8``.
    invert : bool, default False
        Swap foreground / background after thresholding.

    Returns
    -------
    binary : np.ndarray
        Binary image with values in {0, 255}.
    threshold : int
        The threshold chosen by Otsu's method.  Returned in addition to
        the binary image so callers can log or visualize it.

    Algorithm
    ---------
    Let ``h[i]`` be the histogram and ``N`` the total pixel count.
    Probabilities ``p[i] = h[i] / N``.

    Cumulative quantities for class 0 (intensities ≤ t):
        ω0(t) = Σ_{i=0..t} p[i]
        μ0(t) = ( Σ_{i=0..t} i · p[i] ) / ω0(t)

    Class 1 quantities are the complements.

    Define the *global* mean ``μ_T = Σ i · p[i]`` and the cumulative
    mean up to t  ``μ(t) = Σ_{i=0..t} i · p[i]``.  Then

        σ²_b(t) = ( μ_T · ω0(t) − μ(t) )²  /  ( ω0(t) · ( 1 − ω0(t) ) )

    which depends only on cumulative sums and avoids any inner loop.
    """
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(
            "otsu_threshold expects a 2-D uint8 image; got shape "
            f"{image.shape}, dtype {image.dtype}."
        )

    # ------------------------------------------------------------------
    # 1. Histogram and probabilities.
    # ------------------------------------------------------------------
    hist = compute_histogram(image, num_bins=256)
    total = image.size
    p = hist.astype(np.float64) / total          # probability of each level

    # ------------------------------------------------------------------
    # 2. Cumulative sums and global mean.
    # ------------------------------------------------------------------
    levels = np.arange(256, dtype=np.float64)

    omega = np.cumsum(p)                   # ω0(t) for t = 0..255
    mu    = np.cumsum(p * levels)          # μ(t)
    mu_T  = mu[-1]                         # global mean

    # ------------------------------------------------------------------
    # 3. Between-class variance, computed in closed form.
    #
    #    Numerical care: divide-by-zero appears when ω0 is 0 or 1
    #    (a threshold below all values, or above all values).  We mask
    #    those cases to -1 so they cannot win the argmax.  We do the
    #    division with `np.divide`'s ``where=`` argument so NumPy
    #    skips the unsafe entries instead of warning about them.
    # ------------------------------------------------------------------
    denom = omega * (1.0 - omega)
    numer = (mu_T * omega - mu) ** 2

    sigma_b_squared = np.full_like(denom, -1.0)
    np.divide(numer, denom, out=sigma_b_squared, where=denom > 0)

    # ------------------------------------------------------------------
    # 4. Pick the threshold that maximises σ²_b.
    # ------------------------------------------------------------------
    threshold = int(np.argmax(sigma_b_squared))

    # ------------------------------------------------------------------
    # 5. Apply the threshold.
    # ------------------------------------------------------------------
    binary = fixed_threshold(image, threshold=threshold, invert=invert)
    return binary, threshold
