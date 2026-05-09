"""
histogram.py
============

Histogram, cumulative distribution function (CDF), and global histogram
equalization.

Why histograms matter
---------------------
A histogram tells us how the 256 possible 8-bit intensity values are
distributed in an image.  Useful for:
* diagnosing under- / over-exposed images,
* designing thresholds,
* equalizing contrast.

Histogram equalization
----------------------
When most pixels are clustered around a narrow band of intensities
(e.g. a hazy or backlit photograph), characters on the plate become
hard to see.  Histogram equalization remaps intensities so that the
output histogram is approximately uniform, which **stretches contrast**
across the full 0..255 range.

The remap is the (rescaled) cumulative distribution function:

                  L − 1
    T(v) = round( ───── · CDF(v) )
                   N

where ``L = 256`` (number of intensity levels) and ``N = H × W``
(number of pixels).

Limitation
----------
Global equalization treats the whole image identically, so it can
amplify noise in flat areas and over-stretch already-bright regions.
For uneven lighting we use **CLAHE** (next module) instead.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Histogram & CDF
# ---------------------------------------------------------------------------

def compute_histogram(image: np.ndarray, num_bins: int = 256) -> np.ndarray:
    """
    Compute the intensity histogram of a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, dtype ``uint8`` (values 0..255).
    num_bins : int, default 256
        Number of histogram bins.  For 8-bit images leave at 256 so each
        intensity value has its own bin.

    Returns
    -------
    np.ndarray
        ``int64`` array of shape ``(num_bins,)`` where ``hist[v]`` is
        the count of pixels with intensity ``v``.
    """
    if image.dtype != np.uint8:
        raise ValueError(
            f"compute_histogram expects uint8 input; got {image.dtype}."
        )

    # np.bincount is the fastest way to count integer occurrences.
    # `minlength=num_bins` guarantees a fixed-size output even when the
    # image happens not to use all 256 levels.
    return np.bincount(image.ravel(), minlength=num_bins).astype(np.int64)


def compute_cdf(histogram: np.ndarray) -> np.ndarray:
    """
    Compute the cumulative distribution function from a histogram.

    Parameters
    ----------
    histogram : np.ndarray
        Histogram array.

    Returns
    -------
    np.ndarray
        ``cdf[v] = sum(histogram[0..v])``.

    Notes
    -----
    The CDF is monotonically non-decreasing and reaches the total pixel
    count at the last bin.  We do **not** normalize it here so callers
    can do their own scaling; histogram equalization scales by
    ``(L - 1) / N``.
    """
    return np.cumsum(histogram)


# ---------------------------------------------------------------------------
# Histogram equalization
# ---------------------------------------------------------------------------

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply global histogram equalization to a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, dtype ``uint8``.

    Returns
    -------
    np.ndarray
        Equalized image, same shape, dtype ``uint8``.

    Algorithm
    ---------
    1. Histogram h[v] = number of pixels of intensity v.
    2. CDF c[v] = sum(h[0..v]).
    3. Lookup table  T[v] = round( (L-1) · (c[v] - c_min) / (N - c_min) )
       where ``c_min`` is the smallest non-zero CDF value.  Subtracting
       ``c_min`` ensures that the darkest non-empty intensity maps to 0
       so the dynamic range really does start at zero.
    4. Output[y, x] = T[ Input[y, x] ].
    """
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(
            "histogram_equalization expects a 2-D uint8 image; got shape "
            f"{image.shape}, dtype {image.dtype}."
        )

    # --- 1. Histogram ---
    hist = compute_histogram(image, num_bins=256)

    # --- 2. CDF ---
    cdf = compute_cdf(hist)

    # --- 3. Build the lookup table ---
    # Smallest non-zero CDF value: corresponds to the darkest intensity
    # actually present in the image.
    nonzero = cdf[cdf > 0]
    if nonzero.size == 0:
        # Pathological case: empty image.  Return as-is.
        return image.copy()
    cdf_min = nonzero[0]

    total_pixels = image.size

    # The denominator (N - c_min) cannot be zero unless every pixel has
    # the same value (constant image).  In that case equalization is
    # undefined — return the input unchanged.
    denom = total_pixels - cdf_min
    if denom <= 0:
        return image.copy()

    # Compute the LUT in float to avoid integer overflow in the
    # multiplication, then round and clip.
    lut = np.round((cdf - cdf_min) * 255.0 / denom)
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    # --- 4. Apply the LUT.  This is the cheap part: a single fancy-index
    #        operation runs in O(H·W).  ---
    return lut[image]
