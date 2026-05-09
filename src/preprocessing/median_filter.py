"""
median_filter.py
================

Median filter — an order-statistic, edge-preserving denoiser.

When to use this instead of Gaussian blur
-----------------------------------------
* Salt-and-pepper noise (random black/white pixels): the median ignores
  outliers, while a Gaussian blur **spreads** them.
* Old / scanned images, low-light photographs.

Algorithm
---------
For every pixel we look at a ``k × k`` neighbourhood (``k`` odd) and
output the *median* of those ``k²`` values.  Unlike a convolution, this
operation is **non-linear** — it cannot be expressed as a weighted sum
and therefore cannot be made separable.  We accept the ``O(k²)`` cost
because typical kernel sizes are small (3 or 5).

Vectorized implementation
-------------------------
Looping pixel-by-pixel in pure Python is far too slow.  We use NumPy
broadcasting: build a ``(k², H, W)`` stack where each plane is a shifted
copy of the original image.  ``np.median`` along axis 0 then produces
the output in a single call.

Memory
------
The intermediate stack is ``k² × H × W`` floats.  For ``k = 5`` and a
1000×1000 image that is 25 MB — perfectly fine on a laptop.  For very
large images you'd need a streaming implementation; we ignore that here
because license-plate photos are nowhere near that size.
"""

from __future__ import annotations

import numpy as np


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply a median filter to a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, shape ``(H, W)``.
    kernel_size : int, default 3
        Diameter of the square neighbourhood.  Must be odd and ≥ 3.
        Larger kernels remove more noise but also blur more detail.

    Returns
    -------
    np.ndarray
        Filtered image, same shape, dtype ``uint8``.
    """
    # ------------------------------------------------------------------
    # 1. Validate inputs.
    # ------------------------------------------------------------------
    if image.ndim != 2:
        raise ValueError(
            f"median_filter expects a 2-D grayscale image; got shape "
            f"{image.shape}."
        )
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be odd and ≥ 3; got {kernel_size}."
        )

    # ------------------------------------------------------------------
    # 2. Reflect-pad so the kernel can be centred on every pixel,
    #    including those at the borders.
    # ------------------------------------------------------------------
    half = kernel_size // 2
    padded = np.pad(image, pad_width=half, mode="reflect")

    H, W = image.shape

    # ------------------------------------------------------------------
    # 3. Stack every neighbour position as its own plane.
    #
    #    For a 3×3 kernel we end up with 9 planes, each the same size as
    #    the output, where plane (dy, dx) contains the pixel at offset
    #    (dy, dx) from the kernel centre for every output location.
    #
    #    Indexing trick: padded[dy : dy + H, dx : dx + W] is a view that
    #    extracts a (H, W) sub-array starting at (dy, dx) of the padded
    #    image — exactly the contribution we need from one neighbour.
    # ------------------------------------------------------------------
    stack = np.empty((kernel_size * kernel_size, H, W), dtype=image.dtype)
    plane = 0
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            stack[plane] = padded[dy : dy + H, dx : dx + W]
            plane += 1

    # ------------------------------------------------------------------
    # 4. Take the median along the "neighbour" axis.
    #
    #    ``np.median`` returns float64.  We cast back to uint8 at the
    #    end.  For an odd ``k²`` the median of integers is an integer,
    #    so the cast is exact; for an even count it would be the average
    #    of the two middle values, but k² is always odd here.
    # ------------------------------------------------------------------
    out = np.median(stack, axis=0)
    return out.astype(np.uint8)
