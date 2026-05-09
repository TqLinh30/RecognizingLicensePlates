"""
gaussian_blur.py
================

Gaussian smoothing implemented from scratch.

Purpose
-------
A small amount of Gaussian blur is applied **before** edge detection and
thresholding to suppress high-frequency noise (sensor noise, JPEG
artefacts, fine texture) so that gradients reflect actual structure
rather than random fluctuations.

Mathematical background
-----------------------
The 2-D Gaussian function is

                 1                x² + y²
    G(x, y) = -------- · exp( - ---------- )
              2π σ²                2 σ²

A Gaussian filter convolves the image with a discretized version of this
function.

Separability trick
------------------
The 2-D Gaussian factorises:

    G(x, y) = G(x) · G(y)

so a 2-D convolution with a ``k × k`` kernel can be replaced by two 1-D
convolutions of size ``k``.  This reduces complexity from ``O(k²)`` per
pixel to ``O(2k)`` — a huge speed-up for typical kernel sizes (5, 7, 9).
We exploit this here.

Border handling
---------------
We use **reflect** padding (mirror the image at the boundary).  This
avoids the dark frame that zero-padding would produce and the streaking
that "edge / replicate" padding can cause near strong gradients close
to the border.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Kernel construction
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    """
    Build a 1-D, normalized Gaussian kernel.

    Parameters
    ----------
    size : int
        Length of the kernel.  Must be odd and ≥ 3 so it has a
        well-defined centre.
    sigma : float
        Standard deviation, in pixels.

    Returns
    -------
    np.ndarray
        ``float32`` array of shape ``(size,)`` summing to 1.0.
    """
    if size < 3 or size % 2 == 0:
        raise ValueError(
            f"Gaussian kernel size must be odd and ≥ 3; got {size}."
        )
    if sigma <= 0:
        raise ValueError(f"Sigma must be positive; got {sigma}.")

    # Coordinates are centred on zero: e.g. for size=5 we want [-2,-1,0,1,2].
    # The integer division `size // 2` gives the half-width.
    half = size // 2
    x = np.arange(-half, half + 1, dtype=np.float32)

    # Evaluate the Gaussian at each integer offset.
    # The leading 1/(sqrt(2π)σ) constant cancels out after normalization,
    # so we can skip it and just normalize at the end.
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))

    # Normalize so that the kernel sums to 1.  This guarantees that
    # convolving a constant image returns the same constant — i.e. the
    # filter has unit DC gain and does not change image brightness.
    kernel /= kernel.sum()
    return kernel


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

def _reflect_pad(image: np.ndarray, pad: int) -> np.ndarray:
    """
    Pad a 2-D image by mirroring its borders.

    Reflection handling for ``pad=2`` on a 1-D row ``[a b c d e]``::

        [c b | a b c d e | d c]

    This avoids both the black halo of zero-padding and the "stretched
    edge" artefact of replicate padding.

    NumPy ships with ``np.pad(mode='reflect')`` and we use it here —
    this is a layout helper, not a CV algorithm.
    """
    return np.pad(image, pad_width=pad, mode="reflect")


# ---------------------------------------------------------------------------
# 1-D convolution along a single axis (vectorized)
# ---------------------------------------------------------------------------

def _convolve_1d_horizontal(padded: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve every row of ``padded`` with the 1-D kernel.

    Implementation note
    -------------------
    Rather than looping pixel-by-pixel, we perform the convolution as a
    sum of shifted, weighted copies of the image.  For a kernel of size
    ``k`` this is ``k`` element-wise multiply-adds — fully vectorized in
    NumPy.

    Parameters
    ----------
    padded : np.ndarray
        Already-padded 2-D array, shape ``(H, W + 2*half)``.
    kernel : np.ndarray
        1-D kernel of odd length.

    Returns
    -------
    np.ndarray
        Filtered image of original (un-padded) width.
    """
    k = kernel.size
    # The output width equals the padded width minus the kernel "tail".
    out_width = padded.shape[1] - k + 1

    # Accumulator in float32 to avoid overflow during the weighted sum.
    out = np.zeros((padded.shape[0], out_width), dtype=np.float32)

    # For each kernel tap, add a horizontally-shifted, weighted slice.
    # The slice padded[:, i:i+out_width] is a "view" — NumPy does not
    # actually copy, so this loop is cheap.
    for i in range(k):
        out += kernel[i] * padded[:, i:i + out_width]

    return out


def _convolve_1d_vertical(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve every column with the 1-D kernel.

    Mirrors :func:`_convolve_1d_horizontal` along the other axis.
    """
    half = kernel.size // 2
    # Pad rows only.
    padded = np.pad(image, ((half, half), (0, 0)), mode="reflect")

    k = kernel.size
    out_height = padded.shape[0] - k + 1
    out = np.zeros((out_height, padded.shape[1]), dtype=np.float32)

    for i in range(k):
        out += kernel[i] * padded[i:i + out_height, :]

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Apply a 2-D Gaussian blur to a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, shape ``(H, W)``, dtype ``uint8`` or float.
    kernel_size : int, default 5
        Diameter of the Gaussian kernel.  Must be odd.  A common rule of
        thumb is ``size ≈ 6 σ + 1`` so most of the Gaussian's mass falls
        inside the window.
    sigma : float, default 1.0
        Standard deviation of the Gaussian, in pixels.  Larger sigma →
        stronger blur.

    Returns
    -------
    np.ndarray
        Blurred image, shape ``(H, W)``, dtype ``uint8``.

    Notes
    -----
    Implemented using the **separability** of the 2-D Gaussian: we first
    convolve every row with a 1-D kernel, then every column.  This is
    mathematically identical to a full 2-D convolution but ``size``
    times faster.
    """
    if image.ndim != 2:
        raise ValueError(
            f"gaussian_blur expects a 2-D grayscale image; got shape "
            f"{image.shape}."
        )

    # 1. Build the 1-D kernel once.  We will reuse it for both passes
    #    because the Gaussian is symmetric in x and y.
    kernel = _gaussian_kernel_1d(kernel_size, sigma)
    half = kernel_size // 2

    # 2. Promote to float for the arithmetic.  Doing the whole pipeline
    #    in uint8 would round at every multiplication and lose accuracy.
    img_float = image.astype(np.float32)

    # 3. Horizontal pass: pad the columns, then convolve every row.
    padded = np.pad(img_float, ((0, 0), (half, half)), mode="reflect")
    blurred_h = _convolve_1d_horizontal(padded, kernel)

    # 4. Vertical pass: feed the result of the horizontal pass into the
    #    vertical convolution.
    blurred = _convolve_1d_vertical(blurred_h, kernel)

    # 5. Quantize back to uint8.  We round (not truncate) for accuracy.
    return np.clip(np.round(blurred), 0, 255).astype(np.uint8)
