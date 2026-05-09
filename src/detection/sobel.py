"""
sobel.py
========

Sobel edge detection — directional gradient operators implemented from
scratch.

Why Sobel?
----------
The Sobel operator computes a discrete approximation of the image
gradient.  It is the simplest, fastest edge detector that is robust to
small amounts of noise, thanks to its built-in 1-2-1 smoothing along
the axis perpendicular to the gradient direction.

Why specifically the X-gradient for license plates?
---------------------------------------------------
A license plate is a region with **dense vertical strokes** (the left
and right edges of every character).  The X-gradient — sensitive to
horizontal intensity changes — therefore lights up brightly inside a
plate and stays dim on smooth surfaces like a car body or sky.

This is the single most important observation behind the classical
"Sobel + morphology" detector we will build in :mod:`plate_detector`.

Kernels
-------
The 3×3 Sobel kernels are::

       Sx =  [ -1  0 +1 ]        Sy =  [ -1 -2 -1 ]
             [ -2  0 +2 ]              [  0  0  0 ]
             [ -1  0 +1 ]              [ +1 +2 +1 ]

Both factor as outer products of a 1-D smoothing kernel and a 1-D
derivative kernel::

       Sx = [1; 2; 1]  ⊗  [-1, 0, +1]
       Sy = [-1; 0; +1] ⊗  [1, 2, 1]

so we can apply them as **separable** 1-D convolutions, exactly as we
did for Gaussian blur.  This is twice as fast and (for our use case)
gives identical results.

Output range
------------
For an 8-bit image, the per-axis gradient lies in roughly
``[-1020, +1020]`` (sum of four ``±255`` weighted contributions).  We
expose three convenience returns:

* ``gx``, ``gy``  — signed ``float32`` gradients,
* ``magnitude``   — ``sqrt(gx² + gy²)`` in ``float32``,
* ``magnitude_uint8`` — magnitude scaled to ``[0, 255]`` for display
  and for thresholding-based pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SobelResult:
    """Bundle the four arrays produced by :func:`sobel` for convenient access."""
    gx: np.ndarray             # signed horizontal gradient (float32)
    gy: np.ndarray             # signed vertical gradient (float32)
    magnitude: np.ndarray      # gradient magnitude (float32)
    magnitude_uint8: np.ndarray  # magnitude scaled to [0, 255] (uint8)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _convolve_1d(
    image: np.ndarray,
    kernel: np.ndarray,
    axis: int,
) -> np.ndarray:
    """
    Apply a 1-D convolution along the given axis with reflect padding.

    Identical in spirit to the helper used in :mod:`gaussian_blur`, but
    reusable for arbitrary 1-D kernels (which is what we need here
    because the Sobel kernels are not symmetric around their centre on
    one of the two axes).

    Parameters
    ----------
    image : np.ndarray
        2-D float array.
    kernel : np.ndarray
        1-D kernel of odd length.  Stored "as-is" (we do *not* flip it
        — the Sobel literature defines the kernels in correlation form,
        not convolution form, and our shifted-slice loop performs
        correlation).
    axis : int
        ``1`` for horizontal (kernel slides along columns),
        ``0`` for vertical.

    Returns
    -------
    np.ndarray
        Filtered image of the same shape as the input.
    """
    half = kernel.size // 2
    k = kernel.size

    # Pad only along the axis we slide on.
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (half, half)
    padded = np.pad(image, pad_width, mode="reflect")

    out = np.zeros_like(image, dtype=np.float32)

    # Vectorized correlation: each kernel tap is a weighted, axis-shifted
    # slice of the padded image.
    if axis == 1:
        # Horizontal: slide along columns.
        for i in range(k):
            out += kernel[i] * padded[:, i : i + image.shape[1]]
    else:
        # Vertical: slide along rows.
        for i in range(k):
            out += kernel[i] * padded[i : i + image.shape[0], :]

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# 1-D smoothing and derivative kernels for the 3×3 Sobel operator.
# Stored at module scope so they're built once and reused for every call.
_SMOOTH = np.array([1.0, 2.0, 1.0], dtype=np.float32)
_DERIV = np.array([-1.0, 0.0, 1.0], dtype=np.float32)


def sobel(image: np.ndarray) -> SobelResult:
    """
    Compute the Sobel gradients and gradient magnitude of a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, dtype ``uint8`` or float, shape ``(H, W)``.

    Returns
    -------
    SobelResult
        Dataclass with ``gx``, ``gy``, ``magnitude`` (all float32) and
        ``magnitude_uint8`` (scaled to [0, 255]).

    Notes
    -----
    Implementation uses the **separability** of Sobel: each gradient is
    produced by two 1-D passes (one smoothing, one differentiation),
    instead of a 3×3 2-D convolution.  This is roughly 1.5× faster and
    produces bit-identical output to the 2-D form.
    """
    if image.ndim != 2:
        raise ValueError(
            f"sobel expects a 2-D grayscale image; got shape {image.shape}."
        )

    # 1. Promote to float so the negative weights of the derivative
    #    kernel don't underflow uint8.
    img_f = image.astype(np.float32)

    # 2. X-gradient = smooth-vertically then differentiate-horizontally.
    #
    #    Why this order?  Both 1-D passes are linear and they commute,
    #    so either order is mathematically identical; we pick this one
    #    because the smoothing pass over columns is friendlier to row-
    #    major NumPy memory layout.
    smooth_y = _convolve_1d(img_f, _SMOOTH, axis=0)
    gx = _convolve_1d(smooth_y, _DERIV, axis=1)

    # 3. Y-gradient = smooth-horizontally then differentiate-vertically.
    smooth_x = _convolve_1d(img_f, _SMOOTH, axis=1)
    gy = _convolve_1d(smooth_x, _DERIV, axis=0)

    # 4. Magnitude.  We use the L2 norm rather than |gx| + |gy| so the
    #    response is rotation-invariant.
    magnitude = np.sqrt(gx * gx + gy * gy)

    # 5. Scale magnitude to uint8 for visualization / thresholding.
    #    Avoid divide-by-zero on a constant image.
    peak = magnitude.max()
    if peak > 0:
        scaled = (magnitude * (255.0 / peak))
    else:
        scaled = magnitude
    magnitude_uint8 = np.clip(scaled, 0, 255).astype(np.uint8)

    return SobelResult(
        gx=gx,
        gy=gy,
        magnitude=magnitude,
        magnitude_uint8=magnitude_uint8,
    )


def sobel_x(image: np.ndarray) -> np.ndarray:
    """
    Convenience: return only the horizontal-gradient magnitude as ``uint8``.

    For license-plate detection we only really care about ``|∂I/∂x|``
    because plates produce a dense forest of vertical strokes.  This
    helper is exactly that — same as ``sobel(image).gx`` followed by
    absolute value and uint8 scaling — but skips the Y-pass entirely
    so it is roughly twice as fast as the full :func:`sobel`.
    """
    if image.ndim != 2:
        raise ValueError(
            f"sobel_x expects a 2-D grayscale image; got shape {image.shape}."
        )

    img_f = image.astype(np.float32)
    smooth_y = _convolve_1d(img_f, _SMOOTH, axis=0)
    gx = _convolve_1d(smooth_y, _DERIV, axis=1)

    abs_gx = np.abs(gx)
    peak = abs_gx.max()
    if peak > 0:
        scaled = abs_gx * (255.0 / peak)
    else:
        scaled = abs_gx

    return np.clip(scaled, 0, 255).astype(np.uint8)
