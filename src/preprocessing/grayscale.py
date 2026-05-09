"""
grayscale.py
============

RGB → grayscale conversion.

Why grayscale first?
--------------------
Most low-level operators we will write next (Sobel, Otsu, morphology,
Hough, ...) operate on a single intensity channel.  Working on three
channels would triple the computation and rarely improves edge / shape
detection on license plates, where the relevant information is contrast
between dark characters and bright background.

Algorithm
---------
We use the **ITU-R BT.601 luminance formula**, which is the standard
weighting for converting sRGB to perceptual luminance:

    Y = 0.299 * R + 0.587 * G + 0.114 * B

The weights are chosen to match the human eye's sensitivity:
* Green contributes the most (~58.7%) — eyes are most sensitive to green.
* Red contributes ~29.9%.
* Blue contributes the least (~11.4%).

A naive average ``(R + G + B) / 3`` works too but produces a perceptually
flatter image where green objects look too dark and blue objects too
bright, which can hurt edge detection on color-coded plates.
"""

from __future__ import annotations

import numpy as np

# BT.601 luminance coefficients.  They sum to 1.0 by construction so the
# output range stays within [0, 255] for uint8 input.
_LUMINANCE_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale using the BT.601 luminance formula.

    Parameters
    ----------
    image : np.ndarray
        Either:
        * an RGB image of shape ``(H, W, 3)`` and dtype ``uint8``, or
        * an already-grayscale image of shape ``(H, W)`` (returned
          untouched, which makes the function safe to call twice).

    Returns
    -------
    np.ndarray
        Grayscale image of shape ``(H, W)`` and dtype ``uint8``.

    Raises
    ------
    ValueError
        If the input is not 2-D grayscale or 3-D RGB with 3 channels.
    """
    # ------------------------------------------------------------------
    # 1. Idempotency: if the caller hands us a grayscale image, do nothing.
    #    This lets higher-level pipelines stay simple — they can always
    #    call rgb_to_grayscale() without checking the shape themselves.
    # ------------------------------------------------------------------
    if image.ndim == 2:
        return image

    # ------------------------------------------------------------------
    # 2. Validate shape: we only accept three-channel RGB.
    # ------------------------------------------------------------------
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "rgb_to_grayscale expects a (H, W, 3) RGB image or (H, W) "
            f"grayscale image; got shape {image.shape}."
        )

    # ------------------------------------------------------------------
    # 3. Compute the weighted sum.
    #
    #    We promote to float32 because uint8 * 0.299 wraps around in
    #    integer arithmetic.  Using `np.dot` with a 1-D weight vector
    #    contracts the trailing axis, which is the channel axis here:
    #
    #        image[H, W, 3] · weights[3] -> result[H, W]
    #
    #    This is significantly faster than the equivalent expression
    #    `0.299 * R + 0.587 * G + 0.114 * B` because NumPy can dispatch
    #    it to a single BLAS call.
    # ------------------------------------------------------------------
    gray_float = np.dot(image[..., :3].astype(np.float32), _LUMINANCE_WEIGHTS)

    # ------------------------------------------------------------------
    # 4. Round to the nearest integer and clip to the valid uint8 range.
    #    Rounding (rather than truncation) keeps the output as close as
    #    possible to the true luminance.
    # ------------------------------------------------------------------
    gray_uint8 = np.clip(np.round(gray_float), 0, 255).astype(np.uint8)
    return gray_uint8
