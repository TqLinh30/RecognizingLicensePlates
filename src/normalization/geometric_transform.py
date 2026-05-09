"""
geometric_transform.py
======================

Basic geometric image operations implemented with NumPy only:

* crop a candidate box with safe clipping and margin,
* sample floating-point coordinates with bilinear interpolation,
* rotate around the image centre using inverse mapping,
* resize to a fixed shape.

Inverse mapping is important.  Instead of pushing each source pixel to
an output location (which leaves holes), every output pixel asks "where
did I come from in the source image?" and samples that point.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def crop_with_padding(
    image: np.ndarray,
    box: Sequence[int],
    margin_ratio: float = 0.08,
    fill_value: int = 255,
) -> np.ndarray:
    """
    Crop ``(x, y, width, height)`` with a proportional margin.

    Regions that fall outside the image are filled with ``fill_value``.
    This is useful when a detector box touches the frame border but the
    normalizer still wants a little context around the plate.
    """
    if image.ndim not in (2, 3):
        raise ValueError(
            f"crop_with_padding expects a 2-D or 3-D image; got shape {image.shape}."
        )
    if margin_ratio < 0:
        raise ValueError(f"margin_ratio must be non-negative; got {margin_ratio}.")

    x, y, w, h = [int(v) for v in box]
    if w <= 0 or h <= 0:
        raise ValueError(f"box width and height must be positive; got {box}.")

    margin_x = int(round(w * margin_ratio))
    margin_y = int(round(h * margin_ratio))
    out_w = w + 2 * margin_x
    out_h = h + 2 * margin_y

    if image.ndim == 2:
        out = np.full((out_h, out_w), fill_value, dtype=image.dtype)
    else:
        out = np.full((out_h, out_w, image.shape[2]), fill_value, dtype=image.dtype)

    src_x0 = x - margin_x
    src_y0 = y - margin_y
    src_x1 = src_x0 + out_w
    src_y1 = src_y0 + out_h

    H, W = image.shape[:2]
    ix0 = max(0, src_x0)
    iy0 = max(0, src_y0)
    ix1 = min(W, src_x1)
    iy1 = min(H, src_y1)
    if ix0 >= ix1 or iy0 >= iy1:
        return out

    ox0 = ix0 - src_x0
    oy0 = iy0 - src_y0
    out[oy0 : oy0 + (iy1 - iy0), ox0 : ox0 + (ix1 - ix0)] = image[iy0:iy1, ix0:ix1]
    return out


def bilinear_sample(
    image: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    fill_value: int | float = 255,
) -> np.ndarray:
    """
    Sample ``image`` at floating-point coordinates using bilinear interpolation.

    ``xs`` and ``ys`` must have the same shape.  Coordinates outside the
    valid image rectangle receive ``fill_value``.
    """
    if image.ndim not in (2, 3):
        raise ValueError(
            f"bilinear_sample expects a 2-D or 3-D image; got shape {image.shape}."
        )
    if xs.shape != ys.shape:
        raise ValueError("xs and ys must have the same shape.")

    H, W = image.shape[:2]
    valid = (xs >= 0.0) & (ys >= 0.0) & (xs <= W - 1) & (ys <= H - 1)

    x0 = np.floor(np.clip(xs, 0, W - 1)).astype(np.int32)
    y0 = np.floor(np.clip(ys, 0, H - 1)).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)

    wx = xs - x0
    wy = ys - y0
    wx = np.clip(wx, 0.0, 1.0)
    wy = np.clip(wy, 0.0, 1.0)

    if image.ndim == 2:
        Ia = image[y0, x0].astype(np.float32)
        Ib = image[y0, x1].astype(np.float32)
        Ic = image[y1, x0].astype(np.float32)
        Id = image[y1, x1].astype(np.float32)
        out = (
            Ia * (1 - wx) * (1 - wy)
            + Ib * wx * (1 - wy)
            + Ic * (1 - wx) * wy
            + Id * wx * wy
        )
        out[~valid] = fill_value
    else:
        wx3 = wx[..., None]
        wy3 = wy[..., None]
        Ia = image[y0, x0].astype(np.float32)
        Ib = image[y0, x1].astype(np.float32)
        Ic = image[y1, x0].astype(np.float32)
        Id = image[y1, x1].astype(np.float32)
        out = (
            Ia * (1 - wx3) * (1 - wy3)
            + Ib * wx3 * (1 - wy3)
            + Ic * (1 - wx3) * wy3
            + Id * wx3 * wy3
        )
        out[~valid] = fill_value

    return np.clip(np.rint(out), 0, 255).astype(image.dtype)


def rotate_image(
    image: np.ndarray,
    angle_degrees: float,
    fill_value: int = 255,
) -> np.ndarray:
    """
    Rotate an image around its centre while preserving the original shape.

    Positive angles rotate the visible content counter-clockwise in the
    usual image display coordinate system.  The output canvas size is
    unchanged because the plate crop has already been padded.
    """
    if image.ndim not in (2, 3):
        raise ValueError(
            f"rotate_image expects a 2-D or 3-D image; got shape {image.shape}."
        )
    if abs(angle_degrees) < 1e-8:
        return image.copy()

    H, W = image.shape[:2]
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    yy, xx = np.indices((H, W), dtype=np.float32)
    dx = xx - cx
    dy = yy - cy

    angle = np.deg2rad(angle_degrees)
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))

    # Inverse mapping: output coordinate -> source coordinate.
    src_x = cx + cos_a * dx + sin_a * dy
    src_y = cy - sin_a * dx + cos_a * dy
    return bilinear_sample(image, src_x, src_y, fill_value=fill_value)


def resize_bilinear(
    image: np.ndarray,
    target_shape: tuple[int, int],
    fill_value: int = 255,
) -> np.ndarray:
    """
    Resize ``image`` to ``(target_height, target_width)``.

    Pixel centres are aligned: the top-left output centre maps to the
    top-left source centre, and the bottom-right maps to bottom-right.
    This avoids systematic half-pixel shifts when repeatedly resizing.
    """
    if image.ndim not in (2, 3):
        raise ValueError(
            f"resize_bilinear expects a 2-D or 3-D image; got shape {image.shape}."
        )
    out_h, out_w = target_shape
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"target_shape must be positive; got {target_shape}.")

    H, W = image.shape[:2]
    if H == out_h and W == out_w:
        return image.copy()

    if out_w == 1:
        xs = np.zeros(out_w, dtype=np.float32)
    else:
        xs = np.linspace(0, W - 1, out_w, dtype=np.float32)
    if out_h == 1:
        ys = np.zeros(out_h, dtype=np.float32)
    else:
        ys = np.linspace(0, H - 1, out_h, dtype=np.float32)

    grid_x, grid_y = np.meshgrid(xs, ys)
    return bilinear_sample(image, grid_x, grid_y, fill_value=fill_value)
