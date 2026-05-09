"""
clahe.py
========

Contrast Limited Adaptive Histogram Equalization (CLAHE).

Why CLAHE rather than plain histogram equalization?
---------------------------------------------------
A license-plate photo is rarely lit uniformly: there might be glare on
one side and shadow on the other.  Global histogram equalization
processes every pixel with the *same* lookup table, so it cannot
brighten a shadow without also blowing out the highlights.

CLAHE solves this in two steps:

1. **Adaptive** — the image is divided into a grid of small tiles and
   each tile gets its own equalization based on its local histogram.
   This lets dark tiles boost their contrast independently of bright
   tiles.

2. **Contrast limited** — when a tile's histogram has tall spikes
   (e.g. lots of pixels at the same value), the corresponding CDF has
   a steep slope and equalization would massively amplify any noise
   sitting on those values.  We *clip* histogram bins to a maximum
   count and redistribute the excess uniformly to the other bins.
   The clip limit acts as a contrast cap.

3. **Bilinear interpolation between tile maps** — if we naively applied
   each tile's lookup table only to its own pixels, we would see
   blocky, visible tile boundaries in the output.  Instead we
   interpolate between the four nearest tile maps for every pixel,
   producing a smooth result.

Reference: Karel Zuiderveld, *Contrast Limited Adaptive Histogram
Equalization*, Graphics Gems IV, 1994.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Helper: clip a histogram and redistribute the excess
# ---------------------------------------------------------------------------

def _clip_histogram(hist: np.ndarray, clip_limit: int) -> np.ndarray:
    """
    Clip histogram bins to ``clip_limit`` and redistribute the excess
    *uniformly* across all bins.

    The redistribution may itself push some bins above the clip limit
    again (because we add a constant to every bin).  In a fully faithful
    CLAHE one would iterate until convergence; in practice a single
    pass is more than enough for visual quality and we keep the
    implementation simple.

    Parameters
    ----------
    hist : np.ndarray
        Histogram array of shape ``(256,)``.
    clip_limit : int
        Maximum allowed count per bin.

    Returns
    -------
    np.ndarray
        Clipped histogram, same shape and dtype.
    """
    # Total number of "excess" counts above the limit.
    excess = np.maximum(hist - clip_limit, 0).sum()

    # Cap each bin.
    clipped = np.minimum(hist, clip_limit)

    # Spread the excess uniformly across all 256 bins.
    redistribution = excess // hist.size  # integer share per bin
    clipped += redistribution

    # Whatever doesn't divide evenly (the remainder) is simply added
    # to a few low bins.  This is a small pragmatic shortcut over the
    # iterative version.
    leftover = excess - redistribution * hist.size
    if leftover > 0:
        clipped[: int(leftover)] += 1

    return clipped


# ---------------------------------------------------------------------------
# Helper: build the lookup table for one tile
# ---------------------------------------------------------------------------

def _tile_lut(tile: np.ndarray, clip_limit: int) -> np.ndarray:
    """
    Compute the equalization lookup table (256 entries) for a single tile.

    Steps:
    1. histogram of the tile
    2. clip + redistribute
    3. CDF
    4. scale CDF to 0..255

    Returns
    -------
    np.ndarray
        ``uint8`` array of shape ``(256,)``.
    """
    # Histogram of the tile.  bincount + minlength=256 guarantees the
    # full 8-bit range is represented even if some intensities are
    # absent in this tile.
    hist = np.bincount(tile.ravel(), minlength=256)

    # Apply contrast limiting.
    hist = _clip_histogram(hist, clip_limit)

    # Cumulative distribution.
    cdf = np.cumsum(hist)

    # Scale to 0..255.  ``cdf[-1]`` is the total pixel count after
    # clipping (still equal to tile_size because redistribution
    # preserves the total).
    total = cdf[-1]
    if total <= 0:
        # Empty / impossible: fall back to identity LUT.
        return np.arange(256, dtype=np.uint8)

    lut = np.round(cdf * 255.0 / total)
    return np.clip(lut, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clahe(
    image: np.ndarray,
    tile_grid_size: tuple[int, int] = (8, 8),
    clip_limit: float = 2.0,
) -> np.ndarray:
    """
    Apply CLAHE to a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, dtype ``uint8``, shape ``(H, W)``.
    tile_grid_size : (int, int), default (8, 8)
        Number of tiles along (rows, cols).  Smaller grids → more
        global-like behaviour; larger grids → more locally adaptive.
    clip_limit : float, default 2.0
        Multiplier on the average bin height that defines the clip cap.
        ``clip_limit = 2.0`` means bins are capped at twice the average
        bin count, which is a sensible default.  Values of 1.0 disable
        contrast enhancement; values much larger than 4.0 typically
        amplify noise.

    Returns
    -------
    np.ndarray
        Enhanced image, same shape and dtype as the input.
    """
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(
            "clahe expects a 2-D uint8 image; got shape "
            f"{image.shape}, dtype {image.dtype}."
        )

    H, W = image.shape
    n_tiles_y, n_tiles_x = tile_grid_size

    # ------------------------------------------------------------------
    # 1. Pad the image so its dimensions are exact multiples of the tile
    #    size.  This avoids special-casing partial border tiles later.
    # ------------------------------------------------------------------
    tile_h = int(np.ceil(H / n_tiles_y))
    tile_w = int(np.ceil(W / n_tiles_x))

    pad_h = tile_h * n_tiles_y - H
    pad_w = tile_w * n_tiles_x - W
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")

    # ------------------------------------------------------------------
    # 2. Compute the absolute clip count.
    #
    #    A "uniform" tile would have ``tile_size / 256`` counts in every
    #    bin.  Multiplying by ``clip_limit`` gives the cap.  We enforce
    #    a minimum of 1 so that we never effectively disable redistribution.
    # ------------------------------------------------------------------
    tile_size = tile_h * tile_w
    abs_clip = max(1, int(clip_limit * tile_size / 256))

    # ------------------------------------------------------------------
    # 3. Compute one 256-entry LUT per tile, stored in a 4-D array
    #    of shape (n_tiles_y, n_tiles_x, 256) for fast indexing later.
    # ------------------------------------------------------------------
    luts = np.empty((n_tiles_y, n_tiles_x, 256), dtype=np.uint8)
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            tile = padded[
                ty * tile_h : (ty + 1) * tile_h,
                tx * tile_w : (tx + 1) * tile_w,
            ]
            luts[ty, tx] = _tile_lut(tile, abs_clip)

    # ------------------------------------------------------------------
    # 4. Bilinear interpolation between the four nearest tile LUTs.
    #
    #    Strategy: for every pixel (y, x) in the *padded* image we find
    #    the four tile centres surrounding it and combine their LUTs by
    #    distance.  Tile centres lie at:
    #
    #        (cy_t, cx_t) = ( (t + 0.5) * tile_h , (t + 0.5) * tile_w )
    #
    #    Pixels near a corner of the image have fewer than four
    #    neighbours; we handle that by clamping the tile indices.
    # ------------------------------------------------------------------
    Hp, Wp = padded.shape

    # Build coordinate grids.  np.indices produces (2, Hp, Wp) so we
    # split into y_grid and x_grid.
    y_grid, x_grid = np.indices((Hp, Wp), dtype=np.float32)

    # Map each pixel to its position relative to the tile-centre lattice.
    # `ty_f` is in [-0.5, n_tiles_y - 0.5]: -0.5 means we're left of the
    # first tile centre, n_tiles_y - 0.5 means we're past the last one.
    ty_f = y_grid / tile_h - 0.5
    tx_f = x_grid / tile_w - 0.5

    # Integer indices of the upper-left tile of the 2x2 neighbourhood.
    ty0 = np.floor(ty_f).astype(np.int32)
    tx0 = np.floor(tx_f).astype(np.int32)

    # Bilinear weights (distance from upper-left tile centre).
    wy = ty_f - ty0
    wx = tx_f - tx0

    # Clamp tile indices into the valid range.  Border pixels then use
    # the same tile twice on the truncated side, which gracefully
    # degrades bilinear interpolation to linear at edges and to nearest
    # at corners — exactly what CLAHE specifies.
    ty0c = np.clip(ty0, 0, n_tiles_y - 1)
    ty1c = np.clip(ty0 + 1, 0, n_tiles_y - 1)
    tx0c = np.clip(tx0, 0, n_tiles_x - 1)
    tx1c = np.clip(tx0 + 1, 0, n_tiles_x - 1)

    # Look up the LUT value for each of the four corner tiles.
    # Fancy-indexing: luts[ty, tx, intensity] -> selected output value.
    intensity = padded
    v00 = luts[ty0c, tx0c, intensity].astype(np.float32)  # top-left
    v01 = luts[ty0c, tx1c, intensity].astype(np.float32)  # top-right
    v10 = luts[ty1c, tx0c, intensity].astype(np.float32)  # bottom-left
    v11 = luts[ty1c, tx1c, intensity].astype(np.float32)  # bottom-right

    # Bilinear blend.
    # First combine along x (top row, then bottom row), then along y.
    top = v00 * (1.0 - wx) + v01 * wx
    bot = v10 * (1.0 - wx) + v11 * wx
    out = top * (1.0 - wy) + bot * wy

    # ------------------------------------------------------------------
    # 5. Crop back to the original image size and return as uint8.
    # ------------------------------------------------------------------
    out = np.round(out[:H, :W])
    return np.clip(out, 0, 255).astype(np.uint8)
