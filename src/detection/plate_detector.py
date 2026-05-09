"""
plate_detector.py
=================

License-plate detector built from the Step-2 building blocks:

    Sobel-X gradient  →  threshold  →  morphological closing
                      →  connected components  →  geometric filter
                      →  ranked candidate boxes

Why the gradient direction matters
----------------------------------
A license plate is dense with **vertical strokes** (the left/right
edges of every character).  The horizontal Sobel response, ``|∂I/∂x|``,
fires brightly inside the plate and stays calm on car body, sky, and
road, giving us a strong, location-specific cue.

Why a horizontal closing
------------------------
The thresholded gradient image looks like a forest of disconnected
vertical bars.  A morphological closing with a long, narrow,
*horizontal* SE bridges the gaps **between** characters but not
**between** plate and background, fusing the per-character bars into
a single solid blob shaped roughly like the plate itself.

Filtering candidate components
------------------------------
After connected-component labeling we get many blobs — some plates,
many false positives (text on bumpers, road signs, shadows...).  We
score each component against geometric priors derived from the typical
shape of a license plate:

* aspect ratio (width / height),
* absolute and image-relative size,
* fill ratio (area / bounding-box area),
* gradient-density inside the box.

The configuration knobs below are tuned for **Vietnamese-style plates**
in moderately good photographs.  All of them can be loosened or
tightened from the caller side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.detection.sobel import sobel_x
from src.detection.morphology import closing, rect
from src.detection.connected_components import (
    connected_components,
    ComponentStats,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DetectionConfig:
    """
    Tunable parameters for the plate detector.

    Defaults are set for **single-line Vietnamese plates** photographed
    at a roughly frontal angle and resolution where the plate fills
    between 1 % and 30 % of the image area.

    Aspect ratio
    ------------
    Vietnamese single-line plates have aspect ratios around 4:1 — 5:1.
    Two-line "square-ish" plates are about 1.4:1 — 1.8:1.  We allow
    a permissive range that covers both, plus tilt up to a few degrees.

    Geometric thresholds
    --------------------
    * ``min_area_ratio`` / ``max_area_ratio`` — the candidate's bounding
      box area divided by the full image area.  Plates almost never
      fill more than 30 % of the frame and almost never less than 0.5 %.
    * ``min_fill_ratio`` — solid blobs are preferred; a thin frame-like
      contour will have a low fill ratio.
    * ``min_gradient_density`` — fraction of the bounding box that was
      foreground in the thresholded gradient image.  Plates score
      high; smooth body panels score near zero.
    """
    # Sobel + threshold
    sobel_threshold_ratio: float = 0.20  # fraction of max |∂I/∂x| above which we keep
    # Morphology
    closing_kernel: tuple[int, int] = (3, 25)  # (height, width) — long horizontal SE
    # Aspect ratio (width / height)
    min_aspect_ratio: float = 1.5
    max_aspect_ratio: float = 6.5
    # Area as fraction of total image
    min_area_ratio: float = 0.005
    max_area_ratio: float = 0.30
    # Bounding box geometry
    min_fill_ratio: float = 0.30
    min_height_px: int = 15      # avoids picking up specks
    min_width_px: int = 40
    # Gradient-density inside the box
    min_gradient_density: float = 0.10
    # How many top-scoring candidates to return
    max_candidates: int = 5


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class PlateCandidate:
    """One rectangle that may contain a license plate, plus its score."""
    x: int
    y: int
    width: int
    height: int
    score: float                   # higher = better (composite metric)
    aspect_ratio: float
    fill_ratio: float
    gradient_density: float
    component: ComponentStats

    def as_box(self) -> tuple[int, int, int, int]:
        """Return ``(x, y, width, height)`` — convenient for cropping."""
        return (self.x, self.y, self.width, self.height)


@dataclass
class DetectionResult:
    """
    Full output of :func:`detect_plate`.

    Intermediate stages are exposed so callers can debug visually and
    so Step 3 (cropping/normalisation) can directly reuse the gradient
    map without recomputing it.
    """
    gradient: np.ndarray            # uint8 |∂I/∂x| scaled to 0..255
    binary: np.ndarray              # uint8 thresholded gradient {0, 255}
    closed: np.ndarray              # uint8 after morphological closing
    labels: np.ndarray              # int32 connected-component labels
    candidates: list[PlateCandidate]  # sorted, best first
    config: DetectionConfig = field(default_factory=DetectionConfig)


# ---------------------------------------------------------------------------
# Internal scoring
# ---------------------------------------------------------------------------

def _score_candidate(
    component: ComponentStats,
    gradient_density: float,
    cfg: DetectionConfig,
) -> float:
    """
    Composite score for ranking plate candidates.

    The score combines three factors that should each be near 1.0 for
    a good plate:

    * **Aspect-ratio fit** — Gaussian penalty around the most plausible
      aspect ratio (3.5).  Plates close to square (two-line) still get
      a respectable score.
    * **Fill ratio** — solid blobs preferred.
    * **Gradient density** — high inside real plates.

    All three are mapped into ``[0, 1]`` and multiplied so any single
    failure heavily penalises the candidate.  Multiplication (rather
    than averaging) is intentional: a plate-shaped blob with no
    gradient inside is almost certainly not a plate.
    """
    ar = component.aspect_ratio

    # Aspect ratio: Gaussian centred at 3.5 with σ = 1.5.  Width-to-
    # height ratios from ~1.5 to ~6.5 score above 0.13.
    ar_target = 3.5
    ar_sigma = 1.5
    ar_score = float(np.exp(-((ar - ar_target) ** 2) / (2 * ar_sigma ** 2)))

    # Fill ratio: linear ramp from min_fill_ratio (0) to 1.0 (1).
    fill = max(0.0, min(1.0, (component.fill_ratio - cfg.min_fill_ratio) /
                       max(1e-6, 1.0 - cfg.min_fill_ratio)))

    # Gradient density: linear ramp from min_gradient_density (0) to
    # 0.5 (1) — densities above 50 % are rare and saturate the score.
    grad = max(0.0, min(1.0, (gradient_density - cfg.min_gradient_density) /
                       max(1e-6, 0.5 - cfg.min_gradient_density)))

    return ar_score * fill * grad


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_plate(
    enhanced_gray: np.ndarray,
    config: Optional[DetectionConfig] = None,
) -> DetectionResult:
    """
    Detect license-plate candidates in a preprocessed grayscale image.

    Parameters
    ----------
    enhanced_gray : np.ndarray
        Output of Step 1's CLAHE-enhanced image (``PreprocessResult.enhanced``)
        or any contrast-stretched grayscale.  Must be 2-D ``uint8``.
    config : DetectionConfig, optional
        Tuning parameters.

    Returns
    -------
    DetectionResult
        Intermediate maps and a list of :class:`PlateCandidate` objects
        sorted by score (best first).  Even if no candidate passes the
        filter, the full intermediate stack is returned for debugging.

    Pipeline
    --------
    1. Compute the horizontal Sobel response ``|∂I/∂x|``.
    2. Threshold it at ``sobel_threshold_ratio × max`` to obtain a
       binary "active edges" map.
    3. Apply a morphological closing with a long horizontal SE so the
       per-character bars merge into a plate-shaped blob.
    4. Run connected-component labelling.
    5. Filter components by geometric priors (aspect ratio, area,
       fill ratio).
    6. Score the survivors by combining geometry with the gradient
       density inside their bounding box.  Sort and return.
    """
    if enhanced_gray.ndim != 2 or enhanced_gray.dtype != np.uint8:
        raise ValueError(
            "detect_plate expects a 2-D uint8 grayscale image; got shape "
            f"{enhanced_gray.shape}, dtype {enhanced_gray.dtype}."
        )

    cfg = config or DetectionConfig()
    H, W = enhanced_gray.shape
    image_area = H * W

    # --- 1. Horizontal Sobel gradient ---------------------------------
    gradient = sobel_x(enhanced_gray)

    # --- 2. Threshold to a binary "active edges" map ------------------
    # We use a *relative* threshold (a fraction of the maximum) rather
    # than a fixed value.  This makes the detector invariant to global
    # contrast: dark photos still produce a reasonable binary map.
    peak = int(gradient.max()) or 1
    threshold = int(cfg.sobel_threshold_ratio * peak)
    binary = ((gradient > threshold).astype(np.uint8)) * 255

    # --- 3. Morphological closing -------------------------------------
    # The SE is much wider than tall (default 17 × 3) so we close gaps
    # *between characters* without joining the plate to surrounding
    # high-gradient regions like grilles or dashboard text.
    se = rect(cfg.closing_kernel[0], cfg.closing_kernel[1])
    closed = closing(binary, se)

    # --- 4. Connected components --------------------------------------
    cc = connected_components(closed, connectivity=8)

    # --- 5. Geometric filter + score ---------------------------------
    candidates: list[PlateCandidate] = []
    # Pre-compute a binary "edge present" map for gradient-density
    # checks.  We reuse the thresholded `binary` from step 2 because
    # that is exactly the per-pixel "is there an edge here" signal.
    edge_mask = binary > 0

    for comp in cc.stats:
        # Quick geometric rejects.  These are cheap, so we apply them
        # first to avoid the more expensive density check on obvious
        # non-plates.
        if comp.height < cfg.min_height_px or comp.width < cfg.min_width_px:
            continue
        if comp.aspect_ratio < cfg.min_aspect_ratio or \
           comp.aspect_ratio > cfg.max_aspect_ratio:
            continue
        area_frac = (comp.width * comp.height) / image_area
        if area_frac < cfg.min_area_ratio or area_frac > cfg.max_area_ratio:
            continue
        if comp.fill_ratio < cfg.min_fill_ratio:
            continue

        # Gradient density inside this bounding box.
        roi = edge_mask[comp.y : comp.y + comp.height,
                        comp.x : comp.x + comp.width]
        density = float(roi.mean())
        if density < cfg.min_gradient_density:
            continue

        # Score and collect.
        score = _score_candidate(comp, density, cfg)
        candidates.append(
            PlateCandidate(
                x=comp.x,
                y=comp.y,
                width=comp.width,
                height=comp.height,
                score=score,
                aspect_ratio=comp.aspect_ratio,
                fill_ratio=comp.fill_ratio,
                gradient_density=density,
                component=comp,
            )
        )

    # Sort by descending score and trim.
    candidates.sort(key=lambda c: c.score, reverse=True)
    candidates = candidates[: cfg.max_candidates]

    return DetectionResult(
        gradient=gradient,
        binary=binary,
        closed=closed,
        labels=cc.labels,
        candidates=candidates,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Convenience: draw boxes on the original image for visual debugging
# ---------------------------------------------------------------------------

def draw_candidates(
    image: np.ndarray,
    candidates: list[PlateCandidate],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Render bounding boxes for the candidates on a copy of ``image``.

    Parameters
    ----------
    image : np.ndarray
        Either grayscale ``(H, W)`` or RGB ``(H, W, 3)`` ``uint8``.
        A grayscale input is promoted to RGB so the box color is
        meaningful.
    candidates : list[PlateCandidate]
        Boxes to draw.  Drawn in the order given so the first overlays
        the rest — which is fine because we want the top candidate on
        top.
    color : (R, G, B)
        Color of the rectangle outlines.
    thickness : int
        Outline width in pixels.

    Returns
    -------
    np.ndarray
        Annotated RGB image, dtype ``uint8``.
    """
    # Promote to RGB and copy so the caller's array is untouched.
    if image.ndim == 2:
        out = np.stack([image, image, image], axis=-1).copy()
    else:
        out = image.copy()

    H, W, _ = out.shape
    color_arr = np.array(color, dtype=np.uint8)

    for cand in candidates:
        x, y, w, h = cand.as_box()
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)

        # Top and bottom edges.
        out[y0 : min(H, y0 + thickness), x0:x1] = color_arr
        out[max(0, y1 - thickness) : y1, x0:x1] = color_arr
        # Left and right edges.
        out[y0:y1, x0 : min(W, x0 + thickness)] = color_arr
        out[y0:y1, max(0, x1 - thickness) : x1] = color_arr

    return out
