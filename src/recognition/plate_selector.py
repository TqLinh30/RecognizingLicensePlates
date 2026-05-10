"""
plate_selector.py
=================

Choose the best plate region by validating candidates against the
downstream character segmentation stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.detection import PlateCandidate
from src.detection.connected_components import connected_components
from src.normalization import NormalizationResult, normalize_plate
from src.preprocessing.thresholding import otsu_threshold
from src.segmentation import SegmentationConfig, SegmentationResult, segment_characters


@dataclass
class PlateRegionOption:
    """A normalized plate-region candidate plus its segmentation result."""

    source: str
    box: tuple[int, int, int, int]
    normalized: np.ndarray
    segmentation: SegmentationResult
    angle_degrees: float = 0.0
    cropped: np.ndarray | None = None
    edge_image: np.ndarray | None = None
    deskewed: np.ndarray | None = None
    detection_score: float = 0.0
    touches_image_border: bool = False


def select_plate_region(
    enhanced_gray: np.ndarray,
    candidates: Sequence[PlateCandidate],
    segmentation_config: SegmentationConfig | None = None,
    include_full_image: bool = True,
) -> PlateRegionOption | None:
    """
    Select a plate region by checking how well it segments into glyphs.

    The detector can rank grille bars above the real plate in difficult
    images, and plate-only images may produce one candidate per text row.
    Instead of trusting the first detector candidate blindly, every
    candidate is normalized and segmented, then scored by plausible
    character count and detection score.  A full-image option is included
    for already-cropped plate images.
    """
    if enhanced_gray.ndim != 2 or enhanced_gray.dtype != np.uint8:
        raise ValueError(
            "select_plate_region expects a 2-D uint8 enhanced image; got "
            f"shape {enhanced_gray.shape}, dtype {enhanced_gray.dtype}."
        )

    cfg = segmentation_config or SegmentationConfig()
    options: list[PlateRegionOption] = []

    for index, candidate in enumerate(candidates, 1):
        norm = normalize_plate(enhanced_gray, candidate)
        seg = segment_characters(norm.normalized, cfg)
        options.append(_option_from_normalization(index, candidate, norm, seg, enhanced_gray.shape))

    if include_full_image:
        seg = segment_characters(enhanced_gray, cfg)
        H, W = enhanced_gray.shape
        options.append(
            PlateRegionOption(
                source="full-image fallback",
                box=(0, 0, W, H),
                normalized=enhanced_gray,
                segmentation=seg,
                angle_degrees=0.0,
                cropped=enhanced_gray,
                edge_image=np.zeros_like(enhanced_gray),
                deskewed=enhanced_gray,
                detection_score=0.0,
                touches_image_border=False,
            )
        )

    if not any(_has_usable_character_count(option) for option in options):
        options.extend(_bright_region_options(enhanced_gray, cfg))

    if not options:
        return None
    return max(options, key=_plate_region_score)


def _option_from_normalization(
    index: int,
    candidate: PlateCandidate,
    norm: NormalizationResult,
    seg: SegmentationResult,
    image_shape: tuple[int, int],
) -> PlateRegionOption:
    """Convert a detector candidate normalization into a scored option."""
    H, W = image_shape
    x, y, w, h = norm.box
    margin = max(2, int(round(0.02 * min(H, W))))
    touches_border = x <= margin or y <= margin or x + w >= W - margin or y + h >= H - margin
    return PlateRegionOption(
        source=f"detector candidate #{index}",
        box=norm.box,
        normalized=norm.normalized,
        segmentation=seg,
        angle_degrees=norm.angle_degrees,
        cropped=norm.cropped,
        edge_image=norm.edge_image,
        deskewed=norm.deskewed,
        detection_score=float(candidate.score),
        touches_image_border=touches_border,
    )


def _plate_region_score(option: PlateRegionOption) -> tuple[float, float, float]:
    """
    Score candidates by segmentation usefulness, then detector confidence.

    A usable region needs a plausible character count, should not touch
    image borders, and should still respect the detector's geometric
    confidence.  The count score is deliberately not absolute: a strong
    five-character plate such as ``LX570`` must beat a weak false region
    that happens to split into eight grille-like blobs.
    """
    count = len(option.segmentation.characters)
    if 5 <= count <= 8:
        count_score = 0.50 + 0.10 * (count - 5)
    elif 9 <= count <= 10:
        count_score = 0.35
    else:
        count_score = min(0.30, 0.05 * count)

    border_penalty = 0.60 if option.touches_image_border else 0.0
    full_image_bonus = 0.04 if option.source == "full-image fallback" and 5 <= count <= 8 else 0.0
    total = count_score + option.detection_score + full_image_bonus - border_penalty
    return (total, count_score, option.detection_score)


def _has_usable_character_count(option: PlateRegionOption) -> bool:
    """Return whether a candidate already segments into a plausible text row."""
    count = len(option.segmentation.characters)
    return 4 <= count <= 10


def _bright_region_options(
    enhanced_gray: np.ndarray,
    cfg: SegmentationConfig,
) -> list[PlateRegionOption]:
    """
    Find bright plate-like rectangles as a detector fallback.

    Sobel-X detection works well for high-contrast Vietnamese-style
    plates, but it can miss a small bright plate in a large dark vehicle
    image when the surrounding bodywork dominates the gradient map.  This
    fallback thresholds bright connected components, keeps rectangular
    regions with plate-like aspect ratios, and still validates them by
    downstream character segmentation before they can be selected.
    """
    H, W = enhanced_gray.shape
    bright, _threshold = otsu_threshold(enhanced_gray, invert=False)
    cc = connected_components(bright, connectivity=8)
    options: list[PlateRegionOption] = []
    for comp in cc.stats:
        aspect = comp.width / max(1, comp.height)
        box_area_ratio = (comp.width * comp.height) / max(1, H * W)
        if not (1.5 <= aspect <= 6.5):
            continue
        if not (0.001 <= box_area_ratio <= 0.10):
            continue
        if comp.width < 80 or comp.height < 30:
            continue
        if comp.fill_ratio < 0.20:
            continue

        candidate = PlateCandidate(
            x=comp.x,
            y=comp.y,
            width=comp.width,
            height=comp.height,
            score=float(comp.fill_ratio) * 0.35,
            aspect_ratio=aspect,
            fill_ratio=comp.fill_ratio,
            gradient_density=0.0,
            component=comp,
        )
        norm = normalize_plate(enhanced_gray, candidate)
        seg = segment_characters(norm.normalized, cfg)
        if not seg.characters:
            continue
        options.append(
            PlateRegionOption(
                source=f"bright-region fallback #{len(options) + 1}",
                box=norm.box,
                normalized=norm.normalized,
                segmentation=seg,
                angle_degrees=norm.angle_degrees,
                cropped=norm.cropped,
                edge_image=norm.edge_image,
                deskewed=norm.deskewed,
                detection_score=float(candidate.score),
                touches_image_border=False,
            )
        )

    return options[:8]
