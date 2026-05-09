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
from src.normalization import NormalizationResult, normalize_plate
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


def _plate_region_score(option: PlateRegionOption) -> tuple[int, float, int, float]:
    """
    Score candidates by segmentation usefulness, then detector confidence.

    Character count is the strongest evidence.  Counts between 5 and 8
    cover the bundled one-line and two-line examples.  Higher counts are
    possible in the wild, but in this classical detector they are often
    grille bars split into many false glyphs, so they are ranked below a
    clean 5-8 character candidate.
    """
    count = len(option.segmentation.characters)
    if 5 <= count <= 8:
        return (2, float(count), -int(option.touches_image_border), option.detection_score)
    if 9 <= count <= 10:
        return (1, float(10 - count), -int(option.touches_image_border), option.detection_score)
    return (0, float(count), -int(option.touches_image_border), option.detection_score)
