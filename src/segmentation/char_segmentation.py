"""
char_segmentation.py
====================

Step 4 of the ALPR pipeline: split a normalized plate crop into
individual character images.

The input is the canonical plate image from Step 3 (default ``80x240``).
The output is an ordered list of connected components that look like
characters, each normalized to a small fixed canvas (default ``32x32``).

Why connected components?
-------------------------
Projection profiles are very fast but assume clean, vertical gaps
between characters.  Connected components are more tolerant of small
tilts, uneven spacing, and isolated noise.  We still keep the filters
simple and interpretable: height, width, area, aspect ratio, and border
touching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.detection.connected_components import ComponentStats, connected_components
from src.detection.morphology import opening, rect
from src.normalization.geometric_transform import resize_bilinear
from src.preprocessing.thresholding import otsu_threshold


@dataclass
class SegmentationConfig:
    """Tunable parameters for character segmentation."""

    char_shape: tuple[int, int] = (32, 32)  # (height, width)
    padding_ratio: float = 0.18
    connectivity: int = 8
    cleanup_opening_kernel: tuple[int, int] | None = (2, 2)
    min_height_ratio: float = 0.32
    max_height_ratio: float = 0.95
    min_width_ratio: float = 0.015
    max_width_ratio: float = 0.28
    min_area_ratio: float = 0.001
    max_area_ratio: float = 0.18
    min_fill_ratio: float = 0.08
    max_fill_ratio: float = 1.00
    reject_border_touching: bool = True
    border_margin_ratio: float = 0.015
    two_line_gap_ratio: float = 0.22


@dataclass
class CharacterCandidate:
    """One segmented character component."""

    x: int
    y: int
    width: int
    height: int
    image: np.ndarray
    normalized: np.ndarray
    row_index: int
    component: ComponentStats

    @property
    def center_x(self) -> float:
        return self.x + (self.width - 1) / 2.0

    @property
    def center_y(self) -> float:
        return self.y + (self.height - 1) / 2.0

    def as_box(self) -> tuple[int, int, int, int]:
        """Return ``(x, y, width, height)`` for visualization or debugging."""
        return (self.x, self.y, self.width, self.height)


@dataclass
class SegmentationResult:
    """All intermediate outputs produced by :func:`segment_characters`."""

    binary: np.ndarray
    cleaned: np.ndarray
    labels: np.ndarray
    characters: list[CharacterCandidate]
    config: SegmentationConfig = field(default_factory=SegmentationConfig)


def segment_characters(
    plate_gray: np.ndarray,
    config: Optional[SegmentationConfig] = None,
) -> SegmentationResult:
    """
    Segment a normalized grayscale plate into ordered character crops.

    Parameters
    ----------
    plate_gray : np.ndarray
        2-D ``uint8`` normalized plate image.  Characters are expected
        to be darker than the background, which is the usual case for
        white or yellow plates.
    config : SegmentationConfig, optional
        Filtering and normalization parameters.

    Returns
    -------
    SegmentationResult
        Binary image, cleaned binary image, label map, and ordered
        character candidates.
    """
    if plate_gray.ndim != 2 or plate_gray.dtype != np.uint8:
        raise ValueError(
            "segment_characters expects a 2-D uint8 plate image; got shape "
            f"{plate_gray.shape}, dtype {plate_gray.dtype}."
        )

    cfg = config or SegmentationConfig()
    H, W = plate_gray.shape
    plate_area = H * W

    # Dark glyphs become foreground (255).  This convention matches the
    # morphology and feature extraction modules.
    binary, _ = otsu_threshold(plate_gray, invert=True)

    if cfg.cleanup_opening_kernel is not None:
        kh, kw = cfg.cleanup_opening_kernel
        cleaned = opening(binary, rect(kh, kw))
    else:
        cleaned = binary.copy()

    cc = connected_components(cleaned, connectivity=cfg.connectivity)

    raw_chars: list[CharacterCandidate] = []
    for comp in cc.stats:
        if not _looks_like_character(comp, image_shape=(H, W), plate_area=plate_area, cfg=cfg):
            continue
        char_img = _crop_component(cleaned, comp, cfg.padding_ratio)
        normalized = normalize_character(char_img, target_shape=cfg.char_shape)
        raw_chars.append(
            CharacterCandidate(
                x=comp.x,
                y=comp.y,
                width=comp.width,
                height=comp.height,
                image=char_img,
                normalized=normalized,
                row_index=0,
                component=comp,
            )
        )

    ordered = _assign_rows_and_sort(raw_chars, plate_height=H, cfg=cfg)

    return SegmentationResult(
        binary=binary,
        cleaned=cleaned,
        labels=cc.labels,
        characters=ordered,
        config=cfg,
    )


def normalize_character(
    char_binary: np.ndarray,
    target_shape: tuple[int, int] = (32, 32),
) -> np.ndarray:
    """
    Fit a binary character crop onto a fixed black canvas.

    The character aspect ratio is preserved.  The foreground remains
    white (255) and the background black (0), making the output suitable
    for raw-pixel, zoning, and HOG features.
    """
    if char_binary.ndim != 2:
        raise ValueError(
            f"normalize_character expects a 2-D image; got shape {char_binary.shape}."
        )
    out_h, out_w = target_shape
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"target_shape must be positive; got {target_shape}.")

    fg = char_binary > 0
    if not np.any(fg):
        return np.zeros(target_shape, dtype=np.uint8)

    ys, xs = np.nonzero(fg)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    tight = char_binary[y0:y1, x0:x1]

    scale = min(out_h / tight.shape[0], out_w / tight.shape[1])
    resized_h = max(1, int(round(tight.shape[0] * scale)))
    resized_w = max(1, int(round(tight.shape[1] * scale)))
    resized = resize_bilinear(tight, (resized_h, resized_w), fill_value=0)
    resized = ((resized > 127).astype(np.uint8)) * 255

    canvas = np.zeros(target_shape, dtype=np.uint8)
    y = (out_h - resized_h) // 2
    x = (out_w - resized_w) // 2
    canvas[y : y + resized_h, x : x + resized_w] = resized
    return canvas


def draw_character_boxes(
    plate_gray: np.ndarray,
    characters: list[CharacterCandidate],
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Draw segmented character boxes on a copy of the plate image."""
    if plate_gray.ndim == 2:
        out = np.stack([plate_gray, plate_gray, plate_gray], axis=-1).copy()
    else:
        out = plate_gray.copy()

    H, W = out.shape[:2]
    color_arr = np.array(color, dtype=np.uint8)
    for char in characters:
        x0, y0 = max(0, char.x), max(0, char.y)
        x1, y1 = min(W, char.x + char.width), min(H, char.y + char.height)
        out[y0 : min(H, y0 + thickness), x0:x1] = color_arr
        out[max(0, y1 - thickness) : y1, x0:x1] = color_arr
        out[y0:y1, x0 : min(W, x0 + thickness)] = color_arr
        out[y0:y1, max(0, x1 - thickness) : x1] = color_arr
    return out


def _looks_like_character(
    comp: ComponentStats,
    image_shape: tuple[int, int],
    plate_area: int,
    cfg: SegmentationConfig,
) -> bool:
    """Apply simple geometric character filters."""
    H, W = image_shape
    height_ratio = comp.height / max(1, H)
    width_ratio = comp.width / max(1, W)
    area_ratio = comp.area / max(1, plate_area)

    if height_ratio < cfg.min_height_ratio or height_ratio > cfg.max_height_ratio:
        return False
    if width_ratio < cfg.min_width_ratio or width_ratio > cfg.max_width_ratio:
        return False
    if area_ratio < cfg.min_area_ratio or area_ratio > cfg.max_area_ratio:
        return False
    if comp.fill_ratio < cfg.min_fill_ratio or comp.fill_ratio > cfg.max_fill_ratio:
        return False

    if cfg.reject_border_touching:
        margin_x = max(1, int(round(W * cfg.border_margin_ratio)))
        margin_y = max(1, int(round(H * cfg.border_margin_ratio)))
        touches_border = (
            comp.x <= margin_x
            or comp.y <= margin_y
            or comp.x + comp.width >= W - margin_x
            or comp.y + comp.height >= H - margin_y
        )
        if touches_border:
            return False

    return True


def _crop_component(
    binary: np.ndarray,
    comp: ComponentStats,
    padding_ratio: float,
) -> np.ndarray:
    """Crop a component with padding, clipping safely to the image."""
    if padding_ratio < 0:
        raise ValueError(f"padding_ratio must be non-negative; got {padding_ratio}.")
    H, W = binary.shape
    pad_x = int(round(comp.width * padding_ratio))
    pad_y = int(round(comp.height * padding_ratio))
    x0 = max(0, comp.x - pad_x)
    y0 = max(0, comp.y - pad_y)
    x1 = min(W, comp.x + comp.width + pad_x)
    y1 = min(H, comp.y + comp.height + pad_y)
    return binary[y0:y1, x0:x1].copy()


def _assign_rows_and_sort(
    characters: list[CharacterCandidate],
    plate_height: int,
    cfg: SegmentationConfig,
) -> list[CharacterCandidate]:
    """
    Sort characters left-to-right, with optional two-line grouping.

    Vietnamese plates can be single-line or two-line.  We detect two
    rows by looking for a large vertical gap between character centres.
    """
    if not characters:
        return []
    if len(characters) == 1:
        characters[0].row_index = 0
        return characters

    chars_by_y = sorted(characters, key=lambda c: c.center_y)
    centers = np.array([c.center_y for c in chars_by_y], dtype=np.float32)
    gaps = np.diff(centers)

    use_two_rows = False
    split_index = 0
    if gaps.size > 0:
        largest_gap_idx = int(np.argmax(gaps))
        largest_gap = float(gaps[largest_gap_idx])
        use_two_rows = largest_gap >= cfg.two_line_gap_ratio * plate_height
        split_index = largest_gap_idx + 1

    rows: list[list[CharacterCandidate]]
    if use_two_rows:
        rows = [chars_by_y[:split_index], chars_by_y[split_index:]]
    else:
        rows = [chars_by_y]

    ordered: list[CharacterCandidate] = []
    for row_idx, row in enumerate(rows):
        row_sorted = sorted(row, key=lambda c: c.center_x)
        for char in row_sorted:
            char.row_index = row_idx
        ordered.extend(row_sorted)
    return ordered
