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
    cleanup_opening_kernel: tuple[int, int] | None = None
    use_adaptive_threshold: bool = True
    adaptive_window_ratio: float = 0.26
    adaptive_min_window: int = 15
    adaptive_offset: float = 15.0
    combine_otsu_threshold: bool = True
    slot_min_width_to_height: float = 0.40
    slot_vertical_padding_ratio: float = 0.08
    related_fragment_min_area_ratio: float = 0.025
    related_fragment_min_x_overlap_ratio: float = 0.18
    fallback_min_anchor_count: int = 7
    suppress_dense_row_ratio: float = 0.60
    suppress_dense_col_ratio: float = 0.50
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
    keep_largest_glyph_component: bool = True


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

    # Dark glyphs become foreground (255).  Otsu is used for the anchor
    # connected components because it produces cleaner component geometry
    # than local thresholding near plate borders.  The adaptive mask is
    # used later for the actual glyph crop so faint detached strokes can
    # still be recovered.
    anchor_binary, _ = otsu_threshold(plate_gray, invert=True)
    binary = _make_character_binary(plate_gray, cfg, otsu_binary=anchor_binary)

    if cfg.cleanup_opening_kernel is not None:
        kh, kw = cfg.cleanup_opening_kernel
        cleaned = opening(anchor_binary, rect(kh, kw))
    else:
        cleaned = anchor_binary.copy()

    cc = connected_components(cleaned, connectivity=cfg.connectivity)
    anchors = [
        comp
        for comp in cc.stats
        if _looks_like_character(comp, image_shape=(H, W), plate_area=plate_area, cfg=cfg)
    ]
    anchors = _prune_anchor_outliers(anchors)
    if len(anchors) < cfg.fallback_min_anchor_count and cfg.use_adaptive_threshold:
        fallback_binary = _adaptive_anchor_fallback(plate_gray, cfg)
        fallback_cc = connected_components(fallback_binary, connectivity=cfg.connectivity)
        relaxed_cfg = SegmentationConfig(
            **{
                **cfg.__dict__,
                "min_height_ratio": min(cfg.min_height_ratio, 0.20),
                "reject_border_touching": False,
            }
        )
        fallback_anchors = [
            comp
            for comp in fallback_cc.stats
            if _looks_like_character(comp, image_shape=(H, W), plate_area=plate_area, cfg=relaxed_cfg)
        ]
        fallback_anchors = _prune_anchor_outliers(fallback_anchors)
        if len(fallback_anchors) > len(anchors):
            anchors = fallback_anchors
            binary = fallback_binary

    # A glyph can be split into multiple connected components after
    # thresholding.  A typical example is a faint top bar of "7": the
    # main diagonal stroke is a valid tall component, but the top bar is
    # a short detached fragment and would be discarded by pure CCL.  We
    # therefore use tall components as anchors, then crop a character
    # slot around each anchor and keep related fragments inside it.
    raw_chars = _build_slot_candidates(binary, anchors, image_shape=(H, W), cfg=cfg)
    pruned_chars = _prune_edge_artifact_characters(raw_chars, image_shape=(H, W), cfg=cfg)
    projection_chars = _projection_recover_candidates(binary, anchors, image_shape=(H, W), cfg=cfg)
    if _should_use_projection_recovery(pruned_chars, projection_chars):
        pruned_chars = _prune_edge_artifact_characters(
            projection_chars,
            image_shape=(H, W),
            cfg=cfg,
        )

    ordered = _assign_rows_and_sort(pruned_chars, plate_height=H, cfg=cfg)

    return SegmentationResult(
        binary=binary,
        cleaned=cleaned,
        labels=cc.labels,
        characters=ordered,
        config=cfg,
    )


def _make_character_binary(
    plate_gray: np.ndarray,
    cfg: SegmentationConfig,
    otsu_binary: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Binarize a normalized plate for character segmentation.

    Global Otsu works when the whole plate has one clean foreground and
    background distribution.  Real crops often have grill shadows,
    reflections, or one side of a character brighter than another, so a
    local mean threshold is used as the primary cue.  When configured,
    the Otsu mask is OR-ed back in as a conservative fallback.
    """
    if otsu_binary is None:
        otsu_binary, _ = otsu_threshold(plate_gray, invert=True)
    if not cfg.use_adaptive_threshold:
        return otsu_binary

    window = _adaptive_window_size(plate_gray.shape, cfg)
    adaptive = _adaptive_dark_threshold(
        plate_gray,
        window_size=window,
        offset=cfg.adaptive_offset,
    )
    if cfg.combine_otsu_threshold:
        return np.maximum(otsu_binary, adaptive)
    return adaptive


def _prune_anchor_outliers(anchors: list[ComponentStats]) -> list[ComponentStats]:
    """
    Drop skinny low-area components that match plate borders, not glyphs.

    This pass is intentionally relative to the detected text row.  A
    true "1" can be narrow, but it still has area close to the other
    character anchors.  Border slivers tend to be both much narrower and
    much smaller than the median anchor.
    """
    if len(anchors) < 3:
        return anchors

    widths = np.array([comp.width for comp in anchors], dtype=np.float32)
    areas = np.array([comp.area for comp in anchors], dtype=np.float32)
    median_width = float(np.median(widths))
    median_area = float(np.median(areas))
    if median_width <= 0 or median_area <= 0:
        return anchors

    pruned: list[ComponentStats] = []
    for comp in anchors:
        very_skinny = comp.width < 0.45 * median_width
        very_small = comp.area < 0.40 * median_area
        if very_skinny and very_small:
            continue
        pruned.append(comp)
    return pruned


def _adaptive_anchor_fallback(
    plate_gray: np.ndarray,
    cfg: SegmentationConfig,
) -> np.ndarray:
    """
    Build a fallback anchor mask for low-contrast characters near borders.

    Some synthetic or tightly cropped samples place dark characters close
    to a dark plate/background border.  Global Otsu merges those glyphs
    into one huge border component.  The fallback uses only the local
    adaptive mask, then suppresses dense horizontal/vertical border
    lines before connected-component labeling.
    """
    adaptive = _adaptive_dark_threshold(
        plate_gray,
        window_size=_adaptive_window_size(plate_gray.shape, cfg),
        offset=cfg.adaptive_offset,
    )
    return _suppress_dense_foreground_lines(
        adaptive,
        row_ratio=cfg.suppress_dense_row_ratio,
        col_ratio=cfg.suppress_dense_col_ratio,
    )


def _suppress_dense_foreground_lines(
    binary: np.ndarray,
    row_ratio: float,
    col_ratio: float,
) -> np.ndarray:
    """Remove likely plate-border rows/columns from a binary mask."""
    out = binary.copy()
    H, W = out.shape
    if 0.0 < row_ratio < 1.0:
        dense_rows = (out > 0).sum(axis=1) > row_ratio * W
        out[dense_rows, :] = 0
    if 0.0 < col_ratio < 1.0:
        dense_cols = (out > 0).sum(axis=0) > col_ratio * H
        out[:, dense_cols] = 0
    return out


def _adaptive_window_size(
    image_shape: tuple[int, int],
    cfg: SegmentationConfig,
) -> int:
    """Return an odd local-threshold window size tied to plate height."""
    H, W = image_shape
    base = int(round(H * cfg.adaptive_window_ratio))
    window = max(cfg.adaptive_min_window, base)
    window = min(window, max(3, H if H % 2 == 1 else H - 1), max(3, W if W % 2 == 1 else W - 1))
    if window % 2 == 0:
        window -= 1
    return max(3, window)


def _adaptive_dark_threshold(
    gray: np.ndarray,
    window_size: int,
    offset: float,
) -> np.ndarray:
    """
    Threshold dark text against a local mean image.

    A pixel becomes foreground when it is at least ``offset`` intensity
    levels darker than its local neighbourhood.  The local mean is
    computed with an integral image, so the implementation remains
    deterministic and fast without relying on OpenCV.
    """
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError(f"window_size must be a positive odd integer; got {window_size}.")

    arr = gray.astype(np.float32, copy=False)
    H, W = arr.shape
    radius = window_size // 2

    integral = np.pad(arr, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    ones = np.ones((H, W), dtype=np.float32)
    count_integral = np.pad(ones, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)

    y = np.arange(H)
    x = np.arange(W)
    y0 = np.maximum(0, y - radius)
    y1 = np.minimum(H, y + radius + 1)
    x0 = np.maximum(0, x - radius)
    x1 = np.minimum(W, x + radius + 1)

    sums = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    counts = (
        count_integral[y1[:, None], x1[None, :]]
        - count_integral[y0[:, None], x1[None, :]]
        - count_integral[y1[:, None], x0[None, :]]
        + count_integral[y0[:, None], x0[None, :]]
    )
    local_mean = sums / counts
    return ((arr < (local_mean - offset)).astype(np.uint8)) * 255


def _build_slot_candidates(
    binary: np.ndarray,
    anchors: list[ComponentStats],
    image_shape: tuple[int, int],
    cfg: SegmentationConfig,
) -> list[CharacterCandidate]:
    """
    Build character crops from tall anchor components and slot boxes.

    Connected components are excellent for finding the main vertical or
    diagonal body of a glyph.  They are not sufficient as final crop
    boxes, because thin strokes may detach.  Slot boxes use neighbour
    spacing to recover the full character region before normalization.
    """
    if not anchors:
        return []

    H, W = image_shape
    characters: list[CharacterCandidate] = []
    for row in _group_anchor_rows(anchors, plate_height=H, cfg=cfg):
        row_sorted = sorted(row, key=lambda comp: comp.cx)
        heights = np.array([comp.height for comp in row_sorted], dtype=np.float32)
        median_height = float(np.median(heights)) if heights.size else 1.0
        y_pad = max(1, int(round(median_height * cfg.slot_vertical_padding_ratio)))
        row_y0 = max(0, min(comp.y for comp in row_sorted) - y_pad)
        row_y1 = min(H, max(comp.y + comp.height for comp in row_sorted) + y_pad)

        for index, anchor in enumerate(row_sorted):
            x0, x1 = _slot_x_bounds(row_sorted, index, image_width=W, cfg=cfg)
            slot = binary[row_y0:row_y1, x0:x1]
            glyph = _related_slot_glyph(
                slot,
                anchor=anchor,
                slot_origin=(x0, row_y0),
                cfg=cfg,
            )
            if not np.any(glyph):
                continue
            normalized = normalize_character(glyph, target_shape=cfg.char_shape)
            characters.append(
                CharacterCandidate(
                    x=x0,
                    y=row_y0,
                    width=x1 - x0,
                    height=row_y1 - row_y0,
                    image=glyph,
                    normalized=normalized,
                    row_index=0,
                    component=anchor,
                )
            )

    return characters


def _projection_recover_candidates(
    binary: np.ndarray,
    anchors: list[ComponentStats],
    image_shape: tuple[int, int],
    cfg: SegmentationConfig,
) -> list[CharacterCandidate]:
    """
    Recover split slots from vertical projection valleys.

    When decorations or reflective borders connect several glyphs into
    one large component, the anchor-based slotter may build overly wide
    crops that contain two characters.  Vertical projection is a useful
    fallback for this specific failure mode because true printed glyphs
    still create tall dense column runs while separator dots and screws
    stay short.
    """
    if not anchors:
        return []

    H, W = image_shape
    characters: list[CharacterCandidate] = []
    for row in _group_anchor_rows(anchors, plate_height=H, cfg=cfg):
        row_sorted = sorted(row, key=lambda comp: comp.cx)
        heights = np.array([comp.height for comp in row_sorted], dtype=np.float32)
        median_height = float(np.median(heights)) if heights.size else 1.0
        y_pad = max(1, int(round(median_height * cfg.slot_vertical_padding_ratio)))
        row_y0 = max(0, min(comp.y for comp in row_sorted) - y_pad)
        row_y1 = min(H, max(comp.y + comp.height for comp in row_sorted) + y_pad)
        row_height = max(1, row_y1 - row_y0)

        foreground = binary[row_y0:row_y1, :] > 0
        col_counts = foreground.sum(axis=0)
        threshold = max(3, int(round(0.30 * row_height)))
        runs = _merge_projection_runs(
            _column_runs(col_counts > threshold),
            row_height=row_height,
        )

        min_width = max(3, int(round(0.12 * row_height)))
        max_width = max(min_width + 1, int(round(0.85 * row_height)))
        for x0, x1 in runs:
            width = x1 - x0
            if width < min_width or width > max_width:
                continue
            glyph = binary[row_y0:row_y1, x0:x1]
            if not np.any(glyph):
                continue
            area = int(np.count_nonzero(glyph))
            comp = ComponentStats(
                label=0,
                x=x0,
                y=row_y0,
                width=width,
                height=row_height,
                area=area,
                cx=x0 + (width - 1) / 2.0,
                cy=row_y0 + (row_height - 1) / 2.0,
            )
            normalized = normalize_character(glyph, target_shape=cfg.char_shape)
            characters.append(
                CharacterCandidate(
                    x=x0,
                    y=row_y0,
                    width=width,
                    height=row_height,
                    image=glyph.copy(),
                    normalized=normalized,
                    row_index=0,
                    component=comp,
                )
            )

    return characters


def _column_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs of true values in a one-dimensional mask."""
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for idx, value in enumerate(mask):
        if value and not in_run:
            start = idx
            in_run = True
        elif not value and in_run:
            runs.append((start, idx))
            in_run = False
    if in_run:
        runs.append((start, int(mask.shape[0])))
    return runs


def _merge_projection_runs(
    runs: list[tuple[int, int]],
    row_height: int,
) -> list[tuple[int, int]]:
    """
    Merge split stroke pairs that belong to one hollow glyph.

    Characters such as ``0`` can appear as two dense vertical strokes in
    a projection profile, with a low-count hole between them.  Two real
    neighbouring characters are much wider when merged, so the combined
    width guard keeps ordinary inter-character gaps separate.
    """
    if len(runs) <= 1:
        return runs

    max_gap = max(2, int(round(0.20 * row_height)))
    max_merged_width = max(4, int(round(0.58 * row_height)))
    merged: list[tuple[int, int]] = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        gap = start - prev_end
        combined_width = end - prev_start
        if 0 <= gap <= max_gap and combined_width <= max_merged_width:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _should_use_projection_recovery(
    raw_chars: list[CharacterCandidate],
    projection_chars: list[CharacterCandidate],
) -> bool:
    """Decide whether projection recovered a more plausible segmentation."""
    if not (4 <= len(projection_chars) <= 10):
        return False
    if not raw_chars or not projection_chars:
        return False

    widths = np.array([char.width for char in raw_chars], dtype=np.float32)
    median_width = float(np.median(widths))
    if median_width <= 0:
        return False
    has_wide_slot = bool(np.any(widths > 1.45 * median_width))
    if len(projection_chars) > len(raw_chars):
        return len(raw_chars) < 4 or has_wide_slot
    return False


def _prune_edge_artifact_characters(
    characters: list[CharacterCandidate],
    image_shape: tuple[int, int],
    cfg: SegmentationConfig,
) -> list[CharacterCandidate]:
    """
    Remove frame/logo fragments that masquerade as edge characters.

    Real-world plate crops often include coloured borders, screws, or a
    small state/logo mark at the left or right edge.  Those pieces can be
    tall enough to pass the connected-component filters and then become
    fake ``I``/``1`` characters.  This pass is intentionally conservative:
    it only trims row-edge candidates whose foreground is much smaller or
    whose anchor is much shorter than the row's dominant glyphs.  Interior
    narrow glyphs such as ``1`` are left alone.
    """
    if len(characters) < 3:
        return characters

    H, W = image_shape
    pruned: list[CharacterCandidate] = []
    rows = _drop_decorative_short_rows(
        _group_character_rows(characters, plate_height=H, cfg=cfg)
    )
    for row in rows:
        row_sorted = sorted(row, key=lambda c: c.center_x)
        row_pruned = _trim_row_edge_artifacts(row_sorted, image_width=W)
        pruned.extend(row_pruned)
    return pruned


def _drop_decorative_short_rows(
    rows: list[list[CharacterCandidate]],
) -> list[list[CharacterCandidate]]:
    """
    Drop small decorative rows when a main text row is already present.

    Some plates include city/state text above the OCR text.  If that
    text survives thresholding as one or two components, it should not be
    returned as license characters.  True two-line plates, such as a
    3+5 layout, keep both rows because neither row is tiny.
    """
    if len(rows) <= 1:
        return rows
    main_lengths = [len(row) for row in rows]
    if max(main_lengths) < 4:
        return rows
    filtered = [row for row in rows if len(row) > 2]
    return filtered or rows


def _group_character_rows(
    characters: list[CharacterCandidate],
    plate_height: int,
    cfg: SegmentationConfig,
) -> list[list[CharacterCandidate]]:
    """Group already-built character slots into text rows."""
    if len(characters) <= 1:
        return [characters]

    by_y = sorted(characters, key=lambda char: char.center_y)
    centers = np.array([char.center_y for char in by_y], dtype=np.float32)
    gaps = np.diff(centers)
    if gaps.size == 0:
        return [by_y]

    largest_gap_idx = int(np.argmax(gaps))
    largest_gap = float(gaps[largest_gap_idx])
    if largest_gap >= cfg.two_line_gap_ratio * plate_height:
        split = largest_gap_idx + 1
        return [by_y[:split], by_y[split:]]
    return [by_y]


def _trim_row_edge_artifacts(
    row: list[CharacterCandidate],
    image_width: int,
) -> list[CharacterCandidate]:
    """Trim suspicious first/last slots from one text row."""
    if len(row) < 3:
        return row

    metrics = [_normalized_foreground_metrics(char.normalized) for char in row]
    fg_widths = np.array([m[0] for m in metrics], dtype=np.float32)
    fg_areas = np.array([m[1] for m in metrics], dtype=np.float32)
    comp_heights = np.array([char.component.height for char in row], dtype=np.float32)
    median_width = float(np.median(fg_widths))
    median_area = float(np.median(fg_areas))
    median_height = float(np.median(comp_heights))
    if median_width <= 0 or median_area <= 0 or median_height <= 0:
        return row

    keep = [True] * len(row)
    edge_margin = max(2, int(round(0.08 * image_width)))

    def is_artifact(idx: int, current_left: int, current_right: int) -> bool:
        char = row[idx]
        fg_width, fg_area = metrics[idx]
        comp_short = char.component.height < 0.78 * median_height
        low_area = fg_area < 0.46 * median_area
        skinny_low_mass = fg_width <= 0.62 * median_width and fg_area < 0.70 * median_area
        touches_outer_crop = char.x <= edge_margin or char.x + char.width >= image_width - edge_margin
        near_row_edge = idx <= current_left + 1 or idx >= current_right - 1
        current_outer_edge = idx == current_left or idx == current_right
        hard_outer_frame = touches_outer_crop and len(row) > 8 and current_outer_edge
        short_decorative_edge = (
            comp_short
            and current_outer_edge
            and fg_width >= 11
            and (touches_outer_crop or len(row) >= 8)
        )
        outer_skinny_frame = touches_outer_crop and near_row_edge and (low_area or skinny_low_mass)
        far_isolated_edge = False
        if touches_outer_crop and current_outer_edge and len(row) >= 4:
            if idx == current_left and idx + 1 < len(row):
                gap = row[idx + 1].x - (char.x + char.width)
            elif idx == current_right and idx - 1 >= 0:
                gap = char.x - (row[idx - 1].x + row[idx - 1].width)
            else:
                gap = 0
            far_isolated_edge = gap > 1.20 * median_width
        smaller_than_next = False
        if len(row) >= 8 and idx == current_left and idx + 1 < len(row):
            next_width, next_area = metrics[idx + 1]
            edge_prefix_width = char.width <= 18 or fg_width <= 14
            smaller_than_next = (
                edge_prefix_width
                and fg_width >= 11
                and char.width < 0.86 * row[idx + 1].width
                and fg_width < 0.82 * max(1, next_width)
                and fg_area < 0.75 * max(1, next_area)
            )
        return (
            hard_outer_frame
            or short_decorative_edge
            or outer_skinny_frame
            or far_isolated_edge
            or smaller_than_next
            or near_row_edge
            and (touches_outer_crop or len(row) > 8)
            and (comp_short or low_area or skinny_low_mass)
        )

    left = 0
    right = len(row) - 1
    while left < right - 1 and is_artifact(left, left, right):
        keep[left] = False
        left += 1

    while right > left + 1 and is_artifact(right, left, right):
        keep[right] = False
        right -= 1

    return [char for char, keep_char in zip(row, keep) if keep_char]


def _normalized_foreground_metrics(char_image: np.ndarray) -> tuple[int, int]:
    """Return tight foreground width and foreground area for a 32x32 glyph."""
    fg = char_image > 0
    if not np.any(fg):
        return (0, 0)
    _ys, xs = np.nonzero(fg)
    width = int(xs.max() - xs.min() + 1)
    area = int(fg.sum())
    return (width, area)


def _group_anchor_rows(
    anchors: list[ComponentStats],
    plate_height: int,
    cfg: SegmentationConfig,
) -> list[list[ComponentStats]]:
    """Group anchor components into one or two text rows before slotting."""
    if len(anchors) <= 1:
        return [anchors]

    by_y = sorted(anchors, key=lambda comp: comp.cy)
    centers = np.array([comp.cy for comp in by_y], dtype=np.float32)
    gaps = np.diff(centers)
    if gaps.size == 0:
        return [by_y]

    largest_gap_idx = int(np.argmax(gaps))
    largest_gap = float(gaps[largest_gap_idx])
    if largest_gap >= cfg.two_line_gap_ratio * plate_height:
        split = largest_gap_idx + 1
        return [by_y[:split], by_y[split:]]
    return [by_y]


def _slot_x_bounds(
    row: list[ComponentStats],
    index: int,
    image_width: int,
    cfg: SegmentationConfig,
) -> tuple[int, int]:
    """
    Estimate horizontal character-slot boundaries from neighbouring anchors.

    Midpoints between adjacent anchor centres prevent separator dots or
    inter-character whitespace from becoming standalone glyphs, while a
    minimum width tied to character height prevents narrow strokes such
    as "1" and "7" from being cropped too tightly.
    """
    anchor = row[index]
    centers = [comp.cx for comp in row]
    hard_left = 0.0
    hard_right = float(image_width)

    if len(row) == 1:
        pitch = max(anchor.width * (1.0 + 2.0 * cfg.padding_ratio), anchor.height * cfg.slot_min_width_to_height)
        left = anchor.cx - pitch / 2.0
        right = anchor.cx + pitch / 2.0
    else:
        if index == 0:
            pitch = centers[1] - centers[0]
            left = anchor.cx - pitch / 2.0
        else:
            hard_left = (centers[index - 1] + centers[index]) / 2.0
            left = hard_left

        if index == len(row) - 1:
            pitch = centers[-1] - centers[-2]
            right = anchor.cx + pitch / 2.0
        else:
            hard_right = (centers[index] + centers[index + 1]) / 2.0
            right = hard_right

    min_width = max(
        anchor.width * (1.0 + 2.0 * cfg.padding_ratio),
        anchor.height * cfg.slot_min_width_to_height,
    )
    if (right - left) < min_width:
        center = (left + right) / 2.0
        left = center - min_width / 2.0
        right = center + min_width / 2.0

    # Respect neighbour midpoints.  Expanding a slot across the midpoint
    # is what makes tight pairs such as "70" bleed into each other.
    left = max(left, hard_left)
    right = min(right, hard_right)

    # For the outermost slots there is no neighbouring midpoint on one
    # side, so include the anchor itself plus a tiny safety margin.
    if index == 0:
        left = min(left, anchor.x - 1)
    if index == len(row) - 1:
        right = max(right, anchor.x + anchor.width + 1)

    x0 = max(0, int(np.floor(left)))
    x1 = min(image_width, int(np.ceil(right)))
    if x1 <= x0:
        x1 = min(image_width, x0 + 1)
    return x0, x1


def _related_slot_glyph(
    slot: np.ndarray,
    anchor: ComponentStats,
    slot_origin: tuple[int, int],
    cfg: SegmentationConfig,
) -> np.ndarray:
    """
    Keep the anchor and detached fragments that plausibly belong to it.

    The separator dot in a plate should not become part of a neighbouring
    glyph, but a detached top bar of "7" should.  Horizontal overlap with
    the anchor is the strongest cue: real detached strokes usually sit
    above or below the main glyph body, while separators live between
    neighbouring character slots.
    """
    cc = connected_components(slot, connectivity=cfg.connectivity)
    if cc.num_labels == 0:
        return np.zeros_like(slot, dtype=np.uint8)

    origin_x, origin_y = slot_origin
    anchor_x0 = anchor.x - origin_x
    anchor_x1 = anchor_x0 + anchor.width
    anchor_y0 = anchor.y - origin_y
    anchor_y1 = anchor_y0 + anchor.height
    anchor_area = max(1, anchor.area)

    kept = np.zeros_like(slot, dtype=np.uint8)
    for comp in cc.stats:
        if comp.area < max(2, int(round(anchor_area * cfg.related_fragment_min_area_ratio))):
            continue

        overlap_x = _range_overlap(comp.x, comp.x + comp.width, anchor_x0, anchor_x1)
        overlap_y = _range_overlap(comp.y, comp.y + comp.height, anchor_y0, anchor_y1)
        x_overlap_ratio = overlap_x / max(1, min(comp.width, anchor.width))
        y_overlap_ratio = overlap_y / max(1, min(comp.height, anchor.height))
        tall_body = comp.height >= 0.45 * anchor.height and overlap_y > 0
        related_by_x = x_overlap_ratio >= cfg.related_fragment_min_x_overlap_ratio
        related_by_y = y_overlap_ratio >= 0.45 and overlap_x > 0

        touches_slot_edge = (
            comp.x <= 0
            or comp.y <= 0
            or comp.x + comp.width >= slot.shape[1]
            or comp.y + comp.height >= slot.shape[0]
        )
        thin_border_fragment = touches_slot_edge and comp.height <= 0.18 * anchor.height

        if thin_border_fragment:
            continue
        if tall_body or related_by_x or related_by_y:
            kept[cc.labels == comp.label] = 255

    return kept


def _range_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Return the overlap length of two half-open one-dimensional ranges."""
    return max(0.0, min(a1, b1) - max(a0, b0))


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


def clean_character_crop(char_binary: np.ndarray) -> np.ndarray:
    """
    Remove detached border fragments from a character crop.

    Real plate crops often contain tiny horizontal pieces from the top
    or bottom plate border.  Those fragments are white foreground, so if
    we leave them in, the tight bounding box becomes too tall and the
    classifier sees a distorted glyph.  For printed plate characters the
    main glyph is normally one connected component, so keeping the
    dominant component is a robust cleanup step.
    """
    if char_binary.ndim != 2:
        raise ValueError(
            f"clean_character_crop expects a 2-D image; got shape {char_binary.shape}."
        )
    cc = connected_components(char_binary, connectivity=8)
    if cc.num_labels <= 1:
        return char_binary.copy()

    # Prefer the largest foreground component, but add a mild centrality
    # term so a long plate-border strip near the crop edge does not beat
    # the actual glyph in unusual cases.
    H, W = char_binary.shape
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    best = max(
        cc.stats,
        key=lambda comp: comp.area
        - 0.12 * comp.area * (abs(comp.cx - cx) / max(1.0, W) + abs(comp.cy - cy) / max(1.0, H)),
    )
    cleaned = np.zeros_like(char_binary, dtype=np.uint8)
    cleaned[cc.labels == best.label] = 255
    return cleaned


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

    Some plates are single-line and others are two-line.  We detect two
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
