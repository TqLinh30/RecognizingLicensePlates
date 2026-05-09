"""
hough_transform.py
==================

Small, from-scratch Hough transform helpers for plate deskewing.

The detector from Step 2 gives us an axis-aligned bounding box.  Real
photos often contain a plate that is rotated by a few degrees inside
that box, so Step 3 needs a reliable way to estimate the angle before
resizing the plate to a canonical shape.

Line model
----------
We use the standard polar line equation:

    rho = x * cos(theta) + y * sin(theta)

where ``theta`` is the angle of the line normal.  A perfectly horizontal
line has ``theta = 90 degrees`` because its normal points vertically.
The visual orientation of the line is therefore approximately
``theta - 90`` degrees.

For license plates we deliberately search only around horizontal lines
(``90 +/- angle_limit``).  This ignores the many vertical character
strokes and lets the top/bottom plate borders dominate the vote.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class HoughLine:
    """One detected line in polar coordinates."""

    rho: float
    theta_degrees: float
    votes: int

    @property
    def orientation_degrees(self) -> float:
        """Line orientation in image coordinates, where 0 means horizontal."""
        return self.theta_degrees - 90.0


@dataclass
class HoughResult:
    """Accumulator and metadata returned by :func:`hough_lines`."""

    accumulator: np.ndarray
    rho_values: np.ndarray
    theta_values: np.ndarray
    lines: list[HoughLine]


def hough_lines(
    edge_image: np.ndarray,
    theta_values: Sequence[float] | None = None,
    rho_step: float = 1.0,
    min_votes: int | None = None,
    max_lines: int = 10,
    nms_rho_radius: int = 6,
    nms_theta_radius: int = 2,
) -> HoughResult:
    """
    Vote for straight lines in a binary or edge-magnitude image.

    Parameters
    ----------
    edge_image : np.ndarray
        2-D image where non-zero pixels vote.  It can be a binary image
        or an edge magnitude map after thresholding.
    theta_values : sequence of float, optional
        Candidate normal angles in degrees.  If omitted, searches the
        full ``[0, 180)`` range at one-degree resolution.
    rho_step : float
        Spacing between rho bins in pixels.
    min_votes : int, optional
        Minimum accumulator value for accepting a line.  Defaults to
        ``max(8, 5% of the smaller image side)``.
    max_lines : int
        Maximum number of non-maximum-suppressed lines to return.
    nms_rho_radius, nms_theta_radius : int
        Suppression window in accumulator bins.

    Returns
    -------
    HoughResult
        The accumulator, coordinate arrays, and top lines sorted by
        descending vote count.
    """
    if edge_image.ndim != 2:
        raise ValueError(
            f"hough_lines expects a 2-D edge image; got shape {edge_image.shape}."
        )
    if rho_step <= 0:
        raise ValueError(f"rho_step must be positive; got {rho_step}.")
    if max_lines < 1:
        raise ValueError(f"max_lines must be at least 1; got {max_lines}.")

    H, W = edge_image.shape
    ys, xs = np.nonzero(edge_image)
    if theta_values is None:
        theta_values = np.arange(0.0, 180.0, 1.0, dtype=np.float32)
    theta_arr = np.asarray(theta_values, dtype=np.float32)
    if theta_arr.ndim != 1 or theta_arr.size == 0:
        raise ValueError("theta_values must be a non-empty 1-D sequence.")

    rho_max = float(np.hypot(H, W))
    rho_values = np.arange(-rho_max, rho_max + rho_step, rho_step, dtype=np.float32)
    accumulator = np.zeros((rho_values.size, theta_arr.size), dtype=np.int32)

    if xs.size == 0:
        return HoughResult(accumulator, rho_values, theta_arr, [])

    theta_rad = np.deg2rad(theta_arr)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    # Vote one theta column at a time.  This keeps memory small while
    # still using NumPy's vectorized bincount for every angle.
    for j in range(theta_arr.size):
        rho = xs * cos_t[j] + ys * sin_t[j]
        rho_idx = np.rint((rho - rho_values[0]) / rho_step).astype(np.int32)
        valid = (rho_idx >= 0) & (rho_idx < rho_values.size)
        accumulator[:, j] = np.bincount(
            rho_idx[valid],
            minlength=rho_values.size,
        )[: rho_values.size]

    if min_votes is None:
        min_votes = max(8, int(round(0.05 * min(H, W))))

    lines = _extract_peaks(
        accumulator,
        rho_values,
        theta_arr,
        min_votes=min_votes,
        max_lines=max_lines,
        nms_rho_radius=nms_rho_radius,
        nms_theta_radius=nms_theta_radius,
    )
    return HoughResult(accumulator, rho_values, theta_arr, lines)


def estimate_skew_angle(
    edge_image: np.ndarray,
    angle_limit: float = 15.0,
    theta_step: float = 0.5,
    min_votes: int | None = None,
) -> float:
    """
    Estimate plate skew angle from near-horizontal Hough lines.

    The returned value is the line orientation in degrees.  Positive
    values mean the strongest horizontal line tilts downward as x grows
    in image coordinates.  To deskew the plate, rotate by ``-angle``.

    If no reliable line is found, returns ``0.0`` so callers can safely
    continue without deskewing.
    """
    if angle_limit <= 0:
        raise ValueError(f"angle_limit must be positive; got {angle_limit}.")
    if theta_step <= 0:
        raise ValueError(f"theta_step must be positive; got {theta_step}.")

    theta_values = np.arange(
        90.0 - angle_limit,
        90.0 + angle_limit + theta_step,
        theta_step,
        dtype=np.float32,
    )
    result = hough_lines(
        edge_image,
        theta_values=theta_values,
        min_votes=min_votes,
        max_lines=5,
        nms_rho_radius=6,
        nms_theta_radius=max(1, int(round(1.0 / theta_step))),
    )
    if not result.lines:
        return 0.0

    # Use a vote-weighted average of the strongest near-horizontal
    # lines.  Averaging is less jumpy than picking a single peak when
    # top/bottom borders both exist but vote in neighbouring bins.
    votes = np.array([line.votes for line in result.lines], dtype=np.float64)
    angles = np.array([line.orientation_degrees for line in result.lines], dtype=np.float64)
    return float(np.sum(angles * votes) / max(1.0, np.sum(votes)))


def _extract_peaks(
    accumulator: np.ndarray,
    rho_values: np.ndarray,
    theta_values: np.ndarray,
    min_votes: int,
    max_lines: int,
    nms_rho_radius: int,
    nms_theta_radius: int,
) -> list[HoughLine]:
    """Pick strong accumulator peaks with simple rectangular NMS."""
    work = accumulator.copy()
    lines: list[HoughLine] = []

    while len(lines) < max_lines:
        idx = int(np.argmax(work))
        votes = int(work.flat[idx])
        if votes < min_votes:
            break

        rho_idx, theta_idx = np.unravel_index(idx, work.shape)
        lines.append(
            HoughLine(
                rho=float(rho_values[rho_idx]),
                theta_degrees=float(theta_values[theta_idx]),
                votes=votes,
            )
        )

        r0 = max(0, rho_idx - nms_rho_radius)
        r1 = min(work.shape[0], rho_idx + nms_rho_radius + 1)
        t0 = max(0, theta_idx - nms_theta_radius)
        t1 = min(work.shape[1], theta_idx + nms_theta_radius + 1)
        work[r0:r1, t0:t1] = 0

    return lines
