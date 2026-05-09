"""
connected_components.py
=======================

Connected-component labeling (CCL) with the classical **two-pass**
algorithm and a Union-Find / disjoint-set helper.

Problem statement
-----------------
Given a binary image, assign a unique integer label to every maximally
connected blob of foreground pixels.  Two pixels are "connected" if
they are foreground and they touch — for the most useful definition
(8-connectivity) they share an edge or a corner.

Why 8-connectivity here?
------------------------
For license-plate blobs after closing, foreground regions can have
diagonal contacts at the corners of merged characters.  4-connectivity
would split such regions into multiple labels, which would defeat the
whole point of the closing.  8-connectivity is the conservative choice.

Algorithm: two-pass with union-find
-----------------------------------
**Pass 1** scans the image in raster order.  For each foreground pixel
``p`` it inspects the *already-labelled* neighbours that come before it
in scan order (top-left, top, top-right, left for 8-connectivity).
* If none of them is foreground, ``p`` gets a fresh label.
* Otherwise ``p`` inherits the smallest of those labels, and we
  *record* that all the neighbour labels are equivalent in a Union-Find.

**Pass 2** walks the image again and replaces every label with the
representative of its equivalence class.  Then we compactify labels so
they form a contiguous range ``1..K`` (with ``0`` reserved for
background) — this makes downstream array indexing trivial.

Complexity: ``O(N · α(N))`` where ``α`` is the inverse Ackermann (≤ 4
for any conceivable image), thanks to the path-compressed union-find.

Output
------
:class:`CCResult` packages:

* ``labels``      — ``int32`` label image (0 = background, 1..K = blobs)
* ``num_labels``  — K (excludes background)
* ``stats``       — per-blob bounding box, area, centroid in a list of
                    :class:`ComponentStats`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Union-Find / Disjoint-Set
# ---------------------------------------------------------------------------

class _UnionFind:
    """
    Compact disjoint-set with path compression and union-by-rank.

    Used here to merge label equivalence classes during the first CCL
    pass.  Two integer labels live in the same set iff the two pixels
    they label belong to the same connected component.

    We grow the array on demand because we do not know the number of
    initial labels in advance.
    """

    def __init__(self) -> None:
        # parent[i] = parent of label i; parent[i] == i means root.
        self._parent: list[int] = [0]   # index 0 reserved for background
        # rank[i] = upper bound on tree height rooted at i.  Lower-rank
        # trees are attached under higher-rank trees to keep the
        # forest balanced.
        self._rank: list[int] = [0]

    def make_set(self) -> int:
        """Allocate a fresh label and return its integer ID."""
        new_id = len(self._parent)
        self._parent.append(new_id)
        self._rank.append(0)
        return new_id

    def find(self, x: int) -> int:
        """Return the root label of x's equivalence class.

        Uses *path compression*: every node along the lookup path is
        re-parented directly to the root, so subsequent queries are O(1)
        on average.
        """
        # Iterative implementation to avoid Python recursion limits on
        # very long chains.
        root = x
        while self._parent[root] != root:
            root = self._parent[root]

        # Walk again, compressing.
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        """Merge the equivalence classes of a and b (union-by-rank)."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            self._parent[ra] = rb
        elif self._rank[ra] > self._rank[rb]:
            self._parent[rb] = ra
        else:
            self._parent[rb] = ra
            self._rank[ra] += 1

    @property
    def size(self) -> int:
        """Number of allocated labels including the background slot."""
        return len(self._parent)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ComponentStats:
    """
    Geometry summary of a single connected component.

    Attributes
    ----------
    label : int
        Label ID (1-based; 0 is reserved for background).
    x, y : int
        Top-left corner of the axis-aligned bounding box, in pixel
        coordinates.  ``x`` is the column, ``y`` is the row.
    width, height : int
        Bounding box dimensions in pixels (inclusive).
    area : int
        Number of foreground pixels in the component.
    cx, cy : float
        Centroid (mean column, mean row).
    """
    label: int
    x: int
    y: int
    width: int
    height: int
    area: int
    cx: float
    cy: float

    @property
    def aspect_ratio(self) -> float:
        """Width-to-height ratio of the bounding box.  Useful for filtering plates."""
        return self.width / max(1, self.height)

    @property
    def fill_ratio(self) -> float:
        """Area divided by bounding-box area.  ~1.0 for solid blobs, lower for sparse ones."""
        return self.area / max(1, self.width * self.height)


@dataclass
class CCResult:
    """Output of :func:`connected_components`."""
    labels: np.ndarray            # int32 label image, 0 = background
    num_labels: int               # number of foreground components
    stats: list[ComponentStats]   # one entry per label, in order 1..num_labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def connected_components(
    binary: np.ndarray,
    connectivity: int = 8,
) -> CCResult:
    """
    Label the connected components of a binary image.

    Parameters
    ----------
    binary : np.ndarray
        2-D binary image.  Any non-zero value is treated as foreground.
    connectivity : {4, 8}, default 8
        Pixel-adjacency rule.  8-connectivity counts diagonal neighbours.

    Returns
    -------
    CCResult
        Label image, component count, and per-component statistics.

    Notes
    -----
    Implements the classical **two-pass** scan-line algorithm with a
    union-find for label equivalences.  Pass 1 is a Python loop over
    pixels (this is the dominant cost), but it is acceptable for the
    sizes we deal with — license-plate photos rarely exceed a few
    megapixels.  For very large images one would lean on Cython /
    Numba; we deliberately keep things in pure Python+NumPy so the
    algorithm stays didactic.
    """
    if binary.ndim != 2:
        raise ValueError(
            f"connected_components expects a 2-D image; got shape {binary.shape}."
        )
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8; got {connectivity}.")

    H, W = binary.shape
    # Foreground mask as a boolean: avoids casting in the inner loop.
    fg = (binary != 0)

    # Provisional label image; 0 means background or unassigned.
    labels = np.zeros((H, W), dtype=np.int32)

    uf = _UnionFind()

    # ------------------------------------------------------------------
    # Pass 1 — assign provisional labels and record equivalences.
    #
    #   For each foreground pixel we look at its *already-labelled*
    #   neighbours.  In raster order those are:
    #       4-conn :  N (north)         W (west)
    #       8-conn :  NW   N    NE       W
    #
    #   If none of them is foreground, we allocate a new label.  If any
    #   are, we inherit the smallest of their labels and union all of
    #   them so we know later they are the same component.
    # ------------------------------------------------------------------
    for y in range(H):
        for x in range(W):
            if not fg[y, x]:
                continue

            # Collect labels of already-visited foreground neighbours.
            neighbours: list[int] = []

            # West
            if x > 0 and labels[y, x - 1] > 0:
                neighbours.append(int(labels[y, x - 1]))
            # North
            if y > 0 and labels[y - 1, x] > 0:
                neighbours.append(int(labels[y - 1, x]))

            if connectivity == 8:
                # North-west
                if y > 0 and x > 0 and labels[y - 1, x - 1] > 0:
                    neighbours.append(int(labels[y - 1, x - 1]))
                # North-east
                if y > 0 and x + 1 < W and labels[y - 1, x + 1] > 0:
                    neighbours.append(int(labels[y - 1, x + 1]))

            if not neighbours:
                # Fresh component.
                labels[y, x] = uf.make_set()
            else:
                # Inherit the smallest label and union all neighbours.
                smallest = min(neighbours)
                labels[y, x] = smallest
                for n in neighbours:
                    if n != smallest:
                        uf.union(smallest, n)

    # ------------------------------------------------------------------
    # Pass 2 — replace each provisional label with its class root,
    # then compactify labels to a contiguous 1..K range.
    # ------------------------------------------------------------------
    # Build a remap table: provisional label -> compact label.
    # We touch only labels that actually appear, so iteration is bounded
    # by the number of provisional labels created in Pass 1.
    remap: dict[int, int] = {0: 0}     # background stays 0
    next_id = 1
    for prov in range(1, uf.size):
        root = uf.find(prov)
        if root not in remap:
            remap[root] = next_id
            next_id += 1
        # Map the provisional label to whatever its root maps to.
        remap[prov] = remap[root]

    # Vectorized rewrite of the label image using fancy indexing.
    # We build a small lookup array [0, remap[1], remap[2], ...].
    lut = np.zeros(uf.size, dtype=np.int32)
    for k, v in remap.items():
        lut[k] = v
    labels = lut[labels]

    num_labels = next_id - 1  # not counting background

    # ------------------------------------------------------------------
    # Compute per-component statistics in a single pass.
    # ------------------------------------------------------------------
    stats = _compute_stats(labels, num_labels)
    return CCResult(labels=labels, num_labels=num_labels, stats=stats)


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def _compute_stats(labels: np.ndarray, num_labels: int) -> list[ComponentStats]:
    """
    Compute bounding boxes, areas, and centroids for every label > 0.

    Vectorized: we collect per-label sums via :func:`np.add.at`
    (an unbuffered scatter that handles duplicate indices correctly),
    plus per-label min/max for the bounding boxes.

    Parameters
    ----------
    labels : np.ndarray
        Label image with values in ``0..num_labels``.
    num_labels : int
        Number of components (excludes background).

    Returns
    -------
    list[ComponentStats]
        Length ``num_labels``, ordered by label ID (label 1 first).
    """
    if num_labels == 0:
        return []

    # Coordinate grids.
    H, W = labels.shape
    ys, xs = np.indices((H, W))

    # Foreground mask.
    fg = labels > 0

    # Flatten everything to 1-D for scatter ops.
    flat_labels = labels[fg]
    flat_ys = ys[fg]
    flat_xs = xs[fg]

    # Pre-allocate per-label accumulators.  Index 0 is background and
    # will be ignored; we keep it so labels are direct indices.
    K = num_labels + 1
    area = np.zeros(K, dtype=np.int64)
    sum_x = np.zeros(K, dtype=np.int64)
    sum_y = np.zeros(K, dtype=np.int64)
    min_x = np.full(K, W, dtype=np.int64)
    max_x = np.full(K, -1, dtype=np.int64)
    min_y = np.full(K, H, dtype=np.int64)
    max_y = np.full(K, -1, dtype=np.int64)

    # Areas via histogram of labels — fast and exact.
    counts = np.bincount(flat_labels, minlength=K)
    area[: counts.size] = counts

    # Centroid components: weighted sums of x and y per label.
    # np.add.at is the safe, unbuffered scatter that NumPy needs when
    # the indices contain duplicates.
    np.add.at(sum_x, flat_labels, flat_xs)
    np.add.at(sum_y, flat_labels, flat_ys)

    # Bounding boxes: per-label min and max of coordinates.
    # np.minimum.at / np.maximum.at do the same scatter pattern for min/max.
    np.minimum.at(min_x, flat_labels, flat_xs)
    np.maximum.at(max_x, flat_labels, flat_xs)
    np.minimum.at(min_y, flat_labels, flat_ys)
    np.maximum.at(max_y, flat_labels, flat_ys)

    out: list[ComponentStats] = []
    for lbl in range(1, K):
        if area[lbl] == 0:
            continue                  # safety net — shouldn't happen
        x0, y0 = int(min_x[lbl]), int(min_y[lbl])
        x1, y1 = int(max_x[lbl]), int(max_y[lbl])
        out.append(
            ComponentStats(
                label=lbl,
                x=x0,
                y=y0,
                width=x1 - x0 + 1,
                height=y1 - y0 + 1,
                area=int(area[lbl]),
                cx=float(sum_x[lbl] / area[lbl]),
                cy=float(sum_y[lbl] / area[lbl]),
            )
        )
    return out
