"""
Detection module (Pipeline Step 2)
==================================

Locates the bounding box of a license plate in the preprocessed image
produced by Step 1.

Sub-modules
-----------
sobel                   3×3 Sobel gradients (separable).
morphology              Dilation, erosion, opening, closing on binary images.
connected_components    Two-pass labelling with union-find + per-blob stats.
plate_detector          End-to-end detector with geometric + density filtering.

Typical usage
-------------
::

    from src.preprocessing import preprocess
    from src.detection import detect_plate

    pre = preprocess(image)
    det = detect_plate(pre.enhanced)
    if det.candidates:
        x, y, w, h = det.candidates[0].as_box()
"""

from src.detection.sobel import sobel, sobel_x, SobelResult
from src.detection.morphology import (
    rect, cross,
    dilate, erode, opening, closing,
)
from src.detection.connected_components import (
    connected_components,
    ComponentStats,
    CCResult,
)
from src.detection.plate_detector import (
    detect_plate,
    draw_candidates,
    DetectionConfig,
    DetectionResult,
    PlateCandidate,
)

__all__ = [
    # sobel
    "sobel", "sobel_x", "SobelResult",
    # morphology
    "rect", "cross", "dilate", "erode", "opening", "closing",
    # connected components
    "connected_components", "ComponentStats", "CCResult",
    # plate detector
    "detect_plate", "draw_candidates",
    "DetectionConfig", "DetectionResult", "PlateCandidate",
]
