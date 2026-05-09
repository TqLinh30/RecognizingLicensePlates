"""
demo.py
=======

Command-line demo for the Step-2 plate-detection module.

Usage
-----
::

    python -m src.detection.demo path/to/image.jpg
    python -m src.detection.demo path/to/image.jpg --output data/output

The script writes:
    output/05_gradient.png      — |∂I/∂x|
    output/06_binary.png        — thresholded gradient
    output/07_closed.png        — after morphological closing
    output/08_candidates.png    — boxes drawn on original image
    output/strip_step2.png      — gradient / binary / closed / annotated

and prints each candidate's score and bounding box to stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running with `python src/detection/demo.py` as well as the
# preferred `python -m src.detection.demo` form.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.pipeline import preprocess                        # noqa: E402
from src.detection.plate_detector import detect_plate, draw_candidates   # noqa: E402
from src.utils.image_io import load_image, save_image                    # noqa: E402
from src.utils.visualization import save_side_by_side                    # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 2 — license-plate detection demo."
    )
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output"),
        help="Directory where intermediate stages are saved.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1 — preprocess so we have an enhanced grayscale image.
    # ------------------------------------------------------------------
    image = load_image(args.image)
    print(f"[demo] loaded {args.image} with shape {image.shape}")

    pre = preprocess(image)
    print(f"[demo] preprocessing OK, Otsu threshold = {pre.otsu_threshold_value}")

    # ------------------------------------------------------------------
    # Step 2 — detect plate candidates.
    # ------------------------------------------------------------------
    det = detect_plate(pre.enhanced)
    print(f"[demo] {len(det.candidates)} plate candidate(s) found:")
    for i, cand in enumerate(det.candidates, 1):
        print(
            f"  #{i}  box=({cand.x}, {cand.y}, {cand.width}, {cand.height})"
            f"  ar={cand.aspect_ratio:.2f}"
            f"  fill={cand.fill_ratio:.2f}"
            f"  grad={cand.gradient_density:.2f}"
            f"  score={cand.score:.3f}"
        )

    # ------------------------------------------------------------------
    # Persist every intermediate map.
    # ------------------------------------------------------------------
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    save_image(det.gradient, out_dir / "05_gradient.png")
    save_image(det.binary,   out_dir / "06_binary.png")
    save_image(det.closed,   out_dir / "07_closed.png")

    annotated = draw_candidates(image, det.candidates)
    save_image(annotated, out_dir / "08_candidates.png")

    save_side_by_side(
        [det.gradient, det.binary, det.closed, annotated],
        out_dir / "strip_step2.png",
    )

    print(f"[demo] outputs written to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
