"""
demo.py
=======

Command-line demo for the Step-1 preprocessing module.

Usage
-----
::

    python -m src.preprocessing.demo path/to/image.jpg
    python -m src.preprocessing.demo path/to/image.jpg --output data/output

The script writes:
    output/01_grayscale.png
    output/02_blurred.png
    output/03_enhanced.png
    output/04_binary.png
    output/strip.png            (all four stages side-by-side)

and prints the Otsu threshold to stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running with `python src/preprocessing/demo.py` as well as the
# preferred `python -m src.preprocessing.demo` form.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.pipeline import preprocess, PreprocessConfig  # noqa: E402
from src.utils.image_io import load_image, save_image                # noqa: E402
from src.utils.visualization import save_side_by_side                # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 1 — preprocessing demo for license-plate images."
    )
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output"),
        help="Directory where intermediate stages are saved.",
    )
    parser.add_argument(
        "--no-invert",
        action="store_true",
        help="Disable Otsu inversion (keep dark characters as 0).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load.
    # ------------------------------------------------------------------
    image = load_image(args.image)
    print(f"[demo] loaded {args.image} with shape {image.shape}")

    # ------------------------------------------------------------------
    # Run the pipeline.
    # ------------------------------------------------------------------
    cfg = PreprocessConfig(otsu_invert=not args.no_invert)
    result = preprocess(image, cfg)
    print(f"[demo] Otsu threshold = {result.otsu_threshold_value}")

    # ------------------------------------------------------------------
    # Persist every stage.
    # ------------------------------------------------------------------
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    save_image(result.grayscale, out_dir / "01_grayscale.png")
    save_image(result.blurred,   out_dir / "02_blurred.png")
    save_image(result.enhanced,  out_dir / "03_enhanced.png")
    save_image(result.binary,    out_dir / "04_binary.png")

    save_side_by_side(
        [result.grayscale, result.blurred, result.enhanced, result.binary],
        out_dir / "strip.png",
    )

    print(f"[demo] outputs written to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
