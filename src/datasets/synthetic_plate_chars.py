"""
synthetic_plate_chars.py
========================

Synthetic printed-character dataset for license-plate OCR.

EMNIST is useful as a public baseline, but it is handwriting.  License
plates are printed, high-contrast glyphs.  This generator renders
digits/letters with real system fonts, applies small perturbations, and
normalizes the result through the same 32x32 path used by segmentation.

Pillow is used here as a **data generator** (font rasterization), not as
an image-processing dependency for the recognition pipeline itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.detection.morphology import dilate, erode, rect
from src.segmentation import normalize_character


DEFAULT_SYNTHETIC_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_FONT_NAMES = [
    "AGENCYB.TTF",
    "AGENCYR.TTF",
    "arial.ttf",
    "arialbd.ttf",
    "ariblk.ttf",
    "ARIALN.TTF",
    "ARIALNB.TTF",
    "ARIALNBI.TTF",
    "ARIALNI.TTF",
    "bahnschrift.ttf",
    "BRLNSB.TTF",
    "BRLNSDB.TTF",
    "BRLNSR.TTF",
    "calibri.ttf",
    "calibrib.ttf",
    "calibrii.ttf",
    "calibril.ttf",
    "CascadiaMono.ttf",
    "consola.ttf",
    "consolab.ttf",
    "cour.ttf",
    "courbd.ttf",
    "impact.ttf",
    "lucon.ttf",
    "tahoma.ttf",
    "tahomabd.ttf",
    "trebuc.ttf",
    "trebucbd.ttf",
    "verdana.ttf",
    "verdanab.ttf",
]


@dataclass
class SyntheticPlateCharSet:
    """Generated character images and labels."""

    images: np.ndarray
    labels: np.ndarray
    fonts: list[str]
    samples_per_class: int


def generate_synthetic_plate_characters(
    chars: str = DEFAULT_SYNTHETIC_CHARS,
    samples_per_class: int = 600,
    seed: int = 42,
    font_paths: Sequence[str | Path] | None = None,
) -> SyntheticPlateCharSet:
    """
    Generate a balanced synthetic OCR dataset.

    Output images are binary ``uint8`` arrays shaped ``(N, 32, 32)`` with
    white glyphs on a black background, matching the segmentation stage.
    """
    if samples_per_class <= 0:
        raise ValueError(f"samples_per_class must be positive; got {samples_per_class}.")

    rng = np.random.default_rng(seed)
    fonts = _resolve_font_paths(font_paths)
    total = len(chars) * samples_per_class
    images = np.empty((total, 32, 32), dtype=np.uint8)
    labels = np.empty(total, dtype="<U1")

    idx = 0
    for ch in chars:
        for _ in range(samples_per_class):
            font_path = fonts[int(rng.integers(0, len(fonts)))]
            images[idx] = _render_augmented_char(ch, font_path, rng)
            labels[idx] = ch
            idx += 1

    order = rng.permutation(total)
    return SyntheticPlateCharSet(
        images=images[order],
        labels=labels[order],
        fonts=[str(path) for path in fonts],
        samples_per_class=samples_per_class,
    )


def _resolve_font_paths(font_paths: Sequence[str | Path] | None) -> list[Optional[Path]]:
    """Return usable TrueType/OpenType font paths."""
    candidates: list[Path] = []
    if font_paths:
        candidates.extend(Path(path) for path in font_paths)
    else:
        font_dirs = [
            Path("C:/Windows/Fonts"),
            Path("/usr/share/fonts/truetype"),
            Path("/usr/share/fonts"),
            Path("/System/Library/Fonts"),
            Path("/Library/Fonts"),
        ]
        for directory in font_dirs:
            for name in DEFAULT_FONT_NAMES:
                path = directory / name
                if path.is_file():
                    candidates.append(path)
            if candidates:
                break

    usable = []
    for path in candidates:
        if path.is_file():
            usable.append(path)
    if not usable:
        # Last-resort fallback keeps the generator usable on minimal
        # environments.  Real TrueType fonts are strongly preferred.
        return [None]
    return usable


def _render_augmented_char(
    ch: str,
    font_path: Optional[Path],
    rng: np.random.Generator,
) -> np.ndarray:
    """Render one augmented glyph and normalize it to 32x32."""
    canvas_size = 80
    image = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(image)

    font_size = int(rng.integers(42, 62))
    if font_path is None:
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(str(font_path), font_size)
    bbox = draw.textbbox((0, 0), ch, font=font, stroke_width=0)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (canvas_size - text_w) / 2.0 - bbox[0] + float(rng.normal(0, 2.8))
    y = (canvas_size - text_h) / 2.0 - bbox[1] + float(rng.normal(0, 2.8))
    stroke = int(rng.choice([0, 0, 0, 1, 1, 2]))
    draw.text((x, y), ch, fill=255, font=font, stroke_width=stroke, stroke_fill=255)

    # License-plate fonts vary wildly: some are DIN-like and condensed,
    # others are wide, heavy, or camera-stretched after perspective
    # correction.  Scaling the rendered glyph before the affine jitter
    # gives the OCR model a much broader shape prior without relying on
    # any country-specific plate format.
    scale_x = float(rng.uniform(0.72, 1.28))
    scale_y = float(rng.uniform(0.82, 1.16))
    scaled_size = (
        max(1, int(round(canvas_size * scale_x))),
        max(1, int(round(canvas_size * scale_y))),
    )
    scaled = image.resize(scaled_size, resample=Image.Resampling.BILINEAR)
    recentered = Image.new("L", (canvas_size, canvas_size), 0)
    left = (canvas_size - scaled_size[0]) // 2
    top = (canvas_size - scaled_size[1]) // 2
    recentered.paste(scaled, (left, top))
    image = recentered

    # Small geometric perturbations: rotate and shear, both common after
    # plate detection/deskewing and imperfect character segmentation.
    angle = float(rng.uniform(-10.0, 10.0))
    image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)

    shear_x = float(rng.uniform(-0.16, 0.16))
    shift_x = -shear_x * canvas_size / 2.0
    image = image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1.0, shear_x, shift_x, 0.0, 1.0, 0.0),
        resample=Image.Resampling.BILINEAR,
        fillcolor=0,
    )

    if rng.random() < 0.45:
        image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.15, 1.0))))

    arr = np.asarray(image, dtype=np.uint8)

    # Random threshold simulates Otsu jitter.  Tiny morphology changes
    # simulate stroke thickening/thinning after binarization.
    threshold = int(rng.integers(70, 150))
    binary = ((arr > threshold).astype(np.uint8)) * 255
    morph = rng.choice(["none", "none", "none", "dilate", "dilate", "erode"])
    if morph == "dilate":
        binary = dilate(binary, rect(2, 2))
    elif morph == "erode":
        binary = erode(binary, rect(2, 2))

    # Salt-and-pepper specks, kept sparse so they do not dominate HOG.
    if rng.random() < 0.45:
        noise = rng.random(binary.shape)
        binary = binary.copy()
        binary[noise < 0.0015] = 255
        binary[noise > 0.9985] = 0

    return normalize_character(binary, target_shape=(32, 32))
