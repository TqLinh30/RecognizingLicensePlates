"""
app.py
======

Tkinter desktop GUI for the license-plate recognition pipeline.

The command-line demos are useful for development, but a visual pipeline
is much friendlier when testing real photos.  This app lets the user pick
an image from disk, runs the implemented stages, and displays every
intermediate result:

1. preprocessing,
2. plate detection,
3. plate normalization,
4. character segmentation,
5. feature extraction summary,
6. classifier prediction when ``data/models/emnist_mlp.npz`` exists.

If the model file is missing, the GUI shows the exact training command
needed to download EMNIST and build the starter OCR model.
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.classifiers import load_mlp_model
from src.detection import detect_plate, draw_candidates
from src.features import extract_batch_features, feature_length
from src.normalization import normalize_plate
from src.preprocessing import preprocess
from src.recognition import postprocess_predictions
from src.segmentation import draw_character_boxes, segment_characters
from src.utils.image_io import load_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "emnist_mlp.npz"
TRAIN_COMMAND = "python -m scripts.train_emnist_mlp --download"


# ---------------------------------------------------------------------------
# Data passed from worker thread to Tkinter thread
# ---------------------------------------------------------------------------

@dataclass
class StageOutput:
    """One visual or textual stage rendered by the GUI."""

    title: str
    detail: str
    image: Optional[np.ndarray] = None


@dataclass
class AnalysisOutput:
    """Complete GUI-ready analysis result."""

    path: Path
    summary: str
    stages: list[StageOutput]


# ---------------------------------------------------------------------------
# Scrollable container
# ---------------------------------------------------------------------------

class ScrollableFrame(ttk.Frame):
    """A vertical scroll area that hosts ordinary ttk widgets."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas)

        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_content_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------

class LicensePlateApp:
    """Desktop app for visual ALPR pipeline inspection."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Recognizing License Plates - Pipeline Viewer")
        self.root.geometry("1280x850")
        self.root.minsize(980, 680)

        self.photos: list[ImageTk.PhotoImage] = []
        self._worker: Optional[threading.Thread] = None

        self._configure_style()
        self._build_layout()

    def _configure_style(self) -> None:
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10))
        style.configure("Stage.TLabelframe.Label", font=("Segoe UI", 11, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=16)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        title = ttk.Label(header, text="License Plate Pipeline Viewer", style="Title.TLabel")
        title.grid(row=0, column=0, sticky="w")

        controls = ttk.Frame(header)
        controls.grid(row=0, column=1, sticky="e")

        self.choose_button = ttk.Button(
            controls,
            text="Choose Image",
            style="Accent.TButton",
            command=self.choose_image,
        )
        self.choose_button.grid(row=0, column=0, padx=(0, 8))

        self.clear_button = ttk.Button(controls, text="Clear", command=self.clear)
        self.clear_button.grid(row=0, column=1)

        subtitle = ttk.Label(
            outer,
            text=(
                "Select a JPG/PNG/BMP image. The app will show preprocessing, "
                "detection, normalization, segmentation, and feature extraction."
            ),
            style="Subtitle.TLabel",
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(4, 12))

        body = ttk.PanedWindow(outer, orient="horizontal")
        body.grid(row=2, column=0, sticky="nsew")
        outer.grid_rowconfigure(2, weight=1)
        outer.grid_columnconfigure(0, weight=1)

        left = ttk.Frame(body, padding=(0, 0, 12, 0))
        right = ttk.Frame(body)
        body.add(left, weight=1)
        body.add(right, weight=4)

        ttk.Label(left, text="Summary", style="Stage.TLabelframe.Label").grid(
            row=0,
            column=0,
            sticky="w",
        )
        self.summary_text = tk.Text(
            left,
            height=30,
            width=38,
            wrap="word",
            borderwidth=1,
            relief="solid",
            font=("Consolas", 10),
        )
        self.summary_text.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Ready.")
        status = ttk.Label(left, textvariable=self.status_var)
        status.grid(row=2, column=0, sticky="ew", pady=(10, 0))

        self.scroll = ScrollableFrame(right)
        self.scroll.grid(row=0, column=0, sticky="nsew")
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self._set_summary(
            "Ready.\n\n"
            "Click 'Choose Image' to select a photo from your computer.\n\n"
            "If data/models/emnist_mlp.npz exists, the app will also run OCR. "
            f"If it is missing, train it with:\n{TRAIN_COMMAND}"
        )

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._start_analysis(Path(path))

    def clear(self) -> None:
        self.photos.clear()
        for child in self.scroll.content.winfo_children():
            child.destroy()
        self._set_summary("Ready.")
        self.status_var.set("Ready.")

    def _start_analysis(self, path: Path) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("Still processing", "Please wait for the current image to finish.")
            return

        self.clear()
        self.choose_button.configure(state="disabled")
        self.status_var.set(f"Processing: {path.name}")
        self._set_summary(f"Processing image:\n{path}\n\nPlease wait...")

        self._worker = threading.Thread(
            target=self._worker_analyze,
            args=(path,),
            daemon=True,
        )
        self._worker.start()

    def _worker_analyze(self, path: Path) -> None:
        try:
            output = analyze_image(path)
        except Exception as exc:  # pragma: no cover - shown in GUI
            trace = traceback.format_exc()
            self.root.after(0, lambda: self._show_error(exc, trace))
            return
        self.root.after(0, lambda: self._render_output(output))

    def _show_error(self, exc: Exception, trace: str) -> None:
        self.choose_button.configure(state="normal")
        self.status_var.set("Error.")
        self._set_summary(f"Error:\n{exc}\n\nTraceback:\n{trace}")
        messagebox.showerror("Analysis failed", str(exc))

    def _render_output(self, output: AnalysisOutput) -> None:
        self.choose_button.configure(state="normal")
        self.status_var.set(f"Done: {output.path.name}")
        self._set_summary(output.summary)

        for child in self.scroll.content.winfo_children():
            child.destroy()
        self.photos.clear()

        for idx, stage in enumerate(output.stages):
            self._add_stage_card(idx, stage)

    def _add_stage_card(self, idx: int, stage: StageOutput) -> None:
        card = ttk.LabelFrame(
            self.scroll.content,
            text=stage.title,
            style="Stage.TLabelframe",
            padding=10,
        )
        card.grid(row=idx, column=0, sticky="ew", padx=(0, 10), pady=(0, 12))
        self.scroll.content.grid_columnconfigure(0, weight=1)

        if stage.image is not None:
            photo = array_to_photo(stage.image, max_size=(820, 360))
            self.photos.append(photo)
            image_label = ttk.Label(card, image=photo)
            image_label.grid(row=0, column=0, sticky="w")
            detail_row = 1
        else:
            detail_row = 0

        detail = ttk.Label(card, text=stage.detail, justify="left", wraplength=820)
        detail.grid(row=detail_row, column=0, sticky="ew", pady=(8, 0))

    def _set_summary(self, text: str) -> None:
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", text)
        self.summary_text.configure(state="disabled")


# ---------------------------------------------------------------------------
# Pipeline analysis
# ---------------------------------------------------------------------------

def analyze_image(path: Path) -> AnalysisOutput:
    """Run the implemented ALPR stages and package GUI-friendly results."""
    image = load_image(path)
    stages: list[StageOutput] = [
        StageOutput(
            "Input Image",
            f"File: {path}\nShape: {image.shape}\nDtype: {image.dtype}",
            image,
        )
    ]

    # Step 1 - preprocessing.
    pre = preprocess(image)
    stages.extend(
        [
            StageOutput(
                "Step 1.1 - Grayscale",
                "RGB converted to luminance grayscale using the project's BT.601 implementation.",
                pre.grayscale,
            ),
            StageOutput(
                "Step 1.2 - Gaussian Blur",
                "Noise reduced with a separable Gaussian blur.",
                pre.blurred,
            ),
            StageOutput(
                "Step 1.3 - CLAHE Enhanced",
                "Local contrast enhanced with CLAHE before detection.",
                pre.enhanced,
            ),
            StageOutput(
                "Step 1.4 - Otsu Binary",
                f"Otsu threshold = {pre.otsu_threshold_value}. Dark glyph-like pixels become foreground.",
                pre.binary,
            ),
        ]
    )

    # Step 2 - detection.
    det = detect_plate(pre.enhanced)
    annotated = draw_candidates(image, det.candidates)
    candidate_lines = [
        (
            f"#{i}: box=({c.x}, {c.y}, {c.width}, {c.height}), "
            f"score={c.score:.3f}, aspect={c.aspect_ratio:.2f}, "
            f"fill={c.fill_ratio:.2f}, gradient_density={c.gradient_density:.2f}"
        )
        for i, c in enumerate(det.candidates, 1)
    ]
    candidate_text = "\n".join(candidate_lines) if candidate_lines else "No candidate passed the filters."
    stages.extend(
        [
            StageOutput(
                "Step 2.1 - Sobel-X Gradient",
                "Vertical stroke density cue used to locate plate-like regions.",
                det.gradient,
            ),
            StageOutput(
                "Step 2.2 - Thresholded Gradient",
                "Strong Sobel-X responses converted to binary foreground.",
                det.binary,
            ),
            StageOutput(
                "Step 2.3 - Morphological Closing",
                "Horizontal closing merges per-character strokes into plate-shaped blobs.",
                det.closed,
            ),
            StageOutput(
                "Step 2.4 - Candidate Boxes",
                f"Candidates found: {len(det.candidates)}\n{candidate_text}",
                annotated,
            ),
        ]
    )

    if not det.candidates:
        summary = _summary_text(path, image, len(det.candidates), None, None, None)
        stages.append(
            StageOutput(
                "Steps 3-7 - Skipped",
                "No plate candidate was detected, so cropping, segmentation, features, and OCR were skipped.",
            )
        )
        return AnalysisOutput(path=path, summary=summary, stages=stages)

    # Step 3 - normalization.
    best = det.candidates[0]
    norm = normalize_plate(pre.enhanced, best)
    stages.extend(
        [
            StageOutput(
                "Step 3.1 - Cropped Plate",
                f"Best candidate box: {best.as_box()} with margin. Crop shape: {norm.cropped.shape}",
                norm.cropped,
            ),
            StageOutput(
                "Step 3.2 - Hough Edge Map",
                "Sobel magnitude edge map used for near-horizontal Hough voting.",
                norm.edge_image,
            ),
            StageOutput(
                "Step 3.3 - Deskewed Plate",
                f"Estimated skew angle: {norm.angle_degrees:.2f} degrees. Rotated by {-norm.angle_degrees:.2f} degrees.",
                norm.deskewed,
            ),
            StageOutput(
                "Step 3.4 - Normalized Plate",
                f"Resized to canonical shape: {norm.normalized.shape}.",
                norm.normalized,
            ),
        ]
    )

    # Step 4 - segmentation.
    seg = segment_characters(norm.normalized)
    boxes = draw_character_boxes(norm.normalized, seg.characters)
    char_strip = make_character_strip([char.normalized for char in seg.characters])
    char_lines = [
        f"#{i}: box={char.as_box()}, row={char.row_index}"
        for i, char in enumerate(seg.characters, 1)
    ]
    stages.extend(
        [
            StageOutput(
                "Step 4.1 - Character Binary Cleanup",
                "Otsu thresholding and morphology cleanup before connected components.",
                seg.cleaned,
            ),
            StageOutput(
                "Step 4.2 - Character Boxes",
                f"Characters found: {len(seg.characters)}\n" + ("\n".join(char_lines) if char_lines else "No character-like component found."),
                boxes,
            ),
        ]
    )
    if char_strip is not None:
        stages.append(
            StageOutput(
                "Step 4.3 - Normalized Character Crops",
                "Each detected character is centered on a 32x32 binary canvas.",
                char_strip,
            )
        )

    feature_summary = None
    features = None
    if seg.characters:
        # Step 5 - feature extraction.
        features = extract_batch_features(char.normalized for char in seg.characters)
        feature_preview = make_feature_preview(features[0])
        feature_summary = (
            f"Feature matrix shape: {features.shape}\n"
            f"Expected single-character feature length: {feature_length()}\n"
            f"First vector: min={features[0].min():.4f}, max={features[0].max():.4f}, "
            f"mean={features[0].mean():.4f}, std={features[0].std():.4f}"
        )
        stages.append(
            StageOutput(
                "Step 5 - HOG + Zoning Features",
                feature_summary,
                feature_preview,
            )
        )
    else:
        stages.append(
            StageOutput(
                "Step 5 - Feature Extraction",
                "Skipped because no character crops were found.",
            )
        )

    # Step 6/7 - classifier prediction when a trained model exists.
    final_text = None
    classifier_detail = _classifier_missing_message()
    if features is not None:
        final_text, classifier_detail = _classify_with_default_model(features)
    stages.append(
        StageOutput(
            "Step 6-7 - Classification & Post-processing",
            classifier_detail,
        )
    )

    summary = _summary_text(
        path,
        image,
        len(det.candidates),
        norm.angle_degrees,
        len(seg.characters),
        feature_summary,
        final_text,
    )
    return AnalysisOutput(path=path, summary=summary, stages=stages)


def _classify_with_default_model(features: np.ndarray) -> tuple[Optional[str], str]:
    """
    Load the default EMNIST MLP model and classify segmented characters.

    Returns ``(final_text, detail)``.  If the model is missing or
    incompatible, ``final_text`` is ``None`` and ``detail`` explains how
    to fix the situation without failing the whole GUI analysis.
    """
    if not DEFAULT_MODEL_PATH.is_file():
        return None, _classifier_missing_message()

    try:
        model = load_mlp_model(DEFAULT_MODEL_PATH)
        proba = model.predict_proba(features)
        classes = np.asarray(model.classes_).astype(str)
        indices = np.argmax(proba, axis=1)
        labels = classes[indices].tolist()
        confidences = np.max(proba, axis=1).astype(float).tolist()
        post = postprocess_predictions(labels, confidences)
    except Exception as exc:
        return (
            None,
            (
                f"Found model: {DEFAULT_MODEL_PATH}\n"
                f"But prediction failed: {exc}\n\n"
                "Re-train the model with:\n"
                f"{TRAIN_COMMAND}"
            ),
        )

    lines = [
        f"Model: {DEFAULT_MODEL_PATH}",
        f"Raw classifier output: {post.raw_text}",
        f"Corrected compact text: {post.corrected_text}",
        f"Formatted result: {post.formatted_text}",
        f"Average confidence: {post.average_confidence:.3f}",
    ]
    if post.low_confidence_indices:
        lines.append(f"Low-confidence character indices: {post.low_confidence_indices}")
    return post.formatted_text, "\n".join(lines)


def _classifier_missing_message() -> str:
    """Message shown when no trained OCR model is available."""
    return (
        f"No trained OCR model found at:\n{DEFAULT_MODEL_PATH}\n\n"
        "Train a starter EMNIST model with:\n"
        f"{TRAIN_COMMAND}\n\n"
        "After training, run the GUI again. It will load the model automatically."
    )


def _summary_text(
    path: Path,
    image: np.ndarray,
    candidate_count: int,
    angle: Optional[float],
    char_count: Optional[int],
    feature_summary: Optional[str],
    final_text: Optional[str] = None,
) -> str:
    """Build the left-panel text summary."""
    lines = [
        "Analysis Summary",
        "================",
        f"Image: {path.name}",
        f"Shape: {image.shape}",
        "",
        f"Step 1: preprocessing completed",
        f"Step 2: {candidate_count} plate candidate(s)",
    ]
    if angle is None:
        lines.extend(
            [
                "Step 3: skipped",
                "Step 4: skipped",
                "Step 5: skipped",
                "Step 6-7: skipped",
            ]
        )
    else:
        lines.extend(
            [
                f"Step 3: normalized, skew angle = {angle:.2f} deg",
                f"Step 4: {char_count or 0} character candidate(s)",
                "Step 5: features extracted" if feature_summary else "Step 5: skipped",
                f"Step 6-7: OCR result = {final_text}" if final_text else "Step 6-7: waiting for trained classifier model",
            ]
        )
    if feature_summary:
        lines.extend(["", feature_summary])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Image rendering helpers
# ---------------------------------------------------------------------------

def array_to_photo(array: np.ndarray, max_size: tuple[int, int]) -> ImageTk.PhotoImage:
    """Convert a NumPy image into a resized Tk PhotoImage."""
    img = array_to_pil(array)
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(img)


def array_to_pil(array: np.ndarray) -> Image.Image:
    """Convert a uint8 NumPy image to a Pillow image for GUI display."""
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    if arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr, mode="RGB")
    if arr.ndim == 3 and arr.shape[2] == 4:
        return Image.fromarray(arr, mode="RGBA").convert("RGB")
    raise ValueError(f"Unsupported display array shape: {arr.shape}")


def make_character_strip(chars: list[np.ndarray], scale: int = 4, gap: int = 8) -> Optional[np.ndarray]:
    """Create a horizontal preview strip of normalized character crops."""
    if not chars:
        return None
    scaled: list[np.ndarray] = []
    for char in chars:
        rgb = np.stack([char, char, char], axis=-1)
        scaled.append(np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1))

    h = max(img.shape[0] for img in scaled)
    total_w = sum(img.shape[1] for img in scaled) + gap * (len(scaled) - 1)
    canvas = np.full((h, total_w, 3), 245, dtype=np.uint8)
    x = 0
    for img in scaled:
        y = (h - img.shape[0]) // 2
        canvas[y : y + img.shape[0], x : x + img.shape[1]] = img
        x += img.shape[1] + gap
    return canvas


def make_feature_preview(vector: np.ndarray, width: int = 820, height: int = 140) -> np.ndarray:
    """Render a compact bar preview of a feature vector."""
    values = np.abs(np.asarray(vector, dtype=np.float32))
    if values.size == 0:
        return np.full((height, width, 3), 255, dtype=np.uint8)
    if values.max() > 0:
        values = values / values.max()

    columns = min(width, values.size)
    if values.size != columns:
        edges = np.linspace(0, values.size, columns + 1, dtype=np.int32)
        reduced = np.array(
            [values[edges[i] : max(edges[i + 1], edges[i] + 1)].mean() for i in range(columns)],
            dtype=np.float32,
        )
    else:
        reduced = values

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    baseline = height - 18
    canvas[baseline : baseline + 1, :, :] = 210
    left = (width - columns) // 2
    for i, value in enumerate(reduced):
        bar_h = int(round(value * (height - 28)))
        x = left + i
        canvas[baseline - bar_h : baseline, x, 0] = 40
        canvas[baseline - bar_h : baseline, x, 1] = 95
        canvas[baseline - bar_h : baseline, x, 2] = 170
    return canvas


def main() -> None:
    """Start the Tkinter application."""
    root = tk.Tk()
    app = LicensePlateApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
