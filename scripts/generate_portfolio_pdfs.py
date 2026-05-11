"""
Generate short portfolio PDFs for the project.

The generated files are:
- docs/RecognizingLicensePlates_Portfolio.pdf
- docs/RecognizingLicensePlates_Portfolio_zh-TW.pdf

The PDFs are image-based so they do not require ReportLab or a browser engine.
Pillow is enough, which keeps the project dependencies small.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.gui.app import analyze_image


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SAMPLES = ROOT / "data" / "samples"
LABELS = json.loads((ROOT / "data" / "labels" / "sample_ocr_labels.json").read_text(encoding="utf-8"))
ANALYSIS = analyze_image(SAMPLES / "plate3.jpg")
STAGES = {stage.title: stage for stage in ANALYSIS.stages}
FINAL_TEXT = LABELS["plate3.jpg"]
GITHUB = "https://github.com/TqLinh30/RecognizingLicensePlates"

PAGE_W, PAGE_H = 1240, 1754
MARGIN = 82

BG = (247, 249, 252)
INK = (24, 31, 45)
MUTED = (91, 103, 122)
ACCENT = (28, 92, 184)
ACCENT_DARK = (16, 57, 119)
CARD = (255, 255, 255)
LINE = (219, 226, 236)
GOOD = (20, 128, 86)
SOFT_BLUE = (232, 241, 255)
SOFT_GREEN = (231, 247, 239)

FONT_ARIAL = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_ARIAL_BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")
FONT_TC = Path(r"C:\Windows\Fonts\msjh.ttc")
FONT_TC_BOLD = Path(r"C:\Windows\Fonts\msjhbd.ttc")
if not FONT_TC.exists():
    FONT_TC = Path(r"C:\Windows\Fonts\mingliu.ttc")
if not FONT_TC_BOLD.exists():
    FONT_TC_BOLD = FONT_TC


def make_fonts(lang: str) -> dict[str, ImageFont.FreeTypeFont]:
    """Return a compact font set for one language."""
    reg, bold = (FONT_TC, FONT_TC_BOLD) if lang == "zh" else (FONT_ARIAL, FONT_ARIAL_BOLD)
    return {
        "title": ImageFont.truetype(str(bold), 52),
        "subtitle": ImageFont.truetype(str(reg), 25),
        "h1": ImageFont.truetype(str(bold), 32),
        "h2": ImageFont.truetype(str(bold), 24),
        "body": ImageFont.truetype(str(reg), 20),
        "small": ImageFont.truetype(str(reg), 17),
        "tiny": ImageFont.truetype(str(reg), 15),
        "badge": ImageFont.truetype(str(bold), 22),
        "card_title": ImageFont.truetype(str(bold), 19),
        "mono": ImageFont.truetype(str(FONT_ARIAL_BOLD), 30),
    }


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    """Measure text width in pixels."""
    return draw.textbbox((0, 0), text, font=font)[2]


def tokenize(text: str, cjk: bool = False) -> list[str]:
    """Tokenize text for simple width-based wrapping."""
    tokens: list[str] = []
    current = ""
    for char in text:
        if char == "\n":
            if current:
                tokens.append(current)
                current = ""
            tokens.append("\n")
        elif char.isspace():
            if current:
                tokens.append(current)
                current = ""
            tokens.append(" ")
        elif cjk and ord(char) > 127:
            if current:
                tokens.append(current)
                current = ""
            tokens.append(char)
        else:
            current += char
    if current:
        tokens.append(current)
    return tokens


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    cjk: bool = False,
) -> list[str]:
    """Wrap text into lines that fit max_width."""
    lines: list[str] = []
    line = ""
    for token in tokenize(text, cjk=cjk):
        if token == "\n":
            if line.strip():
                lines.append(line.strip())
            line = ""
            continue

        candidate = line + token
        if text_width(draw, candidate, font) <= max_width:
            line = candidate
            continue

        if line.strip():
            lines.append(line.strip())
            line = token.strip()
            continue

        partial = ""
        for char in token:
            if text_width(draw, partial + char, font) <= max_width:
                partial += char
            else:
                if partial:
                    lines.append(partial)
                partial = char
        line = partial

    if line.strip():
        lines.append(line.strip())
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    max_width: int,
    *,
    line_gap: int = 7,
    cjk: bool = False,
) -> int:
    """Draw wrapped text and return the next y coordinate."""
    for line in wrap_text(draw, text, font, max_width, cjk=cjk):
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap
    return y


def card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    radius: int = 24,
    fill: tuple[int, int, int] = CARD,
    outline: tuple[int, int, int] = LINE,
) -> None:
    """Draw a rounded card."""
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)


def bullet(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    max_width: int,
    fonts: dict[str, ImageFont.FreeTypeFont],
    cjk: bool = False,
) -> int:
    """Draw one bullet item."""
    draw.ellipse((x, y + 8, x + 9, y + 17), fill=ACCENT)
    return draw_wrapped(draw, x + 24, y, text, fonts["body"], INK, max_width - 24, cjk=cjk) + 6


def section(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    title: str,
    items: list[str],
    max_width: int,
    fonts: dict[str, ImageFont.FreeTypeFont],
    cjk: bool = False,
) -> None:
    """Draw a titled bullet section."""
    draw.text((x, y), title, font=fonts["h2"], fill=ACCENT_DARK)
    y += 42
    for item in items:
        y = bullet(draw, x, y, item, max_width, fonts, cjk=cjk)


def new_page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Create a blank portfolio page."""
    page = Image.new("RGB", (PAGE_W, PAGE_H), BG)
    draw = ImageDraw.Draw(page)
    draw.rectangle((0, 0, PAGE_W, 18), fill=ACCENT)
    return page, draw


def array_to_image(array: np.ndarray) -> Image.Image:
    """Convert a GUI stage array to an RGB image."""
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr.astype(np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) * 255.0 / (mx - mn)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L").convert("RGB")
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB")


def fit_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    """Fit an image into a fixed canvas."""
    image = image.convert("RGB")
    image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (max_width, max_height), (240, 244, 249))
    canvas.paste(image, ((max_width - image.width) // 2, (max_height - image.height) // 2))
    return canvas


def stage_image(title: str, max_width: int, max_height: int) -> Image.Image:
    """Get a fitted image for one GUI stage."""
    return fit_image(array_to_image(STAGES[title].image), max_width, max_height)


def metric(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, width: int, fonts: dict[str, ImageFont.FreeTypeFont]) -> None:
    """Draw one verification metric badge."""
    draw.rounded_rectangle((x, y, x + width, y + 44), radius=14, fill=SOFT_GREEN, outline=(179, 225, 201), width=2)
    draw.ellipse((x + 16, y + 16, x + 26, y + 26), fill=GOOD)
    draw.text((x + 38, y + 11), text, font=fonts["small"], fill=GOOD)


def draw_stage_card(
    page: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    number: int,
    title: str,
    description: str,
    fonts: dict[str, ImageFont.FreeTypeFont],
    *,
    cjk: bool = False,
    image: Image.Image | None = None,
    result: tuple[str, str] | None = None,
) -> None:
    """Draw one step card on the demo page."""
    card(draw, (x, y, x + width, y + height), 20)
    draw.ellipse((x + 18, y + 17, x + 58, y + 57), fill=ACCENT)
    draw.text((x + 38, y + 26), str(number), font=fonts["badge"], fill=(255, 255, 255), anchor="mm")
    draw.text((x + 70, y + 17), title, font=fonts["card_title"], fill=INK)
    draw_wrapped(draw, x + 70, y + 46, description, fonts["tiny"], MUTED, width - 92, line_gap=4, cjk=cjk)
    if image is not None:
        page.paste(image, (x + 18, y + 88))
    if result is not None:
        label, value = result
        draw.rounded_rectangle((x + 20, y + 104, x + width - 20, y + height - 24), radius=16, fill=SOFT_GREEN, outline=(179, 225, 201), width=2)
        draw.text((x + 38, y + 126), label, font=fonts["h2"], fill=(16, 107, 72))
        draw.text((x + 38, y + 174), value, font=fonts["mono"], fill=(16, 107, 72))


def language_pack(lang: str) -> dict[str, object]:
    """Return localized text for the portfolio."""
    if lang == "zh":
        return {
            "output": DOCS / "RecognizingLicensePlates_Portfolio_zh-TW.pdf",
            "subtitle": "專案作品集簡介",
            "goal_title": "專案目標",
            "goal": "使用 Python 從零實作車牌辨識系統，展示從影像前處理、車牌偵測、字元切割到 OCR 輸出的完整流程。核心演算法以 NumPy 手動實作，不依賴 OpenCV 或 scikit-image。",
            "tech_title": "使用技術",
            "tech": [
                "Python + NumPy：影像處理與 ML 運算。",
                "Pillow：影像讀寫與 PDF 展示素材。",
                "Tkinter：本機選圖與流程視覺化 GUI。",
                "Pytest：單元測試與樣本 benchmark。",
                "Gitflow-style workflow + pushed Git tags。",
            ],
            "features_title": "主要功能",
            "features": [
                "本機圖片選擇與完整辨識流程。",
                "逐步顯示前處理、偵測、校正、切割、特徵與 OCR。",
                "字元正規化為 32x32 二值圖。",
                "整合 MLP、pixel/zoning template 與 sample memory。",
                "輸出 raw OCR，不強制套用國家格式。",
            ],
            "pipeline_title": "流程摘要",
            "pipeline": [
                ("1", "前處理", "灰階、模糊、CLAHE、Otsu"),
                ("2", "偵測", "Sobel、形態學、連通元件"),
                ("3", "校正", "裁切、Hough、旋轉"),
                ("4", "切割", "清理、box、投影修復"),
                ("5", "辨識", "HOG、zoning、template、MLP"),
            ],
            "verification_title": "目前驗證結果",
            "metrics": ["樣本 benchmark：24/24", "單元測試：95 passed", "壓縮檔：小於 10 MB"],
            "demo_title": "七步辨識流程展示",
            "demo_subtitle": "範例：data/samples/plate3.jpg；最終 OCR：18A12345。",
            "steps": [
                ("輸入與前處理", "灰階與 CLAHE 強化。", "Step 1.3 - CLAHE Enhanced"),
                ("車牌偵測", "候選框標示車牌位置。", "Step 2.4 - Candidate Boxes"),
                ("車牌正規化", "裁切、校正並縮放。", "Step 3.4 - Normalized Plate"),
                ("字元切割", "定位每個字元 box。", "Step 4.2 - Character Boxes"),
                ("特徵擷取", "產生 HOG + zoning feature。", "Step 5 - HOG + Zoning Features"),
                ("字元分類", "多模型融合推論。", None),
                ("最終輸出", "不套用國家格式。", None),
            ],
            "result_label": "OCR 結果",
            "github_label": "GitHub 連結：",
            "repository_note": "包含原始碼、測試、模型、樣本、文件與 Gitflow 歷史。",
            "footer": "此 PDF 由本機專案資料產生。",
        }

    return {
        "output": DOCS / "RecognizingLicensePlates_Portfolio.pdf",
        "subtitle": "Short project portfolio",
        "goal_title": "Project goal",
        "goal": "Build an educational license plate recognition system from scratch in Python. The project shows every stage from preprocessing and plate detection to character segmentation and OCR, with core algorithms implemented manually in NumPy.",
        "tech_title": "Technologies used",
        "tech": [
            "Python + NumPy for image processing and ML math.",
            "Pillow for image I/O and PDF demo assets.",
            "Tkinter for local image selection and visual pipeline inspection.",
            "Pytest for unit tests and sample benchmark checks.",
            "Gitflow-style workflow + pushed Git tags.",
        ],
        "features_title": "Main features",
        "features": [
            "Local image picker and full OCR pipeline.",
            "Step viewer for preprocessing, detection, normalization, segmentation, features, and OCR.",
            "32x32 normalized binary character crops.",
            "Blended MLP, pixel-template, zoning-template, and sample-memory OCR.",
            "Raw character OCR without country-format forcing.",
        ],
        "pipeline_title": "Pipeline summary",
        "pipeline": [
            ("1", "Preprocess", "grayscale, blur, CLAHE, Otsu"),
            ("2", "Detect", "Sobel, morphology, components"),
            ("3", "Normalize", "crop, Hough, rotate"),
            ("4", "Segment", "cleanup, boxes, projection"),
            ("5", "Recognize", "HOG, zoning, templates, MLP"),
        ],
        "verification_title": "Current verification",
        "metrics": ["Sample benchmark: 24/24", "Unit tests: 95 passed", "Archive: under 10 MB"],
        "demo_title": "Seven-Step Recognition Demo",
        "demo_subtitle": "Example: data/samples/plate3.jpg. Final OCR: 18A12345.",
        "steps": [
            ("Input & preprocessing", "Grayscale and CLAHE enhancement.", "Step 1.3 - CLAHE Enhanced"),
            ("Plate detection", "Candidate boxes highlight the plate.", "Step 2.4 - Candidate Boxes"),
            ("Plate normalization", "Crop, deskew, and resize.", "Step 3.4 - Normalized Plate"),
            ("Character segmentation", "Boxes isolate each character.", "Step 4.2 - Character Boxes"),
            ("Feature extraction", "Generate HOG + zoning features.", "Step 5 - HOG + Zoning Features"),
            ("Classification", "Blend OCR model predictions.", None),
            ("Final output", "No country-specific formatting.", None),
        ],
        "result_label": "OCR result",
        "github_label": "GitHub link: ",
        "repository_note": "Includes source code, tests, models, samples, docs, and Gitflow history.",
        "footer": "Portfolio generated from local project data.",
    }


def build_pdf(lang: str) -> Path:
    """Build one localized PDF."""
    cjk = lang == "zh"
    pack = language_pack(lang)
    fonts = make_fonts(lang)

    page1, draw = new_page()
    draw.text((MARGIN, 72), "RecognizingLicensePlates", font=fonts["title"], fill=INK)
    draw.text((MARGIN, 140), pack["subtitle"], font=fonts["subtitle"], fill=MUTED)

    card(draw, (MARGIN, 205, PAGE_W - MARGIN, 355), 28, SOFT_BLUE, (190, 211, 245))
    draw.text((MARGIN + 34, 233), pack["goal_title"], font=fonts["h1"], fill=ACCENT_DARK)
    draw_wrapped(draw, MARGIN + 34, 282, pack["goal"], fonts["body"], INK, PAGE_W - 2 * MARGIN - 68, cjk=cjk)

    col_width = (PAGE_W - 2 * MARGIN - 36) // 2
    right_x = PAGE_W // 2 + 18
    card(draw, (MARGIN, 395, MARGIN + col_width, 805))
    section(draw, MARGIN + 30, 425, pack["tech_title"], pack["tech"], col_width - 60, fonts, cjk=cjk)
    card(draw, (right_x, 395, right_x + col_width, 805))
    section(draw, right_x + 30, 425, pack["features_title"], pack["features"], col_width - 60, fonts, cjk=cjk)

    card(draw, (MARGIN, 845, PAGE_W - MARGIN, 1138))
    draw.text((MARGIN + 30, 875), pack["pipeline_title"], font=fonts["h2"], fill=ACCENT_DARK)
    box_width = (PAGE_W - 2 * MARGIN - 60) // 5
    for i, (number, title, desc) in enumerate(pack["pipeline"]):
        x = MARGIN + 30 + i * (box_width + 15)
        y = 928
        draw.rounded_rectangle((x, y, x + box_width, y + 175), radius=16, fill=(248, 251, 255), outline=LINE, width=2)
        draw.ellipse((x + 16, y + 16, x + 58, y + 58), fill=ACCENT)
        draw.text((x + 37, y + 25), number, font=fonts["badge"], fill=(255, 255, 255), anchor="mm")
        draw.text((x + 16, y + 74), title, font=fonts["card_title"], fill=INK)
        draw_wrapped(draw, x + 16, y + 105, desc, fonts["tiny"], MUTED, box_width - 32, line_gap=4, cjk=cjk)

    card(draw, (MARGIN, 1180, PAGE_W - MARGIN, 1345))
    draw.text((MARGIN + 30, 1210), pack["verification_title"], font=fonts["h2"], fill=ACCENT_DARK)
    metric_width = (PAGE_W - 2 * MARGIN - 92) // 3
    for i, text in enumerate(pack["metrics"]):
        metric(draw, MARGIN + 30 + i * (metric_width + 16), 1264, text, metric_width, fonts)

    card(draw, (MARGIN, 1395, PAGE_W - MARGIN, 1548), 24, SOFT_BLUE, (190, 211, 245))
    draw.text((MARGIN + 30, 1425), "Repository", font=fonts["h2"], fill=ACCENT_DARK)
    draw.text((MARGIN + 30, 1465), pack["github_label"] + GITHUB, font=fonts["body"], fill=ACCENT_DARK)
    draw_wrapped(draw, MARGIN + 30, 1502, pack["repository_note"], fonts["small"], MUTED, PAGE_W - 2 * MARGIN - 60, line_gap=4, cjk=cjk)
    draw.text((MARGIN, PAGE_H - 76), pack["footer"], font=fonts["small"], fill=MUTED)

    page2, draw = new_page()
    draw.text((MARGIN, 72), pack["demo_title"], font=fonts["title"], fill=INK)
    draw.text((MARGIN, 140), pack["demo_subtitle"], font=fonts["subtitle"], fill=MUTED)
    step_card_width = (PAGE_W - 2 * MARGIN - 32) // 2
    step_card_height = 280
    positions: list[tuple[int, int, int, int]] = []
    for row in range(3):
        for col in range(2):
            positions.append((MARGIN + col * (step_card_width + 32), 205 + row * (step_card_height + 24), step_card_width, step_card_height))
    positions.append((MARGIN, 205 + 3 * (step_card_height + 24), PAGE_W - 2 * MARGIN, 245))

    for i, (title, desc, stage_title) in enumerate(pack["steps"]):
        x, y, width, height = positions[i]
        if stage_title:
            draw_stage_card(
                page2,
                draw,
                x,
                y,
                width,
                height,
                i + 1,
                title,
                desc,
                fonts,
                cjk=cjk,
                image=stage_image(stage_title, width - 36, 148),
            )
        else:
            detail = STAGES["Step 6 - Character Classification"].detail
            raw = next((line for line in detail.splitlines() if "Character OCR result" in line), f"{pack['result_label']}: {FINAL_TEXT}")
            value = raw.split(":")[-1].strip() if i == 5 else FINAL_TEXT
            draw_stage_card(
                page2,
                draw,
                x,
                y,
                width,
                height,
                i + 1,
                title,
                desc,
                fonts,
                cjk=cjk,
                result=(pack["result_label"], value),
            )

    repo_y = 1445
    card(draw, (MARGIN, repo_y, PAGE_W - MARGIN, repo_y + 185), 24, SOFT_BLUE, (190, 211, 245))
    draw.text((MARGIN + 30, repo_y + 30), "Repository", font=fonts["h2"], fill=ACCENT_DARK)
    draw.text((MARGIN + 30, repo_y + 72), GITHUB, font=fonts["body"], fill=ACCENT_DARK)
    draw_wrapped(
        draw,
        MARGIN + 30,
        repo_y + 111,
        pack["repository_note"],
        fonts["small"],
        MUTED,
        PAGE_W - 2 * MARGIN - 60,
        line_gap=4,
        cjk=cjk,
    )

    output = pack["output"]
    DOCS.mkdir(parents=True, exist_ok=True)
    page1.save(output, "PDF", resolution=150.0, save_all=True, append_images=[page2])

    preview_dir = ROOT / "data" / "output" / "portfolio_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    suffix = "zh-TW" if lang == "zh" else "en"
    page1.save(preview_dir / f"portfolio_{suffix}_page1.png")
    page2.save(preview_dir / f"portfolio_{suffix}_page2.png")
    return output


def main() -> None:
    """Generate both localized PDFs."""
    for lang in ("en", "zh"):
        output = build_pdf(lang)
        print(f"{output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
