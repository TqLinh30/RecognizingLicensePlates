"""
tests/test_gui_pipeline.py
==========================

Smoke test for the GUI analysis path without opening a Tkinter window.
"""

from __future__ import annotations

import numpy as np

from src.gui.app import analyze_image
from src.utils.image_io import save_image


def test_gui_analyze_image_returns_pipeline_stages(tmp_path):
    img = np.full((200, 400, 3), 185, dtype=np.uint8)
    img[50:155, 35:365] = [150, 150, 155]

    x, y, w, h = 90, 88, 190, 52
    img[y : y + h, x : x + w] = 245

    char_w = 14
    char_h = 34
    spacing = (w - 30 - 7 * char_w) // 8
    cx = x + 15 + spacing
    for _ in range(7):
        img[y + 9 : y + 9 + char_h, cx : cx + char_w] = 25
        cx += char_w + spacing

    path = tmp_path / "synthetic_plate.png"
    save_image(img, path)

    result = analyze_image(path)

    assert result.path == path
    assert "Step 2: 1 plate candidate" in result.summary
    assert "Step 4: 7 character candidate" in result.summary
    assert "Feature matrix shape: (7, 340)" in result.summary
    assert len(result.stages) >= 18
