#!/usr/bin/env python3
"""Generate Research Framework.pptx — layout/structure only (no domain text).

Four widescreen slides mirroring the geometry of Research Framework.pptx:
  1. Dual panel: stacked evaluation blocks + chapter flow rows
  2. Dual panel: data grid + chapter task rows
  3. Future plan stage-1 template
  4. Future plan stage-2 template

Run:
    python scripts/build_research_framework_ppt.py
"""

from __future__ import annotations

from pathlib import Path

from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

from ppt_framework_common import (
    ACCENT_BAR,
    BORDER,
    BORDER2,
    GRAY_BORDER,
    GREEN_HDR,
    LIGHT,
    LIGHT2,
    LIGHT3,
    MID_BLUE,
    NAVY,
    NAVY2,
    PANEL_BORDER,
    SLIDE_H,
    SLIDE_W,
    WHITE,
    YELLOW,
    accent_cards_row,
    arrow,
    blank_slide,
    card,
    chapter_strip,
    flow_row,
    header_bar,
    label,
    new_presentation,
    numbered_steps,
    panel_outline,
    rect,
    save_prs,
    stacked_group,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "Research Framework.pptx"


def _slide1(prs):
    """Left: 5 stacked groups. Right: 4 chapter flow strips."""
    slide = blank_slide(prs)

    # --- left panel ---
    lx, lw = 0.15, 5.15
    label(slide, lx, 0.15, lw, 0.32, "L", size=12, bold=True, color=NAVY)

    ys = [0.55, 1.97, 3.05, 4.25, 5.43]
    hs = [1.36, 1.02, 1.14, 1.12, 1.22]
    subs = [2, 3, 3, 5, 3]
    footers = [0.40, 0.0, 0.0, 0.0, 0.0]
    for i, (y, h, n, fh) in enumerate(zip(ys, hs, subs, footers)):
        stacked_group(slide, lx, y, lw, h, str(i + 1), n, footer_h=fh)

    rect(slide, lx, 6.73, lw, 0.45, LIGHT, line=None)

    # --- right panel ---
    rx, rw = 5.45, 7.70
    label(slide, rx, 0.15, rw, 0.32, "R", size=12, bold=True, color=NAVY)

    chapter_heights = [1.18, 1.18, 1.95, 1.30]
    chapter_y = [0.58, 1.86, 3.14, 5.21]
    bx, bw, aw = rx + 0.27, 2.15, 0.33

    for ci, (oy, oh) in enumerate(zip(chapter_y, chapter_heights)):
        chapter_strip(slide, rx + 0.05, oy, rw - 0.10, oh, f"{ci + 1}", "")
        fy, fh = oy + 0.42, 0.66 if ci != 2 else 0.74
        flow_row(slide, bx, fy, bw * 3 + aw * 2, fh, 3, gap=aw, arrow_w=aw - 0.04)
        if ci == 2:
            arrow(slide, bx + bw / 2 - 0.09, fy + fh, 0.18, 0.14,
                  shape=MSO_SHAPE.DOWN_ARROW)
            card(slide, bx, fy + fh + 0.14, bw, 0.52, border=BORDER)


def _slide2(prs):
    """Left: A/B columns + C footer. Right: 4 chapter rows + outputs."""
    slide = blank_slide(prs)

    lx, ly, lw, lh = 0.18, 0.16, 5.55, 7.20
    panel_outline(slide, lx, ly, lw, lh)
    header_bar(slide, lx, ly + 0.06, lw, 0.36, "L", fill=LIGHT2, border=GRAY_BORDER)

    ax, aw = 0.34, 2.62
    bx, bw = 3.06, 2.55

    header_bar(slide, ax, 0.60, aw, 0.30, "A", fill=LIGHT2, border=GRAY_BORDER)
    header_bar(slide, bx, 0.60, bw, 0.30, "B", fill=GREEN_HDR, border=GRAY_BORDER)

    ay, ah, ag = 0.98, 0.92, 0.06
    for i in range(4):
        card(slide, ax, ay + i * (ah + ag), aw, ah, border=GRAY_BORDER)
        rect(slide, ax + 0.02, ay + i * (ah + ag), 0.82, ah, LIGHT3, line=None)
        rect(slide, ax + 0.84, ay + i * (ah + ag) + 0.08, 0.008, ah - 0.16, GRAY_BORDER)

    box_text_h = [1.02, 1.30, 1.00]
    box_y = [0.98, 2.12, 3.54]
    for j, (by, bh) in enumerate(zip(box_y, box_text_h)):
        sp = card(slide, bx, by, bw, bh, border=GRAY_BORDER,
                  dash=(j == 1), fill=YELLOW if j == 2 else WHITE)

    header_bar(slide, ax, 5.00, bx + bw - ax, 0.30, "C", fill=LIGHT2, border=GRAY_BORDER)
    card(slide, ax, 5.38, bx + bw - ax, 0.86, border=GRAY_BORDER)
    card(slide, ax, 6.32, bx + bw - ax, 0.80, border=GRAY_BORDER)

    arrow(slide, ax + aw + 0.02, 1.30, bx - (ax + aw) - 0.04, 0.20)

    rx, ry, rw, rh = 5.92, 0.16, 7.26, 7.20
    panel_outline(slide, rx, ry, rw, rh)
    header_bar(slide, rx, ry + 0.06, rw, 0.36, "R", fill=LIGHT2, border=GRAY_BORDER)

    inner_x, inner_w = rx + 0.16, rw - 0.32
    tag_w, task_w, cgap = 1.50, 2.30, 0.10
    data_w = inner_w - tag_w - task_w - 2 * cgap
    task_x = inner_x + tag_w + cgap
    data_x = task_x + task_w + cgap

    ry0, rh_row, rgap = 0.64, 1.22, 0.10
    for k in range(4):
        oy = ry0 + k * (rh_row + rgap)
        outer = card(slide, inner_x, oy, inner_w, rh_row, rounded=True,
                     border=GRAY_BORDER, dash=True)
        pad, cy, ch = 0.10, oy + 0.10, rh_row - 0.20
        card(slide, inner_x + pad, cy, tag_w, ch, fill=LIGHT3, border=GRAY_BORDER)
        label(slide, inner_x + pad, cy, tag_w, ch, str(k + 1), size=11, bold=True, color=NAVY)
        card(slide, task_x, cy, task_w, ch, border=GRAY_BORDER)
        card(slide, data_x, cy, data_w, ch, border=GRAY_BORDER)
        arrow(slide, rx - 0.30, oy + rh_row / 2 - 0.10, 0.36, 0.20)

    oy = ry0 + 4 * (rh_row + rgap)
    card(slide, inner_x, oy + 0.02, inner_w, ry + rh - (oy + 0.02) - 0.12,
         fill=LIGHT2, border=GRAY_BORDER)
    arrow(slide, rx - 0.30, oy + 0.20, 0.36, 0.20)


def _slide3(prs):
    """Stage-1 future plan layout."""
    slide = blank_slide(prs)

    rect(slide, 0.146, 0.146, 13.041, 0.354, NAVY2, line=None)
    label(slide, 0.292, 0.188, 8.541, 0.292, "1", size=14, bold=True, color=WHITE,
          align=PP_ALIGN.LEFT)
    label(slide, 6.354, 0.198, 6.562, 0.281, "1a", size=11, bold=True, color=WHITE,
          align=PP_ALIGN.RIGHT)

    rect(slide, 0.313, 0.594, 12.708, 0.323, WHITE, line=None)

    card(slide, 0.23, 1.0, 3.708, 1.854, rounded=True, border=BORDER2)
    header_bar(slide, 0.438, 1.167, 1.406, 0.302, "", fill=NAVY2, border=NAVY2, line_w=0)
    card(slide, 0.604, 1.604, 1.375, 0.292, border=BORDER2, fill=WHITE)
    label(slide, 1.948, 1.604, 0.292, 0.292, "+", size=14, bold=True, color=NAVY)
    card(slide, 2.292, 1.604, 1.302, 0.292, border=BORDER2, fill=WHITE)
    arrow(slide, 1.584, 2.042, 0.99, 0.292)
    card(slide, 0.521, 2.406, 3.125, 0.333, border=BORDER2, fill=LIGHT)

    card(slide, 4.146, 1.0, 8.958, 1.854, rounded=True, border=BORDER2)
    header_bar(slide, 4.354, 1.167, 2.5, 0.302, "", fill=NAVY2, border=NAVY2, line_w=0)
    numbered_steps(slide, 4.74, 1.52, 3.5, 0.55, 4, start=1)
    numbered_steps(slide, 9.844, 1.52, 3.0, 0.55, 4, start=1)

    rect(slide, 0.23, 3.031, 12.874, 0.854, MID_BLUE, line=None)
    label(slide, 0.479, 3.177, 2.187, 0.271, "OBJ", size=10, bold=True, color=WHITE,
          align=PP_ALIGN.LEFT)
    rect(slide, 2.761, 3.156, 9.791, 0.354, WHITE, line=BORDER2, line_w=0.5)

    label(slide, 0.292, 4.042, 3.541, 0.26, "Routes", size=10, bold=True, color=NAVY2,
          align=PP_ALIGN.LEFT)
    rect(slide, 2.479, 4.177, 10.572, 0.0, None, line=BORDER2, line_w=0.5)
    accent_cards_row(slide, 0.23, 4.417, 12.874, 1.708, 5)

    rect(slide, 0.23, 6.333, 12.874, 0.562, NAVY2, line=None)
    label(slide, 0.459, 6.479, 2.135, 0.25, "Out", size=10, bold=True, color=WHITE,
          align=PP_ALIGN.LEFT)


def _slide4(prs):
    """Stage-2 future plan layout."""
    slide = blank_slide(prs)

    rect(slide, 0.146, 0.146, 13.041, 0.354, NAVY2, line=None)
    label(slide, 0.292, 0.188, 8.541, 0.292, "2", size=14, bold=True, color=WHITE,
          align=PP_ALIGN.LEFT)
    label(slide, 6.354, 0.198, 6.562, 0.281, "2a", size=11, bold=True, color=WHITE,
          align=PP_ALIGN.RIGHT)

    rect(slide, 0.313, 0.594, 12.708, 0.323, WHITE, line=None)

    card(slide, 0.23, 1.0, 4.083, 1.875, rounded=True, border=BORDER2)
    header_bar(slide, 0.438, 1.167, 1.406, 0.302, "", fill=NAVY2, border=NAVY2, line_w=0)
    card(slide, 0.604, 1.604, 1.375, 0.292, border=BORDER2)
    label(slide, 1.948, 1.604, 0.292, 0.292, "+", size=14, bold=True, color=NAVY)
    card(slide, 2.292, 1.604, 1.302, 0.292, border=BORDER2)
    arrow(slide, 1.677, 2.042, 1.083, 0.292)
    card(slide, 0.5, 2.406, 3.458, 0.333, border=BORDER2, fill=LIGHT)

    card(slide, 4.521, 1.0, 8.583, 1.875, rounded=True, border=BORDER2)
    header_bar(slide, 4.729, 1.167, 2.708, 0.302, "", fill=NAVY2, border=NAVY2, line_w=0)

    tx = [4.948, 5.781, 6.802, 7.948]
    for i, x in enumerate(tx):
        rect(slide, x, 2.0, 0.167, 0.167, WHITE, line=NAVY, line_w=1.2,
             shape=MSO_SHAPE.OVAL)
        label(slide, x, 2.0, 0.167, 0.167, f"t{i+1}", size=6, color=NAVY)
    rect(slide, 5.031, 2.083, 3.0, 0.0, None, line=NAVY, line_w=1.0)
    numbered_steps(slide, 8.333, 1.52, 2.5, 0.55, 4, start=1)
    numbered_steps(slide, 10.958, 1.52, 2.2, 0.55, 4, start=1)

    rect(slide, 0.23, 3.042, 12.874, 0.854, MID_BLUE, line=None)
    label(slide, 0.479, 3.188, 1.979, 0.271, "OBJ", size=10, bold=True, color=WHITE,
          align=PP_ALIGN.LEFT)
    rect(slide, 2.813, 3.156, 9.375, 0.354, WHITE, line=BORDER2, line_w=0.5)

    label(slide, 0.292, 4.052, 4.479, 0.26, "Layer", size=10, bold=True, color=NAVY2,
          align=PP_ALIGN.LEFT)
    accent_cards_row(slide, 0.23, 4.417, 12.874, 1.708, 4, gap=0.11)

    rect(slide, 0.23, 6.333, 12.874, 0.562, NAVY2, line=None)
    label(slide, 0.459, 6.479, 2.135, 0.25, "Out", size=10, bold=True, color=WHITE,
          align=PP_ALIGN.LEFT)


def build(output: Path = OUTPUT) -> Path:
    prs = new_presentation()
    _slide1(prs)
    _slide2(prs)
    _slide3(prs)
    _slide4(prs)
    save_prs(prs, output)
    print(f"Saved: {output}")
    print(f"Slides: {len(prs.slides)} | canvas: {SLIDE_W}x{SLIDE_H} in")
    return output


if __name__ == "__main__":
    build()
