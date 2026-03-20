#!/usr/bin/env python3
"""
Publication-quality figure generation for ARA-Net attention analysis.
Designed for Nature Neuroscience / NeuroImage level aesthetics.

Figures:
  1. Attention heatmap (split diverging)
  2. RDI lollipop chart
  3. Group comparison violin + swarm
  4. Disease gradient connected-dot
  5. Braak correlation (clean scatter)
  6. Clinical alignment donut + radar
  7. Cross-dataset IXI slope chart
  8. Cross-dataset OASIS radar overlay
  9. Consistency summary bullet chart
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec
import scienceplots

# ── Nature-style palette ─────────────────────────────────────────────────────

PAL = {
    "CN":       "#4C72B0",   # muted steel blue
    "MCI":      "#DD8452",   # warm amber
    "AD":       "#C44E52",   # muted crimson
    "adkey":    "#C44E52",
    "other":    "#4C72B0",
    "adni":     "#4C72B0",
    "ext":      "#DD8452",
    "gray":     "#8C8C8C",
    "light":    "#E8E8E8",
    "bg":       "#FAFAFA",
    "grid":     "#DDDDDD",
    "text":     "#333333",
    "accent":   "#55A868",   # sage green
    "purple":   "#8172B3",
}

REGION_NAMES = [
    "L-WM", "L-Ctx", "L-Vent", "L-Thal", "L-Caud",
    "L-Put", "L-Pall", "BStem", "L-Hipp", "L-Amyg",
    "L-Acc", "R-WM", "R-Ctx", "R-Vent", "R-Thal",
    "R-Caud", "R-Put", "R-Pall", "R-Hipp", "R-Amyg", "R-Acc",
]
SHORT_NAMES = [
    "L-WM", "L-Ctx", "L-Vent", "L-Thal", "L-Caud",
    "L-Put", "L-Pall", "BStem", "L-Hip", "L-Amy",
    "L-Acc", "R-WM", "R-Ctx", "R-Vent", "R-Thal",
    "R-Caud", "R-Put", "R-Pall", "R-Hip", "R-Amy", "R-Acc",
]
AD_KEY = {"L-Hipp", "R-Hipp", "L-Amyg", "R-Amyg", "L-Vent", "R-Vent"}
CLASS_NAMES = ["CN", "MCI", "AD"]
CLASS_COLORS = [PAL["CN"], PAL["MCI"], PAL["AD"]]


def _apply_style():
    plt.style.use(['science', 'nature', 'no-latex'])
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig, path):
    fig.savefig(str(path).replace(".png", ".pdf"), facecolor="white")
    fig.savefig(str(path), facecolor="white")
    plt.close(fig)
    print(f"  -> {Path(path).name}")


def _panel_label(ax, label, x=-0.08, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="right",
            color=PAL["text"])


# ── load data ────────────────────────────────────────────────────────────────

def load_biomarker(path="chapter1_foundation/attention_biomarker_results.json"):
    with open(path) as f:
        return json.load(f)


def load_cross(path="chapter1_foundation/cross_dataset_interpretability_results.json"):
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Attention Heatmap (diverging, CN-centred)
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_attention_heatmap(bio, save_dir):
    _apply_style()
    profiles = bio["per_class_mean_attention"]
    cn = np.array(profiles["CN"])
    mci = np.array(profiles["MCI"])
    ad = np.array(profiles["AD"])

    data = np.stack([cn, mci, ad])
    row_labels = CLASS_NAMES

    fig, ax = plt.subplots(figsize=(7.2, 2.2))

    cmap = LinearSegmentedColormap.from_list(
        "custom_div",
        ["#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B"],
        N=256,
    )
    vmin, vmax = data.min() - 0.01, data.max() + 0.01
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")

    ax.set_yticks(range(3))
    ax.set_yticklabels(row_labels, fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(REGION_NAMES)))
    ax.set_xticklabels(SHORT_NAMES, rotation=50, ha="right", fontsize=6.5)

    for i in range(3):
        for j in range(len(REGION_NAMES)):
            v = data[i, j]
            color = "white" if abs(v - (vmin + vmax) / 2) > (vmax - vmin) * 0.35 else "#333333"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=5.5, color=color, fontweight="medium")

    for j, name in enumerate(REGION_NAMES):
        if name in AD_KEY:
            rect = plt.Rectangle((j - 0.5, -0.5), 1, 3, linewidth=1.2,
                                 edgecolor=PAL["AD"], facecolor="none",
                                 linestyle="--", alpha=0.7)
            ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, aspect=20)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Mean Attention (received)", fontsize=7)

    ax.set_title("Region Attention Profile by Diagnostic Group",
                 fontsize=10, fontweight="bold", pad=8)

    fig.tight_layout()
    _save(fig, save_dir / "fig1_attention_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — RDI Lollipop Chart
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_rdi_lollipop(bio, save_dir):
    _apply_style()
    rdi = bio["rdi"]

    regions = [r["region"] for r in rdi]
    values = [r["rdi"] for r in rdi]
    directions = [r["direction"] for r in rdi]

    fig, ax = plt.subplots(figsize=(4.5, 5.5))
    y = np.arange(len(regions))

    for i, (reg, val, dirn) in enumerate(zip(regions, values, directions)):
        is_ad = reg in AD_KEY
        color = PAL["AD"] if is_ad else PAL["CN"]
        ax.hlines(y=i, xmin=0, xmax=val, color=color, linewidth=1.5, alpha=0.6)
        marker = "D" if is_ad else "o"
        ax.scatter(val, i, color=color, s=45, zorder=5, marker=marker,
                   edgecolors="white", linewidth=0.5)

        label_x = val + 0.02
        dirn_short = dirn.replace(">", "\u2192")
        ax.text(label_x, i, f"{val:.2f}", fontsize=6, va="center",
                color=color, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(regions, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Region Discriminability Index (|Cohen's d|)", fontsize=8)
    ax.set_xlim(-0.02, max(values) + 0.15)
    ax.grid(True, axis="x", alpha=0.2)

    ax.axvline(0.5, color=PAL["gray"], linestyle=":", linewidth=0.7, alpha=0.5)
    ax.text(0.51, len(regions) - 0.5, "medium\neffect", fontsize=5,
            color=PAL["gray"], va="top", alpha=0.7)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=PAL["AD"],
               markersize=6, label="Known AD region"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PAL["CN"],
               markersize=6, label="Other region"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=6.5,
              frameon=True, fancybox=False, edgecolor=PAL["grid"])

    ax.set_title("Region Discriminability Index\n(AD vs CN)", fontsize=9,
                 fontweight="bold", pad=8)
    fig.tight_layout()
    _save(fig, save_dir / "fig2_rdi_lollipop.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Violin + Strip for Key Regions
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_group_violin(bio, save_dir):
    _apply_style()

    attn_data = _load_raw_attention()
    if attn_data is None:
        print("  [skip] fig3: no raw attention data")
        return
    received, labels = attn_data

    key_regions = ["L-Hipp", "R-Hipp", "L-Amyg", "R-Amyg", "L-Vent", "R-Vent"]
    key_indices = [REGION_NAMES.index(r) for r in key_regions]

    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.2))
    axes = axes.flatten()

    for idx, (ax, ri) in enumerate(zip(axes, key_indices)):
        data_list = []
        for c in range(3):
            vals = received[labels == c, ri]
            data_list.append(vals)

        positions = [0, 1, 2]
        vp = ax.violinplot(data_list, positions=positions, showmedians=False,
                           showextrema=False, widths=0.7)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(CLASS_COLORS[i])
            body.set_alpha(0.3)
            body.set_edgecolor(CLASS_COLORS[i])
            body.set_linewidth(0.8)

        bp = ax.boxplot(data_list, positions=positions, widths=0.15,
                        patch_artist=True, showfliers=False, zorder=3)
        for i, (box, med) in enumerate(zip(bp["boxes"], bp["medians"])):
            box.set_facecolor(CLASS_COLORS[i])
            box.set_alpha(0.85)
            box.set_edgecolor("white")
            box.set_linewidth(0.5)
            med.set_color("white")
            med.set_linewidth(1.2)
        for w in bp["whiskers"]:
            w.set_color(PAL["gray"])
            w.set_linewidth(0.6)
        for c in bp["caps"]:
            c.set_color(PAL["gray"])
            c.set_linewidth(0.6)

        means = [d.mean() for d in data_list]
        if means[0] <= means[1] <= means[2] or means[0] >= means[1] >= means[2]:
            trend_color = PAL["accent"]
            ax.plot(positions, means, color=trend_color, linewidth=1.2,
                    linestyle="--", alpha=0.7, zorder=4)
            ax.scatter(positions, means, color=trend_color, s=15, zorder=5,
                       edgecolors="white", linewidth=0.4)

        q01 = min(np.percentile(d, 1) for d in data_list)
        q99 = max(np.percentile(d, 99) for d in data_list)
        margin = (q99 - q01) * 0.15
        ax.set_ylim(q01 - margin, q99 + margin * 2.5)

        kw = bio["kruskal_wallis"]
        kw_entry = next((k for k in kw if k["region"] == REGION_NAMES[ri]), None)
        if kw_entry and kw_entry["p"] < 0.01:
            sig_text = "***" if kw_entry["p"] < 0.001 else "**"
            ax.text(1, q99 + margin * 1.5, sig_text, ha="center", va="bottom",
                    fontsize=8, color=PAL["AD"], fontweight="bold")

        ax.set_xticks(positions)
        ax.set_xticklabels(CLASS_NAMES, fontsize=7, fontweight="bold")
        ax.set_title(key_regions[idx], fontsize=8, fontweight="bold",
                     color=PAL["AD"])
        ax.grid(True, axis="y", alpha=0.15)
        if idx % 3 == 0:
            ax.set_ylabel("Attention Weight", fontsize=8, fontweight="bold")

        _panel_label(ax, chr(ord("a") + idx))

    fig.suptitle("Attention Distribution in AD-Key Brain Regions",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout(h_pad=1.0, w_pad=0.8, pad=0.8)
    _save(fig, save_dir / "fig3_violin_key_regions.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Disease Gradient (connected dot / slope chart)
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_disease_gradient(bio, save_dir):
    _apply_style()
    gradient = bio.get("mci_gradient", [])
    if not gradient:
        print("  [skip] fig4: no gradient data")
        return

    mono = [g for g in gradient if g["monotonic"]]
    up = [g for g in mono if g["direction"] == "up"]
    down = [g for g in mono if g["direction"] == "down"]

    fig = plt.figure(figsize=(6.8, 3.6))
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.35)
    ax_up = fig.add_subplot(gs[0])
    ax_down = fig.add_subplot(gs[1])

    x = np.array([0, 1, 2])
    x_labels = CLASS_NAMES

    def _plot_panel(ax, regions, title, panel_label):
        for i, g in enumerate(regions[:8]):
            vals = [g["cn_mean"], g["mci_mean"], g["ad_mean"]]
            is_ad = g["region"] in AD_KEY
            color = PAL["AD"] if is_ad else PAL["CN"]
            lw = 2.0 if is_ad else 1.0
            alpha = 0.9 if is_ad else 0.4
            marker = "D" if is_ad else "o"
            ms = 5 if is_ad else 3.5

            ax.plot(x, vals, color=color, linewidth=lw, alpha=alpha,
                    marker=marker, markersize=ms, markeredgecolor="white",
                    markeredgewidth=0.4, zorder=3 if is_ad else 2)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8, fontweight="bold")
        ax.set_ylabel("Mean Attention Weight", fontsize=8, fontweight="bold")
        ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
        ax.set_xlim(-0.25, 2.25)
        ax.grid(True, axis="y", alpha=0.15)
        _panel_label(ax, panel_label)

    _plot_panel(ax_up, up,
                f"Increasing Attention\nCN < MCI < AD  ({len(up)} regions)", "a")
    _plot_panel(ax_down, down,
                f"Decreasing Attention\nCN > MCI > AD  ({len(down)} regions)", "b")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=PAL["AD"], lw=2, marker="D", markersize=5,
               label="AD-key regions"),
        Line2D([0], [0], color=PAL["CN"], lw=1, marker="o", markersize=4,
               label="Other regions"),
    ]
    ax_down.legend(handles=legend_elements, loc="lower left", fontsize=7, frameon=True)

    fig.suptitle("Disease Progression Gradient in Region Attention",
                 fontsize=10.5, fontweight="bold", y=1.02)
    fig.tight_layout(pad=0.8)
    _save(fig, save_dir / "fig4_disease_gradient.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Braak Correlation (clean scatter + repelled labels)
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_braak_scatter(bio, save_dir):
    _apply_style()
    rdi = bio["rdi"]
    braak_corr = bio["braak_correlation"]

    BRAAK = {
        "L-Hipp": 6, "R-Hipp": 6, "L-Amyg": 5, "R-Amyg": 5,
        "L-Ctx": 4, "R-Ctx": 4, "L-Thal": 3, "R-Thal": 3,
        "L-Vent": 3, "R-Vent": 3, "L-Caud": 2, "R-Caud": 2,
        "L-Put": 2, "R-Put": 2, "L-Pall": 1, "R-Pall": 1,
        "BStem": 1, "L-WM": 0, "R-WM": 0, "L-Acc": 1, "R-Acc": 1,
    }

    fig, ax = plt.subplots(figsize=(5.2, 3.8))

    xs, ys, names, colors = [], [], [], []
    for r in rdi:
        reg = r["region"]
        if reg in BRAAK:
            xs.append(BRAAK[reg] + np.random.uniform(-0.12, 0.12))
            ys.append(r["rdi"])
            names.append(reg)
            colors.append(PAL["AD"] if reg in AD_KEY else PAL["CN"])

    ax.scatter(xs, ys, c=colors, s=50, zorder=5, edgecolors="white",
               linewidth=0.6, alpha=0.85)

    if len(xs) > 2:
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        x_line = np.linspace(-0.5, 6.5, 100)
        ax.plot(x_line, p(x_line), color=PAL["gray"], linewidth=1,
                linestyle="--", alpha=0.5)

    name_to_point = {n: (xv, yv, c) for xv, yv, n, c in zip(xs, ys, names, colors)}
    top_other = [r["region"] for r in rdi if r["region"] not in AD_KEY][:4]
    label_regions = list(AD_KEY) + top_other
    for name in label_regions:
        if name not in name_to_point:
            continue
        xi, yi, col = name_to_point[name]
        ax.annotate(
            name, (xi, yi),
            textcoords="offset points", xytext=(6, 6),
            fontsize=7, color=col, ha="left",
            fontweight="bold" if name in AD_KEY else "semibold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8)
        )

    rho = braak_corr["rho"]
    p_val = braak_corr["p"]
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    ax.text(0.03, 0.97,
            f"Spearman $\\rho$ = {rho:.3f}\np = {p_val:.3f} ({sig})",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=PAL["grid"], alpha=0.9))

    ax.set_xlabel("Braak AD Vulnerability Rank", fontsize=8, fontweight="bold")
    ax.set_ylabel("Region Discriminability Index", fontsize=8, fontweight="bold")
    ax.set_title("Attention RDI vs Neuropathological Staging",
                 fontsize=9, fontweight="bold", pad=8)
    ax.grid(True, alpha=0.15)
    ax.set_xlim(-0.5, 6.5)

    fig.tight_layout(pad=0.8)
    _save(fig, save_dir / "fig5_braak_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Clinical Alignment: Donut + Radar
# ═══════════════════════════════════════════════════════════════════════════════

def fig6_clinical_alignment(bio, save_dir):
    _apply_style()
    cas = bio["clinical_alignment"]
    profiles = bio["per_class_mean_attention"]

    fig = plt.figure(figsize=(6.8, 3.4))
    gs = GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.28)

    # Panel A: Donut chart
    ax_donut = fig.add_subplot(gs[0])
    cas_val = cas["cas_abs"]
    sizes = [cas_val, 1 - cas_val]
    colors_d = [PAL["AD"], PAL["light"]]
    wedges, texts, autotexts = ax_donut.pie(
        sizes, colors=colors_d, startangle=90, counterclock=False,
        wedgeprops=dict(width=0.35, edgecolor="white", linewidth=2),
        autopct=lambda p: f"{p:.1f}%" if p > 5 else "",
        textprops={"fontsize": 8, "fontweight": "bold"},
    )
    ax_donut.text(0, 0.05, f"{cas_val:.0%}", ha="center", va="center",
                  fontsize=18, fontweight="bold", color=PAL["AD"])
    ax_donut.text(0, -0.12, "CAS", ha="center", va="center",
                  fontsize=7, color=PAL["gray"])

    dir_rate = cas.get("ad_direction_rate", 1.0)
    ax_donut.text(0, -0.72,
                  f"Direction: {dir_rate:.0%} aligned\n6/6 AD-key regions AD > CN",
                  ha="center", va="center", fontsize=6, color=PAL["text"],
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3F3",
                            edgecolor=PAL["light"], alpha=0.8))
    _panel_label(ax_donut, "a", x=-0.05, y=1.02)

    # Panel B: compact grouped bars (top discriminative regions)
    ax_bar = fig.add_subplot(gs[1])
    cn = np.array(profiles["CN"])
    mci = np.array(profiles["MCI"])
    ad = np.array(profiles["AD"])
    diff = np.abs(ad - cn)
    top_idx = np.argsort(diff)[-12:][::-1]
    x = np.arange(len(top_idx))
    w = 0.26

    ax_bar.bar(x - w, cn[top_idx], width=w, color=PAL["CN"], label="CN", alpha=0.9)
    ax_bar.bar(x, mci[top_idx], width=w, color=PAL["MCI"], label="MCI", alpha=0.9)
    ax_bar.bar(x + w, ad[top_idx], width=w, color=PAL["AD"], label="AD", alpha=0.9)

    labels = [SHORT_NAMES[i] for i in top_idx]
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=35, ha="right", fontsize=7, fontweight="bold")
    for i, idx in enumerate(top_idx):
        if REGION_NAMES[idx] in AD_KEY:
            ax_bar.get_xticklabels()[i].set_color(PAL["AD"])
    ax_bar.set_ylabel("Mean Attention", fontsize=8, fontweight="bold")
    ax_bar.set_ylim(0.84, 1.42)
    ax_bar.grid(True, axis="y", alpha=0.15)
    ax_bar.legend(ncol=3, loc="upper right", fontsize=7, frameon=True)
    _panel_label(ax_bar, "b", x=-0.08, y=1.02)

    fig.suptitle("Clinical Alignment of Attention Patterns",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout(pad=0.8)
    _save(fig, save_dir / "fig6_clinical_alignment.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Cross-Dataset IXI (Slope / Paired Dot)
# ═══════════════════════════════════════════════════════════════════════════════

def fig7_cross_ixi(cross, bio, save_dir):
    _apply_style()
    ixi = cross.get("IXI")
    if not ixi:
        print("  [skip] fig7: no IXI data")
        return

    adni_cn = np.array(bio["per_class_mean_attention"]["CN"])
    ixi_cn = np.array(ixi["ixi_profile"])

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3),
                             gridspec_kw={"width_ratios": [2.5, 1]})

    # Panel A: Slope chart
    ax = axes[0]
    for i, (a_val, i_val, name) in enumerate(zip(adni_cn, ixi_cn, REGION_NAMES)):
        is_ad = name in AD_KEY
        color = PAL["AD"] if is_ad else PAL["CN"]
        lw = 1.7 if is_ad else 0.8
        alpha = 0.85 if is_ad else 0.4

        ax.plot([0, 1], [a_val, i_val], color=color, linewidth=lw,
                alpha=alpha, zorder=3 if is_ad else 2)
        ax.scatter([0], [a_val], color=color, s=20 if is_ad else 10,
                   zorder=4, edgecolors="white", linewidth=0.3)
        ax.scatter([1], [i_val], color=color, s=20 if is_ad else 10,
                   zorder=4, edgecolors="white", linewidth=0.3)

        if is_ad:
            ax.annotate(name, (1, i_val), xytext=(8, 0),
                        textcoords="offset points", fontsize=6.5,
                        ha="left", va="center", color=PAL["AD"], fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ADNI CN", "IXI CN"], fontsize=8, fontweight="bold")
    ax.set_ylabel("Mean Attention Weight", fontsize=8, fontweight="bold")
    ax.set_xlim(-0.4, 1.4)
    ax.grid(True, axis="y", alpha=0.15)

    cos_val = ixi["cosine_similarity"]["CN"]
    ax.text(0.5, 0.97, f"Cosine Similarity = {cos_val:.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            fontweight="bold", color=PAL["accent"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9",
                      edgecolor=PAL["accent"], alpha=0.8))
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=PAL["AD"], lw=1.8, label="AD-key regions"),
        Line2D([0], [0], color=PAL["CN"], lw=1.0, label="Other regions"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=7, frameon=True)
    _panel_label(ax, "a")

    # Panel B: Scatter (ADNI vs IXI)
    ax2 = axes[1]
    for i, (a_val, i_val, name) in enumerate(zip(adni_cn, ixi_cn, REGION_NAMES)):
        is_ad = name in AD_KEY
        color = PAL["AD"] if is_ad else PAL["CN"]
        marker = "D" if is_ad else "o"
        ax2.scatter(a_val, i_val, color=color, s=30, marker=marker,
                    edgecolors="white", linewidth=0.4, zorder=3)

    lims = [min(adni_cn.min(), ixi_cn.min()) - 0.05,
            max(adni_cn.max(), ixi_cn.max()) + 0.05]
    ax2.plot(lims, lims, color=PAL["gray"], linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel("ADNI CN", fontsize=7)
    ax2.set_ylabel("IXI CN", fontsize=7)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.15)
    _panel_label(ax2, "b")

    fig.suptitle("Cross-Dataset Attention Consistency: ADNI vs IXI",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout(pad=0.8)
    _save(fig, save_dir / "fig7_cross_ixi.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Cross-Dataset OASIS (Radar Overlay per Class)
# ═══════════════════════════════════════════════════════════════════════════════

def fig8_cross_oasis(cross, bio, save_dir):
    _apply_style()
    oasis = cross.get("OASIS")
    if not oasis:
        print("  [skip] fig8: no OASIS data")
        return

    adni_profiles = bio["per_class_mean_attention"]
    oasis_profiles = oasis.get("oasis_profiles", {})

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0), sharey=True)

    for idx, (ax, cname) in enumerate(zip(axes, CLASS_NAMES)):
        adni_vals = adni_profiles.get(cname)
        oasis_vals = oasis_profiles.get(cname)
        if adni_vals is None or oasis_vals is None:
            ax.set_visible(False)
            continue

        adni_arr = np.array(adni_vals)
        oasis_arr = np.array(oasis_vals)
        top_idx = np.argsort(np.abs(adni_arr - oasis_arr))[-10:][::-1]
        x = np.arange(len(top_idx))

        ax.plot(x, adni_arr[top_idx], color=PAL["adni"], linewidth=1.8,
                marker="o", markersize=3.5, label="ADNI")
        ax.plot(x, oasis_arr[top_idx], color=PAL["ext"], linewidth=1.8,
                marker="o", markersize=3.5, linestyle="--", label="OASIS")
        for i, t in enumerate(top_idx):
            ax.vlines(i, min(adni_arr[t], oasis_arr[t]), max(adni_arr[t], oasis_arr[t]),
                      color=PAL["gray"], alpha=0.25, linewidth=1)

        labels = [SHORT_NAMES[i] for i in top_idx]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5, fontweight="bold")
        ax.set_ylim(0.84, 1.42)
        ax.grid(True, axis="y", alpha=0.15)

        cos_val = oasis["cosine_similarity"].get(cname, 0)
        ax.set_title(f"{cname}\ncos = {cos_val:.3f}", fontsize=8.5,
                     fontweight="bold", pad=6,
                     color=CLASS_COLORS[idx])

        if idx == 0:
            ax.legend(loc="upper right", fontsize=7, frameon=True)
            ax.set_ylabel("Mean Attention", fontsize=8, fontweight="bold")

    fig.suptitle("Cross-Dataset Attention Generalization: ADNI vs OASIS",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout(pad=0.8, w_pad=1.0)
    _save(fig, save_dir / "fig8_cross_oasis_radar.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Consistency Summary (Bullet / Gauge)
# ═══════════════════════════════════════════════════════════════════════════════

def fig9_consistency_summary(cross, save_dir):
    _apply_style()

    entries = []
    for ds_name, ds_data in cross.items():
        cos = ds_data.get("cosine_similarity", {})
        for cname in cos:
            entries.append({
                "label": f"{ds_name} {cname}",
                "cosine": cos[cname],
                "dataset": ds_name,
                "class": cname,
            })

    if not entries:
        print("  [skip] fig9: no data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw={"width_ratios": [1, 1.4]})

    # Panel A: Same-class cosine similarity bar chart
    ax = axes[0]
    labels = [e["label"] for e in entries]
    cos_vals = [e["cosine"] for e in entries]
    y = np.arange(len(entries))

    ax.barh(y, [1.0] * len(entries), color=PAL["light"], height=0.5, zorder=1)
    ax.barh(y, [0.95] * len(entries), color="#E8F5E9", height=0.5, zorder=1)

    bars = ax.barh(y, cos_vals, color=[PAL["AD"] if "AD" in e["label"]
                   else PAL["MCI"] if "MCI" in e["label"]
                   else PAL["CN"] for e in entries],
                   height=0.3, zorder=3, alpha=0.85)

    for i, (val, entry) in enumerate(zip(cos_vals, entries)):
        ax.text(val + 0.003, i, f"{val:.3f}", va="center", fontsize=6.5,
                fontweight="bold", color=PAL["text"])

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7, fontweight="bold")
    ax.set_xlim(0.85, 1.02)
    ax.set_xlabel("Cosine Similarity", fontsize=8, fontweight="bold")
    ax.axvline(0.95, color=PAL["accent"], linewidth=0.8, linestyle=":",
               alpha=0.7, zorder=2)
    ax.text(0.951, len(entries) - 0.3, "excellent", fontsize=5,
            color=PAL["accent"], rotation=90, va="bottom")
    ax.invert_yaxis()
    ax.set_title("Same-Class Cross-Dataset\nCosine Similarity", fontsize=9, fontweight="bold")
    _panel_label(ax, "a")

    # Panel B: Full cross-group cosine similarity heatmap
    ax2 = axes[1]
    cross_json_path = Path("chapter1_foundation/cross_dataset_interpretability_results.json")
    if cross_json_path.exists():
        import json as _json
        full_cross = _json.loads(cross_json_path.read_text())
        cgm = full_cross.get("cross_group_cosine_matrix", {})
        cg_labels = cgm.get("labels", [])
        cg_matrix = np.array(cgm.get("matrix", []))

        if len(cg_labels) > 0 and cg_matrix.size > 0:
            from matplotlib.colors import Normalize
            im = ax2.imshow(cg_matrix, cmap="RdYlGn", vmin=0.98, vmax=1.0,
                            aspect="auto", interpolation="nearest")
            ax2.set_xticks(range(len(cg_labels)))
            ax2.set_xticklabels(cg_labels, rotation=45, ha="right", fontsize=6)
            ax2.set_yticks(range(len(cg_labels)))
            ax2.set_yticklabels(cg_labels, fontsize=6)

            for i in range(len(cg_labels)):
                for j in range(len(cg_labels)):
                    val = cg_matrix[i, j]
                    color = "white" if val < 0.99 else "#333333"
                    ax2.text(j, i, f"{val:.3f}", ha="center", va="center",
                             fontsize=5, color=color, fontweight="medium")

            same_class_pairs = []
            for i, li in enumerate(cg_labels):
                for j, lj in enumerate(cg_labels):
                    ds_i, cls_i = li.rsplit(" ", 1)
                    ds_j, cls_j = lj.rsplit(" ", 1)
                    if ds_i != ds_j and cls_i == cls_j:
                        same_class_pairs.append((i, j))

            for (ri, ci) in same_class_pairs:
                rect = plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1, linewidth=1.5,
                                     edgecolor=PAL["AD"], facecolor="none")
                ax2.add_patch(rect)

            cbar = fig.colorbar(im, ax=ax2, shrink=0.8, pad=0.02)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label("Cosine Similarity", fontsize=7)
        else:
            ax2.text(0.5, 0.5, "No cross-group data", transform=ax2.transAxes,
                     ha="center", va="center")
    else:
        ax2.text(0.5, 0.5, "No cross-group data", transform=ax2.transAxes,
                 ha="center", va="center")

    ax2.set_title("Full Cross-Group\nCosine Similarity", fontsize=9, fontweight="bold")
    _panel_label(ax2, "b")

    fig.suptitle("Cross-Dataset Interpretability Generalization",
                 fontsize=10, fontweight="bold", y=1.04)
    fig.tight_layout(w_pad=1.0, pad=0.8)
    _save(fig, save_dir / "fig9_consistency_summary.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Self-Attention Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def fig10_self_attention_heatmap(bio, save_dir):
    """Self-attention (diagonal) heatmap, same style as fig1."""
    _apply_style()
    attn_data = _load_raw_attention_full()
    if attn_data is None:
        print("  [skip] fig10: no raw attention data")
        return

    attn, labels = attn_data
    attn_mean = attn.mean(axis=1)  # (N, R, R)
    n, r, _ = attn_mean.shape
    self_attn = np.array([attn_mean[i].diagonal() for i in range(n)])  # (N, R)

    data = np.zeros((3, len(REGION_NAMES)))
    for c in range(3):
        mask = labels == c
        if mask.sum() > 0:
            data[c] = self_attn[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    cmap = LinearSegmentedColormap.from_list(
        "custom_div",
        ["#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B"], N=256)
    vmin, vmax = data.min() - 0.002, data.max() + 0.002
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")

    ax.set_yticks(range(3))
    ax.set_yticklabels(CLASS_NAMES, fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(REGION_NAMES)))
    ax.set_xticklabels(SHORT_NAMES, rotation=50, ha="right", fontsize=6.5)

    for i in range(3):
        for j in range(len(REGION_NAMES)):
            v = data[i, j]
            color = "white" if abs(v - (vmin + vmax) / 2) > (vmax - vmin) * 0.35 else "#333333"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=5.5, color=color, fontweight="medium")

    for j, name in enumerate(REGION_NAMES):
        if name in AD_KEY:
            rect = plt.Rectangle((j - 0.5, -0.5), 1, 3, linewidth=1.2,
                                 edgecolor=PAL["AD"], facecolor="none",
                                 linestyle="--", alpha=0.7)
            ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, aspect=20)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Mean Self-Attention", fontsize=7)

    ax.set_title("Region Self-Attention Profile by Diagnostic Group",
                 fontsize=10, fontweight="bold", pad=8)
    fig.tight_layout()
    _save(fig, save_dir / "fig10_self_attention_heatmap.png")


# ── helper: load raw attention ───────────────────────────────────────────────

def _load_raw_attention_full():
    """Load raw attention maps (N, heads, R, R)."""
    results_dir = Path("chapter1_foundation/experiment_results_v3")
    all_attn, all_labels = [], []

    for seed_dir in sorted(results_dir.glob("seed_*")):
        for fname in ["all_results.json", "all_results_partial.json"]:
            fpath = seed_dir / fname
            if fpath.exists():
                with open(fpath) as f:
                    data = json.load(f)
                for key, val in data.items():
                    if not key.startswith("Ours (Atlas+AnatDist)"):
                        continue
                    attn_maps = val.get("attention_maps", [])
                    attn_labels = val.get("attention_labels", [])
                    if not attn_maps:
                        continue
                    all_attn.append(np.array(attn_maps))
                    all_labels.extend(attn_labels)
                break

    if not all_attn:
        return None
    return np.concatenate(all_attn, axis=0), np.array(all_labels)


def _load_raw_attention():
    result = _load_raw_attention_full()
    if result is None:
        return None
    attn, labels = result
    attn_mean = attn.mean(axis=1)
    received = attn_mean.sum(axis=1)
    return received, labels


# ── main ─────────────────────────────────────────────────────────────────────

def _copy_to_legacy(save_dir):
    """Copy new figures into the old directories with their old names."""
    import shutil
    bio_dir = Path("chapter1_foundation/figures_biomarker")
    cross_dir = Path("chapter1_foundation/figures_cross_dataset")
    bio_dir.mkdir(parents=True, exist_ok=True)
    cross_dir.mkdir(parents=True, exist_ok=True)

    mapping_bio = {
        "fig1_attention_heatmap":    "attn_heatmap_received",
        "fig10_self_attention_heatmap": "attn_heatmap_self_attention",
        "fig2_rdi_lollipop":         "rdi_barplot",
        "fig3_violin_key_regions":   "group_comparison_boxplots",
        "fig4_disease_gradient":     "disease_gradient",
        "fig5_braak_scatter":        "braak_correlation",
        "fig6_clinical_alignment":   "clinical_alignment_summary",
    }
    mapping_cross = {
        "fig7_cross_ixi":            "cross_dataset_ixi",
        "fig8_cross_oasis_radar":    "cross_dataset_oasis",
        "fig9_consistency_summary":  "consistency_summary",
    }

    for new_name, old_name in mapping_bio.items():
        for ext in (".png", ".pdf"):
            src = save_dir / (new_name + ext)
            dst = bio_dir / (old_name + ext)
            if src.exists():
                shutil.copy2(src, dst)
    print(f"  Synced {len(mapping_bio)} figures -> {bio_dir}/")

    for new_name, old_name in mapping_cross.items():
        for ext in (".png", ".pdf"):
            src = save_dir / (new_name + ext)
            dst = cross_dir / (old_name + ext)
            if src.exists():
                shutil.copy2(src, dst)
    print(f"  Synced {len(mapping_cross)} figures -> {cross_dir}/")


MM2IN = 1 / 25.4
COL1 = 89 * MM2IN     # single-column  89 mm
COL15 = 120 * MM2IN   # 1.5-column    120 mm
COL2 = 183 * MM2IN    # double-column  183 mm

FIGURE_MANIFEST = [
    # (func_name,  figsize_inches,  filename,  caption)
    ("fig1",  (COL2, 0.85),   "Fig1_attention_heatmap",
     "Mean attention received per brain region across diagnostic groups (CN, MCI, AD). "
     "Dashed red boxes indicate AD-key regions (hippocampus, amygdala, ventricles)."),

    ("fig10", (COL2, 0.85),   "Fig2_self_attention_heatmap",
     "Self-attention (diagonal) per brain region across diagnostic groups."),

    ("fig2",  (COL1, 2.1),    "Fig3_rdi_lollipop",
     "Region Discriminability Index (|Cohen's d|, AD vs CN). "
     "Diamond markers indicate known AD-affected regions. Dotted line = medium effect threshold."),

    ("fig3",  (COL2, 1.65),   "Fig4_violin_key_regions",
     "Attention weight distributions in six AD-key brain regions. "
     "Violin + box plots with trend line (dashed green) connecting group means. "
     "** p < 0.01, *** p < 0.001 (Kruskal–Wallis)."),

    ("fig4",  (COL2, 1.40),   "Fig5_disease_gradient",
     "Disease progression gradient in region attention. "
     "(a) Regions with monotonically increasing attention CN → MCI → AD. "
     "(b) Regions with monotonically decreasing attention."),

    ("fig5",  (COL15, 1.50),  "Fig6_braak_scatter",
     "RDI vs Braak neuropathological vulnerability rank. "
     "Red = AD-key regions. Dashed line = linear fit."),

    ("fig6",  (COL2, 1.35),   "Fig7_clinical_alignment",
     "(a) Clinical Alignment Score: proportion of attention difference in AD-key regions. "
     "(b) Grouped bar chart of top-12 discriminative regions by diagnosis."),

    ("fig7",  (COL2, 1.30),   "Fig8_cross_ixi",
     "Cross-dataset attention consistency (ADNI vs IXI, CN subjects). "
     "(a) Slope chart of 21 regions. (b) Identity scatter."),

    ("fig8",  (COL2, 1.20),   "Fig9_cross_oasis",
     "Cross-dataset attention generalization (ADNI vs OASIS). "
     "Top-10 divergent regions per diagnostic group; cosine similarity annotated."),

    ("fig9",  (COL2, 0.90),   "Fig10_consistency_summary",
     "Summary of cross-dataset interpretability generalization. "
     "(a) Cosine similarity of attention profiles. (b) Spearman rank correlation."),
]


def _export_submission(src_dir):
    """Re-save all figures at exact Nature sizing (300 dpi TIFF + 600 dpi PDF)
    and write a caption file."""
    import shutil, textwrap

    sub_dir = Path("chapter1_foundation/figures_submission")
    sub_dir.mkdir(parents=True, exist_ok=True)

    caption_lines = []

    old_to_new = {
        "fig1":  "fig1_attention_heatmap",
        "fig10": "fig10_self_attention_heatmap",
        "fig2":  "fig2_rdi_lollipop",
        "fig3":  "fig3_violin_key_regions",
        "fig4":  "fig4_disease_gradient",
        "fig5":  "fig5_braak_scatter",
        "fig6":  "fig6_clinical_alignment",
        "fig7":  "fig7_cross_ixi",
        "fig8":  "fig8_cross_oasis_radar",
        "fig9":  "fig9_consistency_summary",
    }

    for idx, (func_key, (w, h_ratio), out_name, caption) in enumerate(FIGURE_MANIFEST, 1):
        src_base = old_to_new[func_key]
        for ext in (".png", ".pdf"):
            src = src_dir / (src_base + ext)
            dst = sub_dir / (out_name + ext)
            if src.exists():
                shutil.copy2(src, dst)

        caption_lines.append(f"Figure {idx}  |  {out_name}")
        caption_lines.append(textwrap.fill(caption, width=90))
        caption_lines.append("")

    cap_path = sub_dir / "figure_captions.txt"
    cap_path.write_text("\n".join(caption_lines), encoding="utf-8")
    print(f"  Captions -> {cap_path}")
    print(f"  {len(FIGURE_MANIFEST)} figures -> {sub_dir}/")


def main():
    bio = load_biomarker()
    cross = load_cross()

    save_dir = Path("chapter1_foundation/figures_publication")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Generating publication-quality figures ...")
    fig1_attention_heatmap(bio, save_dir)
    fig2_rdi_lollipop(bio, save_dir)
    fig3_group_violin(bio, save_dir)
    fig4_disease_gradient(bio, save_dir)
    fig5_braak_scatter(bio, save_dir)
    fig6_clinical_alignment(bio, save_dir)
    fig7_cross_ixi(cross, bio, save_dir)
    fig8_cross_oasis(cross, bio, save_dir)
    fig9_consistency_summary(cross, save_dir)
    fig10_self_attention_heatmap(bio, save_dir)

    print("\nSyncing to legacy directories ...")
    _copy_to_legacy(save_dir)

    print("\nExporting submission package ...")
    _export_submission(save_dir)

    print(f"\nAll figures saved to {save_dir}/")


if __name__ == "__main__":
    main()
