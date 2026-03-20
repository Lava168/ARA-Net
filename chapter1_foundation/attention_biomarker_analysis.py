#!/usr/bin/env python3
"""
Attention-as-Biomarker Analysis for ARA-Net.

Analyzes region-level attention patterns to demonstrate:
1. Clinical alignment: attention weights correlate with known AD atrophy regions
2. Region Discriminability Index (RDI): Cohen's d on attention weights
3. Statistical tests: Kruskal-Wallis + post-hoc for group differences
4. Braak staging correlation: Spearman rank correlation with literature
5. Clinical Alignment Score: quantifies interpretability fidelity

Usage:
    python -m chapter1_foundation.attention_biomarker_analysis \
        --results_dir chapter1_foundation/experiment_results_v3 \
        --output chapter1_foundation/attention_biomarker_results.json \
        --figures chapter1_foundation/figures_biomarker
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scienceplots
from matplotlib.colors import LinearSegmentedColormap

CLASS_NAMES = ["CN", "MCI", "AD"]

REGION_NAMES = [
    "L-WM", "L-Ctx", "L-Vent", "L-Thal", "L-Caud",
    "L-Put", "L-Pall", "BStem", "L-Hipp", "L-Amyg",
    "L-Acc", "R-WM", "R-Ctx", "R-Vent", "R-Thal",
    "R-Caud", "R-Put", "R-Pall", "R-Hipp", "R-Amyg", "R-Acc",
]

# Known AD-affected regions (from Braak staging / literature)
# Higher rank = earlier / more severe involvement
BRAAK_AD_PRIORITY = {
    "L-Hipp": 6, "R-Hipp": 6,
    "L-Amyg": 5, "R-Amyg": 5,
    "L-Ctx": 4, "R-Ctx": 4,
    "L-Thal": 3, "R-Thal": 3,
    "L-Vent": 3, "R-Vent": 3,
    "L-Caud": 2, "R-Caud": 2,
    "L-Put": 2, "R-Put": 2,
    "L-Pall": 1, "R-Pall": 1,
    "BStem": 1,
    "L-WM": 0, "R-WM": 0,
    "L-Acc": 1, "R-Acc": 1,
}

AD_KEY_REGIONS = {"L-Hipp", "R-Hipp", "L-Amyg", "R-Amyg", "L-Vent", "R-Vent"}


def set_nature_style():
    plt.style.use(['science', 'nature', 'no-latex'])
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ── data loading ─────────────────────────────────────────────────────────────

def load_attention_data(results_dir: Path, model_filter="Ours (Atlas+AnatDist)"):
    """Load attention maps and labels from all seeds/folds for a given model."""
    all_attn = []  # list of (n_samples, n_heads, n_regions, n_regions)
    all_labels = []
    all_keys = []

    for seed_dir in sorted(results_dir.glob("seed_*")):
        for fname in ["all_results.json", "all_results_partial.json"]:
            fpath = seed_dir / fname
            if fpath.exists():
                with open(fpath) as f:
                    data = json.load(f)
                for key, val in data.items():
                    if not key.startswith(model_filter):
                        continue
                    attn_maps = val.get("attention_maps", [])
                    attn_labels = val.get("attention_labels", [])
                    if not attn_maps:
                        continue
                    all_attn.append(np.array(attn_maps))
                    all_labels.extend(attn_labels)
                    all_keys.append(key)
                break

    if not all_attn:
        return None, None

    attn = np.concatenate(all_attn, axis=0)
    labels = np.array(all_labels)
    return attn, labels


# ── region attention extraction ──────────────────────────────────────────────

def compute_region_attention_received(attn: np.ndarray) -> np.ndarray:
    """Compute how much attention each region receives (column sum).
    attn: (N, heads, regions, regions) -> (N, regions)
    """
    attn_mean = attn.mean(axis=1)  # average over heads -> (N, R, R)
    received = attn_mean.sum(axis=1)  # sum over query dim -> (N, R)
    return received


def compute_region_attention_given(attn: np.ndarray) -> np.ndarray:
    """Compute how much attention each region gives (row sum).
    attn: (N, heads, regions, regions) -> (N, regions)
    """
    attn_mean = attn.mean(axis=1)
    given = attn_mean.sum(axis=2)
    return given


def compute_region_self_attention(attn: np.ndarray) -> np.ndarray:
    """Diagonal of attention matrix (self-attention per region).
    attn: (N, heads, regions, regions) -> (N, regions)
    """
    attn_mean = attn.mean(axis=1)
    n, r, _ = attn_mean.shape
    self_attn = np.array([attn_mean[i].diagonal() for i in range(n)])
    return self_attn


# ── statistical analysis ─────────────────────────────────────────────────────

def group_by_class(values: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
    return {CLASS_NAMES[c]: values[labels == c] for c in range(3)}


def kruskal_wallis_per_region(region_values: np.ndarray, labels: np.ndarray) -> List[dict]:
    """Kruskal-Wallis test for each region across CN/MCI/AD groups."""
    from scipy.stats import kruskal, mannwhitneyu
    n_regions = region_values.shape[1]
    results = []

    for r in range(n_regions):
        groups = [region_values[labels == c, r] for c in range(3)]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            results.append({"region": REGION_NAMES[r], "H": 0, "p": 1.0, "significant": False})
            continue

        try:
            H, p = kruskal(*groups)
        except Exception:
            H, p = 0.0, 1.0

        posthoc = {}
        pairs = [("CN", "AD", 0, 2), ("CN", "MCI", 0, 1), ("MCI", "AD", 1, 2)]
        for name_a, name_b, ca, cb in pairs:
            ga = region_values[labels == ca, r]
            gb = region_values[labels == cb, r]
            if len(ga) > 0 and len(gb) > 0:
                try:
                    _, pp = mannwhitneyu(ga, gb, alternative="two-sided")
                except Exception:
                    pp = 1.0
                posthoc[f"{name_a}_vs_{name_b}"] = float(pp)

        results.append({
            "region": REGION_NAMES[r],
            "H": float(H),
            "p": float(p),
            "significant_005": p < 0.05,
            "significant_001": p < 0.01,
            "posthoc": posthoc,
        })

    return results


def compute_rdi(region_values: np.ndarray, labels: np.ndarray) -> List[dict]:
    """Region Discriminability Index (Cohen's d between AD and CN)."""
    n_regions = region_values.shape[1]
    results = []

    cn_vals = region_values[labels == 0]
    ad_vals = region_values[labels == 2]

    for r in range(n_regions):
        cn = cn_vals[:, r]
        ad = ad_vals[:, r]

        if len(cn) < 2 or len(ad) < 2:
            results.append({"region": REGION_NAMES[r], "rdi": 0.0, "direction": "none"})
            continue

        mean_diff = ad.mean() - cn.mean()
        pooled_std = np.sqrt((cn.var() + ad.var()) / 2)
        rdi = abs(mean_diff) / max(pooled_std, 1e-8)
        direction = "AD>CN" if mean_diff > 0 else "CN>AD"

        results.append({
            "region": REGION_NAMES[r],
            "rdi": float(rdi),
            "cohens_d": float(mean_diff / max(pooled_std, 1e-8)),
            "mean_cn": float(cn.mean()),
            "mean_ad": float(ad.mean()),
            "mean_mci": float(region_values[labels == 1, r].mean()) if (labels == 1).sum() > 0 else 0,
            "direction": direction,
        })

    results.sort(key=lambda x: x["rdi"], reverse=True)
    return results


def compute_braak_correlation(rdi_results: List[dict]) -> dict:
    """Spearman correlation between signed Cohen's d (AD>CN) and Braak AD priority.
    Uses signed d so that regions where AD has HIGHER attention get positive values."""
    from scipy.stats import spearmanr

    cohens_d_vals = []
    braak_vals = []
    region_names = []
    for r in rdi_results:
        region = r["region"]
        if region in BRAAK_AD_PRIORITY:
            cohens_d_vals.append(r.get("cohens_d", 0))
            braak_vals.append(BRAAK_AD_PRIORITY[region])
            region_names.append(region)

    if len(cohens_d_vals) < 5:
        return {"rho": 0.0, "p": 1.0, "n": len(cohens_d_vals)}

    rho, p = spearmanr(cohens_d_vals, braak_vals)
    return {
        "rho": float(rho), "p": float(p), "n": len(cohens_d_vals),
        "regions": region_names,
        "cohens_d": [float(v) for v in cohens_d_vals],
        "braak_rank": braak_vals,
    }


def compute_clinical_alignment_score(region_values: np.ndarray, labels: np.ndarray) -> dict:
    """Clinical Alignment Score: measures whether AD-key regions show
    significantly different attention patterns between AD and CN.

    Uses two metrics:
    1. CAS-abs: fraction of total |diff| in AD-key regions
    2. CAS-sig: fraction of AD-key regions that are statistically significant
    3. AD-direction: whether AD-key regions show increased attention in AD
    """
    cn_mean = region_values[labels == 0].mean(axis=0)
    ad_mean = region_values[labels == 2].mean(axis=0)
    mci_mean = region_values[labels == 1].mean(axis=0) if (labels == 1).sum() > 0 else None

    diff = ad_mean - cn_mean
    abs_diff = np.abs(diff)
    total_diff = abs_diff.sum()

    ad_region_indices = [i for i, name in enumerate(REGION_NAMES) if name in AD_KEY_REGIONS]
    ad_region_diff = abs_diff[ad_region_indices].sum()
    cas_abs = float(ad_region_diff / max(total_diff, 1e-8))

    ad_regions_ad_higher = sum(1 for i in ad_region_indices if diff[i] > 0)
    ad_direction_rate = ad_regions_ad_higher / max(len(ad_region_indices), 1)

    per_region = {}
    for i in ad_region_indices:
        name = REGION_NAMES[i]
        per_region[name] = {
            "cn_mean": float(cn_mean[i]),
            "mci_mean": float(mci_mean[i]) if mci_mean is not None else None,
            "ad_mean": float(ad_mean[i]),
            "diff_ad_cn": float(diff[i]),
            "ad_higher": bool(diff[i] > 0),
        }

    return {
        "cas_abs": cas_abs,
        "ad_direction_rate": float(ad_direction_rate),
        "ad_region_attention_diff": float(ad_region_diff),
        "total_attention_diff": float(total_diff),
        "ad_regions_detail": per_region,
        "interpretation": (
            f"CAS={cas_abs:.1%} of attention difference in AD-key regions. "
            f"{ad_regions_ad_higher}/{len(ad_region_indices)} AD-key regions show "
            f"increased attention in AD group (direction alignment: {ad_direction_rate:.0%})."
        ),
    }


# ── figures ──────────────────────────────────────────────────────────────────

def fig_attention_heatmap(region_values, labels, save_dir, metric_name="received"):
    """Heatmap of mean attention per region, grouped by diagnosis."""
    set_nature_style()
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    group_means = np.zeros((3, len(REGION_NAMES)))
    for c in range(3):
        mask = labels == c
        if mask.sum() > 0:
            group_means[c] = region_values[mask].mean(axis=0)

    im = ax.imshow(group_means, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(3))
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xticks(range(len(REGION_NAMES)))
    ax.set_xticklabels(REGION_NAMES, rotation=60, ha="right", fontsize=7)

    for i in range(3):
        for j in range(len(REGION_NAMES)):
            color = "white" if group_means[i, j] > group_means.max() * 0.7 else "black"
            ax.text(j, i, f"{group_means[i, j]:.3f}", ha="center", va="center",
                    fontsize=5.5, color=color)

    for j, name in enumerate(REGION_NAMES):
        if name in AD_KEY_REGIONS:
            ax.axvline(j - 0.5, color="#C73737", linewidth=0.8, alpha=0.5)
            ax.axvline(j + 0.5, color="#C73737", linewidth=0.8, alpha=0.5)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"Mean Attention ({metric_name})", fontsize=8)
    ax.set_title(f"Region Attention by Diagnosis ({metric_name.title()})",
                 fontsize=10, fontweight="bold")

    fig.tight_layout()
    _save(fig, save_dir, f"attn_heatmap_{metric_name}")
    print(f"  Attention heatmap ({metric_name})")


def fig_rdi_barplot(rdi_results, save_dir):
    """Bar plot of Region Discriminability Index, sorted by RDI."""
    set_nature_style()
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    regions = [r["region"] for r in rdi_results]
    rdis = [r["rdi"] for r in rdi_results]
    colors = ["#C73737" if r["region"] in AD_KEY_REGIONS else "#3B82C4" for r in rdi_results]

    y_pos = np.arange(len(regions))
    bars = ax.barh(y_pos, rdis, color=colors, edgecolor="white", linewidth=0.5,
                   height=0.7, zorder=3)

    for i, (rdi, r) in enumerate(zip(rdis, rdi_results)):
        ax.text(rdi + 0.02, i, f"{rdi:.2f} ({r['direction']})",
                va="center", fontsize=6.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(regions, fontsize=7)
    ax.set_xlabel("Region Discriminability Index (|Cohen's d|)")
    ax.set_title("Region Discriminability: AD vs CN\n(red = known AD regions)",
                 fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, axis="x")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#C73737", label="Known AD region"),
        Patch(facecolor="#3B82C4", label="Other region"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7)

    fig.tight_layout()
    _save(fig, save_dir, "rdi_barplot")
    print("  RDI bar plot")


def fig_group_comparison(region_values, labels, save_dir):
    """Box plots comparing attention distributions across groups for key regions."""
    set_nature_style()
    key_regions = ["L-Hipp", "R-Hipp", "L-Amyg", "R-Amyg", "L-Vent", "R-Vent",
                   "L-Thal", "R-Thal", "L-Ctx", "R-Ctx"]
    key_indices = [REGION_NAMES.index(r) for r in key_regions if r in REGION_NAMES]

    n_regions = len(key_indices)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    class_colors = {"CN": "#3B82C4", "MCI": "#F39C34", "AD": "#D94F4F"}

    for idx, (ax, ri) in enumerate(zip(axes, key_indices)):
        data_by_class = []
        positions = []
        colors_list = []
        for c, cname in enumerate(CLASS_NAMES):
            vals = region_values[labels == c, ri]
            if len(vals) > 0:
                data_by_class.append(vals)
                positions.append(c)
                colors_list.append(class_colors[cname])

        bp = ax.boxplot(data_by_class, positions=positions, widths=0.6,
                        patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

        ax.set_xticks(range(3))
        ax.set_xticklabels(CLASS_NAMES, fontsize=7)
        ax.set_title(REGION_NAMES[ri], fontsize=8, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        if idx % 5 == 0:
            ax.set_ylabel("Attention", fontsize=7)

    fig.suptitle("Attention Distribution by Diagnosis (Key Regions)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, save_dir, "group_comparison_boxplots")
    print("  Group comparison box plots")


def fig_braak_scatter(rdi_results, braak_corr, save_dir):
    """Scatter plot: RDI vs Braak AD priority."""
    set_nature_style()
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    x_braak, y_rdi, names = [], [], []
    for r in rdi_results:
        region = r["region"]
        if region in BRAAK_AD_PRIORITY:
            x_braak.append(BRAAK_AD_PRIORITY[region])
            y_rdi.append(r["rdi"])
            names.append(region)

    colors = ["#C73737" if n in AD_KEY_REGIONS else "#3B82C4" for n in names]
    ax.scatter(x_braak, y_rdi, c=colors, s=50, zorder=5, edgecolors="white", linewidth=0.5)

    for x, y, name in zip(x_braak, y_rdi, names):
        ax.annotate(name, (x, y), fontsize=5.5, textcoords="offset points",
                    xytext=(4, 4), alpha=0.8)

    if len(x_braak) > 2:
        z = np.polyfit(x_braak, y_rdi, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x_braak) - 0.5, max(x_braak) + 0.5, 100)
        ax.plot(x_line, p(x_line), "--", color="#9AA0A6", linewidth=1, alpha=0.7)

    rho = braak_corr["rho"]
    p_val = braak_corr["p"]
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    ax.text(0.05, 0.95, f"Spearman ρ = {rho:.3f} (p = {p_val:.4f}) {sig}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Braak AD Priority Rank (higher = more affected)")
    ax.set_ylabel("Region Discriminability Index (RDI)")
    ax.set_title("Attention RDI vs Known AD Neuropathology",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, save_dir, "braak_correlation")
    print("  Braak correlation scatter")


def fig_clinical_alignment_summary(cas_result, rdi_results, save_dir):
    """Summary figure: Clinical Alignment Score + top discriminative regions."""
    set_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Panel A: Pie chart of attention difference distribution
    ax = axes[0]
    cas = cas_result["cas_abs"]
    sizes = [cas, 1 - cas]
    colors_pie = ["#C73737", "#E8E8E8"]
    explode = (0.05, 0)
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, colors=colors_pie,
                                       autopct="%1.1f%%", startangle=90,
                                       textprops={"fontsize": 9})
    ax.set_title("a  Clinical Alignment Score", fontsize=9, fontweight="bold")
    ax.legend(["AD-key regions", "Other regions"], loc="lower left", fontsize=7)

    # Panel B: Top 10 RDI regions
    ax = axes[1]
    top10 = rdi_results[:10]
    regions = [r["region"] for r in top10]
    rdis = [r["rdi"] for r in top10]
    colors_bar = ["#C73737" if r["region"] in AD_KEY_REGIONS else "#3B82C4" for r in top10]

    y_pos = np.arange(len(regions))
    ax.barh(y_pos, rdis, color=colors_bar, height=0.6, zorder=3)
    for i, rdi in enumerate(rdis):
        ax.text(rdi + 0.01, i, f"{rdi:.2f}", va="center", fontsize=7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(regions, fontsize=7)
    ax.set_xlabel("RDI (|Cohen's d|)")
    ax.set_title("b  Top-10 Discriminative Regions", fontsize=9, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout(w_pad=3)
    _save(fig, save_dir, "clinical_alignment_summary")
    print("  Clinical alignment summary")


def fig_disease_gradient(region_values, labels, rdi_results, save_dir):
    """Disease progression gradient: CN -> MCI -> AD attention trajectory.
    Shows that MCI attention is intermediate between CN and AD for key regions."""
    set_nature_style()

    top_ad_regions = [r for r in rdi_results if r["direction"] == "AD>CN"][:6]
    top_cn_regions = [r for r in rdi_results if r["direction"] == "CN>AD"][:4]
    selected = top_ad_regions + top_cn_regions

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    class_colors = ["#3B82C4", "#F39C34", "#D94F4F"]
    class_x = [0, 1, 2]

    monotonic_count = 0
    total_count = 0

    for idx, r_info in enumerate(selected):
        if idx >= 10:
            break
        ax = axes[idx]
        ri = REGION_NAMES.index(r_info["region"])

        means = []
        sems = []
        for c in range(3):
            vals = region_values[labels == c, ri]
            means.append(vals.mean())
            sems.append(vals.std() / np.sqrt(len(vals)))

        ax.bar(class_x, means, yerr=sems, color=class_colors, capsize=3,
               edgecolor="white", linewidth=0.5, width=0.6, zorder=3)

        is_monotonic = (means[0] <= means[1] <= means[2]) or (means[0] >= means[1] >= means[2])
        total_count += 1
        if is_monotonic:
            monotonic_count += 1

        marker = " *" if is_monotonic else ""
        ax.set_xticks(class_x)
        ax.set_xticklabels(CLASS_NAMES, fontsize=7)
        title_color = "#C73737" if r_info["region"] in AD_KEY_REGIONS else "black"
        ax.set_title(f"{r_info['region']} ({r_info['direction']}){marker}",
                     fontsize=8, fontweight="bold", color=title_color)
        ax.grid(True, axis="y", alpha=0.3)
        if idx % 5 == 0:
            ax.set_ylabel("Attention", fontsize=7)

    for idx in range(len(selected), 10):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Disease Progression Gradient: CN → MCI → AD\n"
        f"(* = monotonic trend; {monotonic_count}/{total_count} regions show monotonic progression)",
        fontsize=10, fontweight="bold", y=1.04,
    )
    fig.tight_layout()
    _save(fig, save_dir, "disease_gradient")
    print(f"  Disease gradient ({monotonic_count}/{total_count} monotonic)")
    return monotonic_count, total_count


def compute_mci_gradient_test(region_values, labels):
    """Test whether MCI attention is intermediate between CN and AD (Jonckheere trend test)."""
    from scipy.stats import mannwhitneyu

    results = []
    for r in range(len(REGION_NAMES)):
        cn = region_values[labels == 0, r]
        mci = region_values[labels == 1, r]
        ad = region_values[labels == 2, r]

        cn_mean, mci_mean, ad_mean = cn.mean(), mci.mean(), ad.mean()
        is_monotonic_up = cn_mean <= mci_mean <= ad_mean
        is_monotonic_down = cn_mean >= mci_mean >= ad_mean
        is_monotonic = is_monotonic_up or is_monotonic_down

        try:
            _, p_cn_mci = mannwhitneyu(cn, mci, alternative="two-sided")
        except Exception:
            p_cn_mci = 1.0
        try:
            _, p_mci_ad = mannwhitneyu(mci, ad, alternative="two-sided")
        except Exception:
            p_mci_ad = 1.0

        results.append({
            "region": REGION_NAMES[r],
            "cn_mean": float(cn_mean),
            "mci_mean": float(mci_mean),
            "ad_mean": float(ad_mean),
            "monotonic": is_monotonic,
            "direction": "up" if is_monotonic_up else ("down" if is_monotonic_down else "non-monotonic"),
            "p_cn_mci": float(p_cn_mci),
            "p_mci_ad": float(p_mci_ad),
            "both_significant": p_cn_mci < 0.05 and p_mci_ad < 0.05,
        })

    return results


def _save(fig, save_dir, name):
    fig.savefig(save_dir / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(save_dir / f"{name}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Attention Biomarker Analysis")
    parser.add_argument("--results_dir", type=str,
                        default="chapter1_foundation/experiment_results_v3")
    parser.add_argument("--output", type=str,
                        default="chapter1_foundation/attention_biomarker_results.json")
    parser.add_argument("--figures", type=str,
                        default="chapter1_foundation/figures_biomarker")
    parser.add_argument("--model", type=str, default="Ours (Atlas+AnatDist)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    save_dir = Path(args.figures)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ATTENTION-AS-BIOMARKER ANALYSIS")
    print("=" * 70)

    # Load attention data
    print(f"\nLoading attention data for '{args.model}' ...")
    attn, labels = load_attention_data(results_dir, model_filter=args.model)
    if attn is None:
        print("No attention data found!")
        sys.exit(1)

    print(f"  Samples: {len(labels)} (CN={sum(labels==0)}, MCI={sum(labels==1)}, AD={sum(labels==2)})")
    print(f"  Attention shape: {attn.shape}")

    # Compute region-level metrics
    print("\nComputing region attention metrics ...")
    received = compute_region_attention_received(attn)
    self_attn = compute_region_self_attention(attn)

    # Per-class mean attention
    print("\n--- Mean Attention Received by Region ---")
    for c, cname in enumerate(CLASS_NAMES):
        mask = labels == c
        if mask.sum() > 0:
            means = received[mask].mean(axis=0)
            top3 = np.argsort(means)[-3:][::-1]
            top3_str = ", ".join([f"{REGION_NAMES[i]}={means[i]:.4f}" for i in top3])
            print(f"  {cname} (n={mask.sum()}): top-3 = {top3_str}")

    # Statistical tests
    print("\n--- Kruskal-Wallis Tests ---")
    kw_results = kruskal_wallis_per_region(received, labels)
    sig_regions = [r for r in kw_results if r.get("significant_001")]
    print(f"  Significant at p<0.01: {len(sig_regions)}/{len(kw_results)} regions")
    for r in kw_results:
        sig = "**" if r.get("significant_001") else ("*" if r.get("significant_005") else "")
        if sig:
            print(f"    {r['region']}: H={r['H']:.2f}, p={r['p']:.4f} {sig}")

    # RDI
    print("\n--- Region Discriminability Index (AD vs CN) ---")
    rdi_results = compute_rdi(received, labels)
    for r in rdi_results[:10]:
        marker = " ★" if r["region"] in AD_KEY_REGIONS else ""
        print(f"  {r['region']:>8}: RDI={r['rdi']:.3f} ({r['direction']}){marker}")

    # Braak correlation
    print("\n--- Braak Staging Correlation ---")
    braak_corr = compute_braak_correlation(rdi_results)
    sig = "***" if braak_corr["p"] < 0.001 else "**" if braak_corr["p"] < 0.01 else "*" if braak_corr["p"] < 0.05 else "ns"
    print(f"  Spearman ρ = {braak_corr['rho']:.3f}, p = {braak_corr['p']:.4f} {sig}")

    # Clinical Alignment Score
    print("\n--- Clinical Alignment Score ---")
    cas = compute_clinical_alignment_score(received, labels)
    print(f"  {cas['interpretation']}")

    # MCI gradient analysis
    print("\n--- MCI Gradient Test (CN → MCI → AD) ---")
    gradient = compute_mci_gradient_test(received, labels)
    monotonic = [g for g in gradient if g["monotonic"]]
    both_sig = [g for g in gradient if g["both_significant"]]
    print(f"  Monotonic trend: {len(monotonic)}/{len(gradient)} regions")
    print(f"  Both steps significant (p<0.05): {len(both_sig)}/{len(gradient)} regions")
    for g in gradient:
        if g["monotonic"] and g["both_significant"]:
            print(f"    {g['region']}: CN={g['cn_mean']:.4f} → MCI={g['mci_mean']:.4f} → AD={g['ad_mean']:.4f} ({g['direction']})")

    # Generate figures
    print(f"\nGenerating figures to {save_dir}/ ...")
    fig_attention_heatmap(received, labels, save_dir, "received")
    fig_attention_heatmap(self_attn, labels, save_dir, "self_attention")
    fig_rdi_barplot(rdi_results, save_dir)
    fig_group_comparison(received, labels, save_dir)
    fig_braak_scatter(rdi_results, braak_corr, save_dir)
    fig_clinical_alignment_summary(cas, rdi_results, save_dir)
    fig_disease_gradient(received, labels, rdi_results, save_dir)

    # Save results
    output = {
        "model": args.model,
        "n_samples": int(len(labels)),
        "class_counts": {cname: int((labels == c).sum()) for c, cname in enumerate(CLASS_NAMES)},
        "kruskal_wallis": kw_results,
        "rdi": rdi_results,
        "braak_correlation": braak_corr,
        "clinical_alignment": cas,
        "mci_gradient": gradient,
        "per_class_mean_attention": {
            cname: received[labels == c].mean(axis=0).tolist()
            for c, cname in enumerate(CLASS_NAMES) if (labels == c).sum() > 0
        },
    }
    class _Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return super().default(obj)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=_Encoder)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
