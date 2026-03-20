"""Attention Analysis for Atlas-Guided Model (ARA-Net)."""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
import torch

FREESURFER_ROI_NAMES = {
    17: "L-Hipp", 53: "R-Hipp", 18: "L-Amyg", 54: "R-Amyg",
    10: "L-Thal", 49: "R-Thal", 11: "L-Caud", 50: "R-Caud",
    12: "L-Put", 51: "R-Put", 13: "L-Pall", 52: "R-Pall",
    4: "L-Vent", 43: "R-Vent", 3: "L-Ctx", 42: "R-Ctx",
    2: "L-WM", 41: "R-WM", 16: "BStem", 26: "L-Acc", 58: "R-Acc",
}
AD_KEY_ROIS = [17, 53, 18, 54, 10, 49]

def extract_regional_attention(attention, seg_downsampled, roi_names=None):
    roi_names = roi_names or FREESURFER_ROI_NAMES
    if attention.dim() == 4:
        attn = attention.mean(dim=1)
    elif attention.dim() == 3 and seg_downsampled.dim() == 3:
        attn = attention.unsqueeze(0)
        seg_downsampled = seg_downsampled.unsqueeze(0)
    else:
        attn = attention
    B = attn.shape[0]
    seg_flat = seg_downsampled.reshape(B, -1)
    attn_received = attn.sum(dim=1)
    region_scores = defaultdict(list)
    for b in range(B):
        for label, name in roi_names.items():
            mask = seg_flat[b] == label
            if mask.any():
                region_scores[name].append(float(attn_received[b, mask].mean()))
    return {k: float(np.mean(v)) for k, v in region_scores.items()}

def attention_entropy(attention):
    attn = attention.mean(dim=1) if attention.dim() == 4 else attention
    return float(-(attn * torch.log(attn + 1e-8)).sum(dim=-1).mean())

def attention_sparsity(attention, threshold=0.01):
    attn = attention.mean(dim=1) if attention.dim() == 4 else attention
    return float((attn < threshold).float().mean())

def top_k_concentration(attention, k=10):
    attn = attention.mean(dim=1) if attention.dim() == 4 else attention
    topk = attn.topk(min(k, attn.shape[-1]), dim=-1).values
    return float((topk.sum(dim=-1) / attn.sum(dim=-1).clamp(min=1e-8)).mean())

def group_attention_comparison(attention_list, seg_list, label_list, class_names=None, roi_names=None):
    class_names = class_names or ["CN", "MCI", "AD"]
    roi_names = roi_names or FREESURFER_ROI_NAMES
    group_vals = {cn: defaultdict(list) for cn in class_names}
    for attn, seg, label in zip(attention_list, seg_list, label_list):
        ra = extract_regional_attention(attn, seg, roi_names)
        for rname, val in ra.items():
            group_vals[class_names[label]][rname].append(val)
    return {g: {r: float(np.mean(v)) for r, v in regions.items()} for g, regions in group_vals.items()}

def plot_regional_attention_comparison(group_data, title="Regional Attention by Diagnosis", figsize=(12, 6)):
    import matplotlib.pyplot as plt
    COLORS = {"CN": "#4E79A7", "MCI": "#F28E2B", "AD": "#E15759"}
    groups = list(group_data.keys())
    all_regions = sorted({r for g in group_data.values() for r in g})
    x = np.arange(len(all_regions))
    width = 0.8 / len(groups)
    fig, ax = plt.subplots(figsize=figsize)
    for i, g in enumerate(groups):
        vals = [group_data[g].get(r, 0.) for r in all_regions]
        ax.bar(x + i * width, vals, width, label=g, color=COLORS.get(g, f"C{i}"),
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x + width * (len(groups) - 1) / 2)
    ax.set_xticklabels(all_regions, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Attention")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig

def plot_attention_statistics(stats_per_group, title="Attention Statistics", figsize=(8, 5)):
    import matplotlib.pyplot as plt
    COLORS = {"CN": "#4E79A7", "MCI": "#F28E2B", "AD": "#E15759"}
    groups = list(stats_per_group.keys())
    keys = list(next(iter(stats_per_group.values())).keys())
    x = np.arange(len(keys))
    width = 0.8 / len(groups)
    fig, ax = plt.subplots(figsize=figsize)
    for i, g in enumerate(groups):
        ax.bar(x + i * width, [stats_per_group[g][m] for m in keys], width,
               label=g, color=COLORS.get(g, f"C{i}"), edgecolor="white")
    ax.set_xticks(x + width * (len(groups) - 1) / 2)
    ax.set_xticklabels(keys)
    ax.set_ylabel("Value")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
