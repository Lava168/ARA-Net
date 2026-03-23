#!/usr/bin/env python3
"""
Error-conditioned interpretability analysis for ARA-Net.

Goal (addresses reviewer concern):
  Quantify whether attention remains anatomically meaningful on misclassified cases,
  e.g., when MCI is misclassified as CN, does attention still concentrate on AD-key regions?

Data source:
  chapter1_foundation/experiment_results_v3/seed_*/all_results.json

Important implementation detail:
  Each run stores attention for only the first `max_samples` test samples
  (see run_experiment_v3.collect_attention(..., max_samples=50)).
  Therefore we align attention samples with the first N entries of test_y_true/test_y_pred.

Outputs:
  - JSON summary + per-sample CSV
  - Figure: heatmaps for CAS / Hit@K by (true, pred)
  - Figure: key error mode bar plot (default focuses on MCI->CN and MCI->AD)
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots


CLASS_NAMES = ["CN", "MCI", "AD"]
REGION_NAMES = [
    "L-WM", "L-Ctx", "L-Vent", "L-Thal", "L-Caud",
    "L-Put", "L-Pall", "BStem", "L-Hipp", "L-Amyg",
    "L-Acc", "R-WM", "R-Ctx", "R-Vent", "R-Thal",
    "R-Caud", "R-Put", "R-Pall", "R-Hipp", "R-Amyg", "R-Acc",
]
AD_KEY_REGIONS = {"L-Hipp", "R-Hipp", "L-Amyg", "R-Amyg", "L-Vent", "R-Vent"}
AD_KEY_IDX = np.array([REGION_NAMES.index(r) for r in sorted(AD_KEY_REGIONS)], dtype=int)


def set_nature_style():
    plt.style.use(['science', 'nature', 'no-latex'])
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig: plt.Figure, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{name}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_received(attn: np.ndarray) -> np.ndarray:
    """
    attn: (N, H, R, R) -> received per region (N, R)
    received[k] = sum over query regions of A_{query, k}
    """
    attn = np.asarray(attn, dtype=float)
    if attn.ndim != 4:
        raise ValueError(f"Expected attention with shape (N,H,R,R), got {attn.shape}")
    attn_mean = attn.mean(axis=1)  # (N, R, R)
    received = attn_mean.sum(axis=1)  # (N, R)
    return received


def compute_cas(received: np.ndarray) -> np.ndarray:
    denom = received.sum(axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    cas = received[:, AD_KEY_IDX].sum(axis=1, keepdims=True) / denom
    return cas[:, 0]


def compute_hit_at_k(received: np.ndarray, k: int) -> np.ndarray:
    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive")
    topk = np.argsort(-received, axis=1)[:, :k]  # (N, k)
    hits = np.isin(topk, AD_KEY_IDX).sum(axis=1) / float(k)
    return hits.astype(float)


@dataclass
class SampleRow:
    run_key: str
    true: int
    pred: int
    cas: float
    hitk: float
    cos_to_cn: float
    cos_to_mci: float
    cos_to_ad: float


def load_runs(results_dir: Path, model_prefix: str) -> List[Tuple[str, dict]]:
    runs = []
    for seed_dir in sorted(results_dir.glob("seed_*")):
        f = seed_dir / "all_results.json"
        if not f.exists():
            continue
        with open(f) as fp:
            d = json.load(fp)
        for k, r in d.items():
            if not k.startswith(model_prefix):
                continue
            if not r.get("attention_maps"):
                continue
            runs.append((k, r))
    return runs


def build_samples(runs: List[Tuple[str, dict]], k_hit: int) -> List[SampleRow]:
    rows: List[SampleRow] = []

    # First pass: collect per-sample vectors + labels + preds for prototype building.
    vecs_by_true_pred: List[Tuple[np.ndarray, int, int, str]] = []

    for run_key, r in runs:
        attn = np.array(r["attention_maps"], dtype=float)  # (N, H, R, R)
        n = attn.shape[0]
        y_true = np.array(r["test_y_true"][:n], dtype=int)
        # Prefer stored y_pred (already aligned to evaluate() order)
        if "test_y_pred" in r and r["test_y_pred"]:
            y_pred = np.array(r["test_y_pred"][:n], dtype=int)
        else:
            y_prob = np.array(r["test_y_prob"][:n], dtype=float)
            y_pred = y_prob.argmax(axis=1)

        received = compute_received(attn)  # (n, R)
        # L1-normalize the vector before cosine for stability/comparability.
        denom = np.clip(received.sum(axis=1, keepdims=True), 1e-12, None)
        vec = received / denom

        for i in range(n):
            vecs_by_true_pred.append((vec[i], int(y_true[i]), int(y_pred[i]), run_key))

    # Prototypes from correctly predicted samples; fallback to all samples of the class.
    proto = {}
    for c in range(3):
        correct = [v for (v, yt, yp, _) in vecs_by_true_pred if yt == c and yp == c]
        if len(correct) >= 10:
            proto[c] = np.mean(np.stack(correct, axis=0), axis=0)
        else:
            allc = [v for (v, yt, _, _) in vecs_by_true_pred if yt == c]
            proto[c] = np.mean(np.stack(allc, axis=0), axis=0) if allc else np.zeros((len(REGION_NAMES),))

    # Second pass: compute metrics per sample.
    for v, yt, yp, run_key in vecs_by_true_pred:
        cas = float(v[AD_KEY_IDX].sum())
        hitk = float(np.isin(np.argsort(-v)[:k_hit], AD_KEY_IDX).sum() / float(k_hit))
        rows.append(SampleRow(
            run_key=run_key,
            true=yt,
            pred=yp,
            cas=cas,
            hitk=hitk,
            cos_to_cn=_cosine(v, proto[0]),
            cos_to_mci=_cosine(v, proto[1]),
            cos_to_ad=_cosine(v, proto[2]),
        ))

    return rows


def summarize(rows: List[SampleRow]) -> Dict:
    by_tp = defaultdict(list)
    for r in rows:
        by_tp[(r.true, r.pred)].append(r)

    out = {
        "n_total_samples": len(rows),
        "metrics_by_true_pred": {},
    }
    for (t, p), lst in sorted(by_tp.items()):
        cas = np.array([x.cas for x in lst], dtype=float)
        hitk = np.array([x.hitk for x in lst], dtype=float)
        cos = np.array([[x.cos_to_cn, x.cos_to_mci, x.cos_to_ad] for x in lst], dtype=float)
        out["metrics_by_true_pred"][f"{CLASS_NAMES[t]}->{CLASS_NAMES[p]}"] = {
            "n": int(len(lst)),
            "cas_mean": float(cas.mean()),
            "cas_std": float(cas.std(ddof=0)),
            "hitk_mean": float(hitk.mean()),
            "hitk_std": float(hitk.std(ddof=0)),
            "cos_mean": {CLASS_NAMES[i]: float(cos[:, i].mean()) for i in range(3)},
            "cos_std": {CLASS_NAMES[i]: float(cos[:, i].std(ddof=0)) for i in range(3)},
        }
    return out


def write_csv(rows: List[SampleRow], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["run_key", "true", "pred", "cas", "hitk", "cos_to_cn", "cos_to_mci", "cos_to_ad"]
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(
                f"{r.run_key},{r.true},{r.pred},"
                f"{r.cas:.6f},{r.hitk:.6f},{r.cos_to_cn:.6f},{r.cos_to_mci:.6f},{r.cos_to_ad:.6f}\n"
            )


def plot_heatmaps(summary: Dict, out_dir: Path, k_hit: int):
    set_nature_style()
    cas_mat = np.full((3, 3), np.nan, dtype=float)
    hit_mat = np.full((3, 3), np.nan, dtype=float)
    n_mat = np.zeros((3, 3), dtype=int)

    for key, v in summary["metrics_by_true_pred"].items():
        tname, pname = key.split("->")
        t = CLASS_NAMES.index(tname)
        p = CLASS_NAMES.index(pname)
        cas_mat[t, p] = v["cas_mean"]
        hit_mat[t, p] = v["hitk_mean"]
        n_mat[t, p] = v["n"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, mat, title in [
        (axes[0], cas_mat, "Error-conditioned CAS"),
        (axes[1], hit_mat, f"Error-conditioned Hit@{k_hit} (AD-key)"),
    ]:
        im = ax.imshow(mat, vmin=np.nanmin(mat), vmax=np.nanmax(mat), cmap="RdBu_r")
        ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title, fontweight="bold")
        for i in range(3):
            for j in range(3):
                if math.isnan(mat[i, j]):
                    txt = "—"
                else:
                    txt = f"{mat[i,j]:.3f}\n(n={n_mat[i,j]})"
                mid = (np.nanmin(mat) + np.nanmax(mat)) / 2
                use_white = not math.isnan(mat[i, j]) and abs(mat[i, j] - mid) > (np.nanmax(mat) - np.nanmin(mat)) * 0.3
                ax.text(j, i, txt, ha="center", va="center", color="white" if use_white else "black", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Interpretability under correct vs misclassified cases", y=1.02, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "FigS_error_conditioned_heatmaps")


def _sem(arr):
    """Standard error of the mean."""
    n = len(arr)
    return np.std(arr, ddof=0) / np.sqrt(n) if n > 1 else 0.0


def plot_key_error_bars(rows: List[SampleRow], out_dir: Path, k_hit: int):
    """
    Unified figure: three MCI error-mode groups, three metrics.
    Left panel: CAS & Hit@k.  Right panel: cosine to class prototypes.
    Error bars = SEM (standard error of the mean), not raw SD.
    Color palette derived from RdBu to match Fig1 attention heatmap.
    """
    set_nature_style()

    def sel(t: int, p: int):
        return [r for r in rows if r.true == t and r.pred == p]

    group_labels = ["MCI\u2192MCI\n(correct)", "MCI\u2192CN\n(error)", "MCI\u2192AD\n(error)"]
    group_data = [sel(1, 1), sel(1, 0), sel(1, 2)]
    keep = [(lbl, g) for lbl, g in zip(group_labels, group_data) if len(g) > 0]
    if not keep:
        return
    group_labels = [k[0] for k in keep]
    group_data = [k[1] for k in keep]

    cas_m, cas_se, hit_m, hit_se, n_counts = [], [], [], [], []
    for g in group_data:
        ca = np.array([x.cas for x in g])
        hi = np.array([x.hitk for x in g])
        cas_m.append(ca.mean()); cas_se.append(_sem(ca))
        hit_m.append(hi.mean()); hit_se.append(_sem(hi))
        n_counts.append(len(g))

    from matplotlib.colors import to_rgba

    CLR_CAS  = "#B2182B"
    CLR_HIT  = "#2166AC"
    CLR_COS2 = "#4393C3"
    CLR_COS3 = "#92C5DE"

    ng = len(group_labels)
    x = np.arange(ng)
    bw = 0.30

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0),
                                    gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.32})

    # ── Left panel: CAS & Hit@k ──
    ax1.bar(x - bw/2, cas_m, bw, label="CAS (AD-key share)",
            color=to_rgba(CLR_CAS, 0.85), edgecolor="white", linewidth=0.8, zorder=3)
    ax1.errorbar(x - bw/2, cas_m, yerr=cas_se, fmt="none",
                 ecolor=to_rgba(CLR_CAS, 0.6), elinewidth=1.2, capsize=4, capthick=1.0, zorder=4)

    ax1.bar(x + bw/2, hit_m, bw, label=f"Hit@{k_hit} (AD-key)",
            color=to_rgba(CLR_HIT, 0.85), edgecolor="white", linewidth=0.8, zorder=3)
    ax1.errorbar(x + bw/2, hit_m, yerr=hit_se, fmt="none",
                 ecolor=to_rgba(CLR_HIT, 0.6), elinewidth=1.2, capsize=4, capthick=1.0, zorder=4)

    lbl_offset = 0.010
    for i in range(ng):
        ax1.text(x[i] - bw/2, cas_m[i] + cas_se[i] + lbl_offset, f"{cas_m[i]:.3f}",
                 ha="center", va="bottom", fontsize=8, color=CLR_CAS, fontweight="bold")
        ax1.text(x[i] + bw/2, hit_m[i] + hit_se[i] + lbl_offset, f"{hit_m[i]:.3f}",
                 ha="center", va="bottom", fontsize=8, color=CLR_HIT, fontweight="bold")

    y_top = max(max(c + e for c, e in zip(cas_m, cas_se)),
                max(h + e for h, e in zip(hit_m, hit_se))) * 1.30
    ax1.set_ylim(0, y_top)
    ax1.set_xticks(x)
    ax1.set_xticklabels(group_labels, fontsize=9, linespacing=1.1)
    ax1.set_ylabel("Score", fontsize=10)
    ax1.set_title("(a)  CAS & Hit@5", fontsize=11, fontweight="bold", pad=10)
    for i in range(ng):
        ax1.text(x[i], -0.18, f"n = {n_counts[i]}", ha="center", va="top",
                 fontsize=7.5, color="#777777", fontstyle="italic",
                 transform=ax1.get_xaxis_transform())

    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax1.spines[sp].set_color("#CCCCCC")
    ax1.tick_params(colors="#333333")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.25, zorder=0)
    ax1.set_axisbelow(True)
    leg1 = ax1.legend(fontsize=8, loc="upper left", frameon=True,
                      framealpha=0.92, edgecolor="#CCCCCC", borderpad=0.6)
    leg1.get_frame().set_linewidth(0.5)

    # ── Right panel: Cosine to each class prototype ──
    cos_cn_m  = [np.mean([r.cos_to_cn for r in g]) for g in group_data]
    cos_mci_m = [np.mean([r.cos_to_mci for r in g]) for g in group_data]
    cos_ad_m  = [np.mean([r.cos_to_ad for r in g]) for g in group_data]
    cos_cn_se  = [_sem([r.cos_to_cn for r in g]) for g in group_data]
    cos_mci_se = [_sem([r.cos_to_mci for r in g]) for g in group_data]
    cos_ad_se  = [_sem([r.cos_to_ad for r in g]) for g in group_data]

    bw2 = 0.24
    offsets = [-bw2, 0, bw2]
    cos_data = [
        ("cos(CN proto.)",  cos_cn_m,  cos_cn_se,  CLR_COS3),
        ("cos(MCI proto.)", cos_mci_m, cos_mci_se, CLR_COS2),
        ("cos(AD proto.)",  cos_ad_m,  cos_ad_se,  CLR_CAS),
    ]
    for idx, (label, vals, ses, clr) in enumerate(cos_data):
        pos = x + offsets[idx]
        ax2.bar(pos, vals, bw2, label=label,
                color=to_rgba(clr, 0.8), edgecolor="white", linewidth=0.8, zorder=3)
        ax2.errorbar(pos, vals, yerr=ses, fmt="none",
                     ecolor=to_rgba(clr, 0.55), elinewidth=1.2, capsize=3, capthick=1.0, zorder=4)
        for p, v, se in zip(pos, vals, ses):
            ax2.text(p, v + se + 0.0008, f".{int(round(v*1000)) % 1000:03d}",
                     ha="center", va="bottom", fontsize=7, color=clr, fontweight="bold")

    all_cos = cos_cn_m + cos_mci_m + cos_ad_m
    all_ses = cos_cn_se + cos_mci_se + cos_ad_se
    lo = min(v - 3*s for v, s in zip(all_cos, all_ses))
    hi = max(v + 3*s for v, s in zip(all_cos, all_ses))
    pad = (hi - lo) * 0.25
    ax2.set_ylim(lo - pad, hi + pad * 1.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(group_labels, fontsize=9, linespacing=1.1)
    ax2.set_ylabel("Cosine Similarity", fontsize=10)
    ax2.set_title("(b)  Cosine to Class Prototypes", fontsize=11, fontweight="bold", pad=10)
    for i in range(ng):
        ax2.text(x[i], -0.18, f"n = {n_counts[i]}", ha="center", va="top",
                 fontsize=7.5, color="#777777", fontstyle="italic",
                 transform=ax2.get_xaxis_transform())

    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax2.spines[sp].set_color("#CCCCCC")
    ax2.tick_params(colors="#333333")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.25, zorder=0)
    ax2.set_axisbelow(True)
    leg2 = ax2.legend(fontsize=8, loc="lower left", frameon=True,
                      framealpha=0.92, edgecolor="#CCCCCC", borderpad=0.6)
    leg2.get_frame().set_linewidth(0.5)

    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.22, top=0.82, wspace=0.30)
    fig.suptitle("MCI Error-Mode Analysis: Interpretability Metrics",
                 fontsize=12.5, fontweight="bold", y=0.93)
    fig.text(0.5, 0.02, "Error bars = SEM (standard error of the mean)",
             ha="center", fontsize=7.5, color="#999999", fontstyle="italic")
    _save(fig, out_dir, "FigS_key_error_modes_MCI")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="chapter1_foundation/experiment_results_v3")
    ap.add_argument("--model_prefix", type=str, default="Ours (Atlas+AnatDist)")
    ap.add_argument("--hit_k", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="chapter1_foundation/figures_supplementary")
    ap.add_argument("--out_json", type=str, default="chapter1_foundation/error_conditioned_interpretability.json")
    ap.add_argument("--out_csv", type=str, default="chapter1_foundation/error_conditioned_interpretability_samples.csv")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    runs = load_runs(results_dir, args.model_prefix)
    if not runs:
        raise SystemExit(f"No runs found with attention for model_prefix={args.model_prefix!r} in {results_dir}")

    rows = build_samples(runs, k_hit=args.hit_k)
    summary = summarize(rows)
    summary["model_prefix"] = args.model_prefix
    summary["hit_k"] = args.hit_k
    summary["note"] = (
        "Attention is collected for first N test samples per run (default N=50). "
        "Metrics are computed on that aligned subset."
    )

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_csv(rows, Path(args.out_csv))

    plot_heatmaps(summary, out_dir, k_hit=args.hit_k)
    plot_key_error_bars(rows, out_dir, k_hit=args.hit_k)

    print(f"Saved JSON: {args.out_json}")
    print(f"Saved CSV:  {args.out_csv}")
    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()

