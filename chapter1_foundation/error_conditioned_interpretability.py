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
        im = ax.imshow(mat, vmin=np.nanmin(mat), vmax=np.nanmax(mat), cmap="viridis")
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
                ax.text(j, i, txt, ha="center", va="center", color="white" if not math.isnan(mat[i,j]) and mat[i,j] > (np.nanmin(mat)+np.nanmax(mat))/2 else "black", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Interpretability under correct vs misclassified cases", y=1.02, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "FigS_error_conditioned_heatmaps")


def plot_key_error_bars(rows: List[SampleRow], out_dir: Path, k_hit: int):
    """
    Focus on clinically relevant confusions:
      - MCI->CN
      - MCI->AD
    Compare against correct MCI->MCI.
    """
    set_nature_style()

    def sel(t: int, p: int):
        return [r for r in rows if r.true == t and r.pred == p]

    groups = [
        ("MCI→MCI (correct)", sel(1, 1)),
        ("MCI→CN (error)", sel(1, 0)),
        ("MCI→AD (error)", sel(1, 2)),
    ]
    # If a group has no samples (rare), drop it.
    groups = [(name, g) for name, g in groups if len(g) > 0]
    if not groups:
        return

    cas_means = [np.mean([x.cas for x in g]) for _, g in groups]
    hit_means = [np.mean([x.hitk for x in g]) for _, g in groups]
    cos_pred_means = []
    for name, g in groups:
        # Cosine to predicted-class prototype: pick per-sample predicted label.
        c = np.mean([ [x.cos_to_cn, x.cos_to_mci, x.cos_to_ad][x.pred] for x in g ])
        cos_pred_means.append(c)

    x = np.arange(len(groups))
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0))
    axes[0].bar(x, cas_means, color="#C73737", edgecolor="white")
    axes[0].set_title("CAS (AD-key share)", fontweight="bold")
    axes[0].set_xticks(x); axes[0].set_xticklabels([n for n, _ in groups], rotation=25, ha="right")
    axes[0].set_ylim(0, max(cas_means) * 1.25)

    axes[1].bar(x, hit_means, color="#3B82C4", edgecolor="white")
    axes[1].set_title(f"Hit@{k_hit} (AD-key)", fontweight="bold")
    axes[1].set_xticks(x); axes[1].set_xticklabels([n for n, _ in groups], rotation=25, ha="right")
    axes[1].set_ylim(0, 1.0)

    axes[2].bar(x, cos_pred_means, color="#2CA6A4", edgecolor="white")
    axes[2].set_title("Cosine to predicted prototype", fontweight="bold")
    axes[2].set_xticks(x); axes[2].set_xticklabels([n for n, _ in groups], rotation=25, ha="right")
    axes[2].set_ylim(0, 1.0)

    for ax in axes:
        ax.grid(axis="y")

    fig.tight_layout()
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

