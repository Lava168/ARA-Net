#!/usr/bin/env python3
"""Biomarker validation for atlas-feature baselines.

Uses the extracted v4 atlas-feature cache to test whether AD-key regions
carry disease signal beyond chance and whether region statistics show
CN->MCI->AD gradients. This is a transparent CAS alternative for reviewer
concerns about attention metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import kruskal, spearmanr

from train_atlas_feature_baseline import FEATURE_NAMES, REGION_NAMES, AD_KEY_REGIONS


def read_rows(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = int(row["label"])
    return rows


def bootstrap_ci(values: Sequence[float], seed: int, n_boot: int = 3000) -> List[float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return [float("nan"), float("nan")]
    stats = []
    for _ in range(n_boot):
        sample = arr[rng.integers(0, len(arr), len(arr))]
        stats.append(float(np.mean(sample)))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def region_feature_matrix(rows: Sequence[dict], feature_prefix: str = "vol") -> np.ndarray:
    return np.array(
        [[float(row[f"{feature_prefix}_{region}"]) for region in REGION_NAMES] for row in rows],
        dtype=float,
    )


def ad_key_score(rows: Sequence[dict], prefix: str, seed: int) -> dict:
    x = region_feature_matrix(rows, prefix)
    y = np.array([int(row["label"]) for row in rows], dtype=int)
    ad_key_idx = [REGION_NAMES.index(region) for region in AD_KEY_REGIONS]
    cn = x[y == 0]
    ad = x[y == 2]
    if len(cn) == 0 or len(ad) == 0:
        return {"error": "requires CN and AD"}
    delta = np.abs(np.nanmean(ad, axis=0) - np.nanmean(cn, axis=0))
    score = float(delta[ad_key_idx].sum() / (delta.sum() + 1e-12))
    uniform = len(ad_key_idx) / len(REGION_NAMES)
    rng = np.random.default_rng(seed)
    perm = []
    all_x = np.concatenate([cn, ad], axis=0)
    labels = np.array([0] * len(cn) + [2] * len(ad))
    for _ in range(3000):
        s = rng.permutation(labels)
        d = np.abs(np.nanmean(all_x[s == 2], axis=0) - np.nanmean(all_x[s == 0], axis=0))
        perm.append(float(d[ad_key_idx].sum() / (d.sum() + 1e-12)))
    boot = []
    for _ in range(3000):
        c = cn[rng.integers(0, len(cn), len(cn))]
        a = ad[rng.integers(0, len(ad), len(ad))]
        d = np.abs(np.nanmean(a, axis=0) - np.nanmean(c, axis=0))
        boot.append(float(d[ad_key_idx].sum() / (d.sum() + 1e-12)))
    return {
        "feature_prefix": prefix,
        "ad_key_score": score,
        "uniform_null": uniform,
        "score_minus_uniform": float(score - uniform),
        "bootstrap_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "permutation_p_greater": float((np.sum(np.array(perm) >= score) + 1) / (len(perm) + 1)),
        "region_delta_rank": [
            {"region": REGION_NAMES[i], "delta": float(delta[i]), "ad_key": REGION_NAMES[i] in AD_KEY_REGIONS}
            for i in np.argsort(-delta)
        ],
        "n_CN": int(len(cn)),
        "n_AD": int(len(ad)),
    }


def region_gradient_tests(rows: Sequence[dict], prefix: str) -> dict:
    y = np.array([int(row["label"]) for row in rows], dtype=int)
    out = {}
    for region in REGION_NAMES:
        vals = np.array([float(row[f"{prefix}_{region}"]) for row in rows], dtype=float)
        means = []
        groups = []
        for label in range(3):
            group = vals[y == label]
            group = group[np.isfinite(group)]
            groups.append(group)
            means.append(float(np.mean(group)) if len(group) else float("nan"))
        try:
            kw_stat, kw_p = kruskal(*groups) if all(len(g) for g in groups) else (float("nan"), float("nan"))
        except Exception:
            kw_stat, kw_p = float("nan"), float("nan")
        rho, sp_p = spearmanr(y, vals, nan_policy="omit")
        out[region] = {
            "CN": means[0],
            "MCI": means[1],
            "AD": means[2],
            "AD_minus_CN": float(means[2] - means[0]) if np.isfinite([means[0], means[2]]).all() else float("nan"),
            "spearman_rho_label": float(rho) if np.isfinite(rho) else None,
            "spearman_p": float(sp_p) if np.isfinite(sp_p) else None,
            "kruskal_p": float(kw_p) if np.isfinite(kw_p) else None,
            "ad_key": region in AD_KEY_REGIONS,
        }
    return {
        "prefix": prefix,
        "regions": out,
        "top_by_abs_spearman": sorted(
            [
                {"region": r, **v}
                for r, v in out.items()
                if v["spearman_rho_label"] is not None
            ],
            key=lambda item: abs(item["spearman_rho_label"]),
            reverse=True,
        ),
        "ad_key": {r: out[r] for r in AD_KEY_REGIONS},
    }


def summarize(rows: Sequence[dict], seed: int) -> dict:
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    selected = {
        "adni_val_internal_test": by_split["val"] + by_split["internal_test"],
        "aibl_adapt_heldout": by_split["aibl_adapt_val"] + by_split["aibl_heldout"],
        "aibl_heldout": by_split["aibl_heldout"],
        "all_labeled_ad": [row for row in rows if row["split"] != "ixi_external"],
    }
    summary = {}
    for name, split_rows in selected.items():
        if not split_rows:
            continue
        summary[name] = {
            "n": len(split_rows),
            "label_counts": dict(Counter(row["label_name"] for row in split_rows)),
            "ad_key_volume_score": ad_key_score(split_rows, "vol", seed),
            "ad_key_mean_intensity_score": ad_key_score(split_rows, "mean", seed + 1),
            "volume_gradients": region_gradient_tests(split_rows, "vol"),
            "mean_intensity_gradients": region_gradient_tests(split_rows, "mean"),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = read_rows(args.feature_csv)
    summary = summarize(rows, args.seed)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[saved] {args.output_json}")
    for name, item in summary.items():
        s = item["ad_key_volume_score"]
        print(name, "n", item["n"], "volume_score", s.get("ad_key_score"), "minus_uniform", s.get("score_minus_uniform"), "p", s.get("permutation_p_greater"))


if __name__ == "__main__":
    main()
