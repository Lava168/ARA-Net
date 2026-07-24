#!/usr/bin/env python3
"""Extract v4 region-readout biomarkers and validate CAS alternatives.

This script loads a v4 checkpoint, exports per-scan region readouts, and
computes reviewer-facing sanity checks:
- CAS with a uniform-null reference and bootstrap CI.
- Disease-gradient monotonicity over CN/MCI/AD.
- Correlation with available biomarker proxy columns when supplied later.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_v4_external_generalization import (
    ARANetV4,
    CLASS_NAMES,
    ManifestDataset,
    read_manifest,
    rows_for_splits,
)


REGION_NAMES = [
    "L-WM", "L-Cortex", "L-Lat-Ventricle", "L-Thalamus", "L-Caudate",
    "L-Putamen", "L-Pallidum", "Brain-Stem", "L-Hippocampus", "L-Amygdala",
    "L-Accumbens", "R-WM", "R-Cortex", "R-Lat-Ventricle", "R-Thalamus",
    "R-Caudate", "R-Putamen", "R-Pallidum", "R-Hippocampus", "R-Amygdala",
    "R-Accumbens",
]
AD_KEY_REGIONS = [
    "L-Hippocampus",
    "R-Hippocampus",
    "L-Amygdala",
    "R-Amygdala",
    "L-Lat-Ventricle",
    "R-Lat-Ventricle",
]
AD_KEY_INDEX = [REGION_NAMES.index(name) for name in AD_KEY_REGIONS]


def parse_split_list(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


@torch.no_grad()
def collect_readouts(
    checkpoint_path: Path,
    manifest_path: Path,
    splits: Sequence[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> List[dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    model = ARANetV4(
        base_channels=int(args.get("base_channels", 32)),
        feature_dim=int(args.get("feature_dim", 160)),
        encoder_max_channels=int(args.get("encoder_max_channels", 192)),
        num_heads=int(args.get("num_heads", 4)),
        num_attn_layers=int(args.get("num_attn_layers", 2)),
        dropout=float(args.get("dropout", 0.25)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows = read_manifest(manifest_path)
    wanted_rows = rows_for_splits(rows, splits)
    ds = ManifestDataset(wanted_rows, augment=False, cache_images=False)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    out_rows: List[dict] = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        seg = batch["segmentation"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            out = model(image, seg, return_attention=True)
            probs = torch.softmax(out["logits"].float(), dim=1).cpu().numpy()
            readout = out["region_readout"].float().cpu().numpy()
        labels = batch["label"].numpy().astype(int)
        for i in range(len(labels)):
            row = {
                "dataset": batch["dataset"][i],
                "split": batch["split"][i],
                "subject_id": batch["subject_id"][i],
                "scan_id": batch["scan_id"][i],
                "label": int(labels[i]),
                "label_name": CLASS_NAMES[int(labels[i])],
                "pred": int(probs[i].argmax()),
                "pred_name": CLASS_NAMES[int(probs[i].argmax())],
                "prob_CN": float(probs[i, 0]),
                "prob_MCI": float(probs[i, 1]),
                "prob_AD": float(probs[i, 2]),
            }
            for ridx, name in enumerate(REGION_NAMES):
                row[f"readout_{name}"] = float(readout[i, ridx])
            out_rows.append(row)
    return out_rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> List[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    stats = []
    for _ in range(n_boot):
        sample = values[rng.integers(0, len(values), len(values))]
        stats.append(float(np.nanmean(sample)))
    return [float(np.nanpercentile(stats, 2.5)), float(np.nanpercentile(stats, 97.5))]


def compute_cas(rows: Sequence[dict], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    by_label: Dict[int, List[np.ndarray]] = defaultdict(list)
    for row in rows:
        vec = np.array([float(row[f"readout_{name}"]) for name in REGION_NAMES], dtype=float)
        by_label[int(row["label"])].append(vec)
    if 0 not in by_label or 2 not in by_label:
        return {"error": "CAS requires both CN and AD labels"}
    cn = np.stack(by_label[0], axis=0)
    ad = np.stack(by_label[2], axis=0)
    delta = np.abs(ad.mean(axis=0) - cn.mean(axis=0))
    total = float(delta.sum())
    cas = float(delta[AD_KEY_INDEX].sum() / total) if total > 0 else float("nan")
    uniform_null = len(AD_KEY_INDEX) / len(REGION_NAMES)

    per_boot = []
    for _ in range(2000):
        cn_s = cn[rng.integers(0, len(cn), len(cn))]
        ad_s = ad[rng.integers(0, len(ad), len(ad))]
        d = np.abs(ad_s.mean(axis=0) - cn_s.mean(axis=0))
        per_boot.append(float(d[AD_KEY_INDEX].sum() / d.sum()) if d.sum() > 0 else float("nan"))
    boot = np.array(per_boot, dtype=float)
    perm_values = []
    all_vec = np.concatenate([cn, ad], axis=0)
    labels = np.array([0] * len(cn) + [2] * len(ad))
    for _ in range(2000):
        shuffled = rng.permutation(labels)
        c = all_vec[shuffled == 0]
        a = all_vec[shuffled == 2]
        d = np.abs(a.mean(axis=0) - c.mean(axis=0))
        perm_values.append(float(d[AD_KEY_INDEX].sum() / d.sum()) if d.sum() > 0 else float("nan"))
    perm = np.array(perm_values, dtype=float)
    return {
        "cas_ad_key": cas,
        "uniform_null": uniform_null,
        "cas_minus_uniform": float(cas - uniform_null),
        "bootstrap_ci": [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))],
        "permutation_p_greater": float((np.nansum(perm >= cas) + 1) / (np.isfinite(perm).sum() + 1)),
        "ad_key_regions": AD_KEY_REGIONS,
        "region_delta_rank": [
            {"region": REGION_NAMES[i], "delta": float(delta[i])}
            for i in np.argsort(-delta)
        ],
        "n_CN": int(len(cn)),
        "n_AD": int(len(ad)),
    }


def monotonic_gradient(rows: Sequence[dict]) -> dict:
    out = {}
    for ridx, name in enumerate(REGION_NAMES):
        means = []
        for label in range(3):
            vals = [float(row[f"readout_{name}"]) for row in rows if int(row["label"]) == label]
            means.append(float(np.mean(vals)) if vals else float("nan"))
        out[name] = {
            "CN": means[0],
            "MCI": means[1],
            "AD": means[2],
            "monotonic_increasing": bool(np.isfinite(means).all() and means[0] <= means[1] <= means[2]),
            "monotonic_decreasing": bool(np.isfinite(means).all() and means[0] >= means[1] >= means[2]),
            "range_AD_minus_CN": float(means[2] - means[0]) if np.isfinite([means[0], means[2]]).all() else float("nan"),
        }
    key_hits = {
        name: out[name]
        for name in AD_KEY_REGIONS
    }
    return {
        "regions": out,
        "ad_key_regions": key_hits,
        "n_monotonic_any": int(sum(v["monotonic_increasing"] or v["monotonic_decreasing"] for v in out.values())),
    }


def summarize(rows: Sequence[dict], seed: int) -> dict:
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    return {
        split: {
            "n": len(split_rows),
            "label_counts": dict(Counter(row["label_name"] for row in split_rows)),
            "cas": compute_cas(split_rows, seed),
            "gradient": monotonic_gradient(split_rows),
        }
        for split, split_rows in sorted(by_split.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", default="val,internal_test,aibl_heldout,oasis_external")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    rows = collect_readouts(
        args.checkpoint,
        args.manifest,
        parse_split_list(args.splits),
        device,
        args.batch_size,
        args.num_workers,
        args.amp,
    )
    write_csv(args.output_csv, rows)
    summary = summarize(rows, args.seed)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[saved] {args.output_csv}")
    print(f"[saved] {args.output_json}")


if __name__ == "__main__":
    main()
