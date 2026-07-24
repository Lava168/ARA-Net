#!/usr/bin/env python3
"""Audit AD-related data assets for ARA-Net rescue experiments.

The script is intentionally read-only. It summarizes candidate cache
directories, sampling NPZ keys/shapes/labels without loading entire datasets
into memory.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOTS = [
    "/home/lry/atlas_guided_attention Alzheimer's Disease Dynamics/chapter1_foundation/sample_data/cache_real",
    "/home/lry/aibl/cache_real",
    "/home/lry/atlas_guided_attention Alzheimer’s Disease Dynamics/chapter2_disentangle/data/aibl_cache_128_adni_norm",
    "/home/lry/atlas_guided_attention Alzheimer’s Disease Dynamics/chapter2_disentangle/data/oasis_cache_128",
    "/home/lry/atlas_guided_attention Alzheimer’s Disease Dynamics/chapter2_disentangle/data/oasis_cache_128_v2",
    "/home/lry/atlas_guided_attention Alzheimer’s Disease Dynamics/chapter2_disentangle/data/full_cache",
    "/home/lry/atlas_guided_attention Alzheimer’s Disease Dynamics/chapter2_disentangle/data/mvp_cache",
    "/home/lry/atlas_guided_attention Alzheimer’s Disease Dynamics/chapter2_disentangle/data/flair_cache",
    "/home/lry/atlas_guided_attention Alzheimer’s Disease Dynamics/chapter2_disentangle/data/pet_cache",
    "/home/lry/atlas_guided_attention Alzheimer’s Disease Dynamics/chapter2_disentangle/data/regional_suvr_cache",
    "/home/lry/atlas_guided_attention Alzheimer’s Disease Dynamics/sample_data/nacc",
    "/home/lry/atlas_guided_attention Alzheimer’s Disease Dynamics/sample_data/nacc_partial_analysis",
    "/home/lry/adni_flair",
    "/home/lry/adni_clinical_2026",
    "/home/lry/atlas_guided_attention Alzheimer’s Disease Dynamics/chapter4_generalization/data",
]


def scalar_value(value: np.ndarray) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, (np.integer, np.floating)):
            return item.item()
        return item
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "preview": str(arr.ravel()[:5])[:120],
    }


def audit_npz_dir(root: Path) -> dict[str, Any]:
    files = sorted(root.glob("*.npz"))
    result: dict[str, Any] = {
        "path": str(root),
        "exists": root.exists(),
        "npz_count": len(files),
        "file_count_top_level": len([p for p in root.iterdir()]) if root.exists() and root.is_dir() else 0,
        "examples": [],
        "label_counts": {},
        "prefix_counts": {},
        "keyset_counts": [],
        "array_shape_counts": [],
        "errors": 0,
    }
    if not files:
        if root.exists() and root.is_dir():
            result["top_entries"] = [p.name for p in list(root.iterdir())[:25]]
        return result

    label_counts: Counter[str] = Counter()
    prefix_counts: Counter[str] = Counter()
    keyset_counts: Counter[tuple[str, ...]] = Counter()
    shape_counts: Counter[tuple[tuple[str, tuple[int, ...], str], ...]] = Counter()

    sample_indices = sorted(set([0, 1, len(files) // 2, max(0, len(files) - 2), len(files) - 1]))
    sample_files = [files[i] for i in sample_indices if 0 <= i < len(files)]

    for f in files:
        prefix_counts[f.name.split("_")[0]] += 1

    # Full label/key scan is cheap for these compressed caches, but cap just in case.
    for f in files[: min(len(files), 5000)]:
        try:
            with np.load(f, allow_pickle=True) as z:
                keys = tuple(sorted(z.keys()))
                keyset_counts[keys] += 1
                if "label" in z:
                    label = str(int(np.asarray(z["label"]).item()))
                elif "y" in z:
                    label = str(int(np.asarray(z["y"]).item()))
                elif "dx" in z:
                    label = str(np.asarray(z["dx"]).item())
                else:
                    label = "NA"
                label_counts[label] += 1
                shapes = []
                for k in keys:
                    arr = np.asarray(z[k])
                    if arr.ndim >= 2:
                        shapes.append((k, tuple(int(x) for x in arr.shape), str(arr.dtype)))
                shape_counts[tuple(shapes)] += 1
        except Exception:
            result["errors"] += 1

    for f in sample_files:
        example: dict[str, Any] = {"file": f.name, "size": f.stat().st_size}
        try:
            with np.load(f, allow_pickle=True) as z:
                example["keys"] = list(z.keys())
                arrays = {}
                meta = {}
                for k in z.keys():
                    arr = np.asarray(z[k])
                    if arr.ndim >= 2:
                        arrays[k] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
                    else:
                        meta[k] = scalar_value(arr)
                example["arrays"] = arrays
                example["meta"] = meta
        except Exception as exc:
            example["error"] = repr(exc)
        result["examples"].append(example)

    result["label_counts"] = dict(label_counts)
    result["prefix_counts"] = dict(prefix_counts.most_common(20))
    result["keyset_counts"] = [
        {"keys": list(keys), "count": count}
        for keys, count in keyset_counts.most_common(8)
    ]
    result["array_shape_counts"] = [
        {
            "count": count,
            "arrays": [
                {"key": key, "shape": list(shape), "dtype": dtype}
                for key, shape, dtype in shapes
            ],
        }
        for shapes, count in shape_counts.most_common(8)
    ]
    return result


def audit_tree(root: Path) -> dict[str, Any]:
    result = audit_npz_dir(root)
    if root.exists() and root.is_dir():
        interesting = []
        for pattern in ("*.csv", "*.xlsx", "*.json", "*.pth", "*.pt", "*.nii", "*.nii.gz", "*.mgz"):
            for p in sorted(root.glob(pattern))[:20]:
                interesting.append({"file": p.name, "size": p.stat().st_size})
        result["interesting_files"] = interesting[:80]
    return result


def main() -> None:
    summary = [audit_tree(Path(root)) for root in ROOTS]
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
