#!/usr/bin/env python3
"""
Preprocess OASIS Cross-Sectional FreeSurfer data → cache_real .npz format.

The OASIS dataset has 6 disc archives (oasis_cs_freesurfer_disc{1..5}.tar.gz).
Each contains FreeSurfer processed subjects: OAS1_xxxx_MR1/mri/{brain.mgz, aseg.mgz}.

Labels derived from CDR (Clinical Dementia Rating):
  CDR 0   → CN  (label 0)
  CDR 0.5 → MCI (label 1)
  CDR 1+  → AD  (label 2)

Usage:
    python -m chapter1_foundation.preprocess_oasis \
        --tar_dir . \
        --metadata oasis_cross-sectional-5708aa0a98d82080.xlsx \
        --output_cache sample_data/cache_real \
        --target_shape 96 112 96
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

CDR_TO_LABEL = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 2, 3.0: 2}

ASEG_REMAP = {
    2: 1, 41: 1,     # WM
    3: 2, 42: 2,     # Cortex
    4: 3, 43: 3, 14: 3, 15: 3,  # Ventricles + 3rd/4th
    10: 4, 49: 4,    # Thalamus
    11: 5, 50: 5,    # Caudate
    12: 6, 51: 6,    # Putamen
    13: 7, 52: 7,    # Pallidum
    16: 8,           # Brain-Stem
    17: 9, 53: 9,    # Hippocampus
    18: 10, 54: 10,  # Amygdala
    26: 11, 58: 11,  # Accumbens
}


def parse_args():
    p = argparse.ArgumentParser(description="OASIS FreeSurfer → cache .npz")
    p.add_argument("--tar_dir", type=Path,
                   default=Path("/home/lry/atlas_guided_attention Alzheimer's Disease Dynamics"))
    p.add_argument("--metadata", type=Path,
                   default=Path("/home/lry/atlas_guided_attention Alzheimer's Disease Dynamics/"
                                "oasis_cross-sectional-5708aa0a98d82080.xlsx"))
    p.add_argument("--output_cache", type=Path, default=Path("sample_data/cache_real"))
    p.add_argument("--target_shape", type=int, nargs=3, default=(96, 112, 96))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--extract_dir", type=Path, default=None,
                   help="Temp directory for extraction (default: /tmp/oasis_extract)")
    return p.parse_args()


def load_metadata(xlsx_path: Path) -> Dict[str, int]:
    """Load OASIS metadata and return {subject_id: label}."""
    try:
        import pandas as pd
        df = pd.read_excel(xlsx_path)
    except ImportError:
        print("pandas/openpyxl required: pip install pandas openpyxl")
        raise

    id_col = None
    cdr_col = None
    for col in df.columns:
        cl = col.strip().lower()
        if "id" in cl and id_col is None:
            id_col = col
        if "cdr" in cl:
            cdr_col = col

    if id_col is None or cdr_col is None:
        cols_str = ", ".join(df.columns.tolist())
        raise ValueError(f"Cannot find ID/CDR columns in: {cols_str}")

    labels = {}
    for _, row in df.iterrows():
        sid = str(row[id_col]).strip()
        cdr = row[cdr_col]
        if isinstance(cdr, str):
            try:
                cdr = float(cdr)
            except ValueError:
                continue
        if np.isnan(cdr):
            continue
        label = CDR_TO_LABEL.get(cdr)
        if label is not None:
            base_id = re.match(r"(OAS1_\d+)_MR\d+", sid)
            if base_id:
                labels[base_id.group(1)] = label
            labels[sid] = label

    return labels


def normalize_image(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    vals = img[np.isfinite(img)]
    if vals.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            return np.zeros_like(img, dtype=np.float32)
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-8)
    return np.clip(img, 0, 1).astype(np.float32)


def resize(vol: np.ndarray, target_shape, order: int) -> np.ndarray:
    factors = [t / s for t, s in zip(target_shape, vol.shape)]
    return zoom(vol, factors, order=order)


def remap_aseg(seg: np.ndarray) -> np.ndarray:
    """Remap FreeSurfer aseg labels to our 21-region scheme."""
    out = np.zeros_like(seg, dtype=np.int64)
    for orig_label, new_label in ASEG_REMAP.items():
        out[seg == orig_label] = new_label
    return out


def process_subject(subject_dir: Path, target_shape: Tuple[int, ...],
                    label: int) -> Optional[dict]:
    """Process one FreeSurfer subject → dict with image, seg, label."""
    mri_dir = subject_dir / "mri"
    if not mri_dir.exists():
        return None

    brain_path = None
    for fname in ["brain.mgz", "brainmask.mgz", "T1.mgz", "norm.mgz"]:
        p = mri_dir / fname
        if p.exists():
            brain_path = p
            break
    if brain_path is None:
        return None

    aseg_path = mri_dir / "aseg.mgz"
    if not aseg_path.exists():
        aseg_path = None

    try:
        img = nib.load(str(brain_path)).get_fdata().astype(np.float32)
        img = normalize_image(img)
        img = resize(img, target_shape, order=1).astype(np.float32)
    except Exception as e:
        print(f"    Error loading brain: {e}")
        return None

    if aseg_path is not None:
        try:
            seg_raw = nib.load(str(aseg_path)).get_fdata().astype(np.int64)
            seg = remap_aseg(seg_raw)
            seg = resize(seg.astype(np.float32), target_shape, order=0).astype(np.int64)
        except Exception as e:
            print(f"    Error loading aseg: {e}")
            seg = np.zeros(target_shape, dtype=np.int64)
    else:
        seg = np.zeros(target_shape, dtype=np.int64)

    return {"image": img, "seg": seg, "label": label}


def extract_and_process(tar_path: Path, extract_dir: Path, labels: Dict[str, int],
                        output_cache: Path, target_shape: Tuple[int, ...],
                        overwrite: bool) -> dict:
    """Extract a tar.gz and process all subjects inside."""
    stats = {"processed": 0, "skipped_no_label": 0, "skipped_exists": 0, "errors": 0}

    # Check if subjects already extracted
    existing_subjects = list(extract_dir.rglob("OAS1_*_MR*"))
    existing_with_mri = [d for d in existing_subjects if d.is_dir() and (d / "mri").is_dir()]

    if existing_with_mri:
        print(f"\n{tar_path.name}: found {len(existing_with_mri)} already-extracted subjects, skipping extraction")
    else:
        print(f"\nExtracting {tar_path.name} ...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_dir)

    # Find all OAS1_* subject directories recursively
    subject_dirs = []
    for d in sorted(extract_dir.rglob("OAS1_*_MR*")):
        if d.is_dir() and (d / "mri").is_dir():
            subject_dirs.append(d)

    print(f"  Found {len(subject_dirs)} subject directories with mri/")

    for subj_dir in subject_dirs:
        subj_name = subj_dir.name
        base_match = re.match(r"(OAS1_\d+)_MR\d+", subj_name)
        base_id = base_match.group(1) if base_match else subj_name

        label = labels.get(subj_name) or labels.get(base_id)
        if label is None:
            stats["skipped_no_label"] += 1
            continue

        out_path = output_cache / f"OASIS_{subj_name}.npz"
        if out_path.exists() and not overwrite:
            stats["skipped_exists"] += 1
            continue

        result = process_subject(subj_dir, target_shape, label)
        if result is None:
            stats["errors"] += 1
            continue

        np.savez_compressed(
            out_path,
            image=result["image"].astype(np.float16),
            seg=result["seg"].astype(np.int8),
            label=np.array(result["label"], dtype=np.int64),
        )
        stats["processed"] += 1
        if stats["processed"] % 20 == 0:
            print(f"  Processed: {stats['processed']}", flush=True)

    return stats


def main():
    args = parse_args()
    args.output_cache.mkdir(parents=True, exist_ok=True)

    extract_dir = args.extract_dir or Path("/tmp/oasis_extract")
    extract_dir.mkdir(parents=True, exist_ok=True)

    print("Loading OASIS metadata ...")
    labels = load_metadata(args.metadata)
    print(f"  Subjects with CDR labels: {len(labels)}")
    label_counts = {}
    for lbl in labels.values():
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    label_names = {0: "CN", 1: "MCI", 2: "AD"}
    for k, v in sorted(label_counts.items()):
        print(f"    {label_names.get(k, k)}: {v}")

    tar_files = sorted(args.tar_dir.glob("oasis_cs_freesurfer_disc*.tar.gz"))
    print(f"\nFound {len(tar_files)} disc archives:")
    for t in tar_files:
        print(f"  {t.name} ({t.stat().st_size / 1e9:.1f} GB)")

    total_stats = {"processed": 0, "skipped_no_label": 0, "skipped_exists": 0, "errors": 0}

    for tar_path in tar_files:
        stats = extract_and_process(
            tar_path, extract_dir, labels,
            args.output_cache, tuple(args.target_shape),
            args.overwrite,
        )
        for k in total_stats:
            total_stats[k] += stats[k]
        print(f"  Disc done: {stats}")

    summary = {
        "n_labels": len(labels),
        "tar_files": [str(t) for t in tar_files],
        "output_cache": str(args.output_cache),
        **total_stats,
    }
    summary_path = args.output_cache / "oasis_preprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print("OASIS Preprocessing Summary")
    print(f"{'='*60}")
    print(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
