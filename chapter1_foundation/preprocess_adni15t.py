#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

LABEL_MAP = {"CN": 0, "MCI": 1, "AD": 2}
ID_RE = re.compile(r"(I\d+)")


def parse_args():
    p = argparse.ArgumentParser(description="ADNI 1.5T (NIfTI+CSV) -> cache_real npz")
    p.add_argument("--adni_root", type=Path, default=Path("/home/lry/adni_extracted/ADNI"))
    p.add_argument("--csv_path", type=Path, default=Path("/home/lry/ADNI1_Complete_1Yr_1.5T_2_27_2026.csv"))
    p.add_argument("--output_cache", type=Path, default=Path("sample_data/cache_real"))
    p.add_argument("--seg_dir", type=Path, default=Path("sample_data/segmentations"))
    p.add_argument("--target_shape", type=int, nargs=3, default=(96, 112, 96))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


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


def build_nifti_index(adni_root: Path):
    idx = {}
    for p in adni_root.glob("**/*.nii"):
        m = ID_RE.search(p.name)
        if m:
            idx.setdefault(m.group(1), p)
    return idx


def find_seg(seg_dir: Path, subject: str, image_id: str):
    if not seg_dir.exists():
        return None
    cands = list(seg_dir.glob(f"ADNI_{subject}_*_{image_id}_seg.npy"))
    if cands:
        return cands[0]
    cands = list(seg_dir.glob(f"*{subject}*{image_id}*_seg.npy"))
    return cands[0] if cands else None


def main():
    args = parse_args()
    args.output_cache.mkdir(parents=True, exist_ok=True)

    with args.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit > 0:
        rows = rows[:args.limit]

    nifti_idx = build_nifti_index(args.adni_root)

    processed = 0
    skipped_no_nifti = 0
    skipped_bad_group = 0
    missing_seg = 0

    print(f"CSV rows: {len(rows)}")
    print(f"Indexed NIfTI IDs: {len(nifti_idx)}")

    for row in rows:
        image_id = (row.get("Image Data ID") or "").strip().strip('"')
        subject = (row.get("Subject") or "").strip().strip('"')
        group = (row.get("Group") or "").strip().strip('"').upper()
        visit = (row.get("Visit") or "na").strip().strip('"').replace(" ", "_")
        downloaded = (row.get("Downloaded") or "").strip().strip('"').lower()

        if downloaded and downloaded not in {"yes", "y", "true", "1"}:
            continue
        if group not in LABEL_MAP:
            skipped_bad_group += 1
            continue
        nii_path = nifti_idx.get(image_id)
        if nii_path is None:
            skipped_no_nifti += 1
            continue

        out = args.output_cache / f"ADNI_{subject}_{visit}_{image_id}.npz"
        if out.exists() and not args.overwrite:
            continue

        img = nib.load(str(nii_path)).get_fdata().astype(np.float32)
        img = normalize_image(img)
        img = resize(img, tuple(args.target_shape), order=1).astype(np.float32)

        seg_path = find_seg(args.seg_dir, subject, image_id)
        if seg_path is not None and seg_path.exists():
            seg = np.load(seg_path).astype(np.int64)
            seg = resize(seg.astype(np.float32), tuple(args.target_shape), order=0).astype(np.int64)
        else:
            seg = np.zeros(tuple(args.target_shape), dtype=np.int64)
            missing_seg += 1

        np.savez_compressed(
            out,
            image=img.astype(np.float16),
            seg=seg.astype(np.int8),
            label=np.array(LABEL_MAP[group], dtype=np.int64),
        )
        processed += 1
        if processed % 50 == 0:
            print(f"Processed: {processed}", flush=True)

    summary = {
        "csv_rows": len(rows),
        "indexed_nifti_ids": len(nifti_idx),
        "processed": processed,
        "skipped_no_nifti": skipped_no_nifti,
        "skipped_bad_group": skipped_bad_group,
        "missing_seg_filled_zero": missing_seg,
        "output_cache": str(args.output_cache),
    }
    sm = args.output_cache / "adni15t_preprocess_summary.json"
    sm.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Summary saved to {sm}")


if __name__ == "__main__":
    main()
