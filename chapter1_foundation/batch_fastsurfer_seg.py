#!/usr/bin/env python3
"""
Batch FastSurfer segmentation for ADNI volumes missing valid segmentations.

Identifies all .npz files in cache_real with zero-sum seg arrays,
finds their corresponding NIfTI source files, runs FastSurfer segmentation
on GPU, and updates the .npz files with the new segmentation.

Uses multiple GPUs in parallel for ~3.5 hour total runtime on 6x RTX 2080 SUPER.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

FASTSURFER_HOME = Path("/home/lry/FastSurfer")
FASTSURFER_SCRIPT = FASTSURFER_HOME / "FastSurferCNN" / "run_prediction.py"

ARA_LABELS = {0, 2, 3, 4, 10, 11, 12, 13, 16, 17, 18, 26,
              41, 42, 43, 49, 50, 51, 52, 53, 54, 58}

ID_RE = re.compile(r"(I\d+)")


def find_zero_seg_files(cache_dir: Path) -> list[Path]:
    """Return .npz files whose seg array is all zeros."""
    zero_files = []
    for f in sorted(cache_dir.glob("ADNI_*.npz")):
        try:
            d = np.load(f, allow_pickle=True)
            seg = d.get("seg", None)
            if seg is None or seg.sum() == 0:
                zero_files.append(f)
        except Exception:
            zero_files.append(f)
    return zero_files


def build_nifti_index(adni_root: Path) -> dict[str, Path]:
    """Map Image ID (e.g. I118692) to NIfTI file path."""
    idx = {}
    for p in adni_root.glob("**/*.nii"):
        m = ID_RE.search(p.name)
        if m:
            idx.setdefault(m.group(1), p)
    return idx


def extract_image_id(npz_stem: str) -> str | None:
    m = ID_RE.search(npz_stem)
    return m.group(1) if m else None


def run_fastsurfer(nii_path: Path, output_dir: Path, sid: str,
                   gpu_id: int) -> Path | None:
    """Run FastSurfer seg-only on a single NIfTI, return aseg.mgz path."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(FASTSURFER_HOME) + ":" + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, str(FASTSURFER_SCRIPT),
        "--t1", str(nii_path),
        "--sd", str(output_dir),
        "--sid", sid,
        "--device", "cuda:0",
        "--batch_size", "8",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, env=env
        )
        aseg_path = output_dir / sid / "mri" / "aseg.auto_noCCseg.mgz"
        if aseg_path.exists():
            return aseg_path
        aparc_path = output_dir / sid / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
        if aparc_path.exists():
            return aparc_path
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass
    return None


def aseg_to_seg_array(aseg_path: Path, target_shape: tuple = (96, 112, 96)) -> np.ndarray:
    """Load aseg.mgz, keep only ARA-Net labels, resize to target shape."""
    seg_full = nib.load(str(aseg_path)).get_fdata().astype(np.int64)
    mask = np.zeros_like(seg_full, dtype=bool)
    for lbl in ARA_LABELS:
        mask |= (seg_full == lbl)
    seg_clean = np.where(mask, seg_full, 0)

    factors = [t / s for t, s in zip(target_shape, seg_clean.shape)]
    seg_resized = zoom(seg_clean.astype(np.float32), factors, order=0).astype(np.int8)
    return seg_resized


def update_npz(npz_path: Path, new_seg: np.ndarray):
    """Replace seg array in existing .npz file."""
    d = dict(np.load(npz_path, allow_pickle=True))
    d["seg"] = new_seg
    np.savez_compressed(npz_path, **d)


def save_seg_npy(seg_dir: Path, npz_stem: str, seg: np.ndarray):
    """Also save to segmentations/ dir for consistency."""
    out = seg_dir / f"{npz_stem}_seg.npy"
    np.save(out, seg)


def worker(args_tuple):
    """Process a single volume: FastSurfer -> extract seg -> update .npz."""
    npz_path, nii_path, gpu_id, seg_dir, target_shape = args_tuple
    stem = npz_path.stem
    sid = f"fs_{stem}"

    with tempfile.TemporaryDirectory(prefix="fastsurfer_") as tmpdir:
        tmpdir = Path(tmpdir)
        aseg_path = run_fastsurfer(nii_path, tmpdir, sid, gpu_id)

        if aseg_path is None:
            return stem, False, "FastSurfer failed"

        try:
            seg = aseg_to_seg_array(aseg_path, target_shape)
            n_labels = len(np.unique(seg))
            if n_labels < 3:
                return stem, False, f"Only {n_labels} unique labels"

            update_npz(npz_path, seg)
            save_seg_npy(seg_dir, stem, seg)
            return stem, True, f"{n_labels} labels"
        except Exception as e:
            return stem, False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=Path,
                        default=Path("sample_data/cache_real"))
    parser.add_argument("--adni_root", type=Path,
                        default=Path("/home/lry/adni15t_extracted/ADNI"))
    parser.add_argument("--seg_dir", type=Path,
                        default=Path("sample_data/segmentations"))
    parser.add_argument("--target_shape", type=int, nargs=3,
                        default=[96, 112, 96])
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    target_shape = tuple(args.target_shape)

    print("=" * 60)
    print("Batch FastSurfer Segmentation")
    print("=" * 60)

    print(f"\n[1/3] Finding zero-seg .npz files in {args.cache_dir} ...")
    zero_files = find_zero_seg_files(args.cache_dir)
    print(f"  Found {len(zero_files)} files with missing segmentation")

    print(f"\n[2/3] Building NIfTI index from {args.adni_root} ...")
    nifti_idx = build_nifti_index(args.adni_root)
    print(f"  Indexed {len(nifti_idx)} NIfTI files")

    work_items = []
    no_nifti = []
    for npz_path in zero_files:
        image_id = extract_image_id(npz_path.stem)
        if image_id and image_id in nifti_idx:
            gpu_id = gpus[len(work_items) % len(gpus)]
            work_items.append((
                npz_path, nifti_idx[image_id], gpu_id,
                args.seg_dir, target_shape
            ))
        else:
            no_nifti.append(npz_path.stem)

    if args.limit > 0:
        work_items = work_items[:args.limit]

    print(f"\n  Matched: {len(work_items)} volumes to process")
    print(f"  No NIfTI found: {len(no_nifti)}")
    if no_nifti[:5]:
        for nm in no_nifti[:5]:
            print(f"    - {nm}")

    if args.dry_run:
        print("\n[DRY RUN] Would process these volumes:")
        for item in work_items[:10]:
            print(f"  {item[0].stem} -> GPU {item[2]}")
        print(f"  ... and {len(work_items) - 10} more" if len(work_items) > 10 else "")
        return

    est_time = len(work_items) * 40 / len(gpus)
    print(f"\n[3/3] Processing {len(work_items)} volumes on {len(gpus)} GPUs ...")
    print(f"  Estimated time: {est_time/3600:.1f} hours ({est_time/60:.0f} min)")
    print(f"  ~{40}s per volume, {len(gpus)} GPUs in parallel\n")

    args.seg_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0
    t0 = time.time()

    with mp.Pool(processes=len(gpus)) as pool:
        for i, (stem, ok, msg) in enumerate(pool.imap_unordered(worker, work_items)):
            if ok:
                success += 1
            else:
                failed += 1

            elapsed = time.time() - t0
            total = success + failed
            rate = elapsed / max(total, 1)
            remaining = rate * (len(work_items) - total)

            if total % 10 == 0 or not ok:
                status = "OK" if ok else "FAIL"
                print(f"  [{total:4d}/{len(work_items)}] {status}: {stem} ({msg}) "
                      f"| {elapsed/60:.0f}m elapsed, ~{remaining/60:.0f}m left",
                      flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/3600:.1f} hours")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Total:   {success + failed}")

    summary = {
        "total_processed": success + failed,
        "success": success,
        "failed": failed,
        "elapsed_seconds": elapsed,
        "gpus_used": len(gpus),
    }
    summary_path = args.cache_dir / "fastsurfer_batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
