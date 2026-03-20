#!/usr/bin/env python3
"""
Synthesize additional training samples from existing MRI scans.

Strategies:
1. Elastic deformation + intensity mixing of same-class samples
2. Atlas-guided regional atrophy simulation (for AD/MCI synthesis)
3. Integration of OASIS FreeSurfer data with clinical labels

This does NOT create random noise — it produces anatomically plausible
variations grounded in known AD pathology (hippocampal/cortical atrophy).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, zoom

AD_ATROPHY_REGIONS_REMAPPED = {
    9: 0.85, 19: 0.85,   # L/R Hippocampus — strongest atrophy
    10: 0.88, 20: 0.88,  # L/R Amygdala
    2: 0.92, 13: 0.92,   # L/R Cortex — mild cortical thinning
    3: 1.15, 14: 1.15,   # L/R Ventricle — enlargement
    4: 0.94, 15: 0.94,   # L/R Thalamus
}

MCI_ATROPHY_REGIONS_REMAPPED = {
    9: 0.92, 19: 0.92,
    10: 0.94, 20: 0.94,
    3: 1.08, 14: 1.08,
}


def elastic_deform_3d(image: np.ndarray, seg: np.ndarray,
                      alpha: float = 8.0, sigma: float = 4.0,
                      seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Apply smooth elastic deformation to both image and segmentation."""
    rng = np.random.RandomState(seed)
    shape = image.shape
    dx = gaussian_filter(rng.randn(*shape) * alpha, sigma, mode='constant')
    dy = gaussian_filter(rng.randn(*shape) * alpha, sigma, mode='constant')
    dz = gaussian_filter(rng.randn(*shape) * alpha, sigma, mode='constant')

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                           np.arange(shape[2]), indexing='ij')
    coords = [np.clip(x + dx, 0, shape[0]-1),
              np.clip(y + dy, 0, shape[1]-1),
              np.clip(z + dz, 0, shape[2]-1)]

    img_def = map_coordinates(image, coords, order=1, mode='constant', cval=0)
    seg_def = map_coordinates(seg.astype(float), coords, order=0, mode='constant', cval=0)
    return img_def.astype(np.float32), seg_def.astype(np.int64)


def simulate_atrophy(image: np.ndarray, seg: np.ndarray,
                     atrophy_map: Dict[int, float],
                     noise_std: float = 0.02,
                     seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate regional brain atrophy by scaling voxel intensities.

    For regions with scale < 1.0 (atrophy): reduce intensity and slightly
    smooth to simulate tissue loss. For scale > 1.0 (ventricle enlargement):
    increase intensity in ventricle boundary voxels.
    """
    rng = np.random.RandomState(seed)
    result = image.copy()

    for region_id, scale in atrophy_map.items():
        mask = seg == region_id
        if not mask.any():
            continue

        variation = rng.uniform(0.95, 1.05)
        effective_scale = 1.0 + (scale - 1.0) * variation

        if effective_scale < 1.0:
            region_vals = result[mask]
            result[mask] = region_vals * effective_scale
            kernel = gaussian_filter(mask.astype(float), sigma=1.0)
            boundary = (kernel > 0.1) & (kernel < 0.9) & (~mask)
            result[boundary] *= (1.0 + (effective_scale - 1.0) * 0.3)
        else:
            result[mask] *= effective_scale

    if noise_std > 0:
        result += rng.randn(*result.shape).astype(np.float32) * noise_std

    result = np.clip(result, 0, 1)
    return result.astype(np.float32), seg


def mixup_same_class(img1: np.ndarray, img2: np.ndarray,
                     seg1: np.ndarray, seg2: np.ndarray,
                     alpha: float = 0.3, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Intensity mixup between two same-class samples."""
    rng = np.random.RandomState(seed)
    lam = rng.beta(alpha, alpha) if alpha > 0 else 0.5
    lam = max(0.6, min(0.9, lam))
    mixed = (lam * img1 + (1 - lam) * img2).astype(np.float32)
    seg_out = seg1 if lam >= 0.5 else seg2
    return mixed, seg_out


def synthesize_from_cache(cache_dir: str, output_dir: str,
                          target_per_class: int = 150,
                          seed: int = 42) -> Dict[str, int]:
    """Generate synthetic samples to balance classes.

    Returns dict of class -> number of synthetic samples created.
    """
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(cache_dir.glob("ADNI_*.npz"))
    by_class: Dict[int, List[Path]] = {0: [], 1: [], 2: []}
    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        lbl = int(d['label'])
        if lbl in by_class:
            by_class[lbl].append(f)

    class_names = {0: 'CN', 1: 'MCI', 2: 'AD'}
    rng = np.random.RandomState(seed)
    counts = {}

    for cls_id, files in by_class.items():
        n_real = len(files)
        n_needed = max(0, target_per_class - n_real)
        if n_needed == 0:
            counts[class_names[cls_id]] = 0
            continue

        print(f"  Synthesizing {n_needed} samples for {class_names[cls_id]} "
              f"(have {n_real}, target {target_per_class})")

        created = 0
        for i in range(n_needed):
            src_idx = rng.randint(0, n_real)
            src_data = np.load(files[src_idx], allow_pickle=True)
            img = src_data['image'].astype(np.float32)
            seg = src_data['seg'].astype(np.int64)

            strategy = rng.choice(['elastic', 'atrophy', 'mixup', 'combined'],
                                  p=[0.3, 0.3, 0.15, 0.25])

            s = seed + i * 7 + cls_id * 1000

            if strategy == 'elastic':
                alpha = rng.uniform(5.0, 12.0)
                sigma = rng.uniform(3.0, 6.0)
                img, seg = elastic_deform_3d(img, seg, alpha=alpha, sigma=sigma, seed=s)

            elif strategy == 'atrophy':
                if cls_id == 2:
                    img, seg = simulate_atrophy(img, seg, AD_ATROPHY_REGIONS_REMAPPED,
                                                noise_std=rng.uniform(0.01, 0.03), seed=s)
                elif cls_id == 1:
                    img, seg = simulate_atrophy(img, seg, MCI_ATROPHY_REGIONS_REMAPPED,
                                                noise_std=rng.uniform(0.01, 0.02), seed=s)
                else:
                    img, seg = elastic_deform_3d(img, seg, alpha=6.0, sigma=4.0, seed=s)

            elif strategy == 'mixup':
                other_idx = rng.randint(0, n_real)
                other_data = np.load(files[other_idx], allow_pickle=True)
                img2 = other_data['image'].astype(np.float32)
                seg2 = other_data['seg'].astype(np.int64)
                img, seg = mixup_same_class(img, img2, seg, seg2, seed=s)

            elif strategy == 'combined':
                img, seg = elastic_deform_3d(img, seg, alpha=6.0, sigma=4.0, seed=s)
                if cls_id == 2:
                    img, seg = simulate_atrophy(img, seg, AD_ATROPHY_REGIONS_REMAPPED,
                                                noise_std=0.015, seed=s+1)
                elif cls_id == 1:
                    img, seg = simulate_atrophy(img, seg, MCI_ATROPHY_REGIONS_REMAPPED,
                                                noise_std=0.01, seed=s+1)

            img = np.clip(img, 0, 1)

            out_name = f"SYNTH_{class_names[cls_id]}_{i:04d}.npz"
            np.savez_compressed(
                output_dir / out_name,
                image=img.astype(np.float16),
                seg=seg.astype(np.int8),
                label=np.array(cls_id, dtype=np.int64),
            )
            created += 1

        counts[class_names[cls_id]] = created

    return counts


def integrate_oasis_disc1(tar_path: str, clinical_xlsx: str,
                          cache_dir: str, seed: int = 42) -> int:
    """Extract and integrate OASIS disc1 FreeSurfer data.

    Extracts brain.mgz and aseg.mgz, converts to numpy, resamples to 96x112x96,
    and saves as .npz with CDR-derived labels.
    """
    import pandas as pd
    import subprocess
    import tempfile

    clinical = pd.read_excel(clinical_xlsx)
    clinical = clinical.dropna(subset=['CDR'])

    label_map = {}
    for _, row in clinical.iterrows():
        sid = row['ID']
        cdr = row['CDR']
        if cdr == 0:
            label_map[sid] = 0
        elif cdr == 0.5:
            label_map[sid] = 1
        else:
            label_map[sid] = 2

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"  Extracting OASIS FreeSurfer data to {tmpdir} (single-pass)...")
        # Fast path: extract once, then iterate files locally.
        try:
            subprocess.run(
                ['tar', 'xzf', tar_path, '-C', tmpdir, 'disc1'],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"  Failed to extract OASIS tar: {e}")
            return 0

        target_shape = (96, 112, 96)
        for sid, label in label_map.items():
            brain_file = Path(tmpdir) / f"disc1/{sid}/mri/brain.mgz"
            aseg_file = Path(tmpdir) / f"disc1/{sid}/mri/aseg.mgz"
            if not brain_file.exists() or not aseg_file.exists():
                continue

            out_name = f"OASIS_{sid}.npz"
            out_path = cache_dir / out_name
            if out_path.exists():
                continue

            try:
                import nibabel as nib
                brain_nii = nib.load(str(brain_file))
                aseg_nii = nib.load(str(aseg_file))
                brain_data = brain_nii.get_fdata().astype(np.float32)
                aseg_data = aseg_nii.get_fdata().astype(np.int64)

                factors = [t / s for t, s in zip(target_shape, brain_data.shape)]
                brain_resized = zoom(brain_data, factors, order=1)
                aseg_resized = zoom(aseg_data.astype(float), factors, order=0).astype(np.int64)

                brain_resized = brain_resized / max(brain_resized.max(), 1e-8)
                brain_resized = np.clip(brain_resized, 0, 1)

                np.savez_compressed(
                    out_path,
                    image=brain_resized.astype(np.float16),
                    seg=aseg_resized.astype(np.int8),
                    label=np.array(label, dtype=np.int64),
                )
                count += 1
                if count % 20 == 0:
                    print(f"    Integrated {count} OASIS subjects...", flush=True)
            except Exception as e:
                print(f"    Failed {sid}: {e}")
                continue

    return count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="sample_data/cache_real")
    parser.add_argument("--synth_output", default="sample_data/cache_real")
    parser.add_argument("--target_per_class", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oasis_tar", default="")
    parser.add_argument("--oasis_xlsx", default="")
    args = parser.parse_args()

    print("=" * 60)
    print("Data Synthesis & Integration Pipeline")
    print("=" * 60)

    if args.oasis_tar and args.oasis_xlsx:
        print("\n[Step 1] Integrating OASIS disc1...")
        n = integrate_oasis_disc1(args.oasis_tar, args.oasis_xlsx,
                                  args.cache_dir, args.seed)
        print(f"  Integrated {n} OASIS subjects\n")

    print("[Step 2] Synthesizing balanced training data...")
    counts = synthesize_from_cache(args.cache_dir, args.synth_output,
                                   args.target_per_class, args.seed)
    print(f"\nSynthesis complete: {counts}")
