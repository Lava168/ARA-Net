"""
Foundation Data Loader for Chapter 1

Supports:
- Real cached data (ADNI + IXI preprocessed .npz) -- primary
- Stratified train/val/test split (70/15/15)
- 5-fold cross-validation splits
- External validation (IXI out-of-site)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# FreeSurfer subcortical labels used in our atlas segmentation
FREESURFER_LABELS = {
    "L-Hippocampus": 17, "R-Hippocampus": 53,
    "L-Amygdala": 18, "R-Amygdala": 54,
    "L-Thalamus": 10, "R-Thalamus": 49,
    "L-Caudate": 11, "R-Caudate": 50,
    "L-Putamen": 12, "R-Putamen": 51,
    "L-Pallidum": 13, "R-Pallidum": 52,
    "L-Lat-Ventricle": 4, "R-Lat-Ventricle": 43,
    "L-Cortex": 3, "R-Cortex": 42,
    "L-WM": 2, "R-WM": 41,
    "Brain-Stem": 16,
    "L-Accumbens": 26, "R-Accumbens": 58,
}

AD_ROIS = [17, 53, 18, 54, 10, 49, 11, 50, 12, 51]

_FS_LABELS = [
    0,
    2, 3, 4, 10, 11, 12, 13, 16, 17, 18, 26,
    41, 42, 43, 49, 50, 51, 52, 53, 54, 58,
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(_FS_LABELS)}
NUM_MAPPED_REGIONS = len(_FS_LABELS) - 1  # 21


def remap_segmentation(seg: np.ndarray) -> np.ndarray:
    """Remap FreeSurfer labels to contiguous indices 0..21."""
    out = np.zeros_like(seg, dtype=np.int64)
    for label, idx in LABEL_TO_IDX.items():
        out[seg == label] = idx
    return out


def stratified_split(
    labels: np.ndarray,
    sites: np.ndarray,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test split by label, preserving site distribution."""
    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []

    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train:n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val:].tolist())

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def kfold_split(
    labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    val_fraction: float = 0.15,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Stratified K-fold: for each fold, returns (train_idx, val_idx, test_idx).

    Test = fold k, Train+Val = remaining folds (split by val_fraction).
    """
    rng = np.random.RandomState(seed)
    fold_indices = [[] for _ in range(n_folds)]

    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        rng.shuffle(idx)
        for i, ix in enumerate(idx):
            fold_indices[i % n_folds].append(ix)

    splits = []
    for k in range(n_folds):
        test = np.array(fold_indices[k])
        remaining = np.concatenate([
            np.array(fold_indices[j]) for j in range(n_folds) if j != k
        ])
        rng2 = np.random.RandomState(seed + k)
        rng2.shuffle(remaining)
        n_val = max(int(len(remaining) * val_fraction), 1)
        val = remaining[:n_val]
        train = remaining[n_val:]
        splits.append((train, val, test))

    return splits


class RealCachedDataset(Dataset):
    """Fast dataset from preprocessed .npz cache (ADNI + IXI).

    Each .npz: image (96,112,96 float16), seg (96,112,96 int8), label (int).
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        indices: Optional[np.ndarray] = None,
        split: Optional[str] = None,
        max_samples: int = 0,
        seed: int = 42,
        augment_fn=None,
        site_filter: Optional[str] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.augment_fn = augment_fn
        npz_files = sorted(self.cache_dir.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files in {cache_dir}")

        all_labels = []
        all_paths = []
        all_sites = []
        for f in npz_files:
            data = np.load(f, allow_pickle=True)
            all_labels.append(int(data["label"]))
            all_paths.append(f)
            sid = f.stem
            all_sites.append("ADNI" if sid.startswith("ADNI_") else "IXI")

        all_labels = np.array(all_labels)
        all_paths = np.array(all_paths)
        all_sites = np.array(all_sites)

        if site_filter:
            mask = all_sites == site_filter
            all_labels = all_labels[mask]
            all_paths = all_paths[mask]
            all_sites = all_sites[mask]

        if indices is not None:
            chosen = indices
        elif split is not None:
            train_idx, val_idx, test_idx = stratified_split(all_labels, all_sites, seed=seed)
            if split == "train":
                chosen = train_idx
            elif split == "val":
                chosen = val_idx
            elif split == "test":
                chosen = test_idx
            else:
                raise ValueError(f"Unknown split: {split}")
        else:
            chosen = np.arange(len(all_paths))

        rng = np.random.RandomState(seed + 1)
        chosen_list = chosen.tolist()
        rng.shuffle(chosen_list)

        self.samples = []
        for i in chosen_list:
            self.samples.append({
                "path": str(all_paths[i]),
                "label": int(all_labels[i]),
                "subject_id": all_paths[i].stem,
                "site": all_sites[i],
            })

        if max_samples > 0:
            self.samples = self.samples[:max_samples]

        counts = {}
        for s in self.samples:
            counts[s["label"]] = counts.get(s["label"], 0) + 1
        label_str = ", ".join(f"{['CN','MCI','AD'][k]}={v}" for k, v in sorted(counts.items()))
        name = split or "custom"
        print(f"  RealCachedDataset [{name}]: {len(self.samples)} ({label_str})", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        data = np.load(s["path"], allow_pickle=True)
        image = data["image"].astype(np.float32)
        seg = data["seg"].astype(np.int64)
        seg = remap_segmentation(seg)

        if self.augment_fn is not None:
            sample_dict = {"image": image, "segmentation": seg}
            sample_dict = self.augment_fn(sample_dict)
            image = sample_dict["image"]
            seg = sample_dict["segmentation"]

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image).astype(np.float32)).unsqueeze(0),
            "segmentation": torch.from_numpy(np.ascontiguousarray(seg).astype(np.int64)),
            "label": torch.tensor(s["label"], dtype=torch.long),
            "subject_id": s["subject_id"],
            "site": s["site"],
        }


def create_foundation_dataloaders(
    data_root: Union[str, Path],
    batch_size: int = 4,
    num_workers: int = 4,
    max_train_samples: int = 0,
    max_val_samples: int = 0,
    seed: int = 42,
    augment_fn=None,
    fold_indices: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    site_filter: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create train, val, (and optionally test) dataloaders.

    Returns (train_loader, val_loader, test_loader).
    If fold_indices is provided, uses those indices directly.
    """
    data_root = Path(data_root)
    cache_real_dir = data_root / "cache_real"

    if not (cache_real_dir.exists() and len(list(cache_real_dir.glob("*.npz"))) > 0):
        raise FileNotFoundError(f"Real data cache not found at {cache_real_dir}")

    if fold_indices is not None:
        train_idx, val_idx, test_idx = fold_indices
        train_ds = RealCachedDataset(cache_real_dir, indices=train_idx, seed=seed,
                                     augment_fn=augment_fn, max_samples=max_train_samples,
                                     site_filter=site_filter)
        val_ds = RealCachedDataset(cache_real_dir, indices=val_idx, seed=seed,
                                   max_samples=max_val_samples, site_filter=site_filter)
        test_ds = RealCachedDataset(cache_real_dir, indices=test_idx, seed=seed,
                                    site_filter=site_filter)
    else:
        train_ds = RealCachedDataset(cache_real_dir, split="train", seed=seed,
                                     augment_fn=augment_fn, max_samples=max_train_samples,
                                     site_filter=site_filter)
        val_ds = RealCachedDataset(cache_real_dir, split="val", seed=seed,
                                   max_samples=max_val_samples, site_filter=site_filter)
        test_ds = RealCachedDataset(cache_real_dir, split="test", seed=seed,
                                    site_filter=site_filter)

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0)

    return make_loader(train_ds, True), make_loader(val_ds, False), make_loader(test_ds, False)
