#!/usr/bin/env python3
"""Train ARA-Net v4 with external-generalization-first objectives.

The model is deliberately self-contained so historical code revisions on the
server do not change the experiment. It trains on the v4 manifest produced by
``build_v4_manifest.py`` and evaluates every requested split after each epoch.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None


CLASS_NAMES = ["CN", "MCI", "AD"]
DATASET_TO_DOMAIN = {"ADNI": 0, "AIBL": 1, "OASIS": 2, "IXI": 3}
FS_LABELS = [
    0,
    2, 3, 4, 10, 11, 12, 13, 16, 17, 18, 26,
    41, 42, 43, 49, 50, 51, 52, 53, 54, 58,
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(FS_LABELS)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def remap_segmentation(seg: np.ndarray) -> np.ndarray:
    out = np.zeros_like(seg, dtype=np.int64)
    for label, idx in LABEL_TO_IDX.items():
        out[seg == label] = idx
    return out


def parse_split_list(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def read_manifest(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = int(row["label"])
        row["domain"] = DATASET_TO_DOMAIN.get(row["dataset"], 0)
    return rows


class ManifestDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict],
        augment: bool = False,
        cache_images: bool = False,
    ):
        self.rows = list(rows)
        self.augment = bool(augment)
        self.cache_images = bool(cache_images)
        self._cache: Dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, index: int) -> dict:
        row = self.rows[index]
        with np.load(row["path"], allow_pickle=True) as data:
            image = data["image"].astype(np.float32)
            seg = data["seg"].astype(np.int64)
        seg = remap_segmentation(seg)
        return {"image": image, "seg": seg}

    def _maybe_load(self, index: int) -> dict:
        if not self.cache_images:
            return self._load(index)
        if index not in self._cache:
            self._cache[index] = self._load(index)
        return self._cache[index]

    @staticmethod
    def _augment(image: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < 0.5:
            axis = np.random.choice([0, 1, 2])
            image = np.flip(image, axis=axis).copy()
            seg = np.flip(seg, axis=axis).copy()
        if np.random.rand() < 0.65:
            mask = image != 0
            scale = 1.0 + np.random.uniform(-0.12, 0.12)
            shift = np.random.uniform(-0.08, 0.08)
            image = (image * scale + shift).astype(np.float32)
            image[~mask] = 0.0
        if np.random.rand() < 0.35:
            mask = image != 0
            image = image + np.random.normal(0.0, 0.035, size=image.shape).astype(np.float32) * mask
        return image, seg

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        loaded = self._maybe_load(index)
        image = loaded["image"].copy()
        seg = loaded["seg"].copy()
        if self.augment:
            image, seg = self._augment(image, seg)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).float(),
            "segmentation": torch.from_numpy(np.ascontiguousarray(seg)).long(),
            "label": torch.tensor(row["label"], dtype=torch.long),
            "domain": torch.tensor(row["domain"], dtype=torch.long),
            "dataset": row["dataset"],
            "split": row["split"],
            "subject_id": row["subject_id"],
            "scan_id": row["scan_id"],
        }


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResBlock3D(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.conv1 = ConvBlock3D(channels, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.drop(self.bn2(self.conv2(x)))
        return F.silu(x + residual)


class FeatureEncoder3D(nn.Module):
    def __init__(self, base_channels: int = 32, max_channels: int = 192, dropout: float = 0.08):
        super().__init__()
        self.stem = ConvBlock3D(1, base_channels)
        self.stages = nn.ModuleList()
        in_ch = base_channels
        for stage in range(4):
            out_ch = min(base_channels * (2 ** stage), max_channels)
            self.stages.append(
                nn.Sequential(
                    ConvBlock3D(in_ch, out_ch, stride=2),
                    ResBlock3D(out_ch, dropout),
                )
            )
            in_ch = out_ch
        self.out_channels = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x


class RegionPooling(nn.Module):
    def __init__(self, num_regions: int = 21):
        super().__init__()
        self.num_regions = num_regions

    def forward(self, features: torch.Tensor, segmentation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seg = F.interpolate(
            segmentation.unsqueeze(1).float(),
            size=features.shape[-3:],
            mode="nearest",
        ).squeeze(1).long()
        batch, channels, _, _, _ = features.shape
        feat_flat = features.view(batch, channels, -1)
        seg_flat = seg.view(batch, -1)
        region_feats = torch.zeros(
            batch, self.num_regions + 1, channels,
            device=features.device, dtype=features.dtype,
        )
        counts = torch.zeros(
            batch, self.num_regions + 1, 1,
            device=features.device, dtype=features.dtype,
        )
        for region in range(self.num_regions + 1):
            mask = (seg_flat == region).unsqueeze(1).to(features.dtype)
            count = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
            region_feats[:, region] = (feat_flat * mask).sum(dim=2) / count.squeeze(-1)
            counts[:, region] = count.squeeze(-1)
        valid = (counts.squeeze(-1) > 0).float()
        return region_feats[:, 1:], valid[:, 1:]


class RegionAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, num_regions: int = 21):
        super().__init__()
        self.region_embed = nn.Embedding(num_regions + 1, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, tokens, _ = x.shape
        region_ids = torch.arange(1, tokens + 1, device=x.device).unsqueeze(0).expand(batch, -1)
        h = x + self.region_embed(region_ids)
        key_padding_mask = valid_mask <= 0
        attn_out, attn = self.attn(
            self.norm1(h),
            self.norm1(h),
            self.norm1(h),
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        h = h + attn_out
        h = h + self.ffn(self.norm2(h))
        return h, attn


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: float) -> torch.Tensor:
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.weight * grad_output, None


def grad_reverse(x: torch.Tensor, weight: float) -> torch.Tensor:
    return GradientReverse.apply(x, weight)


class ARANetV4(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        feature_dim: int = 160,
        encoder_max_channels: int = 192,
        num_heads: int = 4,
        num_attn_layers: int = 2,
        dropout: float = 0.25,
        num_domains: int = 4,
    ):
        super().__init__()
        self.encoder = FeatureEncoder3D(base_channels, encoder_max_channels, dropout * 0.35)
        self.proj = nn.Sequential(
            nn.Conv3d(self.encoder.out_channels, feature_dim, 1, bias=False),
            nn.BatchNorm3d(feature_dim),
            nn.SiLU(inplace=True),
        )
        self.region_pool = RegionPooling(21)
        self.attn_layers = nn.ModuleList([
            RegionAttentionBlock(feature_dim, num_heads, dropout, 21)
            for _ in range(num_attn_layers)
        ])
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.region_readout = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 1),
        )
        self.fuse = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        self.classifier = nn.Linear(feature_dim, 3)
        self.ordinal_head = nn.Linear(feature_dim, 2)
        self.adcn_head = nn.Linear(feature_dim, 1)
        self.domain_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_domains),
        )

    def forward(
        self,
        image: torch.Tensor,
        segmentation: torch.Tensor,
        domain_grl: float = 0.0,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        feats = self.proj(self.encoder(image))
        region_feats, valid = self.region_pool(feats, segmentation)
        attentions = []
        x = region_feats
        for layer in self.attn_layers:
            x, attn = layer(x, valid, return_attention=return_attention)
            if attn is not None:
                attentions.append(attn)
        readout_logits = self.region_readout(x).squeeze(-1)
        readout_logits = readout_logits.masked_fill(valid <= 0, -1e4)
        readout = torch.softmax(readout_logits, dim=1)
        region_repr = (x * readout.unsqueeze(-1)).sum(dim=1)
        global_repr = self.global_pool(feats).flatten(1)
        fused = self.fuse(torch.cat([region_repr, global_repr], dim=1))
        out = {
            "logits": self.classifier(fused),
            "ordinal_logits": self.ordinal_head(fused),
            "adcn_logit": self.adcn_head(fused).squeeze(1),
            "domain_logits": self.domain_head(grad_reverse(fused, domain_grl)) if domain_grl > 0 else self.domain_head(fused.detach()),
            "features": fused,
            "region_readout": readout,
            "valid_regions": valid,
        }
        if attentions:
            out["attention"] = torch.stack(attentions, dim=1)
        return out


def class_weight(labels: Sequence[int], mode: str, device: torch.device) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=3).astype(np.float64)
    weights = counts.sum() / (counts * 3.0 + 1e-8)
    if mode == "sqrt":
        weights = np.sqrt(weights)
    return torch.tensor(weights.astype(np.float32), device=device)


def make_sampler(labels: Sequence[int], mode: str) -> Optional[WeightedRandomSampler]:
    if mode == "none":
        return None
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=3).astype(np.float64)
    weights = counts.sum() / (counts + 1e-8)
    if mode == "sqrt":
        weights = np.sqrt(weights)
    sample_weights = [weights[int(label)] for label in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def focal_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor],
    gamma: float,
    label_smoothing: float,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits,
        target,
        weight=weight,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    if gamma <= 0:
        return ce.mean()
    pt = torch.exp(-ce)
    return (((1.0 - pt) ** gamma) * ce).mean()


def ordinal_loss(ordinal_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    targets = torch.stack([(labels > 0).float(), (labels > 1).float()], dim=1)
    return F.binary_cross_entropy_with_logits(ordinal_logits, targets)


def adcn_loss(adcn_logit: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = (labels == 0) | (labels == 2)
    if not mask.any():
        return adcn_logit.sum() * 0.0
    targets = (labels[mask] == 2).float()
    return F.binary_cross_entropy_with_logits(adcn_logit[mask], targets)


def batch_prior_loss(logits: torch.Tensor, labels: torch.Tensor, smoothing: float = 0.06) -> torch.Tensor:
    """Keep batch-level predictions from collapsing to one class.

    The target prior is the empirical class mix of the current batch with a
    small uniform smoothing term, so the loss does not force an artificial
    global prevalence.
    """
    probs = torch.softmax(logits.float(), dim=1).mean(dim=0)
    target = torch.bincount(labels, minlength=3).float().to(logits.device)
    target = target / target.sum().clamp_min(1.0)
    target = (1.0 - smoothing) * target + smoothing / 3.0
    return F.kl_div(torch.log(probs.clamp_min(1e-6)), target, reduction="sum")


def binary_auc(y_true_binary: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = y_true_binary == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    ece = 0.0
    for start in np.linspace(0.0, 1.0, n_bins, endpoint=False):
        end = start + 1.0 / n_bins
        mask = (conf >= start) & ((conf < end) if end < 1.0 else (conf <= end))
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - conf[mask].mean()))
    return ece


def classification_metrics(y_true: Sequence[int], probs: np.ndarray) -> Dict[str, object]:
    y = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=np.float64)
    pred = probs.argmax(axis=1)
    cm = np.zeros((3, 3), dtype=int)
    for yi, pi in zip(y, pred):
        cm[int(yi), int(pi)] += 1
    support = cm.sum(axis=1)
    per_class = {}
    recalls = []
    f1s = []
    for idx, name in enumerate(CLASS_NAMES):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - tp)
        fn = float(cm[idx, :].sum() - tp)
        tn = float(cm.sum() - tp - fp - fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "support": int(support[idx]),
        }
        if support[idx] > 0:
            recalls.append(recall)
            f1s.append(f1)
    aucs = {}
    valid_aucs = []
    for idx, name in enumerate(CLASS_NAMES):
        y_bin = (y == idx).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            auc = float("nan")
        elif roc_auc_score is not None:
            auc = float(roc_auc_score(y_bin, probs[:, idx]))
        else:
            auc = binary_auc(y_bin, probs[:, idx])
        aucs[name] = None if math.isnan(auc) else auc
        if not math.isnan(auc):
            valid_aucs.append(auc)
    clipped = np.clip(probs, 1e-8, 1.0)
    one_hot = np.eye(3)[y]
    out = {
        "n": int(len(y)),
        "label_counts": {CLASS_NAMES[i]: int(support[i]) for i in range(3) if support[i] > 0},
        "prediction_distribution": {CLASS_NAMES[i]: int((pred == i).sum()) for i in range(3)},
        "acc": float((pred == y).mean()),
        "balanced_acc": float(np.mean(recalls)) if recalls else None,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "macro_auc_ovr": float(np.mean(valid_aucs)) if valid_aucs else None,
        "per_class_auc_ovr": aucs,
        "nll": float(-np.log(clipped[np.arange(len(y)), y]).mean()),
        "brier_multiclass": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "ece_15bin": expected_calibration_error(y, probs),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }
    cn = y == 0
    ad = y == 2
    if cn.sum() and ad.sum():
        score = probs[:, 2] - probs[:, 0]
        yy = np.concatenate([np.zeros(int(cn.sum())), np.ones(int(ad.sum()))])
        ss = np.concatenate([score[cn], score[ad]])
        if roc_auc_score is not None:
            out["ad_vs_cn_auc"] = float(roc_auc_score(yy, ss))
        else:
            out["ad_vs_cn_auc"] = binary_auc(yy, ss)
    if len(set(y.tolist())) == 1 and int(y[0]) == 0:
        out["cn_retention_rate"] = out["acc"]
        out["false_impairment_rate"] = float(1.0 - out["acc"])
    return out


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    output_csv: Optional[Path] = None,
) -> Dict[str, object]:
    model.eval()
    y_true: List[int] = []
    prob_rows: List[np.ndarray] = []
    csv_rows: List[dict] = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        seg = batch["segmentation"].to(device, non_blocking=True)
        labels = batch["label"].numpy().astype(int)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            out = model(image, seg)
            probs = torch.softmax(out["logits"].float(), dim=1).cpu().numpy()
        y_true.extend(labels.tolist())
        prob_rows.extend(list(probs))
        for idx in range(len(labels)):
            pred = int(probs[idx].argmax())
            csv_rows.append(
                {
                    "dataset": batch["dataset"][idx],
                    "split": batch["split"][idx],
                    "subject_id": batch["subject_id"][idx],
                    "scan_id": batch["scan_id"][idx],
                    "y_true": CLASS_NAMES[int(labels[idx])],
                    "y_pred": CLASS_NAMES[pred],
                    "prob_CN": float(probs[idx, 0]),
                    "prob_MCI": float(probs[idx, 1]),
                    "prob_AD": float(probs[idx, 2]),
                }
            )
    probs_np = np.stack(prob_rows, axis=0)
    metrics = classification_metrics(y_true, probs_np)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "dataset", "split", "subject_id", "scan_id", "y_true", "y_pred",
            "prob_CN", "prob_MCI", "prob_AD",
        ]
        with output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
    return metrics


def rows_for_splits(rows: Sequence[dict], splits: Sequence[str]) -> List[dict]:
    wanted = set(splits)
    return [row for row in rows if row["split"] in wanted]


def make_loader(
    rows: Sequence[dict],
    batch_size: int,
    num_workers: int,
    train: bool,
    sampler_mode: str = "none",
    cache_images: bool = False,
) -> DataLoader:
    ds = ManifestDataset(rows, augment=train, cache_images=cache_images)
    sampler = None
    shuffle = train
    if train and sampler_mode != "none":
        sampler = make_sampler([row["label"] for row in rows], sampler_mode)
        shuffle = False
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def selection_score(metrics_by_split: Dict[str, dict], selection_splits: Sequence[str]) -> float:
    values = []
    for split in selection_splits:
        metrics = metrics_by_split.get(split)
        if not metrics:
            continue
        n = max(int(metrics.get("n") or 0), 1)
        pred_dist = metrics.get("prediction_distribution") or {}
        pred_props = np.array([pred_dist.get(name, 0) / n for name in CLASS_NAMES], dtype=float)
        pred_entropy = float(-(pred_props * np.log(pred_props + 1e-8)).sum() / np.log(len(CLASS_NAMES)))
        pred_collapse_penalty = float(max(0.0, pred_props.max() - 0.82))
        bacc = metrics.get("balanced_acc") or 0.0
        auc = metrics.get("macro_auc_ovr") or 0.0
        mci_recall = metrics.get("per_class", {}).get("MCI", {}).get("recall", 0.0)
        ad_recall = metrics.get("per_class", {}).get("AD", {}).get("recall", 0.0)
        ece = metrics.get("ece_15bin") or 0.0
        if metrics.get("label_counts") == {"CN": metrics.get("n")}:
            # Healthy external cohorts are specificity controls: reward keeping
            # CN as CN and strongly penalize false impairment calls.
            cn_retention = metrics.get("cn_retention_rate") or metrics.get("acc") or 0.0
            values.append(0.82 * cn_retention + 0.12 * (1.0 - ece) + 0.06 * pred_entropy)
            continue
        minority_recall = min(mci_recall, ad_recall)
        values.append(
            0.48 * bacc
            + 0.22 * auc
            + 0.18 * minority_recall
            + 0.07 * pred_entropy
            + 0.05 * (1.0 - ece)
            - 0.40 * pred_collapse_penalty
        )
    return float(np.mean(values)) if values else -1e9


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    args_dict = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    rows = read_manifest(args.manifest)
    train_rows = rows_for_splits(rows, parse_split_list(args.train_splits))
    val_rows = rows_for_splits(rows, parse_split_list(args.val_splits))
    if not train_rows:
        raise ValueError(f"No train rows for {args.train_splits}")
    if not val_rows:
        raise ValueError(f"No val rows for {args.val_splits}")

    eval_splits = parse_split_list(args.eval_splits)
    eval_rows_by_split = {split: rows_for_splits(rows, [split]) for split in eval_splits}
    eval_rows_by_split = {k: v for k, v in eval_rows_by_split.items() if v}
    selection_splits = parse_split_list(args.selection_splits)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = ARANetV4(
        base_channels=args.base_channels,
        feature_dim=args.feature_dim,
        encoder_max_channels=args.encoder_max_channels,
        num_heads=args.num_heads,
        num_attn_layers=args.num_attn_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params:,}", flush=True)
    print(f"[data] train={len(train_rows)} val={len(val_rows)} eval={ {k: len(v) for k, v in eval_rows_by_split.items()} }", flush=True)

    train_loader = make_loader(train_rows, args.batch_size, args.num_workers, True, args.sampler, args.cache_images)
    val_loader = make_loader(val_rows, args.batch_size, args.num_workers, False, "none", args.cache_images)
    eval_loaders = {
        split: make_loader(split_rows, args.batch_size, args.num_workers, False, "none", args.cache_images)
        for split, split_rows in eval_rows_by_split.items()
    }

    ce_weight = class_weight([row["label"] for row in train_rows], args.class_weight, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_score = -1e9
    best_epoch = 0
    history = []
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sums = Counter()
        n_seen = 0
        epoch_start = time.time()
        domain_weight = args.domain_loss_weight * min(1.0, epoch / max(args.domain_warmup_epochs, 1))
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            seg = batch["segmentation"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            domain = batch["domain"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                out = model(image, seg, domain_grl=domain_weight)
                ce = focal_ce(out["logits"], label, ce_weight, args.focal_gamma, args.label_smoothing)
                ord_loss = ordinal_loss(out["ordinal_logits"], label)
                bin_loss = adcn_loss(out["adcn_logit"], label)
                dom_loss = F.cross_entropy(out["domain_logits"], domain)
                prior_loss = batch_prior_loss(out["logits"], label)
                loss = (
                    ce
                    + args.ordinal_loss_weight * ord_loss
                    + args.adcn_loss_weight * bin_loss
                    + args.domain_loss_weight * dom_loss
                    + args.prior_loss_weight * prior_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            batch_n = int(label.numel())
            n_seen += batch_n
            loss_sums["loss"] += float(loss.detach()) * batch_n
            loss_sums["ce"] += float(ce.detach()) * batch_n
            loss_sums["ordinal"] += float(ord_loss.detach()) * batch_n
            loss_sums["adcn"] += float(bin_loss.detach()) * batch_n
            loss_sums["domain"] += float(dom_loss.detach()) * batch_n
            loss_sums["prior"] += float(prior_loss.detach()) * batch_n
        scheduler.step()

        metrics_by_split = {"val": evaluate(model, val_loader, device, args.amp)}
        for split, loader in eval_loaders.items():
            metrics_by_split[split] = evaluate(model, loader, device, args.amp)
        score = selection_score(metrics_by_split, selection_splits)
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": float(time.time() - epoch_start),
            "selection_score": score,
            "train_loss": {k: float(v / max(n_seen, 1)) for k, v in loss_sums.items()},
            "metrics": metrics_by_split,
        }
        history.append(row)
        print(
            f"[epoch {epoch:03d}] score={score:.4f} "
            f"val_bacc={metrics_by_split['val'].get('balanced_acc'):.4f} "
            f"val_auc={metrics_by_split['val'].get('macro_auc_ovr')} "
            f"loss={row['train_loss'].get('loss', 0):.4f}",
            flush=True,
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": args_dict,
                    "n_params": n_params,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "metrics": metrics_by_split,
                },
                args.out_dir / "best_model.pt",
            )
            for split, loader in eval_loaders.items():
                evaluate(model, loader, device, args.amp, args.out_dir / f"best_predictions_{split}.csv")
            evaluate(model, val_loader, device, args.amp, args.out_dir / "best_predictions_val.csv")
            print(f"[saved] best_model.pt epoch={best_epoch} score={best_score:.4f}", flush=True)

        with (args.out_dir / "history.json").open("w") as handle:
            json.dump(history, handle, indent=2)

    checkpoint = torch.load(args.out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final_metrics = {"val": evaluate(model, val_loader, device, args.amp, args.out_dir / "final_predictions_val.csv")}
    for split, loader in eval_loaders.items():
        final_metrics[split] = evaluate(model, loader, device, args.amp, args.out_dir / f"final_predictions_{split}.csv")
    summary = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "n_params": n_params,
        "elapsed_seconds": float(time.time() - train_start),
        "args": args_dict,
        "metrics": final_metrics,
    }
    with (args.out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[done] best_epoch={best_epoch} best_score={best_score:.4f}", flush=True)
    for split, metrics in final_metrics.items():
        print(
            f"[final] {split}: n={metrics['n']} acc={metrics['acc']:.4f} "
            f"bacc={metrics.get('balanced_acc')} auc={metrics.get('macro_auc_ovr')}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-splits", default="train")
    parser.add_argument("--val-splits", default="val")
    parser.add_argument(
        "--eval-splits",
        default="internal_test,aibl_adapt_train,aibl_adapt_val,aibl_heldout,oasis_external,ixi_external",
    )
    parser.add_argument("--selection-splits", default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=160)
    parser.add_argument("--encoder-max-channels", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-attn-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--sampler", choices=["none", "sqrt", "balanced"], default="sqrt")
    parser.add_argument("--class-weight", choices=["none", "sqrt", "balanced"], default="sqrt")
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--ordinal-loss-weight", type=float, default=0.20)
    parser.add_argument("--adcn-loss-weight", type=float, default=0.15)
    parser.add_argument("--domain-loss-weight", type=float, default=0.0)
    parser.add_argument("--domain-warmup-epochs", type=int, default=15)
    parser.add_argument("--prior-loss-weight", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cache-images", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
