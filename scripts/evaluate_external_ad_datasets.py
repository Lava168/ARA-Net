#!/usr/bin/env python3
"""External validation for ARA-Net reviewer-response experiments.

This script is intentionally self-contained because the server contains
multiple historical ARA-Net code revisions. It reconstructs the model from
checkpoint keys, loads the exact weights with strict=True, and evaluates
independent AD datasets without changing any training code.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover - sklearn is available on the server.
    roc_auc_score = None


CLASS_NAMES = ["CN", "MCI", "AD"]
FS_LABELS = [
    0,
    2, 3, 4, 10, 11, 12, 13, 16, 17, 18, 26,
    41, 42, 43, 49, 50, 51, 52, 53, 54, 58,
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(FS_LABELS)}


def remap_segmentation(seg: np.ndarray) -> np.ndarray:
    """Map FreeSurfer labels to contiguous 0..21 indices."""
    out = np.zeros_like(seg, dtype=np.int64)
    for label, idx in LABEL_TO_IDX.items():
        out[seg == label] = idx
    return out


class ExternalCachedDataset(Dataset):
    def __init__(
        self,
        name: str,
        cache_dir: Path,
        include_prefix: Optional[str] = None,
        max_samples: int = 0,
    ):
        self.name = name
        self.cache_dir = Path(cache_dir)
        files = sorted(self.cache_dir.glob("*.npz"))
        if include_prefix:
            files = [p for p in files if p.stem.startswith(include_prefix)]
        if max_samples > 0:
            files = files[:max_samples]
        if not files:
            raise FileNotFoundError(f"No .npz files for {name} in {cache_dir}")

        self.samples = []
        label_counts: Counter[int] = Counter()
        for path in files:
            with np.load(path, allow_pickle=True) as data:
                label = int(data["label"]) if "label" in data.files else 0
            self.samples.append({"path": path, "label": label, "subject_id": path.stem})
            label_counts[label] += 1
        self.label_counts = {CLASS_NAMES[k]: int(v) for k, v in sorted(label_counts.items())}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        with np.load(sample["path"], allow_pickle=True) as data:
            image = data["image"].astype(np.float32)
            seg = data["seg"].astype(np.int64)
        seg = remap_segmentation(seg)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0),
            "segmentation": torch.from_numpy(np.ascontiguousarray(seg)),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "subject_id": sample["subject_id"],
            "dataset": self.name,
        }


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResBlock3D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = ConvBlock3D(channels, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm3d(channels)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.drop(self.bn(self.conv2(x)))
        return F.gelu(x + residual)


class FeatureEncoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_stages: int = 4,
        dropout: float = 0.15,
        max_channels: Optional[int] = None,
    ):
        super().__init__()
        self.stem = ConvBlock3D(in_channels, base_channels)
        self.stages = nn.ModuleList()
        in_ch = base_channels
        for i in range(num_stages):
            out_ch = base_channels * (2 ** i)
            if max_channels is not None:
                out_ch = min(out_ch, max_channels)
            self.stages.append(
                nn.Sequential(
                    ConvBlock3D(in_ch, out_ch, stride=2),
                    ResBlock3D(out_ch, dropout=dropout),
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

    def forward(self, features: torch.Tensor, seg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, channels, _, _, _ = features.shape
        feat_flat = features.view(batch, channels, -1)
        seg_flat = seg.view(batch, -1)

        region_feats = torch.zeros(
            batch, self.num_regions + 1, channels,
            device=features.device, dtype=features.dtype,
        )
        region_counts = torch.zeros(
            batch, self.num_regions + 1, 1,
            device=features.device, dtype=features.dtype,
        )

        for region in range(self.num_regions + 1):
            mask = (seg_flat == region).unsqueeze(1).float()
            count = mask.sum(dim=2, keepdim=True).clamp(min=1)
            pooled = (feat_flat * mask).sum(dim=2)
            region_feats[:, region] = pooled / count.squeeze(-1)
            region_counts[:, region] = count.squeeze(-1)

        valid_mask = (region_counts.squeeze(-1) > 0).float()
        return region_feats[:, 1:], valid_mask[:, 1:]


class AnatomyGuidedAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.3, num_regions: int = 21):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.region_embed = nn.Embedding(num_regions + 1, dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        return x.view(batch, tokens, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, tokens, head_dim = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(batch, tokens, heads * head_dim)

    def forward(
        self,
        region_feats: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, tokens, _ = region_feats.shape
        region_ids = torch.arange(1, tokens + 1, device=region_feats.device).unsqueeze(0).expand(batch, -1)
        x = region_feats + self.region_embed(region_ids)

        residual = x
        x = self.norm1(x)
        q = self._split_heads(self.to_q(x))
        k = self._split_heads(self.to_k(x))
        v = self._split_heads(self.to_v(x))
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if valid_mask is not None:
            mask_value = -1e4 if attn.dtype in (torch.float16, torch.bfloat16) else -1e9
            attn = attn.masked_fill(valid_mask.unsqueeze(1).unsqueeze(2) == 0, mask_value)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = self._merge_heads(torch.matmul(attn, v))
        x = residual + self.proj_drop(self.proj(out))
        x = x + self.ffn(self.norm2(x))
        return (x, attn) if return_attention else (x, None)


class RegionTokenARANet(nn.Module):
    """Region-token ARA-Net used by v3/SSL checkpoints."""

    def __init__(
        self,
        feature_dim: int,
        encoder_max_channels: int,
        num_attn_layers: int,
        num_heads: int = 4,
        num_regions: int = 21,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.use_atlas_conditioning = True
        self.encoder = FeatureEncoder3D(
            in_channels=1,
            base_channels=32,
            dropout=dropout * 0.5,
            max_channels=encoder_max_channels,
        )
        self.proj = nn.Sequential(
            nn.Conv3d(self.encoder.out_channels, feature_dim, 1),
            nn.BatchNorm3d(feature_dim),
            nn.GELU(),
        )
        self.region_pool = RegionPooling(num_regions)
        self.attn_layers = nn.ModuleList([
            AnatomyGuidedAttention(feature_dim, num_heads, dropout, num_regions)
            for _ in range(num_attn_layers)
        ])
        self.pool_norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.global_classifier = nn.Sequential(nn.Linear(feature_dim, num_classes))

    def forward(
        self,
        image: torch.Tensor,
        segmentation: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        feats = self.proj(self.encoder(image))
        if segmentation is None:
            pooled = self.global_pool(feats).flatten(1)
            return {"logits": self.global_classifier(pooled), "pooled": pooled}

        seg_down = F.interpolate(
            segmentation.unsqueeze(1).float(),
            size=feats.shape[2:],
            mode="nearest",
        ).squeeze(1).long()
        x, valid_mask = self.region_pool(feats, seg_down)
        attn = None
        for layer in self.attn_layers:
            x, aw = layer(x, valid_mask=valid_mask, return_attention=return_attention)
            if aw is not None:
                attn = aw
        pooled = (x * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = self.pool_norm(pooled)
        out = {"logits": self.classifier(pooled), "pooled": pooled}
        if return_attention:
            out["attention"] = attn
        return out


class LegacySpatialAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_a = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        return x.view(batch, tokens, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, tokens, head_dim = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(batch, tokens, heads * head_dim)

    def forward(
        self,
        features: torch.Tensor,
        atlas_embed: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q = self.to_q(features)
        if atlas_embed is not None:
            q = q + self.to_a(atlas_embed)
        k = self.to_k(features)
        v = self.to_v(features)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.dropout(attn)
        out = self.proj(self._merge_heads(torch.matmul(attn, v)))
        return (out, attn) if return_attention else (out, None)


class LegacySpatialARANet(nn.Module):
    """Original spatial-token ARA-Net used by archived legacy checkpoints."""

    def __init__(
        self,
        feature_dim: int,
        num_regions: int,
        num_heads: int = 8,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = FeatureEncoder3D(in_channels=1, base_channels=32, dropout=0.0, max_channels=None)
        self.proj = nn.Sequential(
            nn.Conv3d(self.encoder.out_channels, feature_dim, 1),
            nn.BatchNorm3d(feature_dim),
            nn.GELU(),
        )
        self.atlas_embed = nn.Embedding(num_regions + 1, feature_dim)
        self.attn = LegacySpatialAttention(feature_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(
        self,
        image: torch.Tensor,
        segmentation: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        feats = self.proj(self.encoder(image))
        atlas_embed = None
        if segmentation is not None:
            seg_down = F.interpolate(
                segmentation.unsqueeze(1).float(),
                size=feats.shape[2:],
                mode="nearest",
            ).squeeze(1).long()
            atlas_embed = self.atlas_embed(seg_down).view(seg_down.shape[0], -1, self.atlas_embed.embedding_dim)
        feats_flat = feats.flatten(2).transpose(1, 2)
        attn_out, attn = self.attn(feats_flat, atlas_embed=atlas_embed, return_attention=return_attention)
        pooled = self.norm(attn_out).mean(dim=1)
        out = {"logits": self.classifier(pooled), "pooled": pooled}
        if return_attention:
            out["attention"] = attn
        return out


def count_attn_layers(state: Dict[str, torch.Tensor]) -> int:
    indices = []
    for key in state:
        match = re.match(r"attn_layers\.(\d+)\.", key)
        if match:
            indices.append(int(match.group(1)))
    return max(indices) + 1 if indices else 0


def build_model_from_state(state: Dict[str, torch.Tensor]) -> Tuple[nn.Module, Dict[str, object]]:
    if "atlas_embed.weight" in state and any(k.startswith("attn.") for k in state):
        feature_dim = int(state["proj.0.weight"].shape[0])
        num_regions = int(state["atlas_embed.weight"].shape[0] - 1)
        num_classes = int(state["classifier.3.weight"].shape[0])
        model = LegacySpatialARANet(feature_dim, num_regions, num_heads=8, num_classes=num_classes)
        meta = {
            "model_family": "legacy_spatial_attention",
            "feature_dim": feature_dim,
            "num_regions": num_regions,
            "num_heads": 8,
            "num_classes": num_classes,
        }
    else:
        feature_dim = int(state["proj.0.weight"].shape[0])
        encoder_max_channels = int(state["encoder.stages.3.0.conv.weight"].shape[0])
        num_regions = int(state["attn_layers.0.region_embed.weight"].shape[0] - 1)
        num_classes = int(state["classifier.3.weight"].shape[0])
        num_attn_layers = count_attn_layers(state)
        model = RegionTokenARANet(
            feature_dim=feature_dim,
            encoder_max_channels=encoder_max_channels,
            num_attn_layers=num_attn_layers,
            num_heads=4,
            num_regions=num_regions,
            num_classes=num_classes,
        )
        meta = {
            "model_family": "region_token_attention",
            "feature_dim": feature_dim,
            "encoder_max_channels": encoder_max_channels,
            "num_regions": num_regions,
            "num_heads": 4,
            "num_attn_layers": num_attn_layers,
            "num_classes": num_classes,
        }
    return model, meta


def load_state(path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state" in obj:
        obj = obj["model_state"]
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")
    return obj


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    mat = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        mat[int(true), int(pred)] += 1
    return mat


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
    for lo in np.linspace(0, 1, n_bins, endpoint=False):
        hi = lo + 1 / n_bins
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - conf[mask].mean()))
    return ece


def classification_metrics(y_true: Sequence[int], probs: np.ndarray) -> Dict[str, object]:
    y_true_np = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true_np, y_pred, len(CLASS_NAMES))
    support = cm.sum(axis=1)
    pred_counts = Counter(int(x) for x in y_pred)

    per_class: Dict[str, Dict[str, float]] = {}
    recalls_present = []
    f1_present = []
    f1_all = []
    for idx, name in enumerate(CLASS_NAMES):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - tp)
        fn = float(cm[idx, :].sum() - tp)
        tn = float(cm.sum() - tp - fp - fn)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "support": int(support[idx]),
        }
        f1_all.append(f1)
        if support[idx] > 0:
            recalls_present.append(recall)
            f1_present.append(f1)

    aucs: Dict[str, Optional[float]] = {}
    valid_aucs = []
    for idx, name in enumerate(CLASS_NAMES):
        y_bin = (y_true_np == idx).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            auc = float("nan")
        elif roc_auc_score is not None:
            auc = float(roc_auc_score(y_bin, probs[:, idx]))
        else:
            auc = binary_auc(y_bin, probs[:, idx])
        aucs[name] = None if math.isnan(auc) else auc
        if not math.isnan(auc):
            valid_aucs.append(auc)

    one_hot = np.eye(len(CLASS_NAMES))[y_true_np]
    clipped = np.clip(probs, 1e-8, 1.0)
    metrics = {
        "n_samples": int(len(y_true_np)),
        "label_counts": {CLASS_NAMES[i]: int(v) for i, v in enumerate(support) if v > 0},
        "prediction_distribution": {CLASS_NAMES[i]: int(pred_counts.get(i, 0)) for i in range(len(CLASS_NAMES))},
        "accuracy": float((y_pred == y_true_np).mean()),
        "balanced_accuracy_present": float(np.mean(recalls_present)) if recalls_present else None,
        "macro_f1_present": float(np.mean(f1_present)) if f1_present else None,
        "macro_f1_all": float(np.mean(f1_all)),
        "macro_auc_ovr_valid": float(np.mean(valid_aucs)) if valid_aucs else None,
        "per_class_auc_ovr": aucs,
        "nll": float(-np.log(clipped[np.arange(len(y_true_np)), y_true_np]).mean()),
        "brier_multiclass": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "ece_15bin": expected_calibration_error(y_true_np, probs),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }
    if len(set(y_true_np.tolist())) == 1 and int(y_true_np[0]) == 0:
        metrics["ixi_cn_retention_rate"] = metrics["accuracy"]
        metrics["ixi_false_impairment_rate"] = float(1.0 - metrics["accuracy"])
    return metrics


def parse_dataset_specs(args: argparse.Namespace) -> Dict[str, Tuple[Path, Optional[str]]]:
    project_root = Path(args.project_root)
    specs = {
        "aibl": (Path(args.aibl_cache), None),
        "oasis": (project_root / "sample_data" / "cache_real", "OASIS"),
        "ixi": (project_root / "sample_data" / "cache_real", "IXI"),
    }
    selected = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]
    return {name: specs[name] for name in selected}


def discover_checkpoints(checkpoint_root: Path, checkpoint_glob: str, max_checkpoints: int) -> List[Path]:
    checkpoints = sorted(checkpoint_root.glob(checkpoint_glob))
    if max_checkpoints > 0:
        checkpoints = checkpoints[:max_checkpoints]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {checkpoint_root} with {checkpoint_glob}")
    return checkpoints


@torch.no_grad()
def predict_checkpoint(
    checkpoint: Path,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[np.ndarray, List[int], List[str], Dict[str, object]]:
    state = load_state(checkpoint)
    model, model_meta = build_model_from_state(state)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Strict load failed: missing={missing}, unexpected={unexpected}")
    model.to(device)
    model.eval()

    all_probs: List[np.ndarray] = []
    all_labels: List[int] = []
    all_subjects: List[str] = []
    amp_enabled = use_amp and device.type == "cuda"

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        seg = batch["segmentation"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy().astype(int).tolist()
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(images, segmentation=seg)["logits"]
            probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.detach().cpu().numpy())
        all_labels.extend(labels)
        all_subjects.extend(batch["subject_id"])
    return np.concatenate(all_probs, axis=0), all_labels, all_subjects, model_meta


def write_prediction_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    fieldnames = [
        "dataset", "checkpoint", "subject_id", "y_true", "y_pred",
        "prob_CN", "prob_MCI", "prob_AD",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Cannot serialize {type(obj)!r}")


def run(args: argparse.Namespace) -> None:
    started = time.time()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_root = Path(args.checkpoint_root)
    checkpoints = discover_checkpoints(checkpoint_root, args.checkpoint_glob, args.max_checkpoints)
    total_discovered_checkpoints = len(checkpoints)
    if args.checkpoint_shard_count > 1:
        if args.checkpoint_shard_index < 0 or args.checkpoint_shard_index >= args.checkpoint_shard_count:
            raise ValueError("--checkpoint-shard-index must be in [0, checkpoint_shard_count)")
        checkpoints = [
            ckpt for i, ckpt in enumerate(checkpoints)
            if i % args.checkpoint_shard_count == args.checkpoint_shard_index
        ]
        if not checkpoints:
            raise ValueError(
                f"Shard {args.checkpoint_shard_index}/{args.checkpoint_shard_count} has no checkpoints"
            )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    per_checkpoint_csv = output_json.with_suffix(".per_checkpoint_predictions.csv")
    ensemble_csv = output_json.with_suffix(".ensemble_predictions.csv")

    datasets = {}
    for name, (cache_dir, prefix) in parse_dataset_specs(args).items():
        ds = ExternalCachedDataset(name, cache_dir, prefix, args.max_samples)
        datasets[name] = ds
        print(f"[data] {name}: n={len(ds)} labels={ds.label_counts} cache={cache_dir}", flush=True)

    result: Dict[str, object] = {
        "created_at_unix": int(time.time()),
        "device": str(device),
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_glob": args.checkpoint_glob,
        "total_discovered_checkpoints": total_discovered_checkpoints,
        "n_checkpoints": len(checkpoints),
        "checkpoint_shard_index": args.checkpoint_shard_index,
        "checkpoint_shard_count": args.checkpoint_shard_count,
        "checkpoints": [str(p) for p in checkpoints],
        "datasets": {},
        "per_checkpoint": defaultdict(dict),
        "ensemble": {},
    }

    per_checkpoint_rows: List[Dict[str, object]] = []
    ensemble_rows: List[Dict[str, object]] = []

    for dataset_name, dataset in datasets.items():
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        result["datasets"][dataset_name] = {
            "n_samples": len(dataset),
            "label_counts": dataset.label_counts,
            "cache_dir": str(dataset.cache_dir),
        }
        checkpoint_probs = []
        labels_ref: Optional[List[int]] = None
        subjects_ref: Optional[List[str]] = None
        for ckpt_idx, checkpoint in enumerate(checkpoints, start=1):
            print(f"[eval] {dataset_name} checkpoint {ckpt_idx}/{len(checkpoints)}: {checkpoint}", flush=True)
            probs, labels, subjects, model_meta = predict_checkpoint(checkpoint, loader, device, args.amp)
            checkpoint_probs.append(probs)
            if labels_ref is None:
                labels_ref, subjects_ref = labels, subjects
                result["model_meta"] = model_meta
            metrics = classification_metrics(labels, probs)
            result["per_checkpoint"][dataset_name][str(checkpoint)] = metrics
            for subject_id, y_true, prob in zip(subjects, labels, probs):
                per_checkpoint_rows.append({
                    "dataset": dataset_name,
                    "checkpoint": str(checkpoint),
                    "subject_id": subject_id,
                    "y_true": CLASS_NAMES[int(y_true)],
                    "y_pred": CLASS_NAMES[int(np.argmax(prob))],
                    "prob_CN": float(prob[0]),
                    "prob_MCI": float(prob[1]),
                    "prob_AD": float(prob[2]),
                })
        assert labels_ref is not None and subjects_ref is not None
        ensemble_probs = np.mean(np.stack(checkpoint_probs, axis=0), axis=0)
        result["ensemble"][dataset_name] = classification_metrics(labels_ref, ensemble_probs)
        for subject_id, y_true, prob in zip(subjects_ref, labels_ref, ensemble_probs):
            ensemble_rows.append({
                "dataset": dataset_name,
                "checkpoint": "ensemble_mean_probability",
                "subject_id": subject_id,
                "y_true": CLASS_NAMES[int(y_true)],
                "y_pred": CLASS_NAMES[int(np.argmax(prob))],
                "prob_CN": float(prob[0]),
                "prob_MCI": float(prob[1]),
                "prob_AD": float(prob[2]),
            })
        print(f"[done] {dataset_name} ensemble: {result['ensemble'][dataset_name]}", flush=True)

    result["runtime_seconds"] = round(time.time() - started, 3)
    with output_json.open("w") as handle:
        json.dump(result, handle, indent=2, default=json_default)
    write_prediction_csv(per_checkpoint_csv, per_checkpoint_rows)
    write_prediction_csv(ensemble_csv, ensemble_rows)
    print(f"[saved] {output_json}", flush=True)
    print(f"[saved] {per_checkpoint_csv}", flush=True)
    print(f"[saved] {ensemble_csv}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        default="/home/lry/atlas_guided_attention Alzheimer's Disease Dynamics/chapter1_foundation",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="/home/lry/atlas_guided_attention Alzheimer's Disease Dynamics/chapter1_foundation/outputs/experiments/experiment_results_v3",
    )
    parser.add_argument("--checkpoint-glob", default="seed_*/best_model_seed*_fold*.pth")
    parser.add_argument("--aibl-cache", default="/home/lry/aibl/cache_real")
    parser.add_argument("--datasets", default="aibl,oasis,ixi")
    parser.add_argument(
        "--output-json",
        default="/home/lry/atlas_guided_attention Alzheimer's Disease Dynamics/chapter1_foundation/outputs/analysis/external_validation_v3_reviewer_grade.json",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-checkpoints", type=int, default=0)
    parser.add_argument("--checkpoint-shard-index", type=int, default=0)
    parser.add_argument("--checkpoint-shard-count", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
