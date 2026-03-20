"""
Chapter 1: Atlas-Guided Attention Model (ARA-Net)

Architecture: 3D CNN encoder -> region-pooled atlas attention -> classifier.

Key insight: instead of running self-attention over all spatial tokens (252),
we pool features by anatomical region (21 regions), reducing the sequence
length dramatically. This makes the attention mechanism tractable with limited
training data while providing directly interpretable region-level attention.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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
    def __init__(self, in_channels: int = 1, base_channels: int = 32,
                 num_stages: int = 4, dropout: float = 0.1):
        super().__init__()
        self.stem = ConvBlock3D(in_channels, base_channels)
        self.stages = nn.ModuleList()
        in_ch = base_channels
        for i in range(num_stages):
            out_ch = base_channels * (2 ** i)
            self.stages.append(nn.Sequential(
                ConvBlock3D(in_ch, out_ch, stride=2),
                ResBlock3D(out_ch, dropout=dropout),
            ))
            in_ch = out_ch
        self.out_channels = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x


class RegionPooling(nn.Module):
    """Pool spatial features into region-level tokens using atlas segmentation.

    Given features (B, C, D, H, W) and downsampled segmentation (B, D, H, W),
    produces (B, num_regions, C) by averaging features within each region.
    """
    def __init__(self, num_regions: int = 21):
        super().__init__()
        self.num_regions = num_regions

    def forward(self, features: torch.Tensor, seg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, D, H, W = features.shape
        feat_flat = features.view(B, C, -1)    # (B, C, N)
        seg_flat = seg.view(B, -1)              # (B, N)

        region_feats = torch.zeros(B, self.num_regions + 1, C,
                                   device=features.device, dtype=features.dtype)
        region_counts = torch.zeros(B, self.num_regions + 1, 1,
                                    device=features.device, dtype=features.dtype)

        for r in range(self.num_regions + 1):
            mask = (seg_flat == r).unsqueeze(1).float()  # (B, 1, N)
            count = mask.sum(dim=2, keepdim=True).clamp(min=1)  # (B, 1, 1)
            pooled = (feat_flat * mask).sum(dim=2)  # (B, C)
            region_feats[:, r] = pooled / count.squeeze(-1)
            region_counts[:, r] = count.squeeze(-1)

        valid_mask = (region_counts.squeeze(-1) > 0).float()
        return region_feats[:, 1:], valid_mask[:, 1:]


class AnatomyGuidedAttention(nn.Module):
    """Multi-head attention on region-pooled tokens with atlas embedding.

    Sequence length = num_regions (21) instead of spatial tokens (252+),
    making it tractable for small datasets.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.3,
                 num_regions: int = 21):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_regions = num_regions

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

    def forward(self, region_feats: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, C = region_feats.shape

        region_ids = torch.arange(1, N + 1, device=region_feats.device).unsqueeze(0).expand(B, -1)
        region_emb = self.region_embed(region_ids)
        x = region_feats + region_emb

        residual = x
        x = self.norm1(x)

        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.to_k(x), "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.to_v(x), "b n (h d) -> b h n d", h=self.num_heads)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if valid_mask is not None:
            mask = valid_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.proj_drop(self.proj(out))
        x = residual + out

        x = x + self.ffn(self.norm2(x))

        if return_attention:
            return x, attn
        return x, None


class AtlasGuidedAttentionModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        feature_dim: int = 128,
        num_heads: int = 4,
        num_classes: int = 3,
        num_regions: int = 21,
        use_atlas_conditioning: bool = True,
        dropout: float = 0.3,
        num_attn_layers: int = 2,
    ):
        super().__init__()
        self.use_atlas_conditioning = use_atlas_conditioning
        self.num_regions = num_regions

        self.encoder = FeatureEncoder3D(in_channels, base_channels, dropout=dropout * 0.5)
        self.proj = nn.Sequential(
            nn.Conv3d(self.encoder.out_channels, feature_dim, 1),
            nn.BatchNorm3d(feature_dim),
            nn.GELU(),
        )
        self.region_pool = RegionPooling(num_regions)

        self.attn_layers = nn.ModuleList([
            AnatomyGuidedAttention(feature_dim, num_heads, dropout=dropout,
                                   num_regions=num_regions)
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
        self.global_classifier = nn.Sequential(
            nn.Linear(feature_dim, num_classes),
        )

    def forward(
        self,
        image: torch.Tensor,
        segmentation: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        feats = self.encoder(image)
        feats = self.proj(feats)

        global_feat = self.global_pool(feats).flatten(1)

        if segmentation is not None and self.use_atlas_conditioning:
            seg_down = F.interpolate(
                segmentation.unsqueeze(1).float(),
                size=feats.shape[2:],
                mode="nearest",
            ).squeeze(1).long()

            region_feats, valid_mask = self.region_pool(feats, seg_down)

            attn_weights = None
            x = region_feats
            for layer in self.attn_layers:
                x, aw = layer(x, valid_mask=valid_mask,
                              return_attention=return_attention)
                if return_attention and aw is not None:
                    attn_weights = aw

            if valid_mask is not None:
                mask_expanded = valid_mask.unsqueeze(-1)
                pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = x.mean(dim=1)

            pooled = self.pool_norm(pooled)
            logits = self.classifier(pooled)

            outputs = {"logits": logits, "pooled": pooled}
            if return_attention:
                outputs["attention"] = attn_weights
            if return_features:
                outputs["spatial_features"] = feats
                outputs["seg_downsampled"] = seg_down
                outputs["region_features"] = x
            return outputs
        else:
            pooled = global_feat
            logits = self.global_classifier(pooled)
            return {"logits": logits, "pooled": pooled}


def create_model(**kwargs) -> AtlasGuidedAttentionModel:
    return AtlasGuidedAttentionModel(**kwargs)
