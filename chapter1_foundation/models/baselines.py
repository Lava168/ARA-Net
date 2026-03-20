"""
Baseline models for comparison:
1. ResNet3D  - standard 3D ResNet-18 with strong regularization
2. ViT3D     - 3D Vision Transformer (reduced depth for small data)
3. PlainCNN  - plain 3D CNN (no attention, no atlas)
"""
from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, downsample=None,
                 dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.downsample = downsample
        self.drop = nn.Dropout3d(dropout)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(self.bn2(self.conv2(out)))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity)


class ResNet3D(nn.Module):
    """3D ResNet-18 baseline with dropout regularization."""

    def __init__(self, in_channels: int = 1, num_classes: int = 3, dropout: float = 0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, 2, dropout=dropout*0.25)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, dropout=dropout*0.5)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, dropout=dropout*0.75)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, dropout=dropout)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def _make_layer(self, in_ch, out_ch, blocks, stride=1, dropout=0.0):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        layers = [BasicBlock3D(in_ch, out_ch, stride, downsample, dropout=dropout)]
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_ch, out_ch, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, image, segmentation=None, return_attention=False, return_features=False):
        x = self.stem(image)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        pooled = self.avgpool(x).flatten(1)
        logits = self.classifier(pooled)
        outputs = {"logits": logits, "pooled": pooled}
        if return_features:
            outputs["spatial_features"] = x
        return outputs


class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels=1, embed_dim=128, patch_size=(8, 8, 8)):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, "b c d h w -> b (d h w) c")


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
        self.drop_path = nn.Dropout(dropout * 0.5)

    def forward(self, x, need_weights=False):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm, need_weights=need_weights)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn_weights


class ViT3D(nn.Module):
    """3D Vision Transformer — reduced depth and width for small datasets."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        embed_dim: int = 128,
        depth: int = 3,
        num_heads: int = 4,
        patch_size: tuple = (8, 8, 8),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size)
        n_patches = (96 // patch_size[0]) * (112 // patch_size[1]) * (96 // patch_size[2])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, image, segmentation=None, return_attention=False, return_features=False):
        x = self.patch_embed(image)
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed[:, :x.shape[1]])

        attn_weights = None
        for blk in self.blocks:
            x, aw = blk(x, need_weights=return_attention)
            if return_attention:
                attn_weights = aw

        x = self.norm(x)
        pooled = x[:, 0]
        logits = self.classifier(pooled)
        outputs = {"logits": logits, "pooled": pooled}
        if return_attention and attn_weights is not None:
            outputs["attention"] = attn_weights
        return outputs


class PlainCNN3D(nn.Module):
    """Plain 3D CNN without attention (ablation baseline)."""

    def __init__(self, in_channels=1, num_classes=3, base_ch=32, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, base_ch, 3, stride=2, padding=1), nn.BatchNorm3d(base_ch), nn.ReLU(),
            nn.Dropout3d(dropout * 0.25),
            nn.Conv3d(base_ch, base_ch*2, 3, stride=2, padding=1), nn.BatchNorm3d(base_ch*2), nn.ReLU(),
            nn.Dropout3d(dropout * 0.5),
            nn.Conv3d(base_ch*2, base_ch*4, 3, stride=2, padding=1), nn.BatchNorm3d(base_ch*4), nn.ReLU(),
            nn.Dropout3d(dropout * 0.75),
            nn.Conv3d(base_ch*4, base_ch*8, 3, stride=2, padding=1), nn.BatchNorm3d(base_ch*8), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_ch * 8, num_classes),
        )

    def forward(self, image, segmentation=None, return_attention=False, return_features=False):
        x = self.features(image)
        pooled = self.pool(x).flatten(1)
        logits = self.classifier(pooled)
        outputs = {"logits": logits, "pooled": pooled}
        if return_features:
            outputs["spatial_features"] = x
        return outputs
