"""Anatomical Distance Attention Loss for atlas-guided attention.

Penalizes attention between anatomically distant brain regions using
centroid-based inter-region Euclidean distances computed from the atlas
segmentation. This encourages the model to preferentially attend to
anatomically proximal structures.

NOTE: The inter-region distances are centroid-based Euclidean distances,
NOT surface geodesic distances. We use the term "anatomical distance" to
accurately describe the constraint. True geodesic computation would require
cortical surface meshes (e.g., from FreeSurfer recon-all), which is beyond
the atlas-based pipeline used here.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AnatomicalDistanceLoss(nn.Module):
    """
    Multi-component loss for anatomy-aware attention:
    1. Distance penalty: attention * centroid_distance (penalize long-range attention)
    2. Entropy regularization: encourage focused attention distributions
    3. Sparsity regularization: L1 on mean attention
    """

    def __init__(
        self,
        distance_weight: float = 1.0,
        entropy_weight: float = 0.1,
        sparsity_weight: float = 0.01,
    ):
        super().__init__()
        self.distance_weight = distance_weight
        self.entropy_weight = entropy_weight
        self.sparsity_weight = sparsity_weight

    def forward(
        self,
        attention: torch.Tensor,
        anatomical_distances: torch.Tensor = None,
        atlas_prior: torch.Tensor = None,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=attention.device)

        if anatomical_distances is not None:
            attn_mean = attention.mean(dim=1)
            if anatomical_distances.dim() == 2:
                dist = anatomical_distances.unsqueeze(0).expand(attn_mean.shape[0], -1, -1)
            elif anatomical_distances.dim() == 3:
                dist = anatomical_distances
            else:
                dist = anatomical_distances
            dist_penalty = (attn_mean * dist).sum(dim=-1).mean()
            loss = loss + self.distance_weight * dist_penalty

        if self.entropy_weight > 0:
            entropy = -(attention * (attention + 1e-8).log()).sum(dim=-1).mean()
            loss = loss - self.entropy_weight * entropy

        if self.sparsity_weight > 0:
            sparsity = attention.mean(dim=1).norm(p=1, dim=-1).mean()
            loss = loss + self.sparsity_weight * sparsity

        return loss


# Backward-compatible alias
GeodesicAttentionLoss = AnatomicalDistanceLoss
