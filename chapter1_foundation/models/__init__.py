"""Models for Chapter 1 / ARA-Net.

Public API:
    :class:`AtlasGuidedAttentionModel` — manuscript-aligned ARA-Net.
    :func:`create_model`               — convenience factory.
    Baselines: :class:`ResNet3D`, :class:`ViT3D`, :class:`PlainCNN3D`.
"""
from .atlas_guided_model import (
    AtlasGuidedAttentionModel,
    create_model,
    EXPECTED_INPUT_SHAPE,
    EXPECTED_FEATURE_SHAPE,
    EXPECTED_NUM_REGION_TOKENS,
    EXPECTED_FEATURE_DIM,
    EXPECTED_NUM_HEADS,
    EXPECTED_HEAD_DIM,
    EXPECTED_NUM_ATTN_LAYERS,
    EXPECTED_NUM_CLASSES,
)
from .baselines import ResNet3D, ViT3D, PlainCNN3D

__all__ = [
    "AtlasGuidedAttentionModel", "create_model",
    "ResNet3D", "ViT3D", "PlainCNN3D",
    "EXPECTED_INPUT_SHAPE", "EXPECTED_FEATURE_SHAPE",
    "EXPECTED_NUM_REGION_TOKENS", "EXPECTED_FEATURE_DIM",
    "EXPECTED_NUM_HEADS", "EXPECTED_HEAD_DIM",
    "EXPECTED_NUM_ATTN_LAYERS", "EXPECTED_NUM_CLASSES",
]
