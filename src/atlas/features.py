"""Public synthetic / template atlas-feature construction helpers."""

from __future__ import annotations

from typing import Mapping

from src.atlas.regions import REGION_NAMES

# Class-conditional templates for smoke-test feature tables (not real MRI).
CLASS_TEMPLATES = {
    "CN": {
        "atlas_hippocampus_volume": 0.0084,
        "atlas_amygdala_volume": 0.0033,
        "atlas_lateral_ventricle_volume": 0.0320,
        "atlas_cortex_volume": 0.420,
        "atlas_ad_like_z": -0.10,
    },
    "MCI": {
        "atlas_hippocampus_volume": 0.0078,
        "atlas_amygdala_volume": 0.0030,
        "atlas_lateral_ventricle_volume": 0.0430,
        "atlas_cortex_volume": 0.405,
        "atlas_ad_like_z": 0.60,
    },
    "AD": {
        "atlas_hippocampus_volume": 0.0073,
        "atlas_amygdala_volume": 0.0029,
        "atlas_lateral_ventricle_volume": 0.0570,
        "atlas_cortex_volume": 0.390,
        "atlas_ad_like_z": 1.25,
    },
}

CORE_CLINICAL_VARIABLES = ("age", "sex", "education", "APOE4", "MMSE", "CDR_SB")


def feature_row_from_label(y_true: str) -> dict[str, float]:
    if y_true not in CLASS_TEMPLATES:
        raise KeyError(f"Unsupported label for synthetic features: {y_true}")
    return dict(CLASS_TEMPLATES[y_true])


def public_feature_names() -> list[str]:
    return [
        "atlas_hippocampus_volume",
        "atlas_amygdala_volume",
        "atlas_lateral_ventricle_volume",
        "atlas_cortex_volume",
        "atlas_ad_like_z",
        *CORE_CLINICAL_VARIABLES,
    ]


def describe_atlas() -> Mapping[str, object]:
    return {
        "n_regions": len(REGION_NAMES),
        "region_names": list(REGION_NAMES),
        "note": "Restricted FreeSurfer volumes are not redistributed; use local site processing.",
    }
