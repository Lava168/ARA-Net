"""FreeSurfer/FastSurfer label definitions for public atlas evidence panels."""

from __future__ import annotations

# Coarse 21-region FreeSurfer-lite atlas used by ARA-Net multimodal features.
FS_LABELS = [
    2, 3, 4, 10, 11, 12, 13, 16, 17, 18, 26,
    41, 42, 43, 49, 50, 51, 52, 53, 54, 58,
]

REGION_NAMES = [
    "L-WM", "L-Cortex", "L-Lat-Ventricle", "L-Thalamus", "L-Caudate",
    "L-Putamen", "L-Pallidum", "Brain-Stem", "L-Hippocampus", "L-Amygdala",
    "L-Accumbens", "R-WM", "R-Cortex", "R-Lat-Ventricle", "R-Thalamus",
    "R-Caudate", "R-Putamen", "R-Pallidum", "R-Hippocampus", "R-Amygdala",
    "R-Accumbens",
]

FREESURFER_LABELS = {
    "left_lateral_ventricle": 4,
    "right_lateral_ventricle": 43,
    "left_hippocampus": 17,
    "right_hippocampus": 53,
    "left_amygdala": 18,
    "right_amygdala": 54,
    "left_accumbens": 26,
    "right_accumbens": 58,
    "left_cortex": 3,
    "right_cortex": 42,
}

AD_KEY_REGIONS = {
    "hippocampus": {
        "labels": [17, 53],
        "names": ["L-Hippocampus", "R-Hippocampus"],
        "expected_ad_direction": "decrease",
        "rationale": "Medial temporal atrophy is a core AD-related structural pattern.",
    },
    "amygdala": {
        "labels": [18, 54],
        "names": ["L-Amygdala", "R-Amygdala"],
        "expected_ad_direction": "decrease",
        "rationale": "Amygdala volume loss supports medial temporal neurodegeneration evidence.",
    },
    "lateral_ventricles": {
        "labels": [4, 43],
        "names": ["L-Lat-Ventricle", "R-Lat-Ventricle"],
        "expected_ad_direction": "increase",
        "rationale": "Ventricular enlargement is consistent with global tissue loss.",
    },
}

# Secondary pathology-sensitive parcels (ranked AD−CN contrast; optional evidence).
SECONDARY_PATHOLOGY_REGIONS = {
    "accumbens": {"labels": [26, 58], "names": ["L-Accumbens", "R-Accumbens"]},
    "cortex": {"labels": [3, 42], "names": ["L-Cortex", "R-Cortex"]},
}


def ad_key_label_set() -> set[int]:
    labels: set[int] = set()
    for spec in AD_KEY_REGIONS.values():
        labels.update(int(x) for x in spec["labels"])
    return labels


def n_atlas_regions() -> int:
    return len(REGION_NAMES)
