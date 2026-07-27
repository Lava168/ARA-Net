"""FreeSurfer/FastSurfer label definitions for public atlas evidence panels."""

FREESURFER_LABELS = {
    "left_lateral_ventricle": 4,
    "right_lateral_ventricle": 43,
    "left_hippocampus": 17,
    "right_hippocampus": 53,
    "left_amygdala": 18,
    "right_amygdala": 54,
}

AD_KEY_REGIONS = {
    "hippocampus": {
        "labels": [17, 53],
        "expected_ad_direction": "decrease",
        "rationale": "Medial temporal atrophy is a core AD-related structural pattern.",
    },
    "amygdala": {
        "labels": [18, 54],
        "expected_ad_direction": "decrease",
        "rationale": "Amygdala volume loss supports medial temporal neurodegeneration evidence.",
    },
    "lateral_ventricles": {
        "labels": [4, 43],
        "expected_ad_direction": "increase",
        "rationale": "Ventricular enlargement is consistent with global tissue loss.",
    },
}
