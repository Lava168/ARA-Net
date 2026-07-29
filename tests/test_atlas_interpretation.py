from __future__ import annotations

from src.atlas import REGION_NAMES, ad_key_label_set, n_atlas_regions
from src.interpretation import ad_key_concentration_score
from src.models import n_locked_streams


def test_atlas_has_twenty_one_regions():
    assert n_atlas_regions() == 21
    assert len(REGION_NAMES) == 21
    assert len(ad_key_label_set()) == 6


def test_six_locked_probability_streams():
    assert n_locked_streams() == 6


def test_ad_key_concentration_exceeds_uniform_when_mass_on_key():
    scores = {name: 0.1 for name in REGION_NAMES}
    for name in ("L-Hippocampus", "R-Hippocampus", "L-Amygdala", "R-Amygdala", "L-Lat-Ventricle", "R-Lat-Ventricle"):
        scores[name] = 1.0
    out = ad_key_concentration_score(scores)
    assert out["score"] > out["null_uniform"]
