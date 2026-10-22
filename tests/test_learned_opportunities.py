import pytest

from musicmix.learned_opportunities import (
    filter_supported_proposals, nearest_support, select_diverse, validate_opportunities,
)


def proposal(identity="x", a="a", b="b", out=1000, incoming=500, score=.8):
    return {"candidate": {"candidate_id": identity, "track_a": a, "track_b": b,
                          "cue_out_sample": out, "cue_in_sample": incoming}, "score": score,
            "raw_features": {"original_structure_cue_samples": {"a": 900, "b": 400}}}


def test_identity_and_timeline_are_required():
    track = {"track_id": "hash", "sample_rate": 100, "duration_samples": 1000}
    payload = dict(track, status="complete", cues=[{"time_seconds": 2.01}])
    assert validate_opportunities(payload, track)[0]["sample"] == 201
    with pytest.raises(ValueError, match="track_id"):
        validate_opportunities(dict(payload, track_id="different_version"), track)
    with pytest.raises(ValueError, match="duration"):
        validate_opportunities(dict(payload, cues=[{"time_seconds": 10.0}]), track)
    with pytest.raises(ValueError, match="complete"):
        validate_opportunities(dict(payload, status="unavailable"), track)
    with pytest.raises(ValueError, match="duplicate"):
        validate_opportunities(dict(payload, cues=[{"time_seconds": 2.01}, {"time_seconds": 2.011}]), track)


def test_nearest_support_inclusive_and_deterministic():
    cues = [{"sample": 200}, {"sample": 100}]
    assert nearest_support(150, cues, 50)["model_cue"]["sample"] == 100
    assert nearest_support(151, cues, 48) is None
    assert nearest_support(150, [], 50) is None


def test_both_rendered_cues_need_support_not_original_anchors():
    p = proposal()
    known = {"a": [{"sample": 900}], "b": [{"sample": 400}]}
    accepted, counts = filter_supported_proposals([p], known, {"a": 10, "b": 10})
    assert accepted == []  # Original beat anchors match, actual word-safe shifts do not.
    known = {"a": [{"sample": 1000}], "b": [{"sample": 500}]}
    accepted, counts = filter_supported_proposals([p], known, {"a": 10, "b": 10})
    assert accepted[0]["score"] == p["score"]
    assert counts["both_supported"] == 1
    assert accepted[0]["learned_opportunity_support"]["incoming"]["calibrated_confidence"] is None


def test_missing_provider_is_not_support():
    accepted, counts = filter_supported_proposals([proposal()], {"a": []}, {"a": 10})
    assert not accepted and counts["missing_provider"] == 1


def test_diversity_covers_same_track_in_opposite_role():
    items = [proposal(), proposal("y", "c", "a", 2000, 1050, .7),
             proposal("z", "c", "a", 3000, 3000, .6)]
    selected = select_diverse(items, maximum=3, sample_rates={k: 100 for k in "abc"}, spacing_seconds=1)
    assert [p["candidate"]["candidate_id"] for p in selected] == ["x", "z"]
