from __future__ import annotations

import math

import pytest

from musicmix.contracts import Candidate, Feature
from musicmix.functional_event import (
    FunctionalEventScorer,
    FunctionalEventThresholds,
    assess_functional_event,
    rerank_functional_event,
)


def _point(t: float, energy: float, events: float, bass: float, timbre: float, tonal: float):
    return {
        "log_rms_mean": math.log(max(energy, 1e-6)),
        "rms_mean": energy,
        "spectral_flux_mean": events,
        "attack_density_per_second": events * 0.5,
        "bass_ratio": bass,
        "high_frequency_ratio": timbre,
        "spectral_centroid_hz": 1000.0 + timbre * 4000.0,
        "chroma_movement": tonal,
    }


def _series(values, *, side, window=8):
    points = []
    for index, values_at_index in enumerate(values):
        point = _point(index, *values_at_index)
        point["beat_index"] = index
        points.append(point)
    return {
        "side": side,
        "cue_sample": 1000,
        "beat_anchor_sample": 1000,
        "windows": [
            {
                "window_beats": window,
                "beat_feature_series": points,
                "boundary_method": "test_observed_beats",
            }
        ],
    }


def _flat(level, n=8):
    return [level for _ in range(n)]


def _rising(start, step, n=8):
    return [(start[0] + step[0] * i, start[1] + step[1] * i, start[2] + step[2] * i,
             start[3] + step[3] * i, start[4] + step[4] * i) for i in range(n)]


def _steady(values):
    return values


def _attack_then_stable():
    # First two observed beats are the local entry event; the remainder keeps
    # most of the level instead of returning to the pre-entry baseline.
    return [
        (0.90, 0.80, 0.48, 0.42, 0.25),
        (0.88, 0.78, 0.47, 0.41, 0.24),
        (0.72, 0.62, 0.42, 0.36, 0.22),
        (0.70, 0.60, 0.41, 0.35, 0.22),
        (0.69, 0.59, 0.41, 0.35, 0.21),
        (0.70, 0.60, 0.42, 0.35, 0.22),
        (0.71, 0.61, 0.42, 0.36, 0.22),
        (0.70, 0.60, 0.41, 0.35, 0.22),
    ]


def _base_profiles():
    a_before = _series(
        _rising((0.12, 0.10, 0.18, 0.10, 0.12), (0.035, 0.028, 0.020, 0.025, 0.018)),
        side="outgoing",
    )
    a_after = _series(
        _rising((0.40, 0.34, 0.30, 0.28, 0.22), (0.020, 0.020, 0.015, 0.015, 0.010)),
        side="incoming",
    )
    b_before = _series([(0.10, 0.08, 0.20, 0.10, 0.10) for _ in range(8)], side="outgoing")
    b_after = _series(_attack_then_stable(), side="incoming")
    return a_before, a_after, b_before, b_after


def test_local_attack_and_persistence_use_b_before_and_b_after():
    profiles = _base_profiles()
    assessment = assess_functional_event(*profiles)
    assert assessment.label == "post_entry_attack_stabilized"
    assert not assessment.abstained
    assert assessment.score is not None and assessment.score > 0
    entry = assessment.entry_evidence
    assert entry is not None and entry.status == "supported"
    assert len(entry.diagnostics["positive_domains"]) >= 2
    assert entry.diagnostics["persistence_domains"]
    assert "B_before" in assessment.diagnostics["profiles"]


def test_salient_reversals_veto_continuation_but_not_independent_arrival():
    a_before, a_after, b_before, _ = _base_profiles()
    # B enters with a sustained local attack, then moves opposite to A.
    # Continuation must be rejected, but the arrival route stays independent.
    b_after = _series(
        [(value, value, value, value, value)
         for value in (0.95, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65)],
        side="incoming",
    )
    assessment = assess_functional_event(a_before, a_after, b_before, b_after)
    continuation = assessment.evidence_by_kind["continuation"]
    arrival = assessment.evidence_by_kind["directed_arrival"]
    assert continuation.status != "supported"
    assert continuation.diagnostics["salient_contradiction"] is True
    assert len(continuation.diagnostics["reversing_domains"]) >= 2
    assert arrival.status == "supported"
    assert assessment.diagnostics["continuation_arrival_conflict"] is True
    assert assessment.diagnostics["continuation_reversal_vetoes_arrival"] is False
    assert assessment.label == "post_entry_attack_stabilized"
    assert not assessment.abstained
    assert assessment.score is not None and assessment.score > 0


def test_flat_A_plus_local_attack_is_not_projected_arrival():
    flat = _flat((0.30, 0.22, 0.25, 0.18, 0.16), n=8)
    a_before = _series(flat, side="outgoing")
    a_after = _series(flat, side="incoming")
    b_before = _series(
        [(0.10, 0.08, 0.20, 0.10, 0.10) for _ in range(8)], side="outgoing"
    )
    b_after = _series(_attack_then_stable(), side="incoming")
    assessment = assess_functional_event(a_before, a_after, b_before, b_after)

    entry = assessment.entry_evidence
    local = assessment.evidence_by_kind["local_entry_event"]
    arrival = assessment.evidence_by_kind["directed_arrival"]
    directed = assessment.evidence_by_kind["a_directed_expectation"]
    assert entry is not None and entry.status == "supported"
    assert local.status == "supported"
    assert local.diagnostics["is_cross_song_resolution"] is False
    assert directed.status == "absent"
    assert arrival.status != "supported"
    assert arrival.diagnostics["flat_A_plus_local_attack_is_unproven"] is True
    assert assessment.label == "unknown"
    assert assessment.abstained is True
    assert assessment.score is None
    assert "directed_a_expectation" in assessment.missing
    assert "unprojected_local_entry_not_arrival" in assessment.missing
    assert assessment.diagnostics["trained_resolution_reward_used"] is False


def test_compatible_continued_motion_is_a_separate_route():
    a_before = _series(
        _rising((0.10, 0.10, 0.10, 0.10, 0.10), (0.035, 0.030, 0.025, 0.025, 0.020)),
        side="outgoing",
    )
    a_after = _series(
        _rising((0.35, 0.32, 0.30, 0.28, 0.25), (0.020, 0.018, 0.015, 0.015, 0.012)),
        side="incoming",
    )
    b_before = _series(
        [(0.60, 0.50, 0.30, 0.25, 0.20) for _ in range(8)], side="outgoing"
    )
    b_after = _series(
        [(0.60 + 0.015 * i, 0.50 + 0.012 * i, 0.30 + 0.008 * i,
          0.25 + 0.008 * i, 0.20 + 0.006 * i) for i in range(8)],
        side="incoming",
    )
    assessment = assess_functional_event(a_before, a_after, b_before, b_after)
    continuation = assessment.evidence_by_kind["continuation"]
    arrival = assessment.evidence_by_kind["directed_arrival"]
    assert continuation.status == "supported"
    assert arrival.status != "supported"
    assert assessment.label in {"sustained_motion", "continuation_without_local_event"}
    assert not assessment.abstained
    assert assessment.score is not None and assessment.score > 0


def test_same_b_after_different_b_before_changes_event_result():
    a_before, a_after, b_before_low, b_after = _base_profiles()
    b_before_high = {
        "side": "outgoing",
        "cue_sample": 1000,
        "beat_anchor_sample": 1000,
        "windows": [
            {
                "window_beats": 8,
                "beat_feature_series": [
                    dict(point)
                    for point in b_before_low["windows"][0]["beat_feature_series"]
                ],
                "boundary_method": "test_observed_beats",
            }
        ],
    }
    for point in b_before_high["windows"][0]["beat_feature_series"]:
        point["log_rms_mean"] = math.log(0.86)
        point["rms_mean"] = 0.86
        point["spectral_flux_mean"] = 0.76
        point["attack_density_per_second"] = 0.38
        point["bass_ratio"] = 0.45
        point["high_frequency_ratio"] = 0.40
        point["spectral_centroid_hz"] = 2600.0
    first = assess_functional_event(a_before, a_after, b_before_low, b_after)
    second = assess_functional_event(a_before, a_after, b_before_high, b_after)
    assert first.label == "post_entry_attack_stabilized"
    assert second.label != first.label
    assert second.entry_evidence.score != first.entry_evidence.score


def test_preparation_and_vocal_pickup_are_pre_entry_not_arrival():
    a_before, a_after, _, _ = _base_profiles()
    b_before = _series(
        _rising((0.10, 0.08, 0.18, 0.10, 0.10), (0.035, 0.030, 0.010, 0.025, 0.004)),
        side="outgoing",
    )
    b_after = _series(
        [(0.38, 0.31, 0.23, 0.20, 0.14) for _ in range(8)],
        side="incoming",
    )
    assessment = assess_functional_event(
        a_before,
        a_after,
        b_before,
        b_after,
        vocal_completion={"pickup": True, "provider": "unit-test"},
    )
    assert assessment.label == "pickup_or_preparation"
    preparation = assessment.evidence_by_kind["pre_entry_preparation"]
    assert preparation.status == "supported"
    assert preparation.diagnostics["explicit_vocal_pickup"] is True


def test_word_gap_only_is_not_phrase_completion_and_section_label_is_not_resolution():
    profiles = _base_profiles()
    assessment = assess_functional_event(
        *profiles,
        vocal_completion={
            "status": "supported",
            "boundary_class": "word_gap_only",
            "provider": "asr-test",
        },
        section_context={
            "before_label": "verse",
            "after_label": "chorus",
            "supported": True,
            "provider": "structure-test",
        },
    )
    voice = assessment.diagnostics["vocal_observation"]
    assert voice["completion"] == "absent"
    assert voice["word_gap_is_not_completion"] is True
    assert assessment.diagnostics["section_label_mapped_to_resolution"] is False
    assert assessment.diagnostics["chorus_is_resolution"] is False
    assert "resolution" not in assessment.label
    assert "independent_phrase_completion" in assessment.missing


def test_word_active_at_cue_blocks_completion_context():
    profiles = _base_profiles()
    assessment = assess_functional_event(
        *profiles,
        vocal_completion={
            "completion": True,
            "phrase_completion": True,
            "provider": "unit-test",
            "word_active_at_cue": True,
        },
    )
    voice = assessment.diagnostics["vocal_observation"]
    assert voice["word_active_at_cue"] is True
    assert voice["word_active_blocks_completion"] is True
    assert voice["completion"] != "supported"
    assert "independent_phrase_completion" in assessment.missing


def test_missing_profile_abstains_without_score_or_fake_beats():
    a_before, a_after, _, b_after = _base_profiles()
    assessment = assess_functional_event(a_before, a_after, None, b_after)
    assert assessment.label == "unknown"
    assert assessment.abstained is True
    assert assessment.score is None
    assert assessment.confidence is None
    assert any("B_before" in item for item in assessment.missing)


def test_rerank_keeps_baseline_and_root_contract():
    profiles = _base_profiles()
    scorer = FunctionalEventScorer(weight=0.2)
    result, assessment = scorer.rerank(0.71, profiles=dict(zip(
        ("A_before", "A_after", "B_before", "B_after"), profiles
    )))
    assert result.baseline_score == 0.71
    assert result.adjusted_score is not None
    assert result.adjusted_score > result.baseline_score
    assert assessment.label == "post_entry_attack_stabilized"

    candidate = Candidate(
        "c1", "a", "b", 0, 0, "test",
        {"baseline_score": Feature(0.71, None, "baseline")}, {},
    )
    report = scorer.score(candidate, dict(zip(
        ("A_before", "A_after", "B_before", "B_after"), profiles
    )))
    assert report.candidate_id == "c1"
    assert report.score is not None and report.score > 0.71
    assert "transfer=unvalidated" in report.notes


def test_abstained_rerank_does_not_change_baseline():
    a_before, a_after, _, b_after = _base_profiles()
    assessment = assess_functional_event(a_before, a_after, None, b_after)
    result = rerank_functional_event(assessment, 0.5, weight=0.8)
    assert result.abstained is True
    assert result.adjusted_score == 0.5
    assert result.enabled is False


def test_threshold_validation():
    with pytest.raises(ValueError):
        FunctionalEventThresholds(min_domains=1)
    with pytest.raises(ValueError):
        FunctionalEventThresholds(event_threshold=-0.1)


def test_unknown_pickup_text_does_not_become_supported_pickup():
    from musicmix.functional_event import _voice
    report=_voice({'status':'unknown','reason':'pickup_alignment_unknown','provider':'test'})
    assert report.pickup=='unknown'

def test_rerank_pool_contract_preserves_rows_and_baseline():
    profiles = _base_profiles()
    lookup = {
        "c1": dict(zip(("A_before", "A_after", "B_before", "B_after"), profiles))
    }
    rows = [{"candidate_id": "c1", "baseline_score": 0.4, "label": "keep"}]
    output = FunctionalEventScorer(weight=0.2).rerank_pool(rows, lookup)
    assert rows[0]["baseline_score"] == 0.4
    assert output[0]["label"] == "keep"
    assert output[0]["event_baseline_score"] == 0.4
    assert output[0]["event_adjusted_score"] > 0.4
    assert output[0]["event_label"] == "post_entry_attack_stabilized"


def test_falling_directed_A_also_projects_arrival():
    # Direction sign is not a resolution judgement: a coherent falling A is
    # still a directed context. Only the joint route is checked here.
    falling = list(reversed(_rising((0.12, 0.10, 0.18, 0.10, 0.12),
                                    (0.035, 0.028, 0.020, 0.025, 0.018))))
    _, a_after, b_before, b_after = _base_profiles()
    a_before = _series(falling, side="outgoing")
    assessment = assess_functional_event(a_before, a_after, b_before, b_after)
    directed = assessment.evidence_by_kind["a_directed_expectation"]
    assert directed.status == "supported"
    assert set(directed.diagnostics["negative_domains"]) >= {"energy", "events"}
    assert assessment.label == "post_entry_attack_stabilized"


def test_pipeline_contract_is_json_serializable_with_routes():
    # EvidencePipeline.assess stores asdict(assessment) in JSON run files.
    import json
    from dataclasses import asdict

    from musicmix.evidence_pipeline import SUPPORTED_EVENTS
    from musicmix.functional_event import LABELS

    assert SUPPORTED_EVENTS <= LABELS
    assessment = assess_functional_event(
        *_base_profiles(),
        vocal_completion={
            "provider": "separate_word_and_measured_vocal_channels",
            "word_active_at_cue": True,
            "completion_supported": False,
            "pickup_supported": False,
        },
    )
    payload = json.loads(json.dumps(asdict(assessment), allow_nan=False))
    routes = payload["diagnostics"]["routes"]
    assert set(routes) >= {"compatible_continued_motion", "directed_arrival", "local_entry_only"}
    kinds = {item["kind"] for item in payload["evidence"]}
    assert {"continuation", "b_before_to_b_after_event", "a_directed_expectation",
            "directed_arrival", "local_entry_event"} <= kinds
    assert payload["diagnostics"]["vocal_observation"]["completion"] == "absent"
    assert payload["confidence"] is None
