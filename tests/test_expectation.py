from __future__ import annotations

import json
import copy
import math

import pytest

from musicmix.expectation import (
    EXPECTATION_VERSION,
    Evidence,
    ExpectationThresholds,
    analyze_expectation,
    rerank_expectation,
)


def _profile(values, *, stability=None, energy=None, structural=None, cue=0, anchor=0, period=100):
    """Small trajectory_to_dict-shaped profile for policy tests."""
    points = []
    for index, value in enumerate(values):
        points.append({
            "log_rms_mean": value if energy is None else energy[index],
            "rms_mean": value if energy is None else energy[index],
            "spectral_flux_mean": value,
            "attack_density_per_second": value,
            "bass_ratio": value,
            "high_frequency_ratio": value,
            "spectral_centroid_hz": value,
            "chroma_movement": value,
        })
    window = {"window_beats": 8, "beat_feature_series": points}
    if stability is not None:
        window["chroma_local_context_similarity"] = stability
    result = {
        "track_id": "synthetic",
        "sample_rate": 100,
        "duration_samples": 10_000,
        "cue_sample": cue,
        "beat_anchor_sample": anchor,
        "beat_period_samples": period,
        "confidence": None,
        "windows": [window],
    }
    if structural is not None:
        result["phrase_or_structural_arrival"] = structural
    return result


def test_energy_direction_does_not_change_under_constant_source_gain():
    reference = _profile([.1, .15, .2, .25])
    for point in reference['windows'][0]['beat_feature_series']:
        point['log_rms_mean'] = math.log(point['rms_mean'])
    quieter = copy.deepcopy(reference)
    for point in quieter['windows'][0]['beat_feature_series']:
        point['rms_mean'] *= .01
        point['log_rms_mean'] += math.log(.01)
    original = analyze_expectation(reference, reference, reference)
    changed = analyze_expectation(quieter, quieter, quieter)
    assert original.continuation_evidence.status == changed.continuation_evidence.status
    a = original.diagnostics['profiles']['A_before']['domains']['energy']
    b = changed.diagnostics['profiles']['A_before']['domains']['energy']
    assert a['score'] == pytest.approx(b['score'])
    assert a['direction'] == b['direction']


def _domain_profile(*, energy, events, bass, timbre, tonal, structural=None,
                    cue=0, anchor=0, period=100):
    points = []
    for index in range(len(energy)):
        points.append({
            "log_rms_mean": energy[index], "rms_mean": energy[index],
            "spectral_flux_mean": events[index],
            "attack_density_per_second": events[index],
            "bass_ratio": bass[index],
            "high_frequency_ratio": timbre[index],
            "spectral_centroid_hz": timbre[index],
            "chroma_movement": tonal[index],
        })
    result = {
        "track_id": "synthetic", "sample_rate": 100,
        "duration_samples": 10_000, "cue_sample": cue,
        "beat_anchor_sample": anchor, "beat_period_samples": period,
        "confidence": None,
        "windows": [{"window_beats": 8, "beat_feature_series": points}],
    }
    if structural is not None:
        result["phrase_or_structural_arrival"] = structural
    return result


def test_crescendo_to_increasing_is_continuation_without_reference_copy():
    assessment = analyze_expectation(
        _profile([1, 2, 3, 4]),
        _profile([1, 2, 3, 4]),
        _profile([1, 2, 3, 4]),
    )
    assert assessment.label == "continuation"
    assert assessment.confidence is None
    assert assessment.evidence_by_kind["continuation"].status == "supported"
    assert assessment.diagnostics["expectation_version"] == EXPECTATION_VERSION
    assert assessment.diagnostics["profiles"]["A_natural_after"]["provenance"]["available"]


def test_crescendo_to_flat_is_interruption_and_flat_flat_is_not_expectation():
    interrupted = analyze_expectation(_profile([1, 2, 3, 4]), None, _profile([2, 2, 2, 2]))
    assert interrupted.label == "unknown"
    assert interrupted.score is None
    flat = analyze_expectation(_profile([1, 1, 1, 1]), None, _profile([1, 1, 1, 1]))
    assert flat.label == "unknown"
    assert flat.evidence_by_kind["continuation"].status != "supported"


def test_arrival_requires_event_and_contextual_stability_not_rms_fall_alone():
    # Only energy falls; flux, density, bass and contextual stability remain
    # flat.  This must not be promoted to closure_candidate.
    volume_drop = _profile([1, 1, 1, 1], energy=[4, 1, 1, 1])
    result = analyze_expectation(_profile([1, 2, 3, 4]), None, volume_drop)
    assert result.label != "closure_candidate"
    assert result.evidence_by_kind["closure_arrival"].status == "unknown"
    assert "phrase_or_structural_arrival" in result.missing

    # A metric entry event plus independent post-entry stabilization is only a
    # closure candidate; tonal/cadence semantics remain explicitly missing.
    arrival = _profile(
        [4, 5, 2, 2], stability=0.9,
        structural={"supported": True, "provider": "synthetic_phrase_provider"},
    )
    result = analyze_expectation(_profile([1, 2, 3, 4]), None, arrival)
    closure = result.evidence_by_kind["closure_arrival"]
    assert result.label == "closure_candidate"
    assert closure.status == "supported"
    assert "tonal_cadence_semantics" in closure.missing

    no_before = analyze_expectation(
        {"windows": [], "cue_sample": 0, "beat_anchor_sample": 0,
         "beat_period_samples": 100},
        None,
        arrival,
    )
    assert no_before.label == "unknown"
    assert "A_before_multifeature_direction" in no_before.missing

    # Without an explicit contextual proxy, flatness of the remaining
    # correlated domains is only weak evidence and cannot prove arrival.
    weak_context = analyze_expectation(_profile([1, 2, 3, 4]), None,
                                       _profile([4, 2, 2, 2]))
    assert weak_context.label != "closure_candidate"
    assert weak_context.evidence_by_kind["closure_arrival"].status == "unknown"
    assert "independent_contextual_stability" in weak_context.missing


def test_continuation_requires_strong_matching_domains_and_conflicts_are_unknown():
    # Only energy matches strongly; the flat domains must not be counted as
    # continuation support merely because their signs multiply to zero.
    weak = _domain_profile(
        energy=[1, 2, 3, 4], events=[1, 1, 1, 1], bass=[1, 1, 1, 1],
        timbre=[1, 1, 1, 1], tonal=[1, 1, 1, 1],
    )
    result = analyze_expectation(_profile([1, 2, 3, 4]), None, weak)
    assert result.label == "unknown"
    assert result.evidence_by_kind["continuation"].status != "supported"

    # Two strong matching domains and two strong divergent domains create a
    # genuine policy conflict.  The conservative output is unknown.
    conflict = _domain_profile(
        energy=[1, 2, 3, 4], events=[1, 2, 3, 4], bass=[4, 3, 2, 1],
        timbre=[4, 3, 2, 1], tonal=[1, 2, 3, 4],
        structural={"supported": False, "provider": "synthetic_phrase_provider"},
    )
    result = analyze_expectation(_profile([1, 2, 3, 4]), None, conflict)
    assert result.label == "unknown"
    assert result.score is None
    assert result.diagnostics["evidence_conflict"] is True
    assert "conflicting_supported_evidence" in result.missing


def test_missing_context_and_null_confidence_propagate_unknown():
    result = analyze_expectation(None, None, _profile([1, 2, 3, 4]))
    assert result.label == "unknown"
    assert result.confidence is None
    assert "A_before" in result.missing
    no_series = {"windows": [{"window_beats": 8}], "cue_sample": 0,
                 "beat_anchor_sample": 0, "beat_period_samples": 100}
    result = analyze_expectation(no_series, None, no_series)
    assert result.label == "unknown"
    assert "beat_feature_series" in result.missing


def test_versioned_thresholds_and_optional_rerank_keep_baseline():
    thresholds = ExpectationThresholds(trend_threshold=0.2)
    assessment = analyze_expectation(_profile([1, 2, 3, 4]), None,
                                     _profile([1, 2, 3, 4]), thresholds=thresholds)
    raw = rerank_expectation(assessment, 0.73)
    assert raw.baseline_score == pytest.approx(0.73)
    assert raw.adjusted_score == pytest.approx(0.73)
    assert raw.enabled is False
    changed = rerank_expectation(assessment, 0.73, weight=0.1)
    assert changed.baseline_score == pytest.approx(0.73)
    assert changed.adjusted_score != changed.baseline_score


def test_evidence_rejects_nonfinite_scores_and_json_diagnostics_are_safe():
    with pytest.raises(ValueError):
        Evidence("continuation", "supported", float("nan"), {})
    assessment = analyze_expectation(_profile([1, 2, 3, 4]), None,
                                     _profile([1, 2, 3, 4]))
    json.dumps(assessment.diagnostics, allow_nan=False)
