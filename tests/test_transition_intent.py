from __future__ import annotations

import json
import math

from musicmix.transition_intent import assess_transition


RATE = 48_000


def _profile(values, *, track_id="track", structural=None, level=None, stability=None, gain=1.0):
    # ``values`` controls direction only.  Features use plausible domains so
    # this fixture does not accidentally rely on ratios outside [0, 1].
    low, high = min(values), max(values)
    span = high - low
    progress = [0.0 if span == 0 else (value - low) / span for value in values]
    points = []
    for fraction in progress:
        log_rms = (-3.2 + 0.8 * fraction) + math.log(gain)
        rms = min(0.99, (0.055 + 0.035 * fraction) * gain)
        points.append({
            "log_rms_mean": log_rms,
            "rms_mean": rms,
            "spectral_flux_mean": 0.018 + 0.018 * fraction,
            "attack_density_per_second": 2.0 + 1.5 * fraction,
            "bass_ratio": 0.22 + 0.18 * fraction,
            "high_frequency_ratio": 0.34 + 0.18 * fraction,
            "spectral_centroid_hz": 1_100.0 + 600.0 * fraction,
            "chroma_movement": 0.06 + 0.09 * fraction,
        })
    window = {"window_beats": 8, "beat_feature_series": points}
    if stability is not None:
        window["chroma_local_context_similarity"] = stability
    result = {
        "track_id": track_id,
        "sample_rate": RATE,
        "duration_samples": RATE * 60,
        "cue_sample": 0,
        "beat_anchor_sample": 0,
        "beat_period_samples": RATE // 2,
        "windows": [window],
    }
    if structural is not None:
        result["phrase_or_structural_arrival"] = structural
    if level is not None:
        result["energy_level"] = level
    return result


def _clear_phrase(status="supported", boundary_class="observed_asr_segment_end"):
    return {
        "status": status,
        "boundary_class": boundary_class,
        "active_utterance": status == "unsafe",
        "pickup_uncertainty": False,
        "reasons": [],
    }


def _segment_only_word_safe_phrase():
    return {
        "status": "supported",
        "timing_status": "supported",
        "word_clearance_status": "supported",
        "word_clearance": {"status": "supported", "supported": True, "unsafe": False},
        "boundary_class": "asr_segment_occupancy_unknown",
        "active_utterance": False,
        "active_asr_segment": True,
        "phrase_continuity_status": "unknown",
        "phrase_closure_status": "unknown",
        "pickup_uncertainty": False,
        "reasons": ["phrase_continuity_unknown_from_asr_segment_occupancy"],
    }


def test_continuation_is_a_motion_intent_and_requires_an_explicit_boundary_for_execution():
    result = assess_transition(
        _profile([1, 2, 3, 4]),
        _profile([1, 2, 3, 4]),
        _profile([1, 2, 3, 4]),
        _profile([2, 3, 4, 5]),
        phrase_a=_clear_phrase(), phrase_b=_clear_phrase(), baseline_score=0.73,
    )

    assert result.intent == "continue_motion"
    assert result.action == "execute"
    assert "multifeature_continuation_support" in result.diagnostics["supported_switch_reasons"]
    assert result.confidence is None
    assert result.utility["baseline_score"] == 0.73
    json.dumps(result.to_dict(), allow_nan=False)


def test_continuation_can_execute_on_word_safe_timing_with_unknown_phrase_closure():
    result = assess_transition(
        _profile([1, 2, 3, 4]),
        _profile([1, 2, 3, 4]),
        _profile([1, 2, 3, 4]),
        _profile([2, 3, 4, 5]),
        phrase_a=_segment_only_word_safe_phrase(),
        phrase_b=_segment_only_word_safe_phrase(),
    )

    assert result.action == "execute"
    assert result.intent == "continue_motion"
    assert "A_phrase_closure_unknown" in result.missing
    assert result.diagnostics["phrase_continuity"] == {
        "A": "unknown", "B": "unknown", "closure_is_required_for_arrive": True,
        "asr_segment_occupancy_is_not_closure": True,
    }


def test_flat_low_groove_stays_and_does_not_invent_a_switch_reason():
    result = assess_transition(
        _profile([1, 1, 1, 1], level="low"),
        None,
        None,
        _profile([1, 1, 1, 1], level="low"),
    )

    assert result.action == "stay"
    assert result.intent == "preserve_groove"
    assert "unmotivated_cut" not in result.reasons
    assert any("flat_low_groove" in reason for reason in result.reasons)


def test_section_label_or_quiet_intro_cannot_fake_arrival():
    result = assess_transition(
        _profile([1, 2, 3, 4]),
        None,
        None,
        _profile([1, 1, 1, 1]),
        structure_b={"section_label": "intro", "quiet_intro": True},
        phrase_a=_clear_phrase(), phrase_b=_clear_phrase(),
    )

    assert result.intent != "arrive"
    assert result.action in {"stay", "defer"}
    assert "arrival_event_and_stability" in result.missing


def test_missing_context_abstains_and_exposes_coverage():
    result = assess_transition(None, None, None, None)

    assert result.action == "defer"
    assert result.abstained is True
    assert result.coverage["fraction"] < 1.0
    assert result.coverage["evidence_slot_count"] == len(result.coverage["available_fields"])
    assert result.coverage["required_count"] == 2
    assert result.coverage["candidate_rating"] is None
    assert result.coverage["claim_10_of_10"] is False
    assert "missing_global_motion_context" in result.reasons


def test_track_identity_is_not_a_natural_reference_reward():
    a = _profile([1, 2, 3, 4], track_id="same")
    b = _profile([2, 3, 4, 5], track_id="same")
    same = assess_transition(a, a, None, b, phrase_a=_clear_phrase(), phrase_b=_clear_phrase())
    different = assess_transition(
        _profile([1, 2, 3, 4], track_id="other-a"),
        _profile([1, 2, 3, 4], track_id="other-natural"),
        None,
        _profile([2, 3, 4, 5], track_id="other-b"),
        phrase_a=_clear_phrase(), phrase_b=_clear_phrase(),
    )

    assert same.intent == different.intent == "continue_motion"
    assert same.action == different.action == "execute"
    assert same.diagnostics["natural_reference_identity_used"] is False


def test_explicit_functional_levels_survive_gain_and_unit_changes():
    base = dict(level="high")
    reference = assess_transition(
        _profile([1, 2, 3, 4], **base),
        _profile([1, 2, 3, 4], **base),
        _profile([1, 2, 3, 4], **base),
        _profile([2, 3, 4, 5], **base),
        phrase_a=_clear_phrase(), phrase_b=_clear_phrase(),
    )
    gained = assess_transition(
        _profile([1, 2, 3, 4], **base, gain=6.0),
        _profile([1, 2, 3, 4], **base, gain=6.0),
        _profile([1, 2, 3, 4], **base, gain=6.0),
        _profile([2, 3, 4, 5], **base, gain=6.0),
        phrase_a=_clear_phrase(), phrase_b=_clear_phrase(),
    )

    assert reference.action == gained.action == "execute"
    assert reference.intent == gained.intent == "continue_motion"
    assert reference.diagnostics["levels"] == gained.diagnostics["levels"] == {
        "A_before": "high", "B_after": "high"
    }


def test_numeric_trajectory_without_provider_level_stays_unknown():
    result = assess_transition(
        _profile([1, 1, 1, 1]), None, _profile([1, 1, 1, 1]),
        _profile([1, 1, 1, 1]), phrase_a=_clear_phrase(), phrase_b=_clear_phrase(),
    )

    assert result.diagnostics["levels"] == {"A_before": None, "B_after": None}
