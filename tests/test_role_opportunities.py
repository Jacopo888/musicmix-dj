"""Contract tests for role-aware opportunities.

All fixtures in this file are synthetic timestamps used to exercise the public
API.  They are not catalog audio and are not evidence about musical quality.
One test feeds a real ``musicmix.structure`` output document (built from a
synthetic zero signal) to check the integration contract, not measured beats.
"""
from __future__ import annotations

import numpy as np
import pytest

from musicmix.audio import AudioTrack
from musicmix.role_opportunities import (
    INCOMING,
    OUTGOING,
    PlanConfig,
    extract_track_opportunities,
    generate_role_plans,
    plan_catalog,
)
from musicmix.structure import ProviderPrediction, TimedEvent, analyze_structure


RATE = 48_000
DURATION = 60 * RATE


def _track(track_id="track_a", **overrides):
    base = {
        "track_id": track_id,
        "sample_rate": RATE,
        "duration_samples": DURATION,
        "structure": {
            "sample_rate": RATE,
            "duration_samples": DURATION,
            "beats": [{"sample": s} for s in range(RATE, DURATION, RATE // 2)],
            "downbeats": [{"sample": s} for s in range(2 * RATE, DURATION, 2 * RATE)],
        },
        "vocals": {"words": [], "asr_segments": [], "coverage_intervals": []},
    }
    base.update(overrides)
    return base


def test_provided_in_out_roles_are_distinct_and_neutral_is_both():
    track = _track(provided_opportunities=[
        {"sample": 10 * RATE, "role": "OUT", "provider": "CUE-DETR"},
        {"sample": 20 * RATE, "role": "IN", "provider": "CUE-DETR"},
        {"sample": 30 * RATE, "role": "boundary", "provider": "LinkSeg"},
    ])
    result = extract_track_opportunities(track)
    outgoing = [op for op in result.opportunities if op.role == OUTGOING]
    incoming = [op for op in result.opportunities if op.role == INCOMING]
    assert any(op.anchor_sample == 10 * RATE and op.source_role == OUTGOING for op in outgoing)
    assert not any(op.anchor_sample == 10 * RATE and op.source_role == INCOMING for op in incoming)
    assert any(op.anchor_sample == 20 * RATE and op.source_role == INCOMING for op in incoming)
    assert not any(op.anchor_sample == 20 * RATE and op.source_role == OUTGOING for op in outgoing)
    # The downbeat grid may independently expose a neutral context anchor.
    assert any(op.anchor_sample == 10 * RATE and op.source_role == "neutral" for op in incoming)
    assert any(op.anchor_sample == 30 * RATE and op.source_role == "neutral" for op in outgoing)
    assert any(op.anchor_sample == 30 * RATE and op.source_role == "neutral" for op in incoming)
    # A provided IN/OUT is a directional opportunity, not a claimed completion.
    explicit_out = next(op for op in outgoing if op.anchor_sample == 10 * RATE)
    assert explicit_out.completion_supported is False
    assert explicit_out.phrase_closure_supported is False


def test_incoming_pickup_and_arrival_are_distinct_with_explicit_delay():
    words = [{"start_sample": 18 * RATE, "end_sample": 18 * RATE + 20_000, "text": "pickup"}]
    track_b = _track("track_b", vocals={"words": words}, provided_opportunities=[
        {"sample": 19 * RATE, "role": "IN", "provider": "CUE-DETR"},
        {"sample": 40 * RATE, "role": "IN", "provider": "CUE-DETR"},
    ])
    result = extract_track_opportunities(track_b, config=PlanConfig(pickup_lookback_seconds=2.0))
    with_pickup = next(op for op in result.opportunities
                       if op.role == INCOMING and op.anchor_sample == 19 * RATE)
    without_pickup = next(op for op in result.opportunities
                          if op.role == INCOMING and op.anchor_sample == 40 * RATE)
    assert with_pickup.pickup_start_sample == 18 * RATE
    assert with_pickup.arrival_sample == 19 * RATE
    assert with_pickup.arrival_delay_samples == RATE
    assert without_pickup.pickup_start_sample == without_pickup.arrival_sample
    assert without_pickup.arrival_delay_samples == 0


def test_anchors_keep_provider_samples_and_never_snap():
    near = 10 * RATE + 37
    far = 16 * RATE + 5_000  # 104.2 ms from the nearest downbeat
    track = _track(provided_opportunities=[
        {"sample": near, "role": "OUT", "provider": "CUE-DETR"},
        {"sample": far, "role": "OUT", "provider": "CUE-DETR"},
    ])
    result = extract_track_opportunities(track, config=PlanConfig(max_grid_snap_seconds=0.080))
    near_op = next(op for op in result.opportunities if op.anchor_sample == near)
    far_op = next(op for op in result.opportunities if op.anchor_sample == far)
    assert near_op.anchor_sample == near
    assert near_op.grid_delta_samples == 37
    assert near_op.grid_alignment.startswith("within_80ms")
    assert far_op.anchor_sample == far
    assert abs(far_op.grid_delta_samples) == 5_000
    assert far_op.grid_alignment.startswith("off_grid")


def test_missing_grid_does_not_invent_beats():
    track = _track(structure=None, provided_opportunities=[
        {"sample": 10 * RATE, "role": "OUT", "provider": "LinkSeg"},
    ])
    result = extract_track_opportunities(track)
    assert result.beats == ()
    assert result.downbeats == ()
    assert result.diagnostics["no_fake_beats"] is True
    assert all("grid" not in op.source or op.source == "LinkSeg" for op in result.opportunities)
    assert not any(op.source.endswith("_grid") for op in result.opportunities)


def test_word_overlap_is_unsafe_and_missing_words_are_unknown():
    words = [{"start_sample": 9 * RATE + 40_000, "end_sample": 10 * RATE + 40_000, "text": "held"}]
    track = _track(vocals={"words": words}, provided_opportunities=[
        {"sample": 10 * RATE, "role": "OUT", "provider": "CUE-DETR"},
        {"sample": 30 * RATE, "role": "OUT", "provider": "CUE-DETR"},
    ])
    result = extract_track_opportunities(track)
    inside = next(op for op in result.opportunities
                  if op.anchor_sample == 10 * RATE and op.role == OUTGOING)
    outside = next(op for op in result.opportunities
                   if op.anchor_sample == 30 * RATE and op.role == OUTGOING)
    assert inside.word_status == "unsafe"
    assert inside.passes_constraints is False
    assert outside.word_status == "supported"

    no_vocals = _track("no_vocals", vocals={"words": []}, provided_opportunities=[
        {"sample": 10 * RATE, "role": "OUT", "provider": "CUE-DETR"},
    ])
    missing = extract_track_opportunities(no_vocals)
    unknown = next(op for op in missing.opportunities
                   if op.anchor_sample == 10 * RATE and op.role == OUTGOING)
    assert unknown.word_status == "unknown"


def test_incoming_can_start_at_word_onset_but_not_inside_word():
    words = [{"start_sample": 28 * RATE, "end_sample": 29 * RATE, "text": "pickup"}]
    track = _track("track_b", vocals={"words": words}, provided_opportunities=[
        {"sample": 28 * RATE, "role": "IN", "provider": "CUE-DETR"},
        {"sample": 28 * RATE + 24_000, "role": "IN", "provider": "CUE-DETR"},
    ])
    result = extract_track_opportunities(track)
    at_start = next(op for op in result.opportunities
                    if op.role == INCOMING and op.anchor_sample == 28 * RATE)
    inside = next(op for op in result.opportunities
                  if op.role == INCOMING and op.anchor_sample == 28 * RATE + 24_000)
    assert at_start.word_status == "supported"
    assert at_start.word_evidence["pickup_at_word_start"] is True
    # An explicit arrival inside a word is converted to a word-onset pickup,
    # while the arrival status remains separately visible.
    assert inside.pickup_start_sample == 28 * RATE
    assert inside.word_status == "supported"
    assert inside.word_evidence["arrival_word_status"] == "unsafe"


def test_unknown_coverage_seconds_are_not_treated_as_silence():
    words = [{"start_sample": 8 * RATE, "end_sample": 8 * RATE + 20_000, "text": "before"}]
    track = _track(vocals={
        "words": words,
        "coverage_intervals": [{
            "start_seconds": 9.5,
            "end_seconds": 10.5,
            "status": "unknown_no_asr",
        }],
    }, provided_opportunities=[
        {"sample": 10 * RATE, "role": "OUT", "provider": "CUE-DETR"},
    ])
    result = extract_track_opportunities(track)
    outgoing = next(op for op in result.opportunities
                    if op.anchor_sample == 10 * RATE and op.role == OUTGOING)
    assert outgoing.word_status == "unknown"
    assert outgoing.word_evidence["reason"] == "canonical_coverage_unknown_at_anchor"


def test_distant_arrival_is_rejected_as_not_a_single_transition():
    track_a = _track("track_a", structure=None, provided_opportunities=[
        {"sample": 10 * RATE, "role": "OUT", "provider": "CUE-DETR"},
    ])
    track_b = _track("track_b", structure=None, provided_opportunities=[
        {"sample": 50 * RATE, "role": "IN", "provider": "CUE-DETR"},
    ])
    result = plan_catalog({"tracks": [track_a, track_b]},
                          config=PlanConfig(max_plans=2, max_per_track_role=1))
    assert result["selected_plans"] == []
    proposal = next(item for item in result["proposals"] if item["track_b"] == "track_b")
    assert proposal["passes_constraints"] is False
    assert "arrival_after_change_exceeds_limit" in proposal["reasons"]


def test_plan_has_wait_prepare_change_and_explicit_entry_window():
    track_a = _track("track_a", provided_opportunities=[
        {"sample": 20 * RATE, "role": "OUT", "provider": "CUE-DETR", "completion": True},
    ])
    words = [{"start_sample": 25 * RATE, "end_sample": 25 * RATE + 20_000, "text": "pickup"}]
    track_b = _track("track_b", vocals={"words": words}, provided_opportunities=[
        {"sample": 26 * RATE, "role": "IN", "provider": "CUE-DETR"},
    ])
    catalog = {"tracks": [track_a, track_b]}
    result = plan_catalog(catalog, config=PlanConfig(max_plans=4, max_per_track_role=2))
    assert result["selected_plans"]
    plan = result["selected_plans"][0]
    actions = [event["action"] for event in plan["events"]]
    assert actions == ["wait", "prepare", "change"]
    assert plan["change_sample"] == 20 * RATE
    assert plan["prepare_sample"] <= plan["change_sample"]
    assert plan["entry_window"]["start_sample"] == 25 * RATE
    assert plan["entry_window"]["end_sample"] == 26 * RATE
    assert plan["entry_window"]["arrival_delay_samples"] == RATE
    assert plan["incoming_role"] == "pickup"
    assert plan["outgoing_completion_supported"] is True
    assert plan["words_protected"] is True


def test_seconds_are_converted_once_and_provider_seconds_are_retained():
    track = _track(provided_opportunities=[
        {"time_seconds": 12.345, "role": "OUT", "provider": "CUE-DETR"},
    ])
    result = extract_track_opportunities(track)
    op = next(op for op in result.opportunities if op.anchor_sample == round(12.345 * RATE))
    assert op.anchor_sample == round(12.345 * RATE)
    assert op.provider_time_seconds == pytest.approx(12.345)


def test_asr_decoder_segment_endpoints_are_not_phrase_opportunities():
    track = _track(
        vocals={
            "words": [],
            "asr_segments": [{"start_sample": 10 * RATE, "end_sample": 12 * RATE, "text": "decoder"}],
        },
        provided_opportunities=[],
        structure={"sample_rate": RATE, "duration_samples": DURATION, "beats": [], "downbeats": []},
    )
    result = extract_track_opportunities(track)
    assert result.opportunities == ()
    assert result.diagnostics["asr_segment_endpoints_used_as_phrase"] is False


def test_named_vocal_completion_is_separate_from_phrase_closure():
    track = _track(provided_opportunities=[], vocal_completions=[
        {"sample": 20 * RATE, "role": "OUT", "provider": "official-vocal-completion", "supported": True},
        {"sample": 30 * RATE, "role": "OUT", "source": "anonymous", "supported": False},
    ])
    result = extract_track_opportunities(track)
    completion = next(
        op for op in result.opportunities
        if op.anchor_sample == 20 * RATE and op.role == OUTGOING
        and op.source == "official-vocal-completion"
    )
    assert completion.completion_supported is True
    assert completion.phrase_closure_supported is False
    assert not any(op.anchor_sample == 30 * RATE for op in result.opportunities if op.source == "anonymous")


def _six_track_catalog():
    tracks = []
    for index in range(6):
        track_id = f"catalog_{index}"
        downbeats = [s for s in range(8 * RATE + index * RATE // 2, 56 * RATE, 4 * RATE)]
        words = [
            {"start_sample": s + RATE, "end_sample": s + RATE + 12_000, "text": f"w{index}"}
            for s in downbeats[:-1]
        ]
        tracks.append({
            "track_id": track_id,
            "sample_rate": RATE,
            "duration_samples": DURATION,
            "structure": {
                "sample_rate": RATE,
                "duration_samples": DURATION,
                "beats": [{"sample": s} for s in range(RATE, DURATION, RATE // 2)],
                "downbeats": [{"sample": s} for s in downbeats],
            },
            "vocals": {"words": words, "asr_segments": [], "coverage_intervals": []},
            "provided_opportunities": [
                {"sample": downbeats[0], "role": "OUT", "provider": "CUE-DETR"},
                {"sample": downbeats[-1], "role": "IN", "provider": "CUE-DETR"},
            ],
        })
    return {"tracks": tracks}


def test_six_track_catalog_api_and_diversity_without_quality_score():
    result = generate_role_plans(
        _six_track_catalog(),
        config=PlanConfig(max_plans=6, max_per_track_role=1, min_region_spacing_seconds=8.0),
    )
    assert result.diagnostics["track_count"] == 6
    assert result.diagnostics["track_ids"] == [f"catalog_{i}" for i in range(6)]
    assert 0 < len(result.selected_plans) <= 6
    assert result.diagnostics["selection"]["selection_basis"] == "evidence_completeness_not_musical_quality"
    assert result.diagnostics["selection"]["no_quality_score_invented"] is True
    assert all(value <= 1 for value in result.diagnostics["selection"]["selected_outgoing_counts"].values())
    assert all(value <= 1 for value in result.diagnostics["selection"]["selected_incoming_counts"].values())
    assert all(plan.passes_constraints for plan in result.selected_plans)


def test_structure_api_document_is_consumed_without_shifting():
    """Synthetic zero audio through the real structure API, not measured beats."""
    audio = AudioTrack(
        "struct_track", "synthetic.wav", RATE,
        np.zeros((20 * RATE, 2), dtype=np.float32),
    )

    class Provider:
        def predict(self, samples, sample_rate):
            del samples, sample_rate
            return ProviderPrediction(
                beats=tuple(TimedEvent(t) for t in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)),
                downbeats=(TimedEvent(2.0), TimedEvent(4.0)),
                provenance={"provider": "synthetic_contract_fixture"},
            )

    analysis = analyze_structure(audio, Provider(), include_heuristic_sections=False)
    document = analysis.to_dict()
    track = {
        "track_id": "struct_track",
        "sample_rate": RATE,
        "duration_samples": analysis.duration_samples,
        "structure": document,
        "vocals": {"words": []},
        "provided_opportunities": [
            {"sample": 2 * RATE + 17, "role": "OUT", "provider": "CUE-DETR"},
        ],
    }
    result = extract_track_opportunities(track)
    assert result.downbeats == (2 * RATE, 4 * RATE)
    op = next(op for op in result.opportunities if op.anchor_sample == 2 * RATE + 17)
    assert op.grid_delta_samples == 17
    assert op.anchor_sample == 2 * RATE + 17
