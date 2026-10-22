from __future__ import annotations

import math
import copy

import numpy as np
import pytest

from musicmix.audio import AudioTrack
from musicmix.directional import DirectionalSearchResult, _audio_descriptor, search_directional as _search_directional


RATE = 1_000
BEATS = tuple(index * 100 for index in range(20))


def search_directional(*args, **kwargs):
    # Unit fixtures are intentionally short; production keeps the default
    # ten-second listening-context guard.
    kwargs.setdefault("min_context_seconds", 0.0)
    return _search_directional(*args, **kwargs)


def _track(name: str, reverse: bool = False) -> AudioTrack:
    duration = 2_000
    time = np.arange(duration, dtype=float) / RATE
    # Different envelopes give the directional descriptors real context to
    # compare while remaining deterministic and cheap for unit tests.
    envelope = np.linspace(0.25, 1.0, duration) if not reverse else np.linspace(1.0, 0.25, duration)
    signal = envelope * (0.5 * np.sin(2 * math.pi * 110 * time) + 0.2 * np.sin(2 * math.pi * 220 * time))
    return AudioTrack(name, f"{name}.wav", RATE, np.column_stack((signal, signal)))


def _structure(track: AudioTrack, cues: tuple[int, ...] | None = None) -> dict:
    selected = cues if cues is not None else BEATS
    return {
        "track_id": track.track_id,
        "sample_rate": RATE,
        "duration_samples": track.duration_samples,
        "beats": [{"sample": sample, "confidence": 0.95} for sample in BEATS],
        "downbeats": [{"sample": sample, "confidence": 0.9} for sample in selected],
    }


def _vocal(words: list[tuple[int, int, float | None]], coverage: list[tuple[int, int]] | None = None) -> dict:
    return {
        "words": [
            {"start_sample": start, "end_sample": end, "text": "x", "confidence": confidence}
            for start, end, confidence in words
        ],
        "coverage": [{"start_sample": start, "end_sample": end} for start, end in (coverage or [(0, 2_000)])],
    }


def test_directional_result_has_selected_and_full_ranked_audit() -> None:
    a, b = _track("a"), _track("b", reverse=True)
    result = search_directional(a, b, _structure(a), _structure(b), top_k=3, max_candidates=40, allow_unknown_vocals=True)

    assert isinstance(result, DirectionalSearchResult)
    assert 0 < len(result.items) <= 3
    assert len(result.proposals) >= len(result.items)
    assert result.selected == tuple(item.candidate.candidate_id for item in result.items)
    assert result.diagnostics["candidate_count_evaluated"] <= 40
    assert result.diagnostics["direction"] == "track_a_outgoing_to_track_b_incoming"
    assert all(item.candidate.track_a == "a" and item.candidate.track_b == "b" for item in result.items)
    assert all(item.raw_features["direction"] == "outgoing_then_incoming" for item in result.items)


def test_provided_opportunities_keep_outgoing_and_incoming_roles():
    a,b=_track('a'),_track('b')
    sa,sb=_structure(a),_structure(b)
    sa['provided_cues']=[{'sample':800,'role':'incoming'},{'sample':1200,'role':'outgoing'}]
    sb['provided_cues']=[{'sample':800,'role':'outgoing'},{'sample':1200,'role':'incoming'}]
    before=copy.deepcopy((sa,sb))
    result=search_directional(a,b,sa,sb,cue_policy='provided',allow_unknown_vocals=True,max_candidates=100)
    assert result.proposals
    assert all(x.candidate.cue_out_sample==1200 and x.candidate.cue_in_sample==1200 for x in result.proposals)
    assert (sa,sb)==before


def test_invalid_provided_role_is_rejected_instead_of_silently_shared():
    a,b=_track('a'),_track('b');sa,sb=_structure(a),_structure(b)
    sa['provided_cues']=[{'sample':1000,'role':'ambiguous_typo'}]
    sb['provided_cues']=[{'sample':1000}]
    with pytest.raises(ValueError,match='cue role'):
        search_directional(a,b,sa,sb,cue_policy='provided',allow_unknown_vocals=True)


def test_order_is_directional_and_deterministic() -> None:
    a, b = _track("a"), _track("b", reverse=True)
    forward = search_directional(a, b, _structure(a), _structure(b), top_k=4, max_candidates=100, allow_unknown_vocals=True)
    repeat = search_directional(a, b, _structure(a), _structure(b), top_k=4, max_candidates=100, allow_unknown_vocals=True)
    reverse = search_directional(b, a, _structure(b), _structure(a), top_k=4, max_candidates=100, allow_unknown_vocals=True)

    assert [item.candidate.candidate_id for item in forward.items] == [item.candidate.candidate_id for item in repeat.items]
    assert [item.score for item in forward.items] == [item.score for item in repeat.items]
    assert forward.items[0].candidate.track_a == "a"
    assert reverse.items[0].candidate.track_a == "b"
    assert [item.candidate.candidate_id for item in forward.items] != [item.candidate.candidate_id for item in reverse.items]
    # A tail window and B head window are distinct semantic units, so the
    # raw directed trend values remain separately inspectable.
    assert "a_rms_trend" in forward.items[0].raw_features
    assert "b_rms_trend" in forward.items[0].raw_features


def test_downbeats_are_preferred_and_short_grids_are_reported() -> None:
    a, b = _track("a"), _track("b")
    structure_a = _structure(a, cues=(800, 900))
    structure_b = _structure(b, cues=(900, 1_000))
    result = search_directional(a, b, structure_a, structure_b, top_k=2, max_candidates=1)

    assert result.diagnostics["cue_source_a"] == "downbeats"
    assert result.diagnostics["cue_source_b"] == "downbeats"
    assert result.diagnostics["candidate_grid_truncated"] is True
    assert result.diagnostics["candidate_count_evaluated"] == 1


def test_known_low_confidence_downbeat_is_filtered_and_reported() -> None:
    a, b = _track("a"), _track("b")
    structure_a = _structure(a, cues=(800, 900))
    structure_b = _structure(b, cues=(900, 1_000))
    structure_b["downbeats"][0]["confidence"] = 0.037
    result = search_directional(a, b, structure_a, structure_b, top_k=2, allow_unknown_vocals=True)

    assert result.diagnostics["low_confidence_downbeats_b"] == 1
    low = [record for record in result.rejected if record["reason"] == "low_downbeat_confidence"]
    assert low and low[0]["confidence"] == 0.037
    assert all(item.candidate.cue_in_sample != 900 for item in result.items)


def test_default_ten_second_context_filter_is_explicit() -> None:
    a, b = _track("a"), _track("b")
    result = _search_directional(a, b, _structure(a), _structure(b), top_k=2, allow_unknown_vocals=True)

    assert not result.items
    assert result.diagnostics["min_context_seconds"] == 10.0
    assert result.diagnostics["listening_context_filter_rejections"] > 0
    assert any(record["reason"] == "outside_listening_context" for record in result.rejected)


def test_known_word_cut_is_rejected_without_fabricated_gap() -> None:
    a, b = _track("a"), _track("b")
    # The only A cue is inside a word.  One word cannot establish a safe
    # internal gap, even though broad coverage is present.
    vocals_a = _vocal([(760, 860, 0.95)])
    result = search_directional(a, b, _structure(a, (800,)), _structure(b, (900,)), vocals_a, None, top_k=5)

    assert not result.items
    assert result.rejected
    assert all(record["reason"] == "vocal_boundary_rejected" for record in result.rejected)
    assert any("no_confident_gap_inside_coverage" in record["reasons"] for record in result.rejected)


def test_safe_internal_gap_can_shift_within_phase_limit() -> None:
    a, b = _track("a"), _track("b")
    vocals_a = _vocal([(100, 760, 0.95), (1_000, 1_100, 0.95)])
    vocals_b = _vocal([(100, 780, 0.95), (1_500, 1_600, 0.95)])
    result = search_directional(a, b, _structure(a, (800,)), _structure(b, (900,)), vocals_a, vocals_b, top_k=2)

    assert result.items
    item = result.items[0]
    assert item.alignment["outgoing"]["status"] == "safe_confirmed_gap"
    assert item.alignment["incoming"]["status"] == "safe_confirmed_gap"
    assert item.candidate.cue_out_sample != 800
    assert item.candidate.cue_in_sample == 900
    assert abs(item.candidate.cue_out_sample - 800) <= 50
    assert item.alignment["fade_interval_a"][1] == item.candidate.cue_out_sample
    assert item.alignment["fade_interval_b"][0] == item.candidate.cue_in_sample


def test_missing_confidence_and_noncontiguous_coverage_are_not_silence() -> None:
    a, b = _track("a"), _track("b")
    missing_conf = _vocal([(100, 400, None), (900, 1_200, 0.95)])
    result_missing = search_directional(a, b, _structure(a, (800,)), _structure(b, (900,)), missing_conf, None)
    assert not result_missing.items
    assert any("no_confident_gap_inside_coverage" in record["reasons"] for record in result_missing.rejected)

    disjoint = _vocal([(100, 300, 0.95), (900, 1_200, 0.95)], [(0, 500), (700, 1_500)])
    result_disjoint = search_directional(a, b, _structure(a, (800,)), _structure(b, (900,)), disjoint, None)
    assert not result_disjoint.items
    assert any("no_confident_gap_inside_coverage" in record["reasons"] for record in result_disjoint.rejected)


def test_explicit_acoustic_silence_can_support_instrumental_boundary() -> None:
    a, b = _track("a"), _track("b")
    acoustic = {
        "coverage_intervals": [
            {"start_sample": 0, "end_sample": 2_000, "status": "acoustic_silence_supported", "evidence": "stem"}
        ],
        "words": [],
    }
    result = search_directional(a, b, _structure(a, (800,)), _structure(b, (900,)), acoustic, acoustic, top_k=1)

    assert result.items
    item = result.items[0]
    assert item.alignment["outgoing"]["safe"] is True
    assert item.alignment["incoming"]["safe"] is True
    assert item.alignment["outgoing"]["status"] == "safe_acoustic_silence"
    assert item.alignment["boundary_class"] == {
        "outgoing": "acoustic_boundary_supported",
        "incoming": "acoustic_boundary_supported",
    }

    occupancy_only = {
        "acoustic_occupancy": [
            {"start_sample": 0, "end_sample": 2_000, "status": "acoustic_silence", "evidence": "stem"}
        ],
        "words": [],
    }
    occupancy_result = search_directional(
        a, b, _structure(a, (800,)), _structure(b, (900,)), occupancy_only, occupancy_only, top_k=1
    )
    assert occupancy_result.items


def test_asr_word_gap_uses_segment_timing_not_occupancy_rows() -> None:
    a, b = _track("a"), _track("b")
    vocals = {
        "asr_segments": [{"start_sample": 100, "end_sample": 1_600, "confidence": 0.9}],
        "words": [
            {"start_sample": 100, "end_sample": 600, "text": "a", "confidence": 0.95},
            {"start_sample": 800, "end_sample": 900, "text": "b", "confidence": 0.95},
            {"start_sample": 1_300, "end_sample": 1_400, "text": "c", "confidence": 0.95},
        ],
        # Production evidence fragments vocal occupancy into rows.  Those
        # rows must not be interpreted as an acoustic silence interval.
        "coverage_intervals": [
            {"start_sample": 0, "end_sample": 500, "status": "vocal_occupancy_supported"},
            {"start_sample": 500, "end_sample": 1_000, "status": "vocal_occupancy_supported"},
            {"start_sample": 1_000, "end_sample": 1_600, "status": "vocal_occupancy_supported"},
        ],
    }
    result = search_directional(
        a,
        b,
        _structure(a, (700,)),
        _structure(b, (700,)),
        vocals,
        vocals,
        top_k=1,
    )

    assert result.items
    for side in ("outgoing", "incoming"):
        alignment = result.items[0].alignment[side]
        assert alignment["status"] == "safe_confirmed_gap"
        assert alignment["evidence_type"] == "asr_word_gap"
        assert alignment["evidence"] == "ASR word timing; no acoustic silence claim"
        assert alignment["status"] != "safe_acoustic_silence"


def test_occupancy_rows_without_asr_do_not_promote_a_word_gap() -> None:
    a, b = _track("a"), _track("b")
    vocals = {
        "words": [
            {"start_sample": 100, "end_sample": 600, "text": "a", "confidence": 0.95},
            {"start_sample": 800, "end_sample": 900, "text": "b", "confidence": 0.95},
        ],
        "coverage_intervals": [
            {"start_sample": 0, "end_sample": 500, "status": "vocal_occupancy_supported"},
            {"start_sample": 500, "end_sample": 1_500, "status": "vocal_occupancy_supported"},
        ],
    }
    result = search_directional(
        a,
        b,
        _structure(a, (700,)),
        _structure(b, (700,)),
        vocals,
        vocals,
        top_k=1,
    )

    assert not result.items
    assert all(record["reason"] == "vocal_boundary_rejected" for record in result.rejected)
    assert any("no_confident_gap_inside_coverage" in record["reasons"] for record in result.rejected)


def test_beats_policy_records_provenance_and_changes_identity() -> None:
    a, b = _track("a"), _track("b")
    downbeats = search_directional(
        a,
        b,
        _structure(a),
        _structure(b),
        top_k=1,
        allow_unknown_vocals=True,
        min_cue_spacing_seconds=0.0,
    )
    beats = search_directional(
        a,
        b,
        _structure(a),
        _structure(b),
        top_k=1,
        allow_unknown_vocals=True,
        min_cue_spacing_seconds=0.0,
        cue_policy="beats",
    )

    assert downbeats.items and beats.items
    assert downbeats.diagnostics["cue_policy"] == "downbeats"
    assert beats.diagnostics["cue_policy"] == "beats"
    assert beats.diagnostics["cue_source_a"] == "beats"
    assert beats.diagnostics["cue_source_b"] == "beats"
    assert beats.items[0].raw_features["cue_source_a"] == "beat"
    assert beats.items[0].raw_features["cue_source_b"] == "beat"
    assert beats.items[0].candidate.candidate_id != downbeats.items[0].candidate.candidate_id


def test_selection_is_diverse_on_both_timelines_and_reports_shortfall() -> None:
    a, b = _track("a"), _track("b", reverse=True)
    result = search_directional(
        a,
        b,
        _structure(a),
        _structure(b),
        top_k=5,
        max_candidates=100,
        allow_unknown_vocals=True,
        min_cue_spacing_seconds=0.3,
    )

    anchors = [item.raw_features["original_structure_cue_samples"] for item in result.items]
    assert len(anchors) == len({(item["a"], item["b"]) for item in anchors})
    assert len({item["a"] for item in anchors}) == len(anchors)
    assert len({item["b"] for item in anchors}) == len(anchors)
    assert all(
        abs(left["a"] - right["a"]) >= 300 and abs(left["b"] - right["b"]) >= 300
        for index, left in enumerate(anchors)
        for right in anchors[index + 1 :]
    )
    assert result.diagnostics["selected_count_lt_top_k"] is True
    assert result.diagnostics["selection_shortfall"] == 5 - len(result.items)
    assert result.diagnostics["cue_reuse_cap"] == 1
    assert result.diagnostics["cue_support_a"]
    assert result.diagnostics["cue_support_b"]
    assert result.diagnostics["position_zone_counts_pregrid"]["a"]["intro"] > 0
    assert result.diagnostics["rejection_reason_counts"]["top_k_cue_reuse"] > 0


def test_invalid_cue_policy_and_spacing_are_rejected() -> None:
    a, b = _track("a"), _track("b")
    with pytest.raises(ValueError):
        search_directional(a, b, _structure(a), _structure(b), cue_policy="sections")
    with pytest.raises(ValueError):
        search_directional(a, b, _structure(a), _structure(b), min_cue_spacing_seconds=-0.1)


def test_phrase_boundary_tier_precedes_internal_word_gap() -> None:
    a, b = _track("a"), _track("b")
    vocals = {
        "words": [
            {"start_sample": 100, "end_sample": 600, "text": "a", "confidence": 0.95},
            {"start_sample": 800, "end_sample": 850, "text": "b", "confidence": 0.95},
            {"start_sample": 1_300, "end_sample": 1_350, "text": "c", "confidence": 0.95},
        ],
        "phrases": [
            {"start_sample": 100, "end_sample": 850, "confidence": 0.95},
            {"start_sample": 1_300, "end_sample": 1_350, "confidence": 0.95},
        ],
        "coverage_intervals": [{"start_sample": 0, "end_sample": 2_000, "status": "asr_observed"}],
    }
    result = search_directional(
        a,
        b,
        _structure(a, (700, 1_000)),
        _structure(b, (700, 1_000)),
        vocals,
        vocals,
        top_k=1,
    )

    assert result.items
    selected = result.items[0]
    assert selected.alignment["boundary_class"] == {
        "outgoing": "phrase_boundary_supported",
        "incoming": "phrase_boundary_supported",
    }
    assert any(
        proposal.alignment["boundary_class"]["outgoing"] == "word_gap_only"
        for proposal in result.proposals
    )
    assert "word_gap_only_secondary_to_phrase_boundary" not in selected.reasons


def test_opposite_safe_shifts_are_rejected_when_relative_phase_exceeds_policy() -> None:
    a, b = _track("a"), _track("b")
    vocals_a = _vocal([(100, 790, 0.95), (1_000, 1_100, 0.95)])
    vocals_b = _vocal([(100, 700, 0.95), (910, 1_000, 0.95)])
    result = search_directional(
        a,
        b,
        _structure(a, (800,)),
        _structure(b, (900,)),
        vocals_a,
        vocals_b,
        top_k=1,
    )

    assert not result.items
    assert any(record["reason"] == "combined_vocal_shift_exceeds_phase" for record in result.rejected)


def test_vocal_cache_does_not_mutate_evidence_payload() -> None:
    a, b = _track("a"), _track("b")
    vocals = _vocal([(100, 760, 0.95), (1_000, 1_100, 0.95)])
    before = copy.deepcopy(vocals)
    search_directional(a, b, _structure(a, (800,)), _structure(b, (900,)), vocals, vocals, top_k=2)
    assert vocals == before


def test_unknown_vocal_input_is_explicit_and_penalized() -> None:
    a, b = _track("a"), _track("b")
    result = search_directional(a, b, _structure(a, (800,)), _structure(b, (900,)), None, None, top_k=1, allow_unknown_vocals=True)
    item = result.items[0]
    assert item.alignment["outgoing"]["status"] == "unknown_no_vocal_input"
    assert item.alignment["incoming"]["status"] == "unknown_no_vocal_input"
    assert "vocal_safety" in item.breakdown
    assert item.breakdown["vocal_safety"] < 1.0
    assert item.candidate.penalties.get("artifact") is None


def test_descriptor_uses_earlier_audio_not_only_tail_frame() -> None:
    duration = 4_096
    time = np.arange(duration, dtype=float) / RATE
    tail = 0.5 * np.sin(2 * math.pi * 110 * time)
    changed = tail.copy()
    changed[:2_000] = 0.5 * np.sin(2 * math.pi * 330 * time[:2_000])
    first = AudioTrack("first", "first.wav", RATE, np.column_stack((tail, tail)))
    second = AudioTrack("second", "second.wav", RATE, np.column_stack((changed, changed)))
    first_chroma = _audio_descriptor(first, 0, duration)[1]
    changed_chroma = _audio_descriptor(second, 0, duration)[1]

    assert first_chroma and changed_chroma
    assert float(np.dot(first_chroma, changed_chroma)) < 0.99


def test_invalid_inputs_are_rejected() -> None:
    a, b = _track("a"), _track("b")
    with pytest.raises(ValueError):
        search_directional(a, a, _structure(a), _structure(a))
    with pytest.raises(ValueError):
        search_directional(a, b, _structure(a), _structure(b), top_k=0)
    with pytest.raises(ValueError):
        search_directional(a, b, _structure(a), _structure(b), max_candidates=0)
    with pytest.raises(ValueError):
        search_directional(a, b, {**_structure(a), "sample_rate": 48_000}, _structure(b))

