from __future__ import annotations

import json

import pytest

from musicmix.phrase_evidence import analyze_phrase_evidence


RATE = 48_000
DURATION = RATE * 10


def _vocals():
    return {
        "schema_version": "musicmix.vocal_boundaries.2",
        "track_id": "track-a",
        "canonical_file_sha256": "a" * 64,
        "canonical_audio_uri": "C:/audio/track-a.wav",
        "sample_rate": RATE,
        "duration_samples": DURATION,
        "words": [
            {"start_sample": RATE, "end_sample": RATE + 4_000, "text": "open"},
            {"start_sample": RATE + 10_000, "end_sample": RATE + 14_000, "text": "phrase."},
        ],
        "asr_segments": [
            {"start_sample": RATE, "end_sample": RATE + 4_000, "text": "open"},
            {"start_sample": RATE + 10_000, "end_sample": RATE + 14_000, "text": "phrase."},
        ],
        "phrases": [
            {"start_sample": RATE, "end_sample": RATE + 14_000,
             "boundary_source": "asr_word_gap_heuristic"},
        ],
    }


def test_uncovered_word_gap_is_unknown_and_not_certified_phrase_exit_and_metadata_is_json_safe():
    result = analyze_phrase_evidence(_vocals(), RATE + 6_000, RATE, DURATION, "outgoing")

    assert result["boundary_class"] == "word_timing_without_asr_coverage"
    assert result["status"] == "unknown"
    assert result["word_gap"] is False
    assert result["true_phrase_closure"] == "unknown"
    assert "local_asr_coverage_missing" in result["reasons"]
    assert result["identity"]["content_identity_complete"] is True
    json.dumps(result, allow_nan=False)


def test_observed_segment_end_is_distinct_from_an_open_phrase_and_active_word_is_unsafe():
    vocals = _vocals()
    segment_end = analyze_phrase_evidence(vocals, RATE + 14_000, RATE, DURATION, "outgoing")
    active = analyze_phrase_evidence(vocals, RATE + 2_000, RATE, DURATION, "outgoing")

    assert segment_end["boundary_class"] == "observed_asr_segment_end"
    assert segment_end["status"] == "supported"
    assert active["boundary_class"] == "active_utterance"
    assert active["status"] == "unsafe"
    assert active["active_utterance"] is True


def test_verified_internal_acoustic_silence_can_clear_segment_overlap_without_phrase_closure():
    vocals = _vocals()
    vocals["words"] = []
    vocals["phrases"] = []
    vocals["asr_segments"] = [
        {"start_sample": RATE, "end_sample": RATE + 14_000, "text": "broad segment"},
    ]
    vocals["coverage_intervals"] = [
        {"start_sample": RATE + 5_000, "end_sample": RATE + 9_000,
         "status": "acoustic_silence_supported"},
    ]

    result = analyze_phrase_evidence(vocals, RATE + 6_000, RATE, DURATION, "incoming")

    assert result["boundary_class"] == "acoustic_clearance_supported"
    assert result["status"] == "supported"
    assert result["active_asr_segment"] is True
    assert result["active_utterance"] is False
    assert result["true_phrase_closure"] == "unknown"
    assert "active_asr_segment_overridden_by_verified_acoustic_silence" in result["reasons"]


def test_segment_only_occupancy_is_word_safe_but_phrase_continuity_unknown():
    vocals = _vocals()
    vocals["words"] = [
        {"start_sample": RATE + 20_000, "end_sample": RATE + 24_000, "text": "later"},
    ]
    vocals["asr_segments"] = [
        {"start_sample": RATE, "end_sample": RATE + 14_000, "text": "broad decoder segment"},
    ]

    result = analyze_phrase_evidence(vocals, RATE + 6_000, RATE, DURATION, "outgoing")

    assert result["status"] == "supported"
    assert result["timing_status"] == "supported"
    assert result["word_clearance_status"] == "supported"
    assert result["active_asr_segment"] is True
    assert result["active_utterance"] is False
    assert result["boundary_class"] == "asr_segment_occupancy_unknown"
    assert result["phrase_continuity_status"] == "unknown"
    assert "active_asr_segment_is_not_word_clearance_failure" in result["reasons"]
    assert "phrase_continuity_unknown_from_asr_segment_occupancy" in result["reasons"]


def test_isolated_words_without_asr_coverage_remain_unknown():
    vocals = _vocals()
    vocals.pop("asr_segments")
    result = analyze_phrase_evidence(vocals, RATE + 6_000, RATE, DURATION, "outgoing")
    assert result["status"] == "unknown"
    assert result["timing_status"] == "unknown"
    assert "missing_asr_coverage_is_unknown" in result["reasons"]


def test_remote_asr_segment_cannot_certify_an_uncovered_local_gap():
    vocals = _vocals()
    vocals["words"] = [
        {"start_sample": RATE + 20_000, "end_sample": RATE + 24_000, "text": "later"},
    ]
    vocals["asr_segments"] = [
        {"start_sample": RATE, "end_sample": RATE + 4_000, "text": "earlier segment"},
    ]
    result = analyze_phrase_evidence(vocals, RATE + 10_000, RATE, DURATION, "outgoing")
    assert result["status"] == "unknown"
    assert result["timing_status"] == "unknown"
    assert result["asr_coverage_present"] is True
    assert result["local_coverage_present"] is False
    assert "local_asr_coverage_missing" in result["reasons"]


def test_missing_vocals_and_incoming_pickup_are_unknown_not_silence():
    missing = analyze_phrase_evidence(None, RATE, RATE, DURATION, "incoming")
    pickup = analyze_phrase_evidence(_vocals(), 100, RATE, DURATION, "incoming")

    assert missing["status"] == "unknown"
    assert missing["boundary_class"] == "true_phrase_closure_unknown"
    assert missing["pickup_uncertainty"] is True
    assert pickup["pickup_uncertainty"] is True
    assert pickup["true_phrase_closure"] == "unknown"


def test_noncanonical_clock_and_mismatched_document_are_rejected_or_unknown():
    with pytest.raises(ValueError, match="canonical"):
        analyze_phrase_evidence({}, 0, 44_100, DURATION, "outgoing")
    vocals = _vocals()
    vocals["sample_rate"] = 44_100
    result = analyze_phrase_evidence(vocals, RATE + 6_000, RATE, DURATION, "outgoing")
    assert result["status"] == "unknown"
    assert "canonical_identity_metadata_mismatch" in result["reasons"]
