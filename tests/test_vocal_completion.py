from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from musicmix.vocal_completion import (
    CANONICAL_SAMPLE_RATE,
    VocalCompletionClockError,
    VocalCompletionInputError,
    _frame_matrix,
    _frame_descriptors,
    _yin_pitch,
    analyze_vocal_completion,
    clock_for_file,
    describe_vocal_cue,
    extract_vocal_features,
    load_source_clocks,
    rank_phrase_end_candidates,
)

SR = CANONICAL_SAMPLE_RATE
START = 3_360_000


def _bursts(duration=6.0, gain=1.0, sample_rate=SR):
    rng = np.random.default_rng(7)
    t = np.arange(int(duration * sample_rate)) / sample_rate
    signal = np.zeros_like(t, dtype=np.float64)
    spans = [(0.4, 0.95, 120.0), (1.15, 1.75, 150.0), (2.20, 2.70, 110.0), (3.35, 4.10, 135.0)]
    for start, end, f0 in spans:
        mask = (t >= start) & (t < end)
        local = t[mask] - start
        envelope = np.minimum(1.0, np.minimum(local / 0.03, (end - start - local) / 0.04))
        signal[mask] += envelope * (
            0.30 * np.sin(2 * np.pi * f0 * local)
            + 0.12 * np.sin(2 * np.pi * 2 * f0 * local)
            + 0.06 * np.sin(2 * np.pi * 3 * f0 * local)
        )
    signal += 0.002 * rng.standard_normal(t.size)
    return (gain * signal).astype(np.float32)


def _clock(tmp_path: Path, name="clip.wav"):
    items = [{
        "file": name,
        "source_sha256": "0" * 64,
        "source_start_sample": START,
        "sample_rate": SR,
        "duration_seconds": 6.0,
        "sha256": "0" * 64,
    }]
    path = tmp_path / "clocks.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    return clock_for_file(load_source_clocks(path), name)


def test_canonical_clock_maps_local_cue_to_source_sample(tmp_path):
    samples = _bursts()
    cue = START + int(2.0 * SR)
    result = analyze_vocal_completion(
        samples, SR, cue, canonical_start_sample=START
    )
    assert result["identity"]["cue_sample"] == cue
    assert result["identity"]["cue_local_sample"] == int(2.0 * SR)
    assert result["identity"]["canonical_start_sample"] == START
    assert result["identity"]["canonical_end_sample"] == START + samples.size
    assert result["window"]["canonical_start_sample"] >= START
    assert result["window"]["canonical_end_sample"] <= START + samples.size

    at_start = analyze_vocal_completion(samples, SR, START, canonical_start_sample=START)
    at_end = analyze_vocal_completion(
        samples, SR, START + samples.size, canonical_start_sample=START
    )
    assert at_start["window"]["local_start_sample"] == 0
    assert at_end["window"]["local_end_sample"] == samples.size

    with pytest.raises(VocalCompletionClockError):
        analyze_vocal_completion(samples, SR, START - 1, canonical_start_sample=START)
    with pytest.raises(VocalCompletionClockError):
        analyze_vocal_completion(samples, SR, START + samples.size + 1, canonical_start_sample=START)


def test_gain_robustness_of_relative_descriptors_and_pitch():
    base = _bursts(gain=1.0)
    quiet = _bursts(gain=0.05)
    a = extract_vocal_features(base, SR, canonical_start_sample=START)
    b = extract_vocal_features(quiet, SR, canonical_start_sample=START)

    np.testing.assert_array_equal(a.active, b.active)
    np.testing.assert_array_equal(a.onsets, b.onsets)
    assert abs((a.threshold_db - b.threshold_db) - (-20.0 * math.log10(0.05))) < 1e-3
    assert abs((a.rms_db.mean() - b.rms_db.mean()) - (-20.0 * math.log10(0.05))) < 0.1
    assert np.allclose(a.spectral_flatness, b.spectral_flatness, rtol=1e-5, atol=1e-7)
    assert np.allclose(a.zero_crossing_rate, b.zero_crossing_rate)
    np.testing.assert_allclose(a.f0_hz, b.f0_hz, rtol=1e-4, atol=0.05, equal_nan=True)

    cue = START + int(2.75 * SR)
    da = describe_vocal_cue(a, cue)
    db = describe_vocal_cue(b, cue)
    assert da["voice_activity"]["active_at_cue"] == db["voice_activity"]["active_at_cue"]
    assert da["outgoing_completion_descriptors"]["unvoiced_gap_seconds"] == pytest.approx(
        db["outgoing_completion_descriptors"]["unvoiced_gap_seconds"]
    )
    assert da["terminal_lengthening"]["ratio"] == pytest.approx(
        db["terminal_lengthening"]["ratio"], nan_ok=True
    )
    assert da["completion_evidence"]["score_uncalibrated"] == pytest.approx(
        db["completion_evidence"]["score_uncalibrated"]
    )


def test_nonfinite_and_unsupported_inputs_are_rejected():
    samples = _bursts(duration=2.0)
    bad = samples.copy()
    bad[100] = np.nan
    with pytest.raises(VocalCompletionInputError):
        extract_vocal_features(bad, SR)
    bad[100] = np.inf
    with pytest.raises(VocalCompletionInputError):
        extract_vocal_features(bad, SR)
    with pytest.raises(VocalCompletionClockError):
        extract_vocal_features(samples, 44_100)
    with pytest.raises(VocalCompletionInputError):
        analyze_vocal_completion(samples, SR, math.nan)


def test_missing_asr_and_breath_channels_are_explicit(tmp_path):
    samples = _bursts()
    cue = START + int(2.75 * SR)
    result = analyze_vocal_completion(samples, SR, cue, canonical_start_sample=START)
    assert result["asr"]["available"] is False
    assert result["asr"]["reason"] == "no_asr_words_provided"
    assert result["asr"]["asr_segment_is_not_prosodic_phrase"] is True
    assert result["asr"]["semantic_or_syntactic_closure_inferred"] is False

    # Segments alone never become words or a prosodic phrase.
    segments = [{"start_sample": START + 100, "end_sample": START + 1000, "text": "la"}]
    only_segments = analyze_vocal_completion(
        samples, SR, cue, canonical_start_sample=START, segments=segments
    )
    assert only_segments["asr"]["available"] is False
    assert only_segments["asr"]["segment_count_near_cue"] == 0
    assert only_segments["completion_evidence"]["available"]["asr_closed_punctuation"] is False

    # A continuous voiced cue has no breath channel rather than a fabricated one.
    continuous = (0.2 * np.sin(2 * np.pi * 150 * np.arange(3 * SR) / SR)).astype(np.float32)
    inside = analyze_vocal_completion(continuous, SR, int(1.5 * SR))
    assert inside["breath"]["available"] is False
    assert inside["voice_activity"]["active_at_cue"] is True

    missing_clock = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_source_clocks(missing_clock)


def test_asr_timing_is_separate_from_prosody_and_punctuation_is_not_closure(tmp_path):
    samples = _bursts()
    clock = _clock(tmp_path)
    cue = START + int(2.75 * SR)
    words = [
        {"start_sample": START + int(2.30 * SR), "end_sample": START + int(2.70 * SR), "text": "last.", "confidence": 0.8},
        {"start_sample": START + int(3.40 * SR), "end_sample": START + int(3.80 * SR), "text": "next", "confidence": 0.7},
    ]
    segments = [{"start_sample": START + int(2.20 * SR), "end_sample": START + int(2.80 * SR), "text": "a segment"}]
    result = analyze_vocal_completion(
        samples, SR, cue, canonical_start_sample=clock.source_start_sample,
        words=words, segments=segments,
    )
    assert result["asr"]["available"] is True
    assert result["asr"]["closed_punctuation"] == "."
    assert result["asr"]["closed_punctuation_evidence"] is True
    assert result["asr"]["asr_segment_is_not_prosodic_phrase"] is True
    assert result["asr"]["segment_count_near_cue"] == 1
    assert result["asr"]["last_word_end_gap_seconds"] == pytest.approx(0.05)

    # Out-of-clock ASR timestamps are rejected rather than silently clipped.
    with pytest.raises(VocalCompletionClockError):
        analyze_vocal_completion(
            samples, SR, cue, canonical_start_sample=START,
            words=[{"start_sample": START - 5, "end_sample": START + 10, "text": "x"}],
        )


def test_ranking_is_bounded_and_marks_uncalibrated_evidence():
    samples = _bursts()
    candidates = [START + int(seconds * SR) for seconds in (0.7, 2.75, 4.3, 5.5)]
    ranked = rank_phrase_end_candidates(
        samples, SR, candidates, canonical_start_sample=START, keep_details=True
    )
    assert [item["rank"] for item in ranked] == [1, 2, 3, 4]
    for item in ranked:
        score = item["completion_score_uncalibrated"]
        assert score is None or 0.0 <= score <= 1.0
        assert item["detail"]["completion_evidence"]["score_kind"].endswith("not_closure")
        assert "closure" in item["detail"]["labels_not_emitted"]

def _tone(frequency: float, duration: float = 1.5, sample_rate: int = SR) -> np.ndarray:
    t = np.arange(int(duration * sample_rate)) / sample_rate
    return (
        0.30 * np.sin(2 * np.pi * frequency * t)
        + 0.12 * np.sin(2 * np.pi * 2 * frequency * t)
        + 0.06 * np.sin(2 * np.pi * 3 * frequency * t)
    ).astype(np.float32)


def test_yin_known_periodic_tones_and_octaves():
    estimates = {}
    for frequency in (80.0, 110.0, 220.0, 330.0, 440.0):
        frames = _frame_matrix(_tone(frequency))
        f0, voicing = _yin_pitch(frames, SR, chunk_frames=64)
        finite = f0[np.isfinite(f0)]
        assert finite.size > f0.size * 0.9
        median = float(np.median(finite))
        estimates[frequency] = median
        assert median == pytest.approx(frequency, rel=0.01), (frequency, median)
        assert float(np.median(voicing)) > 0.9
    assert estimates[220.0] / estimates[110.0] == pytest.approx(2.0, rel=0.02)
    assert estimates[440.0] / estimates[220.0] == pytest.approx(2.0, rel=0.02)
    assert estimates[330.0] / estimates[110.0] == pytest.approx(3.0, rel=0.02)


def test_yin_chunk_size_is_result_invariant():
    frames = _frame_matrix(_tone(180.0, duration=1.0))
    small_f0, small_voicing = _yin_pitch(frames, SR, chunk_frames=7)
    large_f0, large_voicing = _yin_pitch(frames, SR, chunk_frames=256)
    assert np.array_equal(small_f0, large_f0, equal_nan=True)
    assert np.array_equal(small_voicing, large_voicing)


def test_optional_librosa_yin_agrees_on_known_tone():
    librosa = pytest.importorskip("librosa")
    frequency = 220.0
    signal = _tone(frequency, duration=1.0)
    frames = _frame_matrix(signal)
    f0, _ = _yin_pitch(frames, SR, chunk_frames=64)
    reference = librosa.yin(
        signal.astype(np.float32),
        fmin=70.0,
        fmax=500.0,
        sr=SR,
        frame_length=2048,
        hop_length=480,
        center=False,
    )
    reference = reference[np.isfinite(reference)]
    assert float(np.median(f0[np.isfinite(f0)])) == pytest.approx(frequency, rel=0.01)
    assert float(np.median(reference)) == pytest.approx(frequency, rel=0.02)


def _reference_frame_descriptors(signal):
    from scipy.signal import stft

    frames = _frame_matrix(signal)
    spectrum = np.abs(stft(
        signal, fs=SR, window="hann", nperseg=2048, noverlap=2048 - 480,
        boundary=None, padded=False,
    )[2])
    usable = min(frames.shape[0], spectrum.shape[1])
    spectrum = spectrum[:, :usable]
    if usable < frames.shape[0]:
        spectrum = np.pad(spectrum, ((0, 0), (0, frames.shape[0] - usable)))
    power = spectrum.astype(np.float64) ** 2 + 1e-20
    flux = np.zeros(usable, dtype=np.float64)
    if usable > 1:
        difference = np.maximum(0.0, spectrum[:, 1:] - spectrum[:, :-1])
        flux[1:] = difference.sum(axis=0) / (spectrum[:, 1:].sum(axis=0) + 1e-9)
    frequencies = np.linspace(0.0, SR / 2.0, power.shape[0])
    high = frequencies >= 2000.0
    return {
        "rms": np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-18),
        "zero_crossing": np.mean(
            np.abs(np.diff(np.signbit(frames), axis=1)), axis=1
        ),
        "flatness": (
            np.exp(np.mean(np.log(power[1:, :]), axis=0))
            / (np.mean(power[1:, :], axis=0) + 1e-20)
        ),
        "hf_ratio": (
            power[high, :].sum(axis=0) / (power.sum(axis=0) + 1e-20)
        ),
        "flux": flux,
    }


def test_feature_chunk_size_is_result_invariant():
    signal = _bursts(duration=3.0)
    small = _frame_descriptors(signal, SR, chunk_frames=7)
    large = _frame_descriptors(signal, SR, chunk_frames=512)
    for name in ("rms", "zero_crossing", "flatness", "hf_ratio", "flux", "voicing"):
        np.testing.assert_allclose(small[name], large[name], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        small["f0"], large["f0"], rtol=1e-12, atol=1e-9, equal_nan=True
    )


def test_chunked_descriptors_match_reference_short_stft():
    signal = _tone(180.0, duration=1.0)
    chunked = _frame_descriptors(signal, SR, chunk_frames=7)
    reference = _reference_frame_descriptors(signal)
    for name in reference:
        np.testing.assert_allclose(
            chunked[name], reference[name], rtol=1e-5, atol=1e-8
        )

