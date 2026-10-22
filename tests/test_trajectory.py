"""Regression tests for the descriptive beat-synchronous trajectory API."""

from __future__ import annotations

import numpy as np
import pytest

from musicmix.audio import AudioTrack
from musicmix.trajectory import (
    TRAJECTORY_VERSION,
    extract_trajectory,
    trajectory_to_dict,
)


def _track(*, silent: bool = False) -> AudioTrack:
    sample_rate = 1_000
    samples = np.arange(120_000, dtype=np.float64)
    if silent:
        mono = np.zeros_like(samples)
    else:
        # Distinct low/high components keep the brightness and bass ratios
        # observable while a short amplitude step exercises RMS/flux paths.
        mono = 0.12 * np.sin(2.0 * np.pi * 55.0 * samples / sample_rate)
        mono += 0.10 * np.sin(2.0 * np.pi * 220.0 * samples / sample_rate)
        mono += (samples >= 80_000) * 0.15 * np.sin(2.0 * np.pi * 380.0 * samples / sample_rate)
    stereo = np.column_stack([mono, mono]).astype(np.float32)
    return AudioTrack("synthetic-trajectory", "synthetic.wav", sample_rate, stereo)


def _beats() -> list[dict[str, int]]:
    return [{"sample": sample} for sample in range(0, 120_000, 1_000)]


def test_extracts_all_requested_scales_and_explicit_shift() -> None:
    result = extract_trajectory(
        _track(),
        80_000,
        _beats(),
        side="outgoing",
        cue_shift_samples=25,
    )

    assert result.backend == TRAJECTORY_VERSION
    assert result.requested_cue_sample == 80_000
    assert result.cue_sample == 80_025
    assert result.cue_shift_samples == 25
    assert [window.window_beats for window in result.windows] == [8, 16, 32]
    assert all(window.cue_sample == 80_025 for window in result.windows)
    assert result.diagnostics["confidence"] is None
    assert result.diagnostics["confidence_status"] == "unknown_unvalidated"
    assert result.diagnostics["pitch_contour"]["status"] == "unknown"
    assert "not tonal tension" in result.diagnostics["tonal_stability_semantics"]
    for window in result.windows:
        assert window.confidence is None
        assert len(window.beat_feature_series) == window.window_beats
        assert len(window.chroma_mean) == 12
        values = (
            window.rms_mean,
            window.log_rms_mean,
            window.spectral_flux_mean,
            window.attack_density_per_second,
            window.spectral_centroid_hz,
            window.high_frequency_ratio,
            window.bass_entry_delta,
            window.chroma_movement,
        )
        assert all(np.isfinite(values))


def test_incoming_windows_are_beat_bounded_and_json_ready() -> None:
    result = extract_trajectory(_track(), 50_000, _beats(), side="incoming")

    assert [window.window_beats for window in result.windows] == [8, 16, 32]
    assert all(window.start_sample >= 50_000 for window in result.windows)
    assert all(window.end_sample > window.start_sample for window in result.windows)
    payload = trajectory_to_dict(result)
    assert payload["canonical_sample_rate"] == 1_000
    assert payload["tempo_bpm"] == pytest.approx(60.0)
    assert payload["windows"][0]["chroma_tonal_stability_proxy"] is not None
    assert payload["windows"][0]["confidence"] is None


def test_silence_is_finite_but_marked_unreliable() -> None:
    result = extract_trajectory(_track(silent=True), 80_000, _beats(), side="outgoing")

    assert len(result.windows) == 3
    assert result.diagnostics["status"] == "silent_or_near_silent"
    assert result.diagnostics["candidate_status"] == "unreliable_silence"
    assert all(window.rms_mean == 0.0 for window in result.windows)
    assert all(window.confidence is None for window in result.windows)
    assert all(value is not None for window in result.windows for value in window.chroma_mean)


def test_missing_beats_never_invents_tempo_or_windows() -> None:
    result = extract_trajectory(_track(), 80_000, [], side="outgoing")

    assert result.windows == ()
    assert result.beat_period_samples is None
    assert result.tempo_bpm is None
    assert result.diagnostics["status"] == "missing_beats"
    assert result.diagnostics["candidate_status"] == "unreliable_without_provider_beats"


def test_short_context_is_reported_instead_of_padded_to_32_beats() -> None:
    short = _track().samples[:45_000]
    track = AudioTrack("short", "short.wav", 1_000, short)
    result = extract_trajectory(track, 30_000, list(range(0, 45_000, 1_000)), side="outgoing")

    assert [window.window_beats for window in result.windows] == [8, 16]
    assert 32 in result.diagnostics["insufficient_local_windows"]
    assert 16 in result.diagnostics["missing_adjacent_context_windows"]
    assert result.windows[-1].chroma_tonal_stability_proxy is None
