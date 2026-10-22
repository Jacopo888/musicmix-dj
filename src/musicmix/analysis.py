"""Deterministic, offline audio descriptors for the first MusicMix baseline.

This module intentionally provides a modest structural baseline rather than a
learned beat or section model.  It computes short-time FFT descriptors and a
spectral-flux onset grid.  The onset grid is useful for reproducible cue
generation and diagnostics, but ``Cue.kind`` and ``TrackAnalysis.diagnostics``
make the heuristic nature explicit: no downbeat or section labels are
invented when an ML/provider backend is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .audio import AudioTrack


ANALYSIS_VERSION = "numpy-fft-onset-grid-v1"
_CONTEXT_SECONDS = 2.0
_MIN_ONSET_SEPARATION_SECONDS = 0.12
_MAX_STFT_BATCH = 256
_MAX_TEMPO_FRAMES = 131_072
_MAX_ANALYSIS_FRAMES = 250_000


@dataclass(frozen=True)
class Cue:
    """One internally valid cue and its local descriptor vector.

    Cues are expressed in source sample indices.  The baseline uses
    ``kind='onset_grid'`` for all emitted events; this must not be interpreted
    as a downbeat or a semantically labelled section boundary.
    """

    sample: int
    chroma: tuple[float, ...]
    rms: float
    onset: float
    timbre: tuple[float, ...]
    tempo_bpm: float | None
    confidence: float
    kind: str = "onset_grid"

    def __post_init__(self) -> None:
        sample = int(self.sample)
        if sample < 0:
            raise ValueError("cue sample must be non-negative")
        chroma = tuple(float(value) for value in self.chroma)
        timbre = tuple(float(value) for value in self.timbre)
        if len(chroma) != 12:
            raise ValueError("chroma must contain exactly 12 values")
        if not all(math.isfinite(value) for value in chroma + timbre):
            raise ValueError("cue descriptors must be finite")
        rms = float(self.rms)
        onset = float(self.onset)
        confidence = float(self.confidence)
        if not all(math.isfinite(value) for value in (rms, onset, confidence)):
            raise ValueError("cue scalar descriptors must be finite")
        if rms < 0.0 or onset < 0.0:
            raise ValueError("rms and onset must be non-negative")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        tempo = None if self.tempo_bpm is None else float(self.tempo_bpm)
        if tempo is not None and (not math.isfinite(tempo) or tempo <= 0.0):
            raise ValueError("tempo_bpm must be positive or None")
        object.__setattr__(self, "sample", sample)
        object.__setattr__(self, "chroma", chroma)
        object.__setattr__(self, "timbre", timbre)
        object.__setattr__(self, "rms", rms)
        object.__setattr__(self, "onset", onset)
        object.__setattr__(self, "tempo_bpm", tempo)
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class TrackAnalysis:
    """Feature/cue output for one :class:`~musicmix.audio.AudioTrack`."""

    track_id: str
    sample_rate: int
    duration_samples: int
    cues: tuple[Cue, ...]
    backend: str
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.track_id:
            raise ValueError("track_id must be non-empty")
        if int(self.sample_rate) <= 0:
            raise ValueError("sample_rate must be positive")
        if int(self.duration_samples) < 0:
            raise ValueError("duration_samples must be non-negative")
        cues = tuple(self.cues)
        if any(not isinstance(cue, Cue) for cue in cues):
            raise TypeError("cues must contain Cue objects")
        object.__setattr__(self, "sample_rate", int(self.sample_rate))
        object.__setattr__(self, "duration_samples", int(self.duration_samples))
        object.__setattr__(self, "cues", cues)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


def _next_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _analysis_frame_length(sample_rate: int, duration_samples: int) -> int:
    # About 46 ms gives a useful transient/spectral compromise.  The cap
    # bounds FFT work for high-rate sources, while the duration cap handles
    # tiny in-memory fixtures without padding a complete short song.
    target = max(64, min(4096, int(round(sample_rate * 0.046))))
    return min(duration_samples, _next_power_of_two(target))


def _stft_strength(
    mono: np.ndarray,
    n_fft: int,
    hop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frame centres, RMS values, and positive spectral flux.

    FFTs are performed in bounded batches so a long source does not create a
    full ``(frames, bins)`` spectrogram in memory.
    """

    n_samples = int(mono.size)
    if n_fft < 16 or n_samples < n_fft:
        return np.empty(0, dtype=np.int64), np.empty(0), np.empty(0)
    starts = np.arange(0, n_samples - n_fft + 1, hop, dtype=np.int64)
    centres = starts + n_fft // 2
    rms = np.empty(starts.size, dtype=np.float64)
    flux = np.zeros(starts.size, dtype=np.float64)
    fft_window = np.hanning(n_fft).astype(np.float64)
    sample_offsets = np.arange(n_fft, dtype=np.int64)
    previous: np.ndarray | None = None

    for begin in range(0, starts.size, _MAX_STFT_BATCH):
        batch_starts = starts[begin : begin + _MAX_STFT_BATCH]
        indices = batch_starts[:, None] + sample_offsets[None, :]
        raw_frames = mono[indices]
        rms[begin : begin + batch_starts.size] = np.sqrt(
            np.mean(raw_frames * raw_frames, axis=1)
        )
        spectra = np.abs(np.fft.rfft(raw_frames * fft_window[None, :], axis=1))
        log_spectra = np.log1p(spectra)
        # Compare every frame to its immediate predecessor.  Keeping only
        # the previous *batch's* final spectrum would otherwise discard all
        # intra-batch onsets on longer tracks.
        positive_change = np.zeros_like(log_spectra)
        if previous is not None:
            positive_change[0] = np.maximum(log_spectra[0] - previous, 0.0)
        if batch_starts.size > 1:
            positive_change[1:] = np.maximum(
                log_spectra[1:] - log_spectra[:-1], 0.0
            )
        flux[begin : begin + batch_starts.size] = np.mean(positive_change, axis=1)
        previous = log_spectra[-1]

    return centres, rms, flux


def _normalized_onset(flux: np.ndarray) -> np.ndarray:
    if flux.size == 0:
        return np.empty(0, dtype=np.float64)
    finite = np.nan_to_num(flux.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    baseline = float(np.percentile(finite, 50.0))
    upper = float(np.percentile(finite, 95.0))
    peak = float(np.max(finite))
    scale = max(upper - baseline, peak * 0.05, 1e-12)
    return np.clip((finite - baseline) / scale, 0.0, 1.0)


def _onset_candidates(
    centres: np.ndarray,
    rms: np.ndarray,
    flux: np.ndarray,
    sample_rate: int,
    hop: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find reproducible local flux peaks and their [0, 1] onset scores."""

    if flux.size < 3 or rms.size != flux.size:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    scores = _normalized_onset(flux)
    max_rms = float(np.max(rms)) if rms.size else 0.0
    if max_rms <= 1e-8:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    # A small relative gate suppresses numerical noise but retains quiet
    # musical attacks in a track with a louder section elsewhere.
    energy_gate = max(1e-8, max_rms * 0.002)
    score_gate = max(0.20, float(np.median(scores)) + 0.5 * float(np.std(scores)))
    local_max = np.zeros(scores.size, dtype=bool)
    local_max[1:-1] = (scores[1:-1] >= scores[:-2]) & (scores[1:-1] > scores[2:])
    candidate_indices = np.flatnonzero(
        local_max & (scores >= score_gate) & (rms >= energy_gate)
    )
    if candidate_indices.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    # Refractory suppression makes a single attack one cue even when its
    # spectral flux spans adjacent frames.  Resolve close peaks by retaining
    # the stronger one; ties are resolved by source order.
    min_gap = max(1, int(round(_MIN_ONSET_SEPARATION_SECONDS * sample_rate / hop)))
    selected: list[int] = []
    for index in candidate_indices.tolist():
        if not selected or index - selected[-1] >= min_gap:
            selected.append(index)
        elif scores[index] > scores[selected[-1]]:
            selected[-1] = index
    selected_array = np.asarray(selected, dtype=np.int64)
    return centres[selected_array], scores[selected_array]


def _estimate_tempo(
    flux: np.ndarray,
    sample_rate: int,
    hop: int,
    onset_count: int,
) -> tuple[float | None, float]:
    """Estimate a coarse global tempo, or return ``(None, 0)``.

    Autocorrelation is used only as a diagnostic for a regular onset train.
    It is intentionally not exposed as a beat/downbeat claim.
    """

    if onset_count < 4 or flux.size < 8:
        return None, 0.0
    signal = np.maximum(flux - float(np.percentile(flux, 50.0)), 0.0)
    if signal.size > _MAX_TEMPO_FRAMES:
        factor = int(math.ceil(signal.size / _MAX_TEMPO_FRAMES))
        usable = signal.size - signal.size % factor
        signal = signal[:usable].reshape(-1, factor).mean(axis=1)
        hop *= factor
    signal -= float(np.mean(signal))
    energy = float(np.dot(signal, signal))
    if energy <= 1e-12:
        return None, 0.0
    frame_rate = float(sample_rate) / float(hop)
    lag_min = max(1, int(round(frame_rate * 60.0 / 200.0)))
    lag_max = min(signal.size - 1, int(round(frame_rate * 60.0 / 40.0)))
    if lag_max <= lag_min:
        return None, 0.0
    fft_size = _next_power_of_two(2 * signal.size - 1)
    spectrum = np.fft.rfft(signal, n=fft_size)
    autocorrelation = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size)[: signal.size]
    region = autocorrelation[lag_min : lag_max + 1]
    if region.size == 0:
        return None, 0.0
    peak_offset = int(np.argmax(region))
    peak_lag = lag_min + peak_offset
    peak_value = float(autocorrelation[peak_lag])
    ratio = peak_value / max(energy, 1e-12)
    background = float(np.median(region))
    prominence = max(0.0, peak_value - background) / max(energy, 1e-12)
    confidence = float(np.clip((ratio - 0.08) / 0.35, 0.0, 1.0))
    confidence *= float(np.clip(prominence / 0.18, 0.0, 1.0))
    if ratio < 0.12 or confidence < 0.12:
        return None, 0.0
    bpm = 60.0 * frame_rate / float(peak_lag)
    return float(bpm), float(np.clip(confidence, 0.0, 1.0))


def _local_features(
    mono: np.ndarray,
    sample_rate: int,
    sample: int,
    n_fft: int,
    fft_window: np.ndarray,
) -> tuple[tuple[float, ...], float, tuple[float, ...]]:
    """Extract chroma, RMS, and compact timbre descriptors at one sample."""

    start = max(0, min(int(sample) - n_fft // 2, mono.size - n_fft))
    segment = mono[start : start + n_fft]
    if segment.size < n_fft:
        padded = np.zeros(n_fft, dtype=np.float64)
        padded[: segment.size] = segment
        segment = padded
    rms = float(np.sqrt(np.mean(segment * segment))) if segment.size else 0.0
    magnitude = np.abs(np.fft.rfft(segment * fft_window))
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))

    chroma = np.zeros(12, dtype=np.float64)
    chroma_mask = (frequencies >= 55.0) & (frequencies <= min(5000.0, sample_rate / 2.0))
    if np.any(chroma_mask):
        selected_frequencies = frequencies[chroma_mask]
        selected_magnitude = magnitude[chroma_mask]
        midi = 69.0 + 12.0 * np.log2(np.maximum(selected_frequencies, 1e-12) / 440.0)
        pitch_classes = np.mod(np.rint(midi).astype(np.int64), 12)
        for pitch_class in range(12):
            chroma[pitch_class] = float(
                np.sum(selected_magnitude[pitch_classes == pitch_class])
            )
    chroma_total = float(np.sum(chroma))
    if chroma_total > 1e-12:
        chroma /= chroma_total

    timbre_mask = frequencies > 0.0
    timbre_frequencies = frequencies[timbre_mask]
    timbre_magnitude = magnitude[timbre_mask]
    total = float(np.sum(timbre_magnitude))
    nyquist = max(float(sample_rate) / 2.0, 1.0)
    if total <= 1e-12:
        timbre = (0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        centroid = float(np.sum(timbre_frequencies * timbre_magnitude) / total)
        bandwidth = float(
            np.sqrt(
                np.sum(((timbre_frequencies - centroid) ** 2) * timbre_magnitude)
                / total
            )
        )
        cumulative = np.cumsum(timbre_magnitude)
        rolloff_at = int(np.searchsorted(cumulative, 0.85 * total))
        rolloff = float(timbre_frequencies[min(rolloff_at, timbre_frequencies.size - 1)])
        log_mean = float(np.mean(np.log(np.maximum(timbre_magnitude, 1e-12))))
        flatness = float(np.exp(log_mean) / max(float(np.mean(timbre_magnitude)), 1e-12))
        if segment.size > 1:
            zcr = float(np.mean((segment[:-1] * segment[1:]) < 0.0))
        else:
            zcr = 0.0
        timbre = (
            float(np.clip(centroid / nyquist, 0.0, 1.0)),
            float(np.clip(bandwidth / nyquist, 0.0, 1.0)),
            float(np.clip(rolloff / nyquist, 0.0, 1.0)),
            float(np.clip(flatness, 0.0, 1.0)),
            float(np.clip(zcr, 0.0, 1.0)),
        )
    return tuple(float(value) for value in chroma), rms, timbre


def _diagnostics_base(
    track: AudioTrack,
    context_seconds: float,
    n_fft: int | None = None,
    hop: int | None = None,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "feature_backend": "numpy.fft",
        "status": "pending",
        "silence": False,
        "context_seconds_each_side": float(context_seconds),
        "beat_status": "missing_untrusted",
        "downbeat_status": "unavailable",
        "section_status": "unavailable",
        "beat_available": False,
        "downbeat_available": False,
        "section_available": False,
        "heuristic_cue_kind": "onset_grid",
        "warnings": (
            "Cues are deterministic spectral-flux onset heuristics; no beat/downbeat "
            "or section model is loaded.",
        ),
    }
    if n_fft is not None:
        diagnostics["frame_length_samples"] = int(n_fft)
    if hop is not None:
        diagnostics["hop_samples"] = int(hop)
    return diagnostics


def analyze_track(
    track: AudioTrack,
    *,
    context_seconds: float = _CONTEXT_SECONDS,
) -> TrackAnalysis:
    """Compute deterministic descriptors and valid internal onset cues.

    A cue is emitted only when at least ``context_seconds`` (minimum two
    seconds) exists before and after it.  A short or silent track returns an
    empty cue tuple and diagnostics describing why; it never receives made-up
    beats.  ``tempo_bpm`` is populated only when at least four detected onset
    events support a coarse autocorrelation estimate.
    """

    if not isinstance(track, AudioTrack):
        raise TypeError("analyze_track expects an AudioTrack")
    requested_context = float(context_seconds)
    if not math.isfinite(requested_context) or requested_context < 0.0:
        raise ValueError("context_seconds must be finite and non-negative")
    context = max(_CONTEXT_SECONDS, requested_context)
    samples = np.asarray(track.samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] < 1:
        raise ValueError("track samples must have shape (frames, channels)")
    if not np.all(np.isfinite(samples)):
        raise ValueError("track samples must contain only finite values")
    mono = np.mean(samples, axis=1, dtype=np.float64)
    duration = int(mono.size)
    context_samples = int(round(context * track.sample_rate))
    n_fft = _analysis_frame_length(track.sample_rate, duration) if duration else 0
    hop = max(1, n_fft // 4) if n_fft else 1
    if n_fft:
        expected_frames = max(1, 1 + (duration - n_fft) // hop)
        if expected_frames > _MAX_ANALYSIS_FRAMES:
            # Preserve source sample coordinates while bounding the number of
            # FFT frames retained for a very long recording.
            hop = max(hop, int(math.ceil((duration - n_fft) / _MAX_ANALYSIS_FRAMES)))
    diagnostics = _diagnostics_base(track, context, n_fft or None, hop if n_fft else None)
    diagnostics["duration_seconds"] = float(duration / track.sample_rate)
    diagnostics["context_samples_each_side"] = context_samples

    if duration == 0:
        diagnostics["status"] = "empty_track"
        diagnostics["warnings"] = diagnostics["warnings"] + ("Track has no samples.",)
        return TrackAnalysis(track.track_id, track.sample_rate, duration, (), ANALYSIS_VERSION, diagnostics)
    if duration < 2 * context_samples or n_fft < 16:
        diagnostics["status"] = "short_track"
        diagnostics["warnings"] = diagnostics["warnings"] + (
            "Track is shorter than the required two-sided context; no cue emitted.",
        )
        return TrackAnalysis(track.track_id, track.sample_rate, duration, (), ANALYSIS_VERSION, diagnostics)

    centres, frame_rms, flux = _stft_strength(mono, n_fft, hop)
    if frame_rms.size == 0:
        diagnostics["status"] = "insufficient_frames"
        return TrackAnalysis(track.track_id, track.sample_rate, duration, (), ANALYSIS_VERSION, diagnostics)
    if float(np.max(frame_rms)) <= 1e-8:
        diagnostics["status"] = "silence"
        diagnostics["silence"] = True
        diagnostics["warnings"] = diagnostics["warnings"] + (
            "Signal energy is below the silence threshold; no reliable cue emitted.",
        )
        return TrackAnalysis(track.track_id, track.sample_rate, duration, (), ANALYSIS_VERSION, diagnostics)

    onset_samples, onset_scores = _onset_candidates(
        centres, frame_rms, flux, track.sample_rate, hop
    )
    tempo, tempo_confidence = _estimate_tempo(flux, track.sample_rate, hop, onset_samples.size)
    diagnostics["tempo_bpm"] = tempo
    diagnostics["tempo_confidence"] = float(tempo_confidence)
    diagnostics["detected_onsets"] = int(onset_samples.size)
    valid_mask = (onset_samples >= context_samples) & (
        onset_samples <= duration - context_samples
    )
    onset_samples = onset_samples[valid_mask]
    onset_scores = onset_scores[valid_mask]
    diagnostics["context_valid_onsets"] = int(onset_samples.size)
    diagnostics["cue_count"] = int(onset_samples.size)
    if onset_samples.size == 0:
        diagnostics["status"] = "no_reliable_internal_onsets"
        diagnostics["warnings"] = diagnostics["warnings"] + (
            "No sufficiently strong onset exists inside the two-sided context.",
        )
        return TrackAnalysis(track.track_id, track.sample_rate, duration, (), ANALYSIS_VERSION, diagnostics)

    fft_window = np.hanning(n_fft).astype(np.float64)
    max_rms = float(np.max(frame_rms))
    cues: list[Cue] = []
    for sample, onset_score in zip(onset_samples.tolist(), onset_scores.tolist()):
        chroma, rms, timbre = _local_features(
            mono, track.sample_rate, int(sample), n_fft, fft_window
        )
        relative_energy = float(np.clip(rms / max(max_rms, 1e-12), 0.0, 1.0))
        confidence = float(np.clip(0.70 * onset_score + 0.30 * relative_energy, 0.0, 1.0))
        if tempo is not None:
            confidence = float(np.clip(0.75 * confidence + 0.25 * tempo_confidence, 0.0, 1.0))
        cues.append(
            Cue(
                sample=int(sample),
                chroma=chroma,
                rms=rms,
                onset=float(onset_score),
                timbre=timbre,
                tempo_bpm=tempo,
                confidence=confidence,
                kind="onset_grid",
            )
        )
    diagnostics["status"] = "ok"
    diagnostics["cue_count"] = len(cues)
    return TrackAnalysis(
        track.track_id,
        track.sample_rate,
        duration,
        tuple(cues),
        ANALYSIS_VERSION,
        diagnostics,
    )


__all__ = ["ANALYSIS_VERSION", "Cue", "TrackAnalysis", "analyze_track"]
