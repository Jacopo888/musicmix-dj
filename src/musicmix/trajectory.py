"""Beat-synchronous, provider-neutral audio trajectory measurements.

The first MusicMix trajectory baseline is deliberately descriptive.  It uses
one bounded NumPy FFT path for short-time energy, spectral brightness, flux,
bass balance, and chroma distributions.  The same frame-size convention as
``musicmix.analysis`` is reused, but this module does not turn those values
into a score or a semantic claim about suspense, tension, resolution, or
musical sections.

Windows are addressed in beats supplied by a caller (normally a real beat
provider).  An outgoing profile uses the beats immediately before a cue and
an incoming profile uses the beats immediately after it.  ``cue_shift_samples``
is retained in the result so a renderer's safety adjustment cannot be hidden.
No pitch contour provider is bundled; pitch is therefore reported as unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .analysis import _analysis_frame_length
from .audio import AudioTrack


TRAJECTORY_VERSION = "numpy-fft-trajectory-v1"
DEFAULT_WINDOWS_BEATS = (8, 16, 32)
TRAJECTORY_CONFIG_HASH = hashlib.sha256(
    (
        f"{TRAJECTORY_VERSION}|frame=analysis.frame_length|hop=quarter|"
        f"windows={DEFAULT_WINDOWS_BEATS}|"
        "high_hz=2000|bass_hz=150|chroma=power-round-midi"
    ).encode("utf-8")
).hexdigest()
_HOP_FRACTION = 4
_HIGH_FREQUENCY_HZ = 2_000.0
_BASS_FREQUENCY_HZ = 150.0
_MIN_AUDIO_RMS = 1e-8


def _as_finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return None
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))


@dataclass(frozen=True)
class TrajectoryWindow:
    """Descriptors for one one-sided beat window.

    ``bass_entry_delta`` is ``bass_ratio_start - bass_ratio_end``.  It is a
    measurable within-window change; it is not evidence of an arrangement
    entry or a causal musical event.  The two chroma fields are distribution
    similarity/movement proxies, not tonal tension estimates.
    """

    side: str
    window_beats: int
    cue_sample: int
    beat_anchor_sample: int
    start_sample: int
    end_sample: int
    duration_samples: int
    rms_mean: float
    log_rms_mean: float
    spectral_flux_mean: float
    attack_density_per_second: float
    spectral_centroid_hz: float
    high_frequency_ratio: float
    bass_ratio_start: float
    bass_ratio_end: float
    bass_entry_delta: float
    chroma_mean: tuple[float, ...]
    chroma_movement: float
    chroma_local_context_similarity: float | None
    chroma_tonal_stability_proxy: float | None
    confidence: float | None = None
    diagnostics: dict[str, Any] | None = None
    beat_feature_series: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.side not in {"outgoing", "incoming"}:
            raise ValueError("side must be outgoing or incoming")
        if int(self.window_beats) <= 0:
            raise ValueError("window_beats must be positive")
        for name in (
            "cue_sample",
            "beat_anchor_sample",
            "start_sample",
            "end_sample",
            "duration_samples",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if int(self.end_sample) < int(self.start_sample):
            raise ValueError("end_sample must be >= start_sample")
        numeric = (
            self.rms_mean,
            self.log_rms_mean,
            self.spectral_flux_mean,
            self.attack_density_per_second,
            self.spectral_centroid_hz,
            self.high_frequency_ratio,
            self.bass_ratio_start,
            self.bass_ratio_end,
            self.bass_entry_delta,
            self.chroma_movement,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("trajectory descriptors must be finite")
        chroma = tuple(float(value) for value in self.chroma_mean)
        if len(chroma) != 12 or not all(math.isfinite(value) for value in chroma):
            raise ValueError("chroma_mean must contain 12 finite values")
        confidence = None if self.confidence is None else float(self.confidence)
        if confidence is not None and (
            not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError("confidence must be null or in [0, 1]")
        similarity = _as_finite(self.chroma_local_context_similarity)
        stability = _as_finite(self.chroma_tonal_stability_proxy)
        object.__setattr__(self, "side", str(self.side))
        object.__setattr__(self, "window_beats", int(self.window_beats))
        for name in (
            "cue_sample",
            "beat_anchor_sample",
            "start_sample",
            "end_sample",
            "duration_samples",
        ):
            object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(self, "chroma_mean", chroma)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "chroma_local_context_similarity", similarity)
        object.__setattr__(self, "chroma_tonal_stability_proxy", stability)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics or {}))
        object.__setattr__(
            self,
            "beat_feature_series",
            tuple(dict(point) for point in (self.beat_feature_series or ())),
        )


@dataclass(frozen=True)
class TrajectoryAnalysis:
    """All requested multiscale descriptors around one effective cue."""

    track_id: str
    sample_rate: int
    duration_samples: int
    requested_cue_sample: int
    cue_sample: int
    cue_shift_samples: int
    beat_anchor_sample: int | None
    beat_period_samples: float | None
    side: str
    windows: tuple[TrajectoryWindow, ...]
    backend: str
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.track_id:
            raise ValueError("track_id must be non-empty")
        if int(self.sample_rate) <= 0:
            raise ValueError("sample_rate must be positive")
        if int(self.duration_samples) < 0:
            raise ValueError("duration_samples must be non-negative")
        if self.side not in {"outgoing", "incoming"}:
            raise ValueError("side must be outgoing or incoming")
        windows = tuple(self.windows)
        if any(not isinstance(window, TrajectoryWindow) for window in windows):
            raise TypeError("windows must contain TrajectoryWindow objects")
        period = _as_finite(self.beat_period_samples)
        anchor = None if self.beat_anchor_sample is None else int(self.beat_anchor_sample)
        object.__setattr__(self, "sample_rate", int(self.sample_rate))
        object.__setattr__(self, "duration_samples", int(self.duration_samples))
        object.__setattr__(self, "requested_cue_sample", int(self.requested_cue_sample))
        object.__setattr__(self, "cue_sample", int(self.cue_sample))
        object.__setattr__(self, "cue_shift_samples", int(self.cue_shift_samples))
        object.__setattr__(self, "beat_anchor_sample", anchor)
        object.__setattr__(self, "beat_period_samples", period)
        object.__setattr__(self, "windows", windows)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def tempo_bpm(self) -> float | None:
        """Approximate beat-period conversion, not a calibrated tempo model."""

        if self.beat_period_samples is None or self.beat_period_samples <= 0:
            return None
        return float(60.0 * self.sample_rate / self.beat_period_samples)

    @property
    def canonical_sample_rate(self) -> int:
        """The timeline rate of the supplied track (caller-owned canonical input)."""

        return self.sample_rate


def _coerce_beats(
    beat_samples: Sequence[Any], duration_samples: int
) -> tuple[int, ...]:
    """Accept plain sample indices or provider event mappings.

    Provider confidence is intentionally ignored by the descriptor extractor:
    the resulting trajectory confidence remains ``None`` because no
    calibration study is bundled with this baseline.
    """

    values: list[int] = []
    for event in beat_samples:
        if isinstance(event, Mapping):
            value = event.get("sample")
        elif isinstance(event, (tuple, list)) and event:
            value = event[0]
        else:
            value = event
        number = _as_finite(value)
        if number is None:
            continue
        sample = int(round(number))
        if 0 <= sample < int(duration_samples):
            values.append(sample)
    return tuple(sorted(set(values)))


def _mono_samples(track: AudioTrack) -> np.ndarray:
    samples = np.asarray(track.samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] < 1:
        return np.empty(0, dtype=np.float64)
    mono = np.mean(samples, axis=1)
    return np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)


def _empty_frame_features() -> dict[str, Any]:
    return {
        "rms": np.zeros(0, dtype=np.float64),
        "flux": np.zeros(0, dtype=np.float64),
        "centroid": np.zeros(0, dtype=np.float64),
        "high_ratio": np.zeros(0, dtype=np.float64),
        "bass_ratio": np.zeros(0, dtype=np.float64),
        "chroma": np.zeros((0, 12), dtype=np.float64),
    }


def _frame_features(
    mono: np.ndarray,
    start_sample: int,
    end_sample: int,
    *,
    sample_rate: int,
    n_fft: int,
    hop: int,
) -> dict[str, Any]:
    """Compute all descriptors from one bounded FFT matrix for an interval."""

    start = max(0, int(start_sample))
    end = min(int(mono.size), int(end_sample))
    if end <= start or n_fft < 16:
        return _empty_frame_features()
    segment = np.asarray(mono[start:end], dtype=np.float64)
    if segment.size < n_fft:
        padded = np.zeros(n_fft, dtype=np.float64)
        padded[: segment.size] = segment
        frames = padded[None, :]
    else:
        frame_count = 1 + (segment.size - n_fft) // max(1, hop)
        starts = np.arange(frame_count, dtype=np.int64) * max(1, hop)
        offsets = np.arange(n_fft, dtype=np.int64)
        frames = segment[starts[:, None] + offsets[None, :]]
    window = np.hanning(n_fft).astype(np.float64)
    windowed = frames * window[None, :]
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    spectrum = np.abs(np.fft.rfft(windowed, axis=1))
    power = spectrum * spectrum
    log_spectrum = np.log1p(spectrum)
    positive = np.zeros_like(log_spectrum)
    if log_spectrum.shape[0] > 1:
        positive[1:] = np.maximum(log_spectrum[1:] - log_spectrum[:-1], 0.0)
    flux = np.mean(positive, axis=1)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
    power_total = np.sum(power, axis=1)
    safe_total = np.maximum(power_total, 1e-18)
    centroid = np.sum(power * frequencies[None, :], axis=1) / safe_total
    high_mask = frequencies >= _HIGH_FREQUENCY_HZ
    bass_mask = frequencies <= _BASS_FREQUENCY_HZ
    high_ratio = np.sum(power[:, high_mask], axis=1) / safe_total
    bass_ratio = np.sum(power[:, bass_mask], axis=1) / safe_total

    chroma = np.zeros((frames.shape[0], 12), dtype=np.float64)
    pitch_mask = (frequencies >= 55.0) & (frequencies <= min(5_000.0, sample_rate / 2.0))
    pitch_indices = np.flatnonzero(pitch_mask)
    if pitch_indices.size:
        midi = 69.0 + 12.0 * np.log2(np.maximum(frequencies[pitch_indices], 1e-12) / 440.0)
        pitch_classes = np.mod(np.rint(midi).astype(np.int64), 12)
        for index, pitch_class in zip(pitch_indices.tolist(), pitch_classes.tolist()):
            chroma[:, pitch_class] += power[:, index]
        chroma /= np.maximum(np.sum(chroma, axis=1, keepdims=True), 1e-18)
    return {
        "rms": np.nan_to_num(rms, nan=0.0, posinf=0.0, neginf=0.0),
        "flux": np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0),
        "centroid": np.nan_to_num(centroid, nan=0.0, posinf=0.0, neginf=0.0),
        "high_ratio": np.nan_to_num(high_ratio, nan=0.0, posinf=0.0, neginf=0.0),
        "bass_ratio": np.nan_to_num(bass_ratio, nan=0.0, posinf=0.0, neginf=0.0),
        "chroma": np.nan_to_num(chroma, nan=0.0, posinf=0.0, neginf=0.0),
    }


def _aggregate(features: Mapping[str, Any], duration_samples: int, sample_rate: int) -> dict[str, Any]:
    rms = np.asarray(features["rms"], dtype=np.float64)
    flux = np.asarray(features["flux"], dtype=np.float64)
    centroid = np.asarray(features["centroid"], dtype=np.float64)
    high_ratio = np.asarray(features["high_ratio"], dtype=np.float64)
    bass_ratio = np.asarray(features["bass_ratio"], dtype=np.float64)
    chroma = np.asarray(features["chroma"], dtype=np.float64)
    if rms.size == 0:
        return {
            "rms_mean": 0.0,
            "log_rms_mean": math.log(_MIN_AUDIO_RMS),
            "spectral_flux_mean": 0.0,
            "attack_density_per_second": 0.0,
            "spectral_centroid_hz": 0.0,
            "high_frequency_ratio": 0.0,
            "bass_ratio_start": 0.0,
            "bass_ratio_end": 0.0,
            "bass_entry_delta": 0.0,
            "chroma_mean": (0.0,) * 12,
            "chroma_movement": 0.0,
            "mean_rms": 0.0,
            "frame_count": 0,
        }
    rms_mean = float(np.mean(rms))
    flux_mean = float(np.mean(flux))
    if flux.size >= 3:
        threshold = max(float(np.percentile(flux, 75.0)), float(np.median(flux) + 0.5 * np.std(flux)), 1e-12)
        peaks = (flux[1:-1] > flux[:-2]) & (flux[1:-1] >= flux[2:]) & (flux[1:-1] > threshold)
        attacks = int(np.count_nonzero(peaks))
    else:
        attacks = 0
    seconds = max(float(duration_samples) / max(1, sample_rate), 1.0 / max(1, sample_rate))
    attack_density = float(attacks / seconds)
    chroma_mean_array = np.mean(chroma, axis=0) if chroma.size else np.zeros(12)
    chroma_total = float(np.sum(chroma_mean_array))
    if chroma_total > 1e-12:
        chroma_mean_array = chroma_mean_array / chroma_total
    if chroma.shape[0] > 1:
        movement = float(0.5 * np.mean(np.sum(np.abs(chroma[1:] - chroma[:-1]), axis=1)))
    else:
        movement = 0.0
    quarter = max(1, bass_ratio.size // 4)
    bass_start = float(np.mean(bass_ratio[:quarter]))
    bass_end = float(np.mean(bass_ratio[-quarter:]))
    return {
        "rms_mean": rms_mean,
        "log_rms_mean": float(math.log(max(rms_mean, _MIN_AUDIO_RMS))),
        "spectral_flux_mean": flux_mean,
        "attack_density_per_second": attack_density,
        "spectral_centroid_hz": float(np.mean(centroid)),
        "high_frequency_ratio": float(np.mean(high_ratio)),
        "bass_ratio_start": bass_start,
        "bass_ratio_end": bass_end,
        "bass_entry_delta": float(bass_start - bass_end),
        "chroma_mean": tuple(float(value) for value in chroma_mean_array),
        "chroma_movement": movement,
        "mean_rms": rms_mean,
        "frame_count": int(rms.size),
    }


def _window_bounds(
    beats: tuple[int, ...], anchor_index: int, window_beats: int, side: str
) -> tuple[int, int, int | None, int | None] | None:
    n = int(window_beats)
    if side == "outgoing":
        local_start = anchor_index - n
        local_end = anchor_index
        context_start = anchor_index - 2 * n
        context_end = anchor_index - n
    else:
        local_start = anchor_index
        local_end = anchor_index + n
        context_start = anchor_index + n
        context_end = anchor_index + 2 * n
    # Local descriptors remain valid at track edges.  Adjacent context is an
    # optional second interval used only for the chroma proxy; never pad or
    # borrow an unrelated interval when it is unavailable.
    if local_start < 0 or local_end >= len(beats):
        return None
    if context_start < 0 or context_end >= len(beats):
        return beats[local_start], beats[local_end], None, None
    return beats[local_start], beats[local_end], beats[context_start], beats[context_end]


def _beat_feature_series(
    mono: np.ndarray,
    beats: tuple[int, ...],
    anchor_index: int,
    window_beats: int,
    side: str,
    *,
    sample_rate: int,
    n_fft: int,
    hop: int,
) -> tuple[dict[str, Any], ...]:
    """Return chronological descriptors for each beat in a local window."""

    if side == "outgoing":
        first = anchor_index - int(window_beats)
        indices = range(first, anchor_index)
    else:
        first = anchor_index
        indices = range(first, anchor_index + int(window_beats))
    points: list[dict[str, Any]] = []
    for relative_index, index in enumerate(indices):
        start = beats[index]
        end = beats[index + 1]
        values = _aggregate(
            _frame_features(
                mono,
                start,
                end,
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop=hop,
            ),
            end - start,
            sample_rate,
        )
        points.append(
            {
                "beat_index": relative_index,
                "start_sample": start,
                "end_sample": end,
                "duration_samples": end - start,
                "rms_mean": values["rms_mean"],
                "log_rms_mean": values["log_rms_mean"],
                "spectral_flux_mean": values["spectral_flux_mean"],
                "attack_density_per_second": values["attack_density_per_second"],
                "spectral_centroid_hz": values["spectral_centroid_hz"],
                "high_frequency_ratio": values["high_frequency_ratio"],
                "bass_ratio": float((values["bass_ratio_start"] + values["bass_ratio_end"]) / 2.0),
                "chroma_mean": list(values["chroma_mean"]),
                "chroma_movement": values["chroma_movement"],
                "frame_count": values["frame_count"],
            }
        )
    return tuple(points)


def extract_trajectory(
    track: AudioTrack,
    cue_sample: int,
    beat_samples: Sequence[Any],
    *,
    side: str,
    windows_beats: Sequence[int] = DEFAULT_WINDOWS_BEATS,
    cue_shift_samples: int = 0,
) -> TrajectoryAnalysis:
    """Extract one-sided 8/16/32-beat trajectory descriptors.

    ``cue_sample`` is the requested/provider cue and ``cue_shift_samples`` is
    an explicit renderer or safety adjustment.  The effective cue is exposed
    as ``TrajectoryAnalysis.cue_sample``.  A beat anchor is selected by the
    nearest supplied beat; its sample delta is retained in diagnostics.
    Missing beat events produce no windows rather than a synthetic tempo or
    beat grid.  Every output confidence is ``None`` because this baseline has
    no calibration data.
    """

    if side not in {"outgoing", "incoming"}:
        raise ValueError("side must be outgoing or incoming")
    requested = int(cue_sample)
    shift = int(cue_shift_samples)
    effective = requested + shift
    effective = max(0, min(int(track.duration_samples), effective))
    beats = _coerce_beats(beat_samples, track.duration_samples)
    beat_period = None
    if len(beats) >= 2:
        beat_period = float(np.median(np.diff(np.asarray(beats, dtype=np.float64))))
        if beat_period <= 0 or not math.isfinite(beat_period):
            beat_period = None
    anchor_index: int | None = None
    anchor_sample: int | None = None
    anchor_delta: int | None = None
    if beats:
        anchor_index = min(range(len(beats)), key=lambda index: abs(beats[index] - effective))
        anchor_sample = beats[anchor_index]
        anchor_delta = int(effective - anchor_sample)
    mono = _mono_samples(track)
    n_fft = _analysis_frame_length(int(track.sample_rate), int(track.duration_samples))
    hop = max(1, n_fft // _HOP_FRACTION) if n_fft else 0
    windows: list[TrajectoryWindow] = []
    diagnostics: dict[str, Any] = {
        "trajectory_version": TRAJECTORY_VERSION,
        "trajectory_config_hash": TRAJECTORY_CONFIG_HASH,
        "status": "ok",
        "confidence": None,
        "confidence_status": "unknown_unvalidated",
        "beat_provider_confidence_used": False,
        "pitch_contour": {"status": "unknown", "reason": "no physical pitch provider configured"},
        "tonal_stability_semantics": "chroma similarity/movement proxy; not tonal tension",
        "energy_semantics": "log RMS and RMS descriptors; no suspense/resolution inference",
        "requested_windows_beats": tuple(int(value) for value in windows_beats),
        "beat_count": len(beats),
        "beat_period_samples": beat_period,
        "beat_anchor_delta_samples": anchor_delta,
        "sample_rate_source": int(track.sample_rate),
        "canonical_sample_rate": int(track.sample_rate),
        "canonical_sample_rate_assumption": "caller supplied canonical timeline; no resampling here",
    }
    if not beats:
        diagnostics.update({"status": "missing_beats", "candidate_status": "unreliable_without_provider_beats"})
    elif anchor_index is None:
        diagnostics.update({"status": "missing_anchor", "candidate_status": "unreliable_without_anchor"})
    else:
        for value in windows_beats:
            try:
                window_beats = int(value)
            except (TypeError, ValueError):
                continue
            if window_beats <= 0:
                continue
            bounds = _window_bounds(beats, anchor_index, window_beats, side)
            if bounds is None:
                diagnostics.setdefault("insufficient_local_windows", []).append(window_beats)
                continue
            start, end, context_start, context_end = bounds
            local_features = _frame_features(
                mono, start, end, sample_rate=int(track.sample_rate), n_fft=n_fft, hop=hop
            )
            values = _aggregate(local_features, end - start, int(track.sample_rate))
            if context_start is None or context_end is None:
                similarity = None
                diagnostics.setdefault("missing_adjacent_context_windows", []).append(window_beats)
                context_values = None
            else:
                context_features = _frame_features(
                    mono,
                    context_start,
                    context_end,
                    sample_rate=int(track.sample_rate),
                    n_fft=n_fft,
                    hop=hop,
                )
                context_values = _aggregate(context_features, context_end - context_start, int(track.sample_rate))
                similarity = _cosine(values["chroma_mean"], context_values["chroma_mean"])
            series = _beat_feature_series(
                mono,
                beats,
                anchor_index,
                window_beats,
                side,
                sample_rate=int(track.sample_rate),
                n_fft=n_fft,
                hop=hop,
            )
            windows.append(
                TrajectoryWindow(
                    side=side,
                    window_beats=window_beats,
                    cue_sample=effective,
                    beat_anchor_sample=anchor_sample if anchor_sample is not None else effective,
                    start_sample=start,
                    end_sample=end,
                    duration_samples=end - start,
                    rms_mean=values["rms_mean"],
                    log_rms_mean=values["log_rms_mean"],
                    spectral_flux_mean=values["spectral_flux_mean"],
                    attack_density_per_second=values["attack_density_per_second"],
                    spectral_centroid_hz=values["spectral_centroid_hz"],
                    high_frequency_ratio=values["high_frequency_ratio"],
                    bass_ratio_start=values["bass_ratio_start"],
                    bass_ratio_end=values["bass_ratio_end"],
                    bass_entry_delta=values["bass_entry_delta"],
                    chroma_mean=values["chroma_mean"],
                    chroma_movement=values["chroma_movement"],
                    chroma_local_context_similarity=similarity,
                    chroma_tonal_stability_proxy=similarity,
                    confidence=None,
                    diagnostics={
                        "context_start_sample": context_start,
                        "context_end_sample": context_end,
                        "adjacent_context_available": context_start is not None and context_end is not None,
                        "adjacent_context_required_beats": 2 * window_beats,
                        "bass_entry_definition": "start_ratio_minus_end_ratio",
                        "frame_length": n_fft,
                        "hop_samples": hop,
                    },
                    beat_feature_series=series,
                )
            )
    if windows and max(window.rms_mean for window in windows) <= _MIN_AUDIO_RMS:
        diagnostics.update({"status": "silent_or_near_silent", "candidate_status": "unreliable_silence"})
    elif not windows and beats:
        diagnostics.setdefault("candidate_status", "insufficient_context")
    diagnostics["window_count"] = len(windows)
    return TrajectoryAnalysis(
        track_id=track.track_id,
        sample_rate=int(track.sample_rate),
        duration_samples=int(track.duration_samples),
        requested_cue_sample=requested,
        cue_sample=effective,
        cue_shift_samples=shift,
        beat_anchor_sample=anchor_sample,
        beat_period_samples=beat_period,
        side=side,
        windows=tuple(windows),
        backend=TRAJECTORY_VERSION,
        diagnostics=diagnostics,
    )


def trajectory_to_dict(result: TrajectoryAnalysis) -> dict[str, Any]:
    """Return a stable JSON-ready representation for audit tools."""

    return {
        "track_id": result.track_id,
        "sample_rate": result.sample_rate,
        "canonical_sample_rate": result.canonical_sample_rate,
        "duration_samples": result.duration_samples,
        "requested_cue_sample": result.requested_cue_sample,
        "cue_sample": result.cue_sample,
        "cue_shift_samples": result.cue_shift_samples,
        "beat_anchor_sample": result.beat_anchor_sample,
        "beat_period_samples": result.beat_period_samples,
        "tempo_bpm": result.tempo_bpm,
        "side": result.side,
        "backend": result.backend,
        "trajectory_config_hash": TRAJECTORY_CONFIG_HASH,
        "diagnostics": dict(result.diagnostics),
        "windows": [
            {
                "side": window.side,
                "window_beats": window.window_beats,
                "cue_sample": window.cue_sample,
                "beat_anchor_sample": window.beat_anchor_sample,
                "start_sample": window.start_sample,
                "end_sample": window.end_sample,
                "duration_samples": window.duration_samples,
                "rms_mean": window.rms_mean,
                "log_rms_mean": window.log_rms_mean,
                "spectral_flux_mean": window.spectral_flux_mean,
                "attack_density_per_second": window.attack_density_per_second,
                "spectral_centroid_hz": window.spectral_centroid_hz,
                "high_frequency_ratio": window.high_frequency_ratio,
                "bass_ratio_start": window.bass_ratio_start,
                "bass_ratio_end": window.bass_ratio_end,
                "bass_entry_delta": window.bass_entry_delta,
                "chroma_mean": list(window.chroma_mean),
                "chroma_movement": window.chroma_movement,
                "chroma_local_context_similarity": window.chroma_local_context_similarity,
                "chroma_tonal_stability_proxy": window.chroma_tonal_stability_proxy,
                "confidence": window.confidence,
                "diagnostics": dict(window.diagnostics or {}),
                "beat_feature_series": [
                    {
                        **dict(point),
                        "chroma_mean": list(point.get("chroma_mean", ())),
                    }
                    for point in window.beat_feature_series
                ],
            }
            for window in result.windows
        ],
    }


__all__ = [
    "DEFAULT_WINDOWS_BEATS",
    "TRAJECTORY_CONFIG_HASH",
    "TRAJECTORY_VERSION",
    "TrajectoryAnalysis",
    "TrajectoryWindow",
    "extract_trajectory",
    "trajectory_to_dict",
]
