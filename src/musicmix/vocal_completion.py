"""Measured vocal-phrase descriptors for outgoing completion / incoming pickup.

The module is deliberately *not* a phrase classifier.  It extracts independent,
auditable channels around a candidate cue:

* relative-energy voice activity and voice onset/offset times,
* spectral-flux attack/syllable-nucleus timing,
* YIN-style pitch availability, terminal slope and onset movement,
* terminal lengthening of the last active run,
* an uncalibrated breath-likeness index inside the unvoiced gap,
* optional ASR word/segment timing and punctuation evidence.

An ASR segment is never relabelled as a prosodic phrase.  No channel is called
"closure"; the single composite number is an explicitly uncalibrated descriptor
heuristic for ranking candidates, not a probability.

All public sample coordinates are *canonical 48 kHz source samples*.  The
canonical start of an excerpt (or of a full stem) must be supplied explicitly so
that a local array index cannot silently drift against the source clock.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.signal import find_peaks, get_window, resample_poly, stft

from .audio import AudioTrack, load_audio


SCHEMA_VERSION = "musicmix.vocal_completion.1"
CANONICAL_SAMPLE_RATE = 48_000
FRAME_LENGTH = 2_048
FRAME_HOP = 480  # 10 ms at 48 kHz
VAD_GAP_CLOSE_SECONDS = 0.12
VAD_MIN_RUN_SECONDS = 0.04
ONSET_MIN_DISTANCE_SECONDS = 0.09
PITCH_MIN_HZ = 70.0
PITCH_MAX_HZ = 500.0
YIN_THRESHOLD = 0.10
YIN_VOICING_FLOOR = 0.45
YIN_CHUNK_FRAMES = 256
FEATURE_CHUNK_FRAMES = 512
PITCH_SMOOTH_FRAMES = 5
DEFAULT_LEFT_SECONDS = 3.0
DEFAULT_RIGHT_SECONDS = 2.0
BREATH_MIN_GAP_SECONDS = 0.08

# These are engineering priors, not fitted weights.  They are written into the
# JSON so that a downstream consumer can replace or reject the ranking.
COMPLETION_WEIGHTS = {
    "inactive_at_cue": 0.35,
    "gap_length": 0.20,
    "terminal_lengthening": 0.15,
    "breath_index": 0.15,
    "terminal_pitch_available": 0.05,
    "asr_closed_punctuation": 0.10,
}
PICKUP_WEIGHTS = {
    "onset_proximity": 0.35,
    "onset_strength": 0.20,
    "active_after_cue": 0.20,
    "onset_pitch_movement": 0.15,
    "asr_word_onset": 0.10,
}


class VocalCompletionError(ValueError):
    """Base class for invalid vocal-completion inputs."""


class VocalCompletionClockError(VocalCompletionError):
    """Raised when canonical and local sample clocks cannot be reconciled."""


class VocalCompletionInputError(VocalCompletionError):
    """Raised for non-finite, malformed, or unsupported audio input."""


@dataclass(frozen=True)
class SourceClock:
    file: str
    source_start_sample: int
    sample_rate: int
    duration_seconds: float
    sha256: str | None = None
    source_sha256: str | None = None

    @property
    def duration_samples(self) -> int:
        return int(round(self.duration_seconds * self.sample_rate))


@dataclass
class VocalFeatureTrack:
    """Frame-level descriptors computed once for a whole stem/excerpt."""

    sample_rate: int
    canonical_start_sample: int
    duration_samples: int
    frame_centers: np.ndarray
    rms_db: np.ndarray
    active: np.ndarray
    noise_db: float
    speech_db: float
    threshold_db: float
    snr_db: float
    active_runs: list[tuple[int, int]]
    onsets: np.ndarray
    onset_strength: np.ndarray
    onset_strength_percentile: float
    f0_hz: np.ndarray
    voicing: np.ndarray
    zero_crossing_rate: np.ndarray
    spectral_flatness: np.ndarray
    high_frequency_ratio: np.ndarray
    warnings: list[str]
    pretrained_vad: dict[str, Any] | None = None

    @property
    def canonical_end_sample(self) -> int:
        return self.canonical_start_sample + self.duration_samples

    def local_to_canonical(self, local_sample: float) -> int:
        return int(self.canonical_start_sample + round(local_sample))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_clocks(path: str | os.PathLike[str]) -> list[SourceClock]:
    """Load and validate the staged-file source clock manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise VocalCompletionInputError("source clock manifest must be a JSON list")
    clocks: list[SourceClock] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise VocalCompletionInputError("source clock entries must be objects")
        file_name = str(item.get("file", "")).strip()
        if not file_name:
            raise VocalCompletionInputError("source clock entry has no file")
        start = int(item.get("source_start_sample", -1))
        rate = int(item.get("sample_rate", CANONICAL_SAMPLE_RATE))
        duration = float(item.get("duration_seconds", 0.0))
        if start < 0 or rate <= 0 or not math.isfinite(duration) or duration <= 0:
            raise VocalCompletionInputError(f"invalid source clock for {file_name}")
        clocks.append(
            SourceClock(
                file=file_name,
                source_start_sample=start,
                sample_rate=rate,
                duration_seconds=duration,
                sha256=item.get("sha256"),
                source_sha256=item.get("source_sha256"),
            )
        )
    return clocks


def clock_for_file(clocks: Sequence[SourceClock], name: str) -> SourceClock:
    matches = [clock for clock in clocks if clock.file == name or Path(clock.file).name == name]
    if not matches:
        raise VocalCompletionClockError(f"no source clock entry for {name!r}")
    if len(matches) > 1:
        raise VocalCompletionClockError(f"ambiguous source clock entry for {name!r}")
    return matches[0]


def verify_staged_file(path: str | os.PathLike[str], clock: SourceClock) -> dict[str, Any]:
    """Hash the staged file and compare it with the manifest when present."""

    actual = _sha256_file(Path(path))
    expected = clock.sha256
    return {
        "path": str(path),
        "sha256": actual,
        "manifest_sha256": expected,
        "matches_manifest": True if expected is None else actual == expected,
        "source_sha256": clock.source_sha256,
    }


def _as_mono_float32(samples: Any, sample_rate: int) -> tuple[np.ndarray, int]:
    if int(sample_rate) != CANONICAL_SAMPLE_RATE:
        raise VocalCompletionClockError(
            f"canonical analysis requires {CANONICAL_SAMPLE_RATE} Hz, got {sample_rate}; "
            "resample explicitly and preserve the clock mapping"
        )
    array = np.asarray(samples)
    if array.ndim == 2:
        if array.shape[1] < 1:
            raise VocalCompletionInputError("samples must contain at least one channel")
        array = array.mean(axis=1)
    elif array.ndim != 1:
        raise VocalCompletionInputError("samples must be 1-D or (frames, channels)")
    if not np.issubdtype(array.dtype, np.number):
        raise VocalCompletionInputError("samples must be numeric")
    if not np.all(np.isfinite(array)):
        raise VocalCompletionInputError("samples contain NaN or infinite values")
    if array.size == 0:
        raise VocalCompletionInputError("samples are empty")
    return np.ascontiguousarray(array, dtype=np.float32), array.shape[0]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2]) if end > start]


def _close_short_gaps(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged: list[list[int]] = [[runs[0][0], runs[0][1]]]
    for start, end in runs[1:]:
        if start - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _median_filter_1d(values: np.ndarray, width: int) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    half = max(1, width // 2)
    result = values.copy()
    for index in range(values.size):
        lo = max(0, index - half)
        hi = min(values.size, index + half + 1)
        window = values[lo:hi]
        window = window[np.isfinite(window)]
        if window.size:
            result[index] = float(np.median(window))
    return result


def _frame_matrix(mono: np.ndarray) -> np.ndarray:
    if mono.size < FRAME_LENGTH:
        mono = np.pad(mono, (0, FRAME_LENGTH - mono.size))
    view = np.lib.stride_tricks.sliding_window_view(mono, FRAME_LENGTH)
    return view[::FRAME_HOP]


def _frame_count(mono: np.ndarray) -> int:
    if mono.size < FRAME_LENGTH:
        return 1
    return 1 + (mono.size - FRAME_LENGTH) // FRAME_HOP


def _frame_block(mono: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Read only frames ``start:stop`` as a strided view, never the whole stem."""

    first_sample = start * FRAME_HOP
    last_sample = (stop - 1) * FRAME_HOP + FRAME_LENGTH
    segment = mono[first_sample:last_sample]
    if segment.size < FRAME_LENGTH:
        segment = np.pad(segment, (0, FRAME_LENGTH - segment.size))
    view = np.lib.stride_tricks.sliding_window_view(segment, FRAME_LENGTH)
    return view[::FRAME_HOP]


def _frame_descriptors(
    mono: np.ndarray,
    sample_rate: int,
    *,
    chunk_frames: int = FEATURE_CHUNK_FRAMES,
) -> dict[str, np.ndarray]:
    """Per-frame scalar descriptors with bounded memory.

    Frames are visited in chunks; only scalar arrays are retained.  The
    periodic-Hann magnitude of the previous normalised frame is carried across
    chunk boundaries so spectral flux is identical to a whole-song matrix.
    """

    if chunk_frames < 1:
        raise ValueError("chunk_frames must be positive")
    frame_count = _frame_count(mono)
    rms = np.empty(frame_count, dtype=np.float64)
    zero_crossing = np.empty(frame_count, dtype=np.float64)
    flatness = np.empty(frame_count, dtype=np.float64)
    hf_ratio = np.empty(frame_count, dtype=np.float64)
    flux = np.zeros(frame_count, dtype=np.float64)
    f0 = np.full(frame_count, np.nan, dtype=np.float64)
    voicing = np.zeros(frame_count, dtype=np.float64)

    window = get_window("hann", FRAME_LENGTH, fftbins=True).astype(np.float32)
    frequencies = np.fft.rfftfreq(FRAME_LENGTH, d=1.0 / float(sample_rate))
    high = frequencies >= 2000.0
    previous_magnitude: np.ndarray | None = None

    for chunk_start in range(0, frame_count, chunk_frames):
        chunk_end = min(frame_count, chunk_start + chunk_frames)
        block = _frame_block(mono, chunk_start, chunk_end)
        count = block.shape[0]

        numeric = block.astype(np.float64)
        rms[chunk_start:chunk_end] = np.sqrt(
            np.mean(numeric * numeric, axis=1) + 1e-18
        )
        zero_crossing[chunk_start:chunk_end] = np.mean(
            np.abs(np.diff(np.signbit(block), axis=1)), axis=1
        )

        first_sample = chunk_start * FRAME_HOP
        last_sample = (chunk_end - 1) * FRAME_HOP + FRAME_LENGTH
        segment = mono[first_sample:last_sample]
        if segment.size < FRAME_LENGTH:
            segment = np.pad(segment, (0, FRAME_LENGTH - segment.size))
        # One bounded periodic-Hann STFT block; no full-song spectra are kept.
        magnitude = np.abs(stft(
            segment,
            fs=sample_rate,
            window=window,
            nperseg=FRAME_LENGTH,
            noverlap=FRAME_LENGTH - FRAME_HOP,
            boundary=None,
            padded=False,
        )[2]).T
        if magnitude.shape[0] != count:
            raise VocalCompletionInputError("chunked STFT frame count mismatch")
        power = magnitude.astype(np.float64) ** 2 + 1e-20
        flatness[chunk_start:chunk_end] = (
            np.exp(np.mean(np.log(power[:, 1:]), axis=1))
            / (np.mean(power[:, 1:], axis=1) + 1e-20)
        )
        hf_ratio[chunk_start:chunk_end] = (
            power[:, high].sum(axis=1) / (power.sum(axis=1) + 1e-20)
        )

        if previous_magnitude is not None:
            attack = np.maximum(0.0, magnitude[0] - previous_magnitude)
            flux[chunk_start] = (
                attack.sum() / (magnitude[0].sum() + 1e-9)
            )
        if count > 1:
            attack = np.maximum(0.0, magnitude[1:] - magnitude[:-1])
            flux[chunk_start + 1:chunk_end] = (
                attack.sum(axis=1) / (magnitude[1:].sum(axis=1) + 1e-9)
            )
        previous_magnitude = magnitude[-1].copy()

        chunk_f0, chunk_voicing = _yin_pitch(block, sample_rate)
        f0[chunk_start:chunk_end] = chunk_f0
        voicing[chunk_start:chunk_end] = chunk_voicing

    return {
        "rms": rms,
        "zero_crossing": zero_crossing,
        "flatness": flatness,
        "hf_ratio": hf_ratio,
        "flux": flux,
        "f0": f0,
        "voicing": voicing,
    }


def _yin_pitch(frames: np.ndarray, sample_rate: int,
               chunk_frames: int = YIN_CHUNK_FRAMES) -> tuple[np.ndarray, np.ndarray]:
    """Faithful chunked YIN CMND pitch estimator.

    The cumulative mean normalised difference is accumulated over lags
    ``1..max_lag`` and divided by the actual lag number, then the search is
    restricted to ``min_lag..max_lag``.  Frames are processed in bounded chunks
    so a full-length stem never allocates the whole frame-by-lag matrix.
    Frames are independent, so chunk size does not change the per-frame result.
    """

    frame_count, frame_length = frames.shape
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be positive")
    nfft = 1 << int(math.ceil(math.log2(2 * frame_length)))
    min_lag = max(2, int(math.floor(sample_rate / PITCH_MAX_HZ)))
    max_lag = min(frame_length - 2, int(math.ceil(sample_rate / PITCH_MIN_HZ)))
    lags = np.arange(1, max_lag + 1, dtype=np.float64)
    search_start = min_lag - 1

    f0 = np.full(frame_count, np.nan, dtype=np.float64)
    voicing = np.zeros(frame_count, dtype=np.float64)
    for chunk_start in range(0, frame_count, chunk_frames):
        chunk_end = min(frame_count, chunk_start + chunk_frames)
        block = frames[chunk_start:chunk_end]
        count = block.shape[0]
        centred = block - block.mean(axis=1, keepdims=True)
        # No analysis window here: the difference function and the FFT-based
        # autocorrelation must use exactly the same samples.
        spectrum = np.fft.rfft(centred, n=nfft, axis=1)
        autocorrelation = np.fft.irfft(
            np.abs(spectrum) ** 2, n=nfft, axis=1
        )[:, :max_lag + 1]

        squared = centred * centred
        prefix = np.concatenate(
            (np.zeros((count, 1)), np.cumsum(squared, axis=1)), axis=1
        )
        total = prefix[:, -1]
        # Exact energies of the two overlapped segments for lags 1..max_lag.
        first = prefix[:, frame_length - lags.astype(np.int64)]
        second = total[:, None] - prefix[:, lags.astype(np.int64)]
        difference = first + second - 2.0 * autocorrelation[:, 1:max_lag + 1]
        difference = np.maximum(difference, 1e-12)

        cumulative = np.cumsum(difference, axis=1)
        divisor = cumulative / lags[None, :]
        cmnd = difference / np.maximum(divisor, 1e-12)

        for local_index in range(count):
            row = cmnd[local_index]
            candidates = np.flatnonzero(row[search_start:] < YIN_THRESHOLD)
            lag_index = -1
            if candidates.size:
                # First local minimum below the threshold, YIN's standard rule.
                for candidate in candidates:
                    actual = int(candidate) + search_start
                    left_ok = actual == search_start or row[actual] <= row[actual - 1]
                    right_ok = actual == row.size - 1 or row[actual] <= row[actual + 1]
                    if left_ok and right_ok:
                        lag_index = actual
                        break
                if lag_index < 0:
                    lag_index = int(candidates[0]) + search_start
            else:
                lag_index = int(np.argmin(row[search_start:])) + search_start
            confidence = float(np.clip(1.0 - row[lag_index], 0.0, 1.0))
            voicing[chunk_start + local_index] = confidence
            if confidence < YIN_VOICING_FLOOR:
                continue
            lag = float(lag_index + 1)
            if search_start <= lag_index < row.size - 1:
                left, centre, right = row[lag_index - 1], row[lag_index], row[lag_index + 1]
                denominator = left - 2.0 * centre + right
                if abs(denominator) > 1e-12:
                    lag += 0.5 * (left - right) / denominator
            if lag > 0:
                f0[chunk_start + local_index] = sample_rate / lag
    return f0, voicing


def _correct_octave_jumps(f0: np.ndarray) -> np.ndarray:
    """Suppress isolated YIN subharmonic/octave outliers with a running median.

    This is a deterministic continuity heuristic.  It is recorded in the JSON
    method string and should not be read as octave-error-free pitch tracking.
    """

    corrected = f0.copy()
    history: list[float] = []
    for index, value in enumerate(f0):
        if not math.isfinite(value):
            continue
        adjusted = float(value)
        if len(history) >= 3:
            reference = float(np.median(history[-9:]))
            if reference > 0:
                ratio = adjusted / reference
                if ratio < 0.62 and 2.0 * adjusted <= PITCH_MAX_HZ * 1.25:
                    adjusted *= 2.0
                elif ratio > 1.75 and adjusted / 2.0 >= PITCH_MIN_HZ * 0.9:
                    adjusted /= 2.0
        corrected[index] = adjusted
        history.append(adjusted)
    return corrected


_SILERO_MODEL_CACHE: dict[str, Any] = {}


def run_silero_vad(
    samples: Any,
    sample_rate: int = CANONICAL_SAMPLE_RATE,
    *,
    canonical_start_sample: int = 0,
) -> dict[str, Any]:
    """Optional official Silero VAD (MIT) as an independent speech channel.

    The package and its bundled JIT weights are installed under ``runtime/``
    with ``pip --target``.  This function never downloads weights; when the
    package or Torch is absent it returns an explicit unavailable record.
    """

    try:
        import torch  # type: ignore
        from silero_vad import get_speech_timestamps, load_silero_vad  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency path
        return {
            "available": False,
            "reason": f"silero_vad_or_torch_unavailable: {type(exc).__name__}",
            "provider": "silero-vad",
        }
    try:
        import silero_vad as silero_module  # type: ignore
        package_dir = Path(silero_module.__file__).resolve().parent
        model_path = package_dir / "data" / "silero_vad.jit"
        model_hash = _sha256_file(model_path)
        version = getattr(silero_module, "__version__", None)
        torch.set_num_threads(1)
        model = _SILERO_MODEL_CACHE.get(str(model_path))
        if model is None:
            model = load_silero_vad(onnx=False)
            _SILERO_MODEL_CACHE[str(model_path)] = model
        mono, _ = _as_mono_float32(samples, sample_rate)
        resampled = resample_poly(mono, 1, 3).astype(np.float32)
        timestamps = get_speech_timestamps(
            torch.from_numpy(resampled),
            model,
            sampling_rate=16_000,
            return_seconds=True,
        )
        intervals = []
        for item in timestamps:
            start_local = int(round(float(item["start"]) * sample_rate))
            end_local = int(round(float(item["end"]) * sample_rate))
            intervals.append({
                "start_sample": int(canonical_start_sample + start_local),
                "end_sample": int(canonical_start_sample + end_local),
                "start_seconds_in_excerpt": start_local / sample_rate,
                "end_seconds_in_excerpt": end_local / sample_rate,
            })
        return {
            "available": True,
            "reason": None,
            "provider": "silero-vad",
            "version": version,
            "license": "MIT",
            "model_file": str(model_path),
            "model_sha256": model_hash,
            "input_internal_sample_rate": 16_000,
            "resampling": "scipy.signal.resample_poly 48k->16k (3:1)",
            "threshold": 0.5,
            "min_silence_duration_ms": 100,
            "speech_pad_ms": 30,
            "intervals": intervals,
            "segments_are_vad_not_phrases": True,
            "trained_on": "speech/non-speech VAD; not rap phrase boundaries",
        }
    except Exception as exc:  # pragma: no cover - defensive optional path
        return {
            "available": False,
            "reason": f"silero_vad_inference_failed: {type(exc).__name__}: {exc}",
            "provider": "silero-vad",
        }


def extract_vocal_features(
    samples: Any,
    sample_rate: int = CANONICAL_SAMPLE_RATE,
    *,
    canonical_start_sample: int = 0,
    use_pretrained_vad: bool = False,
) -> VocalFeatureTrack:
    """Compute frame descriptors once for a stem, then query many cues."""

    mono, duration_samples = _as_mono_float32(samples, sample_rate)
    if canonical_start_sample < 0:
        raise VocalCompletionClockError("canonical_start_sample must be non-negative")

    frame_count = _frame_count(mono)
    frame_centres = np.arange(frame_count, dtype=np.int64) * FRAME_HOP + FRAME_LENGTH // 2
    descriptors = _frame_descriptors(
        mono, sample_rate, chunk_frames=FEATURE_CHUNK_FRAMES
    )
    rms = descriptors["rms"]
    rms_db = 20.0 * np.log10(rms + 1e-9)

    noise_db = float(np.percentile(rms_db, 10))
    speech_db = float(np.percentile(rms_db, 90))
    snr_db = max(0.0, speech_db - noise_db)
    warnings: list[str] = []
    if snr_db < 3.0 and float(np.max(rms_db)) > -80.0:
        # With almost no relative dynamic range the percentile threshold has no
        # silence contrast.  Mark the energy as active and say so, instead of
        # pretending a constant tone is silence.
        threshold_db = noise_db
        raw_active = np.ones(frame_count, dtype=bool)
        warnings.append("insufficient_relative_dynamic_range_all_frames_active")
    else:
        threshold_db = noise_db + max(4.0, 0.25 * snr_db)
        raw_active = rms_db > threshold_db

    max_gap_frames = int(round(VAD_GAP_CLOSE_SECONDS * sample_rate / FRAME_HOP))
    min_run_frames = int(round(VAD_MIN_RUN_SECONDS * sample_rate / FRAME_HOP))
    active_runs_frames = _close_short_gaps(_runs(raw_active), max_gap_frames)
    active_runs_frames = [(start, end) for start, end in active_runs_frames if end - start >= min_run_frames]
    active = np.zeros(frame_count, dtype=bool)
    for start, end in active_runs_frames:
        active[start:end] = True

    flux = np.convolve(
        descriptors["flux"], np.ones(3, dtype=np.float64) / 3.0, mode="same"
    )
    flux_percentile = float(np.percentile(flux, 70))
    flux_height = max(flux_percentile, float(np.median(flux) + 2.0 * np.median(np.abs(flux - np.median(flux)))))
    peaks, properties = find_peaks(
        flux,
        distance=max(1, int(round(ONSET_MIN_DISTANCE_SECONDS * sample_rate / FRAME_HOP))),
        height=flux_height,
    )
    onset_mask = np.zeros(frame_count, dtype=bool)
    if peaks.size:
        valid = peaks < active.shape[0]
        peaks = peaks[valid]
        onset_mask[np.clip(peaks, 0, frame_count - 1)] = True
        # An attack must fall on or next to an active frame.
        for index in peaks:
            lo, hi = max(0, index - 1), min(frame_count, index + 2)
            if not active[lo:hi].any():
                onset_mask[index] = False
    onset_indices = np.flatnonzero(onset_mask)
    onsets = frame_centres[onset_indices] if onset_indices.size else np.empty(0, dtype=np.int64)
    onset_strengths = flux[onset_indices] if onset_indices.size else np.empty(0, dtype=np.float64)

    f0 = descriptors["f0"]
    voicing = descriptors["voicing"]
    f0 = _median_filter_1d(f0, PITCH_SMOOTH_FRAMES)
    f0 = _correct_octave_jumps(f0)
    # The smoothing must not invent pitch in unvoiced frames.
    f0[voicing < YIN_VOICING_FLOOR] = np.nan

    zero_crossing = descriptors["zero_crossing"]
    flatness = descriptors["flatness"]
    hf_ratio = descriptors["hf_ratio"]

    active_runs = [
        (int(frame_centres[start] - FRAME_LENGTH // 2), int(frame_centres[end - 1] + FRAME_LENGTH // 2))
        for start, end in active_runs_frames
    ]
    active_runs = [(max(0, start), min(duration_samples, end)) for start, end in active_runs]

    if snr_db < 6.0:
        warnings.append("low_relative_snr_activity_threshold_uncertain")
    if frame_count < 20:
        warnings.append("very_short_analysis_context")
    if f0.size and np.all(~np.isfinite(f0)):
        warnings.append("no_pitch_contour_detected")
    pretrained_vad = None
    if use_pretrained_vad:
        pretrained_vad = run_silero_vad(
            mono, sample_rate, canonical_start_sample=canonical_start_sample
        )
        if not pretrained_vad.get("available"):
            warnings.append("pretrained_vad_unavailable")

    return VocalFeatureTrack(
        sample_rate=sample_rate,
        canonical_start_sample=int(canonical_start_sample),
        duration_samples=duration_samples,
        frame_centers=frame_centres,
        rms_db=rms_db,
        active=active,
        noise_db=noise_db,
        speech_db=speech_db,
        threshold_db=float(threshold_db),
        snr_db=float(snr_db),
        active_runs=active_runs,
        onsets=onsets,
        onset_strength=onset_strengths,
        onset_strength_percentile=flux_percentile,
        f0_hz=f0,
        voicing=voicing,
        zero_crossing_rate=zero_crossing,
        spectral_flatness=flatness,
        high_frequency_ratio=hf_ratio,
        warnings=warnings,
        pretrained_vad=pretrained_vad,
    )


def _window_bounds(local_cue: int, duration: int, left_seconds: float, right_seconds: float) -> tuple[int, int]:
    left = max(0, local_cue - int(round(left_seconds * CANONICAL_SAMPLE_RATE)))
    right = min(duration, local_cue + int(round(right_seconds * CANONICAL_SAMPLE_RATE)))
    return left, right


def _frames_in(track: VocalFeatureTrack, start: int, end: int) -> np.ndarray:
    return (track.frame_centers >= start) & (track.frame_centers < end)


def _last_run_before(track: VocalFeatureTrack, cue: int) -> tuple[int, int] | None:
    candidates = [run for run in track.active_runs if run[1] <= cue + FRAME_HOP]
    return max(candidates, key=lambda run: run[1]) if candidates else None


def _first_run_after(track: VocalFeatureTrack, cue: int) -> tuple[int, int] | None:
    candidates = [run for run in track.active_runs if run[0] >= cue - FRAME_HOP]
    return min(candidates, key=lambda run: run[0]) if candidates else None


def _nearest_active_run(track: VocalFeatureTrack, cue: int) -> tuple[int, int] | None:
    if not track.active_runs:
        return None
    return min(track.active_runs, key=lambda run: 0 if run[0] <= cue <= run[1] else min(abs(cue - run[0]), abs(cue - run[1])))


def _pitch_summary(track: VocalFeatureTrack, start: int, end: int) -> dict[str, Any]:
    in_window = _frames_in(track, start, end)
    mask = in_window & np.isfinite(track.f0_hz) & track.active
    values = track.f0_hz[mask]
    voicing = track.voicing[mask]
    if values.size < 5:
        return {
            "available": False,
            "reason": "fewer_than_5_voiced_frames",
            "voiced_frame_count": int(values.size),
            "fraction_of_active_frames": None,
        }
    active_count = max(1, int(np.sum(track.active[_frames_in(track, start, end)])))
    median_hz = float(np.median(values))
    cents = 1200.0 * np.log2(values / median_hz)
    smoothed = _median_filter_1d(cents, 5)
    times = (track.frame_centers[mask] - track.frame_centers[mask][0]) / track.sample_rate
    slope = float(np.polyfit(times, smoothed, 1)[0]) if times.size >= 5 and np.ptp(times) > 0 else None
    quarter = max(1, cents.size // 4)
    first_cents = float(np.median(cents[:quarter]))
    last_cents = float(np.median(cents[-quarter:]))
    octave_jumps = float(np.mean(np.abs(np.diff(np.log2(values))) > 0.8)) if values.size > 1 else 0.0
    return {
        "available": True,
        "reason": None,
        "voiced_frame_count": int(values.size),
        "fraction_of_active_frames": float(values.size / active_count),
        "median_f0_hz": median_hz,
        "first_quartile_f0_hz": float(np.median(values[:quarter])),
        "last_quartile_f0_hz": float(np.median(values[-quarter:])),
        "terminal_or_onset_glide_cents": float(last_cents - first_cents),
        "slope_cents_per_second": slope,
        "large_octave_jump_fraction": octave_jumps,
        "median_periodicity": float(np.median(voicing)),
        "method": "yin_cmnd_full_cumulative_dsp_chunked_256_with_running_median_octave_continuity",
        "periodicity_is_not_probability": True,
    }


def _normalise_events(events: Iterable[Mapping[str, Any]] | None, name: str,
                      canonical_start: int, canonical_end: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not events:
        return result
    for raw in events:
        if not isinstance(raw, Mapping):
            raise VocalCompletionInputError(f"{name} entries must be objects")
        text = str(raw.get("text", "")).strip()
        start = raw.get("start_sample")
        end = raw.get("end_sample")
        if start is None or end is None:
            raise VocalCompletionInputError(f"{name} entry lacks canonical sample bounds")
        start_i, end_i = int(start), int(end)
        if start_i < canonical_start or end_i > canonical_end or end_i <= start_i:
            raise VocalCompletionClockError(
                f"{name} interval [{start_i},{end_i}) is outside "
                f"[{canonical_start},{canonical_end})"
            )
        confidence = raw.get("confidence")
        if confidence is not None:
            confidence = float(confidence)
            if not math.isfinite(confidence):
                raise VocalCompletionInputError(f"{name} confidence must be finite")
            confidence = float(np.clip(confidence, 0.0, 1.0))
        result.append({
            "start_sample": start_i,
            "end_sample": end_i,
            "text": text,
            "confidence": confidence,
        })
    result.sort(key=lambda item: (item["start_sample"], item["end_sample"]))
    return result


def _asr_evidence(words: Sequence[Mapping[str, Any]], segments: Sequence[Mapping[str, Any]],
                  cue: int) -> dict[str, Any]:
    before = [word for word in words if int(word["end_sample"]) <= cue]
    after = [word for word in words if int(word["start_sample"]) >= cue]
    nearest_before = max(before, key=lambda item: int(item["end_sample"])) if before else None
    nearest_after = min(after, key=lambda item: int(item["start_sample"])) if after else None
    punctuation = ".,!?;:…—–"
    trailing = ""
    if nearest_before:
        for character in reversed(str(nearest_before["text"]).strip()):
            if character in punctuation:
                trailing = character
                break
            if character.isalnum() or character in "'\"":
                break
    segment_near = [
        segment for segment in segments
        if int(segment["start_sample"]) <= cue + CANONICAL_SAMPLE_RATE // 10
        and int(segment["end_sample"]) >= cue - CANONICAL_SAMPLE_RATE // 10
    ]
    return {
        "available": bool(words),
        "reason": None if words else "no_asr_words_provided",
        "last_word_before_cue": nearest_before,
        "first_word_after_cue": nearest_after,
        "last_word_end_gap_seconds": None if nearest_before is None else (cue - int(nearest_before["end_sample"])) / CANONICAL_SAMPLE_RATE,
        "first_word_start_gap_seconds": None if nearest_after is None else (int(nearest_after["start_sample"]) - cue) / CANONICAL_SAMPLE_RATE,
        "closed_punctuation": trailing or None,
        "closed_punctuation_evidence": bool(trailing),
        "segment_count_near_cue": len(segment_near),
        "segments_near_cue": segment_near,
        "asr_segment_is_not_prosodic_phrase": True,
        "semantic_or_syntactic_closure_inferred": False,
        "source": "provided_word_and_segment_timings_only",
    }


def _breath_evidence(track: VocalFeatureTrack, gap_start: int, gap_end: int) -> dict[str, Any]:
    span = gap_end - gap_start
    if span < int(BREATH_MIN_GAP_SECONDS * CANONICAL_SAMPLE_RATE):
        return {
            "available": False,
            "reason": "no_unvoiced_gap_at_least_80ms",
            "gap_seconds": span / CANONICAL_SAMPLE_RATE if span >= 0 else None,
        }
    mask = _frames_in(track, gap_start, gap_end)
    if not mask.any():
        return {"available": False, "reason": "gap_shorter_than_one_frame", "gap_seconds": span / CANONICAL_SAMPLE_RATE}
    flatness = float(np.median(track.spectral_flatness[mask]))
    hf_ratio = float(np.median(track.high_frequency_ratio[mask]))
    zcr = float(np.median(track.zero_crossing_rate[mask]))
    # Bounded descriptive index; the ranges are documented heuristics, not a
    # trained breath classifier or a calibrated probability.
    flat_component = float(np.clip(flatness / 0.5, 0.0, 1.0))
    hf_component = float(np.clip(hf_ratio / 0.35, 0.0, 1.0))
    zcr_component = float(np.clip(zcr / 0.25, 0.0, 1.0))
    index = 0.5 * flat_component + 0.3 * hf_component + 0.2 * zcr_component
    return {
        "available": True,
        "reason": None,
        "gap_seconds": span / CANONICAL_SAMPLE_RATE,
        "spectral_flatness_median": flatness,
        "high_frequency_ratio_median": hf_ratio,
        "zero_crossing_rate_median": zcr,
        "breath_likeness_index_uncalibrated": float(np.clip(index, 0.0, 1.0)),
        "method": "broadband_noise_descriptors_uncalibrated",
        "is_not_breath_event_label": True,
    }


def describe_vocal_cue(
    track: VocalFeatureTrack,
    cue_sample: int,
    *,
    words: Iterable[Mapping[str, Any]] | None = None,
    segments: Iterable[Mapping[str, Any]] | None = None,
    left_seconds: float = DEFAULT_LEFT_SECONDS,
    right_seconds: float = DEFAULT_RIGHT_SECONDS,
) -> dict[str, Any]:
    """Describe one canonical-sample cue using a precomputed feature track."""

    if not math.isfinite(float(cue_sample)):
        raise VocalCompletionInputError("cue_sample must be finite")
    cue = int(cue_sample)
    if cue < track.canonical_start_sample or cue > track.canonical_end_sample:
        raise VocalCompletionClockError(
            f"cue {cue} is outside canonical span "
            f"[{track.canonical_start_sample},{track.canonical_end_sample}]"
        )
    local_cue = cue - track.canonical_start_sample
    left_bound, right_bound = _window_bounds(local_cue, track.duration_samples, left_seconds, right_seconds)
    canonical_left = track.local_to_canonical(left_bound)
    canonical_right = track.local_to_canonical(right_bound)

    normalised_words = _normalise_events(words, "words", track.canonical_start_sample, track.canonical_end_sample)
    normalised_segments = _normalise_events(segments, "segments", track.canonical_start_sample, track.canonical_end_sample)

    cue_frame_index = int(np.argmin(np.abs(track.frame_centers - local_cue)))
    active_at_cue = bool(track.active[cue_frame_index])
    active_run = _nearest_active_run(track, local_cue)
    cue_inside_run = bool(active_run and active_run[0] <= local_cue < active_run[1])

    if cue_inside_run:
        # A cue in the middle of a voiced run has no release and no gap; the
        # neighbouring-run durations must not be presented as its terminal
        # lengthening.
        last_before = None
        first_after = None
        gap_start = gap_end = local_cue
    else:
        last_before = _last_run_before(track, local_cue)
        first_after = _first_run_after(track, local_cue)
        if last_before is not None and first_after is not None:
            gap_start, gap_end = last_before[1], first_after[0]
        elif last_before is not None:
            gap_start, gap_end = last_before[1], min(track.duration_samples, last_before[1] + int(0.5 * CANONICAL_SAMPLE_RATE))
        elif first_after is not None:
            gap_start, gap_end = max(0, first_after[0] - int(0.5 * CANONICAL_SAMPLE_RATE)), first_after[0]
        else:
            gap_start = gap_end = local_cue
    gap_seconds = max(0.0, (gap_end - gap_start) / CANONICAL_SAMPLE_RATE)

    run_durations = [(end - start) / CANONICAL_SAMPLE_RATE for start, end in track.active_runs]
    last_duration = None if last_before is None else (last_before[1] - last_before[0]) / CANONICAL_SAMPLE_RATE

    onset_before = track.onsets[track.onsets <= local_cue] if track.onsets.size else np.empty(0)
    onset_after = track.onsets[track.onsets >= local_cue] if track.onsets.size else np.empty(0)
    onset_strength_before = track.onset_strength[track.onsets <= local_cue] if track.onsets.size else np.empty(0)
    onset_strength_after = track.onset_strength[track.onsets >= local_cue] if track.onsets.size else np.empty(0)
    last_onset = int(onset_before[-1]) if onset_before.size else None
    first_onset = int(onset_after[0]) if onset_after.size else None

    # Terminal lengthening: duration from the last attack inside the final
    # voiced run to that run's release, relative to the median attack interval
    # in the analysis context.  This is a measured timing ratio, not a learned
    # "final syllable" label.
    final_syllable_span = None
    median_attack_interval = None
    terminal_ratio = None
    if last_before is not None and last_onset is not None and last_before[0] <= last_onset <= last_before[1]:
        final_syllable_span = (last_before[1] - last_onset) / CANONICAL_SAMPLE_RATE
        context_attacks = track.onsets[
            (track.onsets >= left_bound) & (track.onsets <= right_bound) & (track.onsets <= local_cue)
        ]
        if context_attacks.size >= 4:
            intervals = np.diff(context_attacks) / CANONICAL_SAMPLE_RATE
            intervals = intervals[intervals > 0.04]
            if intervals.size >= 3:
                median_attack_interval = float(np.median(intervals))
                if median_attack_interval > 1e-6:
                    terminal_ratio = float(final_syllable_span / median_attack_interval)

    pitch_before = _pitch_summary(track, max(left_bound, local_cue - int(1.0 * CANONICAL_SAMPLE_RATE)), local_cue)
    pitch_after = _pitch_summary(track, local_cue, min(right_bound, local_cue + int(0.8 * CANONICAL_SAMPLE_RATE)))
    breath = _breath_evidence(track, gap_start, gap_end)
    asr = _asr_evidence(normalised_words, normalised_segments, cue)
    pretrained = track.pretrained_vad
    if pretrained and pretrained.get("available"):
        intervals = pretrained["intervals"]
        containing = [
            item for item in intervals
            if item["start_sample"] <= cue < item["end_sample"]
        ]
        ended_before = [item for item in intervals if item["end_sample"] <= cue]
        starts_after = [item for item in intervals if item["start_sample"] >= cue]
        last_vad = max(ended_before, key=lambda item: item["end_sample"]) if ended_before else None
        next_vad = min(starts_after, key=lambda item: item["start_sample"]) if starts_after else None
        vad_gap = (
            None if last_vad is None or next_vad is None
            else (next_vad["start_sample"] - last_vad["end_sample"]) / CANONICAL_SAMPLE_RATE
        )
        pretrained_evidence = {
            **pretrained,
            "cue_in_speech": bool(containing),
            "last_speech_end_before_cue_seconds": (
                None if last_vad is None else (cue - last_vad["end_sample"]) / CANONICAL_SAMPLE_RATE
            ),
            "first_speech_start_after_cue_seconds": (
                None if next_vad is None else (next_vad["start_sample"] - cue) / CANONICAL_SAMPLE_RATE
            ),
            "gap_between_silero_intervals_seconds": vad_gap,
            "agrees_with_dsp_activity_at_cue": bool(containing) == bool(active_at_cue),
        }
    else:
        pretrained_evidence = pretrained or {
            "available": False,
            "reason": "pretrained_vad_not_requested_or_unavailable",
            "provider": "silero-vad",
        }

    active_mask = _frames_in(track, left_bound, right_bound)
    active_fraction = float(np.mean(track.active[active_mask])) if active_mask.any() else None
    active_after_mask = _frames_in(track, local_cue, right_bound)
    active_after_fraction = float(np.mean(track.active[active_after_mask])) if active_after_mask.any() else None

    onset_strength_scale = max(track.onset_strength_percentile, float(np.median(track.onset_strength)) if track.onset_strength.size else 1e-6, 1e-6)
    first_onset_strength = None
    if first_onset is not None:
        matches = np.flatnonzero(track.onsets == first_onset)
        if matches.size:
            first_onset_strength = float(track.onset_strength[matches[0]] / onset_strength_scale)

    outgoing = {
        "voice_active_at_cue": active_at_cue,
        "last_voice_offset_before_cue_seconds": None if last_before is None else (local_cue - last_before[1]) / CANONICAL_SAMPLE_RATE,
        "last_active_run_seconds": last_duration,
        "final_attack_to_offset_seconds": final_syllable_span,
        "median_attack_interval_seconds": median_attack_interval,
        "terminal_lengthening_ratio": terminal_ratio,
        "terminal_lengthening_status": (
            "measured_final_attack_to_offset_vs_median_attack_interval" if terminal_ratio is not None
            else "unavailable_no_final_attack_fewer_than_3_intervals_or_no_release"
        ),
        "last_attack_before_cue_seconds": None if last_onset is None else (local_cue - last_onset) / CANONICAL_SAMPLE_RATE,
        "last_attack_strength_relative": None if onset_strength_before.size == 0 else float(onset_strength_before[-1] / onset_strength_scale),
        "pitch_terminal": pitch_before,
        "unvoiced_gap_seconds": gap_seconds,
        "breath": breath,
        "asr": asr,
    }
    incoming = {
        "first_voice_onset_after_cue_seconds": None if first_after is None else (first_after[0] - local_cue) / CANONICAL_SAMPLE_RATE,
        "active_fraction_after_cue": active_after_fraction,
        "first_attack_after_cue_seconds": None if first_onset is None else (first_onset - local_cue) / CANONICAL_SAMPLE_RATE,
        "first_attack_strength_relative": first_onset_strength,
        "pitch_onset": pitch_after,
        "pickup_note": "an onset after the cue is pickup evidence, not proof that the incoming phrase completes here",
    }

    components: dict[str, float] = {}
    available: dict[str, bool] = {}
    components["inactive_at_cue"] = 0.0 if active_at_cue else 1.0
    available["inactive_at_cue"] = True
    components["gap_length"] = float(np.clip(gap_seconds / 0.5, 0.0, 1.0))
    available["gap_length"] = True
    if terminal_ratio is not None:
        components["terminal_lengthening"] = float(np.clip((terminal_ratio - 1.0) / 0.75, 0.0, 1.0))
        available["terminal_lengthening"] = True
    else:
        available["terminal_lengthening"] = False
    if breath.get("available"):
        components["breath_index"] = float(breath["breath_likeness_index_uncalibrated"])
        available["breath_index"] = True
    else:
        available["breath_index"] = False
    components["terminal_pitch_available"] = 1.0 if pitch_before.get("available") else 0.0
    available["terminal_pitch_available"] = True
    components["asr_closed_punctuation"] = 1.0 if asr["closed_punctuation_evidence"] else 0.0
    available["asr_closed_punctuation"] = bool(asr["available"])

    completion = _weighted_evidence(components, available, COMPLETION_WEIGHTS)

    pickup_components: dict[str, float] = {}
    pickup_available: dict[str, bool] = {}
    latency = incoming["first_voice_onset_after_cue_seconds"]
    pickup_components["onset_proximity"] = 0.0 if latency is None else float(np.clip(1.0 - latency / 0.5, 0.0, 1.0))
    pickup_available["onset_proximity"] = latency is not None
    pickup_components["onset_strength"] = 0.0 if first_onset_strength is None else float(np.clip(first_onset_strength, 0.0, 1.0))
    pickup_available["onset_strength"] = first_onset_strength is not None
    pickup_components["active_after_cue"] = float(active_after_fraction or 0.0)
    pickup_available["active_after_cue"] = active_after_fraction is not None
    if pitch_after.get("available"):
        glide = pitch_after.get("terminal_or_onset_glide_cents")
        pickup_components["onset_pitch_movement"] = 0.0 if glide is None else float(np.clip(abs(glide) / 200.0, 0.0, 1.0))
        pickup_available["onset_pitch_movement"] = True
    else:
        pickup_available["onset_pitch_movement"] = False
    pickup_components["asr_word_onset"] = 0.0 if asr["first_word_after_cue"] is None else float(np.clip(1.0 - (asr["first_word_start_gap_seconds"] or 0.0) / 0.5, 0.0, 1.0))
    pickup_available["asr_word_onset"] = bool(asr["available"] and asr["first_word_after_cue"] is not None)
    pickup = _weighted_evidence(pickup_components, pickup_available, PICKUP_WEIGHTS)

    uncertainty = list(track.warnings)
    uncertainty.extend([
        "descriptor_heuristic_is_uncalibrated_not_closure_probability",
        "asr_segment_is_not_prosodic_phrase",
        "phrase_closure_and_tension_are_not_identified_by_this_module",
    ])
    if not asr["available"]:
        uncertainty.append("asr_words_unavailable_no_semantic_or_syntactic_evidence")
    if not breath.get("available"):
        uncertainty.append("breath_channel_unavailable")
    if not pitch_before.get("available"):
        uncertainty.append("terminal_pitch_contour_unavailable")
    if track.snr_db < 6.0:
        uncertainty.append("activity_threshold_close_to_noise_floor")
    if not pretrained_evidence.get("available"):
        uncertainty.append("pretrained_silero_vad_unavailable")
    elif not pretrained_evidence.get("agrees_with_dsp_activity_at_cue", True):
        uncertainty.append("pretrained_silero_vad_disagrees_with_dsp_at_cue")

    return {
        "schema": SCHEMA_VERSION,
        "identity": {
            "canonical_start_sample": track.canonical_start_sample,
            "canonical_end_sample": track.canonical_end_sample,
            "duration_samples": track.duration_samples,
            "sample_rate": track.sample_rate,
            "cue_sample": cue,
            "cue_local_sample": local_cue,
            "cue_seconds_in_excerpt": local_cue / track.sample_rate,
        },
        "window": {
            "left_seconds": left_seconds,
            "right_seconds": right_seconds,
            "canonical_start_sample": canonical_left,
            "canonical_end_sample": canonical_right,
            "local_start_sample": left_bound,
            "local_end_sample": right_bound,
            "clamped_at_excerpt_edge": left_bound == 0 or right_bound == track.duration_samples,
        },
        "voice_activity": {
            "method": "energy_vad_relative_percentile_with_120ms_gap_closing",
            "active_at_cue": active_at_cue,
            "active_fraction_in_window": active_fraction,
            "nearest_active_run": None if active_run is None else {
                "start_sample": track.local_to_canonical(active_run[0]),
                "end_sample": track.local_to_canonical(active_run[1]),
            },
            "active_runs_in_context": [
                {
                    "start_sample": track.local_to_canonical(start),
                    "end_sample": track.local_to_canonical(end),
                    "duration_seconds": (end - start) / track.sample_rate,
                }
                for start, end in track.active_runs
                if end >= left_bound and start <= right_bound
            ],
            "noise_floor_db": track.noise_db,
            "speech_level_db": track.speech_db,
            "threshold_db": track.threshold_db,
            "relative_snr_db": track.snr_db,
            "gain_invariant_relative_to_constant_gain": True,
            "caveat": "energy activity is not ASR and not a voiced/unvoiced decision",
        },
        "syllable_attack_timing": {
            "method": "normalised_spectral_flux_peaks_90ms_minimum_spacing",
            "attack_count_in_window": int(np.sum((track.onsets >= left_bound) & (track.onsets <= right_bound))),
            "attacks_in_context_seconds": [
                round(float(value) / track.sample_rate, 4)
                for value in track.onsets[(track.onsets >= left_bound) & (track.onsets <= right_bound)]
            ],
            "last_attack_before_cue_seconds": outgoing["last_attack_before_cue_seconds"],
            "first_attack_after_cue_seconds": incoming["first_attack_after_cue_seconds"],
            "is_syllable_label": False,
            "caveat": "attacks approximate syllabic/percussive timing; they are not verified syllable nuclei",
        },
        "pitch": {
            "method": "yin_cmnd_autocorrelation_dsp",
            "terminal": pitch_before,
            "onset": pitch_after,
            "contour_availability_is_not_intonation_closure": True,
        },
        "terminal_lengthening": {
            "last_active_run_seconds": last_duration,
            "median_active_run_seconds": None if not run_durations else float(np.median(run_durations)),
            "final_attack_to_offset_seconds": final_syllable_span,
            "median_attack_interval_seconds": median_attack_interval,
            "ratio": terminal_ratio,
            "status": outgoing["terminal_lengthening_status"],
            "method": "final_attack_to_offset_vs_median_attack_interval_dsp",
            "caveat": "measured timing ratio; no learned terminal-lengthening classifier",
        },
        "breath": breath,
        "asr": asr,
        "pretrained_vad": pretrained_evidence,
        "outgoing_completion_descriptors": outgoing,
        "incoming_pickup_descriptors": incoming,
        "asymmetry": {
            "outgoing_question": "does A release its vocal phrase near the cue?",
            "incoming_question": "does B start a new vocal phrase after the cue, possibly as a pickup?",
            "single_symmetric_score_used": False,
        },
        "completion_evidence": completion,
        "incoming_pickup_evidence": pickup,
        "uncertainty": uncertainty,
        "labels_not_emitted": ["closure", "resolved", "chorus", "phrase_boundary_certified"],
    }


def _weighted_evidence(components: Mapping[str, float], available: Mapping[str, bool],
                       weights: Mapping[str, float]) -> dict[str, Any]:
    used = {name: float(components[name]) for name in weights if available.get(name)}
    total_weight = sum(weights[name] for name in used)
    all_weight = sum(weights.values())
    score = None if total_weight <= 0 else sum(weights[name] * used[name] for name in used) / total_weight
    return {
        "score_uncalibrated": score,
        "coverage": total_weight / all_weight if all_weight > 0 else 0.0,
        "components": {name: float(components[name]) for name in weights if name in components},
        "available": {name: bool(available.get(name)) for name in weights},
        "weights": dict(weights),
        "score_kind": "descriptor_heuristic_not_probability_not_closure",
        "weights_were_fitted_to_ratings": False,
    }


def analyze_vocal_completion(
    samples: Any,
    sample_rate: int,
    cue_sample: int,
    *,
    canonical_start_sample: int = 0,
    words: Iterable[Mapping[str, Any]] | None = None,
    segments: Iterable[Mapping[str, Any]] | None = None,
    left_seconds: float = DEFAULT_LEFT_SECONDS,
    right_seconds: float = DEFAULT_RIGHT_SECONDS,
    feature_track: VocalFeatureTrack | None = None,
) -> dict[str, Any]:
    """Canonical-sample entry point for one cue on a stem or excerpt."""

    track = feature_track or extract_vocal_features(
        samples, sample_rate, canonical_start_sample=canonical_start_sample
    )
    if track.sample_rate != int(sample_rate):
        raise VocalCompletionClockError("feature_track sample rate does not match sample_rate")
    if track.canonical_start_sample != int(canonical_start_sample):
        raise VocalCompletionClockError("feature_track canonical_start_sample mismatch")
    return describe_vocal_cue(
        track, cue_sample, words=words, segments=segments,
        left_seconds=left_seconds, right_seconds=right_seconds,
    )


def rank_phrase_end_candidates(
    samples: Any,
    sample_rate: int,
    candidate_samples: Sequence[int],
    *,
    canonical_start_sample: int = 0,
    words: Iterable[Mapping[str, Any]] | None = None,
    segments: Iterable[Mapping[str, Any]] | None = None,
    left_seconds: float = DEFAULT_LEFT_SECONDS,
    right_seconds: float = DEFAULT_RIGHT_SECONDS,
    keep_details: bool = False,
) -> list[dict[str, Any]]:
    """Rank candidates by measured descriptor evidence (not calibrated closure)."""

    track = extract_vocal_features(samples, sample_rate, canonical_start_sample=canonical_start_sample)
    results: list[dict[str, Any]] = []
    for cue in candidate_samples:
        detail = describe_vocal_cue(
            track, int(cue), words=words, segments=segments,
            left_seconds=left_seconds, right_seconds=right_seconds,
        )
        completion = detail["completion_evidence"]
        pickup = detail["incoming_pickup_evidence"]
        item: dict[str, Any] = {
            "cue_sample": int(cue),
            "cue_local_sample": detail["identity"]["cue_local_sample"],
            "cue_seconds_in_excerpt": detail["identity"]["cue_seconds_in_excerpt"],
            "completion_score_uncalibrated": completion["score_uncalibrated"],
            "completion_coverage": completion["coverage"],
            "incoming_pickup_score_uncalibrated": pickup["score_uncalibrated"],
            "incoming_pickup_coverage": pickup["coverage"],
            "active_at_cue": detail["voice_activity"]["active_at_cue"],
            "gap_seconds": detail["outgoing_completion_descriptors"]["unvoiced_gap_seconds"],
            "terminal_lengthening_ratio": detail["outgoing_completion_descriptors"]["terminal_lengthening_ratio"],
            "breath_available": detail["breath"]["available"],
            "pitch_terminal_available": detail["pitch"]["terminal"]["available"],
            "asr_available": detail["asr"]["available"],
            "pretrained_vad_available": bool(detail["pretrained_vad"].get("available")),
            "pretrained_vad_cue_in_speech": detail["pretrained_vad"].get("cue_in_speech"),
            "uncertainty_count": len(detail["uncertainty"]),
        }
        if keep_details:
            item["detail"] = detail
        results.append(item)
    results.sort(key=lambda item: (
        item["completion_score_uncalibrated"] is None,
        -(item["completion_score_uncalibrated"] or 0.0),
        item["cue_sample"],
    ))
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank
    return results


def analyze_staged_file(
    path: str | os.PathLike[str],
    clock: SourceClock,
    cue_canonical_sample: int,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load a staged excerpt, verify its hash, and analyse a canonical cue."""

    verify = verify_staged_file(path, clock)
    if not verify["matches_manifest"]:
        raise VocalCompletionInputError(f"staged file hash does not match manifest: {path}")
    track_audio: AudioTrack = load_audio(path)
    if track_audio.sample_rate != clock.sample_rate:
        raise VocalCompletionClockError(
            f"file sample rate {track_audio.sample_rate} != clock rate {clock.sample_rate}"
        )
    if track_audio.duration_samples != clock.duration_samples:
        raise VocalCompletionClockError(
            f"file duration {track_audio.duration_samples} != clock duration {clock.duration_samples}"
        )
    result = analyze_vocal_completion(
        track_audio.samples,
        track_audio.sample_rate,
        cue_canonical_sample,
        canonical_start_sample=clock.source_start_sample,
        **kwargs,
    )
    result["file"] = {
        "path": str(path),
        **verify,
        "clock": {
            "source_start_sample": clock.source_start_sample,
            "source_end_sample": clock.source_start_sample + clock.duration_samples,
            "sample_rate": clock.sample_rate,
        },
    }
    return result


__all__ = [
    "CANONICAL_SAMPLE_RATE",
    "SCHEMA_VERSION",
    "SourceClock",
    "VocalCompletionClockError",
    "VocalCompletionError",
    "VocalCompletionInputError",
    "VocalFeatureTrack",
    "analyze_staged_file",
    "analyze_vocal_completion",
    "clock_for_file",
    "describe_vocal_cue",
    "extract_vocal_features",
    "load_source_clocks",
    "rank_phrase_end_candidates",
    "verify_staged_file",
]
