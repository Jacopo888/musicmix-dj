"""Directional, context-aware transition ranking.

This module is deliberately an adapter around the small domain contracts.  It
accepts the JSON-shaped dictionaries emitted by the structure and vocal
analysis stages, but does not depend on either stage's implementation.  A
candidate is an *outgoing* context from ``track_a`` followed by an *incoming*
context from ``track_b``; swapping the tracks therefore changes both the
features and the candidate identity.

The implementation is conservative about timing evidence.  Word gaps are
accepted only inside one explicitly observed ASR segment (or a compact
legacy ASR coverage row); they are never labelled measured acoustic silence.
Missing coverage is reported as unknown and is never interpreted as silence.
No pitch shift or time stretch is inferred by this ranker.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .audio import AudioTrack
from .contracts import Candidate, Feature


ALGORITHM_VERSION = "directional-v3-role-aware"
DEFAULT_MAX_CANDIDATES = 4096
_WINDOWS = (4, 8)
_VOCAL_MIN_CONFIDENCE = 0.60
_VOCAL_MARGIN_SECONDS = 0.020
_TARGET_FADE_SECONDS = 0.04
_BEAT_PHASE_LIMIT = 0.08
_COMBINED_SHIFT_LIMIT_SECONDS = 0.08


@dataclass(frozen=True)
class DirectionalProposal:
    """A ranked directional transition with auditable feature values."""

    candidate: Candidate
    score: float
    breakdown: Mapping[str, float]
    raw_features: Mapping[str, Any]
    fade_seconds: float
    alignment: Mapping[str, Any]
    reasons: tuple[str, ...] = ()

    @property
    def alignment_diagnostics(self) -> Mapping[str, Any]:
        """Compatibility alias for callers that use the longer name."""

        return self.alignment


@dataclass(frozen=True)
class DirectionalSearchResult:
    """All selected proposals and auditable rejections from one search."""

    items: tuple[DirectionalProposal, ...]
    rejected: tuple[Mapping[str, Any], ...]
    diagnostics: Mapping[str, Any]
    # ``items`` is the bounded, diverse top-k surface used by renderers;
    # ``proposals`` keeps every scored proposal for audit/re-ranking.
    proposals: tuple[DirectionalProposal, ...] = ()
    selected: tuple[str, ...] = ()

    @property
    def selected_items(self) -> tuple[DirectionalProposal, ...]:
        return self.items


@dataclass(frozen=True)
class _Event:
    sample: int
    confidence: float | None
    source: str = "unknown"


@dataclass(frozen=True)
class _CueContext:
    sample: int
    confidence: float | None
    cue_source: str
    window_beats: int
    start_sample: int
    end_sample: int
    segment_bounds: tuple[tuple[int, int], ...]
    tempo_bpm: float | None
    tempo_spread: float | None
    rms: tuple[float, ...]
    chroma: tuple[tuple[float, ...], ...]
    timbre: tuple[float, ...]
    trend: float | None
    direction: str
    beat_offset_samples: int = 0


def _mapping_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _confidence(value: Any) -> float | None:
    number = _finite_number(value)
    if number is None:
        return None
    return max(0.0, min(1.0, number))


def _event_list(payload: Mapping[str, Any], key: str, duration: int) -> tuple[_Event, ...]:
    records = _mapping_value(payload, key, ()) or ()
    events: list[_Event] = []
    for record in records:
        raw_sample = _mapping_value(record, "sample", None)
        sample = _finite_number(raw_sample)
        if sample is None or sample < 0 or sample >= duration:
            continue
        events.append(
            _Event(
                int(round(sample)),
                _confidence(_mapping_value(record, "confidence", None)),
                "provided_cue" if key == "provided_cues" else ("downbeat" if key == "downbeats" else "beat"),
            )
        )
    events.sort(key=lambda event: (event.sample, event.confidence is None, event.confidence or 0.0))
    dedup: list[_Event] = []
    for event in events:
        if dedup and dedup[-1].sample == event.sample:
            # Keep the strongest known confidence deterministically.
            previous = dedup[-1]
            if previous.confidence is None or (event.confidence is not None and event.confidence > previous.confidence):
                dedup[-1] = event
        else:
            dedup.append(event)
    return tuple(dedup)


def _samples_for_beats(payload: Mapping[str, Any], duration: int) -> tuple[int, ...]:
    values: list[int] = []
    for event in _event_list(payload, "beats", duration):
        values.append(event.sample)
    return tuple(values)


def _cue_events(
    payload: Mapping[str, Any],
    duration: int,
    min_downbeat_confidence: float = 0.20,
    cue_policy: str = "downbeats",
    role: str | None = None,
) -> tuple[tuple[_Event, ...], str, tuple[_Event, ...]]:
    if cue_policy not in {"downbeats", "beats", "provided"}:
        raise ValueError("cue_policy must be 'downbeats', 'beats', or 'provided'")
    if cue_policy == "provided":
        if "provided_cues" not in payload:
            raise ValueError("provided cue policy requires explicit provided_cues, including an empty list")
        if not _event_list(payload, "beats", duration):
            raise ValueError("provided cue policy requires an independent beat grid for context")
        if role is not None and role not in {"outgoing", "incoming"}:
            raise ValueError("Unknown transition role")
        retained = []
        for cue in payload["provided_cues"]:
            # Untyped legacy/model opportunities remain usable in either
            # direction. Explicit vocal roles must survive the search adapter.
            declared = cue.get("roles", [cue.get("role", "both")]) if isinstance(cue, Mapping) else ["both"]
            if isinstance(declared, str):
                declared = [declared]
            if not declared or any(value not in {"outgoing", "incoming", "both"} for value in declared):
                raise ValueError("Invalid provided cue role")
            if role is None or "both" in declared or role in declared:
                retained.append(cue)
        return _event_list(dict(payload, provided_cues=retained), "provided_cues", duration), "provided_cues", tuple()
    downbeats = _event_list(payload, "downbeats", duration)
    low = tuple(event for event in downbeats if event.confidence is not None and event.confidence < min_downbeat_confidence)
    if cue_policy == "beats":
        return _event_list(payload, "beats", duration), "beats", low
    if downbeats:
        usable = tuple(event for event in downbeats if event not in low)
        if usable:
            return usable, "downbeats", low
        fallback = _event_list(payload, "beats", duration)
        return fallback, "beats_fallback_low_confidence_downbeats", low
    return _event_list(payload, "beats", duration), "beats_fallback", tuple()


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    if not left or not right:
        return None
    length = min(len(left), len(right))
    a = np.asarray(left[:length], dtype=float)
    b = np.asarray(right[:length], dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return None
    return max(0.0, min(1.0, float(np.dot(a, b) / denominator)))


def _scalar_similarity(left: float | None, right: float | None, scale: float) -> float | None:
    if left is None or right is None:
        return None
    return max(0.0, min(1.0, math.exp(-abs(left - right) / max(scale, 1e-9))))


def _audio_descriptor(track: AudioTrack, start: int, end: int) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    start = max(0, min(track.duration_samples, int(start)))
    end = max(start, min(track.duration_samples, int(end)))
    segment = np.asarray(track.samples[start:end], dtype=float)
    if segment.ndim == 2 and segment.shape[1]:
        signal = np.mean(segment, axis=1)
    else:
        signal = np.asarray(segment, dtype=float).reshape(-1)
    if signal.size == 0:
        return 0.0, tuple(), tuple()
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    rms = float(np.sqrt(np.mean(signal * signal)))
    if signal.size < 8:
        return rms, tuple(), (rms, 0.0, 0.0)
    nfft = min(2048, int(signal.size))
    # Average several bounded frames over the complete beat.  Looking only at
    # ``signal[-nfft:]`` made a long beat's last 43 ms stand in for its entire
    # harmonic context and made early changes invisible.
    if signal.size <= nfft:
        starts = (0,)
    else:
        count = min(8, max(2, int(math.ceil(signal.size / max(nfft // 2, 1)))))
        starts = tuple(int(round(value)) for value in np.linspace(0, signal.size - nfft, count))
    window = np.hanning(nfft)
    spectra = [np.abs(np.fft.rfft(signal[start : start + nfft] * window)) for start in starts]
    spectrum = np.mean(np.asarray(spectra, dtype=float), axis=0)
    frequencies = np.fft.rfftfreq(nfft, 1.0 / float(track.sample_rate))
    usable = (frequencies >= 20.0) & (frequencies <= max(20.0, float(track.sample_rate) / 2.0))
    frequencies = frequencies[usable]
    spectrum = spectrum[usable]
    if not spectrum.size or float(np.sum(spectrum)) <= 1e-12:
        return rms, tuple(), (rms, 0.0, 0.0)
    weights = spectrum / float(np.sum(spectrum))
    centroid = float(np.sum(frequencies * weights)) / max(float(track.sample_rate) / 2.0, 1.0)
    bandwidth = float(np.sqrt(np.sum(((frequencies - centroid * track.sample_rate / 2.0) ** 2) * weights))) / max(float(track.sample_rate) / 2.0, 1.0)
    cumulative = np.cumsum(weights)
    rolloff_index = int(np.searchsorted(cumulative, 0.85, side="left"))
    rolloff = float(frequencies[min(rolloff_index, len(frequencies) - 1)]) / max(float(track.sample_rate) / 2.0, 1.0)
    # A compact chroma proxy based on FFT-bin pitch classes.  It is a
    # similarity descriptor only; no pitch transposition is applied.
    chroma = np.zeros(12, dtype=float)
    positive = frequencies > 0
    midi = np.rint(69.0 + 12.0 * np.log2(np.maximum(frequencies[positive], 1e-6) / 440.0)).astype(int)
    for pitch_class, weight in zip(midi % 12, weights[positive]):
        chroma[int(pitch_class)] += float(weight)
    norm = float(np.linalg.norm(chroma))
    if norm > 1e-12:
        chroma /= norm
    return rms, tuple(float(value) for value in chroma), (centroid, bandwidth, rolloff)


def _context_segments(beat_samples: Sequence[int], beat_index: int, window: int, direction: str) -> tuple[tuple[int, int], ...] | None:
    if direction == "outgoing":
        if beat_index < window:
            return None
        bounds = tuple(int(value) for value in beat_samples[beat_index - window : beat_index + 1])
    else:
        if beat_index + window >= len(beat_samples):
            return None
        bounds = tuple(int(value) for value in beat_samples[beat_index : beat_index + window + 1])
    segments = tuple((bounds[index], bounds[index + 1]) for index in range(len(bounds) - 1) if bounds[index + 1] > bounds[index])
    return segments if len(segments) == window else None


def _tempo(beat_samples: Sequence[int], indices: Iterable[int], rate: int) -> tuple[float | None, float | None]:
    selected = [beat_samples[index] for index in indices if 0 <= index < len(beat_samples)]
    if len(selected) < 2:
        return None, None
    times = np.asarray(selected, dtype=float) / float(rate)
    intervals = np.diff(times)
    intervals = intervals[intervals > 1e-5]
    if not intervals.size:
        return None, None
    # Fit beat index against the full context span.  A single rounded beat
    # interval can move by a whole 20 ms provider frame; the slope averages
    # that quantization over all four/eight beats.
    indices_array = np.arange(len(times), dtype=float)
    slope = float(np.polyfit(times, indices_array, 1)[0]) if len(times) >= 2 else 0.0
    if not math.isfinite(slope) or slope <= 0:
        return None, None
    spread = float(np.std(intervals) / max(float(np.mean(intervals)), 1e-9))
    return float(60.0 * slope), spread


def _context_for(
    track: AudioTrack,
    beat_samples: Sequence[int],
    cue: _Event,
    window: int,
    direction: str,
    cue_source: str = "unknown",
    descriptor_cache: dict[tuple[int, int], tuple[float, tuple[float, ...], tuple[float, ...]]] | None = None,
) -> _CueContext | None:
    if not beat_samples:
        return None
    index = bisect_left(beat_samples, cue.sample)
    if index >= len(beat_samples) or beat_samples[index] != cue.sample:
        # A downbeat can be a rounded sample in a separately produced beat
        # stream.  Use the nearest beat only when it is unambiguous.
        candidates = [max(0, min(len(beat_samples) - 1, index - 1)), max(0, min(len(beat_samples) - 1, index))]
        index = min(candidates, key=lambda candidate: (abs(beat_samples[candidate] - cue.sample), candidate))
    segments = _context_segments(beat_samples, index, window, direction)
    if segments is None:
        return None
    if cue_source == "provided_cue":
        # Preserve the model timestamp. Beat intervals supply context lengths,
        # not a quantization of the proposed transition back to the old grid.
        delta = cue.sample - beat_samples[index]
        segments = tuple((left + delta, right + delta) for left, right in segments)
        if segments[0][0] < 0 or segments[-1][1] > track.duration_samples:
            return None
    rms_values: list[float] = []
    chroma_values: list[tuple[float, ...]] = []
    timbre_values: list[tuple[float, ...]] = []
    for start, end in segments:
        key = (int(start), int(end))
        if descriptor_cache is not None and key in descriptor_cache:
            rms, chroma, timbre = descriptor_cache[key]
        else:
            rms, chroma, timbre = _audio_descriptor(track, start, end)
            if descriptor_cache is not None:
                descriptor_cache[key] = (rms, chroma, timbre)
        rms_values.append(rms)
        chroma_values.append(chroma)
        timbre_values.append(timbre)
    timbre_array = np.asarray([value for value in timbre_values if value], dtype=float)
    timbre = tuple(float(value) for value in np.mean(timbre_array, axis=0)) if timbre_array.size else tuple()
    log_rms = np.log(np.maximum(np.asarray(rms_values, dtype=float), 1e-8))
    trend = float(np.polyfit(np.arange(len(log_rms), dtype=float), log_rms, 1)[0]) if len(log_rms) >= 2 else None
    if direction == "outgoing":
        first = index - window
        tempo_indices = range(first, index + 1)
    else:
        tempo_indices = range(index, index + window + 1)
    tempo_bpm, tempo_spread = _tempo(beat_samples, tempo_indices, track.sample_rate)
    return _CueContext(
        sample=cue.sample,
        confidence=cue.confidence,
        cue_source=cue_source,
        window_beats=window,
        start_sample=segments[0][0],
        end_sample=segments[-1][1],
        segment_bounds=segments,
        tempo_bpm=tempo_bpm,
        tempo_spread=tempo_spread,
        rms=tuple(rms_values),
        chroma=tuple(chroma_values),
        timbre=timbre,
        trend=trend,
        direction=direction,
        beat_offset_samples=cue.sample - beat_samples[index] if cue_source == "provided_cue" else 0,
    )


def _sequence_similarity(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float | None:
    length = min(len(left), len(right))
    values = [_cosine(left[index], right[index]) for index in range(length)]
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _trend_score(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    # Signed trends matter: an outgoing fade-down and incoming fade-up are
    # treated differently from two trends moving in the same direction.
    return max(0.0, min(1.0, math.exp(-abs(left - right) / 0.35)))


def _tempo_score(left: float | None, right: float | None) -> tuple[float | None, float | None, float | None]:
    if left is None or right is None or left <= 0 or right <= 0:
        return None, None, None
    distance = abs(math.log2(left / right))
    octave_distance = min(distance, abs(distance - 1.0), abs(distance + 1.0))
    return math.exp(-octave_distance / 0.35), distance, octave_distance


def _word_records(payload: Mapping[str, Any] | None, duration: int) -> tuple[tuple[int, int, float | None], ...]:
    if payload is None:
        return tuple()
    records = _mapping_value(payload, "words", ()) or ()
    words: list[tuple[int, int, float | None]] = []
    for record in records:
        start = _finite_number(_mapping_value(record, "start_sample", None))
        end = _finite_number(_mapping_value(record, "end_sample", None))
        if start is None or end is None or end <= start:
            continue
        start_i = max(0, min(duration, int(round(start))))
        end_i = max(0, min(duration, int(round(end))))
        if end_i > start_i:
            words.append((start_i, end_i, _confidence(_mapping_value(record, "confidence", None))))
    words.sort(key=lambda word: (word[0], word[1], word[2] is None, word[2] or 0.0))
    return tuple(words)


def _asr_segment_records(
    payload: Mapping[str, Any] | None,
    duration: int,
) -> tuple[tuple[int, int, float | None], ...]:
    """Return explicitly observed ASR segment intervals.

    Segment timing is kept separate from acoustic occupancy.  A segment says
    where the recognizer observed speech-related transcript; it does not say
    that the interval between two words is measured silence.
    """

    if payload is None:
        return tuple()
    records = _mapping_value(payload, "asr_segments", ()) or ()
    segments: list[tuple[int, int, float | None]] = []
    for record in records:
        start = _finite_number(_mapping_value(record, "start_sample", _mapping_value(record, "start", None)))
        end = _finite_number(_mapping_value(record, "end_sample", _mapping_value(record, "end", None)))
        if start is None or end is None or end <= start:
            continue
        start_i = max(0, min(duration, int(round(start))))
        end_i = max(0, min(duration, int(round(end))))
        if end_i > start_i:
            segments.append((start_i, end_i, _confidence(_mapping_value(record, "confidence", None))))
    return tuple(sorted(segments, key=lambda value: (value[0], value[1], value[2] is None, value[2] or 0.0)))


def _coverage_records(payload: Mapping[str, Any] | None, duration: int) -> tuple[tuple[int, int, str], ...]:
    if payload is None:
        return tuple()
    records = _mapping_value(
        payload,
        "coverage_intervals",
        _mapping_value(
            payload,
            "coverage",
            _mapping_value(payload, "acoustic_intervals", _mapping_value(payload, "acoustic_occupancy", ())),
        ),
    ) or ()
    coverage: list[tuple[int, int, str]] = []
    for record in records:
        if isinstance(record, (list, tuple)) and len(record) >= 2:
            start_raw, end_raw = record[0], record[1]
            status = "legacy_observed"
        else:
            start_raw = _mapping_value(record, "start_sample", _mapping_value(record, "start", None))
            end_raw = _mapping_value(record, "end_sample", _mapping_value(record, "end", None))
            status = str(_mapping_value(record, "status", "legacy_observed"))
            if status == "acoustic_silence":
                status = "acoustic_silence_supported"
        start = _finite_number(start_raw)
        end = _finite_number(end_raw)
        if start is None or end is None or end <= start:
            continue
        start_i = max(0, min(duration, int(round(start))))
        end_i = max(0, min(duration, int(round(end))))
        if end_i > start_i:
            coverage.append((start_i, end_i, status))
    coverage.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(coverage)


def _uncertain_records(payload: Mapping[str, Any] | None, duration: int) -> tuple[tuple[int, int], ...]:
    """Read optional ASR uncertainty intervals without treating them as gaps."""

    if payload is None:
        return tuple()
    records = _mapping_value(payload, "uncertain_regions", _mapping_value(payload, "uncertain", ())) or ()
    values: list[tuple[int, int]] = []
    for record in records:
        if isinstance(record, (list, tuple)) and len(record) >= 2:
            start_raw, end_raw = record[0], record[1]
        else:
            start_raw = _mapping_value(record, "start_sample", _mapping_value(record, "start", None))
            end_raw = _mapping_value(record, "end_sample", _mapping_value(record, "end", None))
        start = _finite_number(start_raw)
        end = _finite_number(end_raw)
        if start is None or end is None or end <= start:
            continue
        start_i = max(0, min(duration, int(round(start))))
        end_i = max(0, min(duration, int(round(end))))
        if end_i > start_i:
            values.append((start_i, end_i))
    for start_i, end_i, status in _coverage_records(payload, duration):
        if status.startswith("unknown"):
            values.append((start_i, end_i))
    return tuple(sorted(values))


def _phrase_records(payload: Mapping[str, Any] | None, duration: int) -> tuple[tuple[int, int, float | None], ...]:
    if payload is None:
        return tuple()
    records = _mapping_value(payload, "phrases", ()) or ()
    values: list[tuple[int, int, float | None]] = []
    for record in records:
        start = _finite_number(_mapping_value(record, "start_sample", _mapping_value(record, "start", None)))
        end = _finite_number(_mapping_value(record, "end_sample", _mapping_value(record, "end", None)))
        if start is None or end is None or end <= start:
            continue
        start_i = max(0, min(duration, int(round(start))))
        end_i = max(0, min(duration, int(round(end))))
        if end_i > start_i:
            values.append((start_i, end_i, _confidence(_mapping_value(record, "confidence", None))))
    return tuple(sorted(values, key=lambda value: (value[0], value[1], value[2] is None, value[2] or 0.0)))


def _contains(intervals: Sequence[tuple[int, int, str]], start: int, end: int) -> bool:
    return any(left <= start and end <= right for left, right, _ in intervals)


def _shared_acoustic_boundary_supported(
    payload: Mapping[str, Any] | None,
    side: str,
    cue: int,
    fade_samples: int,
    duration: int,
) -> bool:
    """Use the shared boundary auditor when its full schema is available."""

    if payload is None or "coverage_intervals" not in payload:
        return True
    try:
        from .vocal_boundaries import boundary_safety

        document = dict(payload)
        document.setdefault("duration_samples", int(duration))
        report = boundary_safety(
            document,
            side="A_out" if side == "outgoing" else "B_in",
            cue_sample=cue,
            fade_samples=fade_samples,
        )
    except (ImportError, KeyError, TypeError, ValueError):
        return False
    return report.get("status") == "supported"


def _vocal_alignment(
    cue: int,
    side: str,
    duration: int,
    sample_rate: int,
    payload: Mapping[str, Any] | None,
    fade_seconds: float,
    phase_limit_samples: int | None = None,
) -> dict[str, Any]:
    fade_samples = max(1, int(round(fade_seconds * sample_rate)))
    margin = max(0, int(round(_VOCAL_MARGIN_SECONDS * sample_rate)))
    words = _word_records(payload, duration)
    asr_segments = _asr_segment_records(payload, duration)
    coverage = _coverage_records(payload, duration)
    uncertain = _uncertain_records(payload, duration)
    phrases = _phrase_records(payload, duration)
    base: dict[str, Any] = {"requested_sample": cue, "shift_samples": 0, "safe": False}
    if (side == "outgoing" and cue < fade_samples) or (side == "incoming" and cue + fade_samples > duration):
        base.update(status="unsafe_fade_out_of_range", reason="fade_interval_out_of_track")
        return base
    phase_limit = max(0, int(phase_limit_samples if phase_limit_samples is not None else round(sample_rate * _BEAT_PHASE_LIMIT)))

    # A separated-vocal acoustic interval is the one case where an empty ASR
    # span can be promoted to supported.  It must be explicitly labelled by
    # the provider; ASR coverage gaps remain unknown.
    acoustic_candidates: list[tuple[int, int, int, int]] = []
    for start, end, status in coverage:
        if status != "acoustic_silence_supported" or end - start < fade_samples:
            continue
        lower, upper = (start + fade_samples, end) if side == "outgoing" else (start, end - fade_samples)
        proposed = max(lower, min(upper, cue))
        distance = abs(proposed - cue)
        if distance <= phase_limit:
            fade_start, fade_end = (proposed - fade_samples, proposed) if side == "outgoing" else (proposed, proposed + fade_samples)
            if not any(word_start < fade_end and word_end > fade_start for word_start, word_end, _ in words) and not any(
                uncertain_start < fade_end and uncertain_end > fade_start for uncertain_start, uncertain_end in uncertain
            ) and _shared_acoustic_boundary_supported(payload, side, proposed, fade_samples, duration):
                acoustic_candidates.append((distance, proposed, start, end))
    if acoustic_candidates:
        distance, proposed, gap_start, gap_end = min(acoustic_candidates, key=lambda item: (item[0], item[1], item[2], item[3]))
        base.update(
            status="safe_acoustic_silence",
            safe=True,
            shifted_sample=proposed,
            shift_samples=proposed - cue,
            gap_start=gap_start,
            gap_end=gap_end,
            evidence="acoustic_silence_supported",
            phrase_gap_preferred=False,
            phrase_gap_confidence=None,
            boundary_class="acoustic_boundary_supported",
        )
        return base
    if payload is None:
        base.update(status="unknown_no_vocal_input", reason="vocal_evidence_unavailable")
        return base
    if not coverage:
        base.update(status="unknown_coverage", reason="missing_coverage_is_not_silence")
        return base
    # Every accepted gap is bounded by two reasonably confident words in one
    # explicitly observed ASR segment.  This is temporal transcript evidence,
    # not measured acoustic silence.  Legacy payloads without ASR segments may
    # use one ``legacy_observed``/``asr_observed`` coverage row as a fallback;
    # occupancy and unknown rows are deliberately never promoted.
    gaps: list[tuple[int, int, str]] = []
    for previous, following in zip(words, words[1:]):
        p_start, p_end, p_conf = previous
        f_start, f_end, f_conf = following
        if p_conf is None or f_conf is None or p_conf < _VOCAL_MIN_CONFIDENCE or f_conf < _VOCAL_MIN_CONFIDENCE:
            continue
        observed_segment = next(
            (
                (left, right)
                for left, right, _segment_confidence in asr_segments
                if left <= p_end and f_start <= right
            ),
            None,
        )
        evidence_type = "asr_word_gap" if observed_segment is not None else None
        if observed_segment is None:
            # Compatibility for the older compact schema.  Requiring one row
            # to contain both word endpoints prevents stitched occupancy rows
            # from masquerading as one continuous ASR observation.
            legacy_interval = next(
                (
                    (left, right)
                    for left, right, status in coverage
                    if status in {"legacy_observed", "asr_observed"}
                    and left <= p_end
                    and f_start <= right
                ),
                None,
            )
            if legacy_interval is not None:
                observed_segment = legacy_interval
                evidence_type = "legacy_asr_coverage"
        if observed_segment is None:
            continue
        start = p_end + margin
        end = f_start - margin
        if end > start and not any(start < uncertain_end and uncertain_start < end for uncertain_start, uncertain_end in uncertain):
            gaps.append((start, end, str(evidence_type)))
    if not gaps:
        if not words and (asr_segments or any(status == "asr_observed" for _, _, status in coverage)):
            base.update(status="unknown_asr_gap", reason="asr_gap_without_acoustic_silence_evidence")
            return base
        base.update(status="unsafe_no_confirmed_internal_gap", reason="no_confident_gap_inside_coverage")
        return base
    # The renderer protects [out-fade,out] and [in,in+fade].  A candidate may
    # move only within half of the local beat phase; this is a timing aid, not
    # an implicit time stretch.
    nearest: tuple[int, int, int, int, str] | None = None
    for start, end, evidence_type in gaps:
        if end - start < fade_samples:
            continue
        if side == "outgoing":
            lower, upper = start + fade_samples, end
        else:
            lower, upper = start, end - fade_samples
        proposed = max(lower, min(upper, cue))
        distance = abs(proposed - cue)
        key = (distance, proposed, start, end)
        if nearest is None or key < nearest[:4]:
            nearest = (distance, proposed, start, end, evidence_type)
    if nearest is None:
        base.update(status="unsafe_gap_too_short", reason="confirmed_gaps_shorter_than_fade")
        return base
    distance, proposed, gap_start, nearest_end, evidence_type = nearest
    # Derive a local beat phase cap from the supplied coverage only when no
    # structure beats are available here.  The caller further limits shifts
    # with the candidate's context interval.
    if distance > phase_limit:
        base.update(status="unsafe_shift_exceeds_phase", reason="nearest_safe_gap_exceeds_beat_phase_limit")
        return base
    phrase_confidences: list[float] = []
    for left_phrase, right_phrase in zip(phrases, phrases[1:]):
        left_start, left_end, left_confidence = left_phrase
        right_start, right_end, right_confidence = right_phrase
        # A gap crossing the end of one phrase and start of the next is a
        # phrase boundary.  Gaps wholly inside one phrase remain word-only;
        # ASR-derived phrase labels are never treated as musical section
        # labels.
        if gap_start <= right_start and nearest_end >= left_end:
            if left_confidence is not None and right_confidence is not None:
                phrase_confidences.append(min(left_confidence, right_confidence))
    boundary_class = "phrase_boundary_supported" if phrase_confidences else "word_gap_only"
    base.update(
        status="safe_confirmed_gap",
        safe=True,
        shifted_sample=proposed,
        shift_samples=proposed - cue,
        gap_start=gap_start,
        gap_end=nearest_end,
        uncertain_regions=len(uncertain),
        evidence_type=evidence_type,
        evidence="ASR word timing; no acoustic silence claim",
        phrase_gap_preferred=bool(phrase_confidences),
        phrase_gap_confidence=max(phrase_confidences) if phrase_confidences else None,
        boundary_class=boundary_class,
    )
    return base


def _candidate_id(
    track_a: str,
    track_b: str,
    out_sample: int,
    in_sample: int,
    window_a: int,
    window_b: int,
    cue_source_a: str = "unknown",
    cue_source_b: str = "unknown",
) -> str:
    payload = (
        f"{ALGORITHM_VERSION}|conservative-vocal-v1|{track_a}|{track_b}|"
        f"{out_sample}|{in_sample}|{window_a}|{window_b}|{cue_source_a}|{cue_source_b}|structure"
    )
    return "dir-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _bounded_pairs(left: Sequence[_CueContext], right: Sequence[_CueContext], max_candidates: int) -> tuple[tuple[_CueContext, _CueContext], ...]:
    # Keep semantic units aligned.  A four-beat outgoing context is not
    # silently compared with an eight-beat incoming context.
    grouped: list[tuple[_CueContext, _CueContext]] = []
    for window in _WINDOWS:
        left_window = [context for context in left if context.window_beats == window]
        right_window = [context for context in right if context.window_beats == window]
        grouped.extend((a, b) for a in left_window for b in right_window)
    pairs = len(grouped)
    if pairs <= max_candidates:
        return tuple(grouped)
    # Evenly sample the chronological Cartesian grid.  This keeps early and
    # late sections represented while preserving deterministic order.
    selected: list[tuple[_CueContext, _CueContext]] = []
    for index in range(max_candidates):
        flat = (index * (pairs - 1)) // max(1, max_candidates - 1)
        selected.append(grouped[flat])
    return tuple(selected)


def _position_zone(sample: int, duration: int, sample_rate: int, edge_seconds: float) -> str:
    edge = max(0, int(round(edge_seconds * sample_rate)))
    if sample < edge:
        return "intro"
    if sample >= max(0, duration - edge):
        return "outro"
    return "middle"


def _cue_support_records(
    events: Sequence[_Event],
    contexts: Sequence[_CueContext],
    duration: int,
    sample_rate: int,
    edge_seconds: float,
) -> tuple[Mapping[str, Any], ...]:
    """Describe what reached the context pre-grid, without claiming vocal safety."""

    records: list[Mapping[str, Any]] = []
    for event in events:
        available = sorted({context.window_beats for context in contexts if context.sample == event.sample})
        records.append(
            {
                "cue_sample": event.sample,
                "cue_seconds": event.sample / float(sample_rate),
                "source": event.source,
                "confidence": event.confidence,
                "position_zone": _position_zone(event.sample, duration, sample_rate, edge_seconds),
                "context_windows_available": available,
                "pregrid_eligible": bool(available),
            }
        )
    return tuple(records)


def _vocal_input_summary(payload: Mapping[str, Any] | None, duration: int) -> Mapping[str, Any]:
    if payload is None:
        return {"provided": False, "status": "unknown_no_vocal_input"}
    words = _word_records(payload, duration)
    segments = _asr_segment_records(payload, duration)
    coverage = _coverage_records(payload, duration)
    unknown = _uncertain_records(payload, duration)
    if not words and not segments and not coverage:
        status = "unknown_coverage"
    elif unknown:
        status = "observed_with_unknown_intervals"
    elif segments or words:
        status = "asr_temporal_evidence_available"
    else:
        status = "coverage_present"
    return {
        "provided": True,
        "status": status,
        "word_count": len(words),
        "asr_segment_count": len(segments),
        "coverage_interval_count": len(coverage),
        "unknown_interval_count": len(unknown),
    }


def _context_confidence(context: _CueContext) -> float | None:
    # Null structure confidence remains null; a fabricated midpoint would
    # make an unknown cue appear to have measured confidence.
    return context.confidence


def _translate_context(
    track: AudioTrack,
    context: _CueContext,
    sample: int,
    descriptor_cache: dict[tuple[int, int], tuple[float, tuple[float, ...], tuple[float, ...]]] | None = None,
) -> _CueContext:
    """Re-extract descriptors after a bounded vocal-safe cue adjustment."""

    delta = int(sample) - context.sample
    segments = tuple(
        (max(0, min(track.duration_samples, start + delta)), max(0, min(track.duration_samples, end + delta)))
        for start, end in context.segment_bounds
    )
    segments = tuple((start, end) for start, end in segments if end > start)
    if len(segments) != len(context.segment_bounds):
        return context
    rms_values: list[float] = []
    chroma_values: list[tuple[float, ...]] = []
    timbre_values: list[tuple[float, ...]] = []
    for start, end in segments:
        key = (int(start), int(end))
        if descriptor_cache is not None and key in descriptor_cache:
            rms, chroma, timbre = descriptor_cache[key]
        else:
            rms, chroma, timbre = _audio_descriptor(track, start, end)
            if descriptor_cache is not None:
                descriptor_cache[key] = (rms, chroma, timbre)
        rms_values.append(rms)
        chroma_values.append(chroma)
        timbre_values.append(timbre)
    timbre_array = np.asarray([value for value in timbre_values if value], dtype=float)
    timbre = tuple(float(value) for value in np.mean(timbre_array, axis=0)) if timbre_array.size else tuple()
    log_rms = np.log(np.maximum(np.asarray(rms_values, dtype=float), 1e-8))
    trend = float(np.polyfit(np.arange(len(log_rms), dtype=float), log_rms, 1)[0]) if len(log_rms) >= 2 else None
    return _CueContext(
        sample=int(sample),
        confidence=context.confidence,
        cue_source=context.cue_source,
        window_beats=context.window_beats,
        start_sample=segments[0][0],
        end_sample=segments[-1][1],
        segment_bounds=segments,
        tempo_bpm=context.tempo_bpm,
        tempo_spread=context.tempo_spread,
        rms=tuple(rms_values),
        chroma=tuple(chroma_values),
        timbre=timbre,
        trend=trend,
        direction=context.direction,
        beat_offset_samples=context.beat_offset_samples + delta,
    )


def _proposal_for(
    track_a: AudioTrack,
    track_b: AudioTrack,
    context_a: _CueContext,
    context_b: _CueContext,
    vocals_a: Mapping[str, Any] | None,
    vocals_b: Mapping[str, Any] | None,
    fade_seconds: float,
    max_shift_seconds: float,
    allow_unknown_vocals: bool,
    descriptor_cache_a: dict[tuple[int, int], tuple[float, tuple[float, ...], tuple[float, ...]]] | None = None,
    descriptor_cache_b: dict[tuple[int, int], tuple[float, tuple[float, ...], tuple[float, ...]]] | None = None,
    vocal_cache_a: dict[tuple[int, int, int], dict[str, Any]] | None = None,
    vocal_cache_b: dict[tuple[int, int, int], dict[str, Any]] | None = None,
    min_context_samples_a: int = 0,
    min_context_samples_b: int = 0,
) -> tuple[DirectionalProposal | None, Mapping[str, Any]]:
    def make_candidate_id(out_sample: int, in_sample: int) -> str:
        return _candidate_id(
            track_a.track_id,
            track_b.track_id,
            out_sample,
            in_sample,
            context_a.window_beats,
            context_b.window_beats,
            context_a.cue_source,
            context_b.cue_source,
        )

    phase_a = min(
        int(round(max_shift_seconds * track_a.sample_rate)),
        int(round(0.5 * float(np.median(np.diff(np.asarray([bound[0] for bound in context_a.segment_bounds] + [context_a.segment_bounds[-1][1]], dtype=float)))))) if context_a.segment_bounds else 0,
    )
    phase_b = min(
        int(round(max_shift_seconds * track_b.sample_rate)),
        int(round(0.5 * float(np.median(np.diff(np.asarray([bound[0] for bound in context_b.segment_bounds] + [context_b.segment_bounds[-1][1]], dtype=float)))))) if context_b.segment_bounds else 0,
    )
    fade_samples_a = int(round(fade_seconds * track_a.sample_rate))
    fade_samples_b = int(round(fade_seconds * track_b.sample_rate))
    key_a = (context_a.sample, phase_a, fade_samples_a)
    key_b = (context_b.sample, phase_b, fade_samples_b)
    if vocal_cache_a is not None and key_a in vocal_cache_a:
        vocal_a = dict(vocal_cache_a[key_a])
    else:
        vocal_a = _vocal_alignment(context_a.sample, "outgoing", track_a.duration_samples, track_a.sample_rate, vocals_a, fade_seconds, phase_a)
        if vocal_cache_a is not None:
            vocal_cache_a[key_a] = dict(vocal_a)
    if vocal_cache_b is not None and key_b in vocal_cache_b:
        vocal_b = dict(vocal_cache_b[key_b])
    else:
        vocal_b = _vocal_alignment(context_b.sample, "incoming", track_b.duration_samples, track_b.sample_rate, vocals_b, fade_seconds, phase_b)
        if vocal_cache_b is not None:
            vocal_cache_b[key_b] = dict(vocal_b)
    known_unsafe = [entry for entry in (vocal_a, vocal_b) if str(entry.get("status", "")).startswith("unsafe")]
    alignment_unknown = [entry for entry in (vocal_a, vocal_b) if str(entry.get("status", "")).startswith("unknown")]
    reasons: list[str] = []
    if known_unsafe:
        reasons.extend(str(entry.get("reason", "vocal_boundary_rejected")) for entry in known_unsafe)
    if alignment_unknown:
        reasons.extend(str(entry.get("reason", "vocal_boundary_unknown")) for entry in alignment_unknown)
    if any(entry.get("boundary_class") == "word_gap_only" for entry in (vocal_a, vocal_b)):
        reasons.append("word_gap_only_secondary_to_phrase_boundary")
    safe_count = sum(bool(entry.get("safe")) for entry in (vocal_a, vocal_b))
    if known_unsafe:
        candidate_id = make_candidate_id(context_a.sample, context_b.sample)
        return None, {
            "candidate_id": candidate_id,
            "reason": "vocal_boundary_rejected",
            "reasons": tuple(reasons),
            "alignment": {"outgoing": vocal_a, "incoming": vocal_b},
        }
    if alignment_unknown and not allow_unknown_vocals:
        candidate_id = make_candidate_id(context_a.sample, context_b.sample)
        return None, {
            "candidate_id": candidate_id,
            "reason": "vocal_safety_unknown",
            "reasons": tuple(reasons),
            "alignment": {"outgoing": vocal_a, "incoming": vocal_b},
        }
    out_sample = int(vocal_a.get("shifted_sample", context_a.sample))
    in_sample = int(vocal_b.get("shifted_sample", context_b.sample))
    if out_sample < min_context_samples_a or in_sample + min_context_samples_b > track_b.duration_samples:
        candidate_id = make_candidate_id(out_sample, in_sample)
        return None, {
            "candidate_id": candidate_id,
            "reason": "outside_listening_context_after_vocal_shift",
            "reasons": ("vocal_safe_shift_crosses_listening_context",),
            "alignment": {"outgoing": vocal_a, "incoming": vocal_b},
        }
    relative_shift_seconds = abs(
        (out_sample - context_a.sample) / float(track_a.sample_rate)
        - (in_sample - context_b.sample) / float(track_b.sample_rate)
    )
    if relative_shift_seconds > _COMBINED_SHIFT_LIMIT_SECONDS:
        candidate_id = make_candidate_id(out_sample, in_sample)
        return None, {
            "candidate_id": candidate_id,
            "reason": "combined_vocal_shift_exceeds_phase",
            "reasons": ("relative_boundary_shift_exceeds_80ms_policy",),
            "alignment": {
                "outgoing": vocal_a,
                "incoming": vocal_b,
                "relative_shift_seconds": relative_shift_seconds,
                "combined_shift_limit_seconds": _COMBINED_SHIFT_LIMIT_SECONDS,
            },
        }
    provided_phase_seconds = abs(
        (context_a.beat_offset_samples + out_sample - context_a.sample) / float(track_a.sample_rate)
        - (context_b.beat_offset_samples + in_sample - context_b.sample) / float(track_b.sample_rate)
    )
    if (context_a.cue_source == "provided_cue" or context_b.cue_source == "provided_cue") and provided_phase_seconds > _COMBINED_SHIFT_LIMIT_SECONDS:
        return None, {
            "candidate_id": make_candidate_id(out_sample, in_sample),
            "reason": "provided_cue_phase_mismatch",
            "reasons": ("model_cue_relative_beat_phase_exceeds_80ms_policy",),
            "alignment": {"outgoing": vocal_a, "incoming": vocal_b,
                          "relative_beat_phase_seconds": provided_phase_seconds,
                          "combined_shift_limit_seconds": _COMBINED_SHIFT_LIMIT_SECONDS},
        }
    scored_a = _translate_context(track_a, context_a, out_sample, descriptor_cache_a) if out_sample != context_a.sample else context_a
    scored_b = _translate_context(track_b, context_b, in_sample, descriptor_cache_b) if in_sample != context_b.sample else context_b
    harmony = _sequence_similarity(scored_a.chroma, scored_b.chroma)
    timbre = _cosine(scored_a.timbre, scored_b.timbre)
    rms_level = _scalar_similarity(
        math.log(max(float(np.mean(scored_a.rms)), 1e-8)),
        math.log(max(float(np.mean(scored_b.rms)), 1e-8)),
        1.25,
    )
    trend = _trend_score(scored_a.trend, scored_b.trend)
    tempo, tempo_log2, tempo_octave = _tempo_score(scored_a.tempo_bpm, scored_b.tempo_bpm)
    values = {
        "harmony_context": harmony,
        "timbre_context": timbre,
        "rms_level": rms_level,
        "rms_trend": trend,
        "tempo_compatibility": tempo,
    }
    available = [(name, value) for name, value in values.items() if value is not None]
    base = float(np.mean([value for _, value in available])) if available else 0.0
    # Unknown evidence remains visible with an explicit penalty only when the
    # caller opted into exploratory ranking.
    phrase_count = sum(bool(entry.get("phrase_gap_preferred")) for entry in (vocal_a, vocal_b))
    vocal_score = 1.0 if safe_count == 2 and phrase_count == 2 else (0.97 if safe_count == 2 else (0.82 if safe_count == 1 else 0.70))
    score = max(0.0, min(1.0, 0.82 * base + 0.18 * vocal_score))
    candidate_id = make_candidate_id(out_sample, in_sample)
    confidence_values = [_context_confidence(context_a), _context_confidence(context_b)]
    known_confidence = [value for value in confidence_values if value is not None]
    confidence = min(known_confidence) if known_confidence else None
    feature_values = {
        "horizontal": harmony,
        "continuity": trend if trend is not None else rms_level,
        "vertical": None,
        "creative": None,
    }
    features: dict[str, Feature] = {}
    for name, value in feature_values.items():
        features[name] = Feature(value=value, confidence=(confidence if value is not None else None), source="directional-v1")
    penalties: dict[str, Feature] = {}
    if tempo is not None:
        penalties["transform"] = Feature(value=max(0.0, min(1.0, 1.0 - tempo)), confidence=confidence, source="local_tempo")
    candidate = Candidate(
        candidate_id=candidate_id,
        track_a=track_a.track_id,
        track_b=track_b.track_id,
        cue_out_sample=out_sample,
        cue_in_sample=in_sample,
        mode="structure",
        features=features,
        penalties=penalties,
        passes_constraints=not bool(alignment_unknown),
        constraint_reasons=tuple(reasons),
    )
    breakdown = {name: float(value) for name, value in values.items() if value is not None}
    breakdown.update(vocal_safety=vocal_score, base_context=base)
    raw_features: dict[str, Any] = {
        "window_beats": context_a.window_beats,
        "cue_source_a": context_a.cue_source,
        "cue_source_b": context_b.cue_source,
        "feature_context_samples": {"a": scored_a.sample, "b": scored_b.sample},
        "original_structure_cue_samples": {"a": context_a.sample, "b": context_b.sample},
        "structure_confidence_a": context_a.confidence,
        "structure_confidence_b": context_b.confidence,
        "direction": "outgoing_then_incoming",
        "a_chroma_sequence": [list(row) for row in scored_a.chroma],
        "b_chroma_sequence": [list(row) for row in scored_b.chroma],
        "a_rms": list(scored_a.rms),
        "b_rms": list(scored_b.rms),
        "a_rms_trend": scored_a.trend,
        "b_rms_trend": scored_b.trend,
        "tempo_a_bpm": scored_a.tempo_bpm,
        "tempo_b_bpm": scored_b.tempo_bpm,
        "tempo_a_interval_spread": scored_a.tempo_spread,
        "tempo_b_interval_spread": scored_b.tempo_spread,
        "tempo_log2_distance": tempo_log2,
        "tempo_octave_distance": tempo_octave,
        "boundary_class": {
            "outgoing": vocal_a.get("boundary_class", "unknown"),
            "incoming": vocal_b.get("boundary_class", "unknown"),
        },
        "pitch_shift_applied": False,
        "time_stretch_applied": False,
    }
    alignment = {
        "outgoing": vocal_a,
        "incoming": vocal_b,
        "fade_interval_a": [out_sample - int(round(fade_seconds * track_a.sample_rate)), out_sample],
        "fade_interval_b": [in_sample, in_sample + int(round(fade_seconds * track_b.sample_rate))],
        "vocal_policy": "conservative_confirmed_internal_gaps",
        "cue_source": {"a": context_a.cue_source, "b": context_b.cue_source},
        "boundary_class": {
            "outgoing": vocal_a.get("boundary_class", "unknown"),
            "incoming": vocal_b.get("boundary_class", "unknown"),
        },
        "beat_phase_limit": _BEAT_PHASE_LIMIT,
        "max_shift_seconds": max_shift_seconds,
        "relative_shift_seconds": relative_shift_seconds,
        "relative_beat_phase_seconds": provided_phase_seconds,
        "nominal_beat_offset_samples": {"a": context_a.beat_offset_samples, "b": context_b.beat_offset_samples},
        "combined_shift_limit_seconds": _COMBINED_SHIFT_LIMIT_SECONDS,
    }
    proposal = DirectionalProposal(candidate, score, breakdown, raw_features, fade_seconds, alignment, tuple(reasons))
    return proposal, {"candidate_id": candidate_id, "reason": "accepted", "alignment": alignment}


def search_directional(
    track_a: AudioTrack,
    track_b: AudioTrack,
    structure_a: Mapping[str, Any],
    structure_b: Mapping[str, Any],
    vocals_a: Mapping[str, Any] | None = None,
    vocals_b: Mapping[str, Any] | None = None,
    top_k: int = 5,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    fade_seconds: float = _TARGET_FADE_SECONDS,
    max_shift_seconds: float = _BEAT_PHASE_LIMIT,
    allow_unknown_vocals: bool = False,
    min_context_seconds: float = 10.0,
    min_downbeat_confidence: float = 0.20,
    cue_policy: str = "downbeats",
    min_cue_spacing_seconds: float = 8.0,
    prefilter_vocals: bool = False,
) -> DirectionalSearchResult:
    """Rank outgoing A → incoming B multi-beat transition proposals.

    ``structure_*`` and ``vocals_*`` are JSON-compatible mappings.  Structure
    sample positions remain in each track's own sample-rate timeline; no
    silent resampling or time-stretch assumption is made.  ``top_k`` and
    ``max_candidates`` must be positive.  By default cues need ten seconds of
    listening context on the relevant side and both vocal boundaries must be
    confirmed; ``allow_unknown_vocals=True`` is an explicitly exploratory
    mode.  ``cue_policy="downbeats"`` preserves the primary anchor policy;
    ``cue_policy="beats"`` evaluates every observed beat and records that
    provenance.  Selected proposals use each original cue once and are
    separated on both timelines by ``min_cue_spacing_seconds``.  Two tracks
    with the same identifier are rejected because their direction cannot be
    audited independently.
    """

    if not isinstance(track_a, AudioTrack) or not isinstance(track_b, AudioTrack):
        raise TypeError("track_a and track_b must be AudioTrack instances")
    if track_a.track_id == track_b.track_id:
        raise ValueError("directional search requires distinct track identifiers")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(max_candidates, int) or max_candidates <= 0:
        raise ValueError("max_candidates must be a positive integer")
    if not math.isfinite(float(fade_seconds)) or float(fade_seconds) <= 0:
        raise ValueError("fade_seconds must be positive and finite")
    if not math.isfinite(float(max_shift_seconds)) or float(max_shift_seconds) < 0:
        raise ValueError("max_shift_seconds must be non-negative and finite")
    if not math.isfinite(float(min_context_seconds)) or float(min_context_seconds) < 0:
        raise ValueError("min_context_seconds must be non-negative and finite")
    if not math.isfinite(float(min_downbeat_confidence)) or not 0.0 <= float(min_downbeat_confidence) <= 1.0:
        raise ValueError("min_downbeat_confidence must be between 0 and 1")
    if cue_policy not in {"downbeats", "beats", "provided"}:
        raise ValueError("cue_policy must be 'downbeats', 'beats', or 'provided'")
    if not math.isfinite(float(min_cue_spacing_seconds)) or float(min_cue_spacing_seconds) < 0:
        raise ValueError("min_cue_spacing_seconds must be non-negative and finite")
    if not isinstance(structure_a, Mapping) or not isinstance(structure_b, Mapping):
        raise TypeError("structure_a and structure_b must be mappings")

    for label, track, structure in (("a", track_a, structure_a), ("b", track_b, structure_b)):
        declared_track_id = _mapping_value(structure, "track_id", None)
        if declared_track_id is not None and str(declared_track_id) != track.track_id:
            raise ValueError(f"structure_{label} track_id does not match AudioTrack")
        declared_rate = _finite_number(_mapping_value(structure, "sample_rate", track.sample_rate))
        if declared_rate is not None and int(round(declared_rate)) != track.sample_rate:
            raise ValueError(f"structure_{label} sample_rate does not match AudioTrack")
        declared_duration = _finite_number(_mapping_value(structure, "duration_samples", track.duration_samples))
        if declared_duration is not None and int(round(declared_duration)) != track.duration_samples:
            raise ValueError(f"structure_{label} duration_samples does not match AudioTrack")

    cues_a, source_a, low_downbeats_a = _cue_events(
        structure_a,
        track_a.duration_samples,
        float(min_downbeat_confidence),
        cue_policy,
        role="outgoing",
    )
    cues_b, source_b, low_downbeats_b = _cue_events(
        structure_b,
        track_b.duration_samples,
        float(min_downbeat_confidence),
        cue_policy,
        role="incoming",
    )
    all_cues_a = tuple(cues_a)
    all_cues_b = tuple(cues_b)
    min_context_a = int(round(float(min_context_seconds) * track_a.sample_rate))
    min_context_b = int(round(float(min_context_seconds) * track_b.sample_rate))
    filtered_rejected: list[Mapping[str, Any]] = []
    for side, low_events in (("a", low_downbeats_a), ("b", low_downbeats_b)):
        if cue_policy == "beats":
            continue
        for event in low_events:
            filtered_rejected.append({
                "reason": "low_downbeat_confidence",
                "side": side,
                "cue_sample": event.sample,
                "confidence": event.confidence,
                "threshold": float(min_downbeat_confidence),
            })
    eligible_a: list[_Event] = []
    eligible_b: list[_Event] = []
    for cue in cues_a:
        if cue.sample < min_context_a:
            filtered_rejected.append({"reason": "outside_listening_context", "side": "a", "cue_sample": cue.sample, "required_before_samples": min_context_a})
        else:
            eligible_a.append(cue)
    for cue in cues_b:
        if cue.sample + min_context_b > track_b.duration_samples:
            filtered_rejected.append({"reason": "outside_listening_context", "side": "b", "cue_sample": cue.sample, "required_after_samples": min_context_b})
        else:
            eligible_b.append(cue)
    cues_a = tuple(eligible_a)
    cues_b = tuple(eligible_b)
    beats_a = _samples_for_beats(structure_a, track_a.duration_samples) or tuple(event.sample for event in cues_a)
    beats_b = _samples_for_beats(structure_b, track_b.duration_samples) or tuple(event.sample for event in cues_b)
    contexts_a: list[_CueContext] = []
    contexts_b: list[_CueContext] = []
    descriptor_cache_a: dict[tuple[int, int], tuple[float, tuple[float, ...], tuple[float, ...]]] = {}
    descriptor_cache_b: dict[tuple[int, int], tuple[float, tuple[float, ...], tuple[float, ...]]] = {}
    vocal_cache_a: dict[tuple[int, int, int], dict[str, Any]] = {}
    vocal_cache_b: dict[tuple[int, int, int], dict[str, Any]] = {}
    for cue in cues_a:
        for window in _WINDOWS:
            context = _context_for(
                track_a,
                beats_a,
                cue,
                window,
                "outgoing",
                cue_source=cue.source,
                descriptor_cache=descriptor_cache_a,
            )
            if context is not None:
                contexts_a.append(context)
    for cue in cues_b:
        for window in _WINDOWS:
            context = _context_for(
                track_b,
                beats_b,
                cue,
                window,
                "incoming",
                cue_source=cue.source,
                descriptor_cache=descriptor_cache_b,
            )
            if context is not None:
                contexts_b.append(context)
    contexts_a.sort(key=lambda context: (context.sample, context.window_beats))
    contexts_b.sort(key=lambda context: (context.sample, context.window_beats))
    original_contexts_a = tuple(contexts_a)
    original_contexts_b = tuple(contexts_b)
    original_grid_size = sum(
        sum(c.window_beats == window for c in contexts_a)
        * sum(c.window_beats == window for c in contexts_b) for window in _WINDOWS
    )
    # This is an exact optimization only for an exhaustive original grid.
    # Pruning before deterministic subsampling could otherwise change which
    # candidates are sampled. In that case retain the original algorithm.
    prefilter_applied = bool(prefilter_vocals and original_grid_size <= max_candidates)
    prefilter_counts = {"a": 0, "b": 0}
    if prefilter_applied:
        def retain_contexts(track, contexts, payload, side, cache):
            retained = []
            for context in contexts:
                bounds = [bound[0] for bound in context.segment_bounds] + [context.segment_bounds[-1][1]]
                phase = min(int(round(max_shift_seconds * track.sample_rate)),
                            int(round(0.5 * float(np.median(np.diff(np.asarray(bounds, dtype=float)))))))
                key = (context.sample, phase, int(round(fade_seconds * track.sample_rate)))
                if key not in cache:
                    cache[key] = _vocal_alignment(context.sample, side, track.duration_samples,
                                                  track.sample_rate, payload, fade_seconds, phase)
                status = str(cache[key].get("status", ""))
                rejected_here = status.startswith("unsafe") or (status.startswith("unknown") and not allow_unknown_vocals)
                if not rejected_here:
                    retained.append(context)
            return retained
        contexts_a = retain_contexts(track_a, contexts_a, vocals_a, "outgoing", vocal_cache_a)
        contexts_b = retain_contexts(track_b, contexts_b, vocals_b, "incoming", vocal_cache_b)
        prefilter_counts = {"a": len(original_contexts_a)-len(contexts_a),
                            "b": len(original_contexts_b)-len(contexts_b)}
    filtered_grid_size = sum(
        sum(c.window_beats == window for c in contexts_a)
        * sum(c.window_beats == window for c in contexts_b) for window in _WINDOWS
    )
    pairs = _bounded_pairs(contexts_a, contexts_b, max_candidates)
    proposals: list[DirectionalProposal] = []
    rejected: list[Mapping[str, Any]] = list(filtered_rejected)
    for context_a, context_b in pairs:
        proposal, record = _proposal_for(
            track_a,
            track_b,
            context_a,
            context_b,
            vocals_a,
            vocals_b,
            float(fade_seconds),
            float(max_shift_seconds),
            bool(allow_unknown_vocals),
            descriptor_cache_a,
            descriptor_cache_b,
            vocal_cache_a,
            vocal_cache_b,
            min_context_a,
            min_context_b,
        )
        if proposal is None:
            rejected.append(record)
        else:
            proposals.append(proposal)
    def _boundary_priority(proposal: DirectionalProposal) -> int:
        classes = proposal.alignment.get("boundary_class", {})
        preferred = {"phrase_boundary_supported", "acoustic_boundary_supported"}
        if classes.get("outgoing") in preferred and classes.get("incoming") in preferred:
            return 2
        if any(value in preferred for value in classes.values()):
            return 1
        return 0

    # Confirmed phrase boundaries and explicit instrumental silence are the
    # preferred surface.  Word-gap-only proposals remain available as a
    # labelled secondary tier when no preferred pair exists.
    proposals.sort(key=lambda proposal: (-_boundary_priority(proposal), -proposal.score, proposal.candidate.candidate_id))
    selected: list[DirectionalProposal] = []
    selected_pairs: set[tuple[int, int]] = set()
    selected_out_anchors: list[int] = []
    selected_in_anchors: list[int] = []
    spacing_a = int(round(float(min_cue_spacing_seconds) * track_a.sample_rate))
    spacing_b = int(round(float(min_cue_spacing_seconds) * track_b.sample_rate))
    for proposal in proposals:
        original = proposal.raw_features.get("original_structure_cue_samples", {})
        out_anchor = int(original.get("a", proposal.candidate.cue_out_sample))
        in_anchor = int(original.get("b", proposal.candidate.cue_in_sample))
        pair = (out_anchor, in_anchor)
        if pair in selected_pairs:
            rejected.append(
                {
                    "candidate_id": proposal.candidate.candidate_id,
                    "reason": "top_k_cue_reuse",
                    "cue_out_sample": out_anchor,
                    "cue_in_sample": in_anchor,
                    "reused_sides": ["a" if out_anchor in selected_out_anchors else "b"],
                }
            )
            continue
        reused_sides: list[str] = []
        if out_anchor in selected_out_anchors:
            reused_sides.append("a")
        if in_anchor in selected_in_anchors:
            reused_sides.append("b")
        if reused_sides:
            rejected.append(
                {
                    "candidate_id": proposal.candidate.candidate_id,
                    "reason": "top_k_cue_reuse",
                    "cue_out_sample": out_anchor,
                    "cue_in_sample": in_anchor,
                    "reused_sides": reused_sides,
                }
            )
            continue
        spacing_sides: list[str] = []
        if any(abs(out_anchor - previous) < spacing_a for previous in selected_out_anchors):
            spacing_sides.append("a")
        if any(abs(in_anchor - previous) < spacing_b for previous in selected_in_anchors):
            spacing_sides.append("b")
        if spacing_sides:
            rejected.append(
                {
                    "candidate_id": proposal.candidate.candidate_id,
                    "reason": "top_k_temporal_spacing",
                    "cue_out_sample": out_anchor,
                    "cue_in_sample": in_anchor,
                    "spacing_sides": spacing_sides,
                    "min_cue_spacing_seconds": float(min_cue_spacing_seconds),
                }
            )
            continue
        if len(selected) >= top_k:
            rejected.append({"candidate_id": proposal.candidate.candidate_id, "reason": "below_top_k"})
            continue
        selected_pairs.add(pair)
        selected_out_anchors.append(out_anchor)
        selected_in_anchors.append(in_anchor)
        selected.append(proposal)
    candidate_grid_size = original_grid_size
    edge_seconds = max(10.0, float(min_context_seconds))
    support_a = [dict(record) for record in _cue_support_records(
        all_cues_a,
        original_contexts_a,
        track_a.duration_samples,
        track_a.sample_rate,
        edge_seconds,
    )]
    support_b = [dict(record) for record in _cue_support_records(
        all_cues_b,
        original_contexts_b,
        track_b.duration_samples,
        track_b.sample_rate,
        edge_seconds,
    )]
    for record in support_a:
        if record["cue_sample"] < min_context_a:
            record["pregrid_filter_reason"] = "outside_listening_context_before"
        elif not record["pregrid_eligible"]:
            record["pregrid_filter_reason"] = "no_complete_context_window"
        else:
            record["pregrid_filter_reason"] = None
    for record in support_b:
        if record["cue_sample"] + min_context_b > track_b.duration_samples:
            record["pregrid_filter_reason"] = "outside_listening_context_after"
        elif not record["pregrid_eligible"]:
            record["pregrid_filter_reason"] = "no_complete_context_window"
        else:
            record["pregrid_filter_reason"] = None

    def zone_counts(records: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
        counts = {"intro": 0, "middle": 0, "outro": 0}
        for record in records:
            zone = str(record.get("position_zone", "unknown"))
            counts[zone] = counts.get(zone, 0) + 1
        return counts

    rejection_counts: dict[str, int] = {}
    for record in rejected:
        reason = str(record.get("reason", "unknown"))
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    diagnostics = {
        "algorithm": ALGORITHM_VERSION,
        "policy": "conservative-vocal-v1",
        "direction": "track_a_outgoing_to_track_b_incoming",
        "cue_source_a": source_a,
        "cue_source_b": source_b,
        "cue_policy": cue_policy,
        "cue_provenance": {"a": source_a, "b": source_b},
        "context_windows_beats": list(_WINDOWS),
        "contexts_a": len(original_contexts_a),
        "contexts_b": len(original_contexts_b),
        "vocal_prefilter": {"requested": bool(prefilter_vocals), "applied": prefilter_applied,
                            "disabled_reason": "original_grid_truncated" if prefilter_vocals and not prefilter_applied else None,
                            "rejected_contexts": prefilter_counts,
                            "pruned_candidate_pairs": original_grid_size-filtered_grid_size,
                            "remaining_candidate_grid_size": filtered_grid_size,
                            "audit_semantics": "pruned pairs counted here; no duplicated rejection records materialized"},
        "candidate_grid_size": candidate_grid_size,
        "candidate_count_evaluated": len(pairs),
        "candidate_grid_truncated": candidate_grid_size > max_candidates,
        "proposals_scored": len(proposals),
        "selected_count": len(selected),
        "requested_top_k": int(top_k),
        "selected_count_lt_top_k": len(selected) < top_k,
        "selection_shortfall": max(0, int(top_k) - len(selected)),
        "min_cue_spacing_seconds": float(min_cue_spacing_seconds),
        "cue_reuse_cap": 1,
        "selected_original_cue_anchors": [
            dict(item.raw_features.get("original_structure_cue_samples", {})) for item in selected
        ],
        "selected_position_zones": {
            "a": zone_counts(
                [
                    next(
                        record for record in support_a
                        if record["cue_sample"] == int(item.raw_features.get("original_structure_cue_samples", {}).get("a", -1))
                    )
                    for item in selected
                    if any(
                        record["cue_sample"] == int(item.raw_features.get("original_structure_cue_samples", {}).get("a", -1))
                        for record in support_a
                    )
                ]
            ),
            "b": zone_counts(
                [
                    next(
                        record for record in support_b
                        if record["cue_sample"] == int(item.raw_features.get("original_structure_cue_samples", {}).get("b", -1))
                    )
                    for item in selected
                    if any(
                        record["cue_sample"] == int(item.raw_features.get("original_structure_cue_samples", {}).get("b", -1))
                        for record in support_b
                    )
                ]
            ),
        },
        "position_zone_counts_pregrid": {"a": zone_counts(support_a), "b": zone_counts(support_b)},
        "cue_support_a": support_a,
        "cue_support_b": support_b,
        "pregrid": {
            "cue_count_a": len(all_cues_a),
            "cue_count_b": len(all_cues_b),
            "eligible_cue_count_a": len(cues_a),
            "eligible_cue_count_b": len(cues_b),
            "context_count_a": len(original_contexts_a),
            "context_count_b": len(original_contexts_b),
        },
        "vocal_input_a": dict(_vocal_input_summary(vocals_a, track_a.duration_samples)),
        "vocal_input_b": dict(_vocal_input_summary(vocals_b, track_b.duration_samples)),
        "rejection_reason_counts": rejection_counts,
        "boundary_priority": "phrase_or_acoustic_boundary_supported_then_word_gap_only",
        "selected_boundary_classes": [dict(item.alignment.get("boundary_class", {})) for item in selected],
        "rejected_count": len(rejected),
        "vocal_unknown_is_not_silence": True,
        "allow_unknown_vocals": bool(allow_unknown_vocals),
        "fade_seconds": float(fade_seconds),
        "max_shift_seconds": float(max_shift_seconds),
        "combined_shift_limit_seconds": _COMBINED_SHIFT_LIMIT_SECONDS,
        "min_context_seconds": float(min_context_seconds),
        "min_downbeat_confidence": float(min_downbeat_confidence),
        "low_confidence_downbeats_a": len(low_downbeats_a),
        "low_confidence_downbeats_b": len(low_downbeats_b),
        "listening_context_filter_rejections": len(filtered_rejected),
        "vocal_alignment_cache_entries_a": len(vocal_cache_a),
        "vocal_alignment_cache_entries_b": len(vocal_cache_b),
        "pitch_shift_applied": False,
        "time_stretch_applied": False,
    }
    return DirectionalSearchResult(
        tuple(selected),
        tuple(rejected),
        diagnostics,
        tuple(proposals),
        tuple(proposal.candidate.candidate_id for proposal in selected),
    )


__all__ = ["DirectionalProposal", "DirectionalSearchResult", "search_directional"]
