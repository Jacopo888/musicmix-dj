"""Beat/downbeat and structural-event contracts for MusicMix M02.

The core deliberately knows nothing about a neural provider.  A provider
returns event times in seconds and optional confidences; this module converts
those times to the canonical sample timeline, validates and de-duplicates
events, and records provider provenance.  Provider-specific imports belong in
``tools/analyze_structure.py`` (or another optional adapter), so importing the
package does not require PyTorch or model weights.

The bundled provider adapter currently targets CPJKU's Beat This! checkpoint.
Its published model predicts beats and downbeats, but does not provide song
section labels.  The optional sections emitted here are therefore explicitly
labelled ``heuristic_feature_segment`` and never claim chorus, verse, phrase,
or downbeat semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from .audio import AudioTrack


STRUCTURE_SCHEMA_VERSION = "musicmix.structure.v2"
BEAT_THIS_REPOSITORY = "https://github.com/CPJKU/beat_this"
BEAT_THIS_PAPER = (
    "https://archives.ismir.net/ismir2024/paper/000107.pdf"
)
BEAT_THIS_CHECKPOINT_ROOT = (
    "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp"
)
ALL_IN_ONE_PAPER = "https://arxiv.org/abs/2307.16425"
HEURISTIC_SECTION_LABEL = "heuristic_feature_segment"


@dataclass(frozen=True)
class TimedEvent:
    """A provider event in seconds, with an optional provider confidence."""

    time_seconds: float
    confidence: float | None = None

    def __post_init__(self) -> None:
        time_seconds = float(self.time_seconds)
        if not math.isfinite(time_seconds) or time_seconds < 0.0:
            raise ValueError("event time_seconds must be finite and non-negative")
        confidence = _optional_confidence(self.confidence)
        object.__setattr__(self, "time_seconds", time_seconds)
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class TimedSection:
    """A provider section in seconds (labels remain provider-owned)."""

    start_seconds: float
    end_seconds: float
    label: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        start = float(self.start_seconds)
        end = float(self.end_seconds)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end <= start:
            raise ValueError("section times must be finite and satisfy 0 <= start < end")
        label = str(self.label).strip()
        if not label:
            raise ValueError("section label must be non-empty")
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "confidence", _optional_confidence(self.confidence))


@dataclass(frozen=True)
class ProviderPrediction:
    """Provider-neutral prediction payload consumed by :func:`analyze_structure`."""

    beats: tuple[TimedEvent, ...] = ()
    downbeats: tuple[TimedEvent, ...] = ()
    sections: tuple[TimedSection, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "beats", tuple(_coerce_timed_event(event) for event in self.beats))
        object.__setattr__(
            self,
            "downbeats",
            tuple(_coerce_timed_event(event) for event in self.downbeats),
        )
        object.__setattr__(
            self,
            "sections",
            tuple(_coerce_timed_section(section) for section in self.sections),
        )
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(
            self,
            "warnings",
            tuple(str(warning) for warning in self.warnings if str(warning)),
        )


@dataclass(frozen=True)
class BeatEvent:
    """Validated beat event on the canonical sample timeline."""

    sample: int
    confidence: float | None = None

    def __post_init__(self) -> None:
        sample = int(self.sample)
        if sample < 0:
            raise ValueError("event sample must be non-negative")
        object.__setattr__(self, "sample", sample)
        object.__setattr__(self, "confidence", _optional_confidence(self.confidence))


DownbeatEvent = BeatEvent


@dataclass(frozen=True)
class StructureSection:
    """Validated section interval on the canonical sample timeline."""

    start_sample: int
    end_sample: int
    label: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        start = int(self.start_sample)
        end = int(self.end_sample)
        if start < 0 or end <= start:
            raise ValueError("section samples must satisfy 0 <= start < end")
        label = str(self.label).strip()
        if not label:
            raise ValueError("section label must be non-empty")
        object.__setattr__(self, "start_sample", start)
        object.__setattr__(self, "end_sample", end)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "confidence", _optional_confidence(self.confidence))


@dataclass(frozen=True)
class StructureAnalysis:
    """Serializable output of beat/downbeat and structural analysis."""

    schema_version: str
    track_id: str
    sample_rate: int
    duration_samples: int
    beats: tuple[BeatEvent, ...]
    downbeats: tuple[DownbeatEvent, ...]
    sections: tuple[StructureSection, ...]
    provenance: Mapping[str, Any]
    warnings: tuple[str, ...] = ()
    status: str = "ok"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.track_id:
            raise ValueError("track_id must be non-empty")
        sample_rate = int(self.sample_rate)
        duration = int(self.duration_samples)
        if sample_rate <= 0 or duration < 0:
            raise ValueError("sample_rate and duration_samples are invalid")
        beats = tuple(_coerce_sample_event(event) for event in self.beats)
        downbeats = tuple(_coerce_sample_event(event) for event in self.downbeats)
        sections = tuple(_coerce_sample_section(section) for section in self.sections)
        if any(event.sample >= duration for event in beats + downbeats):
            raise ValueError("event sample must be inside the track duration")
        for previous, current in zip(beats, beats[1:]):
            if current.sample <= previous.sample:
                raise ValueError("beats must be strictly chronological")
        for previous, current in zip(downbeats, downbeats[1:]):
            if current.sample <= previous.sample:
                raise ValueError("downbeats must be strictly chronological")
        for previous, current in zip(sections, sections[1:]):
            if current.start_sample < previous.end_sample:
                raise ValueError("sections must be chronological and non-overlapping")
        object.__setattr__(self, "schema_version", str(self.schema_version))
        object.__setattr__(self, "sample_rate", sample_rate)
        object.__setattr__(self, "duration_samples", duration)
        object.__setattr__(self, "beats", beats)
        object.__setattr__(self, "downbeats", downbeats)
        object.__setattr__(self, "sections", sections)
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(
            self,
            "warnings",
            tuple(str(warning) for warning in self.warnings if str(warning)),
        )
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        """Return the stable event-file representation used by the CLI."""

        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "track_id": self.track_id,
            "sample_rate": self.sample_rate,
            "duration_samples": self.duration_samples,
            "beats": [
                {"sample": event.sample, "confidence": event.confidence}
                for event in self.beats
            ],
            "downbeats": [
                {"sample": event.sample, "confidence": event.confidence}
                for event in self.downbeats
            ],
            "sections": [
                {
                    "start_sample": section.start_sample,
                    "end_sample": section.end_sample,
                    "label": section.label,
                    "confidence": section.confidence,
                }
                for section in self.sections
            ],
            "provenance": _json_safe(dict(self.provenance)),
            "warnings": list(self.warnings),
            "diagnostics": _json_safe(dict(self.diagnostics)),
        }


class StructureProvider(Protocol):
    """Optional provider protocol implemented outside this core module."""

    def predict(self, samples: np.ndarray, sample_rate: int) -> ProviderPrediction: ...


def _optional_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        return None
    return result


def _coerce_timed_event(value: Any) -> TimedEvent:
    if isinstance(value, TimedEvent):
        return value
    if isinstance(value, Mapping):
        time_value = value.get("time_seconds", value.get("time", value.get("seconds")))
        return TimedEvent(time_value, value.get("confidence"))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError("empty event sequence")
        return TimedEvent(value[0], value[1] if len(value) > 1 else None)
    return TimedEvent(value)


def _coerce_timed_section(value: Any) -> TimedSection:
    if isinstance(value, TimedSection):
        return value
    if isinstance(value, Mapping):
        return TimedSection(
            value.get("start_seconds", value.get("start", 0.0)),
            value.get("end_seconds", value.get("end")),
            value.get("label", "provider_section"),
            value.get("confidence"),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) < 2:
            raise ValueError("section sequence requires start and end")
        return TimedSection(value[0], value[1], value[2] if len(value) > 2 else "provider_section", value[3] if len(value) > 3 else None)
    raise TypeError("unsupported section value")


def _coerce_provider_prediction(value: Any) -> ProviderPrediction:
    if isinstance(value, ProviderPrediction):
        return value
    if isinstance(value, Mapping):
        return ProviderPrediction(
            tuple(value.get("beats", ())),
            tuple(value.get("downbeats", ())),
            tuple(value.get("sections", ())),
            value.get("provenance", {}),
            tuple(value.get("warnings", ())),
        )
    if isinstance(value, Sequence) and len(value) >= 2:
        return ProviderPrediction(tuple(value[0]), tuple(value[1]))
    raise TypeError("provider must return ProviderPrediction or a compatible mapping")


def _coerce_sample_event(value: Any) -> BeatEvent:
    if isinstance(value, BeatEvent):
        return value
    if isinstance(value, Mapping):
        return BeatEvent(value.get("sample"), value.get("confidence"))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return BeatEvent(value[0], value[1] if len(value) > 1 else None)
    return BeatEvent(value)


def _coerce_sample_section(value: Any) -> StructureSection:
    if isinstance(value, StructureSection):
        return value
    if isinstance(value, Mapping):
        return StructureSection(
            value.get("start_sample"),
            value.get("end_sample"),
            value.get("label", HEURISTIC_SECTION_LABEL),
            value.get("confidence"),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return StructureSection(value[0], value[1], value[2], value[3] if len(value) > 3 else None)
    raise TypeError("unsupported sample section value")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sample_from_seconds(time_seconds: float, sample_rate: int, duration_samples: int) -> int | None:
    try:
        seconds = float(time_seconds)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(seconds) or seconds < 0.0:
        return None
    sample = int(round(seconds * float(sample_rate)))
    if sample < 0 or sample >= duration_samples:
        return None
    return sample


def _canonical_events(
    values: Iterable[Any],
    sample_rate: int,
    duration_samples: int,
) -> tuple[BeatEvent, ...]:
    """Convert provider seconds to strictly ordered canonical samples."""

    by_sample: dict[int, float | None] = {}
    for raw in values:
        try:
            event = _coerce_timed_event(raw)
        except (TypeError, ValueError, IndexError):
            continue
        sample = _sample_from_seconds(event.time_seconds, sample_rate, duration_samples)
        if sample is None:
            continue
        old = by_sample.get(sample)
        confidence = event.confidence
        if old is None or (confidence is not None and (old is None or confidence > old)):
            by_sample[sample] = confidence
        elif sample not in by_sample:
            by_sample[sample] = confidence
    return tuple(BeatEvent(sample, by_sample[sample]) for sample in sorted(by_sample))


def _canonical_sections(
    values: Iterable[Any],
    sample_rate: int,
    duration_samples: int,
) -> tuple[StructureSection, ...]:
    intervals: list[StructureSection] = []
    for raw in values:
        try:
            section = _coerce_timed_section(raw)
        except (TypeError, ValueError, IndexError):
            continue
        start = _sample_from_seconds(section.start_seconds, sample_rate, duration_samples)
        end_value = section.end_seconds * float(sample_rate)
        if start is None or not math.isfinite(end_value):
            continue
        end = min(duration_samples, max(start + 1, int(round(end_value))))
        if end <= start:
            continue
        intervals.append(StructureSection(start, end, section.label, section.confidence))
    intervals.sort(key=lambda section: (section.start_sample, section.end_sample, section.label))
    clean: list[StructureSection] = []
    for section in intervals:
        if clean and section.start_sample < clean[-1].end_sample:
            if section.end_sample <= clean[-1].end_sample:
                continue
            section = StructureSection(clean[-1].end_sample, section.end_sample, section.label, section.confidence)
        clean.append(section)
    return tuple(clean)


def heuristic_feature_sections(
    track: AudioTrack,
    *,
    window_seconds: float = 8.0,
    hop_seconds: float = 4.0,
    max_sections: int = 32,
) -> tuple[StructureSection, ...]:
    """Create explicitly heuristic segments from coarse spectral changes.

    This is a fallback for providers that do not emit section labels.  It
    computes RMS, spectral centroid, and spectral spread on overlapping
    windows, chooses separated local change maxima, and labels every segment
    ``heuristic_feature_segment``.  Confidence is intentionally ``None``: the
    change score is not a calibrated probability or a section-model output.
    """

    if not isinstance(track, AudioTrack):
        raise TypeError("heuristic_feature_sections expects an AudioTrack")
    if int(max_sections) < 1:
        raise ValueError("max_sections must be positive")
    duration = track.duration_samples
    if duration <= 0:
        return ()
    window = max(1, int(round(float(window_seconds) * track.sample_rate)))
    hop = max(1, int(round(float(hop_seconds) * track.sample_rate)))
    if window >= duration:
        return (StructureSection(0, duration, HEURISTIC_SECTION_LABEL, None),)
    mono = np.asarray(track.samples, dtype=np.float64).mean(axis=1)
    starts = np.arange(0, duration - window + 1, hop, dtype=np.int64)
    features = np.empty((starts.size, 3), dtype=np.float64)
    n_fft = min(4096, max(64, 1 << (min(window, 4096) - 1).bit_length()))
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / float(track.sample_rate))
    positive = frequencies > 0.0
    fft_window = np.hanning(n_fft)
    nyquist = max(track.sample_rate / 2.0, 1.0)
    for index, start in enumerate(starts.tolist()):
        segment = mono[start : start + window]
        if segment.size > n_fft:
            indices = np.linspace(0, segment.size - 1, n_fft, dtype=np.int64)
            segment = segment[indices]
        elif segment.size < n_fft:
            padded = np.zeros(n_fft, dtype=np.float64)
            padded[: segment.size] = segment
            segment = padded
        rms = float(np.sqrt(np.mean(segment * segment)))
        magnitude = np.abs(np.fft.rfft(segment * fft_window))
        total = float(np.sum(magnitude[positive]))
        if total <= 1e-12:
            centroid = 0.0
            spread = 0.0
        else:
            freq = frequencies[positive]
            mag = magnitude[positive]
            centroid = float(np.sum(freq * mag) / total) / nyquist
            spread = float(np.sqrt(np.sum(((freq / nyquist - centroid) ** 2) * mag) / total))
        features[index] = (math.log1p(rms * 1000.0), centroid, spread)
    if features.shape[0] <= 1:
        return (StructureSection(0, duration, HEURISTIC_SECTION_LABEL, None),)
    deltas = np.linalg.norm(np.diff(features, axis=0), axis=1)
    median = float(np.median(deltas))
    mad = float(np.median(np.abs(deltas - median)))
    threshold = max(median + 2.5 * max(mad, 1e-12), float(np.percentile(deltas, 85.0)))
    candidates = np.flatnonzero(deltas >= threshold)
    delta_by_boundary = {
        int(starts[index + 1]): float(deltas[index])
        for index in range(deltas.size)
    }
    # Keep local maxima and enforce a coarse minimum segment duration.
    local: list[int] = []
    for index in candidates.tolist():
        if index <= 0 or index >= deltas.size - 1:
            continue
        if deltas[index] < deltas[index - 1] or deltas[index] < deltas[index + 1]:
            continue
        boundary = int(starts[index + 1])
        if local and boundary - local[-1] < hop * 2:
            if deltas[index] > delta_by_boundary.get(local[-1], -math.inf):
                local[-1] = boundary
        else:
            local.append(boundary)
    if len(local) > max(0, max_sections - 1):
        strongest = sorted(
            local,
            key=lambda boundary: (-delta_by_boundary.get(boundary, 0.0), boundary),
        )[: max_sections - 1]
        local = sorted(strongest)
    boundaries = [0] + [boundary for boundary in local if 0 < boundary < duration] + [duration]
    return tuple(
        StructureSection(start, end, HEURISTIC_SECTION_LABEL, None)
        for start, end in zip(boundaries, boundaries[1:])
        if end > start
    )


def analyze_structure(
    track: AudioTrack,
    provider: StructureProvider | Callable[..., Any] | None = None,
    *,
    include_heuristic_sections: bool = True,
) -> StructureAnalysis:
    """Run an optional beat/downbeat provider and canonicalize its output.

    Provider failures are represented as a serializable ``status='unavailable'``
    result with empty beat/downbeat events and an actionable warning.  This
    keeps offline runs reproducible when optional model dependencies or weights
    are absent.  No confidence is synthesized for events without a provider
    confidence.
    """

    if not isinstance(track, AudioTrack):
        raise TypeError("analyze_structure expects an AudioTrack")
    provider_prediction = ProviderPrediction()
    warnings: list[str] = []
    provenance: dict[str, Any] = {}
    status = "unavailable" if provider is None else "ok"
    if provider is None:
        warnings.append("No beat/downbeat provider was supplied.")
        provenance["provider"] = "none"
    else:
        try:
            if hasattr(provider, "predict"):
                raw_prediction = provider.predict(track.samples, track.sample_rate)
            elif hasattr(provider, "analyze"):
                raw_prediction = provider.analyze(track)
            elif callable(provider):
                raw_prediction = provider(track.samples, track.sample_rate)
            else:
                raise TypeError("provider has no predict/analyze callable")
            provider_prediction = _coerce_provider_prediction(raw_prediction)
            provenance.update(provider_prediction.provenance)
            warnings.extend(provider_prediction.warnings)
        except Exception as exc:  # provider is optional and external
            status = "unavailable"
            warnings.append(f"Beat/downbeat provider failed: {type(exc).__name__}: {exc}")
            provider_metadata = getattr(provider, "provenance", None)
            if isinstance(provider_metadata, Mapping):
                provenance.update(provider_metadata)
            provenance.setdefault("provider", type(provider).__name__)

    beats = _canonical_events(provider_prediction.beats, track.sample_rate, track.duration_samples)
    downbeats = _canonical_events(
        provider_prediction.downbeats, track.sample_rate, track.duration_samples
    )
    sections = _canonical_sections(
        provider_prediction.sections, track.sample_rate, track.duration_samples
    )
    if not sections and include_heuristic_sections:
        sections = heuristic_feature_sections(track)
        warnings.append(
            "Sections are heuristic_feature_segment boundaries from coarse spectral changes; "
            "no provider section labels were available."
        )
    if not beats:
        warnings.append("No provider beat events were available.")
    if not downbeats:
        warnings.append("No provider downbeat events were available.")
    if status == "ok" and not beats and not downbeats:
        status = "ok_empty"
    # Preserve warning order while removing repeated messages from adapters.
    unique_warnings = tuple(dict.fromkeys(str(warning) for warning in warnings if str(warning)))
    diagnostics = {
        "beat_count": len(beats),
        "downbeat_count": len(downbeats),
        "section_count": len(sections),
        "provider_events_have_confidence": any(
            event.confidence is not None for event in beats + downbeats
        ),
        "sections_are_heuristic": bool(
            sections and all(section.label == HEURISTIC_SECTION_LABEL for section in sections)
        ),
    }
    return StructureAnalysis(
        schema_version=STRUCTURE_SCHEMA_VERSION,
        track_id=track.track_id,
        sample_rate=track.sample_rate,
        duration_samples=track.duration_samples,
        beats=beats,
        downbeats=downbeats,
        sections=sections,
        provenance=provenance,
        warnings=unique_warnings,
        status=status,
        diagnostics=diagnostics,
    )


def write_structure(path: str | Path, analysis: StructureAnalysis) -> Path:
    """Write one event JSON file, creating only its parent directories."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


__all__ = [
    "ALL_IN_ONE_PAPER",
    "BEAT_THIS_CHECKPOINT_ROOT",
    "BEAT_THIS_PAPER",
    "BEAT_THIS_REPOSITORY",
    "BeatEvent",
    "DownbeatEvent",
    "HEURISTIC_SECTION_LABEL",
    "ProviderPrediction",
    "STRUCTURE_SCHEMA_VERSION",
    "StructureAnalysis",
    "StructureProvider",
    "StructureSection",
    "TimedEvent",
    "TimedSection",
    "analyze_structure",
    "heuristic_feature_sections",
    "write_structure",
]
