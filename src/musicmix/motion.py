"""Auditable multi-scale motion descriptors for directional cue experiments.

The current directional ranker compares two local snapshots.  This module is
an intentionally separate experiment: it compares beat-level profiles before
and after a handoff on 8/16/32-beat scales.  RMS, novelty, and chroma are
reported as observed descriptors.  They are not silently renamed to musical
tension. Even a complete measured ``tension_proxy`` describes a change,
not whether the incoming passage fulfils the outgoing expectation.
Resolution requires independent evidence of musical closure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_MOTION_WINDOWS = (8, 16, 32)
_SLOPE_EPSILON = 0.015
_TENSION_EPSILON = 0.10


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bounded_confidence(value: Any) -> float | None:
    number = _finite(value)
    if number is None:
        return None
    return max(0.0, min(1.0, number))


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(np.asarray(values, dtype=float))) if values else None


def _slope(values: Sequence[float | None]) -> float | None:
    """Fit a normalized per-beat slope over available finite values."""

    valid = [(index, _finite(value)) for index, value in enumerate(values)]
    valid = [(index, value) for index, value in valid if value is not None]
    if len(valid) < 2:
        return None
    x = np.asarray([index for index, _ in valid], dtype=float)
    y = np.asarray([value for _, value in valid], dtype=float)
    centered = x - float(np.mean(x))
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-12:
        return None
    return float(np.dot(centered, y - float(np.mean(y))) / denominator)


def _cosine(left: Sequence[float] | None, right: Sequence[float] | None) -> float | None:
    if left is None or right is None or len(left) != len(right) or not left:
        return None
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return None
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return None
    return max(0.0, min(1.0, float(np.dot(a, b) / denominator)))


@dataclass(frozen=True)
class MotionBeat:
    """One beat-level observation from a structure/feature provider.

    ``rms`` and ``novelty`` are raw non-negative descriptors.  ``chroma`` is
    a twelve-bin vector in the provider's fixed pitch-class orientation; this
    module never rotates it.  ``tension_proxy`` is optional and must be
    explicitly supplied by a provider that documents what it measures.
    """

    rms: float | None = None
    chroma: tuple[float, ...] | None = None
    novelty: float | None = None
    tension_proxy: float | None = None
    confidence: float | None = None
    sample: int | None = None

    def __post_init__(self) -> None:
        for name in ("rms", "novelty"):
            value = getattr(self, name)
            if value is not None:
                number = _finite(value)
                if number is None or number < 0:
                    raise ValueError(f"{name} must be finite and non-negative")
        if self.tension_proxy is not None:
            value = _finite(self.tension_proxy)
            if value is None or not 0.0 <= value <= 1.0:
                raise ValueError("tension_proxy must be between 0 and 1")
        if self.confidence is not None:
            value = _finite(self.confidence)
            if value is None or not 0.0 <= value <= 1.0:
                raise ValueError("confidence must be between 0 and 1")
        if self.sample is not None and (not isinstance(self.sample, int) or self.sample < 0):
            raise ValueError("sample must be a non-negative integer")
        if self.chroma is not None:
            if len(self.chroma) != 12:
                raise ValueError("chroma must contain twelve bins")
            if not all(_finite(value) is not None for value in self.chroma):
                raise ValueError("chroma bins must be finite")


@dataclass(frozen=True)
class MotionScaleReport:
    """Observed before/after motion at one beat scale."""

    window_beats: int
    before_count: int
    after_count: int
    energy_before_slope: float | None
    energy_after_slope: float | None
    energy_boundary_delta: float | None
    novelty_before_mean: float | None
    novelty_after_mean: float | None
    novelty_boundary_delta: float | None
    chroma_boundary_similarity: float | None
    tension_before_mean: float | None
    tension_after_mean: float | None
    tension_delta: float | None
    confidence: float | None
    classification: str
    missing: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class MotionAssessment:
    """All requested scales plus deterministic experiment diagnostics."""

    scales: tuple[MotionScaleReport, ...]
    preferred_window_beats: int | None
    diagnostics: Mapping[str, Any]


def _coerce_beat(value: MotionBeat | Mapping[str, Any]) -> MotionBeat:
    if isinstance(value, MotionBeat):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("motion beats must be MotionBeat or mappings")
    chroma = value.get("chroma")
    if chroma is not None:
        chroma = tuple(float(item) for item in chroma)
    return MotionBeat(
        rms=value.get("rms"),
        chroma=chroma,
        novelty=value.get("novelty"),
        tension_proxy=value.get("tension_proxy", value.get("tension")),
        confidence=value.get("confidence"),
        sample=value.get("sample"),
    )


def _log_rms(beat: MotionBeat) -> float | None:
    if beat.rms is None:
        return None
    return math.log(max(float(beat.rms), 1e-8))


def _profile_confidence(beats: Sequence[MotionBeat]) -> float | None:
    values = [beat.confidence for beat in beats if beat.confidence is not None]
    if len(values) != len(beats) or not values:
        return None
    return float(min(values))


def _classify(
    energy_boundary_delta: float | None,
    novelty_delta: float | None,
    tension_delta: float | None,
    chroma_similarity: float | None,
) -> tuple[str, tuple[str, ...]]:
    notes: list[str] = []
    if tension_delta is not None:
        if tension_delta <= -_TENSION_EPSILON:
            return "tension_proxy_decrease", ("measured_tension_proxy_decreased",
                                               "resolution_requires_independent_closure_evidence")
        if tension_delta >= _TENSION_EPSILON:
            return "tension_proxy_increase", ("measured_tension_proxy_increased",
                                               "expectation_fulfilment_not_established")
        return "tension_proxy_stable", ("measured_tension_proxy_change_small",
                                         "expectation_fulfilment_not_established")

    notes.append("tension_proxy_missing_or_incomplete")
    if energy_boundary_delta is None:
        return "unknown_tension", tuple(notes + ["energy_arc_incomplete"])
    if abs(energy_boundary_delta) <= _SLOPE_EPSILON:
        return "energy_continuation", tuple(notes)
    if energy_boundary_delta > 0:
        return "energy_rise", tuple(notes)
    if novelty_delta is not None and novelty_delta < -_SLOPE_EPSILON:
        notes.append("novelty_also_falls")
    return "energy_fall", tuple(notes)


def _report_for(
    before: Sequence[MotionBeat],
    after: Sequence[MotionBeat],
    window_beats: int,
) -> MotionScaleReport:
    missing: list[str] = []
    before_logs = [_log_rms(beat) for beat in before]
    after_logs = [_log_rms(beat) for beat in after]
    energy_before = _slope(before_logs)
    energy_after = _slope(after_logs)
    before_levels = [value for value in before_logs if value is not None]
    after_levels = [value for value in after_logs if value is not None]
    if not before_levels or not after_levels:
        missing.append("rms")
        energy_boundary_delta = None
    else:
        energy_boundary_delta = float(np.mean(after_levels) - np.mean(before_levels))

    before_novelty = [beat.novelty for beat in before if beat.novelty is not None]
    after_novelty = [beat.novelty for beat in after if beat.novelty is not None]
    novelty_before_mean = _mean(before_novelty)
    novelty_after_mean = _mean(after_novelty)
    novelty_boundary_delta = (
        float(novelty_after_mean - novelty_before_mean)
        if novelty_before_mean is not None and novelty_after_mean is not None
        else None
    )
    if novelty_boundary_delta is None:
        missing.append("novelty")

    chroma_similarity = _cosine(before[-1].chroma, after[0].chroma)
    if chroma_similarity is None:
        missing.append("chroma")

    before_tension = [beat.tension_proxy for beat in before if beat.tension_proxy is not None]
    after_tension = [beat.tension_proxy for beat in after if beat.tension_proxy is not None]
    # A partial proxy profile is not a complete before/after measurement.  Do
    # not average the available subset: doing so could manufacture a
    # resolution label from one measured beat surrounded by unknown beats.
    if len(before_tension) == len(before) and len(after_tension) == len(after):
        tension_before_mean = _mean(before_tension)
        tension_after_mean = _mean(after_tension)
        tension_delta = (
            float(tension_after_mean - tension_before_mean)
            if tension_before_mean is not None and tension_after_mean is not None
            else None
        )
    else:
        tension_before_mean = None
        tension_after_mean = None
        tension_delta = None
    if tension_delta is None:
        missing.append("tension_proxy")

    confidence_before = _profile_confidence(before)
    confidence_after = _profile_confidence(after)
    confidence = (
        min(confidence_before, confidence_after)
        if confidence_before is not None and confidence_after is not None
        else None
    )
    classification, notes = _classify(
        energy_boundary_delta,
        novelty_boundary_delta,
        tension_delta,
        chroma_similarity,
    )
    if confidence is None:
        notes = tuple(notes) + ("confidence_incomplete",)
    if len(missing) == 0 and classification == "unknown_tension":
        notes = tuple(notes) + ("no_tension_inference_from_observed_descriptors",)
    return MotionScaleReport(
        window_beats=window_beats,
        before_count=len(before),
        after_count=len(after),
        energy_before_slope=energy_before,
        energy_after_slope=energy_after,
        energy_boundary_delta=energy_boundary_delta,
        novelty_before_mean=novelty_before_mean,
        novelty_after_mean=novelty_after_mean,
        novelty_boundary_delta=novelty_boundary_delta,
        chroma_boundary_similarity=chroma_similarity,
        tension_before_mean=tension_before_mean,
        tension_after_mean=tension_after_mean,
        tension_delta=tension_delta,
        confidence=confidence,
        classification=classification,
        missing=tuple(dict.fromkeys(missing)),
        notes=notes,
    )


def analyze_motion(
    outgoing: Sequence[MotionBeat | Mapping[str, Any]],
    incoming: Sequence[MotionBeat | Mapping[str, Any]],
    windows_beats: Sequence[int] = DEFAULT_MOTION_WINDOWS,
) -> MotionAssessment:
    """Compare the last beat windows of A with the first windows of B.

    ``outgoing`` is chronological and ends at the proposed A handoff;
    ``incoming`` is chronological and starts at the proposed B handoff.  A
    scale is reported only with exactly ``window_beats`` observations on both
    sides.  Profiles may be longer; the function uses A's tail and B's head.
    No value is imputed for missing descriptors.
    """

    before = tuple(_coerce_beat(value) for value in outgoing)
    after = tuple(_coerce_beat(value) for value in incoming)
    requested = tuple(int(value) for value in windows_beats)
    if not requested or any(value <= 0 for value in requested) or len(set(requested)) != len(requested):
        raise ValueError("windows_beats must contain distinct positive integers")
    reports: list[MotionScaleReport] = []
    for width in requested:
        if len(before) < width or len(after) < width:
            reports.append(
                MotionScaleReport(
                    window_beats=width,
                    before_count=min(len(before), width),
                    after_count=min(len(after), width),
                    energy_before_slope=None,
                    energy_after_slope=None,
                    energy_boundary_delta=None,
                    novelty_before_mean=None,
                    novelty_after_mean=None,
                    novelty_boundary_delta=None,
                    chroma_boundary_similarity=None,
                    tension_before_mean=None,
                    tension_after_mean=None,
                    tension_delta=None,
                    confidence=None,
                    classification="insufficient_window",
                    missing=("before_window" if len(before) < width else "after_window",),
                    notes=("exact_beat_scale_not_available",),
                )
            )
            continue
        reports.append(_report_for(before[-width:], after[:width], width))

    valid = [report for report in reports if report.classification != "insufficient_window"]
    preferred = max(valid, key=lambda report: report.window_beats).window_beats if valid else None
    diagnostics = {
        "windows_requested": list(requested),
        "windows_valid": [report.window_beats for report in valid],
        "preferred_window_beats": preferred,
        "tension_inference_requires_proxy": True,
        "pitch_rotation_applied": False,
        "time_stretch_applied": False,
    }
    return MotionAssessment(tuple(reports), preferred, diagnostics)


__all__ = [
    "DEFAULT_MOTION_WINDOWS",
    "MotionAssessment",
    "MotionBeat",
    "MotionScaleReport",
    "analyze_motion",
]
