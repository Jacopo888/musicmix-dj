"""Safe word-boundary refinement for an already selected transition.

This module is deliberately a domain guard, not a speech recognizer.  It
accepts word spans supplied by an optional recognizer and moves each candidate
cue into a confidence-qualified internal word gap.  It never uses silence
before the first word or after the last word as a fabricated boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Sequence

from .contracts import Candidate


_VERSION = "word-guard-v1"


@dataclass(frozen=True)
class WordSpan:
    """Minimal recognizer output consumed by :func:`plan_word_boundary`."""

    start_sample: int
    end_sample: int
    confidence: float


class NoSafeBoundaryError(ValueError):
    """Raised when no confidence-qualified, in-range word gap is available."""

    def __init__(self, detail: str = "no safe internal word boundary") -> None:
        self.reason = "no_safe_boundary"
        self.detail = detail
        super().__init__(f"no_safe_boundary: {detail}")


@dataclass(frozen=True)
class WordBoundaryPlan:
    """Revised cue pair and the exact shifts made by the word guard."""

    original_candidate_id: str
    revised: Candidate
    cue_out_shift_samples: int
    cue_in_shift_samples: int
    fade_seconds: float
    notes: tuple[str, ...]

    @property
    def candidate(self) -> Candidate:
        """Alias useful to callers that call the revised object ``candidate``."""

        return self.revised

    @property
    def revised_candidate(self) -> Candidate:
        return self.revised

    @property
    def cue_shifts(self) -> tuple[int, int]:
        return self.cue_out_shift_samples, self.cue_in_shift_samples

    @property
    def shift_out_samples(self) -> int:
        return self.cue_out_shift_samples

    @property
    def shift_in_samples(self) -> int:
        return self.cue_in_shift_samples

    @property
    def revised_candidate_id(self) -> str:
        return self.revised.candidate_id


@dataclass(frozen=True)
class _Gap:
    start: int
    end: int
    left_confidence: float
    right_confidence: float
    had_overlap: bool = False


@dataclass(frozen=True)
class _GuardParameters:
    rate: float
    margin_samples: int
    target_fade_samples: int
    min_fade_samples: int
    max_shift_samples: int
    confidence_threshold: float


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"invalid_{name}")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid_{name}") from exc
    if not math.isfinite(number):
        raise ValueError(f"invalid_{name}")
    return number


def _integer_sample(value: Any, name: str) -> int:
    number = _finite_number(value, name)
    if number < 0 or not number.is_integer():
        raise ValueError(f"invalid_{name}")
    return int(number)


def _coerce_span(span: Any) -> WordSpan:
    """Accept the local dataclass or a compatible recognizer object."""

    try:
        start_value = getattr(span, "start_sample", None)
        end_value = getattr(span, "end_sample", None)
        if start_value is None:
            start_value = getattr(span, "start")
        if end_value is None:
            end_value = getattr(span, "end")
        start = _integer_sample(start_value, "word_start_sample")
        end = _integer_sample(end_value, "word_end_sample")
        confidence_value = getattr(span, "confidence", None)
        # Missing confidence is unavailable evidence, not evidence of a safe
        # boundary.  Treat it as zero so the gap is filtered by the confidence
        # gate and results in ``no_safe_boundary`` when no alternative exists.
        confidence = (
            0.0
            if confidence_value is None
            else _finite_number(confidence_value, "word_confidence")
        )
    except AttributeError as exc:
        raise ValueError("invalid_word_span") from exc
    if end <= start:
        raise ValueError("invalid_word_span_interval")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("invalid_word_confidence")
    return WordSpan(start, end, confidence)


def _internal_gaps(spans: Sequence[Any], margin_samples: int, min_confidence: float) -> list[_Gap]:
    """Return internal gaps after merging overlapping word intervals.

    Overlapping recognizer spans are treated as one occupied region.  The
    confidence of an overlap cluster is the minimum confidence in that
    cluster, so a low-confidence word cannot be silently bypassed when a gap
    follows it.
    """

    ordered = sorted(
        (_coerce_span(span) for span in spans),
        key=lambda span: (span.start_sample, span.end_sample, span.confidence),
    )
    if len(ordered) < 2:
        return []

    gaps: list[_Gap] = []
    cluster_start = ordered[0].start_sample
    cluster_end = ordered[0].end_sample
    cluster_confidence = ordered[0].confidence
    cluster_had_overlap = False
    for span in ordered[1:]:
        if span.start_sample < cluster_end:
            # There is no usable boundary inside an overlap.  Merge it into
            # the occupied region and retain the least confidence.
            cluster_end = max(cluster_end, span.end_sample)
            cluster_confidence = min(cluster_confidence, span.confidence)
            cluster_had_overlap = True
            continue

        raw_start = cluster_end + margin_samples
        raw_end = span.start_sample - margin_samples
        if raw_end > raw_start and cluster_confidence >= min_confidence and span.confidence >= min_confidence:
            gaps.append(_Gap(raw_start, raw_end, cluster_confidence, span.confidence, cluster_had_overlap))
        cluster_start = span.start_sample
        cluster_end = span.end_sample
        cluster_confidence = span.confidence
        cluster_had_overlap = False
    return gaps


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _candidate_id(original_id: str, cue_out: int, cue_in: int, fade_samples: int, sample_rate: float) -> str:
    payload = f"{_VERSION}|{original_id}|{cue_out}|{cue_in}|{fade_samples}|{sample_rate:.12g}".encode("utf-8")
    return "word_" + hashlib.sha256(payload).hexdigest()[:20]


def _guard_parameters(
    sample_rate: float,
    max_shift_seconds: float,
    target_fade_seconds: float,
    min_fade_seconds: float,
    margin_seconds: float,
    min_confidence: float,
) -> _GuardParameters:
    rate = _finite_number(sample_rate, "sample_rate")
    if rate <= 0:
        raise ValueError("invalid_sample_rate")
    max_shift = _finite_number(max_shift_seconds, "max_shift_seconds")
    target_fade = _finite_number(target_fade_seconds, "target_fade_seconds")
    min_fade = _finite_number(min_fade_seconds, "min_fade_seconds")
    margin = _finite_number(margin_seconds, "margin_seconds")
    confidence_threshold = _finite_number(min_confidence, "min_confidence")
    if max_shift < 0 or target_fade <= 0 or min_fade <= 0 or margin < 0:
        raise ValueError("invalid_word_guard_parameters")
    if target_fade < min_fade:
        raise ValueError("target_fade_seconds_below_minimum")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("invalid_min_confidence")
    return _GuardParameters(
        rate=rate,
        # ceil margins so the excluded physical interval is never shorter
        # than requested; ceil minimum fade so the returned fade is never
        # below it.
        margin_samples=int(math.ceil(margin * rate)),
        target_fade_samples=max(1, int(round(target_fade * rate))),
        min_fade_samples=max(1, int(math.ceil(min_fade * rate))),
        max_shift_samples=int(math.floor(max_shift * rate + 1e-9)),
        confidence_threshold=confidence_threshold,
    )


def _candidate_cues(candidate: Candidate) -> tuple[int, int]:
    return (
        _integer_sample(candidate.cue_out_sample, "candidate_cue_out_sample"),
        _integer_sample(candidate.cue_in_sample, "candidate_cue_in_sample"),
    )


def _revised_plan(
    candidate: Candidate,
    original_out: int,
    original_in: int,
    cue_out: int,
    cue_in: int,
    fade_samples: int,
    rate: float,
    notes: Iterable[str],
) -> WordBoundaryPlan:
    revised_id = _candidate_id(candidate.candidate_id, cue_out, cue_in, fade_samples, rate)
    revised = Candidate(
        candidate_id=revised_id,
        track_a=candidate.track_a,
        track_b=candidate.track_b,
        cue_out_sample=cue_out,
        cue_in_sample=cue_in,
        mode=candidate.mode,
        features=candidate.features,
        penalties=candidate.penalties,
        passes_constraints=candidate.passes_constraints,
        constraint_reasons=candidate.constraint_reasons,
    )
    return WordBoundaryPlan(
        original_candidate_id=candidate.candidate_id,
        revised=revised,
        cue_out_shift_samples=cue_out - original_out,
        cue_in_shift_samples=cue_in - original_in,
        fade_seconds=fade_samples / rate,
        notes=tuple(notes),
    )


def plan_word_boundary(
    candidate: Candidate | Sequence[Any] | Iterable[Any],
    words_a: Sequence[Any] | Iterable[Any],
    words_b: Sequence[Any] | Iterable[Any] | Candidate,
    sample_rate: float,
    max_shift_seconds: float = 1.5,
    target_fade_seconds: float = 0.5,
    min_fade_seconds: float = 0.02,
    margin_seconds: float = 0.02,
    min_confidence: float = 0.6,
) -> WordBoundaryPlan:
    """Move candidate cues to safe internal gaps with minimal total shift.

    Each side of the transition receives the same fade duration.  The fade is
    shrunk to the largest common available gap when the target does not fit,
    provided the configured minimum still fits.  Candidate features and
    penalties are retained as provenance, with an explicit stale-score note;
    callers must rerun the ranker after the cue move.
    """

    # Permit the equally natural ``(words_a, words_b, candidate, rate)``
    # ordering as a compatibility convenience while keeping the documented
    # candidate-first signature.  The type check remains strict after swap.
    if not isinstance(candidate, Candidate) and isinstance(words_b, Candidate):
        candidate, words_a, words_b = words_b, candidate, words_a
    if not isinstance(candidate, Candidate):
        raise TypeError("candidate must be a Candidate")
    original_out = _integer_sample(candidate.cue_out_sample, "candidate_cue_out_sample")
    original_in = _integer_sample(candidate.cue_in_sample, "candidate_cue_in_sample")
    parameters = _guard_parameters(
        sample_rate,
        max_shift_seconds,
        target_fade_seconds,
        min_fade_seconds,
        margin_seconds,
        min_confidence,
    )
    original_out, original_in = _candidate_cues(candidate)
    gaps_a = _internal_gaps(
        tuple(words_a), parameters.margin_samples, parameters.confidence_threshold
    )
    gaps_b = _internal_gaps(
        tuple(words_b), parameters.margin_samples, parameters.confidence_threshold
    )
    if not gaps_a or not gaps_b:
        raise NoSafeBoundaryError("no confidence-qualified internal gap on both sides")

    best: tuple[tuple[Any, ...], int, int, int, _Gap, _Gap] | None = None
    for gap_a in gaps_a:
        for gap_b in gaps_b:
            available = min(gap_a.end - gap_a.start, gap_b.end - gap_b.start)
            fade_samples = min(parameters.target_fade_samples, available)
            if fade_samples < parameters.min_fade_samples:
                continue
            out_low, out_high = gap_a.start + fade_samples, gap_a.end
            in_low, in_high = gap_b.start, gap_b.end - fade_samples
            cue_out = _clamp(original_out, out_low, out_high)
            cue_in = _clamp(original_in, in_low, in_high)
            shift_out = cue_out - original_out
            shift_in = cue_in - original_in
            if (abs(shift_out) > parameters.max_shift_samples or
                    abs(shift_in) > parameters.max_shift_samples):
                continue
            total_shift = abs(shift_out) + abs(shift_in)
            # Primary objective: minimum total movement.  Ties prefer the
            # largest common fade, then stable source order and cue values.
            key = (total_shift, -fade_samples, cue_out, cue_in, gap_a.start, gap_b.start)
            option = (key, cue_out, cue_in, fade_samples, gap_a, gap_b)
            if best is None or option[0] < best[0]:
                best = option

    if best is None:
        raise NoSafeBoundaryError("internal gaps exist but all require excessive cue movement or fade shrinkage")

    _, cue_out, cue_in, fade_samples, gap_a, gap_b = best
    notes = [
        _VERSION,
        "score_stale_recompute_required",
        f"fade_samples={fade_samples}",
        f"gap_a=[{gap_a.start},{gap_a.end})",
        f"gap_b=[{gap_b.start},{gap_b.end})",
    ]
    if fade_samples < parameters.target_fade_samples:
        notes.append("fade_shrunk_to_gap_availability")
    if gap_a.had_overlap or gap_b.had_overlap:
        notes.append("overlapping_word_spans_merged_before_gap_check")
    return _revised_plan(
        candidate,
        original_out,
        original_in,
        cue_out,
        cue_in,
        fade_samples,
        parameters.rate,
        notes,
    )


def plan_outgoing_word_boundary(
    candidate: Candidate,
    words_a: Sequence[Any] | Iterable[Any],
    sample_rate: float,
    max_shift_seconds: float = 1.5,
    target_fade_seconds: float = 0.5,
    min_fade_seconds: float = 0.02,
    margin_seconds: float = 0.02,
    min_confidence: float = 0.6,
) -> WordBoundaryPlan:
    """Guard only the outgoing A cue using word spans from A.

    The incoming B cue is preserved exactly because no safe B word boundary is
    asserted by this API.  The returned plan explicitly records that the
    incoming boundary remains unverified; callers may use it for a cautious
    A-only crop, but must not describe B as word-aligned.
    """

    if not isinstance(candidate, Candidate):
        raise TypeError("candidate must be a Candidate")
    parameters = _guard_parameters(
        sample_rate,
        max_shift_seconds,
        target_fade_seconds,
        min_fade_seconds,
        margin_seconds,
        min_confidence,
    )
    original_out, original_in = _candidate_cues(candidate)
    gaps_a = _internal_gaps(
        tuple(words_a), parameters.margin_samples, parameters.confidence_threshold
    )
    if not gaps_a:
        raise NoSafeBoundaryError("no confidence-qualified internal outgoing gap")

    best: tuple[tuple[Any, ...], int, int, _Gap] | None = None
    for gap_a in gaps_a:
        available = gap_a.end - gap_a.start
        fade_samples = min(parameters.target_fade_samples, available)
        if fade_samples < parameters.min_fade_samples:
            continue
        out_low, out_high = gap_a.start + fade_samples, gap_a.end
        cue_out = _clamp(original_out, out_low, out_high)
        shift_out = cue_out - original_out
        if abs(shift_out) > parameters.max_shift_samples:
            continue
        # Preserve the original incoming cue; the outgoing boundary is the
        # only objective being guarded.  Ties prefer a larger fade, then the
        # earliest stable gap/cue coordinate.
        key = (abs(shift_out), -fade_samples, cue_out, gap_a.start)
        option = (key, cue_out, fade_samples, gap_a)
        if best is None or option[0] < best[0]:
            best = option

    if best is None:
        raise NoSafeBoundaryError(
            "internal outgoing gaps exist but all require excessive cue movement or fade shrinkage"
        )

    _, cue_out, fade_samples, gap_a = best
    notes = [
        _VERSION,
        "score_stale_recompute_required",
        "incoming_boundary_unverified",
        "incoming_boundary_unverified_no_word_spans",
        f"fade_samples={fade_samples}",
        f"gap_a=[{gap_a.start},{gap_a.end})",
    ]
    if fade_samples < parameters.target_fade_samples:
        notes.append("fade_shrunk_to_gap_availability")
    if gap_a.had_overlap:
        notes.append("overlapping_word_spans_merged_before_gap_check")
    return _revised_plan(
        candidate,
        original_out,
        original_in,
        cue_out,
        original_in,
        fade_samples,
        parameters.rate,
        notes,
    )


# Friendly aliases for callers that describe this stage as guarding rather
# than planning; all aliases intentionally share the same deterministic API.
guard_word_boundary = plan_word_boundary
protect_word_boundary = plan_word_boundary
guard_outgoing_word_boundary = plan_outgoing_word_boundary


__all__ = [
    "NoSafeBoundaryError",
    "WordBoundaryPlan",
    "WordSpan",
    "guard_outgoing_word_boundary",
    "guard_word_boundary",
    "plan_outgoing_word_boundary",
    "plan_word_boundary",
    "protect_word_boundary",
]
