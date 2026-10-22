"""Deterministic retrieval of pairwise, arbitrary cue transitions.

The search stage deliberately consumes the small :class:`TrackAnalysis`/``Cue``
contract by attribute rather than depending on a feature extractor.  This keeps
it useful with the light-weight backends used by the project and makes missing
features visible in the returned score reports.

No audio is rendered here.  A candidate describes a directed pair of cue
samples and its score explains the measurable evidence used to rank it.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping

from .contracts import Candidate, Feature, ScoreReport
from .ranking import InterpretableRanker


_VERSION = "arbitrary-cue-search-v1"
_EPS = 1e-12


@dataclass(frozen=True)
class SearchResult:
    """Complete, auditable output of :func:`search_transitions`.

    ``candidates`` and ``reports`` are parallel sequences in generation order;
    every generated candidate has exactly one report, including candidates
    that cannot be scored because the available evidence is insufficient.
    ``selected`` contains candidate IDs only so that callers can resolve the
    full candidate and report without duplicating records.
    """

    candidates: tuple[Candidate, ...]
    reports: tuple[ScoreReport, ...]
    selected: tuple[str, ...]
    rejected: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]


def _number(value: Any) -> float | None:
    """Return a finite float, treating booleans and malformed values as absent."""

    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _sample_rate(analysis: Any) -> float | None:
    value = _number(getattr(analysis, "sample_rate", None))
    return value if value is not None and value > 0 else None


def _duration_samples(analysis: Any) -> int | None:
    value = _number(getattr(analysis, "duration_samples", None))
    if value is None or value < 0 or not value.is_integer():
        return None
    return int(value)


def _track_id(analysis: Any) -> str:
    value = getattr(analysis, "track_id", None)
    return str(value) if value is not None else ""


def _cue_sample(cue: Any) -> int | None:
    value = _number(getattr(cue, "sample", None))
    if value is None or value < 0 or not value.is_integer():
        return None
    return int(value)


def _cue_confidence(cue: Any) -> float:
    value = _number(getattr(cue, "confidence", None))
    if value is None:
        return 0.0
    return max(0.0, min(1.0, value))


def _cue_kind(cue: Any) -> str:
    value = getattr(cue, "kind", "")
    return str(value) if value is not None else ""


def _cue_signature(cue: Any) -> tuple[Any, ...]:
    """Stable tie-break key, independent of object identity or repr()."""

    def seq_signature(value: Any) -> tuple[float, ...]:
        if isinstance(value, (str, bytes)):
            return ()
        try:
            return tuple(
                number
                for item in value
                if (number := _number(item)) is not None
            )
        except TypeError:
            return ()

    def scalar_signature(value: Any) -> tuple[int, float]:
        number = _number(value)
        # Include availability before the value so malformed/missing optional
        # descriptors never cause Python to compare ``None`` with a float.
        return (0, 0.0) if number is None else (1, number)

    sample = _cue_sample(cue)
    return (
        -1 if sample is None else sample,
        _cue_kind(cue),
        seq_signature(getattr(cue, "chroma", None)),
        seq_signature(getattr(cue, "timbre", None)),
        scalar_signature(getattr(cue, "rms", None)),
        scalar_signature(getattr(cue, "onset", None)),
        scalar_signature(getattr(cue, "tempo_bpm", None)),
        scalar_signature(getattr(cue, "confidence", None)),
    )


def _usable_cues(analysis: Any, min_spacing_seconds: float) -> tuple[list[Any], list[dict[str, Any]]]:
    """Sort and de-duplicate cues using seconds, not raw sample indices."""

    rate = _sample_rate(analysis)
    duration = _duration_samples(analysis)
    skipped: list[dict[str, Any]] = []
    if rate is None:
        return [], [{"reason": "invalid_sample_rate"}]

    cues = list(getattr(analysis, "cues", ()) or ())
    valid: list[Any] = []
    for index, cue in enumerate(cues):
        sample = _cue_sample(cue)
        if sample is None:
            skipped.append({"index": index, "reason": "invalid_sample"})
            continue
        # A cue at ``duration_samples`` has no sample frame and cannot anchor
        # a transition.  Treat the timeline as the half-open [0, duration)
        # interval used by the data contract.
        if duration is not None and sample >= duration:
            skipped.append({"index": index, "sample": sample, "reason": "cue_out_of_duration"})
            continue
        valid.append(cue)

    valid.sort(key=_cue_signature)
    kept: list[Any] = []
    seen_samples: set[int] = set()
    last_time: float | None = None
    for cue in valid:
        sample = _cue_sample(cue)
        assert sample is not None
        # A candidate identity is defined by track IDs and cue samples.  Two
        # backend records at the same sample therefore describe the same M1
        # anchor; retain the first item in the stable descriptor order instead
        # of creating duplicate candidate IDs and overwriting audit records.
        if sample in seen_samples:
            skipped.append({"sample": sample, "reason": "duplicate_sample"})
            continue
        seen_samples.add(sample)
        time = sample / rate
        if last_time is not None and time - last_time < min_spacing_seconds:
            skipped.append({
                "sample": sample,
                "reason": "min_spacing",
                "min_spacing_seconds": min_spacing_seconds,
            })
            continue
        kept.append(cue)
        last_time = time
    return kept, skipped


def _safe_vector(value: Any, expected: int | None = None) -> tuple[float, ...] | None:
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        vector = tuple(_number(item) for item in value)
    except TypeError:
        return None
    if not vector or any(item is None for item in vector):
        return None
    result = tuple(float(item) for item in vector if item is not None)
    if expected is not None and len(result) != expected:
        return None
    return result


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float | None:
    if len(left) != len(right):
        return None
    denominator = math.sqrt(sum(x * x for x in left) * sum(y * y for y in right))
    if denominator <= _EPS:
        return None
    return max(0.0, min(1.0, sum(x * y for x, y in zip(left, right)) / denominator))


def _harmonic_similarity(left: Any, right: Any) -> float | None:
    """Cosine compatibility of local chroma without an implicit pitch shift.

    The neutral M1 renderer does not transpose audio.  Maximising over circular
    pitch-class shifts would therefore claim that, for example, C and F-sharp
    are identical while applying no corresponding transform.  A future
    transposing renderer can expose a separate shifted metric plus an explicit
    transform cost; this baseline keeps the measured feature faithful to the
    current render policy.
    """

    a = _safe_vector(left, expected=12)
    b = _safe_vector(right, expected=12)
    if a is None or b is None:
        return None
    return _cosine(a, b)


def _positive_scalar(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number >= 0 else None


def _ratio_similarity(left: Any, right: Any) -> float | None:
    """Scale-free similarity for positive descriptors (RMS/onset/BPM)."""

    a, b = _positive_scalar(left), _positive_scalar(right)
    if a is None or b is None:
        return None
    if a <= _EPS and b <= _EPS:
        return 1.0
    # A small floor prevents two quiet cues from being treated as infinitely
    # far apart while still strongly penalising a quiet/loud mismatch.
    ratio = (a + _EPS) / (b + _EPS)
    return max(0.0, min(1.0, math.exp(-abs(math.log(ratio)))))


def _metric_average(values: Iterable[tuple[str, float | None]]) -> tuple[float | None, tuple[str, ...]]:
    available = [(name, value) for name, value in values if value is not None]
    if not available:
        return None, ()
    return sum(value for _, value in available) / len(available), tuple(name for name, _ in available)


def _pair_features(cue_a: Any, cue_b: Any) -> tuple[dict[str, Feature], dict[str, Feature], tuple[str, ...]]:
    """Compute measurable pair evidence and human-readable provenance notes."""

    chroma = _harmonic_similarity(
        getattr(cue_a, "chroma", None), getattr(cue_b, "chroma", None)
    )
    rms = _ratio_similarity(getattr(cue_a, "rms", None), getattr(cue_b, "rms", None))
    onset = _ratio_similarity(getattr(cue_a, "onset", None), getattr(cue_b, "onset", None))
    timbre = _cosine(
        _safe_vector(getattr(cue_a, "timbre", None)) or (),
        _safe_vector(getattr(cue_b, "timbre", None)) or (),
    )
    if not _safe_vector(getattr(cue_a, "timbre", None)) or not _safe_vector(getattr(cue_b, "timbre", None)):
        timbre = None
    tempo = _ratio_similarity(
        getattr(cue_a, "tempo_bpm", None), getattr(cue_b, "tempo_bpm", None)
    )

    horizontal_parts = [("harmonic_chroma", chroma), ("energy_rms", rms),
                        ("timbre_cosine", timbre), ("onset_context", onset),
                        ("tempo_context", tempo)]
    # The weights encode a transparent baseline hypothesis.  Missing metrics
    # are removed before normalisation, rather than turned into zero scores.
    weights = {"harmonic_chroma": 0.40, "energy_rms": 0.20,
               "timbre_cosine": 0.15, "onset_context": 0.10,
               "tempo_context": 0.15}
    available = [(name, value) for name, value in horizontal_parts if value is not None]
    if available:
        weight_total = sum(weights[name] for name, _ in available)
        horizontal_value = sum(weights[name] * value for name, value in available) / weight_total
        horizontal_source = "+".join(name for name, _ in available)
    else:
        horizontal_value = None
        horizontal_source = "no_pair_descriptors"

    continuity_value, continuity_names = _metric_average(
        [("harmonic_chroma", chroma), ("timbre_cosine", timbre),
         ("onset_context", onset), ("energy_rms", rms)]
    )
    continuity_source = "+".join(continuity_names) if continuity_names else "no_continuity_descriptors"
    pair_confidence = min(_cue_confidence(cue_a), _cue_confidence(cue_b))
    features = {
        "horizontal": Feature(horizontal_value, pair_confidence if horizontal_value is not None else None,
                               f"{_VERSION}:{horizontal_source}"),
        # A point-to-point retrieval does not overlap audio; vertical
        # compatibility is consequently an explicit unknown, never a zero.
        "vertical": Feature(None, None, f"{_VERSION}:no_overlap"),
        "continuity": Feature(continuity_value, pair_confidence if continuity_value is not None else None,
                               f"{_VERSION}:{continuity_source}"),
    }

    penalties: dict[str, Feature] = {}
    tempo_a = _positive_scalar(getattr(cue_a, "tempo_bpm", None))
    tempo_b = _positive_scalar(getattr(cue_b, "tempo_bpm", None))
    if tempo_a is not None and tempo_b is not None:
        # Transform penalty is measurable only when both local tempi exist.
        relative_change = abs(tempo_a - tempo_b) / max(tempo_a, tempo_b, _EPS)
        penalties["transform"] = Feature(
            max(0.0, min(1.0, relative_change)), pair_confidence,
            f"{_VERSION}:relative_tempo_change",
        )
    # No stem, clipping, or renderer data is available at M1; omitting the
    # artifact penalty prevents an unsupported audio-quality claim.
    notes = (
        f"pair_metrics={horizontal_source}",
        f"continuity_metrics={continuity_source}",
        "vertical=unknown_no_overlap",
        "creative=unknown_not_measured",
    )
    if "transform" not in penalties:
        notes += ("transform=unknown_missing_tempo",)
    notes += ("artifact=unknown_no_render_measurement",)
    return features, penalties, notes


def _candidate_id(track_a: str, track_b: str, sample_a: int, sample_b: int) -> str:
    payload = f"{_VERSION}|{track_a}|{sample_a}|{track_b}|{sample_b}|arbitrary_cue".encode("utf-8")
    return "cand_" + hashlib.sha256(payload).hexdigest()[:20]


def _bounded_pairs(length_a: int, length_b: int, limit: int) -> list[tuple[int, int]]:
    total = length_a * length_b
    if total <= limit:
        return [(index_a, index_b) for index_a in range(length_a) for index_b in range(length_b)]
    # Flattened row-major grid with evenly spaced, unique indices.  This gives
    # deterministic coverage of both timelines while respecting the bound.
    chosen: list[tuple[int, int]] = []
    previous = -1
    for step in range(limit):
        flat = (step * (total - 1)) // (limit - 1) if limit > 1 else 0
        if flat == previous:
            continue
        previous = flat
        chosen.append((flat // length_b, flat % length_b))
    return chosen


def _candidate_time(candidate: Candidate, analysis: Any, side: str) -> float:
    rate = _sample_rate(analysis) or 1.0
    sample = candidate.cue_out_sample if side == "a" else candidate.cue_in_sample
    return sample / rate


def search_transitions(
    a: Any,
    b: Any,
    top_k: int = 10,
    min_spacing_seconds: float = 2.0,
    max_candidates: int = 10000,
) -> SearchResult:
    """Retrieve and diversify directed A→B cue pairs.

    Candidate generation is a bounded Cartesian grid.  It is deterministic for
    the same analyses and parameters, including when the grid must be sampled.
    ``min_spacing_seconds`` is applied independently on each source timeline
    before the grid is formed, with each source's own sample rate.
    """

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    if not isinstance(max_candidates, int) or isinstance(max_candidates, bool) or max_candidates <= 0:
        raise ValueError("max_candidates must be a positive integer")
    spacing = _number(min_spacing_seconds)
    if spacing is None or spacing < 0:
        raise ValueError("min_spacing_seconds must be finite and non-negative")

    track_a, track_b = _track_id(a), _track_id(b)
    if not track_a or not track_b:
        raise ValueError("analyses must have non-empty track_id")
    if track_a == track_b:
        raise ValueError("search requires two different tracks")
    rate_a, rate_b = _sample_rate(a), _sample_rate(b)
    if rate_a is None or rate_b is None:
        raise ValueError("analyses must have positive sample_rate")

    cues_a, skipped_a = _usable_cues(a, spacing)
    cues_b, skipped_b = _usable_cues(b, spacing)
    grid_size = len(cues_a) * len(cues_b)
    grid = _bounded_pairs(len(cues_a), len(cues_b), max_candidates) if grid_size else []

    candidates: list[Candidate] = []
    reports: list[ScoreReport] = []
    rejected: list[dict[str, Any]] = []
    ranker = InterpretableRanker()

    for index_a, index_b in grid:
        cue_a, cue_b = cues_a[index_a], cues_b[index_b]
        sample_a, sample_b = _cue_sample(cue_a), _cue_sample(cue_b)
        assert sample_a is not None and sample_b is not None
        features, penalties, notes = _pair_features(cue_a, cue_b)
        candidate = Candidate(
            candidate_id=_candidate_id(track_a, track_b, sample_a, sample_b),
            track_a=track_a,
            track_b=track_b,
            cue_out_sample=sample_a,
            cue_in_sample=sample_b,
            # The M1 retrieval is a generic cue grid, but its measurable
            # evidence is harmonic/rhythmic.  Keep the mode within the
            # versioned transition-record enum consumed by the pipeline.
            mode="harmony_rhythm",
            features=features,
            penalties=penalties,
        )
        report = ranker.score(candidate)
        # Add search-specific explanations without changing the shared report
        # contract.  The report remains immutable, so rebuild it explicitly.
        report = ScoreReport(
            report.candidate_id, report.valid, report.score,
            report.contributions, report.penalties, report.missing,
            (*report.notes, *notes),
        )
        candidates.append(candidate)
        reports.append(report)

    ranked = [
        (index, report)
        for index, report in enumerate(reports)
        if report.valid and report.score is not None
    ]
    ranked.sort(key=lambda item: (-float(item[1].score), candidates[item[0]].candidate_id))
    selected_indices: list[int] = []
    selected_ids: list[str] = []
    for index, report in ranked:
        candidate = candidates[index]
        if len(selected_indices) >= top_k:
            rejected.append({"candidate_id": candidate.candidate_id, "reason": "top_k_limit"})
            continue
        candidate_a_time = _candidate_time(candidate, a, "a")
        candidate_b_time = _candidate_time(candidate, b, "b")
        too_close = False
        for selected_index in selected_indices:
            previous = candidates[selected_index]
            previous_a = _candidate_time(previous, a, "a")
            previous_b = _candidate_time(previous, b, "b")
            if (abs(candidate_a_time - previous_a) < spacing or
                    abs(candidate_b_time - previous_b) < spacing):
                too_close = True
                break
        if too_close:
            rejected.append({"candidate_id": candidate.candidate_id, "reason": "diversity_spacing"})
            continue
        selected_indices.append(index)
        selected_ids.append(candidate.candidate_id)

    selected_set = set(selected_ids)
    for candidate, report in zip(candidates, reports):
        if candidate.candidate_id in selected_set:
            continue
        if not report.valid:
            rejected.append({
                "candidate_id": candidate.candidate_id,
                "reason": "invalid_score",
                "details": list(report.notes),
            })

    # Keep rejection order reproducible: ranked reasons first are useful for
    # debugging, while this final diagnostics map is intentionally scalar.
    diagnostics = {
        "algorithm": _VERSION,
        "track_a": track_a,
        "track_b": track_b,
        "sample_rate_a": rate_a,
        "sample_rate_b": rate_b,
        "cue_count_a": len(getattr(a, "cues", ()) or ()),
        "cue_count_b": len(getattr(b, "cues", ()) or ()),
        "usable_cue_count_a": len(cues_a),
        "usable_cue_count_b": len(cues_b),
        "skipped_cue_count_a": len(skipped_a),
        "skipped_cue_count_b": len(skipped_b),
        "skipped_cues_a": tuple(skipped_a),
        "skipped_cues_b": tuple(skipped_b),
        "grid_size": grid_size,
        "generated_count": len(candidates),
        "grid_truncated": grid_size > max_candidates,
        "max_candidates": max_candidates,
        "min_spacing_seconds": spacing,
        "selected_count": len(selected_ids),
        "rejected_count": len(rejected),
        "no_cues": not cues_a or not cues_b,
    }
    return SearchResult(tuple(candidates), tuple(reports), tuple(selected_ids), tuple(rejected), diagnostics)


__all__ = ["SearchResult", "search_transitions"]
