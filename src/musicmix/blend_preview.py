"""Controlled equal-power overlap study renderer.

This module is deliberately separate from the default preview/handoff
renderers.  It answers one narrow question: if two fixed source cue anchors
are mapped to the same output sample and a short equal-power overlap is
placed around that anchor, what exactly is rendered?  It does not choose
cues, time-stretch, beat-match, or claim that the result is musically good.

The existing short vocal guard in the project is not evidence that a one
second overlap is safe.  Root code must call :func:`overlap_vocal_safety`
with explicit separated-vocal acoustic-silence evidence before exposing this
renderer, and must still validate the result by listening.
"""
from __future__ import annotations

import math
from pathlib import Path
import wave
from typing import Any, Mapping

import numpy as np

from .audio import AudioTrack
from .contracts import Candidate
from .ingest import sha256_file


BLEND_PREVIEW_VERSION = "equal_power_overlap_v1"
GLOBAL_GAIN = 0.7
TEMPO_NORMALIZATION = "half_or_double_only"


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _half_overlap_samples(overlap_seconds: float, rate: int) -> int:
    """Return the positive half-overlap used by both renderer and safety audit."""

    requested = _finite(overlap_seconds, "overlap_seconds")
    if requested <= 0:
        raise ValueError("overlap_seconds must be positive")
    return max(1, round(requested * int(rate) / 2.0))


def _normalised_tempo_ratio(tempo_a: float, tempo_b: float) -> tuple[float, str, float]:
    """Compare tempos after explicit half/double normalisation.

    A reported BPM may be a half-time or double-time annotation of the same
    pulse.  Only one octave is normalised; quarter/quadruple relationships are
    not silently accepted.  The returned ratio is the candidate closest to 1.
    """

    ratio = tempo_b / tempo_a
    candidates = (
        (ratio, "as_reported"),
        (ratio * 2.0, "b_is_half_time_doubled_to_compare"),
        (ratio / 2.0, "b_is_double_time_halved_to_compare"),
    )
    normalised, convention = min(candidates, key=lambda item: abs(item[0] - 1.0))
    return float(normalised), convention, float(abs(normalised - 1.0))


def _window_sha256(samples: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.asarray(samples, dtype="<f8").tobytes()).hexdigest()


def render_blend_preview(
    track_a: AudioTrack,
    track_b: AudioTrack,
    candidate: Candidate,
    path: Path,
    *,
    context_seconds: float = 10.0,
    overlap_seconds: float = 1.0,
    tempo_a: float | None = None,
    tempo_b: float | None = None,
    max_relative_tempo_error: float = 0.02,
) -> dict[str, Any]:
    """Render a fixed-anchor equal-power overlap for controlled study.

    Timeline (all output samples)::

        0                       join                2*context
        |---- A ----|---- overlap ----|---- B ----|
                    ^ A cue and B cue both map here

    A's source window is ``[cue_out-context, cue_out+half)``.  B's source
    window is ``[cue_in-half, cue_in+context)``.  The output is exactly
    ``2*context`` samples long.  Both cue anchors map to ``context``.
    """

    context_value = _finite(context_seconds, "context_seconds")
    if context_value <= 0:
        raise ValueError("context_seconds must be positive")
    overlap_value = _finite(overlap_seconds, "overlap_seconds")
    if overlap_value <= 0:
        raise ValueError("overlap_seconds must be positive")
    max_error = _finite(max_relative_tempo_error, "max_relative_tempo_error")
    if max_error < 0:
        raise ValueError("max_relative_tempo_error must be non-negative")

    if track_a.sample_rate != track_b.sample_rate:
        raise ValueError("Blend preview requires identical source sample rates")
    rate = int(track_a.sample_rate)
    if rate <= 0:
        raise ValueError("sample_rate must be positive")
    if track_a.samples.ndim != 2 or track_b.samples.ndim != 2:
        raise ValueError("Source samples must have shape (frames, channels)")
    if track_a.samples.shape[1] != track_b.samples.shape[1]:
        raise ValueError("Source channel counts must match")
    if candidate.track_a != track_a.track_id or candidate.track_b != track_b.track_id:
        raise ValueError("Candidate track identity mismatch")

    context = round(context_value * rate)
    if context < 1:
        raise ValueError("context_seconds is shorter than one sample")
    half = _half_overlap_samples(overlap_value, rate)
    if half > context:
        raise ValueError("overlap_seconds/2 must not exceed context_seconds")

    out_cue = int(candidate.cue_out_sample)
    in_cue = int(candidate.cue_in_sample)
    a_start, a_end = out_cue - context, out_cue + half
    b_start, b_end = in_cue - half, in_cue + context
    if a_start < 0:
        raise ValueError("Insufficient outgoing context before cue_out; padding is forbidden")
    if a_end > track_a.duration_samples:
        raise ValueError("Insufficient outgoing extension after cue_out; padding is forbidden")
    if b_start < 0:
        raise ValueError("Insufficient incoming pre-roll before cue_in; padding is forbidden")
    if b_end > track_b.duration_samples:
        raise ValueError("Insufficient incoming context after cue_in; padding is forbidden")

    a_window = np.asarray(track_a.samples[a_start:a_end], dtype=np.float64)
    b_window = np.asarray(track_b.samples[b_start:b_end], dtype=np.float64)
    if a_window.shape[0] != context + half or b_window.shape[0] != context + half:
        raise ValueError("Internal source window length mismatch")
    if not np.isfinite(a_window).all() or not np.isfinite(b_window).all():
        raise ValueError("Source windows contain non-finite samples")

    beat: dict[str, Any] = {
        "status": "unknown",
        "tempo_a": None if tempo_a is None else _finite(tempo_a, "tempo_a"),
        "tempo_b": None if tempo_b is None else _finite(tempo_b, "tempo_b"),
        "normalization": TEMPO_NORMALIZATION,
        "max_relative_tempo_error": max_error,
        "phase_alignment": "not_evaluated",
        "reasons": [],
    }
    if beat["tempo_a"] is not None and beat["tempo_a"] <= 0:
        raise ValueError("tempo_a must be positive")
    if beat["tempo_b"] is not None and beat["tempo_b"] <= 0:
        raise ValueError("tempo_b must be positive")
    if beat["tempo_a"] is None or beat["tempo_b"] is None:
        if beat["tempo_a"] is None:
            beat["reasons"].append("tempo_a_missing")
        if beat["tempo_b"] is None:
            beat["reasons"].append("tempo_b_missing")
        beat["reasons"].append("beat_coherence_unknown_no_beat_match_claim")
    else:
        normalised, convention, relative_error = _normalised_tempo_ratio(
            beat["tempo_a"], beat["tempo_b"]
        )
        if relative_error > max_error:
            raise ValueError(
                "Tempo mismatch after half/double normalization: "
                f"tempo_a={beat['tempo_a']}, tempo_b={beat['tempo_b']}, "
                f"normalized_ratio={normalised:.6f} ({convention}), "
                f"relative_error={relative_error:.6f} > "
                f"max_relative_tempo_error={max_error}"
            )
        beat.update(
            {
                "status": "tempo_compatible_no_phase_evidence",
                "normalized_ratio": normalised,
                "normalization_convention": convention,
                "relative_error": relative_error,
            }
        )

    output_length = 2 * context
    a_gain = np.zeros((output_length, 1), dtype=np.float64)
    b_gain = np.zeros((output_length, 1), dtype=np.float64)
    a_gain[: context + half] = 1.0
    b_gain[context - half :] = 1.0
    overlap_indices = np.arange(2 * half, dtype=np.float64)
    theta = (np.pi / 2.0) * (overlap_indices / float(2 * half))
    a_gain[context - half : context + half, 0] = np.cos(theta)
    b_gain[context - half : context + half, 0] = np.sin(theta)

    a_placed = np.zeros((output_length, a_window.shape[1]), dtype=np.float64)
    b_placed = np.zeros((output_length, b_window.shape[1]), dtype=np.float64)
    a_placed[: context + half] = a_window
    b_placed[context - half :] = b_window
    signal = (a_placed * a_gain + b_placed * b_gain) * GLOBAL_GAIN
    if not np.isfinite(signal).all():
        raise ValueError("Rendered signal contains non-finite samples")

    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    clipped = int(np.count_nonzero(np.abs(signal) > 1.0))
    pcm = np.round(np.clip(signal, -1.0, 1.0) * 32767.0).astype("<i2")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with destination.open("xb") as raw:
            created = True
            with wave.open(raw, "wb") as stream:
                stream.setnchannels(pcm.shape[1])
                stream.setsampwidth(2)
                stream.setframerate(rate)
                stream.writeframes(pcm.tobytes())
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise

    warnings = [
        "Controlled overlap study only; this does not replace the default renderer.",
        "A extends past cue_out and B begins before cue_in; both cue anchors map to join_sample.",
        "The existing short (~40 ms) vocal guard is not sufficient for the requested "
        "overlap, including a one-second overlap. Root must call overlap_vocal_safety and "
        "require explicit acoustic-silence support over both extensions before exposing "
        "this renderer.",
        "No time stretch, pitch correction, beat tracking, or model downloads; source cues are not re-estimated.",
        "Tempo compatibility, if reported, is ratio-only and does not establish beat phase or downbeat alignment.",
    ]
    if beat["status"] == "unknown":
        warnings.append("Beat coherence unknown: tempo evidence is missing; no beat-match claim is made.")
    else:
        warnings.append("Tempo ratio is within tolerance after half/double normalization; beat phase remains unverified.")
    if clipped:
        warnings.append("Output clipping detected; fixed global gain did not prevent it for these inputs.")

    output_sha256 = sha256_file(destination)
    return {
        "candidate_id": candidate.candidate_id,
        "render_version": BLEND_PREVIEW_VERSION,
        "version": BLEND_PREVIEW_VERSION,
        "path": str(destination.resolve()),
        "sha256": output_sha256,
        "checksum": output_sha256,
        "checksum_algorithm": "sha256",
        "sample_rate": rate,
        "channels": int(pcm.shape[1]),
        "duration_samples": output_length,
        "duration_seconds": output_length / rate,
        "context_seconds": context_value,
        "context_samples": context,
        "overlap_seconds": overlap_value,
        "overlap_samples": 2 * half,
        "half_overlap_samples": half,
        "effective_overlap_seconds": (2 * half) / rate,
        "join_sample": context,
        "join_seconds": context / rate,
        "source_windows": {
            "a": {
                "track_id": track_a.track_id,
                "start_sample": a_start,
                "end_sample": a_end,
                "length_samples": context + half,
                "output_start_sample": 0,
                "cue_offset_samples": context,
                "cue_output_sample": context,
                "window_sha256": _window_sha256(a_window),
            },
            "b": {
                "track_id": track_b.track_id,
                "start_sample": b_start,
                "end_sample": b_end,
                "length_samples": context + half,
                "output_start_sample": context - half,
                "cue_offset_samples": half,
                "cue_output_sample": context,
                "window_sha256": _window_sha256(b_window),
            },
        },
        "anchor_offsets": {
            "join_sample": context,
            "a": {
                "cue_source_sample": out_cue,
                "window_start_source_sample": a_start,
                "cue_offset_in_window_samples": context,
                "output_anchor_sample": context,
            },
            "b": {
                "cue_source_sample": in_cue,
                "window_start_source_sample": b_start,
                "cue_offset_in_window_samples": half,
                "output_anchor_sample": context,
            },
        },
        "gains": {
            "global": GLOBAL_GAIN,
            "curve": "equal_power_cosine_sine",
            "a_at_join_raw": float(a_gain[context, 0]),
            "b_at_join_raw": float(b_gain[context, 0]),
            "a_at_join_applied": float(a_gain[context, 0] * GLOBAL_GAIN),
            "b_at_join_applied": float(b_gain[context, 0] * GLOBAL_GAIN),
            "a_outside_overlap": 1.0,
            "b_outside_overlap": 1.0,
        },
        "overlap_timeline": {
            "anchor_sample": context,
            "a_only": {"start_sample": 0, "end_sample": context - half},
            "overlap": {"start_sample": context - half, "end_sample": context + half},
            "b_only": {"start_sample": context + half, "end_sample": output_length},
            "outgoing_extension": {"start_sample": context, "end_sample": context + half},
            "incoming_preroll": {"start_sample": context - half, "end_sample": context},
        },
        "source_margins": {
            "a_before_cue_samples": context,
            "a_after_cue_samples": half,
            "b_before_cue_samples": half,
            "b_after_cue_samples": context,
        },
        "beat_coherence": beat,
        "beat_coherence_status": beat["status"],
        "vocal_safety": {
            "status": "not_evaluated",
            "required_check": "overlap_vocal_safety",
            "reason": "The renderer does not consume vocal evidence. Root must audit the "
                      "outgoing extension and incoming pre-roll before exposure.",
        },
        "peak": peak,
        "clipped_samples": clipped,
        "warnings": warnings,
    }


def _is_occupancy(status: str) -> bool:
    return status.startswith("vocal_occupancy")


def _is_unknown(status: str) -> bool:
    return status.startswith("unknown")


def _is_explicit_silence(status: str) -> bool:
    return status in {"acoustic_silence_supported", "acoustic_silence"}


def _as_document(vocals: Any, label: str) -> dict[str, Any]:
    """Accept either a boundary document or a separated-vocal AudioTrack."""

    if isinstance(vocals, AudioTrack):
        from .vocal_boundaries import acoustic_intervals

        intervals, diagnostics = acoustic_intervals(vocals)
        coverage = []
        for item in intervals:
            raw_status = str(item.get("status", ""))
            mapped = {
                "acoustic_silence": "acoustic_silence_supported",
                "vocal_occupancy": "vocal_occupancy_supported",
                "unknown_acoustic_margin": "unknown_acoustic_margin",
            }.get(raw_status, raw_status)
            coverage.append(
                {
                    "start_sample": int(item["start_sample"]),
                    "end_sample": int(item["end_sample"]),
                    "status": mapped,
                    "evidence": str(item.get("evidence", "separated_vocals_rms_heuristic")),
                }
            )
        return {
            "track_id": vocals.track_id,
            "sample_rate": vocals.sample_rate,
            "duration_samples": vocals.duration_samples,
            "coverage_intervals": coverage,
            "acoustic_occupancy": list(intervals),
            "acoustic_diagnostics": dict(diagnostics),
            "words": [],
        }
    if not isinstance(vocals, Mapping):
        raise TypeError(f"{label} must be a boundary document or AudioTrack")
    return dict(vocals)


def _document_duration(doc: Mapping[str, Any]) -> int | None:
    duration = doc.get("duration_samples")
    if duration is not None:
        try:
            value = int(duration)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None
    ends: list[int] = []
    for key in ("coverage_intervals", "acoustic_occupancy", "words"):
        for item in doc.get(key, ()) or ():
            try:
                ends.append(int(item.get("end_sample", 0)))
            except (AttributeError, TypeError, ValueError):
                continue
    return max(ends) if ends else None


def _interval_status(
    label: str,
    start: int,
    end: int,
    doc: Mapping[str, Any],
    duration: int | None,
) -> tuple[str, list[str]]:
    if start < 0 or end <= start:
        return "rejected", [f"{label}_invalid_interval"]
    if duration is None:
        return "unknown", [f"{label}_duration_missing"]
    if end > duration:
        return "rejected", [f"{label}_beyond_document_end"]

    words = doc.get("words") or ()
    if any(
        int(word.get("start_sample", 0)) < end and int(word.get("end_sample", 0)) > start
        for word in words
    ):
        return "rejected", [f"{label}_overlaps_known_word"]

    coverage = doc.get("coverage_intervals")
    if coverage:
        cursor = start
        statuses: list[str] = []
        for item in sorted(coverage, key=lambda value: int(value.get("start_sample", 0))):
            item_start = int(item.get("start_sample", 0))
            item_end = int(item.get("end_sample", 0))
            if item_end <= cursor or item_start >= end:
                continue
            if item_start > cursor:
                return "unknown", [f"{label}_coverage_gap"]
            if item_end > cursor:
                statuses.append(str(item.get("status", "")))
                cursor = item_end
            if cursor >= end:
                break
        if cursor < end or not statuses:
            return "unknown", [f"{label}_coverage_gap"]
        if any(_is_occupancy(status) for status in statuses):
            return "rejected", [f"{label}_vocal_occupancy"]
        if all(_is_explicit_silence(status) for status in statuses):
            return "supported", [f"{label}_explicit_acoustic_silence_supported"]
        if any(_is_unknown(status) for status in statuses):
            return "unknown", [f"{label}_unknown_acoustic_evidence"]
        if any(status.startswith("asr") for status in statuses):
            return "unknown", [f"{label}_asr_gap_is_not_acoustic_silence"]
        return "unknown", [f"{label}_not_explicit_acoustic_silence"]

    acoustic = doc.get("acoustic_occupancy")
    if acoustic:
        cursor = start
        statuses = []
        for item in sorted(acoustic, key=lambda value: int(value.get("start_sample", 0))):
            item_start = int(item.get("start_sample", 0))
            item_end = int(item.get("end_sample", 0))
            if item_end <= cursor or item_start >= end:
                continue
            if item_start > cursor:
                return "unknown", [f"{label}_acoustic_coverage_gap"]
            if item_end > cursor:
                statuses.append(str(item.get("status", "")))
                cursor = item_end
            if cursor >= end:
                break
        if cursor < end or not statuses:
            return "unknown", [f"{label}_acoustic_coverage_gap"]
        if any(_is_occupancy(status) for status in statuses):
            return "rejected", [f"{label}_vocal_occupancy"]
        if all(_is_explicit_silence(status) for status in statuses):
            return "supported", [f"{label}_measured_acoustic_silence"]
        return "unknown", [f"{label}_unknown_acoustic_evidence"]

    return "unknown", [f"{label}_missing_acoustic_evidence"]


def overlap_vocal_safety(
    candidate: Candidate,
    vocals_a: Any,
    vocals_b: Any,
    rate: int,
    overlap_seconds: float,
) -> dict[str, Any]:
    """Fail-closed vocal safety audit for the overlap extensions.

    The outgoing extension is ``[cue_out, cue_out + overlap/2)`` because A
    continues past its cue.  The incoming pre-roll is
    ``[cue_in - overlap/2, cue_in)`` because B starts before its cue.

    ``vocals_a``/``vocals_b`` may be boundary documents produced by
    :func:`musicmix.vocal_boundaries.build_boundary_document` or separated
    vocal :class:`AudioTrack` objects.  An AudioTrack is accepted only when
    its ``track_id`` is the candidate source id; a conventional stem content
    hash is rejected unless the caller explicitly binds the stem to its
    source.  Prefer a boundary document built from the canonical source.
    Only explicit separated-vocal acoustic silence supports an interval.
    ASR gaps, missing coverage, and unknown acoustic margins are ``unknown``;
    known words or measured vocal occupancy are ``rejected``.  Missing
    evidence never becomes support.
    """

    rate_value = int(rate)
    if rate_value <= 0:
        raise ValueError("rate must be positive")
    half = _half_overlap_samples(overlap_seconds, rate_value)
    out_start = int(candidate.cue_out_sample)
    out_end = out_start + half
    in_start = int(candidate.cue_in_sample) - half
    in_end = int(candidate.cue_in_sample)

    doc_a = _as_document(vocals_a, "vocals_a")
    doc_b = _as_document(vocals_b, "vocals_b")
    binding_rejected: list[str] = []
    binding_unknown: list[str] = []

    for label, doc, expected in (
        ("vocals_a", doc_a, candidate.track_a),
        ("vocals_b", doc_b, candidate.track_b),
    ):
        track_id = doc.get("track_id")
        if track_id is None:
            binding_unknown.append(f"{label}_track_identity_missing")
        elif str(track_id) != str(expected):
            binding_rejected.append(f"{label}_track_identity_mismatch")
        doc_rate = doc.get("sample_rate")
        if doc_rate is None:
            binding_unknown.append(f"{label}_sample_rate_missing")
        else:
            try:
                parsed_rate = int(doc_rate)
            except (TypeError, ValueError):
                binding_rejected.append(f"{label}_sample_rate_invalid")
            else:
                if parsed_rate != rate_value:
                    binding_rejected.append(f"{label}_sample_rate_mismatch")

    duration_a = _document_duration(doc_a)
    duration_b = _document_duration(doc_b)
    out_status, out_reasons = _interval_status(
        "outgoing_extension", out_start, out_end, doc_a, duration_a
    )
    in_status, in_reasons = _interval_status(
        "incoming_preroll", in_start, in_end, doc_b, duration_b
    )

    binding_reasons = list(dict.fromkeys(binding_rejected + binding_unknown))
    if binding_rejected:
        out_status = "rejected"
        in_status = "rejected"

    side_statuses = (out_status, in_status)
    if "rejected" in side_statuses or binding_rejected:
        status = "rejected"
    elif "unknown" in side_statuses or binding_unknown:
        status = "unknown"
    else:
        status = "supported"

    reasons = list(dict.fromkeys(binding_reasons + out_reasons + in_reasons))
    if status == "supported":
        warnings = [
            "Explicit acoustic silence supports the extensions, but it does not prove "
            "musical continuity, absence of bleed, or a good listening result.",
            "This audit is specific to the requested overlap; the short default vocal "
            "guard is not reused as evidence.",
        ]
    else:
        warnings = [
            "Overlap is not supported by explicit acoustic-silence evidence; do not "
            "expose this renderer for this candidate.",
            "ASR absence, missing coverage, and unknown acoustic margins are not silence.",
        ]

    return {
        "status": status,
        "outgoing_extension": {
            "start_sample": out_start,
            "end_sample": out_end,
            "status": out_status,
            "reasons": out_reasons,
        },
        "incoming_preroll": {
            "start_sample": in_start,
            "end_sample": in_end,
            "status": in_status,
            "reasons": in_reasons,
        },
        "half_overlap_samples": half,
        "overlap_seconds": float(overlap_seconds),
        "effective_overlap_seconds": (2 * half) / rate_value,
        "rate": rate_value,
        "evidence_policy": (
            "requires_explicit_acoustic_silence_supported; ASR gaps are unknown; "
            "known words or vocal occupancy reject"
        ),
        "reasons": reasons,
        "warnings": warnings,
    }


__all__ = [
    "BLEND_PREVIEW_VERSION",
    "GLOBAL_GAIN",
    "overlap_vocal_safety",
    "render_blend_preview",
]
