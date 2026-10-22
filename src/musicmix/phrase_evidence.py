"""Conservative phrase and vocal-boundary evidence for transition policy.

The input is the persisted vocal-boundary document used by
``musicmix.vocal_boundaries``.  This module deliberately keeps three things
separate: an observed word gap, an ASR segment endpoint, and a real musical
phrase closure.  The first two are useful timing observations; neither is a
certificate of syntax or musical completion.

All times in this policy are integer samples on the canonical 48 kHz clock.
ASR punctuation and section labels are retained as diagnostics only.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


CANONICAL_SAMPLE_RATE = 48_000
PHRASE_EVIDENCE_VERSION = "phrase-evidence-v2-timing-separated"
PHRASE_EVIDENCE_SCHEMA = "musicmix.phrase-evidence.1"


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _sample(value: Any, *, name: str) -> int:
    number = _number(value)
    if number is None or number != round(number):
        raise ValueError(f"{name} must be a finite integer sample")
    return int(number)


def _records(payload: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    raw = payload.get(key, ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    result: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        start = _number(item.get("start_sample", item.get("start")))
        end = _number(item.get("end_sample", item.get("end")))
        if start is None or end is None or end <= start:
            continue
        result.append({
            "start_sample": int(round(start)),
            "end_sample": int(round(end)),
            **{str(k): v for k, v in item.items()
               if str(k) not in {"start_sample", "end_sample", "start", "end"}},
        })
    return sorted(result, key=lambda item: (item["start_sample"], item["end_sample"]))


def _overlap(cue: int, item: Mapping[str, Any]) -> bool:
    return int(item["start_sample"]) <= cue < int(item["end_sample"])


def _contains(intervals: Sequence[Mapping[str, Any]], cue: int) -> dict[str, Any] | None:
    for item in intervals:
        if _overlap(cue, item):
            return dict(item)
    return None


def _unknown_at(payload: Mapping[str, Any], cue: int) -> bool:
    for item in _records(payload, "uncertain_regions") + _records(payload, "uncertain"):
        if _overlap(cue, item):
            return True
    for item in _records(payload, "coverage_intervals"):
        if _overlap(cue, item) and str(item.get("status", "")).startswith("unknown"):
            return True
    return False


def _verified_acoustic_silence(payload: Mapping[str, Any], cue: int) -> dict[str, Any] | None:
    """Return explicit separated-vocal silence covering ``cue``.

    A raw ASR gap or an ``acoustic_silence`` heuristic is insufficient here;
    the persisted boundary document's reconciled coverage must say
    ``acoustic_silence_supported``.  This permits a broad ASR segment to be
    superseded by measured internal silence while retaining the distinction
    between clearance and phrase closure.
    """

    for key in ("coverage_intervals", "acoustic_intervals"):
        for item in _records(payload, key):
            if _overlap(cue, item) and str(item.get("status", "")) == "acoustic_silence_supported":
                return dict(item)
    for item in _records(payload, "acoustic_occupancy"):
        if _overlap(cue, item) and str(item.get("status", "")) == "acoustic_silence_supported":
            return dict(item)
    return None


def _identity(payload: Mapping[str, Any], sample_rate: int, duration: int) -> dict[str, Any]:
    """Validate identity metadata without requiring it for small callers.

    The persisted document normally contains a content hash and canonical
    URI.  Minimal test or exploratory payloads may omit those fields, so the
    result reports their absence rather than inventing an identity.
    """

    checks: dict[str, Any] = {
        "sample_rate": {"provided": payload.get("sample_rate"), "valid": True},
        "duration_samples": {"provided": payload.get("duration_samples"), "valid": True},
        "canonical_file_sha256": {
            "provided": payload.get("canonical_file_sha256"),
            "valid": isinstance(payload.get("canonical_file_sha256"), str)
            and len(str(payload.get("canonical_file_sha256"))) == 64,
        },
        "canonical_audio_uri": {
            "provided": payload.get("canonical_audio_uri"),
            "valid": isinstance(payload.get("canonical_audio_uri"), str)
            and bool(str(payload.get("canonical_audio_uri")).strip()),
        },
    }
    declared_rate = _number(payload.get("sample_rate"))
    if declared_rate is not None:
        checks["sample_rate"]["valid"] = int(round(declared_rate)) == sample_rate
    declared_duration = _number(payload.get("duration_samples"))
    if declared_duration is not None:
        checks["duration_samples"]["valid"] = int(round(declared_duration)) == duration
    required = ("sample_rate", "duration_samples")
    valid = all(
        bool(checks[name]["valid"]) and checks[name]["provided"] is not None
        for name in required
    )
    warnings = [
        f"missing_{name}" for name in ("canonical_file_sha256", "canonical_audio_uri")
        if not checks[name]["valid"] and checks[name]["provided"] is None
    ]
    warnings.extend(
        f"missing_{name}" for name in required if checks[name]["provided"] is None
    )
    mismatches = [name for name, check in checks.items() if check["provided"] is not None and not check["valid"]]
    return {
        "track_id": payload.get("track_id"),
        "canonical_sample_rate": sample_rate,
        "checks": checks,
        "valid": valid,
        "content_identity_complete": all(
            bool(checks[name]["valid"])
            for name in ("canonical_file_sha256", "canonical_audio_uri")
        ),
        "warnings": warnings,
        "mismatches": mismatches,
    }


def _clearance(
    *, direction: str, cue: int, words: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]], duration: int,
) -> dict[str, Any]:
    if direction == "outgoing":
        next_word = next((item for item in words if int(item["start_sample"]) >= cue), None)
        previous_word = next((item for item in reversed(words) if int(item["end_sample"]) <= cue), None)
        edge = int(next_word["start_sample"]) - cue if next_word else duration - cue
        return {
            "status": "observed" if edge >= 0 else "unknown",
            "samples": max(0, edge),
            "to": "next_known_word_start" if next_word else "track_end",
            "previous_word_end_sample": previous_word.get("end_sample") if previous_word else None,
        }
    previous_word = next((item for item in reversed(words) if int(item["end_sample"]) <= cue), None)
    next_word = next((item for item in words if int(item["start_sample"]) >= cue), None)
    edge = cue - int(previous_word["end_sample"]) if previous_word else cue
    return {
        "status": "observed" if edge >= 0 else "unknown",
        "samples": max(0, edge),
        "to": "previous_known_word_end" if previous_word else "track_start",
        "next_word_start_sample": next_word.get("start_sample") if next_word else None,
    }


def analyze_phrase_evidence(
    vocals: Mapping[str, Any] | None,
    cue_sample: int,
    sample_rate: int,
    duration_samples: int,
    direction: str,
) -> dict[str, Any]:
    """Return JSON-ready local vocal evidence around a cue.

    ``direction`` is ``outgoing``/``A_out`` for the source deck and
    ``incoming``/``B_in`` for the destination deck.  A missing vocal document
    is an unknown observation, never silence.  A known word overlap is unsafe
    for a boundary, while a word gap or ASR segment endpoint is only timing
    evidence.  No return field claims a syntactic or musical phrase closure
    unless a provider explicitly supplies one under ``phrase_closure``.
    """

    if int(sample_rate) != CANONICAL_SAMPLE_RATE:
        raise ValueError(
            f"sample_rate must be the canonical {CANONICAL_SAMPLE_RATE} Hz timeline"
        )
    duration = _sample(duration_samples, name="duration_samples")
    cue = _sample(cue_sample, name="cue_sample")
    if duration < 0 or cue < 0 or cue > duration:
        raise ValueError("cue_sample must lie on [0, duration_samples]")
    direction_value = str(direction).strip().lower()
    direction_aliases = {"a_out": "outgoing", "out": "outgoing", "b_in": "incoming", "in": "incoming"}
    direction_value = direction_aliases.get(direction_value, direction_value)
    if direction_value not in {"outgoing", "incoming"}:
        raise ValueError("direction must be outgoing/incoming (or A_out/B_in)")

    payload: Mapping[str, Any] = vocals if isinstance(vocals, Mapping) else {}
    words = _records(payload, "words")
    segments = _records(payload, "asr_segments") + _records(payload, "segments")
    segments.sort(key=lambda item: (item["start_sample"], item["end_sample"]))
    phrases = _records(payload, "phrases")
    active_word = _contains(words, cue)
    active_segment = _contains(segments, cue)
    acoustic_silence = _verified_acoustic_silence(payload, cue)
    coverage_records = (
        _records(payload, "coverage_intervals")
        + _records(payload, "coverage")
        + _records(payload, "acoustic_occupancy")
    )
    # Keep global availability for diagnostics, but gate clearance on local
    # evidence at this cue.  A segment elsewhere in the track cannot certify
    # an uncovered gap.  Segment ends are inclusive endpoint observations;
    # ordinary occupancy remains half-open.
    segment_end_observed = any(int(item["end_sample"]) == cue for item in segments)
    local_segment_coverage = active_segment is not None or segment_end_observed
    local_coverage_records = [item for item in coverage_records if _overlap(cue, item)]
    local_coverage_present = bool(local_segment_coverage or local_coverage_records)
    asr_coverage_present = bool(segments or coverage_records)
    # A known word remains unsafe even if an acoustic provider conflicts with
    # it.  Decoder segments are occupancy envelopes, not word boundaries:
    # segment-only overlap therefore leaves phrase continuity unknown and does
    # not become an ``active_utterance`` failure.  A reconciled acoustic
    # interval can support timing clearance inside such an envelope.
    active_utterance = active_word is not None
    unknown = not payload or _unknown_at(payload, cue)

    reasons: list[str] = []
    punctuation = False
    for item in words:
        text = str(item.get("text", "")).strip()
        if text and text[-1:] in ".!?;:":
            punctuation = True
            break
    if punctuation:
        reasons.append("punctuation_present_but_not_certified_syntactic_closure")

    # Explicit closure can only come from a provider that names itself.  The
    # phrase records emitted by vocal_boundaries are ASR word-gap groups and
    # therefore do not satisfy this branch.
    closure_raw = payload.get("phrase_closure")
    if closure_raw is None:
        for alias in ("phrase_boundary", "structural_arrival", "phrase_or_structural_arrival"):
            if alias in payload:
                closure_raw = payload.get(alias)
                break
    if closure_raw is None and isinstance(payload.get("phrase_structure"), Mapping):
        closure_raw = payload["phrase_structure"].get("closure", payload["phrase_structure"].get("arrival"))
    closure_provider = None
    closure_supported = False
    if isinstance(closure_raw, Mapping):
        closure_provider = closure_raw.get("provider") or closure_raw.get("source")
        closure_supported = bool(
            closure_raw.get("supported") is True
            and isinstance(closure_provider, str)
            and bool(closure_provider.strip())
        )
    elif closure_raw is True:
        reasons.append("phrase_closure_provider_missing")

    if active_word is not None:
        boundary_class = "active_utterance"
        boundary_source = "known_asr_word"
        status = "unsafe"
        reasons.append("cue_overlaps_active_utterance")
    elif acoustic_silence is not None:
        boundary_class = "acoustic_clearance_supported"
        boundary_source = "separated_vocal_acoustic_silence"
        status = "supported"
        reasons.append("verified_internal_acoustic_silence")
        if active_segment is not None:
            reasons.append("active_asr_segment_overridden_by_verified_acoustic_silence")
        reasons.append("acoustic_clearance_does_not_certify_phrase_closure")
    elif active_segment is not None:
        boundary_class = "asr_segment_occupancy_unknown"
        boundary_source = "active_asr_segment"
        # Words are the evidence used by the guard to decide whether a cut
        # intersects an utterance.  An ASR segment without an overlapping word
        # is still an open phrase-continuity question, but is not a known word
        # cut.  With no words at all, even timing clearance is unknown.
        status = "supported" if words and local_coverage_present and not unknown else "unknown"
        if words and local_coverage_present and not unknown:
            reasons.append("active_asr_segment_is_not_word_clearance_failure")
        else:
            reasons.append("asr_segment_without_word_timing")
        reasons.append("phrase_continuity_unknown_from_asr_segment_occupancy")
    elif direction_value == "outgoing" and any(
        int(item["end_sample"]) == cue for item in segments
    ):
        boundary_class = "observed_asr_segment_end"
        boundary_source = "asr_segment_endpoint"
        status = "supported" if words and local_coverage_present and not unknown else "unknown"
        reasons.append("observed_asr_segment_end_only")
        reasons.append("phrase_continuity_unknown_from_asr_segment_endpoint")
    elif closure_supported and not unknown:
        boundary_class = "true_phrase_closure_supported"
        boundary_source = str(closure_provider)
        status = "supported"
        reasons.append("explicit_phrase_closure_provider")
    elif words and local_coverage_present:
        boundary_class = "word_gap_only"
        boundary_source = "asr_word_gap_observation"
        status = "unknown" if unknown else "supported"
        reasons.append("word_gap_does_not_certify_phrase_closure")
    elif words:
        boundary_class = "word_timing_without_asr_coverage"
        boundary_source = "asr_word_timing_without_coverage"
        status = "unknown"
        reasons.append("missing_asr_coverage_is_unknown" if not asr_coverage_present
                       else "local_asr_coverage_missing")
    else:
        boundary_class = "true_phrase_closure_unknown"
        boundary_source = "no_phrase_provider"
        status = "unknown"
        reasons.append("true_phrase_closure_unknown")

    # A phrase group is useful provenance but is not promoted to closure.
    phrase_at_cue = next(
        (dict(item) for item in phrases if int(item["end_sample"]) == cue), None
    )
    if phrase_at_cue is not None:
        reasons.append("asr_phrase_group_is_word_gap_heuristic")
    pickup_uncertain = bool(
        direction_value == "incoming"
        and not active_utterance
        and (unknown or (words and cue < int(words[0]["start_sample"])))
    )
    if pickup_uncertain:
        reasons.append("incoming_pickup_span_or_alignment_unknown")
    if direction_value == "outgoing" and unknown:
        reasons.append("outgoing_clearance_intersects_unknown_coverage")

    clearance = _clearance(direction=direction_value, cue=cue, words=words, segments=segments, duration=duration)
    identity = _identity(payload, CANONICAL_SAMPLE_RATE, duration)
    # Mismatched canonical metadata must not be presented as a safe boundary.
    if identity["mismatches"]:
        status = "unknown"
        reasons.append("canonical_identity_metadata_mismatch")

    # ``status`` and ``timing_status`` describe the local word/timing guard.
    # Phrase continuity is deliberately independent: without a named closure
    # provider it remains unknown even when timing clearance is supported.
    word_clearance_status = status
    phrase_continuity_status = "supported" if closure_supported else "unknown"
    word_clearance = {
        "status": word_clearance_status,
        "supported": word_clearance_status == "supported",
        "unsafe": word_clearance_status == "unsafe",
        "active_word": active_word is not None,
        "active_asr_segment": active_segment is not None,
        "evidence": "observed_word_timing" if words else "no_word_timing_observation",
    }

    result: dict[str, Any] = {
        "schema_version": PHRASE_EVIDENCE_SCHEMA,
        "policy_version": PHRASE_EVIDENCE_VERSION,
        "canonical_sample_rate": CANONICAL_SAMPLE_RATE,
        "sample_rate": CANONICAL_SAMPLE_RATE,
        "duration_samples": duration,
        "cue_sample": cue,
        "direction": direction_value,
        "status": status,
        "timing_status": status,
        "word_clearance_status": word_clearance_status,
        "word_clearance": word_clearance,
        "asr_coverage_present": asr_coverage_present,
        "local_coverage_present": local_coverage_present,
        "phrase_continuity_status": phrase_continuity_status,
        "phrase_closure_status": phrase_continuity_status,
        "boundary_class": boundary_class,
        "boundary_source": boundary_source,
        "active_utterance": active_utterance,
        "active_utterance_status": "active" if active_utterance else (
            "unknown_segment_occupancy" if active_segment else "not_observed"
        ),
        "active_asr_segment": active_segment is not None,
        "acoustic_clearance": acoustic_silence,
        "clearance": clearance,
        "clearance_samples": clearance["samples"],
        "pickup_uncertainty": pickup_uncertain,
        "pickup_uncertainty_status": "unknown" if pickup_uncertain else "not_observed",
        "word_gap": boundary_class == "word_gap_only",
        "observed_asr_segment_end": boundary_class == "observed_asr_segment_end",
        "true_phrase_closure": "supported" if closure_supported else "unknown",
        "phrase_group_at_cue": phrase_at_cue,
        "punctuation_present": punctuation,
        "identity": identity,
        "canonical_identity": identity,
        "canonical_identity_valid": identity["valid"],
        "reasons": list(dict.fromkeys(reasons)),
        "evidence_semantics": {
            "timing_status": "word-level clearance only; decoder segment occupancy is not a word cut",
            "word_clearance": "supported only from observed word timing or verified acoustic clearance",
            "phrase_continuity": "unknown for ASR segment occupancy unless a named provider supports closure",
            "missing_asr_coverage": "unknown, even when isolated word records are present",
            "local_coverage": "only evidence covering this cue can support timing; segment ends are inclusive",
            "word_gap": "timing observation; not syntactic or musical closure",
            "observed_asr_segment_end": "provider endpoint; not phrase-completion proof",
            "true_phrase_closure": "unknown unless an explicit named provider supports it",
            "missing_asr": "unknown, never silence",
            "punctuation": "retained as text metadata; never a certified syntactic boundary",
        },
    }
    return result


__all__ = [
    "CANONICAL_SAMPLE_RATE",
    "PHRASE_EVIDENCE_SCHEMA",
    "PHRASE_EVIDENCE_VERSION",
    "analyze_phrase_evidence",
]
