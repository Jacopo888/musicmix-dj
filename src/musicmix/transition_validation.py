"""Validation of rendered transition events.

This validator is intentionally a technical audit.  It keeps measurable
defects (for example clipping or a source identity mismatch) separate from
descriptions (for example a level jump) and unavailable evidence (for example
ASR-only phrase information).  It never turns RMS, a song-level diagnostic,
or an ASR gap into a musical-quality score or certified phrase closure.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping

from .audio import AudioTrack, load_audio
from .mix_quality import evaluate_mix_quality


VALIDATION_VERSION = "musicmix.transition_validation.1"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _entry(container: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in container:
            return container[key]
    return None


def _issue(code: str, message: str, *, severity: str = "defect", evidence: Any = None) -> dict[str, Any]:
    result = {"code": code, "message": message, "severity": severity}
    if evidence is not None:
        result["evidence"] = evidence
    return result


def _actual_join(artifact: Mapping[str, Any], quality: Mapping[str, Any] | None) -> int | None:
    if isinstance(quality, Mapping):
        join = _mapping(quality.get("join"))
        value = join.get("sample")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return int(value)
    clocks = _mapping(artifact.get("clocks"))
    output = _mapping(clocks.get("output"))
    value = _entry(output, "join_sample", "anchor_sample")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return int(value)
    value = _entry(artifact, "join_sample", "output_join_sample")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return int(value)
    return None


def _artifact_join(artifact: Mapping[str, Any]) -> int | None:
    """Read only the renderer's recorded output clock (without quality input)."""

    clocks = _mapping(artifact.get("clocks"))
    output = _mapping(clocks.get("output"))
    value = _entry(output, "join_sample", "anchor_sample")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return int(value)
    value = _entry(artifact, "join_sample", "output_join_sample")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return int(value)
    return None


def _quality_from_artifact(artifact: Mapping[str, Any], quality: Any) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if isinstance(quality, Mapping):
        return dict(quality), []
    if isinstance(quality, AudioTrack):
        join = _actual_join(artifact, None)
        if join is None:
            return None, [_issue("quality_join_missing", "An AudioTrack was supplied without a rendered join anchor", severity="unknown")]
        try:
            return evaluate_mix_quality(quality, join), []
        except (TypeError, ValueError) as exc:
            return None, [_issue("quality_evaluation_failed", str(exc), severity="unknown")]
    path = _entry(artifact, "path", "output_path")
    join = _actual_join(artifact, None)
    if path is None or join is None:
        return None, [_issue("rendered_quality_missing", "No quality report or readable rendered artifact was supplied", severity="unknown")]
    try:
        track = load_audio(Path(path))
        return evaluate_mix_quality(track, join), []
    except (OSError, TypeError, ValueError) as exc:
        return None, [_issue("quality_evaluation_failed", str(exc), severity="unknown")]


def _classify_quality(quality: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split actual waveform flags into hard defects and descriptions."""

    hard: list[dict[str, Any]] = []
    descriptive: list[dict[str, Any]] = []
    for raw in quality.get("flags", ()) or ():
        flag = _mapping(raw)
        code = str(flag.get("code", "quality_flag"))
        message = str(flag.get("description", code))
        # Clipping and empty output are technical failures.  Level changes,
        # silence, DC and relative jump metrics are observations; they can be
        # intentional and never imply that music is bad.
        if code in {"clipping_detected", "empty_track", "nonfinite_output"} or flag.get("severity") == "technical_defect":
            hard.append(_issue(code, message, evidence=flag))
        else:
            descriptive.append(_issue(code, message, severity="descriptive", evidence=flag))
    metrics = _mapping(quality.get("metrics"))
    clipping = _mapping(metrics.get("clipping"))
    if clipping.get("clipped_sample_count", 0):
        if not any(item["code"] == "clipping_detected" for item in hard):
            hard.append(_issue("clipping_detected", "The actual rendered PCM contains clipped samples", evidence=clipping))
    return hard, descriptive


def _phrase_evidence(phrase: Any) -> dict[str, Any]:
    value = _mapping(phrase)
    # Accept the project expectation shape as well as a small direct shape.
    if "phrase_or_structural_arrival" in value and isinstance(value["phrase_or_structural_arrival"], Mapping):
        value = dict(value["phrase_or_structural_arrival"])
    elif "phrase_evidence" in value and isinstance(value["phrase_evidence"], Mapping):
        value = dict(value["phrase_evidence"])
    status = str(_entry(value, "status", "phrase_status", "closure_status") or "unknown").lower()
    provider = str(_entry(value, "provider", "source", "method") or "").lower()
    evidence = str(_entry(value, "evidence", "basis") or "").lower()
    asr_only = "asr" in provider or "asr" in evidence or status in {"asr_gap", "word_gap_only"}
    if asr_only:
        return {"status": "unknown", "certified": False, "reason": "ASR or word gaps do not certify musical phrase closure", "provider": provider or None}
    if value.get("certified") is True and (status == "supported" or value.get("supported") is True):
        return {"status": "supported", "certified": True, "reason": "explicit caller-supplied phrase/structure evidence", "provider": provider or None}
    if status == "supported" or value.get("supported") is True:
        return {"status": "supported", "certified": False, "reason": "phrase evidence is reported but not certified by this technical validator", "provider": provider or None}
    return {"status": "unknown", "certified": False, "reason": "phrase closure evidence is unavailable", "provider": provider or None}


def validate_transition_event(
    event: Mapping[str, Any] | None = None,
    artifact: Mapping[str, Any] | None = None,
    quality: Mapping[str, Any] | AudioTrack | None = None,
    vocal: Mapping[str, Any] | None = None,
    phrase: Mapping[str, Any] | None = None,
    *,
    vocals: Mapping[str, Any] | None = None,
    song_eval: Any = None,
) -> dict[str, Any]:
    """Validate one event against its actual rendered artifact.

    ``event`` contains expected source IDs/clocks and intent metadata;
    ``artifact`` is the renderer return document.  ``quality`` may be an
    existing :func:`evaluate_mix_quality` report, an ``AudioTrack`` to assess,
    or omitted when the artifact path is readable.  The result is JSON-safe
    and keeps hard defects, descriptions and unknown evidence in separate
    arrays.
    """

    event_doc = _mapping(event)
    artifact_doc = _mapping(artifact) or _mapping(event_doc.get("artifact"))
    if not artifact_doc and (event_doc.get("path") is not None or "clocks" in event_doc or "render_version" in event_doc):
        artifact_doc = event_doc
    hard: list[dict[str, Any]] = []
    descriptive: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []

    actual_quality, quality_notes = _quality_from_artifact(artifact_doc, quality)
    unknown.extend(quality_notes)
    if actual_quality is not None:
        quality_hard, quality_descriptive = _classify_quality(actual_quality)
        hard.extend(quality_hard)
        descriptive.extend(quality_descriptive)

    artifact_clock_join = _artifact_join(artifact_doc)
    quality_clock_join = _actual_join({}, actual_quality) if actual_quality is not None else None
    if artifact_clock_join is not None and quality_clock_join is not None and artifact_clock_join != quality_clock_join:
        hard.append(_issue("quality_anchor_mismatch", "Quality was evaluated at a different join than the renderer clock", evidence={"artifact": artifact_clock_join, "quality": quality_clock_join}))
    actual_join = quality_clock_join if quality_clock_join is not None else artifact_clock_join
    expected_join = _entry(event_doc, "output_join_sample", "join_sample", "expected_join_sample")
    if expected_join is not None and actual_join is not None and int(expected_join) != actual_join:
        hard.append(_issue("actual_anchor_mismatch", f"Expected output join {int(expected_join)} differs from rendered join {actual_join}", evidence={"expected": int(expected_join), "actual": actual_join}))
    elif actual_join is None:
        unknown.append(_issue("actual_anchor_unknown", "The rendered join anchor is unavailable", severity="unknown"))

    clocks = _mapping(artifact_doc.get("clocks"))
    source_clocks = _mapping(clocks.get("source"))
    expected_ids = {
        "a": _entry(event_doc, "track_a", "source_a_id"),
        "b": _entry(event_doc, "track_b", "source_b_id"),
    }
    event_sources = {"a": _mapping(event_doc.get("source_a")), "b": _mapping(event_doc.get("source_b"))}
    for side in ("a", "b"):
        source = _mapping(source_clocks.get(side))
        expected_id = expected_ids[side] or event_sources[side].get("track_id")
        actual_id = source.get("track_id")
        if expected_id is not None and actual_id is not None and str(expected_id) != str(actual_id):
            hard.append(_issue("source_identity_mismatch", f"Source {side} identity does not match the event", evidence={"expected": expected_id, "actual": actual_id}))
        elif expected_id is not None and actual_id is None:
            unknown.append(_issue("source_identity_unknown", f"Rendered source {side} identity is unavailable", severity="unknown"))
        expected_rate = event_sources[side].get("sample_rate")
        if expected_rate is None:
            expected_rate = _entry(event_doc, f"sample_rate_{side}", f"source_{side}_sample_rate")
        actual_rate = source.get("sample_rate", clocks.get("rate"))
        if expected_rate is not None and actual_rate is not None and int(expected_rate) != int(actual_rate):
            hard.append(_issue("source_rate_mismatch", f"Source {side} sample rate differs from the rendered clock", evidence={"expected": expected_rate, "actual": actual_rate}))

    artifact_rate = clocks.get("rate", artifact_doc.get("sample_rate"))
    event_rate = _entry(event_doc, "sample_rate", "output_sample_rate")
    if event_rate is not None and artifact_rate is not None and int(event_rate) != int(artifact_rate):
        hard.append(_issue("sample_rate_mismatch", "Event and rendered artifact use different sample rates", evidence={"expected": event_rate, "actual": artifact_rate}))

    # Compare the source cue coordinates recorded by the renderer against any
    # nominal event coordinates.  This catches a stale metadata record while
    # keeping all checks in source and output clocks explicit.
    expected_cues = {"a": _entry(event_doc, "cue_out_sample", "source_a_cue_sample"),
                     "b": _entry(event_doc, "cue_in_sample", "source_b_cue_sample")}
    for side in ("a", "b"):
        actual_cue = _mapping(source_clocks.get(side)).get("cue_sample")
        if expected_cues[side] is not None and actual_cue is not None and int(expected_cues[side]) != int(actual_cue):
            hard.append(_issue("source_anchor_mismatch", f"Source {side} cue differs from the rendered source anchor", evidence={"expected": expected_cues[side], "actual": actual_cue}))

    intent = str(_entry(event_doc, "intent") or artifact_doc.get("intent") or "handoff")
    if intent == "overlap":
        eligibility = _mapping(artifact_doc.get("eligibility"))
        vocal = _mapping(eligibility.get("vocal_safety"))
        vocal = _mapping(vocals if vocals is not None else vocal) if (vocals is not None or vocal is not None) else vocal
        status = str(vocal.get("status", "unknown"))
        if status in {"rejected", "occupied", "unsafe"}:
            hard.append(_issue("overlap_vocal_safety_rejected", "Equal-power overlap has occupied or unsafe vocal evidence", evidence=vocal))
        elif status != "supported":
            unknown.append(_issue("overlap_vocal_safety_unknown", "Overlap requires explicit acoustic-silence support", severity="unknown", evidence=vocal))
        tempo = _mapping(eligibility.get("tempo"))
        if tempo and str(tempo.get("status", "unknown")) not in {"acceptable", "supported"}:
            if str(tempo.get("status")) in {"mismatch", "invalid"}:
                hard.append(_issue("overlap_tempo_incompatible", "Overlap tempo evidence is outside the configured tolerance", evidence=tempo))
            else:
                unknown.append(_issue("overlap_tempo_unknown", "Overlap tempo evidence is unavailable", severity="unknown", evidence=tempo))

    phrase_input = phrase
    if phrase_input is None:
        phrase_input = _entry(event_doc, "phrase", "phrase_evidence", "phrase_or_structural_arrival", "phrase_closure")
    phrase_report = _phrase_evidence(phrase_input)
    if phrase_report["status"] == "unknown":
        unknown.append(_issue("phrase_closure_unknown", phrase_report["reason"], severity="unknown", evidence=phrase_report))
    elif not phrase_report["certified"]:
        descriptive.append(_issue("phrase_evidence_uncertified", phrase_report["reason"], severity="descriptive", evidence=phrase_report))

    diagnostics: dict[str, Any] = {}
    supplied_song_eval = song_eval if song_eval is not None else _entry(event_doc, "song_eval", "song_evaluation")
    if supplied_song_eval is not None:
        # Preserve it for audit, while making clear it did not influence the
        # technical result or award a musical score.
        diagnostics["song_eval"] = deepcopy(supplied_song_eval)
        diagnostics["song_eval_use"] = "diagnostic_only_not_used_as_reward_or_quality_score"

    result_status = "fail" if hard else ("unknown" if unknown else "pass")
    return {
        "schema_version": VALIDATION_VERSION,
        "status": result_status,
        "valid": not bool(hard),
        "certified": not bool(hard or unknown),
        "hard_defects": hard,
        "descriptive": descriptive,
        "unknown": unknown,
        "actual_join_sample": actual_join,
        "actual_quality": actual_quality,
        "phrase_closure": phrase_report,
        "musical_quality": None,
        "musical_quality_reason": "Technical validation does not estimate musical quality.",
        "diagnostics": diagnostics,
        "evidence_policy": {
            "quality_source": "actual rendered waveform and evaluated anchor",
            "level_change": "descriptive_only",
            "asr_phrase_gaps": "unknown_not_closure",
            "song_eval": "diagnostic_only",
        },
    }


def construct_counterfactual(
    event: Mapping[str, Any],
    *,
    order: str | None = None,
    shift_samples: int = 0,
    silence: bool = False,
    clipping: bool = False,
    label: str | None = None,
) -> dict[str, Any]:
    """Describe a controlled counterfactual construction.

    This helper labels known manipulations for an experiment.  It intentionally
    does not infer that changing order, shifting a cue, adding silence, or
    clipping makes a musical result better or worse.
    """

    if not isinstance(event, Mapping):
        raise TypeError("event must be a mapping")
    try:
        shift = int(shift_samples)
    except (TypeError, ValueError) as exc:
        raise ValueError("shift_samples must be an integer") from exc
    result = deepcopy(dict(event))
    properties: list[str] = []
    if order is not None:
        result["counterfactual_order"] = str(order)
        properties.append("order_changed")
    if shift:
        result["counterfactual_shift_samples"] = shift
        properties.append("anchor_shift_applied")
    if silence:
        result["counterfactual_silence"] = True
        properties.append("silence_inserted")
    if clipping:
        result["counterfactual_clipping"] = True
        properties.append("clipping_injected")
    result["counterfactual"] = {"known_properties": properties, "label": label,
                                 "musical_quality": None,
                                 "interpretation": "construction_labels_only_no_universal_bad_music_judgement"}
    return result


# Descriptive aliases make the helper easy to discover without creating a
# second implementation.
build_counterfactual = construct_counterfactual
make_counterfactual_event = construct_counterfactual


__all__ = ["VALIDATION_VERSION", "validate_transition_event", "construct_counterfactual",
           "build_counterfactual", "make_counterfactual_event"]
