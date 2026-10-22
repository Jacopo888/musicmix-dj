"""Typed joint ``when/where`` transition policy.

This is a small policy layer over :func:`musicmix.expectation.analyze_expectation`.
It does not train a model or turn heuristic margins into probabilities.  The
policy keeps functional support (motion, arrival, vocal clearance and
structure) separate from the optional utility/ranking context supplied by a
caller.  Missing evidence remains missing and can lead to ``defer``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .expectation import ExpectationAssessment, analyze_expectation


POLICY_VERSION = "joint-transition-policy-v2-timing-separated"
POLICY_SCHEMA = "musicmix.transition-intent.1"
POLICY_THRESHOLDS_VERSION = "joint-policy-hypothesis-v1"
ACTIONS = frozenset({"stay", "prepare", "execute", "defer"})
INTENTS = frozenset({"preserve_groove", "continue_motion", "arrive", "deliberate_contrast"})


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _json(value: Any) -> Any:
    """Convert dataclass diagnostics to JSON-safe primitives."""

    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    number = _finite(value)
    return number


def _level(profile: Mapping[str, Any] | None) -> str | None:
    """Read only an explicitly declared functional level.

    An isolated RMS scale has no genre-independent high/low meaning.  The
    pairwise fallback lives in :func:`_pair_levels`, where two contexts can be
    compared without treating a file identity or an absolute loudness as a
    learned label.
    """

    if not isinstance(profile, Mapping):
        return None
    for key in ("energy_level", "functional_level", "level", "intensity"):
        value = profile.get(key)
        if isinstance(value, str):
            value = value.strip().lower()
            if value in {"high", "low", "mid", "medium"}:
                return "mid" if value in {"mid", "medium"} else value
    return None


def _pair_levels(a: str | None, b: str | None, a_profile: Mapping[str, Any] | None,
                 b_profile: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    # Absolute RMS, log-RMS, gain, normalization, and genre do not provide a
    # validated psychological high/low scale.  Keep the functional level
    # unknown unless a caller/provider declares it explicitly.  The profile
    # parameters remain in the signature for source-compatible callers and to
    # make it clear that numeric trajectories are never used as this label.
    del a_profile, b_profile
    return a, b


def _support_descriptor(value: Any) -> tuple[bool, dict[str, Any]]:
    if not isinstance(value, Mapping):
        return False, {"present": value is not None, "provider": None}
    provider = value.get("provider") or value.get("source") or value.get("backend")
    supported = bool(
        (value.get("supported") is True or value.get("status") == "supported")
        and isinstance(provider, str)
        and bool(provider.strip())
    )
    return supported, {
        "present": True,
        "supported": supported,
        "provider": provider,
        "status": value.get("status"),
    }


def _structure_evidence(structure: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(structure, Mapping):
        return {
            "available": False, "supported": False, "arrival_supported": False,
            "event_supported": False, "stability_supported": False,
            "section_label_only": False, "details": {},
        }
    section_only = any(key in structure for key in ("section_label", "label", "section"))
    arrival = structure.get("arrival", structure.get("phrase_or_structural_arrival"))
    event = structure.get("event", structure.get("event_evidence"))
    stability = structure.get("stability", structure.get("stability_evidence"))
    arrival_supported, arrival_details = _support_descriptor(arrival)
    event_supported, event_details = _support_descriptor(event)
    stability_supported, stability_details = _support_descriptor(stability)
    # An arrival descriptor by itself is an opportunity provider, not proof of
    # arrival.  Actual event and stability evidence must come from trajectory
    # evidence or explicit separate providers.
    return {
        "available": True,
        "supported": arrival_supported,
        "arrival_supported": arrival_supported,
        "event_supported": event_supported,
        "stability_supported": stability_supported,
        "section_label_only": bool(section_only and not (arrival_supported or event_supported or stability_supported)),
        "details": {"arrival": arrival_details, "event": event_details, "stability": stability_details,
                     "section_label": structure.get("section_label", structure.get("label", structure.get("section")))},
    }


def _phrase_evidence(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"name": name, "available": False, "status": "unknown", "safe": False,
                "unsafe": False, "missing": [f"{name}_phrase_evidence"]}
    # ``status`` used to conflate decoder-segment occupancy with a word cut.
    # Prefer the explicit word/timing contract while retaining the old field
    # as a fallback for callers that provide compact evidence documents.
    word_clearance = value.get("word_clearance")
    if isinstance(word_clearance, Mapping):
        timing_status = str(word_clearance.get("status", value.get("timing_status", value.get("status", "unknown"))))
        word_unsafe = bool(word_clearance.get("unsafe") is True or timing_status == "unsafe")
        timing_safe = bool(word_clearance.get("supported") is True or timing_status == "supported")
    else:
        timing_status = str(value.get("timing_status", value.get("status", "unknown")))
        word_unsafe = bool(value.get("status") == "unsafe")
        timing_safe = timing_status == "supported"
    unsafe = bool(word_unsafe or value.get("active_utterance") is True
                  or value.get("boundary_class") == "active_utterance")
    status = str(value.get("status", timing_status))
    safe = timing_safe and not unsafe
    phrase_closure_status = str(value.get(
        "phrase_closure_status", value.get("phrase_continuity_status", value.get("true_phrase_closure", "unknown"))))
    if phrase_closure_status == "supported" and value.get("true_phrase_closure") == "unknown":
        phrase_closure_status = "unknown"
    return {"name": name, "available": True, "status": status, "safe": safe,
            "unsafe": unsafe, "boundary_class": value.get("boundary_class"),
            "timing_status": timing_status,
            "word_clearance_status": timing_status,
            "phrase_continuity_status": phrase_closure_status,
            "phrase_closure_status": phrase_closure_status,
            "phrase_closure_supported": phrase_closure_status == "supported",
            "active_asr_segment": bool(value.get("active_asr_segment", False)),
            "pickup_uncertainty": bool(value.get("pickup_uncertainty", False)),
            "reasons": list(value.get("reasons", ())) if isinstance(value.get("reasons", ()), Sequence) else []}


def _with_structural_provider(
    profile: Mapping[str, Any] | None,
    structure: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Pass a named structure provider to expectation without using labels."""

    if not isinstance(profile, Mapping):
        return profile
    evidence = _structure_evidence(structure)
    if not evidence["arrival_supported"]:
        return profile
    if any(key in profile for key in ("phrase_or_structural_arrival", "structural_arrival", "phrase_boundary")):
        return profile
    result = dict(profile)
    details = evidence["details"]["arrival"]
    result["phrase_or_structural_arrival"] = {
        "supported": True,
        "provider": details.get("provider"),
    }
    return result


@dataclass(frozen=True)
class TransitionAssessment:
    """Auditable result for one candidate transition."""

    action: str
    intent: str
    functional_support: tuple[Mapping[str, Any], ...]
    reasons: tuple[str, ...]
    missing: tuple[str, ...]
    coverage: Mapping[str, Any]
    utility: Mapping[str, Any]
    confidence: None = None
    baseline_score: float | None = None
    diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"action must be one of {sorted(ACTIONS)}")
        if self.intent not in INTENTS:
            raise ValueError(f"intent must be one of {sorted(INTENTS)}")
        if self.confidence is not None:
            raise ValueError("transition confidence is intentionally uncalibrated")
        if self.baseline_score is not None and not math.isfinite(float(self.baseline_score)):
            raise ValueError("baseline_score must be finite or null")
        object.__setattr__(self, "functional_support", tuple(dict(item) for item in self.functional_support))
        object.__setattr__(self, "reasons", tuple(dict.fromkeys(str(item) for item in self.reasons)))
        object.__setattr__(self, "missing", tuple(dict.fromkeys(str(item) for item in self.missing)))
        object.__setattr__(self, "coverage", dict(self.coverage))
        object.__setattr__(self, "utility", dict(self.utility))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics or {}))

    @property
    def abstained(self) -> bool:
        return self.action == "defer"

    @property
    def action_spec_name(self) -> str:
        return {
            "stay": "stay_in_A", "prepare": "prepare_transition",
            "execute": "execute_supported_event", "defer": "defer_to_later_opportunity",
        }[self.action]

    @property
    def functional_evidence(self) -> tuple[Mapping[str, Any], ...]:
        """Alias for consumers that call the support report evidence."""

        return self.functional_support

    @property
    def utility_score(self) -> float | None:
        value = self.utility.get("value")
        return _finite(value)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible policy document."""

        return _json({
            "schema_version": POLICY_SCHEMA,
            "policy_version": POLICY_VERSION,
            "thresholds_version": POLICY_THRESHOLDS_VERSION,
            "action": self.action,
            "action_spec_name": self.action_spec_name,
            "intent": self.intent,
            "functional_support": self.functional_support,
            "functional_evidence": self.functional_support,
            "reasons": self.reasons,
            "missing": self.missing,
            "coverage": self.coverage,
            "coverage_fraction": self.coverage.get("fraction"),
            "utility": self.utility,
            "confidence": None,
            "baseline_score": self.baseline_score,
            "diagnostics": self.diagnostics,
        })

    def to_json(self) -> dict[str, Any]:
        """Compatibility alias returning the JSON-ready document."""

        return self.to_dict()

    as_dict = to_dict


def assess_transition(
    A_before: Mapping[str, Any] | None,
    A_natural_after: Mapping[str, Any] | None,
    B_before: Mapping[str, Any] | None,
    B_after: Mapping[str, Any] | None,
    *,
    phrase_a: Mapping[str, Any] | None = None,
    phrase_b: Mapping[str, Any] | None = None,
    structure_a: Mapping[str, Any] | None = None,
    structure_b: Mapping[str, Any] | None = None,
    baseline_score: float | None = None,
) -> TransitionAssessment:
    """Assess whether to stay, prepare, execute, or defer a transition.

    The natural continuation is compared by observable trajectory direction;
    track names, hashes, and file identity never contribute to support.  A
    quiet intro or a section label can describe context but cannot satisfy the
    event/stability requirements for ``arrive``.
    """

    baseline = _finite(baseline_score)
    if baseline_score is not None and baseline is None:
        raise ValueError("baseline_score must be finite or null")
    expectation = analyze_expectation(
        A_before,
        A_natural_after,
        _with_structural_provider(B_after, structure_b),
    )
    a_phrase = _phrase_evidence(phrase_a, "A")
    b_phrase = _phrase_evidence(phrase_b, "B")
    struct_a = _structure_evidence(structure_a)
    struct_b = _structure_evidence(structure_b)
    profiles = expectation.diagnostics.get("profiles", {})
    a_summary = profiles.get("A_before", {}) if isinstance(profiles, Mapping) else {}
    b_summary = profiles.get("B_after", {}) if isinstance(profiles, Mapping) else {}
    a_profile = A_before
    b_profile = B_after
    a_level, b_level = _pair_levels(_level(A_before), _level(B_after), a_profile, b_profile)

    closure = expectation.closure_evidence
    continuation = expectation.continuation_evidence
    interruption = expectation.interruption_evidence
    functional_support: list[Mapping[str, Any]] = [
        {"kind": "expectation", "status": expectation.label,
         "score": expectation.score, "supported_evidence": expectation.diagnostics.get("supported_evidence", [])},
        {"kind": "global_motion", "status": "supported" if continuation and continuation.status == "supported" else "unknown",
         "label": expectation.label, "local_voice_closure_is_separate": True},
        {"kind": "phrase_A_local", **a_phrase},
        {"kind": "phrase_B_local", **b_phrase},
        {"kind": "structure_B", **struct_b},
    ]
    missing: list[str] = list(expectation.missing)
    for item in (a_phrase, b_phrase):
        missing.extend(item.get("missing", ()) if isinstance(item.get("missing"), Sequence) else ())
        if item.get("available") and item.get("phrase_closure_status") != "supported":
            missing.append(f"{item['name']}_phrase_closure_unknown")
    if not a_phrase["available"]:
        missing.append("phrase_A")
    if not b_phrase["available"]:
        missing.append("phrase_B")
    if not struct_b["available"]:
        missing.append("structure_B")
    if B_before is None:
        missing.append("B_before")
    if struct_b["section_label_only"]:
        missing.append("arrival_event_and_stability")

    switch_reasons: list[str] = []
    reasons: list[str] = []
    intent = "preserve_groove"
    action = "defer"
    if continuation and continuation.status == "supported":
        intent = "continue_motion"
        switch_reasons.append("multifeature_continuation_support")
    if closure and closure.status == "supported":
        intent = "arrive"
        switch_reasons.append("event_and_stability_arrival_support")
    # Deliberate contrast requires an explicit intent/provider declaration; a
    # direction reversal alone is an interruption, not a reason to cut.
    explicit_intent = None
    if isinstance(structure_b, Mapping):
        explicit_intent = structure_b.get("intent") or structure_b.get("transition_intent")
    contrast_supported = (
        str(explicit_intent).strip().lower() == "deliberate_contrast"
        and struct_b["supported"]
    )
    if contrast_supported:
        intent = "deliberate_contrast"
        switch_reasons.append("explicit_deliberate_contrast_provider")
    elif interruption and interruption.status == "supported":
        reasons.append("motion_interruption_without_explicit_contrast")

    unsafe_vocal = a_phrase["unsafe"] or b_phrase["unsafe"]
    # Unknown phrase continuity is reported as missing provenance but does not
    # block a continuation cut when the word-level timing guard is supported.
    # Missing/unknown timing clearance and pickup uncertainty still do.
    unknown_vocal = (not a_phrase["available"] or not b_phrase["available"]
                     or a_phrase.get("timing_status") == "unknown"
                     or b_phrase.get("timing_status") == "unknown"
                     or a_phrase.get("pickup_uncertainty") or b_phrase.get("pickup_uncertainty"))
    if unsafe_vocal:
        reasons.append("active_utterance_at_candidate_boundary")
    if unknown_vocal:
        missing.append("vocal_boundary_clearance")

    # A flat pair with no supported event is a legitimate stay decision.  Its
    # intent is preserve_groove even when the optional vocal document is absent.
    if (
        a_level in {"low", "high", "mid"} and b_level == a_level
        and expectation.label == "unknown"
        and not switch_reasons
    ):
        action = "stay"
        intent = "preserve_groove"
        reasons.append(f"flat_{a_level}_groove_without_switch_event")
    elif not A_before or not B_after:
        action = "defer"
        reasons.append("missing_global_motion_context")
    elif unsafe_vocal:
        action = "defer"
        reasons.append("vocal_boundary_unsafe")
    elif switch_reasons:
        # Local phrase evidence is a gate for execution, while a missing
        # phrase document remains unknown.  Functional support can therefore
        # produce prepare, but not an ungrounded cut.
        if a_phrase["safe"] and b_phrase["safe"]:
            action = "execute"
            reasons.extend(switch_reasons)
        elif unknown_vocal:
            action = "prepare"
            reasons.append("supported_switch_waits_for_vocal_clearance")
        else:
            action = "defer"
            reasons.append("vocal_clearance_not_supported")
    elif interruption and interruption.status == "supported":
        action = "stay"
        intent = "preserve_groove"
        reasons.append("unmotivated_cut")
    else:
        action = "defer"
        reasons.append("no_explicit_switch_reason")

    if intent == "arrive" and closure is not None and closure.status != "supported":
        # Defensive guard against a future evidence provider accidentally
        # turning section labels into arrival.
        intent = "continue_motion" if continuation and continuation.status == "supported" else "preserve_groove"
    if struct_b["section_label_only"] and intent == "arrive":
        intent = "preserve_groove"
        reasons.append("section_label_or_quiet_intro_is_not_arrival")

    available_slots = {
        "A_before": bool(A_before), "A_natural_after": bool(A_natural_after),
        "B_before": bool(B_before), "B_after": bool(B_after),
        # Expectation is a derived evidence slot only when both global
        # contexts exist.  The diagnostics object itself is always present,
        # including for a fully missing candidate.
        "expectation": bool(A_before) and bool(B_after),
        "phrase_A": a_phrase["available"], "phrase_B": b_phrase["available"],
        "structure_B": struct_b["available"],
    }
    available_count = sum(available_slots.values())
    denominator = len(available_slots)
    required_slots = ("A_before", "B_after")
    optional_slots = tuple(key for key in available_slots if key not in required_slots)
    coverage = {
        "available_fields": available_slots,
        "available_count": available_count,
        "required_count": len(required_slots),
        "evidence_slot_count": denominator,
        "fraction": available_count / denominator if denominator else 0.0,
        "denominator_basis": "fixed_source_and_derived_context_slots_v1",
        "required_slots": required_slots,
        "optional_slots": optional_slots,
        "abstained": action == "defer",
        "candidate_rating": None,
        "claim_10_of_10": False,
    }
    utility: dict[str, Any] = {
        "baseline_score": baseline,
        "expectation_margin": expectation.score,
        "value": None,
        "confidence": None,
        "semantics": "uncalibrated utility context; functional support is reported separately",
    }
    diagnostics = {
        "policy_version": POLICY_VERSION,
        "thresholds_version": POLICY_THRESHOLDS_VERSION,
        "expectation_version": expectation.diagnostics.get("expectation_version"),
        "expectation_label": expectation.label,
        "levels": {"A_before": a_level, "B_after": b_level},
        "B_before_available": B_before is not None,
        "structure_A": struct_a,
        "structure_B": struct_b,
        "closure_status": closure.status if closure is not None else "unknown",
        "global_motion_status": expectation.label,
        "local_voice_closure_is_not_global_motion": True,
        "natural_reference_identity_used": False,
        "identity_shortcut": "track/file identity is retained as provenance by callers but never a support feature",
        "section_labels_are_context_only": True,
        "missing_is_not_zero": True,
        "supported_switch_reasons": list(dict.fromkeys(switch_reasons)),
        "phrase_continuity": {
            "A": a_phrase.get("phrase_continuity_status", "unknown"),
            "B": b_phrase.get("phrase_continuity_status", "unknown"),
            "closure_is_required_for_arrive": True,
            "asr_segment_occupancy_is_not_closure": True,
        },
        "word_timing_guard": {
            "A": a_phrase.get("timing_status", "unknown"),
            "B": b_phrase.get("timing_status", "unknown"),
            "active_word_is_unsafe": True,
            "active_asr_segment_alone_is_not_unsafe": True,
        },
    }
    return TransitionAssessment(
        action=action,
        intent=intent,
        functional_support=tuple(functional_support),
        reasons=tuple(dict.fromkeys(reasons)),
        missing=tuple(dict.fromkeys(missing)),
        coverage=coverage,
        utility=utility,
        confidence=None,
        baseline_score=baseline,
        diagnostics=diagnostics,
    )


__all__ = [
    "ACTIONS", "INTENTS", "POLICY_SCHEMA", "POLICY_THRESHOLDS_VERSION",
    "POLICY_VERSION", "TransitionAssessment", "assess_transition",
]
