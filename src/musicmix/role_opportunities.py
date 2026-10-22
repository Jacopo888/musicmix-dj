"""Role-aware transition opportunity extraction and candidate event plans.

The previous adapters treated every provided cue as a generic pair endpoint.
This module keeps the directional roles explicit:

* an *outgoing* opportunity is an exit/completion point in deck A,
* an *incoming* opportunity is either an audible pickup or a later arrival in
  deck B.

Only provider timestamps are used as anchors.  A nearest beat/downbeat may be
recorded as context, but the anchor is never moved by more than
``PlanConfig.max_grid_snap_seconds`` (and in fact is never moved at all).  No
beat is invented when the real grid is missing.  ASR decoder segment endpoints
are not phrase labels; only word intervals are used for word protection and
optional pickup discovery.

The API is deliberately JSON-friendly so the root pipeline can feed it the six
catalog tracks (structure/vocal documents + optional CUE/LinkSeg opportunities)
after integration.  Synthetic contract fixtures in the tests are labelled
``synthetic``; they are not presented as measured catalog evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence


ROLE_OPPORTUNITY_VERSION = "role-opportunities-v1"
PLAN_SCHEMA = "musicmix.role-transition-plan.1"
OUTGOING = "outgoing"
INCOMING = "incoming"

_OUT_ROLES = {
    "out", "outgoing", "exit", "exit_point", "cue_out", "cue-out", "outro",
    "mix_out", "mix-out", "switch_out", "switch-out", "end", "out_point",
}
_IN_ROLES = {
    "in", "incoming", "entry", "entry_point", "cue_in", "cue-in", "intro",
    "mix_in", "mix-in", "switch_in", "switch-in", "start", "in_point",
    "pickup", "arrival",
}
_NEUTRAL_ROLES = {
    "boundary", "section", "linkseg", "neutral", "both", "unknown", "",
}


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite numeric")
    return number


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    number = _finite(value, label)
    if number != round(number):
        raise ValueError(f"{label} must be an integer sample")
    result = int(number)
    if result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


def _optional_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def _sample_from_record(record: Mapping[str, Any], sample_rate: int, label: str) -> int:
    """Read a canonical sample, or convert a provider seconds field once."""

    for key in ("sample", "sample_index", "canonical_sample", "start_sample"):
        if key in record and record[key] is not None:
            return _integer(record[key], f"{label}.{key}")
    for key in ("time_seconds", "seconds", "time", "start_seconds", "start"):
        if key in record and record[key] is not None:
            seconds = _finite(record[key], f"{label}.{key}")
            if seconds < 0:
                raise ValueError(f"{label}.{key} must be non-negative")
            return int(round(seconds * float(sample_rate)))
    raise ValueError(f"{label} needs sample or provider time_seconds")


def _record_seconds(record: Mapping[str, Any]) -> float | None:
    for key in ("time_seconds", "seconds", "time", "start_seconds", "start"):
        if key in record and record[key] is not None:
            try:
                return _finite(record[key], key)
            except ValueError:
                return None
    return None


def _role_from_record(record: Mapping[str, Any], source: str) -> tuple[str, str]:
    """Return (out/in/neutral, direction evidence)."""

    for key in ("role", "cue_role", "direction", "cue_type", "type", "point_type"):
        raw = record.get(key)
        if raw is None:
            continue
        value = str(raw).strip().lower()
        if value in _OUT_ROLES:
            return OUTGOING, "explicit_role"
        if value in _IN_ROLES:
            return INCOMING, "explicit_role"
        if value in _NEUTRAL_ROLES:
            continue
    source_value = str(source).lower()
    if any(token in source_value for token in ("linkseg", "segment", "section", "boundary")):
        return "neutral", "source_is_neutral_boundary"
    return "neutral", "role_unspecified"


def _dedupe_events(events: Iterable[tuple[int, Mapping[str, Any]]]) -> list[tuple[int, Mapping[str, Any]]]:
    by_sample: dict[int, Mapping[str, Any]] = {}
    for sample, record in events:
        old = by_sample.get(sample)
        if old is None:
            by_sample[sample] = record
            continue
        old_conf = _optional_confidence(old.get("confidence")) or 0.0
        new_conf = _optional_confidence(record.get("confidence")) or 0.0
        if new_conf > old_conf:
            by_sample[sample] = record
    return sorted(by_sample.items(), key=lambda item: item[0])


def _as_records(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _grid(
    structure: Mapping[str, Any] | None,
    key: str,
    sample_rate: int,
    duration_samples: int,
) -> tuple[list[tuple[int, Mapping[str, Any]]], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    if not structure:
        return [], warnings
    events: list[tuple[int, Mapping[str, Any]]] = []
    for index, record in enumerate(_as_records(structure.get(key))):
        try:
            sample = _sample_from_record(record, sample_rate, f"{key}[{index}]")
        except ValueError as exc:
            warnings.append({"reason": "invalid_grid_event", "field": key, "index": index, "detail": str(exc)})
            continue
        if sample >= duration_samples:
            warnings.append({"reason": "grid_event_outside_duration", "field": key, "sample": sample})
            continue
        events.append((sample, record))
    return _dedupe_events(events), warnings


def _normalise_words(
    vocals: Mapping[str, Any] | None,
    sample_rate: int,
    duration_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    if not vocals:
        return words, warnings
    for index, record in enumerate(_as_records(vocals.get("words"))):
        try:
            start = _sample_from_record(record, sample_rate, f"words[{index}].start")
            end = _end_sample_from_record(record, sample_rate, f"words[{index}].end")
        except (ValueError, KeyError) as exc:
            warnings.append({"reason": "invalid_word", "index": index, "detail": str(exc)})
            continue
        if not 0 <= start < end <= duration_samples:
            warnings.append({"reason": "word_interval_outside_duration", "index": index, "start": start, "end": end})
            continue
        words.append({
            "start_sample": start,
            "end_sample": end,
            "text": str(record.get("text", "")).strip(),
            "confidence": _optional_confidence(record.get("confidence")),
            "provider_time_seconds": _record_seconds(record),
        })
    words.sort(key=lambda item: (item["start_sample"], item["end_sample"]))
    return words, warnings


def _end_sample_from_record(record: Mapping[str, Any], sample_rate: int, label: str) -> int:
    for key in ("end_sample", "end_sample_index", "end_canonical_sample"):
        if key in record and record[key] is not None:
            return _integer(record[key], f"{label}.{key}")
    for key in ("end_seconds", "end_time", "end"):
        if key in record and record[key] is not None:
            seconds = _finite(record[key], f"{label}.{key}")
            if seconds < 0:
                raise ValueError(f"{label}.{key} must be non-negative")
            return int(round(seconds * float(sample_rate)))
    raise ValueError(f"{label} needs end_sample or provider end_seconds")


def _coverage_unknown_at(
    vocals: Mapping[str, Any] | None,
    sample: int,
    sample_rate: int,
) -> bool:
    if not vocals:
        return False
    for key in ("coverage_intervals", "coverage", "uncertain_regions"):
        for record in _as_records(vocals.get(key)):
            try:
                start = _sample_from_record(record, sample_rate, f"{key}.start")
                end = _end_sample_from_record(record, sample_rate, f"{key}.end")
            except (ValueError, KeyError):
                continue
            if start <= sample < end and str(record.get("status", "")).startswith("unknown"):
                return True
    return False


def _word_context(
    words: Sequence[Mapping[str, Any]],
    *,
    anchor: int,
    role: str,
    guard_samples: int,
) -> dict[str, Any]:
    """Separate word overlap (unsafe) from timing clearance (not closure)."""

    previous = next((item for item in reversed(words) if int(item["end_sample"]) <= anchor), None)
    following = next((item for item in words if int(item["start_sample"]) >= anchor), None)
    spanning = next(
        (item for item in words if int(item["start_sample"]) < anchor < int(item["end_sample"])),
        None,
    )
    pickup_at_start = next(
        (item for item in words if int(item["start_sample"]) == anchor and anchor < int(item["end_sample"])),
        None,
    )
    result: dict[str, Any] = {
        "status": "unknown",
        "active_word": bool(spanning),
        "previous_word_end_sample": int(previous["end_sample"]) if previous else None,
        "next_word_start_sample": int(following["start_sample"]) if following else None,
        "pickup_at_word_start": bool(pickup_at_start),
        "clearance_samples": None,
        "guard_samples": int(guard_samples),
        "evidence": "word_interval_timing" if words else "no_word_intervals",
    }
    if spanning is not None:
        result["status"] = "unsafe"
        result["reason"] = "anchor_inside_known_word"
        return result
    if role == OUTGOING:
        if previous is not None:
            clearance = anchor - int(previous["end_sample"])
            result["clearance_samples"] = clearance
            if clearance < guard_samples:
                result["status"] = "risk"
                result["reason"] = "word_tail_inside_guard_but_not_cut"
            else:
                result["status"] = "supported"
                result["reason"] = "previous_word_ended_before_exit"
        elif not words:
            result["status"] = "unknown"
            result["reason"] = "missing_asr_is_unknown"
        else:
            result["status"] = "supported"
            result["reason"] = "no_known_word_before_exit"
    else:
        if pickup_at_start is not None:
            result["status"] = "supported"
            result["reason"] = "enters_at_word_onset_not_inside_word"
        elif following is not None:
            clearance = int(following["start_sample"]) - anchor
            result["clearance_samples"] = clearance
            if clearance < guard_samples:
                result["status"] = "risk"
                result["reason"] = "word_onset_inside_guard"
            else:
                result["status"] = "supported"
                result["reason"] = "entry_before_next_known_word"
        elif not words:
            result["status"] = "unknown"
            result["reason"] = "missing_asr_is_unknown"
        else:
            result["status"] = "supported"
            result["reason"] = "no_known_word_after_entry"
    return result


def _nearest_grid(
    grid: Sequence[tuple[int, Mapping[str, Any]]],
    anchor: int,
    max_snap_samples: int,
) -> tuple[int | None, int | None, str]:
    if not grid:
        return None, None, "unavailable"
    nearest = min(grid, key=lambda item: (abs(item[0] - anchor), item[0]))[0]
    delta = anchor - nearest
    if abs(delta) <= max_snap_samples:
        return nearest, delta, "within_80ms_context_only_anchor_not_moved"
    return nearest, delta, "off_grid_anchor_preserved"


def _opportunity_id(track_id: str, role: str, anchor: int, source: str) -> str:
    payload = f"{ROLE_OPPORTUNITY_VERSION}|{track_id}|{role}|{anchor}|{source}".encode("utf-8")
    return "opp_" + hashlib.sha256(payload).hexdigest()[:20]


def _plan_id(track_a: str, track_b: str, out_id: str, in_id: str) -> str:
    payload = f"{ROLE_OPPORTUNITY_VERSION}|{track_a}|{track_b}|{out_id}|{in_id}".encode("utf-8")
    return "plan_" + hashlib.sha256(payload).hexdigest()[:20]


@dataclass(frozen=True)
class PlanConfig:
    max_grid_snap_seconds: float = 0.080
    prepare_beats: float = 4.0
    default_prepare_seconds: float = 8.0
    word_guard_seconds: float = 0.080
    pickup_lookback_seconds: float = 2.0
    min_region_spacing_seconds: float = 8.0
    max_per_track_role: int = 2
    max_plans: int = 24
    max_plans_per_pair: int = 2
    max_arrival_after_change_seconds: float = 8.0
    max_pickup_before_change_seconds: float = 8.0
    max_pickup_to_arrival_seconds: float = 8.0
    include_beat_anchors_when_downbeats: bool = False
    include_neutral_boundaries: bool = True
    allow_unknown_words: bool = True
    require_safe_words: bool = False

    def __post_init__(self) -> None:
        if self.max_grid_snap_seconds < 0 or self.max_grid_snap_seconds > 0.080:
            raise ValueError("max_grid_snap_seconds must be within [0, 0.080]")
        if self.prepare_beats <= 0 or self.default_prepare_seconds < 0:
            raise ValueError("prepare settings must be non-negative/positive")
        if self.word_guard_seconds < 0 or self.pickup_lookback_seconds < 0:
            raise ValueError("word/pickup settings must be non-negative")
        if self.min_region_spacing_seconds < 0:
            raise ValueError("min_region_spacing_seconds must be non-negative")
        if self.max_per_track_role <= 0 or self.max_plans <= 0 or self.max_plans_per_pair <= 0:
            raise ValueError("selection limits must be positive")
        if (self.max_arrival_after_change_seconds < 0
                or self.max_pickup_before_change_seconds < 0
                or self.max_pickup_to_arrival_seconds < 0):
            raise ValueError("transition timing limits must be non-negative")


@dataclass(frozen=True)
class RoleOpportunity:
    opportunity_id: str
    track_id: str
    role: str
    anchor_sample: int
    sample_rate: int
    duration_samples: int
    source: str
    source_role: str
    direction_evidence: str
    evidence_class: str
    provider_time_seconds: float | None
    nearest_grid_sample: int | None
    grid_delta_samples: int | None
    grid_alignment: str
    completion_supported: bool
    phrase_closure_supported: bool
    word_status: str
    word_evidence: Mapping[str, Any]
    pickup_start_sample: int | None
    arrival_sample: int | None
    arrival_source: str | None
    arrival_delay_samples: int | None
    passes_constraints: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "opportunity_id": self.opportunity_id,
            "track_id": self.track_id,
            "role": self.role,
            "anchor_sample": self.anchor_sample,
            "sample_rate": self.sample_rate,
            "duration_samples": self.duration_samples,
            "source": self.source,
            "source_role": self.source_role,
            "direction_evidence": self.direction_evidence,
            "evidence_class": self.evidence_class,
            "provider_time_seconds": self.provider_time_seconds,
            "nearest_grid_sample": self.nearest_grid_sample,
            "grid_delta_samples": self.grid_delta_samples,
            "grid_alignment": self.grid_alignment,
            "completion_supported": self.completion_supported,
            "phrase_closure_supported": self.phrase_closure_supported,
            "word_status": self.word_status,
            "word_evidence": dict(self.word_evidence),
            "pickup_start_sample": self.pickup_start_sample,
            "arrival_sample": self.arrival_sample,
            "arrival_source": self.arrival_source,
            "arrival_delay_samples": self.arrival_delay_samples,
            "passes_constraints": self.passes_constraints,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class PlanEvent:
    action: str
    sample: int
    timeline: str
    from_sample: int | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "sample": self.sample,
            "timeline": self.timeline,
            "from_sample": self.from_sample,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CandidatePlan:
    plan_id: str
    track_a: str
    track_b: str
    outgoing_opportunity_id: str
    incoming_opportunity_id: str
    cue_out_sample: int
    pickup_start_sample: int
    arrival_sample: int
    change_sample: int
    prepare_sample: int
    entry_window_start_sample: int
    entry_window_end_sample: int
    arrival_delay_samples: int
    lookahead_arrival_after_change_seconds: float | None
    overlap_samples: int
    sample_rate_a: int
    sample_rate_b: int
    events: tuple[PlanEvent, ...]
    outgoing_completion_supported: bool
    outgoing_completion_source: str | None
    incoming_role: str
    incoming_arrival_source: str | None
    incoming_pickup_source: str | None
    word_status: Mapping[str, str]
    words_protected: bool
    evidence: Mapping[str, Any]
    passes_constraints: bool
    status: str
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "track_a": self.track_a,
            "track_b": self.track_b,
            "outgoing_opportunity_id": self.outgoing_opportunity_id,
            "incoming_opportunity_id": self.incoming_opportunity_id,
            "cue_out_sample": self.cue_out_sample,
            "pickup_start_sample": self.pickup_start_sample,
            "arrival_sample": self.arrival_sample,
            "change_sample": self.change_sample,
            "prepare_sample": self.prepare_sample,
            "entry_window": {
                "start_sample": self.entry_window_start_sample,
                "end_sample": self.entry_window_end_sample,
                "arrival_sample": self.arrival_sample,
                "pickup_start_sample": self.pickup_start_sample,
                "arrival_delay_samples": self.arrival_delay_samples,
                "incoming_role": self.incoming_role,
            },
            "arrival_delay_samples": self.arrival_delay_samples,
            "lookahead_arrival_after_change_seconds": self.lookahead_arrival_after_change_seconds,
            "overlap_samples": self.overlap_samples,
            "sample_rate_a": self.sample_rate_a,
            "sample_rate_b": self.sample_rate_b,
            "events": [event.to_dict() for event in self.events],
            "outgoing_completion_supported": self.outgoing_completion_supported,
            "outgoing_completion_source": self.outgoing_completion_source,
            "incoming_role": self.incoming_role,
            "incoming_arrival_source": self.incoming_arrival_source,
            "incoming_pickup_source": self.incoming_pickup_source,
            "word_status": dict(self.word_status),
            "words_protected": self.words_protected,
            "evidence": dict(self.evidence),
            "passes_constraints": self.passes_constraints,
            "status": self.status,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class TrackOpportunities:
    track_id: str
    sample_rate: int
    duration_samples: int
    opportunities: tuple[RoleOpportunity, ...]
    beats: tuple[int, ...]
    downbeats: tuple[int, ...]
    words: tuple[Mapping[str, Any], ...]
    warnings: tuple[Mapping[str, Any], ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RolePlanResult:
    schema_version: str
    opportunities: tuple[RoleOpportunity, ...]
    selected_plans: tuple[CandidatePlan, ...]
    proposals: tuple[CandidatePlan, ...]
    rejected: tuple[Mapping[str, Any], ...]
    diagnostics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "opportunities": [item.to_dict() for item in self.opportunities],
            "selected_plans": [item.to_dict() for item in self.selected_plans],
            "proposals": [item.to_dict() for item in self.proposals],
            "rejected": [dict(item) for item in self.rejected],
            "diagnostics": dict(self.diagnostics),
        }


def _provided_records(track: Mapping[str, Any]) -> list[tuple[Mapping[str, Any], str]]:
    records: list[tuple[Mapping[str, Any], str]] = []
    for key in (
        "provided_opportunities", "provided_cues", "opportunities", "cue_points",
        "linkseg_boundaries", "switch_points",
    ):
        for record in _as_records(track.get(key)):
            source = str(record.get("provider") or record.get("source") or key)
            records.append((record, source))
    return records


def _completion_records(track: Mapping[str, Any]) -> list[tuple[Mapping[str, Any], str]]:
    records: list[tuple[Mapping[str, Any], str]] = []
    for key in ("vocal_completions", "completion_events", "vocal_completion_events"):
        for record in _as_records(track.get(key)):
            source = str(record.get("provider") or record.get("source") or key)
            records.append((record, source))
    return records


def _word_pickup(
    words: Sequence[Mapping[str, Any]],
    arrival: int,
    lookback_samples: int,
) -> tuple[int, str] | None:
    """Find a real word onset shortly before arrival, never a decoder endpoint."""

    candidates = [
        int(word["start_sample"]) for word in words
        if arrival - lookback_samples <= int(word["start_sample"]) <= arrival
    ]
    if not candidates:
        return None
    start = max(candidates)
    return start, "asr_word_onset_lookahead"


def _make_opportunity(
    *,
    track_id: str,
    role: str,
    anchor: int,
    sample_rate: int,
    duration_samples: int,
    source: str,
    source_role: str,
    direction_evidence: str,
    evidence_class: str,
    provider_time_seconds: float | None,
    grid: Sequence[tuple[int, Mapping[str, Any]]],
    max_snap_samples: int,
    words: Sequence[Mapping[str, Any]],
    vocals: Mapping[str, Any] | None,
    guard_samples: int,
    pickup_lookback_samples: int,
    completion_supported: bool = False,
    phrase_closure_supported: bool = False,
    explicit_pickup: int | None = None,
    completion_source: str | None = None,
) -> RoleOpportunity:
    nearest, delta, alignment = _nearest_grid(grid, anchor, max_snap_samples)
    reasons: list[str] = []
    if not 0 <= anchor < duration_samples:
        raise ValueError("opportunity anchor outside canonical duration")
    pickup_start: int | None = None
    arrival_sample: int | None = None
    arrival_source: str | None = None
    arrival_delay: int | None = None
    if role == OUTGOING:
        if evidence_class == "downbeat":
            reasons.append("downbeat_is_exit_anchor_not_completion_proof")
        if evidence_class == "beat":
            reasons.append("beat_is_exit_anchor_not_completion_proof")
        if not completion_supported:
            reasons.append("outgoing_completion_not_supported_by_named_event")
    else:
        arrival_sample = anchor
        arrival_source = completion_source or source
        pickup_sample = explicit_pickup
        pickup_source = "explicit_pickup_field" if pickup_sample is not None else None
        if pickup_sample is None:
            found = _word_pickup(words, anchor, pickup_lookback_samples)
            if found is not None:
                pickup_sample, pickup_source = found
        if pickup_sample is None:
            pickup_sample = anchor
            pickup_source = "arrival_without_observed_pickup"
        if not 0 <= pickup_sample <= anchor:
            reasons.append("pickup_after_arrival_ignored")
            pickup_sample = anchor
            pickup_source = "arrival_without_valid_pickup"
        pickup_start = int(pickup_sample)
        arrival_delay = int(anchor - pickup_start)
        if pickup_start < anchor:
            reasons.append("pickup_precedes_arrival_by_lookahead")
        else:
            reasons.append("arrival_only_no_observed_pickup")
        if not any(str(item.get("text", "")).strip() for item in words):
            reasons.append("pickup_has_no_word_evidence")
    word_evidence = _word_context(words, anchor=anchor, role=role, guard_samples=guard_samples)
    if role == INCOMING and pickup_start is not None and pickup_start != anchor:
        entry_evidence = _word_context(words, anchor=pickup_start, role=role, guard_samples=guard_samples)
        word_evidence = dict(entry_evidence)
        word_evidence["arrival_word_status"] = _word_context(
            words, anchor=anchor, role=role, guard_samples=guard_samples
        )["status"]
    if _coverage_unknown_at(vocals, anchor, sample_rate):
        if word_evidence["status"] == "supported":
            word_evidence = dict(word_evidence)
            word_evidence["status"] = "unknown"
            word_evidence["reason"] = "canonical_coverage_unknown_at_anchor"
    passes = True
    if alignment.startswith("off_grid") and delta is not None and abs(delta) > max_snap_samples:
        reasons.append("anchor_is_off_grid_preserved_exact_timestamp")
    if word_evidence["status"] == "unsafe":
        passes = False
        reasons.append("word_overlap_unsafe")
    return RoleOpportunity(
        opportunity_id=_opportunity_id(track_id, role, anchor, f"{source}:{source_role}"),
        track_id=track_id,
        role=role,
        anchor_sample=anchor,
        sample_rate=sample_rate,
        duration_samples=duration_samples,
        source=source,
        source_role=source_role,
        direction_evidence=direction_evidence,
        evidence_class=evidence_class,
        provider_time_seconds=provider_time_seconds,
        nearest_grid_sample=nearest,
        grid_delta_samples=delta,
        grid_alignment=alignment,
        completion_supported=bool(completion_supported),
        phrase_closure_supported=bool(phrase_closure_supported),
        word_status=str(word_evidence["status"]),
        word_evidence=word_evidence,
        pickup_start_sample=pickup_start,
        arrival_sample=arrival_sample,
        arrival_source=arrival_source if role == INCOMING else None,
        arrival_delay_samples=arrival_delay,
        passes_constraints=passes,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _track_identity(track: Mapping[str, Any], structure: Mapping[str, Any] | None) -> tuple[str, int, int]:
    track_id = track.get("track_id")
    if not track_id or not str(track_id).strip():
        raise ValueError("track_id must be non-empty")
    track_id = str(track_id)
    declared_rate = track.get("sample_rate", structure.get("sample_rate") if structure else None)
    if declared_rate is None:
        raise ValueError(f"track {track_id} needs sample_rate")
    sample_rate = _integer(declared_rate, f"{track_id}.sample_rate", minimum=1)
    declared_duration = track.get("duration_samples", structure.get("duration_samples") if structure else None)
    if declared_duration is None:
        raise ValueError(f"track {track_id} needs duration_samples")
    duration = _integer(declared_duration, f"{track_id}.duration_samples", minimum=1)
    if structure:
        for field_name, expected in (("sample_rate", sample_rate), ("duration_samples", duration)):
            if structure.get(field_name) is not None:
                actual = _integer(structure[field_name], f"{track_id}.structure.{field_name}", minimum=1)
                if actual != expected:
                    raise ValueError(
                        f"track {track_id} structure {field_name} does not match canonical track"
                    )
    return track_id, sample_rate, duration


def extract_track_opportunities(
    track: Mapping[str, Any],
    *,
    config: PlanConfig | None = None,
) -> TrackOpportunities:
    """Extract role-specific opportunities for one canonical track.

    The input is a mapping with ``track_id``, ``sample_rate``,
    ``duration_samples``, optional ``structure`` (beats/downbeats), optional
    ``vocals``/``vocal_boundaries`` (words and coverage), and optional
    ``provided_opportunities``/``provided_cues`` plus completion events.
    """

    cfg = config or PlanConfig()
    structure = track.get("structure") if isinstance(track.get("structure"), Mapping) else None
    track_id, sample_rate, duration = _track_identity(track, structure)
    max_snap = int(round(cfg.max_grid_snap_seconds * sample_rate))
    guard = int(round(cfg.word_guard_seconds * sample_rate))
    lookback = int(round(cfg.pickup_lookback_seconds * sample_rate))
    beats, beam_warnings = _grid(structure, "beats", sample_rate, duration)
    downbeats, down_warnings = _grid(structure, "downbeats", sample_rate, duration)
    vocals = track.get("vocals") if isinstance(track.get("vocals"), Mapping) else None
    if vocals is None and isinstance(track.get("vocal_boundaries"), Mapping):
        vocals = track["vocal_boundaries"]
    words, word_warnings = _normalise_words(vocals, sample_rate, duration)
    opportunities: list[RoleOpportunity] = []
    warnings: list[dict[str, Any]] = beam_warnings + down_warnings + word_warnings
    by_role_anchor: dict[tuple[str, int], int] = {}

    def _priority(item: RoleOpportunity) -> tuple[int, int, int]:
        evidence_rank = {
            "vocal_completion": 3,
            "provided_role": 2,
            "neutral_boundary": 1,
            "downbeat": 0,
            "beat": 0,
        }.get(item.evidence_class, 0)
        explicit_rank = 1 if item.source_role in (OUTGOING, INCOMING) else 0
        return (evidence_rank, 1 if item.completion_supported else 0, explicit_rank)

    def add(item: RoleOpportunity) -> None:
        key = (item.role, item.anchor_sample)
        existing_index = by_role_anchor.get(key)
        if existing_index is None:
            by_role_anchor[key] = len(opportunities)
            opportunities.append(item)
            return
        existing = opportunities[existing_index]
        if _priority(item) > _priority(existing):
            opportunities[existing_index] = item

    provided = _provided_records(track)
    for record, source in provided:
        try:
            anchor = _sample_from_record(record, sample_rate, f"provided:{source}")
        except ValueError as exc:
            warnings.append({"reason": "invalid_provided_opportunity", "source": source, "detail": str(exc)})
            continue
        if not 0 <= anchor < duration:
            warnings.append({"reason": "provided_opportunity_outside_duration", "source": source, "sample": anchor})
            continue
        source_role, direction_evidence = _role_from_record(record, source)
        provider_seconds = _record_seconds(record)
        completion = bool(record.get("completion_supported") is True or record.get("completion") is True)
        closure = bool(record.get("phrase_closure_supported") is True or record.get("phrase_closure") is True)
        explicit_pickup = None
        for key in ("pickup_sample", "pickup_start_sample"):
            if record.get(key) is not None:
                try:
                    explicit_pickup = _integer(record[key], f"provided.{key}")
                except ValueError:
                    explicit_pickup = None
                break
        if explicit_pickup is None:
            for key in ("pickup_seconds", "pickup_time_seconds"):
                if record.get(key) is not None:
                    try:
                        explicit_pickup = int(round(_finite(record[key], key) * sample_rate))
                    except ValueError:
                        explicit_pickup = None
                    break
        if source_role == OUTGOING or (source_role == "neutral" and cfg.include_neutral_boundaries):
            add(_make_opportunity(
                track_id=track_id, role=OUTGOING, anchor=anchor, sample_rate=sample_rate,
                duration_samples=duration, source=source, source_role=source_role,
                direction_evidence=direction_evidence,
                evidence_class="provided_role" if source_role == OUTGOING else "neutral_boundary",
                provider_time_seconds=provider_seconds, grid=downbeats or beats,
                max_snap_samples=max_snap, words=words, vocals=vocals, guard_samples=guard,
                pickup_lookback_samples=lookback, completion_supported=completion,
                phrase_closure_supported=closure, explicit_pickup=explicit_pickup,
                completion_source=source,
            ))
        if source_role == INCOMING or (source_role == "neutral" and cfg.include_neutral_boundaries):
            add(_make_opportunity(
                track_id=track_id, role=INCOMING, anchor=anchor, sample_rate=sample_rate,
                duration_samples=duration, source=source, source_role=source_role,
                direction_evidence=direction_evidence,
                evidence_class="provided_role" if source_role == INCOMING else "neutral_boundary",
                provider_time_seconds=provider_seconds, grid=downbeats or beats,
                max_snap_samples=max_snap, words=words, vocals=vocals, guard_samples=guard,
                pickup_lookback_samples=lookback, completion_supported=completion,
                phrase_closure_supported=closure, explicit_pickup=explicit_pickup,
                completion_source=source,
            ))

    for record, source in _completion_records(track):
        supported = record.get("supported") is True or record.get("status") == "supported"
        provider = record.get("provider") or record.get("source") or source
        if not supported or not str(provider).strip():
            warnings.append({"reason": "vocal_completion_without_named_supported_provider", "source": source})
            continue
        try:
            anchor = _sample_from_record(record, sample_rate, f"completion:{source}")
        except ValueError as exc:
            warnings.append({"reason": "invalid_vocal_completion", "source": source, "detail": str(exc)})
            continue
        if not 0 <= anchor < duration:
            warnings.append({"reason": "vocal_completion_outside_duration", "source": source})
            continue
        source_role, direction_evidence = _role_from_record(record, source)
        if source_role == INCOMING:
            add(_make_opportunity(
                track_id=track_id, role=INCOMING, anchor=anchor, sample_rate=sample_rate,
                duration_samples=duration, source=str(provider), source_role=INCOMING,
                direction_evidence=direction_evidence, evidence_class="vocal_completion",
                provider_time_seconds=_record_seconds(record), grid=downbeats or beats,
                max_snap_samples=max_snap, words=words, vocals=vocals, guard_samples=guard,
                pickup_lookback_samples=lookback,
                completion_supported=True, phrase_closure_supported=False,
                completion_source=str(provider),
            ))
        else:
            add(_make_opportunity(
                track_id=track_id, role=OUTGOING, anchor=anchor, sample_rate=sample_rate,
                duration_samples=duration, source=str(provider), source_role=OUTGOING,
                direction_evidence=direction_evidence, evidence_class="vocal_completion",
                provider_time_seconds=_record_seconds(record), grid=downbeats or beats,
                max_snap_samples=max_snap, words=words, vocals=vocals, guard_samples=guard,
                pickup_lookback_samples=lookback,
                completion_supported=True, phrase_closure_supported=False,
                completion_source=str(provider),
            ))

    grid_roles: list[tuple[str, list[tuple[int, Mapping[str, Any]]], str]] = []
    if downbeats:
        grid_roles.append(("downbeat", downbeats))
        if cfg.include_beat_anchors_when_downbeats:
            grid_roles.append(("beat", beats))
    elif beats and cfg.include_beat_anchors_when_downbeats:
        grid_roles.append(("beat", beats))
    for evidence_class, grid in grid_roles:
        for anchor, record in grid:
            for role in (OUTGOING, INCOMING):
                add(_make_opportunity(
                    track_id=track_id, role=role, anchor=anchor, sample_rate=sample_rate,
                    duration_samples=duration, source=f"{evidence_class}_grid",
                    source_role="neutral", direction_evidence="grid_is_context_not_direction",
                    evidence_class=evidence_class, provider_time_seconds=_record_seconds(record),
                    grid=grid, max_snap_samples=max_snap, words=words, vocals=vocals,
                    guard_samples=guard, pickup_lookback_samples=lookback,
                    completion_supported=False, phrase_closure_supported=False,
                ))

    opportunities.sort(key=lambda item: (item.role, item.anchor_sample, item.source, item.opportunity_id))
    diagnostics = {
        "algorithm": ROLE_OPPORTUNITY_VERSION,
        "track_id": track_id,
        "sample_rate": sample_rate,
        "duration_samples": duration,
        "beat_count": len(beats),
        "downbeat_count": len(downbeats),
        "word_count": len(words),
        "provided_count": len(provided),
        "completion_event_count": len(_completion_records(track)),
        "opportunity_counts": {
            role: sum(1 for item in opportunities if item.role == role)
            for role in (OUTGOING, INCOMING)
        },
        "sources": sorted({item.source for item in opportunities}),
        "no_fake_beats": not beats and not downbeats,
        "anchors_are_provider_samples": True,
        "max_grid_snap_seconds": cfg.max_grid_snap_seconds,
        "asr_segment_endpoints_used_as_phrase": False,
        "missing_asr_is_unknown": True,
    }
    return TrackOpportunities(
        track_id=track_id, sample_rate=sample_rate, duration_samples=duration,
        opportunities=tuple(opportunities), beats=tuple(sample for sample, _ in beats),
        downbeats=tuple(sample for sample, _ in downbeats), words=tuple(words),
        warnings=tuple(warnings), diagnostics=diagnostics,
    )


def _median_beat_interval(track: TrackOpportunities) -> float | None:
    samples = track.beats or track.downbeats
    if len(samples) < 2:
        return None
    intervals = sorted(samples[index + 1] - samples[index] for index in range(len(samples) - 1))
    middle = len(intervals) // 2
    if len(intervals) % 2:
        value = float(intervals[middle])
    else:
        value = 0.5 * float(intervals[middle - 1] + intervals[middle])
    return value if value > 0 else None


def _build_plan(
    track_a: TrackOpportunities,
    track_b: TrackOpportunities,
    outgoing: RoleOpportunity,
    incoming: RoleOpportunity,
    config: PlanConfig,
) -> CandidatePlan:
    reasons: list[str] = []
    change = outgoing.anchor_sample
    beat_interval = _median_beat_interval(track_a)
    if beat_interval is not None:
        prepare_lead = int(round(config.prepare_beats * beat_interval))
        prepare_source = "real_beat_interval"
    else:
        prepare_lead = int(round(config.default_prepare_seconds * track_a.sample_rate))
        prepare_source = "constant_without_beat_grid"
    prepare = max(0, change - prepare_lead)
    pickup = int(incoming.pickup_start_sample if incoming.pickup_start_sample is not None else incoming.anchor_sample)
    arrival = int(incoming.arrival_sample if incoming.arrival_sample is not None else incoming.anchor_sample)
    delay = arrival - pickup
    overlap = max(0, change - pickup)
    change_seconds = change / track_a.sample_rate
    pickup_seconds = pickup / track_b.sample_rate
    arrival_seconds = arrival / track_b.sample_rate
    lookahead_seconds = arrival_seconds - change_seconds
    pickup_lead_seconds = change_seconds - pickup_seconds
    delay_seconds = delay / track_b.sample_rate
    word_status = {
        "outgoing": outgoing.word_status,
        "incoming": incoming.word_status,
        "arrival": str(incoming.word_evidence.get("arrival_word_status", incoming.word_status)),
    }
    # The arrival is diagnostic when a pickup already starts the incoming deck.
    # Word protection concerns the two actual entry/exit moments, not a later
    # structural arrival that may legitimately fall inside a sustained word.
    words_protected = (
        word_status["outgoing"] != "unsafe" and word_status["incoming"] != "unsafe"
    )
    passes = bool(outgoing.passes_constraints and incoming.passes_constraints)
    if delay < 0:
        passes = False
        reasons.append("arrival_before_pickup")
    if arrival < change:
        passes = False
        reasons.append("arrival_before_outgoing_change")
    if lookahead_seconds > config.max_arrival_after_change_seconds:
        passes = False
        reasons.append("arrival_after_change_exceeds_limit")
    if pickup_lead_seconds > config.max_pickup_before_change_seconds:
        passes = False
        reasons.append("pickup_before_change_exceeds_limit")
    if delay_seconds > config.max_pickup_to_arrival_seconds:
        passes = False
        reasons.append("pickup_to_arrival_delay_exceeds_limit")
    if prepare > change:
        passes = False
        reasons.append("prepare_after_change")
    entry_word_statuses = (word_status["outgoing"], word_status["incoming"])
    if not config.allow_unknown_words and any(value == "unknown" for value in entry_word_statuses):
        passes = False
        reasons.append("missing_word_evidence_not_allowed")
    if any(value == "unsafe" for value in entry_word_statuses):
        reasons.append("word_overlap_unsafe")
    if any(value == "risk" for value in entry_word_statuses):
        reasons.append("word_within_guard_flagged_not_cut")
    if word_status["arrival"] == "unsafe":
        reasons.append("arrival_inside_word_but_pickup_may_start_at_word_onset")
    if config.require_safe_words and any(value != "supported" for value in entry_word_statuses):
        passes = False
        reasons.append("require_safe_words_not_met")
    if overlap > 0:
        reasons.append("incoming_pickup_starts_before_outgoing_change")
    if pickup < arrival:
        reasons.append("incoming_pickup_distinct_from_arrival")
    if not outgoing.completion_supported:
        reasons.append("outgoing_exit_has_no_named_completion_provider")
    incoming_role = "pickup" if pickup < arrival else "arrival"
    events = (
        PlanEvent("wait", prepare, "A", 0, "hold deck A until preparation window"),
        PlanEvent("prepare", prepare, "A", None, f"prepare lead from {prepare_source}"),
        PlanEvent("change", change, "A", None,
                  "provider exit anchor; completion evidence="
                  + ("supported" if outgoing.completion_supported else "not_supported")),
    )
    evidence = {
        "outgoing_source": outgoing.source,
        "outgoing_evidence_class": outgoing.evidence_class,
        "outgoing_completion_supported": outgoing.completion_supported,
        "incoming_source": incoming.source,
        "incoming_evidence_class": incoming.evidence_class,
        "incoming_arrival_source": incoming.arrival_source,
        "incoming_pickup_source": incoming.word_evidence.get("evidence", "word_interval_timing")
        if pickup < arrival else "arrival_only",
        "outgoing_grid_alignment": outgoing.grid_alignment,
        "incoming_grid_alignment": incoming.grid_alignment,
        "outgoing_grid_delta_samples": outgoing.grid_delta_samples,
        "incoming_grid_delta_samples": incoming.grid_delta_samples,
        "prepare_lead_source": prepare_source,
        "pickup_lead_before_change_seconds": pickup_lead_seconds,
        "pickup_to_arrival_seconds": delay_seconds,
        "timing_limits_seconds": {
            "max_arrival_after_change": config.max_arrival_after_change_seconds,
            "max_pickup_before_change": config.max_pickup_before_change_seconds,
            "max_pickup_to_arrival": config.max_pickup_to_arrival_seconds,
        },
        "selection_semantics": "evidence_completeness_not_musical_quality",
    }
    return CandidatePlan(
        plan_id=_plan_id(track_a.track_id, track_b.track_id, outgoing.opportunity_id, incoming.opportunity_id),
        track_a=track_a.track_id,
        track_b=track_b.track_id,
        outgoing_opportunity_id=outgoing.opportunity_id,
        incoming_opportunity_id=incoming.opportunity_id,
        cue_out_sample=change,
        pickup_start_sample=pickup,
        arrival_sample=arrival,
        change_sample=change,
        prepare_sample=prepare,
        entry_window_start_sample=pickup,
        entry_window_end_sample=arrival,
        arrival_delay_samples=delay,
        lookahead_arrival_after_change_seconds=lookahead_seconds,
        overlap_samples=overlap,
        sample_rate_a=track_a.sample_rate,
        sample_rate_b=track_b.sample_rate,
        events=events,
        outgoing_completion_supported=outgoing.completion_supported,
        outgoing_completion_source=outgoing.source if outgoing.completion_supported else None,
        incoming_role=incoming_role,
        incoming_arrival_source=incoming.arrival_source,
        incoming_pickup_source=str(evidence["incoming_pickup_source"]),
        word_status=word_status,
        words_protected=words_protected,
        evidence=evidence,
        passes_constraints=passes,
        status="ready" if passes else "defer",
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _evidence_rank(plan: CandidatePlan) -> tuple[int, int, int, int, int, str]:
    evidence = plan.evidence
    return (
        1 if plan.incoming_role == "pickup" else 0,
        1 if plan.outgoing_completion_supported else 0,
        1 if evidence.get("outgoing_evidence_class") == "provided_role" else 0,
        1 if evidence.get("incoming_evidence_class") == "provided_role" else 0,
        sum(1 for value in plan.word_status.values() if value == "supported"),
        plan.plan_id,
    )


def select_role_plans(
    plans: Sequence[CandidatePlan],
    *,
    config: PlanConfig | None = None,
) -> tuple[list[CandidatePlan], list[dict[str, Any]], dict[str, Any]]:
    """Diversify tracks/regions without using a fabricated quality score."""

    cfg = config or PlanConfig()
    rejected: list[dict[str, Any]] = []
    candidates = [plan for plan in plans if plan.passes_constraints]
    for plan in plans:
        if not plan.passes_constraints:
            rejected.append({"plan_id": plan.plan_id, "reason": "constraint_failed", "details": list(plan.reasons)})
    candidates.sort(key=lambda plan: tuple(-value if isinstance(value, int) else value
                                           for value in _evidence_rank(plan)))
    selected: list[CandidatePlan] = []
    out_counts: dict[str, int] = {}
    in_counts: dict[str, int] = {}
    pair_counts: dict[tuple[str, str], int] = {}
    used_a: dict[str, list[int]] = {}
    used_b: dict[str, list[int]] = {}
    for plan in candidates:
        if len(selected) >= cfg.max_plans:
            rejected.append({"plan_id": plan.plan_id, "reason": "max_plans"})
            continue
        pair = (plan.track_a, plan.track_b)
        if pair_counts.get(pair, 0) >= cfg.max_plans_per_pair:
            rejected.append({"plan_id": plan.plan_id, "reason": "max_plans_per_pair"})
            continue
        if out_counts.get(plan.track_a, 0) >= cfg.max_per_track_role:
            rejected.append({"plan_id": plan.plan_id, "reason": "outgoing_track_role_cap"})
            continue
        if in_counts.get(plan.track_b, 0) >= cfg.max_per_track_role:
            rejected.append({"plan_id": plan.plan_id, "reason": "incoming_track_role_cap"})
            continue
        spacing_a = int(round(cfg.min_region_spacing_seconds * plan.sample_rate_a))
        spacing_b = int(round(cfg.min_region_spacing_seconds * plan.sample_rate_b))
        if any(abs(plan.change_sample - old) < spacing_a for old in used_a.get(plan.track_a, [])):
            rejected.append({"plan_id": plan.plan_id, "reason": "outgoing_region_spacing"})
            continue
        if any(abs(plan.pickup_start_sample - old) < spacing_b for old in used_b.get(plan.track_b, [])):
            rejected.append({"plan_id": plan.plan_id, "reason": "incoming_region_spacing"})
            continue
        selected.append(plan)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        out_counts[plan.track_a] = out_counts.get(plan.track_a, 0) + 1
        in_counts[plan.track_b] = in_counts.get(plan.track_b, 0) + 1
        used_a.setdefault(plan.track_a, []).append(plan.change_sample)
        used_b.setdefault(plan.track_b, []).append(plan.pickup_start_sample)
    rejection_counts: dict[str, int] = {}
    for item in rejected:
        reason = str(item["reason"])
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    diagnostics = {
        "schema_version": PLAN_SCHEMA,
        "selection_basis": "evidence_completeness_not_musical_quality",
        "input_plans": len(plans),
        "constraint_eligible": len(candidates),
        "selected_count": len(selected),
        "max_plans": cfg.max_plans,
        "max_per_track_role": cfg.max_per_track_role,
        "max_plans_per_pair": cfg.max_plans_per_pair,
        "min_region_spacing_seconds": cfg.min_region_spacing_seconds,
        "selected_outgoing_counts": dict(sorted(out_counts.items())),
        "selected_incoming_counts": dict(sorted(in_counts.items())),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "no_quality_score_invented": True,
    }
    return selected, rejected, diagnostics


def _catalog_tracks(catalog: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if isinstance(catalog, Mapping):
        if "tracks" in catalog:
            raw = catalog["tracks"]
        else:
            raw = [dict(value, track_id=value.get("track_id", key)) if isinstance(value, Mapping) else value
                   for key, value in catalog.items() if isinstance(value, Mapping)]
        if isinstance(raw, Mapping):
            raw = [dict(value, track_id=value.get("track_id", key)) if isinstance(value, Mapping) else value
                   for key, value in raw.items()]
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            return [track for track in raw if isinstance(track, Mapping)]
    if isinstance(catalog, Sequence) and not isinstance(catalog, (str, bytes)):
        return [track for track in catalog if isinstance(track, Mapping)]
    raise TypeError("catalog must be a mapping with tracks or a sequence of track mappings")


def generate_role_plans(
    catalog: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    config: PlanConfig | None = None,
    max_pair_combinations: int = 200_000,
) -> RolePlanResult:
    """Extract all role opportunities and produce diversified candidate plans."""

    cfg = config or PlanConfig()
    track_inputs = _catalog_tracks(catalog)
    extracted = [extract_track_opportunities(track, config=cfg) for track in track_inputs]
    by_id = {track.track_id: track for track in extracted}
    if len(by_id) != len(extracted):
        raise ValueError("catalog track_id values must be unique")
    proposals: list[CandidatePlan] = []
    combination_count = 0
    for track_a in extracted:
        for track_b in extracted:
            if track_a.track_id == track_b.track_id:
                continue
            outgoing = [item for item in track_a.opportunities if item.role == OUTGOING]
            incoming = [item for item in track_b.opportunities if item.role == INCOMING]
            combination_count += len(outgoing) * len(incoming)
            if combination_count > max_pair_combinations:
                raise ValueError("catalog produces too many role combinations; reduce anchors or raise the bound")
            for out in outgoing:
                for inc in incoming:
                    proposals.append(_build_plan(track_a, track_b, out, inc, cfg))
    selected, rejected, selection_diagnostics = select_role_plans(proposals, config=cfg)
    opportunities = tuple(item for track in extracted for item in track.opportunities)
    role_counts = {
        role: {
            "total": sum(1 for item in opportunities if item.role == role),
            "completion_supported": sum(
                1 for item in opportunities if item.role == role and item.completion_supported
            ),
            "safe_words": sum(1 for item in opportunities if item.role == role and item.word_status == "supported"),
            "unsafe_words": sum(1 for item in opportunities if item.role == role and item.word_status == "unsafe"),
            "unknown_words": sum(1 for item in opportunities if item.role == role and item.word_status == "unknown"),
        }
        for role in (OUTGOING, INCOMING)
    }
    diagnostics = {
        "schema_version": PLAN_SCHEMA,
        "algorithm": ROLE_OPPORTUNITY_VERSION,
        "track_count": len(extracted),
        "track_ids": [track.track_id for track in extracted],
        "role_counts": role_counts,
        "track_diagnostics": {track.track_id: dict(track.diagnostics) for track in extracted},
        "warnings": {track.track_id: [dict(item) for item in track.warnings] for track in extracted},
        "proposal_count": len(proposals),
        "selected_count": len(selected),
        "selection": selection_diagnostics,
        "role_semantics": {
            "outgoing": "exit anchor; completion_supported is separate and only from a named event",
            "incoming": "pickup_start may precede arrival; arrival_delay_samples is explicit",
            "grid": "beats/downbeats are context only; provider anchors are never shifted",
            "asr": "word intervals protect cuts and may supply pickups; decoder segment ends are not phrases",
        },
        "tracks_without_grid": [track.track_id for track in extracted if not track.beats and not track.downbeats],
        "no_synthetic_beat_events": True,
        "max_observed_grid_delta_samples": max(
            (abs(item.grid_delta_samples) for item in opportunities if item.grid_delta_samples is not None),
            default=0,
        ),
        "max_anchor_shift_samples": 0,
    }
    return RolePlanResult(
        schema_version=PLAN_SCHEMA,
        opportunities=opportunities,
        selected_plans=tuple(selected),
        proposals=tuple(proposals),
        rejected=tuple(rejected),
        diagnostics=diagnostics,
    )


def plan_catalog(
    catalog: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    config: PlanConfig | None = None,
    max_pair_combinations: int = 200_000,
) -> dict[str, Any]:
    """JSON-ready convenience wrapper for the root pipeline."""

    return generate_role_plans(
        catalog, config=config, max_pair_combinations=max_pair_combinations
    ).to_dict()


__all__ = [
    "CandidatePlan",
    "INCOMING",
    "OUTGOING",
    "PLAN_SCHEMA",
    "PlanConfig",
    "PlanEvent",
    "ROLE_OPPORTUNITY_VERSION",
    "RoleOpportunity",
    "RolePlanResult",
    "TrackOpportunities",
    "extract_track_opportunities",
    "generate_role_plans",
    "plan_catalog",
    "select_role_plans",
]
