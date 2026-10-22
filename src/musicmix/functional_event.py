"""Event-centric, abstaining comparator for A -> B transition evidence.

The comparator is deliberately model-free and unvalidated.  It consumes four
beat-synchronous profiles produced by :mod:`musicmix.pooled_trajectory` (or the
same JSON shape): the outgoing and incoming context of A, and the outgoing and
incoming context of B.  Its central question is not "are two trends equal?" but
"what happens at B's entry, and does the post-entry region persist or merely
spike?".

Evidence is kept per acoustic domain.  A continuation interpretation requires a
weighted majority of salient domains, rejects salient reversals, and is not
created by two matching domains when other strong domains oppose it.  Missing
profiles or missing independent vocal/phrase evidence lead to ``unknown`` and a
null adjustment rather than an invented score.  No section label is mapped to
resolution, and track identity is never rewarded.

v1.2 (integrated 2026-10-20 from the functional_event agent revision, see
``conoscenza_mix/01_psicologia_aspettativa/v12_functional_event/INTEGRAZIONE.md``):
continuation and arrival are separate routes.  A salient A/B reversal vetoes
continuation only; ``post_entry_attack_stabilized`` now requires a directed
``A_before`` context plus a persistent local B entry, and a local B attack after
an undirected A abstains.  Known limit: on real pool audio the 8-beat direction
proxy is close to its noise level, so the directed-A gate passes most cues.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import fmean, median, pstdev
from typing import Any, Iterable, Mapping, Sequence

from .expectation import Evidence


VERSION = "functional-event-v1.2-independent-arrival-route"
TRANSFER_STATUS = "unvalidated_engineering_comparator"
DOMAIN_NAMES = ("energy", "events", "bass", "timbre", "tonal_proxy")
LABELS = frozenset(
    {
        "post_entry_attack_stabilized",
        "post_entry_attack_transient",
        "preparation_not_realized",
        "pickup_or_preparation",
        "sustained_motion",
        "continuation_without_local_event",
        "conflict_entry_vs_continuation",
        "unknown",
    }
)
EVIDENCE_STATUSES = frozenset({"supported", "weak", "absent", "unknown"})

# field -> kind.  "log" is an additive level in log units; "positive" is a
# non-negative quantity compared as a relative change.  These are descriptors,
# not measures of tension or closure.
_DOMAIN_FIELDS: dict[str, tuple[tuple[str, str], ...]] = {
    "energy": (("log_rms_mean", "log"), ("rms_mean", "log_positive")),
    "events": (
        ("spectral_flux_mean", "positive"),
        ("attack_density_per_second", "positive"),
    ),
    "bass": (("bass_ratio", "positive"),),
    "timbre": (
        ("high_frequency_ratio", "positive"),
        ("spectral_centroid_hz", "positive"),
    ),
    "tonal_proxy": (("chroma_movement", "positive"),),
}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bounded(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _mean(values: Sequence[float]) -> float | None:
    return fmean(values) if values else None


@dataclass(frozen=True)
class FunctionalEventThresholds:
    """Versioned engineering hypotheses, not calibrated probabilities."""

    version: str = "functional-event-thresholds-v1"
    trend_threshold: float = 0.18
    flat_threshold: float = 0.08
    event_threshold: float = 0.30
    persistence_retention: float = 0.50
    min_domains: int = 2
    agreement_floor: float = 0.55
    contradiction_weight: float = 0.30
    attack_head_beats: int = 2
    preparation_lead_beats: int = 3
    stabilization_tail_beats: int = 3

    def __post_init__(self) -> None:
        for name in (
            "trend_threshold",
            "flat_threshold",
            "event_threshold",
            "persistence_retention",
            "agreement_floor",
            "contradiction_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if int(self.min_domains) < 2:
            raise ValueError("min_domains must be at least two")
        for name in (
            "attack_head_beats",
            "preparation_lead_beats",
            "stabilization_tail_beats",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(self, "min_domains", int(self.min_domains))


@dataclass(frozen=True)
class DomainEventEvidence:
    """One domain's pre, post, stabilized and local-event observations."""

    domain: str
    a_before_direction: float | None
    a_after_direction: float | None
    b_before_direction: float | None
    b_before_late_direction: float | None
    b_post_direction: float | None
    b_stabilized_direction: float | None
    event_contrast: float | None
    retention: float | None
    status: str
    missing: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.domain not in DOMAIN_NAMES:
            raise ValueError("unknown surface domain")
        if self.status not in EVIDENCE_STATUSES:
            raise ValueError("invalid domain evidence status")
        for name in (
            "a_before_direction",
            "a_after_direction",
            "b_before_direction",
            "b_before_late_direction",
            "b_post_direction",
            "b_stabilized_direction",
            "event_contrast",
            "retention",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite or null")
        object.__setattr__(self, "missing", tuple(dict.fromkeys(self.missing)))


@dataclass(frozen=True)
class FunctionalEventAssessment:
    """Abstaining, per-domain report for one A -> B candidate."""

    label: str
    abstained: bool
    evidence: tuple[Evidence, ...]
    domains: Mapping[str, DomainEventEvidence]
    contributions: Mapping[str, float]
    missing: tuple[str, ...]
    score: float | None
    confidence: None = None
    diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"label must be one of {sorted(LABELS)}")
        if self.confidence is not None:
            raise ValueError("confidence is intentionally uncalibrated")
        if self.abstained and self.score is not None:
            raise ValueError("abstained assessments cannot expose a score")
        evidence = tuple(self.evidence)
        if any(not isinstance(item, Evidence) for item in evidence):
            raise TypeError("evidence must contain Evidence objects")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "domains", dict(self.domains))
        object.__setattr__(
            self,
            "contributions",
            {str(key): float(value) for key, value in (self.contributions or {}).items()},
        )
        object.__setattr__(self, "missing", tuple(dict.fromkeys(self.missing)))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics or {}))
        if self.score is not None:
            object.__setattr__(self, "score", _bounded(float(self.score)))

    @property
    def evidence_by_kind(self) -> dict[str, Evidence]:
        return {item.kind: item for item in self.evidence}

    @property
    def entry_evidence(self) -> Evidence | None:
        return self.evidence_by_kind.get("b_before_to_b_after_event")


@dataclass(frozen=True)
class FunctionalEventRerank:
    """Baseline-preserving optional rerank result."""

    baseline_score: float | None
    event_score: float | None
    adjusted_score: float | None
    weight: float
    enabled: bool
    label: str
    abstained: bool
    rationale: str


@dataclass(frozen=True)
class VocalObservation:
    pickup: str
    completion: str
    word_active_at_cue: bool | None
    details: Mapping[str, Any]
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class SectionObservation:
    status: str
    before_label: str | None
    after_label: str | None
    changed: bool | None
    provider: str | None
    details: Mapping[str, Any]
    missing: tuple[str, ...] = ()


def _records(profile: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(profile, Mapping):
        return []
    raw = profile.get("windows", ())
    records: list[dict[str, Any]] = []
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(value, Mapping):
                record = dict(value)
                record.setdefault("window_beats", key)
                records.append(record)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        records.extend(dict(item) for item in raw if isinstance(item, Mapping))
    if records:
        return records
    series = profile.get("series_per_beat")
    if isinstance(series, Mapping):
        for key, value in series.items():
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                records.append({"window_beats": key, "beat_feature_series": list(value)})
    return records


def _points(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = record.get("beat_feature_series", record.get("series", ()))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [dict(point) for point in raw if isinstance(point, Mapping)]


def _select_series(
    profile: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], tuple[str, ...]]:
    """Pick the shortest window with enough real beat points.

    Nothing is interpolated or padded.  A profile without an observed
    beat-feature series is reported missing.
    """

    if not isinstance(profile, Mapping):
        return None, [], ("profile",)
    usable: list[tuple[int, int, dict[str, Any], list[dict[str, Any]]]] = []
    for record in _records(profile):
        points = _points(record)
        if len(points) >= 6:
            try:
                window = int(record.get("window_beats"))
            except (TypeError, ValueError):
                window = 10**6
            usable.append((window, -len(points), record, points))
    if not usable:
        return None, [], ("beat_feature_series_with_at_least_6_observed_points",)
    usable.sort(key=lambda item: (item[0], item[1]))
    _, _, record, points = usable[0]
    return record, points, ()


def _values(points: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for point in points:
        value = _finite(point.get(field))
        if value is not None:
            values.append(value)
    return values


def _field_direction(values: Sequence[float], kind: str) -> float | None:
    numbers = [float(value) for value in values if math.isfinite(float(value))]
    if len(numbers) < 2:
        return None
    count = max(1, len(numbers) // 4)
    delta = fmean(numbers[-count:]) - fmean(numbers[:count])
    if kind == "log":
        scale = max(pstdev(numbers), 0.12, 1e-6)
    else:
        scale = max(pstdev(numbers), abs(median(numbers)) * 0.2, 1e-6)
    return _bounded(delta / (2.0 * scale))


def _domain_direction(
    points: Sequence[Mapping[str, Any]],
    domain: str,
    *,
    start: int | None = None,
    end: int | None = None,
) -> float | None:
    selected = list(points)[start:end]
    scores: list[float] = []
    for field, kind in _DOMAIN_FIELDS[domain]:
        score = _field_direction(_values(selected, field), kind)
        if score is not None:
            scores.append(score)
    return fmean(scores) if scores else None


def _field_contrast(
    early: Sequence[float], late: Sequence[float], kind: str
) -> float | None:
    if not early or not late:
        return None
    early_mean = fmean(early)
    late_mean = fmean(late)
    if kind == "log":
        # ~0.35 natural-log units (about 3 dB amplitude) maps to a strong
        # bounded contrast.  This is an engineering scale.
        return _bounded(math.tanh((early_mean - late_mean) / 0.35))
    if kind == "log_positive":
        if early_mean <= 0.0 or late_mean <= 0.0:
            return None
        return _bounded(math.tanh(math.log(early_mean / late_mean) / math.log(2.0)))
    floor = max(
        1e-9,
        0.02
        * max(
            abs(early_mean),
            abs(late_mean),
            abs(median(list(early) + list(late))),
        ),
    )
    ratio = math.log((early_mean + floor) / (late_mean + floor))
    return _bounded(math.tanh(ratio / math.log(2.0)))


def _domain_contrast(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    domain: str,
    *,
    head: int,
) -> float | None:
    head = max(1, int(head))
    early_points = list(after)[:head]
    late_points = list(before)[-head:]
    scores: list[float] = []
    for field, kind in _DOMAIN_FIELDS[domain]:
        score = _field_contrast(
            _values(early_points, field),
            _values(late_points, field),
            kind if kind != "log_positive" else "log_positive",
        )
        if score is not None:
            scores.append(score)
    return fmean(scores) if scores else None


def _retention(
    points: Sequence[Mapping[str, Any]], domain: str, *, head: int, tail: int
) -> float | None:
    if len(points) <= head + 1:
        return None
    early_points = list(points)[:head]
    tail_points = list(points)[-max(1, tail):]
    ratios: list[float] = []
    for field, kind in _DOMAIN_FIELDS[domain]:
        early = _values(early_points, field)
        later = _values(tail_points, field)
        if not early or not later:
            continue
        early_mean = fmean(early)
        later_mean = fmean(later)
        if kind == "log":
            ratio = math.exp(later_mean - early_mean)
        elif kind == "log_positive":
            if early_mean <= 0.0 or later_mean <= 0.0:
                continue
            ratio = later_mean / early_mean
        else:
            floor = max(1e-9, 0.05 * abs(early_mean))
            ratio = (later_mean + floor) / (early_mean + floor)
        if math.isfinite(ratio):
            ratios.append(_bounded(ratio, 0.0, 2.0))
    return fmean(ratios) if ratios else None


def _domain_table(
    a_before: list[dict[str, Any]],
    a_after: list[dict[str, Any]],
    b_before: list[dict[str, Any]],
    b_after: list[dict[str, Any]],
    thresholds: FunctionalEventThresholds,
) -> dict[str, DomainEventEvidence]:
    head = thresholds.attack_head_beats
    lead = thresholds.preparation_lead_beats
    tail = thresholds.stabilization_tail_beats
    table: dict[str, DomainEventEvidence] = {}
    for domain in DOMAIN_NAMES:
        pre_lead_start = max(0, len(b_before) - lead)
        post_stab_start = min(len(b_after), head)
        values = DomainEventEvidence(
            domain=domain,
            a_before_direction=_domain_direction(a_before, domain),
            a_after_direction=_domain_direction(a_after, domain),
            b_before_direction=_domain_direction(b_before, domain),
            b_before_late_direction=_domain_direction(
                b_before, domain, start=pre_lead_start
            ),
            b_post_direction=_domain_direction(b_after, domain),
            b_stabilized_direction=_domain_direction(
                b_after, domain, start=post_stab_start
            ),
            event_contrast=_domain_contrast(
                b_before, b_after, domain, head=head
            ),
            retention=_retention(b_after, domain, head=head, tail=tail),
            status="unknown",
        )
        missing: list[str] = []
        if values.a_before_direction is None:
            missing.append("A_before_direction")
        if values.a_after_direction is None:
            missing.append("A_after_reference_direction")
        if values.b_before_direction is None:
            missing.append("B_before_direction")
        if values.b_stabilized_direction is None:
            missing.append("B_stabilized_direction")
        if values.event_contrast is None:
            missing.append("B_before_to_B_after_event_contrast")
        if values.retention is None:
            missing.append("B_post_entry_retention")
        if values.event_contrast is not None:
            values = DomainEventEvidence(
                **{
                    **values.__dict__,
                    "status": "supported"
                    if abs(values.event_contrast) >= thresholds.event_threshold
                    else "weak",
                    "missing": tuple(missing),
                }
            )
        else:
            values = DomainEventEvidence(
                **{**values.__dict__, "status": "unknown", "missing": tuple(missing)}
            )
        table[domain] = values
    return table


def _signed_pairs(
    left: Mapping[str, DomainEventEvidence],
    right_direction: str,
    thresholds: FunctionalEventThresholds,
    *,
    left_direction: str,
) -> tuple[list[tuple[str, float, float, float]], list[tuple[str, float, float, float]], float]:
    matches: list[tuple[str, float, float, float]] = []
    reversals: list[tuple[str, float, float, float]] = []
    total = 0.0
    for domain in DOMAIN_NAMES:
        item = left[domain]
        lvalue = getattr(item, left_direction)
        rvalue = getattr(item, right_direction)
        if lvalue is None or rvalue is None:
            continue
        if abs(lvalue) < thresholds.trend_threshold or abs(rvalue) < thresholds.trend_threshold:
            continue
        weight = (abs(lvalue) + abs(rvalue)) / 2.0
        total += weight
        if lvalue * rvalue > 0:
            matches.append((domain, lvalue, rvalue, weight))
        else:
            reversals.append((domain, lvalue, rvalue, weight))
    return matches, reversals, total


def _continuation_evidence(
    table: Mapping[str, DomainEventEvidence],
    thresholds: FunctionalEventThresholds,
) -> tuple[Evidence, bool]:
    matches, reversals, total = _signed_pairs(
        table,
        "b_stabilized_direction",
        thresholds,
        left_direction="a_before_direction",
    )
    match_weight = sum(item[3] for item in matches)
    reverse_weight = sum(item[3] for item in reversals)
    agreement = match_weight / total if total > 0 else None
    contradiction = (
        len(reversals) >= thresholds.min_domains
        or (reverse_weight > 0.0 and reverse_weight >= thresholds.contradiction_weight * max(total, 1e-9))
    )
    supported = (
        len(matches) >= thresholds.min_domains
        and agreement is not None
        and agreement >= thresholds.agreement_floor
        and not contradiction
    )
    status = "supported" if supported else ("unknown" if total == 0 else "absent")
    score = agreement if supported else None
    return (
        Evidence(
            "continuation",
            status,
            score,
            {domain: weight for domain, _, _, weight in matches},
            tuple(
                ["A_before", "B_after"]
                if total == 0
                else (["salient_domain_reversal"] if contradiction else [])
            ),
            {
                "matching_domains": [item[0] for item in matches],
                "reversing_domains": [item[0] for item in reversals],
                "agreement_weight": agreement,
                "reverse_weight": reverse_weight,
                "salient_contradiction": contradiction,
                "comparison": "A_before_direction_vs_B_stabilized_direction",
            },
        ),
        contradiction,
    )


def _reference_evidence(
    table: Mapping[str, DomainEventEvidence],
    thresholds: FunctionalEventThresholds,
) -> Evidence:
    matches, reversals, total = _signed_pairs(
        table,
        "b_stabilized_direction",
        thresholds,
        left_direction="a_after_direction",
    )
    if total == 0:
        return Evidence(
            "a_after_reference",
            "unknown",
            None,
            {},
            ("A_after_reference",),
            {"comparison": "A_after_reference_direction_vs_B_stabilized_direction"},
        )
    match_weight = sum(item[3] for item in matches)
    reverse_weight = sum(item[3] for item in reversals)
    agreement = match_weight / total
    supported = (
        len(matches) >= thresholds.min_domains
        and agreement >= thresholds.agreement_floor
    )
    return Evidence(
        "a_after_reference",
        "supported" if supported else "weak",
        agreement if supported else None,
        {domain: weight for domain, _, _, weight in matches},
        (),
        {
            "matching_domains": [item[0] for item in matches],
            "reversing_domains": [item[0] for item in reversals],
            "agreement_weight": agreement,
            "reverse_weight": reverse_weight,
            "comparison": "A_after_reference_direction_vs_B_stabilized_direction",
            "identity_or_waveform_similarity_used": False,
        },
    )


def _entry_evidence(
    table: Mapping[str, DomainEventEvidence],
    thresholds: FunctionalEventThresholds,
) -> tuple[Evidence, list[str], list[str], float | None]:
    contrasts = {
        domain: item.event_contrast
        for domain, item in table.items()
        if item.event_contrast is not None
    }
    positive = {
        domain: value
        for domain, value in contrasts.items()
        if value >= thresholds.event_threshold
    }
    negative = {
        domain: value
        for domain, value in contrasts.items()
        if value <= -thresholds.event_threshold
    }
    retention = {
        domain: item.retention
        for domain, item in table.items()
        if item.retention is not None
    }
    persistence_domains = [
        domain
        for domain in positive
        if retention.get(domain) is not None
        and retention[domain] >= thresholds.persistence_retention
    ]
    # A post-entry attack should be visible in energy or event activity, not
    # only in a correlated timbre/chroma proxy.
    core_attack = [domain for domain in positive if domain in {"energy", "events", "bass"}]
    core_release = [domain for domain in negative if domain in {"energy", "events", "bass"}]
    attack_supported = len(positive) >= thresholds.min_domains and bool(core_attack)
    persistence_supported = bool(persistence_domains) and len(persistence_domains) >= 1
    if attack_supported:
        strengths = [positive[domain] for domain in positive]
        entry_score = fmean(strengths)
        if not persistence_supported:
            entry_score = -max(0.0, entry_score) * 0.6
        status = "supported" if persistence_supported else "weak"
    elif core_release:
        # A strong post-entry loss in energy/events/bass means the pre-entry
        # preparation was not carried through the entry.
        entry_score = fmean([negative[domain] for domain in core_release])
        status = "weak"
    elif positive:
        entry_score = fmean(list(positive.values())) * 0.5
        status = "weak"
    elif negative:
        entry_score = fmean(list(negative.values()))
        status = "weak"
    else:
        entry_score = None
        status = "absent"
    missing: list[str] = []
    if contrasts is None or not contrasts:
        missing.append("B_before_to_B_after_event_contrast")
    return (
        Evidence(
            "b_before_to_b_after_event",
            status,
            entry_score,
            {"event_contrast": value for value in positive.values()}
            | {"event_release": value for value in negative.values()},
            tuple(missing),
            {
                "positive_domains": sorted(positive),
                "negative_domains": sorted(negative),
                "persistence_domains": persistence_domains,
                "retention": retention,
                "attack_supported": attack_supported,
                "persistence_supported": persistence_supported,
                "semantics": "local B_before -> B_after contrast; not resolution",
            },
        ),
        sorted(positive),
        persistence_domains,
        entry_score,
    )


def _preparation_evidence(
    table: Mapping[str, DomainEventEvidence],
    thresholds: FunctionalEventThresholds,
    vocal: VocalObservation,
) -> tuple[Evidence, list[str]]:
    domains = [
        domain
        for domain, item in table.items()
        if item.b_before_late_direction is not None
        and item.b_before_late_direction >= thresholds.trend_threshold
    ]
    explicit_pickup = vocal.pickup == "supported"
    supported = len(domains) >= thresholds.min_domains or explicit_pickup
    score = (
        fmean(
            [
                table[domain].b_before_late_direction
                for domain in domains
                if table[domain].b_before_late_direction is not None
            ]
        )
        if domains
        else (0.35 if explicit_pickup else None)
    )
    return (
        Evidence(
            "pre_entry_preparation",
            "supported" if supported else ("absent" if table else "unknown"),
            score if supported else None,
            {domain: table[domain].b_before_late_direction or 0.0 for domain in domains},
            () if supported else (("pre_entry_pickup_evidence",) if not explicit_pickup else ()),
            {
                "rising_domains": domains,
                "explicit_vocal_pickup": explicit_pickup,
                "semantics": "pre-entry preparation/pickup; not a resolution",
            },
        ),
        domains,
    )


def _directed_a_evidence(
    table: Mapping[str, DomainEventEvidence],
    thresholds: FunctionalEventThresholds,
) -> tuple[Evidence, bool, list[str]]:
    """Whether A's outgoing context projects a directed (non-flat) motion.

    This is an expectation gate for the arrival route, not a resolution score.
    A flat or unmeasured A cannot turn a local B attack into a projected
    arrival; it is reported as absent/unknown instead.
    """

    available: list[str] = []
    salient: list[str] = []
    positive: list[str] = []
    negative: list[str] = []
    for domain in DOMAIN_NAMES:
        value = table[domain].a_before_direction
        if value is None:
            continue
        available.append(domain)
        if abs(value) >= thresholds.trend_threshold:
            salient.append(domain)
            if value > 0:
                positive.append(domain)
            else:
                negative.append(domain)
    dominant = positive if len(positive) >= len(negative) else negative
    supported = (
        len(available) >= thresholds.min_domains
        and len(dominant) >= thresholds.min_domains
        and len(dominant) > len(salient) - len(dominant)
    )
    if len(available) < thresholds.min_domains:
        status = "unknown"
    else:
        status = "supported" if supported else "absent"
    score = (
        fmean([table[domain].a_before_direction for domain in dominant])
        if supported
        else None
    )
    missing = ()
    if status == "unknown":
        missing = ("A_before_direction",)
    elif not supported:
        missing = ("directed_A_motion",)
    return (
        Evidence(
            "a_directed_expectation",
            status,
            score,
            {domain: table[domain].a_before_direction or 0.0 for domain in salient},
            missing,
            {
                "salient_domains": salient,
                "positive_domains": positive,
                "negative_domains": negative,
                "comparison": "A_before_direction_is_directed",
                "flat_A_is_not_projected_expectation": True,
            },
        ),
        supported,
        dominant,
    )


def _arrival_evidence(
    entry: Evidence,
    directed_a: Evidence,
    directed_supported: bool,
) -> Evidence:
    """Locally sustained B entry after directed A motion.

    Arrival is an alternative function to continuation: it does not require
    B's stabilized direction to agree with A. It does require both independent
    facts, a directed A expectation and a sustained local B entry. A local
    attack without directed A remains unproven.
    """

    entry_supported = entry.status == "supported" and entry.score is not None
    supported = bool(entry_supported and directed_supported)
    if supported:
        status = "supported"
        score = float(entry.score)
    elif entry_supported:
        status = "absent"
        score = None
    else:
        status = "unknown"
        score = None
    missing: list[str] = []
    if not entry_supported:
        missing.append("stabilized_post_entry_event")
    if not directed_supported:
        missing.append("directed_a_expectation")
    return Evidence(
        "directed_arrival",
        status,
        score,
        {"local_entry": float(score)} if score is not None else {},
        tuple(missing),
        {
            "requires_directed_a_expectation": True,
            "requires_local_sustained_entry": True,
            "continuation_agreement_required": False,
            "entry_status": entry.status,
            "directed_a_status": directed_a.status,
            "flat_A_plus_local_attack_is_unproven": True,
            "is_calibrated_resolution_reward": False,
            "semantics": "arrival after directed A preparation; not a trained resolution reward",
        },
    )


def _local_entry_evidence(entry: Evidence) -> Evidence:
    """The local B_before -> B_after event, explicitly not cross-song arrival."""

    return Evidence(
        "local_entry_event",
        entry.status,
        entry.score,
        dict(entry.contributions),
        tuple(entry.missing),
        {
            **dict(entry.diagnostics),
            "is_cross_song_resolution": False,
            "is_cross_song_arrival": False,
            "semantics": "local B entry only; it cannot certify arrival or release by itself",
        },
    )


def _voice(vocal: Any) -> VocalObservation:
    if not isinstance(vocal, Mapping):
        return VocalObservation(
            "unknown",
            "unknown",
            None,
            {"available": False},
            ("vocal_completion", "pre_entry_pickup"),
        )
    details = dict(vocal)
    text = " ".join(
        str(details.get(key, ""))
        for key in ("evidence_type", "kind", "status", "boundary_class", "reason")
    ).lower()
    pickup = "supported" if (
        details.get("pickup") is True
        or details.get("pre_entry_pickup") is True
        or details.get("pickup_supported") is True
    ) else "unknown"
    word_gap_only = (
        details.get("word_gap_only") is True
        or details.get("boundary_class") == "word_gap_only"
        or "word_gap" in text and "phrase" not in text
    )
    provider = (
        details.get("provider")
        or details.get("source")
        or details.get("backend")
        or details.get("evidence")
    )
    word_active = details.get("word_active_at_cue")
    if not isinstance(word_active, bool):
        word_active = None
    # An audible word spanning the cue is not evidence of phrase completion.
    word_active_blocks_completion = word_active is True
    completion_supported = (
        not word_gap_only
        and not word_active_blocks_completion
        and (
            details.get("completion") is True
            or details.get("phrase_completion") is True
            or details.get("completion_supported") is True
            or (str(details.get("status", "")).lower() == "supported"
                and str(details.get("boundary_class", "")).lower()
                in {"phrase_boundary_supported", "acoustic_boundary_supported"})
        )
        and bool(provider)
    )
    completion = "supported" if completion_supported else (
        "absent" if word_gap_only or word_active_blocks_completion else "unknown"
    )
    missing: list[str] = []
    if completion != "supported":
        missing.append("independent_phrase_completion")
    if pickup != "supported":
        missing.append("pre_entry_pickup")
    return VocalObservation(
        pickup,
        completion,
        word_active,
        details,
        tuple(missing),
    )


def _section(section_context: Any) -> SectionObservation:
    if not isinstance(section_context, Mapping):
        return SectionObservation(
            "unknown", None, None, None, None, {}, ("functional_section_context",)
        )
    details = dict(section_context)
    before = (
        details.get("before_label")
        or details.get("a_label")
        or details.get("B_before_label")
        or details.get("b_before_label")
    )
    after = (
        details.get("after_label")
        or details.get("b_label")
        or details.get("B_after_label")
        or details.get("b_after_label")
    )
    provider = details.get("provider") or details.get("source") or details.get("backend")
    supported = details.get("supported") is True or str(details.get("status", "")).lower() == "supported"
    if not supported or not provider or not isinstance(before, str) or not isinstance(after, str):
        return SectionObservation(
            "unknown",
            str(before) if before is not None else None,
            str(after) if after is not None else None,
            None,
            provider,
            details,
            ("functional_section_context",),
        )
    return SectionObservation(
        "supported",
        before,
        after,
        before != after,
        str(provider),
        details,
        (),
    )


def _score_label(
    label: str,
    continuation: Evidence,
    entry: Evidence,
    persistence_domains: list[str],
    prep_domains: list[str],
    vocal: VocalObservation,
    section: SectionObservation,
    thresholds: FunctionalEventThresholds,
) -> tuple[float | None, dict[str, float]]:
    contributions: dict[str, float] = {}
    if continuation.status == "supported" and continuation.score is not None:
        contributions["continuation"] = float(continuation.score)
    if entry.score is not None:
        contributions["entry_event"] = float(entry.score)
    contributions["persistence_fraction"] = (
        len(persistence_domains) / max(1, len(DOMAIN_NAMES))
    )
    if prep_domains:
        contributions["preparation_fraction"] = len(prep_domains) / len(DOMAIN_NAMES)
    if vocal.completion == "supported":
        contributions["vocal_completion_context"] = 0.25
    if vocal.pickup == "supported":
        contributions["vocal_pickup_context"] = 0.20
    if section.status == "supported":
        contributions["section_change_context"] = 0.10 if section.changed else 0.0
    if label == "post_entry_attack_stabilized":
        score = 0.45 + 0.30 * max(0.0, contributions.get("entry_event", 0.0))
        score += 0.20 * contributions["persistence_fraction"]
        score += contributions.get("vocal_completion_context", 0.0)
        score += contributions.get("section_change_context", 0.0)
    elif label == "pickup_or_preparation":
        score = 0.18 + 0.25 * contributions.get("preparation_fraction", 0.0)
        score += contributions.get("vocal_pickup_context", 0.0)
    elif label == "sustained_motion":
        score = 0.10 + 0.35 * max(0.0, contributions.get("continuation", 0.0))
    elif label == "continuation_without_local_event":
        score = 0.05 + 0.30 * max(0.0, contributions.get("continuation", 0.0))
    elif label == "preparation_not_realized":
        score = -0.15 + 0.15 * contributions.get("preparation_fraction", 0.0)
        score += 0.30 * min(0.0, contributions.get("entry_event", 0.0))
    elif label == "post_entry_attack_transient":
        score = -0.20 - 0.30 * abs(min(0.0, contributions.get("entry_event", 0.0)))
    else:
        return None, contributions
    return _bounded(score), contributions


def assess_functional_event(
    A_before: Mapping[str, Any] | None,
    A_after: Mapping[str, Any] | None,
    B_before: Mapping[str, Any] | None,
    B_after: Mapping[str, Any] | None,
    *,
    vocal_completion: Any = None,
    section_context: Any = None,
    thresholds: FunctionalEventThresholds | None = None,
) -> FunctionalEventAssessment:
    """Compare event preparation, entry, persistence and A->B continuation.

    The four profiles are the actual analysis inputs.  ``B_before`` is not
    treated as a coverage check: its late direction and its contrast with
    ``B_after`` are required for the local event evidence and therefore for the
    final label and adjustment.
    """

    thresholds = thresholds or FunctionalEventThresholds()
    vocal = _voice(vocal_completion)
    section = _section(section_context)
    missing: list[str] = []

    selected: dict[str, tuple[dict[str, Any] | None, list[dict[str, Any]]]] = {}
    for name, profile in (
        ("A_before", A_before),
        ("A_after", A_after),
        ("B_before", B_before),
        ("B_after", B_after),
    ):
        record, points, profile_missing = _select_series(profile)
        selected[name] = (record, points)
        if profile_missing:
            missing.extend(f"{name}.{item}" for item in profile_missing)

    a_before = selected["A_before"][1]
    a_after = selected["A_after"][1]
    b_before = selected["B_before"][1]
    b_after = selected["B_after"][1]
    complete = all((a_before, a_after, b_before, b_after))
    if not complete:
        evidence = (
            Evidence(
                "continuation", "unknown", None, {}, ("all_four_profiles_required",),
                {"reason": "missing observed beat series; no synthetic beat grid used"},
            ),
            Evidence(
                "b_before_to_b_after_event", "unknown", None, {},
                ("all_four_profiles_required",),
                {"reason": "local B event cannot be measured without B_before and B_after"},
            ),
        )
        return FunctionalEventAssessment(
            "unknown",
            True,
            evidence,
            {},
            {},
            tuple(dict.fromkeys(missing + ["all_four_profiles_required"])),
            None,
            None,
            {
                "version": VERSION,
                "transfer_status": TRANSFER_STATUS,
                "confidence": None,
                "abstention_reason": "incomplete_observed_profiles",
            },
        )

    table = _domain_table(a_before, a_after, b_before, b_after, thresholds)
    continuation, contradiction = _continuation_evidence(table, thresholds)
    reference = _reference_evidence(table, thresholds)
    entry, attack_domains, persistence_domains, entry_score = _entry_evidence(
        table, thresholds
    )
    preparation, prep_domains = _preparation_evidence(table, thresholds, vocal)
    directed_a, directed_supported, directed_domains = _directed_a_evidence(
        table, thresholds
    )
    arrival = _arrival_evidence(entry, directed_a, directed_supported)
    local_entry = _local_entry_evidence(entry)

    # B-before -> B-after sustained motion: the same direction survives the
    # entry without a salient local jump.
    sustained_matches, sustained_reversals, sustained_total = _signed_pairs(
        table, "b_stabilized_direction", thresholds,
        left_direction="b_before_direction",
    )
    sustained = (
        len(sustained_matches) >= thresholds.min_domains
        and len(sustained_reversals) < thresholds.min_domains
        and entry.status != "supported"
    )

    # Continuation and arrival are alternative functions. A salient reversal
    # rejects continuation only; it does not veto an independently supported
    # arrival. Conversely, a local B attack without directed A expectation is
    # not promoted to arrival/resolution.
    if arrival.status == "supported":
        label = "post_entry_attack_stabilized"
        abstained = False
    elif entry.status == "supported":
        # Sustained local attack, but A supplied no projected expectation.
        label = "unknown"
        abstained = True
    elif entry.status == "weak" and entry.score is not None and entry.score < 0:
        if preparation.status == "supported":
            label = "preparation_not_realized"
        else:
            label = "post_entry_attack_transient"
        abstained = False
    elif preparation.status == "supported":
        label = "pickup_or_preparation"
        abstained = False
    elif sustained:
        label = "sustained_motion"
        abstained = False
    elif continuation.status == "supported":
        label = "continuation_without_local_event"
        abstained = False
    else:
        label = "unknown"
        abstained = True

    if label == "unknown" and continuation.status != "supported":
        missing.append("supported_continuation_or_local_event")
    if contradiction:
        missing.append("salient_domain_reversal")
    if entry.status != "supported":
        missing.append("stabilized_post_entry_event")
    if not directed_supported:
        missing.append("directed_a_expectation")
    if entry.status == "supported" and arrival.status != "supported":
        missing.append("unprojected_local_entry_not_arrival")
    if section.status != "supported":
        missing.append("functional_section_context")
    if vocal.completion != "supported":
        missing.append("independent_phrase_completion")
    if vocal.pickup != "supported":
        missing.append("pre_entry_pickup")

    evidence = (continuation, reference, entry, preparation, directed_a, arrival, local_entry)
    contributions: dict[str, float] = {}
    for item in evidence:
        for key, value in item.contributions.items():
            contributions[f"{item.kind}.{key}"] = float(value)
    score, score_contributions = _score_label(
        label,
        continuation,
        entry,
        persistence_domains,
        prep_domains,
        vocal,
        section,
        thresholds,
    )
    contributions.update(score_contributions)
    if abstained:
        score = None

    labels_seen = sorted(
        {
            str(value)
            for value in (section.before_label, section.after_label)
            if value is not None
        }
    )
    diagnostics = {
        "version": VERSION,
        "transfer_status": TRANSFER_STATUS,
        "confidence": None,
        "score_semantics": "unvalidated bounded engineering adjustment; not probability",
        "thresholds": {
            "version": thresholds.version,
            "trend_threshold": thresholds.trend_threshold,
            "flat_threshold": thresholds.flat_threshold,
            "event_threshold": thresholds.event_threshold,
            "persistence_retention": thresholds.persistence_retention,
            "min_domains": thresholds.min_domains,
            "agreement_floor": thresholds.agreement_floor,
            "contradiction_weight": thresholds.contradiction_weight,
        },
        "attack_domains": attack_domains,
        "persistence_domains": persistence_domains,
        "preparation_domains": prep_domains,
        "sustained_matching_domains": [item[0] for item in sustained_matches],
        "sustained_reversing_domains": [item[0] for item in sustained_reversals],
        "salient_contradiction": contradiction,
        "directed_a_domains": directed_domains,
        "directed_a_status": directed_a.status,
        "arrival_status": arrival.status,
        "arrival_supported": arrival.status == "supported",
        "continuation_arrival_conflict": bool(
            contradiction and arrival.status == "supported"
        ),
        "continuation_reversal_vetoes_arrival": False,
        "local_entry_event_status": local_entry.status,
        "local_entry_is_cross_song_resolution": False,
        "trained_resolution_reward_used": False,
        "routes": {
            "compatible_continued_motion": continuation.status,
            "directed_arrival": arrival.status,
            "local_entry_only": local_entry.status,
            "preparation_not_realized_preserved": True,
        },
        "section_labels_observed": labels_seen,
        "section_label_mapped_to_resolution": False,
        "chorus_is_resolution": False,
        "track_identity_reward_used": False,
        "waveform_identity_used": False,
        "vocal_observation": {
            "pickup": vocal.pickup,
            "completion": vocal.completion,
            "word_active_at_cue": vocal.word_active_at_cue,
            "word_gap_is_not_completion": True,
            "word_active_blocks_completion": vocal.word_active_at_cue is True,
            "details": dict(vocal.details),
        },
        "section_observation": {
            "status": section.status,
            "changed": section.changed,
            "provider": section.provider,
            "before_label": section.before_label,
            "after_label": section.after_label,
            "label_semantics": "only boundary/change context; no label-to-resolution map",
        },
        "reference_evidence": reference.status,
        "profiles": {
            name: {
                "window_beats": (record or {}).get("window_beats"),
                "points": len(points),
                "side": (
                    profile.get("side")
                    if isinstance(profile, Mapping)
                    else None
                ),
                "boundary_method": (record or {}).get("boundary_method"),
            }
            for name, (record, points), profile in (
                ("A_before", selected["A_before"], A_before),
                ("A_after", selected["A_after"], A_after),
                ("B_before", selected["B_before"], B_before),
                ("B_after", selected["B_after"], B_after),
            )
        },
    }

    return FunctionalEventAssessment(
        label,
        abstained,
        evidence,
        table,
        contributions,
        tuple(dict.fromkeys(missing)),
        score,
        None,
        diagnostics,
    )


def rerank_functional_event(
    assessment: FunctionalEventAssessment,
    baseline_score: float | None,
    *,
    weight: float = 0.20,
) -> FunctionalEventRerank:
    """Optionally add the bounded event adjustment while retaining baseline."""

    baseline = _finite(baseline_score)
    weight_number = _finite(weight)
    if weight_number is None:
        raise ValueError("weight must be finite")
    event_score = assessment.score
    adjusted = None
    if baseline is not None:
        adjusted = baseline + weight_number * (event_score or 0.0)
    return FunctionalEventRerank(
        baseline,
        event_score,
        adjusted,
        weight_number,
        bool(weight_number and not assessment.abstained),
        assessment.label,
        assessment.abstained,
        "baseline retained; event adjustment is unvalidated and abstains on conflict",
    )


class FunctionalEventScorer:
    """Root-facing scorer/reranker over actual pool profiles.

    The four profiles may be supplied either as keyword arguments or as a
    mapping with keys ``A_before``, ``A_after``, ``B_before``, ``B_after``.
    ``rerank`` preserves the caller's baseline; ``score`` returns the project
    ``ScoreReport`` contract; ``rerank_pool`` maps a pool's profile lookup to
    non-mutating rows that a root ranker can sort by ``event_adjusted_score``.
    """

    def __init__(
        self,
        thresholds: FunctionalEventThresholds | None = None,
        *,
        weight: float = 0.20,
    ) -> None:
        self.thresholds = thresholds or FunctionalEventThresholds()
        self.weight = float(weight)
        if not math.isfinite(self.weight):
            raise ValueError("weight must be finite")

    @staticmethod
    def _profile(mapping: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
        direct = mapping.get(name)
        if isinstance(direct, Mapping):
            return direct
        for key, value in mapping.items():
            if str(key).lower() == name.lower() and isinstance(value, Mapping):
                return value
        return None

    def assess(
        self,
        A_before: Mapping[str, Any] | None = None,
        A_after: Mapping[str, Any] | None = None,
        B_before: Mapping[str, Any] | None = None,
        B_after: Mapping[str, Any] | None = None,
        *,
        profiles: Mapping[str, Any] | None = None,
        vocal_completion: Any = None,
        section_context: Any = None,
    ) -> FunctionalEventAssessment:
        if profiles is not None:
            A_before = A_before or self._profile(profiles, "A_before")
            A_after = A_after or self._profile(profiles, "A_after")
            B_before = B_before or self._profile(profiles, "B_before")
            B_after = B_after or self._profile(profiles, "B_after")
        return assess_functional_event(
            A_before,
            A_after,
            B_before,
            B_after,
            vocal_completion=vocal_completion,
            section_context=section_context,
            thresholds=self.thresholds,
        )

    def rerank(
        self,
        baseline_score: float | None,
        *,
        profiles: Mapping[str, Any] | None = None,
        A_before: Mapping[str, Any] | None = None,
        A_after: Mapping[str, Any] | None = None,
        B_before: Mapping[str, Any] | None = None,
        B_after: Mapping[str, Any] | None = None,
        vocal_completion: Any = None,
        section_context: Any = None,
        weight: float | None = None,
    ) -> tuple[FunctionalEventRerank, FunctionalEventAssessment]:
        assessment = self.assess(
            A_before,
            A_after,
            B_before,
            B_after,
            profiles=profiles,
            vocal_completion=vocal_completion,
            section_context=section_context,
        )
        result = rerank_functional_event(
            assessment,
            baseline_score,
            weight=self.weight if weight is None else weight,
        )
        return result, assessment

    def score(
        self,
        candidate,
        profiles: Mapping[str, Any],
        *,
        baseline_score: float | None = None,
        vocal_completion: Any = None,
        section_context: Any = None,
        weight: float | None = None,
    ):
        from .contracts import ScoreReport

        if baseline_score is None:
            feature = candidate.features.get("baseline_score") if candidate.features else None
            baseline_score = None if feature is None else feature.value
        result, assessment = self.rerank(
            baseline_score,
            profiles=profiles,
            vocal_completion=vocal_completion,
            section_context=section_context,
            weight=weight,
        )
        return ScoreReport(
            candidate.candidate_id,
            bool(candidate.passes_constraints),
            result.adjusted_score,
            dict(assessment.contributions),
            {},
            assessment.missing,
            (
                f"functional_event_label={result.label}",
                f"abstained={result.abstained}",
                f"baseline_score={result.baseline_score}",
                f"event_score={result.event_score}",
                "transfer=unvalidated",
            ),
        )

    def rerank_pool(
        self,
        rows: Iterable[Mapping[str, Any]],
        profile_lookup: Mapping[str, Mapping[str, Any]] | Any,
        *,
        baseline_key: str = "baseline_score",
        candidate_key: str = "candidate_id",
        profile_keys: Sequence[str] = ("A_before", "A_after", "B_before", "B_after"),
        vocal_completion: Any = None,
        section_context: Any = None,
        weight: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return rows with baseline and event-adjusted scores, unmutated.

        ``profile_lookup`` may be a mapping from candidate id to a profile
        mapping, or a callable ``candidate_id -> profile mapping``.
        """

        output: list[dict[str, Any]] = []
        for row in rows:
            candidate_id = row.get(candidate_key)
            if callable(profile_lookup):
                profiles = profile_lookup(candidate_id)
            elif isinstance(profile_lookup, Mapping):
                profiles = profile_lookup.get(candidate_id)
                if profiles is None and isinstance(row.get("profiles"), Mapping):
                    profiles = row["profiles"]
            else:
                profiles = None
            payload = dict(row)
            if not isinstance(profiles, Mapping):
                payload.update(
                    {
                        "event_baseline_score": _finite(row.get(baseline_key)),
                        "event_adjusted_score": _finite(row.get(baseline_key)),
                        "event_score": None,
                        "event_label": "unknown",
                        "event_abstained": True,
                        "event_missing": ("profiles",),
                    }
                )
                output.append(payload)
                continue
            selected = {name: self._profile(profiles, name) for name in profile_keys}
            result, assessment = self.rerank(
                _finite(row.get(baseline_key)),
                profiles=selected,
                vocal_completion=vocal_completion,
                section_context=section_context,
                weight=weight,
            )
            payload.update(
                {
                    "event_baseline_score": result.baseline_score,
                    "event_adjusted_score": result.adjusted_score,
                    "event_score": result.event_score,
                    "event_label": result.label,
                    "event_abstained": result.abstained,
                    "event_weight": result.weight,
                    "event_missing": assessment.missing,
                    "event_evidence": {
                        item.kind: item.status for item in assessment.evidence
                    },
                }
            )
            output.append(payload)
        return output


__all__ = [
    "DOMAIN_NAMES",
    "LABELS",
    "TRANSFER_STATUS",
    "VERSION",
    "DomainEventEvidence",
    "FunctionalEventAssessment",
    "FunctionalEventRerank",
    "FunctionalEventScorer",
    "FunctionalEventThresholds",
    "SectionObservation",
    "VocalObservation",
    "assess_functional_event",
    "rerank_functional_event",
]
