"""Provider-neutral, auditable expectation evidence for the V7 experiment.

This module consumes JSON-ready profiles from :mod:`musicmix.trajectory`.  It
describes observable multiscale motion and keeps the semantic boundary clear:
an RMS fall, chroma similarity, or a quiet intro cannot by itself establish a
resolution.  The output is therefore an uncalibrated evidence report, not a
probability and not a replacement for a listening judgement.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import fmean, median, pstdev
from typing import Any, Mapping, Sequence


EXPECTATION_VERSION = "expectation-v11-gain-invariant-log-motion"
THRESHOLDS_VERSION = "hypothesis-v1"
EXPECTATION_LABELS = frozenset({"continuation", "closure_candidate", "interruption", "unknown"})
DOMAIN_NAMES = ("energy", "events", "bass", "timbre", "tonal_proxy")


@dataclass(frozen=True)
class ExpectationThresholds:
    """Versioned hypotheses, not calibrated probabilities.

    Values are deliberately exposed in the result diagnostics so a future
    listening study can replace them without silently changing old reports.
    ``trend_threshold`` and ``flat_threshold`` operate on bounded direction
    scores.  Arrival requires a beat-aligned onset, one entry domain and one
    independent contextual-stability domain.
    """

    version: str = THRESHOLDS_VERSION
    trend_threshold: float = 0.25
    flat_threshold: float = 0.12
    arrival_threshold: float = 0.35
    stability_threshold: float = 0.62
    metric_alignment_fraction: float = 0.25
    min_trend_domains: int = 2

    def __post_init__(self) -> None:
        for name in (
            "trend_threshold",
            "flat_threshold",
            "arrival_threshold",
            "stability_threshold",
            "metric_alignment_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if int(self.min_trend_domains) < 2:
            raise ValueError("min_trend_domains must be at least two")
        object.__setattr__(self, "min_trend_domains", int(self.min_trend_domains))


@dataclass(frozen=True)
class Evidence:
    """One separately inspectable evidence family.

    ``score`` is a bounded, descriptive margin.  It is never a confidence or
    a likelihood.  Missing fields are retained instead of being imputed.
    """

    kind: str
    status: str
    score: float | None
    contributions: dict[str, float]
    missing: tuple[str, ...] = ()
    diagnostics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status not in {"supported", "weak", "absent", "unknown"}:
            raise ValueError("invalid evidence status")
        score = self.score
        if score is not None:
            score = float(score)
            if not math.isfinite(score):
                raise ValueError("evidence score must be finite or null")
            object.__setattr__(self, "score", max(-1.0, min(1.0, score)))
        object.__setattr__(self, "contributions", {
            str(key): float(value) for key, value in (self.contributions or {}).items()
            if math.isfinite(float(value))
        })
        object.__setattr__(self, "missing", tuple(dict.fromkeys(str(item) for item in self.missing)))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics or {}))

    @property
    def name(self) -> str:
        """Alias used by simple audit consumers."""

        return self.kind


@dataclass(frozen=True)
class ExpectationAssessment:
    """Result for ``A_before -> B_after`` and its optional natural reference."""

    label: str
    evidence: tuple[Evidence, ...]
    contributions: dict[str, float]
    missing: tuple[str, ...]
    confidence: None = None
    score: float | None = None
    baseline_score: float | None = None
    diagnostics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.label not in EXPECTATION_LABELS:
            raise ValueError(f"label must be one of {sorted(EXPECTATION_LABELS)}")
        if self.confidence is not None:
            raise ValueError("expectation confidence is intentionally uncalibrated")
        evidence = tuple(self.evidence)
        if any(not isinstance(item, Evidence) for item in evidence):
            raise TypeError("evidence must contain Evidence objects")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "contributions", {
            str(key): float(value) for key, value in (self.contributions or {}).items()
        })
        object.__setattr__(self, "missing", tuple(dict.fromkeys(str(item) for item in self.missing)))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics or {}))

    @property
    def evidence_by_kind(self) -> dict[str, Evidence]:
        return {item.kind: item for item in self.evidence}

    @property
    def classification(self) -> str:
        """Readable alias for callers that use classification terminology."""

        return self.label

    @property
    def continuation_evidence(self) -> Evidence | None:
        return self.evidence_by_kind.get("continuation")

    @property
    def closure_evidence(self) -> Evidence | None:
        return self.evidence_by_kind.get("closure_arrival")

    @property
    def interruption_evidence(self) -> Evidence | None:
        return self.evidence_by_kind.get("interruption")


@dataclass(frozen=True)
class RerankResult:
    """Optional raw-score adjustment, retaining the caller's baseline."""

    baseline_score: float | None
    expectation_score: float | None
    adjusted_score: float | None
    weight: float
    enabled: bool
    rationale: str


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bounded(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _profile_mapping(profile: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    return profile if isinstance(profile, Mapping) else None


def _window_records(profile: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Normalize trajectory ``windows`` and the small series-per-beat variant."""

    if profile is None:
        return []
    raw_windows = profile.get("windows", ())
    records: list[dict[str, Any]] = []
    if isinstance(raw_windows, Mapping):
        iterable = raw_windows.items()
        for key, value in iterable:
            if isinstance(value, Mapping):
                record = dict(value)
                record.setdefault("window_beats", key)
                records.append(record)
    elif isinstance(raw_windows, Sequence) and not isinstance(raw_windows, (str, bytes)):
        records.extend(dict(item) for item in raw_windows if isinstance(item, Mapping))
    if records:
        return records
    raw_series = profile.get("series_per_beat")
    if isinstance(raw_series, Mapping):
        for key, series in raw_series.items():
            if isinstance(series, Sequence) and not isinstance(series, (str, bytes)):
                records.append({"window_beats": key, "beat_feature_series": list(series)})
    return records


def _series(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = record.get("beat_feature_series", record.get("series", ()))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [dict(point) for point in raw if isinstance(point, Mapping)]


def _direction(values: Sequence[Any], *, origin_invariant: bool = False) -> float | None:
    numbers = [number for number in (_finite(value) for value in values) if number is not None]
    if len(numbers) < 2:
        return None
    count = max(1, len(numbers) // 4)
    delta = fmean(numbers[-count:]) - fmean(numbers[:count])
    # A constant audio gain adds an offset to log-RMS. Its absolute median
    # therefore cannot set the scale of a trajectory direction. Other linear
    # descriptors retain the relative-level floor used by the original policy.
    level_floor = 0.0 if origin_invariant else abs(median(numbers)) * 0.2
    scale = max(pstdev(numbers), level_floor, 1e-6)
    return _bounded(delta / (2.0 * scale))


_DOMAIN_FIELDS: dict[str, tuple[str, ...]] = {
    "energy": ("log_rms_mean", "rms_mean"),
    "events": ("spectral_flux_mean", "attack_density_per_second"),
    "bass": ("bass_ratio", "bass_ratio_end"),
    "timbre": ("high_frequency_ratio", "spectral_centroid_hz"),
    "tonal_proxy": ("chroma_movement",),
}


def _domain_scores(record: Mapping[str, Any]) -> dict[str, float | None]:
    points = _series(record)
    result: dict[str, float | None] = {}
    for domain, fields in _DOMAIN_FIELDS.items():
        scores: list[float] = []
        for field in fields:
            values = [point.get(field) for point in points]
            score = _direction(values, origin_invariant=field == "log_rms_mean")
            if score is not None:
                scores.append(score)
        result[domain] = fmean(scores) if scores else None
    return result


def _direction_label(score: float | None, thresholds: ExpectationThresholds) -> str:
    if score is None:
        return "unknown"
    if score >= thresholds.trend_threshold:
        return "up"
    if score <= -thresholds.trend_threshold:
        return "down"
    if abs(score) <= thresholds.flat_threshold:
        return "flat"
    return "weak"


def _summarize(profile: Mapping[str, Any] | None, thresholds: ExpectationThresholds) -> dict[str, Any]:
    profile = _profile_mapping(profile)
    records = _window_records(profile)
    by_domain: dict[str, list[float]] = {domain: [] for domain in DOMAIN_NAMES}
    by_window: dict[str, dict[str, float | None]] = {}
    for record in records:
        try:
            window = int(record.get("window_beats"))
        except (TypeError, ValueError):
            continue
        scores = _domain_scores(record)
        by_window[str(window)] = scores
        for domain, score in scores.items():
            if score is not None:
                by_domain[domain].append(score)
    domains: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for domain in DOMAIN_NAMES:
        values = by_domain[domain]
        score = fmean(values) if values else None
        if score is None:
            missing.append(f"{domain}_trajectory")
        domains[domain] = {
            "score": score,
            "direction": _direction_label(score, thresholds),
            "window_scores": values,
        }
    if not records:
        missing.append("trajectory_windows")
    if records and not any(_series(record) for record in records):
        missing.append("beat_feature_series")
    return {
        "available": profile is not None,
        "domains": domains,
        "windows": by_window,
        "missing": tuple(dict.fromkeys(missing)),
        "provenance": _provenance(profile),
    }


def _provenance(profile: Mapping[str, Any] | None) -> dict[str, Any]:
    if profile is None:
        return {"available": False}
    keys = (
        "track_id", "sample_rate", "canonical_sample_rate", "duration_samples",
        "requested_cue_sample", "cue_sample", "cue_shift_samples",
        "beat_anchor_sample", "beat_period_samples", "side", "backend",
        "trajectory_config_hash", "local_score", "score", "source",
        "candidate_id", "original_cue_sample", "original_structure_cue_sample",
        "structure_cue_sample", "original_downbeat_sample", "downbeat_sample",
        "downbeat_confidence", "shifted_cue_sample", "cue_policy",
    )
    result = {key: profile[key] for key in keys if key in profile}
    for key in ("metadata", "source_metadata", "alignment"):
        value = profile.get(key)
        if isinstance(value, Mapping):
            result[key] = dict(value)
    result["available"] = True
    return result


def _trend_support(summary: Mapping[str, Any], thresholds: ExpectationThresholds) -> tuple[list[str], list[str]]:
    up: list[str] = []
    down: list[str] = []
    for domain, item in summary.get("domains", {}).items():
        direction = item.get("direction")
        if direction == "up":
            up.append(domain)
        elif direction == "down":
            down.append(domain)
    return up, down


def _metric_evidence(profile: Mapping[str, Any] | None, thresholds: ExpectationThresholds) -> tuple[str, float | None, tuple[str, ...], dict[str, Any]]:
    profile = _profile_mapping(profile)
    if profile is None:
        return "unknown", None, ("B_after", "metric_anchor"), {}
    cue = _finite(profile.get("cue_sample"))
    anchor = _finite(profile.get("beat_anchor_sample"))
    period = _finite(profile.get("beat_period_samples"))
    if cue is None or anchor is None or period is None or period <= 0:
        return "unknown", None, ("metric_anchor_or_period",), {"available": False}
    ratio = abs(cue - anchor) / period
    score = _bounded(1.0 - ratio)
    status = "supported" if ratio <= thresholds.metric_alignment_fraction else "absent"
    return status, score, (), {"anchor_delta_fraction": ratio, "period_samples": period}


def _structural_arrival(profile: Mapping[str, Any] | None) -> tuple[str, tuple[str, ...], dict[str, Any]]:
    """Read an explicit phrase/structure provider without inferring one.

    Beat alignment, chroma stability and FFT events are intentionally not
    accepted as a phrase boundary.  Providers may use any of these keys, but
    must state ``supported: true`` (or ``status: supported``) and identify a
    non-empty ``provider``/``source``.
    """

    profile = _profile_mapping(profile)
    if profile is None:
        return "unknown", ("phrase_or_structural_arrival",), {}
    keys = (
        "phrase_or_structural_arrival",
        "structural_arrival",
        "phrase_boundary",
        "phrase_structure",
    )
    for key in keys:
        value = profile.get(key)
        if value is None:
            continue
        candidate: Any = value
        if key == "phrase_structure" and isinstance(value, Mapping) and isinstance(value.get("arrival"), Mapping):
            candidate = value["arrival"]
        if candidate is True:
            return "unknown", ("phrase_or_structural_arrival_provider",), {"field": key, "provider": None}
        if not isinstance(candidate, Mapping):
            return "unknown", ("phrase_or_structural_arrival_provider",), {"field": key}
        provider = candidate.get("provider") or candidate.get("source") or candidate.get("backend")
        supported = candidate.get("supported") is True or candidate.get("status") == "supported"
        details = {"field": key, "provider": provider, "supported": bool(supported)}
        if not isinstance(provider, str) or not provider.strip():
            return "unknown", ("phrase_or_structural_arrival_provider",), details
        return ("supported" if supported else "absent"), (), details
    return "unknown", ("phrase_or_structural_arrival",), {"field": None, "provider": None}


def _arrival_evidence(profile: Mapping[str, Any] | None, thresholds: ExpectationThresholds) -> Evidence:
    profile = _profile_mapping(profile)
    records = _window_records(profile)
    if not records:
        return Evidence("closure_arrival", "unknown", None, {}, ("B_after_windows",))
    record = min(records, key=lambda item: int(item.get("window_beats", 10**9)))
    points = _series(record)
    if len(points) < 3:
        return Evidence("closure_arrival", "unknown", None, {}, ("B_after_beat_feature_series",))
    head_count = max(1, min(2, len(points) // 4))
    tail = points[head_count:]
    if not tail:
        return Evidence("closure_arrival", "unknown", None, {}, ("B_after_post_entry_context",))

    def entry(field: str) -> float | None:
        first = [_finite(point.get(field)) for point in points[:head_count]]
        rest = [_finite(point.get(field)) for point in tail]
        first = [value for value in first if value is not None]
        rest = [value for value in rest if value is not None]
        if not first or not rest:
            return None
        delta = fmean(first) - fmean(rest)
        scale = max(abs(median(rest)) * 0.2, pstdev(rest), 1e-6)
        return _bounded(delta / (2.0 * scale))

    bass = entry("bass_ratio")
    flux = entry("spectral_flux_mean")
    density = entry("attack_density_per_second")
    event_values = [value for value in (flux, density) if value is not None]
    contributions: dict[str, float] = {}
    if bass is not None:
        contributions["bass_entry"] = bass
    if event_values:
        contributions["event_entry"] = max(event_values)
    event_domains = [key for key, value in contributions.items()
                     if value >= thresholds.arrival_threshold]

    metric_status, metric_score, metric_missing, metric_details = _metric_evidence(profile, thresholds)
    if metric_score is not None:
        contributions["metric_alignment"] = metric_score

    stability_values: list[float] = []
    for key in ("chroma_local_context_similarity", "chroma_tonal_stability_proxy"):
        value = _finite(record.get(key))
        if value is not None:
            stability_values.append(value)
    if stability_values:
        # The two trajectory-v1 fields are alternate views of the same
        # chroma-context proxy.  Averaging avoids letting a duplicate field
        # inflate arrival evidence and does not claim independent validation.
        stability = fmean(stability_values)
        stability_status = "supported" if stability >= thresholds.stability_threshold else "absent"
        stability_source = "trajectory_context_proxy_correlated"
    else:
        # A stable post-entry trajectory is useful contextual evidence, but it
        # is deliberately not named tonal closure.
        post_domains = _domain_scores({"beat_feature_series": tail})
        known = [value for value in post_domains.values() if value is not None]
        stable_count = sum(abs(value) <= thresholds.flat_threshold for value in known)
        stability = (stable_count / len(known)) if known else None
        # Flatness of several post-entry trajectory domains is a correlated
        # fallback, useful for diagnostics but insufficient to claim a second
        # independent arrival domain.
        stability_status = (
            "weak" if stability is not None and stable_count >= thresholds.min_trend_domains
            else "absent" if known else "unknown"
        )
        stability_source = "post_entry_correlated_fallback"
    if stability is not None:
        contributions["contextual_stability"] = stability

    structural_status, structural_missing, structural_details = _structural_arrival(profile)
    if structural_status == "supported":
        contributions["phrase_or_structural_arrival"] = 1.0

    missing = list(metric_missing)
    if metric_status == "unknown":
        missing.append("metric_alignment")
    if stability_status == "unknown":
        missing.append("contextual_stability")
    if stability_status == "weak":
        missing.append("independent_contextual_stability")
    missing.extend(structural_missing)
    # No tonal/cadence provider exists in trajectory-v1.  Keep this explicit
    # even when chroma similarity is present; it prevents a false cadence claim.
    diagnostics = _profile_mapping(profile).get("diagnostics", {}) if profile else {}
    tonal = _profile_mapping(profile).get("tonal_evidence") if profile else None
    cadence = _profile_mapping(profile).get("cadence_evidence") if profile else None
    if not tonal and not cadence and not isinstance(diagnostics, Mapping):
        missing.append("tonal_cadence_semantics")
    elif not tonal and not cadence:
        missing.append("tonal_cadence_semantics")

    domains = len(event_domains) + int(stability_status == "supported")
    if (
        metric_status == "supported"
        and event_domains
        and stability_status == "supported"
        and structural_status == "supported"
    ):
        status = "supported"
    elif (
        metric_status == "unknown"
        or stability_status == "unknown"
        or (stability_status == "weak" and event_domains)
        or structural_status == "unknown"
    ):
        status = "unknown"
    else:
        status = "absent"
    score = fmean(list(contributions.values())) if contributions else None
    return Evidence(
        "closure_arrival",
        status,
        score,
        contributions,
        tuple(dict.fromkeys(missing)),
        {
            "metric": metric_details,
            "structural_arrival": structural_details,
            "event_domains": event_domains,
            "independent_domain_count": domains,
            "stability_source": stability_source,
            "stability_independence": "not_calibrated; correlated trajectory proxy",
            "tonal_cadence_evidence_present": bool(tonal or cadence),
        },
    )


def analyze_expectation(
    A_before: Mapping[str, Any] | None,
    A_natural_after: Mapping[str, Any] | None,
    B_after: Mapping[str, Any] | None,
    *,
    thresholds: ExpectationThresholds | None = None,
) -> ExpectationAssessment:
    """Analyze directional motion without claiming calibrated semantics.

    ``A_natural_after`` is an optional reference trajectory.  Only its
    direction is compared; no waveform similarity or copy requirement is
    imposed.  Missing reference context is reported but does not block a
    continuation decision when A-before and B-after are both measurable.
    """

    thresholds = thresholds or ExpectationThresholds()
    a = _summarize(A_before, thresholds)
    natural = _summarize(A_natural_after, thresholds)
    b = _summarize(B_after, thresholds)
    missing: list[str] = []
    if not a["available"]:
        missing.append("A_before")
    if not b["available"]:
        missing.append("B_after")
    if not natural["available"]:
        missing.append("A_natural_after_reference")
    missing.extend(a["missing"] if a["available"] else ())
    missing.extend(b["missing"] if b["available"] else ())

    a_up, a_down = _trend_support(a, thresholds)
    b_up, b_down = _trend_support(b, thresholds)
    natural_up, natural_down = _trend_support(natural, thresholds)
    b_flat = [domain for domain, item in b.get("domains", {}).items()
              if item.get("direction") == "flat"]
    b_known = len(b_up) + len(b_down) + len(b_flat)
    a_supported = len(a_up) >= thresholds.min_trend_domains or len(a_down) >= thresholds.min_trend_domains
    a_is_up = len(a_up) >= thresholds.min_trend_domains and len(a_up) >= len(a_down)
    b_is_up = len(b_up) >= thresholds.min_trend_domains
    b_is_flat = len(b_flat) >= thresholds.min_trend_domains and not b_up and not b_down
    b_is_divergent = len(b_down) >= thresholds.min_trend_domains if a_is_up else len(b_up) >= thresholds.min_trend_domains

    continuation_contributions: dict[str, float] = {}
    matching: list[str] = []
    if a_supported and b_known:
        a_domains = a["domains"]
        b_domains = b["domains"]
        for domain in DOMAIN_NAMES:
            left = a_domains[domain].get("score")
            right = b_domains[domain].get("score")
            if (
                left is not None and right is not None
                and abs(left) >= thresholds.trend_threshold
                and abs(right) >= thresholds.trend_threshold
                and left * right > 0
            ):
                matching.append(domain)
                continuation_contributions[domain] = _bounded(min(abs(left), abs(right)))
    reference_matching: list[str] = []
    if natural["available"]:
        for domain in DOMAIN_NAMES:
            natural_score = natural["domains"][domain].get("score")
            b_score = b["domains"][domain].get("score")
            if natural_score is not None and b_score is not None and natural_score * b_score > 0:
                reference_matching.append(domain)
        if reference_matching:
            continuation_contributions["natural_reference_direction"] = len(reference_matching) / len(DOMAIN_NAMES)
    if len(matching) >= thresholds.min_trend_domains:
        continuation = Evidence(
            "continuation",
            "supported",
            fmean(continuation_contributions.values()),
            continuation_contributions,
            (),
            {"matching_domains": matching, "natural_reference_matching": reference_matching},
        )
    elif not a_supported or b_known < thresholds.min_trend_domains:
        continuation = Evidence(
            "continuation", "unknown", None, continuation_contributions,
            tuple(missing) + ("multifeature_direction",),
            {"matching_domains": matching},
        )
    else:
        continuation = Evidence(
            "continuation", "absent", None, continuation_contributions,
            (), {"matching_domains": matching},
        )

    closure = _arrival_evidence(B_after, thresholds)
    interruption_missing: list[str] = []
    if closure.status == "unknown":
        interruption_missing.extend(closure.missing)
    interruption_supported = (
        a_is_up and b_known >= thresholds.min_trend_domains
        and (b_is_flat or b_is_divergent) and closure.status == "absent"
    )
    interruption_contributions = {
        "A_before_up_domains": float(len(a_up)),
        "B_after_nonmatching_domains": float(len(b_down) if a_is_up else len(b_up)),
    }
    if interruption_supported:
        interruption = Evidence(
            "interruption", "supported", _bounded((len(a_up) + b_known) / (2 * len(DOMAIN_NAMES))),
            interruption_contributions, (),
            {"A_up_domains": a_up, "B_up_domains": b_up, "B_down_domains": b_down},
        )
    elif closure.status == "unknown" or not a_supported or b_known < thresholds.min_trend_domains:
        interruption = Evidence(
            "interruption", "unknown", None, interruption_contributions,
            tuple(interruption_missing) + ("arrival_or_direction",),
            {"A_up_domains": a_up, "B_up_domains": b_up, "B_down_domains": b_down},
        )
    else:
        interruption = Evidence(
            "interruption", "absent", None, interruption_contributions,
            (), {"A_up_domains": a_up, "B_up_domains": b_up, "B_down_domains": b_down},
        )

    evidence = (continuation, closure, interruption)
    supported_kinds = [item.kind for item in evidence if item.status == "supported"]
    # Continuation and interruption are mutually exclusive interpretations.
    # Closure may be the arrival of a continuing arc, but closure plus
    # interruption is still a conservative conflict.
    conflict = (
        (continuation.status == "supported" and interruption.status == "supported")
        or (closure.status == "supported" and interruption.status == "supported")
    )
    closure_eligible = closure.status == "supported" and a_supported
    if closure.status == "supported" and not a_supported:
        missing.append("A_before_multifeature_direction")
    if conflict:
        label = "unknown"
    elif closure_eligible:
        label = "closure_candidate"
    elif continuation.status == "supported":
        label = "continuation"
    elif interruption.status == "supported":
        label = "interruption"
    else:
        label = "unknown"
    contributions: dict[str, float] = {}
    for item in evidence:
        for key, value in item.contributions.items():
            contributions[f"{item.kind}.{key}"] = value
    signed_scores: list[float] = []
    for item in evidence:
        if item.status != "supported" or item.score is None:
            continue
        signed_scores.append(-abs(item.score) if item.kind == "interruption" else abs(item.score))
    # Unknown/conflicting classifications are neutral for any optional
    # reranker; do not let two cancelling raw evidences masquerade as a score.
    raw_score = fmean(signed_scores) if signed_scores and label != "unknown" else None
    diagnostics = {
        "expectation_version": EXPECTATION_VERSION,
        "thresholds_version": thresholds.version,
        "thresholds": {
            "trend_threshold": thresholds.trend_threshold,
            "flat_threshold": thresholds.flat_threshold,
            "arrival_threshold": thresholds.arrival_threshold,
            "stability_threshold": thresholds.stability_threshold,
            "metric_alignment_fraction": thresholds.metric_alignment_fraction,
            "min_trend_domains": thresholds.min_trend_domains,
        },
        "confidence": None,
        "score_semantics": "uncalibrated descriptive evidence margin, not probability",
        "supported_evidence": supported_kinds,
        "evidence_conflict": conflict,
        "A_before_supported": a_supported,
        "profiles": {"A_before": a, "A_natural_after": natural, "B_after": b},
    }
    return ExpectationAssessment(
        label,
        evidence,
        contributions,
        tuple(dict.fromkeys(
            missing + list(closure.missing) + list(interruption_missing)
            + (["conflicting_supported_evidence"] if conflict else [])
        )),
        None,
        raw_score,
        None,
        diagnostics,
    )


def rerank_expectation(
    assessment: ExpectationAssessment,
    baseline_score: float | None,
    *,
    weight: float = 0.0,
) -> RerankResult:
    """Optionally add an explicitly weighted raw evidence margin.

    The default weight is zero, so this function cannot silently alter the
    existing ranker.  ``baseline_score`` is always returned for audit.
    """

    baseline = _finite(baseline_score)
    weight_number = _finite(weight)
    if weight_number is None:
        raise ValueError("weight must be finite")
    evidence_score = assessment.score
    adjusted = None
    if baseline is not None:
        adjusted = baseline + weight_number * (evidence_score or 0.0)
    return RerankResult(
        baseline,
        evidence_score,
        adjusted,
        weight_number,
        bool(weight_number),
        "optional uncalibrated evidence adjustment; baseline retained",
    )


# Descriptive aliases make the experiment easy to call from audit notebooks.
evaluate_expectation = analyze_expectation
assess_expectation = analyze_expectation


__all__ = [
    "DOMAIN_NAMES",
    "EXPECTATION_LABELS",
    "EXPECTATION_VERSION",
    "Evidence",
    "ExpectationAssessment",
    "ExpectationThresholds",
    "RerankResult",
    "THRESHOLDS_VERSION",
    "analyze_expectation",
    "assess_expectation",
    "evaluate_expectation",
    "rerank_expectation",
]
