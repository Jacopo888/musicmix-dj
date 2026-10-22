"""Match pretrained single-track opportunities to an existing candidate pool.

This is coordinate/selection glue, not a learned model. It preserves the
baseline pair score and treats missing model support as unknown. A supported
cue is not a certificate of musical resolution or vocal boundary accuracy.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter
import math
from typing import Any, Mapping, Sequence


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


def validate_opportunities(payload: Mapping[str, Any], track: Mapping[str, Any]) -> list[dict]:
    """Check canonical identity; never silently move an MP3/source timeline."""
    for field in ("track_id", "sample_rate", "duration_samples"):
        if payload.get(field) != track[field]:
            raise ValueError(f"opportunity {field} does not match canonical track")
    if payload.get("status") != "complete":
        raise ValueError("model inference must be complete; unavailable is not empty success")
    rate = _finite(track["sample_rate"], "sample_rate")
    duration = int(track["duration_samples"])
    if rate <= 0 or duration <= 0:
        raise ValueError("positive canonical duration/rate required")
    cues = payload.get("cues")
    if not isinstance(cues, list):
        raise ValueError("cues must be a list")
    result = []
    samples = set()
    for cue in cues:
        seconds = _finite(cue["time_seconds"], "cue time")
        sample = round(seconds * rate)
        if not 0 <= sample < duration:
            raise ValueError("model cue outside canonical duration")
        if sample in samples:
            raise ValueError("duplicate model cue on canonical sample")
        samples.add(sample)
        result.append(dict(cue, sample=sample))
    return sorted(result, key=lambda cue: cue["sample"])


def nearest_support(sample: int, cues: Sequence[Mapping[str, Any]], tolerance_samples: int) -> dict | None:
    if tolerance_samples < 0:
        raise ValueError("tolerance must be nonnegative")
    if not cues:
        return None
    ordered = sorted(cues, key=lambda cue: cue["sample"])
    times = [cue["sample"] for cue in ordered]
    index = bisect_left(times, sample)
    choices = ordered[max(0, index - 1):min(len(ordered), index + 1)]
    nearest = min(choices, key=lambda cue: (abs(cue["sample"] - sample), cue["sample"]))
    delta = sample - nearest["sample"]
    if abs(delta) > tolerance_samples:
        return None
    return {"model_cue": dict(nearest), "actual_minus_model_samples": delta,
            "tolerance_samples": tolerance_samples, "calibrated_confidence": None}


def filter_supported_proposals(
    proposals: Sequence[Mapping[str, Any]],
    opportunities: Mapping[str, Sequence[Mapping[str, Any]]],
    tolerance_samples: Mapping[str, int],
) -> tuple[list[dict], dict]:
    """Retain only pairs supported at BOTH effective (actually rendered) cues."""
    supported = []
    counts = Counter()
    for proposal in proposals:
        c = proposal["candidate"]
        ids = (c["track_a"], c["track_b"])
        if any(identity not in opportunities for identity in ids):
            counts["missing_provider"] += 1
            continue
        a = nearest_support(c["cue_out_sample"], opportunities[ids[0]], tolerance_samples[ids[0]])
        b = nearest_support(c["cue_in_sample"], opportunities[ids[1]], tolerance_samples[ids[1]])
        counts["outgoing_supported"] += a is not None
        counts["incoming_supported"] += b is not None
        if a is None or b is None:
            counts["unsupported_at_one_or_both_cues"] += 1
            continue
        supported.append(dict(proposal, learned_opportunity_support={"outgoing": a, "incoming": b}))
        counts["both_supported"] += 1
    counts["input_proposals"] = len(proposals)
    return supported, dict(counts)


def select_diverse(proposals: Sequence[Mapping[str, Any]], *, maximum: int,
                   sample_rates: Mapping[str, int], spacing_seconds: float = 8.0,
                   role_cap: int = 2) -> list[dict]:
    """Preserve existing pair scores; avoid repeating regions across all roles."""
    if maximum < 0 or role_cap <= 0 or not math.isfinite(spacing_seconds) or spacing_seconds < 0:
        raise ValueError("invalid selection limits")
    selected: list[dict] = []
    used: dict[str, list[int]] = {}
    out_counts, in_counts = Counter(), Counter()
    seen = set()
    for p in sorted(proposals, key=lambda p: (-_finite(p["score"], "score"), p["candidate"]["candidate_id"])):
        if len(selected) >= maximum:
            break
        c = p["candidate"]
        a, b = c["track_a"], c["track_b"]
        geometry = (a, b, c["cue_out_sample"], c["cue_in_sample"])
        if geometry in seen or out_counts[a] >= role_cap or in_counts[b] >= role_cap:
            continue
        points = ((a, c["cue_out_sample"]), (b, c["cue_in_sample"]))
        if any(any(abs(point - old) < spacing_seconds * sample_rates[track] for old in used.get(track, []))
               for track, point in points):
            continue
        selected.append(dict(p)); seen.add(geometry)
        out_counts[a] += 1; in_counts[b] += 1
        for track, point in points:
            used.setdefault(track, []).append(point)
    return selected
