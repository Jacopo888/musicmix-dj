"""Directed candidate pools over every ordered pair of canonical tracks.

This module is deliberately a thin, auditable layer above
:func:`musicmix.directional.search_directional`.  It does not invent a second
ranker: every admitted proposal keeps the directional score, breakdown,
feature provenance, and vocal-boundary alignment produced by that search.
The only additional decisions are pool-level coverage and diversity:

* every ordered pair of distinct canonical tracks is searched;
* exact ``(track_a, track_b, cue_out, cue_in)`` geometries are never repeated;
* at most ``max_per_pair`` proposals survive per directed pair;
* each track may occur at most twice as a source and twice as an incoming
  track (the pool-level default), unless that makes the requested ``top_k``
  infeasible -- in which case the shortfall is reported, never hidden;
* cues on the same track must be at least eight seconds apart, even when the
  track appears in different roles or opposite directions;
* optional heard regions are checked with
  :func:`musicmix.audition_history.exposure` for both source regions.

Inputs are JSON-shaped evidence dictionaries, but their track identity, sample
rate, duration, and content hash must match the canonical :class:`AudioTrack`
exactly.  Missing or ambiguous evidence is rejected rather than repaired.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import math
import numbers
from typing import Any, Mapping, Sequence

from .audio import AudioTrack
from .audition_history import exposure
from .directional import search_directional


SCHEMA_VERSION = "musicmix.pool-search.1"
POOL_ALGORITHM = "pool-search-v1"
DEFAULT_ROLE_CAP = 2
DEFAULT_CUE_SPACING_SECONDS = 8.0

_HASH_KEYS = (
    "canonical_file_sha256",
    "source_sha256",
    "audio_sha256",
    "sha256",
)
_REQUIRED_HEARD_KEYS = (
    "track_a",
    "track_b",
    "cue_out_sample",
    "cue_in_sample",
    "sample_rate",
)


def _exact_integer(payload: Mapping[str, Any], key: str, label: str) -> int:
    if key not in payload:
        raise ValueError(f"{label}: missing {key}")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (numbers.Integral, float)):
        raise ValueError(f"{label}: {key} must be an integer")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise ValueError(f"{label}: {key} must be an exact integer")
    return int(value)


def _declared_hashes(payload: Mapping[str, Any], label: str) -> tuple[tuple[str, str], ...]:
    hashes: list[tuple[str, str]] = []
    for key in _HASH_KEYS:
        if key not in payload:
            continue
        value = payload[key]
        if value is None or str(value).strip() == "":
            raise ValueError(f"{label}: {key} is present but empty")
        hashes.append((key, str(value)))
    if hashes:
        return tuple(hashes)
    # ``structure.py`` persists the canonical content hash as ``track_id`` and
    # does not duplicate it under a separate hash key.  The caller already
    # checked that this identifier matches the canonical AudioTrack, so it is
    # an explicit content binding rather than an inferred one.
    track_id = payload.get("track_id")
    if track_id is None or str(track_id).strip() == "":
        raise ValueError(f"{label}: missing content hash")
    return (("track_id", str(track_id)),)


def _validate_evidence(
    track: AudioTrack,
    payload: Any,
    label: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} must be a mapping")
    declared_id = payload.get("track_id")
    if declared_id is None or str(declared_id) != track.track_id:
        raise ValueError(f"{label}: track_id does not match the canonical AudioTrack")
    sample_rate = _exact_integer(payload, "sample_rate", label)
    if sample_rate != track.sample_rate:
        raise ValueError(f"{label}: sample_rate does not match the canonical AudioTrack")
    duration = _exact_integer(payload, "duration_samples", label)
    if duration != track.duration_samples:
        raise ValueError(f"{label}: duration_samples does not match the canonical AudioTrack")
    hashes = _declared_hashes(payload, label)
    for key, digest in hashes:
        if digest != track.track_id:
            raise ValueError(f"{label}: {key} does not match the canonical content hash")
    return {
        "track_id": track.track_id,
        "sample_rate": track.sample_rate,
        "duration_samples": track.duration_samples,
        "hash_fields": [key for key, _ in hashes],
        "hash_verified": True,
    }


def _validate_key_set(name: str, values: Any, tracks: Mapping[str, AudioTrack]) -> None:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping keyed by canonical track ID")
    expected = set(tracks)
    actual = set(values)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"unknown={extra}")
        raise ValueError(f"{name} keys must match canonical track IDs exactly ({', '.join(details)})")


def _coerce_heard_regions(heard_regions: Any) -> list[dict[str, Any]] | None:
    if heard_regions is None:
        return None
    if isinstance(heard_regions, Mapping):
        if "regions" not in heard_regions:
            raise ValueError("heard_regions mapping must contain 'regions'")
        regions = heard_regions["regions"]
    else:
        regions = heard_regions
    if not isinstance(regions, (list, tuple)):
        raise TypeError("heard_regions must be a sequence or a mapping with 'regions'")
    result: list[dict[str, Any]] = []
    for index, region in enumerate(regions):
        if not isinstance(region, Mapping):
            raise TypeError(f"heard_regions[{index}] must be a mapping")
        missing = [key for key in _REQUIRED_HEARD_KEYS if key not in region]
        if missing:
            raise ValueError(f"heard_regions[{index}] missing {missing}")
        track_a = str(region["track_a"])
        track_b = str(region["track_b"])
        if track_a == track_b:
            raise ValueError(f"heard_regions[{index}] must reference two distinct tracks")
        rate = region["sample_rate"]
        if isinstance(rate, bool) or not isinstance(rate, (numbers.Integral, float)):
            raise ValueError(f"heard_regions[{index}].sample_rate must be numeric")
        if not math.isfinite(float(rate)) or float(rate) <= 0:
            raise ValueError(f"heard_regions[{index}].sample_rate must be positive")
        for key in ("cue_out_sample", "cue_in_sample"):
            value = region[key]
            if isinstance(value, bool) or not isinstance(value, (numbers.Integral, float)):
                raise ValueError(f"heard_regions[{index}].{key} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"heard_regions[{index}].{key} must be non-negative")
        result.append(dict(region))
    return result


def _proposal_geometry(proposal: Any) -> tuple[str, str, int, int]:
    candidate = proposal.candidate
    return (
        str(candidate.track_a),
        str(candidate.track_b),
        int(candidate.cue_out_sample),
        int(candidate.cue_in_sample),
    )


def _proposal_score(proposal: Any) -> float:
    try:
        score = float(proposal.score)
    except (TypeError, ValueError):
        return float("-inf")
    return score if math.isfinite(score) else float("-inf")


def _heard_flags(
    proposal: Any,
    regions: Sequence[Mapping[str, Any]],
    tolerance_seconds: float,
    tracks: Mapping[str, AudioTrack],
) -> dict[str, bool]:
    candidate = proposal.candidate
    rate_a = tracks[candidate.track_a].sample_rate
    rate_b = tracks[candidate.track_b].sample_rate
    # ``exposure`` interprets both cue samples with one sample rate.  Call it
    # once per side so each source region is compared in its own timeline;
    # taking the matching flag from each call avoids a false match when the
    # two tracks were decoded at different rates.
    outgoing = exposure(candidate, list(regions), tolerance_seconds, sample_rate=rate_a)
    incoming = exposure(candidate, list(regions), tolerance_seconds, sample_rate=rate_b)
    outgoing_heard = bool(outgoing.get("outgoing_region_heard"))
    incoming_heard = bool(incoming.get("incoming_region_heard"))
    return {
        "outgoing_region_heard": outgoing_heard,
        "incoming_region_heard": incoming_heard,
        "ordered_pair_heard": bool(
            outgoing.get("ordered_pair_heard")
            or incoming.get("ordered_pair_heard")
            or (outgoing_heard and incoming_heard)
        ),
    }


def _max_flow(
    source_edges: Mapping[str, int],
    incoming_edges: Mapping[str, int],
    pair_capacity: Mapping[tuple[str, str], int],
) -> int:
    """Deterministic Dinic flow for the pool's small bipartite capacity graph.

    Only cardinality is needed here: the pool uses it to reject a greedy choice
    that would make the requested ``top_k`` unreachable under the fixed role
    and per-pair caps.  Scores never enter this feasibility check.
    """

    nodes = ["@source", "@sink"]
    for track_id in source_edges:
        nodes.append(f"src:{track_id}")
    for track_id in incoming_edges:
        nodes.append(f"inc:{track_id}")
    index = {node: position for position, node in enumerate(nodes)}
    graph: list[list[list[int]]] = [[] for _ in nodes]

    def add_edge(left: str, right: str, capacity: int) -> None:
        if capacity <= 0:
            return
        graph[index[left]].append([index[right], capacity, len(graph[index[right]])])
        graph[index[right]].append([index[left], 0, len(graph[index[left]]) - 1])

    for track_id, capacity in source_edges.items():
        add_edge("@source", f"src:{track_id}", capacity)
    for (track_a, track_b), capacity in pair_capacity.items():
        add_edge(f"src:{track_a}", f"inc:{track_b}", capacity)
    for track_id, capacity in incoming_edges.items():
        add_edge(f"inc:{track_id}", "@sink", capacity)

    total = 0
    source_index, sink_index = index["@source"], index["@sink"]
    while True:
        level = [-1] * len(nodes)
        level[source_index] = 0
        queue = [source_index]
        for node in queue:
            for target, capacity, _ in graph[node]:
                if capacity > 0 and level[target] < 0:
                    level[target] = level[node] + 1
                    queue.append(target)
        if level[sink_index] < 0:
            break
        cursor = [0] * len(nodes)

        def send(node: int, limit: int) -> int:
            if node == sink_index:
                return limit
            while cursor[node] < len(graph[node]):
                edge = graph[node][cursor[node]]
                target, capacity, reverse = edge
                if capacity > 0 and level[target] == level[node] + 1:
                    pushed = send(target, min(limit, capacity))
                    if pushed:
                        edge[1] -= pushed
                        graph[target][reverse][1] += pushed
                        return pushed
                cursor[node] += 1
            return 0

        while True:
            pushed = send(source_index, 10**9)
            if not pushed:
                break
            total += pushed
    return total


def _select_globally(
    proposals: Sequence[Any],
    *,
    top_k: int,
    max_per_pair: int,
    role_cap: int,
    cue_spacing_seconds: float,
    tracks: Mapping[str, AudioTrack],
) -> tuple[list[Any], dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    ordered = sorted(
        proposals,
        key=lambda proposal: (-_proposal_score(proposal), str(proposal.candidate.candidate_id)),
    )
    selected: list[Any] = []
    pair_counts: Counter[str] = Counter()
    source_role_counts: Counter[str] = Counter()
    incoming_role_counts: Counter[str] = Counter()
    cue_history: dict[str, list[float]] = defaultdict(list)
    seen_geometry: set[tuple[str, str, int, int]] = set()
    rejection_counts: Counter[str] = Counter()
    remaining_pairs: Counter[str] = Counter(
        f"{proposal.candidate.track_a}->{proposal.candidate.track_b}"
        for proposal in ordered
    )

    initial_capacities: dict[tuple[str, str], int] = {}
    for candidate_pair, count in remaining_pairs.items():
        track_a, track_b = candidate_pair.split("->", 1)
        if track_a != track_b:
            initial_capacities[(track_a, track_b)] = min(max_per_pair, count)
    # The graph target is the number of proposals actually attainable under
    # role and per-pair caps.  Comparing the lookahead against the requested
    # top_k would reject every candidate in a sparse pool even though no
    # selection could ever reach that size.
    attainable_target = min(
        top_k,
        _max_flow(
            {track_id: role_cap for track_id in tracks},
            {track_id: role_cap for track_id in tracks},
            initial_capacities,
        ),
    )

    def feasible_after(proposal: Any) -> bool:
        """Return an optimistic upper bound for role/pair cap feasibility.

        The flow deliberately ignores cue spacing and leaves already
        spacing-rejected records in ``remaining_pairs``.  It is a conservative
        guard against consuming the last cap slot, not a claim that spacing
        can always be satisfied.
        """

        pair_id = f"{proposal.candidate.track_a}->{proposal.candidate.track_b}"
        sources = {
            track_id: max(0, role_cap - source_role_counts[track_id] - (1 if track_id == proposal.candidate.track_a else 0))
            for track_id in tracks
        }
        incoming = {
            track_id: max(0, role_cap - incoming_role_counts[track_id] - (1 if track_id == proposal.candidate.track_b else 0))
            for track_id in tracks
        }
        capacities: dict[tuple[str, str], int] = {}
        for candidate_pair in remaining_pairs:
            if candidate_pair == pair_id:
                # ``remaining_pairs`` still includes the proposal under
                # consideration, so both bounds must exclude it.
                remaining = min(
                    max_per_pair - pair_counts[pair_id] - 1,
                    remaining_pairs[candidate_pair] - 1,
                )
            else:
                remaining = min(
                    max_per_pair - pair_counts[candidate_pair],
                    remaining_pairs[candidate_pair],
                )
            if remaining <= 0:
                continue
            track_a, track_b = candidate_pair.split("->", 1)
            if track_a == track_b:
                continue
            capacities[(track_a, track_b)] = remaining
        return len(selected) + 1 + _max_flow(sources, incoming, capacities) >= attainable_target

    for proposal in ordered:
        geometry = _proposal_geometry(proposal)
        if geometry in seen_geometry:
            rejection_counts["duplicate_pair_cue"] += 1
            continue
        if len(selected) >= top_k:
            rejection_counts["top_k_limit"] += 1
            continue
        candidate = proposal.candidate
        pair_id = f"{candidate.track_a}->{candidate.track_b}"
        if pair_counts[pair_id] >= max_per_pair:
            rejection_counts["max_per_pair"] += 1
            continue
        if source_role_counts[candidate.track_a] >= role_cap:
            rejection_counts["source_role_cap"] += 1
            continue
        if incoming_role_counts[candidate.track_b] >= role_cap:
            rejection_counts["incoming_role_cap"] += 1
            continue
        rate_a = tracks[candidate.track_a].sample_rate
        rate_b = tracks[candidate.track_b].sample_rate
        out_time = candidate.cue_out_sample / float(rate_a)
        in_time = candidate.cue_in_sample / float(rate_b)
        spacing_blocked = False
        if any(abs(out_time - previous) < cue_spacing_seconds for previous in cue_history[candidate.track_a]):
            spacing_blocked = True
        if any(abs(in_time - previous) < cue_spacing_seconds for previous in cue_history[candidate.track_b]):
            spacing_blocked = True
        if spacing_blocked:
            rejection_counts["cue_spacing"] += 1
            continue
        if len(selected) + 1 < attainable_target and not feasible_after(proposal):
            # Accepting this high-scoring record would consume a role or pair
            # slot needed to reach the attainable pool size.  The cap itself is
            # not widened; the record is deferred instead.
            rejection_counts["would_block_capacity"] += 1
            continue
        seen_geometry.add(geometry)
        pair_counts[pair_id] += 1
        source_role_counts[candidate.track_a] += 1
        incoming_role_counts[candidate.track_b] += 1
        cue_history[candidate.track_a].append(out_time)
        cue_history[candidate.track_b].append(in_time)
        remaining_pairs[pair_id] -= 1
        selected.append(proposal)

    return (
        selected,
        dict(rejection_counts),
        dict(pair_counts),
        dict(source_role_counts),
        dict(incoming_role_counts),
    )


def search_pool(
    tracks: dict[str, AudioTrack],
    structures: dict[str, dict],
    vocals: dict[str, dict],
    *,
    top_k: int = 8,
    max_per_pair: int = 4,
    cue_policy: str = "downbeats",
    max_candidates_per_pair: int = 30000,
    heard_regions: Any = None,
    heard_tolerance_seconds: float = 8.0,
) -> dict:
    """Search every ordered pair and return a globally diversified pool.

    The return value is JSON-compatible.  ``proposals`` contains every
    admitted proposal exactly once, ``selected`` contains the globally chosen
    records with their original directional score/provenance/alignment, and
    ``all_pair_counts`` accounts for every directed pair.  Rejected directional
    records are reduced to counts so the result cannot grow with the raw cue
    grid.
    """

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(max_per_pair, int) or isinstance(max_per_pair, bool) or max_per_pair <= 0:
        raise ValueError("max_per_pair must be a positive integer")
    if not isinstance(max_candidates_per_pair, int) or isinstance(max_candidates_per_pair, bool) or max_candidates_per_pair <= 0:
        raise ValueError("max_candidates_per_pair must be a positive integer")
    if cue_policy not in {"downbeats", "beats", "provided"}:
        raise ValueError("cue_policy must be 'downbeats', 'beats', or 'provided'")
    if isinstance(heard_tolerance_seconds, bool) or not isinstance(heard_tolerance_seconds, (int, float)):
        raise ValueError("heard_tolerance_seconds must be finite and non-negative")
    if not math.isfinite(float(heard_tolerance_seconds)) or float(heard_tolerance_seconds) < 0:
        raise ValueError("heard_tolerance_seconds must be finite and non-negative")

    if not isinstance(tracks, Mapping):
        raise TypeError("tracks must be a mapping keyed by canonical track ID")
    if len(tracks) < 3:
        raise ValueError("search_pool requires at least three distinct tracks")
    canonical: dict[str, AudioTrack] = {}
    for key, track in tracks.items():
        if not isinstance(track, AudioTrack):
            raise TypeError(f"tracks[{key!r}] must be an AudioTrack")
        if str(key) != track.track_id:
            raise ValueError(f"tracks key {key!r} is not the canonical track ID {track.track_id!r}")
        if track.track_id in canonical:
            raise ValueError(f"duplicate canonical track ID: {track.track_id}")
        canonical[track.track_id] = track
    if len(canonical) < 3:
        raise ValueError("search_pool requires at least three distinct canonical tracks")

    _validate_key_set("structures", structures, canonical)
    _validate_key_set("vocals", vocals, canonical)

    evidence: dict[str, dict[str, Any]] = {}
    for track_id, track in canonical.items():
        evidence[track_id] = {
            "structure": _validate_evidence(track, structures[track_id], f"structures[{track_id}]"),
            "vocals": _validate_evidence(track, vocals[track_id], f"vocals[{track_id}]"),
        }

    regions = _coerce_heard_regions(heard_regions)
    track_ids = sorted(canonical)
    pair_ids = [
        f"{track_a}->{track_b}"
        for track_a in track_ids
        for track_b in track_ids
        if track_a != track_b
    ]

    admitted_by_geometry: dict[tuple[str, str, int, int], Any] = {}
    pair_counts: dict[str, dict[str, Any]] = {}
    directional_rejections: Counter[str] = Counter()
    duplicate_admitted = 0

    for track_a in track_ids:
        for track_b in track_ids:
            if track_a == track_b:
                continue
            pair_id = f"{track_a}->{track_b}"
            result = search_directional(
                canonical[track_a],
                canonical[track_b],
                structures[track_a],
                structures[track_b],
                vocals[track_a],
                vocals[track_b],
                top_k=max_per_pair,
                max_candidates=max_candidates_per_pair,
                allow_unknown_vocals=False,
                cue_policy=cue_policy,
                min_cue_spacing_seconds=DEFAULT_CUE_SPACING_SECONDS,
                prefilter_vocals=True,
            )
            result_counts = Counter({
                str(reason): int(count)
                for reason, count in (result.diagnostics.get("rejection_reason_counts", {}) or {}).items()
            })
            directional_rejections.update(result_counts)
            # ``items`` is only the directional top-k selection.  The complete
            # admitted pool for global selection is ``proposals``; using the
            # shorter surface would permanently discard viable alternatives
            # before heard-region or spacing filters run.
            admitted_here = list(result.proposals)
            unique_here = 0
            for proposal in admitted_here:
                geometry = _proposal_geometry(proposal)
                previous = admitted_by_geometry.get(geometry)
                if previous is not None:
                    duplicate_admitted += 1
                    # Keep the higher-scored record for an exact geometry; the
                    # directional search can emit the same anchors at two
                    # context-window widths.
                    if _proposal_score(proposal) > _proposal_score(previous):
                        admitted_by_geometry[geometry] = proposal
                    continue
                admitted_by_geometry[geometry] = proposal
                unique_here += 1
            pair_counts[pair_id] = {
                "track_a": track_a,
                "track_b": track_b,
                "cue_policy": result.diagnostics.get("cue_policy", cue_policy),
                "cue_source_a": result.diagnostics.get("cue_source_a"),
                "cue_source_b": result.diagnostics.get("cue_source_b"),
                "candidate_grid_size": result.diagnostics.get("candidate_grid_size"),
                "candidate_count_evaluated": result.diagnostics.get("candidate_count_evaluated"),
                "candidate_grid_truncated": result.diagnostics.get("candidate_grid_truncated"),
                "vocal_prefilter": dict(result.diagnostics.get("vocal_prefilter", {})),
                "proposals_scored": result.diagnostics.get("proposals_scored"),
                "admitted_count": len(admitted_here),
                "admitted_unique_count": unique_here,
                "directional_selected_count": len(result.items),
                "directional_requested_top_k": result.diagnostics.get("requested_top_k", max_per_pair),
                "directional_selection_shortfall": int(result.diagnostics.get("selection_shortfall", 0) or 0),
                "rejection_reason_counts": dict(sorted(result_counts.items())),
                "vocal_input_a": dict(result.diagnostics.get("vocal_input_a", {}) or {}),
                "vocal_input_b": dict(result.diagnostics.get("vocal_input_b", {}) or {}),
                "selected_count": 0,
            }

    admitted = sorted(
        admitted_by_geometry.values(),
        key=lambda proposal: (-_proposal_score(proposal), str(proposal.candidate.candidate_id)),
    )

    heard_rejections = 0
    heard_reason_counts: Counter[str] = Counter()
    if regions:
        retained: list[Any] = []
        for proposal in admitted:
            flags = _heard_flags(proposal, regions, float(heard_tolerance_seconds), canonical)
            if flags["outgoing_region_heard"] or flags["incoming_region_heard"]:
                heard_rejections += 1
                if flags["outgoing_region_heard"]:
                    heard_reason_counts["outgoing_region_heard"] += 1
                if flags["incoming_region_heard"]:
                    heard_reason_counts["incoming_region_heard"] += 1
                if flags["ordered_pair_heard"]:
                    heard_reason_counts["ordered_pair_heard"] += 1
                continue
            retained.append(proposal)
        admitted = retained

    (
        selected,
        selection_rejection_counts,
        selected_pair_counts,
        source_role_counts,
        incoming_role_counts,
    ) = _select_globally(
        admitted,
        top_k=top_k,
        max_per_pair=max_per_pair,
        role_cap=DEFAULT_ROLE_CAP,
        cue_spacing_seconds=DEFAULT_CUE_SPACING_SECONDS,
        tracks=canonical,
    )

    for pair_id, count in selected_pair_counts.items():
        if pair_id in pair_counts:
            pair_counts[pair_id]["selected_count"] = count

    proposals_payload = []
    for proposal in admitted:
        record = asdict(proposal)
        record["pair_id"] = f"{proposal.candidate.track_a}->{proposal.candidate.track_b}"
        proposals_payload.append(record)

    selected_payload = []
    for proposal in selected:
        record = asdict(proposal)
        record["pair_id"] = f"{proposal.candidate.track_a}->{proposal.candidate.track_b}"
        selected_payload.append(record)

    global_rejection_counts = Counter(selection_rejection_counts)
    global_rejection_counts["heard_region"] += heard_rejections
    global_rejection_counts["duplicate_admitted_geometry"] += duplicate_admitted

    expected_pair_count = len(canonical) * (len(canonical) - 1)
    shortfall = max(0, top_k - len(selected))
    role_cap_feasible = top_k <= DEFAULT_ROLE_CAP * len(canonical)
    max_per_pair_feasible = top_k <= max_per_pair * expected_pair_count
    shortfall_reasons: list[str] = []
    if shortfall:
        if not role_cap_feasible:
            shortfall_reasons.append("role_cap_infeasible_for_requested_top_k")
        if not max_per_pair_feasible:
            shortfall_reasons.append("max_per_pair_infeasible_for_requested_top_k")
        if not admitted:
            shortfall_reasons.append("no_admitted_proposals")
        elif len(admitted) < top_k:
            shortfall_reasons.append("fewer_admitted_proposals_than_top_k")
        if not shortfall_reasons:
            shortfall_reasons.append("diversity_constraints_limited_selection")
        if heard_rejections:
            shortfall_reasons.append("heard_regions_removed_candidates")

    diagnostics = {
        "algorithm": POOL_ALGORITHM,
        "schema_version": SCHEMA_VERSION,
        "track_count": len(canonical),
        "expected_ordered_pair_count": expected_pair_count,
        "ordered_pair_count": len(pair_ids),
        "all_ordered_pairs_covered": len(pair_ids) == expected_pair_count,
        "requested_top_k": top_k,
        "selected_count": len(selected),
        "selection_shortfall": shortfall,
        "selected_count_lt_top_k": shortfall > 0,
        "shortfall_reasons": shortfall_reasons,
        "max_per_pair": max_per_pair,
        "role_cap_per_track_per_role": DEFAULT_ROLE_CAP,
        "role_cap_feasible_for_top_k": role_cap_feasible,
        "max_per_pair_feasible_for_top_k": max_per_pair_feasible,
        "capacity_lookahead": "optimistic_role_pair_flow_ignores_spacing_not_exact_feasibility",
        "cue_spacing_seconds": DEFAULT_CUE_SPACING_SECONDS,
        "cue_policy": cue_policy,
        "max_candidates_per_pair": max_candidates_per_pair,
        "allow_unknown_vocals": False,
        "admitted_unique_count": len(admitted),
        "heard_regions_provided": bool(regions),
        "heard_region_count": len(regions or ()),
        "heard_tolerance_seconds": float(heard_tolerance_seconds),
        "heard_rejected_count": heard_rejections,
        "duplicate_admitted_geometry_count": duplicate_admitted,
        "selection_rejection_counts": dict(sorted(selection_rejection_counts.items())),
        "heard_rejection_counts": dict(sorted(heard_reason_counts.items())),
        "directional_rejection_reason_counts": dict(sorted(directional_rejections.items())),
        "selected_source_role_counts": dict(sorted(source_role_counts.items())),
        "selected_incoming_role_counts": dict(sorted(incoming_role_counts.items())),
        "pair_selected_counts": dict(sorted(selected_pair_counts.items())),
        "human_listening_validation": "not_performed",
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm": POOL_ALGORITHM,
        "tracks": {
            track_id: {
                **evidence[track_id]["structure"],
                "structure_hash_fields": evidence[track_id]["structure"]["hash_fields"],
                "vocal_hash_fields": evidence[track_id]["vocals"]["hash_fields"],
            }
            for track_id in track_ids
        },
        "pair_ids": pair_ids,
        "pair_count": len(pair_ids),
        "all_pair_counts": pair_counts,
        "proposals": proposals_payload,
        "selected": selected_payload,
        "rejection_reason_counts": dict(sorted(
            (Counter(directional_rejections) + Counter(global_rejection_counts)).items()
        )),
        "rejection_counts": {
            "directional_by_reason": dict(sorted(directional_rejections.items())),
            "global_by_reason": dict(sorted(global_rejection_counts.items())),
            "by_pair": {
                pair_id: dict(counts["rejection_reason_counts"])
                for pair_id, counts in pair_counts.items()
            },
        },
        "diagnostics": diagnostics,
    }


__all__ = ["search_pool", "SCHEMA_VERSION", "POOL_ALGORITHM"]
