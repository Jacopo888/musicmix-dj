from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest

from musicmix.audio import AudioTrack
from musicmix.contracts import Candidate, Feature
from musicmix.directional import DirectionalProposal, DirectionalSearchResult
from musicmix.pool_search import search_pool


RATE = 100
DURATION = 400 * RATE
TRACK_IDS = ("hash_a", "hash_b", "hash_c", "hash_d")


def _track(track_id: str) -> AudioTrack:
    # The mocked directional search never reads the waveform; a short constant
    # array keeps the canonical duration explicit and cheap.
    samples = np.zeros((DURATION, 2), dtype=np.float32)
    return AudioTrack(track_id, f"{track_id}.wav", RATE, samples)


def _evidence(track: AudioTrack) -> dict:
    return {
        "schema_version": "test.1",
        "track_id": track.track_id,
        "canonical_file_sha256": track.track_id,
        "sample_rate": track.sample_rate,
        "duration_samples": track.duration_samples,
    }


def _fixture():
    tracks = {track_id: _track(track_id) for track_id in TRACK_IDS}
    structures = {track_id: _evidence(track) for track_id, track in tracks.items()}
    vocals = {track_id: _evidence(track) for track_id, track in tracks.items()}
    return tracks, structures, vocals


def _proposal(track_a: str, track_b: str, out_sample: int, in_sample: int, score: float,
              tag: str = "") -> DirectionalProposal:
    candidate = Candidate(
        candidate_id=f"cand_{track_a}_{track_b}_{out_sample}_{in_sample}_{tag}",
        track_a=track_a,
        track_b=track_b,
        cue_out_sample=out_sample,
        cue_in_sample=in_sample,
        mode="structure",
        features={"horizontal": Feature(0.7, 0.9, "directional-v2")},
        penalties={},
    )
    return DirectionalProposal(
        candidate=candidate,
        score=score,
        breakdown={"harmony_context": 0.7, "vocal_safety": 1.0},
        raw_features={
            "original_structure_cue_samples": {"a": out_sample, "b": in_sample},
            "direction": "outgoing_then_incoming",
            "window_beats": 8,
        },
        fade_seconds=0.04,
        alignment={
            "outgoing": {"safe": True, "status": "safe_confirmed_gap"},
            "incoming": {"safe": True, "status": "safe_confirmed_gap"},
            "boundary_class": {
                "outgoing": "phrase_boundary_supported",
                "incoming": "phrase_boundary_supported",
            },
        },
        reasons=(),
    )


def _fake_search(calls: list[dict], *, duplicate_geometry: bool = False):
    pair_index = {pair: index for index, pair in enumerate(
        (a, b) for a in TRACK_IDS for b in TRACK_IDS if a != b
    )}

    def search(track_a, track_b, structure_a, structure_b, vocals_a, vocals_b, **kwargs):
        calls.append({
            "track_a": track_a.track_id,
            "track_b": track_b.track_id,
            "kwargs": dict(kwargs),
        })
        index = pair_index[(track_a.track_id, track_b.track_id)]
        proposals = []
        for offset in range(3):
            # Source cues and incoming cues occupy disjoint, role-specific
            # timelines so the pool's cross-role eight-second spacing rule is
            # satisfied by construction.
            out_sample = (20 + index * 40 + offset * 12) * RATE
            in_sample = (25 + index * 40 + offset * 12) * RATE
            score = 0.9 - index * 0.01 - offset * 0.02
            proposals.append(_proposal(track_a.track_id, track_b.track_id,
                                       out_sample, in_sample, score, tag=str(offset)))
        if duplicate_geometry:
            # Same anchors, different context window: the pool must keep one
            # exact pair/cue geometry and never count it twice.
            duplicate = _proposal(track_a.track_id, track_b.track_id,
                                  proposals[0].candidate.cue_out_sample,
                                  proposals[0].candidate.cue_in_sample, 0.95, tag="dup")
            proposals.insert(0, duplicate)
        # Mirror the real contract: ``items`` is the bounded directional
        # selection, while ``proposals`` is the complete admitted pool.
        selected_items = proposals[: kwargs.get("top_k", len(proposals))]
        diagnostics = {
            "cue_policy": kwargs.get("cue_policy"),
            "cue_source_a": kwargs.get("cue_policy"),
            "cue_source_b": kwargs.get("cue_policy"),
            "candidate_grid_size": 12,
            "candidate_count_evaluated": 12,
            "candidate_grid_truncated": False,
            "proposals_scored": len(proposals),
            "requested_top_k": kwargs.get("top_k"),
            "selected_count": len(selected_items),
            "selection_shortfall": max(0, kwargs.get("top_k", 0) - len(selected_items)),
            "rejection_reason_counts": {"vocal_boundary_rejected": 2},
            "vocal_input_a": {"provided": True, "status": "asr_temporal_evidence_available"},
            "vocal_input_b": {"provided": True, "status": "asr_temporal_evidence_available"},
        }
        return DirectionalSearchResult(
            items=tuple(selected_items),
            rejected=(),
            diagnostics=diagnostics,
            proposals=tuple(proposals),
            selected=tuple(item.candidate.candidate_id for item in selected_items),
        )

    return search


def _synthetic_result(proposals: list[DirectionalProposal], *, top_k: int) -> DirectionalSearchResult:
    selected = proposals[:top_k]
    diagnostics = {
        "cue_policy": "downbeats",
        "cue_source_a": "downbeats",
        "cue_source_b": "downbeats",
        "candidate_grid_size": len(proposals),
        "candidate_count_evaluated": len(proposals),
        "candidate_grid_truncated": False,
        "proposals_scored": len(proposals),
        "requested_top_k": top_k,
        "selected_count": len(selected),
        "selection_shortfall": max(0, top_k - len(selected)),
        "rejection_reason_counts": {"vocal_boundary_rejected": 1},
        "vocal_input_a": {"provided": True},
        "vocal_input_b": {"provided": True},
    }
    return DirectionalSearchResult(
        items=tuple(selected),
        rejected=(),
        diagnostics=diagnostics,
        proposals=tuple(proposals),
        selected=tuple(item.candidate.candidate_id for item in selected),
    )


def test_exhausts_every_ordered_pair_and_never_self_pairs() -> None:
    tracks, structures, vocals = _fixture()
    calls: list[dict] = []
    with patch("musicmix.pool_search.search_directional", side_effect=_fake_search(calls)):
        result = search_pool(tracks, structures, vocals, top_k=8, max_per_pair=2)

    assert result["pair_count"] == 12
    assert len(calls) == 12
    assert len({(call["track_a"], call["track_b"]) for call in calls}) == 12
    assert all(call["track_a"] != call["track_b"] for call in calls)
    assert set(result["all_pair_counts"]) == {
        f"{a}->{b}" for a in TRACK_IDS for b in TRACK_IDS if a != b
    }
    assert result["diagnostics"]["all_ordered_pairs_covered"] is True
    assert all("->" in pair and pair.split("->")[0] != pair.split("->")[1]
               for pair in result["pair_ids"])
    assert all(call["kwargs"]["allow_unknown_vocals"] is False for call in calls)
    assert all(call["kwargs"]["top_k"] == 2 for call in calls)
    assert all(call["kwargs"]["max_candidates"] == 30000 for call in calls)
    assert all(call["kwargs"]["min_cue_spacing_seconds"] == 8.0 for call in calls)


def test_hash_sample_rate_and_duration_mismatches_fail_before_search() -> None:
    tracks, structures, vocals = _fixture()
    calls: list[dict] = []
    search = _fake_search(calls)

    bad_hash = {key: dict(value) for key, value in structures.items()}
    bad_hash["hash_b"]["canonical_file_sha256"] = "0" * 64
    with patch("musicmix.pool_search.search_directional", side_effect=search):
        with pytest.raises(ValueError, match="hash"):
            search_pool(tracks, bad_hash, vocals)
    assert calls == []

    bad_rate = {key: dict(value) for key, value in vocals.items()}
    bad_rate["hash_c"]["sample_rate"] = RATE + 1
    with patch("musicmix.pool_search.search_directional", side_effect=search):
        with pytest.raises(ValueError, match="sample_rate"):
            search_pool(tracks, structures, bad_rate)
    assert calls == []

    bad_duration = {key: dict(value) for key, value in structures.items()}
    bad_duration["hash_d"]["duration_samples"] = DURATION + 1
    with patch("musicmix.pool_search.search_directional", side_effect=search):
        with pytest.raises(ValueError, match="duration_samples"):
            search_pool(tracks, bad_duration, vocals)
    assert calls == []

    missing_hash = {key: dict(value) for key, value in vocals.items()}
    missing_hash["hash_a"]["canonical_file_sha256"] = ""
    with patch("musicmix.pool_search.search_directional", side_effect=search):
        with pytest.raises(ValueError, match="present but empty"):
            search_pool(tracks, structures, missing_hash)
    assert calls == []


def test_global_selection_enforces_pair_role_and_cue_diversity() -> None:
    tracks, structures, vocals = _fixture()
    calls: list[dict] = []
    with patch("musicmix.pool_search.search_directional", side_effect=_fake_search(calls)):
        result = search_pool(tracks, structures, vocals, top_k=8, max_per_pair=1)

    selected = result["selected"]
    assert len(selected) == 8
    assert result["diagnostics"]["selection_shortfall"] == 0

    pair_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    incoming_counts: dict[str, int] = {}
    cues: dict[str, list[float]] = {track_id: [] for track_id in TRACK_IDS}
    seen_geometries = set()
    for record in selected:
        candidate = record["candidate"]
        pair = f"{candidate['track_a']}->{candidate['track_b']}"
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        source_counts[candidate["track_a"]] = source_counts.get(candidate["track_a"], 0) + 1
        incoming_counts[candidate["track_b"]] = incoming_counts.get(candidate["track_b"], 0) + 1
        geometry = (candidate["track_a"], candidate["track_b"],
                    candidate["cue_out_sample"], candidate["cue_in_sample"])
        assert geometry not in seen_geometries
        seen_geometries.add(geometry)
        cues[candidate["track_a"]].append(candidate["cue_out_sample"] / RATE)
        cues[candidate["track_b"]].append(candidate["cue_in_sample"] / RATE)

    assert all(count <= 1 for count in pair_counts.values())
    assert all(count <= 2 for count in source_counts.values())
    assert all(count <= 2 for count in incoming_counts.values())
    for values in cues.values():
        ordered = sorted(values)
        assert all(right - left >= 8.0 for left, right in zip(ordered, ordered[1:]))
    # Every original score/breakdown/provenance/alignment survives the pool.
    admitted = {record["candidate"]["candidate_id"]: record for record in result["proposals"]}
    for record in selected:
        original = admitted[record["candidate"]["candidate_id"]]
        assert record["score"] == original["score"]
        assert record["breakdown"] == original["breakdown"]
        assert record["raw_features"] == original["raw_features"]
        assert record["alignment"] == original["alignment"]


def test_exact_pair_cue_duplicates_are_kept_once() -> None:
    tracks, structures, vocals = _fixture()
    with patch("musicmix.pool_search.search_directional",
               side_effect=_fake_search([], duplicate_geometry=True)):
        result = search_pool(tracks, structures, vocals, top_k=8, max_per_pair=2)

    geometries = [
        (record["candidate"]["track_a"], record["candidate"]["track_b"],
         record["candidate"]["cue_out_sample"], record["candidate"]["cue_in_sample"])
        for record in result["proposals"]
    ]
    assert len(geometries) == len(set(geometries))
    assert result["diagnostics"]["duplicate_admitted_geometry_count"] == 12
    assert result["diagnostics"]["admitted_unique_count"] == 36


def test_sparse_admitted_pool_returns_candidates_with_explicit_shortfall() -> None:
    tracks, structures, vocals = _fixture()

    def search(track_a, track_b, structure_a, structure_b, vocals_a, vocals_b, **kwargs):
        if (track_a.track_id, track_b.track_id) == ("hash_a", "hash_b"):
            proposals = [_proposal("hash_a", "hash_b", 20 * RATE, 25 * RATE, 0.80, tag="only")]
        elif (track_a.track_id, track_b.track_id) == ("hash_c", "hash_d"):
            proposals = [_proposal("hash_c", "hash_d", 60 * RATE, 65 * RATE, 0.70, tag="only")]
        else:
            proposals = []
        return _synthetic_result(proposals, top_k=kwargs["top_k"])

    with patch("musicmix.pool_search.search_directional", side_effect=search):
        result = search_pool(tracks, structures, vocals, top_k=8, max_per_pair=1)

    diagnostics = result["diagnostics"]
    assert len(result["selected"]) == 2
    assert len(result["selected"]) >= 1
    assert diagnostics["selection_shortfall"] == 6
    assert diagnostics["selected_count_lt_top_k"] is True
    assert "fewer_admitted_proposals_than_top_k" in diagnostics["shortfall_reasons"]
    assert diagnostics["selection_rejection_counts"].get("would_block_capacity", 0) == 0
    assert set(diagnostics["capacity_lookahead"].split("_")) >= {"optimistic", "spacing"}


def test_shortfall_is_explicit_and_never_silently_relaxed() -> None:
    tracks, structures, vocals = _fixture()
    with patch("musicmix.pool_search.search_directional", side_effect=_fake_search([])):
        result = search_pool(tracks, structures, vocals, top_k=20, max_per_pair=4)

    diagnostics = result["diagnostics"]
    assert len(result["selected"]) < 20
    assert diagnostics["selection_shortfall"] == 20 - len(result["selected"])
    assert diagnostics["selected_count_lt_top_k"] is True
    assert diagnostics["role_cap_feasible_for_top_k"] is False
    assert "role_cap_infeasible_for_requested_top_k" in diagnostics["shortfall_reasons"]
    # No cap is widened: the observed role counts remain within the default.
    assert all(count <= 2 for count in diagnostics["selected_source_role_counts"].values())
    assert all(count <= 2 for count in diagnostics["selected_incoming_role_counts"].values())


def test_heard_regions_filter_both_source_regions_conservatively() -> None:
    tracks, structures, vocals = _fixture()
    # Pair hash_a -> hash_b, first mocked proposal: out 20 s, in 25 s.
    heard = [{
        "track_a": "hash_a",
        "track_b": "hash_b",
        "cue_out_sample": 20 * RATE,
        "cue_in_sample": 25 * RATE,
        "sample_rate": RATE,
    }]
    with patch("musicmix.pool_search.search_directional", side_effect=_fake_search([])):
        result = search_pool(tracks, structures, vocals, top_k=8, max_per_pair=2,
                             heard_regions=heard, heard_tolerance_seconds=0.5)

    assert result["diagnostics"]["heard_rejected_count"] >= 1
    assert "outgoing_region_heard" in result["diagnostics"]["heard_rejection_counts"]
    for record in result["proposals"]:
        candidate = record["candidate"]
        # Only the heard source region is excluded; the other candidates at
        # nearby incoming cues (different outgoing cues) remain eligible.
        assert not (candidate["track_a"] == "hash_a"
                    and abs(candidate["cue_out_sample"] - 20 * RATE) <= 0.5 * RATE)


def test_lower_ranked_admitted_candidate_survives_heard_top_item() -> None:
    tracks, structures, vocals = _fixture()

    def search(track_a, track_b, structure_a, structure_b, vocals_a, vocals_b, **kwargs):
        if (track_a.track_id, track_b.track_id) == ("hash_a", "hash_b"):
            proposals = [
                _proposal("hash_a", "hash_b", 20 * RATE, 25 * RATE, 0.99, tag="heard"),
                _proposal("hash_a", "hash_b", 40 * RATE, 45 * RATE, 0.40, tag="fallback"),
            ]
        elif (track_a.track_id, track_b.track_id) == ("hash_a", "hash_c"):
            proposals = [_proposal("hash_a", "hash_c", 60 * RATE, 65 * RATE, 0.30, tag="other")]
        else:
            proposals = []
        return _synthetic_result(proposals, top_k=kwargs["top_k"])

    heard = [{
        "track_a": "hash_a", "track_b": "hash_b",
        "cue_out_sample": 20 * RATE, "cue_in_sample": 25 * RATE,
        "sample_rate": RATE,
    }]
    with patch("musicmix.pool_search.search_directional", side_effect=search):
        result = search_pool(tracks, structures, vocals, top_k=2, max_per_pair=2,
                             heard_regions=heard, heard_tolerance_seconds=0.5)

    admitted = {record["candidate"]["candidate_id"] for record in result["proposals"]}
    selected = {record["candidate"]["candidate_id"] for record in result["selected"]}
    assert "cand_hash_a_hash_b_2000_2500_heard" not in admitted
    assert "cand_hash_a_hash_b_4000_4500_fallback" in admitted
    assert "cand_hash_a_hash_b_4000_4500_fallback" in selected
    assert result["all_pair_counts"]["hash_a->hash_b"]["admitted_count"] == 2


def test_lower_ranked_admitted_candidate_survives_spacing_block() -> None:
    tracks, structures, vocals = _fixture()

    def search(track_a, track_b, structure_a, structure_b, vocals_a, vocals_b, **kwargs):
        if (track_a.track_id, track_b.track_id) == ("hash_a", "hash_b"):
            proposals = [
                _proposal("hash_a", "hash_b", 20 * RATE, 25 * RATE, 0.98, tag="blocked"),
                _proposal("hash_a", "hash_b", 40 * RATE, 45 * RATE, 0.50, tag="fallback"),
            ]
        elif (track_a.track_id, track_b.track_id) == ("hash_a", "hash_c"):
            proposals = [_proposal("hash_a", "hash_c", 21 * RATE, 26 * RATE, 0.99, tag="first")]
        else:
            proposals = []
        return _synthetic_result(proposals, top_k=kwargs["top_k"])

    with patch("musicmix.pool_search.search_directional", side_effect=search):
        result = search_pool(tracks, structures, vocals, top_k=2, max_per_pair=2)

    selected = {record["candidate"]["candidate_id"] for record in result["selected"]}
    admitted = {record["candidate"]["candidate_id"] for record in result["proposals"]}
    assert "cand_hash_a_hash_c_2100_2600_first" in selected
    assert "cand_hash_a_hash_b_2000_2500_blocked" in admitted
    assert "cand_hash_a_hash_b_2000_2500_blocked" not in selected
    assert "cand_hash_a_hash_b_4000_4500_fallback" in selected
    assert result["diagnostics"]["selection_rejection_counts"]["cue_spacing"] >= 1


def test_result_is_json_compatible_and_records_pair_counts() -> None:
    tracks, structures, vocals = _fixture()
    with patch("musicmix.pool_search.search_directional", side_effect=_fake_search([])):
        result = search_pool(tracks, structures, vocals, top_k=8, max_per_pair=2,
                             cue_policy="beats", max_candidates_per_pair=500)

    json.dumps(result)
    assert result["diagnostics"]["cue_policy"] == "beats"
    assert result["diagnostics"]["max_candidates_per_pair"] == 500
    assert sum(item["admitted_unique_count"] for item in result["all_pair_counts"].values()) == 36
    assert all(item["selected_count"] >= 0 for item in result["all_pair_counts"].values())
    assert sum(item["selected_count"] for item in result["all_pair_counts"].values()) == len(result["selected"])
    first_pair = result["all_pair_counts"]["hash_a->hash_b"]
    # These names are the actual directional diagnostics, not invented aliases.
    assert first_pair["candidate_grid_size"] == 12
    assert first_pair["candidate_count_evaluated"] == 12
    assert first_pair["proposals_scored"] == 3
    assert first_pair["admitted_count"] == 3
    assert first_pair["directional_selected_count"] == 2


def test_requires_three_distinct_canonical_tracks() -> None:
    tracks, structures, vocals = _fixture()
    two_tracks = {key: tracks[key] for key in TRACK_IDS[:2]}
    two_structures = {key: structures[key] for key in TRACK_IDS[:2]}
    two_vocals = {key: vocals[key] for key in TRACK_IDS[:2]}
    with pytest.raises(ValueError, match="at least three"):
        search_pool(two_tracks, two_structures, two_vocals)

    wrong_key = {**tracks, "not_the_hash": _track("not_the_hash")}
    with pytest.raises(ValueError, match="canonical"):
        search_pool(wrong_key, structures, vocals)
