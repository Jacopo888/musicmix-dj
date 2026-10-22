"""Tests for the provenance-preserving V11 opportunity registry builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.build_vocal_opportunities_v11 import _native_cues, build_registry, sha256_file


def _track_fixture(tmp_path: Path, slug: str, identity: str) -> dict:
    vocal_path = tmp_path / f"{slug}_vocals.json"
    vocal = {
        "schema_version": "musicmix.vocal_boundaries.2",
        "track_id": identity,
        "canonical_file_sha256": identity,
        "canonical_audio_uri": str(tmp_path / f"{slug}.wav"),
        "sample_rate": 48_000,
        "duration_samples": 48_000 * 120,
        "asr_segments": [
            {"start_sample": 48_000 * 10, "end_sample": 48_000 * 12, "confidence": 0.91},
            {"start_sample": 48_000 * 20, "end_sample": 48_000 * 21, "confidence": 0.82},
        ],
        "words": [],
    }
    vocal_path.write_text(json.dumps(vocal), encoding="utf-8")
    return {
        "slug": slug, "track_id": identity, "sample_rate": 48_000,
        "duration_samples": 48_000 * 120, "canonical_path": str(tmp_path / f"{slug}.wav"),
        "vocals_path": str(vocal_path), "vocals_sha256": sha256_file(vocal_path),
    }


def _catalog(tmp_path: Path) -> Path:
    tracks = [_track_fixture(tmp_path, "a", "a" * 64), _track_fixture(tmp_path, "b", "b" * 64)]
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"status": "ready", "tracks": tracks}), encoding="utf-8")
    return path


def test_registry_retains_native_roles_and_exact_identity(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    output = tmp_path / "opportunities.json"
    result = build_registry(catalog, output)
    assert result["status"] == "complete"
    assert len(result["tracks"]) == 2
    first = result["tracks"][0]
    roles = {cue["role"] for cue in first["cues"]}
    assert roles == {"incoming", "outgoing"}
    assert result["counts"]["native_asr_observations"] == 8
    assert all("phrase_closure" not in cue for cue in first["cues"])
    evidence = first["inference_evidence"]
    assert evidence["canonical_sha256"] == first["track_id"]
    assert hashlib.sha256(Path(evidence["artifact_path"]).read_bytes()).hexdigest() == evidence["artifact_sha256"]
    assert output.exists()


def test_provider_cues_are_merged_without_erasing_native_provenance(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    catalog_doc = json.loads(catalog.read_text(encoding="utf-8"))
    track = catalog_doc["tracks"][0]
    provider_artifact = tmp_path / "cue_artifact.json"
    provider_artifact.write_text("{\"source\":\"pinned\"}\n", encoding="utf-8")
    provider = {
        "schema_version": "musicmix.learned-opportunity-registry.1", "status": "complete",
        "tracks": [{
            "track_id": track["track_id"], "sample_rate": track["sample_rate"],
            "duration_samples": track["duration_samples"], "cues": [{
                "time_seconds": 30.0, "label": "provider_boundary"
            }], "inference_evidence": {
                "canonical_sha256": track["track_id"], "artifact_path": str(provider_artifact),
                "artifact_sha256": sha256_file(provider_artifact),
            }
        }]
    }
    provider_path = tmp_path / "provider.json"
    provider_path.write_text(json.dumps(provider), encoding="utf-8")
    result = build_registry(catalog, tmp_path / "merged.json", cue_registry=provider_path)
    cues = result["tracks"][0]["cues"]
    provider_cue = next(cue for cue in cues if cue["sample"] == 48_000 * 30)
    assert provider_cue["provider"] == "cue_detr"
    assert provider_cue["role"] == "provider_opportunity"
    assert provider_cue["inference_evidence"]["artifact_sha256"] == sha256_file(provider_artifact)
    assert result["counts"]["cue_detr_observations"] == 1


def test_changed_vocal_artifact_and_existing_output_fail_closed(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    catalog_doc = json.loads(catalog.read_text(encoding="utf-8"))
    vocal_path = Path(catalog_doc["tracks"][0]["vocals_path"])
    vocal_path.write_text(vocal_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed or missing vocals"):
        build_registry(catalog, tmp_path / "changed.json")
    output = tmp_path / "existing.json"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_registry(_catalog(tmp_path), output)
    assert output.read_text(encoding="utf-8") == "keep"


def test_history_filter_excludes_exposed_side_but_keeps_fresh_observation(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    history = tmp_path / "history.json"
    first_id = "a" * 64
    history.write_text(json.dumps({
        "schema_version": "musicmix.audition-history.1",
        "regions": [{
            "track_a": first_id,
            "track_b": first_id,
            "cue_out_sample": 48_000 * 10,
            "cue_in_sample": 48_000 * 10,
        }],
    }), encoding="utf-8")
    result = build_registry(catalog, tmp_path / "fresh.json", history_path=history)
    first = result["tracks"][0]
    samples = {cue["sample"] for cue in first["cues"]}
    assert 48_000 * 10 not in samples
    assert 48_000 * 20 in samples
    assert result["counts"]["history_rejected_cues"] >= 1
    assert result["history_filter"]["tolerance_seconds"] == 8.0


def test_word_gap_endpoints_are_timing_only_observations() -> None:
    entry = {"slug": "gap", "duration_samples": 48_000 * 60}
    cues = _native_cues({
        "asr_segments": [{"start_sample": 48_000, "end_sample": 48_000 * 4}],
        "words": [
            {"start_sample": 48_000, "end_sample": 48_000 * 2, "confidence": 0.9},
            {"start_sample": 48_000 * 3, "end_sample": 48_000 * 4, "confidence": 0.8},
        ],
    }, entry)
    gap_cues = [cue for cue in cues if cue["provider"] == "native_observed_asr_word_gap"]
    assert {cue["role"] for cue in gap_cues} == {"incoming", "outgoing"}
    assert all(cue["role_semantics"].endswith("timing only") for cue in gap_cues)
    assert all("no phrase closure" in cue["observation"] for cue in gap_cues)
    assert all("acoustic silence claim" in cue["observation"] for cue in gap_cues)

