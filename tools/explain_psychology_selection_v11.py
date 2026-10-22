"""Persist scientific traceability for the six immutable V11 selections.

The release manifest and rendered artifacts are read only.  For each main
selection this tool replays the source-clock trajectory extraction, expectation
evidence, vocal timing evidence, and typed transition decision, then writes a
separate evidence document bound to the release/catalog/module hashes.

The output is descriptive: domain directions and missing arrival evidence are
reported as observations.  ASR segment occupancy is never called a phrase
closure or syntactic completion.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from musicmix.paths import localize  # noqa: E402
RATE = 48_000
SCHEMA = "musicmix.psychology-selection-explanation.1"
TOOL_VERSION = "psychology-selection-explanation-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _json(value: Any) -> Any:
    if is_dataclass(value):
        return _json(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _resolve(path_value: str | Path, base: Path) -> Path:
    path = localize(path_value).expanduser()
    return path if path.is_absolute() else (base / path)


def _module_paths(code_hashes: Mapping[str, Any]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name in code_hashes:
        relative = str(name).replace("\\", "/")
        path = (ROOT / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"manifest module is missing: {path}")
        result[str(name)] = path
    return result


def _check_module_hashes(code_hashes: Mapping[str, Any], release_dir: Path) -> dict[str, Any]:
    paths = _module_paths(code_hashes)
    actual = {name: sha256_file(path) for name, path in paths.items()}
    expected = {str(name): str(value) for name, value in code_hashes.items()}
    archived_paths = {
        name: (release_dir / "source" / name.replace("\\", "/")).resolve()
        for name in expected
    }
    archived_actual = {
        name: sha256_file(path) for name, path in archived_paths.items() if path.is_file()
    }
    current_matches = [name for name in expected if expected[name] == actual.get(name)]
    archived_matches = [name for name in expected if expected[name] == archived_actual.get(name)]
    mismatches = [name for name in expected if name not in current_matches and name not in archived_matches]
    return {
        "status": "pass" if not mismatches else "fail",
        "expected": expected,
        "current_actual": actual,
        "archived_actual": archived_actual,
        "current_matches": current_matches,
        "archived_matches": archived_matches,
        "archived_source_used_for": [name for name in archived_matches if name not in current_matches],
        "mismatches": mismatches,
    }


def _track_index(catalog: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    tracks = catalog.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("catalog tracks must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    for entry in tracks:
        if not isinstance(entry, Mapping):
            raise ValueError("catalog track must be an object")
        identity = str(entry.get("track_id"))
        if identity in result:
            raise ValueError(f"duplicate catalog track: {identity}")
        result[identity] = entry
    return result


def _validate_track(entry: Mapping[str, Any], catalog_path: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from musicmix.audio import load_audio

    audio_path = _resolve(str(entry["canonical_path"]), catalog_path.parent)
    structure_path = _resolve(str(entry["structure_path"]), catalog_path.parent)
    vocals_path = _resolve(str(entry["vocals_path"]), catalog_path.parent)
    audio_hash = sha256_file(audio_path)
    structure_hash = sha256_file(structure_path)
    vocals_hash = sha256_file(vocals_path)
    if audio_hash != entry.get("track_id"):
        raise ValueError(f"canonical PCM hash/identity mismatch: {entry.get('slug')}")
    source_path_value = entry.get("source_path")
    source_hash = None
    if source_path_value:
        source_path = _resolve(str(source_path_value), catalog_path.parent)
        if source_path.is_file():
            source_hash = sha256_file(source_path)
            if source_hash != entry.get("source_sha256"):
                raise ValueError(f"original source hash mismatch: {entry.get('slug')}")
    if structure_hash != entry.get("structure_sha256"):
        raise ValueError(f"structure hash mismatch: {entry.get('slug')}")
    if vocals_hash != entry.get("vocals_sha256"):
        raise ValueError(f"vocals hash mismatch: {entry.get('slug')}")
    track = load_audio(audio_path)
    expected_identity = (entry.get("track_id"), RATE, entry.get("duration_samples"))
    actual_identity = (track.track_id, track.sample_rate, track.duration_samples)
    if actual_identity != expected_identity:
        raise ValueError(f"canonical source clock mismatch: {entry.get('slug')}")
    structure = _read(structure_path)
    vocals = _read(vocals_path)
    for document, name in ((structure, "structure"), (vocals, "vocals")):
        for field in ("track_id", "sample_rate", "duration_samples"):
            if document.get(field) != entry.get(field):
                raise ValueError(f"{name} identity mismatch for {entry.get('slug')}: {field}")
    beats = structure.get("beats")
    if not isinstance(beats, list) or len(beats) < 2:
        raise ValueError(f"insufficient source beats: {entry.get('slug')}")
    beat_samples = []
    for beat in beats:
        if not isinstance(beat, Mapping):
            raise ValueError(f"invalid source beat: {entry.get('slug')}")
        sample = beat.get("sample")
        if isinstance(sample, bool) or not isinstance(sample, (int, float)):
            raise ValueError(f"invalid source beat sample: {entry.get('slug')}")
        beat_samples.append(int(round(sample)))
    return track, structure, vocals


def _candidate(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate = artifact.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("main artifact has no candidate")
    return candidate


def _direction_summary(expectation: Any) -> dict[str, Any]:
    profiles = expectation.diagnostics.get("profiles", {}) if isinstance(expectation.diagnostics, Mapping) else {}
    domains = ("energy", "events", "bass", "timbre", "tonal_proxy")
    summary: dict[str, Any] = {}
    for name in ("A_before", "A_natural_after", "B_after"):
        raw = profiles.get(name, {}) if isinstance(profiles, Mapping) else {}
        domain_values = raw.get("domains", {}) if isinstance(raw, Mapping) else {}
        summary[name] = {
            domain: {
                "direction": domain_values.get(domain, {}).get("direction") if isinstance(domain_values.get(domain), Mapping) else None,
                "score": domain_values.get(domain, {}).get("score") if isinstance(domain_values.get(domain), Mapping) else None,
            }
            for domain in domains
        }
    a = summary.get("A_before", {})
    b = summary.get("B_after", {})
    matching: list[str] = []
    mismatching: list[str] = []
    for domain in domains:
        a_direction = a[domain]["direction"]
        b_direction = b[domain]["direction"]
        if a_direction in {"up", "down"} and a_direction == b_direction:
            matching.append(domain)
        elif a_direction is not None or b_direction is not None:
            mismatching.append(domain)
    observed = [
        value["direction"]
        for side in ("A_before", "B_after")
        for value in summary.get(side, {}).values()
        if value.get("direction") in {"up", "down", "weak", "flat"}
    ]
    falling_or_weak = bool(observed) and all(value in {"down", "weak"} for value in observed)
    return {
        "profiles": summary,
        "A_to_B_matching_up_down_domains": matching,
        "A_to_B_nonmatching_or_unresolved_domains": mismatching,
        "descriptive_flags": ["all_observed_directions_falling_or_weak"] if falling_or_weak else [],
        "semantics": "direction labels are descriptive trajectory evidence, not psychological judgements",
    }


def explain_release(release_dir: str | Path) -> dict[str, Any]:
    from musicmix.expectation import analyze_expectation
    from musicmix.phrase_evidence import analyze_phrase_evidence
    from musicmix.pooled_trajectory import PooledTrajectory
    from musicmix.transition_intent import assess_transition

    release = Path(release_dir).expanduser().resolve()
    manifest_path = release / "manifest.json"
    catalog_path = release / "catalog_input.json"
    if not manifest_path.is_file() or not catalog_path.is_file():
        raise FileNotFoundError("release must contain manifest.json and catalog_input.json")
    manifest = _read(manifest_path)
    catalog = _read(catalog_path)
    module_hashes = _check_module_hashes(manifest.get("code_sha256", {}), release)
    if module_hashes["status"] != "pass":
        raise ValueError(f"release module hash mismatch: {module_hashes['mismatches']}")
    catalog_hash = sha256_file(catalog_path)
    if manifest.get("inputs", {}).get("catalog_sha256") != catalog_hash:
        raise ValueError("release catalog hash mismatch")
    entries = _track_index(catalog)
    loaded: dict[str, tuple[Any, dict[str, Any], dict[str, Any], PooledTrajectory]] = {}
    source_identity: dict[str, Any] = {}
    for identity, entry in entries.items():
        track, structure, vocals = _validate_track(entry, catalog_path)
        beats = [int(round(beat["sample"])) for beat in structure["beats"]]
        loaded[identity] = (track, structure, vocals, PooledTrajectory(track, beats))
        source_identity[identity] = {
            "slug": entry.get("slug"),
            "track_id": identity,
            "canonical_path": str(_resolve(str(entry["canonical_path"]), catalog_path.parent)),
            "canonical_file_sha256": sha256_file(_resolve(str(entry["canonical_path"]), catalog_path.parent)),
            "source_sha256": entry.get("source_sha256"),
            "structure_sha256": sha256_file(_resolve(str(entry["structure_path"]), catalog_path.parent)),
            "vocals_sha256": sha256_file(_resolve(str(entry["vocals_path"]), catalog_path.parent)),
            "sample_rate": RATE,
            "duration_samples": int(entry["duration_samples"]),
            "beat_count": len(beats),
        }

    mains = [item for item in manifest.get("artifacts", []) if item.get("variant") == "main"]
    if len(mains) != int(manifest.get("selected", len(mains))):
        raise ValueError("release main artifact count does not match selected count")
    output_dir = release / "evidence"
    explanation_manifest_path = release / "explanation_manifest.json"
    if output_dir.exists() or explanation_manifest_path.exists():
        raise FileExistsError("release explanation output already exists")
    output_dir.mkdir()
    evidence_entries: list[dict[str, Any]] = []
    try:
        for item in mains:
            candidate = _candidate(item)
            candidate_id = str(candidate["candidate_id"])
            track_a = str(candidate["track_a"])
            track_b = str(candidate["track_b"])
            if track_a not in loaded or track_b not in loaded:
                raise ValueError(f"selected candidate references unknown track: {candidate_id}")
            a_track, _a_structure, a_vocals, a_pool = loaded[track_a]
            b_track, _b_structure, b_vocals, b_pool = loaded[track_b]
            cue_a = int(candidate["cue_out_sample"])
            cue_b = int(candidate["cue_in_sample"])
            a_before = a_pool.profile(cue_a, side="outgoing")
            a_natural_after = a_pool.profile(cue_a, side="incoming")
            b_before = b_pool.profile(cue_b, side="outgoing")
            b_after = b_pool.profile(cue_b, side="incoming")
            expectation = analyze_expectation(a_before, a_natural_after, b_after)
            phrase_a = analyze_phrase_evidence(a_vocals, cue_a, RATE, a_track.duration_samples, "outgoing")
            phrase_b = analyze_phrase_evidence(b_vocals, cue_b, RATE, b_track.duration_samples, "incoming")
            release_assessment = item.get("assessment")
            if not isinstance(release_assessment, Mapping):
                raise ValueError(f"missing release assessment: {candidate_id}")
            replay = assess_transition(
                a_before, a_natural_after, b_before, b_after,
                phrase_a=phrase_a, phrase_b=phrase_b,
                baseline_score=release_assessment.get("baseline_score"),
            ).to_dict()
            equality = {
                "action": replay.get("action") == release_assessment.get("action"),
                "intent": replay.get("intent") == release_assessment.get("intent"),
                "expectation_label": expectation.label == release_assessment.get("diagnostics", {}).get("expectation_label"),
                "all_fields": (
                    replay.get("action") == release_assessment.get("action")
                    and replay.get("intent") == release_assessment.get("intent")
                    and expectation.label == release_assessment.get("diagnostics", {}).get("expectation_label")
                ),
            }
            if not equality["all_fields"]:
                raise ValueError(f"decision replay mismatch: {candidate_id}: {equality}")
            closure = expectation.closure_evidence
            missing_arrival = list(dict.fromkeys(
                list(closure.missing if closure is not None else ("closure_arrival_evidence",))
                + [item for item in expectation.missing if "arrival" in str(item) or "closure" in str(item)]
            ))
            artifact_path = _resolve(str(item["artifact"]["path"]), release)
            artifact_hash = sha256_file(artifact_path)
            if artifact_hash != item["artifact"].get("sha256"):
                raise ValueError(f"release artifact hash mismatch: {candidate_id}")
            evidence = {
                "schema_version": SCHEMA,
                "tool_version": TOOL_VERSION,
                "release_manifest_sha256": sha256_file(manifest_path),
                "release_catalog_sha256": catalog_hash,
                "candidate": _json(candidate),
                "source_identity": {"A": source_identity[track_a], "B": source_identity[track_b]},
                "source_clock": {
                    "sample_rate": RATE,
                    "A_cue_sample": cue_a,
                    "B_cue_sample": cue_b,
                    "A_cue_seconds": cue_a / RATE,
                    "B_cue_seconds": cue_b / RATE,
                    "profile_sides": {"A_before": "outgoing", "A_natural_after": "incoming",
                                      "B_before": "outgoing", "B_after": "incoming"},
                },
                "profiles": {
                    "A_before": _json(a_before),
                    "A_natural_after": _json(a_natural_after),
                    "B_before": _json(b_before),
                    "B_after": _json(b_after),
                },
                "expectation": _json(asdict(expectation)),
                "expectation_diagnostics": _json(expectation.diagnostics),
                "direction_explanation": _direction_summary(expectation),
                "phrase_evidence": {"A": _json(phrase_a), "B": _json(phrase_b),
                                    "phrase_closure_claim": "unknown unless explicitly provider-supported"},
                "missing_arrival_evidence": missing_arrival,
                "release_assessment": _json(release_assessment),
                "replayed_assessment": _json(replay),
                "decision_replay": equality,
                "rendered_artifact": {
                    "path": str(artifact_path),
                    "sha256": artifact_hash,
                    "manifest_sha256": item["artifact"].get("sha256"),
                },
                "module_hash_check": module_hashes,
                "semantics": {
                    "domain_directions": "descriptive trajectory evidence only",
                    "phrase_closure": "ASR segment occupancy and word timing do not certify syntactic or musical completion",
                    "arrival": "requires explicit event/stability evidence; missing arrival remains missing",
                },
            }
            evidence_path = output_dir / f"{candidate_id}.json"
            evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            evidence_entries.append({
                "candidate_id": candidate_id,
                "path": str(evidence_path.resolve()),
                "sha256": sha256_file(evidence_path),
                "action": release_assessment.get("action"),
                "intent": release_assessment.get("intent"),
                "expectation_label": expectation.label,
                "phrase_closure_status": "unknown" if closure is None or closure.status != "supported" else "supported",
                "decision_replay": equality,
            })
    except Exception:
        for path in output_dir.glob("*.json"):
            path.unlink()
        output_dir.rmdir()
        raise

    result = {
        "schema_version": SCHEMA,
        "status": "complete",
        "tool_version": TOOL_VERSION,
        "release_dir": str(release),
        "release_manifest": str(manifest_path),
        "release_manifest_sha256": sha256_file(manifest_path),
        "catalog": str(catalog_path),
        "catalog_sha256": catalog_hash,
        "module_hash_check": module_hashes,
        "selected_count": len(evidence_entries),
        "evidence": evidence_entries,
        "semantics": {
            "source_clock": "all samples are canonical 48 kHz source-clock coordinates",
            "selection": "recomputed trajectory and expectation evidence for existing release selections",
            "phrase_closure": "unknown for ASR segment occupancy unless explicit provider evidence supports it",
            "subjective_claims": "none; directions and missing evidence are descriptive",
        },
    }
    explanation_manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=ROOT / "runs/psychology_v11_release")
    args = parser.parse_args(argv)
    result = explain_release(args.release_dir)
    print(json.dumps({
        "status": result["status"],
        "selected_count": result["selected_count"],
        "output": str((Path(args.release_dir).resolve() / "explanation_manifest.json")),
        "module_hash_status": result["module_hash_check"]["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

