"""Cache trajectory profiles for every admitted structured V4 proposal.

The V7 cache is an observational feature artifact.  It reads the ``proposals``
arrays from the forward/reverse V4 search outputs, keeps each candidate's
baseline fields and cue alignment, and stores deduplicated trajectory profiles
separately.  No candidate is reranked and no listening judgment is used.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, IO, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from musicmix.audio import AudioTrack, load_audio  # noqa: E402
from musicmix.paths import localize  # noqa: E402
from musicmix.trajectory import (  # noqa: E402
    DEFAULT_WINDOWS_BEATS,
    TRAJECTORY_CONFIG_HASH,
    TRAJECTORY_VERSION,
    extract_trajectory,
    trajectory_to_dict,
)


SCHEMA_VERSION = "musicmix.candidate-trajectories.v7"
RUN_SPECS = (
    ("forward", ROOT / "runs" / "structured_forward_v4" / "search.json"),
    ("reverse", ROOT / "runs" / "structured_reverse_v4" / "search.json"),
)
CANONICAL_AUDIO = ROOT / "runs" / "reminder_to_too_many_nights" / "audio"
STRUCTURE_DIR = ROOT / "runs" / "structure_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


TRAJECTORY_SOURCE_SHA256 = _sha256(SRC / "musicmix" / "trajectory.py")
TOOL_SOURCE_SHA256 = _sha256(Path(__file__).resolve())


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or abs(number) == float("inf"):
        return None
    return number


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_write(handle: IO[str], record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    handle.write("\n")
    handle.flush()


def _load_checkpoint(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], set[tuple[str, int, int, Any]]]:
    profiles: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, Any]] = {}
    dedup_seen: set[tuple[str, int, int, Any]] = set()
    if not path.exists():
        return profiles, candidates, dedup_seen
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, Mapping):
                continue
            kind = record.get("type")
            if kind == "profile" and isinstance(record.get("payload"), Mapping):
                key = str(record.get("key") or "")
                if key:
                    profiles[key] = dict(record["payload"])
            elif kind == "candidate" and isinstance(record.get("entry"), Mapping):
                candidate_id = str(record.get("candidate_id") or "")
                if candidate_id:
                    entry = dict(record["entry"])
                    candidates[candidate_id] = entry
                    cue = entry.get("cue_pair_window_key")
                    if isinstance(cue, list) and len(cue) == 3:
                        dedup_seen.add((str(entry.get("direction") or ""), int(cue[0]), int(cue[1]), cue[2]))
    return profiles, candidates, dedup_seen


def _load_reuse_profiles(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    try:
        data = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, Mapping) or not isinstance(data.get("profile_cache"), Mapping):
        return {}
    profiles: dict[str, dict[str, Any]] = {}
    for key, payload in data["profile_cache"].items():
        if not isinstance(payload, Mapping):
            continue
        copy = dict(payload)
        provenance = copy.get("provenance")
        provenance = dict(provenance) if isinstance(provenance, Mapping) else {}
        provenance.setdefault("reused_from", str(path))
        provenance.setdefault("trajectory_source_sha256", TRAJECTORY_SOURCE_SHA256)
        provenance.setdefault("trajectory_config_hash", TRAJECTORY_CONFIG_HASH)
        provenance["reuse_note"] = "profile loaded from prior V7 downbeat cache; trajectory module hash retained"
        copy["provenance"] = provenance
        profiles[str(key)] = copy
    return profiles


def _structure_beats(structure_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    beats_by_track: dict[str, list[dict[str, Any]]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for path in sorted(structure_dir.glob("*.json")):
        try:
            data = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, Mapping):
            continue
        track_id = str(data.get("track_id") or "")
        if not track_id:
            continue
        beats: list[dict[str, Any]] = []
        for event in data.get("beats", ()) or ():
            if not isinstance(event, Mapping):
                continue
            sample = _number(event.get("sample"))
            if sample is None:
                continue
            beats.append(
                {
                    "sample": int(round(sample)),
                    "confidence": _number(event.get("confidence")),
                }
            )
        beats_by_track[track_id] = beats
        provenance[track_id] = {
            "path": str(path),
            "sha256": _sha256(path),
            "schema_version": data.get("schema_version"),
            "beat_count": len(beats),
            "provider": data.get("provenance", {}).get("provider")
            if isinstance(data.get("provenance"), Mapping)
            else None,
        }
    return beats_by_track, provenance


def _audio_specs(
    search_data: Mapping[str, Any], audio_dir: Path, structure_dir: Path
) -> dict[str, Path]:
    specs: dict[str, Path] = {}
    for item in search_data.get("tracks", ()) or ():
        if not isinstance(item, Mapping):
            continue
        track_id = str(item.get("sha256") or item.get("track_id") or "")
        if not track_id:
            continue
        path = localize(str(item.get("path") or ""))
        if not path.exists():
            path = audio_dir / ("a.wav" if len(specs) == 0 else "b.wav")
        specs[track_id] = path
    # Search outputs do not repeat the canonical track list.  Resolve IDs from
    # the already-produced structure JSONs without decoding the WAV here; the
    # caller then loads each canonical track exactly once.
    if not specs:
        for name in ("a.wav", "b.wav"):
            path = audio_dir / name
            structure_path = structure_dir / f"{Path(name).stem}.json"
            try:
                structure = _read_json(structure_path)
            except (OSError, json.JSONDecodeError):
                structure = {}
            track_id = str(structure.get("track_id") or "") if isinstance(structure, Mapping) else ""
            if path.exists() and track_id:
                specs[track_id] = path
    return specs


def _profile_key(track_id: str, side: str, requested: int, shift: int) -> str:
    return f"{track_id}:{side}:requested={int(requested)}:shift={int(shift)}"


def _validate_profile(
    payload: Mapping[str, Any],
    *,
    actual_sample: int,
    requested_sample: int,
    shift: int,
    track: AudioTrack,
) -> dict[str, Any]:
    errors: list[str] = []
    if payload.get("cue_sample") != int(actual_sample):
        errors.append("effective_cue_mismatch")
    if payload.get("requested_cue_sample") != int(requested_sample):
        errors.append("requested_cue_mismatch")
    if payload.get("cue_shift_samples") != int(shift):
        errors.append("cue_shift_mismatch")
    if int(payload.get("sample_rate", -1)) != int(track.sample_rate):
        errors.append("sample_rate_mismatch")
    for window in payload.get("windows", ()) or ():
        if not isinstance(window, Mapping):
            errors.append("invalid_window")
            continue
        start = int(window.get("start_sample", -1))
        end = int(window.get("end_sample", -1))
        if start < 0 or end < start or end > track.duration_samples:
            errors.append("window_out_of_bounds")
        if int(window.get("duration_samples", -1)) != end - start:
            errors.append("window_duration_mismatch")
        series = window.get("beat_feature_series")
        if not isinstance(series, list) or len(series) != int(window.get("window_beats", 0)):
            errors.append("beat_series_length_mismatch")
    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "confidence": payload.get("diagnostics", {}).get("confidence"),
        "confidence_status": payload.get("diagnostics", {}).get("confidence_status"),
    }


def _make_profile(
    *,
    key: str,
    track: AudioTrack,
    track_path: Path,
    beats: Sequence[Any],
    structure_provenance: Mapping[str, Any],
    side: str,
    requested_sample: int,
    actual_sample: int,
    shift: int,
) -> dict[str, Any]:
    result = extract_trajectory(
        track,
        requested_sample,
        beats,
        side=side,
        windows_beats=DEFAULT_WINDOWS_BEATS,
        cue_shift_samples=shift,
    )
    payload = trajectory_to_dict(result)
    validation = _validate_profile(
        payload,
        actual_sample=actual_sample,
        requested_sample=requested_sample,
        shift=shift,
        track=track,
    )
    payload["provenance"] = {
        "cache_key": key,
        "source_path": str(track_path),
        "source_track_id": track.track_id,
        "source_audio_sha256": _sha256(track_path),
        "structure": dict(structure_provenance),
        "trajectory_module_version": TRAJECTORY_VERSION,
        "trajectory_source_sha256": TRAJECTORY_SOURCE_SHA256,
        "tool_source_sha256": TOOL_SOURCE_SHA256,
        "trajectory_config_hash": TRAJECTORY_CONFIG_HASH,
        "parameters": {
            "side": side,
            "windows_beats": list(DEFAULT_WINDOWS_BEATS),
            "requested_cue_sample": int(requested_sample),
            "actual_cue_sample": int(actual_sample),
            "cue_shift_samples": int(shift),
            "canonical_sample_rate": int(track.sample_rate),
        },
        "validation": validation,
    }
    return payload


def _candidate_entry(
    proposal: Mapping[str, Any],
    *,
    direction: str,
    run_path: Path,
    tracks: Mapping[str, AudioTrack],
    track_paths: Mapping[str, Path],
    beats: Mapping[str, Sequence[Any]],
    structure_provenance: Mapping[str, Mapping[str, Any]],
    profile_cache: dict[str, dict[str, Any]],
    stats: dict[str, int],
    checkpoint_handle: IO[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    candidate = proposal.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError(f"proposal has no candidate: {run_path}")
    candidate_id = str(candidate.get("candidate_id") or "")
    track_a_id = str(candidate.get("track_a") or "")
    track_b_id = str(candidate.get("track_b") or "")
    if not candidate_id or track_a_id not in tracks or track_b_id not in tracks:
        raise ValueError(f"invalid candidate/tracks in {run_path}: {candidate_id}")
    raw = proposal.get("raw_features")
    raw = raw if isinstance(raw, Mapping) else {}
    originals = raw.get("original_structure_cue_samples")
    originals = originals if isinstance(originals, Mapping) else {}
    actual_a = int(round(float(candidate["cue_out_sample"])))
    actual_b = int(round(float(candidate["cue_in_sample"])))
    requested_a = int(round(float(originals.get("a", actual_a))))
    requested_b = int(round(float(originals.get("b", actual_b))))
    shift_a = actual_a - requested_a
    shift_b = actual_b - requested_b
    alignment = proposal.get("alignment")
    alignment = alignment if isinstance(alignment, Mapping) else {}
    alignment_a = alignment.get("outgoing")
    alignment_b = alignment.get("incoming")
    alignment_a = alignment_a if isinstance(alignment_a, Mapping) else {}
    alignment_b = alignment_b if isinstance(alignment_b, Mapping) else {}
    alignment_validation = {
        "a_actual_minus_original": shift_a,
        "b_actual_minus_original": shift_b,
        "a_alignment_shift": alignment_a.get("shift_samples"),
        "b_alignment_shift": alignment_b.get("shift_samples"),
        "a_matches_alignment": alignment_a.get("shift_samples") is None
        or int(alignment_a.get("shift_samples")) == shift_a,
        "b_matches_alignment": alignment_b.get("shift_samples") is None
        or int(alignment_b.get("shift_samples")) == shift_b,
    }

    profile_specs = (
        ("a_before", track_a_id, "outgoing", requested_a, actual_a, shift_a),
        ("a_natural_after", track_a_id, "incoming", requested_a, actual_a, shift_a),
        ("b_after", track_b_id, "incoming", requested_b, actual_b, shift_b),
    )
    references: dict[str, str] = {}
    for name, track_id, side, requested, actual, shift in profile_specs:
        key = _profile_key(track_id, side, requested, shift)
        references[name] = key
        if key not in profile_cache:
            profile_cache[key] = _make_profile(
                key=key,
                track=tracks[track_id],
                track_path=track_paths[track_id],
                beats=beats.get(track_id, ()),
                structure_provenance=structure_provenance.get(track_id, {}),
                side=side,
                requested_sample=requested,
                actual_sample=actual,
                shift=shift,
            )
            stats["profile_cache_misses"] += 1
            if checkpoint_handle is not None:
                _checkpoint_write(
                    checkpoint_handle,
                    {"type": "profile", "key": key, "payload": profile_cache[key]},
                )
                if stats["profile_cache_misses"] % 20 == 0:
                    print(
                        f"[trajectory-v7] new profile {stats['profile_cache_misses']} "
                        f"(cache size {len(profile_cache)})",
                        flush=True,
                    )
        else:
            stats["profile_cache_hits"] += 1
    entry = {
        "candidate_id": candidate_id,
        "direction": direction,
        "source_search": str(run_path),
        "cue_pair_window_key": [actual_a, actual_b, raw.get("window_beats")],
        "cue_samples": {
            "a_original": requested_a,
            "a_final": actual_a,
            "a_shift_samples": shift_a,
            "b_original": requested_b,
            "b_final": actual_b,
            "b_shift_samples": shift_b,
        },
        "profiles": references,
        "baseline": {
            "score": proposal.get("score"),
            "breakdown": proposal.get("breakdown"),
            "raw_features": raw,
            "proposal": proposal,
            "alignment": alignment,
            "reasons": proposal.get("reasons"),
        },
        "alignment_validation": alignment_validation,
    }
    return candidate_id, entry


def build_report(
    *,
    output: Path,
    runs: Sequence[tuple[str, Path]] = RUN_SPECS,
    audio_dir: Path = CANONICAL_AUDIO,
    structure_dir: Path = STRUCTURE_DIR,
    limit: int | None = None,
    reuse_cache: Path | None = None,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    loaded_search: list[tuple[str, Path, Mapping[str, Any]]] = []
    all_audio_specs: dict[str, Path] = {}
    run_counts: dict[str, dict[str, int]] = {}
    for direction, search_path in runs:
        data = _read_json(search_path)
        if not isinstance(data, Mapping):
            raise ValueError(f"search output must be an object: {search_path}")
        loaded_search.append((direction, search_path, data))
        all_audio_specs.update(_audio_specs(data, audio_dir, structure_dir))
        proposals = data.get("proposals")
        run_counts[direction] = {
            "proposals_in_search": len(proposals) if isinstance(proposals, list) else 0,
            "selected_in_search": len(data.get("selected", ()) or ()),
            "rejected_in_search": len(data.get("rejected", ()) or ()),
        }
    beats, structure_provenance = _structure_beats(structure_dir)
    tracks: dict[str, AudioTrack] = {}
    track_paths: dict[str, Path] = {}
    for track_id, path in all_audio_specs.items():
        track = load_audio(path)
        if track.track_id != track_id:
            raise ValueError(f"canonical audio hash mismatch for {path}: expected {track_id}, got {track.track_id}")
        tracks[track_id] = track
        track_paths[track_id] = path

    checkpoint_path = checkpoint
    if checkpoint_path is None:
        checkpoint_path = output.with_suffix(".checkpoint.jsonl")
    profile_cache, candidate_map, dedup_seen = _load_checkpoint(checkpoint_path)
    reused_profiles = _load_reuse_profiles(reuse_cache)
    for key, payload in reused_profiles.items():
        profile_cache.setdefault(key, payload)
    duplicate_count = 0
    stats = {
        "profile_cache_hits": 0,
        "profile_cache_misses": 0,
        "profile_cache_reused": len(reused_profiles),
    }
    processed = len(candidate_map)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_handle = checkpoint_path.open("a", encoding="utf-8", newline="\n")
    try:
        if checkpoint_path.stat().st_size == 0:
            _checkpoint_write(
                checkpoint_handle,
                {
                    "type": "meta",
                    "schema_version": SCHEMA_VERSION,
                    "trajectory_module_version": TRAJECTORY_VERSION,
                    "trajectory_source_sha256": TRAJECTORY_SOURCE_SHA256,
                    "tool_source_sha256": TOOL_SOURCE_SHA256,
                },
            )
        for direction, search_path, data in loaded_search:
            proposals = data.get("proposals")
            if not isinstance(proposals, list):
                continue
            for proposal in proposals:
                if not isinstance(proposal, Mapping):
                    continue
                candidate = proposal.get("candidate")
                raw = proposal.get("raw_features")
                if not isinstance(candidate, Mapping) or not isinstance(raw, Mapping):
                    continue
                key = (
                    direction,
                    int(candidate.get("cue_out_sample", -1)),
                    int(candidate.get("cue_in_sample", -1)),
                    raw.get("window_beats"),
                )
                if key in dedup_seen:
                    duplicate_count += 1
                    continue
                dedup_seen.add(key)
                if limit is not None and processed >= int(limit):
                    continue
                candidate_id, entry = _candidate_entry(
                    proposal,
                    direction=direction,
                    run_path=search_path,
                    tracks=tracks,
                    track_paths=track_paths,
                    beats=beats,
                    structure_provenance=structure_provenance,
                    profile_cache=profile_cache,
                    stats=stats,
                    checkpoint_handle=checkpoint_handle,
                )
                if candidate_id in candidate_map:
                    raise ValueError(f"duplicate candidate_id: {candidate_id}")
                candidate_map[candidate_id] = entry
                _checkpoint_write(
                    checkpoint_handle,
                    {"type": "candidate", "candidate_id": candidate_id, "entry": entry},
                )
                processed += 1
    finally:
        checkpoint_handle.close()

    pair_keys = set()
    for entry in candidate_map.values():
        cue = entry["cue_samples"]
        pair_keys.add((entry["direction"], cue["a_final"], cue["b_final"]))
    source_records = {
        track_id: {
            "path": str(track_paths[track_id]),
            "track_id": track.track_id,
            "source_audio_sha256": _sha256(track_paths[track_id]),
            "sample_rate": track.sample_rate,
            "duration_samples": track.duration_samples,
        }
        for track_id, track in tracks.items()
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if limit is None else "partial_smoke_limit",
        "trajectory_module_version": TRAJECTORY_VERSION,
        "code_provenance": {
            "trajectory_source_sha256": TRAJECTORY_SOURCE_SHA256,
            "tool_source_sha256": TOOL_SOURCE_SHA256,
            "hash_capture": "at_tool_start",
        },
        "trajectory_config_hash": TRAJECTORY_CONFIG_HASH,
        "parameters": {
            "windows_beats": list(DEFAULT_WINDOWS_BEATS),
            "profile_sides": ["a_before", "a_natural_after", "b_after"],
            "dedup_key": "direction + final cue_out_sample + final cue_in_sample + proposal window_beats",
            "profile_cache_key": "track_id + side + original cue sample + cue shift samples",
            "cache_policy": "all admitted proposals; no listening or score preselection",
        },
        "search_sources": [
            {
                "direction": direction,
                "path": str(path),
                "sha256": _sha256(path),
                "proposals_in_search": counts["proposals_in_search"],
                "selected_in_search": counts["selected_in_search"],
                "rejected_in_search": counts["rejected_in_search"],
            }
            for (direction, path, _data), counts in zip(loaded_search, run_counts.values())
        ],
        "counts": {
            "candidate_records": len(candidate_map),
            "unique_directional_cue_pairs": len(pair_keys),
            "unique_profile_cache_entries": len(profile_cache),
            "duplicate_cue_pair_window_records_skipped": duplicate_count,
            **stats,
        },
        "source_records": source_records,
        "structure_sources": structure_provenance,
        "confidence_policy": {
            "value": None,
            "status": "unknown_unvalidated",
            "reason": "trajectory descriptors have no calibrated perceptual confidence model",
        },
        "candidate_map": candidate_map,
        "profile_cache": profile_cache,
        "limitations": [
            "Acoustic profiles are descriptive observables; they do not infer suspense, tension, or resolution.",
            "Pitch contour remains unknown because no physical pitch provider is configured.",
            "Missing local/context beat windows are represented by fewer windows and null context proxies; no padding or imputation is performed.",
            "Search score, listening judgments, ASR, and vocal evidence are preserved as provenance only and are not used as trajectory weights.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Compact JSON and gzip keep the all-beat pool below the constrained
    # workspace disk budget; the schema remains ordinary JSON after reading.
    if output.suffix.lower() == ".gz":
        with gzip.open(output, "wt", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, separators=(",", ":"))
    else:
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, separators=(",", ":"))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "candidate_trajectories_v7.json")
    parser.add_argument("--audio-dir", type=Path, default=CANONICAL_AUDIO)
    parser.add_argument("--structure-dir", type=Path, default=STRUCTURE_DIR)
    parser.add_argument("--reuse-cache", type=Path, default=ROOT / "runs" / "candidate_trajectories_v7.json")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--search",
        nargs="+",
        type=Path,
        default=None,
        help="one or more search.json files; direction is inferred from parent directory",
    )
    parser.add_argument("--limit", type=int, default=None, help="process only N deduplicated candidates for a smoke run")
    args = parser.parse_args(argv)
    run_specs = RUN_SPECS
    if args.search:
        inferred: list[tuple[str, Path]] = []
        for path in args.search:
            name = path.parent.name.lower()
            direction = "reverse" if "reverse" in name else "forward"
            inferred.append((direction, path))
        run_specs = tuple(inferred)
    report = build_report(
        output=args.output,
        runs=run_specs,
        audio_dir=args.audio_dir,
        structure_dir=args.structure_dir,
        limit=args.limit,
        reuse_cache=args.reuse_cache if args.reuse_cache != args.output else None,
        checkpoint=args.checkpoint,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report["status"],
                "counts": report["counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
