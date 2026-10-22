"""Build the read-only V6 trajectory audit for selected V4 cues.

This tool measures the canonical source WAVs around already selected cues.  It
does not rerank, render, or change candidate metadata.  The extra
``continuation_after_outgoing`` profile is a descriptive same-track comparison
for the outgoing source; it is not a target or a claim about how an incoming
track should sound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

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


DEFAULT_RUNS = (
    "structured_forward_v4",
    "structured_reverse_v4",
    "structured_forward_beats_v4",
    "structured_reverse_beats_v4",
)
CANONICAL_AUDIO = ROOT / "runs" / "reminder_to_too_many_nights" / "audio"
STRUCTURE_DIR = ROOT / "runs" / "structure_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or abs(number) == float("inf"):
        return None
    return number


def _track_specs(manifest: Mapping[str, Any], audio_dir: Path) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for item in manifest.get("tracks", ()) or ():
        if not isinstance(item, Mapping):
            continue
        track_id = str(item.get("sha256") or item.get("track_id") or "")
        if not track_id:
            continue
        path_value = str(item.get("path") or "")
        path = Path(path_value) if path_value else Path()
        if not path.exists():
            stem = "a.wav" if len(specs) == 0 else "b.wav"
            path = audio_dir / stem
        specs[track_id] = {
            "path": path,
            "manifest_sample_rate": item.get("sample_rate"),
            "manifest_duration_samples": item.get("duration_samples"),
        }
    return specs


def _structure_beats(structure_dir: Path) -> dict[str, list[dict[str, Any]]]:
    values: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(structure_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
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
            beats.append({"sample": int(round(sample)), "confidence": _number(event.get("confidence"))})
        values[track_id] = beats
    return values


def _proposal_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("proposal_*.json"), key=lambda path: path.name)


def _profile(
    track: AudioTrack,
    beats: Sequence[Any],
    requested_sample: int,
    actual_sample: int,
    *,
    side: str,
) -> dict[str, Any]:
    result = extract_trajectory(
        track,
        requested_sample,
        beats,
        side=side,
        windows_beats=DEFAULT_WINDOWS_BEATS,
        cue_shift_samples=int(actual_sample) - int(requested_sample),
    )
    payload = trajectory_to_dict(result)
    payload["confidence_status"] = "unknown_unvalidated"
    payload["provider"] = "none_for_trajectory_features"
    return payload


def _selection_entry(
    proposal: Mapping[str, Any],
    *,
    run_name: str,
    run_path: Path,
    tracks: Mapping[str, AudioTrack],
    beats: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    candidate = proposal.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError(f"missing candidate in {run_path}")
    raw = proposal.get("raw_features")
    raw = raw if isinstance(raw, Mapping) else {}
    candidate_id = str(candidate.get("candidate_id") or "")
    track_a_id = str(candidate.get("track_a") or "")
    track_b_id = str(candidate.get("track_b") or "")
    if track_a_id not in tracks or track_b_id not in tracks:
        raise ValueError(f"unknown candidate track in {run_path}: {candidate_id}")
    actual_a = int(round(float(candidate["cue_out_sample"])))
    actual_b = int(round(float(candidate["cue_in_sample"])))
    originals = raw.get("original_structure_cue_samples")
    originals = originals if isinstance(originals, Mapping) else {}
    requested_a = int(round(float(originals.get("a", actual_a))))
    requested_b = int(round(float(originals.get("b", actual_b))))
    result = {
        "run": run_name,
        "run_file": str(run_path),
        "rank": int(run_path.stem.split("_")[-1]),
        "candidate_id": candidate_id,
        "score": _number(proposal.get("score")),
        "direction": str(raw.get("direction") or "unknown"),
        "cue_samples": {
            "a_requested": requested_a,
            "a_final": actual_a,
            "a_shift": actual_a - requested_a,
            "b_requested": requested_b,
            "b_final": actual_b,
            "b_shift": actual_b - requested_b,
        },
        "profiles": {
            "a_outgoing": _profile(tracks[track_a_id], beats.get(track_a_id, ()), requested_a, actual_a, side="outgoing"),
            "b_incoming": _profile(tracks[track_b_id], beats.get(track_b_id, ()), requested_b, actual_b, side="incoming"),
            "continuation_after_outgoing": _profile(tracks[track_a_id], beats.get(track_a_id, ()), requested_a, actual_a, side="incoming"),
        },
        "interpretation_policy": "descriptive observables only; no suspense, resolution, or section inference",
    }
    return result


def _metric_summary(selections: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = (
        "rms_mean",
        "log_rms_mean",
        "spectral_flux_mean",
        "attack_density_per_second",
        "spectral_centroid_hz",
        "high_frequency_ratio",
        "bass_entry_delta",
        "chroma_movement",
        "chroma_local_context_similarity",
    )
    summary: dict[str, Any] = {}
    for profile_name in ("a_outgoing", "b_incoming", "continuation_after_outgoing"):
        rows: dict[str, Any] = {}
        for beats in DEFAULT_WINDOWS_BEATS:
            values: dict[str, list[float]] = {metric: [] for metric in metrics}
            for selection in selections:
                profiles = selection.get("profiles")
                profile = profiles.get(profile_name) if isinstance(profiles, Mapping) else None
                windows = profile.get("windows") if isinstance(profile, Mapping) else None
                for window in windows or ():
                    if not isinstance(window, Mapping) or int(window.get("window_beats", -1)) != beats:
                        continue
                    for metric in metrics:
                        value = _number(window.get(metric))
                        if value is not None:
                            values[metric].append(value)
            rows[str(beats)] = {
                metric: {
                    "count": len(numbers),
                    "median": statistics.median(numbers) if numbers else None,
                    "min": min(numbers) if numbers else None,
                    "max": max(numbers) if numbers else None,
                }
                for metric, numbers in values.items()
            }
        summary[profile_name] = rows
    return summary


def build_audit(
    *,
    runs: Sequence[str] = DEFAULT_RUNS,
    output: Path = ROOT / "runs" / "trajectory_audit_v6.json",
    audio_dir: Path = CANONICAL_AUDIO,
    structure_dir: Path = STRUCTURE_DIR,
) -> dict[str, Any]:
    track_cache: dict[str, AudioTrack] = {}
    beat_cache = _structure_beats(structure_dir)
    selections: list[dict[str, Any]] = []
    source_records: dict[str, dict[str, Any]] = {}
    run_counts: dict[str, int] = {}
    for run_name in runs:
        run_dir = ROOT / "runs" / run_name
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        specs = _track_specs(manifest, audio_dir)
        for track_id, spec in specs.items():
            if track_id not in track_cache:
                path = localize(spec["path"])
                track_cache[track_id] = load_audio(path)
                source_records[track_id] = {
                    "path": str(path),
                    "sha256": track_cache[track_id].track_id,
                    "source_file_sha256": _sha256(path),
                    "sample_rate": track_cache[track_id].sample_rate,
                    "duration_samples": track_cache[track_id].duration_samples,
                }
        run_entries: list[dict[str, Any]] = []
        for proposal_path in _proposal_files(run_dir):
            proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
            if isinstance(proposal, Mapping):
                run_entries.append(
                    _selection_entry(
                        proposal,
                        run_name=run_name,
                        run_path=proposal_path,
                        tracks=track_cache,
                        beats=beat_cache,
                    )
                )
        selections.extend(run_entries)
        run_counts[run_name] = len(run_entries)

    # The V3 profile is a separately marked observational reference requested
    # by the parent.  It is not folded into V4 counts, ranker inputs, or any
    # training/policy target.
    reference_run = ROOT / "runs" / "structured_forward_v3"
    reference_path = reference_run / "proposal_03.json"
    reference: dict[str, Any] | None = None
    if reference_path.exists():
        proposal = json.loads(reference_path.read_text(encoding="utf-8"))
        candidate = proposal.get("candidate", {})
        if isinstance(candidate, Mapping):
            track_a_id = str(candidate.get("track_a") or "")
            track_b_id = str(candidate.get("track_b") or "")
            if track_a_id in track_cache and track_b_id in track_cache:
                raw = proposal.get("raw_features")
                raw = raw if isinstance(raw, Mapping) else {}
                originals = raw.get("original_structure_cue_samples")
                originals = originals if isinstance(originals, Mapping) else {}
                requested_a = int(round(float(originals.get("a", candidate["cue_out_sample"]))))
                requested_b = int(round(float(originals.get("b", candidate["cue_in_sample"]))))
                actual_a = int(candidate["cue_out_sample"])
                actual_b = int(candidate["cue_in_sample"])
                reference = {
                    "run": "structured_forward_v3",
                    "candidate_id": candidate.get("candidate_id"),
                    "human_reference_annotation": "qualitative sufficient reference only; not a training or policy target",
                    "profiles": {
                        "a_outgoing": _profile(track_cache[track_a_id], beat_cache.get(track_a_id, ()), requested_a, actual_a, side="outgoing"),
                        "b_incoming": _profile(track_cache[track_b_id], beat_cache.get(track_b_id, ()), requested_b, actual_b, side="incoming"),
                    },
                    "cue_samples": {
                        "a_requested": requested_a,
                        "a_final": actual_a,
                        "a_shift": actual_a - requested_a,
                        "b_requested": requested_b,
                        "b_final": actual_b,
                        "b_shift": actual_b - requested_b,
                    },
                }

    report = {
        "schema_version": "musicmix.trajectory-audit.v6",
        "analysis_version": TRAJECTORY_VERSION,
        "analysis_config_hash": TRAJECTORY_CONFIG_HASH,
        "status": "complete",
        "selection_count": len(selections),
        "expected_v4_selection_count": 18,
        "run_counts": run_counts,
        "source_records": source_records,
        "beat_sources": {
            track_id: {"path": str(path), "sha256": _sha256(path)}
            for path in sorted(structure_dir.glob("*.json"))
            for track_id in [str(json.loads(path.read_text(encoding="utf-8")).get("track_id") or "")]
            if track_id
        },
        "windows_beats": list(DEFAULT_WINDOWS_BEATS),
        "confidence": {
            "value": None,
            "status": "unknown_unvalidated",
            "reason": "descriptors have no calibrated event or perceptual confidence model",
        },
        "provider": "none_for_pitch_or_semantic_structure",
        "measured_observables": [
            "RMS and natural-log RMS",
            "spectral centroid and high-frequency energy ratio",
            "spectral flux and deterministic attack-peak density",
            "bass ratio at window start/end and their difference",
            "mean 12-bin chroma, frame-to-frame chroma movement",
            "local-to-context chroma cosine similarity as tonal-stability proxy",
        ],
        "missing_or_explicitly_unknown": [
            "physical pitch contour provider",
            "calibrated confidence",
            "semantic section/chorus labels",
            "suspense, tension, or resolution inference",
            "human-listening validation or ranker changes",
        ],
        "selections": selections,
        "reference_profile": reference,
        "summary": _metric_summary(selections),
        "interpretation_policy": "Numbers are descriptive acoustic observables. They must not be read as suspense/resolution labels; A-after is a same-track comparison only.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "trajectory_audit_v6.json")
    parser.add_argument("--audio-dir", type=Path, default=CANONICAL_AUDIO)
    parser.add_argument("--structure-dir", type=Path, default=STRUCTURE_DIR)
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    args = parser.parse_args(argv)
    report = build_audit(
        runs=args.runs,
        output=args.output,
        audio_dir=args.audio_dir,
        structure_dir=args.structure_dir,
    )
    print(json.dumps({"output": str(args.output), "selection_count": report["selection_count"], "run_counts": report["run_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
