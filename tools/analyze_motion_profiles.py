"""Build the read-only V5 motion audit from the selected V4 cues.

The script deliberately does not rank, render, or alter directional outputs.
It extracts beat-aligned descriptors from the existing canonical WAV files and
calls :func:`musicmix.motion.analyze_motion` at 8/16/32 beats.  Novelty is
reported only as the declared proxy ``1 - cosine(chroma_previous, chroma)``;
it is not a tension estimate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from musicmix.audio import AudioTrack, load_audio  # noqa: E402
from musicmix.directional import _audio_descriptor  # noqa: E402
from musicmix.motion import MotionBeat, MotionAssessment, analyze_motion  # noqa: E402


WINDOWS = (8, 16, 32)
RUNS = (
    ("forward", "downbeats", ROOT / "runs" / "structured_forward_v4" / "search.json"),
    ("reverse", "downbeats", ROOT / "runs" / "structured_reverse_v4" / "search.json"),
    ("forward", "beats", ROOT / "runs" / "beat_search_v4_forward.json"),
    ("reverse", "beats", ROOT / "runs" / "beat_search_v4_reverse.json"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _cosine(left: Sequence[float] | None, right: Sequence[float] | None) -> float | None:
    if left is None or right is None or len(left) != len(right) or not left:
        return None
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    norm_left = sum(float(a) * float(a) for a in left) ** 0.5
    norm_right = sum(float(b) * float(b) for b in right) ** 0.5
    if norm_left <= 1e-12 or norm_right <= 1e-12:
        return None
    return max(0.0, min(1.0, dot / (norm_left * norm_right)))


def _beats(structure: Mapping[str, Any]) -> tuple[tuple[int, float | None], ...]:
    values: list[tuple[int, float | None]] = []
    duration = int(structure.get("duration_samples", 0))
    for event in structure.get("beats", ()) or ():
        sample = _finite(event.get("sample") if isinstance(event, Mapping) else None)
        if sample is None or sample < 0 or sample >= duration:
            continue
        confidence = _finite(event.get("confidence") if isinstance(event, Mapping) else None)
        values.append((int(round(sample)), confidence))
    values.sort(key=lambda value: value[0])
    dedup: list[tuple[int, float | None]] = []
    for event in values:
        if dedup and dedup[-1][0] == event[0]:
            previous = dedup[-1]
            if previous[1] is None or (event[1] is not None and event[1] > previous[1]):
                dedup[-1] = event
        else:
            dedup.append(event)
    return tuple(dedup)


def _cue_index(beats: Sequence[tuple[int, float | None]], cue: int) -> int | None:
    for index, (sample, _confidence) in enumerate(beats):
        if sample == cue:
            return index
    return None


def _descriptor(
    track: AudioTrack,
    start: int,
    end: int,
    cache: dict[tuple[int, int], tuple[float, tuple[float, ...], tuple[float, ...]]],
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    key = (int(start), int(end))
    if key not in cache:
        cache[key] = _audio_descriptor(track, start, end)
    return cache[key]


def _profile(
    track: AudioTrack,
    beats: Sequence[tuple[int, float | None]],
    cue: int,
    side: str,
    max_beats: int,
    cache: dict[tuple[int, int], tuple[float, tuple[float, ...], tuple[float, ...]]],
) -> tuple[tuple[MotionBeat, ...], tuple[str, ...], dict[str, Any]]:
    index = _cue_index(beats, cue)
    if index is None:
        return tuple(), ("cue_not_on_beat_grid",), {"cue_sample": cue, "side": side}
    if side == "outgoing":
        start_index = max(0, index - max_beats)
        end_index = index
    else:
        start_index = index
        end_index = min(len(beats) - 1, index + max_beats)
    if end_index <= start_index:
        return tuple(), ("no_complete_beat_interval",), {"cue_sample": cue, "side": side}

    observations: list[MotionBeat] = []
    beat_activations: list[dict[str, Any]] = []
    missing: list[str] = []
    for beat_index in range(start_index, end_index):
        start, confidence = beats[beat_index]
        end = beats[beat_index + 1][0]
        if end <= start:
            missing.append("non_positive_beat_interval")
            continue
        rms, chroma, _timbre = _descriptor(track, start, end, cache)
        previous_chroma: tuple[float, ...] | None = None
        if beat_index > 0:
            previous_start = beats[beat_index - 1][0]
            if start > previous_start:
                previous_chroma = _descriptor(track, previous_start, start, cache)[1]
        cosine = _cosine(previous_chroma, chroma)
        novelty = None if cosine is None else 1.0 - cosine
        observations.append(
            MotionBeat(
                rms=rms,
                chroma=chroma or None,
                novelty=novelty,
                tension_proxy=None,
                # BeatThis confidence is retained as provenance metadata
                # below.  It is not a calibrated descriptor confidence and
                # must not be aggregated by the motion model.
                confidence=None,
                sample=start,
            )
        )
        beat_activations.append(
            {
                "sample": start,
                "value": confidence,
                "source": "structure.beats.confidence",
                "calibrated": False,
            }
        )
    return tuple(observations), tuple(dict.fromkeys(missing)), {
        "cue_sample": cue,
        "side": side,
        "cue_basis": "original_structure_cue_samples",
        "actual_candidate_cue_not_used": True,
        "cue_beat_index": index,
        "beat_index_range": [start_index, end_index],
        "available_beats": len(observations),
        "descriptor_source": "musicmix.directional._audio_descriptor",
        "novelty_proxy": "1-cosine(chroma_previous,chroma_current)",
        "tension_proxy": None,
        "beat_activation": beat_activations,
        "beat_activation_is_calibrated": False,
    }


def _json_motion(assessment: MotionAssessment) -> dict[str, Any]:
    return {
        "scales": [asdict(report) for report in assessment.scales],
        "preferred_window_beats": assessment.preferred_window_beats,
        "diagnostics": dict(assessment.diagnostics),
    }


def _selected(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if "items" in payload:
        return list(payload.get("items", ()))
    return list(payload.get("selected", ()))


def _cue_pair(item: Mapping[str, Any]) -> tuple[int, int] | None:
    if "candidate" in item:
        candidate = item.get("candidate", {})
        raw = item.get("raw_features", {}).get("original_structure_cue_samples", {})
        if isinstance(raw, Mapping) and "a" in raw and "b" in raw:
            return int(raw["a"]), int(raw["b"])
        if isinstance(candidate, Mapping):
            return int(candidate["cue_out_sample"]), int(candidate["cue_in_sample"])
    raw = item.get("original_cues")
    if isinstance(raw, Mapping) and "a" in raw and "b" in raw:
        return int(raw["a"]), int(raw["b"])
    return None


def _actual_pair(item: Mapping[str, Any], original: tuple[int, int]) -> tuple[int, int]:
    """Return revised candidate cue samples when the search persisted them."""

    if "candidate" in item and isinstance(item.get("candidate"), Mapping):
        candidate = item["candidate"]
        if "cue_out_sample" in candidate and "cue_in_sample" in candidate:
            return int(candidate["cue_out_sample"]), int(candidate["cue_in_sample"])
    if "cue_out_sample" in item and "cue_in_sample" in item:
        return int(item["cue_out_sample"]), int(item["cue_in_sample"])
    return original


def _candidate_id(item: Mapping[str, Any]) -> str:
    candidate = item.get("candidate")
    if isinstance(candidate, Mapping) and candidate.get("candidate_id"):
        return str(candidate["candidate_id"])
    return str(item.get("candidate_id", "unknown"))


def build_audit() -> dict[str, Any]:
    audio_dir = ROOT / "runs" / "reminder_to_too_many_nights" / "audio"
    structures = {
        "a": json.loads((ROOT / "runs" / "structured_forward_v4" / "inputs" / "structure_a.json").read_text()),
        "b": json.loads((ROOT / "runs" / "structured_forward_v4" / "inputs" / "structure_b.json").read_text()),
    }
    tracks = {key: load_audio(audio_dir / f"{key}.wav") for key in ("a", "b")}
    beat_lists = {key: _beats(value) for key, value in structures.items()}
    caches: dict[str, dict[tuple[int, int], tuple[float, tuple[float, ...], tuple[float, ...]]]] = {
        key: {} for key in tracks
    }
    source_hash = _sha256(Path(__file__))
    records: list[dict[str, Any]] = []
    for direction, policy, search_path in RUNS:
        payload = json.loads(search_path.read_text())
        direction_keys = ("a", "b") if direction == "forward" else ("b", "a")
        for selection_index, item in enumerate(_selected(payload), start=1):
            cue_pair = _cue_pair(item)
            if cue_pair is None:
                records.append({
                    "run": search_path.parent.name,
                    "direction": direction,
                    "cue_policy": policy,
                    "selection_index": selection_index,
                    "candidate_id": _candidate_id(item),
                    "missing": ["cue_pair"],
                })
                continue
            cue_a, cue_b = cue_pair
            actual_a, actual_b = _actual_pair(item, cue_pair)
            profile_a, missing_a, provenance_a = _profile(
                tracks[direction_keys[0]],
                beat_lists[direction_keys[0]],
                cue_a,
                "outgoing",
                max(WINDOWS),
                caches[direction_keys[0]],
            )
            profile_b, missing_b, provenance_b = _profile(
                tracks[direction_keys[1]],
                beat_lists[direction_keys[1]],
                cue_b,
                "incoming",
                max(WINDOWS),
                caches[direction_keys[1]],
            )
            assessment = analyze_motion(profile_a, profile_b, windows_beats=WINDOWS)
            record = {
                "run": search_path.parent.name,
                "direction": direction,
                "cue_policy": policy,
                "selection_index": selection_index,
                "candidate_id": _candidate_id(item),
                "original_cues": {"a": cue_a, "b": cue_b},
                "actual_candidate_cues": {"a": actual_a, "b": actual_b},
                "cue_reference": {
                    "profile_reference": "original_structure_cue_samples",
                    "actual_candidate_cue_not_used_for_beat_profiles": True,
                    "shift_samples": {"a": actual_a - cue_a, "b": actual_b - cue_b},
                    "reason": "shifted vocal-safe cues are not beat-grid anchors; no beat or descriptor interpolation was imputed",
                },
                "cue_seconds": {"a": cue_a / tracks[direction_keys[0]].sample_rate, "b": cue_b / tracks[direction_keys[1]].sample_rate},
                "sample_rate": {"a": tracks[direction_keys[0]].sample_rate, "b": tracks[direction_keys[1]].sample_rate},
                "track_lineage": {
                    "a": {"track_key": direction_keys[0], "track_id": tracks[direction_keys[0]].track_id, "path": str(tracks[direction_keys[0]].path), "sha256": tracks[direction_keys[0]].track_id},
                    "b": {"track_key": direction_keys[1], "track_id": tracks[direction_keys[1]].track_id, "path": str(tracks[direction_keys[1]].path), "sha256": tracks[direction_keys[1]].track_id},
                },
                "profiles": {
                    "outgoing": {"provenance": provenance_a, "missing": list(missing_a), "beats": [asdict(value) for value in profile_a]},
                    "incoming": {"provenance": provenance_b, "missing": list(missing_b), "beats": [asdict(value) for value in profile_b]},
                },
                "motion": _json_motion(assessment),
                "local_features": {
                    "score": item.get("score"),
                    "breakdown": item.get("breakdown", {}),
                    "boundary_class": item.get("alignment", {}).get("boundary_class", item.get("boundary_class")),
                },
                "audit_constraints": {
                    "reranking_applied": False,
                    "rendering_applied": False,
                    "tension_proxy_supplied": False,
                    "novelty_is_tension": False,
                },
            }
            records.append(record)
    window_class_counts: dict[str, dict[str, int]] = {}
    for width in WINDOWS:
        counts: dict[str, int] = {}
        for record in records:
            reports = record.get("motion", {}).get("scales", ())
            report = next((value for value in reports if value.get("window_beats") == width), None)
            classification = str(report.get("classification", "missing_report")) if report is not None else "missing_report"
            counts[classification] = counts.get(classification, 0) + 1
        window_class_counts[str(width)] = counts
    all_windows_complete = sum(
        all(
            next(
                (value for value in record.get("motion", {}).get("scales", ()) if value.get("window_beats") == width),
                {"classification": "missing_report"},
            ).get("classification") != "insufficient_window"
            for width in WINDOWS
        )
        for record in records
    )
    return {
        "schema_version": "musicmix.motion_audit.v5",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(Path(__file__)),
        "script_sha256": source_hash,
        "motion_module": "musicmix.motion",
        "windows_beats": list(WINDOWS),
        "selection_inputs": [str(path) for _, _, path in RUNS],
        "records": records,
        "summary": {
            "record_count": len(records),
            "profiles_with_all_requested_windows": all_windows_complete,
            "window_class_counts": window_class_counts,
            "tension_proxy_supplied": False,
            "no_reranking": True,
            "no_wav_outputs": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "motion_audit_v5.json")
    args = parser.parse_args()
    audit = build_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **audit["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
