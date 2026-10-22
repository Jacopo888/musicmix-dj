"""External, reproducible cue-point benchmark for WHENmix.

The only accepted ground truth is a complete public annotation release.  M-DJCUE
is read directly from its JAMS files.  Missing predictions are treated as
abstentions, and predictions outside an empirically covered label region are
reported separately rather than counted as false positives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


BENCHMARK_SCHEMA = "musicmix.external_benchmark.1"
PREDICTION_SCHEMA = "musicmix.external_predictions.1"
EVALUATION_SCHEMA = "musicmix.external_evaluation.1"
DATASET_NAME = "M-DJCUE"
DATASET_REVISION = "7fd692c793a4a4744ab61e409854302269b0974c"
DATASET_URL = "https://github.com/MZehren/M-DJCUE"
PAPER_URL = "https://arxiv.org/abs/2007.08411"
LABELS = ("IN", "OUT")
MERGE_TOLERANCE_S = 0.5
DEFAULT_TOLERANCE_S = 0.5


class BenchmarkError(ValueError):
    """Invalid source annotation, split, or prediction payload."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise BenchmarkError(f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def version_base(title: str) -> str:
    value = title
    previous = None
    while value != previous:
        previous = value
        value = re.sub(r"\([^()]*\)|\[[^\[\]]*\]", " ", value)
    value = re.split(r"[\(\[]", value, maxsplit=1)[0]
    return normalize_text(value)


def component_groups(tracks: Sequence[dict[str, Any]]) -> dict[str, str]:
    parent = list(range(len(tracks)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_artist: dict[str, list[int]] = defaultdict(list)
    by_version: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, track in enumerate(tracks):
        artist = normalize_text(track["artist"])
        by_artist[artist].append(index)
        by_version[(artist, version_base(track["title"]))].append(index)
    for groups in (by_artist, by_version):
        for indices in groups.values():
            for other in indices[1:]:
                union(indices[0], other)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(tracks)):
        components[find(index)].append(index)

    group_of_track: dict[str, str] = {}
    for indices in components.values():
        track_ids = sorted(tracks[index]["track_id"] for index in indices)
        digest = hashlib.sha256("\n".join(track_ids).encode("utf-8")).hexdigest()[:12]
        group_id = f"grp_{digest}"
        for index in indices:
            group_of_track[tracks[index]["track_id"]] = group_id
    return group_of_track


def clusters_for_track(track: dict[str, Any], merge_tolerance: float) -> list[dict[str, Any]]:
    annotators = sorted(track["annotators"])
    total = len(annotators)
    result: list[dict[str, Any]] = []
    for label in LABELS:
        events = sorted(
            (event for event in track["events"] if event["label"] == label),
            key=lambda event: event["time_s"],
        )
        current: list[dict[str, Any]] = []
        for event in events:
            if current and event["time_s"] - current[-1]["time_s"] > merge_tolerance:
                result.append(_cluster(track, label, current, total, len(result)))
                current = []
            current.append(event)
        if current:
            result.append(_cluster(track, label, current, total, len(result)))
    return result


def _cluster(
    track: dict[str, Any],
    label: str,
    events: list[dict[str, Any]],
    annotator_total: int,
    ordinal: int,
) -> dict[str, Any]:
    times = [event["time_s"] for event in events]
    support = len({event["annotator"] for event in events})
    return {
        "cluster_id": f"{track['track_id']}::{label}::{ordinal:03d}",
        "track_id": track["track_id"],
        "label": label,
        "center_s": statistics.median(times),
        "first_s": min(times),
        "last_s": max(times),
        "span_s": max(times) - min(times),
        "support": support,
        "annotator_total": annotator_total,
        "agreement": support / annotator_total if annotator_total else 0.0,
        "annotators": sorted({event["annotator"] for event in events}),
        "times_s": sorted(times),
        "max_duration_s": max((event["duration_s"] for event in events), default=0.0),
        "comments": sorted({event["comment"] for event in events if event["comment"]}),
    }


def label_regions(track: dict[str, Any], clusters: Sequence[dict[str, Any]], merge_tolerance: float) -> dict[str, list[float]]:
    regions: dict[str, list[float]] = {}
    duration = track["duration_s"]
    for label in LABELS:
        centers = [cluster["center_s"] for cluster in clusters if cluster["label"] == label]
        if not centers:
            continue
        if label == "IN":
            regions[label] = [0.0, min(duration, max(centers) + merge_tolerance)]
        else:
            regions[label] = [max(0.0, min(centers) - merge_tolerance), duration]
    return regions


def load_mdjcue(jams_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files = sorted(jams_dir.glob("*.jams"))
    if not files:
        raise BenchmarkError(f"no .jams files below {jams_dir}")
    tracks: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in files:
        payload = read_json(path)
        metadata = payload.get("file_metadata")
        if not isinstance(metadata, dict):
            raise BenchmarkError(f"{path.name}: missing file_metadata")
        for field in ("title", "artist"):
            if not isinstance(metadata.get(field), str) or not metadata[field].strip():
                raise BenchmarkError(f"{path.name}: invalid {field}")
        duration = metadata.get("duration")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
            raise BenchmarkError(f"{path.name}: invalid duration")
        track_id = path.stem
        if track_id in seen_ids:
            raise BenchmarkError(f"duplicate track id {track_id!r}")
        seen_ids.add(track_id)

        annotations = payload.get("annotations")
        if not isinstance(annotations, list) or not annotations:
            raise BenchmarkError(f"{path.name}: no annotations")
        events: list[dict[str, Any]] = []
        annotators: list[str] = []
        namespaces: Counter[str] = Counter()
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise BenchmarkError(f"{path.name}: annotation is not an object")
            namespace = annotation.get("namespace")
            namespaces[str(namespace)] += 1
            if namespace != "cue_point":
                issues.append({"track_id": track_id, "kind": "ignored_namespace", "namespace": namespace})
                continue
            annotator = (
                annotation.get("annotation_metadata", {})
                .get("annotator", {})
                .get("name")
            )
            if not isinstance(annotator, str) or not annotator.strip():
                raise BenchmarkError(f"{path.name}: cue_point annotation without annotator")
            annotators.append(annotator)
            data = annotation.get("data")
            if not isinstance(data, list):
                raise BenchmarkError(f"{path.name}: cue_point data is not a list")
            for row in data:
                if not isinstance(row, dict):
                    raise BenchmarkError(f"{path.name}: cue event is not an object")
                time_s = row.get("time")
                if not isinstance(time_s, (int, float)) or isinstance(time_s, bool) or not math.isfinite(time_s):
                    raise BenchmarkError(f"{path.name}: non-finite event time")
                if time_s < -1e-9:
                    raise BenchmarkError(f"{path.name}: negative event time {time_s}")
                value = row.get("value")
                if not isinstance(value, dict):
                    raise BenchmarkError(f"{path.name}: event value is not an object")
                label = value.get("label")
                if label not in LABELS:
                    raise BenchmarkError(f"{path.name}: unexpected label {label!r}")
                event_duration = row.get("duration", 0.0)
                if event_duration is None:
                    event_duration = 0.0
                if not isinstance(event_duration, (int, float)) or isinstance(event_duration, bool) or event_duration < 0:
                    raise BenchmarkError(f"{path.name}: invalid event duration")
                comment = value.get("comment", "")
                if comment is None:
                    comment = ""
                if not isinstance(comment, str):
                    raise BenchmarkError(f"{path.name}: invalid comment")
                events.append(
                    {
                        "annotator": annotator,
                        "time_s": float(time_s),
                        "label": label,
                        "duration_s": float(event_duration),
                        "comment": comment,
                    }
                )
        if not events:
            raise BenchmarkError(f"{path.name}: no cue_point events")
        annotators = sorted(set(annotators))
        declared_duration = float(duration)
        max_event_time = max(event["time_s"] for event in events)
        effective_duration = max(declared_duration, max_event_time)
        for event in events:
            if event["time_s"] > declared_duration + 1e-6:
                issues.append(
                    {
                        "track_id": track_id,
                        "kind": "event_beyond_declared_duration",
                        "time_s": event["time_s"],
                        "declared_duration_s": declared_duration,
                        "overrun_s": event["time_s"] - declared_duration,
                        "label": event["label"],
                        "annotator": event["annotator"],
                    }
                )
        track = {
            "track_id": track_id,
            "source_filename": path.name,
            "source_sha256": sha256_file(path),
            "title": metadata["title"],
            "artist": metadata["artist"],
            "release": str(metadata.get("release", "")),
            "declared_duration_s": declared_duration,
            "duration_s": effective_duration,
            "jams_version": metadata.get("jams_version"),
            "is_vinyl": bool(metadata.get("identifiers", {}).get("isVinyl")),
            "mirror_count": len(metadata.get("identifiers", {}).get("mirrors", []) or []),
            "namespaces": dict(sorted(namespaces.items())),
            "annotators": annotators,
            "events": sorted(events, key=lambda event: (event["label"], event["time_s"], event["annotator"])),
        }
        tracks.append(track)
    return tracks, issues


def assign_split(tracks: list[dict[str, Any]], seed: int, test_fraction: float) -> dict[str, Any]:
    if not 0.0 < test_fraction < 1.0:
        raise BenchmarkError("test fraction must be between 0 and 1")
    group_of_track = component_groups(tracks)
    groups: dict[str, list[str]] = defaultdict(list)
    for track in tracks:
        groups[group_of_track[track["track_id"]]].append(track["track_id"])
    ordered = sorted(
        groups,
        key=lambda group_id: hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest(),
    )
    test_group_count = min(len(ordered) - 1, max(1, round(len(ordered) * test_fraction)))
    test_groups = set(ordered[:test_group_count])
    for track in tracks:
        track["group_id"] = group_of_track[track["track_id"]]
        track["split"] = "test" if track["group_id"] in test_groups else "train"

    train = [track for track in tracks if track["split"] == "train"]
    test = [track for track in tracks if track["split"] == "test"]
    train_artists = {normalize_text(track["artist"]) for track in train}
    test_artists = {normalize_text(track["artist"]) for track in test}
    train_versions = {
        (normalize_text(track["artist"]), version_base(track["title"])) for track in train
    }
    test_versions = {
        (normalize_text(track["artist"]), version_base(track["title"])) for track in test
    }
    return {
        "seed": seed,
        "test_fraction": test_fraction,
        "group_count": len(groups),
        "test_group_count": len(test_groups),
        "train_tracks": len(train),
        "test_tracks": len(test),
        "train_artists": len(train_artists),
        "test_artists": len(test_artists),
        "artist_overlap": sorted(train_artists & test_artists),
        "version_family_overlap": sorted(f"{artist}|{base}" for artist, base in (train_versions & test_versions)),
        "test_track_ids": sorted(track["track_id"] for track in test),
        "train_track_ids": sorted(track["track_id"] for track in train),
        "group_components": {
            group_id: sorted(track_ids) for group_id, track_ids in sorted(groups.items())
        },
    }


def build_benchmark(jams_dir: Path, out_dir: Path, seed: int, test_fraction: float) -> dict[str, Any]:
    tracks, issues = load_mdjcue(jams_dir)
    split = assign_split(tracks, seed, test_fraction)
    for track in tracks:
        track["clusters"] = clusters_for_track(track, MERGE_TOLERANCE_S)
        track["regions"] = label_regions(track, track["clusters"], MERGE_TOLERANCE_S)
    ordered_tracks = sorted(tracks, key=lambda track: track["track_id"])
    payload = {
        "schema_version": BENCHMARK_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": DATASET_NAME,
            "source_url": DATASET_URL,
            "paper_url": PAPER_URL,
            "revision": DATASET_REVISION,
            "license": "GPL-3.0 for repository annotations/metadata",
            "audio_included": False,
            "audio_note": "Audio is copyrighted and is not distributed; no audio was downloaded.",
            "jams_dir": str(jams_dir),
            "jams_file_count": len(tracks),
            "jams_combined_sha256": hashlib.sha256(
                "".join(track["source_sha256"] for track in ordered_tracks).encode("ascii")
            ).hexdigest(),
            "merge_tolerance_s": MERGE_TOLERANCE_S,
            "available_signals": ["cue_point", "track_metadata", "duration"],
            "downbeat_available": False,
            "downbeat_note": "This release contains only cue_point annotations; no beat/downbeat grid is shipped.",
            "transition_pair_evidence": "none",
            "transition_pair_note": (
                "Cue points are per-track and independent of the next track; no A-to-B pair or "
                "professional DJ set alignment is present in this release."
            ),
            "duration_policy": (
                "duration_s=max(declared duration, last annotated event). Four vinyl tracks have "
                "events beyond the declared duration; these are reported in reading_issues."
            ),
            "region_policy": (
                "IN=[0,max(IN center)+merge_tol]; OUT=[min(OUT center)-merge_tol,duration]. "
                "These are empirical coverage proxies, not claims that every unannotated point is negative."
            ),
        },
        "split": split,
        "reading_issues": issues,
        "tracks": ordered_tracks,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "mdjcue_benchmark.json", payload)
    write_json(out_dir / "split.json", split)
    return payload


def load_benchmark(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != BENCHMARK_SCHEMA:
        raise BenchmarkError(f"expected {BENCHMARK_SCHEMA}, got {payload.get('schema_version')!r}")
    if not isinstance(payload.get("tracks"), list) or not payload["tracks"]:
        raise BenchmarkError("benchmark has no tracks")
    return payload


def benchmark_summary(benchmark: dict[str, Any]) -> dict[str, Any]:
    tracks = benchmark["tracks"]
    events = [event for track in tracks for event in track["events"]]
    clusters = [cluster for track in tracks for cluster in track["clusters"]]
    support = Counter(cluster["support"] for cluster in clusters)
    by_label = {}
    for label in LABELS:
        label_clusters = [cluster for cluster in clusters if cluster["label"] == label]
        label_events = [event for event in events if event["label"] == label]
        by_label[label] = {
            "events": len(label_events),
            "clusters": len(label_clusters),
            "support_ge_2": sum(cluster["support"] >= 2 for cluster in label_clusters),
            "support_ge_3": sum(cluster["support"] >= 3 for cluster in label_clusters),
            "mean_support": statistics.fmean(cluster["support"] for cluster in label_clusters),
            "mean_agreement": statistics.fmean(cluster["agreement"] for cluster in label_clusters),
        }
    return {
        "tracks": len(tracks),
        "events": len(events),
        "clusters": len(clusters),
        "cluster_support_histogram": dict(sorted(support.items())),
        "by_label": by_label,
        "annotators": dict(sorted(Counter(event["annotator"] for event in events).items())),
        "events_per_track_mean": statistics.fmean(
            sum(1 for event in track["events"]) for track in tracks
        ),
        "tracks_with_both_labels": sum(
            bool(track["regions"].get("IN")) and bool(track["regions"].get("OUT"))
            for track in tracks
        ),
    }


def pairwise_agreement(benchmark: dict[str, Any]) -> dict[str, Any]:
    label_scores: dict[str, list[float]] = {label: [] for label in LABELS}
    presence_scores: dict[str, list[float]] = {label: [] for label in LABELS}
    for track in benchmark["tracks"]:
        annotators = sorted(track["annotators"])
        for label in LABELS:
            by_annotator = {
                annotator: sorted(
                    event["time_s"]
                    for event in track["events"]
                    if event["annotator"] == annotator and event["label"] == label
                )
                for annotator in annotators
            }
            for left_index in range(len(annotators)):
                for right_index in range(left_index + 1, len(annotators)):
                    left = by_annotator[annotators[left_index]]
                    right = by_annotator[annotators[right_index]]
                    if not left and not right:
                        continue
                    matched, unused_left, unused_right = greedy_point_match(left, right, MERGE_TOLERANCE_S)
                    precision = matched / len(left) if left else 0.0
                    recall = matched / len(right) if right else 0.0
                    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
                    label_scores[label].append(f1)
                    presence_scores[label].append(1.0 if bool(left) == bool(right) else 0.0)
    result: dict[str, Any] = {}
    for label in LABELS:
        scores = label_scores[label]
        presence = presence_scores[label]
        result[label] = {
            "pairs_scored": len(scores),
            "mean_pairwise_f1": statistics.fmean(scores) if scores else None,
            "mean_label_presence_agreement": statistics.fmean(presence) if presence else None,
        }
    all_scores = label_scores["IN"] + label_scores["OUT"]
    result["all"] = {
        "pairs_scored": len(all_scores),
        "mean_pairwise_f1": statistics.fmean(all_scores) if all_scores else None,
    }
    result["tolerance_s"] = MERGE_TOLERANCE_S
    result["note"] = (
        "Pairwise F1 between raw annotator event lists in the same track/label, "
        "one-to-one greedy matching within the tolerance. Track/label pairs with no "
        "events from either annotator are excluded from the mean."
    )
    return result


def greedy_point_match(
    left: Sequence[float], right: Sequence[float], tolerance: float
) -> tuple[int, list[float], list[float]]:
    candidates = []
    for left_index, left_time in enumerate(left):
        for right_index, right_time in enumerate(right):
            difference = abs(left_time - right_time)
            if difference <= tolerance:
                candidates.append((difference, left_index, right_index))
    candidates.sort()
    used_left: set[int] = set()
    used_right: set[int] = set()
    for _, left_index, right_index in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
    return (
        len(used_left),
        [left[index] for index in range(len(left)) if index not in used_left],
        [right[index] for index in range(len(right)) if index not in used_right],
    )


def exhaustive_gold_coverage(benchmark: dict[str, Any], split_name: str = "test") -> dict[str, Any]:
    tracks = [track for track in benchmark["tracks"] if track["split"] == split_name]
    total_duration = sum(track["duration_s"] for track in tracks)
    result: dict[str, Any] = {"tracks": len(tracks), "total_duration_s": total_duration}
    for label in LABELS:
        durations = 0.0
        covered_tracks = 0
        spans = 0.0
        for track in tracks:
            region = track["regions"].get(label)
            if not region:
                continue
            covered_tracks += 1
            durations += max(0.0, region[1] - region[0])
            events = [event["time_s"] for event in track["events"] if event["label"] == label]
            if events:
                spans += max(events) - min(events)
        result[label] = {
            "tracks_with_region": covered_tracks,
            "track_fraction": covered_tracks / len(tracks) if tracks else 0.0,
            "duration_fraction": durations / total_duration if total_duration else 0.0,
            "mean_annotation_span_s": spans / covered_tracks if covered_tracks else None,
        }
    gaps = []
    for track in tracks:
        in_region = track["regions"].get("IN")
        out_region = track["regions"].get("OUT")
        if in_region and out_region:
            gaps.append(max(0.0, out_region[0] - in_region[1]))
    result["unscored_gap_mean_s"] = statistics.fmean(gaps) if gaps else None
    result["unscored_gap_median_s"] = statistics.median(gaps) if gaps else None
    result["policy_note"] = benchmark["dataset"]["region_policy"]
    return result


def normalize_predictions(path: Path, known_track_ids: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, list):
        rows = payload
        declared_schema = None
    elif isinstance(payload, dict):
        declared_schema = payload.get("schema_version")
        if "predictions" in payload:
            rows = payload["predictions"]
        elif "tracks" in payload:
            rows = []
            for track in payload["tracks"]:
                if not isinstance(track, dict):
                    raise BenchmarkError("each track prediction must be an object")
                track_id = track.get("track_id")
                if not isinstance(track_id, str) or not track_id:
                    raise BenchmarkError("track prediction needs a non-empty track_id")
                cues = track.get("cues", track.get("predictions", []))
                if not isinstance(cues, list):
                    raise BenchmarkError(f"{track_id}: cues must be a list")
                if track.get("abstain") is True and not cues:
                    for label in track.get("abstain_labels", LABELS):
                        rows.append({"track_id": track_id, "label": label, "abstain": True, "reason": track.get("reason", "")})
                for cue in cues:
                    if not isinstance(cue, dict):
                        raise BenchmarkError(f"{track_id}: cue must be an object")
                    row = dict(cue)
                    row["track_id"] = track_id
                    rows.append(row)
        else:
            raise BenchmarkError("prediction object needs a predictions or tracks list")
    else:
        raise BenchmarkError("predictions must be a JSON object or list")
    if declared_schema not in (None, PREDICTION_SCHEMA):
        raise BenchmarkError(f"unexpected prediction schema {declared_schema!r}")
    if not isinstance(rows, list):
        raise BenchmarkError("predictions list is missing")

    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise BenchmarkError(f"prediction {index} is not an object")
        track_id = row.get("track_id")
        if not isinstance(track_id, str) or not track_id:
            raise BenchmarkError(f"prediction {index}: invalid track_id")
        if track_id not in known_track_ids:
            raise BenchmarkError(f"prediction {index}: unknown track_id {track_id!r}")
        label = row.get("label")
        if isinstance(label, str):
            label = label.upper()
        if label not in LABELS:
            raise BenchmarkError(f"prediction {index}: label must be IN or OUT")
        abstain = row.get("abstain") is True
        if abstain:
            normalized.append(
                {
                    "track_id": track_id,
                    "label": label,
                    "kind": "abstain",
                    "reason": str(row.get("reason", "")),
                    "source_index": index,
                }
            )
            continue
        time_s = row.get("time_s", row.get("time"))
        if not isinstance(time_s, (int, float)) or isinstance(time_s, bool) or not math.isfinite(time_s):
            raise BenchmarkError(f"prediction {index}: finite numeric time is required")
        confidence = row.get("confidence")
        if confidence is not None and (
            not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
        ):
            raise BenchmarkError(f"prediction {index}: confidence must be numeric or null")
        normalized.append(
            {
                "track_id": track_id,
                "label": label,
                "kind": "cue",
                "time_s": float(time_s),
                "confidence": None if confidence is None else float(confidence),
                "source_index": index,
            }
        )
    return normalized, {
        "path": str(path),
        "sha256": sha256_file(path),
        "declared_schema": declared_schema,
        "row_count": len(normalized),
    }


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _in_region(time_s: float, region: Sequence[float] | None) -> bool:
    if not region:
        return False
    return region[0] - 1e-9 <= time_s <= region[1] + 1e-9


def evaluate_predictions(
    benchmark: dict[str, Any],
    predictions: list[dict[str, Any]],
    tolerance_s: float,
    split_name: str = "test",
) -> dict[str, Any]:
    tracks = [track for track in benchmark["tracks"] if track["split"] == split_name]
    track_by_id = {track["track_id"]: track for track in tracks}
    cue_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    abstain_rows: set[tuple[str, str]] = set()
    ignored_nontest = 0
    for row in predictions:
        if row["track_id"] not in track_by_id:
            ignored_nontest += 1
            continue
        key = (row["track_id"], row["label"])
        if row["kind"] == "abstain":
            abstain_rows.add(key)
        else:
            cue_rows[key].append(row)

    metrics: dict[str, Any] = {}
    per_track: list[dict[str, Any]] = []
    for min_support in (1, 2):
        label_metrics: dict[str, Any] = {}
        for label in LABELS:
            tp = 0
            gold_count = 0
            in_region_predictions = 0
            out_of_region = 0
            matched_errors: list[float] = []
            role_confusions = 0
            tracks_with_gold = 0
            tracks_emitted = 0
            tracks_abstained = 0
            gold_clusters_total = 0
            for track in tracks:
                track_id = track["track_id"]
                region = track["regions"].get(label)
                gold = [
                    cluster
                    for cluster in track["clusters"]
                    if cluster["label"] == label and cluster["support"] >= min_support
                ]
                gold_clusters_total += len(gold)
                predictions_here = cue_rows.get((track_id, label), [])
                emitted = [row for row in predictions_here if _in_region(row["time_s"], region)]
                outside = [row for row in predictions_here if not _in_region(row["time_s"], region)]
                if gold:
                    tracks_with_gold += 1
                if predictions_here:
                    tracks_emitted += 1
                elif (track_id, label) in abstain_rows or not predictions_here:
                    tracks_abstained += 1
                if not gold:
                    out_of_region += len(outside)
                    continue
                gold_count += len(gold)
                in_region_predictions += len(emitted)
                out_of_region += len(outside)
                candidates = []
                for gold_index, cluster in enumerate(gold):
                    for prediction_index, row in enumerate(emitted):
                        difference = abs(row["time_s"] - cluster["center_s"])
                        if difference <= tolerance_s:
                            candidates.append((difference, gold_index, prediction_index))
                candidates.sort()
                used_gold: set[int] = set()
                used_prediction: set[int] = set()
                for difference, gold_index, prediction_index in candidates:
                    if gold_index in used_gold or prediction_index in used_prediction:
                        continue
                    used_gold.add(gold_index)
                    used_prediction.add(prediction_index)
                    matched_errors.append(difference)
                tp += len(used_gold)
                opposite = [
                    cluster
                    for cluster in track["clusters"]
                    if cluster["label"] != label and cluster["support"] >= min_support
                ]
                for prediction_index, row in enumerate(emitted):
                    if prediction_index in used_prediction:
                        continue
                    if any(abs(row["time_s"] - cluster["center_s"]) <= tolerance_s for cluster in opposite):
                        role_confusions += 1
            fp = in_region_predictions - tp
            fn = gold_count - tp
            precision = tp / in_region_predictions if in_region_predictions else None
            recall = tp / gold_count if gold_count else None
            f1 = None
            if precision is not None and recall is not None and precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            label_metrics[label] = {
                "gold_clusters": gold_count,
                "predictions_in_region": in_region_predictions,
                "predictions_out_of_region": out_of_region,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "timing_mae_s": statistics.fmean(matched_errors) if matched_errors else None,
                "timing_median_s": statistics.median(matched_errors) if matched_errors else None,
                "timing_p90_s": _percentile(matched_errors, 0.9),
                "timing_max_s": max(matched_errors) if matched_errors else None,
                "role_confusions_near_opposite_label": role_confusions,
                "tracks_with_gold": tracks_with_gold,
                "tracks_with_emitted_prediction": tracks_emitted,
                "tracks_abstained": tracks_abstained,
                "abstention_rate": tracks_abstained / len(tracks) if tracks else None,
                "gold_clusters_including_empty_label_tracks": gold_clusters_total,
            }
        total_tp = sum(label_metrics[label]["tp"] for label in LABELS)
        total_pred = sum(label_metrics[label]["predictions_in_region"] for label in LABELS)
        total_gold = sum(label_metrics[label]["gold_clusters"] for label in LABELS)
        total_fp = total_pred - total_tp
        total_fn = total_gold - total_tp
        precision = total_tp / total_pred if total_pred else None
        recall = total_tp / total_gold if total_gold else None
        f1 = None
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        metrics["support_ge_1" if min_support == 1 else "support_ge_2_strict"] = {
            "labels": label_metrics,
            "micro": {
                "gold_clusters": total_gold,
                "predictions_in_region": total_pred,
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
        }

    for track in tracks:
        row = {"track_id": track["track_id"]}
        for label in LABELS:
            gold = [cluster for cluster in track["clusters"] if cluster["label"] == label]
            emitted = cue_rows.get((track["track_id"], label), [])
            row[label] = {
                "gold_clusters": len(gold),
                "predictions": len(emitted),
                "abstained": (track["track_id"], label) in abstain_rows and not emitted,
            }
        per_track.append(row)

    return {
        "schema_version": EVALUATION_SCHEMA,
        "dataset": {
            "name": benchmark["dataset"]["name"],
            "revision": benchmark["dataset"]["revision"],
            "license": benchmark["dataset"]["license"],
        },
        "split": {key: benchmark["split"][key] for key in ("seed", "test_fraction", "train_tracks", "test_tracks", "group_count")},
        "config": {
            "tolerance_s": tolerance_s,
            "merge_tolerance_s": benchmark["dataset"]["merge_tolerance_s"],
            "region_policy": benchmark["dataset"]["region_policy"],
            "matched_by": "one-to-one greedy matching on absolute time difference",
            "missing_prediction_policy": "abstention; lowers recall, contributes no false positive",
            "out_of_region_policy": "reported separately, never counted as false positive",
        },
        "predictions": {
            "evaluated_test_cues": sum(len(rows) for rows in cue_rows.values()),
            "explicit_abstentions": len(abstain_rows),
            "ignored_predictions_for_non_test_tracks": ignored_nontest,
        },
        "metrics": metrics,
        "coverage": exhaustive_gold_coverage(benchmark, split_name),
        "per_track": per_track,
    }


def train_position_prior(
    benchmark: dict[str, Any], mode: str
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    source = [track for track in benchmark["tracks"] if track["split"] == "train"]
    prior: dict[str, dict[str, float]] = {}
    evidence: dict[str, Any] = {}
    for label in LABELS:
        centres = [
            cluster["center_s"]
            for track in source
            for cluster in track["clusters"]
            if cluster["label"] == label
        ]
        fractions = [
            cluster["center_s"] / track["duration_s"]
            for track in source
            for cluster in track["clusters"]
            if cluster["label"] == label
        ]
        if not centres or not fractions:
            continue
        prior[label] = {
            "median_fraction": statistics.median(fractions),
            "median_absolute_s": statistics.median(centres),
        }
        evidence[label] = {
            "train_tracks_used": len(source),
            "train_clusters_used": len(centres),
            "median_fraction": prior[label]["median_fraction"],
            "median_absolute_s": prior[label]["median_absolute_s"],
        }
    return prior, evidence


def position_baseline_predictions(
    benchmark: dict[str, Any], mode: str, out_path: Path
) -> dict[str, Any]:
    if mode not in {"fraction", "absolute"}:
        raise BenchmarkError("baseline mode must be fraction or absolute")
    prior, evidence = train_position_prior(benchmark, mode)
    rows: list[dict[str, Any]] = []
    test_tracks = [track for track in benchmark["tracks"] if track["split"] == "test"]
    for track in test_tracks:
        for label in LABELS:
            if label not in prior:
                rows.append(
                    {
                        "track_id": track["track_id"],
                        "label": label,
                        "abstain": True,
                        "reason": "no train prior",
                    }
                )
                continue
            if mode == "fraction":
                time_s = prior[label]["median_fraction"] * track["duration_s"]
            else:
                time_s = prior[label]["median_absolute_s"]
            time_s = min(max(0.0, time_s), track["duration_s"])
            rows.append(
                {
                    "track_id": track["track_id"],
                    "label": label,
                    "time_s": round(time_s, 6),
                    "confidence": None,
                }
            )
    payload = {
        "schema_version": PREDICTION_SCHEMA,
        "dataset": DATASET_NAME,
        "revision": DATASET_REVISION,
        "baseline": f"train_median_position_{mode}",
        "note": (
            "Metadata/duration-only position prior fitted on the train split. "
            "It does not use audio and does not validate an audio algorithm."
        ),
        "prior_evidence": evidence,
        "predictions": rows,
    }
    write_json(out_path, payload)
    return payload


def summarize_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "tolerance_s": evaluation["config"]["tolerance_s"],
        "coverage": evaluation["coverage"],
        "predictions": evaluation["predictions"],
    }
    for key in ("support_ge_1", "support_ge_2_strict"):
        out[key] = {
            "micro": evaluation["metrics"][key]["micro"],
            "IN": {
                field: evaluation["metrics"][key]["labels"]["IN"][field]
                for field in ("precision", "recall", "f1", "timing_mae_s", "timing_median_s")
            },
            "OUT": {
                field: evaluation["metrics"][key]["labels"]["OUT"][field]
                for field in ("precision", "recall", "f1", "timing_mae_s", "timing_median_s")
            },
        }
    return out


def command_build(args: argparse.Namespace) -> None:
    jams_dir = Path(args.jams_dir) if args.jams_dir else default_jams_dir()
    benchmark = build_benchmark(Path(jams_dir), Path(args.out), args.seed, args.test_fraction)
    summary = benchmark_summary(benchmark)
    write_json(Path(args.out) / "benchmark_summary.json", summary)
    print(json.dumps({"benchmark": summary, "split": {k: benchmark["split"][k] for k in ("group_count", "test_group_count", "train_tracks", "test_tracks", "artist_overlap", "version_family_overlap")}}, indent=2))


def command_evaluate(args: argparse.Namespace) -> None:
    benchmark = load_benchmark(Path(args.dataset))
    known = {track["track_id"] for track in benchmark["tracks"]}
    predictions, prediction_meta = normalize_predictions(Path(args.predictions), known)
    evaluation = evaluate_predictions(benchmark, predictions, args.tolerance, args.split)
    evaluation["predictions"].update(prediction_meta)
    evaluation["agreement"] = pairwise_agreement(benchmark)
    write_json(Path(args.out), evaluation)
    print(json.dumps(summarize_metrics(evaluation), indent=2))


def command_baseline(args: argparse.Namespace) -> None:
    benchmark = load_benchmark(Path(args.dataset))
    out_dir = Path(args.out)
    results = {}
    for mode in ("fraction", "absolute"):
        path = out_dir / f"baseline_position_{mode}_predictions.json"
        payload = position_baseline_predictions(benchmark, mode, path)
        predictions, prediction_meta = normalize_predictions(
            path, {track["track_id"] for track in benchmark["tracks"]}
        )
        evaluation = evaluate_predictions(benchmark, predictions, args.tolerance, "test")
        evaluation["predictions"].update(prediction_meta)
        evaluation["predictions"]["baseline"] = payload["baseline"]
        evaluation["predictions"]["prior_evidence"] = payload["prior_evidence"]
        eval_path = out_dir / f"evaluation_position_{mode}.json"
        write_json(eval_path, evaluation)
        results[mode] = summarize_metrics(evaluation)
    results["downbeat_baseline"] = {
        "status": "not_available",
        "reason": "M-DJCUE ships cue_point annotations only; no beat/downbeat grid is available.",
    }
    write_json(out_dir / "baseline_summary.json", results)
    print(json.dumps(results, indent=2))


def command_all(args: argparse.Namespace) -> None:
    jams_dir = Path(args.jams_dir) if args.jams_dir else default_jams_dir()
    benchmark = build_benchmark(jams_dir, Path(args.out), args.seed, args.test_fraction)
    write_json(Path(args.out) / "benchmark_summary.json", benchmark_summary(benchmark))
    baseline_args = argparse.Namespace(dataset=str(Path(args.out) / "mdjcue_benchmark.json"), out=args.out, tolerance=args.tolerance)
    command_baseline(baseline_args)


def default_jams_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    candidates = sorted((root / "research" / "originals").glob("M-DJCUE-*/annotations"))
    if not candidates:
        raise BenchmarkError("no research/originals/M-DJCUE-*/annotations directory found")
    return candidates[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="parse official JAMS annotations and make a grouped split")
    build.add_argument("--jams-dir", default=None)
    build.add_argument("--out", default="outputs")
    build.add_argument("--seed", type=int, default=20261020)
    build.add_argument("--test-fraction", type=float, default=0.3)
    build.set_defaults(func=command_build)

    baseline = subparsers.add_parser("baseline", help="run metadata-only position prior baselines")
    baseline.add_argument("--dataset", default="outputs/mdjcue_benchmark.json")
    baseline.add_argument("--out", default="outputs")
    baseline.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_S)
    baseline.set_defaults(func=command_baseline)

    evaluate = subparsers.add_parser("evaluate", help="evaluate an external prediction JSON")
    evaluate.add_argument("--dataset", default="outputs/mdjcue_benchmark.json")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--out", default="outputs/evaluation_external.json")
    evaluate.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_S)
    evaluate.add_argument("--split", default="test", choices=("test", "train"))
    evaluate.set_defaults(func=command_evaluate)

    all_parser = subparsers.add_parser("all", help="build and run both baselines")
    all_parser.add_argument("--jams-dir", default=None)
    all_parser.add_argument("--out", default="outputs")
    all_parser.add_argument("--seed", type=int, default=20261020)
    all_parser.add_argument("--test-fraction", type=float, default=0.3)
    all_parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_S)
    all_parser.set_defaults(func=command_all)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
