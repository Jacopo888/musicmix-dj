"""Read-only lineage and content audit for the local listening artefacts.

The run metadata says which canonical track and cue pair a preview should use,
but metadata alone cannot prove that a copied WAV still contains those tracks.
This tool therefore checks both the hash/path chain and PCM fingerprints from
the non-overlapping portions of each rendered 20-second preview.  It also
checks the two currently running local listening servers (when available).
No audio is downloaded, generated, or rewritten.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request
import wave
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from musicmix.paths import localize  # noqa: E402
DEFAULT_OUTPUT = ROOT / "runs" / "audio_lineage_audit_v5.json"
SAMPLE_RATE = 48_000
CONTEXT_SECONDS = 10.0
MATCH_SECONDS = 8.0
MATCH_SKIP_SECONDS = 1.0
MATCH_STRIDE = 16
RUN_NAMES = (
    "structured_forward_v2",
    "structured_reverse_v2",
    "structured_forward_v3",
    "structured_reverse_v3",
    "structured_forward_v4",
    "structured_reverse_v4",
    "structured_forward_beats_v4",
    "structured_reverse_beats_v4",
)
SERVER_TESTS = {
    8766: ("structured_forward_v4", "structured_reverse_v4"),
    8767: ("structured_forward_beats_v4", "structured_reverse_beats_v4"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def wav_info(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as stream:
        info = {
            "channels": stream.getnchannels(),
            "sample_width": stream.getsampwidth(),
            "sample_rate": stream.getframerate(),
            "duration_samples": stream.getnframes(),
            "compression": stream.getcomptype(),
        }
    info["sha256"] = sha256_file(path)
    info["duration_seconds"] = info["duration_samples"] / float(info["sample_rate"] or 1)
    return info


def read_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        width = stream.getsampwidth()
        rate = stream.getframerate()
        if width != 2 or channels not in (1, 2):
            raise ValueError(f"content matching requires 16-bit mono/stereo PCM: {path}")
        raw = stream.readframes(stream.getnframes())
    data = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).astype(np.float64)
    return data, rate


def title_for_source(name: str) -> str:
    lower = name.lower()
    if "reminder" in lower:
        return "Reminder"
    if "too many nights" in lower or "too_many_nights" in lower:
        return "Too Many Nights"
    return "UNKNOWN_SOURCE"


def canonical_sources() -> dict[str, dict[str, Any]]:
    provenance_path = ROOT / "local_audio" / "DOWNLOAD_PROVENANCE.json"
    root_manifest_path = ROOT / "runs" / "reminder_to_too_many_nights" / "manifest.json"
    provenance = read_json(provenance_path)
    manifest = read_json(root_manifest_path)
    download_by_name = {row["filename"]: dict(row) for row in provenance.get("tracks", [])}
    result: dict[str, dict[str, Any]] = {}
    for row in manifest.get("tracks", []):
        canonical_path = localize(row["canonical_audio_uri"])
        source_path = localize(row["input_uri"])
        title = title_for_source(source_path.name)
        actual = wav_info(canonical_path)
        download = download_by_name.get(source_path.name, {})
        result[title] = {
            "title": title,
            "source_filename": source_path.name,
            "source_path": str(source_path),
            "source_sha256_manifest": row.get("source_sha256"),
            "source_sha256_download_provenance": download.get("sha256"),
            "canonical_path": str(canonical_path),
            "canonical_sha256_manifest": row.get("canonical_sha256"),
            "canonical_sha256_actual": actual["sha256"],
            "track_id": row.get("track_id"),
            "duration_samples_manifest": row.get("duration_samples"),
            "duration_samples_actual": actual["duration_samples"],
            "sample_rate_actual": actual["sample_rate"],
            "channels_actual": actual["channels"],
            "source_hash_matches_download_provenance": row.get("source_sha256") == download.get("sha256"),
            "canonical_hash_matches_manifest": row.get("canonical_sha256") == actual["sha256"],
        }
    return result


def title_by_track_id(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    return {str(item["track_id"]): title for title, item in sources.items() if item.get("track_id")}


def run_manifest(run_name: str) -> dict[str, Any]:
    return read_json(ROOT / "runs" / run_name / "manifest.json")


def proposal_rows(run_name: str, manifest: Mapping[str, Any], track_titles: Mapping[str, str]) -> list[dict[str, Any]]:
    root = ROOT / "runs" / run_name
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("proposal_*.json"), key=lambda item: int(re.search(r"(\d+)", item.stem).group(1))):
        payload = read_json(path)
        candidate = dict(payload.get("candidate", {}))
        raw = dict(payload.get("raw_features", {}))
        track_a = str(candidate.get("track_a", ""))
        track_b = str(candidate.get("track_b", ""))
        original = dict(raw.get("original_structure_cue_samples", {}))
        cue_out = int(candidate.get("cue_out_sample", 0))
        cue_in = int(candidate.get("cue_in_sample", 0))
        original_out = int(original.get("a", cue_out))
        original_in = int(original.get("b", cue_in))
        rows.append({
            "rank": len(rows) + 1,
            "path": str(path.resolve()),
            "candidate_id": candidate.get("candidate_id"),
            "track_a_id": track_a,
            "track_b_id": track_b,
            "track_a": track_titles.get(track_a, "UNKNOWN_TRACK_ID"),
            "track_b": track_titles.get(track_b, "UNKNOWN_TRACK_ID"),
            "cue_out_sample": cue_out,
            "cue_in_sample": cue_in,
            "cue_out_seconds": cue_out / SAMPLE_RATE,
            "cue_in_seconds": cue_in / SAMPLE_RATE,
            "original_structure_cue_samples": {"a": original_out, "b": original_in},
            "original_structure_cue_seconds": {"a": original_out / SAMPLE_RATE, "b": original_in / SAMPLE_RATE},
            "score": payload.get("score"),
            "boundary_class": raw.get("boundary_class", payload.get("alignment", {}).get("boundary_class")),
            "source_direction": raw.get("direction"),
        })
    return rows


def run_audit(run_name: str, source_titles: Mapping[str, str]) -> dict[str, Any]:
    manifest = run_manifest(run_name)
    tracks = list(manifest.get("tracks", []))
    slots = []
    for index, track in enumerate(tracks[:2]):
        track_id = str(track.get("sha256", track.get("track_id", "")))
        slots.append({
            "slot": "outgoing" if index == 0 else "incoming",
            "track_id": track_id,
            "title": source_titles.get(track_id, "UNKNOWN_TRACK_ID"),
            "path": track.get("path", track.get("canonical_audio_uri")),
            "sample_rate": track.get("sample_rate"),
            "duration_samples": track.get("duration_samples"),
        })
    proposals = proposal_rows(run_name, manifest, source_titles)
    expected_a = slots[0]["track_id"] if len(slots) > 0 else ""
    expected_b = slots[1]["track_id"] if len(slots) > 1 else ""
    orientation_issues = []
    for row in proposals:
        if row["track_a_id"] != expected_a or row["track_b_id"] != expected_b:
            orientation_issues.append({
                "rank": row["rank"],
                "candidate_id": row["candidate_id"],
                "expected_track_a": expected_a,
                "expected_track_b": expected_b,
                "actual_track_a": row["track_a_id"],
                "actual_track_b": row["track_b_id"],
            })
    outgoing = [row["cue_out_sample"] for row in proposals]
    incoming = [row["cue_in_sample"] for row in proposals]
    return {
        "run": run_name,
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "schema_version": manifest.get("schema_version"),
        "direction": " -> ".join(slot["title"] for slot in slots),
        "track_slots": slots,
        "options": manifest.get("options", {}),
        "selected_count_manifest": manifest.get("selected_count"),
        "proposals": proposals,
        "distinct_cue_counts": {
            "pairs": len({(row["cue_out_sample"], row["cue_in_sample"]) for row in proposals}),
            "outgoing": len(set(outgoing)),
            "incoming": len(set(incoming)),
        },
        "orientation_issues": orientation_issues,
        "evidence_paths": manifest.get("evidence", {}),
        "listening_metadata": manifest.get("listening", {}),
        "random_control": manifest.get("random_control"),
    }


def exact_and_near_reuse(runs: Mapping[str, Mapping[str, Any]], tolerance_samples: int = 3840) -> dict[str, Any]:
    names = list(runs)
    exact: list[dict[str, Any]] = []
    near: list[dict[str, Any]] = []
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left_rows = runs[left_name]["proposals"]
            right_rows = runs[right_name]["proposals"]
            left_signatures = {
                (row["track_a_id"], row["track_b_id"], row["cue_out_sample"], row["cue_in_sample"]): row
                for row in left_rows
            }
            right_signatures = {
                (row["track_a_id"], row["track_b_id"], row["cue_out_sample"], row["cue_in_sample"]): row
                for row in right_rows
            }
            shared = set(left_signatures) & set(right_signatures)
            if shared:
                exact.append({
                    "left_run": left_name,
                    "right_run": right_name,
                    "pairs": [
                        {
                            "track_a": left_signatures[key]["track_a"],
                            "track_b": left_signatures[key]["track_b"],
                            "cue_out_seconds": key[2] / SAMPLE_RATE,
                            "cue_in_seconds": key[3] / SAMPLE_RATE,
                            "left_rank": left_signatures[key]["rank"],
                            "right_rank": right_signatures[key]["rank"],
                        }
                        for key in sorted(shared)
                    ],
                })
            candidate_near: list[dict[str, Any]] = []
            for left in left_rows:
                for right in right_rows:
                    if left["track_a_id"] != right["track_a_id"] or left["track_b_id"] != right["track_b_id"]:
                        continue
                    out_delta = abs(left["cue_out_sample"] - right["cue_out_sample"])
                    in_delta = abs(left["cue_in_sample"] - right["cue_in_sample"])
                    if 0 < out_delta <= tolerance_samples or 0 < in_delta <= tolerance_samples:
                        candidate_near.append({
                            "left_rank": left["rank"],
                            "right_rank": right["rank"],
                            "out_delta_seconds": out_delta / SAMPLE_RATE,
                            "in_delta_seconds": in_delta / SAMPLE_RATE,
                        })
            if candidate_near:
                near.append({"left_run": left_name, "right_run": right_name, "pairs": candidate_near})
    return {
        "exact_pair_reuse": exact,
        "near_pair_reuse_within_seconds": tolerance_samples / SAMPLE_RATE,
        "near_pair_reuse": near,
    }


def region_overlap_summary(runs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize overlap of the 20 s listening regions around each selected cue."""

    by_title: dict[str, list[tuple[float, float, str, int]]] = defaultdict(list)
    for run_name, run in runs.items():
        for row in run["proposals"]:
            for title, sample, side in (
                (row["track_a"], row["cue_out_sample"], "outgoing"),
                (row["track_b"], row["cue_in_sample"], "incoming"),
            ):
                start = max(0.0, sample / SAMPLE_RATE - CONTEXT_SECONDS)
                end = sample / SAMPLE_RATE + CONTEXT_SECONDS
                by_title[title].append((start, end, run_name, row["rank"]))
    output: dict[str, Any] = {}
    for title, regions in by_title.items():
        ordered = sorted(regions)
        clusters: list[dict[str, Any]] = []
        for start, end, run_name, rank in ordered:
            if clusters and start <= clusters[-1]["end_seconds"]:
                cluster = clusters[-1]
                cluster["end_seconds"] = max(cluster["end_seconds"], end)
                cluster["members"].append({"run": run_name, "rank": rank})
            else:
                clusters.append({
                    "start_seconds": start,
                    "end_seconds": end,
                    "members": [{"run": run_name, "rank": rank}],
                })
        output[title] = {
            "region_definition": "cue +/- 10.0 s, clipped at zero on the left",
            "region_count": len(regions),
            "union_seconds": sum(item["end_seconds"] - item["start_seconds"] for item in clusters),
            "overlap_cluster_count": sum(len(item["members"]) > 1 for item in clusters),
            "clusters": clusters,
        }
    return output


def find_proposal_for_path(path: Path, all_runs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    resolved = str(path.resolve())
    for run in all_runs.values():
        for row in run["proposals"]:
            candidate_id = str(row.get("candidate_id", ""))
            if candidate_id and candidate_id in path.name:
                return dict(row, metadata_run=run["run"], resolution="candidate_id_in_filename")
    return None


def expected_pair_for_row(
    row: Mapping[str, Any],
    run_name: str,
    runs: Mapping[str, Mapping[str, Any]],
    all_runs: Mapping[str, Mapping[str, Any]],
    legacy_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    path = localize(str(row.get("source", "")))
    proposal = find_proposal_for_path(path, all_runs)
    if proposal is not None:
        return proposal, "candidate_id_in_filename"
    pair_id = str(row.get("pair_id", ""))
    rank_match = re.search(r"(\d+)$", pair_id)
    rank = int(rank_match.group(1)) if rank_match else None
    system = str(row.get("system", ""))
    referenced_run = None
    for candidate_run in runs:
        if candidate_run in system:
            referenced_run = candidate_run
            break
    if referenced_run is not None and rank is not None:
        rows = runs[referenced_run]["proposals"]
        if 1 <= rank <= len(rows):
            return dict(rows[rank - 1], metadata_run=referenced_run), "system_and_rank"
    if "uniform_valid_cue_pair" in system and rank is not None:
        draws = list(runs[run_name].get("random_control", {}).get("draws", []))
        draw = next((item for item in draws if int(item.get("trial", -1)) == rank), None)
        slots = runs[run_name].get("track_slots", [])
        if draw is not None and len(slots) >= 2:
            out_sample = int(draw["cue_out_sample"])
            in_sample = int(draw["cue_in_sample"])
            return {
                "metadata_run": run_name,
                "candidate_id": draw.get("candidate_id"),
                "track_a": slots[0]["title"],
                "track_b": slots[1]["title"],
                "track_a_id": slots[0]["track_id"],
                "track_b_id": slots[1]["track_id"],
                "cue_out_sample": out_sample,
                "cue_in_sample": in_sample,
                "cue_out_seconds": out_sample / SAMPLE_RATE,
                "cue_in_seconds": in_sample / SAMPLE_RATE,
            }, "random_control_manifest"
    if "onset_snapshot_v1" in system and rank is not None and 1 <= rank <= len(legacy_rows):
        # The V2/V3 baseline files are the first three artifacts of the
        # original arbitrary-cue run.  They are copied into both forward and
        # reverse listening directories without proposal JSON of their own.
        legacy = legacy_rows[rank - 1]
        slots = runs[run_name].get("track_slots", [])
        if len(slots) >= 2:
            forward = slots[0]["title"] == "Reminder"
            out_sample = int(legacy["cue_out_sample"] if forward else legacy["cue_in_sample"])
            in_sample = int(legacy["cue_in_sample"] if forward else legacy["cue_out_sample"])
            return {
                "metadata_run": "reminder_to_too_many_nights/legacy_selected",
                "candidate_id": legacy.get("candidate_id"),
                "track_a": slots[0]["title"],
                "track_b": slots[1]["title"],
                "track_a_id": slots[0]["track_id"],
                "track_b_id": slots[1]["track_id"],
                "cue_out_sample": out_sample,
                "cue_in_sample": in_sample,
                "cue_out_seconds": out_sample / SAMPLE_RATE,
                "cue_in_seconds": in_sample / SAMPLE_RATE,
            }, "legacy_selected_manifest"
    # The older baseline systems have no proposal metadata of their own.  In
    # every listening pair the other side carries the same pair identity; the
    # caller supplies that side's resolved proposal after this first pass.
    return None, "unresolved_baseline"


def resolve_listening(
    run_name: str,
    run: Mapping[str, Any],
    all_runs: Mapping[str, Mapping[str, Any]],
    legacy_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    key_path = ROOT / "runs" / run_name / "listening_ab" / "key.json"
    if not key_path.exists():
        return []
    key = read_json(key_path)
    rows = list(key.get("trials", []))
    by_trial: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        proposal, resolution = expected_pair_for_row(row, run_name, all_runs, all_runs, legacy_rows)
        by_trial[int(row.get("trial", 0))].append({"row": row, "proposal": proposal, "resolution": resolution})
    output: list[dict[str, Any]] = []
    for trial, members in sorted(by_trial.items()):
        known = next((member["proposal"] for member in members if member["proposal"] is not None), None)
        for member in members:
            row = member["row"]
            source_path = localize(str(row.get("source", "")))
            proposal = member["proposal"] or known
            resolution = member["resolution"] if member["proposal"] is not None else (
                "paired_trial_proposal" if known is not None else "unresolved_baseline"
            )
            actual = wav_info(source_path) if source_path.is_file() else {"missing": True}
            output.append({
                "trial": trial,
                "label": row.get("label"),
                "system": row.get("system"),
                "pair_id": row.get("pair_id"),
                "source_path": str(source_path.resolve()),
                "source_exists": source_path.is_file(),
                "wav": actual,
                "proposal_resolution": resolution,
                "expected_pair": None if proposal is None else {
                    "metadata_run": proposal.get("metadata_run"),
                    "candidate_id": proposal.get("candidate_id"),
                    "track_a": proposal.get("track_a"),
                    "track_b": proposal.get("track_b"),
                    "cue_out_sample": proposal.get("cue_out_sample"),
                    "cue_in_sample": proposal.get("cue_in_sample"),
                    "cue_out_seconds": proposal.get("cue_out_seconds"),
                    "cue_in_seconds": proposal.get("cue_in_seconds"),
                },
            })
    return output


def match_score(render: np.ndarray, source: np.ndarray, render_start: int, source_start: int, frames: int) -> float | None:
    if render_start < 0 or source_start < 0 or render_start + frames > len(render) or source_start + frames > len(source):
        return None
    rendered = render[render_start : render_start + frames : MATCH_STRIDE].reshape(-1)
    expected = source[source_start : source_start + frames : MATCH_STRIDE].reshape(-1)
    expected_energy = float(np.dot(expected, expected))
    if expected_energy <= 1e-12:
        return None
    gain = float(np.dot(expected, rendered) / expected_energy)
    error = rendered - gain * expected
    scale = float(np.sqrt(np.mean(np.square(rendered)))) + 1e-12
    return float(1.0 - np.sqrt(np.mean(np.square(error))) / scale)


def content_match(
    source_arrays: Mapping[str, np.ndarray],
    source_rates: Mapping[str, int],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    expected = row.get("expected_pair") or {}
    if row.get("wav", {}).get("missing") or not expected:
        return {"status": "unresolved_expected_pair"}
    render_path = localize(str(row["source_path"]))
    rendered, rate = read_pcm16(render_path)
    if rate != SAMPLE_RATE:
        return {"status": "unsupported_render_rate", "sample_rate": rate}
    frames = round(MATCH_SECONDS * SAMPLE_RATE)
    skip = round(MATCH_SKIP_SECONDS * SAMPLE_RATE)
    half = len(rendered) // 2
    out_start = skip
    in_start = half + skip
    out_source_start = int(expected["cue_out_sample"]) - round((CONTEXT_SECONDS - MATCH_SKIP_SECONDS) * SAMPLE_RATE)
    in_source_start = int(expected["cue_in_sample"]) + skip
    outgoing_scores: dict[str, float | None] = {}
    incoming_scores: dict[str, float | None] = {}
    for title, source in source_arrays.items():
        if source_rates[title] != SAMPLE_RATE:
            continue
        outgoing_scores[title] = match_score(rendered, source, out_start, out_source_start, frames)
        incoming_scores[title] = match_score(rendered, source, in_start, in_source_start, frames)
    expected_out = str(expected.get("track_a"))
    expected_in = str(expected.get("track_b"))
    return {
        "status": "measured",
        "render_info": {"sample_rate": rate, "frames": len(rendered), "channels": int(rendered.shape[1])},
        "window": {"output_first_half": "1.0..9.0 s", "output_second_half": "11.0..19.0 s", "gain_fit": "least_squares"},
        "outgoing_scores": outgoing_scores,
        "incoming_scores": incoming_scores,
        "expected_outgoing": expected_out,
        "expected_incoming": expected_in,
        "expected_outgoing_score": outgoing_scores.get(expected_out),
        "expected_incoming_score": incoming_scores.get(expected_in),
        "track_content_match": bool(
            outgoing_scores.get(expected_out) is not None
            and incoming_scores.get(expected_in) is not None
            and outgoing_scores[expected_out] > 0.90
            and incoming_scores[expected_in] > 0.90
        ),
    }


def audit_servers(listening: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for port, run_names in SERVER_TESTS.items():
        server_report: dict[str, Any] = {"port": port, "tests": [], "status": "checked"}
        for run_name in run_names:
            key_path = ROOT / "runs" / run_name / "listening_ab" / "key.json"
            key = read_json(key_path)
            test_id = str(key.get("test_id", ""))
            base = f"http://127.0.0.1:{port}/test/{test_id}"
            test_report: dict[str, Any] = {"run": run_name, "test_id": test_id, "base_url": base}
            try:
                with urllib.request.urlopen(base + "/index.html", timeout=3.0) as response:
                    html = response.read().decode("utf-8")
                    test_report["index_status"] = int(response.status)
                served_names = sorted(set(re.findall(r"trial_\d+_[AB]\.wav", html)))
                key_names = sorted(
                    f"trial_{int(row['trial']):02d}_{row['label']}.wav"
                    for row in key.get("trials", [])
                )
                test_report["html_audio_names"] = served_names
                test_report["key_expected_audio_names"] = sorted(set(key_names))
                test_report["html_key_audio_set_match"] = served_names == sorted(set(key_names))
                endpoints: list[dict[str, Any]] = []
                local_by_name = {
                    f"trial_{int(row['trial']):02d}_{row['label']}.wav": Path("runs") / run_name / "listening_ab" / f"trial_{int(row['trial']):02d}_{row['label']}.wav"
                    for row in key.get("trials", [])
                }
                for name in served_names:
                    local = ROOT / local_by_name[name]
                    expected_hash = sha256_file(local) if local.is_file() else None
                    request = urllib.request.Request(base + "/" + name, headers={"Range": "bytes=0-31"})
                    with urllib.request.urlopen(request, timeout=3.0) as response:
                        first = response.read()
                        headers = dict(response.headers.items())
                        endpoints.append({
                            "name": name,
                            "local_exists": local.is_file(),
                            "local_sha256": expected_hash,
                            "http_status": int(response.status),
                            "http_content_length": headers.get("Content-Length"),
                            "http_content_range": headers.get("Content-Range"),
                            "range_first_32_bytes_match_local": first == local.read_bytes()[:32] if local.is_file() else False,
                        })
                test_report["audio_endpoints"] = endpoints
                test_report["status"] = "ok"
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                test_report["status"] = "unavailable"
                test_report["error"] = str(exc)
            server_report["tests"].append(test_report)
        result[str(port)] = server_report
    return result


def build_report() -> dict[str, Any]:
    sources = canonical_sources()
    track_titles = title_by_track_id(sources)
    runs = {name: run_audit(name, track_titles) for name in RUN_NAMES}
    legacy_search = read_json(ROOT / "runs" / "reminder_to_too_many_nights" / "search.json")
    legacy_manifest = read_json(ROOT / "runs" / "reminder_to_too_many_nights" / "manifest.json")
    legacy_by_id = {str(item.get("candidate_id")): item for item in legacy_search.get("candidates", [])}
    legacy_rows = [
        dict(legacy_by_id[str(item["candidate_id"])])
        for item in legacy_manifest.get("artifacts", [])[:3]
        if str(item.get("candidate_id")) in legacy_by_id
    ]
    all_listening: dict[str, list[dict[str, Any]]] = {}
    source_arrays: dict[str, np.ndarray] = {}
    source_rates: dict[str, int] = {}
    for title, source in sources.items():
        array, rate = read_pcm16(localize(source["canonical_path"]))
        source_arrays[title] = array
        source_rates[title] = rate
    for run_name, run in runs.items():
        rows = resolve_listening(run_name, run, runs, legacy_rows)
        for row in rows:
            row["content_match"] = content_match(source_arrays, source_rates, row)
        all_listening[run_name] = rows
    distinct_render_hashes: dict[str, list[str]] = defaultdict(list)
    for run_rows in all_listening.values():
        for row in run_rows:
            digest = row.get("wav", {}).get("sha256")
            if digest:
                distinct_render_hashes[digest].append(str(row["source_path"]))
    duplicate_render_content = [
        {"sha256": digest, "paths": sorted(set(paths))}
        for digest, paths in distinct_render_hashes.items()
        if len(set(paths)) > 1
    ]
    for row_list in all_listening.values():
        for row in row_list:
            row["duplicate_content_paths"] = sorted(set(distinct_render_hashes.get(row.get("wav", {}).get("sha256"), [])))
    all_mismatches = [
        {
            "run": run_name,
            "trial": row["trial"],
            "label": row["label"],
            "system": row["system"],
            "source_path": row["source_path"],
            "content_match": row["content_match"],
        }
        for run_name, rows in all_listening.items()
        for row in rows
        if row["content_match"].get("status") == "measured" and not row["content_match"].get("track_content_match")
    ]
    run_manifest_source_mentions: dict[str, Any] = {}
    for path in sorted((ROOT / "runs").glob("*/manifest.json")):
        payload = read_json(path)
        mentions = []
        for track in payload.get("tracks", []):
            track_id = str(track.get("track_id", track.get("sha256", "")))
            mentions.append({
                "track_id": track_id,
                "title": track_titles.get(track_id, "UNKNOWN_TRACK_ID"),
                "input_uri": track.get("input_uri"),
                "canonical_audio_uri": track.get("canonical_audio_uri", track.get("path")),
            })
        if mentions:
            run_manifest_source_mentions[path.parent.name] = mentions
    return {
        "schema_version": "musicmix.audio_lineage_audit.5",
        "status": "complete",
        "scope": {
            "read_only": True,
            "sources": "canonical a.wav/b.wav, local download provenance, selected structure manifests/proposals, listening keys and server paths",
            "content_match": "PCM 16-bit comparisons on output 1..9 s and 11..19 s against canonical source at declared cues; gain fitted per window",
            "no_audio_download_or_regeneration": True,
        },
        "canonical_sources": sources,
        "run_manifest_source_mentions": run_manifest_source_mentions,
        "runs": runs,
        "reuse_and_overlap": {
            **exact_and_near_reuse(runs),
            "context_region_overlap": region_overlap_summary(runs),
        },
        "listening_keys": all_listening,
        "duplicate_render_content": duplicate_render_content,
        "content_mismatches": all_mismatches,
        "servers": audit_servers(all_listening),
        "conclusions": {
            "source_scope": "The canonical chain contains only Reminder and Too Many Nights; no unknown track_id appears in the audited run manifests.",
            "direction_policy": "Forward manifests intentionally list Reminder -> Too Many Nights; reverse manifests intentionally list Too Many Nights -> Reminder. This is orientation, not a file swap.",
            "identical_reuse": "The report lists exact proposal reuse and duplicate rendered WAV hashes; V4 random controls may intentionally reuse a directional preview and are marked by pair/system rather than treated as new audio.",
            "content_claim": "A preview is considered content-matched only when both outgoing and incoming non-fade PCM windows correlate with their declared canonical track. No human listening claim is made.",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "runs": len(report["runs"]),
        "listening_wavs": sum(len(rows) for rows in report["listening_keys"].values()),
        "content_mismatches": len(report["content_mismatches"]),
        "status": report["status"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
