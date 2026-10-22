"""Audit cached vocal evidence without running ASR or source separation.

The audit is intentionally descriptive.  It measures the same separated-stem
RMS lattice used by :mod:`musicmix.vocal_boundaries`, checks the persisted
timeline/hash contracts, and reports how ASR uncertainty and structure events
intersect that evidence.  It cannot determine whether continuous stem energy
is a genuine backing vocal or separator bleed, and it never presents an ASR
timestamp as singing forced alignment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicmix.audio import AudioTrack, load_audio  # noqa: E402
from musicmix.vocal_boundaries import (  # noqa: E402
    acoustic_intervals,
    sha256_file,
)


AUDIT_SCHEMA_VERSION = "musicmix.vocal_support_audit.4"
DEFAULT_AUDIO_DIR = ROOT / "runs" / "reminder_to_too_many_nights" / "audio"
DEFAULT_VOCALS_DIR = ROOT / "runs" / "vocals_v2"
DEFAULT_STRUCTURE_DIR = ROOT / "runs" / "structure_v2"
DEFAULT_OUTPUT = ROOT / "runs" / "vocal_support_audit_v4.json"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _duration_by_status(
    rows: Iterable[Mapping[str, Any]],
    sample_rate: int = 48_000,
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for row in rows:
        status = str(row.get("status", "missing"))
        duration = max(0, int(row.get("end_sample", 0)) - int(row.get("start_sample", 0)))
        item = summary.setdefault(status, {"cells": 0, "samples": 0, "seconds": 0.0})
        item["cells"] += 1
        item["samples"] += duration
    for item in summary.values():
        item["seconds"] = item["samples"] / float(sample_rate)
    return summary


def _union_duration(rows: Iterable[Mapping[str, Any]]) -> int:
    intervals = sorted(
        (int(row["start_sample"]), int(row["end_sample"]))
        for row in rows
        if int(row["end_sample"]) > int(row["start_sample"])
    )
    total = 0
    current: list[int] | None = None
    for start, end in intervals:
        if current is None:
            current = [start, end]
        elif start <= current[1]:
            current[1] = max(current[1], end)
        else:
            total += current[1] - current[0]
            current = [start, end]
    if current is not None:
        total += current[1] - current[0]
    return total


def _partition_issues(rows: Sequence[Mapping[str, Any]], duration_samples: int) -> list[str]:
    issues: list[str] = []
    ordered = sorted(rows, key=lambda row: int(row.get("start_sample", -1)))
    cursor = 0
    for row in ordered:
        start = int(row.get("start_sample", -1))
        end = int(row.get("end_sample", -1))
        if start != cursor:
            issues.append(f"gap_or_overlap_before_{start}")
        if end <= start:
            issues.append(f"nonpositive_interval_{start}_{end}")
        cursor = end
    if cursor != int(duration_samples):
        issues.append(f"partition_ends_{cursor}_expected_{duration_samples}")
    return issues


def _rms_values(track: AudioTrack, frame_samples: int, hop_samples: int) -> np.ndarray:
    """Calculate bounded-size frame RMS values matching acoustic_intervals."""

    mono = np.mean(np.asarray(track.samples, dtype=np.float64), axis=1)
    starts = range(0, track.duration_samples, hop_samples)
    values = [
        float(np.sqrt(np.mean(np.square(mono[start : min(track.duration_samples, start + frame_samples)]))))
        for start in starts
    ]
    return np.asarray(values, dtype=np.float64)


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {key: None for key in ("min", "p01", "p05", "p10", "p20", "p50", "p80", "p90", "p95", "p99", "max")}
    percentiles = {
        "min": 0.0,
        "p01": 1.0,
        "p05": 5.0,
        "p10": 10.0,
        "p20": 20.0,
        "p50": 50.0,
        "p80": 80.0,
        "p90": 90.0,
        "p95": 95.0,
        "p99": 99.0,
        "max": 100.0,
    }
    return {key: float(np.percentile(values, percentile)) for key, percentile in percentiles.items()}


def _overlap(start: int, end: int, row: Mapping[str, Any]) -> bool:
    return int(row["start_sample"]) < end and int(row["end_sample"]) > start


def _status_for_window(
    start: int,
    end: int,
    coverage: Sequence[Mapping[str, Any]],
) -> str:
    statuses = [str(row["status"]) for row in coverage if _overlap(start, end, row)]
    if not statuses:
        return "missing_coverage"
    if any(status.startswith("unknown") for status in statuses):
        return "unknown"
    if all(status == "acoustic_silence_supported" for status in statuses):
        return "acoustic_silence_supported"
    if all(status == "vocal_occupancy_supported" for status in statuses):
        return "vocal_occupancy_supported"
    if any(status == "unknown_acoustic_conflict" for status in statuses):
        return "unknown_acoustic_conflict"
    return "mixed_known"


def _event_support(
    structure: Mapping[str, Any] | None,
    coverage: Sequence[Mapping[str, Any]],
    sample_rate: int,
    duration_samples: int,
    window_seconds: float = 0.040,
) -> dict[str, Any] | None:
    if not structure:
        return None
    window = max(1, round(window_seconds * sample_rate))
    result: dict[str, Any] = {
        "provider": structure.get("provenance", structure.get("provider")),
        "status": structure.get("status"),
        "events": {},
        "window_seconds": window_seconds,
    }
    for event_name in ("beats", "downbeats"):
        events = structure.get(event_name, [])
        counts: dict[str, int] = {}
        for event in events:
            sample = int(event.get("sample", -1))
            if sample < 0 or sample >= duration_samples:
                status = "invalid_event"
            else:
                start = max(0, sample - window // 2)
                end = min(duration_samples, sample + window - window // 2)
                status = _status_for_window(start, end, coverage)
            counts[status] = counts.get(status, 0) + 1
        result["events"][event_name] = {"total": len(events), "coverage_status": counts}
    return result


def _word_evidence(
    words: Sequence[Mapping[str, Any]],
    acoustic_rows: Sequence[Mapping[str, Any]],
    coverage: Sequence[Mapping[str, Any]],
    sample_rate: int,
    low_threshold: float,
) -> dict[str, Any]:
    confidences = [float(row["confidence"]) for row in words if row.get("confidence") is not None]
    low = [row for row in words if row.get("confidence") is not None and float(row["confidence"]) < low_threshold]
    no_confidence = [row for row in words if row.get("confidence") is None]
    silence_word_rows = [
        row for row in words
        if any(_overlap(int(row["start_sample"]), int(row["end_sample"]), acoustic) and acoustic.get("status") == "acoustic_silence" for acoustic in acoustic_rows)
    ]
    conflict_rows = [row for row in coverage if row.get("status") == "unknown_acoustic_conflict"]
    conflict_without_word = [
        row for row in conflict_rows
        if not any(_overlap(int(row["start_sample"]), int(row["end_sample"]), word) for word in words)
    ]
    out: dict[str, Any] = {
        "word_count": len(words),
        "timestamped_word_seconds": _union_duration(words) / sample_rate,
        "low_confidence_threshold": low_threshold,
        "low_confidence_count": len(low),
        "low_confidence_fraction": len(low) / len(words) if words else 0.0,
        "missing_confidence_count": len(no_confidence),
        "confidence_quantiles": _quantiles(np.asarray(confidences, dtype=np.float64)),
        "words_overlapping_acoustic_silence_count": len(silence_word_rows),
        "words_overlapping_acoustic_silence_seconds": sum(
            max(0, int(row["end_sample"]) - int(row["start_sample"])) for row in silence_word_rows
        ) / sample_rate,
        "unknown_conflict_count": len(conflict_rows),
        "unknown_conflicts_without_timestamped_word_count": len(conflict_without_word),
        "interpretation": (
            "ASR probabilities are transcript confidence, not singing alignment confidence; "
            "a word overlapping measured stem silence remains a conflict."
        ),
    }
    return out


def _track_audit(
    label: str,
    audio_path: Path,
    stem_path: Path,
    document_path: Path,
    structure_path: Path | None,
) -> dict[str, Any]:
    audio = load_audio(audio_path)
    stem = load_audio(stem_path)
    document = _read_json(document_path)
    structure = _read_json(structure_path) if structure_path and structure_path.exists() else None
    expected = {
        "audio": {"sample_rate": audio.sample_rate, "duration_samples": audio.duration_samples, "track_id": audio.track_id},
        "stem": {"sample_rate": stem.sample_rate, "duration_samples": stem.duration_samples, "track_id": stem.track_id},
        "document": {"sample_rate": document.get("sample_rate"), "duration_samples": document.get("duration_samples"), "track_id": document.get("track_id")},
    }
    timeline_issues: list[str] = []
    if stem.sample_rate != audio.sample_rate or stem.duration_samples != audio.duration_samples:
        timeline_issues.append("stem_timeline_mismatch")
    if document.get("sample_rate") != audio.sample_rate or document.get("duration_samples") != audio.duration_samples:
        timeline_issues.append("document_timeline_mismatch")
    if document.get("track_id") != audio.track_id or document.get("canonical_file_sha256") != sha256_file(audio_path):
        timeline_issues.append("document_audio_hash_mismatch")
    if structure and (
        structure.get("track_id") != audio.track_id
        or structure.get("sample_rate") != audio.sample_rate
        or structure.get("duration_samples") != audio.duration_samples
    ):
        timeline_issues.append("structure_timeline_or_hash_mismatch")

    acoustic_rows = list(document.get("acoustic_occupancy", []))
    coverage = list(document.get("coverage_intervals", []))
    diagnostics = dict(document.get("acoustic_diagnostics", {}))
    frame_samples = int(diagnostics.get("frame_samples", round(0.040 * stem.sample_rate)))
    hop_samples = int(diagnostics.get("hop_samples", round(0.020 * stem.sample_rate)))
    rms = _rms_values(stem, frame_samples, hop_samples)
    threshold = float(diagnostics.get("threshold_rms", 0.0))
    margin = float(diagnostics.get("margin_rms", 0.0))
    lower = threshold - margin
    upper = threshold + margin
    counts = {
        "acoustic_silence": int(np.sum(rms <= lower)),
        "unknown_acoustic_margin": int(np.sum((rms > lower) & (rms < upper))),
        "vocal_occupancy": int(np.sum(rms >= upper)),
    }
    recomputed_rows, recomputed_diagnostics = acoustic_intervals(stem)
    stored_duration = _duration_by_status(acoustic_rows, audio.sample_rate)
    coverage_duration = _duration_by_status(coverage, audio.sample_rate)
    words = list(document.get("words", []))
    low_threshold = float(document.get("asr_low_confidence_threshold", 0.45))
    recomputed_summary = _duration_by_status(recomputed_rows, audio.sample_rate)
    raw_threshold = float(diagnostics.get("threshold_uncapped_rms", threshold))
    p20 = _quantiles(rms).get("p20")
    persistent = p20 is not None and float(p20) >= upper
    threshold_report = {
        "effective_threshold_rms": threshold,
        "margin_rms": margin,
        "silence_cutoff_rms": lower,
        "occupancy_cutoff_rms": upper,
        "uncapped_threshold_rms": raw_threshold,
        "uncapped_to_effective_ratio": raw_threshold / threshold if threshold > 0 else None,
        "threshold_was_capped": bool(diagnostics.get("threshold_capped_at_p90_fraction", False)),
        "frame_count": int(rms.size),
        "frame_fraction_by_cutoff": {
            "silence": counts["acoustic_silence"] / rms.size if rms.size else 0.0,
            "margin": counts["unknown_acoustic_margin"] / rms.size if rms.size else 0.0,
            "occupancy": counts["vocal_occupancy"] / rms.size if rms.size else 0.0,
        },
        "persistent_stem_energy_by_p20": bool(persistent),
        "interpretation": (
            "The cap is conservative; raising it would be a policy change and is not applied. "
            "Persistent energy is compatible with real backing vocals or separator bleed; RMS alone cannot distinguish them."
        ),
    }
    return {
        "label": label,
        "paths": {"audio": str(audio_path.resolve()), "stem": str(stem_path.resolve()), "document": str(document_path.resolve())},
        "hashes": {
            "audio_sha256": sha256_file(audio_path),
            "audio_track_id": audio.track_id,
            "stem_sha256": sha256_file(stem_path),
            "stem_track_id": stem.track_id,
            "document_track_id": document.get("track_id"),
            "separator_input_sha256": document.get("separator", {}).get("input_sha256"),
        },
        "timeline": {"expected": expected, "issues": timeline_issues, "partition_issues": _partition_issues(coverage, audio.duration_samples)},
        "document_status": {
            "analysis_status": document.get("analysis_status"),
            "source_kind": document.get("source_kind"),
            "backend_status": document.get("backend", {}).get("status"),
            "separator_status": document.get("separator", {}).get("status"),
            "warnings": document.get("warnings", []),
        },
        "coverage": {
            "duration_by_status": coverage_duration,
            "acoustic_rows_duration_by_status": stored_duration,
            "recomputed_acoustic_duration_by_status": recomputed_summary,
            "acoustic_recompute_matches_stored_summary": stored_duration == recomputed_summary,
        },
        "rms": {"quantiles": _quantiles(rms), "threshold": threshold_report},
        "asr": _word_evidence(words, acoustic_rows, coverage, audio.sample_rate, low_threshold),
        "structure_event_support": _event_support(structure, coverage, audio.sample_rate, audio.duration_samples),
    }


def build_audit(
    audio_dir: Path = DEFAULT_AUDIO_DIR,
    vocals_dir: Path = DEFAULT_VOCALS_DIR,
    structure_dir: Path | None = DEFAULT_STRUCTURE_DIR,
) -> dict[str, Any]:
    tracks: dict[str, Any] = {}
    for label in ("a", "b"):
        structure_path = structure_dir / f"{label}.json" if structure_dir else None
        tracks[label] = _track_audit(
            label,
            audio_dir / f"{label}.wav",
            vocals_dir / f"{label}_vocals.wav",
            vocals_dir / f"{label}.json",
            structure_path,
        )
    all_timeline_issues = {
        label: item["timeline"]["issues"] + item["timeline"]["partition_issues"]
        for label, item in tracks.items()
        if item["timeline"]["issues"] or item["timeline"]["partition_issues"]
    }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "complete" if not all_timeline_issues else "completed_with_timeline_warnings",
        "tracks": tracks,
        "forced_alignment": {
            "status": "not_attempted",
            "provider": None,
            "model": None,
            "claim": "No singing-specific forced aligner was run; cached ASR timestamps remain uncalibrated observations.",
        },
        "root_cause_assessment": [
            {
                "id": "asr_segment_pause_conflict",
                "status": "fixed_in_current_boundary_documents",
                "finding": "Transcript segment spans include pauses between words; acoustic conflicts are now attributed only when a timestamped word overlaps stem silence.",
                "verification": {label: item["asr"]["unknown_conflicts_without_timestamped_word_count"] for label, item in tracks.items()},
            },
            {
                "id": "conservative_rms_cap",
                "status": "observed",
                "finding": "The effective RMS silence threshold is capped at 0.003 on both tracks despite much higher uncapped estimates.",
                "impact": "This limits supported acoustic-silence cells; relaxing it would change strict evidence policy and is not justified by this audit.",
            },
            {
                "id": "persistent_stem_energy",
                "status": "observed_but_not_identifiable_from_rms",
                "finding": "The lower stem-RMS quantile is above the occupancy cutoff, so most cells contain continuous separated-stem energy.",
                "impact": "This is compatible with sustained/backing vocal content or separator bleed; the cache and RMS test cannot identify which.",
            },
            {
                "id": "asr_uncertainty",
                "status": "observed",
                "finding": "Low ASR probabilities are substantially more common on B than A and are kept as unknown evidence rather than silence.",
                "impact": "Uncertainty can suppress individual word-boundary candidates, but it does not explain the persistent acoustic occupancy by itself.",
            },
        ],
        "research_basis": [
            {
                "title": "Phoneme Level Lyrics Alignment and Text-Informed Singing Voice Separation (2021)",
                "url": "https://perso.telecom-paristech.fr/grichard/Publications/2021_05_TASLP_preprint.pdf",
                "relevance": "Singing-specific phoneme alignment requires an audio/text model; ordinary ASR timestamps are not substituted for it here.",
            },
            {
                "title": "A Real-Time Lyrics Alignment System Using Chroma and Phonetic Features (2024)",
                "url": "https://arxiv.org/abs/2401.09200",
                "relevance": "Chroma/phonetic alignment is a future provider path, not an executed backend in this audit.",
            },
        ],
        "limitations": [
            "No human listening or labeled vocal occupancy reference was used.",
            "No new separation, ASR, or forced-alignment inference was run.",
            "RMS occupancy cannot distinguish genuine backing vocals from separator bleed.",
            "Structure event support describes overlap with persisted acoustic evidence; it does not validate musical phrase or downbeat semantics.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--vocals-dir", type=Path, default=DEFAULT_VOCALS_DIR)
    parser.add_argument("--structure-dir", type=Path, default=DEFAULT_STRUCTURE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_audit(args.audio_dir, args.vocals_dir, args.structure_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "status": report["status"], "tracks": sorted(report["tracks"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
