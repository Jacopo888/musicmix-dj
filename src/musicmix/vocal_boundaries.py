"""Conservative full-song vocal/word boundary artefacts.

This module deliberately treats ASR timestamps as observations, not as a
silence detector.  Missing words produce unknown intervals.  A separated
vocals stem may additionally provide an acoustic low-energy interval, but the
threshold and margin are stored in the JSON so downstream code can audit the
decision.

The optional Demucs and faster-whisper imports are lazy.  The repository's
small baseline therefore remains usable when the local model runtimes are not
available.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import wave
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .audio import AudioTrack, load_audio
from .paths import models_dir


SCHEMA_VERSION = "musicmix.vocal_boundaries.2"
CANONICAL_SAMPLE_RATE = 48_000
DEFAULT_PHRASE_GAP_SECONDS = 0.75
DEFAULT_FRAME_SECONDS = 0.040
DEFAULT_HOP_SECONDS = 0.020
DEMUCS_CHECKPOINTS = {
    "htdemucs": ("955717e8-8726e21a.th",),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _interval(start: Any, end: Any, duration_samples: int) -> tuple[int, int]:
    start_i = int(start)
    end_i = int(end)
    if start_i < 0 or end_i <= start_i or end_i > int(duration_samples):
        raise ValueError(f"invalid sample interval [{start_i}, {end_i})")
    return start_i, end_i


def _normalise_words(words: Iterable[Mapping[str, Any]], duration_samples: int) -> list[dict[str, Any]]:
    normalised: list[dict[str, Any]] = []
    for raw in words:
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        start, end = _interval(raw.get("start_sample"), raw.get("end_sample"), duration_samples)
        confidence = raw.get("confidence")
        confidence_value = None if confidence is None else float(np.clip(_finite_number(confidence, "confidence"), 0.0, 1.0))
        normalised.append({
            "start_sample": start,
            "end_sample": end,
            "text": text,
            "confidence": confidence_value,
        })
    normalised.sort(key=lambda item: (item["start_sample"], item["end_sample"], item["text"]))
    for previous, current in zip(normalised, normalised[1:]):
        if current["start_sample"] < previous["start_sample"]:
            raise ValueError("word intervals are not ordered")
    return normalised


def _normalise_segments(segments: Iterable[Mapping[str, Any]], duration_samples: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in segments:
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        start, end = _interval(raw.get("start_sample"), raw.get("end_sample"), duration_samples)
        confidence = raw.get("confidence")
        result.append({
            "start_sample": start,
            "end_sample": end,
            "text": text,
            "confidence": None if confidence is None else float(np.clip(_finite_number(confidence, "confidence"), 0.0, 1.0)),
        })
    result.sort(key=lambda item: (item["start_sample"], item["end_sample"]))
    return result


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((int(start), int(end)) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged: list[list[int]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _subtract_intervals(duration_samples: int, observed: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cursor = 0
    for start, end in _merge_intervals(observed):
        if start > cursor:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration_samples:
        result.append((cursor, duration_samples))
    return result


def _overlap(start: int, end: int, intervals: Iterable[Mapping[str, Any]]) -> bool:
    return any(int(item["start_sample"]) < end and int(item["end_sample"]) > start for item in intervals)


def derive_phrases(
    words: Sequence[Mapping[str, Any]],
    duration_samples: int,
    sample_rate: int = CANONICAL_SAMPLE_RATE,
    gap_seconds: float = DEFAULT_PHRASE_GAP_SECONDS,
) -> list[dict[str, Any]]:
    """Group timestamped ASR words by gaps; no section labels are inferred."""

    gap_samples = max(1, round(_finite_number(gap_seconds, "gap_seconds") * sample_rate))
    ordered = _normalise_words(words, duration_samples)
    if not ordered:
        return []
    phrases: list[dict[str, Any]] = []
    group = [ordered[0]]
    for word in ordered[1:]:
        if int(word["start_sample"]) - int(group[-1]["end_sample"]) > gap_samples:
            phrases.append(_phrase_from_group(group))
            group = [word]
        else:
            group.append(word)
    phrases.append(_phrase_from_group(group))
    return phrases


def _phrase_from_group(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    confidences = [word["confidence"] for word in group if word.get("confidence") is not None]
    # This is an aggregate ASR probability, not an alignment-calibrated score.
    confidence = min(confidences) if confidences else None
    return {
        "start_sample": int(group[0]["start_sample"]),
        "end_sample": int(group[-1]["end_sample"]),
        "confidence": confidence,
        "boundary_source": "asr_word_gap_heuristic",
        "word_count": len(group),
    }


def acoustic_intervals(
    track: AudioTrack,
    *,
    frame_seconds: float = DEFAULT_FRAME_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    threshold_multiplier: float = 3.0,
    relative_floor: float = 0.08,
    margin_ratio: float = 0.35,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Estimate occupied/silent intervals from a separated vocal waveform.

    This is an acoustic energy heuristic.  It never labels an ASR gap as
    silence.  Frames near the learned threshold are ``unknown_acoustic_margin``.
    """

    if track.duration_samples == 0:
        return [], {"status": "empty", "sample_rate": track.sample_rate}
    frame = max(1, round(_finite_number(frame_seconds, "frame_seconds") * track.sample_rate))
    hop = max(1, round(_finite_number(hop_seconds, "hop_seconds") * track.sample_rate))
    mono = np.mean(np.asarray(track.samples, dtype=np.float64), axis=1)
    # Measure with overlapping analysis frames, but emit non-overlapping hop
    # cells.  This makes coverage a true partition and avoids two conflicting
    # states being assigned to the same sample interval.
    frame_starts = np.arange(0, track.duration_samples, hop, dtype=np.int64)
    rms = np.asarray([np.sqrt(np.mean(np.square(mono[start : min(track.duration_samples, start + frame)]))) for start in frame_starts], dtype=np.float64)
    positive = rms[rms > 1e-8]
    if positive.size == 0:
        noise_floor = 0.0
        threshold = 1e-4
    else:
        noise_floor = float(np.percentile(positive, 20.0))
        p60 = float(np.percentile(positive, 60.0))
        p90 = float(np.percentile(positive, 90.0))
        raw_threshold = max(1e-5, noise_floor * float(threshold_multiplier), p60 * float(relative_floor))
        # Do not let a loud but continuous vocal stem turn its quietest vocal
        # frames into alleged silence.  The strict cap is deliberately bounded
        # both relative to the voiced peak and at -50 dBFS-ish; intervals near
        # the threshold remain unknown rather than being called silence.
        silence_cap = min(max(1e-5, p90 * 0.03), 0.003)
        threshold = min(raw_threshold, silence_cap)
        threshold_capped = threshold < raw_threshold
    margin = max(1e-6, threshold * float(margin_ratio))
    states: list[tuple[int, int, str]] = []
    for index, start in enumerate(frame_starts.tolist()):
        end = min(track.duration_samples, start + hop)
        value = float(rms[index])
        if value <= threshold - margin:
            state = "acoustic_silence"
        elif value >= threshold + margin:
            state = "vocal_occupancy"
        else:
            state = "unknown_acoustic_margin"
        states.append((start, end, state))
    # A frame lattice is intentionally kept conservative: adjacent states are
    # merged only when they touch, never interpolated across missing audio.
    merged: list[dict[str, Any]] = []
    for start, end, state in states:
        if merged and merged[-1]["end_sample"] == start and merged[-1]["status"] == state:
            merged[-1]["end_sample"] = end
        else:
            merged.append({
                "start_sample": start,
                "end_sample": end,
                "status": state,
                "evidence": "separated_vocals_rms_heuristic",
            })
    diagnostics = {
        "status": "measured",
        "sample_rate": track.sample_rate,
        "frame_samples": frame,
        "hop_samples": hop,
        "threshold_rms": threshold,
        "threshold_uncapped_rms": locals().get("raw_threshold", threshold),
        "threshold_capped_at_p90_fraction": locals().get("threshold_capped", False),
        "silence_cap_p90_fraction": 0.03,
        "silence_cap_absolute_rms": 0.003,
        "margin_rms": margin,
        "noise_floor_rms": noise_floor,
        "threshold_multiplier": float(threshold_multiplier),
        "relative_floor": float(relative_floor),
        "margin_ratio": float(margin_ratio),
        "note": "Acoustic silence is evidence from separated-stem energy, not ASR absence.",
    }
    return merged, diagnostics


def _coverage_intervals(
    duration_samples: int,
    segments: Sequence[Mapping[str, Any]],
    acoustic: Sequence[Mapping[str, Any]] | None = None,
    words: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Return a non-overlapping partition with explicit unknown gaps."""

    observed = _merge_intervals((int(item["start_sample"]), int(item["end_sample"])) for item in segments)
    acoustic = list(acoustic or [])
    cuts = {0, int(duration_samples)}
    for start, end in observed:
        cuts.update((start, end))
    for word in words:
        cuts.update((int(word['start_sample']), int(word['end_sample'])))
    for item in acoustic:
        cuts.update((int(item["start_sample"]), int(item["end_sample"])))
    ordered_cuts = sorted(cut for cut in cuts if 0 <= cut <= duration_samples)
    result: list[dict[str, Any]] = []
    for start, end in zip(ordered_cuts, ordered_cuts[1:]):
        if end <= start:
            continue
        segment_observed = any(s < end and e > start for s, e in observed)
        acoustic_states = [
            str(item["status"])
            for item in acoustic
            if int(item["start_sample"]) < end and int(item["end_sample"]) > start
        ]
        if any(state == "unknown_acoustic_margin" for state in acoustic_states):
            status = "unknown_acoustic_margin"
            evidence = "separated_vocals_rms_margin"
        # A transcript segment includes pauses between words. Only an actual
        # timestamped word contradicts measured low vocal energy here.
        elif _overlap(start, end, words) and any(state == "acoustic_silence" for state in acoustic_states):
            status = "unknown_acoustic_conflict"
            evidence = "asr_and_stem_energy_conflict"
        elif acoustic_states and all(state == "acoustic_silence" for state in acoustic_states):
            status = "acoustic_silence_supported"
            evidence = "separated_vocals_rms_heuristic"
        elif acoustic_states and any(state == "vocal_occupancy" for state in acoustic_states):
            status = "vocal_occupancy_supported"
            evidence = "separated_vocals_rms_heuristic"
        elif segment_observed:
            status = "asr_observed"
            evidence = "asr_segment"
        else:
            status = "unknown_no_asr"
            evidence = "missing_asr_is_not_silence"
        result.append({"start_sample": start, "end_sample": end, "status": status, "evidence": evidence})
    return result


def _uncertain_regions(
    duration_samples: int,
    coverage: Sequence[Mapping[str, Any]],
    words: Sequence[Mapping[str, Any]],
    low_confidence_threshold: float,
) -> list[dict[str, Any]]:
    uncertain: list[dict[str, Any]] = []
    for item in coverage:
        status = str(item["status"])
        if status.startswith("unknown") or status == "unknown_acoustic_margin":
            uncertain.append({
                "start_sample": int(item["start_sample"]),
                "end_sample": int(item["end_sample"]),
                "reason": "missing_asr_not_silence" if status == "unknown_no_asr" else status,
            })
    for word in words:
        confidence = word.get("confidence")
        if confidence is not None and float(confidence) < low_confidence_threshold:
            uncertain.append({
                "start_sample": int(word["start_sample"]),
                "end_sample": int(word["end_sample"]),
                "reason": "low_asr_probability",
            })
    merged: list[dict[str, Any]] = []
    for item in sorted(uncertain, key=lambda value: (value["start_sample"], value["end_sample"])):
        if merged and item["start_sample"] <= merged[-1]["end_sample"]:
            merged[-1]["end_sample"] = max(merged[-1]["end_sample"], item["end_sample"])
            if item["reason"] not in merged[-1]["reason"]:
                merged[-1]["reason"] += ";" + item["reason"]
        else:
            merged.append(dict(item))
    return merged


def build_boundary_document(
    audio_path: Path,
    *,
    words: Iterable[Mapping[str, Any]],
    asr_segments: Iterable[Mapping[str, Any]],
    source_kind: str,
    backend: Mapping[str, Any],
    separator: Mapping[str, Any],
    warnings: Iterable[str] = (),
    acoustic: Sequence[Mapping[str, Any]] | None = None,
    acoustic_diagnostics: Mapping[str, Any] | None = None,
    transcript: str = "",
    low_confidence_threshold: float = 0.45,
) -> dict[str, Any]:
    """Build and validate the persisted, local-only vocal boundary schema."""

    track = load_audio(audio_path)
    words_norm = _normalise_words(words, track.duration_samples)
    segments_norm = _normalise_segments(asr_segments, track.duration_samples)
    acoustic_norm = list(acoustic or [])
    coverage = _coverage_intervals(track.duration_samples, segments_norm, acoustic_norm, words_norm)
    phrases = derive_phrases(words_norm, track.duration_samples, track.sample_rate)
    uncertain = _uncertain_regions(track.duration_samples, coverage, words_norm, low_confidence_threshold)
    if not transcript:
        transcript = " ".join(segment["text"] for segment in segments_norm).strip()
    return {
        "schema_version": SCHEMA_VERSION,
        "track_id": track.track_id,
        "canonical_file_sha256": sha256_file(audio_path),
        "canonical_audio_uri": str(audio_path.resolve()),
        "sample_rate": track.sample_rate,
        "duration_samples": track.duration_samples,
        "duration_seconds": track.duration_samples / track.sample_rate,
        "source_kind": source_kind,
        "words": words_norm,
        "phrases": phrases,
        "asr_segments": segments_norm,
        "transcript": transcript,
        "coverage_intervals": coverage,
        "uncertain_regions": uncertain,
        "asr_low_confidence_threshold": float(low_confidence_threshold),
        "backend": dict(backend),
        "separator": dict(separator),
        "acoustic_occupancy": list(acoustic_norm),
        "acoustic_diagnostics": dict(acoustic_diagnostics or {}),
        "warnings": list(dict.fromkeys(str(item) for item in warnings)),
        "semantics": {
            "asr_confidence": "segment/word ASR probability; not alignment-calibrated",
            "missing_asr": "unknown, never silence",
            "phrase_source": "ASR word gaps only; section labels unavailable",
            "timeline": "canonical sample intervals [start_sample,end_sample) with no unverified offset correction",
        },
    }


def _coverage_status_for_interval(
    start: int,
    end: int,
    coverage_intervals: Sequence[Mapping[str, Any]],
) -> tuple[bool, bool]:
    """Return (fully_supported, intersects_unknown)."""

    if start < 0 or end <= start:
        return False, True
    cursor = start
    unknown = False
    for item in sorted(coverage_intervals, key=lambda value: int(value["start_sample"])):
        item_start = int(item["start_sample"])
        item_end = int(item["end_sample"])
        if item_end <= cursor or item_start >= end:
            continue
        if item_start > cursor:
            unknown = True
        overlap_start = max(cursor, item_start)
        overlap_end = min(end, item_end)
        if str(item.get("status", "")).startswith("unknown"):
            unknown = True
        cursor = max(cursor, overlap_end)
    if cursor < end:
        unknown = True
    return (not unknown and cursor >= end), unknown


def boundary_safety(
    document: Mapping[str, Any],
    *,
    side: str,
    cue_sample: int,
    fade_samples: int,
    word_span: Mapping[str, Any] | None = None,
    alignment_guard_samples: int = 0,
) -> dict[str, Any]:
    """Audit whether a fade boundary is supported, unsafe, or unknown.

    ``side='A_out'`` requires the outgoing word to end before fade-out starts.
    ``side='B_in'`` requires the incoming word to start after fade-in ends.
    If no word is supplied, only an explicitly measured separated-stem acoustic
    silence can support the boundary; an ASR gap returns ``unknown``.
    """

    side = str(side)
    if side not in {"A_out", "B_in", "out", "in"}:
        raise ValueError("side must be A_out/B_in (or out/in)")
    side = "A_out" if side in {"A_out", "out"} else "B_in"
    cue = int(cue_sample)
    fade = int(fade_samples)
    guard = int(alignment_guard_samples)
    if cue < 0 or fade <= 0 or guard < 0:
        raise ValueError("cue/fade/guard samples are invalid")
    fade_start, fade_end = (cue - fade, cue) if side == "A_out" else (cue, cue + fade)
    coverage = list(document.get("coverage_intervals", []))
    uncertain = list(document.get("uncertain_regions", []))
    duration = int(document["duration_samples"])
    result: dict[str, Any] = {
        "status": "unknown",
        "side": side,
        "cue_sample": cue,
        "fade_interval": {"start_sample": fade_start, "end_sample": fade_end},
        "alignment_guard_samples": guard,
        "reasons": [],
    }
    if fade_start < 0:
        result["status"] = "unsafe"
        result["reasons"].append("fade_before_track_start")
        return result
    if fade_end > duration:
        result["status"] = "unsafe"
        result["reasons"].append("fade_beyond_track_end")
        return result
    fade_supported, fade_unknown = _coverage_status_for_interval(fade_start, fade_end, coverage)
    if fade_unknown or not fade_supported:
        result["reasons"].append("alignment_or_coverage_uncertain")
    known_words = list(document.get("words", []))
    if _overlap(fade_start, fade_end, known_words):
        result["status"] = "unsafe"
        result["reasons"].append("fade_overlaps_known_word")
    if word_span is not None:
        word_start, word_end = _interval(word_span.get("start_sample"), word_span.get("end_sample"), duration)
        result["word_interval"] = {"start_sample": word_start, "end_sample": word_end}
        word_supported, word_unknown = _coverage_status_for_interval(word_start, word_end, coverage)
        if word_unknown or not word_supported:
            result["reasons"].append("word_interval_coverage_unknown")
        confidence = word_span.get("confidence")
        if confidence is None:
            result["reasons"].append("word_confidence_missing")
        elif float(confidence) < float(document.get("asr_low_confidence_threshold", 0.45)):
            result["reasons"].append("word_confidence_low")
        if _overlap(word_start, word_end, uncertain):
            result["reasons"].append("word_interval_uncertain")
        if side == "A_out":
            if word_end > fade_start - guard:
                result["status"] = "unsafe"
                result["reasons"].append("outgoing_word_overlaps_fade_or_cut")
            elif not result["reasons"]:
                result["status"] = "supported"
        else:
            if word_start < cue:
                result["status"] = "unsafe"
                result["reasons"].append("incoming_word_already_started_before_cue")
            elif word_start < fade_end + guard:
                result["status"] = "unsafe"
                result["reasons"].append("incoming_word_overlaps_fade_in")
            elif not result["reasons"]:
                result["status"] = "supported"
    else:
        overlap_items = [
            item for item in coverage
            if int(item["start_sample"]) < fade_end and int(item["end_sample"]) > fade_start
        ]
        acoustic_supported = (
            fade_supported
            and bool(overlap_items)
            and all(
            str(item.get("status")) == "acoustic_silence_supported"
            for item in overlap_items
            )
            and not _overlap(fade_start, fade_end, uncertain)
        )
        if acoustic_supported and not result["status"] == "unsafe":
            result["status"] = "supported"
            result["reasons"].append("measured_separated_vocal_low_energy")
        else:
            result["reasons"].append("word_span_missing_or_asr_gap")
    if result["status"] == "unknown" and not result["reasons"]:
        result["reasons"].append("insufficient_boundary_evidence")
    return result


def query_fade_safety(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Stable alias used by retrieval/render integration."""

    return boundary_safety(*args, **kwargs)


def _runtime_paths(runtime_root: Path | None) -> None:
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        runtime_root,
        project_root / ".cache" / "structure_runtime" / "Lib" / "site-packages",
        project_root / ".cache" / "asr_runtime",
    ]
    for candidate in reversed(candidates):
        if candidate is None:
            continue
        candidate = candidate.resolve()
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def transcribe_audio(
    audio_path: Path,
    *,
    model_path: Path,
    device: str = "cpu",
    compute_type: str = "int8",
    cpu_threads: int = 4,
    language: str = "en",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, dict[str, Any]]:
    """Run cached faster-whisper and return words, segments, transcript, metadata."""

    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("CT2_NUM_THREADS", str(cpu_threads))
    from faster_whisper import WhisperModel  # type: ignore

    model = WhisperModel(str(model_path), device=device, compute_type=compute_type,
                         cpu_threads=cpu_threads, num_workers=1)
    segments_iter, info = model.transcribe(
        str(audio_path), language=language, word_timestamps=True,
        vad_filter=False, beam_size=5, temperature=0.0,
        condition_on_previous_text=False,
    )
    words: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    transcript_parts: list[str] = []
    for segment in segments_iter:
        segment_start = float(segment.start)
        segment_end = float(segment.end)
        segment_text = str(segment.text).strip()
        if segment_end <= segment_start or not segment_text:
            continue
        transcript_parts.append(segment_text)
        segment_words: list[dict[str, Any]] = []
        for word in (segment.words or []):
            if word.start is None or word.end is None or word.end <= word.start:
                continue
            text = str(word.word).strip()
            if not text:
                continue
            item = {
                "start_seconds": float(word.start),
                "end_seconds": float(word.end),
                "text": text,
                "confidence": None if word.probability is None else float(np.clip(word.probability, 0.0, 1.0)),
            }
            segment_words.append(item)
            words.append(item)
        confidences = [item["confidence"] for item in segment_words if item["confidence"] is not None]
        segments.append({
            "start_seconds": segment_start,
            "end_seconds": segment_end,
            "text": segment_text,
            "confidence": min(confidences) if confidences else None,
        })
    metadata = {
        "backend": "faster-whisper",
        "model_path": str(model_path.resolve()),
        "model_sha256": _model_hash(model_path),
        "device": device,
        "compute_type": compute_type,
        "cpu_threads": cpu_threads,
        "language": language,
        "detected_language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
    }
    return words, segments, " ".join(transcript_parts), metadata


MULTILINGUAL_WHISPER_REPO = "Systran/faster-whisper-small"
# [H] Soglia sulla probabilita' media della lingua (media sulle finestre di parlato): sotto questa
# soglia, o se la lingua rilevata e' l'inglese, si resta su ``small.en`` (comportamento storico).
LANGUAGE_MIN_PROBABILITY = 0.5


def find_whisper_snapshot(repo: str = MULTILINGUAL_WHISPER_REPO, roots: Sequence[Path] | None = None) -> Path | None:
    """Ultimo snapshot HF in cache di ``repo`` (con ``model.bin``), cercato nelle cache del progetto e della macchina di analisi."""

    project_root = Path(__file__).resolve().parents[2]
    if roots is None:
        roots = [project_root / ".cache" / "models"]
        roots.append(models_dir() / ".cache" / "models")
    name = "models--" + repo.replace("/", "--")
    for root in roots:
        snapshots = Path(root) / name / "snapshots"
        if not snapshots.is_dir():
            continue
        found = sorted(p for p in snapshots.iterdir() if p.is_dir() and (p / "model.bin").is_file())
        if found:
            return found[-1]
    return None


def detect_language(
    audio_path: Path,
    *,
    model_path: Path,
    cpu_threads: int = 2,
    max_windows: int = 8,
    model: Any = None,
) -> dict[str, Any]:
    """Lingua del parlato con il rilevatore ufficiale di Whisper, finestra per finestra.

    Si tiene solo il parlato (Silero VAD di faster-whisper, parametri di default), lo si divide in
    finestre di 30 s e si scelgono fino a ``max_windows`` finestre distribuite su tutto il brano (non
    solo l'inizio, spesso strumentale). Per ogni finestra ``WhisperModel.detect_language`` restituisce
    le probabilita' delle lingue; la lingua scelta e' quella con probabilita' media piu' alta. Si
    registrano anche la lingua prevalente di ogni finestra (un brano bilingue lo mostra) e le prime 5.
    Serve un modello multilingue (``small``, non ``small.en``).
    """

    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("CT2_NUM_THREADS", str(cpu_threads))
    from faster_whisper import WhisperModel, decode_audio  # type: ignore
    from faster_whisper.vad import collect_chunks, get_speech_timestamps  # type: ignore

    if model is None:
        model = WhisperModel(str(model_path), device="cpu", compute_type="int8",
                             cpu_threads=cpu_threads, num_workers=1)
    audio = decode_audio(str(audio_path), sampling_rate=16000)
    chunks = get_speech_timestamps(audio)
    speech_seconds = float(sum(c["end"] - c["start"] for c in chunks)) / 16000.0
    base = {
        "method": "faster-whisper WhisperModel.detect_language per finestra di 30 s di parlato (Silero VAD), "
                  "media delle probabilita' sulle finestre",
        "model_path": str(Path(model_path).resolve()),
        "model_sha256": _model_hash(Path(model_path)),
        "audio_path": str(Path(audio_path).resolve()),
        "speech_seconds": speech_seconds,
    }
    if not chunks:
        return {**base, "language": None, "probability": None, "n_windows": 0, "windows": [], "window_language_counts": {},
                "top5": []}
    pieces, _ = collect_chunks(audio, chunks)
    speech = np.concatenate(pieces, axis=0)
    n_samples = int(model.feature_extractor.n_samples)
    n_full = max(1, len(speech) // n_samples)  # solo finestre piene, salvo parlato < 30 s
    picks = sorted(set(np.linspace(0, n_full - 1, num=min(max_windows, n_full)).round().astype(int).tolist()))
    sums: dict[str, float] = {}
    windows = []
    for k in picks:
        window = speech[k * n_samples:(k + 1) * n_samples]
        lang, prob, all_probs = model.detect_language(audio=window, vad_filter=False, language_detection_segments=1,
                                                      language_detection_threshold=1.01)
        probs = {str(name): float(p) for name, p in all_probs}
        for name, p in probs.items():
            sums[name] = sums.get(name, 0.0) + p
        windows.append({"window": int(k), "speech_start_seconds": k * n_samples / 16000.0,
                        "language": max(probs, key=probs.get), "probability": max(probs.values())})
    mean = {name: s / len(picks) for name, s in sums.items()}
    top = sorted(mean.items(), key=lambda kv: -kv[1])[:5]
    shares: dict[str, int] = {}
    for w in windows:
        shares[w["language"]] = shares.get(w["language"], 0) + 1
    return {**base, "language": top[0][0], "probability": float(top[0][1]), "n_windows": len(picks),
            "windows": windows, "window_language_counts": shares, "top5": [[k, float(v)] for k, v in top]}


def choose_asr_language(
    requested: str | None,
    detection: Mapping[str, Any] | None,
    *,
    min_probability: float = LANGUAGE_MIN_PROBABILITY,
    multilingual_available: bool = True,
) -> dict[str, Any]:
    """Decide la lingua passata a Whisper: richiesta esplicita > rilevamento sicuro > inglese (fallback).

    ``requested`` = ``None``/``"auto"`` attiva il rilevamento. L'inglese usa ``small.en`` (storico);
    ogni altra lingua usa il modello multilingue. Il motivo della scelta e' registrato.
    """

    req = (requested or "auto").strip().lower()
    if req != "auto":
        if req != "en" and not multilingual_available:
            return {"language": "en", "source": "fallback_en", "multilingual": False,
                    "reason": f"lingua richiesta {req!r} ma modello multilingue assente"}
        return {"language": req, "source": "requested", "multilingual": req != "en", "reason": "lingua richiesta"}
    if not detection or not detection.get("language"):
        why = (detection or {}).get("error") or "rilevamento non disponibile o nessun parlato"
        return {"language": "en", "source": "fallback_en", "multilingual": False, "reason": str(why)}
    lang, prob = str(detection["language"]), float(detection.get("probability") or 0.0)
    if lang == "en":
        return {"language": "en", "source": "detected", "multilingual": False, "reason": f"rilevato en p={prob:.2f}"}
    if prob < min_probability:
        return {"language": "en", "source": "fallback_en", "multilingual": False,
                "reason": f"rilevato {lang} con p={prob:.2f} < {min_probability}"}
    if not multilingual_available:
        return {"language": "en", "source": "fallback_en", "multilingual": False,
                "reason": f"rilevato {lang} ma modello multilingue assente"}
    return {"language": lang, "source": "detected", "multilingual": True, "reason": f"rilevato {lang} p={prob:.2f}"}


def _model_hash(model_path: Path) -> str | None:
    if model_path.is_file():
        return sha256_file(model_path)
    model_bin = model_path / "model.bin"
    return sha256_file(model_bin) if model_bin.is_file() else None


def _write_pcm16(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError("expected (channels, samples) or (samples, channels)")
    if array.shape[0] <= 4 and array.shape[1] > array.shape[0]:
        channels_first = array
    else:
        channels_first = array.T
    channels_first = np.clip(channels_first, -1.0, 1.0)
    pcm = np.round(channels_first.T * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(int(channels_first.shape[0]))
        stream.setsampwidth(2)
        stream.setframerate(int(sample_rate))
        stream.writeframes(pcm.tobytes())


def separate_vocals_demucs(
    audio_path: Path,
    output_path: Path,
    *,
    runtime_root: Path,
    model_name: str = "htdemucs",
    threads: int = 4,
) -> dict[str, Any]:
    """Separate only vocals with a local Demucs runtime, preserving length."""

    _runtime_paths(runtime_root)
    import torch  # type: ignore
    from demucs.apply import apply_model  # type: ignore
    from demucs.audio import AudioFile  # type: ignore
    from demucs.pretrained import get_model  # type: ignore
    import torchaudio  # type: ignore

    torch.set_num_threads(int(threads))
    model = get_model(model_name)
    model.to("cpu")
    model.eval()
    original = load_audio(audio_path)
    waveform = AudioFile(audio_path).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
    if not torch.isfinite(waveform).all():
        raise ValueError("Demucs input contains non-finite samples")
    reference = waveform.mean(0)
    reference_mean = reference.mean()
    reference_std = reference.std()
    if not torch.isfinite(reference_std) or float(reference_std) < 1e-8:
        raise ValueError("Demucs input is silent or has unusable reference standard deviation")
    # Match the official demucs.separate.py inference path exactly.
    waveform = waveform - reference_mean
    waveform = waveform / reference_std
    with torch.no_grad():
        separated = apply_model(model, waveform[None], device="cpu", shifts=0,
                                split=True, overlap=0.25, progress=False)
    separated = separated * reference_std
    separated = separated + reference_mean
    if not torch.isfinite(separated).all():
        raise ValueError("Demucs output contains non-finite samples")
    try:  # optional by-product: a failure here must never degrade the vocals stage
        shared_stems = _save_shared_stems(separated[0].cpu(), model, audio_path, model_name)
    except Exception as exc:  # noqa: BLE001
        shared_stems = {"status": "failed", "error": repr(exc)[-300:]}
    source_index = list(model.sources).index("vocals")
    vocals = separated[0, source_index].cpu()
    if int(model.samplerate) != CANONICAL_SAMPLE_RATE:
        vocals = torchaudio.functional.resample(vocals, int(model.samplerate), CANONICAL_SAMPLE_RATE)
    target_length = original.duration_samples
    if vocals.shape[-1] < target_length:
        vocals = torch.nn.functional.pad(vocals, (0, target_length - vocals.shape[-1]))
    vocals = vocals[..., :target_length]
    _write_pcm16(output_path, vocals.numpy(), CANONICAL_SAMPLE_RATE)
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints"
    checkpoint_entries = []
    for filename in DEMUCS_CHECKPOINTS.get(model_name, ()):
        checkpoint_path = checkpoint / filename
        checkpoint_entries.append({
            "filename": filename,
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path) if checkpoint_path.is_file() else None,
            "url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/" + filename,
        })
    return {
        "status": "complete",
        "method": "demucs",
        "model": model_name,
        "model_samplerate": int(model.samplerate),
        "output_sample_rate": CANONICAL_SAMPLE_RATE,
        "threads": int(threads),
        "shifts": 0,
        "checkpoint_files": checkpoint_entries,
        "normalization": "official_demucs_reference_mean_std; rejected nonfinite or std<1e-8",
        "reference_mean": float(reference_mean),
        "reference_std": float(reference_std),
        "input_sha256": sha256_file(audio_path),
        "alignment": "AudioFile model resample, then canonical 48 kHz resample; trim/pad to source duration; no intentional offset",
        **({"shared_stems": shared_stems} if shared_stems else {}),
    }


def _save_shared_stems(sources: Any, model: Any, audio_path: Path, model_name: str) -> dict[str, Any] | None:
    """Optionally save all Demucs stems for reuse by All-In-One (``allin1.demix``).

    Enabled only by ``MUSICMIX_DEMUCS_STEMS_DIR``. Writes ``<dir>/htdemucs/<audio stem>/{bass,drums,other,vocals}.wav``
    exactly as ``demucs.separate`` saves them (model sample rate, ``save_audio`` defaults: int16, clip=rescale), so
    allin1 finds them and skips its own separation. The only difference from allin1's internal call is
    ``shifts=0`` (deterministic) instead of the CLI default ``shifts=1`` (one random time shift). The vocal stem
    used by this stage is computed from the same tensor, so the vocals output is unchanged.
    """
    root = os.environ.get("MUSICMIX_DEMUCS_STEMS_DIR")
    if not root or model_name != "htdemucs":
        return None
    import soundfile as sf  # type: ignore
    from demucs.audio import prevent_clip  # type: ignore

    out = Path(root) / "htdemucs" / Path(audio_path).stem
    tmp = out.with_name(out.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=True)
    for source, name in zip(sources, model.sources):
        # demucs.separate -> save_audio(clip="rescale", int16 PCM); torchaudio.save needs torchcodec here,
        # so the same clip step is followed by a soundfile PCM_16 write (int16 rounding may differ by 1 LSB).
        wav = prevent_clip(source, mode="rescale")
        sf.write(str(tmp / f"{name}.wav"), wav.numpy().T, int(model.samplerate), subtype="PCM_16")
    if out.exists():
        import shutil
        shutil.rmtree(out)
    tmp.rename(out)
    record = {"dir": str(out), "model": model_name, "shifts": 0, "split": True, "overlap": 0.25,
              "samplerate": int(model.samplerate), "format": "wav PCM_16 via soundfile after demucs prevent_clip(rescale) (as demucs.separate defaults)",
              "input_sha256": sha256_file(audio_path), "sources": list(model.sources)}
    prov = Path(root) / "htdemucs_provenance"
    prov.mkdir(parents=True, exist_ok=True)
    (prov / f"{Path(audio_path).stem}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def seconds_to_samples(items: Iterable[Mapping[str, Any]], sample_rate: int, duration_samples: int) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for item in items:
        start_seconds = _finite_number(item["start_seconds"], "start_seconds")
        end_seconds = _finite_number(item["end_seconds"], "end_seconds")
        start = round(start_seconds * sample_rate)
        end = round(end_seconds * sample_rate)
        if end <= start:
            continue
        end = min(duration_samples, end)
        start = max(0, start)
        if end <= start:
            continue
        converted.append({
            "start_sample": start,
            "end_sample": end,
            "text": str(item.get("text", "")).strip(),
            "confidence": item.get("confidence"),
        })
    return converted


__all__ = [
    "SCHEMA_VERSION", "acoustic_intervals", "boundary_safety", "build_boundary_document",
    "choose_asr_language", "derive_phrases", "detect_language", "find_whisper_snapshot",
    "query_fade_safety", "seconds_to_samples", "separate_vocals_demucs", "sha256_file", "transcribe_audio",
]
