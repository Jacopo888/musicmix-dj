"""Independent audit for the psychology-v11 render manifest.

The driver is deliberately kept out of this module.  It reads the manifest
once, decodes the referenced WAV files, and checks facts which are independent
of the candidate scorer: file identity, canonical clocks, quantised PCM at
several interior windows, gain bounds, clipping, history reuse, and the
transition-policy safety contract.

The manifest reader accepts the small spelling differences used by the
experiment drivers (``artifact.path`` versus ``path``, ``tracks`` versus
``catalogtracks`` and so on).  Missing provenance is reported as a failed
check rather than silently being treated as success.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import wave
from typing import Any, Iterable, Mapping

import numpy as np

try:
    from musicmix.paths import localize
except ImportError:  # direct CLI run: use the adjacent package
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from musicmix.paths import localize


CANONICAL_RATE = 48_000
EXPECTED_SECONDS = 48
JOIN_SECONDS = 24
EXPECTED_FRAMES = CANONICAL_RATE * EXPECTED_SECONDS
JOIN_FRAMES = CANONICAL_RATE * JOIN_SECONDS
KNOWN_RENDER_GAIN = 0.9
RAMP_SECONDS = 0.012
MAX_ATTENUATION_DB = 6.0
MAX_BOOST_DB = 3.0
_PATH_KEYS = {
    "path", "wav", "audio_path", "artifact_path", "output_path",
    "canonical_path", "canonical_audio_uri", "canonical_audio_path",
    "source_path", "source_audio_path", "file", "source", "audio",
}
_SHA_KEYS = {"sha256", "sha", "checksum", "output_sha256", "digest"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _first_value(obj: Any, keys: Iterable[str]) -> Any:
    """Return the first value for one of *keys*, walking nested mappings."""

    wanted = {str(key).lower() for key in keys}
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if str(key).lower() in wanted:
                return value
        for value in obj.values():
            found = _first_value(value, wanted)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _first_value(value, wanted)
            if found is not None:
                return found
    return None


def _path_value(obj: Any) -> str | None:
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, Mapping):
        return None
    # Catalog entries commonly carry both the original compressed URI and the
    # decoded canonical PCM.  The independent clock comparison must use the
    # latter, whose sample zero is the audited timeline.
    for preferred in ("canonical_audio_uri", "canonical_path", "canonical_audio_path"):
        value = obj.get(preferred)
        if isinstance(value, (str, Path)):
            return str(value)
    for key, value in obj.items():
        if str(key).lower() in _PATH_KEYS and isinstance(value, (str, Path)):
            return str(value)
    return None


def _resolve_path(value: Any, base: Path) -> Path | None:
    path = _path_value(value)
    if path is None:
        return None
    candidate = localize(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate


class WavData:
    __slots__ = ("path", "rate", "channels", "width", "frames", "pcm", "samples")

    def __init__(self, path: Path, rate: int, channels: int, width: int,
                 frames: int, pcm: np.ndarray, samples: np.ndarray) -> None:
        self.path = path
        self.rate = int(rate)
        self.channels = int(channels)
        self.width = int(width)
        self.frames = int(frames)
        self.pcm = pcm
        self.samples = samples


def _decode_pcm(raw: bytes, width: int, channels: int) -> np.ndarray:
    if width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 4:
        values = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    elif width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values_i32 = (packed[:, 0].astype(np.int32)
                      | (packed[:, 1].astype(np.int32) << 8)
                      | (packed[:, 2].astype(np.int32) << 16))
        negative = (values_i32 & 0x800000) != 0
        values_i32[negative] -= 1 << 24
        values = values_i32.astype(np.float64) / 8388608.0
    else:
        raise ValueError(f"unsupported PCM width {width * 8} bits")
    values = values.reshape(-1, channels)
    if channels == 1:
        values = np.repeat(values, 2, axis=1)
    return np.ascontiguousarray(values, dtype=np.float64)


def _read_wav(path: Path, *, require_output_pcm: bool = False) -> WavData:
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        rate = stream.getframerate()
        width = stream.getsampwidth()
        frames = stream.getnframes()
        if stream.getcomptype() != "NONE":
            raise ValueError(f"compressed WAV is unsupported: {path}")
        raw = stream.readframes(frames)
    expected = int(frames) * int(channels) * int(width)
    if len(raw) != expected:
        raise ValueError(f"truncated WAV payload for {path}")
    if channels not in (1, 2):
        raise ValueError(f"unsupported channel count {channels} for {path}")
    samples = _decode_pcm(raw, width, channels)
    if require_output_pcm:
        if rate != CANONICAL_RATE or channels != 2 or width != 2:
            raise ValueError(
                f"output must be 48 kHz stereo PCM16, got {rate} Hz/{channels}ch/{width * 8}bit"
            )
        pcm = np.frombuffer(raw, dtype="<i2").reshape(-1, 2).copy()
    else:
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).round().astype("<i2")
    return WavData(path, rate, channels, width, frames, pcm, samples)


def _track_entries(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("tracks", "catalogtracks", "catalog_tracks", "track_catalog"):
        value = manifest.get(key)
        if isinstance(value, list):
            return [entry if isinstance(entry, Mapping) else {"canonical_path": entry}
                    for entry in value if isinstance(entry, (Mapping, str, Path))]
        if isinstance(value, Mapping):
            entries = []
            for track_id, entry in value.items():
                if isinstance(entry, Mapping):
                    entries.append({"track_id": track_id, **entry})
                elif isinstance(entry, (str, Path)):
                    entries.append({"track_id": track_id, "canonical_path": entry})
            return entries
    return []


def _track_map(manifest: Mapping[str, Any], base: Path) -> dict[str, tuple[Mapping[str, Any], Path | None]]:
    result: dict[str, tuple[Mapping[str, Any], Path | None]] = {}
    for entry in _track_entries(manifest):
        track_id = _first_value(entry, ("track_id", "id", "key", "name"))
        path = _resolve_path(entry, base)
        if track_id is not None:
            result[str(track_id)] = (entry, path)
        if path is not None:
            result.setdefault(str(path), (entry, path))
    return result


def _candidate_for(entry: Mapping[str, Any], manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate = entry.get("candidate")
    if isinstance(candidate, Mapping):
        return candidate
    candidate_id = entry.get("candidate_id")
    candidates = manifest.get("candidates")
    if isinstance(candidates, Mapping) and candidate_id in candidates:
        value = candidates[candidate_id]
        if isinstance(value, Mapping):
            return value.get("candidate", value)
    if isinstance(candidates, list):
        for value in candidates:
            if isinstance(value, Mapping) and value.get("candidate_id") == candidate_id:
                return value.get("candidate", value)
    return {}


def _cue(candidate: Mapping[str, Any], entry: Mapping[str, Any], side: str) -> int | None:
    aliases = {
        "a": ("cue_out_sample", "outgoing_cue_sample", "a_cue_sample", "cue_a_sample"),
        "b": ("cue_in_sample", "incoming_cue_sample", "b_cue_sample", "cue_b_sample"),
    }
    value = _first_value(candidate, aliases[side])
    if value is None:
        value = _first_value(entry, aliases[side])
    if value is None:
        clocks = entry.get("clocks")
        if isinstance(clocks, Mapping):
            source = clocks.get("source", clocks)
            if isinstance(source, Mapping):
                clock = source.get(side)
                value = _first_value(clock, ("cue_sample", "cue", "anchor_sample"))
    number = _as_number(value)
    return None if number is None else int(round(number))


def _track_for_side(candidate: Mapping[str, Any], entry: Mapping[str, Any], side: str,
                    tracks: Mapping[str, tuple[Mapping[str, Any], Path | None]],
                    base: Path) -> tuple[str | None, Path | None]:
    names = {
        "a": ("track_a", "track_a_id", "outgoing_track_id", "source_a_id"),
        "b": ("track_b", "track_b_id", "incoming_track_id", "source_b_id"),
    }
    value = _first_value(candidate, names[side])
    if value is None:
        value = _first_value(entry, names[side])
    if isinstance(value, Mapping):
        direct = _resolve_path(value, base)
        identifier = _first_value(value, ("track_id", "id", "name"))
        if direct is not None:
            return str(identifier) if identifier is not None else None, direct
        value = identifier
    if value is not None and str(value) in tracks:
        return str(value), tracks[str(value)][1]
    direct = _resolve_path(entry.get(f"source_{side}"), base)
    if direct is None:
        direct = _resolve_path(entry.get(f"track_{side}_path"), base)
    return str(value) if value is not None else None, direct


def _artifact_path(entry: Mapping[str, Any], base: Path) -> Path | None:
    artifact = entry.get("artifact")
    path = _resolve_path(artifact, base)
    if path is not None:
        return path
    for key in ("path", "wav", "audio_path", "artifact_path", "output_path"):
        path = _resolve_path(entry.get(key), base)
        if path is not None:
            return path
    return None


def _declared_sha(entry: Mapping[str, Any]) -> str | None:
    artifact = entry.get("artifact")
    for obj in (artifact, entry):
        value = _first_value(obj, _SHA_KEYS)
        if isinstance(value, str) and len(value) >= 32:
            return value.lower()
    return None


def _gain_info(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Read level-match gains while keeping the known 0.9 render gain separate."""

    objects = [entry.get("artifact"), entry.get("validation"), entry]
    gains: Mapping[str, Any] | None = None
    master: float | None = None
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        value = obj.get("level_matching", obj.get("level_match"))
        if isinstance(value, Mapping):
            maybe = value.get("gains_db")
            if isinstance(maybe, Mapping):
                gains = maybe
                master = _as_number(maybe.get("master_trim_db"))
            if master is None:
                master = _as_number(value.get("master_trim_db"))
        maybe = obj.get("gains_db")
        if isinstance(maybe, Mapping):
            gains = maybe
            if master is None:
                master = _as_number(maybe.get("master_trim_db"))
    if gains is None:
        gains = {}
    result: dict[str, Any] = {
        "a_db": _as_number(gains.get("a")) if isinstance(gains, Mapping) else None,
        "b_db": _as_number(gains.get("b")) if isinstance(gains, Mapping) else None,
        "master_trim_db": master if master is not None else 0.0,
        "base_gain": KNOWN_RENDER_GAIN,
    }
    for side in ("a", "b"):
        if result[f"{side}_db"] is None:
            for key in (f"gain_{side}_db", f"{side}_gain_db"):
                value = _first_value(entry, (key,))
                if value is not None:
                    result[f"{side}_db"] = _as_number(value)
                    break
        if result[f"{side}_db"] is None:
            result[f"{side}_db"] = 0.0
    result["master_trim_db"] = float(result["master_trim_db"])
    result["allowed_base_db"] = {
        side: float(result[f"{side}_db"]) - result["master_trim_db"] for side in ("a", "b")
    }
    result["gain_factors"] = {
        side: KNOWN_RENDER_GAIN * 10.0 ** (float(result[f"{side}_db"]) / 20.0)
        for side in ("a", "b")
    }
    return result


def _metadata_checks(data: WavData, declared_sha: str | None, path: Path) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    digest = sha256_file(path)
    if declared_sha is None:
        issues.append("missing_declared_sha256")
    elif digest != declared_sha:
        issues.append("sha256_mismatch")
    if data.rate != CANONICAL_RATE:
        issues.append("sample_rate_not_48000")
    if data.channels != 2 or data.width != 2:
        issues.append("output_not_stereo_pcm16")
    if data.frames != EXPECTED_FRAMES:
        issues.append("duration_not_48_seconds")
    boundary = int(np.count_nonzero((data.pcm == -32768) | (data.pcm == 32767)))
    actual_peak = float(np.max(np.abs(data.pcm.astype(np.float64))) / 32767.0) if data.pcm.size else 0.0
    if boundary:
        issues.append("actual_pcm_clipping")
    return issues, {
        "path": str(path), "sha256": digest, "declared_sha256": declared_sha,
        "sample_rate": data.rate, "channels": data.channels, "sample_width": data.width,
        "duration_samples": data.frames, "duration_seconds": data.frames / CANONICAL_RATE,
        "join_sample": JOIN_FRAMES, "ramp_samples": round(RAMP_SECONDS * CANONICAL_RATE),
        "boundary_samples": boundary, "peak": actual_peak,
    }


def _source_match(output: WavData, source_a: WavData, source_b: WavData,
                  cue_a: int, cue_b: int, gains: Mapping[str, Any]) -> dict[str, Any]:
    """Compare several source/output windows, excluding ramps and file edges."""

    windows: list[dict[str, Any]] = []
    failures: list[str] = []
    window = 4096
    offsets = (CANONICAL_RATE * 2, CANONICAL_RATE * 8, CANONICAL_RATE * 16)
    for side, source, cue, gain in (("a", source_a, cue_a, gains["gain_factors"]["a"]),
                                    ("b", source_b, cue_b, gains["gain_factors"]["b"])):
        source_start = cue - JOIN_FRAMES if side == "a" else cue
        for offset in offsets:
            output_start = offset if side == "a" else JOIN_FRAMES + offset
            source_start_at = source_start + offset
            source_end = source_start_at + window
            output_end = output_start + window
            check: dict[str, Any] = {"side": side, "output_start": output_start,
                                     "output_end": output_end,
                                     "source_start": source_start_at, "source_end": source_end}
            if source_start_at < 0 or source_end > source.frames or output_end > output.frames:
                check["status"] = "unavailable"
                failures.append(f"{side}_source_window_unavailable")
                windows.append(check)
                continue
            expected = np.rint(np.clip(source.samples[source_start_at:source_end] * gain,
                                       -1.0, 1.0) * 32767.0).astype("<i2")
            actual = output.pcm[output_start:output_end]
            delta = np.abs(actual.astype(np.int32) - expected.astype(np.int32))
            max_error = int(delta.max()) if delta.size else 0
            exact_fraction = float(np.mean(delta == 0)) if delta.size else 0.0
            check.update({"max_abs_error_lsb": max_error, "exact_fraction": exact_fraction,
                          "status": "pass" if max_error <= 1 and exact_fraction >= 0.999 else "fail"})
            if check["status"] == "fail":
                failures.append(f"{side}_source_pcm_mismatch_at_{output_start}")
            windows.append(check)
    fade = round(RAMP_SECONDS * CANONICAL_RATE)
    if cue_a >= fade and cue_b + fade <= source_b.frames and fade > 1:
        ramp_up = np.linspace(0.0, 1.0, fade)
        ramp_down = ramp_up[::-1]
        for side, source, source_slice, output_slice, ramp, gain in (
            ("a", source_a, slice(cue_a - fade, cue_a), slice(JOIN_FRAMES - fade, JOIN_FRAMES),
             ramp_down, gains["gain_factors"]["a"]),
            ("b", source_b, slice(cue_b, cue_b + fade), slice(JOIN_FRAMES, JOIN_FRAMES + fade),
             ramp_up, gains["gain_factors"]["b"]),
        ):
            expected = np.rint(np.clip(source.samples[source_slice] * gain * ramp[:, None],
                                       -1.0, 1.0) * 32767.0).astype("<i2")
            actual = output.pcm[output_slice]
            delta = np.abs(actual.astype(np.int32) - expected.astype(np.int32))
            max_error = int(delta.max()) if delta.size else 0
            ramp_check = {"side": side, "samples": fade, "max_abs_error_lsb": max_error,
                          "status": "pass" if max_error <= 1 else "fail"}
            windows.append({"kind": "join_ramp", **ramp_check})
            if ramp_check["status"] == "fail":
                failures.append(f"{side}_join_ramp_mismatch")
    else:
        failures.append("join_ramp_source_window_unavailable")
    return {"status": "pass" if not failures else "fail", "windows": windows,
            "issues": failures, "window_samples": window, "ramp_samples": fade,
            "ramp_seconds": RAMP_SECONDS,
            "excluded_join_ramp_samples": fade}


def _phrase_audit(value: Any) -> dict[str, Any]:
    """Expose both phrase sides without promoting ASR evidence to closure."""

    if not isinstance(value, Mapping):
        return {"present": False, "unsafe": False, "closure_status": "unknown",
                "word_guard_status": "unknown"}
    closure = _first_value(value, ("phrase_closure_status", "phrase_continuity_status",
                                   "true_phrase_closure", "closure_status"))
    closure_text = str(closure).strip().lower() if closure is not None else "unknown"
    timing = _first_value(value, ("word_clearance_status", "timing_status", "status"))
    timing_text = str(timing).strip().lower() if timing is not None else "unknown"
    unknown_closure = closure_text in {"", "unknown", "missing", "uncertain", "unverified", "false"}
    return {
        "present": True,
        "unsafe": _phrase_unsafe(value),
        "word_guard_status": timing_text,
        "word_gap": value.get("word_gap") is True or value.get("boundary_class") in {"word_gap", "word_gap_only"},
        "active_word": value.get("active_word") is True,
        "active_utterance": value.get("active_utterance") is True,
        "active_asr_segment": value.get("active_asr_segment") is True,
        "acoustic_clearance": value.get("acoustic_clearance"),
        "boundary_class": value.get("boundary_class"),
        "closure_status": closure_text,
        "closure_unknown": unknown_closure,
    }


def _audit_artifact(entry: Mapping[str, Any], index: int, manifest: Mapping[str, Any],
                    base: Path, tracks: Mapping[str, tuple[Mapping[str, Any], Path | None]]) -> dict[str, Any]:
    result: dict[str, Any] = {"index": index, "variant": str(entry.get("variant", "main")),
                              "issues": []}
    result["phrase"] = {
        "a": _phrase_audit(entry.get("phrase_a")),
        "b": _phrase_audit(entry.get("phrase_b")),
    }
    result["word_guard"] = {
        "a_unsafe": result["phrase"]["a"]["unsafe"],
        "b_unsafe": result["phrase"]["b"]["unsafe"],
        "any_unsafe": result["phrase"]["a"]["unsafe"] or result["phrase"]["b"]["unsafe"],
    }
    path = _artifact_path(entry, base)
    if path is None or not path.exists():
        result["issues"].append("artifact_wav_missing")
        result["status"] = "fail"
        return result
    try:
        output = _read_wav(path, require_output_pcm=True)
    except Exception as exc:  # noqa: BLE001 - diagnostic must include malformed artifacts
        result["issues"].append(f"wav_read_error:{exc}")
        result["status"] = "fail"
        return result
    metadata_issues, metadata = _metadata_checks(output, _declared_sha(entry), path)
    result["metadata"] = metadata
    result["issues"].extend(metadata_issues)
    artifact_metadata = entry.get("artifact") if isinstance(entry.get("artifact"), Mapping) else entry
    declared_duration = _as_number(_first_value(artifact_metadata, ("duration_samples", "frames")))
    if declared_duration is not None and int(round(declared_duration)) != EXPECTED_FRAMES:
        result["issues"].append("declared_duration_not_48_seconds")
    declared_join = _as_number(_first_value(artifact_metadata, ("join_sample", "join_frame", "output_join_sample")))
    if declared_join is not None and int(round(declared_join)) != JOIN_FRAMES:
        result["issues"].append("declared_join_not_24_seconds")
    declared_clipped = _first_value(artifact_metadata, ("clipped_samples", "clipped_count", "actual_clipping_samples"))
    clipped_number = _as_number(declared_clipped)
    if clipped_number is not None and clipped_number > 0:
        result["issues"].append("declared_clipping")
    gains = _gain_info(entry)
    result["gain"] = gains
    for side in ("a", "b"):
        value = gains["allowed_base_db"][side]
        if value < -MAX_ATTENUATION_DB - 1e-6 or value > MAX_BOOST_DB + 1e-6:
            result["issues"].append(f"{side}_gain_out_of_bounds")
    if gains["master_trim_db"] > 1e-6:
        result["issues"].append("master_trim_is_positive")

    candidate = _candidate_for(entry, manifest)
    id_a, path_a = _track_for_side(candidate, entry, "a", tracks, base)
    id_b, path_b = _track_for_side(candidate, entry, "b", tracks, base)
    cue_a, cue_b = _cue(candidate, entry, "a"), _cue(candidate, entry, "b")
    result["source"] = {"track_a": id_a, "track_b": id_b, "cue_a": cue_a, "cue_b": cue_b}
    if path_a is None or path_b is None or cue_a is None or cue_b is None:
        result["issues"].append("source_clock_provenance_missing")
    else:
        try:
            source_a = _read_wav(path_a)
            source_b = _read_wav(path_b)
            if source_a.rate != CANONICAL_RATE or source_b.rate != CANONICAL_RATE:
                result["issues"].append("source_rate_not_48000")
            pcm_match = _source_match(output, source_a, source_b, cue_a, cue_b, gains)
            result["source_pcm"] = pcm_match
            result["issues"].extend(pcm_match["issues"])
        except Exception as exc:  # noqa: BLE001
            result["issues"].append(f"source_read_error:{exc}")
    result["status"] = "pass" if not result["issues"] else "fail"
    return result


def _natural_entries(manifest: Mapping[str, Any], base: Path) -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    for key, value in manifest.items():
        normalized = str(key).lower()
        if "natural" in normalized and key not in ("artifacts",):
            if isinstance(value, list):
                result.extend((f"{key}[{index}]", item) for index, item in enumerate(value)
                              if _artifact_path(item if isinstance(item, Mapping)
                                                else {"artifact": item}, base) is not None)
            elif _artifact_path(value if isinstance(value, Mapping) else {"artifact": value},
                                base) is not None:
                result.append((str(key), value))
    return result


def _natural_source_spec(entry: Mapping[str, Any]) -> tuple[str | None, int | None, int | None, float | None, Any]:
    """Return the recorded source window for a natural-A render.

    Natural-A is deliberately checked as a source excerpt, rather than being
    accepted because its metadata looks plausible.  Drivers have used both a
    top-level source description and a nested ``source_pcm`` description, so
    retain the original object for diagnostics while accepting either shape.
    """

    source: Any = entry
    for key in ("source_pcm", "source_window", "natural_source", "source"):
        candidate = entry.get(key)
        if isinstance(candidate, Mapping):
            source = candidate
            break
    track_id = _first_value(source, ("track_id", "source_track_id", "track"))
    start = _as_number(_first_value(source, ("start_sample", "source_start_sample", "start")))
    end = _as_number(_first_value(source, ("end_sample", "source_end_sample", "end")))
    gain = _as_number(_first_value(source, ("gain", "render_gain", "base_gain")))
    return (None if track_id is None else str(track_id),
            None if start is None else int(round(start)),
            None if end is None else int(round(end)), gain, source)


def _natural_source_match(output: WavData, source: WavData, start: int, end: int,
                          gain: float) -> dict[str, Any]:
    """Compare a natural-A file to its catalogued PCM source range.

    The 10 ms edge ramps are part of the recorded render contract.  Keeping
    this comparison quantised to int16 catches shifts and gain changes that a
    metadata-only audit cannot see.
    """

    result: dict[str, Any] = {"status": "fail", "start_sample": start,
                              "end_sample": end, "gain": gain,
                              "ramp_seconds": 0.01,
                              "ramp_samples": round(0.01 * CANONICAL_RATE)}
    if source.rate != CANONICAL_RATE:
        result["issue"] = "source_rate_not_48000"
        return result
    if start < 0 or end <= start or end > source.frames:
        result["issue"] = "source_range_invalid"
        return result
    if output.frames != end - start:
        result["issue"] = "source_range_length_mismatch"
        result["output_frames"] = output.frames
        result["expected_frames"] = end - start
        return result
    fade = min(round(0.01 * CANONICAL_RATE), output.frames // 2)
    envelope = np.ones(output.frames, dtype=np.float64)
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade)
        envelope[:fade] = ramp
        envelope[-fade:] = ramp[::-1]
    expected = np.rint(np.clip(source.samples[start:end] * gain * envelope[:, None],
                               -1.0, 1.0) * 32767.0).astype("<i2")
    actual = output.pcm
    delta = np.abs(actual.astype(np.int32) - expected.astype(np.int32))
    max_error = int(delta.max()) if delta.size else 0
    exact_fraction = float(np.mean(delta == 0)) if delta.size else 0.0
    result.update({"max_abs_error_lsb": max_error, "exact_fraction": exact_fraction,
                   "status": "pass" if max_error <= 1 and exact_fraction >= 0.999 else "fail"})
    if result["status"] == "fail":
        result["issue"] = "natural_source_pcm_mismatch"
    return result


def _audit_natural(name: str, value: Any, base: Path,
                   tracks: Mapping[str, tuple[Mapping[str, Any], Path | None]]) -> dict[str, Any]:
    entry = value if isinstance(value, Mapping) else {"artifact": value, "variant": "natural_a"}
    path = _artifact_path(entry, base)
    result: dict[str, Any] = {"name": name, "variant": "natural_a", "issues": []}
    result["source_pcm"] = {"status": "unknown",
                             "note": "natural-A requires a catalogued source range"}
    if path is None or not path.exists():
        result["issues"].append("natural_wav_missing")
        result["status"] = "fail"
        return result
    try:
        # Natural references are also canonical PCM16 renders.  Keeping the
        # original int16 payload avoids introducing a second quantisation step
        # when comparing against the source slice.
        data = _read_wav(path, require_output_pcm=True)
        issues, metadata = _metadata_checks(data, _declared_sha(entry), path)
        # Natural-A may be a source-length excerpt.  Its duration is checked
        # against the recorded source range below, rather than hard-coded to
        # the 48-second A/B handoff duration.
        issues = [issue for issue in issues if issue != "duration_not_48_seconds"]
        result["metadata"] = metadata
        result["issues"].extend(issues)
        clipped_number = _as_number(_first_value(entry, ("clipped_samples", "clipped_count",
                                                          "actual_clipping_samples")))
        if clipped_number is not None and clipped_number > 0:
            result["issues"].append("declared_clipping")
        track_id, start, end, gain, source_spec = _natural_source_spec(entry)
        result["source"] = {"track_id": track_id, "start_sample": start,
                             "end_sample": end, "gain": gain}
        source_path: Path | None = None
        if track_id is not None and track_id in tracks:
            source_path = tracks[track_id][1]
        if source_path is None:
            source_path = _resolve_path(source_spec, base)
        if source_path is None or start is None or end is None or gain is None:
            result["issues"].append("natural_source_provenance_missing")
        elif not np.isclose(gain, KNOWN_RENDER_GAIN, rtol=0.0, atol=1e-9):
            result["issues"].append("natural_gain_not_0_9")
        elif not source_path.exists():
            result["issues"].append("natural_source_wav_missing")
        else:
            source = _read_wav(source_path, require_output_pcm=False)
            source_match = _natural_source_match(data, source, start, end, gain)
            result["source_pcm"] = source_match
            if source_match.get("status") != "pass":
                result["issues"].append(str(source_match.get("issue", "natural_source_pcm_mismatch")))
    except Exception as exc:  # noqa: BLE001
        result["issues"].append(f"wav_read_error:{exc}")
    result["status"] = "pass" if not result["issues"] else "fail"
    return result


def _phrase_unsafe(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    # A raw ASR segment and a word gap are timing observations, not proof that
    # a cut lands inside speech.  ``phrase_evidence`` may reconcile an ASR
    # segment with acoustic silence; only its actual occupancy/status fields
    # can make execution unsafe.
    explicit = ("active_utterance", "active_word", "occupied", "unsafe")
    for key in explicit:
        if value.get(key) is True:
            return True
    for key in ("status", "actual_status", "vocal_status", "occupancy_status",
                "boundary_status", "boundary_class"):
        raw = value.get(key)
        if raw is None:
            continue
        status = str(raw).strip().lower()
        if status in {"unsafe", "active_utterance", "active_word", "occupied",
                      "vocal_occupancy", "vocal_occupancy_unsafe"}:
            return True
        if status in {"supported", "safe", "acoustic_silence", "acoustic_silence_supported",
                      "acoustic_clearance_supported", "word_gap", "word_gap_only",
                      "active_asr_segment", "observed_asr_segment_end", "unknown",
                      "pickup_uncertain"}:
            continue
    return False


def _walk_named(obj: Any, names: Iterable[str]) -> Iterable[tuple[str, Any]]:
    wanted = {str(name).lower() for name in names}
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if str(key).lower() in wanted:
                yield str(key).lower(), value
            yield from _walk_named(value, wanted)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_named(value, wanted)


def _policy_audit(entries: list[Mapping[str, Any]]) -> dict[str, Any]:
    actions: Counter[str] = Counter()
    coverage_values: list[float] = []
    issues: list[str] = []
    for index, entry in enumerate(entries):
        assessment = entry.get("assessment") if isinstance(entry.get("assessment"), Mapping) else entry
        action = str(_first_value(assessment, ("action", "recommended_action")) or "unknown").lower()
        actions[action] += 1
        coverage = _first_value(assessment, ("coverage_fraction", "coverage"))
        if isinstance(coverage, Mapping):
            coverage = _first_value(coverage, ("fraction", "ratio", "value"))
        number = _as_number(coverage)
        if number is not None and 0.0 <= number <= 1.0:
            coverage_values.append(number)
        phrases = []
        for key in ("phrase_a", "phrase_b", "phrase_evidence", "a_phrase", "b_phrase"):
            value = _first_value(entry, (key,))
            if value is not None:
                phrases.append(value)
        support = assessment.get("functional_support", assessment.get("functional_evidence"))
        if isinstance(support, list):
            phrases.extend(item for item in support
                           if isinstance(item, Mapping)
                           and "phrase" in str(item.get("kind", "")).lower())
        if action == "execute" and any(_phrase_unsafe(value) for value in phrases):
            issues.append(f"execute_with_unsafe_phrase:{index}")
        missing_closure = False
        for key, value in _walk_named(entry, ("true_phrase_closure", "phrase_closure", "closure_status")):
            text = str(value).lower() if value is not None else "unknown"
            if value is None or text in {"unknown", "missing", "uncertain", "unverified", "false"}:
                missing_closure = True
        for key, value in _walk_named(entry, ("closure_unknown", "missing_closure")):
            if value is True:
                missing_closure = True
        certified = False
        for key, value in _walk_named(entry, ("closure_certified", "phrase_closure_certified",
                                                "arrival_certified", "certified_closure",
                                                "closure_status", "phrase_closure_status")):
            if value is True or str(value).lower() in {"certified", "supported", "true"}:
                certified = True
        if missing_closure and certified:
            issues.append(f"missing_closure_claimed_certified:{index}")
    return {
        "status": "pass" if not issues else "fail", "artifact_count": len(entries),
        "action_counts": dict(actions), "coverage_count": len(coverage_values),
        "coverage_mean": (float(np.mean(coverage_values)) if coverage_values else None),
        "coverage_min": (float(np.min(coverage_values)) if coverage_values else None),
        "coverage_max": (float(np.max(coverage_values)) if coverage_values else None),
        "issues": issues,
    }


def _history_paths(obj: Any, base: Path, context: bool = False,
                   seen_json: set[Path] | None = None) -> list[Path]:
    if seen_json is None:
        seen_json = set()
    paths: list[Path] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            key_context = context or any(token in str(key).lower() for token in ("v10", "history", "previous"))
            if key_context and isinstance(value, str):
                if value.lower().endswith(".wav"):
                    resolved = _resolve_path(value, base)
                    if resolved is not None:
                        paths.append(resolved)
                elif value.lower().endswith(".json"):
                    resolved = _resolve_path(value, base)
                    resolved_key = resolved.resolve() if resolved is not None else None
                    if (resolved is not None and resolved.exists()
                            and resolved_key not in seen_json):
                        seen_json.add(resolved_key)
                        try:
                            paths.extend(_history_paths(json.loads(resolved.read_text(encoding="utf-8")),
                                                        resolved.parent, True, seen_json))
                        except (OSError, ValueError, json.JSONDecodeError):
                            pass
            paths.extend(_history_paths(value, base, key_context, seen_json))
    elif isinstance(obj, list):
        for value in obj:
            paths.extend(_history_paths(value, base, context, seen_json))
    return paths


def _history_hashes(obj: Any, context: bool = False) -> set[str]:
    hashes: set[str] = set()
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            key_context = context or any(token in str(key).lower() for token in ("v10", "history", "previous"))
            if key_context and isinstance(value, str) and len(value) == 64:
                try:
                    int(value, 16)
                except ValueError:
                    pass
                else:
                    hashes.add(value.lower())
            hashes.update(_history_hashes(value, key_context))
    elif isinstance(obj, list):
        for value in obj:
            hashes.update(_history_hashes(value, context))
    return hashes


def _history_snapshot(manifest: Mapping[str, Any], base: Path) -> tuple[list[dict[str, Any]], list[str], bool, list[str]]:
    """Load explicit audition-history regions without inferring them from audio.

    The V11 drivers have emitted a mapping under ``inputs.history_snapshot``
    and a JSON path under the same key.  A list item in ``inputs`` is accepted
    as well, since that is the shape used by the older expectation pipeline.
    """

    raw_values: list[Any] = []
    inputs = manifest.get("inputs")
    if isinstance(inputs, Mapping) and "history_snapshot" in inputs:
        raw_values.append(inputs.get("history_snapshot"))
    elif isinstance(inputs, list):
        for item in inputs:
            if isinstance(item, Mapping):
                key_text = " ".join(str(key).lower() for key in item)
                path_text = str(item.get("path", "")).lower()
                if "history_snapshot" in item or "audition_history" in key_text \
                        or "history" in path_text:
                    raw_values.append(item.get("history_snapshot", item))
    if "history_snapshot" in manifest:
        raw_values.append(manifest.get("history_snapshot"))

    regions: list[dict[str, Any]] = []
    paths: list[str] = []
    errors: list[str] = []

    def consume(value: Any, parent: Path) -> None:
        if isinstance(value, (str, Path)):
            resolved = _resolve_path(value, parent)
            if resolved is None:
                return
            paths.append(str(resolved))
            if not resolved.exists():
                errors.append(f"history_snapshot_missing:{resolved}")
                return
            try:
                payload = json.loads(resolved.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"history_snapshot_read_error:{exc}")
                return
            consume(payload, resolved.parent)
            return
        if isinstance(value, list):
            for item in value:
                consume(item, parent)
            return
        if not isinstance(value, Mapping):
            return
        if "path" in value and not isinstance(value.get("regions"), list):
            path_value = value.get("path")
            if isinstance(path_value, (str, Path)):
                consume(path_value, parent)
        explicit = value.get("regions")
        if isinstance(explicit, list):
            regions.extend(item for item in explicit if isinstance(item, Mapping))
        for key in ("history_snapshot", "audition_history", "history"):
            nested = value.get(key)
            if nested is not None and nested is not value:
                consume(nested, parent)

    for value in raw_values:
        consume(value, base)
    unique_paths = list(dict.fromkeys(paths))
    unique_regions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for region in regions:
        try:
            identity = json.dumps(region, sort_keys=True, separators=(",", ":"))
        except TypeError:
            identity = repr(region)
        if identity not in seen:
            seen.add(identity)
            unique_regions.append(dict(region))
    return unique_regions, unique_paths, bool(raw_values), errors


def _history_candidate(entry: Mapping[str, Any], manifest: Mapping[str, Any]) -> Any:
    """Create the tiny candidate view required by audition_history.exposure."""

    from types import SimpleNamespace

    candidate = _candidate_for(entry, manifest)
    if not candidate:
        candidate = entry.get("candidate") if isinstance(entry.get("candidate"), Mapping) else entry
    track_a = _first_value(candidate, ("track_a", "track_a_id", "outgoing_track_id"))
    track_b = _first_value(candidate, ("track_b", "track_b_id", "incoming_track_id"))
    cue_a = _cue(candidate, entry, "a")
    cue_b = _cue(candidate, entry, "b")
    candidate_id = _first_value(candidate, ("candidate_id", "id")) or entry.get("candidate_id")
    if track_a is None or track_b is None or cue_a is None or cue_b is None:
        return None
    return SimpleNamespace(candidate_id=str(candidate_id or "unknown"), track_a=str(track_a),
                           track_b=str(track_b), cue_out_sample=int(cue_a),
                           cue_in_sample=int(cue_b))


def _history_audit(manifest: Mapping[str, Any], base: Path, outputs: list[dict[str, Any]],
                   entries: list[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    regions, snapshot_paths, snapshot_present, snapshot_errors = _history_snapshot(manifest, base)
    # A complete snapshot carries the actual presented artifact in each
    # region.  Restrict byte-level diagnostics to those paths; recursively
    # walking every JSON input also discovers canonical source/stem files and
    # turns a bounded review into a multi-gigabyte search.
    snapshot_audio = [_resolve_path(region.get("source"), base)
                      for region in regions if isinstance(region, Mapping)
                      and region.get("source") is not None]
    snapshot_audio = [path for path in snapshot_audio if path is not None]
    history = snapshot_audio if snapshot_present and snapshot_audio else _history_paths(manifest, base)
    history = list(dict.fromkeys(history))
    history_hashes = _history_hashes(manifest)
    # A byte-identical 8-second interior is expected when a new 48-second
    # preview deliberately contains source context around a new cue.  Keep it
    # as a descriptive context-reuse observation; novelty is decided from
    # coordinate exposure below, or from an exact whole-transition duplicate.
    context_overlaps: list[dict[str, Any]] = []
    whole_transition_matches: list[dict[str, Any]] = []
    for output in outputs:
        path = Path(output["metadata"]["path"]) if output.get("metadata") else None
        if path is None or not path.exists():
            continue
        try:
            generated = _read_wav(path, require_output_pcm=True)
        except Exception:
            continue
        chunk = CANONICAL_RATE * 8
        generated_bytes = generated.pcm.astype("<i2", copy=False).tobytes()
        for historical_path in history:
            if not historical_path.exists():
                continue
            try:
                historical = _read_wav(historical_path, require_output_pcm=True)
            except Exception:
                continue
            historical_bytes = historical.pcm.astype("<i2", copy=False).tobytes()
            if generated.frames == historical.frames and generated_bytes == historical_bytes:
                whole_transition_matches.append({"kind": "whole_transition",
                                                 "artifact": str(path),
                                                 "history": str(historical_path),
                                                 "duration_samples": generated.frames})
            if len(historical_bytes) < chunk * 2 * 2:
                continue
            for start in range(0, max(0, generated.frames - chunk + 1), CANONICAL_RATE):
                piece = generated_bytes[start * 4:(start + chunk) * 4]
                if len(piece) != chunk * 4:
                    continue
                if historical_bytes.find(piece) >= 0:
                    context_overlaps.append({"kind": "context_overlap",
                                             "artifact": str(path), "history": str(historical_path),
                                             "output_start": start, "duration_samples": chunk})
                    break
        for start in range(0, max(0, generated.frames - chunk + 1), CANONICAL_RATE):
            piece = generated_bytes[start * 4:(start + chunk) * 4]
            digest = hashlib.sha256(piece).hexdigest() if len(piece) == chunk * 4 else None
            if digest is not None and digest in history_hashes:
                context_overlaps.append({"kind": "context_overlap", "artifact": str(path),
                                         "history_hash": digest, "output_start": start,
                                         "duration_samples": chunk})
    history_hashes = sorted(history_hashes)
    valid_regions = [region for region in regions
                     if all(key in region for key in ("track_a", "track_b", "cue_out_sample",
                                                      "cue_in_sample", "sample_rate"))]
    exposure_matches: list[dict[str, Any]] = []
    role_counts = {"outgoing_region_heard": 0, "incoming_region_heard": 0,
                   "any_role_heard": 0, "ordered_pair_heard": 0}
    unresolved_candidates = 0
    if valid_regions:
        try:
            from musicmix.audition_history import exposure
        except ImportError as exc:  # pragma: no cover - package invocation supplies PYTHONPATH
            # The documented CLI invocation runs this file directly, where
            # ``src`` is not automatically on sys.path.  Use the adjacent
            # project package without requiring an environment mutation or
            # installation.
            source_root = Path(__file__).resolve().parents[1] / "src"
            if str(source_root) not in sys.path:
                sys.path.insert(0, str(source_root))
            try:
                from musicmix.audition_history import exposure
            except ImportError as fallback_exc:
                snapshot_errors.append(f"history_exposure_import_error:{fallback_exc}")
                exposure = None
        for output in outputs:
            output_index = output.get("index")
            manifest_entries = entries if entries is not None else manifest.get("artifacts", [])
            entry = (manifest_entries[output_index]
                     if isinstance(manifest_entries, list)
                     and isinstance(output_index, int)
                     and 0 <= output_index < len(manifest_entries)
                     and isinstance(manifest_entries[output_index], Mapping)
                     else {})
            candidate = _history_candidate(entry, manifest)
            if candidate is None or exposure is None:
                unresolved_candidates += 1
                continue
            try:
                observed = exposure(candidate, valid_regions, 8.0, sample_rate=CANONICAL_RATE)
            except (KeyError, TypeError, ValueError) as exc:
                unresolved_candidates += 1
                snapshot_errors.append(f"history_exposure_error:{exc}")
                continue
            any_role = bool(observed.get("outgoing_region_heard") or
                            observed.get("incoming_region_heard"))
            row = {"artifact": output.get("metadata", {}).get("path"),
                   "candidate_id": getattr(candidate, "candidate_id", "unknown"),
                   "exposure": {**observed, "any_role_heard": any_role}}
            exposure_matches.append(row)
            for key in ("outgoing_region_heard", "incoming_region_heard", "ordered_pair_heard"):
                if observed.get(key):
                    role_counts[key] += 1
            if any_role:
                role_counts["any_role_heard"] += 1

    coordinate_exposure = [row for row in exposure_matches
                           if isinstance(row.get("exposure"), Mapping)
                           and row["exposure"].get("any_role_heard")]
    hard_reuse = list(coordinate_exposure)
    hard_reuse.extend(whole_transition_matches)
    if hard_reuse:
        status = "fail"
    elif not snapshot_present or not valid_regions:
        status = "unknown"
    elif snapshot_errors and unresolved_candidates:
        status = "unknown"
    else:
        status = "pass"
    matches = list(context_overlaps)
    matches.extend(whole_transition_matches)
    matches.extend(exposure_matches)
    return {"status": status, "history_paths": [str(path) for path in history],
            "history_hashes": history_hashes, "matches": matches,
            "byte_exact_matches": context_overlaps,
            "context_overlaps": context_overlaps,
            "whole_transition_matches": whole_transition_matches,
            "hard_reuse": hard_reuse,
            "history_snapshot": {"present": snapshot_present,
                                 "paths": snapshot_paths,
                                 "region_count": len(regions),
                                 "valid_region_count": len(valid_regions),
                                 "errors": snapshot_errors},
            "exposure": {"tolerance_seconds": 8.0,
                         "candidate_count": len(outputs),
                         "evaluated_count": len(exposure_matches),
                         "unresolved_count": unresolved_candidates,
                         "counts": role_counts,
                         "matches": exposure_matches,
                         "hard": bool(coordinate_exposure)}}


def audit_manifest(manifest_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    if output_path is None:
        output = base / "qa_independent_v2.json"
    else:
        output = Path(output_path).expanduser()
        if not output.is_absolute():
            output = base / output
        output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing QA report: {output}")
    entries = manifest.get("artifacts", [])
    if not isinstance(entries, list):
        raise ValueError("manifest.artifacts must be a list")
    entries = [entry for entry in entries if isinstance(entry, Mapping)]
    tracks = _track_map(manifest, base)
    artifact_reports = [_audit_artifact(entry, index, manifest, base, tracks)
                        for index, entry in enumerate(entries)]
    natural_reports = [_audit_natural(name, value, base, tracks)
                       for name, value in _natural_entries(manifest, base)]
    policy = _policy_audit(entries)
    history = _history_audit(manifest, base, artifact_reports, entries)
    issues = [f"artifact[{item['index']}]:{issue}" for item in artifact_reports for issue in item["issues"]]
    issues.extend(f"natural[{item['name']}]:{issue}" for item in natural_reports for issue in item["issues"])
    issues.extend(f"policy:{issue}" for issue in policy["issues"])
    if history["status"] == "fail":
        if history.get("whole_transition_matches"):
            issues.append("v10_history_repeated_whole_transition")
            # Preserve the V1 hard-failure marker for exact whole repeats;
            # ordinary interior context overlaps never emit it.
            issues.append("v10_history_repeated_8_seconds")
        if history.get("exposure", {}).get("hard"):
            issues.append("v10_history_coordinate_exposure")
    elif history["status"] == "unknown":
        issues.append("audition_history_snapshot_unknown")
    if not entries:
        issues.append("no_outputs")
        policy["status"] = "unknown"
    report: dict[str, Any] = {
        "schema_version": "psychology_v11_independent_qa_v2",
        "status": "pass" if not issues else "fail",
        "manifest": str(manifest_path),
        "canonical": {"sample_rate": CANONICAL_RATE, "duration_seconds": EXPECTED_SECONDS,
                      "duration_samples": EXPECTED_FRAMES, "join_seconds": JOIN_SECONDS,
                      "join_sample": JOIN_FRAMES, "render_gain": KNOWN_RENDER_GAIN,
                      "ramp_seconds": RAMP_SECONDS},
        "artifacts": artifact_reports, "natural": natural_reports,
        "outputs": {"count": len(entries),
                    "status": "no_outputs" if not entries else "validated"},
        "policy": policy, "history": history, "issues": issues,
        "eligibility": {
            "eligible_for_review": not bool(issues),
            "musically_validated": False,
            "musical_status": "unknown",
            "context_reuse_explicit": bool(history.get("context_overlaps")),
            "novelty_hard_failure": bool(history.get("hard_reuse")),
        },
        "audit_code_sha256": sha256_file(Path(__file__).resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    report["qa_path"] = str(output)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="psychology-v11 manifest.json")
    parser.add_argument("--output", type=Path, default=None,
                        help="QA path (default: next to manifest as qa_independent_v2.json)")
    args = parser.parse_args(argv)
    result = audit_manifest(args.manifest, args.output)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"artifacts", "natural", "issues"}}, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

