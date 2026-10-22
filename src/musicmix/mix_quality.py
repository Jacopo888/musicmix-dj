"""Technical artifact assessment for a rendered mix seam.

This module answers a deliberately narrow question: does the rendered stereo
waveform contain measurable technical defects near ``join_sample``?  It does
not estimate musical quality, tension, resolution, or listener preference.
The selector features (chroma, beat grid, structure labels, embeddings) are
never consulted, so this report cannot leak into cue ranking as if it were a
musical critic.

All thresholds are project heuristics.  They are recorded in ``config`` and
bound into ``config_sha256`` so a report can be reproduced and audited.
``confidence`` is always ``None``: no validated perceptual model was run.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from .audio import AudioTrack


MIX_QUALITY_VERSION = "mix-quality-technical-v1"
MIX_QUALITY_SCHEMA = "musicmix.mix_quality.1"

DEFAULT_CONFIG: dict[str, float] = {
    "clip_threshold_full_scale": 0.999,
    "silence_threshold_dbfs": -60.0,
    "seam_guard_seconds": 0.002,
    "join_jump_ratio_threshold": 8.0,
    "seam_silence_seconds": 0.05,
    "dc_offset_threshold_full_scale": 0.01,
    "rms_change_threshold_db": 6.0,
}


def _db(value: float | None) -> float | None:
    if value is None or not math.isfinite(value) or value <= 0.0:
        return None
    return 20.0 * math.log10(value)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _config_sha256(config: dict[str, float]) -> str:
    payload = json.dumps(
        {"version": MIX_QUALITY_VERSION, "config": config},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _samples_sha256(samples: np.ndarray, sample_rate: int) -> str:
    canonical = np.ascontiguousarray(samples, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(b"musicmix.samples.v1\0")
    digest.update(str(int(sample_rate)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(int(canonical.shape[0])).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(int(canonical.shape[1])).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _peak(array: np.ndarray) -> float | None:
    if array.size == 0:
        return None
    return float(np.max(np.abs(array)))


def _rms(array: np.ndarray) -> float | None:
    if array.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))


def _rms_per_channel(array: np.ndarray) -> list[float]:
    if array.size == 0:
        return []
    values = np.sqrt(np.mean(np.square(array, dtype=np.float64), axis=0))
    return [float(value) for value in values]


def _mean_per_channel(array: np.ndarray) -> list[float]:
    if array.size == 0:
        return []
    return [float(value) for value in np.mean(array, axis=0)]


def _edge_silence_seconds(array: np.ndarray, sample_rate: int, reverse: bool) -> float | None:
    if array.size == 0:
        return None
    frame_peak = np.max(np.abs(array), axis=1)
    mask = frame_peak <= _SILENCE_AMPLITUDE
    ordered = mask[::-1] if reverse else mask
    count = 0
    for silent in ordered:
        if not bool(silent):
            break
        count += 1
    return count / float(sample_rate)


def _silence_fraction(array: np.ndarray) -> float | None:
    if array.size == 0:
        return None
    frame_peak = np.max(np.abs(array), axis=1)
    return float(np.mean(frame_peak <= _SILENCE_AMPLITUDE))


def _guard_excluded_samples(
    samples: np.ndarray,
    join: int,
    pre_start: int,
    pre_end: int,
    post_start: int,
    post_end: int,
    guard: int,
) -> np.ndarray | None:
    pieces: list[np.ndarray] = []
    if pre_end > pre_start:
        end = min(pre_end, join - guard)
        if end > pre_start:
            pieces.append(samples[pre_start:end])
    if post_end > post_start:
        start = max(post_start, join + guard)
        if post_end > start:
            pieces.append(samples[start:post_end])
    if not pieces:
        return None
    return np.concatenate(pieces, axis=0)


def _guard_excluded_derivatives(
    samples: np.ndarray,
    join: int,
    pre_start: int,
    pre_end: int,
    post_start: int,
    post_end: int,
    guard: int,
) -> np.ndarray | None:
    pieces: list[np.ndarray] = []
    if pre_end - pre_start >= 2:
        differences = samples[pre_start + 1 : pre_end] - samples[pre_start : pre_end - 1]
        indices = np.arange(pre_start, pre_end - 1)
        keep = indices < join - guard
        if bool(np.any(keep)):
            pieces.append(differences[keep])
    if post_end - post_start >= 2:
        differences = samples[post_start + 1 : post_end] - samples[post_start : post_end - 1]
        indices = np.arange(post_start, post_end - 1)
        keep = indices >= join + guard
        if bool(np.any(keep)):
            pieces.append(differences[keep])
    if not pieces:
        return None
    return np.abs(np.concatenate([piece.reshape(-1) for piece in pieces]))


_SILENCE_AMPLITUDE = 10.0 ** (DEFAULT_CONFIG["silence_threshold_dbfs"] / 20.0)


def evaluate_mix_quality(
    track: AudioTrack,
    join_sample: int,
    *,
    context_seconds: float = 1.0,
) -> dict[str, Any]:
    """Measure technical artifacts around one rendered seam.

    ``track`` is the rendered mix and ``join_sample`` is the first sample of the
    incoming segment.  The report is descriptive only.  It contains no
    ``score`` and no estimated musical-quality value.  A ratio of the seam jump
    to typical nearby sample derivatives is ``None`` when the neighborhood is
    digital/near silence, because a relative jump is undefined there.

    Raises ``ValueError`` for a non-positive/non-finite context or a join
    outside ``[0, duration_samples]``.  Missing context at the track edges is
    reported through availability fields instead of being imputed.
    """

    try:
        context = float(context_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("context_seconds must be a finite positive number") from exc
    if not math.isfinite(context) or context <= 0.0:
        raise ValueError("context_seconds must be a finite positive number")

    samples = np.asarray(track.samples, dtype=np.float64)
    frame_count = int(samples.shape[0])
    join = int(join_sample)
    if join < 0 or join > frame_count:
        raise ValueError("join_sample must be within [0, duration_samples]")

    sample_rate = int(track.sample_rate)
    context_samples = max(1, int(round(context * sample_rate)))
    guard = max(1, int(round(DEFAULT_CONFIG["seam_guard_seconds"] * sample_rate)))

    pre_start = max(0, join - context_samples)
    pre_end = join
    post_start = join
    post_end = min(frame_count, join + context_samples)
    pre = samples[pre_start:pre_end]
    post = samples[post_start:post_end]

    config = dict(DEFAULT_CONFIG)
    config["context_seconds"] = context

    if frame_count == 0:
        peak = None
        peak_dbfs = None
        clipped_count = 0
        clipped_fraction = None
    else:
        absolute = np.abs(samples)
        peak = float(np.max(absolute))
        peak_dbfs = _db(peak)
        clipped_count = int(np.count_nonzero(absolute >= config["clip_threshold_full_scale"]))
        clipped_fraction = clipped_count / float(absolute.size)

    pre_dc = _mean_per_channel(pre)
    post_dc = _mean_per_channel(post)
    dc_values = [abs(value) for value in pre_dc + post_dc]
    dc_max_abs = max(dc_values) if dc_values else None

    pre_rms = _rms(pre)
    post_rms = _rms(post)
    pre_rms_db = _db(pre_rms)
    post_rms_db = _db(post_rms)
    rms_change_db = None
    if pre_rms is not None and post_rms is not None and pre_rms > 0.0 and post_rms > 0.0:
        rms_change_db = 20.0 * math.log10(post_rms / pre_rms)

    pre_edge = _edge_silence_seconds(pre, sample_rate, reverse=True)
    post_edge = _edge_silence_seconds(post, sample_rate, reverse=False)
    pre_silence_fraction = _silence_fraction(pre)
    post_silence_fraction = _silence_fraction(post)

    neighborhood = _guard_excluded_samples(
        samples, join, pre_start, pre_end, post_start, post_end, guard
    )
    neighborhood_peak = _peak(neighborhood) if neighborhood is not None else None
    derivatives = _guard_excluded_derivatives(
        samples, join, pre_start, pre_end, post_start, post_end, guard
    )
    typical_derivative = None
    if derivatives is not None and derivatives.size:
        typical_derivative = float(np.median(derivatives))

    if frame_count == 0:
        jump = None
        jump_availability = "no_samples"
        jump_reason = "the rendered track has no samples"
    elif join == 0:
        jump = None
        jump_availability = "no_preceding_sample"
        jump_reason = "join_sample is at the first sample; no preceding sample exists"
    elif join >= frame_count:
        jump = None
        jump_availability = "no_following_sample"
        jump_reason = "join_sample is at the end of the track; no following sample exists"
    else:
        jump = float(np.max(np.abs(samples[join] - samples[join - 1])))
        jump_availability = "available"
        jump_reason = None

    ratio = None
    ratio_availability = jump_availability
    ratio_reason = jump_reason
    if jump is not None:
        if derivatives is None or not derivatives.size or typical_derivative is None:
            ratio_availability = "insufficient_context"
            ratio_reason = "no non-guard derivative samples are available in the context window"
        elif (
            neighborhood_peak is None
            or neighborhood_peak < _SILENCE_AMPLITUDE
            or typical_derivative <= 0.0
        ):
            ratio_availability = "undefined_in_silence"
            ratio_reason = (
                "relative seam jump is undefined because the nearby signal is at or below "
                f"{config['silence_threshold_dbfs']:.1f} dBFS"
            )
        else:
            ratio = jump / typical_derivative
            ratio_availability = "available"
            ratio_reason = None

    flags: list[dict[str, Any]] = []

    def add_flag(code: str, severity: str, description: str) -> None:
        flags.append({"code": code, "severity": severity, "description": description})

    if frame_count == 0:
        add_flag("empty_track", "technical_defect", "The rendered track contains no samples.")
    if clipped_count > 0:
        add_flag(
            "clipping_detected",
            "technical_defect",
            f"{clipped_count} sample values reach or exceed "
            f"{config['clip_threshold_full_scale']:.3f} full scale.",
        )
    if ratio is not None and ratio >= config["join_jump_ratio_threshold"]:
        add_flag(
            "join_discontinuity_high",
            "heuristic_observation",
            f"Seam jump is {ratio:.3f}x the median nearby sample derivative "
            f"(project threshold {config['join_jump_ratio_threshold']:.3f}x).",
        )
    if ratio is None and frame_count > 0:
        add_flag(
            "join_discontinuity_undefined",
            "missing_evidence",
            f"Relative seam jump unavailable: {ratio_reason}.",
        )
    if (pre_edge is not None and pre_edge >= config["seam_silence_seconds"]) or (
        post_edge is not None and post_edge >= config["seam_silence_seconds"]
    ):
        add_flag(
            "seam_silence_detected",
            "heuristic_observation",
            "At least one side of the seam contains a near-silent edge of at least "
            f"{config['seam_silence_seconds']:.3f} s; this may be intentional.",
        )
    if dc_max_abs is not None and dc_max_abs >= config["dc_offset_threshold_full_scale"]:
        add_flag(
            "dc_offset_detected",
            "heuristic_observation",
            f"Maximum absolute channel DC offset is {dc_max_abs:.6f} full scale "
            f"(project threshold {config['dc_offset_threshold_full_scale']:.3f}).",
        )
    if rms_change_db is not None and abs(rms_change_db) >= config["rms_change_threshold_db"]:
        add_flag(
            "large_level_change_descriptive",
            "descriptive_only",
            f"Post/pre RMS changes by {rms_change_db:.3f} dB; this is a level observation, "
            "not a quality judgement.",
        )

    return {
        "schema_version": MIX_QUALITY_SCHEMA,
        "version": MIX_QUALITY_VERSION,
        "status": "technical_only",
        "musical_continuity": "unknown_without_validated_critic",
        "musical_quality": None,
        "musical_quality_reason": (
            "No validated perceptual critic was executed; technical artifact metrics are not "
            "a musical-quality score."
        ),
        "confidence": None,
        "confidence_reason": (
            "No calibrated perceptual model or human listening evidence is attached to this "
            "technical report."
        ),
        "track": {
            "track_id": track.track_id,
            "track_hash": track.track_id,
            "sample_rate": sample_rate,
            "duration_samples": frame_count,
            "channels": int(samples.shape[1]) if samples.ndim == 2 else None,
            "samples_sha256": _samples_sha256(samples, sample_rate),
            "samples_dtype_for_hash": "float64_le",
        },
        "join": {
            "sample": join,
            "seconds": join / float(sample_rate),
            "context_seconds": context,
            "context_samples": context_samples,
            "guard_samples": guard,
            "pre_samples": int(pre.shape[0]),
            "post_samples": int(post.shape[0]),
        },
        "config": config,
        "config_sha256": _config_sha256(config),
        "metrics": {
            "sample_peak": {
                "value": peak,
                "dbfs": peak_dbfs,
                "unit": "full_scale",
                "availability": "available" if peak is not None else "no_samples",
            },
            "clipping": {
                "clipped_sample_count": clipped_count,
                "clipped_sample_fraction": clipped_fraction,
                "threshold_full_scale": config["clip_threshold_full_scale"],
                "unit": "full_scale",
                "availability": "available" if frame_count else "no_samples",
            },
            "dc_offset": {
                "pre_per_channel": pre_dc,
                "post_per_channel": post_dc,
                "pre_max_abs": max((abs(v) for v in pre_dc), default=None),
                "post_max_abs": max((abs(v) for v in post_dc), default=None),
                "max_abs": dc_max_abs,
                "unit": "full_scale",
                "availability": "available" if dc_values else "missing_context",
            },
            "join_jump": {
                "absolute": jump,
                "typical_neighborhood_derivative": typical_derivative,
                "ratio_to_typical_derivative": ratio,
                "unit": "full_scale",
                "derivative_unit": "full_scale_per_sample",
                "ratio_unit": "dimensionless",
                "availability": ratio_availability,
                "absolute_availability": jump_availability,
                "reason": ratio_reason,
            },
            "seam_silence": {
                "pre_edge_seconds": pre_edge,
                "post_edge_seconds": post_edge,
                "pre_fraction": pre_silence_fraction,
                "post_fraction": post_silence_fraction,
                "threshold_dbfs": config["silence_threshold_dbfs"],
                "unit": "seconds",
                "fraction_unit": "dimensionless",
                "availability": "available" if frame_count else "no_samples",
            },
            "rms": {
                "pre": pre_rms,
                "post": post_rms,
                "pre_dbfs": pre_rms_db,
                "post_dbfs": post_rms_db,
                "change_db": rms_change_db,
                "pre_per_channel": _rms_per_channel(pre),
                "post_per_channel": _rms_per_channel(post),
                "unit": "full_scale_rms",
                "change_unit": "dB",
                "availability": (
                    "available"
                    if pre_rms is not None and post_rms is not None
                    else "missing_context"
                ),
                "descriptive_only": True,
            },
        },
        "flags": flags,
        "evidence": {
            "selector_features_used": False,
            "source_stems_used": False,
            "human_listening_evidence": False,
            "learned_audio_critic_executed": False,
        },
        "limitations": [
            "Technical metrics detect only measurable artifacts; no flags does not prove "
            "good music or valid musical continuity.",
            "Thresholds are project heuristics and are not perceptually calibrated constants.",
            "Clicks, silence, DC offset, or level changes may be intentional artistic choices; "
            "flags are observations, not aesthetic verdicts.",
            "No learned perceptual critic was executed, so musical_continuity remains unknown.",
            "Metrics describe the rendered stereo waveform only; selector features and source "
            "stems are not inspected.",
        ],
    }


__all__ = [
    "DEFAULT_CONFIG",
    "MIX_QUALITY_SCHEMA",
    "MIX_QUALITY_VERSION",
    "evaluate_mix_quality",
]
