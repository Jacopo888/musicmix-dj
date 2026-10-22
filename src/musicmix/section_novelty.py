"""Beat/downbeat-synchronous coarse structural-change evidence.

This module measures *acoustic* change between neighbouring beat or downbeat
intervals.  It deliberately does not emit learned section labels, a phrase
length, a resolution claim, or a calibrated probability.  A high novelty value
only says that the timbre, chroma, or log-energy distribution changed more than
is typical for this track at the same scale.  It cannot establish that a
phrase arrived, that a tension resolved, or that a transition is musically
good.

The implementation uses one bounded, batched FFT pass over the waveform and
then aggregates frame descriptors per beat/downbeat interval.  Novelty is
computed at several context scales and robustly standardised with median/MAD
across the candidate boundaries of the same track.  This keeps the output
descriptive and avoids thresholds fitted to the small qualitative holdouts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any

import numpy as np

from .analysis import _analysis_frame_length
from .audio import AudioTrack


SECTION_NOVELTY_VERSION = "musicmix.section_novelty.v1"
SECTION_NOVELTY_SCHEMA_VERSION = "musicmix.section_novelty.v1"

_BAND_COUNT = 10
_MIN_RMS = 1e-8
_EPS = 1e-12
_FRAME_SECONDS = 0.046
_HOP_FRACTION = 0.5
_PITCH_MIN_HZ = 55.0
_PITCH_MAX_HZ = 5_000.0
_HIGH_HZ = 2_000.0
_BASS_HZ = 150.0
_SCALE_CANDIDATES = (1, 2, 4)
_FAMILY_WEIGHTS = {
    "log_energy": 0.20,
    "timbre": 0.45,
    "chroma": 0.35,
}
_FAMILY_NAMES = tuple(_FAMILY_WEIGHTS)
_CONTRIBUTING_Z = 0.5
_MAX_FAMILY_Z = 12.0
# Raw distances are float64 means of O(1) descriptors, so platform-dependent
# rounding (FFT/SIMD summation order) leaves residuals of ~1e-16 even on
# perfectly stationary audio.  Divided by the minimum family scale (>= 0.02)
# that is < 1e-13 in z units; anything below this floor is treated as exactly
# zero.  Meaningful z values are O(0.1..12) (contributing threshold 0.5), so
# real audio is unaffected.
_NUMERIC_Z_FLOOR = 1e-9
_FAMILY_MIN_SCALE = {
    "log_energy": 0.05,
    "timbre": 0.05,
    "chroma": 0.02,
}

FEATURE_CONFIG: dict[str, Any] = {
    "version": SECTION_NOVELTY_VERSION,
    "frame_seconds": _FRAME_SECONDS,
    "hop_fraction_of_frame": _HOP_FRACTION,
    "fft": "analysis.frame_length (next power of two, bounded)",
    "bands": _BAND_COUNT,
    "band_range_hz": [40.0, 12_000.0],
    "pitch_range_hz": [_PITCH_MIN_HZ, _PITCH_MAX_HZ],
    "high_hz": _HIGH_HZ,
    "bass_hz": _BASS_HZ,
    "descriptor_families": {
        "log_energy": "mean log RMS per interval",
        "timbre": "log band shape + centroid + spread + flatness + high/bass ratio",
        "chroma": "12-bin power chroma, L1-normalised per frame",
    },
    "scale_candidates_units": list(_SCALE_CANDIDATES),
    "include_full_neighborhood_scale": True,
    "family_weights": dict(_FAMILY_WEIGHTS),
    "minimum_family_scale": dict(_FAMILY_MIN_SCALE),
    "robust_scaling": "median/MAD per family and scale (1.4826 factor; IQR/std tie fallbacks)",
    "family_z_cap": _MAX_FAMILY_Z,
    "score_map": "z/(1+z), uncalibrated descriptive index",
    "boundary_unit": "downbeat interval when available, else beat interval",
    "semantic_labels": "none",
}
FEATURE_CONFIG_HASH = hashlib.sha256(
    json.dumps(FEATURE_CONFIG, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return result
    raise TypeError("structure must be a mapping or expose to_dict()")


def _bound_structure(track: AudioTrack, structure: Any) -> Mapping[str, Any]:
    payload = _as_mapping(structure)
    declared_id = payload.get("track_id")
    if declared_id is None:
        raise ValueError("structure is missing track_id; source binding is required")
    if str(declared_id) != track.track_id:
        raise ValueError("structure track_id does not match the canonical audio hash")
    for key, expected in (
        ("sample_rate", track.sample_rate),
        ("duration_samples", track.duration_samples),
    ):
        raw = payload.get(key)
        if raw is None:
            raise ValueError(f"structure is missing {key}; source binding is required")
        number = _finite(raw)
        if number is None or int(number) != int(expected):
            raise ValueError(f"structure {key} does not match the AudioTrack")
    return payload


def _event_samples(records: Any, duration_samples: int) -> tuple[int, ...]:
    if records is None:
        return ()
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("structure beat/downbeat events must be a sequence")
    samples: list[int] = []
    for record in records:
        if isinstance(record, Mapping):
            raw = record.get("sample")
        else:
            raw = getattr(record, "sample", None)
        value = _finite(raw)
        if value is None:
            continue
        sample = int(round(value))
        if 0 <= sample < int(duration_samples):
            samples.append(sample)
    return tuple(sorted(set(samples)))


def _scales_for(neighborhood_units: int) -> tuple[int, ...]:
    values = {scale for scale in _SCALE_CANDIDATES if scale <= neighborhood_units}
    values.add(neighborhood_units)
    return tuple(sorted(value for value in values if value >= 1))


def _frame_descriptors(
    mono: np.ndarray,
    sample_rate: int,
    *,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute bounded-batch frame descriptors once for the whole waveform."""

    n_fft = _analysis_frame_length(sample_rate, mono.size)
    if n_fft < 16 or mono.size < n_fft:
        return None
    hop = max(1, int(round(n_fft * _HOP_FRACTION)))
    starts = np.arange(0, mono.size - n_fft + 1, hop, dtype=np.int64)
    if starts.size == 0:
        return None

    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
    nyquist = max(float(sample_rate) / 2.0, 1.0)
    low = max(1.0, min(40.0, nyquist * 0.9))
    high = max(low * 1.01, min(12_000.0, nyquist))
    edges = np.geomspace(low, high, _BAND_COUNT + 1)
    band_masks: list[np.ndarray] = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (frequencies >= left) & (frequencies < right)
        if not np.any(mask):
            mask = np.zeros(frequencies.size, dtype=bool)
            mask[int(np.argmin(np.abs(frequencies - 0.5 * (left + right))))] = True
        band_masks.append(mask)

    pitch_mask = (frequencies >= _PITCH_MIN_HZ) & (
        frequencies <= min(_PITCH_MAX_HZ, nyquist)
    )
    pitch_indices = np.flatnonzero(pitch_mask)
    chroma_mapping = np.zeros((frequencies.size, 12), dtype=np.float64)
    if pitch_indices.size:
        midi = 69.0 + 12.0 * np.log2(
            np.maximum(frequencies[pitch_indices], _EPS) / 440.0
        )
        pitch_classes = np.mod(np.rint(midi).astype(np.int64), 12)
        chroma_mapping[pitch_indices, pitch_classes] = 1.0

    high_mask = frequencies >= min(_HIGH_HZ, nyquist)
    bass_mask = frequencies <= min(_BASS_HZ, nyquist)
    if not np.any(high_mask):
        high_mask = np.zeros(frequencies.size, dtype=bool)
        high_mask[-1] = True
    if not np.any(bass_mask):
        bass_mask = np.zeros(frequencies.size, dtype=bool)
        bass_mask[0] = True

    window = np.hanning(n_fft).astype(np.float64)
    offsets = np.arange(n_fft, dtype=np.int64)
    timbre_width = _BAND_COUNT + 5
    energy = np.empty(starts.size, dtype=np.float64)
    timbre = np.empty((starts.size, timbre_width), dtype=np.float64)
    chroma = np.empty((starts.size, 12), dtype=np.float64)

    for begin in range(0, starts.size, batch_size):
        end = min(starts.size, begin + batch_size)
        frame_starts = starts[begin:end]
        frames = mono[frame_starts[:, None] + offsets[None, :]]
        spectrum = np.abs(
            np.fft.rfft(frames * window[None, :], axis=1)
        )
        power = spectrum * spectrum
        rms = np.sqrt(np.mean(frames * frames, axis=1))
        energy[begin:end] = np.log(np.maximum(rms, _MIN_RMS))

        total = np.sum(power, axis=1) + _EPS
        band_power = np.stack(
            [np.sum(power[:, mask], axis=1) for mask in band_masks], axis=1
        )
        log_band = np.log1p(band_power)
        log_band -= np.mean(log_band, axis=1, keepdims=True)
        centroid = np.sum(power * frequencies[None, :], axis=1) / total
        spread = np.sqrt(
            np.sum(power * (frequencies[None, :] - centroid[:, None]) ** 2, axis=1)
            / total
        )
        flatness = np.exp(np.mean(np.log(power + _EPS), axis=1)) / (
            np.mean(power, axis=1) + _EPS
        )
        high_ratio = np.sum(power[:, high_mask], axis=1) / total
        bass_ratio = np.sum(power[:, bass_mask], axis=1) / total
        timbre[begin:end] = np.column_stack(
            [
                log_band,
                centroid / nyquist,
                spread / nyquist,
                flatness,
                high_ratio,
                bass_ratio,
            ]
        )

        raw_chroma = power @ chroma_mapping
        raw_chroma /= np.maximum(np.sum(raw_chroma, axis=1, keepdims=True), _EPS)
        chroma[begin:end] = raw_chroma

    centres = starts + n_fft // 2
    return centres, energy, timbre, chroma


def _interval_features(
    centres: np.ndarray,
    energy: np.ndarray,
    timbre: np.ndarray,
    chroma: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate frame descriptors into intervals between grid events."""

    interval_count = max(0, grid.size - 1)
    if interval_count == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros((0, timbre.shape[1]), dtype=np.float64),
            np.zeros((0, 12), dtype=np.float64),
            np.zeros(0, dtype=bool),
        )

    indices = np.searchsorted(grid, centres, side="right") - 1
    keep = (indices >= 0) & (indices < interval_count) & (centres < grid[-1])
    if not np.any(keep):
        return (
            np.zeros(interval_count, dtype=np.float64),
            np.zeros((interval_count, timbre.shape[1]), dtype=np.float64),
            np.zeros((interval_count, 12), dtype=np.float64),
            np.zeros(interval_count, dtype=bool),
        )

    indices = indices[keep]
    kept_energy = energy[keep]
    kept_timbre = timbre[keep]
    kept_chroma = chroma[keep]
    counts = np.bincount(indices, minlength=interval_count).astype(np.float64)
    valid = counts > 0

    energy_mean = np.zeros(interval_count, dtype=np.float64)
    timbre_mean = np.zeros((interval_count, timbre.shape[1]), dtype=np.float64)
    chroma_mean = np.zeros((interval_count, 12), dtype=np.float64)

    energy_sum = np.bincount(
        indices, weights=kept_energy, minlength=interval_count
    )
    timbre_sum = np.zeros_like(timbre_mean)
    chroma_sum = np.zeros_like(chroma_mean)
    np.add.at(timbre_sum, indices, kept_timbre)
    np.add.at(chroma_sum, indices, kept_chroma)

    energy_mean[valid] = energy_sum[valid] / counts[valid]
    timbre_mean[valid] = timbre_sum[valid] / counts[valid, None]
    chroma_mean[valid] = chroma_sum[valid] / counts[valid, None]
    chroma_mean[valid] /= np.maximum(
        np.sum(chroma_mean[valid], axis=1, keepdims=True), _EPS
    )
    return energy_mean, timbre_mean, chroma_mean, valid


def _robust_location_scale(values: Sequence[float]) -> tuple[float, float]:
    """Median/MAD scale with documented IQR/std fallbacks for ties.

    Section changes are sparse: most candidate boundaries can have an
    identical (often zero) distance.  MAD then collapses to zero.  Falling
    back to the IQR, then the standard deviation, keeps the resulting z-scores
    finite without inventing a confidence or fitting a threshold to holdouts.
    """

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0, 1.0
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    if mad > _EPS:
        return median, 1.4826 * mad
    iqr = float(np.percentile(array, 75.0) - np.percentile(array, 25.0))
    if iqr > _EPS:
        return median, iqr / 1.349
    spread = float(np.std(array))
    if spread > _EPS:
        return median, spread
    return median, 1.0


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= _EPS:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, 0.0, 1.0))


def _unavailable_document(
    track: AudioTrack,
    *,
    neighborhood_units: int,
    scales: tuple[int, ...],
    missing: Sequence[str],
    warnings: Sequence[str],
    downbeat_count: int,
    beat_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": SECTION_NOVELTY_SCHEMA_VERSION,
        "status": "unavailable",
        "feature_version": SECTION_NOVELTY_VERSION,
        "feature_config": FEATURE_CONFIG,
        "feature_config_hash": FEATURE_CONFIG_HASH,
        "track_id": track.track_id,
        "sample_rate": track.sample_rate,
        "duration_samples": track.duration_samples,
        "grid_source": "none",
        "grid_unit": "none",
        "grid_sample_count": 0,
        "neighborhood_bars": int(neighborhood_units),
        "neighborhood_units": int(neighborhood_units),
        "neighborhood_unit": "none",
        "phrase_length_assumption_bars": None,
        "descriptor_families": list(_FAMILY_NAMES),
        "scales_units": list(scales),
        "downbeat_count": int(downbeat_count),
        "beat_count": int(beat_count),
        "usable_boundary_samples": [],
        "boundaries": [],
        "insufficient_boundaries": [],
        "missing": tuple(dict.fromkeys(missing)),
        "warnings": tuple(dict.fromkeys(warnings)),
        "section_labels": None,
        "labels_source": "none_acoustic_heuristic",
        "semantic_interpretation": "not_inferred",
        "calibration": None,
        "confidence": None,
        "score_semantics": "uncalibrated_heuristic_novelty_index",
        "phrase_or_structural_arrival": {
            "supported": False,
            "status": "not_supported",
            "reason": "acoustic_boundary_heuristic_cannot_establish_resolution_or_closure",
            "provider": "musicmix.section_novelty",
        },
    }


def analyze_section_novelty(
    track: AudioTrack,
    structure: dict,
    *,
    neighborhood_bars: int = 4,
) -> dict:
    """Measure multiscale acoustic change at beat/downbeat boundaries.

    The returned dictionary is descriptive evidence for candidate exploration.
    ``phrase_or_structural_arrival.supported`` is always ``False``: a spectral
    or timbral boundary is not a phrase, cadence, or resolution.  ``confidence``
    and ``calibration`` are always ``None``.
    """

    if not isinstance(track, AudioTrack):
        raise TypeError("analyze_section_novelty expects an AudioTrack")
    if isinstance(neighborhood_bars, bool) or not isinstance(
        neighborhood_bars, int
    ):
        raise TypeError("neighborhood_bars must be an integer")
    if neighborhood_bars < 1:
        raise ValueError("neighborhood_bars must be positive")

    payload = _bound_structure(track, structure)
    duration = track.duration_samples
    beats = _event_samples(payload.get("beats"), duration)
    downbeats = _event_samples(payload.get("downbeats"), duration)
    scales = _scales_for(neighborhood_bars)

    missing: list[str] = []
    warnings: list[str] = []

    if len(downbeats) >= 3:
        grid = np.asarray(downbeats, dtype=np.int64)
        grid_source = "downbeats"
        grid_unit = "downbeat_interval"
    elif len(beats) >= 2:
        grid = np.asarray(beats, dtype=np.int64)
        grid_source = "beats_fallback_no_usable_downbeats"
        grid_unit = "beat_interval"
        if len(downbeats) == 0:
            missing.append("downbeats")
            warnings.append(
                "downbeats missing; using the beat grid; no bar or downbeat "
                "semantics are inferred"
            )
        else:
            missing.append("downbeats_usable_grid")
            warnings.append(
                "downbeats provided fewer than three usable events; using the "
                "beat grid; no bar semantics are inferred"
            )
    else:
        return _unavailable_document(
            track,
            neighborhood_units=neighborhood_bars,
            scales=scales,
            missing=("beats_or_downbeats",),
            warnings=("no usable beat or downbeat grid was supplied",),
            downbeat_count=len(downbeats),
            beat_count=len(beats),
        )

    if grid.size < 2:
        return _unavailable_document(
            track,
            neighborhood_units=neighborhood_bars,
            scales=scales,
            missing=("beats_or_downbeats",),
            warnings=("the supplied grid has fewer than two events",),
            downbeat_count=len(downbeats),
            beat_count=len(beats),
        )

    mono = np.asarray(track.samples, dtype=np.float64).mean(axis=1)
    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
    descriptors = _frame_descriptors(mono, track.sample_rate)
    if descriptors is None:
        return _unavailable_document(
            track,
            neighborhood_units=neighborhood_bars,
            scales=scales,
            missing=("descriptor_frames",),
            warnings=("waveform is too short for the analysis frame",),
            downbeat_count=len(downbeats),
            beat_count=len(beats),
        )

    centres, energy, timbre, chroma = descriptors
    energy_mean, timbre_mean, chroma_mean, interval_valid = _interval_features(
        centres, energy, timbre, chroma, grid
    )
    interval_count = grid.size - 1

    records: list[dict[str, Any]] = []
    insufficient: list[dict[str, Any]] = []
    for boundary_index in range(1, interval_count):
        sample = int(grid[boundary_index])
        boundary_missing: list[str] = []
        if (
            boundary_index < neighborhood_bars
            or interval_count - boundary_index < neighborhood_bars
        ):
            boundary_missing.append("local_context_full_neighborhood")
        if not boundary_missing:
            left_valid = bool(
                np.all(
                    interval_valid[
                        boundary_index - neighborhood_bars : boundary_index
                    ]
                )
            )
            right_valid = bool(
                np.all(
                    interval_valid[
                        boundary_index : boundary_index + neighborhood_bars
                    ]
                )
            )
            if not (left_valid and right_valid):
                boundary_missing.append("descriptor_frames")
        if boundary_missing:
            insufficient.append(
                {
                    "sample": sample,
                    "usable": False,
                    "missing": tuple(boundary_missing),
                }
            )
            continue

        record: dict[str, Any] = {
            "sample": sample,
            "usable": True,
            "scales": {},
        }
        for scale in scales:
            if scale > boundary_index or boundary_index + scale > interval_count:
                continue
            left = slice(boundary_index - scale, boundary_index)
            right = slice(boundary_index, boundary_index + scale)

            energy_left = float(np.mean(energy_mean[left]))
            energy_right = float(np.mean(energy_mean[right]))
            energy_distance = abs(energy_left - energy_right)

            timbre_left = np.mean(timbre_mean[left], axis=0)
            timbre_right = np.mean(timbre_mean[right], axis=0)
            timbre_distance = float(
                np.sqrt(np.mean((timbre_left - timbre_right) ** 2))
            )

            chroma_left = np.mean(chroma_mean[left], axis=0)
            chroma_right = np.mean(chroma_mean[right], axis=0)
            chroma_distance = 1.0 - _cosine(chroma_left, chroma_right)

            recurrence = float(
                np.mean(
                    [
                        _cosine(timbre_left, timbre_right),
                        _cosine(chroma_left, chroma_right),
                        1.0 / (1.0 + energy_distance),
                    ]
                )
            )
            record["scales"][scale] = {
                "raw": {
                    "log_energy": energy_distance,
                    "timbre": timbre_distance,
                    "chroma": chroma_distance,
                },
                "recurrence_similarity": recurrence,
            }
        records.append(record)

    if not records:
        missing.append("local_context_full_neighborhood")
        return {
            "schema_version": SECTION_NOVELTY_SCHEMA_VERSION,
            "status": "insufficient_context",
            "feature_version": SECTION_NOVELTY_VERSION,
            "feature_config": FEATURE_CONFIG,
            "feature_config_hash": FEATURE_CONFIG_HASH,
            "track_id": track.track_id,
            "sample_rate": track.sample_rate,
            "duration_samples": track.duration_samples,
            "grid_source": grid_source,
            "grid_unit": grid_unit,
            "grid_sample_count": int(grid.size),
            "neighborhood_bars": int(neighborhood_bars),
            "neighborhood_units": int(neighborhood_bars),
            "neighborhood_unit": grid_unit,
            "phrase_length_assumption_bars": None,
            "descriptor_families": list(_FAMILY_NAMES),
            "scales_units": list(scales),
            "downbeat_count": len(downbeats),
            "beat_count": len(beats),
            "usable_boundary_samples": [],
            "boundaries": [],
            "insufficient_boundaries": insufficient,
            "missing": tuple(dict.fromkeys(missing)),
            "warnings": tuple(dict.fromkeys(warnings)),
            "section_labels": None,
            "labels_source": "none_acoustic_heuristic",
            "semantic_interpretation": "not_inferred",
            "calibration": None,
            "confidence": None,
            "score_semantics": "uncalibrated_heuristic_novelty_index",
            "phrase_or_structural_arrival": {
                "supported": False,
                "status": "not_supported",
                "reason": "acoustic_boundary_heuristic_cannot_establish_resolution_or_closure",
                "provider": "musicmix.section_novelty",
            },
        }

    scale_stats: dict[int, dict[str, tuple[float, float]]] = {}
    for scale in scales:
        subset = [record for record in records if scale in record["scales"]]
        if not subset:
            continue
        stats: dict[str, tuple[float, float]] = {}
        for family in _FAMILY_NAMES:
            median, raw_scale = _robust_location_scale(
                [
                    record["scales"][scale]["raw"][family]
                    for record in subset
                ]
            )
            stats[family] = (
                median,
                max(raw_scale, _FAMILY_MIN_SCALE[family]),
            )
        scale_stats[scale] = stats

    boundaries: list[dict[str, Any]] = []
    for record in records:
        family_totals = {family: [] for family in _FAMILY_NAMES}
        novelty_values: list[float] = []
        recurrence_values: list[float] = []
        scale_details: list[dict[str, Any]] = []
        for scale in sorted(record["scales"]):
            raw = record["scales"][scale]
            stats = scale_stats.get(scale)
            if stats is None:
                continue
            family_scores: dict[str, float] = {}
            for family in _FAMILY_NAMES:
                median, scale_value = stats[family]
                distance = float(raw["raw"][family])
                z_value = (distance - median) / scale_value
                if not z_value > _NUMERIC_Z_FLOOR:
                    z_value = 0.0
                family_scores[family] = min(_MAX_FAMILY_Z, z_value)
                family_totals[family].append(family_scores[family])
            novelty_z = float(
                sum(
                    _FAMILY_WEIGHTS[family] * family_scores[family]
                    for family in _FAMILY_NAMES
                )
            )
            novelty_values.append(novelty_z)
            recurrence_values.append(float(raw["recurrence_similarity"]))
            scale_details.append(
                {
                    "units_each_side": int(scale),
                    "novelty_z": novelty_z,
                    "family_z": family_scores,
                }
            )

        if not novelty_values:
            insufficient.append(
                {
                    "sample": record["sample"],
                    "usable": False,
                    "missing": ("descriptor_frames",),
                }
            )
            continue

        novelty_z = float(np.mean(novelty_values))
        novelty_score = novelty_z / (1.0 + novelty_z) if novelty_z > 0.0 else 0.0
        family_mean = {
            family: float(np.mean(values)) if values else 0.0
            for family, values in family_totals.items()
        }
        contributing = tuple(
            family
            for family in sorted(
                family_mean, key=lambda name: (-family_mean[name], name)
            )
            if family_mean[family] >= _CONTRIBUTING_Z
        )
        boundaries.append(
            {
                "sample": record["sample"],
                "usable": True,
                "novelty_score": novelty_score,
                "novelty_z": novelty_z,
                "recurrence_similarity": float(np.mean(recurrence_values)),
                "descriptor_family_scores": family_mean,
                "contributing_descriptor_families": contributing,
                "scales": scale_details,
                "calibration": None,
                "confidence": None,
                "semantic_label": None,
            }
        )

    if not boundaries:
        missing.append("local_context_full_neighborhood")
        status = "insufficient_context"
    else:
        status = "ok"

    return {
        "schema_version": SECTION_NOVELTY_SCHEMA_VERSION,
        "status": status,
        "feature_version": SECTION_NOVELTY_VERSION,
        "feature_config": FEATURE_CONFIG,
        "feature_config_hash": FEATURE_CONFIG_HASH,
        "track_id": track.track_id,
        "sample_rate": track.sample_rate,
        "duration_samples": track.duration_samples,
        "grid_source": grid_source,
        "grid_unit": grid_unit,
        "grid_sample_count": int(grid.size),
        "neighborhood_bars": int(neighborhood_bars),
        "neighborhood_units": int(neighborhood_bars),
        "neighborhood_unit": grid_unit,
        "phrase_length_assumption_bars": None,
        "descriptor_families": list(_FAMILY_NAMES),
        "scales_units": list(scales),
        "downbeat_count": len(downbeats),
        "beat_count": len(beats),
        "usable_boundary_samples": [item["sample"] for item in boundaries],
        "boundaries": boundaries,
        "insufficient_boundaries": insufficient,
        "missing": tuple(dict.fromkeys(missing)),
        "warnings": tuple(dict.fromkeys(warnings)),
        "section_labels": None,
        "labels_source": "none_acoustic_heuristic",
        "semantic_interpretation": "not_inferred",
        "calibration": None,
        "confidence": None,
        "score_semantics": "uncalibrated_heuristic_novelty_index",
        "phrase_or_structural_arrival": {
            "supported": False,
            "status": "not_supported",
            "reason": "acoustic_boundary_heuristic_cannot_establish_resolution_or_closure",
            "provider": "musicmix.section_novelty",
        },
    }


def describe_cue_section(
    document: dict,
    cue_sample: int,
    tolerance_seconds: float = 0.12,
) -> dict:
    """Describe the nearest usable acoustic boundary to a cue.

    The result never asserts phrase/structural arrival.  A matched boundary is
    only a heuristic acoustic change whose novelty is above the track's own
    robust scale; it is not evidence of resolution, closure, or a learned
    section label.
    """

    if not isinstance(document, Mapping):
        raise TypeError("document must be a mapping")
    if document.get("schema_version") != SECTION_NOVELTY_SCHEMA_VERSION:
        raise ValueError("section novelty schema version mismatch")
    if document.get("feature_config_hash") != FEATURE_CONFIG_HASH:
        raise ValueError("section novelty feature config mismatch")

    try:
        track_id = str(document["track_id"])
        sample_rate = int(document["sample_rate"])
        duration_samples = int(document["duration_samples"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("document is missing track provenance") from error
    if not track_id:
        raise ValueError("document track_id must be non-empty")
    if sample_rate <= 0 or duration_samples <= 0:
        raise ValueError("document sample_rate/duration_samples are invalid")

    if isinstance(cue_sample, bool) or not isinstance(cue_sample, int):
        raise TypeError("cue_sample must be an integer")
    if cue_sample < 0 or cue_sample >= duration_samples:
        raise ValueError("cue_sample must be inside the track duration")
    try:
        tolerance = float(tolerance_seconds)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("tolerance_seconds must be finite and non-negative") from error
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance_seconds must be finite and non-negative")
    tolerance_samples = int(round(tolerance * sample_rate))

    matched: Mapping[str, Any] | None = None
    best_distance: int | None = None
    boundaries = document.get("boundaries") or ()
    if isinstance(boundaries, Sequence) and not isinstance(boundaries, (str, bytes)):
        for boundary in boundaries:
            if not isinstance(boundary, Mapping):
                continue
            if boundary.get("usable") is not True:
                continue
            novelty = _finite(boundary.get("novelty_score"))
            if novelty is None or novelty <= 0.0:
                # A flat/stationary section has no acoustic-change evidence to
                # describe even when a grid boundary falls within tolerance.
                continue
            value = _finite(boundary.get("sample"))
            if value is None:
                continue
            sample = int(round(value))
            distance = abs(sample - cue_sample)
            if distance <= tolerance_samples and (
                best_distance is None or distance < best_distance
            ):
                matched = boundary
                best_distance = distance

    phrase_field = {
        "supported": False,
        "status": "not_supported",
        "reason": "heuristic_acoustic_boundary_does_not_establish_phrase_resolution_or_closure",
        "provider": "musicmix.section_novelty",
    }
    result: dict[str, Any] = {
        "schema_version": SECTION_NOVELTY_SCHEMA_VERSION,
        "feature_version": SECTION_NOVELTY_VERSION,
        "feature_config_hash": FEATURE_CONFIG_HASH,
        "track_id": track_id,
        "sample_rate": sample_rate,
        "duration_samples": duration_samples,
        "cue_sample": int(cue_sample),
        "tolerance_seconds": tolerance,
        "tolerance_samples": tolerance_samples,
        "calibration": None,
        "confidence": None,
        "semantic_label": None,
        "phrase_or_structural_arrival": phrase_field,
    }

    if matched is None:
        result.update(
            {
                "boundary_matched": False,
                "usable_boundary": False,
                "matched_boundary_sample": None,
                "distance_samples": None,
                "distance_seconds": None,
                "novelty_score": None,
                "novelty_z": None,
                "recurrence_similarity": None,
                "descriptor_family_scores": {},
                "contributing_descriptor_families": (),
                "scales": (),
                "missing": ("usable_boundary_within_tolerance",),
                "reason": "no_usable_heuristic_boundary_within_tolerance",
            }
        )
        phrase_field["reason"] = "no_usable_heuristic_boundary_within_tolerance"
        return result

    assert best_distance is not None
    result.update(
        {
            "boundary_matched": True,
            "usable_boundary": True,
            "matched_boundary_sample": int(matched["sample"]),
            "distance_samples": int(best_distance),
            "distance_seconds": float(best_distance) / float(sample_rate),
            "novelty_score": matched.get("novelty_score"),
            "novelty_z": matched.get("novelty_z"),
            "recurrence_similarity": matched.get("recurrence_similarity"),
            "descriptor_family_scores": dict(
                matched.get("descriptor_family_scores") or {}
            ),
            "contributing_descriptor_families": tuple(
                matched.get("contributing_descriptor_families") or ()
            ),
            "scales": tuple(matched.get("scales") or ()),
            "missing": (),
            "reason": "heuristic_acoustic_boundary_matched",
        }
    )
    return result


__all__ = [
    "FEATURE_CONFIG",
    "FEATURE_CONFIG_HASH",
    "SECTION_NOVELTY_SCHEMA_VERSION",
    "SECTION_NOVELTY_VERSION",
    "analyze_section_novelty",
    "describe_cue_section",
]
