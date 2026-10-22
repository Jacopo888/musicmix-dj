"""Auditable rendering for a small number of transition intents.

The project already has two deliberately simple renderers.  This module gives
callers one guarded entry point around them.  It keeps clocks in source sample
coordinates and output sample coordinates, uses a bounded *local* RMS match,
and records the provenance needed to reproduce an artifact.  It does not
choose cues, infer phrase completion, beat match, or time stretch audio.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .audio import AudioTrack
from .blend_preview import (
    _half_overlap_samples,
    _normalised_tempo_ratio,
    overlap_vocal_safety,
    render_blend_preview,
)
from .contracts import Candidate
from .ingest import sha256_file
from .musical_preview import HANDOFF_VERSION, render_handoff


INTENT_RENDER_VERSION = "intent_render_v2_twenty_second_default"
MIN_CONTEXT_SECONDS = 10.0
DEFAULT_CONTEXT_SECONDS = 10.0
DEFAULT_FADE_SECONDS = 0.012
DEFAULT_OVERLAP_SECONDS = 1.0
DEFAULT_MAX_RELATIVE_TEMPO_ERROR = 0.02
DEFAULT_MAX_ATTENUATION_DB = 6.0
DEFAULT_MAX_BOOST_DB = 3.0
DEFAULT_SILENCE_FLOOR_DBFS = -45.0
DEFAULT_PEAK_CEILING = 0.98
INTENTS = ("handoff", "overlap")
INCOMING_ROLES = ("arrival", "pickup")
TIME_STRETCH_POLICY = "none"
WORDS_POLICY = "source cues are anchors only; no word dropping, no time-stretch, no fabricated beat grid"


class IntentRenderError(ValueError):
    """The requested render cannot be represented safely."""


class BlendNotEligible(IntentRenderError):
    """A requested overlap lacks explicit acoustic and tempo evidence."""


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise IntentRenderError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise IntentRenderError(f"{name} must be finite")
    return number


def _dbfs(value: float) -> float | None:
    # ``None`` keeps returned provenance JSON-safe for digital silence.
    return None if value <= 0.0 else 20.0 * math.log10(value)


def _window_levels(window: np.ndarray) -> dict[str, float]:
    values = np.asarray(window, dtype=np.float64)
    if values.size == 0:
        raise IntentRenderError("analysis window is empty")
    if values.ndim != 2 or not np.isfinite(values).all():
        raise IntentRenderError("analysis window must be finite frames x channels")
    rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
    peak = float(np.max(np.abs(values)))
    return {"rms_dbfs": _dbfs(rms), "peak_dbfs": _dbfs(peak), "rms": rms, "peak": peak}


def _match_levels(
    level_a: float,
    level_b: float,
    *,
    max_attenuation_db: float,
    max_boost_db: float,
    silence_floor_dbfs: float,
) -> dict[str, Any]:
    """Match loudness with a conservative attenuation-first policy.

    A quiet or digital-silence window is evidence that the level estimate is
    unhelpful, not permission to add gain.  For a real mismatch, the louder
    side is attenuated up to the configured bound before any quiet-side boost.
    """

    result: dict[str, Any] = {"status": "skipped", "gain_a_db": 0.0, "gain_b_db": 0.0, "reasons": []}
    if level_a is None or level_b is None or not math.isfinite(level_a) or not math.isfinite(level_b):
        result.update(status="skipped_near_silence", reasons=["one_or_both_windows_digital_silence"])
        return result
    if level_a < silence_floor_dbfs or level_b < silence_floor_dbfs:
        result.update(
            status="skipped_near_silence",
            reasons=[f"level_below_silence_floor({silence_floor_dbfs}dBFS)", "near_silence_is_never_amplified"],
        )
        return result
    difference = abs(level_a - level_b)
    if difference <= 1e-9:
        result["status"] = "already_matched"
        return result
    attenuation = min(difference, max_attenuation_db)
    remaining = difference - attenuation
    boost = min(remaining, max_boost_db)
    if attenuation <= 0.0:
        result["status"] = "skipped_no_attenuation_available"
        return result
    if level_a > level_b:
        gain_a, gain_b = -attenuation, boost
    else:
        gain_a, gain_b = boost, -attenuation
    result.update(
        status="attenuation_only" if boost == 0.0 else "attenuation_plus_bounded_boost",
        gain_a_db=float(gain_a),
        gain_b_db=float(gain_b),
        measured_difference_db=float(difference),
        residual_difference_db=float(max(0.0, difference - attenuation - boost)),
        reasons=["louder_side_attenuated_first", "bounded_gain_change"],
    )
    return result


def _analysis_windows(
    intent: str,
    track_a: AudioTrack,
    track_b: AudioTrack,
    candidate: Candidate,
    context: int,
    half: int,
) -> dict[str, tuple[int, int]]:
    out_cue, in_cue = int(candidate.cue_out_sample), int(candidate.cue_in_sample)
    if intent == "handoff":
        windows = {"a": (out_cue - context, out_cue), "b": (in_cue, in_cue + context)}
    else:
        windows = {"a": (out_cue - context, out_cue + half), "b": (in_cue - half, in_cue + context)}
    if windows["a"][0] < 0 or windows["a"][1] > track_a.duration_samples:
        raise IntentRenderError("Insufficient outgoing context in source A")
    if windows["b"][0] < 0 or windows["b"][1] > track_b.duration_samples:
        raise IntentRenderError("Insufficient incoming context in source B")
    return windows


def _local_track(track: AudioTrack, window: tuple[int, int], gain_db: float) -> AudioTrack:
    """Copy only the source window required by the chosen renderer."""

    start, end = window
    samples = np.asarray(track.samples[start:end], dtype=np.float64)
    if gain_db:
        samples = samples * (10.0 ** (gain_db / 20.0))
    return AudioTrack(track.track_id, track.path, track.sample_rate, samples.astype(np.float32))


def _local_candidate(candidate: Candidate, windows: Mapping[str, tuple[int, int]], intent: str, half: int) -> Candidate:
    a_start, _ = windows["a"]
    b_start, _ = windows["b"]
    return Candidate(
        candidate.candidate_id,
        candidate.track_a,
        candidate.track_b,
        int(candidate.cue_out_sample) - a_start,
        int(candidate.cue_in_sample) - b_start,
        candidate.mode,
        candidate.features,
        candidate.penalties,
        candidate.passes_constraints,
        candidate.constraint_reasons,
    )


def assess_overlap_eligibility(
    candidate: Candidate,
    vocals_a: Any,
    vocals_b: Any,
    rate: int,
    overlap_seconds: float,
    *,
    tempo_a: float | None = None,
    tempo_b: float | None = None,
    max_relative_tempo_error: float = DEFAULT_MAX_RELATIVE_TEMPO_ERROR,
) -> dict[str, Any]:
    """Return a fail-closed overlap eligibility report."""

    if vocals_a is None or vocals_b is None:
        vocal = {"status": "unknown", "reasons": ["vocal_evidence_missing"], "warnings": []}
    else:
        try:
            vocal = overlap_vocal_safety(candidate, vocals_a, vocals_b, int(rate), overlap_seconds)
        except (TypeError, ValueError, KeyError) as exc:
            vocal = {"status": "unknown", "reasons": [f"vocal_evidence_invalid:{exc}"], "warnings": []}

    tempo: dict[str, Any] = {"status": "unknown", "reasons": []}
    if tempo_a is None or tempo_b is None:
        tempo["reasons"].append("tempo_evidence_missing")
    else:
        try:
            a, b = _finite(tempo_a, "tempo_a"), _finite(tempo_b, "tempo_b")
        except IntentRenderError:
            tempo.update(status="invalid", reasons=["tempo_not_finite"])
        else:
            if a <= 0 or b <= 0:
                tempo.update(status="invalid", reasons=["tempo_not_positive"])
            else:
                normalised, convention, relative = _normalised_tempo_ratio(a, b)
                tempo.update(normalized_ratio=normalised, normalization_convention=convention,
                             relative_error=relative, max_relative_tempo_error=float(max_relative_tempo_error))
                if relative > max_relative_tempo_error:
                    tempo.update(status="mismatch", reasons=["tempo_ratio_outside_tolerance"])
                else:
                    tempo["status"] = "acceptable"

    vocal_ok = vocal.get("status") == "supported"
    tempo_ok = tempo.get("status") == "acceptable"
    reasons: list[str] = []
    if not vocal_ok:
        reasons += [f"vocal_safety_{vocal.get('status', 'unknown')}"] + [str(x) for x in vocal.get("reasons", [])]
    if not tempo_ok:
        reasons += [f"tempo_{tempo.get('status', 'unknown')}"] + [str(x) for x in tempo.get("reasons", [])]
    return {
        "eligible": bool(vocal_ok and tempo_ok),
        "vocal_safety": vocal,
        "tempo": tempo,
        "reasons": list(dict.fromkeys(reasons)),
        "warnings": (["Overlap is technical eligibility only; it does not prove musical continuity."]
                      if vocal_ok and tempo_ok else
                      ["Unknown or occupied vocals and missing tempo evidence deny equal-power overlap."]),
    }


def _source_sha(path: str) -> str | None:
    try:
        candidate = Path(path)
        return sha256_file(candidate) if candidate.is_file() else None
    except (OSError, ValueError):
        return None


def _render_once(intent: str, a: AudioTrack, b: AudioTrack, candidate: Candidate, path: Path, *,
                 context_seconds: float, fade_seconds: float, overlap_seconds: float,
                 tempo_a: float | None, tempo_b: float | None, max_relative_tempo_error: float) -> dict[str, Any]:
    if intent == "handoff":
        return render_handoff(a, b, candidate, path, context_seconds=context_seconds, fade_seconds=fade_seconds)
    return render_blend_preview(a, b, candidate, path, context_seconds=context_seconds,
                                overlap_seconds=overlap_seconds, tempo_a=tempo_a, tempo_b=tempo_b,
                                max_relative_tempo_error=max_relative_tempo_error)


def render_intent(
    track_a: AudioTrack,
    track_b: AudioTrack,
    candidate: Candidate,
    path: Path,
    *,
    intent: str = "handoff",
    context_seconds: float = DEFAULT_CONTEXT_SECONDS,
    fade_seconds: float = DEFAULT_FADE_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
    tempo_a: float | None = None,
    tempo_b: float | None = None,
    max_relative_tempo_error: float = DEFAULT_MAX_RELATIVE_TEMPO_ERROR,
    vocals_a: Any = None,
    vocals_b: Any = None,
    incoming_role: str = "arrival",
    pickup_seconds: float = 0.0,
    level_match: bool = True,
    max_attenuation_db: float = DEFAULT_MAX_ATTENUATION_DB,
    max_boost_db: float = DEFAULT_MAX_BOOST_DB,
    silence_floor_dbfs: float = DEFAULT_SILENCE_FLOOR_DBFS,
    peak_ceiling: float = DEFAULT_PEAK_CEILING,
) -> dict[str, Any]:
    """Render one explicit intent and return provenance plus measured output facts."""

    if intent not in INTENTS:
        raise IntentRenderError(f"intent must be one of {INTENTS}")
    if incoming_role not in INCOMING_ROLES:
        raise IntentRenderError(f"incoming_role must be one of {INCOMING_ROLES}")
    context_value = _finite(context_seconds, "context_seconds")
    fade_value = _finite(fade_seconds, "fade_seconds")
    overlap_value = _finite(overlap_seconds, "overlap_seconds")
    pickup_value = _finite(pickup_seconds, "pickup_seconds")
    max_error = _finite(max_relative_tempo_error, "max_relative_tempo_error")
    max_attenuation = _finite(max_attenuation_db, "max_attenuation_db")
    max_boost = _finite(max_boost_db, "max_boost_db")
    silence_floor = _finite(silence_floor_dbfs, "silence_floor_dbfs")
    ceiling = _finite(peak_ceiling, "peak_ceiling")
    if context_value < MIN_CONTEXT_SECONDS:
        raise IntentRenderError(f"context_seconds must be >= {MIN_CONTEXT_SECONDS}s per side")
    if not 0 < fade_value < context_value:
        raise IntentRenderError("Require 0 < fade_seconds < context_seconds")
    if overlap_value <= 0 or max_error < 0 or pickup_value < 0 or max_attenuation < 0 or max_boost < 0:
        raise IntentRenderError("overlap, tolerance, pickup and gain limits must be non-negative")
    if not 0 < ceiling <= 1.0:
        raise IntentRenderError("peak_ceiling must be in (0, 1]")
    if track_a.sample_rate != track_b.sample_rate:
        raise IntentRenderError("Source sample rates must match")
    if track_a.samples.shape[1] != track_b.samples.shape[1]:
        raise IntentRenderError("Source channel counts must match")
    if candidate.track_a != track_a.track_id or candidate.track_b != track_b.track_id:
        raise IntentRenderError("Candidate track identity mismatch: candidate ids must match source track ids")
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")

    rate = int(track_a.sample_rate)
    context = round(context_value * rate)
    half = _half_overlap_samples(overlap_value, rate) if intent == "overlap" else 0
    if intent == "overlap" and half > context:
        raise IntentRenderError("overlap_seconds/2 must not exceed context_seconds")
    windows = _analysis_windows(intent, track_a, track_b, candidate, context, half)
    if incoming_role == "arrival" and pickup_value != 0:
        raise IntentRenderError("arrival role does not accept a non-zero pickup_seconds")
    pickup_samples = round(pickup_value * rate)
    if incoming_role == "pickup" and (pickup_samples <= 0 or pickup_samples >= context):
        raise IntentRenderError("pickup must be positive and shorter than one context window")
    arrival_source = int(candidate.cue_in_sample) + pickup_samples
    if arrival_source >= track_b.duration_samples:
        raise IntentRenderError("pickup arrival falls beyond the incoming source")

    eligibility = None
    if intent == "overlap":
        eligibility = assess_overlap_eligibility(candidate, vocals_a, vocals_b, rate, overlap_value,
                                                  tempo_a=tempo_a, tempo_b=tempo_b,
                                                  max_relative_tempo_error=max_error)
        if not eligibility["eligible"]:
            raise BlendNotEligible("Equal-power overlap denied: " + "; ".join(eligibility["reasons"]))

    levels = {
        side: _window_levels(np.asarray(track.samples[start:end]))
        for side, track, (start, end) in (("a", track_a, windows["a"]), ("b", track_b, windows["b"]))
    }
    match = _match_levels(levels["a"]["rms_dbfs"], levels["b"]["rms_dbfs"],
                          max_attenuation_db=max_attenuation, max_boost_db=max_boost,
                          silence_floor_dbfs=silence_floor) if level_match else {
                              "status": "disabled_by_caller", "gain_a_db": 0.0, "gain_b_db": 0.0, "reasons": []}
    gains = {"a": float(match["gain_a_db"]), "b": float(match["gain_b_db"])}
    # ``render_handoff`` keeps the source cue as a strict interior index.  Give
    # its outgoing local window one sentinel sample after the cue; that is a
    # one-sample bounded copy and preserves the existing renderer's contract.
    render_windows = dict(windows)
    if intent == "handoff":
        render_windows["a"] = (windows["a"][0], windows["a"][1] + 1)
    local_a = _local_track(track_a, render_windows["a"], gains["a"])
    local_b = _local_track(track_b, render_windows["b"], gains["b"])
    local_candidate = _local_candidate(candidate, render_windows, intent, half)

    # Render through the existing implementation; only the bounded source
    # windows are copied.  A second pass is used if the actual PCM peak needs
    # headroom.  It is intentionally global and recorded separately.
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        artifact = _render_once(intent, local_a, local_b, local_candidate, destination,
                                context_seconds=context_value, fade_seconds=fade_value,
                                overlap_seconds=overlap_value, tempo_a=tempo_a, tempo_b=tempo_b,
                                max_relative_tempo_error=max_error)
        raw_peak = float(artifact.get("peak", 0.0))
        if not math.isfinite(raw_peak):
            raise IntentRenderError("Rendered peak is non-finite")
        master_trim_db = 0.0
        trimmed = False
        if raw_peak > ceiling:
            master_trim_db = 20.0 * math.log10(ceiling / raw_peak)
            destination.unlink(missing_ok=True)
            gains["a"] += master_trim_db
            gains["b"] += master_trim_db
            local_a = _local_track(track_a, render_windows["a"], gains["a"])
            local_b = _local_track(track_b, render_windows["b"], gains["b"])
            artifact = _render_once(intent, local_a, local_b, local_candidate, destination,
                                    context_seconds=context_value, fade_seconds=fade_value,
                                    overlap_seconds=overlap_value, tempo_a=tempo_a, tempo_b=tempo_b,
                                    max_relative_tempo_error=max_error)
            trimmed = True
        final_peak = float(artifact.get("peak", 0.0))
        if not math.isfinite(final_peak) or final_peak > ceiling + 1e-8 or int(artifact.get("clipped_samples", 0)):
            raise IntentRenderError("Rendered output exceeds peak ceiling or contains clipping")
    except FileExistsError:
        # A concurrent writer won the exclusive creation race.  Never delete
        # an output that this invocation did not create.
        raise
    except Exception:
        # Do not leave a misleading partial artifact after validation fails.
        destination.unlink(missing_ok=True)
        raise

    def _after_gain(level: float | None, gain: float) -> float | None:
        return None if level is None else level + gain

    resulting = {
        side: {
            "window_start_sample": windows[side][0], "window_end_sample": windows[side][1],
            "rms_dbfs": _after_gain(levels[side]["rms_dbfs"], gains[side]),
            "peak_dbfs": _after_gain(levels[side]["peak_dbfs"], gains[side]),
        }
        for side in ("a", "b")
    }
    join = context
    output_sha = sha256_file(destination)
    source_a_sha = _source_sha(track_a.path)
    source_b_sha = _source_sha(track_b.path)
    sources = {
        "a": {"track_id": track_a.track_id, "path": str(track_a.path), "file_sha256": source_a_sha, "source_sha256": source_a_sha},
        "b": {"track_id": track_b.track_id, "path": str(track_b.path), "file_sha256": source_b_sha, "source_sha256": source_b_sha},
    }
    level_info = {
        "enabled": bool(level_match), "status": match["status"], "method": "source_window_rms_dbfs",
        "attenuation_preferred": True,
        "limits": {"max_attenuation_db": max_attenuation, "max_boost_db": max_boost, "silence_floor_dbfs": silence_floor},
        "original": {side: {"window_start_sample": windows[side][0], "window_end_sample": windows[side][1],
                             "rms_dbfs": levels[side]["rms_dbfs"], "peak_dbfs": levels[side]["peak_dbfs"]} for side in ("a", "b")},
        "gains_db": {"a": gains["a"], "b": gains["b"], "master_trim_db": master_trim_db},
        "resulting": resulting, "reasons": list(match.get("reasons", [])),
        "notes": ["Levels are measured on source windows and are not a musical quality score."],
    }
    artifact.update({
        "intent": intent, "intent_render_version": INTENT_RENDER_VERSION,
        "renderer_version": artifact.get("render_version"), "minimum_context_seconds": MIN_CONTEXT_SECONDS,
        "context_seconds": context_value,
        "clocks": {"rate": rate, "source": {
            "a": {"track_id": track_a.track_id, "cue_sample": int(candidate.cue_out_sample),
                  "window_start_sample": windows["a"][0], "window_end_sample": windows["a"][1],
                  "duration_samples": track_a.duration_samples},
            "b": {"track_id": track_b.track_id, "cue_sample": int(candidate.cue_in_sample),
                  "window_start_sample": windows["b"][0], "window_end_sample": windows["b"][1],
                  "duration_samples": track_b.duration_samples}},
            "output": {"join_sample": join, "join_seconds": join / rate,
                        "a_cue_output_sample": join, "b_cue_output_sample": join,
                        "b_arrival_output_sample": join + pickup_samples}},
        "source_identity": sources,
        "incoming": {"role": incoming_role, "pickup_seconds": pickup_value if incoming_role == "pickup" else 0.0,
                     "pickup_samples": pickup_samples, "cue_source_sample": int(candidate.cue_in_sample),
                     "arrival_source_sample": arrival_source, "cue_output_sample": join,
                     "arrival_output_sample": join + pickup_samples, "time_stretch": TIME_STRETCH_POLICY,
                     "word_handling": WORDS_POLICY},
        "level_matching": level_info,
        "headroom": {"raw_peak": raw_peak, "ceiling": ceiling, "master_trim_db": master_trim_db,
                      "applied_master_trim": trimmed, "final_peak": final_peak,
                      "clipped_samples": int(artifact.get("clipped_samples", 0))},
        "eligibility": eligibility if eligibility is not None else {"eligible": True, "required": "none_for_sequential_handoff",
                       "vocal_safety": {"status": "not_required"}, "tempo": {"status": "not_required"}, "reasons": []},
        "checksum_algorithm": "sha256", "output_sha256": output_sha,
        "policy": {"time_stretch": TIME_STRETCH_POLICY, "words": WORDS_POLICY,
                    "beat_grid": "none_not_fabricated", "new_models": "none"},
    })
    warnings = list(artifact.get("warnings", []))
    warnings.append("Level matching describes source loudness only; it does not certify tension, groove or phrase completion.")
    if match["status"] == "skipped_near_silence":
        warnings.append("Level matching skipped because a window is near the digital-silence floor; the quiet window was never amplified.")
    if trimmed:
        warnings.append(f"Master trim of {master_trim_db:.2f} dB applied to meet the {ceiling} peak ceiling.")
    artifact.update(warnings=warnings, render_version=INTENT_RENDER_VERSION, sha256=output_sha)
    return artifact


__all__ = ["BlendNotEligible", "HANDOFF_VERSION", "INTENTS", "INCOMING_ROLES", "INTENT_RENDER_VERSION",
           "IntentRenderError", "MIN_CONTEXT_SECONDS", "DEFAULT_CONTEXT_SECONDS", "DEFAULT_MAX_ATTENUATION_DB",
           "DEFAULT_MAX_BOOST_DB", "DEFAULT_PEAK_CEILING", "DEFAULT_SILENCE_FLOOR_DBFS",
           "assess_overlap_eligibility", "render_intent"]
