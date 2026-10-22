"""Fixed neutral preview policy: no tempo/pitch correction or learned effects."""
from __future__ import annotations

import math
from pathlib import Path
import wave

import numpy as np

from .contracts import Candidate
from .ingest import sha256_file

RENDER_VERSION = 'neutral_linear_v1'


def render_preview(track_a, track_b, candidate: Candidate, path: Path,
                   context_seconds: float = 2.0, fade_seconds: float = 0.5) -> dict:
    if not math.isfinite(context_seconds) or not math.isfinite(fade_seconds):
        raise ValueError('Preview durations must be finite')
    if context_seconds <= 0 or not 0 < fade_seconds <= context_seconds:
        raise ValueError('Require 0 < fade_seconds <= context_seconds')
    if track_a.sample_rate != track_b.sample_rate:
        raise ValueError('Preview requires canonical audio with identical sample rates')
    if candidate.track_a != track_a.track_id or candidate.track_b != track_b.track_id:
        raise ValueError('Candidate track identity mismatch')
    rate = track_a.sample_rate
    context = round(context_seconds * rate)
    fade = max(2, round(fade_seconds * rate))
    if context < fade:
        raise ValueError('Preview is shorter than the fade after sample rounding')
    out, incoming = candidate.cue_out_sample, candidate.cue_in_sample
    # A exits exactly at cue_out; B begins exactly at cue_in. Overlap is
    # the final fade interval of A and the first fade interval of B.
    if out < context or incoming < 0 or incoming + context > track_b.duration_samples:
        raise ValueError('Insufficient context for fixed preview policy')
    if out >= track_a.duration_samples:
        raise ValueError('Cue out beyond track duration')
    left = np.asarray(track_a.samples[out-context:out], dtype=np.float64).copy()
    right = np.asarray(track_b.samples[incoming:incoming+context], dtype=np.float64).copy()
    if left.shape[1:] != right.shape[1:]:
        raise ValueError('Channel mismatch')
    weights = np.linspace(0.0, 1.0, fade)[:, None]
    overlap = left[-fade:] * (1.0-weights) + right[:fade] * weights
    signal = np.concatenate([left[:-fade], overlap, right[fade:]], axis=0)
    # Fixed headroom for every system/candidate, without per-preview normalization.
    signal *= 0.90
    edge = min(round(0.01 * rate), len(signal)//2)
    if edge:
        ramp = np.linspace(0.0, 1.0, edge)[:, None]
        signal[:edge] *= ramp
        signal[-edge:] *= ramp[::-1]
    if not np.isfinite(signal).all():
        raise ValueError('Nonfinite audio samples')
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    pcm = np.round(np.clip(signal, -1, 1) * 32767).astype('<i2')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f'Preview already exists: {path}')
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(pcm.shape[1]); stream.setsampwidth(2)
        stream.setframerate(rate); stream.writeframes(pcm.tobytes())
    return {'candidate_id': candidate.candidate_id, 'render_version': RENDER_VERSION,
            'path': str(path.resolve()), 'sha256': sha256_file(path), 'peak': peak,
            'clipped_samples': int(np.count_nonzero(np.abs(signal) > 1)),
            'duration_samples': len(signal), 'sample_rate': rate,
            'context_seconds': context_seconds, 'fade_seconds': fade_seconds,
            'gain': 0.9, 'warnings': ['No beat alignment, pitch correction, or loudness matching.']}
