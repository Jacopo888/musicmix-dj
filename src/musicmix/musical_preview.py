"""A reproducible short handoff with coincident source cue anchors.

No overlap: source B's cue starts exactly when source A's cue is reached.
The two short, sequential ramps avoid clicks without introducing the old
crossfade's early incoming beat. This is not time stretching or beat tracking.
"""
from __future__ import annotations

from pathlib import Path
import math
import wave
import numpy as np

from .contracts import Candidate
from .ingest import sha256_file

HANDOFF_VERSION = 'anchor_handoff_v1'


def render_handoff(track_a, track_b, candidate: Candidate, path: Path,
                   context_seconds: float = 10.0, fade_seconds: float = 0.012) -> dict:
    if not math.isfinite(context_seconds) or not math.isfinite(fade_seconds):
        raise ValueError('Durations must be finite')
    if context_seconds <= 0 or not 0 < fade_seconds < context_seconds:
        raise ValueError('Require 0 < fade_seconds < context_seconds')
    if track_a.sample_rate != track_b.sample_rate:
        raise ValueError('Canonical sample rates must match')
    if candidate.track_a != track_a.track_id or candidate.track_b != track_b.track_id:
        raise ValueError('Candidate identity mismatch')
    rate = track_a.sample_rate
    context = round(context_seconds*rate)
    fade = max(2, round(fade_seconds*rate))
    if context <= fade:
        raise ValueError('Context too short for fade')
    out, incoming = candidate.cue_out_sample, candidate.cue_in_sample
    if not context <= out < track_a.duration_samples:
        raise ValueError('Insufficient outgoing context')
    if not 0 <= incoming < track_b.duration_samples or incoming+context > track_b.duration_samples:
        raise ValueError('Insufficient incoming context')
    left = np.array(track_a.samples[out-context:out], dtype=np.float64, copy=True)
    right = np.array(track_b.samples[incoming:incoming+context], dtype=np.float64, copy=True)
    if left.shape[1:] != right.shape[1:]:
        raise ValueError('Channels must match')
    ramp = np.linspace(0.0, 1.0, fade)[:, None]
    left[-fade:] *= ramp[::-1]
    right[:fade] *= ramp
    signal = 0.9*np.concatenate([left, right], axis=0)
    edge = min(round(0.01*rate), len(signal)//2)
    if edge:
        edge_ramp = np.linspace(0.0, 1.0, edge)[:, None]
        signal[:edge] *= edge_ramp
        signal[-edge:] *= edge_ramp[::-1]
    if not np.isfinite(signal).all():
        raise ValueError('Nonfinite output')
    clipped = int(np.count_nonzero(np.abs(signal)>1))
    peak = float(np.max(np.abs(signal)))
    pcm = np.round(np.clip(signal, -1, 1)*32767).astype('<i2')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects artifacts even during concurrent runs.
    with path.open('xb') as raw:
        with wave.open(raw, 'wb') as stream:
            stream.setnchannels(pcm.shape[1]); stream.setsampwidth(2)
            stream.setframerate(rate); stream.writeframes(pcm.tobytes())
    return {'candidate_id': candidate.candidate_id, 'render_version': HANDOFF_VERSION,
            'path': str(path.resolve()), 'sha256': sha256_file(path),
            'duration_samples': len(signal), 'sample_rate': rate, 'peak': peak,
            'clipped_samples': clipped, 'gain': 0.9,
            'context_seconds': context_seconds, 'fade_seconds': fade_seconds,
            'join_sample': context, 'anchor_offset_samples': 0,
            'warnings': ['No tempo/pitch correction; source cues aligned at a sequential handoff.',
                         'No claim that source cue estimates are correct or word-aligned.']}
