"""Reuse per-beat descriptors while respecting the actual cue boundary.

The source beat grid remains provider-owned and is never modified.  A cue
between two supplied beats is an analysis boundary only: the edge interval
adjacent to the cue is recomputed from the real audio and the other beat
intervals continue to use the cached descriptors.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .analysis import _analysis_frame_length
from .trajectory import (
    DEFAULT_WINDOWS_BEATS,
    TRAJECTORY_CONFIG_HASH,
    _beat_feature_series,
    _coerce_beats,
    _mono_samples,
)


VERSION = "pool-beat-series-v2"
BOUNDARY_METHOD = "nearest_source_beat_with_exact_cue_analysis_boundary"


class PooledTrajectory:
    """Cached beat descriptors for bounded profile extraction.

    ``profile`` keeps its historical signature.  Its ``cue_sample`` plus
    ``cue_shift_samples`` define the actual cue; the nearest source beat is
    retained as an anchor, while the actual cue clips the local edge interval.
    """

    def __init__(self, track, beats):
        self.track_id = track.track_id
        self.rate = int(track.sample_rate)
        self.duration = int(track.duration_samples)
        self.beats = _coerce_beats(beats, self.duration)
        self._mono = _mono_samples(track)
        self._n_fft = _analysis_frame_length(self.rate, self.duration)
        self._hop = max(1, self._n_fft // 4)
        self.series = ()
        if len(self.beats) >= 2:
            self.series = _beat_feature_series(
                self._mono,
                self.beats,
                0,
                len(self.beats) - 1,
                "incoming",
                sample_rate=self.rate,
                n_fft=self._n_fft,
                hop=self._hop,
            )
        self.period = float(np.median(np.diff(self.beats))) if len(self.beats) > 1 else None
        self.config_hash = hashlib.sha256(
            json.dumps(
                {
                    "version": VERSION,
                    "feature_config": TRAJECTORY_CONFIG_HASH,
                    "windows": DEFAULT_WINDOWS_BEATS,
                    "boundary_method": BOUNDARY_METHOD,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def _clipped_point(self, start: int, end: int, *, side: str) -> dict:
        """Compute one edge descriptor through the existing series extractor."""

        if end <= start:
            raise ValueError("clipped edge interval must be positive")
        point = _beat_feature_series(
            self._mono,
            (int(start), int(end)),
            1 if side == "outgoing" else 0,
            1,
            side,
            sample_rate=self.rate,
            n_fft=self._n_fft,
            hop=self._hop,
        )[0]
        return dict(point)

    def profile(self, cue_sample, *, side, cue_shift_samples=0):
        if side not in ("incoming", "outgoing"):
            raise ValueError("Unknown side")
        requested = int(cue_sample)
        shift = int(cue_shift_samples)
        effective = requested + shift
        # Do not let a shifted cue silently clamp to the source edge.  Both
        # requested and effective coordinates are source-clock samples.
        if not 0 <= requested <= self.duration:
            raise ValueError("Cue outside source")
        if not 0 <= effective <= self.duration:
            raise ValueError("Shifted cue outside source")

        anchor = min(
            range(len(self.beats)),
            key=lambda index: abs(self.beats[index] - effective),
        ) if self.beats else None
        anchor_sample = self.beats[anchor] if anchor is not None else None
        anchor_delta = None if anchor_sample is None else int(effective - anchor_sample)
        windows = []
        missing = []
        bounds_diagnostics = []
        clipped_edge_point = None
        clipped_edge_bounds = None

        for count in DEFAULT_WINDOWS_BEATS:
            if anchor is None:
                missing.append(count)
                continue
            count = int(count)
            first = anchor - count if side == "outgoing" else anchor
            last = anchor if side == "outgoing" else anchor + count
            # ``last`` is an exclusive interval index.  There must be count
            # real source-beat intervals; no synthetic beat is added at the
            # track edge.
            if first < 0 or last > len(self.beats) - 1:
                missing.append(count)
                continue

            points = [dict(point) for point in self.series[first:last]]
            if len(points) != count:
                missing.append(count)
                continue

            boundary_method = "source_beat_anchor"
            if effective != anchor_sample:
                if side == "outgoing":
                    edge_start = self.beats[anchor - 1]
                    edge_end = effective
                    clipped_edge_bounds = (edge_start, edge_end)
                    if clipped_edge_point is None:
                        clipped_edge_point = self._clipped_point(edge_start, edge_end, side=side)
                    points[-1] = dict(clipped_edge_point)
                else:
                    edge_start = effective
                    edge_end = self.beats[anchor + 1]
                    clipped_edge_bounds = (edge_start, edge_end)
                    if clipped_edge_point is None:
                        clipped_edge_point = self._clipped_point(edge_start, edge_end, side=side)
                    points[0] = dict(clipped_edge_point)
                boundary_method = "exact_cue_analysis_boundary"

            output_points = [dict(point, beat_index=index) for index, point in enumerate(points)]
            window_start = int(output_points[0]["start_sample"])
            window_end = int(output_points[-1]["end_sample"])
            window = {
                "window_beats": count,
                "beat_feature_series": output_points,
                "start_sample": window_start,
                "end_sample": window_end,
                "duration_samples": window_end - window_start,
                "cue_sample": effective,
                "beat_anchor_sample": anchor_sample,
                "nearest_beat_delta_samples": anchor_delta,
                "boundary_method": boundary_method,
                "chroma_local_context_similarity": None,
                "chroma_tonal_stability_proxy": None,
            }
            windows.append(window)
            bounds_diagnostics.append(
                {
                    "window_beats": count,
                    "start_sample": window_start,
                    "end_sample": window_end,
                    "boundary_method": boundary_method,
                }
            )

        return {
            "track_id": self.track_id,
            "sample_rate": self.rate,
            "canonical_sample_rate": self.rate,
            "duration_samples": self.duration,
            "cue_sample": effective,
            "actual_cue_sample": effective,
            "requested_cue_sample": requested,
            "cue_shift_samples": shift,
            "beat_anchor_sample": anchor_sample,
            "beat_period_samples": self.period,
            "nearest_beat_delta_samples": anchor_delta,
            "beat_anchor_delta_samples": anchor_delta,
            "side": side,
            "backend": VERSION,
            "boundary_method": BOUNDARY_METHOD,
            "trajectory_config_hash": self.config_hash,
            "source_beatgrid": list(self.beats),
            "windows": windows,
            "confidence": None,
            "diagnostics": {
                "missing_windows": missing,
                "confidence_status": "unknown_unvalidated",
                "whole_window_aggregates": "not_computed; continuation policy uses per-beat series",
                "tonal_or_structural_arrival": "unavailable",
                "actual_cue_sample": effective,
                "requested_cue_sample": requested,
                "cue_shift_samples": shift,
                "nearest_beat_delta_samples": anchor_delta,
                "boundary_method": BOUNDARY_METHOD,
                "window_bounds": bounds_diagnostics,
                "clipped_edge_bounds": list(clipped_edge_bounds) if clipped_edge_bounds else None,
                "source_beatgrid_unchanged": True,
                "trajectory_config_hash": self.config_hash,
            },
        }


__all__ = ["BOUNDARY_METHOD", "PooledTrajectory", "VERSION"]
