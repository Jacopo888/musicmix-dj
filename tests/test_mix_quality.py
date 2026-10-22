"""Offline tests for the technical-only mix-quality seam report."""

from __future__ import annotations

import json
import math
import unittest

import numpy as np

from musicmix.audio import AudioTrack
from musicmix.mix_quality import (
    MIX_QUALITY_SCHEMA,
    MIX_QUALITY_VERSION,
    evaluate_mix_quality,
)


SAMPLE_RATE = 8000


def _track(samples: np.ndarray, *, sample_rate: int = SAMPLE_RATE, track_id: str = "fixture") -> AudioTrack:
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 1:
        array = np.repeat(array[:, None], 2, axis=1)
    return AudioTrack(track_id, f"{track_id}.wav", sample_rate, array)


def _sine(frames: int, *, amplitude: float = 0.15, frequency: float = 440.0) -> np.ndarray:
    time = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
    return amplitude * np.sin(2.0 * math.pi * frequency * time)


def _codes(report: dict) -> set[str]:
    return {flag["code"] for flag in report["flags"]}


class TestMixQualityReportContract(unittest.TestCase):
    def test_report_is_explicitly_technical_and_not_a_quality_score(self) -> None:
        track = _track(_sine(SAMPLE_RATE * 2))
        report = evaluate_mix_quality(track, SAMPLE_RATE)
        self.assertEqual(report["schema_version"], MIX_QUALITY_SCHEMA)
        self.assertEqual(report["version"], MIX_QUALITY_VERSION)
        self.assertEqual(report["status"], "technical_only")
        self.assertEqual(report["musical_continuity"], "unknown_without_validated_critic")
        self.assertIsNone(report["musical_quality"])
        self.assertIsNone(report["confidence"])
        self.assertNotIn("score", report)
        self.assertEqual(report["track"]["track_id"], "fixture")
        self.assertEqual(report["track"]["sample_rate"], SAMPLE_RATE)
        self.assertEqual(report["track"]["duration_samples"], SAMPLE_RATE * 2)
        self.assertEqual(len(report["track"]["samples_sha256"]), 64)
        self.assertEqual(len(report["config_sha256"]), 64)
        self.assertEqual(report["join"]["sample"], SAMPLE_RATE)
        self.assertFalse(report["evidence"]["learned_audio_critic_executed"])

    def test_report_is_deterministic_and_json_serializable(self) -> None:
        track = _track(_sine(SAMPLE_RATE))
        first = evaluate_mix_quality(track, SAMPLE_RATE // 2)
        second = evaluate_mix_quality(track, SAMPLE_RATE // 2)
        self.assertEqual(first, second)
        self.assertEqual(first["track"]["samples_sha256"], second["track"]["samples_sha256"])
        self.assertEqual(first["config_sha256"], second["config_sha256"])
        encoded = json.dumps(first, allow_nan=False)
        self.assertIn("technical_only", encoded)

    def test_samples_hash_binds_content_and_sample_rate(self) -> None:
        samples = _sine(SAMPLE_RATE)
        base = evaluate_mix_quality(_track(samples), SAMPLE_RATE // 2)
        other_rate = evaluate_mix_quality(_track(samples, sample_rate=16000), SAMPLE_RATE // 2)
        changed = samples.copy()
        changed[0] += 0.01
        other_content = evaluate_mix_quality(_track(changed), SAMPLE_RATE // 2)
        self.assertNotEqual(base["track"]["samples_sha256"], other_rate["track"]["samples_sha256"])
        self.assertNotEqual(base["track"]["samples_sha256"], other_content["track"]["samples_sha256"])


class TestArtifactDetection(unittest.TestCase):
    def test_injected_click_increases_relative_discontinuity(self) -> None:
        join = SAMPLE_RATE
        clean = _sine(SAMPLE_RATE * 2)
        baseline = evaluate_mix_quality(_track(clean), join)
        corrupted = clean.copy()
        corrupted[join] += 0.8
        clicked = evaluate_mix_quality(_track(corrupted), join)

        baseline_ratio = baseline["metrics"]["join_jump"]["ratio_to_typical_derivative"]
        clicked_ratio = clicked["metrics"]["join_jump"]["ratio_to_typical_derivative"]
        self.assertIsNotNone(baseline_ratio)
        self.assertIsNotNone(clicked_ratio)
        self.assertGreater(clicked_ratio, baseline_ratio * 5.0)
        self.assertIn("join_discontinuity_high", _codes(clicked))
        self.assertNotIn("join_discontinuity_high", _codes(baseline))

    def test_clipped_samples_are_counted_and_flagged(self) -> None:
        samples = np.repeat(_sine(SAMPLE_RATE)[:, None], 2, axis=1)
        samples[100:103, 0] = 1.0
        report = evaluate_mix_quality(_track(samples), SAMPLE_RATE // 2)
        self.assertEqual(report["metrics"]["clipping"]["clipped_sample_count"], 3)
        self.assertGreater(report["metrics"]["clipping"]["clipped_sample_fraction"], 0.0)
        self.assertIn("clipping_detected", _codes(report))

    def test_quiet_valid_fixture_does_not_invent_certainty(self) -> None:
        track = _track(_sine(SAMPLE_RATE * 2, amplitude=0.1))
        report = evaluate_mix_quality(track, SAMPLE_RATE)
        self.assertEqual(report["flags"], [])
        self.assertEqual(report["metrics"]["join_jump"]["availability"], "available")
        self.assertAlmostEqual(
            report["metrics"]["join_jump"]["ratio_to_typical_derivative"], 1.0, delta=0.5
        )
        self.assertIsNone(report["confidence"])
        self.assertEqual(report["musical_continuity"], "unknown_without_validated_critic")

    def test_seam_silence_is_detected_but_not_called_a_defect(self) -> None:
        join = SAMPLE_RATE // 2
        samples = np.zeros(SAMPLE_RATE * 2)
        samples[SAMPLE_RATE // 4 : join - SAMPLE_RATE // 10] = _sine(
            join - SAMPLE_RATE // 4 - SAMPLE_RATE // 10, amplitude=0.2
        )
        samples[join + SAMPLE_RATE // 10 : SAMPLE_RATE] = _sine(
            SAMPLE_RATE - join - SAMPLE_RATE // 10, amplitude=0.2
        )
        report = evaluate_mix_quality(_track(samples), join)
        self.assertIn("seam_silence_detected", _codes(report))
        severity = {
            flag["code"]: flag["severity"] for flag in report["flags"]
        }["seam_silence_detected"]
        self.assertEqual(severity, "heuristic_observation")

    def test_large_level_change_is_descriptive_only(self) -> None:
        samples = np.zeros(SAMPLE_RATE * 2)
        samples[:SAMPLE_RATE] = _sine(SAMPLE_RATE, amplitude=0.2)
        samples[SAMPLE_RATE:] = _sine(SAMPLE_RATE, amplitude=0.002)
        report = evaluate_mix_quality(_track(samples), SAMPLE_RATE)
        self.assertIn("large_level_change_descriptive", _codes(report))
        severity = {
            flag["code"]: flag["severity"] for flag in report["flags"]
        }["large_level_change_descriptive"]
        self.assertEqual(severity, "descriptive_only")


class TestUndefinedAndEdgeEvidence(unittest.TestCase):
    def test_relative_jump_is_undefined_in_silence(self) -> None:
        samples = np.zeros(SAMPLE_RATE)
        join = SAMPLE_RATE // 2
        samples[join] = 0.5
        report = evaluate_mix_quality(_track(samples), join)
        jump = report["metrics"]["join_jump"]
        self.assertAlmostEqual(jump["absolute"], 0.5)
        self.assertIsNone(jump["ratio_to_typical_derivative"])
        self.assertEqual(jump["availability"], "undefined_in_silence")
        self.assertIn("join_discontinuity_undefined", _codes(report))
        self.assertIsNone(report["confidence"])

    def test_join_at_track_start_has_no_preceding_sample(self) -> None:
        report = evaluate_mix_quality(_track(_sine(SAMPLE_RATE)), 0)
        jump = report["metrics"]["join_jump"]
        self.assertIsNone(jump["absolute"])
        self.assertIsNone(jump["ratio_to_typical_derivative"])
        self.assertEqual(jump["absolute_availability"], "no_preceding_sample")
        self.assertEqual(jump["availability"], "no_preceding_sample")
        self.assertIn("join_discontinuity_undefined", _codes(report))

    def test_join_at_track_end_has_no_following_sample(self) -> None:
        frames = SAMPLE_RATE
        report = evaluate_mix_quality(_track(_sine(frames)), frames)
        jump = report["metrics"]["join_jump"]
        self.assertIsNone(jump["absolute"])
        self.assertEqual(jump["absolute_availability"], "no_following_sample")
        self.assertIn("join_discontinuity_undefined", _codes(report))

    def test_context_is_truncated_at_track_edges_instead_of_imputed(self) -> None:
        frames = 100
        report = evaluate_mix_quality(_track(_sine(frames)), 50, context_seconds=1.0)
        self.assertEqual(report["join"]["pre_samples"], 50)
        self.assertEqual(report["join"]["post_samples"], 50)
        self.assertEqual(report["join"]["context_samples"], SAMPLE_RATE)
        self.assertEqual(report["metrics"]["join_jump"]["availability"], "available")

    def test_empty_track_reports_missing_evidence(self) -> None:
        report = evaluate_mix_quality(_track(np.zeros((0, 2))), 0)
        self.assertIn("empty_track", _codes(report))
        self.assertEqual(report["metrics"]["join_jump"]["availability"], "no_samples")
        self.assertEqual(report["metrics"]["sample_peak"]["availability"], "no_samples")
        self.assertIsNone(report["metrics"]["sample_peak"]["value"])
        self.assertIsNone(report["musical_quality"])

    def test_invalid_context_and_join_are_rejected(self) -> None:
        track = _track(_sine(SAMPLE_RATE))
        for context in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                evaluate_mix_quality(track, SAMPLE_RATE // 2, context_seconds=context)
        for join in (-1, SAMPLE_RATE + 1):
            with self.assertRaises(ValueError):
                evaluate_mix_quality(track, join)


if __name__ == "__main__":
    unittest.main()
