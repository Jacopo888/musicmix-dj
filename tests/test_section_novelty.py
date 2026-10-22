"""Focused tests for beat/downbeat-synchronous acoustic novelty evidence."""

from __future__ import annotations

import json
import unittest

import numpy as np

from musicmix.audio import AudioTrack
from musicmix.section_novelty import (
    FEATURE_CONFIG,
    FEATURE_CONFIG_HASH,
    SECTION_NOVELTY_SCHEMA_VERSION,
    analyze_section_novelty,
    describe_cue_section,
)


SAMPLE_RATE = 8_000
DURATION_SECONDS = 20.0
CHANGE_SECONDS = 10.0


def _stereo(mono: np.ndarray) -> np.ndarray:
    mono = np.asarray(mono, dtype=np.float32)
    return np.repeat(mono[:, None], 2, axis=1)


def _tone(
    frequency: float,
    seconds: float,
    *,
    amplitude: float = 0.2,
    harmonics: tuple[tuple[float, float], ...] = (),
) -> np.ndarray:
    t = np.arange(int(round(seconds * SAMPLE_RATE)), dtype=np.float64) / SAMPLE_RATE
    signal = np.sin(2.0 * np.pi * frequency * t)
    for multiplier, weight in harmonics:
        signal += weight * np.sin(2.0 * np.pi * frequency * multiplier * t)
    return amplitude * signal


def section_track() -> AudioTrack:
    # Frequencies are integer multiples of the frame hop rate (31.25 Hz), so
    # each section is stationary across hops.  A and B differ in pitch class
    # and harmonic shape, isolating chroma and timbre from random variation.
    left = _tone(
        218.75,
        CHANGE_SECONDS,
        harmonics=((2.0, 0.45), (3.0, 0.18)),
    )
    right = _tone(
        343.75,
        DURATION_SECONDS - CHANGE_SECONDS,
        harmonics=((2.0, 0.75), (3.0, 0.50), (4.0, 0.28)),
    )
    mono = np.concatenate([left, right])
    return AudioTrack("section-fixture", "fixture.wav", SAMPLE_RATE, _stereo(mono))


def flat_track() -> AudioTrack:
    # 125 Hz is exactly four cycles per analysis hop at 8 kHz, so every frame
    # descriptor is identical: no acoustic change to discover.
    mono = _tone(125.0, DURATION_SECONDS, amplitude=0.15)
    return AudioTrack("flat-fixture", "flat.wav", SAMPLE_RATE, _stereo(mono))


def beat_grid(step_seconds: float = 0.5) -> list[dict]:
    count = int(DURATION_SECONDS / step_seconds)
    return [
        {"sample": int(round(index * step_seconds * SAMPLE_RATE)), "confidence": None}
        for index in range(count)
    ]


def structure_for(
    track: AudioTrack,
    *,
    beats: list[dict] | None = None,
    downbeats: list[dict] | None = None,
) -> dict:
    return {
        "track_id": track.track_id,
        "sample_rate": track.sample_rate,
        "duration_samples": track.duration_samples,
        "beats": beat_grid() if beats is None else beats,
        "downbeats": (
            beat_grid(2.0) if downbeats is None else downbeats
        ),
    }


def boundary_at(document: dict, sample: int) -> dict:
    for boundary in document["boundaries"]:
        if boundary["sample"] == sample:
            return boundary
    raise AssertionError(f"missing boundary {sample}")


class TestSyntheticSectionChange(unittest.TestCase):
    def test_downbeat_synchronous_shift_is_the_strongest_usable_boundary(self) -> None:
        track = section_track()
        document = analyze_section_novelty(
            track, structure_for(track), neighborhood_bars=3
        )

        self.assertEqual(document["status"], "ok")
        self.assertEqual(document["grid_source"], "downbeats")
        self.assertEqual(document["grid_unit"], "downbeat_interval")
        self.assertEqual(document["scales_units"], [1, 2, 3])
        self.assertEqual(document["feature_config_hash"], FEATURE_CONFIG_HASH)
        self.assertEqual(len(document["feature_config_hash"]), 64)
        self.assertIn("missing", document)
        self.assertNotIn("downbeats", document["missing"])

        boundary = boundary_at(document, int(CHANGE_SECONDS * SAMPLE_RATE))
        self.assertTrue(boundary["usable"])
        self.assertGreater(boundary["novelty_score"], 0.5)
        self.assertLessEqual(boundary["novelty_score"], 1.0)
        for value in boundary["descriptor_family_scores"].values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, FEATURE_CONFIG["family_z_cap"])
        self.assertGreater(
            boundary["novelty_score"],
            max(
                item["novelty_score"]
                for item in document["boundaries"]
                if item["sample"] != boundary["sample"]
            ),
        )
        families = set(boundary["contributing_descriptor_families"])
        self.assertIn("timbre", families)
        self.assertIn("chroma", families)
        self.assertIsNone(boundary["confidence"])
        self.assertIsNone(boundary["calibration"])
        self.assertIsNone(boundary["semantic_label"])

        self.assertEqual(
            document["phrase_or_structural_arrival"]["supported"], False
        )
        self.assertEqual(document["semantic_interpretation"], "not_inferred")
        self.assertIsNone(document["section_labels"])
        self.assertIsNone(document["phrase_length_assumption_bars"])
        self.assertIsNone(document["confidence"])
        self.assertIsNone(document["calibration"])
        self.assertEqual(
            document["usable_boundary_samples"],
            [item["sample"] for item in document["boundaries"]],
        )
        # The document must remain a plain JSON-serialisable artifact.
        json.dumps(document)

    def test_beat_fallback_is_explicit_and_reports_missing_downbeats(self) -> None:
        track = section_track()
        structure = structure_for(track, downbeats=[])
        document = analyze_section_novelty(
            track, structure, neighborhood_bars=4
        )

        self.assertEqual(document["status"], "ok")
        self.assertEqual(document["grid_source"], "beats_fallback_no_usable_downbeats")
        self.assertIn("downbeats", document["missing"])
        self.assertTrue(
            any("no bar or downbeat semantics" in warning for warning in document["warnings"])
        )
        self.assertEqual(document["scales_units"], [1, 2, 4])
        boundary = boundary_at(document, int(CHANGE_SECONDS * SAMPLE_RATE))
        self.assertGreater(boundary["novelty_score"], 0.5)
        self.assertFalse(document["phrase_or_structural_arrival"]["supported"])


class TestFlatAudio(unittest.TestCase):
    def test_flat_track_produces_no_semantic_arrival(self) -> None:
        track = flat_track()
        document = analyze_section_novelty(
            track, structure_for(track), neighborhood_bars=4
        )

        self.assertEqual(document["status"], "ok")
        self.assertTrue(document["boundaries"])
        self.assertLess(
            max(item["novelty_score"] for item in document["boundaries"]), 0.05
        )
        self.assertTrue(
            all(item["semantic_label"] is None for item in document["boundaries"])
        )
        self.assertFalse(document["phrase_or_structural_arrival"]["supported"])
        self.assertEqual(document["phrase_or_structural_arrival"]["status"], "not_supported")
        self.assertIsNone(document["confidence"])
        self.assertIsNone(document["calibration"])

        # A zero-novelty grid point is not described as an acoustic boundary.
        cue = document["usable_boundary_samples"][0]
        described = describe_cue_section(document, cue)
        self.assertFalse(described["boundary_matched"])
        self.assertFalse(described["phrase_or_structural_arrival"]["supported"])
        self.assertIn("usable_boundary_within_tolerance", described["missing"])


class TestMissingContextAndProvenance(unittest.TestCase):
    def test_no_beat_or_downbeat_grid_is_unavailable(self) -> None:
        track = section_track()
        document = analyze_section_novelty(
            track, structure_for(track, beats=[], downbeats=[]), neighborhood_bars=4
        )
        self.assertEqual(document["status"], "unavailable")
        self.assertIn("beats_or_downbeats", document["missing"])
        self.assertEqual(document["usable_boundary_samples"], [])
        self.assertEqual(document["boundaries"], [])
        self.assertFalse(document["phrase_or_structural_arrival"]["supported"])

    def test_short_context_is_reported_not_scored(self) -> None:
        track = section_track()
        short_beats = beat_grid()[:3]
        document = analyze_section_novelty(
            track,
            structure_for(track, beats=short_beats, downbeats=[]),
            neighborhood_bars=4,
        )
        self.assertEqual(document["status"], "insufficient_context")
        self.assertIn("local_context_full_neighborhood", document["missing"])
        self.assertEqual(document["usable_boundary_samples"], [])
        self.assertTrue(
            all(
                "local_context_full_neighborhood" in item["missing"]
                for item in document["insufficient_boundaries"]
            )
        )
        self.assertFalse(document["phrase_or_structural_arrival"]["supported"])

    def test_partial_context_boundaries_are_not_scored(self) -> None:
        track = section_track()
        document = analyze_section_novelty(
            track, structure_for(track), neighborhood_bars=4
        )
        scored = {item["sample"] for item in document["boundaries"]}
        for item in document["insufficient_boundaries"]:
            self.assertNotIn(item["sample"], scored)
            self.assertIn("local_context_full_neighborhood", item["missing"])

    def test_mismatched_or_missing_provenance_fails(self) -> None:
        track = section_track()
        base = structure_for(track)
        cases = {
            "track_id": {**base, "track_id": "other-track"},
            "sample_rate": {**base, "sample_rate": SAMPLE_RATE * 2},
            "duration_samples": {**base, "duration_samples": track.duration_samples + 1},
        }
        for field, payload in cases.items():
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    analyze_section_novelty(track, payload)
        missing_id = dict(base)
        missing_id.pop("track_id")
        with self.assertRaises(ValueError):
            analyze_section_novelty(track, missing_id)
        missing_rate = dict(base)
        missing_rate.pop("sample_rate")
        with self.assertRaises(ValueError):
            analyze_section_novelty(track, missing_rate)


class TestDescribeCueSection(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        track = section_track()
        cls.document = analyze_section_novelty(
            track, structure_for(track), neighborhood_bars=3
        )

    def test_nearby_boundary_is_described_without_arrival_claim(self) -> None:
        cue = int(CHANGE_SECONDS * SAMPLE_RATE)
        result = describe_cue_section(self.document, cue, tolerance_seconds=0.12)

        self.assertTrue(result["boundary_matched"])
        self.assertTrue(result["usable_boundary"])
        self.assertEqual(result["matched_boundary_sample"], cue)
        self.assertEqual(result["distance_samples"], 0)
        self.assertGreater(result["novelty_score"], 0.5)
        self.assertIn("timbre", result["contributing_descriptor_families"])
        self.assertIn("chroma", result["contributing_descriptor_families"])
        self.assertFalse(result["phrase_or_structural_arrival"]["supported"])
        self.assertIsNone(result["confidence"])
        self.assertIsNone(result["calibration"])

    def test_far_cue_has_explicit_missing_boundary(self) -> None:
        cue = int(round(12.5 * SAMPLE_RATE))
        result = describe_cue_section(self.document, cue, tolerance_seconds=0.12)
        self.assertFalse(result["boundary_matched"])
        self.assertIn("usable_boundary_within_tolerance", result["missing"])
        self.assertFalse(result["phrase_or_structural_arrival"]["supported"])

    def test_feature_config_or_schema_mismatch_fails(self) -> None:
        cue = int(CHANGE_SECONDS * SAMPLE_RATE)
        wrong_hash = dict(self.document)
        wrong_hash["feature_config_hash"] = "0" * 64
        with self.assertRaises(ValueError):
            describe_cue_section(wrong_hash, cue)
        wrong_schema = dict(self.document)
        wrong_schema["schema_version"] = SECTION_NOVELTY_SCHEMA_VERSION + ".bogus"
        with self.assertRaises(ValueError):
            describe_cue_section(wrong_schema, cue)


if __name__ == "__main__":
    unittest.main()
