"""Provider-neutral structural event tests; no model weights are loaded here."""

from __future__ import annotations

import time
import unittest

import numpy as np

from musicmix.audio import AudioTrack
from musicmix.structure import (
    HEURISTIC_SECTION_LABEL,
    ProviderPrediction,
    StructureSection,
    TimedEvent,
    analyze_structure,
    heuristic_feature_sections,
)
from tools.analyze_structure import _predict_with_timeout


def fixture_track(duration_seconds: float = 12.0, sample_rate: int = 1000) -> AudioTrack:
    frames = int(duration_seconds * sample_rate)
    signal = np.zeros((frames, 2), dtype=np.float32)
    # Two energy changes give the fallback section detector deterministic
    # material without requiring any provider or model import.
    signal[4 * sample_rate : 6 * sample_rate] = 0.8
    signal[8 * sample_rate : 10 * sample_rate] = 0.2
    return AudioTrack("fixture-track", "fixture.wav", sample_rate, signal)


class TestStructureContracts(unittest.TestCase):
    def test_provider_seconds_convert_to_canonical_samples_and_deduplicate(self) -> None:
        track = fixture_track()

        class Provider:
            def predict(self, samples, sample_rate):
                return ProviderPrediction(
                    beats=(TimedEvent(0.1004, 0.2), TimedEvent(0.10049, 0.8), TimedEvent(1.5)),
                    downbeats=(TimedEvent(2.0, 0.9),),
                    provenance={"provider": "fixture", "model": "none"},
                )

        result = analyze_structure(track, Provider(), include_heuristic_sections=False)
        self.assertEqual(result.status, "ok")
        self.assertEqual([event.sample for event in result.beats], [100, 1500])
        self.assertEqual(result.beats[0].confidence, 0.8)
        self.assertEqual([event.sample for event in result.downbeats], [2000])
        self.assertEqual(result.provenance["provider"], "fixture")
        self.assertEqual(result.to_dict()["beats"][0], {"sample": 100, "confidence": 0.8})

    def test_missing_provider_is_explicit_and_does_not_fabricate_events(self) -> None:
        result = analyze_structure(
            fixture_track(), provider=None, include_heuristic_sections=False
        )
        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.beats, ())
        self.assertEqual(result.downbeats, ())
        self.assertEqual(result.provenance["provider"], "none")
        self.assertTrue(any("No beat/downbeat provider" in warning for warning in result.warnings))

    def test_provider_failure_keeps_empty_events_and_warning(self) -> None:
        class BrokenProvider:
            provenance = {"provider": "fixture", "model": "missing"}

            def predict(self, samples, sample_rate):
                raise RuntimeError("weights unavailable")

        result = analyze_structure(
            fixture_track(), BrokenProvider(), include_heuristic_sections=False
        )
        self.assertEqual(result.status, "unavailable")
        self.assertFalse(result.beats)
        self.assertEqual(result.provenance["model"], "missing")
        self.assertTrue(any("weights unavailable" in warning for warning in result.warnings))

    def test_heuristic_sections_are_explicitly_labelled_and_chronological(self) -> None:
        sections = heuristic_feature_sections(
            fixture_track(40.0, 1000), window_seconds=4.0, hop_seconds=2.0
        )
        self.assertGreaterEqual(len(sections), 1)
        self.assertTrue(all(section.label == HEURISTIC_SECTION_LABEL for section in sections))
        self.assertTrue(all(section.confidence is None for section in sections))
        self.assertEqual(sections[0].start_sample, 0)
        self.assertEqual(sections[-1].end_sample, 40000)
        for previous, current in zip(sections, sections[1:]):
            self.assertLessEqual(previous.end_sample, current.start_sample)

    def test_structure_section_validates_timeline(self) -> None:
        with self.assertRaises(ValueError):
            StructureSection(100, 100, HEURISTIC_SECTION_LABEL)

    def test_provider_timeout_disables_reuse_while_daemon_finishes(self) -> None:
        class SlowProvider:
            def __init__(self) -> None:
                self._disabled = False
                self.calls = 0

            @property
            def disabled(self):
                return self._disabled

            def predict(self, samples, sample_rate):
                self.calls += 1
                time.sleep(0.05)
                return ProviderPrediction()

        provider = SlowProvider()
        with self.assertRaises(TimeoutError):
            _predict_with_timeout(provider, fixture_track(), 0.001)
        self.assertTrue(provider._disabled)
        # A later track must be rejected before another worker is created.
        with self.assertRaises(RuntimeError):
            _predict_with_timeout(provider, fixture_track(), 0.1)
        self.assertEqual(provider.calls, 1)


if __name__ == "__main__":
    unittest.main()
