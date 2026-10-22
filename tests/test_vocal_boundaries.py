from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from musicmix.vocal_boundaries import (
    acoustic_intervals,
    boundary_safety,
    build_boundary_document,
    derive_phrases,
)
from musicmix.audio import AudioTrack


def _wav(path: Path, samples: int = 10_000, rate: int = 1_000) -> Path:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(np.zeros(samples, dtype="<i2").tobytes())
    return path


class TestVocalBoundaries(unittest.TestCase):
    def _document(self, words=(), segments=(), acoustic=(), low_threshold=0.45):
        with tempfile.TemporaryDirectory() as directory:
            path = _wav(Path(directory) / "track.wav")
            return build_boundary_document(
                path, words=words, asr_segments=segments,
                source_kind="test", backend={"status": "complete"},
                separator={"status": "test"}, acoustic=acoustic,
                low_confidence_threshold=low_threshold,
            )

    def test_boundary_whole_fade(self):
        document = self._document(
            words=[{"start_sample": 4_000, "end_sample": 6_800, "text": "out", "confidence": 0.95},
                   {"start_sample": 2_000, "end_sample": 3_000, "text": "in", "confidence": 0.95}],
            segments=[{"start_sample": 0, "end_sample": 10_000, "text": "context", "confidence": 0.95}],
        )
        outgoing = boundary_safety(
            document, side="A_out", cue_sample=8_000, fade_samples=1_000,
            word_span=document["words"][1], alignment_guard_samples=100,
        )
        incoming = boundary_safety(
            document, side="B_in", cue_sample=900, fade_samples=1_000,
            word_span=document["words"][0], alignment_guard_samples=100,
        )
        self.assertEqual(outgoing["status"], "supported")
        self.assertEqual(incoming["status"], "supported")

    def test_fade_inside_word_is_unsafe(self):
        document = self._document(
            words=[{"start_sample": 4_000, "end_sample": 7_950, "text": "out", "confidence": 0.95}],
            segments=[{"start_sample": 4_000, "end_sample": 7_950, "text": "out", "confidence": 0.95}],
        )
        result = boundary_safety(
            document, side="A_out", cue_sample=8_000, fade_samples=1_000,
            word_span=document["words"][0],
        )
        self.assertEqual(result["status"], "unsafe")
        self.assertIn("outgoing_word_overlaps_fade_or_cut", result["reasons"])

    def test_low_confidence_returns_unknown(self):
        document = self._document(
            words=[{"start_sample": 4_000, "end_sample": 5_000, "text": "uncertain", "confidence": 0.2}],
            segments=[{"start_sample": 4_000, "end_sample": 5_000, "text": "uncertain", "confidence": 0.2}],
        )
        result = boundary_safety(
            document, side="A_out", cue_sample=8_000, fade_samples=1_000,
            word_span=document["words"][0],
        )
        self.assertEqual(result["status"], "unknown")
        self.assertTrue(any("uncertain" in reason or "word_interval" in reason for reason in result["reasons"]))

    def test_phrase_ends_follow_word_ends(self):
        phrases = derive_phrases(
            [
                {"start_sample": 100, "end_sample": 200, "text": "one", "confidence": 0.9},
                {"start_sample": 300, "end_sample": 400, "text": "two", "confidence": 0.8},
                {"start_sample": 1_200, "end_sample": 1_300, "text": "three", "confidence": 0.7},
            ],
            duration_samples=2_000,
            sample_rate=1_000,
        )
        self.assertEqual([(p["start_sample"], p["end_sample"]) for p in phrases], [(100, 400), (1_200, 1_300)])
        self.assertTrue(all(p["boundary_source"] == "asr_word_gap_heuristic" for p in phrases))

    def test_interval_coverage_is_partition_and_gaps_are_unknown(self):
        document = self._document(
            segments=[{"start_sample": 200, "end_sample": 400, "text": "observed", "confidence": 0.9}],
        )
        intervals = document["coverage_intervals"]
        self.assertEqual(intervals[0]["start_sample"], 0)
        self.assertEqual(intervals[-1]["end_sample"], 10_000)
        self.assertEqual(sum(item["end_sample"] - item["start_sample"] for item in intervals), 10_000)
        self.assertTrue(any(item["status"] == "unknown_no_asr" for item in intervals))
        self.assertTrue(any(item["reason"] == "missing_asr_not_silence" for item in document["uncertain_regions"]))

    def test_measured_stem_silence_can_support_wordless_boundary(self):
        document = self._document(
            acoustic=[{"start_sample": 0, "end_sample": 10_000, "status": "acoustic_silence", "evidence": "test"}],
        )
        result = boundary_safety(document, side="A_out", cue_sample=5_000, fade_samples=1_000)
        self.assertEqual(result["status"], "supported")
        self.assertIn("measured_separated_vocal_low_energy", result["reasons"])

    def test_partial_silence_does_not_support_whole_fade(self):
        document = self._document(
            acoustic=[{"start_sample": 4_500, "end_sample": 5_000, "status": "acoustic_silence", "evidence": "test"}],
        )
        result = boundary_safety(document, side="A_out", cue_sample=5_000, fade_samples=1_000)
        self.assertEqual(result["status"], "unknown")

    def test_segment_spanning_a_measured_pause_does_not_make_it_conflicting(self):
        document = self._document(
            words=[{"start_sample": 1000, "end_sample": 2000, "text": "one", "confidence": 0.9},
                   {"start_sample": 6000, "end_sample": 7000, "text": "two", "confidence": 0.9}],
            segments=[{"start_sample": 1000, "end_sample": 7000, "text": "one two", "confidence": 0.9}],
            acoustic=[{"start_sample": 3000, "end_sample": 5000, "status": "acoustic_silence"}],
        )
        self.assertEqual(boundary_safety(document, side="A_out", cue_sample=5000,
                                         fade_samples=1000)["status"], "supported")

    def test_measured_quiet_overlapping_an_actual_word_remains_unsafe(self):
        document = self._document(
            words=[{"start_sample": 4000, "end_sample": 5000, "text": "word", "confidence": 0.9}],
            segments=[{"start_sample": 1000, "end_sample": 7000, "text": "word", "confidence": 0.9}],
            acoustic=[{"start_sample": 3000, "end_sample": 6000, "status": "acoustic_silence"}],
        )
        self.assertEqual(boundary_safety(document, side="A_out", cue_sample=5000,
                                         fade_samples=1000)["status"], "unsafe")

    def test_segment_pause_is_not_an_acoustic_conflict_without_a_word(self):
        document = self._document(
            words=[{"start_sample": 1000, "end_sample": 2000, "text": "one", "confidence": 0.9},
                   {"start_sample": 6000, "end_sample": 7000, "text": "two", "confidence": 0.9}],
            segments=[{"start_sample": 1000, "end_sample": 7000, "text": "one two", "confidence": 0.9}],
            acoustic=[{"start_sample": 3000, "end_sample": 5000, "status": "acoustic_silence"}],
        )
        pause = [item for item in document["coverage_intervals"]
                 if item["start_sample"] == 3000 and item["end_sample"] == 5000]
        self.assertEqual(len(pause), 1)
        self.assertEqual(pause[0]["status"], "acoustic_silence_supported")
        self.assertFalse(any(item["status"] == "unknown_acoustic_conflict" for item in pause))

    def test_acoustic_silence_over_timestamped_word_is_a_conflict(self):
        document = self._document(
            words=[{"start_sample": 3000, "end_sample": 5000, "text": "word", "confidence": 0.9}],
            segments=[{"start_sample": 1000, "end_sample": 7000, "text": "word", "confidence": 0.9}],
            acoustic=[{"start_sample": 3000, "end_sample": 5000, "status": "acoustic_silence"}],
        )
        conflict = [item for item in document["coverage_intervals"]
                    if item["start_sample"] == 3000 and item["end_sample"] == 5000]
        self.assertEqual(len(conflict), 1)
        self.assertEqual(conflict[0]["status"], "unknown_acoustic_conflict")

    def test_missing_word_confidence_is_unknown(self):
        document = self._document(
            words=[{"start_sample": 4_000, "end_sample": 5_000, "text": "unscored"}],
            segments=[{"start_sample": 0, "end_sample": 10_000, "text": "context", "confidence": 0.9}],
        )
        result = boundary_safety(
            document, side="A_out", cue_sample=7_000, fade_samples=1_000,
            word_span=document["words"][0],
        )
        self.assertEqual(result["status"], "unknown")
        self.assertIn("word_confidence_missing", result["reasons"])

    def test_invalid_spans_are_rejected(self):
        with self.assertRaises(ValueError):
            self._document(words=[{"start_sample": 9_000, "end_sample": 10_001, "text": "bad"}])
        document = self._document()
        with self.assertRaises(ValueError):
            boundary_safety(document, side="B_in", cue_sample=1_000, fade_samples=100,
                            word_span={"start_sample": 100, "end_sample": 10_001})

    def test_acoustic_intervals_are_non_overlapping_hop_cells(self):
        samples = np.zeros((1_000, 2), dtype=np.float32)
        samples[200:800] = 0.2
        track = AudioTrack("fixture", "fixture.wav", 1_000, samples)
        intervals, diagnostics = acoustic_intervals(track, frame_seconds=0.2, hop_seconds=0.1)
        self.assertEqual(intervals[0]["start_sample"], 0)
        self.assertEqual(intervals[-1]["end_sample"], 1_000)
        for previous, current in zip(intervals, intervals[1:]):
            self.assertLessEqual(previous["end_sample"], current["start_sample"])
        self.assertIn("threshold_capped_at_p90_fraction", diagnostics)

    def test_quiet_voiced_region_is_not_promoted_to_silence(self):
        samples = np.zeros((1_000, 2), dtype=np.float32)
        samples[200:400] = 0.2
        samples[600:800] = 0.02
        track = AudioTrack("fixture", "fixture.wav", 1_000, samples)
        intervals, diagnostics = acoustic_intervals(track, frame_seconds=0.2, hop_seconds=0.1)
        soft_region = [
            item for item in intervals
            if item["start_sample"] < 800 and item["end_sample"] > 600
        ]
        self.assertTrue(soft_region)
        self.assertFalse(any(item["status"] == "acoustic_silence" for item in soft_region))
        self.assertLessEqual(diagnostics["threshold_rms"], 0.003)


if __name__ == "__main__":
    unittest.main()
