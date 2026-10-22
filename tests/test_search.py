"""Tests for deterministic arbitrary-cue retrieval."""
from __future__ import annotations

from dataclasses import dataclass
import unittest

from musicmix.search import search_transitions


@dataclass(frozen=True)
class Cue:
    sample: int
    chroma: tuple[float, ...]
    rms: float
    onset: float
    timbre: tuple[float, ...]
    tempo_bpm: float | None
    confidence: float
    kind: str = "cue"


@dataclass(frozen=True)
class TrackAnalysis:
    track_id: str
    sample_rate: int
    duration_samples: int
    cues: tuple[Cue, ...]
    backend: str = "synthetic"
    diagnostics: dict = None


def cue(sample: int, *, chroma=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        rms=1.0, onset=1.0, timbre=(1.0, 0.0), tempo=120.0, confidence=1.0):
    return Cue(sample, tuple(chroma), rms, onset, tuple(timbre), tempo, confidence)


class SearchTests(unittest.TestCase):
    def test_strong_pair_beats_poor_pair_and_explains_missing_creative(self):
        a = TrackAnalysis("a", 100, 1000, (cue(0),))
        b = TrackAnalysis("b", 100, 1000, (
            cue(0),
            cue(500, chroma=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), rms=0.05, onset=0.05,
                    timbre=(0.0, 1.0), tempo=80.0),
        ))
        result = search_transitions(a, b, top_k=1, min_spacing_seconds=0)
        self.assertEqual(len(result.candidates), 2)
        chosen = next(c for c in result.candidates if c.candidate_id == result.selected[0])
        self.assertEqual(chosen.cue_in_sample, 0)
        reports = {report.candidate_id: report for report in result.reports}
        self.assertIn("vertical", reports[chosen.candidate_id].missing)
        self.assertIn("creative", reports[chosen.candidate_id].missing)
        self.assertIn("creative=unknown_not_measured", reports[chosen.candidate_id].notes)

    def test_deterministic_and_directional_with_different_sample_rates(self):
        a = TrackAnalysis("a", 100, 1000, (cue(0), cue(400)))
        b = TrackAnalysis("b", 200, 2000, (cue(0), cue(800)))
        first = search_transitions(a, b, top_k=3, min_spacing_seconds=0)
        second = search_transitions(a, b, top_k=3, min_spacing_seconds=0)
        self.assertEqual(first, second)
        self.assertEqual(first.diagnostics["sample_rate_b"], 200.0)
        reverse = search_transitions(b, a, top_k=3, min_spacing_seconds=0)
        self.assertNotEqual(first.selected, reverse.selected)

    def test_bound_and_spacing_diversity(self):
        a = TrackAnalysis("a", 10, 1000, tuple(cue(i * 20) for i in range(10)))
        b = TrackAnalysis("b", 10, 1000, tuple(cue(i * 20) for i in range(10)))
        result = search_transitions(a, b, top_k=8, min_spacing_seconds=0, max_candidates=7)
        self.assertLessEqual(len(result.candidates), 7)
        self.assertEqual(len(result.candidates), len(result.reports) + 0)
        self.assertEqual(result.diagnostics["grid_size"], 100)
        self.assertTrue(result.diagnostics["grid_truncated"])
        self.assertLessEqual(len(result.selected), 8)

    def test_missing_tempo_is_measured_as_unknown_penalty(self):
        a = TrackAnalysis("a", 100, 1000, (cue(100, tempo=None),))
        b = TrackAnalysis("b", 200, 2000, (cue(200, tempo=None),))
        result = search_transitions(a, b, top_k=1, min_spacing_seconds=0)
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        report = result.reports[0]
        self.assertNotIn("transform", candidate.penalties)
        self.assertIn("penalty:transform", report.missing)
        self.assertIn("transform=unknown_missing_tempo", report.notes)

    def test_chroma_does_not_assume_an_unrequested_pitch_shift(self):
        """The neutral renderer cannot make C and F-sharp harmonically equal."""
        c = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        f_sharp = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        a = TrackAnalysis("a", 100, 1000, (cue(0, chroma=c),))
        b = TrackAnalysis(
            "b", 100, 1000,
            (cue(0, chroma=c), cue(1, chroma=f_sharp)),
        )
        result = search_transitions(a, b, top_k=2, min_spacing_seconds=0)
        reports = {
            candidate.cue_in_sample: report
            for candidate, report in zip(result.candidates, result.reports)
        }
        self.assertGreater(reports[0].score, reports[1].score)

    def test_duplicate_cues_with_missing_scalar_sort_deterministically(self):
        """Optional malformed scalars must not make cue ordering compare None/float."""
        first = cue(0, rms=1.0)
        second = cue(0, rms=1.0)
        # Keep all fields equal except one optional scalar, as can happen when
        # a backend emits a partially populated descriptor record.
        object.__setattr__(second, "rms", None)
        a = TrackAnalysis("a", 100, 1000, (first, second, cue(1)))
        b = TrackAnalysis("b", 100, 1000, (cue(0),))
        result = search_transitions(a, b, top_k=2, min_spacing_seconds=0)
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(len(result.reports), 2)
        self.assertEqual(len({candidate.candidate_id for candidate in result.candidates}), 2)
        self.assertEqual(len(result.selected), len(set(result.selected)))
        self.assertIn(
            "duplicate_sample",
            {item["reason"] for item in result.diagnostics["skipped_cues_a"]},
        )
        self.assertEqual(
            result.candidates,
            search_transitions(a, b, top_k=2, min_spacing_seconds=0).candidates,
        )

    def test_no_cues_and_invalid_parameters(self):
        empty = TrackAnalysis("a", 100, 1000, ())
        filled = TrackAnalysis("b", 100, 1000, (cue(0),))
        result = search_transitions(empty, filled)
        self.assertEqual(result.candidates, ())
        self.assertTrue(result.diagnostics["no_cues"])
        with self.assertRaises(ValueError):
            search_transitions(empty, filled, top_k=-1)
        with self.assertRaises(ValueError):
            search_transitions(empty, filled, max_candidates=0)
        with self.assertRaises(ValueError):
            search_transitions(empty, empty)


if __name__ == "__main__":
    unittest.main()
