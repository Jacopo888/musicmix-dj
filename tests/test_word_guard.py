"""Tests for deterministic, confidence-gated word-boundary refinement."""
from __future__ import annotations

import math
import unittest

from musicmix.contracts import Candidate
from musicmix.word_guard import (
    NoSafeBoundaryError,
    WordSpan,
    plan_outgoing_word_boundary,
    plan_word_boundary,
)


RATE = 100


def candidate(out: int = 160, incoming: int = 130) -> Candidate:
    return Candidate(
        candidate_id="cand-original",
        track_a="track-a",
        track_b="track-b",
        cue_out_sample=out,
        cue_in_sample=incoming,
        mode="harmony_rhythm",
        features={"horizontal": object()},
        penalties={},
    )


def spans(*items: tuple[int, int, float]) -> tuple[WordSpan, ...]:
    return tuple(WordSpan(*item) for item in items)


class WordGuardTests(unittest.TestCase):
    def test_minimal_shift_and_no_word_intersection(self):
        # A gap [100, 200) and B gap [100, 180), after a 2-sample margin.
        result = plan_word_boundary(
            candidate(),
            spans((0, 100, 0.9), (200, 300, 0.9)),
            spans((0, 100, 0.9), (180, 300, 0.9)),
            RATE,
            target_fade_seconds=0.5,
            margin_seconds=0.02,
        )
        self.assertEqual(result.cue_out_shift_samples, 0)
        self.assertEqual(result.cue_in_shift_samples, -2)
        self.assertAlmostEqual(result.fade_seconds, 0.5)
        self.assertEqual(result.revised.track_a, "track-a")
        self.assertEqual(result.revised.track_b, "track-b")
        self.assertIn("score_stale_recompute_required", result.notes)
        for words, start, end, outgoing in [
            (spans((0, 100, 0.9), (200, 300, 0.9)), result.revised.cue_out_sample - 50,
             result.revised.cue_out_sample, True),
            (spans((0, 100, 0.9), (180, 300, 0.9)), result.revised.cue_in_sample,
             result.revised.cue_in_sample + 50, False),
        ]:
            for word in words:
                self.assertTrue(end <= word.start_sample or start >= word.end_sample)

    def test_overlapping_spans_are_merged_and_not_used_as_gap(self):
        result = plan_word_boundary(
            candidate(out=160, incoming=160),
            spans((0, 100, 0.95), (80, 120, 0.9), (200, 300, 0.95)),
            spans((0, 100, 0.95), (200, 300, 0.95)),
            RATE,
            target_fade_seconds=0.4,
            margin_seconds=0.0,
        )
        self.assertEqual(result.revised.cue_out_sample, 160)
        self.assertIn("overlapping_word_spans_merged_before_gap_check", result.notes)
        self.assertEqual(result.revised.cue_in_sample, 160)

    def test_fade_shrinks_to_common_gap(self):
        result = plan_word_boundary(
            candidate(out=130, incoming=130),
            spans((0, 100, 0.9), (130, 200, 0.9)),
            spans((0, 100, 0.9), (140, 200, 0.9)),
            RATE,
            target_fade_seconds=0.5,
            min_fade_seconds=0.1,
            margin_seconds=0.0,
        )
        # The common gap is [100, 130), so the maximum common fade is 30
        # samples and the outgoing cue must finish at its right edge.
        self.assertAlmostEqual(result.fade_seconds, 0.3)
        self.assertIn("fade_shrunk_to_gap_availability", result.notes)
        self.assertEqual(result.revised.cue_out_sample, 130)
        self.assertEqual(result.revised.cue_in_sample, 110)

    def test_low_confidence_neighbor_refuses_boundary(self):
        with self.assertRaisesRegex(NoSafeBoundaryError, "no_safe_boundary"):
            plan_word_boundary(
                candidate(),
                spans((0, 100, 0.59), (200, 300, 0.95)),
                spans((0, 100, 0.95), (200, 300, 0.95)),
                RATE,
                min_confidence=0.6,
            )

    def test_missing_confidence_is_not_treated_as_safe(self):
        class MissingConfidence:
            start_sample = 200
            end_sample = 300

        with self.assertRaisesRegex(NoSafeBoundaryError, "no_safe_boundary"):
            plan_word_boundary(
                candidate(),
                (WordSpan(0, 100, 0.95), MissingConfidence()),
                spans((0, 100, 0.95), (200, 300, 0.95)),
                RATE,
            )

    def test_max_shift_and_outer_silence_are_not_bypassed(self):
        with self.assertRaisesRegex(NoSafeBoundaryError, "no_safe_boundary"):
            plan_word_boundary(
                candidate(out=10, incoming=10),
                spans((0, 100, 0.95), (200, 300, 0.95)),
                spans((0, 100, 0.95), (200, 300, 0.95)),
                RATE,
                max_shift_seconds=0.1,
            )

    def test_invalid_finite_inputs_and_overlap_without_gap(self):
        with self.assertRaisesRegex(ValueError, "invalid_sample_rate"):
            plan_word_boundary(candidate(), spans((0, 100, 0.9), (200, 300, 0.9)),
                               spans((0, 100, 0.9), (200, 300, 0.9)), math.nan)
        with self.assertRaisesRegex(ValueError, "invalid_word_span_interval"):
            plan_word_boundary(candidate(), spans((0, 100, 0.9), (90, 90, 0.9)),
                               spans((0, 100, 0.9), (200, 300, 0.9)), RATE)
        with self.assertRaisesRegex(NoSafeBoundaryError, "no_safe_boundary"):
            plan_word_boundary(candidate(), spans((0, 150, 0.9), (100, 250, 0.9)),
                               spans((0, 150, 0.9), (100, 250, 0.9)), RATE)

    def test_revised_id_is_deterministic_and_features_are_preserved(self):
        original = candidate()
        words = spans((0, 100, 0.9), (200, 300, 0.9))
        one = plan_word_boundary(original, words, words, RATE)
        two = plan_word_boundary(original, words, words, RATE)
        self.assertEqual(one, two)
        self.assertNotEqual(one.revised.candidate_id, original.candidate_id)
        self.assertEqual(one.revised.features, original.features)
        self.assertEqual(one.original_candidate_id, original.candidate_id)

    def test_outgoing_only_preserves_incoming_and_marks_it_unverified(self):
        original = candidate(out=160, incoming=777)
        result = plan_outgoing_word_boundary(
            original,
            spans((0, 100, 0.95), (200, 300, 0.95)),
            RATE,
            target_fade_seconds=0.5,
            margin_seconds=0.02,
        )
        self.assertEqual(result.revised.cue_in_sample, original.cue_in_sample)
        self.assertEqual(result.cue_in_shift_samples, 0)
        self.assertIn("incoming_boundary_unverified", result.notes)
        self.assertIn("incoming_boundary_unverified_no_word_spans", result.notes)
        self.assertAlmostEqual(result.fade_seconds, 0.5)
        outgoing_start = result.revised.cue_out_sample - round(result.fade_seconds * RATE)
        outgoing_end = result.revised.cue_out_sample
        for word in spans((0, 100, 0.95), (200, 300, 0.95)):
            self.assertTrue(outgoing_end <= word.start_sample or outgoing_start >= word.end_sample)

    def test_outgoing_only_refuses_when_A_has_no_safe_internal_gap(self):
        with self.assertRaisesRegex(NoSafeBoundaryError, "no_safe_boundary"):
            plan_outgoing_word_boundary(
                candidate(),
                spans((0, 150, 0.95), (100, 250, 0.95)),
                RATE,
            )


if __name__ == "__main__":
    unittest.main()
