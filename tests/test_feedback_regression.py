"""Anonymized timing regression from the user's rejected third proposal.

Numbers are estimated timing evidence, not a gold lyric alignment or a claim
of perceptual success. No copyrighted audio or lyric text is needed here.
"""
import unittest

from musicmix.contracts import Candidate
from musicmix.vocal_boundaries import boundary_safety
from musicmix.word_guard import WordSpan, plan_outgoing_word_boundary, NoSafeBoundaryError


class ReportedWordCutTests(unittest.TestCase):
    def test_old_downbeat_can_still_split_a_word(self):
        rate = 48000
        document = {'duration_samples': 219*rate,
                    'coverage_intervals': [{'start_sample': 124*rate, 'end_sample': 141*rate,
                                            'status': 'asr_observed'}],
                    'uncertain_regions': [],
                    'words': [{'start_sample': round(132.04*rate), 'end_sample': round(132.58*rate),
                               'confidence': .98}]}
        result = boundary_safety(document, side='A_out', cue_sample=round(132.096*rate),
                                  fade_samples=round(.04*rate))
        self.assertEqual(result['status'], 'unsafe')
        self.assertIn('fade_overlaps_known_word', result['reasons'])

    def test_large_word_fix_cannot_masquerade_as_small_beat_adjustment(self):
        rate = 48000
        candidate = Candidate('reported-third', 'a', 'b', round(132.096*rate),
                              round(10.944*rate), 'structure', {}, {})
        words = [WordSpan(round(132.04*rate), round(132.58*rate), .98),
                 WordSpan(round(132.58*rate), round(133.22*rate), .76),
                 WordSpan(round(133.84*rate), round(134.28*rate), .74)]
        with self.assertRaises(NoSafeBoundaryError):
            plan_outgoing_word_boundary(candidate, words, rate, max_shift_seconds=.08,
                                         target_fade_seconds=.04, margin_seconds=.02)
