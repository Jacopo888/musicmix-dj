"""Test di isolamento del solo ranker, senza librerie AI e senza audio."""
import unittest

from musicmix.contracts import Candidate, Feature
from musicmix.ranking import InterpretableRanker


def mk_candidate(**changes):
    args = dict(candidate_id='test-A-B', track_a='A', track_b='B',
                cue_out_sample=12000, cue_in_sample=48000, mode='word',
                features={'horizontal': Feature(0.8, 0.9),
                          'vertical': Feature(None, None),
                          'continuity': Feature(0.9, 0.9),
                          'creative': Feature(0.5, 0.7)},
                penalties={'transform': Feature(0.2, 0.9),
                           'artifact': Feature(0.1, 0.9)})
    args.update(changes)
    return Candidate(**args)


class TestRanker(unittest.TestCase):
    def test_reproducible(self):
        ranker = InterpretableRanker()
        self.assertEqual(ranker.score(mk_candidate()), ranker.score(mk_candidate()))

    def test_missing_is_not_zero(self):
        rep = InterpretableRanker().score(mk_candidate())
        self.assertTrue(rep.valid)
        self.assertIn('vertical', rep.missing)
        self.assertAlmostEqual(sum(rep.contributions.values()),
                               (0.8*.4+0.9*.35+0.5*.15)/(.4+.35+.15), places=5)

    def test_penalties_reduce_score(self):
        c1 = mk_candidate(penalties={'transform':Feature(0,1), 'artifact':Feature(0,1)})
        c2 = mk_candidate(penalties={'transform':Feature(1,1), 'artifact':Feature(1,1)})
        r=InterpretableRanker()
        self.assertGreater(r.score(c1).score, r.score(c2).score)

    def test_hard_constraint(self):
        rep=InterpretableRanker().score(mk_candidate(passes_constraints=False,
                                  constraint_reasons=('tempo_out_of_range',)))
        self.assertFalse(rep.valid)
        self.assertIsNone(rep.score)
        self.assertIn('tempo_out_of_range', rep.notes)

    def test_low_confidence_excluded(self):
        c=mk_candidate(features={'horizontal': Feature(.8,.2), 'continuity': Feature(.9,.1)})
        rep=InterpretableRanker().score(c)
        self.assertFalse(rep.valid)
        self.assertIn('no_reliable_benefit_features', rep.notes)

    def test_direction_is_distinct_in_contract(self):
        c=mk_candidate()
        rev=mk_candidate(candidate_id='test-B-A', track_a='B', track_b='A',
                         cue_out_sample=c.cue_in_sample, cue_in_sample=c.cue_out_sample)
        self.assertNotEqual((c.track_a,c.track_b,c.cue_out_sample,c.cue_in_sample),
                            (rev.track_a,rev.track_b,rev.cue_out_sample,rev.cue_in_sample))


if __name__ == '__main__':
    unittest.main()
