from musicmix.expectation_pipeline import select_comparisons


def row(name, a, b, baseline, adjusted):
    return dict(candidate_id=name, cue_out_sample=a*48000, cue_in_sample=b*48000,
                baseline_score=baseline, adjusted_score=adjusted)


def test_controlled_selection_requires_changed_incoming_and_spaces_both_sources():
    records = [row('baseline1', 40, 10, .9, .9), row('policy1', 40, 30, .8, 1),
               row('baseline2', 60, 11, .9, .9), row('policy2', 60, 50, .8, .95)]
    chosen, audit = select_comparisons(records)
    assert len(chosen) == 1
    assert chosen[0]['policy']['cue_out_sample'] == chosen[0]['baseline']['cue_out_sample']
    assert audit['skipped']['spacing_between_trials'] == 1


def test_same_winner_or_nearby_beats_do_not_become_comparisons():
    chosen, audit = select_comparisons([row('a', 40, 10, .9, .9), row('b', 40, 11, .8, 1)])
    assert chosen == []
    assert audit['skipped']['unchanged_or_nearby_winner'] == 1
