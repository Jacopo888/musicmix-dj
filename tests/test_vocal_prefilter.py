from dataclasses import asdict
from test_directional import _track,_structure,_vocal
from musicmix.directional import search_directional


def search(fast,*,maximum=10000,unknown=False):
    a,b=_track('a'),_track('b',reverse=True)
    va=_vocal([(250,480,.99),(1050,1280,.99)])
    vb=_vocal([(500,730,.99),(1500,1720,.99)])
    if unknown:va=vb=None
    return search_directional(a,b,_structure(a),_structure(b),va,vb,
               top_k=3,max_candidates=maximum,min_context_seconds=0,prefilter_vocals=fast,
               min_cue_spacing_seconds=0)


def test_prefilter_preserves_every_admitted_proposal_and_score_exactly():
    original=search(False);fast=search(True)
    assert [asdict(p) for p in original.proposals]==[asdict(p) for p in fast.proposals]
    assert original.selected==fast.selected
    assert fast.diagnostics['candidate_count_evaluated']<original.diagnostics['candidate_count_evaluated']
    assert fast.diagnostics['vocal_prefilter']['pruned_candidate_pairs']>0


def test_prefilter_does_not_change_a_truncated_sampling_grid():
    original=search(False,maximum=3);fast=search(True,maximum=3)
    assert [asdict(p) for p in original.proposals]==[asdict(p) for p in fast.proposals]
    assert not fast.diagnostics['vocal_prefilter']['applied']
    assert fast.diagnostics['vocal_prefilter']['disabled_reason']=='original_grid_truncated'


def test_unknown_vocals_still_fail_closed_without_scoring_cartesian_pairs():
    fast=search(True,unknown=True)
    assert not fast.proposals and not fast.items
    assert fast.diagnostics['candidate_count_evaluated']==0
