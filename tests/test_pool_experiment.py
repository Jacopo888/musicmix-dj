from musicmix.pool_experiment import select_pool_trials,label_audio_source


def row(name,a,b,out,inc,base,adjusted):
    return dict(candidate_id=name,track_a=a,track_b=b,cue_out_sample=out*48000,
                cue_in_sample=inc*48000,baseline_score=base,adjusted_score=adjusted)


def test_comparison_keeps_same_outgoing_but_can_change_incoming_song():
    trials,_=select_pool_trials([row('a','source','one',40,30,.9,.9),row('b','source','two',40,20,.8,1)])
    assert len(trials)==1 and trials[0]['kind']=='expectation_rerank'
    assert trials[0]['baseline']['cue_out_sample']==trials[0]['choice']['cue_out_sample']
    assert trials[0]['baseline']['track_b']!=trials[0]['choice']['track_b']


def test_unchanged_policy_gets_explicit_exploration_instead_of_fake_improvement():
    trials,_=select_pool_trials([row('a','source','one',40,30,.9,.9),row('b','source','two',40,20,.8,.8)])
    assert trials[0]['kind']=='pool_alternative_exploration'


def test_track_role_and_cross_role_cue_reuse_are_limited():
    rows=[row('a','source','one',40,30,.9,.9),row('b','source','two',40,20,.8,1),
          row('c','one','three',30,40,.9,.9),row('d','one','two',30,60,.8,1)]
    trials,audit=select_pool_trials(rows)
    assert len(trials)==1 and not audit['search_truncated']


def test_backtracking_does_not_lose_a_song_to_the_first_greedy_incoming_choice():
    rows=[row('a1','a','x',40,30,.9,.9),row('a1alt','a','y',40,20,.8,1.2),
          row('a2','a','x',80,70,.9,.9),row('a2alt','a','y',80,60,.8,1),
          row('b','b','x',100,30,.9,.9),row('balt','b','z',100,40,.8,1)]
    trials,audit=select_pool_trials(rows)
    assert len(trials)==2 and not audit['search_truncated']
    a=next(t for t in trials if t['source_track']=='a')
    assert a['outgoing_sample']==80*48000


def test_source_captions_preserve_seek_player_ids_and_are_idempotent():
    html='<audio id="musicmix_player_0" controls preload="none" src="trial_01_A.wav"></audio><span data-player-version="2"></span>'
    result=label_audio_source(html,'trial_01_A.wav','Artist & song <safe>')
    assert 'Artist &amp; song &lt;safe&gt;' in result
    assert html in result and 'id="musicmix_player_0"' in result
    assert label_audio_source(result,'trial_01_A.wav','Artist & song <safe>')==result
