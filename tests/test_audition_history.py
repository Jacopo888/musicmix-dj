from dataclasses import asdict
import hashlib
import json

from musicmix.audition_history import exposure, load_audition_history,include_presented_runs
from musicmix.contracts import Candidate


def fixture(tmp_path, name, candidate, audio):
    root = tmp_path/'runs'/name
    root.mkdir(parents=True)
    (root/'proposal_01.json').write_text(json.dumps({'candidate': asdict(candidate)}))
    (root/'manifest.json').write_text(json.dumps({
        'schema_version': 'musicmix.structured-run.1',
        'tracks': [{'sha256': candidate.track_a, 'sample_rate': 48000},
                   {'sha256': candidate.track_b, 'sample_rate': 48000}],
        'artifacts': [{'candidate_id': candidate.candidate_id, 'sha256': hashlib.sha256(audio.read_bytes()).hexdigest()}]}))


def test_confirmed_feedback_binds_audio_hash_and_collapses_repeated_exposure(tmp_path):
    audio = tmp_path/'one.wav'; audio.write_bytes(b'actual-audio')
    candidate = Candidate('one', 'a', 'b', 480000, 960000, 'structure', {}, {})
    fixture(tmp_path, 'first', candidate, audio)
    folder = tmp_path/'feedback'; folder.mkdir()
    document = {'schema_version': 'musicmix.evaluation.1', 'mapping': {'confirmed': True},
                'observations': [{'preference_label': 'A', 'candidates': {
                    'A': {'source': str(audio), 'system': 'new'},
                    'B': {'source': str(audio), 'system': 'old'}}}]}
    (folder/'feedback.json').write_text(json.dumps(document))
    result = load_audition_history(folder, tmp_path/'runs')
    assert len(result['regions']) == 1
    assert result['unresolved'] == []
    assert exposure(candidate, result['regions'], 2)['ordered_pair_heard']
    reverse = Candidate('reverse', 'b', 'a', 960000, 480000, 'structure', {}, {})
    flags = exposure(reverse, result['regions'], 2)
    assert flags['outgoing_region_heard'] and flags['incoming_region_heard']
    assert not flags['ordered_pair_heard']


def test_ambiguous_identical_audio_is_not_assigned_a_guessed_timestamp(tmp_path):
    audio = tmp_path/'loop.wav'; audio.write_bytes(b'identical-loop-at-different-cues')
    for index, cue in enumerate((480000, 960000)):
        candidate = Candidate(str(index), 'a', 'b', cue, 960000, 'structure', {}, {})
        fixture(tmp_path, str(index), candidate, audio)
    folder = tmp_path/'feedback'; folder.mkdir()
    (folder/'feedback.json').write_text(json.dumps({'schema_version': 'musicmix.evaluation.1',
        'mapping': {'confirmed': True}, 'observations': [{'preference_label': 'A',
        'candidates': {'A': {'source': str(audio), 'system': 'new'}}}]}))
    result = load_audition_history(folder, tmp_path/'runs')
    assert result['regions'] == []
    assert result['unresolved'][0]['reason'] == 'unresolved_or_ambiguous_geometry'


def test_controlled_experiment_feedback_is_included_in_future_history(tmp_path):
    audio = tmp_path/'new.wav'; audio.write_bytes(b'new-pilot')
    root = tmp_path/'runs'/'pilot'; root.mkdir(parents=True)
    candidate = asdict(Candidate('new', 'a', 'b', 480000, 960000, 'structure', {}, {}))
    (root/'experiment.json').write_text(json.dumps({'schema_version': 'musicmix.expectation-experiment.1',
        'directions': {'forward': {'comparisons': [{'policy': candidate, 'baseline': candidate}],
        'artifacts': [{'candidate_id': 'new', 'sample_rate': 48000,
                       'sha256': hashlib.sha256(audio.read_bytes()).hexdigest()}]}}}))
    folder = tmp_path/'feedback'; folder.mkdir()
    (folder/'feedback.json').write_text(json.dumps({'schema_version': 'musicmix.evaluation.1',
        'mapping': {'confirmed': True}, 'observations': [{'preference_label': 'A',
        'candidates': {'A': {'source': str(audio), 'system': 'expectation_v7'}}}]}))
    result = load_audition_history(folder, tmp_path/'runs')
    assert not result['unresolved']
    assert result['regions'][0]['cue_out_sample'] == 480000


def test_multisong_pool_manifest_preserves_the_actual_pair_not_first_two_tracks(tmp_path):
    audio=tmp_path/'pool.wav';audio.write_bytes(b'pool-audio')
    candidate=Candidate('pool','c','d',480000,960000,'structure',{}, {})
    fixture(tmp_path,'pool',candidate,audio)
    manifest=tmp_path/'runs/pool/manifest.json'
    data=json.loads(manifest.read_text())
    data['tracks']=[{'track_id':name,'sample_rate':48000} for name in ('a','b','c','d')]
    data['schema_version']='musicmix.pool-run.1';manifest.write_text(json.dumps(data))
    folder=tmp_path/'feedback';folder.mkdir()
    (folder/'feedback.json').write_text(json.dumps({'schema_version':'musicmix.evaluation.1',
        'mapping':{'confirmed':True},'observations':[{'preference_label':'A',
        'candidates':{'A':{'source':str(audio),'system':'pool_v8'}}}]}))
    result=load_audition_history(folder,tmp_path/'runs')
    assert not result['unresolved'] and len(result['regions'])==1
    assert result['regions'][0]['track_a']=='c' and result['regions'][0]['track_b']=='d'


def test_presented_proposals_are_excluded_without_becoming_human_feedback(tmp_path):
    audio=tmp_path/'clip.wav';audio.write_bytes(b'presented-audio')
    candidate=asdict(Candidate('c','a','b',480000,960000,'structure',{},{}))
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'listening':{'test_id':'published'},'comparisons':[{'baseline':candidate,'choice':candidate}],
        'artifacts':[{'candidate_id':'c','path':str(audio),'sample_rate':48000,
                      'sha256':hashlib.sha256(audio.read_bytes()).hexdigest()}]}))
    registry=tmp_path/'registry.json'
    registry.write_text(json.dumps({'schema_version':'musicmix.presented-runs.1',
        'runs':[{'manifest':'manifest.json','test_ids':['published']}]}))
    original={'regions':[],'inputs':[],'unresolved':[]}
    history=include_presented_runs(original,registry,tmp_path)
    assert original['regions']==[] and len(history['regions'])==1
    assert history['regions'][0]['proposed_only']
    assert history['inputs']==[]
    assert exposure(Candidate(**candidate),history['regions'],8)['ordered_pair_heard']


def test_pretrained_review_history_deduplicates_shared_controls_and_verifies_audio(tmp_path):
    import pytest
    audio=tmp_path/'clip.wav';audio.write_bytes(b'pretrained-control')
    candidate=asdict(Candidate('c','a','b',480000,960000,'structure',{},{}))
    artifact={'candidate':candidate,'path':str(audio),'sample_rate':48000,
              'sha256':hashlib.sha256(audio.read_bytes()).hexdigest()}
    entries=[]
    for name in ('cue','sections'):
        path=tmp_path/f'{name}.json'
        path.write_text(json.dumps({'schema_version':'musicmix.pretrained-cue-pilot.1','artifacts':[artifact]}))
        entries.append({'manifest':path.name,'presentation':'pretrained_review'})
    registry=tmp_path/'registry.json'
    registry.write_text(json.dumps({'schema_version':'musicmix.presented-runs.1','runs':entries}))
    history=include_presented_runs({'regions':[],'inputs':[]},registry,tmp_path)
    assert len(history['regions'])==1 and history['regions'][0]['proposed_only']
    assert history['inputs']==[] and len(history['presented_inputs'])==2
    audio.write_bytes(b'modified')
    with pytest.raises(ValueError,match='hash mismatch'):
        include_presented_runs({'regions':[],'inputs':[]},registry,tmp_path)


def test_psychology_review_records_main_cues_once_without_inventing_ratings(tmp_path):
    import pytest
    audio=tmp_path/'clip.wav';audio.write_bytes(b'psychology-render')
    candidate=asdict(Candidate('c','a','b',480000,960000,'structure',{},{}))
    render={'path':str(audio),'sample_rate':48000,'sha256':hashlib.sha256(audio.read_bytes()).hexdigest()}
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'schema_version':'musicmix.psychology-pool.1',
        'artifacts':[{'variant':variant,'candidate':candidate,'artifact':render}
                     for variant in ('main','same_cue_no_level_match','main')]}))
    registry=tmp_path/'registry.json'
    registry.write_text(json.dumps({'schema_version':'musicmix.presented-runs.1',
        'runs':[{'manifest':manifest.name,'presentation':'psychology_review'}]}))
    original={'regions':[],'inputs':[],'unresolved':[]}
    history=include_presented_runs(original,registry,tmp_path)
    assert original['regions']==[]
    assert len(history['regions'])==1 and history['regions'][0]['proposed_only']
    assert 'preference_label' not in history['regions'][0]
    audio.write_bytes(b'corrupt-render')
    with pytest.raises(ValueError,match='hash mismatch'):
        include_presented_runs(original,registry,tmp_path)
