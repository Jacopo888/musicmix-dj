"""Audit real pool renders, matched outgoing contexts, copies and pair coverage."""
from pathlib import Path
import argparse
import gzip
import json
import sys
import wave
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from musicmix.audio import load_audio
from musicmix.ingest import sha256_file
from musicmix.paths import localize
from musicmix.trajectory import extract_trajectory,trajectory_to_dict
from musicmix.expectation import analyze_expectation

def audit(root):
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['pair_count']==len(manifest['tracks'])*(len(manifest['tracks'])-1)
    assert manifest['pool_diagnostics']['all_ordered_pairs_covered']
    assert not any(p['candidate_grid_truncated'] for p in manifest['all_pair_counts'].values())
    with gzip.open(root/'search.json.gz','rt',encoding='utf-8') as handle:pool=json.load(handle)
    proposals={r['candidate']['candidate_id']:r for r in pool['proposals']}
    tracks={entry['track_id']:load_audio(entry['canonical_path']) for entry in manifest['tracks']}
    structures={entry['track_id']:json.loads(localize(entry['structure_path']).read_text(encoding='utf-8'))
                for entry in manifest['tracks']}
    reference_profiles={};reference_count=0
    def full_profile(track_id,original,effective,side):
        key=(track_id,original,effective,side)
        if key not in reference_profiles:
            reference_profiles[key]=trajectory_to_dict(extract_trajectory(tracks[track_id],original,
                structures[track_id]['beats'],side=side,cue_shift_samples=effective-original))
        return reference_profiles[key]
    hashes=set();outgoing={};results=[]
    for artifact in manifest['artifacts']:
        digest=sha256_file(localize(artifact['path']))
        assert digest==artifact['sha256'] and digest not in hashes;hashes.add(digest)
        candidate=proposals[artifact['candidate_id']]['candidate']
        comparison=manifest['comparisons'][artifact['trial']-1][artifact['system']]
        assert comparison['candidate_id']==candidate['candidate_id']
        original=proposals[artifact['candidate_id']]['raw_features']['original_structure_cue_samples']
        reference=analyze_expectation(
            full_profile(candidate['track_a'],original['a'],candidate['cue_out_sample'],'outgoing'),
            full_profile(candidate['track_a'],original['a'],candidate['cue_out_sample'],'incoming'),
            full_profile(candidate['track_b'],original['b'],candidate['cue_in_sample'],'incoming'))
        assert reference.label==comparison['label']
        assert reference.score==comparison['expectation_score'],artifact['candidate_id']
        reference_count+=1
        assert artifact['clipped_samples']==0 and artifact['join_sample']==480000
        with wave.open(artifact['path'],'rb') as reader:
            assert (reader.getframerate(),reader.getnchannels(),reader.getsampwidth(),reader.getnframes())==(48000,2,2,960000)
            pcm=np.frombuffer(reader.readframes(960000),dtype='<i2').reshape(-1,2)
        for start,end,identity,cue,offset in [(480,479424,'track_a','cue_out_sample',-480000),
                                             (480576,959520,'track_b','cue_in_sample',-480000)]:
            track=tracks[candidate[identity]];assert track.track_id==candidate[identity]
            begin=candidate[cue]+start+offset
            expected=np.round(.9*track.samples[begin:begin+end-start].astype(np.float64)*32767).astype('<i2')
            assert np.array_equal(pcm[start:end],expected),(artifact['candidate_id'],identity)
        outgoing.setdefault(artifact['trial'],[]).append(pcm[:480000].copy())
        results.append({'candidate_id':artifact['candidate_id'],'sha256':digest,'source_pcm_match':True})
    for pair in outgoing.values():assert len(pair)==2 and np.array_equal(*pair)
    assert results
    key=json.loads((root/'listening/key.json').read_text(encoding='utf-8'))
    for item in key['trials']:
        copy=root/'listening'/f"trial_{item['trial']:02d}_{item['label']}.wav"
        assert sha256_file(copy)==sha256_file(localize(item['source']))
    qa={'status':'pass','wav_count':len(results),'unique_wavs':len(hashes),'pair_count':manifest['pair_count'],
        'same_outgoing_context':True,'canonical_pcm_interiors_match':True,'listening_copies_match':True,
        'pooled_expectation_matches_full_extractor':True,'reference_comparisons':reference_count,
        'files':results,'audit_code_sha256':sha256_file(Path(__file__))}
    (root/'audio_qa.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    return qa

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    args=parser.parse_args();result=audit(args.run)
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
