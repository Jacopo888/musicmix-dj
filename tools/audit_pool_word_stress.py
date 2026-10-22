"""Stress reported ASR envelopes without inventing alignment confidence."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from musicmix.ingest import sha256_file
from musicmix.paths import localize

def audit(root):
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    vocals={}
    for track in manifest['tracks']:
        source=localize(track['vocals_path'])
        if sha256_file(source)!=track['vocals_sha256']:raise ValueError('Vocal evidence changed')
        vocals[track['track_id']]=json.loads(source.read_text(encoding='utf-8'))
    proposals={}
    for path in root.glob('proposal_*.json'):
        item=json.loads(path.read_text(encoding='utf-8'));proposals[item['candidate']['candidate_id']]=item['candidate']
    rows=[]
    for artifact in manifest['artifacts']:
        candidate=proposals[artifact['candidate_id']];rate=artifact['sample_rate']
        fade=round(artifact['fade_seconds']*rate);sides={}
        for side,track_key,cue_key in [('outgoing','track_a','cue_out_sample'),('incoming','track_b','cue_in_sample')]:
            document=vocals[candidate[track_key]];cue=candidate[cue_key]
            start,end=(cue-fade,cue) if side=='outgoing' else (cue,cue+fade)
            cases=[]
            for margin_ms in (0,20,50,100,200):
                margin=round(margin_ms*rate/1000)
                conflicts=[w for w in document.get('words',[]) if int(w['start_sample'])-margin<end
                           and int(w['end_sample'])+margin>start]
                cases.append({'assumed_timestamp_expansion_ms':margin_ms,
                              'overlapping_word_envelopes':len(conflicts),
                              'low_confidence_overlaps':sum(w.get('confidence') is None or w['confidence']<.6 for w in conflicts)})
            sides[side]={'track_id':candidate[track_key],'fade_interval_samples':[start,end],
                         'asr_word_count':len(document.get('words',[])),
                         'cases':cases,'confidence':None,'verified_word_integrity':None}
        rows.append({'candidate_id':candidate['candidate_id'],'artifact_sha256':artifact['sha256'],'sides':sides})
    result={'status':'stress_test_only','rows':rows,'used_for_ranking':False,
            'assumptions_ms':[0,20,50,100,200],'code_sha256':sha256_file(Path(__file__)),
            'limitations':['Expansion values are hypothetical perturbations, not calibrated confidence intervals.',
                           'No overlap with reported ASR spans does not establish absence of a cut sung word.',
                           'Known ASR errors require independent singing alignment or manual boundaries.']}
    (root/'word_boundary_stress.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    args=parser.parse_args();result=audit(args.run)
    print(json.dumps({'status':result['status'],'artifacts':len(result['rows']),
                      'note':'uncertainty stress only; no word-integrity claim'},indent=2))
