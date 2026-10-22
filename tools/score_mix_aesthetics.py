"""Run a local independent neural critic on windows centered on rendered joins."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from musicmix.audio import AudioTrack,load_audio
from musicmix.aesthetic_critic import AudioboxSeamCritic
from musicmix.mix_quality import evaluate_mix_quality
from musicmix.ingest import sha256_file
from musicmix.paths import localize

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--counterfactual',action='store_true');args=parser.parse_args()
    if args.output.exists():raise ValueError('Output must be new')
    manifest=args.run/'manifest.json'
    if not manifest.exists():manifest=args.run/'experiment.json'
    run=json.loads(manifest.read_text(encoding='utf-8'))
    artifacts=run.get('artifacts',[])
    if not artifacts:artifacts=[a for d in run['directions'].values() for a in d['artifacts']]
    critic=AudioboxSeamCritic(ROOT);rows=[]
    for index,artifact in enumerate(artifacts):
        path=localize(artifact['path']);track=load_audio(path);join=int(artifact['join_sample'])
        if track.track_id!=artifact['sha256']:raise ValueError('Render hash mismatch')
        report=critic.score(track,join)
        rows.append({'artifact':artifact,'diagnostic':report})
        print(json.dumps({'event':'scored','file':path.name,'axes':report['axes']}),flush=True)
        if args.counterfactual and index==0:
            damaged=np.array(track.samples,copy=True);half=track.sample_rate
            damaged[max(0,join-half):join+half]=np.clip(damaged[max(0,join-half):join+half]*8,-1,1)
            # This in-memory counterfactual has its own explicit identity and
            # is never exported as a listening candidate or source recording.
            import hashlib
            identity='counterfactual-'+hashlib.sha256(damaged.tobytes()).hexdigest()
            example=AudioTrack(identity,'synthetic corruption of '+str(path),track.sample_rate,damaged)
            counter=critic.score(example,join)
            technical=evaluate_mix_quality(example,join)
            rows.append({'counterfactual':'2s hard-clipped corruption around seam',
                         'source_artifact_sha256':track.track_id,'diagnostic':counter,
                         'technical':technical,'used_for_threshold_fitting':False})
            print(json.dumps({'event':'counterfactual','axes':counter['axes']}),flush=True)
    result={'status':'complete','rows':rows,'input_manifest':str(manifest.resolve()),
            'input_manifest_sha256':sha256_file(manifest),'code_sha256':sha256_file(Path(__file__)),
            'selection_effect':'none; independent diagnostic, not a validated transition reward'}
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
