"""Run cached optional providers sequentially for newly added pool sources."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT/'src', ROOT/'.cache/structure_runtime/Lib/site-packages',
             ROOT/'.cache/structure_runtime/packages'):
    sys.path.insert(0,str(item))
from musicmix.paths import localize

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('structure','vocals'),required=True)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    catalog=json.loads((ROOT/'runs/pool_v8/catalog.json').read_text(encoding='utf-8'))
    provider=None
    if args.stage=='structure':
        from analyze_structure import BeatThisConfig,BeatThisProvider,analyze_one
        provider=BeatThisProvider(BeatThisConfig(str(ROOT/'.cache/structure_runtime/models/beat_this-small0.ckpt'),threads=args.threads))
    else:
        from analyze_vocals import analyse_one
        model_root=ROOT/'.cache/models/models--Systran--faster-whisper-small.en/snapshots'
        model=next(p for p in model_root.iterdir() if (p/'model.bin').is_file())
    for entry in catalog['tracks']:
        if not entry.get('track_id'):continue
        output=localize(entry[args.stage+'_path'])
        if output.is_file():
            print(json.dumps({'event':'reused','slug':entry['slug'],'stage':args.stage}),flush=True);continue
        print(json.dumps({'event':'started','slug':entry['slug'],'stage':args.stage}),flush=True)
        output.parent.mkdir(parents=True,exist_ok=True)
        if args.stage=='structure':
            result=analyze_one(localize(entry['canonical_path']),output,provider,900.0,True)
            print(json.dumps({'event':'finished','slug':entry['slug'],'status':result.get('status'),
                              'beats':len(result.get('beats',[])),'downbeats':len(result.get('downbeats',[]))}),flush=True)
        else:
            result=analyse_one('a',localize(entry['canonical_path']),output.parent,model_path=model,
                               runtime_root=ROOT/'.cache/vocal_runtime',threads=args.threads,skip_separation=False)
            print(json.dumps({'event':'finished','slug':entry['slug'],'status':result.get('analysis_status'),
                              'words':len(result.get('words',[])),'source_kind':result.get('source_kind')}),flush=True)
