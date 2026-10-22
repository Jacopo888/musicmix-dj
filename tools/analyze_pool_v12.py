"""Reuse cached official providers on explicitly selected new V12 sources."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import analyze_pool_v10 as providers
from musicmix.paths import localize

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,default=ROOT/'runs/pool_v12/catalog_initial.json')
    p.add_argument('--slug',default='after_hours')
    p.add_argument('--stage',choices=('structure','vocals'),required=True)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--timeout',type=float,default=900)
    p.add_argument('--pool-root',type=Path,default=ROOT/'runs/pool_v12',
                   help='cartella del pool che contiene structure/, vocals/, evidence/ (default: pool V12)')
    args=p.parse_args();catalog=providers._load_json(args.catalog)
    entry=next(x for x in catalog['tracks'] if x['slug']==args.slug)
    canonical=localize(entry['canonical_path']).resolve(strict=True)
    if entry['track_id']!=providers._sha256(canonical):raise ValueError('Canonical changed')
    source=localize(entry['source_path']).resolve(strict=True)
    if entry['source_sha256']!=providers._sha256(source):raise ValueError('Source changed')
    pool_root=args.pool_root if args.pool_root.is_absolute() else ROOT/args.pool_root
    expected=(pool_root/args.stage/args.slug/'a.json').resolve()
    if localize(entry[args.stage+'_path']).resolve()!=expected:raise ValueError('New pool output required')
    if expected.exists():raise FileExistsError('Do not repeat completed provider inference')
    providers._require_disk_headroom();providers._runtime_sys_path()
    item={'slug':args.slug,'entry':entry,'canonical':canonical,'source':source,
          'canonical_sha256':entry['track_id'],'source_sha256':entry['source_sha256']}
    if args.stage=='structure':
        result=providers._run_structure(item,args,providers._make_structure_provider(args))
    else:result=providers._run_vocals(item,args)
    evidence={'stage':args.stage,'canonical_sha256':entry['track_id'],'source_sha256':entry['source_sha256'],
              'stage_status':result['stage_status'],'output_path':str(expected),'output_sha256':providers._sha256(expected),
              'model':result.get('model'),'separator':result.get('separator'),'warnings':result.get('warnings'),
              'runner_sha256':providers._sha256(Path(__file__)),'reused_provider_runner_sha256':providers._sha256(Path(providers.__file__))}
    out=pool_root/'evidence';out.mkdir(parents=True,exist_ok=True)
    providers._atomic_json(out/f'{args.slug}_{args.stage}.json',evidence)
    print(json.dumps({k:v for k,v in evidence.items() if k not in ('model','separator','warnings')}),flush=True)

if __name__=='__main__':main()
