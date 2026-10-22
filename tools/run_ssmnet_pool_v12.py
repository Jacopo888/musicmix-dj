"""Run the strictly loaded official SSM-Net checkpoint on canonical sources."""
import argparse,importlib.util,json,sys,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from musicmix.paths import localize

def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,default=ROOT/'runs/pool_v12/catalog_initial.json')
    p.add_argument('--output',type=Path,default=ROOT/'runs/ssmnet_pool_v12');a=p.parse_args()
    source=ROOT/'runs/result_swarm_v12/agents/ssmnet_official';cache=ROOT/'.cache/ssmnet_v12'
    if not cache.exists():
        cache.mkdir(parents=True)
        for name in ('source','weights','tools'):shutil.copytree(source/name,cache/name,ignore=shutil.ignore_patterns('__pycache__'))
    runner=cache/'tools/run_ssmnet_official.py'
    spec=importlib.util.spec_from_file_location('ssmnet_v12_official',runner);module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    catalog=json.loads(a.catalog.read_text(encoding='utf8'));a.output.mkdir(parents=True,exist_ok=True)
    config=module.yaml.safe_load((cache/'tools/config_nbatt1.yaml').read_text(encoding='utf8'))
    weights=cache/'weights'/module.CHECKPOINT_NAME
    if module.sha256_file(weights)!=module.CHECKPOINT_SHA256:raise ValueError('Checkpoint changed')
    records=[]
    for entry in catalog['tracks']:
        identity=entry['track_id'];audio=localize(entry['canonical_path']);folder=a.output/entry['slug'];folder.mkdir(exist_ok=True)
        output=folder/'prediction.json'
        if output.exists():
            record=json.loads(output.read_text(encoding='utf8'))
            if record.get('track_id')!=identity:raise ValueError('Stale structural prediction')
        else:
            if module.sha256_file(audio)!=identity:raise ValueError('Canonical audio changed')
            print('Official SSM-Net: '+entry['slug'],flush=True)
            result=module.process_clip(audio,config,weights,folder,entry['slug'],0.0)
            record={'track_id':identity,'sample_rate':entry['sample_rate'],'duration_samples':entry['duration_samples'],
                    'status':'complete','result':result,'runner_sha256':module.sha256_file(runner)}
            output.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
        records.append({'track_id':identity,'slug':entry['slug'],'path':str(output.resolve()),'sha256':module.sha256_file(output)})
    manifest={'schema_version':'musicmix.official-ssmnet-pool.1','status':'complete','tracks':records,
              'checkpoint_sha256':module.CHECKPOINT_SHA256,'source_revision':module.PINNED_COMMIT,
              'meaning':'Pretrained structural novelty; not musical quality or tension resolution'}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps({'status':'complete','tracks':len(records)}),flush=True)

if __name__=='__main__':main()
