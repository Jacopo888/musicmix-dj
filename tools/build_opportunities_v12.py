"""Combine native role-specific timing with verified official structural peaks."""
import json,sys
from pathlib import Path
from collections import Counter
from dataclasses import asdict
from build_vocal_opportunities_v11 import _native_cues, _provider_records, sha256_file
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from musicmix.role_opportunities import extract_track_opportunities
from musicmix.paths import localize

def read(p):return json.loads(localize(p).read_text(encoding='utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
    path=ROOT/'runs/pool_v12/catalog_ready.json';cat=read(path)
    if cat['status']!='ready':raise ValueError('Providers not ready')
    out=ROOT/'runs/opportunities_v12'
    if (out/'registry.json').exists():raise FileExistsError('Frozen registry exists')
    out.mkdir(exist_ok=True)
    by={e['track_id']:e for e in cat['tracks']};providers=[]
    for name,file in [('cue_detr','cue_registry_pool6.json'),('linkseg','linkseg_registry_pool6.json')]:
        values,record=_provider_records(ROOT/'runs/pretrained_v10'/file,by,name)
        providers.append((name,values,record))
    registry=[];counts=Counter()
    for e in cat['tracks']:
        identity=e['track_id'];v=read(e['vocals_path']);structure=read(e['structure_path'])
        if sha256_file(localize(e['vocals_path']))!=e['vocals_sha256']:raise ValueError('Changed vocal evidence')
        raw=_native_cues(v,e)
        for name,values,record in providers:
            raw.extend(dict(c,role='both',provider=name) for c in values.get(identity,[]))
        learned_path=ROOT/'runs/ssmnet_pool_v12'/e['slug']/'prediction.json';learned=read(learned_path)
        if learned['track_id']!=identity:raise ValueError('Structural identity mismatch')
        for peak in learned['result']['events_official']:
            s=round(peak['time_canonical_sec']*48000)
            raw.append({'sample':s,'time_seconds':s/48000,'role':'both','provider':'official_ssmnet',
                        'novelty':peak['novelty_score'],'observation':'structural novelty, not resolution'})
        merged={}
        for cue in raw:
            sample=cue.get('sample',round(cue['time_seconds']*48000))
            if not 10*48000 <=sample <= e['duration_samples']-10*48000:continue
            row=merged.setdefault(sample,{'sample':sample,'time_seconds':sample/48000,'roles':set(),'sources':[]})
            role=cue.get('role','both');row['roles'].update(['outgoing','incoming'] if role not in {'outgoing','incoming'} else [role])
            row['sources'].append(cue)
        cues=[]
        for sample,row in sorted(merged.items()):
            row['roles']=sorted(row['roles']);cues.append(row)
        # Preserve the richer role planner's pickup/arrival and metric evidence
        # for audit; its uncalibrated word-margin proposals do not override
        # the backend's bounded final word/phase guard.
        role_input=dict(e,structure=structure,vocals=v,provided_opportunities=[dict(c,role=('both' if len(c['roles'])==2 else c['roles'][0])) for c in cues])
        role_plan=extract_track_opportunities(role_input)
        write(out/(e['slug']+'_roles.json'),asdict(role_plan))
        evidence=out/(e['slug']+'_evidence.json')
        write(evidence,{'canonical_sha256':identity,'vocal_sha256':e['vocals_sha256'],
             'structure_sha256':e['structure_sha256'],'ssmnet_path':str(learned_path),
             'ssmnet_sha256':sha256_file(learned_path),'provider_registries':[p[2] for p in providers],
             'roles_preserved':True,'phrase_closure':'unknown','cue_count':len(cues)})
        registry.append(dict(track_id=identity,sample_rate=48000,duration_samples=e['duration_samples'],
            slug=e['slug'],status='complete',cues=cues,inference_evidence={
            'canonical_sha256':identity,'artifact_path':str(evidence),'artifact_sha256':sha256_file(evidence)}))
        counts[e['slug']]=len(cues)
    write(out/'registry.json',{'schema_version':'musicmix.learned-opportunity-registry.1','tracks':registry,
          'catalog_sha256':sha256_file(path),'cue_counts':dict(counts)})
    print(json.dumps(dict(counts)),flush=True)
if __name__=='__main__':main()
