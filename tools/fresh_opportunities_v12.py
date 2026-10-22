"""Exclude already presented cue regions before costly pair enumeration.

The extra 80 ms covers the existing maximum alignment shift. Final rendered
coordinates are independently checked again against the original 8 s rule.
"""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=ROOT/'runs/opportunities_v12/registry.json';historypath=ROOT/'runs/audition_history_v11.json'
    target=source.with_name('registry_fresh.json')
    if target.exists():raise FileExistsError(target)
    r=read(source);h=read(historypath)
    if h.get('unresolved'):raise ValueError('Unresolved listening history')
    heard={}
    for row in h['regions']:
        for t,s in [(row['track_a'],row['cue_out_sample']),(row['track_b'],row['cue_in_sample'])]:heard.setdefault(t,[]).append(s)
    counts={}
    for e in r['tracks']:
        before=len(e['cues']);prior=heard.get(e['track_id'],[])
        e['cues']=[c for c in e['cues'] if all(abs(c['sample']-s)>round(8.08*48000) for s in prior)]
        counts[e['slug']]={'before':before,'fresh':len(e['cues'])}
    r['history_prefilter']={'source_sha256':sha(source),'history_sha256':sha(historypath),'exclusion_seconds':8.08,'counts':counts}
    target.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(counts))
if __name__=='__main__':main()
