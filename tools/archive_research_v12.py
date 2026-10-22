"""Archive agent public research with byte hashes and explicit reading limits."""
import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MAPPING={'functional_event':'01_psicologia_aspettativa','vocal_completion':'02_fraseggio_rap_voce',
 'role_planning':'03_struttura_transizioni_dj','ssmnet_official':'04_modelli_preaddestrati',
 'critic_research':'06_valutazione_autofeedback','official_tension':'06_valutazione_autofeedback',
 'external_benchmark':'07_dati_benchmark'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    records=[]
    for worker,section in MAPPING.items():
        source=ROOT/'runs/result_swarm_v12/agents'/worker
        dest=ROOT/'conoscenza_mix'/section/('v12_'+worker)
        if dest.exists():raise FileExistsError(dest)
        dest.mkdir(parents=True)
        shutil.copytree(source/'research',dest/'research')
        for name in ('REPORT.md','RESPONSE.md'):
            if (source/name).is_file():shutil.copy2(source/name,dest/name)
        files=[]
        for f in sorted(dest.rglob('*')):
            if f.is_file():files.append({'path':str(f.relative_to(dest)).replace('\\','/'),'sha256':sha(f),'bytes':f.stat().st_size})
        record={'agent':worker,'folder':str(dest.relative_to(ROOT/'conoscenza_mix')).replace('\\','/'),
                'files':files,'limits':'Archival hash inventory, not a claim that every fetched file is a valid paper or fully read. Read-depth and failures remain in each original sources metadata.'}
        (dest/'ARCHIVE.json').write_text(json.dumps(record,indent=2),encoding='utf8');records.append(record)
    out=ROOT/'conoscenza_mix/_meta/ARCHIVE_V12.json'
    out.write_text(json.dumps({'schema':'musicmix.public-research-archive.1','date':'2026-10-20','bundles':records},indent=2),encoding='utf8')
    print(json.dumps({'bundles':len(records),'files':sum(len(r['files']) for r in records),'bytes':sum(x['bytes'] for r in records for x in r['files'])}))
if __name__=='__main__':main()
