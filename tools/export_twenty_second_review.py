"""Restore 20-second playback from immutable longer diagnostics, preserving cues."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import wave
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from musicmix.paths import localize

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'runs/psychology_v11_release')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/psychology_v11_20s')
    args=parser.parse_args();source=args.source.resolve();out=args.output.resolve()
    if out.exists(): raise FileExistsError('Preserve existing exports')
    review=json.loads((source/'review_manifest.json').read_text(encoding='utf8'))
    run=json.loads((source/'manifest.json').read_text(encoding='utf8'))
    joins={localize(x['artifact']['path']).name:x['artifact']['join_sample'] for x in run['artifacts']}
    joins.update({localize(x['path']).name:x['reference_sample'] for x in run['natural_references']})
    out.mkdir(parents=True);(out/'audio').mkdir();items=[];evidence=[]
    for item in review['items']:
        original=ROOT/item['audio_path'];join=joins[original.name]
        with wave.open(str(original),'rb') as wav:
            rate,channels,width=wav.getframerate(),wav.getnchannels(),wav.getsampwidth()
            if width!=2: raise ValueError('PCM16 source required')
            start=join-10*rate;frames=20*rate
            if start<0 or start+frames>wav.getnframes(): raise ValueError('Insufficient context')
            wav.setpos(start);pcm=np.frombuffer(wav.readframes(frames),dtype='<i2').reshape(-1,channels).copy()
        edge=round(.01*rate);ramp=np.linspace(0,1,edge)[:,None]
        pcm[:edge]=np.rint(pcm[:edge].astype(float)*ramp).astype('<i2')
        pcm[-edge:]=np.rint(pcm[-edge:].astype(float)*ramp[::-1]).astype('<i2')
        identity='v11_'+item['id']+'_20s';dest=out/'audio'/f'{identity}.wav'
        with dest.open('xb') as raw:
            with wave.open(raw,'wb') as wav:
                wav.setnchannels(channels);wav.setsampwidth(width);wav.setframerate(rate);wav.writeframes(pcm.tobytes())
        notes=[n for n in item.get('notes',[]) if not n.startswith('Cambio a ')]
        notes.insert(0,'20 secondi; cambio al secondo 10. Stessi punti della proposta V11, durata ripristinata.')
        items.append(dict(item,id=identity,audio_path=dest.relative_to(ROOT).as_posix(),notes=notes))
        evidence.append({'id':identity,'source_path':str(original),'source_sha256':sha(original),
                         'source_start_sample':start,'source_join_sample':join,'sample_rate':rate,
                         'duration_samples':len(pcm),'join_sample':10*rate,'path':str(dest),'sha256':sha(dest),
                         'only_added_processing':'10ms outer ramps; original transition and gain unchanged'})
    (out/'review_manifest.json').write_text(json.dumps({'schema_version':'musicmix.psychology-review.1','items':items},ensure_ascii=False,indent=2),encoding='utf8')
    (out/'manifest.json').write_text(json.dumps({'schema_version':'musicmix.twenty-second-export.1','source_manifest_sha256':sha(source/'manifest.json'),'artifacts':evidence,'musical_improvement_claim':False},indent=2),encoding='utf8')
    print(json.dumps({'status':'exported','clips':len(items),'seconds':20,'join_seconds':10}))

if __name__=='__main__':main()
