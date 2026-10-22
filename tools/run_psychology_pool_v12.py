"""Render fresh 20 s transitions using separate voice, syntax and entry evidence.

Never reads human judgments. Frozen historical geometries are exclusion only.
All scores are engineering diagnostics, not perceptual ratings.
"""
import argparse,gzip,json,sys,wave,heapq
from collections import Counter
from functools import lru_cache
from pathlib import Path
import numpy as np
from v12_runtime import ROOT,syntax_factory
from musicmix.evidence_pipeline import EvidencePipeline
from musicmix.expectation_pipeline import candidate_from_dict
from musicmix.audition_history import exposure
from musicmix.ingest import sha256_file
from musicmix.intent_render import render_intent
from musicmix.audio import load_audio
from musicmix.mix_quality import evaluate_mix_quality
from musicmix.transition_validation import validate_transition_event
from run_psychology_pool_v11 import natural_reference

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--search',type=Path,default=ROOT/'runs/pool_v12_role_search')
    p.add_argument('--output',type=Path,default=ROOT/'runs/psychology_v12_release');a=p.parse_args();a.output=a.output.resolve();a.search=a.search.resolve()
    if a.output.exists():raise FileExistsError('Preserve earlier run')
    a.output.mkdir();(a.output/'audio').mkdir()
    catpath=ROOT/'runs/pool_v12/catalog_ready.json';catalog=read(catpath)
    history_path=ROOT/'runs/audition_history_v11.json';history=read(history_path)
    if history.get('unresolved'):raise ValueError('History geometry unresolved')
    searchmeta=read(a.search/'manifest.json')
    if searchmeta['inputs']['catalog_sha256']!=sha256_file(catpath) or searchmeta['search_sha256']!=sha256_file(a.search/'search.json.gz'):raise ValueError('Search changed')
    with gzip.open(a.search/'search.json.gz','rt',encoding='utf8') as f:proposals=json.load(f)['proposals']
    code={p.name:sha256_file(p) for p in (ROOT/'src/musicmix').glob('*.py')}
    engine=EvidencePipeline(catalog,pretrained_vad=True,syntax_factory=syntax_factory,cache_dir=ROOT/'runs/vocal_features_v12')
    engine.syntax_at=lru_cache(maxsize=None)(engine.syntax_at)
    entries=engine.entries;counts=Counter();events=Counter();pool=[]
    with gzip.open(a.output/'assessments.jsonl.gz','wt',encoding='utf8') as f:
        for i,proposal in enumerate(proposals):
            c=candidate_from_dict(proposal['candidate']);counts['proposals']+=1
            heard=exposure(c,history['regions'],8.0,sample_rate=48000)
            if heard['outgoing_region_heard'] or heard['incoming_region_heard']:
                counts['history_excluded']+=1;continue
            if any(not 480000<=cue<=entries[t]['duration_samples']-480000 for t,cue in [(c.track_a,c.cue_out_sample),(c.track_b,c.cue_in_sample)]):
                counts['context_excluded']+=1;continue
            r=engine.assess(c,proposal['score']);counts['assessed']+=1;events[r['functional_event']['label']]+=1
            row={'candidate':proposal['candidate'],'rank_key':r['rank_key'],'eligible':r['eligible'],
                 'baseline_score':proposal['score'],'event':r['functional_event']['label'],
                 'syntax_status':r['syntax_A'].get('syntax_status'),'voice_bottleneck':r['voice_bottleneck']}
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            if r['eligible']:
                counts['eligible']+=1
                item=(tuple(r['rank_key']),c.candidate_id,row)
                if len(pool)<3000:heapq.heappush(pool,item)
                elif item[:2]>pool[0][:2]:heapq.heapreplace(pool,item)
            if counts['assessed']%250==0:print(json.dumps(dict(counts)),flush=True)
    chosen=[];used=[];pairs=set();roles=Counter()
    for rank,identity,row in sorted(pool,reverse=True):
        c=candidate_from_dict(row['candidate']);coords=[(c.track_a,c.cue_out_sample),(c.track_b,c.cue_in_sample)]
        pair=c.track_a,c.track_b
        if pair in pairs or roles[(c.track_a,'out')]>=2 or roles[(c.track_b,'in')]>=2:continue
        if any(t==pt and abs(s-ps)<16*48000 for t,s in coords for pt,ps in used):continue
        chosen.append(row);used.extend(coords);pairs.add(pair);roles[(c.track_a,'out')]+=1;roles[(c.track_b,'in')]+=1
        if len(chosen)>=6:break
    artifacts=[];items=[];references=[]
    for number,row in enumerate(chosen,1):
        c=candidate_from_dict(row['candidate']);result=engine.assess(c,row['baseline_score'],include_profiles=True)
        title=entries[c.track_a]['title']+' → '+entries[c.track_b]['title']
        print('Render '+str(number)+' '+title,flush=True)
        write(a.output/f'selected_{number:03d}.json',result)
        for variant,match in [('main',True),('same_cue_no_level_match',False)]:
            path=a.output/'audio'/f'audio_{number:03d}_{variant}.wav'
            artifact=render_intent(engine.track(c.track_a),engine.track(c.track_b),c,path,context_seconds=10,
                                   level_match=match,incoming_role='arrival',max_boost_db=0)
            artifact['incoming']['semantic_arrival_supported']=False
            actual=load_audio(path)
            if actual.duration_samples!=960000 or artifact['join_sample']!=480000:raise ValueError('20 s playback contract')
            quality=evaluate_mix_quality(actual,480000)
            event=dict(row['candidate'],sample_rate=48000,output_join_sample=480000,intent='handoff')
            validation=validate_transition_event(event,artifact,quality,phrase=result['word_guard']['A'])
            if validation['hard_defects']:raise ValueError(validation['hard_defects'])
            artifacts.append({'candidate':row['candidate'],'variant':variant,'artifact':artifact,'validation':validation})
            items.append({'id':f'v12_audio_{number:03d}_{variant}','title':title+(' — controllo volume' if not match else ''),
                'audio_path':path.relative_to(ROOT).as_posix(),'candidate_id':c.candidate_id,
                'A_out':{'track':entries[c.track_a]['title'],'seconds':c.cue_out_sample/48000},
                'B_in':{'track':entries[c.track_b]['title'],'seconds':c.cue_in_sample/48000},
                'action':'execute','intent':result['functional_event']['label'],
                'notes':['V12: 20 secondi, cambio a 10 s.',
                 'Voce separata, parola, sintassi e ingresso B analizzati separatamente.',
                 'Qualità percettiva e risoluzione psicologica non certificate.']})
        path=a.output/'audio'/f'audio_{number:03d}_natural_A.wav'
        references.append(natural_reference(engine.track(c.track_a),c.cue_out_sample,path,10))
        items.append({'id':f'v12_audio_{number:03d}_natural_A','title':entries[c.track_a]['title']+' — continuazione originale',
            'audio_path':path.relative_to(ROOT).as_posix(),'candidate_id':c.candidate_id,
            'A_out':{'track':entries[c.track_a]['title'],'seconds':c.cue_out_sample/48000},
            'B_in':{'track':entries[c.track_a]['title'],'seconds':c.cue_out_sample/48000},
            'action':'stay','intent':'native_reference','notes':['20 s. Confronto con la continuazione originale, senza voto automatico.']})
    if code!={p.name:sha256_file(p) for p in (ROOT/'src/musicmix').glob('*.py')}:raise RuntimeError('Implementation changed during run')
    write(a.output/'manifest.json',{'schema_version':'musicmix.psychology-pool.2','status':'diagnostic_rendered' if chosen else 'abstained',
        'counts':dict(counts),'events':dict(events),'selected':len(chosen),'ordered_pairs_searched':42,
        'artifacts':artifacts,'natural_references':references,'provider_records':engine.model_records,'tracks':catalog['tracks'],
        'inputs':{'catalog_sha256':sha256_file(catpath),'history_sha256':sha256_file(history_path),
                  'search_sha256':sha256_file(a.search/'search.json.gz')},'code_sha256':code,
        'parameters':{'context_seconds':10,'clip_seconds':20,'history_exclusion_seconds':8,'new_region_spacing_seconds':16,
                      'human_labels_used_for_fitting':False,'new_model_training':False},
        'limitations':['Uncalibrated ranking; no claim of 10/10, cross-song psychological resolution or SOTA.',
                       'SSM-Net proposes structure; Silero VAD and spaCy were not trained on rap phrase completion.']})
    items.sort(key=lambda item:(not item['id'].endswith('_main'),item['id']))
    write(a.output/'review_manifest.json',{'schema_version':'musicmix.psychology-review.1','title':'MusicMix V12 — passaggi nuovi, 20 secondi','items':items})
    print(json.dumps({'selected':len(chosen),'counts':dict(counts)}),flush=True)
if __name__=='__main__':main()
