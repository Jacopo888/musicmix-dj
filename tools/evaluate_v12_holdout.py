"""Measure fixed V12 choices against the frozen V10 verbal feedback.

The agent implementations never receive these labels. This small diagnostic
is reported with abstention and common-coverage views, never as SOTA evidence.
"""
from dataclasses import asdict
import argparse,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
# Optional cached providers: append, retaining the already tested main Python
# numerical packages. This exposes no credentials or external service.
for p in (ROOT/'.cache/structure_runtime/Lib/site-packages',ROOT/'.cache/structure_runtime/packages',
          ROOT/'runs/result_swarm_v12/agents/vocal_completion/runtime'):
    if p.is_dir():sys.path.append(str(p))

from musicmix.contracts import Candidate
from musicmix.evidence_pipeline import EvidencePipeline
from musicmix.ingest import sha256_file
from musicmix.paths import localize
from v12_runtime import syntax_factory

def compare(positive,negative):return 1.0 if positive>negative else (.5 if positive==negative else 0.0)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,default=ROOT/'runs/pool_v10/catalog_ready.json')
    p.add_argument('--output',type=Path,default=ROOT/'runs/evaluation_v12_holdout')
    p.add_argument('--no-pretrained-vad',action='store_true')
    args=p.parse_args();out=args.output.resolve()
    if out.exists():raise FileExistsError('Keep frozen evaluation outputs')
    out.mkdir(parents=True)
    plan=ROOT/'runs/result_swarm_v12/EVALUATION_PLAN.json'
    feedback=ROOT.parent/'giudizi json/feedback_verbale_V10_review_8_audio.json'
    labels=json.loads(feedback.read_text(encoding='utf8'))
    gallery=localize(labels['gallery_manifest'])
    if sha256_file(gallery)!=labels['gallery_manifest_sha256']:raise ValueError('Gallery changed')
    metadata=json.loads(gallery.read_text(encoding='utf8'))
    rows={x['audio_id']:x for x in metadata['audio']}
    catalog=json.loads(args.catalog.read_text(encoding='utf8'))
    engine=EvidencePipeline(catalog,pretrained_vad=not args.no_pretrained_vad,syntax_factory=syntax_factory,cache_dir=ROOT/'runs/vocal_features_v12')
    records=[]
    code={p.name:sha256_file(p) for p in (ROOT/'src/musicmix').glob('*.py')}
    for observation in labels['observations']:
        identity=observation['audio_id'];row=rows[identity];pair=row['pair']
        if sha256_file(localize(observation['source']))!=observation['audio_sha256']:raise ValueError('Feedback audio changed')
        c=Candidate(row['candidate_id'],pair['outgoing_track_id'],pair['incoming_track_id'],
                    pair['cue_out_sample'],pair['cue_in_sample'],'handoff',{}, {})
        print('Evaluate frozen '+identity,flush=True)
        result=engine.assess(c,row['score'],include_profiles=True)
        result['audio_id']=identity
        # Label attached only after prediction; never supplied to the engine.
        result['reference_label']='acceptable' if identity=='audio_003' else ('ambiguous_mid' if identity=='audio_001' else 'rejected')
        (out/f'{identity}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
        records.append(result)
    positive=next(x for x in records if x['audio_id']=='audio_003')
    negative=[x for x in records if x['reference_label']=='rejected']
    ranking=sorted(records,key=lambda x:tuple(x['rank_key']),reverse=True)
    old=sorted(records,key=lambda x:x['baseline_score'],reverse=True)
    common=[x for x in negative if x['functional_event']['score'] is not None]
    measurements={
      'positive_rank_before':next(i+1 for i,x in enumerate(old) if x['audio_id']=='audio_003'),
      'positive_rank_after':next(i+1 for i,x in enumerate(ranking) if x['audio_id']=='audio_003'),
      'pairwise_before':sum(compare(positive['baseline_score'],x['baseline_score']) for x in negative)/len(negative),
      'pairwise_after':sum(compare(tuple(positive['rank_key']),tuple(x['rank_key'])) for x in negative)/len(negative),
      'positive_eligible':positive['eligible'],'negative_eligible_count':sum(x['eligible'] for x in negative),
      'evaluated':len(records),'functional_coverage':sum(x['functional_event']['score'] is not None for x in records),
      'common_negative_count':len(common),
      'common_pairwise_after':None if positive['functional_event']['score'] is None or not common else sum(compare(tuple(positive['rank_key']),tuple(x['rank_key'])) for x in common)/len(common)}
    if code!={p.name:sha256_file(p) for p in (ROOT/'src/musicmix').glob('*.py')}:raise RuntimeError('Code changed during evaluation')
    report={'schema_version':'musicmix.v12-holdout-diagnostic.1','measurements':measurements,
       'ranking_before':[x['audio_id'] for x in old],'ranking_after':[x['audio_id'] for x in ranking],
       'cases':[{'id':x['audio_id'],'label':x['reference_label'],'event':x['functional_event']['label'],
                 'event_score':x['functional_event']['score'],'eligible':x['eligible'],'rank_key':x['rank_key']} for x in records],
       'inputs':{'catalog_sha256':sha256_file(args.catalog),'feedback_sha256':sha256_file(feedback),'plan_sha256':sha256_file(plan)},
       'code_sha256':code,'provider_records':engine.model_records,
       'limits':['One acceptable clip, six explicit rejections, one ambiguous mid: a small diagnostic, not independent population validation.',
                 'Unknown context can cause abstention; common-coverage view is reported separately.',
                 'No threshold fitting to these labels; DSP/engineering ranks are not quality probabilities.']}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(measurements),flush=True)

if __name__=='__main__':main()
