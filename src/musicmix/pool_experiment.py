"""Multi-song matched-context listening pilot with render-level feedback."""
from collections import Counter,defaultdict
from dataclasses import asdict
from html import escape
import gzip
import json
import re
from pathlib import Path
import shutil
import uuid

from .audio import load_audio
from .audition_history import exposure
from .expectation import analyze_expectation,rerank_expectation,ExpectationThresholds
from .expectation_pipeline import candidate_from_dict
from .ingest import sha256_file
from .listening import make_listening_test
from .mix_quality import evaluate_mix_quality
from .musical_preview import render_handoff
from .paths import localize
from .pooled_trajectory import PooledTrajectory


def label_audio_source(html,filename,caption):
    marker=f'data-source-label-for="{escape(filename,quote=True)}"'
    if marker in html:return html
    pattern=r'<audio\b[^>]*\bsrc="'+re.escape(filename)+r'"[^>]*>'
    result,count=re.subn(pattern,lambda match:f'<small {marker}>{escape(caption)}</small><br>'+match.group(0),html,count=1)
    if count!=1:raise ValueError('Listening audio tag not found for source caption')
    return result


def select_pool_trials(rows,*,maximum=4,spacing_seconds=8,source_cap=1,incoming_cap=3,rate=48000):
    groups=defaultdict(list)
    for row in rows:groups[(row['track_a'],row['cue_out_sample'])].append(row)
    options=[];counts=Counter()
    def different(left,right):
        return (left['track_b']!=right['track_b'] or
                abs(left['cue_in_sample']-right['cue_in_sample'])>=spacing_seconds*rate)
    for (source,cue),group in groups.items():
        baseline=max(group,key=lambda r:(r['baseline_score'],r['candidate_id']))
        choice=max(group,key=lambda r:(r['adjusted_score'],r['baseline_score'],r['candidate_id']))
        kind='expectation_rerank'
        if not different(baseline,choice):
            alternatives=[r for r in group if r['track_b']!=baseline['track_b']]
            if not alternatives:
                counts['no_distinct_incoming_song']+=1;continue
            choice=max(alternatives,key=lambda r:(r['adjusted_score'],r['candidate_id']))
            kind='pool_alternative_exploration'
        options.append({'source_track':source,'outgoing_sample':cue,'baseline':baseline,'choice':choice,
                        'kind':kind,'adjusted_gain':choice['adjusted_score']-baseline['adjusted_score']})
    options.sort(key=lambda t:(t['kind']!='expectation_rerank',-t['adjusted_gain'],
                               -t['baseline']['baseline_score'],t['source_track'],t['outgoing_sample']))
    if source_cap!=1:raise ValueError('Matched-context trial design requires one trial per outgoing song')
    by_source=defaultdict(list)
    for option in options:by_source[option['source_track']].append(option)
    order=sorted(by_source,key=lambda source:(len(by_source[source]),source))
    selected=[];best_objective=(-1,-1,float('-inf'),float('-inf'));nodes=0;truncated=False
    used=defaultdict(list);incoming=Counter();chosen=[]
    def visit(index):
        nonlocal selected,best_objective,nodes,truncated
        if nodes>=250000:truncated=True;return
        nodes+=1
        objective=(len(chosen),sum(t['kind']=='expectation_rerank' for t in chosen),
                   sum(t['adjusted_gain'] for t in chosen),sum(t['baseline']['baseline_score'] for t in chosen))
        if objective>best_objective:best_objective=objective;selected=list(chosen)
        if index>=len(order) or len(chosen)>=maximum:return
        if len(chosen)+len(order)-index<len(selected):return
        for option in by_source[order[index]]:
            arrivals=[option[s] for s in ('baseline','choice')]
            demand=Counter(r['track_b'] for r in arrivals)
            if any(incoming[t]+n>incoming_cap for t,n in demand.items()):continue
            positions=[(option['source_track'],option['outgoing_sample'])]+[(r['track_b'],r['cue_in_sample']) for r in arrivals]
            if any(abs(sample-prior)<spacing_seconds*rate for track,sample in positions for prior in used[track]):continue
            chosen.append(option);incoming.update(demand)
            for track,sample in positions:used[track].append(sample)
            visit(index+1)
            for track,sample in reversed(positions):used[track].pop()
            incoming.subtract(demand);chosen.pop()
        visit(index+1)
    visit(0)
    selected.sort(key=lambda t:(t['kind']!='expectation_rerank',-t['adjusted_gain'],t['source_track']))
    sources=Counter(t['source_track'] for t in selected)
    incoming=Counter(t[s]['track_b'] for t in selected for s in ('baseline','choice'))
    counts['unselected_comparison_options']=len(options)-len(selected)
    return selected,{'outgoing_groups':len(groups),'comparison_options':len(options),
                     'selected_trials':len(selected),'skipped':dict(counts),
                     'source_role_counts':dict(sources),'incoming_role_counts':dict(incoming),
                     'selection_strategy':'maximize_source_coverage_then_evidence_with_bounded_backtracking',
                     'search_nodes':nodes,'search_truncated':truncated}


def run_pool_experiment(catalog_path:Path,history_path:Path,output:Path,*,maximum=4,cue_policy='beats',max_candidates_per_pair=200000,reuse_run=None):
    from .pool_search import search_pool
    from .section_novelty import analyze_section_novelty,describe_cue_section
    catalog=json.loads(catalog_path.read_text(encoding='utf-8'))
    history=json.loads(history_path.read_text(encoding='utf-8'))
    if catalog['status']!='ready' or history.get('unresolved'):
        raise ValueError('Require complete source/evidence catalog and resolved listening history')
    output=output.resolve();output.mkdir(parents=True,exist_ok=False);(output/'inputs').mkdir()
    for name,source in [('catalog',catalog_path),('history',history_path)]:
        shutil.copyfile(source,output/'inputs'/f'{name}.json')
    tracks={};structures={};vocals={};names={}
    for entry in catalog['tracks']:
        track=load_audio(entry['canonical_path'])
        if track.track_id!=entry['track_id'] or track.sample_rate!=48000:raise ValueError('Canonical binding changed')
        tracks[track.track_id]=track;names[track.track_id]=entry['artist']+' — '+entry['title']
        for kind,target in [('structure',structures),('vocals',vocals)]:
            source=localize(entry[kind+'_path'])
            if sha256_file(source)!=entry[kind+'_sha256']:raise ValueError('Provider changed since catalog')
            target[track.track_id]=json.loads(source.read_text(encoding='utf-8'))
    reuse_provenance=None
    if reuse_run is None:
        print(json.dumps({'event':'searching_pool','tracks':len(tracks),'ordered_pairs':len(tracks)*(len(tracks)-1)}),flush=True)
        pool=search_pool(tracks,structures,vocals,top_k=8,max_per_pair=4,
                         cue_policy=cue_policy,max_candidates_per_pair=max_candidates_per_pair,heard_regions=history['regions'])
    else:
        reuse_run=Path(reuse_run);previous=json.loads((reuse_run/'manifest.json').read_text(encoding='utf-8'))
        if (previous['inputs']['catalog_sha256']!=sha256_file(output/'inputs/catalog.json')
            or previous['parameters']['cue_policy']!=cue_policy
            or previous['parameters']['max_candidates_per_pair']!=max_candidates_per_pair):
            raise ValueError('Cached search catalog/policy mismatch')
        old_history=json.loads((reuse_run/'inputs/history.json').read_text(encoding='utf-8'))
        def geometry(region):return tuple(region[k] for k in ('track_a','track_b','cue_out_sample','cue_in_sample','sample_rate'))
        if not {geometry(r) for r in old_history['regions']}.issubset({geometry(r) for r in history['regions']}):
            raise ValueError('A cached filtered search supports stricter history only')
        with gzip.open(reuse_run/'search.json.gz','rt',encoding='utf-8') as handle:pool=json.load(handle)
        retained=[];additional=0
        for proposal in pool['proposals']:
            flags=exposure(candidate_from_dict(proposal['candidate']),history['regions'],8)
            if flags['outgoing_region_heard'] or flags['incoming_region_heard']:additional+=1
            else:retained.append(proposal)
        pool['proposals']=retained
        from .directional import DirectionalProposal
        from .pool_search import _select_globally
        objects=[DirectionalProposal(candidate_from_dict(p['candidate']),p['score'],p['breakdown'],p['raw_features'],
                    p['fade_seconds'],p['alignment'],tuple(p.get('reasons',()))) for p in retained]
        selected,rejections,pair_counts,sources,incoming=_select_globally(objects,top_k=8,max_per_pair=4,
                      role_cap=2,cue_spacing_seconds=8,tracks=tracks)
        pool['selected']=[dict(asdict(p),pair_id=f'{p.candidate.track_a}->{p.candidate.track_b}') for p in selected]
        pool['diagnostics'].update(admitted_unique_count=len(retained),heard_region_count=len(history['regions']),
             heard_rejected_count=pool['diagnostics']['heard_rejected_count']+additional,
             additional_history_rejected_count=additional,selected_count=len(selected),selection_shortfall=8-len(selected),
             selected_count_lt_top_k=len(selected)<8,selected_source_role_counts=sources,
             selected_incoming_role_counts=incoming,pair_selected_counts=pair_counts,selection_rejection_counts=rejections)
        pool['rejection_counts']['global_by_reason']=dict(rejections,
             heard_regions_removed_total=pool['diagnostics']['heard_rejected_count'],
             duplicate_admitted_geometry_count=pool['diagnostics']['duplicate_admitted_geometry_count'])
        for pair_id,stats in pool['all_pair_counts'].items():
            stats['selected_count']=pair_counts.get(pair_id,0)
            stats['eligible_after_history_count']=sum(p['pair_id']==pair_id for p in retained)
        reuse_provenance={'manifest':str((reuse_run/'manifest.json').resolve()),
                          'manifest_sha256':sha256_file(reuse_run/'manifest.json'),
                          'search_sha256':sha256_file(reuse_run/'search.json.gz'),
                          'additional_history_rejected':additional}
        print(json.dumps({'event':'reused_frozen_search_with_stricter_history','retained':len(retained),'removed':additional}),flush=True)
    with gzip.open(output/'search.json.gz','wt',encoding='utf-8') as handle:json.dump(pool,handle,ensure_ascii=False)
    print(json.dumps({'event':'pool_searched','admitted_unique':len(pool['proposals']),
                      'pair_count':pool['pair_count']}),flush=True)
    sections={};series={}
    for track_id,track in tracks.items():
        print(json.dumps({'event':'caching_track_descriptors','track':names[track_id]}),flush=True)
        sections[track_id]=analyze_section_novelty(track,structures[track_id])
        series[track_id]=PooledTrajectory(track,structures[track_id]['beats'])
    profiles={};records=[];proposals={};labels=Counter()
    def profile(track_id,original,effective,side):
        key=f'{track_id}:{side}:{original}:{effective-original}'
        if key not in profiles:profiles[key]=series[track_id].profile(original,side=side,cue_shift_samples=effective-original)
        return profiles[key],key
    for proposal in pool['proposals']:
        c=proposal['candidate'];original=proposal['raw_features']['original_structure_cue_samples']
        a,ka=profile(c['track_a'],original['a'],c['cue_out_sample'],'outgoing')
        natural,kn=profile(c['track_a'],original['a'],c['cue_out_sample'],'incoming')
        b,kb=profile(c['track_b'],original['b'],c['cue_in_sample'],'incoming')
        assessment=analyze_expectation(a,natural,b);rank=rerank_expectation(assessment,proposal['score'],weight=.20)
        row={key:c[key] for key in ('candidate_id','track_a','track_b','cue_out_sample','cue_in_sample')}
        row.update(baseline_score=proposal['score'],adjusted_score=rank.adjusted_score,
                   expectation_score=rank.expectation_score,label=assessment.label,
                   evidence=[asdict(e) for e in assessment.evidence],missing=assessment.missing,
                   profiles={'a_before':ka,'a_natural_after':kn,'b_after':kb},
                   section_a=describe_cue_section(sections[c['track_a']],c['cue_out_sample']),
                   section_b=describe_cue_section(sections[c['track_b']],c['cue_in_sample']))
        records.append(row);proposals[c['candidate_id']]=proposal;labels[assessment.label]+=1
    bad=set();rendered={};quality={};attempts=[];trials=[];selection={}
    # Feedback is a bounded rejection/refill loop over the frozen candidate
    # pool. Only measured clipping/empty renders are hard failures.
    for iteration in range(9):
        trials,selection=select_pool_trials([r for r in records if r['candidate_id'] not in bad],maximum=maximum)
        new_bad=set()
        for trial in trials:
            for side in ('baseline','choice'):
                row=trial[side];identity=row['candidate_id']
                if identity in rendered:continue
                candidate=candidate_from_dict(proposals[identity]['candidate'])
                artifact=render_handoff(tracks[candidate.track_a],tracks[candidate.track_b],candidate,output/'audio'/f'{identity}.wav')
                qa=evaluate_mix_quality(load_audio(artifact['path']),artifact['join_sample'])
                hard=artifact['clipped_samples']>0 or qa['metrics']['clipping']['clipped_sample_count']>0
                attempts.append({'candidate_id':identity,'iteration':iteration,'hard_rejected':hard,
                                 'reason':'measured_clipping' if hard else None,'artifact':artifact,'quality':qa})
                rendered[identity]=artifact;quality[identity]=qa
                if hard:new_bad.add(identity)
        if not new_bad:break
        bad.update(new_bad)
    else:raise RuntimeError('Technical-feedback iteration limit reached')
    artifacts=[];pairs=[]
    for number,trial in enumerate(trials,1):
        for side in ('baseline','choice'):
            row=trial[side];artifact=dict(rendered[row['candidate_id']],trial=number,system=side)
            artifacts.append(artifact)
            (output/f'proposal_{number:02d}_{side}.json').write_text(json.dumps(proposals[row['candidate_id']],indent=2),encoding='utf-8')
        pairs.append({'pair_id':f'pool_{number:02d}','left_path':rendered[trial['choice']['candidate_id']]['path'],
                      'right_path':rendered[trial['baseline']['candidate_id']]['path'],
                      'left_system':'expectation_pool_v8' if trial['kind']=='expectation_rerank' else 'pool_alternative_v8',
                      'right_system':'directional_pool_baseline_v8'})
    run_id=str(uuid.uuid4())
    policy_title='tutti_beat' if cue_policy=='beats' else 'downbeat'
    listening=make_listening_test(output/'listening',pairs,run_id=run_id,title=f'Pool_{len(tracks)}_brani_V8_{policy_title}',
                                 no_repeated_audio=True,collect_motion=True)
    key=json.loads((output/'listening/key.json').read_text(encoding='utf-8'))
    page=Path(listening['path']);html=page.read_text(encoding='utf-8')
    html=html.replace('<h1>','<p>Ogni confronto mantiene gli stessi primi 10 secondi. Dopo il raccordo '
          'a 10 s può cambiare anche il brano di ingresso. Valuta continuità, andamento e parole tagliate.</p><h1>',1)
    timestamp_rows=['# Tagli del pool V8','','Raccordo a 10 secondi nel player. Tempi riferiti alle sorgenti canoniche.',
                    '', '| Prova | Player | Uscita | Ingresso |','|---|---|---|---|']
    by_path={artifact['path']:artifact for artifact in artifacts}
    for item in key['trials']:
        artifact=by_path[item['source']];row=trials[item['trial']-1][artifact['system']]
        title=f"{names[row['track_a']]} ({row['cue_out_sample']/48000:.3f} s) → {names[row['track_b']]} ({row['cue_in_sample']/48000:.3f} s)"
        html=label_audio_source(html,f'trial_{item["trial"]:02d}_{item["label"]}.wav',title)
        timestamp_rows.append(f"| {item['trial']} | {item['label']} | {names[row['track_a']]} — {row['cue_out_sample']/48000:.3f} s | {names[row['track_b']]} — {row['cue_in_sample']/48000:.3f} s |")
    page.write_text(html,encoding='utf-8');(output/'TIMESTAMPS.md').write_text('\n'.join(timestamp_rows)+'\n',encoding='utf-8')
    def write(name,payload):
        (output/name).write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    write('scores.json',records);write('quality_feedback.json',attempts);write('section_novelty.json',sections)
    with gzip.open(output/'trajectory_profiles.json.gz','wt',encoding='utf-8') as handle:json.dump(profiles,handle)
    report={'schema_version':'musicmix.pool-run.1','run_id':run_id,'status':'ready_for_listening' if pairs else 'no_trials',
            'tracks':catalog['tracks'],'track_names':names,'artifacts':artifacts,'comparisons':trials,
            'listening':listening,'selection':selection,'pool_diagnostics':pool['diagnostics'],
            'pair_count':pool['pair_count'],'all_pair_counts':pool['all_pair_counts'],
            'expectation_counts':dict(labels),'profile_count':len(profiles),
            'feedback':{'iterations':len({a['iteration'] for a in attempts}),
                        'rejected_candidates':sorted(bad),'accepted_artifacts':len(artifacts),
                        'subjective_reward_used':False,'all_attempts':'quality_feedback.json'},
            'parameters':{'cue_policy':cue_policy,'max_candidates_per_pair':max_candidates_per_pair,
                          'motion_weight':.20,'thresholds':asdict(ExpectationThresholds()),
                          'section_novelty_used_for_ranking':False,'analysis_mode':'offline_reference',
                          'maximum_trials':maximum,'source_role_cap_trials':1,'incoming_role_cap_trials':3,
                          'cue_spacing_seconds':8,'hard_feedback_policy':'clipping_only; other flags descriptive'},
            'inputs':{'catalog_sha256':sha256_file(output/'inputs/catalog.json'),
                      'history_sha256':sha256_file(output/'inputs/history.json')},
            'reused_frozen_search':reuse_provenance,
            'code_sha256':{p.name:sha256_file(p) for p in Path(__file__).parent.glob('*.py')},
            'human_validation':'pending','limitations':['Source ASR gaps do not guarantee intact sung words.',
               'Novelty boundaries and continuation proxies do not establish semantic resolution.',
               'Alternative exploration trials collect next-song preferences; they do not prove a reranker improved.',
               'No SOTA or validated perceptual quality claim.']}
    write('manifest.json',report)
    print(json.dumps({'event':'complete','trials':len(pairs),'labels':dict(labels),'listening':listening['test_id']}),flush=True)
    return report
