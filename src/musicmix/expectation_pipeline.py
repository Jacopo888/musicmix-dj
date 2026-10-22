"""Controlled, experimental incoming-selection pilot on a frozen search pool."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import uuid

from .audio import load_audio
from .audition_history import exposure
from .contracts import Candidate, Feature
from .expectation import analyze_expectation, rerank_expectation, EXPECTATION_VERSION, ExpectationThresholds
from .ingest import sha256_file
from .listening import make_listening_test
from .musical_preview import render_handoff


def read_cache(path: Path) -> dict:
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


def candidate_from_dict(raw: dict) -> Candidate:
    data = dict(raw)
    for field in ('features', 'penalties'):
        data[field] = {key: Feature(**value) for key, value in data[field].items()}
    data['constraint_reasons'] = tuple(data.get('constraint_reasons', ()))
    return Candidate(**data)


def select_comparisons(records: list[dict], *, maximum=3, spacing_seconds=12.0,
                       incoming_separation_seconds=8.0, rate=48000) -> tuple[list[dict], dict]:
    """Same outgoing anchor, highest baseline versus changed weighted winner.

    The caller supplies only novel, admissible records. Equal winners and
    nearby incoming alternatives cannot create informative comparisons.
    Fewer than ``maximum`` is a legitimate outcome; no thresholds are relaxed.
    """
    groups = defaultdict(list)
    for record in records:
        groups[record['cue_out_sample']].append(record)
    options = []
    skipped = Counter()
    for outgoing, group in groups.items():
        baseline = max(group, key=lambda r: (r['baseline_score'], r['candidate_id']))
        policy = max(group, key=lambda r: (r['adjusted_score'], r['baseline_score'], r['candidate_id']))
        if abs(policy['cue_in_sample']-baseline['cue_in_sample']) < incoming_separation_seconds*rate:
            skipped['unchanged_or_nearby_winner'] += 1
            continue
        options.append({'outgoing_sample': outgoing, 'policy': policy, 'baseline': baseline,
                        'policy_gain': policy['adjusted_score']-baseline['adjusted_score']})
    options.sort(key=lambda p: (-p['policy_gain'], -p['policy']['baseline_score'], p['outgoing_sample']))
    selected = []
    for option in options:
        new_b = [option[s]['cue_in_sample'] for s in ('policy', 'baseline')]
        if any(abs(option['outgoing_sample']-p['outgoing_sample']) < spacing_seconds*rate
               or any(abs(b-p[s]['cue_in_sample']) < spacing_seconds*rate
                      for b in new_b for s in ('policy', 'baseline')) for p in selected):
            skipped['spacing_between_trials'] += 1
            continue
        if len(selected) >= maximum:
            skipped['maximum_trials'] += 1
            continue
        selected.append(option)
    return selected, {'outgoing_groups': len(groups), 'changed_winner_groups': len(options),
                      'selected_trials': len(selected), 'skipped': dict(skipped)}


def run_experiment(cache_path: Path, history_path: Path, output: Path, *, weight=.20,
                   maximum=3, novelty_seconds=8.0) -> dict:
    cache = read_cache(cache_path)
    history = json.loads(history_path.read_text(encoding='utf-8'))
    if cache.get('status') != 'complete' or history.get('unresolved'):
        raise ValueError('Require complete cache and resolved audition history')
    output.mkdir(parents=True, exist_ok=False)
    history_snapshot = output/'audition_history.json'
    history_snapshot.write_bytes(history_path.read_bytes())
    run_id = str(uuid.uuid4())
    profiles = cache['profile_cache']
    scores, entries, counts = [], cache['candidate_map'], Counter()
    unique = {}
    for candidate_id, entry in entries.items():
        c = candidate_from_dict(entry['baseline']['proposal']['candidate'])
        if not c.passes_constraints:
            raise ValueError('Cache contains an inadmissible candidate')
        if not all(v for k, v in entry['alignment_validation'].items() if k.endswith('matches_alignment')):
            raise ValueError('Cue alignment mismatch')
        assessment = analyze_expectation(*(profiles[entry['profiles'][k]]
                                           for k in ('a_before', 'a_natural_after', 'b_after')))
        rank = rerank_expectation(assessment, entry['baseline']['score'], weight=weight)
        novelty = exposure(c, history['regions'], novelty_seconds)
        novel = not (novelty['outgoing_region_heard'] or novelty['incoming_region_heard'])
        row = {'candidate_id': candidate_id, 'direction': entry['direction'],
               'track_a': c.track_a, 'track_b': c.track_b,
               'cue_out_sample': c.cue_out_sample, 'cue_in_sample': c.cue_in_sample,
               'baseline_score': rank.baseline_score, 'adjusted_score': rank.adjusted_score,
               'expectation_score': rank.expectation_score, 'label': assessment.label,
               'evidence': [asdict(e) for e in assessment.evidence], 'missing': assessment.missing,
               'novelty': novelty, 'eligible_novel': novel}
        scores.append(row)
        counts[f"{entry['direction']}.{assessment.label}"] += 1
        if novel:
            key = (entry['direction'], c.cue_out_sample, c.cue_in_sample)
            if key not in unique or row['baseline_score'] > unique[key]['baseline_score']:
                unique[key] = row
    tracks = {}
    for track_id, source in cache['source_records'].items():
        track = load_audio(source['path'])
        if track.track_id != track_id or track.sample_rate != 48000:
            raise ValueError('Canonical identity/rate mismatch')
        tracks[track_id] = track
    directions = {}
    for direction in ('forward', 'reverse'):
        pool = [r for r in unique.values() if r['direction'] == direction]
        if len({(r['track_a'], r['track_b']) for r in pool}) > 1:
            raise ValueError('This controlled pilot supports one track pair per direction')
        comparisons, selection = select_comparisons(pool, maximum=maximum)
        pairs, artifacts = [], []
        for index, comparison in enumerate(comparisons, 1):
            paths = {}
            for system in ('policy', 'baseline'):
                row = comparison[system]
                candidate = candidate_from_dict(entries[row['candidate_id']]['baseline']['proposal']['candidate'])
                path = output/direction/'audio'/f'{index:02d}_{system}.wav'
                artifact = render_handoff(tracks[candidate.track_a], tracks[candidate.track_b], candidate, path)
                if artifact['clipped_samples']:
                    raise ValueError('Clipped preview')
                artifacts.append({**artifact, 'system': system, 'trial': index})
                paths[system] = path
            pairs.append({'pair_id': f'{direction}_{index:02d}',
                          'left_path': str(paths['policy']), 'right_path': str(paths['baseline']),
                          'left_system': 'expectation_v7', 'right_system': 'local_baseline_v4'})
        title = ('Reminder_verso_Too_Many_Nights' if direction == 'forward'
                 else 'Too_Many_Nights_verso_Reminder') + '_V7_motion'
        listening = make_listening_test(output/direction/'listening', pairs, run_id=run_id,
                                        title=title, no_repeated_audio=True, collect_motion=True)
        html = Path(listening['path'])
        content = html.read_text(encoding='utf-8').replace('<h1>', '<p>In ogni confronto, i primi 10 secondi '
                    'sono identici: cambia il punto di ingresso del secondo brano. Il passaggio avviene '
                    'a 10 secondi. Valuta se il secondo brano continua, risolve o interrompe '
                    'l’aspettativa del primo. Puoi avanzare e tornare indietro.</p><h1>', 1)
        html.write_text(content, encoding='utf-8')
        directions[direction] = {'eligible_novel_pairs': len(pool), 'selection': selection,
                                 'comparisons': comparisons, 'artifacts': artifacts, 'listening': listening}
    report = {'schema_version': 'musicmix.expectation-experiment.1', 'run_id': run_id,
              'status': ('ready_for_listening' if any(v['comparisons'] for v in directions.values())
                         else 'no_informative_comparisons'),
              'version': EXPECTATION_VERSION, 'parameters': {'weight': weight, 'maximum_trials': maximum,
              'analysis_mode': 'offline_reference',
              'natural_future_reference': 'direction only; contributes to continuation score; not a causal live model',
              'thresholds': asdict(ExpectationThresholds()),
              'novelty_seconds_both_sources': novelty_seconds, 'spacing_seconds': 12,
              'incoming_separation_seconds': 8, 'renderer': 'anchor_handoff_v1'},
              'inputs': [{'path': str(p.resolve()), 'sha256': sha256_file(p)} for p in (cache_path, history_snapshot)],
              'code': {p.name: sha256_file(p) for p in [Path(__file__),
                       Path(__file__).with_name('expectation.py'), Path(__file__).with_name('trajectory.py'),
                       Path(__file__).with_name('musical_preview.py'), Path(__file__).with_name('listening.py'),
                       Path(__file__).with_name('audition_history.py'), Path(__file__).with_name('audio.py'),
                       Path(__file__).with_name('contracts.py')]},
              'label_counts_all_records': dict(counts), 'candidate_records': len(scores),
              'unique_novel_pairs': len(unique), 'directions': directions,
              'limitations': ['Experimental thresholds and weight, no human quality validation.',
                              'Closure candidate means arrival evidence, not verified musical resolution.',
                              'Both members share the outgoing context intentionally; full WAVs are unique.',
                              'Beat alignment and ASR gaps do not establish musical phrase or word integrity.']}
    (output/'scores.json').write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'experiment.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report
