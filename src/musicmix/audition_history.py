"""Resolve heard audio to source cue pairs without interpreting comments."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from .ingest import sha256_file
from .paths import localize


def _geometry(candidate: dict, rate: int) -> dict:
    return {'track_a': candidate['track_a'], 'track_b': candidate['track_b'],
            'cue_out_sample': candidate['cue_out_sample'],
            'cue_in_sample': candidate['cue_in_sample'], 'sample_rate': rate}


def load_audition_history(feedback_directory: Path, runs_root: Path) -> dict:
    """Use confirmed normalized feedback only; ambiguous audio stays unresolved."""
    from .__main__ import load_candidate
    from dataclasses import asdict
    audio_index = defaultdict(list)
    manifests = {}
    legacy = []
    for path in sorted(runs_root.glob('*/manifest.json')):
        manifest = json.loads(path.read_text(encoding='utf-8'))
        tracks = manifest.get('tracks', [])
        if len(tracks) < 2:
            continue
        identities = [t.get('track_id', t.get('sha256')) for t in tracks]
        manifests[path.parent.resolve()] = (manifest, identities)
        if len(tracks) == 2 and manifest.get('schema_version') == 'musicmix.run.1':
            legacy.append((path.parent, identities))
        by_id = {}
        for proposal_path in path.parent.glob('proposal_*.json'):
            proposal = json.loads(proposal_path.read_text(encoding='utf-8'))
            if 'candidate' in proposal:
                candidate = proposal['candidate']
                by_id[candidate['candidate_id']] = _geometry(candidate, tracks[0]['sample_rate'])
        for artifact in manifest.get('artifacts', []):
            geometry = by_id.get(artifact.get('candidate_id'))
            if geometry and artifact.get('sha256'):
                audio_index[artifact['sha256']].append(geometry)

    # Controlled V7 pilots keep geometry in an experiment manifest.
    for path in sorted(runs_root.glob('*/experiment.json')):
        experiment = json.loads(path.read_text(encoding='utf-8'))
        if experiment.get('schema_version') != 'musicmix.expectation-experiment.1':
            continue
        for direction in experiment.get('directions', {}).values():
            candidates = {comparison[system]['candidate_id']: comparison[system]
                          for comparison in direction.get('comparisons', [])
                          for system in ('policy', 'baseline')}
            for artifact in direction.get('artifacts', []):
                candidate = candidates.get(artifact['candidate_id'])
                if candidate:
                    audio_index[artifact['sha256']].append(_geometry(candidate, artifact['sample_rate']))

    regions, unresolved, inputs = [], [], []
    known = set()
    file_hashes = {}
    for path in sorted(feedback_directory.glob('*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict) or payload.get('schema_version') != 'musicmix.evaluation.1' or not payload.get('mapping', {}).get('confirmed'):
            continue
        inputs.append({'path': str(path.resolve()), 'sha256': sha256_file(path)})
        for observation in payload.get('observations', []):
            if observation.get('preference_label') is None:
                continue
            for entry in observation.get('candidates', {}).values():
                source = localize(entry['source'])
                if not source.is_file():
                    unresolved.append({'source': str(source), 'reason': 'source_missing'})
                    continue
                if str(source) not in file_hashes:
                    file_hashes[str(source)] = sha256_file(source)
                digest = file_hashes[str(source)]
                matches = audio_index.get(digest, [])
                parent_run = source.parent.parent.resolve()
                owner = manifests.get(parent_run)
                if not matches and source.parent.name == 'random_same_renderer' and source.stem.isdigit() and owner:
                    draw = next((d for d in owner[0].get('random_control', {}).get('draws', [])
                                 if d.get('trial') == int(source.stem)), None)
                    if draw and not draw.get('identical_to_selected'):
                        matches = [{'track_a': owner[1][0], 'track_b': owner[1][1],
                                    'cue_out_sample': draw['cue_out_sample'],
                                    'cue_in_sample': draw['cue_in_sample'],
                                    'sample_rate': owner[0]['tracks'][0]['sample_rate']}]
                if not matches and entry.get('system') == 'onset_snapshot_v1':
                    # Historical baseline handoffs were rerendered by rank.
                    # Bind the legacy search by ordered canonical identities.
                    if owner and source.stem.isdigit():
                        for legacy_root, identities in legacy:
                            if identities != owner[1]:
                                continue
                            search = json.loads((legacy_root/'search.json').read_text(encoding='utf-8'))
                            slot = int(source.stem)-1
                            if 0 <= slot < len(search.get('selected', [])):
                                candidate = load_candidate(legacy_root/'candidates'/f'{search["selected"][slot]}.json')
                                matches.append(_geometry(asdict(candidate), owner[0]['tracks'][0]['sample_rate']))
                unique = {tuple(sorted(g.items())): g for g in matches}
                if len(unique) != 1:
                    unresolved.append({'source': str(source), 'sha256': digest,
                                       'reason': 'unresolved_or_ambiguous_geometry'})
                    continue
                geometry = next(iter(unique.values()))
                identity = tuple(sorted(geometry.items()))
                if identity not in known:
                    regions.append({**geometry, 'source': str(source.resolve()), 'audio_sha256': digest})
                    known.add(identity)
    return {'schema_version': 'musicmix.audition-history.1', 'inputs': inputs,
            'regions': regions, 'unresolved': unresolved,
            'policy': 'confirmed observations only; no semantic labels inferred from comments'}


def exposure(candidate, regions: list[dict], tolerance_seconds: float, *, sample_rate=48000) -> dict:
    """Distinguish a heard pair from reuse of either source region."""
    match_a = match_b = pair = False
    for region in regions:
        rate = region['sample_rate']
        # Compare source regions in either playback direction.
        cues = {region['track_a']: region['cue_out_sample']/rate,
                region['track_b']: region['cue_in_sample']/rate}
        a = candidate.track_a in cues and abs(candidate.cue_out_sample/sample_rate-cues[candidate.track_a]) <= tolerance_seconds
        b = candidate.track_b in cues and abs(candidate.cue_in_sample/sample_rate-cues[candidate.track_b]) <= tolerance_seconds
        match_a = match_a or a
        match_b = match_b or b
        if region['track_a'] == candidate.track_a and region['track_b'] == candidate.track_b:
            pair = pair or (a and b)
    return {'outgoing_region_heard': match_a, 'incoming_region_heard': match_b,
            'ordered_pair_heard': pair, 'tolerance_seconds': tolerance_seconds}


def include_presented_runs(history: dict, registry_path: Path, project_root: Path) -> dict:
    """Exclude explicitly presented proposals without inventing human ratings."""
    result={**history,'regions':list(history['regions']),'presented_inputs':[]}
    if not registry_path.is_file():return result
    registry=json.loads(registry_path.read_text(encoding='utf-8'))
    if registry.get('schema_version')!='musicmix.presented-runs.1':raise ValueError('Unknown presented-run registry')
    keys=('track_a','track_b','cue_out_sample','cue_in_sample','sample_rate')
    known={tuple(r[k] for k in keys) for r in result['regions']}
    root=project_root.resolve()
    for entry in registry['runs']:
        manifest=(root/entry['manifest']).resolve();manifest.relative_to(root)
        data=json.loads(manifest.read_text(encoding='utf-8'))
        if data.get('schema_version') == 'musicmix.psychology-pool.1' and entry.get('presentation') == 'psychology_review':
            for item in data.get('artifacts', []):
                if item.get('variant') != 'main':
                    continue
                artifact = item['artifact']
                path = localize(artifact['path']).resolve(); path.relative_to(root)
                if sha256_file(path) != artifact['sha256']:
                    raise ValueError('Presented audio hash mismatch')
                geometry = _geometry(item['candidate'], artifact['sample_rate'])
                identity = tuple(geometry[k] for k in keys)
                if identity not in known:
                    result['regions'].append(dict(geometry, source=str(path), audio_sha256=artifact['sha256'], proposed_only=True))
                    known.add(identity)
            result['presented_inputs'].append({'path':str(manifest),'sha256':sha256_file(manifest)})
            continue
        if data.get('schema_version') == 'musicmix.pretrained-cue-pilot.1' and entry.get('presentation') == 'pretrained_review':
            for artifact in data.get('artifacts', []):
                path=localize(artifact['path']).resolve();path.relative_to(root)
                if sha256_file(path)!=artifact['sha256']:
                    raise ValueError('Presented audio hash mismatch')
                geometry=_geometry(artifact['candidate'],artifact['sample_rate'])
                identity=tuple(geometry[k] for k in keys)
                if identity not in known:
                    result['regions'].append(dict(geometry,source=str(path),audio_sha256=artifact['sha256'],proposed_only=True))
                    known.add(identity)
            result['presented_inputs'].append({'path':str(manifest),'sha256':sha256_file(manifest)})
            continue
        scopes=list(data.get('directions',{}).values()) if 'directions' in data else [data]
        for scope in scopes:
            if scope.get('listening',{}).get('test_id') not in entry['test_ids']:continue
            artifacts={a['candidate_id']:a for a in scope['artifacts']}
            for comparison in scope['comparisons']:
                for system in ('baseline','policy','choice'):
                    if system not in comparison:continue
                    candidate=comparison[system];artifact=artifacts[candidate['candidate_id']]
                    path=localize(artifact['path']).resolve();path.relative_to(root)
                    if sha256_file(path)!=artifact['sha256']:raise ValueError('Presented audio hash mismatch')
                    geometry=_geometry(candidate,artifact['sample_rate']);identity=tuple(geometry[k] for k in keys)
                    if identity not in known:
                        result['regions'].append(dict(geometry,source=str(path),audio_sha256=artifact['sha256'],
                                                       proposed_only=True))
                        known.add(identity)
        result['presented_inputs'].append({'path':str(manifest),'sha256':sha256_file(manifest)})
    result['policy']='confirmed judgement geometries plus explicitly presented proposals; proposals are not human ratings'
    return result
