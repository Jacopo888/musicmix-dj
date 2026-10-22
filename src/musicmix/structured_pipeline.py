"""Run the directional experiment from explicit, versioned provider outputs."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import shutil
import random
import time
import uuid

from .audio import load_audio
from .ingest import sha256_file
from .musical_preview import render_handoff, HANDOFF_VERSION
from .pipeline import write_json
from .contracts import Candidate, Feature


def _baseline_candidates(run: Path) -> tuple[list[Candidate], str]:
    """Read either the frozen onset baseline or a structured experiment."""
    from .__main__ import load_candidate
    payload = json.loads((run/'search.json').read_text(encoding='utf-8'))
    if 'items' not in payload:
        return ([load_candidate(run/'candidates'/f'{identity}.json')
                 for identity in payload['selected'][:3]], 'onset_snapshot_v1')
    candidates = []
    for item in payload['items'][:3]:
        raw = dict(item['candidate'])
        for name in ('features', 'penalties'):
            raw[name] = {key: Feature(**value) for key, value in raw[name].items()}
        raw['constraint_reasons'] = tuple(raw.get('constraint_reasons', ()))
        candidates.append(Candidate(**raw))
    return candidates, run.name


def _load_evidence(path: Path, track, name: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('track_id') != track.track_id:
        raise ValueError(f'{name}: evidence track_id does not match the canonical WAV hash')
    if payload.get('sample_rate') != track.sample_rate:
        raise ValueError(f'{name}: evidence sample rate mismatch')
    if payload.get('duration_samples') != track.duration_samples:
        raise ValueError(f'{name}: evidence duration mismatch')
    if payload.get('status') in {'unavailable', 'failed', 'timeout', 'error'}:
        raise ValueError(f'{name}: provider status is {payload["status"]}')
    return payload


def run_structured(source_a: Path, source_b: Path, structure_a: Path, structure_b: Path,
                   vocals_a: Path, vocals_b: Path, output: Path, top_k: int = 5,
                   max_candidates: int = 4096, baseline_run: Path | None = None,
                   *, listening_title: str = 'MusicMix', random_controls: bool = False,
                   cue_policy: str = 'downbeats', min_cue_spacing_seconds: float = 8.0,
                   no_repeated_listening_audio: bool = True,
                   collect_motion: bool = False) -> dict:
    """Sources are the exact PCM WAVs bound to the supplied evidence hashes.

    Providers run separately with optional runtimes. This function never
    downloads models or falls back silently to the old onset grid.
    """
    from .directional import search_directional
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError('top_k must be a positive integer')
    if not isinstance(max_candidates, int) or isinstance(max_candidates, bool) or max_candidates <= 0:
        raise ValueError('max_candidates must be a positive integer')
    a, b = load_audio(source_a), load_audio(source_b)
    if a.sample_rate != b.sample_rate or a.sample_rate != 48000:
        raise ValueError('Structured experiment requires canonical 48 kHz WAVs')
    evidence = [_load_evidence(path, track, name) for path, track, name in [
        (structure_a, a, 'structure_a'), (structure_b, b, 'structure_b'),
        (vocals_a, a, 'vocals_a'), (vocals_b, b, 'vocals_b')]]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {'schema_version': 'musicmix.structured-run.1', 'run_id': str(uuid.uuid4()),
                'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
                'tracks': [{'path': str(Path(path).resolve()), 'sha256': track.track_id,
                            'duration_samples': track.duration_samples, 'sample_rate': track.sample_rate}
                           for path, track in [(source_a, a), (source_b, b)]],
                'evidence': {}, 'options': {'top_k': top_k, 'max_candidates': max_candidates},
                'renderer': HANDOFF_VERSION, 'artifacts': [], 'human_validation': 'not_performed'}
    module_root = Path(__file__).resolve().parent
    manifest['code_sha256'] = {name: sha256_file(module_root/name) for name in [
        'directional.py', 'vocal_boundaries.py', 'musical_preview.py',
        'structured_pipeline.py', 'listening.py']}
    manifest['options'].update(context_seconds=10.0, render_fade_seconds=0.012,
                                allow_unknown_vocals=False, safety_fade_seconds=0.04,
                                max_cue_shift_seconds=0.08, min_downbeat_confidence=0.20)
    manifest['options'].update(cue_policy=cue_policy, min_cue_spacing_seconds=min_cue_spacing_seconds)
    manifest['options']['no_repeated_listening_audio'] = no_repeated_listening_audio
    manifest['options']['collect_motion'] = collect_motion
    manifest['options']['random_controls'] = random_controls
    manifest['options']['random_control_seed'] = 314159 if random_controls else None
    manifest['params_sha256'] = hashlib.sha256(
        json.dumps(manifest['options'], sort_keys=True).encode()).hexdigest()
    (output/'inputs').mkdir()
    for name, path in [('structure_a', structure_a), ('structure_b', structure_b),
                       ('vocals_a', vocals_a), ('vocals_b', vocals_b)]:
        target = output/'inputs'/f'{name}.json'
        shutil.copyfile(path, target)
        manifest['evidence'][name] = {'source': str(Path(path).resolve()),
                                      'snapshot': str(target), 'sha256': sha256_file(target)}
    write_json(output/'manifest.json', manifest)
    try:
        result = search_directional(a, b, evidence[0], evidence[1], evidence[2], evidence[3],
                                    top_k=top_k, max_candidates=max_candidates,
                                    fade_seconds=0.04, max_shift_seconds=0.08,
                                    allow_unknown_vocals=False, min_context_seconds=10.0,
                                    min_downbeat_confidence=0.20, cue_policy=cue_policy,
                                    min_cue_spacing_seconds=min_cue_spacing_seconds)
        write_json(output/'search.json', asdict(result))
        rows = ['# MusicMix — struttura e contesti direzionali', '',
                'Previsioni automatiche: musicalità e precisione dei confini richiedono ascolto.', '',
                '| # | Uscita A | Entrata B | Score euristico | Estratto |',
                '|---|---:|---:|---:|---|']
        selected = []
        for item in result.items:
            candidate = item.candidate
            if not all(item.alignment.get(side, {}).get('safe') is True for side in ['outgoing', 'incoming']):
                raise ValueError('A selected proposal lacks supported vocal boundaries on both sides')
            # Ten seconds on each side is fixed for all new/old comparisons.
            if candidate.cue_out_sample < 10*a.sample_rate or candidate.cue_in_sample+10*b.sample_rate > b.duration_samples:
                raise ValueError('Directional proposal lacks the required fixed listening context')
            index = len(selected)+1
            name = f'{index:02d}_{candidate.candidate_id}.wav'
            fade = min(0.012, item.fade_seconds)
            artifact = render_handoff(a, b, candidate, output/'previews'/name,
                                      context_seconds=10, fade_seconds=fade)
            artifact['vocal_alignment'] = dict(item.alignment)
            manifest['artifacts'].append(artifact)
            write_json(output/f'proposal_{index:02d}.json', asdict(item))
            selected.append((item, artifact))
            rows.append(f'| {index} | {candidate.cue_out_sample/a.sample_rate:.3f} s | '
                        f'{candidate.cue_in_sample/b.sample_rate:.3f} s | {item.score:.4f} | '
                        f'[WAV](previews/{name}) |')
        if not selected:
            rows += ['', 'Nessuna proposta soddisfa i vincoli. Le esclusioni sono salvate in search.json.']
        if baseline_run and selected:
            from .listening import make_listening_test
            old, old_system = _baseline_candidates(Path(baseline_run))
            pairs = []
            for index, ((item, artifact), candidate) in enumerate(zip(selected, old), 1):
                baseline = render_handoff(a, b, candidate, output/'baseline_same_renderer'/f'{index:02d}.wav',
                                           context_seconds=10, fade_seconds=0.012)
                pairs.append({'left_path': baseline['path'], 'right_path': artifact['path'],
                              'left_system': old_system, 'right_system': result.diagnostics.get('algorithm', 'structured'),
                              'pair_id': f'rank_{index}'})
            if random_controls:
                unique = {(p.candidate.cue_out_sample, p.candidate.cue_in_sample): p.candidate
                          for p in result.proposals}
                pool = [unique[key] for key in sorted(unique)]
                rng = random.Random(314159)
                draws = []
                for index, (item, artifact) in enumerate(selected[:3], 1):
                    if not pool:
                        break
                    candidate = rng.choice(pool)
                    same = (candidate.cue_out_sample, candidate.cue_in_sample) == (
                        item.candidate.cue_out_sample, item.candidate.cue_in_sample)
                    draws.append({'trial': index, 'candidate_id': candidate.candidate_id,
                                  'cue_out_sample': candidate.cue_out_sample,
                                  'cue_in_sample': candidate.cue_in_sample, 'identical_to_selected': same})
                    if same:
                        continue
                    random_audio = render_handoff(a, b, candidate,
                                                  output/'random_same_renderer'/f'{index:02d}.wav',
                                                  context_seconds=10, fade_seconds=0.012)
                    pairs.append({'left_path': random_audio['path'], 'right_path': artifact['path'],
                                  'left_system': 'uniform_valid_cue_pair',
                                  'right_system': result.diagnostics.get('algorithm', 'structured'),
                                  'pair_id': f'random_{index}'})
                manifest['random_control'] = {'seed': 314159, 'pool_size': len(pool), 'draws': draws,
                                               'scope': 'uniform over unique admitted cue pairs, before top-K diversity; identical draws omitted from listening'}
            manifest['listening'] = make_listening_test(output/'listening_ab', pairs,
                                                       run_id=manifest['run_id'], title=listening_title,
                                                       no_repeated_audio=no_repeated_listening_audio,
                                                       collect_motion=collect_motion)
            if manifest['listening']['trials']:
                rows += ['', '[Confronto A/B con identico renderer](listening_ab/index.html)', '',
                         'La chiave è salvata separatamente; i giudizi non sono ancora stati raccolti.']
            else:
                rows += ['', 'Nessun confronto nuovo disponibile dopo la deduplicazione audio.']
        manifest['status'] = 'complete' if selected else 'no_safe_candidates'
        manifest['selected_count'] = len(selected)
        manifest['diagnostics'] = dict(result.diagnostics)
        (output/'REPORT.md').write_text('\n'.join(rows)+'\n', encoding='utf-8')
    except Exception as error:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        manifest['elapsed_seconds'] = round(time.perf_counter()-started, 3)
        write_json(output/'manifest.json', manifest)
    return manifest
