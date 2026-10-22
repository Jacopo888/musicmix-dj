"""Two-song research run: decode, analyse, retrieve, render, record provenance."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import importlib.metadata
import time
import uuid

from .ingest import decode_audio, sha256_file
from .preview import render_preview


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def run_search(source_a: Path, source_b: Path, output: Path, top_k: int = 10,
               max_candidates: int = 10000) -> dict:
    # Audio dependencies remain optional for the original JSON-only score command.
    from .audio import load_audio
    from .analysis import analyze_track
    from .search import search_transitions
    if top_k < 1 or max_candidates < 1:
        raise ValueError('top-k and max-candidates must be positive')
    if sha256_file(source_a) == sha256_file(source_b):
        raise ValueError('Provide two different audio tracks')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_id = str(uuid.uuid4())
    options = {'top_k': top_k, 'max_candidates': max_candidates,
               'preview_context_seconds': 2.0, 'preview_fade_seconds': 0.5}
    manifest = {'schema_version': 'musicmix.run.1', 'run_id': run_id,
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'status': 'running', 'python': platform.python_version(),
                'dependencies': {'numpy': importlib.metadata.version('numpy')},
                'options': options,
                'params_sha256': hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest(),
                'tracks': [], 'artifacts': [],
                'limitations': ['Heuristic onset/grid cues, not verified downbeats or song sections.',
                                'No lyrics, stems, learned compatibility, or human validation.',
                                'A working baseline does not establish Gate A.']}
    start = time.perf_counter()
    def event(module, status, **details):
        rec = {'run_id': run_id, 'module': module, 'stage': 'pipeline', 'status': status,
               'timestamp_utc': datetime.now(timezone.utc).isoformat(),
               'duration_ms': round((time.perf_counter()-start)*1000, 3),
               'details': details}
        with (output/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(rec, ensure_ascii=False, allow_nan=False)+'\n')
    write_json(output/'manifest.json', manifest)
    try:
        tracks = []
        analyses = []
        for label, source in [('a', source_a), ('b', source_b)]:
            metadata = decode_audio(source, output/'audio'/f'{label}.wav')
            track = load_audio(Path(metadata['canonical_audio_uri']))
            metadata.update(track_id=track.track_id, duration_samples=track.duration_samples)
            manifest['tracks'].append(metadata)
            event('M00', 'decoded', track_id=track.track_id)
            analysis = analyze_track(track)
            write_json(output/f'analysis_{label}.json', asdict(analysis))
            event('M02_M05', 'analysed', track_id=track.track_id, cues=len(analysis.cues))
            tracks.append(track); analyses.append(analysis)
        result = search_transitions(*analyses, top_k=top_k, max_candidates=max_candidates)
        write_json(output/'search.json', asdict(result))
        # Individual records remain compatible with the original scoring CLI/schema.
        report_map = {report.candidate_id: report for report in result.reports}
        candidate_map = {candidate.candidate_id: candidate for candidate in result.candidates}
        (output/'candidates').mkdir()
        for candidate in result.candidates:
            record = {'schema_version': '0.1.0', 'candidate_id': candidate.candidate_id,
                      'track_a': {'track_id': tracks[0].track_id, 'duration_samples': tracks[0].duration_samples},
                      'track_b': {'track_id': tracks[1].track_id, 'duration_samples': tracks[1].duration_samples},
                      'transition': {'cue_out_sample': candidate.cue_out_sample,
                                     'cue_in_sample': candidate.cue_in_sample, 'mode': candidate.mode},
                      'features': {key: asdict(value) for key, value in candidate.features.items()},
                      'penalties': {key: asdict(value) for key, value in candidate.penalties.items()},
                      'constraints': {'passed': candidate.passes_constraints,
                                      'reasons': list(candidate.constraint_reasons)},
                      'provenance': {'run_id': run_id, 'generator_version': result.diagnostics['algorithm'],
                                     'feature_extractor_version': analyses[0].backend,
                                     'audio_sha256_a': manifest['tracks'][0]['canonical_sha256'],
                                     'audio_sha256_b': manifest['tracks'][1]['canonical_sha256'],
                                     'sample_rate': 48000}}
            write_json(output/'candidates'/f'{candidate.candidate_id}.json', record)
        event('M06_M08', 'ranked', candidates=len(result.candidates), selected=len(result.selected))
        rows = ['# MusicMix — risultato baseline', '',
                'Cue euristici; qualità musicale non ancora validata con ascoltatori.', '',
                '| # | Uscita A (s) | Entrata B (s) | Score euristico | Anteprima |',
                '|---|---:|---:|---:|---|']
        for index, candidate_id in enumerate(result.selected, 1):
            candidate = candidate_map[candidate_id]
            name = f'{index:02d}_{candidate_id}.wav'
            artifact = render_preview(*tracks, candidate, output/'previews'/name)
            manifest['artifacts'].append(artifact)
            rows.append(f'| {index} | {candidate.cue_out_sample/48000:.3f} | '
                        f'{candidate.cue_in_sample/48000:.3f} | {report_map[candidate_id].score:.4f} | '
                        f'[WAV](previews/{name}) |')
        if not result.selected:
            rows.extend(['', 'Nessun candidato affidabile: consultare search.json e analysis_*.json.'])
        (output/'REPORT.md').write_text('\n'.join(rows)+'\n', encoding='utf-8')
        manifest['status'] = 'complete' if result.selected else 'no_candidates'
        manifest['candidate_count'] = len(result.candidates)
        manifest['selected_count'] = len(result.selected)
        event('M09', manifest['status'], previews=len(manifest['artifacts']))
    except Exception as error:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(error).__name__}: {error}'
        event('pipeline', 'failed', error=manifest['error'])
        raise
    finally:
        manifest['elapsed_seconds'] = round(time.perf_counter()-start, 3)
        write_json(output/'manifest.json', manifest)
    return manifest
