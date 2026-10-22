"""MusicMix CLI: library import (``ingest``), legacy scoring, onset baseline and structured experiments."""
from __future__ import annotations

import argparse
import json
import sys
import subprocess
from pathlib import Path
from dataclasses import asdict

from .contracts import Candidate, Feature
from .ranking import InterpretableRanker


def load_candidate(path: Path) -> Candidate:
    rec = json.loads(path.read_text(encoding='utf-8'))
    left = rec['track_a']; right = rec['track_b']; trans = rec['transition']
    if trans['cue_out_sample'] >= left['duration_samples'] or trans['cue_in_sample'] >= right['duration_samples']:
        raise ValueError('Cue fuori durata della traccia')
    to_feat = lambda x: Feature(x.get('value'), x.get('confidence'), x.get('source', ''))
    return Candidate(
        candidate_id=rec['candidate_id'],
        track_a=left['track_id'], track_b=right['track_id'],
        cue_out_sample=trans['cue_out_sample'], cue_in_sample=trans['cue_in_sample'],
        mode=trans['mode'],
        features={name: to_feat(f) for name,f in rec['features'].items()},
        penalties={name: to_feat(f) for name,f in rec['penalties'].items()},
        passes_constraints=rec['constraints']['passed'],
        constraint_reasons=tuple(rec['constraints']['reasons']))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'ingest':
        from .library import main as ingest_main
        raise SystemExit(ingest_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == 'search-structured':
        ap = argparse.ArgumentParser(description='Ricerca direzionale da WAV canonici e analisi versionate')
        ap.add_argument('track_a', type=Path)
        ap.add_argument('track_b', type=Path)
        for name in ['structure-a', 'structure-b', 'vocals-a', 'vocals-b']:
            ap.add_argument(f'--{name}', type=Path, required=True)
        ap.add_argument('--output', type=Path, required=True)
        ap.add_argument('--top-k', type=int, default=5)
        ap.add_argument('--max-candidates', type=int, default=4096)
        ap.add_argument('--baseline-run', type=Path)
        ap.add_argument('--cue-policy', choices=['downbeats', 'beats'], default='downbeats')
        ap.add_argument('--min-cue-spacing-seconds', type=float, default=8.0)
        ap.add_argument('--listening-title', default='MusicMix')
        ap.add_argument('--random-controls', action='store_true')
        ap.add_argument('--allow-repeated-listening-audio', action='store_true',
                        help='Consente riusi audio nel pacchetto A/B; il default li esclude')
        ap.add_argument('--collect-motion', action='store_true',
                        help='Chiede annotazioni su continuazione, risoluzione e interruzione')
        args = ap.parse_args(sys.argv[2:])
        try:
            from .structured_pipeline import run_structured
            result = run_structured(args.track_a, args.track_b, args.structure_a, args.structure_b,
                                    args.vocals_a, args.vocals_b, args.output, args.top_k,
                                    args.max_candidates, args.baseline_run,
                                    cue_policy=args.cue_policy,
                                    min_cue_spacing_seconds=args.min_cue_spacing_seconds,
                                    listening_title=args.listening_title,
                                    random_controls=args.random_controls,
                                    no_repeated_listening_audio=not args.allow_repeated_listening_audio,
                                    collect_motion=args.collect_motion)
        except (ValueError, RuntimeError, OSError, ImportError, subprocess.SubprocessError) as error:
            ap.exit(2, f'MusicMix: {error}\n')
        print(json.dumps({'status': result['status'], 'selected': result['selected_count'],
                          'output': str(args.output.resolve())}, ensure_ascii=False, indent=2))
        return
    if len(sys.argv) > 1 and sys.argv[1] == 'search':
        ap = argparse.ArgumentParser(description='Ricerca baseline di transizioni tra due file audio')
        ap.add_argument('track_a', type=Path)
        ap.add_argument('track_b', type=Path)
        ap.add_argument('--output', type=Path, required=True, help='Nuova cartella per gli artefatti della run')
        ap.add_argument('--top-k', type=int, default=10)
        ap.add_argument('--max-candidates', type=int, default=10000)
        args = ap.parse_args(sys.argv[2:])
        try:
            from .pipeline import run_search
            result = run_search(args.track_a, args.track_b, args.output, args.top_k, args.max_candidates)
        except (ValueError, RuntimeError, OSError, ImportError, subprocess.SubprocessError) as error:
            ap.exit(2, f'MusicMix: {error}\n')
        print(json.dumps({'status': result['status'], 'selected': result['selected_count'],
                          'output': str(args.output.resolve())}, ensure_ascii=False, indent=2))
        return
    ap = argparse.ArgumentParser(description='Prototipo di scoring MusicMix da feature precomputate')
    ap.add_argument('candidate_json', type=Path)
    args=ap.parse_args()
    result=InterpretableRanker().score(load_candidate(args.candidate_json))
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
