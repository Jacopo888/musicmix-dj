"""Rebuild geometry-only history from confirmed listening feedback."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from musicmix.audition_history import load_audition_history,include_presented_runs

if __name__ == '__main__':
    history = load_audition_history(ROOT.parent/'giudizi json', ROOT/'runs')
    path = ROOT/'runs/audition_history_v7.json'
    path.write_text(json.dumps(history, indent=2), encoding='utf-8')
    combined=include_presented_runs(history,ROOT/'runs/published_listening_tests.json',ROOT)
    (ROOT/'runs/audition_history_with_proposals.json').write_text(json.dumps(combined,indent=2),encoding='utf-8')
    print(json.dumps({'path': str(path), 'inputs': len(history['inputs']),
                      'regions': len(history['regions']), 'regions_including_presented':len(combined['regions']),
                      'unresolved': history['unresolved']}, indent=2))
