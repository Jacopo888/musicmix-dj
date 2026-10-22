"""Run a controlled motion pilot; never overwrite historical artifacts."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from musicmix.expectation_pipeline import run_experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT/'runs/candidate_trajectories_v7_allbeat.json.gz')
    parser.add_argument('--history', type=Path, default=ROOT/'runs/audition_history_v7.json')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/expectation_v7')
    args = parser.parse_args()
    report = run_experiment(args.cache, args.history, args.output)
    print(json.dumps({d: {'selection': v['selection'], 'listening': v['listening']}
                      for d, v in report['directions'].items()}, indent=2))
