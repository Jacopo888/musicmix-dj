from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from musicmix.pool_experiment import run_pool_experiment
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Multi-song matched-context pilot with technical feedback')
    parser.add_argument('--catalog',type=Path,default=ROOT/'runs/pool_v8/catalog.json')
    latest_history=ROOT/'runs/audition_history_v10.json'
    parser.add_argument('--history',type=Path,default=latest_history if latest_history.is_file() else ROOT/'runs/audition_history_with_proposals.json')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/pool_experiment_v8')
    parser.add_argument('--cue-policy',choices=('beats','downbeats'),default='beats')
    parser.add_argument('--max-candidates-per-pair',type=int,default=200000)
    parser.add_argument('--reuse-run',type=Path,default=None)
    args=parser.parse_args();run_pool_experiment(args.catalog,args.history,args.output,
                  cue_policy=args.cue_policy,max_candidates_per_pair=args.max_candidates_per_pair,reuse_run=args.reuse_run)
