"""Reuse pinned, already installed local providers; install nothing at runtime."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
for relative in ('.cache/structure_runtime/Lib/site-packages','.cache/structure_runtime/packages',
 'runs/result_swarm_v12/agents/vocal_completion/runtime',
 'runs/result_swarm_v12/agents/critic_research/runtime_syntax/Lib/site-packages'):
    p=ROOT/relative
    if p.is_dir():sys.path.append(str(p))

_syntax=None
def syntax_factory(words,entry):
    global _syntax
    from musicmix.syntactic_completion import SyntacticCompletionAnalyzer
    if _syntax is None:_syntax=SyntacticCompletionAnalyzer()
    prepared=_syntax.prepare(words.get('words',[]),entry['sample_rate'],entry['duration_samples'])
    return lambda cue:_syntax.query(prepared,cue)
