"""Dispatch one bounded follow-up to an existing isolated OpenCode folder."""
import argparse,json,time
from pathlib import Path
from opencode_swarm import request,MODEL

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--task',required=True);p.add_argument('--name',required=True)
    p.add_argument('--record',type=Path,required=True);a=p.parse_args()
    folder=a.folder.resolve(strict=True);folder.relative_to(ROOT/'runs')
    task=folder/a.task;task.resolve(strict=True).relative_to(folder)
    s=request('http://127.0.0.1:4096','POST','/session',str(folder),{'title':a.name})
    j={'name':a.name,'directory':str(folder),'task_file':a.task,'session_id':s['id'],'status':'running','started_unix':time.time(),'model':MODEL}
    with a.record.open('x',encoding='utf8') as f:json.dump(j,f,indent=2)
    request('http://127.0.0.1:4096','POST','/session/'+s['id']+'/prompt_async',str(folder),{'model':MODEL,'variant':'high','parts':[{'type':'text','text':task.read_text(encoding='utf8')}]})
    print(json.dumps({'started':a.name,'session':s['id']}))

if __name__=='__main__':main()
