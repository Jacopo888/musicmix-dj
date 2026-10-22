"""Own the single shared `opencode serve` used by every swarm on this machine.

    python tools/swarm_server.py start     # reuse a healthy registered server or start one
    python tools/swarm_server.py status
    python tools/swarm_server.py smoke     # one real DeepSeek call in an empty folder, then delete the session
    python tools/swarm_server.py stop

The pid file and executable are written where tools/opencode_swarm.py expects
them (`--pid-file`, manifest `server_executable`), so launchers can resume.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opencode_platform import (DEFAULT_PORT, MODEL, find_opencode, kill_server_tree,  # noqa: E402
                               resources, spawn_server, verify_registered_server)

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'runs/opencode_server'


def call(base, method, path, directory=None, data=None, timeout=30):
    url = base + path
    if directory is not None:
        url += ('&' if '?' in url else '?') + urllib.parse.urlencode({'directory': str(directory)})
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method=method, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def healthy(base):
    try:
        return bool(call(base, 'GET', '/global/health', timeout=3).get('healthy'))
    except (OSError, ValueError, AttributeError):
        return False


def registered():
    path = STATE/'server.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def start(args):
    base = f'http://127.0.0.1:{args.port}'
    info = registered()
    if healthy(base):
        if info and info.get('port') == args.port:
            try:
                verify_registered_server(info['pid'], info['executable'], args.port)
                print(json.dumps({'status': 'reused', **info})); return 0
            except RuntimeError:
                pass
        print(json.dumps({'status': 'port_owned_by_unregistered_service', 'port': args.port}))
        return 3
    exe = find_opencode()
    STATE.mkdir(parents=True, exist_ok=True)
    process = spawn_server(exe, args.port, STATE, STATE/'server.log',
                           {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    for _ in range(60):
        if healthy(base):
            break
        if process.poll() is not None:
            raise RuntimeError(f'opencode serve exited with {process.returncode}; see {STATE/"server.log"}')
        time.sleep(0.5)
    else:
        raise RuntimeError(f'opencode serve not healthy; see {STATE/"server.log"}')
    info = {'pid': process.pid, 'port': args.port, 'executable': str(exe), 'started_unix': time.time()}
    (STATE/'pid.txt').write_text(str(process.pid))
    (STATE/'server.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'started', **info})); return 0


def status(args):
    base = f'http://127.0.0.1:{args.port}'; info = registered()
    report = {'healthy': healthy(base), 'registered': info}
    if info:
        report['resources'] = resources(info['pid'])
    print(json.dumps(report, indent=2)); return 0 if report['healthy'] else 1


def stop(args):
    info = registered()
    if not info:
        print(json.dumps({'status': 'nothing_registered'})); return 0
    verify_registered_server(info['pid'], info['executable'], info['port'])
    kill_server_tree(info['pid'])
    (STATE/'server.json').unlink(missing_ok=True); (STATE/'pid.txt').unlink(missing_ok=True)
    print(json.dumps({'status': 'stopped', 'pid': info['pid']})); return 0


def smoke(args):
    """End-to-end check: provider reachable through this network, free model answers."""
    base = f'http://127.0.0.1:{args.port}'
    if not healthy(base):
        print(json.dumps({'status': 'server_not_healthy'})); return 1
    folder = Path(tempfile.mkdtemp(prefix='opencode_smoke_'))
    session = call(base, 'POST', '/session', folder, {'title': 'smoke'})
    identity = session['id']; started = time.time(); result = {'status': 'timeout'}
    try:
        call(base, 'POST', f'/session/{identity}/prompt_async', folder,
             {'model': MODEL, 'variant': 'high',
              'parts': [{'type': 'text', 'text': 'Reply with exactly the word PONG and nothing else.'}]})
        while time.time() - started < args.timeout:
            time.sleep(3)
            state = (call(base, 'GET', '/session/status', folder) or {}).get(identity, {'type': 'idle'})
            messages = call(base, 'GET', f'/session/{identity}/message', folder) or []
            done = [m for m in messages if m.get('info', {}).get('role') == 'assistant'
                    and m['info'].get('time', {}).get('completed')]
            if done:
                info = done[-1]['info']
                text = '\n'.join(p.get('text', '') for p in done[-1].get('parts', []) if p.get('type') == 'text')
                result = {'status': 'error' if info.get('error') else 'ok', 'answer': text.strip()[:200],
                          'error': info.get('error'), 'model': info.get('modelID'),
                          'seconds': round(time.time() - started, 1)}
                break
            if state.get('type') == 'retry':
                result = {'status': 'retrying', 'server_status': state}
    finally:
        try: call(base, 'POST', f'/session/{identity}/abort', folder, {})
        finally: call(base, 'DELETE', f'/session/{identity}', folder)
    print(json.dumps(result, ensure_ascii=False)); return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['start', 'status', 'stop', 'smoke'])
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args()
    raise SystemExit({'start': start, 'status': status, 'stop': stop, 'smoke': smoke}[args.command](args))
