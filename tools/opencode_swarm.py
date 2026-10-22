"""Bounded OpenCode API swarm: one server, isolated folders, archived answers.

Uses the normal OpenCode agent and the explicitly requested free model. No API
credentials are read by this launcher. Completed sessions are deleted only
after their final messages have been saved locally.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opencode_platform import (MODEL, kill_server_tree, resources, spawn_server,  # noqa: E402
                               verify_registered_server)


def request(base, method, path, directory=None, data=None):
    url = base + path
    if directory is not None:
        url += ('&' if '?' in url else '?') + urllib.parse.urlencode({'directory': str(directory)})
    payload = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=payload, method=method, headers={'Content-Type': 'application/json'})
    for attempt in range(3 if method == 'GET' else 1):
        try:
            with urllib.request.urlopen(req, timeout=10 if method == 'GET' else 30) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except (TimeoutError, urllib.error.URLError):
            if method != 'GET' or attempt == 2:
                raise
            time.sleep(.5 * (attempt + 1))


def archive(base, job, reason):
    folder = Path(job['directory']); identity = job['session_id']
    messages = request(base, 'GET', f'/session/{identity}/message', folder) or []
    attempt_folder = folder/'sessions'/identity
    attempt_folder.mkdir(parents=True, exist_ok=True)
    (attempt_folder/'messages.json').write_text(json.dumps(messages,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'session_messages.json').write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding='utf-8')
    answers = ['\n'.join(part.get('text', '') for part in message.get('parts', []) if part.get('type') == 'text')
               for message in messages if message.get('info', {}).get('role') == 'assistant']
    (folder/'RESPONSE.md').write_text('\n\n'.join(answers), encoding='utf-8')
    job['status'] = reason
    try: request(base, 'POST', f'/session/{identity}/abort', folder, {})
    finally: request(base, 'DELETE', f'/session/{identity}', folder)
    job['session_deleted'] = True


def atomic_write_json(path, payload):
    """Retry transient Windows sharing locks without losing the last checkpoint."""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    for attempt in range(31):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 30:
                raise
            time.sleep(0.1)


def run(args):
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    jobs = manifest['jobs']; base = args.server.rstrip('/')
    pidfile = args.pid_file.resolve(); pid = int(pidfile.read_text().strip())
    # A monitor can fail while the shared server and its sessions keep working.
    # Resume those exact sessions instead of abandoning or duplicating them.
    active = {job['session_id']: job for job in jobs if job.get('status') == 'running'}
    if active:
        if manifest.get('server_pid') != pid:
            raise RuntimeError('Cannot resume sessions on a different registered server')
        verify_registered_server(pid, manifest['server_executable'], args.port)
    restarts = 0
    def save():
        manifest.update(model=MODEL, variant='high', server=base, server_pid=pid,
                        resources=resources(pid), updated_unix=time.time())
        atomic_write_json(manifest_path, manifest)
    while True:
        memory = resources(pid)
        if memory['server_rss_mb'] > args.max_server_mb or memory['server_private_mb'] > args.max_server_mb:
            # Guard restarts only this task's registered server, after aborting
            # and archiving each active session. Files in the job folders persist.
            if restarts >= 1:
                for job in list(active.values()): archive(base, job, 'memory_guard_stopped')
                save(); return 2
            for job in list(active.values()):
                archive(base, job, 'pending')
                job['resumed_after_memory_guard'] = True
            active.clear()
            verify_registered_server(pid, manifest['server_executable'], args.port)
            kill_server_tree(pid)
            server_folder = pidfile.parent
            exe = manifest['server_executable']
            process = spawn_server(exe, args.port, server_folder, server_folder/'guard_restart.log')
            pid = process.pid; pidfile.write_text(str(pid)); restarts += 1
            for _ in range(30):
                time.sleep(1)
                try:
                    if request(base, 'GET', '/global/health')['healthy']: break
                except Exception: pass
            save(); continue
        for job in jobs:
            if job.get('status', 'pending') != 'pending' or len(active) >= args.concurrency: continue
            memory = resources(pid)
            if memory['free_ram_mb'] < 800 or memory['free_commit_mb'] < 2560: break
            folder = Path(job['directory']).resolve(strict=True)
            session = request(base, 'POST', '/session', folder, {'title': job['name']})
            identity = session['id']
            prompt = (folder/job.get('task_file', 'TASK.md')).read_text(encoding='utf-8')
            if job.get('resumed_after_memory_guard'):
                prompt += '\nResume using existing files in this folder; previous session was stopped by a memory guard.'
            job.update(session_id=identity, status='running', started_unix=time.time())
            active[identity] = job
            request(base, 'POST', f'/session/{identity}/prompt_async', folder,
                    {'model': MODEL, 'variant': 'high', 'parts': [{'type': 'text', 'text': prompt}]})
            print(json.dumps({'event': 'started', 'job': job['name'], 'session': identity}), flush=True)
            save()
        for identity, job in list(active.items()):
            folder = Path(job['directory'])
            try:
                statuses = request(base, 'GET', '/session/status', folder) or {}
            except (TimeoutError, urllib.error.URLError) as exc:
                job['last_poll_error'] = type(exc).__name__
                save()
                continue
            status = statuses.get(identity, {'type': 'idle'})
            job['server_status'] = status
            description = json.dumps(status).lower()
            quota = bool(re.search(r'free.{0,40}(usage|quota).{0,40}(limit|exhaust|reset|exceed)|quota.{0,20}(exhaust|finished|exceed)|insufficient.quota', description))
            reason = None
            if quota: reason = 'free_quota_finished'
            elif time.time()-job['started_unix'] > args.timeout: reason = 'timeout'
            elif status.get('type') == 'idle' and time.time()-job['started_unix'] > 15:
                messages = request(base, 'GET', f'/session/{identity}/message', folder) or []
                assistants = [m for m in messages if m.get('info', {}).get('role') == 'assistant']
                completed = [m for m in assistants if m['info'].get('time', {}).get('completed')]
                if completed:
                    last = completed[-1]['info']
                    job['model_used'] = {k: last.get(k) for k in ('modelID', 'providerID', 'agent', 'cost', 'finish', 'error')}
                    reason = 'error' if last.get('error') else 'complete'
            if reason:
                error = job.get('model_used', {}).get('error') or {}
                retryable = (reason == 'error' and error.get('data', {}).get('isRetryable') is True
                             and job.get('transport_retries', 0) < 1)
                if retryable:
                    job['transport_retries'] = job.get('transport_retries', 0) + 1
                    job.setdefault('failed_transport_attempts', []).append({'session_id':identity,'error':error})
                archive(base, job, 'pending' if retryable else reason); active.pop(identity)
                if retryable:
                    job['resumed_after_memory_guard'] = True
                print(json.dumps({'event': reason, 'job': job['name']}), flush=True)
            save()
        if not active and not any(j.get('status', 'pending') == 'pending' for j in jobs): break
        time.sleep(10)
    save()
    return int(any(j['status'] != 'complete' for j in jobs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--server', default='http://127.0.0.1:4096')
    parser.add_argument('--pid-file', type=Path, default=Path('runs/swarm_v8/server/pid.txt'))
    parser.add_argument('--port', type=int, default=4096)
    parser.add_argument('--concurrency', type=int, choices=range(1, 9), default=5)
    parser.add_argument('--max-server-mb', type=int, default=4096)
    parser.add_argument('--timeout', type=int, default=1800)
    raise SystemExit(run(parser.parse_args()))
