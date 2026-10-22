"""Regression checks for monitor recovery while external sessions stay alive."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def load_launcher():
    path = Path(__file__).resolve().parents[1] / 'tools' / 'opencode_swarm.py'
    spec = importlib.util.spec_from_file_location('swarm_recovery_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_windows_sharing_lock_preserves_checkpoint_then_recovers(tmp_path, monkeypatch):
    launcher = load_launcher()
    checkpoint = tmp_path / 'swarm.json'
    checkpoint.write_text('{"previous": true}', encoding='utf-8')
    original_replace = Path.replace
    attempts = []

    def locked_replace(source, target):
        attempts.append(source)
        if len(attempts) <= 2:
            assert json.loads(checkpoint.read_text()) == {'previous': True}
            raise PermissionError('Transient Windows sharing lock')
        return original_replace(source, target)

    monkeypatch.setattr(Path, 'replace', locked_replace)
    monkeypatch.setattr(launcher.time, 'sleep', lambda _: None)
    launcher.atomic_write_json(checkpoint, {'completed': True})
    assert json.loads(checkpoint.read_text()) == {'completed': True}
    assert len(attempts) == 3


def test_resumed_monitor_archives_existing_session_without_duplicate_launch(tmp_path, monkeypatch):
    launcher = load_launcher()
    folder = tmp_path / 'agent'
    folder.mkdir()
    manifest_path = tmp_path / 'swarm.json'
    manifest_path.write_text(json.dumps({
        'server_executable': 'registered-opencode.exe', 'server_pid': 1234,
        'jobs': [{'name': 'research', 'directory': str(folder), 'status': 'running',
                  'session_id': 'existing-session', 'started_unix': 0}]}), encoding='utf-8')
    pid_file = tmp_path / 'pid.txt'
    pid_file.write_text('1234')
    calls = []

    def request(base, method, path, directory=None, data=None):
        calls.append((method, path))
        assert (method, path) != ('POST', '/session'), 'Must reuse the live session'
        if path == '/session/status':
            return {'existing-session': {'type': 'idle'}}
        if path.endswith('/message'):
            return [{'info': {'role': 'assistant', 'time': {'completed': 1},
                              'modelID': 'deepseek-v4.1-flash-free', 'error': None},
                     'parts': [{'type': 'text', 'text': 'Existing research result'}]}]
        return None

    monkeypatch.setattr(launcher, 'request', request)
    monkeypatch.setattr(launcher, 'verify_registered_server', lambda *args: None)
    monkeypatch.setattr(launcher, 'resources', lambda _: {
        'free_ram_mb': 2000, 'free_commit_mb': 4000,
        'server_rss_mb': 500, 'server_private_mb': 500})
    args = SimpleNamespace(manifest=manifest_path, server='http://127.0.0.1:4096',
                           pid_file=pid_file, port=4096, concurrency=8,
                           max_server_mb=4096, timeout=10**12)
    assert launcher.run(args) == 0
    recovered = json.loads(manifest_path.read_text())['jobs'][0]
    assert recovered['status'] == 'complete' and recovered['session_deleted']
    assert recovered['session_id'] == 'existing-session'
    assert (folder / 'RESPONSE.md').read_text() == 'Existing research result'
    assert ('DELETE', '/session/existing-session') in calls
