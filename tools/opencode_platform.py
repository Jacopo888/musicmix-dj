"""Cross-platform helpers for the shared OpenCode server (Windows desktop and Linux).

Behaviour on Windows is unchanged from the original launcher (DPAPI-free, ctypes
memory probes, PowerShell identity check, taskkill). On Linux the same contracts
are implemented with /proc and process groups, so the swarm guard works on the
Linux machine (Ubuntu 24.04).
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

WINDOWS = sys.platform == 'win32'
DEFAULT_PORT = 4096
MODEL = {'providerID': 'opencode', 'modelID': 'deepseek-v4.1-flash-free'}
# Explicit allow/deny only: an unattended agent waiting on "ask" would hang.
SERVER_CONFIG = {
    'permission': {'edit': 'allow', 'bash': 'allow', 'webfetch': 'allow', 'websearch': 'allow',
                   'external_directory': 'deny', 'question': 'deny'},
    'provider': {'opencode': {'options': {'apiKey': 'public'}}},
}


def find_opencode() -> Path:
    """Locate the OpenCode executable; the native binary is preferred over npm shims."""
    candidates = []
    if os.environ.get('OPENCODE_BIN'):
        candidates.append(Path(os.environ['OPENCODE_BIN']))
    if WINDOWS:
        candidates.append(Path(os.environ.get('APPDATA', '')) / 'npm/node_modules/opencode-ai/bin/opencode.exe')
    else:
        candidates += [Path('/usr/local/bin/opencode-native'), Path.home() / '.opencode/bin/opencode']
    found = shutil.which('opencode')
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError('OpenCode not found: set OPENCODE_BIN or run env/setup_ubuntu.sh')


def server_env(extra=None):
    env = dict(os.environ, OPENCODE_CONFIG_CONTENT=json.dumps(SERVER_CONFIG))
    env.update(extra or {})
    return env


def spawn_server(exe, port, cwd, log_path, extra_env=None):
    """Start `opencode serve` detached from the caller's console/session."""
    cwd = Path(cwd); cwd.mkdir(parents=True, exist_ok=True)
    command = [str(exe), 'serve', '--port', str(port), '--hostname', '127.0.0.1']
    with Path(log_path).open('ab') as log:
        if WINDOWS:
            return subprocess.Popen(command, cwd=cwd, env=server_env(extra_env), stdout=log, stderr=log,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
        return subprocess.Popen(command, cwd=cwd, env=server_env(extra_env), stdout=log, stderr=log,
                                stdin=subprocess.DEVNULL, start_new_session=True)


def _windows_resources(pid):
    import ctypes
    from ctypes import wintypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in ('total_ram', 'free_ram', 'total_commit',
            'free_commit', 'total_virtual', 'free_virtual', 'extended')]

    class ProcessMemory(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('peak_rss', 'rss', 'peak_paged', 'paged',
            'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile', 'private')]

    memory = MemoryStatus(); memory.length = ctypes.sizeof(memory)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
    ctypes.windll.kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = ctypes.windll.kernel32.OpenProcess(0x410, False, pid)
    counters = ProcessMemory(); counters.cb = ctypes.sizeof(counters)
    try:
        if handle:
            ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    finally:
        if handle: ctypes.windll.kernel32.CloseHandle(handle)
    return {'free_ram_mb': memory.free_ram/2**20, 'free_commit_mb': memory.free_commit/2**20,
            'server_rss_mb': counters.rss/2**20, 'server_private_mb': counters.private/2**20}


def _kib_fields(path):
    values = {}
    try:
        for line in Path(path).read_text().splitlines():
            key, _, rest = line.partition(':')
            parts = rest.split()
            if parts and parts[0].isdigit():
                values[key.strip()] = int(parts[0])
    except OSError:
        pass
    return values


def _process_group_rss_kib(pid):
    """RSS of the server plus its descendants (the npm shim may fork the native binary)."""
    total = {'rss': 0, 'anon': 0}
    try:
        pgid = os.getpgid(pid)
    except OSError:
        return total
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.getpgid(int(entry.name)) != pgid:
                continue
        except OSError:
            continue
        status = _kib_fields(entry/'status')
        total['rss'] += status.get('VmRSS', 0); total['anon'] += status.get('RssAnon', 0)
    return total


def resources(pid):
    """Same keys on every platform: free RAM, free commit (RAM+swap on Linux), server RSS/private."""
    if WINDOWS:
        return _windows_resources(pid)
    meminfo = _kib_fields('/proc/meminfo')
    group = _process_group_rss_kib(pid)
    available = meminfo.get('MemAvailable', 0)
    return {'free_ram_mb': available/1024,
            'free_commit_mb': (available + meminfo.get('SwapFree', 0))/1024,
            'server_rss_mb': group['rss']/1024, 'server_private_mb': group['anon']/1024}


def verify_registered_server(pid, executable, port):
    """Refuse to terminate a recycled PID or an unrelated OpenCode service."""
    expected = f'serve --port {port} --hostname 127.0.0.1'
    if WINDOWS:
        command = (f"$taskProcessInfo=Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}'; "
                   "$taskProcessInfo | Select-Object ExecutablePath,CommandLine | ConvertTo-Json -Compress")
        result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                                capture_output=True, text=True, check=True, timeout=20)
        info = json.loads(result.stdout) if result.stdout.strip() else None
        if (not info or Path(info['ExecutablePath']).resolve() != Path(executable).resolve()
                or expected not in info.get('CommandLine', '')):
            raise RuntimeError('Registered server identity changed; no process terminated')
        return
    try:
        argv = Path(f'/proc/{int(pid)}/cmdline').read_bytes().split(b'\0')
    except OSError:
        raise RuntimeError('Registered server is not running; no process terminated')
    argv = [a.decode('utf8', 'replace') for a in argv if a]
    try:
        running = Path(os.readlink(f'/proc/{int(pid)}/exe')).resolve()
    except OSError:
        running = None
    target = Path(executable).resolve()
    same_binary = running == target or any(Path(a).resolve() == target for a in argv[:2] if a.startswith('/'))
    if not same_binary or expected not in ' '.join(argv):
        raise RuntimeError('Registered server identity changed; no process terminated')


def kill_server_tree(pid):
    if WINDOWS:
        subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True, check=True)
        return
    try:
        pgid = os.getpgid(pid)
    except OSError:
        return
    os.killpg(pgid, signal.SIGTERM)
    for _ in range(50):
        try:
            os.killpg(pgid, 0)
        except OSError:
            return
        time.sleep(0.1)
    os.killpg(pgid, signal.SIGKILL)
