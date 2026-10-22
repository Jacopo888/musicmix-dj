"""Local listening UI; save identified judgements to an explicit directory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from .evaluation import MOTION_LABELS


def save_judgements(payload: dict, tests: dict, directory: Path) -> Path:
    if payload.get('schema_version') != 'musicmix.listening.2':
        raise ValueError('Unsupported feedback schema')
    registered = tests.get(payload.get('test_id'))
    if registered is None:
        raise ValueError('Unknown listening test')
    metadata = registered['metadata']
    for name in ('run_id', 'title', 'feedback_filename'):
        if payload.get(name) != metadata.get(name):
            raise ValueError(f'Mismatched {name}')
    rows = payload.get('judgements')
    expected = json.loads((registered['path']/'judgements_template.json').read_text(encoding='utf-8'))
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError('Missing listening trials')
    seen = set()
    allowed = {None, 'A', 'B', 'tie', 'neither'}
    for row, original in zip(rows, expected):
        if (row.get('trial'), row.get('pair_id')) != (original['trial'], original['pair_id']):
            raise ValueError('Trial identity mismatch')
        if row['trial'] in seen or row.get('answer') not in allowed:
            raise ValueError('Invalid answer or duplicate trial')
        seen.add(row['trial'])
        if any(row.get(name) is not None and type(row[name]) is not bool for name in ('word_cut_a', 'word_cut_b')):
            raise ValueError('Invalid word-cut judgement')
        if not isinstance(row.get('comment', ''), str):
            raise ValueError('Invalid comment')
        for name in ('motion_a', 'motion_b'):
            value = row.get(name)
            if value is not None and (not isinstance(value, str) or value not in MOTION_LABELS):
                raise ValueError('Invalid motion judgement')
    directory.mkdir(parents=True, exist_ok=True)
    filename = metadata['feedback_filename']
    if not re.fullmatch(r'[a-zA-Z0-9_.-]+\.json', filename):
        raise ValueError('Invalid filename')
    document = {**payload, 'saved_utc': datetime.now(timezone.utc).isoformat()}
    # Repeated saves are separate revisions; original judgements stay intact.
    for revision in range(10000):
        suffix = '' if revision == 0 else f'_rev{revision:02d}'
        target = directory / f'{Path(filename).stem}{suffix}.json'
        try:
            with target.open('x', encoding='utf-8') as stream:
                json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
            return target.resolve()
        except FileExistsError:
            continue
    raise ValueError('Too many saved revisions')


def make_handler(tests: dict, feedback_directory: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, data, content_type='application/json; charset=utf-8'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(data)

        def serve_audio(self, path: Path, head_only: bool = False):
            size = path.stat().st_size
            start, end = 0, size-1
            partial = False
            requested = self.headers.get('Range') if not head_only else None
            if requested:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested.strip())
                try:
                    if not match or not any(match.groups()) or size == 0:
                        raise ValueError('Invalid byte range')
                    first, last = match.groups()
                    if first:
                        start = int(first)
                        end = min(int(last), size-1) if last else size-1
                    else:
                        suffix = int(last)
                        if suffix <= 0:
                            raise ValueError('Invalid suffix')
                        start = max(0, size-suffix)
                    if start >= size or end < start:
                        raise ValueError('Unsatisfiable byte range')
                    partial = True
                except ValueError:
                    self.send_response(416)
                    self.send_header('Content-Range', f'bytes */{size}')
                    self.send_header('Accept-Ranges', 'bytes')
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
            length = max(0, end-start+1)
            self.send_response(206 if partial else 200)
            self.send_header('Content-Type', 'audio/wav')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(length))
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if partial:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            if head_only:
                return
            try:
                with path.open('rb') as stream:
                    stream.seek(start)
                    remaining = length
                    while remaining:
                        block = stream.read(min(65536, remaining))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # Seeking can cancel an in-flight browser request.
                pass

        def do_GET(self):
            self.get_file()

        def do_HEAD(self):
            self.get_file(head_only=True)

        def get_file(self, head_only=False):
            parts = urlsplit(self.path).path.strip('/').split('/')
            if len(parts) != 3 or parts[0] != 'test' or parts[1] not in tests:
                return self.reply(404, b'{}')
            name = parts[2]
            if name != 'index.html' and not re.fullmatch(r'trial_\d+_[AB]\.wav', name):
                return self.reply(404, b'{}')
            path = tests[parts[1]]['path']/name
            if not path.is_file():
                return self.reply(404, b'{}')
            if name.endswith('.wav'):
                return self.serve_audio(path, head_only=head_only)
            self.reply(200, path.read_bytes(), 'text/html; charset=utf-8')

        def do_POST(self):
            if self.path != '/api/judgements':
                return self.reply(404, b'{}')
            # Only this local UI can submit judgements through a browser.
            host = self.headers.get('Host', '')
            origin = self.headers.get('Origin')
            if host != f'127.0.0.1:{self.server.server_port}' or (origin and origin != f'http://{host}'):
                return self.reply(403, b'{"error":"Invalid origin"}')
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1_000_000:
                    raise ValueError('Invalid request length')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('Invalid JSON object')
                path = save_judgements(payload, tests, feedback_directory)
                self.reply(200, json.dumps({'saved_path': str(path)}).encode())
            except (ValueError, TypeError, KeyError) as error:
                self.reply(400, json.dumps({'error': str(error)}).encode())
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='append', type=Path, required=True)
    parser.add_argument('--feedback-directory', type=Path,
                        default=Path(__file__).resolve().parents[3]/'giudizi json')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    tests = {}
    for path in args.test:
        path = path.resolve(strict=True)
        metadata = json.loads((path/'test.json').read_text(encoding='utf-8'))
        if metadata['test_id'] in tests:
            raise ValueError('Duplicate test identifier')
        tests[metadata['test_id']] = {'metadata': metadata, 'path': path}
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(tests, args.feedback_directory.resolve()))
    for test_id in tests:
        print(f'http://127.0.0.1:{server.server_port}/test/{test_id}/index.html', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
