"""Canonical offline decoding, with source and decoder provenance."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
import uuid


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def decode_audio(source: Path, destination: Path) -> dict:
    source = source.resolve(strict=True)
    if not source.is_file():
        raise ValueError(f'Not an audio file: {source}')
    executable = shutil.which('ffmpeg')
    if not executable:
        raise RuntimeError('FFmpeg is required for canonical decoding; install ffmpeg and add it to PATH.')
    if destination.exists():
        raise ValueError(f'Destination already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    version = subprocess.run([executable, '-version'], capture_output=True, text=True,
                             check=True, timeout=15).stdout.splitlines()[0]
    temporary = destination.with_name(f'.{destination.stem}.{uuid.uuid4().hex}.wav')
    command = [executable, '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
               '-i', str(source), '-map', '0:a:0', '-vn', '-ar', '48000', '-ac', '2',
               '-c:a', 'pcm_s16le', '-map_metadata', '-1', str(temporary)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        if result.returncode:
            raise ValueError(f'Audio decoding failed: {result.stderr.strip()[-2000:]}')
        temporary.rename(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {'input_uri': str(source), 'source_sha256': sha256_file(source),
            'canonical_audio_uri': str(destination.resolve()),
            'canonical_sha256': sha256_file(destination), 'sample_rate': 48000,
            'channels': 2, 'codec': 'pcm_s16le', 'decoder_version': version,
            'offset_policy': 'FFmpeg decoded sample zero; source encoder delay not independently verified'}
