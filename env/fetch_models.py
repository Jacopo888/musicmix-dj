"""Download the large official model weights the pipeline uses, pinned by revision and SHA-256.

    python env/fetch_models.py [--target /opt/musicmix/models] [--critics]
    python env/fetch_models.py --link     # symlink the weights into this checkout's .cache

Small files that sit next to the weights (configs, provenance.json, vendored
sources at fixed revisions, Beat This!/LinkSeg/SSM-Net checkpoints) travel in
the un archivio dati privato bundle; this script fetches only the big weights from their
original hosts. Hashes are the ones of the files used on the desktop runs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = Path(os.environ.get('MUSICMIX_MODELS') or Path(os.environ.get('MUSICMIX_HOME', '/opt/musicmix')) / 'models')
HF = 'https://huggingface.co/{repo}/resolve/{rev}/{file}'
# (repo-relative destination, URL, sha256, group)
FILES = [
    ('.cache/cue_runtime/checkpoint/model.safetensors',
     HF.format(repo='disco-eth/cue-detr', rev='caca89ab48f3187a98beb0fdf782e7f0626dcad2', file='model.safetensors'),
     '195b80f5c94e4b424b417a7c04e0aac238082c584ddcb65d01be8a97c69cadd3', 'core'),
    ('torch/hub/checkpoints/955717e8-8726e21a.th',
     'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th',
     '8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4', 'core'),
    ('.cache/models/audiobox_aesthetics/model.safetensors',
     HF.format(repo='facebook/audiobox-aesthetics', rev='9b1dd8e5df9af7216e836a98974fe3b82c56ded6', file='model.safetensors'),
     'a5a3c2412649cc2384ec525ffd5180ce6c4778f43bed6108e0a1303de04d014e', 'critics'),
    ('.cache/song_eval_runtime/MuQ-large-msd-iter/model.safetensors',
     HF.format(repo='OpenMuQ/MuQ-large-msd-iter', rev='0562a57814f6f8bbd9fdea0a25921a2fce1a841a', file='model.safetensors'),
     '273febab2be02872c37d2c37e48a9d6c52c1c9392f3eeeabd498efa281ccb7a6', 'critics'),
    ('.cache/song_eval_runtime/weights/songeval-model.safetensors',
     'https://github.com/ASLP-lab/SongEval/raw/848fb2ff3a2a9d64dcb20d46f07238c28abd7add/ckpt/model.safetensors',
     '0cb1eb8be02b96e9949ca3a2b66997c5ac7d9da3567240df4dbfc2048eb98efd', 'critics'),
]
# Hugging Face cache snapshots (layout models--org--name/snapshots/<rev>), verified by commit.
SNAPSHOTS = [
    ('.cache/models', 'Systran/faster-whisper-small.en', 'd1d751a5f8271d482d14ca55d9e2deeebbae577f', None),
    ('.cache/models', 'Systran/faster-whisper-base.en', '3d3d5dee26484f91867d81cb899cfcf72b96be6c', None),
    # CUE-DETR loads its image processor by name from the global HF cache.
    ('hf/hub', 'facebook/detr-resnet-50', None, ['preprocessor_config.json', 'config.json']),
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256(destination) == expected:
        return 'present'
    partial = destination.with_suffix(destination.suffix + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'musicmix-cloud-setup'})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as out:
        shutil.copyfileobj(response, out, 1 << 20)
    actual = sha256(partial)
    if actual != expected:
        partial.unlink()
        raise RuntimeError(f'{destination.name}: sha256 {actual} != pinned {expected}')
    partial.replace(destination)
    return 'downloaded'


def fetch(target, critics):
    report = {}
    for relative, url, expected, group in FILES:
        if group == 'critics' and not critics:
            continue
        report[relative] = download(url, target/relative, expected)
    from huggingface_hub import snapshot_download
    for cache, repo, revision, patterns in SNAPSHOTS:
        path = snapshot_download(repo, revision=revision, cache_dir=str(target/cache), allow_patterns=patterns)
        report[f'{cache}/{repo}'] = path
    (target/'MODELS_OK.json').write_text(json.dumps(report, indent=2))
    return report


def link(target):
    """Symlink every fetched weight into the checkout at the path the tools expect."""
    linked = []
    for relative, _, _, _ in FILES:
        source = target/relative
        if not source.is_file() or relative.startswith('torch/'):
            continue
        destination = ROOT/relative; destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() or destination.exists():
            destination.unlink()
        destination.symlink_to(source); linked.append(relative)
    for model in (target/'.cache/models').glob('models--*'):
        destination = ROOT/'.cache/models'/model.name; destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.symlink_to(model); linked.append(str(destination.relative_to(ROOT)))
    return linked


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--target', type=Path, default=DEFAULT_TARGET)
    parser.add_argument('--critics', action='store_true', help='Also fetch Audiobox, MuQ and SongEval (~1.9 GB)')
    parser.add_argument('--link', action='store_true')
    args = parser.parse_args()
    result = {'linked': link(args.target)} if args.link else fetch(args.target, args.critics)
    print(json.dumps(result, indent=2))
    sys.exit(0)
