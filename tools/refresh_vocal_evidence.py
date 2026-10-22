"""Recompute boundary evidence from saved stems and ASR, without model inference."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from musicmix.audio import load_audio
from musicmix.paths import localize
from musicmix.vocal_boundaries import acoustic_intervals, build_boundary_document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    for label in ('a', 'b'):
        path = args.directory / f'{label}.json'
        old = json.loads(path.read_text(encoding='utf-8'))
        if old['source_kind'] != 'separated_vocals' or old['separator']['status'] != 'complete':
            raise ValueError('A completed separated vocal stem is required')
        original = load_audio(localize(old['canonical_audio_uri']))
        stem = load_audio(args.directory / f'{label}_vocals.wav')
        if stem.sample_rate != original.sample_rate or stem.duration_samples != original.duration_samples:
            raise ValueError('Stem timeline mismatch')
        acoustic, diagnostics = acoustic_intervals(stem)
        document = build_boundary_document(
            localize(old['canonical_audio_uri']), words=old['words'],
            asr_segments=old['asr_segments'], source_kind=old['source_kind'],
            backend=old['backend'], separator=old['separator'],
            warnings=old['warnings'], acoustic=acoustic,
            acoustic_diagnostics=diagnostics, transcript=old['transcript'])
        document['analysis_status'] = old['analysis_status']
        document['boundary_code_sha256'] = hashlib.sha256(
            (ROOT / 'src/musicmix/vocal_boundaries.py').read_bytes()).hexdigest()
        document['refresh_source'] = 'cached separated stem and cached ASR; no repeated inference'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        temporary.replace(path)
        print(json.dumps({'track': label, 'words': len(document['words']),
                          'phrases': len(document['phrases']), 'acoustic': diagnostics}))


if __name__ == '__main__':
    main()
