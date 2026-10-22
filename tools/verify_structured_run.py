"""Verify persisted run provenance, known word protection and rendered PCM."""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import sys
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from musicmix.audio import load_audio
from musicmix.paths import localize


def digest(path):
    return hashlib.sha256(localize(path).read_bytes()).hexdigest()


def verify(root):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    search = json.loads((root / 'search.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'complete'
    for track in manifest['tracks']:
        # track_id hashes audio samples and metadata, not the WAV container.
        audio = load_audio(localize(track['path']))
        assert audio.track_id == track['sha256']
        assert audio.sample_rate == track['sample_rate']
        assert audio.duration_samples == track['duration_samples']
    evidence = {}
    for name, entry in manifest['evidence'].items():
        assert digest(entry['snapshot']) == entry['sha256'], name
        evidence[name] = json.loads(localize(entry['snapshot']).read_text(encoding='utf-8'))
    results = []
    spacing = manifest['options'].get('min_cue_spacing_seconds')
    if spacing is not None:
        selected = search['items']
        for index, item in enumerate(selected):
            for other in selected[index+1:]:
                for name in ('cue_out_sample', 'cue_in_sample'):
                    assert abs(item['candidate'][name]-other['candidate'][name]) >= round(spacing*48000)
                for side in ('outgoing', 'incoming'):
                    assert item['alignment'][side]['requested_sample'] != other['alignment'][side]['requested_sample']
    for proposal, artifact in zip(search['items'], manifest['artifacts'], strict=True):
        assert digest(artifact['path']) == artifact['sha256']
        assert artifact['anchor_offset_samples'] == 0
        alignment = proposal['alignment']
        assert alignment['relative_shift_seconds'] <= .08
        for side, doc, interval in (
            ('outgoing', evidence['vocals_a'], alignment['fade_interval_a']),
            ('incoming', evidence['vocals_b'], alignment['fade_interval_b'])):
            assert alignment[side]['safe'] is True
            assert abs(alignment[side]['shift_samples']) <= round(.08 * doc['sample_rate'])
            start, end = interval
            assert not any(w['start_sample'] < end and w['end_sample'] > start for w in doc['words'])
            assert not any(w['start_sample'] < end and w['end_sample'] > start for w in doc['uncertain_regions'])
        with wave.open(artifact['path'], 'rb') as stream:
            assert stream.getnchannels() == 2
            assert stream.getframerate() == 48000
            assert stream.getnframes() == 960000
            assert stream.getsampwidth() == 2
            samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2')
        clipped = int(np.count_nonzero((samples == -32768) | (samples == 32767)))
        assert clipped == 0
        results.append({'candidate_id': artifact['candidate_id'], 'duration_seconds': 20,
                        'pcm_peak': float(np.max(np.abs(samples.astype(float))) / 32768),
                        'clipped_samples': clipped, 'known_words_overlap_fade': False})
    report = {'status': 'passed', 'scope': 'automatic integrity checks; no perceptual validation',
              'run': str(root.resolve()), 'selected_count': len(results), 'audio': results,
              'unique_outgoing_cues': len({p['candidate']['cue_out_sample'] for p in search['items']}),
              'unique_incoming_cues': len({p['candidate']['cue_in_sample'] for p in search['items']}),
              'diversity_spacing_seconds_verified': spacing,
              'rejections': dict(Counter(r['reason'] for r in search['rejected']))}
    (root / 'QA.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    for root in parser.parse_args().runs:
        report = verify(root)
        print(json.dumps({key: value for key, value in report.items() if key != 'audio'}))
