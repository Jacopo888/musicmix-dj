"""Verify source identity, PCM interiors, controlled A context and WAV uniqueness."""
import argparse
import json
from pathlib import Path
import sys
import wave
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from musicmix.audio import load_audio
from musicmix.ingest import sha256_file
from musicmix.expectation_pipeline import read_cache
from musicmix.paths import localize


def audit(output):
    report = json.loads((output/'experiment.json').read_text(encoding='utf-8'))
    cache = read_cache(localize(report['inputs'][0]['path']))
    tracks = {key: load_audio(source['path']) for key, source in cache['source_records'].items()}
    hashes, results = set(), []
    for direction, run in report['directions'].items():
        by_trial = {}
        for artifact in run['artifacts']:
            path = localize(artifact['path'])
            digest = sha256_file(path)
            assert digest == artifact['sha256'] and digest not in hashes
            hashes.add(digest)
            candidate = cache['candidate_map'][artifact['candidate_id']]['baseline']['proposal']['candidate']
            comparison = run['comparisons'][artifact['trial']-1][artifact['system']]
            assert comparison['candidate_id'] == artifact['candidate_id']
            assert artifact['sample_rate'] == 48000 and artifact['duration_samples'] == 960000
            with wave.open(str(path), 'rb') as reader:
                assert reader.getframerate() == 48000 and reader.getnframes() == 960000
                assert reader.getnchannels() == 2 and reader.getsampwidth() == 2
                pcm = np.frombuffer(reader.readframes(reader.getnframes()), dtype='<i2').reshape(-1, 2)
            # Exclude only the documented outer edge ramps and 12 ms join ramps.
            intervals = [(480, 479424, 'track_a', 'cue_out_sample', -480000),
                         (480576, 959520, 'track_b', 'cue_in_sample', -480000)]
            for start, end, track_key, cue_key, offset in intervals:
                source = tracks[candidate[track_key]]
                assert source.track_id == candidate[track_key]
                source_start = candidate[cue_key]+start+offset
                expected = np.round(.9*source.samples[source_start:source_start+end-start].astype(np.float64)*32767).astype('<i2')
                assert np.array_equal(pcm[start:end], expected), (path, track_key)
            by_trial.setdefault(artifact['trial'], []).append(pcm[:480000].copy())
            results.append({'path': str(path), 'sha256': digest, 'canonical_pcm_interiors_match': True})
        for outgoing in by_trial.values():
            assert len(outgoing) == 2 and np.array_equal(*outgoing)
        listening = Path(run['listening']['path']).parent
        key = json.loads((listening/'key.json').read_text(encoding='utf-8'))
        for trial in key['trials']:
            copied = listening/f"trial_{trial['trial']:02d}_{trial['label']}.wav"
            assert sha256_file(copied) == sha256_file(localize(trial['source']))
    assert results, 'No rendered comparisons to audit'
    result = {'status': 'pass', 'wav_count': len(results), 'unique_sha256_count': len(hashes),
              'same_outgoing_context_each_trial': True, 'listening_copies_match': True,
              'audit_code_sha256': sha256_file(Path(__file__)), 'files': results}
    (output/'audio_qa.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = audit(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))
