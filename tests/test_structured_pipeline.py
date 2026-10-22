from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from musicmix.audio import load_audio
from musicmix.contracts import Candidate
from musicmix.directional import DirectionalProposal, DirectionalSearchResult
from musicmix.structured_pipeline import run_structured


def fixture(root):
    rate = 48000
    for name, frequency in [('a', 220), ('b', 330)]:
        t = np.arange(24*rate)/rate
        pcm = np.round(.3*np.sin(2*np.pi*frequency*t)*32767).astype('<i2')
        with wave.open(str(root/f'{name}.wav'), 'wb') as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(rate)
            stream.writeframes(pcm.tobytes())
        track = load_audio(root/f'{name}.wav')
        payload = {'schema_version': 'test.1', 'track_id': track.track_id,
                   'sample_rate': rate, 'duration_samples': track.duration_samples}
        for kind in ['structure', 'vocals']:
            (root/f'{kind}_{name}.json').write_text(json.dumps(payload), encoding='utf-8')
    a, b = load_audio(root/'a.wav'), load_audio(root/'b.wav')
    candidate = Candidate('fixture', a.track_id, b.track_id, 12*rate, 2*rate,
                          'structure', {}, {})
    return candidate


def arguments(root):
    return [root/'a.wav', root/'b.wav', root/'structure_a.json', root/'structure_b.json',
            root/'vocals_a.json', root/'vocals_b.json', root/'run']


class StructuredPipelineTests(unittest.TestCase):
    def test_structured_baseline_and_random_controls_keep_run_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); candidate = fixture(root)
            second = replace(candidate, candidate_id='second', cue_out_sample=13*48000,
                             cue_in_sample=3*48000)
            def proposal(value):
                return DirectionalProposal(value, .7, {}, {}, .02,
                                           {'outgoing': {'safe': True}, 'incoming': {'safe': True}})
            items = (proposal(candidate), proposal(second))
            result = DirectionalSearchResult(items=items, rejected=(), diagnostics={'algorithm': 'test'}, proposals=items)
            baseline = root/'baseline'; baseline.mkdir()
            (baseline/'search.json').write_text(json.dumps({'items': [asdict(items[0])]}))
            with patch('musicmix.directional.search_directional', return_value=result):
                manifest = run_structured(*arguments(root), baseline_run=baseline,
                                          listening_title='Forward V4', random_controls=True)
            self.assertEqual(manifest['random_control']['pool_size'], 2)
            self.assertEqual(manifest['random_control']['seed'], 314159)
            self.assertEqual(manifest['listening']['run_id'], manifest['run_id'])
            self.assertIn('Forward_V4', manifest['listening']['feedback_filename'])
            self.assertTrue(manifest['listening']['dedup']['enabled'])
            key = json.loads((root/'run/listening_ab/key.json').read_text())
            hashes = [item['audio_sha256'] for item in key['trials']]
            self.assertEqual(len(hashes), len(set(hashes)))

    def test_rejects_cross_track_evidence_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); fixture(root)
            payload = json.loads((root/'vocals_a.json').read_text())
            payload['track_id'] = 'another_track'
            (root/'vocals_a.json').write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'track_id'):
                run_structured(*arguments(root))
            self.assertFalse((root/'run').exists())

    def test_supported_proposal_renders_and_input_snapshot_is_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); candidate = fixture(root)
            proposal = DirectionalProposal(candidate, .7, {}, {}, .02,
                                           {'outgoing': {'safe': True}, 'incoming': {'safe': True}})
            result = DirectionalSearchResult(items=(proposal,), rejected=(), diagnostics={})
            with patch('musicmix.directional.search_directional', return_value=result):
                manifest = run_structured(*arguments(root))
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['selected_count'], 1)
            self.assertEqual(manifest['artifacts'][0]['clipped_samples'], 0)
            (root/'vocals_a.json').write_text('{}')
            snapshot = json.loads((root/'run/inputs/vocals_a.json').read_text())
            self.assertEqual(snapshot['track_id'], candidate.track_a)

    def test_unknown_word_boundary_abstains_with_failure_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); candidate = fixture(root)
            proposal = DirectionalProposal(candidate, .7, {}, {}, .02,
                                           {'outgoing': {'safe': False}, 'incoming': {'safe': True}})
            result = DirectionalSearchResult(items=(proposal,), rejected=(), diagnostics={})
            with patch('musicmix.directional.search_directional', return_value=result):
                with self.assertRaisesRegex(ValueError, 'vocal boundaries'):
                    run_structured(*arguments(root))
            manifest = json.loads((root/'run/manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(manifest['artifacts'], [])
