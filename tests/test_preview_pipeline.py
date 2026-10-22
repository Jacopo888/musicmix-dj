"""Integration checks for actual audio boundaries and reproducible run artefacts."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import wave

import numpy as np

from musicmix.audio import load_audio
from musicmix.contracts import Candidate
from musicmix.preview import render_preview


def make_wav(path, frequency=220, seconds=12, rate=8000):
    t = np.arange(round(seconds*rate))/rate
    envelope = 0.3 + 0.7*np.exp(-((t % 0.5)/0.04)**2)
    signal = 0.6*envelope*np.sin(2*np.pi*frequency*t)
    samples = np.round(signal*32767).astype('<i2')
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(rate)
        stream.writeframes(samples.tobytes())


class TestPreview(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg not installed')
    def test_decode_failure_does_not_leave_partial_destination(self):
        from musicmix.ingest import decode_audio
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'invalid.bin'
            source.write_bytes(b'not an audio stream')
            with self.assertRaises(ValueError):
                decode_audio(source, root/'canonical.wav')
            self.assertFalse((root/'canonical.wav').exists())
            self.assertEqual(list(root.glob('*.wav')), [])

    def test_fixed_duration_headroom_and_endpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_wav(root/'a.wav'); make_wav(root/'b.wav', frequency=330)
            a = load_audio(root/'a.wav'); b = load_audio(root/'b.wav')
            c = Candidate('fixture', a.track_id, b.track_id, 4*8000, 3*8000,
                          'harmony_rhythm', {}, {})
            result = render_preview(a, b, c, root/'preview.wav')
            self.assertEqual(result['duration_samples'], 28000)
            self.assertEqual(result['clipped_samples'], 0)
            with wave.open(str(root/'preview.wav')) as stream:
                pcm = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2')
            self.assertEqual(int(pcm[0]), 0)
            self.assertEqual(int(pcm[-1]), 0)
            with self.assertRaises(ValueError):
                render_preview(a, b, c, root/'bad.wav', fade_seconds=3)
            invalid = Candidate('bad', a.track_id, b.track_id, 1, 3*8000,
                                'harmony_rhythm', {}, {})
            with self.assertRaises(ValueError):
                render_preview(a, b, invalid, root/'invalid.wav')

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg not installed')
    def test_end_to_end_provenance_and_replay(self):
        from musicmix.pipeline import run_search
        from musicmix.__main__ import load_candidate
        from musicmix.ranking import InterpretableRanker
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_wav(root/'a.wav'); make_wav(root/'b.wav', frequency=330)
            manifest = run_search(root/'a.wav', root/'b.wav', root/'run', top_k=3, max_candidates=100)
            self.assertEqual(manifest['status'], 'complete')
            self.assertGreater(manifest['selected_count'], 0)
            self.assertLessEqual(manifest['selected_count'], 3)
            self.assertEqual(manifest['tracks'][0]['sample_rate'], 48000)
            self.assertTrue((root/'run'/'REPORT.md').is_file())
            search = json.loads((root/'run'/'search.json').read_text(encoding='utf-8'))
            reports = {r['candidate_id']: r for r in search['reports']}
            for path in (root/'run'/'candidates').glob('*.json'):
                candidate = load_candidate(path)
                score = InterpretableRanker().score(candidate)
                self.assertEqual(score.score, reports[candidate.candidate_id]['score'])
            with self.assertRaises(FileExistsError):
                run_search(root/'a.wav', root/'b.wav', root/'run')


if __name__ == '__main__':
    unittest.main()
