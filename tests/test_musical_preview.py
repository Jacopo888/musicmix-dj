import tempfile
from pathlib import Path
import unittest
import wave
import numpy as np

from musicmix.audio import AudioTrack
from musicmix.contracts import Candidate
from musicmix.musical_preview import render_handoff


class MusicalPreviewTests(unittest.TestCase):
    def test_anchor_join_duration_and_no_overlap(self):
        rate = 1000
        left = AudioTrack('a', 'a', rate, np.full((4000, 2), .4))
        right = AudioTrack('b', 'b', rate, np.full((4000, 2), -.2))
        candidate = Candidate('id', 'a', 'b', 2000, 1000, 'structure', {}, {})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'test.wav'
            artifact = render_handoff(left, right, candidate, path, context_seconds=1, fade_seconds=.01)
            self.assertEqual(artifact['join_sample'], 1000)
            self.assertEqual(artifact['duration_samples'], 2000)
            self.assertEqual(artifact['anchor_offset_samples'], 0)
            self.assertEqual(artifact['clipped_samples'], 0)
            with wave.open(str(path)) as stream:
                signal = np.frombuffer(stream.readframes(stream.getnframes()), '<i2').reshape(-1, 2)
            self.assertEqual(int(signal[999, 0]), 0)
            self.assertEqual(int(signal[1000, 0]), 0)
            self.assertGreater(int(signal[989, 0]), 0)
            self.assertLess(int(signal[1010, 0]), 0)
            with self.assertRaises(FileExistsError):
                render_handoff(left, right, candidate, path, context_seconds=1)

    def test_invalid_context_or_identity(self):
        track = AudioTrack('a', 'a', 1000, np.zeros((4000, 2)))
        other = AudioTrack('b', 'b', 1000, np.zeros((4000, 2)))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'bad.wav'
            candidate = Candidate('id', 'a', 'b', 10, 1000, 'structure', {}, {})
            with self.assertRaises(ValueError):
                render_handoff(track, other, candidate, path)
            with self.assertRaises(ValueError):
                render_handoff(track, other, candidate, path, context_seconds=float('nan'))
