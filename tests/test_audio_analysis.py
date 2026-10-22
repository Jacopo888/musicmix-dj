"""Focused offline tests for PCM ingestion and the M1 descriptor baseline."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from musicmix.analysis import ANALYSIS_VERSION, _stft_strength, analyze_track
from musicmix.audio import AudioFormatError, AudioTrack, load_audio


def _write_pcm(path: Path, samples: np.ndarray, sample_rate: int, width: int) -> None:
    samples = np.asarray(samples)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1 if samples.ndim == 1 else samples.shape[1])
        wav.setsampwidth(width)
        wav.setframerate(sample_rate)
        if width == 2:
            payload = np.asarray(np.clip(samples, -1, 1) * 32767, dtype="<i2").tobytes()
        elif width == 4:
            payload = np.asarray(np.clip(samples, -1, 1) * 2147483647, dtype="<i4").tobytes()
        elif width == 3:
            values = np.asarray(np.clip(samples, -1, 1) * 8388607, dtype=np.int32).reshape(-1)
            payload = bytearray()
            for value in values.tolist():
                payload.extend(int(value).to_bytes(3, "little", signed=True))
            payload = bytes(payload)
        else:
            payload = np.asarray(np.clip(samples, -1, 1) * 127, dtype=np.int8).tobytes()
        wav.writeframes(payload)


class TestAudioLoading(unittest.TestCase):
    def test_mono_is_stereo_and_hash_is_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mono.wav"
            source = np.array([-1.0, -0.25, 0.0, 0.5, 0.99], dtype=np.float32)
            _write_pcm(path, source, 8000, 2)
            track = load_audio(path)
            self.assertEqual(track.samples.shape, (source.size, 2))
            self.assertEqual(track.sample_rate, 8000)
            self.assertEqual(track.duration_samples, source.size)
            self.assertTrue(np.allclose(track.samples[:, 0], track.samples[:, 1]))
            self.assertEqual(track.track_id, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_24_and_32_bit_integer_pcm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for width in (3, 4):
                path = Path(directory) / f"integer_{width}.wav"
                source = np.array([-0.8, -0.1, 0.2, 0.75], dtype=np.float32)
                _write_pcm(path, source, 16000, width)
                track = load_audio(path)
                self.assertEqual(track.samples.shape, (source.size, 2))
                self.assertTrue(np.allclose(track.samples[:, 0], source, atol=2e-6))

    def test_8_bit_pcm_is_explicitly_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "8bit.wav"
            _write_pcm(path, np.zeros(16), 8000, 1)
            with self.assertRaises(AudioFormatError):
                load_audio(path)


class TestAnalysis(unittest.TestCase):
    def test_stft_flux_preserves_first_batch_and_batch_boundary_changes(self) -> None:
        # Use non-overlapping windows so frame 255 is active, frame 256 is
        # silent, and frame 257 is active.  A buggy implementation that only
        # compares a whole batch with its predecessor misses both the first
        # batch's flux and the onset at frame 257.
        n_fft = 64
        hop = 64
        frames = 258
        signal = np.zeros(frames * hop, dtype=np.float64)
        for frame in (5, 255, 257):
            start = frame * hop + 8
            signal[start : start + 48] = np.hanning(48)
        centres, rms, flux = _stft_strength(signal, n_fft, hop)
        self.assertEqual(len(centres), frames)
        self.assertGreater(rms[5], 0.0)
        self.assertGreater(rms[255], 0.0)
        self.assertGreater(rms[257], 0.0)
        self.assertGreater(float(flux[5]), 1e-8)
        self.assertGreater(float(flux[257]), 1e-8)

    def test_context_nan_is_rejected_before_minimum_context_clamp(self) -> None:
        track = AudioTrack("fixture", "fixture.wav", 8000, np.zeros((40000, 2)))
        with self.assertRaises(ValueError):
            analyze_track(track, context_seconds=float("nan"))

    def test_silence_and_short_tracks_emit_no_reliable_cues(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            short_path = Path(directory) / "short.wav"
            _write_pcm(short_path, np.zeros(8000 * 3), 8000, 2)
            short = analyze_track(load_audio(short_path))
            self.assertEqual(short.cues, ())
            self.assertIn(short.diagnostics["status"], {"short_track", "silence"})

    def test_click_train_has_internal_onset_cues_and_fixed_shapes(self) -> None:
        sample_rate = 8000
        duration = 8
        signal = np.zeros(sample_rate * duration, dtype=np.float32)
        for second in np.arange(0.0, duration, 0.5):
            index = int(second * sample_rate)
            signal[index : index + 24] = np.hanning(24)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clicks.wav"
            _write_pcm(path, signal, sample_rate, 2)
            result = analyze_track(load_audio(path))
            self.assertEqual(result.backend, ANALYSIS_VERSION)
            self.assertGreaterEqual(len(result.cues), 4)
            self.assertTrue(all(2 * sample_rate <= cue.sample <= 6 * sample_rate for cue in result.cues))
            self.assertTrue(all(len(cue.chroma) == 12 for cue in result.cues))
            self.assertTrue(all(cue.kind == "onset_grid" for cue in result.cues))
            self.assertTrue(all(cue.confidence >= 0.0 for cue in result.cues))


if __name__ == "__main__":
    unittest.main()
