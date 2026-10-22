"""Focused tests for the controlled equal-power overlap study renderer."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from musicmix.audio import AudioTrack
from musicmix.blend_preview import (
    BLEND_PREVIEW_VERSION,
    GLOBAL_GAIN,
    overlap_vocal_safety,
    render_blend_preview,
)
from musicmix.contracts import Candidate


RATE = 1000
CONTEXT = 0.1
OVERLAP = 0.04
JOIN = 100
HALF = 20


def _track(track_id: str, samples: np.ndarray, rate: int = RATE) -> AudioTrack:
    return AudioTrack(track_id, f"{track_id}.wav", rate, np.asarray(samples, dtype=np.float32))


def _candidate(a: AudioTrack, b: AudioTrack, out_cue: int = JOIN,
               in_cue: int = JOIN) -> Candidate:
    return Candidate("blend_fixture", a.track_id, b.track_id, out_cue, in_cue,
                     "harmony_rhythm", {}, {})


def _impulse_tracks(amplitude: float = 0.5, length: int = 200):
    a_samples = np.zeros((length, 2), dtype=np.float32)
    b_samples = np.zeros((length, 2), dtype=np.float32)
    a_samples[JOIN, :] = amplitude
    b_samples[JOIN, :] = amplitude
    return _track("a", a_samples), _track("b", b_samples)


def _read_pcm(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        rate = stream.getframerate()
        raw = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
    return raw.reshape(-1, channels), rate


def _document(track_id: str, coverage, *, duration: int = 200,
              words=(), rate: int = RATE) -> dict:
    return {
        "track_id": track_id,
        "sample_rate": rate,
        "duration_samples": duration,
        "coverage_intervals": list(coverage),
        "words": list(words),
    }


def _silence_document(track_id: str, status: str = "acoustic_silence_supported") -> dict:
    return _document(track_id, [{"start_sample": 0, "end_sample": 200,
                                 "status": status, "evidence": "test"}])


class TestRenderBlendPreview(unittest.TestCase):
    def test_source_impulse_anchors_coincide_at_join(self):
        a, b = _impulse_tracks()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "blend.wav"
            result = render_blend_preview(a, b, _candidate(a, b), path,
                                          context_seconds=CONTEXT,
                                          overlap_seconds=OVERLAP)
            pcm, rate = _read_pcm(path)
            self.assertEqual(rate, RATE)
            self.assertEqual(result["render_version"], BLEND_PREVIEW_VERSION)
            self.assertEqual(result["join_sample"], JOIN)
            self.assertEqual(result["duration_samples"], 200)
            self.assertEqual(len(pcm), 200)
            # Both source cues are exactly at output sample JOIN.
            expected = GLOBAL_GAIN * 0.5 * (np.cos(np.pi / 4) + np.sin(np.pi / 4))
            self.assertAlmostEqual(pcm[JOIN, 0] / 32767.0, expected, delta=1.5 / 32767)
            self.assertLessEqual(abs(int(pcm[JOIN - 1, 0])), 1)
            self.assertLessEqual(abs(int(pcm[JOIN + 1, 0])), 1)
            self.assertEqual(result["source_windows"]["a"]["start_sample"], 0)
            self.assertEqual(result["source_windows"]["a"]["end_sample"], JOIN + HALF)
            self.assertEqual(result["source_windows"]["b"]["start_sample"], JOIN - HALF)
            self.assertEqual(result["source_windows"]["b"]["end_sample"], 200)
            self.assertEqual(result["source_windows"]["a"]["cue_output_sample"], JOIN)
            self.assertEqual(result["source_windows"]["b"]["cue_output_sample"], JOIN)
            self.assertEqual(result["anchor_offsets"]["a"]["cue_offset_in_window_samples"], JOIN)
            self.assertEqual(result["anchor_offsets"]["b"]["cue_offset_in_window_samples"], HALF)
            self.assertEqual(result["overlap_timeline"]["overlap"],
                             {"start_sample": JOIN - HALF, "end_sample": JOIN + HALF})
            self.assertEqual(result["overlap_timeline"]["outgoing_extension"],
                             {"start_sample": JOIN, "end_sample": JOIN + HALF})
            self.assertEqual(result["overlap_timeline"]["incoming_preroll"],
                             {"start_sample": JOIN - HALF, "end_sample": JOIN})
            self.assertAlmostEqual(result["gains"]["a_at_join_raw"], np.sqrt(0.5), places=12)
            self.assertAlmostEqual(result["gains"]["b_at_join_raw"], np.sqrt(0.5), places=12)
            self.assertLessEqual(result["gains"]["global"], 0.7)
            self.assertEqual(result["vocal_safety"]["status"], "not_evaluated")
            self.assertEqual(result["overlap_samples"], 2 * HALF)
            self.assertIn("sha256", result)
            self.assertEqual(result["checksum_algorithm"], "sha256")
            self.assertTrue(any("overlap_vocal_safety" in warning
                                for warning in result["warnings"]))

    def test_distinct_source_cues_map_to_the_same_join(self):
        length = 260
        a_samples = np.zeros((length, 2), dtype=np.float32)
        b_samples = np.zeros((length, 2), dtype=np.float32)
        a_samples[120, :] = 0.5
        b_samples[150, :] = 0.5
        a = _track("a", a_samples)
        b = _track("b", b_samples)
        candidate = _candidate(a, b, out_cue=120, in_cue=150)
        with tempfile.TemporaryDirectory() as folder:
            result = render_blend_preview(a, b, candidate, Path(folder) / "blend.wav",
                                          context_seconds=CONTEXT, overlap_seconds=OVERLAP)
            pcm, _ = _read_pcm(Path(folder) / "blend.wav")
            expected = GLOBAL_GAIN * 0.5 * (np.cos(np.pi / 4) + np.sin(np.pi / 4))
            self.assertAlmostEqual(pcm[JOIN, 0] / 32767.0, expected, delta=1.5 / 32767)
            self.assertEqual(result["source_windows"]["a"]["start_sample"], 20)
            self.assertEqual(result["source_windows"]["b"]["start_sample"], 130)
            self.assertEqual(result["anchor_offsets"]["a"]["output_anchor_sample"], JOIN)
            self.assertEqual(result["anchor_offsets"]["b"]["output_anchor_sample"], JOIN)

    def test_default_context_is_twenty_seconds(self):
        rate = 8000
        length = 22 * rate
        a = _track("a", np.zeros((length, 2)), rate)
        b = _track("b", np.zeros((length, 2)), rate)
        candidate = Candidate("blend_fixture", a.track_id, b.track_id,
                              11 * rate, 11 * rate, "harmony_rhythm", {}, {})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "blend.wav"
            result = render_blend_preview(a, b, candidate, path)
            self.assertEqual(result["duration_samples"], 20 * rate)
            self.assertEqual(result["join_sample"], 10 * rate)
            self.assertAlmostEqual(result["duration_seconds"], 20.0)
            pcm, _ = _read_pcm(path)
            self.assertEqual(len(pcm), 20 * rate)

    def test_equal_power_full_scale_does_not_clip(self):
        length = 200
        a = _track("a", np.ones((length, 2)), RATE)
        b = _track("b", np.ones((length, 2)), RATE)
        with tempfile.TemporaryDirectory() as folder:
            result = render_blend_preview(a, b, _candidate(a, b), Path(folder) / "blend.wav",
                                          context_seconds=CONTEXT, overlap_seconds=OVERLAP)
            self.assertEqual(result["clipped_samples"], 0)
            self.assertLessEqual(result["peak"], GLOBAL_GAIN * np.sqrt(2.0) + 1e-12)
            self.assertLessEqual(result["peak"], 1.0)

    def test_tempo_mismatch_fails_and_half_double_is_normalized(self):
        a, b = _impulse_tracks()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "blend.wav"
            with self.assertRaisesRegex(ValueError, "Tempo mismatch"):
                render_blend_preview(a, b, _candidate(a, b), path,
                                     context_seconds=CONTEXT, overlap_seconds=OVERLAP,
                                     tempo_a=120.0, tempo_b=150.0)
            self.assertFalse(path.exists())
            result = render_blend_preview(a, b, _candidate(a, b), path,
                                          context_seconds=CONTEXT, overlap_seconds=OVERLAP,
                                          tempo_a=120.0, tempo_b=240.0)
            self.assertEqual(result["beat_coherence"]["normalized_ratio"], 1.0)
            self.assertEqual(result["beat_coherence"]["normalization_convention"],
                             "b_is_double_time_halved_to_compare")
            self.assertEqual(result["beat_coherence"]["status"],
                             "tempo_compatible_no_phase_evidence")

    def test_missing_tempo_reports_unknown_not_beat_match(self):
        a, b = _impulse_tracks()
        with tempfile.TemporaryDirectory() as folder:
            result = render_blend_preview(a, b, _candidate(a, b), Path(folder) / "blend.wav",
                                          context_seconds=CONTEXT, overlap_seconds=OVERLAP,
                                          tempo_a=120.0)
            self.assertEqual(result["beat_coherence"]["status"], "unknown")
            self.assertIn("tempo_b_missing", result["beat_coherence"]["reasons"])
            self.assertTrue(any("Beat coherence unknown" in warning
                                for warning in result["warnings"]))

    def test_missing_source_margins_are_rejected_without_padding(self):
        a, b = _impulse_tracks()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "blend.wav"
            with self.assertRaisesRegex(ValueError, "Insufficient outgoing context"):
                render_blend_preview(a, b, _candidate(a, b, out_cue=50), path,
                                     context_seconds=CONTEXT, overlap_seconds=OVERLAP)
            short_b = _track("b", np.zeros((110, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "Insufficient incoming context"):
                render_blend_preview(a, short_b, _candidate(a, short_b), path,
                                     context_seconds=CONTEXT, overlap_seconds=OVERLAP)
            with self.assertRaisesRegex(ValueError, "Insufficient outgoing extension"):
                render_blend_preview(a, b, _candidate(a, b, out_cue=195), path,
                                     context_seconds=CONTEXT, overlap_seconds=OVERLAP)
            self.assertFalse(path.exists())

    def test_identity_rate_channels_and_nonfinite_are_rejected(self):
        a, b = _impulse_tracks()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "blend.wav"
            wrong = Candidate("blend_fixture", "other", b.track_id, JOIN, JOIN,
                              "harmony_rhythm", {}, {})
            with self.assertRaisesRegex(ValueError, "identity"):
                render_blend_preview(a, b, wrong, path,
                                     context_seconds=CONTEXT, overlap_seconds=OVERLAP)
            other_rate = _track("b", np.zeros((200, 2)), rate=2000)
            with self.assertRaisesRegex(ValueError, "sample rates"):
                render_blend_preview(a, other_rate, _candidate(a, other_rate), path,
                                     context_seconds=CONTEXT, overlap_seconds=OVERLAP)
            mono = _track("b", np.zeros((200, 1), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "channel"):
                render_blend_preview(a, mono, _candidate(a, mono), path,
                                     context_seconds=CONTEXT, overlap_seconds=OVERLAP)
            nonfinite_samples = np.zeros((200, 2), dtype=np.float32)
            nonfinite_samples[JOIN, 0] = np.nan
            nonfinite = _track("b", nonfinite_samples)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                render_blend_preview(a, nonfinite, _candidate(a, nonfinite), path,
                                     context_seconds=CONTEXT, overlap_seconds=OVERLAP)

    def test_exclusive_creation_does_not_overwrite(self):
        a, b = _impulse_tracks()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "blend.wav"
            path.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                render_blend_preview(a, b, _candidate(a, b), path,
                                     context_seconds=CONTEXT, overlap_seconds=OVERLAP)
            self.assertEqual(path.read_bytes(), b"existing")


class TestOverlapVocalSafety(unittest.TestCase):
    def _candidate(self) -> Candidate:
        return Candidate("blend_fixture", "a", "b", JOIN, JOIN,
                         "harmony_rhythm", {}, {})

    def test_explicit_acoustic_silence_is_supported(self):
        result = overlap_vocal_safety(
            self._candidate(),
            _silence_document("a"),
            _silence_document("b"),
            RATE, OVERLAP,
        )
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["outgoing_extension"]["start_sample"], JOIN)
        self.assertEqual(result["outgoing_extension"]["end_sample"], JOIN + HALF)
        self.assertEqual(result["incoming_preroll"]["start_sample"], JOIN - HALF)
        self.assertEqual(result["incoming_preroll"]["end_sample"], JOIN)
        self.assertTrue(any("explicit_acoustic_silence" in reason
                            for reason in result["reasons"]))
        self.assertTrue(any("does not prove musical continuity" in warning
                            for warning in result["warnings"]))

    def test_asr_observed_gap_is_unknown_not_silence(self):
        result = overlap_vocal_safety(
            self._candidate(),
            _silence_document("a", status="asr_observed"),
            _silence_document("b", status="asr_observed"),
            RATE, OVERLAP,
        )
        self.assertEqual(result["status"], "unknown")
        self.assertTrue(any("asr_gap_is_not_acoustic_silence" in reason
                            for reason in result["reasons"]))

    def test_missing_coverage_is_unknown(self):
        empty = _document("a", [])
        result = overlap_vocal_safety(
            self._candidate(), empty, _document("b", []), RATE, OVERLAP,
        )
        self.assertEqual(result["status"], "unknown")
        self.assertTrue(any("missing_acoustic_evidence" in reason
                            for reason in result["reasons"]))

    def test_partial_acoustic_coverage_is_unknown(self):
        partial = _document("a", [{"start_sample": 0, "end_sample": 110,
                                   "status": "acoustic_silence_supported"}])
        result = overlap_vocal_safety(
            self._candidate(), partial, _silence_document("b"), RATE, OVERLAP,
        )
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["outgoing_extension"]["status"], "unknown")

    def test_measured_occupancy_or_known_word_rejects(self):
        occupied = _document("a", [{"start_sample": 0, "end_sample": 200,
                                    "status": "vocal_occupancy_supported"}])
        result = overlap_vocal_safety(
            self._candidate(), occupied, _silence_document("b"), RATE, OVERLAP,
        )
        self.assertEqual(result["status"], "rejected")
        worded = _document(
            "b",
            [{"start_sample": 0, "end_sample": 200,
              "status": "acoustic_silence_supported"}],
            words=[{"start_sample": 85, "end_sample": 90,
                    "text": "word", "confidence": 0.9}],
        )
        result = overlap_vocal_safety(
            self._candidate(), _silence_document("a"), worded, RATE, OVERLAP,
        )
        self.assertEqual(result["status"], "rejected")
        self.assertTrue(any("overlaps_known_word" in reason
                            for reason in result["reasons"]))

    def test_unknown_acoustic_margin_is_unknown(self):
        margin = _silence_document("a", status="unknown_acoustic_margin")
        result = overlap_vocal_safety(
            self._candidate(), margin, _silence_document("b"), RATE, OVERLAP,
        )
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["outgoing_extension"]["status"], "unknown")

    def test_separated_vocal_track_evidence(self):
        silence = _track("a", np.zeros((200, 2), dtype=np.float32))
        silence_b = _track("b", np.zeros((200, 2), dtype=np.float32))
        tone = _track("b", 0.2 * np.ones((200, 2), dtype=np.float32))
        safe = overlap_vocal_safety(
            self._candidate(), silence, silence_b, RATE, OVERLAP,
        )
        self.assertEqual(safe["status"], "supported")
        rejected = overlap_vocal_safety(
            self._candidate(), silence, tone, RATE, OVERLAP,
        )
        self.assertEqual(rejected["status"], "rejected")
        self.assertTrue(any("vocal_occupancy" in reason
                            for reason in rejected["reasons"]))

    def test_identity_mismatch_rejects(self):
        result = overlap_vocal_safety(
            self._candidate(),
            _silence_document("wrong"),
            _silence_document("b"),
            RATE, OVERLAP,
        )
        self.assertEqual(result["status"], "rejected")
        self.assertTrue(any("track_identity_mismatch" in reason
                            for reason in result["reasons"]))

    def test_missing_binding_metadata_is_unknown(self):
        no_rate = _silence_document("a")
        del no_rate["sample_rate"]
        result = overlap_vocal_safety(
            self._candidate(), no_rate, _silence_document("b"), RATE, OVERLAP,
        )
        self.assertEqual(result["status"], "unknown")
        self.assertIn("vocals_a_sample_rate_missing", result["reasons"])


if __name__ == "__main__":
    unittest.main()
