from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import wave
from unittest.mock import patch

import numpy as np

from musicmix.audio import load_audio
from musicmix.pretrained_structure import (
    ALL_IN_ONE_LICENSE,
    ALL_IN_ONE_REPOSITORY,
    AllInOneConfig,
    AllInOneProvider,
)
from musicmix.structure import analyze_structure


def _fixture_track(root: Path):
    rate = 48_000
    t = np.arange(rate * 2, dtype=np.float64) / rate
    pcm = np.round(0.2 * np.sin(2 * np.pi * 220 * t) * 32_767).astype("<i2")
    path = root / "fixture.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm.tobytes())
    return load_audio(path)


class PretrainedStructureTests(unittest.TestCase):
    def test_official_result_maps_events_and_functional_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            track = _fixture_track(root)

            def fake_analyze(path, **kwargs):
                self.assertEqual(path, track.path)
                self.assertEqual(kwargs["model"], "harmonix-fold0")
                return SimpleNamespace(
                    bpm=100,
                    beats=[0.25, 0.75],
                    downbeats=[0.25],
                    segments=[
                        SimpleNamespace(start=0.0, end=0.25, label="start"),
                        SimpleNamespace(start=0.25, end=1.5, label="chorus"),
                        SimpleNamespace(start=1.5, end=2.0, label="end"),
                    ],
                )

            provider = AllInOneProvider(AllInOneConfig(), analyze_fn=fake_analyze)
            analysis = analyze_structure(track, provider, include_heuristic_sections=False)

            self.assertEqual(analysis.status, "ok")
            self.assertEqual([event.sample for event in analysis.beats], [12_000, 36_000])
            self.assertEqual([event.sample for event in analysis.downbeats], [12_000])
            self.assertEqual([section.label for section in analysis.sections], ["start", "chorus", "end"])
            self.assertTrue(all(section.confidence is None for section in analysis.sections))
            self.assertEqual(analysis.provenance["provider_repository"], ALL_IN_ONE_REPOSITORY)
            self.assertEqual(analysis.provenance["provider_license"], ALL_IN_ONE_LICENSE)
            self.assertEqual(analysis.provenance["bpm"], 100)
            self.assertFalse(analysis.diagnostics["sections_are_heuristic"])

    def test_runtime_failure_is_explicit_and_has_no_heuristic_sections(self):
        with tempfile.TemporaryDirectory() as folder:
            track = _fixture_track(Path(folder))

            def failing_analyze(path, **kwargs):
                raise RuntimeError("NATTEN is unavailable on this Windows runtime")

            provider = AllInOneProvider(AllInOneConfig(), analyze_fn=failing_analyze)
            analysis = analyze_structure(track, provider, include_heuristic_sections=False)

            self.assertEqual(analysis.status, "unavailable")
            self.assertEqual(analysis.beats, ())
            self.assertEqual(analysis.downbeats, ())
            self.assertEqual(analysis.sections, ())
            self.assertTrue(any("NATTEN" in warning for warning in analysis.warnings))
            self.assertFalse(analysis.diagnostics["sections_are_heuristic"])

    def test_unknown_model_is_rejected_before_runtime_import(self):
        with self.assertRaises(ValueError):
            AllInOneConfig(model="made-up-model")

    def test_demucs_download_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            track = _fixture_track(root)
            provider = AllInOneProvider(AllInOneConfig(demix_dir=root / "demix"))
            with patch.object(provider, "_load_analyze", return_value=lambda *_args, **_kwargs: None):
                with self.assertRaisesRegex(RuntimeError, "stems are missing"):
                    provider.analyze(track)

    def test_staged_input_rejects_corruption_and_repairs_from_canonical_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            track = _fixture_track(root)
            provider = AllInOneProvider(
                AllInOneConfig(demix_dir=root / "demix"),
                analyze_fn=lambda *_args, **_kwargs: None,
            )
            staged = provider._stage_input(track)
            staged.unlink()
            staged.write_bytes(b"corrupt staged input")
            repaired = provider._stage_input(track)
            self.assertEqual(repaired, staged)
            self.assertEqual(repaired.read_bytes(), Path(track.path).read_bytes())


if __name__ == "__main__":
    unittest.main()
