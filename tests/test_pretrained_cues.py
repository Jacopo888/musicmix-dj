from __future__ import annotations

import importlib.util
import wave
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_cue_detr", ROOT / "tools/run_cue_detr.py")
assert SPEC is not None and SPEC.loader is not None
CUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CUE)


def test_pinned_upstream_constants_and_license_are_present() -> None:
    if not CUE.UPSTREAM_SCRIPT.is_file() or not CUE.UPSTREAM_LICENSE.is_file():
        pytest.skip("Optional upstream checkout not installed")
    constants = CUE._load_upstream_constants()
    assert constants == {"overlap": 0.75, "w_bbox": 21, "w_win": 355, "padding": 266}
    assert CUE.UPSTREAM_LICENSE.read_text(encoding="utf-8").startswith("MIT License")
    assert CUE._sha256(CUE.UPSTREAM_SCRIPT) == "da3149142d6e311cc49b2a2fd3b302e0a44ec10330a18183e951831341aca422"


def test_minmax_preserves_raw_values_and_handles_constant_scores() -> None:
    scaled, stats = CUE._normalise_scores([0.2, 0.5, 0.8])
    assert scaled.tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert stats["min"] == 0.2
    assert stats["max"] == 0.8
    assert stats["degenerate"] is False

    constant, constant_stats = CUE._normalise_scores([0.4, 0.4])
    assert constant.tolist() == [0.0, 0.0]
    assert constant_stats["degenerate"] is True


def test_duplicate_peak_frames_choose_highest_raw_score() -> None:
    peaks = [
        {"spectrogram_frame": 10, "raw_score": 0.20, "normalized_score": 0.1},
        {"spectrogram_frame": 10, "raw_score": 0.90, "normalized_score": 0.8},
        {"spectrogram_frame": 20, "raw_score": 0.70, "normalized_score": 0.6},
    ]
    result = CUE._deduplicate_peaks(peaks)
    assert [item["spectrogram_frame"] for item in result] == [10, 20]
    assert result[0]["raw_score"] == 0.90


def test_canonical_metadata_keeps_its_own_clock(tmp_path) -> None:
    path = tmp_path / "canonical.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes(b"\0" * 4 * 960)
    assert CUE._canonical_audio_info(path) == (48000, 960, .02)


def test_revision_does_not_include_metadata_extension(tmp_path) -> None:
    trees = tmp_path / ".cache/huggingface/trees"
    trees.mkdir(parents=True)
    revision = "a" * 40
    (trees / f"{revision}.json.json").write_text("{}")
    assert CUE._hf_revision(tmp_path) == revision
