"""Independent render-audit fixtures for psychology-v11."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from tools.audit_psychology_v11 import audit_manifest


RATE = 48_000
SOURCE_FRAMES = RATE * 24
OUTPUT_FRAMES = RATE * 48


def _write_pcm(path: Path, samples: np.ndarray) -> None:
    samples = np.asarray(samples, dtype=np.float64)
    pcm = np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(pcm.tobytes())


def _source_values(offset: float) -> np.ndarray:
    # A non-periodic, low-level pair of ramps makes both timestamp and gain
    # errors visible in each independent audit window.
    t = np.arange(SOURCE_FRAMES, dtype=np.float64)
    left = offset + 0.15 * (t / SOURCE_FRAMES) + 0.01 * np.sin(t / 177.0)
    right = offset / 2.0 + 0.11 * (t / SOURCE_FRAMES) + 0.008 * np.cos(t / 241.0)
    return np.column_stack((left, right))


def _clean_fixture(tmp_path: Path, *, action: str = "prepare") -> tuple[Path, dict]:
    source_a = tmp_path / "a.wav"
    source_b = tmp_path / "b.wav"
    values_a = _source_values(0.11)
    values_b = _source_values(-0.08)
    _write_pcm(source_a, values_a)
    _write_pcm(source_b, values_b)

    # Render exactly the two 24-second contexts at the known 0.9 gain.  The
    # source WAV is decoded as int16/32768 by the production loader.
    with wave.open(str(source_a), "rb") as stream:
        a = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").reshape(-1, 2)
    with wave.open(str(source_b), "rb") as stream:
        b = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").reshape(-1, 2)
    expected_a = np.rint((a.astype(np.float64) / 32768.0) * 0.9 * 32767.0).astype("<i2")
    expected_b = np.rint((b.astype(np.float64) / 32768.0) * 0.9 * 32767.0).astype("<i2")
    fade = round(0.012 * RATE)
    expected_a[-fade:] = np.rint(expected_a[-fade:].astype(np.float64)
                                  * np.linspace(1.0, 0.0, fade)[:, None]).astype("<i2")
    expected_b[:fade] = np.rint(expected_b[:fade].astype(np.float64)
                                 * np.linspace(0.0, 1.0, fade)[:, None]).astype("<i2")
    output = tmp_path / "render.wav"
    with wave.open(str(output), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(np.concatenate((expected_a, expected_b), axis=0).tobytes())

    history_path = tmp_path / "history.json"
    history_path.write_text(json.dumps({"regions": [{
        "track_a": "other-a", "track_b": "other-b",
        "cue_out_sample": 0, "cue_in_sample": 0, "sample_rate": RATE,
    }]}), encoding="utf-8")

    manifest = {
        "tracks": [
            {"track_id": "a", "canonical_path": str(source_a)},
            {"track_id": "b", "canonical_path": str(source_b)},
        ],
        "inputs": {"history_snapshot": {"path": str(history_path)}},
        "artifacts": [{
            "variant": "main",
            "candidate": {"track_a": "a", "track_b": "b",
                          "cue_out_sample": SOURCE_FRAMES, "cue_in_sample": 0},
            "artifact": {"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()},
            "assessment": {"action": action, "coverage": {"fraction": 0.75}},
            "phrase_a": {"boundary_class": "verified_phrase_closure"},
            "phrase_b": {"boundary_class": "verified_phrase_closure"},
            "validation": {"level_matching": {"gains_db": {"a": 0.0, "b": 0.0,
                                                               "master_trim_db": 0.0}}},
        }],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest


def test_clean_manifest_checks_independent_pcm_and_policy(tmp_path: Path) -> None:
    manifest_path, _ = _clean_fixture(tmp_path)
    result = audit_manifest(manifest_path)
    assert result["status"] == "pass"
    assert result["policy"]["action_counts"] == {"prepare": 1}
    assert result["policy"]["coverage_mean"] == pytest.approx(0.75)
    checks = result["artifacts"][0]["source_pcm"]["windows"]
    assert len(checks) == 8  # six interior windows plus two 12 ms ramps
    assert all(window["status"] == "pass" for window in checks)
    assert (tmp_path / "qa_independent_v2.json").exists()


def test_wrong_timestamp_or_gain_is_caught_by_interior_windows(tmp_path: Path) -> None:
    manifest_path, manifest = _clean_fixture(tmp_path)
    output = Path(manifest["artifacts"][0]["artifact"]["path"])
    with wave.open(str(output), "rb") as stream:
        pcm = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").reshape(-1, 2).copy()
    # A one-sample source shift and a gain error are both away from the join.
    pcm[RATE * 8:RATE * 8 + 4096] = np.roll(pcm[RATE * 8:RATE * 8 + 4096], 1, axis=0)
    pcm[RATE * 32:RATE * 32 + 4096] = np.rint(pcm[RATE * 32:RATE * 32 + 4096] * 1.03).astype("<i2")
    with wave.open(str(output), "wb") as stream:
        stream.setnchannels(2); stream.setsampwidth(2); stream.setframerate(RATE)
        stream.writeframes(pcm.tobytes())
    # Update the declared digest so this test exercises the independent PCM
    # checks rather than stopping at provenance.
    manifest["artifacts"][0]["artifact"]["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = audit_manifest(manifest_path)
    assert result["status"] == "fail"
    assert any("source_pcm_mismatch" in issue for issue in result["issues"])


def test_natural_a_is_reported_separately_from_main_artifacts(tmp_path: Path) -> None:
    manifest_path, manifest = _clean_fixture(tmp_path)
    source_path = Path(manifest["tracks"][0]["canonical_path"])
    with wave.open(str(source_path), "rb") as stream:
        source = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").reshape(-1, 2)
    fade = round(0.01 * RATE)
    natural_pcm = np.rint(source.astype(np.float64) / 32768.0 * 0.9 * 32767.0).astype("<i2")
    natural_pcm[:fade] = np.rint(natural_pcm[:fade].astype(np.float64)
                                  * np.linspace(0.0, 1.0, fade)[:, None]).astype("<i2")
    natural_pcm[-fade:] = np.rint(natural_pcm[-fade:].astype(np.float64)
                                  * np.linspace(1.0, 0.0, fade)[:, None]).astype("<i2")
    natural_path = tmp_path / "natural_a.wav"
    with wave.open(str(natural_path), "wb") as stream:
        stream.setnchannels(2); stream.setsampwidth(2); stream.setframerate(RATE)
        stream.writeframes(natural_pcm.tobytes())
    manifest["natural_a"] = {
        "path": str(natural_path), "sha256": hashlib.sha256(natural_path.read_bytes()).hexdigest(),
        "source": {"track_id": "a", "start_sample": 0,
                    "end_sample": SOURCE_FRAMES, "gain": 0.9},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = audit_manifest(manifest_path)
    assert result["status"] == "pass"
    assert len(result["artifacts"]) == 1
    assert [item["variant"] for item in result["natural"]] == ["natural_a"]


def test_policy_rejects_execute_with_active_word_and_certified_missing_closure(tmp_path: Path) -> None:
    manifest_path, manifest = _clean_fixture(tmp_path, action="execute")
    entry = manifest["artifacts"][0]
    entry["phrase_a"] = {"active_word": True, "boundary_class": "active_utterance"}
    entry["phrase_b"] = {"true_phrase_closure": "unknown"}
    entry["assessment"]["closure_certified"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = audit_manifest(manifest_path)
    assert result["status"] == "fail"
    assert "policy:execute_with_unsafe_phrase:0" in result["issues"]
    assert "policy:missing_closure_claimed_certified:0" in result["issues"]


def test_policy_does_not_treat_asr_segment_or_word_gap_as_unsafe(tmp_path: Path) -> None:
    manifest_path, manifest = _clean_fixture(tmp_path, action="execute")
    entry = manifest["artifacts"][0]
    entry["phrase_a"] = {"active_asr_segment": True, "boundary_class": "active_asr_segment"}
    entry["phrase_b"] = {"word_gap": True, "boundary_class": "word_gap_only"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = audit_manifest(manifest_path)
    assert not any("execute_with_unsafe_phrase" in issue for issue in result["policy"]["issues"])


def test_existing_qa_report_is_never_overwritten(tmp_path: Path) -> None:
    manifest_path, _ = _clean_fixture(tmp_path)
    output = tmp_path / "qa.json"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        audit_manifest(manifest_path, output)
    assert output.read_text(encoding="utf-8") == "keep"


def test_v10_history_repeated_eight_second_region_is_rejected(tmp_path: Path) -> None:
    manifest_path, manifest = _clean_fixture(tmp_path)
    output = Path(manifest["artifacts"][0]["artifact"]["path"])
    manifest["v10_history"] = [{"path": str(output)}]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = audit_manifest(manifest_path)
    assert result["status"] == "fail"
    assert result["history"]["status"] == "fail"
    assert "v10_history_repeated_whole_transition" in result["issues"]


def test_history_snapshot_exposure_catches_reuse_even_with_different_gain(tmp_path: Path) -> None:
    manifest_path, manifest = _clean_fixture(tmp_path)
    history_path = Path(manifest["inputs"]["history_snapshot"]["path"])
    history_path.write_text(json.dumps({"regions": [{
        "track_a": "a", "track_b": "unrelated",
        "cue_out_sample": SOURCE_FRAMES + 7 * RATE, "cue_in_sample": 0,
        "sample_rate": RATE,
    }]}), encoding="utf-8")
    output = Path(manifest["artifacts"][0]["artifact"]["path"])
    with wave.open(str(output), "rb") as stream:
        pcm = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").reshape(-1, 2).copy()
    pcm[RATE * 8:RATE * 16] = np.rint(pcm[RATE * 8:RATE * 16].astype(np.float64) * 0.8).astype("<i2")
    with wave.open(str(output), "wb") as stream:
        stream.setnchannels(2); stream.setsampwidth(2); stream.setframerate(RATE)
        stream.writeframes(pcm.tobytes())
    manifest["artifacts"][0]["artifact"]["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = audit_manifest(manifest_path)
    assert result["history"]["status"] == "fail"
    assert result["history"]["exposure"]["counts"]["outgoing_region_heard"] == 1
    assert result["history"]["exposure"]["counts"]["incoming_region_heard"] == 0
    assert "v10_history_coordinate_exposure" in result["issues"]


def test_context_overlap_is_descriptive_when_cue_exposure_is_new(tmp_path: Path) -> None:
    manifest_path, manifest = _clean_fixture(tmp_path)
    output = Path(manifest["artifacts"][0]["artifact"]["path"])
    historical = tmp_path / "historical_preview.wav"
    historical.write_bytes(output.read_bytes())
    with wave.open(str(historical), "rb") as stream:
        pcm = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").reshape(-1, 2).copy()
    pcm[0, 0] = 0 if pcm[0, 0] else 1
    with wave.open(str(historical), "wb") as stream:
        stream.setnchannels(2); stream.setsampwidth(2); stream.setframerate(RATE)
        stream.writeframes(pcm.tobytes())
    history_path = Path(manifest["inputs"]["history_snapshot"]["path"])
    history_path.write_text(json.dumps({"regions": [{
        "track_a": "old-a", "track_b": "old-b", "cue_out_sample": 0,
        "cue_in_sample": 0, "sample_rate": RATE, "source": str(historical),
    }]}), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = audit_manifest(manifest_path)
    assert result["history"]["status"] == "pass"
    assert result["history"]["context_overlaps"]
    assert result["eligibility"]["eligible_for_review"] is True
    assert result["eligibility"]["context_reuse_explicit"] is True


def test_empty_artifact_list_is_reported_as_no_outputs(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"tracks": [], "artifacts": []}), encoding="utf-8")
    result = audit_manifest(path)
    assert result["status"] == "fail"
    assert result["outputs"] == {"count": 0, "status": "no_outputs"}
    assert "no_outputs" in result["issues"]


def test_gain_bounds_and_pcm_boundary_clipping_are_reported(tmp_path: Path) -> None:
    manifest_path, manifest = _clean_fixture(tmp_path)
    entry = manifest["artifacts"][0]
    entry["validation"]["level_matching"]["gains_db"]["a"] = 4.0
    output = Path(entry["artifact"]["path"])
    with wave.open(str(output), "rb") as stream:
        pcm = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").reshape(-1, 2).copy()
    pcm[RATE * 8, 0] = 32767
    with wave.open(str(output), "wb") as stream:
        stream.setnchannels(2); stream.setsampwidth(2); stream.setframerate(RATE)
        stream.writeframes(pcm.tobytes())
    entry["artifact"]["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = audit_manifest(manifest_path)
    assert result["status"] == "fail"
    assert "artifact[0]:a_gain_out_of_bounds" in result["issues"]
    assert "artifact[0]:actual_pcm_clipping" in result["issues"]

