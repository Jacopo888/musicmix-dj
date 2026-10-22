from pathlib import Path
import hashlib
import tempfile
import wave

import numpy as np
import pytest

from musicmix.audio import AudioTrack
from musicmix.contracts import Candidate
from musicmix.intent_render import (
    BlendNotEligible,
    IntentRenderError,
    MIN_CONTEXT_SECONDS,
    assess_overlap_eligibility,
    render_intent,
)


RATE = 1000
CONTEXT = 16.0
FRAMES = 40_000
OUT = IN = 17_000


def track(name, value=0.0, *, frames=FRAMES, rate=RATE, channels=2):
    samples = np.full((frames, channels), value, dtype=np.float32)
    return AudioTrack(name, f"{name}.wav", rate, samples)


def candidate(a, b, out=OUT, incoming=IN):
    return Candidate("fixture", a.track_id, b.track_id, out, incoming, "handoff", {}, {})


def pcm(path):
    with wave.open(str(path), "rb") as stream:
        values = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
        return values.reshape(-1, stream.getnchannels()), stream.getframerate()


def silence_doc(track_id):
    return {"track_id": track_id, "sample_rate": RATE, "duration_samples": FRAMES,
            "coverage_intervals": [{"start_sample": 0, "end_sample": FRAMES,
                                     "status": "acoustic_silence_supported"}], "words": []}


def test_handoff_has_source_and_output_clocks_and_actual_pcm_anchor():
    a, b = track("a"), track("b")
    a_values = np.zeros_like(a.samples)
    b_values = np.zeros_like(b.samples)
    a_values[OUT - 20] = 0.5
    b_values[IN + 20] = 0.5
    a = AudioTrack("a", "a.wav", RATE, a_values)
    b = AudioTrack("b", "b.wav", RATE, b_values)
    with tempfile.TemporaryDirectory() as folder:
        destination = Path(folder) / "handoff.wav"
        result = render_intent(a, b, candidate(a, b), destination, level_match=False)
        samples, rate = pcm(destination)
        assert rate == RATE
        assert result["minimum_context_seconds"] == MIN_CONTEXT_SECONDS
        assert result["clocks"]["output"]["join_sample"] == RATE * 10
        assert result["clocks"]["source"]["a"]["cue_sample"] == OUT
        assert len(samples) == RATE * 20
        assert samples[10_000 - 20, 0] == round(0.9 * 0.5 * 32767)
        assert samples[10_000 + 20, 0] == round(0.9 * 0.5 * 32767)
        assert result["sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()


def test_context_identity_rate_and_exclusive_output_are_guarded():
    a, b = track("a"), track("b")
    with tempfile.TemporaryDirectory() as folder:
        destination = Path(folder) / "out.wav"
        destination.write_bytes(b"keep")
        with pytest.raises(FileExistsError):
            render_intent(a, b, candidate(a, b), destination)
        assert destination.read_bytes() == b"keep"
        with pytest.raises(IntentRenderError, match=">= 10"):
            render_intent(a, b, candidate(a, b), Path(folder) / "short.wav", context_seconds=4)
        bad = Candidate("fixture", "other", b.track_id, OUT, IN, "handoff", {}, {})
        with pytest.raises(IntentRenderError, match="identity"):
            render_intent(a, b, bad, Path(folder) / "identity.wav")
    b_other = track("b", rate=2000)
    with tempfile.TemporaryDirectory() as folder:
        with pytest.raises(IntentRenderError, match="sample rates"):
            render_intent(a, b_other, candidate(a, b_other), Path(folder) / "rate.wav")


def test_near_silence_is_not_boosted_and_level_match_is_bounded():
    a, b = track("a", 0.1), track("b", 1e-5)
    with tempfile.TemporaryDirectory() as folder:
        result = render_intent(a, b, candidate(a, b), Path(folder) / "near.wav")
        assert result["level_matching"]["status"] == "skipped_near_silence"
        assert result["level_matching"]["gains_db"] == {"a": 0.0, "b": 0.0, "master_trim_db": 0.0}
        assert any("digital-silence" in warning for warning in result["warnings"])
    a, b = track("a", 0.5), track("b", 0.02)
    with tempfile.TemporaryDirectory() as folder:
        result = render_intent(a, b, candidate(a, b), Path(folder) / "gain.wav")
        gains = result["level_matching"]["gains_db"]
        assert gains["a"] <= 0
        assert gains["b"] <= 3


def test_peak_headroom_is_enforced_using_actual_rendered_peak():
    a, b = track("a", 1.5), track("b", 1.5)
    with tempfile.TemporaryDirectory() as folder:
        destination = Path(folder) / "peak.wav"
        result = render_intent(a, b, candidate(a, b), destination, level_match=False)
        data, _ = pcm(destination)
        assert result["headroom"]["applied_master_trim"]
        assert result["headroom"]["final_peak"] <= 0.98 + 1e-8
        assert result["clipped_samples"] == 0
        assert int(np.max(np.abs(data))) < 32767


def test_overlap_fails_closed_for_unknown_and_occupied_evidence():
    a, b = track("a"), track("b")
    report = assess_overlap_eligibility(candidate(a, b), None, None, RATE, 1.0, tempo_a=120, tempo_b=120)
    assert not report["eligible"]
    with tempfile.TemporaryDirectory() as folder:
        with pytest.raises(BlendNotEligible):
            render_intent(a, b, candidate(a, b), Path(folder) / "unknown.wav", intent="overlap",
                          tempo_a=120, tempo_b=120)
        occupied = {**silence_doc("a"), "coverage_intervals": [{"start_sample": 0, "end_sample": FRAMES,
                         "status": "vocal_occupancy_supported"}]}
        with pytest.raises(BlendNotEligible):
            render_intent(a, b, candidate(a, b), Path(folder) / "occupied.wav", intent="overlap",
                          vocals_a=occupied, vocals_b=occupied, tempo_a=120, tempo_b=120)
