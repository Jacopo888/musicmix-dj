import numpy as np
import pytest

from musicmix.audio import AudioTrack
from musicmix.directional import _Event, _context_for, _cue_events


def test_provided_points_keep_their_own_coordinates_and_null_confidence():
    payload = {"beats": [{"sample": 100}, {"sample": 200}],
               "provided_cues": [{"sample": 123, "confidence": None}]}
    cues, source, low = _cue_events(payload, 1000, cue_policy="provided")
    assert [(cue.sample, cue.confidence, cue.source) for cue in cues] == [(123, None, "provided_cue")]
    assert source == "provided_cues" and low == ()
    empty, _, _ = _cue_events(dict(payload, provided_cues=[]), 1000, cue_policy="provided")
    assert empty == ()  # Empty inference must not turn into the beat fallback.
    with pytest.raises(ValueError, match="explicit"):
        _cue_events({"beats": payload["beats"]}, 1000, cue_policy="provided")
    with pytest.raises(ValueError, match="independent"):
        _cue_events({"provided_cues": payload["provided_cues"]}, 1000, cue_policy="provided")


def test_provided_context_meets_actual_cue_on_both_sides():
    rate = 1000
    track = AudioTrack("canonical", "fixture", rate, np.ones((20000, 2)) * .1)
    beats = tuple(range(0, 20000, 500))
    cue = _Event(10123, None, "provided_cue")
    outgoing = _context_for(track, beats, cue, 4, "outgoing", cue_source=cue.source)
    incoming = _context_for(track, beats, cue, 4, "incoming", cue_source=cue.source)
    assert outgoing.sample == incoming.sample == 10123
    assert outgoing.end_sample == incoming.start_sample == 10123
    assert outgoing.end_sample - outgoing.start_sample == 2000
    assert incoming.end_sample - incoming.start_sample == 2000
    assert (outgoing.tempo_bpm, incoming.tempo_bpm) == pytest.approx((120, 120))
    assert outgoing.beat_offset_samples == incoming.beat_offset_samples == 123


def test_model_points_must_preserve_relative_beat_phase(monkeypatch):
    import musicmix.directional as module
    track = AudioTrack("canonical", "fixture", 1000, np.ones((20000, 2)) * .1)
    track_b = AudioTrack("canonical_b", "fixture_b", 1000, track.samples)
    beats = tuple(range(0, 20000, 500))
    def context(sample, direction):
        return _context_for(track, beats, _Event(sample, None, "provided_cue"), 4,
                            direction, cue_source="provided_cue")
    monkeypatch.setattr(module, "_vocal_alignment", lambda sample, *args: {
        "safe": True, "shifted_sample": sample, "status": "safe"})
    a = context(10123, "outgoing")
    # No vocal adjustment: the original model points themselves differ in phase.
    proposal, audit = module._proposal_for(track, track_b, a, context(10000, "incoming"),
                                           None, None, .04, .08, True)
    assert proposal is None
    assert audit["reason"] == "provided_cue_phase_mismatch"
    assert audit["alignment"]["relative_beat_phase_seconds"] == pytest.approx(.123)
    proposal, audit = module._proposal_for(track, track_b, a, context(10123, "incoming"),
                                           None, None, .04, .08, True)
    assert proposal is not None
    assert proposal.alignment["relative_beat_phase_seconds"] == 0
