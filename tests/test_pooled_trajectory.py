import numpy as np
import pytest
from musicmix.audio import AudioTrack
from musicmix.pooled_trajectory import PooledTrajectory
from musicmix.trajectory import extract_trajectory,trajectory_to_dict
from musicmix.expectation import analyze_expectation


RATE = 4000


def _fixture_track(*, postcue=None, precue=None):
    t = np.arange(RATE * 22) / RATE
    signal = (.1 + .2 * t / t[-1]) * np.sin(2 * np.pi * 220 * t)
    signal = signal.astype(np.float64)
    if postcue is not None:
        signal[int(postcue):] += 0.11 * np.sin(2 * np.pi * 710 * t[int(postcue):])
    if precue is not None:
        signal[:int(precue)] += 0.11 * np.sin(2 * np.pi * 530 * t[:int(precue)])
    return AudioTrack("fixture", "fixture", RATE, np.column_stack([signal, signal]))


def _beats():
    return tuple(i * RATE // 4 for i in range(88))


def test_pooled_beat_series_matches_original_extractor_and_continuation_score():
    rate=4000;t=np.arange(rate*22)/rate
    signal=(.1+.2*t/t[-1])*np.sin(2*np.pi*220*t)
    track=AudioTrack('fixture','fixture',rate,np.column_stack([signal,signal]))
    beats=[{'sample':i*rate//4} for i in range(88)]
    pool=PooledTrajectory(track,beats)
    profiles=[];originals=[]
    for cue,side in ((rate*10,'outgoing'),(rate*10,'incoming'),(rate*11,'incoming')):
        cached=pool.profile(cue,side=side)
        original=trajectory_to_dict(extract_trajectory(track,cue,beats,side=side))
        assert cached['beat_period_samples'] == original['beat_period_samples']
        assert [w['beat_feature_series'] for w in cached['windows']]==[w['beat_feature_series'] for w in original['windows']]
        profiles.append(cached);originals.append(original)
    fast=analyze_expectation(*profiles);reference=analyze_expectation(*originals)
    assert fast.label==reference.label
    assert fast.score==reference.score


def test_missing_real_context_remains_missing():
    track=AudioTrack('fixture','fixture',4000,np.zeros((4000*3,2),dtype=np.float32))
    pool=PooledTrajectory(track,[0,4000,8000,12000])
    assert pool.profile(0,side='outgoing')['windows']==[]
    with pytest.raises(ValueError):pool.profile(-1,side='incoming')


def test_actual_cue_clips_edges_and_postcue_audio_cannot_leak_into_outgoing():
    beats = _beats()
    cue = RATE * 10 + 500
    clean = PooledTrajectory(_fixture_track(), beats).profile(cue, side="outgoing")
    changed = PooledTrajectory(_fixture_track(postcue=cue), beats).profile(cue, side="outgoing")

    assert [w["beat_feature_series"] for w in clean["windows"]] == [
        w["beat_feature_series"] for w in changed["windows"]
    ]
    assert all(w["end_sample"] == cue for w in clean["windows"])
    assert all(w["end_sample"] <= cue for w in clean["windows"])
    assert all(w["beat_feature_series"][-1]["end_sample"] == cue for w in clean["windows"])
    assert all(w["boundary_method"] == "exact_cue_analysis_boundary" for w in clean["windows"])


def test_actual_cue_clips_edges_and_precue_audio_cannot_leak_into_incoming():
    beats = _beats()
    cue = RATE * 10 + 500
    clean = PooledTrajectory(_fixture_track(), beats).profile(cue, side="incoming")
    changed = PooledTrajectory(_fixture_track(precue=cue), beats).profile(cue, side="incoming")

    assert [w["beat_feature_series"] for w in clean["windows"]] == [
        w["beat_feature_series"] for w in changed["windows"]
    ]
    assert all(w["start_sample"] == cue for w in clean["windows"])
    assert all(w["start_sample"] >= cue for w in clean["windows"])
    assert all(w["beat_feature_series"][0]["start_sample"] == cue for w in clean["windows"])
    assert all(w["boundary_method"] == "exact_cue_analysis_boundary" for w in clean["windows"])


def test_shifted_cue_is_reported_on_real_clock_without_mutating_source_grid():
    beats = _beats()
    original = tuple(beats)
    pool = PooledTrajectory(_fixture_track(), beats)
    result = pool.profile(RATE * 10 - 1000, side="outgoing", cue_shift_samples=1500)

    assert tuple(pool.beats) == original
    assert result["source_beatgrid"] == list(original)
    assert 40_500 not in pool.beats
    assert result["requested_cue_sample"] == RATE * 10 - 1000
    assert result["actual_cue_sample"] == 40_500
    assert result["cue_sample"] == 40_500
    assert result["nearest_beat_delta_samples"] == 500
    assert result["boundary_method"] == "nearest_source_beat_with_exact_cue_analysis_boundary"
    assert result["trajectory_config_hash"] == result["diagnostics"]["trajectory_config_hash"]
    assert result["diagnostics"]["window_bounds"]
    assert all(window["end_sample"] <= 40_500 for window in result["windows"])

    with pytest.raises(ValueError):
        pool.profile(0, side="incoming", cue_shift_samples=-1)
    with pytest.raises(ValueError):
        pool.profile(pool.duration, side="incoming", cue_shift_samples=1)
