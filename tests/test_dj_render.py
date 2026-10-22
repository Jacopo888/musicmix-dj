"""Synthetic checks for the DJ-standard renderer (dj_render).

Click tracks at 120 and 126 BPM with marked downbeats (3 kHz, loud) and
offbeats (1.5 kHz, soft), plus a sustained sub-bass sine per track
(A 55 Hz, B 82 Hz) so that the bass swap and pitch preservation can be
measured on the rendered PCM.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

pytest.importorskip("pyloudnorm")
pytest.importorskip("soundfile")
from scipy.signal import butter, resample_poly, sosfiltfilt  # noqa: E402

from musicmix import dj_render as dj  # noqa: E402

SR = 48000
HAS_STRETCH = bool(shutil.which("rubberband") or shutil.which("ffmpeg"))
needs_stretch = pytest.mark.skipif(not HAS_STRETCH, reason="no rubberband/ffmpeg time-stretch backend")


def click_track(bpm, dur, bass_hz, *, start=0.5, bass_level=0.3, gain=1.0, voc_hz=None):
    n = int(dur * SR)
    x = np.zeros(n)
    period = 60.0 / bpm
    burst = np.exp(-np.arange(480) / 60.0)
    beats, t, k = [], start, 0
    while t < dur - 0.2:
        i = int(round(t * SR))
        f, amp = (3000.0, 1.0) if k % 4 == 0 else (1500.0, 0.35)
        x[i : i + 480] += amp * burst * np.sin(2 * np.pi * f * np.arange(480) / SR)
        beats.append(t)
        t += period
        k += 1
    tt = np.arange(n) / SR
    x += bass_level * np.sin(2 * np.pi * bass_hz * tt)
    voc = None
    if voc_hz:
        voc = 0.2 * np.sin(2 * np.pi * voc_hz * tt) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * tt))
        x = x + voc
        voc = voc * gain
    beats = np.array(beats)
    return {"audio": x * gain, "beats": beats, "downbeats": beats[::4], "vocals": voc}


@pytest.fixture(scope="module")
def tracks():
    # A is quiet (about -23 LUFS), B is loud (about -8 LUFS): both must land on -14.
    a = click_track(120, 50, 55, gain=0.3)
    b = click_track(126, 50, 82, gain=1.5)
    return a, b


def _render(a, b, kind="blend", bars=8, ia=8, jb=10, **kw):
    return dj.render_transition(
        a["audio"], b["audio"],
        beats_a=a["beats"], downbeats_a=a["downbeats"],
        beats_b=b["beats"], downbeats_b=b["downbeats"],
        cue_out_a=a["downbeats"][ia], cue_in_b=b["downbeats"][jb],
        kind=kind, bars=bars, sample_rate=SR, **kw,
    )


@pytest.fixture(scope="module")
def blend8(tracks):
    a, b = tracks
    return _render(a, b, bars=8, return_decks=True)


def _band_onsets(x, lo, hi, min_gap_s=0.3, rel=0.5):
    mono = x.mean(axis=1) if x.ndim == 2 else x
    sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    env = np.abs(sosfiltfilt(sos, mono))
    thr = rel * env.max()
    idx = np.flatnonzero(env > thr)
    onsets = []
    for i in idx:
        if not onsets or i - onsets[-1] > min_gap_s * SR:
            onsets.append(i)
    return np.array(onsets)


def _tone_level(x, f, start, length):
    seg = x[start : start + length].mean(axis=1)
    w = np.hanning(len(seg))
    t = np.arange(len(seg)) / SR
    return 2 * abs(np.sum(seg * w * np.exp(-2j * np.pi * f * t))) / w.sum()


# ------------------------------------------------------------------ alignment


@needs_stretch
def test_downbeats_of_stretched_b_land_on_a_grid(blend8, tracks):
    a, _ = tracks
    meta = blend8.metadata
    ov = meta["overlap"]
    off_a = ov["a_source_start_sample"] - ov["output_start_sample"]
    expected = np.round(a["downbeats"][8:17] * SR).astype(int) - off_a
    # Independent measure: 3 kHz downbeat clicks detected in the stretched B deck.
    b_raw = blend8.decks["b_stretched_raw"]
    found = _band_onsets(b_raw, 2500, 3500)
    errors = []
    for e in expected:
        j = np.argmin(np.abs(found - e))
        errors.append((found[j] - e) / SR * 1000.0)
    errors = np.array(errors)
    # Same detector on the unstretched source gives its own bias; remove it.
    src_found = _band_onsets(tracks[1]["audio"][:, None].repeat(2, 1), 2500, 3500)
    src_true = np.round(tracks[1]["downbeats"] * SR).astype(int)
    bias = np.median([(src_found[np.argmin(np.abs(src_found - s))] - s) / SR * 1000.0 for s in src_true[2:12]])
    assert np.max(np.abs(errors - bias)) < 10.0, errors
    assert meta["downbeat_alignment"]["within_10ms"]
    assert meta["downbeat_alignment"]["max_abs_error_ms"] < 10.0
    # Grid-free check: onset cross-correlation between the two decks.
    assert meta["downbeat_alignment"]["interdeck_confident_downbeats"] >= 5
    assert meta["downbeat_alignment"]["interdeck_max_abs_lag_ms"] < 10.0
    st = meta["stretch"]
    assert st["fold_k_b_bars_per_a_bar"] == 1.0
    assert st["tempo_ratio_b"] == pytest.approx(120 / 126, rel=1e-3)
    assert st["b_bpm_effective"] == pytest.approx(120.0, rel=2e-3)


@needs_stretch
def test_stretch_preserves_pitch(blend8):
    st = blend8.metadata["stretch"]
    b_raw = blend8.decks["b_stretched_raw"]
    start = blend8.metadata["overlap"]["output_end_sample"] + SR
    seg = b_raw[start : start + 4 * SR].mean(axis=1)
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1 / SR)
    low = freqs < 300
    assert freqs[low][np.argmax(spec[low])] == pytest.approx(82.0, abs=0.5)
    assert st["applied"] and st["pitch"].startswith("preserved")


@needs_stretch
def test_fit_mode_constant_ratio_also_aligns(tracks):
    a, b = tracks
    r = _render(a, b, bars=4, config=dj.DJRenderConfig(stretch_mode="fit"))
    st = r.metadata["stretch"]
    assert st["mode"].startswith("fit")
    assert st["tempo_ratio_b"] == pytest.approx(120 / 126, rel=1e-3)
    assert r.metadata["downbeat_alignment"]["max_abs_error_ms"] < 10.0


# ---------------------------------------------------------------- bass swap


@needs_stretch
def test_never_two_full_basses(blend8):
    meta = blend8.metadata
    ov, sw = meta["overlap"], meta["switch"]
    out = blend8.audio.astype(np.float64)
    frame = int(0.2 * SR)
    ref_a = _tone_level(out, 55.0, ov["output_start_sample"] - 2 * SR, 2 * SR)
    ref_b = _tone_level(out, 82.0, ov["output_end_sample"] + SR, 2 * SR)
    both_full, a_before, b_after = 0, [], []
    for s in range(ov["output_start_sample"], ov["output_end_sample"] - frame, frame // 2):
        la = 20 * np.log10(_tone_level(out, 55.0, s, frame) / ref_a + 1e-12)
        lb = 20 * np.log10(_tone_level(out, 82.0, s, frame) / ref_b + 1e-12)
        both_full += int(la > -6.0 and lb > -6.0)
        if s + frame < sw["output_sample"]:
            a_before.append(la)
        elif s > sw["output_sample"]:
            b_after.append(lb)
    assert both_full == 0
    # The swap really happens: A's bass holds until the switch, B's after it.
    assert min(a_before) > -3.0
    assert min(b_after) > -3.0
    # Energy below 200 Hz in the overlap never exceeds one full bass by much.
    sos = butter(4, 200, btype="low", fs=SR, output="sos")
    low = sosfiltfilt(sos, out.mean(axis=1))
    ov_low = low[ov["output_start_sample"] : ov["output_end_sample"]]
    ref_low = low[ov["output_start_sample"] - 4 * SR : ov["output_start_sample"]]
    n = len(ov_low) // frame
    rms_ov = np.sqrt(np.mean(ov_low[: n * frame].reshape(n, frame) ** 2, axis=1))
    assert 20 * np.log10(rms_ov.max() / np.sqrt(np.mean(ref_low**2))) < 2.0


@needs_stretch
def test_bass_gains_are_complementary_and_single(blend8):
    g = blend8.decks["gains"]
    assert np.all(g["low_a"] + g["low_b"] <= 1.0 + 1e-9)
    full = (g["low_a"] * g["fader_a"] > 0.5) & (g["low_b"] * g["fader_b"] > 0.5)
    assert not np.any(full)


# ----------------------------------------------------------------- loudness


@needs_stretch
def test_loudness_target_and_true_peak(blend8, tracks):
    import pyloudnorm

    a, b = tracks
    loud = blend8.metadata["loudness"]
    meter = pyloudnorm.Meter(SR)
    for key, track in (("a", a), ("b", b)):
        gain = 10 ** (loud[f"{key}_gain_db"] / 20)
        x = np.repeat(track["audio"][:, None], 2, axis=1) * gain
        assert meter.integrated_loudness(x) == pytest.approx(-14.0, abs=0.1)
        assert loud[f"{key}_integrated_lufs_normalized"] == pytest.approx(-14.0, abs=0.1)
    assert loud["a_integrated_lufs_source"] < -20 and loud["b_integrated_lufs_source"] > -10
    assert abs(loud["loudness_jump_db"]) < 1.5
    up = resample_poly(blend8.audio.astype(np.float64), 4, 1, axis=0)
    assert 20 * np.log10(np.max(np.abs(up))) <= -1.0 + 0.02


def test_limiter_holds_true_peak_on_hot_input():
    rng = np.random.default_rng(0)
    x = np.clip(rng.normal(0, 0.5, (5 * SR, 2)), -1.4, 1.4)
    y, info = dj.true_peak_limit(x, SR, -1.0)
    up = resample_poly(y, 4, 1, axis=0)
    assert 20 * np.log10(np.max(np.abs(up))) <= -1.0 + 0.02
    assert info["max_gain_reduction_db"] > 3.0


def test_band_split_reconstructs_exactly():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.1, (SR, 2))
    low, mid, high = dj.split_bands(x, SR)
    assert np.max(np.abs(low + mid + high - x)) < 1e-12


# ----------------------------------------------------------------- durations


@needs_stretch
def test_blend_clip_is_10s_plus_transition_plus_10s(blend8):
    clip = blend8.metadata["clip"]
    ov = blend8.metadata["overlap"]
    assert ov["output_start_sample"] == 10 * SR
    assert ov["duration_s"] == pytest.approx(16.0, abs=1e-3)  # 8 bars at 120 BPM
    assert clip["length_samples"] == 10 * SR + 16 * SR + 10 * SR
    assert len(blend8.audio) == clip["length_samples"] and clip["matches_rule"]
    sw = blend8.metadata["switch"]["output_sample"]
    assert sw == 10 * SR + 8 * SR  # middle downbeat


def test_cut_clip_is_20s_on_the_downbeat(tracks):
    a, b = tracks
    r = _render(a, b, kind="cut")
    meta = r.metadata
    assert len(r.audio) == 20 * SR and meta["clip"]["matches_rule"]
    assert meta["switch"]["output_sample"] == 10 * SR
    assert meta["switch"]["a_source_sample"] == int(round(a["downbeats"][8] * SR))
    assert meta["switch"]["b_source_sample"] == int(round(b["downbeats"][10] * SR))
    assert meta["overlap"]["duration_s"] == pytest.approx(0.010, abs=1e-4)
    assert meta["stretch"]["tempo_jump_pct"] == pytest.approx(5.0, abs=0.2)
    # B's downbeat click starts right after the switch sample at full gain.
    found = _band_onsets(r.audio.astype(np.float64), 2500, 3500)
    assert np.min(np.abs(found - 10 * SR)) < 0.002 * SR


@needs_stretch
def test_blend_4_bars_kim_convention_ends_on_cue_in(tracks):
    a, b = tracks
    r = _render(a, b, bars=4)
    ov = r.metadata["overlap"]
    assert ov["duration_s"] == pytest.approx(8.0, abs=1e-3)
    assert len(r.audio) == 28 * SR
    assert ov["b_source_end_sample"] == int(round(b["downbeats"][10] * SR))
    assert ov["b_source_start_sample"] == int(round(b["downbeats"][6] * SR))


# ------------------------------------------------------------ tempo folding


def test_tempo_fold_choices():
    k, rate = dj.choose_tempo_fold(2.0, 3.9)  # B in half time
    assert k == 0.5 and rate == pytest.approx(0.975)
    k, rate = dj.choose_tempo_fold(2.0, 1.02)  # B in double time
    assert k == 2.0 and rate == pytest.approx(1.02)
    with pytest.raises(dj.TempoIncompatibleError):
        dj.choose_tempo_fold(2.0, 2.4)


@needs_stretch
def test_half_time_b_maps_two_a_bars_per_b_bar():
    a = click_track(120, 40, 55)
    b = click_track(63, 60, 82)  # bar 3.81 s ~ 2 A bars
    r = _render(a, b, bars=4, ia=6, jb=6)
    st = r.metadata["stretch"]
    assert st["fold_k_b_bars_per_a_bar"] == 0.5
    assert st["tempo_ratio_b"] == pytest.approx(0.5 * (4 * 60 / 63) / 2.0, rel=1e-3)  # 63 -> 60 BPM (120 half time)
    assert r.metadata["overlap"]["b_downbeat_indices"] == [4, 6]
    assert r.metadata["downbeat_alignment"]["max_abs_error_ms"] < 10.0


@needs_stretch
def test_blend_rejected_beyond_8_percent_but_cut_allowed():
    a = click_track(120, 40, 55)
    b = click_track(140, 40, 82)
    with pytest.raises(dj.TempoIncompatibleError):
        _render(a, b, bars=4, ia=6, jb=10)
    r = _render(a, b, kind="cut", ia=6, jb=10)
    assert r.metadata["stretch"]["tempo_jump_pct"] == pytest.approx(100 * (140 / 120 - 1), abs=0.3)


def test_cue_off_downbeat_is_rejected(tracks):
    a, b = tracks
    with pytest.raises(dj.DJRenderError):
        dj.render_transition(
            a["audio"], b["audio"], beats_a=a["beats"], downbeats_a=a["downbeats"],
            beats_b=b["beats"], downbeats_b=b["downbeats"],
            cue_out_a=a["downbeats"][8] + 0.25, cue_in_b=b["downbeats"][10], kind="cut",
        )


# -------------------------------------------------------------------- vocals


@needs_stretch
def test_vocal_ducking_removes_double_voice():
    a = click_track(120, 50, 55, voc_hz=700)
    b = click_track(126, 50, 82, voc_hz=1100)
    kw = dict(bars=8)
    plain = _render(a, b, **kw)
    ducked = _render(a, b, vocals_a=a["vocals"], vocals_b=b["vocals"], **kw)
    voc = ducked.metadata["vocals"]
    assert voc["status"] == "measured"
    assert voc["clash_seconds_before_ducking"] > 2.0
    assert voc["ducking_applied"] and voc["clash_seconds_after_ducking"] == 0.0
    sw = ducked.metadata["switch"]["output_sample"]
    win = (sw + SR, 2 * SR)  # after the switch: A fader still audible, B fully in
    l_plain = _tone_level(plain.audio.astype(np.float64), 700.0, *win)
    l_duck = _tone_level(ducked.audio.astype(np.float64), 700.0, *win)
    assert 20 * np.log10(l_duck / l_plain) < -15.0
    # B's voice is untouched.
    b_plain = _tone_level(plain.audio.astype(np.float64), 1100.0, *win)
    b_duck = _tone_level(ducked.audio.astype(np.float64), 1100.0, *win)
    assert abs(20 * np.log10(b_duck / b_plain)) < 1.0


def test_write_render_roundtrip(tmp_path, tracks):
    import json
    import soundfile as sf

    a, b = tracks
    r = _render(a, b, kind="cut")
    paths = dj.write_render(r, tmp_path, "cut_test")
    data, sr = sf.read(paths["wav"])
    assert sr == SR and len(data) == 20 * SR
    meta = json.loads(paths["json"].read_text())
    assert meta["output"]["wav_sha256"] == paths["sha256"]
    assert meta["schema_version"] == dj.DJ_RENDER_VERSION


# ------------------------------------------------------------ section gain match


def _quiet_before(track, t_end, seconds, factor):
    """Copy of ``track`` with its audio scaled by ``factor`` in [t_end - seconds, t_end)."""
    x = track["audio"].copy()
    i0, i1 = int((t_end - seconds) * SR), int(t_end * SR)
    x[i0:i1] *= factor
    return dict(track, audio=x)


def _lufs(x):
    return dj.gated_short_term_lufs(x, SR)["lufs"]


SG_ON = dj.SectionGainConfig(enabled=True)


def test_gated_short_term_lufs_ignores_silence():
    t = np.arange(4 * SR) / SR
    tone = 0.1 * np.sin(2 * np.pi * 1000 * t)
    x = np.r_[tone, np.zeros(4 * SR)]
    m = dj.gated_short_term_lufs(x, SR)
    assert m["window_s"] == 3.0 and m["blocks"] == 8
    tone_l = dj.gated_short_term_lufs(tone, SR)["lufs"]
    assert m["ungated_lufs"] == pytest.approx(tone_l - 3.01, abs=0.05)  # one block over tone + silence
    assert m["ungated_lufs"] + 0.8 < m["lufs"] < tone_l  # silent blocks gated out, partial blocks kept
    assert dj.gated_short_term_lufs(np.zeros(SR), SR)["lufs"] is None
    assert dj.gated_short_term_lufs(tone[: 2 * SR], SR)["window_s"] == pytest.approx(2.0, abs=1e-3)


def test_section_gain_curve_ramp_and_hold():
    g = dj.section_gain_curve(3000, -6.0, 1000, 1000, "ramp")
    db = 20 * np.log10(g)
    assert np.allclose(db[:1000], -6.0) and np.allclose(db[2000:], 0.0, atol=1e-9)
    assert np.all(np.diff(db) >= -1e-12)  # monotone back to 0 dB
    assert db[1500] == pytest.approx(-3.0, abs=1e-6)  # raised cosine: half way at mid ramp
    assert np.allclose(20 * np.log10(dj.section_gain_curve(3000, 4.0, 1000, 1000, "hold")), 4.0)


def test_section_gain_off_by_default(tracks):
    a, b = tracks
    r = _render(a, b, kind="cut")
    assert r.metadata["section_gain"]["enabled"] is False
    assert r.metadata["loudness"]["section_gain_b_db"] == 0.0


def test_cut_section_gain_matches_a_tail_and_ramps_back(tracks):
    a, b = tracks
    cue = a["downbeats"][8]
    aq = _quiet_before(a, cue, 8.0, 0.5)  # A's last 4 bars 6 dB under the rest of A
    r = _render(aq, b, kind="cut", config=dj.DJRenderConfig(section_gain=SG_ON), return_decks=True)
    sg = r.metadata["section_gain"]
    assert sg["enabled"] and sg["status"] == "applied" and not sg["clamped"]
    assert sg["applied_db"] == pytest.approx(sg["a_lufs"] - sg["b_lufs"], abs=1e-9)
    assert -6.0 < sg["applied_db"] < -4.0
    assert sg["a_region_source_samples"][1] == int(round(cue * SR))
    assert sg["a_region_s"] == pytest.approx(2 * 2.0, abs=1e-3)  # 2 bars of A at 120 BPM
    assert sg["b_region_s"] == pytest.approx(4 * 4 * 60 / 126, abs=1e-3)  # 4 bars of B, no stretch
    b0, b1 = sg["b_region_output_samples"]
    assert b0 == 10 * SR
    deck_b = r.decks["b"]
    assert _lufs(deck_b[b0:b1]) == pytest.approx(sg["a_lufs"], abs=0.3)  # B sits at A's section level
    ramp = sg["ramp"]
    assert ramp["output_start_sample"] == b1 and ramp["bars"] == 16 and ramp["ends_after_clip"]
    curve = r.decks["gains"]["section_gain_b"]
    assert 20 * np.log10(curve[b0]) == pytest.approx(sg["applied_db"], abs=1e-6)
    assert sg["applied_db"] < sg["gain_db_at_clip_end"] < 0.0  # ramp started, not finished in the clip
    assert r.metadata["loudness"]["section_gain_b_db"] == sg["applied_db"]
    hold = _render(aq, b, kind="cut", config=dj.DJRenderConfig(section_gain=dj.SectionGainConfig(enabled=True, after="hold")))
    assert hold.metadata["section_gain"]["gain_db_at_clip_end"] == pytest.approx(sg["applied_db"], abs=1e-6)


def test_section_gain_is_clamped_and_true_peak_holds(tracks):
    a, b = tracks
    cue = a["downbeats"][8]
    # A loud tail (+12 dB) -> B boosted, clamped to +6 dB; the limiter keeps -1 dBTP.
    loud = _quiet_before(a, cue, 8.0, 4.0)
    r = _render(loud, b, kind="cut", config=dj.DJRenderConfig(section_gain=SG_ON))
    sg = r.metadata["section_gain"]
    assert sg["raw_db"] > 6.0 and sg["clamped"] and sg["applied_db"] == pytest.approx(6.0)
    up = resample_poly(r.audio.astype(np.float64), 4, 1, axis=0)
    assert 20 * np.log10(np.max(np.abs(up))) <= -1.0 + 0.02
    r3 = _render(loud, b, kind="cut", config=dj.DJRenderConfig(section_gain=dj.SectionGainConfig(enabled=True, max_db=3.0)))
    assert r3.metadata["section_gain"]["applied_db"] == pytest.approx(3.0)


@needs_stretch
def test_blend_section_gain_measured_on_stretched_b(tracks):
    a, b = tracks
    cue = a["downbeats"][8]
    aq = _quiet_before(a, cue, 8.0, 0.5)
    r = _render(aq, b, bars=8, config=dj.DJRenderConfig(section_gain=SG_ON), return_decks=True)
    sg = r.metadata["section_gain"]
    sw, te = r.metadata["switch"]["output_sample"], r.metadata["overlap"]["output_end_sample"]
    b0, b1 = sg["b_region_output_samples"]
    assert b0 == sw and b1 - b0 == int(round(4 * r.metadata["stretch"]["a_bar_s"] * SR))  # 4 output bars of stretched B
    assert sg["ramp"]["output_start_sample"] == max(b1, te)  # never before the end of the transition
    assert -6.0 < sg["applied_db"] < -4.0
    # deck B after the switch (fader, low and high gains at 1) carries the section gain.
    assert _lufs(r.decks["b"][b0:b1]) == pytest.approx(sg["a_lufs"], abs=0.5)
    assert _lufs(r.decks["b_stretched_raw"][b0:b1]) == pytest.approx(sg["b_lufs"], abs=1e-6)
    assert np.allclose(r.decks["gains"]["section_gain_b"][: b1], 10 ** (sg["applied_db"] / 20))
