"""Synthetic checks for the hip-hop DJ effects (dj_effects).

Same click tracks as test_dj_render: downbeats 3 kHz loud, offbeats 1.5 kHz
soft, plus a sub-bass sine (A 55 Hz, B 82 Hz). A at 120 BPM (beat 0.5 s),
B at 126 BPM, different source loudness.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyloudnorm")
pytest.importorskip("soundfile")
from scipy.signal import butter, sosfiltfilt  # noqa: E402

from musicmix import dj_effects as fx  # noqa: E402
from musicmix import dj_render as dj  # noqa: E402

SR = 48000


def click_track(bpm, dur, bass_hz, *, start=0.5, bass_level=0.3, gain=1.0):
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
    x += bass_level * np.sin(2 * np.pi * bass_hz * np.arange(n) / SR)
    beats = np.array(beats)
    return {"audio": x * gain, "beats": beats, "downbeats": beats[::4]}


@pytest.fixture(scope="module")
def tracks():
    return click_track(120, 50, 55, gain=0.3), click_track(126, 50, 82, gain=1.5)


def _kw(a, b, ia=12, jb=6):
    return dict(
        beats_a=a["beats"], downbeats_a=a["downbeats"], beats_b=b["beats"], downbeats_b=b["downbeats"],
        cue_out_a=a["downbeats"][ia], cue_in_b=b["downbeats"][jb], sample_rate=SR,
    )


def _onsets(x, lo, hi, rel=0.3, min_gap_s=0.1):
    mono = x.mean(axis=1) if x.ndim == 2 else x
    sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    env = np.abs(sosfiltfilt(sos, mono))
    idx = np.flatnonzero(env > rel * env.max())
    out = []
    for i in idx:
        if not out or i - out[-1] > min_gap_s * SR:
            out.append(i)
    return np.array(out)


def _band_db(x, lo=None, hi=None):
    mono = x.mean(axis=1)
    low, _, _ = dj.split_bands(mono[:, None], SR, 200.0, 2500.0)
    return 10 * np.log10(np.mean(low ** 2) + 1e-20)


# ------------------------------------------------------------------ echo out


@pytest.mark.parametrize("delay_beats", [1.0, 0.5])
def test_echo_period_equals_a_beat(tracks, delay_beats):
    a, b = tracks
    r = fx.echo_out(a["audio"], b["audio"], delay_beats=delay_beats, entry_bars=2, return_decks=True, **_kw(a, b))
    m = r.metadata
    expected = int(round(delay_beats * 0.5 * SR))
    assert m["effect"]["delay_samples"] == expected
    reps = [x["output_sample"] for x in m["effect"]["repeats"]]
    assert np.all(np.diff(reps) == expected)
    pre = int(10 * SR)
    assert reps[0] == pre + expected
    # Measured on the PCM of the echo bus: 3 kHz downbeat click repeats every period.
    wet = r.decks["echo"]
    seg = wet[pre : pre + 5 * expected + expected // 2]
    on = _onsets(seg, 2500, 3500, rel=0.05, min_gap_s=0.6 * expected / SR)
    assert len(on) >= 4
    d = np.diff(on[:4])
    assert np.all(np.abs(d - expected) <= int(0.001 * SR)), d
    # Feedback ~0.5: each repeat about -6 dB (with the growing high-pass it can only lose more).
    pk = [np.abs(seg[max(0, o - 50) : o + 480]).max() for o in on[:3]]
    assert 0.2 < pk[1] / pk[0] <= 0.55


def test_echo_tail_highpass_grows(tracks):
    a, b = tracks
    r = fx.echo_out(a["audio"], b["audio"], **_kw(a, b))
    hz = [x["hpf_hz"] for x in r.metadata["effect"]["repeats"]]
    assert np.all(np.diff(hz) > 0)
    gains = [x["gain_db"] for x in r.metadata["effect"]["repeats"]]
    assert np.allclose(np.diff(gains), 20 * np.log10(0.5))


@pytest.mark.parametrize("entry_bars", [0, 1, 2])
def test_echo_b_starts_on_expected_sample_and_clip(tracks, entry_bars):
    a, b = tracks
    r = fx.echo_out(a["audio"], b["audio"], entry_bars=entry_bars, return_decks=True, **_kw(a, b))
    m = r.metadata
    pre = int(10 * SR)
    entry = pre + entry_bars * int(round(2.0 * SR))  # A bar = 2 s at 120 BPM
    assert m["switch"]["output_sample"] == entry
    assert m["clip"]["length_samples"] == entry + int(10 * SR) == len(r.audio)
    assert m["clip"]["matches_rule"]
    sb = int(round(b["downbeats"][6] * SR))
    deck_b = r.decks["b"]
    bn = b["audio"] * 10 ** (m["loudness"]["b_gain_db"] / 20)
    np.testing.assert_allclose(deck_b[entry : entry + 4800, 0], bn[sb : sb + 4800], atol=1e-9)
    d = int(round(0.010 * SR))
    assert np.all(deck_b[: entry - d] == 0.0)
    # B's downbeat click is the first 3 kHz onset of the B deck, on the expected sample.
    on = _onsets(deck_b, 2500, 3500, rel=0.3)
    assert abs(on[0] - entry) <= 24
    assert m["stretch"]["applied"] is False


def test_echo_loudness_and_true_peak(tracks):
    a, b = tracks
    r = fx.echo_out(a["audio"], b["audio"], **_kw(a, b))
    L = r.metadata["loudness"]
    assert abs(L["a_integrated_lufs_normalized"] + 14) < 0.05
    assert abs(L["b_integrated_lufs_normalized"] + 14) < 0.05
    assert L["output_true_peak_dbtp"] <= -1.0 + 1e-6
    assert dj.true_peak_dbtp(r.audio, SR) <= -1.0 + 0.01
    for key in ("lufs_before_effect", "lufs_after_b_entry", "loudness_jump_db"):
        assert L[key] is not None


# ---------------------------------------------------------------- filter out


@pytest.mark.parametrize("sweep_bars", [1, 2, 4])
def test_filter_out_kills_lows_by_20db_at_sweep_end(tracks, sweep_bars):
    a, b = tracks
    r = fx.filter_out(a["audio"], b["audio"], sweep_bars=sweep_bars, return_decks=True, **_kw(a, b))
    m = r.metadata
    pre = int(10 * SR)
    cut = m["switch"]["output_sample"]
    assert cut == pre + sweep_bars * 2 * SR
    win = int(0.25 * SR)
    filt = r.decks["a_filtered_prefader"][cut - win - int(0.01 * SR) : cut - int(0.01 * SR)]
    raw = r.decks["a_raw"][cut - win - int(0.01 * SR) : cut - int(0.01 * SR)]
    drop = _band_db(raw) - _band_db(filt)
    assert drop > 20.0, drop
    # Before the sweep A is untouched.
    np.testing.assert_allclose(r.decks["a"][: pre - 10], r.decks["a_raw"][: pre - 10], atol=1e-12)
    assert m["clip"]["length_samples"] == cut + 10 * SR == len(r.audio)
    assert m["clip"]["matches_rule"]
    assert m["overlap"]["duration_s"] == pytest.approx(2.0 * sweep_bars)


def test_filter_out_b_on_downbeat_and_peaks(tracks):
    a, b = tracks
    r = fx.filter_out(a["audio"], b["audio"], sweep_bars=2, return_decks=True, **_kw(a, b))
    m = r.metadata
    cut = m["switch"]["output_sample"]
    sb = int(round(b["downbeats"][6] * SR))
    bn = b["audio"] * 10 ** (m["loudness"]["b_gain_db"] / 20)
    np.testing.assert_allclose(r.decks["b"][cut : cut + 4800, 0], bn[sb : sb + 4800], atol=1e-9)
    assert np.all(r.decks["a"][cut:] == 0.0)
    assert m["loudness"]["output_true_peak_dbtp"] <= -1.0 + 1e-6


def test_filter_in_opens_low_pass_on_b(tracks):
    a, b = tracks
    r = fx.filter_out(a["audio"], b["audio"], sweep_bars=2, filter_in=True, filter_in_bars=2, return_decks=True, **_kw(a, b))
    m = r.metadata
    fin = m["effect"]["filter_in"]
    assert fin["enabled"]
    cut = m["switch"]["output_sample"]
    deck_b = r.decks["b"]
    # The 3 kHz downbeat click of B is strongly attenuated at entry, intact after the opening.
    hi = lambda seg: np.abs(sosfiltfilt(butter(4, [2500, 3500], btype="band", fs=SR, output="sos"), seg.mean(axis=1))).max()
    bar_b = 4 * 60 / 126
    first = deck_b[cut : cut + 2400]
    late_start = cut + int(round(3 * bar_b * SR))
    late = deck_b[late_start : late_start + 2400]
    assert 20 * np.log10(hi(first) / hi(late)) < -20.0
    assert fin["output_end_sample"] - cut == pytest.approx(2 * bar_b * SR, abs=2)


# ------------------------------------------------------------------- drop in


def test_drop_in_is_cut_on_downbeat(tracks):
    a, b = tracks
    r = fx.drop_in(a["audio"], b["audio"], section_label="chorus", return_decks=True, **_kw(a, b))
    m = r.metadata
    pre = int(10 * SR)
    assert m["kind"] == "drop_in" and m["base_kind"] == "cut"
    assert m["switch"]["output_sample"] == pre
    assert m["clip"]["length_samples"] == 20 * SR == len(r.audio)
    assert m["clip"]["matches_rule"]
    on = _onsets(r.decks["b"], 2500, 3500)
    assert abs(on[on >= pre - 480][0] - pre) <= 24
    assert m["loudness"]["output_true_peak_dbtp"] <= -1.0 + 1e-6
    assert m["effect"]["section_label"] == "chorus"


def test_drop_in_requires_section_start(tracks):
    a, b = tracks
    with pytest.raises(dj.DJRenderError):
        fx.drop_in(a["audio"], b["audio"], cue_in_is_section_start=False, **_kw(a, b))


def test_off_downbeat_and_bad_params_rejected(tracks):
    a, b = tracks
    kw = _kw(a, b)
    kw["cue_out_a"] += 0.1
    with pytest.raises(dj.DJRenderError):
        fx.echo_out(a["audio"], b["audio"], **kw)
    with pytest.raises(dj.DJRenderError):
        fx.echo_out(a["audio"], b["audio"], delay_beats=0.25, **_kw(a, b))
    with pytest.raises(dj.DJRenderError):
        fx.filter_out(a["audio"], b["audio"], sweep_bars=5, **_kw(a, b))
    with pytest.raises(dj.DJRenderError):
        fx.render_effect("spinback", a["audio"], b["audio"], **_kw(a, b))


def test_write_render_roundtrip(tmp_path, tracks):
    import json

    a, b = tracks
    r = fx.render_effect("echo_out", a["audio"], b["audio"], **_kw(a, b))
    paths = dj.write_render(r, tmp_path, "echo")
    meta = json.loads(paths["json"].read_text())
    assert meta["kind"] == "echo_out" and meta["schema_version"] == fx.DJ_EFFECTS_VERSION


# ------------------------------------------------------------ section gain match


def _quiet_before(track, t_end, seconds, factor):
    x = track["audio"].copy()
    x[int((t_end - seconds) * SR) : int(t_end * SR)] *= factor
    return dict(track, audio=x)


def _cfg_sg(**kw):
    return fx.DJEffectsConfig(section_gain=dj.SectionGainConfig(enabled=True, **kw))


@pytest.mark.parametrize("kind", ["echo_out", "filter_out"])
def test_effects_section_gain_matches_a_before_effect(tracks, kind):
    a, b = tracks
    ia = 12
    t_a_end = a["downbeats"][ia] if kind == "echo_out" else a["downbeats"][ia - 2]  # dry A ends at the effect start
    aq = _quiet_before(a, t_a_end, 8.0, 0.5)
    extra = {"sweep_bars": 2} if kind == "filter_out" else {}
    r = fx.render_effect(kind, aq["audio"], b["audio"], config=_cfg_sg(), return_decks=True, **extra, **_kw(aq, b, ia=ia))
    sg = r.metadata["section_gain"]
    assert sg["enabled"] and sg["a_region_source_samples"][1] == int(round(t_a_end * SR))
    assert sg["applied_db"] == pytest.approx(sg["a_lufs"] - sg["b_lufs"], abs=1e-9) and -6.0 < sg["applied_db"] < -4.0
    b0, b1 = sg["b_region_output_samples"]
    assert b0 == r.metadata["switch"]["output_sample"]  # B's entry downbeat
    assert dj.gated_short_term_lufs(r.decks["b"][b0:b1], SR)["lufs"] == pytest.approx(sg["a_lufs"], abs=0.3)
    assert r.metadata["loudness"]["section_gain_b_db"] == sg["applied_db"]
    assert r.metadata["limiter"]["true_peak_after_dbtp"] <= -1.0 + 1e-6
    off = fx.render_effect(kind, aq["audio"], b["audio"], **extra, **_kw(aq, b, ia=ia))
    assert off.metadata["section_gain"]["enabled"] is False and off.metadata["loudness"]["section_gain_b_db"] == 0.0


def test_drop_in_passes_section_gain_to_the_cut(tracks):
    a, b = tracks
    aq = _quiet_before(a, a["downbeats"][12], 8.0, 0.5)
    r = fx.drop_in(aq["audio"], b["audio"], config=_cfg_sg(after="hold"), **_kw(aq, b))
    sg = r.metadata["section_gain"]
    assert sg["enabled"] and sg["after"] == "hold" and -6.0 < sg["applied_db"] < -4.0
    assert r.metadata["config"]["section_gain"]["enabled"] is True
