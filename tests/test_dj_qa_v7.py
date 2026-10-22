"""dj_qa.7 (palestra dei difetti v2): fase di B a un ingresso pulito, frase di B, buco, clic, eco fuori tempo."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.signal import butter, sosfiltfilt  # noqa: E402

from musicmix import dj_critic_books as CB  # noqa: E402
from musicmix import dj_qa  # noqa: E402

SR = 48000
BEAT = 0.5


def _music(n_s: float, seed: int = 0, beat: float = BEAT, shift: float = 0.0) -> np.ndarray:
    """Band-limited (< 15 kHz, like the coded sources) beat-synchronous noise bursts + a bass tone."""
    rng = np.random.default_rng(seed)
    n = int(n_s * SR)
    x = np.zeros(n)
    burst = np.exp(-np.arange(int(0.08 * SR)) / (0.02 * SR)) * rng.standard_normal(int(0.08 * SR))
    for k in range(int(n_s / beat) + 1):
        i = int((k * beat + shift) * SR)
        if 0 <= i < n:
            m = min(len(burst), n - i)
            x[i:i + m] += 0.3 * burst[:m]
    t = np.arange(n) / SR
    x += 0.1 * np.sin(2 * np.pi * 55 * t)
    x = sosfiltfilt(butter(8, 15000, btype="low", fs=SR, output="sos"), x)
    return np.stack([x, x], 1)


def _cut_meta(n: int, **extra) -> dict:
    sw = 10 * SR
    meta = {"kind": "cut", "sample_rate": SR, "overlap": {"output_start_sample": sw - 480, "output_end_sample": sw,
                                                          "a_source_start_sample": sw - 480, "b_source_start_sample": sw - 480},
            "switch": {"output_sample": sw, "b_source_sample": sw}, "stretch": {"applied": False, "a_bar_s": 4 * BEAT, "b_bar_s": 4 * BEAT},
            "clip": {"length_samples": n}}
    meta.update(extra)
    return meta


def test_click_at_the_switch():
    x = _music(20.0)
    meta = _cut_meta(len(x))
    geo = dj_qa.transition_geometry(meta, len(x))
    cfg = dj_qa.QAConfig()
    base = dj_qa.check_click(x, geo, meta, cfg)
    assert base["status"] == "pass" and base["coverage"] == "measured"
    y = x.copy()
    y[10 * SR:10 * SR + 48] += 0.5  # 1 ms at -6 dBFS
    c = dj_qa.check_click(y, geo, meta, cfg)
    assert c["status"] == "fail" and c["worst_point"] == "switch" and c["statistic"] > cfg.click_ratio
    assert dj_qa.check_click(x, geo, meta, dj_qa.QAConfig(click_check="off"))["status"] == "not_applicable"


def _echo_scene(delay_beats: float):
    n = int(16 * SR)
    a = np.zeros((n, 2))
    sw = 8 * SR
    rng = np.random.default_rng(3)
    hit = np.exp(-np.arange(int(0.05 * SR)) / (0.01 * SR)) * rng.standard_normal(int(0.05 * SR))
    for k in range(16):  # A: one transient per beat until the switch
        i = int(k * BEAT * SR)
        if i + len(hit) < sw:
            a[i:i + len(hit), 0] += hit
            a[i:i + len(hit), 1] += hit
    mix = a.copy()
    last = int((sw / SR - BEAT) * SR)
    for k in range(5):  # echo of the last transient after the exit
        i = sw + int(k * delay_beats * BEAT * SR)
        g = 0.6 * 0.6 ** k
        mix[i:i + len(hit)] += g * a[last:last + len(hit)]
    meta = {"kind": "echo_out", "sample_rate": SR, "overlap": {"output_start_sample": sw - 2 * SR, "output_end_sample": sw},
            "switch": {"output_sample": sw}, "stretch": {"a_bar_s": 4 * BEAT}, "effect": {"type": "echo_out"}}
    return mix, {"a": a}, meta


@pytest.mark.parametrize("delay,expected", [(0.5, "pass"), (1.0, "pass"), (0.37, "fail")])
def test_echo_tempo(delay, expected):
    mix, decks, meta = _echo_scene(delay)
    geo = dj_qa.transition_geometry(meta, len(mix))
    c = dj_qa.check_echo_tempo(mix, geo, decks, meta, dj_qa.QAConfig(echo_severity="fail"))
    assert c["status"] == expected, c
    # default severity is warn (natural continuations show off-grid progressions, calibration of 22/10/2026)
    assert dj_qa.check_echo_tempo(mix, geo, decks, meta, dj_qa.QAConfig())["status"] == ("warn" if expected == "fail" else "pass")
    if expected == "fail":
        assert abs(c["spacing_beats"] - 0.37) < 0.01


def test_phrase_position_of_b(tmp_path):
    bar = 4 * BEAT
    rows = [{"bar_index": j, "time_s": 1.0 + j * bar, "phrase_level": (16 if j % 16 == 0 else 8 if j % 8 == 0 else 4 if j % 4 == 0 else 0),
             "section_start": j % 16 == 0} for j in range(64)]
    an = tmp_path / "b.json"
    an.write_text(json.dumps({"phrases": rows}))
    geo = {"sr": SR}
    cfg = dj_qa.QAConfig()
    prov = {"b": {"slug": "nonexistent_slug_for_test", "analysis": str(an)}}
    on = dj_qa.check_phrase(geo, {"cue_in_b_s": 1.0 + 8 * bar, "provenance": prov}, cfg)
    assert on["status"] == "pass" and on["b_cue_bar"] == 8 and on["required"]
    late = dj_qa.check_phrase(geo, {"cue_in_b_s": 1.0 + 7 * bar, "provenance": prov}, cfg)
    assert late["status"] == "fail" and late["off_bars"] == -1 and late["problem"] == "b_off_phrase"
    early = dj_qa.check_phrase(geo, {"cue_in_b_s": 1.0 + 9 * bar + 0.1, "provenance": prov}, cfg)
    assert early["status"] == "fail" and early["off_bars"] == 1
    reloc = dj_qa.check_phrase(geo, {"cue_in_b_s": 1.0 + 7 * bar, "provenance": prov, "b_entry": {"relocated_bars": 1}}, cfg)
    assert reloc["status"] == "warn"
    none = dj_qa.check_phrase(geo, {"cue_in_b_s": 3.0, "provenance": {"b": {"slug": "nonexistent_slug_for_test"}}}, cfg)
    assert none["status"] == "skipped" and none["coverage"] == "unavailable"


def test_undeclared_gap_fails_and_declared_pause_is_discounted():
    x = _music(20.0)
    meta = _cut_meta(len(x))
    geo = dj_qa.transition_geometry(meta, len(x))
    cfg = dj_qa.QAConfig()
    assert dj_qa.check_dropout(x, geo, meta, cfg)["status"] == "pass"
    y = x.copy()
    y[10 * SR:10 * SR + int(BEAT * SR)] = 0.0
    d = dj_qa.check_dropout(y, geo, meta, cfg)
    assert d["status"] == "fail" and d["longest_beats"] > 0.9
    meta2 = dict(meta, creative={"technique": "build_into_chorus", "params": {"gap_beats": 1}})
    assert dj_qa.check_dropout(y, geo, meta2, cfg)["status"] == "pass"  # cut window: the declared beat is discounted
    assert dj_qa.check_dropout(y, geo, meta, dj_qa.QAConfig(dropout_severity="warn"))["status"] == "warn"


def test_critic_v4_reads_the_reliable_entry_phase():
    qa = {"checks": {"rhythm": {"methods": {"beat_this_entry": {"status": "fail", "reliable": True,
                                                                 "beat_phase_offset_at_switch_ms": 180.0, "threshold_ms": 60.0}}}}}
    e = CB.qa_entry_phase(qa)
    assert e["veto"] and e["offset_ms"] == 180.0
    assert not CB.qa_entry_phase({"checks": {"rhythm": {"methods": {"beat_this_entry": {"status": "unreliable"}}}}})["veto"]
    assert CB.qa_dropout({"checks": {"dropout": {"status": "fail", "longest_s": 0.5}}})["veto"]
