"""Test di musicmix.dj_tension (dati sintetici: nessun audio, nessun voto)."""
from __future__ import annotations

import numpy as np
import pytest

from musicmix import dj_tension as T

SR = 48000


def _analysis(sections=None, camelot="5A", level="high", dur=100.0):
    return {"tempo": {"period_s": 0.5}, "grid": {"downbeats_s": list(np.arange(0, dur, 2.0))}, "duration_s": dur,
            "sections": sections if sections is not None else [
                {"label": "intro", "downbeat_s": 0}, {"label": "verse", "downbeat_s": 16}, {"label": "chorus", "downbeat_s": 48},
                {"label": "verse", "downbeat_s": 64}, {"label": "outro", "downbeat_s": 90}],
            "key": {"camelot": camelot, "level": level}}


def _words(spec):
    return {"sample_rate": SR, "lang": "en", "words": [{"on": int(a * SR), "off": int(b * SR), "text": w, "confidence": 0.9}
                                                       for a, b, w in spec]}


def _dsp(dur=100.0, low=-20.0, ramp_at=None):
    n = int(dur * T.DSP_FPS)
    x = np.full((n, 6), low, np.float32)
    x[:, 4] = 1.0
    x[:, 5] = -30.0
    if ramp_at is not None:  # energia che sale nelle 4 battute prima di ramp_at
        a, b = int((ramp_at - 8) * T.DSP_FPS), int(ramp_at * T.DSP_FPS)
        x[a:b, :4] += np.linspace(0, 8, b - a)[:, None]
        x[a:b, 4] += np.linspace(0, 0.5, b - a)
    return x.astype(np.float16)


def test_form_position_keys():
    s = T.build_signals("x", _analysis())
    assert T.form_position(s, 48.0)["key"] == "pre_chorus_end"
    assert T.form_position(s, 64.0)["key"] == "chorus_end"
    assert T.form_position(s, 6.0)["key"] == "intro"
    assert T.form_position(s, 18.0)["key"] == "early_verse"
    assert T.form_position(s, 30.0)["key"] == "mid_verse"
    assert T.form_position(s, 45.0)["key"] == "approach_chorus"
    assert T.FORM_TABLE["pre_chorus_end"] > T.FORM_TABLE["mid_verse"] > T.FORM_TABLE["intro"]


def test_charge_rises_with_energy_ramp_and_minutage():
    flat = T.build_signals("x", _analysis(), dsp=_dsp())
    ramp = T.build_signals("x", _analysis(), dsp=_dsp(ramp_at=30.0))
    assert T.a_charge(ramp, 30.0)["components"]["energy"] > T.a_charge(flat, 30.0)["components"]["energy"]
    assert T.a_charge(ramp, 30.0)["components"]["density"] > T.a_charge(flat, 30.0)["components"]["density"]
    # stessa forma, uscita presto nel brano o dopo pochi secondi suonati: meno carica
    late = T.a_charge(flat, 48.0)
    assert T.a_charge(flat, 48.0, played_s=10.0)["value"] < late["value"]
    s = T.build_signals("x", _analysis(dur=400.0), dsp=_dsp(400.0))
    assert T.minutage(s, 20.0)["value"] == 0.0 and T.minutage(s, 200.0)["value"] == 1.0


def test_resolution_prefers_attack_after_breath():
    breath = T.build_signals("b", _analysis(), _words([(40.0, 40.4, "yeah"), (47.98, 48.3, "go"), (48.4, 48.7, "now")]), dsp=_dsp())
    glued = T.build_signals("b", _analysis(), _words([(47.0, 47.95, "yeah"), (47.97, 48.3, "go"), (48.4, 48.7, "now")]), dsp=_dsp())
    ra, rb = T.b_resolution(breath, 47.9, 48.0), T.b_resolution(glued, 47.9, 48.0)
    assert ra["components"]["vocal_attack"] > rb["components"]["vocal_attack"]
    inst = T.b_resolution(T.build_signals("b", _analysis(), _words([(70.0, 70.5, "late")]), dsp=_dsp()), 48.0, 48.0)
    assert inst["components"]["vocal_attack"] == pytest.approx(T.values()["instrumental_entry_attack"])
    # nessun dato: astensione, non un numero inventato
    assert T.b_resolution(T.build_signals("b", _analysis()), 48.0)["value"] is None


def test_continuity_table_and_gap():
    c = lambda d: T.continuity(d)["value"]  # noqa: E731
    assert c({"type": "blend", "type_key": "blend8"}) == 1.0
    assert c({"type": "filter_out"}) > c({"type": "echo_out"})
    gap = {"type": "build_into_chorus", "creative": True, "render": {"params": {"gap_beats": 1}}}
    nogap = {"type": "build_into_chorus", "creative": True, "render": {"params": {"gap_beats": 0}}}
    assert c(nogap) > c(gap) == T.CONTINUITY_GAP
    assert c({"type": "break_before_entry", "render": {"params": {"mode": "stop"}}}) == 0.0
    assert c({"type": "mystery"}) == T.CONTINUITY_UNKNOWN


def test_continuity_measured_finds_hole():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.1, SR * 10)
    full = T.continuity_measured(x, SR, SR * 5, 0.5)
    y = x.copy()
    y[SR * 5 - SR // 2: SR * 5] = 0.0   # un battito di silenzio prima dello switch
    hole = T.continuity_measured(y, SR, SR * 5, 0.5)
    assert full["value"] == 1.0 and hole["value"] < 0.6 and hole["gap_beats_low"] >= 0.875


def _band(x, lo, hi):
    from scipy.signal import butter, sosfiltfilt
    return sosfiltfilt(butter(6, [lo, hi], btype="bandpass", fs=SR, output="sos"), x)


def test_continuity_measured_filter_out_is_not_a_hole():
    """.2: un passa-alto su A (banda bassa tolta, percussioni alte intatte) prima dello switch non è un buco della base."""
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.1, SR * 10)
    sw = SR * 5
    y = x.copy()
    seg = slice(sw - 2 * SR // 2, sw)               # 2 battiti (periodo 0,5 s) di A filtrato prima dello switch
    y[seg] = _band(x, 2000.0, 20000.0)[seg]          # resta solo la parte alta
    r = T.continuity_measured(y, SR, sw, 0.5)
    assert r["value"] == 1.0 and r["gap_beats"] == 0.0
    assert r["gap_beats_low"] >= 1.5 and r["gap_beats_low_v1"] >= 1.5   # la sola banda bassa (misura .1) vedeva un buco
    assert r["version"] == T.CONTINUITY_MEASURED_VERSION


def test_continuity_measured_echo_tail_and_side_reference():
    rng = np.random.default_rng(2)
    sw = SR * 5
    # A con base piena, poi solo una coda nei medi (eco passa-alto fino a 2,5 kHz) per 1 battito: buco
    x = rng.normal(0, 0.1, SR * 10)
    y = x.copy()
    y[sw - SR // 2: sw] = 0.3 * _band(x, 300.0, 2500.0)[sw - SR // 2: sw]
    hole = T.continuity_measured(y, SR, sw, 0.5)
    assert hole["value"] <= 0.6 and hole["gap_beats"] >= 0.875
    # B più quieta di 20 dB di A ma con la sua base: riferimento del lato B, nessun buco
    z = x.copy()
    z[sw:] *= 0.1
    side = T.continuity_measured(z, SR, sw, 0.5)
    assert side["value"] == 1.0
    # lead-in della sola voce di B prima dello switch: lì vale il riferimento più basso (la base è ancora A)
    w = x.copy()
    w[sw - SR // 2: sw] *= 0.1                      # A cala di 20 dB nell'ultimo battito, B (dopo) è forte
    assert T.continuity_measured(w, SR, sw, 0.5)["value"] < 0.6
    led = T.continuity_measured(z * 0 + np.where(np.arange(len(x)) < sw - SR // 2, x, 0.1 * x), SR, sw, 0.5,
                                b_audible_sample=sw - SR // 2)
    assert led["b_audible_frame_offset_beats"] == -1.0 and led["value"] == 1.0


def test_camelot_helpers():
    assert T.camelot_distance("5A", "5A") == 0
    assert T.camelot_distance("5A", "5B") == 1 and T.camelot_distance("5A", "6A") == 1
    assert T.camelot_distance("12A", "1A") == 1 and T.camelot_distance("5A", "8A") == 3
    assert T.camelot_shift("5A", 1) == "12A" and T.camelot_shift("5A", 0) == "5A"


def test_harmonic_clash_key_and_exposure():
    tonal = np.zeros((1000, 12), np.float16)
    tonal[:, 0] = 1.0
    tonal[:, 7] = 0.5
    other = np.zeros((1000, 12), np.float16)
    other[:, 6] = 1.0
    other[:, 1] = 0.5
    a = T.build_signals("a", _analysis(camelot="5A"), chroma=tonal)
    same = T.build_signals("b", _analysis(camelot="5A"), chroma=tonal)
    far = T.build_signals("b", _analysis(camelot="11A"), chroma=other)
    cut = {"type": "cut"}
    blend = {"type": "blend", "type_key": "blend8"}
    assert T.harmonic_clash(a, 48.0, same, 48.0, cut)["value"] == pytest.approx(0.0, abs=1e-6)
    v_cut = T.harmonic_clash(a, 48.0, far, 48.0, cut)["value"]
    v_blend = T.harmonic_clash(a, 48.0, far, 48.0, blend)["value"]
    assert 0 < v_cut < v_blend          # anche senza sovrapposizione lo scontro conta, meno che nel blend
    low = T.build_signals("b", _analysis(camelot="11A", level="low"), chroma=other)
    assert T.harmonic_clash(a, 48.0, low, 48.0, cut)["key_term"] < T.harmonic_clash(a, 48.0, far, 48.0, cut)["key_term"]


def test_lyric_affinity():
    a = [{"text": "tonight", "rhyme_key": "AY T", "vowel_key": "AY", "confidence": 0.9}]
    b = [{"text": "light", "rhyme_key": "AY T", "vowel_key": "AY", "confidence": 0.9}, {"text": "up", "confidence": 0.9}]
    r = T.lyric_affinity(a, b)
    assert r["value"] > 0.6 and r["b_word"] == "light"
    assert T.lyric_affinity([{"text": "table", "confidence": 0.9}], [{"text": "orange", "confidence": 0.9}])["value"] == 0.0
    k = T.lyric_affinity([{"text": "can't", "confidence": 0.9}], [{"text": "Catch", "confidence": 0.9}])
    assert k["value"] > 0 and k["kind"] == "alliteration"     # legame 'k'/'k' (dj_mashup.sound_affinity)
    assert T.lyric_affinity(a, [dict(b[0], confidence=0.1)])["value"] == 0.0


def test_transition_cost_monotone_and_abstention():
    c0 = T.transition_cost(0.5, 0.5, 0.5, 0.0, 0.0)["total"]
    assert T.transition_cost(0.9, 0.5, 0.5, 0.0, 0.0)["total"] < c0
    assert T.transition_cost(0.5, 0.5, 1.0, 0.0, 0.0)["total"] < c0
    assert T.transition_cost(0.5, 0.5, 0.5, 0.8, 0.0)["total"] > c0
    assert T.transition_cost(0.5, 0.5, 0.5, 0.0, 1.0)["total"] < c0
    ab = T.transition_cost(None, 0.5, 0.5, None, None)
    # astensione = 0,5 (pre-registrato): carica e scontro a metà scala, nessun bonus di testo
    assert ab["abstained"] == ["charge", "clash", "lyric"]
    assert ab["total"] == pytest.approx(T.transition_cost(0.5, 0.5, 0.5, 0.5, 0.0)["total"])


def _pc():
    def cand(a_out, b_in, typ, cost, adm=True):
        return {"plan": {"a_out_s": a_out, "b_in_s": b_in, "a_out_bar": int(a_out / 2), "b_in_bar": int(b_in / 2), "score": 1.0},
                "decision": {"type": typ, "type_key": typ, "fallbacks": []}, "cost": cost, "admissible": adm, "notes": []}
    return {"candidates": [cand(30.0, 48.0, "echo_out", 0.10), cand(48.0, 48.0, "blend", 0.12), cand(20.0, 16.0, "cut", 0.0, False)]}


def test_hook_prefer_and_report():
    sig = {"a": T.build_signals("a", _analysis(), dsp=_dsp()), "b": T.build_signals("b", _analysis(), dsp=_dsp())}
    rep = T.TensionHook(sig.get, mode="report")
    pc = rep.adjust_pair(_pc(), "a", "b")
    assert [c["cost"] for c in pc["candidates"]] == [0.10, 0.12, 0.0] and "tension" in pc["candidates"][0]
    assert "tension" not in pc["candidates"][2]                      # i non ammissibili non si toccano
    pre = T.TensionHook(sig.get, mode="prefer")
    pc = pre.adjust_pair(_pc(), "a", "b")
    # il blend all'uscita prima del ritornello (carica alta, base continua) supera l'eco a metà strofa
    assert pc["candidates"][0]["decision"]["type"] == "blend"
    assert not pc["candidates"][-1]["admissible"]
    rp = pre.report({"order": ["a", "b"], "chosen": [pc["candidates"][0]]})
    assert rp["planned_transitions"][0]["continuity"] == 1.0


def test_order_attempts_continuous_first():
    sig = {"a": T.build_signals("a", _analysis()), "b": T.build_signals("b", _analysis())}
    h = T.TensionHook(sig.get, mode="prefer")
    cands = _pc()["candidates"]
    gap = {"type": "build_into_chorus", "type_key": "build_into_chorus", "creative": True, "render": {"params": {"gap_beats": 1}}}
    att = [(0, gap, "policy"), (0, {"type": "echo_out", "type_key": "echo_out"}, "albero"),
           (0, {"type": "filter_out", "type_key": "filter_out"}, "fallback"), (1, {"type": "cut", "type_key": "cut"}, "altro cue")]
    out = h.order_attempts(att, cands, "a", "b")
    assert [x[1]["type"] for x in out] == ["filter_out", "echo_out", "build_into_chorus", "cut"]
    assert T.TensionHook(sig.get, mode="report").order_attempts(att, cands, "a", "b") == att


def test_no_gap_variant_in_runner():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import run_dj_set_v3_bridge as v3

    dec = {"type": "halving_build", "type_key": "halving_build", "label": "halving", "creative": True,
           "render": {"technique": "halving_build", "params": {"gap_beats": 1, "build_bars": 4}}}
    i, d2, why = v3._no_gap((3, dec, "policy"))
    assert i == 3 and d2["render"]["params"]["gap_beats"] == 0 and dec["render"]["params"]["gap_beats"] == 1
    assert v3._no_gap((3, {"type": "echo_out", "render": {}}, "albero")) is None
