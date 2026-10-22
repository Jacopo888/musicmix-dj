"""Integrazione delle proposte dei libri (21/10/2026) nei moduli di render, effetti, tecniche, selezione e QA.

Ogni test cita la proposta di ``conoscenza_mix/libri/INTEGRAZIONE_PROPOSTE_2026-10-21.md``.
Segnali sintetici come in test_dj_creative / test_dj_effects: click track (downbeat 3 kHz forte,
battiti 1,5 kHz deboli, sub-basso sinusoidale) e "stem vocali" a toni.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("pyloudnorm")
from scipy.signal import butter, sosfiltfilt  # noqa: E402

from musicmix import dj_creative as cr  # noqa: E402
from musicmix import dj_phrase as P  # noqa: E402

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


def vocal_stem(track, bars, freq, level=0.25):
    n = len(track["audio"])
    v = np.zeros(n)
    beats = track["beats"]
    period = float(np.median(np.diff(beats)))
    for b in bars:
        for k in range(4):
            i = 4 * b + k
            if i >= len(beats):
                break
            s = int(round(beats[i] * SR))
            m = int(0.7 * period * SR)
            t = np.arange(m) / SR
            env = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)
            v[s : s + m] += level * env * (np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * 1.5 * freq * t))
    return v


@pytest.fixture(scope="module")
def tracks():
    a = click_track(120, 40, 55, gain=0.4)
    b = click_track(100, 40, 82, gain=1.2)  # far tempo: no-overlap techniques only
    a["vocals"] = vocal_stem(a, list(range(0, 12)), 700.0)  # A sings bars 0..11
    b["vocals"] = vocal_stem(b, list(range(4, 14)), 600.0, level=0.6)
    a["mix"] = a["audio"] + a["vocals"]
    b["mix"] = b["audio"] + b["vocals"]
    return a, b


def _kw(a, b, ia=12, jb=4, vocals=True):
    kw = dict(beats_a=a["beats"], downbeats_a=a["downbeats"], beats_b=b["beats"], downbeats_b=b["downbeats"],
              cue_out_a=a["downbeats"][ia], cue_in_b=b["downbeats"][jb], sample_rate=SR)
    if vocals:
        kw.update(vocals_a=a["vocals"], vocals_b=b["vocals"])
    return kw


def _band(x, lo, hi):
    mono = x.mean(axis=1) if x.ndim == 2 else x
    sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    return sosfiltfilt(sos, mono)


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2))) if len(x) else 0.0


def _db(x):
    return 20 * np.log10(max(x, 1e-12))


# ------------------------------------------------------------------ C1 halving build (Margulis pp. 23, 44)
def test_halving_stages_bar_aligned():
    st = cr.halving_stages(8, 2.0, 0.25)
    assert [s["unit_bars"] for s in st] == [2.0, 1.0, 0.5, 0.25]
    assert [s["repeats"] for s in st] == [2.0, 2.0, 2.0, 4.0]
    assert sum(s["span_bars"] for s in st) == pytest.approx(8.0)
    st4 = cr.halving_stages(4, 2.0, 0.25)  # 2-bar unit cannot repeat twice in half of 4 bars: starts at 1
    assert [s["unit_bars"] for s in st4] == [1.0, 0.5, 0.25]
    assert [s["repeats"] for s in st4] == [2.0, 2.0, 4.0]
    assert all(s["repeats"] >= 2 for s in cr.halving_stages(2, 1.0, 0.25))


def test_halving_build_repeats_shrinking_units(tracks):
    a, b = tracks
    r = cr.halving_build(a["mix"], b["mix"], **_kw(a, b), build_bars=4, start_unit=1.0, trajectory="none", return_decks=True)
    m = r.metadata
    assert m["kind"] == "halving_build" and m["clip"]["matches_rule"]
    cre = m["creative"]
    assert cre["params"]["source"] == "instrumental"  # A sings on the looped bar: words are not repeated
    pieces = cre["pieces"]
    assert [p["unit_bars"] for p in pieces] == [1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25]
    deck = r.decks["a_build_prefader"]
    p1, p2 = pieces[-2]["output"], pieces[-1]["output"]
    x1 = deck[p1[0] + 200 : p1[1] - 200, 0]
    x2 = deck[p2[0] + 200 : p2[0] + 200 + len(x1), 0]
    assert np.corrcoef(x1, x2)[0, 1] > 0.99  # identical quarter-bar repeats
    # the first click of each piece is a downbeat click (3 kHz): every repeat restarts on the loop start
    sw = m["switch"]["output_sample"]
    assert pieces[-1]["output"][1] == sw
    voc = r.decks["vocals_a_heard"]
    assert _rms(voc[pieces[0]["output"][0] : sw]) < 1e-6  # instrumental loop: A's words muted
    # B enters on its downbeat at the switch, nothing of A after it
    assert _rms(r.decks["a"][sw + 960 : sw + SR]) < 1e-6


def test_halving_build_gap_is_silent(tracks):
    a, b = tracks
    r = cr.halving_build(a["mix"], b["mix"], **_kw(a, b), build_bars=2, start_unit=0.5, gap_beats=1, return_decks=True)
    sw = r.metadata["switch"]["output_sample"]
    beat = int(0.5 * SR)
    assert _rms(r.decks["a"][sw - beat + 600 : sw - 100]) < 1e-6
    assert r.metadata["exit_effect"] == "filter"


# ------------------------------------------------------------------ break before B (Huron pp. 248-249, 314-316; Katz p. 106)
def test_break_stop_silences_last_beat(tracks):
    a, b = tracks
    r = cr.break_before_entry(a["mix"], b["mix"], **_kw(a, b), mode="stop", beats=1, return_decks=True)
    sw = r.metadata["switch"]["output_sample"]
    beat = int(0.5 * SR)
    out = r.audio.astype(float)
    assert _db(_rms(out[sw - beat + 600 : sw - 600])) < _db(_rms(out[sw - 2 * beat : sw - beat])) - 60
    assert r.metadata["creative"]["a_vocal_at_break"]["a_vocal_cover_last_beat"] > 0.1  # A was singing: reported


def test_break_beat_mute_keeps_the_voice(tracks):
    a, b = tracks
    r = cr.break_before_entry(a["mix"], b["mix"], **_kw(a, b), mode="beat_mute", beats=1, return_decks=True)
    sw = r.metadata["switch"]["output_sample"]
    beat = int(0.5 * SR)
    deck = r.decks["a"]
    seg, ref = deck[sw - beat + 600 : sw - 600], deck[sw - 2 * beat + 600 : sw - beat - 600]
    inst = deck - r.decks["vocals_a_heard"]
    assert _rms(inst[sw - beat + 600 : sw - 600]) < 1e-9 < _rms(inst[sw - 2 * beat : sw - beat])  # instrumental gone
    assert _rms(_band(seg, 650, 750)) > 0.5 * _rms(_band(ref, 650, 750))  # A's voice still there


def test_break_fill_adds_snare_in_last_beats(tracks):
    a, b = tracks
    r = cr.break_before_entry(a["mix"], b["mix"], **_kw(a, b), mode="fill", beats=2, snare="synth", return_decks=True)
    cre = r.metadata["creative"]
    assert cre["fill"]["events"] == 8
    sw = r.metadata["switch"]["output_sample"]
    beat = int(0.5 * SR)
    ov = r.decks["overlay"]
    assert _rms(ov[: sw - 2 * beat - 10]) == 0.0
    assert _rms(ov[sw - 2 * beat : sw]) > 0.0


# ------------------------------------------------------------------ dj_render: bass gap / mid duck (B&B cap. 28; Butler pp. 53, 246)
import shutil  # noqa: E402

from musicmix import dj_effects as fx  # noqa: E402
from musicmix import dj_render as dj  # noqa: E402

needs_stretch = pytest.mark.skipif(not (shutil.which("rubberband") or shutil.which("ffmpeg")), reason="no stretch backend")


def _blend(cfg, bars=8):
    a = click_track(120, 40, 55, gain=0.3)
    b = click_track(123, 40, 82, gain=1.0)
    return dj.render_transition(a["audio"], b["audio"], beats_a=a["beats"], downbeats_a=a["downbeats"], beats_b=b["beats"],
                                downbeats_b=b["downbeats"], cue_out_a=a["downbeats"][6], cue_in_b=b["downbeats"][4], kind="blend",
                                bars=bars, sample_rate=SR, config=cfg, return_decks=True)


@needs_stretch
def test_bass_gap_slams_b_bass_on_the_switch():
    r = _blend(dj.DJRenderConfig(cue_convention="start", bass_gap_beats=2))
    g, m = r.decks["gains"], r.metadata
    sw = m["switch"]["output_sample"]
    gap = m["eq"]["books"]["bass_gap"]
    assert gap["b_low_on_output_sample"] == sw
    g0 = gap["a_low_off_output_sample"]
    assert sw - g0 == pytest.approx(2 * 0.5 * SR, abs=SR * 0.02)  # two beats at 120 BPM
    assert g["low_a"][g0 + 10] == 0.0 and g["low_b"][sw - 2000] == 0.0  # no bass at all in the gap
    assert g["low_a"][g0 - 2000] == 1.0 and g["low_b"][sw + 10] == 1.0
    assert np.all(g["low_a"] + g["low_b"] <= 1.0 + 1e-9)  # still one bass at a time


@needs_stretch
def test_mid_duck_b_then_back_to_0db():
    r = _blend(dj.DJRenderConfig(cue_convention="start", mid_duck_b_bars=2))
    g, m = r.decks["gains"], r.metadata
    t_s, sw = m["overlap"]["output_start_sample"], m["switch"]["output_sample"]
    assert _db(g["mid_b"][t_s + 100]) == pytest.approx(-12.0, abs=0.01)
    assert g["mid_b"][sw] == 1.0
    assert np.all(np.diff(g["mid_b"][t_s:sw]) >= -1e-12)  # monotone back to 0 dB
    r0 = _blend(dj.DJRenderConfig(cue_convention="start"))
    assert np.all(r0.decks["gains"]["mid_b"] == 1.0) and r0.metadata["eq"]["books"] is None  # default unchanged


# ------------------------------------------------------------------ dj_effects: manual echo (Webber pp. 265-266), keep kick (p. 181)
def _fx_tracks():
    a = click_track(120, 40, 55, gain=0.4)
    b = click_track(100, 40, 82, gain=1.0)
    kw = dict(beats_a=a["beats"], downbeats_a=a["downbeats"], beats_b=b["beats"], downbeats_b=b["downbeats"],
              cue_out_a=a["downbeats"][8], cue_in_b=b["downbeats"][4], sample_rate=SR)
    return a, b, kw


def test_manual_echo_steps_down_6db_per_beat():
    a, b, kw = _fx_tracks()
    r = fx.echo_out(a["audio"], b["audio"], entry_bars=2, config=fx.DJEffectsConfig(echo_manual_repeats=4, echo_step_db=-6.0),
                    return_decks=True, **kw)
    reps = r.metadata["effect"]["repeats"]
    assert len(reps) == 4 and all(x["hpf_hz"] is None for x in reps)
    wet = r.decks["echo"].mean(axis=1)
    D = r.metadata["effect"]["delay_samples"]
    pk = [np.max(np.abs(wet[x["output_sample"] : x["output_sample"] + D // 2])) for x in reps]
    steps = np.diff(20 * np.log10(pk))
    assert np.allclose(steps, -6.0, atol=0.5)


def test_echo_tail_can_stop_on_b():
    a, b, kw = _fx_tracks()
    r = fx.echo_out(a["audio"], b["audio"], entry_bars=1, config=fx.DJEffectsConfig(tail_under_b_db=None), return_decks=True, **kw)
    sw = r.metadata["switch"]["output_sample"]
    assert r.metadata["effect"]["tail_stopped_on_b"]
    assert _rms(r.decks["echo"][sw + 10 :]) == 0.0


def test_filter_out_keep_kick_keeps_low_band():
    a, b, kw = _fx_tracks()
    r0 = fx.filter_out(a["audio"], b["audio"], sweep_bars=2, return_decks=True, **kw)
    r1 = fx.filter_out(a["audio"], b["audio"], sweep_bars=2, config=fx.DJEffectsConfig(filter_keep_low_hz=120.0), return_decks=True, **kw)
    sw = r1.metadata["switch"]["output_sample"]
    seg = slice(sw - int(0.6 * SR), sw - int(0.05 * SR))
    lo0, lo1 = _rms(_band(r0.decks["a"][seg], 45, 65)), _rms(_band(r1.decks["a"][seg], 45, 65))
    ref = _rms(_band(r1.decks["a"][: int(9 * SR)], 45, 65))
    assert _db(lo0) < _db(ref) - 20  # default: A's bass filtered away
    assert _db(lo1) > _db(ref) - 1.5  # keep kick: A's low band intact up to the cut
    hi0, hi1 = _rms(_band(r0.decks["a"][seg], 400, 900)), _rms(_band(r1.decks["a"][seg], 400, 900))
    assert hi1 == pytest.approx(hi0, rel=0.2)  # the sweep above is unchanged


# ------------------------------------------------------------------ dj_policy wiring (profilo libri)
from musicmix import dj_policy as pol  # noqa: E402
from musicmix import dj_qa  # noqa: E402


def _pctx(**kw):
    base = {"stretch_abs": 0.02, "key_relation": "same", "key_confident": True, "key_shift": 0, "has_stems": True,
            "a_vocal_last_beat": 0.0, "a_vocal_bars_before": [1.0, 1.0, 0.0, 0.0], "a_vocal_bars": [0.0] * 16,
            "b_vocal_bars": [0.0] * 16, "a_bars_after": 20, "a_bars_before": 40, "b_section": "chorus",
            "energy_a_db_rel": 0.0, "energy_b_db_rel": 4.0, "key_a": "A minor", "key_b": "A minor", "bar_a_s": 2.0,
            "lyric_links": [], "recent_techniques": [], "a_exit_on_boundary": True, "b_section_start": True}
    base.update(kw)
    return base


def test_policy_halving_build_and_break_only_with_books_profile():
    t0 = [c["technique"] for c in pol.propose(_pctx(), max_proposals=8)["proposals"]]
    assert "halving_build" not in t0 and "break_before_entry" not in t0
    r = pol.propose(_pctx(), max_proposals=8, params=pol.PROFILE_BOOKS)
    by = {p["technique"]: p for p in r["proposals"]}
    assert by["halving_build"]["params"]["build_bars"] == 8 and by["halving_build"]["params"]["gap_beats"] == 1
    assert by["build_into_chorus"]["params"]["gap_beats"] == 1  # Huron pp. 314-316: one beat of stop before the drop
    assert by["break_before_entry"]["params"]["mode"] == "fill"
    sung = pol.propose(_pctx(a_vocal_last_beat=0.9), max_proposals=8, params=pol.PROFILE_BOOKS)
    assert {p["technique"]: p for p in sung["proposals"]}["break_before_entry"]["params"]["mode"] == "beat_mute"
    # A keeps rapping after the cue: no break (it would cut the words, as dj_qa verse_cut showed on heartless -> coehn)
    through = pol.propose(_pctx(a_vocal_last_beat=0.9, a_vocal_bars=[1.0] * 16), max_proposals=8, params=pol.PROFILE_BOOKS)
    assert "break_before_entry" not in [p["technique"] for p in through["proposals"]]


# ------------------------------------------------------------------ dj_qa D4 (Huron p. 52): unintended dropout
def test_qa_dropout_warns_only_when_not_declared(tracks):
    a, b = tracks
    r = cr.break_before_entry(a["mix"], b["mix"], **_kw(a, b), mode="stop", beats=2)
    cfg = dj_qa.QAConfig(dropout_check=True)
    q = dj_qa.evaluate_render(r.audio, r.metadata, tracker=None, config=cfg)
    assert q["checks"]["dropout"]["status"] == "pass" and q["checks"]["dropout"]["intended"]
    meta = {k: v for k, v in r.metadata.items() if k != "creative"}
    meta["kind"] = "cut_like"
    q2 = dj_qa.evaluate_render(r.audio, meta, tracker=None, config=cfg)
    # dj_qa.7: an undeclared silence is a fail (was warn); here the 2-beat stop sits before B's entry (effect region)
    d2 = q2["checks"]["dropout"]
    assert d2["status"] == "fail" and d2["effect_region_longest_s"] >= 0.5
    q3 = dj_qa.evaluate_render(r.audio, meta, tracker=None, config=dataclasses.replace(cfg, dropout_severity="warn"))
    assert q3["checks"]["dropout"]["status"] == "warn"  # behaviour of .4-.6 on request
    assert "dropout" in dj_qa.evaluate_render(r.audio, meta, tracker=None)["checks"]  # on by default since .7


# ------------------------------------------------------------------ dj_set B1 (Huron pp. 207-211): far genres
from musicmix import dj_set as ds  # noqa: E402


def _cand(kind, bars=0, cost=0.0):
    dec = {"type": kind, "bars": bars, "render": {"delay_beats": 1.0, "entry_bars": 1} if kind == "echo_out" else {}}
    return {"decision": dec, "cost": cost, "admissible": True, "plan": {"score": 0.0}, "notes": []}


def test_far_genre_threshold_and_clean_switch():
    D = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]], float)
    assert ds.far_timbre_threshold(D, None) is None
    thr = ds.far_timbre_threshold(D, 90)
    assert 2.0 < thr <= 3.0
    pc = {"candidates": [_cand("echo_out", cost=0.1), _cand("blend", 4, 0.2), _cand("cut", cost=0.3)]}
    near = ds.apply_far_genre(pc, 1.0, thr, 90.0)
    assert not near["far_genre"]["far"] and "tail_stop" not in near["candidates"][0]["decision"]["render"]
    far = ds.apply_far_genre(pc, 3.0, thr, 90.0)
    echo = next(c for c in far["candidates"] if c["decision"]["type"] == "echo_out")
    assert echo["decision"]["render"]["tail_stop"] is True
    assert next(c for c in far["candidates"] if c["decision"]["type"] == "blend")["admissible"]  # 4 bars at 90 BPM = 10.7 s >= 3 s
    fast = ds.apply_far_genre({"candidates": [_cand("blend", 4)]}, 3.0, thr, 400.0)  # 4 bars at 400 BPM = 2.4 s
    assert not fast["candidates"][0]["admissible"]
    assert "tail_stop" not in pc["candidates"][0]["decision"]["render"]  # input untouched
    assert ds.PROFILE_BOOKS["order"]["far_timbre_pct"] == 90.0 and ds.ORDER_PARAMS["far_timbre_pct"].value is None


def test_cut_full_both_sides_costs_only_when_enabled():
    plan = {"cue_score": 5.0}
    dec = {"type": "cut", "type_key": "cut"}
    ctx = {"form_available": True, "a_full_before": True, "b_full_at": True, "a_exit_on_boundary": True, "b_section_start": True}
    c0 = ds.candidate_cost(plan, dec, None, ctx)
    c1 = ds.candidate_cost(plan, dec, {"w_cut_full_both": 0.2}, ctx)
    assert c1 == pytest.approx(c0 + 0.2)
    assert ds.candidate_cost(plan, dec, {"w_cut_full_both": 0.2}, dict(ctx, b_full_at=False)) == pytest.approx(c0)
    blend = {"type": "blend", "type_key": "blend8", "bars": 8}
    assert ds.candidate_cost(plan, blend, {"w_cut_full_both": 0.2}, ctx) == pytest.approx(ds.candidate_cost(plan, blend, None, ctx))


# ------------------------------------------------------------------ F5: syllable onsets from the vocal stem (Ohriner p. 187)
def _syllable_stem(onsets, dur=12.0):
    x = np.zeros(int(dur * SR))
    rng = np.random.default_rng(0)
    for s in onsets:
        n = int(0.11 * SR)
        tt = np.arange(n) / SR
        env = np.minimum(tt / 0.02, 1) * np.exp(-np.maximum(tt - 0.02, 0) / 0.04)
        sig = sum(np.sin(2 * np.pi * f * tt) / k for k, f in enumerate([220, 440, 660, 880], 1)) * env
        i = int(s * SR)
        x[i : i + n] += sig
    return x + rng.normal(0, 0.002, len(x))


def test_stem_onsets_are_half_amplitude_pcentres():
    true = np.arange(1.0, 10.0, 0.25)  # sixteenths at 60 BPM
    r = P.stem_syllable_onsets(_syllable_stem(true), SR)
    on = r["onsets_s"]
    assert len(on) == len(true)
    err = np.abs(on - (true + 0.010))  # linear 20 ms attack: half amplitude 10 ms after the start
    assert np.median(err) < 0.005


def test_with_stem_onsets_snaps_uniform_syllables_and_finds_the_lag():
    per = 1.0  # 60 BPM: a sixteenth = 0.25 s
    lag = 0.05  # the voice is 50 ms late (Ohriner pp. 190-192: emcees sit behind the beat)
    true = np.arange(1.0, 10.0, 0.25) + lag
    stem = _syllable_stem(true)
    words = []
    for k in range(0, len(true) - 1, 2):  # two-syllable words; ASR onset 40 ms early, inner syllable badly placed
        on = int((true[k] - 0.04) * SR)
        words.append({"on": on, "off": int((true[k + 1] + 0.15) * SR), "last_syl_on": int((true[k + 1] + 0.07) * SR),
                      "n_syl": 2, "stress_idx": 0, "clean": "word", "text": "word"})
    model = {"sample_rate": SR, "period_s": per, "downbeats_s": list(np.arange(0.0, 12.0, 4 * per)), "words": words, "lines": []}
    m2, info = P.with_stem_onsets(model, stem, SR)
    assert info["snapped_share"] == 1.0
    sy = P.model_syllables(m2)
    assert np.allclose([s["t_s"] for s in sy], true[: len(sy)] + 0.010, atol=0.006)
    ph = P.vocal_phase_alignment(m2)
    assert ph["onsets"].startswith("vocal stem") and ph["phase_ms"] == pytest.approx(60.0, abs=17.0)  # lag + 10 ms p-centre
    assert model["words"][0].get("syl_on") is None  # input model untouched
    raw = P.vocal_phase_alignment(model, onsets_s=P.stem_syllable_onsets(stem, SR)["onsets_s"])
    assert raw["phase_ms"] == pytest.approx(60.0, abs=17.0)
