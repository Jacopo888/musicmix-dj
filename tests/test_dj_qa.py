"""Synthetic checks for the DJ render QA (dj_qa).

Two synthetic "tracks" at 120 BPM (kick on 1 and 3-and, snare on 2/4, hats on
eighths, a mid stab and a one-bar pad chord changing every bar, a plucked bass line that changes note
every bar) with different timbres, mixed like an 8-bar blend (B fades in
until the switch, A fades out after it, bass swap on the switch). The clip
follows the AGENTS.md rule: 10 s + 16 s transition + 10 s.

Variants: good render; B 40 ms late; no bass swap (double bass); B +6 dB.
The onset cross-correlation path runs always; the Beat This! path runs when
the project checkpoint is available.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("pyloudnorm")
pytest.importorskip("scipy")
from scipy.signal import butter, sosfiltfilt  # noqa: E402

from musicmix import dj_qa  # noqa: E402

SR = 48000
BEAT = 0.5
START, SWITCH, END, N_S = 10.0, 18.0, 26.0, 36.0


def _burst(n, tau):
    return np.exp(-np.arange(n) / (tau * SR))


def synth_track(n_s, *, shift_s=0.0, beat=BEAT, bass_hz=55.0, seed=0, stab_hz=600.0, kick_hz=55.0, extra_kick=False):
    rng = np.random.default_rng(seed)
    n = int(n_s * SR)
    x = np.zeros(n)
    hp = butter(2, 6000, btype="high", fs=SR, output="sos")

    def add(t, sig):
        i = int(round(t * SR))
        if i < 0 or i >= n:
            return
        m = min(len(sig), n - i)
        x[i : i + m] += sig[:m]

    k_len = int(0.25 * SR)
    tk = np.arange(k_len) / SR
    kick = 0.9 * _burst(k_len, 0.06) * np.sin(2 * np.pi * (kick_hz * tk + 80 * 0.01 * (1 - np.exp(-tk / 0.01))))
    snare = 0.35 * _burst(int(0.2 * SR), 0.04) * rng.standard_normal(int(0.2 * SR))
    hat = 0.15 * sosfiltfilt(hp, _burst(int(0.05 * SR), 0.008) * rng.standard_normal(int(0.05 * SR)))
    stab = 0.25 * _burst(int(0.3 * SR), 0.08) * np.sin(2 * np.pi * stab_hz * np.arange(int(0.3 * SR)) / SR)
    b_len = int(0.45 * SR)
    pad_t = np.arange(int(4 * BEAT * SR)) / SR
    pad_env = 0.06 * np.minimum(1.0, pad_t / 0.01) * np.exp(-pad_t / 1.5)
    for k in range(int(n_s / beat) + 2):
        t = k * beat + shift_s
        pos = k % 4
        if pos == 0:
            add(t, kick)
        if pos == 2:
            add(t + 0.25, 0.8 * kick)
        if pos in (1, 3):
            add(t, snare)
        if extra_kick and pos == 3:
            add(t + 0.25, 0.6 * kick)
        add(t, hat)
        add(t + 0.25, 0.6 * hat)
        if pos == 0:
            add(t, stab)
            root = stab_hz * (1.0, 4 / 3, 1.5, 9 / 8)[(k // 4) % 4]
            add(t, pad_env * sum(np.sin(2 * np.pi * root * r * pad_t) for r in (1.0, 1.26, 1.5)))
        f = bass_hz * (1.0, 1.5, 1.25, 1.333)[(k // 4) % 4]
        add(t, 0.35 * _burst(b_len, 0.25) * np.sin(2 * np.pi * f * np.arange(b_len) / SR))
    return np.stack([x, x], 1)


def make_render(*, b_shift_ms=0.0, b_beat=BEAT, bass_swap=True, b_gain_db=0.0, kind="blend", vocals=False, words=None):
    n = int(N_S * SR)
    a = synth_track(N_S, bass_hz=55.0, seed=1, stab_hz=600.0)
    b = synth_track(N_S, shift_s=b_shift_ms / 1000.0, beat=b_beat, bass_hz=73.4, seed=2, stab_hz=880.0, kick_hz=60.0, extra_kick=True)
    b *= 10 ** (b_gain_db / 20)
    t = np.arange(n)
    s, w, e = int(START * SR), int(SWITCH * SR), int(END * SR)
    if kind == "cut":
        s, w, e = int(10 * SR) - 480, int(10 * SR), int(10 * SR)
        n = int(20 * SR)
        a, b, t = a[:n], b[:n], t[:n]
        fade = np.clip((t - s) / (w - s), 0, 1)
        fa, fb = np.cos(0.5 * np.pi * fade), np.sin(0.5 * np.pi * fade)
        low_a = low_b = np.ones(n)
    else:
        v = np.clip((t - s) / (w - s), 0, 1)
        u = np.clip((t - w) / (e - w), 0, 1)
        fb = np.where(t < s, 0.0, np.sin(0.5 * np.pi * v))
        fa = np.where(t >= e, 0.0, np.cos(0.5 * np.pi * u))
        swap = int(0.02 * SR)
        low_b = np.clip((t - (w - swap)) / swap, 0, 1) if bass_swap else np.ones(n)
        low_a = 1.0 - low_b if bass_swap else np.ones(n)
    sos = butter(2, 200, btype="low", fs=SR, output="sos")
    la, lb = sosfiltfilt(sos, a, axis=0), sosfiltfilt(sos, b, axis=0)
    deck_a = (la * low_a[:, None] + (a - la)) * fa[:, None]
    deck_b = (lb * low_b[:, None] + (b - lb)) * fb[:, None]
    mix = deck_a + deck_b
    g = 10 ** (-3.0 / 20) / np.max(np.abs(mix))  # sample peak -3 dBFS, true peak well below -1 dBTP
    meta = {
        "schema_version": "synthetic", "kind": kind, "sample_rate": SR,
        "overlap": {"output_start_sample": s, "output_end_sample": e, "a_source_start_sample": s, "a_source_end_sample": e,
                    "b_source_start_sample": s, "b_source_end_sample": e},
        "switch": {"output_sample": w, "b_source_sample": w},
        "stretch": {"applied": False},
        "clip": {"length_samples": n},
    }
    decks = {"a": deck_a * g, "b": deck_b * g, "b_stretched_raw": b * g, "gains": {"fader_a": fa, "fader_b": fb}}
    out = {"audio": mix * g, "meta": meta, "decks": decks}
    if vocals:
        tt = np.arange(n) / SR
        voc = 0.2 * np.sin(2 * np.pi * 300 * tt)[:, None] * np.ones((1, 2))
        out["vocals_a"], out["vocals_b"] = voc, voc.copy()
    if words is not None:
        out["words_a"] = words
    return out


def run(r, tracker=None, **kw):
    return dj_qa.evaluate_render(r["audio"], r["meta"], decks=r.get("decks"), vocals_a=r.get("vocals_a"), vocals_b=r.get("vocals_b"),
                                 words_a=r.get("words_a"), tracker=tracker, **kw)


def failed(rep):
    return {x["check"] for x in rep["reasons"] if x["severity"] == "fail"}


@pytest.fixture(scope="module")
def renders():
    return {
        "good": make_render(),
        "late40": make_render(b_shift_ms=40.0),
        "double_bass": make_render(bass_swap=False),
        "loud6": make_render(b_gain_db=6.0),
    }


@pytest.fixture(scope="module")
def beat_this():
    pytest.importorskip("beat_this")
    if not dj_qa.default_beat_this_checkpoint().is_file():
        pytest.skip("Beat This! checkpoint not available")
    return dj_qa.BeatThisTracker()


def test_good_render_passes(renders):
    rep = run(renders["good"])
    assert rep["verdict"] == "pass", rep["reasons"]
    xc = rep["checks"]["rhythm"]["methods"]["onset_xcorr_decks"]
    assert xc["status"] == "pass" and xc["median_abs_lag_ms"] < 2.0
    assert rep["checks"]["double_bass"]["longest_s"] <= 0.1
    assert abs(rep["checks"]["loudness_jump"]["max_abs_jump_db"]) <= 2.0
    assert rep["checks"]["duration"]["status"] == "pass"
    json.dumps(rep)  # JSON-serialisable


def test_b_late_40ms_fails_rhythm(renders):
    rep = run(renders["late40"])
    assert failed(rep) == {"rhythm"}, rep["reasons"]
    xc = rep["checks"]["rhythm"]["methods"]["onset_xcorr_decks"]
    assert xc["median_lag_ms"] == pytest.approx(40.0, abs=3.0)


def test_tempo_drift_fails():
    # B's beat slightly longer, phase-aligned at the overlap start: +40 ms after 16 s
    beat_b = BEAT * (1 + 0.040 / (END - START))
    rep = run(make_render(b_shift_ms=-START * (beat_b - BEAT) / BEAT * 1000.0, b_beat=beat_b))
    xc = rep["checks"]["rhythm"]["methods"]["onset_xcorr_decks"]
    assert "rhythm" in failed(rep), rep["reasons"]
    assert xc["drift"]["ms"] == pytest.approx(40.0, abs=8.0) and xc["drift"]["significant"]


def test_double_bass_fails(renders):
    rep = run(renders["double_bass"])
    assert failed(rep) == {"double_bass"}, rep["reasons"]
    assert rep["checks"]["double_bass"]["longest_s"] > 0.1 and rep["checks"]["double_bass"]["total_s"] > 1.0


def test_loudness_jump_fails(renders):
    rep = run(renders["loud6"])
    assert "loudness_jump" in failed(rep), rep["reasons"]
    lj = rep["checks"]["loudness_jump"]
    base = run(renders["good"])["checks"]["loudness_jump"]["signed_jump_db"]
    assert lj["signed_jump_db"] - base == pytest.approx(6.0, abs=1.0)
    assert failed(rep) <= {"loudness_jump", "double_bass"}


def test_beat_this_path(renders, beat_this):
    good = run(renders["good"], tracker=beat_this)["checks"]["rhythm"]["methods"]["beat_this_decks"]
    late = run(renders["late40"], tracker=beat_this)["checks"]["rhythm"]["methods"]["beat_this_decks"]
    assert good["status"] == "pass", good
    assert late["status"] == "fail", late
    assert np.median(late["downbeat_errors_ms"]) == pytest.approx(40.0, abs=12.0)


def test_mix_only_without_decks(renders, beat_this):
    r = dict(renders["good"], decks=None)
    rep = run(r, tracker=beat_this)
    mo = rep["checks"]["rhythm"]["methods"]["beat_this_mix"]
    assert abs(mo["beat_phase_offset_at_switch_ms"]) < 15.0
    assert rep["checks"]["double_bass"]["status"] == "skipped"


def test_true_peak_fails():
    r = make_render()
    r = dict(r, audio=r["audio"] * 10 ** (3.5 / 20))  # sample peak +0.5 dBFS
    rep = run(r)
    assert "true_peak" in failed(rep)


def test_duration_rule():
    r = make_render()
    r = dict(r, audio=r["audio"][: -SR])
    rep = run(r)
    assert "duration" in failed(rep)


def test_vocal_clash_and_ducking():
    r = make_render(vocals=True)
    rep = run(r)
    assert "vocal_clash" in failed(rep)
    duck = np.ones(len(r["audio"]))
    duck[int(START * SR) :] = 10 ** (-30 / 20)  # A's vocal ducked for the whole overlap
    r["decks"] = dict(r["decks"], gains=dict(r["decks"]["gains"], duck_a_vocal=duck))
    assert run(r)["checks"]["vocal_clash"]["status"] == "pass"


def test_verse_cut_on_cut():
    words = [{"start_s": 9.7, "end_s": 10.05, "text": "parola"}]
    r = make_render(kind="cut", words=words)
    rep = run(r)
    assert "verse_cut" in failed(rep), rep["reasons"]
    assert rep["checks"]["duration"]["status"] == "pass"
    r["words_a"] = [{"start_s": 9.2, "end_s": 9.6, "text": "prima"}]
    assert run(r)["checks"]["verse_cut"]["status"] == "pass"
    # dj_qa.4 exempted any exit effect without measuring ("pass"); from .5 the dry stop is checked (see
    # test_effect_exit_word_cut_*); the old behaviour is only reachable explicitly and is reported as not applicable.
    r["meta"] = dict(r["meta"], effect={"exit": "echo_out"})
    r["words_a"] = words
    old = run(r, config=dj_qa.QAConfig(effect_word_cut="off"))["checks"]["verse_cut"]
    assert old["status"] == "not_applicable" and old["coverage"] == "not_applicable"


def test_b_source_position_anchored_on_switch_for_effects():
    # echo_out / filter_out (dj_effects): the overlap starts before B's entry, while
    # overlap.b_source_start_sample is B's entry sample, which plays at switch.output_sample.
    meta = {
        "overlap": {"output_start_sample": 480000, "output_end_sample": 560000,
                    "a_source_start_sample": 1000000, "b_source_start_sample": 300000, "b_source_end_sample": 300000},
        "switch": {"output_sample": 560000, "b_source_sample": 300000},
        "stretch": {"applied": False},
    }
    pos = dj_qa.b_source_position(meta, 600000)
    assert pos[560000] == 300000 and pos[480000] == 300000 - 80000
    # dj_render cut: overlap start and switch anchors agree
    cut = {"overlap": {"output_start_sample": 479520, "b_source_start_sample": 299520},
           "switch": {"output_sample": 480000, "b_source_sample": 300000}, "stretch": {"applied": False}}
    assert dj_qa.b_source_position(cut, 500000)[480000] == 300000


def test_drop_in_is_treated_as_cut():
    meta = {"kind": "drop_in", "base_kind": "cut", "sample_rate": SR, "effect": {"type": "drop_in"},
            "overlap": {"output_start_sample": 10 * SR - 480, "output_end_sample": 10 * SR},
            "switch": {"output_sample": 10 * SR}}
    geo = dj_qa.transition_geometry(meta, 20 * SR)
    assert geo["is_cut"]
    assert dj_qa.check_duration(20 * SR, geo, meta, dj_qa.QAConfig())["status"] == "pass"


def test_blend_fader_exit_is_not_a_verse_cut():
    # a word still sung while A's fader closes at the end of the overlap: gradual exit, not a cut.
    # dj_qa.5: "not applicable" is its own state, no longer reported as a measured pass.
    r = make_render(words=[{"start_s": END - 0.5, "end_s": END + 0.5, "text": "lunga"}])
    vc = run(r)["checks"]["verse_cut"]
    assert vc["status"] == "not_applicable" and vc["coverage"] == "not_applicable"


# ------------------------------------------------- coverage and effect exits (P0.6, 21/10/2026)


def test_every_check_has_a_coverage_state(renders):
    rep = run(renders["good"])
    assert set(rep["coverage"]["checks"]) == set(rep["checks"])
    assert all(c["coverage"] in dj_qa.COVERAGE_STATES for c in rep["checks"].values())
    assert rep["checks"]["vocal_clash"]["coverage"] == "unavailable"  # no stems given
    assert rep["coverage"]["complete"]  # B not beatmatched in this synthetic meta: vocal clash not required


def test_missing_required_check_is_not_a_pass(renders):
    r = dict(renders["good"], meta=dict(renders["good"]["meta"], stretch={"applied": True, "tempo_ratio_b": 1.0}))
    rep = run(r)  # two tracks overlap, no vocal stems: vocal_clash required but unavailable
    assert rep["coverage"]["missing_required"] == ["vocal_clash"]
    assert rep["verdict"] == "warn" and rep["reasons"][-1]["check"] == "coverage"
    legacy = run(r, config=dj_qa.QAConfig(require_coverage=False))
    assert legacy["verdict"] == "pass"


def _echo_exit_render(stop_s=10.5, words=None, vocals=False):
    """A dry until ``stop_s`` (echo_out metadata), then silence in deck A; B from 12 s."""
    n = int(22 * SR)
    a = synth_track(22, bass_hz=55.0, seed=1, stab_hz=600.0)
    b = synth_track(22, bass_hz=73.4, seed=2, stab_hz=880.0, kick_hz=60.0)
    stop, entry = int(stop_s * SR), int(12 * SR)
    deck_a = a.copy()
    deck_a[stop:] = 0.0
    deck_b = np.zeros_like(b)
    deck_b[entry:] = b[: n - entry]
    mix = (deck_a + deck_b) * 0.3
    meta = {"schema_version": "synthetic", "kind": "echo_out", "sample_rate": SR,
            "effect": {"type": "echo_out", "dry_a_cut_output_sample": stop},
            "overlap": {"output_start_sample": 10 * SR, "output_end_sample": entry, "a_source_start_sample": 10 * SR,
                        "b_source_start_sample": 0, "b_source_end_sample": 0},
            "switch": {"output_sample": entry, "b_source_sample": 0}, "stretch": {"applied": False}, "clip": {"length_samples": n}}
    r = {"audio": mix, "meta": meta, "decks": {"a": deck_a * 0.3, "b": deck_b * 0.3, "gains": {}}}
    if words is not None:
        r["words_a"] = words
    if vocals:  # A's source vocal: a sustained tone from 10.2 s to 11.0 s (continues past the stop in the source)
        v = np.zeros((n, 2))
        i0, i1 = int(10.2 * SR), int(11.0 * SR)
        v[i0:i1] = 0.2 * np.sin(2 * np.pi * 300 * np.arange(i1 - i0) / SR)[:, None]
        r["vocals_a"], r["vocals_b"] = v, np.zeros((n, 2))
    return r


def test_effect_exit_word_cut_by_the_dry_stop():
    cut = run(_echo_exit_render(words=[{"start_s": 10.3, "end_s": 10.8, "text": "tagliata"}]))
    vc = cut["checks"]["verse_cut"]
    assert vc["status"] == "fail" and vc["coverage"] == "measured" and "verse_cut" in failed(cut)
    assert vc["a_dry_end_s"] == pytest.approx(10.5) and vc["a_dry_end_source"].startswith("effect.dry_a_cut_output_sample")
    ok = run(_echo_exit_render(words=[{"start_s": 10.0, "end_s": 10.48, "text": "finita"}]))["checks"]["verse_cut"]
    assert ok["status"] == "pass"  # ends within the ASR tolerance before the stop: the echo carries the tail
    warn = run(_echo_exit_render(words=[{"start_s": 10.3, "end_s": 10.8, "text": "x"}]), config=dj_qa.QAConfig(effect_word_cut="warn"))
    assert warn["checks"]["verse_cut"]["status"] == "warn"


def test_effect_exit_word_cut_from_the_stem():
    vc = run(_echo_exit_render(vocals=True))["checks"]["verse_cut"]
    assert vc["status"] == "fail" and vc["stem_active_before"] and vc["stem_active_after"]
    vc = run(_echo_exit_render(stop_s=11.3, vocals=True))["checks"]["verse_cut"]
    assert vc["status"] == "pass"


def test_loop_exit_from_the_a_map_and_unclipped_words():
    # creative-like map: A continuous until 10 s, then a 1-bar loop restarting at an earlier source sample
    r = make_render()
    meta = dict(r["meta"], kind="loop_extend", exit_effect=None, a_map=[
        {"out_start": 0, "out_end": 10 * SR, "src_start": 0, "rate": 1.0, "vocals": True},
        {"out_start": 10 * SR, "out_end": 12 * SR, "src_start": 8 * SR, "rate": 1.0, "vocals": True},
        {"out_start": 12 * SR, "out_end": 14 * SR, "src_start": 8 * SR, "rate": 1.0, "vocals": True}])
    assert dj_qa.a_dry_end(meta)["sample"] == 10 * SR
    word = {"start_s": 9.8, "end_s": 10.0, "end_unclipped_s": 10.3, "text": "loop"}  # clipped by the map at 10 s
    rep = run(dict(r, meta=meta, words_a=[word]))
    assert rep["checks"]["verse_cut"]["status"] == "fail" and rep["checks"]["verse_cut"]["a_dry_end_s"] == pytest.approx(10.0)
    rep = run(dict(r, meta=meta, words_a=[dict(word, end_unclipped_s=10.0)]))
    assert rep["checks"]["verse_cut"]["status"] == "pass"


# ------------------------------------------------- tracker phase ambiguity (MAFIA, v1)


def _tempo(provider="allin1", others=("madmom",), coincidence=1.0, beat_this=0.0):
    agree = {o: {"same_tempo_octave": True, "beat_coincidence": coincidence, "downbeat_coincidence": coincidence} for o in others}
    agree["beat_this"] = {"same_tempo_octave": True, "beat_coincidence": beat_this, "downbeat_coincidence": beat_this}
    return {"provider": provider, "agreement": agree}


def _shifted_tracker(shift_s, from_s=None):
    """Fake tracker: beats every 0.5 s; from ``from_s`` on (or always) shifted by ``shift_s``."""
    def track(audio, sr):
        n = len(audio) / sr
        t = np.arange(0.0, n, BEAT)
        if from_s is None:
            t = t + shift_s
        else:
            t = np.where(t >= from_s, t + shift_s, t)
        t = t[(t >= 0) & (t < n)]
        return {"beats": t, "downbeats": t[::4]}
    return track


def test_grid_phase_support():
    s = dj_qa.grid_phase_support(_tempo())
    assert s["supported"] and s["agreeing_providers"] == ["madmom"] and s["beat_this_off_grid"]
    assert not dj_qa.grid_phase_support(_tempo(others=()))["supported"]  # allin1 grid, no independent confirmation
    assert not dj_qa.grid_phase_support(_tempo(coincidence=0.5))["supported"]
    assert dj_qa.grid_phase_support(_tempo(provider="beat_this", others=("allin1",)))["agreeing_providers"] == ["allin1"]
    assert not dj_qa.grid_phase_support(None)["supported"]


def test_half_beat_mix_only_error_is_a_tracker_phase_warn():
    # dj_qa.7: a clean entry (cut) is measured by B's phase against A's pulse; the measure counts only when the
    # onset-flux detector agrees with the tracker.
    r = dict(make_render(kind="cut"), decks=None)
    trk = _shifted_tracker(0.5 * BEAT, from_s=10.0)  # tracker on the off-beats, audio on the grid: detectors disagree
    rh = run(r, tracker=trk)["checks"]["rhythm"]
    assert rh["status"] == "warn" and rh["coverage"] == "unavailable" and "detectors disagree" in rh["reason"]
    r = dict(make_render(kind="cut", b_shift_ms=0.5 * BEAT * 1000), decks=None)  # B really half a beat late
    rep = run(r, tracker=trk)
    assert "rhythm" in failed(rep), rep["reasons"]
    en = rep["checks"]["rhythm"]["methods"]["beat_this_entry"]
    assert en["reliable"] and abs(abs(en["beat_phase_offset_at_switch_ms"]) - 250.0) < 5.0
    support = {"a": _tempo(beat_this=1.0), "b": _tempo()}  # B's grid confirmed by madmom, Beat This! off it in the analysis
    rep = run(r, tracker=trk, grid_support=support)
    rh = rep["checks"]["rhythm"]
    assert rh["status"] == "warn" and "rhythm" not in failed(rep)
    assert "possibile ambiguità di fase del tracker" in rh["reason"]
    amb = rh["methods"]["beat_this_entry"]["phase_ambiguity"]
    assert amb["half_beat_ms"] == pytest.approx(250.0) and amb["beat_this_off_grid_in_analysis"] == ["B"]
    # only one deck confirmed by allin1/madmom: still a fail
    assert "rhythm" in failed(run(r, tracker=trk, grid_support={"a": _tempo(others=()), "b": _tempo()}))
    # Beat This! agreeing with both grids in the analyses: a half-beat error on the clip is a real error (gym v1 §4.1)
    assert "rhythm" in failed(run(r, tracker=trk, grid_support={"a": _tempo(beat_this=1.0), "b": _tempo(beat_this=1.0)}))
    # an error far from half a beat stays a fail even with confirmed grids
    r100 = dict(make_render(kind="cut", b_shift_ms=100.0), decks=None)
    assert "rhythm" in failed(run(r100, tracker=_shifted_tracker(0.100, from_s=10.0), grid_support=support))


def test_entry_phase_natural_noise_passes():
    r = dict(make_render(kind="cut"), decks=None)
    rep = run(r, tracker=_shifted_tracker(0.0))
    en = rep["checks"]["rhythm"]["methods"]["beat_this_entry"]
    assert en["reliable"] and abs(en["beat_phase_offset_at_switch_ms"]) < 5.0 and rep["checks"]["rhythm"]["status"] == "pass"


def test_half_beat_deck_downbeats_are_a_tracker_phase_warn(renders):
    calls = []

    def trk(audio, sr):  # first call: deck A on the grid; second: deck B on the off-beats
        calls.append(1)
        return _shifted_tracker(0.0 if len(calls) % 2 == 1 else 0.245)(audio, sr)

    r = renders["good"]
    bt = run(r, tracker=trk)["checks"]["rhythm"]["methods"]["beat_this_decks"]
    assert bt["status"] == "fail" and bt["downbeat_median_abs_ms"] == pytest.approx(245.0, abs=1.0)
    rep = run(r, tracker=trk, grid_support={"a": _tempo(), "b": _tempo()})
    bt = rep["checks"]["rhythm"]["methods"]["beat_this_decks"]
    assert bt["status"] == "warn" and "possibile ambiguità di fase del tracker" in bt["reason"]
    assert rep["checks"]["rhythm"]["status"] == "warn" and rep["inputs"]["grid_support"]["b"]["supported"]


def test_section_gain_curve_counts_in_b_vocal_audibility():
    r = make_render(vocals=True)
    n = len(r["audio"])
    gains = dict(r["decks"]["gains"], section_gain_b=np.full(n, 0.5))
    ga, gb, note = dj_qa.deck_gains(r["meta"], n, dict(r["decks"], gains=gains))
    assert np.allclose(gb, 0.5 * np.asarray(r["decks"]["gains"]["fader_b"])) and "section gain" in note


# ------------------------------------- loudness windows for effects (21/10/2026)


def make_effect_render(effect="filter_out", *, b_gain_db=0.0, region_gain_db=None, effect_s=4.0):
    """A full until the effect start (10 s), effect on A until the switch, B clean from the switch, 10 s of B.

    filter_out: A high-passed at 1.5 kHz with a fader dip to -9 dB; echo_out: A's dry signal cut at the
    effect start, its last beat repeated each beat with feedback 0.5. Both are quieter by construction.
    ``region_gain_db`` multiplies the effect region (a bump if > 0).
    """
    s, w = int(START * SR), int((START + effect_s) * SR)
    n = w + int(10 * SR)
    a = synth_track(n / SR, bass_hz=55.0, seed=1, stab_hz=600.0)[:n]
    b = synth_track(n / SR, bass_hz=73.4, seed=2, stab_hz=880.0, kick_hz=60.0, extra_kick=True)[:n] * 10 ** (b_gain_db / 20)
    region = np.zeros((w - s, 2))
    if effect == "filter_out":
        hp = butter(4, 1500, btype="high", fs=SR, output="sos")
        dip = 10 ** (np.linspace(0.0, -9.0, w - s) / 20)[:, None]
        region = sosfiltfilt(hp, a, axis=0)[s:w] * dip
    else:
        beat = int(BEAT * SR)
        frag = a[s - beat : s]
        for k in range(int(np.ceil((w - s) / beat))):
            i0 = k * beat
            m = min(beat, (w - s) - i0)
            region[i0 : i0 + m] = 0.5 ** (k + 1) * frag[:m]
    if region_gain_db is not None:
        region = region * 10 ** (region_gain_db / 20)
    mix = np.zeros((n, 2))
    mix[:s] = a[:s]
    mix[s:w] = region
    mix[w:] = b[w - s : n - s]  # B from its own start (a downbeat) at the switch
    mix *= 10 ** (-3.0 / 20) / np.max(np.abs(mix))
    meta = {"schema_version": "synthetic", "kind": effect, "sample_rate": SR, "effect": {"type": effect},
            "overlap": {"output_start_sample": s, "output_end_sample": w, "a_source_start_sample": s, "b_source_start_sample": 0},
            "switch": {"output_sample": w, "b_source_sample": 0}, "stretch": {"applied": False}, "clip": {"length_samples": n}}
    return {"audio": mix, "meta": meta}


@pytest.mark.parametrize("effect", ["filter_out", "echo_out"])
def test_effect_dip_is_not_a_loudness_jump(effect):
    rep = run(make_effect_render(effect))
    lj = rep["checks"]["loudness_jump"]
    assert lj["status"] == "pass", lj
    assert lj["worst_window"] == "a_before_vs_b_after" and lj["max_abs_jump_db"] <= 2.0
    # the old window around the switch falls inside the effect: a big "jump" that is only the expected dip
    sw = lj["windows"]["switch"]
    assert sw["used_for_verdict"] is False and sw["jump_db"] > 3.0
    assert lj["region_peak"]["status"] == "pass" and lj["region_peak"]["excess_db"] < 0
    assert "loudness_jump" not in failed(rep) and rep["checks"]["duration"]["status"] == "pass"


@pytest.mark.parametrize("effect", ["filter_out", "echo_out"])
def test_effect_real_jump_between_a_before_and_b_after_fails(effect):
    lj = run(make_effect_render(effect, b_gain_db=6.0))["checks"]["loudness_jump"]
    assert lj["status"] == "fail" and lj["worst_window"] == "a_before_vs_b_after"
    base = run(make_effect_render(effect))["checks"]["loudness_jump"]["signed_jump_db"]
    assert lj["signed_jump_db"] - base == pytest.approx(6.0, abs=1.0)


def test_level_bump_inside_effect_fails():
    lj = run(make_effect_render("filter_out", region_gain_db=12.0))["checks"]["loudness_jump"]
    assert lj["region_peak"]["status"] == "fail" and lj["region_peak"]["excess_db"] > 3.0
    assert lj["status"] == "fail" and "level bump inside the effect" in lj["reason"]


def test_cut_keeps_switch_window(renders):
    lj = run(make_render(kind="cut", b_gain_db=6.0))["checks"]["loudness_jump"]
    assert lj["worst_window"] == "switch" and lj["status"] == "fail" and "region_peak" not in lj["windows"]
    blend = run(renders["good"])["checks"]["loudness_jump"]
    assert blend["windows"]["switch"]["used_for_verdict"] is False and blend["region_peak"]["status"] == "pass"


def test_quiet_voice_band_only_discounts_a_crossing_word():
    r = _echo_exit_render(words=[{"start_s": 10.0, "end_s": 10.4, "text": "prima"}])
    r["decks"]["a"][int(9.9 * SR) : int(10.5 * SR)] *= 0.01  # A 40 dB down before its dry stop
    vc = run(r)["checks"]["verse_cut"]
    assert vc["status"] == "pass" and vc["coverage"] == "measured"  # no word crosses: a measured pass, not "not applicable"
    r["words_a"] = [{"start_s": 10.3, "end_s": 10.8, "text": "x"}]
    vc = run(r)["checks"]["verse_cut"]
    assert vc["status"] == "not_applicable" and vc["words_crossing"]
