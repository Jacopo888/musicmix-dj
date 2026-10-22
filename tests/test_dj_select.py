import numpy as np
import pytest

from musicmix import dj_select as djs


# ------------------------------------------------------------------ helpers
def _grid(bpm, n, phase=0.3, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    p = 60.0 / bpm
    t = phase + p * np.arange(n)
    return t + rng.normal(0.0, jitter, n)


def _track(slug, n_bars, sections, vocal, bpm=90.0, camelot="8A", key_conf=0.9, energy=None):
    """Synthetic analysed track: one downbeat every bar, sections as (bar, label)."""
    bar = 4 * 60.0 / bpm
    downbeats = np.arange(n_bars) * bar
    secs = [{"label": lab, "start_s": b * bar, "end_s": 0.0, "bar_index": b, "downbeat_s": b * bar,
             "snap_error_bars": 0.0, "loose": False, "support": ["x"], "source": "test"}
            for b, lab in sections]
    vocal = np.asarray(vocal, dtype=float)
    return {
        "slug": slug, "duration_s": n_bars * bar,
        "tempo": {"bpm": bpm, "confidence": "high"},
        "key": {"camelot": camelot, "confidence": key_conf},
        "sections": secs,
        "phrases": djs.phrase_table(downbeats, secs),
        "bars": {"vocal": vocal.tolist(), "vocal_last_beat": vocal.tolist(),
                 "energy_db_rel": (np.zeros(n_bars) if energy is None else np.asarray(energy)).tolist()},
    }


# -------------------------------------------------------------------- grids
def test_fit_constant_grid_recovers_tempo_and_bar_phase():
    beats = _grid(92.0, 300, phase=0.41, jitter=0.01)
    down = beats[2::4]                       # bar starts on the third detected beat
    fit = djs.fit_constant_grid(beats, down)
    assert fit["bpm"] == pytest.approx(92.0, rel=0.002)
    assert fit["bar_offset_beats"] == 2
    assert fit["beat_f"] > 0.95 and fit["downbeat_f"] > 0.95


def test_downbeats_resolve_noisy_double_time_beats():
    rng = np.random.default_rng(1)
    beats = _grid(90.0, 240, phase=0.2, jitter=0.008)
    half = beats[:-1] + np.diff(beats) / 2
    extra = half[rng.random(half.size) < 0.3]   # spurious double-time detections
    noisy = np.sort(np.concatenate([beats, extra]))
    fit = djs.fit_constant_grid(noisy, beats[::4])
    assert fit["bpm"] == pytest.approx(90.0, rel=0.005)


def test_grid_times_and_agreement():
    beats = _grid(100.0, 200, phase=0.5)
    fit = djs.fit_constant_grid(beats, beats[::4])
    b, d = djs.grid_times(fit, 130.0)
    assert np.allclose(np.diff(b), 0.6, atol=1e-6)
    assert len(d) == pytest.approx(len(b) / 4, abs=1)
    agr = djs.grid_agreement(fit, fit, 130.0)
    assert agr["same_tempo_octave"] and agr["downbeat_coincidence"] == 1.0


def test_choose_grid_confidence_from_agreement():
    beats = _grid(100.0, 200, phase=0.5)
    f1 = djs.fit_constant_grid(beats, beats[::4])
    f2 = djs.fit_constant_grid(beats + 0.01, beats[::4] + 0.01)
    f3 = djs.fit_constant_grid(beats, beats[1::4])   # same tempo, other bar phase
    assert djs.choose_grid({"a": f1, "b": f2}, 130.0)["confidence"] == "high"
    assert djs.choose_grid({"a": f1, "b": f3}, 130.0)["confidence"] == "medium"


# ------------------------------------------------------------------ phrases
def test_phrase_levels_anchor_on_section_starts():
    down = np.arange(40) * 2.0
    secs = djs.merge_section_sources(
        [{"start_s": 0.0, "end_s": 12.1, "label": "intro"}, {"start_s": 12.1, "end_s": 80.0, "label": "verse"}],
        {"other": [12.0]}, down, 80.0)
    assert [s["bar_index"] for s in secs] == [0, 6]
    assert secs[1]["support"] == ["other"]
    rows = djs.phrase_table(down, secs)
    lv = {r["bar_index"]: r["phrase_level"] for r in rows}
    assert lv[0] == 16 and lv[4] == 4 and lv[5] == 0
    assert lv[6] == 16 and lv[10] == 4 and lv[14] == 8 and lv[22] == 16 and lv[7] == 0
    assert rows[6]["section_start"] and rows[6]["section_label"] == "verse"


def test_unsupported_off_grid_boundary_keeps_label_but_not_phrase_grid():
    down = np.arange(40) * 2.0
    secs = djs.merge_section_sources(
        [{"start_s": 0.0, "end_s": 18.0, "label": "verse", "source": "p"},
         {"start_s": 18.0, "end_s": 80.0, "label": "chorus", "source": "p"}],
        {"p": [0.0, 18.0], "other": [0.0]}, down, 80.0)
    assert secs[1]["bar_index"] == 9 and secs[1]["support"] == []
    rows = djs.phrase_table(down, secs)
    assert rows[9]["section_label"] == "chorus" and rows[9]["phrase_level"] == 0
    assert rows[8]["phrase_level"] == 8 and rows[16]["phrase_level"] == 16


# --------------------------------------------------------------------- keys
@pytest.mark.parametrize("name,code", [("A minor", "8A"), ("C major", "8B"), ("Bb major", "6B"),
                                       ("Am", "8A"), ("C#m", "12A"), ("F# minor", "11A"), ("Eb major", "5B")])
def test_camelot_codes(name, code):
    assert djs.camelot(name) == code


@pytest.mark.parametrize("a,b,rel", [("8A", "8A", "same"), ("8A", "9A", "adjacent"), ("12A", "1A", "adjacent"),
                                     ("8A", "8B", "relative"), ("8A", "9B", "diagonal"), ("8A", "2A", "clash")])
def test_camelot_relation(a, b, rel):
    assert djs.camelot_relation(a, b)["relation"] == rel


def test_krumhansl_fallback_and_combination():
    chroma = np.roll(djs.KK_MAJOR, 7)        # G major profile
    assert djs.krumhansl_key(chroma)["key"] == "G major"
    comb = djs.combine_key_estimates({
        "madmom_cnn": {"key": "C minor", "prob": 0.7, "window_stability": 0.8},
        "essentia_edma": {"key": "C minor"}, "essentia_bgate": {"key": "Eb major"}, "krumhansl": {"key": "G minor"}})
    assert comb["camelot"] == "5A" and comb["source"] == "madmom_cnn"
    assert comb["agree_same_or_relative"] == ["essentia_edma", "essentia_bgate"]
    assert comb["confidence"] == pytest.approx(np.mean([0.7, 0.8, 2 / 3]))


# ----------------------------------------------------------- vocals/energy
def test_interval_coverage():
    cov = djs.interval_coverage([(1.0, 3.0), (5.0, 5.5)], np.array([0.0, 2.0, 4.0]), np.array([2.0, 4.0, 6.0]))
    assert np.allclose(cov, [0.5, 0.5, 0.25])


# ------------------------------------------------------------- pair gates
def test_tempo_folds_octaves_and_gates_stretch():
    t = djs.tempo_compatibility(70.0, 141.0)
    assert t["octave_factor"] == 2.0 and t["compatible"] and abs(t["stretch_b_pct"]) < 1
    assert not djs.tempo_compatibility(100.0, 115.0)["compatible"]
    assert djs.tempo_compatibility(100.0, 107.0)["compatible"]
    assert not djs.tempo_compatibility(100.0, 107.0)["long_blend_ok"]


def test_key_compatibility_flags_clash_and_confidence():
    ok = djs.key_compatibility({"camelot": "8A", "confidence": 0.9}, {"camelot": "9A", "confidence": 0.8})
    assert ok["compatible"] and ok["confident"]
    bad = djs.key_compatibility({"camelot": "8A", "confidence": 0.9}, {"camelot": "3B", "confidence": 0.3})
    assert not bad["compatible"] and not bad["confident"]


# ------------------------------------------------------------- cue plans
def test_blend_on_instrumental_intro_and_cues_on_phrase_starts():
    # A: verse with vocals, outro (bar 48-64) instrumental; B: 16-bar intro, then vocal verse
    va = np.r_[np.ones(48), np.zeros(16)]
    vb = np.r_[np.zeros(16), np.ones(48)]
    A = _track("a", 64, [(0, "verse"), (48, "outro")], va)
    B = _track("b", 64, [(0, "intro"), (16, "verse")], vb)
    res = djs.plan_pair(A, B)
    assert res["compatibility"]["compatible"]
    best = res["plans"][0]
    assert best["type"] == "blend" and best["blend_bars"] == 16
    assert best["a_out_bar"] == 48 and best["b_in_bar"] == 0
    for p in res["plans"]:
        assert p["a_out_phrase_level"] >= 4 and p["b_in_phrase_level"] >= 4


def test_both_vocal_gives_cut_or_ducked_short_blend():
    A = _track("a", 64, [(0, "verse"), (32, "chorus")], np.ones(64))
    B = _track("b", 64, [(0, "verse"), (32, "chorus")], np.ones(64))
    for p in djs.plan_pair(A, B, top_k=50)["plans"]:
        assert p["type"] == "cut"           # B starts singing on its first bar
    vb = np.r_[0.0, np.ones(63)]
    B2 = _track("b2", 64, [(0, "verse"), (32, "chorus")], vb)
    types = {p["type"] for p in djs.plan_pair(A, B2, top_k=200)["plans"] if p["b_in_bar"] == 0}
    assert types <= {"blend_duck", "cut"} and "blend_duck" in types


def test_incompatible_tempo_forces_cut_and_ranks_last():
    v = np.zeros(64)
    A = _track("a", 64, [(0, "intro"), (16, "verse")], v, bpm=100.0)
    B = _track("b", 64, [(0, "intro"), (16, "verse")], v, bpm=118.0)
    C = _track("c", 64, [(0, "intro"), (16, "verse")], v, bpm=101.0)
    ab, ac = djs.plan_pair(A, B), djs.plan_pair(A, C)
    assert ab["compatibility"]["status"] == "incompatible"
    assert all(p["type"] == "cut" for p in ab["plans"])
    assert [p["b"] for p in djs.rank_pairs([ab, ac])] == ["c", "b"]


def test_choose_grid_majority_octave_and_ambiguity():
    beats = _grid(70.0, 150, phase=0.5)
    fast = np.sort(np.concatenate([beats, beats[:-1] + np.diff(beats) / 2]))
    f70a = djs.fit_constant_grid(beats, beats[::4])
    f70b = djs.fit_constant_grid(beats + 0.005, beats[::4] + 0.005)
    f140 = djs.fit_constant_grid(fast, fast[::4])
    g = djs.choose_grid({"madmom": f140, "beat_this": f70a, "allin1": f70b}, 140.0)
    assert g["fit"]["bpm"] == pytest.approx(70.0, rel=0.005)
    assert g["octave_ambiguous"] and g["confidence"] == "medium"


def test_bars_before_first_section_count_back_from_it():
    down = np.arange(30) * 2.0
    secs = djs.merge_section_sources([{"start_s": 6.0, "end_s": 60.0, "label": "verse"}], {}, down, 60.0)
    rows = djs.phrase_table(down, secs)
    assert [r["phrase_level"] for r in rows[:4]] == [0, 0, 0, 16]


# ------------------------------------------------------------ cut vocal guard
def _vocal_tracks(last_beat):
    A = _track("a", 64, [(0, "verse"), (32, "chorus")], np.ones(64))
    A["bars"]["vocal_last_beat"] = list(last_beat)
    B = _track("b", 64, [(0, "verse"), (32, "chorus")], np.ones(64))
    return A, B


def test_cut_vocal_guard_turns_cut_into_echo_out_only_when_enabled():
    lb = np.zeros(64)
    lb[31] = 1.0  # A sings through the last beat before bar 32
    A, B = _vocal_tracks(lb)
    pair = djs.pair_compatibility(A, B)
    out = next(c for c in djs.cue_candidates(A, "out") if c["bar_index"] == 32)
    inn = next(c for c in djs.cue_candidates(B, "in") if c["bar_index"] == 0)
    plain = djs.plan_transition(A, B, out, inn, pair)
    guarded = djs.plan_transition(A, B, out, inn, pair, cut_vocal_guard=True)
    assert plain["type"] == "cut" and guarded["type"] == "echo_out"
    assert any("cut non ammesso" in r for r in guarded["reasons"])
    assert not djs.cut_vocal_clear(A, 32) and djs.cut_vocal_clear(A, 36)
    # default planner output is unchanged by the new option
    assert djs.plan_pair(A, B, top_k=50) == djs.plan_pair(A, B, top_k=50, cut_vocal_guard=False)
    assert all(p["type"] != "cut" or djs.cut_vocal_clear(A, p["a_out_bar"])
               for p in djs.plan_pair(A, B, top_k=500, cut_vocal_guard=True)["plans"])


def test_relocate_cut_nearest_clear_phrase_start():
    lb = np.zeros(64)
    lb[31] = lb[27] = 1.0  # bars 32 and 28 not clear; 36 is
    A, B = _vocal_tracks(lb)
    pair = djs.pair_compatibility(A, B)
    plan = djs.plan_pair(A, B, top_k=500)["plans"]
    plan = next(p for p in plan if p["a_out_bar"] == 32 and p["b_in_bar"] == 0)
    res = djs.relocate_cut(A, B, plan, pair)
    assert res["status"] == "moved" and res["to_bar"] == 36 and res["shift_bars"] == 4
    assert [t["bar_index"] for t in res["tried"]] == [28, 36]  # earlier first on equal distance
    assert res["plan"]["a_out_bar"] == 36 and res["plan"]["b_in_bar"] == 0
    assert djs.relocate_cut(A, B, plan, pair, is_clear=lambda c: False)["status"] == "none"
    assert djs.relocate_cut(A, B, plan, pair, is_clear=lambda c: True)["status"] == "clear"


def test_vocal_clear_window():
    t = np.arange(0, 10, 0.02) + 0.01
    act = (t > 4.0) & (t < 4.9)
    assert not djs.vocal_clear_window(t, act, 5.0, before_s=0.5)
    assert djs.vocal_clear_window(t, act, 5.0, before_s=0.05)
    assert not djs.vocal_clear_window(t, act, 3.95, before_s=0.0, after_s=0.15)  # a word starting right after the cut
