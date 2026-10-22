"""Profilo "libri" (2026-10-21) di dj_set e dj_policy: regole dai manuali DJ, default invariati."""

from __future__ import annotations

import math

import numpy as np
import pytest

from musicmix import dj_policy as pol
from musicmix import dj_select as djs
from musicmix import dj_set as ds

BOOKS = ds.profile("books")


def _ctx(**kw):
    base = {
        "a": "A", "b": "B", "a_out_bar": 40, "b_in_bar": 0, "a_out_s": 100.0, "b_in_s": 0.0,
        "bpm_a": 90.0, "bpm_b": 90.0, "stretch_abs": 0.0, "octave_factor": 1.0,
        "tempo_confidence_a": "high", "tempo_confidence_b": "high",
        "key_relation": "same", "key_confident": True, "camelot_a": "8A", "camelot_b": "8A",
        "a_vocal_last_beat": 0.0, "a_vocal_last_bar": 0.0,
        "a_vocal_bars": [0.0] * 16, "b_vocal_bars": [0.0] * 16,
        "a_bars_before": 40, "a_bars_after": 24, "b_bars_after": 60,
        "a_exit_on_boundary": True, "a_exit_section": "chorus",
        "b_section": "verse", "b_section_start": True,
        "a_phrase_level": 16, "b_phrase_level": 16,
        "energy_a_db_rel": 0.0, "energy_b_db_rel": 0.0,
    }
    base.update(kw)
    return base


def _track(slug, n_bars, sections, vocal, bpm=90.0):
    bar = 4 * 60.0 / bpm
    downbeats = np.arange(n_bars) * bar
    secs = [{"label": lab, "start_s": b * bar, "end_s": 0.0, "bar_index": b, "downbeat_s": b * bar,
             "snap_error_bars": 0.0, "loose": False, "support": ["x"], "source": "test"} for b, lab in sections]
    vocal = np.asarray(vocal, dtype=float)
    return {"slug": slug, "duration_s": n_bars * bar,
            "tempo": {"bpm": bpm, "confidence": "high", "period_s": 60.0 / bpm},
            "key": {"camelot": "8A", "confidence": 0.9}, "sections": secs,
            "phrases": djs.phrase_table(downbeats, secs),
            "bars": {"vocal": vocal.tolist(), "vocal_last_beat": vocal.tolist(), "energy_db_rel": np.zeros(n_bars).tolist()}}


# ------------------------------------------------------------------ profile
def test_profile_is_valid_and_defaults_untouched():
    for level, defaults in (("tree", ds.TREE_PARAMS), ("order", ds.ORDER_PARAMS), ("cues", ds.CUE_PARAMS)):
        ds.values(defaults, BOOKS[level])  # no unknown key
        for k in BOOKS[level]:
            assert defaults[k].source  # every overridden parameter has a documented source
    assert ds.profile(None) == {"tree": {}, "order": {}, "cues": {}}
    with pytest.raises(KeyError):
        ds.profile("nope")
    assert ds.values(ds.CUE_PARAMS)["dur_mode"] == "seconds"
    assert ds.values(ds.ORDER_PARAMS)["arc_shape"] == "single"
    for k in ("hiphop_cut", "key_clash_cut", "backspin_far_tempo"):
        assert ds.values(ds.TREE_PARAMS)[k] is False
    for k in ("fx_budget_penalty", "backspin_context_bonus"):
        assert pol.PARAMS[k][0] == 0.0


# --------------------------------------------------------------------- tree
def test_hiphop_cut_after_a_finished_chorus_only_with_profile():
    voc = [1.0] * 16
    b = [0.0] + [1.0] * 15          # B: first bar free (on the one), then raps
    c = _ctx(a_vocal_bars=voc, b_vocal_bars=b)
    d0 = ds.decide_transition(c)
    assert d0["rule"] == "R6_both_vocal" and d0["type"] == "filter_out"
    d1 = ds.decide_transition(c, BOOKS["tree"])
    assert d1["rule"] == "R6b_hiphop_cut" and d1["type"] == "cut"
    assert [f["type"] for f in d1["fallbacks"]][:1] == ["echo_out"]
    # A mid-section, or B raps on the one, or B enters on a chorus: no cut
    for kw in ({"a_exit_on_boundary": False}, {"b_vocal_bars": voc}, {"a_exit_section": "intro"}):
        d = ds.decide_transition(_ctx(**{"a_vocal_bars": voc, "b_vocal_bars": b, **kw}), BOOKS["tree"])
        assert d["type"] != "cut"
    d = ds.decide_transition(_ctx(a_vocal_bars=voc, b_vocal_bars=b, b_section="chorus"), BOOKS["tree"])
    assert d["rule"] == "R3_chorus_entry" and "drop" not in d["type"]


def test_key_clash_cut_instead_of_effect():
    c = _ctx(key_relation="clash")
    assert ds.decide_transition(c)["type"] in ("filter_out", "echo_out")
    d = ds.decide_transition(c, BOOKS["tree"])
    assert d["rule"] == "R4b_key_cut" and d["type"] == "cut"
    d = ds.decide_transition(_ctx(key_relation="clash", a_vocal_last_beat=0.8, a_vocal_last_bar=0.9), BOOKS["tree"])
    assert d["rule"] == "R4_key_no_blend" and d["type"] == "echo_out"


def test_backspin_for_far_tempo_at_phrase_end_into_strong_start():
    c = _ctx(stretch_abs=0.12, b_vocal_bars=[1.0] * 16)       # cut not admissible: B raps on the one
    assert ds.decide_transition(c)["type"] == "echo_out"
    d = ds.decide_transition(c, BOOKS["tree"])
    assert d["rule"] == "R2b_backspin" and d["type"] == "backspin" and d["render"]["spin_beats"] == 2
    assert d["fallbacks"][0]["type"] == "echo_out"
    assert ds.TYPE_COST["backspin"].value > 0
    # B much calmer, or B on a chorus: no backspin
    assert ds.decide_transition(_ctx(stretch_abs=0.12, b_vocal_bars=[1.0] * 16, energy_b_db_rel=-4.0), BOOKS["tree"])["type"] == "echo_out"
    assert ds.decide_transition(_ctx(stretch_abs=0.12, b_section="chorus"), BOOKS["tree"])["type"] == "echo_out"
    # lead of a backspin = spin beats of A
    A = {"tempo": {"period_s": 0.6}}
    assert ds.lead_a_s({"decision": d}, A) == pytest.approx(1.2)


def test_hiphop_blends_are_short():
    assert ds.decide_transition(_ctx(a_exit_section="verse", b_section="intro"))["bars"] == 16
    d = ds.decide_transition(_ctx(a_exit_section="verse", b_section="intro"), BOOKS["tree"])
    assert d["type"] == "blend" and d["bars"] == 8


# ---------------------------------------------------------------- ordering
def test_waves_arc_has_two_peaks_last_highest():
    o = BOOKS["order"]
    L = 10
    t = [ds.arc_target(i, L, o) for i in range(L)]
    peaks = [i for i in range(1, L - 1) if t[i] > t[i - 1] and t[i] >= t[i + 1]]
    assert len(peaks) == 2 and t[peaks[1]] > t[peaks[0]]
    assert t[0] < t[peaks[0]] and t[-1] < t[peaks[1]]
    assert [ds.arc_target(i, 4, o) for i in range(4)] == [ds.arc_target(i, 4) for i in range(4)]  # short set: single
    assert ds.arc_target(0, 10) == pytest.approx(0.35)  # default unchanged


def test_tempo_down_and_type_override_and_placement():
    o = BOOKS["order"]
    assert ds.tempo_down_cost(90, 85) == 0.0
    assert ds.tempo_down_cost(90, 85, o) > 0 and ds.tempo_down_cost(85, 90, o) == 0.0
    assert ds.tempo_down_cost(140, 72, o) == pytest.approx(0.0)  # 72 folded = 144: faster
    cut = {"type_key": "cut"}
    assert ds.type_cost(cut) == 0.5 and ds.type_cost(cut, o["type_cost_override"]) == 0.2
    plan = {"cue_score": 5.0, "a_out_phrase_level": 4, "b_in_phrase_level": 16}
    mid = _ctx(a_exit_on_boundary=False, b_section_start=False)
    assert sum(ds.placement_cost(plan, mid).values()) == 0.0
    pc = ds.placement_cost(plan, mid, o)
    assert pc["exit_midsection"] > 0 and pc["entry_midsection"] > 0 and pc["phrase4"] > 0
    after_chorus = ds.placement_cost(plan, _ctx(), o)
    after_verse = ds.placement_cost(plan, _ctx(a_exit_section="verse"), o)
    assert after_verse["exit_after_verse"] > after_chorus["exit_after_verse"] == 0.0
    dec = {"type_key": "filter_out"}
    assert ds.candidate_cost(plan, dec, o, mid) > ds.candidate_cost(plan, dec, o, _ctx())


# ------------------------------------------------------------- whole sections
SECS = [(0, "intro"), (8, "verse"), (24, "chorus"), (32, "verse"), (48, "chorus"), (56, "outro")]
VOC = [0.0] * 8 + [1.0] * 48 + [0.0] * 16


def test_section_spans_and_played():
    T = _track("t", 72, SECS, VOC)
    assert ds.section_spans(T) == [(0, 8, "intro"), (8, 24, "verse"), (24, 32, "chorus"), (32, 48, "verse"),
                                   (48, 56, "chorus"), (56, 72, "outro")]
    cp = ds.values(ds.CUE_PARAMS, BOOKS["cues"])
    pl = ds.sections_played(T, 8, 32, cp)
    assert pl["complete"] == ["verse", "chorus"] and pl["n_core"] == 2
    assert ds.section_ok(T, pl, "full", cp)
    pl2 = ds.sections_played(T, 8, 30, cp)   # chorus cut off
    assert pl2["complete"] == ["verse"] and not ds.section_ok(T, pl2, "full", cp) and ds.section_ok(T, pl2, "one", cp)
    m = ds._section_matrix(T, np.array([8, 24]), np.array([32, 48]), "full", cp)
    assert math.isfinite(m[0, 0]) and math.isinf(m[1, 0]) and math.isfinite(m[1, 1])


def test_choose_cues_sections_mode_plays_verse_and_chorus():
    tracks = {s: _track(s, 72, SECS, VOC, bpm=90.0 + i) for i, s in enumerate("abcd")}
    order = list("abcd")
    pc = {(x, y): ds.candidates_for_pair(tracks[x], tracks[y], BOOKS["tree"], BOOKS["order"])
          for x, y in zip(order, order[1:])}
    res = ds.choose_cues(order, tracks, pc, BOOKS["cues"])
    assert res["dur_mode"] == "sections" and res["section_requirement"] in ds.SECTION_REQUIREMENTS
    lo, hi = res["play_bounds_s"]
    for d in res["play_s"]:
        assert lo - 1e-6 <= d <= hi + 1e-6
    if res["section_requirement"] == "full":
        for pl in res["sections_played"]:
            assert pl is not None and "verse" in pl["complete"] and "chorus" in pl["complete"]
    # default (seconds) mode is unchanged and reports no section requirement
    res0 = ds.choose_cues(order, tracks, {k: ds.candidates_for_pair(tracks[k[0]], tracks[k[1]]) for k in pc})
    assert res0["dur_mode"] == "seconds" and res0["section_requirement"] == "none"


def test_fx_run_penalty_breaks_consecutive_effects():
    tracks = {s: _track(s, 72, SECS, VOC) for s in "abc"}
    order = list("abc")

    def cand(kind, cost, a_out_bar, b_in_bar, A, B):
        dec = ds._mk(kind, ds.values(ds.TREE_PARAMS))
        dec.update(type_key=kind)
        bar = 4 * A["tempo"]["period_s"]
        return {"plan": {"a_out_s": a_out_bar * bar, "a_out_bar": a_out_bar, "b_in_s": b_in_bar * bar, "b_in_bar": b_in_bar},
                "decision": dec, "cost": cost, "admissible": True}

    pc = {}
    for x, y in zip(order, order[1:]):
        A, B = tracks[x], tracks[y]
        pc[(x, y)] = {"candidates": [cand("echo_out", 0.0, 32, 8, A, B), cand("cut", 0.05, 32, 8, A, B)]}
    base = {"w_repeat": 0.0, "play_min_s": 30.0}
    r0 = ds.choose_cues(order, tracks, pc, base)
    assert [c["decision"]["type"] for c in r0["chosen"]] == ["echo_out", "echo_out"]
    r1 = ds.choose_cues(order, tracks, pc, dict(base, w_fx_run=0.3))
    assert [c["decision"]["type"] for c in r1["chosen"]].count("echo_out") == 1


# ------------------------------------------------------------------ policy
def _pctx(**kw):
    base = {"stretch_abs": 0.15, "key_relation": "same", "key_confident": True, "key_shift": 0, "has_stems": True,
            "a_vocal_last_beat": 0.0, "a_vocal_bars_before": [1.0, 1.0, 1.0, 0.8], "a_vocal_bars": [0.0] * 16,
            "b_vocal_bars": [1.0] * 16, "a_bars_after": 20, "a_bars_before": 40, "b_section": "verse",
            "energy_a_db_rel": 0.0, "energy_b_db_rel": 0.0, "key_a": "A minor", "key_b": "A minor", "bar_a_s": 2.0,
            "lyric_links": [], "recent_techniques": [], "a_exit_on_boundary": True, "b_section_start": True}
    base.update(kw)
    return base


def test_policy_backspin_at_phrase_end_and_fx_moderation():
    t0 = [p["technique"] for p in pol.propose(_pctx())["proposals"]]
    assert t0[0] == "echo_freeze"  # default unchanged
    r = pol.propose(_pctx(), params=pol.PROFILE_BOOKS)
    assert r["proposals"][0]["technique"] == "backspin"
    assert any("B&B" in w for w in r["proposals"][0]["why"])
    # an effect already used recently: every marked effect loses fx_budget_penalty
    r2 = pol.propose(_pctx(recent_techniques=["reverb_wash"]), params=pol.PROFILE_BOOKS)
    s1 = {p["technique"]: p["score"] for p in r["considered"] if "score" in p}
    s2 = {p["technique"]: p["score"] for p in r2["considered"] if "score" in p}
    assert s2["echo_freeze"] == pytest.approx(s1["echo_freeze"] - pol.PROFILE_BOOKS["fx_budget_penalty"])


def test_policy_tape_stop_before_a_punchy_track():
    c = _pctx(energy_b_db_rel=5.0, a_exit_on_boundary=False)
    assert "tape_stop" not in [p["technique"] for p in pol.propose(c, max_proposals=6)["proposals"]]
    t = [p["technique"] for p in pol.propose(c, max_proposals=6, params=pol.PROFILE_BOOKS)["proposals"]]
    assert "tape_stop" in t


# ------------------------------------------------------ Butler form features
def _form(labels, layers, drops=(), breakdowns=(), busy=None, vocals=None):
    n = len(labels)
    return {"labels": list(labels), "n_layers": list(layers), "drops": list(drops), "breakdowns": [list(b) for b in breakdowns],
            "busy": list(busy if busy is not None else [False] * n),
            "active": {"vocals": list(vocals if vocals is not None else [False] * n)}}


def test_form_context_off_without_forms_and_profile_defaults():
    A = _track("A", 64, SECS, VOC)
    B = _track("B", 64, SECS, VOC)
    assert ds.form_context(A, B, 24, 8) == {"form_available": False}
    assert ds.values(ds.TREE_PARAMS)["dense_blend_layers"] is None
    for k in ("w_exit_in_build", "w_exit_breakdown_bonus", "w_entry_drop_bonus", "w_dense_overlap"):
        assert ds.values(ds.ORDER_PARAMS)[k] == 0.0


def test_form_context_and_placement_costs():
    o = BOOKS["order"]
    labels_a = ["intro"] * 8 + ["build"] * 8 + ["core"] * 32 + ["breakdown"] * 8 + ["outro"] * 8
    A = {**_track("A", 64, SECS, VOC), "form": _form(labels_a, [1] * 8 + [3] * 8 + [5] * 32 + [2] * 8 + [2] * 8,
                                                      breakdowns=[(48, 56)])}
    B = {**_track("B", 64, SECS, VOC), "form": _form(["intro"] * 4 + ["breakdown"] * 4 + ["core"] * 56, [1] * 4 + [1] * 4 + [5] * 56,
                                                      drops=[8], breakdowns=[(4, 8)])}
    c_build = ds.form_context(A, B, 12, 8)
    assert c_build["form_available"] and c_build["a_form_before"] == "build" and c_build["b_entry_drop"]
    plan = {"cue_score": 5.0, "a_out_phrase_level": 16, "b_in_phrase_level": 16}
    pc = ds.placement_cost(plan, {**_ctx(), **c_build}, o)
    assert pc["exit_in_build"] == o["w_exit_in_build"] and pc["entry_drop"] == -o["w_entry_drop_bonus"]
    assert ds.placement_cost(plan, {**_ctx(), **c_build})["exit_in_build"] == 0.0  # default off
    c_bd = ds.form_context(A, B, 48, 0)
    assert c_bd["a_exit_in_breakdown"] and ds.placement_cost(plan, {**_ctx(), **c_bd}, o)["exit_breakdown"] < 0
    # full textures in both: blend costs more with the profile, nothing changes without it
    dense = {**_ctx(), "form_available": True, "overlap_layers": 9.0, "overlap_both_busy": 0.5}
    blend = {"type": "blend", "type_key": "blend8"}
    assert ds.dense_overlap_cost(blend, dense) == 0.0
    assert ds.dense_overlap_cost(blend, dense, o) == pytest.approx(o["w_dense_overlap"] * (1.5 + 0.5))
    assert ds.dense_overlap_cost({"type": "cut", "type_key": "cut"}, dense, o) == 0.0


def test_dense_textures_shorten_the_blend():
    base = _ctx(a_exit_section="verse", b_section="intro", form_available=True, overlap_layers=8.0, overlap_both_busy=0.0)
    assert ds.decide_transition(base)["bars"] == 16  # default: rule off
    d = ds.decide_transition(base, BOOKS["tree"])
    assert d["type"] == "blend" and d["bars"] == 4 and d["rule"] == "R7c_dense"
    thin = dict(base, overlap_layers=5.0)
    assert ds.decide_transition(thin, BOOKS["tree"])["bars"] == 8
    busy = dict(base, overlap_layers=5.0, overlap_both_busy=0.75)
    assert ds.decide_transition(busy, BOOKS["tree"])["rule"] == "R7c_dense"


def test_policy_build_before_a_drop_entry():
    c = _pctx(stretch_abs=0.0, b_entry_drop=True, b_vocal_bars=[0.0] * 16)
    assert "build_into_chorus" not in [p["technique"] for p in pol.propose(c, max_proposals=6)["proposals"]]
    r = pol.propose(c, max_proposals=6, params=pol.PROFILE_BOOKS)
    bp = [p for p in r["proposals"] if p["technique"] == "build_into_chorus"]
    assert bp and "drop" in bp[0]["why"][0]
