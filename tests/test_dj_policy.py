"""Rules of dj_policy on hand-made contexts (no audio)."""

from __future__ import annotations

from musicmix import dj_policy as pol


def ctx(**kw):
    base = {
        "stretch_abs": 0.02, "key_relation": "same", "key_confident": True, "key_shift": 0, "has_stems": True,
        "a_vocal_last_beat": 0.0, "a_vocal_bars_before": [1.0, 1.0, 1.0, 0.8], "a_vocal_bars": [0.0] * 16,
        "b_vocal_bars": [1.0] * 16, "a_bars_after": 20, "a_bars_before": 40, "b_section": "verse",
        "energy_a_db_rel": 0.0, "energy_b_db_rel": 0.0, "key_a": "A minor", "key_b": "A minor", "bar_a_s": 2.0,
        "lyric_links": [], "recent_techniques": [],
    }
    base.update(kw)
    return base


def techs(r):
    return [p["technique"] for p in r["proposals"]]


def test_verse_handoff_proposes_acapella_and_vocal_swap():
    r = pol.propose(ctx())
    t = techs(r)
    assert "acapella_over" in t and "vocal_swap" in t
    assert 2 <= len(t) <= 4
    p = next(p for p in r["proposals"] if p["technique"] == "acapella_over")
    assert p["params"]["bars"] == 8 and p["rule"] == "C3_verse_handoff" and p["rule_source"]


def test_far_tempo_has_no_overlap_techniques():
    r = pol.propose(ctx(stretch_abs=0.15))
    t = techs(r)
    assert not set(t) & {"acapella_over", "vocal_swap", "loop_extend"}
    assert "echo_freeze" in t


def test_chorus_entry_gets_a_build_not_a_drop():
    r = pol.propose(ctx(b_section="chorus", b_vocal_bars=[0.0] * 16, a_vocal_bars_before=[0.0] * 4))
    assert techs(r)[0] == "build_into_chorus"


def test_lyric_link_first_when_present():
    link = {"score": 0.8, "a_cue_s": 100.0, "b_cue_s": 10.0, "match": {"type": "rhyme"},
            "a_word": {"text": "money", "start_sample": 99 * 48000}, "b_word": {"text": "honey", "start_sample": 10 * 48000 + 24000}}
    r = pol.propose(ctx(lyric_links=[link]))
    assert techs(r)[0] == "lyric_link"
    assert r["proposals"][0]["params"]["cue_out_a_s"] == 100.0


def test_key_shift_is_passed_when_needed():
    r = pol.propose(ctx(key_relation="clash", key_shift=-1))
    p = next(p for p in r["proposals"] if p["technique"] == "acapella_over")
    assert p["params"]["key_shift_b"] == -1
    r2 = pol.propose(ctx(key_relation="clash", key_shift=None))
    assert "acapella_over" not in techs(r2)


def test_repetition_penalty_and_table():
    r1 = pol.propose(ctx(stretch_abs=0.15))
    r2 = pol.propose(ctx(stretch_abs=0.15, recent_techniques=["echo_freeze"]))
    s1 = next(p["score"] for p in r1["proposals"] if p["technique"] == "echo_freeze")
    s2 = next((p["score"] for p in r2["considered"] if p["technique"] == "echo_freeze" and "score" in p), None)
    assert s2 is not None and s2 < s1
    md = pol.table_markdown()
    assert md.count("\n") == len(pol.POLICY_TABLE) + 1


# ------------------------------------------------------------------ grid guard (come R1_grid_low di dj_set)
def _books_chorus(**kw):
    base = dict(b_section="chorus", b_vocal_bars=[0.0] * 16, a_vocal_bars_before=[1.0] * 4, a_vocal_last_beat=0.6,
                a_exit_on_boundary=True, b_section_start=True, a_vocal_bars=[0.0] * 16)
    base.update(kw)
    return pol.propose(ctx(**base), params=pol.PROFILE_BOOKS, max_proposals=10)


def test_low_grid_blocks_halving_loop_break_and_stutter():
    good = _books_chorus(tempo_confidence_a="high", tempo_confidence_b="high")
    assert "halving_build" in techs(good) and "break_before_entry" in techs(good)
    bg = next(p for p in good["proposals"] if p["technique"] == "build_into_chorus")
    assert bg["params"]["stutter_bars"] == 1 and bg["params"].get("gap_beats") == 1
    for low in ({"tempo_confidence_a": "low", "tempo_confidence_b": "high"},
                {"tempo_confidence_a": "high", "tempo_confidence_b": "low"}):
        r = _books_chorus(**low)
        t = techs(r)
        assert not set(t) & {"halving_build", "break_before_entry", "loop_extend", "acapella_over", "vocal_swap"}
        b = next(p for p in r["proposals"] if p["technique"] == "build_into_chorus")
        assert b["params"]["stutter_bars"] == 0 and "gap_beats" not in b["params"]
        assert r["trace"][0]["inputs"]["grid_low"] is True
        assert any(c["technique"] == "halving_build" and not c["proposed"] and "griglia" in c["why"] for c in r["considered"])


def test_good_grid_unchanged_and_guard_can_be_disabled():
    for kw in ({}, {"tempo_confidence_a": "high", "tempo_confidence_b": "medium"}):
        assert pol.propose(ctx(**kw))["proposals"] == pol.propose(ctx(**kw), params={"grid_guard": False})["proposals"]
    low = ctx(tempo_confidence_a="low", tempo_confidence_b="high")
    assert "acapella_over" in techs(pol.propose(low, params={"grid_guard": False}))
    assert "acapella_over" not in techs(pol.propose(low))
