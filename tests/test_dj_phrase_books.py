"""Tests for the book-derived additions to musicmix.dj_phrase (Ohriner, Krims, Huron, Margulis; 21/10/2026).

Defaults must leave the v1 behaviour unchanged (pillar weight 0, flow weight 0).
"""
import math

import numpy as np
import pytest

from musicmix import dj_phrase as P

SR = P.SR
BEAT = 60.0 / 90.0
BAR = 4 * BEAT


def _model(words, n_bars=16, lines=None):
    """Minimal dj_phrase model: words as (text, on_s, off_s, n_syl), grid from 0.5 s."""
    down = (np.arange(n_bars) * BAR + 0.5).tolist()
    ws = []
    for text, on, off, n in words:
        last = on + (off - on) * (n - 1) / n
        ws.append({"text": text, "clean": text, "on": int(on * SR), "off": int(off * SR), "last_syl_on": int(last * SR),
                   "n_syl": n, "stress_idx": 0, "rhyme": {"score": 0.0}})
    lines = lines or [{"first_word": 0, "last_word": len(ws) - 1, "start_sample": ws[0]["on"], "end_sample": ws[-1]["off"]}]
    return {"sample_rate": SR, "period_s": BEAT, "downbeats_s": down, "words": ws, "lines": lines, "gaps": []}


# ------------------------------------------------------------------ pillar (Ohriner p. 110)
def _gaps(positions, p=0.6, bar=2):
    return [{"after": i, "p_end": p, "bar": bar, "pos_beats": q / 4.0, "terms": {}} for i, q in enumerate(positions)]


def test_pillar_weight_zero_is_v1():
    g = _gaps([12] * 10 + [4] * 2)
    before = [x["p_end"] for x in g]
    info = P.pillar_adjust(g, weight=0.0)
    assert [x["p_end"] for x in g] == before
    assert info["applied"] is False and info["pillar_pos16"] == 12
    assert P.PARAMS["pillar_weight"].value == 0.0


def test_pillar_raises_the_tracks_own_position_and_lowers_rare_ones():
    g = _gaps([4] * 10 + [12] * 2 + [8], p=0.55)
    info = P.pillar_adjust(g, weight=1.0, min_ends=8)
    assert info["applied"]
    at4 = [x for x in g if x["pos_beats"] == 1.0]
    at8 = [x for x in g if x["pos_beats"] == 2.0]
    # position 4 is rare in the MCFlow corpus but frequent in this track -> up; position 8 -> down
    assert all(x["p_end"] > 0.55 and x["terms"]["pillar"] > 0 for x in at4)
    assert all(x["p_end"] < 0.55 for x in at8)
    assert all(x["p_end_base"] == 0.55 for x in g)


def test_pillar_needs_enough_line_ends():
    g = _gaps([4] * 3)
    info = P.pillar_adjust(g, weight=1.0, min_ends=8)
    assert not info["applied"] and all(x["p_end"] == 0.6 for x in g)


def test_reanalyze_with_pillar_is_idempotent_and_rebuilds_lines():
    m = _model([("a", 1.0, 1.2, 1), ("b", 1.3, 1.5, 1), ("c", 1.6, 1.8, 1)])
    m["gaps"] = [{"after": 0, "p_end": 0.52, "bar": 0, "pos_beats": 1.0, "terms": {}},
                 {"after": 1, "p_end": 0.3, "bar": 0, "pos_beats": 1.5, "terms": {}}]
    m["params"] = {}
    m1 = P.reanalyze_with_pillar(m, 1.0, min_ends=1)
    m2 = P.reanalyze_with_pillar(m1, 1.0, min_ends=1)
    assert [g["p_end"] for g in m1["gaps"]] == pytest.approx([g["p_end"] for g in m2["gaps"]])
    assert m1["params"]["pillar_weight"] == 1.0
    assert len(m1["lines"]) == 1 + sum(g["p_end"] >= 0.5 for g in m1["gaps"])


# ------------------------------------------------------------------ nPVI (Patel & Daniele via Huron pp. 188-189)
def test_npvi_isochronous_is_zero_and_alternation_is_high():
    iso = [{"word_i": k, "idx": 0, "n_syl": 1, "t_s": 0.25 * k} for k in range(9)]
    assert P.syllable_npvi(iso) == pytest.approx(0.0)
    t, alt = 0.0, []
    for k in range(9):
        alt.append({"word_i": k, "idx": 0, "n_syl": 1, "t_s": t})
        t += 0.2 if k % 2 == 0 else 0.4
    assert P.syllable_npvi(alt) == pytest.approx(100 * 0.2 / 0.3)


def test_npvi_skips_pairs_across_line_ends():
    s = [{"word_i": k, "idx": 0, "n_syl": 1, "t_s": x} for k, x in enumerate([0.0, 0.25, 0.5, 2.0, 2.25, 2.5])]
    # word 2 ends a line: the long pause 0.5 -> 2.0 must not enter
    assert P.syllable_npvi(s, line_end_words={2}) == pytest.approx(0.0)
    assert P.syllable_npvi(s) > 0


# ------------------------------------------------------------------ hypermeter and first-bar caesura (Krims p. 50)
def test_hypermeter_phase_from_late_line_ends():
    assert P.hypermeter_phase([(3, 12), (7, 13), (11, 12), (5, 4)])["phase"] == 0
    assert P.hypermeter_phase([(4, 12), (8, 13), (12, 12)])["phase"] == 1


def test_first_bar_caesura_counts_mid_bar_ends_in_group_starts():
    ends = [(3, 12), (7, 12), (11, 12), (15, 12), (4, 6), (12, 2)]
    v = P.first_bar_caesura(ends, active_bars=range(16), phase=0)
    assert v == pytest.approx(2 / 4)  # groups starting at bars 4 and 12 have an end before beat 4
    assert P.first_bar_caesura([(4, 12)], active_bars=range(8), phase=0) == 0.0  # beat-4 end is not a caesura


def test_hypermetric_weight_profile():
    assert [P.hypermetric_weight(k) for k in range(4)] == [1.0, 0.25, 0.5, 0.25]
    assert P.hypermetric_weight(4) == 1.0


# ------------------------------------------------------------------ vocal phase (Ohriner ch. 8)
def test_vocal_phase_alignment_recovers_a_constant_lag():
    lag = 0.05 * BEAT  # 3/60 of a beat late
    words = [(f"w{k}", 0.5 + k * BEAT / 2 + lag, 0.5 + k * BEAT / 2 + lag + 0.1, 1) for k in range(24)]
    r = P.vocal_phase_alignment(_model(words))
    assert r["phase_beats"] == pytest.approx(0.05, abs=1 / 60 + 1e-9)
    assert r["misalignment_phase"] < r["misalignment"]
    assert r["phase_ms"] > 0


def test_vocal_phase_alignment_finds_swing():
    # sixteenths with a 2:1 swing inside every eighth
    t, words = 0.5, []
    for k in range(32):
        words.append((f"w{k}", t, t + 0.05, 1))
        t += (BEAT / 2) * (2 / 3 if k % 2 == 0 else 1 / 3)
    r = P.vocal_phase_alignment(_model(words))
    assert r["bur"] == pytest.approx(2.0, abs=0.06)
    assert r["misalignment_phase_swing"] < r["misalignment"]


def test_vocal_phase_alignment_needs_onsets():
    assert "phase_beats" not in P.vocal_phase_alignment(_model([("a", 1.0, 1.2, 1)]))


# ------------------------------------------------------------------ build by halving (Margulis pp. 23, 44)
@pytest.mark.parametrize("total", [4, 8, 16])
def test_build_halving_schedule_sums_and_halves(total):
    st = P.build_halving_schedule(total)
    assert sum(s["unit_bars"] * s["repeats"] for s in st) == pytest.approx(total)
    units = [s["unit_bars"] for s in st]
    assert units == sorted(units, reverse=True) and all(b == a / 2 for a, b in zip(units, units[1:]))
    assert st[0]["start_bar"] == 0 and all(s["repeats"] >= 2 for s in st[1:])


def test_build_halving_schedule_short_span_drops_small_units():
    st = P.build_halving_schedule(2, start_unit=1.0, min_unit=0.25)
    assert sum(s["unit_bars"] * s["repeats"] for s in st) == pytest.approx(2)


# ------------------------------------------------------------------ loops (Huron p. 286, Margulis pp. 51, 173)
def test_loop_comic_threshold_for_short_salient_fragments():
    r = P.loop_repetition_check(1, 4, new_layer_entering=False)
    assert not r["ok"] and any("disco rotto" in w for w in r["warnings"])
    assert r["limits"]["loop_comic_repeats"] == 3 and "Huron" in r["sources"]["loop_comic_repeats"]
    assert P.loop_repetition_check(1, 4, new_layer_entering=True)["ok"]
    assert P.loop_repetition_check(4, 8, new_layer_entering=False)["ok"]


def test_book_rules_carry_sources_and_labels():
    for k in ("loop_comic_repeats", "build_halving_units", "hypermeter_profile", "stm_seconds", "style_schema_seconds",
              "schema_switch_seconds", "phrase_reset_bits", "first_bar_caesura", "npvi_english_speech"):
        src = P.BOOK_RULES[k].source
        assert any(lbl in src for lbl in ("[T]", "[E]", "[A]", "[H]")) and any(a in src for a in ("Huron", "Margulis", "Krims"))


# ------------------------------------------------------------------ flow features keep working with the new keys
def test_flow_features_report_npvi_and_first_bar_caesura():
    words = []
    for b in range(8):
        t0 = 0.5 + b * BAR
        for k in range(6):
            words.append((f"x{b}{k}", t0 + k * BEAT / 2, t0 + k * BEAT / 2 + 0.15, 1))
    lines = []
    for b in range(8):
        i0, i1 = 6 * b, 6 * b + 5
        lines.append({"first_word": i0, "last_word": i1, "start_sample": int((0.5 + b * BAR) * SR),
                      "end_sample": int((0.5 + b * BAR + 2.8 * BEAT) * SR)})
    m = _model(words, lines=lines)
    f = P.flow_features(m)
    assert f["npvi"] is not None and f["npvi"] < 5  # isochronous eighths
    assert f["first_bar_caesura"] is not None and 0.0 <= f["first_bar_caesura"] <= 1.0
    assert f["hypermeter_phase"] in range(4)
    st = P.krims_style(f)
    assert st["n_descriptors"]["speech_effusive"] >= 1 and math.isfinite(sum(st["membership"].values()))


def test_pillar_term_in_list_format_and_removed_on_reanalysis():
    g = [{"after": i, "p_end": 0.6, "bar": 2, "pos_beats": q / 4.0, "terms": [{"name": "rima", "log_lr": 0.1}]}
         for i, q in enumerate([4] * 10 + [8])]
    P.pillar_adjust(g, weight=1.0, min_ends=8)
    assert [t["name"] for t in g[0]["terms"]] == ["rima", "pilastro"]
    m = _model([(f"w{k}", 1.0 + 0.2 * k, 1.1 + 0.2 * k, 1) for k in range(12)])
    m["gaps"], m["params"] = g, {}
    m2 = P.reanalyze_with_pillar(m, 1.0, min_ends=8)
    assert [t["name"] for t in m2["gaps"][0]["terms"]] == ["rima", "pilastro"]
    assert m2["gaps"][0]["p_end_base"] == 0.6


def _exit_case(n_line_words, next_on):
    words = [(f"w{k}", 0.5 + BAR + k * BEAT / 2, 0.5 + BAR + k * BEAT / 2 + 0.2, 1) for k in range(n_line_words)]
    words.append(("next", next_on, next_on + 0.25, 1))
    words += [(f"z{k}", 0.5 + 2 * BAR + (k + 1) * BEAT, 0.5 + 2 * BAR + (k + 1) * BEAT + 0.2, 1) for k in range(3)]
    m = _model(words)
    m["gaps"] = [{"after": i, "p_end": 0.9 if i == n_line_words - 1 else 0.05, "bar": 1 if i < n_line_words else 2, "pos_beats": (i + 1) / 2,
                  "terms": [], "end_sample": m["words"][i]["off"], "next_start_sample": m["words"][i + 1]["on"]}
                 for i in range(len(words) - 1)]
    m["asr"] = {"reliable": True}
    return P.exit_point(m, 0.5 + 2 * BAR)


def test_exit_point_reports_krims_closure():
    r = _exit_case(7, 0.5 + 2 * BAR + 0.05)  # line ends on beat 4, A silent until after the downbeat
    assert r["end_on_beat4"] and r["krims_closure"] == "closed_bc3"
    r = _exit_case(4, 0.5 + 2 * BAR - BEAT * 0.75)  # ends on beat 2-3, A's next syllable before the downbeat
    assert r["krims_closure"] == "pickup" and r["a_pickup_sample"] is not None


# ------------------------------------------------------------------ stem onsets bound to the word (audit 21/10/2026)
def test_stem_onset_past_word_end_is_not_snapped():
    """ROOT_COUNTEREXAMPLES.json 'stem_onset_past_word': word 900-1000 ms, stem onset at 1040 ms. Before the fix the
    onset was given to the word and counted as snapped (100 %). Now it is never assigned past ``off``: the syllable
    keeps the uniform estimate, is marked uncertain and is not counted as snapped."""
    word = {"on": 900, "off": 1000, "last_syl_on": 990, "n_syl": 1, "text": "test"}
    m, info = P.with_stem_onsets({"sample_rate": 1000, "words": [word]}, onsets={"onsets_s": [1.04]})
    w = m["words"][0]
    assert w["syl_on"] == [990] and w["syl_src"] == ["uncertain_beyond_word"]
    assert info["snapped_share"] == 0.0 and info["n_uncertain_beyond_word"] == 1 and info["unused_stem_onsets"] == 1
    # the onset inside the word is still snapped
    m2, info2 = P.with_stem_onsets({"sample_rate": 1000, "words": [dict(word)]}, onsets={"onsets_s": [0.95]})
    assert m2["words"][0]["syl_on"] == [950] and m2["words"][0]["syl_src"] == ["stem"] and info2["snapped_share"] == 1.0
    # an onset just before the ASR onset (inside the window) still counts: ASR onsets are often late
    w2 = {"on": 900, "off": 1100, "last_syl_on": 1000, "n_syl": 2, "text": "test"}
    m3, _ = P.with_stem_onsets({"sample_rate": 1000, "words": [w2]}, onsets={"onsets_s": [0.88, 1.01]})
    assert m3["words"][0]["syl_on"] == [880, 1010] and m3["words"][0]["syl_src"] == ["stem", "stem"]
    # legacy mode reproduces the old behaviour for re-running old comparisons
    m4, info4 = P.with_stem_onsets({"sample_rate": 1000, "words": [dict(word)]}, onsets={"onsets_s": [1.04]}, beyond_word="legacy")
    assert m4["words"][0]["syl_on"] == [1040] and info4["snapped_share"] == 1.0
    with pytest.raises(ValueError):
        P.with_stem_onsets({"sample_rate": 1000, "words": [dict(word)]}, onsets={"onsets_s": [1.04]}, beyond_word="snap")


def test_onset_past_word_goes_to_the_next_word():
    w1 = {"on": 900, "off": 1000, "last_syl_on": 990, "n_syl": 1, "text": "a"}
    w2 = {"on": 1030, "off": 1200, "last_syl_on": 1060, "n_syl": 1, "text": "b"}
    m, info = P.with_stem_onsets({"sample_rate": 1000, "words": [w1, w2]}, onsets={"onsets_s": [1.04]})
    assert m["words"][0]["syl_src"] == ["uncertain_beyond_word"]
    assert m["words"][1]["syl_on"] == [1040] and m["words"][1]["syl_src"] == ["stem"]
    assert info["n_snapped"] == 1 and info["n_uncertain_beyond_word"] == 1
