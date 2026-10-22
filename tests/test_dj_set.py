import itertools
import math

import numpy as np
import pytest

from musicmix import dj_select as djs
from musicmix import dj_set as ds


# ------------------------------------------------------------------ helpers
def _ctx(**kw):
    """Synthetic tree context: tempo/key compatible, instrumental, B on an intro."""
    base = {
        "a": "A", "b": "B", "a_out_bar": 40, "b_in_bar": 0, "a_out_s": 100.0, "b_in_s": 0.0,
        "bpm_a": 90.0, "bpm_b": 90.0, "stretch_abs": 0.0, "octave_factor": 1.0,
        "tempo_confidence_a": "high", "tempo_confidence_b": "high",
        "key_relation": "same", "key_confident": True, "camelot_a": "8A", "camelot_b": "8A",
        "a_vocal_last_beat": 0.0, "a_vocal_last_bar": 0.0,
        "a_vocal_bars": [0.0] * 16, "b_vocal_bars": [0.0] * 16,
        "a_bars_before": 40, "a_bars_after": 24, "b_bars_after": 60,
        "a_exit_on_boundary": True, "a_exit_section": "verse",
        "b_section": "intro", "b_section_start": True,
        "a_phrase_level": 16, "b_phrase_level": 16,
        "energy_a_db_rel": 0.0, "energy_b_db_rel": 0.0,
    }
    base.update(kw)
    return base


def _track(slug, n_bars, sections, vocal, bpm=90.0, camelot="8A", key_conf=0.9, energy=None):
    bar = 4 * 60.0 / bpm
    downbeats = np.arange(n_bars) * bar
    secs = [{"label": lab, "start_s": b * bar, "end_s": 0.0, "bar_index": b, "downbeat_s": b * bar,
             "snap_error_bars": 0.0, "loose": False, "support": ["x"], "source": "test"}
            for b, lab in sections]
    vocal = np.asarray(vocal, dtype=float)
    return {
        "slug": slug, "duration_s": n_bars * bar,
        "tempo": {"bpm": bpm, "confidence": "high", "period_s": 60.0 / bpm},
        "key": {"camelot": camelot, "confidence": key_conf},
        "sections": secs,
        "phrases": djs.phrase_table(downbeats, secs),
        "bars": {"vocal": vocal.tolist(), "vocal_last_beat": vocal.tolist(),
                 "energy_db_rel": (np.zeros(n_bars) if energy is None else np.asarray(energy)).tolist()},
    }


# --------------------------------------------------------------------- tree
def test_tempo_outside_8_percent_never_blends():
    d = ds.decide_transition(_ctx(stretch_abs=0.12, a_vocal_last_beat=0.6, a_vocal_last_bar=0.9))
    assert d["rule"] == "R2_tempo_out" and d["type"] == "echo_out"
    # clean exit on a boundary into an instrumental section start -> cut on the downbeat
    d = ds.decide_transition(_ctx(stretch_abs=0.12, b_section="verse"))
    assert d["type"] == "cut"
    assert all(f["type"] != "blend" for f in d["fallbacks"])


def test_both_vocals_one_voice_at_a_time_never_long_blend():
    voc = [1.0] * 16
    d = ds.decide_transition(_ctx(a_vocal_bars=voc, b_vocal_bars=voc, b_section="verse"))
    assert d["rule"] == "R6_both_vocal" and d["type"] == "filter_out"
    d = ds.decide_transition(_ctx(a_vocal_bars=voc, b_vocal_bars=voc, b_section="verse", a_vocal_last_beat=0.8))
    assert d["type"] == "echo_out"
    assert all(f["type"] != "blend" for f in d["fallbacks"])


def test_instrumental_compatible_parts_blend_8_or_16():
    d = ds.decide_transition(_ctx())
    assert d["rule"] == "R7_instrumental" and d["type"] == "blend" and d["bars"] == 16
    assert [f["type_key"] for f in d["fallbacks"]][:2] == ["blend8", "blend4"]
    d = ds.decide_transition(_ctx(stretch_abs=0.06))  # 5-8 %: no long blend (Kim 2020)
    assert d["type"] == "blend" and d["bars"] == 8
    d = ds.decide_transition(_ctx(b_vocal_bars=[0.0] * 8 + [1.0] * 8))  # B sings after 8 bars
    assert d["bars"] == 8


def test_chorus_entry_is_prepared_never_a_dry_drop():
    d = ds.decide_transition(_ctx(b_section="chorus", b_vocal_bars=[1.0] * 16))
    assert d["rule"] == "R3_chorus_entry" and d["type"] == "filter_into_chorus"
    assert d["render"]["sweep_bars"] in (1, 2)
    d = ds.decide_transition(_ctx(b_section="chorus", a_vocal_last_beat=0.9))
    assert d["type"] == "echo_out"
    for kw in ({}, {"stretch_abs": 0.2}, {"key_relation": "clash"}, {"b_section": "chorus"}):
        dd = ds.decide_transition(_ctx(**kw))
        assert "drop" not in dd["type"] and all("drop" not in f["type"] for f in dd["fallbacks"])


def test_key_is_a_hard_constraint_for_blends_only():
    d = ds.decide_transition(_ctx(key_relation="clash"))
    assert d["rule"] == "R4_key_no_blend" and d["type"] in ("filter_out", "echo_out")
    d = ds.decide_transition(_ctx(key_relation="adjacent", key_confident=False))
    assert d["type"] != "blend"


def test_one_voice_short_blend_and_trace():
    a = [1.0] * 2 + [0.0] * 14
    b = [0.0] * 2 + [1.0] * 14   # voices meet for at most 1 bar on 4 (clash 0) but B sings
    d = ds.decide_transition(_ctx(a_vocal_bars=a, b_vocal_bars=b, b_section="verse"))
    assert d["type"] == "blend" and d["rule"] in ("R7_instrumental", "R8_one_voice")
    assert any(t["rule"] == "R6_both_vocal" and t["result"] is False for t in d["trace"])
    assert {r["id"] for r in ds.TREE_RULES} >= {d["rule"]}


def test_context_from_analyses_and_candidates():
    secs = [(0, "intro"), (8, "verse"), (24, "chorus"), (40, "outro")]
    A = _track("A", 56, secs, [0.0] * 8 + [1.0] * 32 + [0.0] * 16)
    B = _track("B", 56, secs, [0.0] * 8 + [1.0] * 32 + [0.0] * 16)
    pc = ds.candidates_for_pair(A, B)
    assert pc["candidates"]
    c0 = pc["candidates"][0]
    assert c0["cost"] == min(c["cost"] for c in pc["candidates"])
    for c in pc["candidates"]:
        if c["decision"]["type"] == "blend":
            n = c["decision"]["bars"]
            ja, jb = c["plan"]["a_out_bar"], c["plan"]["b_in_bar"]
            va = np.array(A["bars"]["vocal"][ja:ja + n]) >= 0.5
            vb = np.array(B["bars"]["vocal"][jb:jb + n]) >= 0.5
            assert int((va & vb).sum()) <= djs.CLASH_TOLERATED_BARS
        if c["context"]["b_section"] == "chorus":
            assert c["decision"]["type"] in ("filter_into_chorus", "echo_out")


# ---------------------------------------------------------------- features
def test_tempo_octave_equivalence_and_costs():
    assert ds.folded_tempo(70.0, 140.0)["stretch"] == pytest.approx(0.0)
    assert ds.folded_tempo(140.0, 72.0)["stretch"] == pytest.approx(144 / 140 - 1, rel=1e-9)
    assert ds.tempo_cost(90, 90) == 0.0
    assert ds.tempo_cost(90, 95) < ds.tempo_cost(90, 99) < ds.tempo_cost(90, 105)
    assert ds.tempo_cost(90, 99.5) - ds.tempo_cost(90, 98.9) > 0.9  # Ishizaki step beyond 10 %
    assert math.isinf(ds.tempo_cost(90, 112))


def test_energy_scores_and_arc():
    e = ds.energy_scores({"a": 1.0, "b": 3.0, "c": 2.0})
    assert e == {"a": 0.0, "b": 1.0, "c": 0.5}
    L = 10
    tgt = [ds.arc_target(i, L) for i in range(L)]
    peak = int(np.argmax(tgt))
    assert all(np.diff(tgt[: peak + 1]) > 0) and all(np.diff(tgt[peak:]) < 0)
    assert tgt[0] == pytest.approx(0.35) and tgt[-1] == pytest.approx(0.5)


def test_timbre_distance_symmetric():
    v = {"a": [0, 0, 0], "b": [1, 1, 1], "c": [5, 5, 5]}
    D = ds.timbre_distances(v, ["a", "b", "c"])
    assert np.allclose(D, D.T) and np.allclose(np.diag(D), 0)
    assert D[0, 1] < D[0, 2]


# ---------------------------------------------------------------- ordering
def _brute(edge, node, L, start=None, end=None):
    n = edge.shape[0]
    best = (math.inf, None)
    for perm in itertools.permutations(range(n), L):
        if start is not None and perm[0] != start or end is not None and perm[-1] != end:
            continue
        c = sum(node[i, j] for i, j in enumerate(perm)) + sum(edge[a, b] for a, b in zip(perm, perm[1:]))
        if c < best[0]:
            best = (c, list(perm))
    return best


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_held_karp_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n, L = 6, 4
    E = rng.random((n, n)) * 3
    E[rng.random((n, n)) < 0.2] = np.inf
    np.fill_diagonal(E, np.inf)
    N = rng.random((L, n))
    cost, path = ds.held_karp_path(E, N, L)
    bc, bp = _brute(E, N, L)
    assert cost == pytest.approx(bc)
    assert len(path) == L and len(set(path)) == L
    c2, p2 = ds.held_karp_path(E, N, L, start=2, end=4)
    b2, _ = _brute(E, N, L, start=2, end=4)
    assert c2 == pytest.approx(b2) and (not p2 or (p2[0] == 2 and p2[-1] == 4))


def test_order_follows_tempo_chain_and_energy_arc():
    # tempos 80..96: only neighbours within the hard limit chain cheaply
    slugs = [f"t{i}" for i in range(9)]
    bpm = {s: 80.0 + 2.0 * i for i, s in enumerate(slugs)}
    E = np.array([[ds.tempo_cost(bpm[a], bpm[b]) if a != b else np.inf for b in slugs] for a in slugs])
    energy = {s: 0.5 for s in slugs}
    res = ds.order_set(slugs, E, energy, {"len_min": 8, "len_max": 9, "len_target": 9, "w_arc": 0.0})
    o = res["order"]
    assert res["length"] == 9
    assert o in (slugs, slugs[::-1])  # monotone tempo path is optimal
    # energy arc: with flat edges, the order must rise to the peak and close lower
    en = {s: i / 8 for i, s in enumerate(slugs)}
    flat = np.ones((9, 9))
    np.fill_diagonal(flat, np.inf)
    res = ds.order_set(slugs, flat, en, {"len_min": 9, "len_max": 9, "len_target": 9})
    seq = [en[s] for s in res["order"]]
    pk = int(np.argmax(seq))
    assert 4 <= pk <= 7 and seq[0] < seq[pk] and seq[-1] < seq[pk]


def test_order_reports_longest_possible_when_short():
    slugs = ["a", "b", "c", "d"]
    E = np.full((4, 4), np.inf)
    E[0, 1] = E[1, 2] = 1.0   # d is isolated
    res = ds.order_set(slugs, E, {s: 0.5 for s in slugs}, {"len_min": 8})
    assert res["length"] == 3 and res["order"] == ["a", "b", "c"]
    assert "set più lungo possibile" in res["note"]


# ---------------------------------------------------------------- cues
def test_choose_cues_respects_play_duration():
    secs = [(0, "intro"), (8, "verse"), (24, "chorus"), (40, "verse"), (56, "outro")]
    voc = [0.0] * 8 + [1.0] * 48 + [0.0] * 16
    tracks = {s: _track(s, 72, secs, voc, bpm=90.0 + i) for i, s in enumerate("abc")}
    order = ["a", "b", "c"]
    pc = {(x, y): ds.candidates_for_pair(tracks[x], tracks[y]) for x, y in zip(order, order[1:])}
    res = ds.choose_cues(order, tracks, pc)
    for d in res["play_s"]:
        assert res["play_bounds_s"][0] - 1e-6 <= d <= res["play_bounds_s"][1] + 1e-6
    t0, t1 = res["chosen"]
    assert t1["plan"]["a_out_s"] - t0["plan"]["b_in_s"] == pytest.approx(res["play_s"][1])


def test_clip_geometry_mapping():
    meta = {"overlap": {"output_start_sample": 480000, "output_end_sample": 900000, "a_source_start_sample": 5000000,
                        "b_source_start_sample": 100000, "b_source_end_sample": 520000},
            "switch": {"output_sample": 700000, "b_source_sample": 310000},
            "stretch": {"applied": True, "tempo_ratio_b": 1.05}}
    g = ds.clip_geometry(meta, 1380000)
    assert g["a_offset"] == 5000000 - 480000
    assert ds.b_src_of_clip(g, ds.clip_pos_of_b(g, 600000)) == pytest.approx(600000)
    assert ds.clip_pos_of_b(g, 520000 + 1.05 * 1000) == pytest.approx(900000 + 1000)
