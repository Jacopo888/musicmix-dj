"""Piano per eventi (P0.5, musicmix.event_plan.1): schema, fixture della specifica, nessuna regressione dei default,
opzione sperimentale (spenta) che cambia la scelta in modo atteso su dati sintetici."""

import json

import numpy as np
import pytest

from musicmix import dj_events as dje
from musicmix import dj_select as djs
from musicmix import dj_set as ds

SR = dje.SR


def _track(slug, n_bars, sections, vocal, bpm=90.0, camelot="8A", form=None):
    bar = 4 * 60.0 / bpm
    downbeats = np.arange(n_bars) * bar
    secs = [{"label": lab, "start_s": b * bar, "end_s": 0.0, "bar_index": b, "downbeat_s": b * bar,
             "snap_error_bars": 0.0, "loose": False, "support": ["x"], "source": "test"} for b, lab in sections]
    vocal = np.asarray(vocal, dtype=float)
    t = {"slug": slug, "duration_s": n_bars * bar, "tempo": {"bpm": bpm, "confidence": "high", "period_s": 60.0 / bpm},
         "key": {"camelot": camelot, "confidence": 0.9}, "sections": secs, "phrases": djs.phrase_table(downbeats, secs),
         "bars": {"vocal": vocal.tolist(), "vocal_last_beat": vocal.tolist(), "energy_db_rel": [0.0] * n_bars}}
    if form is not None:
        t["form"] = form
    return t


SECS = [(0, "intro"), (8, "verse"), (24, "chorus"), (40, "verse"), (56, "chorus"), (72, "outro")]
VOC = [0.0] * 8 + [1.0] * 64 + [0.0] * 8


def _cand(A, B, a_out_bar, b_in_bar):
    pc = ds.candidates_for_pair(A, B)
    return next(c for c in pc["candidates"] if c["plan"]["a_out_bar"] == a_out_bar and c["plan"]["b_in_bar"] == b_in_bar), pc


def _required(doc):
    assert doc["schema"] == "musicmix.event_plan.1"
    for k in ("a", "b", "grid", "a_event", "b_entry", "b_target", "delay", "choice", "alternatives", "trace"):
        assert k in doc, k
    for k in ("kind", "label_source", "span_bars", "exit_bar", "exit_sample", "fate", "skipped", "line_end", "form_at_exit"):
        assert k in doc["a_event"], k
    for k in ("audible_sample", "lead_in", "downbeat_bar", "downbeat_sample"):
        assert k in doc["b_entry"], k
    for k in ("kind", "bar", "sample", "evidence", "status"):
        assert k in doc["b_target"], k
    for k in ("technique", "bars", "render", "tree_rule", "policy_rule", "cost"):
        assert k in doc["choice"], k
    for k in ("modules", "executed", "cue_windows", "profile", "sources"):
        assert k in doc["trace"], k
    assert doc["a_event"]["fate"] in ("completed", "skipped_rest", "cut_mid_line", "unknown")
    assert doc["b_entry"]["lead_in"]["kind"] in dje.LEAD_IN_KINDS
    assert doc["b_target"]["status"] in ("measured", "estimated", "unavailable")
    assert any(a["vary"] == "continue_A" for a in doc["alternatives"])
    assert {a["vary"] for a in doc["alternatives"]} >= {"duration", "continue_A"}
    json.loads(json.dumps(doc))  # fully serialisable (no numpy scalars)


def test_schema_complete_and_serialisable():
    A, B = _track("a", 80, SECS, VOC), _track("b", 80, SECS, VOC, bpm=91.0)
    c, pc = _cand(A, B, 40, 8)
    doc = dje.event_plan(A, B, c, pair_candidates=pc["candidates"], versions={"dj_phrase": "x"}, profile="books",
                         sources={"a_sha256": "0" * 64})
    _required(doc)
    assert doc["trace"]["modules"]["dj_events"] == dje.DJ_EVENTS_VERSION and doc["trace"]["modules"]["dj_phrase"] == "x"
    assert doc["candidates_table"]["columns"][0] == "a_out_bar" and len(doc["candidates_table"]["rows"]) == len(pc["candidates"])
    assert {a["vary"] for a in doc["alternatives"]} >= {"arrival", "exit", "cue"}


def test_exit_after_chorus_completed_and_next_chorus_skipped():
    """(a) uscita a fine ritornello: evento completato, il ritornello successivo è tra gli eventi saltati (B&B cap. 13)."""
    A, B = _track("a", 80, SECS, VOC), _track("b", 80, SECS, VOC, bpm=91.0)
    c, _ = _cand(A, B, 40, 8)
    ae = dje.event_plan(A, B, c)["a_event"]
    assert ae["kind"] == "chorus" and ae["span_bars"] == [24, 40] and ae["label_source"] == "allin1"
    assert ae["fate"] == "completed" and ae["on_section_boundary"]
    assert {"kind": "chorus", "span_bars": [56, 72]} in ae["skipped"]
    assert ae["skipped_counts"] == {"verse": 1, "chorus": 1, "outro": 1}
    # mid-section exit: the rest of the chorus is skipped
    mid = dje.a_event(A, 32, 32 * 4 * 60 / 90.0)
    assert mid["fate"] == "skipped_rest" and mid["skipped"][0] == {"kind": "chorus", "span_bars": [32, 40], "part": "rest"}
    # a verse running over the downbeat (dj_phrase.exit_point status mid_line) is a truncated line
    cut = dje.a_event(A, 40, 40 * 4 * 60 / 90.0, exit_info={"status": "mid_line", "reasons": ["x"]})
    assert cut["fate"] == "cut_mid_line" and cut["line_end"]["source"] == "dj_phrase.exit_point"


def test_b_pickup_of_two_beats_is_the_lead_in_and_first_vocal():
    """(b) B con pickup di 2 battiti reso da un phrase cut: B è udibile 2 battiti prima del suo downbeat."""
    A, B = _track("a", 80, SECS, VOC), _track("b", 80, SECS, VOC, bpm=90.0)
    c, _ = _cand(A, B, 40, 8)
    per = B["tempo"]["period_s"]
    plan = dict(c["plan"], phrase_switch={"pickup_s": 2 * per, "pickup_words": ["yeah", "uh"], "switch_rel_s": -2 * per})
    cand = dict(c, plan=plan)
    cut = {"type": "cut", "type_key": "cut", "bars": 0, "render": {}, "label": "taglio", "rule": "R6b_hiphop_cut"}
    entry = {"status": "pickup", "pickup_samples": int(round(2 * per * SR)), "pickup_words": ["yeah", "uh"]}
    doc = dje.event_plan(A, B, cand, decision=cut, entry_info=entry)
    li = doc["b_entry"]["lead_in"]
    assert li["kind"] == "pickup" and li["pickup_words"] == ["yeah", "uh"] and li["bars"] == pytest.approx(0.5)
    assert doc["b_entry"]["downbeat_sample"] - doc["b_entry"]["audible_sample"] == pytest.approx(2 * per * SR, abs=1)
    # the first word of B's line (the pickup) is the earliest event: it arrives when B becomes audible
    assert doc["b_target"]["kind"] == "first_vocal" and doc["delay"]["bars"] == pytest.approx(0.0, abs=1e-3)
    assert any(e["kind"] == "first_vocal" and e["bars_after_downbeat"] == pytest.approx(-0.5) for e in doc["b_events"])
    assert any(e["kind"] == "verse" and e["bars_after_downbeat"] == 0 for e in doc["b_events"])
    # an echo out does not play B's pickup: recorded as not played
    echo = dje.lead_in(ds._mk("echo_out", ds.values(ds.TREE_PARAMS)), c["plan"], A, B, entry)
    assert echo["kind"] == "echo_tail" and echo["b_pickup_not_played"] == ["yeah", "uh"]
    # ... and the unheard pickup cannot be B's target event: the verse on the downbeat is
    doc_e = dje.event_plan(A, B, c, decision=ds._mk("echo_out", ds.values(ds.TREE_PARAMS)), entry_info=entry)
    pick = next(e for e in doc_e["b_events"] if e["kind"] == "first_vocal")
    assert pick["audible"] is False and doc_e["b_target"]["kind"] == "verse" and doc_e["delay"]["bars"] == pytest.approx(0.0)


def test_b_chorus_eight_bars_after_entry_gives_the_delay():
    """(c) B entra sull'intro strumentale, il ritornello arriva 8 battute dopo: ritardo 8 battute (descrizione)."""
    secs_b = [(0, "intro"), (8, "chorus"), (24, "verse"), (40, "outro")]
    A = _track("a", 80, SECS, VOC)
    B = _track("b", 48, secs_b, [0.0] * 8 + [1.0] * 32 + [0.0] * 8, bpm=110.0)  # tempo out: no blend
    c, _ = _cand(A, B, 40, 0)
    doc = dje.event_plan(A, B, c)
    assert doc["choice"]["technique"] != "blend"
    assert doc["b_target"]["kind"] == "chorus" and doc["b_target"]["bar"] == 8
    assert doc["delay"]["bars"] == pytest.approx(8.0) and doc["delay"]["from_switch_bars"] == pytest.approx(8.0)
    assert doc["delay"]["s"] == pytest.approx(8 * 4 * 60 / 110.0, rel=1e-3)
    assert doc["event_option"]["preferred"] is False


def test_no_core_track_is_unknown():
    """(d) brano senza sezioni allin1 e senza core (dj_form.2+): kind 'unknown', core_status 'no_core'."""
    form = {"labels": ["unknown"] * 40, "core_status": "no_core", "drops": [], "active": {"kick": [False] * 40}}
    A = _track("a", 40, [], [0.0] * 40, form=form)
    B = _track("b", 80, SECS, VOC, form={"labels": ["core"] * 80, "core_status": "found", "drops": [8],
                                          "active": {"kick": [False] * 8 + [True] * 72}})
    ae = dje.a_event(A, 24, 24 * 4 * 60 / 90.0)
    assert ae["kind"] == "unknown" and ae["label_source"] == "dj_form"
    assert ae["form_at_exit"]["core_status"] == "no_core"
    ev = dje.b_events(B, 0, 0.0)
    assert any(e["kind"] == "chorus" or "dj_form:drop@8" in e["evidence"] for e in ev)
    drop8 = next(e for e in ev if e["bar"] == 8)
    # same bar: merged, the kick return wins over the verse start (target_priority), both kept as evidence
    assert drop8["kind"] == "drop" and "dj_form:drop@8" in drop8["evidence"] and "allin1:verse@8" in drop8["evidence"]


def test_event_plans_do_not_change_costs_or_choices():
    tracks = {s: _track(s, 80, SECS, VOC, bpm=90.0 + i) for i, s in enumerate("abc")}
    order = ["a", "b", "c"]
    base = {(x, y): ds.candidates_for_pair(tracks[x], tracks[y]) for x, y in zip(order, order[1:])}
    ev = {(x, y): ds.candidates_for_pair(tracks[x], tracks[y], event_plans=True) for x, y in zip(order, order[1:])}
    for k in base:
        a, b = base[k]["candidates"], ev[k]["candidates"]
        assert [(c["plan"]["a_out_bar"], c["plan"]["b_in_bar"], c["cost"], c["decision"]["type_key"]) for c in a] == \
               [(c["plan"]["a_out_bar"], c["plan"]["b_in_bar"], c["cost"], c["decision"]["type_key"]) for c in b]
        assert "event_context" not in a[0]["context"] and "event_context" in b[0]["context"] and "event_summary" in b[0]
    r0 = ds.choose_cues(order, tracks, base)
    r1 = ds.choose_cues(order, tracks, ev, event_fn=lambda A, B, c: {"preferred": True})  # w = 0: event_fn ignored
    assert r0["cost"] == r1["cost"] and "event_pref" not in r1
    assert [(c["plan"]["a_out_bar"], c["plan"]["b_in_bar"]) for c in r0["chosen"]] == \
           [(c["plan"]["a_out_bar"], c["plan"]["b_in_bar"]) for c in r1["chosen"]]
    assert ds.CUE_PARAMS["w_event_arrival"].value == 0.0  # experimental option off by default
    for c in r0["chosen"]:
        doc = dje.event_plan(tracks["a"], tracks["b"], c)
        assert doc["choice"]["cost"] == pytest.approx(c["cost"])


def test_experimental_option_prefers_completed_exit_and_near_target():
    """A esce a fine sezione in tutti i candidati migliori; a parità di costo il default fa entrare B sull'intro
    (strofa 8 battute dopo), l'opzione [H] sceglie l'ingresso sulla strofa (evento obiettivo sullo switch)."""
    A, B = _track("a", 80, SECS, VOC, bpm=90.0), _track("b", 80, SECS, VOC, bpm=110.0)  # tempo out: echo/cut only
    tracks = {"a": A, "b": B}
    pc = {("a", "b"): ds.candidates_for_pair(A, B)}
    r0 = ds.choose_cues(["a", "b"], tracks, pc)
    r1 = ds.choose_cues(["a", "b"], tracks, pc, params={"w_event_arrival": 0.05})
    c0, c1 = r0["chosen"][0], r1["chosen"][0]
    s0, s1 = dje.arrival_summary(A, B, c0), dje.arrival_summary(A, B, c1)
    assert c0["plan"]["b_in_bar"] == 0 and s0["b_target"] == "verse" and s0["from_switch_bars"] == 8 and not s0["preferred"]
    assert c1["plan"]["b_in_bar"] == 8 and s1["preferred"] and s1["a_fate"] == "completed" and s1["from_switch_bars"] == 0
    assert c1["cost"] - c0["cost"] < 0.05  # only near-equal technical costs are overturned
    assert r1["event_pref"]["chosen_preferred"] == [True]
    # a bigger cost gap is never overturned by the small weight
    r2 = ds.choose_cues(["a", "b"], tracks, pc, params={"w_event_arrival": 1e-9})
    assert r2["chosen"][0]["plan"]["b_in_bar"] in (0, 8)


def test_transition_context_events_are_additive():
    A, B = _track("a", 80, SECS, VOC), _track("b", 80, SECS, VOC, bpm=91.0)
    c, pc = _cand(A, B, 40, 8)
    ctx0 = ds.transition_context(A, B, c["plan"], pc["compatibility"])
    ctx1 = ds.transition_context(A, B, c["plan"], pc["compatibility"], events=True)
    assert {k: v for k, v in ctx1.items() if k != "event_context"} == ctx0
    ec = ctx1["event_context"]
    assert ec["a_kind"] == "chorus" and ec["a_fate"] == "completed" and ec["b_target"] == "verse"
    assert ds.decide_transition(ctx0) == ds.decide_transition(ctx1)
