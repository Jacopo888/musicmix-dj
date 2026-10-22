"""Ingresso di B (21/10/2026, conoscenza_mix/_meta/INGRESSO_B_2026-10-21.md).

dj_phrase.b_lead_in / entry_clear_test, dj_select.relocate_entry, dj_set.entry_guard (senza ripiego di default),
dj_render.add_b_lead_in su dj_effects, dj_qa.check_b_entry, ripianificazione di run_dj_set_v2_books.build_set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from musicmix import dj_phrase as P
from musicmix import dj_qa
from musicmix import dj_render as dj
from musicmix import dj_select as djs
from musicmix import dj_set as ds

ROOT = Path(__file__).resolve().parents[1]
SR = 48000
BEAT = 0.5


def _model(words, reliable=True):
    return {"sample_rate": SR, "period_s": BEAT, "asr_quality": {"reliable": reliable}, "params": {},
            "words": [{"on": int(a * SR), "off": int(b * SR), "text": t} for a, b, t in words]}


# ------------------------------------------------------------------ dj_phrase
def test_lead_in_keeps_a_pickup_split_by_a_short_pause():
    # "7" ... ".30 in the night": the phrase starts 1.7 s (3.4 beats) before D = 20 s, pause 0.36 s < 1 beat, ".30" crosses D
    m = _model([(10.0, 11.0, "before"), (18.30, 18.70, "7"), (19.06, 20.15, ".30"), (20.2, 20.4, "in"), (20.4, 20.8, "night")])
    r = P.b_lead_in(m, 20.0)
    assert r["status"] == "lead_in" and r["lead_in_words"] == ["7", ".30"]
    assert r["audible_sample"] <= int(18.30 * SR) - int(0.04 * SR)  # before the first word, with the margin
    assert r["audible_sample"] >= int(11.0 * SR)  # inside the rest after the previous phrase
    assert r["truncated_at_audible"] == []


def test_lead_in_on_downbeat_instrumental_and_mid_line():
    on = _model([(10.0, 11.0, "x"), (20.05, 20.5, "yo"), (20.5, 21.0, "go")])
    assert P.b_lead_in(on, 20.0)["status"] == "on_downbeat"
    inst = _model([(10.0, 11.0, "x"), (30.0, 31.0, "late")])
    assert P.b_lead_in(inst, 20.0)["status"] == "instrumental"
    # continuous rap from 16 s (gaps 0.1 s): the phrase began 2 bars before D -> mid_line
    words = [(16.0 + 0.4 * k, 16.3 + 0.4 * k, f"w{k}") for k in range(15)]
    r = P.b_lead_in(_model(words), 20.0)
    assert r["status"] == "mid_line"
    assert not P.entry_clear_test(_model(words))({"time_s": 20.0})["ok"]


def test_lead_in_line_end_with_a_noted_rest_starts_the_phrase():
    words = [(16.0 + 0.4 * k, 16.3 + 0.4 * k, f"w{k}") for k in range(10)] + [(20.15, 20.5, "new"), (20.5, 20.9, "line")]
    m = _model(words)
    # line end after w9 (ends 19.9 s, next word 20.15 s: rest of 0.25 s = half a beat) -> B may enter on D = 20 s
    m["gaps"] = [{"after": i, "p_end": 0.9 if i == 9 else 0.1} for i in range(len(words) - 1)]
    assert P.b_lead_in(m, 20.0)["status"] == "on_downbeat"
    m["gaps"][9]["p_end"] = 0.1  # same pause, no line end: still the same phrase -> mid_line
    assert P.b_lead_in(m, 20.0)["status"] == "mid_line"


def test_lead_in_abstains_without_asr_or_stem():
    r = P.b_lead_in(_model([(19.0, 20.5, "x")], reliable=False), 20.0)
    assert r["status"] == "unknown"


# ------------------------------------------------------------------ dj_select / dj_set
def _track(n_bars=40, bar_s=2.0, sections=((0, "intro"), (8, "verse"), (24, "chorus"))):
    starts = {b: lab for b, lab in sections}
    rows, lab = [], "intro"
    for j in range(n_bars):
        if j in starts:
            lab = starts[j]
        rows.append({"bar_index": j, "time_s": j * bar_s, "phrase_level": 16 if j in starts else (4 if j % 4 == 0 else 0),
                     "section_start": j in starts, "section_label": lab, "bar_in_section": 0, "bars_to_next_section": 8})
    return {"phrases": rows, "duration_s": n_bars * bar_s}


def test_relocate_entry_prefers_the_lead_bar_and_reports_tries(monkeypatch):
    monkeypatch.setattr(djs, "plan_transition", lambda A, B, out, inn, pair: {"a_out_bar": out["bar_index"], "b_in_bar": inn["bar_index"],
                                                                              "b_in_s": inn["time_s"]})
    B = _track()
    calls = []

    def test(c):
        calls.append(c["bar_index"])
        return {"ok": c["bar_index"] == 7, "status": "lead_in" if c["bar_index"] == 7 else "mid_line", "reasons": []}

    plan = {"a_out_bar": 20, "a_out_s": 40.0, "b_in_bar": 8, "b_in_s": 16.0}
    r = djs.relocate_entry(_track(), B, plan, {}, test)
    assert r["status"] == "moved" and r["to_bar"] == 7 and r["kind"] == "lead_bars"
    assert calls[:2] == [8, 7]
    assert djs.relocate_entry(_track(), B, plan, {}, lambda c: {"ok": True})["status"] == "clear"


def _pc(types=("echo_out", "echo_out")):
    cands = [{"plan": {"a_out_bar": 20, "b_in_bar": 8 + 4 * k, "a_out_s": 40.0, "b_in_s": 16.0 + 8 * k, "score": 1.0},
              "decision": {"type": t}, "cost": 1.0 + k, "admissible": True, "notes": []} for k, t in enumerate(types)]
    return {"compatibility": {}, "candidates": cands}


def test_entry_guard_leaves_no_admissible_candidate_by_default(monkeypatch):
    monkeypatch.setattr(djs, "relocate_entry", lambda *a, **k: {"status": "none", "entry": {"reasons": ["mid"]}, "tried": []})
    pc = ds.entry_guard(_pc(), {}, {}, lambda c: {"ok": False})
    assert not any(c["admissible"] for c in pc["candidates"])
    assert pc["entry_guard"]["no_admissible_entry"] and not pc["entry_guard"].get("relaxed")
    e = ds.edge_cost(pc, 90.0, 90.0, 0.0, 0.0, 0.0)
    assert e["total"] == float("inf") and not e["feasible"]
    # the old behaviour is only an explicit option
    pc2 = ds.entry_guard(_pc(), {}, {}, lambda c: {"ok": False}, relax_if_empty=True)
    assert all(c["admissible"] for c in pc2["candidates"]) and pc2["entry_guard"]["relaxed"]


def test_entry_guard_exempts_blends_and_keeps_clear_entries(monkeypatch):
    monkeypatch.setattr(djs, "relocate_entry", lambda A, B, plan, pair, t, **k: (
        {"status": "clear", "plan": plan, "entry": {"status": "lead_in", "lead_in_samples": 100}, "tried": []}))
    pc = ds.entry_guard(_pc(("blend", "echo_out")), {}, {}, lambda c: {"ok": True})
    assert pc["entry_guard"]["exempt"] == 1 and pc["entry_guard"]["clear"] == 1
    echo = next(c for c in pc["candidates"] if c["decision"]["type"] == "echo_out")
    assert echo["plan"]["b_lead_in"]["lead_in_samples"] == 100


# ------------------------------------------------------------------ dj_render.add_b_lead_in on dj_effects
def _click(bpm, dur, f_bass, gain=0.5):
    n = int(dur * SR)
    x = np.zeros(n)
    period = 60.0 / bpm
    beats = np.arange(0.5, dur - 0.2, period)
    for k, t in enumerate(beats):
        i = int(t * SR)
        x[i:i + 480] += (1.0 if k % 4 == 0 else 0.35) * np.exp(-np.arange(480) / 60.0) * np.sin(2 * np.pi * 2000 * np.arange(480) / SR)
    x += 0.2 * np.sin(2 * np.pi * f_bass * np.arange(n) / SR)
    return {"audio": np.stack([x, x], 1) * gain, "beats": beats, "downbeats": beats[::4]}


@pytest.fixture(scope="module")
def ab():
    pytest.importorskip("pyloudnorm")
    a, b = _click(120, 50, 55), _click(120, 50, 82)
    voc = np.zeros_like(b["audio"])
    d = b["downbeats"][6]
    i0, i1 = int((d - 0.75) * SR), int((d + 1.0) * SR)  # B "sings" from 0.75 s before its downbeat
    voc[i0:i1] = 0.3 * np.sin(2 * np.pi * 440 * np.arange(i1 - i0) / SR)[:, None]
    b["audio"] = b["audio"] + voc
    b["vocals"] = voc
    return a, b


def test_add_b_lead_in_rebuilds_the_render_and_plays_b_voice_before_its_downbeat(ab):
    from musicmix import dj_effects as fx

    a, b = ab
    kw = dict(beats_a=a["beats"], downbeats_a=a["downbeats"], beats_b=b["beats"], downbeats_b=b["downbeats"],
              cue_out_a=a["downbeats"][12], cue_in_b=b["downbeats"][6], sample_rate=SR)
    res = fx.echo_out(a["audio"], b["audio"], **kw, entry_bars=1, return_decks=True)
    same = dj.add_b_lead_in(res, b["audio"], 0, vocals_b=b["vocals"])
    assert same is res  # no lead-in: render untouched
    L = int(0.8 * SR)
    new = dj.add_b_lead_in(res, b["audio"], L, vocals_b=b["vocals"], mode="stem", a_close="auto")
    entry = int(res.metadata["switch"]["output_sample"])
    be = new.metadata["b_entry"]
    assert be["audible_output_sample"] == entry - L and be["downbeat_output_sample"] == entry
    assert new.metadata["switch"]["output_sample"] == entry  # clip rule unchanged
    seg_old = res.decks["b"][entry - L + 2400: entry - 2400]
    seg_new = new.decks["b"][entry - L + 2400: entry - 2400]
    assert np.sqrt(np.mean(seg_old ** 2)) < 1e-9 < np.sqrt(np.mean(seg_new ** 2))
    # after the downbeat B is the same as before (the vocal stem hands over to full B)
    assert np.allclose(new.decks["b"][entry + 100: entry + SR], res.decks["b"][entry + 100: entry + SR], atol=1e-9)
    with pytest.raises(dj.DJRenderError):
        dj.add_b_lead_in(res, b["audio"], int(10 * SR), vocals_b=b["vocals"])  # longer than the effect region


# ------------------------------------------------------------------ dj_qa.check_b_entry
def _geo(n=30 * SR, kind="echo_out"):
    return {"kind": kind, "is_cut": False, "sr": SR, "start": 10 * SR, "switch": 12 * SR, "end": 12 * SR, "n": n, "exit_effect": "echo"}


def test_check_b_entry_word_cut_phrase_and_pass():
    cfg = dj_qa.QAConfig()
    meta = {"b_entry": {"audible_output_sample": 12 * SR}, "stretch": {"b_bar_s": 2.0}}
    w_cut = [{"start_s": 11.8, "end_s": 12.3, "text": "thirty"}, {"start_s": 12.3, "end_s": 12.6, "text": "in"}]
    c = dj_qa.check_b_entry(_geo(), None, w_cut, None, cfg, meta)
    assert c["status"] == "fail" and c["problem"] == "b_word_cut" and c["coverage"] == "measured"
    w_mid = [{"start_s": 11.2, "end_s": 11.98, "text": "seven"}, {"start_s": 12.03, "end_s": 12.4, "text": "in"}]
    c = dj_qa.check_b_entry(_geo(), None, w_mid, None, cfg, meta)
    assert c["status"] == "fail" and c["problem"] == "b_phrase_started_before"
    w_ok = [{"start_s": 9.0, "end_s": 9.5, "text": "end"}, {"start_s": 12.05, "end_s": 12.4, "text": "start"}]
    assert dj_qa.check_b_entry(_geo(), None, w_ok, None, cfg, meta)["status"] == "pass"
    blend = dj_qa.check_b_entry(_geo(kind="blend"), None, w_cut, None, cfg, {"stretch": {"applied": True}})
    assert blend["status"] == "not_applicable"
    assert dj_qa.check_b_entry(_geo(), None, None, None, cfg, meta)["coverage"] == "unavailable"
    off = dj_qa.check_b_entry(_geo(), None, w_cut, None, dj_qa.QAConfig(b_entry_check="off"), meta)
    assert off["status"] == "not_applicable"
    assert "b_entry" in dj_qa.REQUIRED_ALWAYS


# ------------------------------------------------------------------ build_set: no kept fail, replanning
v2 = pytest.importorskip("run_dj_set_v2_books") if (sys.path.insert(0, str(ROOT / "tools")) or True) else None


def _fake_pipeline(monkeypatch, tmp_path, fail_arcs):
    """build_set with every heavy step replaced: 3 tracks a, b, c; renders fail on ``fail_arcs``."""
    slugs = ["a", "b", "c"]
    base = v2.base
    monkeypatch.setattr(v2.TF, "filter_sources", lambda s, log: s)
    monkeypatch.setattr(v2.v1, "load_json", lambda p: {"tempo": {"bpm": 90.0, "period_s": 0.66}, "key": {"camelot": "1A"}})
    monkeypatch.setattr(v2, "form_for", lambda *a, **k: {"provenance": None})
    monkeypatch.setattr(v2, "compact_form", lambda f: {})
    monkeypatch.setattr(v2.F, "form_summary", lambda f: {})
    monkeypatch.setattr(base, "sha256", lambda p: "0" * 64)
    monkeypatch.setattr(base, "rel", lambda p: str(p))
    monkeypatch.setattr(v2, "apply_filters", lambda pc, info, oo: pc)
    monkeypatch.setattr(v2.ds, "apply_far_genre", lambda pc, *a, **k: pc)
    monkeypatch.setattr(v2.ds, "far_timbre_calibration", lambda *a, **k: {"status": "off", "threshold": None})

    def cands(A, B, *a, **k):
        plan = {"a_out_s": 100.0, "a_out_bar": 50, "b_in_s": 20.0, "b_in_bar": 10, "type": "cut", "blend_bars": 0, "cue_score": 4.0,
                "score": 1.0, "confidence": "high", "a_section": "verse", "b_section": "verse", "reasons": []}
        dec = {"type": "echo_out", "type_key": "echo_out", "label": "echo", "rule": "R", "rule_source": "x", "render": {}, "fallbacks": []}
        return {"compatibility": {"reasons": [], "status": "compatible"}, "candidates": [
            {"plan": plan, "context": {}, "decision": dec, "cost": 1.0, "admissible": True, "notes": []}]}

    monkeypatch.setattr(v2.ds, "candidates_for_pair", cands)
    monkeypatch.setattr(v2.ds, "edge_cost", lambda pc, *a, **k: {"total": 1.0, "feasible": True})

    def order_set(sl, E, energy, oo):
        import itertools

        best = None
        for perm in itertools.permutations(range(len(sl))):
            c = sum(E[perm[i], perm[i + 1]] for i in range(len(perm) - 1))
            if np.isfinite(c) and (best is None or c < best[0]):
                best = (c, perm)
        if best is None:
            return {"order": [], "length": 0, "cost": float("inf"), "note": "infeasible"}
        return {"order": [sl[i] for i in best[1]], "length": len(sl), "cost": best[0], "note": "fake"}

    monkeypatch.setattr(v2.ds, "order_set", order_set)
    monkeypatch.setattr(v2.ds, "path_breakdown", lambda *a, **k: [])
    monkeypatch.setattr(v2.ds, "choose_cues", lambda order, an, pcs, *a, **k: {
        "chosen": [pcs[(order[t], order[t + 1])]["candidates"][0] for t in range(len(order) - 1)], "cost": 0.0, "play_s": [],
        "play_bounds_s": [0, 1], "relaxations": [], "first": None, "last": None})
    monkeypatch.setattr(v2.ds, "alternatives", lambda *a, **k: [])
    monkeypatch.setattr(v2.ds, "lead_a_s", lambda *a, **k: 0.0)
    monkeypatch.setattr(v2.dj_qa, "BeatThisTracker", lambda *a, **k: None)
    from types import SimpleNamespace

    monkeypatch.setattr(v2.v1, "Track", lambda slug, *a, **k: SimpleNamespace(slug=slug, provenance={"slug": slug}))
    monkeypatch.setattr(v2.ds, "clip_geometry", lambda meta, n: {"transition_start": 0, "switch": 0, "transition_end": 0, "b_rate": 1.0})
    monkeypatch.setattr(v2.ds, "placement_cost", lambda *a, **k: {})
    monkeypatch.setattr(v2.ds, "dense_overlap_cost", lambda *a, **k: 0.0)
    monkeypatch.setattr(v2.F, "overlap_profile", lambda *a, **k: {})
    monkeypatch.setattr(v2.ds, "first_start", lambda *a, **k: None)
    monkeypatch.setattr(v2.ds, "last_end", lambda *a, **k: None)

    class Res:
        metadata, audio = {}, np.zeros((10, 2))

    def qa(res, A, B, decks, tracker):
        bad = (A.slug, B.slug) in fail_arcs
        return {"verdict": "fail" if bad else "pass", "reasons": [{"check": "b_entry", "severity": "fail", "message": "cut"}] if bad else []}

    class Done(Exception):
        pass

    def assemble(order, *a, **k):
        raise Done(order)

    monkeypatch.setattr(base, "assemble", assemble)
    infos = {s: {"timbre": [0.0, 1.0], "energy_features": {"onset_rate": i, "flux_mean": i, "centroid_mean": i}} for i, s in enumerate(slugs)}
    H = v2.SetHooks(pool=lambda log: {s: {"plan_dir": tmp_path, "source": "fake", "catalog": {}} for s in slugs},
                    info=lambda s, src: infos[s], render=lambda dec, A, B, plan, prov: (Res(), None), run_qa=qa,
                    lead_a_s=lambda *a: 0.0, overlap_b_s=lambda *a: 0.0, compare_v1b=False)
    return H, Done


def test_build_set_replans_instead_of_keeping_a_fail(monkeypatch, tmp_path):
    H, Done = _fake_pipeline(monkeypatch, tmp_path, fail_arcs={("a", "b")})
    with pytest.raises(Done) as exc:
        v2.build_set(tmp_path / "run", hooks=H)
    order = exc.value.args[0]
    assert ("a", "b") not in set(zip(order, order[1:]))  # the failed arc was removed and the order redone


def test_build_set_declares_a_set_that_does_not_close(monkeypatch, tmp_path):
    H, Done = _fake_pipeline(monkeypatch, tmp_path, fail_arcs={(x, y) for x in "abc" for y in "abc" if x != y})
    assert v2.build_set(tmp_path / "run", hooks=H) == 2
    doc = json.loads((tmp_path / "run" / "set.json").read_text())
    assert doc["closed"] is False and doc["not_closed"]["banned_arcs"]
    assert "transitions" not in doc  # no kept fail, no audio
