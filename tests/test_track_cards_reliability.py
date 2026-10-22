"""Schede brano (musicmix.track_cards) e affidabilità/copertura/varietà (musicmix.reliability)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from musicmix import dj_phrase as P
from musicmix import dj_select as djs
from musicmix import reliability as R
from musicmix import track_cards as TC

ROOT = Path(__file__).resolve().parents[1]
SR = 48000
BPM = 120.0
BAR = 4 * 60.0 / BPM


def _analysis(n_bars=48, bpm=BPM, confidence="high", agreement=None, octave=False, votes=None, residual=4.0, fit=1.0,
              key_level="high", vocal=None):
    bar = 4 * 60.0 / bpm
    downs = np.arange(n_bars) * bar
    secs_spec = ((0, "intro"), (8, "verse"), (24, "chorus"), (40, "outro"))
    secs = [{"label": lab, "start_s": b * bar, "end_s": 0.0, "bar_index": b, "downbeat_s": b * bar, "snap_error_bars": 0.0,
             "loose": False, "support": ["x"], "source": "test"} for b, lab in secs_spec]
    vocal = [0.0] * 8 + [1.0] * 32 + [0.0] * (n_bars - 40) if vocal is None else vocal
    return {"slug": "t", "track_id": "abc", "duration_s": n_bars * bar,
            "tempo": {"bpm": bpm, "provider": "allin1", "confidence": confidence, "beat_f": fit, "downbeat_f": fit,
                      "residual_ms_median_abs": residual, "period_s": 60.0 / bpm, "phase_s": 0.0, "bar_offset_beats": 0,
                      "providers": {}, "agreement": agreement or {"madmom": {"bpm_ratio": 1.0, "same_tempo_octave": True,
                                                                              "beat_coincidence": 1.0, "downbeat_coincidence": 1.0}},
                      "octave_ambiguous": octave, "octave_votes": votes},
            "grid": {"downbeats_s": downs.tolist(), "n_bars": n_bars},
            "key": {"key": "A minor", "camelot": "8A", "source": "madmom_cnn", "confidence": 0.9 if key_level == "high" else 0.2,
                    "level": key_level, "estimates": {"madmom_cnn": {"camelot": "8A"}, "essentia_edma": {"camelot": "8A" if key_level == "high" else "3A"},
                                                      "krumhansl": {"camelot": "8A" if key_level == "high" else "5B"}}},
            "sections_source": "test", "sections": secs, "phrases": djs.phrase_table(downs, secs),
            "bars": {"vocal": vocal, "vocal_last_beat": vocal, "energy_db_rel": [0.0] * n_bars},
            "vocal_regions": [{"start_s": 8 * bar, "end_s": 40 * bar, "bars": 32, "first_bar": 8}]}


ENTRY = {"slug": "t", "artist": "X", "title": "T", "track_id": "abc"}
INPUTS = {"analysis": {"path": "a.json", "sha256": "1"}}


def _card(analysis=None, **kw):
    kw.setdefault("language", {"language": "en", "admitted": True, "rule": "language_allowed"})
    kw.setdefault("stems", {"vocals": {"ok": True}})
    return TC.build_card(slug="t", entry=ENTRY, analysis=analysis or _analysis(), inputs=INPUTS, audio={"sha256": "abc", "verified": True}, **kw)


# ------------------------------------------------------------------ cards
def test_card_schema_and_flags_without_phrase_model():
    c = _card()
    json.dumps(c)
    assert c["schema_version"] == TC.CARD_SCHEMA and c["audio"]["sha256"] == "abc"
    for k in ("grid", "key", "structure", "voice", "entries", "exits", "usable", "instrumental_regions", "stems", "stretch",
              "uncertain", "corrections", "reliability", "input_key"):
        assert k in c, k
    assert sorted(TC.uncertain_summary(c)) == ["form_missing", "phrase_model_missing"]  # clean grid, confident key, no form
    assert all(r["status"] == "unknown" for r in c["entries"] + c["exits"])
    assert c["instrumental_regions"][0] == {**c["instrumental_regions"][0], "start_bar": 0, "end_bar": 8, "bars": 8}
    assert c["reliability"]["planning"] == pytest.approx(0.7 * 0.95) and c["reliability"]["components"]["phrase"] == pytest.approx(0.7)


def test_grid_flags_octave_suspect_and_disputed_phase():
    a = _analysis(bpm=65.6, confidence="medium", octave=True, votes={"65.58": ["allin1"], "131.14": ["madmom"]}, residual=35.0,
                  agreement={"madmom": {"bpm_ratio": 2.0, "same_tempo_octave": False, "beat_coincidence": 1.0, "downbeat_coincidence": 0.0},
                             "beat_this": {"bpm_ratio": 1.0, "same_tempo_octave": True, "beat_coincidence": 0.2, "downbeat_coincidence": 1.0}},
                  key_level="low")
    c = _card(a)
    ids = TC.uncertain_summary(c)
    for f in ("tempo_octave_suspect", "grid_residual_high", "downbeat_phase_disputed", "beat_phase_disputed",
              "key_confidence_low", "key_estimators_disagree"):
        assert f in ids, f
    assert c["grid"]["downbeat_phase_disputed_by"] == ["madmom"] and c["grid"]["beat_phase_disputed_by"] == ["beat_this"]
    assert c["stretch"]["overlap_recommended"] is False
    # planning ignores the key, value does not
    assert c["reliability"]["planning"] > c["reliability"]["value"]
    assert c["reliability"]["planning"] < R.values()["track_min"]
    # 90 BPM half/double ambiguity inside the plausible range is not a suspect
    c2 = _card(_analysis(bpm=90.0, octave=True, votes={"90": ["allin1"], "180": ["madmom"]}))
    assert "tempo_octave_suspect" not in TC.uncertain_summary(c2)


def test_language_and_asr_flags():
    words = [{"text": "w", "confidence": 0.3}] * 30 + [{"text": "w", "confidence": 0.9}] * 70
    c = _card(vocals_doc={"words": words, "backend": {"language": "it"}},
              language={"language": "it", "admitted": True, "rule": "language_allowed"},
              phrase={"sample_rate": SR, "period_s": 0.5, "downbeats_s": _analysis()["grid"]["downbeats_s"], "lang": "en",
                      "asr_quality": {"reliable": True}, "words": [], "gaps": [], "lines": [], "params": {}, "summary": {"n_words": 0}})
    ids = TC.uncertain_summary(c)
    assert "asr_language_mismatch" in ids and "asr_low_confidence" in ids and "stem_activity_missing" in ids
    c3 = _card(language={"language": "de", "admitted": False, "rule": "language_not_allowed", "why": "de"})
    assert "language_not_allowed" in TC.uncertain_summary(c3) and c3["reliability"]["planning"] == 0.0


def test_entries_from_phrase_model_and_exits(monkeypatch):
    a = _analysis()
    # rap continuo dalla battuta 8 alla 40 (pause brevi): ingressi a metà frase; dopo la battuta 40 niente voce
    words = [{"on": int((8 * BAR + 0.4 * k) * SR), "off": int((8 * BAR + 0.4 * k + 0.3) * SR), "text": f"w{k}"} for k in range(int(32 * BAR / 0.4))]
    model = {"sample_rate": SR, "period_s": 0.5, "downbeats_s": a["grid"]["downbeats_s"], "lang": "en", "asr_quality": {"reliable": True},
             "params": {}, "words": words, "lines": [], "summary": {"n_words": len(words)}}
    monkeypatch.setattr(P, "exit_point", lambda m, t, p=None, act=None: {"status": "mid_line" if 8 * BAR < t < 40 * BAR else "instrumental",
                                                                          "p_line_end": None, "reasons": []})
    c = _card(a, phrase=model)
    by = {r["bar"]: r for r in c["entries"]}
    assert by[0]["ok"] and by[0]["status"] == "instrumental"
    assert by[16]["status"] == "mid_line" and not by[16]["ok"]
    assert 16 not in c["usable"]["entries"] and 0 in c["usable"]["entries"]
    ex = {r["bar"]: r for r in c["exits"]}
    assert all(not r["line_end"] for b, r in ex.items() if 8 < b < 40)
    assert set(c["usable"]["exits_mid_line"]) == {b for b in ex if 8 < b < 40}


def test_corrections_verify_exclude_block_and_stale():
    a = _analysis(bpm=65.6, octave=True, votes={"65.6": ["allin1"], "131.2": ["madmom"]})
    base = _card(a)
    assert "tempo_octave_suspect" in TC.uncertain_summary(base)
    corr = [{"id": "t-1", "kind": "verify_flag", "flag": "tempo_octave_suspect", "audio_sha256": "abc", "why": "ascoltato"},
            {"id": "t-2", "kind": "block_entry", "bar": 0, "audio_sha256": "abc", "why": "intro troppo vuota"},
            {"id": "t-3", "kind": "exclude", "audio_sha256": "other", "why": "altro audio"},
            {"id": "t-4", "kind": "bpm_octave", "value": "double", "why": "da rianalizzare"}]
    c = _card(a, corrections=corr)
    assert "tempo_octave_suspect" not in TC.uncertain_summary(c) and "tempo_octave_suspect" in TC.uncertain_summary(c, True)
    assert next(r for r in c["entries"] if r["bar"] == 0)["ok"] is False and 0 not in c["usable"]["entries"]
    assert [x["id"] for x in c["corrections"]["stale"]] == ["t-3"] and not c["excluded"]
    assert "correction_pending_reanalysis" in TC.uncertain_summary(c)
    assert c["input_key"] != base["input_key"]
    ex = _card(a, corrections=[{"id": "t-5", "kind": "exclude", "audio_sha256": "abc", "why": "no"}])
    assert ex["excluded"] and ex["reliability"]["value"] == 0.0 and ex["reliability"]["planning"] == 0.0


def test_input_key_changes_with_inputs():
    k1 = TC.input_key({"analysis": {"sha256": "1"}})
    assert k1 == TC.input_key({"analysis": {"sha256": "1"}})
    assert k1 != TC.input_key({"analysis": {"sha256": "2"}})
    assert k1 != TC.input_key({"analysis": {"sha256": "1"}}, [{"id": "x"}])


# ------------------------------------------------------------- reliability
def _cardpair():
    clean = _card()
    clean["slug"] = "a"
    risky = _card(_analysis(confidence="low", fit=0.6, key_level="low"))
    risky["slug"] = "b"
    for c in (clean, risky):
        for r in c["exits"]:
            r["status"] = "mid_line" if r["bar"] == 32 else "clean"
        for r in c["entries"]:
            r["status"], r["ok"] = "on_downbeat", True
    return clean, risky


def test_cue_reliability_exit_state_overlap_and_relocated_entry():
    a, b = _cardpair()
    echo = {"type": "echo_out"}
    clean = R.cue_reliability(a, a, {"a_out_bar": 24, "b_in_bar": 8}, echo)
    mid = R.cue_reliability(a, a, {"a_out_bar": 32, "b_in_bar": 8}, echo)
    assert clean["value"] > mid["value"] and mid["factors"]["exit"] == R.values()["exit_factor"]["mid_line"]
    risky_b = R.cue_reliability(a, b, {"a_out_bar": 24, "b_in_bar": 8}, echo)
    assert risky_b["value"] < clean["value"] and "key_b" not in risky_b["factors"]  # key only with overlap
    blend = R.cue_reliability(a, b, {"a_out_bar": 24, "b_in_bar": 8}, {"type": "blend"}, stretch_pct=7.0)
    assert "key_b" in blend["factors"] and "entry" not in blend["factors"]
    assert sum(k.startswith("stretch") for k in blend["factors"]) <= 1
    moved = R.cue_reliability(a, a, {"a_out_bar": 24, "b_in_bar": 7, "b_lead_in": {"status": "lead_in"}}, echo)
    assert moved["factors"]["entry"] == R.values()["entry_factor"]["lead_in"]
    unknown = R.cue_reliability(a, a, {"a_out_bar": 24, "b_in_bar": 7}, echo)
    assert unknown["factors"]["entry"] == R.values()["unknown_cue_factor"]


def _pc(*cands):
    out = []
    for i, (ab, bb, cost) in enumerate(cands):
        out.append({"plan": {"a_out_bar": ab, "b_in_bar": bb, "score": 1.0, "a_out_s": ab * BAR, "b_in_s": bb * BAR},
                    "decision": {"type": "echo_out", "type_key": "echo_out", "rule": "R"}, "admissible": True, "cost": cost, "notes": []})
    return {"candidates": out, "compatibility": {"tempo": {"stretch_b_pct": 0.0}}}


def test_hook_report_changes_nothing_prefer_reorders_and_avoids():
    a, b = _cardpair()
    an = {"a": _analysis(), "b": _analysis()}
    rep = R.ReliabilityHook({"a": a, "b": b}, mode="report")
    pc = rep.adjust_pair(_pc((32, 8, 0.10), (24, 8, 0.12)), "a", "a", {"a": an["a"]})
    assert [c["cost"] for c in pc["candidates"]] == [0.10, 0.12] and all(c["admissible"] for c in pc["candidates"])
    pre = R.ReliabilityHook({"a": a, "b": b}, mode="prefer")
    pc = pre.adjust_pair(_pc((32, 8, 0.10), (24, 8, 0.12)), "a", "a", {"a": an["a"]})
    assert pc["candidates"][0]["plan"]["a_out_bar"] == 24  # the mid-line exit is now more expensive
    assert pre.pairs[("a", "a")]["best_before"] != pre.pairs[("a", "a")]["best_after"]
    # a track below track_min: every candidate of its pairs is inadmissible
    bad = _card(_analysis(bpm=65.6, octave=True, votes={"65.6": ["x"], "131.2": ["y"]}, residual=40.0, confidence="low"))
    bad["slug"] = "z"
    hk = R.ReliabilityHook({"a": a, "z": bad}, mode="prefer")
    pc = hk.adjust_pair(_pc((24, 8, 0.1)), "a", "z", {"a": an["a"], "z": _analysis(bpm=65.6)})
    assert not any(c["admissible"] for c in pc["candidates"]) and hk.tracks["z"]["avoid"]
    doc = hk.report({"order": ["a", "z"], "chosen": pc["candidates"][:1]})
    json.dumps(doc)
    assert doc["tracks_avoided"] == ["z"] and doc["pairs"]["made_unplannable"] == 1
    # a card on another grid is stale: only its slug is reported, its tables are not used
    hk2 = R.ReliabilityHook({"a": a}, mode="prefer")
    hk2.adjust_pair(_pc((24, 8, 0.1)), "a", "a", {"a": _analysis(n_bars=40)})
    assert "a" in hk2.stale


def test_novelty_penalises_repeated_pairs_only_when_on():
    a, _ = _cardpair()
    hist = [{"order": ["a", "q"], "pairs": [["a", "q"]]}]
    off = R.ReliabilityHook({"a": a, "q": a}, mode="prefer", history=hist)
    on = R.ReliabilityHook({"a": a, "q": a}, mode="prefer", history=hist, params=R.NOVELTY_ON)
    c_off = off.adjust_pair(_pc((24, 8, 0.1)), "a", "q", {})["candidates"][0]["cost"]
    c_on = on.adjust_pair(_pc((24, 8, 0.1)), "a", "q", {})["candidates"][0]["cost"]
    assert c_on - c_off == pytest.approx(R.NOVELTY_ON["w_pair_repeat"] + R.NOVELTY_ON["w_track_repeat"])


def test_coverage_variety_and_compare():
    s1 = {"name": "s1", "order": ["a", "b", "c"], "pairs": [["a", "b"], ["b", "c"]],
          "transitions": [{"a": "a", "b": "b", "technique": "echo_out", "a_out_bar": 1, "b_in_bar": 2},
                          {"a": "b", "b": "c", "technique": "echo_out", "a_out_bar": 3, "b_in_bar": 4}]}
    s2 = {"name": "s2", "order": ["a", "b", "d"], "pairs": [["a", "b"], ["b", "d"]],
          "transitions": [{"a": "a", "b": "b", "technique": "echo_out", "a_out_bar": 1, "b_in_bar": 2},
                          {"a": "b", "b": "d", "technique": "blend4", "a_out_bar": 3, "b_in_bar": 4}]}
    cov = R.coverage([s1, s2], ["a", "b", "c", "d", "e"])
    assert [r["cumulative_used"] for r in cov["per_set"]] == [3, 4] and cov["never_used"] == ["e"]
    assert cov["per_set"][1]["new_tracks"] == ["d"]
    var = R.variety([s1, s2], techniques_known=["echo_out", "blend4", "cut"])
    assert var["per_set"][0]["same_technique_twice_in_a_row"] == 1 and var["per_set"][0]["technique_entropy"] == 0.0
    assert var["per_set"][1]["pairs_repeated_from_previous_sets"] == 1 and var["per_set"][1]["transitions_repeated_from_previous_sets"] == 1
    assert var["per_set"][1]["tracks_jaccard_vs_previous"] == pytest.approx(0.5) and var["techniques_never_used"] == ["cut"]
    cmp = R.compare_plans({"ordering": {"order": s1["order"]}, "reliability": {"transitions": [
        {"a": x["a"], "b": x["b"], "type": x["technique"], "a_out_bar": x["a_out_bar"], "b_in_bar": x["b_in_bar"]} for x in s1["transitions"]]}},
        {"ordering": {"order": s2["order"]}, "reliability": {"transitions": [
            {"a": x["a"], "b": x["b"], "type": x["technique"], "a_out_bar": x["a_out_bar"], "b_in_bar": x["b_in_bar"]} for x in s2["transitions"]]}})
    assert cmp["tracks_removed"] == ["c"] and cmp["tracks_added"] == ["d"] and cmp["n_same_pairs"] == 1


# ------------------------------------------------------------------ tools
def test_runner_hook_is_off_by_default():
    sys.path.insert(0, str(ROOT / "tools"))
    import run_dj_set_v2_books as v2

    assert v2.SetHooks().reliability is None


@pytest.mark.data("runs/dj_set_v3_bridge/dj_plan/tracks/gods_plan.json", "runs/pool_bridge_v1/catalog_ready.json")
@pytest.mark.skipif(not (ROOT / "runs/dj_set_v3_bridge/dj_plan/tracks/gods_plan.json").is_file()
                    or not (ROOT / "runs/pool_bridge_v1/catalog_ready.json").is_file(), reason="catalogo del ponte assente")
def test_builder_on_a_real_bridge_track_is_incremental(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    import build_track_cards as B

    args = type("A", (), {"out": str(tmp_path), "catalog": None, "only": ["gods_plan"], "force": False, "compute_phrase": False})()
    idx = B.build(args)
    assert [r["status"] for r in idx["tracks"]] == ["built"]
    card = json.loads((tmp_path / "gods_plan.json").read_text())
    assert card["audio"]["verified"] is True and card["inputs"]["analysis"]["sha256"]
    assert card["usable"]["entries"] and "key_confidence_low" in TC.uncertain_summary(card)
    assert B.build(args)["tracks"][0]["status"] == "unchanged"
