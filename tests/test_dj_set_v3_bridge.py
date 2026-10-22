"""Helper di tools/run_dj_set_v3_bridge.py (set v3 sul catalogo del ponte, galleria A/B cieca)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

v3 = pytest.importorskip("run_dj_set_v3_bridge")
SR = v3.SR


def test_trim_to_rule_cut_and_effect():
    x = np.zeros((int(40 * SR), 2), dtype=np.float32)
    clip, tm = v3.trim_to_rule(x, "cut", 15.0, 15.0, 15.0)
    assert len(clip) == 20 * SR and tm["sw"] == pytest.approx(10.0) and tm["rule_ok"]
    clip, tm = v3.trim_to_rule(x, "blend", 12.0, 14.0, 18.0)  # 10 s + 6 s of transition + 10 s
    assert len(clip) == 26 * SR and tm["t0"] == pytest.approx(10.0) and tm["t1"] == pytest.approx(16.0) and tm["rule_ok"]
    clip, tm = v3.trim_to_rule(x, "echo_out", 8.0, 9.0, 9.0)  # only 8 s before the transition: flagged, not padded
    assert not tm["rule_ok"] and tm["t0"] == pytest.approx(8.0)


def test_signature_separates_phrase_cut_and_techniques():
    plan = {"phrase_switch": {"switch_rel_s": -0.4, "tail_end_a_s": None}}
    cut = {"type": "cut", "type_key": "cut", "render": {}}
    assert v3.signature(cut, plan) != v3.signature(cut, {})
    assert v3.signature(cut, {"phrase_switch": {"switch_rel_s": 0.0, "tail_end_a_s": None}}) == v3.signature(cut, {})
    b4 = {"type": "blend", "type_key": "blend4", "render": {"bars": 4}}
    b8 = {"type": "blend", "type_key": "blend8", "render": {"bars": 8}}
    assert v3.signature(b4, plan) != v3.signature(b8, plan)


def test_creative_spacing_and_decision():
    A = {"tempo": {"period_s": 0.5}}
    p = {"technique": "halving_build", "params": {"build_bars": 4, "gap_beats": 1}, "duration_bars": 4, "rule": "C2b_halving",
         "rule_source": "Margulis 2014 p. 23", "why": ["x"], "prior": 0.7, "score": 0.8, "risks": [], "prior_source": "s"}
    d = v3.creative_decision(p)
    assert d["creative"] and d["type"] == "halving_build" and d["bars"] == 4
    assert v3.lead_a_v3({"decision": d}, A) == pytest.approx(4 * 4 * 0.5)
    brk = v3.creative_decision(dict(p, technique="break_before_entry", params={"mode": "stop", "beats": 1}, duration_bars=0.25, rule="C8_break"))
    assert brk["bars"] == 1 and v3.lead_a_v3({"decision": brk}, A) == pytest.approx(0.5)
    loop = v3.creative_decision(dict(p, technique="loop_extend", params={"loop_bars": 1, "bars": 8}, duration_bars=8, rule="C4_extend"))
    assert v3.overlap_b_v3({"decision": loop}, A) == pytest.approx(8 * 4 * 0.5)


def test_drop_cut_falls_back_to_first_non_cut():
    c = {"decision": {"type": "cut", "type_key": "cut", "rule": "R1", "fallbacks": [{"type": "cut"}, {"type": "echo_out", "type_key": "echo_out", "render": {}}]},
         "notes": []}
    v3._drop_cut(c, "fine verso assente")
    assert c["decision"]["type"] == "echo_out" and c["decision"]["rule"].endswith("+phrase_guard") and c["notes"]


def test_attempts_policy_first_only_for_context_rules(monkeypatch):
    def fake(props):
        monkeypatch.setattr(v3, "policy_for", lambda *a, **k: {"proposals": props})

    mk = lambda tech, rule, sc: {"technique": tech, "rule": rule, "score": sc, "why": ["w"], "params": {}, "duration_bars": 4,
                                 "rule_source": "src", "prior": sc, "risks": [], "prior_source": "s"}
    ctx = {"t": 0, "a": "a", "b": "b", "idx0": 3, "chosen": {"plan": {}}, "pc": {"compatibility": {}}, "recent": [],
           "analyses": {"a": {}, "b": {}}, "plan": [(3, {"type": "cut"}, "albero")]}
    fake([mk("echo_freeze", "C7_default", 0.6), mk("break_before_entry", "C8_break", 0.55)])
    out = v3.attempts_v3(dict(ctx))
    assert [x[1]["type"] for x in out] == ["cut"] and not v3.POLICY[0]["used_policy_first"]
    fake([mk("halving_build", "C2b_halving", 0.8), mk("echo_freeze", "C7_default", 0.6), mk("break_before_entry", "C8_break", 0.55)])
    out = v3.attempts_v3(dict(ctx))
    assert [x[1]["type"] for x in out] == ["halving_build", "break_before_entry", "cut"]


def test_planned_decision_matches_the_first_render_attempt(monkeypatch):
    """--plan-only (piano per eventi) usa la stessa tecnica che il set prova per prima (attempts_v3)."""
    mk = lambda tech, rule, sc: {"technique": tech, "rule": rule, "score": sc, "why": ["w"], "params": {"build_bars": 2}, "duration_bars": 2,
                                 "rule_source": "src", "prior": sc, "risks": [], "prior_source": "s"}
    tree = {"type": "echo_out", "type_key": "echo_out", "rule": "R4_key_no_blend", "fallbacks": []}
    cand = {"plan": {}, "decision": tree}
    pc = {"compatibility": {}}
    for props in ([mk("build_into_chorus", "C2_chorus", 0.85), mk("halving_build", "C2b_halving", 0.8)],
                  [mk("loop_extend", "C4_extend", 0.75), mk("halving_build", "C2b_halving", 0.7)],
                  [mk("echo_freeze", "C7_default", 0.6)]):
        monkeypatch.setattr(v3, "policy_for", lambda *a, **k: {"proposals": props})
        dec, _, why = v3.planned_decision(0, "a", "b", cand, pc, {"a": {}, "b": {}}, [])
        ctx = {"t": 0, "a": "a", "b": "b", "idx0": 0, "chosen": cand, "pc": pc, "recent": [], "analyses": {"a": {}, "b": {}},
               "plan": [(0, tree, "albero")]}
        first = v3.attempts_v3(ctx)[0][1]
        assert dec["type"] == first["type"] and dec.get("rule") == first.get("rule"), why
        rows = v3._policy_rows({"proposals": props}, dec)
        assert len(rows) == len(props) and all(r["why_not"] for r in rows)
    assert any("NOT_SPLICEABLE" in r["why_not"] for r in v3._policy_rows({"proposals": [mk("loop_extend", "C4_extend", 0.7)]}, tree))


def test_declare_entry_relocation_feeds_phrase_check(tmp_path):
    """R-B2 «una battuta prima»: il runner dichiara lo spostamento e dj_qa.check_phrase (.7) lo dà warn, non fail."""
    import json
    from types import SimpleNamespace

    from musicmix import dj_qa

    bar = 2.0
    rows = [{"bar_index": j, "time_s": 1.0 + j * bar, "phrase_level": (8 if j % 8 == 0 else 0), "section_start": j % 16 == 0}
            for j in range(32)]
    an = tmp_path / "b.json"
    an.write_text(json.dumps({"phrases": rows}))
    prov = {"b": {"slug": "nonexistent_slug_for_test", "analysis": str(an)}}
    meta = {"cue_in_b_s": 1.0 + 7 * bar, "provenance": prov, "b_entry": {"audible_output_sample": 123}}
    res = SimpleNamespace(metadata=meta)
    assert dj_qa.check_phrase({"sr": SR}, meta, dj_qa.QAConfig())["status"] == "fail"
    v3.declare_entry_relocation(res, {"entry_relocation": {"from_bar": 8, "to_bar": 7, "kind": "lead_bars", "shift_bars": -1}})
    assert meta["b_entry"]["relocated_bars"] == 1 and meta["b_entry"]["audible_output_sample"] == 123
    assert meta["b_entry"]["relocation"]["kind"] == "lead_bars"
    assert dj_qa.check_phrase({"sr": SR}, meta, dj_qa.QAConfig())["status"] == "warn"
    # un altro cue di frase (cue_candidate) è solo provenienza; senza lead-in b_entry nasce dal nulla
    meta2 = {"cue_in_b_s": 1.0 + 16 * bar}
    v3.declare_entry_relocation(SimpleNamespace(metadata=meta2),
                                {"entry_relocation": {"from_bar": 8, "to_bar": 16, "kind": "cue_candidate", "shift_bars": 8}})
    assert "relocated_bars" not in meta2["b_entry"] and meta2["b_entry"]["relocation"]["shift_bars"] == 8
    meta3 = {}
    v3.declare_entry_relocation(SimpleNamespace(metadata=meta3), {"a_out_s": 1.0})
    assert meta3 == {}
