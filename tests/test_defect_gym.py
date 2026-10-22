"""Palestra dei difetti, versione rapida (suite normale): operatori, classificazione appaiata, gate.

Audio sintetico (le tracce di ``test_dj_qa``), render vero di ``dj_render`` e ``dj_qa`` senza Beat This!: pochi
secondi. La palestra completa sulle transizioni reali è ``tests/test_defect_gym_gate.py`` (``-m gym``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyloudnorm")
pytest.importorskip("scipy")

from musicmix import defect_gym as G  # noqa: E402
from musicmix import dj_qa, dj_render as dj  # noqa: E402
from test_dj_qa import synth_track  # noqa: E402

SR = 48000
BEAT = 0.5


def _grid(n_s):
    beats = np.arange(0.0, n_s, BEAT)
    return beats, beats[::4]


def _cut_bundle():
    a = synth_track(40.0, seed=0, bass_hz=55.0, stab_hz=600.0)
    b = synth_track(40.0, seed=1, bass_hz=49.0, stab_hz=800.0)
    beats, downs = _grid(40.0)
    res = dj.render_transition(a, b, beats_a=beats, downbeats_a=downs, beats_b=beats, downbeats_b=downs, cue_out_a=20.0,
                               cue_in_b=10.0, kind="cut", sample_rate=SR, return_decks=True)
    meta = res.metadata
    off = int(meta["switch"]["a_source_sample"]) - int(meta["switch"]["output_sample"])
    ga = 10 ** (meta["loudness"]["a_gain_db"] / 20)

    def a_source(s0, s1):
        x = np.zeros((s1 - s0, 2))
        lo, hi = max(0, s0 + off), min(len(a), s1 + off)
        x[lo - (s0 + off):hi - (s0 + off)] = a[lo:hi]
        return x * ga

    comps = {k: res.decks[k] for k in ("a", "b")}
    return G.Bundle(audio=np.asarray(res.audio, dtype=np.float64), sr=SR, metadata=meta, components=comps, qa_decks=res.decks,
                    critic_decks=res.decks, kind="cut", a_source=a_source, beat_s=BEAT)


def _qa(b: G.Bundle) -> dict:
    return dj_qa.evaluate_render(b.audio, b.metadata, sample_rate=b.sr, decks=b.qa_decks, tracker=None,
                                 config=dj_qa.QAConfig(dropout_check=True))


@pytest.fixture(scope="module")
def ref():
    return _cut_bundle()


def test_rebuild_reproduces_the_render(ref):
    assert G.rebuild_error(ref) < 1e-5


def test_copy_is_independent(ref):
    v = ref.copy()
    G.apply_deck_defect(v, "loud_jump", "+6dB")
    assert not np.allclose(v.audio, ref.audio)
    assert ref.metadata["loudness"]["b_gain_db"] == pytest.approx(v.metadata["loudness"]["b_gain_db"] - 6.0)
    # deck condivisi tra componenti e deck del QA: la copia li tiene condivisi, il riferimento resta intatto
    assert v.components["b"] is v.qa_decks["b"]
    assert ref.components["b"] is ref.qa_decks["b"]


def test_deck_defects_are_detected_against_the_reference(ref):
    q_ref = _qa(ref)
    sig_ref = G.qa_signals(q_ref)
    out = {}
    for d, lv in (("loud_jump", "+6dB"), ("gap", "1beat"), ("double_bass", "2bars")):
        v = ref.copy()
        info = G.apply_deck_defect(v, d, lv)
        assert info["applied"], info
        q = _qa(v)
        out[d] = G.classify(G.qa_signals(q), sig_ref, G.DEFECTS[d]["expect"], G.qa_coverage(q))
    assert out["loud_jump"]["status"] == "detected" and "qa:loudness_jump" in out["loud_jump"]["new_fail"]
    assert out["double_bass"]["status"] == "detected" and "qa:double_bass" in out["double_bass"]["new_fail"]
    # il buco non dichiarato è solo un avviso del QA (D4): la palestra lo conta come "warned", non come rilevato
    assert out["gap"]["status"] in ("warned", "detected")
    assert "qa:dropout" in out["gap"]["new_warn"] + out["gap"]["new_fail"]


def test_key_clash_only_with_overlap(ref):
    v = ref.copy()
    info = G.op_key_clash(v, 1.0, shifter=lambda x, sr, st: x)
    assert not info["applied"] and "B&B" in info["reason"]


def test_click_and_echo_change_the_mix(ref):
    for d, lv in (("click", "-6dBFS"), ("echo_offtempo", "0.37beat")):
        v = ref.copy()
        assert G.apply_deck_defect(v, d, lv)["applied"]
        sw = v.switch
        assert np.max(np.abs(v.audio[sw:sw + SR] - ref.audio[sw:sw + SR])) > 1e-3


def test_find_straddle_shift():
    downs = np.arange(0.0, 64.0, 2.0)
    words = [{"start_s": 25.7, "end_s": 26.4, "text": "x"}]  # attraversa 26.0 (downbeat 13) di 300/400 ms
    s = G.find_straddle_shift(words, downs, 10, 20.0, min_side_s=0.12)
    assert s["k"] == 3 and s["word"] == "x"
    assert G.find_straddle_shift(words, downs, 10, 20.0, min_side_s=0.5) is None
    w2 = [{"start_sample": int(19.8 * SR), "end_sample": int(20.3 * SR)}]
    assert G.find_straddle_shift(w2, downs, 12, 24.0, sr=SR)["k"] == -2


def test_classify_paired_and_unavailable():
    ref = {"qa:loudness_jump": "fail", "critic:beat_phase_ms": "warn"}
    c = G.classify({"qa:loudness_jump": "fail", "qa:rhythm": "fail"}, ref, ["qa:rhythm"])
    assert c["status"] == "detected" and c["specific"] and c["new_fail"] == ["qa:rhythm"]
    c = G.classify({"qa:loudness_jump": "fail"}, ref, ["qa:loudness_jump"], {"qa:loudness_jump": "measured"})
    assert c["status"] == "missed"  # già fail nel riferimento: non è una rilevazione nuova
    c = G.classify({}, {}, ["qa:click"], {"qa:rhythm": "measured"})
    assert c["status"] == "unavailable"  # controllo inesistente: dichiarato, mai pass
    c = G.classify({"v4:dropout": "warn"}, {}, ["qa:dropout"], {"qa:dropout": "measured"})
    assert c["status"] == "warned"
    c = G.classify({}, {}, ["qa:rhythm"], error="boom")
    assert c["status"] == "error" and not c["detected"]["pipeline"]


def test_runner_veto_rules():
    crt = {"flags": [{"flag": "beat_phase_ms"}, {"flag": "loud_jump_db"}, {"flag": "chroma_dist"}], "declared_phase": {"ms": 0.0}}
    assert G.runner_veto_flags(crt, "cut", False) == ["loud_jump_db"]
    assert G.runner_veto_flags(crt, "blend", True) == ["chroma_dist"]
    crt["declared_phase"] = {"ms": 12.0}
    assert G.runner_veto_flags(crt, "blend", False) == ["beat_phase_ms", "loud_jump_db", "chroma_dist"]


def test_aggregate_and_gate():
    rows = []
    for i in range(10):
        st = "detected" if i < 9 else "missed"
        rows.append({"ref": f"r{i}", "variant": "phase@half_beat", "defects": [["phase", "half_beat"]], "status": "evaluated",
                     "classification": G.classify({"qa:rhythm": "fail"} if st == "detected" else {}, {}, ["qa:rhythm"],
                                                  {"qa:rhythm": "measured"})})
    rows.append({"ref": "r0", "variant": "click@-6dBFS", "defects": [["click", "-6dBFS"]], "status": "not_generated", "reason": "x"})
    refs = [{"id": f"r{i}", "origin": "natural", "clean": True, "signals": {}, "reference_flags": G.reference_flags({})} for i in range(10)]
    refs[0]["reference_flags"] = G.reference_flags({"qa:loudness_jump": "fail"})
    refs[0]["signals"] = {"qa:loudness_jump": "fail"}
    s = G.aggregate(rows, refs)
    e = s["variants"]["phase@half_beat"]
    assert e["n"] == 10 and e["detection_rate"] == pytest.approx(0.9)
    assert s["variants"]["click@-6dBFS"]["not_generated"] == 1 and s["variants"]["click@-6dBFS"]["n"] == 0
    assert s["false_alarms"]["natural"]["rates"]["pipeline"] == pytest.approx(0.1)
    gate = {"detection": [{"variant": "phase@half_beat", "min_rate": 0.9, "min_n": 5},
                          {"variant": "click@-6dBFS", "min_rate": 0.5, "min_n": 1}],
            "false_alarms": [{"group": "natural", "max_rate": 0.1, "min_n": 5}]}
    g = G.check_gate(s, gate)
    st = [r["status"] for r in g["rules"]]
    assert st == ["pass", "insufficient", "pass"] and not g["passed"]  # nessuna variante valutata: mai un pass


def test_gate_config_is_declared():
    cfg = G.load_gate()
    assert cfg["false_alarms"] and cfg["detection"]
    names = {r["variant"] for r in cfg["detection"]}
    assert {"word_cut_a@straddle120", "phase@half_beat"} <= names
    for r in cfg["detection"]:
        d, lv = r["variant"].split("@")
        assert lv in G.DEFECTS[d]["levels"]
    json.dumps(cfg)


def test_variant_plan_has_singles_and_combos():
    plan = G.variant_plan({})
    ids = {G.variant_id(p) for p in plan}
    assert "phase@half_beat" in ids and "loud_jump@+3dB" in ids
    assert any("+" in i for i in ids)
    only = G.variant_plan({}, defects=["gap"], combos=True)
    assert {G.variant_id(p) for p in only} == {"gap@half_beat", "gap@1beat"}


def test_human_effort_set_counts(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("human_effort_report", Path(__file__).resolve().parents[1] / "tools/human_effort_report.py")
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    run = tmp_path / "set_x"
    (run / "gallery").mkdir(parents=True)
    trs = [{"index": 1, "a": "a", "b": "b", "technique": "cut", "qa": {"verdict": "pass"}, "clip_sha256": "x"},
           {"index": 2, "a": "b", "b": "c", "technique": "echo_out", "qa": {"verdict": "fail"}, "clip_sha256": "y"},
           {"index": 3, "a": "c", "b": "d", "technique": "blend4", "qa": {"verdict": "warn"}, "clip_sha256": "z", "kept_despite_veto": True}]
    log = [{"pool": {"skipped": [{"slug": "e", "why": "escluso finché la griglia non è verificata all'ascolto"}]}}]
    (run / "set.json").write_text(json.dumps({"transitions": trs, "log": log}))
    (run / "gallery" / "review_manifest.json").write_text(json.dumps({"groups": [{"items": [{}, {}]}, {"items": [{}]}]}))
    jd = tmp_path / "j"
    jd.mkdir()
    (jd / "1.json").write_text(json.dumps({"score": 6, "note": "B nel secondo sbagliato", "tags": ["punto_frase_sbagliato"]}))
    (jd / "2.json").write_text(json.dumps({"score": None, "note": ""}))
    r = H.set_effort(run, jd)
    assert r["votes_requested"] == 3 and r["votes_given"] == 1 and r["votes_with_notes"] == 1
    assert [t["index"] for t in r["transitions_to_review"]] == [2, 3] and r["transitions_warn"] == []
    assert r["actions"]["manual_track_decisions"] == 1
    assert r["per_10_sets"]["total"] == 10 * r["actions_total"]
