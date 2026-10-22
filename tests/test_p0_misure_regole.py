"""P0 'misure e regole' dell'audit dei nove libri (21/10/2026).

Controesempi riprodotti (ROOT_COUNTEREXAMPLES.json), livelli di evidenza delle regole dei profili libri e finestre
dei cue per tecnica. I test di dj_form e dj_phrase.with_stem_onsets sono in test_dj_form.py e test_dj_phrase_books.py.
"""
import numpy as np
import pytest

from musicmix import dj_phrase as P
from musicmix import dj_policy as pol
from musicmix import dj_select as djs
from musicmix import dj_set as ds

LEVELS = {"E", "T", "A", "P", "H", "U", "S"}


def _cand(kind, bars=0, cost=0.0):
    dec = {"type": kind, "bars": bars, "render": {"delay_beats": 1.0, "entry_bars": 1} if kind == "echo_out" else {}}
    return {"decision": dec, "cost": cost, "admissible": True, "plan": {"score": 0.0}, "notes": []}


# ------------------------------------------------------------------ far genre: degenerate matrix (counterexample 3)
def test_identical_timbre_matrix_abstains_instead_of_far():
    """ROOT_COUNTEREXAMPLES.json 'far_genre_identical_features': matrice tutta 0 -> soglia 0 -> coppia a distanza 0
    'lontana' e coda d'eco fermata. Ora: calibrazione degenere, nessuna soglia, nessuna coppia lontana."""
    Z = np.zeros((3, 3))
    cal = ds.far_timbre_calibration(Z, 90)
    assert cal["status"] == "degenerate" and cal["threshold"] is None and cal["raw_threshold"] == 0.0
    assert ds.far_timbre_threshold(Z, 90) is None
    pc = {"candidates": [_cand("echo_out")]}
    for thr in (cal, ds.far_timbre_threshold(Z, 90), 0.0):  # dict, None, and a legacy float threshold of 0
        out = ds.apply_far_genre(pc, 0.0, thr, 120.0)
        assert out["far_genre"]["far"] is False
        assert "tail_stop" not in out["candidates"][0]["decision"]["render"]
    assert ds.apply_far_genre(pc, 0.0, 0.0, 120.0)["far_genre"]["status"] == "degenerate"


def test_mostly_identical_pool_needs_more_than_the_median():
    D = np.zeros((6, 6))
    D[0, 5] = D[5, 0] = 4.0  # one distant pair, everything else identical: p90 = 0 -> degenerate (IQR 0)
    cal = ds.far_timbre_calibration(D, 90)
    assert cal["status"] == "degenerate"
    # a usable spread: the median check keeps pairs at the median out even if they reach the threshold
    E = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 3], [1, 1, 3, 0]], float)
    calE = ds.far_timbre_calibration(E, 50, min_spread=0.0)
    assert calE["median"] == 1.0
    out = ds.apply_far_genre({"candidates": [_cand("echo_out")]}, 1.0, {**calE, "status": "ok", "threshold": 1.0}, 120.0)
    assert out["far_genre"]["far"] is False  # 1.0 >= threshold but not above the median
    out = ds.apply_far_genre({"candidates": [_cand("echo_out")]}, 3.0, {**calE, "status": "ok", "threshold": 1.0}, 120.0)
    assert out["far_genre"]["far"] is True


def test_far_genre_regular_pool_unchanged_and_options_switch_off():
    D = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]], float)
    cal = ds.far_timbre_calibration(D, 90)
    assert cal["status"] == "ok" and cal["threshold"] == ds.far_timbre_threshold(D, 90)
    pc = {"candidates": [_cand("echo_out", cost=0.1), _cand("blend", 4, 0.2)]}
    a = ds.apply_far_genre(pc, 3.0, cal["threshold"], 400.0)
    b = ds.apply_far_genre(pc, 3.0, cal, 400.0)
    for out in (a, b):  # float and dict give the same decisions as before
        assert out["far_genre"]["far"] and "[H]" in out["far_genre"]["evidence"]
        assert next(c for c in out["candidates"] if c["decision"]["type"] == "echo_out")["decision"]["render"]["tail_stop"]
        assert not next(c for c in out["candidates"] if c["decision"]["type"] == "blend")["admissible"]  # 2,4 s < 3 s
    off = ds.apply_far_genre(pc, 3.0, cal, 400.0, min_blend_s=None, echo_tail_stop=False)
    assert all(c["admissible"] for c in off["candidates"])
    assert "tail_stop" not in next(c for c in off["candidates"] if c["decision"]["type"] == "echo_out")["decision"]["render"]
    # the options are documented [H] parameters with the old values as defaults
    v = ds.values(ds.ORDER_PARAMS)
    assert v["far_min_blend_s"] == 3.0 and v["far_echo_tail_stop"] is True and v["far_timbre_pct"] is None
    for k in ("far_timbre_pct", "far_timbre_min_spread", "far_min_blend_s", "far_echo_tail_stop"):
        assert ds.ORDER_PARAMS[k].source.startswith("[H]")


# ------------------------------------------------------------------ evidence levels of the book rules
def _check_table(table, keys):
    assert set(table) == set(keys), sorted(set(keys) ^ set(table))
    for k, e in table.items():
        assert e["source_level"] and set(e["source_level"].split("+")) <= LEVELS, k
        assert e["rule_level"] and set(e["rule_level"].split("+")) <= LEVELS, k
        assert e["verified"], k


def test_book_rules_have_verified_evidence_levels():
    _check_table(P.BOOK_RULES_EVIDENCE, P.BOOK_RULES)
    # corrections of the audit: thresholds are ours, not the books'
    for k in ("loop_vocal_max_repeats", "schema_switch_seconds", "loop_with_new_layer_max_bars", "loop_focal_repeats",
              "loop_comic_repeats"):
        assert "H" in P.BOOK_RULES_EVIDENCE[k]["rule_level"].split("+"), k
    assert "secondi" in P.BOOK_RULES_EVIDENCE["loop_focal_repeats"]["note"]  # Margulis p. 8: 1/4/8 are seconds
    assert P.BOOK_RULES["loop_vocal_max_repeats"].source.startswith("[H]")


def test_profile_books_have_verified_evidence_levels():
    keys = [f"{lvl}.{k}" for lvl, d in ds.PROFILE_BOOKS.items() for k in d]
    _check_table(ds.PROFILE_BOOKS_EVIDENCE, keys)
    _check_table(pol.PROFILE_BOOKS_EVIDENCE, pol.PROFILE_BOOKS)
    assert ds.PROFILE_BOOKS_EVIDENCE["order.far_timbre_pct"]["rule_level"] == "H"


def test_loop_vocal_warning_is_a_diagnostic_hypothesis():
    r = P.loop_repetition_check(2, 8, loop_has_vocal=True, new_layer_entering=True)
    voc = [f for f in r["findings"] if f["code"] == "vocal_repeats"]
    assert voc and voc[0]["level"] == "H" and voc[0]["diagnostic_only"]
    assert all(w.startswith("[") for w in r["warnings"])  # every warning carries its evidence level
    # the 'ok' contract of the existing callers is unchanged
    assert r["ok"] is False
    assert P.loop_repetition_check(4, 8, new_layer_entering=False)["ok"]
    # without the hypotheses the vocal warning is gone
    r2 = P.loop_repetition_check(2, 8, loop_has_vocal=True, new_layer_entering=True, include_hypotheses=False)
    assert not any(f["code"] == "vocal_repeats" for f in r2["findings"])


def test_krims_closure_is_diagnostic_and_not_from_p32():
    assert "p. 32" in P.KRIMS_CLOSURE_NOTE and "diagnostic" in P.KRIMS_CLOSURE_NOTE
    assert P.PARAMS.get("w_krims_closure") is None or P.PARAMS["w_krims_closure"].value == 0.0


# ------------------------------------------------------------------ cue windows by role and technique
SECS = ((0, "intro"), (8, "verse"), (24, "chorus"), (40, "verse"), (56, "chorus"))


def _track(slug="t", n_bars=96, bpm=80.0, sections=SECS, vocal=None):
    bar = 4 * 60.0 / bpm
    downbeats = np.arange(n_bars) * bar
    secs = [{"label": lab, "start_s": b * bar, "end_s": 0.0, "bar_index": b, "downbeat_s": b * bar,
             "snap_error_bars": 0.0, "loose": False, "support": ["x"], "source": "test"} for b, lab in sections]
    vocal = np.asarray([0.0 if j < 8 else 1.0 for j in range(n_bars)] if vocal is None else vocal, dtype=float)
    return {"slug": slug, "duration_s": n_bars * bar, "tempo": {"bpm": bpm, "confidence": "high", "period_s": 60.0 / bpm},
            "key": {"camelot": "8A", "confidence": 0.9}, "sections": secs, "phrases": djs.phrase_table(downbeats, secs),
            "bars": {"vocal": vocal.tolist(), "vocal_last_beat": vocal.tolist(), "energy_db_rel": [0.0] * n_bars}}


def test_default_windows_unchanged():
    T = _track()
    n = len(T["phrases"])
    base = djs.cue_candidates(T, "in")
    assert [c["bar_index"] for c in base] == [c["bar_index"] for c in djs.cue_candidates(T, "in", windows=djs.CUE_WINDOWS["default"])]
    assert "windows_default_ok" not in base[0]
    assert max(c["position"] for c in base) <= 0.5 and min(n - c["bar_index"] for c in base) >= 16
    outs = djs.cue_candidates(T, "out")
    assert min(c["position"] for c in outs) >= 0.35 and min(n - c["bar_index"] for c in outs) >= 4
    d = djs.CUE_WINDOWS["default"]
    assert (d["out_min_position"], d["out_min_bars_left"], d["in_max_position"], d["in_min_bars_left"]) == (0.35, 4, 0.5, 16)
    assert d["out_min_s"] is None and d["in_min_s_left"] is None


def test_technique_windows_only_add_cues_and_flag_them():
    T = _track()  # 96 bars at 80 BPM = 288 s
    for role in ("in", "out"):
        base = {c["bar_index"] for c in djs.cue_candidates(T, role)}
        wide = djs.cue_candidates(T, role, windows=djs.CUE_WINDOWS["technique"])
        assert base <= {c["bar_index"] for c in wide}
        for c in wide:
            assert c["windows_default_ok"] == (c["bar_index"] in base)
    late = [c for c in djs.cue_candidates(T, "in", windows=djs.CUE_WINDOWS["technique"]) if c["bar_index"] == 56]
    assert late and late[0]["section_label"] == "chorus" and not late[0]["windows_default_ok"]  # chorus at 58 %, 120 s left
    early = [c for c in djs.cue_candidates(T, "out", windows=djs.CUE_WINDOWS["technique"]) if c["bar_index"] == 24]
    assert early and early[0]["prev_label"] == "verse" and not early[0]["windows_default_ok"]  # 72 s in, 25 % of A
    assert all(c["time_s"] >= 60.0 for c in djs.cue_candidates(T, "out", windows=djs.CUE_WINDOWS["technique"])
               if not c["windows_default_ok"])


def test_candidates_for_pair_technique_windows_never_add_blends():
    A, B = _track("A"), _track("B")
    base = ds.candidates_for_pair(A, B)
    wide = ds.candidates_for_pair(A, B, cue_windows="technique")
    key = lambda c: (c["plan"]["a_out_bar"], c["plan"]["b_in_bar"])  # noqa: E731
    kb = {key(c) for c in base["candidates"]}
    assert kb <= {key(c) for c in wide["candidates"]}
    extra = [c for c in wide["candidates"] if key(c) not in kb]
    assert extra, "the technique windows add candidates"
    for c in extra:
        assert c["decision"]["type"] not in ds.OVERLAP_TYPES and c["cue_windows"] == "technique"
        assert c["plan"]["windows_default_ok"] is False
    assert base["cue_windows"] == "default" and wide["cue_windows"] == "technique"
    assert all("cue_windows" not in c for c in base["candidates"])
    # same best candidate on this symmetric pair: extra cues do not win ties
    assert key(base["candidates"][0]) == key(wide["candidates"][0])
    with pytest.raises(KeyError):
        ds.candidates_for_pair(A, B, cue_windows="wide")


# ------------------------------------------------------------------ the audit probe shows the corrected outcomes
def test_book_pipeline_probes_report_the_fixes():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "tools" / "book_pipeline_probes.py"
    spec = importlib.util.spec_from_file_location("book_pipeline_probes", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    res = mod.probe()
    assert res["form_no_core"]["fixed"] and res["onset_after_word"]["fixed"] and res["identical_timbre"]["fixed"]
    assert res["onset_after_word"]["legacy"]["snapped_share"] == 1.0  # the counterexample is still reproducible on request
