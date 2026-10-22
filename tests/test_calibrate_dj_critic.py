"""Calibrazione del giudice: solo giudizi ciechi, ascolti ripetuti separati, selezione/verifica (E-crit, E2)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("calibrate_dj_critic", ROOT / "tools/calibrate_dj_critic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _doc(cid, score, blind=True, plays=1, group=None, saved="2026-10-21T10:00:00Z"):
    return {"run": "g", "type": "versione", "clip_id": cid, "group_id": group or f"grp_{cid}", "score": score,
            "blind": blind, "plays_before_save": plays, "saved_at": saved}


def test_calibration_uses_only_blind_votes_and_separates_repeated_plays(tmp_path):
    T = _tool()
    jd = tmp_path / "judgments"
    jd.mkdir()
    docs = [_doc(f"c{i}", s) for i, s in enumerate([9, 8, 7, 3, 2, 5, 6, 1])]
    docs += [_doc("nb", 10, blind=False), _doc("rep", 9, plays=12)]
    docs += [_doc("c0", 2, saved="2026-10-20T10:00:00Z")]  # voto più vecchio sulla stessa clip: ignorato
    for k, d in enumerate(docs):
        (jd / f"d{k}.json").write_text(json.dumps(d))
    (jd / "export.json").write_text(json.dumps({"run": "g", "judgments": {"x": {**_doc("c9", 4), "doc_id": "x"},
                                                                          "pref": {"type": "preferenza", "group_id": "g"}}}))
    js = T.read_judgments(jd)
    assert {d["clip_id"] for d in js} >= {"c0", "c9", "nb", "rep"}
    scores = {f"c{i}": {"v3_score": s / 10, "v3_n_flags": 0, "summary": {"metric_surprise": 1 - s / 10}}
              for i, s in enumerate([9, 8, 7, 3, 2, 5, 6, 1, 0, 4])}
    scores["nb"] = {"v3_score": 0.0}
    rep = T.calibrate(js, scores, include_nonblind=True, max_plays=3, min_n=20)
    c = rep["counts"]
    assert c["not_blind"] == 1 and c["blind_repeated_plays"] == 1 and c["primary_blind_le_max_plays"] == 9
    p = rep["primary"]["predictors"]["v3_score"]
    assert p["spearman"]["rho"] == pytest.approx(1.0) and p["auc_ge7_vs_le4"]["auc"] == 1.0
    assert rep["primary"]["predictors"]["v4_metric_surprise_neg"]["pairwise"]["accuracy"] == 1.0
    assert "not_blind_diagnostic" in rep and rep["warnings"]  # meno di 20 voti: dichiarato
    assert rep["primary_weighted_all_blind"]["n"] == 10
    assert set(rep["halves"]) == {"selection", "verification"}
    assert sum(rep["halves"][h]["n"] for h in rep["halves"]) == 9


def test_calibration_without_blind_votes_has_empty_primary():
    T = _tool()
    rep = T.calibrate([_doc("a", 9, blind=False), _doc("b", 2, blind=False)], {}, min_n=20)
    assert rep["primary"]["n"] == 0
    assert any("nessun giudizio alla cieca" in w for w in rep["warnings"])
