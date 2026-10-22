"""Supervisione dai brani stessi (conoscenza_mix/_meta/SUPERVISIONE_DAI_BRANI_2026-10-21.md): test sintetici.

Costruzione dei punti e delle giunte, split per artista senza sovrapposizioni, cache e API ``cue_naturalness``.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from musicmix import self_supervision as S

PER = 0.5  # 120 BPM, battuta = 2 s


def _track(slug="t1", artist="Artist A", bpm=120.0, dur=80.0, sections=(8.0, 24.0, 40.0, 56.0), lines=None, snap=None):
    db = np.arange(0.0, dur, 4 * PER)
    secs = [{"label": "verse", "start_s": t, "downbeat_s": t, "snap_error_bars": (snap or {}).get(t, 0.0), "loose": False}
            for t in sections]
    words, lns = [], []
    # versi: (inizio, numero di parole); parole da 0,3 s con 0,1 s tra l'una e l'altra
    for start, n in (lines or [(9.6, 8), (13.6, 6), (30.0, 8), (44.4, 5), (50.0, 6)]):
        fw = len(words)
        for k in range(n):
            on = start + k * 0.4
            words.append({"on_s": on, "off_s": on + 0.3, "text": f"w{len(words)}"})
        lns.append({"start_s": start, "end_s": words[-1]["off_s"], "first_word": fw, "last_word": len(words) - 1, "p_end": 0.9})
    return S.TrackData(slug=slug, artist=artist, group=S.artist_group(artist), sr=48000, duration_s=dur, period_s=60.0 / bpm,
                       downbeats_s=db, sections=secs, words=words, lines=lns, bpm=bpm, camelot="8A")


def test_artist_group_normalization():
    assert S.artist_group("Travi$ Scott") == "travis scott"
    assert S.artist_group("Travis Scott") == "travis scott"
    assert S.artist_group("Future, Drake") == "future"
    assert S.artist_group("18K; Kid Yugi") == "18k"
    assert S.artist_group("Pop Smoke feat. Swae Lee") == "pop smoke"


def test_natural_points_kinds_and_rest_rule():
    tr = _track(snap={40.0: -0.25})  # sezione a 40 s non agganciata al downbeat: esclusa dai positivi
    pts = S.natural_points(tr)
    secs = [p["t"] for p in pts if p["kind"] == "section"]
    assert secs == [8.0, 24.0, 56.0]
    lines = {round(p["t"] + 0.03, 3): p["kind"] for p in pts if p["kind"] in ("line", "reentry")}
    # 9.6: prima frase (nessuna voce prima) -> rientro; 13.6: pausa 13.6-12.7=0.9 s < ... no: 1 battito = 0.5 s -> verso
    assert lines[9.6] == "reentry"
    assert lines[13.6] == "line"
    assert lines[30.0] == "reentry"  # >= 2 battute senza parole
    # verso con pausa < 1 battito: non positivo
    tr2 = _track(lines=[(9.6, 8), (12.9, 4)])
    starts = [round(p["t"] + 0.03, 3) for p in S.natural_points(tr2) if p["kind"] != "section"]
    assert 12.9 not in starts  # 12.9 - 12.7 = 0.2 s < 0.5 s


def test_shifted_negatives_offsets_and_exclusion():
    tr = _track()
    pos = S.natural_points(tr)
    neg = S.shifted_negatives(tr, pos, seed=1)
    excl = np.concatenate([S.plausible_boundaries(tr), [p["t"] for p in pos]])
    expected = {"offbeat": 0.5, "beat": 1.0, "halfbar": 2.0, "bar": 4.0}
    for n in neg:
        assert np.min(np.abs(excl - n["t"])) > S.PARAMS["exclusion_s"]
        assert S.PARAMS["margin_s"] <= n["t"] <= tr.duration_s - S.PARAMS["margin_s"]
        if n["kind"] in expected:
            assert abs(abs(n["shift_s"]) - expected[n["kind"]] * PER) < 1e-9
        if n["kind"] == "midword":
            assert any(w["on_s"] < n["t"] < w["off_s"] for w in tr.words)
        if n["kind"] == "midline":
            assert any(ln["start_s"] < n["t"] < ln["end_s"] for ln in tr.lines)
            assert np.min(np.abs(tr.downbeats_s - n["t"])) < 1e-9
        if n["kind"] == "mid_section":
            sec = np.asarray([8.0, 24.0, 40.0, 56.0])
            prev = sec[sec <= n["t"]][-1]
            assert round((n["t"] - prev) / (4 * PER)) % 4 == 2
    kinds = {n["kind"] for n in neg}
    assert {"offbeat", "beat", "halfbar", "bar", "midword", "midline", "random"} <= kinds


def test_build_points_ids_and_labels():
    rows = S.build_points([_track("a"), _track("b", artist="Artist B")], seed=0)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    pos_ids = {r["id"] for r in rows if r["label"] == 1}
    assert all(r["anchor"] in pos_ids for r in rows if r["label"] == 0)
    assert {r["group"] for r in rows} == {"artist a", "artist b"}


def test_build_pairs_within_jump_cross_groups_and_tempo():
    ta, tb = _track("a", "Artist A", bpm=120.0), _track("b", "Artist B", bpm=121.0)
    tc = _track("c", "Artist C", bpm=100.0)  # tempo incompatibile
    td = _track("d", "Artist A", bpm=120.0)  # stesso artista di a: niente giunte cross a-d
    tracks = [ta, tb, tc, td]
    pts = S.build_points(tracks, seed=0)
    pairs = S.build_pairs(pts, tracks, seed=0)
    byid = {r["id"]: r for r in pts}
    for p in pairs:
        if p["scope"] == "within":
            assert p["x"] == p["y"]
        else:
            assert p["x"] != p["y"]
            assert {p["x"], p["y"]} != {"a", "d"}
            assert "c" not in (p["x"], p["y"])
        if p["label"] == 1:
            assert byid[p["p1"]]["label"] == 1 and byid[p["p2"]]["label"] == 1
            if p["scope"] == "within":
                assert abs(p["t2"] - p["t1"]) >= S.PARAMS["pair_min_jump_bars"] * 4 * PER - 1e-9
        else:
            moved = byid[p["p2"]] if p["side"] == "entry" else byid[p["p1"]]
            kept = byid[p["p1"]] if p["side"] == "entry" else byid[p["p2"]]
            assert moved["label"] == 0 and kept["label"] == 1 and moved["kind"] == p["kind"]
    assert {p["scope"] for p in pairs} == {"within", "cross"}


def test_split_has_no_track_or_artist_overlap():
    groups = {"a": "g1", "b": "g2", "c": "g1"}
    S.check_disjoint({"b"}, {"a", "c"}, groups)
    with pytest.raises(AssertionError):
        S.check_disjoint({"a", "b"}, {"a"}, groups)
    with pytest.raises(AssertionError):
        S.check_disjoint({"a", "b"}, {"c"}, groups)  # stesso artista in train e test
    assert S.group_folds(["g2", "g1", "g2"]) == ["g1", "g2"]


def test_metrics_auc_and_paired():
    assert S.auc([1, 2, 3], [0, 0.5]) == 1.0
    assert S.auc([0], [1]) == 0.0
    assert S.auc([1, 1], [1]) == 0.5
    rows = [{"id": "p", "label": 1, "kind": "section", "anchor": "p"},
            {"id": "n1", "label": 0, "kind": "beat", "anchor": "p"}, {"id": "n2", "label": 0, "kind": "beat", "anchor": "p"}]
    assert S.paired_win_rate({"p": 0.8, "n1": 0.5, "n2": 0.9}, rows, "beat") == 0.5
    assert S.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def _write_cache(tmp_path, tr, seed=0):
    rng = np.random.default_rng(seed)
    n_m = int(tr.duration_s / S.MERT_HOP_S)
    base = rng.normal(size=(n_m, 768)).astype(np.float16)
    (tmp_path / "cache/mert").mkdir(parents=True, exist_ok=True)
    (tmp_path / "cache/dsp").mkdir(parents=True, exist_ok=True)
    np.savez(tmp_path / "cache/mert" / f"{tr.slug}.npz", l6=base, l12=base)
    n_d = int(tr.duration_s * S.DSP_FPS)
    np.savez_compressed(tmp_path / "cache/dsp" / f"{tr.slug}.npz", x=rng.normal(size=(n_d, len(S.DSP_COLS))).astype(np.float16))


def test_cache_features_and_model_roundtrip(tmp_path):
    tr = _track()
    _write_cache(tmp_path, tr)
    cache = S.TrackCache.load(tmp_path, tr.slug)
    f = S.point_features(tr, cache, 20.0)
    assert f["mert_c"].shape == (8,) and f["mert_h"].shape == (4 * 768,)
    assert f["grid"].shape == (6,) and np.all(np.isfinite(f["dsp_mix"])) and f["vox"].shape == (8,)
    assert f["dsp_mix"].size + f["vox"].size == 6 * len(S.DSP_COLS) + 3
    # costante: distanza coseno nulla
    const = {6: np.ones((500, 768), np.float16), 12: np.ones((500, 768), np.float16)}
    comp, _ = S.mert_point_block(const, 30.0)
    assert np.allclose(comp, 0.0, atol=1e-6)
    pts = S.build_points([tr], seed=0)
    X = S.stack([S.point_features(tr, cache, r["t"]) for r in pts], S.ALL_BLOCKS)
    y = np.asarray([r["label"] for r in pts])
    m = S.BlockLogistic(S.FEATURE_BLOCKS["full"], pca_dims=8).fit(X, y)
    p = m.predict_proba(X)
    m.save(tmp_path / "model" / "point_full.npz")
    m2 = S.BlockLogistic.load(tmp_path / "model" / "point_full.npz")
    assert np.allclose(p, m2.predict_proba(X))
    # la griglia da sola separa i ±½ battito dai downbeat delle sezioni (controllo della fuga d'informazione voluta)
    g = S.BlockLogistic(("grid",)).fit(X, y)
    assert g.params["coef"].shape == (6,)


def test_cue_naturalness_cached_computed_unavailable(tmp_path, monkeypatch):
    tr = _track("trk")
    _write_cache(tmp_path, tr)
    pts = S.build_points([tr], seed=0)
    cache = S.TrackCache.load(tmp_path, tr.slug)
    X = S.stack([S.point_features(tr, cache, r["t"]) for r in pts], S.ALL_BLOCKS)
    S.BlockLogistic(S.FEATURE_BLOCKS["full"], pca_dims=8).fit(X, np.asarray([r["label"] for r in pts])).save(
        tmp_path / "model" / "point_full.npz")
    cue_dir = tmp_path / "cue_scores"
    cue_dir.mkdir()
    cands = S.cue_candidates(tr)
    assert any(c["what"] == "line_start" for c in cands) and any(c["what"] == "downbeat" for c in cands)
    for c in cands:
        c["p_full"] = 0.5
    (cue_dir / "trk.json").write_text(json.dumps({"slug": "trk", "oof": True, "model_files": {"full": "model/point_full.npz"},
                                                  "candidates": cands}))
    monkeypatch.setattr(S, "load_track", lambda slug, **kw: tr)
    S._CUE_MEM.clear()
    r = S.cue_naturalness("trk", 24.0, cache_dir=cue_dir, run_dir=tmp_path)
    assert r["status"] == "cached" and r["p"] == 0.5 and r["what"] == "downbeat" and r["oof"] is True
    r2 = S.cue_naturalness("trk", [24.11, 24.0], cache_dir=cue_dir, run_dir=tmp_path)
    assert r2[0]["status"] == "computed" and 0.0 <= r2[0]["p"] <= 1.0 and r2[1]["status"] == "cached"
    r3 = S.cue_naturalness("missing_track", 10.0, cache_dir=cue_dir, run_dir=tmp_path / "nowhere")
    assert r3["status"] == "unavailable" and r3["p"] is None


def _tension_track():
    secs = [(0.0, "intro"), (16.0, "verse"), (48.0, "chorus"), (64.0, "verse"), (96.0, "chorus"), (112.0, "outro")]
    tr = _track(dur=130.0, sections=[t for t, _ in secs])
    for s, (_, lab) in zip(tr.sections, secs):
        s["label"] = lab
    return tr


def test_tension_examples_roles_and_negatives():
    tr = _tension_track()
    ex = S.tension_examples(tr)
    bar = 4 * PER
    assert [r["t"] for r in ex["exit"] if r["label"] == 1] == [48.0, 96.0]
    assert [r["t"] for r in ex["entry"] if r["label"] == 1] == [48.0, 96.0]
    arrivals = np.asarray([48.0, 96.0])
    for r in ex["exit"]:
        if r["label"] == 0 and r["kind"] in ("early_verse", "mid_verse", "intro"):
            # mai nelle 4 battute che precedono un arrivo (lì A sta caricando)
            assert not np.any((arrivals - r["t"] > -1e-6) & (arrivals - r["t"] < 4 * bar))
    kinds = {r["kind"] for r in ex["entry"] if r["label"] == 0}
    assert {"mid_verse", "verse_start", "late_arrival"} <= kinds
    assert set(S.TENSION_NEG_TYPES["exit"]) >= {r["kind"] for r in ex["exit"] if r["label"] == 0}


def test_tension_features_use_only_one_side(tmp_path):
    tr = _tension_track()
    _write_cache(tmp_path, tr)
    cache = S.TrackCache.load(tmp_path, tr.slug)
    f_exit = S.tension_features(tr, cache, 48.0, "exit")
    f_entry = S.tension_features(tr, cache, 48.0, "entry")
    assert set(f_exit) == set(S.TENSION_ALL_BLOCKS) and f_exit["t_mert"].shape == (768,)
    assert np.isnan(f_exit["t_form"]).all()  # senza dj_form il blocco è NaN (il modello si astiene)
    # cambiare l'audio dopo t non cambia l'uscita; prima di t non cambia l'ingresso (a parte il contrasto)
    k = int(48.0 * S.DSP_FPS)
    cache.dsp = cache.dsp.copy()
    cache.dsp[k + 5:] += 3.0
    assert np.allclose(S.tension_features(tr, cache, 48.0, "exit")["t_dsp"], f_exit["t_dsp"])
    cache2 = S.TrackCache.load(tmp_path, tr.slug)
    cache2.dsp = cache2.dsp.copy()
    cache2.dsp[: k - 5] -= 3.0
    assert np.allclose(S.tension_features(tr, cache2, 48.0, "entry")["t_dsp"], f_entry["t_dsp"], atol=1e-3)
