"""Test sintetici del critico automatico (dj_critic): solo funzioni pure e
audio generato, nessun modello preaddestrato richiesto."""

from __future__ import annotations

import json

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
pytest.importorskip("scipy")

from musicmix import dj_critic as C  # noqa: E402

SR = 48000


def test_camelot_distance():
    assert C.camelot_distance("8A", "8A") == 0
    assert C.camelot_distance("8A", "9A") == 1
    assert C.camelot_distance("8A", "8B") == 1  # relativa
    assert C.camelot_distance("12A", "1A") == 1  # cerchio
    assert C.camelot_distance("8A", "11A") == 3
    assert C.camelot_distance(None, "1A") is None


def test_delay_pattern_and_frame_nll():
    codes = np.array([[1, 2, 3], [4, 5, 6]])  # K=2, F=3
    inp, tgt = C.musicgen_delay_targets(codes, pad_id=9, bos_id=9)
    assert tgt.tolist() == [[1, 2, 3, 9], [9, 4, 5, 6]]
    assert inp[:, 0].tolist() == [9, 9] and inp[:, 1:].tolist() == tgt[:, :-1].tolist()
    # log-prob uniformi tranne il target del frame 1 codebook 1 (molto improbabile)
    v = 10
    logp = np.full((2, 4, v), np.log(1 / v))
    logp[1, 2, 5] = np.log(1e-4)
    nll = C.frame_nll_from_logprobs(logp, tgt, 3)
    assert nll.shape == (2, 3)
    assert np.isclose(nll[0, 0], np.log(v))
    assert nll[1, 1] > nll[1, 0] + 5


def test_surprisal_peak_detects_spike_at_seam():
    rng = np.random.default_rng(0)
    fps = 50.0
    nll = 15 + rng.normal(0, 1, int(20 * fps))
    spiky = nll.copy()
    spiky[int(10 * fps): int(10.4 * fps)] += 8
    flat = C.surprisal_seam_stats(nll, fps=fps, start_s=10, switch_s=10, end_s=10)
    peak = C.surprisal_seam_stats(spiky, fps=fps, start_s=10, switch_s=10, end_s=10)
    assert flat["status"] == peak["status"] == "ok"
    assert peak["peak_z"] > flat["peak_z"] + 3
    assert 9.8 <= peak["peak_t_s"] <= 10.6
    assert peak["peak_z_sd"] >= 0


def test_surprisal_requires_context():
    st = C.surprisal_seam_stats(np.ones(50), fps=50, start_s=0.5, switch_s=0.5, end_s=0.5)
    assert st["status"] == "insufficient_context"


def _embeddings(jump: bool, t=300, d=16, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, d)
    x = base + 0.05 * rng.normal(0, 1, (t, d))
    if jump:
        x[t // 2:] += 2.0 * rng.normal(0, 1, d)
    return x


def test_seam_features_separate_jump_from_continuation():
    cont = C.seam_feature_block(_embeddings(False), fps=25, seam_s=6.0)
    jump = C.seam_feature_block(_embeddings(True), fps=25, seam_s=6.0)
    assert jump["compact"]["cos_pre_post"] < cont["compact"]["cos_pre_post"]
    assert jump["compact"]["l2_pre_post"] > cont["compact"]["l2_pre_post"]
    assert jump["compact"]["jump_z"] > cont["compact"]["jump_z"] + 3
    assert jump["compact"]["foote_z"] > cont["compact"]["foote_z"]
    compact, full = C.seam_design_row({6: jump, 12: cont})
    assert compact.shape == (8,) and full.shape == (8 * 16,)


def test_linear_seam_model_roundtrip_and_learns():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (200, 5))
    y = (x[:, 0] + 0.2 * rng.normal(0, 1, 200) > 0).astype(int)
    m = C.fit_linear_seam_model(x, y, c=1.0)
    m2 = C.LinearSeamModel.from_json(json.loads(json.dumps(m.to_json())))
    assert np.allclose(m.proba(x), m2.proba(x))
    acc = np.mean((m.proba(x) > 0.5) == y)
    assert acc > 0.9
    mp = C.fit_linear_seam_model(np.hstack([x, rng.normal(0, 1, (200, 20))]), y, pca=4)
    assert mp.proba(np.hstack([x, rng.normal(0, 1, (200, 20))])).shape == (200,)


def test_technical_from_qa_mapping():
    qa = {"verdict": "warn", "checks": {"rhythm": {"status": "pass"}, "loudness_jump": {"status": "warn", "signed_jump_db": 2.4},
                                         "duration": {"status": "fail"}}}
    t = C.technical_from_qa(qa)
    assert t["status"] == "ok"
    assert t["score"] == C.COMBINATION["technical_map"]["warn"]  # duration esclusa
    assert t["features"]["loudness_jump_db"] == 2.4
    assert C.technical_from_qa(None)["status"] == "missing"


def test_combine_and_missing():
    r = C.combine({"aesthetic": {"score": 0.2, "sd": 0.1}, "seam": {"score": 0.8, "sd": 0.1},
                   "surprisal": {"score": float("nan"), "sd": 0.1}})
    assert r["status"] == "ok" and np.isclose(r["score"], 0.5)
    assert set(r["used"]) == {"aesthetic", "seam"} and "surprisal" in r["missing"]
    assert np.isclose(r["sd"], np.sqrt(0.1 ** 2 + 0.1 ** 2) / 2)
    assert r["disagreement_sd"] > 0.25
    assert C.combine({})["status"] == "missing"


def test_percentile_and_pairwise():
    null = list(range(100))
    assert C.empirical_percentile(-1, null) < 0.01
    assert C.empirical_percentile(1000, null) > 0.99
    pa = C.pairwise_accuracy([3, 2, 1], [30, 20, 10])
    assert pa["n_pairs"] == 3 and pa["accuracy"] == 1.0
    pa = C.pairwise_accuracy([1, 1, 0], [2, 1, 1])  # un pareggio del critico, una coppia a verità pari
    assert pa["n_pairs"] == 2 and pa["accuracy"] == 0.75
    lo, hi = C.wilson_interval(3, 3)
    assert 0.3 < lo < 0.5 and hi == 1.0


def test_aesthetic_relative_deltas():
    mk = lambda v: {a: v for a in C.AUDIOBOX_QUALITY_AXES}
    ws = {"seam@-1": mk(5.0), "seam@+0": mk(5.2), "seam@+1": mk(5.4), "pre": mk(6.0), "post": mk(6.0),
          "srcA": mk(7.0), "srcB": mk(6.0)}
    r = C.aesthetic_relative(ws, C.AUDIOBOX_QUALITY_AXES)
    assert np.isclose(r["seam"], 5.2) and np.isclose(r["delta_vs_sides"], -0.8)
    assert np.isclose(r["delta_vs_srcA"], -1.8) and np.isclose(r["delta_vs_sources"], -1.3)
    assert r["seam_sd_jitter"] > 0


def _write_track(path, seconds=40.0, bpm=120.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = 0.1 * np.sin(2 * np.pi * 220 * t) + 0.01 * rng.normal(0, 1, n)
    beat = 60.0 / bpm
    for k in range(int(seconds / beat)):
        i = int(k * beat * SR)
        x[i:i + 2000] += 0.5 * np.exp(-np.arange(min(2000, n - i)) / 300)
    sf.write(str(path), np.stack([x, x], 1).astype(np.float32), SR)
    return C.TrackInfo(slug=path.stem, wav=str(path), track_id=path.stem, duration_s=seconds, bpm=bpm, camelot="8A",
                       downbeats_s=[k * 4 * beat for k in range(int(seconds / (4 * beat)))],
                       phrase_level=[16 if k % 8 == 0 else 0 for k in range(int(seconds / (4 * beat)))],
                       words=[(19.9, 20.3)], source="synthetic")


def test_synth_example_natural_is_identity_and_defects_differ(tmp_path):
    tr = _write_track(tmp_path / "a.wav")
    tracks = {"a": tr}
    base = {"a": "a", "b": "a", "a_t": 16.0, "b_t": 16.0, "fold": 0}
    nat = C.synth_example({**base, "kind": "natural", "fade": "bar"}, tracks, sr=SR)
    ref, _ = C.load_audio(tr.wav, start_s=10.0, duration_s=12.0, sr=SR)
    assert nat.shape == ref.shape
    assert np.max(np.abs(nat - ref)) < 1e-4  # dissolvenza a guadagno costante su segnale continuo = identità
    gain = C.synth_example({**base, "kind": "gain", "fade": "cut", "gain_db": 6.0}, tracks, sr=SR)
    assert np.allclose(gain[: 5 * SR], ref[: 5 * SR], atol=1e-4)
    assert C._rms(gain[7 * SR:]) / C._rms(ref[7 * SR:]) == pytest.approx(10 ** (6 / 20), rel=1e-3)
    ph = C.synth_example({**base, "kind": "phase", "fade": "cut", "delta_ms": 100.0}, tracks, sr=SR)
    assert np.max(np.abs(ph[7 * SR:] - ref[7 * SR:])) > 0.1


def test_plan_examples_keeps_folds_disjoint(tmp_path):
    trs = []
    for i in range(6):
        tr = _write_track(tmp_path / f"t{i}.wav", seconds=90.0, seed=i)
        tr.slug = f"t{i}"
        trs.append(tr)
    C.assign_folds(trs, k=3)
    recs = C.plan_examples(trs, seed=0, margin_s=7.0)
    assert recs and all(r["label"] in (0, 1) for r in recs)
    fold = {t.slug: t.fold for t in trs}
    for r in recs:
        assert fold[r["a"]] == fold[r["b"]] == r["fold"]
        if "bass_from" in r:
            assert fold[r["bass_from"]] == r["fold"]


def test_monotonicity_metric():
    m = C.monotonicity([0, 1, 2, 3], [0.9, 0.7, 0.5, 0.1])
    assert m["rho"] == pytest.approx(-1.0) and m["frac_non_increasing_steps"] == 1.0 and m["drop_0_to_max"] == pytest.approx(0.8)
    m = C.monotonicity([0, 1, 2, 3], [0.1, 0.5, 0.4, 0.9])
    assert m["rho"] > 0
    assert np.isnan(C.monotonicity([0, 1], [1.0, 0.0])["rho"])


def test_plan_dose_levels(tmp_path):
    trs = []
    for i in range(3):
        tr = _write_track(tmp_path / f"d{i}.wav", seconds=120.0, seed=i)
        tr.slug, tr.fold = f"d{i}", i
        tr.words = [(k * 4.0 + 1.8, k * 4.0 + 2.2) for k in range(60)]  # parola attiva sulle battute dispari
        trs.append(tr)
    recs = C.plan_dose(trs, n_bases=3, margin_s=13.0)
    fams = {r["family"] for r in recs}
    assert {"base", "phase", "gain", "key"} <= fams
    for r in recs:
        if r["family"] == "key":
            assert r["semitones"] == C.CAMELOT_STEP_SEMITONES[int(r["level"])] and r["vocoder"]
    x = C.synth_example([r for r in recs if r["family"] == "gain"][-1], {t.slug: t for t in trs}, sr=SR, length_s=24.0)
    assert len(x) == 24 * SR


def test_api_request_to_item(tmp_path):
    clip = np.zeros((20 * SR, 2), dtype=np.float32)
    it = C._item_from_request({"clip": clip, "sample_rate": SR, "metadata": {"start_s": 10.0, "switch_s": 11.0, "end_s": 12.0, "kind": "blend"},
                               "sources": {"a": {"audio": "/x/a.wav", "switch_s": 100.0}}, "id": "t1"}, 0, tmp_path)
    assert it.item_id == "api/t1" and it.clip_dur_s == pytest.approx(20.0)
    assert (it.start_s, it.switch_s, it.end_s) == (10.0, 11.0, 12.0) and it.a_switch_s == 100.0 and it.b_audio is None


# ------------------------------------------------------------------ v3 (mestiere DSP, ensemble, API)
def _click_track(sr, dur, bpm=90.0, phase_s=0.0, gain=1.0, seed=0):
    """Clic percussivi su ogni battito + accordo (do maggiore) tenuto, mono."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    x = 0.05 * (np.sin(2 * np.pi * 261.6 * t) + np.sin(2 * np.pi * 329.6 * t) + np.sin(2 * np.pi * 392.0 * t))
    period = 60.0 / bpm
    for b in np.arange(phase_s % period, dur, period):
        i = int(b * sr)
        n = min(int(0.03 * sr), len(x) - i)
        x[i:i + n] += 0.8 * rng.standard_normal(n) * np.exp(-np.arange(n) / (0.005 * sr))
    return (gain * x).astype(np.float32)


def test_craft_features_measure_phase_loudness_key():
    sr = C.SR_CRAFT
    a = _click_track(sr, 20.0)
    seam = 10.0
    n = int(seam * sr)
    nat = C.craft_features(a, sr, seam, side_s=8.0, vad=False)
    shifted = np.concatenate([a[:n], _click_track(sr, 20.0, phase_s=0.12)[n:]])
    ph = C.craft_features(shifted, sr, seam, side_s=8.0, vad=False)
    loud = np.concatenate([a[:n], 2.0 * a[n:]])
    lj = C.craft_features(loud, sr, seam, side_s=8.0, vad=False)
    assert nat["beat_phase_ms"] < 20.0
    assert 80.0 < ph["beat_phase_ms"] < 160.0
    assert nat["loud_jump_db"] < 0.5 and abs(lj["loud_jump_db"] - 6.02) < 0.5
    # trasposizione di un tritono del lato "dopo" (accordo diverso): profilo tonale più distante
    t = np.arange(len(a)) / sr
    trit = a.copy()
    trit[n:] = (0.05 * (np.sin(2 * np.pi * 370.0 * t) + np.sin(2 * np.pi * 466.2 * t) + np.sin(2 * np.pi * 554.4 * t)))[n:]
    assert C.craft_features(trit, sr, seam, side_s=8.0, vad=False)["chroma_dist"] > nat["chroma_dist"] + 0.3


def test_craft_flags_and_goodness():
    feats = {"loud_jump_db": 5.0, "beat_phase_ms": 10.0, "chroma_dist": 0.2, "vad_at_cut": 0.95}
    flags = {f["flag"] for f in C.craft_flags(feats)}
    assert flags == {"loud_jump_db", "vad_at_cut"}
    nulls = {k: list(np.linspace(0, 1, 50)) for k in C.CRAFT_FEATURES}
    g = C.craft_goodness({"loud_jump_db": 0.0, "beat_phase_ms": 2.0}, nulls)
    assert g["loud_jump_db"] > 0.95 and g["beat_phase_ms"] < 0.05 and np.isnan(g["chroma_dist"])


def test_fit_nonneg_logistic_signs():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 3))
    y = (x[:, 0] - x[:, 1] + 0.3 * rng.normal(size=400) > 0).astype(int)
    m = C.fit_nonneg_logistic(x, y)
    assert (m.coef >= 0).all()
    assert m.coef[0] > 0.5 and m.coef[1] < 1e-6  # la feature con effetto negativo resta a zero
    p = m.proba(x)
    assert p[y == 1].mean() > p[y == 0].mean()
    m2 = C.LinearSeamModel.from_json(json.loads(json.dumps(m.to_json())))
    assert np.allclose(m2.proba(x), p)


def test_score_transition_fast_api(tmp_path):
    sr = C.SR_CRAFT
    feats = [f"craft_{k}" for k in ("loud_jump_db", "beat_phase_ms")]
    m = C.fit_nonneg_logistic(np.array([[0.9, 0.9], [0.8, 0.7], [0.1, 0.2], [0.2, 0.1]]), np.array([1, 1, 0, 0]))
    (tmp_path / "models").mkdir()
    nulls = {k: list(np.linspace(0, 3 if k == "loud_jump_db" else 60, 40)) for k in C.CRAFT_FEATURES}
    (tmp_path / "models/v3_models.json").write_text(json.dumps({
        "fast": {"features": feats, "models": [m.to_json(), m.to_json()]}, "craft_nulls": nulls,
        "flag_rules": {k: list(v) for k, v in C.CRAFT_FLAG_RULES.items()}}))
    a = _click_track(sr, 20.0)
    n = int(10.0 * sr)
    good = C.score_transition(a, {"start_s": 10.0, "switch_s": 10.0, "end_s": 10.0}, mode="fast", sample_rate=sr,
                              models_dir=tmp_path)
    bad = C.score_transition(np.concatenate([a[:n], 3.0 * _click_track(sr, 20.0, phase_s=0.15)[n:]]),
                             {"start_s": 10.0, "switch_s": 10.0, "end_s": 10.0}, mode="fast", sample_rate=sr, models_dir=tmp_path)
    for r in (good, bad):
        assert set(r) >= {"score", "uncertainty", "parts", "flags", "timings_s"}
        assert r["timings_s"]["total"] < 60
    assert good["score"] > bad["score"]
    assert {f["flag"] for f in bad["flags"]} >= {"loud_jump_db", "beat_phase_ms"}
