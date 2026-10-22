"""Allineamento mix → brani su un mix sintetico con verità nota, prior di riferimento, verità UnmixDB."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from musicmix import mix_alignment as ma
from musicmix import reference_priors as rp

ROOT = Path(__file__).resolve().parents[1]
SR = 22050
BPM = 120.0
BEAT = 60.0 / BPM


def _synth_track(seed: int, dur: float = 48.0) -> np.ndarray:
    """Brano sintetico: triade casuale per battuta (timbro proprio), cassa sui battiti, hi-hat sugli ottavi."""
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.zeros(n, np.float32)
    bar = 4 * BEAT
    harm = rng.uniform(0.2, 1.0, size=6)
    kick_hz = 45 + 25 * rng.random()
    hat_every = int(rng.integers(1, 3))  # ottavi o sedicesimi
    hat_level = 0.05 + 0.25 * rng.random()
    for b in range(int(dur / bar) + 1):
        root = rng.integers(0, 12)
        notes = [root, root + rng.choice([3, 4]), root + 7]
        s, e = int(b * bar * SR), min(n, int((b + 1) * bar * SR))
        if s >= n:
            break
        tt = t[s:e]
        for nn in notes:
            f0 = 220.0 * 2 ** (nn / 12)
            for h, a in enumerate(harm, start=1):
                y[s:e] += (0.08 * a / h) * np.sin(2 * np.pi * f0 * h * tt)
    for k in range(int(dur / BEAT)):
        s = int(k * BEAT * SR)
        m = min(n - s, int(0.12 * SR))
        env = np.exp(-np.arange(m) / (0.03 * SR))
        y[s:s + m] += 0.6 * np.sin(2 * np.pi * (kick_hz + 40 * env) * np.arange(m) / SR) * env
        for sub in range(hat_every):
            h = int((k + 0.5 + sub * 0.25) * BEAT * SR)
            mh = min(n - h, int(0.04 * SR))
            if mh > 0:
                y[h:h + mh] += hat_level * rng.standard_normal(mh) * np.exp(-np.arange(mh) / (0.008 * SR))
    return (y / np.max(np.abs(y)) * 0.5).astype(np.float32)


@pytest.fixture(scope="module")
def synthetic_case():
    tr = {"t1": _synth_track(1), "t2": _synth_track(2), "t3": _synth_track(3)}
    beats_trk = np.arange(0, 48.0, BEAT)
    # mix: t1 sorgente 4..30 s (mix 0..26), taglio secco a 26 s su t2 da 8 s;
    # t2 fino a mix 50 s; blend lineare di 4 battiti (48..50 s) con t3 che entra dal suo secondo 6.
    n = int(70 * SR)
    mix = np.zeros(n, np.float32)

    def place(y, src0, mix0, mix1, gain=None):
        a, b = int(mix0 * SR), int(mix1 * SR)
        seg = y[int(src0 * SR): int(src0 * SR) + (b - a)]
        g = np.ones(len(seg), np.float32) if gain is None else gain[: len(seg)]
        mix[a: a + len(seg)] += seg * g

    place(tr["t1"], 4.0, 0.0, 26.0)
    fade = np.linspace(1, 0, int(2.0 * SR), dtype=np.float32)
    g2 = np.concatenate([np.ones(int(22.0 * SR), np.float32), fade])
    place(tr["t2"], 8.0, 26.0, 50.0, g2)
    g3 = np.concatenate([fade[::-1], np.ones(int(20 * SR), np.float32)])
    place(tr["t3"], 6.0, 48.0, 70.0, g3)
    cfg = ma.AlignConfig(min_diag_beats=8, key_shifts=(0,), tempo_multiples=(1.0,))
    mix_feat = ma.compute_features(mix, np.arange(0, 70.0, BEAT), cfg)
    feats = {k: ma.compute_features(v, beats_trk, cfg) for k, v in tr.items()}
    aligns = {k: ma.align_track(k, mix_feat, f, cfg) for k, f in feats.items()}
    order = sorted([k for k in aligns if aligns[k].found], key=lambda s: aligns[s].diag_mix_range[0])
    gains = ma.estimate_gains(mix_feat, feats, aligns, cfg)
    trans = ma.analyze_transitions(mix_feat, feats, aligns, gains, {}, cfg, order)
    return {"aligns": aligns, "order": order, "trans": trans, "truth": {"t1": (0.0, 4.0), "t2": (26.0, 8.0), "t3": (48.0, 6.0)},
            "mix_feat": mix_feat, "feats": feats, "cfg": cfg}


def test_order_and_time_map(synthetic_case):
    c = synthetic_case
    assert c["order"] == ["t1", "t2", "t3"]
    solo = {"t1": (2.0, 24.0), "t2": (28.0, 46.0), "t3": (52.0, 68.0)}
    for k, (a, b) in solo.items():
        ts = np.arange(a, b, 0.5)
        m0, s0 = c["truth"][k]
        err = np.abs(c["aligns"][k].to_track(ts) - (s0 + ts - m0))
        assert np.nanmedian(err) < 0.06, (k, np.nanmedian(err))
        assert np.mean(np.isfinite(err)) > 0.9


def test_transitions_cut_and_blend(synthetic_case):
    tr = {(t["a"], t["b"]): t for t in synthetic_case["trans"]}
    cut = tr[("t1", "t2")]
    assert abs(cut["mix_s"]["switch"] - 26.0) < BEAT
    assert cut["overlap_beats"] < 2.0
    assert cut["technique"] != "blend"
    blend = tr[("t2", "t3")]
    assert blend["technique"] == "blend"
    assert 1.5 <= blend["overlap_beats"] <= 6.0
    assert 47.0 <= blend["mix_s"]["switch"] <= 50.5
    # cue di B nel brano: secondo 6 + (switch − 48)
    b_src = blend["b_entry"]["at_switch"]["source_s"]
    assert abs(b_src - (6.0 + blend["mix_s"]["switch"] - 48.0)) < 0.1
    assert set(blend["curves_db"]["a"]) == {"full", "low", "mid", "high"}


def test_diag_groups_forgive_small_gaps():
    path = np.array([[i, i + 5] for i in range(20)] + [[20, 26], [21, 26]] + [[i, i + 5] for i in range(22, 40)])
    groups = ma._diag_groups(path, forgivable=2)
    assert max(len(g) for g in groups) >= 35


def test_classify_rules():
    cfg = ma.AlignConfig()
    base = {"overlap_beats": 0.0, "b_before_a_diag_end_beats": -4, "effect_beats": 4, "effect_mix_level_db": -6}
    assert ma.classify_technique({"overlap_beats": 8, "b_before_a_diag_end_beats": 6}, cfg)[0] == "blend"
    assert ma.classify_technique({"overlap_beats": 0.2, "effect_beats": 0.3}, cfg)[0] == "cut"
    assert ma.classify_technique(dict(base, effect_mix_decay_db_per_beat=9, effect_mix_level_db=-18, effect_a_presence_db=-30), cfg)[0] == "echo"
    assert ma.classify_technique(dict(base, effect_mix_decay_db_per_beat=0, effect_a_presence_db=-3, a_low_drop_rel_db=12, residual_rise_db=0), cfg)[0] == "filter"
    assert ma.classify_technique(dict(base, effect_mix_decay_db_per_beat=0, effect_a_presence_db=-3, a_low_drop_rel_db=0), cfg)[0] == "loop"
    assert ma.classify_technique(dict(base, effect_mix_level_db=-60), cfg)[0] == "cut"


def test_track_position_uses_grid_and_phrases():
    an = {"grid": {"downbeats_s": [0.5 + 2.0 * k for k in range(40)]},
          "phrases": [{"bar_index": k, "section_label": "verse" if k < 16 else "chorus", "bar_in_section": k if k < 16 else k - 16,
                       "phrase_level": 16 if k in (0, 16) else 0, "section_start": k in (0, 16), "bars_to_next_section": 0} for k in range(40)],
          "duration_s": 80.0}
    p = ma.track_position(an, 0.5 + 2.0 * 16 - 0.5)  # un battito prima della battuta 16
    assert p["nearest_downbeat_bar"] == 16 and p["section"] == "chorus" and p["bar_in_phrase16"] == 0
    assert p["beats_from_downbeat"] == pytest.approx(-1.0, abs=0.01)


def test_priors_absent_is_noop(tmp_path):
    assert rp.load_reference_priors(tmp_path / "manca.json") is None
    assert rp.prior_cost(None, technique="blend", overlap_beats=4) == (0.0, {})


def test_priors_cost_prefers_frequent_values(tmp_path):
    data = {"schema_version": rp.SCHEMA, "smoothing_alpha": 1.0, "domains": {
        "rap_user": {"weight": 1.0, "distributions": {
            "technique": {"counts": {"cut": 18, "blend": 2}},
            "overlap_beats": {"bins": [-64, 0.5, 4, 64], "counts": [15, 3, 2]},
            "entry_section": {"counts": {"verse": 10, "chorus": 10}}}},
        "electronic_raveform": {"weight": 0.1, "distributions": {"entry_section": {"counts": {"drop": 50, "breakdown": 5}}}}}}
    p = tmp_path / "pri.json"
    p.write_text(json.dumps(data))
    pri = rp.load_reference_priors(p)
    c_cut, _ = rp.prior_cost(pri, technique="cut", overlap_beats=0)
    c_blend, _ = rp.prior_cost(pri, technique="blend", overlap_beats=8)
    assert c_cut < 0 < c_blend
    # etichetta fuori vocabolario del dominio elettronico: nessun costo da quel dominio
    _, parts = rp.prior_cost(pri, entry_section="verse")
    assert "electronic_raveform" not in parts
    with pytest.raises(KeyError):
        rp.prior_cost(pri, sconosciuta=1)


def _load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_unmixdb_truth_parser(tmp_path):
    v = _load_tool("validate_reference_mix")
    lab = tmp_path / "m.labels.txt"
    lab.write_text("\n".join([
        "-2.0\t-2.0\t1\tstart\t\"a.excerpt40.mp3\"", "-2.0\t-2.0\t1\tspeed\t1", "0\t0\t1\tbpm\t120",
        "0\t7.5\t1\tfadein\t", "30\t30\t2\tstart\t\"b.excerpt40.mp3\"", "30\t30\t2\tspeed\t1.02",
        "33\t40\t1\tfadeout\t", "33\t40.5\t2\tfadein\t", "42\t42\t1\tstop\t\"a.excerpt40.mp3\""]) + "\n")
    t = v.truth_from_unmixdb(lab)
    assert t["order"] == ["a.excerpt40", "b.excerpt40"]
    tr = t["transitions"][0]
    assert tr["b_audible_start"] == 33 and tr["a_audible_end"] == 40
    assert tr["b_src_at_switch"] == pytest.approx((36.5 - 30) * 1.02)


def test_stats_distributions_from_reference(tmp_path):
    st = _load_tool("reference_mix_stats")
    ref = {"schema_version": "musicmix.reference-mix.1", "transitions": [
        {"technique": "cut", "overlap_beats": 0.1, "b_entry": {"at_switch": {"bar_in_phrase16": 0, "section": "verse", "track_fraction": 0.3, "beats_from_downbeat": 0.0}},
         "a_exit": {"at_switch": {"bar_in_phrase16": 0, "section": "chorus", "track_fraction": 0.7}}, "tempo": {"b_tempo_change_pct": 0.5}},
        {"technique": "blend", "overlap_beats": 8.0, "b_entry": {"at_switch": {"bar_in_phrase16": 4, "section": "intro"}},
         "a_exit": {"at_switch": {}}, "tempo": {"b_tempo_change_pct": -3.5}}]}
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(ref))
    out = tmp_path / "pri.json"
    assert st.main(["--mix", f"rap_user:1.0:{p}", "--out", str(out)]) == 0
    pri = rp.load_reference_priors(out)
    d = pri.domains["rap_user"]["distributions"]
    assert d["technique"]["counts"] == {"blend": 1, "cut": 1}
    assert d["overlap_beats_by_technique"]["blend"]["n"] == 1
    assert pri.domains["rap_user"]["n_transitions"] == 2


def test_missing_track_marks_pair_not_adjacent(synthetic_case):
    """Se t2 non è in libreria, t1 -> t3 non è una transizione osservata."""
    c = synthetic_case
    aligns = {k: c["aligns"][k] for k in ("t1", "t3")}
    feats = {k: c["feats"][k] for k in ("t1", "t3")}
    gains = ma.estimate_gains(c["mix_feat"], feats, aligns, c["cfg"])
    import dataclasses

    # il brano mancante sintetico dura 48 battiti e i brani sintetici condividono la batteria
    # (residuo più basso che nei mix reali): soglie del test, non quelle di default
    cfg = dataclasses.replace(c["cfg"], max_gap_beats=32.0, missing_residual=0.5)
    tr = ma.analyze_transitions(c["mix_feat"], feats, aligns, gains, {}, cfg, ["t1", "t3"])
    assert tr[0]["adjacent"] is False and tr[0]["technique"] == "unknown"
