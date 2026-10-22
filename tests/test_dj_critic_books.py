"""Test sintetici dei canali v4 del giudice (dj_critic_books) e delle aggiunte additive a
dj_render (deck ``a_raw``) e dj_qa (``rhythm_on_raw_a``). Nessun modello preaddestrato."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("librosa")
pytest.importorskip("pyloudnorm")

from musicmix import dj_critic as C  # noqa: E402
from musicmix import dj_critic_books as B  # noqa: E402
from musicmix import dj_qa  # noqa: E402

SR = 22050
BEAT = 0.5  # 120 BPM


def _burst(kind: str, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng({"kick": 1, "snare": 2, "hat": 3, "clap": 4}[kind])
    if kind == "kick":
        n = int(0.12 * sr)
        t = np.arange(n) / sr
        return np.sin(2 * np.pi * 60 * t) * np.exp(-t / 0.05)
    from scipy.signal import butter, sosfilt

    n = int((0.08 if kind in ("snare", "clap") else 0.03) * sr)
    lo, hi = {"snare": (300, 3000), "clap": (800, 3000), "hat": (6000, 9000)}[kind]
    sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    return sosfilt(sos, rng.standard_normal(n)) * np.exp(-np.arange(n) / (0.02 * sr)) * 3.0


def _place(x, burst, t, sr=SR, gain=1.0):
    i = int(round(t * sr))
    j = min(len(x), i + len(burst))
    if 0 <= i < len(x):
        x[i:j] += gain * burst[: j - i]


def _drums(dur, *, snare_shift=0.0, kick=True, snare=True, t0=0.0):
    x = np.zeros(int(dur * SR))
    k = 0
    t = t0
    while t < dur - 0.2:
        pos = k % 4
        if kick and pos in (0, 2):
            _place(x, _burst("kick"), t)
        if snare and pos in (1, 3):
            _place(x, _burst("snare"), t + snare_shift)
        t += BEAT
        k += 1
    return x


def _grid(dur, t0=0.0):
    beats = t0 + BEAT * np.arange(int((dur - t0) / BEAT) + 1)
    return beats, beats[::4]


# ------------------------------------------------------------------------ Q-2
def test_backbeat_warns_when_only_the_snare_of_b_is_20ms_late():
    dur = 12.0
    a = _drums(dur)
    beats, downs = _grid(dur)
    late = B.backbeat_alignment(a, _drums(dur, snare_shift=0.020), SR, beats, downs, 2.0, 10.0)
    assert late["status"] == "warn"
    assert late["backbeat_median_abs_ms"] == pytest.approx(20.0, abs=3.0)
    assert late["kick_median_abs_ms"] < 3.0  # cassa allineata: lo scarto è solo sul rullante
    same = B.backbeat_alignment(a, _drums(dur), SR, beats, downs, 2.0, 10.0)
    assert same["status"] == "pass" and same["backbeat_median_abs_ms"] < 3.0


def test_backbeat_is_none_without_separate_decks():
    beats, downs = _grid(8.0)
    r = B.backbeat_alignment(None, None, SR, beats, downs, 1.0, 6.0)
    assert r["status"] == "unavailable" and r["backbeat_median_abs_ms"] is None


# ------------------------------------------------------------------------ Q-3
def _intro_then_kick(intro_offset_beats: float, intro_bars=4, kick_bars=6):
    dur = (intro_bars + kick_bars) * 4 * BEAT + 0.5
    x = np.zeros(int(dur * SR))
    for k in range(intro_bars * 4):
        _place(x, _burst("clap"), k * BEAT + intro_offset_beats * BEAT)
    for k in range(intro_bars * 4, (intro_bars + kick_bars) * 4):
        _place(x, _burst("kick"), k * BEAT, gain=2.0)
        _place(x, _burst("hat"), k * BEAT + 0.5 * BEAT, gain=0.3)
    beats, downs = _grid(dur)
    return x, beats, downs, dur


def test_downbeat_after_kick_warns_for_an_intro_an_eighth_off():
    x, beats, downs, dur = _intro_then_kick(0.5)
    r = B.downbeat_after_kick(x, SR, beats, downs, region_s=(0.0, dur), a_audible_until_s=0.0)
    assert r["status"] == "warn", r
    assert abs(abs(r["phase_diff_beats"]) - 0.5) < 0.1
    assert r["kick_entry_s"] == pytest.approx(8.0, abs=0.01)


def test_downbeat_after_kick_passes_for_an_aligned_intro_or_when_a_still_plays():
    x, beats, downs, dur = _intro_then_kick(0.0)
    r = B.downbeat_after_kick(x, SR, beats, downs, region_s=(0.0, dur), a_audible_until_s=0.0)
    assert r["status"] == "pass" and abs(r["phase_diff_beats"]) < 0.1
    x, beats, downs, dur = _intro_then_kick(0.5)
    covered = B.downbeat_after_kick(x, SR, beats, downs, region_s=(0.0, dur), a_audible_until_s=8.0)
    assert covered["status"] == "pass" and "coperta" in covered["reason"]


# ------------------------------------------------------------------------ D1
class ClickTracker:
    """Tracker finto: attacchi sopra metà del massimo, distanti almeno 0,2 s; downbeat ogni 4."""

    def __call__(self, audio, sr):
        x = np.abs(np.asarray(audio, float))
        if x.ndim > 1:
            x = x.mean(axis=1)
        thr = 0.5 * x.max()
        idx, last = [], -10 ** 9
        for i in np.flatnonzero(x > thr):
            if i - last > int(0.2 * sr):
                idx.append(i)
            last = i if i - last > int(0.2 * sr) else last
        t = np.array(idx) / sr
        return {"beats": t, "downbeats": t[::4]}


def _clicks(dur, switch_s, b_shift_s, b_beat=BEAT):
    x = np.zeros(int(dur * SR))
    click = np.hanning(64)
    t = 0.25
    while t < switch_s - 1e-6:
        _place(x, click, t)
        t += BEAT
    t = switch_s + b_shift_s
    while t < dur - 0.1:
        _place(x, click, t)
        t += b_beat
    return x


def _meta(bars_from_cue: float, bar_s=4 * BEAT):
    sr = 48000
    return {"sample_rate": sr, "switch": {"a_source_sample": int((100.0 + bars_from_cue * bar_s) * sr)},
            "stretch": {"a_bar_s": bar_s}, "provenance": {"plan": {"a_out_s": 100.0}}}


def test_hypermetric_entry_profile():
    assert B.hypermetric_entry(_meta(4))["weight"] == 1.0
    assert B.hypermetric_entry(_meta(2))["weight"] == 0.5
    assert B.hypermetric_entry(_meta(1))["weight"] == 0.25
    assert B.hypermetric_entry({})["weight"] is None


def test_metric_surprise_zero_on_grid_and_high_when_b_is_off_grid():
    sw = 10.25
    ok = B.metric_surprise(_clicks(20.0, sw, 0.0), SR, sw, sw, sw, tracker=ClickTracker(), metadata=_meta(4))
    assert ok["status"] == "ok" and ok["grid_dev_ms"] < 2.0
    assert ok["metric_surprise"] == pytest.approx(0.0, abs=0.03) and ok["metric_ok"] is True
    off = B.metric_surprise(_clicks(20.0, sw, 0.060), SR, sw, sw, sw, tracker=ClickTracker(), metadata=_meta(4))
    assert off["grid_dev_ms"] == pytest.approx(60.0, abs=3.0)
    assert off["metric_ok"] is False and off["metric_surprise"] == pytest.approx(0.75, abs=0.05)
    weak = B.metric_surprise(_clicks(20.0, sw, 0.0), SR, sw, sw, sw, tracker=ClickTracker(), metadata=_meta(1))
    assert weak["metric_ok"] is False and weak["metric_surprise"] == pytest.approx(0.75)


def test_content_surprise_is_gated_by_the_metre():
    fps = C.MUSICGEN_FPS
    nll = np.random.default_rng(0).normal(10, 0.5, int(20 * fps))
    nll[int(10.2 * fps): int(10.5 * fps)] += 6
    g = B.content_surprise(nll, start_s=10.2, switch_s=10.2, end_s=10.2, metric={"metric_ok": True})
    assert g["gated"] is False and g["value"] is not None and g["value"] > 3
    off = B.content_surprise(nll, start_s=10.2, switch_s=10.2, end_s=10.2, metric={"metric_ok": False})
    assert off["gated"] is True and off["value"] is None and off["peak_z"] == g["peak_z"]
    assert B.content_surprise(None, start_s=1, switch_s=1, end_s=1)["value"] is None


def test_tempo_break_is_not_a_metric_surprise():
    m = {"creative": {"technique": "backspin", "params": {}}}
    r = B.metric_surprise(np.zeros(SR * 20), SR, 10, 10, 10, tracker=ClickTracker(), metadata=m)
    assert r["status"] == "not_applicable" and r["metric_surprise"] is None


# ------------------------------------------------------------------------ D2
def test_entropy_by_frame_follows_the_delay_pattern_and_uncertainty_fields_are_separate():
    ent = np.arange(2 * 4, dtype=float).reshape(2, 4)  # K=2, S=F+K-1=4 -> F=3
    assert B.entropy_by_frame(ent, 3).tolist() == [[0, 1, 2], [5, 6, 7]]
    fps = C.MUSICGEN_FPS
    n = int(20 * fps)
    e = np.full(n, 8.0)
    nl = np.full(n, 8.0) + np.random.default_rng(1).normal(0, 0.2, n)
    e[int(9 * fps): int(10 * fps)] = 3.0  # il modello è sicuro prima della giunta ...
    nl[int(10 * fps): int(11 * fps)] = 14.0  # ... e sbagliato dopo
    u = B.predictive_uncertainty(nl, e, start_s=10.0, switch_s=10.0, end_s=10.0)
    w = u["windows"]["1s"]
    assert w["entropy_before"] < 4 and w["nll_after"] > 13 and w["realized_minus_expected"] > 5
    assert B.predictive_uncertainty(None, None, start_s=1, switch_s=1, end_s=1)["status"] == "unavailable"


# ------------------------------------------------------------------------ D3
def test_window_profile_and_stability():
    rng = np.random.default_rng(2)
    x = np.concatenate([0.03 * rng.standard_normal(10 * SR), 0.1 * rng.standard_normal(10 * SR)])
    p = B.window_profile(x, SR, 10.0, 10.0)
    for k in ("1s", "3s", "5s"):
        assert p["windows"][k]["loud_jump_db"] == pytest.approx(10.5, abs=1.0)
    profiles = {i: {"windows": {k: {"loud_jump_db": float(i) + 0.1 * j} for j, k in enumerate(("1s", "3s", "5s"))}}
                for i in range(6)}
    st = B.window_stability(profiles, measures=("loud_jump_db",))
    assert st["loud_jump_db"]["1s~5s"]["spearman"] == pytest.approx(1.0)
    assert B.WINDOWS_S == (1.0, 3.0, 5.0)


# ------------------------------------------------------------------- D4, intent
def test_qa_dropout_and_declared_break_are_not_penalised():
    qa = {"verdict": "warn", "checks": {"dropout": {"status": "warn", "longest_s": 0.6, "intended": None}}}
    assert B.qa_dropout(qa)["warning"] is True
    assert B.qa_dropout({"checks": {}})["status"] == "absent"
    # clip: A a -20 dBFS, pausa voluta di un battito prima dello switch, B allo stesso livello
    rng = np.random.default_rng(3)
    sw = 10.0
    x = 0.1 * rng.standard_normal(20 * SR)
    x[int((sw - BEAT) * SR): int(sw * SR)] = 0.0
    craft = {"loud_jump_db": 6.0, "beat_phase_ms": 5.0, "chroma_dist": 0.2}  # salto "visto" includendo la pausa
    meta = {"sample_rate": SR, "switch": {"output_sample": int(sw * SR)}, "stretch": {"a_bar_s": 4 * BEAT},
            "creative": {"technique": "break_before_entry", "params": {"mode": "stop", "beats": 1},
                         "regions": [{"name": "stop before B", "output": [int((sw - BEAT) * SR), int(sw * SR)]}]}}
    intended = B.score_v4(x, SR, meta, start_s=sw, switch_s=sw, end_s=sw, craft=craft, qa=qa, tracker=None, windows=False)
    assert intended["intent"]["intended_gap"] == "technique break_before_entry"
    assert intended["technical"]["n_veto"] == 0 and not intended["technical"]["warnings"]
    plain = B.score_v4(x, SR, {"sample_rate": SR}, start_s=sw, switch_s=sw, end_s=sw, craft=craft, qa=qa, tracker=None,
                       windows=False)
    assert [f["flag"] for f in plain["technical"]["veto_flags"]] == ["loud_jump_db"]
    assert [w["check"] for w in plain["technical"]["warnings"]] == ["dropout"]
    assert plain["musical"]["diagnostic_only"] is True and "veto_flags" not in plain["musical"]


def test_repetition_checks_are_reported_not_recomputed():
    meta = {"creative": {"technique": "halving_build", "params": {"gap_beats": 1},
                         "repetition_checks": [{"unit_bars": 0.25, "repeats": 4, "ok": False, "warnings": ["disco rotto"]}]}}
    it = B.declared_intent(meta)
    assert it["loop_or_build"] is True and it["intended_gap"] == "gap_beats = 1"
    assert it["repetition_warnings"] == [{"unit_bars": 0.25, "repeats": 4, "warning": "disco rotto"}]


# ----------------------------------------------------------- dj_render a_raw / dj_qa
needs_stretch = pytest.mark.skipif(not (shutil.which("rubberband") or shutil.which("ffmpeg")), reason="no stretch backend")


def _click_track48(bpm, dur, bass_hz, gain=1.0):
    sr = 48000
    n = int(dur * sr)
    x = np.zeros(n)
    period = 60.0 / bpm
    burst = np.exp(-np.arange(480) / 60.0)
    beats, t, k = [], 0.5, 0
    while t < dur - 0.2:
        i = int(round(t * sr))
        f, amp = (3000.0, 1.0) if k % 4 == 0 else (1500.0, 0.35)
        x[i: i + 480] += amp * burst * np.sin(2 * np.pi * f * np.arange(480) / sr)
        beats.append(t)
        t += period
        k += 1
    x += 0.3 * np.sin(2 * np.pi * bass_hz * np.arange(n) / sr)
    beats = np.array(beats)
    return {"audio": x * gain, "beats": beats, "downbeats": beats[::4]}


class _GridTracker:
    """Tracker finto (niente Beat This!): attacchi del click oltre metà del massimo."""

    def __call__(self, audio, sr):
        x = np.abs(np.asarray(audio, float))
        x = x.mean(axis=1) if x.ndim > 1 else x
        hop = int(0.005 * sr)
        env = np.array([x[i:i + hop].max() for i in range(0, len(x) - hop, hop)])
        thr = 0.3 * env.max()
        on = np.flatnonzero((env[1:] > thr) & (env[:-1] <= thr)) + 1
        t = on * hop / sr
        return {"beats": t, "downbeats": t[::4]}


@needs_stretch
def test_render_exposes_a_raw_and_default_qa_is_unchanged():
    from musicmix import dj_render as dj

    a = _click_track48(120, 40, 55, gain=0.3)
    b = _click_track48(123, 40, 82, gain=1.0)
    r = dj.render_transition(a["audio"], b["audio"], beats_a=a["beats"], downbeats_a=a["downbeats"], beats_b=b["beats"],
                             downbeats_b=b["downbeats"], cue_out_a=a["downbeats"][6], cue_in_b=b["downbeats"][4], kind="blend",
                             bars=8, sample_rate=48000, config=dj.DJRenderConfig(cue_convention="start", bass_gap_beats=2),
                             return_decks=True)
    d = r.decks
    assert d["a_raw"].shape == d["a"].shape and d["a_raw_opt_in"] is True
    # A grezzo: prima di EQ e fader (alla fine della clip il fader di A è 0, il grezzo no)
    assert np.abs(d["a"][-1000:]).max() == 0.0 and np.abs(d["a_raw"][-48000:]).max() > 0.0
    tr = _GridTracker()
    legacy = {k: v for k, v in d.items() if k not in ("a_raw", "a_raw_opt_in")}
    q_new = dj_qa.evaluate_render(r.audio.astype(np.float64), r.metadata, sample_rate=48000, decks=d, tracker=tr)
    q_old = dj_qa.evaluate_render(r.audio.astype(np.float64), r.metadata, sample_rate=48000, decks=legacy, tracker=tr)
    assert q_new["verdict"] == q_old["verdict"]
    assert q_new["checks"]["rhythm"] == q_old["checks"]["rhythm"]
    assert "a_deck_measured" not in q_new["checks"]["rhythm"]
    q_raw = dj_qa.evaluate_render(r.audio.astype(np.float64), r.metadata, sample_rate=48000, decks=d, tracker=tr,
                                  config=dj_qa.QAConfig(rhythm_on_raw_a=True))
    assert q_raw["checks"]["rhythm"]["a_deck_measured"] == "a_raw"
    # deck grezzi di dj_creative/dj_effects (senza marcatore): usati come prima, con e senza l'opzione
    creative_like = dict(legacy, a_raw=d["a_raw"])
    q_c0 = dj_qa.evaluate_render(r.audio.astype(np.float64), r.metadata, sample_rate=48000, decks=creative_like, tracker=tr)
    q_c1 = dj_qa.evaluate_render(r.audio.astype(np.float64), r.metadata, sample_rate=48000, decks=creative_like, tracker=tr,
                                 config=dj_qa.QAConfig(rhythm_on_raw_a=True))
    assert q_c0["checks"]["rhythm"]["methods"] == q_c1["checks"]["rhythm"]["methods"]


# ------------------------------------------------------------------ API v4 additiva
@pytest.mark.data("runs/dj_critic_v1/models/v3_models.json")
@pytest.mark.skipif(not (C.ROOT / "runs/dj_critic_v1/models/v3_models.json").exists(), reason="modelli v3 assenti")
def test_api_v4_is_additive_to_v3(tmp_path):
    rng = np.random.default_rng(5)
    sr = 48000
    x = np.zeros(int(20 * sr))
    click = np.hanning(96)
    for k in range(40):
        i = int((0.25 + k * BEAT) * sr)
        x[i:i + 96] += click
    x += 0.02 * rng.standard_normal(len(x))
    clip = np.stack([x, x], 1).astype(np.float32)
    meta = {"start_s": 10.25, "switch_s": 10.25, "end_s": 10.25}
    v3 = C.score_transition(clip, meta, mode="fast", sample_rate=sr)
    v4 = C.score_transition(clip, meta, mode="fast", sample_rate=sr, version="v4")
    for k in ("score", "uncertainty", "flags", "version", "semantics", "mode"):
        assert v3[k] == v4[k] or (k == "score" and np.isnan(v3[k]) and np.isnan(v4[k]))
    assert "v4" not in v3
    b = v4["v4"]
    assert set(b) >= {"technical", "musical", "intent", "summary"}
    assert b["musical"]["diagnostic_only"] is True
    assert b["technical"]["backbeat"]["backbeat_median_abs_ms"] is None  # nessun deck: None, non un valore inventato
    with pytest.raises(ValueError):
        C.score_transition(clip, meta, mode="fast", sample_rate=sr, version="v5")


# --------------------------------------- technical block aligned with the set v3 rule (21/10/2026)
def _noise(sec=20, seed=5):
    return 0.1 * np.random.default_rng(seed).standard_normal(int(sec * SR))


def test_phase_flag_is_diagnostic_when_the_render_declares_the_phase():
    craft = {"loud_jump_db": 0.5, "beat_phase_ms": 120.0, "chroma_dist": 0.2}
    kw = dict(start_s=10.0, switch_s=12.0, end_s=14.0, craft=craft, qa=None, tracker=None, windows=False)
    blend = {"sample_rate": SR, "kind": "blend", "stretch": {"applied": True}, "downbeat_alignment": {"max_abs_error_ms": 0.4}}
    t = B.score_v4(_noise(), SR, blend, **kw)["technical"]
    assert t["n_veto"] == 0 and t["declared_phase"]["ms"] == pytest.approx(0.4)
    assert [f["flag"] for f in t["indicative_flags"]] == ["beat_phase_ms"] and t["indicative_flags"][0]["diagnostic_only"]
    # blend whose render residual is above 5 ms: the measured phase stays a veto
    bad = dict(blend, downbeat_alignment={"max_abs_error_ms": 12.0})
    assert [f["flag"] for f in B.score_v4(_noise(), SR, bad, **kw)["technical"]["veto_flags"]] == ["beat_phase_ms"]
    # entry without overlap (echo_out): B's downbeat on A's by construction; a low-confidence grid becomes a warning
    echo = {"sample_rate": SR, "kind": "echo_out", "effect": {"type": "echo_out"}, "stretch": {"applied": False},
            "downbeat_alignment": {"b_entry_output_sample": 14 * SR},
            "provenance": {"a": {"grid": {"confidence": "low"}}, "b": {"grid": {"confidence": "medium"}}}}
    t = B.score_v4(_noise(), SR, echo, **kw)["technical"]
    assert t["n_veto"] == 0 and t["status"] == "warn"
    assert [w["check"] for w in t["warnings"]] == ["grid_confidence"] and "A" in t["warnings"][0]["message"]
    # nothing declared (old metadata): veto as in v3
    assert B.score_v4(_noise(), SR, {"sample_rate": SR}, **kw)["technical"]["n_veto"] == 1


def test_key_distance_is_a_veto_only_with_harmonic_overlap():
    craft = {"loud_jump_db": 0.5, "beat_phase_ms": 5.0, "chroma_dist": 1.3}
    kw = dict(start_s=10.0, switch_s=12.0, end_s=14.0, craft=craft, qa=None, tracker=None, windows=False)
    blend = {"sample_rate": SR, "kind": "blend", "stretch": {"applied": True}, "downbeat_alignment": {"max_abs_error_ms": 0.4}}
    assert [f["flag"] for f in B.score_v4(_noise(), SR, blend, **kw)["technical"]["veto_flags"]] == ["chroma_dist"]
    echo = {"sample_rate": SR, "kind": "echo_out", "stretch": {"applied": False},
            "effect": {"type": "echo_out", "tail_vs_b_rms_db_first_4s": -18.0, "tail_stopped_on_b": False}}
    t = B.score_v4(_noise(), SR, echo, **kw)["technical"]
    assert t["n_veto"] == 0 and not t["harmonic_overlap"]["overlap"] and t["harmonic_overlap"]["echo_tail_vs_b_db"] == -18.0
    assert [f["flag"] for f in t["indicative_flags"]] == ["chroma_dist"]
    build = {"sample_rate": SR, "kind": "build_into_chorus", "creative": {"technique": "build_into_chorus"}, "stretch": {"applied": False}}
    assert B.score_v4(_noise(), SR, build, **kw)["technical"]["n_veto"] == 0
