"""Transizioni "da DJ hip-hop" senza beatmatch lungo: echo out, filter out, drop in.

Complemento di :mod:`musicmix.dj_render` (che fa ``cut`` e ``blend``). Come
quel modulo, qui non si scelgono coppie né cue: ``cue_out_a`` e ``cue_in_b``
arrivano già sul downbeat (tolleranza ``snap_tolerance_s``) da un altro modulo.
Riusa le funzioni pubbliche di ``dj_render`` (loudness BS.1770, limitatore
true-peak, ``split_bands``, ``choose_tempo_fold``, ``render_transition`` per il
drop) e restituisce lo stesso ``DJRenderResult`` con metadati dello stesso tipo
(``kind``, ``overlap``, ``switch``, ``stretch``, ``loudness``, ``limiter``,
``clip``, ``config``) più un blocco ``effect`` con i parametri dell'effetto.

Tutte le forme (tempi di delay, feedback, frequenze, curve) sono ipotesi di
pratica DJ [H], non stime da mix reali: nessuno studio empirico sulle
transizioni hip-hop è stato trovato (REVISIONE_DJ_2026-10-20 §3).

* ``echo_out``: A suona fino al downbeat di uscita ``cue_out_a``; da lì il
  segnale di A entra in un delay a tempo (1/2 o 1 battito della griglia di A)
  per un periodo di delay, poi il dry di A è tagliato. Le ripetizioni hanno
  feedback ``feedback`` (~0,5 = -6 dB a ripetizione) e un passa-alto nel loop
  con frequenza di taglio crescente (zero-phase, quindi il periodo resta
  esatto). B entra pulito sul suo downbeat ``cue_in_b`` dopo 0, 1 o 2 battute
  di A (``entry_bars``); la coda d'eco sotto B è attenuata di
  ``tail_under_b_db``. Nessuno stretch: B suona al suo tempo.
* ``filter_out``: passa-alto (Butterworth, ordine ``order``) in sweep
  geometrico su A per ``sweep_bars`` battute (1-4) che termina sul downbeat di
  uscita; poi taglio come ``dj_render`` (micro-crossfade che finisce sul
  downbeat) e B entra sul downbeat. Opzione ``filter_in``: passa-basso su B che
  si apre da ``lpf_start_hz`` a ``lpf_end_hz`` in ``filter_in_bars`` battute di
  B, poi ritorno al segnale dry con crossfade.
* ``drop_in``: B entra sul primo downbeat del ritornello/hook (``cue_in_b``,
  inizio sezione garantito dal chiamante) al posto di A sul downbeat di
  uscita. È un ``cut`` di ``dj_render`` con metadati del drop; niente riser.

Section gain match: parametro comune ``section_gain`` (``dj_render.SectionGainConfig``,
spento per default) con la stessa misura e la stessa rampa di ``dj_render``: A nelle
ultime battute prima dell'inizio dell'effetto (downbeat di uscita per echo_out,
inizio dello sweep per filter_out: A ancora dry), B nelle prime battute dal suo
ingresso (nessuno stretch); il guadagno si applica solo al deck B (la coda
d'eco resta di A). ``drop_in`` lo passa a ``dj_render.render_transition``.

Clip d'ascolto: 10 s prima dell'inizio dell'effetto (il downbeat di uscita per
echo/drop, l'inizio dello sweep per filter_out) + effetto + 10 s dopo
l'ingresso di B. Per il drop la clip coincide con quella del cut (10 + 10 s).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Optional

import numpy as np

from musicmix import dj_render as dj
from musicmix.dj_render import (
    DJRenderConfig,
    DJRenderError,
    DJRenderResult,
    SectionGainConfig,
    choose_tempo_fold,
    integrated_lufs,
    split_bands,
    true_peak_limit,
)

DJ_EFFECTS_VERSION = "musicmix.dj_effects.3"  # .2: section gain match (20/10/2026); .3: eco manuale, keep low, coda ferma (libri, 21/10/2026, spenti)
EFFECT_KINDS = ("echo_out", "filter_out", "drop_in")
ECHO_DELAY_BEATS = (0.5, 1.0)
ECHO_ENTRY_BARS = (0, 1, 2)
FILTER_SWEEP_BARS = (1, 2, 3, 4)


@dataclass(frozen=True)
class DJEffectsConfig:
    target_lufs: float = -14.0
    ceiling_dbtp: float = -1.0
    crossfade_ms: float = 10.0  # B entry / A cut micro-fade, 5..20 ms, ends on the downbeat
    snap_tolerance_s: float = 0.03
    pre_s: float = 10.0
    post_s: float = 10.0
    limiter_hold_ms: float = 20.0
    max_bar_deviation: float = 0.12
    loudness_window_s: float = 4.0
    # echo_out
    echo_feedback: float = 0.5
    echo_hpf_start_hz: float = 150.0
    echo_hpf_end_hz: float = 2500.0
    echo_hpf_order: int = 2
    echo_floor_db: float = -48.0  # stop repeating below this level
    echo_max_repeats: int = 16
    echo_edge_ms: float = 3.0  # fade at the edges of the captured fragment
    tail_under_b_db: Optional[float] = -12.0  # None: tail stopped on B's downbeat (clean switch, libri B1: Huron pp. 210-211)
    # ---- libri (21/10/2026): default neutri
    # [P] Webber 2008 pp. 265-266 (eco manuale: "Each time you repeat the sample, you stop the up fader at a slightly lower point
    # than the time before"): se impostato, il frammento si ripete a tempo ``echo_manual_repeats`` volte, ognuna echo_step_db
    # più bassa, senza filtro nel loop (le ripetizioni restano uguali). None = eco con feedback e HPF (comportamento .2).
    echo_manual_repeats: Optional[int] = None
    echo_step_db: float = -6.0
    # [P] Webber 2008 p. 181 ("cancel out everything but the kick and bass"); [P] B&B cap. 11 (cut tra parti percussive):
    # nel filter out la banda sotto filter_keep_low_hz di A resta piena fino al cut (A diventa cassa + basso + alti). None = spento.
    filter_keep_low_hz: Optional[float] = None
    # filter_out / filter_in
    hpf_start_hz: float = 30.0
    hpf_end_hz: float = 1200.0
    filter_order: int = 4
    filter_block: int = 128
    filter_engage_ms: float = 30.0  # dry -> filtered crossfade when the sweep starts
    lpf_start_hz: float = 250.0
    lpf_end_hz: float = 18000.0
    lpf_release_ms: float = 50.0  # filtered -> dry crossfade at the end of filter_in
    # section gain match, shared with dj_render (disabled by default)
    section_gain: SectionGainConfig = SectionGainConfig()


# ------------------------------------------------------------------ helpers


def _grid(values: Iterable[float]) -> np.ndarray:
    return np.sort(np.asarray(list(values), dtype=float))


def snap_downbeat(grid: np.ndarray, t: float, tol: float, label: str) -> int:
    """Indice del downbeat più vicino a ``t``; errore se dista più di ``tol``."""
    if len(grid) == 0:
        raise DJRenderError(f"{label}: empty downbeat grid")
    i = int(np.argmin(np.abs(grid - t)))
    if abs(grid[i] - t) > tol:
        raise DJRenderError(f"{label}={t:.3f}s is {1000 * abs(grid[i] - t):.0f} ms from the nearest downbeat")
    return i


def bar_and_beat(downbeats: np.ndarray, beats: np.ndarray, i: int, prefer: int, max_dev: float = 0.12, n: int = 4) -> dict:
    """Durata di battuta e battito vicino al downbeat ``i`` (lato preferito, poi l'altro).

    Battito = battuta / battiti per battuta (mediana dei battiti contati nelle
    stesse battute). Fallback: 4 x intervallo mediano tra battiti.
    """
    for side in (prefer, -prefer):
        for m in range(n, 0, -1):
            i0, i1 = (i - m, i) if side < 0 else (i, i + m)
            if i0 < 0 or i1 >= len(downbeats):
                continue
            bars = np.diff(downbeats[i0 : i1 + 1])
            if np.all(np.abs(bars / np.median(bars) - 1.0) <= max_dev):
                counts = [int(np.sum((beats >= downbeats[k] - 0.01) & (beats < downbeats[k + 1] - 0.01))) for k in range(i0, i1)]
                bpb = int(round(float(np.median(counts)))) if counts else 4
                bpb = bpb if bpb > 0 else 4
                bar = float((downbeats[i1] - downbeats[i0]) / m)
                return {
                    "bar_s": bar,
                    "beats_per_bar": bpb,
                    "beat_s": bar / bpb,
                    "bpm": 60.0 * bpb / bar,
                    "source": f"{m} regular bars {'before' if side < 0 else 'after'} the cue",
                }
    t = downbeats[i]
    sel = beats[(beats >= t - 8.0) & (beats <= t + 8.0)]
    if len(sel) >= 3:
        beat = float(np.median(np.diff(sel)))
        return {"bar_s": 4.0 * beat, "beats_per_bar": 4, "beat_s": beat, "bpm": 60.0 / beat, "source": "median beat interval (irregular downbeat grid)"}
    raise DJRenderError("cannot estimate bar/beat length near the cue")


def _slice(x: np.ndarray, start: int, n: int) -> np.ndarray:
    out = np.zeros((n, x.shape[1]))
    lo, hi = max(0, start), min(len(x), start + n)
    if hi > lo:
        out[lo - start : hi - start] = x[lo:hi]
    return out


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def _as_stereo(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return x[:, :2]


def time_varying_filter(x: np.ndarray, sr: int, cutoff_hz: np.ndarray, btype: str, order: int = 4, block: int = 128) -> np.ndarray:
    """Butterworth causale con frequenza di taglio variabile a blocchi.

    ``cutoff_hz`` ha un valore per campione; ogni blocco usa il valore al suo
    inizio e lo stato dei biquad passa da un blocco all'altro (sweep senza
    reset). Causale come il filtro di un mixer: introduce lo sfasamento del
    Butterworth, che qui non conta perché A esce e B è ancora solo.
    """
    from scipy.signal import butter, sosfilt

    x = _as_stereo(x)
    nyq = 0.5 * sr
    y = np.zeros_like(x)
    zi = None
    last_fc = None
    sos = None
    for s in range(0, len(x), block):
        fc = float(np.clip(cutoff_hz[s], 5.0, 0.98 * nyq))
        if last_fc is None or abs(fc / last_fc - 1.0) > 1e-6:
            sos = butter(order, fc, btype=btype, fs=sr, output="sos")
            last_fc = fc
        if zi is None:
            zi = np.zeros((sos.shape[0], 2, x.shape[1]))
        y[s : s + block], zi = sosfilt(sos, x[s : s + block], axis=0, zi=zi)
    return y


def _geom_ramp(n: int, i0: int, i1: int, f0: float, f1: float) -> np.ndarray:
    """f0 prima di i0, rampa geometrica f0 -> f1 in [i0, i1], f1 dopo."""
    idx = np.arange(n)
    u = np.clip((idx - i0) / max(i1 - i0, 1), 0.0, 1.0)
    return f0 * (f1 / f0) ** u


def _normalize(a: np.ndarray, b: np.ndarray, sr: int, cfg: DJEffectsConfig):
    lufs_a = integrated_lufs(a, sr)
    lufs_b = integrated_lufs(b, sr)
    gain_a_db = (cfg.target_lufs - lufs_a) if np.isfinite(lufs_a) else 0.0
    gain_b_db = (cfg.target_lufs - lufs_b) if np.isfinite(lufs_b) else 0.0
    a = a * 10.0 ** (gain_a_db / 20.0)
    b = b * 10.0 ** (gain_b_db / 20.0)
    loud = {
        "target_lufs": cfg.target_lufs,
        "a_integrated_lufs_source": lufs_a,
        "b_integrated_lufs_source": lufs_b,
        "a_gain_db": gain_a_db,
        "b_gain_db": gain_b_db,
        "a_integrated_lufs_normalized": integrated_lufs(a, sr),
        "b_integrated_lufs_normalized": integrated_lufs(b, sr),
        "scope": "whole track, ITU-R BS.1770-4 via pyloudnorm",
    }
    return a, b, loud


def _b_deck(b: np.ndarray, sb: int, entry_out: int, n: int, sr: int, cfg: DJEffectsConfig) -> np.ndarray:
    """B source sample ``sb`` lands on output sample ``entry_out``; silent before,
    with an equal-power micro fade-in that *ends* on the downbeat (as dj_render cut)."""
    d = int(round(cfg.crossfade_ms / 1000.0 * sr))
    deck = _slice(b, sb - entry_out, n)
    t = np.arange(n)
    g = np.sin(0.5 * np.pi * np.clip((t - (entry_out - d)) / max(d, 1), 0.0, 1.0))
    return deck * g[:, None]


def _a_cut_gain(n: int, cut_out: int, sr: int, cfg: DJEffectsConfig) -> np.ndarray:
    d = int(round(cfg.crossfade_ms / 1000.0 * sr))
    t = np.arange(n)
    g = np.cos(0.5 * np.pi * np.clip((t - (cut_out - d)) / max(d, 1), 0.0, 1.0))
    g[t >= cut_out] = 0.0  # exact silence from the downbeat on (cos(pi/2) != 0 in floating point)
    return g


def _tempo_info(grid: dict, ia: int, jb: int, cfg: DJEffectsConfig) -> tuple[dict, dict, dict]:
    ta = bar_and_beat(grid["downbeats_a"], grid["beats_a"], ia, -1, cfg.max_bar_deviation)
    tb = bar_and_beat(grid["downbeats_b"], grid["beats_b"], jb, +1, cfg.max_bar_deviation)
    k, rate = choose_tempo_fold(ta["bar_s"], tb["bar_s"], max_stretch=10.0)
    stretch = {
        "applied": False,
        "tempo_ratio_b": 1.0,
        "note": "no stretch: B plays at its own tempo after a clean entry",
        "a_bar_s": ta["bar_s"],
        "b_bar_s": tb["bar_s"],
        "a_bar_source": ta["source"],
        "b_bar_source": tb["source"],
        "a_bpm_local": ta["bpm"],
        "b_bpm_local": tb["bpm"],
        "a_beat_s": ta["beat_s"],
        "a_beats_per_bar": ta["beats_per_bar"],
        "fold_k": k,
        "tempo_jump_pct": 100.0 * (1.0 / rate - 1.0),
    }
    return ta, tb, stretch


def _finish(out, decks, meta, loud, sr, cfg, *, kind, grid, ia, jb, cue_out_a, cue_in_b, effect_start, b_entry, clip_rule, provenance, return_decks):
    w = int(round(cfg.loudness_window_s * sr))
    before = integrated_lufs(out[max(0, effect_start - w) : effect_start], sr)
    after = integrated_lufs(out[b_entry : b_entry + w], sr)
    loud["lufs_before_effect"] = before
    loud["lufs_after_b_entry"] = after
    loud["loudness_window_s"] = cfg.loudness_window_s
    loud["loudness_jump_db"] = (after - before) if np.isfinite(before) and np.isfinite(after) else None
    loud["section_gain_b_db"] = (meta.get("section_gain") or {}).get("applied_db", 0.0)
    limited, limiter = true_peak_limit(out, sr, cfg.ceiling_dbtp, cfg.limiter_hold_ms)
    loud["output_integrated_lufs"] = integrated_lufs(limited, sr)
    loud["output_true_peak_dbtp"] = limiter["true_peak_after_dbtp"]
    expected = int(round(cfg.pre_s * sr)) + (b_entry - effect_start) + int(round(cfg.post_s * sr))
    meta.update({
        "schema_version": DJ_EFFECTS_VERSION,
        "renderer_base": dj.DJ_RENDER_VERSION,
        "kind": kind,
        "sample_rate": sr,
        "cue_out_a_s": float(grid["downbeats_a"][ia]),
        "cue_in_b_s": float(grid["downbeats_b"][jb]),
        "cue_out_a_requested_s": float(cue_out_a),
        "cue_in_b_requested_s": float(cue_in_b),
        "loudness": loud,
        "limiter": limiter,
        "clip": {
            "rule": clip_rule,
            "pre_s": cfg.pre_s,
            "post_s": cfg.post_s,
            "effect_start_sample": effect_start,
            "b_entry_sample": b_entry,
            "length_samples": len(limited),
            "duration_s": len(limited) / sr,
            "matches_rule": len(limited) == expected and cfg.pre_s == 10.0 and cfg.post_s == 10.0,
        },
        "config": asdict(cfg),
    })
    if provenance:
        meta["provenance"] = provenance
    result = DJRenderResult(limited.astype(np.float32), sr, _jsonable(meta))
    if return_decks:
        result.decks = decks
    return result


def _prepare(audio_a, audio_b, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b, sr, cfg):
    if not 5.0 <= cfg.crossfade_ms <= 20.0:
        raise DJRenderError("crossfade must be 5..20 ms")
    a = _as_stereo(audio_a)
    b = _as_stereo(audio_b)
    grid = {
        "beats_a": _grid(beats_a),
        "downbeats_a": _grid(downbeats_a),
        "beats_b": _grid(beats_b),
        "downbeats_b": _grid(downbeats_b),
    }
    ia = snap_downbeat(grid["downbeats_a"], cue_out_a, cfg.snap_tolerance_s, "cue_out_a")
    jb = snap_downbeat(grid["downbeats_b"], cue_in_b, cfg.snap_tolerance_s, "cue_in_b")
    a, b, loud = _normalize(a, b, sr, cfg)
    return a, b, grid, ia, jb, loud


# ------------------------------------------------------------------ echo out


def echo_out(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    *,
    beats_a: Iterable[float],
    downbeats_a: Iterable[float],
    beats_b: Iterable[float],
    downbeats_b: Iterable[float],
    cue_out_a: float,
    cue_in_b: float,
    delay_beats: float = 1.0,
    entry_bars: int = 1,
    sample_rate: int = dj.DEFAULT_SAMPLE_RATE,
    config: Optional[DJEffectsConfig] = None,
    provenance: Optional[dict] = None,
    return_decks: bool = False,
) -> DJRenderResult:
    """Echo out di A sul downbeat di uscita, B pulito sul suo downbeat dopo ``entry_bars``."""
    cfg = config or DJEffectsConfig()
    sr = int(sample_rate)
    if delay_beats not in ECHO_DELAY_BEATS:
        raise DJRenderError(f"delay_beats must be one of {ECHO_DELAY_BEATS}")
    if entry_bars not in ECHO_ENTRY_BARS:
        raise DJRenderError(f"entry_bars must be one of {ECHO_ENTRY_BARS}")
    if not 0.0 < cfg.echo_feedback < 1.0:
        raise DJRenderError("echo feedback must be in (0, 1)")
    a, b, grid, ia, jb, loud = _prepare(audio_a, audio_b, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b, sr, cfg)
    ta, tb, stretch = _tempo_info(grid, ia, jb, cfg)
    da = grid["downbeats_a"]

    pre = int(round(cfg.pre_s * sr))
    post = int(round(cfg.post_s * sr))
    sa = int(round(da[ia] * sr))
    off_a = sa - pre  # output n <-> A source n + off_a
    # B enters on the downbeat where A's bar `ia + entry_bars` would fall (A grid if regular, else bar estimate).
    if entry_bars == 0:
        gap_s, gap_src = 0.0, "on the exit downbeat"
    elif ia + entry_bars < len(da) and abs((da[ia + entry_bars] - da[ia]) / (entry_bars * ta["bar_s"]) - 1.0) <= cfg.max_bar_deviation:
        gap_s, gap_src = float(da[ia + entry_bars] - da[ia]), "A downbeat grid continued after the exit"
    else:
        gap_s, gap_src = entry_bars * ta["bar_s"], "A local bar length x entry_bars"
    entry = pre + int(round(gap_s * sr))
    n = entry + post
    sb = int(round(grid["downbeats_b"][jb] * sr))

    D = int(round(delay_beats * ta["beat_s"] * sr))
    if D < int(0.05 * sr):
        raise DJRenderError("echo delay shorter than 50 ms: check A's beat grid")
    dry_end = min(pre + D, entry)  # dry A stops after one delay period (or when B enters)
    deck_a = _slice(a, off_a, n) * _a_cut_gain(n, dry_end, sr, cfg)[:, None]

    # Captured fragment: A from the exit downbeat (with a short pre-roll so the
    # downbeat transient is whole), one delay period long, edge-faded.
    edge = max(1, int(round(cfg.echo_edge_ms / 1000.0 * sr)))
    frag_src0 = sa - edge
    frag = _slice(a, frag_src0, D)
    win = np.ones(D)
    win[:edge] = np.sin(0.5 * np.pi * np.arange(edge) / edge) ** 2
    win[-edge:] = np.minimum(win[-edge:], np.cos(0.5 * np.pi * (np.arange(edge) + 1) / edge) ** 2)
    frag *= win[:, None]

    manual = cfg.echo_manual_repeats is not None
    if manual and not (1 <= int(cfg.echo_manual_repeats) <= 8 and cfg.echo_step_db < 0):
        raise DJRenderError("echo_manual_repeats in 1..8 and echo_step_db < 0")
    fb = 10.0 ** (cfg.echo_step_db / 20.0) if manual else cfg.echo_feedback
    n_rep = int(cfg.echo_manual_repeats) if manual else int(min(cfg.echo_max_repeats, math.floor(cfg.echo_floor_db / (20.0 * math.log10(fb)))))
    n_rep = max(n_rep, 1)
    from scipy.signal import butter, sosfiltfilt

    wet = np.zeros((n, 2))
    repeats = []
    cur = frag.copy()
    pad = min(len(cur) - 1, int(0.1 * sr))
    for k in range(1, n_rep + 1):
        u = (k - 1) / max(n_rep - 1, 1)
        fc = cfg.echo_hpf_start_hz * (cfg.echo_hpf_end_hz / cfg.echo_hpf_start_hz) ** u
        if manual:  # Webber pp. 265-266: same fragment, fader a step lower each time
            fc = None
            cur = cur * fb
        else:
            sos = butter(cfg.echo_hpf_order, fc, btype="high", fs=sr, output="sos")
            cur = sosfiltfilt(sos, cur, axis=0, padlen=pad) * fb  # cumulative: loop filter + feedback
        at = (frag_src0 - off_a) + k * D
        if at >= n:
            break
        hi = min(n, at + D)
        wet[at:hi] += cur[: hi - at]
        repeats.append({"repeat": k, "output_sample": at + edge, "gain_db": 20.0 * k * math.log10(fb), "hpf_hz": fc})
    # Tail under B: attenuate with a 20 ms ramp ending on B's downbeat.
    ramp = max(1, int(0.02 * sr))
    t = np.arange(n)
    under = np.clip((t - (entry - ramp)) / ramp, 0.0, 1.0)
    tail_gain = 1.0 - under * (1.0 - (10.0 ** (cfg.tail_under_b_db / 20.0) if cfg.tail_under_b_db is not None else 0.0))
    wet *= tail_gain[:, None]

    deck_b = _b_deck(b, sb, entry, n, sr, cfg)
    b_ref = b[sb : sb + int(round(cfg.section_gain.b_bars * tb["bar_s"] * sr))]
    sg_curve, sg_info = dj.plan_section_gain(a, sa, ta["bar_s"], b_ref, entry, tb["bar_s"], n, sr, cfg.section_gain,
                                             transition_end_out=entry, b_ref_note="B source (no stretch) from its entry downbeat; A measured before the exit downbeat (dry)")
    deck_b = deck_b * sg_curve[:, None]
    out = deck_a + wet + deck_b

    w = int(min(post, 4 * sr))
    rms = lambda x: float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    tail_rel = None
    if rms(deck_b[entry : entry + w]) > 0 and rms(wet[entry : entry + w]) > 0:
        tail_rel = 20.0 * math.log10(rms(wet[entry : entry + w]) / rms(deck_b[entry : entry + w]))
    meta = {
        "bars": 0,
        "overlap": {
            "output_start_sample": pre,
            "output_end_sample": entry,
            "a_source_start_sample": sa,
            "a_source_end_sample": entry + off_a,
            "b_source_start_sample": sb,
            "b_source_end_sample": sb,
            "duration_s": (entry - pre) / sr,
            "note": "effect region = exit downbeat of A -> entry downbeat of B; the echo tail continues under B, attenuated",
        },
        "switch": {"output_sample": entry, "a_source_sample": entry + off_a, "b_source_sample": sb},
        "transition_duration_s": (entry - pre) / sr,
        "stretch": stretch,
        "downbeat_alignment": {
            "b_entry_output_sample": entry,
            "b_entry_source_sample": sb,
            "note": "B's cue_in downbeat sample is placed exactly on the entry sample; no beatmatch (echo tail keeps A's tempo)",
        },
        "effect": {
            "type": "echo_out",
            "delay_beats": delay_beats,
            "delay_s": D / sr,
            "delay_samples": D,
            "a_beat_s": ta["beat_s"],
            "feedback": fb,
            "hpf_in_loop": {"order": cfg.echo_hpf_order, "start_hz": cfg.echo_hpf_start_hz, "end_hz": cfg.echo_hpf_end_hz, "filter": "zero-phase Butterworth, cumulative across repeats"},
            "captured_fragment": {"a_source_start_sample": frag_src0, "length_samples": D, "edge_ms": cfg.echo_edge_ms},
            "dry_a_cut_output_sample": dry_end,
            "repeats": repeats,
            "entry_bars": entry_bars,
            "entry_gap_s": gap_s,
            "entry_gap_source": gap_src,
            "tail_under_b_db": cfg.tail_under_b_db,
            "tail_stopped_on_b": cfg.tail_under_b_db is None,
            "manual_echo": {"repeats": n_rep, "step_db": cfg.echo_step_db, "source": "[P] Webber 2008 pp. 265-266"} if manual else None,
            "tail_vs_b_rms_db_first_4s": tail_rel,
            "basis": "[H] hip-hop DJ practice (echo out on the one, B in clean on its downbeat); not measured on real mixes",
        },
        "vocals": {"status": "not applicable (no overlap of dry decks)"},
        "section_gain": sg_info,
    }
    decks = {"a": deck_a, "echo": wet, "b": deck_b, "gains": {"tail_gain": tail_gain, "section_gain_b": sg_curve}}
    return _finish(
        out, decks, meta, loud, sr, cfg, kind="echo_out", grid=grid, ia=ia, jb=jb,
        cue_out_a=cue_out_a, cue_in_b=cue_in_b, effect_start=pre, b_entry=entry,
        clip_rule="echo_out: 10 s before the exit downbeat + effect (exit -> B entry) + 10 s after B entry",
        provenance=provenance, return_decks=return_decks,
    )


# ---------------------------------------------------------------- filter out


def filter_out(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    *,
    beats_a: Iterable[float],
    downbeats_a: Iterable[float],
    beats_b: Iterable[float],
    downbeats_b: Iterable[float],
    cue_out_a: float,
    cue_in_b: float,
    sweep_bars: int = 2,
    filter_in: bool = False,
    filter_in_bars: int = 2,
    sample_rate: int = dj.DEFAULT_SAMPLE_RATE,
    config: Optional[DJEffectsConfig] = None,
    provenance: Optional[dict] = None,
    return_decks: bool = False,
) -> DJRenderResult:
    """Passa-alto in sweep su A fino al downbeat di uscita, poi B sul downbeat."""
    cfg = config or DJEffectsConfig()
    sr = int(sample_rate)
    if sweep_bars not in FILTER_SWEEP_BARS:
        raise DJRenderError(f"sweep_bars must be one of {FILTER_SWEEP_BARS}")
    if filter_in and filter_in_bars not in FILTER_SWEEP_BARS:
        raise DJRenderError(f"filter_in_bars must be one of {FILTER_SWEEP_BARS}")
    a, b, grid, ia, jb, loud = _prepare(audio_a, audio_b, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b, sr, cfg)
    ta, tb, stretch = _tempo_info(grid, ia, jb, cfg)
    da, db = grid["downbeats_a"], grid["downbeats_b"]
    if ia - sweep_bars < 0:
        raise DJRenderError(f"filter_out: needs {sweep_bars} A bars before cue_out_a")
    bars = np.diff(da[ia - sweep_bars : ia + 1])
    if np.any(np.abs(bars / ta["bar_s"] - 1.0) > cfg.max_bar_deviation):
        raise DJRenderError(f"filter_out: irregular A bars in the sweep ({np.round(bars, 2).tolist()} s)")

    pre = int(round(cfg.pre_s * sr))
    post = int(round(cfg.post_s * sr))
    sa = int(round(da[ia] * sr))
    s0 = int(round(da[ia - sweep_bars] * sr))
    off_a = s0 - pre
    sweep_start, cut = pre, sa - off_a
    n = cut + post
    sb = int(round(db[jb] * sr))

    # A: filter state warmed at hpf_start before the sweep, dry -> filtered crossfade at sweep start.
    raw_a = _slice(a, off_a, n)
    fc_a = _geom_ramp(n, sweep_start, cut, cfg.hpf_start_hz, cfg.hpf_end_hz)
    filt_a = time_varying_filter(raw_a, sr, fc_a, "highpass", cfg.filter_order, cfg.filter_block)
    eng = max(1, int(round(cfg.filter_engage_ms / 1000.0 * sr)))
    mix = np.clip((np.arange(n) - sweep_start) / eng, 0.0, 1.0)[:, None]
    deck_a_pre = raw_a * (1.0 - mix) + filt_a * mix
    keep = None
    if cfg.filter_keep_low_hz:  # Webber p. 181: everything but the kick and bass goes; the low band stays until the cut
        from scipy.signal import butter, sosfiltfilt

        fk = float(cfg.filter_keep_low_hz)
        lo_raw = sosfiltfilt(butter(4, fk, btype="low", fs=sr, output="sos"), raw_a, axis=0)
        lo_filt = sosfiltfilt(butter(4, fk, btype="low", fs=sr, output="sos"), deck_a_pre, axis=0)
        deck_a_pre = deck_a_pre - lo_filt + lo_raw
        keep = {"low_hz": fk, "source": "[P] Webber 2008 p. 181; [P] B&B cap. 11",
                "note": "A's band under low_hz is kept dry through the sweep (zero-phase LR4-like split)"}
    deck_a = deck_a_pre * _a_cut_gain(n, cut, sr, cfg)[:, None]

    deck_b = _b_deck(b, sb, cut, n, sr, cfg)
    b_ref = b[sb : sb + int(round(cfg.section_gain.b_bars * tb["bar_s"] * sr))]
    sg_curve, sg_info = dj.plan_section_gain(a, s0, ta["bar_s"], b_ref, cut, tb["bar_s"], n, sr, cfg.section_gain,
                                             transition_end_out=cut, b_ref_note="B source (no stretch) from its entry downbeat; A measured before the sweep start (dry)")
    deck_b = deck_b * sg_curve[:, None]
    fin = {"enabled": False}
    if filter_in:
        tb_in = bar_and_beat(db, grid["beats_b"], jb, +1, cfg.max_bar_deviation)
        if jb + filter_in_bars < len(db) and abs((db[jb + filter_in_bars] - db[jb]) / (filter_in_bars * tb_in["bar_s"]) - 1.0) <= cfg.max_bar_deviation:
            open_len = int(round((db[jb + filter_in_bars] - db[jb]) * sr))
            open_src = "B downbeat grid"
        else:
            open_len = int(round(filter_in_bars * tb_in["bar_s"] * sr))
            open_src = "B local bar length x filter_in_bars"
        if open_len > post:
            raise DJRenderError("filter_in longer than the 10 s after B's entry: use fewer bars")
        open_end = cut + open_len
        fc_b = _geom_ramp(n, cut, open_end, cfg.lpf_start_hz, cfg.lpf_end_hz)
        filt_b = time_varying_filter(deck_b, sr, fc_b, "lowpass", cfg.filter_order, cfg.filter_block)
        rel = max(1, int(round(cfg.lpf_release_ms / 1000.0 * sr)))
        back = np.clip((np.arange(n) - open_end) / rel, 0.0, 1.0)[:, None]
        deck_b = filt_b * (1.0 - back) + deck_b * back
        fin = {
            "enabled": True,
            "bars": filter_in_bars,
            "start_hz": cfg.lpf_start_hz,
            "end_hz": cfg.lpf_end_hz,
            "order": cfg.filter_order,
            "output_start_sample": cut,
            "output_end_sample": open_end,
            "length_source": open_src,
            "release_to_dry_ms": cfg.lpf_release_ms,
            "note": "B's attack on the downbeat passes through the closed low-pass: its timing is kept, its brightness opens",
        }
    out = deck_a + deck_b

    meta = {
        "bars": 0,
        "overlap": {
            "output_start_sample": sweep_start,
            "output_end_sample": cut,
            "a_source_start_sample": s0,
            "a_source_end_sample": sa,
            "b_source_start_sample": sb,
            "b_source_end_sample": sb,
            "duration_s": (cut - sweep_start) / sr,
            "note": "effect region = high-pass sweep on A; B enters with a cut on A's exit downbeat",
        },
        "switch": {"output_sample": cut, "a_source_sample": sa, "b_source_sample": sb},
        "transition_duration_s": (cut - sweep_start) / sr,
        "stretch": stretch,
        "downbeat_alignment": {
            "switch_error_ms": 0.0,
            "b_entry_output_sample": cut,
            "b_entry_source_sample": sb,
            "note": "B's cue_in downbeat is placed on A's cue_out downbeat sample (cut)",
        },
        "effect": {
            "type": "filter_out",
            "sweep_bars": sweep_bars,
            "hpf": {
                "order": cfg.filter_order,
                "start_hz": cfg.hpf_start_hz,
                "end_hz": cfg.hpf_end_hz,
                "curve": "geometric (linear in log-frequency) from sweep start to the exit downbeat",
                "filter": "causal Butterworth, coefficients updated every filter_block samples, state carried",
                "engage_crossfade_ms": cfg.filter_engage_ms,
            },
            "sweep_output_samples": [sweep_start, cut],
            "a_downbeat_indices": [ia - sweep_bars, ia],
            "filter_in": fin,
            "keep_low": keep,
            "basis": "[H] hip-hop DJ practice (filter out A, B on the one); not measured on real mixes",
        },
        "vocals": {"status": "not applicable (no overlap)"},
        "section_gain": sg_info,
    }
    decks = {"a": deck_a, "a_filtered_prefader": deck_a_pre, "a_raw": raw_a, "b": deck_b, "gains": {"hpf_a_hz": fc_a, "section_gain_b": sg_curve}}
    return _finish(
        out, decks, meta, loud, sr, cfg, kind="filter_out", grid=grid, ia=ia, jb=jb,
        cue_out_a=cue_out_a, cue_in_b=cue_in_b, effect_start=sweep_start, b_entry=cut,
        clip_rule="filter_out: 10 s before the sweep start + sweep + 10 s after B entry",
        provenance=provenance, return_decks=return_decks,
    )


# ------------------------------------------------------------------- drop in


def drop_in(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    *,
    beats_a: Iterable[float],
    downbeats_a: Iterable[float],
    beats_b: Iterable[float],
    downbeats_b: Iterable[float],
    cue_out_a: float,
    cue_in_b: float,
    cue_in_is_section_start: bool = True,
    section_label: Optional[str] = None,
    sample_rate: int = dj.DEFAULT_SAMPLE_RATE,
    config: Optional[DJEffectsConfig] = None,
    provenance: Optional[dict] = None,
    return_decks: bool = False,
) -> DJRenderResult:
    """B entra sul primo downbeat del ritornello/hook al posto di A (cut sul downbeat).

    ``cue_in_is_section_start`` è la garanzia del chiamante che ``cue_in_b`` sia
    l'inizio di una sezione (hook/ritornello); qui non viene verificata
    acusticamente, solo registrata. Il render è ``dj_render`` ``kind="cut"``.
    """
    cfg = config or DJEffectsConfig()
    if not cue_in_is_section_start:
        raise DJRenderError("drop_in: cue_in_b must be the first downbeat of a section (hook/chorus)")
    base_cfg = DJRenderConfig(
        target_lufs=cfg.target_lufs, ceiling_dbtp=cfg.ceiling_dbtp, crossfade_ms=cfg.crossfade_ms,
        snap_tolerance_s=cfg.snap_tolerance_s, pre_s=cfg.pre_s, post_s=cfg.post_s, limiter_hold_ms=cfg.limiter_hold_ms,
        max_bar_deviation=cfg.max_bar_deviation, section_gain=cfg.section_gain,
    )
    res = dj.render_transition(
        audio_a, audio_b, beats_a=beats_a, downbeats_a=downbeats_a, beats_b=beats_b, downbeats_b=downbeats_b,
        cue_out_a=cue_out_a, cue_in_b=cue_in_b, kind="cut", sample_rate=sample_rate, config=base_cfg,
        provenance=provenance, return_decks=return_decks,
    )
    sr = res.sample_rate
    meta = dict(res.metadata)
    pre = int(round(cfg.pre_s * sr))
    out = res.audio.astype(np.float64)
    w = int(round(cfg.loudness_window_s * sr))
    loud = dict(meta["loudness"])
    # Same windows as the other effects (on the limited clip; dj_render's 4 s fields are kept as well).
    before = integrated_lufs(out[max(0, pre - w) : pre], sr)
    after = integrated_lufs(out[pre : pre + w], sr)
    loud["lufs_before_effect"] = before
    loud["lufs_after_b_entry"] = after
    loud["loudness_window_s"] = cfg.loudness_window_s
    loud["loudness_jump_db"] = (after - before) if np.isfinite(before) and np.isfinite(after) else None
    loud["loudness_jump_scope"] = "limited clip, window before/after the switch"
    meta.update({
        "schema_version": DJ_EFFECTS_VERSION,
        "renderer_base": meta.get("schema_version", dj.DJ_RENDER_VERSION),
        "kind": "drop_in",
        "base_kind": "cut",
        "loudness": loud,
        "effect": {
            "type": "drop_in",
            "cue_in_is_section_start": "asserted by caller (not verified acoustically here)",
            "section_label": section_label,
            "riser": None,
            "note": "B's first hook/chorus downbeat replaces A on A's exit downbeat; micro-crossfade ending on the downbeat",
            "basis": "[H] hip-hop DJ practice (drop on the hook); not measured on real mixes",
        },
        "config": asdict(cfg),
    })
    meta["clip"] = dict(meta["clip"], rule="drop_in: 10 s before + 10 s after the switch downbeat (cut, AGENTS.md)", effect_start_sample=pre, b_entry_sample=pre)
    res.metadata = _jsonable(meta)
    return res


def render_effect(kind: str, audio_a: np.ndarray, audio_b: np.ndarray, **kwargs) -> DJRenderResult:
    """Dispatcher: ``kind`` in ``EFFECT_KINDS``; kwargs come nelle singole funzioni."""
    if kind not in EFFECT_KINDS:
        raise DJRenderError(f"kind must be one of {EFFECT_KINDS}")
    return {"echo_out": echo_out, "filter_out": filter_out, "drop_in": drop_in}[kind](audio_a, audio_b, **kwargs)
