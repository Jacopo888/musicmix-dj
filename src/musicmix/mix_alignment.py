"""Allineamento mix → brani e lettura delle scelte del DJ (``musicmix.reference-mix.1``).

Scopo: dato un mix registrato e la lista dei brani (presenti in libreria), ricavare per
ogni brano dove compare nel mix e con quale corrispondenza temporale, poi per ogni
coppia consecutiva i punti d'ingresso/uscita (secondi, battute, frase, sezione), la
sovrapposizione, le curve di guadagno per banda e la tecnica probabile. Serve a imparare
una *distribuzione* di scelte da set di riferimento, non regole assolute.

Metodo (fonti in ``conoscenza_mix/_meta/SET_DI_RIFERIMENTO_2026-10-21.md``):

1. **Allineamento** — reimplementazione del metodo di Kim et al. 2020 (ISMIR, "A
   computational analysis of real-world DJ mixes using mix-to-track subsequence
   alignment"; codice ``mir-aidj/djmix-dataset`` commit e32d179, *senza licenza
   dichiarata*: il metodo è riscritto, non copiato). Feature a mezzo battito
   (cromagramma + MFCC, standardizzate con media e deviazione del brano), distanza
   coseno, DTW di sottosequenza (``librosa.sequence.dtw(subseq=True)``), tratti
   diagonali del percorso (pendenza 1 in mezzi battiti = stretch assorbito dalla
   sincronizzazione ai battiti), linea principale = tratto più lungo, *match rate*
   come qualità (Raveform scarta < 0,1). Aggiunte nostre [H]: MFCC 0 escluso
   (il fader cambia il livello), multipli di tempo 1 / 2 / ½ del brano (ottava del
   beat tracker), trasposizioni del cromagramma facoltative (Kim: 12).
2. **Curve di guadagno** — come Kim, Yang, Nam (DAFx 2022) il mix è modellato come
   somma non negativa degli spettri (mel, magnitudine) dei brani riportati sul tempo
   del mix; invece dell'ottimizzazione convessa con vincoli di monotonia (cvxpy, non
   installato) si risolve un NNLS per frame e per banda (bassi < 200 Hz, medi
   200–2500 Hz, alti ≥ 2500 Hz) su una finestra di ±2 frame [H]. Le curve sono in dB
   rispetto al livello del brano nel suo tratto allineato.
3. **Transizioni e tecnica** — udibilità (soglia in dB [H]), switch (dominanza),
   sovrapposizione, e un classificatore a regole [H] fra ``blend``, ``echo``,
   ``loop``, ``build``, ``filter`` e ``cut`` da indizi misurati (ripetizione del
   materiale di A dopo la fine del tratto diagonale, decadimento, calo relativo dei
   bassi o degli alti, residuo non spiegato in crescita). Le soglie sono ipotesi
   dichiarate; la validazione sui nostri set è in ``tools/validate_reference_mix.py``.

Il modulo non decide nulla per il pianificatore: produce osservazioni. Il passo verso
i prior è in :mod:`musicmix.reference_priors`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

SCHEMA = "musicmix.reference-mix.1"
METHOD_VERSION = "musicmix.mix_alignment.1"


# ---------------------------------------------------------------------- config


@dataclass
class AlignConfig:
    sr: int = 22050
    n_fft: int = 4096
    hop: int = 1024
    n_mels: int = 96
    n_mfcc: int = 20
    tempo_multiples: tuple = (1.0, 2.0, 0.5)
    key_shifts: tuple = (0,)
    forgivable: int = 2  # Kim 2020: punti non diagonali perdonati (1 battito)
    min_diag_beats: float = 8.0  # Kim usa 16; i brani rap nei set durano poco [H]
    min_match_rate: float = 0.1  # Raveform: soglia di scarto
    max_extension_beats: float = 32.0  # estensione della linea oltre il tratto diagonale [H]
    trim_cost_margin: float = 0.2  # [H] tratto diagonale tenuto dove il costo locale <= 20° percentile + 0,2
    trim_cost_min: float = 0.25
    trim_cost_max: float = 0.5
    step_penalty: float = 0.25  # [H] costo aggiunto ai passi non diagonali del DTW (Kim: 0)
    max_found_cost: float = 0.45  # [H] costo medio sul tratto: i veri allineamenti stanno sotto 0,2, i falsi sopra 0,7 (v3)
    band_edges_hz: tuple = (200.0, 2500.0)
    nnls_halfwidth: int = 1
    audible_db: float = -12.0  # [H] udibile se il guadagno supera -12 dB del livello di riferimento
    gap_bridge_beats: float = 1.0
    min_run_beats: float = 1.0  # [H] tratti udibili più corti di 1 battito ignorati (fughe dell'NNLS)
    audible_low_db: float = -22.0  # [H] soglia bassa dell'isteresi
    blend_min_beats: float = 2.0  # [H] sovrapposizione udibile >= 2 battiti = blend
    repeat_min: float = 0.12  # [H] indice di ripetizione del materiale di A
    echo_decay_db_per_beat: float = 3.0  # [H]
    filter_drop_db: float = 6.0  # [H]
    echo_max_level_db: float = -12.0  # [H] la coda d'eco sta sotto il livello di A prima dell'effetto
    curve_pad_beats: float = 8.0
    max_gap_beats: float = 64.0  # [H] oltre, con residuo alto, la coppia non è adiacente (brano mancante)
    missing_residual: float = 0.8  # [H] residuo relativo medio nella zona (v3: zone d'effetto 0,3-0,8; brani mancanti v1b/v2: 0,86-0,97)

    def to_json(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}


# ---------------------------------------------------------------------- audio & beats


def load_mono(path: str | Path, sr: int) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=True, res_type="soxr_hq")
    return np.ascontiguousarray(y, dtype=np.float32)


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def librosa_beats(y: np.ndarray, sr: int) -> np.ndarray:
    """Ripiego senza modelli (test): beat tracker di librosa."""
    import librosa

    _, frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512, units="frames")
    return librosa.frames_to_time(frames, sr=sr, hop_length=512).astype(float)


def default_tracker() -> Callable[[np.ndarray, int], np.ndarray]:
    """Beat This! (stesso wrapper e checkpoint del QA, ``dj_qa.BeatThisTracker``); ripiego librosa."""
    try:
        from .dj_qa import BeatThisTracker

        bt = BeatThisTracker()

        def _track(y: np.ndarray, sr: int) -> np.ndarray:
            return np.asarray(bt(y, sr)["beats"], float)

        _track.info = bt.info()  # type: ignore[attr-defined]
        return _track
    except Exception as exc:  # pragma: no cover - runtime opzionale
        def _fallback(y: np.ndarray, sr: int) -> np.ndarray:
            return librosa_beats(y, sr)

        _fallback.info = {"model": "librosa.beat.beat_track (ripiego)", "reason": repr(exc)}  # type: ignore[attr-defined]
        return _fallback


def halfbeats(beats: np.ndarray) -> np.ndarray:
    beats = np.asarray(beats, float)
    if len(beats) < 2:
        return beats.copy()
    out = np.empty(2 * len(beats) - 1)
    out[0::2] = beats
    out[1::2] = beats[:-1] + np.diff(beats) / 2.0
    return out


def scale_beats(beats: np.ndarray, multiple: float) -> np.ndarray:
    beats = np.asarray(beats, float)
    if multiple == 1.0 or len(beats) < 2:
        return beats
    if multiple == 2.0:
        return halfbeats(beats)
    if multiple == 0.5:
        return beats[::2]
    raise ValueError("tempo multiple must be 1, 2 or 0.5")


# ---------------------------------------------------------------------- features


@dataclass
class Features:
    sr: int
    hop: int
    duration_s: float
    frame_times: np.ndarray
    mel: np.ndarray  # (n_mels, T) magnitudine
    chroma: np.ndarray  # (12, T)
    mfcc: np.ndarray  # (n_mfcc, T)
    rms_db: np.ndarray  # (T,)
    beats: np.ndarray
    mel_freqs: np.ndarray

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in asdict(self).items()})

    @classmethod
    def load(cls, path: str | Path) -> "Features":
        z = np.load(path)
        kw = {k: z[k] for k in z.files}
        kw["sr"] = int(kw["sr"])
        kw["hop"] = int(kw["hop"])
        kw["duration_s"] = float(kw["duration_s"])
        return cls(**kw)


def compute_features(y: np.ndarray, beats: np.ndarray, cfg: AlignConfig) -> Features:
    import librosa

    S = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop)).astype(np.float32)
    P = S * S
    mel_basis = librosa.filters.mel(sr=cfg.sr, n_fft=cfg.n_fft, n_mels=cfg.n_mels, fmin=20.0, fmax=cfg.sr / 2.0)
    mel = (mel_basis @ S).astype(np.float32)
    mel_pow = mel_basis @ P
    chroma = librosa.feature.chroma_stft(S=P, sr=cfg.sr, n_chroma=12, tuning=0.0).astype(np.float32)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_pow), n_mfcc=cfg.n_mfcc).astype(np.float32)
    rms = np.sqrt(np.maximum(P.sum(axis=0) / (cfg.n_fft / 2.0) ** 2, 1e-20))
    del S, P, mel_pow
    T = mel.shape[1]
    return Features(
        sr=cfg.sr,
        hop=cfg.hop,
        duration_s=float(len(y) / cfg.sr),
        frame_times=librosa.frames_to_time(np.arange(T), sr=cfg.sr, hop_length=cfg.hop).astype(float),
        mel=mel,
        chroma=chroma,
        mfcc=mfcc,
        rms_db=(20 * np.log10(rms)).astype(np.float32),
        beats=np.asarray(beats, float),
        mel_freqs=librosa.mel_frequencies(n_mels=cfg.n_mels + 2, fmin=20.0, fmax=cfg.sr / 2.0)[1:-1],
    )


def grid_features(feat: Features, grid: np.ndarray) -> np.ndarray:
    """Media delle feature (croma 12 + MFCC 1..n) dentro ogni intervallo della griglia (mezzi battiti)."""
    X = np.vstack([feat.chroma, feat.mfcc[1:]])  # MFCC 0 escluso: livello/fader [H]
    grid = np.asarray(grid, float)
    n = len(grid) - 1
    if n < 1:
        return np.zeros((X.shape[0], 0), np.float32)
    idx = np.searchsorted(grid, feat.frame_times, side="right") - 1
    ok = (idx >= 0) & (idx < n)
    sums = np.zeros((X.shape[0], n))
    cnt = np.zeros(n)
    np.add.at(sums.T, idx[ok], X[:, ok].T)
    np.add.at(cnt, idx[ok], 1.0)
    empty = cnt == 0
    if np.any(empty):  # intervallo più corto di un frame: frame più vicino al centro
        centers = (grid[:-1] + grid[1:]) / 2.0
        near = np.clip(np.searchsorted(feat.frame_times, centers[empty]), 0, X.shape[1] - 1)
        sums[:, empty] = X[:, near]
        cnt[empty] = 1.0
    return (sums / cnt).astype(np.float32)


# ---------------------------------------------------------------------- alignment


@dataclass
class TrackAlignment:
    slug: str
    found: bool
    tempo_multiple: float = 1.0
    key_shift: int = 0
    match_rate: float = 0.0
    matched_beats: float = 0.0
    matched_track_s: float = 0.0
    cost: float = float("nan")
    main_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), int))  # (i_trk, j_mix)
    extra_segments: list = field(default_factory=list)
    trk_grid: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mix_grid: np.ndarray = field(default_factory=lambda: np.zeros(0))
    map_mix_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    map_trk_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    diag_mix_range: tuple = (float("nan"), float("nan"))
    candidates: list = field(default_factory=list)

    def to_track(self, t: np.ndarray | float) -> np.ndarray:
        t = np.asarray(t, float)
        if len(self.map_mix_t) < 2:
            return np.full(t.shape, np.nan)
        out = np.interp(t, self.map_mix_t, self.map_trk_t)
        out = np.where((t < self.map_mix_t[0]) | (t > self.map_mix_t[-1]), np.nan, out)
        return out

    def presence(self) -> tuple:
        if len(self.map_mix_t) < 2:
            return (float("nan"), float("nan"))
        return (float(self.map_mix_t[0]), float(self.map_mix_t[-1]))


def _diag_groups(path: np.ndarray, forgivable: int) -> list[np.ndarray]:
    """Gruppi di punti del percorso (ordine in avanti) su tratti diagonali, con buchi perdonati."""
    if len(path) < 2:
        return []
    d = np.diff(path, axis=0)
    diag_step = (d[:, 0] == 1) & (d[:, 1] == 1)
    on = np.zeros(len(path), bool)
    on[:-1] |= diag_step
    on[1:] |= diag_step
    idx = np.flatnonzero(on)
    if len(idx) == 0:
        return []
    groups, cur = [], [idx[0]]
    for a, b in zip(idx[:-1], idx[1:]):
        pa, pb = path[a], path[b]
        jump_i, jump_j = pb[0] - pa[0], pb[1] - pa[1]
        same_line = abs((pb[1] - pb[0]) - (pa[1] - pa[0])) <= forgivable
        if b - a <= forgivable + 1 and same_line and jump_i >= 0 and jump_j >= 0:
            cur.append(b)
        else:
            groups.append(np.asarray(cur))
            cur = [b]
    groups.append(np.asarray(cur))
    out = []
    for g in groups:
        pts = path[g]
        # niente coordinate duplicate (Kim: correct_wp)
        _, ui = np.unique(pts[:, 0], return_index=True)
        pts = pts[np.sort(ui)]
        _, uj = np.unique(pts[:, 1], return_index=True)
        pts = pts[np.sort(uj)]
        out.append(pts)
    return out


def _trim_by_cost(pts: np.ndarray, C: np.ndarray, cfg: AlignConfig) -> np.ndarray:
    """Toglie dalle estremità i punti diagonali con costo locale alto: il DTW può proseguire in
    diagonale anche dove il brano non suona (aggiunta nostra [H] al metodo di Kim)."""
    pts = pts[(pts[:, 0] < C.shape[0]) & (pts[:, 1] < C.shape[1])]
    if len(pts) < 8:
        return pts
    c = C[pts[:, 0], pts[:, 1]]
    k = 4
    sm = np.convolve(c, np.ones(k) / k, mode="same")
    thr = float(np.clip(np.percentile(c, 20) + cfg.trim_cost_margin, cfg.trim_cost_min, cfg.trim_cost_max))
    good = sm <= thr
    good = _bridge(good, 4)  # buchi di costo alto fino a 2 battiti perdonati
    runs = _runs(good)
    if not runs:
        return pts[:0]
    s0, e0 = max(runs, key=lambda r: r[1] - r[0])  # tratto buono più lungo
    return pts[s0:e0]


def _evaluate_candidate(C: np.ndarray, cfg: AlignConfig, n_trk_hb: int):
    import librosa

    D, wp = librosa.sequence.dtw(C=C, subseq=True, backtrack=True,
                                 weights_add=np.array([0.0, cfg.step_penalty, cfg.step_penalty]))
    path = np.asarray(wp[::-1], int)
    groups = _diag_groups(path, cfg.forgivable)
    min_len = int(round(cfg.min_diag_beats * 2))
    long_groups = [g for g in groups if len(g) >= min_len]
    if not long_groups:
        return None
    lengths = [g[-1, 0] - g[0, 0] + 1 for g in long_groups]
    intercepts = [float(np.median(g[:, 1] - g[:, 0])) for g in long_groups]
    k = int(np.argmax(lengths))
    b0 = intercepts[k]
    main = [g for g, b in zip(long_groups, intercepts) if abs(b - b0) <= cfg.forgivable]
    extra = [g for g, b in zip(long_groups, intercepts) if abs(b - b0) > cfg.forgivable]
    pts = np.concatenate(main)
    pts = pts[np.argsort(pts[:, 1])]
    pts = _trim_by_cost(pts, C, cfg)
    if len(pts) < min_len:
        return None
    # monotonia in entrambe le coordinate
    keep = np.r_[True, (np.diff(pts[:, 0]) > 0) & (np.diff(pts[:, 1]) > 0)]
    pts = pts[keep]
    span = int(pts[-1, 0] - pts[0, 0] + 1)
    cost = float(np.mean(C[pts[:, 0], pts[:, 1]]))
    return {"points": pts, "extra": extra, "span_hb": span, "match_rate": span / max(n_trk_hb, 1), "cost": cost}


def align_track(slug: str, mix: Features, trk: Features, cfg: AlignConfig) -> TrackAlignment:
    """DTW di sottosequenza brano→mix su mezzi battiti. I multipli di tempo correggono l'ottava
    del beat tracker affinando sempre la griglia più grossa (×2 sul brano o sul mix), mai
    diradando: la mappa temporale resta fine anche quando il mix è tracciato a metà tempo."""
    from scipy.spatial.distance import cdist

    mix_grids = {1: halfbeats(mix.beats)}
    if any(float(m) < 1.0 for m in cfg.tempo_multiples):
        mix_grids[2] = halfbeats(scale_beats(mix.beats, 2.0))
    mix_feats = {k: grid_features(mix, g) for k, g in mix_grids.items()}
    best, cands = None, []
    for mult in cfg.tempo_multiples:
        mult = float(mult)
        trk_mult, mix_mult = (2, 1) if mult == 2.0 else ((1, 2) if mult == 0.5 else (1, 1))
        tb = scale_beats(trk.beats, float(trk_mult))
        if len(tb) < 8:
            continue
        trk_grid = halfbeats(tb)
        mix_grid = mix_grids[mix_mult]
        Xt = grid_features(trk, trk_grid)
        mu = Xt.mean(axis=1, keepdims=True)
        sd = Xt.std(axis=1, keepdims=True) + 1e-6
        Xt_s = (Xt - mu) / sd
        Xm_s = (mix_feats[mix_mult] - mu) / sd
        for shift in cfg.key_shifts:
            X = Xt_s
            if shift:
                X = Xt_s.copy()
                X[:12] = np.roll(X[:12], int(shift), axis=0)
            C = cdist(X.T, Xm_s.T, metric="cosine").astype(np.float64)
            C = np.nan_to_num(C, nan=1.0, posinf=2.0, neginf=2.0)
            res = _evaluate_candidate(C, cfg, Xt.shape[1])
            del C
            if res is None:
                cands.append({"tempo_multiple": mult, "key_shift": int(shift), "match_rate": 0.0})
                continue
            matched_s = float(trk_grid[min(res["points"][-1, 0] + 1, len(trk_grid) - 1)] - trk_grid[res["points"][0, 0]])
            res.update({"tempo_multiple": mult, "trk_mult": trk_mult, "key_shift": int(shift), "trk_grid": trk_grid,
                        "mix_grid": mix_grid, "matched_s": matched_s})
            cands.append({"tempo_multiple": mult, "key_shift": int(shift), "match_rate": round(res["match_rate"], 4),
                          "matched_track_s": round(matched_s, 2), "cost": round(res["cost"], 4)})
            ok = res["cost"] <= cfg.max_found_cost
            score = (ok, matched_s, -res["cost"])
            if best is None or score > (best["cost"] <= cfg.max_found_cost, best["matched_s"], -best["cost"]):
                best = res
    out = TrackAlignment(slug=slug, found=False, mix_grid=mix_grids[1], candidates=cands)
    if best is None:
        return out
    out.tempo_multiple = float(best["tempo_multiple"])
    out.key_shift = int(best["key_shift"])
    out.match_rate = float(best["match_rate"])
    out.cost = float(best["cost"])
    out.trk_grid = best["trk_grid"]
    out.mix_grid = best["mix_grid"]
    out.main_points = best["points"]
    out.matched_track_s = float(best["matched_s"])
    out.matched_beats = float(best["span_hb"]) / 2.0 / best["trk_mult"]
    mg = best["mix_grid"]
    out.extra_segments = [
        {"mix_s": [float(mg[g[0, 1]]), float(mg[min(g[-1, 1], len(mg) - 1)])],
         "track_s": [float(best["trk_grid"][g[0, 0]]), float(best["trk_grid"][min(g[-1, 0], len(best["trk_grid"]) - 1)])],
         "halfbeats": int(len(g))}
        for g in best["extra"]
    ]
    out.found = (out.match_rate >= cfg.min_match_rate or out.matched_track_s >= 20.0) and out.cost <= cfg.max_found_cost
    _build_time_map(out, cfg)
    return out


def _build_time_map(al: TrackAlignment, cfg: AlignConfig) -> None:
    """Ancore (tempo mix, tempo brano) sui punti diagonali + estensione sulla stessa linea [Kim: extend_wp]."""
    pts = al.main_points
    mg, tg = al.mix_grid, al.trk_grid
    if len(pts) < 2:
        return
    ext = int(round(cfg.max_extension_beats * 2))
    j0, i0 = int(pts[0, 1]), int(pts[0, 0])
    j1, i1 = int(pts[-1, 1]), int(pts[-1, 0])
    pre = [(j, i0 - (j0 - j)) for j in range(max(0, j0 - ext), j0) if i0 - (j0 - j) >= 0]
    post = [(j, i1 + (j - j1)) for j in range(j1 + 1, min(len(mg), j1 + ext + 1)) if i1 + (j - j1) < len(tg)]
    js = [p[0] for p in pre] + list(pts[:, 1]) + [p[0] for p in post]
    is_ = [p[1] for p in pre] + list(pts[:, 0]) + [p[1] for p in post]
    js, is_ = np.asarray(js), np.asarray(is_)
    ok = (js < len(mg)) & (is_ < len(tg))
    mt, tt = mg[js[ok]], tg[is_[ok]]
    keep = np.r_[True, (np.diff(mt) > 0) & (np.diff(tt) > 0)]
    al.map_mix_t, al.map_trk_t = mt[keep], tt[keep]
    al.diag_mix_range = (float(mg[j0]), float(mg[min(j1, len(mg) - 1)]))


# ---------------------------------------------------------------------- gains


def band_masks(feat: Features, cfg: AlignConfig) -> dict:
    f = feat.mel_freqs
    lo, hi = cfg.band_edges_hz
    return {"full": np.ones_like(f, bool), "low": f < lo, "mid": (f >= lo) & (f < hi), "high": f >= hi}


def _track_mel_at(trk: Features, times: np.ndarray) -> np.ndarray:
    """Spettro mel del brano ai tempi (s) dati, interpolato tra frame; NaN fuori."""
    pos = np.asarray(times, float) / (trk.hop / trk.sr)
    T = trk.mel.shape[1]
    out = np.full((trk.mel.shape[0], len(pos)), np.nan, np.float32)
    ok = np.isfinite(pos) & (pos >= 0) & (pos <= T - 1)
    p = pos[ok]
    i = np.floor(p).astype(int)
    i2 = np.minimum(i + 1, T - 1)
    w = (p - i).astype(np.float32)
    out[:, ok] = trk.mel[:, i] * (1 - w) + trk.mel[:, i2] * w
    return out


def estimate_gains(mix: Features, tracks: dict, aligns: dict, cfg: AlignConfig) -> dict:
    """NNLS per frame e banda: mix ≈ Σ g_k · brano_k(t). Restituisce guadagni lineari e residuo."""
    from scipy.optimize import nnls

    slugs = [s for s in aligns if aligns[s].found]
    T = mix.mel.shape[1]
    t = mix.frame_times
    spec = {}
    for s in slugs:
        spec[s] = _track_mel_at(tracks[s], aligns[s].to_track(t))
    masks = band_masks(mix, cfg)
    gains = {s: {b: np.full(T, np.nan, np.float32) for b in masks} for s in slugs}
    resid = np.full(T, np.nan, np.float32)
    hw = cfg.nnls_halfwidth
    present = np.array([[np.isfinite(spec[s][0, f]) for s in slugs] for f in range(T)]) if slugs else np.zeros((T, 0), bool)
    for f in range(T):
        ks = [k for k in range(len(slugs)) if present[f, k]]
        lo, hi = max(0, f - hw), min(T, f + hw + 1)
        for bname, m in masks.items():
            y = mix.mel[m, lo:hi].reshape(-1)
            if not ks:
                if bname == "full":
                    resid[f] = 1.0
                continue
            A = np.stack([np.nan_to_num(spec[slugs[k]][m, lo:hi]).reshape(-1) for k in ks], axis=1)
            if not np.any(A):
                continue
            g, rn = nnls(A.astype(float), y.astype(float), maxiter=200)
            for k, gk in zip(ks, g):
                gains[slugs[k]][bname][f] = gk
            if bname == "full":
                ny = float(np.linalg.norm(y))
                resid[f] = rn / ny if ny > 0 else 0.0
    # riferimento: mediana del guadagno nel tratto diagonale del brano
    ref = {}
    for s in slugs:
        a, b = aligns[s].diag_mix_range
        sel = (t >= a) & (t <= b)
        vals = gains[s]["full"][sel]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        ref[s] = float(np.median(vals)) if len(vals) else 1.0
    gains_db = {}
    for s in slugs:
        gains_db[s] = {}
        for bname in masks:
            # riferimento per banda: mediana della banda sul tratto diagonale
            a, b = aligns[s].diag_mix_range
            sel = (t >= a) & (t <= b)
            vals = gains[s][bname][sel]
            vals = vals[np.isfinite(vals) & (vals > 0)]
            r = float(np.median(vals)) if len(vals) else ref[s]
            g = gains[s][bname]
            gains_db[s][bname] = np.where(np.isfinite(g), 20 * np.log10(np.maximum(g, 1e-6) / max(r, 1e-9)), np.nan).astype(np.float32)
    return {"slugs": slugs, "gain": gains, "gain_db": gains_db, "ref": ref, "residual": resid}


# ---------------------------------------------------------------------- transitions


def _median_filter_nan(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x.astype(float), pad, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(xp, k)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(win, axis=1)


def local_beat_period(beats: np.ndarray, t: float) -> float:
    beats = np.asarray(beats, float)
    if len(beats) < 3:
        return 0.5
    mids = (beats[:-1] + beats[1:]) / 2
    per = np.diff(beats)
    # mediana locale robusta su 8 battiti
    i = int(np.clip(np.searchsorted(mids, t), 0, len(per) - 1))
    lo, hi = max(0, i - 4), min(len(per), i + 5)
    return float(np.median(per[lo:hi]))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.r_[False, mask.astype(bool), False]
    d = np.diff(m.astype(int))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return list(zip(starts, ends))  # [start, end)


def _bridge(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    runs = _runs(mask)
    for (s0, e0), (s1, _e1) in zip(runs[:-1], runs[1:]):
        if s1 - e0 <= max_gap:
            out[e0:s1] = True
    return out


def _presence_db(g: dict) -> np.ndarray:
    """Presenza del brano = massimo fra banda intera, medi e alti (dB): un passa-alto toglie i bassi
    (gran parte dell'energia) ma il brano resta udibile [H]."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(np.vstack([g["full"], g["mid"], g["high"]]), axis=0)


def _run_end_near(aud: np.ndarray, seed: int, lo: int) -> int:
    """Fine del tratto udibile che contiene ``seed``; altrimenti dell'ultimo che finisce prima (non prima di ``lo``)."""
    best = None
    for s0, e0 in _runs(aud):
        if s0 <= seed < e0:
            return e0 - 1
        if e0 - 1 < seed and e0 - 1 >= lo:
            best = e0 - 1
    return best if best is not None else seed


def _run_start_near(aud: np.ndarray, seed: int, hi: int) -> int:
    """Inizio del tratto udibile che contiene ``seed``; altrimenti del primo che inizia dopo (entro ``hi``)."""
    for s0, e0 in _runs(aud):
        if s0 <= seed < e0:
            return s0
        if seed < s0 <= hi:
            return s0
    return seed


def _audible(g_db: np.ndarray, cfg: AlignConfig, beat_frames: int) -> np.ndarray:
    """Udibilità con isteresi [H]: tratti sopra ``audible_low_db`` che contengono almeno
    ``min_run_beats`` sopra ``audible_db`` (le dissolvenze vengono seguite fino al fondo)."""
    g = np.nan_to_num(g_db, nan=-120)
    high = g > cfg.audible_db
    min_len = max(1, int(round(cfg.min_run_beats * beat_frames)))
    for s0, e0 in _runs(high):
        if e0 - s0 < min_len:
            high[s0:e0] = False
    high = _bridge(high, beat_frames)
    low = g > cfg.audible_low_db
    out = np.zeros_like(high)
    for s0, e0 in _runs(low | high):
        if np.any(high[s0:e0]):
            out[s0:e0] = True
    return out


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def _repeat_evidence(mix: Features, trk: Features, al: TrackAlignment, t_from: float, t_to: float) -> dict:
    """Il mix, dopo la fine del tratto diagonale di A e prima che entri B, somiglia più a materiale
    già suonato di A (eco/loop: ultime 4 battute) che al materiale successivo (stessa ampiezza)?
    Indice = media(max sim passato − max sim futuro)."""
    if not np.isfinite(t_from) or t_to - t_from < 0.2 or len(al.main_points) < 4:
        return {"repeat_index": None}
    mg = al.mix_grid
    tg = al.trk_grid
    Xm = grid_features(mix, mg)
    Xt = grid_features(trk, tg)
    mu = Xt.mean(axis=1, keepdims=True)
    sd = Xt.std(axis=1, keepdims=True) + 1e-6
    Xm = (Xm - mu) / sd
    Xt = (Xt - mu) / sd
    i_end = int(al.main_points[-1, 0])
    j_end = int(al.main_points[-1, 1])
    js = [j for j in range(j_end + 1, min(len(mg) - 1, j_end + 33)) if t_from <= mg[j] < t_to]
    if len(js) < 2:
        return {"repeat_index": None}
    past = list(range(max(0, i_end - 32), i_end + 1))  # ultime 4 battute (16 battiti) di A
    diffs, rep_sims, cont_sims = [], [], []
    future = list(range(i_end + 1, min(Xt.shape[1], i_end + 34)))  # stessa ampiezza, dopo: confronto equo
    for j in js:
        rep = max(_cos(Xm[:, j], Xt[:, i]) for i in past)
        cont = max((_cos(Xm[:, j], Xt[:, i]) for i in future), default=0.0)
        diffs.append(rep - cont)
        rep_sims.append(rep)
        cont_sims.append(cont)
    return {"repeat_index": float(np.mean(diffs)), "repeat_sim": float(np.mean(rep_sims)),
            "continuation_sim": float(np.mean(cont_sims)), "n_halfbeats": len(js)}


def _mean_db(x: np.ndarray, sel: np.ndarray) -> float:
    v = x[sel]
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if len(v) else float("nan")


def classify_technique(ev: dict, cfg: AlignConfig) -> tuple[str, list[str]]:
    """Regole [H] dichiarate, sugli indizi misurati in ``ev``.

    * ``blend``: B udibile mentre A suona ancora pulito (prima della fine della sua diagonale) e
      sovrapposizione udibile >= ``blend_min_beats``;
    * zona d'effetto (fra fine della diagonale di A e ingresso di B) < 1 battito → ``cut``;
    * nella zona d'effetto: il mix decade (>= ``echo_decay_db_per_beat``) e la presenza di A come
      livello scende sotto ``echo_max_level_db`` → ``echo``; A resta presente con i bassi (o gli
      alti) calati → ``filter`` (filtro o build: il rullo/riser non è riconosciuto); A presente
      senza filtro → ``loop`` (materiale di A ripetuto o fermato: lo spettro somiglia ad A);
      niente suono → ``cut`` con pausa."""
    why = []
    ov = ev.get("overlap_beats")
    pre = ev.get("b_before_a_diag_end_beats")
    eff = ev.get("effect_beats")
    decay = ev.get("effect_mix_decay_db_per_beat")
    level = ev.get("effect_mix_level_db")
    pres = ev.get("effect_a_presence_db")
    low_drop = ev.get("a_low_drop_rel_db")
    high_drop = ev.get("a_high_drop_rel_db")
    res_rise = ev.get("residual_rise_db")
    if ov is not None and ov >= cfg.blend_min_beats and pre is not None and pre >= 1.0:
        why.append(f"B udibile {pre:.1f} battiti prima che A smetta di suonare pulito; sovrapposizione {ov:.1f} battiti")
        return "blend", why
    if eff is None or eff < 1.0 or level is None:
        why.append("nessuna zona d'effetto fra A pulito e B (< 1 battito)")
        return "cut", why
    if level < -30:
        why.append(f"zona di {eff:.1f} battiti quasi muta ({level:.1f} dB): pausa prima di B")
        return "cut", why
    filt = None
    if low_drop is not None and low_drop >= cfg.filter_drop_db:
        filt = f"bassi di A giù di {low_drop:.1f} dB rispetto a medi/alti (passa-alto)"
    elif high_drop is not None and high_drop >= cfg.filter_drop_db:
        filt = f"alti di A giù di {high_drop:.1f} dB rispetto a bassi/medi (passa-basso)"
    a_present = pres is not None and pres > cfg.audible_db
    if decay is not None and decay >= cfg.echo_decay_db_per_beat and level <= cfg.echo_max_level_db:
        why.append(f"zona di {eff:.1f} battiti: il mix cala ({level:.1f} dB) e decade di {decay:.1f} dB/battito (coda d'eco)")
        return "echo", why
    if filt:
        # sul v3 il residuo non spiegato cresce in tutte le zone d'effetto (eco, filtro, build):
        # filtro e build (rullo/riser sopra il filtro) non sono distinguibili con questi indizi
        why += [filt, "filtro o build: non distinti da questi indizi"]
        return "filter", why
    if a_present:
        why.append(f"zona di {eff:.1f} battiti con materiale simile ad A ma fuori diagonale (ripetuto/fermato)")
        return "loop", why
    why.append(f"zona di {eff:.1f} battiti non riconosciuta")
    return "cut", why


def track_position(analysis: Optional[dict], src_s: float, trk_beats: Optional[np.ndarray] = None) -> dict:
    """Posizione di un istante del brano: battuta, battito nella battuta, frase, sezione."""
    out: dict = {"source_s": None if src_s is None or not np.isfinite(src_s) else round(float(src_s), 3)}
    if out["source_s"] is None:
        return out
    if analysis and (analysis.get("grid") or {}).get("downbeats_s"):
        db = np.asarray(analysis["grid"]["downbeats_s"], float)
        if len(db) >= 2:
            bar_s = float(np.median(np.diff(db)))
            k = np.arange(len(db), dtype=float)
            if src_s < db[0]:
                bar_f = (src_s - db[0]) / bar_s
            elif src_s > db[-1]:
                bar_f = len(db) - 1 + (src_s - db[-1]) / bar_s
            else:
                bar_f = float(np.interp(src_s, db, k))
            near = int(round(bar_f))
            out.update({"bar": round(bar_f, 3), "nearest_downbeat_bar": near,
                        "beats_from_downbeat": round((bar_f - near) * 4.0, 2), "bar_s": round(bar_s, 4)})
            phrases = analysis.get("phrases") or []
            ph = next((p for p in phrases if p.get("bar_index") == near), None)
            if ph:
                out.update({"section": ph.get("section_label"), "bar_in_section": ph.get("bar_in_section"),
                            "phrase_level": ph.get("phrase_level"), "section_start": bool(ph.get("section_start")),
                            "bars_to_next_section": ph.get("bars_to_next_section")})
                bis = ph.get("bar_in_section")
                if isinstance(bis, int):
                    out["bar_in_phrase8"] = bis % 8
                    out["bar_in_phrase16"] = bis % 16
    elif trk_beats is not None and len(trk_beats) > 2:
        out["beat"] = round(float(np.interp(src_s, trk_beats, np.arange(len(trk_beats)))), 2)
    if analysis:
        secs = analysis.get("sections") or []
        cur = next((s for s in secs if s.get("start_s", 0) <= src_s < s.get("end_s", 0)), None)
        if cur:
            out.setdefault("section", cur.get("label"))
            out["section_label_time"] = cur.get("label")
            out["s_from_section_start"] = round(src_s - float(cur.get("start_s", 0)), 3)
        # contesto per "A carica, B risolve" (voti ciechi dell'utente, 21/10/2026): sezione precedente,
        # distanza dal prossimo ritornello e dall'inizio dell'ultimo ritornello [descrizione, non regola]
        bar_s_sec = out.get("bar_s")
        if secs:
            i_cur = next((i for i, x in enumerate(secs) if x is cur), None)
            if i_cur is not None and i_cur > 0:
                out["prev_section"] = secs[i_cur - 1].get("label")
            if cur and bar_s_sec:
                out["bars_to_section_end"] = round((float(cur.get("end_s", src_s)) - src_s) / bar_s_sec, 2)
            nxt = next((x for x in secs if x.get("label") == "chorus" and float(x.get("start_s", 0)) >= src_s - 0.25 * (bar_s_sec or 2.0)), None)
            if nxt is not None and bar_s_sec:
                out["bars_to_next_chorus"] = round((float(nxt["start_s"]) - src_s) / bar_s_sec, 2)
            last = [x for x in secs if x.get("label") == "chorus" and float(x.get("start_s", 0)) < src_s]
            if last and bar_s_sec:
                out["bars_since_chorus_start"] = round((src_s - float(last[-1]["start_s"])) / bar_s_sec, 2)
                out["in_or_after_chorus_end_bars"] = round((src_s - float(last[-1].get("end_s", src_s))) / bar_s_sec, 2)
        dur = analysis.get("duration_s")
        if dur:
            out["track_fraction"] = round(src_s / float(dur), 4)
    return out


def analyze_transitions(mix: Features, tracks: dict, aligns: dict, gains: dict, analyses: dict,
                        cfg: AlignConfig, order: Sequence[str]) -> list[dict]:
    t = mix.frame_times
    dt = mix.hop / mix.sr
    out = []
    smooth = {s: {b: _median_filter_nan(v, 3) for b, v in gains["gain_db"][s].items()} for s in gains["slugs"]}
    resid_db = 20 * np.log10(np.maximum(np.nan_to_num(gains["residual"], nan=1.0), 1e-4))
    for a, b in zip(order[:-1], order[1:]):
        ala, alb = aligns[a], aligns[b]
        ga, gb = _presence_db(smooth[a]), _presence_db(smooth[b])
        bp = local_beat_period(mix.beats, (ala.diag_mix_range[1] + alb.diag_mix_range[0]) / 2)
        beat_frames = max(1, int(round(bp / dt)))
        gap = max(1, int(round(cfg.gap_bridge_beats * bp / dt)))
        aud_a = _audible(ga, cfg, beat_frames)
        aud_b = _audible(gb, cfg, beat_frames)
        a_diag_end = int(np.clip(np.searchsorted(t, ala.diag_mix_range[1]), 0, len(t) - 1))
        b_diag_start = int(np.clip(np.searchsorted(t, alb.diag_mix_range[0]), 0, len(t) - 1))
        a_end_i = _run_end_near(aud_a, a_diag_end, int(np.searchsorted(t, ala.diag_mix_range[0])))
        b_start_i = _run_start_near(aud_b, b_diag_start, int(np.searchsorted(t, alb.diag_mix_range[1])))
        a_end, b_start = float(t[a_end_i]), float(t[b_start_i])
        # switch: prima dominanza di B nella zona
        # switch: primo frame da quando B è udibile in cui B domina A (A assente = -120 dB)
        lo = b_start_i
        hi = min(len(t), max(b_start_i, a_end_i) + gap + 1)
        gbs = np.nan_to_num(gb[lo:hi], nan=-120)
        dom = (gbs >= np.nan_to_num(ga[lo:hi], nan=-120)) & (gbs > cfg.audible_low_db)
        sw_i = lo + int(np.argmax(dom)) if np.any(dom) else b_start_i
        switch = float(t[sw_i])
        overlap_s = a_end - b_start
        ev: dict = {"overlap_s": round(overlap_s, 3), "overlap_beats": round(overlap_s / bp, 2), "beat_period_s": round(bp, 4)}
        # Zona d'effetto E = [fine del tratto diagonale di A, ingresso udibile di B]: sul v3 la fine
        # della diagonale cade sull'inizio dell'effetto (eco, filtro, build, loop) perché il
        # materiale smette di essere "A pulito"; nei blend B entra prima che A smetta.
        d_end = float(ala.diag_mix_range[1])
        ev["effect_start_s"] = round(d_end, 3)
        ev["effect_beats"] = round((b_start - d_end) / bp, 2)
        ev["b_before_a_diag_end_beats"] = round((d_end - b_start) / bp, 2)
        rep = _repeat_evidence(mix, tracks[a], ala, d_end, b_start + 0.25 * bp)
        ev.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep.items()})
        sel_e = (t > d_end) & (t < b_start)
        sel_before = (t > d_end - 8 * bp) & (t <= d_end)
        pa = _presence_db(smooth[a])
        if np.count_nonzero(sel_e) >= 3 and np.any(sel_before):
            y = mix.rms_db[sel_e]
            ev["effect_mix_decay_db_per_beat"] = round(-float(np.polyfit(t[sel_e] / bp, y, 1)[0]), 2)
            ev["effect_mix_level_db"] = round(float(np.mean(y) - np.mean(mix.rms_db[sel_before])), 2)
            ev["effect_a_presence_db"] = round(_mean_db(pa, sel_e), 2)
            sa = smooth[a]
            dl = _mean_db(sa["low"], sel_e) - _mean_db(sa["low"], sel_before)
            dm = _mean_db(sa["mid"], sel_e) - _mean_db(sa["mid"], sel_before)
            dh = _mean_db(sa["high"], sel_e) - _mean_db(sa["high"], sel_before)
            if all(np.isfinite([dl, dm, dh])):
                ev["a_low_drop_rel_db"] = round(-(dl - (dm + dh) / 2), 2)
                ev["a_high_drop_rel_db"] = round(-(dh - (dl + dm) / 2), 2)
            r1, r0 = _mean_db(resid_db, sel_e), _mean_db(resid_db, sel_before)
            if np.isfinite(r1) and np.isfinite(r0):
                ev["residual_rise_db"] = round(r1 - r0, 2)
        technique, why = classify_technique(ev, cfg)
        # brano mancante fra A e B? zona lunga non spiegata fra la fine di A e l'ingresso di B [H]
        # (fra le diagonali: le stime di guadagno possono "coprire" il buco con A o B estesi)
        gap_s = float(alb.diag_mix_range[0] - ala.diag_mix_range[1])
        sel_gap = (t > ala.diag_mix_range[1]) & (t < alb.diag_mix_range[0])
        gap_resid = _mean_db(np.nan_to_num(gains["residual"], nan=1.0).astype(float), sel_gap) if np.any(sel_gap) else 0.0
        ev["diag_gap_beats"] = round(gap_s / bp, 2)
        ev["diag_gap_residual"] = round(gap_resid, 3) if np.isfinite(gap_resid) else None
        adjacent = not (gap_s > cfg.max_gap_beats * bp and gap_resid > cfg.missing_residual)
        if not adjacent:
            technique, why = "unknown", [f"{gap_s / bp:.0f} battiti non spiegati fra A e B: probabile brano mancante in libreria"]
        # bass swap: nei blend, i bassi di A e B si incrociano in meno di 1 battito? [H]
        if technique == "blend":
            sel = (t >= b_start) & (t <= a_end)
            la, lb = smooth[a]["low"][sel], smooth[b]["low"][sel]
            both_low = np.nan_to_num(la, nan=-120) > -6
            both_low &= np.nan_to_num(lb, nan=-120) > -6
            ev["both_bass_full_s"] = round(float(np.count_nonzero(both_low) * dt), 3)
            ev["bass_swap_likely"] = bool(ev["both_bass_full_s"] < bp and overlap_s > 2 * bp)
        # curve per banda campionate a mezzo battito
        c0, c1 = min(b_start, switch) - cfg.curve_pad_beats * bp, max(a_end, switch) + cfg.curve_pad_beats * bp
        grid = np.arange(c0, c1 + 1e-9, bp / 2)
        curves = {"t_mix_s": [round(float(x), 3) for x in grid], "t_rel_switch_beats": [round(float((x - switch) / bp), 2) for x in grid]}
        for name, s in (("a", a), ("b", b)):
            curves[name] = {bn: [None if not np.isfinite(v) else round(float(v), 1) for v in np.interp(grid, t, np.nan_to_num(smooth[s][bn], nan=np.nan))]
                            for bn in ("full", "low", "mid", "high")}
        # posizioni nei brani
        a_src_end = float(ala.to_track(a_end))
        a_src_sw = float(ala.to_track(switch))
        b_src_start = float(alb.to_track(b_start))
        b_src_sw = float(alb.to_track(switch))
        trb_a = tracks[a].beats
        trb_b = tracks[b].beats
        rate_a = _local_rate(ala, "end")
        rate_b = _local_rate(alb, "start")
        out.append({
            "a": a, "b": b, "adjacent": adjacent, "technique": technique, "technique_why": why,
            "mix_s": {"b_audible_start": round(b_start, 3), "switch": round(switch, 3), "a_audible_end": round(a_end, 3),
                      "a_diag_end": round(ala.diag_mix_range[1], 3), "b_diag_start": round(alb.diag_mix_range[0], 3)},
            "overlap_s": round(overlap_s, 3), "overlap_beats": round(overlap_s / bp, 2),
            "a_exit": {"at_audible_end": track_position(analyses.get(a), a_src_end, trb_a),
                       "at_switch": track_position(analyses.get(a), a_src_sw, trb_a)},
            "b_entry": {"at_audible_start": track_position(analyses.get(b), b_src_start, trb_b),
                        "at_switch": track_position(analyses.get(b), b_src_sw, trb_b)},
            "tempo": {"a_rate": rate_a, "b_rate": rate_b,
                      "a_tempo_change_pct": None if rate_a is None else round((rate_a - 1) * 100, 2),
                      "b_tempo_change_pct": None if rate_b is None else round((rate_b - 1) * 100, 2)},
            "key_shift": {"a": ala.key_shift, "b": alb.key_shift},
            "evidence": ev,
            "curves_db": curves,
        })
    return out


def _local_rate(al: TrackAlignment, side: str, span_s: float = 16.0) -> Optional[float]:
    """Secondi di brano per secondo di mix (1 = velocità originale), stimati solo sul tratto
    diagonale (non sull'estensione, che segue la griglia del mix): primi ``span_s`` s per B
    (``side="start"``), ultimi per A (``side="end"``)."""
    pts = al.main_points
    if len(pts) < 8:
        return None
    mt = al.mix_grid[np.minimum(pts[:, 1], len(al.mix_grid) - 1)]
    tt = al.trk_grid[np.minimum(pts[:, 0], len(al.trk_grid) - 1)]
    sel = (mt <= mt[0] + span_s) if side == "start" else (mt >= mt[-1] - span_s)
    if np.count_nonzero(sel) < 8:
        return None
    p = np.polyfit(mt[sel], tt[sel], 1)
    return round(float(p[0]), 4)


# ---------------------------------------------------------------------- top level


@dataclass
class TrackSpec:
    slug: str
    audio: str
    analysis: Optional[dict] = None
    artist: Optional[str] = None
    title: Optional[str] = None
    beats: Optional[np.ndarray] = None


def grid_beats(analysis: Optional[dict]) -> Optional[np.ndarray]:
    """Battiti della scheda del brano (``musicmix.dj_select.1``: griglia a tempo costante
    ``phase + k·period``), così battute e allineamento usano la stessa griglia e non serve il
    beat tracker sui brani di libreria. None se la scheda non ha la griglia."""
    if not analysis:
        return None
    tempo = analysis.get("tempo") or {}
    ph, per, dur = tempo.get("phase_s"), tempo.get("period_s"), analysis.get("duration_s")
    if ph is None or not per or not dur:
        return None
    ph, per = float(ph), float(per)
    k0 = -int(np.floor(ph / per))
    b = ph + per * np.arange(k0, int((float(dur) - ph) / per) + 1)
    return b[(b >= 0) & (b <= float(dur))]


def features_cached(path: str | Path, cfg: AlignConfig, tracker, cache_dir: Optional[Path], beats: Optional[np.ndarray] = None) -> tuple[Features, str]:
    path = Path(path)
    sha = file_sha256(path)
    key = hashlib.sha256(json.dumps({"sha": sha, "sr": cfg.sr, "n_fft": cfg.n_fft, "hop": cfg.hop, "n_mels": cfg.n_mels,
                                     "n_mfcc": cfg.n_mfcc, "beats": None if beats is None else "given",
                                     "tracker": getattr(tracker, "info", {}).get("model") if tracker else None},
                                    sort_keys=True).encode()).hexdigest()[:20]
    if cache_dir is not None:
        cp = Path(cache_dir) / f"{key}.npz"
        if cp.is_file():
            return Features.load(cp), sha
    y = load_mono(path, cfg.sr)
    if beats is None:
        beats = (tracker or librosa_beats)(y, cfg.sr)
    feat = compute_features(y, np.asarray(beats, float), cfg)
    del y
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        feat.save(Path(cache_dir) / f"{key}.npz")
    return feat, sha


def analyze_mix(mix_path: str | Path, specs: Sequence[TrackSpec], cfg: Optional[AlignConfig] = None, *,
                tracker=None, cache_dir: Optional[Path] = None, mix_beats: Optional[np.ndarray] = None,
                track_beats: Optional[dict] = None, log: Callable[[str], None] = lambda m: None) -> dict:
    cfg = cfg or AlignConfig()
    mix, mix_sha = features_cached(mix_path, cfg, tracker, cache_dir, beats=mix_beats)
    log(f"mix: {mix.duration_s:.1f} s, {len(mix.beats)} battiti")
    tracks, shas, aligns, analyses, missing = {}, {}, {}, {}, []
    for sp in specs:
        p = Path(sp.audio) if sp.audio else None
        if p is None or not p.is_file():
            missing.append({"slug": sp.slug, "reason": "audio non in libreria"})
            continue
        tb = (track_beats or {}).get(sp.slug)
        if tb is None:
            tb = sp.beats if sp.beats is not None else grid_beats(sp.analysis)
        tracks[sp.slug], shas[sp.slug] = features_cached(p, cfg, tracker, cache_dir, beats=tb)
        analyses[sp.slug] = sp.analysis
        aligns[sp.slug] = align_track(sp.slug, mix, tracks[sp.slug], cfg)
        al = aligns[sp.slug]
        log(f"  {sp.slug}: found={al.found} match={al.match_rate:.3f} x{al.tempo_multiple:g} key{al.key_shift:+d} diag={al.diag_mix_range[0]:.1f}-{al.diag_mix_range[1]:.1f}")
    found = [s for s in aligns if aligns[s].found]
    order = sorted(found, key=lambda s: (aligns[s].diag_mix_range[0] + aligns[s].diag_mix_range[1]) / 2)
    gains = estimate_gains(mix, tracks, {s: aligns[s] for s in found}, cfg)
    transitions = analyze_transitions(mix, tracks, aligns, gains, analyses, cfg, order)
    t = mix.frame_times
    track_rows = []
    for sp in specs:
        if sp.slug not in aligns:
            continue
        al = aligns[sp.slug]
        row = {
            "slug": sp.slug, "artist": sp.artist, "title": sp.title, "audio_sha256": shas.get(sp.slug),
            "found": al.found, "order": order.index(sp.slug) + 1 if sp.slug in order else None,
            "match_rate": round(al.match_rate, 4), "matched_beats": round(al.matched_beats, 1),
            "matched_track_s": round(al.matched_track_s, 2), "cost": None if not np.isfinite(al.cost) else round(al.cost, 4),
            "tempo_multiple": al.tempo_multiple, "key_shift": al.key_shift,
            "diag_mix_s": [round(x, 3) for x in al.diag_mix_range],
            "presence_mix_s": [round(x, 3) for x in al.presence()],
            "extra_segments": al.extra_segments, "candidates": al.candidates,
        }
        if al.found:
            g = _presence_db(gains["gain_db"][sp.slug])
            bpf = max(1, int(round(local_beat_period(mix.beats, float(np.mean(al.diag_mix_range))) * mix.sr / mix.hop)))
            aud = _audible(_median_filter_nan(g, 5), cfg, bpf)
            runs = [(float(t[s0]), float(t[e0 - 1])) for s0, e0 in _runs(aud)]
            row["audible_runs_mix_s"] = [[round(a, 3), round(b, 3)] for a, b in runs if b - a > 1.0]
            step = max(1, len(al.map_mix_t) // 400)
            row["time_map"] = {"mix_s": [round(float(x), 3) for x in al.map_mix_t[::step]],
                               "track_s": [round(float(x), 3) for x in al.map_trk_t[::step]]}
        track_rows.append(row)
    resid = gains["residual"]
    unexplained = []
    for s0, e0 in _runs(np.nan_to_num(resid, nan=1.0) > 0.7):
        if (e0 - s0) * mix.hop / mix.sr >= 8.0:
            unexplained.append([round(float(t[s0]), 2), round(float(t[e0 - 1]), 2)])
    return {
        "schema_version": SCHEMA,
        "method": METHOD_VERSION,
        "mix": {"path": str(mix_path), "sha256": mix_sha, "duration_s": round(mix.duration_s, 3), "n_beats": int(len(mix.beats))},
        "beat_tracker": getattr(tracker, "info", None) if tracker else {"model": "battiti forniti o librosa"},
        "config": cfg.to_json(),
        "order": order,
        "tracks": track_rows,
        "missing": missing,
        "transitions": transitions,
        "unexplained_regions_mix_s": unexplained,
        "notes": [
            "Osservazioni, non regole: tecnica e soglie di udibilità sono ipotesi [H] (vedi docstring).",
            "Le curve sono guadagni stimati (dB rispetto al livello del brano nel suo tratto allineato), non i movimenti reali del mixer.",
        ],
    }


def load_tracklist(path: str | Path) -> list[dict]:
    """Tracklist JSON (lista o {"tracks": [...]}) o CSV con colonne slug/audio/analysis/artist/title."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        import csv

        with open(p, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    data = json.loads(p.read_text(encoding="utf-8"))
    return data["tracks"] if isinstance(data, dict) else data
