"""Renderer di transizioni "da DJ standard" (P0 di REVISIONE_DJ_2026-10-20).

Questo modulo non sceglie coppie né cue: riceve audio, griglie e cue già
allineati a un downbeat (li sceglie un altro modulo) e rende la transizione
come farebbe un DJ su un mixer a due deck:

* ``cut``: A suona fino al downbeat ``cue_out_a``; B parte dal downbeat
  ``cue_in_b`` nello stesso campione, con un micro-crossfade equal-power di
  5-20 ms che *termina* sul downbeat (l'attacco di B arriva a guadagno pieno).
  Nessuno stretch: il cambio di tempo è dichiarato in ``tempo_jump_pct``.
* ``blend``: sovrapposizione di N battute (N in {4, 8, 16}, contate nella
  griglia di A) con beatmatching di B verso A, EQ a 3 bande per deck con bass
  swap sul downbeat centrale, fader monotoni, normalizzazione LUFS e
  limitatore true-peak sull'uscita.

Convenzione dei cue per ``blend`` (default ``cue_convention="kim2020"``, la
definizione di Kim et al. ISMIR 2020 §4.1): ``cue_out_a`` è il downbeat di A in
cui inizia la transizione (A comincia a uscire, B a entrare); ``cue_in_b`` è il
downbeat di B in cui A è completamente uscito e suona solo B. La regione di
transizione è quindi A[cue_out, cue_out + N battute] sovrapposta a
B[cue_in - N battute, cue_in]. Con ``cue_convention="start"`` entrambi i cue
indicano l'inizio della sovrapposizione (B[cue_in, cue_in + N battute]).

Beatmatching: solo B viene stretchato (Rubber Band, motore R3 "fine",
intonazione invariata; il backend è qualificato a ogni processo su toni di 82/440 Hz,
durata e clic, vedi ``STRETCH_CONTRACT``). Con il backend ``rubberband`` (CLI) il time-map porta
ogni downbeat di B nella sovrapposizione esattamente sul downbeat
corrispondente di A (``stretch_mode="grid"``, default: stretch a tratti,
rapporto per battuta, scarto per battuta <= 5% dal medio); con
``stretch_mode="fit"`` il rapporto è costante (retta ai minimi quadrati sulle
coppie di downbeat), utile quando la griglia di B è instabile (intro senza
batteria) a costo di un errore residuo rispetto alle griglie; con il fallback
``ffmpeg`` (filtro ``rubberband`` con motore R3 real-time e ritardo iniziale
compensato; il vecchio R2 ``crisp`` non conservava il pitch ed è rifiutato) il
rapporto è costante e l'errore residuo è dichiarato. Il raddoppio/dimezzamento di tempo è gestito
mappando 1 battuta di A su 2 o 1/2 battute di B. Dopo l'ultima battuta della
sovrapposizione B *resta stretchato* al rapporto medio fino alla fine della
clip: non c'è rampa di ritorno al tempo originale (scelta documentata).

Curve di fader/EQ: monotone (A scende, B sale), come nel modello fader + EQ a
3 bande di Kim, Yang & Nam (DAFx 2022). Le forme precise sono ipotesi di
progetto [H] ispirate a Vande Veire & De Bie 2018 §4.3 e MosaikBox §3.4, non
stime da mix reali:

* fader B: sale da 0 a 1 (quarto di seno) fino allo switch; fader A: 1 fino
  allo switch, poi scende a 0 (quarto di coseno) alla fine;
* bassi (< ``low_hz``): B a zero fino allo switch, A pieno fino allo switch;
  scambio complementare (gA + gB = 1) in ``bass_swap_ms`` che termina sul
  downbeat di switch: mai due bassi pieni insieme;
* alti (> ``high_hz``): B entra a ``high_floor_db`` e sale in dB lineari fino
  allo switch; A scende da 0 dB a ``high_floor_db`` dopo lo switch.

Il filtro a 3 bande è a fase zero e ricostruzione perfetta: low = Butterworth
del 2° ordine applicato avanti/indietro (modulo |H|^2 = Linkwitz-Riley del 4°
ordine), mid/high ottenuti per complemento; con guadagni unitari la somma è
identica all'ingresso.

Voci: se sono forniti gli stem vocali (Demucs) di A e B, misura i secondi in cui
le due voci sono attive e udibili insieme; se superano ``duck_min_overlap_s``
(2 s come in MosaikBox) attenua la voce di A (A - (1-g)*voce_A) dove B canta.

Loudness: A e B vengono portati a ``target_lufs`` (integrato BS.1770 su tutta
la traccia, pyloudnorm) prima del mix; sull'uscita un limitatore look-ahead
true-peak (sovracampionamento 4x) tiene il picco sotto ``ceiling_dbtp``.

Section gain match (``SectionGainConfig``, opzionale, spento per default per
non cambiare i render precedenti) [H]: la normalizzazione sull'intero brano
non basta quando le sezioni che si incontrano hanno livelli diversi (outro di
A basso, ritornello di B alto); un DJ regola gain/fader sulla sezione. Si
misura la loudness BS.1770 gated a blocchi short-term di 3 s (75% di
sovrapposizione, gate -70 LUFS / -10 LU) di A nelle ultime ``a_bars`` battute
prima dell'inizio della transizione e di B nelle prime ``b_bars`` battute dopo
lo switch (per il blend: B già stretchato, in coordinate dell'uscita); a B si
applica il guadagno statico L_A - L_B limitato a +-``max_db``. Dopo la
finestra di misura di B (e mai prima della fine della transizione) B torna al
suo livello normalizzato con una rampa a coseno rialzato in dB di
``ramp_bars`` battute (default 16: <= 0,4 dB per battuta con 6 dB, pensata per
essere impercettibile); ``after="hold"`` mantiene il guadagno. La rampa di
solito finisce dopo la clip: i metadati riportano inizio/fine in campioni e
il guadagno residuo a fine clip. Il limitatore true-peak resta l'ultimo stadio.

Clip: per ``cut`` 10 s + 10 s attorno allo switch (20 s); per ``blend`` 10 s
prima dell'inizio della transizione, transizione intera, 10 s dopo (regole
AGENTS.md del 20/10/2026). L'uscita del renderer *è* la clip.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Optional, Sequence

import numpy as np

DJ_RENDER_VERSION = "musicmix.dj_render.3"  # (+ backend di stretch qualificato, 21/10/2026: PCM invariato col CLI) (+ deck a_raw additivo, 21/10/2026: PCM invariato) .2: section gain match (20/10/2026); .3: bass_gap_beats, mid_duck_b (libri, 21/10/2026, spenti)
DEFAULT_SAMPLE_RATE = 48000
BLEND_BARS = (4, 8, 16)
TRANSITION_KINDS = ("cut", "blend")


class DJRenderError(ValueError):
    """Input non adatto a una transizione DJ (griglia, tempo, cue, durata)."""


class TempoIncompatibleError(DJRenderError):
    """Il rapporto di tempo (anche ripiegato d'ottava) supera il limite di stretch."""


@dataclass(frozen=True)
class SectionGainConfig:
    """Section gain match di B sul materiale di A che lo precede (vedi docstring del modulo).

    Tutti i valori sono ipotesi di progetto [H] fissate a priori, non stimate
    da mix reali; la finestra short-term di 3 s e il gating sono BS.1770 /
    EBU Tech 3341 [S] (definizione della misura, non della soglia).
    """

    enabled: bool = False
    a_bars: int = 2  # A: last bars before the transition start (2..4)
    b_bars: int = 4  # B: first bars after the switch (4..8), stretched B for blends
    max_db: float = 6.0  # |gain| limit
    window_s: float = 3.0  # BS.1770 short-term block, gated
    after: str = "ramp"  # "ramp": back to B's normalized level | "hold": keep the gain
    ramp_bars: int = 16  # slow raised-cosine (in dB) return, in output bars of B


@dataclass(frozen=True)
class DJRenderConfig:
    target_lufs: float = -14.0
    ceiling_dbtp: float = -1.0
    cue_convention: str = "kim2020"  # "kim2020" | "start"
    crossfade_ms: float = 10.0  # cut: 5..20 ms
    max_stretch: float = 0.08  # |tempo ratio - 1| (MosaikBox)
    max_bar_deviation: float = 0.12  # each bar within +-12% of the local median
    max_bar_rate_deviation: float = 0.05  # per-bar stretch vs mean stretch
    snap_tolerance_s: float = 0.03
    low_hz: float = 200.0
    high_hz: float = 2500.0
    bass_swap_ms: float = 20.0
    high_floor_db: float = -12.0
    duck_vocals: bool = True
    duck_gain_db: float = -24.0
    duck_min_overlap_s: float = 2.0
    vocal_rel_db: float = -24.0
    vocal_abs_floor: float = 0.002
    audible_gain: float = 0.25  # fader >= -12 dB counts as audible
    pre_s: float = 10.0
    post_s: float = 10.0
    stretch_backend: str = "auto"  # "auto" | one of STRETCH_BACKENDS; refused unless it passes STRETCH_CONTRACT
    stretch_mode: str = "grid"  # "grid" (per-bar time map) | "fit" (constant ratio, least squares)
    stretch_preroll_s: float = 1.0
    limiter_hold_ms: float = 20.0
    align_window_s: float = 0.5
    align_max_lag_ms: float = 40.0
    section_gain: SectionGainConfig = SectionGainConfig()
    # ---- libri (21/10/2026), blend: default neutri = comportamento .2 invariato
    # [P] B&B 2003 cap. 28 "How to EQ" ('Slam in the bass': "cut the bass for a few bars during a buildup in the music and then
    # return it, bamm, at exactly the right moment"); [A] Butler 2006 p. 246 (bassi tolti per 4 battiti alla fine di 16 battute,
    # poi "dropping the beat"); Butler pp. 242-243 (togliere la cassa di un disco per agganciarlo al breakdown dell'altro).
    # Battiti di sola parte medio-alta prima dello swap: il basso di A esce bass_gap_beats prima dello switch e quello di B
    # entra di colpo sullo switch (un solo basso alla volta, come prima). 0 = spento.
    bass_gap_beats: float = 0.0
    # [A] Butler 2006 p. 53 ("if two synthesizer lines create too much pitch dissonance, one of them could be excised from the
    # mix"); [P] Webber 2008 p. 181 (kill di banda: "cancel out everything but the kick and bass"); [P] B&B cap. 15 ("a natty bit
    # of filtering"). Medi di B (low_hz..high_hz) a mid_duck_b_db per le prime mid_duck_b_bars battute del blend, poi rampa in dB
    # di una battuta fino a 0 dB (al più tardi sullo switch). 0 battute = spento.
    mid_duck_b_bars: int = 0
    mid_duck_b_db: float = -12.0


@dataclass
class DJRenderResult:
    audio: np.ndarray  # (frames, 2) float32, the listening clip
    sample_rate: int
    metadata: dict
    decks: Optional[dict] = field(default=None, repr=False)


# --------------------------------------------------------------------------- I/O


def load_track(path: os.PathLike | str) -> tuple[np.ndarray, int]:
    """Legge un wav come (frames, 2) float64 e il sample rate."""
    import soundfile as sf

    data, sr = sf.read(os.fspath(path), dtype="float64", always_2d=True)
    return _as_stereo(data), int(sr)


def load_beat_grid(structure_path: os.PathLike | str) -> tuple[np.ndarray, np.ndarray]:
    """Beat e downbeat (secondi) da un JSON ``musicmix.structure.v2`` (Beat This!)."""
    with open(structure_path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    sr = float(doc["sample_rate"])
    beats = np.array([event["sample"] for event in doc.get("beats", [])], dtype=float) / sr
    downbeats = np.array([event["sample"] for event in doc.get("downbeats", [])], dtype=float) / sr
    return np.sort(beats), np.sort(downbeats)


def write_render(result: DJRenderResult, out_dir: os.PathLike | str, name: str) -> dict:
    """Scrive ``<name>.wav`` (PCM 24 bit) e ``<name>.json`` con i metadati."""
    import soundfile as sf

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wav_path = out / f"{name}.wav"
    sf.write(os.fspath(wav_path), result.audio, result.sample_rate, subtype="PCM_24")
    digest = hashlib.sha256(wav_path.read_bytes()).hexdigest()
    meta = dict(result.metadata)
    meta["output"] = {
        **meta.get("output", {}),
        "wav": wav_path.name,
        "wav_sha256": digest,
        "wav_subtype": "PCM_24",
    }
    json_path = out / f"{name}.json"
    json_path.write_text(json.dumps(_jsonable(meta), indent=1, ensure_ascii=False), encoding="utf-8")
    return {"wav": wav_path, "json": json_path, "sha256": digest}


# ------------------------------------------------------------------- loudness


def integrated_lufs(audio: np.ndarray, sr: int) -> float:
    import pyloudnorm

    x = _as_stereo(audio)
    if len(x) < int(0.4 * sr) + 1:
        return float("-inf")
    return float(pyloudnorm.Meter(sr).integrated_loudness(x))


def true_peak_dbtp(audio: np.ndarray, sr: int, oversample: int = 4) -> float:
    peak = float(np.max(_true_peak_envelope(_as_stereo(audio), oversample))) if len(audio) else 0.0
    return 20.0 * math.log10(peak) if peak > 0 else float("-inf")


def _true_peak_envelope(x: np.ndarray, oversample: int = 4) -> np.ndarray:
    """Per-sample max |x| over the 4x-oversampled signal (all channels)."""
    from scipy.signal import resample_poly

    x32 = np.asarray(x, dtype=np.float32)
    n = len(x32)
    env = np.zeros(n, dtype=np.float32)
    for ch in range(x32.shape[1]):
        up = np.abs(resample_poly(x32[:, ch], oversample, 1))
        env = np.maximum(env, up[: n * oversample].reshape(n, oversample).max(axis=1))
    return np.maximum(env, np.max(np.abs(x32), axis=1)).astype(np.float64)


def true_peak_limit(audio: np.ndarray, sr: int, ceiling_dbtp: float = -1.0, hold_ms: float = 20.0) -> tuple[np.ndarray, dict]:
    """Limitatore look-ahead a fase zero (offline) sul picco true-peak 4x.

    Il guadagno richiesto viene filtrato con un minimo mobile di semiampiezza
    ``hold_ms`` e poi lisciato con una finestra di Hann della stessa ampiezza:
    la media di valori <= g(p) resta <= g(p), quindi ogni picco viene ridotto
    abbastanza. Un trim statico finale copre l'errore di interpolazione.
    """
    from scipy.ndimage import minimum_filter1d
    from scipy.signal import fftconvolve

    x = _as_stereo(audio).astype(np.float64)
    ceiling = 10.0 ** (ceiling_dbtp / 20.0)
    if len(x) == 0:
        return x, {"ceiling_dbtp": ceiling_dbtp, "true_peak_before_dbtp": None, "max_gain_reduction_db": 0.0, "final_trim_db": 0.0, "true_peak_after_dbtp": None}
    env = _true_peak_envelope(x)
    peak = float(env.max())
    tp_before = 20.0 * math.log10(peak) if peak > 0 else float("-inf")
    info = {"ceiling_dbtp": ceiling_dbtp, "true_peak_before_dbtp": tp_before}
    env = np.maximum.reduce([env, np.roll(env, 1), np.roll(env, -1)])
    target = ceiling * 0.98
    g_req = np.minimum(1.0, target / np.maximum(env, 1e-12))
    gain = np.ones(len(x))
    if np.any(g_req < 1.0):
        half = max(1, int(round(hold_ms / 1000.0 * sr)))
        g_min = minimum_filter1d(g_req, size=2 * half + 1, mode="nearest")
        window = np.hanning(2 * half + 3)[1:-1]
        window /= window.sum()
        padded = np.pad(g_min, half, mode="edge")
        gain = np.minimum(fftconvolve(padded, window, mode="valid"), 1.0)
    y = x * gain[:, None]
    trim = 1.0
    for _ in range(4):
        tp = true_peak_dbtp(y, sr)
        if tp <= ceiling_dbtp:
            break
        step = 10.0 ** ((ceiling_dbtp - 0.02 - tp) / 20.0)
        y *= step
        trim *= step
    info.update(
        max_gain_reduction_db=float(-20.0 * math.log10(max(float(gain.min()), 1e-12))),
        limited_seconds=float(np.sum(gain < 0.999) / sr),
        final_trim_db=float(20.0 * math.log10(trim)),
        true_peak_after_dbtp=true_peak_dbtp(y, sr),
    )
    return y, info


# ------------------------------------------------------------ section gain match


def gated_short_term_lufs(audio: np.ndarray, sr: int, window_s: float = 3.0) -> dict:
    """Loudness BS.1770 di un tratto: blocchi short-term di ``window_s`` (75% di
    sovrapposizione), gate assoluto -70 LUFS e relativo -10 LU (pyloudnorm con
    ``block_size=window_s``). Se il tratto è più corto della finestra si usa un
    unico blocco lungo quanto il tratto. ``ungated_lufs`` = un blocco su tutto."""
    import pyloudnorm

    x = _as_stereo(audio)
    dur = len(x) / sr
    if len(x) < int(0.4 * sr):
        return {"lufs": None, "ungated_lufs": None, "duration_s": dur, "window_s": window_s, "blocks": 0}
    whole = (len(x) - 1) / sr  # one sample short: pyloudnorm rejects block_size * rate > length by rounding
    block = min(float(window_s), whole)
    v = float(pyloudnorm.Meter(sr, block_size=block).integrated_loudness(x))
    u = float(pyloudnorm.Meter(sr, block_size=whole).integrated_loudness(x))
    n_blocks = int(np.round((dur - block) / (block * 0.25))) + 1
    return {"lufs": v if math.isfinite(v) else None, "ungated_lufs": u if math.isfinite(u) else None,
            "duration_s": dur, "window_s": block, "blocks": n_blocks}


def section_gain_curve(n: int, gain_db: float, ramp_start: int, ramp_len: int, after: str = "ramp") -> np.ndarray:
    """Guadagno lineare per campione: ``gain_db`` statico fino a ``ramp_start``, poi
    (``after="ramp"``) ritorno a 0 dB con coseno rialzato in dB lungo ``ramp_len``."""
    g_db = np.full(n, float(gain_db))
    if after == "ramp" and gain_db != 0.0:
        u = np.clip((np.arange(n) - ramp_start) / max(int(ramp_len), 1), 0.0, 1.0)
        g_db = gain_db * 0.5 * (1.0 + np.cos(np.pi * u))
    return 10.0 ** (g_db / 20.0)


def section_gain_match(a_ref: np.ndarray, b_ref: np.ndarray, sr: int, sg: SectionGainConfig) -> dict:
    """Guadagno statico per B = L(A, ultime battute) - L(B, prime battute), limitato a +-max_db."""
    la = gated_short_term_lufs(a_ref, sr, sg.window_s)
    lb = gated_short_term_lufs(b_ref, sr, sg.window_s)
    out = {"a_lufs": la["lufs"], "b_lufs": lb["lufs"], "a_measure": la, "b_measure": lb}
    if la["lufs"] is None or lb["lufs"] is None:
        out.update(raw_db=None, applied_db=0.0, clamped=False, status="not measurable (silent or too short): no gain")
        return out
    raw = la["lufs"] - lb["lufs"]
    applied = float(np.clip(raw, -sg.max_db, sg.max_db))
    out.update(raw_db=raw, applied_db=applied, clamped=bool(abs(raw) > sg.max_db), status="applied")
    return out


def plan_section_gain(
    a_norm: np.ndarray,
    a_end_src: int,
    a_bar_s: float,
    b_ref: np.ndarray,
    b_ref_out_start: int,
    b_bar_out_s: float,
    n_out: int,
    sr: int,
    sg: SectionGainConfig,
    *,
    transition_end_out: Optional[int] = None,
    b_ref_note: str = "",
) -> tuple[np.ndarray, dict]:
    """Curva di guadagno di B (campioni della clip) e metadati del section gain match.

    ``a_norm``/``a_end_src``: A normalizzato e campione sorgente di inizio
    transizione; ``b_ref``: B (normalizzato, già stretchato se il blend lo
    stretcha) nelle prime ``b_bars`` battute dopo lo switch, che nella clip
    inizia a ``b_ref_out_start``; ``b_bar_out_s``: battuta di B nella clip.
    """
    if not sg.enabled:
        return np.ones(n_out), {"enabled": False, "applied_db": 0.0}
    if sg.after not in ("ramp", "hold"):
        raise DJRenderError("section_gain.after must be 'ramp' or 'hold'")
    if sg.a_bars < 1 or sg.b_bars < 1 or sg.max_db < 0 or sg.ramp_bars < 1:
        raise DJRenderError("section_gain: a_bars, b_bars, ramp_bars >= 1 and max_db >= 0")
    a_len = int(round(sg.a_bars * a_bar_s * sr))
    a0 = max(0, a_end_src - a_len)
    a_ref = a_norm[a0:a_end_src]
    m = section_gain_match(a_ref, b_ref, sr, sg)
    b_end_out = b_ref_out_start + len(b_ref)
    ramp_start = max(b_end_out, transition_end_out if transition_end_out is not None else b_end_out)
    ramp_len = int(round(sg.ramp_bars * b_bar_out_s * sr))
    curve = section_gain_curve(n_out, m["applied_db"], ramp_start, ramp_len, sg.after)
    end_db = float(20.0 * math.log10(curve[-1])) if n_out else m["applied_db"]
    info = {
        "enabled": True,
        **m,
        "a_region_source_samples": [a0, a_end_src],
        "a_region_bars": sg.a_bars,
        "a_region_s": a_len / sr,
        "b_region_output_samples": [b_ref_out_start, b_end_out],
        "b_region_bars": sg.b_bars,
        "b_region_s": len(b_ref) / sr,
        "b_region_note": b_ref_note,
        "max_db": sg.max_db,
        "after": sg.after,
        "ramp": {
            "shape": "raised cosine in dB, applied_db -> 0 dB (B back to its whole-track normalized level)" if sg.after == "ramp" else "none (gain held)",
            "bars": sg.ramp_bars if sg.after == "ramp" else 0,
            "output_start_sample": ramp_start if sg.after == "ramp" else None,
            "output_end_sample": (ramp_start + ramp_len) if sg.after == "ramp" else None,
            "ends_after_clip": bool(sg.after == "ramp" and ramp_start + ramp_len > n_out),
        },
        "gain_db_at_clip_end": end_db,
        "method": "BS.1770 K-weighted loudness, 3 s short-term blocks (75% overlap), gated -70 LUFS / -10 LU (pyloudnorm); gain = L_A - L_B clamped to +-max_db, applied to deck B only",
        "basis": "[H] DJ practice: trim/fader set on the section that meets, not on the whole track",
    }
    return curve, info


# --------------------------------------------------------------------- EQ / bands


def split_bands(x: np.ndarray, sr: int, low_hz: float = 200.0, high_hz: float = 2500.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tre bande a fase zero con low + mid + high == x (ricostruzione esatta).

    low = Butterworth-2 avanti/indietro (|H|^2: modulo di un Linkwitz-Riley 4);
    il complemento x - low ha modulo |H_hp|^2 perché il Butterworth è
    power-complementary. Lo stesso a ``high_hz`` separa mid e high.
    """
    from scipy.signal import butter, sosfiltfilt

    x = np.asarray(x, dtype=np.float64)
    pad = min(len(x) - 1, int(sr * 0.5)) if len(x) > 1 else 0
    sos_low = butter(2, low_hz, btype="low", fs=sr, output="sos")
    low = sosfiltfilt(sos_low, x, axis=0, padlen=pad) if pad > 0 else x * 0.0
    rest = x - low
    sos_mid = butter(2, high_hz, btype="low", fs=sr, output="sos")
    mid = sosfiltfilt(sos_mid, rest, axis=0, padlen=pad) if pad > 0 else rest
    high = rest - mid
    return low, mid, high


# ------------------------------------------------------------------ tempo / grid


def choose_tempo_fold(bar_a_s: float, bar_b_s: float, max_stretch: float = 0.08) -> tuple[float, float]:
    """Sceglie k in {1/2, 1, 2} (battute di B per battuta di A) e la velocità di B.

    ``rate = k * bar_b / bar_a`` è il fattore di tempo da applicare a B
    (>1 accelera). Solleva ``TempoIncompatibleError`` se |rate-1| > max_stretch.
    """
    best = None
    for k in (1.0, 0.5, 2.0):
        rate = k * bar_b_s / bar_a_s
        score = abs(math.log(rate))
        if best is None or score < best[2] - 1e-12:
            best = (k, rate, score)
    k, rate, _ = best
    if abs(rate - 1.0) > max_stretch + 1e-9:
        raise TempoIncompatibleError(
            f"tempo ratio {rate:.4f} (fold k={k:g}) outside +-{max_stretch:.0%}: use a cut, not a blend"
        )
    return k, rate


def _snap(grid: np.ndarray, t: float, tol: float, label: str) -> int:
    if len(grid) == 0:
        raise DJRenderError(f"{label}: empty downbeat grid")
    i = int(np.argmin(np.abs(grid - t)))
    if abs(grid[i] - t) > tol:
        raise DJRenderError(f"{label}={t:.3f}s is {1000*abs(grid[i]-t):.0f} ms from the nearest downbeat")
    return i


def _check_bars(grid: np.ndarray, i0: int, i1: int, max_dev: float, label: str) -> np.ndarray:
    if i0 < 0 or i1 >= len(grid) or i1 <= i0:
        raise DJRenderError(f"{label}: needs downbeats {i0}..{i1}, grid has {len(grid)}")
    bars = np.diff(grid[i0 : i1 + 1])
    med = float(np.median(bars))
    bad = np.where(np.abs(bars / med - 1.0) > max_dev)[0]
    if len(bad):
        raise DJRenderError(
            f"{label}: irregular downbeat grid (bars {', '.join(f'{bars[j]:.2f}s' for j in bad[:4])} vs median {med:.2f}s)"
        )
    return bars


def _local_bar(grid: np.ndarray, i: int, direction: int, n: int = 8) -> float:
    if direction < 0:
        seg = grid[max(0, i - n) : i + 1]
    else:
        seg = grid[i : min(len(grid), i + n + 1)]
    if len(seg) < 2:
        seg = grid[max(0, i - n) : min(len(grid), i + n + 1)]
    if len(seg) < 2:
        raise DJRenderError("downbeat grid too short to estimate the bar length")
    return float(np.median(np.diff(seg)))


def _local_bpm(beats: np.ndarray, downbeats: np.ndarray, t0: float, t1: float) -> Optional[float]:
    """BPM = beats per bar (median count) * 60 / mean bar length in [t0, t1].

    Uses the bar span instead of inter-beat intervals, which Beat This! quantises
    to 20 ms frames (e.g. 0.54/0.56 s -> 111/107 BPM).
    """
    d = downbeats[(downbeats >= t0 - 1e-6) & (downbeats <= t1 + 1e-6)]
    if len(d) < 2:
        return None
    counts = [np.sum((beats >= d[i] - 0.01) & (beats < d[i + 1] - 0.01)) for i in range(len(d) - 1)]
    bpb = float(np.round(np.median(counts))) if counts else 4.0  # 3.5 -> 4
    bpb = bpb if bpb > 0 else 4.0
    return float(bpb * 60.0 * (len(d) - 1) / (d[-1] - d[0]))


def _bar_estimate(downbeats: np.ndarray, beats: np.ndarray, i: int, prefer: int, max_dev: float, n: int = 4) -> tuple[float, str]:
    """Bar length near downbeat i: regular bars on the preferred side, else the
    other side, else 4 x median inter-beat interval (flagged)."""
    for side in (prefer, -prefer):
        i0, i1 = (i - n, i) if side < 0 else (i, i + n)
        if i0 < 0 or i1 >= len(downbeats):
            continue
        bars = np.diff(downbeats[i0 : i1 + 1])
        if np.all(np.abs(bars / np.median(bars) - 1.0) <= max_dev):
            return float((downbeats[i1] - downbeats[i0]) / n), ("downbeats before" if side < 0 else "downbeats after")
    t = downbeats[i]
    sel = beats[(beats >= t - 8.0) & (beats <= t + 8.0)]
    if len(sel) >= 3:
        return float(4.0 * np.median(np.diff(sel))), "4 x median beat interval (irregular downbeat grid)"
    raise DJRenderError("cannot estimate the bar length near the cue")


class _PiecewiseMap:
    """Monotone piecewise-linear map src -> dst with linear extrapolation."""

    def __init__(self, src: Sequence[float], dst: Sequence[float], slope_before: float, slope_after: float):
        self.src = np.asarray(src, dtype=float)
        self.dst = np.asarray(dst, dtype=float)
        self.sb = float(slope_before)
        self.sa = float(slope_after)

    def fwd(self, s):
        s = np.asarray(s, dtype=float)
        out = np.interp(s, self.src, self.dst)
        out = np.where(s < self.src[0], self.dst[0] + (s - self.src[0]) * self.sb, out)
        return np.where(s > self.src[-1], self.dst[-1] + (s - self.src[-1]) * self.sa, out)

    def inv(self, d):
        d = np.asarray(d, dtype=float)
        out = np.interp(d, self.dst, self.src)
        out = np.where(d < self.dst[0], self.src[0] + (d - self.dst[0]) / self.sb, out)
        return np.where(d > self.dst[-1], self.src[-1] + (d - self.dst[-1]) / self.sa, out)


# ----------------------------------------------------------------- time stretch
#
# Qualificazione dei backend (P0.1 dell'audit dei libri, 21/10/2026; RENDER_TRIAGE.md). Il contratto è misurato,
# non presunto: un backend si usa solo se su segnali sintetici conserva il pitch (82 Hz e 440 Hz), la durata e i
# transienti (clic) con stretch +-3/7/12 %. Il vecchio fallback ffmpeg (``rubberband`` R2 real-time,
# ``transients=crisp``) porta 82 Hz a 78,1 Hz con time_ratio 1,05 (Windows ffmpeg 9.0 e Linux ffmpeg 6.1, stessa
# deviazione): non è più usabile. Il nuovo fallback ``ffmpeg`` chiede a librubberband il motore R3 (flag
# OptionEngineFiner passato come valore numerico dell'opzione ``transients``, che ffmpeg OR-a nelle opzioni) e
# compensa a ogni chiamata il ritardo iniziale del motore real-time, che ffmpeg non scarta, con un pilota di clic
# della stessa durata e rapporto. Se la libreria non supporta R3 (< 3.0) il flag è ignorato, il pitch fallisce e il
# backend è rifiutato. Soglie [H] salvo il pitch a 82 Hz +-0,5 Hz, che è il contratto dei test del renderer.

_RB_OPTION_ENGINE_FINER = 0x20000000  # RubberBandOptionEngineFiner (rubberband-c.h, Rubber Band >= 3.0)
STRETCH_TIME_RATIOS = (0.88, 0.93, 0.97, 1.03, 1.07, 1.12)  # output/input duration: stretch +-3, +-7, +-12 %
STRETCH_QUICK_RATIOS = (0.93, 1.07)
STRETCH_CONTRACT = {
    "pitch_82_tol_hz": 0.5,  # contract of tests/test_dj_render.py::test_stretch_preserves_pitch (never relaxed to 78 Hz)
    "pitch_440_tol_hz": 2.0,  # [H] about 8 cents
    "duration_tol_ms": 1.0,  # [H] |output length - round(input length * time_ratio)|
    "click_timing_tol_ms": 2.0,  # [H] one fifth of the 10 ms downbeat tolerance of the renderer/QA
    "click_width_max_ratio": 2.0,  # [H] median width of the click envelope above -6 dB, output / input
    "click_level_min_db": -3.0,  # [H] median click peak, output / input
    "tone_hf_residual_max_db": -60.0,  # [H] energy > 1 kHz on a pure 82 Hz tone (splice clicks, phasiness)
}
_FFMPEG_VARIANTS = {
    # name: (filter template, engine description, compensate start delay with a pilot)
    "ffmpeg": (f"rubberband=tempo={{tempo}}:transients={_RB_OPTION_ENGINE_FINER}:channels=together",
               "librubberband R3 (EngineFiner flag) via ffmpeg af_rubberband, real-time; start delay compensated per call with a click pilot", True),
    "ffmpeg-r2-crisp": ("rubberband=tempo={tempo}:pitchq=quality:transients=crisp:channels=together",
                        "librubberband R2 real-time via ffmpeg, transients=crisp (fallback until 21/10/2026: pitch not preserved)", False),
    "ffmpeg-r2-smooth": ("rubberband=tempo={tempo}:pitchq=quality:transients=smooth:window=short:channels=together",
                         "librubberband R2 real-time via ffmpeg, transients=smooth, window=short", False),
    "ffmpeg-atempo": ("atempo={tempo}", "ffmpeg atempo (WSOLA)", False),
}
STRETCH_BACKENDS = ("rubberband",) + tuple(_FFMPEG_VARIANTS)
_QUALIFICATION_CACHE: dict = {}


def _stretch_descriptor(name: str) -> Optional[dict]:
    """Backend description without qualification; None when the binary is missing."""
    if name == "rubberband":
        cli = shutil.which("rubberband")
        if not cli:
            return None
        version = subprocess.run([cli, "--version"], capture_output=True, text=True).stdout.strip() or None
        version = version or subprocess.run([cli, "-V"], capture_output=True, text=True).stderr.strip()
        return {"backend": "rubberband", "variant": "rubberband", "binary": cli, "version": version,
                "engine": "R3 (--fine), offline, --centre-focus", "keyframes": True}
    if name in _FFMPEG_VARIANTS:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return None
        af, engine, pilot = _FFMPEG_VARIANTS[name]
        return {"backend": "ffmpeg", "variant": name, "binary": ffmpeg, "engine": engine, "filter": af,
                "latency_pilot": pilot, "keyframes": False}
    raise DJRenderError(f"unknown stretch backend {name!r}; expected 'auto' or one of {STRETCH_BACKENDS}")


def stretch_backend_info(name: str = "auto", *, qualify: bool = True, sample_rate: int = DEFAULT_SAMPLE_RATE) -> dict:
    """Pick a time-stretch backend that passes ``STRETCH_CONTRACT``.

    ``auto`` tries the Rubber Band CLI (R3) and then the ffmpeg R3 fallback. A named backend that fails the
    contract is refused (``DJRenderError``), never used silently. ``qualify=False`` only describes the backend
    (measurement tools); the renderer always qualifies. The result carries a short ``qualification`` summary.
    """
    names = ("rubberband", "ffmpeg") if name == "auto" else (name,)
    rejected = []
    for nm in names:
        desc = _stretch_descriptor(nm)
        if desc is None:
            rejected.append({"backend": nm, "reason": "binary not found"})
            continue
        if not qualify:
            return desc
        q = qualify_stretch_backend(desc, sample_rate)
        if q["passed"]:
            return dict(desc, qualification=q["summary"])
        rejected.append({"backend": nm, "reason": "fails the stretch contract", "failed": q["summary"]["failed"]})
    hint = "install rubberband-cli (Rubber Band R3) or an ffmpeg linked to librubberband >= 3.0"
    raise DJRenderError(f"no qualified time-stretch backend ({hint}): {json.dumps(rejected)[:600]}")


def _write_read_tmp(cmd_for, segment: np.ndarray, sr: int, tmp: str, tag: str) -> np.ndarray:
    import soundfile as sf

    src = os.path.join(tmp, f"{tag}_in.wav")
    dst = os.path.join(tmp, f"{tag}_out.wav")
    sf.write(src, np.asarray(segment, dtype=np.float32), sr, subtype="FLOAT")
    proc = subprocess.run(cmd_for(src, dst), capture_output=True, text=True)
    if proc.returncode != 0:
        raise DJRenderError(f"time-stretch failed: {proc.stderr.strip()[:400]}")
    out, out_sr = sf.read(dst, dtype="float64", always_2d=True)
    if out_sr != sr:
        raise DJRenderError(f"stretch output sample rate {out_sr} != {sr}")
    return _as_stereo(out)


def _click_signal(n: int, sr: int, period_s: float, start_s: float, bass_hz: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
    """3 kHz decaying bursts (10 ms) every ``period_s`` (+ optional sustained bass), stereo; nominal onsets."""
    nb = int(round(0.01 * sr))
    k = np.arange(nb)
    burst = np.exp(-k / (0.00125 * sr)) * np.sin(2 * np.pi * 3000.0 * k / sr)
    x = np.zeros(n)
    onsets = np.arange(int(round(start_s * sr)), n - nb - int(0.25 * sr), int(round(period_s * sr)))
    for i in onsets:
        x[i : i + nb] += burst
    if bass_hz:
        x += 0.3 * np.sin(2 * np.pi * bass_hz * np.arange(n) / sr)
    return np.stack([x, x], axis=1), onsets


def _click_features(y: np.ndarray, sr: int, expected: np.ndarray, search_ms: float) -> np.ndarray:
    """Per expected click: (onset sample at -6 dB of the local peak, peak, width ms above -6 dB); NaN if absent.

    One-to-one: each click is searched only in its own window around the expected sample, never reused.
    """
    from scipy.signal import butter, hilbert, sosfiltfilt

    sos = butter(4, [2500.0, 3500.0], btype="band", fs=sr, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, y.mean(axis=1))))
    w = int(round(search_ms / 1000.0 * sr))
    floor = 1e-4 * (env.max() + 1e-12)
    rows = []
    for e in expected:
        lo, hi = max(0, int(e) - w), min(len(env), int(e) + w)
        seg = env[lo:hi]
        if hi - lo < 3 or seg.max() <= floor:
            rows.append((np.nan, 0.0, np.nan))
            continue
        pk = float(seg.max())
        above = np.flatnonzero(seg > 0.5 * pk)
        rows.append((float(lo + above[0]), pk, (above[-1] - above[0]) / sr * 1000.0))
    return np.array(rows, dtype=float)


def _ffmpeg_cmd(backend: dict, time_ratio: float):
    af = backend["filter"].format(tempo=f"{1.0 / time_ratio:.12f}")
    return lambda s, d: [backend["binary"], "-y", "-loglevel", "error", "-i", s, "-af", af, "-c:a", "pcm_f32le", d]


def _pilot_start_delay(backend: dict, n: int, sr: int, time_ratio: float, tmp: str) -> dict:
    """Start delay of the real-time stretcher, measured on a click pilot with the same ratio (output samples).

    The pilot output places clicks at r*t + delay; the median of (onset_out - r*onset_in) over the pilot clicks is
    the delay to drop (negative: pad). Onsets of the pilot input come from the same detector (bias cancels).
    """
    n_p = int(min(n, 12 * sr))
    pilot, nominal = _click_signal(n_p, sr, 0.25, 0.25)
    if len(nominal) < 4:
        return {"start_delay_samples": 0, "pilot_clicks": int(len(nominal)), "pilot_residual_ms": None}
    pout = _write_read_tmp(_ffmpeg_cmd(backend, time_ratio), pilot, sr, tmp, "pilot")
    fin = _click_features(pilot, sr, nominal, 20.0)[:, 0]
    fout = _click_features(pout, sr, np.round(fin * time_ratio), 150.0)[:, 0]
    ok = np.isfinite(fin) & np.isfinite(fout)
    d = fout[ok] - time_ratio * fin[ok]
    delay = int(round(float(np.median(d)))) if ok.sum() >= 4 else 0
    resid = float(np.max(np.abs(d - delay)) / sr * 1000.0) if ok.sum() >= 4 else None
    return {"start_delay_samples": delay, "pilot_clicks": int(ok.sum()), "pilot_residual_ms": resid}


def _run_stretch(segment: np.ndarray, sr: int, time_ratio: float, keyframes: Optional[list[tuple[int, int]]], backend: dict,
                 report: Optional[dict] = None) -> np.ndarray:
    """Stretch ``segment`` by ``time_ratio`` (output/input duration) with the given backend.

    ``report`` (optional dict) receives the start-delay compensation of the ffmpeg R3 fallback.
    """
    with tempfile.TemporaryDirectory(prefix="dj_render_") as tmp:
        if backend["backend"] == "rubberband":
            map_args: list[str] = []
            if keyframes:
                map_path = os.path.join(tmp, "map.txt")
                with open(map_path, "w", encoding="ascii") as handle:
                    for s, t in keyframes:
                        handle.write(f"{int(s)} {int(t)}\n")
                map_args = ["-M", map_path]
            cmd = lambda s, d: [backend["binary"], "-q", "-3", "--centre-focus", "-t", f"{time_ratio:.12f}", *map_args, s, d]
            return _write_read_tmp(cmd, segment, sr, tmp, "seg")
        if "filter" not in backend:
            raise DJRenderError("ffmpeg backend without a filter: build it with stretch_backend_info()")
        out = _write_read_tmp(_ffmpeg_cmd(backend, time_ratio), segment, sr, tmp, "seg")
        if backend.get("latency_pilot"):
            info = _pilot_start_delay(backend, len(segment), sr, time_ratio, tmp)
            delay = info["start_delay_samples"]
            out = out[delay:] if delay >= 0 else np.vstack([np.zeros((-delay, out.shape[1])), out])
            out = _fit_len(out, int(round(len(segment) * time_ratio)))
            if report is not None:
                report.update(info)
        return out


def _peak_hz(y: np.ndarray, sr: int, f0: float, trim_s: float = 0.5) -> float:
    m = y.mean(axis=1)
    t = int(round(trim_s * sr))
    m = m[t : len(m) - t]
    nfft = 8 * len(m)
    spec = np.abs(np.fft.rfft(m * np.hanning(len(m)), nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    band = np.flatnonzero((freqs > f0 / 2) & (freqs < 2 * f0))
    i = int(band[np.argmax(spec[band])])
    a, b, c = np.log(spec[i - 1 : i + 2] + 1e-30)
    den = a - 2 * b + c
    return float(freqs[i] + (0.5 * (a - c) / den if den else 0.0) * (freqs[1] - freqs[0]))


def _hf_residual_db(y: np.ndarray, sr: int, cut_hz: float = 1000.0, trim_s: float = 0.5) -> float:
    m = y.mean(axis=1)
    t = int(round(trim_s * sr))
    m = m[t : len(m) - t]
    p = np.abs(np.fft.rfft(m * np.hanning(len(m)))) ** 2
    f = np.fft.rfftfreq(len(m), 1.0 / sr)
    return float(10 * np.log10(p[f > cut_hz].sum() / (p.sum() + 1e-30) + 1e-30))


def measure_stretch_backend(backend: dict, sample_rate: int = DEFAULT_SAMPLE_RATE, time_ratios: Sequence[float] = STRETCH_TIME_RATIOS) -> dict:
    """Measure pitch (82/440 Hz tones), duration and clicks for each time ratio; check ``STRETCH_CONTRACT``.

    Synthetic signals only: this qualifies the backend's contract (pitch kept, length kept, transients kept), not
    the musical quality of a stretch. A 82 Hz bass under the click train is measured as a diagnostic (the
    renderer test fixture has the same mix).
    """
    sr = int(sample_rate)
    c = STRETCH_CONTRACT
    n_tone = 4 * sr
    tt = np.arange(n_tone) / sr
    tones = {f0: np.repeat((0.3 * np.sin(2 * np.pi * f0 * tt))[:, None], 2, axis=1) for f0 in (82.0, 440.0)}
    clicks, nominal = _click_signal(6 * sr, sr, 0.5, 0.5, bass_hz=82.0)
    fin = _click_features(clicks, sr, nominal, 20.0)
    rows = []
    for r in time_ratios:
        r = float(r)
        row: dict = {"time_ratio": r, "stretch_pct": round(100.0 * (r - 1.0), 3)}
        try:
            for f0, x in tones.items():
                rep: dict = {}
                y = _run_stretch(x, sr, r, None, backend, report=rep)
                key = f"tone_{int(f0)}"
                row[f"{key}_hz"] = _peak_hz(y, sr, f0)
                row[f"{key}_err_hz"] = row[f"{key}_hz"] - f0
                row[f"{key}_cents"] = 1200.0 * math.log2(row[f"{key}_hz"] / f0)
                row[f"{key}_duration_err_samples"] = int(len(y) - round(len(x) * r))
                if f0 == 82.0:
                    row["tone_82_hf_residual_db"] = _hf_residual_db(y, sr)
                    if rep:
                        row["start_delay_samples"] = rep.get("start_delay_samples")
            rep = {}
            y = _run_stretch(clicks, sr, r, None, backend, report=rep)
            fout = _click_features(y, sr, np.round(fin[:, 0] * r), 20.0)
            found = np.isfinite(fout[:, 0]) & (fout[:, 1] > 10 ** (-12 / 20) * fin[:, 1])
            err = (fout[:, 0] - fin[:, 0] * r) / sr * 1000.0
            row["clicks_expected"] = int(len(fin))
            row["clicks_missing"] = int((~found).sum())
            row["click_timing_max_abs_ms"] = float(np.nanmax(np.abs(err[found]))) if found.any() else None
            row["click_timing_median_ms"] = float(np.nanmedian(err[found])) if found.any() else None
            row["click_width_ratio"] = float(np.nanmedian(fout[found, 2]) / np.nanmedian(fin[:, 2])) if found.any() else None
            row["click_level_db"] = float(20 * np.log10(np.median(fout[found, 1] / fin[found, 1]))) if found.any() else None
            row["clicks_duration_err_samples"] = int(len(y) - round(len(clicks) * r))
            row["bass_82_under_clicks_hz"] = _peak_hz(y, sr, 82.0)
        except DJRenderError as exc:
            row["error"] = str(exc)[:300]
        fails = []
        if "error" in row:
            fails.append("run_error")
        else:
            if abs(row["tone_82_err_hz"]) > c["pitch_82_tol_hz"]:
                fails.append("pitch_82")
            if abs(row["tone_440_err_hz"]) > c["pitch_440_tol_hz"]:
                fails.append("pitch_440")
            dur_tol = c["duration_tol_ms"] / 1000.0 * sr
            if max(abs(row["tone_82_duration_err_samples"]), abs(row["clicks_duration_err_samples"])) > dur_tol:
                fails.append("duration")
            if row["clicks_missing"]:
                fails.append("clicks_missing")
            if row["click_timing_max_abs_ms"] is None or row["click_timing_max_abs_ms"] > c["click_timing_tol_ms"]:
                fails.append("click_timing")
            if row["click_width_ratio"] is None or row["click_width_ratio"] > c["click_width_max_ratio"]:
                fails.append("click_smear")
            if row["click_level_db"] is None or row["click_level_db"] < c["click_level_min_db"]:
                fails.append("click_level")
            if row["tone_82_hf_residual_db"] > c["tone_hf_residual_max_db"]:
                fails.append("tone_hf_residual")
        row["failed"] = fails
        row["passed"] = not fails
        rows.append(row)
    failed = sorted({f for r in rows for f in r["failed"]})
    worst = {}
    for key in ("tone_82_err_hz", "tone_440_err_hz", "click_timing_max_abs_ms", "click_width_ratio", "click_level_db"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if vals:
            worst[key] = max(vals, key=abs) if key != "click_level_db" else min(vals)
    durs = [abs(r[k]) for r in rows for k in ("tone_82_duration_err_samples", "clicks_duration_err_samples") if k in r]
    if durs:
        worst["duration_err_samples"] = max(durs)
    return {
        "backend": {k: v for k, v in backend.items() if k != "qualification"},
        "sample_rate": sr,
        "contract": dict(c),
        "rows": rows,
        "passed": not failed,
        "summary": {"passed": not failed, "failed": failed, "time_ratios": [float(r) for r in time_ratios], "worst": worst,
                    "contract": "pitch 82 Hz +-0.5 Hz (test contract), 440 Hz +-2 Hz, duration <= 1 ms, clicks <= 2 ms, width x2, level -3 dB, HF residual <= -60 dB [H]"},
    }


def qualify_stretch_backend(backend: dict, sample_rate: int = DEFAULT_SAMPLE_RATE, time_ratios: Sequence[float] = STRETCH_QUICK_RATIOS) -> dict:
    """``measure_stretch_backend`` on the quick ratios, cached per process (binary, variant, sample rate)."""
    key = (backend.get("binary"), backend.get("variant", backend.get("backend")), int(sample_rate), tuple(float(r) for r in time_ratios))
    if key not in _QUALIFICATION_CACHE:
        _QUALIFICATION_CACHE[key] = measure_stretch_backend(backend, sample_rate, time_ratios)
    return _QUALIFICATION_CACHE[key]


# ---------------------------------------------------------------------- vocals


def vocal_activity(stem: np.ndarray, sr: int, hop_s: float = 0.02, rel_db: float = -24.0, abs_floor: float = 0.002) -> tuple[np.ndarray, np.ndarray, dict]:
    """Attività vocale da stem separato: RMS per frame di 20 ms sopra soglia.

    Soglia = max(``abs_floor``, p95 della traccia + ``rel_db``); lacune < 150 ms
    chiuse. Euristica acustica, non ASR: è evidenza di energia nello stem.
    """
    from scipy.ndimage import binary_closing

    mono = _as_stereo(stem).mean(axis=1)
    hop = max(1, int(round(hop_s * sr)))
    n = len(mono) // hop
    if n == 0:
        return np.zeros(0), np.zeros(0, dtype=bool), {"threshold_rms": None}
    rms = np.sqrt(np.mean(mono[: n * hop].reshape(n, hop) ** 2, axis=1))
    voiced = rms[rms > abs_floor]
    ref = float(np.percentile(voiced, 95)) if len(voiced) else float("inf")
    thr = max(abs_floor, ref * 10.0 ** (rel_db / 20.0))
    active = rms > thr
    close = max(1, int(round(0.15 / hop_s)))
    active = binary_closing(active, structure=np.ones(close, dtype=bool))
    times = (np.arange(n) + 0.5) * hop / sr
    return times, active, {"threshold_rms": thr, "p95_rms": ref if np.isfinite(ref) else None, "hop_s": hop_s, "rel_db": rel_db}


# ------------------------------------------------------------------ alignment


def _flux_env(x: np.ndarray, sr: int, t0: float, t1: float, hop: int, n_fft: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Spectral-flux onset envelope (log magnitude, hop ``hop``) of x[t0:t1] seconds."""
    from scipy.signal import stft

    lo = max(0, int(t0 * sr) - n_fft)
    hi = min(len(x), int(t1 * sr) + n_fft)
    mono = _as_stereo(x[lo:hi]).mean(axis=1)
    if len(mono) < 2 * n_fft:
        return np.zeros(1), np.array([t0])
    _, frames, spec = stft(mono, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    mag = np.log1p(100.0 * np.abs(spec))
    flux = np.maximum(0.0, np.diff(mag, axis=1)).sum(axis=0)
    times = lo / sr + frames[1:]
    return flux, times


def _xcorr_lag(ref: np.ndarray, env: np.ndarray, env_t: np.ndarray, grid: np.ndarray, max_lag: int, step_s: float) -> tuple[float, float]:
    """Lag (s) of ``env`` relative to ``ref`` sampled on ``grid``, parabolic peak."""
    ref_c = ref - ref.mean()
    nr = np.linalg.norm(ref_c)
    corr = np.zeros(2 * max_lag + 1)
    for i, lag in enumerate(range(-max_lag, max_lag + 1)):
        sc = np.interp(grid + lag * step_s, env_t, env)
        sc = sc - sc.mean()
        denom = nr * np.linalg.norm(sc)
        corr[i] = float(ref_c @ sc / denom) if denom > 0 else 0.0
    i = int(np.argmax(corr))
    frac = 0.0
    if 0 < i < len(corr) - 1:
        den = corr[i - 1] - 2 * corr[i] + corr[i + 1]
        frac = float(0.5 * (corr[i - 1] - corr[i + 1]) / den) if den < 0 else 0.0
    return (i - max_lag + frac) * step_s, float(corr[i])


def _measure_alignment(a_out: np.ndarray, b_src: np.ndarray, b_out: np.ndarray, sr: int, tmap: _PiecewiseMap, pairs: list[tuple[float, float]], window_s: float, max_lag_ms: float) -> list[dict]:
    """Misure di allineamento per ogni downbeat della sovrapposizione.

    * ``planned_error_ms``: posizione pianificata del downbeat di B (griglia di
      B attraverso la mappa) meno il downbeat di A (griglia di A);
    * ``stretcher_offset_ms``: ritardo, stimato per cross-correlazione di flusso
      spettrale, tra B stretchato e B sorgente riletto attraverso la mappa
      (fedeltà di Rubber Band alla mappa);
    * ``error_ms`` = pianificato + stretcher: errore rispetto alle griglie;
    * ``interdeck_lag_ms``: ritardo tra contenuto ritmico di B stretchato e di A
      attorno al downbeat (positivo = B in ritardo), misura senza griglie;
      affidabile solo con ``interdeck_xcorr`` alto e pattern ritmici simili.
    """
    step = 1.0 / 1000.0
    hop = max(1, sr // 1000)
    max_lag = int(round(max_lag_ms))
    results = []
    for tb, ta in pairs:
        grid = ta + np.arange(-window_s, window_s, step)
        if grid[0] < 0 or grid[-1] * sr > len(b_out):
            continue
        out_env, out_t = _flux_env(b_out, sr, grid[0] - 0.05, grid[-1] + 0.05, hop)
        src_grid = tmap.inv(grid)
        src_env, src_t = _flux_env(b_src, sr, float(src_grid[0]) - 0.06, float(src_grid[-1]) + 0.06, hop)
        a_env, a_t = _flux_env(a_out, sr, grid[0] - 0.05, grid[-1] + 0.05, hop)
        ref = np.interp(src_grid, src_t, src_env)
        lag_s, c = _xcorr_lag(ref, out_env, out_t, grid, max_lag, step)
        a_ref = np.interp(grid, a_t, a_env)
        ilag_s, ic = _xcorr_lag(a_ref, out_env, out_t, grid, max_lag, step)
        planned_ms = 1000.0 * (float(tmap.fwd(tb)) - ta)
        results.append({
            "output_s": ta,
            "b_source_s": tb,
            "planned_error_ms": planned_ms,
            "stretcher_offset_ms": 1000.0 * lag_s,
            "stretcher_xcorr": c,
            "error_ms": planned_ms + 1000.0 * lag_s,
            "interdeck_lag_ms": 1000.0 * ilag_s,
            "interdeck_xcorr": ic,
        })
    return results


# ----------------------------------------------------------------------- render


def render_transition(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    *,
    beats_a: Iterable[float],
    downbeats_a: Iterable[float],
    beats_b: Iterable[float],
    downbeats_b: Iterable[float],
    cue_out_a: float,
    cue_in_b: float,
    kind: str = "blend",
    bars: int = 8,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    vocals_a: Optional[np.ndarray] = None,
    vocals_b: Optional[np.ndarray] = None,
    config: Optional[DJRenderConfig] = None,
    provenance: Optional[dict] = None,
    return_decks: bool = False,
) -> DJRenderResult:
    """Rende la transizione A -> B e restituisce clip d'ascolto + metadati.

    Coordinate: i secondi in ingresso sono nel tempo di ciascuna sorgente;
    i metadati riportano campioni sia della clip in uscita sia delle sorgenti.
    """
    cfg = config or DJRenderConfig()
    sr = int(sample_rate)
    if kind not in TRANSITION_KINDS:
        raise DJRenderError(f"kind must be one of {TRANSITION_KINDS}")
    if cfg.cue_convention not in ("kim2020", "start"):
        raise DJRenderError("cue_convention must be 'kim2020' or 'start'")
    a = _as_stereo(np.asarray(audio_a, dtype=np.float64))
    b = _as_stereo(np.asarray(audio_b, dtype=np.float64))
    grid = {
        "beats_a": np.sort(np.asarray(list(beats_a), dtype=float)),
        "downbeats_a": np.sort(np.asarray(list(downbeats_a), dtype=float)),
        "beats_b": np.sort(np.asarray(list(beats_b), dtype=float)),
        "downbeats_b": np.sort(np.asarray(list(downbeats_b), dtype=float)),
    }
    va = _fit_len(_as_stereo(np.asarray(vocals_a, dtype=np.float64)), len(a)) if vocals_a is not None else None
    vb = _fit_len(_as_stereo(np.asarray(vocals_b, dtype=np.float64)), len(b)) if vocals_b is not None else None

    # Loudness: deck gain from integrated LUFS over the whole track.
    lufs_a = integrated_lufs(a, sr)
    lufs_b = integrated_lufs(b, sr)
    gain_a_db = (cfg.target_lufs - lufs_a) if np.isfinite(lufs_a) else 0.0
    gain_b_db = (cfg.target_lufs - lufs_b) if np.isfinite(lufs_b) else 0.0
    ga, gb = 10.0 ** (gain_a_db / 20.0), 10.0 ** (gain_b_db / 20.0)
    a, b = a * ga, b * gb
    va = va * ga if va is not None else None
    vb = vb * gb if vb is not None else None
    loudness = {
        "target_lufs": cfg.target_lufs,
        "a_integrated_lufs_source": lufs_a,
        "b_integrated_lufs_source": lufs_b,
        "a_gain_db": gain_a_db,
        "b_gain_db": gain_b_db,
        "a_integrated_lufs_normalized": integrated_lufs(a, sr),
        "b_integrated_lufs_normalized": integrated_lufs(b, sr),
        "scope": "whole track, ITU-R BS.1770-4 via pyloudnorm",
    }

    ia = _snap(grid["downbeats_a"], cue_out_a, cfg.snap_tolerance_s, "cue_out_a")
    jb = _snap(grid["downbeats_b"], cue_in_b, cfg.snap_tolerance_s, "cue_in_b")

    if kind == "cut":
        out, meta, decks = _render_cut(a, b, sr, grid, ia, jb, cfg)
    else:
        if bars not in BLEND_BARS:
            raise DJRenderError(f"blend bars must be one of {BLEND_BARS}")
        out, meta, decks = _render_blend(a, b, va, vb, sr, grid, ia, jb, int(bars), cfg)

    # Loudness jump across the transition (pre-limiter mix, 4 s each side).
    seg = int(4 * sr)
    ts, te = meta["overlap"]["output_start_sample"], meta["overlap"]["output_end_sample"]
    before = integrated_lufs(out[max(0, ts - seg) : ts], sr)
    after = integrated_lufs(out[te : te + seg], sr)
    loudness["lufs_4s_before_transition"] = before
    loudness["lufs_4s_after_transition"] = after
    loudness["loudness_jump_db"] = after - before if np.isfinite(before) and np.isfinite(after) else None
    loudness["section_gain_b_db"] = meta["section_gain"].get("applied_db", 0.0)

    limited, limiter = true_peak_limit(out, sr, cfg.ceiling_dbtp, cfg.limiter_hold_ms)
    loudness["output_integrated_lufs"] = integrated_lufs(limited, sr)
    loudness["output_true_peak_dbtp"] = limiter["true_peak_after_dbtp"]

    clip_len = len(limited)
    if kind == "cut":
        expected = int(round(cfg.pre_s * sr)) + int(round(cfg.post_s * sr))
        rule = "cut: 10 s + 10 s (AGENTS.md)"
    else:
        expected = meta["overlap"]["output_end_sample"] + int(round(cfg.post_s * sr))
        rule = "blend: 10 s before transition + whole transition + 10 s after (AGENTS.md, 20/10/2026)"
    meta.update({
        "schema_version": DJ_RENDER_VERSION,
        "kind": kind,
        "sample_rate": sr,
        "cue_convention": cfg.cue_convention if kind == "blend" else "cut: A stops / B starts on the same downbeat",
        "cue_out_a_s": float(grid["downbeats_a"][ia]),
        "cue_in_b_s": float(grid["downbeats_b"][jb]),
        "cue_out_a_requested_s": float(cue_out_a),
        "cue_in_b_requested_s": float(cue_in_b),
        "loudness": loudness,
        "limiter": limiter,
        "clip": {
            "rule": rule,
            "pre_s": cfg.pre_s,
            "post_s": cfg.post_s,
            "length_samples": clip_len,
            "duration_s": clip_len / sr,
            "matches_rule": clip_len == expected and cfg.pre_s == 10.0 and cfg.post_s == 10.0,
        },
        "config": asdict(cfg),
    })
    if provenance:
        meta["provenance"] = provenance
    result = DJRenderResult(limited.astype(np.float32), sr, _jsonable(meta))
    if return_decks:
        result.decks = decks
    return result


def _render_cut(a, b, sr, grid, ia, jb, cfg):
    if not 5.0 <= cfg.crossfade_ms <= 20.0:
        raise DJRenderError("cut crossfade must be 5..20 ms")
    pre = int(round(cfg.pre_s * sr))
    post = int(round(cfg.post_s * sr))
    sa = int(round(grid["downbeats_a"][ia] * sr))
    sb = int(round(grid["downbeats_b"][jb] * sr))
    d = int(round(cfg.crossfade_ms / 1000.0 * sr))
    n = pre + post
    deck_a = _slice(a, sa - pre, n)
    deck_b = _slice(b, sb - pre, n)
    t = np.arange(n)
    fade = np.clip((t - (pre - d)) / max(d, 1), 0.0, 1.0)
    g_b = np.sin(0.5 * np.pi * fade)
    g_a = np.cos(0.5 * np.pi * fade)
    deck_a *= g_a[:, None]
    deck_b *= g_b[:, None]
    bar_a, src_a = _bar_estimate(grid["downbeats_a"], grid["beats_a"], ia, -1, cfg.max_bar_deviation)
    bar_b, src_b = _bar_estimate(grid["downbeats_b"], grid["beats_b"], jb, +1, cfg.max_bar_deviation)
    b_ref = b[sb : sb + int(round(cfg.section_gain.b_bars * bar_b * sr))]
    sg_curve, sg_info = plan_section_gain(a, sa, bar_a, b_ref, pre, bar_b, n, sr, cfg.section_gain,
                                          transition_end_out=pre, b_ref_note="B source (no stretch in a cut) from cue_in_b")
    deck_b *= sg_curve[:, None]
    out = deck_a + deck_b
    da_, db_ = grid["downbeats_a"], grid["downbeats_b"]
    def span(d, i, bar, source):  # the same 4 bars that gave the bar estimate
        return (d[i] - 4.2 * bar, d[i]) if source.endswith("before") else (d[i], d[i] + 4.2 * bar)

    bpm_a = _local_bpm(grid["beats_a"], da_, *span(da_, ia, bar_a, src_a)) if "downbeats" in src_a else None
    bpm_b = _local_bpm(grid["beats_b"], db_, *span(db_, jb, bar_b, src_b)) if "downbeats" in src_b else None
    fold_k, fold_rate = _fold_no_limit(bar_a, bar_b)
    nxt_b = grid["downbeats_b"][jb + 1] - grid["downbeats_b"][jb] if jb + 1 < len(grid["downbeats_b"]) else None
    meta = {
        "bars": 0,
        "overlap": {
            "output_start_sample": pre - d,
            "output_end_sample": pre,
            "a_source_start_sample": sa - d,
            "a_source_end_sample": sa,
            "b_source_start_sample": sb - d,
            "b_source_end_sample": sb,
            "duration_s": d / sr,
            "crossfade": "equal-power sin/cos ending on the downbeat",
        },
        "switch": {"output_sample": pre, "a_source_sample": sa, "b_source_sample": sb},
        "transition_duration_s": d / sr,
        "stretch": {
            "applied": False,
            "tempo_ratio_b": 1.0,
            "note": "cut: B plays at its own tempo",
            "a_bar_s": bar_a,
            "b_bar_s": bar_b,
            "a_bar_source": src_a,
            "b_bar_source": src_b,
            "a_bpm_local": bpm_a,
            "b_bpm_local": bpm_b,
            "fold_k": fold_k,
            "tempo_jump_pct": 100.0 * (1.0 / fold_rate - 1.0),  # B tempo vs A (octave-folded)
        },
        "downbeat_alignment": {
            "switch_error_ms": 0.0,
            "note": "B's cue_in downbeat is placed on A's cue_out downbeat sample",
            "next_downbeat_drift_ms": (1000.0 * (nxt_b - bar_a)) if nxt_b is not None else None,
        },
        "vocals": {"status": "not applicable (no overlap)"},
        "section_gain": sg_info,
    }
    return out, meta, {"a": deck_a, "b": deck_b, "gains": {"section_gain_b": sg_curve}}


def _fold_no_limit(bar_a, bar_b):
    try:
        return choose_tempo_fold(bar_a, bar_b, max_stretch=10.0)
    except TempoIncompatibleError:  # pragma: no cover - unreachable with limit 10
        return 1.0, bar_b / bar_a


def _render_blend(a, b, va, vb, sr, grid, ia, jb, n_bars, cfg):
    da, db = grid["downbeats_a"], grid["downbeats_b"]
    _check_bars(da, ia, ia + n_bars, cfg.max_bar_deviation, "A overlap")
    bar_a = float(np.median(np.diff(da[ia : ia + n_bars + 1])))
    direction = -1 if cfg.cue_convention == "kim2020" else +1
    bar_b_est = _local_bar(db, jb, direction, n=max(8, n_bars))
    k, rate_est = choose_tempo_fold(bar_a, bar_b_est, cfg.max_stretch)
    kn = k * n_bars
    if abs(kn - round(kn)) > 1e-9:
        raise DJRenderError("half-time fold needs an even number of bars")
    kn = int(round(kn))
    jb0, jb1 = (jb - kn, jb) if cfg.cue_convention == "kim2020" else (jb, jb + kn)
    _check_bars(db, jb0, jb1, cfg.max_bar_deviation, "B overlap")

    pre = int(round(cfg.pre_s * sr))
    post = int(round(cfg.post_s * sr))
    sa0 = int(round(da[ia] * sr))
    off_a = sa0 - pre  # output sample n <-> A source sample n + off_a
    t_a = lambda idx: da[idx] - off_a / sr  # A downbeat in output seconds
    pairs = []
    for m in range(n_bars + 1):
        jm = k * m
        if abs(jm - round(jm)) < 1e-9:
            pairs.append((float(db[jb0 + int(round(jm))]), float(t_a(ia + m))))
    src = np.array([p[0] for p in pairs])
    dst = np.array([p[1] for p in pairs])
    seg_rates = np.diff(src) / np.diff(dst)  # B source seconds per output second
    rate_mean = float((src[-1] - src[0]) / (dst[-1] - dst[0]))
    if abs(rate_mean - 1.0) > cfg.max_stretch + 1e-9:
        raise TempoIncompatibleError(f"mean tempo ratio {rate_mean:.4f} outside +-{cfg.max_stretch:.0%}")
    if np.any(np.abs(seg_rates / rate_mean - 1.0) > cfg.max_bar_rate_deviation):
        raise DJRenderError(f"per-bar stretch too uneven: {np.round(seg_rates, 4).tolist()}")

    if cfg.stretch_mode not in ("fit", "grid"):
        raise DJRenderError("stretch_mode must be 'fit' or 'grid'")
    backend = stretch_backend_info(cfg.stretch_backend, sample_rate=sr)
    if backend["keyframes"] and cfg.stretch_mode == "grid":
        tmap = _PiecewiseMap(src, dst, 1.0 / seg_rates[0], 1.0 / rate_mean)
        mode = "grid: per-bar time map (each overlap downbeat of B -> downbeat of A)"
    else:
        slope, c = np.polyfit(src, dst, 1)
        slope, c = float(slope), float(c)
        rate_mean = 1.0 / slope
        if abs(rate_mean - 1.0) > cfg.max_stretch + 1e-9:
            raise TempoIncompatibleError(f"fitted tempo ratio {rate_mean:.4f} outside +-{cfg.max_stretch:.0%}")
        tmap = _PiecewiseMap([src[0], src[-1]], [c + slope * src[0], c + slope * src[-1]], slope, slope)
        mode = "fit: constant ratio, least-squares line through the overlap downbeat pairs (averages grid jitter)"

    t_s = pre
    t_sw = int(round(da[ia + n_bars // 2] * sr)) - off_a
    t_e = int(round(da[ia + n_bars] * sr)) - off_a
    n_out = t_e + post

    # B source segment (with stretcher pre-roll) and its placement in the output.
    src0 = max(0, int(round((db[jb0] - cfg.stretch_preroll_s) * sr)))
    src1_t = float(tmap.inv((n_out / sr) + 0.5))
    src1 = min(len(b), int(math.ceil(src1_t * sr)))
    if src1 - src0 < sr:
        raise DJRenderError("B too short for the requested transition")
    place0 = int(round(float(tmap.fwd(src0 / sr)) * sr))
    total_tgt = int(round(float(tmap.fwd(src1 / sr)) * sr)) - place0
    time_ratio = total_tgt / (src1 - src0)
    keyframes = None
    if backend["keyframes"]:
        keyframes = []
        for tb, _ in pairs:
            s_rel = int(round(tb * sr)) - src0
            t_rel = int(round(float(tmap.fwd(tb)) * sr)) - place0
            if 0 < s_rel < src1 - src0 and 0 < t_rel < total_tgt:
                keyframes.append((s_rel, t_rel))
    stretch_report: dict = {}
    stretched = _run_stretch(b[src0:src1], sr, time_ratio, keyframes, backend, report=stretch_report)
    deck_b_raw = np.zeros((n_out, 2))
    _place(deck_b_raw, stretched, place0)
    b_end_out = place0 + len(stretched)

    deck_a_raw = _slice(a, off_a, n_out)
    voc_a_raw = _slice(va, off_a, n_out) if va is not None else None

    # Gain curves (output samples).
    n = np.arange(n_out)
    v = np.clip((n - t_s) / max(t_sw - t_s, 1), 0.0, 1.0)
    w = np.clip((n - t_sw) / max(t_e - t_sw, 1), 0.0, 1.0)
    fader_b = np.where(n < t_s, 0.0, np.sin(0.5 * np.pi * v))
    fader_a = np.where(n >= t_e, 0.0, np.cos(0.5 * np.pi * w))
    swap = max(1, int(round(cfg.bass_swap_ms / 1000.0 * sr)))
    low_b = np.clip((n - (t_sw - swap)) / swap, 0.0, 1.0)
    low_a = 1.0 - low_b
    beat_out = bar_a / 4.0  # output beat (A's bar / 4: 4/4 assumed, as dj_select)
    eq_books: dict = {}
    if cfg.bass_gap_beats and cfg.bass_gap_beats > 0:
        g0 = t_sw - int(round(cfg.bass_gap_beats * beat_out * sr))
        if g0 <= t_s:
            raise DJRenderError("bass_gap_beats longer than the first half of the blend")
        low_a = np.where(n < g0 - swap, low_a, np.clip((g0 - n) / swap, 0.0, 1.0))  # A's bass leaves g0, B's slams in on the switch
        eq_books["bass_gap"] = {"beats": cfg.bass_gap_beats, "a_low_off_output_sample": int(g0), "b_low_on_output_sample": int(t_sw),
                                "source": "[P] B&B cap. 28 'Slam in the bass'; [A] Butler 2006 p. 246, pp. 242-243"}
    mid_b = np.ones(n_out)
    if cfg.mid_duck_b_bars and cfg.mid_duck_b_bars > 0:
        hold_end = min(t_s + int(round(cfg.mid_duck_b_bars * bar_a * sr)), t_sw)
        ramp_end = min(hold_end + int(round(bar_a * sr)), t_sw) if hold_end < t_sw else t_sw
        r = np.clip((n - hold_end) / max(ramp_end - hold_end, 1), 0.0, 1.0)
        mid_b = 10.0 ** ((cfg.mid_duck_b_db * (1.0 - r)) / 20.0)
        mid_b[n >= ramp_end] = 1.0
        eq_books["mid_duck_b"] = {"bars": cfg.mid_duck_b_bars, "db": cfg.mid_duck_b_db, "hold_until_output_sample": int(hold_end),
                                  "back_to_0db_output_sample": int(ramp_end),
                                  "source": "[A] Butler 2006 p. 53; [P] Webber 2008 p. 181; [P] B&B cap. 15"}
    floor = cfg.high_floor_db
    high_b = 10.0 ** ((floor * (1.0 - v)) / 20.0)
    high_a = 10.0 ** ((floor * w) / 20.0)

    # Vocals: clash measurement and optional ducking of A's vocal.
    vocals = {"status": "no stems provided"}
    duck = np.ones(n_out)
    if va is not None and vb is not None:
        hop_s = 0.02
        ta_times, act_a, info_a = vocal_activity(va, sr, hop_s, cfg.vocal_rel_db, cfg.vocal_abs_floor)
        tb_times, act_b, info_b = vocal_activity(vb, sr, hop_s, cfg.vocal_rel_db, cfg.vocal_abs_floor)
        frames_out = (np.arange(int(n_out / (hop_s * sr))) + 0.5) * hop_s
        act_a_out = np.interp(frames_out + off_a / sr, ta_times, act_a.astype(float), left=0, right=0) > 0.5
        act_b_out = np.interp(tmap.inv(frames_out), tb_times, act_b.astype(float), left=0, right=0) > 0.5
        act_b_out &= frames_out * sr < b_end_out
        idx = np.minimum((frames_out * sr).astype(int), n_out - 1)
        aud_a = fader_a[idx] >= cfg.audible_gain
        aud_b = fader_b[idx] >= cfg.audible_gain
        in_ov = (idx >= t_s) & (idx < t_e)
        clash = act_a_out & act_b_out & aud_a & aud_b & in_ov
        clash_s = float(clash.sum() * hop_s)
        ducked = False
        if cfg.duck_vocals and clash_s >= cfg.duck_min_overlap_s:
            from scipy.ndimage import binary_dilation

            mask = binary_dilation(act_b_out & aud_b & in_ov & act_a_out, structure=np.ones(int(0.4 / hop_s) + 1, dtype=bool))
            mask_s = np.interp(n / sr, frames_out, mask.astype(float))
            ramp = int(0.08 * sr)
            win = np.hanning(2 * ramp + 1)
            win /= win.sum()
            mask_s = np.clip(np.convolve(mask_s, win, mode="same"), 0.0, 1.0)
            g_floor = 10.0 ** (cfg.duck_gain_db / 20.0)
            duck = 1.0 - mask_s * (1.0 - g_floor)
            deck_a_raw = deck_a_raw - (1.0 - duck)[:, None] * voc_a_raw
            ducked = True
        residual = clash & (duck[idx] > cfg.audible_gain)
        vocals = {
            "status": "measured",
            "clash_seconds_before_ducking": clash_s,
            "ducking_applied": ducked,
            "ducked_seconds": float(np.sum(duck < 0.5) / sr),
            "duck_gain_db": cfg.duck_gain_db if ducked else None,
            "clash_seconds_after_ducking": float(residual.sum() * hop_s),
            "a_vocal_active_in_overlap_s": float((act_a_out & in_ov).sum() * hop_s),
            "b_vocal_active_in_overlap_s": float((act_b_out & in_ov).sum() * hop_s),
            "activity_a": info_a,
            "activity_b": info_b,
            "rule": "clash = both stems active and both faders >= audible_gain inside the overlap; duck A if clash >= duck_min_overlap_s (MosaikBox)",
        }
    elif va is not None or vb is not None:
        vocals = {"status": "only one stem provided: clash not measured"}

    # Section gain match: A before the overlap vs stretched B after the switch.
    b_ref = deck_b_raw[t_sw : min(n_out, t_sw + int(round(cfg.section_gain.b_bars * bar_a * sr)))]
    sg_curve, sg_info = plan_section_gain(a, sa0, bar_a, b_ref, t_sw, bar_a, n_out, sr, cfg.section_gain,
                                          transition_end_out=t_e, b_ref_note="stretched B (output bars = A bars) from the switch downbeat")

    low_a_s, mid_a_s, high_a_s = split_bands(deck_a_raw, sr, cfg.low_hz, cfg.high_hz)
    low_b_s, mid_b_s, high_b_s = split_bands(deck_b_raw, sr, cfg.low_hz, cfg.high_hz)
    deck_a = (low_a_s * low_a[:, None] + mid_a_s + high_a_s * high_a[:, None]) * fader_a[:, None]
    deck_b = (low_b_s * low_b[:, None] + mid_b_s * mid_b[:, None] + high_b_s * high_b[:, None]) * (fader_b * sg_curve)[:, None]
    out = deck_a + deck_b

    # Alignment: planned (grid) and realized (onset cross-correlation).
    b_src_seg = b[src0:src1]
    shift_map = _PiecewiseMap(tmap.src - src0 / sr, tmap.dst, tmap.sb, tmap.sa)
    measured = _measure_alignment(
        deck_a_raw, b_src_seg, deck_b_raw, sr, shift_map,
        [(tb - src0 / sr, ta) for tb, ta in pairs], cfg.align_window_s, cfg.align_max_lag_ms,
    )
    for row in measured:
        row["b_source_s"] += src0 / sr
        row["output_sample"] = int(round(row["output_s"] * sr))
    errs = np.array([abs(r["error_ms"]) for r in measured]) if measured else np.zeros(0)
    conf = [r for r in measured if r["interdeck_xcorr"] >= 0.5]
    jitter = {}
    for key, pts in (("a", dst), ("b", src)):
        idx = np.arange(len(pts))
        line = np.polyval(np.polyfit(idx, pts, 1), idx)
        jitter[f"{key}_max_abs_ms"] = float(1000.0 * np.max(np.abs(pts - line)))
    jitter["note"] = "overlap downbeats vs their own straight line: tracker jitter (Beat This! frames are 20 ms) or real tempo drift"
    ilags = np.array([abs(r["interdeck_lag_ms"]) for r in conf]) if conf else np.zeros(0)
    stretch_off = np.array([abs(r["stretcher_offset_ms"]) for r in measured if r["stretcher_xcorr"] >= 0.5])
    beat_err = _planned_beat_error(grid["beats_a"], grid["beats_b"], tmap, off_a / sr, t_s / sr, t_e / sr)

    bpm_a = _local_bpm(grid["beats_a"], da, da[ia], da[ia + n_bars])
    bpm_b = _local_bpm(grid["beats_b"], db, db[jb0], db[jb1])
    meta = {
        "bars": n_bars,
        "overlap": {
            "output_start_sample": t_s,
            "output_end_sample": t_e,
            "a_source_start_sample": t_s + off_a,
            "a_source_end_sample": t_e + off_a,
            "b_source_start_sample": int(round(db[jb0] * sr)),
            "b_source_end_sample": int(round(db[jb1] * sr)),
            "duration_s": (t_e - t_s) / sr,
            "a_downbeat_indices": [ia, ia + n_bars],
            "b_downbeat_indices": [jb0, jb1],
        },
        "switch": {
            "output_sample": t_sw,
            "a_source_sample": t_sw + off_a,
            "b_source_sample": int(round(float(tmap.inv(t_sw / sr)) * sr)),
            "rule": "bass swap ends on the middle downbeat of the overlap",
            "bass_swap_ms": cfg.bass_swap_ms,
        },
        "transition_duration_s": (t_e - t_s) / sr,
        "stretch": {
            "applied": True,
            "mode": mode,
            "backend": backend,
            "fold_k_b_bars_per_a_bar": k,
            "tempo_ratio_b": rate_mean,
            "time_ratio_b": 1.0 / rate_mean,
            "tempo_ratio_b_per_bar": seg_rates.tolist(),
            "stretch_pct": 100.0 * (rate_mean - 1.0),
            "rubberband_time_ratio_segment": time_ratio,
            "keyframes": len(keyframes) if keyframes else 0,
            "a_bar_s": bar_a,
            "b_bar_s": float(np.median(np.diff(db[jb0 : jb1 + 1]))),
            "a_bpm_local": bpm_a,
            "b_bpm_local": bpm_b,
            "b_bpm_effective": (bpm_b * rate_mean) if bpm_b else None,
            "after_overlap": "B stays stretched at the mean ratio until the end of the clip (no tempo ramp back)",
            "b_source_segment_samples": [src0, src1],
            "b_output_end_sample": int(min(b_end_out, n_out)),
            "pitch": "preserved (time-stretch only, no pitch shift; backend qualified on 82/440 Hz tones, see backend.qualification)",
            "start_delay_compensation": stretch_report or None,
            # additive (21/10/2026, P0.6): the time map actually used, so that QA projects B (stems, words) through the
            # piecewise map instead of a straight line through the switch. B source seconds -> clip seconds; linear
            # extrapolation with slope_before/slope_after (clip seconds per B second) outside the nodes.
            "time_map": {
                "kind": "piecewise" if len(tmap.src) > 2 else "linear",
                "b_source_s": tmap.src.tolist(), "output_s": tmap.dst.tolist(),
                "slope_before": tmap.sb, "slope_after": tmap.sa,
                "realised_by": "rubberband keyframes (-M map) + linear segments to the segment ends" if keyframes else "constant time ratio",
                "units": "B source seconds -> clip output seconds",
            },
        },
        "downbeat_alignment": {
            "max_abs_error_ms": float(errs.max()) if len(errs) else None,
            "median_abs_error_ms": float(np.median(errs)) if len(errs) else None,
            "within_10ms": bool(len(errs) and errs.max() <= 10.0),
            "planned_max_abs_error_ms": float(max(abs(r["planned_error_ms"]) for r in measured)) if measured else None,
            "stretcher_max_abs_offset_ms": float(stretch_off.max()) if len(stretch_off) else None,
            "interdeck_median_abs_lag_ms": float(np.median(ilags)) if len(ilags) else None,
            "interdeck_max_abs_lag_ms": float(ilags.max()) if len(ilags) else None,
            "interdeck_confident_downbeats": len(conf),
            "per_downbeat": measured,
            "method": "error = planned grid residual + stretcher offset (spectral-flux cross-correlation of stretched B vs source B through the map); interdeck = flux cross-correlation of stretched B vs A around each A downbeat (no grids), stats over xcorr >= 0.5",
            "planned_beat_phase_error_ms": beat_err,
            "grid_linearity": jitter,
        },
        "eq": {
            "bands_hz": [cfg.low_hz, cfg.high_hz],
            "filter": "zero-phase Butterworth-2 forward/backward (LR4 magnitude), complementary bands, exact reconstruction",
            "high_floor_db": cfg.high_floor_db,
            "fader_a": "1 until switch, quarter-cosine to 0 at overlap end",
            "fader_b": "quarter-sine 0->1 from overlap start to switch, then 1",
            "low": "A full until switch, B killed until switch; complementary swap ending on switch downbeat",
            "high": "B from high_floor_db to 0 dB (dB-linear) until switch; A 0 dB -> high_floor_db after switch",
            "basis": "monotone fader + 3-band EQ (Kim, Yang, Nam DAFx 2022); shapes are design hypotheses after Vande Veire 2018 §4.3 and MosaikBox §3.4",
            "books": eq_books or None,
        },
        "vocals": vocals,
        "section_gain": sg_info,
    }
    decks = {
        "a": deck_a, "b": deck_b, "b_stretched_raw": deck_b_raw,
        # additive (21/10/2026): A before EQ and fader (after the optional vocal duck). dj_qa ignores it unless
        # QAConfig(rhythm_on_raw_a=True): the marker keeps the default QA identical (rhythm measured on EQ'd A).
        "a_raw": deck_a_raw, "a_raw_opt_in": True,
        "gains": {"fader_a": fader_a, "fader_b": fader_b, "low_a": low_a, "low_b": low_b, "high_a": high_a, "high_b": high_b, "mid_b": mid_b,
                  "duck_a_vocal": duck,
                  "section_gain_b": sg_curve},
    }
    return out, meta, decks


def _planned_beat_error(beats_a, beats_b, tmap, off_a_s, t0, t1) -> Optional[dict]:
    ba = beats_a - off_a_s
    ba = ba[(ba >= t0) & (ba <= t1)]
    if len(ba) < 2 or len(beats_b) == 0:
        return None
    mapped = tmap.fwd(beats_b)
    mapped = mapped[(mapped >= t0 - 0.05) & (mapped <= t1 + 0.05)]
    if len(mapped) == 0:
        return None
    errs = np.array([1000.0 * (m - ba[np.argmin(np.abs(ba - m))]) for m in mapped])
    period = float(np.median(np.diff(ba)))
    errs = errs[np.abs(errs) < 500.0 * period]  # ignore B beats with no A partner
    if len(errs) == 0:
        return None
    return {
        "median_abs_ms": float(np.median(np.abs(errs))),
        "p95_abs_ms": float(np.percentile(np.abs(errs), 95)),
        "n_beats": int(len(errs)),
        "note": "Beat This! beats of B mapped through the time map vs nearest A beat; reflects grid jitter/swing, not the stretcher",
    }


# ------------------------------------------------------------ B lead-in (21/10/2026)
B_LEAD_IN_VERSION = "musicmix.dj_render.b_lead_in.1"  # conoscenza_mix/_meta/INGRESSO_B_2026-10-21.md
B_LEAD_IN_MODES = ("stem", "full")
B_LEAD_IN_KINDS = ("echo_out", "filter_out", "build_into_chorus", "halving_build", "break_before_entry")


B_LEAD_IN_A_VOICE_REL_DB = -24.0  # [H] come dj_qa.VOCAL_REL_DB: voce di A attiva se sopra p95 - 24 dB del suo stem nella clip
B_LEAD_IN_ECHO_DUCK_DB = -6.0  # [H] modo stem: coda d'eco di A abbassata di 6 dB sotto la voce di B fino al downbeat di B
B_LEAD_IN_A_VOICE_SHARE = 0.10  # [H] A si chiude (modo 'auto') se la sua voce è attiva in più del 10 % del lead-in


def add_b_lead_in(res: DJRenderResult, audio_b: np.ndarray, lead_in_samples: int, *, vocals_b: Optional[np.ndarray] = None,
                  mode: str = "stem", a_close: str = "auto", vocals_a_clip: Optional[np.ndarray] = None,
                  info: Optional[dict] = None) -> DJRenderResult:
    """B udibile ``lead_in_samples`` prima del suo downbeat d'ingresso (pickup / prima frase intera) in un render a ingresso pulito.

    Per le tecniche in cui B entra pulito sul suo downbeat senza stretch (:data:`B_LEAD_IN_KINDS`: echo/filter out di
    :mod:`musicmix.dj_effects`, build/halving/break di :mod:`musicmix.dj_creative`). Il mix prima del limitatore è
    ricostruito dai deck del render (``return_decks=True``) e verificato: limitato di nuovo deve dare lo stesso PCM
    (altrimenti ``DJRenderError``). Poi, sullo stesso orologio (B sorgente = campione d'ingresso − anticipo):

    * ``mode='stem'`` [H, come ``dj_phrase.render_phrase_transition``]: nel lead-in suona solo la voce di B (stem Demucs,
      stesso guadagno di B), che sul downbeat passa a B intero (guadagni complementari, voce continua); ``'full'``: B
      intero dal lead-in;
    * ``a_close`` [H]: ``'cut'`` = A (deck secco/filtrato) si chiude con la micro-dissolvenza del render alla partenza del
      lead-in se suona ancora (il campione in cui finisce il secco di A va in ``effect.dry_a_cut_output_sample`` per il
      QA); ``'auto'`` (default) = solo se la voce di A (``vocals_a_clip``, coordinate della clip, o
      ``decks['vocals_a_heard']``) è attiva in più del 10 % del lead-in mentre A suona ancora: le due voci non si
      sovrappongono, altrimenti A resta come base (niente buco); senza voce di A disponibile vale ``'cut'``;
      ``'keep'`` = A invariato. La coda d'eco, dalla partenza del lead-in: modo ``stem`` -6 dB (resta la base fino al
      downbeat di B, dove vale la regola del render); modo ``full`` livello "sotto B" del render (o si ferma).

    ``meta['b_entry']`` dichiara il punto in cui B diventa udibile (campione della clip e di B) per il QA
    (``dj_qa.check_b_entry``). ``switch.output_sample`` resta sul downbeat (regola delle clip invariata).
    """
    if mode not in B_LEAD_IN_MODES:
        raise DJRenderError(f"mode must be one of {B_LEAD_IN_MODES}")
    if a_close not in ("cut", "keep", "auto"):
        raise DJRenderError("a_close must be 'cut', 'keep' or 'auto'")
    meta = dict(res.metadata)
    kind = str(meta.get("kind"))
    if kind not in B_LEAD_IN_KINDS:
        raise DJRenderError(f"b lead-in: technique {kind!r} not supported (clean entries only: {B_LEAD_IN_KINDS})")
    if (meta.get("stretch") or {}).get("applied"):
        raise DJRenderError("b lead-in: B is stretched (only clean entries at B's own tempo)")
    decks = dict(res.decks or {})
    comps = [k for k in ("a", "echo", "b", "overlay") if k in decks]
    if "a" not in decks or "b" not in decks:
        raise DJRenderError("b lead-in needs the render's decks (return_decks=True)")
    sr = int(res.sample_rate)
    n = len(res.audio)
    cfg = meta.get("config") or {}
    ceiling, hold = float(cfg.get("ceiling_dbtp", -1.0)), float(cfg.get("limiter_hold_ms", 20.0))
    parts = {k: _fit_len(_as_stereo(decks[k]), n) for k in comps}
    mix0 = sum(parts.values())
    lim0, _ = true_peak_limit(mix0, sr, ceiling, hold)
    err = float(np.max(np.abs(lim0.astype(np.float32) - res.audio))) if n else 0.0
    if err > 1e-5:
        raise DJRenderError(f"b lead-in: the decks do not rebuild the render (max error {err:.2e})")
    L = int(lead_in_samples)
    if L <= 0:
        return res
    entry = int(meta["switch"]["output_sample"])
    sb = int(meta["switch"]["b_source_sample"])
    start = entry - L
    t0 = int(meta["overlap"]["output_start_sample"])
    xf = int(round(float(cfg.get("crossfade_ms", 10.0)) / 1000.0 * sr))
    if start < t0 + xf:
        raise DJRenderError(f"b lead-in of {L / sr:.2f} s is longer than the transition region ({(entry - t0) / sr:.2f} s)")
    gb = 10 ** (float(meta["loudness"]["b_gain_db"]) / 20.0)
    src = audio_b if mode == "full" else vocals_b
    if src is None:
        raise DJRenderError("b lead-in mode 'stem' needs B's vocal stem")
    b_full = _slice(_as_stereo(audio_b), sb - entry, n) * gb
    # section gain of B at its entry (dj_render/dj_effects curve, else measured on the rendered deck after the micro fade)
    sg = (decks.get("gains") or {}).get("section_gain_b")
    if sg is not None and len(sg) > entry:
        g_sg = float(np.asarray(sg)[entry])
    else:
        w0, w1 = entry + xf, min(n, entry + sr)
        ref = float(np.sqrt(np.mean(b_full[w0:w1] ** 2))) if w1 > w0 else 0.0
        g_sg = float(np.sqrt(np.mean(parts["b"][w0:w1] ** 2))) / ref if ref > 0 else 1.0
    x = _slice(_as_stereo(src), sb - entry, n) * gb * g_sg
    t = np.arange(n)
    g_in = np.sin(0.5 * np.pi * np.clip((t - (start - xf)) / max(xf, 1), 0.0, 1.0))
    g_entry = np.sin(0.5 * np.pi * np.clip((t - (entry - xf)) / max(xf, 1), 0.0, 1.0))  # the render's own fade-in of B
    g_lead = g_in * (1.0 - g_entry)
    g_lead[t >= entry] = 0.0
    lead = x * g_lead[:, None]
    parts["b"] = parts["b"] + lead
    closed = None
    echo_note = None
    a_voice = None
    env = np.abs(parts["a"]).max(axis=1)
    a_alive = np.flatnonzero(env[start:entry] > 1e-6)
    close = a_close == "cut"
    if a_close == "auto":
        va = vocals_a_clip if vocals_a_clip is not None else decks.get("vocals_a_heard")
        if va is None:
            close, a_voice = True, {"decision": "cut", "why": "A's vocal not available: closed (conservative)"}
        elif not a_alive.size:
            a_voice = {"decision": "keep", "why": "A is already silent in the lead-in"}
        else:
            v = _fit_len(_as_stereo(va), n).mean(axis=1)
            hop = max(1, int(0.02 * sr))
            nf = len(v) // hop
            rms = np.sqrt(np.mean(v[: nf * hop].reshape(nf, hop) ** 2, axis=1)) if nf else np.zeros(0)
            voiced = rms[rms > 1e-6]
            thr = float(np.percentile(voiced, 95)) * 10 ** (B_LEAD_IN_A_VOICE_REL_DB / 20.0) if voiced.size else np.inf
            f0, f1 = start // hop, (start + int(a_alive[-1]) + 1) // hop
            share = float(np.mean(rms[f0:max(f0 + 1, f1)] > thr)) if f1 > f0 else 0.0
            close = share > B_LEAD_IN_A_VOICE_SHARE
            a_voice = {"decision": "cut" if close else "keep", "a_voice_share_in_lead_in": share, "threshold_share": B_LEAD_IN_A_VOICE_SHARE,
                       "why": "A's voice active under B's lead-in" if close else "A's voice silent under B's lead-in: A stays as the bed"}
    if close:
        if a_alive.size:
            g_a = np.cos(0.5 * np.pi * np.clip((t - (start - xf)) / max(xf, 1), 0.0, 1.0))
            g_a[t >= start] = 0.0
            parts["a"] = parts["a"] * g_a[:, None]
            closed = start
    if a_close != "keep":
        if "echo" in parts:
            eff = meta.get("effect") or {}
            if mode == "stem":  # B's beat still lands on its downbeat: the tail stays the bed, ducked under B's voice
                lvl = 10 ** (B_LEAD_IN_ECHO_DUCK_DB / 20.0)
            else:
                lvl = 0.0 if eff.get("tail_stopped_on_b") or eff.get("tail_under_b_db") is None else 10 ** (float(eff["tail_under_b_db"]) / 20.0)
            ramp = max(1, int(0.02 * sr))
            under = np.clip((t - (start - ramp)) / ramp, 0.0, 1.0)
            g_e = 1.0 - under * (1.0 - lvl)
            g_e[t >= entry] = 1.0  # after B's downbeat the render's own tail gain already applies
            parts["echo"] = parts["echo"] * g_e[:, None]
            echo_note = f"echo tail at {20 * math.log10(lvl) if lvl > 0 else -np.inf:.0f} dB from the lead-in start (render's level under B)"
    mix = sum(parts.values())
    limited, limiter = true_peak_limit(mix, sr, ceiling, hold)
    loud = dict(meta["loudness"])
    loud["output_integrated_lufs"] = integrated_lufs(limited, sr)
    loud["output_true_peak_dbtp"] = limiter["true_peak_after_dbtp"]
    meta["loudness"] = loud
    meta["limiter"] = limiter
    if closed is not None and kind in ("echo_out", "filter_out"):
        eff = dict(meta.get("effect") or {})
        old = eff.get("dry_a_cut_output_sample")
        eff["dry_a_cut_output_sample"] = int(min(int(old), closed)) if old is not None else int(closed)
        if old is not None:
            eff["dry_a_cut_output_sample_before_lead_in"] = int(old)
        meta["effect"] = eff
    meta["switch"] = dict(meta["switch"], b_audible_output_sample=int(start), downbeat_output_sample=entry)
    meta["b_entry"] = {
        "version": B_LEAD_IN_VERSION, "lead_in_samples": L, "lead_in_s": L / sr, "audible_output_sample": int(start),
        "audible_source_sample": int(sb - L), "downbeat_output_sample": entry, "downbeat_source_sample": sb, "mode": mode,
        "a_close": a_close, "a_closed_output_sample": closed, "a_voice": a_voice, "echo": echo_note, "section_gain_b": g_sg,
        "rebuild_max_error": err, "info": info,
        "note": ("B's vocal stem alone from the lead-in start to its downbeat, then full B (complementary gains) [H]"
                 if mode == "stem" else "full B from the lead-in start [H]"),
    }
    if kind in ("build_into_chorus", "halving_build", "break_before_entry"):
        bm = list(meta.get("b_map") or [])
        bm.insert(0, {"out_start": int(start), "out_end": entry, "src_start": int(sb - L), "rate": 1.0, "vocals": True,
                      "what": f"B lead-in ({mode}) before its downbeat"})
        meta["b_map"] = bm
    gains = dict(decks.get("gains") or {})
    if "fader_b" in gains and len(gains["fader_b"]) == n and kind in ("echo_out", "filter_out"):
        gains["fader_b"] = np.maximum(np.asarray(gains["fader_b"], float), np.where((t >= start) & (t < entry), g_in, 0.0))
    if "fader_a" in gains and len(gains["fader_a"]) == n and closed is not None and kind in ("echo_out", "filter_out"):
        gains["fader_a"] = np.where(t >= closed, 0.0, np.asarray(gains["fader_a"], float))
    new_decks = dict(decks, **{k: v for k, v in parts.items()}, gains=gains)
    if "vocals_b_heard" in decks and mode == "stem":
        new_decks["vocals_b_heard"] = _fit_len(_as_stereo(decks["vocals_b_heard"]), n) + lead
    elif "vocals_b_heard" in decks and vocals_b is not None:
        new_decks["vocals_b_heard"] = _fit_len(_as_stereo(decks["vocals_b_heard"]), n) + \
            _slice(_as_stereo(vocals_b), sb - entry, n) * (gb * g_sg) * g_lead[:, None]
    if "vocals_a_heard" in decks and closed is not None:
        va = _fit_len(_as_stereo(decks["vocals_a_heard"]), n)
        va[t >= closed] = 0.0
        new_decks["vocals_a_heard"] = va
    return DJRenderResult(limited.astype(np.float32), sr, _jsonable(meta), new_decks)


# --------------------------------------------------------------------- helpers


def _as_stereo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    return x


def _fit_len(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.vstack([x, np.zeros((n - len(x), x.shape[1]))])


def _slice(x: np.ndarray, start: int, n: int) -> np.ndarray:
    out = np.zeros((n, x.shape[1]))
    lo, hi = max(0, start), min(len(x), start + n)
    if hi > lo:
        out[lo - start : hi - start] = x[lo:hi]
    return out


def _place(dest: np.ndarray, src: np.ndarray, at: int) -> None:
    lo, hi = max(0, at), min(len(dest), at + len(src))
    if hi > lo:
        dest[lo:hi] += src[lo - at : hi - at]


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj
