"""Mashup e transizioni che usano la musica dei due brani: voce di A sulla base di B, hook di B
sopra A, loop e chop della voce, botta e risposta, beat swap, freeze che risolve nella tonalità
di B, swell inverso della prima parola di B; più la *mashability* con pitch shift che decide la
trasposizione.

Nota della biblioteca: ``conoscenza_mix/_meta/MASHUP_CREATIVI_2026-10-21.md`` (fonti, parametri,
etichette [E]/[S]/[P]/[T]/[H], limiti). Complemento di :mod:`musicmix.dj_creative` (non modificato):
qui si riusano le sue convenzioni e i suoi helper (``_prepare``, ``_finish``, mappe ``a_map``/``b_map``,
``vocals_*_heard``, ``trim_overlay``, ``spectral_freeze``, ``reverb_ir``, ``synth_riser``), così i
render passano per ``dj_qa.evaluate_render`` con ``dj_creative.qa_inputs`` come quelli creativi.

Idea comune (scelta di progetto [H], dichiarata): **B suona com'è** (tempo e tonalità originali,
entra pulito sul suo downbeat); a essere stirato e trasposto è solo il *materiale preso in prestito*
(frammenti dello stem vocale di A, o l'hook di B quando lo si anticipa sopra A). Così una
trasposizione non resta addosso al brano che continua (diverso da ``dj_creative.key_shift_b``) e il
pitch shift serve solo dove due musiche suonano insieme.

* **Mashability** (:func:`harmonic_match`): compatibilità armonica tra cromagrammi sincronizzati ai
  battiti (mediana per battito come AutoMashUpper, Davies et al. 2014 [S]) calcolata nello spazio TIV
  (Bernardes et al. 2016; codice ufficiale TIV.lib, MIT, vendorizzato in
  ``musicmix/_vendor/tivlib``: qui una copia vettoriale di ``TIV.from_pcp`` e
  ``small_scale_compatibility``, verificata contro l'originale nei test) per trasposizioni
  -3..+3 semitoni (limite di pipeline di Huang et al. 2021 [S]); la similarità coseno di AutoMashUpper
  è riportata accanto. Il materiale cantato viene trasposto solo se la compatibilità migliora di
  ``shift_min_gain`` rispetto a 0 [H]: su un rap parlato (cromagramma piatto) le trasposizioni si
  equivalgono e lo shift resta 0.
* **Tempo** (:func:`fold_ratio`): il materiale preso in prestito va al tempo di chi lo ospita con
  piegatura d'ottava (1/2, 1, 2 battute per battuta) e stretch entro ``max_vocal_stretch`` (12 % [H];
  dj_render usa 8 % per i mix interi; Huang 2021 0,8-1,2 [S]). Stretch e trasposizione in un'unica
  chiamata Rubber Band R3 offline con formanti preservate (``-3 -t -p -F``, :func:`stretch_pitch`).
* **Parole intere**: A smette di cantare solo a un fine verso (pausa ASR >= ``phrase_gap_s`` dopo
  l'ultima parola e stem silenzioso subito dopo, :func:`a_exit_point`); i frammenti di A partono da un
  inizio verso; la voce di B si apre solo all'inizio di una sua frase, pickup compreso
  (:func:`b_vocal_gate`), o resta chiusa: B entra su materiale strumentale (mix - stem). Una sola voce
  alla volta (Davies 2014 [E]; Brungart 2001 [E]).

Tecniche (tutte con clip = 10 s prima dell'inizio della transizione + transizione + 10 s dopo):

``acapella_mashup``  verso di A (stem, da un inizio verso) sopra lo strumentale di B per 2-8 battute
                     (B&B, *How to DJ Right*: "Play an acappella of one track over a dub of another" [P]);
``beat_swap``        lo stesso verso di A sopra la sola batteria di B (stem Demucs), poi basso e armonia
                     di B sul downbeat (B&B: con brani che stonano "mix them when one is just percussion" [P]);
``hook_teaser``      la prima frase del ritornello di B, stirata e trasposta, sopra lo strumentale di A
                     nelle ultime battute; poi B entra dal suo cue nella sua tonalità [P][H];
``vocal_loop``       l'ultima parola di A in loop sulla griglia di B (battuta, 1/2, 1 battito o loop roll
                     che si dimezza), intonata a B, passa-alto crescente, coda d'eco o riverbero
                     (Crossfader, *12 high energy DJ transitions*: loop di 1 battito su una parola,
                     dimezzato fino allo stutter [P]);
``vocal_chop``       le ultime parole di A tagliate e risuonate sul ritmo 3+3+2 della griglia di B
                     (Butler 2006 pdf pp. 97-99: 3+3+2 è la distribuzione massimamente uniforme di 3
                     attacchi su 8 [T]; Williams 2013 pdf p. 161: voce sminuzzata e ripetuta [T]);
``call_response``    frasi alternate: verso di A (trasposto) e frase dell'hook di B sopra la base di B
                     (Crossfader, "Wordplay": due brani che si scambiano le righe [P]; Katz 2012 pdf p. 97);
``pivot_freeze``     freeze spettrale dell'ultimo battito di A che scivola (glissando) fino alla tonica
                     di B (o al suo relativo), poi B pulito: un accordo perno che risolve [T][H];
``reverse_swell``    riverbero inverso della prima parola di B: cresce sopra l'ultima battuta di A e
                     sfocia nella parola vera di B [P][H].

Non si scelgono coppie né cue (lo fa il chiamante, es. ``tools/run_dj_mashup_v1.py``). QA e critico
verificano l'integrità tecnica, non il gusto (AGENTS.md).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import os
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from musicmix import dj_creative as cr
from musicmix import dj_effects as fx
from musicmix import dj_render as dj
from musicmix.dj_render import DJRenderError, DJRenderResult

DJ_MASHUP_VERSION = "musicmix.dj_mashup.1"
TECHNIQUES = ("acapella_mashup", "beat_swap", "hook_teaser", "vocal_loop", "vocal_chop", "call_response",
              "pivot_freeze", "reverse_swell")
SHIFT_RANGE = (0, 1, -1, 2, -2, 3, -3)  # Huang et al. 2021 [S]: harmonic stems within +-3 semitones
TIV_WEIGHTS_AUDIO = np.array([3.0, 8.0, 11.5, 15.0, 14.5, 7.5])  # TIV.lib weight_sets['audio'] (Bernardes 2016)
LOOP_MODES = ("bar", "half", "beat", "halving")
CHOP_PATTERNS = {"332": ((0, 3), (3, 3), (6, 2)), "3332": ((0, 3), (3, 3), (6, 3), (9, 3), (12, 4))}  # (onset, length) in 8ths / 16ths
CHOP_UNITS = {"332": 8, "3332": 16}  # pulses per bar


@dataclass(frozen=True)
class MashupConfig:
    creative: cr.DJCreativeConfig = field(default_factory=cr.DJCreativeConfig)
    max_vocal_stretch: float = 0.12  # [H] borrowed fragments only; dj_render keeps 8 % for whole mixes
    phrase_gap_s: float = 0.25  # [H] pause between ASR words that marks a line boundary
    word_pad_s: float = 0.03  # kept after the last word before a mute (ASR end tolerance)
    gate_lead_s: float = 0.04  # B's vocal gate opens this long before the first word of its phrase
    stem_guard_s: float = 0.10  # after a mute point the vocal stem must be (almost) silent this long
    stem_guard_cover: float = 0.15  # [H] max active fraction of the stem in the guard window
    shift_min_gain: float = 0.01  # [H] transpose only if TIV compatibility improves by this much over shift 0
    shift_penalty: float = 0.002  # [H] per semitone (ties -> smaller shift: formant/phasing artefacts grow with |s|)
    soft_line_end_mute_ms: float = 120.0  # [H] fade of A's stem at a line end whose stem is not silent (melodic vocals)
    key_guard: bool = True  # [user] 21/10/2026: no discordant key -> only shifts that make the estimated keys compatible, if any
    mid_cut_db: float = -3.0  # B instrumental mids under a borrowed vocal (dj_creative.acapella_over) [P]
    fade_ms: float = 12.0  # fades of every borrowed fragment
    under_b_db: float = -12.0  # tails under B (dj_creative.tail_under_b_db)


# ------------------------------------------------------------------ small helpers
_as_stereo, _fit, _slice, _rise, _db, _rms = cr._as_stereo, cr._fit, cr._slice, cr._rise, cr._db, cr._rms


def _place(dest: np.ndarray, y: np.ndarray, at: int) -> tuple[int, int]:
    """Somma ``y`` in ``dest`` dal campione ``at`` (taglia ai bordi); restituisce l'intervallo scritto."""
    lo, hi = max(0, at), min(len(dest), at + len(y))
    if hi > lo:
        dest[lo:hi] += y[lo - at : hi - at]
    return lo, hi


def _windowed(x: np.ndarray, fade: int) -> np.ndarray:
    y = _as_stereo(x).copy()
    f = min(fade, len(y) // 2)
    if f > 0:
        r = np.sin(0.5 * np.pi * (np.arange(f) + 0.5) / f)[:, None]
        y[:f] *= r
        y[-f:] *= r[::-1]
    return y


def _hpf(x: np.ndarray, sr: int, hz: float, order: int = 2) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    if len(x) < 64:
        return x
    return sosfiltfilt(butter(order, hz, btype="high", fs=sr, output="sos"), x, axis=0, padlen=min(len(x) - 1, 3 * 64))


# ----------------------------------------------------------------- tempo / pitch
def fold_ratio(bar_src_s: float, bar_dst_s: float, max_stretch: float = 0.12) -> tuple[float, float]:
    """``(k, time_ratio)``: una battuta della sorgente diventa ``k`` battute (1, 1/2, 2) del brano che la
    ospita; ``time_ratio`` = durata in uscita / durata in ingresso. Errore se |ratio - 1| > ``max_stretch``."""
    best = None
    for k in (1.0, 0.5, 2.0):
        r = k * bar_dst_s / bar_src_s
        if best is None or abs(math.log(r)) < abs(math.log(best[1])) - 1e-12:
            best = (k, r)
    if abs(best[1] - 1.0) > max_stretch + 1e-9:
        raise DJRenderError(f"borrowed material would need {100 * (best[1] - 1):+.1f} % stretch (fold k={best[0]:g}), "
                            f"outside +-{100 * max_stretch:.0f} %")
    return best


_RB_QUAL: dict = {}


def rubberband_info() -> dict:
    """Backend Rubber Band CLI qualificato da ``dj_render.stretch_backend_info`` (contratto di pitch, durata e
    transienti, misurato una volta per processo)."""
    if "info" not in _RB_QUAL:
        info = dj.stretch_backend_info("rubberband")
        _RB_QUAL["info"] = {k: info.get(k) for k in ("backend", "version", "engine", "qualification")}
    return _RB_QUAL["info"]


def stretch_pitch(x: np.ndarray, sr: int, time_ratio: float = 1.0, semitones: float = 0.0, formant: bool = True,
                  qualify: bool = True) -> tuple[np.ndarray, dict]:
    """Stretch (``time_ratio`` = durata uscita/ingresso) e trasposizione in una chiamata Rubber Band R3 offline
    (``-3 --centre-focus -t -p`` e ``-F`` per le formanti, voci). Lunghezza = round(len * time_ratio)."""
    import soundfile as sf

    x = _as_stereo(x)
    n_out = int(round(len(x) * time_ratio))
    if abs(time_ratio - 1.0) < 1e-9 and semitones == 0:
        return x.copy(), {"applied": False, "time_ratio": 1.0, "semitones": 0}
    if not np.any(x):
        return np.zeros((n_out, 2)), {"applied": False, "time_ratio": time_ratio, "semitones": semitones, "note": "silent input"}
    cli = shutil.which("rubberband")
    if not cli:
        raise DJRenderError("stretch_pitch needs the rubberband CLI (Rubber Band R3)")
    with tempfile.TemporaryDirectory(prefix="dj_mashup_") as tmp:
        src, dst = os.path.join(tmp, "in.wav"), os.path.join(tmp, "out.wav")
        sf.write(src, x.astype(np.float32), sr, subtype="FLOAT")
        cmd = [cli, "-q", "-3", "--centre-focus", "-t", f"{time_ratio:.9f}", "-p", f"{float(semitones):.6f}"]
        cmd += (["-F"] if formant and semitones else []) + [src, dst]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise DJRenderError(f"rubberband failed: {proc.stderr.strip()[:300]}")
        y, sr2 = sf.read(dst, dtype="float64", always_2d=True)
    if sr2 != sr:
        raise DJRenderError("rubberband changed the sample rate")
    info = {"applied": True, "time_ratio": float(time_ratio), "semitones": float(semitones),
            "formant_preserved": bool(formant and semitones), "cmd": "rubberband -q -3 --centre-focus -t R -p S" + (" -F" if formant and semitones else ""),
            "length_in": len(x), "length_out": n_out, "length_rubberband": len(y)}
    if qualify:
        info["backend"] = rubberband_info()
    return _fit(y, n_out), info


# ----------------------------------------------------------------- chroma / TIV
def beat_chroma(x: np.ndarray, sr: int, beats_s: Sequence[float], analysis_sr: int = 22050, hop: int = 512,
                t0: Optional[float] = None, t1: Optional[float] = None) -> dict:
    """Cromagramma CQT (librosa, ``norm=None``) sincronizzato ai battiti con la mediana per battito
    (AutoMashUpper, Davies et al. 2014 §II-A). ``x`` è l'audio che inizia a 0 s nelle coordinate di
    ``beats_s``; con ``t0``/``t1`` si analizza solo quel tratto. Restituisce ``chroma`` (battiti x 12, non
    normalizzato), ``energy`` (somma per battito) e ``beat_s`` (inizio di ogni battito)."""
    import librosa
    from scipy.signal import resample_poly

    beats = np.asarray(beats_s, float)
    lo = 0.0 if t0 is None else max(0.0, float(t0))
    hi = len(x) / sr if t1 is None else min(len(x) / sr, float(t1))
    beats = beats[(beats >= lo - 1e-6) & (beats <= hi + 1e-6)]
    if len(beats) < 2:
        return {"chroma": np.zeros((0, 12)), "energy": np.zeros(0), "beat_s": beats[:0]}
    s0, s1 = int(beats[0] * sr), int(min(len(x), round(beats[-1] * sr) + hop * sr // analysis_sr + 1))
    mono = _as_stereo(x[s0:s1]).mean(axis=1)
    g = math.gcd(int(analysis_sr), int(sr))
    y = resample_poly(mono, analysis_sr // g, sr // g).astype(np.float32)
    if len(y) < 2048:
        y = np.r_[y, np.zeros(2048 - len(y), np.float32)]
    C = librosa.feature.chroma_cqt(y=y, sr=analysis_sr, hop_length=hop, norm=None)
    ft = beats[0] + librosa.frames_to_time(np.arange(C.shape[1]), sr=analysis_sr, hop_length=hop)
    out = np.zeros((len(beats) - 1, 12))
    for i, (b0, b1) in enumerate(zip(beats[:-1], beats[1:])):
        idx = (ft >= b0) & (ft < b1)
        if idx.any():
            out[i] = np.median(C[:, idx], axis=1)
    return {"chroma": out, "energy": out.sum(axis=1), "beat_s": beats[:-1]}


def tiv(chroma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vettore TIV (Bernardes 2016) come ``TIVCollection.from_pcp`` di TIV.lib: DFT del cromagramma (12 classi),
    coefficienti 1..6 divisi per l'energia (coefficiente 0, + eps) e pesati con ``TIV_WEIGHTS_AUDIO``."""
    c = np.atleast_2d(np.asarray(chroma, float))
    f = np.fft.rfft(c, n=12, axis=-1)
    energy = f[:, 0] + np.finfo(float).eps
    vec = f[:, 1:7] / energy[:, None] * TIV_WEIGHTS_AUDIO
    return energy, vec


def tiv_compatibility(chroma_x: np.ndarray, chroma_y: np.ndarray) -> np.ndarray:
    """``TIV.small_scale_compatibility`` di TIV.lib, frame per frame (più basso = più compatibile):
    (1 - |(Tx + Ty)/2| / |w|) * (|Tx - Ty| / (2 |w|))."""
    _, vx = tiv(chroma_x)
    _, vy = tiv(chroma_y)
    wn = float(np.linalg.norm(TIV_WEIGHTS_AUDIO))
    diss = 1.0 - np.linalg.norm((vx + vy) / 2.0, axis=1) / wn
    rel = np.linalg.norm(vx - vy, axis=1) / (wn * 2.0)
    return diss * rel


def transpose_chroma(chroma: np.ndarray, semitones: int) -> np.ndarray:
    """Cromagramma trasposto di ``semitones`` (classe i -> i + s)."""
    return np.roll(np.asarray(chroma, float), int(semitones), axis=-1)


def resample_beats(chroma: np.ndarray, k: float) -> np.ndarray:
    """Battiti della sorgente portati ai battiti di chi ospita: ``k`` battute ospite per battuta sorgente
    (2: ogni battito si ripete due volte; 1/2: media di coppie di battiti)."""
    c = np.asarray(chroma, float)
    if k == 1.0 or len(c) == 0:
        return c
    if k == 2.0:
        return np.repeat(c, 2, axis=0)
    if k == 0.5:
        m = len(c) // 2
        return c[: 2 * m].reshape(m, 2, 12).mean(axis=1) if m else c[:0]
    raise ValueError("k in (0.5, 1, 2)")


def harmonic_match(top: np.ndarray, bed: np.ndarray, *, top_weight: Optional[np.ndarray] = None,
                   shifts: Sequence[int] = SHIFT_RANGE, min_gain: float = 0.01, penalty: float = 0.002) -> dict:
    """Compatibilità del materiale ``top`` (es. voce di A, battiti x 12) trasposto di s semitoni sopra ``bed``
    (es. strumentale di B, stessi battiti). Per ogni s: TIV ``small_scale_compatibility`` media (pesata con
    ``top_weight``, es. energia della voce per battito) e similarità coseno media (AutoMashUpper). ``best`` =
    minimo della compatibilità + ``penalty`` * |s|; si traspone solo se il guadagno su s = 0 è >= ``min_gain``.
    Con un solo frame (``top`` e ``bed`` medi) è una compatibilità globale."""
    t = np.atleast_2d(np.asarray(top, float))
    b = np.atleast_2d(np.asarray(bed, float))
    n = min(len(t), len(b))
    if n == 0:
        return {"status": "empty", "best_shift": 0, "rows": []}
    t, b = t[:n], b[:n]
    w = np.ones(n) if top_weight is None else np.asarray(top_weight, float)[:n]
    w = np.where(t.sum(axis=1) > 0, w, 0.0) * np.where(b.sum(axis=1) > 0, 1.0, 0.0)
    if w.sum() <= 0:
        return {"status": "silent", "best_shift": 0, "rows": []}
    rows = []
    for s in shifts:
        ts = transpose_chroma(t, s)
        comp = tiv_compatibility(ts, b)
        num = (ts * b).sum(axis=1)
        den = np.linalg.norm(ts, axis=1) * np.linalg.norm(b, axis=1)
        cos = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
        rows.append({"shift": int(s), "tiv_compat": float((comp * w).sum() / w.sum()), "cosine": float((cos * w).sum() / w.sum())})
    for r in rows:
        r["objective"] = r["tiv_compat"] + penalty * abs(r["shift"])
    at0 = next(r for r in rows if r["shift"] == 0) if any(r["shift"] == 0 for r in rows) else None
    best = min(rows, key=lambda r: (r["objective"], abs(r["shift"])))
    gain = (at0["tiv_compat"] - best["tiv_compat"]) if at0 else None
    chosen = best if (at0 is None or best["shift"] == 0 or gain >= min_gain) else at0
    return {"status": "ok", "frames": int(n), "rows": rows, "best_shift": int(chosen["shift"]), "best": chosen,
            "unconstrained_best_shift": int(best["shift"]), "gain_over_no_shift": gain,
            "rule": "TIV small-scale compatibility (Bernardes 2016, TIV.lib) per beat, lower = more compatible; shift only if it "
                    f"improves on 0 by >= {min_gain} [H]; ties -> smaller |shift| (penalty {penalty}/semitone [H])"}


def mashability_score(vertical: Mapping[str, Any], horizontal: Optional[Mapping[str, Any]], time_ratio: float,
                      max_stretch: float = 0.12, weights: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> dict:
    """Punteggio d'ordinamento [H] nello stile di AutoMashUpper (armonia pesata di più, poi altri termini):
    0,6 x armonia verticale (1 - compatibilità TIV / 0,25, tagliata a [0, 1]) + 0,2 x armonia orizzontale
    (stessa scala, A prima e B dopo) + 0,2 x tempo (1 - |log r| / log(1 + max_stretch)). Ordina candidati;
    non stima il gradimento (Davies 2014: correlazione significativa ma debole con voci sovrapposte)."""
    def h(m):
        if not m or m.get("status") != "ok":
            return 0.0
        return float(np.clip(1.0 - m["best"]["tiv_compat"] / 0.25, 0.0, 1.0))

    t = float(np.clip(1.0 - abs(math.log(time_ratio)) / math.log(1.0 + max_stretch), 0.0, 1.0))
    parts = {"vertical": h(vertical), "horizontal": h(horizontal), "tempo": t}
    score = weights[0] * parts["vertical"] + weights[1] * parts["horizontal"] + weights[2] * parts["tempo"]
    return {"score": score, "parts": parts, "weights": list(weights), "time_ratio": time_ratio,
            "basis": "[H] ranking only, AutoMashUpper-style weighting (harmonic first) [S]; tiv scale 0.25 [H]"}


# ------------------------------------------------------------------ words / gates
def vocal_units(words: Optional[Sequence[Mapping[str, Any]]], sr: int, stem: Optional[np.ndarray] = None) -> list[dict]:
    """Parole ASR ordinate (``start``/``end`` in campioni); senza parole, tratti attivi dello stem (lacune
    < 60 ms unite) come unità anonime."""
    out = []
    for w in words or []:
        try:
            s, e = int(w["start_sample"]), int(w["end_sample"])
        except (KeyError, TypeError, ValueError):
            continue
        if e > s:
            out.append({"start": s, "end": e, "text": str(w.get("text", w.get("word", ""))).strip(),
                        "confidence": w.get("confidence"), "source": "asr"})
    if out or stem is None:
        return sorted(out, key=lambda u: u["start"])
    t, act, info = dj.vocal_activity(stem, sr)
    hop = int(round(info.get("hop_s", 0.02) * sr))
    runs, cur = [], None
    for i, a in enumerate(act):
        if a and cur is None:
            cur = i
        if not a and cur is not None:
            runs.append([cur, i])
            cur = None
    if cur is not None:
        runs.append([cur, len(act)])
    merged = []
    for r in runs:
        if merged and (r[0] - merged[-1][1]) * hop < 0.06 * sr:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [{"start": a * hop, "end": b * hop, "text": "", "confidence": None, "source": "stem"} for a, b in merged]


def line_starts(units: Sequence[Mapping[str, Any]], gap: int) -> list[int]:
    return [i for i, u in enumerate(units) if i == 0 or u["start"] - units[i - 1]["end"] >= gap]


def line_ends(units: Sequence[Mapping[str, Any]], gap: int) -> list[int]:
    return [i for i, u in enumerate(units) if i == len(units) - 1 or units[i + 1]["start"] - u["end"] >= gap]


def _stem_quiet(ctx, stem: np.ndarray, s0: int, s1: int, cfg: MashupConfig) -> float:
    return cr._cover(stem, ctx.sr, s0, s1, stem)


def a_exit_point(ctx, units: Sequence[Mapping[str, Any]], deadline: int, cfg: MashupConfig, *, max_back_s: float) -> dict:
    """Punto (campione sorgente di A) da cui la voce di A tace: dopo la fine dell'ultimo verso che termina
    entro ``deadline`` e non prima di ``deadline - max_back_s``; la voce che verrebbe dopo (fino a
    ``deadline``) è tolta solo se inizia almeno ``phrase_gap_s`` dopo. Se lo stem di A è già muto da
    lì a ``deadline`` il punto è ``deadline`` (``kind = 'silent'``). Errore se A non chiude un verso."""
    sr = ctx.sr
    gap = int(round(cfg.phrase_gap_s * sr))
    pad = int(round(cfg.word_pad_s * sr))
    guard = int(round(cfg.stem_guard_s * sr))
    if _stem_quiet(ctx, ctx.va, deadline - int(round(cfg.phrase_gap_s * sr)), deadline + guard, cfg) <= cfg.stem_guard_cover \
            and not any(u["start"] < deadline + pad < u["end"] for u in units):
        last = [u for u in units if u["end"] <= deadline]
        return {"sample": int(deadline), "kind": "silent", "last_word": (last[-1]["text"] if last else None)}
    ends = line_ends(units, gap)
    lo = deadline - int(round(max_back_s * sr))
    soft = []
    for i in sorted(ends, reverse=True):
        u = units[i]
        keep = u["end"] + pad
        if keep > deadline or u["end"] < lo:
            continue
        nxt = units[i + 1]["start"] if i + 1 < len(units) else None
        if nxt is not None and nxt < keep + gap // 2:
            continue
        if _stem_quiet(ctx, ctx.va, keep, keep + guard, cfg) > cfg.stem_guard_cover:
            soft.append((i, keep, nxt))
            continue
        return {"sample": int(keep), "kind": "line_end", "last_word": u["text"], "word_samples": [u["start"], u["end"]],
                "next_word_sample": nxt, "unit_index": i, "mute_ms": None}
    if soft and cfg.soft_line_end_mute_ms:
        # melodic/autotuned vocals rarely leave a silent stem (reverb tails, doubles, ad-libs): the ASR pause still marks the
        # line end, the stem is faded over soft_line_end_mute_ms instead of cut [H]; no ASR word starts inside the fade
        i, keep, nxt = soft[0]
        fade = int(round(cfg.soft_line_end_mute_ms / 1000 * sr))
        if nxt is None or nxt >= keep + fade:
            u = units[i]
            return {"sample": int(keep + fade), "kind": "line_end_soft", "last_word": u["text"], "word_samples": [u["start"], u["end"]],
                    "next_word_sample": nxt, "unit_index": i, "mute_ms": cfg.soft_line_end_mute_ms,
                    "stem_cover_after": _stem_quiet(ctx, ctx.va, keep, keep + guard, cfg),
                    "rule": "[H] ASR pause >= phrase_gap_s but stem still active (tail/ad-lib): faded, not cut"}
    raise DJRenderError("A does not end a line (ASR pause and silent stem) before the cue: the voice would be cut")


def a_line_from(units: Sequence[Mapping[str, Any]], X: int, beat: int, bar: int, cfg: MashupConfig, sr: int,
                max_lead_beats: float = 1.0) -> int:
    """Indice della parola che apre un verso di A vicino al cue ``X`` (da ``max_lead_beats`` prima, come
    pickup, a una battuta dopo), il più vicino a ``X``."""
    gap = int(round(cfg.phrase_gap_s * sr))
    cands = [i for i in line_starts(units, gap) if X - max_lead_beats * beat <= units[i]["start"] <= X + bar]
    if not cands:
        raise DJRenderError("no line of A starts near the cue (needed to borrow A's verse)")
    return min(cands, key=lambda i: abs(units[i]["start"] - X))


def a_line_end_after(units: Sequence[Mapping[str, Any]], i0: int, lo: int, hi: int, cfg: MashupConfig, sr: int,
                     prefer: Optional[int] = None) -> int:
    """Indice dell'ultima parola di un verso di A (pausa dopo >= ``phrase_gap_s``) che finisce in [lo, hi];
    se ``prefer`` è dato, quella più vicina a ``prefer``, altrimenti la più tardiva."""
    gap = int(round(cfg.phrase_gap_s * sr))
    cands = [i for i in line_ends(units, gap) if i >= i0 and lo <= units[i]["end"] <= hi]
    if not cands:
        raise DJRenderError("A's borrowed line does not end inside the allowed span")
    if prefer is None:
        return max(cands, key=lambda i: units[i]["end"])
    return min(cands, key=lambda i: abs(units[i]["end"] - prefer))


def b_vocal_gate(ctx, units_b: Sequence[Mapping[str, Any]], *, t_x: int, earliest: int, n: int, cfg: MashupConfig,
                 latest: Optional[int] = None) -> tuple[np.ndarray, dict]:
    """Guadagno (0/1, rampa 20 ms) della voce di B nella clip: si apre ``gate_lead_s`` prima della prima
    parola che apre una frase (pausa prima >= ``phrase_gap_s``) con inizio >= ``earliest``; prima resta chiuso
    (B suona strumentale). B non è stirato: clip = ``t_x`` + (sorgente - cue di B)."""
    sr = ctx.sr
    sb = ctx.sb()
    gap = int(round(cfg.phrase_gap_s * sr))
    lead = int(round(cfg.gate_lead_s * sr))
    to_clip = lambda s: t_x + (int(s) - sb)  # noqa: E731
    starts = [units_b[i] for i in line_starts(units_b, gap) if to_clip(units_b[i]["start"]) - lead >= earliest]
    gate = np.zeros(n)
    if not starts or (latest is not None and to_clip(starts[0]["start"]) > latest) or to_clip(starts[0]["start"]) >= n:
        return gate, {"status": "closed in the clip", "earliest_clip_sample": int(earliest),
                      "rule": "[H] B's vocal opens only at a phrase start (pause >= phrase_gap_s before its first word)"}
    u = starts[0]
    o = to_clip(u["start"]) - lead
    ramp = int(round(0.02 * sr))
    gate[:] = _rise(n, o - ramp, o, "sin")
    prev = [w for w in units_b if w["end"] <= u["start"]]
    return gate, {"status": "open", "open_clip_sample": int(o), "first_word": u["text"], "first_word_source_sample": int(u["start"]),
                  "pickup_before_downbeat_s": max(0.0, (sb - u["start"]) / sr), "previous_word_end_source_sample": prev[-1]["end"] if prev else None,
                  "earliest_clip_sample": int(earliest),
                  "rule": "[H] B's vocal opens only at a phrase start (pause >= phrase_gap_s before its first word), pickup included"}


# --------------------------------------------------------------------- context
def _ctx(audio_a, audio_b, vocals_a, vocals_b, grids: Mapping[str, Any], cue_out_a, cue_in_b, sr, cfg: MashupConfig):
    return cr._prepare(audio_a, audio_b, vocals_a, vocals_b, grids["beats_a"], grids["downbeats_a"], grids["beats_b"],
                       grids["downbeats_b"], cue_out_a, cue_in_b, sr, cfg.creative, need_vocals_a=True, need_vocals_b=True)


def _bars(ctx) -> dict:
    sr = ctx.sr
    return {"bar_a": int(round(ctx.ta["bar_s"] * sr)), "beat_a": int(round(ctx.ta["beat_s"] * sr)),
            "bar_b": int(round(ctx.tb["bar_s"] * sr)), "beat_b": int(round(ctx.tb["beat_s"] * sr))}


def _key_block(key_a, key_b, shift_a_material: Optional[int] = None, shift_b_material: Optional[int] = None, match: Optional[dict] = None) -> dict:
    out = {"key_a": key_a, "key_b": key_b, "relation_a_b": cr.key_relation(key_a, key_b)}
    ka, kb = cr.parse_key(key_a), cr.parse_key(key_b)
    if shift_a_material is not None:
        out["shift_a_material"] = int(shift_a_material)
        if ka is not None:
            out["a_material_key_after_shift"] = cr.key_name((ka[0] + shift_a_material) % 12, ka[1])
            out["relation_after_shift"] = cr.key_relation(out["a_material_key_after_shift"], key_b)
    if shift_b_material is not None:
        out["shift_b_material"] = int(shift_b_material)
        if kb is not None:
            out["b_material_key_after_shift"] = cr.key_name((kb[0] + shift_b_material) % 12, kb[1])
            out["relation_after_shift"] = cr.key_relation(key_a, out["b_material_key_after_shift"])
    if match is not None:
        out["harmonic_match"] = match
    return out


def key_compatible_shifts(key_top: Optional[str], key_bed: Optional[str], max_semitones: int = 3) -> list[int]:
    """Trasposizioni s (-max..+max, prima le piccole) che rendono la tonalità stimata del materiale trasposto
    stessa/adiacente/relativa a quella dell'ospite (Camelot, ``dj_creative.key_relation``)."""
    out = []
    for s in SHIFT_RANGE:
        if abs(s) > max_semitones:
            continue
        ka = cr.parse_key(key_top)
        if ka is None or cr.parse_key(key_bed) is None:
            return []
        if cr.key_relation(cr.key_name((ka[0] + s) % 12, ka[1]), key_bed) in ("same", "adjacent", "relative"):
            out.append(int(s))
    return out


def _auto_shift(top: np.ndarray, top_beats_s: Sequence[float], top_span: tuple[int, int], bed: np.ndarray,
                bed_beats_s: Sequence[float], bed_span: tuple[int, int], k: float, sr: int, cfg: MashupConfig,
                keys: Optional[tuple[Optional[str], Optional[str]]] = None) -> dict:
    """Trasposizione del materiale ``top`` (tratto in campioni) sopra ``bed`` con :func:`harmonic_match` sui
    battiti; i battiti della sorgente vanno a quelli dell'ospite con :func:`resample_beats`. Con ``cfg.key_guard`` e
    le tonalità ``keys`` = (materiale, ospite) si scelgono solo le trasposizioni che rendono le tonalità stimate
    compatibili (criterio [user] del 21/10/2026: niente tonalità discordante), se ne esiste almeno una; la TIV sceglie
    fra quelle."""
    shifts = SHIFT_RANGE
    guard = None
    if cfg.key_guard and keys is not None:
        ks = key_compatible_shifts(keys[0], keys[1])
        guard = {"keys": list(keys), "compatible_shifts": ks, "applied": bool(ks)}
        if ks:
            shifts = tuple(ks)
    ct = beat_chroma(top, sr, top_beats_s, t0=top_span[0] / sr, t1=top_span[1] / sr)
    cb = beat_chroma(bed, sr, bed_beats_s, t0=bed_span[0] / sr, t1=bed_span[1] / sr)
    t = resample_beats(ct["chroma"], k)
    w = resample_beats(ct["energy"][:, None].repeat(12, 1), k)[:, 0] if len(ct["energy"]) else None
    m = harmonic_match(t, cb["chroma"], top_weight=w, shifts=shifts, min_gain=cfg.shift_min_gain, penalty=cfg.shift_penalty)
    m["key_guard"] = guard
    m["frames_top"], m["frames_bed"] = int(len(t)), int(len(cb["chroma"]))
    if m.get("status") != "ok" or m.get("frames", 0) < 2:  # too short for a beat-wise match: one frame over the whole span
        if len(ct["chroma"]) < 2:
            ct = beat_chroma(top, sr, [top_span[0] / sr, top_span[1] / sr])
        if len(cb["chroma"]) < 2:
            cb = beat_chroma(bed, sr, [bed_span[0] / sr, bed_span[1] / sr])
        g = harmonic_match(ct["chroma"].sum(axis=0, keepdims=True), cb["chroma"].sum(axis=0, keepdims=True), shifts=shifts,
                           min_gain=cfg.shift_min_gain, penalty=cfg.shift_penalty)
        g["mode"] = "global (one chroma frame over the fragment): shorter than 2 beats"
        g["key_guard"] = guard
        if g.get("status") != "ok" and guard and guard.get("compatible_shifts"):
            g["best_shift"] = int(guard["compatible_shifts"][0])
            g["fallback"] = "no usable chroma: smallest key-compatible shift (key guard)"
        return g
    m["mode"] = "beat-synchronous"
    return m


def _resolve_shift(shift, auto: dict) -> int:
    if shift == "auto":
        return int(auto.get("best_shift", 0))
    s = int(shift)
    if abs(s) > 6:
        raise DJRenderError("shift in -6..6")
    return s


def _common_b(ctx, t_x: int, n: int, units_b, earliest: int, cfg: MashupConfig, *, latest: Optional[int] = None,
              mid_cut: Optional[tuple[int, int]] = None, instrumental: Optional[np.ndarray] = None):
    """B al suo tempo dal cue ``t_x``: strumentale (mix - stem, o ``instrumental``) con micro fade-in che finisce
    sul downbeat, voce di B dal suo gate (da :func:`b_vocal_gate`), medi -3 dB in ``mid_cut``."""
    sr, ccfg = ctx.sr, cfg.creative
    sb = ctx.sb()
    ins = cr._b_entry(ctx.b - ctx.vb if instrumental is None else instrumental, sb, t_x, n, ccfg, sr)
    if mid_cut is not None:
        beat = int(round(ctx.tb["beat_s"] * sr))
        m0, m1 = mid_cut
        mid = 1.0 - (1.0 - _db(cfg.mid_cut_db)) * (_rise(n, m0 - beat // 2, m0, "cos") * (1.0 - _rise(n, m1, m1 + beat, "cos")))
        ins = cr._eq(ins, sr, ccfg, mid=mid)
    gate, ginfo = b_vocal_gate(ctx, units_b, t_x=t_x, earliest=earliest, n=n, cfg=cfg, latest=latest)
    voc = _slice(ctx.vb, sb - t_x, n) * gate[:, None]
    return ins, voc, gate, ginfo


def _section(ctx, a_end_src: int, t_x: int, n: int, t_e: int) -> tuple[np.ndarray, dict]:
    sb = ctx.sb()
    ccfg = ctx.cfg
    b_ref = ctx.b[sb : sb + int(round(ccfg.section_gain.b_bars * ctx.tb["bar_s"] * ctx.sr))]
    return cr._section_gain(ctx, a_end_src, b_ref, t_x, ctx.tb["bar_s"], n, t_e,
                            "B source (no stretch) from its entry downbeat; A before the transition (dry)")


def _a_deck(ctx, off_a: int, n: int, mute_at: int, t_x: int, cfg: MashupConfig, *, mute_ms: Optional[float] = None, filter_from: Optional[int] = None,
            mid_cut: Optional[tuple[int, int]] = None, fade: Optional[tuple[int, int]] = None):
    """A nella clip: mix pieno, voce tolta (mix - stem, rampa ``vocal_mute_ms``) da ``mute_at`` (clip), uscita sul
    downbeat ``t_x`` (micro fade) o con dissolvenza ``fade`` (clip); passa-alto crescente 30 -> 1200 Hz da
    ``filter_from``; medi -3 dB in ``mid_cut``. Restituisce deck, voce udibile e guadagno della voce."""
    sr, ccfg = ctx.sr, ctx.cfg
    a_mix = _slice(ctx.a, off_a, n)
    a_voc = _slice(ctx.va, off_a, n)
    mute = _rise(n, mute_at - int(round((mute_ms or ccfg.vocal_mute_ms) / 1000 * sr)), mute_at, "sin")
    a_raw = a_mix - a_voc * mute[:, None]
    if filter_from is not None and filter_from < t_x:
        idx = np.arange(n)
        fc = np.where(idx < filter_from, 30.0, 30.0 * (1200.0 / 30.0) ** np.clip((idx - filter_from) / max(t_x - filter_from, 1), 0, 1))
        filt = fx.time_varying_filter(a_raw, sr, fc, "highpass", 4, 128)
        eng = _rise(n, filter_from, filter_from + int(0.03 * sr))[:, None]
        a_raw = a_raw * (1 - eng) + filt * eng
    if mid_cut is not None:
        beat = int(round(ctx.ta["beat_s"] * sr))
        m0, m1 = mid_cut
        mid = 1.0 - (1.0 - _db(cfg.mid_cut_db)) * (_rise(n, m0 - beat // 2, m0, "cos") * (1.0 - _rise(n, m1, m1 + beat // 4, "cos")))
        a_raw = cr._eq(a_raw, sr, ccfg, mid=mid)
    if fade is not None:
        g = cr._fall_eqpow(n, fade[0], fade[1])
    else:
        g = cr._a_cut(n, t_x, ccfg, sr)
    return a_raw * g[:, None], a_voc * ((1.0 - mute) * g)[:, None], g


def _transform_placed(src: np.ndarray, s0: int, s1: int, anchor_src: int, anchor_clip: int, ratio: float, semis: int,
                      n: int, sr: int, cfg: MashupConfig) -> tuple[np.ndarray, dict]:
    """Frammento ``src[s0:s1]`` stirato di ``ratio`` e trasposto di ``semis`` (formanti preservate), posto
    in modo che il campione sorgente ``anchor_src`` cada su ``anchor_clip``. Restituisce il segnale nella clip
    e la mappa (sorgente -> clip) del frammento."""
    frag = _windowed(_slice(src, s0, s1 - s0), int(round(cfg.fade_ms / 1000 * sr)))
    y, info = stretch_pitch(frag, sr, ratio, semis, formant=True)
    o0 = anchor_clip + int(round((s0 - anchor_src) * ratio))
    out = np.zeros((n, 2))
    lo, hi = _place(out, y, o0)
    return out, {"source_samples": [int(s0), int(s1)], "clip_samples": [int(o0), int(o0 + len(y))], "written": [int(lo), int(hi)],
                 "time_ratio": float(ratio), "semitones": int(semis), "anchor": {"source": int(anchor_src), "clip": int(anchor_clip)},
                 "map": "clip = anchor_clip + (source - anchor_source) * time_ratio", "stretch_pitch": info}


def _echo(frag: np.ndarray, sr: int, period: int, repeats: int, feedback: float = 0.45, hpf0: float = 200.0, hpf1: float = 1500.0) -> np.ndarray:
    """Solo le ripetizioni (niente dry) di ``frag``: ``repeats`` copie ogni ``period`` campioni, guadagno
    ``feedback``^k, passa-alto che sale da ``hpf0`` a ``hpf1`` (eco da mixer DJ [P])."""
    out = np.zeros((len(frag) + repeats * period, 2))
    cur = _as_stereo(frag).copy()
    for k in range(1, repeats + 1):
        hz = hpf0 * (hpf1 / hpf0) ** ((k - 1) / max(repeats - 1, 1))
        cur = _hpf(cur, sr, hz) * feedback
        _place(out, cur, k * period)
    return out


def _duck(sig: np.ndarray, gate: np.ndarray, sr: int) -> np.ndarray:
    """Zittisce ``sig`` dove la voce di B è aperta (una voce alla volta), rampa 30 ms."""
    k = np.hanning(2 * int(0.03 * sr) + 1)
    k /= k.sum()
    g = 1.0 - np.clip(np.convolve(gate, k, mode="same"), 0, 1)
    return sig * g[:, None]


def _next_b_downbeat(t_x: int, t: int, bar_b: int) -> int:
    return t_x + int(math.ceil(max(0, t - t_x) / bar_b - 1e-9)) * bar_b


def _extend_t_e(t_e: int, ginfo: Mapping[str, Any], bar_b: int) -> int:
    """La transizione finisce quando torna la voce di B, se si riapre entro 2 battute di B dalla fine prevista: così il
    confronto di loudness 'dopo' (3 s da ``t_e``) misura B intero, non il suo strumentale in attesa della frase [H]."""
    if ginfo.get("status") == "open" and t_e < ginfo["open_clip_sample"] <= t_e + 2 * bar_b:
        return int(ginfo["open_clip_sample"])
    return int(t_e)


def _b_main_map(t_x: int, n: int, sb: int, ginfo: Mapping[str, Any], what: str) -> list[dict]:
    """``b_map`` di B al suo tempo dal cue (clip ``t_x`` = sorgente ``sb``): senza voce fino al gate, con la voce da lì
    (anche prima di ``t_x`` se la frase ha un pickup). Così le parole di B che il gate tiene chiuse non risultano
    'udibili' per il QA (``words_in_clip``, controllo ``b_entry``)."""
    o = ginfo.get("open_clip_sample") if ginfo.get("status") == "open" else None
    if o is None or o >= n:
        return [{"out_start": t_x, "out_end": n, "src_start": sb, "rate": 1.0, "vocals": False,
                 "what": what + ": instrumental (B's vocal stays closed in the clip)"}]
    segs = []
    if o > t_x:
        segs.append({"out_start": t_x, "out_end": o, "src_start": sb, "rate": 1.0, "vocals": False,
                     "what": what + ": instrumental until B's phrase start"})
    segs.append({"out_start": o, "out_end": n, "src_start": sb + (o - t_x), "rate": 1.0, "vocals": True,
                 "what": what + ": with its vocal from the phrase start (gate)"})
    return segs


def _finish(ctx, technique: str, *, deck_a, deck_b, t_s, t_sw, t_e, off_a, creative: dict, a_map, b_map, va_heard, vb_heard,
            sg_info, provenance, return_decks, overlay_parts: dict, overlay=None, a_voice=None, b_voice=None,
            b_voice_heard=None, exit_effect: Optional[str] = None) -> DJRenderResult:
    """``a_voice``/``b_voice``: materiale cantato preso in prestito, sommato al deck del brano da cui viene (la voce di A
    è A, come il loop di ``dj_creative.loop_extend``) e alla sua voce udibile; non passa per ``trim_overlay`` (che
    abbasserebbe di 12 dB una voce solista sopra un B forte: misurato nella prima prova). ``overlay``: solo effetti
    (freeze, swell), con la riduzione di ``trim_overlay``. ``b_voice_heard``: voce di B dentro un effetto (swell), solo
    per il controllo 'una voce alla volta'."""
    extra = {"overlay_parts": overlay_parts}
    if a_voice is not None and np.any(a_voice):
        n_clip = int(t_e) + int(round(ctx.cfg.post_s * ctx.sr))
        extra["a_raw"] = _fit(deck_a, n_clip)  # A's beat-carrying deck without the borrowed voice (rhythm check of dj_qa)
        deck_a = deck_a + a_voice
        va_heard = va_heard + a_voice
    if b_voice is not None and np.any(b_voice):
        deck_b = deck_b + b_voice
        vb_heard = vb_heard + b_voice
    if b_voice_heard is not None and np.any(b_voice_heard):
        vb_heard = vb_heard + b_voice_heard
    if overlay is not None and not np.any(overlay):
        overlay = None
    creative = dict(creative, technique=technique, module=DJ_MASHUP_VERSION,
                    borrowed_voice_level="unchanged: stem at its track's -14 LUFS normalisation (no overlay trim)")
    res = cr._finish(ctx, technique, deck_a=deck_a, deck_b=deck_b, overlay=overlay, t_s=t_s, t_sw=t_sw, t_e=t_e, off_a=off_a,
                     sb_at_switch=ctx.sb(), stretch=cr._tempo_block(ctx, False), creative=creative, exit_effect=exit_effect,
                     a_map=a_map, b_map=b_map, vocals_a_heard=va_heard, vocals_b_heard=vb_heard, sg_info=sg_info,
                     extra_decks=extra, provenance=provenance, return_decks=return_decks)
    res.metadata["mashup_version"] = DJ_MASHUP_VERSION
    gate = creative.get("b_vocal_gate") or {}
    if gate.get("status") == "open":
        # dj_qa.6 (check_b_entry): the point where B's *voice* becomes audible is the gate, not the instrumental entry;
        # declared like dj_render.add_b_lead_in so the QA checks that no word of B is cut there and that its phrase is not
        # already under way (B's own timeline, independent of the gate's 0.25 s rule).
        o = int(gate["open_clip_sample"])
        res.metadata["b_entry"] = {"audible_output_sample": o, "b_source_sample": int(ctx.sb() + (o - int(t_sw))),
                                   "what": "B's vocal gate (phrase start, pickup included); B's instrumental enters at the switch",
                                   "instrumental_entry_output_sample": int(t_sw), "module": DJ_MASHUP_VERSION}
    return res


def _check_no_word_across(units, at: int, sr: int, tol_s: float = 0.03, label: str = "A") -> None:
    tol = int(round(tol_s * sr))
    for u in units:
        if u["start"] < at - tol and u["end"] > at + tol:
            raise DJRenderError(f"{label}'s word {u['text']!r} crosses the cut point")


# ==================================================================== techniques
def acapella_mashup(audio_a, audio_b, *, grids: Mapping[str, Any], cue_out_a: float, cue_in_b: float, vocals_a, vocals_b,
                    words_a=None, words_b=None, key_a: Optional[str] = None, key_b: Optional[str] = None, bars: int = 4,
                    shift: Any = "auto", echo_repeats: int = 2, sample_rate: int = dj.DEFAULT_SAMPLE_RATE,
                    config: Optional[MashupConfig] = None, provenance: Optional[dict] = None, return_decks: bool = False,
                    _drums_until: Optional[int] = None, _stems_b: Optional[Mapping[str, np.ndarray]] = None,
                    _technique: str = "acapella_mashup") -> DJRenderResult:
    """Verso di A (stem vocale, da un inizio verso vicino al cue, pickup compreso) sopra lo strumentale di B
    al tempo di B per 2..``bars`` battute di A, stirato e trasposto (``shift`` = 'auto' dalla mashability).
    La base di A esce sul downbeat del cue (dove entra B); il verso finisce a un fine verso; la sua ultima
    parola ha ``echo_repeats`` ripetizioni a un battito di B; la voce di B si apre alla sua prima frase dopo."""
    cfg = config or MashupConfig()
    sr = int(sample_rate)
    if bars not in (2, 4, 8):
        raise DJRenderError("bars in (2, 4, 8)")
    ctx = _ctx(audio_a, audio_b, vocals_a, vocals_b, grids, cue_out_a, cue_in_b, sr, cfg)
    g = _bars(ctx)
    pre, post = int(round(ctx.cfg.pre_s * sr)), int(round(ctx.cfg.post_s * sr))
    X, sb = ctx.sa(), ctx.sb()
    k, r = fold_ratio(ctx.ta["bar_s"], ctx.tb["bar_s"], cfg.max_vocal_stretch)
    ua, ub = vocal_units(words_a, sr, ctx.va), vocal_units(words_b, sr, ctx.vb)
    i0 = a_line_from(ua, X, g["beat_a"], g["bar_a"], cfg, sr)
    p0 = ua[i0]["start"] - int(round(cfg.gate_lead_s * sr))
    prev_end = ua[i0 - 1]["end"] if i0 > 0 else None
    if p0 > X and prev_end is not None and prev_end > X - int(0.03 * sr):
        raise DJRenderError("A's previous word crosses the cue")
    if _drums_until is not None:  # beat_swap: the line must end before B's harmony enters
        hi = X + int((_drums_until) / r) - int(round(cfg.word_pad_s * sr))
        i1 = a_line_end_after(ua, i0, p0 + g["bar_a"], hi, cfg, sr)
    else:
        i1 = a_line_end_after(ua, i0, p0 + max(1, bars // 2) * g["bar_a"], p0 + bars * g["bar_a"] + g["beat_a"], cfg, sr,
                              prefer=p0 + bars * g["bar_a"])
    p1 = ua[i1]["end"] + int(round(cfg.word_pad_s * sr))
    stem_after = cr._cover(ctx.va, sr, p1, p1 + int(cfg.stem_guard_s * sr), ctx.va)
    if stem_after > cfg.stem_guard_cover:  # tail/ad-lib after the line (melodic vocals): the borrowed fragment fades out [H]
        p1 += int(round(min(cfg.soft_line_end_mute_ms, 1000 * cfg.phrase_gap_s * 0.8) / 1000 * sr))
    S = min(p0, X)
    off_a = S - pre
    t_s, t_x = pre, X - off_a
    span_b = int(round((p1 - p0) * r))
    if _stems_b is None:
        auto = _auto_shift(ctx.va, ctx.grid["beats_a"], (p0, p1), ctx.b - ctx.vb, ctx.grid["beats_b"], (sb, sb + span_b + g["beat_b"]), k, sr, cfg, keys=(key_a, key_b))
        semis = _resolve_shift(shift, auto)
    else:  # drums only under the voice: no harmony to match
        auto, semis = {"status": "not needed", "note": "only B's drums under A's voice (B&B: mix clashing records when one is just percussion)"}, 0
    n_guess = t_x + span_b + 8 * g["bar_b"] + post
    ov, tmap = _transform_placed(ctx.va, p0, p1, X, t_x, r, semis, n_guess, sr, cfg)
    o0, o1 = tmap["clip_samples"]
    # echo of the last word, one beat of B
    tail = np.zeros_like(ov)
    w = ua[i1]
    if echo_repeats > 0:
        w0 = t_x + int(round((w["start"] - X) * r))
        frag = ov[max(0, w0 - int(0.01 * sr)) : o1].copy()
        if len(frag) > 32:
            _place(tail, _echo(frag, sr, g["beat_b"], echo_repeats), max(0, w0 - int(0.01 * sr)))
    harmony_in = None
    if _drums_until is not None:
        harmony_in = t_x + int(_drums_until)
        t_e = harmony_in + g["bar_b"]
    else:
        t_e = _next_b_downbeat(t_x, o1, g["bar_b"])
    t_e = max(t_e, t_x + g["bar_b"])
    n = t_e + 2 * g["bar_b"] + post
    ov, tail = _fit(ov, n), _fit(tail, n)
    instrumental = None
    if _stems_b is not None:
        drums = _stems_b["drums"] * _db(ctx.loud["b_gain_db"])
        ins_full = ctx.b - ctx.vb
        u = np.clip((np.arange(len(ctx.b)) - (sb + _drums_until - int(0.02 * sr))) / max(int(0.02 * sr), 1), 0, 1)
        instrumental = _fit(drums, len(ctx.b)) * (1 - u)[:, None] + ins_full * u[:, None]
    ins, vocb, gate, ginfo = _common_b(ctx, t_x, n, ub, max(o1 + int(0.15 * sr), harmony_in or 0), cfg,
                                       mid_cut=(o0, o1) if _stems_b is None else None, instrumental=instrumental)
    tail = _duck(tail, gate, sr)
    t_e = _extend_t_e(t_e, ginfo, g["bar_b"])
    deck_a, va_heard, _ = _a_deck(ctx, off_a, n, p0 - off_a, t_x, cfg)
    sg, sg_info = _section(ctx, S, t_x, n, t_e)
    deck_b = (ins + vocb) * sg[:, None]
    vb_heard = vocb * sg[:, None]
    voice = ov + tail
    a_map = [{"out_start": 0, "out_end": min(p0 - off_a, t_x), "src_start": off_a, "rate": 1.0, "vocals": True, "what": "A full mix"}]
    if p0 - off_a < t_x:
        a_map.append({"out_start": p0 - off_a, "out_end": t_x, "src_start": p0, "rate": 1.0, "vocals": False, "what": "A instrumental (mix - stem) until its beat stops"})
    a_map.append({"out_start": o0, "out_end": o1, "src_start": p0, "rate": r, "vocals": True,
                  "what": f"A's line (vocal stem) on B's tempo, stretched x{r:.4f}, shifted {semis:+d} st", "time_map": tmap["map"]})
    b_map = _b_main_map(t_x, n, sb, ginfo, "B drums (Demucs) then full instrumental" if _stems_b is not None else "B at its own tempo")
    regions = [{"name": "A's line over B" + (" drums" if _stems_b is not None else " instrumental"), "output": [o0, o1], "intended": True},
               {"name": "A beat out / B beat in (downbeat)", "output": [t_x, t_x]},
               {"name": "echo of A's last word", "output": [o1, min(n, o1 + echo_repeats * g["beat_b"])], "intended": True}]
    if harmony_in is not None:
        regions.append({"name": "B bass + harmony in (downbeat)", "output": [harmony_in, harmony_in]})
    if ginfo.get("status") == "open":
        regions.append({"name": "B's vocal opens at a phrase start", "output": [ginfo["open_clip_sample"], ginfo["open_clip_sample"]]})
    creative = {
        "params": {"bars": bars, "shift": semis, "shift_requested": shift, "echo_repeats": echo_repeats, "time_ratio_a": r, "fold_k": k,
                   **({"drum_bars": int(round(_drums_until / g["bar_b"]))} if _drums_until is not None else {})},
        "keys": _key_block(key_a, key_b, shift_a_material=semis, match=auto),
        "a_line": {"first_word": ua[i0]["text"], "last_word": ua[i1]["text"], "source_samples": [int(p0), int(p1)],
                   "words": [u["text"] for u in ua[i0 : i1 + 1]], "pickup_before_cue_s": max(0.0, (X - p0) / sr)},
        "a_overlay": tmap, "b_vocal_gate": ginfo, "regions": regions,
        "basis": ("[P] B&B How to DJ Right pdf p. 39 ('mix them when one is just percussion'), beat swap with stems; one voice at a time [E]"
                  if _stems_b is not None else
                  "[P] B&B How to DJ Right pdf p. 86 ('Play an acappella of one track over a dub of another'); [S] AutoMashUpper / "
                  "Huang 2021 (shift chosen by harmonic compatibility, +-3 st); one voice at a time [E] Davies 2014, Brungart 2001"),
    }
    if _stems_b is not None:
        creative["stems_b"] = {"model": "htdemucs (caller)", "used": ["drums"], "then": "B mix - vocal stem from the harmony downbeat"}
    return _finish(ctx, _technique, deck_a=deck_a, deck_b=deck_b, a_voice=voice,
                   t_s=t_s, t_sw=t_x, t_e=t_e, off_a=off_a, creative=creative, a_map=a_map, b_map=b_map, va_heard=va_heard,
                   vb_heard=vb_heard, sg_info=sg_info, provenance=provenance, return_decks=return_decks,
                   overlay_parts={"a_line": ov, "a_echo": tail})


def beat_swap(audio_a, audio_b, *, grids, cue_out_a, cue_in_b, vocals_a, vocals_b, stems_b: Mapping[str, np.ndarray],
              words_a=None, words_b=None, key_a=None, key_b=None, drum_bars: int = 2, echo_repeats: int = 1,
              sample_rate: int = dj.DEFAULT_SAMPLE_RATE, config: Optional[MashupConfig] = None, provenance: Optional[dict] = None,
              return_decks: bool = False) -> DJRenderResult:
    """Verso di A sopra la sola batteria di B (``stems_b['drums']``, stem Demucs allineato a ``audio_b``) per
    ``drum_bars`` battute di B; sul downbeat successivo entrano basso e armonia di B (mix - stem vocale). Il verso
    di A deve finire prima: nessuna trasposizione (sotto c'è solo batteria)."""
    cfg = config or MashupConfig()
    if drum_bars not in (2, 4):
        raise DJRenderError("drum_bars in (2, 4)")
    if "drums" not in stems_b:
        raise DJRenderError("beat_swap needs B's drum stem")
    bar_b = int(round(fx.bar_and_beat(np.sort(np.asarray(grids["downbeats_b"], float)), np.sort(np.asarray(grids["beats_b"], float)),
                                      int(np.argmin(np.abs(np.asarray(grids["downbeats_b"], float) - cue_in_b))), +1)["bar_s"] * sample_rate))
    return acapella_mashup(audio_a, audio_b, grids=grids, cue_out_a=cue_out_a, cue_in_b=cue_in_b, vocals_a=vocals_a, vocals_b=vocals_b,
                           words_a=words_a, words_b=words_b, key_a=key_a, key_b=key_b, bars=4, shift=0, echo_repeats=echo_repeats,
                           sample_rate=sample_rate, config=cfg, provenance=provenance, return_decks=return_decks,
                           _drums_until=drum_bars * bar_b, _stems_b=stems_b, _technique="beat_swap")


def hook_teaser(audio_a, audio_b, *, grids, cue_out_a, cue_in_b, vocals_a, vocals_b, hook_s: float, words_a=None, words_b=None,
                key_a=None, key_b=None, teaser_bars: int = 2, shift: Any = "auto", sample_rate: int = dj.DEFAULT_SAMPLE_RATE,
                config: Optional[MashupConfig] = None, provenance: Optional[dict] = None, return_decks: bool = False) -> DJRenderResult:
    """La prima frase dell'hook di B (da un inizio frase dopo ``hook_s``, secondi sorgente di B; es. il primo
    ritornello) stirata al tempo di A e trasposta nella tonalità di A, sopra lo strumentale di A nelle ultime
    ``teaser_bars`` battute prima del cue; poi B entra pulito dal suo cue nella sua tonalità."""
    cfg = config or MashupConfig()
    sr = int(sample_rate)
    if teaser_bars not in (1, 2, 4):
        raise DJRenderError("teaser_bars in (1, 2, 4)")
    ctx = _ctx(audio_a, audio_b, vocals_a, vocals_b, grids, cue_out_a, cue_in_b, sr, cfg)
    g = _bars(ctx)
    pre, post = int(round(ctx.cfg.pre_s * sr)), int(round(ctx.cfg.post_s * sr))
    X, sb = ctx.sa(), ctx.sb()
    if ctx.ia - teaser_bars < 0:
        raise DJRenderError("not enough A bars before the cue")
    T0 = ctx.sa(ctx.ia - teaser_bars)
    kb, rb = fold_ratio(ctx.tb["bar_s"], ctx.ta["bar_s"], cfg.max_vocal_stretch)
    ua, ub = vocal_units(words_a, sr, ctx.va), vocal_units(words_b, sr, ctx.vb)
    gap = int(round(cfg.phrase_gap_s * sr))
    H = int(round(hook_s * sr))
    starts = [i for i in line_starts(ub, gap) if ub[i]["start"] >= H - g["beat_b"]]
    if not starts:
        raise DJRenderError("no phrase of B starts at/after the hook")
    j0 = starts[0]
    q0 = ub[j0]["start"] - int(round(cfg.gate_lead_s * sr))
    off = int(round(_bar_offset(q0, ctx.db, sr, g["beat_b"]) * rb))  # keep the phrase's place in its bar (pickup -> negative)
    place = T0 + off  # A source sample where the hook starts
    room = X - int(0.12 * sr) - place  # must end before B's own entry
    j1 = a_line_end_after(ub, j0, q0 + g["beat_b"], q0 + int(room / rb), cfg, sr)
    q1 = ub[j1]["end"] + int(round(cfg.word_pad_s * sr))
    exit_a = a_exit_point(ctx, ua, place - int(0.1 * sr), cfg, max_back_s=2 * ctx.ta["bar_s"])
    keep = exit_a["sample"]
    S = min(keep, place)
    off_a = S - pre
    t_s, t_x = pre, X - off_a
    n = t_x + 2 * g["bar_b"] + post
    auto = _auto_shift(ctx.vb, ctx.grid["beats_b"], (q0, q1), ctx.a - ctx.va, ctx.grid["beats_a"],
                       (place, place + int((q1 - q0) * rb) + g["beat_a"]), kb, sr, cfg, keys=(key_b, key_a))
    semis = _resolve_shift(shift, auto)
    ov, tmap = _transform_placed(ctx.vb, q0, q1, q0, place - off_a, rb, semis, n, sr, cfg)
    o0, o1 = tmap["clip_samples"]
    if o1 > t_x:
        raise DJRenderError("the hook teaser would run past B's entry")
    t_e = t_x + g["bar_b"]
    ins, vocb, gate, ginfo = _common_b(ctx, t_x, n, ub, max(o1 + int(0.15 * sr), t_x - g["beat_b"]), cfg)
    t_e = _extend_t_e(t_e, ginfo, g["bar_b"])
    deck_a, va_heard, _ = _a_deck(ctx, off_a, n, keep - off_a, t_x, cfg, mute_ms=exit_a.get("mute_ms"), mid_cut=(o0, o1))
    sg, sg_info = _section(ctx, S, t_x, n, t_e)
    deck_b = (ins + vocb) * sg[:, None]
    a_map = [{"out_start": 0, "out_end": keep - off_a, "src_start": off_a, "rate": 1.0, "vocals": True, "what": "A full mix"},
             {"out_start": keep - off_a, "out_end": t_x, "src_start": keep, "rate": 1.0, "vocals": False, "what": "A instrumental under B's hook"}]
    b_map = [{"out_start": o0, "out_end": o1, "src_start": q0, "rate": rb, "vocals": True,
              "what": f"B's hook (vocal stem) on A's tempo x{rb:.4f}, shifted {semis:+d} st", "time_map": tmap["map"]},
             *_b_main_map(t_x, n, sb, ginfo, "B at its own tempo and key")]
    creative = {
        "params": {"teaser_bars": teaser_bars, "shift": semis, "shift_requested": shift, "time_ratio_b_hook": rb, "fold_k": kb, "hook_s": hook_s},
        "keys": _key_block(key_a, key_b, shift_b_material=semis, match=auto),
        "a_exit": exit_a, "b_hook": {"words": [u["text"] for u in ub[j0 : j1 + 1]], "source_samples": [int(q0), int(q1)]},
        "b_overlay": tmap, "b_vocal_gate": ginfo,
        "regions": [{"name": "A's vocal stops at a line end", "output": [keep - off_a, keep - off_a]},
                    {"name": "B's hook over A's instrumental (teaser)", "output": [o0, o1], "intended": True},
                    {"name": "B enters on its cue (own key)", "output": [t_x, t_x]}],
        "basis": "[P] teasing the next track's hook / acapella over the playing instrumental (B&B pdf p. 86); shift by TIV compatibility [S]; [H] teaser length",
    }
    return _finish(ctx, "hook_teaser", deck_a=deck_a, deck_b=deck_b, b_voice=ov,
                   t_s=t_s, t_sw=t_x, t_e=t_e, off_a=off_a, creative=creative, a_map=a_map, b_map=b_map, va_heard=va_heard,
                   vb_heard=vocb * sg[:, None], sg_info=sg_info, provenance=provenance, return_decks=return_decks,
                   overlay_parts={"b_hook": ov})


def loop_schedule(mode: str, loop_bars: int) -> list[float]:
    """Periodi (in battiti) delle ripetizioni: ``bar`` 4, ``half`` 2, ``beat`` 1, ``halving`` loop roll che si
    dimezza (2 battute: 2,2,1,1,1/2,1/2,1/4 x4; 4 battute: 4,4,2,2,1,1,1/2,1/2,1/4 x4) [P]."""
    total = 4 * loop_bars
    if mode == "halving" and loop_bars not in (2, 4):
        raise DJRenderError("halving needs 2 or 4 bars")
    if mode == "bar":
        return [4.0] * loop_bars
    if mode == "half":
        return [2.0] * (2 * loop_bars)
    if mode == "beat":
        return [1.0] * total
    if mode == "halving":
        seq = ([4.0, 4.0] if loop_bars == 4 else []) + [2.0, 2.0, 1.0, 1.0, 0.5, 0.5] + [0.25] * 4
        assert abs(sum(seq) - total) < 1e-9
        return seq
    raise DJRenderError(f"mode in {LOOP_MODES}")


def _a_chop(ctx, ua, exit_a: dict, cfg: MashupConfig, n_words: int = 1, min_len_s: float = 0.12) -> tuple[int, int, list[str]]:
    """Ultime ``n_words`` parole del verso che chiude A (campioni sorgente), allungate alla precedente se troppo corte."""
    sr = ctx.sr
    i = exit_a.get("unit_index")
    if i is None:
        cands = [k for k, u in enumerate(ua) if u["end"] <= exit_a["sample"]]
        if not cands:
            raise DJRenderError("no word of A to chop before the cue")
        i = cands[-1]
    j = max(0, i - n_words + 1)
    while j > 0 and (ua[i]["end"] - ua[j]["start"]) < min_len_s * sr and ua[j]["start"] - ua[j - 1]["end"] < 0.15 * sr:
        j -= 1
    s0 = ua[j]["start"] - int(0.01 * sr)
    s1 = ua[i]["end"] + int(round(cfg.word_pad_s * sr))
    return s0, s1, [u["text"] for u in ua[j : i + 1]]


def vocal_loop(audio_a, audio_b, *, grids, cue_out_a, cue_in_b, vocals_a, vocals_b, words_a=None, words_b=None, key_a=None,
               key_b=None, mode: str = "halving", loop_bars: int = 2, shift: Any = "auto", n_words: int = 1, tail: str = "echo",
               sample_rate: int = dj.DEFAULT_SAMPLE_RATE, config: Optional[MashupConfig] = None, provenance: Optional[dict] = None,
               return_decks: bool = False) -> DJRenderResult:
    """L'ultima parola (o le ultime ``n_words``) del verso con cui A chiude ripetuta sulla griglia di B dal suo
    downbeat d'ingresso per ``loop_bars`` battute (:func:`loop_schedule`), intonata a B, passa-alto 150 -> 1200 Hz e
    guadagno -0 -> -6 dB lungo il loop; coda ``echo`` (1/2 battito, 3 ripetizioni) o ``reverb`` sotto B."""
    cfg = config or MashupConfig()
    sr = int(sample_rate)
    if loop_bars not in (1, 2, 4) or tail not in ("echo", "reverb", "none"):
        raise DJRenderError("loop_bars in (1, 2, 4), tail echo|reverb|none")
    if mode == "halving" and loop_bars == 1:
        raise DJRenderError("halving needs 2 or 4 bars")
    ctx = _ctx(audio_a, audio_b, vocals_a, vocals_b, grids, cue_out_a, cue_in_b, sr, cfg)
    g = _bars(ctx)
    pre, post = int(round(ctx.cfg.pre_s * sr)), int(round(ctx.cfg.post_s * sr))
    X, sb = ctx.sa(), ctx.sb()
    k, r = fold_ratio(ctx.ta["bar_s"], ctx.tb["bar_s"], cfg.max_vocal_stretch)
    ua, ub = vocal_units(words_a, sr, ctx.va), vocal_units(words_b, sr, ctx.vb)
    exit_a = a_exit_point(ctx, ua, X, cfg, max_back_s=2 * ctx.ta["bar_s"])
    keep = exit_a["sample"]
    s0, s1, chop_words = _a_chop(ctx, ua, exit_a, cfg, n_words)
    S = min(keep, X)
    off_a = S - pre
    t_s, t_x = pre, X - off_a
    sched = loop_schedule(mode, loop_bars)
    loop_end = t_x + loop_bars * g["bar_b"]
    tail_len = {"echo": 2 * g["beat_b"], "reverb": g["bar_b"], "none": 0}[tail]
    t_e = loop_end
    n = t_e + 2 * g["bar_b"] + post
    auto = _auto_shift(ctx.va, ctx.grid["beats_a"], (s0 - g["beat_a"], s1), ctx.b - ctx.vb, ctx.grid["beats_b"],
                       (sb, sb + loop_bars * g["bar_b"]), k, sr, cfg, keys=(key_a, key_b))
    semis = _resolve_shift(shift, auto)
    frag = _windowed(_slice(ctx.va, s0, s1 - s0), int(round(cfg.fade_ms / 1000 * sr)))
    y, sinfo = stretch_pitch(frag, sr, r, semis, formant=True)
    loop = np.zeros((n, 2))
    reps, pos = [], 0.0
    for j, per in enumerate(sched):
        on = t_x + int(round(pos * g["beat_b"]))
        L = min(len(y), int(round(per * g["beat_b"])))
        piece = _windowed(y[:L], int(0.005 * sr)) * _db(-6.0 * pos / (4 * loop_bars))
        _place(loop, piece, on)
        reps.append({"repeat": j + 1, "clip_sample": on, "period_beats": per, "length_samples": int(L)})
        pos += per
    idx = np.arange(n)
    fc = 150.0 * (1200.0 / 150.0) ** np.clip((idx - t_x) / max(loop_end - t_x, 1), 0, 1)
    loop = fx.time_varying_filter(loop, sr, fc, "highpass", 2, 256)
    tl = np.zeros((n, 2))
    last = loop[t_x + int(round((pos - sched[-1]) * g["beat_b"])) : loop_end]
    if tail == "echo" and len(last):
        _place(tl, _echo(last, sr, g["beat_b"] // 2, 3, feedback=0.5, hpf0=400, hpf1=2000), loop_end - len(last))
    elif tail == "reverb" and len(last):
        from scipy.signal import fftconvolve

        ir = cr.reverb_ir(sr, rt60_s=2.0)
        wet = np.stack([fftconvolve(last[:, c], ir[:, c])[: len(last) + tail_len] for c in range(2)], 1)
        wet[: len(last)] *= _rise(len(last), 0, len(last), "lin")[:, None]
        _place(tl, wet * _db(-6.0), loop_end - len(last))
    tl *= (1.0 - (1.0 - _db(cfg.under_b_db)) * _rise(n, loop_end, loop_end + int(0.05 * sr)))[:, None]
    ins, vocb, gate, ginfo = _common_b(ctx, t_x, n, ub, loop_end, cfg, mid_cut=(t_x, loop_end))
    t_e = _extend_t_e(t_e, ginfo, g["bar_b"])
    tl = _duck(tl, gate, sr)
    deck_a, va_heard, _ = _a_deck(ctx, off_a, n, keep - off_a, t_x, cfg, mute_ms=exit_a.get("mute_ms"))
    sg, sg_info = _section(ctx, S, t_x, n, t_e)
    deck_b = (ins + vocb) * sg[:, None]
    voice = loop + tl
    a_map = [{"out_start": 0, "out_end": keep - off_a, "src_start": off_a, "rate": 1.0, "vocals": True, "what": "A full mix"},
             {"out_start": keep - off_a, "out_end": t_x, "src_start": keep, "rate": 1.0, "vocals": False, "what": "A instrumental"}]
    a_map += [{"out_start": rp["clip_sample"], "out_end": rp["clip_sample"] + rp["length_samples"], "src_start": s0, "rate": r,
               "vocals": True, "what": f"loop repeat {rp['repeat']} of {' '.join(chop_words)!r} ({rp['period_beats']:g} beat)"} for rp in reps]
    b_map = _b_main_map(t_x, n, sb, ginfo, "B at its own tempo")
    creative = {
        "params": {"mode": mode, "loop_bars": loop_bars, "shift": semis, "shift_requested": shift, "n_words": n_words, "tail": tail,
                   "time_ratio_a": r, "fold_k": k, "schedule_beats": sched, "hpf_hz": [150, 1200], "gain_db": [0, -6]},
        "keys": _key_block(key_a, key_b, shift_a_material=semis, match=auto), "a_exit": exit_a,
        "chop": {"words": chop_words, "source_samples": [int(s0), int(s1)], "stretch_pitch": sinfo}, "repeats": reps,
        "loop_grid": "B's beat grid from its entry downbeat (B at its own tempo): every repeat starts on a grid point",
        "b_vocal_gate": ginfo,
        "regions": [{"name": "A's vocal stops at a line end", "output": [keep - off_a, keep - off_a]},
                    {"name": f"vocal loop ({mode}) over B", "output": [t_x, loop_end], "intended": True},
                    {"name": f"{tail} tail under B", "output": [loop_end, loop_end + tail_len], "intended": True}],
        "basis": "[P] Crossfader '12 high energy DJ transitions' (1-beat loop on a word, halved to a stutter; loop over the next intro); "
                 "loop roll halving (Serato/rekordbox/Engine DJ Roll) [P]; tuned to B by TIV compatibility [S][H]",
    }
    return _finish(ctx, "vocal_loop", deck_a=deck_a, deck_b=deck_b, a_voice=voice,
                   t_s=t_s, t_sw=t_x, t_e=t_e, off_a=off_a, creative=creative, a_map=a_map, b_map=b_map, va_heard=va_heard,
                   vb_heard=vocb * sg[:, None], sg_info=sg_info, provenance=provenance, return_decks=return_decks,
                   overlay_parts={"loop": loop, "tail": tl})


def vocal_chop(audio_a, audio_b, *, grids, cue_out_a, cue_in_b, vocals_a, vocals_b, words_a=None, words_b=None, key_a=None,
               key_b=None, pattern: str = "332", bars: int = 2, n_words: int = 3, shift: Any = "auto",
               sample_rate: int = dj.DEFAULT_SAMPLE_RATE, config: Optional[MashupConfig] = None, provenance: Optional[dict] = None,
               return_decks: bool = False) -> DJRenderResult:
    """Le ultime ``n_words`` parole del verso con cui A chiude, tagliate parola per parola e suonate a turno sul
    ritmo ``pattern`` (3+3+2 in ottavi, o 3+3+3+3+4 in sedicesimi) della griglia di B per ``bars`` battute,
    intonate a B; ogni taglio dura al più il suo spazio (dissolvenza 5 ms)."""
    cfg = config or MashupConfig()
    sr = int(sample_rate)
    if pattern not in CHOP_PATTERNS or bars not in (1, 2, 4):
        raise DJRenderError(f"pattern in {tuple(CHOP_PATTERNS)}, bars in (1, 2, 4)")
    ctx = _ctx(audio_a, audio_b, vocals_a, vocals_b, grids, cue_out_a, cue_in_b, sr, cfg)
    g = _bars(ctx)
    pre, post = int(round(ctx.cfg.pre_s * sr)), int(round(ctx.cfg.post_s * sr))
    X, sb = ctx.sa(), ctx.sb()
    k, r = fold_ratio(ctx.ta["bar_s"], ctx.tb["bar_s"], cfg.max_vocal_stretch)
    ua, ub = vocal_units(words_a, sr, ctx.va), vocal_units(words_b, sr, ctx.vb)
    exit_a = a_exit_point(ctx, ua, X, cfg, max_back_s=2 * ctx.ta["bar_s"])
    keep = exit_a["sample"]
    i = exit_a.get("unit_index")
    if i is None:
        cands = [kk for kk, u in enumerate(ua) if u["end"] <= keep]
        if not cands:
            raise DJRenderError("no word of A to chop")
        i = cands[-1]
    words = ua[max(0, i - n_words + 1) : i + 1]
    S = min(keep, X)
    off_a = S - pre
    t_s, t_x = pre, X - off_a
    chop_end = t_x + bars * g["bar_b"]
    t_e = chop_end
    n = t_e + 2 * g["bar_b"] + post
    seg0, seg1 = words[0]["start"] - int(0.01 * sr), words[-1]["end"] + int(round(cfg.word_pad_s * sr))
    auto = _auto_shift(ctx.va, ctx.grid["beats_a"], (seg0 - g["beat_a"], seg1), ctx.b - ctx.vb, ctx.grid["beats_b"],
                       (sb, chop_end - t_x + sb), k, sr, cfg, keys=(key_a, key_b))
    semis = _resolve_shift(shift, auto)
    y, sinfo = stretch_pitch(_windowed(_slice(ctx.va, seg0, seg1 - seg0), int(0.005 * sr)), sr, r, semis, formant=True)
    pieces = []
    for w in words:
        a0 = int(round((w["start"] - int(0.01 * sr) - seg0) * r))
        a1 = int(round((w["end"] + int(0.02 * sr) - seg0) * r))
        pieces.append((y[max(0, a0) : max(a0 + 1, min(len(y), a1))], w["text"]))
    units = CHOP_UNITS[pattern]
    pulse = g["bar_b"] / units
    out = np.zeros((n, 2))
    hits, m = [], 0
    for bar in range(bars):
        for on, ln in CHOP_PATTERNS[pattern]:
            p, txt = pieces[m % len(pieces)]
            at = t_x + bar * g["bar_b"] + int(round(on * pulse))
            L = min(len(p), int(round(ln * pulse)) - int(0.004 * sr))
            if L > 32:
                _place(out, _windowed(p[:L], int(0.005 * sr)), at)
            hits.append({"clip_sample": at, "word": txt, "slot_pulses": ln, "length_samples": int(max(L, 0))})
            m += 1
    out = _hpf(out, sr, 120.0)
    ins, vocb, gate, ginfo = _common_b(ctx, t_x, n, ub, chop_end, cfg, mid_cut=(t_x, chop_end))
    t_e = _extend_t_e(t_e, ginfo, g["bar_b"])
    deck_a, va_heard, _ = _a_deck(ctx, off_a, n, keep - off_a, t_x, cfg, mute_ms=exit_a.get("mute_ms"))
    sg, sg_info = _section(ctx, S, t_x, n, t_e)
    deck_b = (ins + vocb) * sg[:, None]
    a_map = [{"out_start": 0, "out_end": keep - off_a, "src_start": off_a, "rate": 1.0, "vocals": True, "what": "A full mix"},
             {"out_start": keep - off_a, "out_end": t_x, "src_start": keep, "rate": 1.0, "vocals": False, "what": "A instrumental"}]
    a_map += [{"out_start": h["clip_sample"], "out_end": h["clip_sample"] + h["length_samples"], "src_start": None, "rate": r,
               "vocals": True, "what": f"chop {h['word']!r}"} for h in hits]
    b_map = _b_main_map(t_x, n, sb, ginfo, "B at its own tempo")
    creative = {
        "params": {"pattern": pattern, "bars": bars, "n_words": n_words, "shift": semis, "shift_requested": shift, "time_ratio_a": r, "fold_k": k},
        "keys": _key_block(key_a, key_b, shift_a_material=semis, match=auto), "a_exit": exit_a,
        "chops": {"words": [w["text"] for w in words], "source_samples": [int(seg0), int(seg1)], "stretch_pitch": sinfo}, "hits": hits,
        "b_vocal_gate": ginfo,
        "regions": [{"name": "A's vocal stops at a line end", "output": [keep - off_a, keep - off_a]},
                    {"name": f"vocal chops on {pattern} over B", "output": [t_x, chop_end], "intended": True}],
        "basis": "[T] Butler 2006 pdf pp. 97-99 (3+3+2: maximally even 3 attacks on 8 pulses); [T] Williams 2013 pdf p. 161 (a voice "
                 "chopped to one syllable and repeated in rhythm); [P] vocal chops with stems/pads; [H] pattern and lengths",
    }
    return _finish(ctx, "vocal_chop", deck_a=deck_a, deck_b=deck_b, a_voice=out,
                   t_s=t_s, t_sw=t_x, t_e=t_e, off_a=off_a, creative=creative, a_map=a_map, b_map=b_map, va_heard=va_heard,
                   vb_heard=vocb * sg[:, None], sg_info=sg_info, provenance=provenance, return_decks=return_decks,
                   overlay_parts={"chops": out})


def _bar_offset(src_sample: int, downbeats_s: np.ndarray, sr: int, beat: int) -> int:
    """Posto della frase nella sua battuta (campioni): negativo se inizia entro un battito prima di un downbeat (pickup)."""
    d = downbeats_s * sr
    nxt = d[d >= src_sample]
    if len(nxt) and nxt[0] - src_sample <= beat:
        return int(src_sample - nxt[0])
    prv = d[d <= src_sample]
    return int(src_sample - prv[-1]) if len(prv) else 0


def call_response(audio_a, audio_b, *, grids, cue_out_a, cue_in_b, vocals_a, vocals_b, hook_s: float, words_a=None, words_b=None,
                  key_a=None, key_b=None, rounds: int = 2, slot_bars: int = 2, shift: Any = "auto", min_line_s: float = 1.0,
                  sample_rate: int = dj.DEFAULT_SAMPLE_RATE, config: Optional[MashupConfig] = None, provenance: Optional[dict] = None,
                  return_decks: bool = False) -> DJRenderResult:
    """Botta e risposta sopra la base di B: spazi di ``slot_bars`` battute di B alternati, nei pari un verso di A
    (dal verso che parte al cue in poi, stirato e trasposto a B), nei dispari una frase dell'hook di B (dal suo
    ``hook_s``, al suo tempo e tonalità), ``rounds`` volte; ogni frase conserva il suo posto nella battuta (pickup)."""
    cfg = config or MashupConfig()
    sr = int(sample_rate)
    if rounds not in (1, 2) or slot_bars not in (1, 2):
        raise DJRenderError("rounds in (1, 2), slot_bars in (1, 2)")
    ctx = _ctx(audio_a, audio_b, vocals_a, vocals_b, grids, cue_out_a, cue_in_b, sr, cfg)
    g = _bars(ctx)
    pre, post = int(round(ctx.cfg.pre_s * sr)), int(round(ctx.cfg.post_s * sr))
    X, sb = ctx.sa(), ctx.sb()
    k, r = fold_ratio(ctx.ta["bar_s"], ctx.tb["bar_s"], cfg.max_vocal_stretch)
    ua, ub = vocal_units(words_a, sr, ctx.va), vocal_units(words_b, sr, ctx.vb)
    gap = int(round(cfg.phrase_gap_s * sr))
    lead = int(round(cfg.gate_lead_s * sr))
    pad = int(round(cfg.word_pad_s * sr))
    slot = slot_bars * g["bar_b"]

    min_len = int(round(min_line_s * sr))

    def lines(units, first_i, n_lines, max_len_out, ratio):
        """Versi consecutivi da ``first_i``; un verso più corto di ``min_line_s`` (es. un solo 'hey') si unisce al
        successivo finché sta nello spazio [H]."""
        out, i = [], first_i
        ends = set(line_ends(units, gap))
        while len(out) < n_lines and i < len(units):
            j = i
            best = None
            while j < len(units):
                if j in ends:
                    fits = (units[j]["end"] + pad - units[i]["start"] + lead) * ratio <= max_len_out
                    if fits:
                        best = j
                    if not fits or units[j]["end"] - units[i]["start"] >= min_len:
                        break
                j += 1
            if best is None:
                raise DJRenderError("a line does not fit its call/response slot")
            out.append((i, best))
            i = best + 1
        if len(out) < n_lines:
            raise DJRenderError("not enough lines for the call and response")
        return out

    i0 = a_line_from(ua, X, g["beat_a"], g["bar_a"], cfg, sr)
    calls = lines(ua, i0, rounds, slot - int(0.12 * sr), r)
    starts_b = [i for i in line_starts(ub, gap) if ub[i]["start"] >= int(hook_s * sr) - g["beat_b"]]
    if not starts_b:
        raise DJRenderError("no phrase of B at the hook")
    resp = lines(ub, starts_b[0], rounds, slot - int(0.12 * sr), 1.0)
    p0 = ua[calls[0][0]]["start"] - lead
    if p0 > X and i0 > 0 and ua[i0 - 1]["end"] > X - int(0.03 * sr):
        raise DJRenderError("A's previous word crosses the cue")
    S = min(p0, X)
    off_a = S - pre
    t_s, t_x = pre, X - off_a
    end = t_x + 2 * rounds * slot
    t_e = end
    n = t_e + 2 * g["bar_b"] + post
    a_first, a_last = ua[calls[0][0]]["start"] - lead, ua[calls[-1][1]]["end"] + pad
    auto = _auto_shift(ctx.va, ctx.grid["beats_a"], (a_first, a_last), ctx.b - ctx.vb, ctx.grid["beats_b"],
                       (sb, sb + 2 * rounds * slot), k, sr, cfg, keys=(key_a, key_b))
    semis = _resolve_shift(shift, auto)
    a_voice, b_voice = np.zeros((n, 2)), np.zeros((n, 2))
    placed, last_end = [], 0
    for m in range(2 * rounds):
        slot0 = t_x + m * slot
        if m % 2 == 0:
            i, j = calls[m // 2]
            s0, s1 = ua[i]["start"] - lead, ua[j]["end"] + pad
            off = int(round(_bar_offset(s0, ctx.da, sr, g["beat_a"]) * r))
            sig, tm = _transform_placed(ctx.va, s0, s1, s0, slot0 + off, r, semis, n, sr, cfg)
            a_voice += sig
            who, words = "A", [u["text"] for u in ua[i : j + 1]]
        else:
            i, j = resp[m // 2]
            s0, s1 = ub[i]["start"] - lead, ub[j]["end"] + pad
            off = _bar_offset(s0, ctx.db, sr, g["beat_b"])
            sig, tm = _transform_placed(ctx.vb, s0, s1, s0, slot0 + off, 1.0, 0, n, sr, cfg)
            b_voice += sig
            who, words = "B", [u["text"] for u in ub[i : j + 1]]
        c0, c1 = tm["clip_samples"]
        if c0 < last_end + int(0.08 * sr):
            raise DJRenderError("call and response phrases overlap")
        last_end = c1
        placed.append({"slot": m, "who": who, "words": words, "clip_samples": [c0, c1], "source_samples": tm["source_samples"],
                       "time_ratio": tm["time_ratio"], "semitones": tm["semitones"]})
    if last_end > end + g["beat_b"]:
        raise DJRenderError("the last phrase runs past the call and response")
    ins, vocb, gate, ginfo = _common_b(ctx, t_x, n, ub, max(end, last_end + int(0.15 * sr)), cfg, mid_cut=(t_x, end))
    t_e = _extend_t_e(t_e, ginfo, g["bar_b"])
    deck_a, va_heard, _ = _a_deck(ctx, off_a, n, p0 - off_a, t_x, cfg)
    sg, sg_info = _section(ctx, S, t_x, n, t_e)
    deck_b = (ins + vocb) * sg[:, None]
    a_map = [{"out_start": 0, "out_end": min(p0 - off_a, t_x), "src_start": off_a, "rate": 1.0, "vocals": True, "what": "A full mix"}]
    if p0 - off_a < t_x:
        a_map.append({"out_start": p0 - off_a, "out_end": t_x, "src_start": p0, "rate": 1.0, "vocals": False, "what": "A instrumental"})
    a_map += [{"out_start": p["clip_samples"][0], "out_end": p["clip_samples"][1], "src_start": p["source_samples"][0], "rate": r,
               "vocals": True, "what": f"call: {' '.join(p['words'])}"} for p in placed if p["who"] == "A"]
    b_map = [{"out_start": p["clip_samples"][0], "out_end": p["clip_samples"][1], "src_start": p["source_samples"][0], "rate": 1.0,
              "vocals": True, "what": f"response (B's hook, moved): {' '.join(p['words'])}"} for p in placed if p["who"] == "B"]
    b_map += _b_main_map(t_x, n, sb, ginfo, "B at its own tempo")
    creative = {
        "params": {"rounds": rounds, "slot_bars": slot_bars, "min_line_s": min_line_s, "shift": semis, "shift_requested": shift, "time_ratio_a": r, "fold_k": k, "hook_s": hook_s},
        "keys": _key_block(key_a, key_b, shift_a_material=semis, match=auto), "phrases": placed, "b_vocal_gate": ginfo,
        "regions": [{"name": f"slot {p['slot']} {p['who']}", "output": p["clip_samples"], "intended": True} for p in placed]
                   + [{"name": "A beat out / B beat in", "output": [t_x, t_x]}],
        "basis": "[P] Crossfader 'Wordplay' (two tracks trade lines like call and response, 4-beat loop + stems); [T] Katz 2012 pdf p. 97 "
                 "(call and response in Flash's routine); one voice at a time [E]; [H] slots of 1-2 bars",
    }
    return _finish(ctx, "call_response", deck_a=deck_a, deck_b=deck_b, a_voice=a_voice,
                   b_voice=b_voice, t_s=t_s, t_sw=t_x, t_e=t_e, off_a=off_a, creative=creative, a_map=a_map, b_map=b_map,
                   va_heard=va_heard, vb_heard=vocb * sg[:, None], sg_info=sg_info, provenance=provenance,
                   return_decks=return_decks, overlay_parts={"calls_a": a_voice, "responses_b": b_voice})


def pivot_target(key_a: Optional[str], key_b: Optional[str]) -> dict:
    """Glissando (semitoni, -5..+6) che porta la tonica di A sulla tonica di B, o sul relativo di B se i modi
    differiscono (l'accordo congelato di A, con la sua qualità, diventa diatonico in B) [T][H]."""
    ka, kb = cr.parse_key(key_a), cr.parse_key(key_b)
    if ka is None or kb is None:
        raise DJRenderError("pivot_freeze needs both keys")
    if ka[1] == kb[1]:
        target, why = kb[0], "tonic of B (same mode)"
    elif kb[1] == "major":
        target, why = (kb[0] + 9) % 12, "relative minor of B (A is minor)"
    else:
        target, why = (kb[0] + 3) % 12, "relative major of B (A is major)"
    d = (target - ka[0]) % 12
    if d > 6:
        d -= 12
    return {"semitones": int(d), "target_pc": cr.PITCH_CLASSES[target], "why": why, "key_a": key_a, "key_b": key_b}


def gliding_freeze(segment: np.ndarray, sr: int, semis_curve: np.ndarray, seed: int = 5, n_fft: int = 4096) -> np.ndarray:
    """Freeze spettrale (modulo STFT medio, fasi casuali, come ``dj_creative.spectral_freeze``) il cui spettro è
    trasposto frame per frame di ``semis_curve`` (semitoni per campione d'uscita): un accordo tenuto che scivola."""
    from scipy.signal import istft, stft

    x = _as_stereo(segment)
    n_out = len(semis_curve)
    rng = np.random.default_rng(seed)
    hop = n_fft // 4
    frames = int(math.ceil(n_out / hop)) + 4
    out = np.zeros((n_out, 2))
    fr = np.clip((np.arange(frames) * hop - n_fft).astype(int), 0, n_out - 1)
    factor = 2.0 ** (np.asarray(semis_curve, float)[fr] / 12.0)
    for ch in range(2):
        f, _, Z = stft(x[:, ch], fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
        mag = np.abs(Z).mean(axis=1)
        M = np.stack([np.interp(f / q, f, mag, left=0.0, right=0.0) for q in factor], axis=1)
        ph = rng.uniform(-np.pi, np.pi, M.shape)
        _, y = istft(M * np.exp(1j * ph), fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
        y = y[n_fft : n_fft + n_out]
        out[: len(y), ch] = y
    r_in, r_out = _rms(x), _rms(out)
    return out * (r_in / r_out if r_out > 0 else 0.0)


def pivot_freeze(audio_a, audio_b, *, grids, cue_out_a, cue_in_b, vocals_a, vocals_b, key_a: str, key_b: str, words_a=None,
                 words_b=None, hold_beats: int = 4, glide_beats: int = 4, release_beats: int = 2, a_fade_beats: int = 2,
                 riser: bool = False, level_db: float = -6.0, sample_rate: int = dj.DEFAULT_SAMPLE_RATE,
                 config: Optional[MashupConfig] = None, provenance: Optional[dict] = None, return_decks: bool = False) -> DJRenderResult:
    """Freeze dell'ultimo battito di A (strumentale, componente armonica HPSS) che entra mentre A sfuma negli
    ultimi ``a_fade_beats`` battiti, resta ``hold_beats`` e scivola in ``glide_beats`` di :func:`pivot_target`
    semitoni (tonica di A -> tonica o relativo di B); B entra pulito sul suo downbeat alla fine del glissando e il
    freeze si spegne sotto B in ``release_beats``. Facoltativo un riser sulla tonica di B."""
    cfg = config or MashupConfig()
    sr = int(sample_rate)
    ctx = _ctx(audio_a, audio_b, vocals_a, vocals_b, grids, cue_out_a, cue_in_b, sr, cfg)
    g = _bars(ctx)
    pre, post = int(round(ctx.cfg.pre_s * sr)), int(round(ctx.cfg.post_s * sr))
    X, sb = ctx.sa(), ctx.sb()
    ua, ub = vocal_units(words_a, sr, ctx.va), vocal_units(words_b, sr, ctx.vb)
    fade = a_fade_beats * g["beat_a"]
    exit_a = a_exit_point(ctx, ua, X - fade, cfg, max_back_s=2 * ctx.ta["bar_s"])
    keep = exit_a["sample"]
    S = min(keep, X - fade)
    off_a = S - pre
    t_s, t_x = pre, X - off_a
    hold, glide = hold_beats * g["beat_a"], glide_beats * g["beat_a"]
    t_sw = t_x + hold + glide
    t_e = t_sw + release_beats * g["beat_b"]
    n = t_e + post
    pv = pivot_target(key_a, key_b)
    cap = ctx.a[X - g["beat_a"] : X] - ctx.va[X - g["beat_a"] : X]
    try:
        import librosa

        cap = np.stack([librosa.effects.harmonic(cap[:, c].astype(np.float32), margin=2.0) for c in range(2)], 1).astype(np.float64)
    except Exception:  # pragma: no cover - librosa is in the venv
        pass
    f0 = t_x - fade
    curve = np.zeros(n - f0)
    gl0 = t_x + hold - f0
    curve[gl0:] = pv["semitones"] * np.clip((np.arange(n - f0 - gl0)) / max(glide, 1), 0, 1)
    fr = gliding_freeze(cap, sr, curve)
    env = _rise(n - f0, 0, fade, "cos") * (1.0 - (1.0 - _db(cfg.under_b_db)) * _rise(n - f0, t_sw - f0 - int(0.02 * sr), t_sw - f0))
    env *= 1.0 - _rise(n - f0, t_sw - f0, t_e - f0, "cos")
    freeze = np.zeros((n, 2))
    freeze[f0:] = fr * env[:, None]
    deck_a, va_heard, ga = _a_deck(ctx, off_a, n, keep - off_a, t_x, cfg, mute_ms=exit_a.get("mute_ms"), fade=(t_x - fade, t_x))
    a_ref = _rms(_slice(ctx.a, X - fade - 2 * g["bar_a"], 2 * g["bar_a"]))
    freeze *= a_ref / max(_rms(freeze[t_x : t_sw]), 1e-9) * _db(level_db)
    rz = None
    if riser:
        kb = cr.parse_key(key_b)
        root = 48 + kb[0]
        rz = np.zeros((n, 2))
        seg = cr.synth_riser(t_sw - t_x, sr, root, glide_semitones=12.0)
        seg *= _rise(t_sw - t_x, 0, t_sw - t_x, "lin")[:, None] ** 2
        rz[t_x:t_sw] = seg * a_ref * _db(level_db - 6.0)
        freeze = freeze + rz
    ins, vocb, gate, ginfo = _common_b(ctx, t_sw, n, ub, t_sw - g["beat_b"], cfg)
    sg, sg_info = _section(ctx, S, t_sw, n, t_e)
    deck_b = (ins + vocb) * sg[:, None]
    a_map = [{"out_start": 0, "out_end": keep - off_a, "src_start": off_a, "rate": 1.0, "vocals": True, "what": "A full mix"},
             {"out_start": keep - off_a, "out_end": t_x, "src_start": keep, "rate": 1.0, "vocals": False, "what": "A instrumental fading"}]
    b_map = _b_main_map(t_sw, n, sb, ginfo, "B at its own tempo")
    creative = {
        "params": {"hold_beats": hold_beats, "glide_beats": glide_beats, "release_beats": release_beats, "a_fade_beats": a_fade_beats,
                   "riser": riser, "level_db": level_db, "gap_bars": (hold_beats + glide_beats) / 4.0},
        "keys": dict(_key_block(key_a, key_b), pivot=pv), "a_exit": exit_a,
        "freeze": {"captured_a_source_samples": [X - g["beat_a"], X], "harmonic_component": "librosa HPSS (margin 2) of A instrumental",
                   "glide": "spectrum rescaled in frequency frame by frame (STFT 4096, random phases)"},
        "b_vocal_gate": ginfo,
        "regions": [{"name": "A fades, freeze fades in", "output": [t_x - fade, t_x]}, {"name": "freeze holds A's chord", "output": [t_x, t_x + hold], "intended": True},
                    {"name": f"freeze glides {pv['semitones']:+d} st to {pv['target_pc']}", "output": [t_x + hold, t_sw], "intended": True},
                    {"name": "B enters clean", "output": [t_sw, t_sw]}, {"name": "freeze releases under B", "output": [t_sw, t_e]}],
        "basis": "[T] pivot / common-chord modulation (the held chord moves to the arriving key); [P] freeze/pad across a change; [H] glide length and target",
    }
    return _finish(ctx, "pivot_freeze", deck_a=deck_a, deck_b=deck_b, overlay=freeze,
                   t_s=t_s, t_sw=t_sw, t_e=t_e, off_a=off_a, creative=creative, a_map=a_map, b_map=b_map, va_heard=va_heard,
                   vb_heard=vocb * sg[:, None], sg_info=sg_info, provenance=provenance, return_decks=return_decks,
                   overlay_parts={"freeze": freeze})


def reverse_swell(audio_a, audio_b, *, grids, cue_out_a, cue_in_b, vocals_a, vocals_b, words_a=None, words_b=None, key_a=None,
                  key_b=None, swell_beats: int = 4, rt60_s: float = 2.0, level_db: float = -4.0,
                  sample_rate: int = dj.DEFAULT_SAMPLE_RATE, config: Optional[MashupConfig] = None, provenance: Optional[dict] = None,
                  return_decks: bool = False) -> DJRenderResult:
    """Riverbero inverso della prima parola della prima frase di B dopo il cue (pickup compreso): la parola convoluta
    con la risposta all'impulso rovesciata (RT60 ``rt60_s``; equivale a rovesciare, riverberare e rovesciare di nuovo),
    tenuto solo ciò che precede l'attacco; cresce per ``swell_beats`` battiti di A sopra A (passa-alto crescente, voce
    di A già ferma a un fine verso) e finisce esattamente sull'attacco della parola vera di B; B entra sul suo downbeat."""
    from scipy.signal import fftconvolve

    cfg = config or MashupConfig()
    sr = int(sample_rate)
    if swell_beats not in (2, 4, 8):
        raise DJRenderError("swell_beats in (2, 4, 8)")
    ctx = _ctx(audio_a, audio_b, vocals_a, vocals_b, grids, cue_out_a, cue_in_b, sr, cfg)
    g = _bars(ctx)
    pre, post = int(round(ctx.cfg.pre_s * sr)), int(round(ctx.cfg.post_s * sr))
    X, sb = ctx.sa(), ctx.sb()
    ua, ub = vocal_units(words_a, sr, ctx.va), vocal_units(words_b, sr, ctx.vb)
    gap = int(round(cfg.phrase_gap_s * sr))
    st = [i for i in line_starts(ub, gap) if sb - g["beat_b"] <= ub[i]["start"] <= sb + 2 * g["bar_b"]]
    if not st:
        raise DJRenderError("B has no phrase start within 2 bars of its cue")
    w = ub[st[0]]
    L = swell_beats * g["beat_a"]
    w_rel = w["start"] - sb  # B unstretched: clip = t_x + w_rel
    deadline_src = X + w_rel - L - int(0.1 * sr)
    exit_a = a_exit_point(ctx, ua, min(X, deadline_src), cfg, max_back_s=2 * ctx.ta["bar_s"])
    keep = exit_a["sample"]
    S = min(keep, X + w_rel - L)
    off_a = S - pre
    t_s, t_x = pre, X - off_a
    t_w = t_x + w_rel
    t_e = max(t_x, t_w) + g["bar_b"]
    n = t_e + post
    frag = _windowed(_slice(ctx.vb, w["start"] - int(0.005 * sr), (w["end"] - w["start"]) + int(0.06 * sr)), int(0.005 * sr))
    ir = cr.reverb_ir(sr, rt60_s=rt60_s)
    # reverse reverb: reverse the word, reverb it (wet only), reverse back = the word convolved with the reversed IR;
    # everything before the word's onset (index len(ir) - 1) is the swell that grows into it
    wet = np.stack([fftconvolve(frag[:, c], ir[::-1, c]) for c in range(2)], 1)[: len(ir) - 1]
    rev = wet[-L:] if len(wet) >= L else np.vstack([np.zeros((L - len(wet), 2)), wet])
    rev = _hpf(rev, sr, 250.0)
    rev *= (_rise(L, 0, L, "lin") ** 2)[:, None]
    rev[-int(0.01 * sr):] *= np.linspace(1, 0, int(0.01 * sr))[:, None]
    swell = np.zeros((n, 2))
    _place(swell, rev, t_w - L)
    a_ref = _rms(_slice(ctx.a, X - 2 * g["bar_a"], 2 * g["bar_a"]))
    swell *= a_ref / max(_rms(swell[t_w - L // 2 : t_w]), 1e-9) * _db(level_db)
    ins, vocb, gate, ginfo = _common_b(ctx, t_x, n, ub, t_w - int(round(cfg.gate_lead_s * sr)) - 1, cfg)
    deck_a, va_heard, _ = _a_deck(ctx, off_a, n, keep - off_a, t_x, cfg, mute_ms=exit_a.get("mute_ms"), filter_from=min(t_x, t_w - L))
    sg, sg_info = _section(ctx, S, t_x, n, t_e)
    deck_b = (ins + vocb) * sg[:, None]
    a_map = [{"out_start": 0, "out_end": keep - off_a, "src_start": off_a, "rate": 1.0, "vocals": True, "what": "A full mix"},
             {"out_start": keep - off_a, "out_end": t_x, "src_start": keep, "rate": 1.0, "vocals": False, "what": "A instrumental, high-passed"}]
    b_map = [{"out_start": t_w - L, "out_end": t_w, "src_start": None, "rate": -1.0, "vocals": True, "what": f"reversed reverb of B's word {w['text']!r}"},
             *_b_main_map(t_x, n, sb, ginfo, "B at its own tempo (vocal pickup before the downbeat if any)")]
    creative = {
        "params": {"swell_beats": swell_beats, "rt60_s": rt60_s, "level_db": level_db},
        "keys": _key_block(key_a, key_b), "a_exit": exit_a,
        "b_word": {"text": w["text"], "source_samples": [int(w["start"]), int(w["end"])], "clip_onset_sample": int(t_w),
                   "pickup_before_downbeat_s": max(0.0, -w_rel / sr)},
        "b_vocal_gate": ginfo,
        "regions": [{"name": "A's vocal stops at a line end", "output": [keep - off_a, keep - off_a]},
                    {"name": "reverse reverb swell of B's first word", "output": [t_w - L, t_w], "intended": True},
                    {"name": "B's word lands", "output": [t_w, t_w]}],
        "basis": "[P] reverse reverb ('reverse swell') into a vocal entry, studio/DJ edit practice; [H] length and level",
    }
    return _finish(ctx, "reverse_swell", deck_a=deck_a, deck_b=deck_b, overlay=swell, b_voice_heard=swell,
                   t_s=t_s, t_sw=t_x, t_e=t_e, off_a=off_a, creative=creative, a_map=a_map, b_map=b_map, va_heard=va_heard,
                   vb_heard=vocb * sg[:, None], sg_info=sg_info, provenance=provenance, return_decks=return_decks,
                   overlay_parts={"swell": swell})


# ------------------------------------------------------------------ dispatcher
def render_mashup(technique: str, audio_a, audio_b, **kwargs) -> DJRenderResult:
    fns = {"acapella_mashup": acapella_mashup, "beat_swap": beat_swap, "hook_teaser": hook_teaser, "vocal_loop": vocal_loop,
           "vocal_chop": vocal_chop, "call_response": call_response, "pivot_freeze": pivot_freeze, "reverse_swell": reverse_swell}
    if technique not in fns:
        raise DJRenderError(f"technique must be one of {TECHNIQUES}")
    return fns[technique](audio_a, audio_b, **kwargs)


def qa_inputs(result: DJRenderResult) -> dict:
    """Come ``dj_creative.qa_inputs`` (deck, voci udibili già con tutti i guadagni)."""
    return cr.qa_inputs(result)


def config_dict(cfg: MashupConfig) -> dict:
    return asdict(cfg)


# ------------------------------------------------------------- sound affinity
VOWELS = set("aeiouy")


def _initial_consonant(w: str) -> str:
    """Prima consonante *pronunciata* (ortografia inglese approssimata [H]: kn/gn -> n, wr -> r, ps -> s, ph -> f,
    c/q -> k; ci/ce -> s)."""
    for cl, rep in (("kn", "n"), ("gn", "n"), ("wr", "r"), ("ps", "s"), ("ph", "f"), ("ch", "c"), ("sh", "s")):
        if w.startswith(cl):
            w = rep + w[len(cl):]
            break
    if not w or w[0] in VOWELS:
        return ""
    if w[0] == "c" and len(w) > 1 and w[1] in "eiy":
        return "s"
    return {"c": "k", "q": "k"}.get(w[0], w[0])


def sound_affinity(wa: str, wb: str, lang_a: Optional[str] = None, lang_b: Optional[str] = None) -> Optional[dict]:
    """Legame di suono tra l'ultima parola di A e la prima di B: prima ``dj_creative.word_match`` (uguale, lemma, rima,
    Soundex), poi due legami più deboli [H] suggeriti dai voti dell'utente (21/10/2026: "A finisce con 'a-a-a-a' e B
    attacca con un suono affine"; Butterfly Effect -> Knife Talk, 'k' e 'k'): **vocalizzo** (entrambe solo vocali con
    h/w/y, es. "ooh" / "ah", 0,4) e **allitterazione** (stessa prima consonante pronunciata, c/k/q uniti, 0,35)."""
    m = cr.word_match(wa, wb, lang_a, lang_b)
    if m:
        return m
    a, b = cr.normalize_word(wa), cr.normalize_word(wb)
    if not a or not b:
        return None
    voc = lambda w: len(w) >= 2 and set(w) <= (VOWELS | set("hw")) and any(c in "aeiou" for c in w)  # noqa: E731
    if voc(a) and voc(b):
        return {"type": "vocalise", "score": 0.4, "a": a, "b": b}
    ca, cb = _initial_consonant(a), _initial_consonant(b)
    if ca and ca == cb and len(a) >= 3 and len(b) >= 3 and a not in cr.STOPWORDS and b not in cr.STOPWORDS:
        return {"type": "alliteration", "score": 0.35, "a": a, "b": b, "consonant": ca}
    return None


def find_sound_links(words_a, words_b, a_cues_s, b_cues_s, *, sr: int, bar_a_s: float, bar_b_s: float, lang_a=None, lang_b=None,
                     min_gap_s: float = 0.15, min_confidence: float = 0.3, top_k: int = 10) -> list[dict]:
    """Come ``dj_creative.find_lyric_links`` (stesse finestre: ultime 3 parole di A entro 2 battute prima del cue, prime 4
    di B da 1/4 di battuta prima a 2 battute dopo) con :func:`sound_affinity` come criterio."""
    out = []
    for X in a_cues_s:
        ca = sorted((w for w in words_a if X - 2 * bar_a_s <= w["start_sample"] / sr and w["end_sample"] / sr <= X - min_gap_s),
                    key=lambda w: w["start_sample"])[-3:]
        for Y in b_cues_s:
            cb = sorted((w for w in words_b if Y - 0.25 * bar_b_s <= w["start_sample"] / sr <= Y + 2 * bar_b_s),
                        key=lambda w: w["start_sample"])[:4]
            for ka, wa in enumerate(ca):
                for kb, wb in enumerate(cb):
                    m = sound_affinity(wa.get("text", ""), wb.get("text", ""), lang_a, lang_b)
                    if not m:
                        continue
                    conf = min(float(wa.get("confidence") or 0.0), float(wb.get("confidence") or 0.0))
                    if conf < min_confidence:
                        continue
                    last, first = ka == len(ca) - 1, kb == 0
                    score = m["score"] * math.sqrt(conf) * (1.1 if last else 1.0) * (1.05 if first else 1.0)
                    out.append({"a_cue_s": float(X), "b_cue_s": float(Y), "a_word": dict(wa), "b_word": dict(wb), "match": m,
                                "asr_confidence_min": conf, "a_word_is_last": last, "b_word_is_first": first, "score": score})
    out.sort(key=lambda r: -r["score"])
    return out[:top_k]
