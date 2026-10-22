"""Transizioni e mashup "creativi" da DJ: loop, acapella, vocal swap, build, nota ponte,
freeze, reverb, backspin, tape stop, lyric link.

Dal 21/10/2026 (libri, ``conoscenza_mix/libri/INTEGRAZIONE_PROPOSTE_2026-10-21.md``) anche
``halving_build`` (loop di A che si dimezza fino al downbeat di B: Margulis 2014 pp. 23, 44) e
``break_before_entry`` (stop, beat-mute sulla punchline o fill di rullante prima dell'uno di B:
Huron 2006 pp. 248-249, 314-316; Katz 2012 p. 106; Krims 2000 pp. 137-139).

Complemento di :mod:`musicmix.dj_render` (cut, blend) e :mod:`musicmix.dj_effects`
(echo out, filter out, drop in). Come quei moduli, qui **non si scelgono coppie
né cue**: ``cue_out_a`` e ``cue_in_b`` arrivano già su un downbeat (tolleranza
``snap_tolerance_s``); la scelta della tecnica è in :mod:`musicmix.dj_policy`.
Catalogo, fonti e limiti: ``conoscenza_mix/_meta/TECNICHE_CREATIVE_2026-10-21.md``.

Riuso (solo funzioni pubbliche, nessuna modifica a ``dj_render``/``dj_effects``):

* beatmatching di B: ``dj_render.render_transition(kind="blend",
  cue_convention="start", return_decks=True)`` e il suo deck
  ``b_stretched_raw`` (time map per battuta di Rubber Band R3, ogni downbeat di
  B sul downbeat di A); lo stem vocale di B passa per la stessa chiamata con le
  stesse griglie, quindi voce e strumentale stretchati restano allineati;
* loudness BS.1770 (``integrated_lufs``), limitatore true-peak
  (``true_peak_limit``), EQ a 3 bande a ricostruzione esatta (``split_bands``),
  section gain match (``plan_section_gain``), attività vocale da stem
  (``vocal_activity``); da ``dj_effects``: ``time_varying_filter``,
  ``bar_and_beat``, ``snap_downbeat``.

Convenzioni comuni (stesse di dj_render/dj_effects):

* A e B normalizzati a ``target_lufs`` (integrato sull'intero brano); gli stem
  vocali ricevono lo stesso guadagno del loro mix; strumentale = mix − stem
  vocale Demucs (come l'uscita ``no_vocals`` a due stem) [S: HT-Demucs];
* tutto è sulla griglia: inizio/fine dei loop, ingressi, swap e build cadono su
  downbeat o battiti della griglia (regolarizzata) del deck che suona;
* B entra sempre su un suo downbeat: o beatmatchato (tecniche con
  sovrapposizione, ``max_stretch`` ±8% come dj_render), o pulito al suo tempo
  (tecniche senza sovrapposizione), con micro-fade che *termina* sul downbeat;
* overlay sintetici o derivati (snare roll, riser, pad, freeze, riverbero, eco
  della parola) sono tarati sul livello di A prima della transizione e, se
  serve, ridotti finché il massimo della loudness momentary (400 ms) nella
  regione non supera A prima di ``overlay_margin_db`` (stessa misura del
  controllo ``loudness_jump`` di dj_qa);
* uscita: limitatore true-peak a ``ceiling_dbtp``; clip d'ascolto = 10 s prima
  dell'inizio della transizione + transizione intera + 10 s dopo la fine
  (AGENTS.md, eccezione del 20/10/2026 per le transizioni lunghe).

Metadati: stesso schema di dj_render (``kind``, ``overlap``, ``switch``,
``stretch``, ``loudness``, ``limiter``, ``clip``, ``section_gain``, ``config``)
più ``creative`` (tecnica, parametri, regioni in campioni della clip, base
della scelta), ``exit_effect`` (``echo``/``filter``/``reverb`` solo se A esce
davvero attraverso quell'effetto: è la chiave che dj_qa usa per l'esenzione dal
controllo "verso tagliato") e le mappe ``a_map``/``b_map`` clip → sorgente per
portare parole ASR e stem nelle coordinate della clip (:func:`words_in_clip`).
I deck restituiti (``return_decks=True``) contengono ``vocals_a_heard`` e
``vocals_b_heard``: la voce effettivamente udibile di ciascun deck nella clip
(dopo fader, mute e duck), da passare a ``dj_qa.evaluate_render`` con
``gains`` unitari.

Tutte le forme, durate e livelli sono pratica DJ [P] o ipotesi di progetto
[H] (TECNICHE_CREATIVE §2), non stime da mix reali: nessuno studio empirico
misura queste tecniche nel rap (REVISIONE_DJ_2026-10-20 §3).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
import shutil
import subprocess
import tempfile
import unicodedata
import os
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from musicmix import dj_effects as fx
from musicmix import dj_render as dj
from musicmix.dj_render import DJRenderError, DJRenderResult, SectionGainConfig

DJ_CREATIVE_VERSION = "musicmix.dj_creative.2"  # .2: halving_build, break_before_entry (libri, 21/10/2026)
TECHNIQUES = (
    "loop_extend", "acapella_over", "vocal_swap", "build_into_chorus", "note_bridge",
    "echo_freeze", "reverb_wash", "backspin", "tape_stop", "lyric_link", "halving_build", "break_before_entry",
)
OVERLAP_TECHNIQUES = ("loop_extend", "acapella_over", "vocal_swap")  # need B beatmatched
LOOP_BARS = (1, 2)
OVERLAP_BARS = (4, 8)
BUILD_BARS = (1, 2, 4)
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)
MINOR_STEPS = (0, 2, 3, 5, 7, 8, 10)  # natural minor


@dataclass(frozen=True)
class DJCreativeConfig:
    target_lufs: float = -14.0
    ceiling_dbtp: float = -1.0
    pre_s: float = 10.0
    post_s: float = 10.0
    snap_tolerance_s: float = 0.03
    max_bar_deviation: float = 0.12
    crossfade_ms: float = 10.0  # B entry / A exit micro fade, ends on the downbeat
    limiter_hold_ms: float = 20.0
    # mixer (same values as dj_render)
    low_hz: float = 200.0
    high_hz: float = 2500.0
    bass_swap_ms: float = 20.0
    high_floor_db: float = -12.0
    # beatmatching (dj_render)
    max_stretch: float = 0.08
    stretch_mode: str = "grid"
    stretch_backend: str = "auto"
    # stems
    vocal_mute_ms: float = 30.0  # A vocal removed (mix - stem) with this ramp ending on the downbeat
    vocal_active_cover: float = 0.1  # loop source "mix" only if A's vocal covers <= 10% of the loop
    loop_xfade_ms: float = 10.0  # loop seam: loop tail -> pre-loop material, ends on the downbeat
    # overlays
    overlay_margin_db: float = 1.0  # region momentary max <= A before + margin (dj_qa warns at +2, fails at +3)
    tail_under_b_db: float = -12.0
    # section gain match (dj_render), enabled as in runs/dj_gallery_v2
    section_gain: SectionGainConfig = field(default_factory=lambda: SectionGainConfig(enabled=True))
    # integrated LUFS of the whole source tracks, when the audio passed is an excerpt (memory):
    # the normalization gain stays the one of the whole track (dj_render convention)
    lufs_a_override: Optional[float] = None
    lufs_b_override: Optional[float] = None
    # normalize the caller's float64 (frames, 2) arrays in place instead of copying them (memory);
    # only for callers that do not reuse the arrays after the call
    inplace_inputs: bool = False


# --------------------------------------------------------------------- helpers


def _as_stereo(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return x[:, :2]


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    x = _as_stereo(x)
    if len(x) >= n:
        return x[:n].copy()
    return np.vstack([x, np.zeros((n - len(x), 2))])


def _slice(x: np.ndarray, start: int, n: int) -> np.ndarray:
    out = np.zeros((n, x.shape[1]))
    lo, hi = max(0, start), min(len(x), start + n)
    if hi > lo:
        out[lo - start : hi - start] = x[lo:hi]
    return out


def _rise(n: int, i0: int, i1: int, shape: str = "lin") -> np.ndarray:
    """0 before ``i0``, 1 from ``i1``; ``lin``, ``sin`` (quarter sine) or ``db`` handled by caller."""
    u = np.clip((np.arange(n) - i0) / max(i1 - i0, 1), 0.0, 1.0)
    if shape == "sin":
        return np.sin(0.5 * np.pi * u)
    if shape == "cos":  # raised cosine 0 -> 1
        return 0.5 - 0.5 * np.cos(np.pi * u)
    return u


def _fall_eqpow(n: int, i0: int, i1: int) -> np.ndarray:
    """1 before ``i0``, quarter cosine to 0 at ``i1``, exactly 0 after."""
    u = np.clip((np.arange(n) - i0) / max(i1 - i0, 1), 0.0, 1.0)
    g = np.cos(0.5 * np.pi * u)
    g[np.arange(n) >= i1] = 0.0
    return g


def _db(x: float) -> float:
    return 10.0 ** (x / 20.0)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, float) ** 2))) if len(x) else 0.0


def _k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    import pyloudnorm

    y = _as_stereo(x).copy()
    meter = pyloudnorm.Meter(sr)
    for stage in meter._filters.values():  # BS.1770 K-weighting (pyloudnorm's own filter stages)
        for ch in range(y.shape[1]):
            y[:, ch] = stage.apply_filter(y[:, ch])
    return y


def momentary_max_lufs(x: np.ndarray, sr: int, s0: int, s1: int, kw: Optional[np.ndarray] = None) -> Optional[float]:
    """Massimo della loudness momentary (400 ms, passo 100 ms, K-weighting BS.1770) in [s0, s1)."""
    y = _k_weight(x, sr) if kw is None else kw
    win, hop = int(round(0.4 * sr)), int(round(0.1 * sr))
    s0, s1 = max(0, int(s0)), min(len(y), int(s1))
    if s1 - s0 < win:
        return None
    c = np.r_[0.0, np.cumsum((y[s0:s1] ** 2).sum(axis=1))]
    starts = np.arange(0, s1 - s0 - win + 1, hop)
    ms = (c[starts + win] - c[starts]) / win
    ms = ms[ms > 0]
    return float(-0.691 + 10.0 * np.log10(ms.max())) if len(ms) else None


def trim_overlay(base: np.ndarray, overlay: np.ndarray, sr: int, region: tuple[int, int], ref: tuple[int, int],
                 margin_db: float) -> tuple[float, dict]:
    """Guadagno (<= 1) per ``overlay`` tale che il massimo momentary di base+overlay nella
    ``region`` non superi quello di ``base`` in ``ref`` (A prima) di più di ``margin_db``.

    K-weighting lineare: K(base + g*ov) = K(base) + g*K(ov), quindi si cerca g per bisezione.
    """
    kb, ko = _k_weight(base, sr), _k_weight(overlay, sr)
    ref_max = momentary_max_lufs(base, sr, *ref, kw=kb)
    info = {"reference_max_momentary_lufs": ref_max, "margin_db": margin_db}
    if ref_max is None or not np.any(overlay):
        info.update(gain_db=0.0, status="not measured")
        return 1.0, info
    limit = ref_max + margin_db

    def peak(g):
        return momentary_max_lufs(None, sr, *region, kw=kb + g * ko)

    p1 = peak(1.0)
    info["region_max_before_trim_lufs"] = p1
    if p1 is None or p1 <= limit:
        info.update(gain_db=0.0, status="within margin")
        return 1.0, info
    p0 = peak(0.0)
    if p0 is not None and p0 > limit:  # the decks alone exceed: keep the overlay quiet (-12 dB)
        info.update(gain_db=-12.0, status="decks alone above the margin: overlay -12 dB", region_max_decks_only_lufs=p0)
        return _db(-12.0), info
    lo, hi = 0.0, 1.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if (peak(mid) or -99.0) <= limit:
            lo = mid
        else:
            hi = mid
    g = max(lo, 1e-4)
    info.update(gain_db=20.0 * math.log10(g), status="trimmed", region_max_after_trim_lufs=peak(g))
    return g, info


# ------------------------------------------------------------------------ keys


def parse_key(name: Optional[str]) -> Optional[tuple[int, str]]:
    """``'B minor'``/``'Bm'``/``'Bb major'`` -> (pitch class 0..11, 'major'|'minor'); None se ignoto."""
    if not name:
        return None
    from musicmix import dj_select as djs

    try:
        tonic, mode = djs.normalize_key(str(name))
    except (ValueError, IndexError):
        return None
    return PITCH_CLASSES.index(tonic), mode


def key_name(pc: int, mode: str) -> str:
    return f"{PITCH_CLASSES[pc % 12]} {mode}"


def camelot_of(pc: int, mode: str) -> str:
    from musicmix import dj_select as djs

    return djs.camelot(key_name(pc, mode))


def key_relation(key_a: Optional[str], key_b: Optional[str], shift_b: int = 0) -> str:
    """Relazione Camelot di A con B trasposto di ``shift_b`` semitoni ('unknown' se manca una chiave)."""
    from musicmix import dj_select as djs

    ka, kb = parse_key(key_a), parse_key(key_b)
    if ka is None or kb is None:
        return "unknown"
    return djs.camelot_relation(camelot_of(*ka), camelot_of((kb[0] + shift_b) % 12, kb[1]))["relation"]


def auto_key_shift(key_a: Optional[str], key_b: Optional[str], max_semitones: int = 1) -> Optional[int]:
    """Trasposizione di B (0, +-1, ...) che rende la coppia stessa/adiacente/relativa.

    Preferisce 0, poi |s| crescente (Kim 2020 §6.2 [E]: il 94,3% delle trasposizioni
    reali è di un semitono; Vande Veire 2018 [S]: ±1 con pitch shift). None se nessuna.
    """
    order = [0] + [s for k in range(1, max_semitones + 1) for s in (k, -k)]
    for s in order:
        if key_relation(key_a, key_b, s) in ("same", "adjacent", "relative"):
            return s
    return None


def scale_pcs(pc: int, mode: str) -> set[int]:
    return {(pc + s) % 12 for s in (MINOR_STEPS if mode == "minor" else MAJOR_STEPS)}


def bridge_chord(key_a: Optional[str], key_b: Optional[str], shift_b: int = 0) -> dict:
    """Accordo ponte con note comuni alle due scale (diatoniche naturali) [H].

    Ordine: triade di tonica di B, triade di tonica di A, altre triadi diatoniche di B
    interamente comuni, bicordo tonica+quinta di B, tonica di B. Restituisce pitch
    class e voicing MIDI (55..76, sopra il registro del basso).
    """
    ka, kb = parse_key(key_a), parse_key(key_b)
    if kb is None and ka is None:
        raise DJRenderError("note_bridge: no key for A or B")
    if kb is None:
        kb = ka
    if ka is None:
        ka = kb
    kb = ((kb[0] + shift_b) % 12, kb[1])
    common = scale_pcs(*ka) & scale_pcs(*kb)

    def triad(root, mode):
        third = 3 if mode == "minor" else 4
        return [root % 12, (root + third) % 12, (root + 7) % 12]

    cands = [("tonic triad of B", triad(*kb)), ("tonic triad of A", triad(*ka))]
    steps = MINOR_STEPS if kb[1] == "minor" else MAJOR_STEPS
    scale_b = [(kb[0] + s) % 12 for s in steps]
    for deg in range(1, 7):
        r = scale_b[deg]
        t = [r, scale_b[(deg + 2) % 7], scale_b[(deg + 4) % 7]]
        cands.append((f"diatonic triad on degree {deg + 1} of B", t))
    chosen = next(((lab, t) for lab, t in cands if set(t) <= common), None)
    if chosen is None:
        fifth = [kb[0], (kb[0] + 7) % 12]
        chosen = ("tonic + fifth of B", fifth) if set(fifth) <= common else ("tonic of B", [kb[0]])
    label, pcs = chosen
    midi = []
    for i, p in enumerate(pcs):
        m = 57 + ((p - 57) % 12)  # first occurrence >= A3 (MIDI 57)
        midi.append(m)
    midi = sorted(midi)
    if len(midi) == 1:
        midi.append(midi[0] + 12)
    midi.append(midi[0] + 12)  # octave doubling of the root for body
    return {"label": label, "pitch_classes": pcs, "names": [PITCH_CLASSES[p] for p in pcs], "midi": sorted(set(midi)),
            "common_pitch_classes": sorted(common), "key_a": key_a, "key_b": key_b, "shift_b": shift_b,
            "basis": "[H] notes common to both diatonic (natural) scales; tonic of the arriving key first"}


def midi_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


# ------------------------------------------------------------------- synthesis


def _saw(freq: np.ndarray, sr: int, max_h: int = 24, rolloff: float = 1.0, phase0: float = 0.0) -> np.ndarray:
    """Dente di sega band-limited additivo con frequenza istantanea ``freq`` (Hz per campione)."""
    ph = phase0 + 2.0 * np.pi * np.cumsum(freq) / sr
    fmax = float(np.max(freq))
    H = int(max(1, min(max_h, math.floor(0.45 * sr / max(fmax, 1.0)))))
    y = np.zeros(len(freq))
    for h in range(1, H + 1):
        y += np.sin(h * ph) / (h ** rolloff)
    return y / max(1e-9, np.max(np.abs(y)))


def synth_pad(midi: Sequence[float], n: int, sr: int, attack: int, release_start: int, release_end: int,
              seed: int = 0, lpf_hz: float = 1800.0) -> np.ndarray:
    """Pad stereo morbido (sega additiva, due voci desintonizzate di +-7 cent per canale, passa-basso)."""
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(seed)
    out = np.zeros((n, 2))
    for m in midi:
        for ch, det in ((0, -7.0), (1, 7.0)):
            for d in (det, -0.5 * det):
                f = np.full(n, midi_hz(m + d / 100.0))
                out[:, ch] += _saw(f, sr, max_h=16, rolloff=1.5, phase0=float(rng.uniform(0, 2 * np.pi)))
    sos = butter(2, lpf_hz, btype="low", fs=sr, output="sos")
    out = sosfilt(sos, out, axis=0)
    sos_h = butter(2, 160.0, btype="high", fs=sr, output="sos")
    out = sosfilt(sos_h, out, axis=0)
    env = _rise(n, 0, max(attack, 1), "cos") * (1.0 - _rise(n, release_start, release_end, "cos"))
    out *= env[:, None]
    return out / max(1e-9, np.max(np.abs(out)))


def synth_riser(n: int, sr: int, root_midi: float, glide_semitones: float = 12.0, seed: int = 1) -> np.ndarray:
    """Riser in tonalità: tonica, quinta e ottava in sega con glissando che *arriva* sulla tonica
    alla fine (da ``glide_semitones`` sotto), filtro passa-basso che si apre (400 Hz -> 6 kHz) e
    strato di rumore con passa-alto crescente (300 Hz -> 6 kHz). Ampiezza normalizzata a 1;
    l'inviluppo (crescendo) lo applica il chiamante."""
    rng = np.random.default_rng(seed)
    u = np.arange(n) / max(n - 1, 1)
    semis = -glide_semitones * (1.0 - u) ** 1.5  # lands on 0 (the tonic) at the end
    tone = np.zeros((n, 2))
    for interval, amp in ((0, 1.0), (7, 0.6), (12, 0.5)):
        for ch, det in ((0, -5.0), (1, 5.0)):
            f = midi_hz(root_midi + interval) * 2.0 ** ((semis + det / 100.0) / 12.0)
            tone[:, ch] += amp * _saw(f, sr, max_h=20, phase0=float(rng.uniform(0, 2 * np.pi)))
    fc = 400.0 * (6000.0 / 400.0) ** u
    tone = fx.time_varying_filter(tone, sr, fc, "lowpass", order=2, block=256)
    noise = rng.standard_normal((n, 2))
    fcn = 300.0 * (6000.0 / 300.0) ** u
    noise = fx.time_varying_filter(noise, sr, fcn, "highpass", order=2, block=256)
    tone /= max(1e-9, np.max(np.abs(tone)))
    noise /= max(1e-9, np.max(np.abs(noise)))
    y = tone + 0.5 * noise
    from scipy.signal import butter, sosfilt

    y = sosfilt(butter(2, 150.0, btype="high", fs=sr, output="sos"), y, axis=0)
    return y / max(1e-9, np.max(np.abs(y)))


def synth_snare(sr: int, seed: int = 2) -> np.ndarray:
    """Rullante sintetico: rumore 1-8 kHz con decadimento 45 ms + corpo a 190 Hz (30 ms)."""
    from scipy.signal import butter, sosfilt

    n = int(0.18 * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    noise = sosfilt(butter(2, [1000.0, 8000.0], btype="band", fs=sr, output="sos"), rng.standard_normal(n))
    body = np.sin(2 * np.pi * 190.0 * t) * np.exp(-t / 0.03)
    y = noise / max(1e-9, np.max(np.abs(noise))) * np.exp(-t / 0.045) + 0.6 * body
    y[: int(0.001 * sr)] *= np.linspace(0, 1, int(0.001 * sr))
    y /= max(1e-9, np.max(np.abs(y)))
    return np.stack([y, y], 1)


def extract_snare(instrumental: np.ndarray, sr: int, backbeats_s: Sequence[float], length_s: float = 0.15) -> tuple[Optional[np.ndarray], dict]:
    """Rullante estratto da A: media dei tagli ai backbeat (battiti 2 e 4) della componente
    percussiva (HPSS, librosa) dello strumentale. Accettato se il centroide spettrale è
    > 1,2 kHz e l'attacco è nei primi 20 ms; altrimenti None (il chiamante usa il sintetico)."""
    x = _as_stereo(instrumental)
    L = int(length_s * sr)
    pre = int(0.005 * sr)
    cuts = []
    for t in backbeats_s:
        s = int(round(t * sr)) - pre
        if s >= 0 and s + L <= len(x):
            cuts.append(x[s : s + L].mean(axis=1))
    info = {"n_slices": len(cuts), "method": "mean of backbeat slices of the percussive (HPSS) component of A's instrumental"}
    if len(cuts) < 4:
        info["status"] = "too few backbeats"
        return None, info
    try:
        import librosa

        perc = [librosa.effects.percussive(c.astype(np.float32), margin=2.0) for c in cuts]
    except Exception as exc:  # optional dependency
        info["status"] = f"HPSS unavailable: {type(exc).__name__}"
        return None, info
    y = np.mean(np.stack(perc), axis=0).astype(np.float64)
    if not np.any(y):
        info["status"] = "silent"
        return None, info
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    centroid = float((spec * freqs).sum() / max(spec.sum(), 1e-12))
    peak_i = int(np.argmax(np.abs(y)))
    info.update(centroid_hz=centroid, attack_ms=1000.0 * (peak_i - pre) / sr)
    if centroid < 1200.0 or peak_i > pre + int(0.02 * sr):
        info["status"] = "rejected (not snare-like)"
        return None, info
    y = y * np.minimum(1.0, np.linspace(4.0, 0.0, len(y)))  # short release
    y /= max(1e-9, np.max(np.abs(y)))
    info["status"] = "extracted"
    return np.stack([y, y], 1), info


def roll_events(beats: np.ndarray, t0: float, t1: float, schedule: Sequence[float], bar_starts: Sequence[float]) -> list[tuple[float, int]]:
    """Istanti del rullo di rullante sulla griglia dei battiti: ``schedule[k]`` = suddivisione (in
    battiti) nella k-esima battuta del build. Restituisce (tempo, indice battuta)."""
    ev = []
    beats = np.asarray(beats, float)
    for k, (b0, sub) in enumerate(zip(bar_starts, schedule)):
        b1 = bar_starts[k + 1] if k + 1 < len(bar_starts) else t1
        sel = beats[(beats >= b0 - 1e-3) & (beats < b1 - 1e-3)]
        if len(sel) == 0:
            continue
        nxt = np.r_[sel[1:], b1]
        for s, e in zip(sel, nxt):
            m = max(1, int(round(1.0 / sub)))
            for j in range(m):
                t = s + j * (e - s) / m
                if t0 - 1e-6 <= t < t1 - 1e-6:
                    ev.append((float(t), k))
    return ev


def spectral_freeze(segment: np.ndarray, sr: int, n_out: int, seed: int = 3, n_fft: int = 4096) -> np.ndarray:
    """Freeze spettrale: modulo STFT medio del frammento risintetizzato con fasi casuali
    (per canale) per ``n_out`` campioni; RMS uguale al frammento."""
    from scipy.signal import istft, stft

    x = _as_stereo(segment)
    rng = np.random.default_rng(seed)
    hop = n_fft // 4
    out = np.zeros((n_out, 2))
    frames = int(math.ceil(n_out / hop)) + 4
    for ch in range(2):
        _, _, Z = stft(x[:, ch], fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
        mag = np.abs(Z).mean(axis=1)
        ph = rng.uniform(-np.pi, np.pi, (len(mag), frames))
        _, y = istft(mag[:, None] * np.exp(1j * ph), fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
        y = y[n_fft : n_fft + n_out]
        out[: len(y), ch] = y
    r_in, r_out = _rms(x), _rms(out)
    return out * (r_in / r_out if r_out > 0 else 0.0)


def reverb_ir(sr: int, rt60_s: float = 3.5, predelay_ms: float = 20.0, seed: int = 4) -> np.ndarray:
    """Risposta all'impulso sintetica stereo: rumore con decadimento esponenziale (RT60), alti
    che decadono al doppio della velocità, pre-delay; energia unitaria."""
    from scipy.signal import butter, sosfiltfilt

    rng = np.random.default_rng(seed)
    n = int(1.2 * rt60_s * sr)
    t = np.arange(n) / sr
    ir = np.zeros((n, 2))
    for ch in range(2):
        nz = rng.standard_normal(n)
        lo = sosfiltfilt(butter(2, 3000.0, btype="low", fs=sr, output="sos"), nz)
        hi = nz - lo
        ir[:, ch] = lo * np.exp(-6.91 * t / rt60_s) + hi * np.exp(-6.91 * t / (0.5 * rt60_s))
    d = int(predelay_ms / 1000.0 * sr)
    ir = np.vstack([np.zeros((d, 2)), ir])
    return ir / np.sqrt((ir ** 2).sum(axis=0, keepdims=True))


def read_varispeed(src: np.ndarray, start: float, speed: np.ndarray, sr: int, lowpass_for: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
    """Legge ``src`` a velocità variabile (vinile/nastro): posizione = start + somma(speed).
    Interpolazione lineare per canale; ``lowpass_for`` = velocità massima per l'anti-aliasing
    (passa-basso a 0,45*sr/v sul tratto letto). Restituisce (audio, posizioni)."""
    from scipy.signal import butter, sosfiltfilt

    x = _as_stereo(src)
    pos = start + np.concatenate([[0.0], np.cumsum(speed[:-1])])
    lo = int(max(0, math.floor(pos.min()) - 8))
    hi = int(min(len(x), math.ceil(pos.max()) + 8))
    seg = x[lo:hi].copy()
    if lowpass_for and lowpass_for > 1.0 and len(seg) > 64:
        fc = min(0.45 * sr / lowpass_for, 0.45 * sr)
        seg = sosfiltfilt(butter(8, fc, btype="low", fs=sr, output="sos"), seg, axis=0)
    idx = np.arange(lo, hi, dtype=float)
    out = np.stack([np.interp(pos, idx, seg[:, c], left=0.0, right=0.0) for c in range(2)], 1)
    return out, pos


def pitch_shift(x: np.ndarray, sr: int, semitones: float, formant: bool = False) -> tuple[np.ndarray, dict]:
    """Trasposizione senza cambio di durata con Rubber Band CLI (motore R3, offline).

    ``formant=True`` preserva l'inviluppo formantico (voci). La lunghezza è riportata a
    quella d'ingresso; Rubber Band offline compensa il ritardo d'avvio.
    """
    import soundfile as sf

    x = _as_stereo(x)
    if semitones == 0 or not np.any(x):
        return x.copy(), {"applied": False, "semitones": 0}
    cli = shutil.which("rubberband")
    if not cli:
        raise DJRenderError("pitch shift needs the rubberband CLI")
    with tempfile.TemporaryDirectory(prefix="dj_creative_") as tmp:
        src, dst = os.path.join(tmp, "in.wav"), os.path.join(tmp, "out.wav")
        sf.write(src, x.astype(np.float32), sr, subtype="FLOAT")
        cmd = [cli, "-q", "-3", "-p", f"{semitones:.6f}"] + (["-F"] if formant else []) + [src, dst]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise DJRenderError(f"pitch shift failed: {proc.stderr.strip()[:300]}")
        y, sr2 = sf.read(dst, dtype="float64", always_2d=True)
    if sr2 != sr:
        raise DJRenderError("pitch shift changed the sample rate")
    return _fit(y, len(x)), {"applied": True, "semitones": semitones, "formant_preserved": formant,
                             "engine": "Rubber Band CLI R3 (-3), offline, pitch only (-p)"}


# --------------------------------------------------------------------- context


@dataclass
class _Ctx:
    sr: int
    cfg: DJCreativeConfig
    a: np.ndarray
    b: np.ndarray
    va: np.ndarray
    vb: np.ndarray
    grid: dict
    ia: int
    jb: int
    ta: dict
    tb: dict
    loud: dict
    has_va: bool
    has_vb: bool

    @property
    def da(self):
        return self.grid["downbeats_a"]

    @property
    def db(self):
        return self.grid["downbeats_b"]

    def sa(self, i: Optional[int] = None) -> int:
        return int(round(self.da[self.ia if i is None else i] * self.sr))

    def sb(self, j: Optional[int] = None) -> int:
        return int(round(self.db[self.jb if j is None else j] * self.sr))


def _prepare(audio_a, audio_b, vocals_a, vocals_b, beats_a, downbeats_a, beats_b, downbeats_b,
             cue_out_a, cue_in_b, sr, cfg, *, need_vocals_a=False, need_vocals_b=False) -> _Ctx:
    if not 5.0 <= cfg.crossfade_ms <= 20.0:
        raise DJRenderError("crossfade must be 5..20 ms")
    a, b = _as_stereo(audio_a), _as_stereo(audio_b)
    if need_vocals_a and vocals_a is None:
        raise DJRenderError("this technique needs A's vocal stem")
    if need_vocals_b and vocals_b is None:
        raise DJRenderError("this technique needs B's vocal stem")
    grid = {k: np.sort(np.asarray(list(v), dtype=float)) for k, v in
            (("beats_a", beats_a), ("downbeats_a", downbeats_a), ("beats_b", beats_b), ("downbeats_b", downbeats_b))}
    ia = fx.snap_downbeat(grid["downbeats_a"], cue_out_a, cfg.snap_tolerance_s, "cue_out_a")
    jb = fx.snap_downbeat(grid["downbeats_b"], cue_in_b, cfg.snap_tolerance_s, "cue_in_b")
    lufs_a = cfg.lufs_a_override if cfg.lufs_a_override is not None else dj.integrated_lufs(a, sr)
    lufs_b = cfg.lufs_b_override if cfg.lufs_b_override is not None else dj.integrated_lufs(b, sr)
    ga_db = (cfg.target_lufs - lufs_a) if np.isfinite(lufs_a) else 0.0
    gb_db = (cfg.target_lufs - lufs_b) if np.isfinite(lufs_b) else 0.0
    ga, gb = _db(ga_db), _db(gb_db)
    def norm(x, n, g):
        if x is None:
            return np.zeros((n, 2))
        if cfg.inplace_inputs and isinstance(x, np.ndarray) and x.dtype == np.float64 and x.ndim == 2 and x.shape == (n, 2):
            x *= g
            return x
        return _fit(x, n) * g

    va = norm(vocals_a, len(a), ga)
    vb = norm(vocals_b, len(b), gb)
    if cfg.inplace_inputs and isinstance(audio_a, np.ndarray) and isinstance(audio_b, np.ndarray) \
            and np.shares_memory(a, audio_a) and np.shares_memory(b, audio_b):
        a *= ga
        b *= gb
    else:
        a, b = a * ga, b * gb
    loud = {"target_lufs": cfg.target_lufs, "a_integrated_lufs_source": lufs_a, "b_integrated_lufs_source": lufs_b,
            "a_gain_db": ga_db, "b_gain_db": gb_db, "scope": "whole track, ITU-R BS.1770-4 via pyloudnorm; vocal stems get their mix's gain"
            + ("; measured by the caller on the whole track (audio passed as an excerpt)" if cfg.lufs_a_override is not None else "")}
    ta = fx.bar_and_beat(grid["downbeats_a"], grid["beats_a"], ia, -1, cfg.max_bar_deviation)
    tb = fx.bar_and_beat(grid["downbeats_b"], grid["beats_b"], jb, +1, cfg.max_bar_deviation)
    return _Ctx(sr, cfg, a, b, va, vb, grid, ia, jb, ta, tb, loud, vocals_a is not None, vocals_b is not None)


def _b_entry(x: np.ndarray, sb: int, entry: int, n: int, cfg: DJCreativeConfig, sr: int) -> np.ndarray:
    """Sorgente ``x`` con il campione ``sb`` sul campione ``entry`` della clip, silenziosa prima,
    micro fade-in equal-power che *termina* sul downbeat (come dj_render cut)."""
    d = int(round(cfg.crossfade_ms / 1000.0 * sr))
    deck = _slice(x, sb - entry, n)
    g = np.sin(0.5 * np.pi * np.clip((np.arange(n) - (entry - d)) / max(d, 1), 0.0, 1.0))
    return deck * g[:, None]


def _a_cut(n: int, cut: int, cfg: DJCreativeConfig, sr: int) -> np.ndarray:
    d = int(round(cfg.crossfade_ms / 1000.0 * sr))
    g = np.cos(0.5 * np.pi * np.clip((np.arange(n) - (cut - d)) / max(d, 1), 0.0, 1.0))
    g[np.arange(n) >= cut] = 0.0
    return g


def _tempo_block(ctx: _Ctx, applied: bool, **extra) -> dict:
    k, rate = dj.choose_tempo_fold(ctx.ta["bar_s"], ctx.tb["bar_s"], max_stretch=10.0)
    out = {"applied": applied, "a_bar_s": ctx.ta["bar_s"], "b_bar_s": ctx.tb["bar_s"], "a_bpm_local": ctx.ta["bpm"],
           "b_bpm_local": ctx.tb["bpm"], "fold_k": k, "tempo_jump_pct": 100.0 * (1.0 / rate - 1.0)}
    if not applied:
        out.update(tempo_ratio_b=1.0, note="no stretch: B enters clean on its downbeat at its own tempo")
    out.update(extra)
    return out


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """``x`` anticipato di ``k`` campioni (k > 0) o ritardato (k < 0), riempiendo di zeri."""
    if k == 0:
        return x
    y = np.zeros_like(x)
    if k > 0:
        y[:-k] = x[k:]
    else:
        y[-k:] = x[:k]
    return y


def _beatmatch(ctx: _Ctx, a_virtual: np.ndarray, da_virtual: np.ndarray, ba_virtual: np.ndarray, cue_out_s: float,
               bars: int, extra_post_s: float, key_shift_b: int, phase_nudge_s: float = 0.0) -> dict:
    """B (mix e voce) beatmatchato sulla griglia (virtuale) di A con dj_render.render_transition.

    Restituisce ``b_full``, ``b_voc``, ``b_instr`` (coordinate della clip, già al livello
    normalizzato di B), la durata della chiamata e i metadati di stretch/allineamento.
    """
    cfg, sr = ctx.cfg, ctx.sr
    rcfg = dj.DJRenderConfig(target_lufs=cfg.target_lufs, ceiling_dbtp=cfg.ceiling_dbtp, cue_convention="start",
                             pre_s=cfg.pre_s, post_s=cfg.post_s + extra_post_s, max_stretch=cfg.max_stretch,
                             stretch_mode=cfg.stretch_mode, stretch_backend=cfg.stretch_backend,
                             snap_tolerance_s=cfg.snap_tolerance_s, max_bar_deviation=cfg.max_bar_deviation, duck_vocals=False)
    kw = dict(beats_a=ba_virtual, downbeats_a=da_virtual, beats_b=ctx.grid["beats_b"], downbeats_b=ctx.db,
              cue_out_a=cue_out_s, cue_in_b=float(ctx.db[ctx.jb]), kind="blend", bars=bars, sample_rate=sr,
              config=rcfg, return_decks=True)
    r1 = dj.render_transition(a_virtual, ctx.b, **kw)
    b_full = r1.decks["b_stretched_raw"] * _db(-r1.metadata["loudness"]["b_gain_db"])
    m = r1.metadata
    del r1  # free the decks of the first call before the second (memory)
    if ctx.has_vb and np.any(ctx.vb):
        r2 = dj.render_transition(a_virtual, ctx.vb, **kw)
        g2 = r2.metadata["loudness"]["b_gain_db"]
        b_voc = r2.decks["b_stretched_raw"] * _db(-(g2 if g2 is not None else 0.0))
        del r2
    else:
        b_voc = np.zeros_like(b_full)
    n = min(len(b_full), len(b_voc))
    b_full, b_voc = b_full[:n], b_voc[:n]
    nudge = int(round(phase_nudge_s * sr))
    if abs(phase_nudge_s) > 0.05:
        raise DJRenderError("phase nudge larger than 50 ms: check the grids instead")
    b_full, b_voc = _shift(b_full, nudge), _shift(b_voc, nudge)
    b_instr = b_full - b_voc
    shift_info = {"applied": False, "semitones": 0}
    if key_shift_b:
        b_instr, shift_info = pitch_shift(b_instr, sr, key_shift_b, formant=False)
        b_voc, vinfo = pitch_shift(b_voc, sr, key_shift_b, formant=True)
        shift_info = {**shift_info, "vocal_stem": vinfo, "note": "instrumental (mix - stem) and vocal stem shifted separately (vocal with formant preservation); B stays transposed until the end of the clip"}
        b_full = b_instr + b_voc
    stretch = {"applied": True, **{k: m["stretch"].get(k) for k in ("mode", "backend", "fold_k_b_bars_per_a_bar", "tempo_ratio_b",
                                                                    "time_ratio_b", "stretch_pct", "a_bar_s", "b_bar_s",
                                                                    "a_bpm_local", "b_bpm_local", "b_bpm_effective", "after_overlap",
                                                                    "keyframes")},
               "via": "dj_render.render_transition(kind='blend', cue_convention='start'): deck b_stretched_raw (mix and vocal stem, same grids)",
               "key_shift_b": shift_info,
               "time_map": _nudged_time_map(m["stretch"].get("time_map"), nudge / sr),
               "phase_nudge": {"seconds": phase_nudge_s, "samples": nudge,
                               "meaning": "B moved earlier by this amount after the grid-based beatmatch (DJ 'nudge'); "
                                          "measured by the caller (e.g. onsets vs grid on the source tracks)"}}
    align = {k: m["downbeat_alignment"].get(k) for k in ("max_abs_error_ms", "median_abs_error_ms", "within_10ms",
                                                          "interdeck_median_abs_lag_ms", "planned_beat_phase_error_ms")}
    return {"b_full": b_full, "b_voc": b_voc, "b_instr": b_instr, "n": n, "stretch": stretch, "alignment": align,
            "switch_b_source_sample": m["switch"]["b_source_sample"], "overlap_meta": m["overlap"]}


def _nudged_time_map(tm: Optional[dict], nudge_s: float) -> Optional[dict]:
    """Time map of dj_render (B source s -> clip s) after B is moved ``nudge_s`` earlier (same clip coordinates:
    the blend render and the creative clip share ``pre_s`` before A's cue)."""
    if not tm:
        return None
    out = dict(tm)
    out["output_s"] = [float(d) - nudge_s for d in tm["output_s"]]
    if nudge_s:
        out["note"] = f"output_s shifted by the phase nudge (-{nudge_s:.4f} s)"
    return out


def _b_map_of(stretch: Mapping[str, Any]):
    tm = (stretch or {}).get("time_map")
    if not (stretch or {}).get("applied") or not tm:
        return None
    return dj._PiecewiseMap(tm["b_source_s"], tm["output_s"], tm["slope_before"], tm["slope_after"])


def _virtual_a_grid(ctx: _Ctx, upto_i: int, extra_bars: int) -> tuple[np.ndarray, np.ndarray]:
    """Griglia di A fino al downbeat ``upto_i`` più ``extra_bars`` battute alla durata locale
    (griglie regolarizzate a tempo costante: coincide con quella reale dove esiste)."""
    da, ba = ctx.da, ctx.grid["beats_a"]
    X = da[upto_i]
    bar, beat, bpb = ctx.ta["bar_s"], ctx.ta["beat_s"], ctx.ta["beats_per_bar"]
    d = np.r_[da[: upto_i + 1], X + bar * np.arange(1, extra_bars + 1)]
    b = np.r_[ba[ba < X - 1e-3], X + beat * np.arange(0, extra_bars * bpb + 1)]
    return np.sort(d), np.sort(b)


def _section_gain(ctx: _Ctx, a_end_src: int, b_ref: np.ndarray, b_ref_out: int, b_bar_out_s: float, n: int, t_end: int, note: str):
    return dj.plan_section_gain(ctx.a, a_end_src, ctx.ta["bar_s"], b_ref, b_ref_out, b_bar_out_s, n, ctx.sr,
                                ctx.cfg.section_gain, transition_end_out=t_end, b_ref_note=note)


def _finish(ctx: _Ctx, technique: str, *, deck_a: np.ndarray, deck_b: np.ndarray, overlay: Optional[np.ndarray],
            t_s: int, t_sw: int, t_e: int, off_a: int, sb_at_switch: int, stretch: dict, creative: dict,
            exit_effect: Optional[str], a_map: list, b_map: list, vocals_a_heard: np.ndarray, vocals_b_heard: np.ndarray,
            sg_info: dict, extra_decks: dict, provenance: Optional[dict], return_decks: bool, trim_region: Optional[tuple] = None) -> DJRenderResult:
    cfg, sr = ctx.cfg, ctx.sr
    n = t_e + int(round(cfg.post_s * sr))
    deck_a, deck_b = _fit(deck_a, n), _fit(deck_b, n)
    overlay = _fit(overlay, n) if overlay is not None else np.zeros((n, 2))
    base = deck_a + deck_b
    trim = {"status": "no overlay"}
    if np.any(overlay):
        w = int(round(3.0 * sr))
        region = trim_region or (t_s, t_sw if exit_effect else t_e)
        g, trim = trim_overlay(base, overlay, sr, region, (max(0, t_s - w), t_s), cfg.overlay_margin_db)
        overlay = overlay * g
        for key in ("overlay_parts",):
            if key in extra_decks:
                extra_decks[key] = {k: v * g for k, v in extra_decks[key].items()}
        if "vocals_a_overlay" in extra_decks:
            vocals_a_heard = vocals_a_heard + g * _fit(extra_decks.pop("vocals_a_overlay"), len(vocals_a_heard))
    elif "vocals_a_overlay" in extra_decks:
        extra_decks.pop("vocals_a_overlay")
    out = base + overlay
    loud = dict(ctx.loud)
    w = int(round(3.0 * sr))
    b_from = t_sw if exit_effect else t_e
    before = dj.integrated_lufs(out[max(0, t_s - w) : t_s], sr)
    after = dj.integrated_lufs(out[b_from : b_from + w], sr)
    loud.update(lufs_3s_before_transition=before, lufs_3s_after_b=after, after_window_from="switch (exit effect)" if exit_effect else "transition end",
                loudness_jump_db=(after - before) if np.isfinite(before) and np.isfinite(after) else None,
                section_gain_b_db=sg_info.get("applied_db", 0.0), overlay_trim=trim)
    limited, limiter = dj.true_peak_limit(out, sr, cfg.ceiling_dbtp, cfg.limiter_hold_ms)
    loud["output_integrated_lufs"] = dj.integrated_lufs(limited, sr)
    loud["output_true_peak_dbtp"] = limiter["true_peak_after_dbtp"]
    pre = int(round(cfg.pre_s * sr))
    expected = t_e + int(round(cfg.post_s * sr))
    a_src = t_s + off_a
    hop = 0.02
    # Stretched B (loop_extend, acapella_over, vocal_swap): B source samples through the renderer's time map, so that
    # splicing (dj_set.clip_geometry) and QA know which B sample plays where (fix 21/10/2026: they were missing).
    bmap = _b_map_of(stretch)
    b_end_src = None
    sb_declared = None
    if bmap is not None:
        b_end_src = int(round(float(bmap.inv(t_e / sr)) * sr))
        # acapella_over / vocal_swap passed B's sample at the *beatmatch render's* switch (middle of its overlap), not
        # at their own swap downbeat t_sw (2 bars later with bars=4): take it from the map instead.
        sb_map = int(round(float(bmap.inv(t_sw / sr)) * sr))
        if abs(sb_map - int(sb_at_switch)) > 1:
            sb_declared = int(sb_at_switch)
        sb_at_switch = sb_map
        b_map = [dict(seg, src_start=int(round(float(bmap.inv(int(seg["out_start"]) / sr)) * sr)), src_via="stretch.time_map")
                 if seg.get("src_start") is None and seg.get("rate") not in (None, 1.0) else seg for seg in b_map]
    va_act, _ = _activity(vocals_a_heard, sr, ctx.va)
    vb_act, _ = _activity(vocals_b_heard, sr, ctx.vb)
    m = min(len(va_act), len(vb_act))
    meta = {
        "schema_version": DJ_CREATIVE_VERSION,
        "renderer_base": dj.DJ_RENDER_VERSION,
        "kind": technique,
        "sample_rate": sr,
        "bars": int(creative.get("params", {}).get("bars", 0) or 0),
        "overlap": {
            "output_start_sample": t_s, "output_end_sample": t_e,
            "a_source_start_sample": a_src, "a_source_end_sample": None,
            "b_source_start_sample": sb_at_switch, "duration_s": (t_e - t_s) / sr,
            "note": "transition region = technique start -> technique end (tails included when they are faded inside it)",
            **({"b_source_end_sample": b_end_src,
                "b_source_note": "b_source_start_sample = B at the switch (as before); b_source_end_sample = B at the transition end "
                                 "through stretch.time_map"} if b_end_src is not None else {}),
        },
        "switch": {"output_sample": t_sw, "a_source_sample": t_sw + off_a, "b_source_sample": sb_at_switch,
                   **({"b_source_sample_before_fix": sb_declared,
                       "b_source_note": "from stretch.time_map at the technique's switch (21/10/2026: was B at the beatmatch render's switch)"}
                      if sb_declared is not None else {})},
        "transition_duration_s": (t_e - t_s) / sr,
        "stretch": stretch,
        "cue_out_a_s": float(ctx.da[ctx.ia]),
        "cue_in_b_s": float(ctx.db[ctx.jb]),
        "exit_effect": exit_effect,
        "creative": creative,
        "a_map": a_map,
        "b_map": b_map,
        "vocals": {
            "a_heard_s": float(va_act.sum() * hop), "b_heard_s": float(vb_act.sum() * hop),
            "both_heard_s": float((va_act[:m] & vb_act[:m]).sum() * hop),
            "method": "stem RMS 20 ms above max(0.002, p95 - 24 dB) of the whole normalized stem (dj_render.vocal_activity threshold), on the vocal actually heard from each deck",
        },
        "loudness": loud,
        "limiter": limiter,
        "section_gain": sg_info,
        "clip": {
            "rule": "10 s before the transition start + whole transition + 10 s after its end (AGENTS.md, 20/10/2026)",
            "pre_s": cfg.pre_s, "post_s": cfg.post_s, "transition_start_sample": t_s, "switch_sample": t_sw,
            "transition_end_sample": t_e, "length_samples": len(limited), "duration_s": len(limited) / sr,
            "matches_rule": bool(len(limited) == expected and t_s == pre and cfg.pre_s == 10.0 and cfg.post_s == 10.0),
        },
        "config": asdict(cfg),
    }
    if provenance:
        meta["provenance"] = provenance
    result = DJRenderResult(limited.astype(np.float32), sr, dj._jsonable(meta))
    if return_decks:
        ones = np.ones(n)
        decks = {"a": deck_a, "b": deck_b, "overlay": overlay, "vocals_a_heard": _fit(vocals_a_heard, n),
                 "vocals_b_heard": _fit(vocals_b_heard, n),
                 "gains": {"fader_a": ones, "fader_b": ones,
                           "note": "vocals_*_heard already include every gain (fader, mute, duck, section gain): pass them to dj_qa with unit gains"}}
        decks.update(extra_decks)
        result.decks = decks
    return result


def _activity(stem_heard: np.ndarray, sr: int, ref_stem: np.ndarray) -> tuple[np.ndarray, float]:
    """Frame attivi (20 ms) della voce udibile con la soglia dello stem intero (come dj_qa)."""
    _, _, info = dj.vocal_activity(ref_stem, sr) if np.any(ref_stem) else (None, None, {"threshold_rms": None})
    thr = info.get("threshold_rms")
    hop = int(round(0.02 * sr))
    mono = _as_stereo(stem_heard).mean(axis=1)
    nf = len(mono) // hop
    if nf == 0 or thr is None or not np.isfinite(thr):
        return np.zeros(max(nf, 0), dtype=bool), float("inf")
    rms = np.sqrt(np.mean(mono[: nf * hop].reshape(nf, hop) ** 2, axis=1))
    return rms > thr, float(thr)


def map_to_clip(src: np.ndarray, segments: Sequence[Mapping[str, Any]], n: int, *, vocal_only: bool = False) -> np.ndarray:
    """Porta una sorgente nella clip con i segmenti ``a_map``/``b_map`` (``out_start``,
    ``out_end``, ``src_start``; rate 1). ``vocal_only``: salta i segmenti senza voce."""
    x = _as_stereo(src)
    out = np.zeros((n, 2))
    for s in segments:
        if s.get("rate", 1.0) != 1.0 or (vocal_only and not s.get("vocals", True)):
            continue
        o0, o1 = int(s["out_start"]), min(int(s["out_end"]), n)
        if o1 > o0:
            out[o0:o1] = _slice(x, int(s["src_start"]), o1 - o0)
    return out


def words_in_clip(meta: Mapping[str, Any], words: Sequence[Mapping[str, Any]], role: str = "a") -> list[dict]:
    """Parole ASR (``start_sample``/``end_sample`` sorgente) -> secondi della clip, con le mappe
    ``a_map``/``b_map`` del render (solo segmenti a velocità 1 in cui la voce del deck suona)."""
    sr = int(meta.get("sample_rate", 48000))
    out = []
    bmap = _b_map_of(meta.get("stretch") or {}) if role == "b" else None
    for s in meta.get(f"{role}_map", []):
        if not s.get("vocals", True):
            continue
        o0, o1 = int(s["out_start"]), int(s["out_end"])
        stretched = s.get("rate", 1.0) not in (None, 1.0)
        if s.get("rate", 1.0) is None or (stretched and bmap is None) or (not stretched and s.get("src_start") is None):
            continue
        for w in words:
            if stretched:  # B stretched: through the renderer's time map (21/10/2026)
                ws, we = (int(round(float(bmap.fwd(int(w[k]) / sr)) * sr)) for k in ("start_sample", "end_sample"))
            else:
                s0 = int(s["src_start"])
                ws, we = int(w["start_sample"]) - s0 + o0, int(w["end_sample"]) - s0 + o0
            if we > o0 and ws < o1:
                # start_unclipped_s / end_unclipped_s: where the word would start/end without the segment bounds
                # (dj_qa uses them to tell a word cut where A's continuous playback stops).
                out.append({"start_s": max(ws, o0) / sr, "end_s": min(we, o1) / sr, "text": w.get("text", w.get("word", "")),
                            "confidence": w.get("confidence"), "start_unclipped_s": ws / sr, "end_unclipped_s": we / sr,
                            "clipped": bool(ws < o0 or we > o1)})
    return sorted(out, key=lambda w: w["start_s"])


def _cover(stem: np.ndarray, sr: int, s0: int, s1: int, ref: np.ndarray) -> float:
    act, _ = _activity(stem[max(0, s0) : max(0, s1)], sr, ref)
    return float(act.mean()) if len(act) else 0.0


def mute_guard(ctx: _Ctx, at_src: int) -> dict:
    """Voce di A attorno al punto in cui la si toglie (stem): copertura nell'ultimo battito e nei
    150 ms dopo. Una voce attiva qui significa una parola tagliata dal mute [H, stesse finestre
    della guardia dei cut di dj_select/dj_qa]."""
    if not ctx.has_va:
        return {"status": "no vocal stem"}
    beat = int(round(ctx.ta["beat_s"] * ctx.sr))
    before = _cover(ctx.va, ctx.sr, at_src - beat, at_src, ctx.va)
    after = _cover(ctx.va, ctx.sr, at_src, at_src + int(0.15 * ctx.sr), ctx.va)
    return {"a_vocal_cover_last_beat": before, "a_vocal_cover_150ms_after": after,
            "clear": bool(before <= 0.1 and after <= 0.0), "rule": "[H] clear if <= 10% of the last beat and nothing in the 150 ms after"}


def _blend_gains(n: int, t_s: int, t_sw: int, t_e: int, sr: int, cfg: DJCreativeConfig) -> dict:
    """Curve di fader/EQ di dj_render (blend): B sale a quarto di seno fino allo switch, bassi di B
    a zero fino allo swap complementare che termina sullo switch, alti di B da high_floor a 0 dB;
    A pieno fino allo switch poi quarto di coseno a 0 alla fine, alti di A che scendono."""
    idx = np.arange(n)
    v = np.clip((idx - t_s) / max(t_sw - t_s, 1), 0.0, 1.0)
    w = np.clip((idx - t_sw) / max(t_e - t_sw, 1), 0.0, 1.0)
    swap = max(1, int(round(cfg.bass_swap_ms / 1000.0 * sr)))
    low_b = np.clip((idx - (t_sw - swap)) / swap, 0.0, 1.0)
    return {
        "fader_b": np.where(idx < t_s, 0.0, np.sin(0.5 * np.pi * v)),
        "fader_a": np.where(idx >= t_e, 0.0, np.cos(0.5 * np.pi * w)),
        "low_b": low_b, "low_a": 1.0 - low_b,
        "high_b": _db(cfg.high_floor_db * (1.0 - v)), "high_a": _db(cfg.high_floor_db * w),
    }


def _eq(x: np.ndarray, sr: int, cfg: DJCreativeConfig, low: Any = 1.0, mid: Any = 1.0, high: Any = 1.0) -> np.ndarray:
    lo, mi, hi = dj.split_bands(x, sr, cfg.low_hz, cfg.high_hz)
    g = lambda v: (np.asarray(v)[:, None] if np.ndim(v) else v)  # noqa: E731
    return lo * g(low) + mi * g(mid) + hi * g(high)


# ================================================================== techniques


def loop_extend(audio_a, audio_b, *, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b,
                vocals_a=None, vocals_b=None, loop_bars: int = 1, bars: int = 8, key_shift_b: int = 0,
                source: str = "auto", phase_nudge_s: float = 0.0, sample_rate: int = dj.DEFAULT_SAMPLE_RATE, config: Optional[DJCreativeConfig] = None,
                provenance: Optional[dict] = None, return_decks: bool = False) -> DJRenderResult:
    """Loop di ``loop_bars`` battute di A (le ultime prima di ``cue_out_a``) ripetuto mentre B entra
    beatmatchato per ``bars`` battute (curve del blend di dj_render, bass swap a metà).

    ``source``: ``mix`` (A così com'è), ``instrumental`` (mix − stem vocale) o ``auto`` (mix se la
    voce di A copre <= ``vocal_active_cover`` del loop, altrimenti strumentale). Giunture del loop:
    gli ultimi ``loop_xfade_ms`` di ogni ripetizione sfumano (equal-power) nel materiale che
    precede l'inizio del loop, così il downbeat ripetuto arriva intero e continuo.
    """
    cfg = config or DJCreativeConfig()
    sr = int(sample_rate)
    if loop_bars not in LOOP_BARS or bars not in OVERLAP_BARS:
        raise DJRenderError(f"loop_bars in {LOOP_BARS}, bars in {OVERLAP_BARS}")
    ctx = _prepare(audio_a, audio_b, vocals_a, vocals_b, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b, sr, cfg,
                   need_vocals_a=(source == "instrumental"))
    ia = ctx.ia
    if ia - loop_bars < 0:
        raise DJRenderError("loop_extend: not enough A bars before cue_out_a")
    bars_src = np.diff(ctx.da[ia - loop_bars : ia + 1])
    if np.any(np.abs(bars_src / ctx.ta["bar_s"] - 1.0) > cfg.max_bar_deviation):
        raise DJRenderError("loop_extend: irregular A bars in the loop")
    X = ctx.sa(ia)
    L0 = ctx.sa(ia - loop_bars)
    L = X - L0
    cover = _cover(ctx.va, sr, L0, X, ctx.va) if ctx.has_va else 0.0
    if source == "auto":
        source = "mix" if (not ctx.has_va or cover <= cfg.vocal_active_cover) else "instrumental"
    if source == "instrumental" and not ctx.has_va:
        raise DJRenderError("loop_extend: instrumental source needs A's vocal stem")
    loop_src = ctx.a if source == "mix" else ctx.a - ctx.va
    loop_voc = ctx.va if source == "mix" else np.zeros_like(ctx.va)
    xf = max(1, int(round(cfg.loop_xfade_ms / 1000.0 * sr)))
    if L0 - xf < 0:
        raise DJRenderError("loop_extend: loop starts too close to the track start")
    pre = int(round(cfg.pre_s * sr))
    post = int(round(cfg.post_s * sr))
    bar_out = L / loop_bars / sr  # output bar length (A's local bar, constant grid)
    copies = int(math.ceil((bars * bar_out + cfg.post_s + 2.0) / (loop_bars * bar_out))) + 1
    fi = np.sin(0.5 * np.pi * (np.arange(xf) + 0.5) / xf)[:, None]
    fo = np.cos(0.5 * np.pi * (np.arange(xf) + 0.5) / xf)[:, None]

    def tile(head: np.ndarray, src: np.ndarray) -> np.ndarray:
        parts = [head[:X].copy()]
        seg, lead = src[L0:X], src[L0 - xf : L0]
        for _ in range(copies):
            parts[-1][-xf:] = parts[-1][-xf:] * fo + lead * fi
            parts.append(seg.copy())
        return np.vstack(parts)

    a_virt = tile(ctx.a, loop_src)  # A plays its own mix up to X, then the loop
    va_virt = tile(ctx.va, loop_voc)
    da_v, ba_v = _virtual_a_grid(ctx, ia, copies * loop_bars)
    bm = _beatmatch(ctx, a_virt, da_v, ba_v, float(ctx.da[ia]), bars, 2.0, key_shift_b, phase_nudge_s)
    off_a = X - pre
    t_s = pre
    t_sw = int(round((da_v[ia + bars // 2] - ctx.da[ia]) * sr)) + pre
    t_e = int(round((da_v[ia + bars] - ctx.da[ia]) * sr)) + pre
    n = t_e + post
    a_raw = _slice(a_virt, off_a, n)
    b_raw = _fit(bm["b_full"], n)
    g = _blend_gains(n, t_s, t_sw, t_e, sr, cfg)
    deck_a = _eq(a_raw, sr, cfg, low=g["low_a"], high=g["high_a"]) * g["fader_a"][:, None]
    b_ref = b_raw[t_sw : t_sw + int(round(cfg.section_gain.b_bars * bar_out * sr))]
    sg_curve, sg_info = _section_gain(ctx, X, b_ref, t_sw, bar_out, n, t_e, "stretched B (output bars = A bars) from the switch downbeat")
    deck_b = _eq(b_raw, sr, cfg, low=g["low_b"], high=g["high_b"]) * (g["fader_b"] * sg_curve)[:, None]
    va_heard = _slice(va_virt, off_a, n) * g["fader_a"][:, None]
    vb_heard = _fit(bm["b_voc"], n) * (g["fader_b"] * sg_curve)[:, None]
    a_map = [{"out_start": 0, "out_end": t_s, "src_start": off_a, "rate": 1.0, "vocals": True, "what": "A before the loop"}]
    for k in range(copies):
        o0 = t_s + k * L
        if o0 >= n:
            break
        a_map.append({"out_start": o0, "out_end": min(n, o0 + L), "src_start": L0, "rate": 1.0, "vocals": source == "mix",
                      "what": f"loop repeat {k + 1} ({source})"})
    sbs = int(bm["switch_b_source_sample"])
    b_map = [{"out_start": t_s, "out_end": n, "src_start": None, "rate": bm["stretch"].get("tempo_ratio_b"), "vocals": True,
              "what": "B beatmatched (time map of dj_render): use decks['vocals_b_heard']"}]
    creative = {
        "technique": "loop_extend",
        "a_vocal_mute_guard": mute_guard(ctx, X) if source == "instrumental" else {"status": "loop from the mix: no vocal mute"},
        "params": {"loop_bars": loop_bars, "bars": bars, "source": source, "key_shift_b": key_shift_b, "loop_xfade_ms": cfg.loop_xfade_ms},
        "loop": {"a_source_samples": [L0, X], "length_samples": L, "a_vocal_cover_in_loop": cover, "repeats_in_clip": len(a_map) - 1,
                 "seam": "equal-power crossfade of the loop tail into the material preceding the loop start, ending on the repeated downbeat"},
        "regions": [{"name": "loop + B blend in", "output": [t_s, t_sw]}, {"name": "bass swap", "output": [t_sw, t_sw]},
                    {"name": "loop fades out", "output": [t_sw, t_e]}],
        "beatmatch_alignment": bm["alignment"],
        "basis": "[P] loop / loop roll to extend an outro (Serato/rekordbox/Traktor loop functions; Broughton & Brewster); fades and EQ as dj_render blend [H]",
    }
    return _finish(ctx, "loop_extend", deck_a=deck_a, deck_b=deck_b, overlay=None, t_s=t_s, t_sw=t_sw, t_e=t_e, off_a=off_a,
                   sb_at_switch=sbs, stretch=bm["stretch"], creative=creative, exit_effect=None, a_map=a_map, b_map=b_map,
                   vocals_a_heard=va_heard, vocals_b_heard=vb_heard, sg_info=sg_info,
                   extra_decks={"a_raw": a_raw, "b_stretched_raw": b_raw}, provenance=provenance, return_decks=return_decks)


def _pickup_samples(ctx: _Ctx, beats: int = 1) -> int:
    """Anacrusi di B: se lo stem di B è attivo nell'ultimo battito prima di ``cue_in_b``, la voce
    di B entra un battito prima (in campioni della clip, B stretchato ~ battito di A)."""
    if not ctx.has_vb or beats <= 0:
        return 0
    sb = ctx.sb()
    span = int(round(beats * ctx.tb["beat_s"] * ctx.sr))
    cover = _cover(ctx.vb, ctx.sr, sb - span, sb, ctx.vb)
    return int(round(beats * ctx.ta["beat_s"] * ctx.sr)) if cover > 0.2 else 0


def acapella_over(audio_a, audio_b, *, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b,
                  vocals_a=None, vocals_b=None, bars: int = 4, key_shift_b: int = 0, mid_cut_db: float = -3.0,
                  swap_release_beats: int = 1, pickup_beats: int = 1, phase_nudge_s: float = 0.0, sample_rate: int = dj.DEFAULT_SAMPLE_RATE,
                  config: Optional[DJCreativeConfig] = None, provenance: Optional[dict] = None, return_decks: bool = False) -> DJRenderResult:
    """Voce di B (stem, beatmatchata, eventualmente trasposta) sopra lo strumentale di A per
    ``bars`` battute da ``cue_out_a``; poi swap completo sul downbeat: strumentale di B pieno,
    bassi di A via con lo swap, medi/alti di A via in ``swap_release_beats``.

    Durante l'acapella i medi dello strumentale di A scendono di ``mid_cut_db`` (spazio alla voce,
    pratica di mixaggio [P]); la voce di A è tolta (mix − stem) con una rampa di
    ``vocal_mute_ms`` che termina sul downbeat d'ingresso. Anacrusi di B (``pickup_beats``)
    inclusa se lo stem di B è attivo nel battito prima del cue.
    """
    cfg = config or DJCreativeConfig()
    sr = int(sample_rate)
    if bars not in OVERLAP_BARS:
        raise DJRenderError(f"bars in {OVERLAP_BARS}")
    ctx = _prepare(audio_a, audio_b, vocals_a, vocals_b, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b, sr, cfg,
                   need_vocals_a=True, need_vocals_b=True)
    ia = ctx.ia
    pre, post = int(round(cfg.pre_s * sr)), int(round(cfg.post_s * sr))
    X = ctx.sa(ia)
    off_a = X - pre
    da_v, ba_v = _virtual_a_grid(ctx, ia, bars + 2)
    bm = _beatmatch(ctx, ctx.a, da_v, ba_v, float(ctx.da[ia]), bars, 2.0 * ctx.ta["bar_s"], key_shift_b, phase_nudge_s)
    t_s = pre
    t_sw = pre + int(round((da_v[ia + bars] - ctx.da[ia]) * sr))
    rel = int(round(swap_release_beats * ctx.ta["beat_s"] * sr))
    t_e = t_sw + max(rel, int(round(cfg.crossfade_ms / 1000 * sr)))
    n = t_e + post
    a_mix = _slice(ctx.a, off_a, n)
    a_voc = _slice(ctx.va, off_a, n)
    mute = _rise(n, t_s - int(round(cfg.vocal_mute_ms / 1000 * sr)), t_s, "sin")
    a_raw = a_mix - a_voc * mute[:, None]  # instrumental from t_s
    beat_out = int(round(ctx.ta["beat_s"] * sr))
    mid = 1.0 - (1.0 - _db(mid_cut_db)) * (_rise(n, t_s, t_s + beat_out, "cos") * (1.0 - _rise(n, t_sw - beat_out, t_sw, "cos")))
    swap = max(1, int(round(cfg.bass_swap_ms / 1000.0 * sr)))
    low_b = np.clip((np.arange(n) - (t_sw - swap)) / swap, 0.0, 1.0)
    fade_a = _fall_eqpow(n, t_sw, t_e)
    deck_a = _eq(a_raw, sr, cfg, low=1.0 - low_b, mid=mid * fade_a, high=fade_a) * (np.arange(n) < t_e)[:, None]
    pk = _pickup_samples(ctx, pickup_beats)
    b_voc = _fit(bm["b_voc"], n)
    b_ins = _fit(bm["b_instr"], n)
    gv = _rise(n, t_s - pk - int(0.02 * sr), t_s - pk, "sin")
    d = int(round(cfg.crossfade_ms / 1000 * sr))
    gi = np.sin(0.5 * np.pi * np.clip((np.arange(n) - (t_sw - d)) / max(d, 1), 0.0, 1.0))
    b_ins_eq = _eq(b_ins, sr, cfg, low=low_b, mid=gi, high=gi)
    bar_out = ctx.ta["bar_s"]
    b_ref = (b_voc + b_ins)[t_sw : t_sw + int(round(cfg.section_gain.b_bars * bar_out * sr))]
    sg_curve, sg_info = _section_gain(ctx, X, b_ref, t_sw, bar_out, n, t_e, "stretched full B from the swap downbeat")
    deck_b = (b_voc * gv[:, None] + b_ins_eq) * sg_curve[:, None]
    va_heard = a_voc * (1.0 - mute)[:, None]
    vb_heard = b_voc * (gv * sg_curve)[:, None]
    sbs = int(bm["switch_b_source_sample"])
    a_map = [{"out_start": 0, "out_end": t_s, "src_start": off_a, "rate": 1.0, "vocals": True, "what": "A full mix"},
             {"out_start": t_s, "out_end": t_e, "src_start": X, "rate": 1.0, "vocals": False, "what": "A instrumental (mix - stem)"}]
    b_map = [{"out_start": t_s - pk, "out_end": n, "src_start": None, "rate": bm["stretch"].get("tempo_ratio_b"), "vocals": True,
              "what": "B beatmatched: vocal stem from the entry (pickup included), full B from the swap"}]
    creative = {
        "technique": "acapella_over",
        "a_vocal_mute_guard": mute_guard(ctx, X),
        "params": {"bars": bars, "key_shift_b": key_shift_b, "mid_cut_db": mid_cut_db, "swap_release_beats": swap_release_beats,
                   "pickup_beats_used": pk / max(beat_out, 1)},
        "regions": [{"name": "B acapella over A instrumental", "output": [t_s - pk, t_sw]},
                    {"name": "swap: B instrumental in, A bass out", "output": [t_sw, t_sw]},
                    {"name": "A mids/highs release", "output": [t_sw, t_e]}],
        "beatmatch_alignment": bm["alignment"],
        "basis": "[P] acapella over instrumental (Serato Stems, rekordbox/Traktor stems; MosaikBox [S]); [E] Davies 2014: overlapping vocals are the main mashup failure, so one voice at a time; mid cut [P]",
    }
    return _finish(ctx, "acapella_over", deck_a=deck_a, deck_b=deck_b, overlay=None, t_s=t_s, t_sw=t_sw, t_e=t_e, off_a=off_a,
                   sb_at_switch=sbs, stretch=bm["stretch"], creative=creative, exit_effect=None, a_map=a_map, b_map=b_map,
                   vocals_a_heard=va_heard, vocals_b_heard=vb_heard, sg_info=sg_info,
                   extra_decks={"a_raw": _slice(ctx.a, off_a, n), "b_stretched_raw": _fit(bm["b_full"], n)},
                   provenance=provenance, return_decks=return_decks)


def vocal_swap(audio_a, audio_b, *, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b,
               vocals_a=None, vocals_b=None, bars: int = 4, out_bars: int = 2, key_shift_b: int = 0, pickup_beats: int = 1,
               phase_nudge_s: float = 0.0, sample_rate: int = dj.DEFAULT_SAMPLE_RATE, config: Optional[DJCreativeConfig] = None,
               provenance: Optional[dict] = None, return_decks: bool = False) -> DJRenderResult:
    """A fine verso di A la voce passa a B (stem) mentre la base resta di A per ``bars`` battute;
    lo strumentale di B entra intanto senza bassi e con gli alti attenuati (curve del blend di
    dj_render), bass swap sul downbeat dopo ``bars`` battute, poi lo strumentale di A esce in
    ``out_bars`` battute (quarto di coseno, alti che scendono). Una sola voce alla volta."""
    cfg = config or DJCreativeConfig()
    sr = int(sample_rate)
    if bars not in OVERLAP_BARS or out_bars not in (1, 2, 4):
        raise DJRenderError(f"bars in {OVERLAP_BARS}, out_bars in (1, 2, 4)")
    ctx = _prepare(audio_a, audio_b, vocals_a, vocals_b, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b, sr, cfg,
                   need_vocals_a=True, need_vocals_b=True)
    ia = ctx.ia
    pre, post = int(round(cfg.pre_s * sr)), int(round(cfg.post_s * sr))
    X = ctx.sa(ia)
    off_a = X - pre
    total = bars + out_bars
    da_v, ba_v = _virtual_a_grid(ctx, ia, total + 2)
    # Beatmatch over the vocal-swap bars; B stays at the mean ratio afterwards (constant grids).
    bm = _beatmatch(ctx, ctx.a, da_v, ba_v, float(ctx.da[ia]), bars, (out_bars + 1) * ctx.ta["bar_s"], key_shift_b, phase_nudge_s)
    t_s = pre
    t_sw = pre + int(round((da_v[ia + bars] - ctx.da[ia]) * sr))
    t_e = pre + int(round((da_v[ia + total] - ctx.da[ia]) * sr))
    n = t_e + post
    a_mix = _slice(ctx.a, off_a, n)
    a_voc = _slice(ctx.va, off_a, n)
    mute = _rise(n, t_s - int(round(cfg.vocal_mute_ms / 1000 * sr)), t_s, "sin")
    a_raw = a_mix - a_voc * mute[:, None]
    g = _blend_gains(n, t_s, t_sw, t_e, sr, cfg)
    deck_a = _eq(a_raw, sr, cfg, low=g["low_a"], high=g["high_a"]) * g["fader_a"][:, None]
    pk = _pickup_samples(ctx, pickup_beats)
    b_voc = _fit(bm["b_voc"], n)
    b_ins = _fit(bm["b_instr"], n)
    gv = _rise(n, t_s - pk - int(0.02 * sr), t_s - pk, "sin")
    bar_out = ctx.ta["bar_s"]
    b_ref = (b_voc + b_ins)[t_sw : t_sw + int(round(cfg.section_gain.b_bars * bar_out * sr))]
    sg_curve, sg_info = _section_gain(ctx, X, b_ref, t_sw, bar_out, n, t_e, "stretched full B from the bass swap downbeat")
    deck_b = (b_voc * gv[:, None] + _eq(b_ins, sr, cfg, low=g["low_b"], high=g["high_b"]) * g["fader_b"][:, None]) * sg_curve[:, None]
    va_heard = a_voc * (1.0 - mute)[:, None]
    vb_heard = b_voc * (gv * sg_curve)[:, None]
    a_map = [{"out_start": 0, "out_end": t_s, "src_start": off_a, "rate": 1.0, "vocals": True, "what": "A full mix"},
             {"out_start": t_s, "out_end": t_e, "src_start": X, "rate": 1.0, "vocals": False, "what": "A instrumental (mix - stem)"}]
    b_map = [{"out_start": t_s - pk, "out_end": n, "src_start": None, "rate": bm["stretch"].get("tempo_ratio_b"), "vocals": True,
              "what": "B beatmatched (vocal from the entry, instrumental blended in)"}]
    creative = {
        "technique": "vocal_swap",
        "a_vocal_mute_guard": mute_guard(ctx, X),
        "params": {"bars": bars, "out_bars": out_bars, "key_shift_b": key_shift_b, "pickup_beats_used": pk / max(1, int(round(ctx.ta['beat_s'] * sr)))},
        "regions": [{"name": "B vocal over A beat, B instrumental creeping in (no bass)", "output": [t_s - pk, t_sw]},
                    {"name": "bass swap", "output": [t_sw, t_sw]}, {"name": "A instrumental out", "output": [t_sw, t_e]}],
        "beatmatch_alignment": bm["alignment"],
        "basis": "[P] vocal swap with stems (Serato Stems / Traktor / djay); [S] Vande Veire 2018 bass swap; one voice at a time [E] Brungart 2001",
    }
    return _finish(ctx, "vocal_swap", deck_a=deck_a, deck_b=deck_b, overlay=None, t_s=t_s, t_sw=t_sw, t_e=t_e, off_a=off_a,
                   sb_at_switch=int(bm["switch_b_source_sample"]), stretch=bm["stretch"], creative=creative, exit_effect=None,
                   a_map=a_map, b_map=b_map, vocals_a_heard=va_heard, vocals_b_heard=vb_heard, sg_info=sg_info,
                   extra_decks={"a_raw": _slice(ctx.a, off_a, n), "b_stretched_raw": _fit(bm["b_full"], n)},
                   provenance=provenance, return_decks=return_decks)


ROLL_SCHEDULES = {1: (0.25,), 2: (0.5, 0.25), 4: (1.0, 0.5, 0.25, 0.125)}


def build_into_chorus(audio_a, audio_b, *, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b,
                      vocals_a=None, vocals_b=None, key_b: Optional[str] = None, build_bars: int = 2,
                      snare: str = "auto", riser: bool = True, stutter_bars: int = 0, gap_beats: float = 0.0,
                      hpf_end_hz: float = 1200.0, roll_level_db: float = -6.0, riser_level_db: float = -8.0,
                      sample_rate: int = dj.DEFAULT_SAMPLE_RATE, config: Optional[DJCreativeConfig] = None,
                      provenance: Optional[dict] = None, return_decks: bool = False) -> DJRenderResult:
    """Build di ``build_bars`` battute su A prima del downbeat del ritornello di B, al posto del drop
    secco (votato 3/10): passa-alto crescente su A (30 Hz -> ``hpf_end_hz``), rullo di rullante
    sulla griglia di A (suddivisioni che si dimezzano a ogni battuta, crescendo da -14 dB a
    ``roll_level_db`` rispetto ad A), riser nella tonalità di B (glissando che arriva sulla tonica),
    facoltativi stutter dell'ultima battuta di A (``stutter_bars``) e pausa prima del drop
    (``gap_beats``). B entra pulito sul suo downbeat (nessuno stretch)."""
    cfg = config or DJCreativeConfig()
    sr = int(sample_rate)
    if build_bars not in BUILD_BARS or snare not in ("auto", "extract", "synth", "none") or stutter_bars not in (0, 1):
        raise DJRenderError("build_bars in (1, 2, 4); snare in auto/extract/synth/none; stutter_bars in (0, 1)")
    ctx = _prepare(audio_a, audio_b, vocals_a, vocals_b, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b, sr, cfg)
    ia = ctx.ia
    if ia - build_bars < 0:
        raise DJRenderError("build: not enough A bars before cue_out_a")
    bars = np.diff(ctx.da[ia - build_bars : ia + 1])
    if np.any(np.abs(bars / ctx.ta["bar_s"] - 1.0) > cfg.max_bar_deviation):
        raise DJRenderError("build: irregular A bars")
    pre, post = int(round(cfg.pre_s * sr)), int(round(cfg.post_s * sr))
    S = ctx.sa(ia - build_bars)
    X = ctx.sa(ia)
    off_a = S - pre
    t_s, t_sw = pre, X - off_a
    t_e = t_sw
    n = t_e + post
    a_raw = _slice(ctx.a, off_a, n)
    # A: HPF sweep (as dj_effects.filter_out), optional stutter of the last bar, optional gap.
    fc = np.where(np.arange(n) < t_s, 30.0, 30.0 * (hpf_end_hz / 30.0) ** np.clip((np.arange(n) - t_s) / max(t_sw - t_s, 1), 0, 1))
    filt = fx.time_varying_filter(a_raw, sr, fc, "highpass", 4, 128)
    eng = _rise(n, t_s, t_s + int(0.03 * sr))[:, None]
    deck_a_pre = a_raw * (1 - eng) + filt * eng
    stutter_info = None
    if stutter_bars:
        b0 = ctx.sa(ia - 1) - off_a
        beat = int(round(ctx.ta["beat_s"] * sr))
        bpb = ctx.ta["beats_per_bar"]
        st = deck_a_pre.copy()
        slices = []
        for k in range(bpb):
            sub = beat // (2 ** (k + 1))
            seg = deck_a_pre[b0 : b0 + sub].copy()
            ramp = min(int(0.002 * sr), sub // 4)
            if ramp > 0:
                seg[:ramp] *= np.linspace(0, 1, ramp)[:, None]
                seg[-ramp:] *= np.linspace(1, 0, ramp)[:, None]
            for r in range(2 ** (k + 1)):
                p = b0 + k * beat + r * sub
                st[p : p + sub] = seg[: max(0, min(sub, n - p))]
            slices.append(sub / sr)
        deck_a_pre = st
        stutter_info = {"bar_output": [b0, b0 + bpb * beat], "slice_s_per_beat": slices,
                        "source": "first 1/2, 1/4, 1/8, 1/16 beat of A's last bar (filtered), repeated (loop roll)"}
    gap = int(round(gap_beats * ctx.ta["beat_s"] * sr))
    deck_a = deck_a_pre * _a_cut(n, t_sw - gap if gap else t_sw, cfg, sr)[:, None]
    # overlays: snare roll + riser, on A's grid, in B's key.
    a_ref_rms = _rms(a_raw[max(0, t_s - int(2 * ctx.ta["bar_s"] * sr)) : t_s])
    ov_parts = {}
    beats_out = ctx.grid["beats_a"] - off_a / sr
    bar_starts = [(ctx.da[ia - build_bars + k] * sr - off_a) / sr for k in range(build_bars)]
    snare_info: dict = {"mode": snare}
    if snare != "none":
        sample = None
        if snare in ("auto", "extract"):
            bb = [t for t in ctx.grid["beats_a"] if S / sr - 8 * ctx.ta["bar_s"] <= t < S / sr]
            back = [t for t in bb if int(round((t - ctx.da[ia]) / ctx.ta["beat_s"])) % ctx.ta["beats_per_bar"] in (1, 3)]
            instr = ctx.a - ctx.va if ctx.has_va else ctx.a
            sample, ex = extract_snare(instr, sr, back)
            snare_info["extraction"] = ex
            if sample is None and snare == "extract":
                raise DJRenderError(f"snare extraction failed: {ex.get('status')}")
        if sample is None:
            sample = synth_snare(sr)
            snare_info["source"] = "synthesized (noise 1-8 kHz + 190 Hz body)"
        else:
            snare_info["source"] = "extracted from A"
        ev = roll_events(beats_out, t_s / sr, (t_sw - gap) / sr, ROLL_SCHEDULES[build_bars], bar_starts)
        roll = np.zeros((n, 2))
        for t, _k in ev:
            u = (t * sr - t_s) / max(t_sw - t_s, 1)
            amp = _db(-14.0 + (roll_level_db + 14.0) * u)
            p = int(round(t * sr))
            seg = sample[: max(0, min(len(sample), n - p))]
            roll[p : p + len(seg)] += amp * seg
        roll *= a_ref_rms / max(_rms(sample[: int(0.05 * sr)]), 1e-9) * 0.5
        ov_parts["snare_roll"] = roll
        snare_info.update(events=len(ev), schedule_beats=ROLL_SCHEDULES[build_bars], crescendo_db=[-14.0, roll_level_db])
    riser_info = None
    if riser:
        kb = parse_key(key_b)
        root = (kb[0] if kb else 9)
        root_midi = 48 + root  # octave 3
        L = t_sw - t_s - gap
        r = synth_riser(L, sr, root_midi + 12, glide_semitones=12.0)
        env = _db(-30.0 + 30.0 * (np.arange(L) / max(L - 1, 1)) ** 1.5)
        riser_sig = np.zeros((n, 2))
        riser_sig[t_s : t_s + L] = r * env[:, None]
        tail = int(0.005 * sr)
        riser_sig[t_s + L - tail : t_s + L] *= np.linspace(1, 0, tail)[:, None]
        riser_sig *= a_ref_rms / max(_rms(r[-int(0.5 * sr):]), 1e-9) * _db(riser_level_db)
        ov_parts["riser"] = riser_sig
        riser_info = {"key_b": key_b, "tonic": PITCH_CLASSES[root], "lands_on_midi": root_midi + 12, "glide_semitones": 12.0,
                      "lowpass_opens_hz": [400, 6000], "noise_highpass_hz": [300, 6000], "level_db_rel_a": riser_level_db,
                      "key_known": kb is not None}
    overlay = sum(ov_parts.values()) if ov_parts else None
    deck_b = _b_entry(ctx.b, ctx.sb(), t_sw, n, cfg, sr)
    vb_src = _b_entry(ctx.vb, ctx.sb(), t_sw, n, cfg, sr)
    b_ref = ctx.b[ctx.sb() : ctx.sb() + int(round(cfg.section_gain.b_bars * ctx.tb["bar_s"] * sr))]
    sg_curve, sg_info = _section_gain(ctx, S, b_ref, t_sw, ctx.tb["bar_s"], n, t_sw, "B source (no stretch) from its chorus downbeat; A before the build (dry)")
    deck_b = deck_b * sg_curve[:, None]
    va_heard = _slice(ctx.va, off_a, n) * _a_cut(n, t_sw - gap if gap else t_sw, cfg, sr)[:, None]
    vb_heard = vb_src * sg_curve[:, None]
    a_map = [{"out_start": 0, "out_end": t_sw, "src_start": off_a, "rate": 1.0, "vocals": True, "what": "A (high-passed during the build)"}]
    if stutter_info is not None:  # 21/10/2026: the map now shows the stutter (A's continuous playback stops inside the last bar)
        b0s, b1s = stutter_info["bar_output"]
        a_map = [{"out_start": 0, "out_end": b0s, "src_start": off_a, "rate": 1.0, "vocals": True, "what": "A (high-passed during the build)"}]
        for k in range(bpb):
            sub = beat // (2 ** (k + 1))
            for r in range(2 ** (k + 1)):
                p0 = b0s + k * beat + r * sub
                if p0 >= t_sw:
                    continue
                a_map.append({"out_start": p0, "out_end": min(p0 + sub, t_sw), "src_start": b0s + off_a, "rate": 1.0, "vocals": True,
                              "what": f"stutter: beat {k + 1}, slice 1/{2 ** (k + 1)} beat, repeat {r + 1}"})
        if b1s < t_sw:
            a_map.append({"out_start": b1s, "out_end": t_sw, "src_start": b1s + off_a, "rate": 1.0, "vocals": True, "what": "A after the stutter bar"})
    b_map = [{"out_start": t_sw, "out_end": n, "src_start": ctx.sb(), "rate": 1.0, "vocals": True, "what": "B from its chorus downbeat"}]
    creative = {
        "technique": "build_into_chorus",
        "params": {"build_bars": build_bars, "snare": snare, "riser": riser, "stutter_bars": stutter_bars, "gap_beats": gap_beats,
                   "hpf_end_hz": hpf_end_hz},
        "hpf": {"start_hz": 30.0, "end_hz": hpf_end_hz, "order": 4, "curve": "geometric over the build, causal Butterworth (dj_effects.time_varying_filter)"},
        "snare_roll": snare_info, "riser": riser_info, "stutter": stutter_info,
        "regions": [{"name": "build (HPF + roll + riser)", "output": [t_s, t_sw]}, {"name": "B chorus downbeat", "output": [t_sw, t_sw]}],
        "basis": "[P] build-up / snare roll / riser before the drop (EDM and open-format practice); [E] Solberg & Dibben 2019 (EDM): drop more arousing after a build; [U] direct drop on B's chorus 3/10",
    }
    return _finish(ctx, "build_into_chorus", deck_a=deck_a, deck_b=deck_b, overlay=overlay, t_s=t_s, t_sw=t_sw, t_e=t_e,
                   off_a=off_a, sb_at_switch=ctx.sb(), stretch=_tempo_block(ctx, False), creative=creative, exit_effect="filter",
                   a_map=a_map, b_map=b_map, vocals_a_heard=va_heard, vocals_b_heard=vb_heard, sg_info=sg_info,
                   extra_decks={"overlay_parts": ov_parts, "a_filtered_prefader": deck_a_pre}, provenance=provenance,
                   return_decks=return_decks)


# ------------------------------------------- libri (21/10/2026): build per dimezzamento, break prima di B
HALVING_BUILD_BARS = (2, 4, 8)
#: techniques without overlap added on 21/10/2026: qa_inputs leaves deck B out (rhythm checked on the mix at the switch)
NO_OVERLAP_QA_MIX_ONLY = ("halving_build", "break_before_entry")
BREAK_MODES = ("stop", "beat_mute", "fill")
BREAK_BEATS = (1, 2, 4)


def halving_stages(build_bars: float, start_unit: float = 2.0, min_unit: float = 0.25) -> list[dict[str, float]]:
    """Stadi del build per dimezzamento allineati alla griglia: unità ``start_unit``, ``start_unit``/2, ...
    fino a ``min_unit`` battute; lo stadio k dura ``build_bars`` / 2^(k+1) battute (l'ultimo prende il
    resto), così ogni confine cade su un downbeat o su un battito e ogni unità si ripete almeno 2 volte
    (Lidov in Margulis 2014 p. 51: due enunciati stabiliscono il pattern). Se il primo stadio non contiene
    2 ripetizioni dell'unità, l'unità iniziale si dimezza. [A] Margulis 2014 p. 23 (unità di 2 battute,
    poi 1, ½, due note: "the cumulative effect is of acceleration"), p. 44; [H] durate degli stadi."""
    u = float(start_unit)
    while u > min_unit and build_bars / 2.0 < 2.0 * u:
        u /= 2.0
    units = []
    while u >= min_unit - 1e-9:
        units.append(u)
        u /= 2.0
    if not units:
        raise DJRenderError("halving_stages: min_unit larger than the build")
    stages, t = [], 0.0
    for k, un in enumerate(units):
        span = build_bars - t if k == len(units) - 1 else build_bars / 2.0 ** (k + 1)
        reps = span / un
        if reps < 2 - 1e-9:  # span too short for this unit: stop here and give the rest to the previous stage
            stages[-1]["span_bars"] += build_bars - t
            stages[-1]["repeats"] = stages[-1]["span_bars"] / stages[-1]["unit_bars"]
            t = build_bars
            break
        stages.append({"start_bar": t, "unit_bars": un, "repeats": reps, "span_bars": span})
        t += span
    return stages


def halving_build(audio_a, audio_b, *, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b,
                  vocals_a=None, vocals_b=None, build_bars: int = 4, start_unit: float = 1.0, min_unit: float = 0.25,
                  source: str = "auto", trajectory: str = "hpf_rise", hpf_end_hz: float = 800.0, gap_beats: float = 0.0,
                  sample_rate: int = dj.DEFAULT_SAMPLE_RATE, config: Optional[DJCreativeConfig] = None,
                  provenance: Optional[dict] = None, return_decks: bool = False) -> DJRenderResult:
    """Build per dimezzamento del loop di A prima del downbeat di B (proposta C1 dei libri).

    Nelle ultime ``build_bars`` battute prima di ``cue_out_a`` A suona un loop che parte dal downbeat
    d'inizio build e si accorcia per stadi (:func:`halving_stages`, es. 8 battute: 2+2, 1+1, ½+½,
    ¼×4): a tempo costante si sente un'accelerazione (Margulis 2014 pp. 23, 44 [A]; Huron 2006
    pp. 323-326 [T]: accelerazione degli eventi come indicatore di climax). ``trajectory='hpf_rise'``:
    passa-alto che sale fino a ``hpf_end_hz`` e massimo d'intensità sull'ultima iterazione (Margulis
    pp. 130-132 [A]). ``gap_beats`` (0, ½, 1, 2): pausa prima del downbeat di B (Huron pp. 314-316 [T];
    oltre 2 battiti si rischia di perdere la tendenza, p. 368). Sorgente del loop: ``mix``,
    ``instrumental`` (mix − stem vocale) o ``auto`` (strumentale se la voce di A copre più di
    ``vocal_active_cover`` dell'unità iniziale: parole ripetute = rischio "disco rotto", Huron p. 286).
    B entra pulito sul suo downbeat al suo tempo (nessuna sovrapposizione, come ``build_into_chorus``).
    """
    cfg = config or DJCreativeConfig()
    sr = int(sample_rate)
    if build_bars not in HALVING_BUILD_BARS or trajectory not in ("hpf_rise", "none") or gap_beats not in (0, 0.0, 0.5, 1, 1.0, 2, 2.0):
        raise DJRenderError(f"halving_build: build_bars in {HALVING_BUILD_BARS}, trajectory hpf_rise|none, gap_beats in 0/0.5/1/2")
    if source not in ("auto", "mix", "instrumental"):
        raise DJRenderError("halving_build: source auto|mix|instrumental")
    ctx = _prepare(audio_a, audio_b, vocals_a, vocals_b, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b, sr, cfg,
                   need_vocals_a=(source == "instrumental"))
    ia = ctx.ia
    if ia - build_bars < 0:
        raise DJRenderError("halving_build: not enough A bars before cue_out_a")
    bars_src = np.diff(ctx.da[ia - build_bars : ia + 1])
    if np.any(np.abs(bars_src / ctx.ta["bar_s"] - 1.0) > cfg.max_bar_deviation):
        raise DJRenderError("halving_build: irregular A bars")
    stages = halving_stages(build_bars, start_unit, min_unit)
    S, X = ctx.sa(ia - build_bars), ctx.sa(ia)
    Lbar = (X - S) / build_bars  # samples per output bar (mean of the build bars)
    u0 = stages[0]["unit_bars"]
    cover = _cover(ctx.va, sr, S, S + int(round(u0 * Lbar)), ctx.va) if ctx.has_va else 0.0
    if source == "auto":
        source = "mix" if (not ctx.has_va or cover <= cfg.vocal_active_cover) else "instrumental"
    loop_src = ctx.a if source == "mix" else ctx.a - ctx.va
    loop_voc = ctx.va if source == "mix" else np.zeros_like(ctx.va)
    pre, post = int(round(cfg.pre_s * sr)), int(round(cfg.post_s * sr))
    off_a = S - pre
    t_s, t_sw = pre, X - off_a
    t_e = t_sw
    n = t_e + post
    a_raw = _slice(ctx.a, off_a, n)
    va_raw = _slice(ctx.va, off_a, n)
    deck_pre = a_raw.copy()
    va_heard = va_raw.copy()
    edge = max(1, int(round(0.002 * sr)))
    a_map = [{"out_start": 0, "out_end": t_s, "src_start": off_a, "rate": 1.0, "vocals": True, "what": "A before the build"}]
    pieces = []
    for st in stages:
        L = int(round(st["unit_bars"] * Lbar))
        reps = int(round(st["repeats"]))
        for r in range(reps):
            o0 = t_s + int(round((st["start_bar"] + r * st["unit_bars"]) * Lbar))
            o1 = min(t_sw, o0 + L)
            if o1 <= o0:
                continue
            seg = loop_src[S : S + (o1 - o0)].copy()
            vseg = loop_voc[S : S + (o1 - o0)].copy()
            if r > 0 or st is not stages[0]:  # repeated pieces: short edge ramps (no clicks at the seams)
                ramp = np.ones(len(seg))
                e = min(edge, len(seg) // 4)
                if e > 0:
                    ramp[:e] = np.linspace(0, 1, e)
                    ramp[-e:] = np.linspace(1, 0, e)
                seg *= ramp[:, None]
                vseg *= ramp[:, None]
            elif source == "instrumental":
                pass
            deck_pre[o0:o1] = seg
            va_heard[o0:o1] = vseg
            pieces.append({"output": [o0, o1], "unit_bars": st["unit_bars"], "repeat": r + 1})
            a_map.append({"out_start": o0, "out_end": o1, "src_start": S, "rate": 1.0, "vocals": source == "mix",
                          "what": f"halving loop {st['unit_bars']:g} bar, repeat {r + 1} ({source})"})
    if trajectory == "hpf_rise":
        fc = np.where(np.arange(n) < t_s, 30.0,
                      30.0 * (hpf_end_hz / 30.0) ** np.clip((np.arange(n) - t_s) / max(t_sw - t_s, 1), 0, 1))
        filt = fx.time_varying_filter(deck_pre, sr, fc, "highpass", 4, 128)
        eng = _rise(n, t_s, t_s + int(0.03 * sr))[:, None]
        deck_pre = deck_pre * (1 - eng) + filt * eng
    gap = int(round(gap_beats * ctx.ta["beat_s"] * sr))
    cut = _a_cut(n, t_sw - gap if gap else t_sw, cfg, sr)
    deck_a = deck_pre * cut[:, None]
    va_heard = va_heard * cut[:, None]
    deck_b = _b_entry(ctx.b, ctx.sb(), t_sw, n, cfg, sr)
    vb_src = _b_entry(ctx.vb, ctx.sb(), t_sw, n, cfg, sr)
    b_ref = ctx.b[ctx.sb() : ctx.sb() + int(round(cfg.section_gain.b_bars * ctx.tb["bar_s"] * sr))]
    sg_curve, sg_info = _section_gain(ctx, S, b_ref, t_sw, ctx.tb["bar_s"], n, t_sw, "B source (no stretch) from its downbeat; A before the build (dry)")
    deck_b = deck_b * sg_curve[:, None]
    b_map = [{"out_start": t_sw, "out_end": n, "src_start": ctx.sb(), "rate": 1.0, "vocals": True, "what": "B from its downbeat"}]
    from musicmix import dj_phrase as _P

    rep_checks = [{"unit_bars": st["unit_bars"], "repeats": st["repeats"],
                   **{k: v for k, v in _P.loop_repetition_check(max(1, int(round(st["unit_bars"]))), int(round(st["span_bars"])) or 1,
                                                                loop_has_vocal=source == "mix" and cover > cfg.vocal_active_cover,
                                                                new_layer_entering=trajectory == "hpf_rise").items()
                      if k in ("ok", "warnings")}} for st in stages]
    creative = {
        "technique": "halving_build",
        "params": {"build_bars": build_bars, "start_unit": start_unit, "min_unit": min_unit, "source": source,
                   "trajectory": trajectory, "hpf_end_hz": hpf_end_hz if trajectory == "hpf_rise" else None, "gap_beats": gap_beats},
        "stages": stages, "pieces": pieces, "a_vocal_cover_first_unit": cover,
        "a_vocal_mute_guard": mute_guard(ctx, S) if source == "instrumental" else {"status": "loop from the mix: no vocal mute"},
        "repetition_checks": rep_checks,
        "regions": [{"name": "halving build", "output": [t_s, t_sw - gap]}] + ([{"name": "stop before B", "output": [t_sw - gap, t_sw]}] if gap else [])
                   + [{"name": "B downbeat", "output": [t_sw, t_sw]}],
        "basis": "[A] Margulis 2014 p. 23, p. 44 (shrinking repeated unit heard as acceleration); [T] Huron 2006 pp. 323-326 "
                 "(accelerating events as climax cue); [A] Margulis pp. 130-132 (trajectory, intensity saved for the last iteration); "
                 "[T] Huron pp. 314-316 (delay before a predictable downbeat); [H] stage lengths, HPF curve, levels",
    }
    return _finish(ctx, "halving_build", deck_a=deck_a, deck_b=deck_b, overlay=None, t_s=t_s, t_sw=t_sw, t_e=t_e, off_a=off_a,
                   sb_at_switch=ctx.sb(), stretch=_tempo_block(ctx, False), creative=creative,
                   exit_effect="filter" if trajectory == "hpf_rise" else None, a_map=a_map, b_map=b_map,
                   vocals_a_heard=va_heard, vocals_b_heard=vb_src * sg_curve[:, None], sg_info=sg_info,
                   extra_decks={"a_build_prefader": deck_pre}, provenance=provenance, return_decks=return_decks)


def break_before_entry(audio_a, audio_b, *, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b,
                       vocals_a=None, vocals_b=None, mode: str = "stop", beats: int = 1, fill_level_db: float = -6.0,
                       snare: str = "auto", sample_rate: int = dj.DEFAULT_SAMPLE_RATE, config: Optional[DJCreativeConfig] = None,
                       provenance: Optional[dict] = None, return_decks: bool = False) -> DJRenderResult:
    """Break di ``beats`` battiti (1, 2, 4) alla fine della frase di A, poi B pulito sul suo downbeat.

    * ``stop``: A tace del tutto negli ultimi battiti (Huron 2006 pp. 314-316 [T]: ritardare un evento
      molto prevedibile ne aumenta la tensione; Krims 2000 pp. 137-139 [A]: lo stop delle tracce è
      l'evento più marcato; Butler 2006 pp. 91, 188 [A]: rimozioni anacrusiche prima del downbeat);
    * ``beat_mute``: tace solo lo strumentale di A (mix − stem), la voce resta: la punchline cade nel
      vuoto (Katz 2012 p. 106 [P]: "his punchline was on the fourth bar and to drop out"; Krims 2000
      p. 139 [A]: tutte le tracce spariscono e resta la voce, evento "drammatico"); serve lo stem;
    * ``fill``: A suona e un fill di rullante a sedicesimi in crescendo riempie gli ultimi battiti
      (Huron pp. 248-249 [A]: il fill rende certo il "quando" del downbeat ipermetrico; "At the Hop":
      fill dal 2° battito a sedicesimi). Rullante estratto da A (``extract_snare``) o sintetico.

    Guardia: ``mute_guard`` riporta la voce di A al punto in cui il break comincia (``stop`` taglia anche
    la voce). Livelli e durate [H].
    """
    cfg = config or DJCreativeConfig()
    sr = int(sample_rate)
    if mode not in BREAK_MODES or beats not in BREAK_BEATS or snare not in ("auto", "extract", "synth"):
        raise DJRenderError(f"break_before_entry: mode in {BREAK_MODES}, beats in {BREAK_BEATS}, snare auto|extract|synth")
    ctx = _prepare(audio_a, audio_b, vocals_a, vocals_b, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b, sr, cfg,
                   need_vocals_a=(mode == "beat_mute"))
    if ctx.ia < 1:
        raise DJRenderError("break_before_entry: no A bar before cue_out_a")
    X = ctx.sa()
    beat = int(round(ctx.ta["beat_s"] * sr))
    B0 = X - beats * beat
    pre, post = int(round(cfg.pre_s * sr)), int(round(cfg.post_s * sr))
    S = ctx.sa(ctx.ia - 1)  # transition region = A's last bar
    off_a = S - pre
    t_s, t_sw = pre, X - off_a
    t_e = t_sw
    n = t_e + post
    b0 = B0 - off_a
    a_raw = _slice(ctx.a, off_a, n)
    va_raw = _slice(ctx.va, off_a, n)
    d = int(round(cfg.crossfade_ms / 1000.0 * sr))
    keep = np.cos(0.5 * np.pi * np.clip((np.arange(n) - (b0 - d)) / max(d, 1), 0.0, 1.0))  # 1 -> 0 ending on the break
    keep[np.arange(n) >= b0] = 0.0
    cut = _a_cut(n, t_sw, cfg, sr)
    overlay = None
    ov_parts: dict = {}
    fill_info = None
    if mode == "stop":
        deck_a = a_raw * keep[:, None]
        va_heard = va_raw * keep[:, None]
    elif mode == "beat_mute":
        inst = a_raw - va_raw
        deck_a = (inst * keep[:, None] + va_raw) * cut[:, None]
        va_heard = va_raw * cut[:, None]
    else:
        deck_a = a_raw * cut[:, None]
        va_heard = va_raw * cut[:, None]
        sample = None
        ex = {"status": "synth requested"}
        if snare in ("auto", "extract"):
            bb = [t for t in ctx.grid["beats_a"] if S / sr - 8 * ctx.ta["bar_s"] <= t < S / sr]
            back = [t for t in bb if int(round((t - ctx.da[ctx.ia]) / ctx.ta["beat_s"])) % ctx.ta["beats_per_bar"] in (1, 3)]
            instr = ctx.a - ctx.va if ctx.has_va else ctx.a
            sample, ex = extract_snare(instr, sr, back)
            if sample is None and snare == "extract":
                raise DJRenderError(f"snare extraction failed: {ex.get('status')}")
        src_name = "extracted from A" if sample is not None else "synthesized"
        if sample is None:
            sample = synth_snare(sr)
        a_ref_rms = _rms(a_raw[max(0, t_s - int(2 * ctx.ta["bar_s"] * sr)) : t_s])
        roll = np.zeros((n, 2))
        steps = beats * 4
        for k in range(steps):
            p = b0 + int(round(k * beat / 4.0))
            amp = _db(-14.0 + (fill_level_db + 14.0) * (k / max(steps - 1, 1)))
            seg = sample[: max(0, min(len(sample), t_sw - p))]
            roll[p : p + len(seg)] += amp * seg
        roll *= a_ref_rms / max(_rms(sample[: int(0.05 * sr)]), 1e-9) * 0.5
        ov_parts["snare_fill"] = roll
        overlay = roll
        fill_info = {"events": steps, "subdivision_beats": 0.25, "crescendo_db": [-14.0, fill_level_db], "source": src_name, "extraction": ex}
    deck_b = _b_entry(ctx.b, ctx.sb(), t_sw, n, cfg, sr)
    vb_src = _b_entry(ctx.vb, ctx.sb(), t_sw, n, cfg, sr)
    b_ref = ctx.b[ctx.sb() : ctx.sb() + int(round(cfg.section_gain.b_bars * ctx.tb["bar_s"] * sr))]
    sg_curve, sg_info = _section_gain(ctx, S, b_ref, t_sw, ctx.tb["bar_s"], n, t_sw, "B source (no stretch) from its downbeat; A's last bar before the break")
    deck_b = deck_b * sg_curve[:, None]
    a_map = [{"out_start": 0, "out_end": t_sw, "src_start": off_a, "rate": 1.0, "vocals": True,
              "what": f"A ({mode} over the last {beats} beat(s))"}]
    b_map = [{"out_start": t_sw, "out_end": n, "src_start": ctx.sb(), "rate": 1.0, "vocals": True, "what": "B from its downbeat"}]
    creative = {
        "technique": "break_before_entry",
        "params": {"mode": mode, "beats": beats, "fill_level_db": fill_level_db if mode == "fill" else None},
        "a_vocal_at_break": mute_guard(ctx, B0),
        "a_vocal_across_downbeat": mute_guard(ctx, X),  # words of A still running at B's downbeat are cut (dj_qa verse_cut)
        "fill": fill_info,
        "regions": [{"name": f"{mode} ({beats} beat(s))", "output": [b0, t_sw]}, {"name": "B downbeat", "output": [t_sw, t_sw]}],
        "basis": {"stop": "[T] Huron 2006 pp. 314-316; [A] Krims 2000 pp. 137-139; [A] Butler 2006 pp. 91, 188",
                  "beat_mute": "[P] Katz 2012 p. 106 (Jazzy Jeff: drop out on the punchline); [A] Krims 2000 p. 139 (all tracks vanish, the voice "
                               "stays: a marked event); [A] Butler 2006 pp. 91, 188",
                  "fill": "[A] Huron 2006 pp. 248-249 (fills make the hypermetric downbeat certain); [P] Webber 2008 p. 260 (drum tip as fill into a downbeat)"}[mode]
                 + "; [H] durations and levels",
    }
    return _finish(ctx, "break_before_entry", deck_a=deck_a, deck_b=deck_b, overlay=overlay, t_s=t_s, t_sw=t_sw, t_e=t_e,
                   off_a=off_a, sb_at_switch=ctx.sb(), stretch=_tempo_block(ctx, False), creative=creative, exit_effect=None,
                   a_map=a_map, b_map=b_map, vocals_a_heard=va_heard, vocals_b_heard=vb_src * sg_curve[:, None], sg_info=sg_info,
                   extra_decks={"overlay_parts": ov_parts} if ov_parts else {}, provenance=provenance, return_decks=return_decks,
                   trim_region=(b0, t_sw) if mode == "fill" else None)


def note_bridge(audio_a, audio_b, *, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b,
                vocals_a=None, vocals_b=None, key_a: Optional[str] = None, key_b: Optional[str] = None,
                pad_bars_before: int = 1, gap_bars: int = 0, pad_bars_after: int = 2, a_fade_beats: int = 2,
                source: str = "synth", pad_level_db: float = -8.0, sample_rate: int = dj.DEFAULT_SAMPLE_RATE,
                config: Optional[DJCreativeConfig] = None, provenance: Optional[dict] = None, return_decks: bool = False) -> DJRenderResult:
    """Nota/pad sostenuto nella tonalità comune che copre la giunta.

    Il pad (``synth``: accordo di :func:`bridge_chord`; ``freeze_a``: freeze spettrale dell'ultimo
    battito della componente armonica di A) entra ``pad_bars_before`` battute prima dell'uscita di
    A, A sfuma negli ultimi ``a_fade_beats`` battiti, dopo ``gap_bars`` battute di solo pad B entra
    pulito sul suo downbeat al suo tempo, il pad sfuma in ``pad_bars_after`` battute di B.
    Utile soprattutto con tempi lontani (nessun beatmatch) e tonalità compatibili."""
    cfg = config or DJCreativeConfig()
    sr = int(sample_rate)
    if source not in ("synth", "freeze_a") or gap_bars not in (0, 1) or pad_bars_before not in (1, 2) or pad_bars_after not in (1, 2, 4):
        raise DJRenderError("note_bridge: source synth|freeze_a, gap_bars 0|1, pad_bars_before 1|2, pad_bars_after 1|2|4")
    ctx = _prepare(audio_a, audio_b, vocals_a, vocals_b, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b, sr, cfg)
    ia = ctx.ia
    if ia - pad_bars_before < 0:
        raise DJRenderError("note_bridge: not enough A bars before cue_out_a")
    pre, post = int(round(cfg.pre_s * sr)), int(round(cfg.post_s * sr))
    P = ctx.sa(ia - pad_bars_before)
    X = ctx.sa(ia)
    off_a = P - pre
    t_s, t_x = pre, X - off_a
    t_sw = t_x + int(round(gap_bars * ctx.ta["bar_s"] * sr))
    t_e = t_sw + int(round(pad_bars_after * ctx.tb["bar_s"] * sr))
    n = t_e + post
    fade = int(round(a_fade_beats * ctx.ta["beat_s"] * sr))
    ga = _fall_eqpow(n, t_x - fade, t_x)
    a_raw = _slice(ctx.a, off_a, n)
    deck_a = a_raw * ga[:, None]
    chord = None
    if source == "synth":
        chord = bridge_chord(key_a, key_b)
        pad = synth_pad(chord["midi"], n - t_s, sr, attack=t_x - t_s, release_start=t_sw - t_s, release_end=t_e - t_s)
        pad_sig = np.zeros((n, 2))
        pad_sig[t_s:] = pad
        src_info = {"source": "synth", "chord": chord}
    else:
        beat = int(round(ctx.ta["beat_s"] * sr))
        seg = ctx.a[X - beat : X] - (ctx.va[X - beat : X] if ctx.has_va else 0.0)
        try:
            import librosa

            seg = np.stack([librosa.effects.harmonic(seg[:, c].astype(np.float32), margin=2.0) for c in range(2)], 1).astype(np.float64)
        except Exception:  # pragma: no cover - librosa is in the project venv
            pass
        fr = spectral_freeze(seg, sr, n - t_s)
        env = _rise(n - t_s, 0, t_x - t_s, "cos") * (1.0 - _rise(n - t_s, t_sw - t_s, t_e - t_s, "cos"))
        pad_sig = np.zeros((n, 2))
        pad_sig[t_s:] = fr * env[:, None]
        pad_sig /= max(1e-9, np.max(np.abs(pad_sig)))
        src_info = {"source": "freeze_a", "captured_a_source_samples": [X - beat, X], "harmonic_component": "librosa HPSS (margin 2) of A instrumental"}
    a_ref_rms = _rms(a_raw[max(0, t_s - int(2 * ctx.ta["bar_s"] * sr)) : t_s])
    pad_sig *= a_ref_rms / max(_rms(pad_sig[t_x - fade : t_sw + 1]), 1e-9) * _db(pad_level_db)
    deck_b = _b_entry(ctx.b, ctx.sb(), t_sw, n, cfg, sr)
    vb_src = _b_entry(ctx.vb, ctx.sb(), t_sw, n, cfg, sr)
    b_ref = ctx.b[ctx.sb() : ctx.sb() + int(round(cfg.section_gain.b_bars * ctx.tb["bar_s"] * sr))]
    sg_curve, sg_info = _section_gain(ctx, P, b_ref, t_sw, ctx.tb["bar_s"], n, t_e, "B source (no stretch) from its entry downbeat; A before the pad (dry)")
    deck_b = deck_b * sg_curve[:, None]
    va_heard = _slice(ctx.va, off_a, n) * ga[:, None]
    vb_heard = vb_src * sg_curve[:, None]
    a_map = [{"out_start": 0, "out_end": t_x, "src_start": off_a, "rate": 1.0, "vocals": True, "what": "A, fading under the pad"}]
    b_map = [{"out_start": t_sw, "out_end": n, "src_start": ctx.sb(), "rate": 1.0, "vocals": True, "what": "B at its own tempo"}]
    creative = {
        "technique": "note_bridge",
        "params": {"pad_bars_before": pad_bars_before, "gap_bars": gap_bars, "pad_bars_after": pad_bars_after, "a_fade_beats": a_fade_beats,
                   "source": source, "pad_level_db": pad_level_db},
        "pad": src_info,
        "regions": [{"name": "pad fades in under A", "output": [t_s, t_x]}, {"name": "A fades out", "output": [t_x - fade, t_x]},
                    {"name": "pad alone" if gap_bars else "junction", "output": [t_x, t_sw]}, {"name": "pad releases under B", "output": [t_sw, t_e]}],
        "basis": "[P] sustained note / pad or ambient bed across a tempo change (open-format practice); chord = common tones [H]",
    }
    return _finish(ctx, "note_bridge", deck_a=deck_a, deck_b=deck_b, overlay=pad_sig, t_s=t_s, t_sw=t_sw, t_e=t_e, off_a=off_a,
                   sb_at_switch=ctx.sb(), stretch=_tempo_block(ctx, False), creative=creative, exit_effect=None, a_map=a_map,
                   b_map=b_map, vocals_a_heard=va_heard, vocals_b_heard=vb_heard, sg_info=sg_info,
                   extra_decks={"overlay_parts": {"pad": pad_sig}}, provenance=provenance, return_decks=return_decks)


def echo_freeze(audio_a, audio_b, *, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b,
                vocals_a=None, vocals_b=None, hold_bars: int = 1, tail_bars: int = 1, capture_beats: int = 1,
                sample_rate: int = dj.DEFAULT_SAMPLE_RATE, config: Optional[DJCreativeConfig] = None,
                provenance: Optional[dict] = None, return_decks: bool = False) -> DJRenderResult:
    """Freeze dell'ultimo battito di A: dal downbeat ``hold_bars`` prima dell'uscita il dry di A è
    tagliato e resta il suo spettro congelato (freeze spettrale) più l'eco a un battito del
    frammento (feedback 0,6), con passa-alto crescente (150 -> 1500 Hz); B entra pulito sul suo
    downbeat, il freeze scende di ``tail_under_b_db`` e sfuma in ``tail_bars`` battute di B."""
    cfg = config or DJCreativeConfig()
    sr = int(sample_rate)
    if hold_bars not in (1, 2) or tail_bars not in (1, 2) or capture_beats not in (1, 2):
        raise DJRenderError("echo_freeze: hold_bars, tail_bars, capture_beats in (1, 2)")
    ctx = _prepare(audio_a, audio_b, vocals_a, vocals_b, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b, sr, cfg)
    ia = ctx.ia
    if ia - hold_bars < 0:
        raise DJRenderError("echo_freeze: not enough A bars")
    pre, post = int(round(cfg.pre_s * sr)), int(round(cfg.post_s * sr))
    F = ctx.sa(ia - hold_bars)
    X = ctx.sa(ia)
    off_a = F - pre
    t_s, t_sw = pre, X - off_a
    t_e = t_sw + int(round(tail_bars * ctx.tb["bar_s"] * sr))
    n = t_e + post
    beat = int(round(ctx.ta["beat_s"] * sr))
    cap = ctx.a[F - capture_beats * beat : F]
    frozen = np.zeros((n, 2))
    frozen[t_s:] = spectral_freeze(cap, sr, n - t_s)
    echo = np.zeros((n, 2))
    frag = cap[-beat:].copy()
    edge = int(0.003 * sr)
    frag[:edge] *= np.linspace(0, 1, edge)[:, None]
    frag[-edge:] *= np.linspace(1, 0, edge)[:, None]
    k = 0
    while t_s + k * beat < t_e:
        p = t_s + k * beat
        seg = frag[: max(0, min(beat, n - p))] * (0.6 ** (k + 1))
        echo[p : p + len(seg)] += seg
        k += 1
    wet = 0.7 * frozen + echo
    fc = np.where(np.arange(n) < t_s, 150.0, 150.0 * 10.0 ** np.clip((np.arange(n) - t_s) / max(t_sw - t_s, 1), 0, 1))
    wet = fx.time_varying_filter(wet, sr, fc, "highpass", 2, 256)
    env = _rise(n, t_s, t_s + int(0.02 * sr), "sin") * (1.0 - (1.0 - _db(cfg.tail_under_b_db)) * _rise(n, t_sw - int(0.02 * sr), t_sw))
    env *= 1.0 - _rise(n, t_sw, t_e, "cos")
    wet *= env[:, None]
    a_ref_rms = _rms(ctx.a[max(0, F - 2 * beat * ctx.ta["beats_per_bar"]) : F])
    wet *= a_ref_rms / max(_rms(wet[t_s:t_sw]), 1e-9) * _db(-3.0)
    a_raw = _slice(ctx.a, off_a, n)
    deck_a = a_raw * _a_cut(n, t_s, cfg, sr)[:, None]
    deck_b = _b_entry(ctx.b, ctx.sb(), t_sw, n, cfg, sr)
    vb_src = _b_entry(ctx.vb, ctx.sb(), t_sw, n, cfg, sr)
    b_ref = ctx.b[ctx.sb() : ctx.sb() + int(round(cfg.section_gain.b_bars * ctx.tb["bar_s"] * sr))]
    sg_curve, sg_info = _section_gain(ctx, F, b_ref, t_sw, ctx.tb["bar_s"], n, t_e, "B source (no stretch); A before the freeze (dry)")
    deck_b = deck_b * sg_curve[:, None]
    va_heard = _slice(ctx.va, off_a, n) * _a_cut(n, t_s, cfg, sr)[:, None]
    vb_heard = vb_src * sg_curve[:, None]
    # A's vocal inside the freeze (captured fragment's vocal, frozen): approximated by the echo of the vocal stem fragment.
    vfrag = ctx.va[F - beat : F]
    v_echo = np.zeros((n, 2))
    for kk in range(k):
        p = t_s + kk * beat
        seg = vfrag[: max(0, min(beat, n - p))] * (0.6 ** (kk + 1))
        v_echo[p : p + len(seg)] += seg
    va_heard = va_heard + v_echo * env[:, None]
    a_map = [{"out_start": 0, "out_end": t_s, "src_start": off_a, "rate": 1.0, "vocals": True, "what": "A dry until the freeze"}]
    b_map = [{"out_start": t_sw, "out_end": n, "src_start": ctx.sb(), "rate": 1.0, "vocals": True, "what": "B at its own tempo"}]
    creative = {
        "technique": "echo_freeze",
        "params": {"hold_bars": hold_bars, "tail_bars": tail_bars, "capture_beats": capture_beats},
        "freeze": {"captured_a_source_samples": [F - capture_beats * beat, F], "spectral_freeze": "mean STFT magnitude (4096) with random phases",
                   "echo": {"delay_beats": 1, "feedback": 0.6}, "hpf_hz": [150, 1500], "level_db_rel_a": -3.0,
                   "under_b_db": cfg.tail_under_b_db},
        "regions": [{"name": "freeze holds", "output": [t_s, t_sw]}, {"name": "freeze tail under B", "output": [t_sw, t_e]}],
        "basis": "[P] echo/freeze out (Pioneer DJM Beat FX, Serato/rekordbox FX); [U] echo out 9/10 and 7/10",
    }
    return _finish(ctx, "echo_freeze", deck_a=deck_a, deck_b=deck_b, overlay=wet, t_s=t_s, t_sw=t_sw, t_e=t_e, off_a=off_a,
                   sb_at_switch=ctx.sb(), stretch=_tempo_block(ctx, False), creative=creative, exit_effect="echo", a_map=a_map,
                   b_map=b_map, vocals_a_heard=va_heard, vocals_b_heard=vb_heard, sg_info=sg_info,
                   extra_decks={"overlay_parts": {"freeze": wet}}, provenance=provenance, return_decks=return_decks)


def reverb_wash(audio_a, audio_b, *, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b,
                vocals_a=None, vocals_b=None, wash_bars: int = 1, tail_bars: int = 2, rt60_s: float = 3.5,
                wet_db: float = -3.0, sample_rate: int = dj.DEFAULT_SAMPLE_RATE, config: Optional[DJCreativeConfig] = None,
                provenance: Optional[dict] = None, return_decks: bool = False) -> DJRenderResult:
    """A mandato in un riverbero lungo (IR sintetica, RT60 ``rt60_s``) con mandata crescente nelle
    ultime ``wash_bars`` battute; dry di A tagliato sul downbeat di uscita, B pulito sul suo
    downbeat, coda (passa-alto 250 Hz) a ``tail_under_b_db`` sotto B e sfumata in ``tail_bars``."""
    cfg = config or DJCreativeConfig()
    sr = int(sample_rate)
    if wash_bars not in (1, 2) or tail_bars not in (1, 2, 4):
        raise DJRenderError("reverb_wash: wash_bars 1|2, tail_bars 1|2|4")
    ctx = _prepare(audio_a, audio_b, vocals_a, vocals_b, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b, sr, cfg)
    ia = ctx.ia
    if ia - wash_bars < 0:
        raise DJRenderError("reverb_wash: not enough A bars")
    from scipy.signal import butter, fftconvolve, sosfiltfilt

    pre, post = int(round(cfg.pre_s * sr)), int(round(cfg.post_s * sr))
    W = ctx.sa(ia - wash_bars)
    X = ctx.sa(ia)
    off_a = W - pre
    t_s, t_sw = pre, X - off_a
    t_e = t_sw + int(round(tail_bars * ctx.tb["bar_s"] * sr))
    n = t_e + post
    a_raw = _slice(ctx.a, off_a, n)
    cut = _a_cut(n, t_sw, cfg, sr)
    deck_a = a_raw * cut[:, None]
    send = _rise(n, t_s, t_sw, "lin") ** 1.5 * cut
    ir = reverb_ir(sr, rt60_s)
    src = a_raw * send[:, None]
    wet = np.stack([fftconvolve(src[:, c], ir[:, c])[:n] for c in range(2)], 1)
    wet = sosfiltfilt(butter(2, 250.0, btype="high", fs=sr, output="sos"), wet, axis=0)
    beat = int(round(ctx.ta["beat_s"] * sr))
    dry_ref = _rms(a_raw[t_sw - beat : t_sw])
    wet_scale = dry_ref / max(_rms(wet[t_sw - beat : t_sw]), 1e-9) * _db(wet_db)
    wet *= wet_scale
    env = 1.0 - (1.0 - _db(cfg.tail_under_b_db)) * _rise(n, t_sw - int(0.02 * sr), t_sw)
    env *= 1.0 - _rise(n, t_sw, t_e, "cos")
    wet *= env[:, None]
    deck_b = _b_entry(ctx.b, ctx.sb(), t_sw, n, cfg, sr)
    vb_src = _b_entry(ctx.vb, ctx.sb(), t_sw, n, cfg, sr)
    b_ref = ctx.b[ctx.sb() : ctx.sb() + int(round(cfg.section_gain.b_bars * ctx.tb["bar_s"] * sr))]
    sg_curve, sg_info = _section_gain(ctx, W, b_ref, t_sw, ctx.tb["bar_s"], n, t_e, "B source (no stretch); A before the wash (dry)")
    deck_b = deck_b * sg_curve[:, None]
    va_src = _slice(ctx.va, off_a, n)
    v_wet = np.stack([fftconvolve(va_src[:, c] * send, ir[:, c])[:n] for c in range(2)], 1)
    v_wet = sosfiltfilt(butter(2, 250.0, btype="high", fs=sr, output="sos"), v_wet, axis=0) * wet_scale * env[:, None]
    va_heard = va_src * cut[:, None]
    vb_heard = vb_src * sg_curve[:, None]
    a_map = [{"out_start": 0, "out_end": t_sw, "src_start": off_a, "rate": 1.0, "vocals": True, "what": "A dry (send to reverb rising)"}]
    b_map = [{"out_start": t_sw, "out_end": n, "src_start": ctx.sb(), "rate": 1.0, "vocals": True, "what": "B at its own tempo"}]
    creative = {
        "technique": "reverb_wash",
        "params": {"wash_bars": wash_bars, "tail_bars": tail_bars, "rt60_s": rt60_s, "wet_db": wet_db},
        "reverb": {"ir": "synthetic stereo exponential noise, highs decay 2x faster, 20 ms pre-delay, unit energy",
                   "send": "rises (u^1.5) over the wash bars, stops with A's dry at the exit downbeat", "wet_highpass_hz": 250,
                   "under_b_db": cfg.tail_under_b_db},
        "regions": [{"name": "wash rises", "output": [t_s, t_sw]}, {"name": "reverb tail under B", "output": [t_sw, t_e]}],
        "basis": "[P] reverb out / wash (DJ mixer FX: Pioneer DJM, Serato FX, rekordbox); levels [H]",
    }
    return _finish(ctx, "reverb_wash", deck_a=deck_a, deck_b=deck_b, overlay=wet, t_s=t_s, t_sw=t_sw, t_e=t_e, off_a=off_a,
                   sb_at_switch=ctx.sb(), stretch=_tempo_block(ctx, False), creative=creative, exit_effect="reverb", a_map=a_map,
                   b_map=b_map, vocals_a_heard=va_heard + v_wet, vocals_b_heard=vb_heard,
                   sg_info=sg_info, extra_decks={"overlay_parts": {"reverb": wet}}, provenance=provenance, return_decks=return_decks)


def _spin_render(ctx: _Ctx, technique: str, length_beats: int, speed: np.ndarray, amp: np.ndarray, start_src: int,
                 lowpass_for: Optional[float], params: dict, basis: str, provenance, return_decks, cfg, sr) -> DJRenderResult:
    pre, post = int(round(cfg.pre_s * sr)), int(round(cfg.post_s * sr))
    X = ctx.sa()
    S = start_src
    off_a = S - pre
    t_s, t_sw = pre, X - off_a
    t_e = t_sw
    n = t_e + post
    L = t_sw - t_s
    fb = float(params.get("fader_out_beats", 0.0))
    if fb > 0:  # DJ pulls the fader while the platter spins/stops [P] (zZounds; LSA 7): gradual exit
        amp = amp[:L] * _fall_eqpow(L, L - min(L, int(round(fb * ctx.ta["beat_s"] * sr))), L)
    spun, pos = read_varispeed(ctx.a, float(S), speed[:L], sr, lowpass_for=lowpass_for)
    spun *= amp[:L, None]
    vspun, _ = read_varispeed(ctx.va, float(S), speed[:L], sr, lowpass_for=lowpass_for)
    vspun *= amp[:L, None]
    a_raw = _slice(ctx.a, off_a, n)
    deck_a = a_raw * (np.arange(n) < t_s)[:, None]
    deck_a[t_s:t_sw] = spun
    d = int(round(cfg.crossfade_ms / 1000 * sr))
    tail = np.cos(0.5 * np.pi * np.clip((np.arange(n) - (t_sw - d)) / max(d, 1), 0, 1))
    tail[np.arange(n) >= t_sw] = 0.0
    deck_a *= tail[:, None]
    va_heard = _slice(ctx.va, off_a, n) * (np.arange(n) < t_s)[:, None]
    va_heard[t_s:t_sw] = vspun
    va_heard *= tail[:, None]
    deck_b = _b_entry(ctx.b, ctx.sb(), t_sw, n, cfg, sr)
    vb_src = _b_entry(ctx.vb, ctx.sb(), t_sw, n, cfg, sr)
    b_ref = ctx.b[ctx.sb() : ctx.sb() + int(round(cfg.section_gain.b_bars * ctx.tb["bar_s"] * sr))]
    sg_curve, sg_info = _section_gain(ctx, S, b_ref, t_sw, ctx.tb["bar_s"], n, t_sw, "B source (no stretch); A before the spin (dry)")
    deck_b = deck_b * sg_curve[:, None]
    a_map = [{"out_start": 0, "out_end": t_s, "src_start": off_a, "rate": 1.0, "vocals": True, "what": "A before the effect"},
             {"out_start": t_s, "out_end": t_sw, "src_start": S, "rate": None, "vocals": True,
              "what": f"A read at variable speed ({technique}); source positions {float(pos.min()):.0f}..{float(pos.max()):.0f}"}]
    b_map = [{"out_start": t_sw, "out_end": n, "src_start": ctx.sb(), "rate": 1.0, "vocals": True, "what": "B at its own tempo"}]
    creative = {
        "technique": technique, "params": {**params, "length_beats": length_beats},
        "speed_profile": {"min": float(speed[:L].min()), "max": float(speed[:L].max()),
                          "source_positions_samples": [float(pos.min()), float(pos.max())],
                          "anti_alias_lowpass_for_speed": lowpass_for},
        "regions": [{"name": technique, "output": [t_s, t_sw]}, {"name": "B on its downbeat", "output": [t_sw, t_sw]}],
        "basis": basis,
    }
    return _finish(ctx, technique, deck_a=deck_a, deck_b=deck_b, overlay=None, t_s=t_s, t_sw=t_sw, t_e=t_e, off_a=off_a,
                   sb_at_switch=ctx.sb(), stretch=_tempo_block(ctx, False), creative=creative, exit_effect=None, a_map=a_map,
                   b_map=b_map, vocals_a_heard=va_heard, vocals_b_heard=vb_src * sg_curve[:, None], sg_info=sg_info,
                   extra_decks={}, provenance=provenance, return_decks=return_decks)


def backspin(audio_a, audio_b, *, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b,
             vocals_a=None, vocals_b=None, spin_beats: int = 2, max_reverse_speed: float = 3.0, fader_out_beats: float = 1.0,
             sample_rate: int = dj.DEFAULT_SAMPLE_RATE, config: Optional[DJCreativeConfig] = None,
             provenance: Optional[dict] = None, return_decks: bool = False) -> DJRenderResult:
    """Backspin (spinback) simulato: negli ultimi ``spin_beats`` battiti prima del downbeat di uscita
    il disco di A viene fermato (40 ms), lanciato all'indietro fino a ``max_reverse_speed`` (80 ms)
    e rallenta esponenzialmente fino a zero; ampiezza ~ sqrt(|velocità|). B entra pulito sul suo
    downbeat (= ``cue_out_a``). Lettura a velocità variabile con anti-aliasing."""
    cfg = config or DJCreativeConfig()
    sr = int(sample_rate)
    if spin_beats not in (1, 2, 4) or not 1.0 <= max_reverse_speed <= 4.0:
        raise DJRenderError("backspin: spin_beats 1|2|4, max_reverse_speed 1..4")
    ctx = _prepare(audio_a, audio_b, vocals_a, vocals_b, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b, sr, cfg)
    L = int(round(spin_beats * ctx.ta["beat_s"] * sr))
    S = ctx.sa() - L
    t = np.arange(L) / sr
    T = L / sr
    t1, t2 = 0.04, 0.12
    v = np.where(t < t1, 1.0 - t / t1, np.where(t < t2, -max_reverse_speed * (t - t1) / (t2 - t1),
                                                   -max_reverse_speed * np.exp(-(t - t2) / max((T - t2) / 4.0, 1e-3))))
    amp = np.sqrt(np.minimum(1.0, np.abs(v)))
    amp *= 1.0 - _rise(L, int(0.85 * L), L, "cos")
    return _spin_render(ctx, "backspin", spin_beats, v, amp, S, max_reverse_speed,
                        {"spin_beats": spin_beats, "max_reverse_speed": max_reverse_speed, "catch_ms": 40, "throw_ms": 80,
                         "fader_out_beats": fader_out_beats},
                        "[P] backspin / spinback (hip-hop and open-format practice; Katz 2012 on Grandmaster Flash's backspin); profile [H]",
                        provenance, return_decks, cfg, sr)


def tape_stop(audio_a, audio_b, *, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b,
              vocals_a=None, vocals_b=None, stop_beats: int = 2, curve_power: float = 1.5, fader_out_beats: float = 1.0,
              sample_rate: int = dj.DEFAULT_SAMPLE_RATE, config: Optional[DJCreativeConfig] = None,
              provenance: Optional[dict] = None, return_decks: bool = False) -> DJRenderResult:
    """Tape stop / brake: negli ultimi ``stop_beats`` battiti A rallenta fino a fermarsi
    (velocità (1-u)^``curve_power``, intonazione che scende come un vinile frenato); B entra pulito
    sul suo downbeat."""
    cfg = config or DJCreativeConfig()
    sr = int(sample_rate)
    if stop_beats not in (1, 2, 4) or not 0.5 <= curve_power <= 3.0:
        raise DJRenderError("tape_stop: stop_beats 1|2|4, curve_power 0.5..3")
    ctx = _prepare(audio_a, audio_b, vocals_a, vocals_b, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b, sr, cfg)
    L = int(round(stop_beats * ctx.ta["beat_s"] * sr))
    S = ctx.sa() - L
    u = np.arange(L) / max(L - 1, 1)
    v = (1.0 - u) ** curve_power
    amp = np.minimum(1.0, 4.0 * v) ** 0.5
    return _spin_render(ctx, "tape_stop", stop_beats, v, amp, S, None,
                        {"stop_beats": stop_beats, "curve_power": curve_power, "fader_out_beats": fader_out_beats},
                        "[P] brake / tape stop (Serato, Traktor, rekordbox vinyl brake); curve [H]", provenance, return_decks, cfg, sr)


# ---------------------------------------------------------------- lyric link

STOPWORDS = frozenset("""
a an the and or but if of to in on at by for with from up out is am are was were be been it its i im i'm me my you your
yo ya he she him her his we us our they them their this that these those what who how so no not yeah uh huh oh ah eh
like just got get gon gonna wanna da de la le les el los las un une una y o e et du des der die das und ein eine ich du
er sie es wir ihr mit von zu je tu il nous vous ils en que qui pas mi me te se lo con por para che non il io
""".split())


def normalize_word(text: str) -> str:
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z']", "", s).strip("'")


WEAK_RHYME_KEYS = frozenset({"er", "es", "ed", "en", "el", "e", "y", "ing", "a", "o", "i", "u"})


def rhyme_key(word: str, syllables: int = 1) -> str:
    """Chiave di rima ortografica approssimata [H]: dagli ultimi ``syllables`` gruppi vocalici
    alla fine, dopo semplificazioni (ph->f, ck->k, e muta finale, doppie). Non è una
    trascrizione fonetica."""
    w = normalize_word(word).replace("'", "")
    if not w:
        return ""
    w = w.replace("ph", "f").replace("ck", "k").replace("qu", "kw")
    w = re.sub(r"(.)\1+", r"\1", w)
    if len(w) > 3 and w.endswith("e") and w[-2] not in "aeiouy":
        w = w[:-1]
    m = re.search(r"(?:[aeiouy]+[^aeiouy]*){%d}$" % max(1, syllables), w)
    if m:
        return m.group(0)
    m = re.search(r"[aeiouy]+[^aeiouy]*$", w)
    return m.group(0) if m else w


def soundex(word: str) -> str:
    """Soundex americano classico (Russell & Odell): lettera iniziale + 3 cifre."""
    w = normalize_word(word).replace("'", "").upper()
    if not w:
        return ""
    codes = {**dict.fromkeys("BFPV", "1"), **dict.fromkeys("CGJKQSXZ", "2"), **dict.fromkeys("DT", "3"), "L": "4",
             **dict.fromkeys("MN", "5"), "R": "6"}
    out, last = w[0], codes.get(w[0], "")
    for ch in w[1:]:
        c = codes.get(ch, "")
        if c and c != last:
            out += c
        if ch not in "HW":
            last = c
    return (out + "000")[:4]


def _lemma(word: str, lang: Optional[str]) -> Optional[str]:
    if lang not in (None, "en"):
        return None
    try:
        import spacy  # en_core_web_sm in the project venv

        global _NLP
        if _NLP is None:
            _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        doc = _NLP(word)
        return doc[0].lemma_.lower() if len(doc) else None
    except Exception:
        return None


_NLP = None


def word_match(wa: str, wb: str, lang_a: Optional[str] = None, lang_b: Optional[str] = None) -> Optional[dict]:
    """Legame tra due parole: ``exact`` (1,0), ``lemma`` (0,85, spaCy inglese), ``rhyme`` (0,7),
    ``soundex`` (0,5). Parole funzionali (STOPWORDS) e token < 3 lettere esclusi salvo match
    esatto lungo. Pesi a priori [H], non stimati."""
    a, b = normalize_word(wa), normalize_word(wb)
    if len(a) < 2 or len(b) < 2:
        return None
    stop = a in STOPWORDS or b in STOPWORDS
    if a == b:
        if stop or len(a) < 3:
            return None
        return {"type": "exact", "score": 1.0, "a": a, "b": b}
    if stop:
        return None
    la, lb = _lemma(a, lang_a), _lemma(b, lang_b)
    if la and lb and la == lb and len(la) >= 3:
        return {"type": "lemma", "score": 0.85, "a": a, "b": b, "lemma": la}
    ra, rb = rhyme_key(a), rhyme_key(b)
    if ra in WEAK_RHYME_KEYS:  # unstressed endings (-er, -es, -ing...): require a two-syllable rhyme
        ra, rb = rhyme_key(a, 2), rhyme_key(b, 2)
        weak = True
    else:
        weak = False
    if len(ra) >= 2 and ra == rb and min(len(a), len(b)) >= 3 and not (weak and len(ra) < 4):
        return {"type": "rhyme", "score": 0.7, "a": a, "b": b, "rhyme_key": ra, "syllables": 2 if weak else 1}
    if min(len(a), len(b)) >= 4 and soundex(a) == soundex(b):
        return {"type": "soundex", "score": 0.5, "a": a, "b": b, "soundex": soundex(a)}
    return None


def find_lyric_links(words_a: Sequence[Mapping[str, Any]], words_b: Sequence[Mapping[str, Any]], a_cues_s: Sequence[float],
                     b_cues_s: Sequence[float], *, sr: int, bar_a_s: float, bar_b_s: float, lang_a: Optional[str] = None,
                     lang_b: Optional[str] = None, a_window_bars: float = 2.0, b_window_bars: float = 2.0,
                     min_gap_s: float = 0.15, min_confidence: float = 0.3, top_k: int = 10) -> list[dict]:
    """Coppie (parola di A che chiude la frase prima di un cue di uscita, parola di B che apre la
    frase dopo un cue d'ingresso) legate da :func:`word_match`.

    A: ultime 3 parole che finiscono almeno ``min_gap_s`` prima del cue ed entro
    ``a_window_bars`` battute; B: prime 4 parole da un battito prima del cue (anacrusi) a
    ``b_window_bars`` battute dopo. Punteggio = match x sqrt(confidenza ASR minima) x bonus se la
    parola di A è l'ultima della frase. Semantica (embedding) non disponibile: spaCy
    ``en_core_web_sm`` non ha vettori di parola; lo si dichiara invece di simularlo."""
    out = []
    for X in a_cues_s:
        cand_a = [w for w in words_a if X - a_window_bars * bar_a_s <= w["start_sample"] / sr and w["end_sample"] / sr <= X - min_gap_s]
        cand_a = sorted(cand_a, key=lambda w: w["start_sample"])[-3:]
        for Y in b_cues_s:
            cand_b = [w for w in words_b if Y - 0.25 * bar_b_s <= w["start_sample"] / sr <= Y + b_window_bars * bar_b_s]
            cand_b = sorted(cand_b, key=lambda w: w["start_sample"])[:4]
            for ka, wa in enumerate(cand_a):
                for kb, wb in enumerate(cand_b):
                    m = word_match(wa.get("text", ""), wb.get("text", ""), lang_a, lang_b)
                    if not m:
                        continue
                    conf = min(float(wa.get("confidence") or 0.0), float(wb.get("confidence") or 0.0))
                    if conf < min_confidence:
                        continue
                    last = ka == len(cand_a) - 1
                    first = kb == 0
                    score = m["score"] * math.sqrt(conf) * (1.1 if last else 1.0) * (1.05 if first else 1.0)
                    out.append({"a_cue_s": float(X), "b_cue_s": float(Y), "a_word": dict(wa), "b_word": dict(wb), "match": m,
                                "asr_confidence_min": conf, "a_word_is_last": last, "b_word_is_first": first, "score": score})
    out.sort(key=lambda r: -r["score"])
    return out[:top_k]


def lyric_link(audio_a, audio_b, *, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b,
               vocals_a=None, vocals_b=None, a_word: Mapping[str, Any], b_word: Mapping[str, Any], match: Optional[dict] = None,
               feedback: float = 0.6, filter_bars: int = 1, entry_delay_bars: Optional[int] = None,
               sample_rate: int = dj.DEFAULT_SAMPLE_RATE,
               config: Optional[DJCreativeConfig] = None, provenance: Optional[dict] = None, return_decks: bool = False) -> DJRenderResult:
    """Lyric link: l'eco della parola di A (isolata dallo stem vocale) porta alla parola legata di B.

    La parola di A (``a_word``, campioni sorgente) è catturata dallo stem; dopo di essa la voce di
    A è tolta (mix − stem) e lo strumentale di A esce con un passa-alto di ``filter_bars`` battute
    fino al downbeat di uscita; l'eco della parola si ripete a un periodo D vicino a 1/2, 1 o 2
    battiti di A scelto in modo che la K-esima ripetizione cada esattamente sull'attacco della
    parola di B (B entra pulito sul suo downbeat ``cue_in_b``): la parola di B "prende il posto"
    dell'ultima eco. Servono almeno K = 2 (una ripetizione udibile fra le due parole): se le parole
    sono troppo vicine B entra 1 o 2 battute dopo l'uscita di A (``entry_delay_bars``, automatico se
    None), come l'``entry_bars`` di ``dj_effects.echo_out``. Le ripetizioni sono zittite dove la voce
    di B è attiva (una voce alla volta).
    """
    cfg = config or DJCreativeConfig()
    sr = int(sample_rate)
    if not 0.0 < feedback < 1.0 or filter_bars not in (1, 2):
        raise DJRenderError("lyric_link: feedback in (0, 1), filter_bars 1|2")
    ctx = _prepare(audio_a, audio_b, vocals_a, vocals_b, beats_a, downbeats_a, beats_b, downbeats_b, cue_out_a, cue_in_b, sr, cfg,
                   need_vocals_a=True)
    from scipy.signal import butter, sosfiltfilt

    pre, post = int(round(cfg.pre_s * sr)), int(round(cfg.post_s * sr))
    X = ctx.sa()
    ws, we = int(a_word["start_sample"]), int(a_word["end_sample"])
    if not (X - (filter_bars + 2) * ctx.ta["bar_s"] * sr <= ws < we <= X):
        raise DJRenderError("lyric_link: A's word must end before cue_out_a, within the last bars")
    F = min(ctx.sa(ctx.ia - filter_bars), ws)
    t_s_src = F
    off_a = t_s_src - pre
    t_s = pre
    t_x = X - off_a  # A's exit downbeat in the clip
    bw_s = int(b_word["start_sample"])
    bw_e = int(b_word["end_sample"])
    bw_rel = bw_s - ctx.sb()  # B word onset relative to B's entry downbeat (B at its own tempo)
    if bw_rel <= -int(0.25 * ctx.tb["bar_s"] * sr) or bw_rel > int(3 * ctx.tb["bar_s"] * sr):
        raise DJRenderError("lyric_link: B's word must be near the entry downbeat")
    a_on = ws - off_a
    beat = ctx.ta["beat_s"] * sr
    # Echo period D (+-6% of 1/2, 1 or 2 beats of A) such that the K-th repeat (K >= 2) lands on
    # B's word; if the words are too close, B enters 1 or 2 bars after A's exit (as echo_out
    # entry_bars) so that at least one echo repeat sits between the two words.
    choice = None
    delays = (0, 1, 2) if entry_delay_bars is None else (int(entry_delay_bars),)
    for delay in delays:
        entry = t_x + int(round(delay * ctx.ta["bar_s"] * sr))
        gap = entry + bw_rel - a_on
        best = None
        for mult in (1.0, 0.5, 2.0):
            for K in range(2, 17):
                D = gap / K
                dev = D / (mult * beat) - 1.0
                if abs(dev) <= 0.06 and D >= 0.12 * sr:
                    cand = (abs(dev), K, D, mult)
                    if best is None or cand < best:
                        best = cand
        if best is not None:
            choice = (delay, entry, gap, *best)
            break
    if choice is None:
        delay = 1 if entry_delay_bars is None else int(entry_delay_bars)
        entry = t_x + int(round(delay * ctx.ta["bar_s"] * sr))
        gap = entry + bw_rel - a_on
        mult, D = 1.0, beat
        K = max(2, int(round(gap / D)))
        phase_err_ms = 1000.0 * (gap - K * D) / sr
        dev = 0.0
    else:
        delay, entry, gap, dev, K, D, mult = choice
        phase_err_ms = 0.0
    t_sw = entry
    b_on = t_sw + bw_rel
    t_e = max(t_sw, t_sw + (bw_e - ctx.sb()))
    n = t_e + post
    Dn = int(round(D))
    # A: vocal muted after the word, instrumental high-passed over the last filter_bars, cut at X.
    a_mix = _slice(ctx.a, off_a, n)
    a_voc = _slice(ctx.va, off_a, n)
    keep_to = we - off_a + int(0.03 * sr)
    mute = _rise(n, keep_to, keep_to + int(round(cfg.vocal_mute_ms / 1000 * sr)), "sin")
    a_raw = a_mix - a_voc * mute[:, None]
    fs0 = ctx.sa(ctx.ia - filter_bars) - off_a
    fc = np.where(np.arange(n) < fs0, 30.0, 30.0 * (1200.0 / 30.0) ** np.clip((np.arange(n) - fs0) / max(t_x - fs0, 1), 0, 1))
    filt = fx.time_varying_filter(a_raw, sr, fc, "highpass", 4, 128)
    eng = _rise(n, fs0, fs0 + int(0.03 * sr))[:, None]
    cut = _a_cut(n, t_x, cfg, sr)
    deck_a = (a_raw * (1 - eng) + filt * eng) * cut[:, None]
    # Echo of the isolated word.
    edge = int(0.01 * sr)
    frag = _slice(ctx.va, ws - edge, (we - ws) + 2 * edge + int(0.03 * sr))
    win = np.ones(len(frag))
    win[:edge] = np.sin(0.5 * np.pi * np.arange(edge) / edge) ** 2
    win[-edge:] = np.cos(0.5 * np.pi * np.arange(edge) / edge) ** 2
    frag *= win[:, None]
    echo = np.zeros((n, 2))
    reps = []
    cur = frag.copy()
    pad = min(len(cur) - 1, int(0.05 * sr))
    for k in range(1, K):
        fc_k = 150.0 * (1500.0 / 150.0) ** ((k - 1) / max(K - 2, 1))
        cur = sosfiltfilt(butter(2, fc_k, btype="high", fs=sr, output="sos"), cur, axis=0, padlen=pad) * feedback
        p = a_on - edge + k * Dn
        if p >= n:
            break
        seg = cur[: max(0, min(len(cur), n - p))]
        echo[p : p + len(seg)] += seg
        reps.append({"repeat": k, "output_sample": p + edge, "gain_db": 20 * k * math.log10(feedback), "hpf_hz": fc_k})
    # Duck the echo where B sings (one voice at a time) and from B's word on.
    vb_entry = _b_entry(ctx.vb, ctx.sb(), t_sw, n, cfg, sr)
    act, _ = _activity(vb_entry, sr, ctx.vb)
    hop = int(round(0.02 * sr))
    mask = np.repeat(act.astype(float), hop)[:n]
    mask = np.r_[mask, np.zeros(n - len(mask))]
    mask = np.maximum(mask, (np.arange(n) >= b_on - int(0.03 * sr)).astype(float))
    k_win = np.hanning(2 * int(0.03 * sr) + 1)
    k_win /= k_win.sum()
    duck = 1.0 - np.clip(np.convolve(mask, k_win, mode="same"), 0, 1)
    echo *= duck[:, None]
    deck_b = _b_entry(ctx.b, ctx.sb(), t_sw, n, cfg, sr)
    b_ref = ctx.b[ctx.sb() : ctx.sb() + int(round(cfg.section_gain.b_bars * ctx.tb["bar_s"] * sr))]
    sg_curve, sg_info = _section_gain(ctx, t_s_src, b_ref, t_sw, ctx.tb["bar_s"], n, t_e, "B source (no stretch) from its entry downbeat; A before the link (dry)")
    deck_b = deck_b * sg_curve[:, None]
    va_heard = a_voc * (1.0 - mute)[:, None] * cut[:, None]
    vb_heard = vb_entry * sg_curve[:, None]
    a_map = [{"out_start": 0, "out_end": keep_to, "src_start": off_a, "rate": 1.0, "vocals": True, "what": "A (vocal kept up to the linked word)"},
             {"out_start": keep_to, "out_end": t_x, "src_start": keep_to + off_a, "rate": 1.0, "vocals": False, "what": "A instrumental, high-passed"}]
    b_map = [{"out_start": t_sw, "out_end": n, "src_start": ctx.sb(), "rate": 1.0, "vocals": True, "what": "B at its own tempo"}]
    creative = {
        "technique": "lyric_link",
        "a_vocal_mute_guard": mute_guard(ctx, keep_to + off_a),
        "params": {"feedback": feedback, "filter_bars": filter_bars, "delay_beats": D / beat, "delay_multiple_of_beat": mult,
                   "entry_delay_bars": delay},
        "link": {"a_word": {k: a_word.get(k) for k in ("text", "start_sample", "end_sample", "confidence")},
                 "b_word": {k: b_word.get(k) for k in ("text", "start_sample", "end_sample", "confidence")}, "match": match,
                 "a_word_output_sample": a_on, "b_word_output_sample": b_on, "echo_period_samples": Dn, "repeats_to_b_word": K,
                 "delay_deviation_from_musical_value": dev, "phase_error_ms": phase_err_ms, "echo_repeats": reps,
                 "semantic_match": "not available (en_core_web_sm has no word vectors); lexical, lemma, orthographic rhyme and Soundex only"},
        "regions": [{"name": "A's linked word + echo", "output": [a_on, b_on]}, {"name": "A filter out", "output": [fs0, t_x]}, {"name": "echo only (B enters after entry_delay_bars)", "output": [t_x, t_sw]},
                    {"name": "B's linked word", "output": [b_on, t_e]}],
        "basis": "[P] word-play / lyric link between tracks (hip-hop DJ practice: 'word play' transitions); echo [U] 9/10; alignment of the K-th repeat on B's word [H]",
    }
    return _finish(ctx, "lyric_link", deck_a=deck_a, deck_b=deck_b, overlay=echo, t_s=t_s, t_sw=t_sw, t_e=t_e, off_a=off_a,
                   sb_at_switch=ctx.sb(), stretch=_tempo_block(ctx, False), creative=creative, exit_effect="echo", a_map=a_map,
                   b_map=b_map, vocals_a_heard=va_heard, vocals_b_heard=vb_heard, sg_info=sg_info,
                   extra_decks={"overlay_parts": {"word_echo": echo}, "vocals_a_overlay": echo}, provenance=provenance,
                   return_decks=return_decks)


# ------------------------------------------------------------------ dispatcher


def render_creative(technique: str, audio_a, audio_b, **kwargs) -> DJRenderResult:
    """Dispatcher: ``technique`` in :data:`TECHNIQUES`; kwargs come nelle singole funzioni."""
    fns = {"loop_extend": loop_extend, "acapella_over": acapella_over, "vocal_swap": vocal_swap,
           "build_into_chorus": build_into_chorus, "note_bridge": note_bridge, "echo_freeze": echo_freeze,
           "reverb_wash": reverb_wash, "backspin": backspin, "tape_stop": tape_stop, "lyric_link": lyric_link,
           "halving_build": halving_build, "break_before_entry": break_before_entry}
    if technique not in fns:
        raise DJRenderError(f"technique must be one of {TECHNIQUES}")
    return fns[technique](audio_a, audio_b, **kwargs)


def qa_inputs(result: DJRenderResult) -> dict:
    """Argomenti per ``dj_qa.evaluate_render`` da un render con ``return_decks=True``: deck, voci
    udibili nella clip (già con tutti i guadagni) e deck 'raw' per il controllo del ritmo."""
    d = result.decks or {}
    decks = {k: d[k] for k in ("a", "b", "a_raw", "b_stretched_raw") if k in d}
    # Until 21/10/2026 deck B was left out for NO_OVERLAP_QA_MIX_ONLY (halving build, break): the deck-vs-deck beat
    # comparison had nothing to pair. Since dj_qa.7 a clean entry is measured by B's phase against A's pulse at the
    # switch (``dj_qa._rhythm_entry``), which uses deck B; without it the double bass could not be measured at all
    # (palestra dei difetti v1, §4.4). Deck B is now always passed.
    decks["gains"] = d.get("gains", {})
    return {"decks": decks, "vocals_a": d.get("vocals_a_heard"), "vocals_b": d.get("vocals_b_heard")}
