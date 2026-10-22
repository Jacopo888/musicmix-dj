"""Carica di A, risoluzione di B, continuità della base, scontro tonale e affinità di testo (``musicmix.dj_tension.1``).

Criteri dichiarati dall'utente nei voti ciechi (torneo dei cue ``giudizi_cloud/b_entry_fix_v1``, galleria del set v3), usati
qui come **specifica** e mai come dati di taratura (AGENTS.md: i giudizi restano diagnostica separata):

* «A deve caricare (alzare la tensione), B deve risolvere» → :func:`a_charge`, :func:`b_resolution`;
* «la base non deve interrompersi tra A e B» → :func:`continuity`;
* «tonalità discordante» anche senza sovrapposizione (voce intonata/autotune) → :func:`harmonic_clash`;
* bonus per testi/suoni affini tra la fine di A e l'inizio di B → :func:`lyric_affinity`.

Ogni punteggio è in [0, 1], **spiegabile** (componenti, pesi, motivi) e si astiene (``None``) quando manca il dato. Pesi e
soglie sono [H] (ipotesi nostre, pre-registrate in ``runs/dj_tension_v1/PREREGISTRATION.json`` prima di guardare i voti);
le fonti (libri con pagina del PDF privato, rilette con ``tools/libri.py search`` il 22/10/2026) motivano la **direzione** di
una componente, non il suo peso:

* [T] Huron 2006, *Sweet Anticipation*, pdf p. 26 (la risposta di tensione è legata al periodo immediatamente prima
  dell'esito atteso), p. 27 (dipende da incertezza e importanza dell'esito), pp. 39–40 (valenza contrastiva: il piacere
  cresce per contrasto), p. 337 (climax = punto di tensione accresciuta);
* [A] Butler 2006, *Unlocking the Groove*, pdf p. 238 (il buildup aumenta l'intensità ispessendo la tessitura e riempiendo
  posizioni ritmiche: densità), p. 237 (forme intro-buildup-core-breakdown), pp. 189, 246–247 (variazioni prima del downbeat
  ipermetrico; la cassa che torna);
* [A] Krims 2000, pdf p. 64 (stili «effusive»: il flow denso come intensificazione); Ohriner 2019, pdf p. 91 (accento come
  intensificazione di un parametro, Roeder);
* [P] Brewster & Broughton 2003, cap. 13 (pdf p. 33: «mix straight after a chorus»; p. 34: «Mixing out of a break prevents
  the energy from dropping… you're bringing in another rising star»), cap. 15 (pdf pp. 38–39: chiavi che cozzano; «Cut them,
  don't blend them» — l'utente lo smentisce per la voce intonata: [user] prevale), cap. 10 (frasi);
* [user] note dei voti ciechi (sopra); [M] supervisione dai brani (``self_supervision.tension_at_exit``: AUC 0,69–0,75 contro
  strofa/intro su brani tenuti fuori; ``resolution_at_entry`` debole, 0,55–0,58, e contrario ai voti: qui pesa poco).

Nessuna funzione equipara tensione a volume né ritornello a risoluzione (AGENTS.md): la risoluzione di B è cercata in
segnali d'attacco (voce che attacca dopo un respiro, salto di densità, cassa/basso che entrano, voce in primo piano, contrasto
di registro con la fine di A).
"""
from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

DJ_TENSION_VERSION = "musicmix.dj_tension.1"
ROOT = Path(__file__).resolve().parents[2]
INSONG = ROOT / "runs" / "insong_v1"
DSP_FPS = 50.0
CHROMA_FPS = 10.0
DSP_COLS = ("low_db", "lowmid_db", "mid_db", "high_db", "flux", "vocal_db")  # = self_supervision.DSP_COLS

# ------------------------------------------------------------------------------------------------ parametri [H]
#: Pesi delle componenti (somma 1 per punteggio). Fissati il 22/10/2026 prima di aprire i voti (pre-registrazione).
W_CHARGE = {"model": 0.30, "energy": 0.20, "density": 0.15, "form": 0.35}
W_RESOLUTION = {"vocal_attack": 0.30, "energy_jump": 0.15, "kick_bass": 0.15, "vocal_foreground": 0.15,
                "register_contrast": 0.15, "model": 0.10}
#: Costo aggiunto al candidato del pianificatore (scala dei costi di dj_set: candidato 0,1–0,6, arco ~1,9).
W_COST = {"charge": 0.40, "resolution": 0.40, "continuity": 0.30, "clash": 0.40, "lyric": 0.15}

PARAMS: dict[str, tuple[Any, str]] = {
    "slope_db_scale": (1.5, "[H] pendenza di energia (dB/battuta) che porta la componente a 0,73 (logistica)"),
    "level_db_scale": (3.0, "[H] livello delle ultime 2 battute sopra la mediana del brano (dB) per 0,73"),
    "flux_slope_scale": (0.08, "[H] pendenza del flusso spettrale (mediana del brano = 1) per battuta"),
    "words_delta_scale": (2.0, "[H] parole per battuta in più nelle ultime 2 battute rispetto alle 4 prima"),
    "pos_min_frac": (0.15, "[H] minutaggio: uscita prima del 15 % del brano = nessun credito di posizione"),
    "pos_full_frac": (0.40, "[H] minutaggio: credito pieno dopo il 40 % del brano ([user]: 'A campionata all'inizio, manca la carica')"),
    "played_min_s": (20.0, "[H] materiale di A suonato nel set: sotto 20 s nessun credito"),
    "played_full_s": (45.0, "[H] materiale di A suonato nel set: credito pieno da 45 s"),
    "minutage_floor": (0.4, "[H] carica = media pesata x (floor + (1 - floor) x minutaggio)"),
    "rest_beats": (1.0, "[H] = R-B1 di INGRESSO_B: un attacco vocale segue una pausa >= 1 battito"),
    "vocal_jump_scale": (6.0, "[H] salto della voce (dB) nella prima battuta di B rispetto alla battuta prima, in B"),
    "energy_jump_scale": (3.0, "[H] salto di energia (dB) della prima battuta di B rispetto alle 2 prima, in B"),
    "flux_jump_scale": (0.15, "[H] salto di densità d'attacchi (flusso) della prima battuta"),
    "low_jump_scale": (4.0, "[H] salto della banda 20-150 Hz (cassa/basso/808) nella prima battuta"),
    "vocal_fg_offset_db": (8.0, "[H] voce in primo piano: livello dello stem rispetto al suo 95° percentile + 8 dB"),
    "register_scale_db": (2.0, "[H] distanza dei profili per banda (dB, livello medio tolto) fine di A / inizio di B"),
    "register_center_db": (4.0, "[H] centro della logistica del contrasto di registro"),
    "instrumental_entry_attack": (0.25, "[H] nessuna voce di B entro 2 battute: attacco vocale neutro-basso"),
    "chroma_peak_floor": (0.10, "[H] picco del croma medio (max/somma) sotto cui il materiale non è tonale (piatto = 1/12)"),
    "chroma_peak_span": (0.15, "[H] campo del picco del croma fino alla presenza tonale piena"),
    "chroma_dist_scale": (0.30, "[H] 1 - coseno dei croma di fine A / inizio B che dà scontro pieno"),
    "key_weight": (0.6, "[H] peso della distanza Camelot nello scontro (il resto al croma misurato)"),
}

#: Continuità della base pianificata per tecnica [H] (motivo nella tabella; misurata sul render da continuity_measured).
CONTINUITY = {
    "blend": (1.0, "sovrapposizione beatmatchata: la base non si ferma"),
    "loop_extend": (0.95, "loop di A sotto B"), "acapella_over": (0.95, "voce di B sulla base di A"),
    "vocal_swap": (0.95, "voce di A sulla base di B"),
    "acapella_mashup": (0.9, "mashup: B suona com'è sotto la voce di A"), "beat_swap": (0.9, "base continua di B"),
    "call_response": (0.9, "botta e risposta su base continua"), "hook_teaser": (0.9, "hook di B sopra A"),
    "vocal_loop": (0.9, "loop della voce di A sulla griglia di B"), "vocal_chop": (0.9, "chop della voce di A sulla griglia di B"),
    "lyric_link": (0.5, "eco della parola di A con strumentale filtrato fino all'uscita"),
    "reverse_swell": (0.6, "riverbero inverso della prima parola di B"),
    "pivot_freeze": (0.3, "freeze dell'accordo di A: la base si ferma"),
    "filter_out": (0.6, "A filtrata fino allo switch: la base si assottiglia ma non si ferma"),
    "filter_into_chorus": (0.6, "filtro verso il ritornello"),
    "cut": (0.55, "taglio sul downbeat: nessun buco, la base cambia di colpo"), "drop_in": (0.55, "drop-in"),
    "echo_out": (0.3, "eco di A: il beat di A si ferma, resta la coda fino a B ('la base si interrompe al taglio')"),
    "build_into_chorus": (0.6, "build sul materiale di A senza pausa fino al drop di B"),
    "halving_build": (0.55, "loop di A che si dimezza fino al downbeat di B"),
    "break_before_entry": (0.3, "break prima dell'ingresso"),
    "backspin": (0.15, "backspin: la base si ferma"),
}
CONTINUITY_GAP = 0.25        # [H] build/halving con una pausa dichiarata (gap_beats > 0): la base si interrompe
CONTINUITY_STOP = 0.0        # [H] break 'stop'
CONTINUITY_UNKNOWN = 0.5
#: Esposizione allo scontro tonale per tecnica [H]: quanto materiale di A e B si sente vicino allo switch.
EXPOSURE = {"overlap": 1.0, "echo_out": 0.7, "filter_out": 0.6, "filter_into_chorus": 0.6, "lyric_link": 0.6,
            "reverse_swell": 0.7, "pivot_freeze": 0.5, "cut": 0.45, "drop_in": 0.45, "build_into_chorus": 0.35,
            "halving_build": 0.35, "break_before_entry": 0.25, "backspin": 0.2}
OVERLAP_KINDS = ("blend", "loop_extend", "acapella_over", "vocal_swap", "acapella_mashup", "beat_swap", "call_response",
                 "hook_teaser", "vocal_loop", "vocal_chop")
KEY_CONF = {"high": 1.0, "medium": 0.7, "low": 0.4}


def values(overrides: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    p = {k: v[0] for k, v in PARAMS.items()}
    for k, v in (overrides or {}).items():
        if k not in p:
            raise KeyError(f"parametro di dj_tension sconosciuto: {k}")
        p[k] = v
    return p


def documented() -> dict[str, Any]:
    return {"version": DJ_TENSION_VERSION, "w_charge": W_CHARGE, "w_resolution": W_RESOLUTION, "w_cost": W_COST,
            "params": {k: {"value": v[0], "source": v[1]} for k, v in PARAMS.items()},
            "continuity": {k: {"value": v[0], "why": v[1]} for k, v in CONTINUITY.items()},
            "continuity_gap": CONTINUITY_GAP, "continuity_stop": CONTINUITY_STOP, "exposure": EXPOSURE}


def _sig(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, float(x))))))


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _wmean(comps: Mapping[str, Optional[float]], weights: Mapping[str, float]) -> Optional[float]:
    num = den = 0.0
    for k, w in weights.items():
        v = comps.get(k)
        if v is None or not np.isfinite(v):
            continue
        num += w * float(v)
        den += w
    return None if den <= 0 else num / den


# ------------------------------------------------------------------------------------------------ dati del brano
@dataclass
class TrackSignals:
    """Segnali di un brano per i punteggi (griglia, sezioni, tonalità, parole, DSP per battuta, croma, forma, modello)."""
    slug: str
    duration_s: float
    period_s: float
    downbeats_s: np.ndarray
    sections: list = field(default_factory=list)          # [{label, t}]
    camelot: Optional[str] = None
    key_level: Optional[str] = None
    words: list = field(default_factory=list)             # [{on_s, off_s, text, rhyme_key, vowel_key, confidence}]
    lang: Optional[str] = None
    bars: Optional[np.ndarray] = None                     # (n_bars, 6) medie per battuta delle colonne DSP
    bar_energy: Optional[np.ndarray] = None               # (n_bars,) energia del mix (dB, somma delle 4 bande)
    chroma_bars: Optional[np.ndarray] = None              # (n_bars, 12)
    form_labels: Optional[list] = None
    form_drops: frozenset = frozenset()
    model_exit: Optional[np.ndarray] = None               # tension_at_exit ai downbeat (percentile nel brano)
    model_entry: Optional[np.ndarray] = None              # resolution_at_entry ai downbeat (percentile nel brano)
    provenance: dict = field(default_factory=dict)

    @property
    def n_bars(self) -> int:
        return int(len(self.downbeats_s))

    def bar_of(self, t: float) -> int:
        return int(np.searchsorted(self.downbeats_s, float(t) + 1e-3) - 1)

    def nearest_bar(self, t: float) -> int:
        return int(np.argmin(np.abs(self.downbeats_s - float(t)))) if self.n_bars else -1

    @property
    def bar_s(self) -> float:
        return 4.0 * self.period_s

    def section_at(self, t: float) -> tuple[Optional[dict], Optional[dict], Optional[dict]]:
        """(sezione che contiene t, sezione precedente, successiva) secondo allin1."""
        secs = self.sections
        k = None
        for i, s in enumerate(secs):
            if s["t"] <= t + 1e-3:
                k = i
        if k is None:
            return None, None, (secs[0] if secs else None)
        return secs[k], (secs[k - 1] if k > 0 else None), (secs[k + 1] if k + 1 < len(secs) else None)


def _bar_table(x: np.ndarray, fps: float, downbeats: np.ndarray, duration: float) -> np.ndarray:
    edges = list(downbeats) + [min(duration, float(downbeats[-1]) + (float(downbeats[-1] - downbeats[-2]) if len(downbeats) > 1 else 2.0))]
    out = np.full((len(downbeats), x.shape[1]), np.nan, np.float32)
    for i in range(len(downbeats)):
        a, b = int(round(edges[i] * fps)), int(round(edges[i + 1] * fps))
        a, b = max(0, a), min(len(x), max(b, a + 1))
        if a < len(x):
            out[i] = np.asarray(x[a:b], np.float32).mean(0)
    return out


def _percentile_rank(v: np.ndarray) -> np.ndarray:
    out = np.full(len(v), np.nan)
    ok = np.isfinite(v)
    if ok.sum() >= 2:
        r = np.argsort(np.argsort(v[ok]))
        out[ok] = r / (ok.sum() - 1)
    elif ok.sum() == 1:
        out[ok] = 0.5
    return out


def model_at_downbeats(cue_doc: Optional[Mapping[str, Any]], downbeats: np.ndarray, key: str) -> Optional[np.ndarray]:
    """Valori del modello (``tension_exit``/``resolution_entry``) della cache ``runs/insong_v1/cue_scores`` ai downbeat."""
    if not cue_doc:
        return None
    cand = [c for c in cue_doc.get("candidates") or [] if c.get("what") == "downbeat" and c.get(key) is not None]
    if not cand:
        return None
    ts = np.asarray([c["t"] for c in cand])
    vs = np.asarray([c[key] for c in cand], float)
    out = np.full(len(downbeats), np.nan)
    for i, d in enumerate(downbeats):
        k = int(np.argmin(np.abs(ts - d)))
        if abs(ts[k] - d) <= 0.06:
            out[i] = vs[k]
    return out if np.isfinite(out).any() else None


def build_signals(slug: str, analysis: Mapping[str, Any], phrase_model: Optional[Mapping[str, Any]] = None, *,
                  dsp: Optional[np.ndarray] = None, chroma: Optional[np.ndarray] = None,
                  form: Optional[Mapping[str, Any]] = None, model_exit: Optional[np.ndarray] = None,
                  model_entry: Optional[np.ndarray] = None, provenance: Optional[dict] = None) -> TrackSignals:
    """Segnali dal documento d'analisi DJ (``dj_plan/tracks/<slug>.json``), dal modello ``dj_phrase`` e dalle cache."""
    tempo = analysis["tempo"]
    db = np.asarray(analysis["grid"]["downbeats_s"], float)
    dur = float(analysis["duration_s"])
    secs = sorted(({"label": str(s.get("label")), "t": float(s.get("downbeat_s", s.get("start_s", 0.0)))}
                   for s in analysis.get("sections") or []), key=lambda s: s["t"])
    words, lang = [], None
    if phrase_model:
        sr = int(phrase_model.get("sample_rate") or analysis.get("sample_rate") or 48000)
        lang = phrase_model.get("lang")
        for w in phrase_model.get("words") or []:
            if w.get("dropped"):
                continue
            words.append({"on_s": w["on"] / sr, "off_s": w["off"] / sr, "text": w.get("text", ""), "rhyme_key": w.get("rhyme_key"),
                          "vowel_key": w.get("vowel_key"), "confidence": w.get("confidence"), "clean": w.get("clean")})
        words.sort(key=lambda w: w["on_s"])
    sig = TrackSignals(slug=slug, duration_s=dur, period_s=float(tempo["period_s"]), downbeats_s=db, sections=secs,
                       camelot=(analysis.get("key") or {}).get("camelot"), key_level=(analysis.get("key") or {}).get("level"),
                       words=words, lang=lang, provenance=dict(provenance or {}))
    if dsp is not None and len(db) >= 2:
        sig.bars = _bar_table(np.asarray(dsp, np.float32), DSP_FPS, db, dur)
        e = sig.bars[:, :4].astype(np.float64)
        sig.bar_energy = (10 * np.log10(np.nansum(10 ** (e / 10.0), axis=1) + 1e-12)).astype(np.float32)
    if chroma is not None and len(db) >= 2:
        sig.chroma_bars = _bar_table(np.asarray(chroma, np.float32), CHROMA_FPS, db, dur)
    if form is not None and len(form.get("labels") or []) == len(db):
        sig.form_labels = list(form["labels"])
        sig.form_drops = frozenset(int(x) for x in form.get("drops") or [])
    sig.model_exit = None if model_exit is None else _percentile_rank(np.asarray(model_exit, float))
    sig.model_entry = None if model_entry is None else _percentile_rank(np.asarray(model_entry, float))
    return sig


# ----------------------------------------------------------------------------------------- caches DSP / croma
def compute_dsp(audio_path: str | Path, vocal_path: str | Path) -> np.ndarray:
    """Stessa ricetta di ``tools/build_insong_dataset.py dsp`` (16 kHz, 50 fps, 4 bande dB, flusso, voce): (n, 6) float16."""
    import librosa
    import soundfile as sf
    from scipy.signal import resample_poly

    def mono(p):
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != 16000:
            g = math.gcd(int(sr), 16000)
            x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
        return x

    sr, hop = 16000, int(16000 / DSP_FPS)
    x, v = mono(audio_path), mono(vocal_path)
    n = min(len(x), len(v))
    X = np.abs(librosa.stft(x[:n], n_fft=1024, hop_length=hop, center=True)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=1024)
    cols = []
    for lo, hi in ((20.0, 150.0), (150.0, 500.0), (500.0, 2000.0), (2000.0, 8000.0)):
        cols.append(10 * np.log10(X[(f >= lo) & (f < hi)].sum(0) + 1e-10))
    logm = np.log1p(1000 * np.sqrt(X))
    flux = np.r_[0.0, np.maximum(np.diff(logm, axis=1), 0).sum(0)]
    cols.append(flux / (np.median(flux) + 1e-6))
    V = np.abs(librosa.stft(v[:n], n_fft=1024, hop_length=hop, center=True)) ** 2
    vdb = 10 * np.log10(V.sum(0) + 1e-10)
    cols.append(vdb - np.percentile(vdb, 95))
    ref = np.percentile(np.stack(cols[:4]).max(0), 95)
    for i in range(4):
        cols[i] = cols[i] - ref
    return np.stack(cols, axis=1).astype(np.float16)


def compute_chroma(audio_path: str | Path) -> np.ndarray:
    """Croma come ``build_insong_dataset.py chroma`` (librosa chroma_stft, 16 kHz, 10 fps)."""
    import librosa
    import soundfile as sf
    from scipy.signal import resample_poly

    x, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        g = math.gcd(int(sr), 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return librosa.feature.chroma_stft(y=x, sr=16000, n_fft=4096, hop_length=1600).T.astype(np.float16)


def cached_array(kind: str, slug: str, dirs: Sequence[Path], compute: Optional[Callable[[], np.ndarray]] = None,
                 write_dir: Optional[Path] = None) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Cache ``<dir>/cache/<kind>/<slug>.npz`` (layout di runs/insong_v1); se assente e ``compute`` dato la crea in ``write_dir``."""
    for d in dirs:
        p = Path(d) / "cache" / kind / f"{slug}.npz"
        if p.is_file():
            return np.load(p)["x"], str(p)
    if compute is None or write_dir is None:
        return None, None
    arr = compute()
    p = Path(write_dir) / "cache" / kind / f"{slug}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, x=arr, fps=np.float32(DSP_FPS if kind == "dsp" else CHROMA_FPS))
    tmp.replace(p)
    return arr, str(p)


# -------------------------------------------------------------------------------------------------- carica di A
def _bar_slice(sig: TrackSignals, i0: int, i1: int) -> slice:
    return slice(max(0, i0), max(0, min(sig.n_bars, i1)))


def words_in(sig: TrackSignals, t0: float, t1: float) -> list[dict]:
    return [w for w in sig.words if w["on_s"] >= t0 and w["on_s"] < t1]


def _slope(y: np.ndarray) -> Optional[float]:
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return None
    x = np.arange(len(y), dtype=float)[ok]
    y = y[ok]
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x ** 2).sum())


FORM_TABLE = {  # [H] posizione nella forma -> carica; motivo in FORM_WHY
    "pre_chorus_end": 1.0, "build_end": 0.95, "approach_chorus": 0.8, "chorus_end": 0.75, "verse_end": 0.55,
    "inside_chorus": 0.5, "inside_other": 0.4, "mid_verse": 0.35, "early_verse": 0.15, "outro": 0.1, "intro": 0.05,
}
FORM_WHY = {
    "pre_chorus_end": "uscita alla fine della sezione che precede un ritornello: massima attesa (Huron p. 26: la tensione è "
                      "nel periodo immediatamente prima dell'esito; B prende il posto dell'arrivo)",
    "build_end": "fine di un build/bridge/strumentale (Butler p. 238: il buildup aumenta l'intensità)",
    "approach_chorus": "a 1-2 battute dalla fine della sezione che precede il ritornello",
    "chorus_end": "fine del ritornello completato (B&B cap. 13, pdf p. 33: 'mix straight after a chorus')",
    "verse_end": "fine di una strofa completata",
    "inside_chorus": "dentro il ritornello (non completato)", "inside_other": "dentro una sezione",
    "mid_verse": "a metà strofa", "early_verse": "nelle prime 4 battute di una strofa (contesto a bassa tensione)",
    "outro": "outro: A si sta scaricando", "intro": "intro: A non ha ancora caricato ([user] 'A campionata all'inizio')",
}


def form_position(sig: TrackSignals, t: float) -> dict[str, Any]:
    """Posizione dell'uscita nella forma di A (sezioni allin1 + etichette dj_form): chiave di :data:`FORM_TABLE`."""
    i = sig.nearest_bar(t)
    on_boundary = i >= 0 and abs(sig.downbeats_s[i] - t) <= 0.5 * sig.bar_s
    cur, prev, nxt = sig.section_at(t - 0.25 * sig.bar_s)   # sezione che A sta suonando prima dell'uscita
    lab = (cur or {}).get("label", "unknown")
    nxt_lab = (nxt or {}).get("label")
    end_t = (nxt or {}).get("t", sig.duration_s)
    bars_to_end = (end_t - t) / sig.bar_s
    bars_in = (t - (cur or {}).get("t", 0.0)) / sig.bar_s
    flab = None
    if sig.form_labels and i >= 1:
        flab = sig.form_labels[min(len(sig.form_labels) - 1, max(0, i - 1))]
    completed = on_boundary and abs(bars_to_end) <= 0.5
    if lab == "intro" or bars_in < 0:
        key = "intro"
    elif lab == "outro":
        key = "outro"
    elif completed and nxt_lab == "chorus" and lab != "chorus":
        key = "build_end" if lab in ("bridge", "inst", "break", "solo") else "pre_chorus_end"
    elif flab == "build" or (completed and lab in ("bridge", "inst", "break", "solo")):
        key = "build_end"
    elif nxt_lab == "chorus" and lab != "chorus" and 0.5 < bars_to_end <= 2.5:
        key = "approach_chorus"
    elif completed and lab == "chorus":
        key = "chorus_end"
    elif completed and lab == "verse":
        key = "verse_end"
    elif lab == "chorus":
        key = "inside_chorus"
    elif lab == "verse":
        key = "early_verse" if bars_in < 4 else "mid_verse"
    else:
        key = "inside_other"
    return {"key": key, "value": FORM_TABLE[key], "why": FORM_WHY[key], "section": lab, "next_section": nxt_lab,
            "bars_into_section": round(float(bars_in), 2), "bars_to_section_end": round(float(bars_to_end), 2),
            "form_label": flab, "completed": bool(completed)}


def minutage(sig: TrackSignals, t: float, played_s: Optional[float] = None, params: Optional[Mapping[str, Any]] = None) -> dict:
    p = values(params)
    frac = float(t) / max(1e-6, sig.duration_s)
    m_pos = _clip01((frac - p["pos_min_frac"]) / (p["pos_full_frac"] - p["pos_min_frac"]))
    m_play = None
    if played_s is not None:
        m_play = _clip01((float(played_s) - p["played_min_s"]) / (p["played_full_s"] - p["played_min_s"]))
    v = m_pos if m_play is None else min(m_pos, m_play)
    return {"value": v, "position_frac": round(frac, 3), "position_credit": round(m_pos, 3),
            "played_s": None if played_s is None else round(float(played_s), 1),
            "played_credit": None if m_play is None else round(m_play, 3)}


def a_charge(sig: TrackSignals, t: float, played_s: Optional[float] = None, params: Optional[Mapping[str, Any]] = None) -> dict:
    """Carica di A all'uscita ``t`` (s, tempo di A): quanto A ha alzato la tensione nelle ultime battute.

    Componenti: ``model`` (``self_supervision.tension_at_exit`` come percentile nel brano), ``energy`` (pendenza di energia
    nelle ultime 4 battute + livello delle ultime 2 rispetto al brano), ``density`` (pendenza del flusso spettrale = densità
    d'attacchi, parole per battuta in aumento: Butler p. 238, Krims p. 64), ``form`` (:func:`form_position`); il risultato è
    scalato dal ``minutage`` (uscita troppo presto nel brano o dopo troppo poco materiale suonato)."""
    p = values(params)
    i = sig.nearest_bar(t)
    comps: dict[str, Optional[float]] = {}
    detail: dict[str, Any] = {}
    if sig.model_exit is not None and 0 <= i < len(sig.model_exit) and np.isfinite(sig.model_exit[i]):
        comps["model"] = float(sig.model_exit[i])
    if sig.bars is not None and i >= 2:
        e = sig.bar_energy[_bar_slice(sig, i - 4, i)]
        sl = _slope(e)
        med = float(np.nanmedian(sig.bar_energy))
        lvl = float(np.nanmean(sig.bar_energy[_bar_slice(sig, i - 2, i)])) - med
        if sl is not None and np.isfinite(lvl):
            comps["energy"] = 0.5 * _sig(sl / p["slope_db_scale"]) + 0.5 * _sig(lvl / p["level_db_scale"])
            detail["energy_slope_db_per_bar"] = round(sl, 3)
            detail["energy_last2_vs_median_db"] = round(lvl, 2)
        fl = _slope(sig.bars[_bar_slice(sig, i - 4, i), 4])
        b = sig.bar_s
        wl = len(words_in(sig, t - 2 * b, t)) / 2.0
        wp = len(words_in(sig, t - 6 * b, t - 2 * b)) / 4.0
        dens = [_sig(fl / p["flux_slope_scale"])] if fl is not None else []
        if sig.words:
            dens.append(_sig((wl - wp) / p["words_delta_scale"]))
        if dens:
            comps["density"] = sum(dens) / len(dens)
            detail.update(flux_slope=None if fl is None else round(fl, 4), words_per_bar_last2=wl, words_per_bar_prev4=wp)
    fp = form_position(sig, t)
    comps["form"] = fp["value"]
    base = _wmean(comps, W_CHARGE)
    mt = minutage(sig, t, played_s, params)
    val = None if base is None else base * (p["minutage_floor"] + (1 - p["minutage_floor"]) * mt["value"])
    why = [f"forma: {fp['why']} ({fp['section']}→{fp['next_section']})"]
    if mt["value"] < 1:
        why.append(f"minutaggio: uscita al {100 * mt['position_frac']:.0f} % del brano" +
                   (f", {mt['played_s']:.0f} s suonati" if mt["played_s"] is not None else "") + f" (credito {mt['value']:.2f})")
    if detail.get("energy_slope_db_per_bar") is not None:
        why.append(f"energia {detail['energy_slope_db_per_bar']:+.2f} dB/battuta, ultime 2 battute {detail['energy_last2_vs_median_db']:+.1f} dB sulla mediana")
    return {"value": None if val is None else round(float(val), 4), "base": None if base is None else round(float(base), 4),
            "components": {k: round(float(v), 4) for k, v in comps.items() if v is not None}, "weights": W_CHARGE,
            "form": fp, "minutage": mt, "detail": detail, "why": why, "t": float(t), "bar": i}


# ------------------------------------------------------------------------------------------------ risoluzione di B
def b_resolution(sig: TrackSignals, t_audible: float, t_downbeat: Optional[float] = None, *, sig_a: Optional[TrackSignals] = None,
                 t_a: Optional[float] = None, params: Optional[Mapping[str, Any]] = None) -> dict:
    """Risoluzione all'ingresso di B, **non** come «ritornello»: segnali d'attacco nel punto in cui B diventa udibile.

    ``vocal_attack`` (prima parola di B entro una battuta dal downbeat, pickup compreso, dopo un respiro >= 1 battito, salto
    della voce nella prima battuta), ``energy_jump`` (energia e densità d'attacchi della prima battuta rispetto alle 2
    precedenti dello stesso B: valenza contrastiva, Huron pp. 39–40), ``kick_bass`` (banda 20–150 Hz che entra, drop di
    dj_form: Butler p. 246), ``vocal_foreground`` (voce in primo piano nelle prime 2 battute), ``register_contrast``
    (profilo per banda della prima battuta di B contro l'ultima di A), ``model`` (``resolution_at_entry``, peso basso: proxy
    «ritornello = arrivo» contrario ai voti)."""
    p = values(params)
    td = float(t_downbeat if t_downbeat is not None else t_audible)
    i = sig.nearest_bar(td)
    b = sig.bar_s
    per = sig.period_s
    comps: dict[str, Optional[float]] = {}
    detail: dict[str, Any] = {}
    # 1. attacco vocale
    if sig.words:
        first = next((w for w in sig.words if w["on_s"] >= float(t_audible) - 0.05), None)
        if first is not None and first["on_s"] <= td + 2 * b:
            k = sig.words.index(first)
            rest = (first["on_s"] - sig.words[k - 1]["off_s"]) / per if k > 0 else 99.0
            near = -b - 0.05 <= first["on_s"] - td <= b
            jump = None
            if sig.bars is not None and i >= 1:
                jump = float(sig.bars[i, 5] - sig.bars[i - 1, 5])
            breath = 1.0 if rest >= p["rest_beats"] else _clip01(rest / p["rest_beats"]) * 0.5
            s = 0.6 * breath + 0.4 * (_sig(jump / p["vocal_jump_scale"]) if jump is not None and np.isfinite(jump) else 0.5)
            comps["vocal_attack"] = s if near else 0.5 * s
            detail.update(first_word=first["text"], first_word_offset_beats=round((first["on_s"] - td) / per, 2),
                          rest_before_beats=round(min(rest, 99.0), 2), vocal_jump_db=None if jump is None else round(jump, 2),
                          first_word_near_downbeat=bool(near))
        else:
            comps["vocal_attack"] = p["instrumental_entry_attack"]
            detail["first_word"] = None
    if sig.bars is not None and 2 <= i < sig.n_bars:
        e = sig.bar_energy
        de = float(e[i] - np.nanmean(e[i - 2:i]))
        dfl = float(sig.bars[i, 4] - np.nanmean(sig.bars[i - 2:i, 4]))
        if np.isfinite(de) and np.isfinite(dfl):
            comps["energy_jump"] = 0.5 * _sig(de / p["energy_jump_scale"]) + 0.5 * _sig(dfl / p["flux_jump_scale"])
            detail.update(energy_jump_db=round(de, 2), flux_jump=round(dfl, 3))
        dl = float(sig.bars[i, 0] - sig.bars[i - 1, 0])
        kb = _sig(dl / p["low_jump_scale"]) if np.isfinite(dl) else None
        if i in sig.form_drops:
            kb = max(kb or 0.0, 0.9)
            detail["form_drop"] = True
        if kb is not None:
            comps["kick_bass"] = kb
            detail["low_jump_db"] = round(dl, 2) if np.isfinite(dl) else None
        v = float(np.nanmean(sig.bars[i:min(sig.n_bars, i + 2), 5]))
        if np.isfinite(v):
            comps["vocal_foreground"] = _sig((v + p["vocal_fg_offset_db"]) / 3.0)
            detail["vocal_level_db_rel95"] = round(v, 2)
    if sig_a is not None and t_a is not None and sig_a.bars is not None and sig.bars is not None:
        ia = sig_a.nearest_bar(float(t_a))
        if ia >= 1 and 0 <= i < sig.n_bars:
            pa = sig_a.bars[ia - 1, :4].astype(float)
            pb = sig.bars[i, :4].astype(float)
            if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                d = float(np.linalg.norm((pa - pa.mean()) - (pb - pb.mean())))
                comps["register_contrast"] = _sig((d - p["register_center_db"]) / p["register_scale_db"])
                detail["register_distance_db"] = round(d, 2)
    if sig.model_entry is not None and 0 <= i < len(sig.model_entry) and np.isfinite(sig.model_entry[i]):
        comps["model"] = float(sig.model_entry[i])
    val = _wmean(comps, W_RESOLUTION)
    why = []
    if detail.get("first_word"):
        why.append(f"prima parola '{detail['first_word']}' a {detail['first_word_offset_beats']:+.2f} battiti dal downbeat, "
                   f"respiro di {detail['rest_before_beats']:.2f} battiti prima")
    elif "vocal_attack" in comps:
        why.append("nessuna voce di B entro 2 battute: ingresso strumentale")
    if detail.get("energy_jump_db") is not None:
        why.append(f"salto di energia {detail['energy_jump_db']:+.1f} dB, densità {detail['flux_jump']:+.2f}")
    return {"value": None if val is None else round(float(val), 4), "components": {k: round(float(v), 4) for k, v in comps.items()
                                                                                   if v is not None},
            "weights": W_RESOLUTION, "detail": detail, "why": why, "t_audible": float(t_audible), "t_downbeat": td, "bar": i}


# ------------------------------------------------------------------------------------------------------ continuità
def technique_kind(decision: Mapping[str, Any]) -> str:
    t = str(decision.get("type_key") or decision.get("type") or "")
    if t.startswith("blend"):
        return "blend"
    return str(decision.get("type") or t)


def continuity(decision: Mapping[str, Any]) -> dict[str, Any]:
    """Continuità della base **pianificata** dalla tecnica e dai suoi parametri (tabella [H] :data:`CONTINUITY`)."""
    kind = technique_kind(decision)
    params = ((decision.get("render") or {}).get("params") or decision.get("params") or {})
    if kind not in CONTINUITY:
        return {"value": CONTINUITY_UNKNOWN, "kind": kind, "why": "tecnica senza voce nella tabella", "gap_beats": None}
    v, why = CONTINUITY[kind]
    gap = params.get("gap_beats")
    if kind in ("build_into_chorus", "halving_build") and gap and float(gap) > 0:
        v, why = CONTINUITY_GAP, f"pausa di {float(gap):g} battiti prima del downbeat di B: la base si interrompe"
    if kind == "break_before_entry" and params.get("mode") == "stop":
        v, why = CONTINUITY_STOP, "stop: la base si ferma"
    return {"value": float(v), "kind": kind, "why": why, "gap_beats": None if gap is None else float(gap)}


#: Versione della continuità misurata sul render (indipendente da :data:`DJ_TENSION_VERSION`: non entra nel costo).
#: .2 (22/10/2026, set v4b): il groove complessivo (banda bassa **o** banda delle percussioni 5-14 kHz sul mix A+B) con
#: riferimenti per lato (A prima del punto udibile di B, B dopo); .1 guardava solo 30-200 Hz con un riferimento unico, e
#: un filter out (passa-alto su A) risultava un buco.
CONTINUITY_MEASURED_VERSION = "musicmix.dj_tension.continuity_measured.2"
GROOVE_LOW_HZ = (30.0, 200.0)      # cassa/basso/808
GROOVE_PERC_HZ = (5000.0, 14000.0)  # charleston/rullante/piatti: restano sotto il passa-alto di un filter out (fino a 1,2 kHz) [H]


def continuity_measured(audio: np.ndarray, sr: int, switch_sample: int, period_s: float, *,
                        window_beats: tuple[float, float] = (-2.0, 1.0), drop_db: float = 15.0,
                        b_audible_sample: Optional[int] = None) -> dict[str, Any]:
    """Buco della base misurato sul render (``CONTINUITY_MEASURED_VERSION``).

    Frame di 1/8 di battito nella finestra ``window_beats`` attorno allo switch. Un frame ha la base **viva** se la banda
    bassa (30-200 Hz: cassa/basso di A o di B) **o** la banda delle percussioni (5-14 kHz: charleston/rullante, che il
    passa-alto di un filter out non toglie) è entro ``drop_db`` [H] dal riferimento **del proprio lato**: mediana delle
    battute prima (frame da -8 a -2 battiti dallo switch: A) prima dello switch, mediana di quelle dopo (da +2 a +8
    battiti: B, la cui base entra sul downbeat) dopo. Fra il punto in cui B diventa udibile (``b_audible_sample``,
    lead-in della sola voce di B) e lo switch vale il più basso dei due riferimenti (la base lì è ancora A). Il buco è la
    corsa più lunga di frame senza base; ``value`` = 1 - buco/2 battiti. ``gap_beats_low`` riporta la sola banda bassa
    (misura .1 con riferimenti per lato), ``gap_beats_low_v1`` la misura .1 com'era (riferimento unico), solo diagnostica.
    Limiti: è energia, non riconoscimento della batteria: sibilanti o coda d'eco nella banda alta possono tenere viva la
    base per qualche frame; un brano B senza batteria né basso all'ingresso dà un buco vero."""
    from scipy.signal import butter, sosfilt

    x = np.asarray(audio, np.float64)
    x = x.mean(axis=1) if x.ndim == 2 else x
    hop = max(1, int(round(period_s / 8 * sr)))
    nyq = sr / 2.0
    bands = {"low": GROOVE_LOW_HZ, "perc": (GROOVE_PERC_HZ[0], min(GROOVE_PERC_HZ[1], 0.95 * nyq))}

    def env(y):
        n = len(y) // hop
        return 10 * np.log10(np.mean(y[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)

    envs = {}
    for name, (lo_hz, hi_hz) in bands.items():
        if hi_hz <= lo_hz:
            continue
        envs[name] = env(sosfilt(butter(4, [lo_hz, hi_hz], btype="bandpass", fs=sr, output="sos"), x))
    envs["full"] = env(x)
    n_fr = len(envs["full"])
    k0 = int(switch_sample) // hop
    kb = min((int(b_audible_sample) // hop) if b_audible_sample is not None else k0, k0)
    a = max(0, k0 + int(window_beats[0] * 8))
    b = min(n_fr, k0 + int(window_beats[1] * 8))
    if b <= a:
        return {"value": None, "why": "finestra fuori dal render", "version": CONTINUITY_MEASURED_VERSION}
    pre_idx = np.arange(max(0, k0 - 64), max(0, k0 - 16))
    post_idx = np.arange(min(n_fr, k0 + 16), min(n_fr, k0 + 64))
    all_idx = np.r_[np.arange(k0 - 64, k0 - 16), np.arange(k0 + 16, k0 + 64)]
    all_idx = all_idx[(all_idx >= 0) & (all_idx < n_fr)]
    fr = np.arange(a, b)
    side_b, lead = fr >= k0, (fr >= kb) & (fr < k0)

    def run_len(mask):
        run = best = 0
        for v in mask:
            run = run + 1 if v else 0
            best = max(best, run)
        return best

    out: dict[str, Any] = {}
    alive_any = np.zeros(b - a, bool)
    measured = False
    for name, e in envs.items():
        refs = {}
        for side, idx in (("a", pre_idx), ("b", post_idx)):
            refs[side] = float(np.median(e[idx])) if idx.size else (float(np.median(e[all_idx])) if all_idx.size else None)
        if refs["a"] is None or refs["b"] is None:
            out[name] = None
            continue
        thr = np.where(side_b, refs["b"], np.where(lead, min(refs["a"], refs["b"]), refs["a"])) - drop_db
        alive = e[a:b] >= thr
        out[name] = {"gap_beats": run_len(~alive) / 8.0, "ref_a_db": round(refs["a"], 1), "ref_b_db": round(refs["b"], 1),
                     "min_db": round(float(e[a:b].min()), 1)}
        if name in ("low", "perc"):
            alive_any |= alive
            measured = True
    if not measured:
        return {"value": None, "why": "riferimenti fuori dal render", "version": CONTINUITY_MEASURED_VERSION}
    g = run_len(~alive_any) / 8.0
    v1 = None
    if all_idx.size:
        ref1 = float(np.median(envs["low"][all_idx]))
        v1 = run_len(envs["low"][a:b] < ref1 - drop_db) / 8.0
    return {"value": round(_clip01(1.0 - g / 2.0), 4), "gap_beats": g,
            "gap_beats_low": (out.get("low") or {}).get("gap_beats"), "gap_beats_perc": (out.get("perc") or {}).get("gap_beats"),
            "gap_beats_full": (out.get("full") or {}).get("gap_beats"), "gap_beats_low_v1": v1,
            "b_audible_frame_offset_beats": (kb - k0) / 8.0, "bands": out, "version": CONTINUITY_MEASURED_VERSION,
            "why": (f"groove (30-200 Hz o 5-14 kHz, riferimento per lato) assente per {g:.2f} battiti attorno allo switch"
                    + (f"; sola banda bassa {out['low']['gap_beats']:.2f}" if out.get("low") else ""))}


# ------------------------------------------------------------------------------------------------- scontro tonale
def parse_camelot(c: Optional[str]) -> Optional[tuple[int, str]]:
    if not c:
        return None
    c = str(c).strip().upper()
    try:
        return int(c[:-1]), c[-1]
    except ValueError:
        return None


def camelot_shift(c: Optional[str], semitones: int) -> Optional[str]:
    """Camelot dopo una trasposizione di ``semitones`` (un semitono in su = +7 posizioni sulla ruota)."""
    k = parse_camelot(c)
    if k is None:
        return None
    n = (k[0] - 1 + 7 * int(semitones)) % 12 + 1
    return f"{n}{k[1]}"


def camelot_distance(a: Optional[str], b: Optional[str]) -> Optional[int]:
    """Passi sulla ruota di Camelot: 0 stessa, 1 adiacente o relativa (stesso numero, lettera diversa), poi +1 per passo."""
    ka, kb = parse_camelot(a), parse_camelot(b)
    if ka is None or kb is None:
        return None
    d = abs(ka[0] - kb[0]) % 12
    d = min(d, 12 - d)
    return d + (1 if ka[1] != kb[1] else 0)


def _chroma_vec(sig: TrackSignals, i0: int, i1: int) -> Optional[np.ndarray]:
    if sig.chroma_bars is None:
        return None
    c = sig.chroma_bars[_bar_slice(sig, i0, i1)]
    c = c[np.all(np.isfinite(c), axis=1)] if len(c) else c
    if not len(c):
        return None
    v = c.astype(float).mean(0)
    return v if v.sum() > 0 else None


def _peak(v: Optional[np.ndarray], p: Mapping[str, Any]) -> Optional[float]:
    if v is None:
        return None
    pk = float(v.max() / (v.sum() + 1e-9))
    return _clip01((pk - p["chroma_peak_floor"]) / p["chroma_peak_span"])


def harmonic_clash(sig_a: TrackSignals, t_a: float, sig_b: TrackSignals, t_b: float, decision: Optional[Mapping[str, Any]] = None,
                   *, shift_b: int = 0, params: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Scontro tonale vicino allo switch, **anche senza sovrapposizione** ([user]: «tonalità discordante» con l'autotune;
    contro B&B pdf p. 39 «Cut them, don't blend them»): distanza Camelot dopo l'eventuale trasposizione di B (pesata dalla
    confidenza delle chiavi) e distanza del croma misurato (ultime 2 battute di A, prime 2 di B), moltiplicate per la
    presenza tonale ai due lati (picco del croma) e per l'esposizione della tecnica (:data:`EXPOSURE`)."""
    p = values(params)
    kind = technique_kind(decision or {"type": "cut"})
    exposure = EXPOSURE["overlap"] if kind in OVERLAP_KINDS else EXPOSURE.get(kind, 0.5)
    cb = camelot_shift(sig_b.camelot, shift_b) if shift_b else sig_b.camelot
    d = camelot_distance(sig_a.camelot, cb)
    conf = min(KEY_CONF.get(str(sig_a.key_level), 0.5), KEY_CONF.get(str(sig_b.key_level), 0.5))
    key_term = None if d is None else (0.0 if d <= 1 else 0.5 if d == 2 else 1.0) * conf
    ia, ib = sig_a.nearest_bar(t_a), sig_b.nearest_bar(t_b)
    va, vb = _chroma_vec(sig_a, ia - 2, ia), _chroma_vec(sig_b, ib, ib + 2)
    chroma_term = cos = None
    if va is not None and vb is not None:
        vbs = np.roll(vb, int(shift_b))
        cos = float(va @ vbs / (np.linalg.norm(va) * np.linalg.norm(vbs) + 1e-9))
        chroma_term = _clip01((1.0 - cos) / p["chroma_dist_scale"])
    pa, pb = _peak(va, p), _peak(vb, p)
    presence = math.sqrt(pa * pb) if pa is not None and pb is not None else 0.5
    terms = {"key": key_term, "chroma": chroma_term}
    w = {"key": p["key_weight"], "chroma": 1.0 - p["key_weight"]}
    dist = _wmean(terms, w)
    val = None if dist is None else exposure * presence * dist
    why = [f"Camelot {sig_a.camelot}→{cb} ({'?' if d is None else d} passi, confidenza {conf:.1f})"]
    if cos is not None:
        why.append(f"croma fine A / inizio B: coseno {cos:.2f}")
    why.append(f"presenza tonale {presence:.2f}, esposizione della tecnica '{kind}' {exposure:.2f}")
    return {"value": None if val is None else round(float(val), 4), "camelot_distance": d, "key_term": key_term,
            "chroma_cosine": None if cos is None else round(cos, 4), "chroma_term": chroma_term,
            "tonal_presence": round(presence, 3), "presence_a": pa, "presence_b": pb, "exposure": exposure, "shift_b": int(shift_b),
            "kind": kind, "why": why}


# ------------------------------------------------------------------------------------------------- affinità di testo
@functools.lru_cache(maxsize=200_000)
def _pair_affinity(ta: str, tb: str, ra: str, rb: str, va: str, vb: str, lang_a: str, lang_b: str) -> tuple[float, str]:
    best, kind = 0.0, ""
    if ra and rb and ra == rb and ta.lower() != tb.lower():
        best, kind = 0.75, "rima (CMUdict)"
    try:  # legami di dj_mashup (uguale, lemma, rima ortografica, Soundex, vocalizzo, allitterazione)
        from musicmix import dj_mashup as M

        m = M.sound_affinity(ta, tb, lang_a or None, lang_b or None)
    except Exception:  # pragma: no cover - dj_mashup non importabile: solo dj_creative
        from musicmix import dj_creative as cr

        m = cr.word_match(ta, tb, lang_a or None, lang_b or None)
    if m and float(m["score"]) > best:
        best, kind = float(m["score"]), str(m["type"])
    if best < 0.3 and va and vb and va == vb and len(ta) >= 3 and len(tb) >= 3:
        from musicmix import dj_creative as cr

        if cr.normalize_word(ta) not in cr.STOPWORDS and cr.normalize_word(tb) not in cr.STOPWORDS:
            best, kind = 0.3, "assonanza (vocale tonica)"
    return best, kind


def lyric_affinity(words_a_end: Sequence[Mapping[str, Any]], words_b_start: Sequence[Mapping[str, Any]], *,
                   lang_a: Optional[str] = None, lang_b: Optional[str] = None, min_confidence: float = 0.3) -> dict[str, Any]:
    """Rime, assonanze, allitterazioni e vocalizzi tra le ultime parole di A (max 3) e le prime di B (max 4).

    Riusa ``dj_mashup.sound_affinity`` (che parte da ``dj_creative.word_match``: uguale, lemma, rima, Soundex; poi vocalizzo e
    allitterazione) e la chiave di rima fonetica CMUdict di ``dj_phrase`` (``rhyme_key``/``vowel_key``) [H]. Punteggio =
    legame x radice della confidenza ASR minima x bonus di posizione (ultima di A 1,1, prima di B 1,05), al massimo 1."""
    A = list(words_a_end)[-3:]
    B = list(words_b_start)[:4]
    best = {"value": 0.0, "a_word": None, "b_word": None, "kind": None}
    n = 0
    for ka, wa in enumerate(A):
        for kb, wb in enumerate(B):
            ca = wa.get("confidence")
            cb = wb.get("confidence")
            conf = min(1.0 if ca is None else float(ca), 1.0 if cb is None else float(cb))
            if conf < min_confidence:
                continue
            s, kind = _pair_affinity(str(wa.get("text", "")), str(wb.get("text", "")), str(wa.get("rhyme_key") or ""),
                                     str(wb.get("rhyme_key") or ""), str(wa.get("vowel_key") or ""), str(wb.get("vowel_key") or ""),
                                     str(lang_a or ""), str(lang_b or ""))
            if s <= 0:
                continue
            n += 1
            v = min(1.0, s * math.sqrt(conf) * (1.1 if ka == len(A) - 1 else 1.0) * (1.05 if kb == 0 else 1.0))
            if v > best["value"]:
                best = {"value": round(v, 4), "a_word": wa.get("text"), "b_word": wb.get("text"), "kind": kind}
    best["n_links"] = n
    best["a_words"] = [w.get("text") for w in A]
    best["b_words"] = [w.get("text") for w in B]
    best["why"] = (f"'{best['a_word']}' → '{best['b_word']}' ({best['kind']})" if best["a_word"] else "nessun legame di suono")
    return best


def words_end_of_a(sig: TrackSignals, t_exit: float, min_gap_s: float = 0.15, window_bars: float = 2.0) -> list[dict]:
    """Ultime parole di A finite almeno ``min_gap_s`` prima dell'uscita, entro ``window_bars`` battute (come find_lyric_links)."""
    return [w for w in sig.words if w["on_s"] >= t_exit - window_bars * sig.bar_s and w["off_s"] <= t_exit - min_gap_s][-3:]


def words_start_of_b(sig: TrackSignals, t_audible: float, window_bars: float = 2.0) -> list[dict]:
    return [w for w in sig.words if t_audible - 0.05 <= w["on_s"] <= t_audible + window_bars * sig.bar_s][:4]


# ------------------------------------------------------------------------------------------------ transizione intera
def b_audible_s(plan: Mapping[str, Any], sr: int = 48000) -> float:
    """Punto in cui B diventa udibile: ``b_lead_in.audible_sample`` della guardia d'ingresso, altrimenti il downbeat."""
    li = plan.get("b_lead_in") or {}
    if li.get("status") == "lead_in" and li.get("audible_sample") is not None:
        return float(li["audible_sample"]) / sr
    return float(plan["b_in_s"])


def score_transition(sig_a: TrackSignals, sig_b: TrackSignals, *, a_out_s: float, b_in_s: float, decision: Mapping[str, Any],
                     b_audible: Optional[float] = None, played_a_s: Optional[float] = None, shift_b: int = 0,
                     params: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Tutti i punteggi di una transizione e il costo [H] per il pianificatore (:data:`W_COST`)."""
    tb_aud = float(b_in_s if b_audible is None else b_audible)
    ch = a_charge(sig_a, a_out_s, played_a_s, params)
    rs = b_resolution(sig_b, tb_aud, b_in_s, sig_a=sig_a, t_a=a_out_s, params=params)
    co = continuity(decision)
    cl = harmonic_clash(sig_a, a_out_s, sig_b, b_in_s, decision, shift_b=shift_b, params=params)
    ly = lyric_affinity(words_end_of_a(sig_a, a_out_s), words_start_of_b(sig_b, tb_aud), lang_a=sig_a.lang, lang_b=sig_b.lang)
    cost = transition_cost(ch["value"], rs["value"], co["value"], cl["value"], ly["value"])
    return {"version": DJ_TENSION_VERSION, "a_charge": ch, "b_resolution": rs, "continuity": co, "harmonic_clash": cl,
            "lyric_affinity": ly, "cost": cost}


def transition_cost(charge: Optional[float], resolution: Optional[float], cont: Optional[float], clash: Optional[float],
                    lyric: Optional[float]) -> dict[str, Any]:
    """Costo [H] = w_c (1 - carica) + w_r (1 - risoluzione) + w_k (1 - continuità) + w_h scontro - w_l affinità.
    Un punteggio assente (astensione) vale 0,5 (né premio né penalità relativa), dichiarato in ``abstained``."""
    vals = {"charge": charge, "resolution": resolution, "continuity": cont, "clash": clash, "lyric": lyric}
    abst = [k for k, v in vals.items() if v is None]
    f = {k: (0.5 if v is None else float(v)) for k, v in vals.items()}
    if lyric is None:
        f["lyric"] = 0.0
    terms = {"charge": W_COST["charge"] * (1 - f["charge"]), "resolution": W_COST["resolution"] * (1 - f["resolution"]),
             "continuity": W_COST["continuity"] * (1 - f["continuity"]), "clash": W_COST["clash"] * f["clash"],
             "lyric": -W_COST["lyric"] * f["lyric"]}
    return {"total": round(float(sum(terms.values())), 4), "terms": {k: round(v, 4) for k, v in terms.items()}, "abstained": abst}


def preference_score(rec: Mapping[str, Any]) -> float:
    """Punteggio complessivo pre-registrato (più alto = più vicino ai criteri dell'utente) = - costo."""
    return -float(rec["cost"]["total"])


# ---------------------------------------------------------------------------------------------------- hook del set
class TensionHook:
    """Hook del runner (``SetHooks.tension``): ``adjust_pair`` dopo affidabilità/novità, ``report`` nel set.json.

    ``signals(slug) -> TrackSignals | None`` fornisce i segnali dei brani. ``mode``: ``prefer`` (costo [H] aggiunto ai
    candidati ammissibili: nessun candidato diventa ammissibile o non ammissibile per la tensione, si sceglie solo tra gli
    ammissibili) o ``report`` (misura senza cambiare)."""

    def __init__(self, signals: Callable[[str], Optional[TrackSignals]], mode: str = "prefer"):
        if mode not in ("prefer", "report"):
            raise ValueError(mode)
        self.signals = signals
        self.mode = mode
        self.pairs: dict[tuple[str, str], dict] = {}
        self._mem: dict[tuple, dict] = {}

    def score_candidate(self, a: str, b: str, c: Mapping[str, Any]) -> Optional[dict]:
        sa, sb = self.signals(a), self.signals(b)
        if sa is None or sb is None:
            return None
        pl, dec = c["plan"], c["decision"]
        key = (a, b, round(float(pl["a_out_s"]), 3), round(float(pl["b_in_s"]), 3), round(b_audible_s(pl), 3),
               json.dumps(continuity(dec), sort_keys=True), technique_kind(dec))
        if key not in self._mem:
            self._mem[key] = score_transition(sa, sb, a_out_s=float(pl["a_out_s"]), b_in_s=float(pl["b_in_s"]), decision=dec,
                                              b_audible=b_audible_s(pl))
        return self._mem[key]

    def adjust_pair(self, pc: dict, a: str, b: str, analyses: Mapping[str, Any] | None = None) -> dict:
        n = 0
        best0 = next((c for c in pc["candidates"] if c["admissible"]), None)
        for c in pc["candidates"]:
            if not c["admissible"]:
                continue
            r = self.score_candidate(a, b, c)
            if r is None:
                continue
            c["tension"] = compact(r)
            n += 1
            if self.mode == "prefer":
                c["cost"] = float(c["cost"]) + float(r["cost"]["total"])
        if self.mode == "prefer" and n:
            pc["candidates"].sort(key=lambda c: (not c["admissible"], c["cost"], -c["plan"]["score"]))
        best1 = next((c for c in pc["candidates"] if c["admissible"]), None)

        def sig(c):
            return None if c is None else [int(c["plan"]["a_out_bar"]), int(c["plan"]["b_in_bar"]), c["decision"]["type_key"]]

        self.pairs[(a, b)] = {"a": a, "b": b, "scored": n, "best_before": sig(best0), "best_after": sig(best1)}
        return pc

    def order_attempts(self, attempts: list, cands: Sequence[Mapping[str, Any]], a: str, b: str) -> list:
        """Tentativi sullo stesso candidato riordinati per (continuità - scontro) della tecnica: la base continua prima
        ([user]); a parità l'ordine originale (policy/albero). Ordine tra candidati invariato."""
        if self.mode != "prefer":
            return attempts
        out, i = [], 0
        while i < len(attempts):
            j = i
            while j < len(attempts) and attempts[j][0] == attempts[i][0]:
                j += 1
            grp = attempts[i:j]

            def key(x, cand=cands[attempts[i][0]]):
                co = continuity(x[1])["value"]
                r = self.score_candidate(a, b, {"plan": cand["plan"], "decision": x[1]})
                cl = (r or {}).get("harmonic_clash", {}).get("value") or 0.0
                return -(W_COST["continuity"] * co - W_COST["clash"] * cl)

            out += [g for _, g in sorted(enumerate(grp), key=lambda e: (key(e[1]), e[0]))]
            i = j
        return out

    def report(self, ctx: Mapping[str, Any]) -> dict:
        order, chosen = list(ctx["order"]), list(ctx["chosen"])
        trans = []
        for t, c in enumerate(chosen):
            r = c.get("tension") or {}
            trans.append({"index": t + 1, "a": order[t], "b": order[t + 1], "a_out_bar": int(c["plan"]["a_out_bar"]),
                          "b_in_bar": int(c["plan"]["b_in_bar"]), "type": c["decision"]["type_key"], **r})
        changed = [x for x in self.pairs.values() if x["best_before"] != x["best_after"]]
        return {"version": DJ_TENSION_VERSION, "mode": self.mode, "params": documented(), "planned_transitions": trans,
                "pairs": {"n": len(self.pairs), "best_changed": len(changed)}}


def compact(r: Mapping[str, Any]) -> dict[str, Any]:
    """Riassunto di :func:`score_transition` per il set.json."""
    return {"a_charge": r["a_charge"]["value"], "a_charge_form": r["a_charge"]["form"]["key"],
            "a_charge_components": r["a_charge"]["components"], "a_minutage": r["a_charge"]["minutage"]["value"],
            "b_resolution": r["b_resolution"]["value"], "b_resolution_components": r["b_resolution"]["components"],
            "continuity": r["continuity"]["value"], "continuity_why": r["continuity"]["why"],
            "harmonic_clash": r["harmonic_clash"]["value"], "camelot_distance": r["harmonic_clash"]["camelot_distance"],
            "lyric_affinity": r["lyric_affinity"]["value"], "lyric_link": r["lyric_affinity"]["why"],
            "tension_cost": r["cost"]["total"], "cost_terms": r["cost"]["terms"]}


# ------------------------------------------------------------------------------------------- caricamento da una run
TENSION_CACHE = ROOT / ".cache" / "dj_tension"
FORM_DIRS = (ROOT / ".cache/track_cards/form", ROOT / ".cache/dj_set_v3_events_plan/form")


def _read(p: Path) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def load_form_labels(slug: str, n_bars: int, dirs: Sequence[Path] = FORM_DIRS) -> tuple[Optional[dict], Optional[str]]:
    for d in dirs:
        p = Path(d) / f"{slug}.json"
        if p.is_file():
            f = (_read(p).get("form") or {})
            if len(f.get("labels") or []) == n_bars:
                return f, str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p)
    return None, None


def load_run_signals(slug: str, run_dir: str | Path, *, compute_missing: bool = True, cache_root: Path = TENSION_CACHE,
                     form: Optional[Mapping[str, Any]] = None) -> TrackSignals:
    """Segnali di un brano preparato da ``run_dj_set_v3_bridge.py prepare`` (``<run>/dj_plan/tracks``, ``<run>/phrase``).

    DSP e croma: cache di ``runs/insong_v1`` se presenti, altrimenti calcolati (stessa ricetta) in ``.cache/dj_tension``.
    Modello di carica/arrivo: ``runs/insong_v1/cue_scores`` (fold che non ha visto l'artista) se il brano c'è; altrimenti
    ``self_supervision.tension_score`` col modello addestrato su tutti i 18 brani (non out-of-fold: dichiarato)."""
    from .paths import localize

    run_dir = Path(run_dir)
    an = _read(run_dir / "dj_plan" / "tracks" / f"{slug}.json")
    pj = run_dir / "phrase" / f"{slug}.json"
    pdoc = _read(pj) if pj.is_file() else None
    model = (pdoc or {}).get("model")
    audio = localize(an["inputs"]["canonical_wav"])
    vocal = None
    if pdoc and (pdoc.get("inputs") or {}).get("vocal_stem"):
        v = Path(pdoc["inputs"]["vocal_stem"])
        vocal = localize(str(v if v.is_absolute() else ROOT / v))
    prov: dict[str, Any] = {"analysis": str((run_dir / "dj_plan" / "tracks" / f"{slug}.json").relative_to(ROOT))
                            if str(run_dir).startswith(str(ROOT)) else str(run_dir), "phrase": str(pj) if pdoc else None}
    can = compute_missing and audio.is_file() and vocal is not None and Path(vocal).is_file()
    dsp, prov["dsp"] = cached_array("dsp", slug, (INSONG, cache_root),
                                    (lambda: compute_dsp(audio, vocal)) if can else None, cache_root)
    chroma, prov["chroma"] = cached_array("chroma", slug, (INSONG, cache_root),
                                          (lambda: compute_chroma(audio)) if compute_missing and audio.is_file() else None, cache_root)
    db = np.asarray(an["grid"]["downbeats_s"], float)
    if form is None:
        form, prov["form"] = load_form_labels(slug, len(db))
    cue = INSONG / "cue_scores" / f"{slug}.json"
    m_exit = m_entry = None
    if cue.is_file():
        doc = _read(cue)
        m_exit = model_at_downbeats(doc, db, "tension_exit")
        m_entry = model_at_downbeats(doc, db, "resolution_entry")
        prov["model"] = {"source": str(cue.relative_to(ROOT)), "oof": bool(doc.get("oof")),
                         "files": {k: v for k, v in (doc.get("model_files") or {}).items() if k in ("tension_exit", "resolution_entry")}}
    elif dsp is not None and model is not None:
        try:
            from . import self_supervision as S

            ok = (db >= 8 * 4 * float(an["tempo"]["period_s"])) & (db <= float(an["duration_s"]) - 4 * 4 * float(an["tempo"]["period_s"]))
            m_exit = np.full(len(db), np.nan)
            m_entry = np.full(len(db), np.nan)
            ts = [float(x) for x in db[ok]]
            if ts:
                kw = dict(run_dir=cache_root, source_run=run_dir)
                ex = S.tension_score(slug, ts, "exit", model_file=str(INSONG / "model" / "tension_exit_dsp_form.npz"), **kw)
                en = S.tension_score(slug, ts, "entry", model_file=str(INSONG / "model" / "tension_entry_dsp_form.npz"), **kw)
                m_exit[ok] = [np.nan if v is None else v for v in ex]
                m_entry[ok] = [np.nan if v is None else v for v in en]
            prov["model"] = {"source": "self_supervision.tension_score (modello su tutti i 18 brani: non out-of-fold)", "oof": False}
        except Exception as exc:  # il modello è una componente: senza, si astiene
            prov["model"] = {"error": f"{type(exc).__name__}: {exc}"}
            m_exit = m_entry = None
    return build_signals(slug, an, model, dsp=dsp, chroma=chroma, form=form, model_exit=m_exit, model_entry=m_entry,
                         provenance=prov)
