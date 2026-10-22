"""Palestra automatica degli errori (defect gym): transizioni di riferimento e varianti con difetti controllati.

Requisito dell'utente (21/10/2026, AGENTS.md "Autonomia"): il sistema non deve aver bisogno dei voti dell'utente
per funzionare. La palestra prende transizioni di riferimento (voto utente >= 6, set v3 con QA pass, continuazioni
naturali dentro lo stesso brano) e ne crea varianti con **un** difetto (o due, nelle combinazioni) di intensità nota;
i valutatori automatici (``dj_qa`` e critico v3/v4) devono rifiutare la variante e non il riferimento.

Due famiglie di operatori (metadati sempre coerenti con l'audio della variante):

* ``rerender``: il difetto nasce a monte, come lo farebbe la pipeline (cue di A o di B spostato di k battute,
  griglia di B sbagliata di una frazione di battito, lead-in di B tolto, eco con ritardo fuori griglia). Il render
  è rifatto dal renderer vero con l'ingresso alterato: time-map, a_map/b_map e parole proiettate sono quelli che il
  renderer scrive. Questo modulo descrive gli operatori (:data:`DEFECTS`) e sceglie i parametri (es.
  :func:`find_straddle_shift`); il rifacimento del render è nel tool (``tools/defect_gym.py``), che conosce le ricette.
* ``deck``: il difetto è un errore di render/missaggio applicato ai deck della clip (salto di loudness, buco non
  dichiarato, clic, basso di A lasciato aperto, B trasposto, coda d'eco fuori tempo). Il mix è ricostruito dai deck e
  limitato come nel renderer (:func:`rebuild`); i metadati non dichiarano il difetto (un errore di render non si
  autodichiara), salvo il guadagno di B, che il renderer dichiara sempre (``loudness.b_gain_db``).

Rilevamento (:func:`classify`): una variante è *rilevata* se compare almeno un segnale di rifiuto (QA ``fail`` o veto
del critico) **assente nel riferimento**: il confronto è appaiato, così un riferimento già segnalato non rende
"rilevate" tutte le sue varianti. Se nessun segnale nuovo compare e il controllo atteso non è stato misurato
(``unavailable``/``not_applicable``) la variante è ``unavailable``: dichiarata, mai contata come pass.

Limite (da non dimenticare): riconoscere un peggioramento non equivale a riconoscere una bella transizione.
Il gate dice che QA e critico vedono i difetti di mestiere elencati, non che il resto suoni bene.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

GYM_VERSION = "musicmix.defect_gym.2"  # .2 (22/10/2026): dose minima del basso duplicato; split sviluppo/verifica nel report
MIX_COMPONENTS = ("a", "echo", "b", "overlay", "echo_gym")
WORD_STRADDLE_MIN_S = 0.12  # [H] la parola deve superare il punto di taglio di >= 120 ms per lato (QA usa 50 ms)
DECK_RAMP_S = 0.005
DOUBLE_BASS_MIN_DOSE_DB = -12.0  # [H] v2: A's injected low band at least this close to B's, else not a double bass
OVERLAP_KINDS = ("blend", "loop_extend", "acapella_over", "vocal_swap")  # due brani suonano insieme (come il runner v3)

# --------------------------------------------------------------------------------------------- catalogue
# Ogni difetto: famiglia, operatore, parametri per livello, segnali attesi (controlli che *dovrebbero* vederlo), base.
# Segnali: "qa:<controllo>" (dj_qa), "critic:<flag>" (veto del runner v3: regole di run_dj_set_v3_bridge.critic),
# "v4:<flag o controllo>" (blocco technical del critico v4). [P] pratica DJ dai libri, [H] ipotesi nostra.
DEFECTS: dict[str, dict[str, Any]] = {
    "word_cut_a": {
        "label": "parola mozzata in A", "family": "voce", "method": "rerender",
        "levels": {"straddle120": {"a_shift": "straddle", "min_side_s": WORD_STRADDLE_MIN_S}},
        "expect": ["qa:verse_cut"],
        "basis": "[P] B&B 2003 cap. 11 (il cut va nel silenzio tra i battiti, non sulla voce); AGENTS.md (gestione delle voci). "
                 "Operatore: cue di A spostato di k battute intere (griglia invariata) finché una parola ASR di A attraversa "
                 "l'uscita secca di A di >= 120 ms per lato [H]",
        "applies": "tutti i riferimenti con uscita secca di A dichiarata o misurabile",
    },
    "word_cut_b": {
        "label": "parola mozzata in B", "family": "voce", "method": "rerender",
        "levels": {"straddle120": {"b_shift": "straddle", "min_side_s": WORD_STRADDLE_MIN_S}},
        "expect": ["qa:b_entry"],
        "basis": "[P] B&B 2003 cap. 10 (iniziare a metà frase); QA .6 b_entry. Operatore: cue di B spostato di k battute "
                 "finché una parola raffinata di B (dj_phrase) attraversa il punto in cui B diventa udibile [H]",
        "applies": "ingressi puliti di B (cut, effetti, tecniche creative); non i blend (ingresso a fader)",
    },
    "pickup_removed": {
        "label": "sillabe di pickup di B tolte", "family": "voce", "method": "rerender",
        "levels": {"no_lead_in": {"no_b_lead_in": True}},
        "expect": ["qa:b_entry"],
        "basis": "[P] B&B 2003 cap. 10; INGRESSO_B (lead-in di dj_phrase). Operatore: stesso render senza il lead-in di B "
                 "dichiarato dal riferimento",
        "applies": "solo riferimenti che dichiarano un lead-in di B (meta.b_entry prima del downbeat)",
    },
    "phase": {
        "label": "sfasamento di B (griglia di B sbagliata)", "family": "ritmo", "method": "rerender",
        "levels": {"half_beat": {"b_grid_offset_beats": 0.5}, "quarter_beat": {"b_grid_offset_beats": 0.25},
                   "30ms": {"b_grid_offset_s": 0.030}},
        "expect": ["qa:rhythm", "critic:beat_phase_ms", "v4:beat_phase_ms"],
        "basis": "[P] Webber 2008 p. 244 (train wreck evidente al pubblico; 'slipped beat' per i DJ); B&B cap. 'How to play out'. "
                 "Operatore: griglia (battiti e downbeat) di B spostata di Δ prima del render, come un errore del tracker: "
                 "il renderer allinea la griglia sbagliata e dichiara un residuo ~0 [H]",
        "applies": "tutti",
    },
    "b_bar_late": {
        "label": "B entra 1 battuta in ritardo (fuori frase)", "family": "frase", "method": "rerender",
        "levels": {"1bar": {"b_shift": -1}},
        "expect": ["qa:phrase"],
        "basis": "[P] B&B 2003 cap. 10 (allineare le frasi); Butler 2006 p. 185. Operatore: cue di B una battuta prima del suo "
                 "inizio frase: la frase di B parte una battuta dopo il confine di A",
        "applies": "tutti",
    },
    "b_bar_early": {
        "label": "B entra 1 battuta in anticipo (fuori frase)", "family": "frase", "method": "rerender",
        "levels": {"1bar": {"b_shift": +1}},
        "expect": ["qa:phrase"],
        "basis": "[P] B&B 2003 cap. 10; Butler 2006 p. 185. Operatore: cue di B una battuta dopo il suo inizio frase",
        "applies": "tutti",
    },
    "double_bass": {
        "label": "basso duplicato (bass swap mancante)", "family": "eq", "method": "deck",
        "levels": {"2bars": {"bars": 2}},
        "expect": ["qa:double_bass"],
        "basis": "[P] B&B 2003 cap. 28 (mai due bassi pieni); REVISIONE_DJ. Operatore: banda < 200 Hz di A (sorgente, stesso "
                 "guadagno) lasciata aperta per 2 battute dopo lo switch",
        "applies": "tutti (A continua oltre lo switch nella sorgente)",
    },
    "loud_jump": {
        "label": "salto di loudness di B", "family": "loudness", "method": "deck",
        "levels": {"+3dB": {"db": 3.0}, "+6dB": {"db": 6.0}},
        "expect": ["qa:loudness_jump", "critic:loud_jump_db", "v4:loud_jump_db"],
        "basis": "[P] Webber 2008 p. 244 (i salti di volume sono train wreck); soglie QA 2/3 dB [H]. Operatore: deck B "
                 "(e voce udibile) +X dB, dichiarato in loudness.b_gain_db",
        "applies": "tutti",
    },
    "gap": {
        "label": "buco non dichiarato", "family": "loudness", "method": "deck",
        "levels": {"half_beat": {"beats": 0.5}, "1beat": {"beats": 1.0}},
        "expect": ["qa:dropout", "v4:dropout"],
        "basis": "[E citata] Huron 2006 p. 52 (un battito atteso omesso è un evento); QA D4 (>= 0,5 battiti, -30 dB) [H]. "
                 "Operatore: tutti i deck a zero dallo switch per X battiti (rampe 5 ms), nessuna pausa dichiarata",
        "applies": "tutti",
    },
    "click": {
        "label": "clic alla giunta", "family": "tecnico", "method": "deck",
        "levels": {"-20dBFS": {"amp": 0.1}, "-6dBFS": {"amp": 0.5}},
        "expect": ["qa:click"],
        "basis": "[H] errore di render (discontinuità). Operatore: impulso rettangolare di 1 ms sullo switch, prima del limitatore",
        "applies": "tutti",
    },
    "key_clash": {
        "label": "tonalità che stride (B trasposto nel blend)", "family": "tonalità", "method": "deck",
        "levels": {"+1st": {"semitones": 1.0}, "-1st": {"semitones": -1.0}},
        "expect": ["critic:chroma_dist", "v4:chroma_dist"],
        "basis": "[P] B&B 2003 cap. 15 (lo scontro si sente quando i brani suonano insieme; nel cut no); Webber 2008 p. 238. "
                 "Operatore: deck B trasposto di ±1 semitono (Rubber Band R3, durata invariata)",
        "applies": "solo riferimenti con sovrapposizione (B beatmatchato sotto A)",
    },
    "echo_offtempo": {
        "label": "coda d'eco fuori tempo", "family": "effetti", "method": "deck",
        "levels": {"0.37beat": {"delay_beats": 0.37}},
        "expect": ["qa:echo_tempo"],
        "basis": "[P] Webber 2008 (eco a tempo, 1 battito o frazioni musicali) [H] 0,37 battiti = ritardo non musicale. "
                 "Operatore: coda dell'ultimo battito di A (passa-alto 300 Hz) con ritardo 0,37 battiti e feedback 0,5 aggiunta "
                 "dopo lo switch, non dichiarata (dj_effects.echo_out accetta solo 0,5 o 1 battito: il difetto è un errore di "
                 "render, non una scelta pianificabile)",
        "applies": "tutti",
    },
}

# Combinazioni di due difetti (operatori applicati in sequenza: al più un rerender, poi i deck).
COMBOS: list[tuple[tuple[str, str], tuple[str, str]]] = [
    (("phase", "quarter_beat"), ("loud_jump", "+3dB")),
    (("word_cut_a", "straddle120"), ("gap", "half_beat")),
    (("b_bar_late", "1bar"), ("echo_offtempo", "0.37beat")),
    (("phase", "30ms"), ("click", "-20dBFS")),
    (("double_bass", "2bars"), ("key_clash", "+1st")),
]


def variant_id(parts: Sequence[tuple[str, str]]) -> str:
    return "+".join(f"{d}@{lv}" for d, lv in parts)


def variant_plan(ref: Mapping[str, Any], *, defects: Optional[Iterable[str]] = None, combos: bool = True,
                 levels: Optional[Mapping[str, Sequence[str]]] = None) -> list[list[tuple[str, str]]]:
    """Varianti da generare per un riferimento: [(difetto, livello), ...] per variante (singoli + combinazioni).

    L'applicabilità vera (es. sovrapposizione, lead-in dichiarato) si decide al momento della generazione: qui solo
    il filtro per nome e livello."""
    want = set(DEFECTS) if defects is None else set(defects)
    out = []
    for d, spec in DEFECTS.items():
        if d not in want:
            continue
        for lv in spec["levels"]:
            if levels and d in levels and lv not in levels[d]:
                continue
            out.append([(d, lv)])
    if combos:
        for c in COMBOS:
            if all(p[0] in want for p in c):
                out.append([c[0], c[1]])
    return out


# --------------------------------------------------------------------------------------------- bundle
@dataclass
class Bundle:
    """Tutto ciò che QA e critico leggono di una clip, con i deck che la compongono.

    ``components``: parti del mix prima del limitatore (``a``, ``echo``, ``b``, ``overlay``...), coordinate della clip;
    ``qa_decks`` / ``critic_decks``: deck passati a ``dj_qa.evaluate_render`` e al critico v4 (possono condividere gli
    array con ``components``); ``vocals_heard``: le voci sono quelle udibili (dj_creative) e seguono i guadagni.
    """

    audio: np.ndarray
    sr: int
    metadata: dict
    components: dict
    qa_decks: Optional[dict] = None
    critic_decks: Optional[dict] = None
    vocals_a: Optional[np.ndarray] = None
    vocals_b: Optional[np.ndarray] = None
    vocal_thresholds: tuple = (None, None)
    vocals_heard: bool = False
    words_a: Optional[list] = None
    words_b: Optional[list] = None
    grid_support: Optional[dict] = None
    kind: str = "unknown"
    declared_gap: bool = False
    a_source: Optional[Callable[[int, int], np.ndarray]] = None  # (s0, s1) clip -> A sorgente con il guadagno di A
    beat_s: Optional[float] = None
    info: dict = field(default_factory=dict)

    def copy(self) -> "Bundle":
        """Copia profonda degli array (le varianti non toccano mai il riferimento)."""
        memo: dict[int, Any] = {}

        def cp(x):
            if x is None:
                return None
            if id(x) in memo:
                return memo[id(x)]
            y = np.array(x, copy=True) if isinstance(x, np.ndarray) else copy.deepcopy(x)
            memo[id(x)] = y
            return y

        def cpd(d):
            if d is None:
                return None
            return {k: (cp(v) if isinstance(v, np.ndarray) else ({kk: cp(vv) if isinstance(vv, np.ndarray) else copy.deepcopy(vv)
                                                                 for kk, vv in v.items()} if isinstance(v, dict) else copy.deepcopy(v)))
                    for k, v in d.items()}

        comps = cpd(self.components)
        return Bundle(audio=np.array(self.audio, copy=True), sr=self.sr, metadata=copy.deepcopy(self.metadata), components=comps,
                      qa_decks=cpd(self.qa_decks), critic_decks=cpd(self.critic_decks), vocals_a=cp(self.vocals_a),
                      vocals_b=cp(self.vocals_b), vocal_thresholds=tuple(self.vocal_thresholds), vocals_heard=self.vocals_heard,
                      words_a=copy.deepcopy(self.words_a), words_b=copy.deepcopy(self.words_b),
                      grid_support=copy.deepcopy(self.grid_support), kind=self.kind, declared_gap=self.declared_gap,
                      a_source=self.a_source, beat_s=self.beat_s, info=copy.deepcopy(self.info))

    @property
    def n(self) -> int:
        return int(len(self.audio))

    @property
    def switch(self) -> int:
        return int((self.metadata.get("switch") or {}).get("output_sample", self.n // 2))


def _stereo(x: np.ndarray, n: Optional[int] = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    if n is not None:
        if len(x) >= n:
            x = x[:n]
        else:
            x = np.concatenate([x, np.zeros((n - len(x), x.shape[1]))])
    return x


def mix_of(components: Mapping[str, Any], n: int) -> np.ndarray:
    parts = [_stereo(components[k], n) for k in MIX_COMPONENTS if components.get(k) is not None]
    return np.sum(parts, axis=0) if parts else np.zeros((n, 2))


def rebuild(bundle: Bundle) -> Bundle:
    """Mix = somma dei componenti, poi il limitatore true-peak del renderer (stessi parametri dei metadati)."""
    from musicmix import dj_render as dj

    cfg = bundle.metadata.get("config") or {}
    lim, _ = dj.true_peak_limit(mix_of(bundle.components, bundle.n), bundle.sr, float(cfg.get("ceiling_dbtp", -1.0)),
                                float(cfg.get("limiter_hold_ms", 20.0)))
    bundle.audio = np.asarray(lim, dtype=np.float64)
    return bundle


def rebuild_error(bundle: Bundle) -> float:
    """Errore massimo tra la clip e il mix ricostruito dai componenti (0 = i deck descrivono esattamente la clip)."""
    b2 = Bundle(audio=bundle.audio, sr=bundle.sr, metadata=bundle.metadata, components=bundle.components)
    ref = np.array(bundle.audio, copy=True)
    rebuild(b2)
    return float(np.max(np.abs(np.asarray(b2.audio, dtype=np.float32) - np.asarray(ref, dtype=np.float32)))) if len(ref) else 0.0


def _decks_with(bundle: Bundle, key: str) -> list[dict]:
    seen, out = set(), []
    for d in (bundle.components, bundle.qa_decks, bundle.critic_decks):
        if d is not None and d.get(key) is not None and id(d) not in seen:
            seen.add(id(d))
            out.append(d)
    return out


def _apply(bundle: Bundle, key: str, fn: Callable[[np.ndarray], np.ndarray]) -> bool:
    """Applica ``fn`` all'array ``key`` in componenti e deck di QA/critico (una volta per array condiviso)."""
    done: dict[int, np.ndarray] = {}
    hit = False
    for d in _decks_with(bundle, key):
        arr = d[key]
        if id(arr) not in done:
            done[id(arr)] = fn(_stereo(arr, bundle.n) if arr.ndim == 2 or arr.ndim == 1 else arr)
        d[key] = done[id(arr)]
        hit = True
    return hit


def _ramp_mask(n: int, s0: int, s1: int, sr: int, ramp_s: float = DECK_RAMP_S) -> np.ndarray:
    """1 in [s0, s1) con rampe di ``ramp_s`` ai bordi, 0 altrove."""
    m = np.zeros(n)
    s0, s1 = max(0, int(s0)), min(n, int(s1))
    if s1 <= s0:
        return m
    m[s0:s1] = 1.0
    r = max(1, int(ramp_s * sr))
    up = np.linspace(0.0, 1.0, r)
    m[s0:min(s1, s0 + r)] *= up[:min(r, s1 - s0)]
    m[max(s0, s1 - r):s1] *= up[::-1][-(s1 - max(s0, s1 - r)):]
    return m


def _beat_samples(bundle: Bundle) -> int:
    if bundle.beat_s:
        return int(round(bundle.beat_s * bundle.sr))
    st = bundle.metadata.get("stretch") or {}
    if st.get("a_bar_s"):
        return int(round(float(st["a_bar_s"]) / 4.0 * bundle.sr))
    return int(round(0.5 * bundle.sr))


# --------------------------------------------------------------------------------------------- deck operators
def op_loud_jump(bundle: Bundle, db: float) -> dict:
    """Deck B (e la sua voce udibile) +``db`` dB; il guadagno di B dichiarato dal renderer segue."""
    g = 10.0 ** (db / 20.0)
    if not _apply(bundle, "b", lambda x: x * g):
        return {"applied": False, "reason": "deck B assente"}
    if bundle.vocals_heard and bundle.vocals_b is not None:
        bundle.vocals_b = np.asarray(bundle.vocals_b) * g
        ta, tb = bundle.vocal_thresholds
        bundle.vocal_thresholds = (ta, tb * g if tb is not None else None)
    loud = bundle.metadata.setdefault("loudness", {})
    if loud.get("b_gain_db") is not None:
        loud["b_gain_db"] = float(loud["b_gain_db"]) + db
    return {"applied": True, "db": db, "region": "intero deck B", "declared": "loudness.b_gain_db"}


def op_gap(bundle: Bundle, beats: float) -> dict:
    """Silenzio non dichiarato di ``beats`` battiti dallo switch, su tutti i deck."""
    L = int(round(beats * _beat_samples(bundle)))
    s0 = bundle.switch
    keep = 1.0 - _ramp_mask(bundle.n, s0, s0 + L, bundle.sr)
    for k in MIX_COMPONENTS + ("a_raw", "b_stretched_raw", "vocals_a_heard", "vocals_b_heard"):
        _apply(bundle, k, lambda x: x * keep[:, None])
    if bundle.vocals_heard:
        for side in ("vocals_a", "vocals_b"):
            v = getattr(bundle, side)
            if v is not None:
                setattr(bundle, side, _stereo(v, bundle.n) * keep[:, None])
    return {"applied": True, "beats": beats, "samples": L, "start_sample": s0, "declared": False}


def op_click(bundle: Bundle, amp: float, width_s: float = 0.001) -> dict:
    """Impulso rettangolare (``amp`` di fondo scala, 1 ms) sullo switch, aggiunto al mix prima del limitatore."""
    n, s0 = bundle.n, bundle.switch
    w = max(1, int(width_s * bundle.sr))
    c = np.zeros((n, 2))
    c[s0:s0 + w] = amp
    prev = bundle.components.get("echo_gym")
    bundle.components["echo_gym"] = c if prev is None else _stereo(prev, n) + c
    return {"applied": True, "amp": amp, "dbfs": 20 * math.log10(amp), "width_ms": 1000 * w / bundle.sr, "at_sample": s0, "declared": False}


def lowpass(x: np.ndarray, sr: int, hz: float = 200.0) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, hz, btype="low", fs=sr, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=0)


def highpass(x: np.ndarray, sr: int, hz: float = 300.0) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    sos = butter(2, hz, btype="high", fs=sr, output="sos")
    return sosfilt(sos, np.asarray(x, dtype=np.float64), axis=0)


def op_double_bass(bundle: Bundle, bars: float = 2.0) -> dict:
    """Banda bassa di A (sorgente, guadagno di A) lasciata aperta ``bars`` battute dopo lo switch, sul deck A."""
    if bundle.a_source is None:
        return {"applied": False, "reason": "sorgente di A non disponibile"}
    n, s0 = bundle.n, bundle.switch
    L = int(round(bars * 4 * _beat_samples(bundle)))
    s1 = min(n, s0 + L)
    if s1 - s0 < bundle.sr // 2:
        return {"applied": False, "reason": "clip troppo corta dopo lo switch"}
    pad = int(0.25 * bundle.sr)
    src = _stereo(bundle.a_source(max(0, s0 - pad), s1 + pad))
    low = lowpass(src, bundle.sr)[s0 - max(0, s0 - pad):][: s1 - s0]
    mask = _ramp_mask(n, s0, s1, bundle.sr, ramp_s=0.02)[s0:s1, None]
    add = np.zeros((n, 2))
    add[s0:s1] = low[: s1 - s0] * mask[: len(low)]
    # Dose (v2, 22/10/2026): A's source may have almost no bass after its cue (a breakdown, the song's end): the variant
    # would then not be a double bass at all (palestra v1: 3 "mancate" with A's injected band 16-45 dB under B's).
    # Not generated when the injected band is more than DOUBLE_BASS_MIN_DOSE_DB under B's low band in the same bars [H].
    b = None
    for d in (bundle.qa_decks, bundle.components):
        if d is not None and d.get("b") is not None:
            b = _stereo(d["b"], n)
            break
    if b is not None:
        lb = lowpass(b[s0:s1], bundle.sr)
        ra, rb = float(np.sqrt(np.mean(add[s0:s1] ** 2))), float(np.sqrt(np.mean(lb ** 2)))
        dose = 20 * math.log10(max(ra, 1e-12) / max(rb, 1e-12))
        if dose < DOUBLE_BASS_MIN_DOSE_DB:
            return {"applied": False, "reason": f"dose: A's low band after its cue {dose:.0f} dB under B's (< {DOUBLE_BASS_MIN_DOSE_DB:g} dB): no double bass to hear"}
    else:
        dose = None
    if not _apply(bundle, "a", lambda x: x + add):
        return {"applied": False, "reason": "deck A assente"}
    return {"applied": True, "bars": bars, "start_sample": s0, "end_sample": s1, "cutoff_hz": 200.0, "declared": False, "dose_db_vs_b": dose}


def pitch_shift(x: np.ndarray, sr: int, semitones: float, workdir: Optional[str] = None) -> np.ndarray:
    """Trasposizione a durata invariata con il CLI Rubber Band (motore R3, ``-3``), file temporanei cancellati."""
    import soundfile as sf

    exe = shutil.which("rubberband")
    if not exe:
        raise RuntimeError("rubberband CLI non disponibile")
    tmp = tempfile.mkdtemp(prefix="gym_ps_", dir=workdir)
    try:
        src, dst = os.path.join(tmp, "in.wav"), os.path.join(tmp, "out.wav")
        x2 = _stereo(x)
        peak = float(np.max(np.abs(x2))) or 1.0
        sf.write(src, (x2 / peak).astype(np.float32), sr, subtype="FLOAT")
        subprocess.run([exe, "-3", "-q", "--pitch", f"{semitones:g}", src, dst], check=True, capture_output=True)
        y, _ = sf.read(dst, dtype="float64", always_2d=True)
        return _stereo(y * peak, len(x2))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def has_overlap(bundle: Bundle) -> bool:
    kind = str(bundle.metadata.get("kind") or bundle.kind)
    if bundle.kind in OVERLAP_KINDS or kind in OVERLAP_KINDS:
        return True
    return bool((bundle.metadata.get("stretch") or {}).get("applied")) and str(kind) not in ("cut",)


def op_key_clash(bundle: Bundle, semitones: float, shifter: Callable[[np.ndarray, int, float], np.ndarray] = pitch_shift) -> dict:
    """Deck B trasposto di ``semitones`` (solo con sovrapposizione: nel cut un cambio di tonalità è ammesso, B&B cap. 15)."""
    if not has_overlap(bundle):
        return {"applied": False, "reason": "nessuna sovrapposizione: un cut tra tonalità diverse non è un difetto (B&B cap. 15)"}
    if not _apply(bundle, "b", lambda x: shifter(x, bundle.sr, semitones)):
        return {"applied": False, "reason": "deck B assente"}
    return {"applied": True, "semitones": semitones, "declared": False}


def op_echo_offtempo(bundle: Bundle, delay_beats: float = 0.37, feedback: float = 0.5, repeats: int = 6,
                     level: float = 0.5) -> dict:
    """Coda dell'ultimo battito di A (deck secco) con ritardo fuori griglia, dopo lo switch, non dichiarata."""
    n, s0 = bundle.n, bundle.switch
    beat = _beat_samples(bundle)
    a = bundle.components.get("a")
    if a is None:
        return {"applied": False, "reason": "deck A assente"}
    a2 = _stereo(a, n)
    seg, back = None, 0
    for back in range(0, 8):  # ultimo battito udibile di A prima dello switch (build e break possono tacere prima)
        e = s0 - back * beat
        cand = a2[max(0, e - beat):e]
        if len(cand) and float(np.sqrt(np.mean(cand ** 2))) > 1e-3:
            seg = cand
            break
    if seg is None:
        return {"applied": False, "reason": "A muto negli 8 battiti prima dello switch"}
    seg = highpass(seg, bundle.sr) * np.hanning(len(seg))[:, None]
    d = int(round(delay_beats * beat))
    tail = np.zeros((n, 2))
    for k in range(1, repeats + 1):
        at = s0 + (k - 1) * d
        if at >= n:
            break
        m = min(len(seg), n - at)
        tail[at:at + m] += level * feedback ** (k - 1) * seg[:m]
    prev = bundle.components.get("echo_gym")
    bundle.components["echo_gym"] = tail if prev is None else _stereo(prev, n) + tail
    return {"applied": True, "delay_beats": delay_beats, "feedback": feedback, "repeats": repeats, "level": level,
            "source_beats_before_switch": back + 1, "declared": False}


DECK_OPS: dict[str, Callable[..., dict]] = {
    "double_bass": lambda b, p: op_double_bass(b, p["bars"]),
    "loud_jump": lambda b, p: op_loud_jump(b, p["db"]),
    "gap": lambda b, p: op_gap(b, p["beats"]),
    "click": lambda b, p: op_click(b, p["amp"]),
    "key_clash": lambda b, p: op_key_clash(b, p["semitones"]),
    "echo_offtempo": lambda b, p: op_echo_offtempo(b, p["delay_beats"]),
}


def apply_deck_defect(bundle: Bundle, defect: str, level: str) -> dict:
    """Applica un difetto di tipo ``deck`` (in place) e ricostruisce il mix; restituisce la descrizione dell'operazione."""
    params = DEFECTS[defect]["levels"][level]
    info = DECK_OPS[defect](bundle, params)
    if info.get("applied"):
        rebuild(bundle)
    return info


# --------------------------------------------------------------------------------------------- word straddles
def find_straddle_shift(words: Sequence[Mapping[str, Any]], downbeats_s: Sequence[float], anchor_index: int, point_s: float,
                        *, min_side_s: float = WORD_STRADDLE_MIN_S, max_bars: int = 32, sr: int = 48000,
                        valid: Optional[Callable[[int], bool]] = None) -> Optional[dict]:
    """Minimo |k| (k != 0) tale che, spostando il cue di k downbeat, una parola attraversi il punto di taglio.

    ``point_s``: il punto di taglio nella sorgente con il cue originale (``downbeats_s[anchor_index]`` + scarto); con il
    cue spostato il punto diventa ``downbeats_s[anchor_index + k]`` + lo stesso scarto (griglia invariata). Le parole
    hanno ``start_sample``/``end_sample`` (campioni della sorgente) o ``start_s``/``end_s``.
    """
    db = np.asarray(downbeats_s, dtype=float)
    off = float(point_s) - float(db[anchor_index])
    ws = []
    for w in words or []:
        if "start_sample" in w:
            s, e = w["start_sample"] / sr, w["end_sample"] / sr
        else:
            s, e = float(w["start_s"]), float(w["end_s"])
        ws.append((s, e, w.get("text")))
    for dist in range(1, max_bars + 1):
        for k in (-dist, dist):
            j = anchor_index + k
            if j < 0 or j >= len(db) or (valid is not None and not valid(k)):
                continue
            p = float(db[j]) + off
            for s, e, txt in ws:
                if s <= p - min_side_s and e >= p + min_side_s:
                    return {"k": k, "point_s": p, "word": txt, "word_start_s": s, "word_end_s": e,
                            "before_s": p - s, "after_s": e - p}
    return None


# --------------------------------------------------------------------------------------------- signals
def qa_signals(qa: Optional[Mapping[str, Any]]) -> dict[str, str]:
    """Segnali del QA: ``qa:<controllo>`` -> ``fail``/``warn`` (controlli con quello stato, più ``coverage``)."""
    out: dict[str, str] = {}
    if not qa:
        return out
    for name, c in (qa.get("checks") or {}).items():
        st = (c or {}).get("status")
        if st in ("fail", "warn"):
            out[f"qa:{name}"] = st
    for r in qa.get("reasons") or []:
        if r.get("check") == "coverage":
            out["qa:coverage"] = r.get("severity", "warn")
    return out


def qa_coverage(qa: Optional[Mapping[str, Any]]) -> dict[str, str]:
    cov = ((qa or {}).get("coverage") or {}).get("checks") or {}
    return {f"qa:{k}": v for k, v in cov.items()}


def critic_signals(crt: Optional[Mapping[str, Any]], veto_flags: Sequence[str] = ()) -> dict[str, str]:
    """Segnali del critico: ``critic:<flag>`` = ``fail`` se veto del runner, ``warn`` se solo flag; ``v4:...``."""
    out: dict[str, str] = {}
    if not crt:
        return out
    for f in crt.get("flags") or []:
        out[f"critic:{f['flag']}"] = "fail" if f["flag"] in veto_flags else "warn"
    t = ((crt.get("v4") or {}).get("technical")) or {}
    for f in t.get("veto_flags") or []:
        out[f"v4:{f['flag']}"] = "fail"
    for w in t.get("warnings") or []:
        key = f"v4:{w.get('check') or w.get('flag')}"
        out.setdefault(key, "warn")
    return out


def runner_veto_flags(crt: Mapping[str, Any], kind: str, declared_gap: bool, *, phase_declared_max_ms: float = 5.0) -> list[str]:
    """Regole di veto del runner v3 (``tools/run_dj_set_v3_bridge.critic``), riscritte per avere i nomi dei flag."""
    dp = (crt.get("declared_phase") or {}).get("ms")
    out = []
    for f in crt.get("flags") or []:
        fl = f["flag"]
        if fl == "beat_phase_ms":
            if dp is not None and dp < phase_declared_max_ms:
                continue
            out.append(fl)
        elif fl == "loud_jump_db":
            if not declared_gap:
                out.append(fl)
        elif fl == "chroma_dist" and kind in OVERLAP_KINDS:
            out.append(fl)
    return out


DETECTORS = {"qa": ("qa:",), "critic": ("critic:",), "v4": ("v4:",), "pipeline": ("qa:", "critic:")}


def classify(var_sig: Mapping[str, str], ref_sig: Mapping[str, str], expected: Sequence[str],
             var_cov: Optional[Mapping[str, str]] = None, error: Optional[str] = None) -> dict[str, Any]:
    """Esito appaiato di una variante rispetto al suo riferimento.

    ``new_fail`` = segnali di rifiuto (``fail``) assenti come ``fail`` nel riferimento; ``new_warn`` = segnali nuovi a
    livello ``warn`` (o peggiorati). Stato: ``detected`` (rifiuto nuovo della pipeline: QA fail o veto del critico),
    ``warned`` (solo avvisi nuovi), ``unavailable`` (nessun segnale nuovo e un controllo atteso non misurato o
    inesistente), ``missed``, ``error`` (valutazione fallita: mai un pass)."""
    if error:
        return {"status": "error", "error": error, "detected": {k: False for k in DETECTORS}, "warned": {k: False for k in DETECTORS},
                "new_fail": [], "new_warn": [], "specific": False}
    new_fail = sorted(k for k, v in var_sig.items() if v == "fail" and ref_sig.get(k) != "fail")
    new_warn = sorted(k for k, v in var_sig.items() if v == "warn" and k not in ref_sig)
    det = {name: any(k.startswith(p) for k in new_fail for p in pref) for name, pref in DETECTORS.items()}
    warned = {name: det[name] or any(k.startswith(p) for k in new_warn for p in pref) for name, pref in DETECTORS.items()}
    exp = set(expected)
    specific = any(k in exp for k in new_fail)
    if det["pipeline"]:
        status = "detected"
    elif warned["pipeline"] or warned["v4"]:
        status = "warned"
    else:
        cov = var_cov or {}
        missing = [e for e in expected if e.startswith("qa:") and cov.get(e, "absent") in ("unavailable", "not_applicable", "absent")]
        status = "unavailable" if (missing or not expected) else "missed"
    return {"status": status, "detected": det, "warned": warned, "new_fail": new_fail, "new_warn": new_warn, "specific": specific}


def reference_flags(ref_sig: Mapping[str, str]) -> dict[str, bool]:
    """Falso allarme sul riferimento, per rivelatore (un ``fail`` qualunque)."""
    return {name: any(v == "fail" and k.startswith(p) for k, v in ref_sig.items() for p in pref) for name, pref in DETECTORS.items()}


# --------------------------------------------------------------------------------------------- aggregation
def _rate(k: int, n: int) -> Optional[float]:
    return None if n == 0 else k / n


def wilson(k: int, n: int, z: float = 1.96) -> Optional[list[float]]:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def aggregate(rows: Sequence[Mapping[str, Any]], refs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Tabelle per difetto/livello e falsi allarmi per origine del riferimento."""
    by: dict[str, dict] = {}
    for r in rows:
        if r.get("status") == "not_generated":
            key = r["variant"]
            e = by.setdefault(key, _empty(r))
            e["not_generated"] += 1
            e["not_generated_reasons"].setdefault(r.get("reason", "?")[:80], 0)
            e["not_generated_reasons"][r.get("reason", "?")[:80]] += 1
            continue
        key = r["variant"]
        e = by.setdefault(key, _empty(r))
        e["n"] += 1
        c = r["classification"]
        e[c["status"]] += 1
        for d in DETECTORS:
            e["detected_by"][d] += int(bool(c["detected"].get(d)))
            e["warned_by"][d] += int(bool(c["warned"].get(d)))
        e["specific"] += int(bool(c.get("specific")))
        if c["status"] in ("missed", "unavailable", "warned", "error") and len(e["examples_missed"]) < 4:
            e["examples_missed"].append({"ref": r["ref"], "status": c["status"], "new_warn": c.get("new_warn"),
                                         "note": (r.get("op") or {}).get("note")})
        for s in c.get("new_fail") or []:
            e["new_fail_counts"][s] = e["new_fail_counts"].get(s, 0) + 1
    for e in by.values():
        n = e["n"]
        e["detection_rate"] = _rate(e["detected"], n)
        e["detection_ci95"] = wilson(e["detected"], n)
        e["rates_by_detector"] = {d: _rate(v, n) for d, v in e["detected_by"].items()}
        e["warn_or_fail_rate"] = _rate(e["detected"] + e["warned"], n)
        e["unavailable_rate"] = _rate(e["unavailable"], n)
    fa: dict[str, dict] = {}
    for ref in refs:
        if ref.get("evaluation_error"):
            continue
        for grp in ("all", ref.get("origin", "?")) + (("clean",) if ref.get("clean") else ()):
            f = fa.setdefault(grp, {"n": 0, **{d: 0 for d in DETECTORS}, "flagged": []})
            f["n"] += 1
            flags = ref.get("reference_flags") or {}
            for d in DETECTORS:
                f[d] += int(bool(flags.get(d)))
            if flags.get("pipeline"):
                f["flagged"].append({"ref": ref["id"], "fail_signals": [k for k, v in (ref.get("signals") or {}).items() if v == "fail"]})
    for f in fa.values():
        f["rates"] = {d: _rate(f[d], f["n"]) for d in DETECTORS}
        f["pipeline_ci95"] = wilson(f["pipeline"], f["n"])
    return {"variants": dict(sorted(by.items())), "false_alarms": fa}


def _empty(r: Mapping[str, Any]) -> dict:
    return {"defects": r.get("defects"), "n": 0, "detected": 0, "warned": 0, "unavailable": 0, "missed": 0, "error": 0,
            "specific": 0, "not_generated": 0, "not_generated_reasons": {}, "detected_by": {d: 0 for d in DETECTORS},
            "warned_by": {d: 0 for d in DETECTORS}, "new_fail_counts": {}, "examples_missed": []}


def check_gate(summary: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    """Confronta il riassunto con le soglie dichiarate (``configs/defect_gym_gate.json``)."""
    results, ok = [], True
    for rule in gate.get("detection", []):
        e = (summary.get("variants") or {}).get(rule["variant"])
        det = rule.get("detector", "pipeline")
        if not e or e["n"] < int(rule.get("min_n", 1)):
            res = {"rule": rule, "status": "insufficient", "n": 0 if not e else e["n"],
                   "message": "varianti valutate meno del minimo dichiarato: il gate non è soddisfatto (non è un pass)"}
            ok = False
        else:
            rate = e["rates_by_detector"][det]
            passed = rate is not None and rate >= float(rule["min_rate"])
            ok &= passed
            res = {"rule": rule, "status": "pass" if passed else "fail", "n": e["n"], "rate": rate, "ci95": e["detection_ci95"]}
        results.append(res)
    for rule in gate.get("false_alarms", []):
        f = (summary.get("false_alarms") or {}).get(rule["group"])
        det = rule.get("detector", "pipeline")
        if not f or f["n"] < int(rule.get("min_n", 1)):
            res = {"rule": rule, "status": "insufficient", "n": 0 if not f else f["n"]}
            ok = False
        else:
            rate = f["rates"][det]
            passed = rate is not None and rate <= float(rule["max_rate"])
            ok &= passed
            res = {"rule": rule, "status": "pass" if passed else "fail", "n": f["n"], "rate": rate}
        results.append(res)
    return {"passed": bool(ok), "rules": results}


def load_gate(path: Optional[os.PathLike | str] = None) -> dict:
    p = Path(path) if path else Path(__file__).resolve().parents[2] / "configs" / "defect_gym_gate.json"
    return json.loads(p.read_text(encoding="utf-8"))


def jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o
