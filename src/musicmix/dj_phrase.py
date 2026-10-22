"""Modello esplicito di versi/barre rap per decidire QUANDO tagliare (fine verso) e da dove far entrare B (pickup).

Problema (misurato sul pool, 20/10/2026): la guardia vocale di ``dj_select``
guarda solo se lo stem vocale è acceso nell'ultimo battito prima del downbeat;
nel rap la voce è quasi sempre accesa, quindi quasi nessun cue risulta pulito,
e un ingresso di B *sul* downbeat tronca il pickup del suo primo verso.

Questo modulo costruisce per ogni brano, con canali separati e auditabili:

(a) **parole** con timestamp ASR (faster-whisper, word timestamps), raffinate
    sull'attività *non chiusa* dello stem vocale (RMS sopra soglia, frame
    10 ms di :mod:`musicmix.vocal_completion`); le parole senza supporto
    acustico (allucinazioni ASR) sono scartate e contate;
(b) **sillabe e rime di fine parola**: CMUdict (inglese, fonetico: dalla
    vocale accentata all'ultima consonante) oppure approssimazione ortografica
    (altre lingue o parole fuori dizionario: gruppo vocalico accentato per
    regola di lingua + coda; documentata come [H]);
(c) **prosodia** dallo stem: pausa acustica dopo la parola, allungamento
    finale (durata dell'ultima sillaba / mediana delle sillabe del brano),
    f0 YIN (abbassamento finale in semitoni, solo diagnostico), energia;
(d) **allineamento alla griglia**: battito della barra su cui cade l'ultima
    sillaba della parola, pausa espressa in sedicesimi del brano;
(e) **pickup**: sillabe del verso di B che iniziano prima del downbeat
    d'ingresso (fino a 2 battiti), in campioni.

Probabilità di fine verso (confine di frase prosodica, MCFlow ``**break`` >= 3)
per ogni confine tra parole: naive Bayes con tabelle **stimate sul corpus
MCFlow** (Condit-Schultz 2017; 124 brani, 54 586 confini di parola,
``configs/dj_phrase_mcflow_v1.json``, ``tools/mcflow_phrase_stats.py``):

* P(confine | battito dell'ultima sillaba, pausa >= 1/16) [E corpus simbolico];
* rapporto di verosimiglianza della rima (accentata) condizionato alla pausa [E];
* allungamento finale, punteggiatura (ASR) [E];
* i canali misurati dall'audio (rima, allungamento, punteggiatura) entrano con
  esponente ``audio_lr_exponent`` = 0,5 [H]: l'estrazione dall'audio è rumorosa
  e le tabelle vengono da trascrizioni manuali. Su MCFlow, a metà dei brani
  (test per brano) l'AUC sale da 0,85 (posizione+pausa) a 0,94 (+rima,
  +allungamento): misura sul corpus simbolico, **non** sull'audio del pool.

Il numero è una probabilità di confine *prosodico* nel senso di MCFlow, non una
"chiusura" né un giudizio su dove un DJ debba tagliare (AGENTS.md: fine frase
non è pausa ASR, chiusura non è volume). La pausa da sola non basta: su MCFlow
il 19 % delle pause tra parole non è fine frase e il 41 % delle fini frase non
ha pausa.

Uscite e ingressi (tutti in campioni canonici 48 kHz di ciascun brano):

* :func:`exit_point` per un downbeat D di A: ultimo fine verso raggiungibile,
  finestra di switch in cui A è vocalmente libero (eventuale pickup del verso
  successivo di A da non troncare, eventuale coda dell'ultima parola oltre D da
  proteggere con lo stem vocale), parole troncate da un taglio secco in D;
* :func:`entry_point` per un downbeat D di B: inizio del verso (pickup), finestra
  in cui B può diventare udibile senza troncare la sua parola precedente;
* :func:`phrase_switch` combina le due finestre sulla linea comune in cui il
  downbeat di B cade sul downbeat di A: lo switch del fader avviene al punto
  più tardo ammissibile (D se possibile, altrimenti sull'ottavo precedente),
  B entra dal pickup e il suo downbeat resta sul downbeat di A.

:func:`render_phrase_transition` rimescola i deck di ``dj_render`` (cut) o
``dj_effects.filter_out`` con lo switch spostato e la coda vocale di A, senza
modificare quei moduli.

Ogni soglia è in ``PARAMS`` con la sua origine ([E] corpus/paper, [H] ipotesi).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from .paths import models_dir

SCHEMA_VERSION = "musicmix.dj_phrase.1"
SR = 48000
ROOT = Path(__file__).resolve().parents[2]
MCFLOW_CONFIG = ROOT / "configs" / "dj_phrase_mcflow_v1.json"
CMUDICT_PATH = Path(os.environ.get("MUSICMIX_CMUDICT", str(models_dir() / "cmudict/cmudict.dict")))


@dataclass(frozen=True)
class Param:
    value: Any
    source: str


PARAMS: dict[str, Param] = {
    "line_end_threshold": Param(0.5, "[H] decisione MAP: confine di verso se p >= 0,5"),
    "audio_lr_exponent": Param(0.5, "[H] rima/allungamento/punteggiatura misurati dall'audio: LR del corpus ridotti (esponente 0,5)"),
    "rest_min_16th": Param(0.75, "[H] silenzio acustico >= 3/4 di sedicesimo = pausa notata MCFlow (gap della VAD chiusi a 120 ms)"),
    "support_min": Param(0.25, "[H] parola ASR tenuta solo se >= 25 % dei frame dello stem è sopra soglia (anti-allucinazione)"),
    "local_rel_db": Param(12.0, "[H] stem attivo se entro 12 dB dal 90° percentile locale: sul pool V12 dà 11-24 % di confini tra parole con pausa >= 3/4 di sedicesimo, contro il 10,4 % di pause notate in MCFlow (calibrazione sulla distribuzione del corpus, non sui giudizi)"),
    "strict_rel_db": Param(20.0, "[H] controllo dello switch più severo: nessun frame entro 20 dB dal 90° percentile locale a +-40 ms dallo switch (parole sussurrate o deboli)"),
    "collapsed_word_s": Param(0.04, "[H] parola ASR ridotta a meno di 40 ms dalla raffinatura: torna all'estensione Whisper ritagliata tra le vicine (incerta, protetta)"),
    "local_window_s": Param(2.0, "[H] finestra del percentile locale"),
    "refine_max_extend_s": Param(0.40, "[H] la fine di una parola si estende sull'attività dello stem al massimo di 400 ms"),
    "word_margin_s": Param(0.03, "[H] tolleranza di troncamento: una parola è troncata se perde >= 30 ms dopo averne suonati >= 30"),
    "switch_margin_s": Param(0.04, "[H] silenzio vocale richiesto attorno allo switch (2 x micro-fade 10 ms + jitter ASR)"),
    "tail_max_beats": Param(0.5, "[H] coda dell'ultima parola di A oltre il downbeat protetta fino a mezzo battito (stem vocale)"),
    "tail_fade_s": Param(0.03, "[H] dissolvenza della coda vocale di A"),
    "pickup_max_beats": Param(2.0, "[E] MCFlow: 42,7 % delle frasi inizia <= 1 battito prima di un downbeat, 13,3 % tra 1 e 2"),
    "pickup_line_p": Param(0.5, "[H] parole prima del downbeat appartengono al verso che attraversa il downbeat se il confine tra loro ha p < 0,5"),
    "rhyme_window_beats": Param(1.0, "[H] una rima si cerca 1 o 2 battute prima/dopo, entro +-1 battito di posizione (rime in posizioni parallele: Condit-Schultz 2017 p. 133; Adams 2009 par. 4)"),
    "rhyme_bar_lags": Param((1, 2), "[E] MCFlow/Adams: rime a fine battuta in distici; [H] ritardi 1 e 2 battute"),
    "switch_grid_div": Param(2, "[H] lo switch anticipato si allinea all'ottavo (mezzo battito) più tardo ammissibile"),
    "f0_in_probability": Param(False, "[H] abbassamento finale di f0 riportato ma escluso dalla probabilità (nessuna stima su corpus)"),
    "pillar_weight": Param(0.0, "[A] Ohriner 2019 p. 110 ('central pillar': ogni strofa ha una posizione preferita delle fini di frase); "
                                "[E] MCFlow held-out per brano (tools/mcflow_flow_stats.py): termine w*log(densità del brano / densità "
                                "del corpus) alla posizione della fine verso, AUC 0,918 -> 0,925 e F1 0,685 -> 0,697 con w = 1. "
                                "Default 0 = comportamento v1; runs/dj_phrase_v2 usa 1"),
    "pillar_min_ends": Param(8, "[H] il termine del pilastro si applica solo se il brano ha almeno 8 fini verso previste (densità stimabile)"),
    # Ingresso di B (21/10/2026, conoscenza_mix/_meta/INGRESSO_B_2026-10-21.md): b_lead_in / entry_clear_test.
    "b_rest_beats": Param(1.0, "[H] confine di frase per l'ingresso di B: pausa della voce di B >= 1 battito, oppure fine verso del "
                               "modello (p >= line_end_threshold) seguita da una pausa notata (>= rest_min_16th sedicesimi). Sui 10 brani "
                               "B del set v3 la regola dà confini nel 4-17 % dei passaggi tra parole, come la frequenza delle frasi di "
                               "MCFlow (~1 ogni 7 parole); con la sola pausa >= 1 battito erano lo 0,5-1 % (il rap è continuo): "
                               "calibrazione sulla distribuzione, non sui voti"),
    "b_lead_in_max_beats": Param(4.0, "[A] Krims 2000 pdf pp. 112-113 ('upbeat layer': una battuta che precede le articolazioni formali, "
                                      "BC 4 = pickup); [A] Ohriner 2019 pdf p. 83 (i pickup stanno nella battuta -1); [E] MCFlow 56 % delle "
                                      "frasi parte entro 2 battiti prima del downbeat; [H] al massimo una battuta di lead-in"),
}

# Function words (English) are kept out of the rhyme test: a rhyme on "the" is noise [H].
_EN_STOP = {"a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with", "i", "you", "he", "she",
            "it", "we", "they", "my", "your", "is", "be", "that", "this", "uh", "yeah", "oh", "like"}
_VOWELS = "aeiouyàáâãäåæèéêëìíîïòóôõöøùúûüýÿœаеёиоуыэюяії"


def params(overrides: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    out = {k: p.value for k, p in PARAMS.items()}
    for k, v in (overrides or {}).items():
        if k not in out:
            raise KeyError(f"unknown dj_phrase parameter {k!r}")
        out[k] = v
    return out


# --------------------------------------------------------------- MCFlow model
_MCFLOW_CACHE: dict[str, Any] = {}


def load_mcflow(path: Optional[os.PathLike | str] = None) -> dict[str, Any]:
    """Tables estimated on MCFlow by ``tools/mcflow_phrase_stats.py`` (tracked config)."""
    p = Path(path) if path else MCFLOW_CONFIG
    key = str(p)
    if key not in _MCFLOW_CACHE:
        _MCFLOW_CACHE[key] = json.loads(p.read_text(encoding="utf-8"))
    return _MCFLOW_CACHE[key]


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def gap_bin(g16: float) -> str:
    """Same bins as ``tools/mcflow_phrase_stats.gap_bin`` (sixteenths of silence)."""
    if g16 < 0.5:
        return "0"
    if g16 < 1.5:
        return "1"
    if g16 < 2.5:
        return "2"
    if g16 < 4.5:
        return "3-4"
    if g16 < 8.5:
        return "5-8"
    return ">8"


def boundary_probability(beat: int, rest: bool, gap16: float, rhyme: Optional[bool], lengthening: Optional[float],
                         punct: Optional[bool], *, mcflow: Optional[Mapping[str, Any]] = None,
                         alpha: float = PARAMS["audio_lr_exponent"].value) -> dict[str, Any]:
    """P(phrase boundary after a word) from MCFlow tables; returns p and the named terms.

    ``beat`` 1..4 = beat of the bar on which the word's last syllable starts;
    ``rest`` = acoustic silence after the word counts as a notated rest;
    ``gap16`` = that silence in sixteenths (>= 4.5 uses the long-gap row);
    ``rhyme``/``lengthening``/``punct`` may be ``None`` (unknown: no term).
    """
    m = mcflow or load_mcflow()
    base = m["base_rate"]
    terms: list[dict[str, Any]] = []
    rk = "rest" if rest else "none"
    if gap16 >= 4.5:
        row = m["gap16"][gap_bin(gap16)]
        lo = _logit(row["p"])
        terms.append({"name": "pausa lunga", "value": f"{gap16:.1f}/16", "p": row["p"], "n": row["n"], "source": "[E] MCFlow gap16"})
    else:
        row = m["pos_gap"][f"{beat}|{rk}"]
        lo = _logit(row["p"])
        terms.append({"name": "posizione+pausa", "value": f"battito {beat}, {'pausa' if rest else 'nessuna pausa'} ({gap16:.1f}/16)",
                      "p": row["p"], "n": row["n"], "source": "[E] MCFlow pos_gap"})
    if rhyme is not None:
        # table of the same detector run on MCFlow IPA when available (else the annotated rhymes)
        tab = m.get("rhyme_gap_detected") or m["rhyme_gap"]
        r = tab[f"{'rhyme' if rhyme else 'no'}|{rk}"]["p"]
        # LR of rhyme given the rest state = odds(rhyme, rest) / odds(rest)
        rest_p = sum(v["p"] * v["n"] for k, v in tab.items() if k.endswith("|" + rk)) / \
            sum(v["n"] for k, v in tab.items() if k.endswith("|" + rk))
        llr = alpha * (_logit(r) - _logit(rest_p))
        lo += llr
        terms.append({"name": "rima", "value": bool(rhyme), "log_lr": llr,
                      "source": f"[E] MCFlow {'rhyme_gap_detected (stesso rilevatore sull IPA del corpus)' if 'rhyme_gap_detected' in m else 'rhyme_gap'}, esponente {alpha} [H]"})
    if lengthening is not None and np.isfinite(lengthening):
        lb = "<=1" if lengthening <= 1.0 else "1-2" if lengthening <= 2.0 else ">2"
        llr = alpha * (_logit(m["lengthening"][lb]["p"]) - _logit(base))
        lo += llr
        terms.append({"name": "allungamento finale", "value": round(float(lengthening), 2), "bin": lb, "log_lr": llr,
                      "source": f"[E] MCFlow lengthening (Wightman 1992: allungamento nella rima della sillaba prima del confine), esponente {alpha} [H]"})
    if punct is not None:
        llr = alpha * (_logit(m["punct"]["yes" if punct else "no"]["p"]) - _logit(base))
        lo += llr
        terms.append({"name": "punteggiatura ASR", "value": bool(punct), "log_lr": llr,
                      "source": f"[E] MCFlow punct (sintassi del trascrittore), esponente {alpha} [H]: la punteggiatura Whisper non è annotazione"})
    return {"p": _sigmoid(lo), "log_odds": lo, "terms": terms}


class IOIModel:
    """IDyOM-style information content of word-onset IOIs (sixteenths), trained on MCFlow.

    Interpolated Witten-Bell n-gram (orders 0..3), as in ``tools/mcflow_phrase_stats.IOIModel``.
    Pearce et al. 2010 [E]: boundaries are perceived before high-IC events. Diagnostic
    channel here: on MCFlow held-out it adds nothing to position+gap+rhyme+lengthening.
    """

    def __init__(self, dump: Mapping[str, Any]):
        self.order = int(dump["order"])
        self.alphabet = list(dump["alphabet"])
        self.counts = {tuple(int(x) for x in k.split()) if k else (): {int(a): int(b) for a, b in v.items()}
                       for k, v in dump["counts"].items()}

    def prob(self, ctx: Sequence[int], x: int) -> float:
        p = 1.0 / len(self.alphabet)
        for n in range(0, self.order + 1):
            if n > len(ctx):
                break
            c = self.counts.get(tuple(ctx[len(ctx) - n:]) if n else ())
            if not c:
                continue
            tot = sum(c.values())
            lam = tot / (tot + len(c))
            p = lam * c.get(x, 0) / tot + (1 - lam) * p
        return p

    def ic(self, seq: Sequence[int]) -> list[float]:
        return [-math.log2(self.prob(seq[max(0, i - self.order):i], x)) for i, x in enumerate(seq)]


# ------------------------------------------------------------------- lexicon
class Lexicon:
    """Syllables, stress and rhyme keys: CMUdict for English, orthographic rules otherwise."""

    def __init__(self, path: Optional[os.PathLike | str] = None):
        self.path = Path(path) if path else CMUDICT_PATH
        self._cmu: Optional[dict[str, list[str]]] = None
        self.available = self.path.is_file()

    def _load(self) -> dict[str, list[str]]:
        if self._cmu is None:
            d: dict[str, list[str]] = {}
            if self.available:
                for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
                    parts = line.split("#")[0].split()
                    if len(parts) < 2:
                        continue
                    w = parts[0].lower()
                    if "(" in w:
                        continue  # keep the first pronunciation
                    d[w] = parts[1:]
            self._cmu = d
        return self._cmu

    @staticmethod
    def clean(text: str) -> str:
        t = text.strip().lower().replace("’", "'")
        t = re.sub(r"[^\w'\-]", "", t, flags=re.UNICODE)
        return t.strip("'-_")

    def analyse(self, text: str, lang: str = "en") -> dict[str, Any]:
        w = self.clean(text)
        if not w:
            return {"word": w, "n_syl": 1, "stress_idx": 0, "rhyme": None, "vowel": None, "method": "empty"}
        if lang == "en":
            ph = self._load().get(w)
            if ph is None and w.endswith("in'"):
                ph = self._load().get(w[:-1] + "g")
            if ph is not None:
                vow = [i for i, p in enumerate(ph) if p[-1].isdigit()]
                if vow:
                    prim = [i for i in vow if ph[i].endswith("1")] or [i for i in vow if ph[i].endswith("2")] or vow
                    s = prim[-1]
                    tail = [re.sub(r"\d", "", p) for p in ph[s:]]
                    return {"word": w, "n_syl": len(vow), "stress_idx": vow.index(s), "rhyme": " ".join(tail),
                            "vowel": tail[0], "method": "cmudict"}
        return self._ortho(w, lang)

    @staticmethod
    def _ortho(w: str, lang: str) -> dict[str, Any]:
        """[H] orthographic approximation: vowel groups = syllables; stressed group by a language rule."""
        base = unicodedata.normalize("NFC", w)
        if lang == "fr":  # silent final letters [H]
            if base.endswith("es") and len(base) > 3:
                base = base[:-2]
            elif base[-1] in "sxtd" and len(base) > 2:
                base = base[:-1]
            if base.endswith("e") and len(base) > 2:
                base = base[:-1]
        groups = [(m.start(), m.end()) for m in re.finditer(f"[{_VOWELS}]+", base)]
        if not groups:
            return {"word": w, "n_syl": 1, "stress_idx": 0, "rhyme": base, "vowel": None, "method": f"ortho-{lang}"}
        n = len(groups)
        if lang in ("es", "it", "pt"):
            s = n - 2 if n >= 2 and re.search(f"[{_VOWELS}ns]$", base) else n - 1
        elif lang == "de":
            s = 0
        elif lang == "en":
            s = max(0, n - 2) if n >= 2 else 0
        else:  # fr and others: final syllable
            s = n - 1
        strip = lambda x: "".join(c for c in unicodedata.normalize("NFD", x) if unicodedata.category(c) != "Mn")  # noqa: E731
        tail = strip(base[groups[s][0]:])
        vowel = strip(base[groups[s][0]:groups[s][1]])
        return {"word": w, "n_syl": n, "stress_idx": s, "rhyme": tail, "vowel": vowel, "method": f"ortho-{lang}"}


def rhyme_match(a: Mapping[str, Any], b: Mapping[str, Any]) -> tuple[float, str]:
    """1.0 rhyme (same tail), 0.5 assonance (same stressed vowel) or identity, else 0."""
    if not a.get("rhyme") or not b.get("rhyme"):
        return 0.0, "none"
    if a["word"] == b["word"]:
        return 0.5, "identity"
    if a["rhyme"] == b["rhyme"]:
        return 1.0, "rhyme"
    if a.get("vowel") and a.get("vowel") == b.get("vowel") and not (
            str(a.get("method", "")).startswith("ortho") or str(b.get("method", "")).startswith("ortho")):
        # [H] assonance only from phonemes (CMUdict): five or six written vowels collide too often
        return 0.5, "assonance"
    return 0.0, "none"


# -------------------------------------------------------------------- words
def normalize_words(words: Iterable[Mapping[str, Any]], sample_rate: int = SR) -> list[dict[str, Any]]:
    out = []
    for w in words:
        if "start_sample" in w:
            s, e = int(w["start_sample"]), int(w["end_sample"])
        else:
            s, e = int(round(float(w["start_seconds"]) * sample_rate)), int(round(float(w["end_seconds"]) * sample_rate))
        if e <= s:
            continue
        out.append({"start": s, "end": e, "text": str(w.get("text", w.get("word", ""))), "confidence": w.get("confidence")})
    out.sort(key=lambda x: x["start"])
    return out


@dataclass
class StemActivity:
    """Frame activity of a vocal stem without gap closing (10 ms hop)."""

    centers: np.ndarray  # sample index of frame centres
    active: np.ndarray  # bool
    rms_db: np.ndarray
    threshold_db: float
    f0_hz: np.ndarray
    onsets: np.ndarray
    hop: int
    active_strict: Optional[np.ndarray] = None  # stricter mask for the switch-point safety check

    @classmethod
    def from_features(cls, vf: Any, local_rel_db: float = None, local_window_s: float = None) -> "StemActivity":
        """Local-relative activity: frame above the global floor of vocal_completion *and* within
        ``local_rel_db`` of the local 90th percentile (``local_window_s``). Demucs rap stems are
        rarely silent (reverb, doubles, ad-libs): a global threshold marks ~85 % of frames active."""
        from scipy.ndimage import percentile_filter

        rel = PARAMS["local_rel_db"].value if local_rel_db is None else local_rel_db
        win = PARAMS["local_window_s"].value if local_window_s is None else local_window_s
        c = np.asarray(vf.frame_centers)
        hop = int(np.median(np.diff(c))) if len(c) > 1 else 480
        r = np.asarray(vf.rms_db, dtype=float)
        size = max(3, int(round(win * SR / hop)) | 1)
        lp = percentile_filter(r, 90, size=size, mode="nearest") if r.size else r
        active = r > np.maximum(float(vf.threshold_db), lp - rel)
        strict = r > np.maximum(float(vf.threshold_db), lp - PARAMS["strict_rel_db"].value)
        return cls(centers=c, active=active, rms_db=r, threshold_db=float(vf.threshold_db), f0_hz=np.asarray(vf.f0_hz),
                   onsets=np.asarray(vf.onsets), hop=hop, active_strict=strict)

    @classmethod
    def from_audio(cls, stem: np.ndarray, sample_rate: int = SR) -> "StemActivity":
        from musicmix.vocal_completion import extract_vocal_features

        x = np.asarray(stem, dtype=np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)
        return cls.from_features(extract_vocal_features(x, sample_rate))

    def frames(self, s: int, e: int) -> np.ndarray:
        return (self.centers >= s) & (self.centers < e)

    def support(self, s: int, e: int) -> float:
        m = self.frames(s, e)
        return float(self.active[m].mean()) if m.any() else 0.0

    def active_near(self, t: int, half: int, strict: bool = False) -> bool:
        m = self.frames(t - half, t + half)
        a = self.active_strict if (strict and self.active_strict is not None) else self.active
        return bool(a[m].any())


def _longest_silence(act: StemActivity, a: int, b: int) -> Optional[tuple[int, int]]:
    """Longest run of inactive frames with centre in [a, b): (start, end) samples, or None."""
    idx = np.flatnonzero((act.centers >= a) & (act.centers < b))
    if not idx.size:
        return None
    off = ~act.active[idx]
    best, cur, start = None, 0, 0
    for k, z in enumerate(off):
        if z:
            if cur == 0:
                start = k
            cur += 1
            if best is None or cur > best[1] - best[0]:
                best = (start, k + 1)
        else:
            cur = 0
    if best is None:
        return None
    half = act.hop // 2
    return int(act.centers[idx[best[0]]] - half), int(act.centers[idx[best[1] - 1]] + half)


def refine_words(words: Sequence[Mapping[str, Any]], act: Optional[StemActivity], sample_rate: int = SR,
                 p: Optional[Mapping[str, Any]] = None) -> list[dict[str, Any]]:
    """Acoustic on/off for each ASR word; low-support words flagged ``dropped`` (kept for audit).

    Whisper word boundaries abut in fast rap (gap 0 for ~85 % of pool words) and can be off by
    100-200 ms, so the pause after word i is searched as the longest silent run of the stem
    (local-relative activity) between the middles of words i and i+1; word i ends where that
    silence starts and word i+1 starts where it ends. No silence = no pause (off = next on).
    """
    pr = params(p)
    out = [dict(w, on=int(w["start"]), off=int(w["end"]), support=None, dropped=False) for w in words]
    if act is None:
        for i in range(len(out) - 1):
            out[i]["off"] = min(out[i]["off"], out[i + 1]["on"])
        return out
    for w in out:
        w["support"] = act.support(w["start"], w["end"])
        w["dropped"] = bool(w["support"] < pr["support_min"])
    kept = [w for w in out if not w["dropped"]]
    mids = [(w["start"] + w["end"]) // 2 for w in kept]
    for i in range(len(kept) - 1):
        w, nx = kept[i], kept[i + 1]
        sil = _longest_silence(act, mids[i], mids[i + 1])
        if sil is not None:
            w["off"], nx["on"] = max(sil[0], w["on"] + 1), min(max(sil[1], sil[0]), nx["end"] - 1)
        else:
            b = min(max(w["end"], nx["start"]), nx["end"] - 1)  # sound continues across the boundary
            w["off"], nx["on"] = b, b
    if kept:
        # first word: skip leading silence; last word: trim trailing silence (bounded by refine_max_extend_s)
        f, l = kept[0], kept[-1]
        sil = _longest_silence(act, f["start"], (f["start"] + f["end"]) // 2)
        if sil is not None and sil[0] <= f["start"] + act.hop:
            f["on"] = sil[1]
        ext = int(pr["refine_max_extend_s"] * sample_rate)
        sil = _longest_silence(act, (l["start"] + l["end"]) // 2, l["end"] + ext)
        if sil is not None:
            l["off"] = sil[0]
    lim = int(pr["collapsed_word_s"] * sample_rate)
    for i, w in enumerate(kept):
        if w["off"] - w["on"] < lim:
            lo = kept[i - 1]["off"] if i > 0 else w["start"]
            hi = kept[i + 1]["on"] if i + 1 < len(kept) else w["end"]
            w["collapsed"] = True
            w["on"], w["off"] = max(w["start"], lo), min(w["end"], hi)
        w["off"] = max(w["off"], w["on"] + 1)
    return out


# --------------------------------------------------------------------- grid
@dataclass
class Grid:
    downbeats: np.ndarray  # seconds
    period_s: float

    @classmethod
    def from_track(cls, track: Mapping[str, Any]) -> "Grid":
        return cls(np.asarray(track["grid"]["downbeats_s"], dtype=float), float(track["tempo"]["period_s"]))

    def locate(self, t_s: float) -> tuple[int, float]:
        """(bar index, position in beats from that bar's downbeat); bar -1 before the first downbeat."""
        j = int(np.searchsorted(self.downbeats, t_s, side="right") - 1)
        if j < 0:
            return -1, (t_s - (self.downbeats[0] - 4 * self.period_s)) / self.period_s
        return j, (t_s - self.downbeats[j]) / self.period_s


# ------------------------------------------------------------------ analysis
def analyze_track(track: Mapping[str, Any], words: Sequence[Mapping[str, Any]], *, act: Optional[StemActivity] = None,
                  lang: str = "en", lexicon: Optional[Lexicon] = None, mcflow: Optional[Mapping[str, Any]] = None,
                  sample_rate: int = SR, p: Optional[Mapping[str, Any]] = None,
                  asr_language_probability: Optional[float] = None) -> dict[str, Any]:
    """Explicit line/bar model of one track: words, word-boundary probabilities, lines."""
    pr = params(p)
    m = mcflow or load_mcflow()
    lex = lexicon or Lexicon()
    grid = Grid.from_track(track)
    sixteenth = grid.period_s / 4.0
    ws = refine_words(normalize_words(words, sample_rate), act, sample_rate, pr)
    quality = asr_quality(ws, asr_language_probability)
    if not quality["reliable"]:
        for w in ws:
            w["dropped"] = True
    kept = [w for w in ws if not w["dropped"]]
    # lexical analysis and syllable timing
    for w in kept:
        lx = lex.analyse(w["text"], lang)
        w.update(n_syl=max(1, int(lx["n_syl"])), stress_idx=int(lx["stress_idx"]), rhyme_key=lx["rhyme"], vowel_key=lx["vowel"],
                 lex_method=lx["method"], clean=lx["word"])
        dur = w["off"] - w["on"]
        # [H] uniform syllables inside the word, last syllable onset snapped to the nearest flux onset within +-60 ms
        last_on = w["on"] + dur * (w["n_syl"] - 1) / w["n_syl"]
        if act is not None and act.onsets.size and w["n_syl"] > 1:
            near = act.onsets[(act.onsets >= w["on"]) & (act.onsets < w["off"])]
            if near.size:
                k = near[np.argmin(np.abs(near - last_on))]
                if abs(k - last_on) <= 0.06 * sample_rate:
                    last_on = float(k)
        w["last_syl_on"] = int(last_on)
        w["syl_dur_s"] = dur / sample_rate / w["n_syl"]
        bar, pos = grid.locate(last_on / sample_rate)
        w["bar"], w["pos_beats"] = bar, pos
        w["beat"] = int(min(3, max(0, math.floor(pos)))) + 1 if bar >= 0 else None
        w["punct"] = bool(re.search(r"[.;:,!?]$", w["text"].strip()))
    med_syl = float(np.median([w["syl_dur_s"] for w in kept])) if kept else 0.0
    # rhyme evidence: same tail 1 or 2 bars before/after at a parallel position (+-1 beat)
    for i, w in enumerate(kept):
        best, kind, partner = 0.0, "none", None
        if w["clean"] not in _EN_STOP and w["rhyme_key"] and w["bar"] is not None and w["bar"] >= 0:
            for v in kept:
                if v is w or v["bar"] is None or v["bar"] < 0 or v["clean"] in _EN_STOP:
                    continue
                if not any(abs(abs((v["last_syl_on"] - w["last_syl_on"]) / sample_rate / grid.period_s) - 4 * L) <= pr["rhyme_window_beats"]
                           for L in pr["rhyme_bar_lags"]):
                    continue
                s, k = rhyme_match({"word": w["clean"], "rhyme": w["rhyme_key"], "vowel": w["vowel_key"], "method": w["lex_method"]},
                                   {"word": v["clean"], "rhyme": v["rhyme_key"], "vowel": v["vowel_key"], "method": v["lex_method"]})
                if s > best:
                    best, kind, partner = s, k, v["text"]
        w["rhyme"] = {"score": best, "kind": kind, "partner": partner}
    # f0 (semitones re 100 Hz) of the last syllable vs the preceding 2 s of voiced frames [diagnostic]
    if act is not None:
        f0 = act.f0_hz
        for w in kept:
            sel = act.frames(w["last_syl_on"], w["off"])
            ctx = act.frames(w["off"] - 2 * sample_rate, w["last_syl_on"])
            a, b = f0[sel], f0[ctx]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            w["f0_final_st"] = float(12 * np.log2(np.median(a) / np.median(b))) if a.size >= 3 and b.size >= 10 else None
    # word-onset IOI information content (MCFlow-trained, diagnostic)
    ioi_model = IOIModel(m["ioi_model"]) if "ioi_model" in m else None
    iois = [min(16, max(1, int(round((kept[i + 1]["on"] - kept[i]["on"]) / sample_rate / sixteenth)))) for i in range(len(kept) - 1)]
    ics = ioi_model.ic(iois) if ioi_model and iois else []
    gaps = []
    for i in range(len(kept) - 1):
        w, nx = kept[i], kept[i + 1]
        g_s = (nx["on"] - w["off"]) / sample_rate
        g16 = max(0.0, g_s / sixteenth)
        rest = g16 >= pr["rest_min_16th"]
        last_dur = (w["off"] - w["last_syl_on"]) / sample_rate
        leng = last_dur / med_syl if med_syl > 0 else None
        beat = w["beat"] or 4
        bp = boundary_probability(beat, rest, g16, w["rhyme"]["score"] >= 0.5, leng, w["punct"], mcflow=m,
                                  alpha=pr["audio_lr_exponent"])
        gaps.append({"after": i, "end_sample": w["off"], "next_start_sample": nx["on"], "gap_s": g_s, "gap16": g16,
                     "rest": rest, "beat": w["beat"], "bar": w["bar"], "pos_beats": w["pos_beats"],
                     "rhyme": w["rhyme"], "lengthening": leng, "punct": w["punct"], "f0_final_st": w.get("f0_final_st"),
                     "ioi_ic_next": ics[i] if i < len(ics) else None,
                     "p_end": bp["p"], "terms": bp["terms"], "word": w["text"], "next_word": nx["text"]})
    thr = pr["line_end_threshold"]
    pillar = pillar_adjust(gaps, weight=pr["pillar_weight"], min_ends=pr["pillar_min_ends"], threshold=thr)
    lines = lines_from_gaps(kept, gaps, thr)
    return {"schema_version": SCHEMA_VERSION, "slug": track.get("slug"), "lang": lang, "sample_rate": sample_rate,
            "period_s": grid.period_s, "downbeats_s": grid.downbeats.tolist(),
            "words": kept, "dropped_words": [w for w in ws if w["dropped"]], "gaps": gaps, "lines": lines,
            "median_syllable_s": med_syl,
            "asr_quality": quality, "pillar": pillar,
            "summary": {"n_words_asr": len(ws), "n_words": len(kept), "n_dropped": len(ws) - len(kept),
                        "asr_reliable": quality["reliable"],
                        "n_lines": len(lines), "lexicon": {k: sum(1 for w in kept if w["lex_method"] == k) for k in {w["lex_method"] for w in kept}},
                        "cmudict_available": lex.available,
                        "mean_words_per_line": (len(kept) / len(lines)) if lines else None},
            "params": pr}


def lines_from_gaps(words: Sequence[Mapping[str, Any]], gaps: Sequence[Mapping[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Lines = runs of words between boundaries with p_end >= threshold (MAP decision)."""
    lines, start = [], 0
    for g in gaps:
        if g["p_end"] >= threshold:
            lines.append(_line(words, start, g["after"], g["p_end"]))
            start = g["after"] + 1
    if words:
        lines.append(_line(words, start, len(words) - 1, 1.0))
    return lines


def _pos16(g: Mapping[str, Any]) -> Optional[int]:
    if g.get("bar") is None or g["bar"] < 0 or g.get("pos_beats") is None:
        return None
    return int(round(4.0 * float(g["pos_beats"]))) % 16


def _circular_density16(positions: Iterable[int], prior: float = 0.5) -> np.ndarray:
    """Density over 16 positions: prior + counts, smoothed 1/4-1/2-1/4 (one sixteenth) as in the MCFlow evaluation."""
    d = np.full(16, prior)
    for q in positions:
        d[int(q) % 16] += 1
    d = 0.25 * np.roll(d, 1) + 0.5 * d + 0.25 * np.roll(d, -1)
    return d / d.sum()


def pillar_corpus_density() -> Optional[np.ndarray]:
    """Density of predicted line ends per position in MCFlow (base model), from configs/dj_phrase_flow_v1.json."""
    try:
        v = json.loads(FLOW_CONFIG.read_text(encoding="utf-8"))["pillar_corpus_density"]
        a = np.asarray(v, dtype=float)
        return a / a.sum() if a.size == 16 and np.all(a > 0) else None
    except (OSError, KeyError, ValueError):
        return None


def _set_pillar_term(g: dict[str, Any], term: Optional[float], q: Optional[int] = None) -> None:
    """Add (or with ``term`` None remove) the pillar term in ``g['terms']`` (a list of dicts as built by
    the line-end model, or a dict in hand-made gaps)."""
    t = g.get("terms")
    if isinstance(t, dict):
        t.pop("pillar", None)
        if term is not None:
            t["pillar"] = term
        return
    t = [x for x in (t or []) if not (isinstance(x, dict) and x.get("name") == "pilastro")]
    if term is not None:
        t.append({"name": "pilastro", "value": q, "log_lr": term,
                  "source": "[A] Ohriner 2019 p. 110 (posizione tipica di fine frase del brano); [E] MCFlow held-out"})
    g["terms"] = t


def pillar_adjust(gaps: list[dict[str, Any]], *, weight: float, min_ends: int = 8, threshold: float = 0.5,
                  corpus: Optional[np.ndarray] = None) -> dict[str, Any]:
    """[A] Ohriner p. 110 central pillar as a per-track prior on line-end positions (in place on ``gaps``).

    Transductive (no labels): the track's density of *predicted* ends (p_end >= threshold) per sixteenth
    is compared with the MCFlow corpus density; each gap gets ``weight * log(d_track / d_corpus)`` added
    to its log-odds (term ``pillar``). Evaluated held-out by song on MCFlow (see PARAMS['pillar_weight']).
    ``weight`` 0 only reports the densities. The original probability is kept as ``p_end_base``.
    """
    ends = [q for q in (_pos16(g) for g in gaps if g["p_end"] >= threshold) if q is not None]
    dc = corpus if corpus is not None else pillar_corpus_density()
    info: dict[str, Any] = {"weight": weight, "n_predicted_ends": len(ends), "applied": False,
                            "source": "[A] Ohriner 2019 p. 110; [E] held-out MCFlow (runs/dj_phrase_v2/mcflow_books_eval.json)"}
    if not ends:
        return info
    ds = _circular_density16(ends)
    info["track_density"] = ds.round(4).tolist()
    info["pillar_pos16"] = int(np.argmax(ds))
    if weight <= 0 or dc is None or len(ends) < min_ends:
        return info
    for g in gaps:
        q = _pos16(g)
        if q is None:
            continue
        term = float(weight * math.log(ds[q] / dc[q]))
        g["p_end_base"] = g.get("p_end_base", g["p_end"])
        g["p_end"] = _sigmoid(_logit(g["p_end_base"]) + term)
        _set_pillar_term(g, term, q)
    info["applied"] = True
    info["n_ends_after"] = sum(1 for g in gaps if g["p_end"] >= threshold)
    return info


def reanalyze_with_pillar(model: Mapping[str, Any], weight: float = 1.0, min_ends: Optional[int] = None) -> dict[str, Any]:
    """Copy of an existing model with the pillar term applied (no ASR/audio needed): new gaps and lines."""
    import copy

    m = copy.deepcopy(dict(model))
    pr = params(m.get("params"))
    for g in m["gaps"]:  # start from the v1 probabilities
        if "p_end_base" in g:
            g["p_end"] = g.pop("p_end_base")
            _set_pillar_term(g, None)
    thr = pr["line_end_threshold"]
    m["pillar"] = pillar_adjust(m["gaps"], weight=weight, min_ends=pr["pillar_min_ends"] if min_ends is None else min_ends,
                                threshold=thr)
    m["lines"] = lines_from_gaps(m["words"], m["gaps"], thr)
    m["params"] = dict(pr, pillar_weight=weight)
    m.setdefault("summary", {})["n_lines"] = len(m["lines"])
    return m


def asr_quality(words: Sequence[Mapping[str, Any]], language_probability: Optional[float] = None) -> dict[str, Any]:
    """[H] Gate against unusable transcripts: the model then abstains (no words, no line ends)."""
    if not words:
        return {"reliable": False, "reason": "no words"}
    dur = np.array([(w["end"] - w["start"]) / SR for w in words])
    toks = [Lexicon.clean(w["text"]) for w in words]
    top = max(set(toks), key=toks.count)
    share = toks.count(top) / len(toks)
    letters = sum(1 for t in toks if len(t) >= 2) / len(toks)
    reasons = []
    if language_probability is not None and language_probability < 0.5:
        reasons.append(f"probabilità della lingua {language_probability:.2f} < 0,5")
    if np.median(dur) > 1.0:
        reasons.append(f"durata mediana delle parole {np.median(dur):.2f} s > 1 s")
    if share > 0.5 and len(toks) >= 10:
        reasons.append(f"la parola «{top}» è il {100 * share:.0f} % del testo")
    if letters < 0.5:
        reasons.append("meno della metà delle parole ha almeno due caratteri")
    return {"reliable": not reasons, "reasons": reasons, "median_word_s": float(np.median(dur)), "top_word_share": share,
            "language_probability": language_probability}


def _line(words: Sequence[Mapping[str, Any]], i0: int, i1: int, p_end: float) -> dict[str, Any]:
    return {"first_word": i0, "last_word": i1, "start_sample": words[i0]["on"], "end_sample": words[i1]["off"],
            "text": " ".join(w["text"] for w in words[i0:i1 + 1]), "p_end": p_end,
            "end_rhyme": words[i1].get("rhyme", {}).get("kind")}


# ------------------------------------------------------------ exits / entries
def _word_at(words: Sequence[Mapping[str, Any]], t: int) -> Optional[int]:
    for i, w in enumerate(words):
        if w["on"] <= t < w["off"]:
            return i
    return None


def truncated_words(words: Sequence[Mapping[str, Any]], cut: int, *, role: str, margin: int,
                    keep_until: Optional[int] = None) -> list[dict[str, Any]]:
    """Words broken by an exit (role 'a': audio stops at ``cut``) or an entry (role 'b': audio starts at ``cut``).

    A word is truncated if it is audible for >= ``margin`` on one side and loses >= ``margin`` on the other.
    ``keep_until`` (role 'a'): A's vocal stays audible until that sample (protected tail).
    """
    out = []
    for w in words:
        if role == "a":
            end_heard = max(cut, keep_until or cut)
            if w["on"] < cut - margin and w["off"] > end_heard + margin:
                out.append({"text": w["text"], "on": w["on"], "off": w["off"], "lost_s": (w["off"] - end_heard) / SR})
        else:
            if w["on"] < cut - margin and w["off"] > cut + margin:
                out.append({"text": w["text"], "on": w["on"], "off": w["off"], "lost_s": (cut - w["on"]) / SR})
    return out


def exit_point(model: Mapping[str, Any], downbeat_s: float, p: Optional[Mapping[str, Any]] = None,
               act: Optional[StemActivity] = None) -> dict[str, Any]:
    """Can A leave at (or just before) the downbeat ``downbeat_s`` at the end of a line?

    Returns ``window`` = [lo, hi] in samples relative to D in which A is vocally silent
    (hi > 0 only with a protected tail), the line-end probability of the last boundary,
    the tail and A's own pickup into its next line, and the words a dry cut at D truncates.
    """
    pr = params(p or model.get("params"))
    sr = model["sample_rate"]
    D = int(round(downbeat_s * sr))
    beat = int(round(model["period_s"] * sr))
    m = int(pr["switch_margin_s"] * sr)
    words, gaps = model["words"], model["gaps"]
    res: dict[str, Any] = {"downbeat_sample": D, "status": "instrumental", "window": None, "p_line_end": None,
                           "reasons": [], "tail_end_sample": None, "a_pickup_sample": None,
                           "dry_cut_truncates": [w["text"] for w in truncated_words(words, D, role="a", margin=int(pr["word_margin_s"] * sr))]}
    lo_lim = D - int(pr["pickup_max_beats"] * beat)
    tail_lim = D + int(pr["tail_max_beats"] * beat)
    if not (model.get("asr_quality") or {}).get("reliable", True):
        res.update(status="asr_unreliable")
        res["reasons"].append("trascrizione non affidabile: il modello si astiene")
        return res
    before = [i for i, w in enumerate(words) if w["on"] < D]
    if not before:
        res.update(status="instrumental", window=[lo_lim - D, 0], p_line_end=None)
        res["reasons"].append("nessuna parola di A prima del downbeat")
        return _acoustic(res, act, D, m, "a")
    last = before[-1]
    if last == len(words) - 1 and words[last]["off"] <= tail_lim:
        # the last word of the track (end of the vocals): a line end by definition
        w = words[last]
        tail = int(w["off"]) if w["off"] + m > D else None
        res.update(status="clean" if tail is None else "tail", window=([max(w["off"] + m, lo_lim) - D, 0] if tail is None else [0, 0]),
                   p_line_end=1.0, tail_end_sample=tail, last_word=w["text"], next_word=None)
        res["reasons"].append(f"ultima parola del brano «{w['text']}» prima del downbeat")
        return _acoustic(res, act, D, m, "a")
    # candidate line ends: gaps after word i with i <= last; the latest one whose silence reaches the window
    best, latest_any = None, None
    for g in reversed(gaps[:last + 1]):
        i = g["after"]
        w, nx = words[i], words[i + 1]
        if nx["on"] - m < lo_lim:
            break  # A's next line started more than pickup_max before D: switch would be too early
        if w["off"] > tail_lim:
            continue  # the word runs too far past D
        lo_c = D if w["off"] + m > D else max(w["off"] + m, lo_lim)
        if min(nx["on"] - m, D) < lo_c:
            continue  # no usable silence after it
        latest_any = latest_any or g
        if g["p_end"] >= pr["line_end_threshold"]:
            best = g
            break
    best = best or latest_any
    if best is None:
        wi = _word_at(words, D)
        res.update(status="mid_line", window=None)
        res["reasons"].append("una parola attraversa il downbeat e nessun fine verso è raggiungibile"
                              + (f" («{words[wi]['text']}»)" if wi is not None else ""))
        return res
    i = best["after"]
    w, nx = words[i], words[i + 1]
    res["p_line_end"] = best["p_end"]
    q = _pos16(best)
    res["end_pos16"] = q
    # [A] Krims pp. 132-133, 192-194: line ends and caesuras on BC 3 (4th beat) mark the sung style and close the line
    res["end_on_beat4"] = (q is not None and q >= 12)
    res["last_word"] = w["text"]
    res["next_word"] = nx["text"]
    res["line_end_terms"] = best["terms"]
    lo = max(w["off"] + m, lo_lim)
    hi = min(nx["on"] - m, D)
    tail = None
    if w["off"] + m > D:
        tail = int(w["off"])
        lo, hi = D, min(nx["on"] - m, D)
    if hi < lo:
        res.update(status="no_gap", window=None)
        res["reasons"].append(f"silenzio tra «{w['text']}» e «{nx['text']}» troppo breve per lo switch")
        return res
    status = "clean" if (hi == D and tail is None) else ("tail" if tail is not None else "a_pickup")
    res.update(status=status, window=[int(lo - D), int(hi - D)], tail_end_sample=tail,
               a_pickup_sample=(int(nx["on"]) if nx["on"] < D else None))
    # [A] Krims p. 194 (one song) and p. 133: a caesura on BC 3 closes the line; a line end before BC 3
    # followed by A's next syllables before the downbeat is an upbeat into A's next line. Diagnostic only:
    # never used in the exit window/score (KRIMS_CLOSURE_NOTE; Krims p. 32 is not a source of a cue rule).
    res["krims_closure"] = ("pickup" if res["a_pickup_sample"] is not None
                            else ("closed_bc3" if res["end_on_beat4"] else "closed_early"))
    if best["p_end"] < pr["line_end_threshold"]:
        res["status"] = "gap_not_line_end"
        res["reasons"].append(f"silenzio dopo «{w['text']}» ma p(fine verso) = {best['p_end']:.2f} < {pr['line_end_threshold']}")
    else:
        res["reasons"].append(f"fine verso dopo «{w['text']}» (p = {best['p_end']:.2f})")
    if tail is not None:
        res["reasons"].append(f"coda di «{w['text']}» protetta per {(tail - D) / sr * 1000:.0f} ms dopo il downbeat (stem vocale)")
    if res["a_pickup_sample"] is not None:
        res["reasons"].append(f"A riparte con «{nx['text']}» {(D - nx['on']) / sr * 1000:.0f} ms prima del downbeat: switch prima del pickup di A")
    return _acoustic(res, act, D, m, "a")


def entry_point(model: Mapping[str, Any], downbeat_s: float, p: Optional[Mapping[str, Any]] = None,
                act: Optional[StemActivity] = None) -> dict[str, Any]:
    """Where may B become audible so that its line through the downbeat ``downbeat_s`` is whole?

    ``pickup_samples`` = D - onset of the line's first word when the line starts before D
    (at most ``pickup_max_beats``); ``window`` = [lo, hi] relative to D in which B can be
    switched on without truncating its previous word or its pickup.
    """
    pr = params(p or model.get("params"))
    sr = model["sample_rate"]
    D = int(round(downbeat_s * sr))
    beat = int(round(model["period_s"] * sr))
    m = int(pr["switch_margin_s"] * sr)
    words, gaps = model["words"], model["gaps"]
    lo_lim = D - int(pr["pickup_max_beats"] * beat)
    res: dict[str, Any] = {"downbeat_sample": D, "status": "instrumental", "pickup_samples": 0, "pickup_words": [],
                           "window": [lo_lim - D, 0], "reasons": []}
    if not (model.get("asr_quality") or {}).get("reliable", True):
        res.update(status="asr_unreliable", window=None)
        res["reasons"].append("trascrizione non affidabile: il modello si astiene")
        return res
    # first word at or after the earliest allowed start
    idx = [i for i, w in enumerate(words) if w["off"] > lo_lim]
    if not idx or words[idx[0]]["on"] >= D + 4 * beat:
        res["reasons"].append("nessuna voce di B tra 2 battiti prima e una battuta dopo il downbeat")
        res["first_word_rel"] = (words[idx[0]]["on"] - D) if idx else None
        return _acoustic(res, act, D, m, "b")
    # line start = first word after the last boundary (p >= pickup_line_p) before D
    j = idx[0]
    start = j
    for i in range(j, len(words)):
        if words[i]["on"] >= D:
            break
        if i > j and gaps[i - 1]["p_end"] >= pr["pickup_line_p"]:
            start = i
    wi = _word_at(words, D)
    if start == j and j > 0 and words[j]["on"] < D and gaps[j - 1]["p_end"] < pr["pickup_line_p"]:
        res.update(status="mid_line", window=None)
        res["reasons"].append(f"B è a metà verso sul downbeat («{words[j]['text']}» segue un confine debole, "
                              f"p = {gaps[j - 1]['p_end']:.2f})")
        return res
    if start == j and words[j]["on"] < lo_lim:
        res.update(status="mid_line", window=None)
        res["reasons"].append(f"B è a metà verso sul downbeat («{words[j]['text']}» inizia più di "
                              f"{pr['pickup_max_beats']:g} battiti prima)")
        return res
    first = words[start]
    prev_off = words[start - 1]["off"] if start > 0 else 0
    lo = max(prev_off + m, lo_lim)
    if first["on"] < D:
        hi = first["on"] - m
        res.update(status="pickup", pickup_samples=int(D - first["on"]),
                   pickup_words=[w["text"] for w in words[start:] if w["on"] < D])
        res["reasons"].append(f"pickup di B: «{' '.join(res['pickup_words'])}» {(D - first['on']) / sr * 1000:.0f} ms prima del downbeat")
    else:
        hi = min(D, first["on"] - m)
        res["status"] = "on_or_after_downbeat"
        res["reasons"].append(f"il verso di B inizia {(first['on'] - D) / sr * 1000:.0f} ms dopo il downbeat («{first['text']}»)")
    if hi < lo:
        res.update(status="no_gap", window=None)
        res["reasons"].append("la parola precedente di B finisce troppo vicino all'inizio del verso")
        return res
    res["window"] = [int(lo - D), int(hi - D)]
    res["first_word"] = first["text"]
    res["first_word_rel"] = int(first["on"] - D)
    res["line_start_p"] = gaps[start - 1]["p_end"] if start > 0 else 1.0
    if res["line_start_p"] < pr["line_end_threshold"]:
        res["reasons"].append(f"«{first['text']}» dopo il downbeat segue un confine debole (p = {res['line_start_p']:.2f})")
    return _acoustic(res, act, D, m, "b")


def _acoustic(res: dict[str, Any], act: Optional[StemActivity], D: int, m: int, role: str) -> dict[str, Any]:
    """Vocal-stem activity in the window, independent of ASR: latest point with >= ``m`` of silence each side."""
    if act is None or res.get("window") is None:
        return res
    if role == "a" and res.get("tail_end_sample") is not None:
        res["acoustic"] = "tail_protected"
        return res
    lo, hi = res["window"]
    s0, s1 = D + lo, D + hi
    sel = act.frames(s0 - m, s1 + act.hop)
    c = act.centers[sel]
    if not c.size:
        res["acoustic"] = "no_frames"
        return res
    res["acoustic_active_fraction"] = float(act.active[sel].mean())
    best = None
    for t in c[::-1]:
        t = int(min(t, s1))
        if t < s0:
            break
        if not act.active_near(t, m, strict=True):
            best = t
            break
    if best is None:
        res["acoustic"] = "stem_active_in_window"
        res["reasons"].append("lo stem vocale non è mai silenzioso nella finestra (ad-lib, cori o riverbero)")
    else:
        res["acoustic"] = "silent_point"
        res["acoustic_latest_silent"] = int(best - D)
    return res


def phrase_switch(exit_a: Mapping[str, Any], entry_b: Mapping[str, Any], period_s: float,
                  p: Optional[Mapping[str, Any]] = None, sample_rate: int = SR) -> dict[str, Any]:
    """Combine A's exit window and B's entry window (both relative to the aligned downbeats).

    The fader switch x (samples, <= 0, relative to the downbeat) is the latest point allowed by
    both windows, snapped down to the eighth-note grid when possible. ``ok`` means: A ends at a
    line end (p >= threshold), no word of A or B is truncated, B's pickup is whole.
    """
    pr = params(p)
    out: dict[str, Any] = {"ok": False, "switch_rel_samples": None, "reasons": []}
    wa, wb = exit_a.get("window"), entry_b.get("window")
    if wa is None or wb is None:
        out["reasons"].append(f"finestra assente: A {exit_a.get('status')}, B {entry_b.get('status')}")
        return out
    lo, hi = max(wa[0], wb[0]), min(wa[1], wb[1], 0)
    for e in (exit_a, entry_b):
        if e.get("acoustic") == "silent_point":
            hi = min(hi, e["acoustic_latest_silent"])
    tail = exit_a.get("tail_end_sample")
    tail_rel = (tail - exit_a["downbeat_sample"]) if tail is not None else None
    if tail_rel is not None:
        # B's first voice must start after A's protected tail
        fw = entry_b.get("first_word_rel")
        if entry_b.get("status") == "pickup" or (fw is not None and fw < tail_rel + int(pr["switch_margin_s"] * sample_rate)):
            out["reasons"].append("coda di A e prima parola di B si sovrappongono")
            return out
    if hi < lo:
        out["reasons"].append(f"finestre incompatibili: A [{wa[0] / sample_rate:+.2f}, {wa[1] / sample_rate:+.2f}] s, "
                              f"B [{wb[0] / sample_rate:+.2f}, {wb[1] / sample_rate:+.2f}] s")
        return out
    step = int(round(period_s * sample_rate / pr["switch_grid_div"]))
    x = hi
    if hi < 0:
        snapped = -int(math.ceil(-hi / step)) * step  # latest grid point <= hi
        if snapped >= lo:
            x = snapped
    line_ok = exit_a.get("status") == "instrumental" or (exit_a.get("p_line_end") or 0) >= pr["line_end_threshold"]
    acoustic_ok = exit_a.get("acoustic") != "stem_active_in_window" and entry_b.get("acoustic") != "stem_active_in_window"
    entry_ok = (entry_b.get("line_start_p") is None) or entry_b["line_start_p"] >= pr["line_end_threshold"]
    out.update(switch_rel_samples=int(x), switch_rel_s=x / sample_rate, tail_rel_samples=tail_rel,
               pickup_samples=int(entry_b.get("pickup_samples") or 0),
               pickup_preserved=bool(entry_b.get("status") != "pickup" or x <= wb[1]),
               line_end=line_ok, acoustic_ok=acoustic_ok, entry_line_start=entry_ok,
               ok=bool(line_ok and acoustic_ok and entry_ok and exit_a.get("status") not in ("gap_not_line_end",)))
    out["reasons"] += [f"switch {x / sample_rate * 1000:+.0f} ms dal downbeat"
                       + (" (sul downbeat)" if x == 0 else " (anticipato all'ottavo)" if x % step == 0 else "")]
    if not line_ok:
        out["reasons"].append("A non è a fine verso")
    if not acoustic_ok:
        out["reasons"].append("stem vocale attivo nella finestra (non coperto dalle parole ASR)")
    if not entry_ok:
        out["reasons"].append("B non entra a inizio verso")
    return out


# ------------------------------------------------------------- flow (books)
# Regole tratte dai libri (note in conoscenza_mix/libri/). Etichette: [T] teoria, [E] evidenza
# empirica (N dichiarato), [A] analisi o algoritmo di un singolo autore, [H] ipotesi/scelta nostra.
# Nessuna di queste regole è validata su giudizi d'ascolto di transizioni DJ: sono diagnostiche
# e opzioni, con i default che lasciano invariato il comportamento v1.

BOOK_RULES: dict[str, Param] = {
    "accent_long_beats": Param(0.34, "[A] Ohriner 2019 p. 57: sillaba 'long' se > 0,34 battiti (più lunga di una terzina di crome); "
                                     "algoritmo d'accento in tre fasi pp. 57-59, 'giusto' nel 96 % dei casi rispetto al suo ascolto (p. 62)"),
    "groove_classes": Param(("2222_2222", "332_2222", "323_2222", "332_332", "333322", "3223222", "333232"),
                            "[A] Ohriner 2019 pp. 84-86: con IAI di 2 o 3 sedicesimi che sommano a 16 esistono esattamente sette classi di groove"),
    "groove_class_share": Param({"2222_2222": 0.44, "332_2222": 0.27, "323_2222": 0.10, "332_332": 0.07, "333322": 0.05,
                                 "3223222": 0.04, "333232": 0.03},
                                "[E] Ohriner 2019 Tab. 4.3 p. 101: quota di battute per classe, ascoltatore adattivo (0 swap), corpus di 75 strofe"),
    "groove_effort_per_bar": Param(2.0, "[A] Ohriner 2019 pp. 96-98: sforzo = distanza DTW tra insiemi di accenti (uno spostamento di "
                                        "un sedicesimo costa 2); 'ascoltatore persistente' a uno swap per battuta"),
    "grooviness_min_bars": Param(2, "[A] Ohriner 2019 p. 117 n. 16: un groove conta se dura almeno due battute (deve ripetersi una volta)"),
    "syllable_speed_mean": Param(5.09, "[E] Ohriner 2019 p. 106: velocità sillabica media 5,09 sillabe/s, N = 75 strofe; non cambia con gli anni (p. 107)"),
    "saturation_change_typical": Param(2.0, "[E] Ohriner 2019 p. 108: da una battuta all'altra si aggiungono o tolgono di solito 1-2 sillabe"),
    "pillar_late_share": Param(2 / 3, "[E] Ohriner 2019 p. 110: due terzi di 75 strofe hanno il 'pilastro centrale' (moda delle fini di frase "
                                      "da respiro) su o dopo la posizione 12; solo 9 su 75 nella prima metà della battuta"),
    "krims_styles": Param(("sung", "percussion_effusive", "speech_effusive"),
                          "[T] Krims 2000 pp. 49-52: tre stili di flow come direzioni su spettri, non categorie separate; possono "
                          "coesistere e cambiare dentro una strofa ('rhythmic acceleration' p. 52)"),
    "four_bar_groups": Param(4, "[T] Krims 2000 pp. 52-53: gruppi di quattro battute come norma del rap; [T]+[corpus d'esempio] Huron 2006 "
                                "pp. 179-180 (ipermetro: battere forte-debole-medio-debole su 4 battute) e pp. 247-250 (fill che "
                                "annunciano il battere ipermetrico). Limite [A] Williams 2013 p. 10: un frammento di durata non "
                                "regolare (non 4/8 battute) o leggermente stonato può segnalare di proposito una citazione: "
                                "l'avviso 'non multiplo di 4' è diagnostico, non un difetto"),
    "loop_establish_repeats": Param(2, "[T] Margulis 2014 p. 51: 'from a discursive perspective, a single repetition is enough to establish "
                                       "a pattern' (modello + copia); [A] Ohriner 2019 p. 117 e n. 16: un groove conta se dura almeno due "
                                       "battute. [H] soglia dell'avviso"),
    "loop_focal_repeats": Param(4, "[T] Margulis 2014 p. 51 (Lidov 2004): categorie teoriche formative/focali/testurali; la ripetizione "
                                   "'three-or-fourfold' tende al focale, oltre 'the three-or-four mark' al testurale. Sono categorie, non "
                                   "un massimo empirico di loop: [H] soglia dell'avviso. (Correzione 21/10/2026: 1/4/8 di Margulis p. 8 "
                                   "sono secondi delle unità ripetute, non numeri di copie)"),
    "loop_with_new_layer_max_bars": Param(8, "[H] con B che entra sopra il loop l'attenzione va al nuovo strato (Margulis p. 51 [T], "
                                             "repetition 'textural'); limite a un'unità ipermetrica di 8 battute (Krims p. 53; Huron "
                                             "pp. 179-180): nessuna fonte dà questo numero"),
    "loop_vocal_max_repeats": Param(2, "[H] ipotesi nostra, solo diagnostica: un loop con parole di A oltre 2 ripetizioni può cambiare il "
                                       "senso del testo. Contesto [E] Margulis 2014 p. 16: nell'illusione speech-to-song (Deutsch) la "
                                       "ripetizione 'excessive and temporally regular' di una frase parlata la fa percepire cantata a circa "
                                       "l'85 % degli ascoltatori; saturazione semantica p. 17. Il libro non dà un numero di ripetizioni "
                                       "(verifica 21/10/2026): nessun limite di 2 copie"),
    "loop_comic_repeats": Param(3, "[A] Huron 2006 p. 286 (Schickele, una registrazione dal vivo di un brano comico: un passaggio di 8 note "
                                   "ripetuto 12 volte, risate durante la 4a ripetizione, 'consistent with a norm of three repetitions'); "
                                   "Margulis 2014 pp. 173-174 (Huron 2004: 629 risate in registrazioni di Schickele; Johnstone 1994: oltre "
                                   "tre ripetizioni si scivola nel comico). [H] usarlo come limite per ripetizioni identiche e salienti in un "
                                   "mix (non per un groove di sfondo)"),
    "build_halving_units": Param((8, 4, 2, 1), "[A] Margulis 2014 p. 23 e p. 44 (Rameau, Tambourin): restringere l'unità ripetuta "
                                                 "(2 battute -> 1 -> 1/2) dà accelerazione percepita a tempo costante; [H] schema di build "
                                                 "per dimezzamento del loop prima del downbeat d'ingresso"),
    "hypermeter_profile": Param((1.0, 0.25, 0.5, 0.25), "[A] Huron 2006 Fig. 10.4 pp. 179-180 (Haydn op. 54 n. 1): attacchi sui quattro "
                                                          "downbeat di un gruppo di 4 battute forte-debole-medio-debole; [T] p. 184: i downbeat "
                                                          "ipermetrici sono i più attesi e i più gratificanti"),
    "stm_seconds": Param((3.0, 5.0, 12.0), "[E] Huron 2006 p. 228: memoria uditiva a breve termine 3-5 s in media, talvolta 10-12 s; "
                                            "~10 eventi (fino a ~25)"),
    "style_schema_seconds": Param((0.25, 1.0), "[E] Huron 2006 pp. 207-208 (Perrott & Gjerdingen): genere riconosciuto in 250 ms, "
                                                "quasi al massimo dopo 1 s; il timbro porta l'informazione di stile"),
    "schema_switch_seconds": Param(3.0, "[H] durata minima di un blend tra generi lontani (dj_set far_min_blend_s). Contesto [E] Huron "
                                        "2006 p. 211 (Krumhansl & Kessler): una modulazione a tonalità VICINA è 'firmly established' in tre "
                                        "accordi ('a few seconds'), con un residuo; per schemi lontani l'autore congettura ('I doubt ...'). "
                                        "Il libro non prova un minimo di 3 s per un blend (verifica 21/10/2026)"),
    "phrase_reset_bits": Param({"within": 5.5, "last_two": 5.0, "across_boundary": 6.2},
                               "[E] Huron 2006 pp. 154-157 (300 canti popolari): informazione media per coppia di gradi 5,5 bit, "
                               "5,0 nelle ultime due note di frase, 6,2 tra fine di una frase e inizio della successiva; intervallo "
                               "medio 2,0 semitoni dentro la frase e 2,9 tra frasi: il confine di frase è un 'reset'"),
    "first_bar_caesura": Param(12, "[T] Krims 2000 p. 50: nello speech-effusive spesso una cesura dentro il primo verso di un nuovo "
                                   "gruppo di 4 battute; [H] fine verso nella prima battuta del gruppo prima della posizione 12 "
                                   "(BC 3), con la fase ipermetrica stimata dalle fini verso. NON validato su MCFlow (r = 0,03 con lo "
                                   "score speech-effusive, N = 124): solo diagnostica"),
    "npvi_english_speech": Param(57.0, "[E] Huron 2006 pp. 188-189 (Patel & Daniele 2003; Huron & Ollen 2003): l'inglese alterna "
                                       "sillabe lunghe e brevi più del francese e le melodie strumentali inglesi lo riflettono; "
                                       "[H] valore di riferimento del nPVI vocalico dell'inglese parlato ~57 (francese ~44) da Patel "
                                       "& Daniele, citato come ordine di grandezza"),
}

#: Livello di evidenza di ogni regola di BOOK_RULES (audit dei nove libri, 21/10/2026). ``source_level``: che cosa
#: offre la fonte ([E] risultato sperimentale, [T] teoria, [A] analisi o algoritmo di un autore, [P] pratica, [H]
#: nostra ipotesi); ``rule_level``: lo stato della soglia o del valore usato qui (quasi sempre [H]: nessuna regola è
#: validata su ascolti di transizioni). ``verified``: libro e pagina STAMPATA riletti nel testo per pagina
#: (``tools/libri.py``; fisica = stampata + scarto del PDF). Nessuna regola è validata su giudizi di mix.
BOOK_RULES_EVIDENCE: dict[str, dict[str, str]] = {
    "accent_long_beats": {"source_level": "A", "rule_level": "A", "verified": "Ohriner 2019 p. 57 (0,34 battiti), p. 62 (96 %)",
                          "note": "replica dell'algoritmo dell'autore su MCFlow; su audio dipende dagli onset"},
    "groove_classes": {"source_level": "A", "rule_level": "A", "verified": "Ohriner 2019 p. 86 (sette classi)", "note": "combinatoria"},
    "groove_class_share": {"source_level": "E", "rule_level": "E", "verified": "Ohriner 2019 Tab. 4.3 p. 101 (fisica 143)",
                           "note": "corpus di 75 strofe; descrittivo, non una preferenza"},
    "groove_effort_per_bar": {"source_level": "A", "rule_level": "A", "verified": "Ohriner 2019 pp. 96-98", "note": ""},
    "grooviness_min_bars": {"source_level": "A", "rule_level": "A", "verified": "Ohriner 2019 p. 117 e n. 16", "note": ""},
    "syllable_speed_mean": {"source_level": "E", "rule_level": "E", "verified": "Ohriner 2019 p. 106 (5,09 sillabe/s)",
                            "note": "esclude le sillabe finali di verso/frase"},
    "saturation_change_typical": {"source_level": "E", "rule_level": "E", "verified": "Ohriner 2019 p. 108", "note": ""},
    "pillar_late_share": {"source_level": "E", "rule_level": "E", "verified": "Ohriner 2019 p. 110",
                          "note": "pilastro = fini di breath phrase, non confini grammaticali"},
    "krims_styles": {"source_level": "T", "rule_level": "H", "verified": "Krims 2000 pp. 49-52",
                     "note": "spettri sovrapposti, non classi; l'operazionalizzazione con i descrittori è nostra"},
    "four_bar_groups": {"source_level": "T", "rule_level": "H", "verified": "Krims 2000 p. 53; Huron 2006 pp. 179-180; limite Williams 2013 p. 10",
                        "note": "avviso diagnostico; un frammento irregolare può essere una citazione voluta (Williams p. 10)"},
    "loop_establish_repeats": {"source_level": "T+A", "rule_level": "H", "verified": "Margulis 2014 p. 51; Ohriner 2019 p. 117",
                               "note": ""},
    "loop_focal_repeats": {"source_level": "T", "rule_level": "H", "verified": "Margulis 2014 p. 51 (Lidov: categorie)",
                           "note": "categorie teoriche, non un massimo di loop; Margulis p. 8: 1/4/8 sono secondi delle unità, "
                                   "non numeri di copie"},
    "loop_with_new_layer_max_bars": {"source_level": "H", "rule_level": "H", "verified": "Margulis 2014 p. 51 (contesto)",
                                     "note": "nessuna fonte dà 8 battute"},
    "loop_vocal_max_repeats": {"source_level": "E", "rule_level": "H", "verified": "Margulis 2014 pp. 16-17",
                               "note": "speech-to-song con ripetizione 'excessive', nessun numero: il limite di 2 è nostro, solo "
                                       "diagnostico"},
    "loop_comic_repeats": {"source_level": "A", "rule_level": "H", "verified": "Huron 2006 p. 286; Margulis 2014 pp. 173-174",
                           "note": "una registrazione dal vivo di un brano comico; 'norma di tre' come ipotesi dell'autore"},
    "build_halving_units": {"source_level": "A", "rule_level": "H", "verified": "Margulis 2014 p. 23, p. 44", "note": ""},
    "hypermeter_profile": {"source_level": "A+T", "rule_level": "H", "verified": "Huron 2006 Fig. 10.4 pp. 179-180, p. 184",
                           "note": "un brano di Haydn; nessun massimo di loop"},
    "stm_seconds": {"source_level": "E", "rule_level": "E", "verified": "Huron 2006 p. 228 (3-5 s, talvolta 10-12 s)", "note": ""},
    "style_schema_seconds": {"source_level": "E", "rule_level": "E", "verified": "Huron 2006 pp. 207-208 (250 ms, 1 s)",
                             "note": "riconoscimento dello stile, non qualità di un mix"},
    "schema_switch_seconds": {"source_level": "E", "rule_level": "H", "verified": "Huron 2006 p. 211",
                              "note": "tre accordi per tonalità VICINE; generi lontani = congettura; il minimo di 3 s per un blend "
                                      "è nostro"},
    "phrase_reset_bits": {"source_level": "E", "rule_level": "E", "verified": "Huron 2006 pp. 154-157", "note": "canti popolari"},
    "first_bar_caesura": {"source_level": "T", "rule_level": "H", "verified": "Krims 2000 p. 50",
                          "note": "non validata su MCFlow (r = 0,03)"},
    "npvi_english_speech": {"source_level": "E", "rule_level": "H", "verified": "Huron 2006 pp. 188-189 (Patel & Daniele)",
                            "note": "valore di riferimento come ordine di grandezza"},
}

#: [A] Krims 2000 p. 133 (fine dell'unità di due distici senza upbeat) e p. 194 (cesura su BC 2 lascia BC 3 per
#: un upbeat): fonte del campo diagnostico ``krims_closure`` di :func:`exit_point`, un brano per pagina.
#: Krims p. 32 ('the problem of musical closure') riguarda invece la chiusura del discorso INTERPRETATIVO
#: dell'analisi, non una cadenza: non è fonte di questa né di altre regole di cue (verifica 21/10/2026).
KRIMS_CLOSURE_NOTE = ("krims_closure: diagnostic only (Krims 2000 p. 133, p. 194 [A], one song each); never in the exit "
                      "score; Krims p. 32 'musical closure' = closure of interpretive systems, not a cadence: not a source")

# CMU function words (short list, used only with ``deaccent_function_words``) [A] Ohriner p. 61 n. 11:
# accented function words needed manual de-accenting 12,5 times as often as content words.
_FUNCTION_WORDS = _EN_STOP | {"as", "if", "so", "from", "by", "up", "out", "me", "him", "her", "us", "them", "his", "our",
                              "their", "was", "are", "am", "were", "been", "do", "did", "can", "will", "just", "when", "what",
                              "who", "how", "than", "then", "there", "where", "cuz", "cause", "ain't", "gon", "wit", "into", "onto"}


def groove_catalog() -> list[dict[str, Any]]:
    """All distinct accent sets of Ohriner's seven groove classes (class, start position, positions mod 16).

    The start is the position of the first IAI of the class string as written (Ohriner p. 84:
    '<332_332> starting at 4'); classes with internal repetition have fewer distinct rotations.
    """
    out, seen = [], set()
    for cls in BOOK_RULES["groove_classes"].value:
        iai = [int(c) for c in cls.replace("_", "")]
        for s in range(16):
            pos, acc = [], s
            for d in iai:
                pos.append(acc % 16)
                acc += d
            key = frozenset(pos)
            if key in seen:
                continue
            seen.add(key)
            out.append({"class": cls, "start": s, "positions": tuple(sorted(key))})
    return out


_GROOVES: Optional[list[dict[str, Any]]] = None


def _dtw_sym2(a: Sequence[int], b: Sequence[int]) -> float:
    """DTW with the 'symmetric2' step pattern (diagonal steps weigh 2): one accent moved by one sixteenth costs 2."""
    if not a or not b:
        return float(2 * 16 * max(len(a), len(b)))
    inf = float("inf")
    D = [[inf] * (len(b) + 1) for _ in range(len(a) + 1)]
    D[0][0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            c = abs(a[i - 1] - b[j - 1])
            D[i][j] = min(D[i - 1][j - 1] + 2 * c, D[i - 1][j] + c, D[i][j - 1] + c)
    return D[len(a)][len(b)]


def groove_effort(accents: Iterable[int], groove: Iterable[int]) -> float:
    """[A]/[H] Ohriner's effort (pp. 96-97) between an accent set and a groove (positions mod 16).

    Ohriner uses dynamic time warping (R ``dtw``, Giorgino 2009) between position sets; we use
    the same step pattern on the sorted sets, taking the smaller of two cuts of the cycle (at 0 and
    at 8) so that accents near the bar line are not penalised by the wrap [H]."""
    a, g = sorted(set(int(x) % 16 for x in accents)), sorted(set(int(x) % 16 for x in groove))
    best = float("inf")
    for off in (0, 8):
        best = min(best, _dtw_sym2(sorted((x + off) % 16 for x in a), sorted((x + off) % 16 for x in g)))
    return best


def best_groove(accents: Iterable[int]) -> dict[str, Any]:
    """Groove (class, start) with the least effort; ties broken by the class share of Ohriner Tab. 4.3 [E]."""
    global _GROOVES
    if _GROOVES is None:
        _GROOVES = groove_catalog()
    share = BOOK_RULES["groove_class_share"].value
    acc = sorted(set(int(x) % 16 for x in accents))
    scored = sorted(((groove_effort(acc, g["positions"]), -share[g["class"]], g["start"], g) for g in _GROOVES),
                    key=lambda x: x[:3])
    e, _, _, g = scored[0]
    return {"class": g["class"], "start": g["start"], "effort": e, "positions": list(g["positions"]), "accents": acc}


def model_syllables(model: Mapping[str, Any], t0_s: Optional[float] = None, t1_s: Optional[float] = None) -> list[dict[str, Any]]:
    """Syllables of a dj_phrase model on the bar grid (16 positions per bar).

    Syllable onsets: uniform inside each word, last syllable at ``last_syl_on`` (as in
    :func:`analyze_track`) [H]. ``abs16`` = 16 * bar + position, rounded; ``off_grid`` = distance from
    the nearest sixteenth in sixteenths (Ohriner discards the 6,6 % of syllables off the grid, p. 53).
    """
    sr = model["sample_rate"]
    grid = Grid(np.asarray(model["downbeats_s"], dtype=float), float(model["period_s"]))
    out = []
    for wi, w in enumerate(model["words"]):
        if w.get("n_syl") is None or w.get("last_syl_on") is None:
            continue
        n = max(1, int(w["n_syl"]))
        on, last_on, off = float(w["on"]), float(w["last_syl_on"]), float(w["off"])
        if isinstance(w.get("syl_on"), (list, tuple)) and len(w["syl_on"]) == n:  # F5: onsets from the vocal stem
            ons = [float(x) for x in w["syl_on"]]
        else:
            ons = [on + (last_on - on) * k / (n - 1) for k in range(n)] if n > 1 else [last_on if last_on >= on else on]
        for k, t in enumerate(ons):
            ts = t / sr
            if (t0_s is not None and ts < t0_s) or (t1_s is not None and ts >= t1_s):
                continue
            nxt = ons[k + 1] if k + 1 < n else off
            bar, pos = grid.locate(ts)
            if bar < 0:
                continue
            p16 = pos * 4.0
            r = int(round(p16))
            bar_r, pos_r = (bar + 1, r - 16) if r >= 16 else (bar, r)
            out.append({"word_i": wi, "idx": k, "n_syl": n, "stress_idx": int(w.get("stress_idx") or 0),
                        "word": w.get("clean") or Lexicon.clean(w.get("text", "")), "t_s": ts,
                        "bar": bar_r, "pos16": pos_r, "abs16": 16 * bar_r + pos_r, "off_grid": abs(p16 - r),
                        "dur_beats": max(0.0, (nxt - t) / sr / grid.period_s)})
    return out


def ohriner_accents(sylls: Sequence[Mapping[str, Any]], *, long_beats: Optional[float] = None,
                    deaccent_function_words: bool = False) -> list[bool]:
    """[A] Ohriner 2019 pp. 57-59 accent discovery on :func:`model_syllables`-style syllables.

    Thin wrapper over :func:`ohriner_accents_official` (port of AppendAccent.r): the CMU accent of each
    syllable is derived from ``n_syl``/``idx``/``stress_idx`` (primary stress only [H]: CMUdict secondary
    stresses are not used). ``deaccent_function_words`` [H] then moves the accent from an on-grid function
    monosyllable to a following off-grid content word (Ohriner's commonest manual correction, p. 59 and
    p. 61 n. 11); on MCFlow it lowers the agreement with the annotated stress (88,5 % -> 87,2 %): off by default.
    """
    lb = BOOK_RULES["accent_long_beats"].value if long_beats is None else long_beats
    conv = [{"beat_index": s["abs16"] / 4.0, "dur_beats": s["dur_beats"],
             "cmu_accent": None if s["n_syl"] == 1 else (1 if s["idx"] == s["stress_idx"] else 0)} for s in sylls]
    acc = ohriner_accents_official(conv, long_beats=lb)
    if deaccent_function_words:
        n = len(sylls)
        for i in range(n - 1):
            if (acc[i] and sylls[i]["abs16"] % 2 == 0 and sylls[i]["n_syl"] == 1 and sylls[i]["word"] in _FUNCTION_WORDS
                    and sylls[i + 1]["abs16"] % 2 == 1 and sylls[i + 1]["word"] not in _FUNCTION_WORDS
                    and sylls[i + 1]["abs16"] - sylls[i]["abs16"] == 1):
                acc[i], acc[i + 1] = False, True
    return acc


def bar_grooves(sylls: Sequence[Mapping[str, Any]], accents: Sequence[bool], min_syllables: int = 3) -> list[dict[str, Any]]:
    """Per bar: occupied positions (saturation, Ohriner p. 107), accents and the least-effort groove."""
    bars: dict[int, dict[str, Any]] = {}
    for s, a in zip(sylls, accents):
        b = bars.setdefault(s["bar"], {"bar": s["bar"], "positions": set(), "accents": set(), "n": 0})
        b["positions"].add(s["pos16"])
        b["n"] += 1
        if a:
            b["accents"].add(s["pos16"])
    out = []
    for k in sorted(bars):
        b = bars[k]
        row = {"bar": k, "n_syllables": b["n"], "occupied": len(b["positions"]), "saturation": len(b["positions"]) / 16.0,
               "accents": sorted(b["accents"])}
        if b["n"] >= min_syllables and len(b["accents"]) >= 2:
            g = best_groove(b["accents"])
            row.update(groove_class=g["class"], groove_start=g["start"], groove_effort=g["effort"])
        out.append(row)
    return out


def _circ_R(positions16: Sequence[float]) -> Optional[float]:
    if not positions16:
        return None
    ang = np.asarray(positions16, dtype=float) * 2 * np.pi / 16.0
    return float(np.hypot(np.cos(ang).mean(), np.sin(ang).mean()))


def central_pillar(line_end_pos16: Sequence[float], bandwidth16: float = 1.0) -> Optional[dict[str, Any]]:
    """[A] Ohriner 2019 p. 110: 'central pillar' = mode of the circular density of phrase endings.

    Von Mises-like kernel on the 16-position cycle [H: bandwidth one sixteenth]."""
    if not line_end_pos16:
        return None
    grid = np.arange(16)
    d = np.zeros(16)
    for p in line_end_pos16:
        dd = np.minimum(np.abs(grid - p) % 16, 16 - np.abs(grid - p) % 16)
        d += np.exp(-0.5 * (dd / bandwidth16) ** 2)
    d /= d.sum()
    return {"pillar_pos16": int(np.argmax(d)), "density": d.tolist(), "R": _circ_R(line_end_pos16), "n": len(line_end_pos16)}


def flow_features(model: Mapping[str, Any], t0_s: Optional[float] = None, t1_s: Optional[float] = None,
                  *, sylls: Optional[Sequence[Mapping[str, Any]]] = None, accents: Optional[Sequence[bool]] = None,
                  lines: Optional[Sequence[Mapping[str, Any]]] = None) -> dict[str, Any]:
    """Flow descriptors of a span of a track (whole track by default), named after their sources.

    * ``saturation``, ``saturation_change`` [A] Ohriner pp. 107-108 (occupied sixteenths per bar);
    * ``syllable_speed`` [A] Ohriner p. 106 (syllables per second, line-final syllables excluded);
    * ``onbeat_accent`` share of accents on the beat, ``offbeat_onset`` share of onsets on 1mod2,
      ``three_share`` share of 3-IAIs among IAIs of 2-3 (Ohriner p. 116: 'average IAI' as syncopation);
    * ``duple_share``, ``groove_adherence``, ``grooviness``, groove of each bar [A] Ohriner pp. 98, 117;
    * ``end_beat4``, ``end_R``, ``pillar`` of line ends [A] Ohriner pp. 108-110, Krims p. 50 (regular
      on-beat pauses near the 4th beat in the sung style);
    * ``enjambment`` share of bar lines crossed inside a line (Krims p. 50: spilling over the meter);
    * ``rhyme_density`` share of words rhyming 1-2 bars away (Krims p. 51: long chains of rhymes);
    * ``articulation`` sounding / onset-to-onset duration of words (Krims p. 51: 'sharply-attacked
      and crisp' percussion-effusive delivery) [H proxy];
    * ``caesura_rate`` pauses inside lines per bar (Krims p. 51: brief caesuras) [H proxy];
    * ``first_bar_caesura`` (Krims p. 50) with ``hypermeter_phase`` of the four-bar groups [H];
    * ``npvi`` of successive syllables inside lines (Patel & Daniele 2003, Huron pp. 188-189) [E].
    """
    sr = model["sample_rate"]
    per = float(model["period_s"])
    sy = list(sylls) if sylls is not None else model_syllables(model, t0_s, t1_s)
    acc = list(accents) if accents is not None else ohriner_accents(sy)
    words = model["words"]
    lines = list(lines) if lines is not None else list(model.get("lines") or [])
    inside = (lambda t: (t0_s is None or t >= t0_s) and (t1_s is None or t < t1_s))  # noqa: E731
    bg = bar_grooves(sy, acc)
    full = [b for b in bg if "groove_class" in b]
    feats: dict[str, Any] = {"n_syllables": len(sy), "n_bars": len(bg), "n_bars_groove": len(full),
                             "off_grid_mean16": float(np.mean([s["off_grid"] for s in sy])) if sy else None}
    occ = [b["occupied"] for b in bg if b["n_syllables"] >= 1]
    feats["saturation"] = float(np.mean(occ)) / 16.0 if occ else None
    feats["saturation_change"] = float(np.mean(np.abs(np.diff(occ)))) if len(occ) > 1 else None
    # syllable speed: Ohriner p. 106 excludes line/phrase-final syllables
    end_words = {L["last_word"] for L in lines}
    durs = [s["dur_beats"] for s in sy if not (s["word_i"] in end_words and s["idx"] == s["n_syl"] - 1) and s["dur_beats"] > 0]
    feats["syllable_speed"] = float(1.0 / (np.mean(durs) * per)) if durs else None
    pos_acc = [s["abs16"] for s, a in zip(sy, acc) if a]
    feats["onbeat_accent"] = float(np.mean([p % 4 == 0 for p in pos_acc])) if pos_acc else None
    feats["offbeat_onset"] = float(np.mean([s["abs16"] % 2 == 1 for s in sy])) if sy else None
    iai = [b - a for a, b in zip(pos_acc, pos_acc[1:]) if 2 <= b - a <= 3]
    feats["three_share"] = float(np.mean([d == 3 for d in iai])) if iai else None
    feats["duple_share"] = float(np.mean([b["groove_class"] == "2222_2222" for b in full])) if full else None
    lim = BOOK_RULES["groove_effort_per_bar"].value
    pairs = [(x, y) for x, y in zip(full, full[1:]) if y["bar"] == x["bar"] + 1]
    feats["groove_adherence"] = (float(np.mean([groove_effort(y["accents"], best_groove(x["accents"])["positions"]) <= lim
                                                for x, y in pairs])) if pairs else None)
    same = [(x["groove_class"], x["groove_start"]) == (y["groove_class"], y["groove_start"]) and y["groove_effort"] == 0 and x["groove_effort"] == 0
            for x, y in pairs]
    groovy = set()
    for (x, y), ok in zip(pairs, same):
        if ok:
            groovy |= {x["bar"], y["bar"]}
    feats["grooviness"] = (len(groovy) / len(full)) if full else None
    cls = [b["groove_class"] for b in full]
    feats["groove_classes"] = {c: cls.count(c) / len(cls) for c in sorted(set(cls))} if cls else {}
    feats["groove_last"] = ({"class": full[-1]["groove_class"], "start": full[-1]["groove_start"], "bar": full[-1]["bar"],
                             "effort": full[-1]["groove_effort"], "accents": full[-1]["accents"]} if full else None)
    feats["groove_first"] = ({"class": full[0]["groove_class"], "start": full[0]["groove_start"], "bar": full[0]["bar"],
                              "effort": full[0]["groove_effort"], "accents": full[0]["accents"]} if full else None)
    # line ends / starts in the span
    grid = Grid(np.asarray(model["downbeats_s"], dtype=float), per)
    ends, starts, spans, ends_bp = [], [], [], []
    for L in lines:
        w = words[L["last_word"]]
        t_end = (w.get("last_syl_on") or w["on"]) / sr
        if not inside(t_end):
            continue
        bar, pos = grid.locate(t_end)
        if bar >= 0:
            ends.append((pos * 4.0) % 16)
            ends_bp.append((bar, (pos * 4.0) % 16))
        spans.append((L["start_sample"] / sr, L["end_sample"] / sr))
    feats["end_beat4"] = float(np.mean([p >= 11.5 for p in ends])) if ends else None
    feats["end_R"] = _circ_R(ends)
    pil = central_pillar(ends)
    feats["pillar_pos16"] = pil["pillar_pos16"] if pil else None
    feats["n_line_ends"] = len(ends)
    # [T] Krims p. 50 first-bar caesura (speech-effusive) with the hypermetric phase of the line ends [H]
    hp = hypermeter_phase(ends_bp)
    feats["hypermeter_phase"] = hp["phase"]
    feats["hypermeter_phase_share"] = hp["share"]
    feats["first_bar_caesura"] = first_bar_caesura(ends_bp, {s["bar"] for s in sy}, phase=hp["phase"])
    # [E] nPVI of syllable onsets inside lines (Patel & Daniele via Huron pp. 188-189)
    feats["npvi"] = syllable_npvi(sy, line_end_words=end_words)
    dbs = [d for d in model["downbeats_s"] if inside(d)]
    if dbs and spans:
        crossed = [any(s0 < d - per and e0 > d + per for s0, e0 in spans) for d in dbs]
        covered = [any(s0 <= d + per and e0 >= d - per for s0, e0 in spans) for d in dbs]
        n_cov = sum(covered)
        feats["enjambment"] = (sum(crossed) / n_cov) if n_cov else None
    else:
        feats["enjambment"] = None
    ws = [w for w in words if inside(w["on"] / sr)]
    feats["rhyme_density"] = float(np.mean([(w.get("rhyme") or {}).get("score", 0) >= 0.5 for w in ws])) if ws else None
    art = []
    for a, b in zip(ws, ws[1:]):
        ioi = b["on"] - a["on"]
        if ioi > 0:
            art.append(min(1.0, (a["off"] - a["on"]) / ioi))
    feats["articulation"] = float(np.mean(art)) if art else None
    gaps = [g for g in model.get("gaps") or [] if inside(g["end_sample"] / sr)]
    n_bars = max(1, len(dbs)) if dbs else max(1, len(bg))
    feats["caesura_rate"] = sum(1 for g in gaps if g.get("rest") and g["p_end"] < PARAMS["line_end_threshold"].value) / n_bars
    return feats


# z-score reference for the Krims scores: MCFlow per-song distribution (tools/mcflow_flow_stats.py),
# stored in configs/dj_phrase_flow_v1.json; fallback to these rounded values if the file is absent.
FLOW_CONFIG = ROOT / "configs" / "dj_phrase_flow_v1.json"
_KRIMS_FALLBACK_REF = {
    "onbeat_accent": (0.62, 0.10), "end_beat4": (0.55, 0.20), "end_R": (0.55, 0.20), "duple_share": (0.45, 0.20),
    "enjambment": (0.35, 0.20), "saturation": (0.45, 0.10), "offbeat_onset": (0.20, 0.08), "three_share": (0.20, 0.10),
    "articulation": (0.85, 0.10), "caesura_rate": (0.15, 0.15), "groove_adherence": (0.55, 0.20),
    "saturation_change": (2.0, 0.8), "rhyme_density": (0.25, 0.10), "first_bar_caesura": (0.25, 0.15),
}
KRIMS_DEFINITION: dict[str, tuple[tuple[str, int], ...]] = {
    # [T] Krims 2000 p. 50: sung = rhythmic repetition, on-beat (strong-beat) accents, regular on-beat
    # pauses near the 4th beat, strict couplets; [H] operationalisation with the descriptors above.
    "sung": (("onbeat_accent", +1), ("end_beat4", +1), ("end_R", +1), ("duple_share", +1), ("enjambment", -1), ("saturation", -1)),
    # [T] Krims pp. 50-51: off-beat attacks, sharp/crisp staccato delivery, brief caesuras; not necessarily fast.
    "percussion_effusive": (("offbeat_onset", +1), ("three_share", +1), ("articulation", -1), ("caesura_rate", +1)),
    # [T] Krims p. 51: speech-like, little sense of the pulse, irregular polyrhythms, profusion of syllables,
    # long rhyme chains, units spilling over bar lines and four-bar groups.
    "speech_effusive": (("saturation", +1), ("enjambment", +1), ("end_R", -1), ("groove_adherence", -1),
                        ("saturation_change", +1), ("rhyme_density", +1)),
    # first_bar_caesura (Krims p. 50) is reported by flow_features but NOT used: on MCFlow it does not track the
    # speech-effusive score computed without it (r = 0,03, N = 124) nor the release year (rho = -0,11):
    # runs/dj_phrase_v2/mcflow_books_eval.json 'books_v2_checks'.
}


def _flow_ref() -> dict[str, tuple[float, float]]:
    try:
        cfg = json.loads(FLOW_CONFIG.read_text(encoding="utf-8"))
        return {k: (float(v["mean"]), float(v["sd"])) for k, v in cfg["krims_reference"].items()}
    except (OSError, KeyError, ValueError):
        return dict(_KRIMS_FALLBACK_REF)


def krims_style(features: Mapping[str, Any], ref: Optional[Mapping[str, tuple[float, float]]] = None) -> dict[str, Any]:
    """[T] Krims's three flow styles as graded scores (z-means of the descriptors, sign by definition) [H].

    ``scores`` are unbounded (0 = MCFlow song average); ``membership`` is a softmax over them, read as
    'which direction dominates' (Krims: spectra, not exclusive classes). Missing descriptors are skipped.
    """
    r = dict(ref or _flow_ref())
    scores, used = {}, {}
    for style, terms in KRIMS_DEFINITION.items():
        zs = []
        for name, sign in terms:
            v = features.get(name)
            if v is None or name not in r or not np.isfinite(v):
                continue
            mu, sd = r[name]
            zs.append(sign * (float(v) - mu) / max(sd, 1e-6))
        scores[style] = float(np.mean(zs)) if zs else None
        used[style] = len(zs)
    valid = {k: v for k, v in scores.items() if v is not None}
    if valid:
        m = max(valid.values())
        ex = {k: math.exp(v - m) for k, v in valid.items()}
        tot = sum(ex.values())
        memb = {k: ex[k] / tot for k in valid}
        top = max(memb, key=memb.get)
    else:
        memb, top = {}, None
    return {"scores": scores, "membership": memb, "dominant": top, "n_descriptors": used,
            "source": "[T] Krims 2000 pp. 49-52; [H] operazionalizzazione con descrittori di Ohriner 2019"}


def flow_window(model: Mapping[str, Any], downbeat_s: float, side: str, bars: int = 4) -> dict[str, Any]:
    """Flow of A's last ``bars`` bars before its exit (side 'a') or of B's first ``bars`` after its entry (side 'b')."""
    bar_s = 4.0 * float(model["period_s"])
    t0, t1 = (downbeat_s - bars * bar_s, downbeat_s) if side == "a" else (downbeat_s, downbeat_s + bars * bar_s)
    f = flow_features(model, t0, t1)
    f["style"] = krims_style(f)
    f["window_s"] = [t0, t1]
    return f


def flow_compatibility(fa: Mapping[str, Any], fb: Mapping[str, Any]) -> dict[str, Any]:
    """[H] Continuity of the flow from A's exit window to B's entry window (both from :func:`flow_window`).

    Components (each in [0, 1], 1 = same): density (saturation ratio; Ohriner pp. 16-17: changing the
    number of syllables per bar is the commonest 'flip' of the flow), speed (syllable-speed ratio, Ohriner
    p. 106), groove (same class and start = 1, same class = 0,6, both with 3-IAIs = 0,3; Ohriner ch. 4),
    style (1 - total-variation distance of the Krims memberships). ``continuity`` = mean of the available
    components. A low value is a *flip*, not an error: a DJ may want the contrast. Diagnostic only
    (AGENTS.md: psychological modules do not drive the selection).
    """
    comp: dict[str, Optional[float]] = {}
    sa, sb = fa.get("saturation"), fb.get("saturation")
    comp["density"] = (min(sa, sb) / max(sa, sb)) if sa and sb else None
    va, vb = fa.get("syllable_speed"), fb.get("syllable_speed")
    comp["speed"] = (min(va, vb) / max(va, vb)) if va and vb else None
    ga, gb = fa.get("groove_last"), fb.get("groove_first")
    if ga and gb:
        if ga["class"] == gb["class"] and ga["start"] == gb["start"]:
            comp["groove"] = 1.0
        elif ga["class"] == gb["class"]:
            comp["groove"] = 0.6
        elif ga["class"] != "2222_2222" and gb["class"] != "2222_2222":
            comp["groove"] = 0.3
        else:
            comp["groove"] = 0.0
    else:
        comp["groove"] = None
    ma, mb = (fa.get("style") or {}).get("membership") or {}, (fb.get("style") or {}).get("membership") or {}
    if ma and mb:
        comp["style"] = 1.0 - 0.5 * sum(abs(ma.get(k, 0) - mb.get(k, 0)) for k in set(ma) | set(mb))
    else:
        comp["style"] = None
    vals = [v for v in comp.values() if v is not None]
    cont = float(np.mean(vals)) if vals else None
    flips = []
    if comp["density"] is not None and comp["density"] < 0.67:
        flips.append("densità sillabica")
    if comp["groove"] == 0.0:
        flips.append("groove (duplo contro terzine)")
    if comp["style"] is not None and comp["style"] < 0.5:
        flips.append(f"stile Krims ({(fa.get('style') or {}).get('dominant')} -> {(fb.get('style') or {}).get('dominant')})")
    return {"continuity": cont, "components": comp, "flips": flips,
            "a_style": (fa.get("style") or {}).get("dominant"), "b_style": (fb.get("style") or {}).get("dominant"),
            "source": "[H] su Ohriner 2019 (flip del flow pp. 16-28, groove cap. 4) e Krims 2000 pp. 49-52"}


#: loop limits with no number in any source (BOOK_RULES_EVIDENCE): dropped by ``include_hypotheses=False``
LOOP_HYPOTHESIS_ONLY = ("loop_with_new_layer_max_bars", "loop_vocal_max_repeats")


def loop_repetition_check(loop_bars: int, bars: int, *, loop_has_vocal: bool = False, new_layer_entering: bool = True,
                          include_hypotheses: bool = True) -> dict[str, Any]:
    """Diagnostic warnings for repeating a loop (for ``dj_creative.loop_extend`` / ``halving_build``), with sources.

    ``repeats`` = bars / loop_bars. Each finding carries its evidence level (BOOK_RULES_EVIDENCE); **all are
    diagnostic**: none of these thresholds is a validated limit for DJ loops (audit 21/10/2026).
    * fewer than ``loop_establish_repeats`` repeats: the loop is not yet heard as a loop [T] + [H] threshold;
    * no new layer and more than ``loop_focal_repeats`` repeats: focal/static repetition [T] Lidov's categories
      (Margulis p. 51) + [H] threshold;
    * no new layer, a short fragment (<= 2 bars) and more than ``loop_comic_repeats``: 'broken record' [A] (Huron
      p. 286, one live recording) + [H];
    * a new layer entering (B) and more than ``loop_with_new_layer_max_bars`` bars [H];
    * words in the loop and more than ``loop_vocal_max_repeats`` repeats: [H] only (Margulis pp. 16-17 give no
      number of repetitions);
    * total span not a multiple of 4 bars [T] (Krims p. 53, Huron pp. 179-180); limit: Williams p. 10, an
      irregular fragment may be an intended quotation.

    ``include_hypotheses=False`` drops the findings whose number comes from no source (:data:`LOOP_HYPOTHESIS_ONLY`:
    the new-layer and vocal-repeat limits). ``ok`` = no finding (unchanged contract for existing callers);
    ``warnings`` = the finding texts, each prefixed by its level.
    """
    R = {k: p.value for k, p in BOOK_RULES.items()}
    reps = bars / max(1, loop_bars)
    findings: list[dict[str, Any]] = []

    def add(code: str, rule: str, text: str) -> None:
        ev = BOOK_RULES_EVIDENCE[rule]
        if not include_hypotheses and rule in LOOP_HYPOTHESIS_ONLY:
            return
        level = "H" if "H" in ev["rule_level"].split("+") else ev["rule_level"]
        findings.append({"code": code, "rule": rule, "level": level, "source_level": ev["source_level"],
                         "diagnostic_only": True, "text": f"[{level}] {text}", "verified": ev["verified"]})

    if reps < R["loop_establish_repeats"]:
        add("not_established", "loop_establish_repeats",
            f"{reps:g} ripetizioni: sotto {R['loop_establish_repeats']} il loop non si stabilisce")
    if not new_layer_entering and reps > R["loop_focal_repeats"]:
        add("focal", "loop_focal_repeats",
            f"{reps:g} ripetizioni senza un nuovo strato: oltre {R['loop_focal_repeats']} la ripetizione tende al focale/statico (Lidov)")
    elif not new_layer_entering and loop_bars <= 2 and reps > R["loop_comic_repeats"]:
        add("broken_record", "loop_comic_repeats",
            f"{reps:g} ripetizioni identiche di un frammento breve senza un nuovo strato: oltre "
            f"{R['loop_comic_repeats']} rischio 'disco rotto' (Huron p. 286)")
    if new_layer_entering and bars > R["loop_with_new_layer_max_bars"]:
        add("long_with_new_layer", "loop_with_new_layer_max_bars",
            f"{bars} battute di loop anche con B che entra: oltre {R['loop_with_new_layer_max_bars']} (un'unità ipermetrica)")
    if loop_has_vocal and reps > R["loop_vocal_max_repeats"]:
        add("vocal_repeats", "loop_vocal_max_repeats",
            f"parole di A ripetute {reps:g} volte: oltre {R['loop_vocal_max_repeats']} il testo potrebbe cambiare senso (ipotesi)")
    if bars % R["four_bar_groups"]:
        add("off_four_bar_grid", "four_bar_groups",
            f"{bars} battute: non multiplo di {R['four_bar_groups']} (raggruppamento a quattro battute; può essere voluto)")
    keys = ("loop_establish_repeats", "loop_focal_repeats", "loop_comic_repeats", "loop_with_new_layer_max_bars",
            "loop_vocal_max_repeats", "four_bar_groups")
    return {"ok": not findings, "repeats": reps, "warnings": [f["text"] for f in findings], "findings": findings,
            "diagnostic_only": True, "limits": {k: R[k] for k in keys},
            "sources": {k: BOOK_RULES[k].source for k in keys},
            "evidence": {k: BOOK_RULES_EVIDENCE[k] for k in keys}}


# ------------------------------------------- Ohriner 2019: port of flowBook
# Reimplementazione in Python del modello di Ohriner (2019) dal codice ufficiale
# github.com/mohriner/flowBook @ faaec23176acf7af339fdda5b119f4aa586bbbda (R; nessuna licenza
# dichiarata: il codice NON è copiato, è riscritto seguendo l'algoritmo; verificato sui dati
# derivati dell'autore con tools/ohriner_port_check.py). File R di riferimento tra parentesi.
#
# Unità: "beat index" in battiti dall'inizio della strofa/brano (i pickup sono negativi);
# posizioni in sedicesimi = round(4 * beat index).

OHRINER_REPO = "github.com/mohriner/flowBook@faaec23176acf7af339fdda5b119f4aa586bbbda"
# book notation <...> -> class string used in the author's data (3s packed left = lexicographic max)
GROOVE_CLASS_LABEL = {"22222222": "<2222_2222>", "3322222": "<332_2222>", "3232222": "<323_2222>", "332332": "<332_332>",
                      "333322": "<333322>", "3223222": "<3223222>", "333232": "<333232>"}
# [A] Ohriner Tab. 4.2 p. 91 (GetGrooveComplexity.r): complexity of a beat-accent type, from -log prevalence
BEAT_ACCENT_COMPLEXITY = {"1010": 0.0, "0010": 0.744, "1001": 0.721, "0100": 0.941, "0101": 1.0}
EFFORT_RATES = (0.0, 0.0625, 0.125, 0.25)  # [A] p. 98: no swap, one per bar, one per half bar, one per beat
GROOVE_SEARCH_RATE_LIMIT = 0.25  # [A] GetVerseGrooves.r: instances kept if effort / length <= 0.25
GROOVE_LONG_UNITS = 32  # [A] GroovinessFeatureFunctions.R: a groove counts if it lasts >= 2 bars (32 sixteenths)


def _groove_dsegs() -> list[str]:
    """All 2/3 duration strings of 6 or 7 IAIs summing to 16, plus the duple string ('groove dsegs (prime form).r')."""
    out = []
    for n in (6, 7):
        for k in range(2 ** n):
            s = "".join(str(int(b) + 2) for b in format(k, f"0{n}b"))
            if sum(int(c) for c in s) == 16:
                out.append(s)
    return ["22222222"] + out


def _rotations(s: str) -> list[str]:
    return [s[i:] + s[:i] for i in range(len(s))]


def groove_prime_form(pattern: str) -> str:
    """Prime form = the rotation that is lexicographically largest (3s packed left), as in the author's data."""
    return max(_rotations(pattern))


def _prime_displacement(pattern: str) -> int:
    pf = groove_prime_form(pattern)
    k = (pattern + pattern).find(pf)
    return sum(int(c) for c in pattern[:k])


def ohriner_syllables_from_model(model: Mapping[str, Any], t0_s: Optional[float] = None, t1_s: Optional[float] = None) -> list[dict[str, Any]]:
    """Syllables of a dj_phrase model in Ohriner's representation (beat index, duration, CMU accent, endings)."""
    sy = model_syllables(model, t0_s, t1_s)
    line_end_words = {L["last_word"] for L in model.get("lines") or []}
    out = []
    for s in sy:
        last = s["idx"] == s["n_syl"] - 1
        out.append({"beat_index": s["abs16"] / 4.0, "dur_beats": s["dur_beats"],
                    "cmu_accent": None if s["n_syl"] == 1 else (1 if s["idx"] == s["stress_idx"] else 0),
                    "line_end": bool(last and s["word_i"] in line_end_words), "breath_end": False,
                    "word_i": s["word_i"], "t_s": s["t_s"], "off_grid": s["off_grid"]})
    return out


def ohriner_accents_official(sylls: Sequence[Mapping[str, Any]], long_beats: float = 0.34) -> list[bool]:
    """[A] Accent discovery as in AppendAccent.r (Ohriner pp. 57-59).

    b = on the eighth-note grid; w = CMU primary accent of a polysyllable; n = CMU non-accent;
    m = monosyllable; l = longer than 0,34 beats. (1) accent if b and (w or m or (n and l)), or if
    not b and (w or l); (2) clear accents within a sixteenth of a word accent or of an off-grid long
    monosyllable; (3) five syllables on adjacent sixteenths accented only at the ends -> accent the middle.
    """
    n = len(sylls)
    bi = [round(4 * float(s["beat_index"])) for s in sylls]  # in sixteenths
    b = [x % 2 == 0 for x in bi]
    w = [s.get("cmu_accent") == 1 for s in sylls]
    nn = [s.get("cmu_accent") == 0 for s in sylls]
    m = [s.get("cmu_accent") is None for s in sylls]
    l_ = [float(s["dur_beats"]) > long_beats for s in sylls]
    acc = [bool((b[i] and (w[i] or m[i] or (nn[i] and l_[i]))) or ((not b[i]) and (w[i] or l_[i]))) for i in range(n)]
    anchors = {bi[i] for i in range(n) if w[i] or ((not b[i]) and m[i] and l_[i])}
    near = {x for a in anchors for x in (a - 1, a, a + 1)} - anchors
    acc = [False if bi[i] in near else acc[i] for i in range(n)]
    add = []
    for j in range(n - 4):
        if [bi[j + k] - bi[j] for k in range(5)] == [0, 1, 2, 3, 4] and acc[j:j + 5] == [True, False, False, False, True]:
            add.append(j + 2)
    for j in add:
        acc[j] = True
    return acc


def l3_rhythm_vector(sylls: Sequence[Mapping[str, Any]], accents: Sequence[bool]) -> list[int]:
    """[A] GetL3RhythmVector (GetL3Durations.R): 'limit-3' accents (p. 94).

    Accented sixteenths are 1, their neighbours 0; in each beat an undetermined first or third position
    becomes 1 (silent 0mod2 positions not adjacent to an accent are heard as accents); pickups dropped.
    """
    pos_a = sorted({round(4 * s["beat_index"]) for s, a in zip(sylls, accents) if a and s["beat_index"] >= 0})
    pos_n = sorted({round(4 * s["beat_index"]) for s, a in zip(sylls, accents) if (not a) and s["beat_index"] >= 0})
    if not pos_a and not pos_n:
        return []
    lim = max(pos_a[-1] if pos_a else -1, pos_n[-1] if pos_n else -1) + 1
    l3: list[Optional[int]] = [None] * lim
    for p in pos_a:
        l3[p] = 1
    for p in pos_a:
        for q in (p - 1, p + 1):
            if 0 <= q < lim:
                l3[q] = 0
    if l3[0] == 0:
        l3[0] = 1
    nb = -(-lim // 4)
    l3 += [None] * (nb * 4 - lim)
    for k in range(nb):
        row = l3[4 * k:4 * k + 4]
        if all(v is None for v in row):
            continue  # (only with a measure range in the original)
        if row[0] is None:
            l3[4 * k] = 1
        if row[2] is None:
            l3[4 * k + 2] = 1
    return [int(v) if v is not None else 0 for v in l3]


def l3_durations(sylls: Sequence[Mapping[str, Any]], accents: Sequence[bool]) -> str:
    """[A] GetL3Durations: string of limit-3 inter-accent durations (4 -> 22)."""
    v = l3_rhythm_vector(sylls, accents)
    ones = [i for i, x in enumerate(v) if x == 1]
    return "".join(str(b - a) for a, b in zip(ones, ones[1:])).replace("4", "22")


def _dtw_sym2_np(a: np.ndarray, b: np.ndarray) -> float:
    """DTW distance, 'symmetric2' step pattern (R dtw default), |x - y| local cost."""
    n, m = len(a), len(b)
    cost = np.abs(a[:, None] - b[None, :]).astype(float)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ci = cost[i - 1]
        row = D[i]
        prev = D[i - 1]
        for j in range(1, m + 1):
            c = ci[j - 1]
            row[j] = min(prev[j - 1] + 2 * c, prev[j] + c, row[j - 1] + c)
    return float(D[n, m])


def groove_instances(l3d: str, *, rate_limit: float = GROOVE_SEARCH_RATE_LIMIT, max_units: int = 64) -> list[dict[str, Any]]:
    """[A] Get.G.dseg.instances / GetVerseGrooves: every span where a groove string, repeated 1..4 times,
    can be heard with effort/length <= ``rate_limit``; effort = DTW (symmetric2) between the groove's
    accent positions and the actual limit-3 accents of the span, which must start and end on accents.
    ``adj_length`` = length * (1 - rate ** 0.25) (p. 97)."""
    if not l3d:
        return []
    x = np.concatenate([[0], np.cumsum([int(c) for c in l3d])])
    xs = set(int(v) for v in x)
    out = []
    for order, pattern in enumerate(_groove_dsegs()):
        pf = groove_prime_form(pattern)
        disp = _prime_displacement(pattern)
        base = [int(c) for c in pattern]
        for reps in range(1, -(-max_units // 16) + 1):
            pat = np.concatenate([[0], np.cumsum(base * reps)])
            total = int(pat[-1])
            for k, i in enumerate(x):
                i = int(i)
                if i + total not in xs:
                    continue
                j = int(np.searchsorted(x, i + total))
                seg = x[k:j + 1] - i
                eff = _dtw_sym2_np(pat.astype(float), seg.astype(float))
                rate = round(eff / total, 2)
                if rate > rate_limit:
                    continue
                rot = (i + disp) % 16
                if pf == "22222222":
                    rot %= 2
                elif pf == "332332":
                    rot %= 8
                out.append({"effort": eff, "start": i, "end": i + total, "rate": rate, "reps": reps, "length": 16 * reps,
                            "adj_length": round(16 * reps * (1 - rate ** 0.25), 2), "groove": pattern, "class": pf,
                            "rotation": int(rot), "order": order})
    return out


def groove_segments(instances: Sequence[Mapping[str, Any]], max_unit: Optional[int] = None,
                    rates: Sequence[float] = EFFORT_RATES) -> dict[float, list[dict[str, Any]]]:
    """[A] SegmentVerseGrooves: at each effort rate, the groove with the longest adjusted length at every
    sixteenth (ties: first found), then run-length segments (class, rotation, start, end, length)."""
    out = {}
    for lim in rates:
        g1 = [g for g in instances if g["rate"] <= lim and (max_unit is None or g["end"] <= max_unit)]
        if not g1:
            out[lim] = []
            continue
        last = max(g["end"] for g in g1)
        lab: list[Optional[tuple]] = []
        for jpos in range(0, last + 1):
            cand = [g for g in g1 if g["start"] <= jpos <= g["end"]]
            if not cand:
                lab.append(None)
                continue
            best = max(g["adj_length"] for g in cand)
            gsel = next(g for g in cand if g["adj_length"] == best)
            lab.append((gsel["class"], gsel["rotation"]))
        segs, s0 = [], 0
        for jpos in range(1, len(lab) + 1):
            if jpos == len(lab) or lab[jpos] != lab[s0]:
                if lab[s0] is not None:
                    segs.append({"class": lab[s0][0], "rotation": lab[s0][1], "start": s0, "end": jpos - 1, "length": jpos - s0})
                s0 = jpos
        out[lim] = segs
    return out


def ohriner_global_features(sylls: Sequence[Mapping[str, Any]], accents: Optional[Sequence[bool]] = None, *,
                            tempo_bpm: Optional[float] = None, segments: Optional[Mapping[float, Sequence[Mapping[str, Any]]]] = None
                            ) -> dict[str, Any]:
    """[A] Global features of a verse as in 'Corpus Feature Generation.R' and GroovinessFeatureFunctions.R.

    ``saturation`` = mean number of occupied sixteenths per bar (bar = beat index // 4); ``sat_delta`` =
    mean |change| bar to bar; ``syls_per_second`` from syllables that do not end a line/breath;
    ``grooviness`` = share of effort-0 segments lasting >= 32 sixteenths; ``adherence`` = for segments
    >= 32 at one swap per half bar, the covered share by effort-0 segments of the same class;
    ``dupleness`` = share of the verse in effort-0 <2222_2222>; ``groove_complexity`` = mean complexity
    (Tab. 4.2) of the groove of each segment at one swap per bar, weighted by length.
    """
    acc = list(accents) if accents is not None else ohriner_accents_official(sylls)
    sixt = sorted({round(4 * s["beat_index"]) for s in sylls})
    bars: dict[int, int] = {}
    for p in sixt:
        bars[p // 16] = bars.get(p // 16, 0) + 1
    counts = [bars[k] for k in sorted(bars)]
    feats: dict[str, Any] = {"n_syllables": len(sylls), "n_bars": len(counts),
                             "saturation": float(np.mean(counts)) if counts else None,
                             "sat_delta": float(np.mean(np.abs(np.diff(counts)))) if len(counts) > 1 else None,
                             "syls_per_second": None}
    if tempo_bpm:
        secs = [60.0 / tempo_bpm * s["dur_beats"] for s in sylls if not (s.get("line_end") or s.get("breath_end")) and s["dur_beats"] > 0]
        feats["syls_per_second"] = (1.0 / float(np.mean(secs))) if secs else None
    if segments is None:
        inst = groove_instances(l3_durations(sylls, acc))
        segments = groove_segments(inst, max_unit=int(round(4 * max(s["beat_index"] for s in sylls))) if sylls else None)
    seg0 = segments.get(0.0) or []
    tot0 = sum(s["length"] for s in seg0)
    feats["grooviness"] = (sum(s["length"] for s in seg0 if s["length"] >= GROOVE_LONG_UNITS) / tot0) if tot0 else None
    end0 = max((s["end"] for s in seg0), default=0)
    feats["dupleness"] = (sum(s["length"] for s in seg0 if s["class"] == "22222222") / end0) if end0 else None
    long_half = [s for s in segments.get(0.125) or [] if s["length"] >= GROOVE_LONG_UNITS]
    if long_half:
        vals = []
        for g in long_half:
            covered = 0
            for z in seg0:
                if z["class"] != g["class"]:
                    continue
                if (g["start"] >= z["start"] >= g["end"]) or (g["start"] <= z["end"] <= g["end"]):
                    covered += min(z["end"], g["end"]) - max(z["start"], g["start"])
            vals.append(covered / g["length"])
        feats["adherence"] = float(np.mean(vals))
    else:
        feats["adherence"] = 0.0
    seg1 = segments.get(0.0625) or []
    if seg1:
        cx = [(groove_complexity(s["class"], s["rotation"]), s["length"]) for s in seg1]
        feats["groove_complexity"] = float(sum(c * l for c, l in cx) / sum(l for _, l in cx))
    else:
        feats["groove_complexity"] = None
    feats["groove_class_share_effort0"] = ({c: sum(s["length"] for s in seg0 if s["class"] == c) / tot0
                                            for c in sorted({s["class"] for s in seg0})} if tot0 else {})
    feats["segments"] = {str(k): v for k, v in segments.items()}
    return feats


def groove_onsets(cls: str, rotation: int) -> list[int]:
    """Accent positions (0-15) of a groove class in a rotation (RotateGroovePF in GrooveSwapDistance.r)."""
    pat = cls if len(cls) >= 6 else {"22": "22222222", "332": "332332"}[cls]
    pos, acc = [], int(rotation)
    for d in pat:
        pos.append(acc % 16)
        acc += int(d)
    return sorted(pos)


def groove_complexity(cls: str, rotation: int) -> float:
    """[A] GetGrooveComplexity.r: sum over the four beats of the complexity of their beat-accent type (max 4)."""
    on = set(groove_onsets(cls, rotation))
    tot = 0.0
    for b in range(4):
        key = "".join("1" if 4 * b + k in on else "0" for k in range(4))
        tot += BEAT_ACCENT_COMPLEXITY.get(key, 0.0)
    return round(tot, 2)


def groove_swap_distance(cls1: str, rot1: int, cls2: str, rot2: int) -> float:
    """[A] GrooveSwapDistance.r: DTW between the accent positions of two grooves (0 = same groove)."""
    a = np.asarray(groove_onsets(cls1, rot1), float)
    b = np.asarray(groove_onsets(cls2, rot2), float)
    return _dtw_sym2_np(a, b)


def central_pillar_vonmises(positions16: Sequence[float], kappa: float = 25.0) -> Optional[float]:
    """[A] CentralPillarFunctions.r: peak of the von Mises kernel density (bw = 25) of line-end positions (0-16)."""
    if not len(positions16):
        return None
    ang = np.asarray(positions16, float) / 16.0 * 2 * np.pi
    grid = np.arange(512) / 512.0 * 2 * np.pi
    dens = np.exp(kappa * np.cos(grid[:, None] - ang[None, :])).sum(axis=1)
    return float(np.argmax(dens) / 512.0 * 16.0)


# ------------------------------- libri: Krims cap. 4-5, Huron 2006, Margulis 2014 (aggiunte 21/10/2026)
# Funzioni pure, diagnostiche o di supporto per dj_creative/dj_policy; nessuna cambia la selezione v1.

def syllable_npvi(sylls: Sequence[Mapping[str, Any]], *, line_end_words: Iterable[int] = (),
                  times: str = "onset") -> Optional[float]:
    """[E] nPVI of successive syllable durations (Patel & Daniele 2003, in Huron 2006 pp. 188-189).

    nPVI = 100 / (m - 1) * sum |d_k - d_{k+1}| / ((d_k + d_{k+1}) / 2) over pairs of consecutive
    syllables of the same line (pairs across a line end, given by the word index of line-final words,
    are skipped). ``times`` 'onset' uses onset-to-onset intervals (``t_s`` in seconds, or ``beat_index``),
    'duration' the annotated ``dur_beats``. [H] On audio, syllables inside a word are uniform
    (:func:`model_syllables`), which biases nPVI downwards: compare audio values with MCFlow values
    computed the same way (tools/mcflow_flow_stats.py reports both).
    """
    ends = set(line_end_words)
    seq = list(sylls)
    if times == "duration":
        d = [float(s["dur_beats"]) for s in seq]
    else:
        key = "t_s" if seq and "t_s" in seq[0] else "beat_index"
        d = [float(b[key]) - float(a[key]) for a, b in zip(seq, seq[1:])] + [float("nan")]
    terms = []
    for k in range(len(seq) - 1):
        if seq[k].get("word_i") in ends and seq[k].get("idx", 0) == seq[k].get("n_syl", 1) - 1:
            continue  # line-final syllable: the next one starts a new line
        if k + 1 < len(seq) and seq[k + 1].get("word_i") in ends and seq[k + 1].get("idx", 0) == seq[k + 1].get("n_syl", 1) - 1 \
                and times != "duration":
            continue  # the onset interval of a line-final syllable includes the pause
        a, b = d[k], d[k + 1]
        if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
            continue
        terms.append(abs(a - b) / ((a + b) / 2.0))
    return float(100.0 * np.mean(terms)) if terms else None


def hypermeter_phase(ends: Sequence[tuple[int, float]], group: int = 4) -> dict[str, Any]:
    """[H] Phase of the 4-bar grouping from line ends ``(bar, pos16)``: the phase that puts the most
    late line ends (pos16 >= 8) in the last bar of a group (Krims pp. 52-53: four-bar groups; p. 98:
    events at the start of a group or in its last bar). Ties: the smaller phase."""
    if not ends:
        return {"phase": 0, "share": None, "n_ends": 0}
    counts = []
    for ph in range(group):
        counts.append(sum(1 for b, q in ends if (b - ph) % group == group - 1 and q >= 8))
    best = int(np.argmax(counts))
    late = sum(1 for _, q in ends if q >= 8)
    return {"phase": best, "share": (counts[best] / late) if late else None, "n_ends": len(ends), "counts": counts}


def first_bar_caesura(ends: Sequence[tuple[int, float]], active_bars: Iterable[int], *, phase: Optional[int] = None,
                      group: int = 4, before16: Optional[int] = None) -> Optional[float]:
    """[T] Krims 2000 p. 50 (speech-effusive): share of 4-bar groups with a line end inside their first bar,
    before position ``before16`` (default BOOK_RULES['first_bar_caesura'] = 12, i.e. before beat 4).
    Only groups with vocal activity (``active_bars``) count. ``phase`` from :func:`hypermeter_phase`."""
    lim = BOOK_RULES["first_bar_caesura"].value if before16 is None else before16
    ph = hypermeter_phase(ends, group)["phase"] if phase is None else phase
    groups = {(b - ph) // group for b in active_bars if b >= 0}
    if not groups:
        return None
    hit = {(b - ph) // group for b, q in ends if b >= 0 and (b - ph) % group == 0 and q < lim}
    return len(hit & groups) / len(groups)


def hypermetric_weight(bar_in_group: int) -> float:
    """[A] Huron 2006 Fig. 10.4 pp. 179-180: strong-weak-medium-weak profile of the downbeats of a 4-bar group."""
    return float(BOOK_RULES["hypermeter_profile"].value[int(bar_in_group) % 4])


def vocal_phase_alignment(model: Mapping[str, Any], t0_s: Optional[float] = None, t1_s: Optional[float] = None, *,
                          onsets_s: Optional[Sequence[float]] = None,
                          max_phase_beats: float = 0.125, step_beats: float = 1.0 / 60.0,
                          burs: Sequence[float] = tuple(np.round(np.arange(1.0, 2.01, 0.05), 2))) -> dict[str, Any]:
    """[A] Ohriner 2019 ch. 8 (pp. 185-196, code 'OptimizeAlignment'), simplified port: how far the vocal
    onsets sit from the sixteenth grid, and how much a constant phase shift and a swing ratio (BUR,
    Benadon 2006: only the second sixteenth of each eighth moves) explain.

    Limit (runs/dj_phrase_v2/compare.json): with Whisper word onsets the median phase is 0 and only ~5 %
    of the misalignment is explained on our tracks, so the result is diagnostic until onsets come from
    the vocal stem; the method recovers lag and swing on exact onsets (tests).

    Onsets = acoustic word onsets and last-syllable onsets of the model (continuous time). Misalignment is
    in sixtieths of a beat ('minutes', Ohriner p. 185). ``phase_beats`` > 0 means the voice is *late*.
    Ties go to the most neutral adjustment (phase 0, BUR 1). Ohriner's genre corpus: optimal phase
    M = -6,84 minutes (his sign: shift applied to the voice), two thirds of bars not improved by swing (p. 195).
    """
    sr = model["sample_rate"]
    per = float(model["period_s"])
    grid = Grid(np.asarray(model["downbeats_s"], dtype=float), per)
    on = []
    stem_based = onsets_s is not None or any(w.get("syl_src") for w in model["words"])
    if onsets_s is not None:  # F5: raw stem p-centres inside the sung words (no ASR assignment)
        sr_ = model["sample_rate"]
        spans = sorted((w["on"] / sr_ - 0.06, w["off"] / sr_) for w in model["words"])
        starts = np.array([a for a, _ in spans])
        for t in onsets_s:
            k = int(np.searchsorted(starts, t, side="right")) - 1
            if k >= 0 and t < spans[k][1] and (t0_s is None or t >= t0_s) and (t1_s is None or t < t1_s):
                bar, pos = grid.locate(float(t))
                if bar >= 0:
                    on.append(bar * 4.0 + pos)
    for w in ([] if onsets_s is not None else model["words"]):
        if stem_based:  # F5: only the onsets measured on the vocal stem (Ohriner p. 187), never the uniform guesses
            pts = [t for t, s in zip(w.get("syl_on") or [], w.get("syl_src") or []) if s == "stem"]
        else:
            pts = list({w["on"], w.get("last_syl_on", w["on"])})
        for t in pts:
            ts = t / sr
            if (t0_s is None or ts >= t0_s) and (t1_s is None or ts < t1_s):
                bar, pos = grid.locate(ts)
                if bar >= 0:
                    on.append(bar * 4.0 + pos)
    res: dict[str, Any] = {"n_onsets": len(on), "unit": "1/60 di battito", "source": "[A] Ohriner 2019 cap. 8 (port semplificato)",
                           "onsets": "vocal stem p-centre (stem_syllable_onsets)" if stem_based else "ASR word onsets"}
    if len(on) < 8:
        return res
    x = np.asarray(sorted(on))

    def mis(xx: np.ndarray, bur: float = 1.0) -> float:
        # grid inside each eighth: 0 and bur / (1 + bur) * 0.5 beat
        e = np.floor(xx / 0.5) * 0.5
        r = xx - e
        cand = np.stack([np.abs(r), np.abs(r - bur / (1 + bur) * 0.5), np.abs(r - 0.5)])
        return float(np.mean(cand.min(axis=0)) * 60.0)

    base = mis(x)
    nph = int(round(max_phase_beats / step_beats))
    phases = np.arange(-nph, nph + 1) * step_beats  # symmetric, includes 0
    pm = [(round(mis(x - ph), 9), abs(ph), ph) for ph in phases]
    best_m, _, best_ph = min(pm)
    # phase and swing searched jointly (the best phase without swing is not the best one with it)
    bm = [(round(mis(x - ph, b), 9), abs(b - 1.0) + abs(ph), ph, b) for ph in phases for b in burs]
    best_bm, _, ph_sw, best_bur = min(bm)
    res.update(misalignment=base, phase_beats=float(best_ph), phase_ms=float(best_ph * per * 1000.0),
               misalignment_phase=best_m, bur=float(best_bur), phase_swing_beats=float(ph_sw), misalignment_phase_swing=best_bm,
               explained_share=(1.0 - best_bm / base) if base > 0 else None)
    return res


# ------------------------------------------------- F5: syllable onsets from the vocal stem
#: [A] Ohriner 2019 p. 187: "I define the beginning of a syllable as the moment its vowel reaches half of its
#: maximum amplitude" (p-centre heuristic, measured on a cappella audio, p. 186). Parameters below are [H].
STEM_ONSET_PARAMS: dict[str, Param] = {
    "band_hz": Param((250.0, 4000.0), "[H] banda delle vocali (prime formanti): toglie rimbombo e soffi sopra i 4 kHz"),
    "hop_s": Param(0.005, "[H] passo dell'inviluppo (5 ms; Ohriner p. 187: correzioni manuali con differenza mediana 5 ms)"),
    "win_s": Param(0.015, "[H] finestra RMS dell'inviluppo"),
    "min_gap_s": Param(0.06, "[H] distanza minima tra due nuclei sillabici (al massimo ~16 sillabe/s)"),
    "prominence": Param(0.3, "[H] prominenza minima di un nucleo rispetto alla sua altezza (valle tra sillabe)"),
    "rel_db": Param(-30.0, "[H] nuclei sotto p95 - 30 dB dell'inviluppo dello stem sono ignorati (riverbero, residui Demucs)"),
    "window_s": Param(0.06, "[H] proposta F5 (PROPOSTE_OHRINER_KRIMS_HURON_MARGULIS.md): finestra di aggancio ±60 ms attorno "
                            "all'onset della parola Whisper; le sillabe interne possono spostarsi fino a metà della spaziatura uniforme"),
}


def stem_syllable_onsets(stem: np.ndarray, sample_rate: int = SR, **overrides: Any) -> dict[str, Any]:
    """[A] Ohriner 2019 p. 187 on a Demucs vocal stem: syllable nuclei = peaks of the band-limited RMS envelope,
    onset (p-centre) = the last time before the peak where the envelope crosses half of the peak amplitude,
    searched back to the trough before it. Returns ``{"onsets_s", "peaks_s", "peak_amp", "threshold"}``."""
    from scipy.signal import butter, find_peaks, sosfiltfilt

    p = {k: v.value for k, v in STEM_ONSET_PARAMS.items()}
    p.update(overrides)
    x = np.asarray(stem, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    sr = int(sample_rate)
    lo, hi = p["band_hz"]
    sos = butter(4, [lo / (sr / 2), min(hi / (sr / 2), 0.99)], btype="band", output="sos")
    y = sosfiltfilt(sos, x)
    hop, win = max(1, int(round(p["hop_s"] * sr))), max(2, int(round(p["win_s"] * sr)))
    nf = max(0, (len(y) - win) // hop + 1)
    if nf < 3:
        return {"onsets_s": np.zeros(0), "peaks_s": np.zeros(0), "peak_amp": np.zeros(0), "threshold": None, "params": p}
    sq = np.concatenate([[0.0], np.cumsum(y * y)])
    starts = np.arange(nf) * hop
    env = np.sqrt((sq[starts + win] - sq[starts]) / win)
    t = (starts + win / 2.0) / sr
    p95 = float(np.percentile(env, 95))
    thr = p95 * 10.0 ** (p["rel_db"] / 20.0)
    dist = max(1, int(round(p["min_gap_s"] / p["hop_s"])))
    pk, props = find_peaks(env, height=thr, distance=dist, prominence=0.0)
    keep = props["prominences"] >= p["prominence"] * env[pk]
    pk = pk[keep]
    ons = np.empty(len(pk))
    prev = 0
    for i, k in enumerate(pk):
        lo_i = prev if i == 0 else int(pk[i - 1] + np.argmin(env[pk[i - 1] : k + 1]))
        half = 0.5 * env[k]
        seg = env[lo_i : k + 1]
        below = np.nonzero(seg < half)[0]
        if below.size:
            j = lo_i + int(below[-1])  # last frame under half amplitude: interpolate the crossing
            a, b = env[j], env[j + 1]
            frac = (half - a) / (b - a) if b > a else 0.0
            ons[i] = t[j] + frac * (t[j + 1] - t[j])
        else:
            ons[i] = t[lo_i]
        prev = k
    return {"onsets_s": ons, "peaks_s": t[pk], "peak_amp": env[pk], "threshold": thr, "params": p,
            "source": "[A] Ohriner 2019 p. 187 (vowel at half of its maximum amplitude), on the Demucs vocal stem"}


def _align_word(targets: Sequence[float], cands: Sequence[float], max_d: Sequence[float]) -> list[Optional[int]]:
    """Monotone assignment of syllable targets to candidate onsets (each used once); a syllable may stay
    unmatched (cost = its max distance). Minimum total |t - c| by dynamic programming."""
    n, m = len(targets), len(cands)
    INF = float("inf")
    D = np.full((n + 1, m + 1), INF)
    D[0, :] = 0.0
    back = np.zeros((n + 1, m + 1), dtype=int)  # 0 = skip syllable, 1 = match, 2 = skip candidate
    for i in range(1, n + 1):
        D[i, 0] = D[i - 1, 0] + max_d[i - 1]
        for j in range(1, m + 1):
            best, how = D[i - 1, j] + max_d[i - 1], 0
            d = abs(targets[i - 1] - cands[j - 1])
            if d <= max_d[i - 1] and D[i - 1, j - 1] + d < best:
                best, how = D[i - 1, j - 1] + d, 1
            if D[i, j - 1] < best:
                best, how = D[i, j - 1], 2
            D[i, j], back[i, j] = best, how
    out: list[Optional[int]] = [None] * n
    i, j = n, m
    while i > 0:
        if j == 0:
            i -= 1
            continue
        how = back[i, j]
        if how == 1:
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif how == 0:
            i -= 1
        else:
            j -= 1
    return out


def with_stem_onsets(model: Mapping[str, Any], stem: Optional[np.ndarray] = None, sample_rate: int = SR, *,
                     onsets: Optional[Mapping[str, Any]] = None, window_s: Optional[float] = None,
                     beyond_word: str = "abstain") -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy of a dj_phrase model whose syllable onsets are snapped to the vocal-stem p-centres (proposal F5).

    Each word keeps ``on``/``off``/``last_syl_on``; new fields ``syl_on`` (sample of every syllable) and
    ``syl_src`` ('stem' if snapped, 'uniform' if no stem onset was close enough, 'uncertain_beyond_word' if
    the only close stem onset lies at or after the word's ASR end). Window: the first syllable within
    ±``window_s`` of the word onset; inner syllables within max(``window_s``, half of the uniform spacing),
    always inside [on - window, off): an onset at or after ``off`` is never assigned to the word.

    ``beyond_word``: ``"abstain"`` (default since 21/10/2026) keeps the uniform estimate for a syllable whose
    nearest onset lies past the word end, marks it ``uncertain_beyond_word`` and does not count it as snapped
    (ASR offset and stem onset disagree: the onset may belong to the next sound; Whisper's offset is not
    assumed perfect either, so the syllable is reported as uncertain, not moved). ``"legacy"`` reproduces the
    pre-audit behaviour (window up to the last target + window, so 1040 ms could be given to a 900-1000 ms
    word: ROOT_COUNTEREXAMPLES.json 'stem_onset_past_word'); only for re-running old comparisons.
    :func:`model_syllables`, :func:`syllable_npvi`, :func:`ohriner_global_features` and
    :func:`vocal_phase_alignment` (stem onsets only) then use them. Returns ``(model, info)``.
    """
    if beyond_word not in ("abstain", "legacy"):
        raise ValueError(f"beyond_word must be 'abstain' or 'legacy', not {beyond_word!r}")
    import copy

    sr = int(model["sample_rate"])
    if onsets is None:
        if stem is None:
            raise ValueError("with_stem_onsets needs the vocal stem or precomputed onsets")
        onsets = stem_syllable_onsets(stem, sample_rate)
    w_s = STEM_ONSET_PARAMS["window_s"].value if window_s is None else float(window_s)
    cand = np.sort(np.asarray(onsets["onsets_s"], dtype=float)) * sr
    used = np.zeros(len(cand), dtype=bool)
    m = copy.deepcopy(dict(model))
    win = w_s * sr
    n_syl = n_snap = n_beyond = 0
    shifts = []
    for w in m["words"]:
        if w.get("n_syl") is None or w.get("last_syl_on") is None:
            continue
        n = max(1, int(w["n_syl"]))
        on, last_on, off = float(w["on"]), float(w["last_syl_on"]), float(w["off"])
        tg = [on + (last_on - on) * k / (n - 1) for k in range(n)] if n > 1 else [last_on if last_on >= on else on]
        sp = (last_on - on) / (n - 1) if n > 1 else 0.0
        md = [win] + [max(win, 0.5 * sp)] * (n - 1)
        lo_i, hi_far = np.searchsorted(cand, on - win), np.searchsorted(cand, max(off, tg[-1] + md[-1]))
        hi_i = hi_far if beyond_word == "legacy" else np.searchsorted(cand, off)  # [on - window, off)
        idx = [k for k in range(lo_i, hi_i) if not used[k]]
        assign = _align_word(tg, [cand[k] for k in idx], md)
        beyond = [k for k in range(hi_i, hi_far) if not used[k]]  # past the word end: never assigned here
        syl, src = [], []
        for t, a, d in zip(tg, assign, md):
            if a is None:
                syl.append(int(round(t)))
                near = any(abs(cand[k] - t) <= d for k in beyond)
                src.append("uncertain_beyond_word" if near else "uniform")
                n_beyond += int(near)
            else:
                used[idx[a]] = True
                syl.append(int(round(cand[idx[a]])))
                src.append("stem")
                shifts.append((cand[idx[a]] - t) / sr * 1000.0)
        for k in range(1, len(syl)):  # keep the order inside the word
            if syl[k] <= syl[k - 1]:
                syl[k] = syl[k - 1] + 1
        w["syl_on"], w["syl_src"] = syl, src
        n_syl += n
        n_snap += src.count("stem")
    sh = np.asarray(shifts, dtype=float)
    info = {"n_syllables": n_syl, "n_snapped": n_snap, "snapped_share": (n_snap / n_syl) if n_syl else None,
            "n_uncertain_beyond_word": n_beyond, "uncertain_share": (n_beyond / n_syl) if n_syl else None,
            "beyond_word": beyond_word,
            "n_stem_onsets": int(len(cand)), "unused_stem_onsets": int((~used).sum()),
            "shift_ms_median_abs": float(np.median(np.abs(sh))) if sh.size else None,
            "shift_ms_median": float(np.median(sh)) if sh.size else None, "window_s": w_s,
            "source": "[A] Ohriner 2019 p. 187 (p-centre = vowel at half amplitude); proposal F5; window [H]"}
    m["syllable_onsets"] = info
    return m, info


def build_halving_schedule(total_bars: float, start_unit: float = 2.0, min_unit: float = 0.25,
                           repeats_per_stage: int = 2) -> list[dict[str, Any]]:
    """[A] Margulis 2014 p. 23 and p. 44: shrinking the repeated unit (2 bars -> 1 -> 1/2 -> ...) is heard as
    acceleration at constant tempo. [H] Schedule for the last ``total_bars`` of A before B's downbeat:
    each stage repeats its unit ``repeats_per_stage`` times (Lidov via Margulis p. 51: two statements
    establish a pattern); the first stage absorbs what is left. Returns stages in order (start bar relative
    to the start of the build). Durations sum to ``total_bars``."""
    units, u = [], float(start_unit)
    while u >= min_unit - 1e-9:
        units.append(u)
        u /= 2.0
    tail = [(u, repeats_per_stage) for u in units[1:]]
    used = sum(u * n for u, n in tail)
    while tail and used > total_bars - units[0] * repeats_per_stage + 1e-9:
        u, n = tail.pop()  # drop the smallest units if the span is short
        used -= u * n
    first_n = (total_bars - used) / units[0]
    stages, t = [], 0.0
    for u, n in [(units[0], first_n)] + tail:
        if n <= 0:
            continue
        stages.append({"start_bar": t, "unit_bars": u, "repeats": n})
        t += u * n
    return stages


# ------------------------------------------------------------------ planning
def exits_for_track(track: Mapping[str, Any], model: Mapping[str, Any], act: Optional[StemActivity] = None,
                    p: Optional[Mapping[str, Any]] = None) -> list[dict[str, Any]]:
    """exit_point for every cue-out candidate of dj_select (phrase-start downbeats)."""
    from musicmix import dj_select as djs

    return [dict(exit_point(model, c["time_s"], p, act), bar_index=c["bar_index"], phrase_level=c["phrase_level"])
            for c in djs.cue_candidates(track, "out")]


def entries_for_track(track: Mapping[str, Any], model: Mapping[str, Any], act: Optional[StemActivity] = None,
                      p: Optional[Mapping[str, Any]] = None) -> list[dict[str, Any]]:
    from musicmix import dj_select as djs

    return [dict(entry_point(model, c["time_s"], p, act), bar_index=c["bar_index"], phrase_level=c["phrase_level"])
            for c in djs.cue_candidates(track, "in")]


def plan_pair_phrase(A: Mapping[str, Any], B: Mapping[str, Any], model_a: Mapping[str, Any], model_b: Mapping[str, Any],
                     *, act_a: Optional[StemActivity] = None, act_b: Optional[StemActivity] = None, top_k: int = 5,
                     p: Optional[Mapping[str, Any]] = None, require_ok: bool = True,
                     flow_weight: float = 0.0, with_flow: bool = False,
                     flow_cache: Optional[dict] = None) -> dict[str, Any]:
    """dj_select plans (cut family) re-checked at verse ends with B's pickup; ranked.

    Score = dj_select score (with ``cut_vocal_guard`` off: the phrase test replaces it)
    + p(line end) of A + 0.5 if B's pickup is preserved [H]. Only plans whose
    phrase switch is ``ok`` are kept when ``require_ok``.

    ``flow_weight`` > 0 adds ``flow_weight * continuity`` of :func:`flow_compatibility` (A's last 4 bars vs
    B's first 4) [H]: default 0 = v1 ranking (AGENTS.md: the flow is diagnostic). ``with_flow`` only
    attaches the flow diagnostics. ``flow_cache`` (dict) reuses flow windows across calls, keyed by
    (slug, side, bar_index).
    """
    from musicmix import dj_select as djs

    pair = djs.pair_compatibility(A, B)
    exits = {c["bar_index"]: (c, exit_point(model_a, c["time_s"], p, act_a)) for c in djs.cue_candidates(A, "out")}
    entries = {c["bar_index"]: (c, entry_point(model_b, c["time_s"], p, act_b)) for c in djs.cue_candidates(B, "in")}
    plans = []
    want_flow = with_flow or flow_weight > 0
    cache = flow_cache if flow_cache is not None else {}
    for ja, (ca, ea) in exits.items():
        if ea.get("window") is None:
            continue
        for jb, (cb, eb) in entries.items():
            if eb.get("window") is None:
                continue
            sw = phrase_switch(ea, eb, model_a["period_s"], p)
            if require_ok and not sw["ok"]:
                continue
            base = djs.plan_transition(A, B, ca, cb, pair)
            bonus = (ea.get("p_line_end") or 0.0) + (0.5 if sw.get("pickup_preserved") and eb.get("status") == "pickup" else 0.0)
            extra: dict[str, Any] = {}
            if want_flow:
                ka, kb = (A.get("slug"), "a", ja), (B.get("slug"), "b", jb)
                if ka not in cache:
                    cache[ka] = flow_window(model_a, ca["time_s"], "a")
                if kb not in cache:
                    cache[kb] = flow_window(model_b, cb["time_s"], "b")
                fc = flow_compatibility(cache[ka], cache[kb])
                extra["flow"] = fc
                if flow_weight > 0 and fc["continuity"] is not None:
                    bonus += flow_weight * fc["continuity"]
            plans.append(dict(base, type_dj_select=base["type"], type="cut_phrase", phrase={"exit": ea, "entry": eb, "switch": sw},
                              score_phrase=base["score"] + bonus,
                              reasons=base["reasons"] + ea["reasons"] + eb["reasons"] + sw["reasons"], **extra))
    plans.sort(key=lambda q: (-q["score_phrase"], q["a_out_s"], q["b_in_s"]))
    return {"a": A.get("slug"), "b": B.get("slug"), "compatibility": pair, "n_candidates": len(plans), "plans": plans[:top_k]}


B_ENTRY_VERSION = "musicmix.dj_phrase.b_entry.1"  # 21/10/2026: b_lead_in + entry_clear_test (INGRESSO_B_2026-10-21.md)
B_ENTRY_OK = ("instrumental", "on_downbeat", "lead_in")


def b_lead_in(model: Mapping[str, Any], downbeat_s: float, p: Optional[Mapping[str, Any]] = None,
              act: Optional[StemActivity] = None) -> dict[str, Any]:
    """Where must B become audible so that its first heard vocal phrase is whole, if it enters on ``downbeat_s``?

    Rule [H] (``b_rest_beats``, ``b_lead_in_max_beats``; sources in PARAMS): starting from the first word of B that is
    not over by the downbeat D, walk back through B's words until a phrase boundary: a silence of at least
    ``b_rest_beats``, or a line end of the model (p >= ``line_end_threshold``) followed by a noted rest
    (>= ``rest_min_16th`` sixteenths); the phrase starts at the first word after it. If it starts before D,
    B must be audible from just before it (``lead_in_samples`` = D - start, snapped down to the eighth inside the
    rest): status ``lead_in`` when that is at most ``b_lead_in_max_beats``, else ``mid_line`` (entering here truncates
    a phrase: choose another cue). ``on_downbeat``: the phrase starts at or after D after a rest; ``instrumental``:
    no voice of B within a bar after D. Without reliable ASR the vocal stem activity (gaps shorter than a rest closed)
    replaces the words (``evidence: stem``); without either the status is ``unknown``.
    Unlike :func:`entry_point` (pickup <= 2 beats, any p >= 0.5 boundary) a line end without an audible rest does not
    start a phrase, and the lead-in may last up to a bar.
    """
    pr = params(p or model.get("params"))
    sr = int(model["sample_rate"])
    D = int(round(downbeat_s * sr))
    beat = int(round(float(model["period_s"]) * sr))
    rest = int(round(pr["b_rest_beats"] * beat))
    max_lead = int(round(pr["b_lead_in_max_beats"] * beat))
    m = int(round(pr["switch_margin_s"] * sr))
    step = max(1, int(round(beat / pr["switch_grid_div"])))
    res: dict[str, Any] = {"version": B_ENTRY_VERSION, "downbeat_sample": D, "status": "unknown", "lead_in_samples": 0,
                           "audible_sample": D, "evidence": None, "reasons": [], "rest_samples": rest, "max_lead_samples": max_lead}
    words = [w for w in (model.get("words") or []) if not w.get("dropped")]
    reliable = bool((model.get("asr_quality") or {}).get("reliable", True)) and bool(words)

    def _place(start: int, what: str) -> dict[str, Any]:
        lead = D - (start - m)
        if lead <= 0:
            res.update(status="on_downbeat", lead_in_samples=0, audible_sample=D)
            res["reasons"].append(f"{what} inizia {(start - D) / sr * 1000:+.0f} ms dal downbeat dopo una pausa >= {pr['b_rest_beats']:g} battiti")
            return res
        if lead > max_lead:
            res.update(status="mid_line", lead_in_samples=int(lead), audible_sample=int(D - lead))
            res["reasons"].append(f"{what} inizia {lead / sr:.2f} s prima del downbeat (> {pr['b_lead_in_max_beats']:g} battiti): "
                                  "entrare qui tronca la frase di B")
            return res
        snapped = int(math.ceil(lead / step) * step)  # earlier eighth inside the rest, never later than start - m
        prev_end = res.get("_prev_end")
        if snapped <= max_lead and (prev_end is None or D - snapped >= prev_end + m):
            lead = snapped
        res.update(status="lead_in", lead_in_samples=int(lead), audible_sample=int(D - lead))
        res["reasons"].append(f"{what} inizia {lead / sr * 1000:.0f} ms prima del downbeat: B udibile da lì (lead-in)")
        return res

    if reliable:
        res["evidence"] = "asr_words"
        on = np.array([int(w["on"]) for w in words])
        off = np.array([int(w["off"]) for w in words])
        later = np.flatnonzero(off > D)
        if not later.size or on[later[0]] >= D + 4 * beat:
            # nothing crosses D and no voice within a bar after it: check that B's previous phrase is not still running
            res.update(status="instrumental", lead_in_samples=0, audible_sample=D)
            res["reasons"].append("nessuna voce di B nella battuta dopo il downbeat")
            return _finish_lead_in(res, words, D, sr, act, m)
        j = int(later[0])
        gp = {int(g["after"]): float(g.get("p_end") or 0.0) for g in (model.get("gaps") or [])} if len(words) == len(model.get("words") or []) else {}
        noted = int(round(pr["rest_min_16th"] * beat / 4))

        def brk(i: int) -> bool:  # boundary between words i and i+1
            gap = on[i + 1] - off[i]
            return bool(gap >= rest or (gp.get(i, 0.0) >= pr["line_end_threshold"] and gap >= noted))

        k = j
        while k > 0 and not brk(k - 1):
            k -= 1
        res["_prev_end"] = int(off[k - 1]) if k > 0 else None
        res["phrase_words"] = [w["text"] for w in words[k:j + 1]]
        res["first_word"] = words[k]["text"]
        res["first_word_rel"] = int(on[k] - D)
        res["lead_in_words"] = [w["text"] for w in words[k:] if int(w["on"]) < D]
        res["rest_before_s"] = (int(on[k]) - int(off[k - 1])) / sr if k > 0 else None
        _place(int(on[k]), f"la frase di B «{' '.join(res['phrase_words'][:6])}»")
        return _finish_lead_in(res, words, D, sr, act, m)
    if act is not None:
        res["evidence"] = "stem"
        a = act.active
        c = act.centers
        if not a.size:
            return res
        idx = int(np.searchsorted(c, D))
        # start of the activity run that is active at (or within a rest after) D, closing gaps shorter than a rest
        nxt = np.flatnonzero(a[idx:] & (c[idx:] < D + 4 * beat))
        if not nxt.size:
            res.update(status="instrumental")
            res["reasons"].append("stem vocale di B silenzioso nella battuta dopo il downbeat")
            return res
        k = idx + int(nxt[0])
        start = int(c[k])
        while True:
            prev = np.flatnonzero(a[:k] & (c[:k] > start - rest))
            if not prev.size:
                break
            k = int(prev[0])
            # move to the beginning of that run
            while k > 0 and a[k - 1]:
                k -= 1
            start = int(c[k])
        _place(start - act.hop // 2, "la voce di B (stem)")
        res.pop("_prev_end", None)
        return res
    res["reasons"].append("trascrizione non affidabile e nessuno stem: ingresso non verificabile")
    return res


def _finish_lead_in(res: dict[str, Any], words: Sequence[Mapping[str, Any]], D: int, sr: int, act: Optional[StemActivity],
                    m: int) -> dict[str, Any]:
    res.pop("_prev_end", None)
    start = int(res["audible_sample"])
    res["truncated_at_audible"] = [w["text"] for w in truncated_words(words, start, role="b", margin=int(PARAMS["word_margin_s"].value * sr))]
    if res["truncated_at_audible"] and res["status"] in B_ENTRY_OK:
        res["reasons"].append("parola di B troncata nel punto udibile: " + " ".join(res["truncated_at_audible"]))
        res["status"] = "mid_line"
    if act is not None and res["status"] in B_ENTRY_OK:
        res["acoustic"] = "stem_active" if act.active_near(start, m, strict=True) else "silent_point"
        if res["acoustic"] == "stem_active":
            res["reasons"].append("stem vocale di B attivo nel punto udibile (ad-lib, cori o parole non trascritte): diagnostica")
    return res


def entry_clear_test(model: Mapping[str, Any], act: Optional[StemActivity] = None, p: Optional[Mapping[str, Any]] = None):
    """``is_clear(cue) -> dict`` for :func:`musicmix.dj_select.relocate_entry`: B can enter at that downbeat with its first
    phrase whole (``ok``), possibly from a lead-in before the downbeat (:func:`b_lead_in`)."""
    pr = params(p or model.get("params"))

    def is_clear(cue: Mapping[str, Any]) -> dict[str, Any]:
        r = b_lead_in(model, float(cue["time_s"]), pr, act)
        return dict(r, ok=r["status"] in B_ENTRY_OK)
    return is_clear


def exit_clear_test(model: Mapping[str, Any], act: Optional[StemActivity] = None, p: Optional[Mapping[str, Any]] = None):
    """``is_clear(cue)`` for :func:`musicmix.dj_select.relocate_cut`: A's exit at that downbeat is a verse end."""
    pr = params(p or model.get("params"))

    def is_clear(cue: Mapping[str, Any]) -> bool:
        e = exit_point(model, float(cue["time_s"]), pr, act)
        return bool(e["status"] in ("clean", "tail", "a_pickup", "instrumental")
                    and (e["status"] == "instrumental" or (e.get("p_line_end") or 0) >= pr["line_end_threshold"])
                    and e.get("acoustic") != "stem_active_in_window")
    return is_clear


# ----------------------------------------------------------------- rendering
def render_phrase_transition(kind: str, audio_a: np.ndarray, audio_b: np.ndarray, *, beats_a, downbeats_a, beats_b, downbeats_b,
                             cue_out_a: float, cue_in_b: float, switch_rel_s: float = 0.0, tail_end_a_s: Optional[float] = None,
                             vocals_a: Optional[np.ndarray] = None, sample_rate: int = SR, sweep_bars: int = 2,
                             render_config: Any = None, effects_config: Any = None, provenance: Optional[dict] = None,
                             tail_fade_s: float = PARAMS["tail_fade_s"].value, pickup_mode: str = "switch",
                             vocals_b: Optional[np.ndarray] = None):
    """Cut or filter_out with the fader switch moved to ``switch_rel_s`` (<= 0) and A's vocal tail kept.

    Renders with ``dj_render.render_transition(kind='cut')`` or ``dj_effects.filter_out`` (same
    loudness normalisation, section gain and metadata), then rebuilds the decks from the source
    audio on the same output clock (B's downbeat stays on A's downbeat) with the new faders,
    adds A's vocal stem from the downbeat to ``tail_end_a_s`` (+ fade) and re-limits.

    ``pickup_mode``: ``"switch"`` = the whole of B replaces A at the early switch; ``"stem"`` (cut
    only, needs both vocal stems) = A's instrumental (mix minus its Demucs vocal) keeps playing to
    the downbeat while only B's vocal stem plays its pickup, then full B on the downbeat [H]: the
    bar of A is not cut off-beat and B's beat starts on the one.
    """
    from musicmix import dj_effects as fx
    from musicmix import dj_render as dj

    sr = int(sample_rate)
    grids = dict(beats_a=beats_a, downbeats_a=downbeats_a, beats_b=beats_b, downbeats_b=downbeats_b)
    if kind == "cut":
        cfg = render_config or dj.DJRenderConfig()
        res = dj.render_transition(audio_a, audio_b, **grids, cue_out_a=cue_out_a, cue_in_b=cue_in_b, kind="cut",
                                   sample_rate=sr, config=cfg, provenance=provenance, return_decks=True)
        xf_ms, ceiling, hold = cfg.crossfade_ms, cfg.ceiling_dbtp, cfg.limiter_hold_ms
    elif kind == "filter_out":
        cfg = effects_config or fx.DJEffectsConfig()
        res = fx.filter_out(audio_a, audio_b, **grids, cue_out_a=cue_out_a, cue_in_b=cue_in_b, sweep_bars=sweep_bars,
                            sample_rate=sr, config=cfg, provenance=provenance, return_decks=True)
        xf_ms, ceiling, hold = cfg.crossfade_ms, cfg.ceiling_dbtp, cfg.limiter_hold_ms
    else:
        raise ValueError("kind must be 'cut' or 'filter_out'")
    meta = dict(res.metadata)
    n = len(res.audio)
    sw = int(meta["switch"]["output_sample"])
    sa, sb = int(meta["switch"]["a_source_sample"]), int(meta["switch"]["b_source_sample"])
    ga = 10 ** (meta["loudness"]["a_gain_db"] / 20.0)
    gb = 10 ** (meta["loudness"]["b_gain_db"] / 20.0)
    x = int(round(switch_rel_s * sr))
    if x > 0:
        raise ValueError("switch_rel_s must be <= 0 (the switch never comes after the downbeat)")
    new_sw = sw + x
    d = int(round(xf_ms / 1000.0 * sr))
    t = np.arange(n)
    fade = np.clip((t - (new_sw - d)) / max(d, 1), 0.0, 1.0)
    fader_a = np.cos(0.5 * np.pi * fade)
    fader_a[t >= new_sw] = 0.0
    fader_b = np.sin(0.5 * np.pi * fade)
    a = dj._as_stereo(np.asarray(audio_a, dtype=np.float64))
    b = dj._as_stereo(np.asarray(audio_b, dtype=np.float64))
    if kind == "cut":
        a_pre = dj._slice(a, sa - sw, n) * ga
    else:
        a_pre = np.asarray(res.decks["a_filtered_prefader"], dtype=np.float64)
    sg = np.asarray(res.decks["gains"].get("section_gain_b", np.ones(n)), dtype=np.float64)[:n]
    b_src = dj._slice(b, sb - sw, n) * gb
    deck_a = a_pre * fader_a[:, None]
    deck_b = b_src * (fader_b * sg)[:, None]
    pickup_info = {"mode": "switch" if x < 0 else "none"}
    fader_a_qa, fader_b_qa = fader_a, fader_b
    if pickup_mode == "stem" and x < 0 and kind == "cut" and vocals_a is not None and vocals_b is not None:
        fade_d = np.clip((t - (sw - d)) / max(d, 1), 0.0, 1.0)  # the plain cut fade, ending on the downbeat
        g_full_a = np.cos(0.5 * np.pi * fade_d)
        g_full_a[t >= sw] = 0.0
        g_full_b = np.sin(0.5 * np.pi * fade_d)
        va_src = dj._slice(dj._as_stereo(np.asarray(vocals_a, dtype=np.float64)), sa - sw, n) * ga
        vb_src = dj._slice(dj._as_stereo(np.asarray(vocals_b, dtype=np.float64)), sb - sw, n) * gb
        g_inst = (1.0 - fader_a) * g_full_a  # A instrumental: from the early switch to the downbeat
        g_bv = fader_b * (1.0 - g_full_b)  # B vocal alone: from the early switch to the downbeat
        deck_a = a_pre * fader_a[:, None] + (a_pre - va_src) * g_inst[:, None]
        deck_b = b_src * (g_full_b * sg)[:, None] + vb_src * (g_bv * sg)[:, None]
        fader_a_qa, fader_b_qa = np.maximum(fader_a, g_inst), fader_b
        pickup_info = {"mode": "stem", "a_instrumental_output": [int(new_sw - d), int(sw)],
                       "b_vocal_only_output": [int(new_sw - d), int(sw)],
                       "note": "A instrumental = mix - Demucs vocal stem; B vocal stem alone until its downbeat [H]"}
    tail_info = None
    deck_tail = np.zeros_like(deck_a)
    if tail_end_a_s is not None and vocals_a is not None:
        tail_end = sw + int(round(tail_end_a_s * sr)) - sa
        if tail_end > new_sw:
            va = dj._as_stereo(np.asarray(vocals_a, dtype=np.float64))
            v = dj._slice(va, sa - sw, n) * ga
            f = max(1, int(round(tail_fade_s * sr)))
            # stem fades in while the dry deck fades out (same 10 ms), holds until the word's end, then fades out
            g = fader_b.copy()
            g[t >= tail_end] = 0.0
            k = np.arange(tail_end, min(n, tail_end + f))
            g[k] = np.cos(0.5 * np.pi * (k - tail_end) / f) ** 2
            deck_tail = v * g[:, None]
            tail_info = {"output_start_sample": int(new_sw - d), "output_end_sample": int(min(n, tail_end + f)),
                         "duration_s": (tail_end - new_sw) / sr, "fade_s": tail_fade_s, "source": "A vocal stem (Demucs), A loudness gain"}
    out = deck_a + deck_b + deck_tail
    limited, limiter = dj.true_peak_limit(out, sr, ceiling, hold)
    loud = dict(meta["loudness"])
    loud["output_integrated_lufs"] = dj.integrated_lufs(limited, sr)
    loud["output_true_peak_dbtp"] = limiter["true_peak_after_dbtp"]
    meta["loudness"] = loud
    meta["limiter"] = limiter
    # ``switch.output_sample`` stays on the downbeat (B's downbeat on A's, as in dj_render/dj_effects and
    # for the 10 s + 10 s clip rule); the early fader switch is an explicit "pickup" region before it.
    meta["switch"] = dict(meta["switch"], downbeat_output_sample=sw, fader_switch_output_sample=int(new_sw),
                          fader_switch_a_source_sample=int(sa + x), fader_switch_b_source_sample=int(sb + x),
                          note="dj_phrase: B audible from fader_switch_output_sample (pickup), B's downbeat at output_sample")
    ov = dict(meta["overlap"])
    if kind == "cut" and x < 0:
        # source samples of the pickup start on the same clock (A: sa - sw + k, B: sb - sw + k); before 21/10/2026 only the
        # output samples moved, so dj_set.clip_geometry spliced the whole track of A up to 0.5 s off (runs/dj_set_v3_bridge)
        ov.update(output_start_sample=int(new_sw - d), output_end_sample=int(sw),
                  a_source_start_sample=int(sa - sw + new_sw - d), b_source_start_sample=int(sb - sw + new_sw - d),
                  a_source_end_sample=int(sa), b_source_end_sample=int(sb), duration_s=(sw - (new_sw - d)) / sr,
                  note=f"pickup region ({pickup_info['mode']}): B's pickup from the fader switch to its downbeat")
    meta["overlap"] = ov
    meta["phrase"] = {"schema_version": SCHEMA_VERSION, "switch_rel_s": switch_rel_s, "switch_rel_samples": x,
                      "tail": tail_info, "pickup": pickup_info, "base_render": kind,
                      "note": "decks rebuilt from source audio on the renderer's output clock; same loudness gains and section gain"}
    decks = {"a": deck_a + deck_tail, "b": deck_b,
             "gains": {"fader_a": fader_a_qa, "fader_b": fader_b_qa, "section_gain_b": sg}}
    return dj.DJRenderResult(limited.astype(np.float32), sr, dj._jsonable(meta), decks)


__all__ = [name for name in dir() if not name.startswith("_") and name not in (
    "annotations", "dataclass", "json", "math", "os", "re", "unicodedata", "np", "Any", "Iterable", "Mapping",
    "Optional", "Sequence", "Path")]
