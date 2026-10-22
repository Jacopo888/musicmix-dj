"""Supervisione dai brani stessi: dove un brano si articola da solo (punti naturali) e versioni spostate.

Requisito dell'utente (21/10/2026, regola "Autonomia" di AGENTS.md): niente voti continui e niente mix da
fornire. Dentro ogni brano ci sono già passaggi riusciti (inizio di sezione sul downbeat, inizio di verso
con pickup dopo una pausa, rientro della voce). Sono esempi positivi di "punto giusto"; le loro versioni
spostate dentro lo stesso brano (±½ battito, ±1 battito, ±½ battuta, ±1 battuta, metà parola, downbeat a
metà verso, downbeat a metà sezione) sono negativi controllati. Nota: :doc:`conoscenza_mix/_meta/SUPERVISIONE_DAI_BRANI_2026-10-21.md`.

Contenuto (puro numpy; i modelli preaddestrati sono caricati solo dai tool):

* :func:`load_track` — griglia, sezioni allin1/dj_form e parole/versi di ``dj_phrase`` di un brano già preparato
  (``runs/dj_set_v3_bridge``), senza toccare i JSON congelati (percorsi via :func:`musicmix.paths.localize`).
* :func:`natural_points` / :func:`shifted_negatives` / :func:`build_points` — dataset di punti (positivi e
  negativi per tipo), con l'esclusione dei negativi vicini a qualunque confine plausibile.
* :func:`build_pairs` — giunte (uscita da X a t1, ingresso in Y a t2) dentro lo stesso brano e tra brani
  diversi, naturali contro spostate, per staccare il segnale "stesso brano".
* :func:`artist_group` / :func:`group_folds` — split per artista (mai lo stesso brano o artista in train e test).
* :func:`point_features` / :func:`pair_features` — feature dalle cache compatte per brano (MERT medio per
  finestra, DSP per banda, voce dallo stem, fase della griglia).
* :class:`BlockLogistic` — logistica con standardizzazione e PCA per blocco ad alta dimensione, salvabile in npz.
* :func:`auc`, :func:`paired_win_rate` — metriche per tipo di negativo.
* :func:`cue_naturalness` — API per schede brano e pianificatore: probabilità che ``t`` sia un punto in cui il
  brano si articola da solo, letta da ``runs/insong_v1/cue_scores/<slug>.json`` o calcolata dalle cache.

* :func:`tension_examples` / :func:`tension_features` / :func:`tension_at_exit` / :func:`resolution_at_entry` — compito
  aggiunto (carica prima dell'arrivo di un ritornello, arrivo): proxy [H] ritornello = arrivo, debole e contraddetto dal
  controllo esterno sui tornei dei cue (vedi la nota, §5–6).

Cosa NON dice: impara la continuità interna ai brani (dove il brano stesso cambia), non le scelte di un DJ
(quale brano, quale sezione di B, ordine, durata della sovrapposizione). Un punteggio alto non è una
transizione approvata: è un "dove" plausibile. Le etichette vengono da modelli (allin1, Whisper + dj_phrase).
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from musicmix.paths import localize

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "musicmix.insong.1"
RUN_DIR = ROOT / "runs" / "insong_v1"
CUE_DIR = RUN_DIR / "cue_scores"
SOURCE_RUN = ROOT / "runs" / "dj_set_v3_bridge"

#: tipi di negativo (nome -> descrizione); l'ordine è quello dei report
NEG_TYPES: dict[str, str] = {
    "offbeat": "punto naturale spostato di ±½ battito (taglio fuori battito)",
    "beat": "punto naturale spostato di ±1 battito",
    "halfbar": "punto naturale spostato di ±½ battuta (2 battiti)",
    "bar": "punto naturale spostato di ±1 battuta (downbeat sbagliato)",
    "midword": "a metà di una parola vicina (parole raffinate di dj_phrase)",
    "midline": "downbeat dentro un verso (la voce attraversa il downbeat)",
    "mid_section": "downbeat a metà sezione (battuta 3 di un gruppo di 4, lontano dai confini)",
    "random": "istante casuale nel brano (riferimento facile)",
}
POS_KINDS = ("section", "line", "reentry")

PARAMS: dict[str, Any] = {
    "margin_s": 5.5,             # contesto richiesto ai due lati (finestre COCOLA da 5 s, MERT da 2 s)
    "section_snap_max_bars": 0.05,  # sezione allin1 positiva solo se il confine coincide col downbeat della griglia
    "line_rest_beats": 1.0,      # = dj_phrase.PARAMS['b_rest_beats'] (R-B1): una frase comincia dopo una pausa >= 1 battito
    "reentry_rest_beats": 8.0,   # rientro della voce: almeno 2 battute senza parole
    "line_lead_s": 0.03,         # il punto di un verso è 30 ms prima dell'attacco della prima parola
    "exclusion_s": 0.15,         # un negativo non può stare entro 150 ms da un confine plausibile
    "near_beats": 8.0,           # metà parola / metà verso cercati entro ±2 battute dal positivo
    "min_word_s": 0.12,          # parole troppo brevi non danno un "metà parola" affidabile
    "mid_section_bars": 16.0,
    "pair_min_jump_bars": 8.0,   # salto minimo dentro lo stesso brano (non una semplice continuazione)
    "pair_tempo_tol": 0.04,      # coppie tra brani: tempo entro ±4 % (anche metà/doppio)
}

#: gruppi d'artista: nome principale normalizzato; le collaborazioni del catalogo restano col primo nome
GROUP_ALIASES = {"travi$ scott": "travis scott"}


# ======================================================================= dati del brano
@dataclass
class TrackData:
    slug: str
    artist: str
    group: str
    sr: int
    duration_s: float
    period_s: float
    downbeats_s: np.ndarray
    sections: list[dict]
    words: list[dict]            # on_s, off_s, text (parole raffinate, senza le scartate)
    lines: list[dict]            # start_s, end_s, first_word, last_word, p_end
    bpm: float = 0.0
    camelot: Optional[str] = None
    audio_path: Optional[str] = None
    vocal_path: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def beats_s(self) -> np.ndarray:
        db = self.downbeats_s
        if db.size == 0:
            return db
        k = np.arange(4)
        b = (db[:, None] + k[None, :] * self.period_s).ravel()
        return b[b < self.duration_s]

    def bar_of(self, t: float) -> int:
        return int(np.searchsorted(self.downbeats_s, t + 1e-6) - 1)


def artist_group(artist: str) -> str:
    a = (artist or "").strip().lower()
    a = GROUP_ALIASES.get(a, a)
    first = re.split(r"\s*(?:,|;|&| feat\.?| ft\.?| x )\s*", a)[0].strip()
    return GROUP_ALIASES.get(first, first) or "unknown"


def _read(p: Path) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def load_track(slug: str, *, source_run: Path = SOURCE_RUN, artist: Optional[str] = None) -> TrackData:
    """Brano preparato da ``tools/run_dj_set_v3_bridge.py prepare`` (analisi DJ + modello dj_phrase)."""
    tj = _read(Path(source_run) / "dj_plan" / "tracks" / f"{slug}.json")
    pj = _read(Path(source_run) / "phrase" / f"{slug}.json")
    m = pj["model"]
    sr = int(m.get("sample_rate") or tj.get("sample_rate") or 48000)
    words = [{"on_s": w["on"] / sr, "off_s": w["off"] / sr, "text": w.get("text", "")}
             for w in m.get("words", []) if not w.get("dropped")]
    lines = [{"start_s": ln["start_sample"] / sr, "end_s": ln["end_sample"] / sr, "first_word": int(ln["first_word"]),
              "last_word": int(ln["last_word"]), "p_end": ln.get("p_end")} for ln in m.get("lines", [])]
    inputs = pj.get("inputs") or {}
    audio = (tj.get("inputs") or {}).get("canonical_wav")
    vocal = inputs.get("vocal_stem")
    vp = None
    if vocal:
        cand = Path(vocal)
        vp = str(cand if cand.is_absolute() else ROOT / cand)
        vp = str(localize(vp))
    art = artist or tj.get("artist") or ""
    return TrackData(slug=slug, artist=art, group=artist_group(art), sr=sr, duration_s=float(tj["duration_s"]),
                     period_s=float(tj["tempo"]["period_s"]), downbeats_s=np.asarray(tj["grid"]["downbeats_s"], float),
                     sections=list(tj.get("sections") or []), words=words, lines=lines, bpm=float(tj["tempo"]["bpm"]),
                     camelot=(tj.get("key") or {}).get("camelot"), audio_path=str(localize(audio)) if audio else None,
                     vocal_path=vp, meta={"asr_reliable": bool((m.get("asr_quality") or {}).get("reliable")),
                                          "lang": m.get("lang"), "octave_ambiguous": bool(tj["tempo"].get("octave_ambiguous"))})


# ======================================================================= punti naturali
def _rest_before(track: TrackData, line: dict) -> float:
    fw = line["first_word"]
    if fw <= 0:
        return line["start_s"]  # prima frase: tutto ciò che precede è senza voce
    return line["start_s"] - track.words[fw - 1]["off_s"]


def natural_points(track: TrackData, params: Mapping[str, Any] = PARAMS) -> list[dict]:
    """Positivi: inizi di sezione sul downbeat, inizi di verso dopo una pausa, rientri della voce.

    * ``section``: confine allin1/dj_form agganciato a un downbeat della griglia con errore <= 0,05 battute
      (i confini a ¼ di battuta dalla griglia sono ambigui: esclusi dai positivi e protetti dai negativi);
    * ``line``: prima parola di un verso di dj_phrase preceduta da una pausa della voce >= 1 battito (R-B1),
      punto = 30 ms prima dell'attacco (il pickup resta intero);
    * ``reentry``: come ``line`` con almeno 2 battute senza parole prima.
    """
    m, dur, per = float(params["margin_s"]), track.duration_s, track.period_s
    out: list[dict] = []
    for i, s in enumerate(track.sections):
        t = float(s.get("downbeat_s", s.get("start_s", 0.0)))
        ok = abs(float(s.get("snap_error_bars") or 0.0)) <= float(params["section_snap_max_bars"]) and not s.get("loose")
        if ok and m <= t <= dur - m:
            out.append({"t": t, "kind": "section", "label_section": s.get("label"), "ref": f"section{i}"})
    for j, ln in enumerate(track.lines):
        rest = _rest_before(track, ln)
        if rest < float(params["line_rest_beats"]) * per:
            continue
        t = ln["start_s"] - float(params["line_lead_s"])
        if not (m <= t <= dur - m):
            continue
        kind = "reentry" if rest >= float(params["reentry_rest_beats"]) * per else "line"
        out.append({"t": t, "kind": kind, "rest_beats": rest / per, "ref": f"line{j}"})
    out.sort(key=lambda p: p["t"])
    # due positivi a meno di 120 ms sono lo stesso punto: si tiene il primo (il verso col pickup prima del downbeat)
    merged: list[dict] = []
    for p in out:
        if merged and p["t"] - merged[-1]["t"] < 0.12:
            merged[-1].setdefault("also", []).append(p["kind"])
            continue
        merged.append(p)
    return merged


def plausible_boundaries(track: TrackData) -> np.ndarray:
    """Tutti gli istanti che potrebbero essere confini (anche quelli esclusi dai positivi): i negativi li evitano."""
    ts = [float(s.get("start_s", 0.0)) for s in track.sections] + [float(s.get("downbeat_s", 0.0)) for s in track.sections]
    ts += [ln["start_s"] for ln in track.lines]
    return np.asarray(sorted(ts), float)


def _far(t: float, ref: np.ndarray, tol: float) -> bool:
    return ref.size == 0 or float(np.min(np.abs(ref - t))) > tol


def _section_bounds(track: TrackData) -> np.ndarray:
    return np.asarray(sorted(float(s.get("downbeat_s", s.get("start_s", 0.0))) for s in track.sections), float)


def shifted_negatives(track: TrackData, positives: Sequence[dict], *, seed: int = 0,
                      params: Mapping[str, Any] = PARAMS) -> list[dict]:
    """Negativi controllati per ogni positivo (stesso brano). Ogni negativo porta ``anchor`` = indice del positivo."""
    rng = np.random.default_rng(seed)
    per, dur, m = track.period_s, track.duration_s, float(params["margin_s"])
    tol = float(params["exclusion_s"])
    excl = np.concatenate([plausible_boundaries(track), np.asarray([p["t"] for p in positives], float)])
    db = track.downbeats_s
    sec = _section_bounds(track)
    words = [w for w in track.words if w["off_s"] - w["on_s"] >= float(params["min_word_s"])]
    near = float(params["near_beats"]) * per
    out: list[dict] = []

    def ok(t: float) -> bool:
        return m <= t <= dur - m and _far(t, excl, tol)

    for a, p in enumerate(positives):
        t0 = p["t"]
        for typ, k in (("offbeat", 0.5), ("beat", 1.0), ("halfbar", 2.0), ("bar", 4.0)):
            for sgn in (-1, 1):
                t = t0 + sgn * k * per
                if ok(t):
                    out.append({"t": t, "kind": typ, "anchor": a, "shift_s": t - t0})
        # metà parola: la parola più vicina con almeno 120 ms, centro della parola
        cands = [(abs((w["on_s"] + w["off_s"]) / 2 - t0), w) for w in words if abs((w["on_s"] + w["off_s"]) / 2 - t0) <= near]
        for _, w in sorted(cands, key=lambda c: c[0]):
            t = 0.5 * (w["on_s"] + w["off_s"])
            if ok(t):
                out.append({"t": t, "kind": "midword", "anchor": a, "shift_s": t - t0, "word": w["text"]})
                break
        # downbeat dentro un verso
        inside = []
        for d in db[np.abs(db - t0) <= near]:
            for ln in track.lines:
                if ln["start_s"] + 0.5 * per < d < ln["end_s"] - 0.25 * per:
                    inside.append(float(d))
                    break
        for d in sorted(inside, key=lambda d: abs(d - t0)):
            if ok(d):
                out.append({"t": d, "kind": "midline", "anchor": a, "shift_s": d - t0})
                break
        # downbeat a metà sezione: battuta 3 di un gruppo di 4 dall'inizio di sezione, >= 2 battute dai confini
        mids = []
        for i, d in enumerate(db):
            if abs(d - t0) > float(params["mid_section_bars"]) * 4 * per or sec.size == 0:
                continue
            prev = sec[sec <= d + 1e-6]
            if prev.size == 0:
                continue
            bars_in = round((d - prev[-1]) / (4 * per))
            if bars_in % 4 == 2 and float(np.min(np.abs(sec - d))) >= 2 * 4 * per - 1e-3:
                mids.append(float(d))
        for d in sorted(mids, key=lambda d: abs(d - t0)):
            if ok(d):
                out.append({"t": d, "kind": "mid_section", "anchor": a, "shift_s": d - t0})
                break
        for _ in range(20):
            t = float(rng.uniform(m, dur - m))
            if ok(t):
                out.append({"t": t, "kind": "random", "anchor": a, "shift_s": t - t0})
                break
    return out


def build_points(tracks: Sequence[TrackData], *, seed: int = 0, params: Mapping[str, Any] = PARAMS) -> list[dict]:
    """Dataset di punti: ``id``, ``slug``, ``group``, ``t``, ``label`` (1 positivo / 0 negativo), ``kind``, ``anchor``."""
    rows: list[dict] = []
    for ti, tr in enumerate(tracks):
        pos = natural_points(tr, params)
        neg = shifted_negatives(tr, pos, seed=seed + ti, params=params)
        for i, p in enumerate(pos):
            rows.append({"id": f"{tr.slug}:p{i}", "slug": tr.slug, "group": tr.group, "t": round(p["t"], 4), "label": 1,
                         "kind": p["kind"], "anchor": f"{tr.slug}:p{i}", "ref": p.get("ref")})
        for j, n in enumerate(neg):
            rows.append({"id": f"{tr.slug}:n{j}", "slug": tr.slug, "group": tr.group, "t": round(n["t"], 4), "label": 0,
                         "kind": n["kind"], "anchor": f"{tr.slug}:p{n['anchor']}", "shift_s": round(n["shift_s"], 4)})
    return rows


def tempo_compatible(bpm_a: float, bpm_b: float, tol: float = PARAMS["pair_tempo_tol"]) -> bool:
    for f in (1.0, 2.0, 0.5):
        if abs(bpm_b * f / bpm_a - 1.0) <= tol:
            return True
    return False


def build_pairs(points: Sequence[dict], tracks: Sequence[TrackData], *, seed: int = 0, within_per_track: int = 24,
                cross_per_pair: int = 4, exit_neg_types: int = 3, params: Mapping[str, Any] = PARAMS) -> list[dict]:
    """Giunte artificiali (uscita da X a t1 -> ingresso in Y a t2).

    Positivo: entrambi i punti naturali. Negativo: lo stesso con l'ingresso (``side='entry'``, tutti i tipi) o
    l'uscita (``side='exit'``, ``exit_neg_types`` tipi a caso) spostati. ``scope='within'`` (stesso brano, salto
    >= 8 battute) o ``'cross'`` (brani diversi con tempo compatibile). Le coppie sono le stesse per i due modelli
    e servono al controllo "stesso brano vs brani diversi".
    """
    rng = np.random.default_rng(seed)
    by_slug: dict[str, dict] = {}
    for r in points:
        d = by_slug.setdefault(r["slug"], {"pos": [], "neg": {}})
        if r["label"] == 1:
            d["pos"].append(r)
        else:
            d["neg"].setdefault(r["anchor"], {}).setdefault(r["kind"], []).append(r)
    tmap = {t.slug: t for t in tracks}
    out: list[dict] = []

    def emit(scope: str, p1: dict, p2: dict) -> None:
        base = {"scope": scope, "x": p1["slug"], "y": p2["slug"], "groups": sorted({p1["group"], p2["group"]})}
        pid = f"{scope}:{p1['id']}>{p2['id']}"
        out.append({**base, "id": pid, "t1": p1["t"], "t2": p2["t"], "p1": p1["id"], "p2": p2["id"], "label": 1,
                    "kind": "natural", "side": None})
        for typ, lst in by_slug[p2["slug"]]["neg"].get(p2["id"], {}).items():
            n = lst[int(rng.integers(len(lst)))]
            out.append({**base, "id": f"{pid}|entry:{typ}", "t1": p1["t"], "t2": n["t"], "p1": p1["id"], "p2": n["id"],
                        "label": 0, "kind": typ, "side": "entry"})
        exits = list(by_slug[p1["slug"]]["neg"].get(p1["id"], {}).items())
        for k in rng.permutation(len(exits))[:exit_neg_types]:
            typ, lst = exits[int(k)]
            n = lst[int(rng.integers(len(lst)))]
            out.append({**base, "id": f"{pid}|exit:{typ}", "t1": n["t"], "t2": p2["t"], "p1": n["id"], "p2": p2["id"],
                        "label": 0, "kind": typ, "side": "exit"})

    jump = float(params["pair_min_jump_bars"])
    for slug, d in sorted(by_slug.items()):
        pos = d["pos"]
        per = tmap[slug].period_s
        cand = [(a, b) for a in pos for b in pos if a is not b and abs(b["t"] - a["t"]) >= jump * 4 * per]
        for k in rng.permutation(len(cand))[:within_per_track]:
            emit("within", *cand[int(k)])
    slugs = sorted(by_slug)
    for x in slugs:
        for y in slugs:
            if x == y or tmap[x].group == tmap[y].group or not tempo_compatible(tmap[x].bpm, tmap[y].bpm,
                                                                                   float(params["pair_tempo_tol"])):
                continue
            px, py = by_slug[x]["pos"], by_slug[y]["pos"]
            for _ in range(cross_per_pair):
                if px and py:
                    emit("cross", px[int(rng.integers(len(px)))], py[int(rng.integers(len(py)))])
    return out


# ======================================================================= split per artista
def group_folds(groups: Sequence[str]) -> list[str]:
    """Leave-one-artist-out: un fold per gruppo d'artista (ordinato)."""
    return sorted(set(groups))


def check_disjoint(train_slugs: Iterable[str], test_slugs: Iterable[str], track_groups: Mapping[str, str]) -> None:
    tr, te = set(train_slugs), set(test_slugs)
    if tr & te:
        raise AssertionError(f"brani in train e test: {sorted(tr & te)}")
    gtr, gte = {track_groups[s] for s in tr}, {track_groups[s] for s in te}
    if gtr & gte:
        raise AssertionError(f"artisti in train e test: {sorted(gtr & gte)}")


# ======================================================================= cache per brano e feature
MERT_LAYERS = (6, 12)
MERT_HOP_S = 12 / 75.0  # 75 Hz di MERT-v1-95M, media su 12 frame (0,16 s)
DSP_FPS = 50.0
DSP_BANDS = ((20.0, 150.0), (150.0, 500.0), (500.0, 2000.0), (2000.0, 8000.0))
DSP_COLS = ("low_db", "lowmid_db", "mid_db", "high_db", "flux", "vocal_db")


def _span(n: int, hop: float, a: float, b: float) -> slice:
    i0 = max(0, int(math.floor(a / hop + 1e-9)))
    i1 = min(n, max(i0 + 1, int(math.ceil(b / hop - 1e-9))))
    return slice(i0, i1)


def _mean_rows(x: np.ndarray, hop: float, a: float, b: float) -> np.ndarray:
    s = _span(len(x), hop, a, b)
    seg = x[s].astype(np.float32)
    return seg.mean(axis=0) if len(seg) else np.zeros(x.shape[1:], np.float32)


def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return 1.0 - float(a @ b) / (na * nb)


def mert_side(mert: Mapping[int, np.ndarray], t: float, side: str, long_s: float = 2.0, short_s: float = 0.48) -> dict:
    """Medie MERT per layer di una finestra prima (``pre``) o dopo (``post``) ``t``."""
    out = {}
    for L, x in mert.items():
        if side == "pre":
            out[(L, "long")] = _mean_rows(x, MERT_HOP_S, t - long_s, t)
            out[(L, "short")] = _mean_rows(x, MERT_HOP_S, t - short_s, t)
        else:
            out[(L, "long")] = _mean_rows(x, MERT_HOP_S, t, t + long_s)
            out[(L, "short")] = _mean_rows(x, MERT_HOP_S, t, t + short_s)
    return out


def mert_point_block(mert: Mapping[int, np.ndarray], t_pre: float, t_post: Optional[float] = None,
                     mert_post: Optional[Mapping[int, np.ndarray]] = None) -> tuple[np.ndarray, np.ndarray]:
    """(compatte, alta dimensione) tra il contesto prima di ``t_pre`` e quello dopo ``t_post`` (stesso brano se
    ``mert_post`` è None). Compatte: distanza coseno e norma della differenza (lunga 2 s, corta 0,48 s) per layer;
    alta dimensione: differenze post-pre (lunga e corta) per layer."""
    t_post = t_pre if t_post is None else t_post
    pre = mert_side(mert, t_pre, "pre")
    post = mert_side(mert if mert_post is None else mert_post, t_post, "post")
    comp, hi = [], []
    for L in sorted(mert):
        for w in ("long", "short"):
            a, b = pre[(L, w)], post[(L, w)]
            comp += [_cos_dist(a, b), float(np.linalg.norm(b - a))]
            hi.append(b - a)
    return np.asarray(comp, np.float32), np.concatenate(hi).astype(np.float32)


def dsp_side(dsp: np.ndarray, t: float, side: str) -> np.ndarray:
    hop = 1.0 / DSP_FPS
    if side == "pre":
        lo, sh = _mean_rows(dsp, hop, t - 2.0, t), _mean_rows(dsp, hop, t - 0.25, t)
    else:
        lo, sh = _mean_rows(dsp, hop, t, t + 2.0), _mean_rows(dsp, hop, t, t + 0.25)
    return np.concatenate([lo, sh])


def dsp_point_block(dsp: np.ndarray, t_pre: float, t_post: Optional[float] = None,
                    dsp_post: Optional[np.ndarray] = None) -> np.ndarray:
    """Energie per banda, flusso spettrale e voce (stem) ai due lati e salti; picco d'attacco vicino al punto."""
    t_post = t_pre if t_post is None else t_post
    other = dsp if dsp_post is None else dsp_post
    pre, post = dsp_side(dsp, t_pre, "pre"), dsp_side(other, t_post, "post")
    hop = 1.0 / DSP_FPS
    flux_col = DSP_COLS.index("flux")
    s = _span(len(other), hop, t_post - 0.04, t_post + 0.06)
    ctx = _span(len(other), hop, t_post - 4.0, t_post + 4.0)
    med = max(float(np.median(other[ctx, flux_col].astype(np.float32))), 0.0) + 1e-3
    peak = float(other[s, flux_col].astype(np.float32).max()) / med if s.stop > s.start else 0.0
    voc = DSP_COLS.index("vocal_db")
    vs = _span(len(dsp), hop, t_pre - 0.12, t_pre)
    vp = _span(len(other), hop, t_post, t_post + 0.12)
    v_edge = [float(dsp[vs, voc].astype(np.float32).mean()), float(other[vp, voc].astype(np.float32).mean())]
    return np.concatenate([pre, post, post - pre, [math.log1p(max(peak, 0.0))], v_edge]).astype(np.float32)


def grid_block(track: TrackData, t: float) -> np.ndarray:
    """Fase del battito e della battuta (seno/coseno). Non usa sezioni né parole (eviterebbe la fuga d'etichetta)."""
    per = track.period_s
    db = track.downbeats_s
    i = max(0, track.bar_of(t))
    ph_bar = ((t - db[i]) / (4 * per)) % 1.0 if db.size else 0.0
    ph_beat = (ph_bar * 4) % 1.0
    f = []
    for ph, ks in ((ph_beat, (1,)), (ph_bar, (1, 2))):
        for k in ks:
            f += [math.sin(2 * math.pi * k * ph), math.cos(2 * math.pi * k * ph)]
    return np.asarray(f, np.float32)


@dataclass
class TrackCache:
    """Cache compatta di un brano: MERT medio (float16, 6,25 Hz, layer 6 e 12), DSP (float16, 50 Hz)."""
    mert: Optional[dict[int, np.ndarray]] = None
    dsp: Optional[np.ndarray] = None
    cocola_prepost: Optional[dict[float, float]] = None

    @classmethod
    def load(cls, run_dir: Path, slug: str) -> "TrackCache":
        c = cls()
        mp = Path(run_dir) / "cache" / "mert" / f"{slug}.npz"
        if mp.is_file():
            z = np.load(mp)
            c.mert = {L: z[f"l{L}"] for L in MERT_LAYERS if f"l{L}" in z.files}
        dp = Path(run_dir) / "cache" / "dsp" / f"{slug}.npz"
        if dp.is_file():
            c.dsp = np.load(dp)["x"]
        return c


#: colonne della voce (stem Demucs) nel vettore di :func:`dsp_point_block`: 6 gruppi da len(DSP_COLS) (pre lunga/corta,
#: post lunga/corta, salti lungo/corto), poi il picco d'attacco e i due bordi vocali
_N = len(DSP_COLS)
DSP_VOX_IDX = np.asarray([g * _N + DSP_COLS.index("vocal_db") for g in range(6)] + [6 * _N + 1, 6 * _N + 2])
DSP_MIX_IDX = np.asarray([i for i in range(6 * _N + 3) if i not in set(DSP_VOX_IDX.tolist())])
ALL_BLOCKS = ("mert_c", "mert_h", "dsp_mix", "vox", "grid")

#: varianti dei modelli. ``vox`` (energia dello stem vocale) riproduce in parte la definizione dei positivi di verso
#: (pausa della voce >= 1 battito prima dell'attacco): le varianti ``*_novox`` misurano cosa resta senza.
FEATURE_BLOCKS = {
    "grid": ("grid",),
    "dsp_novox": ("dsp_mix",),
    "dsp": ("dsp_mix", "vox"),
    "mert": ("mert_c", "mert_h"),
    "audio_novox": ("mert_c", "mert_h", "dsp_mix"),
    "audio": ("mert_c", "mert_h", "dsp_mix", "vox"),
    "full": ("mert_c", "mert_h", "dsp_mix", "vox", "grid"),
}


def _split_dsp(d: np.ndarray) -> dict[str, np.ndarray]:
    return {"dsp_mix": d[DSP_MIX_IDX], "vox": d[DSP_VOX_IDX]}


def point_features(track: TrackData, cache: TrackCache, t: float) -> dict[str, np.ndarray]:
    """Blocchi di feature di un punto. Senza cache MERT i blocchi ``mert_*`` sono NaN (i modelli che li usano si
    astengono: vedi :class:`PointScorer`)."""
    if cache.mert is not None:
        mc, mh = mert_point_block(cache.mert, t)
    else:
        mc = np.full(4 * len(MERT_LAYERS), np.nan, np.float32)
        mh = np.full(2 * 768 * len(MERT_LAYERS), np.nan, np.float32)
    return {"mert_c": mc, "mert_h": mh, **_split_dsp(dsp_point_block(cache.dsp, t)), "grid": grid_block(track, t)}


def pair_features(tx: TrackData, cx: TrackCache, t1: float, ty: TrackData, cy: TrackCache, t2: float) -> dict[str, np.ndarray]:
    """Giunta X(…t1) + Y(t2…): contesto prima da X e dopo da Y (salti tra i due lati), più la fase di griglia dei due punti."""
    mc, mh = mert_point_block(cx.mert, t1, t2, cy.mert)
    d = dsp_point_block(cx.dsp, t1, t2, cy.dsp)
    g = np.concatenate([grid_block(tx, t1), grid_block(ty, t2)])
    return {"mert_c": mc, "mert_h": mh, **_split_dsp(d), "grid": g}


def stack(rows: Sequence[Mapping[str, np.ndarray]], blocks: Sequence[str]) -> dict[str, np.ndarray]:
    return {b: np.stack([r[b] for r in rows]).astype(np.float32) for b in blocks}


# ======================================================================= modello
class BlockLogistic:
    """Logistica L2 (bilanciata) su blocchi standardizzati; i blocchi con più di ``pca_dims`` colonne passano da una
    PCA stimata sul solo training. Salvabile/caricabile in npz senza sklearn al momento dell'uso."""

    def __init__(self, blocks: Sequence[str], *, pca_dims: int = 24, c: float = 0.3):
        self.blocks, self.pca_dims, self.c = tuple(blocks), int(pca_dims), float(c)
        self.params: dict[str, np.ndarray] = {}

    def _transform(self, X: Mapping[str, np.ndarray]) -> np.ndarray:
        cols = []
        for b in self.blocks:
            x = (X[b] - self.params[f"{b}.mu"]) / self.params[f"{b}.sd"]
            if f"{b}.pca" in self.params:
                x = x @ self.params[f"{b}.pca"].T
                x = x / self.params[f"{b}.pca_sd"]
            cols.append(x)
        return np.concatenate(cols, axis=1)

    def fit(self, X: Mapping[str, np.ndarray], y: np.ndarray) -> "BlockLogistic":
        from sklearn.linear_model import LogisticRegression

        for b in self.blocks:
            x = X[b].astype(np.float64)
            mu, sd = x.mean(0), x.std(0) + 1e-6
            self.params[f"{b}.mu"], self.params[f"{b}.sd"] = mu, sd
            if x.shape[1] > self.pca_dims:
                z = (x - mu) / sd
                from sklearn.utils.extmath import randomized_svd

                _, _, vt = randomized_svd(z - z.mean(0), n_components=self.pca_dims, n_iter=4, random_state=0)
                comp = vt
                self.params[f"{b}.pca"] = comp
                self.params[f"{b}.pca_sd"] = (z @ comp.T).std(0) + 1e-6
        Z = self._transform(X)
        clf = LogisticRegression(C=self.c, max_iter=4000, class_weight="balanced")
        clf.fit(Z, y)
        self.params["coef"], self.params["intercept"] = clf.coef_[0], np.asarray(clf.intercept_)
        return self

    def decision(self, X: Mapping[str, np.ndarray]) -> np.ndarray:
        return self._transform(X) @ self.params["coef"] + float(self.params["intercept"][0])

    def predict_proba(self, X: Mapping[str, np.ndarray]) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.decision(X)))

    def save(self, path: Path, meta: Optional[dict] = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{k.replace(".", "__"): v for k, v in self.params.items()},
                 _meta=np.asarray(json.dumps({"blocks": self.blocks, "pca_dims": self.pca_dims, "c": self.c, **(meta or {})})))

    @classmethod
    def load(cls, path: Path) -> "BlockLogistic":
        z = np.load(path)
        meta = json.loads(str(z["_meta"]))
        m = cls(meta["blocks"], pca_dims=meta["pca_dims"], c=meta["c"])
        m.params = {k.replace("__", "."): z[k] for k in z.files if k != "_meta"}
        m.meta = meta
        return m


# ======================================================================= metriche
def auc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    """AUC di Mann-Whitney (pareggi = ½)."""
    p, n = np.asarray(pos, float), np.asarray(neg, float)
    if p.size == 0 or n.size == 0:
        return None
    allv = np.concatenate([p, n])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), float)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i: j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    rp = ranks[: p.size].sum()
    return float((rp - p.size * (p.size + 1) / 2) / (p.size * n.size))


def paired_win_rate(scores: Mapping[str, float], rows: Sequence[dict], kind: str) -> Optional[float]:
    """Quota dei negativi di tipo ``kind`` con punteggio minore del proprio positivo (stesso brano, ancora)."""
    wins, n = 0.0, 0
    for r in rows:
        if r["label"] == 0 and r["kind"] == kind and r["anchor"] in scores and r["id"] in scores:
            s_pos, s_neg = scores[r["anchor"]], scores[r["id"]]
            wins += 1.0 if s_pos > s_neg else (0.5 if s_pos == s_neg else 0.0)
            n += 1
    return wins / n if n else None


def bootstrap_auc_by_group(pos: Sequence[tuple[str, float]], neg: Sequence[tuple[str, float]], *, n_boot: int = 500,
                           seed: int = 0) -> tuple[Optional[float], Optional[float]]:
    """IC95 dell'AUC ricampionando i brani (cluster bootstrap)."""
    rng = np.random.default_rng(seed)
    slugs = sorted({s for s, _ in pos} | {s for s, _ in neg})
    if len(slugs) < 3:
        return None, None
    P: dict[str, list[float]] = {}
    N: dict[str, list[float]] = {}
    for s, v in pos:
        P.setdefault(s, []).append(v)
    for s, v in neg:
        N.setdefault(s, []).append(v)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(slugs, size=len(slugs), replace=True)
        pp = [v for s in pick for v in P.get(s, [])]
        nn = [v for s in pick for v in N.get(s, [])]
        a = auc(pp, nn)
        if a is not None:
            vals.append(a)
    if not vals:
        return None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3:
        return None

    def rank(v):
        order = v.argsort(kind="mergesort")
        r = np.empty(len(v), float)
        sv = v[order]
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1] == sv[i]:
                j += 1
            r[order[i: j + 1]] = 0.5 * (i + j)
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


# ======================================================================= API
def cue_candidates(track: TrackData, params: Mapping[str, Any] = PARAMS) -> list[dict]:
    """Istanti candidati di un brano: ogni battito e mezzo battito della griglia, più l'inizio di ogni verso."""
    m, dur, per = 2.5, track.duration_s, track.period_s
    out = []
    for i, d in enumerate(track.downbeats_s):
        for k in range(8):
            t = float(d + k * per / 2)
            if m <= t <= dur - m:
                out.append({"t": round(t, 4), "bar": i, "beat": k / 2 + 1, "what": "downbeat" if k == 0 else
                            ("beat" if k % 2 == 0 else "offbeat")})
    for j, ln in enumerate(track.lines):
        t = ln["start_s"] - float(params["line_lead_s"])
        if m <= t <= dur - m:
            out.append({"t": round(t, 4), "bar": track.bar_of(t), "beat": None, "what": "line_start",
                        "rest_beats": round(_rest_before(track, ln) / per, 2)})
    out.sort(key=lambda c: c["t"])
    return out


_CUE_MEM: dict[str, dict] = {}


def _cue_doc(slug: str, cache_dir: Path) -> Optional[dict]:
    key = f"{cache_dir}:{slug}"
    if key not in _CUE_MEM:
        p = Path(cache_dir) / f"{slug}.json"
        if not p.is_file():
            return None
        _CUE_MEM[key] = _read(p)
    return _CUE_MEM[key]


def cue_naturalness(track: str, t: float | Sequence[float], *, variant: str = "full", cache_dir: Path | str = CUE_DIR,
                    tol_s: float = 0.06, run_dir: Path | str = RUN_DIR, source_run: Path | str = SOURCE_RUN) -> Any:
    """Probabilità che ``t`` (s, tempo del brano) sia un punto in cui il brano ``track`` si articola da solo.

    Uso: ingresso di B (``t`` = dove B diventa udibile) e uscita di A (``t`` = dove A smette). Ritorna un dict
    ``{"t", "p", "variant", "status", "nearest_t", "what", "oof"}`` (o una lista per più istanti). ``status``:

    * ``cached``: candidato precalcolato entro ``tol_s`` (battito, mezzo battito o inizio verso);
    * ``computed``: calcolato al volo dalle cache del brano (MERT/DSP) col modello salvato;
    * ``unavailable``: né cache né feature: ``p`` è None (il chiamante deve astenersi, non indovinare).

    ``variant``: ``full`` (audio + fase della griglia, default) o ``audio`` (solo audio). ``oof`` = True se il
    punteggio viene da un modello che non ha visto l'artista del brano (vale per i brani di addestramento).
    """
    ts = [float(t)] if np.isscalar(t) else [float(v) for v in t]
    doc = _cue_doc(track, Path(cache_dir))
    res = []
    todo = []
    for i, tv in enumerate(ts):
        if doc:
            cand = doc["candidates"]
            arr = np.asarray([c["t"] for c in cand])
            k = int(np.argmin(np.abs(arr - tv))) if arr.size else -1
            if k >= 0 and abs(arr[k] - tv) <= tol_s and cand[k].get(f"p_{variant}") is not None:
                c = cand[k]
                res.append({"t": tv, "p": c[f"p_{variant}"], "variant": variant, "status": "cached", "nearest_t": c["t"],
                            "what": c["what"], "oof": bool(doc.get("oof"))})
                continue
        res.append(None)
        todo.append(i)
    if todo:
        try:
            mf = ((doc or {}).get("model_files") or {}).get(variant)
            scorer = PointScorer(Path(run_dir), variant=variant, model_file=mf, source_run=Path(source_run))
            scorer.oof = bool(doc.get("oof")) if doc else False
            ps = scorer.score(track, [ts[i] for i in todo])
            for i, p in zip(todo, ps):
                res[i] = {"t": ts[i], "p": p, "variant": variant, "status": "computed" if p is not None else "unavailable",
                          "nearest_t": None, "what": None, "oof": scorer.oof}
        except (FileNotFoundError, KeyError, ValueError) as e:
            for i in todo:
                res[i] = {"t": ts[i], "p": None, "variant": variant, "status": "unavailable", "nearest_t": None,
                          "what": None, "oof": None, "why": str(e)[:200]}
    return res[0] if np.isscalar(t) else res


class PointScorer:
    """Punteggio dei punti di un brano dalle cache (``runs/insong_v1/cache``) e dal modello salvato.

    ``model_file`` (relativo a ``run_dir``): quello registrato nella cache del brano; per i brani di addestramento è il
    modello del fold che ha escluso il suo artista. Senza, il modello addestrato su tutti i brani."""

    def __init__(self, run_dir: Path = RUN_DIR, *, variant: str = "full", model_file: Optional[str] = None,
                 source_run: Path = SOURCE_RUN):
        self.run_dir, self.variant, self.source_run = Path(run_dir), variant, Path(source_run)
        self.oof = False
        path = self.run_dir / (model_file or f"model/point_{variant}.npz")
        self.model = BlockLogistic.load(path)

    def score(self, slug: str, times: Sequence[float]) -> list[Optional[float]]:
        tr = load_track(slug, source_run=self.source_run)
        cache = TrackCache.load(self.run_dir, slug)
        need_mert = any(b.startswith("mert") for b in self.model.blocks)
        if (need_mert and cache.mert is None) or cache.dsp is None:
            return [None] * len(times)
        rows = [point_features(tr, cache, float(t)) for t in times]
        return [float(p) for p in self.model.predict_proba(stack(rows, self.model.blocks))]


def _safe(s: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "none").lower()).strip("_")


# ======================================================================= tensione -> risoluzione
#: Compito aggiunto dall'orchestratore (21/10/2026, dai tornei dei cue: "A deve caricare, B deve risolvere").
#: Dentro un brano, la fine di ciò che precede l'arrivo di un ritornello (strofa che sale, pre-ritornello, build) è un
#: esempio di "carica" e l'arrivo del ritornello è un esempio di "arrivo". [H] Proxy operativo: ritornello (allin1) =
#: arrivo. NON è un'equivalenza ritornello = risoluzione (AGENTS.md): è l'etichetta disponibile senza ascolti.
TENSION_ARRIVAL_LABELS = ("chorus",)
TENSION_NEG_TYPES = {
    "exit": {
        "early_verse": "2 o 4 battute dopo l'inizio di una strofa (contesto prima a bassa tensione)",
        "mid_verse": "a metà strofa, ad almeno 4 battute dal ritornello successivo",
        "intro": "2-4 battute dopo l'inizio del brano/intro",
        "post_arrival": "2 battute dopo l'arrivo del ritornello (la carica è già scaricata)",
    },
    "entry": {
        "mid_verse": "ripresa a metà strofa (downbeat ad almeno 4 battute dai confini)",
        "verse_start": "inizio di strofa (confine naturale ma senza arrivo: diagnostico)",
        "intro": "inizio o prime battute dell'intro",
        "late_arrival": "2 battute dopo l'arrivo del ritornello (arrivo già avvenuto)",
    },
}
FORM_DIRS = (".cache/dj_set_v3_events_plan/form", ".cache/track_cards/form")


def load_form(slug: str, n_bars: int, root: Path = ROOT) -> Optional[dict]:
    """Etichette dj_form.3 per battuta (cache dei runner, stessa griglia di dj_select); None se assenti/incoerenti."""
    for d in FORM_DIRS:
        p = Path(root) / d / f"{slug}.json"
        if p.is_file():
            f = _read(p).get("form") or {}
            if str(f.get("version", "")).startswith("musicmix.dj_form.") and len(f.get("labels") or []) == n_bars:
                return {"labels": f["labels"], "n_layers": f.get("n_layers_smooth") or f.get("n_layers"),
                        "drops": set(f.get("drops") or []), "dropouts": {int(x["end_bar"]) for x in f.get("kick_dropouts") or []},
                        "version": f.get("version"), "path": str(p.relative_to(root))}
    return None


def tension_examples(track: TrackData) -> dict[str, list[dict]]:
    """Esempi di carica (uscita) e di arrivo (ingresso) dentro un brano, con i negativi per tipo.

    Positivo (``exit`` e ``entry``): downbeat d'inizio di un ritornello preceduto da una sezione diversa.
    Il modello d'uscita guarda solo il contesto **prima** di t, quello d'ingresso solo **dopo** t."""
    per, dur = track.period_s, track.duration_s
    bar = 4 * per
    secs = sorted(({"t": float(s.get("downbeat_s", s.get("start_s", 0.0))), "label": str(s.get("label"))}
                   for s in track.sections), key=lambda s: s["t"])
    db = track.downbeats_s
    arrivals = [s["t"] for i, s in enumerate(secs) if s["label"] in TENSION_ARRIVAL_LABELS and i > 0
                and secs[i - 1]["label"] not in TENSION_ARRIVAL_LABELS]
    arr = np.asarray(arrivals, float)
    out: dict[str, list[dict]] = {"exit": [], "entry": []}
    lo, hi = 8 * bar, dur - 4 * bar

    def far_from_arrivals(t: float, before_bars: float = 4.0) -> bool:
        return arr.size == 0 or not np.any((arr - t > -1e-6) & (arr - t < before_bars * bar))

    for i, t in enumerate(arrivals):
        prev = next(s["label"] for s in reversed(secs) if s["t"] < t - 1e-6)
        if t >= 8 * bar and t <= dur - 8 * bar:
            out["exit"].append({"t": t, "label": 1, "kind": "pre_arrival", "prev": prev, "ref": f"arrival{i}"})
            out["entry"].append({"t": t, "label": 1, "kind": "arrival", "prev": prev, "ref": f"arrival{i}"})
        for role, kind in (("exit", "post_arrival"), ("entry", "late_arrival")):
            tt = t + 2 * bar
            if lo <= tt <= hi:
                out[role].append({"t": tt, "label": 0, "kind": kind, "ref": f"arrival{i}"})
    for j, s in enumerate(secs):
        end = secs[j + 1]["t"] if j + 1 < len(secs) else dur
        if s["label"] == "verse":
            for k in (2, 4):
                tt = s["t"] + k * bar
                if lo <= tt <= hi and tt < end - 2 * bar and far_from_arrivals(tt):
                    out["exit"].append({"t": tt, "label": 0, "kind": "early_verse", "ref": f"section{j}"})
            if lo <= s["t"] <= hi:
                out["entry"].append({"t": s["t"], "label": 0, "kind": "verse_start", "ref": f"section{j}"})
            mids = [float(d) for d in db if s["t"] + 4 * bar - 1e-6 <= d <= end - 4 * bar + 1e-6]
            if mids:
                tt = mids[len(mids) // 2]
                if lo <= tt <= hi:
                    out["entry"].append({"t": tt, "label": 0, "kind": "mid_verse", "ref": f"section{j}"})
                    if far_from_arrivals(tt):
                        out["exit"].append({"t": tt, "label": 0, "kind": "mid_verse", "ref": f"section{j}"})
        if s["label"] == "intro":
            for k, role in ((2, "exit"), (4, "exit"), (0, "entry"), (2, "entry")):
                tt = s["t"] + k * bar
                ok = (tt >= 2 * bar if role == "exit" else tt >= 4 * per) and tt < end and tt <= hi
                if ok and far_from_arrivals(tt):
                    out[role].append({"t": tt, "label": 0, "kind": "intro", "ref": f"section{j}"})
    return out


def _bar_means(x: np.ndarray, hop: float, t0: float, bar: float, n: int, cols: Sequence[int]) -> np.ndarray:
    """Media per battuta di ``n`` battute a partire da ``t0`` (n<0: le battute prima di t0)."""
    rows = []
    rng_ = range(n, 0) if n < 0 else range(n)
    for k in rng_:
        rows.append(_mean_rows(x[:, cols], hop, t0 + k * bar, t0 + (k + 1) * bar))
    return np.asarray(rows, np.float32)


def _slope(y: np.ndarray) -> np.ndarray:
    if len(y) < 2:
        return np.zeros(y.shape[1:], np.float32)
    xs = np.arange(len(y), dtype=np.float32)
    xs = xs - xs.mean()
    return (xs[:, None] * (y - y.mean(0))).sum(0) / (xs ** 2).sum()


def tension_features(track: TrackData, cache: TrackCache, t: float, role: str, form: Optional[dict] = None,
                     chroma: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
    """Feature di un lato: ``role='exit'`` usa solo [t-8 battute, t), ``'entry'`` solo [t, t+4 battute).

    Blocchi: ``t_dsp`` (energia per banda per battuta, pendenze, flusso = densità d'attacchi, voce), ``t_form``
    (strati attivi di dj_form, pendenza, etichetta della battuta, drop/dropout), ``t_chroma`` (somiglianza del croma
    col profilo medio del brano: "ritorno a casa" armonico, proxy [H]), ``t_mert`` (media MERT layer 12 su 4 s)."""
    bar = 4 * track.period_s
    hop = 1.0 / DSP_FPS
    cols = list(range(len(DSP_COLS)))
    d = cache.dsp.astype(np.float32)
    if role == "exit":
        bm = _bar_means(d, hop, t, bar, -8, cols)                     # 8 battute prima, in ordine temporale
        last = _mean_rows(d, hop, t - bar / 2, t)
        feats = np.concatenate([bm[-1], bm[-2:].mean(0), bm.mean(0), _slope(bm), _slope(bm[-4:]), last,
                                bm[-2:].mean(0) - bm[:4].mean(0)])
    else:
        bm = _bar_means(d, hop, t, bar, 4, cols)
        first = _mean_rows(d, hop, t, t + bar / 2)
        hit = _mean_rows(d, hop, t, t + 0.1)
        feats = np.concatenate([bm[0], bm[:2].mean(0), bm.mean(0), _slope(bm), first, hit])
    out = {"t_dsp": feats.astype(np.float32)}
    # contrasto dentro il brano: uscita = salita nelle 8 battute; ingresso = 2 battute dopo meno le 2 battute prima di t
    # nello stesso brano (un arrivo è un cambiamento; nel mix ciò che precede è A, quindi è una proprietà del punto di B)
    if role == "exit":
        con = bm[-2:].mean(0) - bm[:2].mean(0)
    else:
        con = bm[:2].mean(0) - _bar_means(d, hop, t, bar, -2, cols).mean(0)
    # forma
    nb = len(track.downbeats_s)
    i = max(0, track.bar_of(t + 1e-3))
    if form is not None:
        nl = np.asarray(form["n_layers"], float)
        labs = form["labels"]
        cats = ("intro", "build", "core", "groove", "breakdown", "outro")
        if role == "exit":
            seg = nl[max(0, i - 8): i] if i > 0 else nl[:1]
            lab = labs[max(0, i - 1)]
            ev = [float((i - 1) in form["dropouts"] or i in form["dropouts"]), float(i in form["drops"])]
        else:
            seg = nl[i: min(nb, i + 4)]
            lab = labs[min(i, len(labs) - 1)]
            ev = [float(i in form["drops"]), float(i in form["dropouts"])]
        seg = seg if len(seg) else np.zeros(1)
        f = [float(seg.mean()), float(seg[-1] if role == "exit" else seg[0]), float(_slope(seg[:, None])[0])] + ev
        f += [float(lab == c) for c in cats]
        out["t_form"] = np.asarray(f, np.float32)
        if role == "exit":
            dl = float(nl[max(0, i - 2): max(1, i)].mean() - nl[max(0, i - 8): max(1, i - 6)].mean())
        else:
            dl = float(nl[i: min(nb, i + 2)].mean() - nl[max(0, i - 2): max(1, i)].mean()) if i > 0 else 0.0
        out["t_contrast"] = np.concatenate([con, [dl]]).astype(np.float32)
    else:
        out["t_form"] = np.full(11, np.nan, np.float32)
        out["t_contrast"] = np.concatenate([con, [np.nan]]).astype(np.float32)
    if chroma is not None:
        prof = chroma.astype(np.float32).mean(0)
        w = (t - 2 * bar, t) if role == "exit" else (t, t + 2 * bar)
        c = _mean_rows(chroma, 0.1, *w)
        out["t_chroma"] = np.asarray([1.0 - _cos_dist(c, prof), float(c.max() / (c.sum() + 1e-6))], np.float32)
    else:
        out["t_chroma"] = np.full(2, np.nan, np.float32)
    if cache.mert is not None and 12 in cache.mert:
        w = (t - 4.0, t) if role == "exit" else (t, t + 4.0)
        out["t_mert"] = _mean_rows(cache.mert[12], MERT_HOP_S, *w).astype(np.float32)
    else:
        out["t_mert"] = np.full(768, np.nan, np.float32)
    return out


TENSION_BLOCKS = {
    "dsp": ("t_dsp",),
    "dsp_form": ("t_dsp", "t_form", "t_chroma"),
    "contrast": ("t_dsp", "t_form", "t_chroma", "t_contrast"),
    "full": ("t_dsp", "t_form", "t_chroma", "t_contrast", "t_mert"),
}
TENSION_ALL_BLOCKS = ("t_dsp", "t_form", "t_chroma", "t_contrast", "t_mert")


def tension_score(track: str, t: float | Sequence[float], role: str, *, variant: str = "dsp_form",
                  run_dir: Path | str = RUN_DIR, source_run: Path | str = SOURCE_RUN, model_file: Optional[str] = None) -> Any:
    """Probabilità del modello di carica (``role='exit'``) o di arrivo (``role='entry'``) nel brano ``track`` a ``t``.

    Modello: ``runs/insong_v1/model/tension_{role}_{variant}.npz`` (o ``model_file``); per un brano di addestramento
    il chiamante può passare il modello del fold che ha escluso il suo artista. None se mancano le cache."""
    run_dir = Path(run_dir)
    m = BlockLogistic.load(run_dir / (model_file or f"model/tension_{role}_{variant}.npz"))
    tr = load_track(track, source_run=Path(source_run))
    cache = TrackCache.load(run_dir, track)
    if cache.dsp is None:
        return None if np.isscalar(t) else [None] * len(t)
    form = load_form(track, len(tr.downbeats_s))
    cp = run_dir / "cache" / "chroma" / f"{track}.npz"
    chroma = np.load(cp)["x"] if cp.is_file() else None
    ts = [float(t)] if np.isscalar(t) else [float(v) for v in t]
    rows = [tension_features(tr, cache, v, role, form, chroma) for v in ts]
    X = stack(rows, m.blocks)
    if any(np.isnan(X[b]).any() for b in X):
        ps = [None] * len(ts)
    else:
        ps = [float(p) for p in m.predict_proba(X)]
    return ps[0] if np.isscalar(t) else ps


def tension_at_exit(track: str, t: float | Sequence[float], **kw) -> Any:
    """Quanto il contesto **prima** di ``t`` somiglia alla carica che precede l'arrivo di un ritornello nel brano."""
    return tension_score(track, t, "exit", **kw)


def resolution_at_entry(track: str, t: float | Sequence[float], **kw) -> Any:
    """Quanto il contesto **dopo** ``t`` somiglia all'arrivo di un ritornello (proxy [H] di "B risolve")."""
    return tension_score(track, t, "entry", **kw)
