#!/usr/bin/env python3
"""Galleria d'ascolto DJ v1: tre versioni dello stesso punto di cambio per piano.

Legge (senza modificarli) ``runs/dj_plan_v1`` (tracks/, pairs.json, summary.json),
il catalogo del pool, gli audio canonici, gli stem vocali Demucs e le parole ASR;
scrive soltanto in una cartella NUOVA (default ``runs/dj_gallery_v1``)::

    clips/<item>.wav + clips/<item>.json   render tenuti (pass / warn), metadati + QA
    rejected/<item>.wav + .json            render con QA fail (scartati, con motivi)
    review_manifest.json                   schema ``musicmix.dj-review.1``
    gallery_log.json                       scelte, aggiustamenti dei cue, scarti

Scelte (fissate a priori, non adattate ai giudizi d'ascolto):

* Piani: solo coppie ``compatible`` di dj_plan_v1. Per ogni coppia il piano
  migliore, poi il miglior piano con un cue di uscita di A diverso (>= 4
  battute di distanza) per le coppie col secondo piano più alto, fino a
  ``--max-plans`` (6), al massimo 2 per coppia. I piani vengono ricalcolati con
  ``dj_select.plan_pair`` sulle analisi salvate in ``tracks/`` e si verifica che
  i primi 5 coincidano con ``pairs.json`` (che ne conserva solo 5 per coppia).
* Griglie: quelle regolarizzate di dj_select (``tracks/<slug>.json``: periodo,
  fase, offset di battuta della griglia a tempo costante; downbeat =
  ``grid.downbeats_s``), ricostruite con ``dj_select.grid_times``; mai le
  Beat This! grezze. I cue del piano cadono esattamente su questi downbeat.
* Convenzione dei cue: dj_select usa "start" (``a_out_s`` = downbeat di A in cui
  B comincia a suonare, ``b_in_s`` = downbeat di B suonato in quel momento),
  quindi i blend sono resi con ``DJRenderConfig(cue_convention="start")``.
  "Stesso punto di cambio" = il downbeat ``b_in`` di B arriva sul downbeat
  ``a_out`` di A in tutte le versioni: blend (inizio sovrapposizione), cut e
  drop_in (cambio secco), filter_out (sweep sulle 2 battute prima di a_out),
  echo_out (eco agganciato al downbeat precedente ``a_out - 1 battuta``,
  ``entry_bars=1``: B entra pulito su a_out).
* Uscita in dissolvenza: se la loudness di A nei 4 s prima di ``a_out``
  (BS.1770 K-weighting, un blocco non gated) è più di 6 dB sotto la mediana
  della sua loudness short-term (3 s, passo 1 s, sull'intero brano), si sceglie
  il precedente cue di uscita valido di A secondo ``dj_select.cue_candidates``
  (inizio frase su downbeat) che non cade in dissolvenza, mantenendo l'ingresso
  di B, e il piano viene rivalutato con ``dj_select.plan_transition``.
* Versioni: (a) il tipo del piano (cut / blend N / blend_duck = blend 4 con
  ducking automatico della voce di A); (b) natura diversa: se (a) è un blend,
  echo_out quando la voce di A è attiva nell'ultimo beat prima del cue o B
  entra su un ritornello (allin1), altrimenti cut sul downbeat; se (a) è un cut,
  blend 4 se ammesso dalle regole di dj_select (tempo, tonalità, voci) altrimenti
  filter_out 2 battute; (c) drop_in se B entra su un ritornello secondo
  allin1 e il cut non rischia una parola di A (e non sarebbe identico a (b)),
  altrimenti un blend di 8/16 battute diverso da (a) se ammesso, altrimenti
  echo_out.
* QA: ``dj_qa.evaluate_render`` con i deck (``return_decks=True``), gli stem
  vocali in coordinate della clip, le parole ASR di A e Beat This! small0.
  Per echo_out / filter_out i guadagni udibili dei deck (non esposti da
  dj_effects) sono ricostruiti dai metadati dell'effetto. ``fail`` = scartato.

    python tools/run_dj_gallery_v1.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicmix import dj_effects as fx  # noqa: E402
from musicmix import dj_qa  # noqa: E402
from musicmix import dj_render as dj  # noqa: E402
from musicmix import dj_select as djs  # noqa: E402
from musicmix.paths import localize  # noqa: E402

SCHEMA = "musicmix.dj-review.1"
RUN = "dj_gallery_v1"
SR = 48000
FADE_WINDOW_S = 4.0
FADE_MAX_DB = 6.0
ST_WINDOW_S = 3.0
ST_HOP_S = 1.0
ECHO_DELAY_BEATS = 1.0
FILTER_SWEEP_BARS = 2
SECOND_PLAN_MIN_BAR_GAP = 4

KIND_LABEL = {
    "cut": "taglio sul downbeat",
    "blend": "blend {bars} battute",
    "echo_out": "echo out",
    "filter_out": "filter out",
    "drop_in": "drop sul ritornello",
}


# ----------------------------------------------------------------- utilities
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(o: Any) -> Any:
    return dj_qa._jsonable(o)


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# -------------------------------------------------------------------- tracks
def regular_grid(t: dict) -> tuple[np.ndarray, np.ndarray]:
    """Beat e downbeat della griglia regolarizzata di dj_select per la traccia ``t``.

    ``dj_plan_pool`` salva i downbeat (arrotondati a 0,1 ms e limitati all'intervallo
    coperto dal provider) e i parametri della griglia: qui si rigenera la griglia a
    piena precisione con ``dj_select.grid_times`` e la si limita allo stesso
    intervallo; i beat coprono le stesse battute.
    """
    tempo = t["tempo"]
    fit = {"period_s": tempo["period_s"], "phase_s": tempo["phase_s"], "bar_offset_beats": tempo["bar_offset_beats"]}
    beats, downs = djs.grid_times(fit, t["duration_s"])
    stored = np.asarray(t["grid"]["downbeats_s"], dtype=float)
    downs = downs[(downs >= stored[0] - 1e-3) & (downs <= stored[-1] + 1e-3)]
    if len(downs) != len(stored) or np.max(np.abs(downs - stored)) > 1e-3:
        raise RuntimeError(f"{t['slug']}: regenerated grid does not match tracks/{t['slug']}.json")
    bar = djs.BEATS_PER_BAR * tempo["period_s"]
    beats = beats[(beats >= downs[0] - 1e-3) & (beats < downs[-1] + bar - 1e-3)]
    return beats, downs


def short_term_profile(audio: np.ndarray, sr: int) -> np.ndarray:
    import pyloudnorm

    w, hop = int(ST_WINDOW_S * sr), int(ST_HOP_S * sr)
    meter = pyloudnorm.Meter(sr, block_size=ST_WINDOW_S)
    vals = []
    for s in range(0, len(audio) - w + 1, hop):
        v = meter.integrated_loudness(audio[s : s + w])
        vals.append(v if math.isfinite(v) else -np.inf)
    return np.asarray(vals)


def block_lufs(audio: np.ndarray, sr: int, t0: float, t1: float) -> float | None:
    import pyloudnorm

    seg = audio[max(0, int(t0 * sr)) : int(t1 * sr)]
    if len(seg) < int(0.4 * sr):
        return None
    v = float(pyloudnorm.Meter(sr, block_size=len(seg) / sr).integrated_loudness(seg))
    return v if math.isfinite(v) else None


class Track:
    def __init__(self, slug: str, plan_dir: Path, catalog: dict):
        self.slug = slug
        self.analysis = load_json(plan_dir / "tracks" / f"{slug}.json")
        entry = next(t for t in catalog["tracks"] if t["slug"] == slug)
        wav = localize(self.analysis["inputs"]["canonical_wav"])
        if sha256(wav) != self.analysis["track_id"] or entry["track_id"] != self.analysis["track_id"]:
            raise RuntimeError(f"{slug}: canonical audio identity mismatch")
        self.audio, sr = dj.load_track(wav)
        vjson = localize(entry["vocals_path"])
        from musicmix.stems import vocal_stem_path  # 21/10/2026: stem dalla provenienza della fase vocals, verificato

        stem = vocal_stem_path(entry, check_sha=False)
        self.vocals, sr_v = dj.load_track(stem)
        if sr != SR or sr_v != SR:
            raise RuntimeError(f"{slug}: sample rate {sr}/{sr_v} != {SR}")
        self.words = load_json(vjson)["words"]
        self.beats, self.downbeats = regular_grid(self.analysis)
        self.vocal_threshold = dj_qa.stem_threshold(self.vocals, SR)
        st = short_term_profile(self.audio, SR)
        self.st_median = float(np.median(st[np.isfinite(st) & (st > -70)]))
        self.provenance = {
            "slug": slug, "title": self.analysis.get("title"), "artist": self.analysis.get("artist"),
            "audio": rel(wav), "audio_sha256": self.analysis["track_id"],
            "vocal_stem": rel(stem), "vocal_stem_sha256": sha256(stem),
            "vocals_json": rel(vjson),
            "analysis": rel(plan_dir / "tracks" / f"{slug}.json"),
            "analysis_sha256": sha256(plan_dir / "tracks" / f"{slug}.json"),
            "grid": {"source": "dj_select regularised constant-tempo grid (tracks/<slug>.json)",
                     "provider": self.analysis["tempo"]["provider"], "confidence": self.analysis["tempo"]["confidence"],
                     "bpm": self.analysis["tempo"]["bpm"], "n_downbeats": int(len(self.downbeats)),
                     "n_beats": int(len(self.beats))},
        }

    @property
    def title(self) -> str:
        return self.analysis.get("title") or self.slug

    def fade_check(self, cue_s: float) -> dict:
        lv = block_lufs(self.audio, SR, cue_s - FADE_WINDOW_S, cue_s)
        drop = (self.st_median - lv) if lv is not None else None
        return {"lufs_4s_before_cue": lv, "track_short_term_median_lufs": self.st_median,
                "below_median_db": drop, "in_fade": bool(drop is None or drop > FADE_MAX_DB)}


# --------------------------------------------------------------------- plans
def pick_plans(tracks_an: dict, pairs_doc: dict, max_plans: int, log: list) -> list[dict]:
    stored = {(p["a"], p["b"]): p for p in pairs_doc["pairs"]}
    compatible = [p for p in pairs_doc["pairs"] if p["compatibility"]["status"] == "compatible"]
    full = {}
    for p in compatible:
        r = djs.plan_pair(tracks_an[p["a"]], tracks_an[p["b"]], top_k=100000)
        key = lambda q: (q["a_out_bar"], q["b_in_bar"], q["type"], round(q["score"], 6))  # noqa: E731
        if [key(q) for q in r["plans"][:5]] != [key(q) for q in stored[(p["a"], p["b"])]["plans"]]:
            raise RuntimeError(f"{p['a']}->{p['b']}: recomputed plans differ from pairs.json")
        full[(p["a"], p["b"])] = r
    firsts = [(k, r["plans"][0], 1) for k, r in full.items()]
    seconds = []
    for k, r in full.items():
        a0 = r["plans"][0]["a_out_bar"]
        nxt = next((q for q in r["plans"][1:] if abs(q["a_out_bar"] - a0) >= SECOND_PLAN_MIN_BAR_GAP), None)
        if nxt is not None:
            seconds.append((k, nxt, 2))
    firsts.sort(key=lambda x: -x[1]["score"])
    seconds.sort(key=lambda x: -x[1]["score"])
    chosen = (firsts + seconds)[:max_plans]
    log.append({"plan_selection": [{"pair": f"{k[0]}->{k[1]}", "rank_in_pair": rk, "a_out_bar": q["a_out_bar"],
                                    "b_in_bar": q["b_in_bar"], "type": q["type"], "score": q["score"],
                                    "in_pairs_json_top5": any(q["a_out_bar"] == s["a_out_bar"] and q["b_in_bar"] == s["b_in_bar"]
                                                              for s in stored[k]["plans"])}
                                   for k, q, rk in chosen]})
    return [{"a": k[0], "b": k[1], "plan": q, "rank_in_pair": rk, "compatibility": full[k]["compatibility"]}
            for k, q, rk in chosen]


def adjust_for_fade(A: Track, B: Track, sel: dict) -> tuple[dict, dict | None, dict]:
    """Sposta l'uscita di A al precedente cue valido se ``a_out`` cade in dissolvenza."""
    plan = sel["plan"]
    check = A.fade_check(plan["a_out_s"])
    if not check["in_fade"]:
        return plan, None, check
    outs = sorted((c for c in djs.cue_candidates(A.analysis, "out") if c["bar_index"] < plan["a_out_bar"]),
                  key=lambda c: -c["bar_index"])
    inn = next(c for c in djs.cue_candidates(B.analysis, "in") if c["bar_index"] == plan["b_in_bar"])
    tried = []
    for c in outs:
        chk = A.fade_check(c["time_s"])
        tried.append({"bar": c["bar_index"], "time_s": c["time_s"], **chk})
        if not chk["in_fade"]:
            new = djs.plan_transition(A.analysis, B.analysis, c, inn, sel["compatibility"])
            adj = {"rule": f"A loudness in the {FADE_WINDOW_S:.0f} s before a_out more than {FADE_MAX_DB:.0f} dB below its "
                           "short-term median: previous valid phrase-start cue of A (dj_select.cue_candidates)",
                   "original": {"a_out_bar": plan["a_out_bar"], "a_out_s": plan["a_out_s"], "type": plan["type"],
                                "blend_bars": plan["blend_bars"], "score": plan["score"], "fade_check": check},
                   "replacement": {"a_out_bar": new["a_out_bar"], "a_out_s": new["a_out_s"], "type": new["type"],
                                   "blend_bars": new["blend_bars"], "score": new["score"], "fade_check": chk},
                   "tried": tried}
            return new, adj, chk
    raise RuntimeError(f"{A.slug}: no earlier out cue outside the fade")


# ------------------------------------------------------------------ versions
def vocal_bars(t: dict, start: int, n: int) -> np.ndarray:
    return np.asarray(t["bars"]["vocal"], dtype=float)[start:start + n] >= djs.VOCAL_BAR_ACTIVE


def blend_allowed(A: dict, B: dict, ja: int, jb: int, n: int, comp: dict) -> tuple[bool, str]:
    """Stesse regole di dj_select.plan_transition per un blend di ``n`` battute."""
    t, k = comp["tempo"], comp["key"]
    if not t["compatible"]:
        return False, "tempo oltre +-8%"
    max_bars = 16 if t["long_blend_ok"] else 8
    if k["relation"] in ("diagonal", "clash", "unknown") or not k["confident"]:
        max_bars = min(max_bars, 4)
    if n > max_bars:
        return False, f"blend {n} oltre il massimo {max_bars} per tempo/tonalita"
    if ja + n > len(A["phrases"]) or jb + n > len(B["phrases"]):
        return False, "battute insufficienti"
    clash = int(np.sum(vocal_bars(A, ja, n) & vocal_bars(B, jb, n)))
    if clash > djs.CLASH_TOLERATED_BARS:
        return False, f"{clash} battute con entrambe le voci"
    return True, f"ammesso (voci sovrapposte {clash} battute)"


def b_enters_on_chorus(B: dict, jb: int) -> bool:
    return any(s["bar_index"] == jb and s["label"] == "chorus" and s.get("source") == "allin1" for s in B["sections"])


def a_word_risk(A: dict, ja: int) -> bool:
    lb = A["bars"].get("vocal_last_beat") or []
    return bool(ja > 0 and ja - 1 < len(lb) and lb[ja - 1] >= djs.VOCAL_BAR_ACTIVE)


def choose_versions(A: dict, B: dict, plan: dict, comp: dict) -> tuple[list[dict], dict]:
    ja, jb = plan["a_out_bar"], plan["b_in_bar"]
    chorus = b_enters_on_chorus(B, jb)
    risk = a_word_risk(A, ja)
    ctx = {"b_enters_on_allin1_chorus": chorus, "a_vocal_in_last_beat_before_cue": risk}
    out: list[dict] = []
    if plan["type"] == "cut":
        out.append({"role": "a", "kind": "cut", "why": "tipo suggerito dal piano"})
        ok4, why4 = blend_allowed(A, B, ja, jb, 4, comp)
        ctx["blend4"] = why4
        if ok4:
            out.append({"role": "b", "kind": "blend", "bars": 4, "why": f"alternativa al taglio: blend 4 {why4}"})
        else:
            out.append({"role": "b", "kind": "filter_out", "sweep_bars": FILTER_SWEEP_BARS,
                        "why": f"alternativa al taglio: blend 4 non ammesso ({why4}), filter out"})
        out.append({"role": "c", "kind": "echo_out", "why": "terza via: echo out sull'ultima battuta di A, B pulito sul cue"
                    + (" (drop_in escluso: identico al taglio)" if chorus else "")})
        return out, ctx
    n = 4 if plan["type"] == "blend_duck" else int(plan["blend_bars"])
    out.append({"role": "a", "kind": "blend", "bars": n, "why": "tipo suggerito dal piano"
                + (" (blend_duck: ducking della voce di A se lo scontro supera 2 s)" if plan["type"] == "blend_duck" else "")})
    if risk or chorus:
        out.append({"role": "b", "kind": "echo_out", "why": "alternativa al blend: echo out ("
                    + ("voce di A nell'ultimo beat prima del cue" if risk else "B entra su un ritornello: il taglio diventa la terza versione") + ")"})
    else:
        out.append({"role": "b", "kind": "cut", "why": "alternativa al blend: taglio sul downbeat"})
    if chorus and not risk:
        out.append({"role": "c", "kind": "drop_in", "why": "B entra sul primo downbeat di un ritornello (allin1)"})
        return out, ctx
    for m in ((16, 8) if n == 8 else (8, 16)):
        if m == n:
            continue
        ok, why = blend_allowed(A, B, ja, jb, m, comp)
        ctx[f"blend{m}"] = why
        if ok:
            out.append({"role": "c", "kind": "blend", "bars": m, "why": f"durata diversa: blend {m} {why}"})
            return out, ctx
    used = {v["kind"] for v in out}
    third = "echo_out" if "echo_out" not in used else "cut"
    out.append({"role": "c", "kind": third, "why": "terza via: nessun blend 8/16 alternativo ammesso"})
    return out, ctx


def version_label(v: dict) -> str:
    return KIND_LABEL[v["kind"]].format(bars=v.get("bars"))


# -------------------------------------------------------------------- render
def render_version(v: dict, A: Track, B: Track, plan: dict, prov: dict) -> tuple[dj.DJRenderResult, dict, dict]:
    grids = dict(beats_a=A.beats, downbeats_a=A.downbeats, beats_b=B.beats, downbeats_b=B.downbeats)
    ia = int(np.argmin(np.abs(A.downbeats - plan["a_out_s"])))
    a_out, b_in = float(A.downbeats[ia]), float(plan["b_in_s"])
    kind = v["kind"]
    params: dict[str, Any] = {"cue_out_a_s": a_out, "cue_in_b_s": b_in}
    if kind == "blend":
        cfg = dj.DJRenderConfig(cue_convention="start")
        params.update(bars=v["bars"], cue_convention="start", stretch_mode=cfg.stretch_mode, vocal_ducking="auto (>= 2 s clash)")
        res = dj.render_transition(A.audio, B.audio, **grids, cue_out_a=a_out, cue_in_b=b_in, kind="blend", bars=v["bars"],
                                   sample_rate=SR, vocals_a=A.vocals, vocals_b=B.vocals, config=cfg, provenance=prov,
                                   return_decks=True)
    elif kind == "cut":
        res = dj.render_transition(A.audio, B.audio, **grids, cue_out_a=a_out, cue_in_b=b_in, kind="cut", sample_rate=SR,
                                   provenance=prov, return_decks=True)
        params.update(crossfade_ms=dj.DJRenderConfig().crossfade_ms)
    elif kind == "drop_in":
        label = next(s["label"] for s in B.analysis["sections"] if s["bar_index"] == plan["b_in_bar"])
        res = fx.drop_in(A.audio, B.audio, **grids, cue_out_a=a_out, cue_in_b=b_in, cue_in_is_section_start=True,
                         section_label=label, sample_rate=SR, provenance=prov, return_decks=True)
        params.update(section_label=label, crossfade_ms=fx.DJEffectsConfig().crossfade_ms)
    elif kind == "echo_out":
        if ia < 1:
            raise dj.DJRenderError("echo_out: no A downbeat before a_out")
        echo_at = float(A.downbeats[ia - 1])
        params.update(echo_cue_out_a_s=echo_at, delay_beats=ECHO_DELAY_BEATS, entry_bars=1,
                      note="echo engaged on the downbeat one bar before a_out; B enters clean on a_out (same change point)")
        res = fx.echo_out(A.audio, B.audio, **grids, cue_out_a=echo_at, cue_in_b=b_in, delay_beats=ECHO_DELAY_BEATS,
                          entry_bars=1, sample_rate=SR, provenance=prov, return_decks=True)
    elif kind == "filter_out":
        params.update(sweep_bars=v["sweep_bars"], filter_in=False)
        res = fx.filter_out(A.audio, B.audio, **grids, cue_out_a=a_out, cue_in_b=b_in, sweep_bars=v["sweep_bars"],
                            sample_rate=SR, provenance=prov, return_decks=True)
    else:
        raise ValueError(kind)
    meta = res.metadata
    decks = dict(res.decks or {})
    gains_note = None
    n = len(res.audio)
    t = np.arange(n)
    if kind == "echo_out":
        dry_end = int(meta["effect"]["dry_a_cut_output_sample"])
        entry = int(meta["switch"]["output_sample"])
        decks["gains"] = dict(decks.get("gains", {}), fader_a=(t < dry_end).astype(float), fader_b=(t >= entry).astype(float))
        gains_note = "reconstructed from dj_effects metadata: A dry until dry_a_cut_output_sample, B from its entry; echo tail of A's vocal (-12 dB under B) not counted"
    elif kind == "filter_out":
        cut = int(meta["switch"]["output_sample"])
        decks["gains"] = dict(decks.get("gains", {}), fader_a=(t < cut).astype(float), fader_b=(t >= cut).astype(float))
        gains_note = "reconstructed from dj_effects metadata: A (high-passed) until the cut, B from the cut"
    return res, params, {"decks": decks, "gains_note": gains_note}


def markers(meta: dict) -> dict:
    sr = int(meta["sample_rate"])
    ov, sw = meta["overlap"], meta["switch"]
    return {"transition_start_s": ov["output_start_sample"] / sr, "switch_s": sw["output_sample"] / sr,
            "transition_end_s": ov["output_end_sample"] / sr}


def run_qa(res: dj.DJRenderResult, A: Track, B: Track, decks: dict, tracker) -> dict:
    n = len(res.audio)
    meta = res.metadata
    va, vb = dj_qa.stems_to_output(meta, n, A.vocals, B.vocals)
    words = dj_qa.words_to_output(meta, A.words, n)
    return dj_qa.evaluate_render(res.audio.astype(np.float64), meta, sample_rate=res.sample_rate, decks=decks,
                                 vocals_a=va, vocals_b=vb, vocal_thresholds=(A.vocal_threshold, B.vocal_threshold),
                                 words_a=words, tracker=tracker)


# ---------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default=str(ROOT / "runs/dj_plan_v1"))
    ap.add_argument("--out", default=str(ROOT / "runs" / RUN))
    ap.add_argument("--max-plans", type=int, default=6)
    args = ap.parse_args(argv)
    plan_dir, out = Path(args.plan), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty: refusing to overwrite")
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "rejected").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pairs_doc = load_json(plan_dir / "pairs.json")
    summary = load_json(plan_dir / "summary.json")
    catalog_path = localize(pairs_doc["catalog"])
    if not catalog_path.is_absolute():
        catalog_path = ROOT / catalog_path
    catalog = load_json(catalog_path)
    slugs = sorted({p[k] for p in pairs_doc["pairs"] if p["compatibility"]["status"] == "compatible" for k in ("a", "b")})
    analyses = {s: load_json(plan_dir / "tracks" / f"{s}.json") for s in slugs}
    log: list = []
    selection = pick_plans(analyses, pairs_doc, args.max_plans, log)
    tracks = {s: Track(s, plan_dir, catalog) for s in sorted({x[k] for x in selection for k in ("a", "b")})}
    tracker = dj_qa.BeatThisTracker()
    counts = {"pass": 0, "warn": 0, "fail": 0, "render_error": 0}
    groups, rejected = [], []
    for gi, sel in enumerate(selection, 1):
        A, B = tracks[sel["a"]], tracks[sel["b"]]
        plan, adj, fade = adjust_for_fade(A, B, sel)
        comp = sel["compatibility"]
        gid = f"g{gi:02d}_{A.slug}__{B.slug}_a{plan['a_out_bar']}_b{plan['b_in_bar']}"
        versions, ctx = choose_versions(A.analysis, B.analysis, plan, comp)
        print(f"[{gid}] {plan['type']} {plan['blend_bars']} -> " + ", ".join(version_label(v) for v in versions)
              + (f"  (a_out spostato da battuta {adj['original']['a_out_bar']})" if adj else ""), flush=True)
        prov = {"a": A.provenance, "b": B.provenance, "plan_run": rel(plan_dir), "plan": {k: plan[k] for k in (
            "a_out_s", "a_out_bar", "b_in_s", "b_in_bar", "type", "blend_bars", "score", "confidence")},
            "script": "tools/run_dj_gallery_v1.py", "cue_adjustment": adj}
        items = []
        for v in versions:
            iid = f"{gid}__{v['role']}_{v['kind']}" + (str(v["bars"]) if v["kind"] == "blend" else "")
            try:
                res, params, extra = render_version(v, A, B, plan, prov)
            except dj.DJRenderError as exc:
                counts["render_error"] += 1
                rejected.append({"group": gid, "id": iid, "kind": v["kind"], "stage": "render", "reasons": [str(exc)]})
                print(f"   RENDER ERROR {iid}: {exc}", flush=True)
                continue
            qa = run_qa(res, A, B, extra["decks"], tracker)
            verdict = qa["verdict"]
            counts[verdict] += 1
            dest = out / ("rejected" if verdict == "fail" else "clips")
            written = dj.write_render(res, dest, iid)
            meta = load_json(written["json"])
            mk = markers(res.metadata)
            record = {"gallery": {"schema": SCHEMA, "run": RUN, "group": gid, "item": iid, "role": v["role"], "kind": v["kind"],
                                  "label": version_label(v), "why": v["why"], "params": params, "markers_s": mk,
                                  "qa_inputs": {"decks": True, "vocal_stems": True, "asr_words_a": True,
                                                "tracker": qa["inputs"]["tracker"], "gains_note": extra["gains_note"]}},
                      "render": meta, "qa": qa}
            written["json"].write_text(json.dumps(jsonable(record), indent=1, ensure_ascii=False), encoding="utf-8")
            reasons = [f"{r['check']}: {r['message']}" for r in qa["reasons"]]
            print(f"   {verdict.upper():4s} {iid} ({res.metadata['clip']['duration_s']:.2f} s) " + "; ".join(reasons), flush=True)
            if verdict == "fail":
                rejected.append({"group": gid, "id": iid, "kind": v["kind"], "stage": "qa", "reasons": reasons,
                                 "audio_path": rel(written["wav"]), "json_path": rel(written["json"]), "sha256": written["sha256"]})
                continue
            items.append({
                "id": iid, "role": v["role"], "kind": v["kind"], "label": version_label(v), "why": v["why"], "params": params,
                "audio_path": rel(written["wav"]), "json_path": rel(written["json"]),
                "duration_s": res.metadata["clip"]["duration_s"], "sample_rate": res.sample_rate, **mk,
                "qa": {"verdict": verdict, "warn_reasons": reasons if verdict == "warn" else []},
                "sha256": written["sha256"],
            })
        a_secs = A.analysis["phrases"][plan["a_out_bar"]]
        groups.append({
            "id": gid,
            "pair": {"a": {"slug": A.slug, "title": A.title, "artist": A.analysis.get("artist")},
                     "b": {"slug": B.slug, "title": B.title, "artist": B.analysis.get("artist")}},
            "title": f"{A.title} → {B.title}",
            "bpm": {"a": A.analysis["tempo"]["bpm"], "b": B.analysis["tempo"]["bpm"],
                    "stretch_b_pct": comp["tempo"]["stretch_b_pct"]},
            "key": {"a": A.analysis["key"]["camelot"], "b": B.analysis["key"]["camelot"],
                    "a_name": A.analysis["key"]["key"], "b_name": B.analysis["key"]["key"], "relation": comp["key"]["relation"]},
            "A_out_s": plan["a_out_s"], "B_in_s": plan["b_in_s"], "A_out_bar": plan["a_out_bar"], "B_in_bar": plan["b_in_bar"],
            "sections": {"a": plan["a_section"], "b": plan["b_section"], "a_bar_in_section": a_secs["bar_in_section"]},
            "plan": {"type": plan["type"], "blend_bars": plan["blend_bars"], "score": plan["score"],
                     "confidence": plan["confidence"], "rank_in_pair": sel["rank_in_pair"]},
            "reasons": list(comp["reasons"]) + list(plan["reasons"]),
            "version_context": ctx,
            "fade_check": fade,
            "cue_adjustment": adj,
            "items": items,
        })
    manifest = {
        "schema_version": SCHEMA,
        "title": "MusicMix — galleria DJ v1: tre versioni dello stesso cambio",
        "run": RUN,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"plan_run": rel(plan_dir), "pairs_sha256": sha256(plan_dir / "pairs.json"),
                   "summary_sha256": sha256(plan_dir / "summary.json"), "catalog": rel(catalog_path)},
        "conventions": {
            "audio_path": "relative to the repository root",
            "markers": "seconds in the clip: transition_start_s / switch_s / transition_end_s from the renderer metadata (overlap, switch)",
            "change_point": "B's b_in downbeat lands on A's a_out downbeat in every version (echo_out: echo on the bar before)",
            "clip_rules": "cut/drop_in 10 s + 10 s; blend/effects 10 s + whole transition + 10 s (AGENTS.md, 20/10/2026)",
            "grids": "dj_select regularised grids (runs/dj_plan_v1/tracks), not raw Beat This!",
            "qa": "dj_qa.evaluate_render with decks, vocal stems, ASR words of A and Beat This! small0; fail = discarded",
            "no_stay_items": "continuing A is not offered as an item to rate (AGENTS.md)",
        },
        "counts": counts,
        "groups": groups,
        "rejected": rejected,
    }
    (out / "review_manifest.json").write_text(json.dumps(jsonable(manifest), indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "gallery_log.json").write_text(json.dumps(jsonable({"log": log, "counts": counts, "rejected": rejected,
                                                               "elapsed_s": time.time() - t0,
                                                               "summary_top_plans": summary.get("top_plans")}),
                                                      indent=1, ensure_ascii=False), encoding="utf-8")
    if not any((out / "rejected").iterdir()):
        shutil.rmtree(out / "rejected")
    print(json.dumps(counts), f"{time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
