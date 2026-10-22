#!/usr/bin/env python3
"""dj_phrase v3 (proposta F5): onset sillabici dallo stem vocale invece dei tempi Whisper.

Ohriner 2019 p. 187 [A]: l'inizio di una sillaba è il momento in cui la sua vocale raggiunge metà
dell'ampiezza massima, misurato su a cappella (p. 186). Qui: stem vocale Demucs →
``dj_phrase.stem_syllable_onsets`` → ``dj_phrase.with_stem_onsets`` (aggancio delle sillabe Whisper,
finestra ±60 ms) e confronto, sugli stessi brani di ``runs/dj_phrase_v2`` (modelli congelati di
``runs/dj_phrase_v1``, pilastro w = 1 come v2), delle misure fini di Ohriner prima e dopo:
nPVI (Huron pp. 188-189), grooviness (Ohriner Tab. 5.2), disallineamento dalla griglia e fase vocale
(Ohriner pp. 185-196). Riferimenti MCFlow da ``runs/dj_phrase_v2/mcflow_books_eval.json``.

Legge solo; scrive ``runs/dj_phrase_v3_stem_onsets/compare.json`` (niente modelli completi).

    python tools/run_dj_phrase_v3_stem_onsets.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from musicmix import dj_phrase as P  # noqa: E402
from musicmix import dj_render as dj  # noqa: E402
import run_dj_phrase_v1 as V1  # noqa: E402

V1_DIR = ROOT / "runs/dj_phrase_v1/tracks"
OUT = ROOT / "runs/dj_phrase_v3_stem_onsets"
MCFLOW_EVAL = ROOT / "runs/dj_phrase_v2/mcflow_books_eval.json"


def measures(model: dict) -> dict[str, Any]:
    sy = P.model_syllables(model)
    ends = {L["last_word"] for L in model.get("lines") or []}
    out: dict[str, Any] = {"npvi": P.syllable_npvi(sy, line_end_words=ends)}
    # MCFlow nPVI is computed on onsets quantized to the sixteenth grid (books_v2_checks: r = 0,998 with the grid):
    # the comparable value here uses the quantized positions of our syllables, not continuous time.
    q = [{k: v for k, v in s.items() if k != "t_s"} | {"beat_index": s["abs16"] / 4.0} for s in sy]
    out["npvi_quantized"] = P.syllable_npvi(q, line_end_words=ends)
    f = P.flow_features(model, sylls=sy)
    out["flow_grooviness"] = f.get("grooviness")
    out["off_grid_mean16"] = f.get("off_grid_mean16")
    osy = P.ohriner_syllables_from_model(model)
    if len(osy) >= 8:
        og = P.ohriner_global_features(osy, tempo_bpm=60.0 / float(model["period_s"]))
        out.update({k: og.get(k) for k in ("grooviness", "dupleness", "adherence", "syls_per_second", "saturation")})
    ph = P.vocal_phase_alignment(model)
    out["phase"] = {k: ph.get(k) for k in ("n_onsets", "onsets", "misalignment", "phase_beats", "phase_ms", "misalignment_phase",
                                            "bur", "explained_share")}
    return out


def stat(vals) -> dict[str, Any]:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    return {"n": int(v.size), "median": float(np.median(v)) if v.size else None, "mean": float(v.mean()) if v.size else None,
            "q25": float(np.quantile(v, 0.25)) if v.size else None, "q75": float(np.quantile(v, 0.75)) if v.size else None}


def mcflow_refs() -> dict[str, Any]:
    if not MCFLOW_EVAL.is_file():
        return {}
    d = json.loads(MCFLOW_EVAL.read_text())
    songs = d.get("songs") or []
    return {"npvi_mean_sd": (d.get("books_v2_checks") or {}).get("means", {}).get("npvi"),
            "flow_grooviness": stat([s.get("grooviness") for s in songs]),
            "note": "MCFlow annotato a mano (onset quantizzati alla griglia): disallineamento e fase non definiti",
            "ohriner_tab_5_2": "grooviness .51, adherence .67, duple .45, sill/s 5,09 (75 strofe)",
            "ohriner_phase": "pp. 190-196: aggiustamento ottimo -6,84/60 di battito (voce in ritardo ~0,11 battiti); "
                             "2/3 delle battute senza miglioramento dallo swing (p. 195)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    t0 = time.time()
    files = sorted(V1_DIR.glob("*.json"))[: args.limit]
    for f in files:
        doc = json.loads(f.read_text())
        inp = V1.track_inputs(doc["pool"], doc["slug"])
        if inp is None or "stem" not in inp:
            skipped.append({"slug": doc["slug"], "why": "stem or analysis missing on this machine"})
            continue
        stem, sr = dj.load_track(inp["stem"])
        m2 = P.reanalyze_with_pillar(doc["model"], 1.0)
        on = P.stem_syllable_onsets(stem, sr)
        del stem
        m3, info = P.with_stem_onsets(m2, onsets=on)
        raw = P.vocal_phase_alignment(m2, onsets_s=on["onsets_s"])
        rows.append({"lang": doc["model"].get("lang"), "stem_raw_phase": {k: raw.get(k) for k in ("n_onsets", "misalignment", "phase_beats", "phase_ms",
                                                                 "misalignment_phase", "bur", "explained_share")},"slug": doc["slug"], "pool": doc["pool"], "asr_reliable": (doc["model"].get("asr_quality") or {}).get("reliable", True),
                     "bpm": 60.0 / float(m2["period_s"]), "snap": info, "whisper": measures(m2), "stem": measures(m3)})
        fmt = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
        print(f"{doc['slug']:40s} snapped {fmt(info['snapped_share'])}  nPVI {fmt(rows[-1]['whisper']['npvi'])} -> "
              f"{fmt(rows[-1]['stem']['npvi'])}  mis {fmt(rows[-1]['whisper']['phase']['misalignment'])} -> "
              f"{fmt(rows[-1]['stem']['phase']['misalignment'])}", flush=True)
    rel = [r for r in rows if r["asr_reliable"]]
    keys = ("npvi", "npvi_quantized", "flow_grooviness", "grooviness", "dupleness", "adherence", "syls_per_second", "off_grid_mean16")
    summary: dict[str, Any] = {"n_tracks": len(rows), "n_reliable": len(rel)}
    for v in ("whisper", "stem"):
        summary[v] = {k: stat([r[v].get(k) for r in rel]) for k in keys}
        summary[v]["phase"] = {k: stat([r[v]["phase"].get(k) for r in rel]) for k in
                               ("misalignment", "phase_beats", "phase_ms", "misalignment_phase", "bur", "explained_share")}
    summary["stem_raw_phase"] = {k: stat([r["stem_raw_phase"].get(k) for r in rel]) for k in
                                 ("misalignment", "phase_beats", "phase_ms", "misalignment_phase", "bur", "explained_share")}
    summary["snap"] = {k: stat([r["snap"].get(k) for r in rel]) for k in ("snapped_share", "shift_ms_median_abs", "shift_ms_median")}
    summary["paired"] = {}
    for k in ("npvi", "npvi_quantized", "flow_grooviness", "grooviness"):
        d = [r["stem"][k] - r["whisper"][k] for r in rel if r["stem"].get(k) is not None and r["whisper"].get(k) is not None]
        summary["paired"][k + "_stem_minus_whisper"] = stat(d)
    d = [r["stem"]["phase"]["misalignment"] - r["whisper"]["phase"]["misalignment"] for r in rel
         if r["stem"]["phase"].get("misalignment") is not None and r["whisper"]["phase"].get("misalignment") is not None]
    summary["paired"]["misalignment_stem_minus_whisper"] = stat(d)
    by_lang: dict[str, Any] = {}
    for lab, sel in (("en", [r for r in rel if r.get("lang") == "en"]), ("other", [r for r in rel if r.get("lang") != "en"])):
        by_lang[lab] = {"n": len(sel)}
        for v in ("whisper", "stem"):
            by_lang[lab][v] = {k: stat([r[v].get(k) for r in sel])["median"] for k in ("npvi_quantized", "grooviness", "adherence")}
            by_lang[lab][v].update({"phase_" + k: stat([r[v]["phase"].get(k) for r in sel])["median"]
                                    for k in ("misalignment", "phase_beats", "explained_share")})
        by_lang[lab]["stem_raw"] = {"phase_" + k: stat([r["stem_raw_phase"].get(k) for r in sel])["median"]
                                    for k in ("misalignment", "phase_beats", "explained_share")}
    summary["by_language"] = by_lang
    summary["random_onsets_misalignment"] = ("3,75/60 di battito = media della distanza dalla sedicesima più vicina per onset "
                                             "uniformi (nessun aggancio alla griglia)")
    doc = {"schema": "musicmix.dj_phrase_v3_stem_onsets.1", "params": {k: p.value for k, p in P.STEM_ONSET_PARAMS.items()},
           "method": "Ohriner 2019 p. 187 (p-centre) sullo stem Demucs; sillabe Whisper agganciate (dj_phrase.with_stem_onsets)",
           "mcflow_reference": mcflow_refs(), "summary": summary, "tracks": rows, "skipped": skipped,
           "elapsed_s": time.time() - t0}
    (OUT / "compare.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=float))
    print(json.dumps(summary, indent=1, default=float)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
