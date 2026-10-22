#!/usr/bin/env python3
"""dj_phrase v2 (libri): effetto delle aggiunte da Ohriner, Krims, Huron e Margulis sui brani di runs/dj_phrase_v1.

Legge i modelli congelati di ``runs/dj_phrase_v1/tracks`` (non li modifica) e scrive solo in
``runs/dj_phrase_v2``:

* ``tracks/<slug>.json``: modello con il termine del pilastro centrale (Ohriner p. 110; peso 1, valutato
  held-out su MCFlow), uscite v1 vs v2 per ogni cue-out di dj_select, descrittori di flow (Ohriner,
  Krims, nPVI), allineamento vocale (Ohriner cap. 8);
* ``compare.json``: totali (uscite cambiate, versi, posizioni delle fini verso contro MCFlow e contro
  la lunghezza di frase di Ohriner), stili di Krims, e sulle coppie ordinate dello stesso pool il
  cambio del piano migliore con ``flow_weight`` = 1 (diagnostica, default 0).

    python tools/run_dj_phrase_v2.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from musicmix import dj_phrase as P  # noqa: E402
from musicmix import dj_render as dj  # noqa: E402
from musicmix.paths import external_dir  # noqa: E402
from musicmix import dj_select as djs  # noqa: E402
import run_dj_phrase_v1 as V1  # noqa: E402

V1_DIR = ROOT / "runs/dj_phrase_v1/tracks"
OUT = ROOT / "runs/dj_phrase_v2"
SR = P.SR
PILLAR_W = 1.0


def exit_ok(e: dict) -> bool:
    """Same acceptance as runs/dj_phrase_v1 (tools/run_dj_phrase_v1.py, guard 'new')."""
    return bool(e["status"] in ("clean", "tail", "a_pickup", "instrumental")
                and (e["status"] == "instrumental" or (e.get("p_line_end") or 0) >= P.PARAMS["line_end_threshold"].value)
                and e.get("acoustic") != "stem_active_in_window")


def pillar_term(terms) -> Any:
    if isinstance(terms, dict):
        return terms.get("pillar")
    return next((t.get("log_lr") for t in terms or [] if isinstance(t, dict) and t.get("name") == "pilastro"), None)


def end_positions(model: dict) -> list[int]:
    sr, grid = model["sample_rate"], P.Grid(np.asarray(model["downbeats_s"], float), float(model["period_s"]))
    out = []
    for L in model["lines"][:-1]:  # the last 'line' closes the track by definition
        w = model["words"][L["last_word"]]
        bar, pos = grid.locate((w.get("last_syl_on") or w["on"]) / sr)
        if bar >= 0:
            out.append(int(round(pos * 4)) % 16)
    return out


def line_durations(model: dict) -> list[float]:
    sr = model["sample_rate"]
    return [(L["end_sample"] - L["start_sample"]) / sr for L in model["lines"][:-1]]


def hist16(pos: list[int]) -> np.ndarray:
    h = np.bincount(np.asarray(pos, int) % 16, minlength=16).astype(float) if pos else np.zeros(16)
    return h / h.sum() if h.sum() else h


def mcflow_annotated_ends() -> dict:
    """Annotated line-end positions in MCFlow (prosodic phrases, break >= 3) and phrase durations."""
    import mcflow_flow_stats as MF

    files = sorted(((external_dir() / "computational-cognitive-musicology-lab/mcflow") / "Humdrum").glob("*.rap"))
    lex = P.Lexicon()
    pos, durs = [], []
    for f in files:
        s = MF.song(f, lex)
        if not s:
            continue
        pos += end_positions(s["model"])
        durs += line_durations(s["model"])
    return {"positions": pos, "durations": durs, "n_songs": len(files)}


def analyze_track(doc: dict) -> dict:
    slug, pool = doc["slug"], doc["pool"]
    inp = V1.track_inputs(pool, slug)
    if inp is None:  # analysis or stem not available on this machine
        return None
    track = inp["track"]
    stem_audio, sr = dj.load_track(inp["stem"])
    act = P.StemActivity.from_audio(stem_audio, sr)
    m1 = doc["model"]
    m2 = P.reanalyze_with_pillar(m1, PILLAR_W)
    exits = []
    for c in djs.cue_candidates(track, "out"):
        e1 = P.exit_point(m1, c["time_s"], act=act)
        e2 = P.exit_point(m2, c["time_s"], act=act)
        exits.append({"bar_index": c["bar_index"], "time_s": c["time_s"],
                      "v1": {"ok": exit_ok(e1), "status": e1["status"], "p_line_end": e1.get("p_line_end"), "last_word": e1.get("last_word"),
                             "end_pos16": e1.get("end_pos16")},
                      "v2": {"ok": exit_ok(e2), "status": e2["status"], "p_line_end": e2.get("p_line_end"), "last_word": e2.get("last_word"),
                             "end_pos16": e2.get("end_pos16"), "end_on_beat4": e2.get("end_on_beat4"),
                             "krims_closure": e2.get("krims_closure"),
                             "pillar_term": pillar_term(e2.get("line_end_terms"))}})
    f = P.flow_features(m2)
    f["style"] = P.krims_style(f)
    sy = P.ohriner_syllables_from_model(m2)
    og = P.ohriner_global_features(sy, tempo_bpm=60.0 / float(m2["period_s"])) if len(sy) >= 8 else {}
    og.pop("segments", None)
    phase = P.vocal_phase_alignment(m2)
    return {"slug": slug, "pool": pool, "v1_track": str(V1_DIR / f"{slug}.json"), "pillar": m2["pillar"],
            "n_lines": {"v1": len(m1["lines"]), "v2": len(m2["lines"])},
            "line_end_pos16": {"v1": end_positions(m1), "v2": end_positions(m2)},
            "line_dur_s": {"v1": line_durations(m1), "v2": line_durations(m2)},
            "exits": exits, "flow": {k: v for k, v in f.items() if not isinstance(v, (list,)) and k not in ("groove_last", "groove_first")},
            "ohriner_global": og, "vocal_phase": phase,
            "asr_reliable": (m1.get("asr_quality") or {}).get("reliable", True),
            "_model_v2": m2, "_track": track}


def summarize(rows: list[dict], mc: dict) -> dict:
    out: dict[str, Any] = {}
    ex = [e for r in rows for e in r["exits"]]
    out["exits"] = {"n": len(ex), "ok_v1": sum(e["v1"]["ok"] for e in ex), "ok_v2": sum(e["v2"]["ok"] for e in ex),
                    "ok_changed": sum(e["v1"]["ok"] != e["v2"]["ok"] for e in ex),
                    "gained": sum((not e["v1"]["ok"]) and e["v2"]["ok"] for e in ex),
                    "lost": sum(e["v1"]["ok"] and not e["v2"]["ok"] for e in ex),
                    "status_changed": sum(e["v1"]["status"] != e["v2"]["status"] for e in ex),
                    "line_end_word_changed": sum(e["v1"]["last_word"] != e["v2"]["last_word"] for e in ex if e["v1"]["ok"] or e["v2"]["ok"]),
                    "v2_ok_on_beat4": sum(bool(e["v2"].get("end_on_beat4")) for e in ex if e["v2"]["ok"]),
                    "v2_ok_krims_closure": dict(Counter(e["v2"].get("krims_closure") for e in ex if e["v2"]["ok"])),
                    "changed_examples": [{"slug": r["slug"], "bar": e["bar_index"], "v1": e["v1"], "v2": e["v2"]}
                                         for r in rows for e in r["exits"] if e["v1"]["ok"] != e["v2"]["ok"]][:20]}
    rel = [r for r in rows if r["asr_reliable"]]
    out["lines"] = {"n_tracks": len(rel), "v1": sum(r["n_lines"]["v1"] for r in rel), "v2": sum(r["n_lines"]["v2"] for r in rel),
                    "tracks_with_pillar_applied": sum(bool(r["pillar"].get("applied")) for r in rel)}
    mc_h = hist16(mc["positions"])
    for v in ("v1", "v2"):
        pos = [q for r in rel for q in r["line_end_pos16"][v]]
        h = hist16(pos)
        d = [x for r in rel for x in r["line_dur_s"][v]]
        out["lines"][v + "_positions"] = {"n": len(pos), "hist": h.round(4).tolist(), "tv_distance_to_mcflow": float(0.5 * np.abs(h - mc_h).sum()),
                                          "share_pos_ge_12": float(np.mean([q >= 12 for q in pos])) if pos else None,
                                          "median_line_s": float(np.median(d)) if d else None,
                                          "mean_line_s": float(np.mean(d)) if d else None}
    out["lines"]["mcflow_annotated"] = {"n": len(mc["positions"]), "hist": mc_h.round(4).tolist(),
                                        "share_pos_ge_12": float(np.mean([q >= 12 for q in mc["positions"]])),
                                        "median_line_s": float(np.median(mc["durations"])), "mean_line_s": float(np.mean(mc["durations"]))}
    out["lines"]["reference"] = ("Ohriner 2019 Tab. 5.2: lunghezza di frase media 2,37 s (DS 0,43) su 75 strofe; p. 110: due terzi "
                                 "delle strofe con il pilastro su o dopo la posizione 12")
    # flow / Ohriner globals / phase
    def stat(vals):
        v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
        return {"n": int(v.size), "median": float(np.median(v)) if v.size else None, "mean": float(v.mean()) if v.size else None,
                "q25": float(np.quantile(v, 0.25)) if v.size else None, "q75": float(np.quantile(v, 0.75)) if v.size else None}
    out["flow"] = {k: stat([r["flow"].get(k) for r in rel]) for k in ("syllable_speed", "saturation", "npvi", "first_bar_caesura",
                                                                      "end_beat4", "end_R", "enjambment", "duple_share", "grooviness")}
    out["ohriner_global"] = {k: stat([r["ohriner_global"].get(k) for r in rel]) for k in ("saturation", "sat_delta", "syls_per_second",
                                                                                         "grooviness", "dupleness", "adherence")}
    out["ohriner_global"]["reference"] = "Ohriner 2019 Tab. 5.2 (75 strofe): saturazione 11,1, sat. change 2,07, sill/s 5,09, duple .45, grooviness .51, adherence .67"
    out["krims_dominant"] = dict(Counter(r["flow"]["style"]["dominant"] for r in rel))
    out["vocal_phase"] = {k: stat([r["vocal_phase"].get(k) for r in rel]) for k in ("phase_beats", "phase_ms", "misalignment", "misalignment_phase",
                                                                                 "bur", "explained_share")}
    out["vocal_phase"]["reference"] = ("Ohriner 2019 pp. 190-196: voce in ritardo nel genere, aggiustamento ottimo -6,84/60 di battito (circa "
                                       "+0,11 battiti di ritardo); swing raro (2/3 delle battute senza miglioramento)")
    return out


def pairs_flow(rows: list[dict], max_pairs: int | None = None) -> dict:
    by_pool = defaultdict(list)
    for r in rows:
        if r["asr_reliable"]:
            by_pool[r["pool"]].append(r)
    cache: dict = {}
    res = Counter()
    cont, flips, examples = [], Counter(), []
    t0 = time.time()
    n = 0
    for pool, rs in by_pool.items():
        for a in rs:
            for b in rs:
                if a is b:
                    continue
                if max_pairs and n >= max_pairs:
                    break
                n += 1
                q0 = P.plan_pair_phrase(a["_track"], b["_track"], a["_model_v2"], b["_model_v2"], top_k=1, with_flow=True, flow_cache=cache)
                q1 = P.plan_pair_phrase(a["_track"], b["_track"], a["_model_v2"], b["_model_v2"], top_k=1, flow_weight=1.0, flow_cache=cache)
                res["pairs"] += 1
                if not q0["plans"]:
                    res["no_plan"] += 1
                    continue
                p0, p1 = q0["plans"][0], q1["plans"][0]
                c = (p0.get("flow") or {}).get("continuity")
                if c is not None:
                    cont.append(c)
                for fl in (p0.get("flow") or {}).get("flips", []):
                    flips[fl.split(" (")[0]] += 1
                changed = (p0["a_out_s"], p0["b_in_s"]) != (p1["a_out_s"], p1["b_in_s"])
                res["top_changed_with_flow"] += int(changed)
                if changed and len(examples) < 10:
                    examples.append({"a": a["slug"], "b": b["slug"], "v_flow0": [p0["a_out_s"], p0["b_in_s"], (p0.get("flow") or {}).get("continuity")],
                                     "v_flow1": [p1["a_out_s"], p1["b_in_s"], (p1.get("flow") or {}).get("continuity")]})
    return {"counts": dict(res), "continuity_top_plan": {"median": float(np.median(cont)) if cont else None, "n": len(cont)},
            "flips_top_plan": dict(flips), "examples": examples, "elapsed_s": time.time() - t0,
            "note": "flow_weight 1 = +1 x continuity (0-1) nel punteggio del piano; diagnostica, default 0"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-pairs", type=int, default=None)
    args = ap.parse_args()
    (OUT / "tracks").mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for f in sorted(V1_DIR.glob("*.json")):
        doc = json.loads(f.read_text())
        t0 = time.time()
        r = analyze_track(doc)
        if r is None:
            skipped.append(doc["slug"])
            print(f"{doc['slug']}: input mancanti, saltato", flush=True)
            continue
        rows.append(r)
        keep = {k: v for k, v in r.items() if not k.startswith("_")}
        keep["model_v2_lines"] = r["_model_v2"]["lines"]
        (OUT / "tracks" / f"{r['slug']}.json").write_text(json.dumps(V1.jsonable(keep), ensure_ascii=False))
        print(f"{r['slug']}: versi {r['n_lines']['v1']} -> {r['n_lines']['v2']}, uscite ok "
              f"{sum(e['v1']['ok'] for e in r['exits'])} -> {sum(e['v2']['ok'] for e in r['exits'])} [{time.time() - t0:.0f}s]", flush=True)
    mc = mcflow_annotated_ends()
    summ = summarize(rows, mc)
    summ["pairs_flow"] = pairs_flow(rows, args.max_pairs)
    summ["skipped_tracks"] = skipped
    summ["inputs"] = {"v1_tracks": str(V1_DIR.relative_to(ROOT)), "pillar_weight": PILLAR_W,
                      "flow_config": str(P.FLOW_CONFIG.relative_to(ROOT)), "mcflow_config": str(P.MCFLOW_CONFIG.relative_to(ROOT))}
    (OUT / "compare.json").write_text(json.dumps(V1.jsonable(summ), indent=1, ensure_ascii=False))
    print(json.dumps(V1.jsonable({k: summ[k] for k in ("exits", "lines", "flow", "ohriner_global", "krims_dominant", "vocal_phase", "pairs_flow")}),
                     indent=1, ensure_ascii=False)[:9000])


if __name__ == "__main__":
    main()
