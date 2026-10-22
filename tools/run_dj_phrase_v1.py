#!/usr/bin/env python3
"""dj_phrase v1: modello di versi/barre per brano, confronto guardia vecchia/nuova, clip d'esempio.

Fasi (riprendibili; scrive solo in ``runs/dj_phrase_v1``):

* ``analyze``: per ogni brano con analisi DJ completa (``runs/dj_plan_v1/tracks`` = pool V12,
  ``runs/dj_plan_cc_v1/tracks`` = pool CC, solo file già scritti da dj_plan_pool) costruisce il
  modello di ``musicmix.dj_phrase`` (parole ASR raffinate sullo stem, rime, prosodia, griglia,
  pickup) e valuta ogni cue-out/cue-in candidato di dj_select con le tre guardie:

  - ``old``: ``dj_select.cut_vocal_clear`` (voce di A <= 10 % dell'ultimo battito);
  - ``v2``: guardia della galleria v2 (stem silenzioso nell'ultimo battito e 150 ms dopo, nessuna
    parola ASR entro +-150 ms);
  - ``new``: ``dj_phrase.exit_point`` (fine verso p >= 0,5, finestra di switch senza parole, stem
    silenzioso in almeno un punto della finestra; coda protetta fino a mezzo battito).

  Parole troncate: misurate sia con le parole raffinate (quelle che la guardia nuova usa) sia con le
  parole ASR grezze (misura indipendente dalla raffinatura).
* ``compare``: tabella per brano e totali, più la fattibilità sulle coppie ordinate (stesso pool).
* ``render``: 6-10 coppie; per ciascuna il miglior piano dj_phrase reso come ``cut`` e ``filter_out``
  con switch e pickup (``dj_phrase.render_phrase_transition``), lo stesso cue reso con
  ``dj_render`` senza pickup (B sul downbeat) e il piano di default di dj_select; QA ``dj_qa``.

    python tools/run_dj_phrase_v1.py analyze compare render
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from musicmix import dj_phrase as P  # noqa: E402
from musicmix import dj_qa  # noqa: E402
from musicmix import dj_render as dj  # noqa: E402
from musicmix import dj_select as djs  # noqa: E402
from musicmix.paths import localize  # noqa: E402

OUT = ROOT / "runs/dj_phrase_v1"
SR = 48000
POOLS = {"v12": ROOT / "runs/dj_plan_v1", "cc": ROOT / "runs/dj_plan_cc_v1"}
# CC tracks whose analysis finished before runs/dj_plan_cc_v1/tracks is written (tools/dj_phrase_plan_cc.py)
EXTRA_TRACKS = {"cc": ROOT / "runs/dj_phrase_v1/plan_cc/tracks"}
ASR_DIR = OUT / "asr"
from musicmix import dj_effects as fx  # noqa: E402

SECTION_GAIN = dj.SectionGainConfig(enabled=True)  # as runs/dj_gallery_v2, for every version
CUT_CFG = dj.DJRenderConfig(section_gain=SECTION_GAIN)
FX_CFG = fx.DJEffectsConfig(section_gain=SECTION_GAIN)
V2_POST_S = dj_qa.WORD_GUARD_S
V2_ASR_S = dj_qa.WORD_GUARD_S


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


# ------------------------------------------------------------------ inputs
def track_inputs(pool: str, slug: str) -> dict[str, Any] | None:
    tj = POOLS[pool] / "tracks" / f"{slug}.json"
    if not tj.is_file() and pool in EXTRA_TRACKS:
        tj = EXTRA_TRACKS[pool] / f"{slug}.json"
    track = json.loads(tj.read_text())
    vjson = localize(track["inputs"]["vocals"])
    # NB (21/10/2026): tool congelato; nome dello stem indovinato. Nel codice attivo usare musicmix.stems.vocal_stem_path.
    stem = Path(vjson).parent / ("a_vocals.wav" if Path(vjson).name == "a.json" else Path(vjson).stem + "_vocals.wav")
    if not stem.is_file():
        return None
    if pool == "cc":
        asr = ASR_DIR / f"{slug}.json"
        if not asr.is_file():
            return {"skip": "multilingual ASR missing", "track": track}
        a = json.loads(asr.read_text())
        words = a["words"]
        lang = a["backend"].get("detected_language") or "und"
        asr_src = {"file": rel(asr), "asr_sha256": sha256(asr), "model": a["backend"].get("model_repo"), "revision": a["backend"].get("model_revision"),
                   "language": lang, "language_probability": a["backend"].get("language_probability"),
                   "stem_sha256": a.get("stem_sha256")}
    else:
        v = json.loads(Path(vjson).read_text())
        words = v["words"]
        lang = "en"
        asr_src = {"file": rel(Path(vjson)), "asr_sha256": sha256(Path(vjson)), "model": "Systran/faster-whisper-small.en (pool V12 analysis)", "language": "en"}
    return {"track": track, "track_json": tj, "stem": stem, "words": words, "lang": lang, "asr": asr_src}


def complete_slugs(pool: str) -> list[str]:
    out = set()
    for d in (POOLS[pool] / "tracks", EXTRA_TRACKS.get(pool)):
        if d is not None and d.is_dir():
            out |= {p.stem for p in d.glob("*.json")}
    return sorted(out)


# ---------------------------------------------------------------- guards
def v2_guard(track: dict, act_v2: np.ndarray, times: np.ndarray, words_raw: list, cut_s: float) -> dict:
    beat = float(track["tempo"]["period_s"])
    stem_clear = djs.vocal_clear_window(times, act_v2, cut_s, beat, V2_POST_S)
    near = [w for w in words_raw if w["end_sample"] / SR > cut_s - V2_ASR_S and w["start_sample"] / SR < cut_s + V2_ASR_S]
    return {"clear": bool(stem_clear and not near), "stem_clear": bool(stem_clear), "asr_words": [w.get("text") for w in near]}


def raw_words(words: list) -> list[dict]:
    return [{"on": int(w["start_sample"]), "off": int(w["end_sample"]), "text": w.get("text", "")} for w in words]


def analyze(slugs_by_pool: dict[str, list[str]], force: bool = False) -> None:
    (OUT / "tracks").mkdir(parents=True, exist_ok=True)
    lex = P.Lexicon()
    for pool, slugs in slugs_by_pool.items():
        for slug in slugs:
            dst = OUT / "tracks" / f"{slug}.json"
            inp = track_inputs(pool, slug)
            if dst.is_file() and not force:
                old = json.loads(dst.read_text())["inputs"]["asr"].get("asr_sha256")
                cur = (inp or {}).get("asr", {}).get("asr_sha256")
                if old == cur:
                    continue
            if inp is None or "skip" in inp:
                print(f"{slug}: salto ({(inp or {}).get('skip', 'stem mancante')})", flush=True)
                continue
            t0 = time.time()
            track = inp["track"]
            stem_audio, sr = dj.load_track(inp["stem"])
            assert sr == SR
            act = P.StemActivity.from_audio(stem_audio, sr)
            model = P.analyze_track(track, inp["words"], act=act, lang=inp["lang"], lexicon=lex,
                                    asr_language_probability=inp["asr"].get("language_probability"))
            act_v2, _ = dj_qa.vocal_activity(stem_audio, SR, None, dj_qa.stem_threshold(stem_audio, SR))
            times = (np.arange(len(act_v2)) + 0.5) * dj_qa.VOCAL_HOP_S
            m = int(P.PARAMS["word_margin_s"].value * SR)
            rw = raw_words(inp["words"])
            lp = np.full_like(act.rms_db, np.percentile(act.rms_db, 90))  # global speech level of the stem

            def level(sample: int) -> float:
                """Max stem level in the 60 ms before a cut, dB re the stem's 90th percentile (ASR-independent)."""
                sel = act.frames(sample - int(0.06 * SR), sample)  # the last 60 ms that are still heard
                return float(np.max(act.rms_db[sel] - lp[sel])) if sel.any() else float("nan")

            exits = []
            for c in djs.cue_candidates(track, "out"):
                e = P.exit_point(model, c["time_s"], act=act)
                D = e["downbeat_sample"]
                new_ok = bool(e["status"] in ("clean", "tail", "a_pickup", "instrumental")
                              and (e["status"] == "instrumental" or (e.get("p_line_end") or 0) >= P.PARAMS["line_end_threshold"].value)
                              and e.get("acoustic") != "stem_active_in_window")
                # the new exit's own switch point: latest silent point of the window (or D with the tail)
                if new_ok:
                    x = e["window"][1]
                    if e.get("acoustic") == "silent_point":
                        x = min(x, e["acoustic_latest_silent"])
                    cut = D + x
                    keep = e.get("tail_end_sample")
                else:
                    cut, keep = D, None
                exits.append({
                    "bar_index": c["bar_index"], "time_s": c["time_s"], "phrase_level": c["phrase_level"],
                    "section_start": c["section_start"],
                    "old": {"clear": bool(djs.cut_vocal_clear(track, c["bar_index"])),
                            "vocal_last_beat": float(track["bars"]["vocal_last_beat"][c["bar_index"] - 1]) if c["bar_index"] > 0 else 0.0},
                    "v2": v2_guard(track, act_v2, times, inp["words"], c["time_s"]),
                    "new": {"ok": new_ok, "status": e["status"], "p_line_end": e.get("p_line_end"), "acoustic": e.get("acoustic"),
                            "switch_rel_s": (cut - D) / SR, "tail_s": ((keep - D) / SR) if keep else None,
                            "reasons": e["reasons"], "last_word": e.get("last_word"), "next_word": e.get("next_word")},
                    "detail": e,
                    "level_at_downbeat_db": level(D), "level_at_new_cut_db": level(cut) if new_ok and keep is None else None,
                    "dry_cut_truncated_refined": [w["text"] for w in P.truncated_words(model["words"], D, role="a", margin=m)],
                    "dry_cut_truncated_raw": [w["text"] for w in P.truncated_words(rw, D, role="a", margin=m)],
                    "new_cut_truncated_refined": [w["text"] for w in P.truncated_words(model["words"], cut, role="a", margin=m, keep_until=keep)] if new_ok else None,
                    "new_cut_truncated_raw": [w["text"] for w in P.truncated_words(rw, cut, role="a", margin=m, keep_until=keep)] if new_ok else None,
                })
            entries = []
            for c in djs.cue_candidates(track, "in"):
                e = P.entry_point(model, c["time_s"], act=act)
                D = e["downbeat_sample"]
                new_ok = bool(e["status"] in ("instrumental", "on_or_after_downbeat", "pickup")
                              and e.get("acoustic") != "stem_active_in_window"
                              and (e.get("line_start_p") is None or e["line_start_p"] >= P.PARAMS["line_end_threshold"].value))
                start = D + (e["window"][1] if new_ok else 0)
                if new_ok and e.get("acoustic") == "silent_point":
                    start = min(start, D + e["acoustic_latest_silent"])
                entries.append({
                    "bar_index": c["bar_index"], "time_s": c["time_s"], "phrase_level": c["phrase_level"],
                    "section_start": c["section_start"],
                    "new": {"ok": new_ok, "status": e["status"], "pickup_s": e["pickup_samples"] / SR, "pickup_words": e["pickup_words"],
                            "start_rel_s": (start - D) / SR, "acoustic": e.get("acoustic"), "line_start_p": e.get("line_start_p"),
                            "reasons": e["reasons"]},
                    # B entering exactly on the downbeat (current renderer): words cut at the entry
                    "detail": e,
                    "level_at_downbeat_db": level(D), "level_at_new_start_db": level(start) if new_ok else None,
                    "at_downbeat_truncated_refined": [w["text"] for w in P.truncated_words(model["words"], D, role="b", margin=m)],
                    "at_downbeat_truncated_raw": [w["text"] for w in P.truncated_words(rw, D, role="b", margin=m)],
                    "at_downbeat_pickup_lost": bool(e["status"] == "pickup"),
                    "new_truncated_refined": [w["text"] for w in P.truncated_words(model["words"], start, role="b", margin=m)] if new_ok else None,
                    "new_truncated_raw": [w["text"] for w in P.truncated_words(rw, start, role="b", margin=m)] if new_ok else None,
                })
            doc = {"schema_version": "musicmix.dj_phrase_run.1", "pool": pool, "slug": slug,
                   "inputs": {"track_analysis": rel(inp["track_json"]), "track_analysis_sha256": sha256(inp["track_json"]),
                              "vocal_stem": rel(inp["stem"]), "vocal_stem_sha256": sha256(inp["stem"]), "asr": inp["asr"],
                              "cmudict": {"path": str(P.CMUDICT_PATH), "sha256": sha256(P.CMUDICT_PATH) if P.CMUDICT_PATH.is_file() else None,
                                          "source": "https://github.com/cmusphinx/cmudict @ 74790861f652b15e4ac49015a90074ad62a27690 (BSD-2)"},
                              "mcflow_config": rel(P.MCFLOW_CONFIG), "mcflow_config_sha256": sha256(P.MCFLOW_CONFIG)},
                   "model": model, "exits": exits, "entries": entries, "elapsed_s": time.time() - t0}
            dst.write_text(json.dumps(jsonable(doc), ensure_ascii=False))
            s = model["summary"]
            print(f"{pool}/{slug}: {s['n_words']} parole ({s['n_dropped']} scartate), {s['n_lines']} versi, "
                  f"uscite new {sum(e['new']['ok'] for e in exits)}/{len(exits)} old {sum(e['old']['clear'] for e in exits)} "
                  f"v2 {sum(e['v2']['clear'] for e in exits)}; ingressi new {sum(e['new']['ok'] for e in entries)}/{len(entries)} "
                  f"[{time.time() - t0:.0f}s]", flush=True)


# ----------------------------------------------------------------- compare
def compare() -> dict:
    docs = {p.stem: json.loads(p.read_text()) for p in sorted((OUT / "tracks").glob("*.json"))}
    rows = []
    tot = Counter()
    for slug, d in docs.items():
        ex, en = d["exits"], d["entries"]
        r = {"pool": d["pool"], "slug": slug, "lang": d["model"]["lang"], "n_words": d["model"]["summary"]["n_words"],
             "n_dropped": d["model"]["summary"]["n_dropped"], "n_lines": d["model"]["summary"]["n_lines"],
             "exit_candidates": len(ex), "entry_candidates": len(en)}
        for g in ("old", "v2"):
            ok = [e for e in ex if e[g]["clear"]]
            r[f"exits_{g}"] = len(ok)
            r[f"exits_{g}_truncating_refined"] = sum(1 for e in ok if e["dry_cut_truncated_refined"])
            r[f"exits_{g}_truncated_words_refined"] = sum(len(e["dry_cut_truncated_refined"]) for e in ok)
            r[f"exits_{g}_truncated_words_raw"] = sum(len(e["dry_cut_truncated_raw"]) for e in ok)
        ok = [e for e in ex if e["new"]["ok"]]
        r["exits_new"] = len(ok)
        r["exits_new_status"] = dict(Counter(e["new"]["status"] for e in ok))
        r["exits_new_truncated_words_refined"] = sum(len(e["new_cut_truncated_refined"]) for e in ok)
        r["exits_new_truncated_words_raw"] = sum(len(e["new_cut_truncated_raw"]) for e in ok)
        r["exits_new_switch_before_downbeat"] = sum(1 for e in ok if e["new"]["switch_rel_s"] < 0)
        r["exits_new_tail"] = sum(1 for e in ok if e["new"]["tail_s"])
        lv = lambda xs: float(np.nanmedian(xs)) if len(xs) else None  # noqa: E731
        r["level_db_median_old_ok"] = lv([e["level_at_downbeat_db"] for e in ex if e["old"]["clear"]])
        r["level_db_median_all_downbeats"] = lv([e["level_at_downbeat_db"] for e in ex])
        r["level_db_median_new_cut"] = lv([e["level_at_new_cut_db"] for e in ok if e["level_at_new_cut_db"] is not None])
        r["exits_all_dry_cut_truncating_refined"] = sum(1 for e in ex if e["dry_cut_truncated_refined"])
        r["exits_all_dry_cut_truncating_raw"] = sum(1 for e in ex if e["dry_cut_truncated_raw"])
        r["exits_status_all"] = dict(Counter(e["new"]["status"] for e in ex))
        r["entries_with_pickup"] = sum(1 for e in en if e["new"]["status"] == "pickup")
        r["entries_at_downbeat_truncating_refined"] = sum(1 for e in en if e["at_downbeat_truncated_refined"])
        r["entries_at_downbeat_truncating_raw"] = sum(1 for e in en if e["at_downbeat_truncated_raw"])
        r["entries_at_downbeat_pickup_lost"] = sum(1 for e in en if e["at_downbeat_pickup_lost"])
        okb = [e for e in en if e["new"]["ok"]]
        r["entries_new"] = len(okb)
        r["entries_new_pickup_preserved"] = sum(1 for e in okb if e["new"]["status"] == "pickup")
        r["entries_new_truncated_words_refined"] = sum(len(e["new_truncated_refined"]) for e in okb)
        r["entries_new_truncated_words_raw"] = sum(len(e["new_truncated_raw"]) for e in okb)
        r["entries_status_all"] = dict(Counter(e["new"]["status"] for e in en))
        rows.append(r)
        r["asr_reliable"] = bool(d["model"]["summary"].get("asr_reliable", True))
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                tot[(d["pool"], k)] += v
                tot[("all", k)] += v
                if r["asr_reliable"]:
                    tot[(d["pool"] + "_reliable", k)] += v
                    tot[("all_reliable", k)] += v
    totals = {}
    for (pool, k), v in tot.items():
        totals.setdefault(pool, {})[k] = v
    for pool in totals:
        base = pool.replace("_reliable", "")
        totals[pool]["n_tracks"] = sum(1 for r in rows if (base == "all" or r["pool"] == base)
                                       and (not pool.endswith("_reliable") or r["asr_reliable"]))
    # pair feasibility (ordered pairs within each pool): any (exit, entry) with phrase_switch ok
    pairs = []
    models = {}
    for slug, d in docs.items():
        models[slug] = d
    for pool in ("v12", "cc"):
        sl = [s for s, d in docs.items() if d["pool"] == pool]
        for a in sl:
            for b in sl:
                if a == b:
                    continue
                da, db = docs[a], docs[b]
                n_comb = n_ok = n_ok_pick = 0
                n_old = 0
                for ea in da["exits"]:
                    for eb in db["entries"]:
                        n_comb += 1
                        n_old += int(ea["old"]["clear"])
                        if ea["new"]["ok"] and eb["new"]["ok"]:
                            sw = P.phrase_switch(ea["detail"], eb["detail"], da["model"]["period_s"])
                            if sw["ok"]:
                                n_ok += 1
                                n_ok_pick += int(eb["new"]["status"] == "pickup")
                # the plan each policy would actually pick for this pair
                TA = json.loads((ROOT / da["inputs"]["track_analysis"]).read_text())
                TB = json.loads((ROOT / db["inputs"]["track_analysis"]).read_text())
                m = int(P.PARAMS["word_margin_s"].value * SR)
                top = {}
                d0 = djs.plan_pair(TA, TB, top_k=1)["plans"]
                if d0:
                    q = d0[0]
                    Da, Db = int(round(q["a_out_s"] * SR)), int(round(q["b_in_s"] * SR))
                    eb = P.entry_point(db["model"], q["b_in_s"])
                    top["default"] = {"a_out_bar": q["a_out_bar"], "b_in_bar": q["b_in_bar"], "type": q["type"],
                                      "a_truncated": len(P.truncated_words(da["model"]["words"], Da, role="a", margin=m)),
                                      "b_truncated": len(P.truncated_words(db["model"]["words"], Db, role="b", margin=m)),
                                      "b_pickup_lost": eb["status"] == "pickup"}
                    g = djs.plan_pair(TA, TB, top_k=1, cut_vocal_guard=True)["plans"][0]
                    top["default_guarded_type"] = g["type"]
                ph = P.plan_pair_phrase(TA, TB, da["model"], db["model"], top_k=1)["plans"]
                if ph:
                    q = ph[0]
                    sw = q["phrase"]["switch"]
                    Da, Db = int(round(q["a_out_s"] * SR)), int(round(q["b_in_s"] * SR))
                    x = sw["switch_rel_samples"]
                    keep = q["phrase"]["exit"].get("tail_end_sample")
                    top["phrase"] = {"a_out_bar": q["a_out_bar"], "b_in_bar": q["b_in_bar"], "switch_rel_s": sw["switch_rel_s"],
                                     "a_truncated": len(P.truncated_words(da["model"]["words"], Da + x, role="a", margin=m, keep_until=keep)),
                                     "b_truncated": len(P.truncated_words(db["model"]["words"], Db + x, role="b", margin=m)),
                                     "b_pickup_preserved": q["phrase"]["entry"]["status"] == "pickup",
                                     "p_line_end": q["phrase"]["exit"].get("p_line_end")}
                pairs.append({"pool": pool, "a": a, "b": b, "combinations": n_comb, "old_clear_cut": n_old,
                              "phrase_ok": n_ok, "phrase_ok_with_pickup": n_ok_pick, "top": top})
    # distribution check against MCFlow: where audio lines start/end relative to the bar
    line_stats = {}
    for pool in ("v12", "cc", "all"):
        starts, ends, wpl = Counter(), Counter(), []
        for slug, d in docs.items():
            if pool != "all" and d["pool"] != pool:
                continue
            md = d["model"]
            if not md["lines"]:
                continue
            down = np.asarray(md["downbeats_s"])
            per = md["period_s"]
            for L in md["lines"]:
                wpl.append(L["last_word"] - L["first_word"] + 1)
                t0 = L["start_sample"] / SR
                j = int(np.searchsorted(down, t0, side="right") - 1)
                if j < 0 or j + 1 >= len(down):
                    continue
                bar_beats = (down[j + 1] - down[j]) / per
                pos = round((t0 - down[j]) / per * 4) / 4  # quantised to sixteenths, as MCFlow
                if pos >= bar_beats - 1e-6:
                    pos = 0.0
                to_next = bar_beats - pos
                if pos < 1e-6:
                    starts["on_downbeat"] += 1
                elif to_next <= 1.0:
                    starts["pickup_le_1_beat"] += 1
                elif to_next <= 2.0:
                    starts["pickup_1_2_beats"] += 1
                elif pos <= 1.0:
                    starts["after_downbeat_le_1_beat"] += 1
                else:
                    starts["mid_bar"] += 1
                w = md["words"][L["last_word"]]
                if w.get("beat"):
                    ends[str(w["beat"])] += 1
        tot_s, tot_e = sum(starts.values()), sum(ends.values())
        line_stats[pool] = {"n_lines": len(wpl), "words_per_line_mean": float(np.mean(wpl)) if wpl else None,
                            "line_start_vs_downbeat": {k: v / tot_s for k, v in starts.items()} if tot_s else {},
                            "line_end_beat": {k: v / tot_e for k, v in sorted(ends.items())} if tot_e else {}}
    mc = P.load_mcflow()
    tot = sum(mc["phrase_start_vs_downbeat"].values())
    line_stats["mcflow_reference"] = {"words_per_phrase_mean": mc["n_word_boundaries"] / max(1, tot),
                                      "line_start_vs_downbeat": {k: v / tot for k, v in mc["phrase_start_vs_downbeat"].items()},
                                      "note": "same categories; audio line onsets quantised to the nearest sixteenth of the dj_select grid"}
    res = {"schema_version": "musicmix.dj_phrase_compare.1", "rows": rows, "totals": totals, "pairs": pairs,
           "line_stats": line_stats,
           "top_plan_totals": {pool: {
               "pairs": sum(1 for p in pairs if p["pool"] == pool),
               "default_top_truncating_a": sum(1 for p in pairs if p["pool"] == pool and p["top"].get("default", {}).get("type") == "cut" and p["top"].get("default", {}).get("a_truncated")),
               "default_top_truncating_b": sum(1 for p in pairs if p["pool"] == pool and p["top"].get("default", {}).get("type") == "cut" and p["top"].get("default", {}).get("b_truncated")),
               "default_top_b_pickup_lost": sum(1 for p in pairs if p["pool"] == pool and p["top"].get("default", {}).get("b_pickup_lost")),
               "default_top_is_cut": sum(1 for p in pairs if p["pool"] == pool and p["top"].get("default", {}).get("type") == "cut"),
               "default_guarded_echo_out": sum(1 for p in pairs if p["pool"] == pool and p["top"].get("default_guarded_type") == "echo_out"),
               "phrase_top_available": sum(1 for p in pairs if p["pool"] == pool and "phrase" in p["top"]),
               "phrase_top_truncating_a": sum(1 for p in pairs if p["pool"] == pool and p["top"].get("phrase", {}).get("a_truncated")),
               "phrase_top_truncating_b": sum(1 for p in pairs if p["pool"] == pool and p["top"].get("phrase", {}).get("b_truncated")),
               "phrase_top_b_pickup_preserved": sum(1 for p in pairs if p["pool"] == pool and p["top"].get("phrase", {}).get("b_pickup_preserved")),
           } for pool in ("v12", "cc")},
           "pair_totals": {pool: {"pairs": sum(1 for p in pairs if p["pool"] == pool),
                                  "pairs_with_old_clear_cut": sum(1 for p in pairs if p["pool"] == pool and p["old_clear_cut"]),
                                  "pairs_with_phrase_ok": sum(1 for p in pairs if p["pool"] == pool and p["phrase_ok"]),
                                  "pairs_with_phrase_ok_pickup": sum(1 for p in pairs if p["pool"] == pool and p["phrase_ok_with_pickup"])}
                           for pool in ("v12", "cc")},
           "definitions": {
               "old": "dj_select.cut_vocal_clear: vocal stem coverage of the last beat before the cue <= 0.10",
               "v2": "gallery v2 guard: stem (dj_qa activity) silent in the last beat and 150 ms after, no ASR word within +-150 ms",
               "new": "dj_phrase.exit_point: line end p >= 0.5, word-free switch window, one stem-silent point; tail <= 1/2 beat protected",
               "truncated": "word audible >= 30 ms before the cut and losing >= 30 ms after it (refined = stem-refined words used by the new guard; raw = ASR words as stored)",
               "entries_at_downbeat": "B made audible exactly at its downbeat, as dj_render/dj_effects do today"}}
    (OUT / "compare.json").write_text(json.dumps(jsonable(res), indent=1, ensure_ascii=False))
    return res


# ------------------------------------------------------------------ render
class Loaded:
    def __init__(self, pool: str, slug: str):
        inp = track_inputs(pool, slug)
        self.pool, self.slug = pool, slug
        self.track = inp["track"]
        wav = localize(self.track["inputs"]["canonical_wav"])
        if sha256(wav) != self.track["track_id"]:
            raise RuntimeError(f"{slug}: canonical audio identity mismatch")
        self.wav = wav
        self.audio, _ = dj.load_track(wav)
        self.vocals, _ = dj.load_track(inp["stem"])
        self.stem = inp["stem"]
        self.doc = json.loads((OUT / "tracks" / f"{slug}.json").read_text())
        self.model = self.doc["model"]
        self.act = P.StemActivity.from_audio(self.vocals, SR)
        p, ph, off = self.track["tempo"]["period_s"], self.track["tempo"]["phase_s"], self.track["tempo"]["bar_offset_beats"]
        beats, downs = djs.grid_times(self.track["tempo"], self.track["duration_s"])
        self.beats, self.downbeats = beats, np.asarray(self.track["grid"]["downbeats_s"])
        self.words_raw = inp["words"]
        self.vocal_threshold = dj_qa.stem_threshold(self.vocals, SR)


def out_truncations(meta: dict, A: Loaded, B: Loaded, n: int) -> dict:
    """Words of A and B cut by the rendered clip (refined words mapped to the clip clock)."""
    sw = int(meta["switch"].get("fader_switch_output_sample", meta["switch"]["output_sample"]))
    a_src = int(meta["switch"].get("fader_switch_a_source_sample", meta["switch"]["a_source_sample"]))
    b_src = int(meta["switch"].get("fader_switch_b_source_sample", meta["switch"]["b_source_sample"]))
    tail = (meta.get("phrase") or {}).get("tail")
    m = int(P.PARAMS["word_margin_s"].value * SR)
    keep = None
    if tail:
        keep = a_src + (int(tail["output_end_sample"]) - sw)
    ta = P.truncated_words(A.model["words"], a_src, role="a", margin=m, keep_until=keep)
    tb = P.truncated_words(B.model["words"], b_src, role="b", margin=m)
    ra = P.truncated_words(raw_words(A.words_raw), a_src, role="a", margin=m, keep_until=keep)
    rb = P.truncated_words(raw_words(B.words_raw), b_src, role="b", margin=m)
    return {"a_truncated_refined": [w["text"] for w in ta], "b_truncated_refined": [w["text"] for w in tb],
            "a_truncated_raw": [w["text"] for w in ra], "b_truncated_raw": [w["text"] for w in rb],
            "a_exit_source_sample": a_src, "b_entry_source_sample": b_src}


def choose_render_pairs(max_pairs: int) -> list[dict]:
    docs = {p.stem: json.loads(p.read_text()) for p in sorted((OUT / "tracks").glob("*.json"))}
    cand = []
    loaded_tracks = {}
    for slug, d in docs.items():
        loaded_tracks[slug] = json.loads((ROOT / d["inputs"]["track_analysis"]).read_text())
    for pool in ("v12", "cc"):
        sl = [s for s, d in docs.items() if d["pool"] == pool]
        for a in sl:
            for b in sl:
                if a == b:
                    continue
                A, B = loaded_tracks[a], loaded_tracks[b]
                pr = P.plan_pair_phrase(A, B, docs[a]["model"], docs[b]["model"], top_k=1)
                if not pr["plans"]:
                    continue
                q = pr["plans"][0]
                comp = pr["compatibility"]
                cand.append({"pool": pool, "a": a, "b": b, "plan": q, "compat": comp["status"],
                             "tempo_jump_pct": comp["tempo"]["stretch_b_pct"], "has_pickup": q["phrase"]["entry"]["status"] == "pickup",
                             "tail": bool(q["phrase"]["exit"].get("tail_end_sample"))})
    # prefer compatible tempo, then variety: pickup / tail / early switch, distinct tracks
    cand.sort(key=lambda c: (c["compat"] == "incompatible", -int(c["has_pickup"]), -c["plan"]["score_phrase"]))
    chosen, used = [], Counter()
    quota = {"cc": min(3, max_pairs // 3), "v12": max_pairs}  # keep some CC (non-English) examples
    for pool in ("cc", "v12"):
        for c in cand:
            if c["pool"] != pool or used[c["a"]] >= 2 or used[c["b"]] >= 2 or sum(x["pool"] == pool for x in chosen) >= quota[pool]:
                continue
            chosen.append(c)
            used[c["a"]] += 1
            used[c["b"]] += 1
            if len(chosen) >= max_pairs:
                break
    return chosen


def render(max_pairs: int = 8, qa: bool = True) -> None:
    clips = OUT / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    chosen = choose_render_pairs(max_pairs)
    cache: dict[str, Loaded] = {}
    manifest = []
    tracker = "auto" if qa else None
    for k, c in enumerate(chosen, 1):
        base_name = f"{k:02d}_{c['a'][:24]}__{c['b'][:24]}"
        gpath = clips / f"{base_name}.group.json"
        if gpath.is_file():  # resumable: this group is already rendered
            manifest.append(json.loads(gpath.read_text()))
            continue
        for s in list(cache):  # keep memory low (no swap on Linux)
            if s not in (c["a"], c["b"]):
                del cache[s]
        for s in (c["a"], c["b"]):
            if s not in cache:
                cache[s] = Loaded(c["pool"], s)
        A, B = cache[c["a"]], cache[c["b"]]
        q = c["plan"]
        sw = q["phrase"]["switch"]
        ex = q["phrase"]["exit"]
        tail_end_s = (ex["tail_end_sample"] / SR) if ex.get("tail_end_sample") else None
        grids = dict(beats_a=A.beats, downbeats_a=A.downbeats, beats_b=B.beats, downbeats_b=B.downbeats)
        prov = {"a": {"slug": A.slug, "audio": rel(A.wav), "audio_sha256": A.track["track_id"], "vocal_stem": rel(A.stem)},
                "b": {"slug": B.slug, "audio": rel(B.wav), "audio_sha256": B.track["track_id"], "vocal_stem": rel(B.stem)},
                "plan": {k2: q[k2] for k2 in ("a_out_s", "a_out_bar", "b_in_s", "b_in_bar", "type_dj_select", "score", "score_phrase")},
                "phrase": q["phrase"], "script": "tools/run_dj_phrase_v1.py"}
        versions = []
        # 1-2: phrase-aware cut and filter_out at the verse end with B's pickup
        for kind in ("cut", "filter_out"):
            try:
                res = P.render_phrase_transition(kind, A.audio, B.audio, **grids, cue_out_a=q["a_out_s"], cue_in_b=q["b_in_s"],
                                                 switch_rel_s=sw["switch_rel_s"], tail_end_a_s=tail_end_s, vocals_a=A.vocals,
                                                 vocals_b=B.vocals, pickup_mode="stem" if kind == "cut" else "switch",
                                                 sample_rate=SR, provenance=prov, render_config=CUT_CFG, effects_config=FX_CFG)
                versions.append((f"phrase_{kind}", res))
            except Exception as exc:  # noqa: BLE001
                versions.append((f"phrase_{kind}", exc))
        # 3: same cues, B on its downbeat, no tail (dj_render cut as today)
        try:
            res = dj.render_transition(A.audio, B.audio, **grids, cue_out_a=q["a_out_s"], cue_in_b=q["b_in_s"], kind="cut",
                                       sample_rate=SR, provenance=prov, return_decks=True, config=CUT_CFG)
            versions.append(("same_cue_downbeat_cut", res))
        except Exception as exc:  # noqa: BLE001
            versions.append(("same_cue_downbeat_cut", exc))
        # 4: dj_select default top plan rendered as a cut (old behaviour, no guard)
        dp = djs.plan_pair(A.track, B.track, top_k=1)["plans"]
        if dp:
            d0 = dp[0]
            try:
                res = dj.render_transition(A.audio, B.audio, **grids, cue_out_a=d0["a_out_s"], cue_in_b=d0["b_in_s"], kind="cut",
                                           sample_rate=SR, provenance=dict(prov, plan_default={k2: d0[k2] for k2 in ("a_out_s", "a_out_bar", "b_in_s", "b_in_bar", "type", "score")}),
                                           return_decks=True, config=CUT_CFG)
                versions.append(("dj_select_default_cut", res))
            except Exception as exc:  # noqa: BLE001
                versions.append(("dj_select_default_cut", exc))
        items = []
        for name, res in versions:
            if isinstance(res, Exception):
                items.append({"version": name, "error": repr(res)})
                print(f"  {base_name} {name}: errore {res!r}", flush=True)
                continue
            n = len(res.audio)
            meta = res.metadata
            tr = out_truncations(meta, A, B, n)
            meta = dict(meta, truncations=tr, version=name)
            qa_res = None
            if qa:
                try:
                    va, vb = dj_qa.stems_to_output(meta, n, A.vocals, B.vocals)
                    a_voc_exit = tr["a_exit_source_sample"]  # A's vocal is not heard after the fader switch
                    heard = [w for w in A.model["words"] if w["on"] < a_voc_exit]  # same rule for every version
                    words = dj_qa.words_to_output(meta, [{"start_sample": w["on"], "end_sample": w["off"], "text": w["text"]} for w in heard], n)
                    qa_res = dj_qa.evaluate_render(res.audio.astype(np.float64), meta, sample_rate=SR, decks=res.decks,
                                                   vocals_a=va, vocals_b=vb, vocal_thresholds=(A.vocal_threshold, B.vocal_threshold),
                                                   words_a=words, tracker=tracker,
                                                   grid_support={"a": A.track["tempo"], "b": B.track["tempo"]})
                except Exception as exc:  # noqa: BLE001
                    qa_res = {"error": repr(exc)}
            meta["qa"] = qa_res
            res.metadata = jsonable(meta)
            w = dj.write_render(res, clips, f"{base_name}__{name}")
            items.append({"version": name, "wav": rel(w["wav"]), "json": rel(w["json"]), "sha256": w["sha256"],
                          "duration_s": n / SR, "truncations": tr,
                          "qa_verdict": (qa_res or {}).get("verdict"), "qa_fail": (qa_res or {}).get("fail_reasons"),
                          "qa_warn": (qa_res or {}).get("warn_reasons")})
            print(f"  {base_name} {name}: A tronc {len(tr['a_truncated_refined'])}/{len(tr['a_truncated_raw'])} "
                  f"B tronc {len(tr['b_truncated_refined'])}/{len(tr['b_truncated_raw'])} QA {(qa_res or {}).get('verdict')}", flush=True)
        group = {"id": base_name, "pool": c["pool"], "a": A.slug, "b": B.slug, "compat": c["compat"],
                         "plan": prov["plan"], "switch_rel_s": sw["switch_rel_s"], "pickup_s": sw.get("pickup_samples", 0) / SR,
                         "tail_s": ((ex["tail_end_sample"] - ex["downbeat_sample"]) / SR) if ex.get("tail_end_sample") else None,
                         "exit": {k2: ex.get(k2) for k2 in ("status", "p_line_end", "last_word", "next_word", "reasons")},
                         "entry": {k2: q["phrase"]["entry"].get(k2) for k2 in ("status", "pickup_words", "first_word", "reasons")},
                         "items": items}
        gpath.write_text(json.dumps(jsonable(group), indent=1, ensure_ascii=False))
        manifest.append(group)
        del versions
    (OUT / "render_manifest.json").write_text(json.dumps(jsonable({"schema_version": "musicmix.dj_phrase_render.1",
                                                                   "clip_rule": "cut/filter_out: 10 s before the downbeat (filter: before the sweep) + 10 s after (AGENTS.md); the phrase switch moves the fader up to 2 beats before the downbeat inside the clip",
                                                                   "groups": manifest}), indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stages", nargs="+", choices=["analyze", "compare", "render"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=8)
    ap.add_argument("--no-qa", action="store_true")
    args = ap.parse_args()
    if "analyze" in args.stages:
        analyze({pool: complete_slugs(pool) for pool in POOLS}, force=args.force)
    if "compare" in args.stages:
        res = compare()
        print(json.dumps({"totals": res["totals"], "pair_totals": res["pair_totals"], "top_plan_totals": res["top_plan_totals"]}, indent=1))
    if "render" in args.stages:
        render(args.max_pairs, qa=not args.no_qa)


if __name__ == "__main__":
    main()
