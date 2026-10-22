"""Valida ``musicmix.reference-mix.1`` contro una verità nota.

Due sorgenti di verità, entrambe senza dati dell'utente:

* ``set``: i nostri set renderizzati (``runs/dj_set_*/set.json``): segmenti grezzi con
  campioni sorgente esatti, tempi di inizio/switch/fine delle transizioni, stretch di B
  nella clip, punto di giunzione. Opzione ``--time-scale`` per mix degradati con
  velocità cambiata (es. 1.03 = +3 % con ``asetrate``: i tempi del mix si dividono per 1.03).
* ``unmixdb``: file ``.labels.txt`` di UnmixDB (Schwarz & Fourer 2018; mix sintetici di
  3 estratti CC con dissolvenze lineari, verità esatta: start, speed, fadein, fadeout).

Metriche: ordine, errore della mappa temporale nei tratti a brano solo (ms e battiti),
switch, ingresso udibile di B, fine udibile di A, sovrapposizione (battiti), cue nel brano
(battiti e battute, con la griglia dell'analisi), battuta d'ingresso esatta, tecnica.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicmix.paths import localize  # noqa: E402

# famiglie: l'eco è un effetto di coda, filtro/build/loop sono preparazioni su A, blend e cut a parte
FAMILY = {"blend": "blend", "cut": "cut", "echo": "echo", "filter": "prep", "build": "prep", "loop": "prep"}
# build_into_chorus = passa-alto + rullo + riser su A: il classificatore lo chiama "filter" (vedi docstring)
TECH_MAP = {"blend4": "blend", "blend8": "blend", "blend": "blend", "echo_out": "echo", "filter_out": "filter",
            "build_into_chorus": "filter", "halving_build": "loop", "loop_extend": "loop", "cut": "cut"}


# ---------------------------------------------------------------------- verità dai nostri set


def truth_from_set(set_json: Path, time_scale: float = 1.0) -> dict:
    d = json.loads(Path(set_json).read_text(encoding="utf-8"))
    sr = int(d["output"].get("sample_rate", 48000))
    order = [t["slug"] for t in d["tracklist"]]
    raws = {}
    for seg in d["segments"]:
        if seg["kind"] == "raw":
            raws.setdefault(seg["slug"], []).append(seg)
    bpm = {t["slug"]: float(t["bpm"]) for t in d["tracklist"]}
    analyses = {}
    for s in order:
        ap = d["sources"][s].get("analysis")
        p = localize(ap) if ap else None
        analyses[s] = json.loads(Path(p).read_text(encoding="utf-8")) if p and Path(p).is_file() else None
    tracks = {}
    for s in order:
        segs = raws.get(s, [])
        if not segs:
            continue
        seg = segs[0]
        s0, s1 = seg["source_samples"]
        tracks[s] = {"raw_set_s": [seg["set_start_s"], seg["set_end_s"]], "raw_src_s": [s0 / sr, s1 / sr], "bpm": bpm[s]}
    trans = []
    for t in d["transitions"]:
        a, b = t["a"], t["b"]
        st = t["set_times_s"]
        typ = t.get("technique") or t["type"]
        coarse = TECH_MAP.get(typ, typ)
        blend = coarse == "blend"
        ta, tb = tracks[a], tracks[b]
        rate_b = 1.0 + float(t.get("stretch_pct") or 0.0) / 100.0 if blend else 1.0

        def src_b(x):
            return tb["raw_src_s"][0] - (tb["raw_set_s"][0] - x) * rate_b

        def src_a(x):
            return ta["raw_src_s"][1] + (x - ta["raw_set_s"][1])

        b_start = st["start"] if blend else st["switch"]
        # fine di A come *continuazione* del brano: nell'eco e nel loop A smette di avanzare
        # all'inizio dell'effetto (poi ripete/riecheggia materiale già suonato); nel filtro e nel
        # build A continua (filtrato) fino allo switch; nel blend fino alla fine della dissolvenza.
        a_end = st["end"] if blend else (st["start"] if coarse in ("echo", "loop") else st["switch"])
        bp = 60.0 / tb["bpm"] if blend else 60.0 / ta["bpm"]
        bp_a = 60.0 / ta["bpm"]
        trans.append({
            "index": t["index"], "a": a, "b": b, "type": typ, "technique": coarse,
            "start": st["start"] / time_scale, "switch": st["switch"] / time_scale, "end": st["end"] / time_scale,
            "b_audible_start": b_start / time_scale, "a_audible_end": a_end / time_scale,
            "overlap_beats": (a_end - b_start) / bp_a,
            "b_src_at_switch": src_b(st["switch"]), "a_src_at_switch": src_a(st["switch"]),
            "b_src_at_start": src_b(b_start), "a_src_at_end": src_a(a_end),
            "plan": {"a_out_s": t["plan"]["a_out_s"], "b_in_s": t["plan"]["b_in_s"]},
            "beat_s_mix": bp_a / time_scale,
        })
    return {"kind": "set", "order": order, "tracks": tracks, "transitions": trans, "analyses": analyses,
            "time_scale": time_scale}


def truth_from_unmixdb(labels: Path) -> dict:
    rows = []
    for line in Path(labels).read_text(encoding="utf-8").splitlines():
        p = line.split("\t")
        if len(p) >= 4:
            rows.append((float(p[0]), float(p[1]), int(p[2]), p[3], p[4].strip('"') if len(p) > 4 else ""))
    tr = {}
    for t0, t1, k, lab, par in rows:
        e = tr.setdefault(k, {})
        if lab == "start":
            e["start"], e["file"] = t0, par
        elif lab == "speed":
            e["speed"] = float(par)
        elif lab in ("fadein", "fadeout"):
            e[lab] = (t0, t1)
        elif lab == "stop":
            e["stop"] = t0
        elif lab == "bpm":
            e["bpm"] = float(par)
    ks = sorted(tr)
    order = [Path(tr[k]["file"]).stem for k in ks]
    tracks = {}
    for k in ks:
        e = tr[k]
        tracks[Path(e["file"]).stem] = {"start": e["start"], "speed": e.get("speed", 1.0), "fadein": e.get("fadein"),
                                       "fadeout": e.get("fadeout"), "stop": e.get("stop"), "file": e["file"]}
    trans = []
    for ka, kb in zip(ks[:-1], ks[1:]):
        a, b = Path(tr[ka]["file"]).stem, Path(tr[kb]["file"]).stem
        ta, tb = tracks[a], tracks[b]
        b_start, a_end = tb["fadein"][0], ta["fadeout"][1]
        sw = (max(tb["fadein"][0], ta["fadeout"][0]) + min(tb["fadein"][1], ta["fadeout"][1])) / 2
        trans.append({"a": a, "b": b, "technique": "blend", "b_audible_start": b_start, "a_audible_end": a_end,
                      "switch": sw, "overlap_s": a_end - b_start,
                      "b_src_at_switch": (sw - tb["start"]) * tb["speed"], "a_src_at_switch": (sw - ta["start"]) * ta["speed"]})
    return {"kind": "unmixdb", "order": order, "tracks": tracks, "transitions": trans}


# ---------------------------------------------------------------------- confronto


def _interp_map(row: dict, t: np.ndarray) -> np.ndarray:
    tm = row.get("time_map") or {}
    mx, tk = np.asarray(tm.get("mix_s", []), float), np.asarray(tm.get("track_s", []), float)
    if len(mx) < 2:
        return np.full(len(t), np.nan)
    out = np.interp(t, mx, tk)
    out[(t < mx[0]) | (t > mx[-1])] = np.nan
    return out


def _kendall(pred: list, truth: list) -> float:
    common = [s for s in truth if s in pred]
    if len(common) < 2:
        return float("nan")
    pos = {s: pred.index(s) for s in common}
    c = dcount = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            if pos[common[i]] < pos[common[j]]:
                c += 1
            else:
                dcount += 1
    return (c - dcount) / (c + dcount)


def _q(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(x):
        return None
    return {"n": int(len(x)), "median": round(float(np.median(x)), 3), "mean": round(float(np.mean(x)), 3),
            "p90": round(float(np.percentile(x, 90)), 3), "max": round(float(np.max(x)), 3)}


def compare(res: dict, truth: dict) -> dict:
    rows = {r["slug"]: r for r in res["tracks"]}
    pred_order = res["order"]
    out: dict = {"order": {"truth": truth["order"], "pred": pred_order,
                           "exact": pred_order == [s for s in truth["order"] if s in pred_order] and len(pred_order) == len(truth["order"]),
                           "found": f"{sum(1 for s in truth['order'] if rows.get(s, {}).get('found'))}/{len(truth['order'])}",
                           "kendall_tau": round(_kendall(pred_order, truth["order"]), 3),
                           "false_positives": [s for s in pred_order if s not in truth["order"]]}}
    # mappa temporale nei tratti a brano solo
    errs_ms, errs_beats = [], []
    per_track = {}
    for s, tt in truth["tracks"].items():
        if s not in rows:
            continue
        if truth["kind"] == "set":
            a0, a1 = tt["raw_set_s"]
            ts = np.arange(a0 + 2.0, a1 - 2.0, 0.25)
            true_src = tt["raw_src_s"][0] + (ts - a0)
            ts_mix = ts / truth["time_scale"]
            beat = 60.0 / tt["bpm"]
        else:
            fi, fo = tt["fadein"], tt["fadeout"]
            ts_mix = np.arange(fi[1] + 1.0, fo[0] - 1.0, 0.25)
            true_src = (ts_mix - tt["start"]) * tt["speed"]
            beat = 0.47
        pred = _interp_map(rows[s], ts_mix)
        e = np.abs(pred - true_src)
        e_ok = e[np.isfinite(e)]
        cover = float(np.mean(np.isfinite(e))) if len(e) else 0.0
        per_track[s] = {"coverage": round(cover, 3), "median_ms": None if not len(e_ok) else round(float(np.median(e_ok)) * 1000, 1),
                        "p90_ms": None if not len(e_ok) else round(float(np.percentile(e_ok, 90)) * 1000, 1)}
        errs_ms += list(e_ok * 1000)
        errs_beats += list(e_ok / beat)
    em = np.asarray(errs_ms)
    out["time_map_solo"] = {"per_track": per_track, "error_ms": _q(errs_ms), "error_beats": _q(errs_beats),
                            "share_within_50ms": round(float(np.mean(em <= 50)), 4) if len(em) else None,
                            "share_within_quarter_beat": round(float(np.mean(np.asarray(errs_beats) <= 0.25)), 4) if len(em) else None}
    # transizioni
    pred_tr = {(t["a"], t["b"]): t for t in res["transitions"]}
    trs = []
    conf: dict = {}
    for gt in truth["transitions"]:
        p = pred_tr.get((gt["a"], gt["b"]))
        row = {"a": gt["a"], "b": gt["b"], "truth_technique": gt["technique"], "matched_pair": p is not None}
        if p is None:
            trs.append(row)
            conf.setdefault(gt["technique"], {}).setdefault("(coppia non trovata)", 0)
            conf[gt["technique"]]["(coppia non trovata)"] += 1
            continue
        bp = gt.get("beat_s_mix") or p["evidence"].get("beat_period_s") or 0.5
        ms = p["mix_s"]
        row.update({
            "pred_technique": p["technique"],
            "switch_err_beats": round((ms["switch"] - gt["switch"]) / bp, 2),
            "b_start_err_beats": round((ms["b_audible_start"] - gt["b_audible_start"]) / bp, 2),
            "a_end_err_beats": round((ms["a_audible_end"] - gt["a_audible_end"]) / bp, 2),
            "overlap_pred_beats": p["overlap_beats"],
            "effect_start_err_beats": None if gt["technique"] in ("blend", "cut") or p["evidence"].get("effect_start_s") is None
            else round((p["evidence"]["effect_start_s"] - gt["start"]) / bp, 2),
            "overlap_truth_beats": round(gt.get("overlap_beats", gt.get("overlap_s", 0) / bp), 2),
        })
        bsrc = p["b_entry"]["at_switch"].get("source_s")
        asrc = p["a_exit"]["at_switch"].get("source_s")
        b_in = p["b_entry"]["at_audible_start"].get("source_s")
        a_out = p["a_exit"]["at_audible_end"].get("source_s")
        if truth["kind"] == "set":
            an_b = truth["analyses"].get(gt["b"]) or {}
            an_a = truth["analyses"].get(gt["a"]) or {}
            bb = 60.0 / truth["tracks"][gt["b"]]["bpm"]
            ba = 60.0 / truth["tracks"][gt["a"]]["bpm"]
            if bsrc is not None:
                row["b_cue_err_beats"] = round((bsrc - gt["b_src_at_switch"]) / bb, 2)
                row["b_cue_bar_exact"] = _bar(an_b, bsrc) == _bar(an_b, gt["b_src_at_switch"])
                row["b_cue_bar_truth"] = _bar(an_b, gt["b_src_at_switch"])
                row["b_cue_bar_pred"] = _bar(an_b, bsrc)
            if asrc is not None:
                row["a_cue_err_beats"] = round((asrc - gt["a_src_at_switch"]) / ba, 2)
                row["a_cue_bar_exact"] = _bar(an_a, asrc) == _bar(an_a, gt["a_src_at_switch"])
            # cue d'ingresso/uscita come evento: sorgente al punto udibile stimato vs vero
            if b_in is not None:
                row["b_entry_cue_err_beats"] = round((b_in - gt["b_src_at_start"]) / bb, 2)
                row["b_entry_bar_exact"] = _bar(an_b, b_in) == _bar(an_b, gt["b_src_at_start"])
            if a_out is not None:
                row["a_exit_cue_err_beats"] = round((a_out - gt["a_src_at_end"]) / ba, 2)
                row["a_exit_bar_exact"] = _bar(an_a, a_out) == _bar(an_a, gt["a_src_at_end"])
            # mappa temporale nella zona di transizione: sorgente di B allo switch VERO
            tmb = _interp_map(rows[gt["b"]], np.asarray([gt["switch"]]))[0] if gt["b"] in rows else np.nan
            if np.isfinite(tmb):
                row["b_map_err_at_true_switch_beats"] = round((tmb - gt["b_src_at_switch"]) / bb, 3)
        else:
            if bsrc is not None:
                row["b_cue_err_s"] = round(bsrc - gt["b_src_at_switch"], 3)
            if asrc is not None:
                row["a_cue_err_s"] = round(asrc - gt["a_src_at_switch"], 3)
        # tempo di B all'inizio del suo tratto allineato: verità = speed di UnmixDB, oppure 0 % nei nostri
        # set fuori dai blend (B entra alla sua velocità; nei blend lo stretch vale solo dentro la clip)
        tb = (p.get("tempo") or {}).get("b_tempo_change_pct")
        if tb is not None:
            if truth["kind"] == "unmixdb":
                row["b_tempo_err_pct"] = round(tb - (truth["tracks"][gt["b"]]["speed"] - 1) * 100, 2)
            elif gt["technique"] != "blend":
                row["b_tempo_err_pct"] = round(tb - (truth.get("time_scale", 1.0) - 1) * 100, 2)
        trs.append(row)
        conf.setdefault(gt["technique"], {}).setdefault(p["technique"], 0)
        conf[gt["technique"]][p["technique"]] += 1
    out["transitions"] = trs
    ok = [r for r in trs if r.get("matched_pair")]

    def absq(key):
        return _q([abs(r[key]) for r in ok if r.get(key) is not None])

    out["summary"] = {
        "pairs_matched": f"{len(ok)}/{len(trs)}",
        "switch_abs_err_beats": absq("switch_err_beats"),
        "b_start_abs_err_beats": absq("b_start_err_beats"),
        "a_end_abs_err_beats": absq("a_end_err_beats"),
        "overlap_abs_err_beats": _q([abs(r["overlap_pred_beats"] - r["overlap_truth_beats"]) for r in ok]),
        "b_cue_abs_err_beats": absq("b_cue_err_beats"),
        "a_cue_abs_err_beats": absq("a_cue_err_beats"),
        "b_cue_bar_exact": _rate([r.get("b_cue_bar_exact") for r in ok]),
        "b_entry_cue_abs_err_beats": absq("b_entry_cue_err_beats"),
        "a_exit_cue_abs_err_beats": absq("a_exit_cue_err_beats"),
        "b_entry_bar_exact": _rate([r.get("b_entry_bar_exact") for r in ok]),
        "a_exit_bar_exact": _rate([r.get("a_exit_bar_exact") for r in ok]),
        "b_map_abs_err_at_true_switch_beats": absq("b_map_err_at_true_switch_beats"),
        "a_cue_bar_exact": _rate([r.get("a_cue_bar_exact") for r in ok]),
        "b_cue_abs_err_s": absq("b_cue_err_s"),
        "a_cue_abs_err_s": absq("a_cue_err_s"),
        "effect_start_abs_err_beats": absq("effect_start_err_beats"),
        "b_tempo_abs_err_pct": absq("b_tempo_err_pct"),
        "technique_accuracy": _rate([r.get("pred_technique") == r["truth_technique"] for r in ok]),
        "technique_family_accuracy": _rate([FAMILY.get(r.get("pred_technique")) == FAMILY.get(r["truth_technique"]) for r in ok]),
        "blend_vs_not_accuracy": _rate([(r.get("pred_technique") == "blend") == (r["truth_technique"] == "blend") for r in ok]),
        "technique_confusion": conf,
    }
    return out


def _bar(an: dict, src: float):
    db = np.asarray(((an or {}).get("grid") or {}).get("downbeats_s") or [], float)
    if len(db) < 2 or src is None:
        return None
    # battuta del downbeat più vicino: i cue si contano sul downbeat (convenzione DJ)
    return int(np.argmin(np.abs(db - src)))


def _rate(xs):
    xs = [x for x in xs if x is not None]
    return None if not xs else {"rate": round(sum(bool(x) for x in xs) / len(xs), 3), "n": len(xs)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", help="JSON musicmix.reference-mix.1")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--set", dest="set_json")
    g.add_argument("--unmixdb-labels")
    ap.add_argument("--time-scale", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    res = json.loads(Path(args.result).read_text(encoding="utf-8"))
    truth = truth_from_set(Path(args.set_json), args.time_scale) if args.set_json else truth_from_unmixdb(Path(args.unmixdb_labels))
    rep = compare(res, truth)
    rep["schema_version"] = "musicmix.reference-mix-validation.1"
    rep["result"] = args.result
    rep["truth"] = args.set_json or args.unmixdb_labels
    rep["time_scale"] = args.time_scale
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"order": {k: rep["order"][k] for k in ("exact", "found", "kendall_tau")},
                      "time_map": {k: rep["time_map_solo"][k] for k in ("error_ms", "share_within_50ms")},
                      "summary": rep["summary"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
