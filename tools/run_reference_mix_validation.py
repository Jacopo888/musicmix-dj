"""Validazione della pipeline dei mix di riferimento su verità note (nessun dato dell'utente).

  python tools/run_reference_mix_validation.py sets [--degraded DIR]
  python tools/run_reference_mix_validation.py unmixdb --root DIR_SET_UNMIXDB [--variants none-none,stretch-none,...]
  python tools/run_reference_mix_validation.py summary

* ``sets``: ``runs/dj_set_v3_bridge`` (sviluppo), ``runs/dj_set_v1b`` e ``runs/dj_set_v2_books``
  (tenuti da parte: soglie non ritoccate dopo averli visti), con i brani del catalogo non suonati
  come distrattori. ``--degraded`` aggiunge le varianti del v3 (mp3 128 kbps; +3 % di velocità
  con ``asetrate`` = tempo +3 % e +0,51 semitoni; +1 semitone con rubberband a tempo invariato).
* ``unmixdb``: mix sintetici di UnmixDB (Zenodo 1422385, CC BY-NC-ND: solo uso di ricerca,
  niente ridistribuzione) con 2 estratti di altri mix come distrattori.

Uscite in ``runs/reference_mix_v1/validation/`` e ``runs/reference_mix_v1/VALIDATION_SUMMARY.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from analyze_reference_mix import rows_from_set, specs_from_rows  # noqa: E402
from musicmix.mix_alignment import AlignConfig, TrackSpec, analyze_mix, default_tracker  # noqa: E402
from validate_reference_mix import FAMILY, compare, truth_from_set, truth_from_unmixdb  # noqa: E402

OUT = ROOT / "runs" / "reference_mix_v1"
VAL = OUT / "validation"
CACHE = OUT / "cache"
SETS = {"v3_bridge": "runs/dj_set_v3_bridge", "v1b": "runs/dj_set_v1b", "v2_books": "runs/dj_set_v2_books"}


def _run(name: str, mix: Path, specs, truth: dict, tracker, cfg: AlignConfig, log) -> dict:
    t0 = time.time()
    res = analyze_mix(mix, specs, cfg, tracker=tracker, cache_dir=CACHE, log=log)
    res["elapsed_s"] = round(time.time() - t0, 1)
    VAL.mkdir(parents=True, exist_ok=True)
    (VAL / f"{name}.reference_mix.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    rep = compare(res, truth)
    rep.update({"schema_version": "musicmix.reference-mix-validation.1", "name": name, "mix": str(mix),
                "time_scale": truth.get("time_scale", 1.0)})
    (VAL / f"{name}.validation.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    s = rep["summary"]
    log(f"[{name}] ordine esatto={rep['order']['exact']} trovati={rep['order']['found']} FP={rep['order']['false_positives']} "
        f"mappa mediana={rep['time_map_solo']['error_ms']} switch={s['switch_abs_err_beats']} tecnica={s['technique_accuracy']}")
    return rep


def run_sets(args, tracker, log):
    cfg = AlignConfig()
    for name, d in SETS.items():
        if args.only and name not in args.only.split(","):
            continue
        set_json = ROOT / d / "set.json"
        specs = specs_from_rows(rows_from_set(set_json, include_pool=True))
        _run(name, ROOT / d / "set.wav", specs, truth_from_set(set_json), tracker, cfg, log)
    if args.degraded:
        set_json = ROOT / SETS["v3_bridge"] / "set.json"
        specs = specs_from_rows(rows_from_set(set_json, include_pool=True))
        deg = Path(args.degraded)
        variants = [("v3_mp3_128k", "v3_128k.mp3", 1.0, (0,)), ("v3_vinyl_plus3pct_128k", "v3_vinyl103_128k.mp3", 1.03, (0, 1)),
                    ("v3_pitch_plus1st_128k", "v3_pitch1_128k.mp3", 1.0, (0, 1))]
        for name, fn, scale, keys in variants:
            if (deg / fn).is_file():
                cfg = AlignConfig(key_shifts=keys)
                _run(name, deg / fn, specs, truth_from_set(set_json, time_scale=scale), tracker, cfg, log)


def run_unmixdb(args, tracker, log):
    root = Path(args.root)
    mixes = sorted((root / "mixes").glob("*.mp3"))
    variants = args.variants.split(",") if args.variants else None
    pool = sorted((root / "refsongs").glob("*.excerpt40.mp3"))
    rng = random.Random(20261021)
    cfg = AlignConfig(min_diag_beats=8, key_shifts=(-1, 0, 1))
    for mp3 in mixes:
        stem = mp3.name[: -len(".mp3")]
        if variants and not any(f"-{v}-" in stem for v in variants):
            continue
        labels = mp3.with_suffix("").with_suffix(".labels.txt") if False else root / "mixes" / f"{stem}.labels.txt"
        truth = truth_from_unmixdb(labels)
        names = [truth["tracks"][s]["file"] for s in truth["order"]]
        distract = [p for p in pool if p.name not in names]
        extra = rng.sample(distract, 2)
        specs = [TrackSpec(slug=Path(n).stem, audio=str(root / "refsongs" / n)) for n in names]
        specs += [TrackSpec(slug=p.stem, audio=str(p)) for p in extra]
        _run(f"unmixdb_{stem}", mp3, specs, truth, tracker, cfg, log)


def _collect(prefix_filter) -> list[dict]:
    out = []
    for p in sorted(VAL.glob("*.validation.json")):
        rep = json.loads(p.read_text(encoding="utf-8"))
        if prefix_filter(rep["name"]):
            out.append(rep)
    return out


def _agg(reps: list[dict]) -> dict:
    def q(vals):
        v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
        if not len(v):
            return None
        return {"n": int(len(v)), "median": round(float(np.median(v)), 3), "p90": round(float(np.percentile(v, 90)), 3),
                "share_le_0.5": round(float(np.mean(v <= 0.5)), 3), "share_le_1": round(float(np.mean(v <= 1.0)), 3)}

    trs = [t for r in reps for t in r["transitions"] if t.get("matched_pair")]
    all_trs = [t for r in reps for t in r["transitions"]]
    conf: dict = {}
    for t in all_trs:
        k = t.get("pred_technique", "(coppia non trovata)")
        conf.setdefault(t["truth_technique"], {}).setdefault(k, 0)
        conf[t["truth_technique"]][k] += 1
    tm = [r["time_map_solo"] for r in reps]
    return {
        "n_mixes": len(reps),
        "order_exact": f"{sum(r['order']['exact'] for r in reps)}/{len(reps)}",
        "found": [r["order"]["found"] for r in reps],
        "false_positives": sum(len(r["order"]["false_positives"]) for r in reps),
        "time_map_median_ms_per_mix": [None if not x["error_ms"] else x["error_ms"]["median"] for x in tm],
        "time_map_share_within_50ms_per_mix": [x["share_within_50ms"] for x in tm],
        "pairs_matched": f"{len(trs)}/{len(all_trs)}",
        "switch_abs_err_beats": q([abs(t["switch_err_beats"]) for t in trs]),
        "b_start_abs_err_beats": q([abs(t["b_start_err_beats"]) for t in trs]),
        "a_end_abs_err_beats": q([abs(t["a_end_err_beats"]) for t in trs]),
        "overlap_abs_err_beats": q([abs(t["overlap_pred_beats"] - t["overlap_truth_beats"]) for t in trs]),
        "b_cue_abs_err_beats": q([abs(t["b_cue_err_beats"]) for t in trs if t.get("b_cue_err_beats") is not None]),
        "a_cue_abs_err_beats": q([abs(t["a_cue_err_beats"]) for t in trs if t.get("a_cue_err_beats") is not None]),
        "b_cue_bar_exact": _rate([t.get("b_cue_bar_exact") for t in trs]),
        "b_entry_cue_abs_err_beats": q([abs(t["b_entry_cue_err_beats"]) for t in trs if t.get("b_entry_cue_err_beats") is not None]),
        "a_exit_cue_abs_err_beats": q([abs(t["a_exit_cue_err_beats"]) for t in trs if t.get("a_exit_cue_err_beats") is not None]),
        "b_entry_bar_exact": _rate([t.get("b_entry_bar_exact") for t in trs]),
        "a_exit_bar_exact": _rate([t.get("a_exit_bar_exact") for t in trs]),
        "b_map_abs_err_at_true_switch_beats": q([abs(t["b_map_err_at_true_switch_beats"]) for t in trs if t.get("b_map_err_at_true_switch_beats") is not None]),
        "a_cue_bar_exact": _rate([t.get("a_cue_bar_exact") for t in trs]),
        "b_cue_abs_err_s": q([abs(t["b_cue_err_s"]) for t in trs if t.get("b_cue_err_s") is not None]),
        "b_tempo_abs_err_pct": q([abs(t["b_tempo_err_pct"]) for t in trs if t.get("b_tempo_err_pct") is not None]),
        "effect_start_abs_err_beats": q([abs(t["effect_start_err_beats"]) for t in trs if t.get("effect_start_err_beats") is not None]),
        "technique_accuracy": _rate([t.get("pred_technique") == t["truth_technique"] for t in trs]),
        "technique_family_accuracy": _rate([FAMILY.get(t.get("pred_technique")) == FAMILY.get(t["truth_technique"]) for t in trs]),
        "blend_vs_not_accuracy": _rate([(t.get("pred_technique") == "blend") == (t["truth_technique"] == "blend") for t in trs]),
        "technique_confusion": conf,
    }


def _rate(xs):
    xs = [x for x in xs if x is not None]
    return None if not xs else {"rate": round(sum(bool(x) for x in xs) / len(xs), 3), "n": len(xs)}


def summary(log):
    groups = {
        "dev_v3_bridge": lambda n: n == "v3_bridge",
        "heldout_v1b_v2_books": lambda n: n in ("v1b", "v2_books"),
        "degraded_v3": lambda n: n.startswith("v3_") and n != "v3_bridge",
        "unmixdb": lambda n: n.startswith("unmixdb_"),
    }
    out = {"schema_version": "musicmix.reference-mix-validation-summary.1", "groups": {}}
    for g, f in groups.items():
        reps = _collect(f)
        if reps:
            out["groups"][g] = _agg(reps)
            out["groups"][g]["mixes"] = [r["name"] for r in reps]
    for name in ("v3_mp3_128k", "v3_vinyl_plus3pct_128k", "v3_pitch_plus1st_128k"):
        reps = _collect(lambda n, name=name: n == name)
        if reps:
            out["groups"][name] = _agg(reps)
    (OUT / "VALIDATION_SUMMARY.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log(json.dumps(out, ensure_ascii=False, indent=1)[:6000])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["sets", "unmixdb", "summary"])
    ap.add_argument("--degraded")
    ap.add_argument("--only")
    ap.add_argument("--root")
    ap.add_argument("--variants")
    ap.add_argument("--tracker", choices=["beat_this", "librosa"], default="beat_this")
    args = ap.parse_args(argv)

    def log(m):
        print(m, flush=True)

    if args.what == "summary":
        summary(log)
        return 0
    tracker = default_tracker()
    if args.what == "sets":
        run_sets(args, tracker, log)
    else:
        run_unmixdb(args, tracker, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
