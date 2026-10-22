#!/usr/bin/env python3
"""Confronto dei piani con e senza affidabilità (``runs/plan_reliability_v1``) e metriche di copertura e varietà.

Legge i ``set.json`` di sola pianificazione prodotti da ``tools/run_dj_set_v3_bridge.py set --plan-only --event-plan``
nelle varianti (``off``, ``report``, ``prefer``, ...) e scrive ``compare.json`` nella stessa cartella: che cosa cambia
(brani, coppie, cue, tecniche, coppie rese non pianificabili e perché), affidabilità media dei set, copertura della
libreria (schede ammesse e pool del set) e varietà nei set successivi (senza e con l'opzione di novità).

    python tools/plan_reliability_report.py runs/plan_reliability_v1
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicmix import reliability as R  # noqa: E402


def load(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def variant_row(name: str, doc: dict) -> dict:
    s = R.set_summary(doc)
    rel = doc.get("reliability") or {}
    return {"name": name, "order": s["order"], "order_cost": (doc.get("ordering") or {}).get("cost"),
            "cue_cost": (doc.get("cues") or {}).get("cost"), "techniques": [x["technique"] for x in s["transitions"]],
            "set_reliability": rel.get("set_reliability"), "tracks_avoided": rel.get("tracks_avoided"),
            "pairs_unplannable": (rel.get("pairs") or {}).get("made_unplannable"), "pairs_best_changed": (rel.get("pairs") or {}).get("best_changed"),
            "excluded_by_runner": [x for x in (doc.get("log") or []) if "pool_v3" in x][0]["pool_v3"]["skipped"] if any("pool_v3" in x for x in doc.get("log") or []) else None}


def explain(base_doc: dict, report_doc: dict, new_doc: dict) -> list:
    """Per ogni transizione del piano base che cambia: affidabilità della scelta base (variante report) e della nuova."""
    rep = {(t["a"], t["b"]): t for t in (report_doc.get("reliability") or {}).get("transitions") or []}
    new = {(t["a"], t["b"]): t for t in (new_doc.get("reliability") or {}).get("transitions") or []}
    cmp = R.compare_plans(base_doc, new_doc)
    rows = []
    for x in R.set_summary(base_doc)["transitions"]:
        k = (x["a"], x["b"])
        y = new.get(k)
        same = y is not None and [y["a_out_bar"], y["b_in_bar"]] == [x["a_out_bar"], x["b_in_bar"]]
        if same:
            continue
        r = rep.get(k) or {}
        rows.append({"a": x["a"], "b": x["b"], "base_cue": [x["a_out_bar"], x["b_in_bar"]], "base_technique": x["technique"],
                     "base_reliability": r.get("reliability"), "base_why": r.get("why"),
                     "new": None if y is None else {"cue": [y["a_out_bar"], y["b_in_bar"]], "type": y["type"], "reliability": y["reliability"],
                                                    "why": y["why"]},
                     "pair_dropped": y is None})
    return [{"summary": {k: cmp[k] for k in ("same_order", "tracks_removed", "tracks_added", "pairs_removed", "pairs_added",
                                              "n_same_pairs", "n_same_cue", "n_same_technique")}}, rows]


def main(argv=None) -> int:
    run = Path((argv or sys.argv[1:] or ["runs/plan_reliability_v1"])[0])
    run = run if run.is_absolute() else ROOT / run
    docs = {p.parent.name: load(p) for p in sorted(run.glob("*/set.json"))}
    out = {"schema_version": "musicmix.plan-reliability-compare.1", "run": str(run.relative_to(ROOT)),
           "variants": {k: variant_row(k, d) for k, d in docs.items()}, "comparisons": {}}
    pairs = [("off", "report"), ("off", "prefer"), ("off", "prefer_w1.5"), ("off", "prefer_w3"), ("off_noexclude", "prefer_noexclude"),
             ("prefer", "repeat_2"), ("prefer", "novelty_2"), ("novelty_2", "novelty_3")]
    for a, b in pairs:
        if a in docs and b in docs:
            summ, rows = explain(docs[a], docs.get("report", docs[b]) if a == "off" else docs[a], docs[b])
            out["comparisons"][f"{a} -> {b}"] = {**summ, "changed_transitions": rows}
    if "prefer" in docs:
        rel = docs["prefer"].get("reliability") or {}
        out["prefer_pairs_unplannable"] = rel.get("pairs", {}).get("unplannable")
        out["prefer_tracks"] = rel.get("tracks")
    if "prefer_noexclude" in docs:
        out["prefer_noexclude_tracks_avoided"] = (docs["prefer_noexclude"].get("reliability") or {}).get("tracks_avoided")
        out["prefer_noexclude_unplannable"] = len(((docs["prefer_noexclude"].get("reliability") or {}).get("pairs") or {}).get("unplannable") or [])
    # copertura e varietà nei set successivi
    cards = R.load_cards(ROOT / "track_cards")
    pool = sorted((docs.get("off") or next(iter(docs.values())))["sources"])
    frozen = ROOT / "runs/dj_set_v3_bridge/set.json"
    seqs = {"stesso piano ripetuto (senza novità)": ["prefer", "repeat_2"], "con novità [H]": ["prefer", "novelty_2", "novelty_3"]}
    out["coverage_variety"] = {"library_cards": sorted(cards), "pool_set": pool}
    for label, names in seqs.items():
        sets = [dict(R.set_summary(docs[n]), name=n) for n in names if n in docs]
        if frozen.is_file():
            sets = [dict(R.set_summary(load(frozen)), name="set v3 congelato (render)")] + sets
        techniques = sorted({x["technique"] for d in docs.values() for x in R.set_summary(d)["transitions"] if x["technique"]})
        out["coverage_variety"][label] = {"sets": [s["name"] for s in sets], "coverage_pool": R.coverage(sets, pool),
                                          "coverage_cards": R.coverage(sets, sorted(cards)), "variety": R.variety(sets, techniques)}
    (run / "compare.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    for k, v in out["variants"].items():
        print(f"{k:18s} costo ordine {v['order_cost']:.3f}  affidabilità {v['set_reliability']}  evitati {v['tracks_avoided']}  "
              f"coppie non pianificabili {v['pairs_unplannable']}")
    for k, v in out["comparisons"].items():
        print(f"{k}: {v['summary']}; transizioni cambiate {len(v['changed_transitions'])}")
    print(f"scritto {run / 'compare.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
