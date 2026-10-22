#!/usr/bin/env python3
"""Riverifica QA (dj_qa.5) e critico v4 (dj_critic_books.2) sulle 13 transizioni finali del set v3 "ponte".

Non tocca la run congelata ``runs/dj_set_v3_bridge`` (legge ``set.json``, catalogo, analisi, modelli di frase e cache):
per ciascuna transizione rifà il render con la stessa decisione, lo stesso cue e gli stessi ingressi del runner
(``tools/run_dj_set_v3_bridge.py``, funzioni ``render_v3``/``qa_v3``/``critic``), verifica che il PCM 24 bit sia
identico al WAV accettato (sha256), poi calcola:

* ``qa_saved``: verdetto e motivi salvati in ``set.json`` (dj_qa.4, run congelata);
* ``qa_legacy``: QA di oggi con ``effect_word_cut="off"`` e ``require_coverage=False`` (stessa misura di .4 salvo la
  proiezione di B col time-map): separa l'effetto dei controlli nuovi da eventuali derive del tracker;
* ``qa_new``: QA di default (dj_qa.5: parola tagliata all'uscita con effetto, copertura, stati per controllo);
* ``critic_v4_saved`` / ``critic_v4_new``: blocco ``technical`` del critico v4 prima (set.json) e dopo.

Scrive ``runs/dj_set_v3_bridge/qa_v5_recheck.json`` (file nuovo; i file originali della run restano invariati).

    python tools/recheck_set_v3_qa.py [--only 2 6 7]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import run_dj_set_v3_bridge as R  # noqa: E402

from musicmix import dj_critic_books, dj_qa, dj_render as dj, dj_select as djs  # noqa: E402

OUT = R.OUT / "qa_v5_recheck.json"
R.B_ENTRY["on"] = False  # 21/10/2026: riproduce i render congelati (senza guardia/lead-in dell'ingresso di B, INGRESSO_B)


def _tech(v4: dict | None) -> dict | None:
    t = (v4 or {}).get("technical")
    if not t:
        return None
    return {"status": t.get("status"), "veto_flags": [f["flag"] for f in t.get("veto_flags") or []],
            "warnings": [w["check"] for w in t.get("warnings") or []],
            "indicative_flags": [f["flag"] for f in t.get("indicative_flags") or []],
            "declared_phase": t.get("declared_phase"), "harmonic_overlap": t.get("harmonic_overlap")}


def _qa_short(q: dict) -> dict:
    out = {"verdict": q["verdict"], "reasons": [f"{r['check']}: {r['message']}" for r in q["reasons"]]}
    if "coverage" in q:
        out["coverage"] = q["coverage"]
    vc = (q.get("checks") or {}).get("verse_cut") or {}
    out["verse_cut"] = {k: vc.get(k) for k in ("status", "coverage", "rule", "a_dry_end_s", "a_dry_end_source", "a_exit_s",
                                                "words_cut", "words_near_exit", "evidence", "reason", "a_voice_band_at_dry_end_db")
                        if vc.get(k) is not None}
    out["b_projection"] = (q.get("inputs") or {}).get("b_projection")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", type=int, nargs="*", help="transition indices (1-based)")
    ap.add_argument("--out", help="file di uscita (default runs/dj_set_v3_bridge/qa_v5_recheck.json); dj_qa.7: qa_v7_recheck.json")
    ap.add_argument("--previous", help="ricontrollo precedente da confrontare (es. runs/dj_set_v3_bridge/qa_v5_recheck.json)")
    args = ap.parse_args()
    global OUT
    if args.out:
        OUT = Path(args.out)
    prev = {}
    if args.previous and Path(args.previous).exists():
        pd = json.loads(Path(args.previous).read_text(encoding="utf-8"))
        prev = {t["index"]: t for t in pd["transitions"] if "qa_new" in t}
        prev_version = (pd.get("versions") or {}).get("dj_qa")
    doc = R.load(R.OUT / "set.json")
    cat = R.load(R.FROZEN)
    src = {"plan_dir": R.PLAN, "catalog": cat, "catalog_path": R.FROZEN}
    trs = [t for t in doc["transitions"] if not args.only or t["index"] in args.only]
    R.load_phrase({x for t in trs for x in (t["a"], t["b"])})
    tracker = dj_qa.BeatThisTracker()
    legacy_cfg = dataclasses.replace(R.QA_CFG, effect_word_cut="off", require_coverage=False)
    rows, loaded = [], {}
    t0 = time.time()
    for tr in trs:
        k, a, b = tr["index"], tr["a"], tr["b"]
        for key in list(loaded):
            if key not in (a, b):
                del loaded[key]
        for s in (a, b):
            if s not in loaded:
                loaded[s] = R.v1.Track(s, R.PLAN, cat)
        A, B = loaded[a], loaded[b]
        pp = djs.plan_pair(A.analysis, B.analysis, top_k=10 ** 9)
        plan = next(p for p in pp["plans"] if p["a_out_bar"] == tr["plan"]["a_out_bar"] and p["b_in_bar"] == tr["plan"]["b_in_bar"])
        opt = next(o for o in tr["books_options_same_cue"] if o["type_key"] == tr["technique"])
        dec = R.opt_decision(opt)
        prov = {"a": A.provenance, "b": B.provenance, "script": R.SCRIPT, "run": R.RUN, "recheck": "tools/recheck_set_v3_qa.py"}
        row = {"index": k, "a": a, "b": b, "technique": tr["technique"], "phrase_exit": (tr.get("phrase") or {}).get("exit", {}).get("status")}
        try:
            res, decks = R.render_v3(dec, A, B, plan, prov)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"render: {type(exc).__name__}: {exc}"
            rows.append(row)
            continue
        with tempfile.TemporaryDirectory() as tmp:
            sha = dj.write_render(res, tmp, "x")["sha256"]
        row["pcm_identical"] = sha == tr["clip_sha256"]
        row["pcm_sha256"] = sha
        q_new = R.qa_v3(res, A, B, decks, tracker, cfg=R.QA_CFG)
        q_old = R.qa_v3(res, A, B, decks, tracker, cfg=legacy_cfg)
        crt = R.critic(res, dec["type"], f"t{k:02d}_recheck", decks=res.decks, qa=q_new, declared_gap=R.declared_gap(dec))
        row.update({
            "qa_saved": {"verdict": tr["qa"]["verdict"], "reasons": tr["qa"]["reasons"], "schema": "musicmix.dj_qa.4 (set.json)"},
            "qa_legacy": _qa_short(q_old),
            "qa_new": _qa_short(q_new),
            "critic_v4_saved": _tech((tr.get("critic") or {}).get("v4")),
            "critic_v4_new": _tech(crt.get("v4")),
            "critic_runner_veto_saved": (tr.get("critic") or {}).get("veto"),
            "critic_runner_veto_new": crt.get("veto"),
        })
        row["qa_verdict_changed"] = row["qa_new"]["verdict"] != row["qa_saved"]["verdict"]
        if k in prev:
            row["qa_previous"] = {"version": prev_version, "verdict": prev[k]["qa_new"]["verdict"], "reasons": prev[k]["qa_new"]["reasons"]}
            row["qa_verdict_changed_vs_previous"] = row["qa_new"]["verdict"] != prev[k]["qa_new"]["verdict"]
        row["critic_v4_status_changed"] = (row["critic_v4_new"] or {}).get("status") != (row["critic_v4_saved"] or {}).get("status")
        rows.append(row)
        print(f"T{k:02d} {tr['technique']:18s} pcm={'=' if row['pcm_identical'] else 'DIFF'} QA {row['qa_saved']['verdict']} "
              f"(prev {(row.get('qa_previous') or {}).get('verdict')}) -> "
              f"{row['qa_new']['verdict']} (legacy {row['qa_legacy']['verdict']}); v4 {(row['critic_v4_saved'] or {}).get('status')} -> "
              f"{(row['critic_v4_new'] or {}).get('status')}  [{time.time() - t0:.0f} s]", flush=True)
    ok = [r for r in rows if "error" not in r]
    summary = {
        "transitions": len(rows), "errors": len(rows) - len(ok),
        "pcm_identical": sum(1 for r in ok if r["pcm_identical"]),
        "qa_saved": _count(r["qa_saved"]["verdict"] for r in ok),
        "qa_legacy": _count(r["qa_legacy"]["verdict"] for r in ok),
        "qa_new": _count(r["qa_new"]["verdict"] for r in ok),
        "qa_verdicts_changed": [r["index"] for r in ok if r["qa_verdict_changed"]],
        "qa_previous": _count(r["qa_previous"]["verdict"] for r in ok if "qa_previous" in r),
        "qa_verdicts_changed_vs_previous": [r["index"] for r in ok if r.get("qa_verdict_changed_vs_previous")],
        "verse_cut_new": _count(r["qa_new"]["verse_cut"].get("status") for r in ok),
        "critic_v4_saved": _count((r["critic_v4_saved"] or {}).get("status") for r in ok),
        "critic_v4_new": _count((r["critic_v4_new"] or {}).get("status") for r in ok),
        "critic_v4_status_changed": [r["index"] for r in ok if r["critic_v4_status_changed"]],
    }
    out = {"schema": "musicmix.qa_recheck.1", "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "run": "runs/dj_set_v3_bridge (frozen: set.json and transitions/ unchanged)", "script": "tools/recheck_set_v3_qa.py",
           "versions": {"dj_qa": dj_qa.DJ_QA_VERSION, "dj_critic_books": dj_critic_books.DJ_CRITIC_BOOKS_VERSION,
                        "dj_render": dj.DJ_RENDER_VERSION, "tracker": tracker.info()},
           "qa_config_new": dataclasses.asdict(R.QA_CFG), "qa_config_legacy": dataclasses.asdict(legacy_cfg),
           "note": "integrità tecnica, non qualità musicale; nessuna transizione riascoltata",
           "summary": summary, "transitions": rows}
    OUT.write_text(json.dumps(dj_qa._jsonable(out), indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print(OUT)
    return 0


def _count(it) -> dict:
    c: dict = {}
    for v in it:
        c[str(v)] = c.get(str(v), 0) + 1
    return c


if __name__ == "__main__":
    raise SystemExit(main())
