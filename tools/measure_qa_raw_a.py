#!/usr/bin/env python3
"""Quanti verdetti del QA cambierebbero misurando il ritmo su A grezzo (``QAConfig(rhythm_on_raw_a=True)``).

``dj_render`` restituisce il deck A già equalizzato: ``dj_qa.check_rhythm`` traccia A dopo l'EQ (nota dei libri,
runs/books_integration_check). Da oggi ``dj_render`` passa anche ``decks["a_raw"]`` (additivo, marcato
``a_raw_opt_in``) e il QA lo usa solo con ``rhythm_on_raw_a=True``. Questo script rifà i **blend di dj_render**
già pubblicati (gli unici toccati dall'opzione: cut senza sovrapposizione, echo/filter/creative hanno deck propri)
con gli stessi brani, cue, griglie e configurazioni delle run, e confronta per ciascuno il QA di default con
quello su A grezzo (stesso tracker, stessi altri controlli). Il default non cambia: decide l'orchestratore.

Run coperte (se presenti): runs/dj_gallery_v2 (clips + rejected), runs/dj_set_v1b, runs/dj_set_v2_books e
runs/books_integration_check (blend 4 default e con le opzioni EQ dei libri). Scrive
``runs/dj_critic_v4/qa_raw_a.json``. Riproducibilità: per ogni clip si confronta lo sha256 del PCM rifatto e il
verdetto di default con quelli salvati nella run.

    python tools/measure_qa_raw_a.py
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from musicmix import dj_qa  # noqa: E402
from musicmix import dj_render as dj  # noqa: E402

OUT = ROOT / "runs/dj_critic_v4/qa_raw_a.json"


def pcm_sha(res) -> str:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, res.audio, res.sample_rate, subtype="PCM_24", format="WAV")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def qa_pair(res, A, B, decks, tracker, base_cfg: dj_qa.QAConfig, grid_support: bool) -> dict:
    n = len(res.audio)
    meta = res.metadata
    va, vb = dj_qa.stems_to_output(meta, n, A.vocals, B.vocals)
    words = dj_qa.words_to_output(meta, A.words, n)
    kw = {"grid_support": {"a": A.analysis["tempo"], "b": B.analysis["tempo"]}} if grid_support else {}
    out = {}
    for name, cfg in (("default", base_cfg), ("raw_a", dj_qa.QAConfig(**{**base_cfg.__dict__, "rhythm_on_raw_a": True}))):
        q = dj_qa.evaluate_render(res.audio.astype(np.float64), meta, sample_rate=res.sample_rate, decks=decks, vocals_a=va,
                                  vocals_b=vb, vocal_thresholds=(A.vocal_threshold, B.vocal_threshold), words_a=words,
                                  tracker=tracker, config=cfg, **kw)
        rh = q["checks"]["rhythm"]
        bt = (rh.get("methods") or {}).get("beat_this_decks") or {}
        out[name] = {"verdict": q["verdict"], "rhythm": rh["status"], "rhythm_reason": rh.get("reason"),
                     "downbeat_median_abs_ms": bt.get("downbeat_median_abs_ms"), "beat_median_abs_ms": bt.get("beat_median_abs_ms"),
                     "reasons": [f"{r['check']}: {r['message']}" for r in q["reasons"]]}
    return out


def gallery_v2(rows: list, tracker):
    import run_dj_gallery_v2 as g2

    run = ROOT / "runs/dj_gallery_v2"
    if not run.is_dir():
        return
    plan_dir = ROOT / "runs/dj_plan_v1"
    pairs_doc = g2.v1.load_json(plan_dir / "pairs.json")
    cat = g2.v1.localize(pairs_doc["catalog"])
    catalog = g2.v1.load_json(cat if cat.is_absolute() else ROOT / cat)
    tracks: dict = {}
    for sub in ("clips", "rejected"):
        for p in sorted((run / sub).glob("*.json")):
            d = json.loads(p.read_text())
            gal, r = d["gallery"], d["render"]
            if gal["kind"] != "blend":
                continue
            pr = r["provenance"]
            sa, sb = pr["a"]["slug"], pr["b"]["slug"]
            for s in (sa, sb):
                if s not in tracks:
                    tracks[s] = g2.Track(s, plan_dir, catalog)
            A, B = tracks[sa], tracks[sb]
            v = {"kind": "blend", "bars": gal["params"]["bars"], "a_out_s": gal["params"]["cue_out_a_s"]}
            plan = {"a_out_s": gal["params"]["cue_out_a_s"], "b_in_s": gal["params"]["cue_in_b_s"]}
            prov = {k: pr[k] for k in pr}
            t0 = time.time()
            res, _params, extra = g2.render_version(v, A, B, plan, prov)
            q = qa_pair(res, A, B, extra["decks"], tracker, dj_qa.QAConfig(), True)
            rows.append({"run": "dj_gallery_v2", "id": p.stem, "published": sub == "clips",
                         "stored_verdict": d["qa"]["verdict"], "stored_rhythm": d["qa"]["checks"]["rhythm"]["status"],
                         "pcm_identical": pcm_sha(res) == r["output"]["wav_sha256"], **q, "seconds": time.time() - t0})
            print(rows[-1]["run"], p.stem, rows[-1]["stored_verdict"], "->", q["default"]["verdict"], "/", q["raw_a"]["verdict"],
                  "pcm", rows[-1]["pcm_identical"], flush=True)


def sets(rows: list, tracker):
    import run_dj_set_v1 as base

    log: list = []
    sources = base.discover(log)
    for run, extra_cfgs in (("dj_set_v1b", {"default": {}}),
                            ("dj_set_v2_books", {"default": {}, "books_eq": {"bass_gap_beats": 2.0, "mid_duck_b_bars": 2}})):
        sp = ROOT / "runs" / run / "set.json"
        if not sp.exists():
            continue
        doc = json.loads(sp.read_text())
        for t in doc["transitions"]:
            if not t["type"].startswith("blend"):
                continue
            if t["a"] not in sources or t["b"] not in sources:
                rows.append({"run": run, "id": f"t{t['index']:02d}_{t['type']}", "error": "sorgenti non trovate sulla macchina di analisi"})
                continue
            src_a, src_b = sources[t["a"]], sources[t["b"]]
            A = base.v1.Track(t["a"], src_a["plan_dir"], src_a["catalog"])
            B = base.v1.Track(t["b"], src_b["plan_dir"], src_b["catalog"])
            plan = t["plan"]
            stored = sorted((ROOT / "runs" / run / "transitions").glob(f"t{t['index']:02d}_*{t['type']}.json"))
            stored_d = json.loads(stored[-1].read_text()) if stored else {}
            for name, extra in extra_cfgs.items():
                cfg = dj.DJRenderConfig(**{**base.BLEND_CFG.__dict__, **extra})
                grids = dict(beats_a=A.beats, downbeats_a=A.downbeats, beats_b=B.beats, downbeats_b=B.downbeats)
                ia = int(np.argmin(np.abs(A.downbeats - plan["a_out_s"])))
                t0 = time.time()
                res = dj.render_transition(A.audio, B.audio, **grids, cue_out_a=float(A.downbeats[ia]), cue_in_b=float(plan["b_in_s"]),
                                           kind="blend", bars=int(t["type"][5:]), sample_rate=base.SR, vocals_a=A.vocals,
                                           vocals_b=B.vocals, config=cfg, return_decks=True)
                for qname, qcfg in (("set_qa", dj_qa.QAConfig()), ("books_check_qa", dj_qa.QAConfig(dropout_check=True))):
                    if run == "dj_set_v1b" and qname == "books_check_qa":
                        continue
                    if name == "books_eq" and qname == "set_qa":
                        continue
                    q = qa_pair(res, A, B, res.decks, tracker, qcfg, qname == "set_qa" and base._QA_HAS_GRID)
                    rid = f"t{t['index']:02d}_{t['type']}" + ("" if name == "default" else f"_{name}")
                    row = {"run": run if qname == "set_qa" else "books_integration_check", "id": rid, "render": name, "qa_config": qname,
                           **q, "seconds": time.time() - t0}
                    if qname == "set_qa":
                        row["stored_verdict"] = (stored_d.get("qa") or {}).get("verdict")
                        row["pcm_identical"] = pcm_sha(res) == ((stored_d.get("render") or {}).get("output") or {}).get("wav_sha256")
                    rows.append(row)
                    print(row["run"], rid, row.get("stored_verdict"), "->", q["default"]["verdict"], "/", q["raw_a"]["verdict"], flush=True)
            del A, B


def main() -> int:
    tracker = dj_qa.BeatThisTracker()
    rows: list = []
    t0 = time.time()
    for fn in (gallery_v2, sets):
        try:
            fn(rows, tracker)
        except Exception as exc:  # registrato, non nascosto
            rows.append({"run": fn.__name__, "error": f"{type(exc).__name__}: {exc}"})
            print("ERRORE", fn.__name__, exc, flush=True)
    ok = [r for r in rows if "default" in r]
    changed = [r for r in ok if r["default"]["verdict"] != r["raw_a"]["verdict"]]
    rchanged = [r for r in ok if r["default"]["rhythm"] != r["raw_a"]["rhythm"]]
    summary = {"n_blends": len(ok), "verdict_changes": len(changed), "rhythm_status_changes": len(rchanged),
               "changes": [{"run": r["run"], "id": r["id"], "default": r["default"]["verdict"], "raw_a": r["raw_a"]["verdict"],
                            "rhythm": [r["default"]["rhythm"], r["raw_a"]["rhythm"]]} for r in rchanged],
               "reproduced_default_equals_stored": sum(1 for r in ok if r.get("stored_verdict") and r["stored_verdict"] == r["default"]["verdict"]),
               "with_stored_verdict": sum(1 for r in ok if r.get("stored_verdict")),
               "pcm_identical": sum(1 for r in ok if r.get("pcm_identical")), "errors": [r for r in rows if "error" in r]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"schema": "musicmix.qa_raw_a_check.1", "qa_version": dj_qa.DJ_QA_VERSION,
                               "note": "solo blend di dj_render; QA tecnico, nessun ascolto; il default di QAConfig non cambia",
                               "summary": summary, "rows": rows, "elapsed_s": time.time() - t0},
                              indent=1, ensure_ascii=False, default=str))
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
