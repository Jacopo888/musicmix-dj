#!/usr/bin/env python3
"""Verifica su audio reale delle opzioni dei libri integrate il 21/10/2026 (diagnostica tecnica, non ascolto).

Rende alcune transizioni di ``runs/dj_set_v2_books`` (stessi brani, cue e griglie del set; nessun file
sorgente modificato) con le nuove opzioni e passa ogni clip a ``dj_qa.evaluate_render`` (Beat This!,
controllo opzionale del buco nella giunta acceso):

* blend: default vs ``bass_gap_beats=2`` + ``mid_duck_b_bars=2`` (B&B cap. 28; Butler pp. 53, 246; Webber p. 181);
* ``dj_creative.halving_build`` (Margulis pp. 23, 44; Huron pp. 314-316) e ``break_before_entry``
  (Huron pp. 248-249, 314-316; Katz p. 106).

Scrive ``runs/books_integration_check/report.json`` e le clip WAV in ``clips/`` (per un ascolto successivo).
Un ``pass`` del QA è integrità tecnica, non qualità musicale (AGENTS.md).

    python tools/check_books_integration_real.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import run_dj_set_v1 as base  # noqa: E402
from musicmix import dj_creative as cr  # noqa: E402
from musicmix import dj_qa  # noqa: E402
from musicmix import dj_render as dj  # noqa: E402
from musicmix import track_filter as TF  # noqa: E402

OUT = ROOT / "runs/books_integration_check"
SET = ROOT / "runs/dj_set_v2_books/set.json"
SR = base.SR
QA_CFG = dj_qa.QAConfig(dropout_check=True)
CR_CFG = cr.DJCreativeConfig(section_gain=base.SG) if base.SG is not None else cr.DJCreativeConfig()  # as run_dj_set_v2_books
JOBS = {  # transition index in set.json -> renders
    1: ["blend_default", "blend_books_eq", "halving_build", "break_beat_mute"],
    4: ["blend_default", "blend_books_eq", "break_fill"],
    5: ["halving_build", "break_stop"],
}


def summary(qa: dict) -> dict:
    return {"verdict": qa["verdict"], "reasons": [f"{r['check']}: {r['message']}" for r in qa["reasons"]],
            "dropout": {k: qa["checks"].get("dropout", {}).get(k) for k in ("status", "longest_s", "intended")},
            "loudness_jump": {k: qa["checks"]["loudness_jump"].get(k) for k in ("status", "signed_jump_db", "worst_window")},
            "double_bass": qa["checks"]["double_bass"].get("status")}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "clips").mkdir(exist_ok=True)
    doc = json.loads(SET.read_text())
    log: list = []
    sources = base.discover(log)
    tracker = dj_qa._get_tracker("auto")
    rows = []
    t0 = time.time()
    for idx, kinds in JOBS.items():
        t = next(x for x in doc["transitions"] if x["index"] == idx)
        if not (TF.is_admitted(t["a"]) and TF.is_admitted(t["b"])):
            print(f"t{idx:02d}: saltata, brano escluso (configs/excluded_tracks.json o lingua)", flush=True)
            continue
        src_a, src_b = sources[t["a"]], sources[t["b"]]
        A = base.v1.Track(t["a"], src_a["plan_dir"], src_a["catalog"])
        B = base.v1.Track(t["b"], src_b["plan_dir"], src_b["catalog"])
        plan = t["plan"]
        ia = int(np.argmin(np.abs(A.downbeats - plan["a_out_s"])))
        grids = dict(beats_a=A.beats, downbeats_a=A.downbeats, beats_b=B.beats, downbeats_b=B.downbeats,
                     cue_out_a=float(A.downbeats[ia]), cue_in_b=float(plan["b_in_s"]), sample_rate=SR)
        for kind in kinds:
            name = f"t{idx:02d}_{kind}"
            try:
                if kind.startswith("blend"):
                    extra = {"bass_gap_beats": 2.0, "mid_duck_b_bars": 2} if kind == "blend_books_eq" else {}
                    cfg = dj.DJRenderConfig(**{**base.BLEND_CFG.__dict__, **extra})
                    nb = int(t["type"][5:]) if t["type"].startswith("blend") else 8  # same bars as the set
                    res = dj.render_transition(A.audio, B.audio, **grids, kind="blend", bars=nb, vocals_a=A.vocals,
                                               vocals_b=B.vocals, config=cfg, return_decks=True)
                    n = len(res.audio)
                    va, vb = dj_qa.stems_to_output(res.metadata, n, A.vocals, B.vocals)
                    words = dj_qa.words_to_output(res.metadata, A.words, n)
                    qa = dj_qa.evaluate_render(res.audio.astype(np.float64), res.metadata, sample_rate=SR, decks=res.decks,
                                               vocals_a=va, vocals_b=vb, vocal_thresholds=(A.vocal_threshold, B.vocal_threshold),
                                               words_a=words, tracker=tracker, config=QA_CFG)
                    extra_info = {"eq_books": res.metadata["eq"].get("books")}
                else:
                    fn = {"halving_build": (cr.halving_build, {"build_bars": 4, "start_unit": 1.0, "gap_beats": 1}),
                          "break_beat_mute": (cr.break_before_entry, {"mode": "beat_mute", "beats": 1}),
                          "break_fill": (cr.break_before_entry, {"mode": "fill", "beats": 1}),
                          "break_stop": (cr.break_before_entry, {"mode": "stop", "beats": 1})}[kind]
                    res = fn[0](A.audio, B.audio, **grids, vocals_a=A.vocals, vocals_b=B.vocals, config=CR_CFG, return_decks=True, **fn[1])
                    qi = cr.qa_inputs(res)
                    meta = res.metadata
                    ga = 10 ** (meta["loudness"]["a_gain_db"] / 20.0)
                    gb = 10 ** (meta["loudness"]["b_gain_db"] / 20.0)
                    qa = dj_qa.evaluate_render(res.audio.astype(np.float64), meta, sample_rate=SR, decks=qi["decks"],
                                               vocals_a=qi["vocals_a"], vocals_b=qi["vocals_b"],
                                               vocal_thresholds=(A.vocal_threshold * ga, B.vocal_threshold * gb),
                                               words_a=cr.words_in_clip(meta, A.words, "a"), tracker=tracker, config=QA_CFG)
                    cre = meta["creative"]
                    extra_info = {k: cre.get(k) for k in ("params", "a_vocal_at_break", "a_vocal_mute_guard", "repetition_checks", "fill")}
                dj.write_render(res, OUT / "clips", name)
                rows.append({"name": name, "a": t["a"], "b": t["b"], "set_type": t["type"], "kind": kind,
                             "duration_s": len(res.audio) / SR, "qa": summary(qa), "info": extra_info})
            except Exception as exc:  # recorded, not hidden
                rows.append({"name": name, "a": t["a"], "b": t["b"], "kind": kind, "error": f"{type(exc).__name__}: {exc}"})
            print(name, rows[-1].get("qa", {}).get("verdict"), rows[-1].get("error", ""), rows[-1].get("qa", {}).get("reasons", ""), flush=True)
        del A, B
    report = {"schema": "musicmix.books_integration_check.1", "set": str(SET.relative_to(ROOT)), "qa_config": QA_CFG.__dict__,
              "note": "QA tecnico (dj_qa); nessun ascolto. Le clip sono in clips/ per un confronto in ascolto.",
              "rows": rows, "elapsed_s": time.time() - t0}
    (OUT / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
