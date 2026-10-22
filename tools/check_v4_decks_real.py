#!/usr/bin/env python3
"""Canali v4 con i deck separati su audio reale (Q-1, Q-2, Q-3): blend di runs/dj_gallery_v2.

Le clip salvate non hanno i deck: questo script rifà i blend della galleria v2 (stessi brani, cue, griglie e
configurazione; PCM verificato sullo sha256 salvato) con ``return_decks=True`` e, per i brani che hanno lo stem
drums di HT-Demucs (``.cache/dj_allin1/demix/htdemucs/<slug>/drums.wav``), rifà la **stessa** transizione con gli
stem drums al posto dei mix (stesse griglie, stesso stretch: stessa mappa temporale) per avere i rullanti di A e B
in coordinate della clip. Confronta il backbeat (Q-2) misurato sugli stem drums con quello sul proxy "deck mix"
(affidabilità del proxy) e calcola ``dj_critic_books.score_v4`` con i deck. Diagnostica tecnica, nessun ascolto.

    python tools/check_v4_decks_real.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import run_dj_gallery_v2 as g2  # noqa: E402
from measure_qa_raw_a import pcm_sha  # noqa: E402
from musicmix import dj_critic as C  # noqa: E402
from musicmix import dj_critic_books as B  # noqa: E402
from musicmix import dj_qa  # noqa: E402
from musicmix import dj_render as dj  # noqa: E402

OUT = ROOT / "runs/dj_critic_v4/decks_real.json"
DRUMS = ROOT / ".cache/dj_allin1/demix/htdemucs"


def load_drums(slug: str, n: int, sr: int):
    p = DRUMS / slug / "drums.wav"
    if not p.is_file():
        return None, "stem drums assente"
    x, srd = dj.load_track(p)
    if srd != sr:
        x = C.resample(x, srd, sr)
    if abs(len(x) - n) > int(0.05 * sr):
        return None, f"durata dello stem diversa dal brano ({len(x) / sr:.2f} s vs {n / sr:.2f} s)"
    out = np.zeros((n, x.shape[1] if x.ndim > 1 else 1), dtype=np.float64)
    m = min(n, len(x))
    out[:m] = x[:m] if x.ndim > 1 else x[:m, None]
    return out, None


def main() -> int:
    run = ROOT / "runs/dj_gallery_v2"
    plan_dir = ROOT / "runs/dj_plan_v1"
    pairs_doc = g2.v1.load_json(plan_dir / "pairs.json")
    cat = g2.v1.localize(pairs_doc["catalog"])
    catalog = g2.v1.load_json(cat if cat.is_absolute() else ROOT / cat)
    tracker = dj_qa.BeatThisTracker()
    tracks: dict = {}
    rows = []
    t00 = time.time()
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
            A, Bt = tracks[sa], tracks[sb]
            t0 = time.time()
            v = {"kind": "blend", "bars": gal["params"]["bars"], "a_out_s": gal["params"]["cue_out_a_s"]}
            plan = {"a_out_s": gal["params"]["cue_out_a_s"], "b_in_s": gal["params"]["cue_in_b_s"]}
            res, _params, extra = g2.render_version(v, A, Bt, plan, dict(pr))
            sr = res.sample_rate
            decks = {k: res.decks[k] for k in ("a", "b", "a_raw", "b_stretched_raw") if k in res.decks}
            row = {"id": p.stem, "published": sub == "clips", "pcm_identical": pcm_sha(res) == r["output"]["wav_sha256"]}
            da, ea = load_drums(sa, len(A.audio), sr)
            db, eb = load_drums(sb, len(Bt.audio), sr)
            if da is not None and db is not None:
                grids = dict(beats_a=A.beats, downbeats_a=A.downbeats, beats_b=Bt.beats, downbeats_b=Bt.downbeats)
                ia = int(np.argmin(np.abs(A.downbeats - plan["a_out_s"])))
                rd = dj.render_transition(da, db, **grids, cue_out_a=float(A.downbeats[ia]), cue_in_b=float(plan["b_in_s"]),
                                          kind="blend", bars=v["bars"], sample_rate=sr, config=g2.BLEND_CFG, return_decks=True)
                same_map = (rd.metadata["overlap"] == res.metadata["overlap"] and rd.metadata["switch"]["b_source_sample"]
                            == res.metadata["switch"]["b_source_sample"])
                decks["drums_a"] = rd.decks["a_raw"]
                decks["drums_b"] = rd.decks["b_stretched_raw"]
                row["drums"] = {"status": "ok", "same_time_map": bool(same_map)}
            else:
                row["drums"] = {"status": "missing", "a": ea, "b": eb}
            # canali v4 con i deck (mono, stessa sr del render)
            meta = res.metadata
            st = meta["overlap"]["output_start_sample"] / sr
            sw = meta["switch"]["output_sample"] / sr
            en = meta["overlap"]["output_end_sample"] / sr
            mono = res.audio.mean(axis=1)
            craft = C.craft_features(C.resample(mono, sr, C.SR_CRAFT), C.SR_CRAFT, st, en,
                                     side_s=min(C.CRAFT_SIDE_S, st - 0.2, len(mono) / sr - en - 0.2))
            v4 = B.score_v4(res.audio, sr, meta, start_s=st, switch_s=sw, end_s=en, craft=craft, qa=d.get("qa"),
                            decks=decks, tracker=tracker, windows=False)
            grid = B.output_grid(meta, len(mono), sr)
            proxy = B.backbeat_alignment(decks["a_raw"], decks["b_stretched_raw"], sr, grid["beats"], grid["downbeats"], st, en,
                                         source="decks (proxy)")
            row.update({"technical": {k: v4["technical"][k] for k in ("status", "veto_flags", "warnings", "indicative_flags")},
                        "backbeat_used": v4["technical"]["backbeat"], "backbeat_proxy_decks": proxy,
                        "downbeat_after_kick": v4["technical"]["downbeat_after_kick"], "texture": v4["musical"]["texture"],
                        "metric_surprise": v4["musical"]["metric_surprise"], "render_downbeat_alignment": {
                            k: meta["downbeat_alignment"].get(k) for k in ("median_abs_error_ms", "max_abs_error_ms",
                                                                          "interdeck_median_abs_lag_ms")},
                        "stored_qa": d["qa"]["verdict"], "seconds": time.time() - t0})
            rows.append(row)
            bb = row["backbeat_used"]
            print(p.stem, "pcm", row["pcm_identical"], "drums", row["drums"]["status"], "backbeat", bb.get("status"),
                  bb.get("backbeat_median_abs_ms"), "proxy", proxy.get("status"), proxy.get("backbeat_median_abs_ms"),
                  "q3", row["downbeat_after_kick"].get("status"), flush=True)
    pairs = [(r["backbeat_used"].get("backbeat_median_abs_ms"), r["backbeat_proxy_decks"].get("backbeat_median_abs_ms"))
             for r in rows if r["drums"]["status"] == "ok"]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    summary = {"n_blends": len(rows), "with_drums": sum(r["drums"]["status"] == "ok" for r in rows),
               "backbeat_drums_vs_proxy_ms": pairs,
               "proxy_agrees_status": sum(1 for r in rows if r["drums"]["status"] == "ok"
                                          and r["backbeat_used"].get("status") == r["backbeat_proxy_decks"].get("status"))}
    OUT.write_text(json.dumps({"schema": "musicmix.dj_critic_v4_decks_real.1", "summary": summary, "rows": rows,
                               "elapsed_s": time.time() - t00}, indent=1, ensure_ascii=False, default=lambda o: o.tolist()
                              if isinstance(o, np.ndarray) else str(o)))
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
