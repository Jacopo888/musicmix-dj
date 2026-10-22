#!/usr/bin/env python3
"""Analisi dj_select dei brani CC gia' completi, in una cartella di dj_phrase (non tocca runs/dj_plan_cc_v1).

L'analisi del pool CC (``tools/run_cc_pool_analysis.py``) scrive ``runs/dj_plan_cc_v1/tracks`` solo alla
fine. Per i brani le cui tre fasi sono gia' complete (struttura Beat This!, voce Demucs+ASR, provider
allin1/madmom/essentia ``status == complete``) questa utility chiama la stessa funzione
``tools/dj_plan_pool.analyse_track`` e scrive in ``runs/dj_phrase_v1/plan_cc/tracks/<slug>.json``.
Legge solo file completi; i brani gia' presenti nella cartella ufficiale sono saltati.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import dj_plan_pool as dpp  # noqa: E402

POOL = ROOT / "runs/pool_cc_v1"
PROV = ROOT / "runs/dj_plan_cc_v1/providers"
OFFICIAL = ROOT / "runs/dj_plan_cc_v1/tracks"
OUT = ROOT / "runs/dj_phrase_v1/plan_cc/tracks"


def complete(slug: str) -> bool:
    if not ((POOL / "structure" / slug / "a.json").is_file() and (POOL / "vocals" / slug / "a.json").is_file()):
        return False
    p = PROV / f"{slug}.json"
    if not p.is_file():
        return False
    try:
        rec = json.loads(p.read_text())
    except json.JSONDecodeError:
        return False  # being written
    return all(rec.get(k, {}).get("status") == "complete" for k in ("allin1", "madmom_key", "essentia_key"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cat = json.loads((POOL / "catalog_initial.json").read_text())
    done = []
    for t in cat["tracks"]:
        slug = t["slug"]
        if (OFFICIAL / f"{slug}.json").is_file() or (OUT / f"{slug}.json").is_file() or not complete(slug):
            continue
        try:
            a = dpp.analyse_track(t, PROV)
        except Exception as exc:  # noqa: BLE001
            print(f"{slug}: errore {exc!r}", flush=True)
            continue
        a["inputs"]["note"] = "computed by tools/dj_phrase_plan_cc.py with tools/dj_plan_pool.analyse_track (same code)"
        (OUT / f"{slug}.json").write_text(dpp.dump(a), encoding="utf-8")
        done.append(slug)
        print(f"{slug}: ok", flush=True)
    print(json.dumps({"new": done}))


if __name__ == "__main__":
    main()
