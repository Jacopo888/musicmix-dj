#!/usr/bin/env python3
"""Analisi DJ completa e riprendibile del pool CC (Beat This!, Demucs+ASR, All-In-One, madmom, essentia).

Per ogni traccia, in sottoprocessi separati e saltando cio' che esiste gia':
  1. ``tools/analyze_pool_v12.py --stage structure``   (venv core, Beat This!)
  2. ``tools/analyze_pool_v12.py --stage vocals``      (venv core, Demucs + faster-whisper)
  3. ``tools/dj_providers.py``                         (venv-allin1: allin1, madmom, essentia)
I worker dividono le tracce a round-robin sull'ordine del catalogo (gia' mescolato dalla selezione),
cosi' ogni prefisso delle tracce completate resta vario. Al termine (o con ``--plan-only``) rigenera
``catalog_ready.json`` e lancia ``tools/dj_plan_pool.py`` sulle tracce pronte.

Ripresa dopo una pausa della macchina di analisi: rilanciare lo stesso comando (le fasi complete vengono saltate)::

    nohup python tools/run_cc_pool_analysis.py --workers 2 --threads 2 \
        > runs/cloud_session/pool_cc_v1_analysis.log 2>&1 &

Altri pool (default invariati = pool CC): ``--pool runs/pool_bridge_v1 --plan runs/dj_plan_bridge_v1
--work .cache/dj_allin1_bridge --log-prefix pool_bridge_v1 --slugs a b --no-shuffle --skip-plan``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from musicmix.paths import data_root, python_allin1, python_main, torch_home  # noqa: E402

POOL = ROOT / "runs/pool_cc_v1"
PLAN = ROOT / "runs/dj_plan_cc_v1"
CORE = python_main()
ALLIN1 = python_allin1()
LOGS = data_root() / "runs/cloud_session"
WORK = ROOT / ".cache/dj_allin1_cc"
PREFIX = "pool_cc_v1"
SHARE_STEMS = False


def run(cmd: list[str], log: Path, env: dict) -> int:
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n[{time.strftime('%H:%M:%S')}]\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        fh.write(f"[{time.strftime('%H:%M:%S')}] exit {rc}\n")
    return rc


def providers_complete(slug: str) -> bool:
    p = PLAN / "providers" / f"{slug}.json"  # PLAN/POOL/WORK/PREFIX: reimpostati da main() con le opzioni
    if not p.is_file():
        return False
    rec = json.loads(p.read_text())
    return all(rec.get(k, {}).get("status") == "complete" for k in ("allin1", "madmom_key", "essentia_key"))


def worker(slugs: list[str], threads: int, env: dict, catalog: Path) -> None:
    for slug in slugs:
        t0 = time.time()
        for stage in ("structure", "vocals"):
            if (POOL / stage / slug / "a.json").is_file():
                continue
            stage_env = dict(env, MUSICMIX_DEMUCS_STEMS_DIR=str(WORK / "demix")) if SHARE_STEMS else env
            rc = run([CORE, "tools/analyze_pool_v12.py", "--catalog", str(catalog), "--slug", slug, "--stage", stage,
                      "--pool-root", str(POOL), "--threads", str(threads)], LOGS / f"{PREFIX}_{stage}.log", stage_env)
            print(f"{time.strftime('%H:%M:%S')} {slug} {stage} exit {rc}", flush=True)
        if not providers_complete(slug):
            rc = run([ALLIN1, "tools/dj_providers.py", "--catalog", str(catalog), "--out", str(PLAN / "providers"),
                      "--work", str(WORK), "--slugs", slug, "--threads", str(threads),
                      "--skip-complete"], LOGS / f"{PREFIX}_providers.log", env)
            print(f"{time.strftime('%H:%M:%S')} {slug} providers exit {rc}", flush=True)
        if SHARE_STEMS and providers_complete(slug):  # stem non piu' necessari (~170 MB/brano, disco limitato)
            shutil.rmtree(WORK / "demix" / "htdemucs" / slug, ignore_errors=True)
        print(f"{time.strftime('%H:%M:%S')} {slug} fatto in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    global POOL, PLAN, WORK, PREFIX, SHARE_STEMS, LOGS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--plan-only", action="store_true", help="salta l'analisi, esegue solo refresh + dj_plan_pool")
    ap.add_argument("--pool", type=Path, default=POOL, help="cartella del pool (catalog_initial, structure, vocals)")
    ap.add_argument("--plan", type=Path, default=PLAN, help="cartella dell'analisi DJ (providers/, tracks/)")
    ap.add_argument("--work", type=Path, default=WORK, help="cartella di lavoro allin1 (stem demix)")
    ap.add_argument("--log-prefix", default=PREFIX, help="prefisso dei log (cartella --logs)")
    ap.add_argument("--logs", type=Path, default=LOGS, help="cartella dei log (default: runs/cloud_session)")
    ap.add_argument("--slugs", nargs="*", help="solo queste tracce (default: tutte quelle del catalogo)")
    ap.add_argument("--no-shuffle", action="store_true", help="mantiene l'ordine di --slugs/catalogo")
    ap.add_argument("--share-stems", action="store_true",
                    help="la fase vocals salva i 4 stem htdemucs in <work>/demix: allin1 non ripete la separazione")
    ap.add_argument("--skip-plan", action="store_true", help="non esegue refresh + dj_plan_pool alla fine")
    args = ap.parse_args()
    POOL, PLAN, WORK = (p if p.is_absolute() else ROOT / p for p in (args.pool, args.plan, args.work))
    PREFIX = args.log_prefix
    SHARE_STEMS = args.share_stems
    LOGS = args.logs if args.logs.is_absolute() else ROOT / args.logs
    LOGS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TORCH_HOME=str(torch_home()), OMP_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads), PYTHONHASHSEED="0")
    catalog = POOL / "catalog_initial.json"
    slugs = [t["slug"] for t in json.loads(catalog.read_text())["tracks"]]
    if args.slugs:
        unknown = set(args.slugs) - set(slugs)
        if unknown:
            raise SystemExit(f"slug non nel catalogo: {sorted(unknown)}")
        slugs = list(args.slugs)
    if not args.no_shuffle:
        random.Random(20261020).shuffle(slugs)  # ogni prefisso delle tracce completate resta vario (shard, licenze)
    if not args.plan_only:
        shards = [slugs[i::args.workers] for i in range(args.workers)]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(lambda s: worker(s, args.threads, env, catalog), shards))
    if args.skip_plan:
        return
    subprocess.run([CORE, "tools/prepare_cc_pool.py", "refresh", "--pool", str(POOL)], cwd=ROOT, check=True)
    rc = run([CORE, "tools/dj_plan_pool.py", "--catalog", str(POOL / "catalog_ready.json"), "--out", str(PLAN)],
             LOGS / f"{PREFIX}_dj_plan.log", env)
    print(f"{time.strftime('%H:%M:%S')} dj_plan_pool exit {rc}", flush=True)


if __name__ == "__main__":
    main()
