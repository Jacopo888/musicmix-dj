#!/usr/bin/env python3
"""Rifa la fase vocals (Demucs htdemucs + VAD + ASR, lingua automatica) di un brano V12 del ponte.

Serve per ``reminder`` e ``too_many_nights``: i JSON vocals congelati (``runs/vocals_v2``) puntano a stem
Windows assenti sulla macchina di analisi e il vecchio ``a_vocals.wav`` di too_many_nights era la voce di reminder. Il
risultato va in una cartella NUOVA (default ``runs/pool_bridge_v1/vocals_redo/<slug>/``); i JSON congelati
non si toccano. Stesso codice dei brani del ponte (``analyze_vocals.analyse_one``, stesso modello ASR,
stesso runtime Demucs, ``MUSICMIX_DEMUCS_STEMS_DIR`` opzionale per gli stem a 4 sorgenti).

Un brano alla volta::

    python tools/redo_vocals_pair.py reminder --threads 2
    python tools/redo_vocals_pair.py too_many_nights --threads 2

Salva anche ``evidence.json`` (sha della sorgente in ``local_audio/``, del WAV canonico, dello stem,
modello e checkpoint). Si ferma se i GB liberi o la RAM disponibile sono sotto soglia.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from musicmix.paths import localize, torch_home  # noqa: E402

MIN_FREE_GB = 2.0
MIN_AVAIL_GB = 5.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def avail_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576
    return 0.0


def guard(where: str) -> None:
    free = shutil.disk_usage(ROOT).free / 1e9
    avail = avail_gb()
    print(f"[{time.strftime('%H:%M:%S')}] {where}: disco libero {free:.2f} GB, RAM disponibile {avail:.1f} GB", flush=True)
    if free < MIN_FREE_GB:
        raise SystemExit(f"disco libero {free:.2f} GB < {MIN_FREE_GB} GB: mi fermo")
    if avail < MIN_AVAIL_GB:
        raise SystemExit(f"RAM disponibile {avail:.1f} GB < {MIN_AVAIL_GB} GB: Demucs non parte")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("slug", choices=("reminder", "too_many_nights"))
    ap.add_argument("--catalog", type=Path, default=ROOT / "runs/pool_bridge_v1/catalog_ready.json")
    ap.add_argument("--out-root", type=Path, default=ROOT / "runs/pool_bridge_v1/vocals_redo")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--stems-dir", type=Path, default=None,
                    help="se dato: MUSICMIX_DEMUCS_STEMS_DIR (4 stem htdemucs condivisi, ~160 MB/brano)")
    ap.add_argument("--language", default="auto")
    args = ap.parse_args(argv)

    os.environ.setdefault("TORCH_HOME", str(torch_home()))
    if args.stems_dir:
        os.environ["MUSICMIX_DEMUCS_STEMS_DIR"] = str(args.stems_dir)
    import analyze_pool_v10 as providers  # noqa: E402  (stesso runtime/modello ASR del ponte)
    from analyze_vocals import analyse_one  # noqa: E402

    entry = next(t for t in json.loads(args.catalog.read_text(encoding="utf-8"))["tracks"] if t["slug"] == args.slug)
    canonical = localize(entry["canonical_path"]).resolve(strict=True)
    source = localize(entry["source_path"]).resolve(strict=True)
    if sha256(canonical) != entry["track_id"]:
        raise SystemExit("audio canonico diverso dal catalogo")
    src_sha = sha256(source)
    if src_sha != entry["source_sha256"]:
        raise SystemExit("sorgente in local_audio diversa dal catalogo")
    out_dir = (args.out_root if args.out_root.is_absolute() else ROOT / args.out_root) / args.slug
    if (out_dir / "a.json").exists():
        raise SystemExit(f"{out_dir / 'a.json'} esiste gia': non ripeto l'inferenza")
    guard("prima di Demucs")
    t0 = time.time()
    doc = analyse_one("a", canonical, out_dir, model_path=providers._find_whisper_model(),
                      runtime_root=providers.VOCAL_RUNTIME, threads=args.threads, skip_separation=False,
                      language=args.language)
    guard("dopo la fase vocals")
    sep, be = doc.get("separator") or {}, doc.get("backend") or {}
    stem = out_dir / "a_vocals.wav"
    evidence = {
        "schema_version": "musicmix.vocals_redo.1", "slug": args.slug, "reason": "stem vocale assente sulla macchina di analisi (JSON V12 congelati "
        "puntano a percorsi Windows); il vecchio a_vocals.wav di too_many_nights era la voce di reminder",
        "source_path": str(source), "source_sha256": src_sha,
        "canonical_path": str(canonical), "canonical_sha256": entry["track_id"],
        "vocals_json": str(out_dir / "a.json"), "vocals_json_sha256": sha256(out_dir / "a.json"),
        "stem_path": str(stem), "stem_sha256": sha256(stem), "separator_output_sha256": sep.get("output_sha256"),
        "separator_status": sep.get("status"), "separator_model": sep.get("model"),
        "separator_checkpoint_files": sep.get("checkpoint_files"), "separator_input_sha256": sep.get("input_sha256"),
        "shared_stems": sep.get("shared_stems"),
        "asr_backend": {k: be.get(k) for k in ("backend", "model_path", "model_sha256", "compute_type", "language",
                                               "detected_language", "language_probability", "language_source", "status")},
        "language_selection": doc.get("language_selection"),
        "n_words": len(doc.get("words") or []), "analysis_status": doc.get("analysis_status"),
        "runner_sha256": sha256(Path(__file__)), "wall_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: evidence[k] for k in ("slug", "stem_sha256", "separator_status", "n_words", "wall_seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
