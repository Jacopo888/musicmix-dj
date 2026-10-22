#!/usr/bin/env python3
"""Misura il contratto di time-stretch per ogni backend disponibile (P0.1, RENDER_TRIAGE.md, 21/10/2026).

Per ciascun backend di ``dj_render.STRETCH_BACKENDS`` presente sulla macchina (CLI Rubber Band R3, ffmpeg R3
compensato, vecchio ffmpeg R2 ``crisp``, R2 ``smooth``, ``atempo``) misura su segnali sintetici, con stretch
+-3/7/12 % e 1,05 (il caso del triage): frequenza di picco di un tono di 82 Hz e 440 Hz, scarto di durata, residuo
> 1 kHz sul tono (clic di giunzione), e su un treno di clic a 3 kHz con basso a 82 Hz: clic mancanti, errore di
posizione (uno-a-uno), allargamento e livello dei transienti. Il verdetto usa ``dj_render.STRETCH_CONTRACT``
(82 Hz +-0,5 Hz è il contratto dei test; le altre soglie sono [H]). Scrive
``runs/stretch_backend_qualification/qualification_<host>.json``.

    python tools/qualify_stretch_backends.py [--sr 48000 44100]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicmix import dj_render as dj  # noqa: E402


def _version(cmd: list[str]) -> str | None:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return ((p.stdout or "") + (p.stderr or "")).strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sr", type=int, nargs="+", default=[48000, 44100])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ratios = tuple(sorted(set(dj.STRETCH_TIME_RATIOS) | {1.05}))
    env = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "rubberband_cli": shutil.which("rubberband"),
        "rubberband_cli_version": _version(["rubberband", "--version"]) if shutil.which("rubberband") else None,
        "ffmpeg": shutil.which("ffmpeg"),
        "ffmpeg_version": _version(["ffmpeg", "-version"]) if shutil.which("ffmpeg") else None,
    }
    results = []
    for sr in args.sr:
        for name in dj.STRETCH_BACKENDS:
            desc = dj.stretch_backend_info(name, qualify=False) if _available(name) else None
            if desc is None:
                results.append({"backend": name, "sample_rate": sr, "available": False})
                continue
            t0 = time.time()
            m = dj.measure_stretch_backend(desc, sr, ratios)
            m["seconds"] = round(time.time() - t0, 1)
            m["available"] = True
            m["used_by_auto"] = None
            results.append(m)
            print(f"{sr} {name:18s} passed={m['passed']} failed={m['summary']['failed']} ({m['seconds']} s)", flush=True)
    for sr in args.sr:
        try:
            chosen = dj.stretch_backend_info("auto", sample_rate=sr)["variant"]
        except dj.DJRenderError as exc:
            chosen = f"refused: {exc}"
        for r in results:
            if r.get("sample_rate") == sr and r.get("available"):
                r["used_by_auto"] = r["backend"]["variant"] == chosen
    table = []
    for r in results:
        if not r.get("available"):
            continue
        for row in r["rows"]:
            table.append({
                "backend": r["backend"]["variant"], "sample_rate": r["sample_rate"], "time_ratio": row["time_ratio"],
                "tone_82_hz": _round(row.get("tone_82_hz"), 3), "tone_440_hz": _round(row.get("tone_440_hz"), 3),
                "duration_err_samples": row.get("tone_82_duration_err_samples"),
                "click_timing_max_abs_ms": _round(row.get("click_timing_max_abs_ms"), 2),
                "click_width_ratio": _round(row.get("click_width_ratio"), 2), "click_level_db": _round(row.get("click_level_db"), 2),
                "clicks_missing": row.get("clicks_missing"), "passed": row["passed"], "failed": row["failed"],
            })
    out = {
        "schema": "musicmix.stretch_backend_qualification.1",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dj_render_version": dj.DJ_RENDER_VERSION,
        "environment": env,
        "time_ratios": list(ratios),
        "contract": dj.STRETCH_CONTRACT,
        "labels": "82 Hz +-0.5 Hz = contract of tests/test_dj_render.py; other thresholds [H]; synthetic signals only (not musical quality)",
        "table": table,
        "results": results,
    }
    dest = Path(args.out) if args.out else ROOT / "runs" / "stretch_backend_qualification" / f"qualification_{platform.node() or 'host'}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(dest)
    return 0


def _available(name: str) -> bool:
    try:
        return dj.stretch_backend_info(name, qualify=False) is not None
    except dj.DJRenderError:
        return False


def _round(v, n):
    return None if v is None else round(float(v), n)


if __name__ == "__main__":
    raise SystemExit(main())
