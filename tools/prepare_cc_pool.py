#!/usr/bin/env python3
"""Catalogo canonico del pool CC (``pool_cc_v1``) nello stesso formato di ``runs/pool_v12``.

``init``     decodifica ogni mp3 in WAV 48 kHz stereo (``musicmix.ingest.decode_audio``) e scrive
             ``runs/pool_cc_v1/catalog_initial.json`` (status ``canonical_ready_evidence_pending``).
``refresh``  rilegge catalog_initial (``--pool`` per un altro pool, es. ``runs/pool_bridge_v1``), verifica che structure/vocals esistano e abbiano la stessa identita'
             (track_id, sample_rate, duration_samples) e scrive ``catalog_ready.json`` con status ``ready``
             solo per le tracce complete. Non modifica nulla di quanto e' gia' stato prodotto.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from musicmix.ingest import decode_audio, sha256_file  # noqa: E402
from musicmix.paths import localize  # noqa: E402

POOL = ROOT / "runs/pool_cc_v1"
AUDIO = ROOT / "local_audio/pool_cc_v1"


def wav_info(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as w:
        return w.getframerate(), w.getnframes()


def canonical_entry(pool: Path, slug: str, artist: str, title: str, source: Path) -> dict:
    """Decodifica ``source`` in ``<pool>/audio/<slug>.wav`` (48 kHz stereo, se manca) e restituisce la voce
    di catalogo con identita' canonica e percorsi attesi di structure/vocals (riusata da run_bridge_pool)."""
    wav = pool / "audio" / f"{slug}.wav"
    ingest = wav.with_suffix(".ingest.json")
    if not wav.exists():
        ingest.parent.mkdir(parents=True, exist_ok=True)
        ingest.write_text(json.dumps(decode_audio(source, wav), indent=2), encoding="utf-8")
    rate, frames = wav_info(wav)
    assert rate == 48000
    return {
        "slug": slug, "artist": artist, "title": title, "source_path": str(source),
        "canonical_path": str(wav.resolve()),
        "structure_path": str((pool / "structure" / slug / "a.json").resolve()),
        "vocals_path": str((pool / "vocals" / slug / "a.json").resolve()),
        "status": "canonical_ready_evidence_pending", "missing": ["structure", "vocals"],
        "source_sha256": sha256_file(source), "track_id": sha256_file(wav), "sample_rate": rate,
        "duration_samples": frames, "duration_seconds": frames / rate,
    }


def init() -> None:
    sources = json.loads((POOL / "sources.json").read_text(encoding="utf-8"))
    tracks = []
    for s in sources["tracks"]:
        mp3 = (AUDIO / s["file"]).resolve(strict=True)
        tracks.append({
            **canonical_entry(POOL, s["slug"], s["artist"], s["title"], mp3),
            "license": s["license"]["code"], "license_url": s["license"]["url"], "jamendo_url": s["jamendo_url"],
            "tags": s["tags"],
        })
    ids = [t["track_id"] for t in tracks]
    if len(set(ids)) != len(ids):
        raise SystemExit("audio canonico duplicato nel pool")
    doc = {"schema_version": "musicmix.pool-inventory.1", "requested_count": len(tracks), "ready_count": 0,
           "status": "incomplete", "config_path": str(POOL / "sources.json"),
           "config_sha256": sha256_file(POOL / "sources.json"), "tracks": tracks,
           "directed_pair_count_when_complete": len(tracks) * (len(tracks) - 1)}
    (POOL / "catalog_initial.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(tracks)} tracce in catalog_initial.json")


def refresh(pool: Path = POOL) -> None:
    cat = json.loads((pool / "catalog_initial.json").read_text(encoding="utf-8"))
    ready = 0
    for t in cat["tracks"]:
        missing = []
        for ev in ("structure", "vocals"):
            p = localize(t[ev + "_path"])
            if not p.is_file():
                missing.append(ev)
                continue
            doc = json.loads(p.read_text(encoding="utf-8"))
            if (doc.get("track_id"), doc.get("sample_rate"), doc.get("duration_samples")) != (
                    t["track_id"], t["sample_rate"], t["duration_samples"]):
                raise SystemExit(f"{t['slug']}: identita' {ev} diversa")
            if doc.get("status") in {"unavailable", "failed", "timeout", "error"} or doc.get("analysis_status") == "fallback":
                missing.append(ev + "_provider_unavailable")
            t[ev + "_sha256"] = sha256_file(p)
        t["missing"] = missing
        t["status"] = "ready" if not missing else "canonical_ready_evidence_pending"
        ready += not missing
    cat.update(ready_count=ready, status="ready" if ready == len(cat["tracks"]) else "incomplete")
    (pool / "catalog_ready.json").write_text(json.dumps(cat, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"ready {ready}/{len(cat['tracks'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["init", "refresh"])
    ap.add_argument("--pool", type=Path, default=POOL, help="cartella del pool per refresh (default: pool CC)")
    a = ap.parse_args()
    if a.cmd == "init":
        init()
    else:
        refresh(a.pool if a.pool.is_absolute() else ROOT / a.pool)
