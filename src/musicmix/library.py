"""Import the user's own audio files into a MusicMix pool ("bring your own library").

``musicmix ingest <folder>`` (or ``python tools/ingest_library.py <folder>``) scans a local folder,
never downloads anything, and for every audio file it:

1. hashes the original bytes (SHA-256) and reads container metadata with ``ffprobe``;
2. skips duplicates (same bytes already in the pool) and rejects files that are not plausible
   songs (no audio stream, unreadable, shorter/longer than the configured bounds);
3. decodes a canonical 48 kHz stereo PCM WAV with :func:`musicmix.ingest.decode_audio`
   (the canonical hash becomes the ``track_id``);
4. records provenance (original path, size, modification time, hashes, decoder, import time,
   ``"downloaded": false``) in ``<pool>/sources.json``;
5. writes ``<pool>/catalog_initial.json`` in the pool-inventory format used by the rest of the
   pipeline (``musicmix.pool-inventory.1``: same fields as ``catalog_ready.json``), with
   ``structure``/``vocals`` still pending.

The existing analysis tools then fill the pending evidence: ``tools/run_cc_pool_analysis.py``
(Beat This!, Demucs + ASR, All-In-One/madmom/essentia), ``tools/prepare_cc_pool.py refresh``
(``catalog_ready.json``) and ``tools/dj_plan_pool.py`` (per-track DJ analysis and pair plan);
:func:`analysis_commands` lists them and ``ingest --analyze`` runs them.

Only import music you have the rights to use. Original files are never modified; by default
they are referenced in place (the analysis re-checks their hash), ``copy_sources=True`` copies
them into ``<pool>/sources/`` first.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import unicodedata
import wave
from typing import Any, Iterable, Optional

from .ingest import decode_audio, sha256_file
from .paths import data_root

AUDIO_EXTENSIONS = frozenset({
    ".aac", ".aif", ".aifc", ".aiff", ".alac", ".ape", ".flac", ".m4a", ".mp3", ".mp4", ".oga", ".ogg",
    ".opus", ".wav", ".wave", ".webm", ".wma", ".wv",
})
INVENTORY_SCHEMA = "musicmix.pool-inventory.1"
SOURCES_SCHEMA = "musicmix.library-sources.1"
DEFAULT_MIN_SECONDS = 30.0
DEFAULT_MAX_SECONDS = 1200.0


def default_pool() -> Path:
    return data_root() / "runs" / "library"


def scan(folder: Path, recursive: bool = True) -> list[Path]:
    """Audio files under ``folder`` (by extension), sorted, skipping hidden files and folders."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"not a folder: {folder}")
    walker = folder.rglob("*") if recursive else folder.glob("*")
    found = []
    for path in walker:
        rel = path.relative_to(folder)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            found.append(path)
    return sorted(found, key=lambda p: str(p.relative_to(folder)).lower())


def probe(path: Path) -> dict[str, Any]:
    """Container/stream metadata from ``ffprobe`` (raises ``RuntimeError`` if it cannot read the file)."""
    exe = shutil.which("ffprobe")
    if not exe:
        raise RuntimeError("ffprobe (FFmpeg) is required to import audio; install ffmpeg and add it to PATH")
    out = subprocess.run([exe, "-v", "error", "-show_entries",
                          "format=duration,bit_rate,format_name:format_tags:stream=codec_type,codec_name,sample_rate,channels",
                          "-of", "json", str(path)], capture_output=True, text=True, timeout=60)
    if out.returncode:
        raise RuntimeError(f"ffprobe could not read the file: {out.stderr.strip()[-300:]}")
    doc = json.loads(out.stdout or "{}")
    fmt = doc.get("format") or {}
    audio = next((s for s in doc.get("streams", []) if s.get("codec_type") == "audio"), None)
    tags = {str(k).lower(): str(v).strip() for k, v in (fmt.get("tags") or {}).items()}
    try:
        duration = float(fmt.get("duration")) if fmt.get("duration") not in (None, "N/A") else None
    except ValueError:
        duration = None
    return {
        "has_audio": audio is not None, "duration_seconds": duration, "format": fmt.get("format_name"),
        "bit_rate": int(fmt["bit_rate"]) if str(fmt.get("bit_rate", "")).isdigit() else None,
        "codec": (audio or {}).get("codec_name"),
        "sample_rate": int((audio or {}).get("sample_rate") or 0) or None,
        "channels": (audio or {}).get("channels"),
        "tags": {k: tags[k] for k in ("title", "artist", "album_artist", "album", "date", "genre", "track") if tags.get(k)},
    }


def slugify(text: str, fallback: str = "track") -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return slug[:60].rstrip("_") or fallback


_TRACK_NO = re.compile(r"^\s*(?:\d{1,3}|[A-Da-d]\d{1,2})\s*[-_.)]?\s+")


def guess_artist_title(path: Path, tags: dict[str, str]) -> tuple[str, str]:
    """Artist and title from tags, else from an ``Artist - Title`` file name, else the bare file name."""
    artist = tags.get("artist") or tags.get("album_artist") or ""
    title = tags.get("title") or ""
    if not title:
        stem = _TRACK_NO.sub("", path.stem.replace("_", " ")).strip()
        parts = [p.strip() for p in re.split(r"\s+[-–—]\s+", stem) if p.strip()]
        if len(parts) >= 2 and not artist:
            artist, title = parts[0], " - ".join(parts[1:])
        else:
            title = parts[-1] if parts else path.stem
    return artist, title


def _wav_info(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as w:
        return w.getframerate(), w.getnframes()


def _load(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def catalog_entry(pool: Path, slug: str, artist: str, title: str, source: Path, canonical: Path) -> dict[str, Any]:
    """Pool-inventory entry (same fields as ``tools/prepare_cc_pool.canonical_entry``)."""
    rate, frames = _wav_info(canonical)
    return {
        "slug": slug, "artist": artist, "title": title, "source_path": str(source.resolve()),
        "canonical_path": str(canonical.resolve()),
        "structure_path": str((pool / "structure" / slug / "a.json").resolve()),
        "vocals_path": str((pool / "vocals" / slug / "a.json").resolve()),
        "status": "canonical_ready_evidence_pending", "missing": ["structure", "vocals"],
        "source_sha256": sha256_file(source), "track_id": sha256_file(canonical), "sample_rate": rate,
        "duration_samples": frames, "duration_seconds": frames / rate,
    }


def ingest_library(folder: Path, pool: Optional[Path] = None, *, recursive: bool = True,
                   min_seconds: float = DEFAULT_MIN_SECONDS, max_seconds: float = DEFAULT_MAX_SECONDS,
                   copy_sources: bool = False, dry_run: bool = False, limit: Optional[int] = None,
                   log=print) -> dict[str, Any]:
    """Import every audio file under ``folder`` into ``pool``; idempotent (already imported bytes are skipped).

    Returns a summary with ``added``, ``duplicates``, ``rejected`` and the catalog path.
    """
    folder = Path(folder).resolve()
    pool = Path(pool or default_pool()).resolve()
    sources_path, catalog_path = pool / "sources.json", pool / "catalog_initial.json"
    sources = _load(sources_path, {"schema_version": SOURCES_SCHEMA, "tracks": [], "rejected": []})
    catalog = _load(catalog_path, {"schema_version": INVENTORY_SCHEMA, "tracks": []})
    by_sha = {s["source_sha256"]: s for s in sources["tracks"]}
    by_track = {t["track_id"]: t for t in catalog["tracks"]}
    slugs = {t["slug"] for t in catalog["tracks"]}
    summary: dict[str, Any] = {"folder": str(folder), "pool": str(pool), "scanned": 0, "added": [],
                               "duplicates": [], "rejected": [], "dry_run": dry_run}
    files = scan(folder, recursive)
    for path in files[:limit] if limit else files:
        summary["scanned"] += 1
        rel = str(path.relative_to(folder))
        stat = path.stat()
        digest = sha256_file(path)
        if digest in by_sha:
            summary["duplicates"].append({"file": rel, "slug": by_sha[digest]["slug"]})
            continue
        reason, info = None, None
        try:
            info = probe(path)
        except (RuntimeError, ValueError, subprocess.SubprocessError) as error:
            if "required" in str(error):
                raise
            reason = str(error)
        if info is not None:
            dur = info["duration_seconds"]
            if not info["has_audio"]:
                reason = "no audio stream"
            elif dur is None:
                reason = "unknown duration"
            elif not min_seconds <= dur <= max_seconds:
                reason = f"duration {dur:.1f}s outside [{min_seconds:g}, {max_seconds:g}] s"
        if reason:
            rec = {"file": rel, "source_path": str(path), "source_sha256": digest, "reason": reason, "at": _now()}
            summary["rejected"].append(rec)
            if not dry_run:
                sources["rejected"] = [r for r in sources.get("rejected", []) if r.get("source_sha256") != digest] + [rec]
            log(f"rejected {rel}: {reason}")
            continue
        artist, title = guess_artist_title(path, info["tags"])
        base = slugify(f"{artist} {title}" if artist else title, fallback=slugify(path.stem))
        slug, n = base, 2
        while slug in slugs:
            slug, n = f"{base}_{n}", n + 1
        slugs.add(slug)
        if dry_run:
            summary["added"].append({"file": rel, "slug": slug, "artist": artist, "title": title})
            continue
        source = path
        if copy_sources:
            source = pool / "sources" / f"{slug}{path.suffix.lower()}"
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, source)
            if sha256_file(source) != digest:
                raise RuntimeError(f"copy of {path} does not match the original hash")
        wav = pool / "audio" / f"{slug}.wav"
        decoded = decode_audio(source, wav)
        _save(wav.with_suffix(".ingest.json"), decoded)
        entry = catalog_entry(pool, slug, artist, title, source, wav)
        if entry["track_id"] in by_track:  # different file bytes, identical decoded PCM
            wav.unlink()
            wav.with_suffix(".ingest.json").unlink()
            summary["duplicates"].append({"file": rel, "slug": by_track[entry["track_id"]]["slug"], "same_pcm": True})
            slugs.discard(slug)
            continue
        album = info["tags"].get("album")
        provenance = {
            "kind": "local_file", "downloaded": False, "imported_at": _now(),
            "original_path": str(path), "library_folder": str(folder), "relative_path": rel,
            "size_bytes": stat.st_size, "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
            .replace(microsecond=0).isoformat(), "source_sha256": digest, "copied_into_pool": copy_sources,
            "probe": info, "decoder_version": decoded.get("decoder_version"),
            "offset_policy": decoded.get("offset_policy"),
        }
        entry.update(origin="local_library", **({"album": album} if album else {}), provenance=provenance)
        catalog["tracks"].append(entry)
        by_track[entry["track_id"]] = entry
        src_rec = {"slug": slug, "artist": artist, "title": title, "file": rel, "source_path": str(source),
                   "source_sha256": digest, "track_id": entry["track_id"], "provenance": provenance}
        sources["tracks"].append(src_rec)
        by_sha[digest] = src_rec
        summary["added"].append({"file": rel, "slug": slug, "artist": artist, "title": title,
                                 "duration_seconds": round(entry["duration_seconds"], 3)})
        log(f"added {slug} <- {rel}")
    if not dry_run:
        sources["updated_at"] = _now()
        _save(sources_path, sources)
        tracks = catalog["tracks"]
        catalog.update(schema_version=INVENTORY_SCHEMA, requested_count=len(tracks),
                       ready_count=sum(t.get("status") == "ready" for t in tracks),
                       status="ready" if tracks and all(t.get("status") == "ready" for t in tracks) else "incomplete",
                       config_path=str(sources_path), config_sha256=sha256_file(sources_path),
                       directed_pair_count_when_complete=len(tracks) * (len(tracks) - 1),
                       note="local library import (musicmix ingest): user-supplied files, nothing downloaded")
        catalog["tracks"] = tracks
        _save(catalog_path, catalog)
    summary["catalog"] = str(catalog_path)
    summary["total_tracks"] = len(catalog["tracks"]) + (len(summary["added"]) if dry_run else 0)
    return summary


def analysis_commands(pool: Path, *, python: str, root: Path, slugs: Optional[Iterable[str]] = None,
                      workers: int = 1, threads: int = 2, plan: Optional[Path] = None, work: Optional[Path] = None,
                      logs: Optional[Path] = None, share_stems: bool = True,
                      track_cards: Optional[Path] = None) -> list[list[str]]:
    """Commands that run the existing analyses on an imported pool, in order.

    1. ``run_cc_pool_analysis.py``: per-track structure (Beat This!), vocals (Demucs + faster-whisper) and
       providers (All-In-One, madmom, essentia), then ``prepare_cc_pool.py refresh`` and ``dj_plan_pool.py``;
    2. optionally ``build_track_cards.py build`` (persistent per-track cards) when ``track_cards`` is given.
    """
    pool = Path(pool).resolve()
    plan = Path(plan or pool / "dj_plan").resolve()
    work = Path(work or data_root() / ".cache" / f"allin1_{pool.name}").resolve()
    logs = Path(logs or pool / "logs").resolve()
    cmd = [python, str(root / "tools/run_cc_pool_analysis.py"), "--pool", str(pool), "--plan", str(plan),
           "--work", str(work), "--logs", str(logs), "--log-prefix", pool.name, "--workers", str(workers),
           "--threads", str(threads), "--no-shuffle"]
    if share_stems:
        cmd.append("--share-stems")
    slugs = list(slugs or [])
    if slugs:
        cmd += ["--slugs", *slugs]
    cmds = [cmd]
    if track_cards is not None:
        cmds.append([python, str(root / "tools/build_track_cards.py"), "build", "--compute-phrase",
                     "--catalog", str(pool / "catalog_ready.json"), "--out", str(Path(track_cards).resolve()),
                     "--analysis-dir", str(plan / "tracks"), "--spec-dir", str(work / "spec"),
                     "--demix-dir", str(work / "demix" / "htdemucs")])
    return cmds


def run_analysis(pool: Path, *, python: str, root: Path, env: Optional[dict] = None, log=print, **kwargs) -> int:
    """Run :func:`analysis_commands` in order; stop at the first failure and return its exit code."""
    for cmd in analysis_commands(pool, python=python, root=root, **kwargs):
        log("$ " + " ".join(cmd))
        rc = subprocess.run(cmd, cwd=root, env=env or dict(os.environ)).returncode
        if rc:
            log(f"exit {rc}")
            return rc
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    from .paths import python_main, repo_root

    ap = argparse.ArgumentParser(prog="musicmix ingest",
                                 description="Import your own audio files into a MusicMix pool (no downloads) "
                                             "and run the existing analyses. Use only music you have the rights to.")
    ap.add_argument("folder", type=Path, help="folder with your audio files (mp3, flac, wav, m4a, ogg, opus, ...)")
    ap.add_argument("--pool", type=Path, default=None,
                    help="pool folder (default: $MUSICMIX_DATA/runs/library)")
    ap.add_argument("--no-recursive", action="store_true", help="do not descend into subfolders")
    ap.add_argument("--min-seconds", type=float, default=DEFAULT_MIN_SECONDS)
    ap.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    ap.add_argument("--copy-sources", action="store_true",
                    help="copy originals into <pool>/sources/ (default: reference them in place)")
    ap.add_argument("--limit", type=int, default=None, help="import at most N files (first in name order)")
    ap.add_argument("--dry-run", action="store_true", help="only list what would be imported")
    ap.add_argument("--no-analyze", action="store_true",
                    help="import and catalog only; do not run Beat This!/Demucs/ASR/All-In-One/key analyses")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--track-cards", type=Path, default=None,
                    help="also build persistent track cards into this folder after the analysis")
    ap.add_argument("--print-commands", action="store_true", help="print the analysis commands and exit after import")
    args = ap.parse_args(argv)
    try:
        summary = ingest_library(args.folder, args.pool, recursive=not args.no_recursive,
                                 min_seconds=args.min_seconds, max_seconds=args.max_seconds,
                                 copy_sources=args.copy_sources, dry_run=args.dry_run, limit=args.limit)
    except (ValueError, RuntimeError, OSError) as error:
        ap.exit(2, f"musicmix ingest: {error}\n")
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in summary.items()}, indent=2))
    if args.dry_run:
        for rec in summary["added"]:
            print(f"  would add {rec['slug']}: {rec['file']}")
        return 0
    pool = Path(summary["pool"])
    kwargs = dict(python=python_main(), root=repo_root(), workers=args.workers, threads=args.threads,
                  track_cards=args.track_cards)
    if args.print_commands or args.no_analyze:
        for cmd in analysis_commands(pool, **kwargs):
            print(("next: " if args.no_analyze else "") + " ".join(cmd))
        return 0
    if not summary["total_tracks"]:
        print("no tracks to analyse")
        return 0
    return run_analysis(pool, **kwargs)
