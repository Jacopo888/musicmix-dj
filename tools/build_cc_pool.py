#!/usr/bin/env python3
"""Pool Creative Commons hip-hop/rap/trap dal dataset MTG-Jamendo (ricerca non commerciale).

Fonte: Bogdanov et al., "The MTG-Jamendo Dataset for Automatic Music Tagging", ICML 2019
(https://github.com/MTG/mtg-jamendo-dataset). Metadati in ``runs/pool_cc_v1/meta`` (copiati dal
repository, mai modificati); audio dal mirror ufficiale https://cdn.freesound.org/mtg-jamendo/
(``raw_30s/audio``: tracce INTERE a 320 kbps, il "30s" del nome indica solo il filtro durata > 30 s).

Sottocomandi::

    select  filtra (tag genre, durata, licenza registrata, niente ND) e sceglie le tracce
    manifest scrive LICENSES.json e sources.json per le tracce scaricate
    fetch   legge dai tar solo i membri scelti (HTTP Range, nessun archivio intero su disco),
            verifica sha256 per traccia (``raw_30s_audio_sha256_tracks.txt``) e scrive mp3 + JSON
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import shutil
import sys
import tarfile
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "runs/pool_cc_v1/meta"
SELECTION = ROOT / "runs/pool_cc_v1/selection.json"
AUDIO_DIR = ROOT / "local_audio/pool_cc_v1"
CDN = "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio/raw_30s_audio-{shard}.tar"

PRIMARY = {"genre---hiphop", "genre---rap", "genre---trap"}
EXTENDED = {"genre---gangstarap", "genre---rapfrancais", "genre---instrumentalhiphop",
            "genre---abstracthiphop", "genre---christianrap", "genre---rapcore"}
CITATION = ("Bogdanov, D., Won, M., Tovstogan, P., Porter, A., Serra, X. (2019). The MTG-Jamendo Dataset "
            "for Automatic Music Tagging. Machine Learning for Music Discovery Workshop, ICML 2019, "
            "Long Beach, CA. http://hdl.handle.net/10230/42015")
DATASET_NOTE = ("Dataset distribuito per uso non commerciale di ricerca e accademico (README MTG-Jamendo); "
                "metadati CC BY-NC-SA 4.0; audio sotto le licenze CC individuali di ogni traccia.")


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48]


def sha256_bytes_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_metadata() -> list[dict]:
    lic = {}
    for block in (META / "audio_licenses.txt").read_text(encoding="utf-8").split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            m = re.search(r": (https?://\S+)", lines[2])
            lic[lines[0]] = {"text": lines[2].strip(), "url": m.group(1) if m else None}
    meta = {}
    for line in (META / "raw.meta.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        f = line.split("\t")
        meta[f[0]] = {"album_id": f[2], "title": f[3], "artist": f[4], "album": f[5], "release": f[6], "url": f[7]}
    sha = {}
    for line in (META / "raw_30s_audio_sha256_tracks.txt").read_text().splitlines():
        h, p = line.split(" ", 1)
        sha[p.strip()] = h
    rows = []
    for line in (META / "raw_30s_cleantags.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        f = line.split("\t")
        track_id, artist_id, album_id, path, duration = f[:5]
        m = meta.get(track_id, {})
        rows.append({"track_id": track_id, "jamendo_id": int(track_id.split("_")[1]), "artist_id": artist_id,
                     "path": path, "duration_s": float(duration), "tags": f[5:],
                     "title": m.get("title"), "artist": m.get("artist"), "album": m.get("album"),
                     "release": m.get("release"), "jamendo_url": m.get("url"),
                     "license": lic.get(path), "sha256_expected": sha.get(path)})
    return rows


def eligible(r: dict, dmin: float, dmax: float) -> tuple[bool, str]:
    if not (set(r["tags"]) & (PRIMARY | EXTENDED)):
        return False, "tag"
    if not dmin <= r["duration_s"] <= dmax:
        return False, "duration"
    if not r["license"] or not r["license"]["url"]:
        return False, "license_missing"
    if not re.search(r"creativecommons\.org/licenses/", r["license"]["url"]):
        return False, "license_not_cc"
    if re.search(r"/by(-nc)?-nd/", r["license"]["url"]):
        return False, "license_nd"
    if not r["sha256_expected"] or not r["title"] or not r["artist"]:
        return False, "metadata_incomplete"
    return True, "ok"


def cmd_select(args: argparse.Namespace) -> None:
    rows = load_metadata()
    reasons: Counter = Counter()
    pool = []
    for r in rows:
        ok, why = eligible(r, args.min_s, args.max_s)
        if set(r["tags"]) & (PRIMARY | EXTENDED):
            reasons[why] += 1
        if ok:
            pool.append(r)
    # Tracce rap/trap/hiphop vere (tag primari) e con voce probabile; sottogeneri solo come riempimento.
    def tier(r: dict) -> int:
        t = set(r["tags"])
        if t & {"genre---rap", "genre---trap"} and "genre---hiphop" in t:
            return 0
        if t & {"genre---rap", "genre---trap"}:
            return 1
        if "genre---hiphop" in t:
            return 2
        return 3
    # Si sceglie un numero limitato di shard (lettura Range = una richiesta per membro del tar).
    by_shard = defaultdict(list)
    for r in pool:
        if tier(r) <= 2:
            by_shard[r["path"].split("/")[0]].append(r)
    shards = sorted(by_shard, key=lambda s: -sum(1 for r in by_shard[s] if tier(r) <= 1))[:args.shards]
    rng = random.Random(args.seed)
    cand = [r for s in shards for r in by_shard[s]]
    rng.shuffle(cand)
    cand.sort(key=tier)
    chosen, per_artist = [], Counter()
    for r in cand:
        if per_artist[r["artist_id"]] >= args.per_artist:
            continue
        per_artist[r["artist_id"]] += 1
        chosen.append(r)
        if len(chosen) >= args.count:
            break
    used, slugs = Counter(), set()
    for r in chosen:
        base = slugify(f"{r['artist']}_{r['title']}") or "track"
        slug = f"{base}_j{r['jamendo_id']}"
        r["slug"] = slug
        slugs.add(slug)
    out = {"schema_version": "musicmix.cc_pool_selection.1", "dataset_citation": CITATION,
           "dataset_note": DATASET_NOTE, "criteria": {"genre_primary": sorted(PRIMARY), "genre_extended": sorted(EXTENDED),
                                                    "duration_s": [args.min_s, args.max_s], "exclude": "ND",
                                                    "per_artist_max": args.per_artist, "seed": args.seed,
                                                    "shards": shards},
           "metadata_files": {p.name: sha256_bytes_file(p) for p in sorted(META.iterdir())},
           "filter_counts_tagged_tracks": dict(reasons), "eligible_total": len(pool),
           "tracks": sorted(chosen, key=lambda r: (r["path"].split("/")[0], r["jamendo_id"]))}
    SELECTION.parent.mkdir(parents=True, exist_ok=True)
    SELECTION.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"eligible": len(pool), "filters": dict(reasons), "shards": shards, "chosen": len(chosen),
                      "licenses": Counter(re.search(r"licenses/([a-z-]+)/", r["license"]["url"]).group(1) for r in chosen),
                      "tiers": Counter(tier(r) for r in chosen), "artists": len(per_artist)}, indent=1, default=dict))


class RangeFile(io.RawIOBase):
    """File seekable su HTTP Range (tarfile salta tra gli header: una richiesta per header)."""

    def __init__(self, url: str, session: requests.Session):
        self.url, self.s, self.pos = url, session, 0
        r = self.s.head(url, timeout=60, allow_redirects=True)
        r.raise_for_status()
        self.size = int(r.headers["content-length"])
        assert r.headers.get("accept-ranges") == "bytes"
        self.requests = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n <= 0:
            return b""
        end = self.pos + n - 1
        for attempt in range(6):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120)
                if r.status_code != 206:
                    raise IOError(f"HTTP {r.status_code}")
                if len(r.content) != n:
                    raise IOError("short range read")
                break
            except (requests.RequestException, IOError):
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        self.requests += 1
        self.pos += n
        return r.content

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def fetch_shard(shard: str, wanted: dict[str, dict], dry: bool) -> dict:
    url = CDN.format(shard=shard)
    s = requests.Session()
    rf = RangeFile(url, s)
    found, t0 = {}, time.time()
    with tarfile.open(fileobj=rf, mode="r:") as tf:
        for member in tf:
            if member.name not in wanted:
                continue
            r = wanted[member.name]
            data = tf.extractfile(member).read()
            digest = hashlib.sha256(data).hexdigest()
            ok = digest == r["sha256_expected"]
            found[member.name] = {"sha256": digest, "sha256_ok": ok, "bytes": len(data)}
            if ok and not dry:
                dest = AUDIO_DIR / f"{r['slug']}.mp3"
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(dest)
            print(f"[{shard}] {member.name} {len(data)/1e6:.1f} MB sha256 {'OK' if ok else 'MISMATCH'}", flush=True)
            if len(found) == len(wanted):
                break
    return {"shard": shard, "tar_url": url, "tar_bytes": rf.size, "range_requests": rf.requests,
            "seconds": round(time.time() - t0, 1), "members": found,
            "missing": sorted(set(wanted) - set(found))}


def cmd_fetch(args: argparse.Namespace) -> None:
    sel = json.loads(SELECTION.read_text(encoding="utf-8"))
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(AUDIO_DIR).free
    print(f"spazio libero {free/1e9:.1f} GB", flush=True)
    by_shard = defaultdict(dict)
    for r in sel["tracks"]:
        by_shard[r["path"].split("/")[0]][r["path"]] = r
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        reports = list(ex.map(lambda kv: fetch_shard(kv[0], kv[1], args.dry_run), sorted(by_shard.items())))
    (ROOT / "runs/pool_cc_v1/fetch_report.json").write_text(json.dumps(reports, indent=1), encoding="utf-8")
    bad = [m for rep in reports for m, v in rep["members"].items() if not v["sha256_ok"]]
    miss = [m for rep in reports for m in rep["missing"]]
    print(json.dumps({"mismatch": bad, "missing": miss}))


def cmd_manifest(args: argparse.Namespace) -> None:
    """Scrive LICENSES.json (accanto agli mp3) e runs/pool_cc_v1/sources.json."""
    import subprocess
    sel = json.loads(SELECTION.read_text(encoding="utf-8"))
    tar_sha = {}
    for line in (META / "raw_30s_audio_sha256_tars.txt").read_text().splitlines():
        h, n = line.split(" ", 1)
        tar_sha[n.strip()] = h
    fetch = {}
    rep_path = ROOT / "runs/pool_cc_v1/fetch_report.json"
    if rep_path.exists():
        for rep in json.loads(rep_path.read_text()):
            fetch[rep["shard"]] = rep
    tracks, licenses = [], {}
    for r in sel["tracks"]:
        mp3 = AUDIO_DIR / f"{r['slug']}.mp3"
        if not mp3.is_file():
            continue
        digest = sha256_bytes_file(mp3)
        if digest != r["sha256_expected"]:
            raise SystemExit(f"{mp3}: sha256 diverso dal dataset")
        probe = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration,bit_rate:stream=codec_name,sample_rate,channels",
             "-of", "json", str(mp3)], capture_output=True, text=True, check=True).stdout)
        shard = r["path"].split("/")[0]
        tar_name = f"raw_30s_audio-{shard}.tar"
        lic_url = r["license"]["url"]
        code = re.search(r"licenses/([a-z-]+)/([0-9.]+)", lic_url)
        entry = {
            "slug": r["slug"], "file": f"{r['slug']}.mp3", "jamendo_track_id": r["jamendo_id"],
            "mtg_track_id": r["track_id"], "jamendo_artist_id": r["artist_id"], "artist": r["artist"], "title": r["title"],
            "album": r["album"], "release_date": r["release"], "jamendo_url": r["jamendo_url"],
            "license": {"text": r["license"]["text"], "url": lic_url,
                        "code": f"CC {code.group(1).upper()} {code.group(2)}" if code else None},
            "tags": r["tags"], "duration_s_dataset": r["duration_s"],
            "duration_s_ffprobe": float(probe["format"]["duration"]), "bit_rate": int(probe["format"]["bit_rate"]),
            "bytes": mp3.stat().st_size, "sha256": digest, "sha256_source": "raw_30s_audio_sha256_tracks.txt (dataset), verificato",
            "archive": {"name": tar_name, "member": r["path"], "url": CDN.format(shard=shard),
                        "tar_sha256_published": tar_sha.get(tar_name),
                        "tar_sha256_verified": False,
                        "note": "membro letto con HTTP Range; il tar intero non e' stato scaricato, "
                                "quindi vale la verifica sha256 per traccia"}}
        tracks.append(entry)
        licenses[r["slug"]] = {k: entry[k] for k in ("artist", "title", "jamendo_url", "license", "sha256")}
    common = {"dataset": "MTG-Jamendo", "dataset_url": "https://github.com/MTG/mtg-jamendo-dataset",
              "audio_mirror": "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio/", "citation": CITATION,
              "dataset_note": DATASET_NOTE, "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "selection_criteria": sel["criteria"], "metadata_files_sha256": sel["metadata_files"],
              "attribution_note": "Le licenze CC BY/BY-SA richiedono attribuzione (artista, titolo, URL Jamendo, licenza); "
                                  "BY-SA/NC-SA impongono la stessa licenza alle opere derivate (i mix); "
                                  "NC esclude uso commerciale. Nessuna traccia ND."}
    (AUDIO_DIR / "LICENSES.json").write_text(json.dumps({**common, "tracks": licenses}, indent=1, ensure_ascii=False), encoding="utf-8")
    out = ROOT / "runs/pool_cc_v1/sources.json"
    out.write_text(json.dumps({**common, "count": len(tracks), "license_counts": Counter(t["license"]["code"] for t in tracks),
                               "fetch_report": fetch, "tracks": tracks}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"tracks": len(tracks), "licenses": Counter(t["license"]["code"] for t in tracks),
                      "bytes": sum(t["bytes"] for t in tracks)}, default=dict))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--count", type=int, default=56)
    s.add_argument("--shards", type=int, default=6)
    s.add_argument("--per-artist", type=int, default=2)
    s.add_argument("--min-s", type=float, default=120)
    s.add_argument("--max-s", type=float, default=360)
    s.add_argument("--seed", type=int, default=20261020)
    s.set_defaults(fn=cmd_select)
    f = sub.add_parser("fetch")
    f.add_argument("--jobs", type=int, default=3)
    f.add_argument("--dry-run", action="store_true")
    f.set_defaults(fn=cmd_fetch)
    m = sub.add_parser("manifest")
    m.set_defaults(fn=cmd_manifest)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
