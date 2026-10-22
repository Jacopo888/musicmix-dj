"""Build a self-contained listening page (Artifact) from a 20-second export manifest.

Input (detected from ``schema_version``; manifests are read only):
- ``musicmix.twenty-second-export.1`` (e.g. ``runs/psychology_v11_20s/manifest.json``):
  ``artifacts`` with id, path, sha256, sample_rate, duration_samples, join_sample.
  Recorded Windows paths are resolved with :func:`musicmix.paths.localize`.
- ``musicmix.psychology-review.1`` (``review_manifest.json`` written by
  ``tools/run_psychology_pool_v12.py``): ``items`` with id, title, audio_path
  (relative to the repo root), candidate_id, A_out/B_in {track, seconds},
  action (``execute``/``stay``), intent, notes.  The join is fixed at 10 s;
  sample rate and length are read from the WAV.  There is no per-item hash:
  the sha256 is computed from the WAV and checked against
  ``manifest.json`` next to it (``artifacts[].artifact`` and
  ``natural_references``) when that file records one; otherwise the clip is
  kept with ``hash_status = "hash_non_verificabile"``.
- ``musicmix.dj-review.1`` (``review_manifest.json`` written by
  ``tools/run_dj_gallery_v1.py``): ``groups`` (one per transition plan: pair
  A→B with titles, BPM, Camelot keys, A_out/B_in seconds, sections, plan
  reasons) each with ``items`` (versions of the same change point: id, kind,
  label, params, audio_path relative to the repo root, duration_s, markers
  ``transition_start_s``/``switch_s``/``transition_end_s`` in clip seconds,
  ``qa`` verdict and warn reasons, sha256 of the WAV).  Clips whose WAV hash
  differs from ``sha256`` are excluded.  The page shows one block per group,
  the versions side by side (stacked on phones), each with its own player,
  transition region and switch line; per version a 1–10 score, DJ quick tags
  and a note (doc ``<run>__<item id>``), per group the question "Quale
  versione preferisci?" (doc ``<run>__<group id>__preferenza``), both in the
  ``judgments`` collection.

- ``set.json`` of a continuous DJ set (``musicmix.dj_set.1``, written by
  ``tools/run_dj_set_v1.py``) or its normalized form ``musicmix.dj-set.1``
  (:func:`convert_set`; written to ``<out>/set_manifest.json``): tracklist
  (artist, title, BPM, Camelot, set_start_s/set_end_s), transitions (index,
  type, label, rule, ``set_times_s`` start/switch/end, ``qa``) and
  ``output.wav``/``wav_sha256``/``duration_s``.  ``tracklist.csv`` beside the
  manifest is the fallback when the JSON lacks them.  Audio: ``--set-mp3``
  (or ``set_128k.mp3`` beside the manifest) is copied as is; otherwise the WAV
  is hash-checked and encoded at 128 kbps CBR.  The page has one player with
  a long bar (track bands and boundaries, transition regions coloured by type,
  switch lines), a clickable tracklist, one row per transition (go to −4 s,
  replay −4 s…end+4 s with optional loop, 1–10 score, DJ quick tags, note →
  doc ``<run>__t<NN>``) and an overall set judgment (score, smoothness,
  sensible order yes/no/partly, note → doc ``<run>__set``).

- ``musicmix.dj-review.2``: same manifest as ``.1``; since 21/10/2026 every dj-review page (``.1`` or ``.2``)
  saves ``schema_version: musicmix.dj-review.2`` judgments with structured ``corrections`` (A exit / B entry
  ±bars, pickup, section, wrong B, stay on A, technique, grid; :mod:`listening_corrections`) and the group cue.
- ``--kind set-approval`` (set.json) and ``--kind cue-tournament`` (``musicmix.cue-variants.1``, auto-detected):
  see :mod:`listening_pages` (whole-set approval with per-transition verdicts; blind cue tournament).

Output folder (``--out``):
- ``index.html``: standalone page, one player per clip with seek buttons, a
  "vai al cambio" jump and a clickable progress bar marking the join; a
  judgment form saved to the Artifact ``db`` capability (collection
  ``judgments``, doc id ``<run>__<clip id>``), with a localStorage fallback.
- ``audio/<id>.mp3`` (ffmpeg libmp3lame 192 kbps; venv soundfile MP3 as a
  second choice; WAV PCM16 if no MP3 encoder exists).  Each WAV is checked
  against the manifest sha256 before encoding; missing or mismatching files
  are excluded and recorded.
- ``listening_manifest.json``: provenance, included/excluded clips, hashes.

Usage:
    python tools/build_listening_artifact.py runs/psychology_v11_20s/manifest.json \\
        --out runs/psychology_v11_20s/listening_artifact [--run NAME] [--limit N]
    python tools/build_listening_artifact.py runs/psychology_v12_release/review_manifest.json \\
        --out runs/psychology_v12_release/listening_artifact [--repo-root DIR]
    python tools/build_listening_artifact.py runs/dj_set_v1/set.json --out OUT [--set-mp3 FILE.mp3]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from musicmix.paths import localize, python_main, repo_root  # noqa: E402

if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
import listening_corrections as LC  # noqa: E402

SCHEMA_IN = "musicmix.twenty-second-export.1"
SCHEMA_REVIEW = "musicmix.psychology-review.1"
SCHEMA_DJ = "musicmix.dj-review.1"
SCHEMA_DJ2 = "musicmix.dj-review.2"       # stesso manifest; i giudizi salvati hanno anche ``corrections``
SCHEMAS_DJ = (SCHEMA_DJ, SCHEMA_DJ2)
SCHEMA_SET = "musicmix.dj-set.1"          # normalized set manifest (written as set_manifest.json)
SCHEMA_SET_RAW = "musicmix.dj_set.1"      # set.json written by tools/run_dj_set_v1.py
SCHEMAS_SET = (SCHEMA_SET, SCHEMA_SET_RAW)
SCHEMAS_IN = (SCHEMA_IN, SCHEMA_REVIEW) + SCHEMAS_DJ + SCHEMAS_SET
SET_MP3_BITRATE = "128k"
SET_MP3_MAX_BYTES = 15 * 1000 * 1000
SET_LEAD_S = 4.0
SET_DOC_SUFFIX = "set"
SET_TOLERANCE_S = 0.05
REVIEW_JOIN_S = 10
REVIEW_RATE = 48000
SCHEMA_OUT = "musicmix.listening-artifact.1"
VENV_PY = Path(python_main())
MP3_BITRATE = "192k"
COLLECTION = "judgments"
GO_TO_JOIN_LEAD_S = 2.0
PREFERENCE_SUFFIX = "preferenza"
NO_PREFERENCE = "nessuna"
CLIP_SIDE_S = 10.0
CLIP_TOLERANCE_S = 0.002
KIND_LABELS = {"cut": "taglio", "blend": "blend {bars} battute", "echo_out": "echo out",
               "filter_out": "filter out", "drop_in": "drop in"}
_SEGMENT_BAD = re.compile(r"[^A-Za-z0-9_\-.~:@+]")


# --------------------------------------------------------------------------- ids

def sanitize_segment(text: str, max_bytes: int = 120) -> str:
    """Make ``text`` a valid db path segment (letters, digits, ``_ - . ~ : @ +``)."""
    seg = _SEGMENT_BAD.sub("_", str(text)).strip()
    if seg in ("", ".", ".."):
        seg = seg.replace(".", "_") or "_"
    return seg.encode("ascii")[:max_bytes].decode("ascii")


def doc_id(run: str, clip_id: str) -> str:
    return f"{sanitize_segment(run)}__{sanitize_segment(clip_id)}"


def preference_doc_id(run: str, group_id: str) -> str:
    """Doc id of the per-group "which version do you prefer?" answer."""
    return f"{doc_id(run, group_id)}__{PREFERENCE_SUFFIX}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------- encoders

def _ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _ffmpeg_has_lame(ffmpeg: str) -> bool:
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "libmp3lame" in out


def _venv_soundfile_mp3() -> bool:
    if not VENV_PY.exists():
        return False
    code = "import soundfile as sf, numpy; print('MP3' in sf.available_formats())"
    try:
        out = subprocess.run([str(VENV_PY), "-c", code], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return out.strip() == "True"


def choose_encoder(requested: str = "auto") -> str:
    """Return one of ``ffmpeg-mp3``, ``soundfile-mp3``, ``wav16``."""
    ffmpeg = _ffmpeg()
    if requested in ("auto", "ffmpeg-mp3") and ffmpeg and _ffmpeg_has_lame(ffmpeg):
        return "ffmpeg-mp3"
    if requested == "ffmpeg-mp3":
        raise RuntimeError("ffmpeg con libmp3lame non disponibile")
    if requested in ("auto", "soundfile-mp3") and _venv_soundfile_mp3():
        return "soundfile-mp3"
    if requested == "soundfile-mp3":
        raise RuntimeError("soundfile con supporto MP3 non disponibile nella venv")
    return "wav16"


_SF_SCRIPT = r"""
import sys, soundfile as sf
src, dst, fmt = sys.argv[1:4]
data, rate = sf.read(src, always_2d=True)
if fmt == 'MP3':
    sf.write(dst, data, rate, format='MP3')
else:
    sf.write(dst, data, rate, format='WAV', subtype='PCM_16')
"""


def encode(src: Path, dst: Path, encoder: str, bitrate: str = MP3_BITRATE) -> None:
    ffmpeg = _ffmpeg()
    if encoder == "ffmpeg-mp3":
        # libmp3lame with -b:a is CBR: needed for accurate seeking in long files
        cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
               "-map_metadata", "-1", "-codec:a", "libmp3lame", "-b:a", bitrate, str(dst)]
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    elif encoder == "soundfile-mp3":
        subprocess.run([str(VENV_PY), "-c", _SF_SCRIPT, str(src), str(dst), "MP3"], check=True,
                       capture_output=True, timeout=600)
    elif encoder == "wav16":
        if ffmpeg:
            cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                   "-map_metadata", "-1", "-codec:a", "pcm_s16le", str(dst)]
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            return
        try:
            with wave.open(str(src), "rb") as r:
                if r.getsampwidth() != 2:
                    raise ValueError("not PCM16")
                params, frames = r.getparams(), r.readframes(r.getnframes())
            with wave.open(str(dst), "wb") as w:
                w.setparams(params)
                w.writeframes(frames)
        except (wave.Error, ValueError, EOFError):
            if not VENV_PY.exists():
                raise RuntimeError("WAV non PCM16 e nessun convertitore disponibile")
            subprocess.run([str(VENV_PY), "-c", _SF_SCRIPT, str(src), str(dst), "WAV"], check=True,
                           capture_output=True, timeout=600)
    else:
        raise ValueError(f"encoder sconosciuto: {encoder}")


# ----------------------------------------------------------------------- manifest

def resolve_audio(recorded: str, manifest_dir: Path) -> Path:
    path = localize(recorded)
    if not path.is_absolute() and not path.exists():
        path = manifest_dir / path
    return path


def _check_numbers(item: dict) -> str | None:
    try:
        rate = int(item["sample_rate"])
        dur = int(item["duration_samples"])
        join = int(item["join_sample"])
    except (KeyError, TypeError, ValueError) as exc:
        return f"campi numerici mancanti o non validi: {exc}"
    if rate <= 0 or dur <= 0:
        return "sample_rate o duration_samples non positivi"
    if not 0 <= join <= dur:
        return "join_sample fuori dalla clip"
    return None


def wav_info(path: Path) -> tuple[int, int]:
    """Return ``(sample_rate, frames)`` from a RIFF/WAVE header (PCM, float or extensible)."""
    with open(path, "rb") as fh:
        head = fh.read(12)
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError("non è un file RIFF/WAVE")
        rate = block = None
        while True:
            chunk = fh.read(8)
            if len(chunk) < 8:
                break
            cid, size = chunk[:4], int.from_bytes(chunk[4:], "little")
            if cid == b"fmt ":
                fmt = fh.read(size)
                if len(fmt) < 16:
                    raise ValueError("chunk fmt troncato")
                rate = int.from_bytes(fmt[4:8], "little")
                block = int.from_bytes(fmt[12:14], "little")
                if size % 2:
                    fh.seek(1, 1)
            elif cid == b"data":
                if rate is None or not block:
                    raise ValueError("chunk data prima di fmt")
                here = fh.tell()
                avail = os.fstat(fh.fileno()).st_size - here
                return rate, min(size, avail) // block
            else:
                fh.seek(size + (size % 2), 1)
    raise ValueError("chunk fmt/data mancante")


def _basename(recorded: str) -> str:
    return str(recorded).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def load_sibling_hashes(manifest_dir: Path) -> tuple[dict[str, str], dict | None]:
    """Hashes recorded in ``manifest.json`` beside a review manifest, keyed by file name."""
    path = manifest_dir / "manifest.json"
    if not path.is_file():
        return {}, None
    raw = path.read_bytes()
    try:
        data = json.loads(raw.decode("utf8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "readable": False}
    records = []
    for entry in data.get("artifacts", []) or []:
        if isinstance(entry, dict) and isinstance(entry.get("artifact"), dict):
            records.append(entry["artifact"])
    records += [r for r in data.get("natural_references", []) or [] if isinstance(r, dict)]
    hashes: dict[str, str] = {}
    for rec in records:
        recorded = rec.get("path") or rec.get("output_path")
        digest = rec.get("sha256") or rec.get("output_sha256")
        if recorded and isinstance(digest, str) and len(digest) == 64:
            hashes[_basename(recorded)] = digest.lower()
    meta = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "readable": True,
            "schema_version": data.get("schema_version"), "hashes_found": len(hashes)}
    return hashes, meta


def resolve_review_audio(recorded: str, manifest_dir: Path, root: Path) -> Path:
    """``audio_path`` is relative to the repo root; fall back to the manifest's ancestors."""
    path = localize(recorded)
    if path.is_absolute():
        return path
    candidates = [Path(root) / path] + [parent / path for parent in [manifest_dir, *manifest_dir.parents]]
    for cand in candidates:
        if cand.is_file():
            return cand
    return candidates[0]


def _prepare_legacy(item: dict, manifest_dir: Path) -> tuple[dict | None, tuple[str, str] | None]:
    problem = _check_numbers(item)
    if problem:
        return None, ("campi_non_validi", problem)
    if not item.get("path") or not item.get("sha256"):
        return None, ("path_o_sha256_mancante", "")
    src = resolve_audio(item["path"], manifest_dir)
    if not src.is_file():
        return None, ("wav_mancante", str(src))
    actual = sha256_file(src)
    if actual != str(item["sha256"]).lower():
        return None, ("sha256_diverso", f"atteso {item['sha256']}, trovato {actual}")
    return {"src": src, "path_recorded": item["path"], "source_sha256": actual, "hash_status": "verificato",
            "sample_rate": int(item["sample_rate"]), "duration_samples": int(item["duration_samples"]),
            "join_sample": int(item["join_sample"])}, None


def _side(value) -> dict | None:
    if not isinstance(value, dict):
        return None
    try:
        seconds = float(value.get("seconds"))
    except (TypeError, ValueError):
        seconds = None
    return {"track": str(value.get("track", "")), "seconds": seconds}


def _prepare_review(item: dict, manifest_dir: Path, root: Path,
                    hashes: dict[str, str]) -> tuple[dict | None, tuple[str, str] | None]:
    recorded = item.get("audio_path")
    if not recorded:
        return None, ("audio_path_mancante", "")
    src = resolve_review_audio(str(recorded), manifest_dir, root)
    if not src.is_file():
        return None, ("wav_mancante", str(src))
    try:
        rate, dur = wav_info(src)
    except (OSError, ValueError) as exc:
        return None, ("wav_non_leggibile", str(exc))
    join = REVIEW_JOIN_S * rate
    if rate <= 0 or dur <= 0 or dur < join:
        return None, ("campi_non_validi", f"wav di {dur} campioni a {rate} Hz: cambio a {REVIEW_JOIN_S} s fuori dalla clip")
    actual = sha256_file(src)
    expected = hashes.get(_basename(str(recorded)))
    if expected and expected != actual:
        return None, ("sha256_diverso", f"atteso {expected} (manifest.json), trovato {actual}")
    action = str(item.get("action", ""))
    notes = item.get("notes") or []
    if not isinstance(notes, list):
        notes = [notes]
    return {"src": src, "path_recorded": recorded, "source_sha256": actual,
            "hash_status": "verificato" if expected else "hash_non_verificabile",
            "sample_rate": rate, "duration_samples": dur, "join_sample": join,
            "title": str(item.get("title", "")), "candidate_id": item.get("candidate_id"),
            "A_out": _side(item.get("A_out")), "B_in": _side(item.get("B_in")),
            "action": action, "intent": item.get("intent"), "notes": [str(n) for n in notes],
            "marker_label": "continuazione originale" if action == "stay" else "cambio"}, None


def kind_label(kind: str, params: dict | None = None) -> str:
    params = params or {}
    label = KIND_LABELS.get(kind, kind.replace("_", " "))
    return label.format(bars=params.get("bars", "?"))


def _float(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def _prepare_dj(item: dict, group: dict, manifest_dir: Path, root: Path) -> tuple[dict | None, tuple[str, str] | None]:
    recorded = item.get("audio_path")
    if not recorded:
        return None, ("audio_path_mancante", "")
    src = resolve_review_audio(str(recorded), manifest_dir, root)
    if not src.is_file():
        return None, ("wav_mancante", str(src))
    try:
        rate, dur = wav_info(src)
    except (OSError, ValueError) as exc:
        return None, ("wav_non_leggibile", str(exc))
    if rate <= 0 or dur <= 0:
        return None, ("campi_non_validi", f"wav di {dur} campioni a {rate} Hz")
    length = dur / rate
    marks = {k: _float(item.get(k)) for k in ("transition_start_s", "switch_s", "transition_end_s")}
    t0, sw, t1 = marks["transition_start_s"], marks["switch_s"], marks["transition_end_s"]
    tol = CLIP_TOLERANCE_S
    if None in (t0, sw, t1) or not (-tol <= t0 <= sw + tol and sw <= t1 + tol and t1 <= length + tol):
        return None, ("marcatori_non_validi", f"transizione {t0}–{t1} s, switch {sw} s in una clip di {length:.3f} s")
    actual = sha256_file(src)
    expected = str(item.get("sha256") or "").lower()
    if expected and expected != actual:
        return None, ("sha256_diverso", f"atteso {expected}, trovato {actual}")
    kind = str(item.get("kind", ""))
    params = item.get("params") if isinstance(item.get("params"), dict) else {}
    qa = item.get("qa") if isinstance(item.get("qa"), dict) else {}
    warn = qa.get("warn_reasons") or []
    if not isinstance(warn, list):
        warn = [warn]
    is_cut = kind in ("cut", "drop_in")
    if is_cut:
        rule_ok = abs(length - 2 * CLIP_SIDE_S) <= tol and abs(sw - CLIP_SIDE_S) <= tol
    else:
        rule_ok = abs(t0 - CLIP_SIDE_S) <= tol and abs(length - (t1 + CLIP_SIDE_S)) <= tol
    return {"src": src, "path_recorded": recorded, "source_sha256": actual,
            "hash_status": "verificato" if expected else "hash_non_verificabile",
            "sample_rate": rate, "duration_samples": dur, "join_sample": int(round(sw * rate)),
            "group_id": str(group.get("id", "")), "kind": kind,
            "label": str(item.get("label") or kind_label(kind, params)), "role": item.get("role"),
            "params": params, "why": item.get("why"),
            "transition_start_s": t0, "switch_s": sw, "transition_end_s": t1,
            "qa_verdict": str(qa.get("verdict") or "non_registrato"), "qa_warn_reasons": [str(w) for w in warn],
            "clip_rule_ok": bool(rule_ok)}, None


def _group_record(group: dict, run: str, clips: list[dict]) -> dict:
    keep = ("id", "title", "pair", "bpm", "key", "A_out_s", "B_in_s", "A_out_bar", "B_in_bar", "sections",
            "plan", "reasons", "cue_adjustment")
    rec = {k: group.get(k) for k in keep}
    rec["id"] = str(group.get("id", ""))
    rec["preference_doc_id"] = preference_doc_id(run, rec["id"])
    rec["items"] = [c["id"] for c in clips]
    return rec


KINDS = ("auto", "set-approval", "cue-tournament")


def build(manifest_path: Path, out: Path, run: str | None = None, limit: int | None = None,
          encoder: str = "auto", overwrite: bool = False, log=print, root: Path | None = None,
          set_mp3: Path | None = None, blind: bool = False, kind: str = "auto", seed: int = 0,
          set_bitrate: str | None = None) -> dict:
    """Pagina d'ascolto; ``kind`` = ``set-approval`` / ``cue-tournament`` usa :mod:`listening_pages`."""
    manifest_path = Path(manifest_path)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf8"))
    schema = manifest.get("schema_version")
    if kind not in KINDS:
        raise ValueError(f"kind sconosciuto: {kind!r} (attesi {', '.join(KINDS)})")
    if kind == "cue-tournament" or (kind == "auto" and schema == "musicmix.cue-variants.1"):
        import listening_pages as LP
        return LP.build_cue_tournament(manifest_path, out, run=run, encoder=encoder, overwrite=overwrite, log=log,
                                       root=root, seed=seed)
    if kind == "set-approval":
        import listening_pages as LP
        return LP.build_set_approval(manifest_path, out, run=run, encoder=encoder, overwrite=overwrite, log=log,
                                     root=root, set_mp3=set_mp3, blind=blind, bitrate=set_bitrate)
    if schema not in SCHEMAS_IN:
        raise ValueError(f"schema non supportato: {schema!r} (attesi {', '.join(SCHEMAS_IN)})")
    if schema in SCHEMAS_SET:
        return build_set(manifest_path, out, run=run, encoder=encoder, overwrite=overwrite, log=log,
                         root=root, set_mp3=set_mp3)
    review = schema == SCHEMA_REVIEW
    dj = schema in SCHEMAS_DJ
    root = Path(root) if root is not None else repo_root()
    run = run or manifest_path.resolve().parent.name
    out = Path(out)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} esiste e non è vuota (usa --overwrite)")
    (out / "audio").mkdir(parents=True, exist_ok=True)

    hashes, hash_source = load_sibling_hashes(manifest_path.parent) if review else ({}, None)
    if dj:
        source_groups = [g for g in manifest.get("groups", []) or [] if isinstance(g, dict)]
        group_of: dict[int, dict] = {}
        all_items = []
        for g in source_groups:
            for it in g.get("items", []) or []:
                group_of[id(it)] = g
                all_items.append(it)
    else:
        all_items = list(manifest.get("items" if review else "artifacts", []))
    artifacts = all_items
    if limit is not None:
        artifacts = artifacts[: max(0, limit)]
    enc = choose_encoder(encoder)
    ext = "wav" if enc == "wav16" else "mp3"
    log(f"encoder: {enc}")

    included, excluded, seen = [], [], set()
    for item in artifacts:
        item = item if isinstance(item, dict) else {}
        clip_id = str(item.get("id", ""))
        path_recorded = item.get("audio_path" if (review or dj) else "path")

        def skip(reason: str, detail: str = "") -> None:
            excluded.append({"id": clip_id, "reason": reason, "detail": detail,
                             "path_recorded": path_recorded})
            log(f"ESCLUSA {clip_id or '?'}: {reason} {detail}".rstrip())

        if not clip_id:
            skip("id_mancante"); continue
        fname = sanitize_segment(clip_id, 150)
        if fname in seen:
            skip("id_duplicato"); continue
        if dj:
            prep, problem = _prepare_dj(item, group_of.get(id(item), {}), manifest_path.parent, root)
        elif review:
            prep, problem = _prepare_review(item, manifest_path.parent, root, hashes)
        else:
            prep, problem = _prepare_legacy(item, manifest_path.parent)
        if problem:
            skip(*problem); continue
        src = prep.pop("src")
        dst = out / "audio" / f"{fname}.{ext}"
        try:
            encode(src, dst, enc)
        except (subprocess.CalledProcessError, OSError, RuntimeError) as exc:
            err = getattr(exc, "stderr", b"") or b""
            skip("codifica_fallita", (err.decode("utf8", "replace") if isinstance(err, bytes) else str(err))[-400:] or str(exc))
            dst.unlink(missing_ok=True)
            continue
        seen.add(fname)
        rate, dur, join = prep["sample_rate"], prep["duration_samples"], prep["join_sample"]
        record = {
            "id": clip_id,
            "doc_id": doc_id(run, clip_id),
            "audio": f"audio/{fname}.{ext}",
            "audio_sha256": sha256_file(dst),
            "audio_bytes": dst.stat().st_size,
            "path_recorded": prep.pop("path_recorded"),
            "path_resolved": str(src),
            "source_sha256": prep.pop("source_sha256"),
            "hash_status": prep.pop("hash_status"),
            "sample_rate": rate,
            "duration_samples": dur,
            "join_sample": join,
            "duration_s": dur / rate,
            "join_s": join / rate,
            "twenty_s_ten_each_side": dur == 20 * rate and join == 10 * rate,
        }
        for key in ("sample_rate", "duration_samples", "join_sample"):
            prep.pop(key)
        record.update(prep)
        included.append(record)
        if dj:
            if not record["clip_rule_ok"]:
                log(f"AVVISO {clip_id}: durata fuori regola AGENTS.md ({dur / rate:.2f} s, transizione "
                    f"{record['transition_start_s']:.2f}–{record['transition_end_s']:.2f} s)")
        elif dur != 20 * rate or join != 10 * rate:
            log(f"AVVISO {clip_id}: non è 20 s con cambio a 10 s ({dur / rate:.2f} s, cambio {join / rate:.2f} s)")
        if (review or dj) and rate != REVIEW_RATE:
            log(f"AVVISO {clip_id}: sample rate {rate} Hz invece di {REVIEW_RATE} Hz")
        if record["hash_status"] != "verificato":
            log(f"AVVISO {clip_id}: {record['hash_status']}")
        log(f"ok {clip_id} -> {dst.name}")

    ffmpeg = _ffmpeg()
    ffmpeg_version = None
    if ffmpeg and enc != "soundfile-mp3":
        try:
            ffmpeg_version = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True,
                                            timeout=30).stdout.splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError):
            pass
    listing = {
        "schema_version": SCHEMA_OUT,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": run,
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "schema_version": schema,
            "artifact_count": len(all_items),
            "title": manifest.get("title"),
        },
        "hash_source": hash_source,
        "limit": limit,
        "encoder": {"kind": enc, "bitrate": MP3_BITRATE if enc == "ffmpeg-mp3" else None,
                    "ffmpeg": ffmpeg_version},
        "db": {"collection": COLLECTION, "doc_id": "<run>__<clip id> (segmenti sanificati)"},
        "included": included,
        "excluded": excluded,
    }
    groups_out = None
    if dj:
        groups_out = []
        for g in source_groups:
            clips = [c for c in included if c["group_id"] == str(g.get("id", ""))]
            if clips:
                groups_out.append(dict(_group_record(g, run, clips), clips=clips))
        listing["groups"] = [{k: v for k, v in g.items() if k != "clips"} for g in groups_out]
        listing["db"]["preference_doc_id"] = f"<run>__<group id>__{PREFERENCE_SUFFIX}"
    (out / "listening_manifest.json").write_text(json.dumps(listing, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf8")
    page = (render_dj_html(run, groups_out, manifest.get("title"), blind=blind) if dj
            else render_html(run, included, manifest.get("title")))
    (out / "index.html").write_text(page, encoding="utf8")
    log(f"{len(included)} incluse, {len(excluded)} escluse -> {out}")
    return listing


# --------------------------------------------------------------------------- html

TAGS = [
    ("troppo_presto", "cambio troppo presto"),
    ("troppo_tardi", "troppo tardi"),
    ("punto_giusto", "punto giusto"),
    ("meglio_continuare_A", "meglio continuare A"),
    ("stacco_udibile", "stacco udibile"),
]


def render_html(run: str, clips: list[dict], heading: str | None = None) -> str:
    data = {
        "run": run,
        "collection": COLLECTION,
        "lead": GO_TO_JOIN_LEAD_S,
        "tags": [{"key": k, "label": v} for k, v in TAGS],
        "clips": [{k: c[k] for k in ("id", "doc_id", "audio", "source_sha256", "duration_s", "join_s",
                                     "join_sample", "sample_rate")} for c in clips],
    }
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
    cards = "\n".join(_card(c) for c in clips) or '<p class="muted">Nessuna clip inclusa.</p>'
    subtitle = f'<p class="small">{html.escape(heading)}</p>' if heading else ""
    return (HTML_TEMPLATE
            .replace("__SUBTITLE__", subtitle)
            .replace("__RUN__", html.escape(run))
            .replace("__COUNT__", str(len(clips)))
            .replace("__CARDS__", cards)
            .replace("__DATA__", payload))


def _fmt_side(side: dict | None) -> str:
    if not side:
        return "—"
    sec = side.get("seconds")
    where = f" a {sec:.2f} s" if isinstance(sec, (int, float)) else ""
    return f"{html.escape(side.get('track') or '?')}{where}"


def _details(c: dict) -> str:
    if "action" not in c:
        return ""
    rows = [("A esce", _fmt_side(c.get("A_out")))]
    if c.get("action") == "stay":
        rows.append(("Azione", "resta su A (continuazione originale)"))
    else:
        rows.append(("B entra", _fmt_side(c.get("B_in"))))
    if c.get("intent"):
        rows.append(("Intent", f"<code>{html.escape(str(c['intent']))}</code>"))
    if c.get("hash_status") and c["hash_status"] != "verificato":
        rows.append(("Hash", html.escape(c["hash_status"].replace("_", " "))))
    dl = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in c.get("notes") or [])
    return (f'<dl class="meta small">{dl}</dl>'
            + (f'<ul class="notes small muted">{notes}</ul>' if notes else ""))


def _card(c: dict) -> str:
    cid = html.escape(c["id"])
    title = html.escape(c.get("title") or "")
    label = html.escape(c.get("marker_label") or "cambio")
    pct = 100.0 * c["join_s"] / c["duration_s"]
    scores = "".join(
        f'<label class="chip"><input type="radio" name="score-{cid}" value="{n}"><span>{n}</span></label>'
        for n in range(1, 11))
    tags = "".join(
        f'<label class="chip"><input type="checkbox" value="{k}"><span>{html.escape(v)}</span></label>'
        for k, v in TAGS)
    return f"""<article class="card" data-clip="{cid}" data-doc="{html.escape(c['doc_id'])}">
  <h2>{title or cid}</h2>
  <p class="muted small">{(cid + " · ") if title else ""}{label} a {c['join_s']:.2f} s (campione {c['join_sample']}) · doc <code>{html.escape(c['doc_id'])}</code></p>
  {_details(c)}
  <audio preload="metadata" controls src="{html.escape(c['audio'])}"></audio>
  <div class="track" role="slider" tabindex="0" aria-label="Posizione in {cid}" aria-valuemin="0" aria-valuemax="{c['duration_s']:.2f}" aria-valuenow="0">
    <div class="fill"></div>
    <div class="join-marker" data-join-s="{c['join_s']:.6f}" style="left:{pct:.4f}%" title="{'punto di cambio' if label == 'cambio' else label}"></div>
  </div>
  <div class="row between small muted"><span class="time">0:00</span><span>{label} {c['join_s']:.1f} s</span><span class="dur">{c['duration_s']:.1f} s</span></div>
  <div class="row buttons">
    <button type="button" data-act="back">−5 s</button>
    <button type="button" data-act="play">▶︎ / ❚❚</button>
    <button type="button" data-act="fwd">+5 s</button>
    <button type="button" data-act="join">{'vai al cambio' if label == 'cambio' else 'vai a ' + format(c['join_s'], '.0f') + ' s'}</button>
  </div>
  <form class="judge">
    <fieldset><legend>Voto</legend><div class="chips">{scores}</div></fieldset>
    <fieldset><legend>Scelte rapide</legend><div class="chips">{tags}</div></fieldset>
    <label class="note-label">Nota<textarea rows="2" name="note" placeholder="cosa si sente al cambio…"></textarea></label>
    <div class="row between"><button type="submit" class="primary">Salva</button><span class="status small muted">non salvato</span></div>
    <p class="existing small muted"></p>
  </form>
</article>"""


HTML_TEMPLATE = r"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ascolti MusicMix</title>
<style>
:root{--bg:#f6f5f2;--surface:#ffffff;--text:#1d1d1b;--muted:#6a6862;--border:#dddad3;--accent:#2f5bd3;--accent-text:#ffffff;--fill:#2f5bd3;--track:#e7e4dd;--marker:#d2462f;--ok:#2f7d4f;--warn:#a5651a;--warn-bg:#fbf0de;color-scheme:light}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141413;--surface:#1f1f1d;--text:#ecebe6;--muted:#a3a19a;--border:#34332f;--accent:#7c9cf5;--accent-text:#0e1220;--fill:#7c9cf5;--track:#2d2c29;--marker:#ff7a5f;--ok:#6fcf97;--warn:#f0b35a;--warn-bg:#2f2615;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#141413;--surface:#1f1f1d;--text:#ecebe6;--muted:#a3a19a;--border:#34332f;--accent:#7c9cf5;--accent-text:#0e1220;--fill:#7c9cf5;--track:#2d2c29;--marker:#ff7a5f;--ok:#6fcf97;--warn:#f0b35a;--warn-bg:#2f2615;color-scheme:dark}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:760px;margin:0 auto;padding:16px}
h1{font-size:1.3rem;margin:.2rem 0}
h2{font-size:1rem;margin:0 0 .2rem;overflow-wrap:anywhere}
code{font-size:.85em;overflow-wrap:anywhere}
.muted{color:var(--muted)}.small{font-size:.85rem}
.banner{display:none;background:var(--warn-bg);color:var(--text);border:1px solid var(--warn);border-radius:8px;padding:.6rem .8rem;margin:.8rem 0}
.banner.show{display:block}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px;margin:14px 0}
audio{width:100%;margin:.4rem 0}
.track{position:relative;height:22px;background:var(--track);border-radius:6px;cursor:pointer;touch-action:none;outline-offset:2px}
.fill{position:absolute;left:0;top:0;bottom:0;width:0;background:var(--fill);border-radius:6px;opacity:.55}
.join-marker{position:absolute;top:-4px;bottom:-4px;width:3px;margin-left:-1px;background:var(--marker);border-radius:2px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:.4rem 0}
.between{justify-content:space-between}
button{font:inherit;padding:.45rem .7rem;border-radius:8px;border:1px solid var(--border);background:var(--surface);color:var(--text);cursor:pointer;min-height:40px}
button.primary{background:var(--accent);color:var(--accent-text);border-color:var(--accent)}
button:disabled{opacity:.6;cursor:default}
.buttons button{flex:1 1 auto}
fieldset{border:0;padding:0;margin:.6rem 0}
legend{font-weight:600;font-size:.9rem;margin-bottom:.3rem}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{position:relative}
.chip input{position:absolute;opacity:0;inset:0}
.chip span{display:inline-block;padding:.35rem .6rem;border:1px solid var(--border);border-radius:999px;min-width:2.2rem;text-align:center;font-size:.9rem;cursor:pointer}
.chip input:checked+span{background:var(--accent);color:var(--accent-text);border-color:var(--accent)}
.chip input:focus-visible+span{outline:2px solid var(--accent);outline-offset:2px}
.note-label{display:block;font-weight:600;font-size:.9rem}
textarea{display:block;width:100%;margin-top:.3rem;font:inherit;padding:.5rem;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);resize:vertical}
.meta{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;margin:.4rem 0}
.meta dt{color:var(--muted)}.meta dd{margin:0;overflow-wrap:anywhere}
.notes{margin:.2rem 0 .4rem;padding-left:1.1rem}
.status.ok{color:var(--ok)}.status.err{color:var(--marker)}
</style>
</head>
<body>
<main>
<header>
  <h1>Ascolti MusicMix</h1>
  __SUBTITLE__
  <p class="muted small">Run <code>__RUN__</code> · __COUNT__ clip · la linea rossa segna il punto di cambio.</p>
  <div class="banner" id="banner"></div>
  <div class="row"><span class="small muted" id="dbstate">Collegamento al database in corso…</span><button type="button" id="copy" class="small">Copia JSON</button></div>
</header>
__CARDS__
</main>
<script id="clip-data" type="application/json">__DATA__</script>
<script>
(function(){
"use strict";
const CFG = JSON.parse(document.getElementById("clip-data").textContent);
const LS_KEY = "musicmix:judgments:" + CFG.run;
const byId = new Map(CFG.clips.map(c => [c.doc_id, c]));
const state = {db: null, mode: "pending", saved: {}, local: {}};
const $ = (s, el) => (el || document).querySelector(s);
const fmt = s => { s = Math.max(0, s || 0); const m = Math.floor(s / 60); const r = Math.floor(s % 60); return m + ":" + String(r).padStart(2, "0"); };
const sleep = ms => new Promise(r => setTimeout(r, ms));

function lsRead(){ try { return JSON.parse(localStorage.getItem(LS_KEY) || "{}") || {}; } catch (e) { return {}; } }
function lsWrite(obj){ try { localStorage.setItem(LS_KEY, JSON.stringify(obj)); return true; } catch (e) { return false; } }
function banner(msg){ const b = $("#banner"); b.textContent = msg; b.classList.toggle("show", !!msg); }

// ---------------------------------------------------------------- players
document.querySelectorAll(".card").forEach(card => {
  const clip = byId.get(card.dataset.doc);
  const audio = $("audio", card), track = $(".track", card), fill = $(".fill", card), time = $(".time", card);
  const dur = () => (isFinite(audio.duration) && audio.duration > 0) ? audio.duration : clip.duration_s;
  const seek = t => { audio.currentTime = Math.min(Math.max(0, t), Math.max(0, dur() - 0.05)); };
  const paint = () => {
    const p = Math.min(1, audio.currentTime / dur());
    fill.style.width = (p * 100) + "%";
    time.textContent = fmt(audio.currentTime) + " / " + fmt(dur());
    track.setAttribute("aria-valuenow", audio.currentTime.toFixed(1));
  };
  ["timeupdate", "seeked", "loadedmetadata"].forEach(ev => audio.addEventListener(ev, paint));
  audio.addEventListener("play", () => document.querySelectorAll("audio").forEach(a => { if (a !== audio) a.pause(); }));
  const fromPointer = ev => { const r = track.getBoundingClientRect(); seek(((ev.clientX - r.left) / r.width) * dur()); };
  track.addEventListener("pointerdown", ev => { track.setPointerCapture(ev.pointerId); fromPointer(ev); });
  track.addEventListener("pointermove", ev => { if (track.hasPointerCapture(ev.pointerId)) fromPointer(ev); });
  track.addEventListener("keydown", ev => {
    if (ev.key === "ArrowLeft") { seek(audio.currentTime - 1); ev.preventDefault(); }
    if (ev.key === "ArrowRight") { seek(audio.currentTime + 1); ev.preventDefault(); }
  });
  card.querySelectorAll("button[data-act]").forEach(btn => btn.addEventListener("click", () => {
    const act = btn.dataset.act;
    if (act === "back") seek(audio.currentTime - 5);
    else if (act === "fwd") seek(audio.currentTime + 5);
    else if (act === "join") { seek(clip.join_s - CFG.lead); audio.play().catch(() => {}); }
    else if (act === "play") { audio.paused ? audio.play().catch(() => {}) : audio.pause(); }
  }));
  card._audio = audio;
});

// ---------------------------------------------------------------- form
function readForm(card){
  const f = $("form", card);
  const score = $("input[type=radio]:checked", f);
  return {
    score: score ? Number(score.value) : null,
    tags: Array.from(f.querySelectorAll("input[type=checkbox]:checked")).map(i => i.value),
    note: f.note.value.trim()
  };
}
function fillForm(card, j){
  const f = $("form", card);
  f.querySelectorAll("input[type=radio]").forEach(i => { i.checked = Number(i.value) === j.score; });
  f.querySelectorAll("input[type=checkbox]").forEach(i => { i.checked = (j.tags || []).includes(i.value); });
  f.note.value = j.note || "";
}
function setStatus(card, text, cls){ const s = $(".status", card); s.textContent = text; s.className = "status small " + (cls || "muted"); }
function showExisting(card, j, where){
  const p = $(".existing", card);
  if (!j) { p.textContent = ""; return; }
  const labels = (j.tags || []).map(k => (CFG.tags.find(t => t.key === k) || {label: k}).label);
  const when = j.saved_at ? new Date(j.saved_at).toLocaleString("it-IT") : "";
  p.textContent = "Giudizio " + where + ": " + (j.score != null ? j.score + "/10" : "senza voto") +
    (labels.length ? " · " + labels.join(", ") : "") + (j.note ? " · “" + j.note + "”" : "") +
    (when ? " · " + when : "") + (j.position_s != null ? " · a " + Number(j.position_s).toFixed(1) + " s" : "");
}
function applyJudgment(docId, j, where){
  const card = document.querySelector('.card[data-doc="' + CSS.escape(docId) + '"]');
  if (!card) return;
  showExisting(card, j, where);
  if (j && !card._dirty && !card._saving) { fillForm(card, j); setStatus(card, "salvato", "ok"); }
}

const queue = {};  // docId -> {busy, pending}
async function save(card){
  const clip = byId.get(card.dataset.doc);
  const docId = clip.doc_id;
  const q = queue[docId] || (queue[docId] = {busy: false, pending: false});
  if (q.busy) { q.pending = true; setStatus(card, "in coda…"); return; }
  q.busy = true; card._saving = true;
  const btn = $("button[type=submit]", card); btn.disabled = true;
  try {
    do {
      q.pending = false;
      const body = Object.assign({
        run: CFG.run, clip_id: clip.id, source_sha256: clip.source_sha256
      }, readForm(card), {
        position_s: Math.round(card._audio.currentTime * 1000) / 1000,
        saved_at: new Date().toISOString()
      });
      card._dirty = false;
      setStatus(card, "salvataggio…");
      if (state.db) {
        try {
          await state.db.collection(CFG.collection).doc(docId).set(body);
          state.saved[docId] = body;
          setStatus(card, "salvato nel database", "ok");
        } catch (e) {
          const code = e && e.code || "errore";
          state.local[docId] = body; lsWrite(state.local);
          setStatus(card, "non salvato nel database (" + code + "): copia locale", "err");
          if (code === "invalid_argument") banner("Non hai permesso di scrittura sul database: i giudizi restano in questo browser. Usa “Copia JSON”.");
        }
      } else {
        state.local[docId] = body;
        const ok = lsWrite(state.local);
        setStatus(card, ok ? "salvato in questo browser" : "non salvato (storage bloccato): usa Copia JSON", ok ? "ok" : "err");
      }
      showExisting(card, state.saved[docId] || state.local[docId], state.saved[docId] ? "salvato" : "locale");
    } while (q.pending);
  } finally {
    q.busy = false; card._saving = false; btn.disabled = false;
  }
}

document.querySelectorAll(".card").forEach(card => {
  const f = $("form", card);
  f.addEventListener("input", () => { card._dirty = true; setStatus(card, "modifiche non salvate"); });
  f.addEventListener("submit", ev => { ev.preventDefault(); save(card); });
});

// ---------------------------------------------------------------- copy JSON
$("#copy").addEventListener("click", async () => {
  const all = Object.assign({}, state.local, state.saved);
  const text = JSON.stringify({run: CFG.run, collection: CFG.collection, judgments: all}, null, 2);
  try { await navigator.clipboard.writeText(text); $("#copy").textContent = "Copiato"; }
  catch (e) {
    const ta = document.createElement("textarea"); ta.value = text; ta.rows = 8;
    $("header").appendChild(ta); ta.select();
    $("#copy").textContent = "Seleziona e copia il testo qui sotto";
  }
  setTimeout(() => { $("#copy").textContent = "Copia JSON"; }, 3000);
});

// ---------------------------------------------------------------- db
state.local = lsRead();
Object.entries(state.local).forEach(([id, j]) => applyJudgment(id, j, "locale"));

async function getDb(){
  const deadline = Date.now() + 10000;
  while (!(window.claude && typeof window.claude.use === "function") && Date.now() < deadline) await sleep(100);
  if (!(window.claude && typeof window.claude.use === "function")) return null;
  try { return await Promise.race([window.claude.use("db"), sleep(12000).then(() => null)]); }
  catch (e) { return null; }
}

getDb().then(db => {
  if (!db) {
    state.mode = "local";
    $("#dbstate").textContent = "Database non disponibile: salvataggio in questo browser.";
    banner("Database dell'Artifact non disponibile in questa vista: i giudizi vengono salvati solo in questo browser. Usa “Copia JSON” per conservarli.");
    return;
  }
  state.db = db; state.mode = "db";
  $("#dbstate").textContent = "Giudizi salvati nel database dell'Artifact (" + CFG.collection + ").";
  try {
    db.collection(CFG.collection).onSnapshot(snap => {
      snap.docs.forEach(d => {
        const j = d.data();
        if (!j || j.run !== CFG.run || !byId.has(d.id)) return;
        state.saved[d.id] = j;
        applyJudgment(d.id, j, "salvato");
      });
    }, err => {
      $("#dbstate").textContent = "Lettura dal database interrotta (" + (err && err.code || "errore") + ").";
    });
  } catch (e) {
    $("#dbstate").textContent = "Lettura dal database non riuscita.";
  }
});
})();
</script>
</body>
</html>
"""


# ------------------------------------------------------------ dj-review html

DJ_TAGS = [
    ("battute_sfasate", "battute sfasate"),
    ("punto_frase_sbagliato", "cambio nel punto sbagliato della frase"),
    ("volume_salta", "volume che salta"),
    ("due_voci", "due voci insieme"),
    ("troppo_lunga", "transizione troppo lunga"),
    ("troppo_corta", "troppo corta"),
    ("suona_dj", "suona come un DJ"),
]


def _num(value, fmt: str = ".2f", suffix: str = "") -> str:
    return (format(value, fmt) + suffix) if isinstance(value, (int, float)) else "—"


def _group_meta(g: dict) -> str:
    bpm, key, sec = g.get("bpm") or {}, g.get("key") or {}, g.get("sections") or {}
    stretch = bpm.get("stretch_b_pct")
    rows = [
        ("BPM", f"{_num(bpm.get('a'), '.1f')} → {_num(bpm.get('b'), '.1f')}"
                + (f" (B {stretch:+.1f}%)" if isinstance(stretch, (int, float)) else "")),
        ("Tonalità", f"{html.escape(str(key.get('a') or '?'))} → {html.escape(str(key.get('b') or '?'))}"
                     + (f" · {html.escape(str(key['relation']))}" if key.get("relation") else "")),
        ("A esce", f"{_num(g.get('A_out_s'), '.2f', ' s')}"
                   + (f" · battuta {g['A_out_bar']}" if g.get("A_out_bar") is not None else "")
                   + (f" · {html.escape(str(sec['a']))}" if sec.get("a") else "")),
        ("B entra", f"{_num(g.get('B_in_s'), '.2f', ' s')}"
                    + (f" · battuta {g['B_in_bar']}" if g.get("B_in_bar") is not None else "")
                    + (f" · {html.escape(str(sec['b']))}" if sec.get("b") else "")),
    ]
    adj = g.get("cue_adjustment")
    if isinstance(adj, dict) and adj.get("original"):
        rows.append(("Cue spostato", f"A esce prima: battuta {adj['original'].get('a_out_bar')} in dissolvenza"))
    dl = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
    reasons = [str(r) for r in g.get("reasons") or []]
    short, rest = reasons[:4], reasons[4:]
    notes = "".join(f"<li>{html.escape(r)}</li>" for r in short)
    more = ("<details class=\"small muted\"><summary>altre motivazioni</summary><ul class=\"notes\">"
            + "".join(f"<li>{html.escape(r)}</li>" for r in rest) + "</ul></details>") if rest else ""
    return (f'<dl class="meta small">{dl}</dl>'
            + (f'<ul class="notes small muted">{notes}</ul>' if notes else "") + more)


#: Proposta E1 dei libri (Margulis 2014 p. 15 [E]: le descrizioni date prima dell'ascolto riducono il piacere, "When Program
#: Notes Don't Help", 2010): con ``blind`` la tecnica, i warn QA e le motivazioni del piano restano nascosti finché la versione
#: (o il gruppo) non ha un giudizio salvato. Spento di default.
BLIND_NOTE = ("Ascolto alla cieca: tecnica, controlli e motivazioni compaiono dopo il voto "
              "(Margulis 2014 p. 15: le descrizioni prima dell'ascolto cambiano il giudizio).")


def _is_reference(c: dict) -> bool:
    """``role: riferimento`` (es. la scelta del set v3 accanto alle varianti): si ascolta, non si vota."""
    return str(c.get("role") or "") == "riferimento"


def _dj_reference(c: dict) -> str:
    cid = html.escape(c["id"])
    dur = c["duration_s"] or 1.0
    t0, sw, t1 = c["transition_start_s"], c["switch_s"], c["transition_end_s"]
    left, width, sw_pct = 100.0 * t0 / dur, 100.0 * max(0.0, t1 - t0) / dur, 100.0 * sw / dur
    return f"""<article class="version reference" data-clip="{cid}" data-doc="{html.escape(c['doc_id'])}">
  <div class="row vhead"><span class="badge kind">{html.escape(c["label"])}</span></div>
  <p class="muted small id"><code>{cid}</code> · solo per confronto, non si vota</p>
  <audio preload="metadata" controls src="{html.escape(c['audio'])}"></audio>
  <div class="track" role="slider" tabindex="0" aria-label="Posizione in {cid}" aria-valuemin="0" aria-valuemax="{dur:.2f}" aria-valuenow="0">
    <div class="region" data-start-s="{t0:.6f}" data-end-s="{t1:.6f}" style="left:{left:.4f}%;width:{width:.4f}%" title="regione di transizione"></div>
    <div class="fill"></div>
    <div class="switch-marker" data-switch-s="{sw:.6f}" style="left:{sw_pct:.4f}%" title="switch"></div>
  </div>
  <div class="row between small muted"><span class="time">0:00</span><span>cambio attorno a {sw:.1f} s</span><span class="dur">{dur:.1f} s</span></div>
  <div class="row buttons">
    <button type="button" data-act="back">−5 s</button>
    <button type="button" data-act="play">▶︎ / ❚❚</button>
    <button type="button" data-act="fwd">+5 s</button>
    <button type="button" data-act="transition">vai alla transizione</button>
  </div>
</article>"""


def _dj_version(c: dict, blind: bool = False, number: int = 0) -> str:
    if _is_reference(c):
        return _dj_reference(c)
    cid = html.escape(c["id"])
    dur = c["duration_s"] or 1.0
    t0, sw, t1 = c["transition_start_s"], c["switch_s"], c["transition_end_s"]
    left, width, sw_pct = 100.0 * t0 / dur, 100.0 * max(0.0, t1 - t0) / dur, 100.0 * sw / dur
    warn = c.get("qa_verdict") == "warn"
    reasons = c.get("qa_warn_reasons") or []
    hide = " data-reveal hidden" if blind else ""
    badges = f'<span class="badge kind"{hide}>{html.escape(c["label"])}</span>'
    if blind:
        badges = f'<span class="badge kind">versione {number}</span>' + badges
    if warn:
        badges += f'<span class="badge warnqa"{hide} title="{html.escape("; ".join(reasons))}">warn QA</span>'
    warn_list = (f"<ul class=\"qa-reasons small\"{hide}>" + "".join(f"<li>{html.escape(r)}</li>" for r in reasons)
                 + "</ul>") if warn and reasons else ""
    scores = "".join(
        f'<label class="chip"><input type="radio" name="score-{cid}" value="{n}"><span>{n}</span></label>'
        for n in range(1, 11))
    tags = "".join(
        f'<label class="chip"><input type="checkbox" value="{k}"><span>{html.escape(v)}</span></label>'
        for k, v in DJ_TAGS)
    span = f"taglio a {sw:.2f} s" if t1 - t0 < 0.05 else f"transizione {t0:.1f}–{t1:.1f} s · switch {sw:.1f} s"
    if blind:
        span = f"cambio attorno a {sw:.1f} s"
    return f"""<article class="version" data-clip="{cid}" data-doc="{html.escape(c['doc_id'])}">
  <div class="row vhead">{badges}</div>
  <p class="muted small id"><code>{cid}</code></p>
  <audio preload="metadata" controls src="{html.escape(c['audio'])}"></audio>
  <div class="track" role="slider" tabindex="0" aria-label="Posizione in {cid}" aria-valuemin="0" aria-valuemax="{dur:.2f}" aria-valuenow="0">
    <div class="region" data-start-s="{t0:.6f}" data-end-s="{t1:.6f}" style="left:{left:.4f}%;width:{width:.4f}%" title="regione di transizione"></div>
    <div class="fill"></div>
    <div class="switch-marker" data-switch-s="{sw:.6f}" style="left:{sw_pct:.4f}%" title="switch"></div>
  </div>
  <div class="row between small muted"><span class="time">0:00</span><span>{span}</span><span class="dur">{dur:.1f} s</span></div>
  <div class="row buttons">
    <button type="button" data-act="back">−5 s</button>
    <button type="button" data-act="play">▶︎ / ❚❚</button>
    <button type="button" data-act="fwd">+5 s</button>
    <button type="button" data-act="transition">vai alla transizione</button>
  </div>
  {warn_list}
  <form class="judge">
    <fieldset><legend>Voto</legend><div class="chips">{scores}</div></fieldset>
    <fieldset><legend>Scelte rapide</legend><div class="chips">{tags}</div></fieldset>
    {LC.fieldset_html("c-" + c["id"])}
    <label class="note-label">Nota<textarea rows="2" name="note" placeholder="cosa si sente nella transizione…"></textarea></label>
    <div class="row between"><button type="submit" class="primary">Salva</button><span class="status small muted">non salvato</span></div>
    <p class="existing small muted"></p>
  </form>
</article>"""


def _dj_group(g: dict, blind: bool = False) -> str:
    gid = html.escape(g["id"])
    title = html.escape(g.get("title") or g["id"])
    votable = [c for c in g["clips"] if not _is_reference(c)]
    refs = [c for c in g["clips"] if _is_reference(c)]
    versions = "\n".join([_dj_version(c, blind, k + 1) for k, c in enumerate(votable)] + [_dj_reference(c) for c in refs])
    radios = "".join(
        f'<label class="chip"><input type="radio" name="pref-{gid}" value="{html.escape(c["id"])}"><span>'
        f'{("versione " + str(k + 1)) if blind else html.escape(c["label"])}</span></label>'
        for k, c in enumerate(votable))
    radios += f'<label class="chip"><input type="radio" name="pref-{gid}" value="{NO_PREFERENCE}"><span>nessuna</span></label>'
    return f"""<section class="group" data-group="{gid}" data-pref="{html.escape(g['preference_doc_id'])}">
  <h2>{title}</h2>
  <p class="muted small"><code>{gid}</code> · {len(votable)} versioni dello stesso cambio{(" + " + str(len(refs)) + " riferimento non votabile") if refs else ""}</p>
  {('<div class="group-meta" data-reveal-group hidden>' + _group_meta(g) + '</div>') if blind else _group_meta(g)}
  <div class="versions">
{versions}
  </div>
  <form class="pref">
    <fieldset><legend>Quale versione preferisci?</legend><div class="chips">{radios}</div></fieldset>
    <label class="note-label">Perché (facoltativo)<textarea rows="2" name="note" placeholder="cosa decide la scelta…"></textarea></label>
    <div class="row between"><button type="submit" class="primary">Salva preferenza</button><span class="status small muted">non salvato</span></div>
    <p class="existing small muted"></p>
  </form>
</section>"""


def _group_cue(g: dict) -> dict:
    """Cue del gruppo (uguale per tutte le versioni), copiato nei giudizi ``.2`` per applicare le correzioni."""
    pair = g.get("pair") if isinstance(g.get("pair"), dict) else {}
    side = lambda k: (pair.get(k) or {}).get("slug") if isinstance(pair.get(k), dict) else pair.get(k)  # noqa: E731
    return {"a": side("a"), "b": side("b"), "a_out_bar": g.get("A_out_bar"), "b_in_bar": g.get("B_in_bar"),
            "a_out_s": g.get("A_out_s"), "b_in_s": g.get("B_in_s")}


def render_dj_html(run: str, groups: list[dict], heading: str | None = None, blind: bool = False) -> str:
    clips = [c for g in groups for c in g["clips"]]
    data = {
        "run": run,
        "blind": bool(blind),
        "collection": COLLECTION,
        "lead": GO_TO_JOIN_LEAD_S,
        "none": NO_PREFERENCE,
        "tags": [{"key": k, "label": v} for k, v in DJ_TAGS],
        "clips": [{k: c[k] for k in ("id", "doc_id", "audio", "source_sha256", "duration_s", "transition_start_s",
                                     "switch_s", "transition_end_s", "group_id", "kind", "label", "qa_verdict")}
                  for c in clips],
        "judgment_schema": SCHEMA_DJ2,
        "groups": [{"id": g["id"], "pref_doc_id": g["preference_doc_id"], "cue": _group_cue(g),
                    "items": [{"id": c["id"], "label": f"versione {k + 1}" if blind else c["label"]}
                              for k, c in enumerate([x for x in g["clips"] if not _is_reference(x)])]} for g in groups],
    }
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
    blocks = "\n".join(_dj_group(g, blind) for g in groups) or '<p class="muted">Nessuna versione inclusa.</p>'
    subtitle = f'<p class="small">{html.escape(heading)}</p>' if heading else ""
    if blind:
        subtitle += f'<p class="small muted">{html.escape(BLIND_NOTE)}</p>'
    return (DJ_HTML_TEMPLATE
            .replace("__CORR_CSS__", LC.CSS)
            .replace("__CORR_JS__", LC.js())
            .replace("__SUBTITLE__", subtitle)
            .replace("__RUN__", html.escape(run))
            .replace("__NGROUPS__", str(len(groups)))
            .replace("__COUNT__", str(len(clips)))
            .replace("__GROUPS__", blocks)
            .replace("__DATA__", payload))


DJ_HTML_TEMPLATE = r"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ascolti DJ MusicMix</title>
<style>
:root{--bg:#f6f5f2;--surface:#ffffff;--surface2:#faf9f6;--text:#1d1d1b;--muted:#6a6862;--border:#dddad3;--accent:#2f5bd3;--accent-text:#ffffff;--fill:#2f5bd3;--track:#e7e4dd;--region:#f2c94c;--marker:#d2462f;--ok:#2f7d4f;--warn:#a5651a;--warn-bg:#fbf0de;color-scheme:light}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141413;--surface:#1f1f1d;--surface2:#191918;--text:#ecebe6;--muted:#a3a19a;--border:#34332f;--accent:#7c9cf5;--accent-text:#0e1220;--fill:#7c9cf5;--track:#2d2c29;--region:#8a6d1f;--marker:#ff7a5f;--ok:#6fcf97;--warn:#f0b35a;--warn-bg:#2f2615;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#141413;--surface:#1f1f1d;--surface2:#191918;--text:#ecebe6;--muted:#a3a19a;--border:#34332f;--accent:#7c9cf5;--accent-text:#0e1220;--fill:#7c9cf5;--track:#2d2c29;--region:#8a6d1f;--marker:#ff7a5f;--ok:#6fcf97;--warn:#f0b35a;--warn-bg:#2f2615;color-scheme:dark}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:16px}
h1{font-size:1.3rem;margin:.2rem 0}
h2{font-size:1.1rem;margin:0 0 .2rem;overflow-wrap:anywhere}
code{font-size:.85em;overflow-wrap:anywhere}
.muted{color:var(--muted)}.small{font-size:.85rem}
.banner{display:none;background:var(--warn-bg);color:var(--text);border:1px solid var(--warn);border-radius:8px;padding:.6rem .8rem;margin:.8rem 0}
.banner.show{display:block}
.group{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px;margin:18px 0}
.versions{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:12px;margin:.6rem 0}
.version{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:12px;min-width:0}
.vhead{margin:0 0 .2rem}
.id{margin:.1rem 0}
[hidden]{display:none!important}
.badge{display:inline-block;padding:.15rem .55rem;border-radius:999px;font-size:.8rem;font-weight:600;border:1px solid var(--border)}
.badge.kind{background:var(--accent);color:var(--accent-text);border-color:var(--accent)}
.badge.warnqa{background:var(--warn-bg);color:var(--warn);border-color:var(--warn);cursor:help}
.qa-reasons{color:var(--warn);margin:.2rem 0;padding-left:1.1rem;overflow-wrap:anywhere}
audio{width:100%;margin:.4rem 0}
.track{position:relative;height:24px;background:var(--track);border-radius:6px;cursor:pointer;touch-action:none;outline-offset:2px}
.region{position:absolute;top:0;bottom:0;min-width:3px;background:var(--region);opacity:.75;border-radius:3px}
.fill{position:absolute;left:0;top:0;bottom:0;width:0;background:var(--fill);border-radius:6px;opacity:.45}
.switch-marker{position:absolute;top:-4px;bottom:-4px;width:3px;margin-left:-1px;background:var(--marker);border-radius:2px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:.4rem 0}
.between{justify-content:space-between}
button{font:inherit;padding:.45rem .7rem;border-radius:8px;border:1px solid var(--border);background:var(--surface);color:var(--text);cursor:pointer;min-height:40px}
button.primary{background:var(--accent);color:var(--accent-text);border-color:var(--accent)}
button:disabled{opacity:.6;cursor:default}
.buttons button{flex:1 1 auto}
fieldset{border:0;padding:0;margin:.6rem 0;min-width:0}
legend{font-weight:600;font-size:.9rem;margin-bottom:.3rem}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{position:relative}
.chip input{position:absolute;opacity:0;inset:0}
.chip span{display:inline-block;padding:.35rem .6rem;border:1px solid var(--border);border-radius:999px;min-width:2.2rem;text-align:center;font-size:.9rem;cursor:pointer}
.chip input:checked+span{background:var(--accent);color:var(--accent-text);border-color:var(--accent)}
.chip input:focus-visible+span{outline:2px solid var(--accent);outline-offset:2px}
.note-label{display:block;font-weight:600;font-size:.9rem}
textarea{display:block;width:100%;margin-top:.3rem;font:inherit;padding:.5rem;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);resize:vertical}
.meta{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;margin:.4rem 0}
.meta dt{color:var(--muted)}.meta dd{margin:0;overflow-wrap:anywhere}
.notes{margin:.2rem 0 .4rem;padding-left:1.1rem}
.pref{border-top:1px solid var(--border);padding-top:.4rem}
.legend-key{display:inline-block;width:14px;height:10px;border-radius:2px;vertical-align:middle;margin:0 4px 0 8px}
.status.ok{color:var(--ok)}.status.err{color:var(--marker)}
__CORR_CSS__</style>
</head>
<body>
<main>
<header>
  <h1>Ascolti DJ MusicMix</h1>
  __SUBTITLE__
  <p class="muted small">Run <code>__RUN__</code> · __NGROUPS__ cambi · __COUNT__ versioni ·<span class="legend-key" style="background:var(--region)"></span>regione di transizione<span class="legend-key" style="background:var(--marker);width:3px"></span>switch</p>
  <div class="banner" id="banner"></div>
  <div class="row"><span class="small muted" id="dbstate">Collegamento al database in corso…</span><button type="button" id="copy" class="small">Copia JSON</button></div>
</header>
__GROUPS__
</main>
<script id="clip-data" type="application/json">__DATA__</script>
<script>
(function(){
"use strict";
const CFG = JSON.parse(document.getElementById("clip-data").textContent);
const LS_KEY = "musicmix:judgments:" + CFG.run;
const clipByDoc = new Map(CFG.clips.map(c => [c.doc_id, c]));
const groupByPref = new Map(CFG.groups.map(g => [g.pref_doc_id, g]));
const state = {db: null, saved: {}, local: {}};
const $ = (s, el) => (el || document).querySelector(s);
const fmt = s => { s = Math.max(0, s || 0); const m = Math.floor(s / 60); const r = Math.floor(s % 60); return m + ":" + String(r).padStart(2, "0"); };
const sleep = ms => new Promise(r => setTimeout(r, ms));
function lsRead(){ try { return JSON.parse(localStorage.getItem(LS_KEY) || "{}") || {}; } catch (e) { return {}; } }
function lsWrite(obj){ try { localStorage.setItem(LS_KEY, JSON.stringify(obj)); return true; } catch (e) { return false; } }
function banner(msg){ const b = $("#banner"); b.textContent = msg; b.classList.toggle("show", !!msg); }

// ---------------------------------------------------------------- players
document.querySelectorAll(".version").forEach(card => {
  const clip = clipByDoc.get(card.dataset.doc);
  const audio = $("audio", card), track = $(".track", card), fill = $(".fill", card), time = $(".time", card);
  const dur = () => (isFinite(audio.duration) && audio.duration > 0) ? audio.duration : clip.duration_s;
  const seek = t => { audio.currentTime = Math.min(Math.max(0, t), Math.max(0, dur() - 0.05)); };
  const paint = () => {
    const p = Math.min(1, audio.currentTime / dur());
    fill.style.width = (p * 100) + "%";
    time.textContent = fmt(audio.currentTime) + " / " + fmt(dur());
    track.setAttribute("aria-valuenow", audio.currentTime.toFixed(1));
  };
  ["timeupdate", "seeked", "loadedmetadata"].forEach(ev => audio.addEventListener(ev, paint));
  audio.addEventListener("play", () => document.querySelectorAll("audio").forEach(a => { if (a !== audio) a.pause(); }));
  const fromPointer = ev => { const r = track.getBoundingClientRect(); seek(((ev.clientX - r.left) / r.width) * dur()); };
  track.addEventListener("pointerdown", ev => { track.setPointerCapture(ev.pointerId); fromPointer(ev); });
  track.addEventListener("pointermove", ev => { if (track.hasPointerCapture(ev.pointerId)) fromPointer(ev); });
  track.addEventListener("keydown", ev => {
    if (ev.key === "ArrowLeft") { seek(audio.currentTime - 1); ev.preventDefault(); }
    if (ev.key === "ArrowRight") { seek(audio.currentTime + 1); ev.preventDefault(); }
  });
  card.querySelectorAll("button[data-act]").forEach(btn => btn.addEventListener("click", () => {
    const act = btn.dataset.act;
    if (act === "back") seek(audio.currentTime - 5);
    else if (act === "fwd") seek(audio.currentTime + 5);
    else if (act === "transition") { seek(clip.transition_start_s - CFG.lead); audio.play().catch(() => {}); }
    else if (act === "play") { audio.paused ? audio.play().catch(() => {}) : audio.pause(); }
  }));
  card._audio = audio;
  card._plays = 0;  // E2 (Margulis p. 101; Huron pp. 133-134): listens before the vote are recorded, repeats are not independent
  audio.addEventListener("play", () => { card._plays += 1; });
});

// E1 (Margulis 2014 p. 15): in blind mode the technique and the plan reasons appear only after a saved judgment
function reveal(docId){
  if (!CFG.blind) return;
  const t = target(docId);
  if (!t) return;
  t.el.querySelectorAll("[data-reveal]").forEach(e => { e.hidden = false; });
  const sec = t.el.closest(".group") || t.el;
  const cards = Array.from(sec.querySelectorAll(".version:not(.reference)"));
  const all = cards.every(c => state.saved[c.dataset.doc] || state.local[c.dataset.doc]);
  if (all || t.el.classList.contains("group")) sec.querySelectorAll("[data-reveal-group]").forEach(e => { e.hidden = false; });
}

// ---------------------------------------------------------------- forms
__CORR_JS__
function readVersion(card){
  const f = $("form", card);
  const score = $('input[type=radio][name^="score-"]:checked', f);
  return {score: score ? Number(score.value) : null,
          tags: Array.from(f.querySelectorAll("input[type=checkbox][value]:checked")).map(i => i.value),
          note: f.note.value.trim(), corrections: readCorr(f)};
}
function fillVersion(card, j){
  const f = $("form", card);
  f.querySelectorAll('input[type=radio][name^="score-"]').forEach(i => { i.checked = Number(i.value) === j.score; });
  f.querySelectorAll("input[type=checkbox][value]").forEach(i => { i.checked = (j.tags || []).includes(i.value); });
  fillCorr(f, j.corrections);
  f.note.value = j.note || "";
}
function readPref(sec){
  const f = $("form.pref", sec);
  const c = $("input[type=radio]:checked", f);
  return {choice: c ? c.value : null, note: f.note.value.trim()};
}
function fillPref(sec, j){
  const f = $("form.pref", sec);
  f.querySelectorAll("input[type=radio]").forEach(i => { i.checked = i.value === j.choice; });
  f.note.value = j.note || "";
}
function setStatus(form, text, cls){ const s = $(".status", form); s.textContent = text; s.className = "status small " + (cls || "muted"); }
function describe(j){
  if (j.type === "preferenza") {
    const g = groupByPref.get(j.doc_id) || {items: []};
    const it = g.items.find(x => x.id === j.choice);
    return "Preferenza: " + (j.choice === CFG.none ? "nessuna" : (it ? it.label : (j.choice || "—"))) + (j.note ? " · “" + j.note + "”" : "");
  }
  const labels = (j.tags || []).map(k => (CFG.tags.find(t => t.key === k) || {label: k}).label);
  const corr = corrText(j.corrections);
  return (j.score != null ? j.score + "/10" : "senza voto") + (labels.length ? " · " + labels.join(", ") : "") +
    (corr ? " · correzioni: " + corr : "") +
    (j.note ? " · “" + j.note + "”" : "") + (j.position_s != null ? " · a " + Number(j.position_s).toFixed(1) + " s" : "");
}
function showExisting(form, j, where){
  const p = $(".existing", form);
  if (!j) { p.textContent = ""; return; }
  const when = j.saved_at ? new Date(j.saved_at).toLocaleString("it-IT") : "";
  p.textContent = "Giudizio " + where + ": " + describe(j) + (when ? " · " + when : "");
}
function target(docId){
  if (clipByDoc.has(docId)) {
    const card = document.querySelector('.version[data-doc="' + CSS.escape(docId) + '"]');
    return card ? {el: card, form: $("form", card), fill: j => fillVersion(card, j)} : null;
  }
  if (groupByPref.has(docId)) {
    const sec = document.querySelector('.group[data-pref="' + CSS.escape(docId) + '"]');
    return sec ? {el: sec, form: $("form.pref", sec), fill: j => fillPref(sec, j)} : null;
  }
  return null;
}
function applyJudgment(docId, j, where){
  const t = target(docId);
  if (!t) return;
  showExisting(t.form, j, where);
  if (j && !t.form._dirty && !t.form._saving) { t.fill(j); setStatus(t.form, "salvato", "ok"); }
  if (j) reveal(docId);
}

const queue = {};
async function save(docId, form, makeBody){
  const q = queue[docId] || (queue[docId] = {busy: false, pending: false});
  if (q.busy) { q.pending = true; setStatus(form, "in coda…"); return; }
  q.busy = true; form._saving = true;
  const btn = $("button[type=submit]", form); btn.disabled = true;
  try {
    do {
      q.pending = false;
      const body = Object.assign(makeBody(), {doc_id: docId, saved_at: new Date().toISOString()});
      form._dirty = false;
      setStatus(form, "salvataggio…");
      if (state.db) {
        try {
          await state.db.collection(CFG.collection).doc(docId).set(body);
          state.saved[docId] = body;
          setStatus(form, "salvato nel database", "ok");
        } catch (e) {
          const code = e && e.code || "errore";
          state.local[docId] = body; lsWrite(state.local);
          setStatus(form, "non salvato nel database (" + code + "): copia locale", "err");
          if (code === "invalid_argument") banner("Non hai permesso di scrittura sul database: i giudizi restano in questo browser. Usa “Copia JSON”.");
        }
      } else {
        state.local[docId] = body;
        const ok = lsWrite(state.local);
        setStatus(form, ok ? "salvato in questo browser" : "non salvato (storage bloccato): usa Copia JSON", ok ? "ok" : "err");
      }
      showExisting(form, state.saved[docId] || state.local[docId], state.saved[docId] ? "salvato" : "locale");
      if (state.saved[docId] || state.local[docId]) reveal(docId);
    } while (q.pending);
  } finally {
    q.busy = false; form._saving = false; btn.disabled = false;
  }
}

document.querySelectorAll(".version:not(.reference)").forEach(card => {
  const clip = clipByDoc.get(card.dataset.doc);
  const f = $("form", card);
  f.addEventListener("input", () => { f._dirty = true; setStatus(f, "modifiche non salvate"); });
  f.addEventListener("submit", ev => {
    ev.preventDefault();
    const grp = CFG.groups.find(x => x.id === clip.group_id) || {};
    save(clip.doc_id, f, () => Object.assign({
      run: CFG.run, type: "versione", schema_version: CFG.judgment_schema, clip_id: clip.id, group_id: clip.group_id,
      kind: clip.kind, label: clip.label, qa_verdict: clip.qa_verdict, source_sha256: clip.source_sha256, cue: grp.cue || null
    }, readVersion(card), {position_s: Math.round(card._audio.currentTime * 1000) / 1000, plays_before_save: card._plays,
       blind: !!CFG.blind}));
  });
});
document.querySelectorAll(".group").forEach(sec => {
  const g = groupByPref.get(sec.dataset.pref);
  const f = $("form.pref", sec);
  f.addEventListener("input", () => { f._dirty = true; setStatus(f, "modifiche non salvate"); });
  f.addEventListener("submit", ev => {
    ev.preventDefault();
    const r = readPref(sec);
    if (!r.choice) { setStatus(f, "scegli una versione o “nessuna”", "err"); return; }
    save(g.pref_doc_id, f, () => Object.assign({run: CFG.run, type: "preferenza", group_id: g.id,
      options: g.items.map(x => x.id).concat([CFG.none]), blind: !!CFG.blind,
      plays_before_save: Array.from(sec.querySelectorAll(".version")).reduce((n, c) => n + (c._plays || 0), 0)}, readPref(sec)));
  });
});

// ---------------------------------------------------------------- copy JSON
$("#copy").addEventListener("click", async () => {
  const all = Object.assign({}, state.local, state.saved);
  const text = JSON.stringify({run: CFG.run, collection: CFG.collection, judgments: all}, null, 2);
  try { await navigator.clipboard.writeText(text); $("#copy").textContent = "Copiato"; }
  catch (e) {
    const ta = document.createElement("textarea"); ta.value = text; ta.rows = 8;
    $("header").appendChild(ta); ta.select();
    $("#copy").textContent = "Seleziona e copia il testo qui sotto";
  }
  setTimeout(() => { $("#copy").textContent = "Copia JSON"; }, 3000);
});

// ---------------------------------------------------------------- db
state.local = lsRead();
Object.entries(state.local).forEach(([id, j]) => applyJudgment(id, j, "locale"));

async function getDb(){
  const deadline = Date.now() + 10000;
  while (!(window.claude && typeof window.claude.use === "function") && Date.now() < deadline) await sleep(100);
  if (!(window.claude && typeof window.claude.use === "function")) return null;
  try { return await Promise.race([window.claude.use("db"), sleep(12000).then(() => null)]); }
  catch (e) { return null; }
}

getDb().then(db => {
  if (!db) {
    $("#dbstate").textContent = "Database non disponibile: salvataggio in questo browser.";
    banner("Database dell'Artifact non disponibile in questa vista: i giudizi vengono salvati solo in questo browser. Usa “Copia JSON” per conservarli.");
    return;
  }
  state.db = db;
  $("#dbstate").textContent = "Giudizi salvati nel database dell'Artifact (" + CFG.collection + ").";
  try {
    db.collection(CFG.collection).onSnapshot(snap => {
      snap.docs.forEach(d => {
        const j = d.data();
        if (!j || j.run !== CFG.run || !(clipByDoc.has(d.id) || groupByPref.has(d.id))) return;
        state.saved[d.id] = j;
        applyJudgment(d.id, j, "salvato");
      });
    }, err => {
      $("#dbstate").textContent = "Lettura dal database interrotta (" + (err && err.code || "errore") + ").";
    });
  } catch (e) {
    $("#dbstate").textContent = "Lettura dal database non riuscita.";
  }
});
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------- musicmix.dj-set.1

# prefix of the transition type -> colour family (first match wins)
SET_TYPE_FAMILIES = (("blend", "blend"), ("filter_into_chorus", "filter_chorus"), ("filter", "filter"),
                     ("echo", "echo"), ("cut", "cut"), ("drop", "cut"))
SET_FAMILY_LABELS = {"blend": "blend", "filter_chorus": "filter + ritornello", "filter": "filter out",
                     "echo": "echo out", "cut": "taglio", "other": "altro"}
SET_ORDER_CHOICES = (("si", "sì"), ("in_parte", "in parte"), ("no", "no"))


def type_family(kind: str | None) -> str:
    k = str(kind or "").lower()
    for prefix, fam in SET_TYPE_FAMILIES:
        if k.startswith(prefix):
            return fam
    return "other"


def transition_doc_id(run: str, index: int) -> str:
    """Doc id of one set transition: ``<run>__t<NN>``."""
    return f"{sanitize_segment(run)}__t{int(index):02d}"


def set_doc_id(run: str) -> str:
    """Doc id of the overall set judgment: ``<run>__set``."""
    return f"{sanitize_segment(run)}__{SET_DOC_SUFFIX}"


def _mmss(seconds: float | None) -> str:
    if not isinstance(seconds, (int, float)):
        return "—"
    s = max(0, int(seconds))
    return f"{s // 60}:{s % 60:02d}"


def _read_csv(path: Path) -> list[dict]:
    import csv
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf8") as fh:
        return list(csv.DictReader(fh))


def convert_set(manifest: dict, manifest_dir: Path) -> dict:
    """Normalize a ``set.json`` (``musicmix.dj_set.1``) into ``musicmix.dj-set.1``.

    A manifest already in ``musicmix.dj-set.1`` is returned as a copy.  The
    tracklist and transitions fall back to ``tracklist.csv`` beside the
    manifest when ``set.json`` lacks them.  Nothing is validated here.
    """
    if manifest.get("schema_version") == SCHEMA_SET:
        return json.loads(json.dumps(manifest))
    sources = manifest.get("sources") if isinstance(manifest.get("sources"), dict) else {}
    output = manifest.get("output") if isinstance(manifest.get("output"), dict) else {}
    csv_rows = None
    tracks_raw = [t for t in manifest.get("tracklist") or [] if isinstance(t, dict)]
    from_csv = False
    if not tracks_raw:
        csv_rows = _read_csv(manifest_dir / "tracklist.csv")
        tracks_raw, from_csv = csv_rows, bool(csv_rows)
    tracks = []
    for i, t in enumerate(tracks_raw, 1):
        slug = str(t.get("slug") or "")
        src = sources.get(slug) if isinstance(sources.get(slug), dict) else {}
        pos = t.get("position")
        tracks.append({
            "position": int(pos) if str(pos or "").strip().isdigit() else i,
            "slug": slug,
            "artist": str(t.get("artist") or src.get("artist") or ""),
            "title": str(t.get("title") or src.get("title") or slug),
            "start_s": _float(t.get("set_start_s")), "end_s": _float(t.get("set_end_s")),
            "bpm": _float(t.get("bpm") if t.get("bpm") not in (None, "") else src.get("bpm")),
            "camelot": str(t.get("camelot") or src.get("camelot") or "") or None,
            "energy": _float(t.get("energy")),
        })
    transitions = []
    raw_tr = [t for t in manifest.get("transitions") or [] if isinstance(t, dict)]
    if raw_tr:
        for t in raw_tr:
            times = t.get("set_times_s") if isinstance(t.get("set_times_s"), dict) else {}
            qa = t.get("qa") if isinstance(t.get("qa"), dict) else {}
            reasons = qa.get("reasons", qa.get("warn_reasons")) or []
            kind = str(t.get("type") or "")
            transitions.append({
                "index": t.get("index"), "type": kind, "family": type_family(kind),
                "label": str(t.get("label") or kind_label(kind)), "rule": t.get("rule"),
                "a": t.get("a"), "b": t.get("b"),
                "start_s": _float(times.get("start")), "switch_s": _float(times.get("switch")),
                "end_s": _float(times.get("end")),
                "stretch_pct": _float(t.get("stretch_pct")),
                "qa_verdict": str(qa.get("verdict") or "non_registrato"),
                "qa_reasons": [str(r) for r in (reasons if isinstance(reasons, list) else [reasons])],
                "qa_kept_despite_fail": bool(qa.get("kept_despite_fail")),
            })
    else:
        rows = csv_rows if csv_rows is not None else _read_csv(manifest_dir / "tracklist.csv")
        for i, row in enumerate(rows):
            kind = str(row.get("transition_out_type") or "")
            if not kind:
                continue
            nxt = rows[i + 1].get("slug") if i + 1 < len(rows) else None
            transitions.append({
                "index": i + 1, "type": kind, "family": type_family(kind), "label": kind_label(kind),
                "rule": row.get("rule") or None, "a": row.get("slug"), "b": nxt,
                "start_s": _float(row.get("transition_start_s")), "switch_s": _float(row.get("switch_s")),
                "end_s": _float(row.get("transition_end_s")), "stretch_pct": None,
                "qa_verdict": str(row.get("qa") or "non_registrato"), "qa_reasons": [],
                "qa_kept_despite_fail": False,
            })
    duration = _float(output.get("duration_s"))
    if duration is None:
        ends = [t["end_s"] for t in tracks if t["end_s"] is not None]
        duration = max(ends) if ends else None
    return {
        "schema_version": SCHEMA_SET,
        "converted_from": manifest.get("schema_version"),
        "run": manifest.get("run"),
        "title": manifest.get("title"),
        "script": manifest.get("script"),
        "created_at": manifest.get("created_at"),
        "duration_s": duration,
        "sample_rate": output.get("sample_rate"),
        "audio": {"wav": output.get("wav"), "wav_sha256": output.get("wav_sha256"),
                  "mp3": output.get("mp3"), "mp3_sha256": output.get("mp3_sha256")},
        "tracklist_source": "tracklist.csv" if from_csv else "set.json",
        "tracks": tracks,
        "transitions": transitions,
        "qa_counts_final": manifest.get("qa_counts_final"),
        "notes": [str(n) for n in manifest.get("notes") or []],
    }


def validate_set(norm: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Return ``(tracks, transitions, excluded)``; invalid entries are excluded, never fixed."""
    dur = _float(norm.get("duration_s"))
    tol = SET_TOLERANCE_S
    if dur is None or dur <= 0:
        raise ValueError("durata del set mancante o non valida")
    tracks, transitions, excluded = [], [], []
    for t in norm.get("tracks") or []:
        s, e = _float(t.get("start_s")), _float(t.get("end_s"))
        if s is None or e is None or not (-tol <= s < e <= dur + tol):
            excluded.append({"kind": "brano", "id": t.get("slug"), "reason": "tempi_non_validi",
                             "detail": f"{s}–{e} s in un set di {dur:.3f} s"})
            continue
        tracks.append(dict(t))
    tracks.sort(key=lambda t: t["start_s"])
    seen = set()
    for t in norm.get("transitions") or []:
        try:
            idx = int(t.get("index"))
        except (TypeError, ValueError):
            idx = None
        s, w, e = (_float(t.get(k)) for k in ("start_s", "switch_s", "end_s"))
        if idx is None or idx < 1 or idx in seen:
            excluded.append({"kind": "transizione", "id": t.get("index"), "reason": "indice_non_valido"})
            continue
        if None in (s, w, e) or not (-tol <= s <= w + tol and w <= e + tol and e <= dur + tol):
            excluded.append({"kind": "transizione", "id": idx, "reason": "marcatori_non_validi",
                             "detail": f"inizio {s}, switch {w}, fine {e} in un set di {dur:.3f} s"})
            continue
        seen.add(idx)
        transitions.append(dict(t, index=idx))
    transitions.sort(key=lambda t: t["index"])
    return tracks, transitions, excluded


def _probe(path: Path) -> dict:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {}
    try:
        out = subprocess.run([ffprobe, "-v", "error", "-show_entries",
                              "format=duration,bit_rate:stream=codec_name,sample_rate", "-of", "json", str(path)],
                             capture_output=True, text=True, timeout=60).stdout
        data = json.loads(out or "{}")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {}
    fmt = data.get("format") or {}
    st = (data.get("streams") or [{}])[0]
    br = _float(fmt.get("bit_rate"))
    return {"duration_s": _float(fmt.get("duration")), "bit_rate": int(br) if br else None,
            "codec": st.get("codec_name"), "sample_rate": int(st["sample_rate"]) if st.get("sample_rate") else None}


def build_set(manifest_path: Path, out: Path, run: str | None = None, encoder: str = "auto",
              overwrite: bool = False, log=print, root: Path | None = None,
              set_mp3: Path | None = None) -> dict:
    """Build the listening page of a continuous DJ set (one player, 9-ish transitions)."""
    manifest_path = Path(manifest_path)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf8"))
    manifest_dir = manifest_path.parent
    root = Path(root) if root is not None else repo_root()
    norm = convert_set(manifest, manifest_dir)
    run = run or norm.get("run") or manifest_path.resolve().parent.name
    norm["run"] = run
    out = Path(out)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} esiste e non è vuota (usa --overwrite)")
    tracks, transitions, excluded = validate_set(norm)
    if not tracks:
        raise ValueError("nessun brano valido nel set")
    for e in excluded:
        log(f"ESCLUSA {e['kind']} {e.get('id')}: {e['reason']} {e.get('detail', '')}".rstrip())
    (out / "audio").mkdir(parents=True, exist_ok=True)

    recorded = norm.get("audio") or {}
    if set_mp3 is None and (manifest_dir / "set_128k.mp3").is_file():
        set_mp3 = manifest_dir / "set_128k.mp3"
        log(f"uso l'mp3 già codificato {set_mp3}")
    if set_mp3 is not None:
        src = Path(set_mp3)
        if not src.is_file():
            raise FileNotFoundError(f"mp3 del set non trovato: {src}")
        dst = out / "audio" / "set.mp3"
        shutil.copyfile(src, dst)
        actual = sha256_file(src)
        rec_mp3 = recorded.get("mp3")
        expected = str(recorded.get("mp3_sha256") or "").lower()
        same_file = bool(rec_mp3) and _basename(str(rec_mp3)) == src.name
        if same_file and expected and expected != actual:
            raise ValueError(f"sha256 dell'mp3 diverso: atteso {expected}, trovato {actual}")
        audio_src = {"kind": "mp3_esistente", "path": str(src), "sha256": actual,
                     "hash_status": "verificato" if same_file and expected else "hash_non_verificabile"}
        enc_info = {"kind": "copia", "bitrate": None, "ffmpeg": None}
    else:
        if not recorded.get("wav"):
            raise ValueError("il manifest non indica il WAV del set (output.wav): usa --set-mp3")
        src = resolve_review_audio(str(recorded["wav"]), manifest_dir, root)
        if not src.is_file() and (manifest_dir / _basename(str(recorded["wav"]))).is_file():
            src = manifest_dir / _basename(str(recorded["wav"]))
        if not src.is_file():
            raise FileNotFoundError(f"WAV del set non trovato: {src}")
        actual = sha256_file(src)
        expected = str(recorded.get("wav_sha256") or "").lower()
        if expected and expected != actual:
            raise ValueError(f"sha256 del WAV diverso: atteso {expected}, trovato {actual}")
        enc = choose_encoder(encoder)
        dst = out / "audio" / ("set.wav" if enc == "wav16" else "set.mp3")
        log(f"encoder: {enc} ({SET_MP3_BITRATE} CBR)" if enc == "ffmpeg-mp3" else f"encoder: {enc}")
        encode(src, dst, enc, SET_MP3_BITRATE)
        audio_src = {"kind": "wav", "path": str(src), "path_recorded": recorded["wav"], "sha256": actual,
                     "hash_status": "verificato" if expected else "hash_non_verificabile"}
        enc_info = {"kind": enc, "bitrate": SET_MP3_BITRATE if enc == "ffmpeg-mp3" else None, "ffmpeg": None}
    probe = _probe(dst)
    size = dst.stat().st_size
    dur = float(norm["duration_s"])
    if dst.suffix == ".mp3" and size > SET_MP3_MAX_BYTES:
        log(f"AVVISO: {dst.name} pesa {size / 1e6:.1f} MB (> {SET_MP3_MAX_BYTES / 1e6:.0f} MB)")
    if probe.get("duration_s") is not None and abs(probe["duration_s"] - dur) > 1.0:
        log(f"AVVISO: audio di {probe['duration_s']:.2f} s, il set ne dichiara {dur:.2f}")
    if audio_src["hash_status"] != "verificato":
        log(f"AVVISO audio del set: {audio_src['hash_status']}")

    for t in transitions:
        t["doc_id"] = transition_doc_id(run, t["index"])
    by_slug = {t["slug"]: t for t in tracks}
    for t in transitions:
        for side in ("a", "b"):
            tr = by_slug.get(t.get(side))
            t[f"{side}_name"] = (f"{tr['artist']} – {tr['title']}" if tr and tr["artist"]
                                 else (tr["title"] if tr else str(t.get(side) or "?")))
    audio = {"path": f"audio/{dst.name}", "sha256": sha256_file(dst), "bytes": size, "probe": probe,
             "source": audio_src}
    listing = {
        "schema_version": SCHEMA_OUT,
        "kind": "dj-set",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": run,
        "source_manifest": {"path": str(manifest_path), "sha256": hashlib.sha256(raw).hexdigest(),
                            "schema_version": manifest.get("schema_version"), "normalized_schema": SCHEMA_SET,
                            "title": norm.get("title")},
        "set_manifest": "set_manifest.json",
        "encoder": enc_info,
        "audio": audio,
        "duration_s": dur,
        "db": {"collection": COLLECTION, "transition_doc_id": "<run>__t<NN>", "set_doc_id": set_doc_id(run)},
        "tracks": tracks,
        "transitions": transitions,
        "included": [{"id": "set", "audio": audio["path"], "source_sha256": audio_src["sha256"]}],
        "excluded": excluded,
    }
    (out / "set_manifest.json").write_text(json.dumps(norm, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    (out / "listening_manifest.json").write_text(json.dumps(listing, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf8")
    (out / "index.html").write_text(render_set_html(run, norm, tracks, transitions, audio, dur,
                                                    audio_src["sha256"]), encoding="utf8")
    log(f"{len(tracks)} brani, {len(transitions)} transizioni, {len(excluded)} escluse, "
        f"audio {size / 1e6:.1f} MB -> {out}")
    return listing


def _set_bar(tracks: list[dict], transitions: list[dict], dur: float) -> tuple[str, str]:
    pct = lambda s: 100.0 * max(0.0, min(dur, s)) / dur  # noqa: E731
    parts = []
    for i, t in enumerate(tracks):
        parts.append(f'<div class="band b{i % 2}" style="left:{pct(t["start_s"]):.4f}%;'
                     f'width:{pct(t["end_s"]) - pct(t["start_s"]):.4f}%"></div>')
    for t in tracks:
        if t["start_s"] > SET_TOLERANCE_S:
            parts.append(f'<div class="bound" style="left:{pct(t["start_s"]):.4f}%" '
                         f'title="inizio brano {t["position"]}"></div>')
    for t in transitions:
        left, right = pct(t["start_s"]), pct(t["end_s"])
        parts.append(f'<div class="region f-{t["family"]}" data-index="{t["index"]}" '
                     f'style="left:{left:.4f}%;width:{right - left:.4f}%" '
                     f'title="T{t["index"]:02d} {html.escape(t["label"])}"></div>')
        parts.append(f'<div class="switch" data-switch-s="{t["switch_s"]:.6f}" style="left:{pct(t["switch_s"]):.4f}%" '
                     f'title="switch T{t["index"]:02d}"></div>')
    parts.append('<div class="fill"></div><div class="head"></div>')
    nums = "".join(f'<span style="left:{pct((t["start_s"] + t["end_s"]) / 2):.4f}%">{t["index"]}</span>'
                   for t in transitions)
    return "".join(parts), nums


def _set_track_row(t: dict) -> str:
    name = (f"{html.escape(t['artist'])} – {html.escape(t['title'])}" if t["artist"] else html.escape(t["title"]))
    meta = [f"{_num(t.get('bpm'), '.1f')} BPM", html.escape(t.get("camelot") or "—"), f"entra a {_mmss(t['start_s'])}"]
    return (f'<li><button type="button" class="trk" data-start-s="{t["start_s"]:.6f}" data-end-s="{t["end_s"]:.6f}">'
            f'<span class="pos">{t["position"]}</span><span class="tname">{name}</span>'
            f'<span class="tmeta small muted">{" · ".join(meta)}</span></button></li>')


def _qa_badge(t: dict) -> str:
    v = t.get("qa_verdict")
    if v not in ("warn", "fail"):
        return ""
    tip = html.escape("; ".join(t.get("qa_reasons") or []))
    kept = " (tenuta)" if v == "fail" and t.get("qa_kept_despite_fail") else ""
    return f'<span class="badge qa-{v}" title="{tip}">QA {v}{kept}</span>'


def _set_transition(t: dict) -> str:
    idx = t["index"]
    doc = html.escape(t["doc_id"])
    scores = "".join(f'<label class="chip"><input type="radio" name="score-t{idx}" value="{n}"><span>{n}</span></label>'
                     for n in range(1, 11))
    tags = "".join(f'<label class="chip"><input type="checkbox" value="{k}"><span>{html.escape(v)}</span></label>'
                   for k, v in DJ_TAGS)
    reasons = t.get("qa_reasons") or []
    qa_list = ("<ul class=\"qa-reasons small\">" + "".join(f"<li>{html.escape(r)}</li>" for r in reasons) + "</ul>"
               if t.get("qa_verdict") in ("warn", "fail") and reasons else "")
    span = (f"taglio a {_mmss(t['switch_s'])}" if t["end_s"] - t["start_s"] < 0.05 else
            f"{_mmss(t['start_s'])}–{_mmss(t['end_s'])} · switch {_mmss(t['switch_s'])} · {t['end_s'] - t['start_s']:.1f} s")
    rule = f' · regola <code>{html.escape(str(t["rule"]))}</code>' if t.get("rule") else ""
    return f"""<article class="tr" id="t{idx:02d}" data-doc="{doc}" data-index="{idx}">
  <div class="row thead"><span class="tnum">T{idx:02d}</span><span class="badge f-{t['family']}">{html.escape(t['label'])}</span>{_qa_badge(t)}</div>
  <p class="pair">{html.escape(t['a_name'])} → {html.escape(t['b_name'])}</p>
  <p class="muted small">{span}{rule}</p>
  {qa_list}
  <div class="row buttons">
    <button type="button" data-act="goto">vai alla transizione</button>
    <button type="button" data-act="replay" aria-pressed="false">riascolta</button>
    <label class="loop small"><input type="checkbox" class="loopbox"> loop</label>
  </div>
  <form class="judge">
    <fieldset><legend>Voto</legend><div class="chips">{scores}</div></fieldset>
    <fieldset><legend>Scelte rapide</legend><div class="chips">{tags}</div></fieldset>
    <label class="note-label">Nota<textarea rows="2" name="note" placeholder="cosa si sente nella transizione…"></textarea></label>
    <div class="row between"><button type="submit" class="primary">Salva</button><span class="status small muted">non salvato</span></div>
    <p class="existing small muted"></p>
  </form>
</article>"""


def _set_form(run: str) -> str:
    score = "".join(f'<label class="chip"><input type="radio" name="set-score" value="{n}"><span>{n}</span></label>'
                    for n in range(1, 11))
    smooth = "".join(f'<label class="chip"><input type="radio" name="set-smooth" value="{n}"><span>{n}</span></label>'
                     for n in range(1, 11))
    order = "".join(f'<label class="chip"><input type="radio" name="set-order" value="{k}"><span>{v}</span></label>'
                    for k, v in SET_ORDER_CHOICES)
    return f"""<section class="card" id="set-judge" data-doc="{html.escape(set_doc_id(run))}">
  <h2>Giudizio complessivo del set</h2>
  <form class="judge setform">
    <fieldset><legend>Voto al set</legend><div class="chips">{score}</div></fieldset>
    <fieldset><legend>Scorrevolezza</legend><div class="chips">{smooth}</div></fieldset>
    <fieldset><legend>Ordine dei brani sensato?</legend><div class="chips">{order}</div></fieldset>
    <label class="note-label">Nota<textarea rows="3" name="note" placeholder="impressione generale, punti da rifare…"></textarea></label>
    <div class="row between"><button type="submit" class="primary">Salva giudizio del set</button><span class="status small muted">non salvato</span></div>
    <p class="existing small muted"></p>
  </form>
</section>"""


def render_set_html(run: str, norm: dict, tracks: list[dict], transitions: list[dict], audio: dict,
                    dur: float, source_sha256: str) -> str:
    bar, nums = _set_bar(tracks, transitions, dur)
    fams = [f for f in SET_FAMILY_LABELS if any(t["family"] == f for t in transitions)]
    legend = "".join(f'<span class="lg"><span class="legend-key f-{f}"></span>{SET_FAMILY_LABELS[f]}</span>' for f in fams)
    legend += ('<span class="lg"><span class="legend-key line"></span>switch</span>'
               '<span class="lg"><span class="legend-key bnd"></span>confine brano</span>')
    qa = {}
    for t in transitions:
        qa[t["qa_verdict"]] = qa.get(t["qa_verdict"], 0) + 1
    qa_txt = " · ".join(f"QA {k} {v}" for k, v in sorted(qa.items()))
    data = {
        "run": run, "collection": COLLECTION, "lead": SET_LEAD_S, "duration_s": dur,
        "audio_sha256": audio["sha256"], "source_sha256": source_sha256,
        "set_doc_id": set_doc_id(run),
        "tags": [{"key": k, "label": v} for k, v in DJ_TAGS],
        "order": [{"key": k, "label": v} for k, v in SET_ORDER_CHOICES],
        "tracks": [{"position": t["position"], "start_s": t["start_s"], "end_s": t["end_s"],
                    "name": (f"{t['artist']} – {t['title']}" if t["artist"] else t["title"])} for t in tracks],
        "transitions": [{k: t.get(k) for k in ("index", "doc_id", "type", "family", "label", "a", "b",
                                                 "start_s", "switch_s", "end_s", "qa_verdict")} for t in transitions],
    }
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
    heading = norm.get("title")
    subtitle = f'<p class="small">{html.escape(heading)}</p>' if heading else ""
    rows = "\n".join(_set_transition(t) for t in transitions) or '<p class="muted">Nessuna transizione valida.</p>'
    return (SET_HTML_TEMPLATE
            .replace("__SUBTITLE__", subtitle)
            .replace("__RUN__", html.escape(run))
            .replace("__NTRACKS__", str(len(tracks)))
            .replace("__NTRANS__", str(len(transitions)))
            .replace("__DUR__", _mmss(dur))
            .replace("__QA__", html.escape(qa_txt))
            .replace("__LEGEND__", legend)
            .replace("__AUDIO__", html.escape(audio["path"]))
            .replace("__DURS__", f"{dur:.2f}")
            .replace("__BAR__", bar)
            .replace("__NUMS__", nums)
            .replace("__TRACKS__", "\n".join(_set_track_row(t) for t in tracks))
            .replace("__TRANSITIONS__", rows)
            .replace("__SETFORM__", _set_form(run))
            .replace("__DATA__", payload))


SET_HTML_TEMPLATE = r"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ascolto set DJ</title>
<style>
:root{--bg:#f6f5f2;--surface:#ffffff;--surface2:#faf9f6;--text:#1d1d1b;--muted:#6a6862;--border:#dddad3;--accent:#2f5bd3;--accent-text:#ffffff;--fill:#1d1d1b;--band0:#e7e4dd;--band1:#d9d5cc;--bound:#8d897f;--switch:#1d1d1b;--head:#2f5bd3;--ok:#2f7d4f;--warn:#a5651a;--warn-bg:#fbf0de;--fail:#b3261e;--fail-bg:#fbe4e1;--current:#eef2fd;
--f-blend:#2f6fd3;--f-filter:#1a9a85;--f-filter_chorus:#8a4fd8;--f-echo:#d9761c;--f-cut:#c43d5a;--f-other:#7a776f;color-scheme:light}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141413;--surface:#1f1f1d;--surface2:#191918;--text:#ecebe6;--muted:#a3a19a;--border:#34332f;--accent:#7c9cf5;--accent-text:#0e1220;--fill:#ecebe6;--band0:#2d2c29;--band1:#3a3935;--bound:#8b887f;--switch:#f4f3ee;--head:#7c9cf5;--ok:#6fcf97;--warn:#f0b35a;--warn-bg:#2f2615;--fail:#ff8a80;--fail-bg:#3a1c1a;--current:#1f2740;
--f-blend:#6f9cf2;--f-filter:#3cc7ad;--f-filter_chorus:#b48af0;--f-echo:#f0a052;--f-cut:#ef7790;--f-other:#a3a19a;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#141413;--surface:#1f1f1d;--surface2:#191918;--text:#ecebe6;--muted:#a3a19a;--border:#34332f;--accent:#7c9cf5;--accent-text:#0e1220;--fill:#ecebe6;--band0:#2d2c29;--band1:#3a3935;--bound:#8b887f;--switch:#f4f3ee;--head:#7c9cf5;--ok:#6fcf97;--warn:#f0b35a;--warn-bg:#2f2615;--fail:#ff8a80;--fail-bg:#3a1c1a;--current:#1f2740;
--f-blend:#6f9cf2;--f-filter:#3cc7ad;--f-filter_chorus:#b48af0;--f-echo:#f0a052;--f-cut:#ef7790;--f-other:#a3a19a;color-scheme:dark}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:860px;margin:0 auto;padding:16px}
h1{font-size:1.3rem;margin:.2rem 0}
h2{font-size:1.05rem;margin:0 0 .4rem}
code{font-size:.85em;overflow-wrap:anywhere}
.muted{color:var(--muted)}.small{font-size:.85rem}
p{margin:.25rem 0}
.banner{display:none;background:var(--warn-bg);color:var(--text);border:1px solid var(--warn);border-radius:8px;padding:.6rem .8rem;margin:.8rem 0}
.banner.show{display:block}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px;margin:14px 0}
.player{position:sticky;top:0;z-index:5;background:var(--surface);border:1px solid var(--border);border-radius:0 0 12px 12px;padding:10px 12px;margin:10px -4px 14px;box-shadow:0 2px 8px rgba(0,0,0,.08)}
.setbar{position:relative;height:40px;border-radius:6px;overflow:hidden;cursor:pointer;touch-action:none;background:var(--band0);outline-offset:2px}
.band{position:absolute;top:0;bottom:0}.band.b0{background:var(--band0)}.band.b1{background:var(--band1)}
.bound{position:absolute;top:0;bottom:0;width:0;border-left:1px dashed var(--bound)}
.region{position:absolute;top:6px;bottom:6px;min-width:4px;margin-left:-1px;border-radius:3px;opacity:.9}
.switch{position:absolute;top:0;bottom:0;width:2px;margin-left:-1px;background:var(--switch)}
.fill{position:absolute;left:0;top:0;bottom:0;width:0;background:var(--fill);opacity:.12;pointer-events:none}
.head{position:absolute;top:0;bottom:0;width:4px;margin-left:-2px;left:0;background:var(--head);box-shadow:0 0 0 2px var(--surface);border-radius:2px;pointer-events:none}
.nums{position:relative;height:16px;font-size:.72rem;color:var(--muted)}
.nums span{position:absolute;transform:translateX(-50%);top:1px}
.f-blend{background:var(--f-blend)}.f-filter{background:var(--f-filter)}.f-filter_chorus{background:var(--f-filter_chorus)}
.f-echo{background:var(--f-echo)}.f-cut{background:var(--f-cut)}.f-other{background:var(--f-other)}
.badge.f-blend,.badge.f-filter,.badge.f-filter_chorus,.badge.f-echo,.badge.f-cut,.badge.f-other{color:#fff;border-color:transparent}
:root[data-theme="dark"] .badge[class*=" f-"]{color:#111}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .badge[class*=" f-"]{color:#111}}
.timerow{display:flex;justify-content:space-between;gap:8px;align-items:baseline;margin:.3rem 0}
.time{font-variant-numeric:tabular-nums;font-weight:600}
.now{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0;text-align:right}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:.4rem 0}
.between{justify-content:space-between}
button{font:inherit;padding:.45rem .7rem;border-radius:8px;border:1px solid var(--border);background:var(--surface);color:var(--text);cursor:pointer;min-height:40px}
button.primary{background:var(--accent);color:var(--accent-text);border-color:var(--accent)}
button:disabled{opacity:.6;cursor:default}
button[aria-pressed="true"]{background:var(--accent);color:var(--accent-text);border-color:var(--accent)}
.buttons button{flex:1 1 auto}
.legend{display:flex;flex-wrap:wrap;gap:4px 12px;margin:.4rem 0}
.lg{display:inline-flex;align-items:center;gap:5px}
.legend-key{display:inline-block;width:14px;height:10px;border-radius:2px}
.legend-key.line{width:2px;height:14px;background:var(--switch)}
.legend-key.bnd{width:0;height:14px;border-left:1px dashed var(--bound);border-radius:0}
.tracks{list-style:none;margin:0;padding:0}
.tracks li{margin:0}
.trk{display:grid;grid-template-columns:2rem 1fr;gap:0 6px;width:100%;text-align:left;border:0;border-bottom:1px solid var(--border);border-radius:0;background:transparent;padding:.5rem .2rem}
.trk:hover{background:var(--surface2)}
.trk.current{background:var(--current)}
.trk .pos{grid-row:span 2;font-weight:600;color:var(--muted);font-variant-numeric:tabular-nums}
.tname{overflow-wrap:anywhere;font-weight:500}
.tmeta{grid-column:2}
.tr{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px 14px;margin:12px 0;scroll-margin-top:170px}
.tr.current{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.thead{margin:0 0 .2rem}
.tnum{font-weight:700;font-variant-numeric:tabular-nums}
.pair{font-weight:500;overflow-wrap:anywhere}
.badge{display:inline-block;padding:.15rem .55rem;border-radius:999px;font-size:.8rem;font-weight:600;border:1px solid var(--border)}
.badge.qa-warn{background:var(--warn-bg);color:var(--warn);border-color:var(--warn);cursor:help}
.badge.qa-fail{background:var(--fail-bg);color:var(--fail);border-color:var(--fail);cursor:help}
.qa-reasons{color:var(--muted);margin:.2rem 0;padding-left:1.1rem;overflow-wrap:anywhere}
.loop{display:inline-flex;align-items:center;gap:4px;min-height:40px;padding:0 .3rem}
fieldset{border:0;padding:0;margin:.6rem 0;min-width:0}
legend{font-weight:600;font-size:.9rem;margin-bottom:.3rem}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{position:relative}
.chip input{position:absolute;opacity:0;inset:0}
.chip span{display:inline-block;padding:.35rem .6rem;border:1px solid var(--border);border-radius:999px;min-width:2.2rem;text-align:center;font-size:.9rem;cursor:pointer}
.chip input:checked+span{background:var(--accent);color:var(--accent-text);border-color:var(--accent)}
.chip input:focus-visible+span{outline:2px solid var(--accent);outline-offset:2px}
.note-label{display:block;font-weight:600;font-size:.9rem}
textarea{display:block;width:100%;margin-top:.3rem;font:inherit;padding:.5rem;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);resize:vertical}
.status.ok{color:var(--ok)}.status.err{color:var(--fail)}
</style>
</head>
<body>
<main>
<header>
  <h1>Ascolto set DJ</h1>
  __SUBTITLE__
  <p class="muted small">Run <code>__RUN__</code> · __NTRACKS__ brani · __NTRANS__ transizioni · __DUR__ · __QA__</p>
  <div class="legend small muted">__LEGEND__</div>
  <div class="banner" id="banner"></div>
  <div class="row"><span class="small muted" id="dbstate">Collegamento al database in corso…</span><button type="button" id="copy" class="small">Copia JSON</button></div>
</header>
<section class="player" aria-label="Player del set">
  <audio id="audio" preload="metadata" src="__AUDIO__"></audio>
  <div class="setbar" id="bar" role="slider" tabindex="0" aria-label="Posizione nel set" aria-valuemin="0" aria-valuemax="__DURS__" aria-valuenow="0">__BAR__</div>
  <div class="nums" aria-hidden="true">__NUMS__</div>
  <div class="timerow small"><span class="time" id="time">0:00 / __DUR__</span><span class="now muted" id="now"></span></div>
  <div class="row buttons">
    <button type="button" id="back">−10 s</button>
    <button type="button" id="play">▶︎ / ❚❚</button>
    <button type="button" id="fwd">+10 s</button>
  </div>
</section>
<section class="card">
  <h2>Tracklist</h2>
  <ol class="tracks">
__TRACKS__
  </ol>
</section>
<h2>Transizioni</h2>
__TRANSITIONS__
__SETFORM__
</main>
<script id="set-data" type="application/json">__DATA__</script>
<script>
(function(){
"use strict";
const CFG = JSON.parse(document.getElementById("set-data").textContent);
const LS_KEY = "musicmix:judgments:" + CFG.run;
const trByDoc = new Map(CFG.transitions.map(t => [t.doc_id, t]));
const state = {db: null, saved: {}, local: {}};
const $ = (s, el) => (el || document).querySelector(s);
const fmt = s => { s = Math.max(0, s || 0); const m = Math.floor(s / 60); const r = Math.floor(s % 60); return m + ":" + String(r).padStart(2, "0"); };
const sleep = ms => new Promise(r => setTimeout(r, ms));
function lsRead(){ try { return JSON.parse(localStorage.getItem(LS_KEY) || "{}") || {}; } catch (e) { return {}; } }
function lsWrite(obj){ try { localStorage.setItem(LS_KEY, JSON.stringify(obj)); return true; } catch (e) { return false; } }
function banner(msg){ const b = $("#banner"); b.textContent = msg; b.classList.toggle("show", !!msg); }

// ---------------------------------------------------------------- player
const audio = $("#audio"), bar = $("#bar"), fill = $(".fill", bar), head = $(".head", bar);
const timeEl = $("#time"), nowEl = $("#now");
const trkBtns = Array.from(document.querySelectorAll(".trk"));
const trCards = Array.from(document.querySelectorAll(".tr"));
const dur = () => (isFinite(audio.duration) && audio.duration > 0) ? audio.duration : CFG.duration_s;
let win = null;  // {from, to, loop, idx}: active "riascolta" window
let lastTrack = -1, lastTr = -1;
function setWin(w){
  win = w;
  trCards.forEach(c => { const b = $('[data-act="replay"]', c); b.setAttribute("aria-pressed", String(!!w && w.idx === Number(c.dataset.index))); });
}
function seek(t, keepWin){
  if (!keepWin) setWin(null);
  audio.currentTime = Math.min(Math.max(0, t), Math.max(0, dur() - 0.05));
  paint();
}
function play(){ audio.play().catch(() => {}); }
function paint(){
  const t = audio.currentTime, d = dur();
  const p = Math.min(100, 100 * t / d);
  fill.style.width = p + "%"; head.style.left = p + "%";
  timeEl.textContent = fmt(t) + " / " + fmt(d);
  bar.setAttribute("aria-valuenow", t.toFixed(1));
  bar.setAttribute("aria-valuetext", fmt(t));
  let ti = CFG.tracks.findIndex(x => t >= x.start_s && t < x.end_s);
  if (ti < 0 && CFG.tracks.length && t >= CFG.tracks[CFG.tracks.length - 1].start_s) ti = CFG.tracks.length - 1;
  if (ti !== lastTrack) {
    lastTrack = ti;
    trkBtns.forEach((b, i) => b.classList.toggle("current", i === ti));
    nowEl.textContent = ti >= 0 ? CFG.tracks[ti].position + ". " + CFG.tracks[ti].name : "";
  }
  const tr = CFG.transitions.findIndex(x => t >= x.start_s - CFG.lead && t <= x.end_s + CFG.lead);
  if (tr !== lastTr) { lastTr = tr; trCards.forEach((c, i) => c.classList.toggle("current", i === tr)); }
}
function guard(){
  if (win && audio.currentTime >= win.to) {
    if (win.loop) audio.currentTime = win.from;
    else { audio.pause(); setWin(null); }
  }
}
function tick(){ guard(); paint(); if (!audio.paused) requestAnimationFrame(tick); }
audio.addEventListener("play", () => requestAnimationFrame(tick));
["timeupdate", "seeked", "loadedmetadata", "pause"].forEach(ev => audio.addEventListener(ev, () => { guard(); paint(); }));
const fromPointer = ev => { const r = bar.getBoundingClientRect(); seek(((ev.clientX - r.left) / r.width) * dur()); };
bar.addEventListener("pointerdown", ev => { bar.setPointerCapture(ev.pointerId); fromPointer(ev); });
bar.addEventListener("pointermove", ev => { if (bar.hasPointerCapture(ev.pointerId)) fromPointer(ev); });
bar.addEventListener("keydown", ev => {
  if (ev.key === "ArrowLeft") { seek(audio.currentTime - 5); ev.preventDefault(); }
  if (ev.key === "ArrowRight") { seek(audio.currentTime + 5); ev.preventDefault(); }
  if (ev.key === " ") { audio.paused ? play() : audio.pause(); ev.preventDefault(); }
});
$("#back").addEventListener("click", () => seek(audio.currentTime - 10));
$("#fwd").addEventListener("click", () => seek(audio.currentTime + 10));
$("#play").addEventListener("click", () => { audio.paused ? play() : audio.pause(); });
trkBtns.forEach(b => b.addEventListener("click", () => { seek(Number(b.dataset.startS)); play(); }));
trCards.forEach(card => {
  const t = CFG.transitions.find(x => x.index === Number(card.dataset.index));
  const loopBox = $(".loopbox", card);
  $('[data-act="goto"]', card).addEventListener("click", () => { seek(t.start_s - CFG.lead); play(); });
  $('[data-act="replay"]', card).addEventListener("click", () => {
    if (win && win.idx === t.index) { audio.pause(); setWin(null); return; }
    const from = Math.max(0, t.start_s - CFG.lead), to = Math.min(dur(), t.end_s + CFG.lead);
    seek(from); setWin({from: from, to: to, loop: loopBox.checked, idx: t.index}); play();
  });
  loopBox.addEventListener("change", () => { if (win && win.idx === t.index) win.loop = loopBox.checked; });
});

// ---------------------------------------------------------------- forms
function readTr(form){
  const s = $("input[type=radio]:checked", form);
  return {score: s ? Number(s.value) : null,
          tags: Array.from(form.querySelectorAll("input[type=checkbox]:checked")).map(i => i.value),
          note: form.note.value.trim()};
}
function fillTr(form, j){
  form.querySelectorAll("input[type=radio]").forEach(i => { i.checked = Number(i.value) === j.score; });
  form.querySelectorAll("input[type=checkbox]").forEach(i => { i.checked = (j.tags || []).includes(i.value); });
  form.note.value = j.note || "";
}
function readSet(form){
  const v = n => { const c = $('input[name="' + n + '"]:checked', form); return c ? c.value : null; };
  const s = v("set-score"), m = v("set-smooth");
  return {score: s ? Number(s) : null, smoothness: m ? Number(m) : null, order: v("set-order"), note: form.note.value.trim()};
}
function fillSet(form, j){
  form.querySelectorAll('input[name="set-score"]').forEach(i => { i.checked = Number(i.value) === j.score; });
  form.querySelectorAll('input[name="set-smooth"]').forEach(i => { i.checked = Number(i.value) === j.smoothness; });
  form.querySelectorAll('input[name="set-order"]').forEach(i => { i.checked = i.value === j.order; });
  form.note.value = j.note || "";
}
function setStatus(form, text, cls){ const s = $(".status", form); s.textContent = text; s.className = "status small " + (cls || "muted"); }
function describe(j){
  const note = j.note ? " · “" + j.note + "”" : "";
  if (j.type === "set") {
    const o = CFG.order.find(x => x.key === j.order);
    return "voto " + (j.score != null ? j.score + "/10" : "—") + " · scorrevolezza " + (j.smoothness != null ? j.smoothness + "/10" : "—") +
      " · ordine sensato: " + (o ? o.label : "—") + note;
  }
  const labels = (j.tags || []).map(k => (CFG.tags.find(t => t.key === k) || {label: k}).label);
  return (j.score != null ? j.score + "/10" : "senza voto") + (labels.length ? " · " + labels.join(", ") : "") + note;
}
function showExisting(form, j, where){
  const p = $(".existing", form);
  if (!j) { p.textContent = ""; return; }
  const when = j.saved_at ? new Date(j.saved_at).toLocaleString("it-IT") : "";
  p.textContent = "Giudizio " + where + ": " + describe(j) + (when ? " · " + when : "");
}
function target(docId){
  if (trByDoc.has(docId)) {
    const card = document.querySelector('.tr[data-doc="' + CSS.escape(docId) + '"]');
    if (!card) return null;
    const form = $("form", card);
    return {form: form, fill: j => fillTr(form, j)};
  }
  if (docId === CFG.set_doc_id) { const form = $("#set-judge form"); return {form: form, fill: j => fillSet(form, j)}; }
  return null;
}
function applyJudgment(docId, j, where){
  const t = target(docId);
  if (!t) return;
  showExisting(t.form, j, where);
  if (j && !t.form._dirty && !t.form._saving) { t.fill(j); setStatus(t.form, "salvato", "ok"); }
}
const queue = {};
async function save(docId, form, makeBody){
  const q = queue[docId] || (queue[docId] = {busy: false, pending: false});
  if (q.busy) { q.pending = true; setStatus(form, "in coda…"); return; }
  q.busy = true; form._saving = true;
  const btn = $("button[type=submit]", form); btn.disabled = true;
  try {
    do {
      q.pending = false;
      const body = Object.assign(makeBody(), {doc_id: docId, position_s: Math.round(audio.currentTime * 1000) / 1000,
                                              saved_at: new Date().toISOString()});
      form._dirty = false;
      setStatus(form, "salvataggio…");
      if (state.db) {
        try {
          await state.db.collection(CFG.collection).doc(docId).set(body);
          state.saved[docId] = body;
          setStatus(form, "salvato nel database", "ok");
        } catch (e) {
          const code = e && e.code || "errore";
          state.local[docId] = body; lsWrite(state.local);
          setStatus(form, "non salvato nel database (" + code + "): copia locale", "err");
          if (code === "invalid_argument") banner("Non hai permesso di scrittura sul database: i giudizi restano in questo browser. Usa “Copia JSON”.");
        }
      } else {
        state.local[docId] = body;
        const ok = lsWrite(state.local);
        setStatus(form, ok ? "salvato in questo browser" : "non salvato (storage bloccato): usa Copia JSON", ok ? "ok" : "err");
      }
      showExisting(form, state.saved[docId] || state.local[docId], state.saved[docId] ? "salvato" : "locale");
    } while (q.pending);
  } finally {
    q.busy = false; form._saving = false; btn.disabled = false;
  }
}
trCards.forEach(card => {
  const t = trByDoc.get(card.dataset.doc);
  const f = $("form", card);
  f.addEventListener("input", () => { f._dirty = true; setStatus(f, "modifiche non salvate"); });
  f.addEventListener("submit", ev => {
    ev.preventDefault();
    save(t.doc_id, f, () => Object.assign({
      run: CFG.run, type: "transizione", transition_index: t.index, transition_type: t.type, family: t.family,
      label: t.label, a: t.a, b: t.b, set_start_s: t.start_s, set_switch_s: t.switch_s, set_end_s: t.end_s,
      qa_verdict: t.qa_verdict, source_sha256: CFG.source_sha256, audio_sha256: CFG.audio_sha256
    }, readTr(f)));
  });
});
(function(){
  const f = $("#set-judge form");
  f.addEventListener("input", () => { f._dirty = true; setStatus(f, "modifiche non salvate"); });
  f.addEventListener("submit", ev => {
    ev.preventDefault();
    save(CFG.set_doc_id, f, () => Object.assign({run: CFG.run, type: "set", n_tracks: CFG.tracks.length,
      n_transitions: CFG.transitions.length, source_sha256: CFG.source_sha256, audio_sha256: CFG.audio_sha256}, readSet(f)));
  });
})();

// ---------------------------------------------------------------- copy JSON
$("#copy").addEventListener("click", async () => {
  const all = Object.assign({}, state.local, state.saved);
  const text = JSON.stringify({run: CFG.run, collection: CFG.collection, judgments: all}, null, 2);
  try { await navigator.clipboard.writeText(text); $("#copy").textContent = "Copiato"; }
  catch (e) {
    const ta = document.createElement("textarea"); ta.value = text; ta.rows = 8;
    $("header").appendChild(ta); ta.select();
    $("#copy").textContent = "Seleziona e copia il testo qui sotto";
  }
  setTimeout(() => { $("#copy").textContent = "Copia JSON"; }, 3000);
});

// ---------------------------------------------------------------- db
state.local = lsRead();
Object.entries(state.local).forEach(([id, j]) => applyJudgment(id, j, "locale"));
paint();
async function getDb(){
  const deadline = Date.now() + 10000;
  while (!(window.claude && typeof window.claude.use === "function") && Date.now() < deadline) await sleep(100);
  if (!(window.claude && typeof window.claude.use === "function")) return null;
  try { return await Promise.race([window.claude.use("db"), sleep(12000).then(() => null)]); }
  catch (e) { return null; }
}
getDb().then(db => {
  if (!db) {
    $("#dbstate").textContent = "Database non disponibile: salvataggio in questo browser.";
    banner("Database dell'Artifact non disponibile in questa vista: i giudizi vengono salvati solo in questo browser. Usa “Copia JSON” per conservarli.");
    return;
  }
  state.db = db;
  $("#dbstate").textContent = "Giudizi salvati nel database dell'Artifact (" + CFG.collection + ").";
  try {
    db.collection(CFG.collection).onSnapshot(snap => {
      snap.docs.forEach(d => {
        const j = d.data();
        if (!j || j.run !== CFG.run || !(trByDoc.has(d.id) || d.id === CFG.set_doc_id)) return;
        state.saved[d.id] = j;
        applyJudgment(d.id, j, "salvato");
      });
    }, err => {
      $("#dbstate").textContent = "Lettura dal database interrotta (" + (err && err.code || "errore") + ").";
    });
  } catch (e) {
    $("#dbstate").textContent = "Lettura dal database non riuscita.";
  }
});
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- cli

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", type=Path,
                   help="manifest musicmix.twenty-second-export.1 o review_manifest musicmix.psychology-review.1 "
                        "o musicmix.dj-review.1 o set.json (musicmix.dj_set.1 / musicmix.dj-set.1)")
    p.add_argument("--out", type=Path, required=True, help="cartella di output")
    p.add_argument("--run", help="nome della run (default: cartella del manifest)")
    p.add_argument("--limit", type=int, help="usa solo le prime N clip")
    p.add_argument("--encoder", default="auto", choices=["auto", "ffmpeg-mp3", "soundfile-mp3", "wav16"])
    p.add_argument("--overwrite", action="store_true", help="consenti una cartella di output non vuota")
    p.add_argument("--repo-root", type=Path, help="root per gli audio_path relativi (default: MUSICMIX_ROOT o questo checkout)")
    p.add_argument("--set-mp3", type=Path,
                   help="solo set DJ: mp3 già codificato da usare (default: set_128k.mp3 accanto al set.json se "
                        "esiste, altrimenti codifica output.wav a 128 kbps CBR)")
    p.add_argument("--blind", action="store_true",
                   help="pagine DJ a versioni e approvazione del set: tecnica, warn QA e motivazioni nascosti fino al "
                        "voto (Margulis 2014 p. 15); il torneo dei cue è sempre alla cieca")
    p.add_argument("--kind", default="auto", choices=KINDS,
                   help="set-approval: approvazione del set intero da un set.json (musicmix.set-approval.1); "
                        "cue-tournament: torneo dei cue da un manifest musicmix.cue-variants.1 (vedi tools/listening_pages.py)")
    p.add_argument("--seed", type=int, default=0, help="torneo dei cue: seme aggiuntivo dell'ordine casuale")
    p.add_argument("--set-bitrate", help="approvazione del set: bitrate MP3 (default: il più alto <= 128k sotto il limite)")
    a = p.parse_args(argv)
    listing = build(a.manifest, a.out, run=a.run, limit=a.limit, encoder=a.encoder, overwrite=a.overwrite,
                    root=a.repo_root, set_mp3=a.set_mp3, blind=a.blind, kind=a.kind, seed=a.seed,
                    set_bitrate=a.set_bitrate)
    return 0 if listing["included"] else 1


if __name__ == "__main__":
    sys.exit(main())
