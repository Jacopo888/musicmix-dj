"""Risoluzione verificata dello stem vocale Demucs di un brano.

Il percorso dello stem non si indovina dal nome (``<cartella del JSON>/a_vocals.wav``): si legge dalla
provenienza registrata dalla fase vocals (``tools/analyze_vocals.py`` / ``musicmix.vocal_boundaries``):
``separator.reused_stem_path`` (ASR rifatto su uno stem esistente), ``separator.output_path``,
``backend.audio_path``. I percorsi registrati (anche Windows) passano da :func:`musicmix.paths.localize`.

Verifiche (errore :class:`StemError` con il motivo se una non torna):

* fase vocals con separazione Demucs completa (``source_kind == "separated_vocals"``,
  ``separator.status == "complete"``), non il ripiego ASR sul mix;
* identità della sorgente: ``separator.input_sha256`` == ``track_id`` del JSON vocals == ``track_id``
  della voce di catalogo (se data);
* il file esiste, ha la frequenza del JSON (48 kHz canonici) e la stessa durata in campioni del
  brano canonico (Demucs taglia/allunga alla durata della sorgente);
* se la provenienza registra lo SHA-256 dello stem (``reused_stem_sha256``/``output_sha256``), coincide.

Override additivo (``runs/pool_bridge_v1/vocals_overrides.json``, oppure il file di ``$MUSICMIX_VOCALS_OVERRIDES``):
per le voci di catalogo il cui JSON vocals congelato non ha uno stem utilizzabile (es. ``reminder`` e
``too_many_nights``, fase vocals rifatta in ``runs/pool_bridge_v1/vocals_redo/<slug>/``) il risolutore usa il
JSON indicato, ma solo se il suo ``track_id`` coincide con quello della voce e (se dato) lo SHA-256 del JSON
torna. I JSON congelati non si modificano; un ``vocals_json=`` esplicito ha la precedenza.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from .paths import localize, repo_root


class StemError(RuntimeError):
    """Lo stem vocale registrato manca o non corrisponde al brano."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def overrides_file() -> Path:
    env = os.environ.get("MUSICMIX_VOCALS_OVERRIDES")
    return Path(env) if env else repo_root() / "runs" / "pool_bridge_v1" / "vocals_overrides.json"


def vocals_json_override(entry: Mapping[str, Any]) -> Optional[Path]:
    """JSON vocals sostitutivo per la voce di catalogo ``entry`` (o ``None``); :class:`StemError` se l'override non torna."""
    f = overrides_file()
    slug = entry.get("slug")
    if not slug or not f.is_file():
        return None
    rec = (json.loads(f.read_text(encoding="utf-8")).get("overrides") or {}).get(slug)
    if not rec:
        return None
    vj = localize(rec["vocals_json"])
    if not vj.is_absolute():
        vj = repo_root() / vj
    if not vj.is_file():
        raise StemError(f"{slug}: JSON vocals di override assente ({vj})")
    if rec.get("vocals_json_sha256") and _sha256(vj) != rec["vocals_json_sha256"]:
        raise StemError(f"{slug}: SHA-256 del JSON vocals di override diverso dal registrato ({vj})")
    want = entry.get("track_id") or rec.get("track_id")
    got = json.loads(vj.read_text(encoding="utf-8")).get("track_id")
    if want and got != want:
        raise StemError(f"{slug}: l'override e' di un altro audio (track_id {str(got)[:12]} != {str(want)[:12]})")
    return vj


def vocals_json_for(entry: Mapping[str, Any]) -> Path:
    """JSON vocals da leggere per la voce di catalogo: l'override se c'e', altrimenti ``vocals_path`` (localizzato)."""
    return vocals_json_override(entry) or localize(entry["vocals_path"])


def _load(source: Union[str, Path, Mapping[str, Any]]) -> tuple[dict, Optional[dict], Optional[Path]]:
    """(documento vocals, voce di catalogo o None, percorso del JSON vocals o None)."""
    if isinstance(source, Mapping):
        if "vocals_path" in source and "words" not in source:
            vj = vocals_json_for(source)
            if not vj.is_file():
                raise StemError(f"{source.get('slug')}: JSON vocals assente ({vj})")
            return json.loads(vj.read_text(encoding="utf-8")), dict(source), vj
        return dict(source), None, None
    vj = localize(source)
    if not vj.is_file():
        raise StemError(f"JSON vocals assente ({vj})")
    return json.loads(vj.read_text(encoding="utf-8")), None, vj


def vocal_stem_record(source: Union[str, Path, Mapping[str, Any]], *, vocals_json: Union[str, Path, None] = None,
                      check_sha: bool = True) -> dict[str, Any]:
    """Stem verificato e provenienza: ``{path, recorded_as, checks, model, checkpoint_sha256, input_sha256}``.

    ``source``: voce di catalogo (con ``vocals_path``, ``track_id``...), percorso del JSON vocals o il
    documento stesso. ``vocals_json`` sostituisce il JSON indicato dalla voce di catalogo (es. ASR rifatto
    in ``runs/asr_language_v1``), mantenendo la verifica d'identità contro la voce.
    """
    doc, entry, vj = _load(source)
    if vocals_json is not None:
        vj = localize(vocals_json)
        doc = json.loads(vj.read_text(encoding="utf-8"))
    name = (entry or {}).get("slug") or (str(vj) if vj else "vocals")
    sep = doc.get("separator") or {}
    if doc.get("source_kind") != "separated_vocals" or sep.get("status") != "complete":
        raise StemError(f"{name}: la fase vocals non ha separato la voce (source_kind={doc.get('source_kind')!r}, "
                        f"separator.status={sep.get('status')!r}): nessuno stem")
    tid = doc.get("track_id")
    if sep.get("input_sha256") and tid and sep["input_sha256"] != tid:
        raise StemError(f"{name}: lo stem viene da un audio diverso (separator.input_sha256 {sep['input_sha256'][:12]} "
                        f"!= track_id {tid[:12]})")
    if entry is not None and entry.get("track_id") and tid and entry["track_id"] != tid:
        raise StemError(f"{name}: il JSON vocals è di un altro audio (track_id {tid[:12]} != catalogo {entry['track_id'][:12]})")
    be = doc.get("backend") or {}
    recorded = [("separator.reused_stem_path", sep.get("reused_stem_path")), ("separator.output_path", sep.get("output_path")),
                ("backend.audio_path", be.get("audio_path"))]
    recorded = [(k, v) for k, v in recorded if v]
    if not recorded:
        raise StemError(f"{name}: nessun percorso dello stem registrato nella fase vocals")
    tried = []
    path = key = None
    for k, v in recorded:
        p = localize(v)
        tried.append(f"{k}={p}")
        if p.is_file():
            path, key = p, k
            break
    if path is None:
        raise StemError(f"{name}: stem vocale registrato assente su questa macchina ({'; '.join(tried)})")
    import soundfile as sf

    info = sf.info(str(path))
    sr, n = doc.get("sample_rate"), doc.get("duration_samples")
    if entry is not None:
        sr = entry.get("sample_rate", sr)
        n = entry.get("duration_samples", n)
    if sr and int(info.samplerate) != int(sr):
        raise StemError(f"{name}: stem a {info.samplerate} Hz, brano a {sr} Hz ({path})")
    if n is not None and int(info.frames) != int(n):
        raise StemError(f"{name}: stem di {info.frames} campioni, brano di {n} ({path})")
    checks = {"separated": True, "input_sha256_matches_track": bool(sep.get("input_sha256")),
              "catalog_track_id_matches": entry is not None and bool(entry.get("track_id")),
              "sample_rate": int(info.samplerate), "frames": int(info.frames), "stem_sha256_checked": False}
    want = sep.get("reused_stem_sha256") if key == "separator.reused_stem_path" else sep.get("output_sha256")
    if check_sha and want:
        got = _sha256(path)
        if got != want:
            raise StemError(f"{name}: SHA-256 dello stem {got[:12]} != registrato {want[:12]} ({path})")
        checks["stem_sha256_checked"] = True
    ck = sep.get("checkpoint_files") or []
    return {"path": path, "recorded_as": key, "checks": checks, "model": sep.get("model"),
            "checkpoint_sha256": [c.get("sha256") for c in ck], "input_sha256": sep.get("input_sha256"),
            "vocals_json": str(vj) if vj else None}


def vocal_stem_path(source: Union[str, Path, Mapping[str, Any]], **kw: Any) -> Path:
    """Percorso verificato dello stem vocale (vedi :func:`vocal_stem_record`); :class:`StemError` se non torna."""
    return vocal_stem_record(source, **kw)["path"]
