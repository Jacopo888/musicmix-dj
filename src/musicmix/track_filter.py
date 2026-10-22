"""Ammissione dei brani alla selezione: lista di esclusione esplicita e lingua cantata.

Regole (utente, 21/10/2026): solo brani cantati in inglese o italiano (configurabile in
``configs/track_languages.json``); i brani strumentali o senza voce rilevabile restano ammessi;
i brani in ``configs/excluded_tracks.json`` sono esclusi sempre. Il filtro non modifica cataloghi
ne' JSON delle run: restituisce i brani tenuti e, per gli esclusi, il motivo da registrare.

Lingua di un brano, in ordine di precedenza:
1. campo ``language`` della voce di catalogo (dichiarato nella richiesta del ponte o a mano);
2. JSON vocals prodotto con la selezione automatica della lingua (``language_selection`` di
   ``tools/analyze_vocals.py``): non i JSON storici di ``small.en``, la cui lingua e' forzata;
3. indici di rilevamento (``runs/asr_language_v1/*_languages.json``, faster-whisper-small
   ``detect_language`` sullo stem vocale o, se assente, sul mix);
4. altrimenti sconosciuta: ammesso, segnalato.

Sotto ``min_probability`` [H] la lingua e' "incerta" e il brano resta ammesso (segnalato).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_PATH = ROOT / "configs" / "excluded_tracks.json"
POLICY_PATH = ROOT / "configs" / "track_languages.json"
DEFAULT_POLICY = {"allowed_languages": ["en", "it"], "min_probability": 0.5, "language_indexes": []}


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_excluded(path: Optional[Path] = None) -> dict[str, str]:
    """slug -> motivo, da ``configs/excluded_tracks.json`` (vuoto se il file manca)."""
    p = Path(path) if path else EXCLUDED_PATH
    if not p.is_file():
        return {}
    tracks = _read(p).get("tracks") or {}
    if isinstance(tracks, list):
        return {str(s): "lista di esclusione" for s in tracks}
    return {str(s): str((v or {}).get("reason") if isinstance(v, Mapping) else v or "lista di esclusione")
            for s, v in tracks.items()}


def load_policy(path: Optional[Path] = None) -> dict[str, Any]:
    p = Path(path) if path else POLICY_PATH
    doc = dict(DEFAULT_POLICY)
    if p.is_file():
        doc.update({k: v for k, v in _read(p).items() if k in DEFAULT_POLICY})
    return doc


def load_language_index(paths: Iterable[str | Path], root: Path = ROOT) -> dict[str, dict[str, Any]]:
    """slug -> {language, probability, source} dagli indici di rilevamento (i primi hanno precedenza)."""
    out: dict[str, dict[str, Any]] = {}
    for raw in paths:
        p = Path(raw)
        p = p if p.is_absolute() else root / p
        if not p.is_file():
            continue
        doc = _read(p)
        for row in doc.get("tracks") or []:
            slug = row.get("slug")
            if not slug or slug in out or row.get("status") not in (None, "ok"):
                continue
            out[slug] = {"language": row.get("language"), "probability": row.get("probability"),
                         "source": f"{p.relative_to(root) if p.is_relative_to(root) else p}: {row.get('source') or 'detect_language'}",
                         "speech_seconds": row.get("speech_seconds")}
    return out


def language_from_vocals(doc: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    """Lingua da un JSON vocals solo se e' stata scelta o rilevata (non forzata a ``en`` come in ``small.en``)."""
    if not doc:
        return None
    sel = doc.get("language_selection")
    if not isinstance(sel, Mapping):
        return None
    det = sel.get("detection") or {}
    if sel.get("source") == "requested":
        return {"language": sel.get("language"), "probability": None, "source": "vocals JSON (lingua richiesta)"}
    if det.get("language") is None and det.get("n_windows") == 0:
        return {"language": None, "probability": None, "source": "vocals JSON: nessun parlato rilevato",
                "speech_seconds": det.get("speech_seconds")}
    if det.get("language"):
        return {"language": det["language"], "probability": det.get("probability"),
                "source": "vocals JSON (detect_language sullo stem)", "speech_seconds": det.get("speech_seconds")}
    return None


class TrackFilter:
    """Ammissione dei brani; ``check`` per un brano, ``split`` per un insieme."""

    def __init__(self, *, allowed_languages: Optional[Iterable[str]] = None, min_probability: Optional[float] = None,
                 excluded: Optional[Mapping[str, str]] = None, language_index: Optional[Mapping[str, Mapping[str, Any]]] = None,
                 policy_path: Optional[Path] = None, excluded_path: Optional[Path] = None, root: Path = ROOT):
        pol = load_policy(policy_path)
        self.allowed = {str(x).lower() for x in (allowed_languages if allowed_languages is not None else pol["allowed_languages"])}
        self.min_probability = float(min_probability if min_probability is not None else pol["min_probability"])
        self.excluded = dict(excluded) if excluded is not None else load_excluded(excluded_path)
        self.index = dict(language_index) if language_index is not None else load_language_index(pol["language_indexes"], root)

    def language(self, slug: str, entry: Optional[Mapping[str, Any]] = None,
                 vocals_doc: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        if entry and entry.get("language"):
            return {"language": str(entry["language"]).lower(), "probability": None, "source": "catalogo (campo language)"}
        rec = language_from_vocals(vocals_doc)
        if rec is not None:
            return rec
        if slug in self.index:
            return dict(self.index[slug])
        return {"language": None, "probability": None, "source": "sconosciuta (nessun rilevamento disponibile)"}

    def check(self, slug: str, entry: Optional[Mapping[str, Any]] = None,
              vocals_doc: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        lang = self.language(slug, entry, vocals_doc)
        row = {"slug": slug, "language": lang.get("language"), "probability": lang.get("probability"),
               "language_source": lang.get("source")}
        if slug in self.excluded:
            return {**row, "admitted": False, "rule": "excluded_list", "why": f"lista di esclusione: {self.excluded[slug]}"}
        code, p = lang.get("language"), lang.get("probability")
        if code is None:
            # [H] "senza voce" solo se il VAD ha girato sullo stem vocale; sul mix intero Silero e' poco affidabile
            src = str(lang.get("source"))
            known = "nessun parlato" in src or (lang.get("speech_seconds") == 0 and "mix" not in src)
            return {**row, "admitted": True, "rule": "no_voice" if known else "language_unknown",
                    "why": "senza voce rilevabile: ammesso" if known else "lingua sconosciuta: ammesso (da verificare)"}
        if code in self.allowed:
            return {**row, "admitted": True, "rule": "language_allowed", "why": f"lingua {code} ammessa"}
        if p is not None and float(p) < self.min_probability:
            return {**row, "admitted": True, "rule": "language_uncertain",
                    "why": f"lingua {code} incerta (p={float(p):.2f} < {self.min_probability}): ammesso, da verificare"}
        pt = "" if p is None else f" p={float(p):.2f}"
        return {**row, "admitted": False, "rule": "language_not_allowed",
                "why": f"lingua cantata {code}{pt} non in {sorted(self.allowed)} (utente 21/10/2026)"}

    def split(self, slugs: Iterable[str], entries: Optional[Mapping[str, Mapping[str, Any]]] = None,
              vocals_docs: Optional[Mapping[str, Mapping[str, Any]]] = None) -> tuple[list[str], list[dict[str, Any]]]:
        """(slug ammessi nell'ordine dato, righe degli esclusi con ``why``)."""
        kept, dropped = [], []
        for s in slugs:
            r = self.check(s, (entries or {}).get(s), (vocals_docs or {}).get(s))
            (kept.append(s) if r["admitted"] else dropped.append(r))
        return kept, dropped

    def describe(self) -> dict[str, Any]:
        return {"allowed_languages": sorted(self.allowed), "min_probability": self.min_probability,
                "excluded": dict(self.excluded), "n_language_index": len(self.index),
                "policy": str(POLICY_PATH.relative_to(ROOT)), "excluded_list": str(EXCLUDED_PATH.relative_to(ROOT))}


_DEFAULT: Optional[TrackFilter] = None


def default_filter() -> TrackFilter:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = TrackFilter()
    return _DEFAULT


def filter_sources(sources: Mapping[str, Any], log: Optional[list] = None,
                   track_filter: Optional[TrackFilter] = None) -> dict[str, Any]:
    """Filtra un dizionario ``slug -> sorgente`` (come quelli dei runner di set); registra gli esclusi in ``log``."""
    tf = track_filter or default_filter()
    entries = {}
    for s, src in sources.items():
        cat = (src or {}).get("catalog") if isinstance(src, Mapping) else None
        if isinstance(cat, Mapping):
            entries[s] = next((t for t in cat.get("tracks", []) if t.get("slug") == s), None) or {}
    vocals = {}
    for s, e in entries.items():  # brani analizzati con la selezione automatica della lingua (fuori dagli indici)
        if s in tf.index or not e or e.get("language") or not e.get("vocals_path"):
            continue
        try:
            from .paths import localize
            vp = localize(e["vocals_path"])
            if vp.is_file():
                vocals[s] = json.loads(vp.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - percorso illeggibile: lingua sconosciuta
            pass
    kept, dropped = tf.split(list(sources), entries, vocals)
    if log is not None:
        log.append({"track_filter": {**tf.describe(), "dropped": dropped}})
    return {s: sources[s] for s in kept}


def is_admitted(slug: str, track_filter: Optional[TrackFilter] = None) -> bool:
    return (track_filter or default_filter()).check(slug)["admitted"]
