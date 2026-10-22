#!/usr/bin/env python3
"""Schede brano persistenti (``track_cards/<slug>.json``, schema ``musicmix.track-card.1``): costruzione incrementale.

Raccoglie per ogni brano dei cataloghi (default: ponte ``runs/pool_bridge_v1/catalog_ready.json`` + pool V12
``configs/music_pool_v12.json``; solo lingue ammesse da ``musicmix.track_filter``) le analisi **già presenti**:

* analisi DJ (``dj_select``): ``runs/dj_set_v3_bridge/dj_plan/tracks`` (quella del set v3), poi
  ``runs/dj_plan_bridge_v1/tracks``, poi ``runs/dj_plan_v1/tracks`` (V12); deve avere lo stesso ``track_id`` del catalogo;
* modello di versi ``dj_phrase``: prima uno nella lingua del brano (``runs/asr_language_v1/phrase_<slug>_auto.json``),
  poi ``runs/dj_set_v3_bridge/phrase``, ``runs/dj_phrase_v1/tracks``, ``track_cards/_phrase``; attività dello stem da
  ``.cache/dj_set_v3_bridge/act`` o ``.cache/track_cards/act``. Con ``--compute-phrase`` un modello mancante si calcola
  (parole ASR del JSON vocals + stem vocale; nessun modello neurale) in ``track_cards/_phrase``;
* forma ``dj_form`` dagli spettrogrammi per stem di allin1 (``.cache/dj_allin1*/spec``), altrimenti dal mix;
* stem vocale verificato (``musicmix.stems.vocal_stem_record``), stem Demucs a 4 sorgenti e loro provenienza;
* correzioni manuali ``track_cards/corrections.json`` (sottocomando ``correct``).

Incrementale: una scheda si rifà solo se la chiave degli ingressi (hash di analisi, versi, attività, JSON vocals,
spettrogramma, voce di catalogo, lingua, audio, correzioni, versione) è cambiata (``--force`` rifà tutto).
Scrive anche ``track_cards/index.json`` (affidabilità e casi incerti per brano).

    python tools/build_track_cards.py build
    python tools/build_track_cards.py correct --slug antidote --kind verify_flag \\
        --flag tempo_octave_suspect --why "ascoltato: 65,6 BPM giusto, downbeat sull'uno" --author utente
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicmix import dj_form as F  # noqa: E402
from musicmix import dj_phrase as P  # noqa: E402
from musicmix import dj_select as djs  # noqa: E402
from musicmix import track_cards as TC  # noqa: E402
from musicmix import track_filter as TF  # noqa: E402
from musicmix.paths import localize  # noqa: E402

OUT = ROOT / "track_cards"
CATALOGS = (ROOT / "runs/pool_bridge_v1/catalog_ready.json", ROOT / "configs/music_pool_v12.json")
ANALYSIS_DIRS = (ROOT / "runs/dj_set_v3_bridge/dj_plan/tracks", ROOT / "runs/dj_plan_bridge_v1/tracks", ROOT / "runs/dj_plan_v1/tracks")
PHRASE_DIRS = (ROOT / "runs/dj_set_v3_bridge/phrase", ROOT / "runs/dj_phrase_v1/tracks")
ACT_DIRS = (ROOT / ".cache/dj_set_v3_bridge/act", ROOT / ".cache/track_cards/act")
SPEC_DIRS = (ROOT / ".cache/dj_allin1/spec", ROOT / ".cache/dj_allin1_cc/spec", ROOT / ".cache/dj_allin1_bridge/spec")
DEMIX_DIRS = (ROOT / ".cache/dj_allin1/demix/htdemucs", ROOT / ".cache/dj_allin1_bridge/demix/htdemucs", ROOT / ".cache/dj_allin1_cc/demix/htdemucs")
DEMIX_DIRS_V12 = DEMIX_DIRS[:1]  # stem V12 nominati per WAV canonico (a/b)
CACHE = ROOT / ".cache/track_cards"
SR = 48000
REVIEW_MIN = 0.3  # [H] gravità da cui un caso incerto entra nella coda di revisione umana


# ------------------------------------------------------------------ utils
def rel(p: Optional[Path]) -> Optional[str]:
    if p is None:
        return None
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p: Path) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def dump(p: Path, obj: Any) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_jsonable), encoding="utf-8")


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def ref(p: Optional[Path], **extra: Any) -> Optional[dict]:
    return None if p is None else {"path": rel(p), "sha256": sha256(p), **extra}


def stat_ref(p: Optional[Path]) -> Optional[dict]:
    """Riferimento leggero per file grandi (spettrogrammi): dimensione e mtime al posto dell'hash."""
    if p is None:
        return None
    st = Path(p).stat()
    return {"path": rel(p), "sha256": f"size={st.st_size};mtime={int(st.st_mtime)}"}


# --------------------------------------------------------------- catalogs
def catalog_entries(paths) -> tuple[dict[str, dict], dict[str, str]]:
    """slug -> voce di catalogo (il primo catalogo vince), slug -> catalogo."""
    out, where = {}, {}
    for p in paths:
        p = Path(p)
        if not p.is_file():
            continue
        for t in load(p).get("tracks") or []:
            if t["slug"] in out:
                continue
            if t.get("status") not in (None, "ready"):
                continue
            out[t["slug"]] = t
            where[t["slug"]] = rel(p)
    return out, where


def audio_ref(entry: dict) -> dict:
    """SHA-256 dell'audio canonico (memorizzato per percorso, dimensione e mtime) confrontato con ``track_id``."""
    p = localize(entry.get("canonical_path") or "")
    tid = entry.get("track_id")
    if not p.is_file():
        return {"sha256": tid, "canonical_path": rel(p), "present": False, "verified": None}
    st = p.stat()
    memo_p = CACHE / "audio_sha.json"
    memo = load(memo_p) if memo_p.is_file() else {}
    k = f"{p}|{st.st_size}|{int(st.st_mtime)}"
    if k not in memo:
        memo[k] = sha256(p)
        dump(memo_p, memo)
    got = memo[k]
    if tid is None:  # pool V12 configs: no track_id
        tid = got
    return {"sha256": tid, "canonical_path": rel(p), "present": True, "verified": got == tid, "measured_sha256": got,
            "sample_rate": entry.get("sample_rate"), "duration_s": entry.get("duration_seconds")}


def find_analysis(slug: str, entry: dict) -> Optional[Path]:
    cands = [d / f"{slug}.json" for d in ANALYSIS_DIRS]
    if entry.get("analysis_ref"):
        cands.append(localize(entry["analysis_ref"]))
    for p in cands:
        if p.is_file():
            a = load(p)
            if entry.get("track_id") is None or a.get("track_id") == entry.get("track_id"):
                return p
    return None


def _act_from_npz(path: Path) -> P.StemActivity:
    z = np.load(path)
    strict = z["active_strict"]
    return P.StemActivity(centers=z["centers"], active=z["active"], rms_db=z["rms_db"], threshold_db=float(z["threshold_db"]),
                          f0_hz=z["f0_hz"], onsets=z["onsets"], hop=int(z["hop"]), active_strict=strict if strict.size else None)


def _act_to_npz(act: P.StemActivity, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, centers=act.centers, active=act.active, rms_db=act.rms_db, threshold_db=np.array(act.threshold_db),
                        f0_hz=act.f0_hz, onsets=act.onsets, hop=np.array(act.hop),
                        active_strict=act.active_strict if act.active_strict is not None else np.zeros(0, bool))


def grid_matches(model: dict, analysis: dict) -> bool:
    return abs(float(model.get("period_s") or 0) - float(analysis["tempo"]["period_s"])) < 1e-6 and \
        len(model.get("downbeats_s") or []) == len(analysis["grid"]["downbeats_s"])


def find_phrase(slug: str, analysis: dict, track_lang: Optional[str], out: Path) -> tuple[Optional[Path], list[str]]:
    """Modello di versi sulla stessa griglia, prima nella lingua del brano; restituisce anche i candidati scartati."""
    cands = [ROOT / f"runs/asr_language_v1/phrase_{slug}_auto.json"] + [d / f"{slug}.json" for d in PHRASE_DIRS] + [out / "_phrase" / f"{slug}.json"]
    ok, notes = [], []
    for p in cands:
        if not p.is_file():
            continue
        m = load(p).get("model") or {}
        if not grid_matches(m, analysis):
            notes.append(f"{rel(p)}: altra griglia")
            continue
        ok.append((p, m.get("lang")))
    if not ok:
        return None, notes
    if track_lang:
        for p, lang in ok:
            if lang and str(lang).lower() == str(track_lang).lower():
                return p, notes + [f"{rel(q)}: scartato (lingua {lg})" for q, lg in ok if q != p]
    return ok[0][0], notes + [f"{rel(q)}: alternativa" for q, _ in ok[1:]]


def find_act(slug: str) -> Optional[Path]:
    return next((d / f"{slug}.npz" for d in ACT_DIRS if (d / f"{slug}.npz").is_file()), None)


def compute_phrase(slug: str, entry: dict, analysis: dict, out: Path) -> tuple[Optional[Path], Optional[str]]:
    """Modello di versi da ASR già fatto + stem vocale (DSP, nessun modello neurale)."""
    from musicmix.stems import StemError, vocal_stem_path, vocals_json_for
    import soundfile as sf

    try:
        vj = vocals_json_for(entry) if entry.get("vocals_path") else localize("")
    except StemError as exc:
        return None, f"stem: {exc}"
    if not vj.is_file():
        return None, "JSON vocals assente"
    try:
        stem_path = vocal_stem_path(entry)
    except StemError as exc:
        return None, f"stem: {exc}"
    voc = load(vj)
    be = voc.get("backend") or {}
    lang = be.get("detected_language") or be.get("language") or "en"
    stem, sr = sf.read(str(stem_path), dtype="float32", always_2d=False)
    if sr != SR:
        return None, f"stem a {sr} Hz"
    act = P.StemActivity.from_audio(stem, sr)
    del stem
    model = P.analyze_track(analysis, voc["words"], act=act, lang=lang, asr_language_probability=be.get("language_probability"))
    npz = CACHE / "act" / f"{slug}.npz"
    _act_to_npz(act, npz)
    pj = out / "_phrase" / f"{slug}.json"
    dump(pj, {"schema_version": "musicmix.dj_phrase_run.1", "slug": slug,
              "inputs": {"vocals_json": rel(vj), "vocals_json_sha256": sha256(vj), "vocal_stem": rel(stem_path),
                         "asr_model": be.get("model_path"), "asr_language": lang,
                         "asr_language_probability": be.get("language_probability"), "built_by": "tools/build_track_cards.py --compute-phrase"},
              "model": model})
    return pj, None


def find_spec(slug: str, entry: dict, duration_s: float) -> tuple[Optional[Path], Optional[str]]:
    names = [f"{slug}.npy"]
    can = entry.get("canonical_path")
    if can:
        names.append(Path(str(can).replace("\\", "/")).stem + ".npy")  # V12: .cache/dj_allin1/spec/a.npy = reminder
    for d in SPEC_DIRS:
        for n in names:
            p = d / n
            if p.is_file():
                spec = np.load(p, mmap_mode="r")
                dur = spec.shape[1] / F.ALLIN1_FPS
                if abs(dur - duration_s) <= 1.0:
                    return p, None
    return None, "nessuno spettrogramma allin1 della durata giusta"


def form_for(slug: str, entry: dict, analysis: dict, analysis_sha: str) -> tuple[Optional[dict], Optional[dict], Optional[str]]:
    """Forma dj_form sulla griglia regolare di dj_select (cache per hash dell'analisi e dello spettrogramma)."""
    t = analysis["tempo"]
    beats, downs = djs.grid_times({"period_s": t["period_s"], "phase_s": t["phase_s"], "bar_offset_beats": t["bar_offset_beats"]},
                                  analysis["duration_s"])
    stored = np.asarray(analysis["grid"]["downbeats_s"], dtype=float)
    downs = downs[(downs >= stored[0] - 1e-3) & (downs <= stored[-1] + 1e-3)]
    if len(downs) != len(stored):
        return None, None, "griglia rigenerata diversa da quella dell'analisi"
    bar = djs.BEATS_PER_BAR * t["period_s"]
    beats = beats[(beats >= downs[0] - 1e-3) & (beats < downs[-1] + bar - 1e-3)]
    spec_p, why = find_spec(slug, entry, float(analysis["duration_s"]))
    key = {"analysis_sha256": analysis_sha, "form": F.FORM_VERSION, "spec": stat_ref(spec_p),
           "params": {k: v[0] for k, v in F.PARAMS.items()}}
    cp = CACHE / "form" / f"{slug}.json"
    if cp.is_file():
        doc = load(cp)
        if doc.get("key") == json.loads(json.dumps(key)):
            return doc["form"], stat_ref(spec_p), None
    if spec_p is not None:
        form = F.analyse_from_allin1_spec(np.asarray(np.load(spec_p, mmap_mode="r")), downs, beats)
    else:
        wav = localize(entry.get("canonical_path") or "")
        if not wav.is_file():
            return None, None, (why or "") + "; audio canonico assente"
        import soundfile as sf

        audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        form = F.analyse(audio, sr, downs, beats)
        del audio
    form["provenance"] = {"spec": rel(spec_p), "fallback_reason": why if spec_p is None else None,
                          "grid": "dj_select regularised grid (dj_select.grid_times)"}
    dump(cp, {"key": key, "form": form})
    return form, stat_ref(spec_p), None


def stems_info(slug: str, entry: dict, vocals_json: Optional[Path]) -> dict:
    from musicmix.stems import StemError, vocal_stem_record

    out: dict[str, Any] = {}
    errors = []
    # prima lo stem della voce di catalogo (fase vocals corrente), poi quello del JSON vocals del modello di versi
    for vj in ([None] + ([vocals_json] if vocals_json else [])):
        try:
            r = vocal_stem_record(entry, vocals_json=vj) if vj else vocal_stem_record(entry)
            out["vocals"] = {"ok": True, "path": rel(r["path"]), "recorded_as": r["recorded_as"], "model": r["model"],
                             "checkpoint_sha256": r["checkpoint_sha256"], "input_sha256": r["input_sha256"], "checks": r["checks"],
                             "vocals_json": rel(Path(r["vocals_json"])) if r.get("vocals_json") else None}
            break
        except (StemError, OSError, KeyError) as exc:
            errors.append(str(exc))
    else:
        out["vocals"] = {"ok": False, "error": "; ".join(errors)}
    demix = next((d / slug for d in DEMIX_DIRS if (d / slug / "vocals.wav").is_file()), None)
    can = Path(str(entry.get("canonical_path") or "").replace("\\", "/")).stem
    if demix is None and can:
        demix = next((d / can for d in DEMIX_DIRS_V12 if (d / can / "vocals.wav").is_file()), None)
    if demix is not None:
        prov_p = demix.parent.parent / "htdemucs_provenance" / f"{demix.name}.json"
        prov = load(prov_p) if prov_p.is_file() else None
        ok = prov is None or entry.get("track_id") is None or prov.get("input_sha256") in (None, entry.get("track_id"),
                                                                                             (out.get("vocals") or {}).get("input_sha256"))
        out["demucs_4stem"] = {"dir": rel(demix), "sources": sorted(p.stem for p in demix.glob("*.wav")),
                               "provenance": rel(prov_p) if prov else None, "model": (prov or {}).get("model"),
                               "input_sha256": (prov or {}).get("input_sha256"), "identity_ok": bool(ok)}
    else:
        out["demucs_4stem"] = None
    return out


# ------------------------------------------------------------------ build
def build(args) -> dict:
    global ANALYSIS_DIRS, SPEC_DIRS, DEMIX_DIRS
    ANALYSIS_DIRS = tuple(getattr(args, "analysis_dir", None) or ()) + ANALYSIS_DIRS
    SPEC_DIRS = tuple(getattr(args, "spec_dir", None) or ()) + SPEC_DIRS
    DEMIX_DIRS = tuple(getattr(args, "demix_dir", None) or ()) + DEMIX_DIRS
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    corr_p = out / "corrections.json"
    if not corr_p.is_file():
        dump(corr_p, TC.new_corrections_doc())
    corr_doc = load(corr_p)
    entries, where = catalog_entries(args.catalog or CATALOGS)
    tf = TF.default_filter()
    rows, skipped = [], []
    only = set(args.only or [])
    t_all = time.time()
    for slug, entry in sorted(entries.items()):
        if only and slug not in only:
            continue
        t0 = time.time()
        try:
            from musicmix.stems import StemError, vocals_json_for
            vj = vocals_json_for(entry) if entry.get("vocals_path") else localize("")  # override additivo se c'e'
        except StemError:
            vj = localize(entry.get("vocals_path") or "")
        vdoc = load(vj) if vj.is_file() else None
        lang = tf.check(slug, entry, vdoc)
        if not lang["admitted"]:
            skipped.append({"slug": slug, "why": lang["why"]})
            continue
        ap = find_analysis(slug, entry)
        analysis = load(ap) if ap else None
        phrase_p, phrase_notes = (None, [])
        if analysis is not None:
            phrase_p, phrase_notes = find_phrase(slug, analysis, lang.get("language"), out)
            if phrase_p is None and args.compute_phrase:
                print(f"{slug}: calcolo del modello di versi (--compute-phrase)", flush=True)
                phrase_p, why = compute_phrase(slug, entry, analysis, out)
                if why:
                    phrase_notes.append(f"--compute-phrase: {why}")
        phrase_doc = load(phrase_p) if phrase_p else None
        act_p = find_act(slug) if phrase_p else None
        pv = None
        if phrase_doc:
            pin = phrase_doc.get("inputs") or {}
            pvs = pin.get("vocals_json") or (pin.get("asr") or {}).get("file")
            pv = (ROOT / pvs) if pvs and not Path(pvs).is_absolute() else (Path(pvs) if pvs else None)
            if pv is not None and not pv.is_file():
                pv = None
        vocals_used = pv or (vj if vj.is_file() else None)
        spec_p = find_spec(slug, entry, float(analysis["duration_s"]))[0] if analysis else None
        corrections = TC.corrections_for(corr_doc, slug)
        audio = audio_ref(entry)
        sep = (vdoc or {}).get("separator") or {}
        stem_files = [sep[k] for k in ("reused_stem_path", "output_path") if isinstance(sep.get(k), str)]
        inputs = {"catalog_entry": {"path": where.get(slug), "sha256": TC.sha256_json(entry)},
                  "audio": {"path": audio.get("canonical_path"), "sha256": audio.get("sha256")},
                  "analysis": ref(ap), "phrase_model": ref(phrase_p), "stem_activity": ref(act_p),
                  "vocals_json": ref(vocals_used), "allin1_spec": stat_ref(spec_p),
                  "vocal_stem": stat_ref(next((localize(p) for p in stem_files if localize(p).is_file()), None)),
                  "language": {"sha256": TC.sha256_json({k: lang.get(k) for k in ("language", "probability", "rule")})}}
        key = TC.input_key(inputs, corrections)
        cp = out / f"{slug}.json"
        if cp.is_file() and not args.force:
            old = load(cp)
            if old.get("input_key") == key and old.get("builder_version") == TC.BUILDER_VERSION:
                rows.append(index_row(old, "unchanged"))
                continue
        form = fs = None
        form_why = None
        if analysis is not None:
            form, _sref, form_why = form_for(slug, entry, analysis, inputs["analysis"]["sha256"])
            fs = F.form_summary(form) if form else None
        act = _act_from_npz(act_p) if act_p else None
        notes = [f"analisi: {rel(ap)}" if ap else "analisi DJ assente"] + phrase_notes + ([f"forma: {form_why}"] if form_why else [])
        card = TC.build_card(slug=slug, entry=entry, analysis=analysis, inputs=inputs, phrase=(phrase_doc or {}).get("model"), act=act,
                             vocals_doc=load(vocals_used) if vocals_used else None, language=lang, form=form, form_summary=fs,
                             stems=stems_info(slug, entry, pv), audio=audio, corrections=corrections,
                             built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), notes=notes)
        card["input_key"] = key
        dump(cp, card)
        rows.append(index_row(card, "built"))
        print(f"{slug}: affidabilità {card['reliability']['value']:.2f} (pianificazione {card['reliability']['planning']:.2f}); "
              f"incerti: {', '.join(TC.uncertain_summary(card)) or '-'} [{time.time() - t0:.1f} s]", flush=True)
    index = {"schema_version": "musicmix.track-card-index.1", "card_schema": TC.CARD_SCHEMA, "builder_version": TC.BUILDER_VERSION,
             "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "catalogs": [rel(Path(p)) for p in (args.catalog or CATALOGS)], "language_policy": tf.describe(),
             "thresholds": {k: {"value": v[0], "why": v[1]} for k, v in TC.THRESHOLDS.items()}, "flags": TC.FLAGS,
             "tracks": sorted(rows, key=lambda r: r["slug"]), "skipped": skipped, "elapsed_s": round(time.time() - t_all, 1)}
    if only and (out / "index.json").is_file():  # aggiornamento parziale: tiene le altre righe
        prev = {r["slug"]: r for r in load(out / "index.json").get("tracks", [])}
        prev.update({r["slug"]: r for r in rows})
        index["tracks"] = sorted(prev.values(), key=lambda r: r["slug"])
    # code d'intervento (misura dell'intervento umano richiesto dai brani nuovi), [H]: ascolto solo per i casi aperti "listen"
    # con gravità >= REVIEW_MIN; "data" = analisi da fare/rifare senza ascolto; "avoid" = il pianificatore li aggira
    index["review_rule"] = (f"[H] ascolto se un caso aperto con fix='listen' ha gravità >= {REVIEW_MIN}; dati se un caso ha fix='data'; "
                            "i casi 'avoid' (tonalità incerta, forma senza core) li aggira il pianificatore")
    index["review_queue"] = [{"slug": r["slug"], "flags": [f["id"] for f in r["uncertain"] if f.get("fix") == "listen" and f["severity"] >= REVIEW_MIN]}
                             for r in index["tracks"] if any(f.get("fix") == "listen" and f["severity"] >= REVIEW_MIN for f in r["uncertain"])]
    index["data_queue"] = [{"slug": r["slug"], "flags": [f["id"] for f in r["uncertain"] if f.get("fix") == "data"]}
                           for r in index["tracks"] if any(f.get("fix") == "data" for f in r["uncertain"])]
    index["n_review"] = len(index["review_queue"])
    index["n_data"] = len(index["data_queue"])
    dump(out / "index.json", index)
    print(f"{len(rows)} schede ({sum(r['status'] == 'built' for r in rows)} rifatte), {len(skipped)} brani esclusi dalla lingua; "
          f"{time.time() - t_all:.0f} s", flush=True)
    return index


def index_row(card: dict, status: str) -> dict:
    r = card.get("reliability") or {}
    return {"slug": card["slug"], "status": status, "title": card.get("title"), "artist": card.get("artist"),
            "audio_sha256": (card.get("audio") or {}).get("sha256"), "input_key": card.get("input_key"), "built_at": card.get("built_at"),
            "bpm": (card.get("grid") or {}).get("bpm"), "camelot": (card.get("key") or {}).get("camelot"),
            "language": ((card.get("voice") or {}).get("language") or {}).get("language"),
            "reliability": r.get("value"), "planning": r.get("planning"), "components": r.get("components"),
            "excluded": bool(card.get("excluded")),
            "n_entries_clean": len((card.get("usable") or {}).get("entries") or []), "n_entries": (card.get("usable") or {}).get("n_entries"),
            "n_exits_line_end": len((card.get("usable") or {}).get("exits_line_end") or []), "n_exits": (card.get("usable") or {}).get("n_exits"),
            "uncertain": [{"id": f["id"], "severity": f["severity"], "fix": f.get("fix"), "what": f["what"]}
                          for f in card.get("uncertain") or [] if not f.get("resolved_by")],
            "resolved": [f["id"] for f in card.get("uncertain") or [] if f.get("resolved_by")]}


# ---------------------------------------------------------------- correct
def correct(args) -> dict:
    """Aggiunge una correzione manuale (legata allo SHA-256 dell'audio della scheda) e ricostruisce la scheda."""
    out = Path(args.out)
    corr_p = out / "corrections.json"
    doc = load(corr_p) if corr_p.is_file() else TC.new_corrections_doc()
    if args.kind not in TC.CORRECTION_KINDS:
        raise SystemExit(f"--kind tra {TC.CORRECTION_KINDS}")
    card_p = out / f"{args.slug}.json"
    sha = load(card_p)["audio"]["sha256"] if card_p.is_file() else None
    lst = doc["tracks"].setdefault(args.slug, [])
    c = {"id": f"{args.slug}-{len(lst) + 1}", "date": datetime.now(timezone.utc).date().isoformat(), "author": args.author,
         "kind": args.kind, "audio_sha256": sha, "flag": args.flag, "bar": args.bar, "value": args.value, "why": args.why}
    lst.append({k: v for k, v in c.items() if v is not None})
    dump(corr_p, doc)
    print(f"correzione registrata: {c['id']}", flush=True)
    args.only = [args.slug]
    args.force = False
    return build(args)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build", help="costruisce/aggiorna le schede (incrementale)")
    for p in (b,):
        p.add_argument("--catalog", nargs="*", default=None, help="cataloghi (default: ponte + V12)")
        p.add_argument("--only", nargs="*", default=None, help="solo questi slug")
        p.add_argument("--force", action="store_true")
        p.add_argument("--compute-phrase", action="store_true", help="calcola i modelli di versi mancanti (ASR + stem, niente reti)")
        p.add_argument("--analysis-dir", type=Path, action="append", default=[],
                       help="cartella extra di analisi dj_select (<slug>.json), cercata per prima (es. pool importato)")
        p.add_argument("--spec-dir", type=Path, action="append", default=[], help="cartella extra di spettrogrammi allin1")
        p.add_argument("--demix-dir", type=Path, action="append", default=[], help="cartella extra di stem htdemucs")
    c = sub.add_parser("correct", help="registra una correzione manuale nella scheda")
    c.add_argument("--slug", required=True)
    c.add_argument("--kind", required=True, choices=TC.CORRECTION_KINDS)
    c.add_argument("--flag")
    c.add_argument("--bar", type=int)
    c.add_argument("--value")
    c.add_argument("--why", required=True)
    c.add_argument("--author", default="utente")
    c.add_argument("--catalog", nargs="*", default=None)
    c.add_argument("--compute-phrase", action="store_true")
    for p in (b, c):
        p.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    if args.cmd in (None, "build"):
        if args.cmd is None:
            args = ap.parse_args(["build"] + list(argv or sys.argv[1:]))
        build(args)
    else:
        correct(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
