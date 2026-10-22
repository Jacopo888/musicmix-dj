"""Schede brano persistenti (``musicmix.track-card.1``): tutto ciò che il set deve sapere di un brano, preparato una volta.

Una scheda per brano (``track_cards/<slug>.json``), legata allo SHA-256 dell'audio canonico (= ``track_id``) e alle
analisi da cui è stata costruita (SHA-256 di ogni ingresso, ``input_key``): griglia e confidenza (accordo tra
Beat This!, allin1 e madmom; mezzo/doppio tempo sospetto), tonalità e confidenza, sezioni allin1 e forma
(``dj_form``), voce (lingua, qualità dell'ASR, versi di ``dj_phrase``), **ingressi utilizzabili** (cue IN di
``dj_select`` in cui B entra con la prima frase intera, verificati con ``dj_phrase.entry_clear_test``, la stessa guardia
di ``dj_set.entry_guard``), **uscite utilizzabili** (cue OUT con lo stato del verso di ``dj_phrase.exit_point``; per il cut
``exit_clear_test``), regioni strumentali, stem con provenienza, limiti di stretching consigliati e l'elenco dei **casi
incerti** da rivedere. Le correzioni manuali (``track_cards/corrections.json``) entrano nella scheda e valgono in ogni
set.

Questo modulo è puro (nessun I/O): riceve analisi, modello di versi, attività dello stem, forma, record dello stem,
lingua e correzioni già letti (``tools/build_track_cards.py`` li raccoglie). Le soglie dei casi incerti sono [H]
(ipotesi nostre), con il motivo nel campo ``why`` di ogni caso; nessuna è tarata su un ascolto.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, Optional, Sequence

CARD_SCHEMA = "musicmix.track-card.1"
CORRECTIONS_SCHEMA = "musicmix.track-card-corrections.1"
BUILDER_VERSION = "track_cards.1"

#: Soglie dei casi incerti [H] (nessuna tarata su ascolti; motivo accanto).
THRESHOLDS: dict[str, tuple[Any, str]] = {
    "grid_fit_min": (0.70, "[H] beat_f/downbeat_f di dj_select (accordo della griglia regolare con il provider) sotto 0,70: "
                           "più di un battito su quattro fuori dalla tolleranza"),
    "grid_residual_max_ms": (20.0, "[H] residuo mediano |battito - griglia| oltre 20 ms: metà della tolleranza di fase del "
                                   "veto del set (40 ms, run_dj_set_v3_bridge)"),
    "tempo_plausible_bpm": ((70.0, 160.0), "[H] fuori da 70-160 BPM con un provider all'ottava: probabile mezzo/doppio tempo "
                                           "(antidote 65,6 vs 131,1 BPM, orchestratore 21/10)"),
    "phase_coincidence_min": (0.5, "[H] coincidenza dei downbeat (o dei battiti) con un altro provider sotto 0,5"),
    "asr_low_conf": (0.5, "[H] confidenza della parola di faster-whisper sotto 0,5 (come asr_language_v1)"),
    "asr_low_conf_share_max": (0.20, "[H] più del 20 % delle parole sotto 0,5 di confidenza"),
    "language_prob_min": (0.5, "[H] come configs/track_languages.json"),
    "min_clean_entries": (2, "[H] meno di due ingressi puliti: il brano si può aprire quasi solo in un punto"),
    "instrumental_min_bars": (2, "[H] regione strumentale = almeno 2 battute senza voce (bars.vocal di dj_select)"),
    "vocal_bar_share_heavy": (0.75, "[H] voce in più del 75 % delle battute: lo stretching sulla voce si sente di più"),
}

#: Gravità di default dei casi incerti [H]: 0 = informativo, 1 = inutilizzabile. ``category``: a che cosa si applica nel
#: calcolo dell'affidabilità (reliability.py): grid/phrase/track sempre, key solo con sovrapposizione. ``fix``: che cosa lo
#: risolve: ``listen`` (verifica umana a campione), ``data`` (un'analisi da fare o rifare, nessun ascolto), ``avoid`` (il
#: pianificatore lo aggira: es. niente blend con tonalità incerta), ``none``.
FLAGS: dict[str, dict[str, Any]] = {
    "analysis_missing": {"severity": 1.0, "category": "track", "fix": "data"},
    "audio_sha_mismatch": {"severity": 1.0, "category": "track", "fix": "data"},
    "language_not_allowed": {"severity": 1.0, "category": "track", "fix": "none"},
    "language_uncertain": {"severity": 0.1, "category": "track", "fix": "listen"},
    "vocal_stem_missing": {"severity": 0.5, "category": "track", "fix": "data"},
    "tempo_octave_suspect": {"severity": 0.7, "category": "grid", "fix": "listen"},  # ottava sbagliata = ogni confine di battuta sbagliato
    "grid_confidence_low": {"severity": 0.35, "category": "grid", "fix": "listen"},
    "grid_fit_low": {"severity": 0.15, "category": "grid", "fix": "listen"},
    "grid_residual_high": {"severity": 0.2, "category": "grid", "fix": "listen"},
    "downbeat_phase_disputed": {"severity": 0.1, "category": "grid", "fix": "listen"},
    "beat_phase_disputed": {"severity": 0.15, "category": "grid", "fix": "listen"},
    "key_confidence_low": {"severity": 0.3, "category": "key", "fix": "avoid"},
    "key_estimators_disagree": {"severity": 0.1, "category": "key", "fix": "avoid"},
    "phrase_model_missing": {"severity": 0.3, "category": "phrase", "fix": "data"},
    "asr_unreliable": {"severity": 0.3, "category": "phrase", "fix": "data"},
    "asr_language_mismatch": {"severity": 0.3, "category": "phrase", "fix": "data"},
    "asr_low_confidence": {"severity": 0.1, "category": "phrase", "fix": "listen"},
    "stem_activity_missing": {"severity": 0.1, "category": "phrase", "fix": "data"},
    "no_clean_entries": {"severity": 0.5, "category": "track", "fix": "listen"},
    "few_clean_entries": {"severity": 0.2, "category": "track", "fix": "listen"},
    "no_line_end_exits": {"severity": 0.2, "category": "track", "fix": "listen"},
    "form_no_core": {"severity": 0.05, "category": "track", "fix": "avoid"},
    "form_missing": {"severity": 0.05, "category": "track", "fix": "data"},
    "correction_pending_reanalysis": {"severity": 0.3, "category": "grid", "fix": "data"},
    "excluded_by_correction": {"severity": 1.0, "category": "track", "fix": "none"},
}

ENTRY_OK = ("instrumental", "on_downbeat", "lead_in")
EXIT_LINE_END = ("clean", "tail", "a_pickup", "instrumental")
CORRECTION_KINDS = ("exclude", "include", "verify_flag", "block_entry", "block_exit", "approve_entry", "approve_exit",
                    "bpm_octave", "downbeat_offset_beats", "key", "note")


def thr(name: str) -> Any:
    return THRESHOLDS[name][0]


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def input_key(inputs: Mapping[str, Any], corrections: Sequence[Mapping[str, Any]] = ()) -> str:
    """Chiave d'incrementalità: SHA-256 degli ingressi (percorsi + hash), delle correzioni e della versione."""
    return sha256_json({"builder": BUILDER_VERSION, "schema": CARD_SCHEMA, "thresholds": {k: v[0] for k, v in THRESHOLDS.items()},
                        "flags": FLAGS, "inputs": {k: (v or {}).get("sha256") if isinstance(v, Mapping) else v for k, v in inputs.items()},
                        "corrections": list(corrections)})


def _flag(fid: str, what: str, why: str, evidence: Any = None, action: Optional[str] = None) -> dict[str, Any]:
    f = FLAGS[fid]
    return {"id": fid, "severity": f["severity"], "category": f["category"], "fix": f["fix"], "what": what, "why": why,
            "evidence": evidence, "action": action, "resolved_by": None}


# ------------------------------------------------------------------- grid
def grid_section(analysis: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    t = analysis["tempo"]
    flags: list[dict[str, Any]] = []
    agreement = t.get("agreement") or {}
    providers = t.get("providers") or {}
    other = {}
    db_disputed, b_disputed = [], []
    pmin = thr("phase_coincidence_min")
    for name, ag in agreement.items():
        ratio = float(ag.get("bpm_ratio") or 1.0)
        same = bool(ag.get("same_tempo_octave", abs(ratio - 1.0) < 0.04))
        other[name] = {"bpm": (providers.get(name) or {}).get("bpm"), "bpm_ratio": ratio, "same_octave": same,
                       "beat_coincidence": ag.get("beat_coincidence"), "downbeat_coincidence": ag.get("downbeat_coincidence")}
        if ag.get("downbeat_coincidence") is not None and float(ag["downbeat_coincidence"]) < pmin:
            db_disputed.append(name)
        if same and ag.get("beat_coincidence") is not None and float(ag["beat_coincidence"]) < pmin:
            b_disputed.append(name)
    bpm = float(t["bpm"])
    lo, hi = thr("tempo_plausible_bpm")
    octave = bool(t.get("octave_ambiguous"))
    octave_suspect = octave and not (lo <= bpm <= hi)
    fit = min(float(t.get("beat_f") or 0.0), float(t.get("downbeat_f") or 0.0))
    res = t.get("residual_ms_median_abs")
    n_agree = sum(1 for v in other.values() if v["same_octave"] and (v["downbeat_coincidence"] or 0) >= pmin)
    g = {"bpm": bpm, "period_s": t.get("period_s"), "provider": t.get("provider"), "confidence": t.get("confidence"),
         "beat_f": t.get("beat_f"), "downbeat_f": t.get("downbeat_f"), "residual_ms_median_abs": res,
         "drift_pct_max": t.get("drift_pct_max"), "phase_s": t.get("phase_s"), "bar_offset_beats": t.get("bar_offset_beats"),
         "n_bars": (analysis.get("grid") or {}).get("n_bars"), "providers": providers, "agreement": other,
         "providers_agreeing_fully": n_agree, "octave_ambiguous": octave, "octave_votes": t.get("octave_votes"),
         "tempo_octave_suspect": octave_suspect, "downbeat_phase_disputed_by": db_disputed, "beat_phase_disputed_by": b_disputed}
    if octave_suspect:
        alt = [k for k in (t.get("octave_votes") or {}) if abs(float(k) - bpm) > 1.0]
        flags.append(_flag("tempo_octave_suspect", f"tempo {bpm:.2f} BPM fuori da {lo:g}-{hi:g} con voti d'ottava {t.get('octave_votes')}",
                           THRESHOLDS["tempo_plausible_bpm"][1], {"bpm": bpm, "alternatives_bpm": alt},
                           "verificare all'ascolto mezzo/doppio tempo e il downbeat (correzione bpm_octave o verify_flag)"))
    if t.get("confidence") == "low":
        flags.append(_flag("grid_confidence_low", "confidenza della griglia di dj_select 'low'",
                           "dj_select: accordo debole tra griglia regolare e provider", {"beat_f": t.get("beat_f"), "downbeat_f": t.get("downbeat_f")},
                           "controllare il downbeat nelle clip; preferire ingressi senza sovrapposizione"))
    if fit < thr("grid_fit_min"):
        flags.append(_flag("grid_fit_low", f"accordo della griglia {fit:.2f} < {thr('grid_fit_min'):.2f}",
                           THRESHOLDS["grid_fit_min"][1], {"beat_f": t.get("beat_f"), "downbeat_f": t.get("downbeat_f")}))
    if res is not None and float(res) > thr("grid_residual_max_ms"):
        flags.append(_flag("grid_residual_high", f"residuo mediano {float(res):.0f} ms > {thr('grid_residual_max_ms'):.0f} ms",
                           THRESHOLDS["grid_residual_max_ms"][1], {"residual_ms_median_abs": res}))
    if db_disputed:
        flags.append(_flag("downbeat_phase_disputed", f"downbeat non condivisi da {', '.join(db_disputed)}",
                           THRESHOLDS["phase_coincidence_min"][1] + "; con un provider all'ottava il downbeat può cadere sul terzo battito",
                           {n: other[n] for n in db_disputed}))
    if b_disputed:
        flags.append(_flag("beat_phase_disputed", f"battiti fuori fase rispetto a {', '.join(b_disputed)} (stesso tempo)",
                           THRESHOLDS["phase_coincidence_min"][1], {n: other[n] for n in b_disputed}))
    return g, flags


# -------------------------------------------------------------------- key
def key_section(analysis: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    k = analysis.get("key") or {}
    est = {n: (e or {}).get("camelot") for n, e in (k.get("estimates") or {}).items()}
    cam = k.get("camelot")
    n_agree = sum(1 for v in est.values() if v == cam)
    out = {"key": k.get("key"), "camelot": cam, "source": k.get("source"), "confidence": k.get("confidence"), "level": k.get("level"),
           "agree_exact": k.get("agree_exact"), "agree_same_or_relative": k.get("agree_same_or_relative"), "estimates_camelot": est,
           "n_estimators_agree": n_agree, "n_estimators": len(est)}
    flags = []
    if k.get("level") != "high":
        flags.append(_flag("key_confidence_low", f"tonalità {k.get('key')} ({cam}) con livello '{k.get('level')}' (confidenza "
                           f"{float(k.get('confidence') or 0):.2f})", "dj_select: stimatori discordi o probabilità bassa",
                           {"estimates": est}, "nei blend la compatibilità tonale non è garantita: preferire ingressi senza sovrapposizione"))
    if est and n_agree < 2:
        flags.append(_flag("key_estimators_disagree", f"solo {n_agree}/{len(est)} stimatori danno {cam}", "[H] meno di due stimatori d'accordo",
                           {"estimates": est}))
    return out, flags


# -------------------------------------------------------------- structure
def structure_section(analysis: Mapping[str, Any], form: Optional[Mapping[str, Any]], form_summary: Optional[Mapping[str, Any]],
                      form_ref: Optional[Mapping[str, Any]] = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    secs = analysis.get("sections") or []
    n = len(analysis.get("phrases") or [])
    rows = []
    for i, s in enumerate(secs):
        start = int(s.get("bar_index", 0))
        end = int(secs[i + 1].get("bar_index", n)) if i + 1 < len(secs) else n
        rows.append({"label": s.get("label"), "start_bar": start, "end_bar": end, "bars": max(0, end - start),
                     "start_s": s.get("downbeat_s", s.get("start_s")), "loose": bool(s.get("loose"))})
    flags = []
    fs = None
    if form is not None:
        fs = dict(form_summary or {})
        fs["version"] = form.get("version") or (form_ref or {}).get("version")
        fs["provenance"] = form.get("provenance")
        if form.get("core_status") == "no_core":
            flags.append(_flag("form_no_core", f"dj_form senza core ({form.get('no_core_reason')})",
                               "dj_form.2: senza core le battute sono 'unknown' (intro/build/outro non definiti)"))
    else:
        flags.append(_flag("form_missing", "forma dj_form non calcolata (spettrogramma allin1 e audio assenti)",
                           "nessuna categoria di Butler per questo brano"))
    return {"sections_source": analysis.get("sections_source"), "sections": rows, "form": fs}, flags


# ------------------------------------------------------------------ voice
def asr_stats(vocals_doc: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    if not vocals_doc:
        return None
    words = vocals_doc.get("words") or []
    conf = [float(w["confidence"]) for w in words if w.get("confidence") is not None]
    be = vocals_doc.get("backend") or {}
    lo = thr("asr_low_conf")
    import statistics

    return {"model": be.get("model_path"), "language": be.get("language"), "detected_language": be.get("detected_language"),
            "language_probability": be.get("language_probability"), "n_words": len(words),
            "median_confidence": statistics.median(conf) if conf else None,
            "share_low_confidence": (sum(c < lo for c in conf) / len(conf)) if conf else None,
            "source_kind": vocals_doc.get("source_kind")}


def voice_section(analysis: Mapping[str, Any], phrase: Optional[Mapping[str, Any]], phrase_ref: Optional[Mapping[str, Any]],
                  act_available: bool, asr: Optional[Mapping[str, Any]], language: Optional[Mapping[str, Any]]
                  ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    flags = []
    lang = dict(language or {})
    if lang and not lang.get("admitted", True):
        flags.append(_flag("language_not_allowed", f"lingua cantata {lang.get('language')}", lang.get("why") or "track_filter"))
    elif lang.get("rule") in ("language_uncertain", "language_unknown"):
        flags.append(_flag("language_uncertain", lang.get("why") or "lingua incerta", "track_filter"))
    track_lang = lang.get("language")
    pm = None
    if phrase is None:
        flags.append(_flag("phrase_model_missing", "nessun modello di versi dj_phrase per questo brano",
                           "senza versi: fini verso, pickup e ingressi puliti non verificabili", None,
                           "tools/build_track_cards.py --compute-phrase (ASR + stem vocale: nessun modello pesante)"))
    else:
        q = phrase.get("asr_quality") or {}
        lines = phrase.get("lines") or []
        sr = float(phrase.get("sample_rate") or 48000)
        pm = {"source": (phrase_ref or {}).get("path"), "sha256": (phrase_ref or {}).get("sha256"), "lang": phrase.get("lang"),
              "asr_reliable": bool(q.get("reliable", False)), "asr_reasons": q.get("reasons"),
              "n_words": (phrase.get("summary") or {}).get("n_words"), "n_lines": len(lines),
              "stem_activity": bool(act_available),
              "lines": [{"start_s": round(ln["start_sample"] / sr, 3), "end_s": round(ln["end_sample"] / sr, 3),
                         "p_end": round(float(ln.get("p_end") or 0.0), 3), "text": (ln.get("text") or "")[:80]} for ln in lines]}
        if not q.get("reliable", False):
            flags.append(_flag("asr_unreliable", "trascrizione giudicata non affidabile da dj_phrase.asr_quality",
                               "il modello si astiene: niente fini verso", q.get("reasons")))
        if track_lang and phrase.get("lang") and str(phrase["lang"]).lower() != str(track_lang).lower():
            flags.append(_flag("asr_language_mismatch", f"versi da un ASR in '{phrase.get('lang')}' su un brano in '{track_lang}'",
                               "le parole e quindi i fini verso sono poco credibili", {"phrase_model": (phrase_ref or {}).get("path")},
                               "rifare l'ASR nella lingua rilevata (runs/asr_language_v1) e ricostruire la scheda"))
        if not act_available:
            flags.append(_flag("stem_activity_missing", "attività dello stem vocale non disponibile",
                               "ingressi e uscite verificati solo sulle parole ASR (niente controllo acustico)"))
    if asr and asr.get("share_low_confidence") is not None and asr["share_low_confidence"] > thr("asr_low_conf_share_max"):
        flags.append(_flag("asr_low_confidence", f"{100 * asr['share_low_confidence']:.0f} % delle parole con confidenza < "
                           f"{thr('asr_low_conf'):.1f} (mediana {asr.get('median_confidence') or 0:.2f})",
                           THRESHOLDS["asr_low_conf_share_max"][1], {"model": asr.get("model"), "language": asr.get("language")},
                           "fini verso da verificare all'ascolto"))
    bars = (analysis.get("bars") or {}).get("vocal") or []
    share = (sum(1 for v in bars if float(v or 0) >= 0.5) / len(bars)) if bars else None
    regions = analysis.get("vocal_regions") or []
    out = {"language": lang or None, "asr": dict(asr) if asr else None, "phrase_model": pm,
           "vocal_regions": regions, "vocal_bar_share": share,
           "first_vocal_s": float(regions[0]["start_s"]) if regions else None}
    return out, flags


def instrumental_regions(analysis: Mapping[str, Any], min_bars: Optional[int] = None) -> list[dict[str, Any]]:
    """Battute senza voce (``bars.vocal`` < 0,5 di dj_select) in corse di almeno ``min_bars``."""
    mb = int(thr("instrumental_min_bars") if min_bars is None else min_bars)
    voc = (analysis.get("bars") or {}).get("vocal") or []
    downs = (analysis.get("grid") or {}).get("downbeats_s") or []
    out, start = [], None
    for i, v in enumerate(list(voc) + [1.0]):
        quiet = i < len(voc) and float(v or 0) < 0.5
        if quiet and start is None:
            start = i
        elif not quiet and start is not None:
            if i - start >= mb:
                out.append({"start_bar": start, "end_bar": i, "bars": i - start,
                            "start_s": downs[start] if start < len(downs) else None,
                            "end_s": downs[i] if i < len(downs) else analysis.get("duration_s")})
            start = None
    return out


# ------------------------------------------------------------ cues (I/O-free)
def cue_rows(analysis: Mapping[str, Any], role: str) -> list[dict[str, Any]]:
    """Cue di dj_select con le finestre di default e «technique» (etichetta ``window``)."""
    from musicmix import dj_select as djs

    rows = djs.cue_candidates(analysis, role, windows=djs.CUE_WINDOWS["technique"])
    return [{"bar": int(c["bar_index"]), "time_s": float(c["time_s"]), "phrase_level": int(c["phrase_level"]),
             "section_label": c.get("section_label"), "section_start": bool(c.get("section_start")),
             "window": "default" if c.get("windows_default_ok", True) else "technique"} for c in rows]


def entries_section(analysis: Mapping[str, Any], phrase: Optional[Mapping[str, Any]], act: Any = None
                    ) -> list[dict[str, Any]]:
    from musicmix import dj_phrase as P

    rows = cue_rows(analysis, "in")
    test = P.entry_clear_test(phrase, act) if phrase is not None else None
    for r in rows:
        if test is None:
            r.update(status="unknown", ok=None, reasons=["nessun modello di versi"])
            continue
        e = test({"time_s": r["time_s"]})
        r.update(status=e.get("status"), ok=bool(e.get("ok")), evidence=e.get("evidence"),
                 lead_in_s=round(int(e.get("lead_in_samples") or 0) / float(phrase.get("sample_rate") or 48000), 3),
                 first_word=e.get("first_word"), acoustic=e.get("acoustic"), reasons=(e.get("reasons") or [])[:3])
    return rows


def exits_section(analysis: Mapping[str, Any], phrase: Optional[Mapping[str, Any]], act: Any = None) -> list[dict[str, Any]]:
    from musicmix import dj_phrase as P

    rows = cue_rows(analysis, "out")
    if phrase is None:
        for r in rows:
            r.update(status="unknown", line_end=None, clean_for_cut=None, reasons=["nessun modello di versi"])
        return rows
    pr = P.params(phrase.get("params"))
    for r in rows:
        e = P.exit_point(phrase, r["time_s"], pr, act)
        p_end = e.get("p_line_end")
        line_end = e["status"] in EXIT_LINE_END and (e["status"] == "instrumental" or (p_end or 0) >= pr["line_end_threshold"])
        r.update(status=e["status"], p_line_end=None if p_end is None else round(float(p_end), 3), last_word=e.get("last_word"),
                 line_end=bool(line_end), clean_for_cut=bool(line_end and e.get("acoustic") != "stem_active_in_window"),
                 truncates=(e.get("dry_cut_truncates") or [])[:4], reasons=(e.get("reasons") or [])[:3])
    return rows


def stretch_section(grid_flags: Sequence[Mapping[str, Any]], vocal_share: Optional[float]) -> dict[str, Any]:
    """Limiti di stretching consigliati [H]: ±8 % (MosaikBox [S]) e ±5 % per i blend lunghi (Kim 2020 [E]); ±5 % se la voce
    copre quasi tutto il brano; nessuna sovrapposizione consigliata se la griglia è incerta."""
    from musicmix import dj_select as djs

    blend = 100.0 * djs.TEMPO_MAX_STRETCH
    long_blend = 100.0 * djs.TEMPO_LONG_BLEND_STRETCH
    why = ["dj_select.TEMPO_MAX_STRETCH [S] MosaikBox ±8 %; TEMPO_LONG_BLEND_STRETCH [E] Kim 2020 §6.1: 86,1 % entro 5 %"]
    if vocal_share is not None and vocal_share > thr("vocal_bar_share_heavy"):
        blend = min(blend, long_blend)
        why.append(f"voce in {100 * vocal_share:.0f} % delle battute > {100 * thr('vocal_bar_share_heavy'):.0f} %: ±{long_blend:.0f} % [H]")
    risky = [f["id"] for f in grid_flags if f["id"] in ("tempo_octave_suspect", "grid_confidence_low", "grid_residual_high")]
    if risky:
        why.append("griglia incerta (" + ", ".join(risky) + "): sovrapposizioni sconsigliate [H]")
    return {"blend_max_pct": blend, "long_blend_max_pct": long_blend, "overlap_recommended": not risky, "why": why}


# ------------------------------------------------------------- corrections
def corrections_for(doc: Optional[Mapping[str, Any]], slug: str) -> list[dict[str, Any]]:
    if not doc:
        return []
    return [dict(c) for c in ((doc.get("tracks") or {}).get(slug) or [])]


def apply_corrections(card: dict[str, Any], corrections: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Registra e applica le correzioni manuali alla scheda (in place). Una correzione legata a un altro audio
    (``audio_sha256`` diverso) non si applica ed è segnalata come ``stale``."""
    applied, stale, pending = [], [], []
    sha = card["audio"]["sha256"]
    by_bar_in = {r["bar"]: r for r in card["entries"]}
    by_bar_out = {r["bar"]: r for r in card["exits"]}
    for c in corrections:
        c = dict(c)
        kind = c.get("kind")
        if kind not in CORRECTION_KINDS:
            stale.append({**c, "why_not": f"tipo sconosciuto {kind!r}"})
            continue
        if c.get("audio_sha256") and c["audio_sha256"] != sha:
            stale.append({**c, "why_not": "correzione legata a un altro audio (sha256 diverso)"})
            continue
        if kind == "exclude":
            card["excluded"] = {"by": c.get("id"), "why": c.get("why")}
        elif kind == "include":
            card["excluded"] = None
        elif kind == "verify_flag":
            hit = [f for f in card["uncertain"] if f["id"] == c.get("flag")]
            if not hit:
                stale.append({**c, "why_not": f"nessun caso incerto {c.get('flag')!r} in questa scheda"})
                continue
            for f in hit:
                f["resolved_by"] = c.get("id")
        elif kind in ("block_entry", "approve_entry", "block_exit", "approve_exit"):
            table = by_bar_in if kind.endswith("entry") else by_bar_out
            row = table.get(int(c.get("bar", -1)))
            if row is None:
                stale.append({**c, "why_not": f"battuta {c.get('bar')} non è un cue {'IN' if kind.endswith('entry') else 'OUT'}"})
                continue
            ok = kind.startswith("approve")
            row["manual"] = {"by": c.get("id"), "ok": ok, "why": c.get("why")}
            if kind.endswith("entry"):
                row["ok"] = ok
            else:
                row["line_end"] = ok
                row["clean_for_cut"] = ok and row.get("clean_for_cut", ok)
        elif kind in ("bpm_octave", "downbeat_offset_beats", "key"):
            pending.append(c)  # cambiano la griglia o la tonalità: valgono solo dopo una nuova analisi
            card["uncertain"].append(_flag("correction_pending_reanalysis", f"correzione {kind}={c.get('value')!r} in attesa di rianalisi",
                                           "la scheda descrive ancora l'analisi automatica; finché non è rifatta il brano è più rischioso",
                                           c, "rifare l'analisi DJ con la correzione e ricostruire la scheda"))
        elif kind == "note":
            card.setdefault("notes", []).append(f"[{c.get('id')}] {c.get('why') or c.get('value')}")
        applied.append(c)
    if any(c.get("kind") == "exclude" for c in applied) and card.get("excluded"):
        card["uncertain"].append(_flag("excluded_by_correction", "escluso a mano", card["excluded"].get("why") or "correzione manuale"))
    card["corrections"] = {"applied": applied, "stale": stale, "pending_reanalysis": pending}
    return card


# ------------------------------------------------------------------ card
def build_card(*, slug: str, entry: Mapping[str, Any], analysis: Optional[Mapping[str, Any]], inputs: Mapping[str, Any],
               phrase: Optional[Mapping[str, Any]] = None, act: Any = None, vocals_doc: Optional[Mapping[str, Any]] = None,
               language: Optional[Mapping[str, Any]] = None, form: Optional[Mapping[str, Any]] = None,
               form_summary: Optional[Mapping[str, Any]] = None, stems: Optional[Mapping[str, Any]] = None,
               audio: Optional[Mapping[str, Any]] = None, corrections: Sequence[Mapping[str, Any]] = (),
               built_at: Optional[str] = None, notes: Iterable[str] = ()) -> dict[str, Any]:
    """Scheda ``musicmix.track-card.1`` di un brano (vedi docstring del modulo)."""
    from musicmix import reliability as R

    audio = dict(audio or {"sha256": entry.get("track_id")})
    card: dict[str, Any] = {"schema_version": CARD_SCHEMA, "builder_version": BUILDER_VERSION, "slug": slug,
                            "artist": entry.get("artist"), "title": entry.get("title"), "audio": audio, "inputs": dict(inputs),
                            "input_key": input_key(inputs, corrections), "built_at": built_at, "excluded": None,
                            "uncertain": [], "notes": list(notes)}
    if audio.get("verified") is False:
        card["uncertain"].append(_flag("audio_sha_mismatch", "lo SHA-256 dell'audio canonico non coincide con track_id",
                                       "la scheda non è legata all'audio presente", audio))
    if analysis is None:
        card["uncertain"].append(_flag("analysis_missing", "analisi DJ (dj_select) assente", "nessuna griglia: scheda vuota", None,
                                       "attendere l'analisi del ponte (runs/dj_plan_bridge_v1/tracks)"))
        card.update(grid=None, key=None, structure=None, voice=None, entries=[], exits=[], instrumental_regions=[],
                    stems=dict(stems or {}), stretch=None)
        apply_corrections(card, corrections)
        card["usable"] = usable(card)
        card["reliability"] = R.track_reliability(card)
        return card
    if phrase is not None and analysis.get("grid"):
        same = abs(float(phrase.get("period_s") or 0) - float(analysis["tempo"]["period_s"])) < 1e-6 and \
            len(phrase.get("downbeats_s") or []) == len(analysis["grid"]["downbeats_s"])
        if not same:
            card["notes"].append("modello di versi su un'altra griglia: non usato")
            phrase, act = None, None
    g, gf = grid_section(analysis)
    k, kf = key_section(analysis)
    st, sf = structure_section(analysis, form, form_summary)
    asr = asr_stats(vocals_doc)
    v, vf = voice_section(analysis, phrase, inputs.get("phrase_model"), act is not None, asr, language)
    card.update(duration_s=analysis.get("duration_s"), grid=g, key=k, structure=st, voice=v)
    card["entries"] = entries_section(analysis, phrase, act)
    card["exits"] = exits_section(analysis, phrase, act)
    card["instrumental_regions"] = instrumental_regions(analysis)
    card["stems"] = dict(stems or {})
    card["uncertain"] += gf + kf + sf + vf
    voc = (stems or {}).get("vocals") or {}
    if voc and not voc.get("ok"):
        card["uncertain"].append(_flag("vocal_stem_missing", "stem vocale Demucs non disponibile o non verificato", voc.get("error") or "",
                                       None, "senza stem il brano non entra nel set (run_dj_set_v2_books.renderable)"))
    card["stretch"] = stretch_section(gf, v.get("vocal_bar_share"))
    clean_in = [r["bar"] for r in card["entries"] if r.get("ok")]
    line_out = [r["bar"] for r in card["exits"] if r.get("line_end")]
    if phrase is not None:
        if not clean_in:
            card["uncertain"].append(_flag("no_clean_entries", "nessun ingresso con la prima frase intera", "dj_phrase.entry_clear_test",
                                           {"n_in_cues": len(card["entries"])}, "rivedere i cue IN (approve_entry) o la trascrizione"))
        elif len(clean_in) < thr("min_clean_entries"):
            card["uncertain"].append(_flag("few_clean_entries", f"solo {len(clean_in)} ingresso pulito", THRESHOLDS["min_clean_entries"][1],
                                           {"bars": clean_in}))
        if card["exits"] and not line_out:
            card["uncertain"].append(_flag("no_line_end_exits", "nessuna uscita a fine verso tra i cue OUT", "dj_phrase.exit_point",
                                           {"n_out_cues": len(card["exits"])}))
    apply_corrections(card, corrections)  # dopo i casi automatici: verify_flag li può risolvere, block/approve cambiano i cue
    card["usable"] = usable(card)
    card["reliability"] = R.track_reliability(card)
    return card


def usable(card: Mapping[str, Any]) -> dict[str, Any]:
    """Ingressi e uscite utilizzabili (battute), dopo le correzioni manuali."""
    ent, ex = card.get("entries") or [], card.get("exits") or []
    return {"entries": [r["bar"] for r in ent if r.get("ok")],
            "entries_default_window": [r["bar"] for r in ent if r.get("ok") and r.get("window") == "default"],
            "exits_line_end": [r["bar"] for r in ex if r.get("line_end")],
            "exits_for_cut": [r["bar"] for r in ex if r.get("clean_for_cut")],
            "exits_mid_line": [r["bar"] for r in ex if r.get("status") == "mid_line"],
            "n_entries": len(ent), "n_exits": len(ex)}


def uncertain_summary(card: Mapping[str, Any], include_resolved: bool = False) -> list[str]:
    return [f["id"] for f in card.get("uncertain") or [] if include_resolved or not f.get("resolved_by")]


def new_corrections_doc() -> dict[str, Any]:
    return {"schema_version": CORRECTIONS_SCHEMA,
            "note": ("Correzioni manuali delle schede brano (track_cards/<slug>.json). Ogni voce: id, date, author, kind "
                     f"({', '.join(CORRECTION_KINDS)}), audio_sha256 (lega la correzione all'audio), flag/bar/value, why. "
                     "Si aggiungono con tools/build_track_cards.py correct; valgono in ogni set (musicmix.reliability)."),
            "tracks": {}}
