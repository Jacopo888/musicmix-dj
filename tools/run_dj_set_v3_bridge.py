#!/usr/bin/env python3
"""Set DJ v3 "ponte": catalogo del ponte congelato, profilo ``books``, dj_phrase, tecniche di dj_policy, veto del critico.

Strato sottile sopra ``tools/run_dj_set_v2_books.py`` (:func:`build_set` con :class:`SetHooks`): stessa
pipeline di ordine (Held–Karp), cue (Viterbi), render, QA e montaggio; qui cambiano solo gli ingressi e
quattro punti di estensione.

Fasi (riprendibili; scrive solo in ``runs/dj_set_v3_bridge``, ``.cache/dj_set_v3_bridge`` e nella cartella
``--page-out``):

* ``prepare``: legge ``runs/dj_set_v3_bridge/catalog_frozen.json`` (copia congelata di
  ``runs/pool_bridge_v1/catalog_ready.json``: il pool continua a crescere in parallelo), copia i JSON dei
  provider dei brani ``ready`` da ``runs/dj_plan_bridge_v1/providers`` (sola lettura), rifà l'analisi DJ con
  ``tools/dj_plan_pool.analyse_track`` in ``dj_plan/tracks`` e costruisce il modello di versi di
  :mod:`musicmix.dj_phrase` (parole ASR del JSON vocale, stem Demucs) in ``phrase/``.
* ``set``: :func:`run_dj_set_v2_books.build_set` con
  - guardia del cut = fine verso di ``dj_phrase.exit_clear_test`` (al posto della guardia stem/ASR della
    galleria v2) quando la trascrizione è affidabile; per la coppia ``dj_phrase.phrase_switch``
    (pickup di B intero, coda dell'ultima parola di A protetta); i cut si rendono con
    ``dj_phrase.render_phrase_transition`` (switch prima del downbeat, pickup con lo stem);
  - tecnica: se la prima proposta di ``dj_policy.propose(..., PROFILE_BOOKS)`` sul cue scelto è una regola
    di contesto giuntabile nel set (C2 build verso ritornello/drop, C2b halving build, C8 break; non C3/C4, vedi
    ``RENDERABLE``) si rende quella con :mod:`musicmix.dj_creative`, altrimenti la decisione dell'albero ``books``;
  - veto: QA ``dj_qa`` (fail, incluso doppio basso e parola tagliata) e flag del critico v3
    (``dj_critic.score_transition(..., mode='fast', version='v4')``: salto di loudness > 3 dB senza pausa dichiarata,
    distanza tonale > 1 solo con due brani sovrapposti, fase > 40 ms solo se il render non dichiara un residuo < 5 ms)
    → si prova il candidato successivo; blocco v4 e QA con ``rhythm_on_raw_a`` registrati come diagnostica;
  - set finale a -14 LUFS integrati, true peak <= -1 dBTP.
* ``gallery``: per ogni transizione (max 12) due versioni dello stesso cue: profilo default (albero di
  ``dj_set`` senza profilo, guardia v2, come ``dj_set_v1b``) e profilo libri (la tecnica del set, o la
  migliore alternativa dei libri se coincidono); clip secondo AGENTS.md; ordine casuale con seed salvato.
  Più le clip di ``runs/books_integration_check/clips`` come sezione separata.
* ``page``: ``tools/build_listening_artifact.py --blind`` sul manifest della galleria.

Piano per eventi (P0.5, 21/10/2026; ``conoscenza_mix/_meta/PIANO_EVENTI_IMPL_2026-10-21.md``), solo fase ``set``:

* ``--event-plan``: scrive nel ``set.json`` un documento ``musicmix.event_plan.1`` (:mod:`musicmix.dj_events`) per ogni
  transizione pianificata (``event_plans``: cue del Viterbi + tecnica che il set proverebbe per prima, cioè la policy dei
  libri se giuntabile, altrimenti l'albero) e, con il render, per la transizione accettata (``transitions[].event_plan``);
  registra anche che cosa avrebbe scelto l'opzione sperimentale (``event_pref_counterfactual``). Non cambia scelte né costi.
* ``--event-pref`` [H] sperimentale (spenta): ``dj_set.CUE_PARAMS['w_event_arrival']`` = ``EVENT_PREF_W``; il controfattuale
  diventa allora il default.
* ``--plan-only``: niente render, QA né montaggio (``build_set(no_render=True)``); ``--out`` scrive in un'altra run (gli
  ingressi congelati restano letti da ``runs/dj_set_v3_bridge``; cache delle forme separata); ``--form-params`` sostituisce
  ``dj_form.PARAMS`` (es. ``'{"kick_peak_on": null}'`` = regola della cassa di dj_form.2, per il confronto dell'808).

    python tools/run_dj_set_v3_bridge.py set --plan-only --event-plan --out runs/dj_set_v3_events_plan

Il QA e il critico verificano integrità tecnica, non gusto (AGENTS.md).

    python tools/run_dj_set_v3_bridge.py prepare set gallery page
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import run_dj_set_v2_books as v2  # noqa: E402

from musicmix import dj_creative as cr  # noqa: E402
from musicmix.paths import scratch_dir  # noqa: E402
from musicmix import dj_events as dje  # noqa: E402
from musicmix import dj_phrase as P  # noqa: E402
from musicmix import dj_policy as pol  # noqa: E402
from musicmix import dj_qa  # noqa: E402
from musicmix import dj_render as dj  # noqa: E402
from musicmix import dj_select as djs  # noqa: E402
from musicmix import dj_set as ds  # noqa: E402
from musicmix.paths import localize  # noqa: E402
from musicmix import track_filter as TF  # noqa: E402
from musicmix.stems import vocal_stem_path, vocals_json_for  # noqa: E402

base = v2.base
v1 = base.v1
RUN = "dj_set_v3_bridge"
SCRIPT = "tools/run_dj_set_v3_bridge.py"
OUT = ROOT / "runs" / RUN
FROZEN = OUT / "catalog_frozen.json"
PLAN = OUT / "dj_plan"
PHRASE = OUT / "phrase"
CACHE = ROOT / ".cache" / RUN
PROV_SRC = ROOT / "runs/dj_plan_bridge_v1/providers"
CHECK_DIR = ROOT / "runs/books_integration_check"
PAGE_OUT = scratch_dir("ascolti_set_v3_bridge")  # override with --page-out
DELIVER = ROOT / "deliverables"
SR = base.SR
QA_CFG = dj_qa.QAConfig(dropout_check=True)
QA_RAW_A = dj_qa.QAConfig(dropout_check=True, rhythm_on_raw_a=True)  # solo registrato per le transizioni finali (giudice v4)  # dj_qa.4 con il controllo D4 (Huron p. 52) acceso: warn solo se il buco non è dichiarato
ORDER_OVERRIDES = {"len_min": 12, "len_target": 14}  # richiesta: 12-16 brani; max_exact 15 => al massimo 15
CONTEXT_RULES = ("C2_chorus", "C2b_halving", "C3_verse_handoff", "C4_extend", "C8_break")
#: tecniche della policy rese nel set. loop_extend, acapella_over e vocal_swap (B stirato) sono escluse: i metadati di
#: dj_creative non danno il campione sorgente di B alla fine della regione (overlap.b_source_end_sample, b_map.src_start = None),
#: quindi dj_set.clip_geometry non può giuntare la clip con il resto di B (KeyError nella seconda esecuzione, 21/10); nelle
#: due esecuzioni precedenti erano comunque tutte bocciate dal QA del ritmo (downbeat 258 ms, 0 downbeat appaiati).
RENDERABLE = ("halving_build", "break_before_entry", "build_into_chorus")
NOT_SPLICEABLE = ("acapella_over", "vocal_swap", "loop_extend")
OVERLAP_KINDS = ("blend", "loop_extend", "acapella_over", "vocal_swap")  # due brani suonano insieme: conta la tonalità
VETO_FLAGS = ("beat_phase_ms", "loud_jump_db")  # + chroma_dist solo in OVERLAP_KINDS (critico v3, REPORT §10)
MAX_GALLERY = 12
#: brani esclusi a mano dopo la verifica (orchestratore 21/10: "antidote a 65,6 BPM, probabilmente metà tempo")
ASR_LANG_DIR = ROOT / "runs/asr_language_v1/vocals"  # usato solo per i modelli dj_phrase non ancora calcolati
EXCLUDE = {"antidote": "tempo ambiguo d'ottava: allin1 e Beat This! 65,58 BPM, madmom 131,14 BPM (octave_ambiguous); con la griglia a "
                       "metà tempo una 'battuta' sono due battute reali e il downbeat può cadere sul terzo battito: escluso finché la "
                       "griglia non è verificata all'ascolto"}
GALLERY_SEED = 20261021
TECH_LABEL = {"halving_build": "halving build (loop che si dimezza)", "break_before_entry": "break prima dell'ingresso",
              "build_into_chorus": "build verso il ritornello", "acapella_over": "acapella di B sullo strumentale di A",
              "vocal_swap": "vocal swap", "loop_extend": "loop di A sotto B", "backspin": "backspin"}


# ------------------------------------------------------------------ utils
def load(p: Path) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def dump(p: Path, obj: Any) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(base.jsonable(obj), indent=1, ensure_ascii=False, default=str), encoding="utf-8")


def sha(p: Path) -> str:
    return base.sha256(Path(p))


def ready_entries(cat: dict) -> tuple[list[dict], list[dict]]:
    ok, skipped = [], []
    for t in cat["tracks"]:
        if t.get("status") == "ready":
            ok.append(t)
        else:
            skipped.append({"slug": t["slug"], "why": f"stato '{t.get('status')}' nel catalogo congelato, mancano {t.get('missing')}"})
    return ok, skipped


# ---------------------------------------------------------------- prepare
def _act_to_npz(act: P.StemActivity, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, centers=act.centers, active=act.active, rms_db=act.rms_db, threshold_db=np.array(act.threshold_db),
                        f0_hz=act.f0_hz, onsets=act.onsets, hop=np.array(act.hop),
                        active_strict=act.active_strict if act.active_strict is not None else np.zeros(0, bool))


def _act_from_npz(path: Path) -> P.StemActivity:
    z = np.load(path)
    strict = z["active_strict"]
    return P.StemActivity(centers=z["centers"], active=z["active"], rms_db=z["rms_db"], threshold_db=float(z["threshold_db"]),
                          f0_hz=z["f0_hz"], onsets=z["onsets"], hop=int(z["hop"]), active_strict=strict if strict.size else None)


def prepare() -> dict:
    import dj_plan_pool  # tools/dj_plan_pool.py: stessa analisi DJ della pipeline

    cat = load(FROZEN)
    ready, skipped = ready_entries(cat)
    (PLAN / "providers").mkdir(parents=True, exist_ok=True)
    (PLAN / "tracks").mkdir(parents=True, exist_ok=True)
    lex = P.Lexicon()
    rows = []
    for t in ready:
        slug = t["slug"]
        t0 = time.time()
        src, dst = PROV_SRC / f"{slug}.json", PLAN / "providers" / f"{slug}.json"
        if not dst.is_file():
            if not src.is_file():
                skipped.append({"slug": slug, "why": "provider assenti in runs/dj_plan_bridge_v1/providers"})
                continue
            shutil.copy2(src, dst)
        prov = load(dst)
        missing = [k for k in ("allin1", "madmom_key", "essentia_key") if (prov.get(k) or {}).get("status") != "complete"]
        if missing:
            skipped.append({"slug": slug, "why": f"provider incompleti: {missing}"})
            continue
        tj = PLAN / "tracks" / f"{slug}.json"
        if not tj.is_file():
            print(f"analisi DJ di {slug}", flush=True)
            a = dj_plan_pool.analyse_track(t, PLAN / "providers")
            tj.write_text(dj_plan_pool.dump(a), encoding="utf-8")
        track = load(tj)
        why = v2.renderable(slug, {"plan_dir": PLAN, "catalog": cat})
        if why:  # e.g. the V12 pair reminder/too_many_nights: canonical audio restored, Demucs stems not on Linux
            skipped.append({"slug": slug, "why": why})
            print(f"{slug}: escluso ({why})", flush=True)
            continue
        pj, npz = PHRASE / f"{slug}.json", CACHE / "act" / f"{slug}.npz"
        if not (pj.is_file() and npz.is_file()):
            print(f"modello dj_phrase di {slug}", flush=True)
            vjson = vocals_json_for(t)  # catalogo, oppure override additivo (vocals_overrides.json)
            redo = ASR_LANG_DIR / slug / "a.json"  # ASR rifatto nella lingua rilevata (es. 18k_no_kizzy in italiano)
            if redo.is_file():
                vjson = redo
            voc = load(vjson)
            be = voc.get("backend") or {}
            lang = be.get("detected_language") or be.get("language") or "en"
            stem_path = vocal_stem_path(t, vocals_json=vjson)  # stem registrato e verificato (anche con ASR rifatto)
            stem, sr = dj.load_track(stem_path)
            if sr != SR:
                raise RuntimeError(f"{slug}: stem a {sr} Hz")
            act = P.StemActivity.from_audio(stem, sr)
            del stem
            model = P.analyze_track(track, voc["words"], act=act, lang=lang, lexicon=lex,
                                    asr_language_probability=be.get("language_probability"))
            _act_to_npz(act, npz)
            dump(pj, {"schema_version": "musicmix.dj_phrase_run.1", "slug": slug,
                      "inputs": {"track_analysis": base.rel(tj), "track_analysis_sha256": sha(tj),
                                 "vocals_json": base.rel(Path(vjson)), "vocals_json_sha256": sha(Path(vjson)),
                                 "vocal_stem": base.rel(stem_path), "asr_model": be.get("model_path"), "asr_language": lang,
                                 "asr_language_probability": be.get("language_probability")},
                      "model": model})
        m = load(pj)["model"]
        rows.append({"slug": slug, "bpm": track["tempo"]["bpm"], "camelot": track["key"]["camelot"], "key": track["key"]["key"],
                     "key_level": track["key"]["level"], "n_bars": track["grid"]["n_bars"],
                     "asr_reliable": bool((m.get("asr_quality") or {}).get("reliable", False)),
                     "asr_reasons": (m.get("asr_quality") or {}).get("reasons"), "n_lines": m["summary"].get("n_lines"),
                     "elapsed_s": round(time.time() - t0, 1)})
        print(rows[-1], flush=True)
    doc = {"schema_version": "musicmix.dj_set_v3_prepare.1", "catalog_frozen": base.rel(FROZEN), "catalog_frozen_sha256": sha(FROZEN),
           "catalog_source": "runs/pool_bridge_v1/catalog_ready.json (copiato all'avvio, 21/10/2026)",
           "providers_source": base.rel(PROV_SRC), "tracks": rows, "skipped": skipped}
    dump(OUT / "prepare.json", doc)
    return doc


# --------------------------------------------------------------- phrase state
MODELS: dict[str, dict] = {}
ACTS: dict[str, P.StemActivity] = {}
_EXIT: dict[tuple, dict] = {}
_ENTRY: dict[tuple, dict] = {}


def load_phrase(slugs) -> None:
    for s in slugs:
        if s not in MODELS:
            MODELS[s] = load(PHRASE / f"{s}.json")["model"]
            ACTS[s] = _act_from_npz(CACHE / "act" / f"{s}.npz")


def reliable(slug: str) -> bool:
    return bool((MODELS[slug].get("asr_quality") or {}).get("reliable", False))


def exit_at(slug: str, t: float) -> dict:
    k = (slug, round(float(t), 4))
    if k not in _EXIT:
        _EXIT[k] = P.exit_point(MODELS[slug], float(t), act=ACTS[slug])
    return _EXIT[k]


def entry_at(slug: str, t: float) -> dict:
    k = (slug, round(float(t), 4))
    if k not in _ENTRY:
        _ENTRY[k] = P.entry_point(MODELS[slug], float(t), act=ACTS[slug])
    return _ENTRY[k]


def phrase_summary(a: str, b: str, a_out: float, b_in: float, period_a: float) -> dict:
    ea, eb = exit_at(a, a_out), entry_at(b, b_in)
    sw = P.phrase_switch(ea, eb, period_a)
    return {"exit": {k: ea.get(k) for k in ("status", "p_line_end", "last_word", "next_word", "krims_closure", "acoustic")},
            "exit_reasons": ea.get("reasons"), "entry": {k: eb.get(k) for k in ("status", "pickup_words", "first_word", "line_start_p")},
            "entry_pickup_s": (eb.get("pickup_samples") or 0) / SR, "switch_ok": bool(sw.get("ok")), "switch_rel_s": sw.get("switch_rel_s"),
            "tail_s": ((ea["tail_end_sample"] - ea["downbeat_sample"]) / SR) if ea.get("tail_end_sample") else None,
            "tail_end_a_s": (ea["tail_end_sample"] / SR) if ea.get("tail_end_sample") else None,
            "switch_reasons": sw.get("reasons"), "a_asr_reliable": reliable(a), "b_asr_reliable": reliable(b)}


# --------------------------------------------------------------------- hooks
def sources_v3(log: list) -> dict[str, dict]:
    cat = load(FROZEN)
    ready, skipped = ready_entries(cat)
    keep: dict[str, dict] = {}
    for t in ready:
        s = t["slug"]
        if s in EXCLUDE:
            skipped.append({"slug": s, "why": EXCLUDE[s]})
            continue
        if not (PLAN / "tracks" / f"{s}.json").is_file() or not (PHRASE / f"{s}.json").is_file():
            skipped.append({"slug": s, "why": "analisi DJ o modello dj_phrase assenti (fase prepare)"})
            continue
        src = {"plan_dir": PLAN, "catalog": cat, "catalog_path": FROZEN, "source": "pool_bridge_v1 (catalogo congelato, runs/dj_set_v3_bridge/dj_plan)"}
        why = v2.renderable(s, src)
        if why:
            skipped.append({"slug": s, "why": why})
            continue
        keep[s] = src
    load_phrase(keep)
    log.append({"pool_v3": {"tracks": sorted(keep), "skipped": skipped, "catalog_frozen_sha256": sha(FROZEN)}})
    return keep


def info_v3(slug: str, src: dict) -> dict:
    """Feature del brano (cache di run_dj_set_v1) con la guardia del cut sostituita dalla fine verso di dj_phrase."""
    info = dict(base.track_info(slug, src))
    info["cut_clear_v2"] = list(info["cut_clear"])
    if reliable(slug):
        test = P.exit_clear_test(MODELS[slug], ACTS[slug])
        info["cut_clear"] = [bool(test({"time_s": float(t)})) for t in info["downbeats_s"]]
        info["cut_guard"] = "dj_phrase.exit_clear_test (fine verso p >= 0,5, finestra senza parole, stem silenzioso)"
    else:
        info["cut_guard"] = "guardia v2 (stem + parole ASR): trascrizione non affidabile per dj_phrase"
    return info


def _order_over() -> dict:
    return dict(v2.PROFILE["order"], **ORDER_OVERRIDES)


def _drop_cut(c: dict, why: str) -> None:
    """Come run_dj_set_v1.apply_filters: niente cut, prima alternativa non-cut dell'albero."""
    d = c["decision"]
    fbs = [f for f in d["fallbacks"] if f["type"] != "cut"]
    if d["type"] == "cut":
        alt = fbs[0] if fbs else ds._mk("echo_out", ds.values(ds.TREE_PARAMS))
        c["decision"] = {**d, **alt, "type_key": alt.get("type_key", alt["type"]), "fallbacks": fbs[1:], "rule": d["rule"] + "+phrase_guard"}
    else:
        c["decision"] = {**d, "fallbacks": fbs}
    c["notes"].append(why)


#: Ingresso di B (21/10/2026, conoscenza_mix/_meta/INGRESSO_B_2026-10-21.md): acceso di default. Guardia dei cue di B
#: (dj_set.entry_guard + dj_phrase.entry_clear_test), lead-in di B nel render (dj_render.add_b_lead_in, modo stem) e parole
#: raffinate di B nel QA (dj_qa.6, controllo b_entry). ``--no-b-entry-fix`` (o B_ENTRY["on"] = False) riproduce il set v3.
B_ENTRY = {"on": True, "mode": "stem", "a_close": "auto"}


def entry_test_for(slug: str):
    return P.entry_clear_test(MODELS[slug], ACTS.get(slug))


def b_lead_in_at(slug: str, t: float) -> dict | None:
    if slug not in MODELS:
        return None
    return P.b_lead_in(MODELS[slug], float(t), act=ACTS.get(slug))


def words_b_refined(slug: str) -> list | None:
    """Parole di B raffinate da dj_phrase (on/off sullo stem) nel formato di dj_qa (start_sample/end_sample)."""
    if slug not in MODELS:
        return None
    return [{"start_sample": int(w["on"]), "end_sample": int(w["off"]), "text": w["text"]} for w in MODELS[slug]["words"]
            if not w.get("dropped")]


def with_b_lead_in(res, decks, B, plan: dict, dec: dict, A=None):
    """Lead-in di B (pickup / prima frase intera) sulle tecniche a ingresso pulito, se dj_phrase lo chiede."""
    kind = res.metadata.get("kind")
    if not B_ENTRY["on"] or kind not in dj.B_LEAD_IN_KINDS:
        return res, decks
    li = plan.get("b_lead_in") or b_lead_in_at(B.slug, plan["b_in_s"])
    if not li or li.get("status") != "lead_in" or int(li.get("lead_in_samples") or 0) <= 0:
        return res, decks
    va = None
    if decks is not None and A is not None:  # dj_effects: A's vocal stem in clip coordinates (creative renders carry vocals_a_heard)
        va = dj_qa.stems_to_output(res.metadata, len(res.audio), A.vocals)[0]
    res2 = dj.add_b_lead_in(res, B.audio, int(li["lead_in_samples"]), vocals_b=B.vocals, mode=B_ENTRY["mode"],
                            a_close=B_ENTRY["a_close"], vocals_a_clip=va, info={k: li.get(k) for k in ("status", "lead_in_words", "first_word",
                                                                                       "phrase_words", "evidence", "reasons")})
    if decks is None:
        return res2, None
    n = len(res2.audio)
    t = np.arange(n)
    meta = res2.metadata
    new = dict(res2.decks)
    a_end = int((meta.get("effect") or {}).get("dry_a_cut_output_sample") or meta["switch"]["output_sample"])
    if kind == "filter_out":
        a_end = int(meta["b_entry"].get("a_closed_output_sample") or meta["switch"]["output_sample"])
    new["gains"] = dict(new.get("gains", {}), fader_a=(t < a_end).astype(float),
                        fader_b=(t >= int(meta["b_entry"]["audible_output_sample"])).astype(float))
    return res2, new


def filter_pair_v3(pc: dict, a: str, b: str, analyses: dict) -> dict:
    """Cut solo con fine verso di A e pickup di B interi (dj_phrase.phrase_switch); piano di switch per il render.

    Con ``B_ENTRY['on']`` prima la guardia dell'ingresso di B (dj_set.entry_guard): cue di B spostati o non ammissibili
    se B entrerebbe a metà frase."""
    if B_ENTRY["on"] and b in MODELS:
        ds.entry_guard(pc, analyses[a], analyses[b], entry_test_for(b), dict(v2.PROFILE["tree"]), _order_over())
    period = float(analyses[a]["tempo"]["period_s"])
    changed = False
    for c in pc["candidates"]:
        plan = c["plan"]
        has_cut = c["decision"]["type"] == "cut" or any(f["type"] == "cut" for f in c["decision"]["fallbacks"])
        if not has_cut or not (reliable(a) or reliable(b)):
            continue
        ps = phrase_summary(a, b, plan["a_out_s"], plan["b_in_s"], period)
        if ps["switch_ok"]:
            plan["phrase_switch"] = {"switch_rel_s": ps["switch_rel_s"], "tail_end_a_s": ps["tail_end_a_s"],
                                     "pickup_s": ps["entry_pickup_s"], "pickup_words": ps["entry"]["pickup_words"],
                                     "last_word": ps["exit"]["last_word"]}
        else:
            _drop_cut(c, "cut non ammesso da dj_phrase: " + "; ".join(ps["switch_reasons"] or [str(ps["exit"]["status"])]))
            c["cost"] = ds.candidate_cost(plan, c["decision"], _order_over(), c["context"])
            changed = True
    if changed:
        pc["candidates"].sort(key=lambda c: (not c["admissible"], c["cost"], -c["plan"]["score"]))
    return pc


def creative_decision(p: dict) -> dict:
    tech = p["technique"]
    params = {k: v for k, v in p["params"].items()}
    return {"type": tech, "type_key": tech, "label": TECH_LABEL.get(tech, tech), "rule": p["rule"], "rule_source": p["rule_source"],
            "render": {"technique": tech, "params": params}, "bars": max(1, int(round(float(p["duration_bars"])))),
            "fallbacks": [], "creative": True,
            "trace": {"policy_why": p["why"], "prior": p["prior"], "score": p["score"], "risks": p["risks"], "prior_source": p["prior_source"]}}


POLICY: dict[int, dict] = {}
ATTEMPTS: dict[int, list] = {}


def policy_for(a_an: dict, b_an: dict, plan: dict, pair: dict, recent: list) -> dict:
    pctx = pol.creative_context(a_an, b_an, plan, pair, has_stems=True, recent_techniques=recent[-2:])
    return pol.propose(pctx, params=pol.PROFILE_BOOKS)


def attempts_v3(ctx: dict) -> list:
    t, a, b, i0 = ctx["t"], ctx["a"], ctx["b"], ctx["idx0"]
    c0 = ctx["chosen"]
    props = policy_for(ctx["analyses"][a], ctx["analyses"][b], c0["plan"], ctx["pc"]["compatibility"], ctx["recent"])
    top = props["proposals"][0] if props["proposals"] else None
    first = []
    if top is not None and top["rule"] in CONTEXT_RULES and top["technique"] in RENDERABLE:
        for p in props["proposals"]:
            if p["rule"] in CONTEXT_RULES and p["technique"] in RENDERABLE:
                first.append((i0, creative_decision(p), f"policy dei libri ({p['rule']}): {'; '.join(p['why'])}"))
    POLICY[t] = {"top": None if top is None else {k: top[k] for k in ("technique", "rule", "score")},
                 "proposals": [{k: p[k] for k in ("technique", "rule", "score", "why", "duration_bars", "params")} for p in props["proposals"]],
                 "used_policy_first": bool(first),
                 "why_not": None if first else (
                     "nessuna proposta" if top is None else
                     f"prima proposta {top['technique']} ({top['rule']}) con B stirato: non giuntabile nel set (NOT_SPLICEABLE); vale l'albero libri"
                     if top["technique"] in NOT_SPLICEABLE else
                     "prima proposta generica (C5-C7: uscite con effetto): vale la decisione dell'albero libri")}
    if TENSION["mode"] == "prefer":  # base continua (dj_tension, [user]): la stessa tecnica senza pausa prima di quella con pausa
        first = [x for f in first for x in ((_no_gap(f),) if _no_gap(f) is not None else ()) + (f,)]
    plan = first + list(ctx["plan"])
    ATTEMPTS[t] = plan
    return plan


def _no_gap(att):
    """Copia di un tentativo creativo con ``gap_beats`` > 0 senza la pausa (None se non serve)."""
    i, dec, why = att
    pr = ((dec.get("render") or {}).get("params") or {})
    if not dec.get("creative") or not float(pr.get("gap_beats") or 0) > 0:
        return None
    d2 = copy.deepcopy(dec)
    d2["render"]["params"]["gap_beats"] = 0
    d2["label"] = f"{dec['label']} (senza pausa)"
    d2["no_gap_variant"] = True
    return (i, d2, why + "; variante senza pausa (dj_tension: base continua, [user])")


def lead_a_v3(cand: dict, A: dict) -> float:
    d = cand["decision"]
    if not d.get("creative"):
        return ds.lead_a_s(cand, A)
    per = float(A["tempo"]["period_s"])
    pr = d["render"]["params"]
    if d["type"] in ("halving_build", "build_into_chorus"):
        return float(pr.get("build_bars", 4)) * 4 * per
    if d["type"] == "break_before_entry":
        return float(pr.get("beats", 1)) * per
    return 0.0


def overlap_b_v3(cand: dict, B: dict) -> float:
    d = cand["decision"]
    if not d.get("creative"):
        return ds.overlap_b_s(cand, B)
    bar = 4 * float(B["tempo"]["period_s"])
    pr = d["render"]["params"]
    if d["type"] in ("loop_extend", "acapella_over"):
        return float(pr.get("bars", 4)) * bar
    if d["type"] == "vocal_swap":
        return float(pr.get("bars", 4) + pr.get("out_bars", 2)) * bar
    return 0.0


def declare_entry_relocation(res, plan: dict):
    """Provenienza dello spostamento del cue di B (guardia d'ingresso, INGRESSO_B R-B2) nei metadati del render.

    ``dj_set.entry_guard`` -> ``dj_select.relocate_entry`` lascia ``plan['entry_relocation']`` (``from_bar``, ``to_bar``,
    ``kind``, ``shift_bars``). Con ``kind == 'lead_bars'`` lo stesso cue è anticipato di 1-2 battute (l'evento di B arriva
    ancora): l'ingresso è fuori dalla griglia di frase **per scelta**, e ``dj_qa.check_phrase`` (.7) lo tratta come ``warn``
    dichiarato se trova ``meta['b_entry']['relocated_bars']`` (> 0, battute prima dell'inizio frase). Un ``cue_candidate``
    è già un inizio frase (``dj_select.cue_candidates``): solo provenienza, nessun ``relocated_bars``. Non cambia il PCM."""
    rel = plan.get("entry_relocation") if isinstance(plan, dict) else None
    if not rel:
        return res
    be = res.metadata.get("b_entry")
    if not isinstance(be, dict):
        be = res.metadata["b_entry"] = {}
    shift = int(rel.get("shift_bars") or 0)
    be["relocation"] = {"from_bar": rel.get("from_bar"), "to_bar": rel.get("to_bar"), "kind": rel.get("kind"), "shift_bars": shift,
                        "source": f"{ds.ENTRY_GUARD_VERSION} / dj_select.relocate_entry (INGRESSO_B R-B2)"}
    if rel.get("kind") == "lead_bars" and shift < 0:
        be["relocated_bars"] = -shift
    return res


def render_v3(dec: dict, A, B, plan: dict, prov: dict):
    res, decks = _render_v3(dec, A, B, plan, prov)
    return declare_entry_relocation(res, plan), decks


def _render_v3(dec: dict, A, B, plan: dict, prov: dict):
    grids = dict(beats_a=A.beats, downbeats_a=A.downbeats, beats_b=B.beats, downbeats_b=B.downbeats)
    ia = int(np.argmin(np.abs(A.downbeats - plan["a_out_s"])))
    a_out, b_in = float(A.downbeats[ia]), float(plan["b_in_s"])
    if dec.get("creative"):
        r = dec["render"]
        res = cr.render_creative(r["technique"], A.audio, B.audio, **grids, cue_out_a=a_out, cue_in_b=b_in, vocals_a=A.vocals,
                                 vocals_b=B.vocals, sample_rate=SR, config=v2.CR_CFG, provenance=prov, return_decks=True, **r["params"])
        return with_b_lead_in(res, None, B, plan, dec, A)
    if dec["type"] == "cut" and plan.get("phrase_switch"):
        ps = plan["phrase_switch"]
        res = P.render_phrase_transition("cut", A.audio, B.audio, **grids, cue_out_a=a_out, cue_in_b=b_in,
                                         switch_rel_s=float(ps["switch_rel_s"] or 0.0), tail_end_a_s=ps.get("tail_end_a_s"),
                                         vocals_a=A.vocals, vocals_b=B.vocals, pickup_mode="stem", sample_rate=SR,
                                         render_config=base.CUT_CFG, provenance=dict(prov, phrase_switch=ps))
        return res, res.decks
    res, decks = v2.render(dec, A, B, plan, prov)
    return with_b_lead_in(res, decks, B, plan, dec, A)


def qa_v3(res, A, B, decks, tracker, cfg: dj_qa.QAConfig | None = None) -> dict:
    """QA dj_qa.4 con il controllo del buco acceso (stessi ingressi di run_dj_set_v1/v2)."""
    cfg = cfg or QA_CFG
    if cfg is QA_CFG:
        _LAST_QA.update(res=res, decks=decks, tracker=tracker)
    meta = res.metadata
    n = len(res.audio)
    wb_src = (words_b_refined(B.slug) if B_ENTRY["on"] else None) or None
    words_b = dj_qa.words_to_output(meta, wb_src, n, deck="b") if wb_src else None
    if decks is None:  # dj_creative: deck e voci udibili dal render
        qi = cr.qa_inputs(res)
        ga = 10 ** (meta["loudness"]["a_gain_db"] / 20.0)
        gb = 10 ** (meta["loudness"]["b_gain_db"] / 20.0)
        return dj_qa.evaluate_render(res.audio.astype(np.float64), meta, sample_rate=res.sample_rate, decks=qi["decks"],
                                     vocals_a=qi["vocals_a"], vocals_b=qi["vocals_b"],
                                     vocal_thresholds=(A.vocal_threshold * ga, B.vocal_threshold * gb),
                                     words_a=cr.words_in_clip(meta, A.words, "a"), tracker=tracker, config=cfg,
                                     grid_support={"a": A.analysis["tempo"], "b": B.analysis["tempo"]}, words_b=words_b)
    va, vb = dj_qa.stems_to_output(meta, n, A.vocals, B.vocals)
    words = dj_qa.words_to_output(meta, A.words, n)
    ph = meta.get("phrase") or {}
    if ph.get("switch_rel_samples"):  # phrase cut: A's words after its fader switch are not heard (except the protected tail)
        sw = meta["switch"]
        cut_src = int(sw["fader_switch_a_source_sample"])
        tail = (ph.get("tail") or {})
        keep_to = int(sw["a_source_sample"]) + (int(tail["output_end_sample"]) - int(sw["output_sample"])) if tail else cut_src
        heard = [w for w in A.words if w["start_sample"] < max(cut_src, keep_to)]
        words = dj_qa.words_to_output(meta, heard, n)
    return dj_qa.evaluate_render(res.audio.astype(np.float64), meta, sample_rate=res.sample_rate, decks=decks, vocals_a=va, vocals_b=vb,
                                 vocal_thresholds=(A.vocal_threshold, B.vocal_threshold), words_a=words, tracker=tracker,
                                 config=cfg, grid_support={"a": A.analysis["tempo"], "b": B.analysis["tempo"]}, words_b=words_b)


PHASE_DECLARED_MAX_MS = 5.0  # orchestratore 21/10 (giudice v4): il flag di fase non è un veto se il render dichiara residuo < 5 ms


def declared_phase(meta: dict) -> dict:
    """Fase di B rispetto alla griglia di A dichiarata dal render (non misurata sulla clip).

    Blend (dj_render): ``downbeat_alignment.max_abs_error_ms`` (griglia pianificata + offset dello stretcher).
    Ingresso pulito senza sovrapposizione (cut, echo, filter, tecniche creative, cut di dj_phrase): il downbeat
    di B è posto esattamente sul campione d'ingresso, che è un downbeat della griglia di A (residuo 0 per
    costruzione; resta vero solo quanto le griglie delle analisi)."""
    da = meta.get("downbeat_alignment") or {}
    if da.get("max_abs_error_ms") is not None:
        return {"ms": float(da["max_abs_error_ms"]), "source": "downbeat_alignment.max_abs_error_ms (blend beatmatchato)"}
    if "b_entry_output_sample" in da or meta.get("kind") in ("cut",) or meta.get("creative") or meta.get("phrase"):
        return {"ms": 0.0, "source": "downbeat di B posto sul downbeat di A (nessuna sovrapposizione ritmica; griglie delle analisi)"}
    return {"ms": None, "source": "non dichiarata"}


def critic(res, kind: str, ident: str, decks=None, qa=None, declared_gap: bool = False) -> dict:
    """Critico v3 'fast' (DSP: Beat This!, BS.1770, Krumhansl-Kessler, Silero) con il blocco v4 come diagnostica.

    Veto: salto di loudness > 3 dB (non con una pausa dichiarata dal render), distanza tonale > 1 solo con due brani
    sovrapposti; la fase > 40 ms è veto solo se il render non dichiara un residuo < 5 ms (falsi positivi di Beat This!
    sulla clip mixata, runs/dj_critic_v4/REPORT.md)."""
    from musicmix import dj_critic

    try:
        r = dj_critic.score_transition(res.audio, res.metadata, None, mode="fast", sample_rate=res.sample_rate, id=ident,
                                       version="v4", decks=decks, qa=qa)
    except Exception as exc:  # misura impossibile (lati troppo corti) o v4 non calcolabile: riprova v3, poi registra
        try:
            r = dj_critic.score_transition(res.audio, res.metadata, None, mode="fast", sample_rate=res.sample_rate, id=ident)
            r["v4_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc2:
            return {"error": f"{type(exc2).__name__}: {exc2}", "flags": [], "veto": [], "declared_phase": declared_phase(res.metadata)}
    dp = declared_phase(res.metadata)
    veto, diag = [], []
    for f in r["flags"]:
        msg = f"critico: {f['message']} ({f['value']:.2f} > {f['threshold']:g})"
        if f["flag"] == "beat_phase_ms":
            if dp["ms"] is not None and dp["ms"] < PHASE_DECLARED_MAX_MS:
                diag.append(msg + f" — non veto: fase dichiarata dal render {dp['ms']:.2f} ms ({dp['source']})")
                continue
            veto.append(msg)
        elif f["flag"] == "loud_jump_db":
            if declared_gap:
                diag.append(msg + " — non veto: pausa dichiarata dal render (break/halving)")
                continue
            veto.append(msg)
        elif f["flag"] == "chroma_dist" and kind in OVERLAP_KINDS:
            veto.append(msg)
        else:
            diag.append(msg + " — diagnostica")
    craft = r["parts"]["craft"]
    return {"score": r["score"], "uncertainty": r["uncertainty"], "version": r["version"], "flags": r["flags"], "veto": veto,
            "diagnostic_flags": diag, "declared_phase": dp, "v4": r.get("v4"), "v4_error": r.get("v4_error"),
            "craft": {k: craft.get(k) for k in ("loud_jump_db", "beat_phase_ms", "chroma_dist", "chroma_shift_gain", "vad_at_cut",
                                                "loud_pre_lufs", "loud_post_lufs")},
            "note": "vad_at_cut registrato ma non usato come veto (Silero non informativo sul rap, REPORT §10); chroma_dist è un "
                    "veto solo con due brani sovrapposti; beat_phase_ms solo se il render non dichiara residuo < 5 ms (giudice v4)"}


def declared_gap(dec: dict) -> bool:
    pr = (dec.get("render") or {}).get("params") or {}
    return bool((dec["type"] == "halving_build" and float(pr.get("gap_beats") or 0) > 0)
                or (dec["type"] == "break_before_entry" and pr.get("mode") == "stop")
                or (dec["type"] == "build_into_chorus" and float(pr.get("gap_beats") or 0) > 0))


_LAST_QA: dict = {}


def veto_v3(res, qa, dec, A, B, rec, t) -> list:
    cr_ = critic(res, dec["type"], f"t{t + 1:02d}_{rec['candidate']}_{dec['type_key']}", decks=res.decks, qa=qa,
                 declared_gap=declared_gap(dec))
    rec["critic"] = cr_
    if qa["verdict"] != "fail" and not cr_["veto"] and _LAST_QA.get("res") is res:
        # transizione che sarà accettata: QA ripetuto con il ritmo di A misurato prima dell'EQ (solo registrato)
        L = _LAST_QA
        try:
            q2 = qa_v3(res, A, B, L["decks"], L["tracker"], cfg=QA_RAW_A)
            rec["qa_rhythm_on_raw_a"] = {"verdict": q2["verdict"], "reasons": [f"{r['check']}: {r['message']}" for r in q2["reasons"]]}
        except Exception as exc:
            rec["qa_rhythm_on_raw_a"] = {"error": f"{type(exc).__name__}: {exc}"}
    return cr_["veto"]


BOOK_SOURCES: dict[int, list] = {}


def on_accept_v3(t, a, b, res, dec, c, rec, qa) -> dict:
    name = f"t{t + 1:02d}_accepted"
    w = dj.write_render(res, Path(_CTX.get("out", OUT)) / "transitions", name)  # la run di --out, mai quella congelata
    plan = c["plan"]
    ps = phrase_summary(a, b, plan["a_out_s"], plan["b_in_s"], float(load(PLAN / "tracks" / f"{a}.json")["tempo"]["period_s"]))
    rules = [{"what": "decisione", "rule": dec["rule"], "source": dec.get("rule_source")}]
    ev_extra = {}
    if EVENT_PLAN["on"] or EVENT_PLAN["pref"]:
        try:
            props = {"proposals": (POLICY.get(t) or {}).get("proposals") or []}
            an = _CTX["analyses"]
            pcs = _CTX["pcs"].get((a, b)) or {"candidates": [c]}
            ev_extra["event_plan"] = dje.event_plan(
                an[a], an[b], c, decision=dec, exit_fn=_exit_fn(a), entry_fn=_entry_fn(b), pair_candidates=pcs["candidates"],
                policy=_policy_rows(props, dec), policy_rule=dec.get("rule"), order_params=_CTX["order_over"],
                executed=["dj_set.choose_cues", "dj_render/dj_creative (render)", "dj_qa.evaluate_render", "dj_critic.score_transition (veto)"],
                versions={"dj_phrase": P.SCHEMA_VERSION, "dj_policy": pol.POLICY_VERSION, "dj_qa": dj_qa.DJ_QA_VERSION,
                          "dj_critic": (rec.get("critic") or {}).get("version")}, profile="books")
            ev_extra["event_plan"]["stage"] = "accettata (dopo render, QA e veto)"
        except Exception as exc:  # the event plan is only recorded
            ev_extra["event_plan_error"] = f"{type(exc).__name__}: {exc}"
    _B_IN_ACC[t] = float(plan["b_in_s"])
    if TENSION["mode"] != "off":
        ev_extra["tension"] = tension_record(t, a, b, res, dec, c)
        ev_extra["reliability"] = {k: (c.get("reliability") or {}).get(k) for k in ("value", "factors", "why", "novelty_cost")}
    if dec.get("creative"):
        rules.append({"what": "tecnica della policy (profilo libri)", "rule": dec["rule"], "source": dec.get("rule_source"),
                      "prior_source": dec["trace"].get("prior_source")})
    if dec["type"] == "cut" and plan.get("phrase_switch"):
        rules.append({"what": "uscita a fine verso e pickup di B", "rule": "dj_phrase.phrase_switch",
                      "source": "[M] MCFlow (Condit-Schultz 2017) per p(fine verso); [A] Krims 2000 pp. 132-133, 192-194 (chiusura sul 4° battito, "
                                "diagnostica); [P] B&B cap. 11 'How to cut' (il taglio nel silenzio tra i battiti); "
                                "conoscenza_mix/_meta/QUANDO_TAGLIARE_2026-10-21.md"})
    cands_same = [x for x in ATTEMPTS.get(t, []) if x[0] == rec["candidate"]]
    options = []
    for i, d, why in cands_same:
        options.append({"type_key": d["type_key"], "label": d["label"], "rule": d["rule"], "creative": bool(d.get("creative")),
                        "render": d.get("render"), "bars": d.get("bars"), "why": why})
    return {
        "technique": dec["type_key"], "duration_bars": dec.get("bars"),
        "books_rules": rules, "policy_books_v3": POLICY.get(t), "phrase": ps,
        "camelot": {"a": c["context"].get("camelot_a"), "b": c["context"].get("camelot_b")},
        "critic": rec.get("critic"), "qa_detail": {"verdict": qa["verdict"], "reasons": rec["reasons"]},
        "qa_rhythm_on_raw_a": rec.get("qa_rhythm_on_raw_a"),
        "vetoes": [{"candidate": x["candidate"], "type": x["type"], "a_out_bar": x.get("a_out_bar"), "b_in_bar": x.get("b_in_bar"),
                    "verdict": x.get("verdict"), "veto": x.get("veto"), "qa_reasons": x.get("reasons")}
                   for x in rec_list_for(t) if x.get("veto") or x.get("verdict") == "fail"],
        "kept_despite_veto": bool(rec.get("veto")),
        "books_options_same_cue": options,
        "clip_wav": base.rel(Path(w["wav"])), "clip_sha256": w["sha256"],
        **ev_extra,
    }


_TRIED: dict[int, list] = {}
_CTX: dict = {}  # analisi (con forme), candidati e parametri del set: per gli event plan delle transizioni accettate


def print_event_table(doc: dict) -> None:
    for x in doc["event_plans"]["transitions"]:
        ep = x["event_plan"]
        ae, tg, dl = ep["a_event"], ep["b_target"], ep["delay"]
        print(f"T{x['index']:02d} {x['a']} -> {x['b']}: A {ae['kind']} {ae['fate']} (verso {ae['line_end']['status']}), "
              f"B {ep['b_entry']['lead_in']['kind']} -> {tg['kind']}@{tg['bar']} ritardo {dl.get('bars')} batt. "
              f"(dallo switch {dl.get('from_switch_bars')}); {ep['choice']['type_key']}; preferito {ep['event_option']['preferred']}",
              flush=True)
    cf = doc.get("event_pref_counterfactual") or {}
    print(f"controfattuale ({cf.get('label')}): {cf.get('n_changed')} transizioni cambierebbero", flush=True)


def rec_list_for(t: int) -> list:
    return _TRIED.get(t, [])


def _track_tried(veto_fn):
    def wrapped(res, qa, dec, A, B, rec, t):
        out = veto_fn(res, qa, dec, A, B, rec, t)
        _TRIED.setdefault(t, []).append(rec)
        return out
    return wrapped


def finalize_v3(audio: np.ndarray) -> tuple[np.ndarray, dict]:
    """-14 LUFS integrati (BS.1770, pyloudnorm) e true peak <= -1 dBTP (limiter a blocchi di run_dj_set_v1)."""
    info: dict[str, Any] = {"target_lufs": -14.0, "ceiling_dbtp": -1.0, "steps": []}
    for _ in range(3):
        L = dj.integrated_lufs(audio.astype(np.float64), SR)
        g = -14.0 - L
        info["steps"].append({"lufs_before": L, "gain_db": g})
        if abs(g) <= 0.05:
            break
        audio *= np.float32(10 ** (g / 20.0))
        audio, lim = base.limit_long(audio)
        audio = audio.astype(np.float32)
        info["steps"][-1]["limiter"] = lim
    info["integrated_lufs"] = dj.integrated_lufs(audio.astype(np.float64), SR)
    return audio, info


# ------------------------------------------------------------- event plans (P0.5)
EVENT_PLAN = {"on": False, "pref": False}
#: Affidabilità dalle schede brano (21/10/2026, conoscenza_mix/_meta/SCHEDE_BRANO_AFFIDABILITA_2026-10-21.md): "off" (default,
#: comportamento invariato) | "report" (misura senza cambiare scelte) | "prefer" (evita brani e cue rischiosi); "novelty":
#: set.json di set precedenti (penalità [H] per coppie e brani già suonati, musicmix.reliability.NOVELTY_ON).
RELIABILITY = {"mode": "off", "cards": ROOT / "track_cards", "novelty": [], "params": {}}


def reliability_hook():
    if RELIABILITY["mode"] == "off":
        return None
    from musicmix import reliability as REL

    hist = [dict(REL.set_summary(load(Path(p))), name=str(p)) for p in RELIABILITY["novelty"]]
    params = dict(REL.NOVELTY_ON if hist else {}, **RELIABILITY["params"])
    return REL.ReliabilityHook(REL.load_cards(RELIABILITY["cards"]), mode=RELIABILITY["mode"], params=params or None,
                               history=hist, cards_dir=base.rel(Path(RELIABILITY["cards"])))
#: Carica di A / risoluzione di B / continuità / scontro tonale / affinità di testo (22/10/2026, musicmix.dj_tension,
#: conoscenza_mix/_meta/TENSIONE_RISOLUZIONE_2026-10-22.md): "off" (default) | "report" | "prefer" (costo [H] pre-registrato tra i
#: candidati ammissibili, tecniche a base continua prima sullo stesso cue, variante senza pausa dei build).
TENSION = {"mode": "off"}
_B_IN_ACC: dict[int, float] = {}
_TSIG: dict = {}


def tension_signals(slug: str):
    from musicmix import dj_tension as DT

    if slug not in _TSIG:
        try:
            _TSIG[slug] = DT.load_run_signals(slug, OUT)
        except Exception as exc:  # il punteggio si astiene sul brano
            print(f"dj_tension: {slug} senza segnali ({type(exc).__name__}: {exc})", flush=True)
            _TSIG[slug] = None
    return _TSIG[slug]


def tension_hook():
    if TENSION["mode"] == "off":
        return None
    from musicmix import dj_tension as DT

    return DT.TensionHook(tension_signals, mode=TENSION["mode"])


def tension_record(t: int, a: str, b: str, res, dec: dict, c: dict) -> dict:
    """Punteggi dj_tension della transizione accettata (tecnica reale, A suonata dal suo ingresso nel set, base misurata)."""
    from musicmix import dj_tension as DT

    sa, sb = tension_signals(a), tension_signals(b)
    if sa is None or sb is None:
        return {"error": "segnali assenti"}
    pl = c["plan"]
    played = float(pl["a_out_s"]) - _B_IN_ACC[t - 1] if t > 0 and (t - 1) in _B_IN_ACC else None
    aud = ((res.metadata.get("b_entry") or {}).get("audible_source_sample"))
    b_aud = float(aud) / SR if aud is not None else DT.b_audible_s(pl, SR)
    r = DT.score_transition(sa, sb, a_out_s=float(pl["a_out_s"]), b_in_s=float(pl["b_in_s"]), decision=dec, b_audible=b_aud,
                            played_a_s=played)
    out = DT.compact(r)
    out.update(a_charge_why=r["a_charge"]["why"], b_resolution_why=r["b_resolution"]["why"],
               harmonic_clash_why=r["harmonic_clash"]["why"], b_audible_s=b_aud, a_played_s=played)
    try:
        g = ds.clip_geometry(res.metadata, len(res.audio))
        b_aud_out = (res.metadata.get("b_entry") or {}).get("audible_output_sample")
        out["continuity_measured"] = DT.continuity_measured(res.audio, SR, int(g["switch"]), float(sa.period_s),
                                                            b_audible_sample=None if b_aud_out is None else int(b_aud_out))
    except Exception as exc:
        out["continuity_measured"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


EVENT_PREF_W = 0.05  # [H] sperimentale: "a parità di costo tecnico" = differenza di costo < 0,05 (scala dei costi di dj_set)


def _exit_fn(slug: str):
    return (lambda t: exit_at(slug, t)) if slug in MODELS else None


def _entry_fn(slug: str):
    return (lambda t: entry_at(slug, t)) if slug in MODELS else None


def event_fn_v3(A: dict, B: dict, c: dict) -> dict:
    """Riassunto eventi con dj_phrase (verso di A, pickup/prima parola di B) per l'opzione sperimentale."""
    pl = c["plan"]
    a, b = A["slug"], B["slug"]
    return dje.arrival_summary(A, B, c, exit_info=exit_at(a, pl["a_out_s"]) if a in MODELS else None,
                               entry_info=entry_at(b, pl["b_in_s"]) if b in MODELS else None,
                               params={"arrival_max_bars": ds.values(ds.CUE_PARAMS)["event_arrival_max_bars"]})


def _policy_rows(props: dict, chosen: dict | None) -> list:
    rows = []
    for p in props["proposals"]:
        if p["rule"] in CONTEXT_RULES and p["technique"] in RENDERABLE:
            why = "regola di contesto giuntabile: proposta successiva (ordine per punteggio della policy)"
        elif p["technique"] in NOT_SPLICEABLE:
            why = "B stirato: non giuntabile nel set (NOT_SPLICEABLE)"
        else:
            why = "regola generica (C5-C7): nel set vale la decisione dell'albero libri"
        rows.append({"technique": p["technique"], "rule": p["rule"], "score": round(float(p["score"]), 4),
                     "duration_bars": p.get("duration_bars"), "why_not": why})
    return rows


def planned_decision(t: int, a: str, b: str, cand: dict, pc: dict, analyses: dict, recent: list) -> tuple[dict, dict, str]:
    """Tecnica che il set prova per prima sul cue del Viterbi (come attempts_v3): policy dei libri se è una regola di
    contesto giuntabile, altrimenti l'albero. Restituisce (decisione, proposte, motivo)."""
    props = policy_for(analyses[a], analyses[b], cand["plan"], pc["compatibility"], recent)
    for p in props["proposals"]:
        if props["proposals"][0]["rule"] in CONTEXT_RULES and props["proposals"][0]["technique"] in RENDERABLE \
                and p["rule"] in CONTEXT_RULES and p["technique"] in RENDERABLE:
            return creative_decision(p), props, f"policy dei libri ({p['rule']})"
    return cand["decision"], props, "albero libri (prima proposta della policy generica o non giuntabile)"


def plan_one(t: int, a: str, b: str, cand: dict, dec: dict, props: dict, pc: dict, analyses: dict, order_over: dict) -> dict:
    sa, sb = PLAN / "tracks" / f"{a}.json", PLAN / "tracks" / f"{b}.json"
    return dje.event_plan(
        analyses[a], analyses[b], cand, decision=dec, exit_fn=_exit_fn(a), entry_fn=_entry_fn(b),
        pair_candidates=pc["candidates"], policy=_policy_rows(props, dec), policy_rule=dec.get("rule"),
        order_params=order_over, executed=["dj_set.choose_cues", "run_dj_set_v3_bridge.filter_pair_v3 (guardia dj_phrase dei cut)"],
        versions={"dj_phrase": P.SCHEMA_VERSION, "dj_policy": pol.POLICY_VERSION, "dj_critic": None, "dj_qa": None},
        cue_windows=pc.get("cue_windows", "default"), profile="books",
        sources={"a_analysis_sha256": sha(sa), "b_analysis_sha256": sha(sb),
                 "a_phrase_asr_reliable": reliable(a) if a in MODELS else None, "b_phrase_asr_reliable": reliable(b) if b in MODELS else None})


def _cues_of(chosen: list) -> list:
    return [(int(c["plan"]["a_out_bar"]), int(c["plan"]["b_in_bar"]), c["decision"]["type_key"]) for c in chosen]


def after_cues_v3(ctx: dict) -> dict:
    order, chosen, pcs, analyses = ctx["order"], ctx["chosen"], ctx["pair_cands"], ctx["analyses"]
    _CTX.update(analyses=analyses, pcs=pcs, order_over=ctx["order_over"])
    recent: list = []
    plans = []
    for t, c in enumerate(chosen):
        a, b = order[t], order[t + 1]
        pc = pcs[(a, b)]
        dec, props, why = planned_decision(t, a, b, c, pc, analyses, recent)
        ep = plan_one(t, a, b, c, dec, props, pc, analyses, ctx["order_over"])
        ep["planned_by"] = why
        plans.append({"index": t + 1, "a": a, "b": b, "event_plan": ep})
        recent.append(dec["type"])
    # what the experimental option would choose (or the default, when the option is on)
    w_now = float(ctx["cue_over"].get("w_event_arrival") or 0.0)
    alt_over = dict(ctx["cue_over"], w_event_arrival=0.0 if w_now > 0 else EVENT_PREF_W)
    try:
        alt = ds.choose_cues(order, analyses, pcs, alt_over, event_fn=event_fn_v3)
        changed = []
        for t, (x, y) in enumerate(zip(_cues_of(chosen), _cues_of(alt["chosen"]))):
            if x != y:
                s0 = event_fn_v3(analyses[order[t]], analyses[order[t + 1]], chosen[t])
                s1 = event_fn_v3(analyses[order[t]], analyses[order[t + 1]], alt["chosen"][t])
                changed.append({"index": t + 1, "a": order[t], "b": order[t + 1], "run": {"cue": x, "event": s0, "cost": chosen[t]["cost"]},
                                "counterfactual": {"cue": y, "event": s1, "cost": alt["chosen"][t]["cost"]}})
        cf = {"w_event_arrival": alt_over["w_event_arrival"], "event_arrival_max_bars": alt_over.get("event_arrival_max_bars", 2.0),
              "label": "[H] sperimentale" if alt_over["w_event_arrival"] > 0 else "default (opzione spenta)",
              "cost": alt["cost"], "changed": changed, "n_changed": len(changed),
              "chosen_preferred": alt.get("event_pref", {}).get("chosen_preferred")}
    except ValueError as exc:
        cf = {"error": str(exc)}
    run_pref = [bool(event_fn_v3(analyses[order[t]], analyses[order[t + 1]], c)["preferred"]) for t, c in enumerate(chosen)]
    return {"event_plans": {"schema": dje.EVENT_PLAN_SCHEMA, "version": dje.DJ_EVENTS_VERSION, "stage": "planned (prima del render)",
                            "experimental_option": {"w_event_arrival": w_now, "on": w_now > 0},
                            "chosen_preferred_by_option": run_pref, "transitions": plans},
            "event_pref_counterfactual": cf}


def frozen_comparison(doc: dict) -> dict:
    """Confronto con il set congelato (runs/dj_set_v3_bridge/set.json): stesse coppie, cue e tecniche?"""
    ref = load(OUT / "set.json")
    rt = {(t["a"], t["b"]): t for t in ref["transitions"]}
    rows = []
    for x in doc.get("event_plans", {}).get("transitions", []):
        ep = x["event_plan"]
        r = rt.get((x["a"], x["b"]))
        ch = ep["choice"]
        rows.append({"index": x["index"], "a": x["a"], "b": x["b"], "in_frozen_set": r is not None,
                     "cue": [ep["a_event"]["exit_bar"], ep["b_entry"]["downbeat_bar"]],
                     "frozen_cue": [r["plan"]["a_out_bar"], r["plan"]["b_in_bar"]] if r else None,
                     "technique": ch["type_key"], "frozen_technique": r["type"] if r else None,
                     "same_cue": bool(r and [r["plan"]["a_out_bar"], r["plan"]["b_in_bar"]] == [ep["a_event"]["exit_bar"], ep["b_entry"]["downbeat_bar"]]),
                     "same_technique": bool(r and r["type"] == ch["type_key"])})
    return {"reference": base.rel(OUT / "set.json"), "reference_sha256": sha(OUT / "set.json"),
            "same_order": [x["a"] for x in doc.get("event_plans", {}).get("transitions", [])] == [t["a"] for t in ref["transitions"]],
            "n_same_cue": sum(r["same_cue"] for r in rows), "n_same_technique": sum(r["same_technique"] for r in rows), "rows": rows}


def run_set(overwrite: bool, out: Path = OUT, plan_only: bool = False, form_params: dict | None = None) -> None:
    out = Path(out)
    _CTX["out"] = out
    if out.resolve() != OUT.resolve():  # separate form cache (under .cache/<run>/, the run being the first folder under runs/)
        try:
            run_root = out.resolve().relative_to(ROOT / "runs").parts[0]
        except ValueError:
            run_root = out.name
        v2.FORM_CACHE = ROOT / ".cache" / run_root / ("form" if not form_params else "form_" + hashlib.sha1(
            json.dumps(form_params, sort_keys=True).encode()).hexdigest()[:8])
    if out.resolve() == OUT.resolve() and plan_only:
        raise SystemExit("--plan-only riscriverebbe la run congelata: usare --out")
    cue_over = {"w_event_arrival": EVENT_PREF_W} if EVENT_PLAN["pref"] else {}
    hooks = v2.SetHooks(
        run=out.name, script=SCRIPT, title="set DJ v3 'ponte' (profilo libri, dj_phrase, veto del critico)",
        pool=sources_v3, info=info_v3, filter_pair=filter_pair_v3, attempts=attempts_v3, render=render_v3, run_qa=qa_v3,
        veto=_track_tried(veto_v3), lead_a_s=lead_a_v3, overlap_b_s=overlap_b_v3, on_accept=on_accept_v3,
        finalize_audio=finalize_v3, mp3_bitrate="192k", order_overrides=dict(ORDER_OVERRIDES), compare_v1b=False,
        notes=("integrità tecnica (QA, critico) non è qualità musicale: il set non è stato ascoltato",
               "profilo 'books' (dj_set, dj_policy.PROFILE_BOOKS): regole di pratica/analisi dai libri, pesi e soglie [H]",
               "guardia del cut: dj_phrase (fine verso stimata su MCFlow) quando l'ASR è affidabile, altrimenti guardia v2",
               "veto del critico v3 (fase > 40 ms, loudness > 3 dB, distanza tonale > 1 con sovrapposizione) e QA fail: il critico non ordina le alternative",
               "pool: catalogo del ponte congelato (runs/dj_set_v3_bridge/catalog_frozen.json), solo brani 'ready'"),
        form_params=form_params, cue_overrides=cue_over, reliability=reliability_hook(), tension=tension_hook(),
        event_fn=event_fn_v3 if (EVENT_PLAN["on"] or EVENT_PLAN["pref"]) else None,
        after_cues=after_cues_v3 if (EVENT_PLAN["on"] or EVENT_PLAN["pref"]) else None)
    v2.build_set(out, overwrite=overwrite, no_render=plan_only, hooks=hooks)
    doc = load(out / "set.json")
    if plan_only:
        if doc.get("event_plans"):
            doc["frozen_set_comparison"] = frozen_comparison(doc)
            dump(out / "set.json", doc)
            print_event_table(doc)
        return
    if out.resolve() != OUT.resolve():
        return
    DELIVER.mkdir(exist_ok=True)
    shutil.copy2(OUT / "set.mp3", DELIVER / "set_dj_v3_bridge.mp3")
    shutil.copy2(OUT / "tracklist.txt", DELIVER / "set_dj_v3_bridge_tracklist.txt")
    print(json.dumps({"durata_s": doc["output"]["duration_s"], "lufs": (doc["output"].get("finalize") or {}).get("integrated_lufs"),
                      "true_peak": doc["output"]["true_peak_dbtp"], "qa": doc["qa_counts_final"]}), flush=True)


# ------------------------------------------------------------------ gallery
def default_decision(a_an: dict, b_an: dict, plan: dict, pair: dict, info_a: dict) -> dict:
    """Decisione dell'albero di dj_set senza profilo (dj_set_v1b) con la guardia del cut v2."""
    ctx = ds.transition_context(a_an, b_an, plan, pair)
    pc = {"candidates": [{"plan": plan, "context": ctx, "decision": ds.decide_transition(ctx, None), "notes": [], "admissible": True,
                          "cost": 0.0}], "compatibility": pair}
    inf = dict(info_a, cut_clear=info_a.get("cut_clear_v2", info_a["cut_clear"]))
    pc = base.apply_filters(pc, inf)
    return pc["candidates"][0]["decision"]


def signature(dec: dict, plan: dict) -> str:
    r = dict(dec.get("render") or {})
    ph = plan.get("phrase_switch") if dec["type"] == "cut" else None
    if ph and (abs(float(ph.get("switch_rel_s") or 0.0)) > 1e-6 or ph.get("tail_end_a_s")):
        r["phrase"] = True
    return dec["type_key"] + "|" + json.dumps(base.jsonable(r), sort_keys=True, default=str)


def clip_markers(res) -> dict:
    g = ds.clip_geometry(res.metadata, len(res.audio))
    return {"t0": g["transition_start"] / SR, "sw": g["switch"] / SR, "t1": g["transition_end"] / SR, "n": len(res.audio) / SR}


def trim_to_rule(audio: np.ndarray, kind: str, t0: float, sw: float, t1: float) -> tuple[np.ndarray, dict]:
    """AGENTS.md: cut 10 s + 10 s attorno al cambio; blend/effetti 10 s prima dell'inizio + transizione + 10 s dopo."""
    n = len(audio)
    is_cut = kind in ("cut", "drop_in")
    s0 = (sw if is_cut else t0) - 10.0
    s1 = (sw if is_cut else t1) + 10.0
    a, b = int(round(s0 * SR)), int(round(s1 * SR))
    ok = a >= 0 and b <= n
    a, b = max(0, a), min(n, b)
    return audio[a:b], {"t0": t0 - a / SR, "sw": sw - a / SR, "t1": t1 - a / SR, "rule_ok": bool(ok), "src_window_s": [a / SR, b / SR]}


def write_clip(audio: np.ndarray, path: Path) -> str:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SR, subtype="PCM_24")
    return sha(path)


def gallery() -> dict:
    import soundfile as sf

    doc = load(OUT / "set.json")
    gdir = OUT / "gallery"
    (gdir / "clips").mkdir(parents=True, exist_ok=True)
    cat = load(FROZEN)
    src = {"plan_dir": PLAN, "catalog": cat, "catalog_path": FROZEN}
    tf = TF.default_filter()  # brani esclusi (lista o lingua): niente clip nella galleria
    trs = [t for t in doc["transitions"] if tf.check(t["a"])["admitted"] and tf.check(t["b"])["admitted"]][:MAX_GALLERY]
    load_phrase({x for t in trs for x in (t["a"], t["b"])})
    tracker = dj_qa.BeatThisTracker()
    rng = random.Random(GALLERY_SEED)
    groups, key, notes = [], {}, []
    loaded: dict[str, Any] = {}

    def track(slug, keep):
        for k in list(loaded):
            if k not in keep:
                del loaded[k]
        if slug not in loaded:
            loaded[slug] = v1.Track(slug, PLAN, cat)
        return loaded[slug]

    for tr in trs:
        k = tr["index"]
        a, b = tr["a"], tr["b"]
        A, B = track(a, (a, b)), track(b, (a, b))
        a_an, b_an = A.analysis, B.analysis
        pp = djs.plan_pair(a_an, b_an, top_k=10 ** 9)
        plan = next(p for p in pp["plans"] if p["a_out_bar"] == tr["plan"]["a_out_bar"] and p["b_in_bar"] == tr["plan"]["b_in_bar"])
        info_a = info_v3(a, src)
        # the set's own phrase switch for this cue (same as filter_pair_v3)
        ps = phrase_summary(a, b, plan["a_out_s"], plan["b_in_s"], float(a_an["tempo"]["period_s"]))
        plan_books = dict(plan)
        if ps["switch_ok"] and (reliable(a) or reliable(b)):
            plan_books["phrase_switch"] = {"switch_rel_s": ps["switch_rel_s"], "tail_end_a_s": ps["tail_end_a_s"],
                                           "pickup_s": ps["entry_pickup_s"], "pickup_words": ps["entry"]["pickup_words"],
                                           "last_word": ps["exit"]["last_word"]}
        prov = {"a": A.provenance, "b": B.provenance, "script": SCRIPT, "run": RUN, "gallery_transition": k}

        def attempt(dec, pl, role):
            res, decks = (v2.render(dec, A, B, pl, prov) if role == "default" else render_v3(dec, A, B, pl, prov))
            qa = qa_v3(res, A, B, decks, tracker)
            crt = critic(res, dec["type"], f"g{k:02d}_{role}_{dec['type_key']}", decks=res.decks, qa=qa, declared_gap=declared_gap(dec))
            if qa["verdict"] != "fail" and not crt["veto"]:  # versione che entrerà in galleria: QA anche con il ritmo di A prima dell'EQ
                q2 = qa_v3(res, A, B, decks, tracker, cfg=QA_RAW_A)
                qa = dict(qa, rhythm_on_raw_a={"verdict": q2["verdict"], "reasons": [f"{r['check']}: {r['message']}" for r in q2["reasons"]]})
            return res, qa, crt

        # (A) default profile: tree without profile + v2 cut guard, its own fallbacks on the same cue if vetoed
        d0 = default_decision(a_an, b_an, plan, pp["compatibility"], info_a)
        tried_a, ver_a = [], None
        for dec in [d0] + list(d0.get("fallbacks") or []):
            dec = {**dec, "type_key": dec.get("type_key", dec["type"])}
            try:
                res, qa, crt = attempt(dec, plan, "default")
            except (dj.DJRenderError, ValueError) as exc:
                tried_a.append({"type": dec["type_key"], "error": str(exc)})
                continue
            row = {"type": dec["type_key"], "label": dec.get("label"), "rule": dec.get("rule"), "qa": qa["verdict"],
                   "qa_reasons": [f"{r['check']}: {r['message']}" for r in qa["reasons"]], "critic": crt}
            tried_a.append(row)
            if qa["verdict"] != "fail" and not crt["veto"]:
                ver_a = (dec, res, qa, crt, plan)
                break
        # (B) books: the set's clip, or the best books alternative on the same cue if it equals A
        opts = tr.get("books_options_same_cue") or []
        acc_opt = next((o for o in opts if o["type_key"] == tr["technique"]), None)
        acc_sig = signature(opt_decision(acc_opt), plan_books) if acc_opt is not None else None
        sig_a = signature(ver_a[0], plan) if ver_a else None
        ver_b, tried_b, b_is_set_clip = None, [], False
        if ver_a is not None and acc_sig is not None and acc_sig != sig_a:
            audio, sr = sf.read(str(ROOT / tr["clip_wav"]), dtype="float32", always_2d=True)
            res_b = dj.DJRenderResult(audio, sr, load((ROOT / tr["clip_wav"]).with_suffix(".json")), None)
            qa_b = {"verdict": tr["qa"]["verdict"], "reasons": [{"check": r.split(":")[0], "message": r} for r in tr["qa"]["reasons"]],
                    "rhythm_on_raw_a": tr.get("qa_rhythm_on_raw_a")}
            ver_b = (opt_decision(acc_opt), res_b, qa_b, tr.get("critic") or {}, plan_books)
            b_is_set_clip = True
        elif ver_a is not None:
            for o in opts:
                dec = opt_decision(o)
                if signature(dec, plan_books) == sig_a:
                    continue
                try:
                    res, qa, crt = attempt(dec, plan_books, "books")
                except (dj.DJRenderError, ValueError, KeyError) as exc:
                    tried_b.append({"type": o["type_key"], "error": str(exc)})
                    continue
                tried_b.append({"type": o["type_key"], "qa": qa["verdict"], "critic_veto": crt["veto"]})
                if qa["verdict"] != "fail" and not crt["veto"]:
                    ver_b = (dec, res, qa, crt, plan_books)
                    break
        gid = f"g{k:02d}"
        entry = {"id": gid, "title": f"{k}. {A.title} → {B.title}",
                 "pair": {"a": {"slug": a, "title": A.title, "artist": a_an.get("artist")}, "b": {"slug": b, "title": B.title, "artist": b_an.get("artist")}},
                 "bpm": {"a": a_an["tempo"]["bpm"], "b": b_an["tempo"]["bpm"]},
                 "key": {"a": a_an["key"]["camelot"], "b": b_an["key"]["camelot"], "a_name": a_an["key"]["key"], "b_name": b_an["key"]["key"]},
                 "A_out_s": plan["a_out_s"], "B_in_s": plan["b_in_s"], "A_out_bar": plan["a_out_bar"], "B_in_bar": plan["b_in_bar"],
                 "sections": {"a": plan.get("a_section"), "b": plan.get("b_section")},
                 "plan": {"type": plan.get("type"), "score": plan.get("score")},
                 "reasons": (ps.get("exit_reasons") or [])[:2], "items": []}
        rec_key = {"group": gid, "transition": k, "default_attempts": tried_a, "books_attempts": tried_b, "b_is_set_clip": b_is_set_clip}
        versions = []
        if ver_a is not None:
            versions.append(("default", ver_a))
        if ver_b is not None:
            versions.append(("books", ver_b))
        if len(versions) < 2:
            notes.append({"transition": k, "note": "una sola versione: " + ("nessuna alternativa dei libri diversa dal default ammessa da QA e veto"
                                                                           if ver_a is not None else "default bocciato da QA/veto su tutte le varianti")})
        rng.shuffle(versions)
        for role, (dec, res, qa, crt, pl) in versions:
            mk = clip_markers(res)
            kind = dec["type"]
            clip, tm = trim_to_rule(res.audio, kind, mk["t0"], mk["sw"], mk["t1"])
            cid = f"{gid}_{hashlib.sha1(f'{GALLERY_SEED}{gid}{role}'.encode()).hexdigest()[:5]}"
            wav = gdir / "clips" / f"{cid}.wav"
            h = write_clip(clip, wav)
            item = {"id": cid, "kind": "cut" if kind == "cut" else kind, "role": "versione",
                    "label": ("profilo default: " if role == "default" else "profilo libri: ") + str(dec.get("label") or dec["type_key"]),
                    "params": {"technique": dec["type_key"], "render": dec.get("render"), "phrase_switch": pl.get("phrase_switch") if kind == "cut" else None},
                    "why": dec.get("rule"), "audio_path": base.rel(wav), "duration_s": len(clip) / SR, "sample_rate": SR,
                    "transition_start_s": tm["t0"], "switch_s": tm["sw"], "transition_end_s": max(tm["t1"], tm["sw"]),
                    "qa": {"verdict": qa["verdict"], "warn_reasons": [r["message"] for r in qa["reasons"]],
                           "rhythm_on_raw_a": qa.get("rhythm_on_raw_a")},
                    "critic": {k2: crt.get(k2) for k2 in ("score", "flags", "veto", "diagnostic_flags", "declared_phase", "craft", "v4")},
                    "sha256": h}
            entry["items"].append(item)
            rec_key[cid] = {"profile": role, "technique": dec["type_key"], "rule": dec.get("rule"), "clip_rule_ok": tm["rule_ok"],
                            "qa": qa["verdict"], "critic_flags": [f["flag"] for f in crt.get("flags", [])]}
        key[gid] = rec_key
        groups.append(entry)
        print(f"{gid}: {[(i['id'], key[gid][i['id']]['profile'], key[gid][i['id']]['technique']) for i in entry['items']]}", flush=True)
        del res
    loaded.clear()
    groups += check_groups(gdir, key)
    manifest = {"schema_version": "musicmix.dj-review.1", "title": "Set DJ v3 ponte: A/B cieco profilo default vs libri",
                "run": RUN, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": {"set": base.rel(OUT / "set.json"), "set_sha256": sha(OUT / "set.json")},
                "conventions": {"clip_rule": "cut: 10 s + 10 s attorno al cambio; blend/effetti: 10 s prima dell'inizio della transizione + "
                                             "transizione + 10 s dopo (AGENTS.md, 20/10/2026)",
                                "blind": "ordine delle versioni casuale (seed in blind_key.json); profilo e tecnica rivelati dopo il voto",
                                "no_continue_A": "nessuna clip 'continua A' da votare"},
                "counts": {"groups": len(groups), "clips": sum(len(g["items"]) for g in groups)},
                "groups": groups, "rejected": notes}
    dump(gdir / "review_manifest.json", manifest)
    dump(gdir / "blind_key.json", {"seed": GALLERY_SEED, "note": "NON pubblicare: chiave delle versioni (profilo, tecnica)", "groups": key})
    return manifest


def _tree_type(type_key: str) -> str:
    return "blend" if type_key.startswith("blend") else type_key


def opt_decision(o: dict) -> dict:
    """Decisione da una voce di ``books_options_same_cue`` del set."""
    return {"type": o["type_key"] if o["creative"] else _tree_type(o["type_key"]), "type_key": o["type_key"], "label": o["label"],
            "rule": o["rule"], "render": o["render"], "bars": o["bars"], "creative": o["creative"]}


CHECK_GROUPS = {  # runs/books_integration_check: transizione -> clip (le due 'break' con verse_cut fail sono escluse)
    "t01": ["t01_blend_default", "t01_blend_books_eq", "t01_halving_build"],
    "t04": ["t04_blend_default", "t04_blend_books_eq"],
    "t05": ["t05_halving_build", "t05_break_stop"],
}


def check_groups(gdir: Path, key: dict) -> list:
    import soundfile as sf

    rep = load(CHECK_DIR / "report.json")
    rows = {r["name"]: r for r in rep["rows"]}
    rng = random.Random(GALLERY_SEED + 1)
    out = []
    tf = TF.default_filter()
    for tkey, names in CHECK_GROUPS.items():
        if any(not tf.check(rows[n][r])["admitted"] for n in names if n in rows for r in ("a", "b")):
            continue  # clip di verifica con un brano escluso (lista o lingua)
        gid = f"verifica_{tkey}"
        items, rec = [], {"group": gid, "source": base.rel(CHECK_DIR)}
        names = list(names)
        rng.shuffle(names)
        title = None
        for nm in names:
            wav = CHECK_DIR / "clips" / f"{nm}.wav"
            meta = load(wav.with_suffix(".json"))
            meta = meta.get("metadata", meta)
            audio, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            g = ds.clip_geometry(meta, len(audio))
            kind = meta.get("kind") or nm.split("_", 1)[1]
            clip, tm = trim_to_rule(audio, kind, g["transition_start"] / sr, g["switch"] / sr, g["transition_end"] / sr)
            cid = f"{gid}_{hashlib.sha1(f'{GALLERY_SEED}{nm}'.encode()).hexdigest()[:5]}"
            out_wav = gdir / "clips" / f"{cid}.wav"
            h = write_clip(clip, out_wav)
            r = rows.get(nm, {})
            title = f"Verifica libri ({tkey}, da dj_set_v2_books): {r.get('a')} → {r.get('b')}"
            items.append({"id": cid, "kind": kind, "role": "versione", "label": f"verifica libri: {nm.split('_', 1)[1]}",
                          "params": {"clip": nm}, "why": "runs/books_integration_check (clip già pronte, non ascoltate)",
                          "audio_path": base.rel(out_wav), "duration_s": len(clip) / sr, "sample_rate": sr,
                          "transition_start_s": tm["t0"], "switch_s": tm["sw"], "transition_end_s": max(tm["t1"], tm["sw"]),
                          "qa": {"verdict": (r.get("qa") or {}).get("verdict"), "warn_reasons": (r.get("qa") or {}).get("reasons", [])},
                          "sha256": h})
            rec[cid] = {"clip": nm, "clip_rule_ok": tm["rule_ok"], "source_wav_sha256": sha(wav)}
        key[gid] = rec
        out.append({"id": gid, "title": title, "pair": {"a": {"slug": rows[names[0]]["a"]}, "b": {"slug": rows[names[0]]["b"]}},
                    "reasons": ["sezione separata: clip di verifica dei libri (halving build, break stop, EQ dei libri) già pronte"],
                    "items": items})
    return out


def page(out: Path) -> dict:
    import build_listening_artifact as bla

    man = OUT / "gallery" / "review_manifest.json"
    listing = bla.build(man, out, run=RUN, blind=True, overwrite=True)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"pagina {out / 'index.html'}: {len(listing['included'])} clip, {total / 1e6:.1f} MB", flush=True)
    if total >= 60e6:
        raise SystemExit(f"pagina troppo grande: {total / 1e6:.1f} MB")
    return listing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stages", nargs="+", choices=("prepare", "set", "gallery", "page"))
    ap.add_argument("--overwrite", action="store_true", help="rifà il set (mantiene catalogo congelato, analisi e modelli)")
    ap.add_argument("--page-out", default=str(PAGE_OUT))
    ap.add_argument("--out", default=str(OUT), help="run di uscita della fase set (default: la run congelata)")
    ap.add_argument("--plan-only", action="store_true", help="fase set senza render/QA/montaggio (solo ordine, cue e piani)")
    ap.add_argument("--event-plan", action="store_true", help="scrive i piani per eventi (musicmix.event_plan.1) nel set.json")
    ap.add_argument("--event-pref", action="store_true", help="[H] sperimentale: preferenza per eventi a parità di costo")
    ap.add_argument("--form-params", default=None, help="JSON di sostituzioni di dj_form.PARAMS (es. '{\"kick_peak_on\": null}')")
    ap.add_argument("--no-b-entry-fix", action="store_true", help="spegne guardia/lead-in/QA dell'ingresso di B (riproduce il set v3)")
    ap.add_argument("--reliability", choices=("off", "report", "prefer"), default="off",
                    help="affidabilità dalle schede brano: report = solo misura, prefer = evita brani/cue rischiosi")
    ap.add_argument("--cards", default=str(ROOT / "track_cards"), help="cartella delle schede brano (tools/build_track_cards.py)")
    ap.add_argument("--novelty", nargs="*", default=[], help="set.json di set precedenti: penalità [H] per coppie e brani ripetuti")
    ap.add_argument("--reliability-params", default=None, help="JSON di sostituzioni di musicmix.reliability.PARAMS (es. '{\"w_risk\": 1.5}')")
    ap.add_argument("--no-manual-exclude", action="store_true", help="ignora EXCLUDE (esclusioni a mano): decidono le schede brano")
    ap.add_argument("--tension", choices=("off", "report", "prefer"), default="off",
                    help="dj_tension: carica di A, risoluzione di B, base continua, scontro tonale, testo affine (costo [H] pre-registrato)")
    args = ap.parse_args(argv)
    B_ENTRY["on"] = not args.no_b_entry_fix
    RELIABILITY.update(mode=args.reliability, cards=Path(args.cards), novelty=list(args.novelty),
                       params=json.loads(args.reliability_params) if args.reliability_params else {})
    if args.no_manual_exclude:
        EXCLUDE.clear()
    EVENT_PLAN.update(on=bool(args.event_plan), pref=bool(args.event_pref))
    TENSION["mode"] = args.tension
    for st in args.stages:
        t0 = time.time()
        if st == "prepare":
            prepare()
        elif st == "set":
            run_set(args.overwrite, Path(args.out), plan_only=args.plan_only,
                    form_params=json.loads(args.form_params) if args.form_params else None)
        elif st == "gallery":
            gallery()
        elif st == "page":
            page(Path(args.page_out))
        print(f"[{st}] {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
