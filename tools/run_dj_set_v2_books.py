"""Set DJ v2 "libri": stesso pool e stessa pipeline di dj_set_v1b, profilo ``books`` di dj_set/dj_policy.

Differenze rispetto a ``tools/run_dj_set_v1.py`` (di cui riusa sorgenti, cache delle feature,
render, QA e montaggio):

* parametri del profilo ``books`` (``musicmix.dj_set.profile("books")``) per albero, ordine e cue:
  cut sull'uno nell'hip-hop, cut quando le tonalità cozzano, backspin per tempi lontani, blend
  hip-hop corti, durata a sezioni intere, arco a onde, costi di collocazione, budget degli effetti
  (fonti in ``conoscenza_mix/libri/``);
* categorie formali di Butler (``musicmix.dj_form``) calcolate per ogni brano dagli spettrogrammi per
  stem di All-In-One (Demucs htdemucs) se presenti e della durata giusta, altrimenti dal mix meno lo
  stem vocale; allegate alle analisi come ``form`` e usate dal profilo (uscita durante un build,
  breakdown, drop, tessiture piene);
* render del tipo ``backspin`` con ``musicmix.dj_creative.backspin`` (QA con ``dj_creative.qa_inputs``);
* per ogni transizione: profilo formale della sovrapposizione (per il giudice) e proposte di
  ``dj_policy.propose`` con il profilo libri (solo registrate, non renderizzate);
* pool: gli stessi 25 brani di ``runs/dj_set_v1b`` (confronto omogeneo) più i brani del ponte
  (``runs/dj_plan_bridge_v1``) con analisi DJ completa; confronto finale con v1b.

Uso: ``python tools/run_dj_set_v2_books.py --out runs/dj_set_v2_books``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import run_dj_set_v1 as base  # noqa: E402  (sorgenti, feature, render, QA, montaggio)

from musicmix import dj_creative as cr  # noqa: E402
from musicmix import dj_form as F  # noqa: E402
from musicmix import dj_policy as pol  # noqa: E402
from musicmix import dj_qa  # noqa: E402
from musicmix import dj_render as dj  # noqa: E402
from musicmix import dj_select as djs  # noqa: E402
from musicmix import dj_set as ds  # noqa: E402
from musicmix import track_filter as TF  # noqa: E402

v1 = base.v1
RUN = "dj_set_v2_books"
SCRIPT = "tools/run_dj_set_v2_books.py"
SR = base.SR
FORM_CACHE = ROOT / ".cache" / RUN / "form"
SPEC_DIRS = (ROOT / ".cache/dj_allin1/spec", ROOT / ".cache/dj_allin1_cc/spec", ROOT / ".cache/dj_allin1_bridge/spec")
V1B = ROOT / "runs/dj_set_v1b/set.json"
BRIDGE_PLAN = ROOT / "runs/dj_plan_bridge_v1"
PROFILE = ds.profile("books")
CR_CFG = cr.DJCreativeConfig(section_gain=base.SG) if base.SG is not None else cr.DJCreativeConfig()
KEEP_ON_OVERWRITE = ("catalog_frozen.json", "catalog_frozen_18.json", "dj_plan", "phrase", "prepare.json", "prepare.log", "set.log", "set_run1.json", "set_run1.log", "set_run2_keyerror.log", "compare_kick_legacy", "compare_event_pref", "plans", "defect_gym")  # ingressi congelati di una run (v3) da non cancellare


# ------------------------------------------------------------------- pool
def pool_sources(log: list) -> dict[str, dict]:
    """Pool di v1b (stessi slug) + brani del ponte con analisi DJ (``tracks/<slug>.json``)."""
    allsrc = base.discover(log)
    v1b_slugs = json.loads(V1B.read_text(encoding="utf-8"))["pool"]["edge_matrix"]["slugs"]
    keep = {s: allsrc[s] for s in v1b_slugs if s in allsrc}
    bridge_added, bridge_skipped = [], []
    tracks_dir = BRIDGE_PLAN / "tracks"
    if tracks_dir.is_dir():
        cat_path = ROOT / "runs/pool_bridge_v1/catalog_initial.json"
        cat = v1.load_json(cat_path) if cat_path.is_file() else None
        for f in sorted(tracks_dir.glob("*.json")):
            if f.stem in keep:
                continue
            if cat is None:
                bridge_skipped.append({"slug": f.stem, "why": "catalogo del ponte assente"})
                continue
            keep[f.stem] = {"plan_dir": BRIDGE_PLAN, "catalog": cat, "catalog_path": cat_path, "source": "pool_bridge_v1 (runs/dj_plan_bridge_v1)"}
            bridge_added.append(f.stem)
    prov_dir = BRIDGE_PLAN / "providers"
    if prov_dir.is_dir():
        for p in sorted(prov_dir.glob("*.json")):
            if p.stem in keep or p.stem in bridge_added:
                continue
            rec = json.loads(p.read_text())
            missing = [k for k in ("allin1", "madmom_key", "essentia_key") if (rec.get(k) or {}).get("status") != "complete"]
            bridge_skipped.append({"slug": p.stem, "why": f"provider incompleti: {missing}" if missing else "nessuna analisi DJ (tracks/) nel ponte"})
    unrenderable = []
    for slug in list(keep):
        why = renderable(slug, keep[slug])
        if why:
            unrenderable.append({"slug": slug, "why": why})
            del keep[slug]
    log.append({"pool_v2": {"from_v1b": sorted(s for s in keep if s in v1b_slugs), "missing_from_v1b": [s for s in v1b_slugs if s not in allsrc],
                            "unrenderable_here": unrenderable,
                            "bridge_added": bridge_added, "bridge_skipped": bridge_skipped,
                            "excluded_new_cc": sorted(s for s in allsrc if s not in keep)}})
    return keep


def renderable(slug: str, src: dict) -> str | None:
    """Motivo per cui il brano non si può caricare su questa macchina (audio o stem vocale assenti), o None."""
    an = v1.load_json(src["plan_dir"] / "tracks" / f"{slug}.json")
    wav = base.localize(an["inputs"]["canonical_wav"])
    if not wav.is_file():
        return f"audio canonico assente: {base.rel(wav)}"
    entry = next((t for t in src["catalog"]["tracks"] if t["slug"] == slug), None)
    if entry is None:
        return "assente dal catalogo"
    from musicmix.stems import StemError, vocal_stem_path  # percorso e identità dalla fase vocals, non dal nome

    try:
        vocal_stem_path(entry, check_sha=False)
    except StemError as exc:
        return f"stem vocale: {exc}"
    return None


# ------------------------------------------------------------------- form
def form_for(slug: str, src: dict, analysis: dict, params: dict | None = None) -> dict:
    """Documento ``dj_form`` sulla griglia di battute di dj_select (cache per hash dell'analisi e della sorgente).

    ``params``: sostituzioni di ``dj_form.PARAMS`` (es. ``{"kick_peak_on": None}`` = regola della cassa di dj_form.2);
    entrano nella chiave della cache."""
    beats, downs = v1.regular_grid(analysis)
    if len(downs) != len(analysis["phrases"]):
        raise RuntimeError(f"{slug}: griglia di {len(downs)} battute, tabella frasi di {len(analysis['phrases'])}")
    spec_path = next((d / f"{slug}.npy" for d in SPEC_DIRS if (d / f"{slug}.npy").is_file()), None)
    an_path = src["plan_dir"] / "tracks" / f"{slug}.json"
    key = {"analysis_sha256": base.sha256(an_path), "form": F.FORM_VERSION,
           "params": {k: (params[k] if params and k in params else v[0]) for k, v in F.PARAMS.items()},
           "spec": {"path": base.rel(spec_path), "size": spec_path.stat().st_size, "mtime": int(spec_path.stat().st_mtime)} if spec_path else None}
    cp = FORM_CACHE / f"{slug}.json"
    if cp.is_file():
        doc = v1.load_json(cp)
        if doc.get("key") == json.loads(json.dumps(key)):
            return doc["form"]
    form = None
    why = None
    if spec_path is not None:
        spec = np.load(spec_path, mmap_mode="r")
        dur_spec = spec.shape[1] / F.ALLIN1_FPS
        if abs(dur_spec - float(analysis["duration_s"])) <= 1.0:
            form = F.analyse_from_allin1_spec(np.asarray(spec), downs, beats, params=params)
        else:
            why = f"spettrogramma allin1 di {dur_spec:.1f} s, brano di {analysis['duration_s']:.1f} s"
    if form is None:
        T = v1.Track(slug, src["plan_dir"], src["catalog"])
        form = F.analyse(T.audio, SR, T.downbeats, T.beats, vocals=T.vocals, params=params)
        why = why or "nessuno spettrogramma allin1 per questo brano"
    form["provenance"] = {"spec": base.rel(spec_path) if spec_path else None, "fallback_reason": why,
                          "grid": "dj_select regularised grid (run_dj_gallery_v1.regular_grid)"}
    FORM_CACHE.mkdir(parents=True, exist_ok=True)
    base.dump(cp, {"key": key, "form": form})
    return form


def compact_form(form: dict) -> dict:
    """Solo ciò che serve a dj_set (le serie per battuta) e un riassunto leggibile."""
    return {"labels": form["labels"], "n_layers": form["n_layers"], "drops": form["drops"], "breakdowns": form["breakdowns"],
            "busy": form["busy"], "active": {"vocals": form["active"]["vocals"], "kick": form["active"]["kick"]},
            "core_level": form["core_level"], "core_status": form.get("core_status", "found"), "source": form["layers"]["source"]}


# ------------------------------------------------------------ candidates
def apply_filters(pc: dict, info_a: dict, order_over: dict) -> dict:
    """Come ``run_dj_set_v1.apply_filters`` ma con i costi del profilo (contesto incluso)."""
    pc = base.apply_filters(pc, info_a)
    for c in pc["candidates"]:
        c["cost"] = ds.candidate_cost(c["plan"], c["decision"], order_over, c["context"])
    pc["candidates"].sort(key=lambda c: (not c["admissible"], c["cost"], -c["plan"]["score"]))
    return pc


# ------------------------------------------------------------------ render
def render(dec: dict, A, B, plan: dict, prov: dict):
    if dec["type"] == "echo_out" and dec["render"].get("tail_stop"):  # B1 (Huron pp. 207-211): echo tail stops on B
        import dataclasses

        saved = base.FX_CFG
        base.FX_CFG = dataclasses.replace(saved, tail_under_b_db=None)
        try:
            return base.render(dec, A, B, plan, prov)
        finally:
            base.FX_CFG = saved
    if dec["type"] != "backspin":
        return base.render(dec, A, B, plan, prov)
    ia = int(np.argmin(np.abs(A.downbeats - plan["a_out_s"])))
    a_out, b_in = float(A.downbeats[ia]), float(plan["b_in_s"])
    res = cr.backspin(A.audio, B.audio, beats_a=A.beats, downbeats_a=A.downbeats, beats_b=B.beats, downbeats_b=B.downbeats,
                      cue_out_a=a_out, cue_in_b=b_in, vocals_a=A.vocals, vocals_b=B.vocals,
                      spin_beats=int(dec["render"].get("spin_beats", 2)), sample_rate=SR, config=CR_CFG,
                      provenance=prov, return_decks=True)
    return res, None


def run_qa(res, A, B, decks, tracker) -> dict:
    if decks is not None:
        return base.run_qa(res, A, B, decks, tracker)
    meta = res.metadata
    qi = cr.qa_inputs(res)
    ga = 10 ** (meta["loudness"]["a_gain_db"] / 20.0)
    gb = 10 ** (meta["loudness"]["b_gain_db"] / 20.0)
    words = cr.words_in_clip(meta, A.words, "a")
    return dj_qa.evaluate_render(res.audio.astype(np.float64), meta, sample_rate=res.sample_rate, decks=qi["decks"],
                                 vocals_a=qi["vocals_a"], vocals_b=qi["vocals_b"],
                                 vocal_thresholds=(A.vocal_threshold * ga, B.vocal_threshold * gb), words_a=words,
                                 tracker=tracker, grid_support={"a": A.analysis["tempo"], "b": B.analysis["tempo"]})


# ----------------------------------------------------------------- compare
def compare_with_v1b(doc: dict, v1b: dict) -> dict:
    def types(d):
        out: dict[str, int] = {}
        for tr in d["transitions"]:
            out[tr["type"]] = out.get(tr["type"], 0) + 1
        return out

    def durations(d):
        return {e["slug"]: round(e["set_seconds"], 1) for e in d["tracklist"]}

    def rules(d):
        out: dict[str, int] = {}
        for tr in d["transitions"]:
            out[tr["rule"]] = out.get(tr["rule"], 0) + 1
        return out

    d2, d1 = durations(doc), durations(v1b)
    return {
        "v1b": {"n_tracks": len(v1b["tracklist"]), "duration_s": v1b["output"]["duration_s"], "types": types(v1b),
                "rules": rules(v1b), "qa_final": v1b["qa_counts_final"], "qa_attempts": v1b["qa_counts_attempts"],
                "order": [e["slug"] for e in v1b["tracklist"]], "set_seconds": d1,
                "mean_set_seconds": float(np.mean(list(d1.values())))},
        "v2_books": {"n_tracks": len(doc["tracklist"]), "duration_s": doc["output"]["duration_s"], "types": types(doc),
                     "rules": rules(doc), "qa_final": doc["qa_counts_final"], "qa_attempts": doc["qa_counts_attempts"],
                     "order": [e["slug"] for e in doc["tracklist"]], "set_seconds": d2,
                     "mean_set_seconds": float(np.mean(list(d2.values())))},
        "common_tracks": sorted(set(d1) & set(d2)),
        "same_order": [e["slug"] for e in v1b["tracklist"]] == [e["slug"] for e in doc["tracklist"]],
    }


# ------------------------------------------------------------------- hooks
@dataclasses.dataclass
class SetHooks:
    """Punti di estensione di :func:`build_set` (default = comportamento di dj_set_v2_books).

    Usati da ``tools/run_dj_set_v3_bridge.py`` (catalogo congelato, dj_phrase, tecniche di dj_policy,
    veto del critico) senza duplicare questa pipeline.
    """
    run: str = RUN
    script: str = SCRIPT
    title: str = "set DJ v2 'libri'"
    pool: Callable[[list], dict] = None  # log -> sources (default: pool_sources)
    info: Callable[[str, dict], dict] = None  # (slug, src) -> feature/guardie per brano (default: run_dj_set_v1.track_info)
    filter_pair: Optional[Callable[..., dict]] = None  # (pc, a, b, analyses) -> pc, dopo i filtri del profilo
    attempts: Optional[Callable[..., list]] = None  # (ctx dict) -> lista (indice candidato, decisione, motivo)
    render: Callable = None  # (dec, A, B, plan, prov) -> (res, decks|None)
    run_qa: Callable = None  # (res, A, B, decks, tracker) -> qa
    veto: Optional[Callable[..., list]] = None  # (res, qa, dec, A, B, rec, t) -> motivi di veto (rec arricchito)
    lead_a_s: Callable = ds.lead_a_s
    overlap_b_s: Callable = ds.overlap_b_s
    on_accept: Optional[Callable[..., dict]] = None  # (t, a, b, res, dec, c, rec, qa) -> campi extra della transizione
    finalize_audio: Optional[Callable[..., tuple]] = None  # (audio) -> (audio, info) prima della scrittura del set
    mp3_bitrate: str = "256k"
    order_overrides: dict = dataclasses.field(default_factory=dict)  # sostituzioni di dj_set.ORDER_PARAMS (es. len_min)
    notes: tuple = ()
    compare_v1b: bool = True
    # ---- piano per eventi (P0.5, 21/10/2026): default = comportamento invariato
    form_params: Optional[dict] = None  # sostituzioni di dj_form.PARAMS (es. regola della cassa di dj_form.2)
    cue_overrides: dict = dataclasses.field(default_factory=dict)  # sostituzioni di dj_set.CUE_PARAMS (es. w_event_arrival)
    event_fn: Optional[Callable[..., dict]] = None  # (A, B, cand) -> riassunto eventi, per l'opzione sperimentale di choose_cues
    after_cues: Optional[Callable[[dict], dict]] = None  # (ctx) -> campi extra del set.json (anche con no_render)
    # ---- ripianificazione (21/10/2026): "replan" (default) o "keep_least_bad" (opzione esplicita, comportamento precedente)
    on_all_fail: str = "replan"
    max_replans: int = 6
    # ---- correzioni dei giudizi (21/10/2026, tools/apply_corrections.py): None = comportamento invariato
    corrections: Optional[dict] = None  # corrections_plan.json: cue fissati, coppie vietate, riuso dei render congelati
    # ---- affidabilità dalle schede brano (21/10/2026, musicmix.reliability.ReliabilityHook): None = comportamento invariato
    reliability: Optional[Any] = None  # .adjust_pair(pc, a, b, analyses) -> pc dopo i filtri; .report(ctx) -> set.json["reliability"]
    # ---- carica di A / risoluzione di B / continuità / tonalità / testo (22/10/2026, musicmix.dj_tension.TensionHook): None = invariato
    tension: Optional[Any] = None  # .adjust_pair(pc, a, b, analyses) dopo l'affidabilità; .order_attempts(plan, cands, a, b); .report(ctx)


# ------------------------------------------------------- replanning (21/10/2026)
#: Nessun ripiego che porti all'ascolto ciò che le analisi sconsigliano (orchestratore, 21/10/2026, revisione esterna): se
#: tutti i tentativi di un arco falliscono QA/veto, prima si provano altre uscite di A (prolungata alla frase successiva)
#: e altri ingressi di B compatibili con i vicini, poi l'arco viene tolto (costo infinito) e ordine e cue si rifanno; se il
#: set non si chiude più lo si dichiara nel set.json (``closed: false``). Tenere il tentativo meno grave è solo l'opzione
#: esplicita ``SetHooks.on_all_fail = 'keep_least_bad'`` (comportamento fino a questa data).
ON_ALL_FAIL = ("replan", "keep_least_bad")


class ArcFailed(Exception):
    def __init__(self, a: str, b: str, tried: list):
        super().__init__(f"{a} -> {b}: all attempts failed")
        self.a, self.b, self.tried = a, b, tried


def extended_attempts(t: int, order, analyses, cands, prev_b_in, nxt, cue_over, idx0: int, plan: list, limit: int = 12) -> list:
    """Seconda passata prima di togliere l'arco: candidati della coppia compatibili con i vicini non ancora in ``plan``,
    prima quelli con l'uscita di A più tardi di quella scelta (A prolungata alla frase successiva), poi gli altri (altri
    ingressi di B), ciascuno con la decisione dell'albero e le sue alternative."""
    seen = {i for i, _d, _w in plan}
    idx = ds.alternatives(t, order, analyses, cands, prev_b_in, nxt, cue_over, exclude=seen | {idx0}, limit=10 ** 6)
    a0 = cands[idx0]["plan"]["a_out_s"]
    b0 = cands[idx0]["plan"]["b_in_s"]
    later = sorted((i for i in idx if cands[i]["plan"]["a_out_s"] > a0 + 1e-6), key=lambda i: (cands[i]["plan"]["a_out_s"], cands[i]["cost"]))
    other_b = sorted((i for i in idx if i not in later and abs(cands[i]["plan"]["b_in_s"] - b0) > 1e-6), key=lambda i: cands[i]["cost"])
    out = []
    for i in (later + other_b)[:limit]:
        why = ("ripianificazione: A prolungata alla frase successiva" if i in later else "ripianificazione: altro ingresso di B")
        out.append((i, cands[i]["decision"], why))
        out += [(i, fb, why + " (alternativa dell'albero)") for fb in cands[i]["decision"].get("fallbacks", [])[:1]]
    return out


def _not_closed(out: Path, H, why: str, banned: list, replans: list, log: list) -> int:
    """Dichiara nel set.json che il set non si chiude (nessun fail tenuto, nessun audio)."""
    doc = {"schema_version": ds.SCHEMA_VERSION, "run": H.run, "script": H.script,
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "closed": False,
           "not_closed": {"why": why, "banned_arcs": [list(x) for x in banned], "replans": replans, "policy": H.on_all_fail},
           "note": "nessun set scritto: nessun ripiego che porti all'ascolto ciò che QA/veto sconsigliano", "log": log}
    base.dump(Path(out) / "set.json", doc)
    print(f"SET NON CHIUSO: {why}", flush=True)
    return 2


# -------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "runs" / RUN))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--length", type=int, default=None)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args(argv)
    return build_set(Path(args.out), overwrite=args.overwrite, length=args.length, no_render=args.no_render)


def build_set(out: Path, *, overwrite: bool = False, length: int | None = None, no_render: bool = False,
              hooks: SetHooks | None = None) -> int:
    """Pipeline del set (ordine, cue, render, QA, montaggio); ``hooks`` per le varianti (vedi :class:`SetHooks`)."""
    H = hooks or SetHooks()
    pool_fn = H.pool or pool_sources
    render_fn = H.render or render
    qa_fn = H.run_qa or run_qa
    out = Path(out)
    if out.exists() and any(p for p in out.iterdir() if p.name not in KEEP_ON_OVERWRITE):
        if not overwrite:
            raise SystemExit(f"{out} esiste e non è vuota: usare --out nuovo o --overwrite")
        for p in list(out.iterdir()):
            if p.name in KEEP_ON_OVERWRITE:
                continue
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    (out / "transitions").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log: list = []
    tree_over = dict(PROFILE["tree"])
    order_over = dict(PROFILE["order"], **H.order_overrides, **({"len_target": length} if length else {}))
    cue_over = dict(PROFILE["cues"], **H.cue_overrides)
    pol_params = dict(pol.PROFILE_BOOKS)

    # 1. tracks + forms -------------------------------------------------------
    sources = pool_fn(log)
    # lista di esclusione + lingua cantata (configs/excluded_tracks.json, configs/track_languages.json)
    sources = TF.filter_sources(sources, log)
    analyses = {s: v1.load_json(src["plan_dir"] / "tracks" / f"{s}.json") for s, src in sources.items()}
    infos = {s: (H.info or base.track_info)(s, src) for s, src in sources.items()}
    forms = {}
    for s in sorted(sources):
        print(f"forma di {s}", flush=True)
        forms[s] = form_for(s, sources[s], analyses[s], H.form_params)
        analyses[s]["form"] = compact_form(forms[s])
    slugs = sorted(analyses)
    print(f"{len(slugs)} brani: {', '.join(slugs)}", flush=True)

    # 2. candidates, edges, order ---------------------------------------------
    timbre = ds.timbre_distances({s: infos[s]["timbre"] for s in slugs}, slugs)
    ov = ds.values(ds.ORDER_PARAMS, order_over)
    far_thr = ds.far_timbre_calibration(timbre, ov["far_timbre_pct"], ov["far_timbre_min_spread"])  # [H] B1, astensione se degenere
    print(f"generi lontani [H]: {far_thr['status']} soglia {far_thr['threshold']}", flush=True)
    feats = {s: infos[s]["energy_features"] for s in slugs}
    z = ds.zscore_rows(np.array([[feats[s]["onset_rate"], feats[s]["flux_mean"], feats[s]["centroid_mean"]] for s in slugs]))
    energy_raw = {s: float(z[i].mean()) for i, s in enumerate(slugs)}
    energy = ds.energy_scores(energy_raw, slugs)
    pair_cands, edges = {}, {}
    E = np.full((len(slugs), len(slugs)), np.inf)
    for i, a in enumerate(slugs):
        for j, b in enumerate(slugs):
            if a == b:
                continue
            pc = apply_filters(ds.candidates_for_pair(analyses[a], analyses[b], tree_over, order_over), infos[a], order_over)
            pc = ds.apply_far_genre(pc, float(timbre[i, j]), far_thr, analyses[a]["tempo"]["bpm"],  # B1 [H]
                                    min_blend_s=ov["far_min_blend_s"], echo_tail_stop=ov["far_echo_tail_stop"])
            if H.filter_pair is not None:
                pc = H.filter_pair(pc, a, b, analyses)
            if H.corrections is not None:  # correzioni dell'utente: cue fissato o coppia vietata
                import apply_corrections as CORR
                pc = CORR.pin_pair(pc, analyses[a], analyses[b], H.corrections, tree_over, order_over)
            if H.reliability is not None:  # rischio dalle schede brano: costo in più, candidati/brani rischiosi evitati
                pc = H.reliability.adjust_pair(pc, a, b, analyses)
            if H.tension is not None:  # criteri dichiarati dall'utente (dj_tension): costo [H] tra i candidati ammissibili
                pc = H.tension.adjust_pair(pc, a, b, analyses)
            pair_cands[(a, b)] = pc
            e = ds.edge_cost(pc, analyses[a]["tempo"]["bpm"], analyses[b]["tempo"]["bpm"], float(timbre[i, j]),
                             energy[a], energy[b], order_over)
            edges[(a, b)] = e
            E[i, j] = e["total"]
    feasible_pairs = int(np.isfinite(E).sum())
    print(f"coppie ammissibili {feasible_pairs}/{len(slugs) * (len(slugs) - 1)}", flush=True)
    # 2b-4. ordine, cue, render; un arco che fallisce al render viene tolto e si ripianifica (21/10/2026)
    banned: list = []
    replans: list = []
    while True:
        E_it = E.copy()
        for ba, bb in banned:  # archi falliti al render in un giro precedente: costo infinito per l'ordinatore
            E_it[slugs.index(ba), slugs.index(bb)] = np.inf
        ordering = ds.order_set(slugs, E_it, energy, order_over)
        if not ordering["order"]:
            return _not_closed(out, H, "l'ordinatore non trova più un set con gli archi rimasti", banned, replans, log)
        order = ordering["order"]
        breakdown = ds.path_breakdown(order, edges, energy, order_over)
        print(f"ordine ({ordering['length']} brani, costo {ordering['cost']:.3f}): {' -> '.join(order)}", flush=True)

        # 3. cues -------------------------------------------------------------------
        try:
            cues = ds.choose_cues(order, analyses, pair_cands, cue_over, event_fn=H.event_fn)
        except ValueError as exc:
            return _not_closed(out, H, f"nessuna sequenza di cue: {exc}", banned, replans, log)
        chosen = cues["chosen"]
        import gc

        gc.collect()
        for t, c in enumerate(chosen):
            print(f"  {t + 1}. {order[t]} -> {order[t + 1]}: {c['decision']['label']} ({c['decision']['rule']}), "
                  f"A battuta {c['plan']['a_out_bar']}, B battuta {c['plan']['b_in_bar']}", flush=True)

        doc: dict[str, Any] = {
            "schema_version": ds.SCHEMA_VERSION, "run": H.run, "script": H.script,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "versions": {"dj_set": ds.SCHEMA_VERSION, "dj_select": djs.SCHEMA_VERSION, "dj_render": dj.DJ_RENDER_VERSION,
                         "dj_effects": base.fx.DJ_EFFECTS_VERSION, "dj_creative": cr.DJ_CREATIVE_VERSION, "dj_qa": dj_qa.DJ_QA_VERSION,
                         "dj_form": F.FORM_VERSION, "dj_policy": pol.POLICY_VERSION},
            "profile": {"name": "books", "dj_set": PROFILE, "dj_policy": pol_params,
                        "sources": "conoscenza_mix/libri/MAPPA_PIPELINE_KATZ_BB_WEBBER_BUTLER_WILLIAMS.md"},
            "params": {"tree": ds.documented(ds.TREE_PARAMS, tree_over), "type_cost": ds.documented(ds.TYPE_COST),
                       "order": ds.documented(ds.ORDER_PARAMS, order_over), "cues": ds.documented(ds.CUE_PARAMS, cue_over),
                       "tree_rules": ds.TREE_RULES, "form": {k: {"value": v[0], "source": v[1]} for k, v in F.PARAMS.items()},
                       "render": {"section_gain": base.SG.__dict__ if base.SG is not None else None, "blend_cue_convention": "start",
                                  "splice_crossfade_ms": 1000 * base.XFADE / SR, "fade_in_s": base.FADE_IN_S,
                                  "fade_out_bars": base.FADE_OUT_BARS}},
            "sources": {s: {"source": src["source"], "analysis": base.rel(src["plan_dir"] / "tracks" / f"{s}.json"),
                            "analysis_sha256": base.sha256(src["plan_dir"] / "tracks" / f"{s}.json"),
                            "bpm": analyses[s]["tempo"]["bpm"], "camelot": analyses[s]["key"]["camelot"], "energy": energy[s],
                            "title": infos[s].get("title"), "artist": infos[s].get("artist"),
                            "form": {**F.form_summary(forms[s]), "provenance": forms[s].get("provenance")}}
                        for s, src in sources.items()},
            "pool": {"n_tracks": len(slugs), "feasible_ordered_pairs": feasible_pairs,
                     "edge_matrix": {"slugs": slugs, "cost": [[None if not math.isfinite(x) else x for x in row] for row in E]}},
            "ordering": {**ordering, "breakdown": breakdown},
            "cues": {"cost": cues["cost"], "play_s": cues["play_s"], "play_bounds_s": cues["play_bounds_s"],
                     "relaxations": cues["relaxations"], "first": cues["first"], "last": cues["last"],
                     "section_requirement": cues.get("section_requirement"), "dur_mode": cues.get("dur_mode"),
                     "sections_played": cues.get("sections_played")},
            "log": log,
            "replanning": {"policy": H.on_all_fail, "replans": replans, "banned_arcs": [list(x) for x in banned],
                           "rule": "se tutti i tentativi di un arco falliscono QA/veto: altra uscita di A (prolungata), altro ingresso "
                                   "di B, poi arco tolto e ordine rifatto; nessun fail tenuto (opzione esplicita on_all_fail="
                                   "'keep_least_bad')"},
        }
        if cues.get("event_pref"):
            doc["cues"]["event_pref"] = cues["event_pref"]
        if H.form_params:
            doc["params"]["form_overrides"] = dict(H.form_params)
        if H.corrections is not None:
            import apply_corrections as CORR
            doc["corrections"] = CORR.provenance(H.corrections)
        if H.reliability is not None:
            doc["reliability"] = H.reliability.report({"order": order, "chosen": chosen})
        if H.tension is not None:
            doc["tension"] = H.tension.report({"order": order, "chosen": chosen})
        if H.after_cues is not None:
            doc.update(H.after_cues({"order": order, "chosen": chosen, "pair_cands": pair_cands, "analyses": analyses, "edges": edges,
                                     "cue_over": cue_over, "order_over": order_over, "tree_over": tree_over, "cues": cues,
                                     "no_render": no_render}))
        if no_render:
            base.dump(out / "set.json", doc)
            print(f"solo piano: {out / 'set.json'}")
            return 0

        try:
            # 4. render + QA ------------------------------------------------------------
            tracker = dj_qa.BeatThisTracker()
            loaded: dict[str, Any] = {}

            def load(slug, keep):
                for k in list(loaded):
                    if k not in keep:
                        del loaded[k]
                if slug not in loaded:
                    src = sources[slug]
                    loaded[slug] = v1.Track(slug, src["plan_dir"], src["catalog"])
                return loaded[slug]

            results, transitions = [], []
            counts = {"pass": 0, "warn": 0, "fail": 0, "render_error": 0}
            final_counts = {"pass": 0, "warn": 0, "fail": 0}
            prev_b_in = prev_ovl = None
            recent: list[str] = []
            cp = ds.values(ds.CUE_PARAMS, cue_over)
            for t in range(len(order) - 1):
                a, b = order[t], order[t + 1]
                A, B = load(a, (a, b)), load(b, (a, b))
                pc = pair_cands[(a, b)]
                cands = pc["candidates"]
                idx0 = next(i for i, c in enumerate(cands) if c is chosen[t])
                nxt = None
                if t + 1 < len(chosen):
                    nc = chosen[t + 1]
                    nxt = (nc["plan"]["a_out_s"], ds.lead_a_s(nc, analyses[b]))

                def fits(c, dec):
                    cc = {"plan": c["plan"], "decision": dec}
                    if prev_b_in is not None and (c["plan"]["a_out_s"] - H.lead_a_s(cc, analyses[a])) - (prev_b_in + prev_ovl) < cp["min_gap_s"]:
                        return False
                    if nxt is not None and (nxt[0] - nxt[1]) - (c["plan"]["b_in_s"] + H.overlap_b_s(cc, analyses[b])) < cp["min_gap_s"]:
                        return False
                    return True

                attempts_plan = [(idx0, chosen[t]["decision"], "scelta del Viterbi")]
                attempts_plan += [(idx0, fb, "alternativa dell'albero sullo stesso cue") for fb in chosen[t]["decision"]["fallbacks"]]
                for i in ds.alternatives(t, order, analyses, cands, prev_b_in, nxt, cue_over, exclude=[idx0]):
                    c = cands[i]
                    attempts_plan.append((i, c["decision"], "altro cue compatibile con i vicini"))
                    attempts_plan += [(i, fb, "alternativa dell'albero su altro cue") for fb in c["decision"]["fallbacks"]]
                if H.attempts is not None:
                    attempts_plan = H.attempts({"t": t, "a": a, "b": b, "A": A, "B": B, "cands": cands, "idx0": idx0, "pc": pc,
                                                "chosen": chosen[t], "plan": attempts_plan, "recent": list(recent),
                                                "analyses": analyses})
                if H.tension is not None and hasattr(H.tension, "order_attempts"):  # base continua prima, sullo stesso cue
                    attempts_plan = H.tension.order_attempts(attempts_plan, cands, a, b)
                if H.corrections is not None:  # tecnica fissata dalle correzioni per prima
                    import apply_corrections as CORR
                    attempts_plan = CORR.prefer_technique(attempts_plan, a, b, H.corrections)
                if H.on_all_fail == "replan":  # seconda passata: A prolungata alla frase successiva, poi altri ingressi di B
                    attempts_plan += extended_attempts(t, order, analyses, cands, prev_b_in, nxt, cue_over, idx0, attempts_plan)
                tried, accepted, best_fail = [], None, None
                cap = base.MAX_ATTEMPTS * (2 if H.on_all_fail == "replan" else 1)
                for i, dec, why in attempts_plan:
                    if len(tried) >= cap:
                        break
                    c = cands[i]
                    plan = c["plan"]
                    if not fits(c, dec):
                        tried.append({"candidate": i, "type": dec["type_key"], "why": why, "skipped": "spazio insufficiente con le transizioni vicine"})
                        continue
                    if any(x.get("candidate") == i and x.get("type") == dec["type_key"] and "verdict" in x for x in tried):
                        continue
                    prov = {"a": A.provenance, "b": B.provenance, "script": H.script, "run": H.run, "transition": t,
                            "plan": {k: plan[k] for k in ("a_out_s", "a_out_bar", "b_in_s", "b_in_bar", "type", "blend_bars", "score")},
                            "decision": {k: dec.get(k) for k in ("type", "bars", "render", "rule")}}
                    rec = {"candidate": i, "type": dec["type_key"], "label": dec["label"], "why": why,
                           "a_out_bar": plan["a_out_bar"], "b_in_bar": plan["b_in_bar"], "a_out_s": plan["a_out_s"], "b_in_s": plan["b_in_s"]}
                    reused = None
                    if H.corrections is not None:  # transizione non corretta: render e QA del set congelato
                        import apply_corrections as CORR
                        reused = CORR.reuse_render(H.corrections, a, b, plan, dec)
                        rec["reused"] = bool(reused)
                    try:
                        res, decks = (reused[0], None) if reused else render_fn(dec, A, B, plan, prov)
                    except (dj.DJRenderError, ValueError) as exc:
                        counts["render_error"] += 1
                        rec.update(verdict="render_error", reasons=[str(exc)])
                        tried.append(rec)
                        print(f"   RENDER ERROR {a}->{b} {dec['type_key']}: {exc}", flush=True)
                        continue
                    qa = reused[1] if reused else qa_fn(res, A, B, decks, tracker)
                    counts[qa["verdict"]] += 1
                    rec.update(verdict=qa["verdict"], reasons=[f"{r['check']}: {r['message']}" for r in qa["reasons"]])
                    vetoes = H.veto(res, qa, dec, A, B, rec, t) if H.veto is not None and not reused else []
                    if vetoes:
                        rec["veto"] = vetoes
                        counts["vetoed"] = counts.get("vetoed", 0) + 1
                    tried.append(rec)
                    name = f"t{t + 1:02d}_{len(tried):02d}_{dec['type_key']}"
                    base.dump(out / "transitions" / f"{name}.json", {"attempt": rec, "render": res.metadata, "qa": qa})
                    print(f"   {qa['verdict'].upper():4s} {a} -> {b} {dec['label']} (A {plan['a_out_bar']}, B {plan['b_in_bar']}) "
                          + "; ".join(rec["reasons"]) + (f" | VETO: {'; '.join(vetoes)}" if vetoes else ""), flush=True)
                    if qa["verdict"] != "fail" and not vetoes:
                        accepted = (i, dec, res, qa, rec)
                        break
                    nfail = sum(1 for r in qa["reasons"] if r["severity"] == "fail") + len(vetoes)
                    if best_fail is None or nfail < best_fail[0]:
                        best_fail = (nfail, (i, dec, res, qa, rec))
                kept_fail = False
                if accepted is None:
                    if H.on_all_fail != "keep_least_bad" or best_fail is None:
                        print(f"   ARCO FALLITO {a} -> {b}: {len(tried)} tentativi falliscono QA/veto: arco tolto, ordine rifatto", flush=True)
                        raise ArcFailed(a, b, tried)
                    accepted = best_fail[1]  # opzione esplicita (comportamento fino al 21/10/2026)
                    kept_fail = True
                    print(f"   ATTENZIONE: {a} -> {b}: tutti i tentativi falliscono il QA, tenuto il meno grave (on_all_fail='keep_least_bad')", flush=True)
                i, dec, res, qa, rec = accepted
                c = cands[i]
                final_counts[qa["verdict"]] += 1
                prev_b_in = c["plan"]["b_in_s"]
                prev_ovl = H.overlap_b_s({"plan": c["plan"], "decision": dec}, analyses[b])
                if i != idx0:
                    chosen[t] = c
                geo = ds.clip_geometry(res.metadata, len(res.audio))
                results.append({"audio": res.audio, "meta": res.metadata, "geo": geo})
                ja, jb = int(c["plan"]["a_out_bar"]), int(c["plan"]["b_in_bar"])
                nb = max(1, int(dec.get("bars") or 1))
                form_qa = F.overlap_profile(analyses[a]["form"], analyses[b]["form"], ja, jb, nb)
                try:
                    pctx = pol.creative_context(analyses[a], analyses[b], c["plan"], pc["compatibility"], has_stems=True,
                                                recent_techniques=recent[-2:])
                    props = pol.propose(pctx, params=pol_params)
                    policy = [{"technique": p["technique"], "rule": p["rule"], "score": round(p["score"], 3), "why": p["why"]} for p in props["proposals"]]
                except Exception as exc:  # proposals are only recorded
                    policy = [{"error": str(exc)}]
                recent.append(dec["type"])
                extra = H.on_accept(t, a, b, res, dec, c, rec, qa) if H.on_accept is not None else {}
                transitions.append({
                    "index": t + 1, "a": a, "b": b, "type": dec["type_key"], "label": dec["label"],
                    "rule": dec["rule"], "rule_source": dec["rule_source"], "trace": dec.get("trace"),
                    "context": c["context"], "notes": c["notes"],
                    "placement_cost": ds.placement_cost(c["plan"], c["context"], order_over),
                    "dense_overlap_cost": ds.dense_overlap_cost(dec, c["context"], order_over),
                    "form_overlap": form_qa, "policy_proposals_books": policy,
                    "plan": {k: c["plan"][k] for k in ("a_out_s", "a_out_bar", "b_in_s", "b_in_bar", "type", "blend_bars", "cue_score",
                                                       "score", "confidence", "a_section", "b_section")},
                    "plan_reasons": c["plan"]["reasons"], "pair_reasons": pc["compatibility"]["reasons"],
                    "pair_status": pc["compatibility"]["status"], "edge": edges[(a, b)],
                    "qa": {"verdict": qa["verdict"], "reasons": rec["reasons"], "kept_despite_fail": kept_fail,
                           "loudness_windows_db": {k: w.get("jump_db") for k, w in
                                                   ((qa.get("checks") or {}).get("loudness_jump") or {}).get("windows", {}).items()}},
                    "attempts": tried,
                    "clip": {"duration_s": len(res.audio) / SR, "transition_start_s": geo["transition_start"] / SR,
                             "switch_s": geo["switch"] / SR, "transition_end_s": geo["transition_end"] / SR},
                    "stretch_pct": 100.0 * (geo["b_rate"] - 1.0),
                    "section_gain_b_db": (res.metadata.get("section_gain") or {}).get("applied_db"),
                    **extra,
                })

        except ArcFailed as exc:
            banned.append((exc.a, exc.b))
            replans.append({"arc": [exc.a, exc.b], "order": list(order), "tried": exc.tried})
            if len(replans) > H.max_replans:
                return _not_closed(out, H, f"{len(replans)} archi falliti (max_replans={H.max_replans})", banned, replans, log)
            continue
        break

    # 5. assemble -----------------------------------------------------------------
    req = cues.get("section_requirement", "none")
    first = ds.first_start(analyses[order[0]], chosen[0]["plan"]["a_out_s"], cp, chosen[0]["plan"].get("a_out_bar"), req)
    last = ds.last_end(analyses[order[-1]], chosen[-1]["plan"]["b_in_s"], cp, chosen[-1]["plan"].get("b_in_bar"), req)
    loaded.clear()
    cache: dict[str, Any] = {}

    def loader(slug):
        if slug not in cache:
            if len(cache) >= 2:
                cache.pop(next(iter(cache)))
            src = sources[slug]
            cache[slug] = v1.Track(slug, src["plan_dir"], src["catalog"])
        return cache[slug]

    audio, seg_info, splices = base.assemble(order, loader, results, first, last)
    cache.clear()
    for tr, sp, r in zip(transitions, splices, results):
        o = sp["clip_origin_set_sample"]
        g = r["geo"]
        tr["set_times_s"] = {"start": (o + g["transition_start"]) / SR, "switch": (o + g["switch"]) / SR, "end": (o + g["transition_end"]) / SR}
        tr["splice"] = sp
    junctions = base.junction_check(audio, seg_info)
    final_info = None
    if H.finalize_audio is not None:
        audio, final_info = H.finalize_audio(audio)
    tp = max(dj.true_peak_dbtp(audio[s:s + 30 * SR], SR) for s in range(0, len(audio), 30 * SR))
    import soundfile as sf

    wav = out / "set.wav"
    sf.write(str(wav), audio, SR, subtype="PCM_24")
    mp3 = out / "set.mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", H.mp3_bitrate, str(mp3)], check=True)
    dur = len(audio) / SR

    entries = []
    for k, slug in enumerate(order):
        start = 0.0 if k == 0 else transitions[k - 1]["set_times_s"]["switch"]
        end = dur if k == len(order) - 1 else transitions[k]["set_times_s"]["switch"]
        entries.append({"position": k + 1, "slug": slug, "artist": infos[slug].get("artist"), "title": infos[slug].get("title"),
                        "set_start_s": start, "set_end_s": end, "set_seconds": end - start,
                        "bpm": analyses[slug]["tempo"]["bpm"], "camelot": analyses[slug]["key"]["camelot"], "energy": energy[slug],
                        "source_play_s": cues["play_s"][k],
                        "sections_played": (cues.get("sections_played") or [None] * len(order))[k]})
    lines = [f"MusicMix — {H.title} ({len(order)} brani, {base.fmt_t(dur)}), {doc['created_at']}",
             f"ordine: costo {ordering['cost']:.3f} ({ordering['note']}); cue: costo {cues['cost']:.3f}; "
             f"durata a sezioni: requisito '{req}'", ""]
    for k, e in enumerate(entries):
        who = " - ".join(x.strip() for x in (e["artist"] or "", e["title"] or e["slug"]) if x and x.strip())
        lines.append(f"{e['position']:2d}. [{base.fmt_t(e['set_start_s'])}] {who}  ({e['bpm']:.1f} BPM, {e['camelot']}, energia {e['energy']:.2f}; "
                     f"{e['set_seconds']:.0f} s nel set)")
        if k < len(transitions):
            tr = transitions[k]
            st = tr["set_times_s"]
            fo = tr["form_overlap"]
            lines.append(f"      -> transizione {tr['index']}: {tr['label']} [{tr['rule']}] inizio {base.fmt_t(st['start'])}, switch "
                         f"{base.fmt_t(st['switch'])}, fine {base.fmt_t(st['end'])}; QA {tr['qa']['verdict']}"
                         + (" (fail tenuto)" if tr["qa"]["kept_despite_fail"] else "")
                         + f"; forma A '{fo.get('a_form_before')}'→'{fo.get('a_form_at')}', B '{fo.get('b_form_at')}'"
                         + (" (drop)" if fo.get("b_entry_drop") else ""))
    (out / "tracklist.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / "tracklist.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["position", "slug", "artist", "title", "set_start_s", "set_end_s", "bpm", "camelot", "energy",
                    "transition_out_type", "rule", "transition_start_s", "switch_s", "transition_end_s", "qa"])
        for k, e in enumerate(entries):
            tr = transitions[k] if k < len(transitions) else None
            w.writerow([e["position"], e["slug"], (e["artist"] or "").strip(), (e["title"] or "").strip(), f"{e['set_start_s']:.3f}",
                        f"{e['set_end_s']:.3f}", f"{e['bpm']:.2f}", e["camelot"], f"{e['energy']:.3f}",
                        tr["type"] if tr else "", tr["rule"] if tr else "",
                        f"{tr['set_times_s']['start']:.3f}" if tr else "", f"{tr['set_times_s']['switch']:.3f}" if tr else "",
                        f"{tr['set_times_s']['end']:.3f}" if tr else "", tr["qa"]["verdict"] if tr else ""])
    doc.update({
        "closed": True,
        "tracklist": entries, "transitions": transitions, "segments": seg_info, "junctions": junctions,
        "qa_counts_attempts": counts, "qa_counts_final": final_counts,
        "output": {"wav": base.rel(wav), "wav_sha256": base.sha256(wav), "mp3": base.rel(mp3), "mp3_sha256": base.sha256(mp3),
                   "duration_s": dur, "sample_rate": SR, "true_peak_dbtp": tp, "finalize": final_info,
                   "tracklist_txt": base.rel(out / "tracklist.txt"), "tracklist_csv": base.rel(out / "tracklist.csv")},
        "elapsed_s": time.time() - t0,
        "notes": [
            "integrità tecnica (QA) non è qualità musicale: il set non è stato ascoltato",
            "le regole del profilo 'books' sono pratica/analisi dai libri, non evidenza; i pesi sono [H]",
            "le categorie di Butler (dj_form) hanno soglie [H] non validate su ascolti; nell'hip-hop la 'cassa' include l'808",
            "pool: stessi brani di dj_set_v1b + brani del ponte con analisi DJ completa (vedi log.pool_v2)",
        ] if not H.notes else list(H.notes),
    })
    if H.compare_v1b and V1B.is_file():
        doc["compare_v1b"] = compare_with_v1b(doc, json.loads(V1B.read_text(encoding="utf-8")))
    base.dump(out / "set.json", doc)
    print(json.dumps({"durata": base.fmt_t(dur), "qa_finale": final_counts, "tentativi": counts}), f"{time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
