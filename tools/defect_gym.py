#!/usr/bin/env python3
"""Palestra automatica degli errori: riferimenti, varianti con difetti controllati, gate di QA e critico.

Fasi (``conoscenza_mix/_meta/PALESTRA_DIFETTI_2026-10-21.md``; libreria :mod:`musicmix.defect_gym`):

* ``refs``: scrive ``runs/defect_gym_v1/refs.json``: transizioni con voto utente >= 6 (``giudizi_cloud/``, clip e
  run dai manifest), transizioni del set v3 con QA pass (verdetto salvato), set v3 con lead-in di B (comportamento
  corrente del runner, per il difetto "pickup tolto"; tenute solo se QA e veto le accettano), e continuazioni
  naturali dentro i brani (cut e blend di un brano con se stesso su un confine di sezione libero da parole).
  Solo brani ammessi da ``musicmix.track_filter`` (en/it, non esclusi).
* ``run``: per ogni riferimento rifà il render con la ricetta originale (verifica dell'identità PCM quando la clip
  salvata esiste), lo valuta (``dj_qa`` come il runner v3, critico v3 fast + blocco v4 ``technical``), genera le
  varianti (:data:`musicmix.defect_gym.DEFECTS`, livelli e combinazioni) e le valuta. Risultati incrementali in
  ``results.jsonl`` (ripresa automatica), nessun WAV salvato salvo ``--keep-wav`` (clip di 20 s per l'ascolto).
* ``report``: aggregazione, gate (``configs/defect_gym_gate.json``), ``report.json`` e ``REPORT.md``.

    python tools/defect_gym.py refs
    python tools/defect_gym.py run [--refs nat_ ...] [--defects phase gap] [--no-combos]
    python tools/defect_gym.py report
    python tools/defect_gym.py refs run report --profile gate --out <cartella>   # profilo del test -m gym

Un processo pesante alla volta (Beat This!, Rubber Band): RAM ~2-3 GB. QA e critico verificano integrità tecnica,
non gusto: riconoscere un peggioramento non equivale a riconoscere una bella transizione.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from musicmix import defect_gym as G  # noqa: E402

OUT = ROOT / "runs" / "defect_gym_v1"
GATE = ROOT / "configs" / "defect_gym_gate.json"
SR = 48000
NATURAL_N_TRACKS = 8
NATURAL_BLEND_TRACKS = 4
NATURAL_SECTION_TRACKS = 3
NATURAL_CUTS_PER_TRACK = 2
NATURAL_GUARD_PRE_S = 0.15   # [H] come WORD_GUARD_S di dj_qa: nessuna parola negli ultimi 150 ms prima del downbeat
NATURAL_GUARD_POST_S = 0.05  # [H] né nei primi 50 ms dopo
NATURAL_REST_BEATS = 0.25    # [H] respiro attorno al downbeat >= 1/4 di battito (QA b_entry: 3/16)
NATURAL_MIN_PRE_S = 14.0
NATURAL_MIN_POST_S = 30.0
MIN_VOTE = 6
# Varianti per origine del riferimento (costo: ~25 s per variante su questa macchina). I naturali di inizio sezione servono
# solo ai falsi allarmi; i riferimenti con lead-in solo ai difetti dell'ingresso di B; le combinazioni sui naturali.
ORIGIN_POLICY = {
    "natural": {"combos": True},
    "natural_blend": {"combos": True},
    "natural_section": {"defects": []},
    "set_v3_lead_in": {"defects": ["pickup_removed", "word_cut_b"]},
    "user_vote": {"combos": False},
    "set_v3_qa_pass": {"combos": False},
}


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(p: Path) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def dump(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(G.jsonable(obj), indent=1, ensure_ascii=False), encoding="utf-8")


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ===================================================================================== runtime (lazy)
class RT:
    """Moduli pesanti caricati una volta (runner v3, galleria v1, QA, tracker)."""

    _ok = False

    @classmethod
    def init(cls) -> None:
        if cls._ok:
            return
        import run_dj_set_v3_bridge as R
        import run_dj_gallery_v1 as G1
        from musicmix import dj_creative as cr, dj_qa, dj_render as dj, dj_select as djs
        from musicmix import track_filter as TF

        cls.R, cls.G1, cls.cr, cls.qa, cls.dj, cls.djs, cls.TF = R, G1, cr, dj_qa, dj, djs, TF
        cls.tracker = None
        cls._ok = True

    @classmethod
    def get_tracker(cls):
        if cls.tracker is None:
            cls.tracker = cls.qa.BeatThisTracker()
        return cls.tracker


_TRACKS: dict[tuple, Any] = {}


def track(slug: str, plan_dir: Path, catalog: dict, keep: tuple = ()) -> Any:
    key = (str(plan_dir), slug)
    for k in list(_TRACKS):
        if k != key and k[1] not in keep:
            del _TRACKS[k]
    if key not in _TRACKS:
        _TRACKS[key] = RT.R.v1.Track(slug, plan_dir, catalog)
    return _TRACKS[key]


def shifted(T: Any, offset_s: float) -> Any:
    """Copia superficiale del brano con la griglia (battiti e downbeat) spostata di ``offset_s`` (errore del tracker)."""
    if not offset_s:
        return T
    U = copy.copy(T)
    U.beats = np.asarray(T.beats, dtype=float) + offset_s
    U.downbeats = np.asarray(T.downbeats, dtype=float) + offset_s
    return U


# ===================================================================================== references
def _judgements() -> list[dict]:
    out = []
    for f in sorted((ROOT / "giudizi_cloud").glob("*/judgments/*.json")):
        d = load(f)
        out.append({**d, "file": str(f.relative_to(ROOT))})
    return out


def _admitted(slugs, tf) -> Optional[str]:
    for s in slugs:
        chk = tf.check(s)
        if not chk["admitted"]:
            return f"{s}: {chk.get('why')}"
    return None


def build_refs() -> dict:
    RT.init()
    R, TF = RT.R, RT.TF
    tf = TF.default_filter()
    refs, skipped = [], []
    judg = _judgements()
    by_clip = {}
    for j in judg:
        if j.get("score") is not None:
            by_clip.setdefault((j["run"], j["clip_id"]), []).append(j)
    # (1) voti utente >= 6 -------------------------------------------------------------------------------------
    g1 = load(ROOT / "runs/dj_gallery_v1/review_manifest.json")
    g1_items = {it["id"]: (g, it) for g in g1["groups"] for it in g["items"]}
    v3m = load(ROOT / "runs/dj_set_v3_bridge/gallery/review_manifest.json")
    v3_items = {it["id"]: (g, it) for g in v3m["groups"] for it in g["items"]}
    key = load(ROOT / "runs/dj_set_v3_bridge/gallery/blind_key.json")["groups"]
    setdoc = load(ROOT / "runs/dj_set_v3_bridge/set.json")
    trs = {t["index"]: t for t in setdoc["transitions"]}
    for (run, cid), js in sorted(by_clip.items()):
        last = sorted(js, key=lambda j: j.get("saved_at") or "")[-1]
        score = last["score"]
        if score < MIN_VOTE:
            continue
        jinfo = {"score": score, "blind": bool(last.get("blind")), "file": last["file"], "n_judgements": len(js)}
        if run == "dj_gallery_v1":
            g, it = g1_items[cid]
            clip_json = load(ROOT / "runs/dj_gallery_v1/clips" / f"{cid}.json")
            r = clip_json["render"]
            a, b = r["provenance"]["a"]["slug"], r["provenance"]["b"]["slug"]
            why = _admitted((a, b), tf)
            if why:
                skipped.append({"clip": cid, "run": run, "reason": why})
                continue
            plan = r["provenance"]["plan"]
            v = {"kind": r["kind"], "bars": r.get("bars")}
            if r["kind"] == "filter_out":
                v["sweep_bars"] = RT.G1.FILTER_SWEEP_BARS
            refs.append({"id": f"uv_g1_{cid.split('__')[0]}_{r['kind']}{r.get('bars') or ''}", "origin": "user_vote", "clean": False,
                         "judgement": jinfo, "a": a, "b": b, "kind": r["kind"],
                         "recipe": {"type": "gallery_v1", "clip": cid, "version": v,
                                    "plan": {"a_out_s": plan["a_out_s"], "b_in_s": plan["b_in_s"], "b_in_bar": plan["b_in_bar"],
                                             "a_out_bar": plan["a_out_bar"]},
                                    "expected_wav_sha256": sha_file(ROOT / "runs/dj_gallery_v1/clips" / f"{cid}.wav")}})
        elif run == "dj_set_v3_bridge":
            g, it = v3_items[cid]
            gid = g["id"]
            if gid.startswith("verifica"):
                a, b = "heartless", "coehn_foehrb_alles_coehn_und_gut_j111121"
                why = _admitted((a, b), tf) or "runs/books_integration_check: ricetta non riprodotta"
                skipped.append({"clip": cid, "run": run, "group": gid, "reason": why, "score": score})
                continue
            kinfo = key[gid][cid]
            k = int(key[gid]["transition"])
            tr = trs[k]
            why = _admitted((tr["a"], tr["b"]), tf)
            if why:
                skipped.append({"clip": cid, "run": run, "reason": why})
                continue
            role = kinfo["profile"]
            is_set_clip = role == "books" and key[gid].get("b_is_set_clip")
            refs.append({"id": f"uv_v3_{gid}_{role}_{kinfo['technique']}", "origin": "user_vote", "clean": False, "judgement": jinfo,
                         "a": tr["a"], "b": tr["b"], "kind": kinfo["technique"],
                         "recipe": {"type": "set_v3" if is_set_clip else "set_v3_gallery", "index": k, "group": gid, "role": role,
                                    "technique": kinfo["technique"], "clip": cid, "b_entry": False,
                                    "expected_trimmed_sha256": it.get("sha256"),
                                    "expected_wav_sha256": tr.get("clip_sha256") if is_set_clip else None}})
    # (2) set v3 con QA pass (verdetto salvato, dj_qa.4) ---------------------------------------------------------
    have = {(r["recipe"].get("type"), r["recipe"].get("index")) for r in refs}
    for k, tr in sorted(trs.items()):
        if tr["qa"]["verdict"] != "pass" or (tr.get("critic") or {}).get("veto"):
            continue
        why = _admitted((tr["a"], tr["b"]), tf)
        if why:
            skipped.append({"transition": k, "reason": why})
            continue
        if ("set_v3", k) in have:
            continue
        refs.append({"id": f"v3_t{k:02d}_{tr['technique']}", "origin": "set_v3_qa_pass", "clean": False, "a": tr["a"], "b": tr["b"],
                     "kind": tr["technique"], "saved_qa": tr["qa"]["verdict"],
                     "recipe": {"type": "set_v3", "index": k, "technique": tr["technique"], "b_entry": False,
                                "expected_wav_sha256": tr.get("clip_sha256")}})
    # (3) set v3 con lead-in di B (runner corrente): candidati, tenuti solo se accettati al run --------------------
    from musicmix import dj_render as dj
    for k, tr in sorted(trs.items()):
        if tr["technique"] not in dj.B_LEAD_IN_KINDS:
            continue
        if _admitted((tr["a"], tr["b"]), tf):
            continue
        refs.append({"id": f"li_t{k:02d}_{tr['technique']}", "origin": "set_v3_lead_in", "clean": False, "a": tr["a"], "b": tr["b"],
                     "kind": tr["technique"], "candidate": True,
                     "recipe": {"type": "set_v3", "index": k, "technique": tr["technique"], "b_entry": True, "require_lead_in": True}})
    # (4) continuazioni naturali ------------------------------------------------------------------------------------
    cat = load(R.FROZEN)
    ok_tracks = []
    for t in cat["tracks"]:
        s = t["slug"]
        if s in R.EXCLUDE or not (R.PHRASE / f"{s}.json").is_file() or _admitted((s,), tf):
            continue
        an = load(R.PLAN / "tracks" / f"{s}.json")
        if an["tempo"].get("confidence") not in ("high", "medium"):
            continue
        ok_tracks.append((hashlib.sha256(f"defect_gym_v1:{s}".encode()).hexdigest(), s, an))
    ok_tracks.sort()
    nat = []
    n_tracks = n_sec = 0
    for _, s, an in ok_tracks:
        if n_tracks >= NATURAL_N_TRACKS and n_sec >= NATURAL_SECTION_TRACKS:
            break
        R.load_phrase({s})
        words = load(Path(_vocals_json(cat, s)))["words"] + [
            {"start_sample": int(w["on"]), "end_sample": int(w["off"])} for w in R.MODELS[s]["words"] if not w.get("dropped")]
        bounds = _natural_boundaries(an, words)
        inner = [x for x in bounds if not x["section_start"]]
        starts = [x for x in bounds if x["section_start"]]
        if n_sec < NATURAL_SECTION_TRACKS and starts:
            n_sec += 1
            x = starts[0]
            nat.append({"id": f"nsec_{s}_bar{x['bar']}_cut", "origin": "natural_section", "clean": True, "a": s, "b": s, "kind": "cut",
                        "recipe": {"type": "natural", "slug": s, "bar": x["bar"], "kind": "cut"}, "boundary": x})
        if n_tracks >= NATURAL_N_TRACKS:
            continue
        if not inner:
            skipped.append({"natural": s, "reason": "nessun inizio frase interno a una sezione libero da parole"})
            continue
        i = n_tracks
        n_tracks += 1
        x = inner[0]
        picks = [x] + [y for y in inner[1:] if abs(y["bar"] - x["bar"]) >= 8][:NATURAL_CUTS_PER_TRACK - 1]
        for y in picks:
            nat.append({"id": f"nat_{s}_bar{y['bar']}_cut", "origin": "natural", "clean": True, "a": s, "b": s, "kind": "cut",
                        "recipe": {"type": "natural", "slug": s, "bar": y["bar"], "kind": "cut"}, "boundary": y})
        if i < NATURAL_BLEND_TRACKS:
            far = [y for y in bounds if abs(y["bar"] - x["bar"]) >= 8 and y["section"] == x["section"]] or \
                  [y for y in bounds if abs(y["bar"] - x["bar"]) >= 8]
            if far:
                y = far[0]
                nat.append({"id": f"nbl_{s}_bar{x['bar']}to{y['bar']}_blend4", "origin": "natural_blend", "clean": False, "a": s, "b": s,
                            "kind": "blend", "recipe": {"type": "natural", "slug": s, "bar": x["bar"], "bar_b": y["bar"], "kind": "blend",
                                                        "bars": 4}, "boundary": x, "boundary_b": y})
    refs += nat
    doc = {"schema": "musicmix.defect_gym.refs.1", "version": G.GYM_VERSION, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "rules": {"user_vote": f"ultimo voto per clip >= {MIN_VOTE} in giudizi_cloud/ (cieco o no, dichiarato)",
                     "set_v3_qa_pass": "transizioni finali di runs/dj_set_v3_bridge con QA pass salvato (dj_qa.4) e nessun veto",
                     "set_v3_lead_in": "stesse transizioni rifatte col runner corrente (B_ENTRY on): tenute se c'è un lead-in di B "
                                       "(riferimenti per 'pickup tolto' e 'parola di B'). Emendamento dopo la prima run (n=2 < min_n): "
                                       "non si richiede più che QA/veto le accettino; un rifiuto del riferimento è riportato come falso "
                                       "allarme e conta solo un segnale nuovo rispetto a esso (regola appaiata, 'masked' se b_entry era già fail)",
                     "natural": "cut di un brano con se stesso (stesso campione: continuazione senza giunta) sul downbeat di un inizio "
                                "frase di dj_select interno a una sezione (frase >= 4 battute, non inizio sezione), nessuna parola (ASR "
                                "o raffinata di dj_phrase) in [d - 150 ms, d + 50 ms] e respiro >= 1/4 di battito attorno a d, >= 14 s "
                                "prima e >= 30 s dopo; brani con "
                                "griglia 'high'/'medium' e modello dj_phrase; scelta deterministica (sha256 dello slug)",
                     "natural_section": "come natural ma su un inizio sezione: la dinamica propria del brano (es. intro -> strofa) "
                                        "resta nel riferimento; gruppo separato, solo per i falsi allarmi",
                     "natural_blend": "blend 4 battute tra due inizi frase diversi (>= 8 battute di distanza) dello stesso brano: "
                                      "stessa tonalità e tempo, contenuto diverso (un blend del brano con se stesso nello stesso punto "
                                      "somma due segnali identici, +6 dB: escluso dopo la prima prova)",
                     "filter": "musicmix.track_filter (en/it, configs/excluded_tracks.json)"},
           "refs": refs, "skipped": skipped}
    dump(OUT / "refs.json", doc)
    log(f"refs: {len(refs)} riferimenti ({sum(1 for r in refs if r['origin']=='natural')} naturali), {len(skipped)} esclusi")
    return doc


def _vocals_json(cat: dict, slug: str) -> Path:
    from musicmix.paths import localize

    entry = next(t for t in cat["tracks"] if t["slug"] == slug)
    return localize(entry["vocals_path"])


def _natural_boundaries(an: dict, words: list) -> list[dict]:
    RT.init()
    beats, downs = RT.G1.regular_grid(an)
    period = float(an["tempo"]["period_s"])
    ws = [(w["start_sample"] / SR, w["end_sample"] / SR) for w in words]
    out = []
    seen = set()
    for s in an["phrases"]:  # inizi frase di dj_select (frase >= 4 battute o inizio sezione)
        if not (s.get("section_start") or int(s.get("phrase_level") or 0) >= 4):
            continue
        j = int(s["bar_index"])
        if j in seen or j <= 0 or j >= len(downs):
            continue
        seen.add(j)
        d = float(downs[j])
        if d < NATURAL_MIN_PRE_S + 2 or d + NATURAL_MIN_POST_S > float(an["duration_s"]):
            continue
        if any(e > d - NATURAL_GUARD_PRE_S and st < d + NATURAL_GUARD_POST_S for st, e in ws):
            continue
        prev_end = max((e for st, e in ws if e <= d), default=-1e9)
        next_start = min((st for st, e in ws if st >= d), default=1e9)
        if next_start - prev_end < NATURAL_REST_BEATS * period:
            continue
        out.append({"bar": j, "time_s": d, "section": s.get("section_label"), "phrase_level": s.get("phrase_level"),
                    "section_start": bool(s.get("section_start"))})
    return out


# ===================================================================================== recipes
def _set_ctx() -> dict:
    R = RT.R
    if not hasattr(RT, "set_ctx"):
        RT.set_ctx = {"doc": R.load(R.OUT / "set.json"), "cat": R.load(R.FROZEN)}
    return RT.set_ctx


def _set_plan(tr: dict, A, B) -> dict:
    pp = RT.djs.plan_pair(A.analysis, B.analysis, top_k=10 ** 9)
    hit = next((p for p in pp["plans"] if p["a_out_bar"] == tr["plan"]["a_out_bar"] and p["b_in_bar"] == tr["plan"]["b_in_bar"]), None)
    rel = tr.get("entry_relocation")
    if hit is None and rel:  # cue di B spostato dalla guardia d'ingresso (INGRESSO_B R-B2): non è tra i piani di plan_pair
        djs = RT.djs
        out = ({c["bar_index"]: c for c in djs.cue_candidates(A.analysis, "out")}.get(int(tr["plan"]["a_out_bar"]))
               or djs.bar_cue(A.analysis, int(tr["plan"]["a_out_bar"])))
        inn = djs.bar_cue(B.analysis, int(tr["plan"]["b_in_bar"]))
        hit = djs.plan_transition(A.analysis, B.analysis, out, inn, pp["compatibility"])
    if hit is None:
        raise StopIteration(f"cue {tr['plan']['a_out_bar']}/{tr['plan']['b_in_bar']} non tra i piani di plan_pair")
    if rel:  # il render lo dichiara (meta['b_entry']['relocated_bars']) come nel set
        hit = dict(hit, entry_relocation=rel)
    return hit


def _apply_cue_overrides(plan: dict, A, B, ov: dict) -> tuple[dict, Any, Any]:
    plan = dict(plan)
    B2 = B
    if ov.get("b_grid_offset_s"):
        off = float(ov["b_grid_offset_s"])
        B2 = shifted(B, off)
        plan["b_in_s"] = float(plan["b_in_s"]) + off
    if ov.get("a_shift"):
        ia = int(np.argmin(np.abs(A.downbeats - plan["a_out_s"])))
        j = ia + int(ov["a_shift"])
        if not 0 <= j < len(A.downbeats):
            raise ValueError("a_shift fuori dal brano")
        plan["a_out_s"] = float(A.downbeats[j])
    if ov.get("b_shift"):
        jb = int(np.argmin(np.abs(B2.downbeats - plan["b_in_s"])))
        j = jb + int(ov["b_shift"])
        if not 0 <= j < len(B2.downbeats):
            raise ValueError("b_shift fuori dal brano")
        plan["b_in_s"] = float(B2.downbeats[j])
    if ov.get("a_shift") or ov.get("b_shift") or ov.get("b_grid_offset_s"):
        plan.pop("phrase_switch", None)
        plan.pop("b_lead_in", None)
    return plan, A, B2


def render_ref(ref: dict, ov: Optional[dict] = None) -> tuple[Any, Optional[dict], Any, Any, dict, bool]:
    """Rende il riferimento (o una sua variante a monte, ``ov``) con la ricetta originale.

    Restituisce ``(res, decks_qa_o_None, A, B, dec, creative)``; ``decks`` None = render di dj_creative (deck e voci
    dal render, come ``run_dj_set_v3_bridge.qa_v3``)."""
    RT.init()
    R, dj = RT.R, RT.dj
    ov = dict(ov or {})
    rec = ref["recipe"]
    typ = rec["type"]
    if typ == "natural":
        cat = _set_ctx()["cat"]
        X = track(rec["slug"], R.PLAN, cat)
        plan = {"a_out_s": float(X.downbeats[rec["bar"]]), "b_in_s": float(X.downbeats[rec.get("bar_b", rec["bar"])])}
        if ov.get("b_grid_offset_beats"):
            ov["b_grid_offset_s"] = float(ov["b_grid_offset_beats"]) * float(X.analysis["tempo"]["period_s"])
        plan, A, B = _apply_cue_overrides(plan, X, X, ov)
        if rec["kind"] == "cut":
            dec = {"type": "cut", "type_key": "cut", "render": {}}
        else:
            dec = {"type": "blend", "type_key": f"blend{rec['bars']}", "render": {"bars": rec["bars"]}}
        prov = {"a": A.provenance, "b": B.provenance, "script": "tools/defect_gym.py", "natural": True}
        res, decks = R.v2.render(dec, A, B, plan, prov)
        return res, decks, A, B, dec, False
    if typ == "gallery_v1":
        G1 = RT.G1
        plan_dir = ROOT / "runs/dj_plan_v1"
        pairs = load(plan_dir / "pairs.json")
        from musicmix.paths import localize

        cpath = localize(pairs["catalog"])
        cat = load(cpath if cpath.is_absolute() else ROOT / cpath)
        A = track(ref["a"], plan_dir, cat, keep=(ref["a"], ref["b"]))
        B = track(ref["b"], plan_dir, cat, keep=(ref["a"], ref["b"]))
        plan = dict(rec["plan"])
        if ov.get("b_grid_offset_beats"):
            ov["b_grid_offset_s"] = float(ov["b_grid_offset_beats"]) * float(B.analysis["tempo"]["period_s"])
        plan, A2, B2 = _apply_cue_overrides(plan, A, B, ov)
        v = dict(rec["version"])
        prov = {"a": A.provenance, "b": B.provenance, "plan_run": "runs/dj_plan_v1", "script": "tools/run_dj_gallery_v1.py (defect_gym)"}
        saved = G1.ECHO_DELAY_BEATS
        if ov.get("echo_delay_beats") and v["kind"] == "echo_out":
            G1.ECHO_DELAY_BEATS = float(ov["echo_delay_beats"])
        try:
            res, params, extra = G1.render_version(v, A2, B2, plan, prov)
        finally:
            G1.ECHO_DELAY_BEATS = saved
        dec = {"type": v["kind"], "type_key": v["kind"], "render": {k: v.get(k) for k in ("bars", "sweep_bars")}}
        return res, extra["decks"], A2, B2, dec, False
    # set v3 (transizione del set o versione default della galleria)
    ctx = _set_ctx()
    tr = next(t for t in ctx["doc"]["transitions"] if t["index"] == rec["index"])
    R.load_phrase({tr["a"], tr["b"]})
    A = track(tr["a"], R.PLAN, ctx["cat"], keep=(tr["a"], tr["b"]))
    B = track(tr["b"], R.PLAN, ctx["cat"], keep=(tr["a"], tr["b"]))
    plan = _set_plan(tr, A, B)
    if ov.get("b_grid_offset_beats"):
        ov["b_grid_offset_s"] = float(ov["b_grid_offset_beats"]) * float(B.analysis["tempo"]["period_s"])
    prov = {"a": A.provenance, "b": B.provenance, "script": R.SCRIPT, "run": R.RUN, "defect_gym": True}
    saved_entry = dict(R.B_ENTRY)
    R.B_ENTRY["on"] = bool(rec.get("b_entry")) and not ov.get("no_b_lead_in")
    try:
        if typ == "set_v3":
            opt = next(o for o in tr["books_options_same_cue"] if o["type_key"] == tr["technique"])
            dec = copy.deepcopy(R.opt_decision(opt))
            plan2, A2, B2 = _apply_cue_overrides(plan, A, B, ov)
            if ov.get("echo_delay_beats") and dec["type"] == "echo_out":
                dec["render"]["delay_beats"] = float(ov["echo_delay_beats"])
            res, decks = R.render_v3(dec, A2, B2, plan2, prov)
            return res, decks, A2, B2, dec, bool(dec.get("creative"))
        if typ == "set_v3_gallery":
            info_a = R.info_v3(tr["a"], {"plan_dir": R.PLAN, "catalog": ctx["cat"], "catalog_path": R.FROZEN})
            pp = RT.djs.plan_pair(A.analysis, B.analysis, top_k=10 ** 9)
            d0 = R.default_decision(A.analysis, B.analysis, plan, pp["compatibility"], info_a)
            cands = [d0] + list(d0.get("fallbacks") or [])
            dec = next(d for d in cands if d.get("type_key", d["type"]) == rec["technique"])
            dec = copy.deepcopy({**dec, "type_key": dec.get("type_key", dec["type"])})
            plan2, A2, B2 = _apply_cue_overrides(plan, A, B, ov)
            if ov.get("echo_delay_beats") and dec["type"] == "echo_out":
                dec["render"]["delay_beats"] = float(ov["echo_delay_beats"])
            if rec["role"] == "default":
                res, decks = R.v2.render(dec, A2, B2, plan2, prov)
                return res, decks, A2, B2, dec, False
            res, decks = R.render_v3(dec, A2, B2, plan2, prov)
            return res, decks, A2, B2, dec, bool(dec.get("creative"))
    finally:
        R.B_ENTRY.clear()
        R.B_ENTRY.update(saved_entry)
    raise ValueError(f"ricetta sconosciuta: {typ}")


def make_bundle(res, decks, A, B, dec: dict, creative: bool) -> G.Bundle:
    """Ingressi di QA e critico come ``run_dj_set_v3_bridge.qa_v3``/``critic`` (parole di B raffinate di dj_phrase)."""
    R, cr, dj_qa = RT.R, RT.cr, RT.qa
    meta = copy.deepcopy(res.metadata)
    n = len(res.audio)
    sr = int(res.sample_rate)
    if B.slug not in R.MODELS and (R.PHRASE / f"{B.slug}.json").is_file():
        R.load_phrase({B.slug})
    wb_src = R.words_b_refined(B.slug)
    words_b = dj_qa.words_to_output(meta, wb_src, n, deck="b") if wb_src else None
    comps = {k: res.decks[k] for k in G.MIX_COMPONENTS if (res.decks or {}).get(k) is not None}
    grid = {"a": A.analysis["tempo"], "b": B.analysis["tempo"]}
    kind = dec["type"]
    gap = RT.R.declared_gap(dec) if dec.get("render") is not None else False
    off = _a_offset(meta)
    gain_a = 10 ** (float(meta["loudness"]["a_gain_db"]) / 20.0)

    def a_source(s0: int, s1: int, _A=A, _off=off, _g=gain_a) -> np.ndarray:
        x = G._stereo(_A.audio)
        lo, hi = s0 + _off, s1 + _off
        out = np.zeros((s1 - s0, 2))
        a0, a1 = max(0, lo), min(len(x), hi)
        if a1 > a0:
            out[a0 - lo:a1 - lo] = x[a0:a1]
        return out * _g

    beat_s = float(A.analysis["tempo"]["period_s"])
    if decks is None:
        qi = cr.qa_inputs(res)
        ga = 10 ** (meta["loudness"]["a_gain_db"] / 20.0)
        gb = 10 ** (meta["loudness"]["b_gain_db"] / 20.0)
        return G.Bundle(audio=np.asarray(res.audio, dtype=np.float64), sr=sr, metadata=meta, components=comps,
                        qa_decks=qi["decks"], critic_decks=res.decks, vocals_a=qi["vocals_a"], vocals_b=qi["vocals_b"],
                        vocal_thresholds=(A.vocal_threshold * ga, B.vocal_threshold * gb), vocals_heard=True,
                        words_a=cr.words_in_clip(meta, A.words, "a"), words_b=words_b, grid_support=grid, kind=kind,
                        declared_gap=gap, a_source=a_source, beat_s=beat_s, info={"creative": True})
    va, vb = dj_qa.stems_to_output(meta, n, A.vocals, B.vocals)
    words = dj_qa.words_to_output(meta, A.words, n)
    ph = meta.get("phrase") or {}
    if ph.get("switch_rel_samples"):
        sw = meta["switch"]
        cut_src = int(sw["fader_switch_a_source_sample"])
        tail = (ph.get("tail") or {})
        keep_to = int(sw["a_source_sample"]) + (int(tail["output_end_sample"]) - int(sw["output_sample"])) if tail else cut_src
        heard = [w for w in A.words if w["start_sample"] < max(cut_src, keep_to)]
        words = dj_qa.words_to_output(meta, heard, n)
    return G.Bundle(audio=np.asarray(res.audio, dtype=np.float64), sr=sr, metadata=meta, components=comps, qa_decks=decks,
                    critic_decks=res.decks, vocals_a=va, vocals_b=vb, vocal_thresholds=(A.vocal_threshold, B.vocal_threshold),
                    vocals_heard=False, words_a=words, words_b=words_b, grid_support=grid, kind=kind, declared_gap=gap,
                    a_source=a_source, beat_s=beat_s, info={"creative": False})


def _a_offset(meta: dict) -> int:
    """Campione di A sorgente = campione della clip + offset, per la riproduzione secca di A (prima segmento di a_map)."""
    amap = sorted((m for m in (meta.get("a_map") or []) if m.get("src_start") is not None and m.get("out_start") is not None),
                  key=lambda m: int(m["out_start"]))
    if amap:
        return int(amap[0]["src_start"]) - int(amap[0]["out_start"])
    sw = meta.get("switch") or {}
    if sw.get("a_source_sample") is not None and sw.get("output_sample") is not None:
        return int(sw["a_source_sample"]) - int(sw["output_sample"])
    ov = meta["overlap"]
    return int(ov["a_source_start_sample"]) - int(ov["output_start_sample"])


# ===================================================================================== evaluation
def evaluate(bundle: G.Bundle, ident: str) -> dict:
    R, dj_qa = RT.R, RT.qa
    qa = dj_qa.evaluate_render(bundle.audio, bundle.metadata, sample_rate=bundle.sr, decks=bundle.qa_decks,
                               vocals_a=bundle.vocals_a, vocals_b=bundle.vocals_b, vocal_thresholds=bundle.vocal_thresholds,
                               words_a=bundle.words_a, tracker=RT.get_tracker(), config=R.QA_CFG,
                               grid_support=bundle.grid_support, words_b=bundle.words_b)
    res_like = SimpleNamespace(audio=bundle.audio.astype(np.float32), metadata=bundle.metadata, sample_rate=bundle.sr,
                               decks=bundle.critic_decks)
    crt = R.critic(res_like, bundle.kind, ident, decks=bundle.critic_decks, qa=qa, declared_gap=bundle.declared_gap)
    if crt.get("error"):
        raise RuntimeError(f"critico: {crt['error']}")
    veto = G.runner_veto_flags(crt, bundle.kind, bundle.declared_gap, phase_declared_max_ms=R.PHASE_DECLARED_MAX_MS)
    if len(veto) != len(crt.get("veto") or []):
        raise RuntimeError(f"regole di veto disallineate col runner: {veto} vs {crt.get('veto')}")
    sig = {**G.qa_signals(qa), **G.critic_signals(crt, veto)}
    tech = ((crt.get("v4") or {}).get("technical") or {})
    return {"signals": sig, "coverage": G.qa_coverage(qa), "qa_version": qa.get("schema_version"), "details": qa_details(qa),
            "qa": {"verdict": qa["verdict"], "reasons": [f"{r['check']}: {r['message']}" for r in qa["reasons"]],
                   "missing_required": (qa.get("coverage") or {}).get("missing_required")},
            "critic": {"veto": veto, "flags": {f["flag"]: f["value"] for f in crt.get("flags") or []},
                       "declared_phase": crt.get("declared_phase"), "craft": crt.get("craft"),
                       "v4_status": tech.get("status"), "v4_veto": [f["flag"] for f in tech.get("veto_flags") or []],
                       "v4_warnings": [w.get("check") for w in tech.get("warnings") or []], "v4_error": crt.get("v4_error")}}


DETAIL_KEYS = {
    "phrase": ("status", "b_cue_bar", "off_bars", "bars_after_phrase_start", "bars_to_next_phrase_start", "phrase_table"),
    "dropout": ("status", "longest_s", "longest_beats", "declared_gap_s", "excess_beats", "at_s"),
    "click": ("status", "statistic", "worst_point", "threshold"),
    "echo_tempo": ("status", "statistic", "spacing_beats", "threshold"),
    "double_bass": ("status", "longest_s", "method"),
    "verse_cut": ("status", "coverage"),
    "b_entry": ("status", "coverage", "problem"),
}


def qa_details(qa: dict) -> dict:
    """Misure compatte del QA per il report (dal .7: offset di fase all'ingresso, frase, buco, clic, eco)."""
    ch = qa.get("checks") or {}
    out = {}
    rh = ch.get("rhythm") or {}
    en = (rh.get("methods") or {}).get("beat_this_entry")
    out["rhythm"] = {"status": rh.get("status"), "overlap": rh.get("rhythmic_overlap")}
    if en:
        out["rhythm"].update(entry_status=en.get("status"), entry_offset_ms=en.get("beat_phase_offset_at_switch_ms"),
                             entry_problems=en.get("reliability_problems"))
    for name, keys in DETAIL_KEYS.items():
        if name in ch:
            out[name] = {k: ch[name].get(k) for k in keys if k in ch[name]}
    return out


def pcm_sha(res) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        return RT.dj.write_render(res, tmp, "x")["sha256"]


def trimmed_sha(res, kind: str) -> str:
    R = RT.R
    mk = R.clip_markers(res)
    clip, _ = R.trim_to_rule(res.audio, kind, mk["t0"], mk["sw"], mk["t1"])
    with tempfile.TemporaryDirectory() as tmp:
        return R.write_clip(clip, Path(tmp) / "x.wav")


# ===================================================================================== variants
def _a_point(bundle: G.Bundle) -> int:
    """Uscita secca di A nella clip: effetto/a_map dichiarati, altrimenti lo switch."""
    d = RT.qa.a_dry_end(bundle.metadata)
    return int(d["sample"]) if d else bundle.switch


def _b_point(bundle: G.Bundle) -> tuple[int, int]:
    """Punto in cui B diventa udibile (clip, sorgente di B)."""
    m = bundle.metadata
    be = m.get("b_entry") or {}
    if be.get("audible_output_sample") is not None and be.get("audible_source_sample") is not None:
        return int(be["audible_output_sample"]), int(be["audible_source_sample"])
    sw = m["switch"]
    return int(sw["output_sample"]), int(sw["b_source_sample"])


def rerender_overrides(ref: dict, refb: G.Bundle, A, B, defect: str, level: str) -> tuple[Optional[dict], dict]:
    """Ingressi alterati del render per un difetto a monte; (None, motivo) se non applicabile."""
    p = G.DEFECTS[defect]["levels"][level]
    kind = refb.kind
    overlap = G.has_overlap(refb)
    if defect == "word_cut_a":
        if overlap:
            return None, {"reason": "uscita a fader (blend): il taglio di parola non si applica"}
        pt = _a_point(refb)
        pt_src = pt + _a_offset(refb.metadata)
        plan_cue = _plan_a_cue(ref, refb)
        ia = int(np.argmin(np.abs(A.downbeats - plan_cue)))
        s = G.find_straddle_shift(A.words, A.downbeats, ia, pt_src / SR, min_side_s=p["min_side_s"], sr=SR)
        if not s:
            return None, {"reason": "nessuna parola di A attraversa l'uscita entro ±32 battute"}
        return {"a_shift": s["k"]}, {"straddle": s}
    if defect == "word_cut_b":
        if overlap:
            return None, {"reason": "ingresso a fader (blend): non si applica"}
        R = RT.R
        R.load_phrase({B.slug}) if (R.PHRASE / f"{B.slug}.json").is_file() else None
        wb = R.words_b_refined(B.slug)
        if not wb:
            return None, {"reason": "parole raffinate di B non disponibili"}
        _, src = _b_point(refb)
        jb = int(np.argmin(np.abs(B.downbeats - _plan_b_cue(ref, refb))))
        s = G.find_straddle_shift(wb, B.downbeats, jb, src / SR, min_side_s=p["min_side_s"], sr=SR)
        if not s:
            return None, {"reason": "nessuna parola di B attraversa l'ingresso entro ±32 battute"}
        return {"b_shift": s["k"], "no_b_lead_in": True}, {"straddle": s, "note": "lead-in di B spento (altrimenti la guardia lo ripara)"}
    if defect == "pickup_removed":
        be = refb.metadata.get("b_entry") or {}
        if not be.get("lead_in_samples"):
            return None, {"reason": "il riferimento non dichiara un lead-in di B"}
        return {"no_b_lead_in": True}, {"lead_in_s": be.get("lead_in_s"), "lead_in_words": (be.get("info") or {}).get("lead_in_words")}
    if defect == "phase":
        if "b_grid_offset_beats" in p:
            return {"b_grid_offset_beats": p["b_grid_offset_beats"]}, {}
        return {"b_grid_offset_s": p["b_grid_offset_s"]}, {}
    if defect in ("b_bar_late", "b_bar_early"):
        return {"b_shift": p["b_shift"]}, {}
    if defect == "echo_offtempo":
        return {"echo_delay_beats": p["delay_beats"]}, {}
    raise ValueError(defect)


def _plan_a_cue(ref: dict, refb: G.Bundle) -> float:
    """Un downbeat di A solidale col cue (gli spostamenti sono differenze di indici sulla stessa griglia regolare)."""
    m = refb.metadata
    if ref["recipe"].get("plan", {}).get("a_out_s") is not None:
        return float(ref["recipe"]["plan"]["a_out_s"])
    if m.get("cue_out_a_s") is not None:
        return float(m["cue_out_a_s"])
    return float(m["switch"]["a_source_sample"]) / SR


def _plan_b_cue(ref: dict, refb: G.Bundle) -> float:
    m = refb.metadata
    if m.get("cue_in_b_s") is not None:
        return float(m["cue_in_b_s"])
    if ref["recipe"].get("plan", {}).get("b_in_s") is not None:
        return float(ref["recipe"]["plan"]["b_in_s"])
    return float(m["switch"]["b_source_sample"]) / SR


def is_rerender(defect: str, refb: G.Bundle) -> bool:
    m = G.DEFECTS[defect]["method"]
    if m == "rerender":
        return True
    if m == "deck_or_rerender":
        return refb.kind == "echo_out"
    return False


def build_variant(ref: dict, refb: G.Bundle, A, B, parts: list) -> tuple[Optional[G.Bundle], dict]:
    """Variante (bundle) per ``parts`` = [(difetto, livello), ...]; (None, info) se non generabile."""
    info: dict = {"parts": []}
    re_parts = [p for p in parts if is_rerender(p[0], refb)]
    deck_parts = [p for p in parts if not is_rerender(p[0], refb)]
    if len(re_parts) > 1:
        return None, {"reason": "più di un difetto a monte nella stessa variante"}
    if re_parts:
        d, lv = re_parts[0]
        ov, extra = rerender_overrides(ref, refb, A, B, d, lv)
        if ov is None:
            return None, {"reason": f"{d}: {extra['reason']}"}
        try:
            res, decks, A2, B2, dec, creative = render_ref(ref, ov)
        except Exception as exc:  # noqa: BLE001 - il motivo resta nel risultato
            return None, {"reason": f"{d}: render: {type(exc).__name__}: {exc}"}
        vb = make_bundle(res, decks, A2, B2, dec, creative)
        if d == "pickup_removed" and (vb.metadata.get("b_entry") or {}).get("lead_in_samples"):
            return None, {"reason": "pickup_removed: il render senza lead-in ne ha ancora uno"}
        info["parts"].append({"defect": d, "level": lv, "method": "rerender", "overrides": ov, **extra})
    else:
        vb = refb.copy()
    for d, lv in deck_parts:
        try:
            op = G.apply_deck_defect(vb, d, lv)
        except Exception as exc:  # noqa: BLE001
            return None, {"reason": f"{d}: {type(exc).__name__}: {exc}"}
        if not op.get("applied"):
            return None, {"reason": f"{d}: {op.get('reason')}"}
        info["parts"].append({"defect": d, "level": lv, "method": "deck", **op})
    return vb, info


# ===================================================================================== run
def run(args) -> None:
    RT.init()
    refs_doc = load(OUT / "refs.json")
    refs = [r for r in refs_doc["refs"] if not args.refs or any(r["id"].startswith(p) for p in args.refs)]
    if getattr(args, "max_per_origin", None):
        kept, cnt = [], {}
        for r in refs:
            cap = args.max_per_origin.get(r["origin"])
            if cap is not None and cnt.get(r["origin"], 0) >= cap:
                continue
            cnt[r["origin"]] = cnt.get(r["origin"], 0) + 1
            kept.append(r)
        refs = kept
    res_path = OUT / (args.results or "results.jsonl")
    done = set()
    ref_rows = {}
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            row = json.loads(line)
            if row.get("type") == "reference":
                ref_rows[row["ref"]] = row
            else:
                done.add((row["ref"], row["variant"]))
    fh = open(res_path, "a", encoding="utf-8")
    levels = None
    if args.levels:
        levels = {}
        for x in args.levels:
            d, lv = x.split("@")
            levels.setdefault(d, []).append(lv)
    t_all = time.time()
    for ref in refs:
        pol = ORIGIN_POLICY.get(ref["origin"], {})
        defects = args.defects if pol.get("defects") is None else [d for d in pol["defects"] if not args.defects or d in args.defects]
        plan = G.variant_plan(ref, defects=defects, combos=(not args.no_combos) and pol.get("combos", False), levels=levels)
        todo = [p for p in plan if (ref["id"], G.variant_id(p)) not in done]
        if ref["id"] in ref_rows and not todo:
            continue
        log(f"== {ref['id']} ({ref['origin']}, {ref['kind']}): {len(todo)} varianti")
        t0 = time.time()
        try:
            res, decks, A, B, dec, creative = render_ref(ref)
            refb = make_bundle(res, decks, A, B, dec, creative)
        except Exception as exc:  # noqa: BLE001
            row = {"type": "reference", "ref": ref["id"], "evaluation_error": f"render: {type(exc).__name__}: {exc}",
                   "trace": traceback.format_exc()[-1500:]}
            fh.write(json.dumps(G.jsonable(row), ensure_ascii=False) + "\n")
            fh.flush()
            log("   ERRORE render riferimento:", exc)
            continue
        if ref["id"] not in ref_rows:
            row = {"type": "reference", "ref": ref["id"], "origin": ref["origin"], "kind": refb.kind, "clean": ref.get("clean", False)}
            rec = ref["recipe"]
            try:
                if rec.get("expected_wav_sha256") and rec["type"] in ("gallery_v1", "set_v3"):
                    h = pcm_sha(res) if rec["type"] == "set_v3" else _g1_sha(res, ref)
                    row["identity"] = {"expected": rec["expected_wav_sha256"], "got": h, "identical": h == rec["expected_wav_sha256"]}
                elif rec.get("expected_trimmed_sha256"):
                    h = trimmed_sha(res, dec["type"])
                    row["identity"] = {"expected": rec["expected_trimmed_sha256"], "got": h, "identical": h == rec["expected_trimmed_sha256"],
                                       "note": "clip della galleria (tagliata secondo AGENTS.md)"}
            except Exception as exc:  # noqa: BLE001
                row["identity"] = {"error": f"{type(exc).__name__}: {exc}"}
            row["rebuild_error"] = G.rebuild_error(refb)
            row["b_lead_in_s"] = (refb.metadata.get("b_entry") or {}).get("lead_in_s")
            try:
                ev = evaluate(refb, f"gym_{ref['id']}")
                row.update(ev)
                row["reference_flags"] = G.reference_flags(ev["signals"])
            except Exception as exc:  # noqa: BLE001
                row["evaluation_error"] = f"{type(exc).__name__}: {exc}"
                row["trace"] = traceback.format_exc()[-1500:]
            if rec.get("require_lead_in"):
                acc = not row.get("evaluation_error") and not row.get("reference_flags", {}).get("pipeline")
                row["candidate_accepted"] = bool(row["b_lead_in_s"]) and acc
            fh.write(json.dumps(G.jsonable(row), ensure_ascii=False) + "\n")
            fh.flush()
            ref_rows[ref["id"]] = row
            log(f"   rif: QA {row.get('qa', {}).get('verdict')} veto {row.get('critic', {}).get('veto')} "
                f"v4 {row.get('critic', {}).get('v4_status')} id={row.get('identity', {}).get('identical')} "
                f"rebuild={row['rebuild_error']:.1e} [{time.time() - t0:.0f} s]")
        rrow = ref_rows[ref["id"]]
        if rrow.get("evaluation_error") or (ref["recipe"].get("require_lead_in") and not rrow.get("b_lead_in_s")):
            continue
        ref_sig = rrow["signals"]
        if rrow.get("rebuild_error", 0) > 1e-4:
            log(f"   attenzione: i deck non ricostruiscono la clip (errore {rrow['rebuild_error']:.2e}); varianti a deck sul mix ricostruito")
        for parts in todo:
            vid = G.variant_id(parts)
            t1 = time.time()
            vb, info = build_variant(ref, refb, A, B, parts)
            exp = sorted({e for d, _ in parts for e in G.DEFECTS[d]["expect"]})
            row = {"type": "variant", "ref": ref["id"], "origin": ref["origin"], "variant": vid, "defects": [list(p) for p in parts],
                   "expected": exp}
            if vb is None:
                row.update(status="not_generated", reason=info.get("reason"))
            else:
                row["op"] = info
                try:
                    ev = evaluate(vb, f"gym_{ref['id']}_{vid}")
                    row.update(status="evaluated", **ev)
                    row["classification"] = classify_masked(ev, ref_sig, exp)
                except Exception as exc:  # noqa: BLE001
                    row.update(status="evaluated", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-1500:])
                    row["classification"] = G.classify({}, ref_sig, exp, error=row["error"])
                if args.keep_wav:
                    _keep_wav(vb, ref["id"], vid)
            row["seconds"] = time.time() - t1
            fh.write(json.dumps(G.jsonable(row), ensure_ascii=False) + "\n")
            fh.flush()
            st = row.get("classification", {}).get("status", row.get("status"))
            log(f"   {vid:42s} {st:12s} {row.get('classification', {}).get('new_fail', row.get('reason', ''))} [{row['seconds']:.0f} s]")
            del vb
        log(f"   fatto in {time.time() - t0:.0f} s (totale {time.time() - t_all:.0f} s)")
    fh.close()


def classify_masked(ev: dict, ref_sig: dict, expected: list) -> dict:
    """Come :func:`musicmix.defect_gym.classify`, con ``masked``: un controllo atteso già ``fail`` nel riferimento."""
    c = G.classify(ev["signals"], ref_sig, expected, ev.get("coverage"))
    masked = [e for e in expected if ref_sig.get(e) == "fail"]
    if c["status"] != "detected" and masked:
        c["status"] = "masked"
        c["masked_by"] = masked
    return c


def _g1_sha(res, ref) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        out = RT.dj.write_render(res, tmp, "x")
        return sha_file(Path(out["wav"]))


def _keep_wav(vb: G.Bundle, ref_id: str, vid: str) -> None:
    import soundfile as sf

    d = OUT / "clips"
    d.mkdir(parents=True, exist_ok=True)
    m = vb.metadata
    sr = vb.sr
    sw = vb.switch
    s0, s1 = max(0, sw - 10 * sr), min(vb.n, sw + 10 * sr)
    safe = vid.replace("@", "_").replace("+", "__").replace("/", "_")
    sf.write(str(d / f"{ref_id}__{safe}.wav"), vb.audio[s0:s1].astype(np.float32), sr, subtype="PCM_16")


# ===================================================================================== report
def report(args) -> dict:
    refs_doc = load(OUT / "refs.json")
    res_path = OUT / (args.results or "results.jsonl")
    rows = [json.loads(x) for x in res_path.read_text().splitlines() if x.strip()]
    ref_rows = {}
    for r in rows:
        if r.get("type") == "reference":
            ref_rows[r["ref"]] = r
    var_rows = [r for r in rows if r.get("type") == "variant"]
    # ultima riga per (ref, variante)
    last = {}
    for r in var_rows:
        last[(r["ref"], r["variant"])] = r
    var_rows = list(last.values())
    refs_eval = []
    for ref in refs_doc["refs"]:
        rr = ref_rows.get(ref["id"])
        if rr is None:
            continue
        if ref["recipe"].get("require_lead_in") and not rr.get("b_lead_in_s"):
            continue
        refs_eval.append({**ref, **{k: rr.get(k) for k in ("signals", "reference_flags", "evaluation_error", "identity", "qa", "critic",
                                                          "rebuild_error", "b_lead_in_s")}})
    used = {r["id"] for r in refs_eval}
    var_rows = [r for r in var_rows if r["ref"] in used]
    for r in var_rows:  # masked: escluso dal denominatore, dichiarato
        pass
    summ = aggregate_masked(var_rows, refs_eval)
    split = load(OUT / "split.json") if (OUT / "split.json").exists() else None
    if split:  # falsi allarmi per metà dichiarata (sviluppo / verifica), dal v2
        for r in refs_eval:
            r["half"] = split["refs"].get(r["id"], "?")
        for half in ("dev", "verify"):
            sub = [dict(r, origin=f"{r['origin']}@{half}", clean=False) for r in refs_eval if r.get("half") == half]
            fa = G.aggregate([], sub)["false_alarms"]
            fa.pop("all", None)
            for k, v in fa.items():
                summ["false_alarms"][k] = v
            allh = G.aggregate([], [dict(r, origin=f"all@{half}") for r in refs_eval if r.get("half") == half])["false_alarms"]
            summ["false_alarms"][f"all@{half}"] = allh[f"all@{half}"]
        # gate per metà: rilevamento sulle varianti dei riferimenti di quella metà
        summ["gate_by_half"] = {}
        for half in ("dev", "verify"):
            vh = [r for r in var_rows if split["refs"].get(r["ref"]) == half]
            rh = [r for r in refs_eval if r.get("half") == half]
            sh = aggregate_masked(vh, rh)
            gh = G.check_gate(sh, {**G.load_gate(GATE), "false_alarms": [dict(fr, min_n=1) for fr in G.load_gate(GATE)["false_alarms"]]})
            summ["gate_by_half"][half] = {"passed": gh["passed"], "rules": [{"rule": x["rule"].get("variant") or x["rule"].get("group"),
                                                                              "n": x.get("n"), "rate": x.get("rate"), "status": x["status"]}
                                                                             for x in gh["rules"]]}
    gate = G.load_gate(GATE)
    g = G.check_gate(summ, gate)
    weak = weak_points(summ)
    rep = {"schema": "musicmix.defect_gym.report.1", "version": G.GYM_VERSION,
           "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "refs": {"n": len(refs_eval), "by_origin": _count(r["origin"] for r in refs_eval),
                    "rows": [{k: r.get(k) for k in ("id", "origin", "kind", "a", "b", "judgement", "identity", "rebuild_error", "b_lead_in_s")}
                             | {"qa": (r.get("qa") or {}).get("verdict"), "qa_reasons": (r.get("qa") or {}).get("reasons"),
                                "critic_veto": (r.get("critic") or {}).get("veto"), "v4": (r.get("critic") or {}).get("v4_status"),
                                "flagged": (r.get("reference_flags") or {}).get("pipeline")} for r in refs_eval]},
           "n_variant_rows": len(var_rows), "summary": summ, "gate": g, "gate_config": gate, "weak_points": weak,
           "qa_versions": _count(r.get("qa_version") for r in var_rows if r.get("status") == "evaluated"),
           "split": None if not split else {"rule": split["rule"], "counts": split["counts"], "declared": split["declared"]},
           "baseline": _baseline(args, summ),
           "detection_rule": "rilevata = segnale di rifiuto nuovo rispetto al riferimento (QA fail o veto del critico del runner); "
                             "warned = solo avvisi nuovi; unavailable = nessun segnale e controllo atteso non misurato/inesistente; "
                             "masked = controllo atteso già fail nel riferimento (escluso dal denominatore, dichiarato); "
                             "error = valutazione fallita (nel denominatore, mai pass); not_generated = variante non costruibile",
           "limits": ["riconoscere un peggioramento non equivale a riconoscere una bella transizione",
                      "i difetti sono quelli simulati; il gate non vede difetti non elencati",
                      "riferimenti naturali = un brano con se stesso: puliti per costruzione, non transizioni tra brani"]}
    dump(OUT / "report.json", rep)
    (OUT / "REPORT.md").write_text(render_md(rep), encoding="utf-8")
    log(f"report: gate {'PASS' if g['passed'] else 'FAIL'}; {len(var_rows)} varianti, {len(refs_eval)} riferimenti")
    return rep


def _baseline(args, summ: dict) -> Optional[dict]:
    """Confronto prima/dopo con una run precedente (``--baseline``, es. runs/defect_gym_v1/report.json)."""
    bp = getattr(args, "baseline", None)
    if not bp or not Path(bp).exists():
        return None
    b = load(Path(bp))["summary"]
    rows = {}
    for vid, e in summ["variants"].items():
        o = (b.get("variants") or {}).get(vid) or {}
        rows[vid] = {"before": o.get("detection_rate"), "before_n": o.get("n"), "after": e.get("detection_rate"), "after_n": e.get("n"),
                     "before_warn_or_fail": o.get("warn_or_fail_rate"), "after_warn_or_fail": e.get("warn_or_fail_rate")}
    fa = {}
    for gname, f in summ["false_alarms"].items():
        if "@" in gname:
            continue
        o = (b.get("false_alarms") or {}).get(gname) or {}
        fa[gname] = {"before": (o.get("rates") or {}).get("pipeline"), "after": f["rates"]["pipeline"], "n": f["n"],
                     "before_flagged": [x["ref"] for x in o.get("flagged", [])], "after_flagged": [x["ref"] for x in f.get("flagged", [])]}
    return {"path": str(bp), "variants": rows, "false_alarms": fa}


def aggregate_masked(var_rows: list, refs_eval: list) -> dict:
    keep = [r for r in var_rows if (r.get("classification") or {}).get("status") != "masked"]
    summ = G.aggregate(keep, refs_eval)
    masked: dict = {}
    for r in var_rows:
        if (r.get("classification") or {}).get("status") == "masked":
            masked.setdefault(r["variant"], []).append({"ref": r["ref"], "masked_by": r["classification"].get("masked_by")})
    for k, v in masked.items():
        summ["variants"].setdefault(k, G._empty({"defects": None}))["masked"] = v
    # per difetto (tutti i livelli insieme) e per famiglia
    per_def: dict = {}
    for r in keep:
        if r.get("status") != "evaluated" or len(r["defects"]) != 1:
            continue
        d = r["defects"][0][0]
        e = per_def.setdefault(d, {"n": 0, "detected": 0, "warned": 0, "unavailable": 0, "missed": 0, "error": 0})
        e["n"] += 1
        e[r["classification"]["status"]] += 1
    for e in per_def.values():
        e["detection_rate"] = e["detected"] / e["n"] if e["n"] else None
    summ["per_defect"] = per_def
    return summ


def weak_points(summ: dict) -> list:
    out = []
    for vid, e in summ["variants"].items():
        if not e.get("n"):
            continue
        rate = e.get("detection_rate")
        if rate is None or rate < 0.9:
            out.append({"variant": vid, "n": e["n"], "detection_rate": rate, "warn_or_fail_rate": e.get("warn_or_fail_rate"),
                        "unavailable": e.get("unavailable"), "missed": e.get("missed"), "examples": e.get("examples_missed")})
    out.sort(key=lambda x: (x["detection_rate"] if x["detection_rate"] is not None else -1))
    return out


def _count(it) -> dict:
    c: dict = {}
    for v in it:
        c[str(v)] = c.get(str(v), 0) + 1
    return c


def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.0f}%"


def render_md(rep: dict) -> str:
    s = rep["summary"]
    L = [f"# Palestra dei difetti {OUT.name.replace('defect_gym_', '')} — report", "",
         f"Generato {rep['created_at']} da `tools/defect_gym.py report` ({rep['version']}). Riferimenti: {rep['refs']['n']} "
         f"({', '.join(f'{k} {v}' for k, v in rep['refs']['by_origin'].items())}); righe variante: {rep['n_variant_rows']}.", "",
         "Regola: " + rep["detection_rule"] + ".", "",
         f"## Gate: **{'PASS' if rep['gate']['passed'] else 'FAIL'}** (`configs/defect_gym_gate.json`, soglie [H])", "",
         "| regola | n | valore | soglia | esito |", "|---|---|---|---|---|"]
    for r in rep["gate"]["rules"]:
        rule = r["rule"]
        name = rule.get("variant") or f"falsi allarmi {rule.get('group')}"
        thr = f">= {rule['min_rate']}" if "min_rate" in rule else f"<= {rule['max_rate']}"
        L.append(f"| {name} ({rule.get('detector', 'pipeline')}) | {r.get('n')} | {_pct(r.get('rate'))} | {thr} | {r['status']} |")
    if rep.get("qa_versions"):
        L += ["", f"Versioni del QA nelle righe valutate: {rep['qa_versions']}."]
    bl = rep.get("baseline")
    if bl:
        L += ["", f"## Prima / dopo (baseline `{bl['path']}`)", "", "| variante | prima | dopo | warn+ prima | warn+ dopo |", "|---|---|---|---|---|"]
        for vid, e in bl["variants"].items():
            L.append(f"| `{vid}` | {_pct(e['before'])} (n {e['before_n']}) | {_pct(e['after'])} (n {e['after_n']}) | "
                     f"{_pct(e['before_warn_or_fail'])} | {_pct(e['after_warn_or_fail'])} |")
        L += ["", "| falsi allarmi | n | prima | dopo | segnalati dopo |", "|---|---|---|---|---|"]
        for gname, e in bl["false_alarms"].items():
            L.append(f"| {gname} | {e['n']} | {_pct(e['before'])} | {_pct(e['after'])} | {', '.join(e['after_flagged']) or '—'} |")
    gb = s.get("gate_by_half")
    if gb:
        sp = rep.get("split") or {}
        L += ["", "## Metà di sviluppo e di verifica", "", f"Regola (dichiarata {sp.get('declared')}): {sp.get('rule')}", "",
              "| regola | sviluppo n | sviluppo | verifica n | verifica |", "|---|---|---|---|---|"]
        for rd, rv in zip(gb["dev"]["rules"], gb["verify"]["rules"]):
            L.append(f"| {rd['rule']} | {rd['n']} | {_pct(rd.get('rate'))} ({rd['status']}) | {rv['n']} | {_pct(rv.get('rate'))} ({rv['status']}) |")
        L.append(f"| **gate (min_n dei falsi allarmi = 1)** | | {'PASS' if gb['dev']['passed'] else 'FAIL'} | | {'PASS' if gb['verify']['passed'] else 'FAIL'} |")
    L += ["", "## Rilevamento per variante", "",
          "pipeline = QA fail o veto del critico (come il runner del set); qa / critic / v4 = singoli rivelatori; "
          "warn+ = anche solo avvisi nuovi.", "",
          "| variante | n | rilevate | IC95 | QA | critico | v4 | warn+ | non misurabili | mancate | errori | mascherate | non generate |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for vid, e in s["variants"].items():
        ci = e.get("detection_ci95")
        rb = e.get("rates_by_detector") or {}
        L.append(f"| `{vid}` | {e['n']} | {_pct(e.get('detection_rate'))} | {'' if not ci else f'{ci[0]:.2f}–{ci[1]:.2f}'} | "
                 f"{_pct(rb.get('qa'))} | {_pct(rb.get('critic'))} | {_pct(rb.get('v4'))} | {_pct(e.get('warn_or_fail_rate'))} | "
                 f"{e.get('unavailable', 0)} | {e.get('missed', 0)} | {e.get('error', 0)} | {len(e.get('masked') or [])} | {e.get('not_generated', 0)} |")
    L += ["", "## Falsi allarmi sui riferimenti", "", "| gruppo | n | pipeline | QA | critico | v4 | riferimenti segnalati |", "|---|---|---|---|---|---|---|"]
    for gname, f in s["false_alarms"].items():
        flagged = "; ".join(f"{x['ref']} ({', '.join(x['fail_signals'])})" for x in f["flagged"][:6])
        L.append(f"| {gname} | {f['n']} | {_pct(f['rates']['pipeline'])} | {_pct(f['rates']['qa'])} | {_pct(f['rates']['critic'])} | "
                 f"{_pct(f['rates']['v4'])} | {flagged} |")
    L += ["", "## Punti deboli (rilevamento < 90%)", ""]
    for w in rep["weak_points"]:
        ex = "; ".join(f"{x['ref']}: {x['status']}" + (f" (avvisi {', '.join(x['new_warn'])})" if x.get("new_warn") else "")
                       for x in (w.get("examples") or [])[:3])
        L.append(f"- `{w['variant']}`: {_pct(w['detection_rate'])} su {w['n']} (warn+ {_pct(w['warn_or_fail_rate'])}, "
                 f"non misurabili {w['unavailable']}, mancate {w['missed']}). Esempi: {ex}")
    L += ["", "## Riferimenti", "", "| id | origine | tipo | voto | identità PCM | QA | veto | v4 |", "|---|---|---|---|---|---|---|---|"]
    for r in rep["refs"]["rows"]:
        j = r.get("judgement") or {}
        idt = r.get("identity") or {}
        L.append(f"| {r['id']} | {r['origin']} | {r['kind']} | {j.get('score', '')} | {idt.get('identical', '—')} | {r.get('qa')} | "
                 f"{', '.join(r.get('critic_veto') or []) or '—'} | {r.get('v4')} |")
    L += ["", "## Limiti", ""] + [f"- {x}" for x in rep["limits"]] + [""]
    return "\n".join(L)


# ===================================================================================== main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stages", nargs="+", choices=["refs", "run", "report"])
    ap.add_argument("--refs", nargs="*", help="prefissi degli id dei riferimenti")
    ap.add_argument("--defects", nargs="*", help="solo questi difetti")
    ap.add_argument("--levels", nargs="*", help="solo questi livelli, es. phase@half_beat")
    ap.add_argument("--no-combos", action="store_true")
    ap.add_argument("--results", help="file dei risultati in runs/defect_gym_v1 (default results.jsonl)")
    ap.add_argument("--keep-wav", action="store_true", help="salva le clip di 20 s delle varianti in runs/defect_gym_v1/clips")
    ap.add_argument("--out", help="cartella della run (default runs/defect_gym_v1)")
    ap.add_argument("--baseline", help="report.json di una run precedente per la tabella prima/dopo")
    ap.add_argument("--profile", choices=["full", "gate"], default="full",
                    help="gate: solo i difetti con soglia in configs/defect_gym_gate.json, niente combinazioni (test -m gym)")
    args = ap.parse_args(argv)
    global OUT
    if args.out:
        OUT = Path(args.out)
    args.max_per_origin = None
    if args.profile == "gate":
        gate = G.load_gate(GATE)
        prof = gate.get("gym_profile") or {}
        args.refs = args.refs or prof.get("ref_prefixes")
        args.levels = args.levels or [r["variant"] for r in gate["detection"]]
        args.defects = args.defects or sorted({r["variant"].split("@")[0] for r in gate["detection"]})
        args.no_combos = True
        args.max_per_origin = prof.get("max_refs_per_origin")
    for st in args.stages:
        if st == "refs":
            build_refs()
        elif st == "run":
            run(args)
        elif st == "report":
            rep = report(args)
            if "run" not in args.stages and not rep["gate"]["passed"]:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
