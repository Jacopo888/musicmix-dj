#!/usr/bin/env python3
"""Ingresso di B: diagnosi della galleria cieca del set v3 e verifica della correzione (21/10/2026).

Nota: ``conoscenza_mix/_meta/INGRESSO_B_2026-10-21.md``. Legge (senza modificarli) ``runs/dj_set_v3_bridge`` (set.json,
gallery/blind_key.json, gallery/review_manifest.json, catalogo congelato, analisi, modelli dj_phrase),
``giudizi_cloud/dj_set_v3_bridge/judgments`` e ``runs/dj_set_v3_events_plan/set.json``. Per ciascuna delle 12 transizioni
della galleria e per ciascuna versione votata (profilo default / libri):

1. **vecchio**: rifà il render con la stessa decisione, lo stesso cue e gli stessi ingressi della galleria
   (``run_dj_set_v3_bridge`` con ``B_ENTRY['on'] = False``), verifica che la clip tagliata sia identica al WAV votato
   (sha256), misura dove B diventa udibile, le parole di B lì attorno (proiettate con la time-map del render,
   ``dj_qa.words_to_output(deck="b")``, parole raffinate di ``dj_phrase`` e parole ASR grezze), lo stato del verso di B
   (``dj_phrase.entry_point`` e ``dj_phrase.b_lead_in``), la sezione allin1 e il QA ``dj_qa.6`` (controllo ``b_entry``);
   → ``runs/b_entry_diagnosis/diagnosis.json``;
2. **nuovo**: stessa coppia, stessa uscita di A e stessa tecnica; cue di B da ``dj_select.relocate_entry`` con
   ``dj_phrase.entry_clear_test`` (lo stesso cue se l'ingresso è pulito, eventualmente con lead-in; 1-2 battute prima;
   un altro cue di B), render con il lead-in di B (``dj_render.add_b_lead_in``, modo stem), QA ``dj_qa.6`` e critico v4;
   clip secondo AGENTS.md in mp3 192 kbps → ``runs/b_entry_fix_v1/``.

Riprendibile (un file per transizione). Un render pesante alla volta; nessun WAV intermedio tenuto.

    python tools/run_b_entry_fix_v1.py run [--only 3 9] && \\
    python tools/run_b_entry_fix_v1.py summary
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import run_dj_set_v3_bridge as R  # noqa: E402

from musicmix import dj_critic_books, dj_qa, dj_render as dj, dj_select as djs, dj_set as ds  # noqa: E402
from musicmix import dj_phrase as P  # noqa: E402

v2 = R.v2
SR = R.SR
GAL = R.OUT / "gallery"
JUDG = ROOT / "giudizi_cloud" / "dj_set_v3_bridge" / "judgments"
EVENTS = ROOT / "runs" / "dj_set_v3_events_plan" / "set.json"
DIAG = ROOT / "runs" / "b_entry_diagnosis"
OUT = ROOT / "runs" / "b_entry_fix_v1"
SEED = 20261022
CHECK_T01 = ROOT / "runs" / "books_integration_check" / "clips" / "t01_halving_build.json"
CHECK_T01_MODEL = ROOT / "runs" / "dj_phrase_v1" / "tracks" / "coehn_foehrb_alles_coehn_und_gut_j111121.json"
EPS = dj_qa.B_ENTRY_EPS_S


def load(p: Path) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def dump(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dj_qa._jsonable(obj), indent=1, ensure_ascii=False, default=str), encoding="utf-8")


def judgments() -> dict[str, dict]:
    out = {}
    for f in sorted(JUDG.glob("*.json")):
        d = load(f)
        out[d["clip_id"]] = {k: d.get(k) for k in ("score", "note", "tags", "label", "position_s", "kind", "group_id")}
    return out


def clip_sha(audio: np.ndarray) -> str:
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "c.wav"
        sf.write(str(p), audio, SR, subtype="PCM_24")
        return hashlib.sha256(p.read_bytes()).hexdigest()


def write_mp3(audio: np.ndarray, path: Path) -> str:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        w = Path(tmp) / "c.wav"
        sf.write(str(w), audio, SR, subtype="PCM_24")
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(w), "-codec:a", "libmp3lame", "-b:a", "192k",
                        str(path)], check=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ helpers
def section_at(an: dict, bar: int) -> dict:
    rows = an["phrases"]
    bar = max(0, min(int(bar), len(rows) - 1))
    r = rows[bar]
    return {"bar": bar, "label": r["section_label"], "bar_in_section": r["bar_in_section"], "section_start": r["section_start"],
            "bars_to_next_section": r.get("bars_to_next_section")}


def bar_of(an: dict, t_s: float) -> float:
    """Battuta (frazionaria) della griglia di B a ``t_s`` secondi sorgente."""
    db = np.array([r["time_s"] for r in an["phrases"]])
    j = int(np.searchsorted(db, t_s, side="right") - 1)
    j = max(0, min(j, len(db) - 2))
    return float(j + (t_s - db[j]) / (db[j + 1] - db[j]))


def words_near(words_clip: list, t: float, span: float = 3.0) -> list:
    return [{"text": w["text"], "start_s": round(float(w["start_s"]), 3), "end_s": round(float(w["end_s"]), 3)}
            for w in words_clip if float(w["end_s"]) > t - span and float(w["start_s"]) < t + span]


def entry_words(words_clip: list, t: float, beat: float) -> dict:
    """Parole di B attorno al punto in cui B diventa udibile (secondi della clip)."""
    ws = sorted(words_clip, key=lambda w: float(w["start_s"]))
    cut = [w["text"] for w in ws if float(w["start_s"]) < t - EPS and float(w["end_s"]) > t + EPS]
    heard = [w for w in ws if float(w["start_s"]) >= t - EPS and float(w["end_s"]) > t + EPS]
    before = [w for w in ws if float(w["end_s"]) <= t + EPS]
    first = heard[0] if heard else None
    prev = before[-1] if before else None
    # diagnostic only: B's words before the audible point chained to the first heard one by gaps < 1 beat (a "breath
    # group"; in continuous rap it can span the whole verse). The QA/phrase rule uses a noted rest and line ends instead.
    lost = []
    if first is not None:
        k = ws.index(first)
        while k > 0 and float(ws[k]["start_s"]) - float(ws[k - 1]["end_s"]) < beat:
            k -= 1
            if float(ws[k]["end_s"]) <= t + EPS:
                lost.insert(0, ws[k]["text"])
    return {"cut_word": cut, "first_heard": None if first is None else first["text"],
            "first_heard_s": None if first is None else round(float(first["start_s"]) - t, 3),
            "previous_word": None if prev is None else prev["text"],
            "gap_prev_to_first_s": None if (first is None or prev is None) else round(float(first["start_s"]) - float(prev["end_s"]), 3),
            "phrase_words_lost_before": lost}


def refined_words(slug: str) -> list:
    return R.words_b_refined(slug) or []


def event_targets() -> dict[int, dict]:
    if not EVENTS.is_file():
        return {}
    d = load(EVENTS)
    out = {}
    for t in (d.get("event_plans") or {}).get("transitions", []):
        ep = t["event_plan"]
        out[int(t["index"])] = {"b_target": ep.get("b_target"), "b_entry": {k: (ep.get("b_entry") or {}).get(k)
                                                                            for k in ("lead_in", "section_at_entry", "section_start")}}
    return out


def b_audible(res, decks) -> dict:
    meta = res.metadata
    n = len(res.audio)
    geo = dj_qa.transition_geometry(meta, n)
    if decks is None:
        decks = R.cr.qa_inputs(res)["decks"] if res.decks else None
    st = dj_qa.b_audible_start(meta, geo, decks, R.QA_CFG)
    if st["sample"] is None:
        return dict(st, src_s=None)
    src = float(dj_qa.b_source_position(meta, n)[int(st["sample"])]) / SR
    return dict(st, out_s=st["sample"] / SR, src_s=src)


def qa_short(q: dict) -> dict:
    c = q["checks"]
    be, vc = c.get("b_entry") or {}, c.get("verse_cut") or {}
    return {"verdict": q["verdict"], "reasons": [f"{r['check']}: {r['message']}" for r in q["reasons"]],
            "b_entry": {k: be.get(k) for k in ("status", "coverage", "problem", "reason", "b_audible_s", "b_audible_source",
                                                "gap_before_first_s") if be.get(k) is not None},
            "b_entry_words": {"cut": [w.get("text") for w in be.get("words_cut") or []],
                              "first": (be.get("first_heard_word") or {}).get("text"),
                              "previous": (be.get("previous_word") or {}).get("text")},
            "verse_cut": {k: vc.get(k) for k in ("status", "coverage", "reason") if vc.get(k) is not None},
            "coverage_missing": (q.get("coverage") or {}).get("missing_required"), "schema": q.get("schema_version")}


def critic_short(crt: dict) -> dict:
    t = ((crt.get("v4") or {}).get("technical") or {})
    return {"veto_runner": crt.get("veto"), "v4_technical": t.get("status"),
            "v4_veto_flags": [f["flag"] for f in t.get("veto_flags") or []], "v4_warnings": [w["check"] for w in t.get("warnings") or []],
            "diagnostic_flags": crt.get("diagnostic_flags"), "error": crt.get("error") or crt.get("v4_error")}


# ------------------------------------------------------------------ one transition
def decision_for(role: str, tech: str, tr: dict, a_an: dict, b_an: dict, plan: dict, pair: dict, info_a: dict) -> dict:
    if role == "default":
        d0 = R.default_decision(a_an, b_an, plan, pair, info_a)
        for d in [d0] + list(d0.get("fallbacks") or []):
            d = {**d, "type_key": d.get("type_key", d["type"])}
            if d["type_key"] == tech:
                return d
        raise RuntimeError(f"T{tr['index']}: default technique {tech} not in the tree chain")
    o = next(o for o in tr["books_options_same_cue"] if o["type_key"] == tech)
    return R.opt_decision(o)


def render(role: str, dec: dict, A, B, plan: dict, prov: dict):
    if role == "default":
        res, decks = v2.render(dec, A, B, plan, prov)
        return R.with_b_lead_in(res, decks, B, plan, dec, A)
    return R.render_v3(dec, A, B, plan, prov)


def trimmed(res, kind: str):
    mk = R.clip_markers(res)
    return R.trim_to_rule(res.audio, kind, mk["t0"], mk["sw"], mk["t1"])


def do_transition(tr: dict, key: dict, man: dict, judg: dict, ev: dict, tracker, loaded: dict) -> tuple[dict, dict]:
    k, a, b = tr["index"], tr["a"], tr["b"]
    cat = R.load(R.FROZEN)
    src = {"plan_dir": R.PLAN, "catalog": cat, "catalog_path": R.FROZEN}
    for s in list(loaded):
        if s not in (a, b):
            del loaded[s]
    gc.collect()
    for s in (a, b):
        if s not in loaded:
            loaded[s] = R.v1.Track(s, R.PLAN, cat)
    A, B = loaded[a], loaded[b]
    a_an, b_an = A.analysis, B.analysis
    pp = djs.plan_pair(a_an, b_an, top_k=10 ** 9)
    pair = pp["compatibility"]
    plan = next(p for p in pp["plans"] if p["a_out_bar"] == tr["plan"]["a_out_bar"] and p["b_in_bar"] == tr["plan"]["b_in_bar"])
    info_a = R.info_v3(a, src)
    ps = R.phrase_summary(a, b, plan["a_out_s"], plan["b_in_s"], float(a_an["tempo"]["period_s"]))
    plan_books = dict(plan)
    if ps["switch_ok"] and (R.reliable(a) or R.reliable(b)):
        plan_books["phrase_switch"] = {"switch_rel_s": ps["switch_rel_s"], "tail_end_a_s": ps["tail_end_a_s"],
                                       "pickup_s": ps["entry_pickup_s"], "pickup_words": ps["entry"]["pickup_words"],
                                       "last_word": ps["exit"]["last_word"]}
    gid = f"g{k:02d}"
    gkey = key["groups"][gid]
    items = {i["id"]: i for g in man["groups"] if g["id"] == gid for i in g["items"]}
    beat_b = float(b_an["tempo"]["period_s"])
    prov = {"a": A.provenance, "b": B.provenance, "script": "tools/run_b_entry_fix_v1.py", "gallery_transition": k}
    wb = refined_words(b)
    ep_b = P.entry_point(R.MODELS[b], plan["b_in_s"], act=R.ACTS.get(b))
    li_b = P.b_lead_in(R.MODELS[b], plan["b_in_s"], act=R.ACTS.get(b))
    diag_rows, fix_rows = [], []
    entry_test = R.entry_test_for(b)
    for cid, rk in ((c, v) for c, v in gkey.items() if isinstance(v, dict) and "profile" in v):
        role, tech = rk["profile"], rk["technique"]
        dec = decision_for(role, tech, tr, a_an, b_an, plan, pair, info_a)
        pl = plan if role == "default" else plan_books
        # ---------------- old (frozen gallery)
        R.B_ENTRY["on"] = False
        t0 = time.time()
        res, decks = render(role, dec, A, B, pl, prov)
        kind = dec["type"]
        clip, tm = trimmed(res, kind)
        same = clip_sha(clip) == items[cid]["sha256"]
        n = len(res.audio)
        R.B_ENTRY["on"] = True  # QA .6 with B's refined words on the old render
        q_old = R.qa_v3(res, A, B, decks, tracker)
        aud = b_audible(res, decks)
        wclip = dj_qa.words_to_output(res.metadata, wb, n, deck="b")
        wraw = dj_qa.words_to_output(res.metadata, B.words, n, deck="b")
        off_clip = float(tm["src_window_s"][0])  # clip second 0 = render second off_clip
        ew = entry_words(wclip, aud["out_s"], beat_b) if aud.get("out_s") is not None else {}
        ew_raw = entry_words(wraw, aud["out_s"], beat_b) if aud.get("out_s") is not None else {}
        j = judg.get(cid, {})
        causes = []
        be = q_old["checks"]["b_entry"]
        if be.get("problem") == "b_word_cut" or ew.get("cut_word"):
            causes.append("b_word_cut")
        if be.get("problem") == "b_phrase_started_before" or (ew.get("phrase_words_lost_before") and not ew.get("cut_word")):
            causes.append("b_phrase_started_before")
        if aud.get("abrupt") and li_b["status"] == "lead_in":
            causes.append("pickup_not_played")
        if aud.get("abrupt") and li_b["status"] == "mid_line":
            causes.append("b_mid_line_entry")
        if q_old["checks"]["verse_cut"]["status"] == "fail":
            causes.append("a_word_cut")
        tgt = (ev.get(k) or {}).get("b_target") or {}
        if tgt.get("bar") is not None and aud.get("src_s") is not None and float(tgt["bar"]) - bar_of(b_an, aud["src_s"]) > 4:
            causes.append("b_event_late")
        if not causes:
            causes.append("none_measured")
        row = {"group": gid, "clip_id": cid, "transition": k, "a": a, "b": b, "profile": role, "technique": tech,
               "judgment": j, "clip_identical_to_voted": same,
               "a_exit": {"bar": plan["a_out_bar"], "s": plan["a_out_s"], "phrase": ps["exit"], "qa_verse_cut": q_old["checks"]["verse_cut"].get("status")},
               "b_cue": {"bar": plan["b_in_bar"], "s": plan["b_in_s"], "section": section_at(b_an, plan["b_in_bar"]),
                         "entry_point_dj_phrase_1": {k2: ep_b.get(k2) for k2 in ("status", "pickup_words", "first_word", "line_start_p", "reasons")},
                         "b_lead_in": {k2: li_b.get(k2) for k2 in ("status", "lead_in_samples", "lead_in_words", "phrase_words", "first_word",
                                                                    "rest_before_s", "acoustic", "evidence", "reasons")}},
               "b_audible_old": {"clip_s": None if aud.get("out_s") is None else aud["out_s"] - off_clip, "render_s": aud.get("out_s"),
                                 "b_source_s": aud.get("src_s"), "b_bar": None if aud.get("src_s") is None else round(bar_of(b_an, aud["src_s"]), 2),
                                 "abrupt": aud.get("abrupt"), "source": aud.get("source")},
               "b_words_at_entry_refined": ew, "b_words_at_entry_raw_asr": ew_raw,
               "b_words_near_entry_refined": words_near(wclip, aud["out_s"]) if aud.get("out_s") is not None else [],
               "b_target_event": tgt, "qa_dj_qa6_old": qa_short(q_old), "causes": causes}
        diag_rows.append(row)
        del res, decks
        gc.collect()
    # ---------------- verification: technique FIXED (the set's / books technique), B's entry varied, then A's exit varied
    tech = tr["technique"]  # the set's technique (the 'books' clip of the gallery; T8 had only one version, the same echo out)
    dec = R.opt_decision(next(o for o in tr["books_options_same_cue"] if o["type_key"] == tech))
    kind = dec["type"]
    fix_rows = []
    R.B_ENTRY["on"] = True
    if kind in ds.OVERLAP_TYPES:
        reloc = {"status": "exempt", "reason": "B enters with a fader (blend): no truncation"}
        new_bar = plan["b_in_bar"]
    else:
        r = djs.relocate_entry(a_an, b_an, plan, pair, entry_test)
        reloc = {"status": r["status"], "kind": r.get("kind"), "from_bar": r.get("from_bar"), "to_bar": r.get("to_bar"), "tried": r["tried"],
                 "entry": {k2: (r.get("entry") or {}).get(k2) for k2 in ("status", "lead_in_samples", "lead_in_words", "phrase_words",
                                                                         "first_word", "reasons")}}
        new_bar = int(r["plan"]["b_in_bar"]) if r["status"] == "moved" else plan["b_in_bar"]
    outs = {c["bar_index"]: c for c in djs.cue_candidates(a_an, "out")}
    ins = {c["bar_index"]: c for c in djs.cue_candidates(b_an, "in")}
    out0 = outs.get(plan["a_out_bar"]) or djs.bar_cue(a_an, plan["a_out_bar"])

    def in_cue(j):
        return ins.get(j) or djs.bar_cue(b_an, j)

    def make_plan(out_c, in_c):
        pv = djs.plan_transition(a_an, b_an, out_c, in_c, pair)
        li = P.b_lead_in(R.MODELS[b], pv["b_in_s"], act=R.ACTS.get(b))
        pv["b_lead_in"] = ds._lead_in_summary(li)
        return pv, li

    # entry variants (A's exit = the voted one): old, new, new -1 bar, next phrase start of B after the new cue
    entry_bars = [("E0_old", plan["b_in_bar"]), ("E1_new", new_bar)]
    alt = new_bar - 1 if (new_bar - 1 >= 0 and new_bar - 1 != plan["b_in_bar"]) else new_bar + 1
    entry_bars.append(("E2_new_minus_1_bar" if alt < new_bar else "E2_new_plus_1_bar", alt))
    nxt_phrase = next((j for j in sorted(ins) if j > new_bar and j not in [x[1] for x in entry_bars]), None)
    if nxt_phrase is not None:
        entry_bars.append(("E3_next_phrase_of_B", nxt_phrase))
    # exit variants (B's entry = the new one): old exit, previous phrase start of A, next phrase start of A (A prolonged)
    prev_out = next((j for j in sorted(outs, reverse=True) if j < plan["a_out_bar"]), None)
    next_out = next((j for j in sorted(outs) if j > plan["a_out_bar"]), None)
    exit_bars = [("X0_old_exit", plan["a_out_bar"])] + ([("X1_previous_phrase_of_A", prev_out)] if prev_out is not None else []) \
        + ([("X2_next_phrase_of_A_prolonged", next_out)] if next_out is not None else [])
    variants = []
    for lab, j in entry_bars:
        variants.append({"label": lab, "a_bar": plan["a_out_bar"], "b_bar": j, "lead_in": lab != "E0_old", "comparisons": ["entry"]})
    for lab, ja in exit_bars:
        if ja == plan["a_out_bar"]:
            variants[1]["comparisons"].append("exit")
            variants[1]["exit_label"] = lab
            continue
        variants.append({"label": lab, "a_bar": ja, "b_bar": new_bar, "lead_in": True, "comparisons": ["exit"]})
    for v in variants:
        t0 = time.time()
        out_c = outs.get(v["a_bar"]) or djs.bar_cue(a_an, v["a_bar"])
        if out_c is None or in_cue(v["b_bar"]) is None:
            fix_rows.append({"transition": k, "variant": v, "error": "cue outside the track"})
            continue
        pv, li = make_plan(out_c, in_cue(v["b_bar"]))
        ea = R.exit_at(a, pv["a_out_s"])
        R.B_ENTRY["on"] = bool(v["lead_in"])
        try:
            res, decks = R.render_v3(dec, A, B, pv, dict(prov, b_entry_fix="musicmix.b_entry_fix_v1", variant=v["label"]))
        except (dj.DJRenderError, ValueError) as exc:
            fix_rows.append({"transition": k, "variant": v, "error": f"{type(exc).__name__}: {exc}"})
            R.B_ENTRY["on"] = True
            continue
        R.B_ENTRY["on"] = True
        n = len(res.audio)
        q = R.qa_v3(res, A, B, decks, tracker)
        crt = R.critic(res, kind, f"t{k:02d}_{v['label']}", decks=res.decks, qa=q, declared_gap=R.declared_gap(dec))
        aud_n = b_audible(res, decks)
        wclip_n = dj_qa.words_to_output(res.metadata, wb, n, deck="b")
        clip_n, tm_n = trimmed(res, kind)
        off_n = float(tm_n["src_window_s"][0])
        be_meta = res.metadata.get("b_entry") or {}
        vlab = v["label"]
        cid = f"t{k:02d}_{hashlib.sha1(f'{SEED}{k}{vlab}'.encode()).hexdigest()[:6]}"
        mp3 = OUT / "clips" / f"{cid}.mp3"
        h = write_mp3(clip_n, mp3)
        fix_rows.append({
            "transition": k, "a": a, "b": b, "technique": tech, "variant": v["label"], "comparisons": v["comparisons"],
            "exit_label": v.get("exit_label"), "clip_id": cid, "relocation": reloc if v["label"] == "E1_new" else None,
            "a_exit": {"bar": pv["a_out_bar"], "s": pv["a_out_s"], "section": section_at(a_an, pv["a_out_bar"]),
                       "dj_phrase_exit": {k2: ea.get(k2) for k2 in ("status", "p_line_end", "last_word", "next_word", "reasons")}},
            "b_entry": {"bar": pv["b_in_bar"], "downbeat_s": pv["b_in_s"], "section": section_at(b_an, pv["b_in_bar"]),
                        "analysis": {k2: li.get(k2) for k2 in ("status", "lead_in_samples", "lead_in_words", "phrase_words", "first_word", "reasons")},
                        "audible_source_s": aud_n.get("src_s"), "audible_bar": None if aud_n.get("src_s") is None else round(bar_of(b_an, aud_n["src_s"]), 2),
                        "lead_in_s": be_meta.get("lead_in_s", 0.0), "pickup_played": bool(be_meta.get("lead_in_samples")),
                        "lead_in_mode": be_meta.get("mode"), "a_voice_under_lead_in": be_meta.get("a_voice"),
                        "words": entry_words(wclip_n, aud_n["out_s"], beat_b) if aud_n.get("out_s") is not None else {}},
            "qa": qa_short(q), "critic": critic_short(crt),
            "clip": {"path": str(mp3.relative_to(ROOT)), "sha256": h, "duration_s": len(clip_n) / SR, "rule_ok": tm_n["rule_ok"],
                     "transition_start_s": tm_n["t0"], "switch_s": tm_n["sw"], "transition_end_s": max(tm_n["t1"], tm_n["sw"]),
                     "b_audible_clip_s": None if aud_n.get("out_s") is None else aud_n["out_s"] - off_n},
            "versions": {"dj_qa": dj_qa.DJ_QA_VERSION, "b_lead_in": dj.B_LEAD_IN_VERSION, "b_entry_rule": P.B_ENTRY_VERSION,
                         "entry_guard": ds.ENTRY_GUARD_VERSION, "dj_render": dj.DJ_RENDER_VERSION},
            "seconds": round(time.time() - t0, 1)})
        print(f"T{k:02d} {tech:18s} {v['label']:30s} A {pv['a_out_bar']} B {pv['b_in_bar']} ({li['status']}) lead "
              f"{be_meta.get('lead_in_s', 0):.2f}s QA {q['verdict']}/{q['checks']['b_entry']['status']} [{time.time() - t0:.0f} s]", flush=True)
        del res, decks, clip_n
        gc.collect()
    print(f"T{k:02d} diagnosis: " + "; ".join(f"{r['profile']} {r['technique']} {r['causes']}" for r in diag_rows), flush=True)
    return {"transition": k, "rows": diag_rows}, {"transition": k, "rows": fix_rows}


def check_t01(judg: dict) -> dict:
    """verifica_t01 (runs/books_integration_check, halving build): diagnosi dai soli metadati della clip (nessun render)."""
    meta = load(CHECK_T01)
    meta = meta.get("metadata", meta)
    model = load(CHECK_T01_MODEL)
    model = model.get("model", model)
    li = P.b_lead_in(model, float(meta["cue_in_b_s"]))
    ep = P.entry_point(model, float(meta["cue_in_b_s"]))
    words = [{"start_sample": int(w["on"]), "end_sample": int(w["off"]), "text": w["text"]} for w in model["words"] if not w.get("dropped")]
    n = int(meta["clip"]["length_samples"])
    wclip = dj_qa.words_to_output(meta, words, n, deck="b")
    t = int(meta["switch"]["output_sample"]) / SR
    ew = entry_words(wclip, t, float(model["period_s"]))
    cid = next((c for c in judg if c.startswith("verifica_t01") and judg[c].get("kind") == "halving_build"), None)
    return {"group": "verifica_t01", "clip_id": cid, "judgment": judg.get(cid), "a": "heartless", "b": model.get("slug"),
            "technique": "halving_build", "b_cue_s": meta["cue_in_b_s"], "b_lead_in": {k: li.get(k) for k in
            ("status", "lead_in_samples", "lead_in_words", "phrase_words", "first_word", "reasons", "evidence")},
            "entry_point_dj_phrase_1": {k: ep.get(k) for k in ("status", "pickup_words", "first_word", "reasons")},
            "b_words_at_entry_refined": ew, "b_words_near_entry_refined": words_near(wclip, t),
            "asr_quality": model.get("asr_quality"), "lang": model.get("lang"),
            "note": "metadati della clip congelata (runs/books_integration_check); nessun render rifatto"}


# ------------------------------------------------------------------ stages
def run(only: list[int] | None) -> None:
    doc = R.load(R.OUT / "set.json")
    key = R.load(GAL / "blind_key.json")
    man = R.load(GAL / "review_manifest.json")
    judg = judgments()
    ev = event_targets()
    trs = [t for t in doc["transitions"][:R.MAX_GALLERY] if not only or t["index"] in only]
    R.load_phrase({x for t in trs for x in (t["a"], t["b"])})
    tracker = dj_qa.BeatThisTracker()
    loaded: dict = {}
    for tr in trs:
        k = tr["index"]
        dpath, fpath = DIAG / "transitions" / f"t{k:02d}.json", OUT / "transitions" / f"t{k:02d}.json"
        if dpath.is_file() and fpath.is_file() and not only:
            continue
        d, f = do_transition(tr, key, man, judg, ev, tracker, loaded)
        dump(dpath, d)
        dump(fpath, f)
    loaded.clear()
    dump(DIAG / "verifica_t01.json", check_t01(judg))


def summary() -> None:
    diag = [r for p in sorted((DIAG / "transitions").glob("t*.json")) for r in load(p)["rows"]]
    for r in diag:  # causes from the QA rule (dj_qa.6) and dj_phrase.b_lead_in; the 1-beat "breath group" is only a diagnostic
        q = r["qa_dj_qa6_old"]["b_entry"]
        w = r["b_words_at_entry_refined"]
        aud_abrupt = r["b_audible_old"]["abrupt"]
        li = r["b_cue"]["b_lead_in"]["status"]
        c = []
        if q.get("problem") == "b_word_cut" or w.get("cut_word"):
            c.append("b_word_cut")
        if q.get("problem") == "b_phrase_started_before":
            c.append("b_phrase_started_before")
        if aud_abrupt and li == "lead_in":
            c.append("pickup_not_played")
        if aud_abrupt and li == "mid_line":
            c.append("b_mid_line_entry")
        c += [x for x in r["causes"] if x in ("a_word_cut", "b_event_late")]
        r["causes_v1_loose"] = r["causes"]
        r["causes"] = c or ["none_measured"]
    fix = [r for p in sorted((OUT / "transitions").glob("t*.json")) for r in load(p)["rows"]]
    vt01 = load(DIAG / "verifica_t01.json") if (DIAG / "verifica_t01.json").is_file() else None
    noted = [r for r in diag if (r["judgment"] or {}).get("note") or (r["judgment"] or {}).get("tags")]
    b_noted = [r for r in diag if any(s in ((r["judgment"] or {}).get("note") or "").lower() for s in ("b ", "b,", "parola"))
               or "punto_frase_sbagliato" in ((r["judgment"] or {}).get("tags") or [])]
    count: dict = {}
    for r in b_noted:
        for c in r["causes"]:
            count[c] = count.get(c, 0) + 1
    count_all: dict = {}
    for r in diag:
        for c in r["causes"]:
            count_all[c] = count_all.get(c, 0) + 1
    out = {"schema": "musicmix.b_entry_diagnosis.1", "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "script": "tools/run_b_entry_fix_v1.py", "sources": {"gallery": "runs/dj_set_v3_bridge/gallery", "judgments": str(JUDG.relative_to(ROOT))},
           "note": ("diagnosi: dove B diventa udibile nelle clip votate e che cosa succede alle parole e alla frase di B; i voti sono "
                    "diagnostica tenuta separata (AGENTS.md), nessuna soglia adattata a una clip"),
           "rules": {"b_word_cut": f"parola di B che inizia prima e finisce dopo il punto udibile di oltre {EPS * 1000:.0f} ms [H]",
                     "b_phrase_started_before": "parola precedente di B a meno di 1 battito dalla prima udibile [H] (dj_qa.6)",
                     "pickup_not_played": "dj_phrase.b_lead_in = lead_in (la frase di B comincia <= 1 battuta prima del downbeat) e B entra sul downbeat",
                     "b_mid_line_entry": "dj_phrase.b_lead_in = mid_line (frase di B iniziata > 1 battuta prima o parola troncata)",
                     "a_word_cut": "dj_qa verse_cut fail (lato A)", "b_event_late": "evento obiettivo di B > 4 battute dopo il punto udibile"},
           "versions": {"dj_qa": dj_qa.DJ_QA_VERSION, "b_entry_rule": P.B_ENTRY_VERSION},
           "summary": {"clips": len(diag), "clips_identical_to_voted": sum(1 for r in diag if r["clip_identical_to_voted"]),
                       "clips_with_notes_or_tags": len(noted), "clips_with_b_notes": len(b_noted),
                       "causes_in_b_noted_clips": count, "causes_all_clips": count_all,
                       "qa6_b_entry_old": _count(r["qa_dj_qa6_old"]["b_entry"].get("status") for r in diag)},
           "clips": diag, "verifica_t01": vt01}
    dump(DIAG / "diagnosis.json", out)
    ok = [r for r in fix if "error" not in r]
    table = []
    for k in sorted({r["transition"] for r in ok}):
        rs = {r["variant"]: r for r in ok if r["transition"] == k}
        e0, e1 = rs.get("E0_old"), rs.get("E1_new")
        if not (e0 and e1):
            continue
        table.append({"transition": k, "a": e1["a"], "b": e1["b"], "technique": e1["technique"], "relocation": (e1["relocation"] or {}).get("status"),
                      "old": {"b_bar": e0["b_entry"]["bar"], "b_audible_s": e0["b_entry"]["audible_source_s"], "b_audible_bar": e0["b_entry"]["audible_bar"],
                              "first_word": e0["b_entry"]["words"].get("first_heard"), "cut_word": e0["b_entry"]["words"].get("cut_word"),
                              "previous_word": e0["b_entry"]["words"].get("previous_word"),
                              "gap_prev_to_first_s": e0["b_entry"]["words"].get("gap_prev_to_first_s"), "analysis": e0["b_entry"]["analysis"]["status"],
                              "qa": e0["qa"]["verdict"], "b_entry_qa": e0["qa"]["b_entry"].get("status")},
                      "new": {"b_bar": e1["b_entry"]["bar"], "b_audible_s": e1["b_entry"]["audible_source_s"], "b_audible_bar": e1["b_entry"]["audible_bar"],
                              "lead_in_s": e1["b_entry"]["lead_in_s"], "pickup_played": e1["b_entry"]["pickup_played"],
                              "first_word": e1["b_entry"]["words"].get("first_heard"), "cut_word": e1["b_entry"]["words"].get("cut_word"),
                              "analysis": e1["b_entry"]["analysis"]["status"], "qa": e1["qa"]["verdict"], "b_entry_qa": e1["qa"]["b_entry"].get("status"),
                              "qa_reasons": e1["qa"]["reasons"], "critic_v4": e1["critic"]["v4_technical"], "critic_veto": e1["critic"]["veto_runner"]}})
    res = {"schema": "musicmix.b_entry_fix.2", "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script": "tools/run_b_entry_fix_v1.py",
           "note": ("integrità tecnica, non qualità musicale: nessuna clip ascoltata. Tecnica FISSA (quella del set) per transizione; "
                    "confronto 'entry' = stessa uscita di A, ingressi di B diversi; confronto 'exit' = stesso ingresso nuovo di B, uscite di A diverse"),
           "versions": ok[0]["versions"] if ok else None,
           "summary": {"renders": len(fix), "errors": len(fix) - len(ok),
                       "old_vs_new": {"qa_old": _count(t["old"]["qa"] for t in table), "qa_new": _count(t["new"]["qa"] for t in table),
                                      "b_entry_old": _count(t["old"]["b_entry_qa"] for t in table),
                                      "b_entry_new": _count(t["new"]["b_entry_qa"] for t in table),
                                      "relocation": _count(t["relocation"] for t in table),
                                      "pickup_played_new": sum(1 for t in table if t["new"]["pickup_played"]),
                                      "critic_v4_new": _count(t["new"]["critic_v4"] for t in table)},
                       "all_variants_qa": _count(r["qa"]["verdict"] for r in ok),
                       "all_variants_b_entry": _count(r["qa"]["b_entry"].get("status") for r in ok)},
           "table": table, "variants": fix}
    dump(OUT / "results.json", res)
    # blind comparison manifest for a future gallery (not published): one group per comparison, technique fixed
    rng = random.Random(SEED)
    groups, keyd = [], {}
    for k in sorted({r["transition"] for r in ok}):
        for cmp_name in ("entry", "exit"):
            its = [r for r in ok if r["transition"] == k and cmp_name in r["comparisons"]]
            if len(its) < 2:
                continue
            rng.shuffle(its)
            gid = f"t{k:02d}_{cmp_name}"
            q = ("stessa tecnica e stessa uscita di A: quale ingresso di B è giusto?" if cmp_name == "entry"
                 else "stessa tecnica e stesso ingresso di B: quale uscita di A è giusta?")
            groups.append({"id": gid, "title": f"{k}. {its[0]['a']} → {its[0]['b']} ({its[0]['technique']}): {q}", "question": q,
                           "items": [{"id": r["clip_id"], "kind": r["technique"], "role": "versione", "audio_path": r["clip"]["path"],
                                      "duration_s": r["clip"]["duration_s"], "transition_start_s": r["clip"]["transition_start_s"],
                                      "switch_s": r["clip"]["switch_s"], "transition_end_s": r["clip"]["transition_end_s"],
                                      "qa": {"verdict": r["qa"]["verdict"], "warn_reasons": r["qa"]["reasons"]}, "sha256": r["clip"]["sha256"]}
                                     for r in its]})
            keyd[gid] = {r["clip_id"]: {"variant": r["variant"] if cmp_name == "entry" else (r.get("exit_label") or r["variant"]),
                                        "a_exit_bar": r["a_exit"]["bar"], "b_entry_bar": r["b_entry"]["bar"],
                                        "b_analysis": r["b_entry"]["analysis"]["status"], "lead_in_s": r["b_entry"]["lead_in_s"],
                                        "a_exit_analysis": r["a_exit"]["dj_phrase_exit"]["status"], "qa": r["qa"]["verdict"]} for r in its}
    dump(OUT / "clips_manifest.json", {"schema_version": "musicmix.dj-review.1", "title": "Ingresso di B / uscita di A a tecnica fissa (b_entry_fix_v1), non pubblicata",
                                       "run": "b_entry_fix_v1",
                                       "conventions": {"clip_rule": "AGENTS.md: blend/effetti 10 s prima dell'inizio + transizione + 10 s dopo",
                                                       "blind": "ordine casuale (seed in clips_key.json); varianti rivelate dopo il voto",
                                                       "comparisons": "gruppo tNN_entry: tecnica e uscita di A fisse, ingressi di B diversi (vecchio, nuovo, "
                                                                      "+-1 battuta, frase successiva di B); gruppo tNN_exit: tecnica e ingresso nuovo di B "
                                                                      "fissi, uscite di A diverse (vecchia, frase precedente, frase successiva = A prolungata). "
                                                                      "La clip 'nuovo' è condivisa dai due gruppi"},
                                       "groups": groups})
    dump(OUT / "clips_key.json", {"seed": SEED, "note": "NON pubblicare: chiave delle varianti", "groups": keyd})
    # same clips as tournaments for tools/listening_pages.py (musicmix.cue-variants.1): one per transition, technique fixed
    tours = []
    for k in sorted({r["transition"] for r in ok}):
        its = [r for r in ok if r["transition"] == k]
        an = {x: load(R.PLAN / "tracks" / f"{x}.json") for x in (its[0]["a"], its[0]["b"])}
        side = lambda x: {"slug": x, "title": an[x].get("title"), "artist": an[x].get("artist"), "audio_sha256": an[x].get("track_id")}
        e1 = next((r for r in its if r["variant"] == "E1_new"), its[0])
        tours.append({"id": f"t{k:02d}", "transition_index": k, "a": side(its[0]["a"]), "b": side(its[0]["b"]),
                      "technique": its[0]["technique"], "system": {"a_out_bar": e1["a_exit"]["bar"], "b_in_bar": e1["b_entry"]["bar"]},
                      "comparisons": {c: [r["clip_id"] for r in its if c in r["comparisons"]] for c in ("entry", "exit")},
                      "variants": [{"id": r["clip_id"], "audio_path": r["clip"]["path"], "sha256": r["clip"]["sha256"],
                                    "technique": r["technique"], "a_out_bar": r["a_exit"]["bar"], "b_in_bar": r["b_entry"]["bar"],
                                    "a_out_s": r["a_exit"]["s"], "b_in_s": r["b_entry"]["downbeat_s"],
                                    "b_lead_in_samples": int(round(float(r["b_entry"]["lead_in_s"] or 0.0) * SR)),
                                    "transition_start_s": r["clip"]["transition_start_s"], "switch_s": r["clip"]["switch_s"],
                                    "transition_end_s": r["clip"]["transition_end_s"], "duration_s": r["clip"]["duration_s"],
                                    "label": r["variant"], "qa": {"verdict": r["qa"]["verdict"], "warn_reasons": r["qa"]["reasons"]}}
                                   for r in its]})
    dump(OUT / "cue_variants.json", {"schema_version": "musicmix.cue-variants.1", "run": "b_entry_fix_v1",
                                     "title": "Ingresso di B e uscita di A a tecnica fissa (b_entry_fix_v1)",
                                     "source_set": {"path": "runs/dj_set_v3_bridge/set.json", "sha256": R.sha(R.OUT / "set.json")},
                                     "tournaments": tours})
    print(json.dumps(out["summary"], ensure_ascii=False))
    print(json.dumps(res["summary"], ensure_ascii=False))
    for t in table:
        print(t["transition"], t["technique"], t["relocation"], "old", t["old"], "new", {k: t["new"][k] for k in
              ("b_bar", "b_audible_bar", "lead_in_s", "first_word", "cut_word", "qa", "b_entry_qa")})


def _count(it) -> dict:
    c: dict = {}
    for v in it:
        c[str(v)] = c.get(str(v), 0) + 1
    return c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("run", "summary"))
    ap.add_argument("--only", type=int, nargs="*")
    args = ap.parse_args(argv)
    if args.stage == "run":
        run(args.only)
    else:
        summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
