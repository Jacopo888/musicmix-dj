#!/usr/bin/env python3
"""Galleria d'ascolto DJ v2: stessi piani di v1 con section gain, guardia vocale dei cut e QA di fase.

Rispetto a ``tools/run_dj_gallery_v1.py`` (che resta invariato, come la sua run
``runs/dj_gallery_v1``) cambiano solo tre cose, fissate a priori e non
adattate ai giudizi d'ascolto:

1. **Section gain match** (``dj_render.SectionGainConfig(enabled=True)``,
   default del modulo: A nelle ultime 2 battute prima dell'inizio della
   transizione, B nelle prime 4 battute dopo lo switch, B stretchato per i
   blend; guadagno statico su B limitato a +-6 dB; dopo la finestra di B, e
   mai prima della fine della transizione, rampa lenta di 16 battute che
   riporta B al suo livello normalizzato -14 LUFS). Stesso parametro per
   cut/blend (dj_render) ed echo_out/filter_out/drop_in (dj_effects).
2. **Guardia vocale dei cut** (cut e drop_in, che sono cut): ammessi solo se lo
   stem vocale di A (RMS 20 ms sopra la soglia dello stem intero, come
   dj_qa) è silenzioso nell'ultimo battito prima del cut e nei 150 ms dopo, e
   nessuna parola ASR di A cade entro +-150 ms dal cut (la stessa finestra del
   controllo ``verse_cut`` di dj_qa). Altrimenti ``dj_select.relocate_cut``
   cerca il downbeat di inizio frase valido più vicino (prima il precedente a
   pari distanza, entro 16 battute) che superi la guardia e non sia in
   dissolvenza (stessa regola di v1); in mancanza si ripiega su echo_out (o
   filter_out se echo_out è già una versione del gruppo), uscite esenti.
3. **QA con provenienza delle griglie**: ``dj_qa.evaluate_render(...,
   grid_support=...)`` riceve i blocchi ``tempo`` di dj_select dei due brani,
   così un errore a mezzo battito con griglie confermate da allin1/madmom è un
   warn ("possibile ambiguità di fase del tracker"), non un fail.

Scrive soltanto in una cartella NUOVA (default ``runs/dj_gallery_v2``) con la
stessa struttura di v1 (``clips/``, ``rejected/``, ``review_manifest.json``
schema ``musicmix.dj-review.1``, ``gallery_log.json`` con il confronto con v1).

    python tools/run_dj_gallery_v2.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dj_gallery_v1 as v1  # noqa: E402  (also puts src/ on sys.path)

from musicmix import dj_effects as fx  # noqa: E402
from musicmix import dj_qa  # noqa: E402
from musicmix import dj_render as dj  # noqa: E402
from musicmix import dj_select as djs  # noqa: E402

ROOT = v1.ROOT
SR = v1.SR
SCHEMA = v1.SCHEMA
RUN = "dj_gallery_v2"
SCRIPT = "tools/run_dj_gallery_v2.py"

SECTION_GAIN = dj.SectionGainConfig(enabled=True)  # module defaults: 2 / 4 bars, +-6 dB, 16-bar ramp
BLEND_CFG = dj.DJRenderConfig(cue_convention="start", section_gain=SECTION_GAIN)
CUT_CFG = dj.DJRenderConfig(section_gain=SECTION_GAIN)
FX_CFG = fx.DJEffectsConfig(section_gain=SECTION_GAIN)

# Cut vocal guard [H] (same windows as dj_qa.verse_cut where they overlap).
GUARD_WINDOW_BEATS = 1  # last beat before the cut (2 = last half bar)
GUARD_POST_S = dj_qa.WORD_GUARD_S  # stem silent also 150 ms after the cut
GUARD_ASR_S = dj_qa.WORD_GUARD_S  # no ASR word within +-150 ms
GUARD_MAX_BARS = djs.CUT_GUARD_MAX_BARS
CUT_KINDS = ("cut", "drop_in")


# -------------------------------------------------------------------- tracks
class Track(v1.Track):
    def __init__(self, slug: str, plan_dir: Path, catalog: dict):
        super().__init__(slug, plan_dir, catalog)
        act, _ = dj_qa.vocal_activity(self.vocals, SR, None, self.vocal_threshold)
        self.vocal_act = act
        self.vocal_times = (np.arange(len(act)) + 0.5) * dj_qa.VOCAL_HOP_S

    def cut_guard(self, cut_s: float) -> dict:
        beat = float(self.analysis["tempo"]["period_s"])
        before = GUARD_WINDOW_BEATS * beat
        stem_clear = djs.vocal_clear_window(self.vocal_times, self.vocal_act, cut_s, before, GUARD_POST_S)
        words = [w for w in self.words
                 if w["end_sample"] / SR > cut_s - GUARD_ASR_S and w["start_sample"] / SR < cut_s + GUARD_ASR_S]
        return {"cut_s": cut_s, "window_s": [cut_s - before, cut_s + GUARD_POST_S], "window_beats": GUARD_WINDOW_BEATS,
                "stem_clear": stem_clear, "asr_words": [w.get("text") for w in words],
                "clear": bool(stem_clear and not words)}


# ------------------------------------------------------------------ versions
def apply_cut_guard(versions: list[dict], A: Track, B: Track, plan: dict, comp: dict) -> tuple[list[dict], list[dict]]:
    """Cut/drop_in ammessi solo con l'uscita di A libera dalla voce; altrimenti spostati o sostituiti."""
    out, log = [], []
    for v in versions:
        if v["kind"] not in CUT_KINDS:
            out.append(v)
            continue
        g = A.cut_guard(plan["a_out_s"])
        if g["clear"]:
            out.append(dict(v, cut_guard=g))
            continue

        def is_clear(c: dict) -> bool:
            return A.cut_guard(c["time_s"])["clear"] and not A.fade_check(c["time_s"])["in_fade"]

        res = djs.relocate_cut(A.analysis, B.analysis, plan, comp, is_clear=is_clear, max_bars=GUARD_MAX_BARS)
        entry = {"role": v["role"], "kind": v["kind"], "a_out_bar": plan["a_out_bar"], "guard": g,
                 "relocation": {k: res.get(k) for k in ("status", "from_bar", "to_bar", "shift_bars", "tried")}}
        if res["status"] == "moved":
            new = res["plan"]
            nv = dict(v, cut_guard=A.cut_guard(new["a_out_s"]), a_out_s=new["a_out_s"], a_out_bar=new["a_out_bar"],
                      recovered={"how": "moved", "from_bar": plan["a_out_bar"], "to_bar": new["a_out_bar"],
                                 "shift_bars": res["shift_bars"], "guard_at_plan_cue": g,
                                 "rescored_plan": {"type": new.get("type"), "score": new.get("score")}},
                      why=v["why"] + f"; cut spostato da battuta {plan['a_out_bar']} a {new['a_out_bar']}"
                      f" ({res['shift_bars']:+d}): voce di A nell'ultimo battito prima del cut")
            out.append(nv)
            entry["action"] = f"moved to bar {new['a_out_bar']}"
        else:
            used = {x["kind"] for x in versions}
            kind = "echo_out" if "echo_out" not in used else "filter_out" if "filter_out" not in used else None
            if kind is None:
                entry["action"] = "dropped: no clear cue and both exempt exits already used"
                log.append(entry)
                continue
            nv = {"role": v["role"], "kind": kind, "why": f"{kind} al posto di {v['kind']}: voce di A sull'uscita e nessun "
                  f"downbeat di inizio frase libero entro {GUARD_MAX_BARS} battute (uscita con effetto, esente)",
                  "recovered": {"how": "fallback", "replaced_kind": v["kind"], "guard_at_plan_cue": g}}
            if kind == "filter_out":
                nv["sweep_bars"] = v1.FILTER_SWEEP_BARS
            out.append(nv)
            entry["action"] = f"fallback {kind}"
        log.append(entry)
    return out, log


def version_label(v: dict) -> str:
    lab = v1.version_label(v)
    if (v.get("recovered") or {}).get("how") == "moved":
        lab += f" (A battuta {v['a_out_bar']})"
    return lab


# -------------------------------------------------------------------- render
def render_version(v: dict, A: Track, B: Track, plan: dict, prov: dict):
    grids = dict(beats_a=A.beats, downbeats_a=A.downbeats, beats_b=B.beats, downbeats_b=B.downbeats)
    a_out_req = v.get("a_out_s", plan["a_out_s"])
    ia = int(np.argmin(np.abs(A.downbeats - a_out_req)))
    a_out, b_in = float(A.downbeats[ia]), float(plan["b_in_s"])
    kind = v["kind"]
    params: dict[str, Any] = {"cue_out_a_s": a_out, "cue_in_b_s": b_in, "section_gain": "enabled (dj_render.SectionGainConfig defaults)"}
    if kind == "blend":
        params.update(bars=v["bars"], cue_convention="start", stretch_mode=BLEND_CFG.stretch_mode, vocal_ducking="auto (>= 2 s clash)")
        res = dj.render_transition(A.audio, B.audio, **grids, cue_out_a=a_out, cue_in_b=b_in, kind="blend", bars=v["bars"],
                                   sample_rate=SR, vocals_a=A.vocals, vocals_b=B.vocals, config=BLEND_CFG, provenance=prov,
                                   return_decks=True)
    elif kind == "cut":
        res = dj.render_transition(A.audio, B.audio, **grids, cue_out_a=a_out, cue_in_b=b_in, kind="cut", sample_rate=SR,
                                   config=CUT_CFG, provenance=prov, return_decks=True)
        params.update(crossfade_ms=CUT_CFG.crossfade_ms)
    elif kind == "drop_in":
        label = next(s["label"] for s in B.analysis["sections"] if s["bar_index"] == plan["b_in_bar"])
        res = fx.drop_in(A.audio, B.audio, **grids, cue_out_a=a_out, cue_in_b=b_in, cue_in_is_section_start=True,
                         section_label=label, sample_rate=SR, config=FX_CFG, provenance=prov, return_decks=True)
        params.update(section_label=label, crossfade_ms=FX_CFG.crossfade_ms)
    elif kind == "echo_out":
        if ia < 1:
            raise dj.DJRenderError("echo_out: no A downbeat before a_out")
        echo_at = float(A.downbeats[ia - 1])
        params.update(echo_cue_out_a_s=echo_at, delay_beats=v1.ECHO_DELAY_BEATS, entry_bars=1,
                      note="echo engaged on the downbeat one bar before a_out; B enters clean on a_out (same change point)")
        res = fx.echo_out(A.audio, B.audio, **grids, cue_out_a=echo_at, cue_in_b=b_in, delay_beats=v1.ECHO_DELAY_BEATS,
                          entry_bars=1, sample_rate=SR, config=FX_CFG, provenance=prov, return_decks=True)
    elif kind == "filter_out":
        params.update(sweep_bars=v["sweep_bars"], filter_in=False)
        res = fx.filter_out(A.audio, B.audio, **grids, cue_out_a=a_out, cue_in_b=b_in, sweep_bars=v["sweep_bars"],
                            sample_rate=SR, config=FX_CFG, provenance=prov, return_decks=True)
    else:
        raise ValueError(kind)
    meta = res.metadata
    decks = dict(res.decks or {})
    gains_note = None
    n = len(res.audio)
    t = np.arange(n)
    if kind == "echo_out":
        dry_end = int(meta["effect"]["dry_a_cut_output_sample"])
        entry = int(meta["switch"]["output_sample"])
        decks["gains"] = dict(decks.get("gains", {}), fader_a=(t < dry_end).astype(float), fader_b=(t >= entry).astype(float))
        gains_note = "reconstructed from dj_effects metadata: A dry until dry_a_cut_output_sample, B from its entry (x section gain); echo tail of A's vocal (-12 dB under B) not counted"
    elif kind == "filter_out":
        cut = int(meta["switch"]["output_sample"])
        decks["gains"] = dict(decks.get("gains", {}), fader_a=(t < cut).astype(float), fader_b=(t >= cut).astype(float))
        gains_note = "reconstructed from dj_effects metadata: A (high-passed) until the cut, B from the cut (x section gain)"
    return res, params, {"decks": decks, "gains_note": gains_note}


def run_qa(res, A: Track, B: Track, decks: dict, tracker) -> dict:
    n = len(res.audio)
    meta = res.metadata
    va, vb = dj_qa.stems_to_output(meta, n, A.vocals, B.vocals)
    words = dj_qa.words_to_output(meta, A.words, n)
    return dj_qa.evaluate_render(res.audio.astype(np.float64), meta, sample_rate=res.sample_rate, decks=decks,
                                 vocals_a=va, vocals_b=vb, vocal_thresholds=(A.vocal_threshold, B.vocal_threshold),
                                 words_a=words, tracker=tracker,
                                 grid_support={"a": A.analysis["tempo"], "b": B.analysis["tempo"]})


# ------------------------------------------------------------- v1 comparison
def v1_verdicts(v1_dir: Path) -> dict:
    """(group, role) -> {id, kind, verdict, reasons} from the frozen v1 run (read only)."""
    out = {}
    if not (v1_dir / "review_manifest.json").is_file():
        return out
    man = v1.load_json(v1_dir / "review_manifest.json")
    for g in man["groups"]:
        for it in g["items"]:
            out[(g["id"], it["role"])] = {"id": it["id"], "kind": it["kind"], "verdict": it["qa"]["verdict"],
                                          "reasons": it["qa"]["warn_reasons"]}
    for r in man.get("rejected", []):
        role = r["id"].split("__")[-1].split("_")[0]
        out[(r["group"], role)] = {"id": r["id"], "kind": r["kind"], "verdict": "fail" if r["stage"] == "qa" else "render_error",
                                   "reasons": r["reasons"]}
    return out


# ---------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default=str(ROOT / "runs/dj_plan_v1"))
    ap.add_argument("--out", default=str(ROOT / "runs" / RUN))
    ap.add_argument("--compare", default=str(ROOT / "runs/dj_gallery_v1"), help="frozen v1 run to compare with (read only)")
    ap.add_argument("--max-plans", type=int, default=6)
    args = ap.parse_args(argv)
    plan_dir, out = Path(args.plan), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty: refusing to overwrite")
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "rejected").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pairs_doc = v1.load_json(plan_dir / "pairs.json")
    summary = v1.load_json(plan_dir / "summary.json")
    catalog_path = v1.localize(pairs_doc["catalog"])
    if not catalog_path.is_absolute():
        catalog_path = ROOT / catalog_path
    catalog = v1.load_json(catalog_path)
    slugs = sorted({p[k] for p in pairs_doc["pairs"] if p["compatibility"]["status"] == "compatible" for k in ("a", "b")})
    analyses = {s: v1.load_json(plan_dir / "tracks" / f"{s}.json") for s in slugs}
    log: list = []
    selection = v1.pick_plans(analyses, pairs_doc, args.max_plans, log)
    tracks = {s: Track(s, plan_dir, catalog) for s in sorted({x[k] for x in selection for k in ("a", "b")})}
    tracker = dj_qa.BeatThisTracker()
    prev = v1_verdicts(Path(args.compare))
    counts = {"pass": 0, "warn": 0, "fail": 0, "render_error": 0}
    groups, rejected, guard_log, comparison, gains = [], [], [], [], []
    for gi, sel in enumerate(selection, 1):
        A, B = tracks[sel["a"]], tracks[sel["b"]]
        plan, adj, fade = v1.adjust_for_fade(A, B, sel)
        comp = sel["compatibility"]
        gid = f"g{gi:02d}_{A.slug}__{B.slug}_a{plan['a_out_bar']}_b{plan['b_in_bar']}"
        versions, ctx = v1.choose_versions(A.analysis, B.analysis, plan, comp)
        versions, glog = apply_cut_guard(versions, A, B, plan, comp)
        for e in glog:
            guard_log.append({"group": gid, **e})
        print(f"[{gid}] {plan['type']} {plan['blend_bars']} -> " + ", ".join(version_label(v) for v in versions)
              + (f"  (a_out spostato da battuta {adj['original']['a_out_bar']})" if adj else ""), flush=True)
        prov = {"a": A.provenance, "b": B.provenance, "plan_run": v1.rel(plan_dir), "plan": {k: plan[k] for k in (
            "a_out_s", "a_out_bar", "b_in_s", "b_in_bar", "type", "blend_bars", "score", "confidence")},
            "script": SCRIPT, "cue_adjustment": adj}
        items = []
        for v in versions:
            iid = f"{gid}__{v['role']}_{v['kind']}" + (str(v["bars"]) if v["kind"] == "blend" else "")
            if (v.get("recovered") or {}).get("how") == "moved":
                iid += f"_a{v['a_out_bar']}"
            before = prev.get((gid, v["role"]))
            try:
                res, params, extra = render_version(v, A, B, plan, dict(prov, version=v.get("recovered")))
            except dj.DJRenderError as exc:
                counts["render_error"] += 1
                rejected.append({"group": gid, "id": iid, "kind": v["kind"], "stage": "render", "reasons": [str(exc)]})
                comparison.append({"group": gid, "role": v["role"], "v1": before, "v2": {"id": iid, "kind": v["kind"], "verdict": "render_error"}})
                print(f"   RENDER ERROR {iid}: {exc}", flush=True)
                continue
            qa = run_qa(res, A, B, extra["decks"], tracker)
            verdict = qa["verdict"]
            counts[verdict] += 1
            dest = out / ("rejected" if verdict == "fail" else "clips")
            written = dj.write_render(res, dest, iid)
            meta = v1.load_json(written["json"])
            mk = v1.markers(res.metadata)
            sg = res.metadata.get("section_gain") or {}
            sg_short = {k: sg.get(k) for k in ("applied_db", "raw_db", "clamped", "a_lufs", "b_lufs", "gain_db_at_clip_end", "status")}
            record = {"gallery": {"schema": SCHEMA, "run": RUN, "group": gid, "item": iid, "role": v["role"], "kind": v["kind"],
                                  "label": version_label(v), "why": v["why"], "params": params, "markers_s": mk,
                                  "cut_guard": v.get("cut_guard"), "recovered": v.get("recovered"), "section_gain": sg_short,
                                  "qa_inputs": {"decks": True, "vocal_stems": True, "asr_words_a": True, "grid_support": True,
                                                "tracker": qa["inputs"]["tracker"], "gains_note": extra["gains_note"]}},
                      "render": meta, "qa": qa}
            written["json"].write_text(json.dumps(v1.jsonable(record), indent=1, ensure_ascii=False), encoding="utf-8")
            reasons = [f"{r['check']}: {r['message']}" for r in qa["reasons"]]
            print(f"   {verdict.upper():4s} {iid} ({res.metadata['clip']['duration_s']:.2f} s, gain B {sg_short['applied_db'] or 0:+.2f} dB) "
                  + "; ".join(reasons), flush=True)
            gains.append({"id": iid, "kind": v["kind"], **sg_short,
                          "limiter_max_gain_reduction_db": res.metadata["limiter"].get("max_gain_reduction_db"),
                          "true_peak_dbtp": res.metadata["limiter"].get("true_peak_after_dbtp")})
            comparison.append({"group": gid, "role": v["role"], "v1": before,
                               "v2": {"id": iid, "kind": v["kind"], "verdict": verdict, "reasons": reasons,
                                      "section_gain_b_db": sg_short["applied_db"], "recovered": (v.get("recovered") or {}).get("how")}})
            if verdict == "fail":
                rejected.append({"group": gid, "id": iid, "kind": v["kind"], "stage": "qa", "reasons": reasons,
                                 "audio_path": v1.rel(written["wav"]), "json_path": v1.rel(written["json"]), "sha256": written["sha256"]})
                continue
            items.append({
                "id": iid, "role": v["role"], "kind": v["kind"], "label": version_label(v), "why": v["why"], "params": params,
                "audio_path": v1.rel(written["wav"]), "json_path": v1.rel(written["json"]),
                "duration_s": res.metadata["clip"]["duration_s"], "sample_rate": res.sample_rate, **mk,
                "qa": {"verdict": verdict, "warn_reasons": reasons if verdict == "warn" else []},
                "section_gain_b_db": sg_short["applied_db"], "recovered": v.get("recovered"),
                "sha256": written["sha256"],
            })
        a_secs = A.analysis["phrases"][plan["a_out_bar"]]
        groups.append({
            "id": gid,
            "pair": {"a": {"slug": A.slug, "title": A.title, "artist": A.analysis.get("artist")},
                     "b": {"slug": B.slug, "title": B.title, "artist": B.analysis.get("artist")}},
            "title": f"{A.title} → {B.title}",
            "bpm": {"a": A.analysis["tempo"]["bpm"], "b": B.analysis["tempo"]["bpm"],
                    "stretch_b_pct": comp["tempo"]["stretch_b_pct"]},
            "key": {"a": A.analysis["key"]["camelot"], "b": B.analysis["key"]["camelot"],
                    "a_name": A.analysis["key"]["key"], "b_name": B.analysis["key"]["key"], "relation": comp["key"]["relation"]},
            "A_out_s": plan["a_out_s"], "B_in_s": plan["b_in_s"], "A_out_bar": plan["a_out_bar"], "B_in_bar": plan["b_in_bar"],
            "sections": {"a": plan["a_section"], "b": plan["b_section"], "a_bar_in_section": a_secs["bar_in_section"]},
            "plan": {"type": plan["type"], "blend_bars": plan["blend_bars"], "score": plan["score"],
                     "confidence": plan["confidence"], "rank_in_pair": sel["rank_in_pair"]},
            "reasons": list(comp["reasons"]) + list(plan["reasons"]),
            "version_context": ctx,
            "cut_guard": [e for e in glog],
            "fade_check": fade,
            "cue_adjustment": adj,
            "items": items,
        })
    manifest = {
        "schema_version": SCHEMA,
        "title": "MusicMix — galleria DJ v2: section gain, cut senza parole troncate",
        "run": RUN,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"plan_run": v1.rel(plan_dir), "pairs_sha256": v1.sha256(plan_dir / "pairs.json"),
                   "summary_sha256": v1.sha256(plan_dir / "summary.json"), "catalog": v1.rel(catalog_path),
                   "compared_with": v1.rel(Path(args.compare)) if Path(args.compare).exists() else None},
        "conventions": {
            "audio_path": "relative to the repository root",
            "markers": "seconds in the clip: transition_start_s / switch_s / transition_end_s from the renderer metadata (overlap, switch)",
            "change_point": "B's b_in downbeat lands on A's a_out downbeat in every version (echo_out: echo on the bar before); a cut moved by the vocal guard exits A on another phrase-start downbeat (label shows the bar)",
            "clip_rules": "cut/drop_in 10 s + 10 s; blend/effects 10 s + whole transition + 10 s (AGENTS.md, 20/10/2026)",
            "grids": "dj_select regularised grids (runs/dj_plan_v1/tracks), not raw Beat This!",
            "section_gain": "B gain = L(A, last 2 bars before the transition) - L(B, first 4 bars after the switch), BS.1770 3 s gated, clamped +-6 dB; 16-bar raised-cosine return to -14 LUFS after B's window (mostly after the clip)",
            "cut_guard": "cut/drop_in only if A's vocal stem is silent in the last beat before the cut and 150 ms after, and no ASR word within +-150 ms; else nearest clear phrase-start cue (<= 16 bars) or echo_out/filter_out",
            "qa": "dj_qa.evaluate_render with decks, vocal stems, ASR words of A, Beat This! small0 and grid provenance (phase ambiguity warn); fail = discarded",
            "no_stay_items": "continuing A is not offered as an item to rate (AGENTS.md)",
        },
        "counts": counts,
        "groups": groups,
        "rejected": rejected,
    }
    (out / "review_manifest.json").write_text(json.dumps(v1.jsonable(manifest), indent=1, ensure_ascii=False), encoding="utf-8")
    v1_counts = None
    if (Path(args.compare) / "gallery_log.json").is_file():
        v1_counts = v1.load_json(Path(args.compare) / "gallery_log.json").get("counts")
    (out / "gallery_log.json").write_text(json.dumps(v1.jsonable({
        "log": log, "counts": counts, "v1_counts": v1_counts, "rejected": rejected, "cut_guard": guard_log,
        "section_gain": gains, "comparison_v1_v2": comparison, "elapsed_s": time.time() - t0,
        "settings": {"section_gain": SECTION_GAIN.__dict__, "guard": {"window_beats": GUARD_WINDOW_BEATS, "post_s": GUARD_POST_S,
                                                                     "asr_s": GUARD_ASR_S, "max_bars": GUARD_MAX_BARS}},
        "summary_top_plans": summary.get("top_plans")}), indent=1, ensure_ascii=False), encoding="utf-8")
    if not any((out / "rejected").iterdir()):
        shutil.rmtree(out / "rejected")
    print(json.dumps(counts), "v1:", json.dumps(v1_counts), f"{time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
