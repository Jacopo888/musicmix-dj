#!/usr/bin/env python3
"""Galleria d'ascolto delle tecniche creative v1 (dj_creative + dj_policy) con QA.

Per 6-8 coppie compatibili (pool V12 ``runs/dj_plan_v1`` + brani CC con analisi
completa, scoperti come in ``tools/run_dj_set_v1.py``):

1. sceglie il punto di cambio (piani di ``dj_select.plan_pair``: cue su downbeat
   di inizio frase) che abilita più regole specifiche della policy (lyric link,
   ritornello, passaggio di verso, estensione), scartando le uscite in
   dissolvenza come nelle gallerie precedenti;
2. chiede a ``dj_policy.propose`` 2-4 tecniche e ne renderizza fino a 3 con
   ``dj_creative`` (se una fallisce il QA si prova la successiva), più un
   riferimento standard dell'albero ``dj_set.decide_transition`` sullo stesso
   cue (blend/echo/filter/cut con dj_render/dj_effects);
3. esegue ``dj_qa.evaluate_render`` su ogni clip (deck, voce udibile di ciascun
   deck nella clip, parole ASR di A mappate con ``dj_creative.words_in_clip``,
   Beat This!, provenienza delle griglie) e scrive ``review_manifest.json``
   schema ``musicmix.dj-review.1`` (come ``runs/dj_gallery_v2``), leggibile da
   ``tools/build_listening_artifact.py``.

Scrive solo in una cartella NUOVA (default ``runs/dj_creative_v1``). Le run e i
giudizi esistenti sono solo letti.

    python tools/run_dj_creative_v1.py [--pairs 8] [--per-pair 3]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import run_dj_gallery_v1 as v1  # noqa: E402  (Track, fade check, regular grid)
import run_dj_set_v1 as sv  # noqa: E402  (track discovery, baseline render)

from musicmix import dj_creative as cr  # noqa: E402
from musicmix import dj_policy as pol  # noqa: E402
from musicmix import dj_qa  # noqa: E402
from musicmix import dj_render as dj  # noqa: E402
from musicmix import dj_select as djs  # noqa: E402
from musicmix import dj_set as ds  # noqa: E402

RUN = "dj_creative_v1"
SCRIPT = "tools/run_dj_creative_v1.py"
SCHEMA = "musicmix.dj-review.1"
SR = 48000
CFG = cr.DJCreativeConfig()
CONTEXT_RULES = ("C1_lyric", "C2_chorus", "C3_verse_handoff", "C4_extend")
MAX_PLANS_PER_PAIR = 60

LABELS = {
    "loop_extend": lambda p: f"loop di A ({p.get('loop_bars', 1)} batt.) + blend {p.get('bars', 8)}",
    "acapella_over": lambda p: f"acapella di B su strumentale di A ({p.get('bars', 4)} batt.)" + (f", B {p['key_shift_b']:+d} st" if p.get("key_shift_b") else ""),
    "vocal_swap": lambda p: f"vocal swap ({p.get('bars', 4)}+{p.get('out_bars', 2)} batt.)" + (f", B {p['key_shift_b']:+d} st" if p.get("key_shift_b") else ""),
    "build_into_chorus": lambda p: f"build {p.get('build_bars', 2)} batt. verso il ritornello" + (" + stutter" if p.get("stutter_bars") else ""),
    "note_bridge": lambda p: "nota ponte (pad)" + (" con 1 battuta sola" if p.get("gap_bars") else ""),
    "echo_freeze": lambda p: "echo freeze",
    "reverb_wash": lambda p: "reverb wash",
    "backspin": lambda p: "backspin",
    "tape_stop": lambda p: "tape stop",
    "lyric_link": lambda p: f"lyric link «{p['a_word'].get('text', '').strip()}» → «{p['b_word'].get('text', '').strip()}»",
}


def jsonable(o: Any) -> Any:
    return dj_qa._jsonable(o)


STATS_DIR = ROOT / ".cache" / "dj_creative_v1" / "track_stats"


class LightTrack:
    """Brano senza audio in memoria (OOM del cgroup condiviso, 21/10): analisi, griglia
    regolarizzata, parole ASR e statistiche dell'intero brano (LUFS integrata, soglia dello stem
    vocale, mediana short-term per il controllo di dissolvenza) calcolate una volta e salvate in
    ``.cache/dj_creative_v1/track_stats``; l'audio si legge a estratti con ``read()``."""

    def __init__(self, slug: str, plan_dir: Path, catalog: dict):
        import soundfile as sf

        self.slug = slug
        self.analysis = v1.load_json(plan_dir / "tracks" / f"{slug}.json")
        entry = next(t for t in catalog["tracks"] if t["slug"] == slug)
        self.wav = v1.localize(self.analysis["inputs"]["canonical_wav"])
        vjson = v1.localize(entry["vocals_path"])
        # NB (21/10/2026): tool congelato; nome dello stem indovinato. Nel codice attivo usare musicmix.stems.vocal_stem_path.
        self.stem = vjson.parent / "a_vocals.wav"
        if entry["track_id"] != self.analysis["track_id"]:
            raise RuntimeError(f"{slug}: catalog/analysis track id mismatch")
        self.words = v1.load_json(vjson)["words"]
        self.beats, self.downbeats = v1.regular_grid(self.analysis)
        self.n = sf.info(str(self.wav)).frames
        STATS_DIR.mkdir(parents=True, exist_ok=True)
        sp = STATS_DIR / f"{slug}.json"
        st = v1.load_json(sp) if sp.is_file() else None
        if not st or st.get("track_id") != self.analysis["track_id"]:
            audio, sr = dj.load_track(self.wav)
            if v1.sha256(self.wav) != self.analysis["track_id"] or sr != SR:
                raise RuntimeError(f"{slug}: canonical audio identity mismatch")
            lufs = dj.integrated_lufs(audio, SR)
            prof = v1.short_term_profile(audio, SR)
            del audio
            voc, _ = dj.load_track(self.stem)
            thr = dj_qa.stem_threshold(voc, SR)
            del voc
            st = {"track_id": self.analysis["track_id"], "lufs": lufs, "vocal_threshold": thr,
                  "st_median": float(np.median(prof[np.isfinite(prof) & (prof > -70)])),
                  "stem_sha256": v1.sha256(self.stem)}
            sp.write_text(json.dumps(st), encoding="utf-8")
        self.lufs, self.vocal_threshold, self.st_median = st["lufs"], st["vocal_threshold"], st["st_median"]
        self.provenance = {
            "slug": slug, "title": self.analysis.get("title"), "artist": self.analysis.get("artist"),
            "audio": v1.rel(self.wav), "audio_sha256": self.analysis["track_id"], "vocal_stem": v1.rel(self.stem),
            "vocal_stem_sha256": st["stem_sha256"], "vocals_json": v1.rel(vjson),
            "analysis": v1.rel(plan_dir / "tracks" / f"{slug}.json"),
            "grid": {"source": "dj_select regularised constant-tempo grid", "provider": self.analysis["tempo"]["provider"],
                     "confidence": self.analysis["tempo"]["confidence"], "bpm": self.analysis["tempo"]["bpm"]},
        }

    @property
    def title(self) -> str:
        return self.analysis.get("title") or self.slug

    def read(self, lo: int, hi: int, stem: bool = False) -> np.ndarray:
        import soundfile as sf

        x, _ = sf.read(str(self.stem if stem else self.wav), start=max(0, lo), stop=min(self.n, hi), dtype="float64", always_2d=True)
        return dj._as_stereo(x)

    def fade_check(self, cue_s: float) -> dict:
        lo = max(0, int((cue_s - v1.FADE_WINDOW_S) * SR))
        seg = self.read(lo, int(cue_s * SR))
        lv = v1.block_lufs(seg, SR, 0.0, len(seg) / SR)
        drop = (self.st_median - lv) if lv is not None else None
        return {"lufs_4s_before_cue": lv, "track_short_term_median_lufs": self.st_median, "below_median_db": drop,
                "in_fade": bool(drop is None or drop > v1.FADE_MAX_DB)}


def load_words(entry: dict) -> tuple[list, str | None]:
    doc = v1.load_json(v1.localize(entry["vocals_path"]))
    lang = (doc.get("backend") or {}).get("detected_language") or (doc.get("backend") or {}).get("language")
    return doc.get("words") or [], lang


# ------------------------------------------------------------------- planning
def pair_candidates(analyses: dict, sources: dict, log: list) -> list[dict]:
    """Ordered pairs with tempo within +-8% and key compatible or fixable by +-1 semitone."""
    words = {}
    for s, src in sources.items():
        entry = next(t for t in src["catalog"]["tracks"] if t["slug"] == s)
        words[s] = load_words(entry)
    out = []
    slugs = sorted(analyses)
    for a in slugs:
        for b in slugs:
            if a == b:
                continue
            A, B = analyses[a], analyses[b]
            comp = djs.pair_compatibility(A, B)
            if not comp["tempo"]["compatible"] or A["tempo"]["confidence"] == "low" or B["tempo"]["confidence"] == "low":
                continue
            shift = cr.auto_key_shift(A["key"].get("key"), B["key"].get("key"), 1)
            if shift is None or not comp["key"]["confident"]:
                continue
            r = djs.plan_pair(A, B, top_k=MAX_PLANS_PER_PAIR)
            outs = [c["time_s"] for c in djs.cue_candidates(A, "out")]
            ins = [c["time_s"] for c in djs.cue_candidates(B, "in")]
            links = cr.find_lyric_links(words[a][0], words[b][0], outs, ins, sr=SR, bar_a_s=4 * A["tempo"]["period_s"],
                                        bar_b_s=4 * B["tempo"]["period_s"], lang_a=words[a][1], lang_b=words[b][1], top_k=20)
            best = None
            for plan in r["plans"]:
                pl_links = [l for l in links if abs(l["a_cue_s"] - plan["a_out_s"]) < 1e-3 and abs(l["b_cue_s"] - plan["b_in_s"]) < 1e-3]
                ctx = pol.creative_context(A, B, plan, comp, has_stems=True, lyric_links=pl_links)
                ctx["sample_rate"] = SR
                prop = pol.propose(ctx)
                ctx_rules = sorted({p["rule"] for p in prop["proposals"] if p["rule"] in CONTEXT_RULES})
                value = plan["score"] + 1.5 * len(ctx_rules) + (1.0 if pl_links else 0.0)
                if best is None or value > best["value"]:
                    best = {"value": value, "plan": plan, "ctx": ctx, "proposals": prop, "links": pl_links, "context_rules": ctx_rules}
            if best is None:
                continue
            out.append({"a": a, "b": b, "compatibility": comp, "key_shift": shift, **best, "n_links_pair": len(links)})
    log.append({"pair_candidates": [{"a": p["a"], "b": p["b"], "value": p["value"], "rules": p["context_rules"],
                                     "stretch_pct": p["compatibility"]["tempo"]["stretch_b_pct"], "key": p["compatibility"]["key"]["relation"],
                                     "shift": p["key_shift"], "links": p["n_links_pair"]} for p in out]})
    return out


def pick_pairs(cands: list[dict], n: int, sources: dict) -> list[dict]:
    """Greedy: best value first, each track at most twice, no reverse duplicate, mix of pools, varied rules."""
    cands = sorted(cands, key=lambda p: -p["value"])
    used: dict[str, int] = {}
    seen_pairs = set()
    rule_count: dict[str, int] = {}
    chosen = []
    for p in cands:
        if len(chosen) >= n:
            break
        if used.get(p["a"], 0) >= 2 or used.get(p["b"], 0) >= 2 or (p["b"], p["a"]) in seen_pairs:
            continue
        main = p["context_rules"][0] if p["context_rules"] else "none"
        if rule_count.get(main, 0) >= max(2, n // 3) and any(c["context_rules"] and c["context_rules"][0] != main for c in cands):
            continue
        chosen.append(p)
        seen_pairs.add((p["a"], p["b"]))
        for s in (p["a"], p["b"]):
            used[s] = used.get(s, 0) + 1
        rule_count[main] = rule_count.get(main, 0) + 1
    return chosen


# --------------------------------------------------------------------- render
def deck_phase_error(qa: dict) -> dict | None:
    """Errore di fase B - A misurato da dj_qa (Beat This! sui deck nella sovrapposizione), in s.

    Usato per il "nudge" in seconda passata: le griglie regolarizzate di dj_select allineano i
    downbeat delle griglie, ma gli attacchi reali possono differire di 10-30 ms (run preliminare:
    loop_extend 21 ms costanti). Un DJ correggerebbe a orecchio; qui si usa la stessa misura del
    QA, quindi dopo il nudge il controllo ritmico non è più indipendente (dichiarato nel record).
    """
    bt = (((qa.get("checks") or {}).get("rhythm") or {}).get("methods") or {}).get("beat_this_decks")
    if not bt:
        return None
    errs = list(bt.get("beat_errors_ms") or [])
    if len(errs) < 8:
        errs = errs + list(bt.get("downbeat_errors_ms") or [])
    if len(errs) < 6:
        return None
    e = np.asarray(errs, float)
    med = float(np.median(e))
    iqr = float(np.percentile(e, 75) - np.percentile(e, 25))
    return {"median_ms": med, "iqr_ms": iqr, "n": int(len(e)), "usable": bool(5.0 <= abs(med) <= 40.0 and iqr <= 25.0)}


A_WINDOW_S = (45.0, 60.0)  # excerpt of A around its cue (before, after): memory, the clip needs < 35 s before
B_WINDOW_S = (30.0, 90.0)
_LUFS: dict = {}


def whole_lufs(T) -> float:
    return T.lufs


def excerpt(T, t: float, before: float, after: float) -> dict:
    """Estratto di audio, stem e griglie attorno a ``t`` (secondi), con l'offset in campioni.

    Solo per la memoria (OOM del cgroup condiviso con gli altri agenti nella run del 21/10):
    ogni tempo resta coerente perché griglie, cue e parole sono traslati dello stesso offset."""
    lo = max(0, int((t - before) * SR))
    hi = min(T.n, int((t + after) * SR))
    off = lo / SR
    sel = lambda g: g[(g >= off - 1e-6) & (g < hi / SR)] - off  # noqa: E731
    return {"lo": lo, "off_s": off, "audio": T.read(lo, hi), "vocals": T.read(lo, hi, stem=True),
            "beats": sel(T.beats), "downbeats": sel(T.downbeats)}


def shift_word(w: dict, lo: int) -> dict:
    return dict(w, start_sample=int(w["start_sample"]) - lo, end_sample=int(w["end_sample"]) - lo)


def render_creative(prop: dict, A, B, plan: dict, prov: dict, nudge_s: float = 0.0):
    params = dict(prop["params"])
    a_out = params.pop("cue_out_a_s", plan["a_out_s"])
    b_in = params.pop("cue_in_b_s", plan["b_in_s"])
    ia = int(np.argmin(np.abs(A.downbeats - a_out)))
    jb = int(np.argmin(np.abs(B.downbeats - b_in)))
    a_cue, b_cue = float(A.downbeats[ia]), float(B.downbeats[jb])
    ea = excerpt(A, a_cue, *A_WINDOW_S)
    eb = excerpt(B, b_cue, *B_WINDOW_S)
    cfg = cr.DJCreativeConfig(lufs_a_override=whole_lufs(A), lufs_b_override=whole_lufs(B), inplace_inputs=True)
    prov = dict(prov, excerpt={"a_source_offset_samples": ea["lo"], "b_source_offset_samples": eb["lo"],
                               "note": "audio passed as excerpts (memory); source samples in the render metadata are relative to them; LUFS of the whole tracks"})
    if "a_word" in params:
        params["a_word"] = shift_word(params["a_word"], ea["lo"])
        params["b_word"] = shift_word(params["b_word"], eb["lo"])
    kw = dict(beats_a=ea["beats"], downbeats_a=ea["downbeats"], beats_b=eb["beats"], downbeats_b=eb["downbeats"],
              cue_out_a=a_cue - ea["off_s"], cue_in_b=b_cue - eb["off_s"], vocals_a=ea["vocals"], vocals_b=eb["vocals"],
              sample_rate=SR, config=cfg, provenance=prov, return_decks=True)
    if prop["technique"] in ("note_bridge",):
        params.setdefault("key_a", A.analysis["key"].get("key"))
        params.setdefault("key_b", B.analysis["key"].get("key"))
    if prop["technique"] == "build_into_chorus":
        params.setdefault("key_b", B.analysis["key"].get("key"))
    if nudge_s and prop["technique"] in cr.OVERLAP_TECHNIQUES:
        params["phase_nudge_s"] = float(nudge_s)
    res = cr.render_creative(prop["technique"], ea["audio"], eb["audio"], **kw, **params)
    res.metadata["excerpt_offsets_samples"] = {"a": ea["lo"], "b": eb["lo"]}
    if "a_word" in params:
        params["a_word"] = shift_word(params["a_word"], -ea["lo"])
        params["b_word"] = shift_word(params["b_word"], -eb["lo"])
    return res, {"cue_out_a_s": a_cue, "cue_in_b_s": b_cue, **{k: v for k, v in params.items() if k not in ("a_word", "b_word")},
                 **({"a_word": params["a_word"].get("text"), "b_word": params["b_word"].get("text")} if "a_word" in params else {})}


def qa_creative(res, A, B, tracker) -> dict:
    meta = res.metadata
    qi = cr.qa_inputs(res)
    ga = 10 ** (meta["loudness"]["a_gain_db"] / 20.0)
    gb = 10 ** (meta["loudness"]["b_gain_db"] / 20.0)
    lo = (meta.get("excerpt_offsets_samples") or {}).get("a", 0)
    words = cr.words_in_clip(meta, [shift_word(w, lo) for w in A.words], "a")
    return dj_qa.evaluate_render(res.audio.astype(np.float64), meta, sample_rate=res.sample_rate, decks=qi["decks"],
                                 vocals_a=qi["vocals_a"], vocals_b=qi["vocals_b"],
                                 vocal_thresholds=(A.vocal_threshold * ga, B.vocal_threshold * gb), words_a=words, tracker=tracker,
                                 grid_support={"a": A.analysis["tempo"], "b": B.analysis["tempo"]})


def markers(meta: dict) -> dict:
    sr = int(meta["sample_rate"])
    ov, sw = meta["overlap"], meta["switch"]
    return {"transition_start_s": ov["output_start_sample"] / sr, "switch_s": sw["output_sample"] / sr,
            "transition_end_s": ov["output_end_sample"] / sr}


# ------------------------------------------------------------------ children
def run_child(spec: dict, work: Path, retries: int = 3) -> dict:
    """Ogni render in un sottoprocesso (memoria liberata all'uscita; OOM del cgroup condiviso nella
    notte del 21/10): risultato in ``_work/<iid>.result.json``; se il figlio viene ucciso si riprova."""
    import subprocess

    rp = work / f"{spec['iid']}.result.json"
    if rp.is_file():
        return v1.load_json(rp)
    sp = work / f"{spec['iid']}.spec.json"
    sp.write_text(json.dumps(jsonable(spec), ensure_ascii=False), encoding="utf-8")
    for attempt in range(1, retries + 1):
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", str(sp), "--result", str(rp)],
                              cwd=str(ROOT), capture_output=True, text=True)
        for line in proc.stdout.splitlines():
            if line.strip():
                print(line, flush=True)
        if rp.is_file():
            return v1.load_json(rp)
        print(f"   child exited {proc.returncode} (attempt {attempt}): {proc.stderr.strip()[-400:]}", flush=True)
    res = {"counter": "render_error", "rejected": {"group": spec["gid"], "id": spec["iid"], "kind": (spec.get("prop") or {}).get("technique", "baseline"),
                                                    "stage": "render", "reasons": [f"child process failed {retries} times (memory?)"]}}
    rp.write_text(json.dumps(res), encoding="utf-8")
    return res


def _child_tracks(spec: dict):
    out = []
    for k in ("a", "b"):
        src = spec[f"src_{k}"]
        out.append(LightTrack(spec[k], Path(src["plan_dir"]), v1.load_json(Path(src["catalog_path"]))))
    return out


def child_creative(spec: dict) -> dict:
    out = Path(spec["out"])
    A, B = _child_tracks(spec)
    tracker = dj_qa.BeatThisTracker()
    prop, plan, gid, iid, role, prov = spec["prop"], spec["plan"], spec["gid"], spec["iid"], spec["role"], spec["prov"]
    ts = time.time()
    try:
        pv = dict(prov, proposal={k: prop[k] for k in ("technique", "rule", "why", "prior")})
        res, params = render_creative(prop, A, B, plan, pv)
    except (dj.DJRenderError, ValueError) as exc:
        print(f"   RENDER ERROR {iid}: {exc}", flush=True)
        return {"counter": "render_error", "rejected": {"group": gid, "id": iid, "kind": prop["technique"], "stage": "render", "reasons": [str(exc)]}}
    except Exception as exc:  # keep the run going, record the traceback
        print(f"   RENDER CRASH {iid}: {exc!r}", flush=True)
        return {"counter": "render_error", "rejected": {"group": gid, "id": iid, "kind": prop["technique"], "stage": "render", "reasons": [repr(exc)],
                                                        "traceback": traceback.format_exc()[-2000:]}}
    qa = qa_creative(res, A, B, tracker)
    nudge_log = None
    if prop["technique"] in cr.OVERLAP_TECHNIQUES and qa["checks"]["rhythm"]["status"] != "pass":
        pe = deck_phase_error(qa)
        nudge_log = {"first_pass": {"verdict": qa["verdict"], "reasons": [f"{r['check']}: {r['message']}" for r in qa["reasons"]],
                                    "phase_error": pe}}
        if pe and pe["usable"]:
            first_verdict = qa["verdict"]
            del res  # memory: never hold two renders at once
            import gc

            gc.collect()
            res, params = render_creative(prop, A, B, plan, dict(pv, phase_nudge_first_pass=nudge_log), nudge_s=pe["median_ms"] / 1000.0)
            qa = qa_creative(res, A, B, tracker)
            nudge_log["second_pass"] = {"nudge_ms": pe["median_ms"], "verdict": qa["verdict"]}
            nudge_log["note"] = "B nudged by the Beat This! deck error of the first pass: the rhythm check of the kept render uses the same tracker, so it is not independent"
            order = {"pass": 0, "warn": 1, "fail": 2}
            if order[qa["verdict"]] <= order[first_verdict]:
                nudge_log["kept"] = "second pass (nudged)"
            else:
                del res
                gc.collect()
                res, params = render_creative(prop, A, B, plan, pv)  # deterministic: same as the first pass
                qa = qa_creative(res, A, B, tracker)
                nudge_log["kept"] = "first pass (nudge did not help; re-rendered)"
        print(f"      nudge: {json.dumps(nudge_log, default=str)[:300]}", flush=True)
    verdict = qa["verdict"]
    dest = out / ("rejected" if verdict == "fail" else "clips")
    written = dj.write_render(res, dest, iid)
    meta = v1.load_json(written["json"])
    mk = markers(res.metadata)
    label = LABELS[prop["technique"]](prop["params"])
    reasons = [f"{r['check']}: {r['message']}" for r in qa["reasons"]]
    record = {"gallery": {"schema": SCHEMA, "run": RUN, "group": gid, "item": iid, "role": role, "kind": prop["technique"],
                          "label": label, "why": "; ".join(prop["why"]), "rule": prop["rule"], "rule_source": prop["rule_source"],
                          "params": params, "markers_s": mk, "phase_nudge": nudge_log},
              "render": meta, "qa": qa}
    written["json"].write_text(json.dumps(jsonable(record), indent=1, ensure_ascii=False), encoding="utf-8")
    import resource

    print(f"   {verdict.upper():4s} {iid} ({res.metadata['clip']['duration_s']:.1f} s, {time.time() - ts:.0f} s, "
          f"RSS max {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB) " + "; ".join(reasons), flush=True)
    tm = {"id": iid, "s": time.time() - ts}
    if verdict == "fail":
        return {"counter": verdict, "timing": tm, "rejected": {"group": gid, "id": iid, "kind": prop["technique"], "stage": "qa", "reasons": reasons,
                "audio_path": v1.rel(written["wav"]), "json_path": v1.rel(written["json"]), "sha256": written["sha256"]}}
    return {"counter": verdict, "timing": tm, "item": {"id": iid, "role": role, "kind": prop["technique"], "label": label,
                  "why": "; ".join(prop["why"]) + f" [{prop['rule']}: {prop['rule_source']}]", "params": params,
                  "audio_path": v1.rel(written["wav"]), "json_path": v1.rel(written["json"]),
                  "duration_s": res.metadata["clip"]["duration_s"], "sample_rate": res.sample_rate, **mk,
                  "qa": {"verdict": verdict, "warn_reasons": reasons if verdict == "warn" else []},
                  "section_gain_b_db": (res.metadata.get("section_gain") or {}).get("applied_db"),
                  "creative": {k: res.metadata["creative"].get(k) for k in ("technique", "params", "regions", "basis", "a_vocal_mute_guard")},
                  "phase_nudge": nudge_log,
                  "sha256": written["sha256"]}}


def child_baseline(spec: dict) -> dict:
    out = Path(spec["out"])
    A, B = _child_tracks(spec)
    tracker = dj_qa.BeatThisTracker()
    dec, plan, gid, iid, prov = spec["dec"], spec["plan"], spec["gid"], spec["iid"], spec["prov"]
    try:
        from types import SimpleNamespace

        a_cue = float(A.downbeats[int(np.argmin(np.abs(A.downbeats - plan["a_out_s"])))])
        b_cue = float(B.downbeats[int(np.argmin(np.abs(B.downbeats - plan["b_in_s"])))])
        ea, eb = excerpt(A, a_cue, *A_WINDOW_S), excerpt(B, b_cue, *B_WINDOW_S)
        NA = SimpleNamespace(audio=ea["audio"], vocals=ea["vocals"], beats=ea["beats"], downbeats=ea["downbeats"], analysis=A.analysis,
                             words=[shift_word(w, ea["lo"]) for w in A.words], vocal_threshold=A.vocal_threshold)
        NB = SimpleNamespace(audio=eb["audio"], vocals=eb["vocals"], beats=eb["beats"], downbeats=eb["downbeats"], analysis=B.analysis,
                             words=[shift_word(w, eb["lo"]) for w in B.words], vocal_threshold=B.vocal_threshold)
        plan_x = dict(plan, a_out_s=a_cue - ea["off_s"], b_in_s=b_cue - eb["off_s"])
        res, decks = sv.render(dec, NA, NB, plan_x, dict(prov, reference="dj_set.decide_transition",
                                                         excerpt={"a_source_offset_samples": ea["lo"], "b_source_offset_samples": eb["lo"],
                                                                  "note": "dj_render/dj_effects normalize the excerpt (not the whole track) to -14 LUFS"}))
        qa = sv.run_qa(res, NA, NB, decks, tracker)
    except (dj.DJRenderError, ValueError) as exc:
        print(f"   RENDER ERROR {iid}: {exc}", flush=True)
        return {"counter": "render_error", "rejected": {"group": gid, "id": iid, "kind": dec["type"], "stage": "render", "reasons": [str(exc)]}}
    verdict = qa["verdict"]
    dest = out / ("rejected" if verdict == "fail" else "clips")
    written = dj.write_render(res, dest, iid)
    meta = v1.load_json(written["json"])
    mk = markers(res.metadata)
    reasons = [f"{r['check']}: {r['message']}" for r in qa["reasons"]]
    written["json"].write_text(json.dumps(jsonable({"gallery": {"schema": SCHEMA, "run": RUN, "group": gid, "item": iid, "role": "z", "kind": dec["type"],
                                                                "label": "riferimento: " + dec["label"], "rule": dec["rule"], "markers_s": mk},
                                                    "render": meta, "qa": qa}), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"   {verdict.upper():4s} {iid} (riferimento {dec['label']}) " + "; ".join(reasons), flush=True)
    baseline = {"id": iid, "type": dec["type"], "label": dec["label"], "rule": dec["rule"], "verdict": verdict}
    if verdict == "fail":
        return {"counter": verdict, "baseline": baseline, "rejected": {"group": gid, "id": iid, "kind": dec["type"], "stage": "qa", "reasons": reasons,
                "audio_path": v1.rel(written["wav"]), "json_path": v1.rel(written["json"]), "sha256": written["sha256"]}}
    return {"counter": verdict, "baseline": baseline, "item": {
        "id": iid, "role": "z", "kind": dec["type"], "label": "riferimento standard: " + dec["label"],
        "why": f"albero dj_set {dec['rule']} ({dec['rule_source']}) sullo stesso cue, per confronto; dj_render normalizza l'estratto, non il brano intero",
        "params": {"render": dec["render"]}, "audio_path": v1.rel(written["wav"]), "json_path": v1.rel(written["json"]),
        "duration_s": res.metadata["clip"]["duration_s"], "sample_rate": res.sample_rate, **mk,
        "qa": {"verdict": verdict, "warn_reasons": reasons if verdict == "warn" else []}, "sha256": written["sha256"]}}


def child_stats(slug: str, plan_dir: str, catalog_path: str) -> None:
    LightTrack(slug, Path(plan_dir), v1.load_json(Path(catalog_path)))


# ----------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "runs" / RUN))
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--per-pair", type=int, default=3)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--resume", action="store_true", help="reuse _work/*.result.json of an interrupted run")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--result", help=argparse.SUPPRESS)
    ap.add_argument("--stats", nargs=3, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.stats:
        child_stats(*args.stats)
        return 0
    if args.child:
        spec = v1.load_json(Path(args.child))
        res = child_creative(spec) if spec["mode"] == "creative" else child_baseline(spec)
        Path(args.result).write_text(json.dumps(jsonable(res), ensure_ascii=False), encoding="utf-8")
        return 0
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise SystemExit(f"{out} exists and is not empty: refusing to overwrite")
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "rejected").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log: list = []
    sources = sv.discover(log)
    analyses = {s: v1.load_json(src["plan_dir"] / "tracks" / f"{s}.json") for s, src in sources.items()}
    print(f"brani disponibili: {len(analyses)}", flush=True)
    cands = pair_candidates(analyses, sources, log)
    print(f"coppie compatibili (tempo +-8%, tonalità compatibile o a +-1): {len(cands)}", flush=True)
    # fade check needs the audio: verify while picking
    tracks: dict[str, Any] = {}

    def load(slug):
        if slug not in tracks:
            if len(tracks) >= 12:
                tracks.pop(next(iter(tracks)))
            src = sources[slug]
            if not (STATS_DIR / f"{slug}.json").is_file():  # whole-track stats in a child (memory)
                import subprocess

                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--stats", slug, str(src["plan_dir"]), str(src["catalog_path"])],
                               cwd=str(ROOT), check=True)
            tracks[slug] = LightTrack(slug, src["plan_dir"], src["catalog"])
        return tracks[slug]

    pool = sorted(cands, key=lambda p: -p["value"])
    selection: list[dict] = []
    rejected_pairs = []
    while len(selection) < args.pairs and pool:
        pick = pick_pairs(pool, args.pairs - len(selection), sources)
        if not pick:
            break
        for p in pick:
            pool.remove(p)
            A = load(p["a"])
            fc = A.fade_check(p["plan"]["a_out_s"])
            if fc["in_fade"]:
                rejected_pairs.append({"a": p["a"], "b": p["b"], "why": "A exit in a fade", "fade": fc})
                continue
            p["fade_check"] = fc
            selection.append(p)
            # keep the per-track limit across rounds
            pool = [q for q in pool if sum(1 for s in selection for k in ("a", "b") if s[k] in (q["a"], q["b"])) < 3
                    and (q["a"], q["b"]) not in {(s["b"], s["a"]) for s in selection} and (q["a"], q["b"]) not in {(s["a"], s["b"]) for s in selection}]
            if len(selection) >= args.pairs:
                break
    log.append({"selection": [{"a": p["a"], "b": p["b"], "rules": p["context_rules"], "plan": {k: p["plan"][k] for k in ("a_out_bar", "b_in_bar", "type", "score")}}
                              for p in selection], "rejected_pairs": rejected_pairs})
    counts = {"pass": 0, "warn": 0, "fail": 0, "render_error": 0}
    groups, rejected, timing = [], [], []
    work = out / "_work"
    work.mkdir(exist_ok=True)
    src_of = lambda slug: {"plan_dir": str(sources[slug]["plan_dir"]), "catalog_path": str(sources[slug]["catalog_path"])}  # noqa: E731
    for gi, sel in enumerate(selection, 1):
        A, B = load(sel["a"]), load(sel["b"])
        plan, comp = sel["plan"], sel["compatibility"]
        gid = f"c{gi:02d}_{A.slug[:28]}__{B.slug[:28]}_a{plan['a_out_bar']}_b{plan['b_in_bar']}"
        props = sel["proposals"]["proposals"]
        print(f"[{gid}] regole {sel['context_rules']} -> " + ", ".join(p["technique"] for p in props), flush=True)
        prov = {"a": A.provenance, "b": B.provenance, "plan": {k: plan[k] for k in ("a_out_s", "a_out_bar", "b_in_s", "b_in_bar", "type", "score")},
                "script": SCRIPT, "policy_version": pol.POLICY_VERSION}
        items = []
        good = tried = 0
        roles = iter("abcdefgh")
        for prop in props:
            if good >= args.per_pair or tried >= args.per_pair + 2:
                break
            tried += 1
            role = next(roles)
            iid = f"{gid}__{role}_{prop['technique']}"
            spec = {"mode": "creative", "out": str(out), "gid": gid, "iid": iid, "role": role, "prop": prop, "plan": plan,
                    "a": sel["a"], "b": sel["b"], "src_a": src_of(sel["a"]), "src_b": src_of(sel["b"]), "prov": prov}
            r = run_child(spec, work)
            counts[r["counter"]] += 1
            if r.get("timing"):
                timing.append(r["timing"])
            if r.get("rejected"):
                rejected.append(r["rejected"])
            if r.get("item"):
                items.append(r["item"])
                good += 1
        baseline = None
        if not args.no_baseline:
            dec = ds.decide_transition(sel["ctx"])
            iid = f"{gid}__z_ref_{dec['type_key']}"
            spec = {"mode": "baseline", "out": str(out), "gid": gid, "iid": iid, "dec": dec, "plan": plan,
                    "a": sel["a"], "b": sel["b"], "src_a": src_of(sel["a"]), "src_b": src_of(sel["b"]), "prov": prov}
            r = run_child(spec, work)
            counts[r["counter"]] += 1
            if r.get("rejected"):
                rejected.append(r["rejected"])
            if r.get("item"):
                items.append(r["item"])
            baseline = r.get("baseline")
        ctx = sel["ctx"]
        import gc

        gc.collect()
        groups.append({
            "id": gid,
            "pair": {"a": {"slug": A.slug, "title": A.title, "artist": A.analysis.get("artist")},
                     "b": {"slug": B.slug, "title": B.title, "artist": B.analysis.get("artist")}},
            "title": f"{A.title} → {B.title}",
            "bpm": {"a": A.analysis["tempo"]["bpm"], "b": B.analysis["tempo"]["bpm"], "stretch_b_pct": comp["tempo"]["stretch_b_pct"]},
            "key": {"a": A.analysis["key"]["camelot"], "b": B.analysis["key"]["camelot"], "a_name": A.analysis["key"]["key"],
                    "b_name": B.analysis["key"]["key"], "relation": comp["key"]["relation"], "key_shift_b_if_needed": sel["key_shift"]},
            "A_out_s": plan["a_out_s"], "B_in_s": plan["b_in_s"], "A_out_bar": plan["a_out_bar"], "B_in_bar": plan["b_in_bar"],
            "sections": {"a": plan["a_section"], "b": plan["b_section"]},
            "plan": {"type": plan["type"], "blend_bars": plan["blend_bars"], "score": plan["score"], "confidence": plan["confidence"]},
            "reasons": list(comp["reasons"]) + list(plan["reasons"]),
            "policy": {"context_rules": sel["context_rules"], "proposals": [{k: p[k] for k in ("technique", "rule", "duration_bars", "prior", "score", "why")}
                                                                            for p in props],
                       "inputs": sel["proposals"]["trace"][0]["inputs"]},
            "lyric_links": [{"a": l["a_word"].get("text"), "b": l["b_word"].get("text"), "match": l["match"]["type"], "score": l["score"]} for l in sel["links"]],
            "baseline": baseline,
            "fade_check": sel.get("fade_check"),
            "items": items,
        })
    manifest = {
        "schema_version": SCHEMA,
        "title": "MusicMix — tecniche creative v1: loop, acapella, vocal swap, build, nota ponte, freeze, riverbero, backspin, tape stop, lyric link",
        "run": RUN,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"plans": "dj_select.plan_pair on runs/dj_plan_v1 + CC analyses (run_dj_set_v1.discover)",
                   "renderer": cr.DJ_CREATIVE_VERSION, "policy": pol.POLICY_VERSION,
                   "knowledge": "conoscenza_mix/_meta/TECNICHE_CREATIVE_2026-10-21.md"},
        "conventions": {
            "audio_path": "relative to the repository root",
            "markers": "seconds in the clip: transition_start_s / switch_s / transition_end_s from the renderer metadata (overlap, switch)",
            "clip_rules": "10 s before the transition start + whole transition + 10 s after its end (AGENTS.md, 20/10/2026)",
            "grids": "dj_select regularised grids (tracks/<slug>.json), not raw Beat This!",
            "qa": "dj_qa.evaluate_render with decks, the vocal actually heard from each deck in the clip, ASR words of A mapped with dj_creative.words_in_clip, Beat This! small0 and grid provenance; fail = discarded",
            "reference": "role z = standard transition of the dj_set tree on the same cue (for comparison, not a creative technique)",
            "no_stay_items": "continuing A is not offered as an item to rate (AGENTS.md)",
            "prior": "policy prior/score only orders proposals; it is not a quality estimate",
        },
        "counts": counts,
        "groups": groups,
        "rejected": rejected,
    }
    (out / "review_manifest.json").write_text(json.dumps(jsonable(manifest), indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "gallery_log.json").write_text(json.dumps(jsonable({"log": log, "counts": counts, "rejected": rejected, "timing": timing,
                                                               "elapsed_s": time.time() - t0, "config": CFG.__dict__}),
                                                     indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    if not any((out / "rejected").iterdir()):
        shutil.rmtree(out / "rejected")
    import resource

    print(json.dumps(counts), f"{time.time() - t0:.0f} s", f"max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
