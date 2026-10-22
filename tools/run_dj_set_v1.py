#!/usr/bin/env python3
"""Set DJ continuo v1: ordine dei brani, tipo di ogni transizione, render e QA.

Legge (senza modificarli) le analisi DJ già fatte e scrive soltanto in una
cartella NUOVA (default ``runs/dj_set_v1``) e nella cache ``.cache/dj_set_v1``:

* brani: ``runs/dj_plan_v1/tracks`` (pool V12) + ``runs/dj_plan_cc_v1/tracks``
  (pool CC) + i brani CC con struttura, voce e provider già completi ma non
  ancora pianificati (analizzati qui con ``tools/dj_plan_pool.analyse_track``,
  la stessa funzione della pipeline, in cache). L'analisi CC in corso
  (``runs/cloud_session/pool_cc_v1_analysis.log``) non viene toccata: a fine
  analisi basta rilanciare con ``--out`` nuovo (o ``--overwrite``);
* per ogni brano: MFCC medi/deviazione (timbro), densità di onset, flusso
  spettrale e centroide (energia relativa, [H]), loudness prima di ogni
  downbeat (uscita in dissolvenza, regola della galleria v1) e guardia vocale
  dei cut (stem + parole ASR, regola della galleria v2);
* candidati per ogni coppia ordinata: ``dj_select.plan_pair`` + albero di
  ``dj_set.decide_transition``; ordine con ``dj_set.order_set`` (Held–Karp);
  cue lungo il set con ``dj_set.choose_cues`` (Viterbi, 60–120 s per brano);
* render di ogni transizione con ``dj_render`` / ``dj_effects`` (section gain
  match se disponibile), QA con ``dj_qa.evaluate_render``; se fallisce si
  provano le alternative dell'albero sullo stesso cue, poi altri cue
  compatibili con i vicini; tutto registrato;
* giunte: tra due transizioni il brano suona dall'audio intero normalizzato
  (stesso guadagno LUFS della clip, section gain che continua la sua rampa),
  con crossfade di 10 ms su campioni equivalenti. Dopo un blend B torna al
  suo tempo originale sul primo downbeat dopo la fine della transizione
  (salto dichiarato in ``tempo_reset_pct``) [H, limite noto].

Uscite: ``set.wav`` (PCM 24 bit), ``set.mp3``, ``tracklist.txt``,
``tracklist.csv``, ``set.json`` (parametri con fonte, ordine e costi, cue,
decisioni dell'albero con traccia, tentativi e QA), ``transitions/*.json``.

    python tools/run_dj_set_v1.py [--out runs/dj_set_v1] [--length 10]
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import run_dj_gallery_v1 as v1  # noqa: E402  (frozen helpers: Track, fade check, regular grid)

from musicmix import dj_effects as fx  # noqa: E402
from musicmix import dj_qa  # noqa: E402
from musicmix import dj_render as dj  # noqa: E402
from musicmix import dj_select as djs  # noqa: E402
from musicmix import dj_set as ds  # noqa: E402
from musicmix.paths import localize  # noqa: E402

RUN = "dj_set_v1"
SCRIPT = "tools/run_dj_set_v1.py"
SR = 48000
CACHE = ROOT / ".cache" / "dj_set_v1"
V12_PLAN = ROOT / "runs/dj_plan_v1"
CC_PLAN = ROOT / "runs/dj_plan_cc_v1"
CC_POOL = ROOT / "runs/pool_cc_v1"
CC_LOG = ROOT / "runs/cloud_session/pool_cc_v1_analysis.log"
XFADE = int(0.010 * SR)  # 10 ms splice crossfade between equivalent samples [H]
FADE_IN_S = 0.5  # set start when the first track does not start at 0 [H]
FADE_OUT_BARS = 2  # set end when the last track is cut on a phrase downbeat [H]
MAX_ATTEMPTS = 10
GUARD_POST_S = dj_qa.WORD_GUARD_S
GUARD_ASR_S = dj_qa.WORD_GUARD_S


def section_gain_cfg():
    if hasattr(dj, "SectionGainConfig"):
        return dj.SectionGainConfig(enabled=True, ramp_bars=8)  # 8 bars [H]: return to -14 LUFS well before the next exit
    return None


SG = section_gain_cfg()
_kw = {"section_gain": SG} if SG is not None else {}
BLEND_CFG = dj.DJRenderConfig(cue_convention="start", **_kw)
CUT_CFG = dj.DJRenderConfig(**_kw)
FX_CFG = fx.DJEffectsConfig(**({"section_gain": SG} if SG is not None and "section_gain" in fx.DJEffectsConfig.__dataclass_fields__ else {}))


# -------------------------------------------------------------------- utils
def sha256(path: Path) -> str:
    return v1.sha256(Path(path))


def jsonable(o: Any) -> Any:
    return dj_qa._jsonable(o)


def dump(path: Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(jsonable(obj), indent=1, ensure_ascii=False), encoding="utf-8")


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def fmt_t(s: float) -> str:
    m, r = divmod(max(0.0, s), 60.0)
    return f"{int(m):02d}:{r:06.3f}"


# ------------------------------------------------------------ track sources
def providers_complete(slug: str) -> bool:
    p = CC_PLAN / "providers" / f"{slug}.json"
    if not p.is_file():
        return False
    try:
        rec = json.loads(p.read_text())
    except json.JSONDecodeError:
        return False
    return all(rec.get(k, {}).get("status") == "complete" for k in ("allin1", "madmom_key", "essentia_key"))


def evidence_ok(entry: dict) -> bool:
    for ev in ("structure", "vocals"):
        p = Path(entry[ev + "_path"])
        if not p.is_file():
            return False
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        if (doc.get("track_id"), doc.get("sample_rate"), doc.get("duration_samples")) != (
                entry["track_id"], entry["sample_rate"], entry["duration_samples"]):
            return False
    from musicmix.stems import StemError, vocal_stem_path  # 21/10/2026: niente nome indovinato

    try:
        vocal_stem_path(entry, check_sha=False)
    except StemError:
        return False
    return True


def discover(log: list) -> dict[str, dict]:
    """slug -> {plan_dir, catalog, source}; analyses CC tracks complete but not yet planned."""
    out: dict[str, dict] = {}
    pairs_doc = v1.load_json(V12_PLAN / "pairs.json")
    cat_v12_path = localize(pairs_doc["catalog"])
    if not cat_v12_path.is_absolute():
        cat_v12_path = ROOT / cat_v12_path
    cat_v12 = v1.load_json(cat_v12_path)
    for f in sorted((V12_PLAN / "tracks").glob("*.json")):
        out[f.stem] = {"plan_dir": V12_PLAN, "catalog": cat_v12, "catalog_path": cat_v12_path, "source": "pool_v12 (runs/dj_plan_v1)"}
    cat_cc_path = CC_POOL / "catalog_initial.json"
    cat_cc = v1.load_json(cat_cc_path)
    pending, analysed = [], []
    for entry in cat_cc["tracks"]:
        slug = entry["slug"]
        if (CC_PLAN / "tracks" / f"{slug}.json").is_file():
            out[slug] = {"plan_dir": CC_PLAN, "catalog": cat_cc, "catalog_path": cat_cc_path, "source": "pool_cc_v1 (runs/dj_plan_cc_v1)"}
            continue
        if not (providers_complete(slug) and evidence_ok(entry)):
            pending.append(slug)
            continue
        cached = CACHE / "tracks" / f"{slug}.json"
        key = {"track_id": entry["track_id"], "providers_sha256": sha256(CC_PLAN / "providers" / f"{slug}.json"),
               "structure_sha256": sha256(Path(entry["structure_path"])), "vocals_sha256": sha256(Path(entry["vocals_path"])),
               "dj_select": djs.SCHEMA_VERSION}
        keyf = cached.with_suffix(".key.json")
        if not (cached.is_file() and keyf.is_file() and v1.load_json(keyf) == key):
            import dj_plan_pool  # tools/dj_plan_pool.py (same analysis as the pipeline)

            print(f"analisi DJ di {slug} (provider completi, non ancora in dj_plan_cc_v1)", flush=True)
            a = dj_plan_pool.analyse_track(entry, CC_PLAN / "providers")
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(dj_plan_pool.dump(a), encoding="utf-8")
            dump(keyf, key)
        analysed.append(slug)
        out[slug] = {"plan_dir": CACHE, "catalog": cat_cc, "catalog_path": cat_cc_path,
                     "source": "pool_cc_v1 (analizzato qui con dj_plan_pool.analyse_track, .cache/dj_set_v1/tracks)"}
    log.append({"sources": {"v12": sorted(s for s, v in out.items() if v["plan_dir"] == V12_PLAN),
                            "cc_planned": sorted(s for s, v in out.items() if v["plan_dir"] == CC_PLAN),
                            "cc_analysed_here": analysed, "cc_pending": len(pending),
                            "cc_log_tail": CC_LOG.read_text(encoding="utf-8").splitlines()[-3:] if CC_LOG.is_file() else None}})
    return out


# ----------------------------------------------------------- per-track info
def track_info(slug: str, src: dict) -> dict:
    """Timbre/energy features, fade check and cut guard per downbeat (cached)."""
    an_path = src["plan_dir"] / "tracks" / f"{slug}.json"
    an = v1.load_json(an_path)
    key = {"track_id": an["track_id"], "analysis_sha256": sha256(an_path), "version": 1}
    cp = CACHE / "info" / f"{slug}.json"
    if cp.is_file():
        doc = v1.load_json(cp)
        if doc.get("key") == key:
            return doc
    import librosa
    from scipy.signal import resample_poly

    print(f"feature di {slug}", flush=True)
    T = v1.Track(slug, src["plan_dir"], src["catalog"])
    gain = 10 ** ((-14.0 - dj.integrated_lufs(T.audio, SR)) / 20.0)
    mono = resample_poly(T.audio.mean(axis=1) * gain, 22050, SR).astype(np.float32)
    mf = librosa.feature.mfcc(y=mono, sr=22050, n_mfcc=20, hop_length=512)
    onset_env = librosa.onset.onset_strength(y=mono, sr=22050, hop_length=512)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=22050, hop_length=512)
    cent = librosa.feature.spectral_centroid(y=mono, sr=22050, hop_length=512)
    dur = len(mono) / 22050
    # per-downbeat fade check (gallery v1 rule) and cut guard (gallery v2 rule)
    beat = float(an["tempo"]["period_s"])
    act, _ = dj_qa.vocal_activity(T.vocals, SR, None, T.vocal_threshold)
    times = (np.arange(len(act)) + 0.5) * dj_qa.VOCAL_HOP_S
    words = [(w["start_sample"] / SR, w["end_sample"] / SR) for w in T.words]
    fade, guard = [], []
    for t in T.downbeats:
        try:  # only exits matter (>= 35 % of the track): the first seconds cannot be measured
            fade.append(bool(T.fade_check(float(t))["in_fade"]) if t > v1.FADE_WINDOW_S + 0.5 else False)
        except ValueError:
            fade.append(False)
        stem_clear = djs.vocal_clear_window(times, act, float(t), beat, GUARD_POST_S)
        asr = any(e > t - GUARD_ASR_S and s < t + GUARD_ASR_S for s, e in words)
        guard.append(bool(stem_clear and not asr))
    doc = {"key": key, "slug": slug, "title": an.get("title"), "artist": an.get("artist"),
           "timbre": np.concatenate([mf.mean(axis=1), mf.std(axis=1)]).tolist(),
           "energy_features": {"onset_rate": float(len(onsets) / dur), "flux_mean": float(onset_env.mean()),
                               "centroid_mean": float(cent.mean())},
           "downbeats_s": [float(x) for x in T.downbeats], "in_fade": fade, "cut_clear": guard,
           "st_median_lufs": T.st_median}
    cp.parent.mkdir(parents=True, exist_ok=True)
    dump(cp, doc)
    return doc


def bar_flag(info: dict, t: float, name: str) -> bool:
    d = np.asarray(info["downbeats_s"])
    j = int(np.argmin(np.abs(d - t)))
    return bool(info[name][j]) if abs(d[j] - t) < 0.05 else False


def apply_filters(pc: dict, info_a: dict) -> dict:
    """Uscita in dissolvenza -> candidato non ammesso; cut con voce/parola di A
    sull'uscita -> prima alternativa non-cut dell'albero (uscita con effetto)."""
    for c in pc["candidates"]:
        a_out = c["plan"]["a_out_s"]
        if bar_flag(info_a, a_out, "in_fade"):
            c["admissible"] = False
            c["notes"].append("A in dissolvenza nei 4 s prima di a_out (> 6 dB sotto la mediana): regola galleria v1")
            continue
        d = c["decision"]
        clear = bar_flag(info_a, a_out, "cut_clear")
        if not clear:
            d["fallbacks"] = [f for f in d["fallbacks"] if f["type"] != "cut"]
            if d["type"] == "cut":
                alt = d["fallbacks"][0] if d["fallbacks"] else ds._mk("echo_out", ds.values(ds.TREE_PARAMS))
                rest = d["fallbacks"][1:]
                c["notes"].append("cut non ammesso: voce (stem) o parola ASR di A sull'uscita (guardia galleria v2) -> "
                                  + alt["type"])
                c["decision"] = {**d, **alt, "type_key": alt.get("type_key", alt["type"]), "fallbacks": rest,
                                 "rule": d["rule"] + "+cut_guard"}
        c["cost"] = ds.candidate_cost(c["plan"], c["decision"])
    pc["candidates"].sort(key=lambda c: (not c["admissible"], c["cost"], -c["plan"]["score"]))
    return pc


# ------------------------------------------------------------------- render
def render(dec: dict, A, B, plan: dict, prov: dict):
    grids = dict(beats_a=A.beats, downbeats_a=A.downbeats, beats_b=B.beats, downbeats_b=B.downbeats)
    ia = int(np.argmin(np.abs(A.downbeats - plan["a_out_s"])))
    a_out, b_in = float(A.downbeats[ia]), float(plan["b_in_s"])
    kind = dec["type"]
    r = dec["render"]
    if kind == "blend":
        res = dj.render_transition(A.audio, B.audio, **grids, cue_out_a=a_out, cue_in_b=b_in, kind="blend", bars=r["bars"],
                                   sample_rate=SR, vocals_a=A.vocals, vocals_b=B.vocals, config=BLEND_CFG, provenance=prov,
                                   return_decks=True)
    elif kind == "cut":
        res = dj.render_transition(A.audio, B.audio, **grids, cue_out_a=a_out, cue_in_b=b_in, kind="cut", sample_rate=SR,
                                   config=CUT_CFG, provenance=prov, return_decks=True)
    elif kind == "echo_out":
        if ia < r["entry_bars"]:
            raise dj.DJRenderError("echo_out: no A downbeat before a_out")
        echo_at = float(A.downbeats[ia - r["entry_bars"]])
        res = fx.echo_out(A.audio, B.audio, **grids, cue_out_a=echo_at, cue_in_b=b_in, delay_beats=r["delay_beats"],
                          entry_bars=r["entry_bars"], sample_rate=SR, config=FX_CFG, provenance=prov, return_decks=True)
    elif kind in ("filter_out", "filter_into_chorus"):
        res = fx.filter_out(A.audio, B.audio, **grids, cue_out_a=a_out, cue_in_b=b_in, sweep_bars=r["sweep_bars"],
                            sample_rate=SR, config=FX_CFG, provenance=prov, return_decks=True)
    else:
        raise ValueError(kind)
    meta = res.metadata
    decks = dict(res.decks or {})
    n = len(res.audio)
    t = np.arange(n)
    if kind == "echo_out":
        dry_end = int(meta["effect"]["dry_a_cut_output_sample"])
        entry = int(meta["switch"]["output_sample"])
        decks["gains"] = dict(decks.get("gains", {}), fader_a=(t < dry_end).astype(float), fader_b=(t >= entry).astype(float))
    elif kind in ("filter_out", "filter_into_chorus"):
        cut = int(meta["switch"]["output_sample"])
        decks["gains"] = dict(decks.get("gains", {}), fader_a=(t < cut).astype(float), fader_b=(t >= cut).astype(float))
    return res, decks


_QA_HAS_GRID = "grid_support" in inspect.signature(dj_qa.evaluate_render).parameters


def run_qa(res, A, B, decks, tracker) -> dict:
    n = len(res.audio)
    meta = res.metadata
    va, vb = dj_qa.stems_to_output(meta, n, A.vocals, B.vocals)
    words = dj_qa.words_to_output(meta, A.words, n)
    kw = {"grid_support": {"a": A.analysis["tempo"], "b": B.analysis["tempo"]}} if _QA_HAS_GRID else {}
    return dj_qa.evaluate_render(res.audio.astype(np.float64), meta, sample_rate=res.sample_rate, decks=decks,
                                 vocals_a=va, vocals_b=vb, vocal_thresholds=(A.vocal_threshold, B.vocal_threshold),
                                 words_a=words, tracker=tracker, **kw)


# --------------------------------------------------------------- assembly
def limit_long(x: np.ndarray, chunk_s: float = 30.0, margin_s: float = 0.2) -> tuple[np.ndarray, dict]:
    """dj_render.true_peak_limit a blocchi (memoria); solo se il true peak supera il tetto."""
    n = len(x)
    if n == 0:
        return x, {"applied": False}
    c, m = int(chunk_s * SR), int(margin_s * SR)
    peaks = [dj.true_peak_dbtp(x[s:s + c], SR) for s in range(0, n, c)]
    if max(peaks) <= -1.0:
        return x, {"applied": False, "true_peak_dbtp": max(peaks)}
    out = np.empty_like(x)
    red = 0.0
    for s in range(0, n, c):
        lo, hi = max(0, s - m), min(n, s + c + m)
        y, info = dj.true_peak_limit(x[lo:hi], SR, -1.0)
        red = max(red, info.get("max_gain_reduction_db", 0.0) or 0.0)
        out[s:min(n, s + c)] = y[s - lo:s - lo + min(c, n - s)]
    return out, {"applied": True, "true_peak_before_dbtp": max(peaks), "max_gain_reduction_db": red}


class Piece:
    """Tratto del set: ``audio`` più ``tail`` (XFADE campioni equivalenti oltre la fine)."""

    def __init__(self, audio: np.ndarray, tail: np.ndarray, label: str, info: dict):
        self.audio, self.tail, self.label, self.info = audio, tail, label, info


def raw_piece(T, gain: float, s0: int, s1: int, label: str, sg: dict | None = None, sg_offset: int = 0) -> Piece:
    """Audio intero normalizzato di T in [s0, s1) (+ coda), con eventuale rampa di section gain."""
    n = len(T.audio)
    s0, s1 = max(0, s0), min(n, s1)
    seg = T.audio[s0:min(n, s1 + XFADE)] * gain
    if sg and sg.get("enabled") and (sg.get("applied_db") or 0.0) != 0.0 and sg.get("after", "ramp") == "ramp":
        ramp = sg["ramp"]
        r0, r1 = ramp["output_start_sample"], ramp["output_end_sample"]
        curve = dj.section_gain_curve(len(seg), sg["applied_db"], r0 - sg_offset, r1 - r0, "ramp")
        seg = seg * curve[:, None]
    body, tail = seg[: s1 - s0], seg[s1 - s0:]
    return Piece(body, tail, label, {"kind": "raw", "slug": T.slug, "source_samples": [s0, s1], "gain_db": 20 * math.log10(gain)})


def assemble(order, tracks_loader, results, first, last) -> tuple[np.ndarray, list, list]:
    """Concatena raw A0 | clip0 | raw B | clip1 | ... | raw ultimo; restituisce audio, pezzi, marcatori."""
    pieces: list[Piece] = []
    clip_origin = []  # set sample of each clip's sample 0 (filled after concatenation)
    T0 = tracks_loader(order[0])
    r0 = results[0]
    g0 = r0["geo"]
    ga = 10 ** (r0["meta"]["loudness"]["a_gain_db"] / 20.0)
    s_start = int(round(first["start_s"] * SR))
    pieces.append(raw_piece(T0, ga, s_start, g0["a_offset"], f"{order[0]} (inizio)"))
    q_in = 0
    splices = []
    for t, r in enumerate(results):
        g, clip = r["geo"], r["audio"]
        B = tracks_loader(order[t + 1])
        # splice out of the clip: first B downbeat after the pure-B start
        db = B.downbeats * SR
        pos = np.array([ds.clip_pos_of_b(g, x) for x in db])
        ok = np.where((pos >= g["b_pure_start"] + 0.25 * SR) & (pos <= g["n"] - XFADE - 0.5 * SR))[0]
        if ok.size:
            k = int(ok[0])
            p_out, b_src = int(round(pos[k])), int(round(db[k]))
        else:  # no downbeat left in the clip: splice at the last safe sample
            p_out = int(g["n"] - XFADE - 1)
            b_src = int(round(ds.b_src_of_clip(g, p_out)))
        reset = 100.0 * (g["b_rate"] - 1.0)
        pieces.append(Piece(clip[q_in:p_out], clip[p_out:p_out + XFADE], f"transizione {t + 1}",
                            {"kind": "clip", "transition": t, "clip_samples": [q_in, p_out]}))
        clip_origin.append((len(pieces) - 1, q_in))
        gb = 10 ** (r["meta"]["loudness"]["b_gain_db"] / 20.0)
        sg = r["meta"].get("section_gain")
        if t + 1 < len(results):
            nxt = results[t + 1]["geo"]
            a_off = nxt["a_offset"]
            if b_src <= a_off:
                pieces.append(raw_piece(B, gb, b_src, a_off, order[t + 1], sg, p_out))
                q_in = 0
            elif b_src - a_off <= nxt["a_pure_end"] - XFADE:
                q_in = b_src - a_off  # direct clip -> clip splice inside the pure regions
            else:
                raise RuntimeError(f"no room to splice {order[t + 1]}: B source {b_src} after next transition start")
            splices.append({"after_transition": t, "b_source_sample": b_src, "clip_out_sample": p_out,
                            "next_clip_in_sample": q_in, "tempo_reset_pct": reset if abs(reset) > 1e-9 else 0.0,
                            "raw_seconds": max(0, a_off - b_src) / SR})
        else:
            s_end = int(round(last["end_s"] * SR))
            pc = raw_piece(B, gb, b_src, s_end, f"{order[t + 1]} (fine)", sg, p_out)
            if not last["to_track_end"]:
                nf = min(len(pc.audio), int(FADE_OUT_BARS * 4 * B.analysis["tempo"]["period_s"] * SR))
                pc.audio[-nf:] *= np.linspace(1.0, 0.0, nf)[:, None]
                pc.tail = np.zeros_like(pc.tail)
            pieces.append(pc)
            splices.append({"after_transition": t, "b_source_sample": b_src, "clip_out_sample": p_out,
                            "tempo_reset_pct": reset if abs(reset) > 1e-9 else 0.0})
    # limit raw pieces (clips are already limited by the renderer)
    for p in pieces:
        if p.info["kind"] == "raw":
            both = np.concatenate([p.audio, p.tail]) if len(p.tail) else p.audio
            lim, info = limit_long(both)
            lim = lim.astype(np.float32)
            p.audio, p.tail = lim[: len(p.audio)], lim[len(p.audio):]
            p.info["limiter"] = info
    if not first["from_track_start"]:
        nf = int(FADE_IN_S * SR)
        pieces[0].audio[:nf] *= np.linspace(0.0, 1.0, nf)[:, None]
    # concatenate with 10 ms crossfades between equivalent samples
    total = sum(len(p.audio) for p in pieces)
    out = np.zeros((total, 2), dtype=np.float32)
    starts, pos = [], 0
    for i, p in enumerate(pieces):
        a = p.audio.astype(np.float32)
        starts.append(pos)
        out[pos:pos + len(a)] = a
        if i > 0 and len(pieces[i - 1].tail) and len(a):
            m = min(XFADE, len(pieces[i - 1].tail), len(a))
            w = np.linspace(0.0, 1.0, m, dtype=np.float32)[:, None]
            out[pos:pos + m] = pieces[i - 1].tail[:m].astype(np.float32) * (1 - w) + a[:m] * w
        pos += len(a)
    origins = [starts[pi] - q for pi, q in clip_origin]
    seg_info = [{**p.info, "label": p.label, "set_start_s": s / SR, "set_end_s": (s + len(p.audio)) / SR}
                for p, s in zip(pieces, starts)]
    return out, seg_info, [{"clip_origin_set_sample": o, **sp} for o, sp in zip(origins, splices)]


def junction_check(x: np.ndarray, seg_info: list) -> list:
    """Salto di loudness short-term (3 s prima / dopo) a ogni giunta: deve essere ~0."""
    rows = []
    for s in seg_info[1:]:
        p = int(s["set_start_s"] * SR)
        w = int(3 * SR)
        if p - w < 0 or p + w > len(x):
            continue
        a = dj.integrated_lufs(x[p - w:p].astype(np.float64), SR)
        b = dj.integrated_lufs(x[p:p + w].astype(np.float64), SR)
        rows.append({"set_s": p / SR, "label": s["label"], "lufs_before": a, "lufs_after": b,
                     "jump_db": (b - a) if np.isfinite(a) and np.isfinite(b) else None})
    return rows


# ---------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "runs" / RUN))
    ap.add_argument("--overwrite", action="store_true", help="sovrascrive una cartella di uscita esistente")
    ap.add_argument("--length", type=int, default=None, help="numero di brani obiettivo (8-15, default 10)")
    ap.add_argument("--play-min", type=float, default=None)
    ap.add_argument("--play-max", type=float, default=None)
    ap.add_argument("--start", default=None, help="slug del primo brano (facoltativo)")
    ap.add_argument("--end", default=None, help="slug dell'ultimo brano (facoltativo)")
    ap.add_argument("--no-render", action="store_true", help="solo ordine e cue (niente audio)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{out} esiste e non è vuota: usare --out nuovo o --overwrite")
        shutil.rmtree(out)
    (out / "transitions").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log: list = []
    order_over = {k: v for k, v in {"len_target": args.length}.items() if v is not None}
    cue_over = {k: v for k, v in {"play_min_s": args.play_min, "play_max_s": args.play_max}.items() if v is not None}
    if args.play_min or args.play_max:
        lo = cue_over.get("play_min_s", 60.0)
        hi = cue_over.get("play_max_s", 120.0)
        cue_over["play_target_s"] = 0.5 * (lo + hi)

    # 1. tracks -------------------------------------------------------------
    sources = discover(log)
    analyses = {s: v1.load_json(src["plan_dir"] / "tracks" / f"{s}.json") for s, src in sources.items()}
    infos = {s: track_info(s, src) for s, src in sources.items()}
    slugs = sorted(analyses)
    print(f"{len(slugs)} brani analizzati: {', '.join(slugs)}", flush=True)

    # 2. candidates, edges, order ----------------------------------------------
    timbre = ds.timbre_distances({s: infos[s]["timbre"] for s in slugs}, slugs)
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
            pc = apply_filters(ds.candidates_for_pair(analyses[a], analyses[b]), infos[a])
            pair_cands[(a, b)] = pc
            e = ds.edge_cost(pc, analyses[a]["tempo"]["bpm"], analyses[b]["tempo"]["bpm"], float(timbre[i, j]),
                             energy[a], energy[b], order_over)
            edges[(a, b)] = e
            E[i, j] = e["total"]
    feasible_pairs = int(np.isfinite(E).sum())
    blend_pairs = sum(1 for k, pc in pair_cands.items() if any(c["admissible"] and c["decision"]["type"] == "blend" for c in pc["candidates"]))
    print(f"coppie ordinate ammissibili {feasible_pairs}/{len(slugs) * (len(slugs) - 1)}, con almeno un blend {blend_pairs}", flush=True)
    ordering = ds.order_set(slugs, E, energy, order_over, args.start, args.end)
    order = ordering["order"]
    if len(order) < 2:
        raise SystemExit("nessun cammino ammissibile")
    breakdown = ds.path_breakdown(order, edges, energy, order_over)
    print(f"ordine ({ordering['length']} brani, costo {ordering['cost']:.3f}): {' -> '.join(order)}", flush=True)

    # 3. cues along the set --------------------------------------------------
    cues = ds.choose_cues(order, analyses, pair_cands, cue_over)
    chosen = cues["chosen"]
    for t, c in enumerate(chosen):
        print(f"  {t + 1}. {order[t]} -> {order[t + 1]}: {c['decision']['label']} ({c['decision']['rule']}), "
              f"A battuta {c['plan']['a_out_bar']}, B battuta {c['plan']['b_in_bar']}", flush=True)

    doc: dict[str, Any] = {
        "schema_version": ds.SCHEMA_VERSION, "run": RUN, "script": SCRIPT,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"dj_set": ds.SCHEMA_VERSION, "dj_select": djs.SCHEMA_VERSION, "dj_render": dj.DJ_RENDER_VERSION,
                     "dj_effects": fx.DJ_EFFECTS_VERSION, "dj_qa": dj_qa.DJ_QA_VERSION},
        "params": {"tree": ds.documented(ds.TREE_PARAMS), "type_cost": ds.documented(ds.TYPE_COST),
                   "order": ds.documented(ds.ORDER_PARAMS, order_over), "cues": ds.documented(ds.CUE_PARAMS, cue_over),
                   "tree_rules": ds.TREE_RULES,
                   "render": {"section_gain": SG.__dict__ if SG is not None else None, "blend_cue_convention": "start",
                              "splice_crossfade_ms": 1000 * XFADE / SR, "fade_in_s": FADE_IN_S, "fade_out_bars": FADE_OUT_BARS},
                   "energy_feature": "[H] media degli z-score (sul pool) di densità di onset, flusso spettrale medio e centroide medio, audio a -14 LUFS; poi rango percentile in [0,1]",
                   "timbre_feature": "[S] Bittner 2017 / [E] Kell & Tzanetakis 2013: MFCC 20 (media e deviazione) standardizzati sul pool, distanza euclidea / sqrt(40)"},
        "sources": {s: {"source": src["source"], "analysis": rel(src["plan_dir"] / "tracks" / f"{s}.json"),
                        "analysis_sha256": sha256(src["plan_dir"] / "tracks" / f"{s}.json"),
                        "bpm": analyses[s]["tempo"]["bpm"], "tempo_confidence": analyses[s]["tempo"]["confidence"],
                        "camelot": analyses[s]["key"]["camelot"], "key_confidence": analyses[s]["key"]["confidence"],
                        "energy": energy[s], "energy_raw": energy_raw[s], "energy_features": feats[s],
                        "title": infos[s].get("title"), "artist": infos[s].get("artist")} for s, src in sources.items()},
        "pool": {"n_tracks": len(slugs), "feasible_ordered_pairs": feasible_pairs, "pairs_with_blend": blend_pairs,
                 "edge_matrix": {"slugs": slugs, "cost": [[None if not math.isfinite(x) else x for x in row] for row in E]}},
        "ordering": {**ordering, "breakdown": breakdown},
        "cues": {"cost": cues["cost"], "play_s": cues["play_s"], "play_bounds_s": cues["play_bounds_s"],
                 "relaxations": cues["relaxations"], "first": cues["first"], "last": cues["last"]},
        "log": log,
    }
    if args.no_render:
        dump(out / "set.json", doc)
        print(f"solo piano: {out / 'set.json'}")
        return 0

    # 4. render each transition with QA and fallbacks ---------------------------
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
        cp = ds.values(ds.CUE_PARAMS, cue_over)

        def fits(c, dec):
            cc = {"plan": c["plan"], "decision": dec}
            if prev_b_in is not None:
                if (c["plan"]["a_out_s"] - ds.lead_a_s(cc, analyses[a])) - (prev_b_in + prev_ovl) < cp["min_gap_s"]:
                    return False
            if nxt is not None and (nxt[0] - nxt[1]) - (c["plan"]["b_in_s"] + ds.overlap_b_s(cc, analyses[b])) < cp["min_gap_s"]:
                return False
            return True

        attempts_plan = [(idx0, chosen[t]["decision"], "scelta del Viterbi")]
        attempts_plan += [(idx0, fb, "alternativa dell'albero sullo stesso cue") for fb in chosen[t]["decision"]["fallbacks"]]
        for i in ds.alternatives(t, order, analyses, cands, prev_b_in, nxt, cue_over, exclude=[idx0]):
            c = cands[i]
            attempts_plan.append((i, c["decision"], "altro cue compatibile con i vicini"))
            attempts_plan += [(i, fb, "alternativa dell'albero su altro cue") for fb in c["decision"]["fallbacks"]]
        tried, accepted, best_fail = [], None, None
        for i, dec, why in attempts_plan:
            if len(tried) >= MAX_ATTEMPTS:
                break
            c = cands[i]
            plan = c["plan"]
            if not fits(c, dec):
                tried.append({"candidate": i, "type": dec["type_key"], "why": why, "skipped": "spazio insufficiente con le transizioni vicine"})
                continue
            if any(x.get("candidate") == i and x.get("type") == dec["type_key"] and "verdict" in x for x in tried):
                continue
            prov = {"a": A.provenance, "b": B.provenance, "script": SCRIPT, "run": RUN, "transition": t,
                    "plan": {k: plan[k] for k in ("a_out_s", "a_out_bar", "b_in_s", "b_in_bar", "type", "blend_bars", "score")},
                    "decision": {k: dec.get(k) for k in ("type", "bars", "render", "rule")}}
            rec = {"candidate": i, "type": dec["type_key"], "label": dec["label"], "why": why,
                   "a_out_bar": plan["a_out_bar"], "b_in_bar": plan["b_in_bar"], "a_out_s": plan["a_out_s"], "b_in_s": plan["b_in_s"]}
            try:
                res, decks = render(dec, A, B, plan, prov)
            except (dj.DJRenderError, ValueError) as exc:
                counts["render_error"] += 1
                rec.update(verdict="render_error", reasons=[str(exc)])
                tried.append(rec)
                print(f"   RENDER ERROR {a}->{b} {dec['type_key']}: {exc}", flush=True)
                continue
            qa = run_qa(res, A, B, decks, tracker)
            counts[qa["verdict"]] += 1
            rec.update(verdict=qa["verdict"], reasons=[f"{r['check']}: {r['message']}" for r in qa["reasons"]])
            tried.append(rec)
            name = f"t{t + 1:02d}_{len(tried):02d}_{dec['type_key']}"
            dump(out / "transitions" / f"{name}.json", {"attempt": rec, "render": res.metadata, "qa": qa})
            print(f"   {qa['verdict'].upper():4s} {a} -> {b} {dec['label']} (A {plan['a_out_bar']}, B {plan['b_in_bar']}) "
                  + "; ".join(rec["reasons"]), flush=True)
            if qa["verdict"] != "fail":
                accepted = (i, dec, res, qa, rec)
                break
            nfail = sum(1 for r in qa["reasons"] if r["severity"] == "fail")
            if best_fail is None or nfail < best_fail[0]:
                best_fail = (nfail, (i, dec, res, qa, rec))
        kept_fail = False
        if accepted is None:
            if best_fail is None:
                raise SystemExit(f"transition {a} -> {b}: no renderable attempt ({tried})")
            accepted = best_fail[1]
            kept_fail = True
            print(f"   ATTENZIONE: {a} -> {b}: tutti i tentativi falliscono il QA, tenuto il meno grave", flush=True)
        i, dec, res, qa, rec = accepted
        c = cands[i]
        final_counts[qa["verdict"]] += 1
        prev_b_in = c["plan"]["b_in_s"]
        prev_ovl = ds.overlap_b_s({"plan": c["plan"], "decision": dec}, analyses[b])
        if i != idx0:
            chosen[t] = c
        geo = ds.clip_geometry(res.metadata, len(res.audio))
        results.append({"audio": res.audio, "meta": res.metadata, "geo": geo})
        transitions.append({
            "index": t + 1, "a": a, "b": b, "type": dec["type_key"], "label": dec["label"],
            "rule": dec["rule"], "rule_source": dec["rule_source"], "trace": dec.get("trace"),
            "context": c["context"], "notes": c["notes"],
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
        })

    # 5. assemble the set -----------------------------------------------------
    first = ds.first_start(analyses[order[0]], chosen[0]["plan"]["a_out_s"], ds.values(ds.CUE_PARAMS, cue_over))
    last = ds.last_end(analyses[order[-1]], chosen[-1]["plan"]["b_in_s"], ds.values(ds.CUE_PARAMS, cue_over))
    loaded.clear()
    cache: dict[str, Any] = {}

    def loader(slug):
        if slug not in cache:
            if len(cache) >= 2:
                cache.pop(next(iter(cache)))
            src = sources[slug]
            cache[slug] = v1.Track(slug, src["plan_dir"], src["catalog"])
        return cache[slug]

    audio, seg_info, splices = assemble(order, loader, results, first, last)
    cache.clear()
    for tr, sp, r in zip(transitions, splices, results):
        o = sp["clip_origin_set_sample"]
        g = r["geo"]
        tr["set_times_s"] = {"start": (o + g["transition_start"]) / SR, "switch": (o + g["switch"]) / SR,
                             "end": (o + g["transition_end"]) / SR}
        tr["splice"] = sp
    junctions = junction_check(audio, seg_info)
    tp = max(dj.true_peak_dbtp(audio[s:s + 30 * SR], SR) for s in range(0, len(audio), 30 * SR))
    import soundfile as sf

    wav = out / "set.wav"
    sf.write(str(wav), audio, SR, subtype="PCM_24")
    mp3 = out / "set.mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", "256k", str(mp3)], check=True)
    dur = len(audio) / SR

    # tracklist
    entries = []
    for k, slug in enumerate(order):
        start = 0.0 if k == 0 else transitions[k - 1]["set_times_s"]["switch"]
        end = dur if k == len(order) - 1 else transitions[k]["set_times_s"]["switch"]
        entries.append({"position": k + 1, "slug": slug, "artist": infos[slug].get("artist"), "title": infos[slug].get("title"),
                        "set_start_s": start, "set_end_s": end, "set_seconds": end - start,
                        "bpm": analyses[slug]["tempo"]["bpm"], "camelot": analyses[slug]["key"]["camelot"], "energy": energy[slug],
                        "source_play_s": cues["play_s"][k]})
    lines = [f"MusicMix — set DJ v1 ({len(order)} brani, {fmt_t(dur)}), {doc['created_at']}",
             f"ordine: costo {ordering['cost']:.3f} ({ordering['note']}); cue: costo {cues['cost']:.3f}", ""]
    for k, e in enumerate(entries):
        who = " - ".join(x.strip() for x in (e["artist"] or "", e["title"] or e["slug"]) if x and x.strip())
        lines.append(f"{e['position']:2d}. [{fmt_t(e['set_start_s'])}] {who}  ({e['bpm']:.1f} BPM, {e['camelot']}, energia {e['energy']:.2f}; "
                     f"{e['set_seconds']:.0f} s nel set)")
        if k < len(transitions):
            tr = transitions[k]
            st = tr["set_times_s"]
            lines.append(f"      -> transizione {tr['index']}: {tr['label']} [{tr['rule']}] inizio {fmt_t(st['start'])}, switch {fmt_t(st['switch'])}, "
                         f"fine {fmt_t(st['end'])}; QA {tr['qa']['verdict']}" + (" (fail tenuto)" if tr["qa"]["kept_despite_fail"] else ""))
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
        "tracklist": entries, "transitions": transitions, "segments": seg_info, "junctions": junctions,
        "qa_counts_attempts": counts, "qa_counts_final": final_counts,
        "output": {"wav": rel(wav), "wav_sha256": sha256(wav), "mp3": rel(mp3), "mp3_sha256": sha256(mp3),
                   "duration_s": dur, "sample_rate": SR, "true_peak_dbtp": tp,
                   "tracklist_txt": rel(out / "tracklist.txt"), "tracklist_csv": rel(out / "tracklist.csv")},
        "elapsed_s": time.time() - t0,
        "notes": [
            "integrità tecnica (QA) non è qualità musicale: il set non è stato ascoltato",
            "i 6 giudizi v1 entrano solo come priorità dichiarate [U] nei parametri, non come validazione",
            "dopo un blend B torna al tempo originale sul primo downbeat dopo la transizione (tempo_reset_pct nelle giunte)",
            "il primo brano parte dall'inizio se la durata rientra, l'ultimo finisce col brano o su un downbeat con fade di 2 battute",
        ],
    })
    dump(out / "set.json", doc)
    print(json.dumps({"durata": fmt_t(dur), "qa_finale": final_counts, "tentativi": counts}), f"{time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
