#!/usr/bin/env python3
"""Galleria creativa v2: mashup e transizioni che usano la musica dei due brani (``musicmix.dj_mashup``).

Fasi (riprendibili; scrive in ``runs/dj_mashup_v1``, ``.cache/dj_mashup_v1`` e nella cartella ``--page-out``):

* ``features``: per ogni brano del catalogo congelato del ponte (``runs/dj_set_v3_bridge/catalog_frozen.json``; solo inglese
  o italiano; esclusi ``times_real_scelta_j1393805``, ``porrazzo_glide_beat_j1398420``, il pool CC e i brani esclusi dal set
  v3: stem assenti, tempo ambiguo) cromagramma CQT sincronizzato ai battiti dello strumentale (mix - stem) e dello stem
  vocale, parole ASR e LUFS integrata (cache ``.cache/dj_mashup_v1/feat``).
* ``pairs``: per ogni coppia ordinata e ogni piano di ``dj_select.plan_pair`` (cue a inizio frase) la mashability
  (:func:`dj_mashup.harmonic_match` su TIV, trasposizioni -3..+3, :func:`dj_mashup.mashability_score`) e le condizioni
  delle tecniche (verso di A che parte o finisce al cue, frase di B all'hook); si tengono 10 coppie con la mashability
  migliore, al più 2 apparizioni per brano, metà almeno con tonalità oggi incompatibile (Camelot) se la trasposizione
  la sblocca.
* ``render``: per ogni coppia fino a 9 varianti (le 8 tecniche di dj_mashup + lyric link di dj_creative quando l'ASR
  trova un legame; vocal swap di dj_creative con B trasposto se i tempi sono entro 8 %), ciascuna sul cue del piano o
  sul più vicino inizio frase di A che soddisfa la tecnica; Demucs htdemucs (4 stem) solo sull'estratto di B per il beat
  swap, uno alla volta, stem cancellati a fine fase; QA ``dj_qa`` (configurazione del set v3) e critico v4 come veto
  tecnico (``run_dj_set_v3_bridge.critic``); il riferimento del set v3 per la coppia (clip del set se la coppia c'è,
  altrimenti la decisione del set v3 sul suo cue migliore).
* ``select``: 12-16 clip ammesse (QA non fail, nessun veto) il più possibile diverse (tecniche e coppie), manifest
  ``musicmix.dj-review.1`` con il riferimento del set v3 come voce non votabile (``role: riferimento``), chiave cieca.
* ``page``: ``tools/build_listening_artifact.py --blind`` (mp3 192 kbps) in ``--page-out``.

QA e critico verificano integrità tecnica, non gusto (AGENTS.md). Nulla è stato ascoltato.

    python tools/run_dj_mashup_v1.py features pairs render select page
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import run_dj_set_v3_bridge as v3  # noqa: E402

from musicmix import dj_creative as cr  # noqa: E402
from musicmix.paths import scratch_dir  # noqa: E402
from musicmix import dj_mashup as mx  # noqa: E402
from musicmix import dj_qa  # noqa: E402
from musicmix import dj_render as dj  # noqa: E402
from musicmix import dj_select as djs  # noqa: E402
from musicmix import dj_set as ds  # noqa: E402

v2, base = v3.v2, v3.base
v1 = base.v1
RUN = "dj_mashup_v1"
SCRIPT = "tools/run_dj_mashup_v1.py"
OUT = ROOT / "runs" / RUN
CACHE = ROOT / ".cache" / RUN
PAGE_OUT = scratch_dir("ascolti_mashup_v1")  # override with --page-out
FROZEN, PLAN = v3.FROZEN, v3.PLAN
SR = v3.SR
SEED = 20261021
N_PAIRS = 10
MAX_PER_SLUG = 2
EXCLUDE_ALWAYS = {"times_real_scelta_j1393805": "esclusione richiesta dall'utente", "porrazzo_glide_beat_j1398420": "esclusione richiesta dall'utente"}
LANGS = ("en", "it")
#: ASR rifatta nella lingua giusta (runs/asr_language_v1, stesso stem e stesso track_id): sostituisce le parole del catalogo
#: congelato (es. 18k_no_kizzy: l'inglese era allucinato, CHANGELOG 21/10/2026)
ASR_OVERRIDE_DIR = ROOT / "runs" / "asr_language_v1" / "vocals"
CFG = mx.MashupConfig(creative=v2.CR_CFG)
QA_CFG = v3.QA_CFG
TECH_LABEL = {
    "acapella_mashup": "verso di A sulla base di B (mashup)", "beat_swap": "verso di A sulla batteria di B (beat swap)",
    "hook_teaser": "hook di B anticipato sopra A", "vocal_loop": "loop della voce di A sulla griglia di B",
    "vocal_chop": "chop della voce di A sul 3+3+2 di B", "call_response": "botta e risposta A/B",
    "pivot_freeze": "accordo congelato di A che scivola in B", "reverse_swell": "swell inverso della prima parola di B",
    "lyric_link": "legame di parole (eco)", "vocal_swap_shift": "vocal swap con B trasposto",
}


def load(p: Path) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def dump(p: Path, obj: Any) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(base.jsonable(obj), indent=1, ensure_ascii=False, default=str), encoding="utf-8")


def disk_free_gb() -> float:
    return shutil.disk_usage("/").free / 1e9


def check_disk(stage: str, need_gb: float = 0.5) -> None:
    free = disk_free_gb()
    print(f"[{stage}] disco libero {free:.2f} GB", flush=True)
    if free < need_gb:
        raise SystemExit(f"disco libero {free:.2f} GB < {need_gb} GB: interrotto prima di {stage}")


# ------------------------------------------------------------------ pool
def asr_words(slug: str, track_id: str, words: list) -> tuple[list, dict]:
    """Parole ASR: quelle rifatte nella lingua rilevata se esistono per lo stesso ``track_id``, altrimenti quelle date."""
    p = ASR_OVERRIDE_DIR / slug / "a.json"
    if p.is_file():
        d = load(p)
        if d.get("track_id") == track_id and d.get("words"):
            be = d.get("backend") or {}
            return d["words"], {"source": base.rel(p), "sha256": v3.sha(p), "language": be.get("language"), "n_words": len(d["words"])}
    return words, {"source": "catalogo congelato (vocals_path)", "n_words": len(words)}


def pool() -> tuple[dict, list]:
    cat = load(FROZEN)
    ready, skipped = v3.ready_entries(cat)
    keep, out_skip = {}, list(skipped)
    for t in ready:
        s = t["slug"]
        why = EXCLUDE_ALWAYS.get(s) or v3.EXCLUDE.get(s) or v2.renderable(s, {"plan_dir": PLAN, "catalog": cat})
        if why is None and (t.get("origin") or "").lower().startswith("cc"):
            why = "pool CC escluso"
        if why is None:
            ph = v3.PHRASE / f"{s}.json"
            lang = load(ph)["inputs"].get("asr_language") if ph.is_file() else None
            if lang not in LANGS:
                why = f"lingua ASR {lang!r} non in {LANGS}"
        if why is None:
            why = v2.renderable(s, {"plan_dir": PLAN, "catalog": cat})
        if why:
            out_skip.append({"slug": s, "why": why})
            continue
        keep[s] = {"plan_dir": PLAN, "catalog": cat, "catalog_path": FROZEN}
    from musicmix import track_filter

    log: list = []
    kept = track_filter.filter_sources(keep, log)  # lingua (en/it) ed esclusioni dell'utente, filtro condiviso del progetto
    for d in (log[0]["track_filter"]["dropped"] if log else []):
        out_skip.append({"slug": d["slug"], "why": "track_filter: " + d["why"]})
    return kept, out_skip


# -------------------------------------------------------------- features
def features() -> dict:
    keep, skipped = pool()
    (CACHE / "feat").mkdir(parents=True, exist_ok=True)
    rows = []
    for s, src in sorted(keep.items()):
        fp = CACHE / "feat" / f"{s}.npz"
        an = load(PLAN / "tracks" / f"{s}.json")
        if not fp.is_file():
            t0 = time.time()
            T = v1.Track(s, PLAN, src["catalog"])
            T.words, _ = asr_words(s, T.analysis["track_id"], T.words)
            ins = mx.beat_chroma(T.audio - T.vocals, SR, T.beats)
            voc = mx.beat_chroma(T.vocals, SR, T.beats)
            lufs = dj.integrated_lufs(T.audio, SR)
            ws = np.array([[w["start_sample"], w["end_sample"]] for w in T.words], dtype=np.int64).reshape(-1, 2)
            np.savez_compressed(fp, ins=ins["chroma"], ins_e=ins["energy"], voc=voc["chroma"], voc_e=voc["energy"], beat_s=ins["beat_s"],
                                lufs=np.array(lufs), words=ws, duration_s=np.array(len(T.audio) / SR))
            print(f"feature {s}: {len(ins['chroma'])} battiti, {len(ws)} parole, {time.time() - t0:.0f} s", flush=True)
            del T
            gc.collect()
        z = np.load(fp)
        _, asr = asr_words(s, an["track_id"], [])
        rows.append({"asr_words": asr,"slug": s, "title": an.get("title"), "artist": an.get("artist"), "bpm": an["tempo"]["bpm"], "key": an["key"]["key"],
                     "camelot": an["key"]["camelot"], "key_level": an["key"]["level"], "n_beats": int(len(z["beat_s"])),
                     "n_words": int(len(z["words"])), "lufs": float(z["lufs"])})
    doc = {"schema_version": "musicmix.dj_mashup_features.1", "catalog_frozen": base.rel(FROZEN), "catalog_frozen_sha256": v3.sha(FROZEN),
           "tracks": rows, "skipped": skipped,
           "chroma": "librosa chroma_cqt (22050 Hz, hop 512, norm=None), median per beat of the dj_select regular grid (AutoMashUpper)"}
    dump(OUT / "features.json", doc)
    return doc


# ----------------------------------------------------------------- pairs
def _units(words: np.ndarray) -> list[dict]:
    return [{"start": int(a), "end": int(b), "text": ""} for a, b in words]


def _hook_s(an: dict) -> float | None:
    """Inizio del primo ritornello (allin1) del brano, o None."""
    for sec in an.get("sections") or []:
        if sec.get("label") == "chorus":
            return float(sec.get("downbeat_s", sec["start_s"]))
    return None


def _feasible(ua, ub, an_a, an_b, X: float, Y: float, hook: float | None) -> dict:
    gap = int(CFG.phrase_gap_s * SR)
    beat_a, bar_a = an_a["tempo"]["period_s"], 4 * an_a["tempo"]["period_s"]
    beat_b, bar_b = an_b["tempo"]["period_s"], 4 * an_b["tempo"]["period_s"]
    sa = mx.line_starts(ua, gap)
    ea = mx.line_ends(ua, gap)
    xs = X * SR
    starts_near = any(xs - beat_a * SR <= ua[i]["start"] <= xs + bar_a * SR for i in sa)
    ends_before = any(xs - 2 * bar_a * SR <= ua[i]["end"] <= xs - 0.05 * SR for i in ea)
    crossing = any(u["start"] < xs - 0.03 * SR and u["end"] > xs + 0.03 * SR for u in ua)
    sb_ = mx.line_starts(ub, gap)
    hook_ok = hook is not None and any(ub[i]["start"] >= (hook - beat_b) * SR for i in sb_)
    b_phrase_near = any((Y - beat_b) * SR <= ub[i]["start"] <= (Y + 2 * bar_b) * SR for i in sb_)
    return {"a_line_starts_at_cue": starts_near, "a_line_ends_before_cue": ends_before and not crossing,
            "a_word_crosses_cue": crossing, "b_hook_phrase": hook_ok, "b_phrase_after_cue": b_phrase_near}


def techniques_ok(f: dict) -> list[str]:
    out = []
    if f["a_line_starts_at_cue"]:
        out += ["acapella_mashup", "beat_swap"]
        if f["b_hook_phrase"]:
            out.append("call_response")
    if f["a_line_ends_before_cue"]:
        out += ["vocal_loop", "vocal_chop", "pivot_freeze"]
        if f["b_hook_phrase"]:
            out.append("hook_teaser")
        if f["b_phrase_after_cue"]:
            out.append("reverse_swell")
    return out


def _frames(z: dict, key: str, b0: int, nb: int) -> np.ndarray:
    c = z[key]
    b0 = max(0, b0)
    return c[b0 : b0 + nb]


#: criteri dell'utente dai voti ciechi del torneo b_entry_fix_v1 (orchestratore, 21/10/2026) [user]; pesi [H]
USER_W = {"charge": 0.15, "arrival": 0.15, "key_today": 0.12, "key_after_shift": 0.08, "sound_link": 0.10}
CHARGE_FROM = ("chorus", "bridge", "inst")  # A esce dopo un ritornello/bridge/strumentale (punto di carica)...
CHARGE_TO = ("chorus",)  # ...o subito prima del proprio ritornello (fine del pre-ritornello)
ARRIVAL = ("chorus",)  # B entra su un arrivo forte (ritornello/hook)


def user_terms(p: dict, A: dict, B: dict, shift: int, X: float) -> dict:
    sec = str(p.get("a_section") or "")
    prev, _, nxt = sec.partition("->")
    nxt = nxt or prev
    charge = (prev in CHARGE_FROM or nxt in CHARGE_TO) and X / float(A["duration_s"]) >= 0.35
    arrival = str(p.get("b_section") or "").split("->")[-1] in ARRIVAL
    rel_today = cr.key_relation(A["key"]["key"], B["key"]["key"])
    ka = cr.parse_key(A["key"]["key"])
    rel_after = cr.key_relation(cr.key_name((ka[0] + shift) % 12, ka[1]), B["key"]["key"]) if ka else "unknown"
    ok = ("same", "adjacent", "relative")
    t = {"charge": bool(charge), "arrival": bool(arrival), "key_today": rel_today in ok, "key_after_shift": rel_after in ok,
         "relation_today": rel_today, "relation_after_shift": rel_after}
    t["bonus"] = sum(USER_W[k] for k in ("charge", "arrival", "key_today", "key_after_shift") if t[k])
    return t


_WORDS: dict = {}


def words_of(slug: str) -> list:
    if slug not in _WORDS:
        cat = load(FROZEN)
        e = next(t for t in cat["tracks"] if t["slug"] == slug)
        an = load(PLAN / "tracks" / f"{slug}.json")
        w = load(v3.localize(e["vocals_path"])).get("words", [])
        _WORDS[slug], _ = asr_words(slug, an["track_id"], w)
    return _WORDS[slug]


def pairs() -> dict:
    keep, _ = pool()
    feats = {s: dict(np.load(CACHE / "feat" / f"{s}.npz")) for s in keep}
    an = {s: load(PLAN / "tracks" / f"{s}.json") for s in keep}
    units = {s: _units(feats[s]["words"]) for s in keep}
    hooks = {s: _hook_s(an[s]) for s in keep}
    rows = []
    for a in sorted(keep):
        for b in sorted(keep):
            if a == b:
                continue
            A, B = an[a], an[b]
            try:
                k, r = mx.fold_ratio(4 * A["tempo"]["period_s"], 4 * B["tempo"]["period_s"], CFG.max_vocal_stretch)
            except dj.DJRenderError as exc:
                rows.append({"a": a, "b": b, "status": "tempo", "why": str(exc)})
                continue
            plans = djs.plan_pair(A, B, top_k=10 ** 9)
            key_shifts = tuple(mx.key_compatible_shifts(A["key"]["key"], B["key"]["key"])) if CFG.key_guard else ()
            cand_rows = []
            fa, fb = feats[a], feats[b]
            bta, btb = fa["beat_s"], fb["beat_s"]
            best = None
            hook_b = hooks[b]
            for p in plans["plans"]:
                X, Y = float(p["a_out_s"]), float(p["b_in_s"])
                if X < 30.0 or X > float(fa["duration_s"]) - 20.0:
                    continue
                f = _feasible(units[a], units[b], A, B, X, Y, hook_b)
                techs = techniques_ok(f)
                if len(techs) < 4:
                    continue
                ia, ib = int(np.searchsorted(bta, X - 1e-3)), int(np.searchsorted(btb, Y - 1e-3))
                nb_b = int(round(16 * k))
                top = mx.resample_beats(_frames(fa, "voc", ia, 16), k)
                w = mx.resample_beats(_frames(fa, "voc_e", ia, 16)[:, None].repeat(12, 1), k)[:, 0]
                vert = mx.harmonic_match(top, _frames(fb, "ins", ib, nb_b), top_weight=w, shifts=key_shifts or mx.SHIFT_RANGE,
                                         min_gain=CFG.shift_min_gain, penalty=CFG.shift_penalty)
                chop = mx.harmonic_match(_frames(fa, "voc", ia - 8, 8).sum(0, keepdims=True), _frames(fb, "ins", ib, 8).sum(0, keepdims=True),
                                         min_gain=CFG.shift_min_gain, penalty=CFG.shift_penalty)
                horiz = mx.harmonic_match(_frames(fa, "ins", ia - 16, 16).sum(0, keepdims=True), _frames(fb, "ins", ib, 16).sum(0, keepdims=True),
                                          shifts=(0,))
                sc = mx.mashability_score(vert, horiz, r, CFG.max_vocal_stretch)
                vsh = float(np.mean(B["bars"]["vocal"][p["b_in_bar"] : p["b_in_bar"] + 4] or [0.0]))
                ut = user_terms(p, A, B, int(vert.get("best_shift", 0)), X)
                score = sc["score"] + 0.01 * len(techs) + ut["bonus"]  # [H] + [user] criteria
                cand_rows.append((score, p, ut))
                if best is None or score > best["score"]:
                    best = {"score": score, "user_terms": ut, "mashability": sc, "plan": {k2: p[k2] for k2 in ("a_out_s", "a_out_bar", "b_in_s", "b_in_bar", "type", "score", "a_section", "b_section")},
                            "vertical": {k2: vert.get(k2) for k2 in ("best_shift", "best", "gain_over_no_shift", "unconstrained_best_shift")},
                            "chop": {k2: chop.get(k2) for k2 in ("best_shift", "best", "gain_over_no_shift")},
                            "horizontal": horiz.get("best"), "feasible": f, "techniques": techs, "b_vocal_share_4_bars": vsh}
            if best is None:
                rows.append({"a": a, "b": b, "status": "no_plan", "why": "nessun piano con almeno 4 tecniche possibili"})
                continue
            # [user] sound affinity between A's last words and B's first words (lyric link) on the 5 best plans
            for sc_, p_, ut_ in sorted(cand_rows, key=lambda r_: -r_[0])[:5]:
                links = mx.find_sound_links(words_of(a), words_of(b), [p_["a_out_s"]], [p_["b_in_s"]], sr=SR,
                                            bar_a_s=4 * A["tempo"]["period_s"], bar_b_s=4 * B["tempo"]["period_s"])
                if links and sc_ + USER_W["sound_link"] * links[0]["match"]["score"] > best["score"]:
                    best = dict(best, score=sc_ + USER_W["sound_link"] * links[0]["match"]["score"], user_terms=ut_,
                                plan={k2: p_[k2] for k2 in ("a_out_s", "a_out_bar", "b_in_s", "b_in_bar", "type", "score", "a_section", "b_section")})
                if links:
                    best.setdefault("sound_links", []).append({"a_out_s": p_["a_out_s"], "b_in_s": p_["b_in_s"],
                                                               "a_word": links[0]["a_word"].get("text"), "b_word": links[0]["b_word"].get("text"),
                                                               "match": links[0]["match"]})
            rel = djs.camelot_relation(A["key"]["camelot"], B["key"]["camelot"])["relation"]
            rows.append({"a": a, "b": b, "status": "ok", "fold_k": k, "time_ratio_a": r, "camelot": [A["key"]["camelot"], B["key"]["camelot"]],
                         "key_relation": rel, "key_compatible_today": rel in ("same", "adjacent", "relative"), "hook_b_s": hook_b, **best})
    ok = sorted((x for x in rows if x["status"] == "ok"), key=lambda x: -x["score"])
    chosen, count = [], {}

    def take(x):
        chosen.append(x)
        for s in (x["a"], x["b"]):
            count[s] = count.get(s, 0) + 1

    def free(x):
        return (count.get(x["a"], 0) < MAX_PER_SLUG and count.get(x["b"], 0) < MAX_PER_SLUG
                and not any(c["a"] == x["b"] and c["b"] == x["a"] for c in chosen))

    for x in ok:
        if len(chosen) >= N_PAIRS:
            break
        if x not in chosen and free(x):
            take(x)
    doc = {"schema_version": "musicmix.dj_mashup_pairs.1", "n_pairs_scored": len(rows), "n_ok": len(ok),
           "selection_rule": f"[H] per pair the best plan by mashability + [user] terms (A exits at a charge point, B arrives on a chorus, "
                             f"key compatible today / after the shift, sound link between A's last and B's first words; weights {USER_W}); "
                             f"{N_PAIRS} pairs by score, <= {MAX_PER_SLUG} per track, no A<->B duplicates",
           "user_criteria_source": "orchestrator relay of the blind tournament votes (giudizi_cloud/b_entry_fix_v1/), 21/10/2026",
           "chosen": chosen, "ranking_top30": ok[:30], "rejected": [x for x in rows if x["status"] != "ok"]}
    dump(OUT / "pairs.json", doc)
    for c in chosen:
        print(f"{c['a']} -> {c['b']}: {c['score']:.3f} shift {c['vertical']['best_shift']:+d} {c['key_relation']} {c['user_terms']} "
              f"{c['plan']['a_section']} | {c['plan']['b_section']} links {[(l_['a_word'], l_['b_word']) for l_ in c.get('sound_links', [])][:2]}", flush=True)
    return doc


# ---------------------------------------------------------------- render
def excerpt(T, lo_s: float, hi_s: float) -> SimpleNamespace:
    T.words, asr = asr_words(T.slug, T.analysis["track_id"], T.words)
    lo, hi = max(0, int(lo_s * SR)), min(len(T.audio), int(hi_s * SR))
    ws = [dict(w, start_sample=int(w["start_sample"]) - lo, end_sample=int(w["end_sample"]) - lo) for w in T.words
          if lo <= int(w["start_sample"]) and int(w["end_sample"]) <= hi]
    sel = lambda g: g[(g >= lo / SR) & (g <= hi / SR)] - lo / SR  # noqa: E731
    return SimpleNamespace(slug=T.slug, title=T.title, audio=np.ascontiguousarray(T.audio[lo:hi]), vocals=np.ascontiguousarray(T.vocals[lo:hi]),
                           words=ws, beats=sel(T.beats), downbeats=sel(T.downbeats), vocal_threshold=T.vocal_threshold,
                           analysis=T.analysis, provenance=dict(T.provenance, excerpt_s=[lo / SR, hi / SR], asr_words=asr), offset=lo,
                           lang=asr.get("language") or "en",
                           lufs=dj.integrated_lufs(T.audio, SR))


def wait_for_memory(gb: float, timeout_s: float = 3600.0) -> None:
    """Attende che la memoria disponibile (``MemAvailable``) sia >= ``gb`` GB (Demucs condiviso con altri agenti)."""
    t0 = time.time()
    while True:
        avail = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable")) / 1e6
        if avail >= gb:
            return
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"memoria disponibile {avail:.1f} GB < {gb} GB da {timeout_s:.0f} s: Demucs non avviato")
        print(f"  attesa memoria: {avail:.1f} GB < {gb} GB", flush=True)
        time.sleep(30)


def demucs4(audio: np.ndarray) -> tuple[dict, dict]:
    """htdemucs (4 stem) su un estratto, stesso percorso d'inferenza ufficiale (normalizzazione per media/std del
    riferimento, apply_model split overlap 0.25, shifts 0); ricampionamento 48 <-> 44,1 kHz con torchaudio."""
    import torch
    import torchaudio
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    torch.set_num_threads(4)
    model = get_model("htdemucs")
    model.eval()
    wav = torch.tensor(np.ascontiguousarray(audio.T), dtype=torch.float32)
    wav = torchaudio.functional.resample(wav, SR, model.samplerate)
    ref = wav.mean(0)
    m, s = ref.mean(), ref.std()
    with torch.no_grad():
        out = apply_model(model, ((wav - m) / s)[None], device="cpu", shifts=0, split=True, overlap=0.25, progress=False)[0]
    out = out * s + m
    stems = {}
    for i, name in enumerate(model.sources):
        y = torchaudio.functional.resample(out[i], model.samplerate, SR).numpy().T.astype(np.float64)
        stems[name] = cr._fit(y, len(audio))
    ck = Path(torch.hub.get_dir()) / "checkpoints" / "955717e8-8726e21a.th"
    info = {"model": "htdemucs", "sources": list(model.sources), "checkpoint": str(ck), "checkpoint_sha256": base.sha256(ck) if ck.is_file() else None,
            "inference": "official normalisation (reference mean/std), apply_model split=True overlap=0.25 shifts=0, torchaudio resample 48k<->44.1k"}
    del model, out
    gc.collect()
    return stems, info


def _words_b_excerpt(B) -> list:
    """Parole di B per il controllo b_entry come nel set v3 (raffinate da dj_phrase sullo stem, se il modello c'è),
    portate nelle coordinate dell'estratto."""
    v3.load_phrase([B.slug])
    ref = v3.words_b_refined(B.slug) if v3.B_ENTRY["on"] else None
    src = ref if ref else [{"start_sample": w["start_sample"] + B.offset, "end_sample": w["end_sample"] + B.offset, "text": w.get("text", "")}
                           for w in B.words]
    hi = B.offset + len(B.audio)
    return [dict(w, start_sample=int(w["start_sample"]) - B.offset, end_sample=int(w["end_sample"]) - B.offset)
            for w in src if B.offset <= int(w["start_sample"]) and int(w["end_sample"]) <= hi]


def _words_b_clip(meta: dict, B) -> list:
    """Parole di B nella clip. dj_mashup (B non stirato, ``meta['b_entry']`` = gate della voce): tutte le parole sul
    tempo proprio di B (clip = switch + sorgente - B allo switch), così ``check_b_entry`` vede anche la parola prima
    del gate e verifica che la frase non sia già cominciata. dj_creative: solo dove la voce di B suona (mappe)."""
    words = _words_b_excerpt(B)
    if meta.get("mashup_version") and meta.get("b_entry"):
        sr = int(meta["sample_rate"])
        t_sw, sb = int(meta["switch"]["output_sample"]), int(meta["switch"]["b_source_sample"])
        n = int(meta["clip"]["length_samples"])
        out = []
        for w in words:
            s0, s1 = t_sw + int(w["start_sample"]) - sb, t_sw + int(w["end_sample"]) - sb
            if s1 > 0 and s0 < n:
                out.append({"start_s": s0 / sr, "end_s": s1 / sr, "text": w.get("text", ""), "start_unclipped_s": s0 / sr,
                            "end_unclipped_s": s1 / sr})
        return out
    return cr.words_in_clip(meta, words, "b")


def qa_mashup(res, A, B, tracker) -> dict:
    """``run_dj_set_v3_bridge.qa_v3`` per i render di dj_creative/dj_mashup, ma con le parole di B nelle coordinate
    dell'estratto e portate nella clip con le mappe del render (``dj_creative.words_in_clip``: solo dove la voce di B
    suona davvero, gate compreso)."""
    meta = res.metadata
    qi = cr.qa_inputs(res)
    ga = 10 ** (meta["loudness"]["a_gain_db"] / 20.0)
    gb = 10 ** (meta["loudness"]["b_gain_db"] / 20.0)
    return dj_qa.evaluate_render(res.audio.astype(np.float64), meta, sample_rate=res.sample_rate, decks=qi["decks"],
                                 vocals_a=qi["vocals_a"], vocals_b=qi["vocals_b"],
                                 vocal_thresholds=(A.vocal_threshold * ga, B.vocal_threshold * gb),
                                 words_a=cr.words_in_clip(meta, A.words, "a"), words_b=_words_b_clip(meta, B),
                                 tracker=tracker, config=QA_CFG, grid_support={"a": A.analysis["tempo"], "b": B.analysis["tempo"]})


def qa_and_critic(res, A, B, ident: str, declared_gap: bool, tracker) -> tuple[dict, dict]:
    qa = qa_mashup(res, A, B, tracker)
    crt = v3.critic(res, res.metadata.get("kind", "mashup"), ident, decks=res.decks, qa=qa, declared_gap=declared_gap)
    return qa, crt


def _cue_candidates(A, an_a: dict, X0: float) -> list[float]:
    """Cue del piano poi gli altri inizi frase di A (dj_select) entro 16 battute, dal più vicino; tutti dentro l'estratto."""
    cands = [c["time_s"] for c in djs.cue_candidates(an_a, "out")]
    lo, hi = A.offset / SR + 25.0, A.offset / SR + len(A.audio) / SR - 25.0
    near = sorted((t for t in cands if abs(t - X0) <= 24 * 4 * an_a["tempo"]["period_s"] and lo <= t <= hi and abs(t - X0) > 1e-3),
                  key=lambda t: abs(t - X0))
    return [X0] + near[:9]


def beat_swap_precheck(A, B, Xl: float, Yl: float, an_a: dict, an_b: dict, drum_bars: int) -> str | None:
    """Prima di Demucs: c'è un verso di A che parte al cue e finisce entro ``drum_bars`` battute di B? (stesse regole
    di ``dj_mashup.acapella_mashup``.)"""
    try:
        k, r = mx.fold_ratio(4 * an_a["tempo"]["period_s"], 4 * an_b["tempo"]["period_s"], CFG.max_vocal_stretch)
        ua = mx.vocal_units(A.words, SR)
        X = int(round(Xl * SR))
        beat, bar = int(round(an_a["tempo"]["period_s"] * SR)), int(round(4 * an_a["tempo"]["period_s"] * SR))
        i0 = mx.a_line_from(ua, X, beat, bar, CFG, SR)
        p0 = ua[i0]["start"] - int(round(CFG.gate_lead_s * SR))
        hi = X + int(drum_bars * 4 * an_b["tempo"]["period_s"] * SR / r) - int(round(CFG.word_pad_s * SR))
        mx.a_line_end_after(ua, i0, p0 + bar, hi, CFG, SR)
    except dj.DJRenderError as exc:
        return str(exc)
    return None


def render_pair(pc: dict, tracker, done: dict) -> dict:
    a, b = pc["a"], pc["b"]
    cat = load(FROZEN)
    pdir = OUT / "renders" / f"{a}__{b}"
    pdir.mkdir(parents=True, exist_ok=True)
    plan = pc["plan"]
    X0, Y = float(plan["a_out_s"]), float(plan["b_in_s"])
    hook = pc.get("hook_b_s")
    TA = v1.Track(a, PLAN, cat)
    A = excerpt(TA, X0 - 24 * 4 * TA.analysis["tempo"]["period_s"] - 30.0, X0 + 24 * 4 * TA.analysis["tempo"]["period_s"] + 40.0)
    del TA
    TB = v1.Track(b, PLAN, cat)
    lo_b = min(Y, hook if hook is not None else Y) - 25.0
    hi_b = max(Y, hook if hook is not None else Y) + 60.0
    B = excerpt(TB, lo_b, hi_b)
    del TB
    gc.collect()
    an_a, an_b = A.analysis, B.analysis
    cfg = dataclasses.replace(CFG, creative=dataclasses.replace(CFG.creative, lufs_a_override=A.lufs, lufs_b_override=B.lufs))
    grids = {"beats_a": A.beats, "downbeats_a": A.downbeats, "beats_b": B.beats, "downbeats_b": B.downbeats}
    Yl = Y - B.offset / SR
    yi = int(np.argmin(np.abs(B.downbeats - Yl)))
    Yl = float(B.downbeats[yi])
    hook_l = None if hook is None else hook - B.offset / SR
    keys = {"key_a": an_a["key"]["key"], "key_b": an_b["key"]["key"]}
    prov = {"a": A.provenance, "b": B.provenance, "script": SCRIPT, "run": RUN, "pair_selection": base.rel(OUT / "pairs.json")}
    # B's words for its vocal gate = the same dj_phrase-refined words (on/off on the stem) that dj_qa.6 uses for b_entry
    common = dict(vocals_a=A.vocals, vocals_b=B.vocals, words_a=A.words, words_b=_words_b_excerpt(B), sample_rate=SR, config=cfg,
                  provenance=prov, return_decks=True, **keys)
    variants = [("acapella_mashup", dict(bars=4)), ("vocal_loop", dict(mode="halving", loop_bars=2)),
                ("vocal_chop", dict(pattern="332", bars=2)), ("pivot_freeze", dict(hold_beats=4, glide_beats=4)),
                ("reverse_swell", dict(swell_beats=4))]
    if hook_l is not None and B.offset / SR <= hook <= B.offset / SR + len(B.audio) / SR - 20:
        variants += [("hook_teaser", dict(teaser_bars=2, hook_s=hook_l)), ("call_response", dict(rounds=2, slot_bars=2, hook_s=hook_l)),
                     ("call_response", dict(rounds=2, slot_bars=1, hook_s=hook_l))]
    variants += [("beat_swap", dict(drum_bars=2)), ("beat_swap", dict(drum_bars=4))]
    stems_b, stems_info = None, None
    out = done.get("renders", {})
    seen_tech = {v.get("technique") for v in out.values() if v.get("status") == "ok"}
    for tech, kw in variants:
        name = {"call_response": f"call_response_s{kw.get('slot_bars')}", "beat_swap": f"beat_swap_d{kw.get('drum_bars')}"}.get(tech, tech)
        if (name in out and out[name].get("status") in ("ok", "rejected_all")) or (tech in ("call_response", "beat_swap") and tech in seen_tech):
            continue
        rec = {"technique": tech, "params": {k: v for k, v in kw.items() if k != "hook_s"}, "attempts": []}
        rejected = None
        for X in _cue_candidates(A, an_a, X0):
            Xl = X - A.offset / SR
            xi = int(np.argmin(np.abs(A.downbeats - Xl)))
            if abs(A.downbeats[xi] - Xl) > 0.03:
                continue
            Xl = float(A.downbeats[xi])
            extra = {}
            if tech == "beat_swap":
                why = beat_swap_precheck(A, B, Xl, Yl, an_a, an_b, kw["drum_bars"])
                if why:
                    rec["attempts"].append({"a_out_s": X, "error": "precheck: " + why})
                    continue
                if stems_b is None:
                    check_disk("demucs", 0.6)
                    seg0 = max(0, int((Yl - 3.0) * SR))  # only B's first 4 bars need the drum stem (+ margins): Demucs is slow on Linux
                    seg1 = min(len(B.audio), int((Yl + 16 * an_b["tempo"]["period_s"] + 4.0) * SR))
                    t0 = time.time()
                    cp = CACHE / "stems" / f"{b}_{Y:.3f}.npz"
                    if cp.is_file():
                        z = np.load(cp, allow_pickle=False)
                        st = {k: z[k].astype(np.float64) for k in ("drums", "bass", "other", "vocals")}
                        stems_info = json.loads(str(z["info"]))
                    else:
                        wait_for_memory(5.0)  # coordinator: Demucs only with >= 5 GB free, one at a time
                        st, stems_info = demucs4(B.audio[seg0:seg1])
                        cp.parent.mkdir(parents=True, exist_ok=True)
                        np.savez(cp, info=np.array(json.dumps(base.jsonable(stems_info))), **{k: v.astype(np.float16) for k, v in st.items()})
                    stems_b = {k: np.vstack([np.zeros((seg0, 2)), v, np.zeros((len(B.audio) - seg1, 2))]) for k, v in st.items()}
                    stems_info = dict(stems_info, excerpt_b_s=[seg0 / SR, seg1 / SR], elapsed_s=round(time.time() - t0, 1),
                                      cache="float16 npz in .cache/dj_mashup_v1/stems (deleted by the cleanup stage)")
                    print(f"  demucs {b}: {stems_info['elapsed_s']} s", flush=True)
                extra["stems_b"] = stems_b
            try:
                t0 = time.time()
                res = mx.render_mashup(tech, A.audio, B.audio, grids=grids, cue_out_a=Xl, cue_in_b=Yl, **common, **kw, **extra)
            except (dj.DJRenderError, ValueError) as exc:
                rec["attempts"].append({"a_out_s": X, "error": str(exc)[:300]})
                continue
            except Exception as exc:  # a bug: record it and go on with the next technique
                rec["attempts"].append({"a_out_s": X, "error": f"BUG {type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-1500:]})
                out[name] = dict(rec, status="error")
                print(f"  {name}: ERRORE {type(exc).__name__}: {exc}", flush=True)
                break
            ident = f"{a}__{b}__{name}"
            try:
                qa, crt = qa_and_critic(res, A, B, ident, declared_gap=(tech == "pivot_freeze"), tracker=tracker)
            except Exception as exc:
                rec["attempts"].append({"a_out_s": X, "error": f"QA/critic {type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-1500:]})
                out[name] = dict(rec, status="error")
                print(f"  {name}: ERRORE QA {type(exc).__name__}: {exc}", flush=True)
                break
            meta = res.metadata
            meta["mashup_run"] = {"pair": [a, b], "a_out_s_track": X, "b_in_s_track": Y, "excerpt_offsets_s": [A.offset / SR, B.offset / SR],
                                  "demucs": stems_info if tech == "beat_swap" else None}
            ok = qa["verdict"] != "fail" and not crt.get("veto")
            rec["attempts"].append({"a_out_s": X, "qa": qa["verdict"], "veto": crt.get("veto"), "elapsed_s": round(time.time() - t0, 1)})
            entry = {"status": "ok" if ok else "rejected", "technique": tech, "name": name, "a_out_s": X, "b_in_s": Y,
                     "qa": {"verdict": qa["verdict"], "reasons": [f"{r_['check']}: {r_['message']}" for r_ in qa["reasons"]],
                            "coverage_missing": qa["coverage"]["missing_required"]},
                     "critic": {k2: crt.get(k2) for k2 in ("score", "flags", "veto", "diagnostic_flags", "declared_phase", "craft")},
                     "transition_s": [meta["overlap"]["output_start_sample"] / SR, meta["switch"]["output_sample"] / SR,
                                      meta["overlap"]["output_end_sample"] / SR],
                     "duration_s": len(res.audio) / SR, "clip_rule_ok": meta["clip"]["matches_rule"],
                     "keys": meta["creative"].get("keys", {}), "params": meta["creative"].get("params", {})}
            jp = pdir / f"{name}.json"
            dump(jp, {"metadata": meta, "qa": qa, "critic": crt, "summary": entry})
            if ok:
                import soundfile as sf

                fl = pdir / f"{name}.flac"
                sf.write(str(fl), res.audio, SR, subtype="PCM_24")
                entry["audio"] = base.rel(fl)
            print(f"  {name} @ {X:.2f}: QA {qa['verdict']} veto {crt.get('veto')} ({time.time() - t0:.0f} s)", flush=True)
            del res
            gc.collect()
            if ok:
                out[name] = dict(rec, **entry)
                break
            rejected = {**rec, **entry, "status": "rejected_all"}  # QA fail or veto: try the next phrase-start cue of A
        else:
            if rejected is not None:
                out[name] = rejected
                print(f"  {name}: tutti i cue bocciati da QA/veto", flush=True)
            else:
                out[name] = dict(rec, status="infeasible")
                print(f"  {name}: non realizzabile ({rec['attempts'][-1]['error'] if rec['attempts'] else 'nessun cue'})", flush=True)
        if out[name].get("status") == "ok":
            seen_tech.add(tech)
        done["renders"] = out
        dump(OUT / "renders" / f"{a}__{b}" / "index.json", done)
    # dj_creative techniques on the same cue: lyric link (if the ASR finds a link) and vocal swap with B transposed
    for name in ("lyric_link", "vocal_swap_shift"):
        if name in out:
            continue
        rec = {"technique": name, "attempts": []}
        try:
            Xl = float(A.downbeats[int(np.argmin(np.abs(A.downbeats - (X0 - A.offset / SR))))])
            ckw = dict(beats_a=A.beats, downbeats_a=A.downbeats, beats_b=B.beats, downbeats_b=B.downbeats, cue_out_a=Xl, cue_in_b=Yl,
                       vocals_a=A.vocals, vocals_b=B.vocals, sample_rate=SR, config=cfg.creative, provenance=prov, return_decks=True)
            if name == "lyric_link":
                # [user] sound affinity (word, lemma, rhyme, Soundex, vocalise, alliteration) on A's phrase-start cues near the plan
                a_cues = [x - A.offset / SR for x in _cue_candidates(A, an_a, X0)]
                links = mx.find_sound_links(A.words, B.words, a_cues, [Yl], sr=SR, bar_a_s=4 * an_a["tempo"]["period_s"],
                                            bar_b_s=4 * an_b["tempo"]["period_s"], lang_a=A.lang, lang_b=B.lang)
                if not links:
                    raise dj.DJRenderError("no sound link between A's last words and B's first words on these cues")
                res, err = None, None
                for L in links:
                    try:
                        xa = float(A.downbeats[int(np.argmin(np.abs(A.downbeats - L["a_cue_s"])))])
                        res = cr.lyric_link(A.audio, B.audio, a_word=L["a_word"], b_word=L["b_word"], match=L["match"],
                                            **dict(ckw, cue_out_a=xa))
                        rec["link"] = {"a_word": L["a_word"].get("text"), "b_word": L["b_word"].get("text"), "match": L["match"],
                                       "a_out_s": xa + A.offset / SR}
                        break
                    except dj.DJRenderError as exc:
                        err = exc
                if res is None:
                    raise err
                tech_kind = "lyric_link"
            else:
                s = -int(pc["vertical"]["best_shift"])  # A's material up s == B down s
                if s == 0:
                    raise dj.DJRenderError("no transposition from the mashability: the plain vocal swap is not this gallery's subject")
                res = cr.vocal_swap(A.audio, B.audio, key_shift_b=s, bars=4, **ckw)
                tech_kind = "vocal_swap"
        except (dj.DJRenderError, ValueError, KeyError) as exc:
            out[name] = dict(rec, status="infeasible", error=str(exc)[:300])
            print(f"  {name}: non realizzabile ({str(exc)[:120]})", flush=True)
            continue
        except Exception as exc:
            out[name] = dict(rec, status="error", error=f"BUG {type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-1500:])
            print(f"  {name}: ERRORE {type(exc).__name__}: {exc}", flush=True)
            continue
        qa, crt = qa_and_critic(res, A, B, f"{a}__{b}__{name}", False, tracker)
        if tech_kind == "vocal_swap":  # two tracks overlap harmonically: tonal distance is a veto (set v3 rule)
            crt = v3.critic(res, "vocal_swap", f"{a}__{b}__{name}", decks=res.decks, qa=qa)
        ok = qa["verdict"] != "fail" and not crt.get("veto")
        meta = res.metadata
        meta["mashup_run"] = {"pair": [a, b], "a_out_s_track": X0, "b_in_s_track": Y, "excerpt_offsets_s": [A.offset / SR, B.offset / SR]}
        entry = {"status": "ok" if ok else "rejected", "technique": name, "name": name, "a_out_s": X0, "b_in_s": Y,
                 "qa": {"verdict": qa["verdict"], "reasons": [f"{r_['check']}: {r_['message']}" for r_ in qa["reasons"]],
                        "coverage_missing": qa["coverage"]["missing_required"]},
                 "critic": {k2: crt.get(k2) for k2 in ("score", "flags", "veto", "diagnostic_flags", "declared_phase", "craft")},
                 "transition_s": [meta["overlap"]["output_start_sample"] / SR, meta["switch"]["output_sample"] / SR, meta["overlap"]["output_end_sample"] / SR],
                 "duration_s": len(res.audio) / SR, "clip_rule_ok": meta["clip"]["matches_rule"],
                 "keys": {"key_shift_b": (meta.get("stretch") or {}).get("key_shift_b")} if tech_kind == "vocal_swap" else {},
                 "params": (meta.get("creative") or {}).get("params", {})}
        dump(pdir / f"{name}.json", {"metadata": meta, "qa": qa, "critic": crt, "summary": entry})
        if ok:
            import soundfile as sf

            fl = pdir / f"{name}.flac"
            sf.write(str(fl), res.audio, SR, subtype="PCM_24")
            entry["audio"] = base.rel(fl)
        out[name] = dict(rec, **entry)
        print(f"  {name}: QA {qa['verdict']} veto {crt.get('veto')}", flush=True)
        del res
        gc.collect()
    done["renders"] = out
    done["demucs"] = stems_info
    dump(pdir / "index.json", done)
    del A, B, stems_b
    gc.collect()
    return done


# -------------------------------------------------------------- reference
def reference(pc: dict, tracker) -> dict:
    """La tecnica del set v3 per la coppia: la clip del set se la coppia c'è; altrimenti la decisione del set v3
    (policy dei libri se giuntabile, altrimenti l'albero ``books``) sul candidato di costo minimo della coppia
    (senza il filtro dei generi lontani e senza il Viterbi del set: dichiarato)."""
    import soundfile as sf

    a, b = pc["a"], pc["b"]
    rdir = OUT / "references"
    rdir.mkdir(parents=True, exist_ok=True)
    jp = rdir / f"{a}__{b}.json"
    if jp.is_file():
        return load(jp)
    setdoc = load(v3.OUT / "set.json")
    tr = next((t for t in setdoc["transitions"] if t["a"] == a and t["b"] == b), None)
    if tr is not None:
        wav = ROOT / tr["clip_wav"]
        audio, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        meta = load(wav.with_suffix(".json"))
        meta = meta.get("metadata", meta)
        g = ds.clip_geometry(meta, len(audio))
        clip, tm = v3.trim_to_rule(audio, meta.get("kind", tr["technique"]), g["transition_start"] / sr, g["switch"] / sr, g["transition_end"] / sr)
        out = rdir / f"{a}__{b}.wav"
        h = v3.write_clip(clip, out)
        doc = {"source": "set v3 clip", "set_transition": tr["index"], "technique": tr["technique"], "label": tr.get("label"),
               "rule": tr.get("rule"), "qa": tr.get("qa", {}).get("verdict"), "audio": base.rel(out), "sha256": h, "markers": tm,
               "kind": meta.get("kind", tr["technique"]), "duration_s": len(clip) / sr}
        dump(jp, doc)
        return doc
    cat = load(FROZEN)
    src = {"plan_dir": PLAN, "catalog": cat, "catalog_path": FROZEN}
    v3.load_phrase([a, b])
    analyses = {}
    for s in (a, b):
        an = load(PLAN / "tracks" / f"{s}.json")
        an["form"] = v2.compact_form(v2.form_for(s, src, an))
        analyses[s] = an
    tree_over = dict(v2.PROFILE["tree"])
    order_over = v3._order_over()
    info_a = v3.info_v3(a, src)
    pcand = v2.apply_filters(ds.candidates_for_pair(analyses[a], analyses[b], tree_over, order_over), info_a, order_over)
    pcand = v3.filter_pair_v3(pcand, a, b, analyses)
    cands = [c for c in pcand["candidates"] if c["admissible"]][:4]
    if not cands:
        doc = {"source": "none", "why": "nessun candidato ammissibile per il set v3"}
        dump(jp, doc)
        return doc
    A = v1.Track(a, PLAN, cat)
    B = v1.Track(b, PLAN, cat)
    prov = {"a": A.provenance, "b": B.provenance, "script": SCRIPT, "run": RUN, "reference": "set v3 decision"}
    tried = []
    doc = None
    for rank, c0 in enumerate(cands):  # like the set: if every technique fails on this cue, the next candidate (cost order)
        plan = c0["plan"]
        props = v3.policy_for(analyses[a], analyses[b], plan, pcand["compatibility"], [])
        decs = []
        top = props["proposals"][0] if props["proposals"] else None
        if top is not None and top["rule"] in v3.CONTEXT_RULES and top["technique"] in v3.RENDERABLE:
            decs += [v3.creative_decision(p) for p in props["proposals"] if p["rule"] in v3.CONTEXT_RULES and p["technique"] in v3.RENDERABLE]
        d0 = c0["decision"]
        decs += [{**d, "type_key": d.get("type_key", d["type"])} for d in [d0] + list(d0.get("fallbacks") or [])]
        for dec in decs:
            try:
                res, decks = v3.render_v3(dec, A, B, plan, prov)
            except (dj.DJRenderError, ValueError, KeyError) as exc:
                tried.append({"candidate": rank, "type": dec.get("type_key"), "error": str(exc)[:200]})
                continue
            qa = v3.qa_v3(res, A, B, decks, tracker)
            crt = v3.critic(res, dec["type"], f"ref_{a}__{b}_{dec['type_key']}", decks=res.decks, qa=qa, declared_gap=v3.declared_gap(dec))
            tried.append({"candidate": rank, "type": dec.get("type_key"), "qa": qa["verdict"], "veto": crt.get("veto"),
                          "a_out_s": plan.get("a_out_s"), "b_in_s": plan.get("b_in_s")})
            if qa["verdict"] == "fail" or crt.get("veto"):
                continue
            g = ds.clip_geometry(res.metadata, len(res.audio))
            clip, tm = v3.trim_to_rule(res.audio, dec["type"], g["transition_start"] / SR, g["switch"] / SR, g["transition_end"] / SR)
            out = rdir / f"{a}__{b}.wav"
            h = v3.write_clip(clip, out)
            doc = {"source": "set v3 decision on the pair's candidates in cost order (no far-genre filter, no set Viterbi)",
                   "candidate_rank": rank, "technique": dec["type_key"], "label": dec.get("label"), "rule": dec.get("rule"), "qa": qa["verdict"],
                   "qa_reasons": [f"{r_['check']}: {r_['message']}" for r_ in qa["reasons"]], "critic_veto": crt.get("veto"),
                   "audio": base.rel(out), "sha256": h, "markers": tm, "kind": dec["type"], "duration_s": len(clip) / SR,
                   "plan": {k: plan.get(k) for k in ("a_out_s", "b_in_s", "a_out_bar", "b_in_bar", "type")}, "tried": tried}
            break
        if doc is not None:
            break
    if doc is None:
        doc = {"source": "none", "why": "tutte le decisioni del set v3 bocciate da QA o veto", "tried": tried}
    dump(jp, doc)
    del A, B
    gc.collect()
    return doc


def render() -> None:
    pdoc = load(OUT / "pairs.json")
    tracker = dj_qa.BeatThisTracker()
    for pc in pdoc["chosen"]:
        check_disk("render", 0.8)
        a, b = pc["a"], pc["b"]
        print(f"== {a} -> {b}", flush=True)
        idx = OUT / "renders" / f"{a}__{b}" / "index.json"
        done = load(idx) if idx.is_file() else {"pair": [a, b], "pair_selection": pc}
        try:
            render_pair(pc, tracker, done)
        except Exception as exc:  # keep going with the other pairs, record the error
            done["error"] = f"{type(exc).__name__}: {exc}"
            done["traceback"] = traceback.format_exc()[-2000:]
            dump(idx, done)
            print(f"  ERRORE {done['error']}", flush=True)
        try:
            ref = reference(pc, tracker)
            print(f"  riferimento set v3: {ref.get('technique')} ({ref.get('source')})", flush=True)
        except Exception as exc:
            dump(OUT / "references" / f"{a}__{b}.json", {"source": "none", "why": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]})
            print(f"  riferimento: ERRORE {exc}", flush=True)
        gc.collect()


# ---------------------------------------------------------------- select
def select(min_clips: int = 12, max_clips: int = 16) -> dict:
    import soundfile as sf

    pdoc = load(OUT / "pairs.json")
    cands = []
    for pc in pdoc["chosen"]:
        idx = OUT / "renders" / f"{pc['a']}__{pc['b']}" / "index.json"
        if not idx.is_file():
            continue
        for name, r in load(idx).get("renders", {}).items():
            if r.get("status") == "ok" and r.get("audio") and r.get("clip_rule_ok"):
                cands.append(dict(r, pair=(pc["a"], pc["b"]), mash=pc["score"]))
    # greedy diversity: rarest technique first, at most 2 clips per technique while others are available, at most 2 per pair,
    # every pair at least once; QA pass before warn [H]
    rng = random.Random(SEED)
    rng.shuffle(cands)
    # [user] priorities (votes relayed 21/10/2026): continuous beat and sound links first, the freeze (beat stops) last
    prio = {"lyric_link": 0, "acapella_mashup": 0, "beat_swap": 0, "call_response": 0, "hook_teaser": 1, "vocal_loop": 1,
            "vocal_chop": 1, "reverse_swell": 2, "vocal_swap_shift": 2, "pivot_freeze": 3}
    cands.sort(key=lambda r: (r["qa"]["verdict"] != "pass", prio.get(r["technique"], 2), -r["mash"]))
    chosen: list[dict] = []
    per_t: dict = {}
    per_p: dict = {}

    def ok_add(r, cap_t, cap_p):
        return r not in chosen and per_t.get(r["technique"], 0) < cap_t and per_p.get(r["pair"], 0) < cap_p

    def add(r):
        chosen.append(r)
        per_t[r["technique"]] = per_t.get(r["technique"], 0) + 1
        per_p[r["pair"]] = per_p.get(r["pair"], 0) + 1

    for cap_t, cap_p in ((1, 1), (2, 1), (2, 2), (3, 2)):
        pairs_first = sorted({r["pair"] for r in cands}, key=lambda p: per_p.get(p, 0))
        for p in pairs_first:
            opts = [r for r in cands if r["pair"] == p and ok_add(r, cap_t, cap_p)]
            opts.sort(key=lambda r: (per_t.get(r["technique"], 0), prio.get(r["technique"], 2), r["qa"]["verdict"] != "pass"))
            if opts and len(chosen) < max_clips:
                add(opts[0])
        if len(chosen) >= min_clips and len(per_t) >= 6:
            break
    gdir = OUT / "gallery"
    (gdir / "clips").mkdir(parents=True, exist_ok=True)
    groups, key = [], {}
    by_pair: dict = {}
    for r in chosen:
        by_pair.setdefault(r["pair"], []).append(r)
    feats = {x["slug"]: x for x in load(OUT / "features.json")["tracks"]}
    for gi, pc in enumerate(pdoc["chosen"]):
        p = (pc["a"], pc["b"])
        if p not in by_pair:
            continue
        gid = f"m{gi + 1:02d}"
        items, rec = [], {"group": gid, "pair": list(p)}
        rs = list(by_pair[p])
        rng.shuffle(rs)
        for r in rs:
            audio, sr = sf.read(str(ROOT / r["audio"]), dtype="float32", always_2d=True)
            seed_s = f"{SEED}{gid}{r['name']}"
            cid = f"{gid}_{hashlib.sha1(seed_s.encode()).hexdigest()[:5]}"
            wav = gdir / "clips" / f"{cid}.wav"
            h = v3.write_clip(audio, wav)
            t0, sw, t1 = r["transition_s"]
            items.append({"id": cid, "kind": r["technique"], "role": "versione", "label": TECH_LABEL.get(r["technique"], r["technique"]),
                          "params": {"technique": r["technique"], **{k: v for k, v in (r.get("params") or {}).items() if not isinstance(v, (list, dict))}},
                          "why": "dj_mashup (galleria creativa v2)", "audio_path": base.rel(wav), "duration_s": len(audio) / sr, "sample_rate": sr,
                          "transition_start_s": t0, "switch_s": sw, "transition_end_s": t1,
                          "qa": {"verdict": r["qa"]["verdict"], "warn_reasons": r["qa"]["reasons"]},
                          "critic": {k: r["critic"].get(k) for k in ("score", "flags", "veto", "diagnostic_flags")}, "sha256": h})
            rec[cid] = {"technique": r["technique"], "name": r["name"], "render": r["audio"], "qa": r["qa"]["verdict"],
                        "shift": (r.get("keys") or {}).get("shift_a_material", (r.get("keys") or {}).get("shift_b_material")),
                        "a_out_s": r["a_out_s"], "b_in_s": r["b_in_s"], "duration_s": len(audio) / sr}
        ref_p = OUT / "references" / f"{p[0]}__{p[1]}.json"
        ref = load(ref_p) if ref_p.is_file() else {"source": "none"}
        if ref.get("audio"):
            audio, sr = sf.read(str(ROOT / ref["audio"]), dtype="float32", always_2d=True)
            rid = f"{gid}_rif"
            m = ref["markers"]
            items.append({"id": rid, "kind": ref.get("kind", ref["technique"]), "role": "riferimento",
                          "label": "riferimento: scelta del set v3 (non votabile)",
                          "params": {"technique": ref["technique"]}, "why": ref["source"], "audio_path": ref["audio"],
                          "duration_s": len(audio) / sr, "sample_rate": sr, "transition_start_s": m["t0"], "switch_s": m["sw"],
                          "transition_end_s": max(m["t1"], m["sw"]), "qa": {"verdict": ref.get("qa"), "warn_reasons": ref.get("qa_reasons", [])},
                          "sha256": ref["sha256"]})
            rec[rid] = {"technique": ref["technique"], "source": ref["source"], "role": "riferimento"}
        fa, fb = feats[p[0]], feats[p[1]]
        groups.append({"id": gid, "title": f"{fa['title']} → {fb['title']}",
                       "pair": {"a": {"slug": p[0], "title": fa["title"], "artist": fa["artist"]}, "b": {"slug": p[1], "title": fb["title"], "artist": fb["artist"]}},
                       "bpm": {"a": fa["bpm"], "b": fb["bpm"]}, "key": {"a": fa["camelot"], "b": fb["camelot"], "a_name": fa["key"], "b_name": fb["key"]},
                       "A_out_s": pc["plan"]["a_out_s"], "B_in_s": pc["plan"]["b_in_s"], "A_out_bar": pc["plan"]["a_out_bar"], "B_in_bar": pc["plan"]["b_in_bar"],
                       "sections": {"a": pc["plan"].get("a_section"), "b": pc["plan"].get("b_section")},
                       "plan": {"type": "mashup", "score": pc["score"]},
                       "reasons": [f"mashability {pc['score']:.2f}; tonalità oggi {pc['key_relation']}; trasposizione del materiale di A "
                                   f"{pc['vertical']['best_shift']:+d} st"], "items": items})
        key[gid] = rec
    manifest = {"schema_version": "musicmix.dj-review.1", "title": "Galleria creativa v2: mashup e transizioni con la musica dei due brani",
                "run": RUN, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": {"pairs": base.rel(OUT / "pairs.json"), "module": mx.DJ_MASHUP_VERSION},
                "conventions": {"clip_rule": "blend/effetti/mashup: 10 s prima dell'inizio della transizione + transizione + 10 s dopo (AGENTS.md)",
                                "blind": "ordine casuale (seed in blind_key.json), tecnica rivelata dopo il voto",
                                "reference": "la voce 'riferimento' è la scelta del set v3 sulla stessa coppia: solo confronto, non si vota",
                                "no_continue_A": "nessuna clip 'continua A' da votare"},
                "counts": {"groups": len(groups), "clips": sum(1 for g in groups for i in g["items"] if i["role"] == "versione"),
                           "references": sum(1 for g in groups for i in g["items"] if i["role"] == "riferimento"),
                           "techniques": sorted(per_t), "per_technique": per_t},
                "groups": groups}
    dump(gdir / "review_manifest.json", manifest)
    dump(gdir / "blind_key.json", {"seed": SEED, "note": "NON pubblicare: chiave delle versioni", "groups": key,
                                    "candidates": [{"pair": list(r["pair"]), "name": r["name"], "qa": r["qa"]["verdict"]} for r in cands]})
    print(f"selezione: {manifest['counts']}", flush=True)
    return manifest


def page(out: Path) -> dict:
    import build_listening_artifact as bla

    man = OUT / "gallery" / "review_manifest.json"
    listing = bla.build(man, out, run=RUN, blind=True, overwrite=True)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"pagina {out / 'index.html'}: {len(listing['included'])} clip, {total / 1e6:.1f} MB", flush=True)
    if total >= 40e6:
        raise SystemExit(f"pagina troppo grande: {total / 1e6:.1f} MB")
    return listing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stages", nargs="+", choices=("features", "pairs", "render", "refs", "select", "page", "cleanup"))
    ap.add_argument("--page-out", default=str(PAGE_OUT))
    args = ap.parse_args(argv)
    for st in args.stages:
        t0 = time.time()
        if st == "features":
            features()
        elif st == "pairs":
            pairs()
        elif st == "render":
            render()
        elif st == "refs":  # references only (e.g. after a failed one was deleted)
            tracker = dj_qa.BeatThisTracker()
            for pc in load(OUT / "pairs.json")["chosen"]:
                ref = reference(pc, tracker)
                print(f"{pc['a']} -> {pc['b']}: {ref.get('technique')} ({ref.get('source')})", flush=True)
        elif st == "select":
            select()
        elif st == "page":
            page(Path(args.page_out))
        elif st == "cleanup":
            shutil.rmtree(CACHE / "stems", ignore_errors=True)
        print(f"[{st}] {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
