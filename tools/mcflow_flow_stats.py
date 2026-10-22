#!/usr/bin/env python3
"""Verifiche su MCFlow delle regole tratte dai libri (Ohriner 2019, Krims 2000) per ``dj_phrase``.

Legge il clone locale del corpus MCFlow (Condit-Schultz 2017; NON copiato nel repo, vedi
``tools/mcflow_phrase_stats.py``) e salva solo statistiche derivate:

1. accordo tra l'algoritmo d'accento di Ohriner (pp. 57-59, implementato in
   ``dj_phrase.ohriner_accents`` con l'accento lessicale CMUdict, come sull'audio) e l'accento
   annotato a mano in MCFlow (``**stress``);
2. distribuzione delle sette classi di groove (``dj_phrase.best_groove``) contro la Tab. 4.3 di
   Ohriner (p. 101), e i due esempi del libro presenti anche in MCFlow ("The Breaks",
   "Lose Yourself");
3. descrittori di flow per brano e tendenze con l'anno di uscita (Ohriner p. 107: tempo in calo,
   saturazione in crescita, velocità sillabica costante; Krims p. 49: stili "effusive" più recenti),
   punteggi di Krims e controllo sui Beastie Boys (esempio di "sung style", Krims p. 50);
4. modello di fine frase held-out (stessa divisione per brano di ``mcflow_phrase_stats.heldout``)
   con i canali nuovi: posizione dell'attacco della parola seguente (Ohriner pp. 108-110: le frasi
   iniziano vicino al battere), accento dell'ultima sillaba, adattamento al "pilastro" del brano.

    python tools/mcflow_flow_stats.py \
        --out runs/dj_phrase_v2/mcflow_books_eval.json --config configs/dj_phrase_flow_v1.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import mcflow_phrase_stats as M  # noqa: E402
from musicmix import dj_phrase as P  # noqa: E402
from musicmix.paths import external_dir  # noqa: E402

SR = 48000


def release_year(path: Path) -> int | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("!!!RRD:"):
            m = re.search(r"(19|20)\d\d", line)
            return int(m.group(0)) if m else None
    return None


def tempo_bpm(path: Path) -> float | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\*MM(\d+(?:\.\d+)?)", line)
        if m:
            return float(m.group(1))
    return None


def song(path: Path, lex: P.Lexicon) -> dict | None:
    """Syllables (dj_phrase format), pseudo-model and word-boundary records of one MCFlow file."""
    ev = M.parse(path)
    bpm = tempo_bpm(path)
    if not ev or not bpm:
        return None
    q_s = 60.0 / bpm
    sylls_ev = [e for e in ev if not e["rest"]]
    for s in sylls_ev:
        s["bound_before"] = bool(s["break"] is not None and s["break"] >= 3)
    for i, s in enumerate(sylls_ev[:-1]):
        if s["break"] is not None and s["break"] <= -3:
            sylls_ev[i + 1]["bound_before"] = True
    words = M.words_of(ev)
    # bar numbers restart at every verse in MCFlow (=0, =1, ... after each *>VerseN): index bars by their start time
    starts_q = sorted({round(e["t"] - e["pos"], 6) for e in ev if e["bar"] is not None})
    bar_idx = {b: k for k, b in enumerate(starts_q)}
    for e in ev:
        if e["bar"] is not None:
            e["bar"] = bar_idx[round(e["t"] - e["pos"], 6)]
    downbeats = starts_q
    sylls, mw = [], []
    for wi, w in enumerate(words):
        text = "".join(s["lyr"].strip("-") for s in w["syl"])
        lx = lex.analyse(text, "en")
        n = len(w["syl"])
        si = lx["stress_idx"] if lx["n_syl"] == n else 0  # [H] CMU syllable count differs: first syllable (English left-dominant feet, Ohriner p. 52)
        rh_annot = any(re.search(r"[A-Za-z]", s["rhyme"] or "") for s in w["syl"])
        for k, s in enumerate(w["syl"]):
            if s["bar"] is None:
                continue
            p16 = s["pos"] * 4.0
            r = int(round(p16))
            bar_r, pos_r = (s["bar"] + 1, r - 16) if r >= 16 else (s["bar"], r)
            sylls.append({"word_i": wi, "idx": k, "n_syl": n, "stress_idx": si, "word": lx["word"], "t_s": s["t"] * q_s,
                          "bar": bar_r, "pos16": pos_r, "abs16": 16 * bar_r + pos_r, "off_grid": abs(p16 - r),
                          "dur_beats": s["dur"], "stress_annot": s["stress"] == "1", "bound_before": s["bound_before"]})
        last = w["syl"][-1]
        mw.append({"on": int(w["syl"][0]["t"] * q_s * SR), "off": int((last["t"] + last["dur"]) * q_s * SR),
                   "last_syl_on": int(last["t"] * q_s * SR), "n_syl": n, "stress_idx": si, "clean": lx["word"], "text": text,
                   "rhyme": {"score": 1.0 if rh_annot else 0.0}})
    # lines = prosodic phrases (break >= 3)
    lines, start = [], 0
    starts = [wi for wi, w in enumerate(words) if w["syl"][0]["bound_before"]]
    for k, wi in enumerate(starts):
        nxt = starts[k + 1] if k + 1 < len(starts) else len(words)
        if nxt - 1 >= wi:
            lines.append({"first_word": wi, "last_word": nxt - 1, "start_sample": mw[wi]["on"], "end_sample": mw[nxt - 1]["off"]})
    gaps = []
    for wi in range(len(words) - 1):
        gap16 = (words[wi + 1]["syl"][0]["t"] - (words[wi]["syl"][-1]["t"] + words[wi]["syl"][-1]["dur"])) * 4
        y = words[wi + 1]["syl"][0]["bound_before"]
        gaps.append({"after": wi, "end_sample": mw[wi]["off"], "rest": gap16 >= 0.5, "p_end": 1.0 if y else 0.0})
    model = {"sample_rate": SR, "period_s": q_s, "downbeats_s": [d * q_s for d in downbeats], "words": mw, "lines": lines,
             "gaps": gaps}
    return {"name": path.name, "artist": path.name.split("_")[0], "year": release_year(path), "bpm": bpm,
            "sylls": sylls, "model": model, "words_ev": words}


# ------------------------------------------------------------- 1. accents
def accent_agreement(songs: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    tp_f = fp_f = fn_f = tn_f = 0
    for s in songs:
        sy = [x for x in s["sylls"] if x["off_grid"] < 1e-6]
        a = P.ohriner_accents(sy)
        af = P.ohriner_accents(sy, deaccent_function_words=True)
        for x, y, yf in zip(sy, a, af):
            t = x["stress_annot"]
            tp += y and t
            fp += y and not t
            fn += (not y) and t
            tn += (not y) and (not t)
            tp_f += yf and t
            fp_f += yf and not t
            fn_f += (not yf) and t
            tn_f += (not yf) and (not t)
    n = tp + fp + fn + tn
    return {"n_syllables_on_grid": n,
            "ohriner_cmudict": {"accuracy": (tp + tn) / n, "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
                                "annotated_accent_rate": (tp + fn) / n, "predicted_accent_rate": (tp + fp) / n},
            "ohriner_cmudict_function_word_rule": {"accuracy": (tp_f + tn_f) / n, "precision": tp_f / max(1, tp_f + fp_f),
                                                   "recall": tp_f / max(1, tp_f + fn_f)},
            "reference": "Ohriner 2019 p. 62: l'algoritmo è 'giusto' nel 96 % dei casi rispetto al suo ascolto (sul suo corpus); "
                         "qui il confronto è con l'accento annotato in MCFlow da Condit-Schultz, con CMUdict al posto del CMUPD sillabato"}


# --------------------------------------------------------------- 2. grooves
def groove_stats(songs: list[dict], use_annot: bool) -> dict:
    by_eff = {0: Counter(), 2: Counter(), 4: Counter(), 8: Counter()}
    per_song = {}
    for s in songs:
        sy = s["sylls"]
        acc = [x["stress_annot"] for x in sy] if use_annot else P.ohriner_accents(sy)
        bg = [b for b in P.bar_grooves(sy, acc) if "groove_class" in b]
        per_song[s["name"]] = Counter(b["groove_class"] for b in bg if b["groove_effort"] <= 4)
        for b in bg:
            for lim in by_eff:
                if b["groove_effort"] <= lim:
                    by_eff[lim][b["groove_class"]] += 1
    out = {}
    for lim, c in by_eff.items():
        tot = sum(c.values())
        out[f"effort_le_{lim}"] = {"n_bars": tot, "share": {k: v / tot for k, v in c.most_common()}}
    return out, per_song


def book_examples(songs: list[dict]) -> dict:
    out = {}
    for name in ("KurtisBlow_TheBreaks.rap", "Eminem_LoseYourself.rap"):
        s = next((x for x in songs if x["name"] == name), None)
        if s is None:
            continue
        sy = s["sylls"]
        acc = P.ohriner_accents(sy)
        bg = [b for b in P.bar_grooves(sy, acc) if "groove_class" in b]
        c = Counter((b["groove_class"], b["groove_start"]) for b in bg if b["groove_effort"] <= 2)
        out[name] = {"n_bars": len(bg), "top_grooves_effort_le_2": [[f"{k[0]}@{k[1]}", v] for k, v in c.most_common(4)],
                     "bars": [[b["bar"], b["groove_class"], b["groove_start"], b["groove_effort"]] for b in bg[:40]]}
    out["reference"] = ("Ohriner 2019 Tab. 5.1 p. 123: 'The Breaks' (Kurtis Blow) 92 % in <332_332>; pp. 83, 98: seconda metà della "
                        "prima strofa di 'Lose Yourself' in <332_332> da posizione 4 (trascrizioni di Ohriner, non di MCFlow)")
    return out


# ------------------------------------------------------- 3. features/years
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return None, len(x)
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]), int(len(x))


def song_features(songs: list[dict]) -> list[dict]:
    rows = []
    for s in songs:
        acc = P.ohriner_accents(s["sylls"])
        f = P.flow_features(s["model"], sylls=s["sylls"], accents=acc)
        ends = {L["last_word"] for L in s["model"]["lines"]}
        # nPVI three ways: annotated syllable onsets (as in flow_features), annotated durations, and the audio
        # approximation (syllables uniform inside the word, as dj_phrase does on Whisper words)
        f["npvi_duration"] = P.syllable_npvi(s["sylls"], line_end_words=ends, times="duration")
        f["npvi_uniform"] = P.syllable_npvi(P.model_syllables(s["model"]), line_end_words=ends)
        rows.append({"name": s["name"], "artist": s["artist"], "year": s["year"], "bpm": s["bpm"], **{k: v for k, v in f.items()
                     if isinstance(v, (int, float)) or v is None}})
    return rows


KEYS = ["saturation", "saturation_change", "syllable_speed", "onbeat_accent", "offbeat_onset", "three_share", "duple_share",
        "groove_adherence", "grooviness", "end_beat4", "end_R", "enjambment", "rhyme_density", "articulation", "caesura_rate",
        "first_bar_caesura", "npvi", "npvi_duration", "npvi_uniform", "hypermeter_phase_share"]


def reference(rows: list[dict]) -> dict:
    ref = {}
    for k in KEYS:
        v = np.asarray([r[k] for r in rows if r.get(k) is not None], float)
        if v.size:
            ref[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 1.0, "n": int(v.size)}
    return ref


# ------------------------------------------------------ 4. line-end model
def next_class(next_pos_q: float) -> str:
    """Onset of the next word relative to the bar: within one beat before a downbeat, on it (+-1/16), elsewhere."""
    to_db = 4.0 - (next_pos_q % 4.0)
    if next_pos_q % 4.0 < 0.25 - 1e-9:
        return "db"
    if to_db <= 1.0 + 1e-9:
        return "pickup"
    return "other"


def line_end_records(path: Path, s: dict) -> list[dict]:
    recs = M.detected_rhyme_records(path)  # same order as words_of pairs
    words = s["words_ev"]
    acc_by_word = {}
    sy = s["sylls"]
    acc = P.ohriner_accents(sy)
    for x, a in zip(sy, acc):
        if x["idx"] == x["n_syl"] - 1:
            acc_by_word[x["word_i"]] = a
    for wi, r in enumerate(recs):
        nx = words[wi + 1]["syl"][0]
        last = words[wi]["syl"][-1]
        r["next"] = next_class(nx["pos"])
        r["last_acc"] = bool(acc_by_word.get(wi, False))
        r["pos16"] = int(math.floor(last["pos"] * 4 + 1e-6)) % 16
    return recs


def auc_f1(scores, ys):
    tp = sum(1 for a, b in zip(scores, ys) if a > 0 and b)
    fp = sum(1 for a, b in zip(scores, ys) if a > 0 and not b)
    fn_ = sum(1 for a, b in zip(scores, ys) if a <= 0 and b)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn_, 1)
    return {"auc": M._auc(scores, ys), "precision_at_map": prec, "recall_at_map": rec,
            "f1_at_map": 2 * prec * rec / max(prec + rec, 1e-9), "n": len(ys)}


def heldout_new_terms(per: dict[str, list[dict]], folds: int = 2) -> dict:
    names = sorted(per)
    base_keys = lambda r: (f"{r['beat']}|{r['rest']}", f"rh{r['rhyme']}|{r['rest']}", f"l{r['leng']}")  # noqa: E731
    variants = {
        "base(pos_gap+rhyme_det+leng)": base_keys,
        "+next_onset": lambda r: base_keys(r) + (f"nx{r['next']}|{r['rest']}",),
        "+last_accent": lambda r: base_keys(r) + (f"acc{r['last_acc']}|{r['rest']}",),
        "+next_onset+last_accent": lambda r: base_keys(r) + (f"nx{r['next']}|{r['rest']}", f"acc{r['last_acc']}|{r['rest']}"),
    }
    out = {}
    tables_full = {}
    scores = {k: ([], []) for k in list(variants) + ["base+pillar(w=1)", "base+pillar(w=0.5)", "+next_onset+pillar(w=0.5)"]}
    for k in range(folds):
        train = [n for i, n in enumerate(names) if i % folds != k]
        test = [n for i, n in enumerate(names) if i % folds == k]
        tr = [r for n in train for r in per[n]]
        base = sum(r["y"] for r in tr) / len(tr)
        lb = math.log(base / (1 - base))
        fitted = {}
        for name, fn in variants.items():
            tabs = defaultdict(lambda: [1, 2])
            for r in tr:
                for kk in fn(r):
                    tabs[kk][0] += int(r["y"])
                    tabs[kk][1] += 1
            fitted[name] = tabs

        def sc(name, r):
            lo = lb
            for kk in variants[name](r):
                b, nn = fitted[name][kk]
                p = b / nn
                lo += math.log(p / (1 - p)) - lb
            return lo

        # corpus density of predicted line ends (train songs), per position
        def pillar_density(recs, name):
            d = np.full(16, 0.5)
            for r in recs:
                if sc(name, r) > 0:
                    d[r["pos16"]] += 1
            # circular smoothing, one sixteenth [H]
            d = 0.25 * np.roll(d, 1) + 0.5 * d + 0.25 * np.roll(d, -1)
            return d / d.sum()
        dens_corpus = {nm: pillar_density(tr, nm) for nm in ("base(pos_gap+rhyme_det+leng)", "+next_onset")}
        for n in test:
            recs = per[n]
            for name in variants:
                for r in recs:
                    scores[name][0].append(sc(name, r))
                    scores[name][1].append(r["y"])
            for nm, w, key in (("base(pos_gap+rhyme_det+leng)", 1.0, "base+pillar(w=1)"),
                               ("base(pos_gap+rhyme_det+leng)", 0.5, "base+pillar(w=0.5)"),
                               ("+next_onset", 0.5, "+next_onset+pillar(w=0.5)")):
                ds = pillar_density(recs, nm)  # transductive: no labels of the test song
                for r in recs:
                    scores[key][0].append(sc(nm, r) + w * math.log(ds[r["pos16"]] / dens_corpus[nm][r["pos16"]]))
                    scores[key][1].append(r["y"])
    for name, (s, y) in scores.items():
        out[name] = auc_f1(s, y)
    # full-corpus tables for the new channels (for the config)
    allr = [r for n in names for r in per[n]]
    for key, fn in (("next_onset_gap", lambda r: f"{r['next']}|{'rest' if r['rest'] else 'none'}"),
                    ("last_accent_gap", lambda r: f"{r['last_acc']}|{'rest' if r['rest'] else 'none'}")):
        c = defaultdict(lambda: [0, 0])
        for r in allr:
            c[fn(r)][0] += int(r["y"])
            c[fn(r)][1] += 1
        tables_full[key] = {k: {"p": b / n, "n": n} for k, (b, n) in sorted(c.items())}
    # density of predicted line ends (base model fitted on all songs): denominator of the pillar term in dj_phrase
    base_all = sum(r["y"] for r in allr) / len(allr)
    lb_all = math.log(base_all / (1 - base_all))
    tabs = defaultdict(lambda: [1, 2])
    for r in allr:
        for kk in base_keys(r):
            tabs[kk][0] += int(r["y"])
            tabs[kk][1] += 1
    d = np.full(16, 0.5)
    for r in allr:
        lo = lb_all + sum(math.log((tabs[kk][0] / tabs[kk][1]) / (1 - tabs[kk][0] / tabs[kk][1])) - lb_all for kk in base_keys(r))
        if lo > 0:
            d[r["pos16"]] += 1
    d = 0.25 * np.roll(d, 1) + 0.5 * d + 0.25 * np.roll(d, -1)
    tables_full["pillar_corpus_density"] = (d / d.sum()).tolist()
    return {"folds": folds, "split": "by song, alphabetical index mod folds (as mcflow_phrase_stats.heldout)",
            "models": out}, tables_full


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(external_dir() / "computational-cognitive-musicology-lab/mcflow"))
    ap.add_argument("--out", default="runs/dj_phrase_v2/mcflow_books_eval.json")
    ap.add_argument("--config", default="configs/dj_phrase_flow_v1.json")
    args = ap.parse_args()
    files = sorted((Path(args.corpus) / "Humdrum").glob("*.rap"))
    lex = P.Lexicon()
    songs = [s for s in (song(f, lex) for f in files) if s]
    res = {"schema": "musicmix.dj_phrase_flow_eval.1", "n_songs": len(songs),
           "corpus": "MCFlow (Condit-Schultz 2017), clone locale; solo statistiche derivate"}
    res["accent_agreement"] = accent_agreement(songs)
    g_alg, per_song = groove_stats(songs, use_annot=False)
    g_ann, _ = groove_stats(songs, use_annot=True)
    res["groove_distribution"] = {"ohriner_accents_cmudict": g_alg, "mcflow_annotated_stress": g_ann,
                                  "ohriner_tab_4_3_effort_0": P.BOOK_RULES["groove_class_share"].value}
    res["book_examples"] = book_examples(songs)
    rows = song_features(songs)
    ref = reference(rows)
    for r in rows:
        st = P.krims_style(r, {k: (v["mean"], v["sd"]) for k, v in ref.items()})
        r.update({f"krims_{k}": v for k, v in st["scores"].items()}, krims_dominant=st["dominant"])
    years = [r["year"] for r in rows]
    trends = {}
    for k in KEYS + ["bpm", "krims_sung", "krims_percussion_effusive", "krims_speech_effusive"]:
        rho, n = spearman(years, [r.get(k) if r.get(k) is not None else np.nan for r in rows])
        trends[k] = {"spearman_rho_vs_year": rho, "n": n}
    res["trends_vs_year"] = trends
    res["trend_reference"] = ("Ohriner 2019 p. 107 (75 strofe): tempo in calo con gli anni, saturazione in crescita, velocità sillabica "
                              "costante; Krims 2000 p. 49: stili più veloci e 'complessi' dagli inizi degli anni '90, quelli vecchi persistono")
    by_artist = defaultdict(list)
    for r in rows:
        by_artist[r["artist"]].append(r)
    med = {k: float(np.nanmedian([r[k] for r in rows if r.get(k) is not None])) for k in ("krims_sung", "krims_speech_effusive",
                                                                                      "krims_percussion_effusive")}
    res["krims_check"] = {
        "beastie_boys_sung": [round(r["krims_sung"], 2) for r in by_artist["BeastieBoys"]],
        "beastie_boys_sung_above_median": sum(r["krims_sung"] > med["krims_sung"] for r in by_artist["BeastieBoys"]),
        "beastie_boys_n": len(by_artist["BeastieBoys"]),
        "corpus_median": med,
        "dominant_counts": dict(Counter(r["krims_dominant"] for r in rows)),
        "by_decade_mean": {str(d): {k: float(np.mean([r[k] for r in rows if r["year"] and r["year"] // 10 * 10 == d]))
                                    for k in ("krims_sung", "krims_percussion_effusive", "krims_speech_effusive", "saturation", "bpm",
                                              "syllable_speed")}
                           for d in (1970, 1980, 1990, 2000, 2010) if any(r["year"] and r["year"] // 10 * 10 == d for r in rows)},
        "reference": "Krims 2000 p. 50: i Beastie Boys 'often exemplify sung delivery'"}
    # books v2 checks (Krims p. 50 first-bar caesura; Huron pp. 188-189 nPVI; audio approximation of nPVI)
    def corr(a, b):
        x = np.asarray([r.get(a) for r in rows], float)
        y = np.asarray([r.get(b) for r in rows], float)
        ok = np.isfinite(x) & np.isfinite(y)
        return {"pearson_r": float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 else None, "n": int(ok.sum())}
    res["books_v2_checks"] = {
        "npvi_onset_vs_uniform": corr("npvi", "npvi_uniform"),
        "npvi_onset_vs_duration": corr("npvi", "npvi_duration"),
        "npvi_vs_krims_sung": corr("npvi", "krims_sung"),
        "npvi_vs_krims_speech": corr("npvi", "krims_speech_effusive"),
        "first_bar_caesura_vs_krims_speech_without_it": None,
        "first_bar_caesura_vs_year": trends.get("first_bar_caesura"),
        "npvi_vs_year": trends.get("npvi"),
        "means": {k: ref[k] for k in ("npvi", "npvi_duration", "npvi_uniform", "first_bar_caesura", "hypermeter_phase_share") if k in ref},
        "reference": "Patel & Daniele 2003 (via Huron 2006 pp. 188-189): nPVI vocalico dell'inglese parlato ~57, francese ~44"}
    # first-bar caesura against the speech-effusive score computed without it (no circularity)
    ref_wo = {k: (v["mean"], v["sd"]) for k, v in ref.items() if k != "first_bar_caesura"}
    sp_wo = [P.krims_style(r, ref_wo)["scores"]["speech_effusive"] for r in rows]
    x = np.asarray([r.get("first_bar_caesura") for r in rows], float)
    y = np.asarray(sp_wo, float)
    ok = np.isfinite(x) & np.isfinite(y)
    res["books_v2_checks"]["first_bar_caesura_vs_krims_speech_without_it"] = {
        "pearson_r": float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 else None, "n": int(ok.sum())}
    res["songs"] = rows
    per = {s["name"]: line_end_records(f, s) for f, s in zip([f for f in files], songs) if s}
    held, tabs = heldout_new_terms(per)
    res["line_end_heldout"] = held
    res["line_end_tables"] = tabs
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    cfg = {"schema": "musicmix.dj_phrase_flow.1",
           "source": "tools/mcflow_flow_stats.py su MCFlow (stesso clone e commit di configs/dj_phrase_mcflow_v1.json)",
           "krims_reference": ref, "next_onset_gap": tabs["next_onset_gap"], "last_accent_gap": tabs["last_accent_gap"],
           "pillar_corpus_density": tabs["pillar_corpus_density"],
           "line_end_heldout": held["models"],
           "definitions": {"pillar_corpus_density": "share of predicted line ends (pos_gap+rhyme_det+leng, p >= 0.5) per sixteenth of the bar, "
                                                    "smoothed (0.25, 0.5, 0.25) on the cycle", "next_onset": "db = next word starts on the downbeat (< 1/16 after), pickup = within one beat before a "
                                         "downbeat, other = elsewhere", "krims_reference": "per-song mean/sd of flow_features on MCFlow (124 songs)"}}
    Path(args.config).write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("accent_agreement", "book_examples", "trends_vs_year", "krims_check")}, indent=1,
                     default=float)[:6000])
    print(json.dumps(res["groove_distribution"], indent=1)[:3000])
    print(json.dumps(held, indent=1))


if __name__ == "__main__":
    main()
