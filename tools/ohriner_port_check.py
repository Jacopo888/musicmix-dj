#!/usr/bin/env python3
"""Verifica della reimplementazione Python del modello di Ohriner (2019) in ``musicmix.dj_phrase``.

Confronta, sul corpus dell'autore (clone locale di github.com/mohriner/flowBook @ faaec231, NON
copiato nel repo), le nostre funzioni con i dati derivati pubblicati dall'autore:

1. ``ohriner_accents_official`` contro la colonna ``accent`` di ``DerivedData/FlowBookCorpus.txt``
   sulle sillabe non corrette a mano (``toggled == FALSE``);
2. ``groove_instances`` + ``groove_segments`` (calcolati dalla colonna ``accent`` dell'autore) contro
   ``DerivedData/corpus_groove_segments.txt``: quota di sedicesimi con la stessa (classe, rotazione);
3. ``ohriner_global_features`` contro ``DerivedData/CorpusGlobalFeatures.txt`` (correlazione e
   differenza assoluta media).

    python tools/ohriner_port_check.py --out runs/dj_phrase_v2/ohriner_port_check.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from musicmix import dj_phrase as P  # noqa: E402
from musicmix.paths import external_dir  # noqa: E402


def read_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def fnum(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(external_dir() / "flowBook"))
    ap.add_argument("--out", default="runs/dj_phrase_v2/ohriner_port_check.json")
    ap.add_argument("--corpora", default="genre1,eminem,blackThought,kendrickLamar,andre3k,talibKweli,otherEminem,blackThoughtOther")
    ap.add_argument("--max-verses", type=int, default=0)
    args = ap.parse_args()
    repo = Path(args.repo)
    rows = read_tsv(repo / "DerivedData/FlowBookCorpus.txt")
    meta = {r["title"]: r for r in read_tsv(repo / "SourceData/verseMetadata.txt")}
    keep = set(args.corpora.split(","))
    verses = defaultdict(list)
    for r in rows:
        if r["corpus"] in keep:
            verses[r["verse"]].append(r)
    names = sorted(verses)
    if args.max_verses:
        names = names[:args.max_verses]
    # ---- 1. accents
    agree = tot = 0
    agree_all = tot_all = 0
    per_verse_sylls = {}
    for v in names:
        sy = []
        for r in verses[v]:
            bi, q, d = fnum(r["beatIndex"]), fnum(r["quant"]), fnum(r["duration"])
            if bi is None or q is None or d is None:
                continue
            ca = r["CMU_accent"]
            sy.append({"beat_index": bi, "dur_beats": 4.0 / q * d,
                       "cmu_accent": None if ca in ("NA", "") else int(float(ca)),
                       "line_end": r["lineEnding"] == "1", "breath_end": r["breathEnding"] == "1",
                       "author_accent": r["accent"] in ("TRUE", "1"), "toggled": r["toggled"] in ("TRUE", "1")})
        per_verse_sylls[v] = sy
        mine = P.ohriner_accents_official(sy)
        for s, a in zip(sy, mine):
            tot_all += 1
            agree_all += int(a == s["author_accent"])
            if not s["toggled"]:
                tot += 1
                agree += int(a == s["author_accent"])
    res = {"schema": "musicmix.ohriner_port_check.1", "repo": P.OHRINER_REPO, "n_verses": len(names),
           "accent": {"agreement_untoggled": agree / max(tot, 1), "n_untoggled": tot,
                      "agreement_all_incl_manual_toggles": agree_all / max(tot_all, 1), "n_all": tot_all}}
    print(json.dumps(res["accent"]), flush=True)
    # ---- 2. grooves and 3. global features
    seg_auth = defaultdict(lambda: defaultdict(list))
    for r in read_tsv(repo / "DerivedData/corpus_groove_segments.txt"):
        seg_auth[r["verse"]][float(r["effort.rate"])].append(
            {"class": r["class"], "rotation": int(float(r["rotation"])), "start": int(r["start"]), "end": int(r["end"])})
    gf_auth = {r["verse"]: r for r in read_tsv(repo / "DerivedData/CorpusGlobalFeatures.txt")}
    lab_agree = defaultdict(lambda: [0, 0])
    feat_pairs = defaultdict(list)
    t0 = time.time()
    per_verse = {}
    per_verse_agree: dict = {}
    for k, v in enumerate(names):
        sy = per_verse_sylls[v]
        if not sy:
            continue
        acc = [s["author_accent"] for s in sy]
        inst = P.groove_instances(P.l3_durations(sy, acc))
        m = int(round(4 * max(s["beat_index"] for s in sy)))
        segs = P.groove_segments(inst, max_unit=m)
        for rate, mine in segs.items():
            auth = seg_auth.get(v, {}).get(rate)
            if auth is None:
                continue
            la = {}
            for s in auth:
                for j in range(s["start"], s["end"] + 1):
                    la[j] = (s["class"], s["rotation"])
            lm = {}
            for s in mine:
                for j in range(s["start"], s["end"] + 1):
                    lm[j] = (s["class"], s["rotation"])
            keys = set(la) | set(lm)
            ok = sum(1 for j in keys if la.get(j) == lm.get(j))
            lab_agree[rate][0] += ok
            lab_agree[rate][1] += len(keys)
            per_verse_agree.setdefault(v, {})[str(rate)] = ok / max(1, len(keys))
        bpm = fnum(meta.get(v, {}).get("bpm"))
        f = P.ohriner_global_features(sy, acc, tempo_bpm=bpm, segments=segs)
        a = gf_auth.get(v)
        if a:
            for mk, ak in (("saturation", "saturation"), ("sat_delta", "sat.delta"), ("syls_per_second", "sylsPerSecond"),
                           ("grooviness", "grooviness"), ("adherence", "adherence"), ("dupleness", "dupleness")):
                if f.get(mk) is not None and fnum(a.get(ak)) is not None:
                    feat_pairs[mk].append((f[mk], fnum(a[ak])))
        per_verse[v] = {k2: f.get(k2) for k2 in ("saturation", "sat_delta", "syls_per_second", "grooviness", "adherence", "dupleness",
                                             "groove_complexity")}
        if k % 20 == 0:
            print(f"{k}/{len(names)} {v} [{time.time() - t0:.0f}s]", flush=True)
    res["groove_segments_label_agreement"] = {str(r): {"agreement": a / max(n, 1), "n_positions": n} for r, (a, n) in sorted(lab_agree.items())}
    res["global_features_vs_author"] = {}
    for k2, pairs in feat_pairs.items():
        x, y = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        res["global_features_vs_author"][k2] = {"pearson_r": float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else None,
                                                "mean_abs_diff": float(np.mean(np.abs(x - y))), "author_mean": float(y.mean()),
                                                "ours_mean": float(x.mean()), "n": len(x)}
    res["per_verse"] = per_verse
    res["per_verse_segment_agreement"] = per_verse_agree
    a0 = [d.get("0.0") for d in per_verse_agree.values() if d.get("0.0") is not None]
    if a0:
        res["segment_agreement_effort0_quantiles"] = {q: float(np.quantile(a0, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
        res["verses_with_effort0_agreement_ge_0.95"] = sum(1 for x in a0 if x >= 0.95)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k2: res[k2] for k2 in ("accent", "groove_segments_label_agreement", "global_features_vs_author")}, indent=1))


if __name__ == "__main__":
    main()
