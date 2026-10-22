#!/usr/bin/env python3
"""Calibrazione onesta del giudice automatico sui giudizi d'ascolto (schema ``musicmix.dj-review.1``).

Legge una cartella di giudizi salvati dalla pagina d'ascolto (``tools/build_listening_artifact.py``: un JSON per
documento, oppure l'esportazione "Copia JSON" ``{run, collection, judgments: {doc_id: body}}``), i punteggi del
giudice per le stesse clip (``--scores``, oppure calcolati con ``dj_critic.score_transition(..., mode="fast",
version="v4")`` dalle clip del ``--manifest``) e scrive un report con:

* **solo giudizi alla cieca** (``blind: true``) nell'analisi principale (Margulis 2014 p. 15 [E]: una descrizione
  letta prima dell'ascolto cambia il giudizio); quelli non ciechi solo come diagnostica separata (``--include-nonblind``);
* **ascolti ripetuti** (E2; Margulis p. 101 [E], Huron pp. 133-134 [E]): giudizi con ``plays_before_save`` oltre
  ``--max-plays`` esclusi dall'analisi principale e riportati a parte; analisi secondaria pesata con
  ``w = min(1, max_plays / plays)``; per ogni clip conta un solo voto (l'ultimo salvato);
* **selezione separata dalla verifica**: i gruppi (coppie di brani) sono divisi in modo deterministico (sha256 del
  ``group_id``) in due metà; un'eventuale scelta (``--choose``: il predittore migliore sulla metà di selezione) è
  verificata solo sull'altra metà. Di default nessun peso e nessuna soglia vengono stimati;
* Spearman (IC bootstrap sugli item, p di permutazione), AUC voti >= 7 contro <= 4 (IC bootstrap), accuratezza a
  coppie (dentro i gruppi e complessiva, IC bootstrap). Con meno di ``--min-n`` voti il report lo dichiara.

    python tools/calibrate_dj_critic.py --judgments giudizi_cloud/<run> \
        --manifest runs/<run>/review_manifest.json --out runs/<run>/critic_calibration.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from musicmix import dj_critic as C  # noqa: E402
import run_dj_critic_v4 as V4  # noqa: E402

SCHEMA = "musicmix.dj-critic-calibration.1"


# ---------------------------------------------------------------- giudizi
def read_judgments(folder: Path) -> list[dict]:
    """Documenti ``type == "versione"`` con un voto numerico (singoli file o esportazioni "Copia JSON")."""
    docs: list[dict] = []
    for p in sorted(Path(folder).rglob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(d, dict) and isinstance(d.get("judgments"), dict):
            for doc_id, body in d["judgments"].items():
                if isinstance(body, dict):
                    docs.append({"doc_id": doc_id, "_file": str(p), **body})
        elif isinstance(d, dict) and d.get("clip_id"):
            docs.append({"_file": str(p), **d})
    out = [d for d in docs if d.get("type", "versione") == "versione" and isinstance(d.get("score"), (int, float))]
    return out


def latest_per_clip(docs: list[dict]) -> list[dict]:
    """Un voto per clip (l'ultimo salvato): voti ripetuti della stessa clip non sono indipendenti (E2)."""
    best: dict[tuple, dict] = {}
    for d in docs:
        k = (d.get("run"), d["clip_id"])
        if k not in best or str(d.get("saved_at", "")) >= str(best[k].get("saved_at", "")):
            best[k] = d
    return list(best.values())


def half_of(group_id: str) -> str:
    return "selection" if int(hashlib.sha256(str(group_id).encode()).hexdigest(), 16) % 2 == 0 else "verification"


# ---------------------------------------------------------------- punteggi
def score_clips(manifest: Path, clip_ids: set, cache: Path) -> dict:
    """Punteggi v3 + v4 (modo fast) delle clip del manifest dj-review.1, con cache."""
    man = json.loads(manifest.read_text(encoding="utf-8"))
    res = json.loads(cache.read_text()) if cache.exists() else {}
    clips = [it for g in man.get("groups", []) for it in g.get("items", [])]
    for it in clips:
        cid = it["id"]
        if cid not in clip_ids or cid in res:
            continue
        meta_doc = json.loads((ROOT / it["json_path"]).read_text(encoding="utf-8"))
        meta = meta_doc.get("render", meta_doc)
        r = C.score_transition(str(ROOT / it["audio_path"]), meta, mode="fast", version="v4", qa=meta_doc.get("qa"),
                               id=cid)
        res[cid] = {"v3_score": r["score"], "v3_n_flags": len(r["flags"]), "summary": r["v4"]["summary"],
                    "mode": "fast (v3 solo DSP + canali v4)"}
        C.write_json(cache, res)
    return res


def predictor_table(scores: dict) -> dict:
    def get(k, f, neg=False):
        v = scores.get(k) or {}
        x = v.get(f, (v.get("summary") or {}).get(f))
        return None if x is None else (-float(x) if neg else float(x))

    return {
        "v3_score": lambda k: get(k, "v3_score"),
        "v3_n_flags_neg": lambda k: get(k, "v3_n_flags", True),
        "v4_technical_veto_neg": lambda k: get(k, "v4_n_veto", True),
        "v4_technical_issues_neg": lambda k: get(k, "v4_n_issues", True),
        "v4_metric_surprise_neg": lambda k: get(k, "metric_surprise", True),
        "v4_content_surprise_gated_neg": lambda k: get(k, "content_surprise_gated", True),
        "v4_backbeat_neg": lambda k: get(k, "backbeat_ms", True),
    }


def weighted_pairwise(pred: np.ndarray, truth: np.ndarray, w: np.ndarray) -> Optional[float]:
    num = den = 0.0
    for i in range(len(pred)):
        for j in range(i + 1, len(pred)):
            if truth[i] == truth[j] or not (np.isfinite(pred[i]) and np.isfinite(pred[j])):
                continue
            ww = w[i] * w[j]
            d = pred[i] - pred[j]
            num += ww * (0.5 if d == 0 else float(np.sign(d) == np.sign(truth[i] - truth[j])))
            den += ww
    return num / den if den else None


def evaluate(rows: list[dict], preds: dict) -> dict:
    ids = [r["clip_id"] for r in rows]
    y = np.array([float(r["score"]) for r in rows])
    groups = [r.get("group_id") or "" for r in rows]
    w = np.array([float(r.get("weight", 1.0)) for r in rows])
    out: dict[str, Any] = {"n": len(rows), "n_groups": len(set(groups)), "n_ge7": int((y >= 7).sum()), "n_le4": int((y <= 4).sum())}
    res = {}
    for name, fn in preds.items():
        p = np.array([np.nan if fn(k) is None else fn(k) for k in ids], dtype=float)
        pa = C.pairwise_accuracy(p, y)
        pw = C.pairwise_accuracy(p, y, groups=groups, within_groups_only=True)
        res[name] = {
            "n_scored": int(np.isfinite(p).sum()),
            "spearman": V4.spearman_with_ci(p, y),
            "auc_ge7_vs_le4": V4.auc_with_ci(p[y >= 7], p[y <= 4]),
            "pairwise": {"n_pairs": pa["n_pairs"], "accuracy": pa["accuracy"],
                         "bootstrap95_items": C.bootstrap_pairwise_ci(p, y) if pa["n_pairs"] else None},
            "pairwise_within_groups": {"n_pairs": pw["n_pairs"], "accuracy": pw["accuracy"]},
            "pairwise_weighted_plays": weighted_pairwise(p, y, w),
        }
    out["predictors"] = res
    return out


def calibrate(docs: list[dict], scores: dict, *, include_nonblind: bool = False, max_plays: int = 3, min_n: int = 20,
              choose: bool = False) -> dict:
    docs = latest_per_clip(docs)
    for d in docs:
        plays = d.get("plays_before_save")
        d["plays"] = None if plays is None else int(plays)
        d["weight"] = 1.0 if d["plays"] is None or d["plays"] <= max_plays else max_plays / d["plays"]
        d["half"] = half_of(d.get("group_id") or d["clip_id"])
    blind = [d for d in docs if d.get("blind") is True]
    primary = [d for d in blind if d["plays"] is None or d["plays"] <= max_plays]
    repeated = [d for d in blind if d["plays"] is not None and d["plays"] > max_plays]
    nonblind = [d for d in docs if d.get("blind") is not True]
    preds = predictor_table(scores)
    rep: dict[str, Any] = {
        "schema": SCHEMA,
        "counts": {"judgments": len(docs), "blind": len(blind), "primary_blind_le_max_plays": len(primary),
                   "blind_repeated_plays": len(repeated), "not_blind": len(nonblind),
                   "with_scores": sum(1 for d in docs if d["clip_id"] in scores)},
        "rules": {"blind_only": True, "max_plays": max_plays, "one_vote_per_clip": "latest saved_at",
                  "split": "sha256(group_id) % 2: selection / verification", "fitting": "nessuno" if not choose else
                  "scelta del predittore singolo migliore sulla selezione, verificata sulla verifica",
                  "sources": "Margulis 2014 p. 15 [E] (blind); Margulis p. 101, Huron pp. 133-134 [E] (ascolti ripetuti)"},
        "primary": evaluate(primary, preds),
        "primary_weighted_all_blind": evaluate(blind, preds),
        "halves": {h: evaluate([d for d in primary if d["half"] == h], preds) for h in ("selection", "verification")},
    }
    if repeated:
        rep["repeated_plays_only"] = evaluate(repeated, preds)
    if include_nonblind and nonblind:
        rep["not_blind_diagnostic"] = evaluate(nonblind, preds)
    if choose:
        sel = rep["halves"]["selection"]["predictors"]
        best = max(sel, key=lambda k: (sel[k]["pairwise"]["accuracy"] if sel[k]["pairwise"]["accuracy"] is not None else -1))
        rep["choice"] = {"chosen_on_selection": best, "selection": sel[best]["pairwise"],
                         "verification": rep["halves"]["verification"]["predictors"][best]["pairwise"]}
    warn = []
    if len(primary) < min_n:
        warn.append(f"solo {len(primary)} giudizi ciechi utilizzabili (< {min_n}): nessuna conclusione, IC larghi")
    if not blind:
        warn.append("nessun giudizio alla cieca: l'analisi principale è vuota (i non ciechi sono solo diagnostica)")
    rep["warnings"] = warn
    rep["note"] = "Un'associazione con i voti non prova che il giudice misuri il gusto; il veto tecnico resta separato."
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--judgments", required=True, help="cartella dei giudizi (dj-review.1)")
    ap.add_argument("--scores", help="JSON {clip_id: {v3_score, v3_n_flags, summary{...}}} già calcolato")
    ap.add_argument("--manifest", help="review_manifest.json (musicmix.dj-review.1) per calcolare i punteggi mancanti")
    ap.add_argument("--cache", help="cache dei punteggi calcolati (default accanto a --out)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-nonblind", action="store_true")
    ap.add_argument("--max-plays", type=int, default=3)
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--choose", action="store_true", help="scegli il predittore sulla metà di selezione, verifica sull'altra")
    a = ap.parse_args(argv)
    docs = read_judgments(Path(a.judgments))
    scores: dict = json.loads(Path(a.scores).read_text()) if a.scores else {}
    if a.manifest:
        cache = Path(a.cache) if a.cache else Path(a.out).with_suffix(".scores.json")
        scores = {**score_clips(Path(a.manifest), {d["clip_id"] for d in docs}, cache), **scores}
    rep = calibrate(docs, scores, include_nonblind=a.include_nonblind, max_plays=a.max_plays, min_n=a.min_n, choose=a.choose)
    C.write_json(a.out, rep)
    print(json.dumps({"counts": rep["counts"], "warnings": rep["warnings"]}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
