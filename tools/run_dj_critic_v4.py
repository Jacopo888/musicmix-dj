#!/usr/bin/env python3
"""Giudice automatico v4 (canali dei libri, ``musicmix.dj_critic_books``): run ``runs/dj_critic_v4``.

Fasi (riprendibili; ``--out`` = runs/dj_critic_v4):

  musicgen   [venv-np1]  NLL **ed entropia** per frame di MusicGen-small sugli item di runs/dj_critic_v1
                         (stessa finestra di 30 s della v1); parte solo se ``free`` mostra >= 6 GB disponibili
  channels   [venv]      canali v4 per ogni item (Beat This!, DSP; NLL dalla cache v1, entropia da quella v4)
  validate   [venv]      v3 contro v4 sugli stessi giudizi (Spearman, AUC voti >= 7 vs <= 4, accuratezza a coppie,
                         bootstrap sugli item), separando i giudizi di selezione da quelli di verifica
                         (``PREREGISTRATION.json``, scritto prima di calcolare i canali sugli item giudicati)
  stability  [venv]      D3: stabilità dei punteggi tra finestre di 1, 3 e 5 s sulle stesse clip

    python tools/run_dj_critic_v4.py channels validate stability
    $MUSICMIX_PY_NP1 tools/run_dj_critic_v4.py musicgen
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from musicmix import dj_critic as C  # noqa: E402

V1 = ROOT / "runs/dj_critic_v1"
OUT = ROOT / "runs/dj_critic_v4"
SR = 22050
LABELLED = ("dj_gallery_v1", "v10_reconstructed", "v12_release", "v12_natural_reference", "v8_reconstructed")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_items() -> list[C.TransitionItem]:
    d = json.loads((V1 / "items.json").read_text(encoding="utf-8"))
    items = [C.TransitionItem.from_json(x) for x in d["items"]]
    return sorted(items, key=lambda it: (it.dataset not in LABELLED, it.item_id))


def cache_name(it: C.TransitionItem) -> str:
    return it.item_id.replace("/", "__") + ".npz"


def mem_available_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return float("nan")


def item_metadata(it: C.TransitionItem) -> Optional[dict]:
    """Metadati del render della clip (gallerie, set), se esistono; None per le ricostruzioni storiche."""
    run, stem = it.item_id.split("/", 1)
    cands = []
    if run.startswith("dj_gallery"):
        cands = [ROOT / "runs" / run / sub / f"{stem}.json" for sub in ("clips", "rejected")]
    elif run == "dj_set_v1b":
        cands = [ROOT / "runs/dj_set_v1b/transitions" / f"{stem}.json"]
    for p in cands:
        if p.is_file():
            d = json.loads(p.read_text(encoding="utf-8"))
            return d.get("render", d)
    return None


# ================================================================ musicgen
def stage_musicgen(out: Path, threads: int, limit: Optional[int], items_json: Optional[str] = None,
                   cache_dir: Optional[str] = None):
    from musicmix import dj_critic_books as B

    avail = mem_available_gb()
    if not avail >= 6.0:
        log(f"musicgen: solo {avail:.1f} GB disponibili (< 6): rinviato")
        (out / "musicgen_skipped.json").write_text(json.dumps({"mem_available_gb": avail, "rule": ">= 6 GB"}))
        return
    cache = Path(cache_dir) if cache_dir else out / "cache/musicgen_ent"
    cache.mkdir(parents=True, exist_ok=True)
    mg = B.MusicGenEntropy(threads=threads)
    if items_json:  # API (dj_critic.score_transitions(..., version="v4", content_surprise=True))
        items = [C.TransitionItem.from_json(x) for x in json.loads(Path(items_json).read_text())["items"]]
    else:
        items = load_items()[: limit or None]
    t0 = time.time()
    for n, it in enumerate(items):
        p = cache / cache_name(it)
        if p.exists():
            continue
        dur = min(it.clip_dur_s, C.SURPRISAL_MAX_S)
        span = it.end_s - it.start_s
        side = max(4.0, (dur - span) / 2)
        s0 = max(0.0, min(it.start_s - side, it.clip_dur_s - dur))
        nll, ent = mg.frame_nll_entropy(it.read_clip(C.SR_MUSICGEN, s0, dur))
        np.savez(p, nll=nll.astype(np.float32), ent=ent.astype(np.float32), offset_s=s0, dtype=mg.dtype)
        log(f"musicgen {n + 1}/{len(items)} {it.item_id} ({dur:.1f} s, {(time.time() - t0) / (n + 1):.0f} s/item, "
            f"{mem_available_gb():.1f} GB liberi)")


# ================================================================ channels
def stage_channels(out: Path, limit: Optional[int]):
    from musicmix import dj_critic_books as B
    from musicmix import dj_qa

    sc3 = json.loads((V1 / "scores_v3.json").read_text())["items"]
    path = out / "scores_v4.json"
    res = json.loads(path.read_text())["items"] if path.exists() else {}
    tracker = dj_qa._get_tracker("auto")
    items = load_items()[: limit or None]
    for n, it in enumerate(items):
        if it.item_id in res:
            continue
        t1 = time.time()
        x = it.read_clip(SR)
        meta = item_metadata(it)
        nll = ent = None
        off = 0.0
        pe = out / "cache/musicgen_ent" / cache_name(it)
        p1 = V1 / "cache/musicgen" / cache_name(it)
        if pe.exists():
            z = np.load(pe)
            nll, ent, off = z["nll"], z["ent"], float(z["offset_s"])
        elif p1.exists():
            z = np.load(p1)
            nll, off = z["nll"], float(z["offset_s"])
        craft = (sc3.get(it.item_id) or {}).get("craft")
        v4 = B.score_v4(x, SR, meta, start_s=it.start_s, switch_s=it.switch_s, end_s=it.end_s, craft=craft, qa=it.qa,
                        decks=None, nll=nll, entropy=ent, nll_offset_s=off, tracker=tracker)
        res[it.item_id] = {"item": {"dataset": it.dataset, "kind": it.kind, "group": it.group, "label": it.label},
                           "summary": B.summary_v4(v4), "v4": v4, "inputs": {
                               "metadata": meta is not None, "craft_v3": craft is not None, "qa": it.qa is not None,
                               "musicgen_nll": "v4 cache" if pe.exists() else ("v1 cache" if p1.exists() else None),
                               "musicgen_entropy": pe.exists(), "decks": False},
                           "seconds": time.time() - t1}
        C.write_json(path, {"schema_version": B.DJ_CRITIC_BOOKS_VERSION, "items": res})
        s = res[it.item_id]["summary"]
        log(f"channels {n + 1}/{len(items)} {it.item_id}: veto {s['v4_n_veto']:.0f}, issues {s['v4_n_issues']:.0f}, "
            f"metric {s['metric_surprise']}, content {s['content_surprise_gated']} ({time.time() - t1:.1f} s)")


def stage_metric41(out: Path):
    """D1 v4.1 (correzione di misurabilità, decisa per il tasso di rette fallite, non sui voti): Beat This! in una
    sola passata sul tratto continuo; ricalcola metric_surprise e il gating della sorpresa di contenuto."""
    from musicmix import dj_critic_books as B
    from musicmix import dj_qa

    sc4 = json.loads((out / "scores_v4.json").read_text())["items"]
    tracker = dj_qa._get_tracker("auto")
    res = {}
    for n, it in enumerate(load_items()):
        if it.item_id not in sc4:
            continue
        x = it.read_clip(SR)
        meta = item_metadata(it)
        m = B.metric_surprise(x, SR, it.start_s, it.switch_s, it.end_s, tracker=tracker, metadata=meta, joint_tracking=True)
        cs = sc4[it.item_id]["v4"]["musical"]["content_surprise"]
        ok = m.get("metric_ok")
        res[it.item_id] = {"metric": m, "metric_surprise": m.get("metric_surprise"), "grid_dev_ms": m.get("grid_dev_ms"),
                           "content_surprise_gated": cs.get("peak_z") if ok is True else None}
        log(f"metric41 {n + 1}: {it.item_id} {m['status']} {m.get('grid_dev_ms')}")
    C.write_json(out / "metric_v41.json", {"items": res})


# ================================================================ validate
def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0) + 0.5 * (d == 0)).mean())


def auc_with_ci(pos, neg, n_boot=2000, seed=0) -> dict:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    a = _auc(pos, neg)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        bp = pos[rng.integers(0, len(pos), len(pos))] if len(pos) else pos
        bn = neg[rng.integers(0, len(neg), len(neg))] if len(neg) else neg
        v = _auc(bp, bn)
        if np.isfinite(v):
            boots.append(v)
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None
    return {"auc": a, "n_pos": int(np.isfinite(pos).sum()), "n_neg": int(np.isfinite(neg).sum()), "bootstrap95": ci}


def spearman_with_ci(pred, truth, n_boot=2000, seed=0) -> dict:
    from scipy.stats import spearmanr

    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    ok = np.isfinite(pred) & np.isfinite(truth)
    pred, truth = pred[ok], truth[ok]
    if len(pred) < 4 or np.all(pred == pred[0]):
        return {"n": int(len(pred)), "rho": None, "bootstrap95": None}
    rho = float(spearmanr(pred, truth).statistic)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(pred), len(pred))
        if len(set(pred[i].tolist())) > 1 and len(set(truth[i].tolist())) > 1:
            boots.append(float(spearmanr(pred[i], truth[i]).statistic))
    perm = C.spearman_with_permutation(pred, truth, n_perm=5000)
    return {"n": int(len(pred)), "rho": rho, "bootstrap95": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]
            if boots else None, "p_perm_two_sided": perm.get("p_perm_two_sided")}


def predictors(sc3: dict, sc4: dict) -> dict:
    """Bontà (più alto = meglio) di v3 e v4, come dichiarate in PREREGISTRATION.json."""
    def g3(f):
        return lambda k: (sc3.get(k) or {}).get(f)

    def neg4(f):
        def fn(k):
            v = ((sc4.get(k) or {}).get("summary") or {}).get(f)
            return None if v is None else -float(v)
        return fn

    return {
        "v3_ensemble": g3("score"),
        "v3_fast_dsp": g3("score_fast"),
        "v3_n_flags_neg": lambda k: -len((sc3.get(k) or {}).get("flags", [])) if k in sc3 else None,
        "v4_technical_veto_neg": neg4("v4_n_veto"),
        "v4_technical_issues_neg": neg4("v4_n_issues"),
        "v4_metric_surprise_neg": neg4("metric_surprise"),
        "v4_metric_grid_dev_neg": neg4("metric_grid_dev_ms"),
        "v4_content_surprise_gated_neg": neg4("content_surprise_gated"),
        "v4_content_surprise_raw_neg": neg4("content_surprise_raw"),
        "v4_confident_wrong_1s_neg": neg4("confident_wrong_1s"),
        "v4_entropy_before_3s": lambda k: ((sc4.get(k) or {}).get("summary") or {}).get("ent_before_3s"),
        "v4_win1s_timbre_neg": neg4("win1s_timbre_dist"),
        "v4_win3s_loud_neg": neg4("win3s_loud_jump_db"),
    }


def label_sets(sc3: dict) -> dict:
    s1 = [(k, v["item"]["label"]["score"]) for k, v in sc3.items() if v["item"]["dataset"] == "dj_gallery_v1" and v["item"]["label"]]
    v10 = [(k, v["item"]["label"]["ordinal"]) for k, v in sc3.items() if v["item"]["dataset"] == "v10_reconstructed"]
    v12 = [k for k, v in sc3.items() if v["item"]["dataset"] == "v12_release" and v["item"]["label"]]
    return {"S1": s1, "V10": v10, "V12": v12}


def stage_validate(out: Path):
    sc3 = json.loads((V1 / "scores_v3.json").read_text())["items"]
    sc4 = json.loads((out / "scores_v4.json").read_text())["items"]
    pre = json.loads((out / "PREREGISTRATION.json").read_text())
    m41 = json.loads((out / "metric_v41.json").read_text())["items"] if (out / "metric_v41.json").exists() else {}
    L = label_sets(sc3)
    preds = predictors(sc3, sc4)
    if m41:
        def n41(f):
            return lambda k: None if (m41.get(k) or {}).get(f) is None else -float(m41[k][f])
        preds["v41_metric_surprise_neg"] = n41("metric_surprise")
        preds["v41_metric_grid_dev_neg"] = n41("grid_dev_ms")
        preds["v41_content_surprise_gated_neg"] = n41("content_surprise_gated")

    def val(k, key):
        try:
            v = preds[key](k)
        except (KeyError, TypeError):
            return float("nan")
        return float("nan") if v is None else float(v)

    s1_ids = [k for k, _ in L["S1"]]
    s1_y = np.array([y for _, y in L["S1"]], float)
    v10_ids = [k for k, _ in L["V10"]]
    v10_y = np.array([y for _, y in L["V10"]], float)
    pos_s1 = [k for k, y in L["S1"] if y >= 7]
    neg_s1 = [k for k, y in L["S1"] if y <= 4]
    neg_v10 = [k for k, y in L["V10"] if y <= 1.5]  # [H] "non va bene", "pessima"
    res: dict[str, Any] = {"preregistration_sha256": C.sha256_file(out / "PREREGISTRATION.json"), "n": {
        "S1_votes": len(s1_ids), "S1_ge7": len(pos_s1), "S1_le4": len(neg_s1), "V10_verbal": len(v10_ids), "V12_main": len(L["V12"]),
        "V10_le1.5": len(neg_v10)}, "critics": {}}
    for key in preds:
        r: dict[str, Any] = {}
        # --- selezione (S1, già usata per la v3b e per formulare D1)
        p1 = np.array([val(k, key) for k in s1_ids])
        r["selection_S1_spearman"] = spearman_with_ci(p1, s1_y)
        r["selection_S1_auc_ge7_vs_le4"] = auc_with_ci([val(k, key) for k in pos_s1], [val(k, key) for k in neg_s1])
        pa = C.pairwise_accuracy(p1, s1_y)
        r["selection_S1_pairwise"] = {"n_pairs": pa["n_pairs"], "accuracy": pa["accuracy"],
                                      "bootstrap95_items": C.bootstrap_pairwise_ci(p1, s1_y) if pa["n_pairs"] else None}
        # --- verifica (giudizi non usati per scegliere le regole v4)
        p2 = np.array([val(k, key) for k in v10_ids])
        r["verify_V10_spearman"] = spearman_with_ci(p2, v10_y)
        pa = C.pairwise_accuracy(p2, v10_y)
        r["verify_V10_pairwise"] = {"n_pairs": pa["n_pairs"], "accuracy": pa["accuracy"],
                                    "bootstrap95_items": C.bootstrap_pairwise_ci(p2, v10_y) if pa["n_pairs"] else None}
        r["verify_S1ge7_vs_V12main_auc"] = auc_with_ci([val(k, key) for k in pos_s1], [val(k, key) for k in L["V12"]])
        # sensibilità [H]: negativi = S1 <= 4 + V12 main + V10 codificati <= 1,5 (sessioni e scale diverse)
        r["sensitivity_ge7_vs_all_neg_auc"] = auc_with_ci([val(k, key) for k in pos_s1],
                                                          [val(k, key) for k in neg_s1 + L["V12"] + neg_v10])
        res["critics"][key] = r
    # V8: unica preferenza netta
    v8 = {}
    for k, v in sc3.items():
        if v["item"]["dataset"] == "v8_reconstructed":
            lab = v["item"]["label"]
            v8.setdefault(lab["pair_id"], {})[lab["candidate_label"]] = (k, lab["preference"])
    res["v8_pool_04"] = {key: {"A": val(v8["pool_04"]["A"][0], key), "B": val(v8["pool_04"]["B"][0], key), "preference": "A"}
                         for key in preds} if "pool_04" in v8 else None
    # D1: la sorpresa metrica deve essere ~0 sulle clip votate >= 8 (proposta dei libri)
    res["D1_check_votes_ge8"] = {k: {"vote": y, "metric_surprise": ((sc4.get(k) or {}).get("summary") or {}).get("metric_surprise"),
                                     "grid_dev_ms": ((sc4.get(k) or {}).get("summary") or {}).get("metric_grid_dev_ms"),
                                     "hyper_weight": ((sc4.get(k) or {}).get("summary") or {}).get("hyper_weight"),
                                     "v41_metric_surprise": (m41.get(k) or {}).get("metric_surprise"),
                                     "v41_grid_dev_ms": (m41.get(k) or {}).get("grid_dev_ms")}
                                 for k, y in L["S1"] if y >= 8}
    res["notes"] = pre["honesty_notes"]
    C.write_json(out / "validation_v4.json", res)
    log("validate: fatto")


# =============================================================== stability
def stage_stability(out: Path):
    from musicmix import dj_critic_books as B

    sc4 = json.loads((out / "scores_v4.json").read_text())["items"]
    prof = {k: v["v4"]["musical"]["windows"] for k, v in sc4.items() if (v["v4"]["musical"].get("windows") or {}).get("windows")}
    st = B.window_stability(prof)
    C.write_json(out / "window_stability.json", {"n_clips": len(prof), "stability": st,
                                                  "rule": "Spearman tra finestre sulle stesse clip (D3; Huron pp. 207-208, 228)"})
    log("stability: fatto")


def _f(v, nd=2):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def _ci(c):
    return "–" if not c else f"[{_f(c[0])}, {_f(c[1])}]"


def stage_report(out: Path):
    """REPORT.md con le tabelle di validazione, stabilità, QA su A grezzo e canali con i deck."""
    val = json.loads((out / "validation_v4.json").read_text())
    L = ["# Giudice automatico v4 (canali dei libri) — run dj_critic_v4", "",
         "Codice: `src/musicmix/dj_critic_books.py`, `tools/run_dj_critic_v4.py`, `tools/measure_qa_raw_a.py`, "
         "`tools/check_v4_decks_real.py`, `tools/calibrate_dj_critic.py`. Regole dichiarate prima del calcolo: "
         "`PREREGISTRATION.json` (con gli emendamenti). Nota: `conoscenza_mix/_meta/AUTO_CRITICO_V4_2026-10-21.md`.", "",
         f"Numerosità: {json.dumps(val['n'], ensure_ascii=False)}. Un solo ascoltatore; galleria v1 non alla cieca; "
         "tutti i giudizi erano già noti durante lo sviluppo v1-v3.", "",
         "## Validazione v3 contro v4 (più alto = migliore; IC 95% bootstrap sugli item)", "",
         "| predittore | S1 Spearman (6) | S1 AUC ≥7 vs ≤4 (5 vs 1) | V10 Spearman (8) [verifica] | S1≥7 vs V12 main AUC (5 vs 2) [verifica] | sensibilità AUC ≥7 vs tutti i negativi |",
         "|---|---|---|---|---|---|"]
    for k, r in val["critics"].items():
        s1, a1 = r["selection_S1_spearman"], r["selection_S1_auc_ge7_vs_le4"]
        v10, a2, a3 = r["verify_V10_spearman"], r["verify_S1ge7_vs_V12main_auc"], r["sensitivity_ge7_vs_all_neg_auc"]
        L.append(f"| {k} | {_f(s1['rho'])} {_ci(s1['bootstrap95'])} (n {s1['n']}) | {_f(a1['auc'])} {_ci(a1['bootstrap95'])} | "
                 f"{_f(v10['rho'])} {_ci(v10['bootstrap95'])} (n {v10['n']}) | {_f(a2['auc'])} {_ci(a2['bootstrap95'])} | "
                 f"{_f(a3['auc'])} {_ci(a3['bootstrap95'])} ({a3['n_pos']} vs {a3['n_neg']}) |")
    L += ["", "D1 (sorpresa metrica sulle clip votate ≥ 8): " + "; ".join(
        f"`{k.split('/')[-1]}` voto {v['vote']:g}: metric {_f(v['metric_surprise'])}, scarto {_f(v['grid_dev_ms'], 1)} ms, "
        f"peso ipermetrico {_f(v['hyper_weight'])}; v4.1 metric {_f(v.get('v41_metric_surprise'))}, scarto {_f(v.get('v41_grid_dev_ms'), 1)} ms"
        for k, v in val["D1_check_votes_ge8"].items()), ""]
    if val.get("v8_pool_04"):
        L.append("V8 pool_04 (preferenza A): " + "; ".join(f"{k} A {_f(v['A'])} / B {_f(v['B'])}" for k, v in val["v8_pool_04"].items()))
        L.append("")
    st = out / "window_stability.json"
    if st.exists():
        sd = json.loads(st.read_text())
        L += [f"## D3: stabilità tra finestre (Spearman sulle stesse {sd['n_clips']} clip)", "", "| misura | 1 s ~ 3 s | 3 s ~ 5 s | 1 s ~ 5 s |", "|---|---|---|---|"]
        for m, rr in sd["stability"].items():
            L.append(f"| {m} | " + " | ".join(f"{_f((rr.get(k) or {}).get('spearman'))} (n {(rr.get(k) or {}).get('n')})"
                                              for k in ("1s~3s", "3s~5s", "1s~5s")) + " |")
        L.append("")
    qa = out / "qa_raw_a.json"
    if qa.exists():
        q = json.loads(qa.read_text())["summary"]
        L += ["## QA: ritmo su A grezzo (`QAConfig(rhythm_on_raw_a=True)`)", "",
              f"Blend rifatti: {q['n_blends']}; verdetti che cambierebbero: **{q['verdict_changes']}**; stato del ritmo che cambia: "
              f"{q['rhythm_status_changes']}; PCM identico al salvato: {q['pcm_identical']}; verdetto di default rifatto = salvato: "
              f"{q['reproduced_default_equals_stored']}/{q['with_stored_verdict']}.", ""]
        for c in q["changes"]:
            L.append(f"- {c['run']} `{c['id']}`: verdetto {c['default']} → {c['raw_a']}, ritmo {c['rhythm'][0]} → {c['rhythm'][1]}")
        L.append("")
    dr = out / "decks_real.json"
    if dr.exists():
        q = json.loads(dr.read_text())
        L += ["## Canali con i deck su audio reale (galleria v2, blend)", ""]
        for r in q["rows"]:
            bb, px = r["backbeat_used"], r["backbeat_proxy_decks"]
            L.append(f"- `{r['id']}`: backbeat {bb.get('status')}, mediana con segno {_f(bb.get('backbeat_median_signed_ms'), 1)} ms, "
                     f"|scarto| {_f(bb.get('backbeat_median_abs_ms'), 1)} ms ({bb.get('source')}, {bb.get('n_backbeats')} rullanti); "
                     f"proxy deck {_f(px.get('backbeat_median_signed_ms'), 1)} ms con segno; allineamento del render "
                     f"{_f((r.get('render_downbeat_alignment') or {}).get('median_abs_error_ms'))} ms; "
                     f"Beat This! solo-clip {_f((r.get('metric_surprise') or {}).get('grid_dev_ms'), 0)} ms; "
                     f"beat girato {r['downbeat_after_kick'].get('status')}; tessitura both_busy {_f((r['texture'] or {}).get('both_busy_share'))}; "
                     f"veto {[f['flag'] for f in r['technical']['veto_flags']]}, warn {[w['check'] for w in r['technical']['warnings']]}")
        L.append("")
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("report: fatto")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stages", nargs="+", choices=["musicgen", "channels", "metric41", "validate", "stability", "report"])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--items-json", help="musicgen: item (schema dj_critic) da valutare al posto di runs/dj_critic_v1")
    ap.add_argument("--cache-dir", help="musicgen: cartella dei .npz (default <out>/cache/musicgen_ent)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in a.stages:
        t0 = time.time()
        if s == "musicgen":
            stage_musicgen(out, a.threads, a.limit, a.items_json, a.cache_dir)
        elif s == "channels":
            stage_channels(out, a.limit)
        elif s == "metric41":
            stage_metric41(out)
        elif s == "validate":
            stage_validate(out)
        elif s == "stability":
            stage_stability(out)
        elif s == "report":
            stage_report(out)
        log(f"{s}: {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
