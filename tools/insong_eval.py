"""insong_v1: addestramento e valutazione dei modelli dei punti e delle giunte, controllo esterno, cache dei cue.

Chiamato da ``tools/build_insong_dataset.py <fase>`` con fase in ``train``, ``pairs``, ``external``, ``cues``, ``report``.
Tutto su feature già in cache (``runs/insong_v1/cache``): nessun modello preaddestrato caricato qui.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

from build_insong_dataset import OUT, ROOT, dump, load, log, tracks_by_slug  # noqa: F401
from musicmix import self_supervision as S

VARIANTS = tuple(S.FEATURE_BLOCKS)
SAVE_VARIANTS = ("full", "audio")


# =========================================================================== feature dei punti
def point_matrix(points: list[dict], tracks: dict[str, S.TrackData]) -> dict[str, np.ndarray]:
    caches = {s: S.TrackCache.load(OUT, s) for s in tracks}
    rows = [S.point_features(tracks[r["slug"]], caches[r["slug"]], r["t"]) for r in points]
    return S.stack(rows, S.ALL_BLOCKS)


def subset(X: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[idx] for k, v in X.items()}


def metrics_points(points: list[dict], score: np.ndarray) -> dict:
    sc = {r["id"]: float(s) for r, s in zip(points, score)}
    pos = [(r["slug"], sc[r["id"]]) for r in points if r["label"] == 1]
    out = {"all": {"auc": S.auc([v for _, v in pos], [sc[r["id"]] for r in points if r["label"] == 0])}}
    for typ in S.NEG_TYPES:
        neg = [(r["slug"], sc[r["id"]]) for r in points if r["label"] == 0 and r["kind"] == typ]
        lo, hi = S.bootstrap_auc_by_group(pos, neg, n_boot=300)
        out[typ] = {"n_neg": len(neg), "auc": S.auc([v for _, v in pos], [v for _, v in neg]), "ci95": [lo, hi],
                    "paired_win": S.paired_win_rate(sc, points, typ)}
    negall = [sc[r["id"]] for r in points if r["label"] == 0]
    for kind in S.POS_KINDS:
        pk = [sc[r["id"]] for r in points if r["label"] == 1 and r["kind"] == kind]
        out[f"pos_{kind}"] = {"n_pos": len(pk), "auc_vs_all_neg": S.auc(pk, negall)}
    return out


def loao(points: list[dict], X: dict[str, np.ndarray], blocks, *, c: float = 0.3, pca: int = 24, save_prefix: Optional[str] = None,
         track_groups: Optional[dict] = None):
    y = np.asarray([r["label"] for r in points])
    groups = np.asarray([r["group"] for r in points])
    slugs = np.asarray([r["slug"] for r in points])
    oof = np.full(len(points), np.nan)
    for g in S.group_folds(groups):
        te, tr = groups == g, groups != g
        S.check_disjoint(set(slugs[tr]), set(slugs[te]), track_groups)
        m = S.BlockLogistic(blocks, pca_dims=pca, c=c).fit(subset(X, np.where(tr)[0]), y[tr])
        oof[te] = m.predict_proba(subset(X, np.where(te)[0]))
        if save_prefix:
            m.save(OUT / "model" / f"{save_prefix}__wo_{S._safe(g)}.npz", meta={"held_out_group": g, "train_slugs": sorted(set(slugs[tr]))})
    return oof


def stage_train() -> None:
    pts = load(OUT / "points.json")["points"]
    tracks = tracks_by_slug()
    track_groups = {s: t.group for s, t in tracks.items()}
    t0 = time.time()
    X = point_matrix(pts, tracks)
    log(f"feature dei punti: {len(pts)} in {time.time() - t0:.0f} s; dimensioni {{k: v.shape[1] for k, v in X.items()}}")
    y = np.asarray([r["label"] for r in pts])
    res = {"schema_version": S.SCHEMA + ".points_eval", "n_points": len(pts), "n_pos": int(y.sum()),
           "split": "leave-one-artist-out (gruppi: " + ", ".join(S.group_folds([r["group"] for r in pts])) + ")",
           "classifier": "logistica L2 bilanciata, C=0.3, standardizzazione e PCA 24 per blocco > 24 colonne (stimate sul training)",
           "variants": {}}
    oofs = {}
    for v in VARIANTS:
        blocks = S.FEATURE_BLOCKS[v]
        oof = loao(pts, X, blocks, save_prefix=f"point_{v}" if v in SAVE_VARIANTS else None, track_groups=track_groups)
        oofs[v] = oof
        res["variants"][v] = {"blocks": list(blocks), "metrics": metrics_points(pts, oof)}
        mm = res["variants"][v]["metrics"]
        log(f"{v:6s} AUC tutti {mm['all']['auc']:.3f} | " + " ".join(f"{t}:{mm[t]['auc']:.2f}" for t in S.NEG_TYPES if mm[t]["auc"] is not None))
        if v in SAVE_VARIANTS:
            S.BlockLogistic(blocks).fit(X, y).save(OUT / "model" / f"point_{v}.npz", meta={"train_slugs": sorted(tracks)})
    dump(OUT / "model" / "folds.json", {"track_groups": track_groups, "groups": S.group_folds(list(track_groups.values()))})
    np.savez(OUT / "oof_points.npz", ids=np.asarray([r["id"] for r in pts]), **{v: oofs[v] for v in oofs})
    dump(OUT / "results_points.json", res)


# =========================================================================== giunte
def pair_matrix(pairs: list[dict], tracks) -> dict[str, np.ndarray]:
    caches = {s: S.TrackCache.load(OUT, s) for s in tracks}
    rows = [S.pair_features(tracks[p["x"]], caches[p["x"]], p["t1"], tracks[p["y"]], caches[p["y"]], p["t2"]) for p in pairs]
    return S.stack(rows, S.ALL_BLOCKS)


def pair_oof(pairs, X, blocks, train_mask=None, test_mask=None):
    """OOF per gruppi: una giunta è di test nel fold di ciascun artista che tocca; addestramento senza quell'artista."""
    y = np.asarray([p["label"] for p in pairs])
    gs = [set(p["groups"]) for p in pairs]
    allg = sorted(set().union(*gs))
    acc, cnt = np.zeros(len(pairs)), np.zeros(len(pairs))
    train_mask = np.ones(len(pairs), bool) if train_mask is None else train_mask
    test_mask = np.ones(len(pairs), bool) if test_mask is None else test_mask
    for g in allg:
        te = np.asarray([g in s for s in gs]) & test_mask
        tr = np.asarray([g not in s for s in gs]) & train_mask
        if te.sum() == 0 or len(set(y[tr])) < 2:
            continue
        m = S.BlockLogistic(blocks).fit(subset(X, np.where(tr)[0]), y[tr])
        acc[te] += m.predict_proba(subset(X, np.where(te)[0]))
        cnt[te] += 1
    out = np.full(len(pairs), np.nan)
    out[cnt > 0] = acc[cnt > 0] / cnt[cnt > 0]
    return out


def pair_metrics(pairs, sc) -> dict:
    out = {}
    for scope in ("within", "cross"):
        pos = [s for p, s in zip(pairs, sc) if p["scope"] == scope and p["label"] == 1 and np.isfinite(s)]
        d = {"n_pos": len(pos)}
        neg_all = [s for p, s in zip(pairs, sc) if p["scope"] == scope and p["label"] == 0 and np.isfinite(s)]
        d["all"] = S.auc(pos, neg_all)
        for side in ("entry", "exit"):
            for typ in S.NEG_TYPES:
                neg = [s for p, s in zip(pairs, sc) if p["scope"] == scope and p["label"] == 0 and p["side"] == side
                       and p["kind"] == typ and np.isfinite(s)]
                d[f"{side}:{typ}"] = S.auc(pos, neg)
        out[scope] = d
    wp = [s for p, s in zip(pairs, sc) if p["scope"] == "within" and p["label"] == 1 and np.isfinite(s)]
    cp = [s for p, s in zip(pairs, sc) if p["scope"] == "cross" and p["label"] == 1 and np.isfinite(s)]
    wn = [s for p, s in zip(pairs, sc) if p["scope"] == "within" and p["label"] == 0 and np.isfinite(s)]
    out["same_song_bias"] = {
        "auc_within_pos_vs_cross_pos": S.auc(wp, cp),
        "auc_cross_pos_vs_within_neg": S.auc(cp, wn),
        "mean_within_pos": float(np.mean(wp)) if wp else None, "mean_cross_pos": float(np.mean(cp)) if cp else None,
        "note": "0,5 = il punteggio non premia lo stesso brano a parità di punto naturale; >0,5 = premia 'stesso brano'"}
    return out


def stage_pairs() -> None:
    pairs = load(OUT / "pairs.json")["pairs"]
    tracks = tracks_by_slug()
    t0 = time.time()
    X = pair_matrix(pairs, tracks)
    log(f"feature delle giunte: {len(pairs)} in {time.time() - t0:.0f} s")
    within = np.asarray([p["scope"] == "within" for p in pairs])
    res = {"schema_version": S.SCHEMA + ".pairs_eval", "n_pairs": len(pairs), "runs": {}}
    specs = {
        "full_all": (S.FEATURE_BLOCKS["full"], None),
        "audio_all": (S.FEATURE_BLOCKS["audio"], None),
        "audio_novox_all": (S.FEATURE_BLOCKS["audio_novox"], None),
        "full_within_only": (S.FEATURE_BLOCKS["full"], within),
        "mert_compact_only": (("mert_c",), None),
    }
    for name, (blocks, tm) in specs.items():
        sc = pair_oof(pairs, X, blocks, train_mask=tm)
        res["runs"][name] = {"blocks": list(blocks), "train": "solo giunte nello stesso brano" if tm is not None else "tutte",
                             "metrics": pair_metrics(pairs, sc)}
        m = res["runs"][name]["metrics"]
        log(f"{name:18s} within {m['within']['all']:.3f} cross {m['cross']['all']:.3f} | bias W>C {m['same_song_bias']['auc_within_pos_vs_cross_pos']:.3f}")
    dump(OUT / "results_pairs.json", res)


# =========================================================================== COCOLA
def stage_cocola_eval() -> dict:
    doc = load(OUT / "cocola_stems.json")
    rows = doc["rows"]
    out = {}
    for scope in ("within", "cross"):
        d = {}
        for mode in ("both", "harmonic", "percussive"):
            key = f"s_{mode}"
            pos = [r[key] for r in rows if r["scope"] == scope and r["shift"] == "aligned"]
            dm = {"n_pos": len(pos)}
            for sh in ("offbeat", "beat", "halfbar", "bar", "two_bars"):
                neg = [r[key] for r in rows if r["scope"] == scope and r["shift"] == sh]
                # vittorie appaiate: stessa ancora (x, t_x) e stessa coppia
                anc = {(r["x"], r["y"], round(r["t_x"], 3)): r[key] for r in rows if r["scope"] == scope and r["shift"] == "aligned"}
                wins = [1.0 if anc[(r["x"], r["y"], round(r["t_x"], 3))] > r[key] else 0.0 for r in rows
                        if r["scope"] == scope and r["shift"] == sh and (r["x"], r["y"], round(r["t_x"], 3)) in anc]
                dm[sh] = {"auc": S.auc(pos, neg), "paired_win": float(np.mean(wins)) if wins else None, "n": len(neg)}
            d[mode] = dm
        out[scope] = d
    # armonia: distanza Camelot tra brani (allineati) contro punteggio armonico
    cam = []
    for r in rows:
        if r["scope"] == "cross" and r["shift"] == "aligned":
            dist = camelot_distance(r.get("camelot_x"), r.get("camelot_y"))
            if dist is not None:
                cam.append((dist, r["s_harmonic"], r["s_both"]))
    if cam:
        a = np.asarray(cam, float)
        out["camelot"] = {"n": len(cam), "spearman_harmonic_vs_distance": S.spearman(a[:, 0], a[:, 1]),
                          "spearman_both_vs_distance": S.spearman(a[:, 0], a[:, 2])}
    # stesso brano contro brani diversi a parità di allineamento
    wpos = [r["s_both"] for r in rows if r["scope"] == "within" and r["shift"] == "aligned"]
    cpos = [r["s_both"] for r in rows if r["scope"] == "cross" and r["shift"] == "aligned"]
    out["same_song"] = {"auc_within_aligned_vs_cross_aligned": S.auc(wpos, cpos)}
    dump(OUT / "results_cocola.json", out)
    return out


def camelot_distance(a: Optional[str], b: Optional[str]) -> Optional[int]:
    try:
        na, la, nb, lb = int(a[:-1]), a[-1], int(b[:-1]), b[-1]
    except (TypeError, ValueError, IndexError):
        return None
    d = min((na - nb) % 12, (nb - na) % 12)
    return d + (0 if la == lb else 1)


# =========================================================================== controllo esterno
B_WRONG = re.compile(r"(secondo sbagl|parola tagliata in b|b tagliat|punto sbagliato|momento sbagliato|b nel secondo|b sbagliat)", re.I)


def point_scorer(variant: str, group: Optional[str]) -> S.BlockLogistic:
    p = OUT / "model" / (f"point_{variant}__wo_{S._safe(group)}.npz" if group else f"point_{variant}.npz")
    return S.BlockLogistic.load(p)


def stage_external() -> None:
    man = load(ROOT / "runs/dj_set_v3_bridge/gallery/review_manifest.json")
    jdir = ROOT / "giudizi_cloud/dj_set_v3_bridge/judgments"
    judg = {}
    for f in sorted(jdir.glob("*.json")):
        d = load(f)
        judg[d["clip_id"]] = d
    tracks = tracks_by_slug()
    caches = {}
    rows = []
    for g in man["groups"]:
        if "A_out_s" not in g:
            continue  # clip di verifica dei libri: brani del pool CC (fuori dal dataset)
        a, b = g["pair"]["a"]["slug"], g["pair"]["b"]["slug"]
        if a not in tracks or b not in tracks:
            continue
        for s in (a, b):
            caches.setdefault(s, S.TrackCache.load(OUT, s))
        ta, tb = tracks[a], tracks[b]
        rec = {"group": g["id"], "a": a, "b": b, "A_out_s": g["A_out_s"], "B_in_s": g["B_in_s"]}
        for v in SAVE_VARIANTS:
            ma, mb = point_scorer(v, ta.group), point_scorer(v, tb.group)
            rec[f"p_exit_{v}"] = float(ma.predict_proba(S.stack([S.point_features(ta, caches[a], g["A_out_s"])], ma.blocks))[0])
            rec[f"p_entry_{v}"] = float(mb.predict_proba(S.stack([S.point_features(tb, caches[b], g["B_in_s"])], mb.blocks))[0])
            # c'era un punto più naturale entro ±2 battute dall'ingresso di B?
            cands = [c["t"] for c in S.cue_candidates(tb) if abs(c["t"] - g["B_in_s"]) <= 8 * tb.period_s]
            if cands:
                ps = mb.predict_proba(S.stack([S.point_features(tb, caches[b], t) for t in cands], mb.blocks))
                k = int(np.argmax(ps))
                rec[f"entry_best_nearby_{v}"] = {"t": cands[k], "p": float(ps[k]), "delta_s": cands[k] - g["B_in_s"],
                                                 "rank_of_used": int((ps > rec[f"p_entry_{v}"]).sum()) + 1, "n": len(cands)}
        for it in g["items"]:
            j = judg.get(it["id"])
            if not j:
                continue
            note = (j.get("note") or "") + " " + " ".join(j.get("tags") or [])
            rows.append({**rec, "clip": it["id"], "kind": it["kind"], "score": j.get("score"), "note": j.get("note"),
                         "tags": j.get("tags"), "b_wrong_note": bool(B_WRONG.search(note)) or "punto_frase_sbagliato" in (j.get("tags") or []),
                         "critic_v3": (it.get("critic") or {}).get("score"),
                         "critic_v4_technical": (((it.get("critic") or {}).get("v4") or {}).get("technical") or {}).get("status")})
    scored = [r for r in rows if r["score"] is not None]
    res = {"schema_version": S.SCHEMA + ".external", "n_clips": len(rows), "n_scored": len(scored),
           "n_groups": len({r["group"] for r in rows}), "rows": rows, "tests": {}}
    for key in ("p_entry_full", "p_entry_audio", "p_exit_full", "p_exit_audio", "critic_v3"):
        xs = [r[key] for r in scored if r[key] is not None]
        ys = [r["score"] for r in scored if r[key] is not None]
        flagged = [r[key] for r in rows if r["b_wrong_note"] and r[key] is not None]
        clean = [r[key] for r in rows if not r["b_wrong_note"] and r[key] is not None]
        # per gruppo (stessi cue nelle due versioni): media dei voti
        gm = {}
        for r in scored:
            gm.setdefault(r["group"], {"x": r[key], "y": []})["y"].append(r["score"])
        gx = [v["x"] for v in gm.values() if v["x"] is not None]
        gy = [float(np.mean(v["y"])) for v in gm.values() if v["x"] is not None]
        res["tests"][key] = {"spearman_clip": S.spearman(xs, ys), "n_clip": len(xs), "spearman_group": S.spearman(gx, gy),
                             "n_group": len(gx), "auc_clean_vs_b_wrong_note": S.auc(clean, flagged),
                             "n_flagged": len(flagged), "n_clean": len(clean)}
    dump(OUT / "results_external.json", res)
    for k, v in res["tests"].items():
        log(f"esterno {k:14s} rho clip {fmt(v['spearman_clip'])} (n={v['n_clip']}) gruppo {fmt(v['spearman_group'])} (n={v['n_group']}) "
            f"AUC pulite>note 'B sbagliato' {fmt(v['auc_clean_vs_b_wrong_note'])} ({v['n_clean']}/{v['n_flagged']})")


def fmt(x) -> str:
    return "n/d" if x is None else f"{x:.2f}"


# =========================================================================== cache dei cue
def stage_cues(slugs: Optional[list[str]] = None) -> None:
    tracks = tracks_by_slug()
    folds = load(OUT / "model" / "folds.json")
    for slug in slugs or sorted(tracks):
        tr = tracks.get(slug) or S.load_track(slug)
        cache = S.TrackCache.load(OUT, slug)
        if cache.mert is None or cache.dsp is None:
            log(f"{slug}: cache MERT/DSP assenti, salto")
            continue
        seen = slug in folds["track_groups"] or tr.group in folds["groups"]
        group = tr.group if seen else None
        cands = S.cue_candidates(tr)
        F = [S.point_features(tr, cache, c["t"]) for c in cands]
        model_files = {}
        for v in SAVE_VARIANTS:
            m = point_scorer(v, group)
            model_files[v] = f"model/point_{v}__wo_{S._safe(group)}.npz" if group else f"model/point_{v}.npz"
            ps = m.predict_proba(S.stack(F, m.blocks))
            for c, p in zip(cands, ps):
                c[f"p_{v}"] = round(float(p), 4)
        # tensione -> risoluzione (dsp_form: nessun modello preaddestrato richiesto), stesso criterio di fold
        form = S.load_form(slug, len(tr.downbeats_s))
        cp = OUT / "cache/chroma" / f"{slug}.npz"
        chroma = np.load(cp)["x"] if cp.is_file() else None
        for role, key in (("exit", "tension_exit"), ("entry", "resolution_entry")):
            fn = f"model/tension_{role}_dsp_form__wo_{S._safe(group)}.npz" if group else f"model/tension_{role}_dsp_form.npz"
            model_files[key] = fn
            m = S.BlockLogistic.load(OUT / fn)
            TF = [S.tension_features(tr, cache, c["t"], role, form, chroma) for c in cands]
            X = S.stack(TF, m.blocks)
            bad = np.isnan(np.concatenate([X[b] for b in m.blocks], axis=1)).any(1)
            ps = m.predict_proba({b: np.nan_to_num(v) for b, v in X.items()})
            for c, p, bd in zip(cands, ps, bad):
                c[key] = None if bd else round(float(p), 4)
        dump(S.CUE_DIR / f"{slug}.json", {
            "schema_version": S.SCHEMA + ".cue_scores", "slug": slug, "artist": tr.artist, "group": tr.group,
            "oof": bool(group), "artist_seen_in_training": False,
            "note": "punteggio da un modello che non ha visto l'artista del brano (fold leave-one-artist-out)" if group else
                    "brano nuovo: modello addestrato su tutti i brani di insong_v1",
            "model_files": model_files, "period_s": tr.period_s,
            "meaning": "p_full/p_audio: probabilità che il brano si articoli da solo in t (inizio sezione sul downbeat, inizio "
                       "verso dopo pausa, rientro voce); tension_exit/resolution_entry: somiglianza del contesto prima/dopo t "
                       "alla carica che precede l'arrivo di un ritornello / all'arrivo (proxy [H], debole: vedi la nota). "
                       "Ordinamenti dentro il brano, non probabilità calibrate; non sono giudizi sulla transizione né scelte di DJ",
            "candidates": cands})
        log(f"cue {slug}: {len(cands)} candidati")


# =========================================================================== tensione -> risoluzione
def _tension_inputs(tracks):
    caches = {s: S.TrackCache.load(OUT, s) for s in tracks}
    forms = {s: S.load_form(s, len(t.downbeats_s)) for s, t in tracks.items()}
    chroma = {}
    for s in tracks:
        p = OUT / "cache/chroma" / f"{s}.npz"
        chroma[s] = np.load(p)["x"] if p.is_file() else None
    return caches, forms, chroma


def tension_metrics(rows, sc, role) -> dict:
    pos = [(r["slug"], v) for r, v in zip(rows, sc) if r["label"] == 1]
    out = {"all": S.auc([v for _, v in pos], [v for r, v in zip(rows, sc) if r["label"] == 0]), "n_pos": len(pos)}
    for typ in S.TENSION_NEG_TYPES[role]:
        neg = [(r["slug"], v) for r, v in zip(rows, sc) if r["label"] == 0 and r["kind"] == typ]
        lo, hi = S.bootstrap_auc_by_group(pos, neg, n_boot=300)
        out[typ] = {"n_neg": len(neg), "auc": S.auc([v for _, v in pos], [v for _, v in neg]), "ci95": [lo, hi]}
    return out


def stage_tension() -> None:
    tracks = tracks_by_slug()
    caches, forms, chroma = _tension_inputs(tracks)
    track_groups = {s: t.group for s, t in tracks.items()}
    data = {"exit": [], "entry": []}
    for slug, tr in tracks.items():
        ex = S.tension_examples(tr)
        for role in data:
            for i, r in enumerate(ex[role]):
                data[role].append({"id": f"{slug}:{role}{i}", "slug": slug, "group": tr.group, **r})
    res = {"schema_version": S.SCHEMA + ".tension_eval", "arrival_labels": S.TENSION_ARRIVAL_LABELS,
           "neg_types": S.TENSION_NEG_TYPES, "form_source": {s: (f or {}).get("path") for s, f in forms.items()},
           "note": "[H] proxy: arrivo del ritornello allin1 = arrivo; non equivale a 'risoluzione' (AGENTS.md)", "roles": {}}
    for role, rows in data.items():
        F = [S.tension_features(tracks[r["slug"]], caches[r["slug"]], r["t"], role, forms[r["slug"]], chroma[r["slug"]])
             for r in rows]
        X = S.stack(F, S.TENSION_ALL_BLOCKS)
        ok = ~np.isnan(np.concatenate([X[b] for b in S.TENSION_ALL_BLOCKS], axis=1)).any(1)
        rows = [r for r, k in zip(rows, ok) if k]
        X = {b: v[ok] for b, v in X.items()}
        y = np.asarray([r["label"] for r in rows])
        counts = {}
        for r in rows:
            counts[("pos:" if r["label"] else "neg:") + r["kind"]] = counts.get(("pos:" if r["label"] else "neg:") + r["kind"], 0) + 1
        rr = {"n": len(rows), "n_dropped_missing": int((~ok).sum()), "counts": counts, "variants": {}}
        groups = np.asarray([r["group"] for r in rows])
        slugs = np.asarray([r["slug"] for r in rows])
        for v, blocks in S.TENSION_BLOCKS.items():
            oof = np.full(len(rows), np.nan)
            for g in S.group_folds(groups):
                te, trm = groups == g, groups != g
                S.check_disjoint(set(slugs[trm]), set(slugs[te]), track_groups)
                m = S.BlockLogistic(blocks, pca_dims=16, c=0.3).fit(subset(X, np.where(trm)[0]), y[trm])
                oof[te] = m.predict_proba(subset(X, np.where(te)[0]))
                m.save(OUT / "model" / f"tension_{role}_{v}__wo_{S._safe(g)}.npz", meta={"held_out_group": g})
            S.BlockLogistic(blocks, pca_dims=16, c=0.3).fit(X, y).save(OUT / "model" / f"tension_{role}_{v}.npz",
                                                                       meta={"train_slugs": sorted(tracks)})
            rr["variants"][v] = {"blocks": list(blocks), "metrics": tension_metrics(rows, oof, role)}
            mm = rr["variants"][v]["metrics"]
            log(f"tensione {role:5s} {v:8s} AUC {mm['all']:.3f} | " + " ".join(f"{k}:{mm[k]['auc']:.2f}(n={mm[k]['n_neg']})"
                                                                         for k in S.TENSION_NEG_TYPES[role] if mm[k]["auc"] is not None))
        res["roles"][role] = rr
    dump(OUT / "tension_examples.json", data)
    dump(OUT / "results_tension.json", res)


def stage_tournament() -> None:
    """Controllo esterno: voti ciechi del torneo dei cue (giudizi_cloud/b_entry_fix_v1), mai usati per addestrare."""
    jdir = ROOT / "giudizi_cloud/b_entry_fix_v1/judgments"
    tracks = tracks_by_slug()
    caches, forms, chroma = _tension_inputs(tracks)
    rows = []
    for f in sorted(jdir.glob("*__torneo.json")):
        T = load(f)
        a, b = T["pair"]["a"], T["pair"]["b"]
        if a not in tracks or b not in tracks:
            continue
        acc = set(T.get("acceptable") or [])
        for vid in T["order_shown"]:
            vp = jdir / f.name.replace("torneo", vid)
            if not vp.is_file():
                continue
            J = load(vp)
            c = J["cue"]
            r = {"tournament": T["tournament_id"], "a": a, "b": b, "variant": vid, "technique": T.get("technique"),
                 "a_out_s": c["a_out_s"], "b_in_s": c["b_in_s"], "b_lead_in_s": (c.get("b_lead_in_samples") or 0) / 48000,
                 "accepted": vid in acc, "best": T.get("best") == vid, "tournament_any_accepted": bool(acc),
                 "tags": J.get("tags"), "note": J.get("note")}
            for v in ("dsp_form", "contrast", "full"):
                for role, slug, t in (("exit", a, c["a_out_s"]), ("entry", b, c["b_in_s"])):
                    m = S.BlockLogistic.load(OUT / "model" / f"tension_{role}_{v}__wo_{S._safe(tracks[slug].group)}.npz")
                    X = S.stack([S.tension_features(tracks[slug], caches[slug], t, role, forms[slug], chroma[slug])], m.blocks)
                    r[f"{'tension_exit' if role == 'exit' else 'resolution_entry'}_{v}"] = float(m.predict_proba(X)[0])
                r[f"tr_combined_{v}"] = float(np.sqrt(r[f"tension_exit_{v}"] * r[f"resolution_entry_{v}"]))
            for v in SAVE_VARIANTS:
                ma, mb = point_scorer(v, tracks[a].group), point_scorer(v, tracks[b].group)
                r[f"cue_exit_{v}"] = float(ma.predict_proba(S.stack([S.point_features(tracks[a], caches[a], c["a_out_s"])], ma.blocks))[0])
                t_aud = c["b_in_s"] - r["b_lead_in_s"]
                r[f"cue_entry_{v}"] = float(mb.predict_proba(S.stack([S.point_features(tracks[b], caches[b], t_aud)], mb.blocks))[0])
            rows.append(r)
    keys = [k for k in rows[0] if k.startswith(("tension_", "resolution_", "tr_combined", "cue_"))] if rows else []
    tests = {}
    for k in keys:
        acc = [r[k] for r in rows if r["accepted"]]
        rej = [r[k] for r in rows if not r["accepted"]]
        # dentro i tornei con accettate e rifiutate (cue diversi, stessa coppia)
        within = []
        for tid in sorted({r["tournament"] for r in rows}):
            rs = [r for r in rows if r["tournament"] == tid]
            A = [r[k] for r in rs if r["accepted"]]
            R = [r[k] for r in rs if not r["accepted"]]
            if A and R:
                within.append({"tournament": tid, "auc": S.auc(A, R), "n_acc": len(A), "n_rej": len(R)})
        best_rank = []
        for tid in sorted({r["tournament"] for r in rows}):
            rs = [r for r in rows if r["tournament"] == tid]
            bs = [r for r in rs if r["best"]]
            if bs:
                best_rank.append({"tournament": tid, "rank": 1 + sum(r[k] > bs[0][k] for r in rs), "n": len(rs)})
        # per torneo: media del punteggio contro "almeno una accettata"
        tm = {}
        for r in rows:
            tm.setdefault(r["tournament"], {"x": [], "y": r["tournament_any_accepted"]})["x"].append(r[k])
        tpos = [float(np.mean(v["x"])) for v in tm.values() if v["y"]]
        tneg = [float(np.mean(v["x"])) for v in tm.values() if not v["y"]]
        tests[k] = {"auc_accepted_vs_rejected": S.auc(acc, rej), "n_acc": len(acc), "n_rej": len(rej),
                    "auc_tournament_any_accepted": S.auc(tpos, tneg), "n_t_pos": len(tpos), "n_t_neg": len(tneg),
                    "within_tournament": within, "best_rank": best_rank}
        log(f"torneo {k:24s} AUC accettate/rifiutate {fmt(tests[k]['auc_accepted_vs_rejected'])} ({len(acc)}/{len(rej)}) "
            f"tornei {fmt(tests[k]['auc_tournament_any_accepted'])} ({len(tpos)}/{len(tneg)}) "
            f"migliore: {[b['rank'] for b in best_rank]}")
    dump(OUT / "results_tournament.json", {"schema_version": S.SCHEMA + ".tournament_check",
                                           "source": "giudizi_cloud/b_entry_fix_v1/judgments (solo controllo esterno)",
                                           "n_variants": len(rows), "n_tournaments": len({r["tournament"] for r in rows}),
                                           "rows": rows, "tests": tests})


def main(stage: str, args) -> None:
    if stage == "train":
        stage_train()
    elif stage == "pairs":
        stage_pairs()
    elif stage == "cocola-eval":
        log(json.dumps(stage_cocola_eval(), indent=1)[:3000])
    elif stage == "external":
        stage_external()
    elif stage == "tension":
        stage_tension()
    elif stage == "tournament":
        stage_tournament()
    elif stage == "cues":
        stage_cues()
    else:
        raise SystemExit(f"fase sconosciuta: {stage}")
