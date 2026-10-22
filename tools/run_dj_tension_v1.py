#!/usr/bin/env python3
"""Pre-registrazione e controllo esterno di ``musicmix.dj_tension`` (run ``runs/dj_tension_v1``).

Fasi:

* ``prereg``: scrive ``runs/dj_tension_v1/PREREGISTRATION.json`` (pesi, soglie, ipotesi, metriche, SHA-256 del modulo)
  **prima** di aprire i voti; si rifiuta di sovrascriverla (gli emendamenti post-hoc vanno in ``AMENDMENTS.json``).
* ``external``: calcola i punteggi sulle varianti del torneo dei cue (``runs/b_entry_fix_v1/cue_variants.json``) e sulle clip
  della galleria cieca del set v3 (``runs/dj_set_v3_bridge/gallery/review_manifest.json``), poi li confronta con i voti
  (``giudizi_cloud/b_entry_fix_v1/judgments``, ``giudizi_cloud/dj_set_v3_bridge/judgments``): AUC con IC bootstrap per
  torneo/gruppo, Spearman con p di permutazione. I voti sono **solo** controllo esterno: nessun peso è stimato su di essi.

    python tools/run_dj_tension_v1.py prereg
    python tools/run_dj_tension_v1.py external
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicmix import dj_tension as T  # noqa: E402
from musicmix.paths import localize  # noqa: E402

OUT = ROOT / "runs" / "dj_tension_v1"
SRC_RUN = ROOT / "runs" / "dj_set_v3_bridge"
TOURN = ROOT / "runs" / "b_entry_fix_v1" / "cue_variants.json"
TOURN_J = ROOT / "giudizi_cloud" / "b_entry_fix_v1" / "judgments"
GAL = SRC_RUN / "gallery" / "review_manifest.json"
GAL_J = ROOT / "giudizi_cloud" / "dj_set_v3_bridge" / "judgments"
SEED = 20261022
N_BOOT = 2000
SCORES = ("preference", "a_charge", "b_resolution", "continuity", "neg_clash", "lyric_affinity", "continuity_measured")


def load(p: Path) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def dump(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                 encoding="utf-8")


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ prereg
def prereg() -> dict:
    p = OUT / "PREREGISTRATION.json"
    if p.exists():
        raise SystemExit(f"{p} esiste: la pre-registrazione non si sovrascrive (emendamenti in AMENDMENTS.json)")
    doc = {
        "schema_version": "musicmix.preregistration.1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "module": "src/musicmix/dj_tension.py", "module_version": T.DJ_TENSION_VERSION,
        "module_sha256": sha(ROOT / "src/musicmix/dj_tension.py"),
        "tool_sha256": sha(Path(__file__)),
        "status_of_knowledge": (
            "Pesi e soglie fissati prima di aprire i file dei voti (giudizi_cloud/*/judgments). Il progettista conosceva però i "
            "criteri dell'utente riassunti dall'orchestratore (A carica / B risolve, base continua, tonalità, testi affini, esempi "
            "positivi 5% TINT->Creature, Butterfly Effect->Knife Talk, A Lot->Nonstop) e i risultati aggregati già pubblicati in "
            "conoscenza_mix/_meta/SUPERVISIONE_DAI_BRANI_2026-10-21.md §6 (tension_at_exit AUC 0,76 tra tornei, resolution_at_entry "
            "0,35). Il controllo non è quindi del tutto cieco: le componenti sono state scelte sapendo che 'ritornello = risoluzione' "
            "era contrario ai voti."),
        "weights": T.documented(),
        "inputs": {"tracks": "runs/dj_set_v3_bridge (dj_plan/tracks, phrase: le analisi con cui le clip sono state fatte)",
                   "dsp_chroma": "runs/insong_v1/cache (stessa ricetta per i brani mancanti)",
                   "model": "runs/insong_v1/cue_scores (fold che non ha visto l'artista)",
                   "tournament": "runs/b_entry_fix_v1/cue_variants.json (cue, lead-in, tecnica, clip)",
                   "gallery": "runs/dj_set_v3_bridge/gallery/review_manifest.json (cue, tecnica, parametri, clip)",
                   "played_a": "A suonata dal suo ingresso nel set v3 congelato (runs/dj_set_v3_bridge/set.json) fino all'uscita"},
        "score_definitions": {
            "preference": "- costo dj_tension.transition_cost (combinazione pre-registrata, W_COST)",
            "a_charge": "dj_tension.a_charge(A, a_out_s, played_s)",
            "b_resolution": "dj_tension.b_resolution(B, punto udibile, downbeat, A)",
            "continuity": "dj_tension.continuity(tecnica): pianificata",
            "neg_clash": "- dj_tension.harmonic_clash",
            "lyric_affinity": "dj_tension.lyric_affinity (ultime 3 parole di A, prime 4 di B)",
            "continuity_measured": "dj_tension.continuity_measured sulla clip votata (secondaria)"},
        "hypotheses": {
            "H1": "torneo: AUC(preference) accettate vs rifiutate > 0,5 (IC bootstrap per torneo)",
            "H2": "torneo: media delle AUC dentro i tornei con accettate e rifiutate > 0,5",
            "H3": "torneo: la variante 'migliore' ha un rango percentile medio sopra 0,5 nel suo torneo",
            "H4": "torneo: tornei con almeno un'accettata hanno preference media più alta di quelli senza (AUC > 0,5)",
            "H5": "torneo: AUC(a_charge) > 0,5 e AUC(b_resolution) > 0,5 (atteso: b_resolution migliore dello 0,35 di resolution_at_entry)",
            "H6": "galleria v3: Spearman(preference, voto) per clip > 0 e per gruppo (media dei voti) > 0",
            "H7": "galleria v3: dentro un gruppo (stessi cue, tecniche diverse) la versione con continuità pianificata più alta "
                  "ha il voto più alto (conteggio dei gruppi con continuità diversa, test del segno)",
        },
        "metrics": {"auc": "Mann-Whitney (pari = 0,5)", "ci": f"bootstrap percentile 95 % ricampionando tornei/gruppi, {N_BOOT} "
                                                             f"repliche, seed {SEED}",
                    "spearman": f"rho con p di permutazione bilaterale ({N_BOOT} permutazioni, seed {SEED})",
                    "labels": {"tournament": "accettata = id in 'acceptable' del torneo; migliore = 'best'",
                               "gallery": "voto 1-10 della clip ('score')"}},
        "rules": ["pesi e soglie non si ritoccano dopo aver visto i risultati; ogni modifica = emendamento post-hoc in AMENDMENTS.json "
                  "con motivo, data e nuova SHA del modulo, e i risultati pre-emendamento restano nel report",
                  "un solo ascoltatore, pochi voti: nessun risultato è una validazione; le direzioni contano più dei decimali"],
    }
    dump(p, doc)
    print(f"pre-registrazione: {p} (modulo {doc['module_sha256'][:12]})")
    return doc


# ---------------------------------------------------------------------------------------------- statistiche
def auc(pos, neg) -> Optional[float]:
    pos, neg = [x for x in pos if x is not None], [x for x in neg if x is not None]
    if not pos or not neg:
        return None
    s = 0.0
    for p in pos:
        for n in neg:
            s += 1.0 if p > n else 0.5 if p == n else 0.0
    return s / (len(pos) * len(neg))


def boot_auc(rows: list, key: str, label: str, group: str, rng: np.random.Generator) -> dict:
    groups = sorted({r[group] for r in rows})
    by = {g: [r for r in rows if r[group] == g] for g in groups}
    pt = auc([r[key] for r in rows if r[label]], [r[key] for r in rows if not r[label]])
    bs = []
    for _ in range(N_BOOT):
        gs = rng.choice(groups, size=len(groups), replace=True)
        rr = [r for g in gs for r in by[g]]
        a = auc([r[key] for r in rr if r[label]], [r[key] for r in rr if not r[label]])
        if a is not None:
            bs.append(a)
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None
    return {"auc": pt, "ci95": ci, "n_pos": sum(1 for r in rows if r[label] and r[key] is not None),
            "n_neg": sum(1 for r in rows if not r[label] and r[key] is not None), "n_groups": len(groups)}


def spearman(x, y) -> Optional[float]:
    from scipy.stats import spearmanr

    xy = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(xy) < 3:
        return None
    r = spearmanr([a for a, _ in xy], [b for _, b in xy]).correlation
    return None if r is None or not np.isfinite(r) else float(r)


def perm_p(x, y, rng) -> Optional[float]:
    r0 = spearman(x, y)
    if r0 is None:
        return None
    xy = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    xs = np.asarray([a for a, _ in xy])
    ys = np.asarray([b for _, b in xy])
    c = 0
    for _ in range(N_BOOT):
        r = spearman(xs, rng.permutation(ys))
        if r is not None and abs(r) >= abs(r0) - 1e-12:
            c += 1
    return (c + 1) / (N_BOOT + 1)


# ------------------------------------------------------------------------------------------------ punteggi
_SIG: dict = {}


def sig(slug: str) -> T.TrackSignals:
    if slug not in _SIG:
        _SIG[slug] = T.load_run_signals(slug, SRC_RUN, compute_missing=True)
    return _SIG[slug]


def played_in_v3(a: str, a_out_s: float) -> Optional[float]:
    doc = load(SRC_RUN / "set.json")
    for t in doc["transitions"]:
        if t["b"] == a:
            return float(a_out_s) - float(t["plan"]["b_in_s"])
    if doc["tracklist"] and doc["tracklist"][0]["slug"] == a:
        return float(a_out_s)
    return None


def score_row(a: str, b: str, a_out_s: float, b_in_s: float, b_aud: float, decision: dict, clip: Optional[Path],
              switch_s: Optional[float]) -> dict:
    sa, sb = sig(a), sig(b)
    r = T.score_transition(sa, sb, a_out_s=a_out_s, b_in_s=b_in_s, decision=decision, b_audible=b_aud,
                           played_a_s=played_in_v3(a, a_out_s))
    row = {"preference": T.preference_score(r), "a_charge": r["a_charge"]["value"], "b_resolution": r["b_resolution"]["value"],
           "continuity": r["continuity"]["value"], "neg_clash": None if r["harmonic_clash"]["value"] is None else -r["harmonic_clash"]["value"],
           "lyric_affinity": r["lyric_affinity"]["value"], "detail": T.compact(r),
           "a_charge_why": r["a_charge"]["why"], "b_resolution_why": r["b_resolution"]["why"], "lyric_why": r["lyric_affinity"]["why"]}
    row["continuity_measured"] = None
    if clip is not None and switch_s is not None and Path(clip).is_file():
        import soundfile as sf

        try:
            x, sr = sf.read(str(clip), dtype="float32", always_2d=True)
        except Exception:
            import librosa

            y, sr = librosa.load(str(clip), sr=48000, mono=True)
            x = y[:, None]
        cm = T.continuity_measured(x, sr, int(round(switch_s * sr)), sa.period_s)
        row["continuity_measured"] = cm["value"]
        row["continuity_measured_detail"] = {k: cm.get(k) for k in ("gap_beats_low", "gap_beats_full", "why")}
    return row


def tournament_rows() -> list:
    cv = load(TOURN)
    v3 = {t["index"]: t for t in load(SRC_RUN / "set.json")["transitions"]}
    rows = []
    for tt in cv["tournaments"]:
        a, b = tt["a"]["slug"], tt["b"]["slug"]
        tr = v3.get(tt.get("transition_index")) or {}
        opt = next((o for o in tr.get("books_options_same_cue") or [] if o["type_key"] == tr.get("type")), None)
        for v in tt["variants"]:
            if v.get("error") or v.get("a_out_s") is None:
                continue
            aud = float(v["b_in_s"]) - (v.get("b_lead_in_samples") or 0) / 48000.0
            # tecnica fissa del torneo = quella del set v3, con i suoi parametri (es. gap_beats del build)
            render = (opt or {}).get("render") if opt and opt["type_key"] == v["technique"] else (v.get("render") or {})
            dec = {"type": "blend" if v["technique"].startswith("blend") else v["technique"], "type_key": v["technique"],
                   "render": render or {}}
            r = score_row(a, b, float(v["a_out_s"]), float(v["b_in_s"]), aud, dec, ROOT / v["audio_path"], v.get("switch_s"))
            rows.append({"tournament": tt["id"], "variant": v["id"], "label": v.get("label"), "a": a, "b": b,
                         "technique": v["technique"], "a_out_bar": v.get("a_out_bar"), "b_in_bar": v.get("b_in_bar"), **r})
    return rows


def gallery_rows() -> list:
    man = load(GAL)
    rows = []
    for g in man["groups"]:
        if "A_out_s" not in g:
            continue
        a, b = g["pair"]["a"]["slug"], g["pair"]["b"]["slug"]
        for it in g["items"]:
            pr = it.get("params") or {}
            render = pr.get("render") or {}
            kind = it["kind"]
            dec = {"type": "blend" if str(pr.get("technique", kind)).startswith("blend") else kind,
                   "type_key": pr.get("technique", kind), "render": render}
            r = score_row(a, b, float(g["A_out_s"]), float(g["B_in_s"]), float(g["B_in_s"]), dec, ROOT / it["audio_path"],
                          it.get("switch_s"))
            rows.append({"group": g["id"], "clip": it["id"], "a": a, "b": b, "technique": pr.get("technique", kind), **r})
    return rows


# ------------------------------------------------------------------------------------------------ esterno
def external() -> dict:
    pre = OUT / "PREREGISTRATION.json"
    if not pre.is_file():
        raise SystemExit("manca la pre-registrazione: eseguire prima 'prereg'")
    P = load(pre)
    msha = sha(ROOT / "src/musicmix/dj_tension.py")
    amend = load(OUT / "AMENDMENTS.json") if (OUT / "AMENDMENTS.json").is_file() else {"amendments": []}
    if msha != P["module_sha256"] and not any(a.get("module_sha256") == msha for a in amend["amendments"]):
        raise SystemExit("il modulo è cambiato dopo la pre-registrazione senza un emendamento registrato in AMENDMENTS.json")
    rng = np.random.default_rng(SEED)
    # --- punteggi (senza voti)
    trow, grow = tournament_rows(), gallery_rows()
    # --- voti (solo ora)
    for r in trow:
        tj = load(TOURN_J / f"b_entry_fix_v1__{r['tournament']}__torneo.json") if (TOURN_J / f"b_entry_fix_v1__{r['tournament']}__torneo.json").is_file() else None
        r["judged"] = tj is not None and r["variant"] in (tj.get("order_shown") or [])
        acc = set((tj or {}).get("acceptable") or [])
        r["accepted"] = r["variant"] in acc
        r["best"] = (tj or {}).get("best") == r["variant"]
        r["tournament_any_accepted"] = bool(acc)
        vj = TOURN_J / f"b_entry_fix_v1__{r['tournament']}__{r['variant']}.json"
        if vj.is_file():
            J = load(vj)
            r["note"], r["tags"] = J.get("note"), J.get("tags")
    trow = [r for r in trow if r["judged"]]
    for r in grow:
        jp = GAL_J / f"dj_set_v3_bridge__{r['clip']}.json"
        J = load(jp) if jp.is_file() else {}
        r["score"] = J.get("score")
        r["note"] = J.get("note")
    grow = [r for r in grow if r["score"] is not None]
    res: dict[str, Any] = {"schema_version": "musicmix.dj_tension_external.1", "preregistration": str(pre.relative_to(ROOT)),
                           "module_sha256": msha, "preregistered_module_sha256": P["module_sha256"],
                           "n": {"tournament_variants": len(trow), "tournaments": len({r['tournament'] for r in trow}),
                                 "accepted": sum(r["accepted"] for r in trow), "gallery_clips": len(grow),
                                 "gallery_groups": len({r['group'] for r in grow})},
                           "caveat": "un solo ascoltatore, pochi voti: diagnostica, non validazione; nessun peso stimato sui voti",
                           "tournament": {}, "gallery": {}}
    for k in SCORES:
        t = {"H1_auc_accepted": boot_auc(trow, k, "accepted", "tournament", rng)}
        within = []
        for tid in sorted({r["tournament"] for r in trow}):
            rs = [r for r in trow if r["tournament"] == tid]
            A = [r[k] for r in rs if r["accepted"]]
            R = [r[k] for r in rs if not r["accepted"]]
            if A and R:
                within.append({"tournament": tid, "auc": auc(A, R), "n_acc": len(A), "n_rej": len(R)})
        ws = [w["auc"] for w in within if w["auc"] is not None]
        t["H2_within"] = {"mean_auc": float(np.mean(ws)) if ws else None, "n_tournaments": len(ws), "rows": within}
        ranks = []
        for tid in sorted({r["tournament"] for r in trow}):
            rs = [r for r in trow if r["tournament"] == tid and r[k] is not None]
            bs = [r for r in rs if r["best"]]
            if bs and len(rs) > 1:
                below = sum(1 for r in rs if r[k] < bs[0][k]) + 0.5 * sum(1 for r in rs if r[k] == bs[0][k] and r is not bs[0])
                ranks.append({"tournament": tid, "percentile": below / (len(rs) - 1), "n": len(rs)})
        t["H3_best_rank"] = {"mean_percentile": float(np.mean([x["percentile"] for x in ranks])) if ranks else None,
                             "n": len(ranks), "rows": ranks}
        tm: dict = {}
        for r in trow:
            if r[k] is not None:
                tm.setdefault(r["tournament"], {"x": [], "y": r["tournament_any_accepted"]})["x"].append(r[k])
        t["H4_tournament_level"] = {"auc": auc([float(np.mean(v["x"])) for v in tm.values() if v["y"]],
                                               [float(np.mean(v["x"])) for v in tm.values() if not v["y"]]),
                                    "n_pos": sum(1 for v in tm.values() if v["y"]), "n_neg": sum(1 for v in tm.values() if not v["y"])}
        res["tournament"][k] = t
        g = {"H6_spearman_clip": {"rho": spearman([r[k] for r in grow], [r["score"] for r in grow]),
                                  "p_perm": perm_p([r[k] for r in grow], [r["score"] for r in grow], rng),
                                  "n": sum(1 for r in grow if r[k] is not None)}}
        gm: dict = {}
        for r in grow:
            if r[k] is not None:
                d = gm.setdefault(r["group"], {"x": [], "y": []})
                d["x"].append(r[k])
                d["y"].append(r["score"])
        gx = [float(np.mean(v["x"])) for v in gm.values()]
        gy = [float(np.mean(v["y"])) for v in gm.values()]
        g["H6_spearman_group"] = {"rho": spearman(gx, gy), "p_perm": perm_p(gx, gy, rng), "n": len(gx)}
        pairs = []
        for gid, v in gm.items():
            rs = [r for r in grow if r["group"] == gid and r[k] is not None]
            if len(rs) == 2 and rs[0][k] != rs[1][k] and rs[0]["score"] != rs[1]["score"]:
                hi = max(rs, key=lambda r: r[k])
                lo = min(rs, key=lambda r: r[k])
                pairs.append({"group": gid, "agree": hi["score"] > lo["score"], "x": [lo[k], hi[k]], "votes": [lo["score"], hi["score"]]})
        n_ag = sum(p["agree"] for p in pairs)
        from scipy.stats import binomtest

        g["H7_within_group_sign"] = {"agree": n_ag, "n": len(pairs),
                                     "p_sign": binomtest(n_ag, len(pairs), 0.5).pvalue if pairs else None, "rows": pairs}
        res["gallery"][k] = g
    res["rows"] = {"tournament": trow, "gallery": grow}
    dump(OUT / "external.json", res)
    for k in SCORES:
        t, g = res["tournament"][k], res["gallery"][k]
        h1 = t["H1_auc_accepted"]
        print(f"{k:20s} torneo AUC {fmt(h1['auc'])} {fmtci(h1['ci95'])} ({h1['n_pos']}/{h1['n_neg']}); dentro {fmt(t['H2_within']['mean_auc'])} "
              f"(n={t['H2_within']['n_tournaments']}); migliore pct {fmt(t['H3_best_rank']['mean_percentile'])} (n={t['H3_best_rank']['n']}); "
              f"tornei {fmt(t['H4_tournament_level']['auc'])} | galleria rho {fmt(g['H6_spearman_clip']['rho'])} p={fmt(g['H6_spearman_clip']['p_perm'])} "
              f"(n={g['H6_spearman_clip']['n']}), gruppi {fmt(g['H6_spearman_group']['rho'])} (n={g['H6_spearman_group']['n']}); "
              f"segno {g['H7_within_group_sign']['agree']}/{g['H7_within_group_sign']['n']}", flush=True)
    return res


def fmt(x) -> str:
    return "n/d" if x is None else f"{x:.2f}"


def fmtci(ci) -> str:
    return "[n/d]" if not ci else f"[{ci[0]:.2f}-{ci[1]:.2f}]"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stages", nargs="+", choices=("prereg", "external"))
    args = ap.parse_args(argv)
    for s in args.stages:
        prereg() if s == "prereg" else external()
    return 0


if __name__ == "__main__":
    sys.exit(main())
