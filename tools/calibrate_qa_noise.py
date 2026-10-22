#!/usr/bin/env python3
"""Rumore delle misure del QA sulle giunte naturali (calibrazione di dj_qa.7, palestra dei difetti v2).

Una giunta naturale è un brano che continua sé stesso nel punto in cui il renderer cambierebbe brano: cut, echo out
(1 battuta, eco di 1 battito) e filter out (2 battute) dallo stesso downbeat di inizio frase (B = A, stessa posizione:
la griglia di B prosegue quella di A), e blend di 4 battute tra due inizi frase diversi dello stesso brano (B
beatmatchato sotto A). Ogni misura che il QA fa su queste clip è rumore o la dinamica propria del brano: nessun
difetto è presente per costruzione (salvo le parole che attraversano il punto, che qui non contano: si misurano solo
ritmo, buco, clic, eco, frase e basso).

Brani e punti sono **disgiunti** dai riferimenti naturali della palestra (`runs/defect_gym_v1/refs.json`: brani
`nat_*`, `nsec_*`, `nbl_*` esclusi del tutto), così i riferimenti restano una verifica tenuta separata. Nessun voto
dell'utente è letto.

    python tools/calibrate_qa_noise.py run      # misure (incrementale) -> calibration_points.jsonl
    python tools/calibrate_qa_noise.py summary  # quantili e soglie -> calibration.json

Le soglie proposte seguono regole scritte qui prima di guardare le varianti della palestra (``RULES``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import defect_gym as T  # noqa: E402

OUT = ROOT / "runs" / "defect_gym_v2"
POINTS = OUT / "calibration_points.jsonl"
SUMMARY = OUT / "calibration.json"
MAX_POINTS_PER_TRACK = 4
BLENDS_PER_TRACK = 2
KINDS = {
    "cut": {"type": "cut", "type_key": "cut", "render": {}},
    "echo_out": {"type": "echo_out", "type_key": "echo_out", "render": {"delay_beats": 1.0, "entry_bars": 1}},
    "filter_out": {"type": "filter_out", "type_key": "filter_out", "render": {"sweep_bars": 2}},
}
RULES = {
    "declared": "regole scritte il 22/10/2026 prima della run v2 della palestra; la prima versione (1.5 x p95, minimo 40 ms) è stata "
                "sostituita dopo aver visto SOLO i punti di calibrazione (non la palestra): il rumore affidabile ha code (Beat This! "
                "sbilanciato di ~40 ms su un brano a mezzo tempo) e il p95 non lo copre",
    "excluded_tracks": "brani la cui scheda chiede di ascoltare la griglia con gravità >= 0.3 (rich_flex: griglia low e cambio di "
                       "beat; antidote: ottava sospetta): una continuazione naturale lì non è ritmicamente pulita",
    "entry_phase_fail_ms": "1.5 x max |offset| sulle giunte naturali pulite misurate in modo affidabile (brani esclusi a parte), "
                           "arrotondata a 5 ms; deve restare sotto 1/4 di battito del brano più veloce (altrimenti limite dichiarato)",
    "overlap": "xcorr dei deck: max(15 ms dell'utente, 1.5 x max dei blend naturali puliti); deck-vs-deck di Beat This!: solo warn se "
               "il suo rumore p95 supera 1/8 di battito; linee di battito sovrapposte: stessa soglia dell'ingresso pulito se il loro "
               "massimo affidabile x 1.5 non la supera",
    "dropout": "silenzio non dichiarato: lato B >= max(0.5 battiti, max naturale + 0.1); regione dell'effetto >= max(1.5 battiti, "
               "max naturale + 0.25)",
    "click": "fail sopra 1.5 x max naturale (minimo 4); warn sopra il max naturale",
    "echo": "se le giunte naturali hanno progressioni fuori griglia (B = A ripete il proprio materiale) il controllo resta warn",
}


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gym_natural_slugs() -> set:
    refs = T.load(ROOT / "runs/defect_gym_v1/refs.json")["refs"]
    return {r["a"] for r in refs if r["origin"].startswith("natural")}


def candidate_points(X) -> list[int]:
    an = X.analysis
    downs = X.downbeats
    dur = float(an["duration_s"])
    out = []
    for s in an["phrases"]:
        if not (s.get("section_start") or int(s.get("phrase_level") or 0) >= 4):
            continue
        j = int(s["bar_index"])
        if j < 2 or j >= len(downs):
            continue
        d = float(downs[j])
        if d < T.NATURAL_MIN_PRE_S + 4 or d + T.NATURAL_MIN_POST_S > dur:
            continue
        out.append(j)
    out = sorted(set(out))
    if len(out) > MAX_POINTS_PER_TRACK:
        idx = np.linspace(0, len(out) - 1, MAX_POINTS_PER_TRACK).round().astype(int)
        out = [out[i] for i in idx]
    return out


def compact(q: dict) -> dict:
    """Misure del QA utili alla calibrazione (niente array lunghi)."""
    ch = q["checks"]
    rh = ch.get("rhythm") or {}
    meth = rh.get("methods") or {}
    out = {"verdict": q["verdict"], "reasons": [f"{r['check']}: {r['message']}" for r in q["reasons"]],
           "rhythm": {"status": rh.get("status"), "coverage": rh.get("coverage"), "overlap": rh.get("rhythmic_overlap")}}
    en = meth.get("beat_this_entry")
    if en:
        out["rhythm"]["entry"] = {k: en.get(k) for k in ("status", "reliable", "beat_phase_offset_at_switch_ms", "period_s",
                                                         "consensus_spread_ms", "reliability_problems", "a_beats_used", "b_beats_used")}
        out["rhythm"]["entry"]["flux"] = en.get("flux")
        out["rhythm"]["entry"]["rms"] = [(en.get("a_line") or {}).get("rms_resid_ms"), (en.get("b_line") or {}).get("rms_resid_ms")]
    bt = meth.get("beat_this_decks")
    if bt:
        out["rhythm"]["decks"] = {k: bt.get(k) for k in ("status", "downbeat_median_abs_ms", "beat_median_abs_ms", "paired_downbeats",
                                                         "paired_beats", "beat_period_s")}
        out["rhythm"]["decks"]["downbeat_drift"] = bt.get("downbeat_drift")
        out["rhythm"]["decks"]["beat_drift"] = bt.get("beat_drift")
    xc = meth.get("onset_xcorr_decks")
    if xc:
        out["rhythm"]["xcorr"] = {k: xc.get(k) for k in ("status", "median_abs_lag_ms", "median_lag_ms", "confident_windows", "windows")}
        out["rhythm"]["xcorr"]["drift"] = xc.get("drift")
    ol = meth.get("beat_this_overlap_lines")
    if ol:
        out["rhythm"]["overlap_lines"] = {k: ol.get(k) for k in ("status", "reliable", "beat_phase_offset_at_switch_ms", "reliability_problems")}
    mo = meth.get("beat_this_mix")
    if mo:
        out["rhythm"]["mix"] = {k: mo.get(k) for k in ("status", "beat_phase_offset_at_switch_ms")}
    for name in ("dropout", "click", "echo_tempo", "phrase", "double_bass", "loudness_jump"):
        c = ch.get(name)
        if c is None:
            continue
        out[name] = {k: v for k, v in c.items() if not isinstance(v, (list, dict)) or k in ("measures", "stats", "evidence_summary")}
    return out


def run(args) -> None:
    T.RT.init()
    R = T.RT.R
    cat = T._set_ctx()["cat"]
    excluded = gym_natural_slugs()
    done = set()
    if POINTS.exists():
        for line in POINTS.read_text().splitlines():
            row = json.loads(line)
            done.add((row["slug"], row["kind"], row["bar"], row.get("bar_b")))
    OUT.mkdir(parents=True, exist_ok=True)
    fh = open(POINTS, "a", encoding="utf-8")
    slugs = [e["slug"] for e in cat["tracks"]] if isinstance(cat.get("tracks"), list) else sorted(cat)
    tf = T.RT.TF.default_filter()
    for slug in slugs:
        if slug in excluded or (args.tracks and slug not in args.tracks):
            continue
        if not tf.check(slug)["admitted"]:
            continue
        try:
            X = T.track(slug, R.PLAN, cat)
        except Exception as exc:  # noqa: BLE001
            log("skip", slug, exc)
            continue
        pts = candidate_points(X)
        jobs = [(k, j, None) for j in pts for k in KINDS]
        for i in range(min(BLENDS_PER_TRACK, len(pts))):
            far = [q for q in pts if abs(q - pts[i]) >= 8]
            if far:
                jobs.append(("blend4", pts[i], far[-1] if i % 2 == 0 else far[0]))
        log(f"== {slug}: {len(pts)} punti, {len(jobs)} clip")
        for kind, j, jb in jobs:
            if (slug, kind, j, jb) in done:
                continue
            t0 = time.time()
            row = {"slug": slug, "kind": kind, "bar": j, "bar_b": jb, "excluded_gym_slugs": sorted(excluded)}
            try:
                plan = {"a_out_s": float(X.downbeats[j]), "b_in_s": float(X.downbeats[jb if jb is not None else j])}
                dec = dict(KINDS[kind]) if kind in KINDS else {"type": "blend", "type_key": "blend4", "render": {"bars": 4}}
                prov = {"a": X.provenance, "b": X.provenance, "script": "tools/calibrate_qa_noise.py", "natural": True}
                res, decks = R.v2.render(dec, X, X, plan, prov)
                bd = T.make_bundle(res, decks, X, X, dec, False)
                q = T.RT.qa.evaluate_render(bd.audio, bd.metadata, sample_rate=bd.sr, decks=bd.qa_decks, vocals_a=bd.vocals_a,
                                            vocals_b=bd.vocals_b, vocal_thresholds=bd.vocal_thresholds, words_a=bd.words_a,
                                            tracker=T.RT.get_tracker(), config=R.QA_CFG, grid_support=bd.grid_support,
                                            words_b=bd.words_b)
                row.update(qa_version=q["schema_version"], **compact(q))
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["trace"] = traceback.format_exc()[-1200:]
            row["seconds"] = time.time() - t0
            fh.write(json.dumps(T.G.jsonable(row), ensure_ascii=False) + "\n")
            fh.flush()
            en = ((row.get("rhythm") or {}).get("entry") or {})
            log(f"   {kind:10s} bar {j}{'' if jb is None else '->' + str(jb)}: {row.get('verdict', row.get('error'))} "
                f"entry={en.get('beat_phase_offset_at_switch_ms')} rel={en.get('reliable')} [{row['seconds']:.0f} s]")
    fh.close()


def _q(v, p):
    v = [x for x in v if x is not None]
    return float(np.percentile(v, p)) if v else None


def summary(args) -> dict:
    rows = [json.loads(x) for x in POINTS.read_text().splitlines() if x.strip()]
    excl = set()
    for slug in {r["slug"] for r in rows}:
        card = ROOT / "track_cards" / f"{slug}.json"
        if card.exists():
            unc = json.loads(card.read_text()).get("uncertain") or []
            if any(u.get("category") == "grid" and float(u.get("severity") or 0) >= 0.3 for u in unc):
                excl.add(slug)
    all_ok = [r for r in rows if not r.get("error")]
    ok = [r for r in all_ok if r["slug"] not in excl]
    entry = [r["rhythm"]["entry"] for r in ok if (r.get("rhythm") or {}).get("entry")]
    rel = [e for e in entry if e.get("reliable")]
    offs = [abs(e["beat_phase_offset_at_switch_ms"]) for e in rel if e.get("beat_phase_offset_at_switch_ms") is not None]
    all_offs = [abs(e["beat_phase_offset_at_switch_ms"]) for e in entry if e.get("beat_phase_offset_at_switch_ms") is not None]
    periods = [e["period_s"] for e in entry if e.get("period_s")]
    p95 = _q(offs, 95)
    mx = _q(offs, 100)
    thr = None if mx is None else float(np.ceil(1.5 * mx / 5.0) * 5.0)
    ols = [abs(r["rhythm"]["overlap_lines"]["beat_phase_offset_at_switch_ms"]) for r in ok
           if (r.get("rhythm") or {}).get("overlap_lines", {}).get("reliable")]
    clicks = [((r.get("click") or {}).get("statistic")) for r in ok if (r.get("click") or {}).get("statistic") is not None]
    blends = [r["rhythm"] for r in ok if r["kind"] == "blend4"]
    dmed = [(b.get("decks") or {}).get("downbeat_median_abs_ms") for b in blends]
    bmed = [(b.get("decks") or {}).get("beat_median_abs_ms") for b in blends]
    xmed = [(b.get("xcorr") or {}).get("median_abs_lag_ms") for b in blends if (b.get("xcorr") or {}).get("status") != "unreliable"]

    def stat(key, field):
        return [((r.get(key) or {}).get(field)) for r in ok if (r.get(key) or {}).get(field) is not None]

    doc = {"schema": "musicmix.qa_noise_calibration.1", "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n_points": len(rows), "n_ok": len(ok), "excluded_slugs": sorted(excl), "n_excluded_rows": len(all_ok) - len(ok),
           "overlap_lines": {"n": sum(1 for r in ok if (r.get("rhythm") or {}).get("overlap_lines")), "n_reliable": len(ols),
                             "abs_offset_ms": {"p50": _q(ols, 50), "p95": _q(ols, 95), "max": _q(ols, 100)}},
           "proposed": {"entry_phase_fail_ms": thr, "click_ratio": None if not clicks else max(4.0, round(1.5 * max(clicks), 1)),
                        "click_warn_ratio": None if not clicks else round(max(clicks), 1)}, "errors": [r.get("error") for r in rows if r.get("error")][:10],
           "tracks": sorted({r["slug"] for r in rows}), "excluded_gym_slugs": rows[0].get("excluded_gym_slugs") if rows else None,
           "by_kind": {k: sum(1 for r in ok if r["kind"] == k) for k in sorted({r["kind"] for r in ok})},
           "rules": RULES,
           "entry_phase": {"n": len(entry), "n_reliable": len(rel), "reliable_rate": len(rel) / len(entry) if entry else None,
                           "abs_offset_ms_reliable": {"p50": _q(offs, 50), "p90": _q(offs, 90), "p95": p95, "max": _q(offs, 100)},
                           "abs_offset_ms_all": {"p50": _q(all_offs, 50), "p95": _q(all_offs, 95), "max": _q(all_offs, 100)},
                           "quarter_beat_ms_min": 250.0 * min(periods) if periods else None,
                           "quarter_beat_ms_max": 250.0 * max(periods) if periods else None,
                           "proposed_fail_ms": thr,
                           "unreliable_reasons": _count(p for e in entry if not e.get("reliable") for p in (e.get("reliability_problems") or ["?"]))},
           "overlap_rhythm": {"n": len(blends), "downbeat_median_abs_ms": {"p50": _q(dmed, 50), "p95": _q(dmed, 95), "max": _q(dmed, 100)},
                              "beat_median_abs_ms": {"p50": _q(bmed, 50), "p95": _q(bmed, 95), "max": _q(bmed, 100)},
                              "xcorr_median_abs_lag_ms": {"p50": _q(xmed, 50), "p95": _q(xmed, 95), "max": _q(xmed, 100)}},
           "dropout_longest_s": {"p50": _q(stat("dropout", "longest_s"), 50), "p99": _q(stat("dropout", "longest_s"), 99),
                                 "max": _q(stat("dropout", "longest_s"), 100)},
           "dropout_longest_beats": {"p99": _q(stat("dropout", "longest_beats"), 99), "max": _q(stat("dropout", "longest_beats"), 100)},
           "dropout_effect_region_beats": {"p99": _q(stat("dropout", "effect_region_longest_beats"), 99),
                                           "max": _q(stat("dropout", "effect_region_longest_beats"), 100)},
           "click_stat": {"p50": _q(stat("click", "statistic"), 50), "p99": _q(stat("click", "statistic"), 99), "max": _q(stat("click", "statistic"), 100)},
           "echo_stat": {"p50": _q(stat("echo_tempo", "statistic"), 50), "p99": _q(stat("echo_tempo", "statistic"), 99),
                         "max": _q(stat("echo_tempo", "statistic"), 100)},
           "verdicts": _count(r["verdict"] for r in ok),
           "fail_reasons": _count(x.split(":")[0] for r in ok for x in r.get("reasons", []) if x),
           "status_by_check": {c: _count((r.get(c) or {}).get("status") for r in ok) for c in ("dropout", "click", "echo_tempo", "phrase", "double_bass")},
           "rhythm_status": _count((r.get("rhythm") or {}).get("status") for r in ok)}
    T.dump(SUMMARY, doc)
    print(json.dumps({k: doc[k] for k in ("n_ok", "by_kind", "entry_phase", "overlap_rhythm", "dropout_longest_s", "dropout_longest_beats",
                                          "click_stat", "echo_stat", "rhythm_status", "status_by_check")}, indent=1))
    return doc


def _count(it) -> dict:
    c: dict = {}
    for v in it:
        c[str(v)] = c.get(str(v), 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stages", nargs="+", choices=["run", "summary"])
    ap.add_argument("--tracks", nargs="*")
    ap.add_argument("--fresh", action="store_true", help="ricomincia (cancella calibration_points.jsonl)")
    args = ap.parse_args(argv)
    if args.fresh and POINTS.exists():
        POINTS.unlink()
    for st in args.stages:
        run(args) if st == "run" else summary(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
