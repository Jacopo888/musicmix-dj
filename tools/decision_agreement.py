#!/usr/bin/env python3
"""Metrica di accordo: quante decisioni del sistema l'utente approva senza correzioni, e di quanto sposta le altre.

Per ogni set (``--case SET_JSON JUDGMENTS_DIR``, ripetibile) legge i giudizi della pagina di approvazione del set
(``musicmix.set-approval.1``: un doc ``<run>__set_tNN`` per transizione, :mod:`listening_pages`) e li traduce con
:mod:`apply_corrections` (stesse regole delle correzioni). Quattro decisioni per transizione:

* **ordine** (il brano B scelto per quel momento): approvata se il verdetto non è "da rifare con altro brano" e
  "brano B sbagliato" non è segnato;
* **uscita** (punto di A): approvata se il cue corretto di A coincide con quello del sistema; altrimenti lo spostamento
  in battute sulla griglia di A (``fine_frase`` e "resta su A" tradotti in battute dalle regole [H] del piano);
* **ingresso** (punto di B): approvato se cue di B e lead-in coincidono; altrimenti lo spostamento in battute (pickup
  e sezione inclusi);
* **tecnica**: approvata se non è chiesta un'altra tecnica.

Le transizioni "da rifare con altro brano" contano solo per l'ordine; quelle "da correggere" senza campi compilati
(``manuale``) contano solo per l'ordine e come transizione non approvata per intero. Un campo non compilato in una
transizione "da correggere" conta come approvazione implicita (riportata a parte da quella esplicita).

Intervalli: bootstrap percentile al 95 % ricampionando le transizioni (a grappoli per set quando i set sono più di
uno), seme fisso. Con poche transizioni gli intervalli sono larghi e il report lo dice; nessuna soglia è stimata qui.
Le pagine non cieche sono contate a parte (``blind``).

    python tools/decision_agreement.py \\
        --case runs/dj_set_v3_bridge/set.json giudizi_cloud/dj_set_v3_bridge/approvazione --out runs/.../agreement.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import apply_corrections as AC  # noqa: E402
import listening_corrections as LC  # noqa: E402

SCHEMA = "musicmix.decision-agreement.1"
DECISIONS = ("order", "exit", "entry", "technique")
LABELS = {"order": "ordine (brano B)", "exit": "uscita di A", "entry": "ingresso di B", "technique": "tecnica",
          "all": "transizione approvata per intero"}
MIN_N = 10


def rows_for_case(set_path: Path, judgments: list[Path], run: Optional[str] = None) -> tuple[list[dict], dict]:
    """Una riga per transizione giudicata nella pagina di approvazione (le altre fonti sono escluse)."""
    ctx = AC.Context(set_path)
    docs = [d for d in LC.read_docs(judgments) if d.get("type") in ("set_approval_transizione", "set_approval")]
    entries, ignored, set_doc = AC.collect_entries(ctx, docs, run)
    rows = []
    for i in sorted(ctx.transitions):
        items = [e for e in entries.get(i, []) if e["source"] == "set_approval"]
        if not items:
            continue
        e, _ = AC.choose_entry(items)
        r = AC.resolve(ctx, i, e)
        d = r["decisions"] or {}
        row: dict[str, Any] = {"set": ctx.run, "index": i, "a": r["a"], "b": r["b"], "status": r["status"],
                               "verdict": r["verdict"], "blind": bool(e["doc"].get("blind"))}
        row["order"] = bool((d.get("order") or {}).get("approved"))
        moot = r["status"] in ("ripianifica", "manuale")
        for k in ("exit", "entry", "technique"):
            x = d.get(k) or {}
            row[k] = None if moot or x.get("approved") is None else bool(x["approved"])
            row[k + "_explicit"] = bool(x.get("explicit"))
        row["exit_shift_bars"] = None if moot else (d.get("exit") or {}).get("shift_bars")
        row["entry_shift_bars"] = None if moot else (d.get("entry") or {}).get("shift_bars")
        row["entry_pickup"] = bool((d.get("entry") or {}).get("pickup"))
        row["technique_to"] = (d.get("technique") or {}).get("to")
        row["all"] = bool(row["order"] and not moot and row["exit"] and row["entry"] and row["technique"])
        rows.append(row)
    meta = {"set": AC.rel(Path(set_path)), "set_sha256": ctx.sha, "run": ctx.run, "n_transitions": len(ctx.transitions),
            "n_judged": len(rows), "ignored_docs": len(ignored),
            "set_judgment": {k: set_doc.get(k) for k in ("set_verdict", "flow", "energy", "variety", "repetitiveness")}
            if set_doc else None}
    return rows, meta


def _rate(vals: list) -> Optional[float]:
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def _mean_abs(vals: list) -> Optional[float]:
    v = [abs(x) for x in vals if x not in (None, 0)]
    return float(np.mean(v)) if v else None


def bootstrap(rows: list[dict], stat, n_boot: int = 2000, seed: int = 20261021) -> Optional[list[float]]:
    """IC 95 % percentile; ricampiona i set (grappoli) se più di uno, altrimenti le transizioni."""
    if not rows:
        return None
    rng = np.random.default_rng(seed)
    sets = sorted({r["set"] for r in rows})
    groups = [[r for r in rows if r["set"] == s] for s in sets] if len(sets) > 1 else [[r] for r in rows]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        sample = [r for g in pick for r in groups[g]]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    if len(vals) < n_boot // 2:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def summarize(rows: list[dict], n_boot: int = 2000) -> dict:
    out: dict[str, Any] = {"n_transitions": len(rows), "n_sets": len({r["set"] for r in rows})}
    for k in DECISIONS + ("all",):
        vals = [r[k] for r in rows]
        n = sum(v is not None for v in vals)
        out[k] = {"label": LABELS[k], "n": n, "approved": sum(1 for v in vals if v), "rate": _rate(vals),
                  "ci95": bootstrap(rows, lambda s, k=k: _rate([r[k] for r in s]), n_boot)}
        if k in ("exit", "entry", "technique"):
            out[k]["explicit_approvals"] = sum(1 for r in rows if r[k] and r[k + "_explicit"])
    for k in ("exit", "entry"):
        sh = [r[f"{k}_shift_bars"] for r in rows if r[k] is False]
        nz = [x for x in sh if x not in (None, 0)]
        out[k]["corrected_shift_bars"] = {
            "n": len(nz), "values": nz, "mean_abs": _mean_abs(nz), "median": float(np.median(nz)) if nz else None,
            "earlier": sum(1 for x in nz if x < 0), "later": sum(1 for x in nz if x > 0),
            "ci95_mean_abs": bootstrap([r for r in rows if r[k] is False and r[f"{k}_shift_bars"] not in (None, 0)],
                                       lambda s, k=k: _mean_abs([r[f"{k}_shift_bars"] for r in s]), n_boot)}
    out["entry"]["pickup_requested"] = sum(1 for r in rows if r["entry_pickup"])
    out["technique"]["requested"] = {}
    for r in rows:
        if r["technique_to"]:
            out["technique"]["requested"][r["technique_to"]] = out["technique"]["requested"].get(r["technique_to"], 0) + 1
    out["status_counts"] = {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})}
    out["blind_share"] = _rate([r["blind"] for r in rows])
    if len(rows) < MIN_N:
        out["warning"] = f"solo {len(rows)} transizioni giudicate (< {MIN_N}): intervalli larghi, nessuna conclusione"
    return out


def agreement(cases: list[tuple[Path, list[Path]]], run: Optional[str] = None, n_boot: int = 2000) -> dict:
    rows, metas = [], []
    for set_path, judgments in cases:
        r, m = rows_for_case(set_path, judgments, run)
        rows += r
        metas.append(m)
    return {"schema_version": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "tools/decision_agreement.py", "cases": metas, "summary": summarize(rows, n_boot),
            "rows": rows,
            "notes": ["accordo = decisione del sistema approvata dall'utente all'ascolto (pagina di approvazione del set)",
                      "campo non compilato in 'da correggere' = approvazione implicita (vedi explicit_approvals)",
                      "spostamenti in battute sulla griglia del brano (A per l'uscita, B per l'ingresso); fine frase, resta su A, "
                      "sezione e pickup tradotti con le regole [H] di tools/apply_corrections.py",
                      "IC bootstrap percentile 95 % (grappoli per set se più set); nessuna soglia stimata"]}


def _fmt(x: Optional[float], pct: bool = True) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.0f}%" if pct else f"{x:.2f}"


def report(res: dict) -> str:
    s = res["summary"]
    lines = [f"Accordo sulle decisioni: {s['n_transitions']} transizioni giudicate in {s['n_sets']} set"]
    for k in DECISIONS + ("all",):
        d = s[k]
        ci = d["ci95"]
        lines.append(f"  {d['label']}: {d['approved']}/{d['n']} = {_fmt(d['rate'])}"
                     + (f" (IC95 {_fmt(ci[0])}–{_fmt(ci[1])})" if ci else ""))
    for k in ("exit", "entry"):
        c = s[k]["corrected_shift_bars"]
        if c["n"]:
            ci = c["ci95_mean_abs"]
            lines.append(f"  {LABELS[k]}: {c['n']} spostamenti, |Δ| medio {_fmt(c['mean_abs'], False)} battute"
                         + (f" (IC95 {ci[0]:.2f}–{ci[1]:.2f})" if ci else "") + f", prima {c['earlier']}, dopo {c['later']}")
    if s.get("warning"):
        lines.append("  ATTENZIONE: " + s["warning"])
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", nargs="+", action="append", required=True, metavar="PATH",
                    help="SET_JSON seguito da una o più cartelle/file di giudizi (ripetibile per più set)")
    ap.add_argument("--run", help="nome della run nei giudizi (default: run di ciascun set)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    cases = []
    for c in a.case:
        if len(c) < 2:
            ap.error("--case vuole SET_JSON e almeno una cartella di giudizi")
        cases.append((Path(c[0]), [Path(x) for x in c[1:]]))
    res = agreement(cases, a.run, a.n_boot)
    print(report(res))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
