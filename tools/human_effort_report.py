#!/usr/bin/env python3
"""Misura dello sforzo umano: quante azioni umane richiede una run di set o un lotto di brani nuovi.

Requisito dell'utente (AGENTS.md, "Autonomia", 21/10/2026): nuovi brani e nuovi set non devono richiedere il voto
dell'utente per funzionare; va misurato quanto intervento umano richiedono. Questo strumento legge le run esistenti
(sola lettura, JSON con ``musicmix.paths.localize``) e conta le azioni umane **richieste** o **già compiute**, per
categoria, poi le normalizza a "100 brani nuovi" e "10 set nuovi". Le categorie sono definizioni nostre [H]:

Set (``set --run runs/dj_set_v3_bridge``):

* ``votes_requested``: clip proposte all'ascolto nella galleria della run (``gallery/review_manifest.json``);
* ``votes_given`` / ``votes_with_notes``: giudizi salvati (``giudizi_cloud/<run>/judgments``), di cui con nota o tag
  (correzioni verbali); ``notes_by_tag``: tag delle correzioni (es. ``punto_frase_sbagliato``);
* ``structured_corrections`` e ``needs_manual``: correzioni strutturate dei giudizi e voci ``needs_manual``/``manuale``
  del piano di correzione (``corrections_plan.json``, se esiste in una run ``<run>_corr*``);
* ``transitions_to_review``: transizioni che la macchina non chiude da sola: QA ``fail`` al ricontrollo più recente
  (``qa_v5_recheck.json`` se c'è, altrimenti il verdetto salvato), tenute nonostante un veto (``kept_despite_veto``) o
  senza tentativo accettato; ``transitions_warn``: QA ``warn`` (da campionare, non obbligatorie);
* ``manual_track_decisions``: brani del set esclusi o sospesi a mano (lista di esclusione, ``EXCLUDE`` del runner)
  e brani con griglia a confidenza ``low`` usati nel set (controllo manuale della griglia).

Lotto di brani (``tracks --run runs/pool_bridge_v1``):

* ``download_failed`` / ``download_pending``: richieste fallite o ancora in attesa sul PC (servono scelta di
  un'alternativa o un nuovo tentativo dell'utente);
* ``duration_mismatch``: durata scaricata diversa dall'attesa di oltre il 5 % [H] (versione da verificare);
* ``language_review``: lingua sconosciuta o incerta per ``musicmix.track_filter`` (da verificare);
* ``excluded_by_user``: brani nella lista di esclusione (decisione umana già presa);
* ``grid_review``: griglia a confidenza ``low`` o tempo ambiguo d'ottava escluso a mano;
* ``automatic_degradations`` (non umane, riportate a parte): ASR non affidabile (dj_phrase usa i ripieghi), ottava
  ambigua risolta dai provider.

Stima: azioni per 100 brani = azioni del lotto / brani richiesti x 100; per 10 set = azioni della run x 10. Le stime
assumono gli stessi tassi: sono un ordine di grandezza, non una previsione.

    python tools/human_effort_report.py set --run runs/dj_set_v3_bridge
    python tools/human_effort_report.py tracks --run runs/pool_bridge_v1
    python tools/human_effort_report.py all   # le due run esistenti, scrive runs/human_effort_v1/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicmix.paths import localize  # noqa: E402

OUT = ROOT / "runs" / "human_effort_v1"
DURATION_TOL = 0.05  # [H]


def load(p: Path) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _resolve(p: str | Path, base: Path = ROOT) -> Path:
    q = localize(str(p))
    return q if q.is_absolute() else base / q


# ------------------------------------------------------------------------------------------- set
def set_effort(run: Path, judgments: Optional[Path] = None) -> dict:
    run = Path(run)
    name = run.name
    doc = load(run / "set.json")
    trs = doc.get("transitions") or []
    out: dict[str, Any] = {"run": _rel(run), "kind": "set", "n_transitions": len(trs), "n_tracks": len(doc.get("tracklist") or []) or (len(trs) + 1)}
    # voti chiesti: clip della galleria della run (e di eventuali pagine d'ascolto registrate)
    req = 0
    man = run / "gallery" / "review_manifest.json"
    if man.is_file():
        m = load(man)
        req = sum(len(g.get("items") or []) for g in m.get("groups") or [])
    out["votes_requested"] = req
    jdir = Path(judgments) if judgments else ROOT / "giudizi_cloud" / name / "judgments"
    js = [load(f) for f in sorted(jdir.glob("*.json"))] if jdir.is_dir() else []
    scored = [j for j in js if j.get("score") is not None]
    notes = [j for j in js if (j.get("note") or "").strip() or j.get("tags")]
    tags: dict[str, int] = {}
    for j in js:
        for t in j.get("tags") or []:
            tags[t] = tags.get(t, 0) + 1
    out.update(judgments_dir=_rel(jdir), votes_given=len(scored), judgments_saved=len(js), votes_with_notes=len(notes),
               notes_by_tag=tags, structured_corrections=sum(1 for j in js if j.get("corrections")))
    # piano di correzione (run sorelle <run>_corr*)
    nm, cp_files = 0, sorted(run.parent.glob(f"{name}_corr*/corrections_plan.json"))
    statuses: dict[str, int] = {}
    for f in cp_files:
        cp = load(f)
        for r in cp.get("transitions") or cp.get("records") or []:
            nm += len(r.get("needs_manual") or [])
            statuses[r.get("status", "?")] = statuses.get(r.get("status", "?"), 0) + 1
    out.update(corrections_plans=[_rel(f) for f in cp_files], needs_manual=nm, corrections_status=statuses)
    # transizioni da rivedere
    recheck = run / "qa_v5_recheck.json"
    latest = {}
    src = "set.json (verdetto salvato)"
    if recheck.is_file():
        rc = load(recheck)
        latest = {r["index"]: (r.get("qa_new") or {}).get("verdict") for r in rc.get("transitions") or [] if "error" not in r}
        src = f"{_rel(recheck)} (QA corrente sullo stesso PCM)"
    review, warn = [], []
    for t in trs:
        v = latest.get(t["index"], (t.get("qa") or {}).get("verdict"))
        why = []
        if v == "fail":
            why.append("QA fail")
        if t.get("kept_despite_veto"):
            why.append("tenuta nonostante il veto")
        if not t.get("clip_sha256") and not t.get("clip_wav"):
            why.append("nessun tentativo accettato")
        if why:
            review.append({"index": t["index"], "a": t["a"], "b": t["b"], "technique": t.get("technique"), "why": why})
        elif v == "warn":
            warn.append(t["index"])
    out.update(qa_source=src, transitions_to_review=review, n_transitions_to_review=len(review), transitions_warn=warn)
    # decisioni manuali sui brani
    manual = []
    try:
        excl = load(ROOT / "configs" / "excluded_tracks.json").get("tracks") or {}
    except FileNotFoundError:
        excl = {}
    skipped = []

    def _walk(o):
        if isinstance(o, dict):
            if isinstance(o.get("skipped"), list):
                skipped.extend(x for x in o["skipped"] if isinstance(x, dict) and x.get("slug"))
            for v in o.values():
                _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)

    _walk(doc.get("log"))
    seen_s: set = set()
    skipped = [x for x in skipped if not (x["slug"] in seen_s or seen_s.add(x["slug"]))]
    for e in skipped:
        slug, why = e.get("slug"), str(e.get("why") or e.get("reason") or "")
        low = why.lower()
        if slug in excl or any(w in low for w in ("all'ascolto", "verificat", "utente", "a mano", "manual")):
            manual.append({"slug": slug, "why": why, "kind": "esclusione in attesa di verifica umana"})
        elif any(w in low for w in ("assent", "missing")):
            manual.append({"slug": slug, "why": why, "kind": "dati mancanti (rifare l'analisi: automatico se la pipeline gira)",
                           "human": False})
    manual_h = [m for m in manual if m.get("human", True)]
    used = {x for t in trs for x in (t["a"], t["b"])}
    grid_low = []
    plan_dir = run / "dj_plan" / "tracks"
    for s in sorted(used):
        p = plan_dir / f"{s}.json"
        if p.is_file():
            tempo = load(p).get("tempo") or {}
            if tempo.get("confidence") == "low":
                grid_low.append(s)
    out.update(manual_track_decisions=manual, grid_low_tracks_used=grid_low)
    actions = {
        "votes_requested": req,
        "notes_and_corrections": len(notes),
        "needs_manual": nm,
        "transitions_to_review": len(review),
        "manual_track_decisions": len(manual_h),
        "grid_checks": len(grid_low),
    }
    out["actions"] = actions
    out["actions_total"] = sum(actions.values())
    out["per_10_sets"] = {k: 10 * v for k, v in actions.items()} | {"total": 10 * out["actions_total"]}
    out["autonomy_rule"] = ("AGENTS.md: la verifica umana è periodica e su un campione, mai a ogni set: votes_requested e "
                            "transitions_to_review sono l'intervento da portare a ~0 per set (le altre voci sono una tantum)")
    return out


# ------------------------------------------------------------------------------------------- tracks
def tracks_effort(run: Path, plan_dirs: Optional[list[Path]] = None, phrase_dir: Optional[Path] = None) -> dict:
    run = Path(run)
    st = load(run / "state.json")
    issues = st.get("issues") or {}
    out: dict[str, Any] = {"run": _rel(run), "kind": "tracks", "n_requested": len(issues)}
    failed = [{"issue": k, "slug": v.get("slug"), "reasons": v.get("reasons")} for k, v in issues.items() if v.get("status") == "failed"]
    pending = [{"issue": k, "slug": v.get("slug")} for k, v in issues.items() if v.get("status") in ("pending", "requested", "waiting")]
    analysed = [v.get("slug") for v in issues.values() if v.get("status") == "analysed"]
    mism = []
    for k, v in issues.items():
        pr = v.get("probe") or {}
        if pr.get("expected_seconds") and pr.get("duration"):
            d = abs(float(pr["duration"]) - float(pr["expected_seconds"])) / float(pr["expected_seconds"])
            if d > DURATION_TOL:
                mism.append({"issue": k, "slug": v.get("slug"), "duration_s": pr["duration"], "expected_s": pr["expected_seconds"],
                             "rel_diff": round(d, 3)})
    cat = load(run / "catalog_ready.json")
    slugs = [t["slug"] for t in cat.get("tracks") or []]
    from musicmix import track_filter as TF

    tf = TF.default_filter()
    lang_review, excluded = [], []
    for t in cat.get("tracks") or []:
        vdoc = None
        try:
            vdoc = load(_resolve(t["vocals_path"]))
        except Exception:  # noqa: BLE001 - JSON vocals mancante: la lingua resta sconosciuta
            pass
        chk = tf.check(t["slug"], t, vdoc)
        if chk.get("rule") == "excluded_list":
            excluded.append({"slug": t["slug"], "why": chk.get("why")})
        elif chk.get("rule") in ("language_unknown", "language_uncertain"):
            lang_review.append({"slug": t["slug"], "rule": chk["rule"], "why": chk.get("why")})
        elif chk.get("rule") == "language_not_allowed":
            excluded.append({"slug": t["slug"], "why": chk.get("why"), "automatic": True})
    plan_dirs = plan_dirs or [ROOT / "runs/dj_plan_bridge_v1/tracks", ROOT / "runs/dj_set_v3_bridge/dj_plan/tracks"]
    grid_low, octave_auto = [], []
    try:  # esclusioni a mano del runner del set v3 (es. tempo ambiguo d'ottava da verificare all'ascolto)
        sys.path.insert(0, str(ROOT / "tools"))
        import run_dj_set_v3_bridge as _R

        manual_excl = dict(_R.EXCLUDE)
    except Exception:  # noqa: BLE001
        manual_excl = {}
    for s in slugs:
        if s in manual_excl:
            grid_low.append(s)
            continue
        p = next((d / f"{s}.json" for d in plan_dirs if (d / f"{s}.json").is_file()), None)
        if p is None:
            continue
        tempo = load(p).get("tempo") or {}
        if tempo.get("confidence") == "low":
            grid_low.append(s)
        elif tempo.get("octave_ambiguous"):
            octave_auto.append(s)
    asr_unrel = []
    phrase_dir = phrase_dir or ROOT / "runs/dj_set_v3_bridge/phrase"
    for s in slugs:
        p = phrase_dir / f"{s}.json"
        if p.is_file():
            q = (load(p).get("model") or {}).get("asr_quality") or {}
            if q and not q.get("reliable", False):
                asr_unrel.append(s)
    actions = {"download_failed": len(failed), "download_pending": len(pending), "duration_mismatch": len(mism),
               "language_review": len(lang_review), "grid_review": len(grid_low)}
    n = max(1, len(issues))
    out.update(analysed=len(analysed), catalog_tracks=len(slugs), download_failed=failed, download_pending=pending,
               duration_mismatch=mism, language_review=lang_review, excluded=excluded, grid_low=grid_low,
               automatic_degradations={"asr_unreliable": asr_unrel, "octave_ambiguous_resolved": octave_auto},
               actions=actions, actions_total=sum(actions.values()),
               per_100_tracks={k: round(100.0 * v / n, 1) for k, v in actions.items()} | {"total": round(100.0 * sum(actions.values()) / n, 1)},
               note="esclusioni dell'utente già prese non sono contate come azioni future; i brani richiesti includono quelli "
                    "non arrivati (tasso di richiesta, non di catalogo)")
    return out


# ------------------------------------------------------------------------------------------- report
def to_md(res: dict) -> str:
    L = [f"# Sforzo umano: {res['run']}", "", f"Generato {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} da "
         "`tools/human_effort_report.py`. Categorie [H] nel docstring dello strumento.", ""]
    L += ["| azione | conteggio | stima |", "|---|---|---|"]
    scale = res.get("per_10_sets") or res.get("per_100_tracks") or {}
    lab = "per 10 set" if res["kind"] == "set" else "per 100 brani"
    for k, v in res["actions"].items():
        L.append(f"| {k} | {v} | {scale.get(k)} {lab} |")
    L.append(f"| **totale** | **{res['actions_total']}** | **{scale.get('total')} {lab}** |")
    L.append("")
    if res["kind"] == "set":
        L += [f"Transizioni: {res['n_transitions']}; voti dati {res['votes_given']} su {res['votes_requested']} chiesti; "
              f"giudizi con nota/tag {res['votes_with_notes']} (tag: {res['notes_by_tag']}); QA da {res['qa_source']}.", ""]
        for r in res["transitions_to_review"]:
            L.append(f"- T{r['index']:02d} {r['a']} → {r['b']} ({r['technique']}): {', '.join(r['why'])}")
    else:
        L += [f"Richieste {res['n_requested']}, analizzate {res['analysed']}, catalogo {res['catalog_tracks']}.", ""]
        for k in ("download_failed", "duration_mismatch", "language_review", "grid_low"):
            L.append(f"- {k}: {', '.join(str(x.get('slug') if isinstance(x, dict) else x) for x in res[k]) or '—'}")
        L.append(f"- degradazioni automatiche (non umane): {res['automatic_degradations']}")
    L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["set", "tracks", "all"])
    ap.add_argument("--run", help="cartella della run")
    ap.add_argument("--judgments", help="cartella dei giudizi (default giudizi_cloud/<run>/judgments)")
    ap.add_argument("--out", default=str(OUT), help="cartella di uscita (mai la run letta)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.mode == "all":
        jobs = [("set", ROOT / "runs/dj_set_v3_bridge"), ("tracks", ROOT / "runs/pool_bridge_v1")]
    else:
        jobs = [(args.mode, Path(args.run))]
    summary = {}
    for mode, run in jobs:
        res = set_effort(run, args.judgments) if mode == "set" else tracks_effort(run)
        name = Path(run).name
        (out / f"{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        (out / f"{name}.md").write_text(to_md(res), encoding="utf-8")
        summary[name] = {"actions": res["actions"], "total": res["actions_total"],
                         "scaled": res.get("per_10_sets") or res.get("per_100_tracks")}
        print(name, json.dumps(summary[name], ensure_ascii=False))
    if args.mode == "all":
        (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
