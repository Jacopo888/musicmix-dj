#!/usr/bin/env python3
"""Set DJ v4: catalogo en/it del ponte con ``reminder``/``too_many_nights`` (voci rifatte), ASR italiano per 18K, ingresso di B
corretto, ripianificazione senza ripieghi, schede + affidabilità + novità + tensione (``musicmix.dj_tension``).

Strato sottile sopra ``tools/run_dj_set_v3_bridge.py`` (stessi hook: profilo libri, dj_phrase, policy, guardia e lead-in di B,
QA ``dj_qa.6``, veto tecnico del critico, ripianificazione), con gli ingressi in una run nuova ``runs/dj_set_v4`` (le run
congelate non si toccano):

* ``prepare``: ``catalog_frozen.json`` = copia di ``runs/pool_bridge_v1/catalog_ready.json`` in cui ``vocals_path`` punta al JSON
  vocals che si deve usare (override di ``runs/pool_bridge_v1/vocals_overrides.json`` per ``reminder``/``too_many_nights``,
  ASR nella lingua rilevata di ``runs/asr_language_v1`` per ``18k_no_kizzy``; il valore del catalogo resta in
  ``vocals_path_catalog``); analisi DJ e modelli ``dj_phrase``: copiati dalla run v3 congelata quando gli ingressi sono identici
  (stesso JSON vocals e stesso audio), altrimenti rifatti (``run_dj_set_v3_bridge.prepare``).
* ``set``: ``run_dj_set_v2_books.build_set`` con gli hook del v3 e ``--reliability prefer --cards track_cards --novelty
  runs/dj_set_v3_bridge/set.json --tension prefer --event-plan`` (default di questo runner).
* ``export``: ``deliverables/set_dj_v4.mp3`` e la tracklist (con carica, risoluzione, continuità e tecnica per transizione).

    python tools/run_dj_set_v4.py prepare set export
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import run_dj_set_v3_bridge as v3  # noqa: E402

from musicmix.paths import localize  # noqa: E402
from musicmix.stems import vocals_json_for  # noqa: E402

v2, base = v3.v2, v3.base
RUN = "dj_set_v4"
OUT = ROOT / "runs" / RUN
V3 = ROOT / "runs" / "dj_set_v3_bridge"
CAT_SRC = ROOT / "runs" / "pool_bridge_v1" / "catalog_ready.json"
DELIVER = ROOT / "deliverables"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rebind() -> None:
    """Gli hook del v3 leggono i percorsi dai globali del modulo: qui puntano alla run v4."""
    v3.RUN, v3.SCRIPT, v3.OUT = RUN, "tools/run_dj_set_v4.py", OUT
    v3.FROZEN, v3.PLAN, v3.PHRASE = OUT / "catalog_frozen.json", OUT / "dj_plan", OUT / "phrase"
    v3.CACHE = ROOT / ".cache" / RUN
    v2.FORM_CACHE = ROOT / ".cache" / RUN / "form"


def vocals_choice(entry: dict) -> tuple[Path, str]:
    redo = v3.ASR_LANG_DIR / entry["slug"] / "a.json"
    if redo.is_file():
        return redo, "ASR rifatto nella lingua rilevata (runs/asr_language_v1)"
    vj = vocals_json_for(entry)
    if Path(vj) != localize(entry["vocals_path"]):
        return Path(vj), "override di runs/pool_bridge_v1/vocals_overrides.json (fase vocals rifatta)"
    return localize(entry["vocals_path"]), "catalogo"


def prepare() -> dict:
    rebind()
    OUT.mkdir(parents=True, exist_ok=True)
    cat = json.loads(CAT_SRC.read_text(encoding="utf-8"))
    v3cat = {t["slug"]: t for t in json.loads((V3 / "catalog_frozen.json").read_text(encoding="utf-8"))["tracks"]}
    changed, copied = [], []
    for t in cat["tracks"]:
        vj, why = vocals_choice(t)
        t["vocals_path_catalog"] = t["vocals_path"]
        t["vocals_path"] = str(Path(vj).resolve())  # assoluto, come nel catalogo del ponte
        t["vocals_path_source"] = why
        t["vocals_json_sha256"] = sha(vj)
    doc = dict(cat, frozen_from=str(CAT_SRC.relative_to(ROOT)), frozen_from_sha256=sha(CAT_SRC),
               frozen_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               note="copia per la run v4; vocals_path risolto (override / ASR nella lingua del brano), valore del catalogo in vocals_path_catalog")
    if not v3.FROZEN.is_file():
        v3.FROZEN.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    cat = json.loads(v3.FROZEN.read_text(encoding="utf-8"))
    (v3.PLAN / "tracks").mkdir(parents=True, exist_ok=True)
    v3.PHRASE.mkdir(parents=True, exist_ok=True)
    (v3.CACHE / "act").mkdir(parents=True, exist_ok=True)
    for t in cat["tracks"]:
        s = t["slug"]
        old = v3cat.get(s)
        same = (old is not None and old.get("track_id") == t["track_id"]
                and localize(old["vocals_path"]) == localize(t["vocals_path"]) and not (v3.ASR_LANG_DIR / s / "a.json").is_file())
        src_t, src_p, src_a = V3 / "dj_plan" / "tracks" / f"{s}.json", V3 / "phrase" / f"{s}.json", ROOT / ".cache/dj_set_v3_bridge/act" / f"{s}.npz"
        if same and src_t.is_file() and src_p.is_file() and src_a.is_file():
            for a_, b_ in ((src_t, v3.PLAN / "tracks" / f"{s}.json"), (src_p, v3.PHRASE / f"{s}.json"), (src_a, v3.CACHE / "act" / f"{s}.npz")):
                if not b_.is_file():
                    shutil.copy2(a_, b_)
            copied.append(s)
        else:
            changed.append(s)
    print(f"copiati dalla run v3 (ingressi identici): {copied}\nda rifare: {changed}", flush=True)
    res = v3.prepare()   # rifà analisi DJ e modelli dj_phrase mancanti (stesse funzioni del v3)
    res["copied_from_v3"] = copied
    res["recomputed"] = changed
    v3.dump(OUT / "prepare.json", res)
    return res


def run_set(overwrite: bool, reliability: str, novelty: list, tension: str, plan_only: Path | None = None) -> None:
    rebind()
    v3.RELIABILITY.update(mode=reliability, cards=ROOT / "track_cards", novelty=[str(ROOT / n) for n in novelty], params={})
    v3.TENSION["mode"] = tension
    v3.EVENT_PLAN.update(on=True, pref=False)
    hooks = v2.SetHooks(
        run=RUN, script="tools/run_dj_set_v4.py",
        title="set DJ v4 (libri, dj_phrase, ingresso di B, schede/affidabilità/novità, tensione)",
        pool=v3.sources_v3, info=v3.info_v3, filter_pair=v3.filter_pair_v3, attempts=v3.attempts_v3, render=v3.render_v3,
        run_qa=v3.qa_v3, veto=v3._track_tried(v3.veto_v3), lead_a_s=v3.lead_a_v3, overlap_b_s=v3.overlap_b_v3,
        on_accept=v3.on_accept_v3, finalize_audio=v3.finalize_v3, mp3_bitrate="192k", order_overrides=dict(v3.ORDER_OVERRIDES),
        compare_v1b=False,
        notes=("integrità tecnica (QA dj_qa.6, veto del critico, palestra) non è qualità musicale: il set non è stato ascoltato",
               "profilo 'books', dj_phrase, guardia e lead-in dell'ingresso di B, ripianificazione senza ripieghi",
               f"affidabilità '{reliability}' dalle schede brano, novità rispetto a {novelty}",
               f"dj_tension '{tension}': carica di A, risoluzione di B, base continua, scontro tonale, testo affine (pesi [H] "
               "pre-registrati in runs/dj_tension_v1/PREREGISTRATION.json)",
               "catalogo: runs/dj_set_v4/catalog_frozen.json (en/it; voci rifatte per reminder/too_many_nights, ASR italiano per 18K)"),
        cue_overrides={}, reliability=v3.reliability_hook(), tension=v3.tension_hook(),
        event_fn=v3.event_fn_v3, after_cues=v3.after_cues_v3)
    out = Path(plan_only) if plan_only else OUT
    v3._CTX["out"] = out
    v2.build_set(out, overwrite=overwrite, hooks=hooks, no_render=bool(plan_only))


def export() -> dict:
    doc = json.loads((OUT / "set.json").read_text(encoding="utf-8"))
    if not doc.get("closed"):
        raise SystemExit("il set non si è chiuso: niente export")
    # versione del QA usata per accettare le transizioni (orchestratore 22/10: dj_qa.7 in preparazione da un altro agente)
    import importlib
    from musicmix import dj_qa
    importlib.reload(dj_qa)
    used = doc["versions"]["dj_qa"]
    doc["qa_used"] = {"dj_qa": used, "current_dj_qa": dj_qa.DJ_QA_VERSION, "needs_recheck": used != "musicmix.dj_qa.7",
                      "note": "transizioni accettate con QA " + used + " e veto tecnico del critico (v3 fast + blocco v4); la palestra "
                              "dei difetti (runs/defect_gym_v1/REPORT.md, gate FAIL) mostra limiti di questo QA: ricontrollare con dj_qa.7"}
    (OUT / "set.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    DELIVER.mkdir(exist_ok=True)
    shutil.copy2(OUT / "set.mp3", DELIVER / "set_dj_v4.mp3")
    lines = (OUT / "tracklist.txt").read_text(encoding="utf-8").rstrip("\n").split("\n")
    lines += ["", "Per transizione (dj_tension, pesi [H] pre-registrati; descrizione, non giudizio):"]
    for tr in doc["transitions"]:
        tz = tr.get("tension") or {}
        cm = (tz.get("continuity_measured") or {}).get("value")
        f = lambda x: "n/d" if x is None else f"{x:.2f}"  # noqa: E731
        lines.append(f"  T{tr['index']:02d} {tr['a']} -> {tr['b']}: {tr['label']}; carica A {f(tz.get('a_charge'))} "
                     f"({tz.get('a_charge_form')}), risoluzione B {f(tz.get('b_resolution'))}, continuità {f(tz.get('continuity'))} "
                     f"(misurata {f(cm)}), scontro tonale {f(tz.get('harmonic_clash'))}, testo {f(tz.get('lyric_affinity'))} "
                     f"[{tz.get('lyric_link')}], affidabilità {f((tr.get('reliability') or {}).get('value'))}, QA {tr['qa']['verdict']}")
    (DELIVER / "set_dj_v4_tracklist.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(OUT / "tracklist.csv", DELIVER / "set_dj_v4_tracklist.csv")
    print("\n".join(lines[-len(doc["transitions"]) - 1:]), flush=True)
    return doc


def gym(extra: list | None = None) -> None:
    """Palestra dei difetti (``tools/defect_gym.py``, non modificato) sulle transizioni accettate del set v4.

    Le ricette ``set_v3`` della palestra leggono ``set.json``, catalogo e analisi dai globali di ``run_dj_set_v3_bridge``:
    con :func:`rebind` puntano alla run v4. Riferimenti = transizioni del set (lead-in di B acceso, come nel set); varianti =
    difetti del gate (``configs/defect_gym_gate.json``); uscite in ``runs/dj_set_v4/defect_gym``."""
    rebind()
    import defect_gym as DG

    doc = json.loads((OUT / "set.json").read_text(encoding="utf-8"))
    gdir = OUT / "defect_gym"
    refs = [{"id": f"v4_t{tr['index']:02d}_{tr['type']}", "origin": "set_v4", "clean": False, "a": tr["a"], "b": tr["b"],
             "kind": tr["type"], "saved_qa": tr["qa"]["verdict"],
             "recipe": {"type": "set_v3", "index": tr["index"], "technique": tr["type"], "b_entry": True}}
            for tr in doc["transitions"]]
    DG.dump(gdir / "refs.json", {"schema_version": "musicmix.defect_gym_refs.1", "source": "runs/dj_set_v4/set.json",
                                 "note": "riferimenti = transizioni accettate del set v4 (QA dj_qa.6 + veto); ricetta set_v3 sui globali "
                                         "di run_dj_set_v3_bridge ricollegati alla run v4", "refs": refs})
    DG.main(["run", "report", "--out", str(gdir), "--refs", "v4_", "--profile", "gate"] + list(extra or []))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stages", nargs="+", choices=("prepare", "set", "export", "gym"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--reliability", choices=("off", "report", "prefer"), default="prefer")
    ap.add_argument("--novelty", nargs="*", default=["runs/dj_set_v3_bridge/set.json"])
    ap.add_argument("--tension", choices=("off", "report", "prefer"), default="prefer")
    ap.add_argument("--plan-only", default=None, help="cartella di uscita di un piano senza render (es. runs/dj_set_v4/plans/report)")
    args = ap.parse_args(argv)
    for st in args.stages:
        t0 = time.time()
        if st == "prepare":
            prepare()
        elif st == "set":
            run_set(args.overwrite, args.reliability, args.novelty, args.tension,
                    plan_only=(ROOT / args.plan_only) if args.plan_only else None)
        elif st == "gym":
            gym()
        else:
            export()
        print(f"[{st}] {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
