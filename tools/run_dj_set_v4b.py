#!/usr/bin/env python3
"""Set DJ v4b: il set v4 ricontrollato con ``dj_qa.7``; rifatte solo le transizioni che non passano.

Strato sottile sopra ``tools/run_dj_set_v4.py`` / ``tools/run_dj_set_v3_bridge.py``. La run ``runs/dj_set_v4`` resta
congelata: ``prepare`` copia catalogo, analisi DJ, modelli ``dj_phrase`` e cache delle attività in ``runs/dj_set_v4b``
(con gli sha256), così pianificazione e render ripartono dagli stessi ingressi.

* ``set``: stessa pianificazione del v4 (``--reliability prefer --cards track_cards --novelty runs/dj_set_v3_bridge/set.json
  --tension prefer``, piano per eventi acceso: il piano è deterministico). Per ogni transizione il **primo** tentativo è la
  transizione accettata nel v4 (stesso candidato, stessa tecnica, stesso motivo), rifatta e valutata con il QA di oggi
  (``dj_qa.7``) e il veto: se passa resta (PCM confrontato con lo sha256 del v4), altrimenti seguono gli altri tentativi del
  runner e, se nessuno passa, la ripianificazione senza ripieghi (arco tolto, ordine e cue rifatti). Il runner dichiara lo
  spostamento del cue di B (``meta['b_entry']['relocated_bars']``, INGRESSO_B R-B2) e la continuità misurata è la .2.
* ``recheck``: ``runs/dj_set_v4b/recheck_v4_qa7.json``: verdetto v4 (``dj_qa.6``) contro ``dj_qa.7`` della stessa
  transizione, transizioni rifatte e perché.
* ``export``: ``deliverables/set_dj_v4b.mp3`` + tracklist (tecnica, carica, risoluzione, continuità misurata .2).
* ``gym``: palestra dei difetti sulle transizioni del set v4b (``runs/dj_set_v4b/defect_gym``).

    python tools/run_dj_set_v4b.py prepare set recheck export gym
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import run_dj_set_v4 as v4  # noqa: E402

v3, v2 = v4.v3, v4.v2
RUN = "dj_set_v4b"
SCRIPT = "tools/run_dj_set_v4b.py"
OUT = ROOT / "runs" / RUN
V4 = ROOT / "runs" / "dj_set_v4"
DELIVER = ROOT / "deliverables"
FIRST_WHY = "transizione accettata nel set v4 (ricontrollo con dj_qa.7)"


def rebind() -> None:
    v4.RUN, v4.OUT = RUN, OUT
    v4.rebind()
    v3.SCRIPT = SCRIPT


def prepare() -> dict:
    rebind()
    OUT.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name in ("catalog_frozen.json", "prepare.json"):
        if not (OUT / name).is_file():
            shutil.copy2(V4 / name, OUT / name)
        copied[name] = v4.sha(OUT / name)
    for d in ("dj_plan", "phrase"):
        if not (OUT / d).is_dir():
            shutil.copytree(V4 / d, OUT / d)
    for sub in ("act", "form"):
        src, dst = ROOT / ".cache" / "dj_set_v4" / sub, ROOT / ".cache" / RUN / sub
        if src.is_dir() and not dst.is_dir():
            shutil.copytree(src, dst)
    files = {str(p.relative_to(OUT)): v4.sha(p) for d in ("dj_plan", "phrase") for p in sorted((OUT / d).rglob("*.json"))}
    same = all(v4.sha(V4 / k) == h for k, h in files.items())
    doc = {"copied_from": "runs/dj_set_v4 (congelata, non modificata)", "files": {**copied, **files}, "identical_to_v4": same,
           "cache": f".cache/{RUN} (copia di .cache/dj_set_v4: act, form)", "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    v3.dump(OUT / "copied_inputs.json", doc)
    print(f"ingressi copiati dal v4: {len(files) + len(copied)} file, identici: {same}", flush=True)
    return doc


def _v4_accepted() -> dict:
    doc = json.loads((V4 / "set.json").read_text(encoding="utf-8"))
    out = {}
    for tr in doc["transitions"]:
        acc = next(r for r in reversed(tr["attempts"]) if r.get("verdict") and r["verdict"] != "fail" and not r.get("veto"))
        out[(tr["a"], tr["b"])] = {"index": tr["index"], "type": acc["type"], "why": acc["why"], "a_out_bar": acc["a_out_bar"],
                                   "b_in_bar": acc["b_in_bar"], "clip_sha256": tr.get("clip_sha256"), "qa": tr["qa"]}
    return out


class FirstV4:
    """Proxy di ``TensionHook``: dopo ``order_attempts`` mette per primo il tentativo accettato nel v4 per la coppia."""

    def __init__(self, inner, accepted: dict):
        self.inner, self.accepted, self.log = inner, accepted, {}

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def order_attempts(self, attempts: list, cands, a: str, b: str) -> list:
        attempts = self.inner.order_attempts(attempts, cands, a, b)
        acc = self.accepted.get((a, b))
        if acc is None:
            self.log[(a, b)] = {"status": "coppia assente nel v4"}
            return attempts

        def hit(x, strict):
            i, dec, why = x
            pl = cands[i]["plan"]
            ok = (dec.get("type_key") == acc["type"] and int(pl["a_out_bar"]) == int(acc["a_out_bar"])
                  and int(pl["b_in_bar"]) == int(acc["b_in_bar"]))
            return ok and (why == acc["why"] or not strict)

        k = next((n for n, x in enumerate(attempts) if hit(x, True)), None)
        if k is None:
            k = next((n for n, x in enumerate(attempts) if hit(x, False)), None)
        if k is None:
            self.log[(a, b)] = {"status": "tentativo del v4 non trovato tra quelli del runner"}
            return attempts
        i, dec, why = attempts[k]
        self.log[(a, b)] = {"status": "primo", "candidate": i, "type": dec.get("type_key"), "why_v4": why}
        return [(i, dec, f"{FIRST_WHY}: {why}")] + attempts[:k] + attempts[k + 1:]


def on_accept_v4b(t, a, b, res, dec, c, rec, qa) -> dict:
    extra = v3.on_accept_v3(t, a, b, res, dec, c, rec, qa)
    extra["entry_relocation"] = c["plan"].get("entry_relocation")
    extra["b_entry_declared"] = {k: (res.metadata.get("b_entry") or {}).get(k) for k in ("relocated_bars", "relocation", "audible_output_sample")}
    return extra


def run_set(overwrite: bool) -> None:
    rebind()
    v3.RELIABILITY.update(mode="prefer", cards=ROOT / "track_cards", novelty=[str(ROOT / "runs/dj_set_v3_bridge/set.json")], params={})
    v3.TENSION["mode"] = "prefer"
    v3.EVENT_PLAN.update(on=True, pref=False)
    first = FirstV4(v3.tension_hook(), _v4_accepted())
    hooks = v2.SetHooks(
        run=RUN, script=SCRIPT,
        title="set DJ v4b (set v4 ricontrollato con dj_qa.7; rifatte solo le transizioni che non passano)",
        pool=v3.sources_v3, info=v3.info_v3, filter_pair=v3.filter_pair_v3, attempts=v3.attempts_v3, render=v3.render_v3,
        run_qa=v3.qa_v3, veto=v3._track_tried(v3.veto_v3), lead_a_s=v3.lead_a_v3, overlap_b_s=v3.overlap_b_v3,
        on_accept=on_accept_v4b, finalize_audio=v3.finalize_v3, mp3_bitrate="192k", order_overrides=dict(v3.ORDER_OVERRIDES),
        compare_v1b=False,
        notes=("integrità tecnica (QA dj_qa.7, veto del critico, palestra) non è qualità musicale: il set non è stato ascoltato",
               "stessa pianificazione del set v4; primo tentativo di ogni transizione = transizione accettata nel v4",
               "spostamento del cue di B dichiarato nel render (meta['b_entry']['relocated_bars'], INGRESSO_B R-B2)",
               "continuità misurata dj_tension.continuity_measured.2 (groove: banda bassa o percussioni, riferimento per lato)",
               "catalogo: runs/dj_set_v4b/catalog_frozen.json = copia di runs/dj_set_v4/catalog_frozen.json"),
        cue_overrides={}, reliability=v3.reliability_hook(), tension=first,
        event_fn=v3.event_fn_v3, after_cues=v3.after_cues_v3)
    v3._CTX["out"] = OUT
    if "copied_inputs.json" not in v2.KEEP_ON_OVERWRITE:
        v2.KEEP_ON_OVERWRITE = tuple(v2.KEEP_ON_OVERWRITE) + ("copied_inputs.json",)
    v2.build_set(OUT, overwrite=overwrite, hooks=hooks)
    v3.dump(OUT / "first_attempts.json", {f"{a}->{b}": v for (a, b), v in first.log.items()})


def remeasure() -> dict:
    """Ricalcola ``tension.continuity_measured`` sulle clip accettate (``transitions/tNN_accepted.wav``) con la versione
    corrente di ``dj_tension.continuity_measured`` (stessi argomenti di ``run_dj_set_v3_bridge.tension_record``); il valore
    calcolato durante ``set`` resta in ``continuity_measured_at_set``. Non tocca audio né verdetti."""
    import soundfile as sf
    from musicmix import dj_tension as DT

    rebind()
    doc = json.loads((OUT / "set.json").read_text(encoding="utf-8"))
    for tr in doc["transitions"]:
        name = f"t{tr['index']:02d}_accepted"
        meta = json.loads((OUT / "transitions" / f"{name}.json").read_text(encoding="utf-8"))
        x, sr = sf.read(str(OUT / "transitions" / f"{name}.wav"), dtype="float64", always_2d=True)
        tz = tr.setdefault("tension", {})
        sa = v3.tension_signals(tr["a"])
        g = v3.ds.clip_geometry(meta, len(x))
        aud = (meta.get("b_entry") or {}).get("audible_output_sample")
        if "continuity_measured_at_set" not in tz:
            tz["continuity_measured_at_set"] = tz.get("continuity_measured")
        tz["continuity_measured"] = DT.continuity_measured(x, sr, int(g["switch"]), float(sa.period_s),
                                                           b_audible_sample=None if aud is None else int(aud))
        cm = tz["continuity_measured"]
        print(f"T{tr['index']:02d} {tr['a']} -> {tr['b']} {tr['type']}: continuità misurata {cm.get('value')} "
              f"(buco {cm.get('gap_beats')}; bassa {cm.get('gap_beats_low')}, .1 {cm.get('gap_beats_low_v1')})", flush=True)
    (OUT / "set.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return doc


def recheck() -> dict:
    """Confronto v4 (dj_qa.6) / v4b (dj_qa.7) per transizione: il primo tentativo del v4b è la transizione del v4."""
    old = json.loads((V4 / "set.json").read_text(encoding="utf-8"))
    new = json.loads((OUT / "set.json").read_text(encoding="utf-8"))
    acc4 = {(t["a"], t["b"]): t for t in old["transitions"]}
    rows = []
    for t4 in old["transitions"]:
        key = (t4["a"], t4["b"])
        t5 = next((t for t in new["transitions"] if (t["a"], t["b"]) == key), None)
        row = {"index_v4": t4["index"], "a": t4["a"], "b": t4["b"], "technique_v4": t4["type"],
               "qa_v4": {"version": old["versions"]["dj_qa"], "verdict": t4["qa"]["verdict"], "reasons": t4["qa"]["reasons"]}}
        # primo tentativo con il motivo FIRST_WHY in qualunque passata (set.json: replanning[].tried o transitions[].attempts)
        tries = []
        for rp in (new.get("replanning") or {}).get("replans") or []:
            if rp.get("arc") == list(key):
                tries += rp.get("tried") or []
        if t5 is not None:
            tries += t5["attempts"]
        f = next((r for r in tries if str(r.get("why", "")).startswith(FIRST_WHY)), None)
        if f is not None:
            row["qa7_same_transition"] = {"verdict": f.get("verdict"), "reasons": f.get("reasons"), "veto": f.get("veto")}
        row["kept"] = bool(t5 is not None and t5.get("clip_sha256") == t4.get("clip_sha256"))
        row["pcm_identical_to_v4"] = row["kept"]
        if t5 is not None:
            row.update(index_v4b=t5["index"], technique_v4b=t5["type"], qa_v4b={"verdict": t5["qa"]["verdict"], "reasons": t5["qa"]["reasons"]},
                       cue_v4=[t4["plan"]["a_out_bar"], t4["plan"]["b_in_bar"]], cue_v4b=[t5["plan"]["a_out_bar"], t5["plan"]["b_in_bar"]])
        else:
            row["arc_removed_in_v4b"] = True
        rows.append(row)
    newpairs = [{"index": t["index"], "a": t["a"], "b": t["b"], "technique": t["type"], "qa": t["qa"]["verdict"]}
                for t in new["transitions"] if (t["a"], t["b"]) not in acc4]
    cnt = lambda it: {k: sum(1 for x in it if x == k) for k in ("pass", "warn", "fail", None)}  # noqa: E731
    doc = {"schema": "musicmix.set_recheck.1", "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "v4": "runs/dj_set_v4/set.json (congelata)", "v4b": f"runs/{RUN}/set.json", "dj_qa_v4": old["versions"]["dj_qa"],
           "dj_qa_v4b": new["versions"]["dj_qa"],
           "summary": {"v4_verdicts_qa6": cnt([r["qa_v4"]["verdict"] for r in rows]),
                       "v4_transitions_with_qa7": cnt([(r.get("qa7_same_transition") or {}).get("verdict") for r in rows]),
                       "v4_vetoed_with_qa7": [r["index_v4"] for r in rows if (r.get("qa7_same_transition") or {}).get("veto")],
                       "kept_identical": [r["index_v4"] for r in rows if r["kept"]],
                       "redone": [r["index_v4"] for r in rows if not r["kept"]],
                       "v4b_verdicts": cnt([t["qa"]["verdict"] for t in new["transitions"]]),
                       "replans": len((new.get("replanning") or {}).get("replans") or []),
                       "banned_arcs": (new.get("replanning") or {}).get("banned_arcs"), "new_pairs": newpairs},
           "transitions": rows, "note": "integrità tecnica, non qualità musicale"}
    v3.dump(OUT / "recheck_v4_qa7.json", doc)
    print(json.dumps(doc["summary"], ensure_ascii=False, indent=1), flush=True)
    return doc


def _decode(mp3: Path):
    import subprocess

    import numpy as np

    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp3), "-f", "f32le", "-ac", "2", "-ar", str(v3.SR), "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2)


def _tp(x) -> float:
    c = 30 * v3.SR
    return max(v3.dj.true_peak_dbtp(x[s:s + c], v3.SR) for s in range(0, len(x), c))


def deliverable_mp3(bitrate: str = "192k", max_tp: float = -1.0) -> dict:
    """MP3 consegnato con true peak <= ``max_tp`` **dopo la decodifica** (l'encoder MP3 alza i picchi: il set.mp3 del v4,
    identico, decodificato arriva a circa -0,2 dBTP). Se serve: limitatore true-peak di ``dj_render`` sul set.wav con un
    tetto abbassato della sovraelongazione misurata (a blocchi di 30 s, come ``run_dj_set_v1.limit_long``), codifica
    da pipe (nessun WAV intermedio), misura di nuovo; al più 4 passi. ``runs/<run>/set.wav``/``set.mp3`` non cambiano."""
    import subprocess

    import numpy as np
    import soundfile as sf

    src = OUT / "set.mp3"
    dst = OUT / "set_deliverable.mp3"
    wav, sr = sf.read(str(OUT / "set.wav"), dtype="float32", always_2d=True)
    tp_wav = _tp(wav)
    dec = _decode(src)
    steps = [{"mp3": "set.mp3", "ceiling_dbtp": None, "true_peak_decoded_dbtp": _tp(dec),
              "lufs_decoded": v3.dj.integrated_lufs(dec.astype(np.float64), sr)}]
    del dec
    out = src
    ceiling = -1.0
    while steps[-1]["true_peak_decoded_dbtp"] > max_tp and len(steps) <= 4:
        ceiling = ceiling - (steps[-1]["true_peak_decoded_dbtp"] - max_tp) - 0.1
        y = np.empty_like(wav)
        c, m, n = 30 * sr, int(0.2 * sr), len(wav)
        for s0 in range(0, n, c):
            lo, hi = max(0, s0 - m), min(n, s0 + c + m)
            z, _ = v3.dj.true_peak_limit(wav[lo:hi], sr, ceiling)
            y[s0:min(n, s0 + c)] = z[s0 - lo:s0 - lo + min(c, n - s0)]
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "-", "-codec:a", "libmp3lame",
                        "-b:a", bitrate, str(dst)], input=y.astype(np.float32).tobytes(), check=True)
        del y
        dec = _decode(dst)
        steps.append({"mp3": dst.name, "ceiling_dbtp": ceiling, "true_peak_decoded_dbtp": _tp(dec),
                      "lufs_decoded": v3.dj.integrated_lufs(dec.astype(np.float64), sr)})
        del dec
        out = dst
    info = {"source_wav_true_peak_dbtp": tp_wav, "steps": steps, "delivered": str(out.relative_to(ROOT)),
            "true_peak_decoded_dbtp": steps[-1]["true_peak_decoded_dbtp"], "lufs_decoded": steps[-1]["lufs_decoded"],
            "ok": steps[-1]["true_peak_decoded_dbtp"] <= max_tp, "sha256": v4.sha(out)}
    print(json.dumps(info, indent=1), flush=True)
    return info


def export() -> dict:
    doc = json.loads((OUT / "set.json").read_text(encoding="utf-8"))
    if not doc.get("closed"):
        raise SystemExit("il set non si è chiuso: niente export")
    from musicmix import dj_qa

    used = doc["versions"]["dj_qa"]
    doc["qa_used"] = {"dj_qa": used, "current_dj_qa": dj_qa.DJ_QA_VERSION, "needs_recheck": used != dj_qa.DJ_QA_VERSION,
                      "note": "set v4 ricontrollato con " + used + " e veto tecnico del critico; rifatte solo le transizioni che non passano "
                              f"(runs/{RUN}/recheck_v4_qa7.json)"}
    dl = deliverable_mp3()
    doc.setdefault("output", {})["deliverable_mp3"] = dl
    (OUT / "set.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    DELIVER.mkdir(exist_ok=True)
    shutil.copy2(ROOT / dl["delivered"], DELIVER / "set_dj_v4b.mp3")
    lines = (OUT / "tracklist.txt").read_text(encoding="utf-8").rstrip("\n").split("\n")
    lines += ["", "Per transizione (dj_tension, pesi [H] pre-registrati; continuità misurata .2 = groove; descrizione, non giudizio):"]
    for tr in doc["transitions"]:
        tz = tr.get("tension") or {}
        cm = tz.get("continuity_measured") or {}
        f = lambda x: "n/d" if x is None else f"{x:.2f}"  # noqa: E731
        lines.append(f"  T{tr['index']:02d} {tr['a']} -> {tr['b']}: {tr['label']}; carica A {f(tz.get('a_charge'))} "
                     f"({tz.get('a_charge_form')}), risoluzione B {f(tz.get('b_resolution'))}, continuità {f(tz.get('continuity'))} "
                     f"(misurata {f(cm.get('value'))}, buco {f(cm.get('gap_beats'))} battiti; sola banda bassa {f(cm.get('gap_beats_low'))}), "
                     f"scontro tonale {f(tz.get('harmonic_clash'))}, testo {f(tz.get('lyric_affinity'))} [{tz.get('lyric_link')}], "
                     f"affidabilità {f((tr.get('reliability') or {}).get('value'))}, QA {tr['qa']['verdict']}")
    (DELIVER / "set_dj_v4b_tracklist.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(OUT / "tracklist.csv", DELIVER / "set_dj_v4b_tracklist.csv")
    print("\n".join(lines), flush=True)
    return doc


def gym(extra: list | None = None) -> None:
    """Palestra dei difetti (``tools/defect_gym.py``) sulle transizioni del set v4b, come ``run_dj_set_v4.gym``."""
    rebind()
    import defect_gym as DG

    doc = json.loads((OUT / "set.json").read_text(encoding="utf-8"))
    gdir = OUT / "defect_gym"
    refs = [{"id": f"v4b_t{tr['index']:02d}_{tr['type']}", "origin": "set_v4b", "clean": False, "a": tr["a"], "b": tr["b"],
             "kind": tr["type"], "saved_qa": tr["qa"]["verdict"],
             "recipe": {"type": "set_v3", "index": tr["index"], "technique": tr["type"], "b_entry": True}}
            for tr in doc["transitions"]]
    DG.dump(gdir / "refs.json", {"schema_version": "musicmix.defect_gym_refs.1", "source": f"runs/{RUN}/set.json",
                                 "note": "riferimenti = transizioni accettate del set v4b (QA dj_qa.7 + veto); ricetta set_v3 sui "
                                         "globali di run_dj_set_v3_bridge ricollegati alla run v4b", "refs": refs})
    DG.main(["run", "report", "--out", str(gdir), "--refs", "v4b_", "--profile", "gate"] + list(extra or []))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stages", nargs="+", choices=("prepare", "set", "remeasure", "recheck", "export", "gym"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    for st in args.stages:
        t0 = time.time()
        {"prepare": prepare, "set": lambda: run_set(args.overwrite), "remeasure": remeasure, "recheck": recheck, "export": export, "gym": gym}[st]()
        print(f"[{st}] {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
