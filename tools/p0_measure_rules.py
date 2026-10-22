#!/usr/bin/env python3
"""P0 "misure e regole" (audit dei libri, 21/10/2026): effetto misurato delle correzioni su dati reali.

Tre misure, solo lettura dei run congelati, risultati in ``runs/p0_misure_regole/``:

* ``onsets``: sugli stessi brani di ``runs/dj_phrase_v3_stem_onsets`` (modelli ``runs/dj_phrase_v1``, pilastro
  w = 1), quante sillabe ricevevano un onset dello stem oltre la fine della parola Whisper (modo ``legacy``) e
  diventano ``uncertain_beyond_word`` con il default corretto (``abstain``); quota snapped prima/dopo.
* ``forms``: quanti documenti ``dj_form`` in cache non hanno un core (prima: tutte ``outro``).
* ``cues``: finestre dei cue di ``dj_select`` sul catalogo del ponte (``runs/dj_set_v3_bridge/dj_plan/tracks``):
  quanti downbeat di inizio frase (livello >= 4) ciascuna soglia esclude (OUT >= 35 %, OUT con < 4 battute
  residue, IN <= 50 %, IN con < 16 battute residue) e quali sono inizi di sezione plausibili (ritornello/strofa
  di B, fine di strofa/ritornello di A); coppie di cue escluse; candidati in più con le finestre per tecnica.

    python tools/p0_measure_rules.py [onsets|forms|cues ...]
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from musicmix import dj_select as djs  # noqa: E402

OUT = ROOT / "runs" / "p0_misure_regole"
BRIDGE_TRACKS = ROOT / "runs" / "dj_set_v3_bridge" / "dj_plan" / "tracks"


def measure_onsets() -> dict[str, Any]:
    from musicmix import dj_phrase as P
    from musicmix import dj_render as dj
    import run_dj_phrase_v1 as V1

    rows, skipped = [], []
    for f in sorted((ROOT / "runs/dj_phrase_v1/tracks").glob("*.json")):
        doc = json.loads(f.read_text())
        inp = V1.track_inputs(doc["pool"], doc["slug"])
        if inp is None or "stem" not in inp:
            skipped.append(doc["slug"])
            continue
        stem, sr = dj.load_track(inp["stem"])
        m2 = P.reanalyze_with_pillar(doc["model"], 1.0)
        on = P.stem_syllable_onsets(stem, sr)
        del stem
        ml, legacy = P.with_stem_onsets(m2, onsets=on, beyond_word="legacy")
        m3, fixed = P.with_stem_onsets(m2, onsets=on)
        ph = P.vocal_phase_alignment(m3)
        phl = P.vocal_phase_alignment(ml)
        moved = sum(1 for wl, wf in zip(ml["words"], m3["words"]) for a, b in zip(wl.get("syl_on") or [], wf.get("syl_on") or []) if a != b)
        past = 0  # legacy: syllables placed at or after their word's ASR end
        for w in ml["words"]:
            past += sum(1 for t, src in zip(w.get("syl_on") or [], w.get("syl_src") or []) if src == "stem" and t >= w["off"])
        rows.append({"slug": doc["slug"], "lang": doc["model"].get("lang"),
                     "asr_reliable": (doc["model"].get("asr_quality") or {}).get("reliable", True),
                     "n_syllables": fixed["n_syllables"], "snapped_legacy": legacy["snapped_share"],
                     "snapped_fixed": fixed["snapped_share"], "uncertain_beyond_word": fixed["n_uncertain_beyond_word"],
                     "uncertain_share": fixed["uncertain_share"], "phase_beats_fixed": ph.get("phase_beats"),
                     "phase_beats_legacy": phl.get("phase_beats"), "n_snapped_legacy": legacy["n_snapped"],
                     "n_snapped_fixed": fixed["n_snapped"], "legacy_past_word_end": past, "syllables_moved": moved})
        print(f"{doc['slug']:40s} snapped {legacy['n_snapped']} -> {fixed['n_snapped']}  past-end(legacy) {past}  "
              f"uncertain {fixed['n_uncertain_beyond_word']}  moved {moved}", flush=True)
    rel = [r for r in rows if r["asr_reliable"]]
    med = lambda k, rr: float(np.median([r[k] for r in rr if r[k] is not None])) if rr else None  # noqa: E731
    tot_syl = sum(r["n_syllables"] for r in rows)
    tot_unc = sum(r["uncertain_beyond_word"] for r in rows)
    return {"n_tracks": len(rows), "n_reliable": len(rel), "skipped": skipped,
            "syllables": tot_syl, "uncertain_beyond_word": tot_unc, "uncertain_share_total": tot_unc / tot_syl if tot_syl else None,
            "legacy_past_word_end": sum(r["legacy_past_word_end"] for r in rows),
            "syllables_moved": sum(r["syllables_moved"] for r in rows),
            "n_snapped_legacy": sum(r["n_snapped_legacy"] for r in rows), "n_snapped_fixed": sum(r["n_snapped_fixed"] for r in rows),
            "median_reliable": {k: med(k, rel) for k in ("snapped_legacy", "snapped_fixed", "uncertain_share", "phase_beats_fixed",
                                                         "phase_beats_legacy")},
            "tracks": rows,
            "note": "legacy = finestra fino all'ultimo bersaglio + 60 ms (prima dell'audit); fixed = [on - 60 ms, off) e "
                    "sillaba incerta se l'onset vicino è oltre la fine della parola. La fase di riferimento era +0,033 battiti "
                    "(runs/dj_phrase_v3_stem_onsets/compare.json, 29 brani affidabili)."}


def measure_forms() -> dict[str, Any]:
    out = {}
    for d in sorted((ROOT / ".cache").glob("*/form")):
        rows = []
        for f in sorted(d.glob("*.json")):
            form = json.loads(f.read_text())["form"]
            if form.get("first_core_bar") is None:
                rows.append({"slug": f.stem, "n_bars": len(form["labels"]), "old_labels": dict(Counter(form["labels"])),
                             "kick_bars": int(sum(form["active"]["kick"])), "core_level": form.get("core_level"),
                             "version": form.get("version")})
        out[str(d.relative_to(ROOT))] = {"n_forms": len(list(d.glob("*.json"))), "no_core": rows}
    return out


SECTION_IN_PLAUSIBLE = ("chorus", "verse")
SECTION_OUT_PLAUSIBLE = ("chorus", "verse")


def _all_phrase_cues(track: dict, min_level: int = 4) -> list[dict]:
    rows, dur, n = track["phrases"], track["duration_s"], len(track["phrases"])
    out = []
    for r in rows:
        if r["phrase_level"] < min_level:
            continue
        j = r["bar_index"]
        out.append({"bar": j, "time_s": r["time_s"], "frac": r["time_s"] / dur, "left": n - j, "level": r["phrase_level"],
                    "section_start": r["section_start"], "label": r["section_label"],
                    "prev_label": rows[j - 1]["section_label"] if j > 0 else "start"})
    return out


def measure_cues() -> dict[str, Any]:
    tracks = {f.stem: json.loads(f.read_text()) for f in sorted(BRIDGE_TRACKS.glob("*.json"))}
    per = {}
    tot = Counter()
    plausible_excl: dict[str, list] = {"in_after_50pct": [], "in_last_16_bars": [], "out_before_35pct": [], "out_last_4_bars": []}
    for slug, T in tracks.items():
        cues = _all_phrase_cues(T)
        c = Counter()
        for q in cues:
            c["phrase_cues"] += 1
            # OUT
            o35, o4 = q["frac"] < djs.OUT_MIN_POSITION, q["left"] < 4
            c["out_kept"] += int(not (o35 or o4))
            c["out_excl_35pct"] += int(o35)
            c["out_excl_last4"] += int(o4)
            out_plaus = q["section_start"] and q["prev_label"] in SECTION_OUT_PLAUSIBLE
            c["out_plausible"] += int(out_plaus)
            if out_plaus and o35:
                c["out_plausible_excl_35pct"] += 1
                plausible_excl["out_before_35pct"].append({"slug": slug, "bar": q["bar"], "frac": round(q["frac"], 3),
                                                           "after": q["prev_label"], "time_s": round(q["time_s"], 1)})
            if out_plaus and o4:
                c["out_plausible_excl_last4"] += 1
                plausible_excl["out_last_4_bars"].append({"slug": slug, "bar": q["bar"], "after": q["prev_label"]})
            # IN
            i50, i16 = q["frac"] > djs.IN_MAX_POSITION, q["left"] < 16
            c["in_kept"] += int(not (i50 or i16))
            c["in_excl_50pct"] += int(i50)
            c["in_excl_last16"] += int(i16)
            c["in_excl_last16_only"] += int(i16 and not i50)
            in_plaus = q["section_start"] and q["label"] in SECTION_IN_PLAUSIBLE
            c["in_plausible"] += int(in_plaus)
            if in_plaus and i50:
                c["in_plausible_excl_50pct"] += 1
                plausible_excl["in_after_50pct"].append({"slug": slug, "bar": q["bar"], "frac": round(q["frac"], 3), "label": q["label"],
                                                         "left_bars": q["left"], "left_s": round(T["duration_s"] - q["time_s"], 1)})
            if in_plaus and i16:
                c["in_plausible_excl_last16"] += 1
                plausible_excl["in_last_16_bars"].append({"slug": slug, "bar": q["bar"], "label": q["label"], "left_bars": q["left"]})
        # first chorus of the track: is it a cue-in candidate?
        first_ch = next((q for q in cues if q["section_start"] and q["label"] == "chorus"), None)
        c["has_chorus"] = int(first_ch is not None)
        c["first_chorus_in_ok"] = int(first_ch is not None and first_ch["frac"] <= djs.IN_MAX_POSITION and first_ch["left"] >= 16)
        per[slug] = dict(c, duration_s=round(T["duration_s"], 1), n_bars=len(T["phrases"]),
                         first_chorus_frac=round(first_ch["frac"], 3) if first_ch else None)
        tot.update(c)
    # pairs (ordered A != B): plan_pair candidates = outs(A) x ins(B)
    n_out = {s: len(djs.cue_candidates(T, "out")) for s, T in tracks.items()}
    n_in = {s: len(djs.cue_candidates(T, "in")) for s, T in tracks.items()}
    n_out_all = {s: sum(1 for q in _all_phrase_cues(T) if q["left"] >= 1) for s, T in tracks.items()}
    pairs = [(a, b) for a in tracks for b in tracks if a != b]
    kept = sum(n_out[a] * n_in[b] for a, b in pairs)
    allp = sum(n_out_all[a] * n_out_all[b] for a, b in pairs)
    tech = {}
    for name, w in djs.CUE_WINDOWS.items():
        no = {s: len(djs.cue_candidates(T, "out", windows=w)) for s, T in tracks.items()}
        ni = {s: len(djs.cue_candidates(T, "in", windows=w)) for s, T in tracks.items()}
        tech[name] = {"windows": w, "out_cues": sum(no.values()), "in_cues": sum(ni.values()),
                      "pair_plans": sum(no[a] * ni[b] for a, b in pairs)}
    # through the set planner (books profile, no forms, before the runner's QA/vocal filters)
    from musicmix import dj_set as ds
    prof = ds.profile("books")
    pipe = Counter()
    changed = []
    for a, b in pairs:
        base = ds.candidates_for_pair(tracks[a], tracks[b], prof["tree"], prof["order"])
        wide = ds.candidates_for_pair(tracks[a], tracks[b], prof["tree"], prof["order"], cue_windows="technique")
        pipe["cands_default"] += len(base["candidates"])
        pipe["cands_technique"] += len(wide["candidates"])
        pipe["dropped_overlap_outside_default"] += wide["dropped_overlap_outside_default"]
        extra = [c for c in wide["candidates"] if c.get("cue_windows") == "technique"]
        pipe["extra_candidates"] += len(extra)
        for c in extra:
            pipe["extra_type_" + c["decision"]["type"]] += 1
        if base["candidates"] and wide["candidates"]:
            b0, w0 = base["candidates"][0], wide["candidates"][0]
            if (b0["plan"]["a_out_bar"], b0["plan"]["b_in_bar"]) != (w0["plan"]["a_out_bar"], w0["plan"]["b_in_bar"]):
                pipe["best_changed"] += 1
                changed.append({"a": a, "b": b, "default": [b0["plan"]["a_out_bar"], b0["plan"]["b_in_bar"], b0["decision"]["type"],
                                                            round(b0["cost"], 3), b0["plan"]["b_section"]],
                                "technique": [w0["plan"]["a_out_bar"], w0["plan"]["b_in_bar"], w0["decision"]["type"],
                                              round(w0["cost"], 3), w0["plan"]["b_section"], w0.get("cue_windows")]})
    return {"catalog": str(BRIDGE_TRACKS.relative_to(ROOT)), "n_tracks": len(tracks), "n_ordered_pairs": len(pairs),
            "set_planner_books_profile": {**dict(pipe), "best_changed_examples": changed[:40],
                                          "note": "dj_set.candidates_for_pair con profilo libri, senza forme dj_form e prima dei "
                                                  "filtri del runner (guardia vocale, QA): conta i candidati, non la qualità"},
            "thresholds": {"OUT_MIN_POSITION": djs.OUT_MIN_POSITION, "IN_MAX_POSITION": djs.IN_MAX_POSITION,
                           "out_min_bars_left": 4, "in_min_bars_left": 16, "min_phrase_level": 4},
            "totals": dict(tot), "pair_plans_default": kept, "pair_plans_without_windows": allp,
            "pair_plans_excluded_share": 1 - kept / allp if allp else None,
            "windows_profiles": tech, "plausible_excluded": plausible_excl, "per_track": per,
            "plausible_definition": "[H] proxy senza annotazioni: IN plausibile = inizio di sezione allin1 'chorus'/'verse' su "
                                    "downbeat di frase >= 4; OUT plausibile = inizio di sezione dopo 'chorus'/'verse' (fine di "
                                    "strofa/ritornello). Non è recall su cue annotati da DJ: nel repo non esistono per questi brani."}


def main() -> int:
    what = sys.argv[1:] or ["forms", "cues", "onsets"]
    OUT.mkdir(parents=True, exist_ok=True)
    for w in what:
        t0 = time.time()
        res = {"forms": measure_forms, "cues": measure_cues, "onsets": measure_onsets}[w]()
        res["elapsed_s"] = round(time.time() - t0, 1)
        res["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        (OUT / f"{w}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
        print(w, "->", OUT / f"{w}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
