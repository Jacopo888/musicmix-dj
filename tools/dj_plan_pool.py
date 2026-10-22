#!/usr/bin/env python3
"""DJ analysis and cue planning for a MusicMix catalog (core venv).

Reads (never modifies) the catalog, Beat This! structure JSON, Demucs vocal
occupancy, LinkSeg / SSM-Net sections and the provider JSON written by
``tools/dj_providers.py`` (All-In-One, madmom, essentia; isolated venv), then
uses the pure functions of :mod:`musicmix.dj_select` to write::

    <out>/tracks/<slug>.json   per-track grid, phrases, sections, key, vocals, energy
    <out>/pairs.json           all ordered pairs: compatibility + top cue plans
    <out>/summary.json         compact table + best plans across pairs

Usage::

    python tools/dj_plan_pool.py \
        --catalog runs/pool_v12/catalog_ready.json --out runs/dj_plan_v1
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicmix import dj_select as djs  # noqa: E402
from musicmix.paths import localize  # noqa: E402

LINKSEG_DIR = ROOT / "runs/pretrained_v10/linkseg_pool6"
SSMNET_DIR = ROOT / "runs/ssmnet_pool_v12"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _np_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def dump(obj: Any) -> str:
    return json.dumps(obj, indent=1, default=_np_default)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def krumhansl_from_audio(mono: np.ndarray, sr: int) -> dict[str, Any]:
    import librosa
    from scipy.signal import resample_poly
    y = resample_poly(mono, 22050, sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=22050, hop_length=2048).mean(axis=1)
    return djs.krumhansl_key(chroma)


def section_inputs(slug: str, prov: dict, duration: float) -> tuple[list[dict], dict[str, list[float]], str]:
    """Primary labelled sections and boundary supports from available providers."""
    supports: dict[str, list[float]] = {}
    primary: list[dict] = []
    source = "none"
    a1 = prov.get("allin1", {})
    if a1.get("status") == "complete":
        primary = [{"start_s": s["start"], "end_s": s["end"], "label": s["label"], "source": "allin1"}
                   for s in a1["segments"]]
        supports["allin1"] = [s["start"] for s in a1["segments"]]
        source = "allin1"
    ls_path = LINKSEG_DIR / f"{slug}.json"
    if ls_path.exists():
        ls = load_json(ls_path)
        secs = [s for s in ls.get("sections", [])]
        supports["linkseg"] = [s["start_seconds"] for s in secs]
        if not primary:
            primary = [{"start_s": s["start_seconds"], "end_s": s["end_seconds"], "label": s["label"],
                        "source": "linkseg"} for s in secs]
            source = "linkseg"
    sm_path = SSMNET_DIR / slug / "prediction.json"
    if sm_path.exists():
        sm = load_json(sm_path)["result"]
        supports["ssmnet"] = [e["time_canonical_sec"] for e in sm["events_official"]]
        if not primary:
            source = "ssmnet"
    if not primary and "ssmnet" in supports:
        supports = {"ssmnet": supports["ssmnet"], **{k: v for k, v in supports.items() if k != "ssmnet"}}
    return primary, supports, source


def key_estimates(prov: dict, mono: np.ndarray, sr: int) -> dict[str, dict]:
    est: dict[str, dict] = {}
    mk = prov.get("madmom_key", {})
    if mk.get("status") == "complete":
        wins = mk.get("windows", [])
        stab = (sum(1 for w in wins if w["key"] == mk["key"]) / len(wins)) if wins else None
        est["madmom_cnn"] = {"key": mk["key"], "prob": mk["prob"], "window_stability": stab,
                             "second": mk.get("second"), "prob2": mk.get("prob2")}
    ek = prov.get("essentia_key", {})
    if ek.get("status") == "complete":
        for prof in ("edma", "bgate"):
            p = ek["profiles"][prof]
            est[f"essentia_{prof}"] = {"key": p["key"], "strength": p["strength"]}
    kk = krumhansl_from_audio(mono, sr)
    est["krumhansl"] = {"key": kk["key"], "r": kk["r"], "margin": kk["margin"]}
    return est


def analyse_track(t: dict, prov_dir: Path) -> dict[str, Any]:
    slug = t["slug"]
    wav = localize(t["canonical_path"])
    if _sha(wav) != t["track_id"]:
        raise RuntimeError(f"canonical audio identity mismatch for {slug}")
    audio, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    duration = len(mono) / sr
    prov_path = prov_dir / f"{slug}.json"
    prov = load_json(prov_path) if prov_path.exists() else {}

    # ---- grids ------------------------------------------------------------
    st = load_json(localize(t["structure_path"]))
    ssr = st["sample_rate"]
    raw = {"beat_this": ([e["sample"] / ssr for e in st["beats"]], [e["sample"] / ssr for e in st["downbeats"]])}
    if prov.get("allin1", {}).get("status") == "complete":
        raw["allin1"] = (prov["allin1"]["beats"], prov["allin1"]["downbeats"])
    if prov.get("madmom_downbeats", {}).get("status") == "complete":
        raw["madmom"] = (prov["madmom_downbeats"]["beats"], prov["madmom_downbeats"]["downbeats"])
    fits = {name: djs.fit_constant_grid(b, d) for name, (b, d) in raw.items() if len(b) >= 8}
    grid = djs.choose_grid(fits, duration)
    beats, downbeats = djs.grid_times(grid["fit"], duration)
    prov_beats = np.asarray(raw[grid["provider"]][0])
    lo, hi = prov_beats.min() - grid["fit"]["period_s"], prov_beats.max() + grid["fit"]["period_s"]
    downbeats = downbeats[(downbeats >= lo) & (downbeats <= hi)]

    # ---- sections and phrases --------------------------------------------
    primary, supports, sec_source = section_inputs(slug, prov, duration)
    sections = djs.merge_section_sources(primary, supports, downbeats, duration)
    phrases = djs.phrase_table(downbeats, sections)

    # ---- vocals and energy -----------------------------------------------
    voc = load_json(localize(t["vocals_path"]))
    vsr = voc["sample_rate"]
    vint = [(i["start_sample"] / vsr, i["end_sample"] / vsr) for i in voc["acoustic_occupancy"]
            if i["status"] == "vocal_occupancy"]
    starts, ends = djs.bar_bounds(downbeats, duration)
    bar_vocal = djs.interval_coverage(vint, starts, ends)
    p = grid["fit"]["period_s"]
    last_beat = djs.interval_coverage(vint, np.maximum(ends - p, starts), ends)
    bar_db = np.array([20 * np.log10(np.sqrt(np.mean(mono[int(s * sr):max(int(s * sr) + 1, int(e * sr))] ** 2)) + 1e-9)
                       for s, e in zip(starts, ends)])
    bar_rel = bar_db - float(np.median(bar_db))
    sec_energy = djs.section_energy(bar_db, bar_vocal, sections, len(downbeats))

    # ---- key ------------------------------------------------------------------
    key = djs.combine_key_estimates(key_estimates(prov, mono, sr))

    alt = {name: {"bpm": f["bpm"], "beat_f": f["beat_f"], "downbeat_f": f["downbeat_f"]} for name, f in fits.items()}
    return {
        "schema_version": djs.SCHEMA_VERSION, "slug": slug, "artist": t.get("artist"), "title": t.get("title"),
        "track_id": t["track_id"], "duration_s": duration, "sample_rate": sr,
        "tempo": {"bpm": grid["fit"]["bpm"], "provider": grid["provider"], "confidence": grid["confidence"],
                  "beat_f": grid["fit"]["beat_f"], "downbeat_f": grid["fit"]["downbeat_f"],
                  "residual_ms_median_abs": grid["fit"]["residual_ms_median_abs"],
                  "drift_pct_max": grid["fit"]["drift_pct_max"], "bar_offset_beats": grid["fit"]["bar_offset_beats"],
                  "phase_s": grid["fit"]["phase_s"], "period_s": grid["fit"]["period_s"],
                  "providers": alt, "agreement": grid["agreement"],
                  "octave_ambiguous": grid["octave_ambiguous"], "octave_votes": grid["octave_votes"],
                  "allin1_bpm": prov.get("allin1", {}).get("bpm"),
                  "raw_beat_this_ibi_cv": float(np.std(np.diff(raw["beat_this"][0])) / np.mean(np.diff(raw["beat_this"][0])))},
        "grid": {"downbeats_s": [round(float(x), 4) for x in downbeats], "n_bars": int(len(downbeats)),
                 "note": "constant-tempo grid fitted to the chosen provider; beats = phase + k*period"},
        "sections_source": sec_source, "sections": sections, "section_energy": sec_energy,
        "phrases": phrases,
        "key": key,
        "vocal_regions": djs.vocal_regions(bar_vocal, downbeats, ends),
        "bars": {"vocal": [round(float(x), 3) for x in bar_vocal],
                 "vocal_last_beat": [round(float(x), 3) for x in last_beat],
                 "energy_db_rel": [round(float(x), 2) for x in bar_rel]},
        "inputs": {"canonical_wav": str(wav), "structure": str(localize(t["structure_path"])),
                   "vocals": str(localize(t["vocals_path"])), "providers": str(prov_path) if prov else None},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalog", default=str(ROOT / "runs/pool_v12/catalog_ready.json"))
    ap.add_argument("--out", default=str(ROOT / "runs/dj_plan_v1"))
    ap.add_argument("--providers", default=None, help="default: <out>/providers")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out)
    prov_dir = Path(args.providers) if args.providers else out / "providers"
    (out / "tracks").mkdir(parents=True, exist_ok=True)
    cat = load_json(Path(args.catalog))
    tracks = {}
    for t in cat["tracks"]:
        if t.get("status") != "ready":
            continue
        print(f"analysing {t['slug']}", flush=True)
        a = analyse_track(t, prov_dir)
        tracks[t["slug"]] = a
        (out / "tracks" / f"{t['slug']}.json").write_text(dump(a), encoding="utf-8")
    pairs = [djs.plan_pair(tracks[a], tracks[b], top_k=args.top_k)
             for a, b in itertools.permutations(tracks, 2)]
    ranked = djs.rank_pairs(pairs)
    (out / "pairs.json").write_text(dump({"schema_version": djs.SCHEMA_VERSION,
                                           "catalog": args.catalog, "pairs": ranked}), encoding="utf-8")
    table = [{"slug": s, "bpm": round(a["tempo"]["bpm"], 2), "tempo_provider": a["tempo"]["provider"],
              "tempo_confidence": a["tempo"]["confidence"], "key": a["key"]["key"], "camelot": a["key"]["camelot"],
              "key_confidence": round(a["key"]["confidence"], 2), "key_level": a["key"]["level"],
              "sections_source": a["sections_source"], "n_bars": a["grid"]["n_bars"]} for s, a in tracks.items()]
    counts = {}
    for p in pairs:
        counts[p["compatibility"]["status"]] = counts.get(p["compatibility"]["status"], 0) + 1
    best = [{"a": p["a"], "b": p["b"], "status": p["compatibility"]["status"],
             "pair_reasons": p["compatibility"]["reasons"], **{k: v for k, v in p["plans"][0].items()}}
            for p in ranked if p["plans"]]
    top, per_pair = [], {}
    for p in ranked:
        for plan in p["plans"]:
            if per_pair.get((p["a"], p["b"]), 0) >= 2:
                break
            top.append({"a": p["a"], "b": p["b"], "status": p["compatibility"]["status"], **plan})
            per_pair[(p["a"], p["b"])] = per_pair.get((p["a"], p["b"]), 0) + 1
    status_rank = {"compatible": 0, "compatible_uncertain": 1, "incompatible": 2}
    top.sort(key=lambda x: (status_rank[x["status"]], -x["score"]))
    summary = {"schema_version": djs.SCHEMA_VERSION, "catalog": args.catalog, "tracks": table,
               "ordered_pairs": len(pairs), "pair_status_counts": counts,
               "compatible_pairs": [f"{p['a']}->{p['b']}" for p in ranked
                                    if p["compatibility"]["compatible"]],
               "best_plan_per_pair": best,
               "top_plans": top[:6],
               "top_plans_note": "at most 2 plans per ordered pair; compatible pairs first, then score",
               "rules": {"tempo_max_stretch": djs.TEMPO_MAX_STRETCH, "long_blend_stretch": djs.TEMPO_LONG_BLEND_STRETCH,
                         "phrase_levels": djs.PHRASE_LEVELS, "clash_tolerated_bars": djs.CLASH_TOLERATED_BARS,
                         "key_confident": djs.KEY_CONFIDENT,
                         "source": "conoscenza_mix/_meta/REVISIONE_DJ_2026-10-20.md (a priori, not fitted)"}}
    (out / "summary.json").write_text(dump(summary), encoding="utf-8")
    for row in table:
        print(row)
    print(counts)


if __name__ == "__main__":
    main()
