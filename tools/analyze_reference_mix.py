"""Analizza un mix di riferimento: allineamento mix → brani, cue, sovrapposizioni, curve, tecnica.

Uso:
  python tools/analyze_reference_mix.py MIX.wav --tracklist tracklist.json --out runs/reference_mix_v1/<nome>.json
  python tools/analyze_reference_mix.py runs/dj_set_v3_bridge/set.wav --from-set runs/dj_set_v3_bridge/set.json --out ...

Tracklist (JSON lista o {"tracks": [...]}, oppure CSV) con, per brano: ``slug`` e uno fra
``audio`` (file audio) e ``analysis`` (JSON ``musicmix.dj_select.1``, da cui si prende
``inputs.canonical_wav``); facoltativi ``artist``, ``title``. I brani senza audio in libreria
finiscono in ``missing`` (vedi la nota della biblioteca su cosa succede con i brani mancanti).
L'ordine della tracklist NON è usato dall'allineamento: l'ordine è ricostruito dal mix.

Uscita ``musicmix.reference-mix.1``. Le feature (battiti Beat This!, mel, croma, MFCC) sono
in cache in ``--cache`` (npz piccoli, per hash del file audio).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicmix.mix_alignment import AlignConfig, TrackSpec, analyze_mix, default_tracker, librosa_beats, load_tracklist  # noqa: E402
from musicmix.paths import localize  # noqa: E402


def specs_from_rows(rows: list[dict]) -> list[TrackSpec]:
    out = []
    for r in rows:
        analysis = None
        if r.get("analysis"):
            ap = localize(r["analysis"])
            if Path(ap).is_file():
                analysis = json.loads(Path(ap).read_text(encoding="utf-8"))
        audio = r.get("audio")
        if not audio and analysis:
            audio = (analysis.get("inputs") or {}).get("canonical_wav")
        audio = str(localize(audio)) if audio else ""
        out.append(TrackSpec(slug=r["slug"], audio=audio, analysis=analysis,
                             artist=r.get("artist") or (analysis or {}).get("artist"),
                             title=r.get("title") or (analysis or {}).get("title")))
    return out


def rows_from_set(set_json: Path, include_pool: bool = False) -> list[dict]:
    d = json.loads(set_json.read_text(encoding="utf-8"))
    slugs = [t["slug"] for t in d["tracklist"]]
    if include_pool:
        slugs = sorted(set(slugs) | set(d["sources"]))
    else:
        slugs = sorted(slugs)  # ordine alfabetico: l'ordine va ritrovato dal mix
    return [{"slug": s, "analysis": d["sources"][s].get("analysis")} for s in slugs]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mix")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tracklist")
    g.add_argument("--from-set", help="set.json di una nostra run (tracklist dai sources; ordine ignorato)")
    ap.add_argument("--include-pool", action="store_true", help="con --from-set: aggiungi anche i brani del catalogo non suonati (distrattori)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=str(ROOT / "runs" / "reference_mix_v1" / "cache"))
    ap.add_argument("--tracker", choices=["beat_this", "librosa"], default="beat_this")
    ap.add_argument("--key-shifts", default="0", help="es. 0 oppure -1,0,1 oppure all")
    ap.add_argument("--audible-db", type=float, default=None)
    args = ap.parse_args(argv)

    cfg = AlignConfig()
    if args.key_shifts == "all":
        cfg.key_shifts = tuple(range(-5, 7))
    else:
        cfg.key_shifts = tuple(int(x) for x in args.key_shifts.split(","))
    if args.audible_db is not None:
        cfg.audible_db = args.audible_db
    rows = load_tracklist(args.tracklist) if args.tracklist else rows_from_set(Path(args.from_set), args.include_pool)
    specs = specs_from_rows(rows)
    tracker = default_tracker() if args.tracker == "beat_this" else librosa_beats
    if args.tracker == "librosa":
        tracker.info = {"model": "librosa.beat.beat_track"}  # type: ignore[attr-defined]
    t0 = time.time()
    res = analyze_mix(args.mix, specs, cfg, tracker=tracker, cache_dir=Path(args.cache), log=lambda m: print(m, flush=True))
    res["elapsed_s"] = round(time.time() - t0, 1)
    res["inputs"] = {"tracklist": args.tracklist, "from_set": args.from_set, "include_pool": args.include_pool}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"scritto {out} in {res['elapsed_s']} s; ordine: {res['order']}")
    for tr in res["transitions"]:
        print(f"  {tr['a']} -> {tr['b']}: {tr['technique']}, switch {tr['mix_s']['switch']:.2f} s, sovrapp. {tr['overlap_beats']:.1f} batt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
