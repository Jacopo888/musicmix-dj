#!/usr/bin/env python3
"""Effetto della regola dell'808 di ``dj_form.3`` (``kick_peak_on``) sulle forme in cache di una run.

Per ogni forma (``.cache/<run>/form/<slug>.json``, scritta da ``run_dj_set_v2_books.form_for`` con dj_form.3) rifà le
etichette con la regola della cassa di dj_form.2 (``kick_peak_on=None``) sugli stessi livelli per battuta e conta
etichette cambiate, battute con cassa solo dal picco, core, drop e breakdown. Solo lettura delle cache.

    python tools/form_kick_808_effect.py .cache/dj_set_v3_events_plan/form \\
        --out runs/dj_set_v3_events_plan/kick_808_effect.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicmix import dj_form as F  # noqa: E402


def effect(form_dir: Path) -> dict:
    rows = []
    for f in sorted(Path(form_dir).glob("*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        fm = doc["form"]
        if "kick_peak" not in fm["layers"]:
            rows.append({"slug": f.stem, "skipped": "livelli senza kick_peak (cache precedente a dj_form.3)"})
            continue
        old = F.form_segments(fm["layers"], {"kick_peak_on": None})
        new = F.form_segments(fm["layers"])
        changed = [i for i, (a, b) in enumerate(zip(old["labels"], new["labels"])) if a != b]
        rows.append({"slug": f.stem, "bars": len(new["labels"]), "labels_changed": len(changed),
                     "kick_bars_old": sum(old["active"]["kick"]), "kick_bars_new": sum(new["active"]["kick"]),
                     "kick_peak_only_bars": sum(new["active"]["kick_peak_only"]),
                     "core_old": old["core_status"], "core_new": new["core_status"], "no_core_reason_old": old["no_core_reason"],
                     "drops_old": old["drops"], "drops_new": new["drops"],
                     "breakdowns_old": old["breakdowns"], "breakdowns_new": new["breakdowns"],
                     "segments_old": F.form_summary(old)["segments"], "segments_new": F.form_summary(new)["segments"],
                     "source": fm["layers"].get("source")})
    ok = [r for r in rows if "skipped" not in r]
    tot = {"tracks": len(ok), "bars": sum(r["bars"] for r in ok), "labels_changed": sum(r["labels_changed"] for r in ok),
           "tracks_with_changes": sum(1 for r in ok if r["labels_changed"]),
           "kick_peak_only_bars": sum(r["kick_peak_only_bars"] for r in ok),
           "drops_old": sum(len(r["drops_old"]) for r in ok), "drops_new": sum(len(r["drops_new"]) for r in ok),
           "breakdowns_old": sum(len(r["breakdowns_old"]) for r in ok), "breakdowns_new": sum(len(r["breakdowns_new"]) for r in ok),
           "core_changed": [r["slug"] for r in ok if r["core_old"] != r["core_new"]]}
    return {"schema_version": "musicmix.form_kick_808_effect.1", "form_version": F.FORM_VERSION, "form_dir": str(form_dir),
            "kick_peak_on": F.PARAMS["kick_peak_on"][0], "totals": tot, "tracks": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("form_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    doc = effect(Path(a.form_dir))
    print(json.dumps(doc["totals"], ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
