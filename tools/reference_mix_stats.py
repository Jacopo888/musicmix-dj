"""Statistiche apprese da mix di riferimento → prior del pianificatore (``musicmix.reference-priors.1``).

Sorgenti (ognuna in un *dominio* con il suo peso, mai mescolate senza dichiararlo):

* ``--mix DOMINIO:PESO:file.json`` — uscite ``musicmix.reference-mix.1`` di
  ``tools/analyze_reference_mix.py`` (es. ``rap_user:1.0:runs/reference_mix_v1/x.json``):
  tecnica, sovrapposizione per tecnica, cue d'ingresso/uscita rispetto alla frase (battuta
  nella sezione modulo 16), distanza dal downbeat, sezione, frazione del brano, tempo.
* ``--raveform raveform.zip`` — metadati pubblici Raveform (Kim et al., TISMIR 2026; CC BY 4.0;
  allineamenti mix–brano a livello di battito, mix-in/mix-out, struttura annotata a mano
  per 1.423 brani). Nessun audio. Dominio ``electronic_raveform`` (house/techno/trance/DnB:
  **non rap**, peso di default 0,15) e ``hiphop_tagged_raveform`` (mix con tag MixesDB
  ``Hip Hop``: solo sovrapposizioni e tempo, peso di default 0,3). I mix-in/out di Raveform
  sono gli estremi del tratto allineato (Kim 2020), non l'inizio udibile: la sovrapposizione
  è quindi la "transizione" nel senso di Kim, tipicamente più corta di quella udibile.

Uscita: ``--out`` (default ``runs/reference_mix_v1/reference_priors.json``). Per attivarla
nel pianificatore copiarla in ``configs/reference_priors.json`` (assente = default invariato).
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicmix.reference_priors import SCHEMA, transition_features  # noqa: E402

BINS = {
    "overlap_beats": [-64, -4, -0.5, 0.5, 2, 4, 8, 16, 32, 64, 128, 512],
    "entry_beats_from_downbeat": [-2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2.0001],
    "exit_beats_from_downbeat": [-2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2.0001],
    "entry_track_fraction": [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001],
    "exit_track_fraction": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001],
    "tempo_change_pct": [-50, -10, -6, -3, -1, 1, 3, 6, 10, 50],
    "play_beats": [0, 32, 64, 96, 128, 192, 256, 384, 512, 4096],
    "entry_bars_to_chorus": [-0.5, 0.5, 2, 4, 8, 16, 32, 128],
    "exit_bars_to_section_end": [-0.5, 0.5, 2, 4, 8, 16, 128],
    "b_drop_vs_a_exit_beats": [-512, -64, -32, -16, -4, 4, 16, 32, 64, 512],
}


def hist(values, edges) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    v = np.clip(v, edges[0], edges[-1] - 1e-9)
    c, _ = np.histogram(v, bins=edges)
    return {"bins": edges, "counts": [int(x) for x in c], "n": int(len(v)),
            "quantiles": None if not len(v) else {q: round(float(np.percentile(v, p)), 3) for q, p in (("p10", 10), ("p25", 25), ("p50", 50), ("p75", 75), ("p90", 90))}}


def counts(values) -> dict:
    c = collections.Counter(str(x) for x in values if x is not None)
    return {"counts": dict(sorted(c.items())), "n": int(sum(c.values()))}


def distributions(rows: list[dict], extra_play: list | None = None) -> dict:
    out = {}
    for f in ("technique", "entry_bar_in_phrase16", "exit_bar_in_phrase16", "entry_section", "exit_section",
              "exit_prev_section", "base_continuous"):
        vals = [r.get(f) for r in rows]
        if any(v is not None for v in vals):
            out[f] = counts(vals)
    for f, edges in BINS.items():
        if f == "play_beats":
            continue
        vals = [r.get(f) for r in rows]
        if any(v is not None for v in vals):
            out[f] = hist(vals, edges)
    if extra_play:
        out["play_beats"] = hist(extra_play, BINS["play_beats"])
    # sovrapposizione per tecnica
    by_t = collections.defaultdict(list)
    for r in rows:
        if r.get("technique") and r.get("overlap_beats") is not None:
            by_t[r["technique"]].append(r["overlap_beats"])
    if by_t:
        out["overlap_beats_by_technique"] = {t: hist(v, BINS["overlap_beats"]) for t, v in sorted(by_t.items())}
    ent = [r for r in rows if r.get("entry_section")]
    if ent:
        out["entry_on_section_start_rate"] = round(float(np.mean([r.get("entry_bar_in_phrase16") == 0 for r in ent])), 3)
    bc = [r["base_continuous"] for r in rows if r.get("base_continuous") is not None]
    if bc:
        out["base_continuous_rate"] = round(float(np.mean(bc)), 3)
    ch = [r["entry_bars_to_chorus"] for r in rows if r.get("entry_bars_to_chorus") is not None]
    if ch:
        out["entry_on_chorus_rate"] = round(float(np.mean([abs(x) <= 0.5 for x in ch])), 3)
    dr = [r["b_drop_vs_a_exit_beats"] for r in rows if r.get("b_drop_vs_a_exit_beats") is not None]
    if dr:
        out["b_drop_within_4_beats_of_a_exit_rate"] = round(float(np.mean([abs(x) <= 4 for x in dr])), 3)
        out["b_drop_before_a_exit_rate"] = round(float(np.mean([x < -4 for x in dr])), 3)
    return out


# ---------------------------------------------------------------------- nostri mix analizzati


def rows_from_reference(path: Path) -> list[dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("schema_version") != "musicmix.reference-mix.1":
        raise ValueError(f"{path}: non è musicmix.reference-mix.1")
    rows = []
    for tr in d["transitions"]:
        if tr.get("adjacent") is False or tr.get("technique") == "unknown":
            continue  # coppia separata da un brano mancante: non è una transizione osservata
        r = transition_features(tr)
        if r.get("tempo_change_pct") is None:  # file prodotti prima del cambio di nome del campo
            st = (tr.get("tempo") or {}).get("b_stretch_pct")
            r["tempo_change_pct"] = None if st is None else st
        r["entry_chorus"] = (r.get("entry_section") == "chorus")
        rows.append(r)
    return rows


# ---------------------------------------------------------------------- Raveform


def _section_at(sections, t):
    for s in sections:
        if s["start"] <= t < s["end"]:
            return s
    return None


def raveform_rows(zpath: Path, log=print) -> tuple[dict, dict]:
    z = zipfile.ZipFile(zpath)
    mixes = [json.loads(x) for x in z.read("raveform/mixes.jsonl").decode("utf-8").splitlines() if x.strip()]
    segs = json.loads(z.read("raveform/structures/segments.json"))
    ann = {s["id"]: s for s in segs}
    beat_files = {n.split("/")[-1].split(".", 1)[1].rsplit(".beat.csv", 1)[0]: n
                  for n in z.namelist() if n.startswith("raveform/structures/beats/") and n.endswith(".beat.csv")}
    beat_cache: dict = {}

    def beats_of(tid):
        if tid not in beat_cache:
            n = beat_files.get(tid)
            if not n:
                beat_cache[tid] = None
            else:
                lines = z.read(n).decode("utf-8").splitlines()[1:]
                arr = [(float(a), int(b), c) for a, b, c in (ln.split(",") for ln in lines if ln.strip())]
                beat_cache[tid] = arr
        return beat_cache[tid]

    def position(tid, t):
        a = ann.get(tid)
        bt = beats_of(tid)
        if not a or not bt:
            return {}
        times = np.asarray([x[0] for x in bt])
        i = int(np.clip(np.searchsorted(times, t), 0, len(times) - 1))
        if i > 0 and abs(times[i - 1] - t) < abs(times[i] - t):
            i -= 1
        sec = _section_at(a["sections"], t)
        # battuta nella sezione: downbeat contati dall'inizio della sezione
        dbs = [k for k, x in enumerate(bt) if x[1] == 1]
        db_times = times[dbs] if dbs else np.zeros(0)
        out = {"track_fraction": t / a["duration"] if a.get("duration") else None}
        if sec:
            k = a["sections"].index(sec)
            if k > 0:
                out["prev_section"] = a["sections"][k - 1]["name"]
            bar_len = 4 * (float(np.median(np.diff(times))) if len(times) > 2 else 0.5)
            out["bars_to_section_end"] = (sec["end"] - t) / bar_len
        if len(db_times):
            j = int(np.argmin(np.abs(db_times - t)))
            bpb = float(np.median(np.diff(times))) if len(times) > 2 else 0.5
            out["beats_from_downbeat"] = (t - db_times[j]) / bpb
            if sec:
                n_before = int(np.sum((db_times >= sec["start"] - 0.05) & (db_times <= db_times[j] + 1e-6))) - 1
                out["bar_in_phrase16"] = max(n_before, 0) % 16
                out["section"] = sec["name"]
        return out

    genre = {m["id"]: set(m.get("genres") or []) for m in mixes}
    by_domain = {"electronic_raveform": [], "hiphop_tagged_raveform": []}
    play = {"electronic_raveform": [], "hiphop_tagged_raveform": []}
    n_mix = collections.Counter()
    skipped = collections.Counter()
    for m in mixes:
        name = f"raveform/alignments/{m['id']}.align.jsonl"
        try:
            raw = z.read(name).decode("utf-8")
        except KeyError:
            continue
        best = {}
        for ln in raw.splitlines():
            if not ln.strip():
                continue
            # i campi utili precedono "wp": evitiamo di decodificare il percorso intero
            head = ln.split(',"wp"', 1)[0] + "}"
            r = json.loads(head)
            if r.get("match_rate", 0) < 0.1:
                continue
            if r["track_id"] not in best or r["match_rate"] > best[r["track_id"]]["match_rate"]:
                best[r["track_id"]] = r
        dom = "hiphop_tagged_raveform" if "Hip Hop" in genre[m["id"]] else "electronic_raveform"
        n_mix[dom] += 1
        ids = [t.get("id") for t in m.get("tracklist") or []]
        for r in best.values():
            play[dom].append(r["mixout_beat_mix"] - r["mixin_beat_mix"])
        for pa, nb in zip(ids[:-1], ids[1:]):
            if pa not in best or nb not in best:
                continue
            A, B = best[pa], best[nb]
            ov = A["mixout_beat_mix"] - B["mixin_beat_mix"]
            if ov <= 0:  # Raveform conta una transizione solo se i tratti allineati si sovrappongono
                skipped[dom] += 1
                continue
            rate_b = B["matched_time_track"] / B["matched_time_mix"] if B.get("matched_time_mix") else None
            pe = position(nb, B["mixin_time_track"])
            px = position(pa, A["mixout_time_track"])
            # dove cade il primo drop di B rispetto all'uscita di A (nel tempo del mix, in battiti)
            drop_vs_exit = None
            if nb in ann and rate_b and B["mixout_beat_mix"] > B["mixin_beat_mix"]:
                drops = [x["start"] for x in ann[nb]["sections"] if x["name"] == "drop" and x["start"] >= B["mixin_time_track"] - 0.5]
                if drops:
                    bp_mix = B["matched_time_mix"] / (B["mixout_beat_mix"] - B["mixin_beat_mix"])
                    drop_mix = B["mixin_time_mix"] + (drops[0] - B["mixin_time_track"]) / rate_b
                    drop_vs_exit = (drop_mix - A["mixout_time_mix"]) / bp_mix
            by_domain[dom].append({
                "overlap_beats": ov,
                "tempo_change_pct": None if rate_b is None else (rate_b - 1) * 100,
                "entry_bar_in_phrase16": pe.get("bar_in_phrase16"), "entry_section": pe.get("section"),
                "entry_beats_from_downbeat": pe.get("beats_from_downbeat"), "entry_track_fraction": pe.get("track_fraction"),
                "exit_bar_in_phrase16": px.get("bar_in_phrase16"), "exit_section": px.get("section"),
                "exit_beats_from_downbeat": px.get("beats_from_downbeat"), "exit_track_fraction": px.get("track_fraction"),
                "exit_prev_section": px.get("prev_section"), "exit_bars_to_section_end": px.get("bars_to_section_end"),
                "b_drop_vs_a_exit_beats": drop_vs_exit,
            })
    log(f"raveform: mix per dominio {dict(n_mix)}; transizioni {({k: len(v) for k, v in by_domain.items()})}; "
        f"coppie senza sovrapposizione escluse {dict(skipped)}")
    return by_domain, {"play": play, "n_mix": dict(n_mix), "skipped_no_overlap": dict(skipped)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mix", action="append", default=[], help="DOMINIO:PESO:file.json (ripetibile)")
    ap.add_argument("--raveform", help="raveform.zip (metadati CC BY 4.0, nessun audio)")
    ap.add_argument("--electronic-weight", type=float, default=0.15)
    ap.add_argument("--hiphop-tagged-weight", type=float, default=0.3)
    ap.add_argument("--out", default=str(ROOT / "runs" / "reference_mix_v1" / "reference_priors.json"))
    args = ap.parse_args(argv)
    domains: dict = {}
    for spec in args.mix:
        dom, w, path = spec.split(":", 2)
        rows = rows_from_reference(Path(path))
        d = domains.setdefault(dom, {"weight": float(w), "domain": "rap" if "rap" in dom else "other", "sources": [], "_rows": []})
        d["sources"].append({"file": path, "sha256": sha256(Path(path)), "n_transitions": len(rows)})
        d["_rows"] += rows
    if args.raveform:
        rv, meta = raveform_rows(Path(args.raveform))
        src = {"file": Path(args.raveform).name, "sha256": sha256(Path(args.raveform)),
               "url": "https://huggingface.co/datasets/taejunkim/raveform/resolve/main/raveform.zip", "license": "CC BY 4.0"}
        domains["electronic_raveform"] = {
            "weight": args.electronic_weight, "domain": "electronic (house/techno/trance/DnB; NON rap)",
            "evidence": "[M] metadati Raveform; mix point = estremi del tratto allineato (Kim 2020)", "sources": [dict(src, n_mixes=meta["n_mix"].get("electronic_raveform", 0),
                                                                                                             pairs_without_overlap_excluded=meta["skipped_no_overlap"].get("electronic_raveform", 0))],
            "_rows": rv["electronic_raveform"], "_play": meta["play"]["electronic_raveform"]}
        domains["hiphop_tagged_raveform"] = {
            "weight": args.hiphop_tagged_weight, "domain": "mix con tag MixesDB 'Hip Hop' (spesso misti con elettronica)",
            "evidence": "[M] metadati Raveform; pochi mix, struttura dei brani non annotata", "sources": [dict(src, n_mixes=meta["n_mix"].get("hiphop_tagged_raveform", 0),
                                                                                       pairs_without_overlap_excluded=meta["skipped_no_overlap"].get("hiphop_tagged_raveform", 0))],
            "_rows": rv["hiphop_tagged_raveform"], "_play": meta["play"]["hiphop_tagged_raveform"]}
    out_domains = {}
    for name, d in domains.items():
        rows = d.pop("_rows")
        play = d.pop("_play", None)
        d["n_transitions"] = len(rows)
        d["distributions"] = distributions(rows, play)
        out_domains[name] = d
    out = {
        "schema_version": SCHEMA,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generator": "tools/reference_mix_stats.py",
        "smoothing_alpha": 1.0,
        "usage": ("costo additivo -Σ_d w_d·log(p_d(x)/u_d) (musicmix.reference_priors.prior_cost); "
                  "assente configs/reference_priors.json = default invariato. I domini elettronici valgono "
                  "come baseline generica a peso basso, non come misura del rap."),
        "domains": out_domains,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"scritto {args.out}: " + ", ".join(f"{k} n={v['n_transitions']} w={v['weight']}" for k, v in out_domains.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
