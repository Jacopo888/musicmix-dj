"""Evaluate pretrained cue support on a frozen, vocal-filtered candidate pool.

Does not retrain or reconstruct a model. Scores/rendering are held fixed.
Inference results must be canonical-bound and normalized by the model adapter.
No human judgments are used to tune tolerance, model scores or pair weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
from html import escape
import json
from pathlib import Path
import statistics
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicmix.audio import load_audio
from musicmix.audition_history import exposure
from musicmix.expectation_pipeline import candidate_from_dict
from musicmix.ingest import sha256_file
from musicmix.learned_opportunities import filter_supported_proposals, select_diverse, validate_opportunities
from musicmix.mix_quality import evaluate_mix_quality
from musicmix.musical_preview import render_handoff
from musicmix.paths import localize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cues", required=True, type=Path)
    parser.add_argument("--catalog", type=Path, default=ROOT / "runs/pool_v8/catalog.json")
    parser.add_argument("--frozen-run", type=Path, default=ROOT / "runs/pool_experiment_v8_final")
    parser.add_argument("--provided-run", type=Path, help="Compare exact model-cue search against the all-beat control")
    latest_history = ROOT / "runs/audition_history_v10.json"
    parser.add_argument("--history", type=Path, default=latest_history if latest_history.is_file() else ROOT / "runs/audition_history_with_proposals.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory to preserve frozen artifacts")
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    registry = json.loads(args.cues.read_text(encoding="utf-8"))
    history = json.loads(args.history.read_text(encoding="utf-8"))
    frozen_manifest = json.loads((args.frozen_run / "manifest.json").read_text(encoding="utf-8"))
    if frozen_manifest["inputs"]["catalog_sha256"] != sha256_file(args.catalog):
        raise ValueError("Frozen search does not belong to this canonical catalog")
    if frozen_manifest["parameters"]["cue_policy"] != "beats":
        raise ValueError("Comparison requires the full-beat baseline")
    frozen_search_path = args.frozen_run / "search.json.gz"
    expected_frozen_search_sha256 = frozen_manifest.get("search_sha256")
    if (expected_frozen_search_sha256 is not None
            and expected_frozen_search_sha256 != sha256_file(frozen_search_path)):
        raise ValueError("Frozen baseline search changed")
    with gzip.open(frozen_search_path, "rt", encoding="utf-8") as handle:
        pool = json.load(handle)
    by_id = {entry["track_id"]: entry for entry in catalog["tracks"]}
    opportunities, beat_steps = {}, {}
    registry_ids = {entry["track_id"] for entry in registry["tracks"]}
    if len(registry_ids) != len(registry["tracks"]):
        raise ValueError("Duplicate cue-provider identity")
    if registry_ids != set(by_id):
        raise ValueError("Cue registry must cover exactly the canonical catalog")
    for prediction in registry["tracks"]:
        identity = prediction["track_id"]
        if identity not in by_id:
            raise ValueError("Unknown cue-provider track")
        evidence = prediction.get("inference_evidence", {})
        if evidence.get("canonical_sha256") != identity:
            raise ValueError("Per-track inference must record the canonical WAV hash")
        evidence_path = localize(evidence.get("artifact_path", ""))
        if not evidence_path.is_file() or sha256_file(evidence_path) != evidence.get("artifact_sha256"):
            raise ValueError("Original per-track inference artifact missing or changed")
        opportunities[identity] = validate_opportunities(prediction, by_id[identity])
    for identity, entry in by_id.items():
        structure_path = localize(entry["structure_path"])
        if sha256_file(structure_path) != entry["structure_sha256"]:
            raise ValueError("Beat grid changed since catalog")
        structure = json.loads(structure_path.read_text(encoding="utf-8"))
        for field in ("track_id", "sample_rate", "duration_samples"):
            if structure.get(field) != entry[field]:
                raise ValueError(f"Structure {field} does not match canonical track")
        beats = [b["sample"] for b in structure["beats"]]
        steps = [right - left for left, right in zip(beats, beats[1:]) if right > left]
        if not steps:
            raise ValueError("Cannot derive timing tolerance without observed beats")
        beat_steps[identity] = statistics.median(steps)
    fresh = []
    heard_rejected = 0
    for proposal in pool["proposals"]:
        c = candidate_from_dict(proposal["candidate"])
        flags = exposure(c, history["regions"], 8.0, sample_rate=by_id[c.track_a]["sample_rate"])
        if flags["outgoing_region_heard"] or flags["incoming_region_heard"]:
            heard_rejected += 1
        else:
            fresh.append(proposal)
    # Fixed before results: primary half-beat; quarter/one-beat are diagnostics.
    sweeps = {}
    primary = []
    for fraction in (.25, .5, 1.0):
        tolerance = {identity: round(step * fraction) for identity, step in beat_steps.items()}
        retained, counts = filter_supported_proposals(fresh, opportunities, tolerance)
        sweeps[str(fraction)] = dict(counts, tolerance_samples=tolerance,
                                     ordered_pair_counts=dict(Counter(p["pair_id"] for p in retained)))
        if fraction == .5:
            primary = retained
    direct = None
    if args.provided_run:
        direct_manifest_path = args.provided_run / "manifest.json"
        direct_manifest = json.loads(direct_manifest_path.read_text(encoding="utf-8"))
        if (direct_manifest["inputs"]["catalog_sha256"] != sha256_file(args.catalog)
                or direct_manifest["inputs"].get("opportunities_sha256") != sha256_file(args.cues)
                or direct_manifest["parameters"]["cue_policy"] != "provided"):
            raise ValueError("Exact model-cue search inputs do not match this pilot")
        direct_path = args.provided_run / "search.json.gz"
        if sha256_file(direct_path) != direct_manifest["search_sha256"]:
            raise ValueError("Exact model-cue search changed")
        with gzip.open(direct_path, "rt", encoding="utf-8") as handle:
            direct_pool = json.load(handle)
        direct_fresh = []
        for proposal in direct_pool["proposals"]:
            c = candidate_from_dict(proposal["candidate"])
            flags = exposure(c, history["regions"], 8.0, sample_rate=by_id[c.track_a]["sample_rate"])
            if not flags["outgoing_region_heard"] and not flags["incoming_region_heard"]:
                direct_fresh.append(proposal)
        primary, support_counts = filter_supported_proposals(direct_fresh, opportunities,
                        {identity: round(.08 * entry["sample_rate"]) for identity, entry in by_id.items()})
        direct = {"manifest": str(direct_manifest_path.resolve()), "manifest_sha256": sha256_file(direct_manifest_path),
                  "proposals": len(direct_pool["proposals"]), "unheard": len(direct_fresh),
                  "support": support_counts, "support_tolerance_seconds": .08,
                  "meaning": "Actual model coordinates; tolerance is the unchanged allowed vocal-alignment shift, not grid snapping."}
    rates = {identity: entry["sample_rate"] for identity, entry in by_id.items()}
    if len(set(rates.values())) != 1:
        raise ValueError("This cached-pool pilot requires identical canonical sample rates")
    provider_id = registry.get("provider_id", "cue_detr")
    if provider_id not in {"cue_detr", "linkseg"}:
        raise ValueError("Unknown model provider identifier")
    model_system = provider_id + ("_direct" if direct is not None else "_supported")
    selection = {"all_beats_control": select_diverse(fresh, maximum=args.maximum, sample_rates=rates),
                 model_system: select_diverse(primary, maximum=args.maximum, sample_rates=rates)}
    args.output.mkdir(parents=True)
    (args.output / "audio").mkdir()
    canonical = {}
    artifacts = []
    def track(identity):
        if identity not in canonical:
            item = load_audio(by_id[identity]["canonical_path"])
            if item.track_id != identity:
                raise ValueError("Canonical audio identity changed")
            canonical[identity] = item
        return canonical[identity]
    for system, selected in selection.items():
        for number, proposal in enumerate(selected, 1):
            c = candidate_from_dict(proposal["candidate"])
            artifact = render_handoff(track(c.track_a), track(c.track_b), c,
                                      args.output / "audio" / f"{system}_{number:02d}_{c.candidate_id}.wav")
            qa = evaluate_mix_quality(load_audio(artifact["path"]), artifact["join_sample"])
            artifacts.append(dict(artifact, system=system, candidate=proposal["candidate"],
                                  learned_opportunity_support=proposal.get("learned_opportunity_support"),
                                  technical_quality=qa))
            (args.output / f"proposal_{system}_{number:02d}.json").write_text(
                json.dumps(proposal, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {"schema_version": "musicmix.pretrained-cue-pilot.1", "run_id": str(uuid.uuid4()),
                "status": "rendered_diagnostic_pilot" if primary else "no_primary_supported_candidates",
                "tracks": catalog["tracks"], "artifacts": artifacts,
                "opportunity_provider": provider_id,
                "inputs": {"catalog": str(args.catalog.resolve()), "catalog_sha256": sha256_file(args.catalog),
                           "cues": str(args.cues.resolve()), "cues_sha256": sha256_file(args.cues),
                           "search_sha256": sha256_file(frozen_search_path),
                           "history_sha256": sha256_file(args.history)},
                "parameters": {"primary_tolerance_beats": None if direct is not None else .5, "diagnostic_tolerances_beats": [.25, .5, 1.0],
                               "history_tolerance_seconds": 8, "spacing_seconds": 8, "role_cap": 2,
                               "pair_score": "unchanged directional baseline", "render": "unchanged anchor_handoff_v1"},
                "pool_counts": {"frozen_proposals": len(pool["proposals"]), "new_history_rejected": heard_rejected,
                                "unheard_proposals": len(fresh), "primary_supported": len(primary)},
                "selection_overlap_candidate_ids": sorted(
                    {p["candidate"]["candidate_id"] for p in selection["all_beats_control"]}
                    & {p["candidate"]["candidate_id"] for p in selection[model_system]}),
                "exact_model_search": direct,
                "sensitivity": sweeps, "selected": selection,
                "limitations": ["Cue model supplies single-track opportunities, not cross-song resolution.",
                                "Primary tolerance was fixed before results; sensitivity is not quality validation.",
                                "All-beat support sweep is restricted to the frozen grid; exact_model_search, when present, evaluates native model timestamps separately.",
                                "Both arms use the same new greedy diversity selector; not a replay of the old V8 listening selection.",
                                "Model support is tested at actual rendered cue, not nominal beat.",
                                "ASR gaps and technical QA do not certify complete sung words.",
                                "No new subjective reward, training, or SOTA claim."]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Pilot pretrained V10", "", "Confronto diagnostico, non miglioramento percettivo dimostrato.", "",
             "| Sistema | Uscita A (s) | Ingresso B (s) | File |", "|---|---|---|---|"]
    for a in artifacts:
        c = a["candidate"]
        lines.append(f"| {a['system']} | {by_id[c['track_a']]['title']} {c['cue_out_sample']/rates[c['track_a']]:.3f} | "
                     f"{by_id[c['track_b']]['title']} {c['cue_in_sample']/rates[c['track_b']]:.3f} | "
                     f"[Audio](audio/{Path(a['path']).name}) |")
    (args.output / "TIMESTAMPS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cards = []
    for artifact in artifacts:
        c = artifact["candidate"]
        title = (f"{artifact['system']}: {by_id[c['track_a']]['title']} "
                 f"({c['cue_out_sample']/rates[c['track_a']]:.3f}s) → {by_id[c['track_b']]['title']} "
                 f"({c['cue_in_sample']/rates[c['track_b']]:.3f}s)")
        cards.append(f'<section><h2>{escape(title)}</h2><audio controls preload="metadata" '
                     f'src="audio/{escape(Path(artifact["path"]).name, quote=True)}"></audio>'
                     '<p><button onclick="seek(this,-5)">−5 s</button> '
                     '<button onclick="seek(this,5)">+5 s</button></p></section>')
    page = ('<!doctype html><html lang="it"><meta charset="utf-8"><title>MusicMix V10</title>'
            '<style>body{font:17px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;'
            'background:#14181d;color:#edf2f7}section{padding:20px 0;border-bottom:1px solid #556}'
            'h2{font-size:18px}audio{width:100%}button{padding:8px 16px}</style>'
            '<h1>MusicMix V10 — cue preaddestrati</h1><p>Esportazione diagnostica. '
            'Raccordo a10s, stessi punteggi e renderer della baseline. Il supporto del modello '
            'non certifica risoluzione musicale o parole integre. Nessun voto richiesto.</p>'
            + ''.join(cards) + '<script>function seek(b,d){const a=b.closest("section").querySelector("audio");'
            'a.currentTime=Math.max(0,Math.min(Number.isFinite(a.duration)?a.duration:Infinity,a.currentTime+d))}'
            '</script></html>')
    (args.output / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "counts": manifest["pool_counts"], "artifacts": len(artifacts)}), flush=True)


if __name__ == "__main__":
    main()
