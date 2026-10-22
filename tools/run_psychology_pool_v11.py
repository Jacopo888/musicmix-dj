"""Run joint transition decisions on immutable six-song pretrained searches.

This is an auditable engineering policy, not a trained psychological reward.
Prior human judgments are not read. New previews require supported timing and
motion; lack of phrase-closure evidence remains visible after rendering.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import gzip
import json
from pathlib import Path
import shutil
import sys
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from musicmix.audio import load_audio
from musicmix.audition_history import exposure
from musicmix.expectation_pipeline import candidate_from_dict
from musicmix.ingest import sha256_file
from musicmix.intent_render import render_intent
from musicmix.mix_quality import evaluate_mix_quality
from musicmix.paths import localize
from musicmix.phrase_evidence import analyze_phrase_evidence
from musicmix.pooled_trajectory import PooledTrajectory
from musicmix.transition_intent import assess_transition
from musicmix.transition_validation import validate_transition_event


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def validate_document(entry, kind):
    path = localize(entry[f"{kind}_path"])
    if sha256_file(path) != entry[f"{kind}_sha256"]:
        raise ValueError(f"Changed {kind}: {entry['slug']}")
    document = read(path)
    for key in ("track_id", "sample_rate", "duration_samples"):
        if document.get(key) != entry[key]:
            raise ValueError(f"Wrong {kind} identity/clock: {key}")
    return document


def load_search(folder, catalog_hash):
    manifest = read(folder / "manifest.json")
    path = folder / "search.json.gz"
    if manifest["inputs"]["catalog_sha256"] != catalog_hash or sha256_file(path) != manifest["search_sha256"]:
        raise ValueError(f"Frozen search provenance mismatch: {folder}")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        proposals = json.load(stream)["proposals"]
    return proposals, {"path": str(folder.resolve()), "manifest_sha256": sha256_file(folder / "manifest.json"),
                       "search_sha256": manifest["search_sha256"], "count": len(proposals)}


def rank_key(record):
    """Lexicographic, uncalibrated selection; no fabricated quality probability."""
    timing = record["phrase_a"]["boundary_class"], record["phrase_b"]["boundary_class"]
    preferred = sum(item in {"observed_asr_segment_end", "acoustic_clearance_supported"} for item in timing)
    return (preferred, len(record["search_origins"]) > 1,
            record["proposal"].get("score", 0.0), record["candidate_id"])


def diverse(records, maximum, rate=48000, spacing=16):
    chosen, used, roles, pairs = [], [], Counter(), set()
    for record in sorted(records, key=rank_key, reverse=True):
        c = record["proposal"]["candidate"]
        coords = [(c["track_a"], c["cue_out_sample"]), (c["track_b"], c["cue_in_sample"])]
        pair = c["track_a"], c["track_b"]
        if pair in pairs or roles[(pair[0], "out")] >= 2 or roles[(pair[1], "in")] >= 2:
            continue
        if any(t == prior_t and abs(s - prior_s) < spacing * rate for t, s in coords for prior_t, prior_s in used):
            continue
        chosen.append(record)
        used.extend(coords)
        pairs.add(pair)
        roles[(pair[0], "out")] += 1
        roles[(pair[1], "in")] += 1
        if len(chosen) >= maximum:
            break
    return chosen


def natural_reference(track, cue, path, context):
    n = round(context * track.sample_rate)
    signal = np.array(track.samples[cue - n:cue + n], dtype=np.float64, copy=True) * .9
    edge = round(.01 * track.sample_rate)
    signal[:edge] *= np.linspace(0, 1, edge)[:, None]
    signal[-edge:] *= np.linspace(1, 0, edge)[:, None]
    pcm = np.round(signal * 32767).astype("<i2")
    with Path(path).open("xb") as raw:
        with wave.open(raw, "wb") as wav:
            wav.setnchannels(signal.shape[1]); wav.setsampwidth(2)
            wav.setframerate(track.sample_rate); wav.writeframes(pcm.tobytes())
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path), "track_id": track.track_id,
            "start_sample": cue - n, "end_sample": cue + n, "reference_sample": n,
            "gain": .9, "meaning": "Native A continuation; comparison reference, never an automatic quality label"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "runs/pool_v10/catalog_ready.json")
    latest_history = ROOT / "runs/audition_history_v11.json"
    parser.add_argument("--history", type=Path, default=latest_history if latest_history.is_file() else ROOT / "runs/audition_history_v10.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/psychology_v11")
    parser.add_argument("--context", type=float, default=24.0)
    parser.add_argument("--maximum", type=int, default=6)
    parser.add_argument("--extra-search", type=Path, action="append", default=[])
    args = parser.parse_args()
    args.catalog = args.catalog.resolve()
    args.history = args.history.resolve()
    args.output = args.output.resolve()
    args.extra_search = [path.resolve() for path in args.extra_search]
    implementation_paths = [Path(__file__), *[ROOT / "src/musicmix" / f"{name}.py" for name in (
        "phrase_evidence", "transition_intent", "pooled_trajectory", "intent_render", "transition_validation", "expectation")]]
    implementation_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in implementation_paths}
    if args.output.exists():
        raise FileExistsError("Use a new output folder; previous runs are immutable")
    if args.context < 16 or args.maximum < 1:
        raise ValueError("Require context >=16 seconds and positive maximum")
    catalog, history = read(args.catalog), read(args.history)
    if history.get("schema_version") != "musicmix.audition-history.1" or not isinstance(history.get("regions"), list):
        raise ValueError("Invalid audition history schema")
    if history.get("unresolved"):
        raise ValueError("History has unresolved geometries; cannot claim cue novelty")
    entries = {entry["track_id"]: entry for entry in catalog["tracks"]}
    if catalog["ready_count"] != len(entries) or len(entries) < 3:
        raise ValueError("A ready multi-song catalog is required")
    args.output.mkdir(parents=True)
    (args.output / "audio").mkdir()
    for source in implementation_paths:
        snapshot = args.output / "source" / source.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, snapshot)
    write(args.output / "history_input.json", history)
    write(args.output / "catalog_input.json", catalog)
    tracks, vocals, structures, trajectories = {}, {}, {}, {}
    for identity, entry in entries.items():
        print(f"Analyze canonical source: {entry['slug']}", flush=True)
        track = load_audio(entry["canonical_path"])
        if (track.track_id, track.sample_rate, track.duration_samples) != (identity, 48000, entry["duration_samples"]):
            raise ValueError("Canonical audio changed")
        tracks[identity] = track
        vocals[identity] = validate_document(entry, "vocals")
        structures[identity] = validate_document(entry, "structure")
        trajectories[identity] = PooledTrajectory(track, [b["sample"] for b in structures[identity]["beats"]])
    searches = [ROOT / "runs" / name for name in (
        "pretrained_v10_pool6_search", "pretrained_v10_direct6_search", "pretrained_v10_linkseg6_search")]
    searches.extend(args.extra_search)
    proposals, provenance = {}, []
    for folder in searches:
        pool, evidence = load_search(folder, sha256_file(args.catalog))
        provenance.append(evidence)
        for proposal in pool:
            identity = proposal["candidate"]["candidate_id"]
            if identity not in proposals:
                proposals[identity] = {"proposal": proposal, "origins": []}
            proposals[identity]["origins"].append(folder.name)
    print(f"Frozen distinct proposals: {len(proposals)}", flush=True)

    @lru_cache(maxsize=None)
    def profile(identity, cue, side):
        value = trajectories[identity].profile(cue, side=side)
        return value if value["windows"] else None

    @lru_cache(maxsize=None)
    def phrase(identity, cue, direction):
        return analyze_phrase_evidence(vocals[identity], cue, 48000, entries[identity]["duration_samples"], direction)

    counts, actions, intents, pair_counts = Counter(), Counter(), Counter(), {}
    eligible = []
    n = round(args.context * 48000)
    assessment_path = args.output / "assessments.jsonl.gz"
    with gzip.open(assessment_path, "wt", encoding="utf-8") as out:
        for index, item in enumerate(proposals.values(), 1):
            p = item["proposal"]; c = candidate_from_dict(p["candidate"])
            counts["distinct_proposals"] += 1
            heard = exposure(c, history["regions"], 8.0, sample_rate=48000)
            if heard["outgoing_region_heard"] or heard["incoming_region_heard"]:
                counts["history_rejected"] += 1; continue
            a, b = c.track_a, c.track_b
            ca, cb = c.cue_out_sample, c.cue_in_sample
            # Require equal full native contexts on both sides for controls,
            # natural-A and B-before analysis; do not manufacture intro history.
            if not (n <= ca <= tracks[a].duration_samples - n and n <= cb <= tracks[b].duration_samples - n):
                counts["context_rejected"] += 1; continue
            pa, pb = phrase(a, ca, "outgoing"), phrase(b, cb, "incoming")
            assessed = assess_transition(profile(a, ca, "outgoing"), profile(a, ca, "incoming"),
                                         profile(b, cb, "outgoing"), profile(b, cb, "incoming"),
                                         phrase_a=pa, phrase_b=pb, baseline_score=p.get("score")).to_dict()
            record = {"candidate_id": c.candidate_id, "proposal": p, "search_origins": item["origins"],
                      "assessment": assessed, "phrase_a": pa, "phrase_b": pb}
            # Retain decisions, not repeated large chroma payloads in every row.
            row = {k: v for k, v in record.items() if k != "proposal"}
            row["candidate"] = p["candidate"]
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            actions[assessed["action"]] += 1; intents[assessed["intent"]] += 1
            pair = f"{entries[a]['slug']}->{entries[b]['slug']}"
            pair_counts.setdefault(pair, Counter())[assessed["action"]] += 1
            counts["assessed"] += 1
            if assessed["action"] == "execute":
                eligible.append(record)
            if counts["assessed"] % 500 == 0:
                print(json.dumps({"assessed": counts["assessed"], "actions": actions}), flush=True)
    selected = diverse(eligible, args.maximum)
    artifacts, review, references = [], [], []
    for number, record in enumerate(selected, 1):
        c = candidate_from_dict(record["proposal"]["candidate"])
        a, b = tracks[c.track_a], tracks[c.track_b]
        title = f"{entries[c.track_a]['title']} -> {entries[c.track_b]['title']}"
        print(f"Render diagnostic {number}: {title}", flush=True)
        for variant, match in (("main", True), ("same_cue_no_level_match", False)):
            path = args.output / "audio" / f"audio_{number:03d}_{variant}.wav"
            artifact = render_intent(a, b, c, path, context_seconds=args.context, level_match=match,
                                     incoming_role="arrival", max_boost_db=0.0)
            # 'arrival' here is a renderer clock role; no semantic resolution
            # evidence is inferred from this setting.
            artifact["incoming"]["semantic_arrival_supported"] = False
            quality = evaluate_mix_quality(load_audio(path), artifact["join_sample"])
            event = dict(record["proposal"]["candidate"], sample_rate=48000,
                         output_join_sample=n, intent="handoff")
            validation = validate_transition_event(event, artifact, quality, phrase=record["phrase_a"])
            validation["phrase_evidence_by_side"] = {"A": record["phrase_a"], "B": record["phrase_b"]}
            if validation["hard_defects"]:
                raise ValueError(f"Rendered artifact failed validation: {validation['hard_defects']}")
            artifacts.append({"candidate": record["proposal"]["candidate"], "assessment": record["assessment"],
                              "phrase_a": record["phrase_a"], "phrase_b": record["phrase_b"],
                              "artifact": artifact, "validation": validation, "variant": variant})
            review.append({"id": f"audio_{number:03d}_{variant}", "title": title + (" — controllo volume originale" if not match else ""),
                           "audio_path": path.relative_to(ROOT).as_posix(), "candidate_id": c.candidate_id,
                           "A_out": {"track": entries[c.track_a]["title"], "seconds": c.cue_out_sample / 48000},
                           "B_in": {"track": entries[c.track_b]["title"], "seconds": c.cue_in_sample / 48000},
                           "action": record["assessment"]["action"], "intent": record["assessment"]["intent"],
                           "notes": ["Diagnostica: continuità misurata, qualità percettiva non certificata.",
                                     f"Cambio a {args.context:g}s; contesto totale {args.context * 2:g}s.",
                                     "Chiusura musicale della frase e risoluzione della tensione: ignote."]})
        reference_path = args.output / "audio" / f"audio_{number:03d}_natural_A.wav"
        reference = natural_reference(a, c.cue_out_sample, reference_path, args.context)
        references.append(reference)
        review.append({"id": f"audio_{number:03d}_natural_A", "title": entries[c.track_a]["title"] + " — continuazione originale A",
                       "audio_path": reference_path.relative_to(ROOT).as_posix(), "candidate_id": c.candidate_id,
                       "A_out": {"track": entries[c.track_a]["title"], "seconds": c.cue_out_sample / 48000},
                       "B_in": {"track": entries[c.track_a]["title"], "seconds": c.cue_out_sample / 48000},
                       "action": "stay", "intent": "native_reference", "notes": ["Stesso contesto A; nessuna sostituzione con B. Non è una risposta corretta per definizione."]})
        write(args.output / f"selected_{number:03d}.json", record)
    if implementation_hashes != {str(p.relative_to(ROOT)): sha256_file(p) for p in implementation_paths}:
        raise RuntimeError("Implementation changed during the run; preserve this incomplete run and retry after edits finish")
    manifest = {"schema_version": "musicmix.psychology-pool.1", "status": "diagnostic_rendered" if artifacts else "abstained_no_executable_candidates",
                "tracks": catalog["tracks"], "inputs": {"catalog_sha256": sha256_file(args.catalog),
                "history_sha256": sha256_file(args.history), "history_snapshot": str((args.output / "history_input.json").resolve()),
                "searches": provenance}, "counts": dict(counts), "actions": dict(actions), "intents": dict(intents),
                "pair_actions": pair_counts, "ordered_pairs_searched": len(entries) * (len(entries) - 1),
                "ordered_pairs_assessed": len(pair_counts), "execute_candidates": len(eligible), "selected": len(selected),
                "artifacts": artifacts, "natural_references": references, "assessment_sha256": sha256_file(assessment_path),
                "parameters": {"context_seconds": args.context, "history_exclusion_seconds": 8, "new_region_spacing_seconds": 16,
                "ranking": "execute gate, timing class, independent search-source support, unchanged baseline tie-breaker",
                "max_boost_db": 0, "human_labels_used_for_fitting": False, "new_model_training": False},
                "code_sha256": implementation_hashes,
                "limitations": ["Engineering hypotheses; no calibrated transition reward or demonstrated SOTA.",
                    "B-before is retained as context but the current expectation scorer uses A-before, natural-A-after and B-after.",
                    "No certified cross-song tension resolution provider; arrival and deliberate contrast can remain unavailable.",
                    "Existing CUE-DETR and LinkSeg outputs are opportunities, not perceptual approval.",
                    "Same-cue control isolates level matching; native-A reference is never assigned an automatic positive label.",
                    "V10 human judgments are held out; no claimed listening improvement from synthetic or technical tests."]}
    write(args.output / "manifest.json", manifest)
    write(args.output / "review_manifest.json", {"schema_version": "musicmix.psychology-review.1", "items": review})
    print(json.dumps({k: manifest[k] for k in ("status", "counts", "actions", "execute_candidates", "selected")}), flush=True)


if __name__ == "__main__":
    main()
