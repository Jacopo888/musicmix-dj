"""Freeze an existing MusicMix all-beat search for a pretrained-model pilot.

This runner calls the established pool search unchanged. History is applied
later by run_pretrained_pool_pilot, so coverage before/after exposure can be
distinguished without recomputing audio features.
"""
from pathlib import Path
import argparse
import gzip
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from musicmix.audio import load_audio
from musicmix.ingest import sha256_file
from musicmix.paths import localize
from musicmix.pool_search import search_pool
from musicmix.learned_opportunities import validate_opportunities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-candidates-per-pair", type=int, default=1000000)
    parser.add_argument("--opportunities", type=Path, help="Use exact model timestamps instead of all-beat cues")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new frozen run directory")
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    if catalog.get("status") != "ready":
        raise ValueError("All canonical/provider evidence must be ready")
    tracks, structures, vocals = {}, {}, {}
    for item in catalog["tracks"]:
        track = load_audio(item["canonical_path"])
        if track.track_id != item["track_id"]:
            raise ValueError("Canonical input changed")
        tracks[track.track_id] = track
        for key, target in (("structure", structures), ("vocals", vocals)):
            path = localize(item[key + "_path"])
            if sha256_file(path) != item[key + "_sha256"]:
                raise ValueError("Provider artifact changed")
            target[track.track_id] = json.loads(path.read_text(encoding="utf-8"))
    policy = "beats"
    if args.opportunities:
        registry = json.loads(args.opportunities.read_text(encoding="utf-8"))
        by_id = {t["track_id"]: t for t in catalog["tracks"]}
        registry_ids = [t["track_id"] for t in registry["tracks"]]
        if len(set(registry_ids)) != len(registry_ids):
            raise ValueError("Exact model-cue registry must not contain duplicate tracks")
        if set(registry_ids) != set(by_id):
            raise ValueError("Exact model-cue registry must cover all canonical tracks")
        for record in registry["tracks"]:
            identity = record["track_id"]
            cues = validate_opportunities(record, by_id[identity])
            evidence = record["inference_evidence"]
            if evidence.get("canonical_sha256") != identity:
                raise ValueError("Model evidence must name the canonical track hash")
            if sha256_file(localize(evidence["artifact_path"])) != evidence["artifact_sha256"]:
                raise ValueError("Model artifact changed")
            structures[identity] = dict(structures[identity], provided_cues=[
                dict(cue, sample=cue["sample"], confidence=None) for cue in cues])
        policy = "provided"
    print(json.dumps({"event": "search", "tracks": len(tracks), "pairs": len(tracks) * (len(tracks) - 1)}), flush=True)
    pool = search_pool(tracks, structures, vocals, cue_policy=policy,
                       max_candidates_per_pair=args.max_candidates_per_pair,
                       top_k=8, max_per_pair=4, heard_regions=None)
    args.output.mkdir(parents=True)
    with gzip.open(args.output / "search.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(pool, stream, ensure_ascii=False)
    manifest = {"schema_version": "musicmix.frozen-pool-search.1", "status": "complete",
                "inputs": {"catalog": str(args.catalog.resolve()), "catalog_sha256": sha256_file(args.catalog)},
                "parameters": {"cue_policy": policy, "max_candidates_per_pair": args.max_candidates_per_pair,
                               "history": "not applied; required downstream before new presentation"},
                "search_sha256": sha256_file(args.output / "search.json.gz"),
                "diagnostics": pool["diagnostics"], "pair_count": pool["pair_count"],
                "all_pair_counts": pool["all_pair_counts"],
                "code_sha256": {name: sha256_file(ROOT / "src/musicmix" / name)
                                for name in ("directional.py", "pool_search.py")}}
    if args.opportunities:
        manifest["inputs"]["opportunities_sha256"] = sha256_file(args.opportunities)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "proposals": len(pool["proposals"]), "pairs": pool["pair_count"]}), flush=True)


if __name__ == "__main__":
    main()
