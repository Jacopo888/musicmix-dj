"""Merge separately inferred, canonical-bound cue records without rerunning models."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from musicmix.ingest import sha256_file
from musicmix.paths import localize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new merged registry path")
    tracks, sources, seen, model_hashes, policies = [], [], set(), set(), set()
    for path in args.registry:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "complete":
            raise ValueError("Cannot merge unavailable or partial inference")
        source = localize(record["source"])
        if sha256_file(source) != record["source_sha256"]:
            raise ValueError("Source inference changed since registry creation")
        native = json.loads(source.read_text(encoding="utf-8"))
        model_hashes.add(native["checkpoint"]["model_sha256"])
        policies.add(json.dumps(native["parameters"], sort_keys=True))
        for track in record["tracks"]:
            if track["track_id"] in seen:
                raise ValueError("Duplicate canonical track in registries")
            evidence = track["inference_evidence"]
            if sha256_file(localize(evidence["artifact_path"])) != evidence["artifact_sha256"]:
                raise ValueError("Per-track evidence changed")
            seen.add(track["track_id"]); tracks.append(track)
        sources.append({"registry": str(path.resolve()), "sha256": sha256_file(path),
                        "source": str(source.resolve()), "source_sha256": record["source_sha256"]})
    if len(model_hashes) != 1 or len(policies) != 1:
        raise ValueError("Models or postprocessing policies differ; use a separately declared experiment")
    result = {"schema_version": "musicmix.learned-opportunity-registry.1", "status": "complete",
              "provider": "CUE-DETR official checkpoint", "source_registries": sources,
              "checkpoint_sha256": next(iter(model_hashes)), "tracks": tracks,
              "note": "Per-track inference provenance preserved across separate runs; no prediction recomputed."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"tracks": len(tracks), "cues": sum(len(t["cues"]) for t in tracks)}))


if __name__ == "__main__":
    main()
