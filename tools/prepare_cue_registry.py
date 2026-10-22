"""Bind official CUE-DETR inference artifacts to the pilot input contract.

No model predictions are recomputed. Out-of-recording padded-window detections
remain in the original audit and are counted explicitly when excluded here.
"""
from pathlib import Path
import argparse
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from musicmix.ingest import sha256_file
from musicmix.paths import localize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new registry path")
    original = json.loads(args.input.read_text(encoding="utf-8"))
    if original.get("schema_version") != "musicmix.pretrained-cue-detr.1" or original.get("status") != "complete":
        raise ValueError("Require completed official CUE-DETR adapter output")
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if original["checkpoint"]["loading_info"].get(key):
            raise ValueError("Checkpoint not fully compatible")
    original_sha = sha256_file(args.input)
    extracted_dir = args.output.parent / f"{args.output.stem}_tracks"
    extracted_dir.mkdir(parents=True, exist_ok=False)
    tracks = []
    for item in original["tracks"]:
        if item["status"] != "complete" or item["canonical_sha256"] != item["track_id"]:
            raise ValueError("Incomplete or unbound track inference")
        if sha256_file(localize(item["canonical_path"])) != item["track_id"]:
            raise ValueError("Canonical recording changed since inference")
        slug = str(item["slug"])
        if not slug or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in slug):
            raise ValueError("Unsafe or ambiguous track slug")
        raw_path = extracted_dir / f"{slug}.json"
        with raw_path.open("x", encoding="utf-8") as handle:
            json.dump({"source_artifact": str(args.input.resolve()), "source_sha256": original_sha,
                       "upstream": original["upstream"], "checkpoint": original["checkpoint"],
                       "runtime": original["runtime"], "track": item}, handle, ensure_ascii=False, indent=2)
        accepted, outside = [], []
        for cue in item["cues"]:
            seconds = float(cue["time_seconds"])
            if not math.isfinite(seconds):
                raise ValueError("Nonfinite model timestamp")
            if 0 <= round(seconds * item["sample_rate"]) < item["duration_samples"]:
                accepted.append(cue)
            else:
                outside.append(cue)
        tracks.append({**{k: item[k] for k in ("track_id", "sample_rate", "duration_samples", "slug", "status")},
                       "cues": accepted, "excluded_outside_recording": outside,
                       "inference_evidence": {"canonical_sha256": item["canonical_sha256"],
                                              "artifact_path": str(raw_path.resolve()),
                                              "artifact_sha256": sha256_file(raw_path),
                                              "original_artifact_sha256": original_sha}})
    payload = {"schema_version": "musicmix.learned-opportunity-registry.1",
               "status": "complete", "provider": "CUE-DETR official checkpoint",
               "source": str(args.input.resolve()), "source_sha256": original_sha,
               "tracks": tracks, "selection_effect": "coordinates only; model scores not calibrated or retuned"}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"tracks": len(tracks), "retained_cues": sum(len(t["cues"]) for t in tracks),
                      "excluded_outside_recording": sum(len(t["excluded_outside_recording"]) for t in tracks)}))


if __name__ == "__main__":
    main()
