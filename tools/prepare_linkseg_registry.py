"""Normalize verified official LinkSeg pool6 sections into opportunities.

This is coordinate glue only. It exposes internal LinkSeg section boundaries as
learned opportunities and never infers arrival, safety, or reward semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from musicmix.paths import localize  # noqa: E402

EXPECTED_SOURCE_REVISION = "38dc0dabed14daa5fbef7a61ea3cb046517e74a0"
EXPECTED_CHECKPOINT_SHA256 = "5f7b6d749719086a133ebe215c8e37a55d1089dd2ab7c334613004cdd7c2d9a9"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_identity(observed: Mapping[str, Any], expected: Mapping[str, Any], context: str) -> None:
    for field in ("track_id", "sample_rate", "duration_samples"):
        if observed.get(field) != expected.get(field):
            raise ValueError(f"{context}: {field} does not match canonical catalog")


def require_sha256(observed: str, expected: str, context: str) -> None:
    if observed != expected:
        raise ValueError(f"{context}: SHA256 mismatch")


def validate_manifest(manifest: Mapping[str, Any], manifest_path: Path, catalog: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if manifest.get("schema_version") != "musicmix.linkseg_pool6_manifest.v1" or manifest.get("status") != "complete":
        raise ValueError("require complete LinkSeg pool6 manifest")
    if manifest.get("source_revision") != EXPECTED_SOURCE_REVISION:
        raise ValueError("unexpected LinkSeg source revision")
    if manifest.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("unexpected LinkSeg checkpoint hash")
    catalog_tracks = {item["slug"]: item for item in catalog.get("tracks", [])}
    entries = manifest.get("tracks")
    if not isinstance(entries, list) or set(item.get("slug") for item in entries) != set(catalog_tracks):
        raise ValueError("manifest must cover exactly the canonical catalog tracks")
    result: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        catalog_entry = catalog_tracks[entry["slug"]]
        require_identity(entry, catalog_entry, f"manifest[{entry['slug']}]")
        output_path = localize(entry.get("output_path", ""))
        if not output_path.is_file():
            raise ValueError(f"missing LinkSeg artifact: {entry['slug']}")
        require_sha256(sha256_file(output_path), entry.get("output_sha256", ""),
                       f"LinkSeg artifact {entry['slug']}")
        if entry.get("status") != "complete" or entry.get("load_missing_keys") or entry.get("load_unexpected_keys"):
            raise ValueError(f"incomplete LinkSeg artifact: {entry['slug']}")
        artifact = json.loads(output_path.read_text(encoding="utf-8"))
        artifact_identity = {
            "track_id": artifact.get("canonical_track_id"),
            "sample_rate": artifact.get("canonical_sample_rate"),
            "duration_samples": artifact.get("canonical_duration_samples"),
        }
        require_identity(artifact_identity, catalog_entry, f"artifact[{entry['slug']}]")
        if artifact.get("status") != "complete" or artifact.get("model", {}).get("source_revision") != EXPECTED_SOURCE_REVISION:
            raise ValueError(f"artifact provenance mismatch: {entry['slug']}")
        if artifact.get("model", {}).get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError(f"artifact checkpoint mismatch: {entry['slug']}")
        if artifact.get("inputs", {}).get("audio", {}).get("sha256") != catalog_entry.get("track_id"):
            raise ValueError(f"artifact audio identity mismatch: {entry['slug']}")
        result[entry["slug"]] = {"manifest": entry, "catalog": catalog_entry, "artifact": artifact, "path": output_path}
    return result


def normalize_track(item: Mapping[str, Any], manifest_path: Path, manifest_sha256: str) -> dict[str, Any]:
    artifact = item["artifact"]
    catalog_entry = item["catalog"]
    rate = int(catalog_entry["sample_rate"])
    duration = int(catalog_entry["duration_samples"])
    sections = artifact.get("sections")
    if not isinstance(sections, list):
        raise ValueError(f"sections must be a list: {catalog_entry['slug']}")
    cues: list[dict[str, Any]] = []
    for index, section in enumerate(sections):
        start_sample = int(section["start_sample"])
        if not 0 < start_sample < duration:
            # Exclude the recording's left edge and final edge. The final edge
            # is represented only as an end_sample by the last section.
            continue
        if index == 0 or sections[index - 1].get("end_sample") != start_sample:
            raise ValueError(f"non-contiguous LinkSeg boundary: {catalog_entry['slug']}:{index}")
        before = str(sections[index - 1]["label"])
        after = str(section["label"])
        seconds = start_sample / rate
        if not math.isfinite(seconds):
            raise ValueError("nonfinite LinkSeg boundary")
        cues.append({
            "time_seconds": seconds,
            "sample": start_sample,
            "confidence": None,
            "label_before": before,
            "label_after": after,
        })
    evidence = {
        "canonical_sha256": catalog_entry["track_id"],
        "artifact_path": str(item["path"].resolve()),
        "artifact_sha256": sha256_file(item["path"]),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": manifest_sha256,
        "source_revision": EXPECTED_SOURCE_REVISION,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
    }
    return {
        "track_id": catalog_entry["track_id"],
        "sample_rate": rate,
        "duration_samples": duration,
        "slug": catalog_entry["slug"],
        "status": "complete",
        "provider_id": "linkseg",
        "cues": cues,
        "inference_evidence": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing registry")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    manifest_sha256 = sha256_file(args.manifest)
    verified = validate_manifest(manifest, args.manifest, catalog)
    tracks = [normalize_track(verified[slug], args.manifest, manifest_sha256)
              for slug in sorted(verified)]
    payload = {
        "schema_version": "musicmix.learned-opportunity-registry.1",
        "status": "complete",
        "provider": "LinkSeg official checkpoint",
        "provider_id": "linkseg",
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": manifest_sha256,
        "source_revision": EXPECTED_SOURCE_REVISION,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "selection_effect": "boundary coordinates only; confidence unavailable; no arrival or reward inference",
        "tracks": tracks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"tracks": len(tracks), "retained_boundaries": sum(len(t["cues"]) for t in tracks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
