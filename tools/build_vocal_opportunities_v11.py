"""Build a canonical opportunity registry from observed vocal boundaries.

This is coordinate and provenance glue.  ASR segment starts and ends remain
observations with explicit roles: an end is an outgoing timing opportunity and
a start is an incoming timing opportunity.  Neither is labelled phrase
closure or acoustic silence.  Existing CUE-DETR and LinkSeg registries are
copied as provider opportunities only when their canonical identity and
artifact hashes still match.

The output follows ``musicmix.learned-opportunity-registry.1`` so it can be
passed to ``freeze_pool_search.py --opportunities``.  Additional role and
source fields are retained for a downstream driver that understands them;
the older search contract safely ignores those fields.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from musicmix.paths import localize  # noqa: E402
RATE = 48_000
SCHEMA = "musicmix.learned-opportunity-registry.1"
BUILDER_VERSION = "psychology-v11-vocal-opportunities-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric")
    return result


def _sample(value: Any, *, duration: int, name: str) -> int:
    number = _number(value, name)
    result = int(round(number))
    if not 0 <= result < duration:
        raise ValueError(f"{name} outside canonical duration: {result}")
    return result


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve(path_value: Any, base: Path) -> Path:
    path = localize(str(path_value)).expanduser()
    return path if path.is_absolute() else (base / path)


def _require_track_identity(document: Mapping[str, Any], entry: Mapping[str, Any], kind: str) -> None:
    for field in ("track_id", "sample_rate", "duration_samples"):
        if document.get(field) != entry.get(field):
            raise ValueError(f"{kind} identity mismatch for {entry.get('slug')}: {field}")
    if int(document.get("sample_rate", 0)) != RATE:
        raise ValueError(f"{kind} is not on the 48 kHz canonical timeline")


def _provider_records(path: Path, catalog_by_id: Mapping[str, Mapping[str, Any]],
                      provider: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Load and validate one existing CUE/LinkSeg registry."""

    if not path.is_file():
        raise FileNotFoundError(f"missing {provider} registry: {path}")
    registry = _read(path)
    if registry.get("schema_version") != SCHEMA or registry.get("status") != "complete":
        raise ValueError(f"incomplete {provider} registry: {path}")
    records: dict[str, list[dict[str, Any]]] = {}
    for track in registry.get("tracks", []):
        if not isinstance(track, Mapping):
            raise ValueError(f"invalid {provider} track record")
        identity = str(track.get("track_id"))
        if identity not in catalog_by_id:
            raise ValueError(f"{provider} registry contains unknown track {identity}")
        catalog_entry = catalog_by_id[identity]
        _require_track_identity(track, catalog_entry, provider)
        evidence = track.get("inference_evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError(f"{provider} record has no inference evidence: {identity}")
        if evidence.get("canonical_sha256") != identity:
            raise ValueError(f"{provider} evidence is not bound to canonical track: {identity}")
        artifact_path = _resolve(evidence.get("artifact_path", ""), path.parent)
        artifact_sha = evidence.get("artifact_sha256")
        if not artifact_path.is_file() or not isinstance(artifact_sha, str):
            raise ValueError(f"missing {provider} evidence artifact: {identity}")
        if sha256_file(artifact_path) != artifact_sha:
            raise ValueError(f"changed {provider} evidence artifact: {identity}")
        cues = track.get("cues")
        if not isinstance(cues, list):
            raise ValueError(f"{provider} cues must be a list: {identity}")
        bound_cues = []
        for cue in cues:
            item = dict(cue)
            item["inference_evidence"] = {
                "artifact_path": str(artifact_path.resolve()),
                "artifact_sha256": artifact_sha,
                "canonical_sha256": identity,
                "registry_path": str(path.resolve()),
                "registry_sha256": sha256_file(path),
            }
            bound_cues.append(item)
        records[identity] = bound_cues
    return records, {"provider": provider, "path": str(path.resolve()), "sha256": sha256_file(path)}


def _native_cues(vocals: Mapping[str, Any], entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    duration = int(entry["duration_samples"])
    result: list[dict[str, Any]] = []
    segments = vocals.get("asr_segments", [])
    if not isinstance(segments, list):
        raise ValueError(f"asr_segments must be a list: {entry['slug']}")
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            continue
        start = _sample(segment.get("start_sample", segment.get("start")), duration=duration,
                        name=f"{entry['slug']} ASR segment start")
        end_number = _number(segment.get("end_sample", segment.get("end")),
                             f"{entry['slug']} ASR segment end")
        end = int(round(end_number))
        if not start < end <= duration:
            raise ValueError(f"invalid ASR segment interval: {entry['slug']}:{index}")
        confidence = segment.get("confidence")
        cue_common = {
            "provider": "native_observed_asr",
            "observation": "ASR segment timing; no phrase closure or acoustic silence claim",
            "segment_index": index,
            "confidence": confidence if confidence is None else max(0.0, min(1.0, _number(confidence, "ASR confidence"))),
        }
        result.extend((
            {"sample": end, "time_seconds": end / RATE, "role": "outgoing",
             "role_semantics": "observed ASR segment end; timing only", **cue_common},
            {"sample": start, "time_seconds": start / RATE, "role": "incoming",
             "role_semantics": "observed ASR segment start; timing only", **cue_common},
        ))
    # Adjacent ASR word endpoints are additional observed timing candidates.
    # They make the internal gap available to the unchanged vocal guard, which
    # can then require the existing confidence, margin, and phase checks.  The
    # endpoints remain timing evidence only: a word gap is not certified phrase
    # closure or acoustic silence.
    words = vocals.get("words", [])
    if not isinstance(words, list):
        raise ValueError(f"words must be a list: {entry['slug']}")
    normalized_words: list[tuple[int, int, float | None, int]] = []
    for index, word in enumerate(words):
        if not isinstance(word, Mapping):
            continue
        start = _sample(word.get("start_sample", word.get("start")), duration=duration,
                        name=f"{entry['slug']} ASR word start")
        end = _sample(word.get("end_sample", word.get("end")), duration=duration,
                      name=f"{entry['slug']} ASR word end")
        if not start < end:
            raise ValueError(f"invalid ASR word interval: {entry['slug']}:{index}")
        confidence_value = word.get("confidence")
        confidence = None if confidence_value is None else max(
            0.0, min(1.0, _number(confidence_value, "ASR word confidence")))
        normalized_words.append((start, end, confidence, index))
    normalized_words.sort(key=lambda item: (item[0], item[1], item[3]))
    for gap_index, (previous, following) in enumerate(zip(normalized_words, normalized_words[1:])):
        _previous_start, previous_end, previous_confidence, previous_index = previous
        following_start, _following_end, following_confidence, following_index = following
        if following_start <= previous_end:
            continue
        common = {
            "provider": "native_observed_asr_word_gap",
            "observation": "ASR word timing gap; no phrase closure or acoustic silence claim",
            "evidence_type": "asr_word_gap",
            "word_index_before": previous_index,
            "word_index_after": following_index,
            "gap_confidence": None if previous_confidence is None or following_confidence is None
            else min(previous_confidence, following_confidence),
        }
        result.extend((
            {"sample": previous_end, "time_seconds": previous_end / RATE, "role": "outgoing",
             "role_semantics": "observed ASR word end; gap timing only", **common},
            {"sample": following_start, "time_seconds": following_start / RATE, "role": "incoming",
             "role_semantics": "observed ASR word start; gap timing only", **common},
        ))
    return result


def _merge_cues(cues: list[dict[str, Any]], duration: int, slug: str) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for raw in cues:
        sample_value = raw.get("sample")
        if sample_value is None:
            seconds = _number(raw.get("time_seconds"), f"{slug} provider cue time")
            sample = int(round(seconds * RATE))
        else:
            sample = _sample(sample_value, duration=duration, name=f"{slug} provider cue")
        if not 0 <= sample < duration:
            raise ValueError(f"provider cue outside duration: {slug}:{sample}")
        item = dict(raw)
        item["sample"] = sample
        item["time_seconds"] = sample / RATE
        item.setdefault("role", "provider_opportunity")
        item.setdefault("role_semantics", "provider opportunity; role not supplied by provider")
        prior = merged.get(sample)
        if prior is None:
            item["sources"] = [dict(item)]
            merged[sample] = item
            continue
        sources = list(prior.get("sources", []))
        sources.extend(item.get("sources", [dict(item)]))
        roles = {str(prior.get("role"))}
        roles.update(str(source.get("role")) for source in sources)
        prior["role"] = next(iter(roles)) if len(roles) == 1 else "multi_role"
        prior["role_semantics"] = "multiple independent observations at one canonical sample"
        prior["sources"] = sources
    result = []
    for sample in sorted(merged):
        item = merged[sample]
        # Do not leak implementation-only nested duplication into the primary
        # cue while retaining every exact source observation under `sources`.
        item["sources"] = [
            {key: value for key, value in source.items() if key != "sources"}
            for source in item.get("sources", [])
        ]
        result.append(item)
    return result


def _history_hit(track_id: str, sample: int, role: str, regions: list[Mapping[str, Any]],
                 tolerance_seconds: float) -> bool:
    """Return whether a cue is already exposed on the corresponding track side.

    The current frozen-search adapter carries exact samples but does not carry
    directional role metadata into its candidate geometry.  Every cue is
    therefore checked against both source sides, even when its registry role is
    directional.  This conservative exclusion is for candidate generation
    only; it does not label the remaining cue as safe or as a phrase closure.
    """

    def close(value: Any) -> bool:
        try:
            return abs(sample - int(value)) / RATE <= tolerance_seconds
        except (TypeError, ValueError):
            return False

    # Keep the role argument in the contract for downstream role-aware drivers,
    # but use both checks for compatibility with the role-blind frozen search.
    _ = str(role)
    check_outgoing = check_incoming = True
    for region in regions:
        if check_outgoing and region.get("track_a") == track_id and close(region.get("cue_out_sample")):
            return True
        if check_incoming and region.get("track_b") == track_id and close(region.get("cue_in_sample")):
            return True
    return False


def _exclude_history(cues: list[dict[str, Any]], track_id: str, history: Mapping[str, Any] | None,
                     tolerance_seconds: float) -> tuple[list[dict[str, Any]], int]:
    if history is None:
        return cues, 0
    regions = history.get("regions")
    if not isinstance(regions, list):
        raise ValueError("history document must contain a regions list")
    kept: list[dict[str, Any]] = []
    rejected = 0
    for cue in cues:
        if _history_hit(track_id, int(cue["sample"]), str(cue.get("role", "provider_opportunity")),
                        regions, tolerance_seconds):
            rejected += 1
        else:
            kept.append(cue)
    return kept, rejected


def build_registry(catalog_path: str | Path, output_path: str | Path,
                   *, cue_registry: str | Path | None = None,
                   linkseg_registry: str | Path | None = None,
                   history_path: str | Path | None = None,
                   history_tolerance_seconds: float = 8.0) -> dict[str, Any]:
    catalog_path = Path(catalog_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing registry: {output_path}")
    catalog = _read(catalog_path)
    if catalog.get("status") != "ready":
        raise ValueError("catalog must be ready")
    tracks = catalog.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        raise ValueError("catalog must contain tracks")
    if not math.isfinite(float(history_tolerance_seconds)) or history_tolerance_seconds < 0:
        raise ValueError("history tolerance must be finite and nonnegative")
    history: dict[str, Any] | None = None
    history_path_resolved: Path | None = None
    if history_path is not None:
        history_path_resolved = Path(history_path).expanduser().resolve()
        history = _read(history_path_resolved)
        if history.get("schema_version") != "musicmix.audition-history.1":
            raise ValueError("history must use musicmix.audition-history.1")
    by_id: dict[str, Mapping[str, Any]] = {}
    for entry in tracks:
        if not isinstance(entry, Mapping):
            raise ValueError("invalid catalog track")
        identity = str(entry.get("track_id"))
        if identity in by_id:
            raise ValueError(f"duplicate catalog track: {identity}")
        by_id[identity] = entry

    providers: dict[str, dict[str, list[dict[str, Any]]]] = {}
    provider_sources: list[dict[str, Any]] = []
    for provider, registry_value in (("cue_detr", cue_registry), ("linkseg", linkseg_registry)):
        if registry_value is None:
            continue
        path = Path(registry_value).expanduser().resolve()
        records, source = _provider_records(path, by_id, provider)
        providers[provider] = records
        provider_sources.append(source)

    sidecar_dir = output_path.parent / f"{output_path.stem}_evidence"
    sidecar_dir.mkdir(parents=True, exist_ok=False)
    output_tracks: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    for entry in tracks:
        identity = str(entry["track_id"])
        slug = str(entry.get("slug", identity[:12]))
        vocal_path = _resolve(entry.get("vocals_path", ""), catalog_path.parent)
        vocal_sha = entry.get("vocals_sha256")
        if not vocal_path.is_file() or not isinstance(vocal_sha, str) or sha256_file(vocal_path) != vocal_sha:
            raise ValueError(f"changed or missing vocals artifact: {slug}")
        vocals = _read(vocal_path)
        _require_track_identity(vocals, entry, "vocals")
        cues = _native_cues(vocals, entry)
        native_count = len(cues)
        totals["native_asr_observations"] += sum(1 for cue in cues
                                                  if cue.get("provider") == "native_observed_asr")
        totals["native_asr_word_gap_observations"] += sum(
            1 for cue in cues if cue.get("provider") == "native_observed_asr_word_gap")
        source_artifacts = [{"provider": "native_observed_asr", "artifact_path": str(vocal_path.resolve()),
                             "artifact_sha256": vocal_sha, "canonical_sha256": identity}]
        for provider, records in providers.items():
            provider_cues = records.get(identity, [])
            for cue in provider_cues:
                item = dict(cue)
                item["provider"] = provider
                item.setdefault("role", "provider_opportunity")
                item.setdefault("role_semantics", "provider opportunity; role not supplied by provider")
                cues.append(item)
            totals[f"{provider}_observations"] += len(provider_cues)
            if provider_cues:
                evidence_values = [cue.get("inference_evidence") for cue in provider_cues
                                   if isinstance(cue.get("inference_evidence"), Mapping)]
                source_artifacts.append({"provider": provider,
                                         "registry_provider_record_count": len(provider_cues),
                                         "inference_evidence": evidence_values})
        merged = _merge_cues(cues, int(entry["duration_samples"]), slug)
        totals["unique_canonical_samples_before_history"] += len(merged)
        merged, history_rejected = _exclude_history(merged, identity, history, history_tolerance_seconds)
        totals["history_rejected_cues"] += history_rejected
        totals["unique_canonical_samples"] += len(merged)
        evidence = {
            "canonical_sha256": identity,
            "canonical_sample_rate": int(entry["sample_rate"]),
            "canonical_duration_samples": int(entry["duration_samples"]),
            "native_vocal_artifact": {"path": str(vocal_path.resolve()), "sha256": vocal_sha},
            "provider_sources": provider_sources,
            "history_filter": None if history_path_resolved is None else {
                "path": str(history_path_resolved),
                "sha256": sha256_file(history_path_resolved),
                "tolerance_seconds": float(history_tolerance_seconds),
                "rejected_cues_for_track": history_rejected,
            },
            "source_artifacts": source_artifacts,
            "semantics": {
                "native_asr_segment_end": "outgoing timing observation; not phrase closure",
                "native_asr_segment_start": "incoming timing observation; not phrase closure",
                "native_asr_word_gap": "word endpoint timing observation; not phrase closure or acoustic silence",
                "provider_opportunity": "coordinate supplied by provider; no safety or reward claim",
                "beat_alignment": "no fake beat snapping; samples remain native provider coordinates",
            },
        }
        sidecar = sidecar_dir / f"{slug}.json"
        sidecar.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        sidecar_sha = sha256_file(sidecar)
        output_tracks.append({
            "track_id": identity,
            "sample_rate": int(entry["sample_rate"]),
            "duration_samples": int(entry["duration_samples"]),
            "slug": slug,
            "status": "complete",
            "provider_id": "native_asr_and_verified_opportunities",
            "cues": merged,
            "inference_evidence": {
                "canonical_sha256": identity,
                "artifact_path": str(sidecar.resolve()),
                "artifact_sha256": sidecar_sha,
                "native_vocal_artifact_path": str(vocal_path.resolve()),
                "native_vocal_artifact_sha256": vocal_sha,
                "provider_sources": provider_sources,
            },
            "role_contract": {
                "outgoing": "native observed ASR segment ends only",
                "incoming": "native observed ASR segment starts only",
            "word_gap": "native observed ASR word endpoints; timing only",
                "provider_opportunity": "role remains provider-supplied or multi-role",
                "closure": "never certified by this registry",
            },
        })

    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "complete",
        "provider": "native observed ASR boundaries + verified provider opportunities",
        "provider_id": "native_asr_v11",
        "builder_version": BUILDER_VERSION,
        "source_catalog": str(catalog_path),
        "source_catalog_sha256": sha256_file(catalog_path),
        "source_registries": provider_sources,
        "history_filter": None if history_path_resolved is None else {
            "path": str(history_path_resolved),
            "sha256": sha256_file(history_path_resolved),
            "tolerance_seconds": float(history_tolerance_seconds),
            "semantics": "history exclusion only; remaining opportunities are not certified safe or unheard beyond this exact registry",
        },
        "tracks": output_tracks,
        "counts": dict(totals),
        "selection_effect": "coordinates only; no threshold fitting, beat snapping, phrase-closure, safety, or reward inference",
        "semantics": {
            "native_asr_segment_end": "outgoing timing evidence only",
            "native_asr_segment_start": "incoming timing evidence only",
            "native_asr_word_gap": "ASR word endpoint timing evidence only; no closure or silence claim",
            "asr_missing": "unknown, never silence",
            "provider_cues": "retained with provider provenance; no automatic phrase or arrival claim",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cue-registry", type=Path, default=None)
    parser.add_argument("--linkseg-registry", type=Path, default=None)
    parser.add_argument("--history", dest="history_path", type=Path, default=None,
                        help="optional audition history used only to exclude already exposed cue samples")
    parser.add_argument("--history-tolerance-seconds", type=float, default=8.0)
    args = parser.parse_args(argv)
    result = build_registry(args.catalog, args.output, cue_registry=args.cue_registry,
                            linkseg_registry=args.linkseg_registry,
                            history_path=args.history_path,
                            history_tolerance_seconds=args.history_tolerance_seconds)
    print(json.dumps({"output": str(Path(args.output).resolve()), "tracks": len(result["tracks"]),
                      "counts": result["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

