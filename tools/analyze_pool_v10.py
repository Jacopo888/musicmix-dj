#!/usr/bin/env python3
"""Run the cached Beat This!/Demucs/faster-whisper providers for pool v10.

This is a deliberately thin, v10-only orchestration layer.  The provider
implementations and preprocessing remain in ``analyze_structure.py`` and
``analyze_vocals.py``; this file only binds their outputs to the immutable
canonical tracks in ``runs/pool_v10/catalog.json``.  It refuses the existing
v8 tracks so an accidental broad rerun cannot change the frozen benchmark.

The default invocation is a dry run.  Provider weights are loaded only when
``--run`` is supplied, which makes catalog/path/hash checks safe while another
CPU job is using the machine.  Each completed track gets an evidence sidecar
under ``runs/pool_v10/evidences`` containing the source/canonical hashes,
provider provenance, output hash, and this runner's hash.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable
import wave


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
STRUCTURE_SITE = ROOT / ".cache" / "structure_runtime" / "Lib" / "site-packages"
STRUCTURE_PACKAGES = ROOT / ".cache" / "structure_runtime" / "packages"
NEW_SLUGS = ("heartless", "mafia")
DEFAULT_CONFIG = ROOT / "configs" / "music_pool_v10.json"
DEFAULT_CATALOG = ROOT / "runs" / "pool_v10" / "catalog.json"
DEFAULT_EVIDENCE_DIR = ROOT / "runs" / "pool_v10" / "evidences"
if str(SRC.resolve()) not in sys.path:
    sys.path.insert(0, str(SRC.resolve()))
from musicmix.paths import localize  # noqa: E402
MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024
BEAT_THIS_CHECKPOINT = (
    ROOT / ".cache" / "structure_runtime" / "models" / "beat_this-small0.ckpt"
)
VOCAL_RUNTIME = ROOT / ".cache" / "vocal_runtime"
WHISPER_SNAPSHOT_ROOT = (
    ROOT / ".cache" / "models" / "models--Systran--faster-whisper-small.en" / "snapshots"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_if_file(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


def _wav_metadata(path: Path) -> tuple[int, int]:
    """Read canonical WAV identity without allocating the full waveform."""

    with wave.open(str(path), "rb") as stream:
        return int(stream.getframerate()), int(stream.getnframes())


def _require_disk_headroom() -> None:
    free = int(shutil.disk_usage(ROOT).free)
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"insufficient disk headroom for provider outputs: {free} bytes free; "
            f"at least {MIN_FREE_BYTES} required"
        )


def _runtime_sys_path() -> None:
    """Prefer project source and the isolated structure runtime.

    Insert in reverse priority because ``sys.path.insert(0, ...)`` is used.
    In particular, this keeps the project's ``src`` ahead of a globally
    installed copy of the MusicMix package while exposing the cached provider
    packages and their dependencies.
    """

    for item in (STRUCTURE_SITE, STRUCTURE_PACKAGES, SRC):
        value = str(item.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)


def _recorded(entry: dict[str, Any], key: str) -> Path:
    return localize(str(entry.get(key, "")))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _find_whisper_model() -> Path:
    candidates = sorted(
        path for path in WHISPER_SNAPSHOT_ROOT.iterdir() if path.is_dir()
    ) if WHISPER_SNAPSHOT_ROOT.is_dir() else []
    for candidate in reversed(candidates):
        if (candidate / "model.bin").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"No cached faster-whisper-small.en snapshot under {WHISPER_SNAPSHOT_ROOT}"
    )


def _select_tracks(
    config: dict[str, Any], catalog: dict[str, Any], slugs: Iterable[str]
) -> list[dict[str, Any]]:
    requested = list(slugs)
    if not requested:
        raise ValueError("at least one --slug is required")
    unknown = sorted(set(requested) - set(NEW_SLUGS))
    if unknown:
        raise ValueError(
            "pool-v10 provider runner is restricted to new tracks heartless/mafia; "
            f"received {', '.join(unknown)}"
        )
    config_tracks = {str(item["slug"]): item for item in config.get("tracks", [])}
    catalog_tracks = {str(item["slug"]): item for item in catalog.get("tracks", [])}
    selected: list[dict[str, Any]] = []
    for slug in requested:
        spec = config_tracks.get(slug)
        entry = catalog_tracks.get(slug)
        if spec is None or entry is None:
            raise ValueError(f"{slug}: missing from config or catalog")
        if slug not in NEW_SLUGS:
            raise ValueError(f"{slug}: not an allowed v10 addition")
        if str(_recorded(entry, "canonical_path").resolve()) != str((ROOT / spec["canonical_path"]).resolve()):
            raise ValueError(f"{slug}: catalog canonical_path does not match config")
        if str(_recorded(entry, "structure_path").resolve()) != str((ROOT / spec["structure_path"]).resolve()):
            raise ValueError(f"{slug}: catalog structure_path does not match config")
        if str(_recorded(entry, "vocals_path").resolve()) != str((ROOT / spec["vocals_path"]).resolve()):
            raise ValueError(f"{slug}: catalog vocals_path does not match config")
        canonical = _recorded(entry, "canonical_path").resolve(strict=True)
        source = _recorded(entry, "source_path").resolve(strict=True)
        actual_source_sha = _sha256(source)
        if actual_source_sha != str(entry.get("source_sha256")):
            raise ValueError(f"{slug}: source SHA does not match catalog")
        actual_rate, actual_frames = _wav_metadata(canonical)
        if actual_rate != 48000 or int(entry.get("sample_rate", 0)) != 48000:
            raise ValueError(f"{slug}: canonical sample rate is not 48 kHz")
        if actual_frames != int(entry.get("duration_samples", -1)):
            raise ValueError(f"{slug}: canonical duration does not match catalog")
        actual_canonical_sha = _sha256(canonical)
        if actual_canonical_sha != str(entry.get("track_id")):
            raise ValueError(f"{slug}: canonical SHA does not match catalog track_id")
        selected.append({
            "slug": slug,
            "config": spec,
            "entry": entry,
            "canonical": canonical,
            "source": source,
            "source_sha256": actual_source_sha,
            "canonical_sha256": actual_canonical_sha,
        })
    return selected


def _common_evidence(
    item: dict[str, Any],
    *,
    config_path: Path,
    catalog_path: Path,
    stage: str,
    runner_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    entry = item["entry"]
    canonical = item["canonical"]
    return {
        "schema_version": "musicmix.pool-v10-provider-evidence.1",
        "status": "pending",
        "slug": item["slug"],
        "stage": stage,
        "track_id": str(entry["track_id"]),
        "canonical_path": str(canonical),
        "canonical_sha256": item["canonical_sha256"],
        "source_path": str(item["source"]),
        "source_sha256": item["source_sha256"],
        "sample_rate": int(entry["sample_rate"]),
        "duration_samples": int(entry["duration_samples"]),
        "duration_seconds": float(entry["duration_seconds"]),
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256_before_stage": _sha256(catalog_path),
        "runner_path": str(runner_path.resolve()),
        "runner_sha256": _sha256(runner_path),
        "threads": int(args.threads),
        "command": [str(value) for value in sys.argv],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _dry_run(
    selected: list[dict[str, Any]],
    *,
    config_path: Path,
    catalog_path: Path,
    stage: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model = None
    model_hash = None
    checkpoint_hash = _sha256_if_file(BEAT_THIS_CHECKPOINT)
    if stage == "vocals":
        model = _find_whisper_model()
        model_hash = _sha256_if_file(model / "model.bin")
    return {
        "status": "dry_run",
        "stage": stage,
        "threads": int(args.threads),
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256": _sha256(catalog_path),
        "beat_this_checkpoint": str(BEAT_THIS_CHECKPOINT.resolve()),
        "beat_this_checkpoint_sha256": checkpoint_hash,
        "faster_whisper_model": None if model is None else str(model),
        "faster_whisper_model_bin_sha256": model_hash,
        "tracks": [
            {
                "slug": item["slug"],
                "canonical_path": str(item["canonical"]),
                "canonical_sha256": item["canonical_sha256"],
                "source_path": str(item["source"]),
                "source_sha256": item["source_sha256"],
                "duration_samples": item["entry"].get("duration_samples"),
                "duration_seconds": item["entry"].get("duration_seconds"),
                "output_path": str(item["entry"][stage + "_path"]),
            }
            for item in selected
        ],
    }


def _make_structure_provider(args: argparse.Namespace) -> Any:
    from analyze_structure import BeatThisConfig, BeatThisProvider

    checkpoint = BEAT_THIS_CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Beat This checkpoint not found: {checkpoint}")
    config = BeatThisConfig(
        checkpoint_path=str(checkpoint),
        model_name="small0",
        device="cpu",
        threads=int(args.threads),
        fps=50,
    )
    return BeatThisProvider(config)


def _run_structure(
    item: dict[str, Any], args: argparse.Namespace, provider: Any
) -> dict[str, Any]:
    from analyze_structure import analyze_one

    output = _recorded(item["entry"], "structure_path")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = analyze_one(
        item["canonical"], output, provider, float(args.timeout), True
    )
    return {
        "stage_status": str(result.get("status")),
        "provider": result.get("provider"),
        "warnings": result.get("warnings", []),
        "result": result,
        "model": provider.provenance,
        "output_path": output,
    }


def _run_vocals(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from analyze_vocals import analyse_one

    model = _find_whisper_model()
    output = _recorded(item["entry"], "vocals_path")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = analyse_one(
        "a",
        item["canonical"],
        output.parent,
        model_path=model,
        runtime_root=VOCAL_RUNTIME,
        threads=int(args.threads),
        skip_separation=False,
        # campo opzionale del catalogo (o della richiesta del ponte); senza: rilevamento sullo stem
        language=item["entry"].get("language") or "auto",
    )
    selection = result.get("language_selection") or {}
    backend = result.get("backend") or {}
    return {
        "stage_status": str(result.get("analysis_status")),
        "backend": result.get("backend"),
        "separator": result.get("separator"),
        "warnings": result.get("warnings", []),
        "result": result,
        "model": {
            "provider": "faster-whisper",
            "model_path": backend.get("model_path") or str(model),
            "model_bin_sha256": backend.get("model_sha256") or _sha256_if_file(model / "model.bin"),
            "compute_type": "int8",
            "language": selection.get("language", "en"),
            "language_source": selection.get("source"),
            "language_probability": backend.get("language_probability"),
            "threads": int(args.threads),
        },
        "output_path": output,
        "secondary_artifacts": [output.parent / "a_vocals.wav"],
    }


def _write_stage_evidence(
    item: dict[str, Any],
    *,
    stage: str,
    run_result: dict[str, Any],
    config_path: Path,
    catalog_path: Path,
    evidence_dir: Path,
    args: argparse.Namespace,
) -> Path:
    output = Path(run_result["output_path"]).resolve(strict=True)
    evidence = _common_evidence(
        item,
        config_path=config_path,
        catalog_path=catalog_path,
        stage=stage,
        runner_path=Path(__file__),
        args=args,
    )
    stage_status = str(run_result.get("stage_status"))
    successful = stage_status in {"ok", "ok_empty", "complete"}
    evidence.update({
        "status": "complete" if successful else "provider_unavailable",
        "stage_status": run_result.get("stage_status"),
        "artifact_path": str(output),
        "artifact_sha256": _sha256(output),
        "model": run_result.get("model"),
        "provider": run_result.get("provider") or run_result.get("backend"),
        "separator": run_result.get("separator"),
        "warnings": run_result.get("warnings", []),
        "secondary_artifacts": [
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path) if path.is_file() else None,
                "exists": path.is_file(),
            }
            for path in run_result.get("secondary_artifacts", [])
        ],
    })
    evidence_path = evidence_dir / f"{item['slug']}_{stage}.json"
    _atomic_json(evidence_path, evidence)
    return evidence_path


def _write_stage_error(
    item: dict[str, Any],
    *,
    stage: str,
    error: BaseException,
    config_path: Path,
    catalog_path: Path,
    evidence_dir: Path,
    args: argparse.Namespace,
) -> Path:
    evidence = _common_evidence(
        item,
        config_path=config_path,
        catalog_path=catalog_path,
        stage=stage,
        runner_path=Path(__file__),
        args=args,
    )
    evidence.update({
        "status": "error",
        "stage_status": "error",
        "error": f"{type(error).__name__}: {error}",
        "artifact_path": None,
        "artifact_sha256": None,
    })
    evidence_path = evidence_dir / f"{item['slug']}_{stage}.json"
    _atomic_json(evidence_path, evidence)
    return evidence_path


def _reusable_evidence(path: Path, item: dict[str, Any], stage: str) -> bool:
    """Reuse only a successful sidecar still bound to the current canonical file."""

    try:
        evidence = _load_json(path)
        if evidence.get("status") != "complete" or evidence.get("stage") != stage:
            return False
        if evidence.get("track_id") != item["canonical_sha256"]:
            return False
        if evidence.get("canonical_sha256") != item["canonical_sha256"]:
            return False
        if evidence.get("source_sha256") != item["source_sha256"]:
            return False
        artifact = _recorded(evidence, "artifact_path")
        if not artifact.is_file():
            return False
        return evidence.get("artifact_sha256") == _sha256(artifact)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--stage", choices=("structure", "vocals"), required=True)
    parser.add_argument("--slug", dest="slugs", action="append", choices=NEW_SLUGS,
                        default=None, help="new v10 track (repeatable)")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--run", action="store_true", help="load cached provider and write outputs")
    parser.add_argument("--force", action="store_true", help="re-run when an evidence sidecar exists")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.slugs is None:
        args.slugs = list(NEW_SLUGS)
    if args.threads < 1 or args.timeout <= 0.0:
        raise SystemExit("--threads must be positive and --timeout must be > 0")
    config_path = args.config.resolve(strict=True)
    catalog_path = args.catalog.resolve(strict=True)
    evidence_dir = args.evidence_dir.resolve()
    config = _load_json(config_path)
    catalog = _load_json(catalog_path)
    expected_config_sha = catalog.get("config_sha256")
    actual_config_sha = _sha256(config_path)
    if expected_config_sha and expected_config_sha != actual_config_sha:
        raise ValueError("catalog config_sha256 does not match the supplied config")
    selected = _select_tracks(config, catalog, args.slugs)
    if not args.run:
        print(json.dumps(_dry_run(selected, config_path=config_path,
                                  catalog_path=catalog_path, stage=args.stage,
                                  args=args), ensure_ascii=False, indent=2))
        return 0
    _runtime_sys_path()
    _require_disk_headroom()
    structure_provider = _make_structure_provider(args) if args.stage == "structure" else None
    reports = []
    for item in selected:
        evidence_path = evidence_dir / f"{item['slug']}_{args.stage}.json"
        if evidence_path.is_file() and not args.force and _reusable_evidence(
            evidence_path, item, args.stage
        ):
            reports.append({"slug": item["slug"], "stage": args.stage,
                            "status": "reused", "evidence_path": str(evidence_path)})
            continue
        print(json.dumps({"event": "started", "slug": item["slug"],
                          "stage": args.stage, "threads": args.threads}), flush=True)
        _require_disk_headroom()
        try:
            run_result = (_run_structure(item, args, structure_provider) if args.stage == "structure"
                          else _run_vocals(item, args))
        except Exception as exc:
            evidence_path = _write_stage_error(
                item, stage=args.stage, error=exc,
                config_path=config_path, catalog_path=catalog_path,
                evidence_dir=evidence_dir, args=args,
            )
            report = {"slug": item["slug"], "stage": args.stage,
                      "status": "error", "error": f"{type(exc).__name__}: {exc}",
                      "evidence_path": str(evidence_path)}
            reports.append(report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
            continue
        evidence_path = _write_stage_evidence(
            item, stage=args.stage, run_result=run_result,
            config_path=config_path, catalog_path=catalog_path,
            evidence_dir=evidence_dir, args=args,
        )
        stage_status = str(run_result.get("stage_status"))
        reports.append({
            "slug": item["slug"],
            "stage": args.stage,
            "status": "complete" if stage_status in {"ok", "ok_empty", "complete"} else "provider_unavailable",
            "stage_status": stage_status,
            "artifact_path": str(Path(run_result["output_path"]).resolve()),
            "artifact_sha256": _sha256(Path(run_result["output_path"])),
            "evidence_path": str(evidence_path),
        })
        print(json.dumps(reports[-1], ensure_ascii=False), flush=True)
    if any(item["status"] == "error" for item in reports):
        overall = "error"
    elif any(item["status"] == "provider_unavailable" for item in reports):
        overall = "complete_with_unavailable"
    else:
        overall = "complete"
    print(json.dumps({"status": overall, "reports": reports}, ensure_ascii=False, indent=2))
    return 0 if overall == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
