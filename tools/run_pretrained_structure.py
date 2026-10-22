#!/usr/bin/env python3
"""Run the official All-In-One pretrained structure analyser.

The command writes MusicMix ``structure.v2`` JSON while preserving the
provider's functional section labels (intro, verse, chorus, ...).  It does not
fall back to the project's heuristic feature segments: when the optional
All-In-One runtime is unavailable, output is explicitly ``status=unavailable``
with empty events and sections.

Use the isolated structure runtime, for example on Windows::

    .cache\\structure_runtime\\Scripts\\python.exe tools\\run_pretrained_structure.py \\
      --pool-config configs\\music_pool_v8.json \\
      --output-dir runs\\pretrained_v10

All-In-One requires the official NATTEN and madmom dependencies.  The script
does not build NATTEN or download Demucs implicitly outside the supplied
isolated runtime; existing Demucs stems can be reused through ``--demix-dir``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# The official madmom 0.16.1 release uses a ragged two-column Viterbi result
# that NumPy >=1.24 now rejects without ``dtype=object``.  A copy containing
# that one upstream compatibility edit is kept inside this project runtime;
# prefer it when present, without modifying a global Python installation.
MADMOM_COMPAT = ROOT / ".cache" / "madmom_compat"
if MADMOM_COMPAT.is_dir() and str(MADMOM_COMPAT) not in sys.path:
    sys.path.insert(0, str(MADMOM_COMPAT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from musicmix.audio import load_audio  # noqa: E402
from musicmix.pretrained_structure import (  # noqa: E402
    ALL_IN_ONE_REPOSITORY,
    AllInOneConfig,
    AllInOneProvider,
)
from musicmix.structure import analyze_structure, write_structure  # noqa: E402


def _path_from_root(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def load_pool(config_path: Path) -> list[tuple[str, Path]]:
    """Resolve canonical pool paths and stable slugs from a MusicMix pool JSON."""

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("tracks"), list):
        raise ValueError(f"invalid pool config: {config_path}")
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else ROOT
    items: list[tuple[str, Path]] = []
    for item in payload["tracks"]:
        if not isinstance(item, dict):
            raise ValueError("pool tracks must be objects")
        slug = str(item.get("slug", "")).strip()
        raw_path = item.get("canonical_path")
        if not slug or not isinstance(raw_path, str) or not raw_path:
            raise ValueError("each pool track requires slug and canonical_path")
        items.append((slug, _path_from_root(project_root, raw_path)))
    return items


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="*", type=Path, help="canonical WAV paths")
    parser.add_argument(
        "--pool-config",
        type=Path,
        help="MusicMix pool config; canonical_path entries are analyzed in listed order",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "pretrained_v10",
    )
    parser.add_argument("--model", default="harmonix-fold0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--demix-dir",
        type=Path,
        default=ROOT / ".cache" / "allin1_demix",
        help="Demucs cache root; expected layout is <root>/htdemucs/<wav stem>/{bass,drums,other,vocals}.wav",
    )
    parser.add_argument(
        "--spec-dir",
        type=Path,
        default=ROOT / ".cache" / "allin1_spec",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / ".cache" / "allin1_hf",
        help="isolated Hugging Face cache for All-In-One checkpoints",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-demucs-download",
        action="store_true",
        help="allow the official HTDemucs step to download its checkpoint when stems are absent",
    )
    parser.add_argument(
        "--fail-on-unavailable",
        action="store_true",
        help="return exit code 2 if any track cannot run the official provider",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.pool_config and args.audio:
        raise SystemExit("use --pool-config or positional audio paths, not both")
    if args.pool_config:
        pool_items = load_pool(args.pool_config.resolve())
    else:
        pool_items = [(path.stem, path if path.is_absolute() else ROOT / path) for path in args.audio]
    if not pool_items:
        raise SystemExit("provide at least one canonical WAV or --pool-config")
    if args.threads < 1:
        raise SystemExit("--threads must be positive")

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = AllInOneConfig(
        model=args.model,
        device=args.device,
        threads=args.threads,
        demix_dir=args.demix_dir if args.demix_dir.is_absolute() else ROOT / args.demix_dir,
        spec_dir=args.spec_dir if args.spec_dir.is_absolute() else ROOT / args.spec_dir,
        cache_dir=args.cache_dir if args.cache_dir.is_absolute() else ROOT / args.cache_dir,
        overwrite=args.overwrite,
        allow_demucs_download=args.allow_demucs_download,
    )
    provider = AllInOneProvider(config)
    manifest: dict[str, Any] = {
        "schema_version": "musicmix.pretrained_structure_run.1",
        "status": "complete",
        "provider": provider.provenance,
        "provider_repository": ALL_IN_ONE_REPOSITORY,
        "model": args.model,
        "device": args.device,
        "threads": args.threads,
        "heuristic_sections": False,
        "tracks": [],
    }
    unavailable = False
    for slug, path in pool_items:
        output_path = output_dir / f"{slug}.json"
        item: dict[str, Any] = {
            "slug": slug,
            "audio_path": str(path),
            "output_path": str(output_path),
        }
        try:
            track = load_audio(path)
            analysis = analyze_structure(
                track,
                provider=provider,
                include_heuristic_sections=False,
            )
            write_structure(output_path, analysis)
            item.update(
                {
                    "status": analysis.status,
                    "track_id": analysis.track_id,
                    "beat_count": len(analysis.beats),
                    "downbeat_count": len(analysis.downbeats),
                    "section_count": len(analysis.sections),
                    "warnings": list(analysis.warnings),
                }
            )
            unavailable = unavailable or analysis.status == "unavailable"
        except Exception as exc:
            unavailable = True
            item.update(
                {
                    "status": "unavailable",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        manifest["tracks"].append(item)

    if unavailable:
        manifest["status"] = "complete_with_unavailable"
    # Runtime versions, checkpoint paths, and compatibility provenance are
    # populated lazily after the first official provider call.
    manifest["provider"] = provider.provenance
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2))
    return 2 if unavailable and args.fail_on_unavailable else 0


if __name__ == "__main__":
    raise SystemExit(main())
