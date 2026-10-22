"""Optional adapter for the official All-In-One structure analyser.

This module deliberately keeps the ``allin1`` import at the provider boundary.
The base MusicMix package can therefore still be imported on machines without
PyTorch, NATTEN, madmom, or the All-In-One weights.  The adapter consumes the
official Python API and maps its timestamps and functional labels into the
provider-neutral :mod:`musicmix.structure` contract.  It never invents event
or section confidence values: All-In-One's public result does not expose a
calibrated confidence for those post-processed events, so confidence remains
``None``.

The upstream implementation is MIT licensed and its released Harmonix
checkpoints are hosted by the project's Hugging Face repository.  The
provenance emitted by :class:`AllInOneProvider` records these sources and the
requested model so a run can be audited without treating a missing optional
runtime as a neural result.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import collections
import collections.abc
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .audio import AudioTrack
from .structure import ProviderPrediction, TimedEvent, TimedSection


ALL_IN_ONE_REPOSITORY = "https://github.com/mir-aidj/all-in-one"
ALL_IN_ONE_PAPER = "https://arxiv.org/abs/2307.16425"
ALL_IN_ONE_LICENSE = "MIT"
ALL_IN_ONE_HUGGINGFACE = "https://huggingface.co/taejunkim/allinone"
# Commit used for the isolated allin1==1.1.0 source checkout in this pilot.
ALL_IN_ONE_SOURCE_REVISION = "18e78903c0365147a2c5d4e5e57ebf88cb7d800e"

# Keep this table in sync with the official loader.  It is used only for
# provenance and preflight diagnostics; weights are downloaded by allin1.
MODEL_CHECKPOINT_FILES: Mapping[str, str] = {
    "harmonix-fold0": "harmonix-fold0-0vra4ys2.pth",
    "harmonix-fold1": "harmonix-fold1-3ozjhtsj.pth",
    "harmonix-fold2": "harmonix-fold2-gmgo0nsy.pth",
    "harmonix-fold3": "harmonix-fold3-i92b7m8p.pth",
    "harmonix-fold4": "harmonix-fold4-1bql5qo0.pth",
    "harmonix-fold5": "harmonix-fold5-x4z5zeef.pth",
    "harmonix-fold6": "harmonix-fold6-x7t226rq.pth",
    "harmonix-fold7": "harmonix-fold7-qwwskhg6.pth",
}
MODEL_ENSEMBLES: Mapping[str, tuple[str, ...]] = {
    "harmonix-all": tuple(MODEL_CHECKPOINT_FILES),
}


@dataclass(frozen=True)
class AllInOneConfig:
    """Runtime settings for one official All-In-One provider.

    ``harmonix-fold0`` is the default to make a CPU pilot deterministic and
    avoid downloading all eight checkpoints.  ``harmonix-all`` remains
    available for the official ensemble.  Demucs and spectrogram byproducts
    are kept by default so repeated pool runs can reuse existing stems.
    """

    model: str = "harmonix-fold0"
    device: str = "cpu"
    threads: int = 1
    demix_dir: str | os.PathLike[str] = ".cache/allin1_demix"
    spec_dir: str | os.PathLike[str] = ".cache/allin1_spec"
    cache_dir: str | os.PathLike[str] | None = None
    keep_byproducts: bool = True
    overwrite: bool = False
    multiprocess: bool = False
    allow_demucs_download: bool = False
    package_root: str | os.PathLike[str] | None = None

    def __post_init__(self) -> None:
        if self.model not in MODEL_CHECKPOINT_FILES and self.model not in MODEL_ENSEMBLES:
            known = ", ".join((*MODEL_CHECKPOINT_FILES, *MODEL_ENSEMBLES))
            raise ValueError(f"unknown All-In-One model {self.model!r}; expected one of {known}")
        if int(self.threads) < 1:
            raise ValueError("threads must be positive")
        if not str(self.device):
            raise ValueError("device must be non-empty")


def _package_version() -> str | None:
    try:
        return importlib.metadata.version("allin1")
    except importlib.metadata.PackageNotFoundError:
        return None


def _checkpoint_files(model: str) -> tuple[str, ...]:
    if model in MODEL_CHECKPOINT_FILES:
        return (MODEL_CHECKPOINT_FILES[model],)
    return tuple(MODEL_CHECKPOINT_FILES[name] for name in MODEL_ENSEMBLES[model])


def _result_field(result: Any, name: str, default: Any = ()) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


class AllInOneProvider:
    """Lazy wrapper around ``allin1.analyze``.

    The provider implements ``analyze(track)`` rather than ``predict`` because
    the official package takes an audio path and performs (or reuses) Demucs
    separation internally.  Callers should pass canonical WAV paths; the
    upstream project documents WAV input for stable timing.
    """

    def __init__(
        self,
        config: AllInOneConfig,
        *,
        analyze_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self._analyze_fn = analyze_fn
        self._injected_analyze = analyze_fn is not None
        self.provenance = self._base_provenance()

    def _base_provenance(self) -> dict[str, Any]:
        checkpoint_names = _checkpoint_files(self.config.model)
        checkpoints = [f"{ALL_IN_ONE_HUGGINGFACE}/resolve/main/{name}" for name in checkpoint_names]
        payload: dict[str, Any] = {
            "provider": "All-In-One Music Structure Analyzer",
            "provider_id": "allin1",
            "provider_repository": ALL_IN_ONE_REPOSITORY,
            "provider_source_revision": ALL_IN_ONE_SOURCE_REVISION,
            "provider_paper": ALL_IN_ONE_PAPER,
            "provider_license": ALL_IN_ONE_LICENSE,
            "provider_version": _package_version(),
            "model": self.config.model,
            "model_checkpoint_files": checkpoint_names,
            "model_source_urls": checkpoints,
            "device": self.config.device,
            "threads": int(self.config.threads),
            "frame_rate_hz": 100,
            "section_labels": True,
            "confidence_source": "none (official post-processed result exposes no calibrated event confidence)",
            "demix_dir": str(Path(self.config.demix_dir).resolve()),
            "spec_dir": str(Path(self.config.spec_dir).resolve()),
            "keep_byproducts": bool(self.config.keep_byproducts),
            "allow_demucs_download": bool(self.config.allow_demucs_download),
        }
        if self.config.cache_dir is not None:
            payload["cache_dir"] = str(Path(self.config.cache_dir).resolve())
        return payload

    @property
    def available(self) -> tuple[bool, str | None]:
        """Return whether the official runtime can be imported in this process."""

        if self._analyze_fn is not None:
            return True, None
        try:
            self._load_analyze()
        except Exception as exc:  # optional dependency boundary
            return False, f"{type(exc).__name__}: {exc}"
        return True, None

    def _load_analyze(self) -> Callable[..., Any]:
        if self._analyze_fn is not None:
            return self._analyze_fn
        # The import is intentionally lazy.  In particular, allin1 imports
        # NATTEN and madmom, both of which are optional and platform-sensitive.
        # madmom 0.16.1 is the official release required by All-In-One, but
        # still imports MutableSequence from collections on Python 3.11.
        # Restore that removed alias in-process before importing the official
        # package; this does not alter the installed upstream sources.
        for name in ("MutableSequence", "MutableMapping", "Sequence"):
            if not hasattr(collections, name) and hasattr(collections.abc, name):
                setattr(collections, name, getattr(collections.abc, name))
        # The same pinned madmom release uses NumPy aliases removed in NumPy
        # 2.0.  These aliases are process-local compatibility shims only.
        for name, replacement in (("float", float), ("int", int), ("bool", bool)):
            if not hasattr(np, name):
                setattr(np, name, replacement)
        module = importlib.import_module("allin1")
        analyze_fn = getattr(module, "analyze", None)
        if not callable(analyze_fn):
            raise RuntimeError("installed allin1 package does not expose analyze()")
        self._analyze_fn = analyze_fn
        return analyze_fn

    def _record_checkpoint_paths(self) -> None:
        """Resolve the exact Hugging Face files used by the official loader.

        ``allin1.analyze`` does not return its checkpoint path.  Resolving the
        same files first makes successful runs auditable while sharing the
        normal Hugging Face cache with the upstream loader.  This is deferred
        until after the optional runtime imports successfully, so a blocked
        Windows runtime never starts a weight download.
        """

        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:  # pragma: no cover - only optional runtime
            self.provenance["checkpoint_resolution_warning"] = (
                f"{type(exc).__name__}: {exc}"
            )
            return
        paths: list[str] = []
        hashes: dict[str, str] = {}
        for filename in _checkpoint_files(self.config.model):
            kwargs: dict[str, Any] = {"repo_id": "taejunkim/allinone", "filename": filename}
            if self.config.cache_dir is not None:
                kwargs["cache_dir"] = str(Path(self.config.cache_dir).resolve())
            path = Path(hf_hub_download(**kwargs))
            paths.append(str(path.resolve()))
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[filename] = digest.hexdigest()
        self.provenance["model_checkpoint_paths"] = paths
        self.provenance["model_checkpoint_sha256"] = hashes

    def _record_runtime_provenance(self) -> None:
        """Record the isolated runtime actually imported by ``allin1``."""

        for package in ("torch", "natten", "madmom", "demucs"):
            try:
                self.provenance[f"{package}_version"] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                continue
        try:
            madmom = importlib.import_module("madmom")
            madmom_path = Path(str(getattr(madmom, "__file__", ""))).resolve()
            self.provenance["madmom_module_path"] = str(madmom_path)
            if "madmom_compat" in str(madmom_path).replace("\\", "/"):
                self.provenance["madmom_numpy_compatibility"] = (
                    "isolated copy of madmom/features/downbeats.py: "
                    "np.asarray(results, dtype=object) for NumPy ragged DBN candidates"
                )
        except Exception:
            # Runtime version metadata is audit enrichment; provider execution
            # remains governed by the official allin1 import/result.
            pass

    def _stage_input(self, track: AudioTrack) -> Path:
        """Give the official cache a content-addressed, collision-free name.

        All-In-One derives both its Demucs stem directory and its spectrogram
        filename from ``Path(audio).stem``.  Pool inputs commonly reuse names
        such as ``a.wav`` across source folders, so passing canonical paths
        directly can make one track reuse another track's byproducts.  A hard
        link keeps this staging step zero-copy on the usual same-volume
        Windows/WSL filesystem; the copy fallback is used only when links are
        unavailable.
        """

        source = Path(track.path).resolve()
        source_digest = self._sha256_file(source)
        if source_digest != track.track_id:
            raise RuntimeError(
                f"canonical audio changed after load for {source.name}: "
                f"expected {track.track_id}, got {source_digest}"
            )
        # A pilot may have produced a legacy stem directory keyed by the
        # source basename before content-addressed staging was introduced.
        # Reuse it only when a sidecar written by this adapter proves the
        # exact canonical input identity.  An unbound ``a`` directory is
        # deliberately ignored and gets a fresh collision-free staging name.
        legacy_stem_dir = Path(self.config.demix_dir).resolve() / "htdemucs" / source.stem
        legacy_meta = legacy_stem_dir / ".musicmix_input.json"
        if legacy_meta.is_file():
            try:
                metadata = json.loads(legacy_meta.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                metadata = None
            if (
                isinstance(metadata, dict)
                and str(metadata.get("track_id", "")) == track.track_id
                # The content hash is the source identity.  A path string is
                # retained for audit, but cannot be compared literally here:
                # the same cache is often inspected as ``C:\\...`` on Windows
                # and ``/mnt/c/...`` inside WSL.
                and bool(str(metadata.get("canonical_path", "")).strip())
                and all((legacy_stem_dir / f"{name}.wav").is_file() for name in ("bass", "drums", "other", "vocals"))
            ):
                return source

        stage_dir = Path(self.config.demix_dir).resolve() / ".inputs"
        stage_dir.mkdir(parents=True, exist_ok=True)
        staged = stage_dir / f"{track.track_id}.wav"
        if staged.exists() and self._sha256_file(staged) != track.track_id:
            staged.unlink()
        if not staged.exists():
            try:
                os.link(source, staged)
            except OSError:
                shutil.copy2(source, staged)
        if self._sha256_file(staged) != track.track_id:
            raise RuntimeError(f"staged input integrity check failed: {staged}")
        return staged

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _record_cache_binding(self, track: AudioTrack, analyze_path: Path) -> None:
        """Persist the source binding after a successful official run.

        The metadata is intentionally written only after ``allin1.analyze``
        returns, so a partial Demucs cache can never become trusted merely by
        entering the provider.
        """

        stem_dir = Path(self.config.demix_dir).resolve() / "htdemucs" / analyze_path.stem
        if not stem_dir.is_dir():
            return
        if not all((stem_dir / f"{name}.wav").is_file() for name in ("bass", "drums", "other", "vocals")):
            return
        metadata = {
            "schema_version": "musicmix.allin1_cache_binding.1",
            "track_id": track.track_id,
            "canonical_path": str(Path(track.path).resolve()),
            "staged_input_path": str(analyze_path),
            "stem_sha256": {
                name: self._sha256_file(stem_dir / f"{name}.wav")
                for name in ("bass", "drums", "other", "vocals")
            },
        }
        (stem_dir / ".musicmix_input.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def analyze(self, track: AudioTrack) -> ProviderPrediction:
        if not isinstance(track, AudioTrack):
            raise TypeError("AllInOneProvider expects an AudioTrack")
        if Path(track.path).suffix.lower() != ".wav":
            raise ValueError(
                "All-In-One timing runs require canonical WAV input; convert source audio during ingest"
            )

        try:
            import torch
        except Exception as exc:  # optional dependency boundary
            raise RuntimeError("All-In-One requires PyTorch in the isolated runtime") from exc
        torch.set_num_threads(max(1, int(self.config.threads)))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        # The released API currently does not expose cache_dir.  Setting
        # HF_HOME before importing/loading the model gives callers a local,
        # reproducible cache while leaving the global environment untouched.
        if self.config.cache_dir is not None:
            cache = Path(self.config.cache_dir).resolve()
            cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(cache))

        analyze_fn = self._load_analyze()
        analyze_path = Path(track.path).resolve()
        if not self._injected_analyze:
            analyze_path = self._stage_input(track)
        # The official ``allin1.analyze`` calls HTDemucs automatically when
        # the four stems are absent.  Keep a CPU pilot from silently pulling
        # the multi-gigabyte Demucs checkpoint: callers must explicitly opt
        # into that download, while an existing official stem cache remains
        # fully reusable.  Check the runtime first so a missing dependency is
        # reported accurately before a data-cache prerequisite.
        if not self._injected_analyze and not self.config.allow_demucs_download:
            stem_dir = Path(self.config.demix_dir) / "htdemucs" / analyze_path.stem
            missing = tuple(
                name for name in ("bass", "drums", "other", "vocals")
                if not (stem_dir / f"{name}.wav").is_file()
            )
            if missing:
                raise RuntimeError(
                    "official HTDemucs stems are missing for "
                    f"{Path(track.path).name}: {', '.join(missing)}; "
                    "provide the existing cache or set allow_demucs_download=True"
                )
        if not self._injected_analyze:
            self._record_runtime_provenance()
            self._record_checkpoint_paths()
        result = analyze_fn(
            str(analyze_path),
            model=self.config.model,
            device=self.config.device,
            demix_dir=str(self.config.demix_dir),
            spec_dir=str(self.config.spec_dir),
            keep_byproducts=bool(self.config.keep_byproducts),
            overwrite=bool(self.config.overwrite),
            multiprocess=bool(self.config.multiprocess),
        )
        if not self._injected_analyze:
            self._record_cache_binding(track, analyze_path)
        if isinstance(result, (list, tuple)):
            if len(result) != 1:
                raise RuntimeError(f"All-In-One returned {len(result)} results for one track")
            result = result[0]
        if result is None:
            raise RuntimeError("All-In-One returned no analysis result")

        beats = tuple(TimedEvent(float(value)) for value in _result_field(result, "beats", ()))
        downbeats = tuple(
            TimedEvent(float(value)) for value in _result_field(result, "downbeats", ())
        )
        sections: list[TimedSection] = []
        for item in _result_field(result, "segments", ()):
            start = float(_result_field(item, "start", None))
            end = float(_result_field(item, "end", None))
            label = str(_result_field(item, "label", "provider_section"))
            sections.append(TimedSection(start, end, label, None))

        provenance = dict(self.provenance)
        bpm = _result_field(result, "bpm", None)
        if bpm is not None:
            provenance["bpm"] = int(bpm)
        provenance["track_path"] = str(Path(track.path).resolve())
        if not self._injected_analyze:
            provenance["cache_input_path"] = str(analyze_path)
        return ProviderPrediction(
            beats=beats,
            downbeats=downbeats,
            sections=tuple(sections),
            provenance=provenance,
            warnings=(
                "All-In-One functional segment labels are provider model outputs; event and label confidence is unavailable and remains null.",
            ),
        )


__all__ = [
    "ALL_IN_ONE_HUGGINGFACE",
    "ALL_IN_ONE_LICENSE",
    "ALL_IN_ONE_PAPER",
    "ALL_IN_ONE_REPOSITORY",
    "ALL_IN_ONE_SOURCE_REVISION",
    "AllInOneConfig",
    "AllInOneProvider",
    "MODEL_CHECKPOINT_FILES",
    "MODEL_ENSEMBLES",
]
