#!/usr/bin/env python3
"""Run the optional CPJKU Beat This! provider and write M02 event JSON.

The script keeps model imports out of ``musicmix.structure``.  The default
checkpoint is the official ``small0`` model downloaded by the isolated runtime
setup used for this project; a missing runtime or checkpoint produces an
explicit ``status=unavailable`` JSON file rather than falling back to the
onset-grid analyzer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import queue
import sys
import threading
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from musicmix.audio import AudioTrack, load_audio  # noqa: E402
from musicmix.structure import (  # noqa: E402
    BEAT_THIS_CHECKPOINT_ROOT,
    BEAT_THIS_PAPER,
    BEAT_THIS_REPOSITORY,
    ProviderPrediction,
    TimedEvent,
    analyze_structure,
    write_structure,
)


@dataclass(frozen=True)
class BeatThisConfig:
    """Configuration captured in output provenance."""

    checkpoint_path: str
    model_name: str = "small0"
    device: str = "cpu"
    threads: int = 1
    fps: int = 50


class BeatThisProvider:
    """Lazy, CPU-safe adapter around the official CPJKU Beat This API."""

    def __init__(self, config: BeatThisConfig) -> None:
        self.config = config
        self._model: Any = None
        self._postprocessor: Any = None
        self._torch: Any = None
        # A timed-out inference runs on a daemon thread that cannot be safely
        # cancelled.  Once that happens, refuse all later predictions so a
        # subsequent track can never share the provider while the old call is
        # still using its model.
        self._disabled = False
        checkpoint = Path(config.checkpoint_path)
        self._checkpoint_sha256 = (
            _sha256(checkpoint) if checkpoint.is_file() else None
        )
        self.provenance = {
            "provider": "CPJKU Beat This!",
            "provider_id": "beat_this",
            "provider_repository": BEAT_THIS_REPOSITORY,
            "provider_paper": BEAT_THIS_PAPER,
            "provider_version": _package_version("beat-this"),
            "model": config.model_name,
            "model_checkpoint": str(checkpoint.resolve()),
            "model_source_url": f"{BEAT_THIS_CHECKPOINT_ROOT}/{config.model_name}.ckpt",
            "model_checkpoint_sha256": self._checkpoint_sha256,
            "device": config.device,
            "threads": int(config.threads),
            "frame_rate_hz": int(config.fps),
            "confidence_source": "sigmoid(frame logits at provider event frame)",
            "confidence_note": (
                "Sigmoid frame activation is not a calibrated probability of "
                "event timing accuracy."
            ),
            "section_labels": False,
        }

    def _load(self) -> None:
        if self._disabled:
            raise RuntimeError(
                "Beat This provider disabled after a prior timeout; "
                "restart the provider before analyzing another track"
            )
        if self._model is not None:
            return
        try:
            import torch
            from beat_this.inference import Audio2Frames
            from beat_this.inference import Postprocessor
        except Exception as exc:  # optional provider boundary
            raise RuntimeError(
                "Beat This runtime unavailable; install beat-this, torch, "
                "torchaudio, soxr, einops, and rotary-embedding-torch"
            ) from exc
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA device is unavailable")
        torch.set_num_threads(max(1, int(self.config.threads)))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch rejects changing this after parallel work has started;
            # intra-op threads are still bounded above.
            pass
        checkpoint = Path(self.config.checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Beat This checkpoint not found: {checkpoint}")
        self._torch = torch
        self._model = Audio2Frames(
            checkpoint_path=str(checkpoint), device=self.config.device, float16=False
        )
        self._postprocessor = Postprocessor(type="minimal", fps=self.config.fps)

    def predict(self, samples: np.ndarray, sample_rate: int) -> ProviderPrediction:
        self._load()
        signal = np.asarray(samples, dtype=np.float64)
        if signal.ndim != 2 or signal.shape[1] < 1:
            raise ValueError("Beat This adapter expects (frames, channels) samples")
        beat_logits, downbeat_logits = self._model(signal, int(sample_rate))
        beat_times, downbeat_times = self._postprocessor(beat_logits, downbeat_logits)
        beat_values = _events_from_times_and_logits(
            beat_times, beat_logits, self._torch, fps=self.config.fps
        )
        downbeat_values = _events_from_times_and_logits(
            downbeat_times, downbeat_logits, self._torch, fps=self.config.fps
        )
        return ProviderPrediction(
            beats=tuple(beat_values),
            downbeats=tuple(downbeat_values),
            provenance=self.provenance,
            warnings=(
                "Beat This! provides beat/downbeat predictions but no functional section labels.",
                "Beat/downbeat confidence is sigmoid model activation, not calibrated timing accuracy.",
            ),
        )

    @property
    def disabled(self) -> bool:
        """Whether an uncancellable timeout permanently stopped this provider."""

        return self._disabled


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _events_from_times_and_logits(
    times: Any, logits: Any, torch: Any, *, fps: float = 50.0
) -> list[TimedEvent]:
    """Pair provider postprocessed times with sigmoid model confidences."""

    time_array = np.asarray(times, dtype=float).reshape(-1)
    logits_array = logits.detach().cpu().numpy().reshape(-1)
    events: list[TimedEvent] = []
    for value in time_array.tolist():
        if not np.isfinite(value) or value < 0.0:
            continue
        frame = int(round(float(value) * float(fps)))
        frame = max(0, min(frame, logits_array.size - 1))
        logit = float(logits_array[frame])
        # This is the model's sigmoid output, retained as evidence rather than
        # a calibrated human confidence estimate.
        confidence = float(1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0))))
        events.append(TimedEvent(float(value), confidence))
    return events


def _predict_with_timeout(
    provider: BeatThisProvider,
    track: AudioTrack,
    timeout_seconds: float,
) -> ProviderPrediction:
    """Run provider work on a daemon thread so a timeout returns a JSON result."""

    if provider.disabled:
        raise RuntimeError(
            "Beat This provider is disabled after a prior timeout; "
            "no concurrent inference was started"
        )
    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            result_queue.put((True, provider.predict(track.samples, track.sample_rate)))
        except BaseException as exc:  # propagate to the caller as data
            result_queue.put((False, exc))

    thread = threading.Thread(target=worker, daemon=True, name="beat-this")
    thread.start()
    thread.join(timeout=max(0.0, float(timeout_seconds)))
    if thread.is_alive():
        # Python threads cannot be forcefully cancelled.  Mark the provider
        # unusable before returning so later tracks never call into a model
        # that may still be running on this daemon thread.
        provider._disabled = True
        raise TimeoutError(f"Beat This provider exceeded {timeout_seconds:.1f}s")
    try:
        succeeded, value = result_queue.get_nowait()
    except queue.Empty as exc:
        raise RuntimeError("Beat This provider returned no result") from exc
    if not succeeded:
        raise value
    return value


def _unavailable_provider(config: BeatThisConfig, exc: BaseException) -> Any:
    """Build a provider-like callable carrying provenance after a failure."""

    class FailedProvider:
        provenance = BeatThisProvider(config).provenance

        def predict(self, samples: np.ndarray, sample_rate: int) -> ProviderPrediction:
            raise exc

    return FailedProvider()


def analyze_one(
    path: Path,
    output_path: Path,
    provider: BeatThisProvider | None,
    timeout_seconds: float,
    include_heuristic_sections: bool,
) -> dict[str, Any]:
    track = load_audio(path)
    if provider is None:
        result = analyze_structure(
            track, provider=None, include_heuristic_sections=include_heuristic_sections
        )
    else:
        try:
            prediction = _predict_with_timeout(provider, track, timeout_seconds)
            result = analyze_structure(
                track,
                provider=lambda _samples, _sample_rate: prediction,
                include_heuristic_sections=include_heuristic_sections,
            )
        except BaseException as exc:
            failed = _unavailable_provider(provider.config, exc)
            result = analyze_structure(
                track,
                provider=failed,
                include_heuristic_sections=include_heuristic_sections,
            )
        write_structure(output_path, result)
        return result.to_dict()
    write_structure(output_path, result)
    return result.to_dict()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="+", type=Path, help="canonical WAV files")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "structure_v2")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / ".cache" / "structure_runtime" / "models" / "beat_this-small0.ckpt",
    )
    parser.add_argument("--model", default="small0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--no-heuristic-sections", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads < 1 or args.timeout <= 0.0:
        raise SystemExit("--threads must be positive and --timeout must be > 0")
    config = BeatThisConfig(
        checkpoint_path=str(args.checkpoint),
        model_name=args.model,
        device=args.device,
        threads=args.threads,
    )
    provider = BeatThisProvider(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, path in enumerate(args.audio):
        output_name = f"{chr(ord('a') + index)}.json" if index < 26 else f"{index + 1}.json"
        outputs.append(
            analyze_one(
                path,
                args.output_dir / output_name,
                provider,
                args.timeout,
                not args.no_heuristic_sections,
            )
        )
    print(json.dumps({"status": "complete", "outputs": outputs}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
