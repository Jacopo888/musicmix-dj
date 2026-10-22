"""Run the official ETH-DISCO CUE-DETR checkpoint on MusicMix canonical audio.

The detector and its preprocessing are intentionally taken from the pinned
``ETH-DISCO/cue-detr`` repository.  This wrapper only supplies canonical-pool
I/O, bounded CPU batches, provenance, and an auditable raw/normalised score
record.  CUE-DETR predicts cue opportunities within one track; this script
does not infer cross-song compatibility or a successful transition.
"""

from __future__ import annotations

import argparse
import hashlib
import ast
import json
import math
import os
import sys
import wave
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

def _configure_runtime_paths() -> None:
    # Do not alter package discovery during test collection or --help.
    # The last prepended directory has priority: NumPy1.26, then staged deps.
    for path in (ROOT / ".cache/cue_runtime/site-packages", ROOT / ".cache/cue_runtime/numpy126"):
        if path.is_dir():
            sys.path.insert(0, str(path))

UPSTREAM_DIR = ROOT / ".cache/cue_detr_upstream"
UPSTREAM_SCRIPT = UPSTREAM_DIR / "cue_points.py"
UPSTREAM_LICENSE = UPSTREAM_DIR / "LICENSE"


def _load_upstream_constants() -> dict[str, Any]:
    """Read constants from the pinned upstream example without importing its CLI."""

    if not UPSTREAM_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Pinned upstream source is missing: {UPSTREAM_SCRIPT}. "
            "Clone ETH-DISCO/cue-detr at the recorded revision first."
        )
    tree = ast.parse(UPSTREAM_SCRIPT.read_text(encoding="utf-8"), filename=str(UPSTREAM_SCRIPT))
    values: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"OVERLAP", "W_BBOX", "W_WIN", "PADDING"}:
                values[node.targets[0].id] = ast.literal_eval(node.value)
    required = {"OVERLAP", "W_BBOX", "W_WIN", "PADDING"}
    missing = required.difference(values)
    if missing:
        raise RuntimeError(f"Pinned upstream constants missing from {UPSTREAM_SCRIPT}: {sorted(missing)}")
    return {
        "overlap": float(values["OVERLAP"]),
        "w_bbox": int(values["W_BBOX"]),
        "w_win": int(values["W_WIN"]),
        "padding": int(values["PADDING"]),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_audio_info(path: Path) -> tuple[int, int, float]:
    """Read immutable canonical WAV metadata without resampling it."""

    with wave.open(str(path), "rb") as handle:
        sample_rate = int(handle.getframerate())
        frames = int(handle.getnframes())
    return sample_rate, frames, frames / sample_rate if sample_rate else math.nan


def _hf_revision(checkpoint: Path) -> str | None:
    """Return a local Hugging Face snapshot revision when snapshot metadata exists."""

    cache_root = checkpoint / ".cache/huggingface/trees"
    revisions = sorted(cache_root.iterdir()) if cache_root.is_dir() else []
    return revisions[-1].stem.removesuffix(".json") if revisions else None


def _rgb_image_from_audio(y: np.ndarray, upstream: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """The preprocessing block copied in behaviour from upstream cue_points.py."""

    import librosa  # type: ignore
    import numpy as np  # type: ignore
    from matplotlib import cm  # type: ignore
    from PIL import Image  # type: ignore

    # Upstream librosa.load uses its default sr=22050 and mono=True.  The
    # explicit sr here is part of the official mel-spectrogram recipe.
    m = librosa.feature.melspectrogram(y=y, sr=22050, n_fft=2048)
    m_db = librosa.power_to_db(m, ref=np.max)

    arr = m_db[::-1]
    sm = cm.ScalarMappable(cmap="viridis")
    sm.set_clim(arr.min(), arr.max())
    rgba = sm.to_rgba(arr, bytes=True)
    rgb_shape = (rgba.shape[1], rgba.shape[0])
    rgba = np.require(rgba, requirements="C")
    im = Image.frombuffer("RGBA", rgb_shape, rgba, "raw", "RGBA", 0, 1)
    image = np.array(im)[:, :, :3]

    image_w = int(image.shape[1])
    image_w_padded = image_w + int(upstream["padding"])
    step = upstream["w_win"] * (1.0 - upstream["overlap"])
    n_windows = int(np.floor(image_w_padded / step))
    images: list[np.ndarray] = []
    borders: list[int] = []

    # This is the upstream 75%-overlap window and linear-ramp padding logic.
    for i in range(n_windows):
        left = int(np.floor(i * step)) - upstream["padding"]
        right = left + upstream["w_win"]
        borders.append(left)
        if left < 0:
            segment = image[:, :right]
            segment = np.pad(segment, ((0, 0), (-left, 0), (0, 0)), mode="linear_ramp")
        elif right > image.shape[1]:
            segment = image[:, left:]
            segment = np.pad(
                segment,
                ((0, 0), (0, right - left - segment.shape[1]), (0, 0)),
                mode="linear_ramp",
            )
        else:
            segment = image[:, left:right]
        images.append(segment)

    return image, {
        "mel_sample_rate": 22050,
        "mel_n_fft": 2048,
        "mel_bins": int(m.shape[0]),
        "spectrogram_frames": int(m.shape[1]),
        "image_width": image_w,
        "window_count": len(images),
        "window_height": int(upstream["w_bbox"] * 0 + 128),
        "window_width": int(upstream["w_win"]),
        "window_overlap": float(upstream["overlap"]),
        "padding_pixels": int(upstream["padding"]),
        "window_borders": borders,
        "images": images,
    }


def _normalise_scores(scores: list[float]) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the upstream min-max score scaling, recording degenerate inputs."""

    import numpy as np  # type: ignore

    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return values, {"min": None, "max": None, "degenerate": False}
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    denominator = maximum - minimum
    # The upstream lambda is undefined for a constant score vector.  Keep the
    # run JSON-valid and explicit instead of emitting NaN into downstream data.
    if denominator == 0.0:
        scaled = np.zeros_like(values)
        degenerate = True
    else:
        scaled = (values - minimum) / denominator
        degenerate = False
    return scaled, {"min": minimum, "max": maximum, "degenerate": degenerate}


def _deduplicate_peaks(peaks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse duplicate frames caused by overlapping windows deterministically."""

    selected: dict[int, dict[str, Any]] = {}
    for peak in peaks:
        frame = int(peak["spectrogram_frame"])
        previous = selected.get(frame)
        if previous is None or float(peak["raw_score"]) > float(previous["raw_score"]):
            selected[frame] = peak
    return [selected[frame] for frame in sorted(selected)]


def _predict_track(
    *,
    track: dict[str, Any],
    processor: DetrImageProcessor,
    model: DetrForObjectDetection,
    upstream: dict[str, Any],
    sensitivity: float,
    radius: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    import librosa  # type: ignore
    import torch  # type: ignore
    from scipy.signal import find_peaks  # type: ignore

    path = Path(track["canonical_path"])
    record: dict[str, Any] = {
        "status": "error",
        "slug": track.get("slug"),
        "artist": track.get("artist"),
        "title": track.get("title"),
        "canonical_path": str(path),
    }
    if not path.is_file():
        record["error"] = f"canonical audio missing: {path}"
        return record

    canonical_sha = _sha256(path)
    sample_rate, duration_samples, duration_seconds = _canonical_audio_info(path)
    record.update(
        {
            "track_id": canonical_sha,
            "canonical_sha256": canonical_sha,
            "sample_rate": sample_rate,
            "duration_samples": duration_samples,
            "duration_seconds": duration_seconds,
        }
    )

    # Faithful upstream decode: default librosa.load => mono, 22050 Hz.
    y, sr = librosa.load(str(path))
    image, prep = _rgb_image_from_audio(y, upstream)
    images = prep.pop("images")
    borders = prep.pop("window_borders")

    raw_scores: list[float] = []
    positions: list[int] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            encoding = processor.preprocess(batch, do_resize=False, return_tensors="pt")
            pixel_values = encoding["pixel_values"].to(device)
            outputs = model(pixel_values)
            target_sizes = [(128, 355)] * pixel_values.shape[0]
            predictions = processor.post_process_object_detection(outputs, 0, target_sizes)
            for prediction, border in zip(predictions, borders[start : start + batch_size]):
                raw_scores.extend(float(value) for value in prediction["scores"].tolist())
                pos = (prediction["boxes"][:, 0] + prediction["boxes"][:, 2]) // 2 + border
                positions.extend(int(value) for value in pos.long().tolist())

    scaled, score_stats = _normalise_scores(raw_scores)
    # The official code sorts positions together with min-max-normalised scores
    # and runs find_peaks(distance=radius) on this *prediction sequence*.  The
    # radius is therefore not converted from bars or seconds.
    ordered = sorted(zip(positions, raw_scores, scaled.tolist()), key=lambda row: (row[0], row[2]))
    sorted_positions = [row[0] for row in ordered]
    sorted_raw = [row[1] for row in ordered]
    sorted_normalized = [row[2] for row in ordered]
    peak_idx, _ = find_peaks(
        sorted_normalized,
        height=sensitivity,
        distance=radius,
    )

    raw_peaks: list[dict[str, Any]] = []
    for index in peak_idx.tolist():
        frame = int(sorted_positions[index])
        time_seconds = float(librosa.frames_to_time(frame))
        raw_peaks.append(
            {
                "prediction_index_sorted": int(index),
                "spectrogram_frame": frame,
                "time_seconds": time_seconds,
                "canonical_sample_approx": int(round(time_seconds * sample_rate)),
                "raw_score": float(sorted_raw[index]),
                "normalized_score": float(sorted_normalized[index]),
            }
        )
    deduplicated_peaks = _deduplicate_peaks(raw_peaks)
    # Preserve every upstream peak in raw_peaks, while exposing only canonical
    # timeline coordinates to downstream matchers.  Padded windows can produce
    # negative frame coordinates; those remain audit evidence and are excluded
    # from the valid cue collection.
    cues = [
        peak
        for peak in deduplicated_peaks
        if 0.0 <= float(peak["time_seconds"]) <= duration_seconds
    ]

    record.update(
        {
            "status": "complete",
            "decode_sample_rate": int(sr),
            "preprocessing": prep,
            "score_scaling": {
                "method": "upstream_minmax",
                "raw_prediction_count": len(raw_scores),
                "ordered_prediction_count": len(ordered),
                **score_stats,
            },
            "peak_filter": {
                "sensitivity": float(sensitivity),
                "radius": int(radius),
                "radius_units": "sorted_prediction_sequence_indices",
                "upstream_cli_label": "minimum distance in bars (upstream label; implementation uses scipy.find_peaks distance on sorted prediction sequence)",
                "raw_peak_count": len(raw_peaks),
                "deduplicated_peak_count": len(deduplicated_peaks),
                "duplicate_peak_count": len(raw_peaks) - len(deduplicated_peaks),
                "valid_peak_count": len(cues),
                "invalid_coordinate_peak_count": len(deduplicated_peaks) - len(cues),
            },
            # Keep the undeduplicated peak audit so any collapsing caused by
            # overlapping windows is visible to downstream reviewers.
            "raw_peaks": raw_peaks,
            "cues": cues,
        }
    )
    return record


def _load_pool(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tracks: list[dict[str, Any]] = []
    for entry in payload.get("tracks", []):
        item = dict(entry)
        item["canonical_path"] = str((ROOT / item["canonical_path"]).resolve())
        tracks.append(item)
    return tracks


def _load_model(checkpoint: Path, processor_name: str, device: torch.device) -> tuple[Any, Any, dict[str, Any]]:
    from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor  # type: ignore

    processor = DetrImageProcessor.from_pretrained(processor_name)
    config = DetrConfig.from_pretrained(str(checkpoint), local_files_only=True)
    original_use_pretrained_backbone = bool(config.use_pretrained_backbone)
    # The official checkpoint contains all ResNet weights.  Suppress timm's
    # redundant ImageNet bootstrap download; architecture remains use_timm=True
    # and loading_info below proves every checkpoint tensor was consumed.
    config.use_pretrained_backbone = False
    model, loading_info = DetrForObjectDetection.from_pretrained(
        str(checkpoint),
        config=config,
        local_files_only=True,
        output_loading_info=True,
    )
    model.to(device)
    model.eval()
    load_summary = {
        "config_use_timm_backbone": bool(config.use_timm_backbone),
        "config_original_use_pretrained_backbone": original_use_pretrained_backbone,
        "config_effective_use_pretrained_backbone": bool(config.use_pretrained_backbone),
        "missing_keys": list(loading_info.get("missing_keys", [])),
        "unexpected_keys": list(loading_info.get("unexpected_keys", [])),
        "mismatched_keys": list(loading_info.get("mismatched_keys", [])),
        "error_msgs": list(loading_info.get("error_msgs", [])),
    }
    if any(load_summary[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Checkpoint did not load cleanly: {load_summary}")
    return processor, model, load_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-config", type=Path, default=ROOT / "configs/music_pool_v8.json")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/pretrained_v10/cue_detr.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / ".cache/cue_runtime/checkpoint")
    parser.add_argument("--processor", default="facebook/detr-resnet-50")
    parser.add_argument("--sensitivity", type=float, default=0.9)
    parser.add_argument("--radius", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--only-slug", action="append", default=[])
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1 or args.radius < 1:
        parser.error("batch-size, threads, and radius must be >= 1")

    args.out = args.out if args.out.is_absolute() else ROOT / args.out
    if args.out.exists():
        parser.error("Output exists; choose a new path to preserve frozen inference")
    _configure_runtime_paths()
    import librosa  # type: ignore
    import torch  # type: ignore
    args.checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    upstream = _load_upstream_constants()
    processor, model, loading_info = _load_model(args.checkpoint, args.processor, device)
    tracks = _load_pool(args.pool_config if args.pool_config.is_absolute() else ROOT / args.pool_config)
    if args.only_slug:
        selected = set(args.only_slug)
        tracks = [track for track in tracks if track.get("slug") in selected]

    code_hash = _sha256(Path(__file__).resolve())
    upstream_hash = _sha256(UPSTREAM_SCRIPT)
    license_hash = _sha256(UPSTREAM_LICENSE)
    checkpoint_hash = _sha256(args.checkpoint / "model.safetensors")
    config_hash = _sha256(args.checkpoint / "config.json")
    revision = _hf_revision(args.checkpoint)
    output_tracks: list[dict[str, Any]] = []
    for track in tracks:
        print(json.dumps({"event": "started", "slug": track.get("slug")}), flush=True)
        try:
            output = _predict_track(
                track=track,
                processor=processor,
                model=model,
                upstream=upstream,
                sensitivity=args.sensitivity,
                radius=args.radius,
                batch_size=args.batch_size,
                device=device,
            )
        except Exception as exc:  # keep remaining pool tracks auditable
            output = {
                "status": "error",
                "slug": track.get("slug"),
                "canonical_path": track.get("canonical_path"),
                "error": f"{type(exc).__name__}: {exc}",
            }
        output_tracks.append(output)
        print(
            json.dumps(
                {
                    "event": "finished",
                    "slug": track.get("slug"),
                    "status": output.get("status"),
                    "cues": len(output.get("cues", [])),
                }
            ),
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "musicmix.pretrained-cue-detr.1",
        "status": "complete" if all(item.get("status") == "complete" for item in output_tracks) else "partial",
        "task_scope": "single-track cue opportunity detection; cross-song compatibility is not evaluated",
        "pool_config": str((args.pool_config if args.pool_config.is_absolute() else ROOT / args.pool_config).resolve()),
        "pool_config_sha256": _sha256(args.pool_config if args.pool_config.is_absolute() else ROOT / args.pool_config),
        "runtime": {
            "device": str(device),
            "torch_threads": int(args.threads),
            "torch_interop_threads": 1,
            "batch_size": int(args.batch_size),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "librosa": librosa.__version__,
        },
        "upstream": {
            "repository": "https://github.com/ETH-DISCO/cue-detr",
            "commit": "d0462856ed2f59a1fb65267cfbe87340a65ad1bb",
            "script": str(UPSTREAM_SCRIPT),
            "script_sha256": upstream_hash,
            "license": str(UPSTREAM_LICENSE),
            "license_sha256": license_hash,
            "constants": upstream,
        },
        "checkpoint": {
            "repository": "https://huggingface.co/disco-eth/cue-detr",
            "revision": revision,
            "path": str(args.checkpoint.resolve()),
            "model_file": "model.safetensors",
            "model_sha256": checkpoint_hash,
            "config_sha256": config_hash,
            "loading_info": loading_info,
        },
        "processor": {"id": args.processor},
        "parameters": {
            "sensitivity": float(args.sensitivity),
            "radius": int(args.radius),
            "radius_units": "sorted_prediction_sequence_indices",
        },
        "code_sha256": code_hash,
        "tracks": output_tracks,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "written", "path": str(args.out), "status": payload["status"]}), flush=True)
    return 0 if payload["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
