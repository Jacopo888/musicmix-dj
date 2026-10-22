#!/usr/bin/env python3
"""Pinned official LinkSeg CPU pilot for one canonical MusicMix track."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any

EXPECTED_CHECKPOINT_SHA256 = "5f7b6d749719086a133ebe215c8e37a55d1089dd2ab7c334613004cdd7c2d9a9"
EXPECTED_SOURCE_SHA256 = {
    "models.py": "538614ff0fb29b5ed1b7e9de11c3fa8ec0a9fc441550f9251840d97293c37d91",
    "modules.py": "e90561c161dee8e1f914aa50042b66a6976775ebb4baa35541921429fd65b278",
    "EMA.py": "c6b9b23db7e3a286bff0b5d15a20e62fd1221a9fcba3f52a2765adc3d2107dd6",
    "post_processing.py": "4cc5633b61a739b9a9ea7f97155f78a3225d9271805a6f350686ca9a66275bc3",
}

# Keep this process bounded before importing torch/torchaudio.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("DGLBACKEND", "pytorch")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def load_beat_times(path: Path) -> tuple[list[float], dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    sr = int(obj.get("sample_rate", 48000))
    raw = obj.get("beats", [])
    times: list[float] = []
    for item in raw:
        if isinstance(item, dict):
            if "time" in item:
                times.append(float(item["time"]))
            elif "sample" in item:
                times.append(float(item["sample"]) / sr)
        elif isinstance(item, (int, float)):
            # Existing structure artifacts store samples when sample_rate is present.
            times.append(float(item) / sr)
    if len(times) < 3:
        raise ValueError(f"beat artifact has too few beat times: {len(times)}")
    return times, {
        "source_sample_rate": sr,
        "source_count": len(times),
        "source_provider": obj.get("provenance", {}).get("provider"),
        "source_provider_id": obj.get("provenance", {}).get("provider_id"),
        "source_provider_revision": obj.get("provenance", {}).get("provider_revision"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--beats-json", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--expected-track-id", required=True)
    ap.add_argument("--expected-sample-rate", type=int, required=True)
    ap.add_argument("--expected-duration-samples", type=int, required=True)
    args = ap.parse_args()

    audio = Path(args.audio).resolve()
    beats_json = Path(args.beats_json).resolve()
    source = Path(args.source).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    result: dict[str, Any] = {
        "schema_version": "musicmix.linkseg_official_pilot.v1",
        "status": "unavailable",
        "model": {
            "name": "LinkSeg model_7_classes",
            "repository": "https://github.com/morgan76/LinkSeg",
            "source_revision": "38dc0dabed14daa5fbef7a61ea3cb046517e74a0",
            "checkpoint_url": "https://raw.githubusercontent.com/morgan76/LinkSeg/38dc0dabed14daa5fbef7a61ea3cb046517e74a0/data/model_7_classes.pt",
            "source_urls": {
                name: f"https://raw.githubusercontent.com/morgan76/LinkSeg/38dc0dabed14daa5fbef7a61ea3cb046517e74a0/src/{name}"
                for name in ("models.py", "modules.py", "EMA.py", "post_processing.py")
            },
            "functional_label_map": {
                "0": "silence", "1": "verse", "2": "chorus", "3": "intro",
                "4": "outro", "5": "inst", "6": "bridge",
            },
            "interpretation": "diagnostic functional section labels; no structural-arrival or reward claim",
        },
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version,
            "threads": 1,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        },
        "inputs": {
            "audio": {"path": str(audio)},
            "beats": {"path": str(beats_json)},
        },
    }

    try:
        for p in (audio, beats_json, checkpoint):
            if not p.is_file():
                raise FileNotFoundError(str(p))
        source_files = [source / n for n in ("models.py", "modules.py", "EMA.py", "post_processing.py")]
        for p in source_files:
            if not p.is_file():
                raise FileNotFoundError(str(p))
        result["model"]["checkpoint_sha256"] = sha256_file(checkpoint)
        result["model"]["checkpoint_bytes"] = checkpoint.stat().st_size
        if result["model"]["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError("pinned checkpoint SHA256 mismatch")
        result["model"]["source_files"] = {
            p.name: {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in source_files
        }
        for p in source_files:
            if result["model"]["source_files"][p.name]["sha256"] != EXPECTED_SOURCE_SHA256[p.name]:
                raise RuntimeError(f"pinned source SHA256 mismatch: {p.name}")
        result["inputs"]["audio"].update({"sha256": sha256_file(audio), "bytes": audio.stat().st_size})
        result["inputs"]["beats"].update({"sha256": sha256_file(beats_json), "bytes": beats_json.stat().st_size})
        if result["inputs"]["audio"]["sha256"] != args.expected_track_id:
            raise RuntimeError("canonical audio SHA256 does not match expected track_id")

        # Import the pinned model source only after paths and hashes are recorded.
        sys.path.insert(0, str(source))
        import numpy as np
        import librosa
        import torch
        import dgl
        from models import FrameEncoder, LinkSeg
        # The official decoder module imports jams for its optional JAMS exporter.
        # This pilot emits canonical JSON and never calls that exporter, so keep
        # the model run independent of its unrelated pandas validation stack.
        import types
        sys.modules["jams"] = types.ModuleType("jams")
        result["runtime"]["post_processing_import"] = (
            "official post_processing.py; no-op jams module because JAMS export is not invoked"
        )
        from post_processing import post_process

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        result["runtime"].update({"torch": torch.__version__, "dgl": dgl.__version__, "numpy": np.__version__, "librosa": librosa.__version__})

        encoder = FrameEncoder(
            n_mels=64, conv_ndim=32, sample_rate=22050, n_fft=1024, hop_length=256,
            n_embedding=64, f_min=0, f_max=11025, dropout=0.1, hidden_dim=32,
            attention_ndim=32, attention_nlayers=2, attention_nheads=8,
        )
        model = LinkSeg(
            encoder, nb_ssm_classes=3, nb_section_labels=7, hidden_size=32,
            output_channels=16, dropout_gnn=0.1, dropout_cnn=0.2,
            dropout_egat=0.5, max_len=1500,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(state, (dict, OrderedDict)):
            raise TypeError(f"unexpected checkpoint type: {type(state)!r}")
        if "state_dict" in state:
            state = state["state_dict"]
        remapped = OrderedDict((key.split("network.")[-1], value) for key, value in state.items())
        incompatible = model.load_state_dict(remapped, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        result["model"]["load_missing_keys"] = missing
        result["model"]["load_unexpected_keys"] = unexpected
        if missing or unexpected:
            raise RuntimeError(f"checkpoint compatibility failure: missing={missing}, unexpected={unexpected}")
        model.load_state_dict(remapped, strict=True)
        model.eval()
        result["model"]["parameter_count"] = sum(p.numel() for p in model.parameters())

        beat_obj = json.loads(beats_json.read_text(encoding="utf-8"))
        beat_track_id = beat_obj.get("track_id")
        if beat_track_id != args.expected_track_id or beat_track_id != result["inputs"]["audio"]["sha256"]:
            raise RuntimeError("beat artifact track_id does not match canonical audio SHA256")
        if int(beat_obj.get("sample_rate", -1)) != args.expected_sample_rate:
            raise RuntimeError("beat artifact sample_rate does not match catalog/header expectation")
        if int(beat_obj.get("duration_samples", -1)) != args.expected_duration_samples:
            raise RuntimeError("beat artifact duration_samples does not match catalog expectation")
        import soundfile as sf
        audio_info = sf.info(str(audio))
        if int(audio_info.samplerate) != args.expected_sample_rate or int(audio_info.frames) != args.expected_duration_samples:
            raise RuntimeError(
                f"canonical audio header mismatch: samplerate={audio_info.samplerate}, frames={audio_info.frames}"
            )
        result["canonical_track_id"] = args.expected_track_id
        result["canonical_sample_rate"] = args.expected_sample_rate
        result["canonical_duration_samples"] = args.expected_duration_samples
        result["canonical_duration_seconds"] = args.expected_duration_samples / args.expected_sample_rate
        beat_times_raw, beat_meta = load_beat_times(beats_json)
        waveform, sample_rate = librosa.load(str(audio), sr=22050, mono=True)
        duration = float(librosa.get_duration(y=waveform, sr=sample_rate))
        beat_frames = librosa.time_to_frames(np.asarray(beat_times_raw), sr=22050, hop_length=256)
        beat_frames = librosa.util.fix_frames(beat_frames)
        beat_times = librosa.frames_to_time(beat_frames, sr=22050, hop_length=256)
        # This is the exact second conversion used by the official predict.py.
        feature_frames = librosa.time_to_frames(beat_times, sr=22050, hop_length=1)
        if len(feature_frames) > 1500:
            raise ValueError(f"official max_len exceeded: {len(feature_frames)} > 1500")
        pad_width = ((256 * 64) - 2) // 2
        waveform_padded = np.pad(waveform, pad_width=((pad_width, pad_width),), mode="edge")
        features = np.stack([waveform_padded[i:i + pad_width * 2] for i in feature_frames], axis=0)
        x = torch.tensor(features, dtype=torch.float32)
        result["inputs"]["audio"].update({"sample_rate": sample_rate, "duration_seconds": duration, "samples_22050": int(waveform.shape[0])})
        result["inputs"]["beats"].update({
            **beat_meta,
            "raw_count": len(beat_times_raw),
            "quantized_count": int(len(beat_frames)),
            "clock_conversion": "source samples -> seconds using source sample_rate; librosa.time_to_frames(sr=22050, hop_length=256); librosa.frames_to_time; official feature frame conversion hop_length=1",
            "first_quantized_seconds": float(beat_times[0]),
            "last_quantized_seconds": float(beat_times[-1]),
        })

        with torch.inference_mode():
            _, bound_curve, class_curves, _ = model(x)
        est_times, est_labels = post_process(
            str(audio), beat_times.tolist(), duration,
            bound_curve.detach().cpu().numpy(), class_curves.detach().cpu().numpy(),
            max_past=8, max_future=8, tau=0,
        )
        est_times = [float(v) for v in est_times]
        est_labels = [str(v) for v in est_labels]
        sections = [
            {
                "start_seconds": est_times[i],
                "end_seconds": est_times[i + 1],
                "start_sample": int(round(est_times[i] * args.expected_sample_rate)),
                "end_sample": (
                    args.expected_duration_samples
                    if i == min(len(est_labels), max(0, len(est_times) - 1)) - 1
                    else int(round(est_times[i + 1] * args.expected_sample_rate))
                ),
                "label": est_labels[i],
            }
            for i in range(min(len(est_labels), max(0, len(est_times) - 1)))
        ]
        result["status"] = "complete"
        result["sections"] = sections
        result["diagnostics"] = {
            "section_count": len(sections),
            "boundary_count": max(0, len(est_times) - 2),
            "labels": [s["label"] for s in sections],
            "raw_model_outputs": {"bound_curve_length": int(bound_curve.shape[0]), "class_curve_shape": list(class_curves.shape)},
        }
    except Exception as exc:
        result["status"] = "unavailable"
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}

    result["runtime"]["elapsed_seconds"] = time.time() - started
    output.write_text(json.dumps(json_safe(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output), "sections": len(result.get("sections", [])), "error": result.get("error", {}).get("message")}, ensure_ascii=False))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
