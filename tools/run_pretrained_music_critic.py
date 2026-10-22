"""Run the released SongEval critic on a short diagnostic audio window.

This is a thin inference wrapper around the official SongEval ``Generator``
and ``MuQ`` feature extractor.  It deliberately limits each input to 30 s:
SongEval was trained for holistic full-song aesthetics, so short-window
scores are diagnostic measurements around a join and are not a seam-validated
acceptance signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / ".cache" / "song_eval_runtime"
SONGEVAL_ROOT = RUNTIME_ROOT / "SongEval"
MUQ_ROOT = RUNTIME_ROOT / "MuQ-large-msd-iter"
HEAD_PATH = RUNTIME_ROOT / "weights" / "songeval-model.safetensors"


def _prepend_runtime_paths() -> None:
    """Reuse the existing CPU Torch and staged audio Python runtimes."""

    paths = [
        RUNTIME_ROOT / "site-packages",
        PROJECT_ROOT / ".cache" / "cue_runtime" / "numpy126",
        PROJECT_ROOT / ".cache" / "cue_runtime" / "site-packages",
        PROJECT_ROOT / ".cache" / "structure_runtime" / "packages",
        PROJECT_ROOT / ".cache" / "structure_runtime" / "Lib" / "site-packages",
    ]
    for path in reversed(paths):
        if path.is_dir():
            sys.path.insert(0, str(path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_models(threads: int):
    _prepend_runtime_paths()
    import torch
    from safetensors.torch import load_file

    # The staged SongEval source is the official model.py, kept outside src/.
    sys.path.insert(0, str(SONGEVAL_ROOT))
    from model import Generator  # type: ignore
    from muq import MuQ

    torch.set_num_threads(max(1, threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    head = Generator(
        in_features=1024,
        ffd_hidden_size=4096,
        num_classes=5,
        attn_layer_num=4,
    )
    state = load_file(str(HEAD_PATH), device="cpu")
    # The released head is expected to match the official Generator exactly.
    # A mismatch must never yield plausible-looking but invalid scores.
    head.load_state_dict(state, strict=True)
    head.eval()

    # MuQ.from_pretrained reads only the staged local directory; no implicit
    # network fetch is permitted during an inference run.
    muq = MuQ.from_pretrained(str(MUQ_ROOT), local_files_only=True)
    muq.eval()
    return torch, head, muq


def _score_one(torch, head, muq, audio_path: Path, start: float, duration: float) -> dict[str, Any]:
    import librosa

    wav, sample_rate = librosa.load(
        str(audio_path),
        sr=24000,
        mono=True,
        offset=max(0.0, start),
        duration=duration,
    )
    if wav.size == 0:
        raise ValueError(f"empty audio window: {audio_path}")
    audio = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        output = muq(audio, output_hidden_states=True)
        features = output["hidden_states"][6]
        values = head(features).squeeze(0)
    scores = {
        "Coherence": round(float(values[0].item()), 6),
        "Musicality": round(float(values[1].item()), 6),
        "Memorability": round(float(values[2].item()), 6),
        "Clarity": round(float(values[3].item()), 6),
        "Naturalness": round(float(values[4].item()), 6),
    }
    return {
        "audio_path": str(audio_path),
        "audio_sha256": _sha256(audio_path),
        "window_start_s": round(max(0.0, start), 6),
        "window_duration_s": round(float(len(wav) / sample_rate), 6),
        "sample_rate_hz": int(sample_rate),
        "scores_raw": scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", action="append", required=False, type=Path)
    parser.add_argument(
        "--cases",
        type=Path,
        help="JSON case manifest with cases[{label,path,note}]; replaces --audio",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window-start", type=float, default=0.0)
    parser.add_argument("--window-seconds", type=float, default=20.0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    if not 0.0 < args.window_seconds <= 30.0:
        parser.error("--window-seconds must be >0 and <=30")
    if args.cases is not None:
        if not args.cases.is_file():
            parser.error(f"case manifest not found: {args.cases}")
        case_manifest = json.loads(args.cases.read_text(encoding="utf-8"))
        case_specs = [
            {
                "label": str(case.get("label", "unlabeled")),
                "note": str(case.get("note", "")),
                "audio": Path(case["path"]),
            }
            for case in case_manifest.get("cases", [])
        ]
        if not case_specs:
            parser.error("case manifest contains no cases")
    else:
        if not args.audio:
            parser.error("provide --audio at least once or --cases")
        case_specs = [{"label": "unlabeled", "note": "", "audio": audio} for audio in args.audio]

    for spec in case_specs:
        audio = spec["audio"]
        if not audio.is_file():
            parser.error(f"audio file not found: {audio}")
    required = [HEAD_PATH, MUQ_ROOT / "config.json", MUQ_ROOT / "model.safetensors"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("missing staged model assets: " + ", ".join(missing))

    _prepend_runtime_paths()
    import torch as torch_module

    started = time.perf_counter()
    payload = {
        "schema_version": "musicmix.pretrained_music_critic.1",
        "status": "running",
        "critic": "SongEval",
        "task_scope": {
            "measurement": "released SongEval five-axis aesthetic score",
            "window_limit_s": 30.0,
            "interpretation": "diagnostic short-window score around a join",
            "full_song_gap": "SongEval was trained/evaluated on full-song aesthetics; no cross-song seam validation is claimed",
            "order_counterfactual": "run separate files with reversed order; raw differences are controls, not labels",
        },
        "provenance": {
            "head_source": "https://github.com/ASLP-lab/SongEval/blob/main/ckpt/model.safetensors",
            "head_source_revision": "848fb2ff3a2a9d64dcb20d46f07238c28abd7add",
            "head_sha256": _sha256(HEAD_PATH),
            "head_license": "CC-BY-NC-SA-4.0",
            "backbone_source": "https://huggingface.co/OpenMuQ/MuQ-large-msd-iter",
            "backbone_source_revision": "0562a57814f6f8bbd9fdea0a25921a2fce1a841a",
            "backbone_sha256": _sha256(MUQ_ROOT / "model.safetensors"),
            "backbone_license": "CC-BY-NC-4.0",
            "device": "cpu",
            "torch_version": torch_module.__version__,
        },
        "runtime": {
            "threads": max(1, args.threads),
            "elapsed_s": round(time.perf_counter() - started, 3),
        },
        "results": [],
    }
    if args.cases is not None:
        payload["case_manifest"] = str(args.cases)
    try:
        torch, head, muq = _load_models(args.threads)
        results = []
        for spec in case_specs:
            result = _score_one(
                torch, head, muq, spec["audio"], args.window_start, args.window_seconds
            )
            result["label"] = spec["label"]
            result["note"] = spec["note"]
            results.append(result)
        payload["results"] = results
        payload["status"] = "complete"
    except RuntimeError as exc:
        # Covers strict head-load mismatches and runtime/ABI failures. Keep a
        # machine-readable artifact so a failed run cannot be mistaken for a
        # completed critic result.
        payload["status"] = "unavailable"
        payload["failure"] = {
            "type": "runtime_error",
            "message": str(exc),
            "head_mismatch": "Error(s) in loading state_dict" in str(exc),
        }
    except Exception as exc:  # pragma: no cover - depends on local model runtime
        payload["status"] = "unavailable"
        payload["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    payload["runtime"]["elapsed_s"] = round(time.perf_counter() - started, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
