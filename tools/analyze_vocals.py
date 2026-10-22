"""Run bounded vocal separation + full-song ASR and save local JSON artefacts.

The command prints counts and statuses only.  The transcript remains in the
requested JSON output and is never printed to the terminal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from musicmix.audio import load_audio  # noqa: E402
from musicmix.vocal_boundaries import (  # noqa: E402
    acoustic_intervals,
    build_boundary_document,
    choose_asr_language,
    detect_language,
    find_whisper_snapshot,
    seconds_to_samples,
    separate_vocals_demucs,
    sha256_file,
    transcribe_audio,
)


def _default_model(root: Path) -> Path:
    for name in ("base.en", "small.en"):
        snapshots = root / ".cache" / "models" / f"models--Systran--faster-whisper-{name}" / "snapshots"
        candidates = sorted(path for path in snapshots.iterdir() if path.is_dir()) if snapshots.is_dir() else []
        if candidates and (candidates[-1] / "model.bin").is_file():
            return candidates[-1]
    raise FileNotFoundError("No cached faster-whisper model; pass --model-path")


def _write_json(path: Path, document: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def analyse_one(
    label: str,
    input_path: Path,
    output_dir: Path,
    *,
    model_path: Path,
    runtime_root: Path,
    threads: int,
    skip_separation: bool,
    language: str | None = "en",
    multilingual_model_path: Path | None = None,
    existing_stem: Path | None = None,
    existing_separator: dict | None = None,
) -> dict:
    """Separazione Demucs + ASR. ``language``: codice Whisper, oppure ``None``/``"auto"`` per rilevare la
    lingua sullo stem vocale con il modello multilingue (``faster-whisper-small``); l'inglese resta su
    ``model_path`` (``small.en``) ed e' anche il fallback. Lingua usata, probabilita' e motivo vanno in
    ``backend`` e ``language_selection`` del JSON."""
    input_path = input_path.resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    separator: dict = {
        "status": "skipped" if skip_separation else "pending",
        "method": None,
        "model": None,
        "input_path": str(input_path),
    }
    asr_path = input_path
    source_kind = "mixed_audio_fallback"
    stem_path = output_dir / f"{label}_vocals.wav"
    if existing_stem is not None:  # rifare solo l'ASR su uno stem Demucs gia' calcolato (non si ripete la separazione)
        stem_path = Path(existing_stem).resolve(strict=True)
        separator = dict(existing_separator or {"status": "complete", "method": "demucs", "model": "htdemucs"})
        separator["reused_stem_path"] = str(stem_path)
        separator["reused_stem_sha256"] = sha256_file(stem_path)
        asr_path = stem_path
        source_kind = "separated_vocals"
    elif not skip_separation:
        try:
            separator = separate_vocals_demucs(
                input_path, stem_path, runtime_root=runtime_root,
                model_name="htdemucs", threads=threads,
            )
            separator["input_path"] = str(input_path)
            separator["output_path"] = str(stem_path.resolve())
            separator["output_sha256"] = sha256_file(stem_path)
            asr_path = stem_path
            source_kind = "separated_vocals"
        except Exception as exc:  # pragma: no cover - depends on optional runtime/model
            separator = {
                "status": "failed",
                "method": "demucs",
                "model": "htdemucs",
                "input_path": str(input_path),
                "error": f"{type(exc).__name__}: {exc}",
            }
            warnings.append("Demucs separation unavailable; ASR ran on mixed canonical audio.")
    else:
        warnings.append("Separation disabled by request; ASR ran on mixed canonical audio.")

    track = load_audio(input_path)
    acoustic = []
    acoustic_diagnostics = {}
    if source_kind == "separated_vocals":
        stem_track = load_audio(asr_path)
        acoustic, acoustic_diagnostics = acoustic_intervals(stem_track)

    asr_segments_seconds: list[dict] = []
    asr_words_seconds: list[dict] = []
    transcript = ""
    multilingual = multilingual_model_path or find_whisper_snapshot()
    detection = None
    if (language or "auto").strip().lower() == "auto":
        if multilingual is None:
            detection = {"language": None, "error": "modello multilingue faster-whisper-small non in cache"}
        else:
            try:
                detection = detect_language(asr_path, model_path=multilingual, cpu_threads=threads)
            except Exception as exc:  # pragma: no cover - depends on optional runtime/model
                detection = {"language": None, "error": f"{type(exc).__name__}: {exc}"}
    selection = choose_asr_language(language, detection, multilingual_available=multilingual is not None)
    selection["requested"] = language
    selection["detection"] = detection
    if selection["source"] == "fallback_en" and (language or "auto").strip().lower() == "auto":
        warnings.append(f"ASR language fallback to English: {selection['reason']}")
    asr_model = multilingual if selection["multilingual"] else model_path
    try:
        asr_words_seconds, asr_segments_seconds, transcript, backend = transcribe_audio(
            asr_path, model_path=asr_model, device="cpu", compute_type="int8",
            cpu_threads=threads, language=selection["language"],
        )
        backend["status"] = "complete"
        backend["audio_path"] = str(asr_path.resolve())
        backend["language_source"] = selection["source"]
        if detection and detection.get("language"):
            # probabilita' dal rilevamento sul parlato, non quella banale (1.0) della lingua forzata
            backend["detected_language"] = detection["language"]
            backend["language_probability"] = detection["probability"]
    except Exception as exc:  # pragma: no cover - depends on optional runtime/model
        backend = {
            "status": "failed",
            "backend": "faster-whisper",
            "model_path": str(asr_model.resolve()),
            "language": selection["language"],
            "error": f"{type(exc).__name__}: {exc}",
        }
        warnings.append("ASR unavailable; no word boundary is treated as silence.")

    words = seconds_to_samples(asr_words_seconds, track.sample_rate, track.duration_samples)
    segments = seconds_to_samples(asr_segments_seconds, track.sample_rate, track.duration_samples)
    document = build_boundary_document(
        input_path, words=words, asr_segments=segments,
        source_kind=source_kind, backend=backend, separator=separator,
        warnings=warnings, acoustic=acoustic,
        acoustic_diagnostics=acoustic_diagnostics, transcript=transcript,
    )
    document["analysis_status"] = "complete" if backend.get("status") == "complete" else "fallback"
    document["language_selection"] = selection
    output_path = output_dir / f"{label}.json"
    _write_json(output_path, document)
    unknown = sum(1 for item in document["coverage_intervals"] if str(item["status"]).startswith("unknown"))
    print(f"{label}: source={source_kind} lang={selection['language']} ({selection['source']}) words={len(words)} phrases={len(document['phrases'])} unknown_intervals={unknown} output={output_path}")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="canonical 48 kHz WAV files")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "vocals_v2")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--runtime-root", type=Path, default=PROJECT_ROOT / ".cache" / "vocal_runtime")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-separation", action="store_true")
    parser.add_argument("--language", default="auto",
                        help="codice lingua Whisper, oppure 'auto' (rilevamento sullo stem; fallback inglese)")
    parser.add_argument("--multilingual-model-path", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    model_path = (args.model_path or _default_model(PROJECT_ROOT)).resolve(strict=True)
    for index, input_path in enumerate(args.inputs):
        label = chr(ord("a") + index) if len(args.inputs) <= 26 else str(index + 1)
        analyse_one(
            label, input_path, args.output_dir,
            model_path=model_path, runtime_root=args.runtime_root,
            threads=args.threads, skip_separation=args.skip_separation,
            language=args.language, multilingual_model_path=args.multilingual_model_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
