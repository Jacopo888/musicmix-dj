#!/usr/bin/env python3
"""ASR multilingue (faster-whisper small, word timestamps) sugli stem vocali Demucs per dj_phrase.

Il pool CC e' in buona parte non inglese (francese, tedesco, spagnolo, creolo...): le parole in
``runs/pool_cc_v1/vocals/*/a.json`` vengono da ``small.en`` (lingua forzata ``en``) e sono spesso
allucinazioni inglesi. Qui si ritrascrive lo stesso stem con il modello multilingue ufficiale
``Systran/faster-whisper-small`` e rilevamento automatico della lingua, con gli stessi parametri
di ``musicmix.vocal_boundaries.transcribe_audio`` (beam 5, temperatura 0, niente VAD, niente
condizionamento sul testo precedente). I file del pool non sono toccati: l'uscita va in
``runs/dj_phrase_v1/asr/<slug>.json`` (campioni canonici 48 kHz, stesso orologio dello stem).

    python tools/dj_phrase_asr.py --slugs a b c --threads 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from musicmix.stems import StemError, vocal_stem_path  # noqa: E402
from musicmix.vocal_boundaries import transcribe_audio  # noqa: E402
from musicmix.paths import models_dir  # noqa: E402

MODEL = (models_dir() / ".cache/models/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120")
SR = 48000


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def detect_language(stem: Path, model_path: Path, threads: int) -> dict:
    """Language from up to 6 x 30 s of *speech* (faster-whisper detect_language with its Silero VAD).

    The default detection of ``transcribe`` looks only at the first 30 s, which in a rap track is
    often an instrumental intro (first pass: French verses detected as English with p = 0.46).
    """
    from faster_whisper import WhisperModel, decode_audio

    model = WhisperModel(str(model_path), device="cpu", compute_type="int8", cpu_threads=threads, num_workers=1)
    audio = decode_audio(str(stem), sampling_rate=16000)
    lang, prob, all_probs = model.detect_language(audio, vad_filter=True, language_detection_segments=6,
                                                  language_detection_threshold=0.9)
    top = sorted(all_probs, key=lambda x: -x[1])[:5] if all_probs else []
    return {"language": lang, "probability": float(prob), "top5": [(k, float(v)) for k, v in top],
            "method": "faster-whisper detect_language, vad_filter=True, language_detection_segments=6, threshold=0.9"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs", nargs="+", required=True)
    ap.add_argument("--pool", default=str(ROOT / "runs/pool_cc_v1"))
    ap.add_argument("--out", default=str(ROOT / "runs/dj_phrase_v1/asr"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--model", default=str(MODEL))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cat_path = Path(args.pool) / "catalog_initial.json"
    entries = {t["slug"]: t for t in json.loads(cat_path.read_text(encoding="utf-8"))["tracks"]} if cat_path.is_file() else {}
    for slug in args.slugs:
        dst = out / f"{slug}.json"
        if dst.is_file():
            print(f"{slug}: gia' presente", flush=True)
            continue
        entry = entries.get(slug)
        try:
            if entry is None:
                raise StemError(f"{slug}: assente dal catalogo {cat_path}")
            stem = vocal_stem_path(entry, check_sha=False)  # percorso e identità registrati dalla fase vocals
        except StemError as exc:
            print(f"{slug}: stem non utilizzabile ({exc})", flush=True)
            continue
        t0 = time.time()
        lang_info = detect_language(stem, Path(args.model), args.threads)
        words, segments, transcript, meta = transcribe_audio(stem, model_path=Path(args.model), cpu_threads=args.threads,
                                                             language=lang_info["language"])
        meta["detected_language"] = lang_info["language"]
        meta["language_probability"] = lang_info["probability"]
        meta["language_detection"] = lang_info
        conv = lambda it: dict(start_sample=int(round(it["start_seconds"] * SR)), end_sample=int(round(it["end_seconds"] * SR)),  # noqa: E731
                               text=it["text"], confidence=it["confidence"])
        doc = {"schema_version": "musicmix.dj_phrase_asr.1", "slug": slug, "stem": str(stem.relative_to(ROOT)),
               "stem_sha256": sha256(stem), "sample_rate": SR, "words": [conv(w) for w in words],
               "asr_segments": [conv(s) for s in segments], "transcript": transcript,
               "backend": dict(meta, model_repo="Systran/faster-whisper-small",
                               model_revision=Path(args.model).name, language_requested=None),
               "elapsed_s": time.time() - t0}
        dst.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
        print(f"{slug}: {meta.get('detected_language')} p={meta.get('language_probability'):.2f} {len(words)} parole "
              f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
