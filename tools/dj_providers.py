#!/usr/bin/env python3
"""Run official pretrained providers for DJ selection (isolated runtime).

Runs in the All-In-One venv (``$MUSICMIX_PY_ALLIN1``) (torch 2.2.2+cpu, numpy 1.26.4, NATTEN
0.14.6, madmom @27f032e, allin1 @18e7890, essentia 2.1b6.dev1110; see the
commented block at the end of ``env/requirements-core.txt``).  For each
track of a MusicMix catalog it writes ``<out>/<slug>.json`` with:

* ``allin1``: All-In-One (Kim & Nam, WASPAA 2023) beats, downbeats, BPM and
  functional segments, ``harmonix-all`` 8-fold ensemble from the official
  Hugging Face repository ``taejunkim/allinone`` (sha256 of every checkpoint);
* ``madmom_key``: CNN key classifier of Korzeniowski & Widmer (ISMIR 2018),
  madmom's bundled ``key_cnn.pkl``: 24-class probabilities for the whole
  track and for 30 s windows (hop 15 s);
* ``madmom_downbeats``: RNNDownBeatProcessor + DBN (beats_per_bar 3/4);
* ``essentia_key``: essentia ``KeyExtractor`` with profiles edma, bgate,
  temperley (key, scale, strength).

Nothing here ranks or selects; ``tools/dj_plan_pool.py`` consumes the JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musicmix.paths import localize, torch_home  # noqa: E402

ALLIN1_REPO = "https://github.com/mir-aidj/all-in-one"
ALLIN1_COMMIT = "18e78903c0365147a2c5d4e5e57ebf88cb7d800e"
MADMOM_REPO = "https://github.com/CPJKU/madmom"
MADMOM_COMMIT = "27f032e8947204902c675e5e341a3faf5dc86dae"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_allin1(slug: str, wav: Path, work: Path, model: str) -> dict:
    import allin1
    from allin1.models.loaders import NAME_TO_FILE
    from huggingface_hub import hf_hub_download

    link_dir = work / "inputs"
    link_dir.mkdir(parents=True, exist_ok=True)
    link = link_dir / f"{slug}.wav"   # allin1 names its demix folder after the file stem
    if not link.exists():
        link.symlink_to(wav)
    shared = work / "demix" / "htdemucs_provenance" / f"{slug}.json"
    stems = work / "demix" / "htdemucs" / slug
    demucs_note = "htdemucs (allin1 internal demix)"
    if shared.is_file() and all((stems / f"{n}.wav").is_file() for n in ("bass", "drums", "other", "vocals")):
        prov = json.loads(shared.read_text(encoding="utf-8"))
        if prov.get("input_sha256") == sha256(wav):
            demucs_note = {"note": "htdemucs stems shared with the vocals stage (same checkpoint, shifts=0); "
                                   "allin1 skipped its own demix (CLI default shifts=1, one random shift)", **prov}
    t0 = time.time()
    res = allin1.analyze(link, model=model, device="cpu", demix_dir=work / "demix",
                         spec_dir=work / "spec", keep_byproducts=True, multiprocess=False)
    ckpts = {}
    names = [model] if model in NAME_TO_FILE else [f"harmonix-fold{i}" for i in range(8)]
    for name in names:
        p = Path(hf_hub_download(repo_id="taejunkim/allinone", filename=NAME_TO_FILE[name]))
        ckpts[NAME_TO_FILE[name]] = {"sha256": sha256(p), "hf_snapshot": p.parent.name}
    return {
        "status": "complete", "runtime_s": round(time.time() - t0, 1),
        "provider": "All-In-One Music Structure Analyzer", "repository": ALLIN1_REPO,
        "source_revision": ALLIN1_COMMIT, "paper": "https://arxiv.org/abs/2307.16425", "license": "MIT",
        "model": model, "hf_repo": "taejunkim/allinone", "checkpoints": ckpts,
        "demucs": demucs_note,
        "bpm": res.bpm, "beats": [float(x) for x in res.beats],
        "downbeats": [float(x) for x in res.downbeats],
        "beat_positions": [int(x) for x in res.beat_positions],
        "segments": [{"start": float(s.start), "end": float(s.end), "label": s.label} for s in res.segments],
        "note": "segment labels are model outputs (Harmonix taxonomy); no calibrated confidence",
    }


def run_madmom(wav: Path) -> tuple[dict, dict]:
    import numpy as np
    from madmom.audio.signal import Signal
    from madmom.features.key import CNNKeyRecognitionProcessor, KEY_LABELS
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
    from madmom.models import KEY_CNN

    sig = Signal(str(wav), sample_rate=44100, num_channels=1)
    proc = CNNKeyRecognitionProcessor()
    t0 = time.time()
    probs = proc(sig)[0]
    win, hop, sr = 30.0, 15.0, 44100
    windows = []
    start = 0.0
    while start + win <= len(sig) / sr + 1e-6:
        seg = Signal(np.asarray(sig[int(start * sr):int((start + win) * sr)]), sample_rate=sr)
        p = proc(seg)[0]
        windows.append({"start_s": start, "key": KEY_LABELS[int(p.argmax())], "prob": float(p.max())})
        start += hop
    key = {
        "status": "complete", "provider": "madmom CNNKeyRecognitionProcessor (Korzeniowski & Widmer, ISMIR 2018)",
        "repository": MADMOM_REPO, "source_revision": MADMOM_COMMIT,
        "model_files": {Path(f).name: sha256(Path(f)) for f in KEY_CNN},
        "labels": list(KEY_LABELS), "probs": [float(x) for x in probs],
        "key": KEY_LABELS[int(probs.argmax())], "prob": float(probs.max()),
        "second": KEY_LABELS[int(np.argsort(probs)[-2])], "prob2": float(np.sort(probs)[-2]),
        "windows": windows, "window_s": win, "hop_s": hop, "runtime_s": round(time.time() - t0, 1),
    }
    t0 = time.time()
    act = RNNDownBeatProcessor()(sig)
    beats = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)(act)
    down = {
        "status": "complete", "provider": "madmom RNNDownBeatProcessor + DBNDownBeatTrackingProcessor",
        "repository": MADMOM_REPO, "source_revision": MADMOM_COMMIT, "beats_per_bar": [3, 4],
        "beats": [float(t) for t, _ in beats], "beat_positions": [int(p) for _, p in beats],
        "downbeats": [float(t) for t, p in beats if int(p) == 1], "runtime_s": round(time.time() - t0, 1),
    }
    return key, down


def run_essentia(wav: Path) -> dict:
    import essentia
    import essentia.standard as es

    audio = es.MonoLoader(filename=str(wav), sampleRate=44100)()
    out = {"status": "complete", "provider": "essentia KeyExtractor", "version": essentia.__version__,
           "profiles": {}}
    for prof in ("edma", "bgate", "temperley"):
        k, s, strength = es.KeyExtractor(profileType=prof, sampleRate=44100)(audio)
        out["profiles"][prof] = {"key": f"{k} {s}", "strength": float(strength)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalog", default=str(ROOT / "runs/pool_v12/catalog_ready.json"))
    ap.add_argument("--out", default=str(ROOT / "runs/dj_plan_v1/providers"))
    ap.add_argument("--work", default=str(ROOT / ".cache/dj_allin1"))
    ap.add_argument("--model", default="harmonix-all")
    ap.add_argument("--slugs", nargs="*")
    ap.add_argument("--only", choices=["allin1", "madmom", "essentia"], nargs="*")
    ap.add_argument("--threads", type=int, default=None, help="thread torch (default: tutti i core)")
    ap.add_argument("--skip-complete", action="store_true",
                    help="salta i provider gia' completi nel JSON della traccia (ripresa dopo interruzione)")
    args = ap.parse_args()
    os.environ.setdefault("TORCH_HOME", str(torch_home()))
    import torch
    torch.set_num_threads(args.threads or os.cpu_count() or 1)
    cat = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = set(args.only or ["allin1", "madmom", "essentia"])
    for t in cat["tracks"]:
        if args.slugs and t["slug"] not in args.slugs:
            continue
        wav = localize(t["canonical_path"])
        path = out_dir / f"{t['slug']}.json"
        rec = json.loads(path.read_text()) if path.exists() else {}
        rec.update({"schema_version": "musicmix.dj_providers.1", "slug": t["slug"], "track_id": t["track_id"],
                    "audio_sha256": sha256(wav)})
        assert rec["audio_sha256"] == t["track_id"], f"audio identity mismatch for {t['slug']}"
        for name, fn in (("madmom", None), ("essentia", None), ("allin1", None)):
            if name not in only:
                continue
            if args.skip_complete and rec.get("madmom_key" if name == "madmom" else name, {}).get("status") == "complete":
                continue
            print(f"[{t['slug']}] {name} ...", flush=True)
            try:
                if name == "allin1":
                    rec["allin1"] = run_allin1(t["slug"], wav, Path(args.work), args.model)
                elif name == "madmom":
                    rec["madmom_key"], rec["madmom_downbeats"] = run_madmom(wav)
                else:
                    rec["essentia_key"] = run_essentia(wav)
            except Exception as exc:  # recorded, never silently replaced
                rec[name if name != "madmom" else "madmom_key"] = {"status": "failed", "error": repr(exc)}
                print(f"  failed: {exc!r}", flush=True)
            path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
