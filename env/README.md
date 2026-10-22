# Environment setup

MusicMix uses up to three Python 3.11 virtual environments because some official model packages pin
incompatible versions of numpy/transformers. Paths are configurable (see `src/musicmix/paths.py`).

| venv (default under `$MUSICMIX_HOME`, default `/opt/musicmix`) | variable | requirements | used for |
|---|---|---|---|
| `venv` | `MUSICMIX_PY_MAIN` | `requirements-core.txt` (+ `requirements-critics.txt`) | pipeline, Beat This!, Demucs, faster-whisper, Silero, spaCy, Audiobox, tests |
| `venv-np1` | `MUSICMIX_PY_NP1` | `requirements-legacy-np1.txt` | CUE-DETR, SongEval/MuQ, MusicGen (numpy<2) |
| `venv-allin1` | `MUSICMIX_PY_ALLIN1` | see `tools/dj_providers.py` | All-In-One, madmom, essentia (torch 2.2, NATTEN 0.14) |

```bash
export MUSICMIX_HOME=$HOME/.local/musicmix     # anywhere you like
bash env/install_python.sh                     # uv-managed venvs + pinned model weights (env/fetch_models.py)
python env/fetch_models.py --target "$MUSICMIX_HOME/models" --critics   # optional critics
```

`setup_ubuntu.sh` is a reference script for a fresh Ubuntu 24.04 machine run as root (apt packages:
ffmpeg, libsndfile1, rubberband-cli; uv; then `install_python.sh`). `RUNTIME_MAP.md` records how the
original runtimes were mapped to Linux and the known numerical risks (e.g. FFmpeg version and canonical hashes).

Model weights are downloaded from their official sources at pinned revisions and verified by SHA-256; several
are **non-commercial** (see `THIRD_PARTY.md`). For CI and quick tests `requirements-ci.txt` is enough.
