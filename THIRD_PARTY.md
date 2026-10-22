# Third-party code, models and data

MusicMix DJ is licensed under Apache-2.0. It **uses** (imports, calls or downloads at run time) the projects
below, which keep their own licences. Only TIV.lib is vendored in this repository. **No model weights, audio,
datasets or book/paper full texts are redistributed here**: `env/fetch_models.py` downloads pinned weights from
the official sources, and you must accept their terms.

Licence fields marked "verified" were read from the installed package metadata or the Hugging Face model card
on 22/10/2026; the others are as stated by the upstream project at the time of writing. Always check upstream
before any commercial use.

## Vendored

| Component | Where | Licence |
|---|---|---|
| TIV.lib (Tonal Interval Vectors, A. Ramires) | `src/musicmix/_vendor/tivlib/` | MIT (copy in `_vendor/tivlib/LICENSE`, provenance in `PROVENANCE.json`) |

## Analysis models and libraries

| Component | Used for | Code licence | Weights licence | Notes |
|---|---|---|---|---|
| Demucs (htdemucs), Meta | vocal / 4-stem separation | MIT (verified, demucs 4.0.1) | MIT (upstream) | |
| faster-whisper + CTranslate2, SYSTRAN | ASR word timestamps, language detection | MIT (verified) | Whisper conversions `Systran/faster-whisper-*`: MIT (verified) | original Whisper weights by OpenAI, MIT |
| Silero VAD | speech activity | MIT (verified, silero-vad 6.2.3) | MIT (bundled) | |
| spaCy + `en_core_web_sm` | syntactic cues in lyrics | MIT (verified) | MIT (verified) | |
| Beat This! (CPJKU) | beats / downbeats | MIT (verified, beat-this 1.1.0) | released with the code (check upstream) | |
| All-In-One `allin1` (mir-aidj) | tempo, beats, sections | MIT (verified, allin1 1.1.0) | released with the code (check upstream) | depends on madmom and NATTEN |
| madmom (CPJKU) | key (CNN), beat tracking | BSD (verified) | **CC BY-NC-SA 4.0** (verified: "free for non-commercial use") | non-commercial models |
| essentia (MTG) | key estimation | **AGPL-3.0** (verified) | – | optional dependency; installing it in your environment brings AGPL obligations |
| CUE-DETR (ETH DISCO) | DJ cue-point detection | see upstream repository | MIT (verified, `disco-eth/cue-detr`) | runs in the numpy<2 venv |
| LinkSeg, SSM-Net | structure boundaries | see upstream repositories | see upstream | optional research providers |

## Critics (diagnostic only, never used to select transitions)

| Component | Code licence | Weights licence |
|---|---|---|
| Audiobox Aesthetics, Meta | CC BY 4.0 (verified, package 0.0.4) | CC BY 4.0 (verified) |
| MuQ (Tencent AI Lab) | MIT (verified, muq 0.1.0) | **CC BY-NC 4.0** (verified, `OpenMuQ/MuQ-large-msd-iter`) |
| SongEval (ASLP lab) | see upstream | **non-commercial** (upstream; uses MuQ) |
| MERT-v1-95M (m-a-p) | – | **CC BY-NC 4.0** (verified) |
| MusicGen (Meta, audiocraft) | MIT (upstream) | **CC BY-NC 4.0** (verified, `facebook/musicgen-small`) |
| CLAP (LAION) | CC0-1.0 (upstream `laion_clap`) | Apache-2.0 (verified, `laion/clap-htsat-unfused`, `laion/larger_clap_music`) |
| COCOLA | **no explicit licence** found | research use only; evaluated, not adopted |

## External programs

| Program | Licence | How it is used |
|---|---|---|
| FFmpeg / ffprobe | LGPL-2.1+ or GPL depending on the build | external binary for decoding and probing |
| Rubber Band (`rubberband-cli`) | **GPL-2.0-or-later** (commercial licence available from the authors) | external binary called by `musicmix.dj_render`; not linked |

## Python dependencies

PyTorch (BSD-3 / Apache-2.0 components), NumPy, SciPy (BSD), librosa (ISC), soundfile (BSD), transformers
(Apache-2.0) and the other packages pinned in `env/requirements-*.txt` keep their own licences.

## Research sources

Papers and books are cited in `conoscenza_mix/` and `ricerca/` with DOI/URL and licence metadata
(`SOURCES.json`). Only our own summaries and short, page-referenced quotations are included.
