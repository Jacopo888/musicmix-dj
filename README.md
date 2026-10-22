# MusicMix DJ

Research toolkit for **automatic DJ transitions and sets** built from your own music library:
it analyses tracks, chooses where to leave one track and where to enter the next, renders the
transition with DJ techniques, checks the result automatically and keeps every decision traceable.

> **Status: research prototype.** Useful for experiments, not a finished product. No claim of
> state-of-the-art quality: automatic checks catch technical defects (off-beat cuts, clipped words,
> phase/level problems), they do not prove that a transition sounds good. Human listening is still
> needed, periodically and on samples.

*Sezione in italiano più sotto.*

## What it does

| Stage | What happens | Main modules |
|---|---|---|
| Import | your local audio files are hashed, probed and decoded to canonical 48 kHz PCM; nothing is downloaded | `musicmix ingest`, `musicmix.library`, `musicmix.ingest` |
| Analysis | beats/downbeats (Beat This!), sections, vocal stem (Demucs), word timestamps (faster-whisper + Silero VAD), tempo/key (All-In-One, madmom, essentia), phrase model of the vocals | `tools/run_cc_pool_analysis.py`, `musicmix.dj_phrase`, `musicmix.stems` |
| Track cards | one persistent, provenance-hashed card per track with reliability flags | `tools/build_track_cards.py`, `musicmix.track_cards` |
| Selection | pair compatibility (tempo, key, energy), cue on the downbeat at a phrase start, vocal-clash guard, "keep playing A" as an explicit option | `musicmix.dj_select`, `musicmix.dj_policy`, `musicmix.dj_set` |
| Rendering | beatmatching (Rubber Band), EQ / bass swap blends, cuts, echo/filter outs, lead-ins, loudness and true-peak control | `musicmix.dj_render`, `musicmix.dj_effects`, `musicmix.dj_creative`, `musicmix.dj_mashup` |
| Automatic QA | checks on the rendered PCM; a "defect gym" injects known defects into good transitions to measure what the checks catch | `musicmix.dj_qa`, `musicmix.defect_gym`, `tools/defect_gym.py` |
| Critics (diagnostic) | optional pretrained critics (Audiobox Aesthetics, MuQ/SongEval, MERT, CLAP); they do **not** drive selection | `musicmix.dj_critic`, `tools/run_dj_critic.py` |
| Listening | self-contained HTML pages with 20 s clips (10 s per side) and a judgement form | `tools/build_listening_artifact.py`, `tools/listening_pages.py` |

The research notes in `conoscenza_mix/` (Italian) link papers and books to the code that implements
them; they contain summaries and short page-referenced quotations only, never full texts.

## Requirements and installation

- Linux or macOS (Windows works for most tools), Python 3.11, FFmpeg/ffprobe on `PATH`,
  `rubberband-cli` for grid-locked time-stretching (optional: an FFmpeg fallback exists).
- CPU is enough; a full analysis takes several minutes per track.

Minimal install (synthetic tests, import, catalogue, rendering on precomputed analyses):

```bash
git clone https://github.com/Jacopo888/musicmix-dj.git && cd musicmix-dj
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e ".[audio,test]" scipy soundfile
pytest -q -m "not data and not gym"
```

Full analysis stack (pinned versions, three virtual environments because of incompatible numpy
requirements; see `env/README.md`):

```bash
export MUSICMIX_HOME=$HOME/.local/musicmix        # default /opt/musicmix
bash env/install_python.sh                        # core venv + numpy<2 venv, then model weights
python env/fetch_models.py --critics              # optional critics (non-commercial weights!)
```

Every location is configurable with environment variables (`MUSICMIX_HOME`, `MUSICMIX_MODELS`,
`MUSICMIX_DATA`, `MUSICMIX_PY_MAIN`, `MUSICMIX_PY_NP1`, `MUSICMIX_PY_ALLIN1`, `MUSICMIX_SCRATCH`,
`TORCH_HOME`); see `src/musicmix/paths.py`.

## Use your own music library

```bash
musicmix ingest ~/Music/crate --pool runs/library_crate            # import + full analysis
musicmix ingest ~/Music/crate --pool runs/library_crate --no-analyze  # catalogue only
```

The pool folder receives `sources.json` (provenance: original path, size, mtime, SHA-256, decoder),
canonical WAVs and `catalog_initial.json`; the analyses then write `catalog_ready.json` and the
per-track DJ analysis used by the set builder.

**Only use music you own or have the rights to use.** This repository contains no commercial audio and
must not be used to distribute it. Publishing mixes or mashups may need permissions beyond private analysis.

## Licences

Code: [Apache-2.0](LICENSE). Third-party code and model weights keep their own licences, listed in
[THIRD_PARTY.md](THIRD_PARTY.md). **Several model weights are non-commercial** (madmom models CC BY-NC-SA,
MERT / MusicGen / MuQ CC BY-NC, SongEval), essentia is AGPL-3.0 (optional) and Rubber Band is GPL
(used as an external binary). Weights are downloaded from their official sources, never redistributed here.

## Repository layout

`src/musicmix/` library · `tools/` command-line runners and experiments · `tests/` pytest suite
(`-m data` tests need private run data and are skipped without it) · `env/` pinned requirements and setup ·
`configs/`, `schemas/` · `docs/` design documents (`docs/history/` version diary) · `conoscenza_mix/`,
`ricerca/` research notes and source metadata.

Contributions: see [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
Security reports: [SECURITY.md](SECURITY.md).

---

## In italiano

**MusicMix DJ** è un insieme di strumenti di ricerca per creare transizioni e set DJ automatici a partire
dalla propria libreria musicale: analizza i brani (battiti e downbeat, sezioni, voce separata, parole con
timestamp, tempo e tonalità), sceglie quando uscire da A e dove entrare in B (cue sul downbeat a inizio
frase, compatibilità di tempo e tonalità, gestione delle voci, anche l'opzione di continuare A), realizza la
transizione con tecniche DJ (beatmatching, EQ/bass swap, tagli, echo e filtri, loudness) e la controlla con
prove automatiche sul PCM prodotto.

**Stato:** prototipo di ricerca. I controlli automatici trovano difetti tecnici, non dimostrano la qualità
musicale; l'ascolto umano resta necessario, a campione. Nessun risultato "SOTA" dichiarato.

**Installazione:** vedi sopra (Python 3.11, FFmpeg, `pip install -e ".[audio,test]"`; per le analisi complete
`env/install_python.sh`). **Libreria personale:** `musicmix ingest <cartella>` importa i tuoi file senza
scaricare nulla e lancia le analisi.

**Avvertenze:** usa solo musica di cui hai i diritti. Il codice è Apache-2.0, ma i pesi di alcuni modelli
sono **non commerciali** (vedi [THIRD_PARTY.md](THIRD_PARTY.md)). Le note in `conoscenza_mix/` riportano solo
schede e citazioni brevi con pagina; i testi integrali di libri e articoli non sono inclusi.
