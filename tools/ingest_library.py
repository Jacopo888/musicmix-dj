#!/usr/bin/env python3
"""Import your own audio library into a MusicMix pool and run the analyses (same as ``musicmix ingest``).

    python tools/ingest_library.py ~/Music/dj_crate --pool runs/library_crate
    python tools/ingest_library.py ~/Music/dj_crate --no-analyze      # catalog only, prints the next commands

Nothing is downloaded: files are hashed, probed, decoded to canonical 48 kHz WAV and catalogued
(``<pool>/sources.json``, ``<pool>/catalog_initial.json``). Use only music you have the rights to use.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from musicmix.library import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
