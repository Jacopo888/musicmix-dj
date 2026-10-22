"""``musicmix ingest``: local library import on synthetic audio (no network, no models)."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import wave

import numpy as np
import pytest

from musicmix import library as L

ROOT = Path(__file__).resolve().parents[1]
needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg missing")


def write_wav(path: Path, seconds: float, freq: float = 220.0, sr: int = 44100) -> Path:
    t = np.arange(int(seconds * sr)) / sr
    tone = 0.2 * np.sin(2 * np.pi * freq * t)
    tone[(np.arange(t.size) % (sr // 2)) < 200] += 0.4  # clicks at 120 BPM
    pcm = (np.clip(np.stack([tone, tone], 1), -1, 1) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def encode(src: Path, dst: Path, **tags: str) -> Path:
    meta = [x for k, v in tags.items() for x in ("-metadata", f"{k}={v}")]
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), *meta, str(dst)], check=True)
    return dst


@pytest.fixture
def library(tmp_path):
    lib = tmp_path / "my music"
    a = write_wav(lib / "01 - Test Artist - First Song.wav", 40.0, 220.0)
    write_wav(lib / "sub" / "loop.wav", 35.0, 330.0)
    write_wav(lib / "too_short.wav", 5.0, 440.0)
    (lib / "notes.txt").write_text("not audio")
    (lib / "broken.mp3").write_bytes(b"this is not an mp3 file at all" * 10)
    (lib / ".hidden").mkdir()
    write_wav(lib / ".hidden" / "skip.wav", 40.0)
    return lib, a


def test_scan_filters_extensions_and_hidden(library):
    lib, _ = library
    names = [p.relative_to(lib).as_posix() for p in L.scan(lib)]
    assert names == ["01 - Test Artist - First Song.wav", "broken.mp3", "sub/loop.wav", "too_short.wav"]
    assert [p.name for p in L.scan(lib, recursive=False)] == ["01 - Test Artist - First Song.wav", "broken.mp3",
                                                              "too_short.wav"]


def test_guess_artist_title_and_slug():
    assert L.guess_artist_title(Path("03 Some One - A Title (Mix).flac"), {}) == ("Some One", "A Title (Mix)")
    assert L.guess_artist_title(Path("x.mp3"), {"artist": "Tag Artist", "title": "Tag Title"}) == ("Tag Artist", "Tag Title")
    assert L.guess_artist_title(Path("lonely_name.wav"), {}) == ("", "lonely name")
    assert L.slugify("Beyoncé & Co. — Ça va!") == "beyonce_co_ca_va"
    assert L.slugify("!!!", fallback="x") == "x"


@needs_ffmpeg
def test_ingest_catalog_provenance_and_idempotence(library, tmp_path):
    lib, first = library
    tagged = encode(write_wav(tmp_path / "raw.wav", 32.0, 550.0), lib / "tagged.flac",
                    artist="Tag Artist", title="Tag Title", album="Demo")
    pool = tmp_path / "pool"
    summary = L.ingest_library(lib, pool, log=lambda *_: None)
    added = {r["slug"]: r for r in summary["added"]}
    assert set(added) == {"test_artist_first_song", "loop", "tag_artist_tag_title"}
    assert {Path(r["file"]).name for r in summary["rejected"]} == {"too_short.wav", "broken.mp3"}

    cat = json.loads((pool / "catalog_initial.json").read_text())
    assert cat["schema_version"] == "musicmix.pool-inventory.1" and cat["requested_count"] == 3
    assert cat["status"] == "incomplete" and cat["directed_pair_count_when_complete"] == 6
    src = pool / "sources.json"
    assert cat["config_path"] == str(src.resolve()) and cat["config_sha256"] == L.sha256_file(src)
    entry = next(t for t in cat["tracks"] if t["slug"] == "tag_artist_tag_title")
    assert entry["artist"] == "Tag Artist" and entry["album"] == "Demo" and entry["origin"] == "local_library"
    assert entry["sample_rate"] == 48000 and abs(entry["duration_seconds"] - 32.0) < 0.05
    assert entry["track_id"] == L.sha256_file(Path(entry["canonical_path"]))
    assert entry["source_sha256"] == L.sha256_file(tagged) and entry["source_path"] == str(tagged.resolve())
    assert entry["missing"] == ["structure", "vocals"]
    assert entry["structure_path"] == str((pool / "structure" / entry["slug"] / "a.json").resolve())
    prov = entry["provenance"]
    assert prov["downloaded"] is False and prov["kind"] == "local_file" and prov["probe"]["codec"] == "flac"
    assert prov["size_bytes"] == tagged.stat().st_size
    with wave.open(entry["canonical_path"]) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (48000, 2, 2)
    assert Path(entry["canonical_path"]).with_suffix(".ingest.json").is_file()

    # same bytes elsewhere, re-run: nothing new, nothing rewritten
    shutil.copy2(first, lib / "copy of first.wav")
    again = L.ingest_library(lib, pool, log=lambda *_: None)
    assert again["added"] == [] and {d["slug"] for d in again["duplicates"]} >= {"test_artist_first_song"}
    assert json.loads((pool / "catalog_initial.json").read_text())["requested_count"] == 3

    # a new file with an already-used title gets a distinct slug
    write_wav(lib / "new" / "loop.wav", 36.0, 660.0)
    third = L.ingest_library(lib, pool, log=lambda *_: None)
    assert [r["slug"] for r in third["added"]] == ["loop_2"]


@needs_ffmpeg
def test_ingested_pool_is_accepted_by_refresh(library, tmp_path):
    """prepare_cc_pool.refresh (the existing next step) marks tracks ready once structure/vocals exist."""
    lib, _ = library
    pool = tmp_path / "pool"
    L.ingest_library(lib, pool, log=lambda *_: None)
    cat = json.loads((pool / "catalog_initial.json").read_text())
    for t in cat["tracks"]:
        for ev in ("structure", "vocals"):
            p = Path(t[f"{ev}_path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"track_id": t["track_id"], "sample_rate": t["sample_rate"],
                                     "duration_samples": t["duration_samples"], "status": "complete"}))
    sys.path.insert(0, str(ROOT / "tools"))
    import prepare_cc_pool

    prepare_cc_pool.refresh(pool)
    ready = json.loads((pool / "catalog_ready.json").read_text())
    assert ready["status"] == "ready" and ready["ready_count"] == 2


@needs_ffmpeg
def test_dry_run_and_copy_sources(library, tmp_path):
    lib, _ = library
    pool = tmp_path / "pool"
    dry = L.ingest_library(lib, pool, dry_run=True, log=lambda *_: None)
    assert len(dry["added"]) == 2 and not pool.exists()
    L.ingest_library(lib, pool, copy_sources=True, limit=1, log=lambda *_: None)
    entry = json.loads((pool / "catalog_initial.json").read_text())["tracks"][0]
    assert Path(entry["source_path"]).parent == (pool / "sources").resolve()
    assert entry["provenance"]["copied_into_pool"] is True


def test_analysis_commands_use_existing_tools(tmp_path):
    pool = tmp_path / "pool"
    cmds = L.analysis_commands(pool, python="py", root=ROOT, slugs=["a", "b"], track_cards=tmp_path / "cards")
    first, cards = cmds
    assert first[:2] == ["py", str(ROOT / "tools/run_cc_pool_analysis.py")]
    assert first[first.index("--pool") + 1] == str(pool.resolve())
    assert first[-3:] == ["--slugs", "a", "b"] and "--share-stems" in first and "--logs" in first
    assert cards[1].endswith("build_track_cards.py") and "--analysis-dir" in cards
    for cmd in cmds:  # every referenced tool exists and accepts the flags
        help_out = subprocess.run([sys.executable, cmd[1], *(["build"] if "build" in cmd else []), "--help"],
                                  capture_output=True, text=True, cwd=ROOT)
        assert help_out.returncode == 0, help_out.stderr
        for flag in (x for x in cmd[2:] if x.startswith("--")):
            assert flag in help_out.stdout, flag


@needs_ffmpeg
def test_cli_no_analyze(library, tmp_path):
    lib, _ = library
    pool = tmp_path / "pool"
    out = subprocess.run([sys.executable, str(ROOT / "tools/ingest_library.py"), str(lib), "--pool", str(pool),
                          "--no-analyze"], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert "next: " in out.stdout and "run_cc_pool_analysis.py" in out.stdout
    assert json.loads((pool / "catalog_initial.json").read_text())["requested_count"] == 2
    rc = subprocess.run([sys.executable, "-m", "musicmix", "ingest", str(tmp_path / "missing")],
                        capture_output=True, text=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert rc.returncode == 2 and "not a folder" in rc.stderr
