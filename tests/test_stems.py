"""Risolutore dello stem vocale (musicmix.stems): percorso registrato, identità verificata."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

from musicmix.stems import StemError, vocal_stem_path, vocal_stem_record

TID = "ab" * 32


def _setup(tmp_path, *, n=4800, stem_name="voce_x.wav", stem_frames=None, input_sha=TID, kind="separated_vocals"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    stem = tmp_path / "stems" / stem_name
    stem.parent.mkdir()
    sf.write(str(stem), np.zeros((stem_frames or n, 2), dtype="float32"), 48000, subtype="PCM_16")
    vj = tmp_path / "vocals" / "a.json"
    vj.parent.mkdir()
    (vj.parent / "a_vocals.wav").write_bytes(b"not the stem")  # il nome indovinato non deve essere usato
    doc = {"track_id": TID, "sample_rate": 48000, "duration_samples": n, "source_kind": kind,
           "separator": {"status": "complete", "model": "htdemucs", "input_sha256": input_sha, "output_path": str(stem),
                         "checkpoint_files": [{"sha256": "c" * 64}]},
           "backend": {"audio_path": str(stem)}, "words": []}
    vj.write_text(json.dumps(doc))
    entry = {"slug": "x", "track_id": TID, "sample_rate": 48000, "duration_samples": n, "vocals_path": str(vj)}
    return entry, stem, vj


def test_resolves_recorded_path_not_guessed_name(tmp_path):
    entry, stem, vj = _setup(tmp_path)
    assert vocal_stem_path(entry) == stem
    assert vocal_stem_path(vj) == stem
    r = vocal_stem_record(entry)
    assert r["recorded_as"] == "separator.output_path" and r["checks"]["frames"] == 4800 and r["model"] == "htdemucs"


def test_mismatches_raise_clear_errors(tmp_path):
    entry, stem, vj = _setup(tmp_path, stem_frames=4000)
    with pytest.raises(StemError, match="campioni"):
        vocal_stem_path(entry)
    entry, stem, vj = _setup(tmp_path / "b", input_sha="cd" * 32)
    with pytest.raises(StemError, match="audio diverso"):
        vocal_stem_path(entry)
    entry, stem, vj = _setup(tmp_path / "c")
    with pytest.raises(StemError, match="altro audio"):
        vocal_stem_path(dict(entry, track_id="ef" * 32))
    stem.unlink()
    with pytest.raises(StemError, match="assente"):
        vocal_stem_path(entry)
    entry, stem, vj = _setup(tmp_path / "d", kind="mixed_audio_fallback")
    with pytest.raises(StemError, match="non ha separato"):
        vocal_stem_path(entry)


def test_reused_stem_sha_is_checked(tmp_path):
    entry, stem, vj = _setup(tmp_path)
    doc = json.loads(vj.read_text())
    doc["separator"]["reused_stem_path"] = str(stem)
    doc["separator"]["reused_stem_sha256"] = hashlib.sha256(stem.read_bytes()).hexdigest()
    vj.write_text(json.dumps(doc))
    assert vocal_stem_record(entry)["checks"]["stem_sha256_checked"]
    doc["separator"]["reused_stem_sha256"] = "0" * 64
    vj.write_text(json.dumps(doc))
    with pytest.raises(StemError, match="SHA-256"):
        vocal_stem_path(entry)
    assert vocal_stem_path(entry, check_sha=False) == stem


def test_additive_override_replaces_unusable_frozen_vocals(tmp_path, monkeypatch):
    good, stem, vj = _setup(tmp_path / "redo")
    frozen = dict(good, vocals_path=str(tmp_path / "congelato" / "a.json"))  # JSON congelato assente (percorso altrui)
    ov = tmp_path / "ov.json"
    monkeypatch.setenv("MUSICMIX_VOCALS_OVERRIDES", str(ov))
    with pytest.raises(StemError, match="JSON vocals assente"):
        vocal_stem_path(frozen)
    sha = hashlib.sha256(vj.read_bytes()).hexdigest()
    ov.write_text(json.dumps({"overrides": {"x": {"vocals_json": str(vj), "vocals_json_sha256": sha}}}))
    assert vocal_stem_path(frozen) == stem
    with pytest.raises(StemError, match="altro audio"):  # l'override non vale per un altro track_id
        vocal_stem_path(dict(frozen, track_id="ef" * 32))
    ov.write_text(json.dumps({"overrides": {"x": {"vocals_json": str(vj), "vocals_json_sha256": "0" * 64}}}))
    with pytest.raises(StemError, match="SHA-256 del JSON vocals di override"):
        vocal_stem_path(frozen)
