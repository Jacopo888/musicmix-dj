"""Listening artifact builder: exclusion on bad hash, HTML markers, doc ids."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import wave

import pytest

from tools.build_listening_artifact import build, doc_id, sanitize_segment

RATE = 8000


def _write_wav(path: Path, seconds: float, freq: int) -> str:
    n = int(seconds * RATE)
    frames = b"".join(struct.pack("<h", int(8000 * ((i * freq // RATE) % 2 * 2 - 1))) for i in range(n))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(frames)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def gallery(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    run_dir = repo / "runs" / "synthetic_20s"
    (run_dir / "audio").mkdir(parents=True)
    monkeypatch.setenv("MUSICMIX_ROOT", str(repo))
    good = _write_wav(run_dir / "audio" / "good.wav", 2.0, 440)
    _write_wav(run_dir / "audio" / "bad.wav", 2.0, 220)
    manifest = {
        "schema_version": "musicmix.twenty-second-export.1",
        "artifacts": [
            {   # recorded Windows path, resolved through musicmix.paths.localize
                "id": "clip good/1",
                "path": r"C:\Data\x\Mix in C\music_mixing_progetto\runs\synthetic_20s\audio\good.wav",
                "sha256": good, "sample_rate": RATE, "duration_samples": 2 * RATE, "join_sample": RATE,
            },
            {
                "id": "clip_bad",
                "path": str(run_dir / "audio" / "bad.wav"),
                "sha256": "0" * 64, "sample_rate": RATE, "duration_samples": 2 * RATE, "join_sample": RATE,
            },
        ],
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf8")
    return path


def test_build_excludes_bad_hash_and_marks_join(gallery, tmp_path):
    before = gallery.read_bytes()
    out = tmp_path / "out"
    listing = build(gallery, out, encoder="wav16", log=lambda *_: None)

    assert gallery.read_bytes() == before  # manifest untouched
    assert listing["run"] == "synthetic_20s"
    assert [c["id"] for c in listing["included"]] == ["clip good/1"]
    assert listing["excluded"] == [{
        "id": "clip_bad", "reason": "sha256_diverso",
        "detail": listing["excluded"][0]["detail"], "path_recorded": str(gallery.parent / "audio" / "bad.wav"),
    }]
    assert "atteso " + "0" * 64 in listing["excluded"][0]["detail"]

    clip = listing["included"][0]
    assert clip["doc_id"] == "synthetic_20s__clip_good_1"
    audio = out / clip["audio"]
    assert audio.is_file() and hashlib.sha256(audio.read_bytes()).hexdigest() == clip["audio_sha256"]

    saved = json.loads((out / "listening_manifest.json").read_text(encoding="utf8"))
    assert saved["source_manifest"]["sha256"] == hashlib.sha256(before).hexdigest()

    page = (out / "index.html").read_text(encoding="utf8")
    assert "clip good/1" in page and "synthetic_20s__clip_good_1" in page
    assert "clip_bad" not in page
    assert 'class="join-marker" data-join-s="1.000000" style="left:50.0000%"' in page
    assert 'preload="metadata"' in page and "<title>Ascolti MusicMix</title>" in page
    assert 'window.claude.use("db")' in page and "onSnapshot" in page


def test_run_and_limit_options(gallery, tmp_path):
    listing = build(gallery, tmp_path / "o", run="my run!", limit=1, encoder="wav16", log=lambda *_: None)
    assert listing["included"][0]["doc_id"] == "my_run___clip_good_1"
    assert listing["excluded"] == []


def test_existing_output_is_preserved(gallery, tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        build(gallery, out, encoder="wav16", log=lambda *_: None)


@pytest.mark.parametrize("raw,expected", [
    ("v11_audio_001_main_20s", "v11_audio_001_main_20s"),
    ("a b/c\\d", "a_b_c_d"),
    ("x~y:z@w+v.-", "x~y:z@w+v.-"),
    ("..", "__"),
    ("", "_"),
    ("città", "citt_"),
])
def test_sanitize_segment(raw, expected):
    assert sanitize_segment(raw) == expected


def test_doc_id_joins_sanitized_segments():
    assert doc_id("psychology_v11_20s", "v11 audio/001") == "psychology_v11_20s__v11_audio_001"


def test_auto_encoder_produces_mp3_when_ffmpeg_has_lame(gallery, tmp_path):
    from tools.build_listening_artifact import choose_encoder
    if choose_encoder("auto") != "ffmpeg-mp3":
        pytest.skip("ffmpeg con libmp3lame non disponibile")
    listing = build(gallery, tmp_path / "mp3", log=lambda *_: None)
    clip = listing["included"][0]
    assert clip["audio"].endswith(".mp3") and listing["encoder"]["bitrate"] == "192k"
    assert (tmp_path / "mp3" / clip["audio"]).stat().st_size > 0


# ------------------------------------------------- musicmix.psychology-review.1

def _write_wav48(path: Path, seconds: float, level: int) -> str:
    n = int(seconds * 48000)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(struct.pack("<hh", level, -level) * n)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def review(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    run_dir = repo / "runs" / "synthetic_v12"
    (run_dir / "audio").mkdir(parents=True)
    monkeypatch.setenv("MUSICMIX_ROOT", str(repo))
    main = _write_wav48(run_dir / "audio" / "audio_001_main.wav", 20.0, 1000)
    _write_wav48(run_dir / "audio" / "audio_001_natural_A.wav", 20.0, 2000)
    _write_wav48(run_dir / "audio" / "audio_002_main.wav", 20.0, 3000)
    _write_wav48(run_dir / "audio" / "audio_003_main.wav", 12.0, 4000)
    rel = "runs/synthetic_v12/audio/"
    side = lambda t, s: {"track": t, "seconds": s}  # noqa: E731
    items = [
        {"id": "v12_audio_001_main", "title": "Uno → Due", "audio_path": rel + "audio_001_main.wav",
         "candidate_id": "c1", "A_out": side("Uno", 61.5), "B_in": side("Due", 33.25), "action": "execute",
         "intent": "handoff_after_phrase", "notes": ["V12: 20 secondi, cambio a 10 s.", "nota <b>due</b>"]},
        {"id": "v12_audio_001_natural_A", "title": "Uno — continuazione originale",
         "audio_path": rel + "audio_001_natural_A.wav", "candidate_id": "c1", "A_out": side("Uno", 61.5),
         "B_in": side("Uno", 61.5), "action": "stay", "intent": "native_reference", "notes": ["20 s."]},
        {"id": "v12_audio_002_main", "title": "Tre → Quattro", "audio_path": rel + "audio_002_main.wav",
         "candidate_id": "c2", "A_out": side("Tre", 10.0), "B_in": side("Quattro", 20.0), "action": "execute",
         "intent": "handoff", "notes": []},
        {"id": "v12_audio_003_main", "title": "Corta", "audio_path": rel + "audio_003_main.wav",
         "candidate_id": "c3", "A_out": side("A", 1.0), "B_in": side("B", 2.0), "action": "execute",
         "intent": "handoff", "notes": []},
        {"id": "v12_audio_004_main", "title": "Mancante", "audio_path": rel + "audio_004_main.wav",
         "action": "execute", "notes": []},
    ]
    win = "C:\\Data\\x\\music_mixing_progetto\\runs\\synthetic_v12\\audio\\"
    (run_dir / "manifest.json").write_text(json.dumps({
        "schema_version": "musicmix.psychology-pool.2",
        "artifacts": [
            {"variant": "main", "artifact": {"path": win + "audio_001_main.wav", "sha256": main}},
            {"variant": "main", "artifact": {"path": win + "audio_002_main.wav", "sha256": "f" * 64}},
        ],
        "natural_references": [],
    }), encoding="utf8")
    path = run_dir / "review_manifest.json"
    path.write_text(json.dumps({"schema_version": "musicmix.psychology-review.1",
                                "title": "MusicMix V12 — sintetico", "items": items}), encoding="utf8")
    return path


def test_review_manifest_is_supported(review, tmp_path):
    before = review.read_bytes()
    out = tmp_path / "out"
    listing = build(review, out, encoder="wav16", log=lambda *_: None)
    assert review.read_bytes() == before

    assert listing["source_manifest"]["schema_version"] == "musicmix.psychology-review.1"
    assert listing["hash_source"]["hashes_found"] == 2
    by_id = {c["id"]: c for c in listing["included"]}
    assert list(by_id) == ["v12_audio_001_main", "v12_audio_001_natural_A", "v12_audio_003_main"]
    reasons = {e["id"]: e["reason"] for e in listing["excluded"]}
    assert reasons == {"v12_audio_002_main": "sha256_diverso", "v12_audio_004_main": "wav_mancante"}

    main = by_id["v12_audio_001_main"]
    assert main["hash_status"] == "verificato"
    assert main["sample_rate"] == 48000 and main["duration_samples"] == 960000 and main["join_sample"] == 480000
    assert main["twenty_s_ten_each_side"] and main["marker_label"] == "cambio"
    assert main["A_out"] == {"track": "Uno", "seconds": 61.5} and main["intent"] == "handoff_after_phrase"
    natural = by_id["v12_audio_001_natural_A"]
    assert natural["hash_status"] == "hash_non_verificabile"
    assert natural["source_sha256"] == hashlib.sha256(
        (review.parent / "audio" / "audio_001_natural_A.wav").read_bytes()).hexdigest()
    assert natural["join_sample"] == 480000 and natural["marker_label"] == "continuazione originale"
    short = by_id["v12_audio_003_main"]
    assert short["duration_samples"] == 12 * 48000 and not short["twenty_s_ten_each_side"]

    page = (out / "index.html").read_text(encoding="utf8")
    assert "MusicMix V12 — sintetico" in page
    assert "<h2>Uno → Due</h2>" in page and "Uno a 61.50 s" in page and "Due a 33.25 s" in page
    assert "handoff_after_phrase" in page and "nota &lt;b&gt;due&lt;/b&gt;" in page
    assert 'class="join-marker" data-join-s="10.000000" style="left:50.0000%" title="punto di cambio"' in page
    assert 'style="left:50.0000%" title="continuazione originale"' in page
    assert "continuazione originale 10.0 s" in page and "hash non verificabile" in page
    assert "Tre → Quattro" not in page


def test_review_without_sibling_manifest_keeps_clips(review, tmp_path):
    (review.parent / "manifest.json").unlink()
    listing = build(review, tmp_path / "o", encoder="wav16", log=lambda *_: None)
    assert listing["hash_source"] is None
    assert {c["hash_status"] for c in listing["included"]} == {"hash_non_verificabile"}
    assert len(listing["included"]) == 4


def test_unknown_schema_is_rejected(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"schema_version": "altro", "items": []}), encoding="utf8")
    with pytest.raises(ValueError, match="schema non supportato"):
        build(path, tmp_path / "o", encoder="wav16", log=lambda *_: None)


# ------------------------------------------------------------ musicmix.dj-review.1

@pytest.fixture
def dj_review(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    run_dir = repo / "runs" / "synthetic_dj"
    (run_dir / "clips").mkdir(parents=True)
    monkeypatch.setenv("MUSICMIX_ROOT", str(repo))
    blend = _write_wav48(run_dir / "clips" / "g01__a_blend8.wav", 36.0, 1000)
    cut = _write_wav48(run_dir / "clips" / "g01__b_cut.wav", 20.0, 2000)
    _write_wav48(run_dir / "clips" / "g01__c_echo_out.wav", 22.0, 3000)
    rel = "runs/synthetic_dj/clips/"
    items = [
        {"id": "g01__a_blend8", "role": "a", "kind": "blend", "label": "blend 8 battute", "params": {"bars": 8},
         "audio_path": rel + "g01__a_blend8.wav", "duration_s": 36.0, "transition_start_s": 10.0, "switch_s": 18.0,
         "transition_end_s": 26.0, "qa": {"verdict": "pass", "warn_reasons": []}, "sha256": blend},
        {"id": "g01__b_cut", "role": "b", "kind": "cut", "params": {},
         "audio_path": rel + "g01__b_cut.wav", "duration_s": 20.0, "transition_start_s": 9.99, "switch_s": 10.0,
         "transition_end_s": 10.0, "qa": {"verdict": "warn", "warn_reasons": ["loudness_jump: salto <b>+2.4</b> dB"]},
         "sha256": cut},
        {"id": "g01__c_echo_out", "role": "c", "kind": "echo_out", "label": "echo out", "params": {},
         "audio_path": rel + "g01__c_echo_out.wav", "duration_s": 22.0, "transition_start_s": 10.0, "switch_s": 12.0,
         "transition_end_s": 12.0, "qa": {"verdict": "pass", "warn_reasons": []}, "sha256": "e" * 64},
    ]
    group = {"id": "g01_uno__due_a95_b16", "title": "Uno → Due",
             "pair": {"a": {"slug": "uno", "title": "Uno"}, "b": {"slug": "due", "title": "Due"}},
             "bpm": {"a": 110.0, "b": 109.0, "stretch_b_pct": 0.92}, "key": {"a": "5A", "b": "5A", "relation": "same"},
             "A_out_s": 207.75, "B_in_s": 44.15, "A_out_bar": 95, "B_in_bar": 16,
             "sections": {"a": "chorus->outro", "b": "chorus"},
             "reasons": ["blend 8 battute: voci sovrapposte 0", "A esce a confine di sezione", "r3", "r4", "quinta motivazione"],
             "items": items}
    empty = {"id": "g02_vuoto", "title": "Vuoto", "items": [
        {"id": "g02__a_cut", "kind": "cut", "audio_path": rel + "missing.wav", "transition_start_s": 9.99,
         "switch_s": 10.0, "transition_end_s": 10.0, "sha256": "0" * 64}]}
    path = run_dir / "review_manifest.json"
    path.write_text(json.dumps({"schema_version": "musicmix.dj-review.1", "title": "Galleria DJ sintetica",
                                "run": "synthetic_dj", "groups": [group, empty]}), encoding="utf8")
    return path


def test_dj_review_manifest_groups_versions_and_preference(dj_review, tmp_path):
    from tools.build_listening_artifact import preference_doc_id
    before = dj_review.read_bytes()
    out = tmp_path / "out"
    listing = build(dj_review, out, encoder="wav16", log=lambda *_: None)
    assert dj_review.read_bytes() == before
    assert listing["source_manifest"]["schema_version"] == "musicmix.dj-review.1"

    by_id = {c["id"]: c for c in listing["included"]}
    assert list(by_id) == ["g01__a_blend8", "g01__b_cut"]
    reasons = {e["id"]: e["reason"] for e in listing["excluded"]}
    assert reasons == {"g01__c_echo_out": "sha256_diverso", "g02__a_cut": "wav_mancante"}
    blend = by_id["g01__a_blend8"]
    assert blend["hash_status"] == "verificato" and blend["clip_rule_ok"]
    assert blend["doc_id"] == "synthetic_dj__g01__a_blend8" and blend["group_id"] == "g01_uno__due_a95_b16"
    cut = by_id["g01__b_cut"]
    assert cut["label"] == "taglio" and cut["clip_rule_ok"] and cut["qa_verdict"] == "warn"

    assert [g["id"] for g in listing["groups"]] == ["g01_uno__due_a95_b16"]
    pref = preference_doc_id("synthetic_dj", "g01_uno__due_a95_b16")
    assert pref == "synthetic_dj__g01_uno__due_a95_b16__preferenza"
    assert listing["groups"][0]["preference_doc_id"] == pref
    assert listing["groups"][0]["items"] == ["g01__a_blend8", "g01__b_cut"]

    page = (out / "index.html").read_text(encoding="utf8")
    assert "<title>Ascolti DJ MusicMix</title>" in page and "Galleria DJ sintetica" in page
    assert "<h2>Uno → Due</h2>" in page and "110.0 → 109.0" in page and "(B +0.9%)" in page
    assert "5A → 5A · same" in page and "207.75 s · battuta 95 · chorus-&gt;outro" in page
    assert "44.15 s · battuta 16 · chorus" in page and "altre motivazioni" in page
    # transition region 10..26 s of 36 s, switch at 18 s
    assert 'class="region" data-start-s="10.000000" data-end-s="26.000000" style="left:27.7778%;width:44.4444%"' in page
    assert 'class="switch-marker" data-switch-s="18.000000" style="left:50.0000%"' in page
    assert 'data-act="transition">vai alla transizione' in page and "−5 s" in page and "+5 s" in page
    assert '<span class="badge kind">blend 8 battute</span>' in page and '<span class="badge kind">taglio</span>' in page
    assert page.count('class="badge warnqa"') == 1 and "salto &lt;b&gt;+2.4&lt;/b&gt; dB" in page
    for label in ("battute sfasate", "cambio nel punto sbagliato della frase", "volume che salta", "due voci insieme",
                  "transizione troppo lunga", "troppo corta", "suona come un DJ"):
        assert label in page
    assert "Quale versione preferisci?" in page and f'data-pref="{pref}"' in page
    assert 'name="pref-g01_uno__due_a95_b16" value="g01__a_blend8"' in page
    assert 'name="pref-g01_uno__due_a95_b16" value="nessuna"' in page
    assert "g02_vuoto" not in page and "g01__c_echo_out" not in page
    assert 'window.claude.use("db")' in page and "onSnapshot" in page and 'preload="metadata"' in page


def test_dj_review_blind_hides_technique_until_vote(dj_review, tmp_path):
    """Proposta E1 dei libri (Margulis 2014 p. 15): niente descrizione tecnica visibile prima del voto."""
    page0 = (tmp_path / "o0")
    build(dj_review, page0, encoder="wav16", log=lambda *_: None)
    normal = (page0 / "index.html").read_text(encoding="utf8")
    assert "data-reveal" not in normal.split("<script")[0] and '"blind": false' in normal
    out = tmp_path / "blind"
    build(dj_review, out, encoder="wav16", log=lambda *_: None, blind=True)
    page = (out / "index.html").read_text(encoding="utf8")
    body = page.split("<script")[0]
    assert '<span class="badge kind">versione 1</span><span class="badge kind" data-reveal hidden>blend 8 battute</span>' in body
    assert 'class="badge warnqa" data-reveal hidden' in body and 'class="qa-reasons small" data-reveal hidden' in body
    assert '<div class="group-meta" data-reveal-group hidden>' in body  # BPM, keys and plan reasons hidden
    assert "transizione 10.0–26.0 s" not in body and "cambio attorno a 18.0 s" in body
    assert '<span>versione 2</span></label>' in body and "Margulis 2014 p. 15" in body
    assert '"blind": true' in page and "plays_before_save" in page and "function reveal(docId)" in page
    assert 'class="region" data-start-s="10.000000"' in body  # seek to the transition stays available


# ------------------------------------------------------------ musicmix.dj_set.1 / dj-set.1

def _synthetic_set(run_dir: Path, wav_sha: str, duration: float = 30.0) -> dict:
    tracks = [("uno", "Artista Uno", "Primo <brano>", 0.0, 10.0, 85.0, "2A"),
              ("due", "", "Secondo", 10.0, 21.0, 83.0, "2A"),
              ("tre", "Artista Tre", "Terzo", 21.0, duration, 90.0, "8A")]
    tracklist = [{"position": i + 1, "slug": s, "artist": a, "title": t, "set_start_s": st, "set_end_s": en,
                  "bpm": b, "camelot": k, "energy": 0.5} for i, (s, a, t, st, en, b, k) in enumerate(tracks)]
    transitions = [
        {"index": 1, "a": "uno", "b": "due", "type": "filter_out", "label": "filter out 2 batt.",
         "rule": "R6_both_vocal:alternativa", "set_times_s": {"start": 7.0, "switch": 10.0, "end": 10.0},
         "qa": {"verdict": "warn", "reasons": ["loudness_jump: salto <b>+2.8</b> dB"], "kept_despite_fail": False}},
        {"index": 2, "a": "due", "b": "tre", "type": "blend4", "label": "blend 4 battute", "rule": "R8_one_voice",
         "set_times_s": {"start": 18.0, "switch": 21.0, "end": 24.0},
         "qa": {"verdict": "fail", "reasons": ["loudness_jump: +7.0 dB"], "kept_despite_fail": True}},
        {"index": 3, "a": "tre", "b": "x", "type": "echo_out", "label": "echo out",   # outside the set: excluded
         "set_times_s": {"start": 29.0, "switch": 31.0, "end": 31.0}, "qa": {"verdict": "pass", "reasons": []}},
    ]
    return {"schema_version": "musicmix.dj_set.1", "run": run_dir.name, "tracklist": tracklist,
            "transitions": transitions, "qa_counts_final": {"warn": 1, "fail": 1},
            "output": {"wav": f"runs/{run_dir.name}/set.wav", "wav_sha256": wav_sha, "duration_s": duration,
                       "sample_rate": 48000}}


@pytest.fixture
def dj_set(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    run_dir = repo / "runs" / "synthetic_set"
    run_dir.mkdir(parents=True)
    monkeypatch.setenv("MUSICMIX_ROOT", str(repo))
    sha = _write_wav48(run_dir / "set.wav", 30.0, 1000)
    path = run_dir / "set.json"
    path.write_text(json.dumps(_synthetic_set(run_dir, sha)), encoding="utf8")
    return path


def test_dj_set_page_player_tracklist_transitions_and_set_judgment(dj_set, tmp_path):
    from tools.build_listening_artifact import set_doc_id, transition_doc_id
    before = dj_set.read_bytes()
    out = tmp_path / "out"
    listing = build(dj_set, out, encoder="wav16", log=lambda *_: None)
    assert dj_set.read_bytes() == before
    assert listing["kind"] == "dj-set" and listing["source_manifest"]["normalized_schema"] == "musicmix.dj-set.1"
    assert listing["audio"]["source"]["hash_status"] == "verificato" and (out / "audio" / "set.wav").is_file()
    assert [t["index"] for t in listing["transitions"]] == [1, 2]
    assert [(e["kind"], e["id"], e["reason"]) for e in listing["excluded"]] == [("transizione", 3, "marcatori_non_validi")]
    assert transition_doc_id("synthetic_set", 1) == "synthetic_set__t01" == listing["transitions"][0]["doc_id"]
    assert set_doc_id("synthetic_set") == "synthetic_set__set"
    norm = json.loads((out / "set_manifest.json").read_text(encoding="utf8"))
    assert norm["schema_version"] == "musicmix.dj-set.1" and norm["converted_from"] == "musicmix.dj_set.1"

    page = (out / "index.html").read_text(encoding="utf8")
    assert "<title>Ascolto set DJ</title>" in page and 'src="audio/set.wav"' in page
    assert page.count("<audio") == 1 and 'preload="metadata"' in page
    # regions coloured by type: T01 7..10 s, T02 18..24 s of 30 s; switch lines; track boundaries at 10 and 21 s
    assert 'class="region f-filter" data-index="1" style="left:23.3333%;width:10.0000%"' in page
    assert 'class="region f-blend" data-index="2" style="left:60.0000%;width:20.0000%"' in page
    assert 'class="switch" data-switch-s="21.000000" style="left:70.0000%"' in page
    assert page.count('class="bound"') == 2 and 'class="bound" style="left:33.3333%"' in page
    # tracklist: artist – title, BPM, Camelot, entry minute, clickable
    assert "Artista Uno – Primo &lt;brano&gt;" in page and "85.0 BPM · 2A · entra a 0:00" in page
    assert 'data-start-s="21.000000"' in page and "entra a 0:21" in page
    assert '<span class="tname">Secondo</span>' in page
    # transition rows
    assert 'id="t01" data-doc="synthetic_set__t01"' in page and 'data-doc="synthetic_set__t02"' in page
    assert '<span class="badge f-blend">blend 4 battute</span>' in page
    assert '<span class="badge qa-warn"' in page and "QA fail (tenuta)" in page
    assert "salto &lt;b&gt;+2.8&lt;/b&gt; dB" in page and "R6_both_vocal:alternativa" in page
    assert "vai alla transizione" in page and "riascolta" in page and 'class="loopbox"' in page
    assert "−10 s" in page and "+10 s" in page
    for label in ("battute sfasate", "cambio nel punto sbagliato della frase", "volume che salta", "due voci insieme",
                  "transizione troppo lunga", "troppo corta", "suona come un DJ"):
        assert label in page
    # overall set judgment
    assert 'id="set-judge" data-doc="synthetic_set__set"' in page and "Scorrevolezza" in page
    assert "Ordine dei brani sensato?" in page and 'name="set-order" value="in_parte"' in page
    assert 'window.claude.use("db")' in page and "onSnapshot" in page and '"lead": 4.0' in page
    assert "Artista Uno – Primo \\u003cbrano\\u003e" in page   # JSON payload escaped
    assert "echo out" not in page.split("<main>")[1].split("<script")[0]


def test_dj_set_uses_given_mp3_and_rejects_bad_wav_hash(dj_set, tmp_path):
    mp3 = tmp_path / "set_128k.mp3"
    mp3.write_bytes(b"ID3fake-mp3")
    listing = build(dj_set, tmp_path / "o1", log=lambda *_: None, set_mp3=mp3)
    assert listing["encoder"]["kind"] == "copia" and listing["audio"]["path"] == "audio/set.mp3"
    assert listing["audio"]["source"]["hash_status"] == "hash_non_verificabile"
    assert (tmp_path / "o1" / "audio" / "set.mp3").read_bytes() == b"ID3fake-mp3"

    data = json.loads(dj_set.read_text(encoding="utf8"))
    data["output"]["wav_sha256"] = "0" * 64
    bad = dj_set.with_name("bad_set.json")
    bad.write_text(json.dumps(data), encoding="utf8")
    with pytest.raises(ValueError, match="sha256 del WAV diverso"):
        build(bad, tmp_path / "o2", encoder="wav16", log=lambda *_: None)


def test_dj_set_normalized_manifest_and_csv_fallback(dj_set, tmp_path):
    from tools.build_listening_artifact import convert_set
    run_dir = dj_set.parent
    (run_dir / "tracklist.csv").write_text(
        "position,slug,artist,title,set_start_s,set_end_s,bpm,camelot,energy,transition_out_type,rule,"
        "transition_start_s,switch_s,transition_end_s,qa\n"
        "1,uno,A,Uno,0.000,15.000,85.00,2A,0.1,echo_out,R4,12.000,15.000,15.000,warn\n"
        "2,due,B,Due,15.000,30.000,83.00,3A,0.2,,,,,,\n", encoding="utf8")
    data = json.loads(dj_set.read_text(encoding="utf8"))
    del data["tracklist"], data["transitions"]
    norm = convert_set(data, run_dir)
    assert norm["tracklist_source"] == "tracklist.csv" and [t["slug"] for t in norm["tracks"]] == ["uno", "due"]
    assert norm["transitions"][0]["family"] == "echo" and norm["transitions"][0]["b"] == "due"
    # the normalized musicmix.dj-set.1 file is itself a valid input
    path = run_dir / "set_manifest.json"
    path.write_text(json.dumps(norm), encoding="utf8")
    listing = build(path, tmp_path / "o", encoder="wav16", log=lambda *_: None)
    assert listing["source_manifest"]["schema_version"] == "musicmix.dj-set.1"
    assert [t["doc_id"] for t in listing["transitions"]] == ["synthetic_set__t01"]


@pytest.mark.skipif(not __import__("shutil").which("ffmpeg"), reason="ffmpeg assente")
def test_dj_set_auto_encoder_is_128k_mp3(dj_set, tmp_path):
    from tools.build_listening_artifact import choose_encoder
    if choose_encoder() != "ffmpeg-mp3":
        pytest.skip("ffmpeg senza libmp3lame")
    listing = build(dj_set, tmp_path / "o", log=lambda *_: None)
    assert listing["encoder"] == {"kind": "ffmpeg-mp3", "bitrate": "128k", "ffmpeg": None}
    assert listing["audio"]["path"] == "audio/set.mp3"
    br = listing["audio"]["probe"].get("bit_rate")
    assert br is None or abs(br - 128000) < 4000
