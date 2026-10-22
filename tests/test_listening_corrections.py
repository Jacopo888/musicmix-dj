"""Correzioni strutturate (dj-review.2), approvazione del set, torneo dei cue, piano di correzioni, metrica di accordo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys
import wave

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import apply_corrections as AC  # noqa: E402
import build_listening_artifact as BLA  # noqa: E402
import decision_agreement as DA  # noqa: E402
import listening_corrections as LC  # noqa: E402
import listening_pages as LP  # noqa: E402

RATE = 48000


def _wav(path: Path, seconds: float, level: int = 1000, rate: int = 8000) -> str:
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", level if (i // 40) % 2 else -level) for i in range(n)))
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ campi di correzione
def test_corrections_of_reads_v1_and_v2_docs():
    v1 = LC.corrections_of({"score": 4, "tags": ["battute_sfasate"]})
    assert v1["schema"] == 1 and v1["a_exit"] is None and v1["grid_offbeat"] is True
    assert LC.is_corrected(v1)  # la battuta sfasata (tag già esistente) è una segnalazione
    v2 = LC.corrections_of({"corrections": {"a_exit": "-1", "b_entry": "ok", "b_pickup": True, "b_section": "chorus",
                                            "technique": "blend8", "grid": "meta_tempo", "bogus": 1}})
    assert v2["schema"] == 2 and v2["a_exit"] == "-1" and v2["b_entry"] == "ok" and v2["b_section"] == "chorus"
    assert LC.bars_shift(v2["a_exit"]) == -1 and LC.bars_shift("ok") == 0 and LC.bars_shift("fine_frase") is None
    assert "B entra con il pickup" in LC.describe(v2) and "metà tempo" in LC.describe(v2)
    ok = LC.corrections_of({"corrections": {"a_exit": "ok", "b_entry": "ok"}})
    assert not LC.is_corrected(ok)
    bad = LC.corrections_of({"corrections": {"a_exit": "+7", "b_section": "x"}})
    assert bad["a_exit"] is None and bad["b_section"] is None


def test_read_docs_accepts_single_docs_and_copy_json_exports(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"doc_id": "r__t01", "type": "versione", "clip_id": "x", "saved_at": "1"}))
    (tmp_path / "exp.json").write_text(json.dumps({"run": "r", "judgments": {"r__t01": {"type": "versione", "saved_at": "2"},
                                                                         "r__t02": {"type": "versione", "saved_at": "1"}}}))
    docs = LC.read_docs([tmp_path])
    assert len(docs) == 3 and all(d["_file_sha256"] for d in docs)
    last = LC.latest(docs)
    assert set(last) == {"r__t01", "r__t02"} and last["r__t01"]["saved_at"] == "2"


@pytest.fixture
def dj_review2(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    run_dir = repo / "runs" / "synthetic_dj2"
    (run_dir / "clips").mkdir(parents=True)
    monkeypatch.setenv("MUSICMIX_ROOT", str(repo))
    sha = _wav(run_dir / "clips" / "g01__a.wav", 20.0)
    group = {"id": "g01", "title": "Uno → Due", "pair": {"a": {"slug": "uno"}, "b": {"slug": "due"}},
             "A_out_s": 100.0, "B_in_s": 20.0, "A_out_bar": 40, "B_in_bar": 8,
             "items": [{"id": "g01__a", "kind": "cut", "audio_path": "runs/synthetic_dj2/clips/g01__a.wav",
                        "transition_start_s": 9.99, "switch_s": 10.0, "transition_end_s": 10.0, "sha256": sha}]}
    path = run_dir / "review_manifest.json"
    path.write_text(json.dumps({"schema_version": "musicmix.dj-review.2", "groups": [group]}), encoding="utf8")
    return path


def test_dj_review2_page_has_structured_corrections(dj_review2, tmp_path):
    listing = BLA.build(dj_review2, tmp_path / "o", encoder="wav16", log=lambda *_: None, blind=True)
    assert listing["source_manifest"]["schema_version"] == "musicmix.dj-review.2"
    page = (tmp_path / "o" / "index.html").read_text(encoding="utf8")
    body = page.split("<script")[0]
    assert '<details class="corr">' in body and "Uscita di A" in body and "Ingresso di B" in body
    assert 'name="c-g01__a-a_exit" value="fine_frase"' in body and 'name="c-g01__a-b_entry" value="-2"' in body
    assert 'data-corr-flag="b_pickup"' in body and 'data-corr-select="b_section"' in body
    assert 'data-corr-flag="wrong_b"' in body and 'data-corr-flag="stay_longer_a"' in body
    assert '<option value="halving_build">' in body and 'value="doppio_tempo"' in body and 'value="meta_tempo"' in body
    assert "battute sfasate" in body  # tag già esistente, letto come grid_offbeat
    assert '"judgment_schema": "musicmix.dj-review.2"' in page and "corrections: readCorr(f)" in page
    assert '"cue": {"a": "uno", "b": "due", "a_out_bar": 40, "b_in_bar": 8' in page
    assert "plays_before_save" in page and '"blind": true' in page and 'preload="metadata"' in page


# ------------------------------------------------------------------ approvazione del set
def _set_doc(run: str, wav_sha: str, sources: dict | None = None) -> dict:
    tracklist = [{"position": 1, "slug": "uno", "artist": "A", "title": "Uno", "set_start_s": 0.0, "set_end_s": 12.0},
                 {"position": 2, "slug": "due", "artist": "B", "title": "Due", "set_start_s": 12.0, "set_end_s": 22.0},
                 {"position": 3, "slug": "tre", "artist": "", "title": "Tre", "set_start_s": 22.0, "set_end_s": 30.0}]
    tr = [{"index": 1, "a": "uno", "b": "due", "type": "blend4", "label": "blend 4 battute",
           "set_times_s": {"start": 10.0, "switch": 12.0, "end": 14.0}, "qa": {"verdict": "warn", "reasons": ["x"]},
           "plan": {"a_out_bar": 40, "b_in_bar": 8, "a_out_s": 80.0, "b_in_s": 16.0, "type": "blend"}},
          {"index": 2, "a": "due", "b": "tre", "type": "cut", "label": "taglio",
           "set_times_s": {"start": 21.99, "switch": 22.0, "end": 22.0}, "qa": {"verdict": "pass", "reasons": []},
           "plan": {"a_out_bar": 16, "b_in_bar": 4, "a_out_s": 32.0, "b_in_s": 8.0, "type": "cut"}}]
    doc = {"schema_version": "musicmix.dj_set.1", "run": run, "tracklist": tracklist, "transitions": tr,
           "output": {"wav": f"runs/{run}/set.wav", "wav_sha256": wav_sha, "duration_s": 30.0, "sample_rate": RATE}}
    if sources:
        doc["sources"] = sources
    return doc


@pytest.fixture
def approval_set(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    run_dir = repo / "runs" / "synthetic_appr"
    run_dir.mkdir(parents=True)
    monkeypatch.setenv("MUSICMIX_ROOT", str(repo))
    sha = _wav(run_dir / "set.wav", 30.0, rate=RATE)
    path = run_dir / "set.json"
    path.write_text(json.dumps(_set_doc("synthetic_appr", sha)), encoding="utf8")
    return path


def test_set_approval_page(approval_set, tmp_path):
    before = approval_set.read_bytes()
    out = tmp_path / "appr"
    listing = BLA.build(approval_set, out, encoder="wav16", log=lambda *_: None, kind="set-approval")
    assert approval_set.read_bytes() == before
    assert listing["kind"] == "set-approval" and listing["page_schema"] == "musicmix.set-approval.1"
    assert [t["doc_id"] for t in listing["transitions"]] == ["synthetic_appr__set_t01", "synthetic_appr__set_t02"]
    assert listing["db"]["set_doc_id"] == "synthetic_appr__set" == LP.approval_set_doc_id("synthetic_appr")
    assert listing["transitions"][0]["plan"]["a_out_bar"] == 40 and listing["order"] == ["uno", "due", "tre"]
    page = (out / "index.html").read_text(encoding="utf8")
    body = page.split("<script")[0]
    assert "<title>Approvazione del set</title>" in page and page.count("<audio") == 1 and 'preload="metadata"' in page
    assert 'id="t01" data-doc="synthetic_appr__set_t01"' in body and 'data-act="zone"' in body
    assert "ascolta la zona (−10 s)" in body and '"zone_lead": 10.0' in page
    for v in ("approvata", "da_correggere", "da_rifare_altro_brano"):
        assert f'name="verdict-t1" value="{v}"' in body
    assert "da rifare con altro brano" in body and '<details class="corr">' in body
    assert 'id="set-judge" data-doc="synthetic_appr__set"' in body
    for k in ("set-flow", "set-energy", "set-variety", "set-repetitiveness", "set-verdict"):
        assert f'name="{k}"' in body
    assert "Ripetitività" in body and "approvato dall&#x27;inizio alla fine" in body
    assert "A esce alla battuta 40 · B entra dalla battuta 8" in body
    assert 'class="badge f-blend">blend 4 battute' in body  # non cieca: tecnica visibile
    assert "−10 s" in body and "+10 s" in body and 'window.claude.use("db")' in page and "onSnapshot" in page
    assert '"type": "set_approval_transizione"' not in page  # il tipo è nel JS, non nei dati
    assert 'type: "set_approval_transizione"' in page and "plays_before_save: card._plays" in page


def test_set_approval_blind_hides_technique(approval_set, tmp_path):
    BLA.build(approval_set, tmp_path / "b", encoder="wav16", log=lambda *_: None, kind="set-approval", blind=True)
    body = (tmp_path / "b" / "index.html").read_text(encoding="utf8").split("<script")[0]
    assert 'class="badge f-blend" data-reveal hidden>blend 4 battute' in body
    assert 'class="region f-other"' in body and 'class="region f-blend"' not in body
    assert '<p class="muted small" data-reveal hidden>A esce alla battuta 40' in body
    assert "Margulis 2014 p. 15" in body


def test_set_bitrate_stays_under_limit():
    assert LP.choose_set_bitrate(600.0) == "128k"          # 9,6 MB
    assert LP.choose_set_bitrate(1049.37) == "96k"         # il set v3: 128k = 16,8 MB, 112k = 14,7 MB > 95 % di 15 MB
    assert LP.choose_set_bitrate(5000.0) == "64k"


def test_set_approval_copies_given_mp3(approval_set, tmp_path):
    mp3 = tmp_path / "set_96k.mp3"
    mp3.write_bytes(b"ID3fake")
    listing = BLA.build(approval_set, tmp_path / "m", log=lambda *_: None, kind="set-approval", set_mp3=mp3)
    assert listing["encoder"]["kind"] == "copia" and listing["audio"]["path"] == "audio/set.mp3"


# ------------------------------------------------------------------ torneo dei cue
@pytest.fixture
def variants(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    clips = repo / "runs" / "b_entry_fix_synth" / "clips"
    clips.mkdir(parents=True)
    monkeypatch.setenv("MUSICMIX_ROOT", str(repo))
    cues = [("v0", 48, 0, 0), ("v1", 48, 1, 0), ("v2", 48, 1, 12000), ("v3", 47, 0, 0), ("v4", 49, 2, 0)]
    vs = []
    for k, (vid, a, b, li) in enumerate(cues):
        sha = _wav(clips / f"{vid}.wav", 27.4, level=500 + 100 * k)
        vs.append({"id": vid, "audio_path": f"runs/b_entry_fix_synth/clips/{vid}.wav", "sha256": sha, "a_out_bar": a,
                   "b_in_bar": b, "b_lead_in_samples": li, "transition_start_s": 10.0, "switch_s": 13.7,
                   "transition_end_s": 17.4})
    (clips / "v5.mp3").write_bytes(b"ID3fake-mp3")
    vs.append({"id": "v5", "audio_path": "runs/b_entry_fix_synth/clips/v5.mp3", "a_out_bar": 48, "b_in_bar": 2,
               "transition_start_s": 10.0, "switch_s": 13.7, "transition_end_s": 17.4, "duration_s": 27.4})
    vs.append(dict(vs[0], id="w_tech", technique="echo_out"))
    vs.append(dict(vs[0], id="w_hash", sha256="0" * 64))
    m = {"schema_version": "musicmix.cue-variants.1", "run": "b_entry_fix_synth", "title": "Torneo sintetico",
         "tournaments": [{"id": "t04", "transition_index": 4, "a": {"slug": "goosebumps", "title": "Goosebumps"},
                          "b": {"slug": "tint", "title": "Tint"}, "technique": "blend4",
                          "system": {"a_out_bar": 48, "b_in_bar": 0}, "variants": vs},
                         {"id": "t05", "technique": "cut", "variants": [vs[0]]}]}
    path = repo / "runs" / "b_entry_fix_synth" / "variants.json"
    path.write_text(json.dumps(m), encoding="utf8")
    return path


def test_cue_tournament_page_and_comparisons(variants, tmp_path):
    before = variants.read_bytes()
    listing = BLA.build(variants, tmp_path / "t", encoder="wav16", log=lambda *_: None)  # schema riconosciuto
    assert variants.read_bytes() == before
    assert listing["kind"] == "cue-tournament" and listing["blind"] is True
    reasons = {(e.get("id")): e["reason"] for e in listing["excluded"]}
    assert reasons == {"w_tech": "tecnica_diversa", "w_hash": "sha256_diverso", "t05": "meno_di_due_varianti"}
    t = listing["tournaments"][0]
    assert t["doc_id"] == "b_entry_fix_synth__t04__torneo" and len(t["variants"]) == 6
    assert sorted(t["order"]) == ["v0", "v1", "v2", "v3", "v4", "v5"]
    again = BLA.build(variants, tmp_path / "t2", encoder="wav16", log=lambda *_: None)
    assert again["tournaments"][0]["order"] == t["order"]  # ordine casuale riproducibile
    other = BLA.build(variants, tmp_path / "t3", encoder="wav16", log=lambda *_: None, seed=7)
    assert sorted(other["tournaments"][0]["order"]) == sorted(t["order"])
    by = {v["id"]: v for v in t["variants"]}
    assert by["v0"]["is_system"] and not by["v1"]["is_system"] and by["v4"]["a_shift_bars"] == 1 and by["v4"]["b_shift_bars"] == 2
    assert by["v5"]["encoding"] == "copia" and by["v5"]["hash_status"] == "hash_non_verificabile"
    cmp = {(c["x"], c["y"]): c for c in t["comparisons"]}
    assert cmp[("v0", "v1")]["varies"] == "solo_cue_B" and cmp[("v0", "v1")]["delta_b_bars"] == 1
    assert cmp[("v0", "v3")]["varies"] == "solo_cue_A" and cmp[("v0", "v3")]["delta_a_bars"] == -1
    assert cmp[("v1", "v2")]["varies"] == "solo_pickup_B" and cmp[("v0", "v4")]["varies"] == "cue_A_e_B"
    page = (tmp_path / "t" / "index.html").read_text(encoding="utf8")
    body = page.split("<script")[0]
    assert "<title>Torneo dei cue</title>" in page and "tecnica fissa: blend 4 battute" in body
    assert body.count('<span class="badge kind">variante ') == 6 and body.count("data-reveal hidden") == 6
    first = t["order"][0]
    assert body.index(f'data-variant="{first}"') < min(body.index(f'data-variant="{v}"') for v in t["order"][1:])
    assert 'name="best-t04" value="nessuna"' in body and body.count("data-acceptable") == 6
    assert body.count('<details class="corr">') == 6 and "scelta del sistema" in body
    assert f'data-doc="{LP.variant_doc_id("b_entry_fix_synth", "t04", "v0")}"' in body
    assert 'type: "torneo"' in page and "best_cue" in page and "order_shown" in page


# ------------------------------------------------------------------ piano di correzioni
def _analysis(slug: str, n_bars: int, period: float, sections: list[tuple[str, int]]) -> dict:
    bar = 4 * period
    labels = []
    for j in range(n_bars):
        lab = [s for s, b in sections if b <= j][-1]
        labels.append(lab)
    starts = {b for _, b in sections}
    phrases = [{"bar_index": j, "time_s": 1.0 + j * bar, "section_label": labels[j], "section_start": j in starts,
                "phrase_level": 16 if j in starts else (8 if j % 8 == 0 else (4 if j % 4 == 0 else 0))}
               for j in range(n_bars)]
    return {"slug": slug, "title": slug.title(), "track_id": hashlib.sha256(slug.encode()).hexdigest(),
            "duration_s": 1.0 + n_bars * bar, "tempo": {"bpm": 60 / period, "period_s": period, "provider": "test",
                                                         "confidence": "high"},
            "grid": {"downbeats_s": [p["time_s"] for p in phrases], "n_bars": n_bars}, "phrases": phrases,
            "sections": [{"label": s, "bar_index": b} for s, b in sections]}


@pytest.fixture
def corr_case(tmp_path):
    run_dir = tmp_path / "runs" / "synthetic_corr"
    (run_dir / "dj_plan" / "tracks").mkdir(parents=True)
    an = {"uno": _analysis("uno", 64, 0.5, [("intro", 0), ("verse", 8), ("chorus", 24), ("verse", 40)]),
          "due": _analysis("due", 48, 0.5, [("intro", 0), ("verse", 4), ("chorus", 20), ("verse", 36)]),
          "tre": _analysis("tre", 48, 0.5, [("intro", 0), ("verse", 8), ("chorus", 16)])}
    sources = {}
    for s, a in an.items():
        p = run_dir / "dj_plan" / "tracks" / f"{s}.json"
        p.write_text(json.dumps(a))
        sources[s] = {"analysis": str(p), "title": s.title(), "artist": "X"}
    doc = _set_doc("synthetic_corr", "0" * 64, sources)
    set_path = run_dir / "set.json"
    set_path.write_text(json.dumps(doc))
    return set_path, an


def _jdoc(run, i, a, b, verdict, corr=None, saved="2026-10-21T10:00:00Z", **kw):
    return {"doc_id": f"{run}__set_t{i:02d}", "run": run, "type": "set_approval_transizione", "transition_index": i,
            "a": a, "b": b, "verdict": verdict, "corrections": dict(LC.EMPTY, **(corr or {})), "saved_at": saved,
            "blind": True, **kw}


def test_plan_translates_corrections_on_the_grid(corr_case, tmp_path):
    set_path, an = corr_case
    before = set_path.read_bytes()
    jd = tmp_path / "judg"
    jd.mkdir()
    run = "synthetic_corr"
    docs = {"j1": _jdoc(run, 1, "uno", "due", "da_correggere", {"a_exit": "-1", "b_section": "chorus", "b_pickup": True,
                                                                "technique": "blend8"}),
            "j2": _jdoc(run, 2, "due", "tre", "approvata"),
            "old": _jdoc(run, 2, "due", "tre", "da_rifare_altro_brano", saved="2026-10-21T09:00:00Z"),
            "set": {"doc_id": f"{run}__set", "run": run, "type": "set_approval", "set_verdict": "da_correggere", "flow": 3}}
    (jd / "export.json").write_text(json.dumps({"run": run, "judgments": {d["doc_id"] + ("_old" if k == "old" else ""): d
                                                                           for k, d in docs.items()}}))
    plan = AC.build_plan(set_path, [jd], reuse=False)
    assert set_path.read_bytes() == before
    t1, t2 = plan["transitions"]
    assert t1["status"] == "corretta" and t1["original"]["a_out_bar"] == 40 and t1["original"]["b_in_bar"] == 8
    c = t1["corrected"]
    assert c["a_out_bar"] == 39 and c["b_in_bar"] == 20 and c["type_key"] == "blend8"
    assert c["a_out_s"] == pytest.approx(1.0 + 39 * 2.0) and c["b_in_s"] == pytest.approx(1.0 + 20 * 2.0)
    assert c["b_lead_in_samples"] == 24000  # un battito (0,5 s) senza modello dj_phrase [H]
    assert t1["needs_manual"] == []
    d = t1["decisions"]
    assert d["order"]["approved"] and not d["exit"]["approved"] and d["exit"]["shift_bars"] == -1
    assert not d["entry"]["approved"] and d["entry"]["shift_bars"] == 12 and not d["technique"]["approved"]
    assert t2["status"] == "approvata" and t2["decisions"]["exit"]["explicit"]
    assert plan["counts"]["corretta"] == 1 and plan["counts"]["approvata"] == 1
    assert plan["set_judgment"]["set_verdict"] == "da_correggere"
    pins = {p["index"]: p for p in plan["runner"]["pins"]}
    assert pins[1]["status"] == "corretta" and pins[1]["a_out_bar"] == 39 and pins[2]["status"] == "approvata"
    assert plan["runner"]["rerender"] == [1] and plan["runner"]["forbid_pairs"] == []


def test_plan_rules_phrase_end_stay_longer_wrong_b_and_grid(corr_case, tmp_path):
    set_path, an = corr_case
    run = "synthetic_corr"
    jd = tmp_path / "j"
    jd.mkdir()
    (jd / "1.json").write_text(json.dumps(_jdoc(run, 1, "uno", "due", "da_correggere", {"a_exit": "fine_frase"})))
    (jd / "2.json").write_text(json.dumps(_jdoc(run, 2, "due", "tre", "da_rifare_altro_brano")))
    plan = AC.build_plan(set_path, [jd], reuse=False)
    t1, t2 = plan["transitions"]
    assert t1["corrected"]["a_out_bar"] == 44  # prima battuta dopo la 40 con phrase_level >= 4 (nessun modello dj_phrase)
    assert "phrase_level >= 4" in t1["changes"][0]["rule"]
    assert t2["status"] == "ripianifica" and plan["runner"]["forbid_pairs"] == [["due", "tre"]]
    assert t2["decisions"]["order"]["approved"] is False
    (jd / "1.json").write_text(json.dumps(_jdoc(run, 1, "uno", "due", "da_correggere", {"stay_longer_a": True})))
    (jd / "2.json").write_text(json.dumps(_jdoc(run, 2, "due", "tre", "da_correggere", {"grid": "doppio_tempo"})))
    plan = AC.build_plan(set_path, [jd], reuse=False)
    t1, t2 = plan["transitions"]
    assert t1["corrected"]["a_out_bar"] == 48 and t1["status"] == "corretta"  # >= 40 + 8 con phrase_level >= 8
    assert t2["status"] == "manuale" and any("griglia" in m for m in t2["needs_manual"])
    assert 2 not in plan["runner"]["rerender"]


def test_tournament_and_gallery_entries(corr_case, tmp_path):
    set_path, an = corr_case
    run = "synthetic_corr"
    jd = tmp_path / "j"
    jd.mkdir()
    tour = {"doc_id": "x__t1__torneo", "run": "x", "type": "torneo", "tournament_id": "t1", "transition_index": 1,
            "pair": {"a": "uno", "b": "due"}, "technique": "blend4", "best": "v2",
            "best_cue": {"a_out_bar": 40, "b_in_bar": 9, "b_lead_in_samples": 0}, "saved_at": "2026-10-21T11:00:00Z"}
    gal = {"doc_id": "g__c", "run": "g", "type": "versione", "schema_version": "musicmix.dj-review.2", "clip_id": "c",
           "kind": "cut", "cue": {"a": "due", "b": "tre", "a_out_bar": 16, "b_in_bar": 4},
           "corrections": dict(LC.EMPTY, b_entry="-1"), "score": 5, "saved_at": "2026-10-21T11:00:00Z"}
    v1 = {"doc_id": "g__old", "run": "g", "type": "versione", "clip_id": "old", "score": 3, "tags": [], "saved_at": "1"}
    for k, d in {"t": tour, "g": gal, "v1": v1}.items():
        (jd / f"{k}.json").write_text(json.dumps(d))
    plan = AC.build_plan(set_path, [jd], reuse=False)
    t1, t2 = plan["transitions"]
    assert t1["source"] == "cue_tournament" and t1["corrected"]["b_in_bar"] == 9 and t1["status"] == "corretta"
    assert t2["source"] == "gallery" and t2["corrected"]["b_in_bar"] == 3 and t2["corrected"]["type_key"] == "cut"
    assert {x["reason"] for x in plan["ignored_docs"]} == {"senza_correzioni_strutturate (dj-review.1)"}
    refs = AC.reference_records(plan)
    assert [r["id"] for r in refs] == ["uno__due__A40_B9_blend4"]  # il torneo è una verifica d'ascolto; la galleria no
    r = refs[0]
    assert r["schema_version"] == "musicmix.reference-transition.1" and r["use"] == "verifica"
    assert r["entry_b"]["sample"] == round((1.0 + 9 * 2.0) * 48000) and r["exit_a"]["bar"] == 40
    assert r["tracks"]["a"]["audio_sha256"] == an["uno"]["track_id"] and r["verification"]["method"] == "cue_tournament"
    assert r["exit_a"]["words"]["status"] == "modello_dj_phrase_assente"
    out = tmp_path / "refs"
    assert len(AC.write_references(refs, out)) == 1
    assert AC.write_references(refs, out) == []  # stesso contenuto: niente duplicati
    changed = [dict(refs[0], verified_by="altro")]
    w = AC.write_references(changed, out)
    assert len(w) == 1 and w[0].endswith(".json") and "__" in Path(w[0]).stem.split("_B9_blend4")[1]
    assert json.loads((out / "uno__due__A40_B9_blend4.json").read_text())["verified_by"] == "utente"  # mai sovrascritto


def test_runner_hooks_pin_forbid_prefer_and_reuse(tmp_path):
    corr = {"runner": {"pins": [{"index": 1, "a": "uno", "b": "due", "a_out_bar": 40, "b_in_bar": 9, "type_key": "blend8",
                                 "b_lead_in_samples": 12000, "status": "corretta"},
                                {"index": 2, "a": "due", "b": "tre", "a_out_bar": 16, "b_in_bar": 4, "type_key": "cut",
                                 "status": "approvata"}],
                       "forbid_pairs": [["tre", "uno"]], "rerender": [1], "reuse": {"transitions": []}}}
    cands = [{"plan": {"a_out_bar": 40, "b_in_bar": 8}, "admissible": True, "notes": [], "decision": {"type_key": "blend4"}},
             {"plan": {"a_out_bar": 40, "b_in_bar": 9}, "admissible": True, "notes": [], "decision": {"type_key": "blend4"}}]
    pc = AC.pin_pair({"candidates": cands, "compatibility": {}}, {"slug": "uno"}, {"slug": "due"}, corr)
    assert [c["admissible"] for c in pc["candidates"]] == [True, False] and pc["candidates"][0]["plan"]["b_in_bar"] == 9
    assert pc["candidates"][0]["plan"]["b_lead_in"]["lead_in_samples"] == 12000 and pc["corrections"]["status"] == "fissata"
    pc2 = AC.pin_pair({"candidates": [dict(cands[0], notes=[])], "compatibility": {}}, {"slug": "tre"}, {"slug": "uno"}, corr)
    assert not any(c["admissible"] for c in pc2["candidates"]) and pc2["corrections"]["status"] == "vietata"
    att = [(0, {"type_key": "blend4"}, "x"), (0, {"type_key": "blend8"}, "y"), (1, {"type_key": "echo_out"}, "z")]
    assert [x[1]["type_key"] for x in AC.prefer_technique(att, "uno", "due", corr)] == ["blend8", "blend4", "echo_out"]
    assert AC.prefer_technique(att, "x", "y", corr) == att
    assert AC.reuse_render(corr, "uno", "due", {"a_out_bar": 40, "b_in_bar": 9}, {"type_key": "blend8"}) is None  # corretta
    assert AC.reuse_render(corr, "due", "tre", {"a_out_bar": 16, "b_in_bar": 4}, {"type_key": "cut"}) is None  # niente file


def test_reuse_render_loads_frozen_transition(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    wav = tmp_path / "t02_accepted.wav"
    sf.write(str(wav), np.zeros((4800, 2), dtype="float32"), RATE, subtype="PCM_24")
    (tmp_path / "t02_accepted.json").write_text(json.dumps({"kind": "cut", "switch": {"output_sample": 2400}}))
    (tmp_path / "t02_01_cut.json").write_text(json.dumps({"qa": {"verdict": "pass", "reasons": []}}))
    corr = {"runner": {"pins": [{"a": "due", "b": "tre", "a_out_bar": 16, "b_in_bar": 4, "status": "approvata"}],
                       "reuse": {"transitions": [{"index": 2, "a": "due", "b": "tre", "a_out_bar": 16, "b_in_bar": 4,
                                                  "type_key": "cut", "wav": str(wav), "wav_sha256": AC.sha256_file(wav),
                                                  "render_json": str(tmp_path / "t02_accepted.json"),
                                                  "qa_json": str(tmp_path / "t02_01_cut.json")}]}}}
    res, qa = AC.reuse_render(corr, "due", "tre", {"a_out_bar": 16, "b_in_bar": 4}, {"type_key": "cut"})
    assert res.audio.shape == (4800, 2) and res.metadata["reused_from"]["index"] == 2 and qa["verdict"] == "pass"
    assert AC.reuse_render(corr, "due", "tre", {"a_out_bar": 16, "b_in_bar": 4}, {"type_key": "blend4"}) is None


FROZEN = ROOT / "runs" / "dj_set_v3_bridge" / "dj_plan" / "tracks"


@pytest.mark.data("runs/dj_set_v3_bridge/dj_plan/tracks/goosebumps.json")
@pytest.mark.skipif(not (FROZEN / "goosebumps.json").is_file(), reason="analisi del set v3 assenti")
def test_pin_pair_builds_missing_candidate_on_real_analyses():
    from musicmix import dj_set as ds
    A = json.loads((FROZEN / "goosebumps.json").read_text())
    B = json.loads((FROZEN / "5_percent_tint.json").read_text())
    pc = ds.candidates_for_pair(A, B)
    have = {(c["plan"]["a_out_bar"], c["plan"]["b_in_bar"]) for c in pc["candidates"]}
    a_bar, b_bar = next((x, y) for x in range(49, 60) for y in range(1, 9) if (x, y) not in have)
    corr = {"runner": {"pins": [{"a": "goosebumps", "b": "5_percent_tint", "a_out_bar": a_bar, "b_in_bar": b_bar,
                                 "type_key": "blend4", "status": "corretta"}]}}
    n = len(pc["candidates"])
    AC.pin_pair(pc, A, B, corr)
    assert len(pc["candidates"]) == n + 1 and pc["corrections"]["constructed"]
    keep = [c for c in pc["candidates"] if c["admissible"]]
    assert len(keep) == 1 and (keep[0]["plan"]["a_out_bar"], keep[0]["plan"]["b_in_bar"]) == (a_bar, b_bar)
    assert keep[0]["decision"]["type_key"] and keep[0]["cost"] is not None


# ------------------------------------------------------------------ metrica di accordo
def test_decision_agreement_counts_and_bootstrap(corr_case, tmp_path):
    set_path, an = corr_case
    run = "synthetic_corr"
    jd = tmp_path / "j"
    jd.mkdir()
    (jd / "1.json").write_text(json.dumps(_jdoc(run, 1, "uno", "due", "da_correggere", {"b_entry": "+2", "a_exit": "ok"})))
    (jd / "2.json").write_text(json.dumps(_jdoc(run, 2, "due", "tre", "approvata")))
    res = DA.agreement([(set_path, [jd])], n_boot=300)
    s = res["summary"]
    assert s["n_transitions"] == 2 and s["order"]["rate"] == 1.0 and s["exit"]["approved"] == 2
    assert s["exit"]["explicit_approvals"] == 2  # "ok" esplicito + verdetto approvata
    assert s["entry"]["approved"] == 1 and s["entry"]["corrected_shift_bars"]["values"] == [2]
    assert s["entry"]["corrected_shift_bars"]["later"] == 1 and s["all"]["approved"] == 1
    assert s["entry"]["ci95"] is not None and 0.0 <= s["entry"]["ci95"][0] <= s["entry"]["ci95"][1] <= 1.0
    assert "warning" in s and "Accordo sulle decisioni" in DA.report(res)
    again = DA.agreement([(set_path, [jd])], n_boot=300)
    assert again["summary"]["entry"]["ci95"] == s["entry"]["ci95"]  # seme fisso
    (jd / "2.json").write_text(json.dumps(_jdoc(run, 2, "due", "tre", "da_rifare_altro_brano")))
    s2 = DA.agreement([(set_path, [jd])], n_boot=200)["summary"]
    assert s2["order"]["approved"] == 1 and s2["exit"]["n"] == 1  # la transizione da rifare conta solo per l'ordine


def test_prefer_technique_adds_requested_tree_technique():
    corr = {"runner": {"pins": [{"a": "uno", "b": "due", "a_out_bar": 40, "b_in_bar": 9, "type_key": "blend8"},
                                {"a": "due", "b": "tre", "a_out_bar": 16, "b_in_bar": 4, "type_key": "halving_build"}]}}
    att = [(3, {"type_key": "echo_out", "type": "echo_out"}, "Viterbi")]
    out = AC.prefer_technique(att, "uno", "due", corr)
    assert out[0][0] == 3 and out[0][1]["type_key"] == "blend8" and out[0][1]["type"] == "blend" and out[0][1]["bars"] == 8
    assert out[0][1]["rule"] == "correzione_utente" and out[1:] == att
    assert AC.prefer_technique(att, "due", "tre", corr) == att  # tecnica creativa non proposta: non costruibile qui
    assert AC.tree_decision("cut")["type_key"] == "cut" and AC.tree_decision("loop_extend") is None


def test_order_lock_keeps_only_consecutive_frozen_pairs():
    corr = {"runner": {"pins": [], "forbid_pairs": [], "order_lock": ["uno", "due", "tre"]}}
    mk = lambda: {"candidates": [{"plan": {"a_out_bar": 1, "b_in_bar": 2}, "admissible": True, "notes": []}], "compatibility": {}}  # noqa: E731
    assert AC.pin_pair(mk(), {"slug": "uno"}, {"slug": "due"}, corr)["candidates"][0]["admissible"]
    off = AC.pin_pair(mk(), {"slug": "due"}, {"slug": "uno"}, corr)
    assert not off["candidates"][0]["admissible"] and off["corrections"]["status"] == "fuori_ordine"


def test_pickup_is_not_forced_when_dj_phrase_sees_mid_line(monkeypatch):
    from musicmix import dj_phrase as P
    monkeypatch.setattr(P, "b_lead_in", lambda m, t: {"status": "mid_line", "lead_in_samples": 10 ** 6, "reasons": ["metà frase"]})
    monkeypatch.setattr(P, "entry_point", lambda m, t: {"status": "mid_line", "pickup_samples": 0, "reasons": []})
    n, why = AC.pickup_samples({"words": []}, {"tempo": {"period_s": 0.5}}, 10.0)
    assert n == 0 and "mid_line" in why
    monkeypatch.setattr(P, "b_lead_in", lambda m, t: {"status": "lead_in", "lead_in_samples": 9000, "reasons": ["ok"]})
    assert AC.pickup_samples({"words": []}, {"tempo": {"period_s": 0.5}}, 10.0)[0] == 9000
    assert AC.pickup_samples(None, {"tempo": {"period_s": 0.5}}, 10.0)[0] == 24000


def test_unchanged_pin_puts_frozen_decision_first():
    fz = {"type": "build_into_chorus", "label": "build", "rule": "C2_chorus", "duration_bars": 2,
          "policy_books_v3": {"proposals": [{"technique": "build_into_chorus", "params": {"build_bars": 2}}]}}
    corr = {"runner": {"pins": [{"a": "x", "b": "y", "a_out_bar": 60, "b_in_bar": 25, "type_key": "build_into_chorus",
                                 "status": "senza_giudizio"}],
                       "reuse": {"transitions": [{"a": "x", "b": "y", "a_out_bar": 60, "b_in_bar": 25,
                                                  "type_key": "build_into_chorus", "decision": AC.frozen_decision(fz)}]}}}
    att = [(5, {"type_key": "echo_out"}, "Viterbi")]
    out = AC.prefer_technique(att, "x", "y", corr)
    assert out[0][0] == 5 and out[0][1]["type_key"] == "build_into_chorus" and out[0][1]["creative"] and out[1:] == att
    assert out[0][1]["render"]["params"] == {"build_bars": 2}
    assert AC.frozen_decision({"type": "blend4"})["type"] == "blend" and AC.frozen_decision({"type": "blend4"})["bars"] == 4
    assert "entry_bars" in AC.frozen_decision({"type": "echo_out"})["render"]
