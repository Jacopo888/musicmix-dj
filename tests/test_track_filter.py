"""Ammissione dei brani: lista di esclusione e lingua cantata (musicmix.track_filter)."""

from __future__ import annotations

import json

from musicmix import track_filter as TF
from musicmix.vocal_boundaries import choose_asr_language

INDEX = {
    "kizzy": {"language": "it", "probability": 0.84, "source": "idx"},
    "drake": {"language": "en", "probability": 0.97, "source": "idx"},
    "rap_fr": {"language": "fr", "probability": 0.95, "source": "idx"},
    "unsure_de": {"language": "de", "probability": 0.40, "source": "idx"},
    "beat": {"language": None, "probability": None, "source": "idx: stem vocale", "speech_seconds": 0.0},
    "mix_silent": {"language": None, "probability": None, "source": "idx: mix canonico", "speech_seconds": 0.0},
}


def make(**kw):
    return TF.TrackFilter(allowed_languages=kw.pop("allowed", ("en", "it")), min_probability=0.5,
                          excluded=kw.pop("excluded", {"times_real_scelta_j1393805": "utente"}), language_index=INDEX)


def test_language_filter_keeps_en_it_and_instrumentals():
    tf = make()
    kept, dropped = tf.split(["kizzy", "drake", "rap_fr", "unsure_de", "beat", "mix_silent", "nuovo"])
    assert kept == ["kizzy", "drake", "unsure_de", "beat", "mix_silent", "nuovo"]
    assert [d["slug"] for d in dropped] == ["rap_fr"]
    assert dropped[0]["rule"] == "language_not_allowed" and "fr" in dropped[0]["why"] and dropped[0]["probability"] == 0.95
    assert tf.check("beat")["rule"] == "no_voice"
    assert tf.check("mix_silent")["rule"] == "language_unknown"  # VAD sul mix: non basta per dire "senza voce"
    assert tf.check("unsure_de")["rule"] == "language_uncertain"
    assert make(allowed=("en",)).check("kizzy")["admitted"] is False  # configurabile


def test_catalog_language_field_and_vocals_selection_take_precedence():
    tf = make()
    assert tf.check("rap_fr", entry={"language": "it"})["admitted"]
    voc = {"language_selection": {"source": "detected", "language": "es",
                                  "detection": {"language": "es", "probability": 0.9, "n_windows": 3}}}
    r = tf.check("drake", vocals_doc=voc)
    assert not r["admitted"] and r["language"] == "es"
    legacy = {"backend": {"language": "en", "detected_language": "en", "language_probability": 1}}  # small.en forzato
    assert TF.language_from_vocals(legacy) is None


def test_exclusion_list_wins_and_file_is_read(tmp_path):
    tf = make(excluded={"drake": "utente"})
    r = tf.check("drake")
    assert not r["admitted"] and r["rule"] == "excluded_list"
    p = tmp_path / "ex.json"
    p.write_text(json.dumps({"tracks": {"a": {"reason": "x"}, "b": "y"}}))
    assert TF.load_excluded(p) == {"a": "x", "b": "y"}
    real = TF.load_excluded()
    assert {"times_real_scelta_j1393805", "porrazzo_glide_beat_j1398420"} <= set(real)


def test_filter_sources_logs_dropped():
    tf = make()
    cat = {"tracks": [{"slug": "rap_fr"}, {"slug": "drake"}, {"slug": "times_real_scelta_j1393805"}]}
    src = {s["slug"]: {"catalog": cat} for s in cat["tracks"]}
    log: list = []
    kept = TF.filter_sources(src, log, track_filter=tf)
    assert list(kept) == ["drake"]
    dropped = log[0]["track_filter"]["dropped"]
    assert {d["slug"]: d["rule"] for d in dropped} == {"rap_fr": "language_not_allowed",
                                                         "times_real_scelta_j1393805": "excluded_list"}


def test_choose_asr_language():
    assert choose_asr_language("auto", {"language": "it", "probability": 0.84})["language"] == "it"
    assert choose_asr_language(None, {"language": "it", "probability": 0.3})["source"] == "fallback_en"
    assert choose_asr_language("auto", {"language": None, "error": "x"})["language"] == "en"
    assert choose_asr_language("auto", {"language": "it", "probability": 0.9}, multilingual_available=False)["language"] == "en"
    r = choose_asr_language("it", None)
    assert r["language"] == "it" and r["source"] == "requested" and r["multilingual"]
    assert choose_asr_language("auto", {"language": "en", "probability": 0.95})["multilingual"] is False


def test_bridge_catalog_languages_recorded():
    """Indice reale: 18k_no_kizzy rilevato italiano; nessun brano del ponte escluso per lingua."""
    tf = TF.TrackFilter()
    r = tf.check("18k_no_kizzy")
    if r["language"] is None:  # indice non presente su questa macchina
        return
    assert r["language"] == "it" and r["admitted"]
