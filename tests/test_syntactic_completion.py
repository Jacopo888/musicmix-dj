"""Tests against the actual pretrained English spaCy dependency parser."""

from __future__ import annotations

import json
import os

import pytest

from musicmix.syntactic_completion import (
    CORE_CATEGORIES,
    MODEL_BYTES,
    MODEL_LICENSE,
    MODEL_NAME,
    MODEL_SHA256,
    MODEL_VERSION,
    SPACY_VERSION_PIN,
    SyntacticCompletionAnalyzer,
)

try:
    import spacy  # noqa: F401
except ImportError as exc:
    if not os.environ.get("MUSICMIX_REQUIRE_RUNTIMES"):
        pytest.skip("spaCy runtime not installed (set MUSICMIX_REQUIRE_RUNTIMES=1 to make this an error)",
                    allow_module_level=True)
    raise RuntimeError(
        "These tests require the isolated runtime: "
        "python -m venv runtime_syntax --system-site-packages; "
        "runtime_syntax\\Scripts\\python.exe -m pip install spacy==3.8.16 "
        "en_core_web_sm==3.8.0"
    ) from exc


SAMPLE_RATE = 48_000
SENTENCE_A = ["She", "doesn't", "finish", "the", "line."]
SENTENCE_B = ["When", "the", "beat", "drops", ",", "the", "crowd", "goes", "wild", "."]
SENTENCE_C = ["The", "rapper", "drops", "the", "beat."]


def _make_words(tokens, *, with_confidence=True, sample_rate=SAMPLE_RATE):
    words = []
    cursor = 0
    gap = int(round(sample_rate * 0.07))
    for index, token in enumerate(tokens):
        punctuation_only = all(character in ".,!?;:" for character in token)
        duration = int(round(sample_rate * (0.12 if punctuation_only else 0.38)))
        word = {
            "word": token,
            "start_sample": cursor,
            "end_sample": cursor + duration,
        }
        if with_confidence:
            word["confidence"] = 0.93 + (index % 3) * 0.01
        words.append(word)
        cursor += duration + gap
    duration_samples = cursor + sample_rate
    return words, duration_samples


@pytest.fixture(scope="session")
def analyzer():
    return SyntacticCompletionAnalyzer()


def _prepare(analyzer, tokens, **kwargs):
    words, duration = _make_words(tokens, **kwargs)
    prepared = analyzer.prepare(words, SAMPLE_RATE, duration)
    return words, duration, prepared


def _token(prepared, text):
    return next(token for token in prepared.doc if token.text == text)


def _span(prepared, text):
    return prepared.token_spans[_token(prepared, text).i]


def test_pinned_pretrained_parser_is_actually_loaded(analyzer):
    meta = analyzer.model_meta
    assert meta["name"] in {MODEL_NAME, "core_web_sm"}
    assert meta["distribution_name"] == MODEL_NAME
    assert meta["version"] == MODEL_VERSION
    assert meta["license"] == MODEL_LICENSE
    assert meta["spacy_version"] == SPACY_VERSION_PIN
    assert meta["has_parser"] is True
    assert meta["matches_pinned_model"] is True
    assert len(MODEL_SHA256) == 64
    assert MODEL_BYTES == 12_806_118
    assert "parser" in analyzer.nlp.pipe_names
    assert "ner" not in analyzer.nlp.pipe_names

    doc = analyzer.nlp("She doesn't finish the line.")
    assert doc.has_annotation("DEP")
    assert doc.has_annotation("SENT_START")
    assert len(doc) > 0
    assert all(token.dep_ for token in doc)
    assert any(token.head.i != token.i for token in doc)


def test_contraction_tokens_map_to_one_asr_word(analyzer):
    words, _, prepared = _prepare(analyzer, SENTENCE_A)
    word_index = next(
        index for index, word in enumerate(words) if word["word"] == "doesn't"
    )
    do_span = _span(prepared, "does")
    ntt_span = _span(prepared, "n't")
    assert do_span.mapped and ntt_span.mapped
    assert do_span.word_indices == (word_index,)
    assert ntt_span.word_indices == (word_index,)
    assert words[word_index]["start_sample"] <= do_span.start_sample
    assert do_span.end_sample <= ntt_span.start_sample
    assert ntt_span.end_sample <= words[word_index]["end_sample"]
    assert prepared.mapping_stats["tokens_with_shared_word"] >= 1
    assert do_span.interpolated and ntt_span.interpolated


def test_split_punctuation_maps_to_the_right_asr_words(analyzer):
    words_b, _, prepared_b = _prepare(analyzer, SENTENCE_B)
    comma_index = next(
        index for index, word in enumerate(words_b) if word["word"] == ","
    )
    comma_span = _span(prepared_b, ",")
    assert comma_span.word_indices == (comma_index,)
    assert comma_span.start_sample == words_b[comma_index]["start_sample"]
    assert comma_span.end_sample == words_b[comma_index]["end_sample"]

    words_c, _, prepared_c = _prepare(analyzer, SENTENCE_C)
    attached_index = next(
        index for index, word in enumerate(words_c) if word["word"] == "beat."
    )
    period_span = next(
        prepared_c.token_spans[token.i]
        for token in prepared_c.doc
        if token.text == "."
    )
    assert period_span.word_indices == (attached_index,)
    assert period_span.interpolated is True
    assert period_span.start_sample > words_c[attached_index]["start_sample"]
    assert period_span.end_sample == words_c[attached_index]["end_sample"]


def test_cue_before_verb_reports_subject_aux_neg_crossings(analyzer):
    words, duration, _ = _prepare(analyzer, SENTENCE_A)
    cue = words[2]["start_sample"]
    result = analyzer.analyze(words, cue, SAMPLE_RATE, duration)
    evidence = result["dependency_evidence"]
    assert result["coverage"]["status"] == "word_boundary"
    assert set(evidence["core_arc_categories_crossing_cut"]) == {
        "subject",
        "aux",
        "neg",
    }
    assert result["syntax_status"] == "open_core_dependency_arc"
    assert result["syntactic_end_candidate"] is False
    assert result["certified_musical_closure"] is False
    assert result["sentence_end_evidence"]["sentence_end_after_cue"] is True
    assert result["sentence_end_evidence"]["sentence_end_before_or_at_cue"] is False
    assert set(CORE_CATEGORIES) >= {"subject", "object", "aux", "neg", "complement"}


def test_comma_is_punctuation_not_sentence_closure(analyzer):
    words, duration, _ = _prepare(analyzer, SENTENCE_B)
    comma_index = next(
        index for index, word in enumerate(words) if word["word"] == ","
    )
    cue = words[comma_index]["start_sample"]
    result = analyzer.analyze(words, cue, SAMPLE_RATE, duration)
    punctuation = result["punctuation"]
    sentence = result["sentence_end_evidence"]
    clause = result["clause_end_evidence"]

    assert punctuation["punctuation_at_cue"] is True
    assert punctuation["punctuation_token"] == ","
    assert punctuation["terminal_punctuation_at_cue"] is False
    assert punctuation["punctuation_alone_is_not_closure"] is True
    assert punctuation["punctuation_only_signal"] is True
    assert punctuation["non_punctuation_end_evidence"] == []
    assert punctuation["punctuation_at_cue_but_sentence_continues"] is True
    assert sentence["sentence_end_before_or_at_cue"] is False
    assert sentence["sentence_end_after_cue"] is True
    assert result["syntactic_end_candidate"] is False
    assert result["syntax_status"] == "open_sentence_continues"
    assert "advcl" in clause["clause_arc_categories_crossing_cut"]
    assert any(
        item["dependent_text"] == "drops"
        for item in clause["clause_ends_before_cue"]
    )


def test_object_and_complement_arcs_cross_the_cut(analyzer):
    words_c, duration_c, _ = _prepare(analyzer, SENTENCE_C)
    cue_object = words_c[4]["start_sample"]
    object_result = analyzer.analyze(words_c, cue_object, SAMPLE_RATE, duration_c)
    assert "object" in object_result["dependency_evidence"][
        "core_arc_categories_crossing_cut"
    ]
    assert object_result["syntax_status"] == "open_core_dependency_arc"

    words_b, duration_b, _ = _prepare(analyzer, SENTENCE_B)
    cue_complement = words_b[8]["start_sample"]
    complement_result = analyzer.analyze(
        words_b, cue_complement, SAMPLE_RATE, duration_b
    )
    assert "complement" in complement_result["dependency_evidence"][
        "core_arc_categories_crossing_cut"
    ]
    assert complement_result["syntax_status"] == "open_core_dependency_arc"


def test_sentence_end_candidate_is_not_musical_closure(analyzer):
    words, duration, _ = _prepare(analyzer, SENTENCE_C)
    cue = words[-1]["end_sample"]
    result = analyzer.analyze(words, cue, SAMPLE_RATE, duration)
    sentence = result["sentence_end_evidence"]
    punctuation = result["punctuation"]

    assert result["coverage"]["status"] == "word_boundary"
    assert sentence["sentence_end_before_or_at_cue"] is True
    assert sentence["sentence_end_after_cue"] is False
    assert sentence["sentence_boundary_evidence"]
    assert result["dependency_evidence"]["core_arcs_crossing_cut"] == []
    assert result["syntax_status"] == "sentence_end_candidate"
    assert result["syntactic_end_candidate"] is True
    assert result["certified_musical_closure"] is False
    assert "not certified musical closure" in result["closure_scope"]
    assert punctuation["terminal_punctuation_at_cue"] is True
    assert punctuation["punctuation_only_signal"] is False
    assert punctuation["non_punctuation_end_evidence"]


def test_cue_inside_word_is_unknown_and_touches_arcs(analyzer):
    words, duration, _ = _prepare(analyzer, SENTENCE_A)
    finish = words[2]
    cue = (finish["start_sample"] + finish["end_sample"]) // 2
    result = analyzer.analyze(words, cue, SAMPLE_RATE, duration)
    assert result["coverage"]["status"] == "inside_word"
    assert result["coverage"]["cue_inside_word"] is True
    assert result["unknown"]["is_unknown"] is True
    assert result["unknown"]["coverage_unknown"] is True
    assert "cue_inside_word" in result["unknown"]["reasons"]
    assert result["dependency_evidence"]["core_arcs_touching_cut"]
    assert result["syntax_status"] == "unknown"
    assert result["syntactic_end_candidate"] is False


def test_reuses_one_parse_for_many_cue_queries(analyzer):
    fresh = SyntacticCompletionAnalyzer(nlp=analyzer.nlp)
    words, duration = _make_words(SENTENCE_B)
    cues = [
        words[0]["start_sample"],
        words[4]["start_sample"],
        words[-1]["end_sample"],
    ]
    results = fresh.analyze_many(words, cues, SAMPLE_RATE, duration)
    assert len(results) == 3
    assert fresh.parse_count == 1
    assert results[1]["punctuation"]["punctuation_token"] == ","
    assert results[2]["syntax_status"] == "sentence_end_candidate"


def test_asr_confidence_reports_available_and_missing_values(analyzer):
    words_conf, duration_conf = _make_words(SENTENCE_A, with_confidence=True)
    result_conf = analyzer.analyze(
        words_conf, words_conf[2]["start_sample"], SAMPLE_RATE, duration_conf
    )
    confidence = result_conf["asr_confidence"]
    assert confidence["unknown"] is False
    assert confidence["local_mean"] is not None
    assert 0.0 <= confidence["local_mean"] <= 1.0
    assert confidence["words_with_confidence"] == confidence["local_words"]

    words_no, duration_no = _make_words(SENTENCE_A, with_confidence=False)
    result_no = analyzer.analyze(
        words_no, words_no[2]["start_sample"], SAMPLE_RATE, duration_no
    )
    confidence_no = result_no["asr_confidence"]
    assert confidence_no["unknown"] is True
    assert confidence_no["local_mean"] is None
    assert result_no["unknown"]["confidence_unknown"] is True
    assert "asr_confidence_missing" in result_no["unknown"]["reasons"]
    assert result_no["unknown"]["analysis_unknown"] is False


def test_result_is_json_serializable(analyzer):
    words, duration = _make_words(SENTENCE_B)
    result = analyzer.analyze(words, words[4]["start_sample"], SAMPLE_RATE, duration)
    encoded = json.dumps(result)
    decoded = json.loads(encoded)
    assert decoded["schema"] == "musicmix.syntactic-completion.1"
    assert decoded["certified_musical_closure"] is False
    assert decoded["transcript"]["model"]["model_sha256"] == MODEL_SHA256


def test_validation_rejects_bad_samples(analyzer):
    with pytest.raises(ValueError):
        analyzer.prepare(
            [{"word": "x", "start_sample": 0, "end_sample": 10}],
            0,
            100,
        )
    with pytest.raises(ValueError):
        analyzer.prepare(
            [{"word": "x", "start_sample": 10, "end_sample": 5}],
            SAMPLE_RATE,
            100,
        )
    with pytest.raises(ValueError):
        analyzer.prepare(
            [{"word": "x", "start_sample": 0}],
            SAMPLE_RATE,
            100,
        )
    with pytest.raises(ValueError):
        analyzer.analyze(
            [{"word": "x", "start_sample": 0, "end_sample": 10}],
            1.5,
            SAMPLE_RATE,
            100,
        )
