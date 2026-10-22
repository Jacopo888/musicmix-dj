"""Dependency-syntax evidence for arbitrary cues in ASR word timelines.

The canonical input is a whole ASR word sequence and one cue in samples.  A
pretrained English parser analyses the whole transcript once, then each cue
query reports which core dependency arcs cross the cut, what sentence or
clause boundaries exist, and how much of the relevant context is observed.

This is syntactic evidence only.  It is not a certificate of musical closure,
and punctuation alone is never treated as closure.  The motivation comes from
rap-syntax analyses in the supplied knowledge base, but the parser output is
not a validation of any particular musical transition policy.
"""

from __future__ import annotations

import bisect
import math
import os
import threading
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

for _thread_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")


SCHEMA = "musicmix.syntactic-completion.1"
VERSION = "syntactic-completion-v1"
CANONICAL_SAMPLE_RATE = 48_000

SPACY_VERSION_PIN = "3.8.16"
SPACY_SOURCE_URL = "https://github.com/explosion/spaCy/releases/tag/v3.8.16"
SPACY_LICENSE = "MIT"
MODEL_NAME = "en_core_web_sm"
MODEL_VERSION = "3.8.0"
MODEL_LICENSE = "MIT"
MODEL_URL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)
MODEL_SHA256 = "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
MODEL_BYTES = 12_806_118
MODEL_PIPELINE = ("tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner")

CORE_DEPENDENCIES = {
    "subject": frozenset({"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}),
    "object": frozenset({"obj", "dobj", "iobj", "pobj", "dative"}),
    "aux": frozenset({"aux", "auxpass"}),
    "neg": frozenset({"neg"}),
    "complement": frozenset(
        {"acomp", "ccomp", "xcomp", "pcomp", "oprd", "attr", "prt", "prep"}
    ),
}
CORE_CATEGORIES = tuple(CORE_DEPENDENCIES)
CORE_LABELS = frozenset(
    label for labels in CORE_DEPENDENCIES.values() for label in labels
)
CLAUSE_LABELS = frozenset(
    {"advcl", "ccomp", "xcomp", "acl", "relcl", "csubj", "csubjpass", "parataxis", "conj"}
)
TERMINAL_PUNCTUATION = frozenset({".", "!", "?", "…", "؟", "。", "！", "？"})
CONFIDENCE_KEYS = ("confidence", "asr_confidence", "score", "probability", "prob")
CLITIC_WORDS = frozenset(
    {
        "n't",
        "n’t",
        "'s",
        "’s",
        "'re",
        "’re",
        "'ve",
        "’ve",
        "'ll",
        "’ll",
        "'d",
        "’d",
        "'m",
        "’m",
    }
)
OPENING_PUNCTUATION = frozenset("([{‘“«¿¡")
DEFAULT_CONFIDENCE_WINDOW_SECONDS = 1.0


def _label_category(label: str) -> str | None:
    for category, labels in CORE_DEPENDENCIES.items():
        if label in labels:
            return category
    return None


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_sample(value: Any, *, name: str) -> int:
    number = _finite_number(value)
    if number is None or number != round(number):
        raise ValueError(f"{name} must be a finite integer sample")
    return int(number)


def _lookup(item: Any, names: Sequence[str]) -> Any:
    if isinstance(item, Mapping):
        for name in names:
            if name in item:
                return item[name]
        return None
    for name in names:
        if hasattr(item, name):
            return getattr(item, name)
    return None


def _extract_confidence(item: Any) -> tuple[float | None, str | None]:
    for key in CONFIDENCE_KEYS:
        value = _lookup(item, (key,))
        number = _finite_number(value)
        if number is not None and 0.0 <= number <= 1.0:
            return number, key
    return None, None


def _is_punctuation_word(text: str) -> bool:
    return bool(text) and all(
        unicodedata.category(char).startswith("P") or char in "$+<=>^`|~"
        for char in text
    )


def _is_clitic(text: str) -> bool:
    lowered = text.lower()
    return lowered in CLITIC_WORDS or text[:1] in {"'", "’"}


def _separator(previous: str, current: str) -> str:
    if not previous:
        return ""
    if _is_punctuation_word(current):
        if current[:1] in OPENING_PUNCTUATION:
            return " "
        return ""
    if _is_clitic(current):
        return ""
    if _is_punctuation_word(previous) and previous[:1] in OPENING_PUNCTUATION:
        return ""
    return " "


@dataclass
class _Word:
    index: int
    text: str
    start_sample: int
    end_sample: int
    confidence: float | None
    confidence_field: str | None
    original_index: int
    char_start: int = 0
    char_end: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "word": self.text,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "confidence": self.confidence,
            "confidence_field": self.confidence_field,
        }


@dataclass(frozen=True)
class _TokenSpan:
    token_index: int
    char_start: int
    char_end: int
    start_sample: int
    end_sample: int
    word_indices: tuple[int, ...]
    interpolated: bool
    mapped: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "token_index": self.token_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "word_indices": list(self.word_indices),
            "interpolated": self.interpolated,
            "mapped": self.mapped,
        }


@dataclass(frozen=True)
class _SentenceSpan:
    index: int
    token_start: int
    token_end: int
    text: str
    start_sample: int
    lexical_end_sample: int
    end_sample: int
    terminal_punctuation: str | None
    root_token_index: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "text": self.text,
            "start_sample": self.start_sample,
            "lexical_end_sample": self.lexical_end_sample,
            "end_sample": self.end_sample,
            "terminal_punctuation": self.terminal_punctuation,
            "root_token_index": self.root_token_index,
        }


def _normalize_words(
    words: Sequence[Mapping[str, Any] | Any],
    *,
    duration_samples: int,
) -> tuple[list[_Word], list[str]]:
    if isinstance(words, Mapping):
        if "words" not in words:
            raise ValueError("words mapping must contain a 'words' sequence")
        words = words["words"]
    if isinstance(words, (str, bytes)) or not isinstance(words, Sequence):
        raise ValueError("words must be a sequence of word records")

    normalized: list[_Word] = []
    warnings: list[str] = []
    for original_index, item in enumerate(words):
        text_value = _lookup(item, ("word", "text", "value"))
        start_value = _lookup(item, ("start_sample", "start"))
        end_value = _lookup(item, ("end_sample", "end"))
        if text_value is None and isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            if len(item) < 3:
                raise ValueError("word tuple must contain text, start_sample, end_sample")
            text_value, start_value, end_value = item[0], item[1], item[2]
        if text_value is None or start_value is None or end_value is None:
            raise ValueError(
                "each word needs text and start_sample/end_sample "
                f"(record {original_index})"
            )
        text = str(text_value).strip()
        start_sample = _int_sample(start_value, name=f"words[{original_index}].start_sample")
        end_sample = _int_sample(end_value, name=f"words[{original_index}].end_sample")
        if end_sample < start_sample:
            raise ValueError(f"words[{original_index}] ends before it starts")
        if end_sample == start_sample:
            warnings.append("zero_duration_word")
        if start_sample < 0:
            warnings.append("negative_word_start")
        if end_sample > duration_samples:
            warnings.append("word_after_duration")
        confidence, confidence_field = _extract_confidence(item)
        if not text:
            warnings.append("empty_word_text_ignored")
            continue
        normalized.append(
            _Word(
                index=0,
                text=text,
                start_sample=start_sample,
                end_sample=end_sample,
                confidence=confidence,
                confidence_field=confidence_field,
                original_index=original_index,
            )
        )

    normalized.sort(key=lambda word: (word.start_sample, word.end_sample, word.original_index))
    for index, word in enumerate(normalized):
        word.index = index
        if index and word.start_sample < normalized[index - 1].end_sample:
            warnings.append("overlapping_word_intervals")
            break
    return normalized, sorted(set(warnings))


def _build_text(words: Sequence[_Word]) -> str:
    pieces: list[str] = []
    position = 0
    previous: str | None = None
    for word in words:
        separator = "" if previous is None else _separator(previous, word.text)
        if separator:
            pieces.append(separator)
            position += len(separator)
        word.char_start = position
        pieces.append(word.text)
        position += len(word.text)
        word.char_end = position
        previous = word.text
    return "".join(pieces)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _interpolate_samples(
    *, token_start_char: int, token_end_char: int,
    char_start: int, char_end: int, start_sample: int, end_sample: int,
) -> tuple[int, int]:
    width = max(1, char_end - char_start)
    fraction_start = _clamp((token_start_char - char_start) / width)
    fraction_end = _clamp((token_end_char - char_start) / width)
    if end_sample <= start_sample:
        return start_sample, start_sample
    mapped_start = start_sample + int(round(fraction_start * (end_sample - start_sample)))
    mapped_end = start_sample + int(round(fraction_end * (end_sample - start_sample)))
    mapped_start = max(start_sample, min(end_sample, mapped_start))
    mapped_end = max(start_sample, min(end_sample, mapped_end))
    if mapped_end < mapped_start:
        mapped_end = mapped_start
    if mapped_end == mapped_start and end_sample > start_sample:
        if mapped_end < end_sample:
            mapped_end += 1
        elif mapped_start > start_sample:
            mapped_start -= 1
    return mapped_start, mapped_end


def _map_tokens(doc: Any, words: Sequence[_Word], text: str) -> tuple[list[_TokenSpan], dict[str, Any]]:
    char_ends = [word.char_end for word in words]
    spans: list[_TokenSpan] = []
    unmapped = 0
    interpolated = 0
    shared = 0
    multiword = 0
    word_token_counts: dict[int, int] = {}

    for token in doc:
        token_start = token.idx
        token_end = token.idx + len(token.text)
        first = bisect.bisect_right(char_ends, token_start)
        overlapping: list[int] = []
        cursor = first
        while cursor < len(words) and words[cursor].char_start < token_end:
            overlapping.append(cursor)
            cursor += 1
        if not overlapping:
            spans.append(
                _TokenSpan(
                    token_index=token.i,
                    char_start=token_start,
                    char_end=token_end,
                    start_sample=0,
                    end_sample=0,
                    word_indices=(),
                    interpolated=False,
                    mapped=False,
                )
            )
            unmapped += 1
            continue

        first_word = words[overlapping[0]]
        last_word = words[overlapping[-1]]
        if len(overlapping) == 1:
            mapped_start, mapped_end = _interpolate_samples(
                token_start_char=token_start,
                token_end_char=token_end,
                char_start=first_word.char_start,
                char_end=first_word.char_end,
                start_sample=first_word.start_sample,
                end_sample=first_word.end_sample,
            )
            is_interpolated = (
                token_start > first_word.char_start or token_end < first_word.char_end
            )
            if is_interpolated:
                shared += 1
        else:
            mapped_start, mapped_end = _interpolate_samples(
                token_start_char=token_start,
                token_end_char=token_end,
                char_start=first_word.char_start,
                char_end=last_word.char_end,
                start_sample=first_word.start_sample,
                end_sample=last_word.end_sample,
            )
            is_interpolated = True
            multiword += 1

        spans.append(
            _TokenSpan(
                token_index=token.i,
                char_start=token_start,
                char_end=token_end,
                start_sample=mapped_start,
                end_sample=mapped_end,
                word_indices=tuple(overlapping),
                interpolated=is_interpolated,
                mapped=True,
            )
        )
        if is_interpolated:
            interpolated += 1
        for word_index in overlapping:
            word_token_counts[word_index] = word_token_counts.get(word_index, 0) + 1

    stats = {
        "word_count": len(words),
        "token_count": len(doc),
        "tokens_mapped": len(doc) - unmapped,
        "tokens_unmapped": unmapped,
        "interpolated_tokens": interpolated,
        "tokens_with_shared_word": shared,
        "tokens_spanning_multiple_words": multiword,
        "max_tokens_per_word": max(word_token_counts.values(), default=0),
        "text_characters": len(text),
    }
    return spans, stats


def _build_sentences(doc: Any, token_spans: Sequence[_TokenSpan]) -> list[_SentenceSpan]:
    if not len(doc):
        return []
    if doc.has_annotation("SENT_START"):
        starts = [token.i for token in doc if bool(token.is_sent_start)]
    else:
        starts = []
    if not starts or starts[0] != 0:
        starts = [0] + [index for index in starts if index != 0]
    bounds = list(zip(starts, starts[1:] + [len(doc)]))
    sentences: list[_SentenceSpan] = []
    for sentence_index, (start, end) in enumerate(bounds):
        if end <= start:
            continue
        mapped_spans = [
            token_spans[index]
            for index in range(start, end)
            if token_spans[index].mapped
        ]
        if not mapped_spans:
            continue
        lexical = [
            token_spans[index]
            for index in range(start, end)
            if token_spans[index].mapped and not doc[index].is_punct
        ]
        start_sample = min(span.start_sample for span in mapped_spans)
        end_sample = max(span.end_sample for span in mapped_spans)
        lexical_end = max((span.end_sample for span in lexical), default=end_sample)
        last_token = doc[end - 1]
        terminal = (
            last_token.text
            if last_token.is_punct and last_token.text in TERMINAL_PUNCTUATION
            else None
        )
        root_index = next(
            (token.i for token in doc[start:end] if token.dep_ == "ROOT"),
            None,
        )
        sentences.append(
            _SentenceSpan(
                index=sentence_index,
                token_start=start,
                token_end=end,
                text=doc[start:end].text.strip(),
                start_sample=start_sample,
                lexical_end_sample=lexical_end,
                end_sample=end_sample,
                terminal_punctuation=terminal,
                root_token_index=root_index,
            )
        )
    return sentences


def _filtered_model_meta(nlp: Any) -> dict[str, Any]:
    meta = getattr(nlp, "meta", {}) or {}
    name = meta.get("name")
    version = meta.get("version")
    try:
        import spacy

        spacy_version = spacy.__version__
    except Exception:
        spacy_version = None
    accepted_names = {MODEL_NAME, "core_web_sm"}
    return {
        "name": name,
        "distribution_name": MODEL_NAME,
        "version": version,
        "lang": meta.get("lang"),
        "license": meta.get("license"),
        "spacy_version": spacy_version,
        "pipeline": list(getattr(nlp, "pipe_names", ())),
        "model_url": MODEL_URL,
        "model_sha256": MODEL_SHA256,
        "model_bytes": MODEL_BYTES,
        "pinned_model": MODEL_NAME,
        "pinned_model_version": MODEL_VERSION,
        "pinned_model_license": MODEL_LICENSE,
        "pinned_spacy_version": SPACY_VERSION_PIN,
        "spacy_source_url": SPACY_SOURCE_URL,
        "spacy_license": SPACY_LICENSE,
        "matches_pinned_model": name in accepted_names and version == MODEL_VERSION,
        "has_parser": "parser" in list(getattr(nlp, "pipe_names", ())),
    }


@dataclass(frozen=True)
class PreparedSyntax:
    words: tuple[_Word, ...]
    sample_rate: int
    duration_samples: int
    text: str
    doc: Any = field(repr=False, compare=False)
    token_spans: tuple[_TokenSpan, ...]
    sentences: tuple[_SentenceSpan, ...]
    mapping_stats: Mapping[str, Any]
    model_meta: Mapping[str, Any]
    warnings: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "version": VERSION,
            "sample_rate": self.sample_rate,
            "duration_samples": self.duration_samples,
            "text": self.text,
            "mapping": dict(self.mapping_stats),
            "model": dict(self.model_meta),
            "warnings": list(self.warnings),
            "has_dependency_annotation": bool(self.doc.has_annotation("DEP")),
            "has_sentence_annotation": bool(self.doc.has_annotation("SENT_START")),
            "sentence_count": len(self.sentences),
        }


class SyntacticCompletionAnalyzer:
    """Parse one ASR word sequence once and answer many cue queries."""

    def __init__(
        self,
        model: str = MODEL_NAME,
        *,
        nlp: Any | None = None,
        parse_cache_size: int = 16,
    ) -> None:
        self.model_name = model
        if nlp is None:
            try:
                import spacy
            except ImportError as exc:
                raise RuntimeError(
                    "spaCy is not importable. Install the pinned isolated runtime "
                    "with: python -m venv runtime_syntax --system-site-packages; "
                    "runtime_syntax\\Scripts\\python.exe -m pip install spacy==3.8.16 "
                    f"en_core_web_sm=={MODEL_VERSION}"
                ) from exc
            try:
                nlp = spacy.load(model, disable=["ner"])
            except OSError as exc:
                raise RuntimeError(
                    f"spaCy model {model!r} is not installed. Install the pinned "
                    f"wheel {MODEL_URL} (sha256 {MODEL_SHA256})."
                ) from exc
        if "parser" not in list(getattr(nlp, "pipe_names", ())):
            raise ValueError("the supplied spaCy pipeline has no dependency parser")
        self.nlp = nlp
        self.model_meta = _filtered_model_meta(nlp)
        self.parse_count = 0
        self._cache_size = max(1, int(parse_cache_size))
        self._cache: OrderedDict[Any, PreparedSyntax] = OrderedDict()
        self._lock = threading.Lock()

    def prepare(
        self,
        words: Sequence[Mapping[str, Any] | Any],
        sample_rate: int,
        duration_samples: int,
    ) -> PreparedSyntax:
        sample_rate = _int_sample(sample_rate, name="sample_rate")
        duration_samples = _int_sample(duration_samples, name="duration_samples")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if duration_samples <= 0:
            raise ValueError("duration_samples must be positive")

        normalized, warnings = _normalize_words(words, duration_samples=duration_samples)
        key = (
            sample_rate,
            duration_samples,
            tuple(
                (
                    word.text,
                    word.start_sample,
                    word.end_sample,
                    word.confidence,
                    word.confidence_field,
                )
                for word in normalized
            ),
        )
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached

        text = _build_text(normalized)
        doc = self.nlp(text, disable=["ner"]) if text else self.nlp("", disable=["ner"])
        token_spans, mapping_stats = _map_tokens(doc, normalized, text)
        sentences = _build_sentences(doc, token_spans)
        prepared = PreparedSyntax(
            words=tuple(normalized),
            sample_rate=sample_rate,
            duration_samples=duration_samples,
            text=text,
            doc=doc,
            token_spans=tuple(token_spans),
            sentences=tuple(sentences),
            mapping_stats=mapping_stats,
            model_meta=self.model_meta,
            warnings=tuple(warnings),
        )
        with self._lock:
            self.parse_count += 1
            self._cache[key] = prepared
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return prepared

    def query(self, prepared: PreparedSyntax, cue_sample: int) -> dict[str, Any]:
        cue = _int_sample(cue_sample, name="cue_sample")
        coverage = _coverage(prepared, cue)
        confidence = _asr_confidence(prepared, cue, coverage)
        dependency, clause = _dependency_evidence(prepared, cue)
        sentence, clause_end = _sentence_evidence(prepared, cue, dependency, clause)
        punctuation = _punctuation_evidence(prepared, cue, sentence, dependency)
        unknown = _unknown_evidence(
            prepared,
            cue,
            coverage,
            confidence,
            dependency,
        )
        syntax_status, open_reasons = _syntax_status(
            prepared,
            cue,
            coverage,
            dependency,
            sentence,
            unknown,
        )
        non_punctuation_evidence = []
        if (
            syntax_status == "sentence_end_candidate"
            and not dependency["core_arcs_crossing_cut"]
            and not dependency["core_arcs_touching_cut"]
        ):
            non_punctuation_evidence.append(
                "spacy_parser_sentence_boundary_without_crossing_core_dependency"
            )
        punctuation["non_punctuation_end_evidence"] = non_punctuation_evidence
        punctuation["punctuation_only_signal"] = bool(
            punctuation["punctuation_at_cue"] and not non_punctuation_evidence
        )
        punctuation["punctuation_at_cue_but_sentence_continues"] = bool(
            punctuation["punctuation_at_cue"] and sentence["sentence_end_after_cue"]
        )
        punctuation["terminal_punctuation_at_cue_is_final"] = bool(
            punctuation["terminal_punctuation_at_cue"]
            and sentence["sentence_end_before_or_at_cue"]
            and not sentence["sentence_end_after_cue"]
        )

        sentence["non_punctuation_end_evidence"] = list(non_punctuation_evidence)
        sentence["core_dependency_crosses_cut"] = bool(
            dependency["core_arcs_crossing_cut"]
        )
        if not punctuation["punctuation_at_cue"]:
            punctuation["punctuation_token"] = None

        return {
            "schema": SCHEMA,
            "version": VERSION,
            "cue_sample": cue,
            "sample_rate": prepared.sample_rate,
            "duration_samples": prepared.duration_samples,
            "syntax_status": syntax_status,
            "syntactic_end_candidate": syntax_status == "sentence_end_candidate",
            "certified_musical_closure": False,
            "closure_scope": (
                "dependency and sentence-boundary evidence only; "
                "not certified musical closure"
            ),
            "open_reasons": open_reasons,
            "coverage": coverage,
            "asr_confidence": confidence,
            "sentence_end_evidence": sentence,
            "clause_end_evidence": clause_end,
            "dependency_evidence": dependency,
            "punctuation": punctuation,
            "unknown": unknown,
            "transcript": prepared.summary(),
        }

    def analyze(
        self,
        words: Sequence[Mapping[str, Any] | Any],
        cue_sample: int,
        sample_rate: int,
        duration_samples: int,
    ) -> dict[str, Any]:
        return self.query(
            self.prepare(words, sample_rate, duration_samples),
            cue_sample,
        )

    def analyze_many(
        self,
        words: Sequence[Mapping[str, Any] | Any],
        cue_samples: Sequence[int],
        sample_rate: int,
        duration_samples: int,
    ) -> list[dict[str, Any]]:
        prepared = self.prepare(words, sample_rate, duration_samples)
        return [self.query(prepared, cue) for cue in cue_samples]


def _word_dict(word: _Word | None) -> dict[str, Any] | None:
    return None if word is None else word.as_dict()


def _intersects(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return max(start_a, start_b) < min(end_a, end_b)


def _merge_length(intervals: Sequence[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += max(0, current_end - current_start)
            current_start, current_end = start, end
    return total + max(0, current_end - current_start)


def _coverage(prepared: PreparedSyntax, cue: int) -> dict[str, Any]:
    words = prepared.words
    window_half = max(1, int(round(prepared.sample_rate * DEFAULT_CONFIDENCE_WINDOW_SECONDS)))
    window_start = max(0, cue - window_half)
    window_end = min(prepared.duration_samples, cue + window_half)
    interval_length = max(0, window_end - window_start)
    if not words:
        return {
            "status": "no_words",
            "transcript_start_sample": None,
            "transcript_end_sample": None,
            "cue_inside_transcript_range": False,
            "cue_inside_word": False,
            "cue_at_word_boundary": False,
            "active_word": None,
            "nearest_word_before": None,
            "nearest_word_after": None,
            "window": {
                "start_sample": window_start,
                "end_sample": window_end,
                "covered_samples": 0,
                "coverage_ratio": None,
            },
            "unknown": True,
        }

    first = words[0]
    last = words[-1]
    active = [word for word in words if word.start_sample < cue < word.end_sample]
    at_boundary = any(
        cue == word.start_sample or cue == word.end_sample for word in words
    )
    before = next((word for word in reversed(words) if word.end_sample <= cue), None)
    after = next((word for word in words if word.start_sample >= cue), None)
    if cue < first.start_sample:
        status = "before_transcript"
    elif cue > last.end_sample:
        status = "after_transcript"
    elif active:
        status = "inside_word"
    elif at_boundary:
        status = "word_boundary"
    else:
        status = "gap"
    clipped = [
        (max(word.start_sample, window_start), min(word.end_sample, window_end))
        for word in words
        if _intersects(word.start_sample, word.end_sample, window_start, window_end)
    ]
    covered = _merge_length(clipped)
    unknown = status in {"before_transcript", "after_transcript", "inside_word"}
    return {
        "status": status,
        "transcript_start_sample": first.start_sample,
        "transcript_end_sample": last.end_sample,
        "cue_inside_transcript_range": first.start_sample <= cue <= last.end_sample,
        "cue_inside_word": status == "inside_word",
        "cue_at_word_boundary": at_boundary,
        "active_word": _word_dict(active[0]) if active else None,
        "active_words": [_word_dict(word) for word in active],
        "nearest_word_before": _word_dict(before),
        "nearest_word_after": _word_dict(after),
        "window": {
            "start_sample": window_start,
            "end_sample": window_end,
            "covered_samples": covered,
            "coverage_ratio": (covered / interval_length) if interval_length else None,
        },
        "unknown": unknown,
    }


def _asr_confidence(
    prepared: PreparedSyntax,
    cue: int,
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    words = prepared.words
    window = coverage["window"]
    local = [
        word
        for word in words
        if _intersects(
            word.start_sample,
            word.end_sample,
            window["start_sample"],
            window["end_sample"],
        )
    ]
    if not local:
        nearest = coverage["nearest_word_before"] or coverage["nearest_word_after"]
        if nearest is not None:
            local = [word for word in words if word.index == nearest.get("index")]
    values = [word.confidence for word in local if word.confidence is not None]
    fields = sorted(
        {
            word.confidence_field
            for word in local
            if word.confidence_field is not None
        }
    )
    return {
        "field": fields[0] if len(fields) == 1 else None,
        "fields_present": fields,
        "local_words": len(local),
        "words_with_confidence": len(values),
        "coverage_ratio": (len(values) / len(local)) if local else None,
        "local_mean": (sum(values) / len(values)) if values else None,
        "local_min": min(values) if values else None,
        "local_max": max(values) if values else None,
        "window_start_sample": window["start_sample"],
        "window_end_sample": window["end_sample"],
        "unknown": not bool(values),
    }


def _side(span: _TokenSpan, cue: int) -> str:
    if not span.mapped:
        return "unknown"
    if span.end_sample <= cue:
        return "left"
    if span.start_sample >= cue:
        return "right"
    return "straddle"


def _arc_direction(head_side: str, dependent_side: str) -> str:
    if head_side == "left" and dependent_side == "right":
        return "head_left_dependent_right"
    if head_side == "right" and dependent_side == "left":
        return "head_right_dependent_left"
    if "straddle" in {head_side, dependent_side}:
        return "straddles_cut"
    return "same_side"


def _arc_dict(
    prepared: PreparedSyntax,
    token: Any,
    cue: int,
    *,
    category: str | None,
) -> dict[str, Any]:
    head_span = prepared.token_spans[token.head.i]
    dependent_span = prepared.token_spans[token.i]
    head_side = _side(head_span, cue)
    dependent_side = _side(dependent_span, cue)
    sides = {head_side, dependent_side}
    crosses = sides == {"left", "right"}
    touches = "straddle" in sides or "unknown" in sides
    return {
        "category": category,
        "dependency": token.dep_,
        "head_text": token.head.text,
        "dependent_text": token.text,
        "head_token_index": token.head.i,
        "dependent_token_index": token.i,
        "head_side": head_side,
        "dependent_side": dependent_side,
        "crosses_cut": crosses,
        "touches_cut": touches,
        "crossing_direction": _arc_direction(head_side, dependent_side),
        "head_interval": {
            "start_sample": head_span.start_sample,
            "end_sample": head_span.end_sample,
            "mapped": head_span.mapped,
        },
        "dependent_interval": {
            "start_sample": dependent_span.start_sample,
            "end_sample": dependent_span.end_sample,
            "mapped": dependent_span.mapped,
        },
        "mapping_unknown": not head_span.mapped or not dependent_span.mapped,
    }


def _dependency_evidence(
    prepared: PreparedSyntax,
    cue: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    core_crossing: list[dict[str, Any]] = []
    core_touching: list[dict[str, Any]] = []
    core_present: set[str] = set()
    clause_crossing: list[dict[str, Any]] = []
    clause_records: list[dict[str, Any]] = []
    unknown_arcs: list[dict[str, Any]] = []

    for token in prepared.doc:
        label = token.dep_
        category = _label_category(label)
        if category is not None and token.head.i != token.i:
            arc = _arc_dict(prepared, token, cue, category=category)
            core_present.add(category)
            if arc["crosses_cut"]:
                core_crossing.append(arc)
            elif arc["touches_cut"]:
                core_touching.append(arc)
            if arc["mapping_unknown"]:
                unknown_arcs.append(arc)

        if label in CLAUSE_LABELS and token.head.i != token.i:
            arc = _arc_dict(prepared, token, cue, category=None)
            if arc["crosses_cut"]:
                clause_crossing.append(arc)
            subtree_spans = [
                prepared.token_spans[subtree_token.i]
                for subtree_token in token.subtree
                if prepared.token_spans[subtree_token.i].mapped
            ]
            lexical_spans = [
                prepared.token_spans[subtree_token.i]
                for subtree_token in token.subtree
                if prepared.token_spans[subtree_token.i].mapped
                and not subtree_token.is_punct
            ]
            if subtree_spans:
                end_sample = max(span.end_sample for span in subtree_spans)
                lexical_end = max(
                    (span.end_sample for span in lexical_spans),
                    default=end_sample,
                )
                clause_records.append(
                    {
                        "dependency": label,
                        "head_text": token.head.text,
                        "dependent_text": token.text,
                        "dependent_token_index": token.i,
                        "head_token_index": token.head.i,
                        "start_sample": min(span.start_sample for span in subtree_spans),
                        "end_sample": end_sample,
                        "lexical_end_sample": lexical_end,
                        "head_side": arc["head_side"],
                        "dependent_side": arc["dependent_side"],
                        "crosses_cut": arc["crosses_cut"],
                    }
                )

    crossing_categories = sorted(
        {arc["category"] for arc in core_crossing if arc["category"] is not None}
    )
    touching_categories = sorted(
        {arc["category"] for arc in core_touching if arc["category"] is not None}
    )
    dependency = {
        "core_categories_expected": list(CORE_CATEGORIES),
        "core_categories_present_in_transcript": sorted(core_present),
        "core_arc_categories_crossing_cut": crossing_categories,
        "core_arc_categories_touching_cut": touching_categories,
        "core_arcs_crossing_cut": core_crossing,
        "core_arcs_touching_cut": core_touching,
        "unknown_mapping_arcs": unknown_arcs,
        "noncore_arcs_crossing_count": sum(
            1
            for token in prepared.doc
            if token.head.i != token.i
            and _label_category(token.dep_) is None
            and token.dep_ not in CLAUSE_LABELS
            and {
                _side(prepared.token_spans[token.head.i], cue),
                _side(prepared.token_spans[token.i], cue),
            }
            == {"left", "right"}
        ),
    }

    clause_ends_before = [
        {**record, "distance_samples": cue - record["end_sample"]}
        for record in clause_records
        if record["end_sample"] <= cue
    ]
    clause_ends_after = [
        {**record, "distance_samples": record["end_sample"] - cue}
        for record in clause_records
        if record["end_sample"] > cue
    ]
    clause_ends_before.sort(key=lambda item: item["end_sample"], reverse=True)
    clause_ends_after.sort(key=lambda item: item["end_sample"])
    clause = {
        "clause_arcs_crossing_cut": clause_crossing,
        "clause_arc_categories_crossing_cut": sorted(
            {arc["dependency"] for arc in clause_crossing}
        ),
        "clause_ends_before_cue": clause_ends_before,
        "clause_ends_after_cue": clause_ends_after,
        "nearest_clause_end_before_cue": clause_ends_before[0] if clause_ends_before else None,
        "nearest_clause_end_after_cue": clause_ends_after[0] if clause_ends_after else None,
        "clause_boundary_at_or_before_cue": bool(clause_ends_before),
        "clause_boundary_after_cue": bool(clause_ends_after),
    }
    return dependency, clause


def _sentence_at(prepared: PreparedSyntax, cue: int) -> _SentenceSpan | None:
    for sentence in prepared.sentences:
        if sentence.start_sample <= cue <= sentence.end_sample:
            return sentence
    return None


def _previous_sentence(prepared: PreparedSyntax, cue: int) -> _SentenceSpan | None:
    previous = None
    for sentence in prepared.sentences:
        if sentence.end_sample <= cue:
            previous = sentence
        else:
            break
    return previous


def _next_sentence(prepared: PreparedSyntax, cue: int) -> _SentenceSpan | None:
    for sentence in prepared.sentences:
        if sentence.start_sample >= cue:
            return sentence
    return None


def _sentence_root(prepared: PreparedSyntax, sentence: _SentenceSpan | None) -> dict[str, Any] | None:
    if sentence is None or sentence.root_token_index is None:
        return None
    token = prepared.doc[sentence.root_token_index]
    span = prepared.token_spans[token.i]
    return {
        "text": token.text,
        "dependency": token.dep_,
        "token_index": token.i,
        "start_sample": span.start_sample,
        "end_sample": span.end_sample,
    }


def _sentence_evidence(
    prepared: PreparedSyntax,
    cue: int,
    dependency: Mapping[str, Any],
    clause: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    current = _sentence_at(prepared, cue)
    previous = _previous_sentence(prepared, cue)
    following = _next_sentence(prepared, cue)
    if current is not None:
        relation = "inside_sentence"
        if cue >= current.end_sample:
            relation = "at_or_after_sentence_end"
        elif cue >= current.lexical_end_sample:
            relation = "between_lexical_end_and_terminal_punctuation"
        end_before = cue >= current.end_sample
        end_after = cue < current.end_sample
        distance = cue - current.end_sample
        evidence_sentence = current
    elif previous is not None and following is not None:
        relation = "between_sentences"
        end_before = True
        end_after = True
        distance = cue - previous.end_sample
        evidence_sentence = previous
    elif following is not None:
        relation = "before_first_sentence"
        end_before = False
        end_after = True
        distance = cue - following.end_sample
        evidence_sentence = following
    elif previous is not None:
        relation = "after_last_sentence"
        end_before = True
        end_after = False
        distance = cue - previous.end_sample
        evidence_sentence = previous
    else:
        relation = "no_sentence_evidence"
        end_before = False
        end_after = False
        distance = None
        evidence_sentence = None

    boundary_evidence: list[str] = []
    if end_before and prepared.doc.has_annotation("SENT_START"):
        boundary_evidence.append("spacy_parser_sentence_boundary")
    if evidence_sentence is not None and evidence_sentence.terminal_punctuation and end_before:
        boundary_evidence.append("terminal_punctuation_present")
    if relation == "between_sentences":
        boundary_evidence.append("between_sentences")

    sentence = {
        "cue_relation": relation,
        "sentence_index": evidence_sentence.index if evidence_sentence else None,
        "sentence_text": evidence_sentence.text if evidence_sentence else None,
        "sentence_start_sample": evidence_sentence.start_sample if evidence_sentence else None,
        "sentence_lexical_end_sample": (
            evidence_sentence.lexical_end_sample if evidence_sentence else None
        ),
        "sentence_end_sample": evidence_sentence.end_sample if evidence_sentence else None,
        "sentence_end_before_or_at_cue": end_before,
        "sentence_end_after_cue": end_after,
        "distance_to_sentence_end_samples": distance,
        "terminal_punctuation_present": bool(
            evidence_sentence and evidence_sentence.terminal_punctuation
        ),
        "terminal_punctuation": (
            evidence_sentence.terminal_punctuation if evidence_sentence else None
        ),
        "sentence_root": _sentence_root(prepared, evidence_sentence),
        "sentence_boundary_evidence": boundary_evidence,
        "previous_sentence_end_sample": previous.end_sample if previous else None,
        "next_sentence_start_sample": following.start_sample if following else None,
        "sentence_boundary_before_cue": end_before,
        "sentence_boundary_after_cue": end_after,
        "core_dependency_crosses_cut": False,
        "non_punctuation_end_evidence": [],
    }

    clause_end = {
        "cue_relation": relation,
        "nearest_clause_end_before_cue": clause["nearest_clause_end_before_cue"],
        "nearest_clause_end_after_cue": clause["nearest_clause_end_after_cue"],
        "clause_boundary_at_or_before_cue": clause["clause_boundary_at_or_before_cue"],
        "clause_boundary_after_cue": clause["clause_boundary_after_cue"],
        "clause_arc_categories_crossing_cut": clause[
            "clause_arc_categories_crossing_cut"
        ],
        "clause_arcs_crossing_cut": clause["clause_arcs_crossing_cut"],
        "clause_ends_before_cue": clause["clause_ends_before_cue"],
        "clause_ends_after_cue": clause["clause_ends_after_cue"],
        "core_dependency_crosses_cut": bool(dependency["core_arcs_crossing_cut"]),
    }
    return sentence, clause_end


def _punctuation_evidence(
    prepared: PreparedSyntax,
    cue: int,
    sentence: Mapping[str, Any],
    dependency: Mapping[str, Any],
) -> dict[str, Any]:
    punctuation_token = None
    punctuation_span = None
    for token in prepared.doc:
        if not token.is_punct:
            continue
        span = prepared.token_spans[token.i]
        if span.mapped and span.start_sample <= cue <= span.end_sample:
            punctuation_token = token
            punctuation_span = span
            break
    terminal = bool(
        punctuation_token is not None
        and punctuation_token.text in TERMINAL_PUNCTUATION
    )
    return {
        "punctuation_at_cue": punctuation_token is not None,
        "punctuation_token": punctuation_token.text if punctuation_token is not None else None,
        "punctuation_token_index": (
            punctuation_token.i if punctuation_token is not None else None
        ),
        "punctuation_interval": (
            {
                "start_sample": punctuation_span.start_sample,
                "end_sample": punctuation_span.end_sample,
            }
            if punctuation_span is not None
            else None
        ),
        "terminal_punctuation_at_cue": terminal,
        "punctuation_alone_is_not_closure": True,
        "punctuation_only_signal": False,
        "non_punctuation_end_evidence": [],
        "punctuation_at_cue_but_sentence_continues": False,
        "terminal_punctuation_at_cue_is_final": False,
        "punctuation_and_sentence_boundary_disagree": bool(
            punctuation_token is not None
            and not sentence["sentence_end_before_or_at_cue"]
            and not dependency["core_arcs_crossing_cut"]
        ),
    }


def _unknown_evidence(
    prepared: PreparedSyntax,
    cue: int,
    coverage: Mapping[str, Any],
    confidence: Mapping[str, Any],
    dependency: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    parse_unknown = not prepared.doc.has_annotation("DEP") or not prepared.words
    mapping_unknown = bool(prepared.mapping_stats.get("tokens_unmapped"))
    coverage_unknown = bool(coverage["unknown"])
    confidence_unknown = bool(confidence["unknown"])
    if not prepared.words:
        reasons.append("no_words")
    if not prepared.doc.has_annotation("DEP"):
        reasons.append("no_dependency_annotation")
    if coverage["status"] == "inside_word":
        reasons.append("cue_inside_word")
    if coverage["status"] == "before_transcript":
        reasons.append("cue_before_transcript")
    if coverage["status"] == "after_transcript":
        reasons.append("cue_after_transcript")
    if mapping_unknown:
        reasons.append("unmapped_tokens")
    if prepared.mapping_stats.get("tokens_with_shared_word"):
        reasons.append("char_span_time_interpolated")
    if "overlapping_word_intervals" in prepared.warnings:
        reasons.append("overlapping_word_intervals")
    if confidence_unknown:
        reasons.append("asr_confidence_missing")
    if cue < 0 or cue > prepared.duration_samples:
        reasons.append("cue_outside_duration")
    if dependency["unknown_mapping_arcs"]:
        reasons.append("dependency_arc_mapping_unknown")
    return {
        "is_unknown": bool(parse_unknown or coverage_unknown or mapping_unknown),
        "analysis_unknown": bool(parse_unknown or coverage_unknown or mapping_unknown),
        "parse_unknown": bool(parse_unknown),
        "coverage_unknown": bool(coverage_unknown),
        "confidence_unknown": bool(confidence_unknown),
        "mapping_unknown": bool(mapping_unknown),
        "any_unknown": bool(
            parse_unknown or coverage_unknown or mapping_unknown or confidence_unknown
        ),
        "reasons": sorted(set(reasons)),
    }


def _syntax_status(
    prepared: PreparedSyntax,
    cue: int,
    coverage: Mapping[str, Any],
    dependency: Mapping[str, Any],
    sentence: Mapping[str, Any],
    unknown: Mapping[str, Any],
) -> tuple[str, list[str]]:
    if unknown["analysis_unknown"]:
        return "unknown", list(unknown["reasons"])
    crossing = dependency["core_arcs_crossing_cut"]
    touching = dependency["core_arcs_touching_cut"]
    if crossing or touching:
        reasons = [
            f"core_dependency_crosses_cut:{arc['category']}"
            for arc in crossing
        ] + [
            f"core_dependency_touches_cut:{arc['category']}"
            for arc in touching
        ]
        return "open_core_dependency_arc", sorted(set(reasons))
    if sentence["sentence_end_before_or_at_cue"]:
        return "sentence_end_candidate", [
            "sentence_boundary_at_or_before_cue_without_crossing_core_dependency"
        ]
    if sentence["sentence_end_after_cue"]:
        return "open_sentence_continues", ["sentence_end_after_cue"]
    return "unknown", ["no_sentence_boundary_evidence"]


_DEFAULT_ANALYZER: SyntacticCompletionAnalyzer | None = None
_DEFAULT_ANALYZER_LOCK = threading.Lock()


def get_default_analyzer() -> SyntacticCompletionAnalyzer:
    global _DEFAULT_ANALYZER
    with _DEFAULT_ANALYZER_LOCK:
        if _DEFAULT_ANALYZER is None:
            _DEFAULT_ANALYZER = SyntacticCompletionAnalyzer()
        return _DEFAULT_ANALYZER


def analyze(
    words: Sequence[Mapping[str, Any] | Any],
    cue_sample: int,
    sample_rate: int,
    duration_samples: int,
    *,
    analyzer: SyntacticCompletionAnalyzer | None = None,
) -> dict[str, Any]:
    """Canonical cue query: words, cue_sample, sample_rate, duration_samples."""
    active = analyzer if analyzer is not None else get_default_analyzer()
    return active.analyze(words, cue_sample, sample_rate, duration_samples)


analyze_cue = analyze
analyze_syntax = analyze
