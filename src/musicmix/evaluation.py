"""Import and audit human A/B listening judgements.

The listening page deliberately stores only labels (A/B) in the exported
file.  The run's private ``listening_ab/key.json`` is therefore a required
second input: it is the only source used to associate a label with a system
and a candidate.  This module keeps that join explicit and records enough
hashes to make a later re-import auditable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


LISTENING_SCHEMA = "musicmix.listening.1"
LISTENING_SCHEMA_V2 = "musicmix.listening.2"
LISTENING_SCHEMAS = frozenset({LISTENING_SCHEMA, LISTENING_SCHEMA_V2})
EVALUATION_SCHEMA = "musicmix.evaluation.1"
ALLOWED_ANSWERS = frozenset({"A", "B", "tie", "neither"})
MOTION_LABELS = frozenset({"continues", "resolves", "interrupts", "unclear"})


class EvaluationError(ValueError):
    """Base error for invalid listening evidence or an unsafe key join."""


class SchemaValidationError(EvaluationError):
    """The judgement export or run key does not satisfy its schema."""


class AmbiguousMappingError(EvaluationError):
    """A selected run key cannot identify exactly one A/B pair."""


class MissingTrialError(EvaluationError):
    """The judgement export and selected run do not cover the same trials."""


class DuplicateJudgementError(EvaluationError):
    """The same trial has conflicting annotations."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise EvaluationError(f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"invalid JSON in {path}: {exc}") from exc


def _is_trial(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _require_string(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise SchemaValidationError(f"{field} must be a non-empty string")
    return value


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaValidationError(f"{field} must be a boolean")
    return value


def _judgement_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["trial"],
        row["pair_id"],
        row["answer"],
        row["word_cut_a"],
        row["word_cut_b"],
        row["comment"],
        # Keep optional motion fields in the duplicate/conflict identity.  A
        # legacy row with no motion annotation remains distinct from a V2 row
        # that explicitly records an unresolved annotation as null.
        ("motion_a" in row, row.get("motion_a")),
        ("motion_b" in row, row.get("motion_b")),
    )


def load_judgements(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate and deterministically de-duplicate one listening export.

    Exact repeated rows are harmless and collapse to one observation.
    Repeated ``trial``/``pair_id`` rows with different values are rejected;
    silently choosing one would make the result impossible to audit.
    """

    path = Path(path).expanduser().resolve()
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise SchemaValidationError("judgement export must be a JSON object")
    schema_version = payload.get("schema_version")
    if schema_version not in LISTENING_SCHEMAS:
        raise SchemaValidationError(
            f"expected schema_version in {sorted(LISTENING_SCHEMAS)!r}, got {schema_version!r}"
        )
    if schema_version == LISTENING_SCHEMA_V2:
        for field in ("test_id", "run_id", "title", "feedback_filename"):
            _require_string(payload.get(field), f"{field}")
    rows = payload.get("judgements")
    if not isinstance(rows, list):
        raise SchemaValidationError("judgements must be a list")

    by_identity: dict[tuple[int, str], tuple[Any, ...]] = {}
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise SchemaValidationError(f"judgements[{index}] must be an object")
        if not _is_trial(raw.get("trial")):
            raise SchemaValidationError(f"judgements[{index}].trial must be a positive integer")
        pair_id = _require_string(raw.get("pair_id"), f"judgements[{index}].pair_id")
        answer = raw.get("answer")
        if answer not in ALLOWED_ANSWERS:
            raise SchemaValidationError(
                f"judgements[{index}].answer must be one of {sorted(ALLOWED_ANSWERS)}"
            )
        word_cut_a = _require_bool(raw.get("word_cut_a"), f"judgements[{index}].word_cut_a")
        word_cut_b = _require_bool(raw.get("word_cut_b"), f"judgements[{index}].word_cut_b")
        comment = raw.get("comment", "")
        if not isinstance(comment, str):
            raise SchemaValidationError(f"judgements[{index}].comment must be a string")
        motion_present = "motion_a" in raw or "motion_b" in raw
        motion_values: dict[str, Any] = {}
        if motion_present:
            for field in ("motion_a", "motion_b"):
                value = raw.get(field)
                if value is not None and value not in MOTION_LABELS:
                    raise SchemaValidationError(
                        f"judgements[{index}].{field} must be null or one of {sorted(MOTION_LABELS)}"
                    )
                # Canonicalize a partially supplied optional pair to two
                # fields so duplicate exports compare predictably.
                motion_values[field] = value
        row = {
            "trial": raw["trial"],
            "pair_id": pair_id,
            "answer": answer,
            "word_cut_a": word_cut_a,
            "word_cut_b": word_cut_b,
            "comment": comment,
        }
        row.update(motion_values)
        identity = (row["trial"], row["pair_id"])
        signature = _judgement_signature(row)
        previous = by_identity.get(identity)
        if previous is not None:
            if previous != signature:
                raise DuplicateJudgementError(
                    f"conflicting judgements for trial={identity[0]} pair_id={identity[1]!r}"
                )
            continue
        by_identity[identity] = signature
        normalized.append(row)

    normalized.sort(key=lambda row: (row["trial"], row["pair_id"]))
    return payload, normalized


def _run_root_and_key(run_path: str | Path) -> tuple[Path, Path]:
    selected = Path(run_path).expanduser().resolve()
    if selected.is_file():
        if selected.name != "key.json":
            raise EvaluationError("--run must be a run directory or listening_ab/key.json")
        key_path = selected
        listening_dir = selected.parent
        run_root = listening_dir.parent if listening_dir.name == "listening_ab" else listening_dir
    else:
        if (selected / "listening_ab" / "key.json").is_file():
            run_root = selected
            key_path = selected / "listening_ab" / "key.json"
        elif (selected / "key.json").is_file():
            key_path = selected / "key.json"
            run_root = selected.parent if selected.name == "listening_ab" else selected
        else:
            raise EvaluationError(f"no listening_ab/key.json found below --run {selected}")
    return run_root, key_path


def load_run_key(run_path: str | Path) -> tuple[Path, Path, dict[str, Any], dict[tuple[int, str], dict[str, Any]]]:
    """Load a selected run and verify that every trial has exactly A and B."""

    run_root, key_path = _run_root_and_key(run_path)
    payload = _read_json(key_path)
    if not isinstance(payload, dict):
        raise SchemaValidationError("run key must be a JSON object")
    if not _is_trial(payload.get("seed")):
        raise SchemaValidationError("run key seed must be a positive integer")
    for field in ("test_id", "run_id", "title", "feedback_filename"):
        if field in payload:
            _require_string(payload[field], f"key.{field}")
    raw_trials = payload.get("trials")
    if not isinstance(raw_trials, list) or not raw_trials:
        raise SchemaValidationError("run key trials must be a non-empty list")

    by_label: dict[tuple[int, str], dict[str, Any]] = {}
    by_pair: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for index, raw in enumerate(raw_trials):
        if not isinstance(raw, dict) or not _is_trial(raw.get("trial")):
            raise SchemaValidationError(f"key.trials[{index}].trial must be a positive integer")
        label = raw.get("label")
        if label not in {"A", "B"}:
            raise SchemaValidationError(f"key.trials[{index}].label must be A or B")
        pair_id = _require_string(raw.get("pair_id"), f"key.trials[{index}].pair_id")
        identity = (raw["trial"], label)
        if identity in by_label:
            raise AmbiguousMappingError(f"duplicate key entry for trial={identity[0]} label={label}")
        entry = {
            "trial": raw["trial"],
            "label": label,
            "pair_id": pair_id,
            "system": _require_string(raw.get("system"), f"key.trials[{index}].system"),
            "source": _require_string(raw.get("source"), f"key.trials[{index}].source"),
        }
        by_label[identity] = entry
        by_pair.setdefault((entry["trial"], pair_id), {})[label] = entry

    pair_map: dict[tuple[int, str], dict[str, Any]] = {}
    for identity, labels in by_pair.items():
        if set(labels) != {"A", "B"}:
            raise AmbiguousMappingError(
                f"key pair trial={identity[0]} pair_id={identity[1]!r} does not contain exactly A and B"
            )
        pair_map[identity] = {"A": labels["A"], "B": labels["B"]}

    return run_root, key_path, payload, pair_map


def _manifest(run_root: Path) -> tuple[str | None, Path | None, str | None]:
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        return None, None, None
    payload = _read_json(manifest_path)
    if not isinstance(payload, dict):
        raise SchemaValidationError("run manifest must be a JSON object")
    run_id = payload.get("run_id")
    if run_id is not None:
        _require_string(run_id, "manifest.run_id")
    return run_id, manifest_path, sha256_file(manifest_path)


def _candidate_observation(
    row: dict[str, Any], pair: dict[str, dict[str, Any]], mapping_confirmed: bool
) -> dict[str, Any]:
    candidates: dict[str, dict[str, Any]] = {}
    for label, cut_key in (("A", "word_cut_a"), ("B", "word_cut_b")):
        entry = pair[label]
        candidates[label] = {
            "label": label,
            "system": entry["system"],
            "source": entry["source"],
            "pair_id": entry["pair_id"],
            "word_cut": row[cut_key],
        }
        motion_key = f"motion_{label.lower()}"
        if motion_key in row:
            candidates[label]["motion"] = row[motion_key]
    preferred = None
    if mapping_confirmed and row["answer"] in {"A", "B"}:
        preferred = dict(candidates[row["answer"]])
    return {
        "trial": row["trial"],
        "pair_id": row["pair_id"],
        "preference_label": row["answer"],
        "preferred_candidate": preferred,
        "candidates": candidates,
        "comment": row["comment"],
        "annotation": {
            "sufficient_for_single_observation": True,
            "basis": "valid_preference_and_word_cut_fields",
            "confidence": None,
        },
    }


def import_judgements(
    judgements_path: str | Path,
    run_path: str | Path,
    *,
    mapping_confirmed: bool = False,
) -> dict[str, Any]:
    """Join one export to one explicitly selected run key.

    ``mapping_confirmed=False`` is the safe default.  It keeps the preferred
    label while withholding a preferred system, which prevents accidentally
    presenting a label-only export as a system comparison.  Set it only when
    the human has confirmed which export belongs to the selected run.
    """

    input_path = Path(judgements_path).expanduser().resolve()
    payload, rows = load_judgements(input_path)
    run_root, key_path, key_payload, pair_map = load_run_key(run_path)
    run_id, manifest_path, manifest_hash = _manifest(run_root)

    if payload["schema_version"] == LISTENING_SCHEMA_V2:
        for field in ("test_id", "run_id"):
            expected = key_payload.get(field)
            if expected is None:
                raise SchemaValidationError(
                    f"V2 judgement export requires key.{field} for explicit identity checking"
                )
            if payload[field] != expected:
                raise SchemaValidationError(
                    f"V2 {field} mismatch between judgement export and selected run key"
                )
        if run_id is not None and payload["run_id"] != run_id:
            raise SchemaValidationError("V2 run_id mismatch between judgement export and manifest")

    expected = set(pair_map)
    actual = {(row["trial"], row["pair_id"]) for row in rows}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing trials/pairs: {missing}")
        if extra:
            details.append(f"unknown trials/pairs: {extra}")
        raise MissingTrialError("; ".join(details))

    observations = [
        _candidate_observation(row, pair_map[(row["trial"], row["pair_id"])], mapping_confirmed)
        for row in rows
    ]
    return {
        "schema_version": EVALUATION_SCHEMA,
        "source_schema_version": payload["schema_version"],
        "mapping": {
            "confirmed": bool(mapping_confirmed),
            "status": "confirmed_by_user" if mapping_confirmed else "pending_human_confirmation",
            "method": "explicit_judgements_and_run_key_arguments",
            "key_join": "trial+pair_id with exactly labels A and B",
        },
        "input": {
            "judgements_path": str(input_path),
            "judgements_sha256": sha256_file(input_path),
            "judgements_schema_version": payload["schema_version"],
        },
        "listening_identity": {
            "test_id": payload.get("test_id"),
            "run_id": payload.get("run_id"),
            "title": payload.get("title"),
            "feedback_filename": payload.get("feedback_filename"),
        },
        "run": {
            "root": str(run_root),
            "run_id": run_id,
            "seed": key_payload["seed"],
            "key_path": str(key_path),
            "key_sha256": sha256_file(key_path),
            "manifest_path": str(manifest_path) if manifest_path else None,
            "manifest_sha256": manifest_hash,
        },
        "observations": observations,
        "usage_policy": {
            "descriptive_only": True,
            "training_eligible": False,
            "reason": "single listening observations are not automatic training labels",
        },
        "summary": {
            "observation_count": len(observations),
            "scope": "descriptive evidence for these listening observations only",
            "confidence": None,
            "statistical_inference": None,
            "generalized_winner_claim": None,
        },
    }


def write_json(document: dict[str, Any], path: str | Path) -> Path:
    """Write normalized evidence with stable formatting and no NaN values."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


def render_report(document: dict[str, Any]) -> str:
    """Render a cautious Italian report for a normalized evidence document."""

    mapping = document["mapping"]
    run = document["run"]
    lines = [
        "# Valutazione umana delle transizioni",
        "",
        "Questo report riassume osservazioni A/B ascoltate per i soli trial indicati. "
        "Non è una stima di qualità generale né uno studio con potenza statistica.",
        "",
        f"- Run: `{run['run_id'] or 'non disponibile nel manifest'}`",
        f"- Seed: `{run['seed']}`",
        f"- Input giudizi SHA-256: `{document['input']['judgements_sha256']}`",
        f"- Key SHA-256: `{run['key_sha256']}`",
        f"- Associazione label/sistema: **{mapping['status']}**",
        f"- Osservazioni: `{document['summary']['observation_count']}`",
        "",
        "## Osservazioni",
        "",
        "| Trial | Pair | Preferenza | A (sistema; parola tagliata) | B (sistema; parola tagliata) | Nota |",
        "|---:|---|---|---|---|---|",
    ]
    for observation in document["observations"]:
        candidates = observation["candidates"]
        preferred = observation["preferred_candidate"]
        preference = observation["preference_label"]
        if preferred is not None:
            preference = f"{preference} ({preferred['system']})"
        comment = observation["comment"].replace("|", "\\|").replace("\n", " ")
        cells = []
        for label in ("A", "B"):
            candidate = candidates[label]
            cells.append(f"`{candidate['system']}`; {str(candidate['word_cut']).lower()}")
        lines.append(
            f"| {observation['trial']} | `{observation['pair_id']}` | {preference} | "
            f"{cells[0]} | {cells[1]} | {comment} |"
        )
    lines.extend(
        [
            "",
            "## Limiti di interpretazione",
            "",
            "Il mapping è stato usato solo dopo conferma esplicita dell'associazione tra "
            "ciascun file di giudizi e il run selezionato. La preferenza è riportata per "
            "trial e la segnalazione di parola tagliata per candidato; non viene calcolato "
            "un vincitore generale, un intervallo di confidenza o una significatività. "
            "`confidence` resta volutamente `null`: una singola osservazione di ascolto "
            "non fornisce una confidenza statistica.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "ALLOWED_ANSWERS",
    "AmbiguousMappingError",
    "DuplicateJudgementError",
    "EVALUATION_SCHEMA",
    "EvaluationError",
    "LISTENING_SCHEMA",
    "LISTENING_SCHEMA_V2",
    "LISTENING_SCHEMAS",
    "MOTION_LABELS",
    "MissingTrialError",
    "SchemaValidationError",
    "import_judgements",
    "load_judgements",
    "load_run_key",
    "render_report",
    "sha256_file",
    "write_json",
]
