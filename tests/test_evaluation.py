import json
from pathlib import Path
import tempfile
import unittest

from musicmix.evaluation import (
    AmbiguousMappingError,
    DuplicateJudgementError,
    MissingTrialError,
    MOTION_LABELS,
    SchemaValidationError,
    import_judgements,
    load_judgements,
)


class EvaluationTests(unittest.TestCase):
    def _fixture(self, *, rows=None, key_trials=None, manifest=True):
        folder = tempfile.TemporaryDirectory()
        root = Path(folder.name)
        run = root / "run"
        listening = run / "listening_ab"
        listening.mkdir(parents=True)
        if key_trials is None:
            key_trials = [
                {"trial": 1, "label": "A", "system": "new", "source": "new-1.wav", "pair_id": "p1"},
                {"trial": 1, "label": "B", "system": "old", "source": "old-1.wav", "pair_id": "p1"},
                {"trial": 2, "label": "A", "system": "new", "source": "new-2.wav", "pair_id": "p2"},
                {"trial": 2, "label": "B", "system": "old", "source": "old-2.wav", "pair_id": "p2"},
            ]
        (listening / "key.json").write_text(
            json.dumps({"seed": 17, "trials": key_trials}), encoding="utf-8"
        )
        if manifest:
            (run / "manifest.json").write_text(
                json.dumps({"run_id": "run-17"}), encoding="utf-8"
            )
        if rows is None:
            rows = [
                {"trial": 1, "pair_id": "p1", "answer": "A", "word_cut_a": False, "word_cut_b": True, "comment": "ok"},
                {"trial": 2, "pair_id": "p2", "answer": "neither", "word_cut_a": False, "word_cut_b": False, "comment": ""},
            ]
        judgement = root / "judgements.json"
        judgement.write_text(
            json.dumps({"schema_version": "musicmix.listening.1", "judgements": rows}),
            encoding="utf-8",
        )
        return folder, judgement, run

    def test_invalid_schema_rejected(self):
        folder, judgement, run = self._fixture()
        self.addCleanup(folder.cleanup)
        judgement.write_text(json.dumps({"schema_version": "wrong", "judgements": []}), encoding="utf-8")
        with self.assertRaises(SchemaValidationError):
            import_judgements(judgement, run)

    def test_ambiguous_key_rejected(self):
        key = [
            {"trial": 1, "label": "A", "system": "one", "source": "one.wav", "pair_id": "p1"},
            {"trial": 1, "label": "A", "system": "two", "source": "two.wav", "pair_id": "p1"},
            {"trial": 1, "label": "B", "system": "old", "source": "old.wav", "pair_id": "p1"},
        ]
        folder, judgement, run = self._fixture(key_trials=key, rows=[
            {"trial": 1, "pair_id": "p1", "answer": "A", "word_cut_a": False, "word_cut_b": False, "comment": ""}
        ])
        self.addCleanup(folder.cleanup)
        with self.assertRaises(AmbiguousMappingError):
            import_judgements(judgement, run)

    def test_missing_trial_rejected(self):
        folder, judgement, run = self._fixture(rows=[
            {"trial": 1, "pair_id": "p1", "answer": "A", "word_cut_a": False, "word_cut_b": False, "comment": ""}
        ])
        self.addCleanup(folder.cleanup)
        with self.assertRaises(MissingTrialError):
            import_judgements(judgement, run)

    def test_bad_answer_rejected(self):
        folder, judgement, run = self._fixture(rows=[
            {"trial": 1, "pair_id": "p1", "answer": "C", "word_cut_a": False, "word_cut_b": False, "comment": ""},
            {"trial": 2, "pair_id": "p2", "answer": "A", "word_cut_a": False, "word_cut_b": False, "comment": ""},
        ])
        self.addCleanup(folder.cleanup)
        with self.assertRaises(SchemaValidationError):
            import_judgements(judgement, run)

    def test_exact_duplicates_collapse_but_conflicts_reject(self):
        row = {"trial": 1, "pair_id": "p1", "answer": "A", "word_cut_a": False, "word_cut_b": True, "comment": ""}
        folder, judgement, run = self._fixture(rows=[row, dict(row),
            {"trial": 2, "pair_id": "p2", "answer": "B", "word_cut_a": False, "word_cut_b": False, "comment": ""}])
        self.addCleanup(folder.cleanup)
        document = import_judgements(judgement, run, mapping_confirmed=True)
        self.assertEqual(document["summary"]["observation_count"], 2)
        self.assertEqual(document["observations"][0]["candidates"]["B"]["word_cut"], True)

        conflicting = dict(row)
        conflicting["answer"] = "B"
        judgement.write_text(json.dumps({"schema_version": "musicmix.listening.1", "judgements": [row, conflicting,
            {"trial": 2, "pair_id": "p2", "answer": "B", "word_cut_a": False, "word_cut_b": False, "comment": ""}]}), encoding="utf-8")
        with self.assertRaises(DuplicateJudgementError):
            import_judgements(judgement, run)

    def test_mapping_pending_withholds_system_preference_and_confidence(self):
        folder, judgement, run = self._fixture()
        self.addCleanup(folder.cleanup)
        document = import_judgements(judgement, run)
        self.assertFalse(document["mapping"]["confirmed"])
        self.assertIsNone(document["observations"][0]["preferred_candidate"])
        self.assertIsNone(document["summary"]["confidence"])
        self.assertIsNone(document["observations"][0]["annotation"]["confidence"])

        confirmed = import_judgements(judgement, run, mapping_confirmed=True)
        self.assertEqual(confirmed["observations"][0]["preferred_candidate"]["system"], "new")

    def test_v2_identity_fields_must_match_selected_key(self):
        folder, judgement, run = self._fixture()
        self.addCleanup(folder.cleanup)
        key_path = run / "listening_ab" / "key.json"
        key = json.loads(key_path.read_text(encoding="utf-8"))
        key.update({"test_id": "test-1", "run_id": "run-17", "title": "AB", "feedback_filename": "x.json"})
        key_path.write_text(json.dumps(key), encoding="utf-8")
        judgement.write_text(json.dumps({
            "schema_version": "musicmix.listening.2",
            "test_id": "test-1", "run_id": "run-17", "title": "AB", "feedback_filename": "x.json",
            "judgements": json.loads(judgement.read_text(encoding="utf-8"))["judgements"],
        }), encoding="utf-8")
        document = import_judgements(judgement, run)
        self.assertEqual(document["source_schema_version"], "musicmix.listening.2")

        bad = json.loads(judgement.read_text(encoding="utf-8"))
        bad["run_id"] = "different-run"
        judgement.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaises(SchemaValidationError):
            import_judgements(judgement, run)

    def test_motion_annotations_are_preserved_per_candidate_and_in_duplicates(self):
        rows = [
            {"trial": 1, "pair_id": "p1", "answer": "A", "word_cut_a": False,
             "word_cut_b": True, "comment": "ok", "motion_a": "continues",
             "motion_b": "unclear"},
            {"trial": 1, "pair_id": "p1", "answer": "A", "word_cut_a": False,
             "word_cut_b": True, "comment": "ok", "motion_a": "continues",
             "motion_b": "unclear"},
            {"trial": 2, "pair_id": "p2", "answer": "B", "word_cut_a": False,
             "word_cut_b": False, "comment": "", "motion_a": None,
             "motion_b": "resolves"},
        ]
        folder, judgement, run = self._fixture(rows=rows)
        self.addCleanup(folder.cleanup)
        payload, normalized = load_judgements(judgement)
        self.assertEqual(len(normalized), 2)
        self.assertEqual(MOTION_LABELS, {"continues", "resolves", "interrupts", "unclear"})
        document = import_judgements(judgement, run, mapping_confirmed=True)
        first = document["observations"][0]["candidates"]
        self.assertEqual(first["A"]["motion"], "continues")
        self.assertEqual(first["B"]["motion"], "unclear")
        second = document["observations"][1]["candidates"]
        self.assertIsNone(second["A"]["motion"])
        self.assertEqual(second["B"]["motion"], "resolves")

        conflicting = dict(rows[0])
        conflicting["motion_a"] = "interrupts"
        judgement.write_text(json.dumps({"schema_version": "musicmix.listening.1",
                                         "judgements": [rows[0], conflicting, rows[2]]}),
                             encoding="utf-8")
        with self.assertRaises(DuplicateJudgementError):
            import_judgements(judgement, run)

    def test_invalid_motion_annotation_is_rejected(self):
        row = {"trial": 1, "pair_id": "p1", "answer": "A", "word_cut_a": False,
               "word_cut_b": False, "comment": "", "motion_a": "good"}
        folder, judgement, run = self._fixture(rows=[row,
            {"trial": 2, "pair_id": "p2", "answer": "B", "word_cut_a": False,
             "word_cut_b": False, "comment": ""}])
        self.addCleanup(folder.cleanup)
        with self.assertRaises(SchemaValidationError):
            load_judgements(judgement)


if __name__ == "__main__":
    unittest.main()
