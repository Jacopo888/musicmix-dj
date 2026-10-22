"""Reproduce three audit observations without audio models or source music.

Run from this checkout: python tools/book_pipeline_probes.py
These are diagnostic probes, not assertions that the observed behavior is right.
The output includes inputs and the imported production function results.

Update 21/10/2026 (P0 "misure e regole"): the three counterexamples of
conoscenza_mix/libri/audit_2026-10-21/ROOT_COUNTEREXAMPLES.json are fixed in the production
functions; each probe now prints the current result, the pre-fix behaviour where it can still be
reproduced (``legacy``), and ``fixed`` = whether the current result avoids the counterexample.
Tests: tests/test_dj_form.py::test_no_core_is_unknown_not_outro,
tests/test_dj_phrase_books.py::test_stem_onset_past_word_end_is_not_snapped,
tests/test_p0_misure_regole.py::test_identical_timbre_matrix_abstains_instead_of_far.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from musicmix import dj_form, dj_phrase, dj_set  # noqa: E402


def probe() -> dict:
    layers = {"kick": [0.0] * 8, "onsets_per_beat": [1.0] * 8}
    layers.update({f"{k}_db": [-12.0] * 8 for k in ("bass", "mid", "high", "vocals")})
    form = dj_form.form_segments(layers)
    word = {"on": 900, "off": 1000, "last_syl_on": 990, "n_syl": 1, "text": "test"}
    model, info = dj_phrase.with_stem_onsets(
        {"sample_rate": 1000, "words": [word]}, onsets={"onsets_s": [1.04]}
    )
    model_l, info_l = dj_phrase.with_stem_onsets(
        {"sample_rate": 1000, "words": [dict(word)]}, onsets={"onsets_s": [1.04]}, beyond_word="legacy"
    )
    Z = np.zeros((3, 3))
    cal = dj_set.far_timbre_calibration(Z, 90)
    threshold = dj_set.far_timbre_threshold(Z, 90)
    pc = {"candidates": [{"decision": {"type": "echo_out", "render": {}},
                           "admissible": True, "cost": 0.0, "plan": {"score": 1.0}}]}
    changed = dj_set.apply_far_genre(pc, 0.0, cal, 120.0)
    legacy_float = dj_set.apply_far_genre(pc, 0.0, cal["raw_threshold"], 120.0)  # the old float threshold 0.0
    return {
        "form_no_core": {"input": layers, "first_core": form["first_core_bar"], "labels": form["labels"],
                         "core_status": form["core_status"], "no_core_reason": form["no_core_reason"],
                         "version": form["version"], "legacy": "8 x 'outro' (dj_form.1)",
                         "fixed": "outro" not in form["labels"] and form["core_status"] == "no_core"},
        "onset_after_word": {"input_word": word, "onsets_s": [1.04],
                             "output_word": model["words"][0], "snapped_share": info["snapped_share"],
                             "n_uncertain_beyond_word": info["n_uncertain_beyond_word"],
                             "legacy": {"output_word": model_l["words"][0], "snapped_share": info_l["snapped_share"]},
                             "fixed": model["words"][0]["syl_on"][0] < word["off"] and info["snapped_share"] == 0.0},
        "identical_timbre": {"input_distance_matrix": Z.tolist(), "percentile": 90,
                             "calibration": cal, "threshold": threshold, "far_genre": changed["far_genre"],
                             "render": changed["candidates"][0]["decision"]["render"],
                             "legacy_float_threshold_0": legacy_float["far_genre"],
                             "fixed": not changed["far_genre"]["far"] and not legacy_float["far_genre"]["far"]
                                      and "tail_stop" not in changed["candidates"][0]["decision"]["render"]},
    }


if __name__ == "__main__":
    print(json.dumps(probe(), ensure_ascii=False, indent=2))
