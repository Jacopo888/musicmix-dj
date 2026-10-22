import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.data("conoscenza_mix/07_dati_benchmark/benchmark_v12/mdjcue_benchmark.json")


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "conoscenza_mix/07_dati_benchmark/benchmark_v12"


def load(name):
    with (OUTPUTS / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_real_mdjcue_counts_and_grouped_split():
    benchmark = load("mdjcue_benchmark.json")
    assert benchmark["dataset"]["name"] == "M-DJCUE"
    assert benchmark["dataset"]["audio_included"] is False
    assert len(benchmark["tracks"]) == 134
    assert sum(len(track["events"]) for track in benchmark["tracks"]) == 1849
    assert sum(len(track["clusters"]) for track in benchmark["tracks"]) == 1082
    split = benchmark["split"]
    assert split["test_tracks"] == 40
    assert split["artist_overlap"] == []
    assert split["version_family_overlap"] == []
    train_ids = set(split["train_track_ids"])
    test_ids = set(split["test_track_ids"])
    assert not train_ids & test_ids


def test_position_baseline_was_evaluated_on_test_split():
    evaluation = load("evaluation_position_fraction.json")
    micro = evaluation["metrics"]["support_ge_1"]["micro"]
    assert evaluation["config"]["tolerance_s"] == 0.5
    assert micro["gold_clusters"] == 346
    assert micro["tp"] == 3
    assert micro["fp"] == 70
    assert micro["fn"] == 343
    assert evaluation["coverage"]["tracks"] == 40
    assert evaluation["predictions"]["sha256"]


def test_annotator_agreement_and_abstention_fields():
    evaluation = load("evaluation_cli_baseline.json")
    assert evaluation["agreement"]["all"]["pairs_scored"] == 802
    assert 0.5 < evaluation["agreement"]["all"]["mean_pairwise_f1"] < 0.55
    assert "abstention_rate" in evaluation["metrics"]["support_ge_1"]["labels"]["IN"]
    assert evaluation["predictions"]["explicit_abstentions"] == 0
