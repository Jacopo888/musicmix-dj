from musicmix.transition_validation import construct_counterfactual, validate_transition_event


def artifact(**overrides):
    value = {
        "path": "missing.wav",
        "intent": "handoff",
        "clocks": {"rate": 48_000, "source": {
            "a": {"track_id": "a", "cue_sample": 96_000},
            "b": {"track_id": "b", "cue_sample": 48_000}},
            "output": {"join_sample": 48_000}},
    }
    value.update(overrides)
    return value


def test_level_jump_and_phrase_asr_are_not_claimed_as_musical_defects():
    report = validate_transition_event(
        {"track_a": "a", "track_b": "b", "sample_rate": 48_000,
         "cue_out_sample": 96_000, "cue_in_sample": 48_000,
         "output_join_sample": 48_000,
         "phrase": {"status": "unknown", "provider": "ASR"},
         "song_eval": {"diagnostic": "positive"}},
        artifact=artifact(),
        quality={"join": {"sample": 48_000}, "flags": [
            {"code": "large_level_change_descriptive", "severity": "descriptive_only", "description": "12 dB"},
        ]},
    )
    assert report["hard_defects"] == []
    assert any(item["code"] == "large_level_change_descriptive" for item in report["descriptive"])
    assert any(item["code"] == "phrase_closure_unknown" for item in report["unknown"])
    assert report["musical_quality"] is None
    assert report["diagnostics"]["song_eval_use"] == "diagnostic_only_not_used_as_reward_or_quality_score"


def test_clipping_and_actual_anchor_mismatch_are_hard_defects():
    report = validate_transition_event(
        {"track_a": "a", "track_b": "b", "output_join_sample": 48_001},
        artifact=artifact(),
        quality={"join": {"sample": 48_000}, "flags": [
            {"code": "clipping_detected", "severity": "technical_defect", "description": "clip"},
        ]},
    )
    codes = {item["code"] for item in report["hard_defects"]}
    assert {"clipping_detected", "actual_anchor_mismatch"} <= codes
    assert report["status"] == "fail"


def test_counterfactual_labels_known_properties_without_quality_verdict():
    result = construct_counterfactual({"intent": "handoff"}, order="reverse", shift_samples=10, silence=True, clipping=True)
    assert result["counterfactual"]["known_properties"] == ["order_changed", "anchor_shift_applied", "silence_inserted", "clipping_injected"]
    assert result["counterfactual"]["musical_quality"] is None
