from __future__ import annotations

import pytest

from musicmix.motion import MotionBeat, analyze_motion


def _beat(rms: float, *, tension: float | None = None, novelty: float = 0.2, confidence: float | None = 0.95) -> MotionBeat:
    chroma = (1.0,) + (0.0,) * 11
    return MotionBeat(
        rms=rms,
        chroma=chroma,
        novelty=novelty,
        tension_proxy=tension,
        confidence=confidence,
    )


def test_tension_drop_and_similar_chroma_do_not_prove_resolution() -> None:
    outgoing = [_beat(0.30 + 0.01 * index, tension=0.75, novelty=0.5) for index in range(8)]
    incoming = [_beat(0.25 - 0.01 * index, tension=0.25, novelty=0.2) for index in range(8)]

    assessment = analyze_motion(outgoing, incoming, windows_beats=(8,))
    report = assessment.scales[0]

    assert report.classification == "tension_proxy_decrease"
    assert report.tension_delta < -0.1
    assert report.chroma_boundary_similarity == pytest.approx(1.0)
    assert report.confidence == pytest.approx(0.95)
    assert "measured_tension_proxy_decreased" in report.notes
    assert "resolution_requires_independent_closure_evidence" in report.notes


def test_missing_tension_never_claims_musical_resolution() -> None:
    outgoing = [_beat(0.20 + 0.01 * index, novelty=0.5) for index in range(8)]
    incoming = [_beat(0.18 - 0.01 * index, novelty=0.1) for index in range(8)]

    report = analyze_motion(outgoing, incoming, windows_beats=(8,)).scales[0]

    assert report.classification == "energy_fall"
    assert "tension_proxy" in report.missing
    assert "tension_proxy_missing_or_incomplete" in report.notes
    assert report.classification != "resolution_likely"


def test_partial_tension_proxy_is_unknown_instead_of_an_averaged_resolution() -> None:
    outgoing = [_beat(0.30, tension=0.80) for _ in range(8)]
    incoming = [_beat(0.25, tension=0.20) for _ in range(7)] + [_beat(0.25, tension=None)]

    report = analyze_motion(outgoing, incoming, windows_beats=(8,)).scales[0]

    assert report.tension_before_mean is None
    assert report.tension_after_mean is None
    assert report.tension_delta is None
    assert "tension_proxy" in report.missing
    assert report.classification != "resolution_likely"


def test_energy_rise_and_novelty_are_kept_as_observed_descriptors() -> None:
    outgoing = [_beat(0.20, novelty=0.10) for _ in range(8)]
    incoming = [_beat(0.45, novelty=0.50) for _ in range(8)]

    report = analyze_motion(outgoing, incoming, windows_beats=(8,)).scales[0]

    assert report.classification == "energy_rise"
    assert report.energy_boundary_delta > 0
    assert report.novelty_boundary_delta > 0
    assert "tension_proxy_missing_or_incomplete" in report.notes


def test_all_requested_scales_are_reported_and_short_windows_are_explicit() -> None:
    outgoing = [_beat(0.3) for _ in range(16)]
    incoming = [_beat(0.3) for _ in range(16)]

    assessment = analyze_motion(outgoing, incoming)

    assert [report.window_beats for report in assessment.scales] == [8, 16, 32]
    assert assessment.scales[0].classification == "energy_continuation"
    assert assessment.scales[1].classification == "energy_continuation"
    assert assessment.scales[2].classification == "insufficient_window"
    assert assessment.scales[2].missing == ("before_window",)
    assert assessment.preferred_window_beats == 16
    assert assessment.diagnostics["tension_inference_requires_proxy"] is True


def test_chroma_is_unrotated_and_confidence_stays_unknown_when_incomplete() -> None:
    outgoing = [MotionBeat(rms=0.3, chroma=(1.0,) + (0.0,) * 11, confidence=0.9) for _ in range(8)]
    incoming = [MotionBeat(rms=0.3, chroma=(0.0,) * 6 + (1.0,) + (0.0,) * 5, confidence=None) for _ in range(8)]

    report = analyze_motion(outgoing, incoming, windows_beats=(8,)).scales[0]

    assert report.chroma_boundary_similarity == pytest.approx(0.0)
    assert report.confidence is None
    assert "confidence_incomplete" in report.notes


def test_mapping_input_is_supported_and_analysis_is_deterministic() -> None:
    outgoing = [{"rms": 0.3, "novelty": 0.2, "chroma": [1.0] + [0.0] * 11, "confidence": 0.8} for _ in range(8)]
    incoming = [{"rms": 0.3, "novelty": 0.2, "chroma": [1.0] + [0.0] * 11, "confidence": 0.8} for _ in range(8)]

    first = analyze_motion(outgoing, incoming, windows_beats=(8,))
    second = analyze_motion(outgoing, incoming, windows_beats=(8,))

    assert first == second


def test_invalid_motion_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        MotionBeat(rms=-0.1)
    with pytest.raises(ValueError):
        MotionBeat(chroma=(1.0, 0.0))
    with pytest.raises(ValueError):
        MotionBeat(tension_proxy=1.2)
    with pytest.raises(ValueError):
        analyze_motion([], [], windows_beats=(8, 8))
    with pytest.raises(TypeError):
        analyze_motion(["bad"], [])  # type: ignore[list-item]
