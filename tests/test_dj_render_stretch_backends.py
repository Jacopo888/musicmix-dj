"""Qualification of the time-stretch backends (P0.1, RENDER_TRIAGE.md, 21/10/2026).

The 82 Hz pitch contract of tests/test_dj_render.py is kept as is: a backend that moves 82 Hz (the old ffmpeg R2
``transients=crisp`` fallback gave 78.1 Hz with time_ratio 1.05) is refused, never accommodated by the test.
The three triage tests (pitch, never two full basses, downbeats of stretched B) are repeated with the ffmpeg
fallback forced, so that the Linux machine (Rubber Band CLI present) also covers the path the Windows machine takes.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

pytest.importorskip("pyloudnorm")
pytest.importorskip("soundfile")

from musicmix import dj_render as dj  # noqa: E402

import test_dj_render as base  # noqa: E402  (helpers and assertions of the renderer tests)

SR = 48000
HAS_CLI = bool(shutil.which("rubberband"))
HAS_FFMPEG = bool(shutil.which("ffmpeg"))


def _ffmpeg_qualified() -> bool:
    if not HAS_FFMPEG:
        return False
    try:
        dj.stretch_backend_info("ffmpeg", sample_rate=SR)
        return True
    except dj.DJRenderError:
        return False


needs_ffmpeg_r3 = pytest.mark.skipif(not _ffmpeg_qualified(), reason="ffmpeg without librubberband >= 3.0 (fallback refused)")


def _sine(f0, seconds=4.0, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    x = 0.3 * np.sin(2 * np.pi * f0 * t)
    return np.stack([x, x], axis=1)


@pytest.mark.skipif(not HAS_FFMPEG, reason="no ffmpeg")
def test_legacy_r2_crisp_fallback_moves_82hz_and_is_refused():
    desc = dj.stretch_backend_info("ffmpeg-r2-crisp", qualify=False)
    y = dj._run_stretch(_sine(82.0), SR, 1.05, None, desc)
    # The defect reproduced on Linux: the measured fundamental is far from 82 Hz (78.1 Hz in the triage).
    assert abs(dj._peak_hz(y, SR, 82.0) - 82.0) > 0.5
    with pytest.raises(dj.DJRenderError, match="no qualified time-stretch backend"):
        dj.stretch_backend_info("ffmpeg-r2-crisp", sample_rate=SR)


@pytest.mark.skipif(not HAS_FFMPEG, reason="no ffmpeg")
@pytest.mark.parametrize("name", ["ffmpeg-r2-smooth", "ffmpeg-atempo"])
def test_other_ffmpeg_variants_are_refused(name):
    q = dj.qualify_stretch_backend(dj.stretch_backend_info(name, qualify=False), SR)
    assert not q["passed"] and q["summary"]["failed"]
    with pytest.raises(dj.DJRenderError):
        dj.stretch_backend_info(name, sample_rate=SR)


@pytest.mark.skipif(not HAS_CLI, reason="no rubberband CLI")
def test_rubberband_cli_passes_the_contract():
    info = dj.stretch_backend_info("rubberband", sample_rate=SR)
    assert info["qualification"]["passed"] and info["keyframes"]


@needs_ffmpeg_r3
@pytest.mark.parametrize("ratio", [0.95, 1.05])
def test_forced_ffmpeg_fallback_keeps_82hz_and_length(ratio):
    desc = dj.stretch_backend_info("ffmpeg", sample_rate=SR)
    assert desc["qualification"]["passed"]
    x = _sine(82.0)
    rep: dict = {}
    y = dj._run_stretch(x, SR, ratio, None, desc, report=rep)
    assert dj._peak_hz(y, SR, 82.0) == pytest.approx(82.0, abs=0.5)
    assert abs(len(y) - round(len(x) * ratio)) <= 1
    assert rep["pilot_clicks"] >= 4 and rep["pilot_residual_ms"] < 2.0


def test_unknown_backend_name_is_an_error():
    with pytest.raises(dj.DJRenderError, match="unknown stretch backend"):
        dj.stretch_backend_info("sox")


# ------------------------------------------- the three triage tests with the fallback forced


@pytest.fixture(scope="module")
def tracks():
    a = base.click_track(120, 50, 55, gain=0.3)
    b = base.click_track(126, 50, 82, gain=1.5)
    return a, b


@pytest.fixture(scope="module")
def blend8_ffmpeg(tracks):
    if not _ffmpeg_qualified():
        pytest.skip("ffmpeg fallback refused on this machine")
    a, b = tracks
    return base._render(a, b, bars=8, return_decks=True, config=dj.DJRenderConfig(stretch_backend="ffmpeg"))


def test_forced_fallback_metadata(blend8_ffmpeg):
    st = blend8_ffmpeg.metadata["stretch"]
    assert st["backend"]["variant"] == "ffmpeg" and st["backend"]["qualification"]["passed"]
    assert st["mode"].startswith("fit")  # constant ratio: no keyframes through ffmpeg
    assert st["start_delay_compensation"]["pilot_residual_ms"] < 2.0


def test_forced_fallback_stretch_preserves_pitch(blend8_ffmpeg):
    base.test_stretch_preserves_pitch(blend8_ffmpeg)


def test_forced_fallback_never_two_full_basses(blend8_ffmpeg):
    base.test_never_two_full_basses(blend8_ffmpeg)


def test_forced_fallback_downbeats_land_on_a_grid(blend8_ffmpeg, tracks):
    base.test_downbeats_of_stretched_b_land_on_a_grid(blend8_ffmpeg, tracks)
