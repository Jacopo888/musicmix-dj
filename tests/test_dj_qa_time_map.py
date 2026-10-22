"""QA in the time actually rendered (P0.6, 21/10/2026): B projected through the renderer's piecewise time map.

B has an uneven bar grid (bars alternately 1.5 % slower in the first half of the overlap and 1.5 % faster in the
second), so the per-bar time map of dj_render ("grid" mode, Rubber Band keyframes) differs from the straight line
through the switch that dj_qa used before. Markers in samples: B's downbeats (the map nodes) and short 700 Hz
"word" bursts in the middle of bars. Their positions are detected independently in the stretched B deck and
compared with (1) the time-map projection and (2) the old linear approximation.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

pytest.importorskip("pyloudnorm")
pytest.importorskip("soundfile")
from scipy.signal import butter, hilbert, sosfiltfilt  # noqa: E402

from musicmix import dj_qa  # noqa: E402
from musicmix import dj_render as dj  # noqa: E402

SR = 48000
needs_cli = pytest.mark.skipif(not shutil.which("rubberband"), reason="per-bar time map needs the Rubber Band CLI (keyframes)")


def _burst(f, n, tau):
    k = np.arange(n)
    return np.exp(-k / (tau * SR)) * np.sin(2 * np.pi * f * k / SR)


def uneven_b(n_bars=24, bar0=60.0 / 126 * 4, dev=0.015, jb0=8, ov=8):
    """B at ~126 BPM; bars jb0..jb0+ov/2 are (1+dev) long, the next ov/2 are (1-dev); 700 Hz markers at 40 % of each bar."""
    lens = np.full(n_bars, bar0)
    lens[jb0 : jb0 + ov // 2] *= 1 + dev
    lens[jb0 + ov // 2 : jb0 + ov] *= 1 - dev
    downs = 0.5 + np.r_[0.0, np.cumsum(lens)]
    beats = np.concatenate([d + np.arange(4) * l / 4 for d, l in zip(downs[:-1], lens)] + [downs[-1:]])
    n = int((downs[-1] + 2.0) * SR)
    x = 0.3 * np.sin(2 * np.pi * 82.0 * np.arange(n) / SR)
    for i, t in enumerate(beats):
        p = int(round(t * SR))
        x[p : p + 480] += (1.0 if i % 4 == 0 else 0.35) * _burst(3000.0 if i % 4 == 0 else 1500.0, 480, 0.00125)[: len(x[p : p + 480])]
    words = []
    for d, l in zip(downs[:-1], lens):
        p = int(round((d + 0.4 * l) * SR))
        x[p : p + 1440] += 0.5 * _burst(700.0, 1440, 0.01)
        words.append(p)
    return {"audio": x, "beats": beats, "downbeats": downs, "markers": np.array(words)}


def steady_a(bpm=120.0, dur=60.0):
    beats = np.arange(0.5, dur - 0.5, 60.0 / bpm)
    n = int(dur * SR)
    x = 0.2 * np.sin(2 * np.pi * 55.0 * np.arange(n) / SR)
    for i, t in enumerate(beats):
        p = int(round(t * SR))
        x[p : p + 480] += (1.0 if i % 4 == 0 else 0.35) * _burst(2000.0, 480, 0.00125)
    return {"audio": x, "beats": beats, "downbeats": beats[::4]}


def _onsets(x, lo, hi, expected, search_ms=80.0):
    env = np.abs(hilbert(sosfiltfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), x.mean(axis=1))))
    w = int(search_ms / 1000 * SR)
    out = []
    for e in expected:
        e = int(round(e))
        lo_i, hi_i = max(0, e - w), min(len(env), e + w)
        seg = env[lo_i:hi_i]
        out.append(lo_i + np.flatnonzero(seg > 0.5 * seg.max())[0] if len(seg) and seg.max() > 0 else np.nan)
    return np.array(out, dtype=float)


@pytest.fixture(scope="module")
def blend():
    a, b = steady_a(), uneven_b()
    r = dj.render_transition(a["audio"], b["audio"], beats_a=a["beats"], downbeats_a=a["downbeats"], beats_b=b["beats"],
                             downbeats_b=b["downbeats"], cue_out_a=a["downbeats"][10], cue_in_b=b["downbeats"][8], kind="blend",
                             bars=8, sample_rate=SR, config=dj.DJRenderConfig(cue_convention="start", stretch_mode="grid"),
                             return_decks=True)
    return r, b


@needs_cli
def test_time_map_is_in_the_metadata(blend):
    r, _ = blend
    tm = r.metadata["stretch"]["time_map"]
    assert tm["kind"] == "piecewise" and len(tm["b_source_s"]) == 9
    assert dj_qa.b_projection_info(r.metadata)["method"] == "time_map"


@needs_cli
def test_nodes_and_words_land_where_the_map_says(blend):
    r, b = blend
    meta = r.metadata
    raw = r.decks["b_stretched_raw"]
    n = len(r.audio)
    linear_meta = dict(meta, stretch={k: v for k, v in meta["stretch"].items() if k != "time_map"})
    assert dj_qa.b_projection_info(linear_meta)["method"] == "linear_approximation"
    # source-side reference positions from the same detector (detector bias cancels)
    src = np.repeat(b["audio"][:, None], 2, axis=1)
    db_src = _onsets(src, 2500, 3500, b["downbeats"][8:17] * SR, 20.0)
    mk_src = _onsets(src, 600, 800, b["markers"][8:16], 20.0)
    for name, ref, lo, hi in (("nodes", db_src, 2500, 3500), ("words", mk_src, 600, 800)):
        proj = dj_qa.b_output_position(meta, ref)
        lin = dj_qa.b_output_position(linear_meta, ref)
        inside = (proj > 0) & (proj < n)
        found = _onsets(raw, lo, hi, proj[inside])
        err_map = np.abs(found - proj[inside]) / SR * 1000
        err_lin = np.abs(found - lin[inside]) / SR * 1000
        assert np.all(np.isfinite(found)), name
        assert err_map.max() < 3.0, (name, err_map)
        # the straight line through the switch misses the per-bar map by tens of ms at the overlap edges
        assert err_lin.max() > 15.0 and err_lin.max() > 5 * err_map.max(), (name, err_lin)
    # nodes: each B downbeat of the overlap is on the corresponding A downbeat (output seconds)
    tm = meta["stretch"]["time_map"]
    proj_nodes = dj_qa.b_output_position(meta, np.asarray(tm["b_source_s"]) * SR) / SR
    assert np.allclose(proj_nodes, tm["output_s"], atol=1e-6)


@needs_cli
def test_stems_and_words_of_b_use_the_map(blend):
    r, b = blend
    meta = r.metadata
    n = len(r.audio)
    pos = dj_qa.b_source_position(meta, n)
    probe = np.array([meta["overlap"]["output_start_sample"] + 12345, meta["switch"]["output_sample"] + 777])
    back = dj_qa.b_output_position(meta, pos[probe])
    assert np.allclose(back, probe, atol=1e-6)
    words = [{"start_sample": int(p), "end_sample": int(p) + 1440, "text": f"w{i}"} for i, p in enumerate(b["markers"][8:16])]
    out = dj_qa.words_to_output(meta, words, n, deck="b")
    assert len(out) == 8 and all(w["projection"] == "time_map" for w in out)
    exp = dj_qa.b_output_position(meta, b["markers"][8:16]) / SR
    assert np.allclose([w["start_s"] for w in out], exp, atol=1e-9)
