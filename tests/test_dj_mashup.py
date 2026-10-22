"""Synthetic checks for dj_mashup (mashability, borrowed vocals on B's grid, freeze glide, swell).

A: click track 120 BPM, sub-bass 55 Hz; its "vocal" sings tone bursts (700 Hz + fifth) on beats 1-3 of every
bar (beat 4 silent = a line end every bar). B: click track 126 BPM (A material stretched x0.952), sub-bass
73 Hz (D), a sustained D major triad (pad) and a vocal (600 Hz) on beats 1-3 of bars 8.. (bars 0-7 instrumental).
Words = the bursts, in source samples.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

pytest.importorskip("pyloudnorm")
pytest.importorskip("soundfile")
pytest.importorskip("librosa")
from scipy.signal import butter, sosfiltfilt  # noqa: E402

from musicmix import dj_creative as cr  # noqa: E402
from musicmix import dj_mashup as mx  # noqa: E402
from musicmix import dj_qa  # noqa: E402
from musicmix import dj_render as dj  # noqa: E402

SR = 48000
HAS_RB = shutil.which("rubberband") is not None
needs_rb = pytest.mark.skipif(not HAS_RB, reason="rubberband CLI not installed")


def click_track(bpm, dur, bass_hz, *, start=0.5, bass_level=0.25, gain=1.0):
    n = int(dur * SR)
    drums = np.zeros(n)
    period = 60.0 / bpm
    burst = np.exp(-np.arange(480) / 60.0)
    beats, t, k = [], start, 0
    while t < dur - 0.2:
        i = int(round(t * SR))
        f, amp = (3000.0, 1.0) if k % 4 == 0 else (1500.0, 0.35)
        drums[i : i + 480] += amp * burst * np.sin(2 * np.pi * f * np.arange(480) / SR)
        beats.append(t)
        t += period
        k += 1
    bass = bass_level * np.sin(2 * np.pi * bass_hz * np.arange(n) / SR)
    beats = np.array(beats)
    return {"drums": drums * gain, "bass": bass * gain, "audio": (drums + bass) * gain, "beats": beats, "downbeats": beats[::4]}


def sung(track, bars, freqs, level=0.25, beats_in_bar=(0, 1, 2)):
    """Tone bursts (70 % of the beat) on the chosen beats of the chosen bars; ``freqs`` cycles over the beats."""
    n = len(track["audio"])
    v = np.zeros(n)
    words = []
    beats = track["beats"]
    period = float(np.median(np.diff(beats)))
    m = int(0.7 * period * SR)
    t = np.arange(m) / SR
    env = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)
    for b in bars:
        for k in beats_in_bar:
            i = 4 * b + k
            if i >= len(beats):
                break
            f = freqs[k % len(freqs)]
            s = int(round(beats[i] * SR))
            v[s : s + m] += level * env * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t))
            words.append({"start_sample": s, "end_sample": s + m, "text": f"w{i}", "confidence": 0.9})
    return v, words


def triad(n, midi, level=0.08):
    t = np.arange(n) / SR
    return level * sum(np.sin(2 * np.pi * 440.0 * 2 ** ((m - 69) / 12) * t) for m in midi)


@pytest.fixture(scope="module")
def tracks():
    a = click_track(120, 70, 55, gain=0.5)
    b = click_track(126, 70, 73.42, gain=1.0)  # bass on D
    b["pad"] = triad(len(b["audio"]), (62, 66, 69))  # D major
    b["audio"] = b["audio"] + b["pad"]
    a["vocals"], a["words"] = sung(a, range(0, 30), (523.25, 659.26, 783.99))  # C E G (C major)
    b["vocals"], b["words"] = sung(b, range(8, 30), (600.0,), level=0.4)
    a["mix"], b["mix"] = a["audio"] + a["vocals"], b["audio"] + b["vocals"]
    return a, b


def _kw(a, b, ia=12, jb=2):
    return dict(grids={"beats_a": a["beats"], "downbeats_a": a["downbeats"], "beats_b": b["beats"], "downbeats_b": b["downbeats"]},
                cue_out_a=a["downbeats"][ia], cue_in_b=b["downbeats"][jb], vocals_a=a["vocals"], vocals_b=b["vocals"],
                words_a=a["words"], words_b=b["words"], sample_rate=SR, return_decks=True)


def _common(r, technique):
    m = r.metadata
    assert m["kind"] == technique and m["creative"]["technique"] == technique
    assert m["mashup_version"] == mx.DJ_MASHUP_VERSION
    ov, sw = m["overlap"], m["switch"]
    assert ov["output_start_sample"] == int(round(10.0 * SR))
    assert ov["output_start_sample"] <= sw["output_sample"] <= ov["output_end_sample"]
    assert m["clip"]["matches_rule"]
    assert len(r.audio) == ov["output_end_sample"] + int(round(10.0 * SR))
    assert m["loudness"]["output_true_peak_dbtp"] <= -1.0 + 1e-6
    assert float(np.max(np.abs(r.audio))) < 1.0  # no clipping
    for seg in m["a_map"] + m["b_map"]:
        assert seg["out_start"] <= seg["out_end"]
    return m


def _active(x, thr=1e-3, hop=960):
    mono = np.asarray(x).mean(axis=1)
    k = len(mono) // hop
    return np.sqrt(np.mean(mono[: k * hop].reshape(k, hop) ** 2, axis=1)) > thr


def _one_voice(r):
    va, vb = _active(r.decks["vocals_a_heard"]), _active(r.decks["vocals_b_heard"])
    m = min(len(va), len(vb))
    return int((va[:m] & vb[:m]).sum())


def _peak_hz(x, lo=300.0, hi=2000.0):
    mono = np.asarray(x).mean(axis=1) if np.ndim(x) == 2 else np.asarray(x)
    w = np.hanning(len(mono))
    spec = np.abs(np.fft.rfft(mono * w, n=8 * len(mono)))
    f = np.fft.rfftfreq(8 * len(mono), 1 / SR)
    sel = (f >= lo) & (f <= hi)
    return float(f[sel][np.argmax(spec[sel])])


# ------------------------------------------------------------ mashability / TIV
def test_tiv_compatibility_matches_vendored_tivlib():
    from musicmix._vendor.tivlib import load

    tiv = load()
    rng = np.random.default_rng(0)
    x, y = rng.random((6, 12)), rng.random((6, 12))
    ours = mx.tiv_compatibility(x, y)
    ref = [tiv.TIV.from_pcp(x[i]).small_scale_compatibility(tiv.TIV.from_pcp(y[i])) for i in range(6)]
    assert np.allclose(ours, ref, rtol=1e-9, atol=1e-12)
    coll = tiv.TIVCollection.from_pcp(x.T).small_scale_compatibility(tiv.TIVCollection.from_pcp(y.T))
    assert np.isclose(ours.sum(), coll, rtol=1e-6)


def _chroma(pcs, n=8):
    c = np.zeros((n, 12))
    c[:, list(pcs)] = 1.0
    return c


def test_harmonic_match_finds_the_transposition():
    top = _chroma((0, 4, 7))  # C major
    bed = _chroma((2, 6, 9))  # D major
    m = mx.harmonic_match(top, bed)
    assert m["best_shift"] == 2 and m["gain_over_no_shift"] > 0.05
    assert next(r for r in m["rows"] if r["shift"] == 2)["cosine"] == pytest.approx(1.0)
    flat = np.ones((8, 12))  # unpitched (rap-like) material: every shift is the same, keep 0
    assert mx.harmonic_match(flat, bed)["best_shift"] == 0
    assert np.allclose(mx.transpose_chroma(top, 2), bed)


def test_fold_ratio_and_resample_beats():
    k, r = mx.fold_ratio(2.0, 1.0)  # source 60 BPM bar, host 120 BPM bar: one source bar = two host bars
    assert k == 2.0 and r == pytest.approx(1.0)
    k, r = mx.fold_ratio(2.0, 1.9)
    assert k == 1.0 and r == pytest.approx(0.95)
    with pytest.raises(dj.DJRenderError):
        mx.fold_ratio(2.0, 1.6, max_stretch=0.12)
    c = np.arange(24.0).reshape(2, 12)
    assert mx.resample_beats(c, 2.0).shape == (4, 12) and mx.resample_beats(c, 0.5).shape == (1, 12)


def test_beat_chroma_is_beat_synchronous(tracks):
    _, b = tracks
    bc = mx.beat_chroma(b["pad"], SR, b["beats"], t0=2.0, t1=8.0)
    assert bc["chroma"].shape[1] == 12 and len(bc["chroma"]) == len(bc["beat_s"]) >= 10
    top3 = set(np.argsort(bc["chroma"].sum(axis=0))[-3:])
    assert top3 == {2, 6, 9}  # D F# A


@needs_rb
def test_stretch_pitch_keeps_length_and_moves_pitch():
    t = np.arange(int(1.5 * SR)) / SR
    x = 0.3 * np.sin(2 * np.pi * 440.0 * t)
    y, info = mx.stretch_pitch(x, SR, 0.95, 2, formant=False, qualify=False)
    assert len(y) == round(len(x) * 0.95)
    assert _peak_hz(y[int(0.2 * SR) : -int(0.2 * SR)]) == pytest.approx(440.0 * 2 ** (2 / 12), abs=2.0)
    assert info["applied"] and info["semitones"] == 2


def test_units_and_lines(tracks):
    a, _ = tracks
    u = mx.vocal_units(a["words"], SR)
    gap = int(0.25 * SR)
    st, en = mx.line_starts(u, gap), mx.line_ends(u, gap)
    assert st[:3] == [0, 3, 6] and en[:3] == [2, 5, 8]  # one line per bar (beat 4 silent)
    stem_units = mx.vocal_units(None, SR, stem=a["vocals"][: 6 * SR])
    assert len(stem_units) >= 3 and all(x["source"] == "stem" for x in stem_units)


# ------------------------------------------------------------------ techniques
@needs_rb
def test_acapella_mashup_line_on_b_grid_shifted_and_one_voice(tracks):
    a, b = tracks
    r = mx.acapella_mashup(a["mix"], b["mix"], key_a="C major", key_b="D major", bars=4, shift=2, **_kw(a, b))
    m = _common(r, "acapella_mashup")
    c = m["creative"]
    ov = c["a_overlay"]
    t_x = m["switch"]["output_sample"]
    ratio = c["params"]["time_ratio_a"]
    assert ratio == pytest.approx(120.0 / 126.0, rel=1e-3)
    p0 = c["a_line"]["source_samples"][0]
    X = int(round(a["downbeats"][12] * SR))
    assert ov["clip_samples"][0] == t_x + int(round((p0 - X) * ratio))
    # the line starts on A's downbeat -> its first word lands on B's entry downbeat (+- the 40 ms gate lead)
    assert abs(ov["clip_samples"][0] - t_x) <= int(0.05 * SR)
    seg = next(s for s in m["a_map"] if s["rate"] != 1.0)
    assert seg["rate"] == pytest.approx(ratio) and seg["src_start"] == p0
    # shift +2 measured on the borrowed vocal: C5 (523 Hz) -> D5 (587 Hz)
    first = r.decks["overlay_parts"]["a_line"][t_x + int(0.05 * SR) : t_x + int(0.3 * SR)]
    assert _peak_hz(first, 450, 700) == pytest.approx(523.25 * 2 ** (2 / 12), abs=4.0)
    assert c["keys"]["a_material_key_after_shift"] == "D major" and c["keys"]["relation_after_shift"] == "same"
    assert _one_voice(r) == 0
    # A's line ends at a line end; B's vocal opens at its phrase start (bar 8) after it
    gate = c["b_vocal_gate"]
    assert gate["status"] == "open" and gate["open_clip_sample"] >= ov["clip_samples"][1]
    jb8 = int(round(b["downbeats"][8] * SR))
    sb = int(round(b["downbeats"][2] * SR))
    assert gate["open_clip_sample"] == t_x + (jb8 - sb) - int(0.04 * SR)
    # A's beat stops on B's entry downbeat; the borrowed voice is in deck A (it is A's material), not trimmed
    assert np.sqrt(np.mean(r.decks["a_raw"][t_x + int(0.05 * SR) :] ** 2)) < 1e-6
    voice = r.decks["a"] - r.decks["a_raw"]
    assert np.allclose(voice[: ov["clip_samples"][0]], 0.0) and np.abs(voice).max() > 0.05


@needs_rb
def test_acapella_mashup_auto_shift_follows_the_bed(tracks):
    a, b = tracks
    r = mx.acapella_mashup(a["mix"], b["mix"], key_a="C major", key_b="D major", bars=2, shift="auto", **_kw(a, b))
    hm = r.metadata["creative"]["keys"]["harmonic_match"]
    assert hm["status"] == "ok" and r.metadata["creative"]["params"]["shift"] == 2


@needs_rb
def test_vocal_loop_repeats_on_b_grid(tracks):
    a, b = tracks
    r = mx.vocal_loop(a["mix"], b["mix"], key_a="C major", key_b="D major", mode="halving", loop_bars=2, **_kw(a, b))
    m = _common(r, "vocal_loop")
    r_bar = mx.vocal_loop(a["mix"], b["mix"], key_a="C major", key_b="D major", mode="bar", loop_bars=2, **_kw(a, b))
    c = m["creative"]
    t_x = m["switch"]["output_sample"]
    beat_b = 60.0 / 126 * SR
    sched = c["params"]["schedule_beats"]
    assert sum(sched) == pytest.approx(8.0)
    pos = 0.0
    for rp, per in zip(c["repeats"], sched):
        assert rp["clip_sample"] == t_x + int(round(pos * round(beat_b)))
        pos += per
    # A's vocal stopped at the line end (beat 3 of the bar before the cue), the chop is that last word
    assert c["a_exit"]["kind"] in ("line_end", "silent") and c["chop"]["words"]
    assert _one_voice(r) == 0
    # mode 'bar': silence between repeats, and every repeat's energy starts on its grid point (10 ms pre-roll of the chop)
    loop = np.abs(r_bar.decks["overlay_parts"]["loop"]).mean(axis=1)
    for rp in r_bar.metadata["creative"]["repeats"]:
        seg = loop[rp["clip_sample"] - int(0.05 * SR) : rp["clip_sample"] + int(0.1 * SR)]
        on = int(np.argmax(seg > 0.1 * seg.max())) - int(0.05 * SR)
        assert 0 <= on <= int(0.02 * SR)


@needs_rb
def test_vocal_chop_on_332(tracks):
    a, b = tracks
    r = mx.vocal_chop(a["mix"], b["mix"], key_a="C major", key_b="D major", bars=2, **_kw(a, b))
    m = _common(r, "vocal_chop")
    t_x = m["switch"]["output_sample"]
    bar_b = round(4 * 60.0 / 126 * SR)
    ons = [h["clip_sample"] - t_x for h in m["creative"]["hits"]]
    exp = [bb * bar_b + int(round(p * bar_b / 8)) for bb in range(2) for p in (0, 3, 6)]
    assert ons == exp
    assert _one_voice(r) == 0


@needs_rb
def test_hook_teaser_is_b_voice_before_entry(tracks):
    a, b = tracks
    r = mx.hook_teaser(a["mix"], b["mix"], hook_s=float(b["downbeats"][10]), key_a="C major", key_b="D major", teaser_bars=2,
                       shift=-2, **_kw(a, b))
    m = _common(r, "hook_teaser")
    c = m["creative"]
    t_x = m["switch"]["output_sample"]
    o0, o1 = c["b_overlay"]["clip_samples"]
    assert o1 <= t_x and o0 >= m["overlap"]["output_start_sample"]
    assert c["b_overlay"]["time_ratio"] == pytest.approx(126.0 / 120.0, rel=1e-3)
    hook = r.decks["overlay_parts"]["b_hook"][o0 + int(0.05 * SR) : o0 + int(0.3 * SR)]
    assert _peak_hz(hook, 450, 700) == pytest.approx(600.0 * 2 ** (-2 / 12), abs=4.0)
    assert _one_voice(r) == 0
    assert np.abs(r.decks["vocals_a_heard"][o0:o1]).max() < 1e-3  # A's vocal silent under the teaser


@needs_rb
def test_call_response_alternates(tracks):
    a, b = tracks
    r = mx.call_response(a["mix"], b["mix"], hook_s=float(b["downbeats"][10]), key_a="C major", key_b="D major", rounds=2,
                         slot_bars=1, **_kw(a, b))
    m = _common(r, "call_response")
    ph = m["creative"]["phrases"]
    assert [p["who"] for p in ph] == ["A", "B", "A", "B"]
    for p, q in zip(ph, ph[1:]):
        assert p["clip_samples"][1] < q["clip_samples"][0]
    assert all(p["time_ratio"] == 1.0 and p["semitones"] == 0 for p in ph if p["who"] == "B")
    assert _one_voice(r) == 0


@needs_rb
def test_beat_swap_drums_then_bass(tracks):
    a, b = tracks
    stems = {"drums": b["drums"], "bass": b["bass"]}
    r = mx.beat_swap(a["mix"], b["mix"], stems_b=stems, key_a="C major", key_b="D major", drum_bars=2, **_kw(a, b))
    m = _common(r, "beat_swap")
    t_x = m["switch"]["output_sample"]
    bar_b = round(4 * 60.0 / 126 * SR)
    h = t_x + 2 * bar_b
    low = sosfiltfilt(butter(4, 120, btype="low", fs=SR, output="sos"), r.decks["b"].mean(axis=1))
    before = np.sqrt(np.mean(low[t_x + int(0.1 * SR) : h - int(0.1 * SR)] ** 2))
    after = np.sqrt(np.mean(low[h + int(0.1 * SR) : h + bar_b] ** 2))
    assert after > 20 * max(before, 1e-9)  # B's bass only from the harmony downbeat
    assert m["creative"]["params"]["shift"] == 0
    assert m["creative"]["a_overlay"]["clip_samples"][1] <= h + int(0.01 * SR)
    assert _one_voice(r) == 0


def test_pivot_target():
    assert mx.pivot_target("A major", "B major")["semitones"] == 2
    assert mx.pivot_target("A minor", "C major")["semitones"] == 0  # A minor is C major's relative minor
    assert mx.pivot_target("C major", "F# major")["semitones"] == 6


def test_pivot_freeze_glides_to_b(tracks):
    a, b = tracks
    t = np.arange(len(a["audio"])) / SR
    mix = a["mix"] + 0.15 * np.sin(2 * np.pi * 440.0 * t)  # sustained A4 in A's instrumental
    kw = _kw(a, b)
    r = mx.pivot_freeze(mix, b["mix"], key_a="A major", key_b="B major", hold_beats=4, glide_beats=4, **kw)
    m = _common(r, "pivot_freeze")
    fr = r.decks["overlay_parts"]["freeze"]
    reg = {x["name"]: x["output"] for x in m["creative"]["regions"]}
    h0, h1 = reg["freeze holds A's chord"]
    g1 = m["switch"]["output_sample"]
    assert _peak_hz(fr[h0 + int(0.1 * SR) : h1 - int(0.1 * SR)], 300, 700) == pytest.approx(440.0, abs=6.0)
    assert _peak_hz(fr[g1 - int(0.35 * SR) : g1 - int(0.05 * SR)], 300, 700) == pytest.approx(440.0 * 2 ** (2 / 12), abs=8.0)
    assert m["creative"]["params"]["gap_bars"] == 2.0


def test_reverse_swell_ends_on_b_word(tracks):
    a, b = tracks
    kw = _kw(a, b, ia=12, jb=8)  # B's cue on its first sung bar: the swell leads into its first word
    r = mx.reverse_swell(a["mix"], b["mix"], key_a="C major", key_b="D major", swell_beats=4, **kw)
    m = _common(r, "reverse_swell")
    sw = r.decks["overlay_parts"]["swell"]
    t_w = m["creative"]["b_word"]["clip_onset_sample"]
    env = np.abs(sw).mean(axis=1)
    assert env[t_w:].max() < 1e-6
    L = int(round(4 * 60.0 / 120 * SR))
    k = int(np.argmax(env[t_w - L : t_w])) + t_w - L
    assert t_w - k < L // 8  # the swell grows into B's word (reverse reverb)
    assert m["creative"]["b_vocal_gate"]["open_clip_sample"] == t_w - int(0.04 * SR)


@needs_rb
def test_dj_qa_runs_on_mashup_renders(tracks):
    a, b = tracks
    r = mx.vocal_loop(a["mix"], b["mix"], key_a="C major", key_b="D major", mode="beat", loop_bars=1, **_kw(a, b))
    words = cr.words_in_clip(r.metadata, a["words"], "a")
    q = dj_qa.evaluate_render(r.audio, r.metadata, tracker=None, words_a=words, **mx.qa_inputs(r))
    assert q["checks"]["duration"]["status"] == "pass"
    assert q["checks"]["true_peak"]["status"] == "pass"
    assert q["checks"]["vocal_clash"]["status"] == "pass"
    assert q["checks"]["double_bass"]["status"] in ("pass", "not_applicable")
    vc = q["checks"]["verse_cut"]
    assert vc["status"] == "pass" and vc["coverage"] == "measured"


def test_no_line_end_raises(tracks):
    a, b = tracks
    v, w = sung(a, range(0, 30), (523.25,), beats_in_bar=(0, 1, 2, 3))  # sings every beat: no line end
    kw = _kw(a, b)
    kw.update(vocals_a=v, words_a=w)
    with pytest.raises(dj.DJRenderError):
        mx.vocal_loop(a["audio"] + v, b["mix"], key_a="C major", key_b="D major", **kw)
    with pytest.raises(dj.DJRenderError):
        mx.loop_schedule("halving", 1)


def test_listening_page_reference_is_not_votable(tmp_path, monkeypatch):
    """build_listening_artifact: an item with role 'riferimento' is played but has no vote form and no preference option."""
    import hashlib
    import json

    import soundfile as sf

    from tools.build_listening_artifact import build

    repo = tmp_path / "checkout"
    clips = repo / "runs" / "syn_mashup" / "clips"
    clips.mkdir(parents=True)
    monkeypatch.setenv("MUSICMIX_ROOT", str(repo))
    items = []
    for cid, role in (("m01_a", "versione"), ("m01_b", "versione"), ("m01_rif", "riferimento")):
        p = clips / f"{cid}.wav"
        sf.write(str(p), 0.1 * np.sin(2 * np.pi * 440 * np.arange(int(30 * SR)) / SR), SR, subtype="PCM_16")
        items.append({"id": cid, "role": role, "kind": "vocal_loop", "label": "riferimento: set v3" if role != "versione" else "loop",
                      "audio_path": f"runs/syn_mashup/clips/{cid}.wav", "duration_s": 30.0, "transition_start_s": 10.0, "switch_s": 12.0,
                      "transition_end_s": 20.0, "qa": {"verdict": "pass", "warn_reasons": []},
                      "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    man = repo / "runs" / "syn_mashup" / "review_manifest.json"
    man.write_text(json.dumps({"schema_version": "musicmix.dj-review.1", "run": "syn_mashup",
                               "groups": [{"id": "m01", "title": "A → B", "items": items}]}), encoding="utf8")
    out = tmp_path / "out"
    listing = build(man, out, encoder="wav16", blind=True, log=lambda *_: None)
    page = (out / "index.html").read_text(encoding="utf8")
    assert len(listing["included"]) == 3
    assert 'class="version reference" data-clip="m01_rif"' in page and "non si vota" in page
    assert 'value="m01_rif"' not in page and 'name="score-m01_rif"' not in page
    assert 'name="pref-m01" value="m01_a"' in page and "2 versioni dello stesso cambio + 1 riferimento non votabile" in page
    data = json.loads(page.split('<script id="clip-data" type="application/json">')[1].split("</script>")[0])
    assert [x["id"] for x in data["groups"][0]["items"]] == ["m01_a", "m01_b"]


@needs_rb
def test_b_map_hides_gated_vocal_from_qa(tracks):
    """B's words that the gate keeps closed are not 'heard' for dj_qa (b_entry): B's cue inside its sung bars."""
    a, b = tracks
    r = mx.vocal_loop(a["mix"], b["mix"], key_a="C major", key_b="D major", mode="beat", loop_bars=1, **_kw(a, b, ia=12, jb=9))
    m = _common(r, "vocal_loop")
    seg = [x for x in m["b_map"] if x["out_start"] >= m["switch"]["output_sample"]]
    assert seg[0]["vocals"] is False and seg[-1]["vocals"] is True
    heard = cr.words_in_clip(m, b["words"], "b")
    gate = m["creative"]["b_vocal_gate"]["open_clip_sample"] / SR
    assert heard and min(w["start_unclipped_s"] for w in heard) >= gate - 1e-6
    words_a = cr.words_in_clip(m, a["words"], "a")
    # B's vocal entry is declared at the gate (dj_qa.6 b_entry): all of B's words on its own timeline, the word before the
    # gate included, so the QA also checks that B's phrase is not already under way there
    assert m["b_entry"]["audible_output_sample"] == m["creative"]["b_vocal_gate"]["open_clip_sample"]
    t_sw, sb = m["switch"]["output_sample"], m["switch"]["b_source_sample"]
    all_b = [{"start_s": (t_sw + w["start_sample"] - sb) / SR, "end_s": (t_sw + w["end_sample"] - sb) / SR, "text": w["text"]} for w in b["words"]]
    q = dj_qa.evaluate_render(r.audio, m, tracker=None, words_a=words_a, words_b=all_b, **mx.qa_inputs(r))
    be = q["checks"]["b_entry"]
    assert be["status"] == "pass" and be["coverage"] == "measured" and be["b_audible_source"].startswith("meta.b_entry")


def test_sound_affinity():
    assert mx.sound_affinity("cold", "killer")["type"] == "alliteration"
    assert mx.sound_affinity("knife", "nasty")["type"] == "alliteration" and mx.sound_affinity("knife", "killer") is None
    assert mx.sound_affinity("aaah", "ooh")["type"] == "vocalise"
    assert mx.sound_affinity("night", "light")["type"] == "rhyme"
    assert mx.sound_affinity("the", "this") is None
    wa = [{"start_sample": 0, "end_sample": 4800, "text": "cold", "confidence": 0.9}]
    wb = [{"start_sample": 10 * SR, "end_sample": 10 * SR + 4800, "text": "killa", "confidence": 0.9}]
    links = mx.find_sound_links(wa, wb, [0.5], [10.0], sr=SR, bar_a_s=2.0, bar_b_s=2.0)
    assert links and links[0]["match"]["type"] == "alliteration"


def test_auto_shift_on_a_fragment_shorter_than_two_beats(tracks):
    a, b = tracks
    cfg = mx.MashupConfig()
    s0 = int(round(a["beats"][0] * SR))
    m = mx._auto_shift(a["vocals"], a["beats"], (s0, s0 + int(0.3 * SR)), b["audio"], b["beats"], (2 * SR, 6 * SR), 1.0, SR, cfg,
                       keys=("C major", "D major"))
    assert m["status"] == "ok" and m["mode"].startswith("global") and m["best_shift"] in m["key_guard"]["compatible_shifts"]
