"""Synthetic checks for the creative DJ techniques (dj_creative).

Click tracks as in test_dj_effects (downbeat 3 kHz loud, offbeats 1.5 kHz soft,
sub-bass sine: A 55 Hz, B 82 Hz), A at 120 BPM and B at 123 BPM (stretch 2.4%),
plus synthetic "vocal stems" (tone bursts 700 Hz / 1050 Hz on chosen bars) that
are added to the mix, so mix - stem is the exact instrumental.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

pytest.importorskip("pyloudnorm")
pytest.importorskip("soundfile")
from scipy.signal import butter, sosfiltfilt  # noqa: E402

from musicmix import dj_creative as cr  # noqa: E402
from musicmix import dj_qa  # noqa: E402
from musicmix import dj_render as dj  # noqa: E402

SR = 48000
HAS_RB = shutil.which("rubberband") is not None


def click_track(bpm, dur, bass_hz, *, start=0.5, bass_level=0.3, gain=1.0):
    n = int(dur * SR)
    x = np.zeros(n)
    period = 60.0 / bpm
    burst = np.exp(-np.arange(480) / 60.0)
    beats, t, k = [], start, 0
    while t < dur - 0.2:
        i = int(round(t * SR))
        f, amp = (3000.0, 1.0) if k % 4 == 0 else (1500.0, 0.35)
        x[i : i + 480] += amp * burst * np.sin(2 * np.pi * f * np.arange(480) / SR)
        beats.append(t)
        t += period
        k += 1
    x += bass_level * np.sin(2 * np.pi * bass_hz * np.arange(n) / SR)
    beats = np.array(beats)
    return {"audio": x * gain, "beats": beats, "downbeats": beats[::4]}


def vocal_stem(track, bars, freq, level=0.25):
    """Tone bursts (one per beat, 70% of the beat) on the given bar indices."""
    n = len(track["audio"])
    v = np.zeros(n)
    beats = track["beats"]
    period = float(np.median(np.diff(beats)))
    for b in bars:
        for k in range(4):
            i = 4 * b + k
            if i >= len(beats):
                break
            s = int(round(beats[i] * SR))
            m = int(0.7 * period * SR)
            t = np.arange(m) / SR
            env = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)
            v[s : s + m] += level * env * (np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * 1.5 * freq * t))
    return v


@pytest.fixture(scope="module")
def tracks():
    a = click_track(120, 60, 55, gain=0.4)
    b = click_track(123, 60, 82, gain=1.2)
    a["vocals"] = vocal_stem(a, list(range(0, 12)), 700.0)  # A sings bars 0..11, instrumental from bar 12
    b["vocals"] = vocal_stem(b, list(range(4, 14)), 600.0, level=0.6)  # B sings from bar 4
    a["mix"] = a["audio"] + a["vocals"]
    b["mix"] = b["audio"] + b["vocals"]
    return a, b


def _kw(a, b, ia=12, jb=4, vocals=True):
    kw = dict(beats_a=a["beats"], downbeats_a=a["downbeats"], beats_b=b["beats"], downbeats_b=b["downbeats"],
              cue_out_a=a["downbeats"][ia], cue_in_b=b["downbeats"][jb], sample_rate=SR)
    if vocals:
        kw.update(vocals_a=a["vocals"], vocals_b=b["vocals"])
    return kw


def _band(x, lo, hi):
    mono = x.mean(axis=1) if x.ndim == 2 else x
    sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    return sosfiltfilt(sos, mono)


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2))) if len(x) else 0.0


def _common(r, technique):
    m = r.metadata
    assert m["kind"] == technique
    assert m["schema_version"] == cr.DJ_CREATIVE_VERSION
    ov, sw = m["overlap"], m["switch"]
    assert ov["output_start_sample"] == int(round(10.0 * SR))
    assert ov["output_start_sample"] <= sw["output_sample"] <= ov["output_end_sample"]
    assert m["clip"]["matches_rule"]
    assert len(r.audio) == ov["output_end_sample"] + int(round(10.0 * SR))
    assert m["loudness"]["output_true_peak_dbtp"] <= -1.0 + 1e-6
    assert m["creative"]["technique"] == technique
    return m


# --------------------------------------------------------------- keys / words


def test_auto_key_shift_and_relations():
    assert cr.auto_key_shift("A minor", "A minor") == 0
    assert cr.auto_key_shift("C major", "A minor") == 0  # relative
    assert cr.auto_key_shift("C major", "C# major") == -1  # one semitone down makes it the same key
    assert cr.auto_key_shift("C major", "F# major") == 1  # G major: a fifth above C
    assert cr.auto_key_shift("C major", "D# minor") is None  # E minor / D minor are only diagonal
    assert cr.key_relation("C major", "C# major", -1) == "same"
    assert cr.auto_key_shift(None, "C major") is None


def test_bridge_chord_uses_common_tones():
    ch = cr.bridge_chord("C major", "A minor")
    assert set(ch["pitch_classes"]) <= set(ch["common_pitch_classes"])
    assert ch["label"] == "tonic triad of B" and ch["names"] == ["A", "C", "E"]
    ch2 = cr.bridge_chord("C major", "G major")  # B's tonic triad G-B-D is in C major
    assert ch2["names"] == ["G", "B", "D"]
    assert all(55 <= m <= 80 for m in ch2["midi"])


def test_word_matching():
    assert cr.word_match("Creek", "creek")["type"] == "exact"
    assert cr.word_match("the", "the") is None  # function word
    assert cr.word_match("creek", "seek")["type"] == "rhyme"
    assert cr.word_match("night", "light")["type"] == "rhyme"
    assert cr.soundex("Robert") == "R163" and cr.soundex("Rupert") == "R163"
    assert cr.word_match("money", "honey")["type"] == "rhyme"
    assert cr.word_match("cat", "dog") is None
    assert cr.word_match("we're", "answer") is None  # weak "-er" ending needs a two-syllable rhyme
    assert cr.word_match("Japan!", "man")["type"] == "rhyme"


def test_find_lyric_links_window():
    words_a = [{"start_sample": int(9.0 * SR), "end_sample": int(9.3 * SR), "text": "go", "confidence": 0.9},
               {"start_sample": int(10.5 * SR), "end_sample": int(11.0 * SR), "text": "money", "confidence": 0.9}]
    words_b = [{"start_sample": int(20.1 * SR), "end_sample": int(20.4 * SR), "text": "honey", "confidence": 0.8},
               {"start_sample": int(30.0 * SR), "end_sample": int(30.4 * SR), "text": "money", "confidence": 0.9}]
    links = cr.find_lyric_links(words_a, words_b, [12.0], [20.0], sr=SR, bar_a_s=2.0, bar_b_s=2.0)
    assert len(links) == 1
    assert links[0]["match"]["type"] == "rhyme" and links[0]["b_word"]["text"] == "honey"  # 30 s is outside B's window


def test_roll_events_halve_each_bar():
    beats = np.arange(0, 8) * 0.5
    ev = cr.roll_events(beats, 0.0, 4.0, (0.5, 0.25), [0.0, 2.0])
    assert len(ev) == 4 * 2 + 4 * 4
    assert np.allclose(np.diff([t for t, _ in ev[:8]]), 0.25)
    assert np.allclose(np.diff([t for t, _ in ev[8:]]), 0.125)


# ---------------------------------------------------------- overlap techniques


@pytest.mark.skipif(not HAS_RB, reason="rubberband CLI needed for beatmatching")
def test_loop_extend_repeats_instrumental_loop(tracks):
    a, b = tracks
    # loop the bars before bar 12: bar 11 has A's vocal -> auto source must be the instrumental
    r = cr.loop_extend(a["mix"], b["mix"], loop_bars=1, bars=4, return_decks=True, **_kw(a, b, ia=12, jb=0))
    m = _common(r, "loop_extend")
    assert m["creative"]["params"]["source"] == "instrumental"
    d = r.decks
    t_s = m["overlap"]["output_start_sample"]
    L = m["creative"]["loop"]["length_samples"]
    assert L == int(round(a["downbeats"][12] * SR)) - int(round(a["downbeats"][11] * SR))
    raw = d["a_raw"].mean(axis=1)
    xf = int(0.012 * SR)
    seg1, seg2 = raw[t_s + xf : t_s + L - xf], raw[t_s + L + xf : t_s + 2 * L - xf]
    assert np.corrcoef(seg1, seg2)[0, 1] > 0.999  # identical repeats
    # A's vocal (700 Hz bursts) is gone from the loop, and nobody's vocal overlaps (B starts singing at bar 4)
    assert _rms(_band(raw[t_s : t_s + 2 * L], 650, 750)) < 0.05 * _rms(_band(raw[t_s - 2 * L : t_s], 650, 750))
    assert m["vocals"]["both_heard_s"] < 0.5
    # B is beatmatched onto the loop grid
    assert m["creative"]["beatmatch_alignment"]["max_abs_error_ms"] < 10.0
    assert m["stretch"]["applied"] and abs(m["stretch"]["stretch_pct"] + 100 * (1 - 120 / 123)) < 0.5
    # a_map: before the loop + repeats without vocals
    assert m["a_map"][0]["out_end"] == t_s and not m["a_map"][1]["vocals"]


@pytest.mark.skipif(not HAS_RB, reason="rubberband CLI needed for beatmatching")
def test_acapella_over_one_voice_and_hard_swap(tracks):
    a, b = tracks
    r = cr.acapella_over(a["mix"], b["mix"], bars=4, return_decks=True, **_kw(a, b, ia=8, jb=4))
    m = _common(r, "acapella_over")
    d = r.decks
    t_s, t_sw = m["overlap"]["output_start_sample"], m["switch"]["output_sample"]
    assert t_sw - t_s == int(round((a["downbeats"][12] - a["downbeats"][8]) * SR))
    deck_b = d["b"]
    # before the swap deck B has no instrumental (B's 82 Hz bass is absent), after it is there
    pre_low = _rms(_band(deck_b[t_s + SR // 2 : t_sw - SR // 4], 70, 95))
    post_low = _rms(_band(deck_b[t_sw + SR // 2 : t_sw + 2 * SR], 70, 95))
    assert pre_low < 0.05 * post_low
    # B's vocal is in deck B during the acapella, A's vocal is muted from t_s
    assert _rms(_band(deck_b[t_s + SR // 2 : t_sw], 550, 650)) > 0.0
    assert _rms(d["vocals_a_heard"][t_s + SR // 10 :]) < 1e-6
    assert m["vocals"]["both_heard_s"] < 0.2
    # A's bass gone after the swap, A fully out at the end
    assert _rms(_band(d["a"][t_sw + SR // 10 : t_sw + SR], 45, 65)) < 0.05 * _rms(_band(d["a"][t_s : t_sw], 45, 65))
    assert not np.any(d["a"][m["overlap"]["output_end_sample"] :])
    assert "a_vocal_mute_guard" in m["creative"]


@pytest.mark.skipif(not HAS_RB, reason="rubberband CLI needed")
def test_acapella_key_shift_moves_the_vocal_up_a_semitone(tracks):
    a, b = tracks
    r = cr.acapella_over(a["mix"], b["mix"], bars=4, key_shift_b=1, return_decks=True, **_kw(a, b, ia=8, jb=4))
    m = r.metadata
    assert m["stretch"]["key_shift_b"]["applied"] and m["stretch"]["key_shift_b"]["semitones"] == 1
    t_s, t_sw = m["overlap"]["output_start_sample"], m["switch"]["output_sample"]
    v = r.decks["vocals_b_heard"][t_s + SR // 2 : t_sw].mean(axis=1)
    spec = np.abs(np.fft.rfft(v * np.hanning(len(v))))
    f = np.fft.rfftfreq(len(v), 1 / SR)
    sel = (f > 400) & (f < 800)
    peak = f[sel][np.argmax(spec[sel])]
    assert abs(peak / (600.0 * 2 ** (1 / 12)) - 1) < 0.01


@pytest.mark.skipif(not HAS_RB, reason="rubberband CLI needed")
def test_vocal_swap_bass_swap_on_the_downbeat(tracks):
    a, b = tracks
    r = cr.vocal_swap(a["mix"], b["mix"], bars=4, out_bars=2, return_decks=True, **_kw(a, b, ia=8, jb=4))
    m = _common(r, "vocal_swap")
    d = r.decks
    t_s, t_sw, t_e = m["overlap"]["output_start_sample"], m["switch"]["output_sample"], m["overlap"]["output_end_sample"]
    assert (t_e - t_sw) == int(round((a["downbeats"][14] - a["downbeats"][12]) * SR))
    low_a_after = _rms(_band(d["a"][t_sw + SR // 10 : t_e], 45, 65))
    low_a_before = _rms(_band(d["a"][t_s : t_sw - SR // 10], 45, 65))
    low_b_before = _rms(_band(d["b"][t_s + SR // 10 : t_sw - SR // 10], 70, 95))
    low_b_after = _rms(_band(d["b"][t_sw + SR // 10 : t_e], 70, 95))
    assert low_a_after < 0.05 * low_a_before and low_b_before < 0.05 * low_b_after
    assert m["vocals"]["both_heard_s"] < 0.2


# ------------------------------------------------------- entry techniques


def test_build_into_chorus_roll_riser_and_entry(tracks):
    a, b = tracks
    r = cr.build_into_chorus(a["mix"], b["mix"], key_b="A minor", build_bars=2, snare="synth", return_decks=True,
                             **_kw(a, b, ia=12, jb=4))
    m = _common(r, "build_into_chorus")
    assert m["exit_effect"] == "filter"
    t_s, t_sw = m["overlap"]["output_start_sample"], m["switch"]["output_sample"]
    assert t_sw - t_s == int(round((a["downbeats"][12] - a["downbeats"][10]) * SR))
    assert m["creative"]["snare_roll"]["events"] == 4 * 2 + 4 * 4
    # nothing of A after the drop, B starts on the downbeat
    assert not np.any(r.decks["a"][t_sw:])
    assert not np.any(r.decks["b"][: t_sw - int(0.011 * SR)])
    # riser lands on the tonic (A): strongest partial in the last 150 ms near 220/440 Hz multiples
    ris = r.decks["overlay_parts"]["riser"][t_sw - int(0.15 * SR) - int(0.006 * SR) : t_sw - int(0.006 * SR)].mean(axis=1)
    spec = np.abs(np.fft.rfft(ris * np.hanning(len(ris))))
    f = np.fft.rfftfreq(len(ris), 1 / SR)
    sel = (f > 150) & (f < 500)
    peak = f[sel][np.argmax(spec[sel])]
    assert min(abs(peak - 220.0), abs(peak - 440.0), abs(peak - 330.0)) < 15.0
    # overlays kept under the margin used by dj_qa's region-peak check
    tr = m["loudness"]["overlay_trim"]
    assert tr["status"] in ("within margin", "trimmed")


def test_build_extract_falls_back_to_synth(tracks):
    a, b = tracks
    r = cr.build_into_chorus(a["mix"], b["mix"], key_b="A minor", build_bars=1, snare="auto", riser=False, stutter_bars=1,
                             **_kw(a, b, ia=12, jb=4))
    sn = r.metadata["creative"]["snare_roll"]
    assert sn["source"] in ("synthesized (noise 1-8 kHz + 190 Hz body)", "extracted from A")
    assert r.metadata["creative"]["stutter"]["slice_s_per_beat"][0] == pytest.approx(0.25, abs=1e-3)


def test_note_bridge_pad_covers_the_junction(tracks):
    a, b = tracks
    r = cr.note_bridge(a["mix"], b["mix"], key_a="C major", key_b="A minor", gap_bars=1, return_decks=True, **_kw(a, b, ia=12, jb=4))
    m = _common(r, "note_bridge")
    d = r.decks
    t_sw = m["switch"]["output_sample"]
    t_x = t_sw - int(round((a["downbeats"][13] - a["downbeats"][12]) * SR))
    pad = d["overlay"]
    assert _rms(pad[t_x : t_sw]) > 0  # pad alone in the gap bar
    assert not np.any(d["a"][t_x:]) and not np.any(d["b"][: t_sw - int(0.011 * SR)])
    assert _rms(pad[m["overlap"]["output_end_sample"] :]) < 1e-6
    assert m["creative"]["pad"]["chord"]["names"] == ["A", "C", "E"]


def test_note_bridge_freeze_source(tracks):
    a, b = tracks
    r = cr.note_bridge(a["mix"], b["mix"], key_a="C major", key_b="A minor", source="freeze_a", **_kw(a, b, ia=12, jb=4))
    assert r.metadata["creative"]["pad"]["source"] == "freeze_a"


@pytest.mark.parametrize("fn,effect", [(cr.echo_freeze, "echo"), (cr.reverb_wash, "reverb")])
def test_freeze_and_reverb_tails_under_b(tracks, fn, effect):
    a, b = tracks
    r = fn(a["mix"], b["mix"], return_decks=True, **_kw(a, b, ia=12, jb=4))
    m = _common(r, fn.__name__)
    assert m["exit_effect"] == effect
    t_s, t_sw, t_e = m["overlap"]["output_start_sample"], m["switch"]["output_sample"], m["overlap"]["output_end_sample"]
    ov = r.decks["overlay"]
    assert _rms(ov[t_sw - SR // 2 : t_sw]) > 3 * _rms(ov[t_sw + SR // 2 : t_sw + SR])  # tail ducked under B
    assert _rms(ov[t_e + SR // 10 :]) < 1e-6
    if effect == "echo":
        assert not np.any(r.decks["a"][t_s:])  # dry A cut where the freeze starts
    else:
        assert not np.any(r.decks["a"][t_sw:])


def test_backspin_reverses_and_tape_stop_slows(tracks):
    a, b = tracks
    r = cr.backspin(a["mix"], b["mix"], spin_beats=2, return_decks=True, **_kw(a, b, ia=12, jb=4))
    m = _common(r, "backspin")
    sp = m["creative"]["speed_profile"]
    assert sp["min"] < -2.5 and sp["max"] == pytest.approx(1.0)
    assert m["switch"]["output_sample"] - m["overlap"]["output_start_sample"] == int(round(2 * 0.5 * SR))
    assert not np.any(r.decks["a"][m["switch"]["output_sample"] :])
    r2 = cr.tape_stop(a["mix"], b["mix"], stop_beats=2, **_kw(a, b, ia=12, jb=4))
    sp2 = r2.metadata["creative"]["speed_profile"]
    assert sp2["max"] == pytest.approx(1.0) and sp2["min"] >= 0.0
    assert r2.metadata["a_map"][1]["rate"] is None  # variable speed: words not mapped


def test_lyric_link_echo_lands_on_b_word(tracks):
    a, b = tracks
    beat = 0.5
    X = a["downbeats"][12]
    a_word = {"start_sample": int(round((X - 2 * beat) * SR)), "end_sample": int(round((X - 1.7 * beat) * SR)), "text": "money", "confidence": 0.9}
    yb = b["downbeats"][4]
    b_word = {"start_sample": int(round(b["beats"][17] * SR)), "end_sample": int(round((b["beats"][17] + 0.3) * SR)), "text": "honey", "confidence": 0.9}
    r = cr.lyric_link(a["mix"], b["mix"], a_word=a_word, b_word=b_word, return_decks=True, **_kw(a, b, ia=12, jb=4))
    m = _common(r, "lyric_link")
    link = m["creative"]["link"]
    gap = link["b_word_output_sample"] - link["a_word_output_sample"]
    assert abs(gap - link["repeats_to_b_word"] * link["echo_period_samples"]) <= link["repeats_to_b_word"]
    assert abs(link["delay_deviation_from_musical_value"]) <= 0.06
    assert m["exit_effect"] == "echo"
    assert link["b_word_output_sample"] == m["switch"]["output_sample"] + int(round((b["beats"][17] - yb) * SR))
    # the echo is silent from B's word on
    echo = r.decks["overlay_parts"]["word_echo"]
    assert _rms(echo[link["b_word_output_sample"] + int(0.05 * SR) :]) < 1e-6


def test_lyric_link_rejects_far_words(tracks):
    a, b = tracks
    X = a["downbeats"][12]
    w = {"start_sample": int((X - 20) * SR), "end_sample": int((X - 19.7) * SR), "text": "x"}
    with pytest.raises(dj.DJRenderError):
        cr.lyric_link(a["mix"], b["mix"], a_word=w, b_word=w, **_kw(a, b, ia=12, jb=4))


# ------------------------------------------------------------ maps and QA


def test_words_in_clip_follow_the_maps():
    meta = {"sample_rate": SR, "a_map": [{"out_start": 0, "out_end": 100, "src_start": 1000, "rate": 1.0, "vocals": True},
                                         {"out_start": 100, "out_end": 200, "src_start": 1000, "rate": 1.0, "vocals": False}]}
    words = [{"start_sample": 1010, "end_sample": 1050, "text": "yo"}, {"start_sample": 1150, "end_sample": 1160, "text": "no"}]
    out = cr.words_in_clip(meta, words)
    assert [w["text"] for w in out] == ["yo"] and out[0]["start_s"] == pytest.approx(10 / SR)


def test_dj_qa_runs_on_creative_renders(tracks):
    a, b = tracks
    r = cr.echo_freeze(a["mix"], b["mix"], return_decks=True, **_kw(a, b, ia=12, jb=4))
    q = dj_qa.evaluate_render(r.audio, r.metadata, tracker=None, **cr.qa_inputs(r))
    assert q["checks"]["duration"]["status"] == "pass"
    assert q["checks"]["true_peak"]["status"] == "pass"
    assert q["checks"]["vocal_clash"]["status"] == "pass"
    # dj_qa.5: the exit through the freeze is checked where A's dry playback stops. Without ASR words the heard stem,
    # active up to the stop, cannot show whether the word continues in the source: unavailable, not a silent pass.
    vc = q["checks"]["verse_cut"]
    assert vc["status"] == "skipped" and vc["coverage"] == "unavailable" and vc["a_dry_end_s"] == pytest.approx(10.0)
    words = cr.words_in_clip(r.metadata, _burst_words(a), "a")
    vc = dj_qa.evaluate_render(r.audio, r.metadata, tracker=None, words_a=words, **cr.qa_inputs(r))["checks"]["verse_cut"]
    assert vc["status"] == "pass" and vc["coverage"] == "measured"  # the freeze starts on a beat, between two bursts


def _burst_words(track, bars=range(0, 12)):
    """'Words' = the synthetic vocal bursts of vocal_stem (70 % of each beat), in source samples."""
    beats = track["beats"]
    period = float(np.median(np.diff(beats)))
    out = []
    for bar in bars:
        for k in range(4):
            i = 4 * bar + k
            if i < len(beats):
                s0 = int(round(beats[i] * SR))
                out.append({"start_sample": s0, "end_sample": s0 + int(0.7 * period * SR), "text": f"b{i}"})
    return out


def test_bad_parameters_raise(tracks):
    a, b = tracks
    with pytest.raises(dj.DJRenderError):
        cr.loop_extend(a["mix"], b["mix"], loop_bars=3, **_kw(a, b))
    with pytest.raises(dj.DJRenderError):
        cr.render_creative("scratch", a["mix"], b["mix"])
    with pytest.raises(dj.DJRenderError):
        cr.acapella_over(a["mix"], b["mix"], **_kw(a, b, vocals=False))


@pytest.mark.skipif(not HAS_RB, reason="rubberband CLI needed")
def test_phase_nudge_moves_b_earlier(tracks):
    a, b = tracks
    r0 = cr.vocal_swap(a["mix"], b["mix"], bars=4, return_decks=True, **_kw(a, b, ia=8, jb=4))
    r1 = cr.vocal_swap(a["mix"], b["mix"], bars=4, phase_nudge_s=0.010, return_decks=True, **_kw(a, b, ia=8, jb=4))
    x0, x1 = r0.decks["b_stretched_raw"].mean(axis=1), r1.decks["b_stretched_raw"].mean(axis=1)
    k = int(0.010 * SR)
    assert np.allclose(x1[: len(x1) - k], x0[k:], atol=1e-9)
    assert r1.metadata["stretch"]["phase_nudge"]["samples"] == k
    with pytest.raises(dj.DJRenderError):
        cr.vocal_swap(a["mix"], b["mix"], bars=4, phase_nudge_s=0.2, **_kw(a, b, ia=8, jb=4))


def test_lyric_link_delays_b_when_words_are_too_close(tracks):
    a, b = tracks
    X = a["downbeats"][12]
    a_word = {"start_sample": int(round((X - 0.25) * SR)), "end_sample": int(round((X - 0.1) * SR)), "text": "man", "confidence": 0.9}
    b_word = {"start_sample": int(round(b["downbeats"][4] * SR)), "end_sample": int(round((b["downbeats"][4] + 0.3) * SR)), "text": "Japan", "confidence": 0.9}
    r = cr.lyric_link(a["mix"], b["mix"], a_word=a_word, b_word=b_word, return_decks=True, **_kw(a, b, ia=12, jb=4))
    m = r.metadata
    link = m["creative"]["link"]
    assert m["creative"]["params"]["entry_delay_bars"] >= 1
    assert link["repeats_to_b_word"] >= 2 and link["echo_repeats"]
    x_out = int(round(10.0 * SR)) + int(round(X * SR)) - int(round(a["downbeats"][11] * SR))
    assert not np.any(r.decks["a"][x_out:])  # A's dry exits on its downbeat, B later
    assert m["switch"]["output_sample"] > x_out


def test_inplace_inputs_gives_the_same_render(tracks):
    a, b = tracks
    r0 = cr.echo_freeze(a["mix"], b["mix"], **_kw(a, b, ia=12, jb=4))
    am, bm = np.stack([a["mix"]] * 2, 1).copy(), np.stack([b["mix"]] * 2, 1).copy()
    va, vb = np.stack([a["vocals"]] * 2, 1).copy(), np.stack([b["vocals"]] * 2, 1).copy()
    kw = _kw(a, b, ia=12, jb=4)
    kw.update(vocals_a=va, vocals_b=vb, config=cr.DJCreativeConfig(inplace_inputs=True))
    r1 = cr.echo_freeze(am, bm, **kw)
    assert np.allclose(r0.audio, r1.audio, atol=1e-6)
    assert not np.allclose(am[:, 0], a["mix"])  # the caller's array was normalized in place


def test_spin_fader_makes_a_gradual_exit(tracks):
    a, b = tracks
    r = cr.backspin(a["mix"], b["mix"], spin_beats=2, fader_out_beats=1.0, return_decks=True, **_kw(a, b, ia=12, jb=4))
    m = r.metadata
    t_s, t_sw = m["overlap"]["output_start_sample"], m["switch"]["output_sample"]
    d = r.decks["a"][t_s:t_sw].mean(axis=1)
    half = len(d) // 2
    assert _rms(d[-len(d) // 8 :]) < 0.3 * _rms(d[:half])
    words = cr.words_in_clip(m, _burst_words(a), "a")
    q = dj_qa.evaluate_render(r.audio, m, tracker=None, words_a=words, **cr.qa_inputs(r))
    # dj_qa.5: the spin stops A's dry playback at its start (a_map), checked with the words; the fader makes the
    # exit gradual but the spin itself interrupts the playback
    assert q["checks"]["verse_cut"]["status"] == "pass" and q["checks"]["verse_cut"]["coverage"] == "measured"


# ------------------------------- stretched B: source samples for splicing and QA (fix 21/10/2026, set v3)


@pytest.mark.skipif(not HAS_RB, reason="rubberband CLI needed for the per-bar map")
@pytest.mark.parametrize("technique,kw", [
    ("loop_extend", dict(loop_bars=1, bars=4, ia=12, jb=0)),
    ("acapella_over", dict(bars=4, ia=8, jb=4)),
    ("vocal_swap", dict(bars=4, out_bars=2, ia=8, jb=4)),
])
def test_stretched_b_declares_its_source_samples(tracks, technique, kw):
    """dj_set.clip_geometry raised KeyError (overlap.b_source_end_sample missing, b_map.src_start None)."""
    from musicmix import dj_set as ds

    a, b = tracks
    ia, jb = kw.pop("ia"), kw.pop("jb")
    r = cr.render_creative(technique, a["mix"], b["mix"], return_decks=True, **kw, **_kw(a, b, ia=ia, jb=jb))
    m = r.metadata
    n = len(r.audio)
    ov, st = m["overlap"], m["stretch"]
    assert st["applied"] and st["time_map"]["kind"] in ("piecewise", "linear")
    assert ov["b_source_end_sample"] is not None
    assert all(seg["src_start"] is not None for seg in m["b_map"])
    geo = ds.clip_geometry(m, n)  # no KeyError any more
    assert ds.b_src_of_clip(geo, ov["output_end_sample"]) == pytest.approx(ov["b_source_end_sample"])
    # after the overlap the straight line of clip_geometry and the renderer's map agree (B stays at the mean ratio)
    later = ov["output_end_sample"] + 2 * SR
    assert ds.b_src_of_clip(geo, later) == pytest.approx(dj_qa.b_source_position(m, n)[later], abs=2.0)
    # the switch is consistent with the map, and the stretched B deck really plays B's downbeat there
    assert dj_qa.b_source_position(m, n)[m["switch"]["output_sample"]] == pytest.approx(m["switch"]["b_source_sample"], abs=2.0)
    jsw = int(np.argmin(np.abs(b["downbeats"] * SR - m["switch"]["b_source_sample"])))
    out_db = dj_qa.b_output_position(m, [b["downbeats"][jsw + 1] * SR])[0]
    env = np.abs(_band(r.decks["b_stretched_raw"], 2500, 3500))
    w = int(0.03 * SR)
    seg = env[int(out_db) - w : int(out_db) + w]
    assert abs(int(np.argmax(seg > 0.5 * seg.max())) - w) < int(0.005 * SR)
    # B's words follow the map (no longer skipped on stretched segments)
    words_b = [{"start_sample": int(b["downbeats"][jsw + 1] * SR), "end_sample": int(b["downbeats"][jsw + 1] * SR) + 4800, "text": "b"}]
    wb = cr.words_in_clip(m, words_b, "b")
    assert len(wb) == 1 and wb[0]["start_s"] * SR == pytest.approx(out_db, abs=1.0)


def test_build_stutter_is_in_the_a_map_and_qa_checks_the_word(tracks):
    a, b = tracks
    r = cr.build_into_chorus(a["mix"], b["mix"], key_b="A minor", build_bars=2, snare="synth", stutter_bars=1, return_decks=True,
                             **_kw(a, b, ia=11, jb=4))
    m = r.metadata
    st = m["creative"]["stutter"]
    b0 = st["bar_output"][0]
    beat = int(round(0.5 * SR))
    dry = dj_qa.a_dry_end(m)
    assert dry["sample"] == b0 + beat // 2  # the first slice (half a beat) continues A, then it repeats
    # a word of A crossing the end of A's continuous playback: cut; QA says so (A sings bars 0..11 in the fixture)
    off = m["overlap"]["a_source_start_sample"] - m["overlap"]["output_start_sample"]
    src = dry["sample"] + off
    words = [{"start_sample": src - 4800, "end_sample": src + 9600, "text": "tagliata"}]
    clip_words = cr.words_in_clip(m, words, "a")
    assert any(w["clipped"] and w["end_unclipped_s"] * SR == pytest.approx(src + 9600 - off) for w in clip_words)
    q = dj_qa.evaluate_render(r.audio, m, tracker=None, words_a=clip_words, **cr.qa_inputs(r))
    vc = q["checks"]["verse_cut"]
    assert vc["status"] == "fail" and vc["a_dry_end_s"] * SR == pytest.approx(dry["sample"])
    words_ok = [{"start_sample": src - 9600, "end_sample": src - 4800, "text": "prima"}]
    q = dj_qa.evaluate_render(r.audio, m, tracker=None, words_a=cr.words_in_clip(m, words_ok, "a"), **cr.qa_inputs(r))
    assert q["checks"]["verse_cut"]["status"] == "pass"
