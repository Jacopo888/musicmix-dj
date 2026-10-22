"""Tests for musicmix.dj_phrase: line-end probability, lexicon, exits/entries with pickup, render remix."""
import numpy as np
import pytest

from musicmix import dj_phrase as P
from musicmix import dj_select as djs

SR = P.SR
BPM = 90.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT


# ------------------------------------------------------------------ helpers
def _track(n_bars=40, slug="t", vocal=1.0, sections=((0, "intro"), (8, "verse"), (24, "chorus"))):
    down = np.arange(n_bars) * BAR + 0.5
    secs = [{"label": lab, "start_s": float(down[b]), "end_s": 0.0, "bar_index": b, "downbeat_s": float(down[b]),
             "snap_error_bars": 0.0, "loose": False, "support": ["x"], "source": "test"} for b, lab in sections]
    v = np.full(n_bars, vocal)
    return {"slug": slug, "duration_s": float(down[-1] + BAR),
            "tempo": {"bpm": BPM, "period_s": BEAT, "phase_s": 0.5, "bar_offset_beats": 0, "confidence": "high"},
            "grid": {"downbeats_s": down.tolist()}, "key": {"camelot": "8A", "confidence": 0.9},
            "sections": secs, "phrases": djs.phrase_table(down, secs),
            "bars": {"vocal": v.tolist(), "vocal_last_beat": v.tolist(), "energy_db_rel": np.zeros(n_bars).tolist()}}


def _w(text, on_s, off_s):
    return {"start_sample": int(on_s * SR), "end_sample": int(off_s * SR), "text": text, "confidence": 0.9}


def _activity(words, dur_s):
    """Synthetic stem activity: active exactly inside the words (10 ms frames)."""
    hop = 480
    c = np.arange(int(dur_s * SR / hop)) * hop + hop // 2
    act = np.zeros(c.size, bool)
    for w in words:
        act[(c >= w["start_sample"]) & (c < w["end_sample"])] = True
    rms = np.where(act, -10.0, -70.0)
    return P.StemActivity(centers=c, active=act, rms_db=rms, threshold_db=-50.0, f0_hz=np.full(c.size, np.nan),
                          onsets=np.empty(0, int), hop=hop, active_strict=act.copy())


def _verse(D, pickup_s=0.3, last_word="night"):
    """A line ending with a rest on beat 3-4 of the bar before ``D``; next line with a pickup."""
    t0 = D - BAR
    words = [_w("I", t0 + 0.0, t0 + 0.15), _w("ride", t0 + 0.17, t0 + 0.4), _w("all", t0 + 0.42, t0 + 0.6),
             _w("through", t0 + 0.62, t0 + 0.85), _w("the", t0 + 0.87, t0 + 1.0), _w("cold", t0 + 1.02, t0 + 1.3),
             _w(last_word, t0 + 1.40, t0 + 1.95)]
    # previous bar: an end rhyme one bar earlier at the same position
    words = [_w("we", t0 - BAR + 1.0, t0 - BAR + 1.3), _w("light", t0 - BAR + 1.40, t0 - BAR + 1.95)] + words
    nxt = [_w("so", D - pickup_s, D - pickup_s + 0.12), _w("bring", D - pickup_s + 0.14, D + 0.05),
           _w("it", D + 0.07, D + 0.25), _w("back", D + 0.27, D + 0.6)]
    return words + nxt


@pytest.fixture(scope="module")
def lex():
    return P.Lexicon()


# ------------------------------------------------------------- probability
def test_mcflow_tables_and_boundary_probability():
    m = P.load_mcflow()
    assert m["source"]["n_files"] >= 120 and 0.1 < m["base_rate"] < 0.2
    hi = P.boundary_probability(4, True, 2.0, True, 1.5, None)["p"]
    lo = P.boundary_probability(1, False, 0.0, False, 0.8, None)["p"]
    assert hi > 0.9 and lo < 0.05
    # rhyme and lengthening move the estimate in the corpus direction
    base = P.boundary_probability(3, False, 0.0, None, None, None)["p"]
    assert P.boundary_probability(3, False, 0.0, True, None, None)["p"] > base
    assert P.boundary_probability(3, False, 0.0, None, 2.5, None)["p"] > base
    # a long gap uses the long-gap row regardless of position
    assert P.boundary_probability(1, True, 12.0, None, None, None)["p"] > 0.9


def test_mcflow_heldout_documents_the_cues():
    h = P.load_mcflow()["heldout"]["models"]
    assert h["pos_gap+rhyme+leng"]["auc"] > h["pos_gap"]["auc"] > h["position_only"]["auc"]
    # a pause is evidence, not a phrase end: rests alone miss many boundaries
    assert h["gap_only"]["recall_at_map"] < 0.7


def test_ioi_information_content_prefers_common_rhythms():
    m = P.IOIModel(P.load_mcflow()["ioi_model"])
    ic = m.ic([2, 2, 2, 2, 2, 13])
    assert ic[-1] > ic[-2]


# ----------------------------------------------------------------- lexicon
def test_orthographic_rhymes_french_and_spanish():
    a = P.Lexicon._ortho("parole", "fr")
    b = P.Lexicon._ortho("école", "fr")
    assert P.rhyme_match(a, b)[1] == "rhyme"
    c = P.Lexicon._ortho("corazón", "es")
    assert c["n_syl"] == 3


def test_cmudict_rhyme(lex):
    if not lex.available:
        pytest.skip("CMUdict not staged")
    a, b, c = lex.analyse("night"), lex.analyse("light,"), lex.analyse("cold")
    assert P.rhyme_match(a, b)[1] == "rhyme"
    assert P.rhyme_match(a, c)[0] == 0.0
    assert lex.analyse("heartless")["n_syl"] == 2


# --------------------------------------------------------- exits / entries
def test_line_model_exit_with_a_pickup_and_entry_with_b_pickup(lex):
    tr = _track()
    D = tr["grid"]["downbeats_s"][16]
    words = _verse(D, pickup_s=0.3)
    act = _activity(words, tr["duration_s"])
    model = P.analyze_track(tr, words, act=act, lang="en", lexicon=lex)
    night = next(g for g in model["gaps"] if g["word"] == "night")
    assert night["p_end"] >= 0.5 and night["rest"]
    cold = next(g for g in model["gaps"] if g["word"] == "cold")
    assert cold["p_end"] < night["p_end"]
    ex = P.exit_point(model, D, act=act)
    assert ex["status"] == "a_pickup" and ex["last_word"] == "night"
    m = int(P.PARAMS["switch_margin_s"].value * SR)
    assert ex["window"][1] == pytest.approx(-int(0.3 * SR) - m, abs=480)  # 10 ms frames
    # a dry cut on the downbeat would truncate "bring" (the pickup runs across D)
    assert "bring" in ex["dry_cut_truncates"]
    en = P.entry_point(model, D, act=act)
    assert en["status"] == "pickup"
    assert en["pickup_samples"] == pytest.approx(int(0.3 * SR), abs=480)
    assert en["pickup_words"] == ["so", "bring"]


def test_entry_mid_line_is_rejected(lex):
    tr = _track()
    D = tr["grid"]["downbeats_s"][16]
    # continuous line from 3 beats before D through D, no pause
    t = D - 3 * BEAT
    words, vocab, k = [], ["money", "on", "my", "mind", "and", "time", "on", "the", "line", "we", "keep", "it", "real"], 0
    while t < D + BAR:
        words.append(_w(vocab[k % len(vocab)], t, t + 0.18))
        t += 0.2
        k += 1
    act = _activity(words, tr["duration_s"])
    model = P.analyze_track(tr, words, act=act, lang="en", lexicon=lex)
    assert P.entry_point(model, D, act=act)["status"] == "mid_line"
    assert P.exit_point(model, D, act=act)["status"] in ("mid_line", "gap_not_line_end", "no_gap")
    assert not P.exit_clear_test(model, act)({"time_s": D})


def test_phrase_switch_moves_the_fader_before_b_pickup(lex):
    trA, trB = _track(slug="a"), _track(slug="b")
    Da, Db = trA["grid"]["downbeats_s"][16], trB["grid"]["downbeats_s"][8]
    wa = _verse(Da, pickup_s=0.0)[:-4]  # A: line ends, nothing after
    wb = _verse(Db, pickup_s=0.45)
    acta, actb = _activity(wa, trA["duration_s"]), _activity(wb, trB["duration_s"])
    ma = P.analyze_track(trA, wa, act=acta, lexicon=lex)
    mb = P.analyze_track(trB, wb, act=actb, lexicon=lex)
    ea, eb = P.exit_point(ma, Da, act=acta), P.entry_point(mb, Db, act=actb)
    assert ea["status"] in ("clean", "instrumental") and eb["status"] == "pickup"
    sw = P.phrase_switch(ea, eb, BEAT)
    assert sw["ok"] and sw["pickup_preserved"]
    x = sw["switch_rel_samples"]
    assert x <= -int(0.45 * SR)
    assert x % int(round(BEAT * SR / 2)) == 0  # on the eighth-note grid
    # no word of A or B is truncated at the chosen switch
    m = int(P.PARAMS["word_margin_s"].value * SR)
    assert not P.truncated_words(ma["words"], int(Da * SR) + x, role="a", margin=m)
    assert not P.truncated_words(mb["words"], int(Db * SR) + x, role="b", margin=m)


def test_tail_of_last_word_is_protected():
    tr = _track()
    D = tr["grid"]["downbeats_s"][16]
    words = [_w("we", D - 1.2, D - 0.8), _w("go", D - 0.7, D - 0.3), _w("far", D - 0.25, D + 0.12),
             _w("next", D + 1.0, D + 1.3)]
    act = _activity(words, tr["duration_s"])
    model = P.analyze_track(tr, words, act=act, lexicon=P.Lexicon())
    ex = P.exit_point(model, D, act=act)
    if (ex.get("p_line_end") or 0) >= 0.5:
        assert ex["status"] == "tail"
    assert ex["tail_end_sample"] is not None and ex["tail_end_sample"] > int(D * SR)
    assert ex["window"] == [0, 0]


def test_truncated_words_definition():
    words = [{"on": 0, "off": 1000, "text": "a"}, {"on": 2000, "off": 3000, "text": "b"}]
    assert [w["text"] for w in P.truncated_words(words, 500, role="a", margin=100)] == ["a"]
    assert P.truncated_words(words, 1500, role="a", margin=100) == []
    assert P.truncated_words(words, 500, role="a", margin=100, keep_until=1000) == []
    assert [w["text"] for w in P.truncated_words(words, 2500, role="b", margin=100)] == ["b"]


def test_unreliable_asr_abstains():
    tr = _track()
    words = [_w("ლ", 1.0 + 3 * k, 3.5 + 3 * k) for k in range(12)]
    model = P.analyze_track(tr, words, act=None, asr_language_probability=0.7)
    assert not model["asr_quality"]["reliable"]
    D = tr["grid"]["downbeats_s"][16]
    assert P.exit_point(model, D)["status"] == "asr_unreliable"
    assert P.entry_point(model, D)["window"] is None


def test_refine_words_finds_the_pause_inside_abutting_asr_words():
    # whisper glues the words; the stem has a 200 ms silence inside "night"'s ASR span
    words = [_w("cold", 1.0, 1.5), _w("night", 1.5, 2.2), _w("so", 2.2, 2.5)]
    real = [_w("cold", 1.0, 1.5), _w("night", 1.5, 1.9), _w("so", 2.1, 2.5)]
    act = _activity(real, 4.0)
    out = P.refine_words(P.normalize_words(words), act)
    assert out[1]["off"] == pytest.approx(int(1.9 * SR), abs=480)
    assert out[2]["on"] == pytest.approx(int(2.1 * SR), abs=480)


# ------------------------------------------------------------ dj_select hook
def test_plan_pair_phrase_option_leaves_defaults_unchanged(lex):
    A, B = _track(slug="a"), _track(slug="b")
    Da, Db = A["grid"]["downbeats_s"][24], B["grid"]["downbeats_s"][8]
    wa, wb = _verse(Da, pickup_s=0.0)[:-4], _verse(Db, pickup_s=0.3)
    acta, actb = _activity(wa, A["duration_s"]), _activity(wb, B["duration_s"])
    ma, mb = P.analyze_track(A, wa, act=acta, lexicon=lex), P.analyze_track(B, wb, act=actb, lexicon=lex)
    default = djs.plan_pair(A, B, top_k=3)
    assert all(p["type"] != "cut_phrase" for p in default["plans"])
    ph = djs.plan_pair(A, B, top_k=500, phrase={"a": ma, "b": mb, "act_a": acta, "act_b": actb})
    assert ph["plans"] and all(p["type"] == "cut_phrase" and p["phrase"]["switch"]["ok"] for p in ph["plans"])
    best = next(p for p in ph["plans"] if p["a_out_bar"] == 24 and p["b_in_bar"] == 8)
    assert best["phrase"]["entry"]["status"] == "pickup" and best["phrase"]["switch"]["switch_rel_s"] <= -0.3
    # mid-line exits of A never appear
    assert all(p["phrase"]["exit"]["status"] != "mid_line" for p in ph["plans"])
    # the same test plugged into relocate_cut
    test = P.exit_clear_test(ma, acta)
    assert test({"time_s": Da})


# ------------------------------------------------------------------ render
def test_render_phrase_cut_keeps_b_pickup_and_a_tail():
    rng = np.random.default_rng(0)
    dur = 40.0
    n = int(dur * SR)
    beats = np.arange(0.5, dur, BEAT)
    downs = beats[::4]
    a = 0.05 * rng.standard_normal((n, 2))
    b = 0.05 * rng.standard_normal((n, 2))
    Da, Db = float(downs[6]), float(downs[4])
    # B: silence in the bar before its downbeat except a loud "pickup" 0.4 s before it
    b[int((Db - BAR) * SR):int(Db * SR)] = 0.0
    b[int((Db - 0.4) * SR):int((Db - 0.05) * SR)] = 0.3
    va = np.zeros((n, 2))
    va[int((Da - 0.5) * SR):int((Da + 0.15) * SR)] = 0.2
    res = P.render_phrase_transition("cut", a, b, beats_a=beats, downbeats_a=downs, beats_b=beats, downbeats_b=downs,
                                     cue_out_a=Da, cue_in_b=Db, switch_rel_s=-0.5, tail_end_a_s=Da + 0.15, vocals_a=va)
    meta = res.metadata
    pre = int(10 * SR)
    assert meta["switch"]["output_sample"] == meta["switch"]["downbeat_output_sample"] == pre
    assert meta["switch"]["fader_switch_output_sample"] == pre - int(0.5 * SR)
    assert len(res.audio) == int(20 * SR)
    seg = res.audio[pre - int(0.35 * SR):pre - int(0.1 * SR)]
    assert np.sqrt(np.mean(seg ** 2)) > 0.05  # B's pickup is audible before the downbeat
    tail = res.decks["a"][pre + int(0.02 * SR):pre + int(0.12 * SR)]
    assert np.abs(tail).max() > 0.0  # A's vocal tail continues after the downbeat
    assert np.abs(res.decks["a"][pre + int(0.25 * SR):]).max() == 0.0
    # the plain dj_render cut of the same cues loses the pickup
    from musicmix import dj_render as dj
    base = dj.render_transition(a, b, beats_a=beats, downbeats_a=downs, beats_b=beats, downbeats_b=downs,
                                cue_out_a=Da, cue_in_b=Db, kind="cut")
    seg0 = base.audio[pre - int(0.35 * SR):pre - int(0.1 * SR)]
    assert np.sqrt(np.mean(seg0 ** 2)) < np.sqrt(np.mean(seg ** 2))


def test_render_stem_pickup_keeps_a_instrumental_until_the_downbeat():
    rng = np.random.default_rng(1)
    dur = 40.0
    n = int(dur * SR)
    beats = np.arange(0.5, dur, BEAT)
    downs = beats[::4]
    inst_a = 0.05 * rng.standard_normal((n, 2))
    va = np.zeros((n, 2))
    b = 0.05 * rng.standard_normal((n, 2))
    vb = np.zeros((n, 2))
    Da, Db = float(downs[6]), float(downs[4])
    vb[int((Db - 0.4) * SR):int((Db - 0.05) * SR)] = 0.2
    b = b + vb
    a = inst_a + va
    res = P.render_phrase_transition("cut", a, b, beats_a=beats, downbeats_a=downs, beats_b=beats, downbeats_b=downs,
                                     cue_out_a=Da, cue_in_b=Db, switch_rel_s=-0.5, vocals_a=va, vocals_b=vb, pickup_mode="stem")
    pre = int(10 * SR)
    assert res.metadata["phrase"]["pickup"]["mode"] == "stem"
    # A's instrumental is still there just before the downbeat, B's full mix is not (only its vocal)
    a_seg = res.decks["a"][pre - int(0.3 * SR):pre - int(0.02 * SR)]
    assert np.sqrt(np.mean(a_seg ** 2)) > 0.02
    b_seg = res.decks["b"][pre - int(0.45 * SR):pre - int(0.42 * SR)]  # before B's vocal pickup: B silent
    assert np.abs(b_seg).max() < 1e-9


def test_render_phrase_cut_overlap_source_samples_follow_the_pickup():
    """The pickup region moves the overlap start: A's source sample must move with it (clip_geometry splices on it)."""
    from musicmix import dj_set as ds

    rng = np.random.default_rng(2)
    dur = 40.0
    n = int(dur * SR)
    beats = np.arange(0.5, dur, BEAT)
    downs = beats[::4]
    a = 0.05 * rng.standard_normal((n, 2))
    b = 0.05 * rng.standard_normal((n, 2))
    Da, Db = float(downs[6]), float(downs[4])
    res = P.render_phrase_transition("cut", a, b, beats_a=beats, downbeats_a=downs, beats_b=beats, downbeats_b=downs,
                                     cue_out_a=Da, cue_in_b=Db, switch_rel_s=-0.5)
    ov, sw = res.metadata["overlap"], res.metadata["switch"]
    k = int(ov["output_start_sample"])
    assert int(sw["output_sample"]) - k > int(0.5 * SR) - 1
    # same clock for every output sample: source = switch source - (switch output - k)
    assert int(ov["a_source_start_sample"]) == int(sw["a_source_sample"]) - (int(sw["output_sample"]) - k)
    assert int(ov["b_source_start_sample"]) == int(sw["b_source_sample"]) - (int(sw["output_sample"]) - k)
    g = ds.clip_geometry(res.metadata, len(res.audio))
    assert g["a_offset"] == int(sw["a_source_sample"]) - int(sw["output_sample"])
    j = k - int(1.0 * SR)  # A alone before the pickup: clip sample j is A's source sample j + a_offset (gain applied)
    ga = 10 ** (res.metadata["loudness"]["a_gain_db"] / 20.0)
    src = a[j + g["a_offset"]] * ga
    assert np.allclose(res.decks["a"][j], src, atol=1e-6)
