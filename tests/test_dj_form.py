"""Test delle feature formali di Butler (musicmix.dj_form) su audio sintetico."""

import numpy as np

from musicmix import dj_form as F

SR = 16000
BPM = 120.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT


def _kick(n):
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * 60 * t) * np.exp(-t / 0.08)


def _synth(plan, seed=0):
    """plan: lista di set di strati per battuta ('kick', 'bass', 'hat', 'pad', 'voc')."""
    rng = np.random.default_rng(seed)
    n_bar = int(BAR * SR)
    mix = np.zeros(len(plan) * n_bar)
    voc = np.zeros_like(mix)
    k = _kick(int(0.3 * SR))
    for j, layers in enumerate(plan):
        a = j * n_bar
        t = np.arange(n_bar) / SR
        if "kick" in layers:
            for b in range(4):
                s = a + int(b * BEAT * SR)
                mix[s:s + len(k)] += 0.8 * k[: len(mix) - s]
        if "bass" in layers:
            mix[a:a + n_bar] += 0.3 * np.sin(2 * np.pi * 110 * t)
        if "pad" in layers:
            mix[a:a + n_bar] += 0.1 * (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 660 * t))
        if "hat" in layers:
            for b in range(16):
                s = a + int(b * BEAT / 4 * SR)
                L = int(0.02 * SR)
                mix[s:s + L] += 0.2 * rng.standard_normal(L) * np.hanning(L)
        if "voc" in layers:
            v = 0.2 * np.sin(2 * np.pi * 300 * t) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t))
            voc[a:a + n_bar] += v
            mix[a:a + n_bar] += v
    downbeats = np.arange(len(plan)) * BAR
    beats = np.arange(len(plan) * 4) * BEAT
    return mix, voc, downbeats, beats


FULL = {"kick", "bass", "hat", "pad", "voc"}
PLAN = ([{"kick"}] * 4 + [{"kick", "hat"}] * 2 + [{"kick", "hat", "bass"}] * 2 + [FULL] * 8
        + [{"pad"}] * 4 + [FULL] * 8 + [{"kick", "hat"}] * 4)


def test_kick_and_layers_follow_the_plan():
    mix, voc, db, beats = _synth(PLAN)
    lay = F.bar_layers(mix, SR, db, beats, vocals=voc, params={"n_fft": 1024, "hop": 256})
    act = F.layer_activity(lay)
    assert lay["source"] == "bands" and lay["n_bars"] == len(PLAN)
    for j, layers in enumerate(PLAN):
        assert act["kick"][j] == ("kick" in layers), j
    assert act["vocals"][10] and not act["vocals"][0]
    assert act["bass"][10] and not act["bass"][0]


def test_form_segments_intro_build_core_breakdown_drop_outro():
    mix, voc, db, beats = _synth(PLAN)
    form = F.analyse(mix, SR, db, beats, vocals=voc, params={"n_fft": 1024, "hop": 256})
    lab = form["labels"]
    assert lab[0] == "intro"
    assert "build" in lab[4:8]
    assert set(lab[8:16]) == {"core"}
    assert set(lab[16:20]) == {"breakdown"}
    assert form["breakdowns"] == [[16, 20]]
    assert 20 in form["drops"]
    assert set(lab[28:32]) == {"outro"}
    s = F.form_summary(form)
    assert s["segments"][0][0] == "intro" and s["bars_by_label"]["breakdown"] == 4


def test_short_kick_dropout_is_not_a_breakdown():
    plan = [FULL] * 8 + [{"bass", "hat", "pad", "voc"}] + [FULL] * 8
    mix, voc, db, beats = _synth(plan)
    form = F.analyse(mix, SR, db, beats, vocals=voc, params={"n_fft": 1024, "hop": 256})
    assert form["breakdowns"] == []
    assert form["kick_dropouts"] == [{"start_bar": 8, "end_bar": 9, "bars": 1}]
    assert 9 in form["drops"]


def test_busy_and_overlap_profile():
    mix, voc, db, beats = _synth(PLAN)
    form = F.analyse(mix, SR, db, beats, vocals=voc, params={"n_fft": 1024, "hop": 256, "busy_onsets_per_beat": 3.0})
    assert any(form["busy"][8:16]) and not any(form["busy"][0:4])
    prof = F.overlap_profile(form, form, 8, 0, 4)
    assert prof["bars"] == 4 and prof["a_layers"] > prof["b_layers"]
    assert prof["both_busy_share"] == 0.0
    assert prof["b_form_at"] == "intro" and prof["a_form_before"] == "build"
    prof2 = F.overlap_profile(form, form, 10, 10, 4)
    assert prof2["both_vocals_share"] == 1.0
    assert F.overlap_profile(form, form, 16, 20, 2)["b_entry_drop"] is True


def test_stems_path_uses_drums_stem():
    mix, voc, db, beats = _synth(PLAN)
    drums = np.zeros_like(mix)
    k = _kick(int(0.3 * SR))
    for j, layers in enumerate(PLAN):
        if "kick" in layers:
            for b in range(4):
                s = int(j * BAR * SR + b * BEAT * SR)
                drums[s:s + len(k)] += 0.8 * k[: len(drums) - s]
    stems = {"drums": drums, "bass": np.zeros_like(mix), "other": mix - drums - voc}
    lay = F.bar_layers(mix, SR, db, beats, vocals=voc, stems=stems, params={"n_fft": 1024, "hop": 256})
    assert lay["source"] == "stems"
    act = F.layer_activity(lay)
    assert act["kick"][:4] == [True] * 4 and act["kick"][16:20] == [False] * 4


def test_allin1_spec_path():
    """Spettrogrammi per stem sintetici nel formato di All-In-One (4, frame, 81), 100 fps."""
    fps, n_bars = 100, 12
    frames = int(n_bars * BAR * fps)
    c = np.asarray(F.ALLIN1_SPEC_CENTERS_HZ)
    spec = np.zeros((4, frames, len(c)))
    kick_bins = c < 120
    for j in range(n_bars):
        if 4 <= j < 8:  # breakdown: no drums
            continue
        for b in range(4):
            k = int((j * BAR + b * BEAT) * fps)
            spec[1, k:k + 5, kick_bins] = np.log1p(10.0)  # drums stem, low bins
        spec[0, int(j * BAR * fps):int((j + 1) * BAR * fps), c < 200] = np.log1p(3.0)  # bass stem always
    db = np.arange(n_bars) * BAR
    beats = np.arange(n_bars * 4) * BEAT
    form = F.analyse_from_allin1_spec(spec, db, beats)
    assert form["layers"]["source"] == "allin1_spec"
    assert form["active"]["kick"] == [True] * 4 + [False] * 4 + [True] * 4
    assert form["breakdowns"] == [[4, 8]] and 8 in form["drops"]
    with __import__("pytest").raises(ValueError):
        F.bar_layers_from_allin1_spec(np.zeros((3, 10, 81)), db, beats)


def _flat_layers(n=8, kick=0.0, db=-12.0):
    lay = {"kick": [kick] * n, "onsets_per_beat": [1.0] * n}
    lay.update({f"{k}_db": [db] * n for k in ("bass", "mid", "high", "vocals")})
    return lay


def test_no_core_is_unknown_not_outro():
    """Controesempio dell'audit 21/10/2026 (ROOT_COUNTEREXAMPLES.json 'form_no_kick'): senza cassa le otto battute
    erano tutte 'outro'. Senza core intro/build/outro non sono definiti: 'unknown' e core_status esplicito."""
    form = F.form_segments(_flat_layers())
    assert form["first_core_bar"] is None and form["last_core_bar"] is None
    assert form["labels"] == ["unknown"] * 8 and "outro" not in form["labels"]
    assert form["core_status"] == "no_core" and form["no_core_reason"] == "no_kick"
    s = F.form_summary(form)
    assert s["core_status"] == "no_core" and s["bars_by_label"] == {"unknown": 8}


def test_no_core_with_kick_but_thin_texture():
    lay = _flat_layers(kick=1.0)
    for k in ("bass", "mid", "high", "vocals"):
        lay[f"{k}_db"] = [-80.0] * 8  # only the kick: below the -60 dB floor, n_layers = 1 < core_level 2
    form = F.form_segments(lay)
    assert form["labels"] == ["unknown"] * 8
    assert form["core_status"] == "no_core" and form["no_core_reason"] == "kick_without_full_texture"


def test_core_found_keeps_the_previous_labels():
    mix, voc, db, beats = _synth(PLAN)
    form = F.analyse(mix, SR, db, beats, vocals=voc, params={"n_fft": 1024, "hop": 256})
    assert form["core_status"] == "found" and form["no_core_reason"] is None and "unknown" not in form["labels"]


# ------------------------------------------------- dj_form.3: 808 lunghi e sincopati
def _808(n, f_start=70.0, f_end=45.0, glide_s=0.12, decay_s=0.8):
    """808 tonale: attacco secco, coda lunga, glissando verso il basso."""
    t = np.arange(n) / SR
    f = f_end + (f_start - f_end) * np.exp(-t / glide_s)
    phase = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(phase) * np.exp(-t / decay_s)


def _synth_808(plan):
    """plan: per battuta 'intro' (hat + pad + voce, niente basso) o '808' (+ 808 sul battito 1 e sul 'e' del 2)."""
    rng = np.random.default_rng(3)
    n_bar = int(BAR * SR)
    mix = np.zeros(len(plan) * n_bar)
    voc = np.zeros_like(mix)
    hit = _808(int(1.5 * BEAT * SR))
    for j, kind in enumerate(plan):
        a = j * n_bar
        t = np.arange(n_bar) / SR
        for b in range(16):
            s = a + int(b * BEAT / 4 * SR)
            L = int(0.02 * SR)
            mix[s:s + L] += 0.15 * rng.standard_normal(L) * np.hanning(L)
        mix[a:a + n_bar] += 0.1 * (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 660 * t))
        v = 0.2 * np.sin(2 * np.pi * 300 * t) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t))
        voc[a:a + n_bar] += v
        mix[a:a + n_bar] += v
        if kind in ("808", "pickup"):
            for pos in ((0.0, 1.5) if kind == "808" else (3.0,)):  # beat 1 and the 'and' of 2 (off the grid) / lone beat-4 hit
                s = a + int(pos * BEAT * SR)
                seg = hit[: min(len(hit), len(mix) - s)]
                mix[s:s + len(seg)] += 0.9 * seg
    downbeats = np.arange(len(plan)) * BAR
    beats = np.arange(len(plan) * 4) * BEAT
    return mix, voc, downbeats, beats


def test_long_808_with_pitch_glide_is_a_kick():
    """gods_plan (audit 21/10/2026): un 808 lungo, sincopato e con glissando non supera la media sui battiti (0,30)
    ma è la cassa del brano; dj_form.3 lo rileva dal picco d'attacco basso nella battuta."""
    plan = ["intro"] * 4 + ["808"] * 16 + ["intro"] * 4
    mix, voc, db, beats = _synth_808(plan)
    lay = F.bar_layers(mix, SR, db, beats, vocals=voc, params={"n_fft": 1024, "hop": 256})
    on = [j for j, k in enumerate(plan) if k == "808"]
    off = [j for j, k in enumerate(plan) if k != "808"]
    assert max(lay["kick"][j] for j in on) < F.PARAMS["kick_on"][0]  # the dj_form.2 rule misses it
    legacy = F.layer_activity(lay, {"kick_peak_on": None})
    assert not any(legacy["kick"])
    act = F.layer_activity(lay)
    assert all(act["kick"][j] for j in on) and not any(act["kick"][j] for j in off)
    assert all(act["kick_peak_only"][j] for j in on)
    assert min(lay["kick_sustain"][j] for j in on) > 0.2  # long tonal tail (diagnostic only)
    form = F.analyse(mix, SR, db, beats, vocals=voc, params={"n_fft": 1024, "hop": 256})
    assert form["core_status"] == "found" and form["version"] == "musicmix.dj_form.3"
    assert form["drops"] == []  # the 808 enters after the intro: first kick, not a return
    assert set(form["labels"][5:19]) == {"core"}
    old = F.form_segments(lay, {"kick_peak_on": None})
    assert old["core_status"] == "no_core" and old["no_core_reason"] == "no_kick"


def test_four_on_the_floor_kick_unchanged_by_the_808_rule():
    mix, voc, db, beats = _synth(PLAN)
    lay = F.bar_layers(mix, SR, db, beats, vocals=voc, params={"n_fft": 1024, "hop": 256})
    assert F.layer_activity(lay)["kick"] == F.layer_activity(lay, {"kick_peak_on": None})["kick"]
    assert F.form_segments(lay)["labels"] == F.form_segments(lay, {"kick_peak_on": None})["labels"]


def test_layers_cached_before_dj_form_3_keep_the_old_rule():
    lay = _flat_layers(kick=0.2)
    assert "kick_peak" not in lay
    assert F.form_segments(lay)["no_core_reason"] == "no_kick"


def test_lone_808_on_the_last_beat_is_an_anacrusis_not_a_kick_bar():
    """Un colpo isolato sull'ultimo battito prima del ritorno dell'808 appartiene al gesto anacrusico (Butler p. 189):
    la battuta resta senza cassa e il drop cade sul downbeat del ritorno (casi 18k_no_kizzy@42, knife_talk@24)."""
    plan = ["808"] * 8 + ["intro"] * 3 + ["pickup"] + ["808"] * 8
    mix, voc, db, beats = _synth_808(plan)
    lay = F.bar_layers(mix, SR, db, beats, vocals=voc, params={"n_fft": 1024, "hop": 256})
    act = F.layer_activity(lay)
    assert not act["kick"][11] and act["kick"][12]
    form = F.form_segments(lay)
    assert 12 in form["drops"] and 11 not in form["drops"]
