"""DJ-craft track analysis, pair compatibility and cue candidates (P0/P1).

Implements the selection side of ``conoscenza_mix/_meta/REVISIONE_DJ_2026-10-20.md``:
beat/downbeat grid, phrase boundaries on downbeats, key in Camelot notation,
vocal bars, section energy, tempo/key gates and cue candidates with a
suggested transition type.  Everything here is a pure function over plain
arrays and dicts: no audio I/O, no model inference.  Provider outputs (Beat
This!, All-In-One, madmom, LinkSeg, SSM-Net, Demucs vocal occupancy) are
loaded by ``tools/dj_plan_pool.py``.

Rules and constants are fixed a priori from the review, never fitted on the
user's listening judgements:

* tempo: B is stretched to A, at most +-8 % (MosaikBox [S]); 86 % of real
  DJ stretches stay within 5 % (Kim 2020 [E]), so long blends need <= 5 %.
  Half/double time is folded (70 <-> 140 BPM) [H].
* key: same key, +-1 on the Camelot wheel, or relative major/minor
  (Vande Veire 2018 [S]; key lock in practice, Kim 2020 [E]); anything else
  gets a strong penalty.
* cues: A's cue-out and B's cue-in only on a downbeat that starts a 4/8/16
  bar phrase (Zehren 2020 rule 2 [P]; CUE-DETR [E]); phrases are counted from
  the downbeat nearest to each structural section boundary.
* vocals: >= 2 bars with both vocals active is a vocal clash, 1 tolerated
  (Vande Veire 2018 [S]); if both are vocal: cut, or a short blend with the
  outgoing vocal ducked (MosaikBox [S]).

Transition plan convention (all times in seconds on each track's own
canonical timeline):

* ``a_out_s`` is a phrase-start downbeat of A where B starts sounding;
* ``b_in_s`` is a phrase-start downbeat of B that is played at ``a_out_s``;
* ``cut``: A stops at ``a_out_s`` and B continues from ``b_in_s``;
* ``blend`` N bars: A keeps playing ``[a_out_s, a_out_s + N bars)`` while B
  plays ``[b_in_s, b_in_s + N bars)``; bass swap on the middle downbeat of the
  overlap, A gone at the end (renderer decides the exact curves);
* ``blend_duck``: a 4-bar blend with A's vocal stem attenuated.

Phrase guard (option, ``plan_pair(..., phrase=...)``, 21/10/2026): cues at verse ends with B's
pickup from :mod:`musicmix.dj_phrase` (MCFlow-estimated line-end probability, word timestamps,
stem prosody); ``dj_phrase.exit_clear_test`` plugs the same test into :func:`relocate_cut`.

Cut vocal guard (option, off by default so earlier plans are unchanged):
``plan_transition(..., cut_vocal_guard=True)`` admits a cut only when A's
vocal covers at most ``CUT_GUARD_MAX_COVER`` of the last beat before the cue
(``bars.vocal_last_beat``); otherwise the plan becomes ``echo_out`` (an exit
through an effect is exempt from the truncated-word rule).  ``relocate_cut``
searches the nearest earlier or later valid phrase-start cue of A whose exit
is clear, with a caller-supplied test (e.g. stem RMS in the last beat or half
bar, ``tools/run_dj_gallery_v2.py``) [H].

The score is a sum of named, interpretable terms (``reasons`` lists each
one).  It ranks candidates; it is not a quality estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

SCHEMA_VERSION = "musicmix.dj_select.1"

# ---------------------------------------------------------------- constants
TEMPO_MAX_STRETCH = 0.08          # MosaikBox +-8 % [S]
TEMPO_LONG_BLEND_STRETCH = 0.05   # Kim 2020: 86.1 % within 5 % [E]
TEMPO_FREE_STRETCH = 0.02         # below this the stretch term is zero [H]
BEAT_TOLERANCE_S = 0.07           # standard beat-tracking tolerance (70 ms)
BPM_RANGE = (55.0, 215.0)
BEATS_PER_BAR = 4                 # 4/4 assumed for this pool [H]
PHRASE_LEVELS = (16, 8, 4)
BLEND_BARS = (16, 8, 4)
VOCAL_BAR_ACTIVE = 0.5            # bar counted vocal if >= 50 % covered [H]
CLASH_TOLERATED_BARS = 1          # Vande Veire: >= 2 overlapping vocal bars [S]
SECTION_SNAP_MAX_BARS = 0.5       # a boundary farther than this from a downbeat is "loose"
MIN_SECTION_BARS = 2              # shorter grid sections are merged into a neighbour [H]
KEY_CONFIDENT = 0.5               # combined key confidence threshold [H]
# Hard cue windows as a fraction of track duration [H]: A has played its core
# (M-DJCUE "OUT not before the core" [P]; hip-hop tracklists change track about
# every 2 min, DJMIX_STATS_HIPHOP [M]) and B keeps at least half of itself.
OUT_MIN_POSITION = 0.35
IN_MAX_POSITION = 0.5
#: Cue windows by name [H]. "default" = the values above (v1, unchanged). "technique" (option, audit P0
#: 21/10/2026, cue per ruolo e tecnica): wider generation of cues for transitions WITHOUT overlap (cut,
#: filter/echo exits, filter into the chorus, backspin). Role-based limits replace the fractions: B must keep
#: at least 60 s and 16 bars after its entry and may enter up to 75 % of its duration (a chorus or verse of B
#: after the middle of the track); A may leave from 60 s on, e.g. right after an early chorus ([P] B&B 2003
#: cap. 13 'Even tighter': "mix straight after a chorus ... very effective with breakbeat genres (hip hop ...)";
#: [A] Katz 2012 p. 138: quick-mixing = segue of short song fragments). 60 s = dj_set.CUE_PARAMS play_min_s
#: (user request 60-120 s per track); choose_cues still checks the real play time. A blend keeps the default
#: windows: musicmix.dj_set.candidates_for_pair(cue_windows="technique") drops overlaps outside them.
#: Measured on the bridge catalog: runs/p0_misure_regole/cues.json.
CUE_WINDOWS: dict[str, dict[str, Optional[float]]] = {
    "default": {"out_min_position": OUT_MIN_POSITION, "out_min_bars_left": 4, "out_min_s": None,
                "in_max_position": IN_MAX_POSITION, "in_min_bars_left": 16, "in_min_s_left": None},
    "technique": {"out_min_position": 0.0, "out_min_bars_left": 4, "out_min_s": 60.0,
                  "in_max_position": 0.75, "in_min_bars_left": 16, "in_min_s_left": 60.0},
}
# Cut vocal guard [H]: at most 10 % of the last beat before a cut may be
# covered by A's vocal occupancy (about 55 ms at 110 BPM, shorter than a
# syllable); relocation looks at most 16 bars (one long phrase) either side.
CUT_GUARD_MAX_COVER = 0.1
CUT_GUARD_MAX_BARS = 16

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E",
          "E#": "F", "B#": "C"}
# Camelot number for each tonic: major (B) and minor (A).
_CAMELOT_MAJOR = {"B": 1, "F#": 2, "C#": 3, "G#": 4, "D#": 5, "A#": 6,
                  "F": 7, "C": 8, "G": 9, "D": 10, "A": 11, "E": 12}
_CAMELOT_MINOR = {"G#": 1, "D#": 2, "A#": 3, "F": 4, "C": 5, "G": 6,
                  "D": 7, "A": 8, "E": 9, "B": 10, "F#": 11, "C#": 12}
# Krumhansl-Kessler probe-tone profiles (fallback only).
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

SECTION_OUT_FRIENDLY = {"outro", "break", "inst", "bridge", "end"}
SECTION_IN_FRIENDLY = {"intro", "inst", "break", "start"}
SECTION_CORE = {"verse", "chorus"}


# ------------------------------------------------------------------- grids
def grid_fmeasure(events: np.ndarray, period: float, phase: float,
                  tol: float = BEAT_TOLERANCE_S) -> tuple[float, float, float]:
    """Precision, recall, F of ``events`` against the grid ``phase + k*period``.

    Recall counts grid points between the first and last event that have an
    event within ``tol``; precision counts events within ``tol`` of a grid
    point.
    """
    events = np.asarray(events, dtype=float)
    if events.size < 2 or period <= 0:
        return 0.0, 0.0, 0.0
    k = np.round((events - phase) / period)
    err = np.abs(events - (phase + k * period))
    hit = err <= tol
    precision = float(hit.mean())
    k_lo = math.ceil((events[0] - tol - phase) / period)
    k_hi = math.floor((events[-1] + tol - phase) / period)
    n_grid = max(1, k_hi - k_lo + 1)
    recall = float(min(1.0, np.unique(k[hit]).size / n_grid))
    f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f


def _best_phase(events: np.ndarray, period: float) -> float:
    ang = 2 * np.pi * (events % period) / period
    mean = np.angle(np.mean(np.exp(1j * ang)))
    return float((mean % (2 * np.pi)) / (2 * np.pi) * period)


def _refine(events: np.ndarray, period: float, phase: float,
            tol: float = BEAT_TOLERANCE_S) -> tuple[float, float, np.ndarray]:
    """Least-squares (period, phase) on matched events; returns residuals (s)."""
    for _ in range(3):
        k = np.round((events - phase) / period)
        err = events - (phase + k * period)
        hit = np.abs(err) <= max(tol, 0.12 * period)
        if hit.sum() < 4:
            break
        slope, intercept = np.polyfit(k[hit], events[hit], 1)
        period, phase = float(slope), float(intercept)
    k = np.round((events - phase) / period)
    resid = events - (phase + k * period)
    return period, phase, resid


def fit_constant_grid(beats: Sequence[float], downbeats: Sequence[float] = (),
                      bpm_range: tuple[float, float] = BPM_RANGE,
                      beats_per_bar: int = BEATS_PER_BAR,
                      step_bpm: float = 0.05) -> dict[str, Any]:
    """Fit one constant-tempo beat grid and its bar phase to provider events.

    The tempo octave is chosen jointly with the downbeats: each candidate beat
    period P is scored as F(beats, P) + F(downbeats, beats_per_bar * P) with
    the bar phase restricted to the beat grid.  This rejects the double/half
    tempo that the beat F-measure alone cannot separate when the provider's
    beats are noisy.
    """
    b = np.sort(np.asarray(beats, dtype=float))
    d = np.sort(np.asarray(downbeats, dtype=float))
    if b.size < 8:
        raise ValueError("at least 8 beats are needed to fit a grid")
    bpms = np.arange(bpm_range[0], bpm_range[1] + step_bpm, step_bpm)
    fs = np.empty(bpms.size)
    for i, bpm in enumerate(bpms):
        p = 60.0 / bpm
        fs[i] = grid_fmeasure(b, p, _best_phase(b, p))[2]
    # candidate octaves around the best beat-level fits
    order = np.argsort(fs)[::-1]
    seeds: list[float] = []
    for i in order:
        if all(abs(bpms[i] / s - 1) > 0.02 for s in seeds):
            seeds.append(float(bpms[i]))
        if len(seeds) >= 3:
            break
    cands = set()
    for s in seeds:
        for m in (0.5, 2 / 3, 1.0, 1.5, 2.0):
            if bpm_range[0] <= s * m <= bpm_range[1]:
                cands.add(round(s * m, 3))
    best: dict[str, Any] | None = None
    for bpm0 in sorted(cands):
        # local fine search +-1 %
        local = np.arange(bpm0 * 0.99, bpm0 * 1.01, 0.01)
        lf = [grid_fmeasure(b, 60 / x, _best_phase(b, 60 / x))[2] for x in local]
        bpm = float(local[int(np.argmax(lf))])
        period, phase, resid = _refine(b, 60.0 / bpm, _best_phase(b, 60.0 / bpm))
        pb, rb, fb = grid_fmeasure(b, period, phase)
        bar_offset, fd, pd, rd = 0, 0.0, 0.0, 0.0
        if d.size >= 4:
            for m in range(beats_per_bar):
                pp, rr, ff = grid_fmeasure(d, period * beats_per_bar, phase + m * period)
                if ff > fd:
                    bar_offset, fd, pd, rd = m, ff, pp, rr
        score = fb + fd
        if best is None or score > best["score"] + 1e-9:
            best = {"bpm": 60.0 / period, "period_s": period, "phase_s": phase,
                    "bar_offset_beats": bar_offset, "beat_precision": pb, "beat_recall": rb,
                    "beat_f": fb, "downbeat_precision": pd, "downbeat_recall": rd,
                    "downbeat_f": fd, "score": score,
                    "residual_ms_median_abs": float(np.median(np.abs(resid[np.abs(resid) <= 0.12 * period])) * 1000)
                    if np.any(np.abs(resid) <= 0.12 * period) else None}
    assert best is not None
    best["drift_pct_max"] = tempo_drift_pct(b, best["period_s"])
    best["n_beats"] = int(b.size)
    best["n_downbeats"] = int(d.size)
    return best


def tempo_drift_pct(beats: np.ndarray, period: float, window_beats: int = 32) -> float | None:
    """Max deviation (%) of local fitted period over sliding windows."""
    b = np.asarray(beats, dtype=float)
    if b.size < window_beats:
        return None
    devs = []
    for s in range(0, b.size - window_beats + 1, window_beats // 2):
        seg = b[s:s + window_beats]
        ph = _best_phase(seg, period)
        p, _, _ = _refine(seg, period, ph)
        if 0.5 * period < p < 1.5 * period:
            devs.append(abs(p / period - 1) * 100)
    return float(max(devs)) if devs else None


def grid_times(fit: Mapping[str, Any], duration_s: float,
               beats_per_bar: int = BEATS_PER_BAR) -> tuple[np.ndarray, np.ndarray]:
    """Regularised beat and downbeat times covering ``[0, duration_s)``."""
    p, ph = fit["period_s"], fit["phase_s"]
    k0 = math.ceil((0.0 - ph) / p)
    k1 = math.floor((duration_s - 1e-6 - ph) / p)
    k = np.arange(k0, k1 + 1)
    beats = ph + k * p
    is_down = ((k - fit["bar_offset_beats"]) % beats_per_bar) == 0
    return beats, beats[is_down]


def grid_agreement(fit_a: Mapping[str, Any], fit_b: Mapping[str, Any], duration_s: float,
                   tol: float = BEAT_TOLERANCE_S) -> dict[str, Any]:
    """Compare two fitted grids: tempo ratio, beat and downbeat coincidence."""
    ratio = fit_b["bpm"] / fit_a["bpm"]
    ba, da = grid_times(fit_a, duration_s)
    bb, db = grid_times(fit_b, duration_s)
    same_tempo = bool(abs(ratio - 1) < 0.01)

    def frac(x: np.ndarray, y: np.ndarray) -> float:
        if x.size == 0 or y.size == 0:
            return 0.0
        idx = np.clip(np.searchsorted(y, x), 1, y.size - 1)
        near = np.minimum(np.abs(x - y[idx - 1]), np.abs(x - y[idx]))
        return float((near <= tol).mean())

    return {"bpm_ratio": ratio, "same_tempo_octave": same_tempo,
            "beat_coincidence": frac(ba, bb), "downbeat_coincidence": frac(da, db)}


def choose_grid(fits: Mapping[str, Mapping[str, Any]], duration_s: float,
                preference: Sequence[str] = ("allin1", "beat_this", "madmom")) -> dict[str, Any]:
    """Pick the grid to use and grade its reliability from provider agreement.

    Providers are grouped by tempo octave (BPM within 1 %); the largest group
    wins (majority vote on half/double time), ties broken by the best joint
    score (beat F + downbeat F).  Inside the group the best score wins, ties
    follow ``preference``.  Confidence: ``high`` when another provider agrees
    on tempo octave and on >= 80 % of downbeats; ``medium`` when the tempo
    agrees but the bar phase does not, or for a lone grid with beat F >= 0.8
    and downbeat F >= 0.6; else ``low``.  A provider in another octave caps
    the confidence at ``medium`` (``octave_ambiguous``).
    """
    if not fits:
        raise ValueError("no grid fits")
    rank = {name: i for i, name in enumerate(preference)}
    names = sorted(fits, key=lambda n: (-round(fits[n]["score"], 2), rank.get(n, 99)))
    groups: list[list[str]] = []
    for n in names:
        for g in groups:
            if abs(fits[n]["bpm"] / fits[g[0]]["bpm"] - 1) < 0.01:
                g.append(n)
                break
        else:
            groups.append([n])
    groups.sort(key=lambda g: (-len(g), -max(round(fits[n]["score"], 2) for n in g)))
    chosen = groups[0][0]
    agreements = {}
    for other in names:
        if other != chosen:
            agreements[other] = grid_agreement(fits[chosen], fits[other], duration_s)
    f = fits[chosen]
    level = "low"
    if any(a["same_tempo_octave"] and a["downbeat_coincidence"] >= 0.8 for a in agreements.values()):
        level = "high"
    elif any(a["same_tempo_octave"] and a["beat_coincidence"] >= 0.8 for a in agreements.values()):
        level = "medium"
    elif f["beat_f"] >= 0.8 and f["downbeat_f"] >= 0.6:
        level = "medium"
    ambiguous = len(groups) > 1
    if ambiguous and level == "high":
        level = "medium"
    if f["beat_f"] < 0.5:
        level = "low"
    return {"provider": chosen, "fit": dict(f), "agreement": agreements, "confidence": level,
            "octave_ambiguous": ambiguous,
            "octave_votes": {f"{fits[g[0]]['bpm']:.2f}": list(g) for g in groups}}


# ------------------------------------------------------------------ phrases
def snap_to_downbeats(times: Iterable[float], downbeats: np.ndarray) -> list[dict[str, Any]]:
    """Snap section boundaries to the nearest downbeat (index, error in bars)."""
    out = []
    if downbeats.size < 2:
        return out
    bar = float(np.median(np.diff(downbeats)))
    for t in times:
        j = int(np.argmin(np.abs(downbeats - t)))
        err = (t - downbeats[j]) / bar
        out.append({"time_s": float(t), "bar_index": j, "snap_error_bars": float(err),
                    "loose": bool(abs(err) > SECTION_SNAP_MAX_BARS)})
    return out


def merge_section_sources(primary: Sequence[Mapping[str, Any]],
                          supports: Mapping[str, Sequence[float]],
                          downbeats: np.ndarray, duration_s: float) -> list[dict[str, Any]]:
    """Structural sections snapped to downbeats, with cross-provider support.

    ``primary`` are labelled sections ``{start_s, end_s, label}`` (All-In-One
    or LinkSeg); ``supports`` maps provider name to boundary times.  When
    ``primary`` is empty the first support source defines unlabeled sections.
    Each section start gets ``bar_index`` (downbeat index) and the list of
    providers with a boundary within one bar.
    """
    if not primary:
        name = next(iter(supports), None)
        times = sorted(supports[name]) if name else []
        times = [0.0] + [t for t in times if t > 1.0]
        ends = times[1:] + [duration_s]
        primary = [{"start_s": s, "end_s": e, "label": "unlabeled", "source": name}
                   for s, e in zip(times, ends)]
    bar = float(np.median(np.diff(downbeats))) if downbeats.size > 1 else 2.0
    sections: list[dict[str, Any]] = []
    for sec in primary:
        label = str(sec.get("label", "unlabeled")).lower()
        if label in ("silence", "start", "end") and sec["end_s"] - sec["start_s"] < bar:
            continue
        snap = snap_to_downbeats([sec["start_s"]], downbeats)
        if not snap:
            continue
        s = snap[0]
        if abs(s["snap_error_bars"]) > 1.0 and sec["start_s"] > bar:
            continue  # boundary outside the bar grid (e.g. after the last downbeat)
        support = [n for n, ts in supports.items() if n != sec.get("source")
                   and any(abs(t - sec["start_s"]) <= bar for t in ts)]
        entry = {"label": label, "start_s": float(sec["start_s"]), "end_s": float(sec["end_s"]),
                 "bar_index": s["bar_index"], "downbeat_s": float(downbeats[s["bar_index"]]),
                 "snap_error_bars": s["snap_error_bars"], "loose": s["loose"],
                 "support": support, "source": sec.get("source"),
                 "n_other_sources": len([n for n in supports if n != sec.get("source")])}
        if sections and entry["bar_index"] - sections[-1]["bar_index"] < MIN_SECTION_BARS:
            # boundaries collapse onto one downbeat, or a section shorter than
            # MIN_SECTION_BARS on the grid (e.g. an intro that starts before the
            # first beat): keep the better supported boundary, the later on ties
            if len(entry["support"]) >= len(sections[-1]["support"]):
                sections[-1] = entry
            continue
        sections.append(entry)
    return sections


def phrase_table(downbeats: np.ndarray, sections: Sequence[Mapping[str, Any]],
                 levels: Sequence[int] = PHRASE_LEVELS) -> list[dict[str, Any]]:
    """For each downbeat: anchoring section, bar offset and phrase level.

    Phrases are counted from the downbeat of the latest *anchoring* section
    start; level is the largest of 16/8/4 that divides the offset (0 = not a
    phrase start).  A section start anchors (resets the count) when another
    provider supports it, when no other provider exists, or when it already
    lies on the 4-bar grid of the previous anchor; an unsupported, off-grid
    boundary keeps its label but does not shift the phrase grid.  Before the
    first section the count runs backwards from the first anchor.
    """
    secs = sorted((s for s in sections if s["bar_index"] >= 0), key=lambda s: s["bar_index"])
    anchors: list[int] = []
    for s in secs:
        j = int(s["bar_index"])
        last = anchors[-1] if anchors else 0
        supported = bool(s.get("support")) and not s.get("loose", False)
        if not anchors or supported or s.get("n_other_sources", 1) == 0 or (j - last) % 4 == 0:
            anchors.append(j)
    by_bar = {int(s["bar_index"]): s for s in secs}
    starts = sorted(by_bar)
    rows = []
    for j, t in enumerate(downbeats):
        prev_anchor = [a for a in anchors if a <= j]
        # before the first section, count backwards from the first anchor
        anchor = prev_anchor[-1] if prev_anchor else (anchors[0] if anchors else 0)
        offset = j - anchor
        level = next((lv for lv in levels if offset % lv == 0), 0)
        prev = [b for b in starts if b <= j]
        sec = by_bar.get(prev[-1]) if prev else None
        nxt = [b for b in starts if b > j]
        rows.append({"bar_index": j, "time_s": float(t), "anchor_bar": anchor,
                     "bar_in_section": offset, "phrase_level": level,
                     "section_start": j in by_bar, "anchors_phrases": j in anchors,
                     "section_label": sec["label"] if sec else "unlabeled",
                     "bars_to_next_section": (nxt[0] - j) if nxt else None})
    return rows


# --------------------------------------------------------------------- keys
def normalize_key(name: str) -> tuple[str, str]:
    """``'A minor'``, ``'Am'``, ``'a min'``, ``'Bb major'`` -> (tonic, 'major'|'minor')."""
    s = name.strip().replace("♯", "#").replace("♭", "b")
    parts = s.split()
    if len(parts) == 1:
        tok = parts[0]
        mode = "minor" if tok.endswith("m") or tok.endswith("min") else "major"
        if tok.endswith("min"):
            tok = tok[:-3]
        elif tok.endswith("m"):
            tok = tok[:-1]
        tonic = tok
    else:
        tonic, mode_raw = parts[0], parts[1].lower()
        mode = "minor" if mode_raw.startswith("min") else "major"
    tonic = tonic[0].upper() + tonic[1:]
    tonic = _FLATS.get(tonic, tonic)
    if tonic not in PITCH_CLASSES:
        raise ValueError(f"unknown tonic in key {name!r}")
    return tonic, mode


def camelot(name: str) -> str:
    tonic, mode = normalize_key(name)
    return f"{(_CAMELOT_MINOR if mode == 'minor' else _CAMELOT_MAJOR)[tonic]}{'A' if mode == 'minor' else 'B'}"


def camelot_relation(a: str, b: str) -> dict[str, Any]:
    """Relation of Camelot codes ``a`` -> ``b``: same/adjacent/relative/diagonal/clash."""
    na, la = int(a[:-1]), a[-1]
    nb, lb = int(b[:-1]), b[-1]
    step = min((na - nb) % 12, (nb - na) % 12)
    if step == 0 and la == lb:
        rel = "same"
    elif step == 0:
        rel = "relative"
    elif step == 1 and la == lb:
        rel = "adjacent"
    elif step == 1:
        rel = "diagonal"
    else:
        rel = "clash"
    return {"relation": rel, "wheel_steps": step, "mode_change": la != lb}


def krumhansl_key(chroma_mean: Sequence[float]) -> dict[str, Any]:
    """Fallback key from a mean chroma vector (Krumhansl-Kessler correlation)."""
    ch = np.asarray(chroma_mean, dtype=float)
    scores = []
    for k in range(12):
        scores.append((float(np.corrcoef(ch, np.roll(KK_MAJOR, k))[0, 1]), f"{PITCH_CLASSES[k]} major"))
        scores.append((float(np.corrcoef(ch, np.roll(KK_MINOR, k))[0, 1]), f"{PITCH_CLASSES[k]} minor"))
    scores.sort(reverse=True)
    return {"key": scores[0][1], "r": scores[0][0], "second": scores[1][1], "r2": scores[1][0],
            "margin": scores[0][0] - scores[1][0]}


def combine_key_estimates(estimates: Mapping[str, Mapping[str, Any]],
                          primary: Sequence[str] = ("madmom_cnn", "essentia_edma", "essentia_bgate",
                                                    "krumhansl")) -> dict[str, Any]:
    """Merge key estimates; confidence from probability, stability and agreement.

    Each estimate: ``{key, prob?, window_stability?}``.  The first available
    estimator in ``primary`` gives the key.  Confidence (0-1) =
    mean of: its probability (if any), its windowed stability (if any), and the
    fraction of other estimators whose key is the same or the relative key.
    """
    avail = [n for n in primary if n in estimates] + [n for n in estimates if n not in primary]
    if not avail:
        return {"key": None, "camelot": None, "confidence": 0.0, "level": "unknown"}
    name = avail[0]
    est = estimates[name]
    cam = camelot(est["key"])
    parts = []
    if est.get("prob") is not None:
        parts.append(float(est["prob"]))
    if est.get("window_stability") is not None:
        parts.append(float(est["window_stability"]))
    others = [n for n in avail[1:]]
    agree_exact = [n for n in others if camelot(estimates[n]["key"]) == cam]
    agree_rel = [n for n in others
                 if camelot_relation(cam, camelot(estimates[n]["key"]))["relation"] in ("same", "relative")]
    if others:
        parts.append(len(agree_rel) / len(others))
    conf = float(np.mean(parts)) if parts else 0.0
    level = "high" if conf >= 0.7 else "medium" if conf >= KEY_CONFIDENT else "low"
    return {"key": est["key"], "camelot": cam, "source": name, "confidence": conf, "level": level,
            "agree_exact": agree_exact, "agree_same_or_relative": agree_rel,
            "estimates": {n: {**estimates[n], "camelot": camelot(estimates[n]["key"])} for n in avail}}


# ------------------------------------------------------------ vocals/energy
def interval_coverage(intervals: Sequence[tuple[float, float]], starts: np.ndarray,
                      ends: np.ndarray) -> np.ndarray:
    """Fraction of each ``[starts[i], ends[i])`` covered by ``intervals``."""
    cov = np.zeros(len(starts))
    if not len(intervals):
        return cov
    iv = sorted((float(a), float(b)) for a, b in intervals)
    for i, (s, e) in enumerate(zip(starts, ends)):
        tot = 0.0
        for a, b in iv:
            if b <= s:
                continue
            if a >= e:
                break
            tot += min(b, e) - max(a, s)
        cov[i] = tot / max(e - s, 1e-9)
    return np.clip(cov, 0.0, 1.0)


def bar_bounds(downbeats: np.ndarray, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(downbeats, dtype=float)
    ends = np.append(starts[1:], min(duration_s, starts[-1] + (np.median(np.diff(starts)) if starts.size > 1 else 2.0)))
    return starts, ends


def vocal_regions(bar_vocal: np.ndarray, downbeats: np.ndarray, ends: np.ndarray,
                  thr: float = VOCAL_BAR_ACTIVE) -> list[dict[str, Any]]:
    out, start = [], None
    for i, v in enumerate(list(bar_vocal) + [0.0]):
        if v >= thr and start is None:
            start = i
        elif v < thr and start is not None:
            out.append({"start_s": float(downbeats[start]), "end_s": float(ends[i - 1]),
                        "bars": i - start, "first_bar": start})
            start = None
    return out


def section_energy(bar_db: np.ndarray, bar_vocal: np.ndarray,
                   sections: Sequence[Mapping[str, Any]], n_bars: int) -> list[dict[str, Any]]:
    """Mean bar RMS (dB, relative to the track median bar) and vocal share per section."""
    med = float(np.median(bar_db)) if bar_db.size else 0.0
    starts = [int(max(0, s["bar_index"])) for s in sections]
    out = []
    for i, s in enumerate(sections):
        a = starts[i]
        b = starts[i + 1] if i + 1 < len(starts) else n_bars
        if b <= a:
            continue
        out.append({"label": s["label"], "start_bar": a, "end_bar": b, "bars": b - a,
                    "start_s": s["downbeat_s"], "energy_db_rel": float(np.mean(bar_db[a:b]) - med),
                    "vocal_share": float(np.mean(bar_vocal[a:b]))})
    return out


# -------------------------------------------------------------- pair gates
def tempo_compatibility(bpm_a: float, bpm_b: float,
                        max_stretch: float = TEMPO_MAX_STRETCH) -> dict[str, Any]:
    """Stretch of B to A after folding half/double time."""
    best = None
    for factor in (0.5, 1.0, 2.0):
        target = bpm_a * factor   # B played at A's tempo (or half/double of it)
        stretch = target / bpm_b - 1.0
        if best is None or abs(stretch) < abs(best[1]):
            best = (factor, stretch)
    factor, stretch = best
    return {"bpm_a": bpm_a, "bpm_b": bpm_b, "octave_factor": factor,
            "stretch_b_pct": float(100 * stretch), "compatible": bool(abs(stretch) <= max_stretch),
            "long_blend_ok": bool(abs(stretch) <= TEMPO_LONG_BLEND_STRETCH)}


def key_compatibility(key_a: Mapping[str, Any], key_b: Mapping[str, Any]) -> dict[str, Any]:
    if not key_a.get("camelot") or not key_b.get("camelot"):
        return {"relation": "unknown", "compatible": None, "confident": False, "confidence": 0.0}
    rel = camelot_relation(key_a["camelot"], key_b["camelot"])
    conf = min(float(key_a.get("confidence", 0.0)), float(key_b.get("confidence", 0.0)))
    ok = rel["relation"] in ("same", "adjacent", "relative")
    return {**rel, "camelot_a": key_a["camelot"], "camelot_b": key_b["camelot"],
            "compatible": bool(ok), "confidence": conf, "confident": bool(conf >= KEY_CONFIDENT)}


def pair_compatibility(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """Tempo and key gate for the ordered pair A -> B (track analyses)."""
    t = tempo_compatibility(a["tempo"]["bpm"], b["tempo"]["bpm"])
    k = key_compatibility(a["key"], b["key"])
    grid_levels = {a["tempo"]["confidence"], b["tempo"]["confidence"]}
    tempo_conf = "low" if "low" in grid_levels else "medium" if "medium" in grid_levels else "high"
    compatible = bool(t["compatible"] and k["compatible"])
    reasons = []
    reasons.append(f"tempo {t['bpm_a']:.1f}->{t['bpm_b']:.1f} BPM, stretch B {t['stretch_b_pct']:+.1f}%"
                   + (" (ottava piegata x%g)" % t["octave_factor"] if t["octave_factor"] != 1 else "")
                   + (" entro" if t["compatible"] else " OLTRE") + f" +-{100 * TEMPO_MAX_STRETCH:.0f}%")
    reasons.append(f"tonalita {k.get('camelot_a')}->{k.get('camelot_b')}: {k['relation']}"
                   f" (confidenza {k['confidence']:.2f})")
    score = 0.0
    s = abs(t["stretch_b_pct"]) / 100
    if not t["compatible"]:
        score -= 3.0
    elif s > TEMPO_FREE_STRETCH:
        score -= (s - TEMPO_FREE_STRETCH) / (TEMPO_MAX_STRETCH - TEMPO_FREE_STRETCH)
    key_pen = {"same": 0.0, "relative": -0.25, "adjacent": -0.25, "diagonal": -1.0,
               "clash": -2.0, "unknown": -1.0}[k["relation"]]
    if not k["confident"]:
        key_pen = min(key_pen, -0.25) if k["relation"] != "same" else -0.25
    score += key_pen
    return {"tempo": t, "key": k, "tempo_confidence": tempo_conf, "compatible": compatible,
            "status": ("compatible" if compatible and k["confident"] and tempo_conf != "low"
                       else "compatible_uncertain" if compatible else "incompatible"),
            "score": score, "reasons": reasons}


# ------------------------------------------------------------ cue planning
def _vocal_bars(track: Mapping[str, Any], start_bar: int, n: int) -> np.ndarray:
    v = np.asarray(track["bars"]["vocal"], dtype=float)
    seg = v[start_bar:start_bar + n]
    return seg


def _in_windows(role: str, frac: float, left: int, t_s: float, dur_s: float, w: Mapping[str, Optional[float]]) -> bool:
    if role == "out":
        return (left >= w["out_min_bars_left"] and frac >= w["out_min_position"]
                and (w.get("out_min_s") is None or t_s >= w["out_min_s"]))
    return (left >= w["in_min_bars_left"] and frac <= w["in_max_position"]
            and (w.get("in_min_s_left") is None or dur_s - t_s >= w["in_min_s_left"]))


def cue_candidates(track: Mapping[str, Any], role: str, min_level: int = 4,
                   windows: Optional[Mapping[str, Optional[float]]] = None) -> list[dict[str, Any]]:
    """Phrase-start downbeats usable as cue-out (role 'out') or cue-in ('in').

    ``windows`` (option): a :data:`CUE_WINDOWS` entry whose cues are added to the default ones (never removes a
    default cue); None = the default windows (v1). With ``windows`` each candidate also says whether it lies
    inside the default windows (``windows_default_ok``).
    """
    rows = track["phrases"]
    n_bars = len(rows)
    dur = track["duration_s"]
    w = CUE_WINDOWS["default"] if windows is None else windows
    out = []
    for r in rows:
        if r["phrase_level"] < min_level:
            continue
        j = r["bar_index"]
        frac = r["time_s"] / dur
        if role not in ("out", "in"):
            continue
        ok_default = _in_windows(role, frac, n_bars - j, r["time_s"], dur, CUE_WINDOWS["default"])
        # a named window only ADDS cues: the default candidates are always kept
        if not (ok_default or (windows is not None and _in_windows(role, frac, n_bars - j, r["time_s"], dur, w))):
            continue
        c = {"bar_index": j, "time_s": r["time_s"], "phrase_level": r["phrase_level"],
             "section_start": r["section_start"], "section_label": r["section_label"],
             "prev_label": rows[j - 1]["section_label"] if j > 0 else "start",
             "position": frac}
        if windows is not None:
            c["windows_default_ok"] = ok_default
        out.append(c)
    return out


def _energy(track: Mapping[str, Any], a: int, b: int) -> float:
    e = np.asarray(track["bars"]["energy_db_rel"], dtype=float)
    a, b = max(0, a), min(len(e), b)
    return float(np.mean(e[a:b])) if b > a else 0.0


def cut_vocal_clear(track: Mapping[str, Any], bar_index: int,
                    max_cover: float = CUT_GUARD_MAX_COVER) -> bool:
    """True if A's vocal covers at most ``max_cover`` of the beat before ``bar_index``."""
    return _last_beat_vocal(track, bar_index) <= max_cover


def vocal_clear_window(times_s: np.ndarray, active: np.ndarray, cut_s: float,
                       before_s: float, after_s: float = 0.0) -> bool:
    """True if no frame of a vocal activity mask is active in ``[cut - before, cut + after)``.

    ``times_s`` are frame centres, ``active`` the boolean mask (e.g. stem RMS
    above threshold).  Pure helper for a driver-side, audio-based cut guard.
    """
    t = np.asarray(times_s, dtype=float)
    a = np.asarray(active, dtype=bool)
    sel = (t >= cut_s - before_s) & (t < cut_s + after_s)
    return not bool(a[sel].any())


def relocate_cut(A: Mapping[str, Any], B: Mapping[str, Any], plan: Mapping[str, Any],
                 pair: Mapping[str, Any], is_clear=None, max_bars: int = CUT_GUARD_MAX_BARS) -> dict[str, Any]:
    """Nearest valid phrase-start cue-out of A (earlier or later) whose exit is clear.

    ``is_clear(cue) -> bool`` defaults to :func:`cut_vocal_clear` on the cue's
    bar.  Candidates come from :func:`cue_candidates` (same phrase and position
    rules as the planner), ordered by distance in bars, the earlier one first
    on ties; B's cue-in is kept and the new pair of cues is re-scored with
    :func:`plan_transition`.  Returns ``status`` = ``"clear"`` (the original
    cue is already clear), ``"moved"`` (with ``plan``) or ``"none"``.
    """
    test = is_clear or (lambda c: cut_vocal_clear(A, c["bar_index"]))
    ja = int(plan["a_out_bar"])
    outs = {c["bar_index"]: c for c in cue_candidates(A, "out")}
    orig = outs.get(ja) or {"bar_index": ja, "time_s": plan["a_out_s"]}
    if test(orig):
        return {"status": "clear", "plan": dict(plan), "tried": []}
    inn = next((c for c in cue_candidates(B, "in") if c["bar_index"] == plan["b_in_bar"]), None)
    order = sorted((j for j in outs if j != ja and abs(j - ja) <= max_bars), key=lambda j: (abs(j - ja), j > ja))
    tried = []
    for j in order:
        ok = bool(test(outs[j]))
        tried.append({"bar_index": j, "time_s": outs[j]["time_s"], "clear": ok})
        if ok:
            new = plan_transition(A, B, outs[j], inn, pair) if inn is not None else {"a_out_bar": j, "a_out_s": outs[j]["time_s"]}
            return {"status": "moved", "plan": new, "from_bar": ja, "to_bar": j, "shift_bars": j - ja, "tried": tried}
    return {"status": "none", "plan": None, "tried": tried}


#: [H] (21/10/2026, INGRESSO_B): B may be cued up to 2 bars before the chosen section start (the event still arrives,
#: whole, 1-2 bars after B is audible: Krims 2000 pdf pp. 112-113 'upbeat' bars; B&B 2003 cap. 10 'How to match phrases' (EPUB unit 28):
#: "start halfway through a phrase ... will throw your dancers off"); farther cues come from cue_candidates.
ENTRY_BACK_BARS = (1, 2)
ENTRY_MAX_BARS = 16


def bar_cue(track: Mapping[str, Any], bar_index: int) -> Optional[dict[str, Any]]:
    """Cue dict (as :func:`cue_candidates`) for any bar of the track's phrase table, or None outside it."""
    rows = track["phrases"]
    j = int(bar_index)
    if not 0 <= j < len(rows):
        return None
    r = rows[j]
    return {"bar_index": j, "time_s": r["time_s"], "phrase_level": r["phrase_level"], "section_start": r["section_start"],
            "section_label": r["section_label"], "prev_label": rows[j - 1]["section_label"] if j > 0 else "start",
            "position": r["time_s"] / track["duration_s"], "lead_bar": True}


def relocate_entry(A: Mapping[str, Any], B: Mapping[str, Any], plan: Mapping[str, Any], pair: Mapping[str, Any],
                   is_clear, back_bars: Sequence[int] = ENTRY_BACK_BARS, max_bars: int = ENTRY_MAX_BARS) -> dict[str, Any]:
    """Nearest cue-in of B whose entry keeps B's first phrase whole; A's cue-out is kept (symmetric of :func:`relocate_cut`).

    ``is_clear(cue) -> bool | dict`` (a dict needs ``ok``; :func:`musicmix.dj_phrase.entry_clear_test`). Order [H]: the
    original cue (``clear``); the same cue 1-2 bars earlier (``lead_bars``: the section start of B still arrives); the
    other :func:`cue_candidates` of B by distance in bars, the earlier first on ties. Returns ``status`` = ``clear``,
    ``moved`` (with ``plan`` re-scored by :func:`plan_transition`) or ``none``; ``tried`` lists every test.
    """
    def ok(c):
        r = is_clear(c)
        return (bool(r.get("ok")), r) if isinstance(r, Mapping) else (bool(r), {"ok": bool(r)})

    jb = int(plan["b_in_bar"])
    ins = {c["bar_index"]: c for c in cue_candidates(B, "in")}
    orig = ins.get(jb) or bar_cue(B, jb) or {"bar_index": jb, "time_s": plan["b_in_s"]}
    good, info = ok(orig)
    if good:
        return {"status": "clear", "plan": dict(plan), "entry": info, "tried": []}
    outs = {c["bar_index"]: c for c in cue_candidates(A, "out")}
    out = outs.get(int(plan["a_out_bar"])) or bar_cue(A, int(plan["a_out_bar"]))
    order = [(jb - k, "lead_bars") for k in back_bars if jb - k >= 0]
    order += [(j, "cue_candidate") for j in sorted((j for j in ins if j != jb and abs(j - jb) <= max_bars),
                                                   key=lambda j: (abs(j - jb), j > jb))]
    tried = [{"bar_index": jb, "time_s": orig["time_s"], "clear": False, "kind": "original", "status": info.get("status"),
              "reasons": info.get("reasons")}]
    seen = {jb}
    for j, kind in order:
        if j in seen:
            continue
        seen.add(j)
        c = ins.get(j) if kind == "cue_candidate" else (ins.get(j) or bar_cue(B, j))
        if c is None:
            continue
        g, inf = ok(c)
        tried.append({"bar_index": j, "time_s": c["time_s"], "clear": g, "kind": kind, "status": inf.get("status"),
                      "reasons": inf.get("reasons")})
        if g:
            new = plan_transition(A, B, out, c, pair) if out is not None else dict(plan, b_in_bar=j, b_in_s=c["time_s"])
            new["entry_relocation"] = {"from_bar": jb, "to_bar": j, "kind": kind, "shift_bars": j - jb}
            return {"status": "moved", "plan": new, "entry": inf, "from_bar": jb, "to_bar": j, "shift_bars": j - jb, "kind": kind,
                    "tried": tried}
    return {"status": "none", "plan": None, "entry": info, "tried": tried}


def plan_transition(A: Mapping[str, Any], B: Mapping[str, Any], out: Mapping[str, Any],
                    inn: Mapping[str, Any], pair: Mapping[str, Any],
                    cut_vocal_guard: bool = False) -> dict[str, Any]:
    """Choose transition type and score one (cue_out, cue_in) candidate.

    ``cut_vocal_guard``: a cut whose last beat before the cue carries A's vocal
    (:func:`cut_vocal_clear` false) becomes ``echo_out`` (exempt exit).
    """
    reasons: list[str] = []
    score = 0.0
    t, k = pair["tempo"], pair["key"]
    nA, nB = len(A["phrases"]), len(B["phrases"])
    ja, jb = out["bar_index"], inn["bar_index"]

    # --- transition type --------------------------------------------------
    ttype, bars, clash = "cut", 0, None
    duck = False
    if not t["compatible"]:
        reasons.append("cut: tempo fuori +-8%, niente sovrapposizione")
    elif k["relation"] == "clash" and k["confident"]:
        reasons.append("cut: tonalita incompatibili con confidenza, niente sovrapposizione armonica")
    else:
        max_bars = 16
        if not t["long_blend_ok"]:
            max_bars = 8
        if k["relation"] in ("diagonal", "clash", "unknown") or not k["confident"]:
            max_bars = min(max_bars, 4)
        for n in BLEND_BARS:
            if n > max_bars or ja + n > nA or jb + n > nB:
                continue
            va = _vocal_bars(A, ja, n) >= VOCAL_BAR_ACTIVE
            vb = _vocal_bars(B, jb, n) >= VOCAL_BAR_ACTIVE
            c = int(np.sum(va & vb))
            if c <= CLASH_TOLERATED_BARS:
                ttype, bars, clash = "blend", n, c
                reasons.append(f"blend {n} battute: voci sovrapposte {c} battute (tollerata <= {CLASH_TOLERATED_BARS})")
                break
        if ttype == "cut":
            vb0 = _vocal_bars(B, jb, 1)
            if ja + 4 <= nA and jb + 4 <= nB and abs(t["stretch_b_pct"]) <= 100 * TEMPO_LONG_BLEND_STRETCH \
                    and not (vb0.size and vb0[0] >= VOCAL_BAR_ACTIVE):
                ttype, bars, duck = "blend_duck", 4, True
                va = _vocal_bars(A, ja, 4) >= VOCAL_BAR_ACTIVE
                vb = _vocal_bars(B, jb, 4) >= VOCAL_BAR_ACTIVE
                clash = int(np.sum(va & vb))
                reasons.append(f"blend 4 battute con ducking della voce di A: {clash} battute con entrambe le voci")
            else:
                reasons.append("cut sull'uno: entrambe le voci attive nella sovrapposizione"
                               + (" e B entra cantando" if vb0.size and vb0[0] >= VOCAL_BAR_ACTIVE else ""))
    if cut_vocal_guard and ttype == "cut" and not cut_vocal_clear(A, ja):
        ttype = "echo_out"
        reasons.append(f"cut non ammesso: voce di A su {_last_beat_vocal(A, ja):.2f} dell'ultimo battito"
                       f" (> {CUT_GUARD_MAX_COVER:.2f}); echo out (uscita con effetto, esente)")

    # --- cue terms -------------------------------------------------------
    lvl_bonus = {16: 1.0, 8: 0.6, 4: 0.3}
    score += lvl_bonus.get(out["phrase_level"], 0.0)
    score += lvl_bonus.get(inn["phrase_level"], 0.0)
    reasons.append(f"A esce su downbeat di inizio frase da {out['phrase_level']} (battuta {ja}),"
                   f" B entra su frase da {inn['phrase_level']} (battuta {jb})")
    if out["section_start"]:
        score += 1.0
        reasons.append(f"A esce a confine di sezione ({out['prev_label']} -> {out['section_label']})")
    if inn["section_start"]:
        score += 1.0
        reasons.append(f"B entra a inizio sezione '{inn['section_label']}'")
    if ttype in ("cut", "echo_out"):
        if out["prev_label"] in SECTION_CORE and out["section_start"]:
            score += 0.3
            reasons.append(f"cut a fine {out['prev_label']} di A")
        if inn["section_label"] in SECTION_CORE | {"intro"} and inn["section_start"]:
            score += 0.3
        last_beat = _last_beat_vocal(A, ja)
        if ttype == "cut" and last_beat >= VOCAL_BAR_ACTIVE:
            score -= 0.5
            reasons.append("rischio parola troncata: voce di A attiva nell'ultimo beat prima del cut")
    else:
        if out["section_label"] in SECTION_OUT_FRIENDLY:
            score += 0.5
            reasons.append(f"A in sovrapposizione su '{out['section_label']}'")
        if inn["section_label"] in SECTION_IN_FRIENDLY:
            score += 0.5
            reasons.append(f"B in sovrapposizione su '{inn['section_label']}'")
        va = float(np.mean(_vocal_bars(A, ja, bars))) if bars else 0.0
        vb = float(np.mean(_vocal_bars(B, jb, bars))) if bars else 0.0
        if va < 0.25:
            score += 0.5
            reasons.append(f"poca voce di A nella sovrapposizione ({va:.2f})")
        if B["phrases"][min(jb + bars, nB - 1)]["section_start"]:
            score += 0.3
            reasons.append("dopo la sovrapposizione B arriva a un inizio sezione")
        if duck:
            score -= 0.3
        reasons.append(f"quota voce nella sovrapposizione A {va:.2f}, B {vb:.2f}")
    # position (M-DJCUE: OUT not before A's core, IN within B's intro region) [P]/[H]
    if out["position"] < 0.5:
        score -= 0.5
        reasons.append(f"A esce presto ({100 * out['position']:.0f}% del brano)")
    if inn["position"] > 0.35:
        score -= 0.5
        reasons.append(f"B entra tardi ({100 * inn['position']:.0f}% del brano)")
    # energy continuity (relative to each track's median bar) [H]
    e_a = _energy(A, ja - 4, ja)
    e_b = _energy(B, jb, jb + 4)
    if e_b - e_a < -6.0:
        score -= 0.5
        reasons.append(f"calo di energia {e_b - e_a:+.1f} dB (relativo)")
    score_total = score + pair["score"]
    conf = _plan_confidence(A, B, pair, out, inn)
    return {"a_out_s": out["time_s"], "a_out_bar": ja, "b_in_s": inn["time_s"], "b_in_bar": jb,
            "type": ttype, "blend_bars": bars, "duck_a_vocal": duck, "vocal_clash_bars": clash,
            "a_out_phrase_level": out["phrase_level"], "b_in_phrase_level": inn["phrase_level"],
            "a_section": f"{out['prev_label']}->{out['section_label']}" if out["section_start"] else out["section_label"],
            "b_section": inn["section_label"], "energy_a_db_rel": e_a, "energy_b_db_rel": e_b,
            "cue_score": score, "pair_score": pair["score"], "score": score_total,
            "confidence": conf, "reasons": reasons}


def _last_beat_vocal(track: Mapping[str, Any], bar_index: int) -> float:
    lb = track["bars"].get("vocal_last_beat")
    if lb is None or bar_index <= 0:
        return 0.0
    return float(lb[bar_index - 1])


def _plan_confidence(A, B, pair, out, inn) -> str:
    levels = [A["tempo"]["confidence"], B["tempo"]["confidence"]]
    if not pair["key"]["confident"]:
        levels.append("low")
    for t, c in ((A, out), (B, inn)):
        if c["section_start"]:
            sec = next((s for s in t["sections"] if s["bar_index"] == c["bar_index"]), None)
            if sec is not None and (sec["loose"] or not sec["support"]):
                levels.append("medium")
    return "low" if "low" in levels else "medium" if "medium" in levels else "high"


def plan_pair(A: Mapping[str, Any], B: Mapping[str, Any], top_k: int = 5,
              cut_vocal_guard: bool = False, phrase: Optional[Mapping[str, Any]] = None,
              windows: Optional[Mapping[str, float]] = None) -> dict[str, Any]:
    """All cue candidates for A -> B, ranked; returns pair gate and top plans.

    ``phrase`` (option, default ``None`` = unchanged plans): ``{"a": model_a, "b": model_b,
    "act_a": ..., "act_b": ...}`` with :func:`musicmix.dj_phrase.analyze_track` models (and
    optional stem activities). The plans are then the cut-family plans of
    :func:`musicmix.dj_phrase.plan_pair_phrase`: A leaves at a verse end (switch up to two beats
    before the downbeat, last word's tail protected), B enters from its pickup with its downbeat
    on A's downbeat; ``cut_vocal_guard`` is ignored (the phrase test replaces it).

    ``windows`` (option, default None = v1 windows): a :data:`CUE_WINDOWS` entry for the cue candidates;
    each plan then carries ``windows_default_ok`` (both cues inside the default windows).
    """
    if phrase is not None:
        from musicmix import dj_phrase  # lazy: dj_phrase imports this module

        return dj_phrase.plan_pair_phrase(A, B, phrase["a"], phrase["b"], act_a=phrase.get("act_a"),
                                          act_b=phrase.get("act_b"), top_k=top_k)
    pair = pair_compatibility(A, B)
    if windows is None:
        plans = [plan_transition(A, B, o, i, pair, cut_vocal_guard=cut_vocal_guard)
                 for o in cue_candidates(A, "out") for i in cue_candidates(B, "in")]
    else:
        plans = []
        for o in cue_candidates(A, "out", windows=windows):
            for i in cue_candidates(B, "in", windows=windows):
                pl = plan_transition(A, B, o, i, pair, cut_vocal_guard=cut_vocal_guard)
                pl["windows_default_ok"] = bool(o["windows_default_ok"] and i["windows_default_ok"])
                plans.append(pl)
    plans.sort(key=lambda p: (-p["score"], p["a_out_s"], p["b_in_s"]))
    return {"a": A["slug"], "b": B["slug"], "compatibility": pair,
            "n_candidates": len(plans), "plans": plans[:top_k]}


def rank_pairs(pairs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Order pairs: compatible (confident) first, then by best plan score."""
    status_rank = {"compatible": 0, "compatible_uncertain": 1, "incompatible": 2}
    return sorted(pairs, key=lambda p: (status_rank[p["compatibility"]["status"]],
                                         -(p["plans"][0]["score"] if p["plans"] else -99)))


__all__ = [name for name in dir() if not name.startswith("_") and name not in (
    "annotations", "dataclass", "math", "np", "Any", "Iterable", "Mapping", "Optional", "Sequence")]
