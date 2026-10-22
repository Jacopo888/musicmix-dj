#!/usr/bin/env python3
"""Statistiche di fine frase e pickup dal corpus MCFlow (Condit-Schultz 2017), per dj_phrase.

Legge i file Humdrum ``*.rap`` del repository ufficiale
https://github.com/Computational-Cognitive-Musicology-Lab/MCFlow (clone locale, NON
copiato nel repo: nessuna licenza esplicita nel repository; il paper EMR e' CC BY-NC 4.0
e il sito dichiara il dataset "freely available"). Salva solo statistiche derivate.

Convenzioni (paper EMR 11(2) Tab. 2): ``**break`` codifica la disgiunzione *prima*
della sillaba (0 debole, 1 normale, 2 sotto-frase, 3 frase, 4 iper-frase). Quattro
file usano anche valori negativi sull'ultima sillaba della frase: trattati come
confine *dopo* quella sillaba (ridondante con il valore positivo seguente).
Frase = break >= 3, come nel paper (p. 141). Unita' metrica: sedicesimi in 4/4.

Uscita: probabilita' di confine di frase per confine di parola condizionata a
posizione metrica dell'ultima sillaba, pausa dopo la parola, rima sulla sillaba
accentata finale, punteggiatura, allungamento; distribuzione dei pickup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from musicmix.paths import external_dir  # noqa: E402


def recip_quarters(tok: str) -> float | None:
    tok = tok.strip()
    m = re.match(r"^(\d+)%(\d+)(\.*)$", tok)
    if m:
        base = 4.0 * int(m.group(2)) / int(m.group(1))
        dots = len(m.group(3))
    else:
        m = re.match(r"^(\d+)(\.*)", tok)
        if not m:
            return None
        n = int(m.group(1))
        base = 16.0 if n == 0 else 4.0 / n
        dots = len(m.group(2))
    return base * (2 - 0.5 ** dots)


def parse(path: Path) -> list[dict]:
    """Syllable/rest events with absolute time (quarters), bar index and in-bar position."""
    events = []
    t = 0.0
    bar_start = 0.0
    bar = None
    meter_ok = True
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("!"):
            continue
        f = line.split("\t")
        if f[0].startswith("*"):
            if f[0].startswith("*M") and f[0][2:3].isdigit() and f[0] != "*M4/4":
                meter_ok = False
            continue
        if f[0].startswith("="):
            m = re.match(r"=(\d+)", f[0])
            bar = int(m.group(1)) if m else (bar + 1 if bar is not None else 0)
            bar_start = t
            continue
        if not meter_ok:
            continue
        d = recip_quarters(f[0])
        if d is None:
            continue
        ipa = f[5] if len(f) > 5 else "."
        lyr = f[6] if len(f) > 6 else "."
        rest = ipa.startswith("R") or lyr == "."
        brk = f[3] if len(f) > 3 else "."
        try:
            b = int(brk)
        except ValueError:
            b = None
        events.append({"t": t, "dur": d, "bar": bar, "pos": t - bar_start, "rest": rest,
                       "stress": f[1], "break": b, "rhyme": f[4], "lyr": lyr})
        t += d
    return events


def words_of(events: list[dict]) -> list[dict]:
    """Group syllables into words via the '-' convention; attach rests after each word."""
    sylls = [e for e in events if not e["rest"]]
    words: list[dict] = []
    cur: list[dict] = []
    for s in sylls:
        lyr = s["lyr"]
        if cur and not lyr.startswith("-"):
            words.append({"syl": cur})
            cur = []
        cur.append(s)
        if not lyr.endswith("-"):
            words.append({"syl": cur})
            cur = []
    if cur:
        words.append({"syl": cur})
    return words


def pos16(q: float) -> int:
    return int(math.floor((q % 4.0) * 4 + 1e-6)) % 16


def gap_bin(g16: float) -> str:
    if g16 < 0.5:
        return "0"
    if g16 < 1.5:
        return "1"
    if g16 < 2.5:
        return "2"
    if g16 < 4.5:
        return "3-4"
    if g16 < 8.5:
        return "5-8"
    return ">8"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(external_dir() / "computational-cognitive-musicology-lab/mcflow"))
    ap.add_argument("--out", default="runs/dj_phrase_v1/mcflow_stats.json")
    ap.add_argument("--config", default="configs/dj_phrase_mcflow_v1.json")
    args = ap.parse_args()
    root = Path(args.corpus)
    files = sorted((root / "Humdrum").glob("*.rap"))
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())

    tab = {k: defaultdict(lambda: [0, 0]) for k in ("end_pos16", "gap16", "rhyme", "punct", "lengthening", "end_beat", "pos_gap", "rhyme_gap")}
    n_bound = n_all = 0
    pickup16 = Counter()
    start_pos = Counter()
    end_pos = Counter()
    hyper_end_bar_mod = Counter()
    across_downbeat = [0, 0]
    db_pickup = [0, 0]  # bar lines just after a pickup start / strictly inside a phrase
    phrase_start_near_db = Counter()
    rests_total = [0, 0]  # (word gaps with rest >= 1/16, of which phrase boundaries)
    n_phrases = 0
    for f in files:
        ev = parse(f)
        if not ev:
            continue
        # boundaries: positive break >= 3 on a syllable = boundary before it; negative <= -3 = after it
        sylls = [e for e in ev if not e["rest"]]
        for i, s in enumerate(sylls):
            s["bound_before"] = bool(s["break"] is not None and s["break"] >= 3)
            s["hyper_before"] = bool(s["break"] is not None and s["break"] >= 4)
        for i, s in enumerate(sylls[:-1]):
            if s["break"] is not None and s["break"] <= -3:
                sylls[i + 1]["bound_before"] = True
                if s["break"] <= -4:
                    sylls[i + 1]["hyper_before"] = True
        words = words_of(ev)
        durs = [s["dur"] for s in sylls]
        med = sorted(durs)[len(durs) // 2] if durs else 0.25
        # every bar line inside the rapped span: inside a phrase, or between phrases
        # (clean = previous phrase ended before it and the next one starts on/after it: no pickup)
        bar_t = {}
        for e in ev:
            if e["bar"] is not None and e["bar"] not in bar_t:
                bar_t[e["bar"]] = e["t"] - e["pos"]
        for b, tb in bar_t.items():
            before = [s for s in sylls if s["t"] < tb - 1e-6]
            after = [s for s in sylls if s["t"] >= tb - 1e-6]
            if not before or not after:
                continue
            seg = [s for s in after[:1]]
            # boundary between the last syllable before tb and the first at/after tb?
            between_bound = after[0]["bound_before"]
            # pickup: a phrase starts within the beat before tb
            pick = [s for s in before if s["bound_before"] and tb - s["t"] <= 1.0 + 1e-6]
            across_downbeat[1] += 1
            if between_bound:
                across_downbeat[0] += 1  # clean: phrase boundary exactly between the two syllables
            elif pick:
                db_pickup[0] += 1
            else:
                db_pickup[1] += 1
        for wi in range(len(words) - 1):
            w, nx = words[wi], words[wi + 1]
            last = w["syl"][-1]
            nfirst = nx["syl"][0]
            end_t = last["t"] + last["dur"]
            gap = (nfirst["t"] - end_t) * 4  # sixteenths of silence after the word
            is_b = nfirst["bound_before"]
            n_all += 1
            n_bound += int(is_b)
            p = pos16(last["pos"])
            # rhyme on the last stressed syllable of the word (uppercase letter in **rhyme)
            stressed = [s for s in w["syl"] if s["stress"] == "1"] or w["syl"]
            rh = any(re.search(r"[A-Z]", s["rhyme"] or "") for s in stressed[-1:])
            rh_any = any(re.search(r"[A-Za-z]", s["rhyme"] or "") for s in w["syl"])
            punct = bool(re.search(r"[.;:,!?]$", last["lyr"].rstrip("-")))
            leng = last["dur"] / med
            lbin = "<=1" if leng <= 1.0 + 1e-6 else "1-2" if leng <= 2.0 + 1e-6 else ">2"
            beat = int(last["pos"] // 1) + 1
            g = gap_bin(gap)
            for key, val in (("end_pos16", str(p)), ("gap16", g), ("rhyme", "stressed" if rh else ("other" if rh_any else "none")),
                             ("punct", "yes" if punct else "no"), ("lengthening", lbin), ("end_beat", str(beat)),
                             ("pos_gap", f"{beat}|{'rest' if gap >= 0.5 else 'none'}"),
                             ("rhyme_gap", f"{'rhyme' if rh else 'no'}|{'rest' if gap >= 0.5 else 'none'}")):
                tab[key][val][0] += int(is_b)
                tab[key][val][1] += 1
            if gap >= 0.5:
                rests_total[0] += 1
                rests_total[1] += int(is_b)
        # phrase starts / ends
        starts = [i for i, s in enumerate(sylls) if s["bound_before"]]
        for k, i in enumerate(starts):
            s = sylls[i]
            n_phrases += 1
            start_pos[pos16(s["pos"])] += 1
            j = (starts[k + 1] - 1) if k + 1 < len(starts) else None
            if j is not None and j >= i:
                end_pos[pos16(sylls[j]["pos"])] += 1
                if sylls[j + 1]["hyper_before"] and sylls[j]["bar"] is not None:
                    hyper_end_bar_mod[str(sylls[j]["bar"] % 4)] += 1
            # pickup: phrase starts in the last 4 sixteenths... up to one beat before a downbeat, and continues across it
            to_next_db = 4.0 - (s["pos"] % 4.0)
            if s["pos"] % 4.0 < 1e-6:
                phrase_start_near_db["on_downbeat"] += 1
            elif to_next_db <= 1.0 + 1e-6:
                phrase_start_near_db["pickup_le_1_beat"] += 1
                pickup16[str(int(round(to_next_db * 4)))] += 1
            elif to_next_db <= 2.0 + 1e-6:
                phrase_start_near_db["pickup_1_2_beats"] += 1
            elif s["pos"] % 4.0 <= 1.0:
                phrase_start_near_db["after_downbeat_le_1_beat"] += 1
            else:
                phrase_start_near_db["mid_bar"] += 1

    base = n_bound / max(n_all, 1)

    def table(d):
        out = {}
        for k, (b, n) in sorted(d.items()):
            p = b / n if n else None
            out[k] = {"boundary": b, "n": n, "p_boundary": p,
                      "likelihood_ratio": (p / (1 - p)) / (base / (1 - base)) if p not in (None, 0, 1) else None}
        return out

    res = {
        "schema": "musicmix.mcflow_phrase_stats.1",
        "source": {"corpus": "MCFlow (Condit-Schultz 2017, EMR 11(2):124-147, doi:10.18061/emr.v11i2.4961)",
                   "repo": "https://github.com/Computational-Cognitive-Musicology-Lab/MCFlow", "commit": commit,
                   "n_files": len(files), "files_sha256": h.hexdigest(),
                   "license_note": "nessun file LICENSE nel repository; paper CC BY-NC 4.0; corpus non ridistribuito, solo statistiche"},
        "definitions": {"boundary": "next word starts with **break >= 3 (or this word ends with break <= -3)",
                        "unit": "sixteenth notes in 4/4; end_pos16 = onset position of the word's last syllable",
                        "gap16": "rest between word offset and next word onset (sixteenths)"},
        "n_word_boundaries": n_all, "n_phrase_boundaries": n_bound, "base_rate": base,
        "n_phrases": n_phrases,
        "tables": {k: table(v) for k, v in tab.items()},
        "phrase_start_pos16": dict(sorted(start_pos.items())),
        "phrase_end_pos16": dict(sorted(end_pos.items())),
        "phrase_start_vs_downbeat": dict(phrase_start_near_db),
        "pickup_len16_le_1_beat": dict(sorted(pickup16.items(), key=lambda x: int(x[0]))),
        "hyper_phrase_end_bar_mod4": dict(hyper_end_bar_mod),
        "bar_lines_in_rapped_span": {"n": across_downbeat[1],
                                     "clean_boundary": across_downbeat[0],
                                     "after_pickup_le_1_beat": db_pickup[0],
                                     "inside_phrase": db_pickup[1],
                                     "fraction_clean": across_downbeat[0] / max(across_downbeat[1], 1),
                                     "note": "clean = phrase boundary falls between the last syllable before the bar line and the first at/after it, with no phrase start in the beat before"},
        "word_gaps_with_rest": {"n": rests_total[0], "phrase_boundary": rests_total[1],
                                "p_boundary_given_rest": rests_total[1] / max(rests_total[0], 1)},
    }
    res["heldout"] = heldout(files)
    res["detected_rhyme"] = detected_rhyme_eval(files)
    tabd = defaultdict(lambda: [0, 0])
    for f in files:
        for r in detected_rhyme_records(f):
            k = f"{'rhyme' if r['rhyme'] else 'no'}|{'rest' if r['rest'] else 'none'}"
            tabd[k][0] += int(r["y"])
            tabd[k][1] += 1
    res["tables"]["rhyme_gap_detected"] = {k: {"boundary": b, "n": n, "p_boundary": b / n, "likelihood_ratio": None}
                                           for k, (b, n) in sorted(tabd.items())}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    write_config(res, files, Path(args.config))
    print(json.dumps({k: res[k] for k in ("n_word_boundaries", "n_phrase_boundaries", "base_rate", "n_phrases",
                                          "phrase_start_vs_downbeat", "pickup_len16_le_1_beat",
                                          "bar_lines_in_rapped_span", "word_gaps_with_rest",
                                          "phrase_end_pos16", "hyper_phrase_end_bar_mod4")}, indent=1))
    for k in ("end_beat", "gap16", "rhyme", "punct", "lengthening", "pos_gap", "rhyme_gap"):
        print(k, {kk: (round(v["p_boundary"], 3), v["n"]) for kk, v in res["tables"][k].items()})




# ---------------------------------------------------------------- held-out
def word_records(path: Path) -> list[dict]:
    """Per word boundary: features used by dj_phrase + label, in MCFlow units."""
    ev = parse(path)
    sylls = [e for e in ev if not e["rest"]]
    for s in sylls:
        s["bound_before"] = bool(s["break"] is not None and s["break"] >= 3)
    for i, s in enumerate(sylls[:-1]):
        if s["break"] is not None and s["break"] <= -3:
            sylls[i + 1]["bound_before"] = True
    words = words_of(ev)
    durs = sorted(s["dur"] for s in sylls)
    med = durs[len(durs) // 2] if durs else 0.25
    recs = []
    onsets = [w["syl"][0]["t"] for w in words]
    for wi in range(len(words) - 1):
        w, nx = words[wi], words[wi + 1]
        last = w["syl"][-1]
        gap = (nx["syl"][0]["t"] - (last["t"] + last["dur"])) * 4
        stressed = [s for s in w["syl"] if s["stress"] == "1"] or w["syl"]
        rh = any(re.search(r"[A-Z]", s["rhyme"] or "") for s in stressed[-1:])
        leng = last["dur"] / med
        recs.append({"beat": int(last["pos"] // 1) + 1, "rest": gap >= 0.5, "rhyme": rh,
                     "leng": "<=1" if leng <= 1.0 + 1e-6 else "1-2" if leng <= 2.0 + 1e-6 else ">2",
                     "punct": bool(re.search(r"[.;:,!?]$", last["lyr"].rstrip("-"))),
                     "ioi_next": min(16, max(1, int(round((onsets[wi + 1] - onsets[wi]) * 4)))),
                     "y": nx["syl"][0]["bound_before"]})
    return recs


def _auc(scores, ys):
    pos = [s for s, y in zip(scores, ys) if y]
    neg = [s for s, y in zip(scores, ys) if not y]
    if not pos or not neg:
        return None
    import bisect
    neg_sorted = sorted(neg)
    tot = 0.0
    for s in pos:
        lo = bisect.bisect_left(neg_sorted, s)
        hi = bisect.bisect_right(neg_sorted, s)
        tot += lo + 0.5 * (hi - lo)
    return tot / (len(pos) * len(neg))


class IOIModel:
    """Interpolated (Witten-Bell) n-gram over word-onset IOIs in sixteenths, orders 0..N (PPM-like, IDyOM-style IC)."""

    def __init__(self, order: int = 3):
        self.order = order
        self.counts: dict[tuple, Counter] = defaultdict(Counter)
        self.alphabet = list(range(1, 17))

    def fit(self, seqs):
        for seq in seqs:
            for i, x in enumerate(seq):
                for n in range(self.order + 1):
                    if i - n < 0:
                        break
                    self.counts[tuple(seq[i - n:i])][x] += 1
        return self

    def prob(self, ctx: tuple, x: int) -> float:
        p = 1.0 / len(self.alphabet)
        for n in range(0, self.order + 1):
            if n > len(ctx):
                break
            c = self.counts.get(tuple(ctx[len(ctx) - n:]) if n else ())
            if not c:
                continue
            tot = sum(c.values())
            types = len(c)
            lam = tot / (tot + types)
            p = lam * c.get(x, 0) / tot + (1 - lam) * p
        return p

    def ic(self, seq):
        return [-math.log2(self.prob(tuple(seq[max(0, i - self.order):i]), x)) for i, x in enumerate(seq)]


def heldout(files: list[Path], folds: int = 2) -> dict:
    """Naive-Bayes boundary model trained on half the songs, tested on the other half (and swapped)."""
    per_file = {f.name: word_records(f) for f in files}
    names = sorted(per_file)
    res = {"folds": folds, "split": "by song, alphabetical index mod folds", "models": {}}
    feats = {
        "pos_gap": lambda r: (f"{r['beat']}|{r['rest']}",),
        "pos_gap+rhyme": lambda r: (f"{r['beat']}|{r['rest']}", f"rh{r['rhyme']}|{r['rest']}"),
        "pos_gap+rhyme+leng": lambda r: (f"{r['beat']}|{r['rest']}", f"rh{r['rhyme']}|{r['rest']}", f"l{r['leng']}"),
        "pos_gap+rhyme+leng+punct": lambda r: (f"{r['beat']}|{r['rest']}", f"rh{r['rhyme']}|{r['rest']}", f"l{r['leng']}", f"p{r['punct']}"),
        "gap_only": lambda r: (f"{r['rest']}",),
        "position_only": lambda r: (f"{r['beat']}",),
        "rhyme_only": lambda r: (f"rh{r['rhyme']}",),
    }
    scores = {k: ([], []) for k in list(feats) + ["ioi_ic_next", "pos_gap+rhyme+leng+ic"]}
    for k in range(folds):
        train = [n for i, n in enumerate(names) if i % folds != k]
        test = [n for i, n in enumerate(names) if i % folds == k]
        tr = [r for n in train for r in per_file[n]]
        base = sum(r["y"] for r in tr) / len(tr)
        ioi = IOIModel(3).fit([[r["ioi_next"] for r in per_file[n]] for n in train])
        # IC terciles on train for the NB channel
        tr_ic = []
        for n in train:
            seq = [r["ioi_next"] for r in per_file[n]]
            tr_ic += ioi.ic(seq)
        q = sorted(tr_ic)
        cuts = (q[len(q) // 3], q[2 * len(q) // 3])
        icbin = lambda v: "ic_lo" if v <= cuts[0] else "ic_mid" if v <= cuts[1] else "ic_hi"  # noqa: E731
        for n in train:
            for r, v in zip(per_file[n], ioi.ic([r["ioi_next"] for r in per_file[n]])):
                r["_ic"] = v
        for n in test:
            for r, v in zip(per_file[n], ioi.ic([r["ioi_next"] for r in per_file[n]])):
                r["_ic"] = v

        def nb(fn, extra=None):
            tabs = defaultdict(lambda: [1, 2])  # Laplace
            for r in tr:
                keys = fn(r) + ((extra(r),) if extra else ())
                for kk in keys:
                    tabs[kk][0] += int(r["y"])
                    tabs[kk][1] += 1

            def score(r):
                lo = math.log(base / (1 - base))
                for kk in fn(r) + ((extra(r),) if extra else ()):
                    b, nn = tabs[kk]
                    p = b / nn
                    lo += math.log(p / (1 - p)) - math.log(base / (1 - base))
                return lo
            return score
        te = [r for n in test for r in per_file[n]]
        for name, fn in feats.items():
            sc = nb(fn)
            scores[name][0].extend(sc(r) for r in te)
            scores[name][1].extend(r["y"] for r in te)
        scores["ioi_ic_next"][0].extend(r["_ic"] for r in te)
        scores["ioi_ic_next"][1].extend(r["y"] for r in te)
        sc = nb(feats["pos_gap+rhyme+leng"], extra=lambda r: icbin(r["_ic"]))
        scores["pos_gap+rhyme+leng+ic"][0].extend(sc(r) for r in te)
        scores["pos_gap+rhyme+leng+ic"][1].extend(r["y"] for r in te)
    for name, (s, y) in scores.items():
        # F1 at the MAP threshold (log-odds > 0) for NB models
        if name != "ioi_ic_next":
            tp = sum(1 for a, b in zip(s, y) if a > 0 and b)
            fp = sum(1 for a, b in zip(s, y) if a > 0 and not b)
            fn_ = sum(1 for a, b in zip(s, y) if a <= 0 and b)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn_, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        else:
            prec = rec = f1 = None
        res["models"][name] = {"auc": _auc(s, y), "precision_at_map": prec, "recall_at_map": rec, "f1_at_map": f1, "n": len(y)}
    return res


def ioi_model_dump(files: list[Path], order: int = 3) -> dict:
    m = IOIModel(order).fit([[r["ioi_next"] for r in word_records(f)] for f in files])
    return {"order": order, "alphabet": m.alphabet, "smoothing": "interpolated Witten-Bell (PPM-like), orders 0..order",
            "counts": {" ".join(map(str, k)): dict(v) for k, v in m.counts.items()}}


def write_config(res: dict, files: list[Path], path: Path) -> None:
    """Compact tables for src/musicmix/dj_phrase.py (tracked in configs/)."""
    t = res["tables"]
    keep = lambda d: {k: {"p": v["p_boundary"], "n": v["n"]} for k, v in d.items()}  # noqa: E731
    cfg = {"schema": "musicmix.dj_phrase_mcflow.1", "source": res["source"], "definitions": res["definitions"],
           "base_rate": res["base_rate"], "n_word_boundaries": res["n_word_boundaries"],
           "pos_gap": keep(t["pos_gap"]), "rhyme_gap": keep(t["rhyme_gap"]), "lengthening": keep(t["lengthening"]),
           "punct": keep(t["punct"]), "gap16": keep(t["gap16"]),
           "rhyme_gap_detected": keep(t["rhyme_gap_detected"]), "detected_rhyme": res["detected_rhyme"],
           "phrase_start_vs_downbeat": res["phrase_start_vs_downbeat"],
           "pickup_len16_le_1_beat": res["pickup_len16_le_1_beat"],
           "bar_lines_in_rapped_span": res["bar_lines_in_rapped_span"],
           "heldout": res["heldout"], "ioi_model": ioi_model_dump(files)}
    path.write_text(json.dumps(cfg, indent=0, ensure_ascii=False))



# ------------------------------------------------- detected-rhyme robustness
_IPA_V = set("aeiouæɑɒɔəɛɜɪʊʌIUEAOɝɚ")


def _ipa_tail(syl_ipas: list[str], stress_idx: int) -> tuple[str, str]:
    """Rhyme key (from the stressed vowel to the end of the word) and the stressed vowel, from MCFlow IPA."""
    s = syl_ipas[stress_idx]
    k = next((i for i, c in enumerate(s) if c in _IPA_V), 0)
    j = k
    while j < len(s) and (s[j] in _IPA_V or s[j] in "˞"):
        j += 1
    return s[k:] + "".join(syl_ipas[stress_idx + 1:]), s[k:j]


def detected_rhyme_records(path: Path, window_beats: float = 1.0, lags=(1, 2)) -> list[dict]:
    """word_records() with the annotated rhyme replaced by the dj_phrase-style detector run on MCFlow IPA."""
    ev = parse(path)
    recs = word_records(path)
    words = words_of(ev)
    info = []
    for w in words:
        ipas = []
        for s in w["syl"]:
            pass
        sy = w["syl"]
        stressed = [i for i, s in enumerate(sy) if s["stress"] == "1"]
        si = stressed[-1] if stressed else len(sy) - 1
        # MCFlow keeps the IPA in column 6; parse() did not store it, re-read from lyrics order
        info.append({"t": sy[-1]["t"], "si": si, "n": len(sy), "lyr": "".join(s["lyr"].strip("-") for s in sy).lower()})
    # re-read IPA per syllable (parse() skipped it): second pass over the file
    ipa_seq = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line[0] in "!*=":
            continue
        f = line.split("\t")
        if len(f) > 6 and not (f[5].startswith("R") or f[6] == "."):
            ipa_seq.append(f[5])
    k = 0
    for w, inf in zip(words, info):
        syl_ipas = ipa_seq[k:k + len(w["syl"])]
        k += len(w["syl"])
        inf["key"], inf["vowel"] = _ipa_tail(syl_ipas, inf["si"]) if syl_ipas else ("", "")
    stop = {"a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with", "i", "you", "he", "she",
            "it", "we", "they", "my", "your", "is", "be", "that", "this", "uh", "yeah", "oh", "like"}
    for wi, r in enumerate(recs):
        me = info[wi]
        best = False
        if me["lyr"].strip(".,;:!?") not in stop and me["key"]:
            for o in info:
                if o is me or o["lyr"].strip(".,;:!?") in stop:
                    continue
                dt = abs(o["t"] - me["t"])
                if not any(abs(dt - 4 * L) <= window_beats for L in lags):
                    continue
                if o["key"] == me["key"] or (o["vowel"] and o["vowel"] == me["vowel"]) or o["lyr"] == me["lyr"]:
                    best = True
                    break
        r["rhyme_annot"] = r["rhyme"]
        r["rhyme"] = best
    return recs


def detected_rhyme_eval(files: list[Path], folds: int = 2) -> dict:
    """Held-out NB (pos_gap + rhyme + lengthening) with detected instead of annotated rhyme."""
    per = {f.name: detected_rhyme_records(f) for f in files}
    names = sorted(per)
    agree = [(r["rhyme"], r["rhyme_annot"]) for n in names for r in per[n]]
    tp = sum(1 for a, b in agree if a and b)
    out = {"detector": "same tail or same stressed vowel or same word, 1-2 bars away +-1 beat, function words excluded (MCFlow IPA)",
           "precision_vs_annotation": tp / max(1, sum(1 for a, _ in agree if a)),
           "recall_vs_annotation": tp / max(1, sum(1 for _, b in agree if b))}
    scores, ys = [], []
    for k in range(folds):
        train = [n for i, n in enumerate(names) if i % folds != k]
        test = [n for i, n in enumerate(names) if i % folds == k]
        tr = [r for n in train for r in per[n]]
        base = sum(r["y"] for r in tr) / len(tr)
        tabs = defaultdict(lambda: [1, 2])
        keys = lambda r: (f"{r['beat']}|{r['rest']}", f"rh{r['rhyme']}|{r['rest']}", f"l{r['leng']}")  # noqa: E731
        for r in tr:
            for kk in keys(r):
                tabs[kk][0] += int(r["y"])
                tabs[kk][1] += 1
        for n in test:
            for r in per[n]:
                lo = math.log(base / (1 - base))
                for kk in keys(r):
                    b, nn = tabs[kk]
                    p = b / nn
                    lo += math.log(p / (1 - p)) - math.log(base / (1 - base))
                scores.append(lo)
                ys.append(r["y"])
    out["auc_pos_gap_rhyme_leng_detected"] = _auc(scores, ys)
    tp = sum(1 for a, b in zip(scores, ys) if a > 0 and b)
    fp = sum(1 for a, b in zip(scores, ys) if a > 0 and not b)
    fn_ = sum(1 for a, b in zip(scores, ys) if a <= 0 and b)
    out["precision_at_map"] = tp / max(tp + fp, 1)
    out["recall_at_map"] = tp / max(tp + fn_, 1)
    return out


if __name__ == "__main__":
    main()
