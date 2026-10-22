"""Affidabilità (scegliere i problemi che il sistema sa risolvere), copertura della libreria e varietà dei set.

Tre parti, tutte additive (nessuna modifica a ``dj_set``/``dj_select``; il runner le usa tramite
``SetHooks.reliability`` di ``tools/run_dj_set_v2_books.py``):

1. **Affidabilità del brano** (:func:`track_reliability`) dalla scheda ``musicmix.track-card.1``
   (:mod:`musicmix.track_cards`): prodotto di ``1 - gravità`` dei casi incerti non risolti, per categoria
   (``grid``, ``key``, ``phrase``, ``track``). ``planning`` esclude ``key`` (la tonalità conta solo quando A e B
   suonano insieme).
2. **Affidabilità di un candidato** (coppia + cue + tecnica, :func:`cue_reliability`): griglie di A e B, versi di A e B,
   stato del verso di A all'uscita e dell'ingresso di B (tabelle della scheda), tonalità e stretching solo con
   sovrapposizione. Con l'opzione ``prefer`` (:class:`ReliabilityHook`) il costo del candidato cresce di
   ``w_risk * (1 - affidabilità)`` e sotto ``pair_min`` il candidato non è ammissibile; un brano con ``planning`` sotto
   ``track_min`` non entra nel set (tutti i suoi archi vietati). ``report`` misura senza cambiare nulla.
3. **Copertura e varietà** (:func:`coverage`, :func:`variety`): quanti brani della libreria i set usano (anche nei set
   successivi), tecniche, coppie e ripetizioni tra set. Opzione di novità (spenta: pesi 0) che penalizza coppie e brani
   già usati nei set precedenti.

Tutti i pesi e le soglie sono [H] (ipotesi nostre, mai tarate su un ascolto): servono a preferire le situazioni con
analisi solide a parità di costo musicale, non a dire che una transizione è buona.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

RELIABILITY_VERSION = "musicmix.reliability.1"

PARAMS: dict[str, tuple[Any, str]] = {
    "track_min": (0.35, "[H] affidabilità di pianificazione del brano sotto 0,35: il brano non entra nel set (es. un caso con "
                        "gravità 0,5 e altri due medi)"),
    "pair_min": (0.30, "[H] candidato (coppia+cue+tecnica) sotto 0,30: non ammissibile"),
    "w_risk": (0.6, "[H] costo aggiunto per unità di rischio (1 - affidabilità); scala dei costi di dj_set (un arco del set v3 "
                    "costa in media ~1,9, un candidato 0,1-0,6)"),
    "exit_factor": ({"clean": 1.0, "tail": 1.0, "a_pickup": 1.0, "instrumental": 1.0, "gap_not_line_end": 0.85, "no_gap": 0.75,
                     "mid_line": 0.6, "asr_unreliable": 0.8, "unknown": 0.8},
                    "[H] stato del verso di A all'uscita (dj_phrase.exit_point): a metà verso il verso troncato si sente anche con "
                    "un effetto (PIANO_EVENTI_IMPL §2: T2, T6, T7 'da ascoltare')"),
    "cut_not_clean_factor": (0.5, "[H] cut su un'uscita che non è fine verso pulita (exit_clear_test falso)"),
    "entry_factor": ({"instrumental": 1.0, "on_downbeat": 1.0, "lead_in": 0.9, "mid_line": 0.5, "unknown": 0.8},
                     "[H] ingresso di B (dj_phrase.b_lead_in): il lead-in richiede un render in più; a metà frase la guardia "
                     "dell'ingresso di B lo sposta o lo vieta già"),
    "unknown_cue_factor": (0.8, "[H] cue non presente nella scheda (spostato o fuori finestra): non verificato"),
    "overlap_not_recommended_factor": (0.6, "[H] sovrapposizione su un brano con griglia incerta (scheda: stretch.overlap_recommended)"),
    "stretch_over_limit_factor": (0.6, "[H] stretching oltre il limite consigliato dalla scheda"),
    "overlap_types": (("blend",), "come dj_set.OVERLAP_TYPES: A e B suonano insieme per più battute"),
    "w_pair_repeat": (0.0, "[H] novità (spenta): costo in più per una coppia A->B già suonata in un set precedente"),
    "w_track_repeat": (0.0, "[H] novità (spenta): costo in più per entrare in un brano usato nel set precedente"),
}
NOVELTY_ON = {"w_pair_repeat": 0.3, "w_track_repeat": 0.15}  # [H] valori dell'opzione --novelty (vedi nota)


def values(overrides: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    p = {k: v[0] for k, v in PARAMS.items()}
    for k, v in (overrides or {}).items():
        if k not in p:
            raise KeyError(f"parametro di affidabilità sconosciuto: {k}")
        p[k] = v
    return p


def documented(overrides: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    p = values(overrides)
    return {k: {"value": list(p[k]) if isinstance(p[k], tuple) else p[k], "source": PARAMS[k][1]} for k in PARAMS}


# ------------------------------------------------------------------ track
def open_flags(card: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [f for f in card.get("uncertain") or [] if not f.get("resolved_by")]


def _prod(xs: Iterable[float]) -> float:
    out = 1.0
    for x in xs:
        out *= max(0.0, min(1.0, float(x)))
    return out


def category_factor(card: Mapping[str, Any], category: str) -> float:
    return _prod(1.0 - float(f["severity"]) for f in open_flags(card) if f.get("category") == category)


def track_reliability(card: Mapping[str, Any]) -> dict[str, Any]:
    """Affidabilità del brano dalla scheda: ``value`` (tutte le categorie), ``planning`` (senza ``key``)."""
    comps = {c: category_factor(card, c) for c in ("grid", "key", "phrase", "track")}
    excluded = bool(card.get("excluded"))
    value = 0.0 if excluded else _prod(comps.values())
    planning = 0.0 if excluded else comps["grid"] * comps["phrase"] * comps["track"]
    return {"version": RELIABILITY_VERSION, "value": round(value, 4), "planning": round(planning, 4),
            "components": {k: round(v, 4) for k, v in comps.items()}, "excluded": excluded,
            "open_flags": [{"id": f["id"], "severity": f["severity"], "category": f.get("category")} for f in open_flags(card)],
            "rule": "[H] prodotto di (1 - gravità) dei casi incerti non risolti; planning esclude la tonalità"}


# ------------------------------------------------------------------ cards
def load_cards(directory: str | Path) -> dict[str, dict[str, Any]]:
    d = Path(directory)
    out = {}
    for p in sorted(d.glob("*.json")):
        if p.name in ("index.json", "corrections.json"):
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        if doc.get("schema_version") == "musicmix.track-card.1":
            out[doc["slug"]] = doc
    return out


def card_matches(card: Mapping[str, Any], analysis: Mapping[str, Any]) -> bool:
    """La scheda descrive la stessa griglia dell'analisi usata dal runner (stesso tempo e stesse battute)?"""
    g = card.get("grid") or {}
    try:
        return abs(float(g.get("period_s")) - float(analysis["tempo"]["period_s"])) < 1e-6 and \
            int(g.get("n_bars") or -1) == len(analysis["grid"]["downbeats_s"])
    except (TypeError, KeyError, ValueError):
        return False


# ------------------------------------------------------------------- cues
def _row(rows: Sequence[Mapping[str, Any]], bar: int) -> Optional[Mapping[str, Any]]:
    for r in rows:
        if int(r["bar"]) == int(bar):
            return r
    return None


def cue_reliability(card_a: Optional[Mapping[str, Any]], card_b: Optional[Mapping[str, Any]], plan: Mapping[str, Any],
                    decision: Mapping[str, Any], params: Optional[Mapping[str, Any]] = None,
                    stretch_pct: Optional[float] = None) -> dict[str, Any]:
    """Affidabilità [H] di un candidato A -> B (cue ``plan``, tecnica ``decision``) dalle due schede."""
    p = values(params)
    f: dict[str, float] = {}
    why: list[str] = []
    overlap = decision.get("type") in p["overlap_types"]
    if card_a is None or card_b is None:
        f["no_card"] = p["unknown_cue_factor"]
        why.append("scheda assente: " + ", ".join(x for x, c in (("A", card_a), ("B", card_b)) if c is None))
    for side, card in (("a", card_a), ("b", card_b)):
        if card is None:
            continue
        f[f"grid_{side}"] = category_factor(card, "grid")
        f[f"phrase_{side}"] = category_factor(card, "phrase")
        if overlap:
            f[f"key_{side}"] = category_factor(card, "key")
            st = card.get("stretch") or {}
            if st and not st.get("overlap_recommended", True):
                f[f"overlap_{side}"] = p["overlap_not_recommended_factor"]
                why.append(f"sovrapposizione sconsigliata per {card['slug']} (griglia incerta)")
    if overlap and stretch_pct is not None:  # un solo fattore: lo stretching è uno, il limite è il più stretto dei due
        lims = [(float((c.get("stretch") or {})["blend_max_pct"]), c["slug"]) for c in (card_a, card_b)
                if c is not None and (c.get("stretch") or {}).get("blend_max_pct") is not None]
        if lims and abs(stretch_pct) > min(lims)[0] + 1e-9:
            f["stretch"] = p["stretch_over_limit_factor"]
            why.append(f"stretching {abs(stretch_pct):.1f} % > {min(lims)[0]:g} % consigliato per {min(lims)[1]}")
    if card_a is not None:
        r = _row(card_a.get("exits") or [], plan["a_out_bar"])
        if r is None:
            f["exit"] = p["unknown_cue_factor"]
            why.append(f"uscita di A a battuta {plan['a_out_bar']} non nella scheda")
        else:
            if r.get("manual"):
                f["exit"] = 1.0 if r["manual"].get("ok") else p["exit_factor"]["mid_line"]
            else:
                f["exit"] = p["exit_factor"].get(r.get("status"), p["exit_factor"]["unknown"])
            if decision.get("type") == "cut" and r.get("clean_for_cut") is False:
                f["exit_cut"] = p["cut_not_clean_factor"]
                why.append("cut senza fine verso pulita")
            if f["exit"] < 1.0:
                why.append(f"A esce in stato '{r.get('status')}' (battuta {r['bar']})")
    if card_b is not None and not overlap:
        r = _row(card_b.get("entries") or [], plan["b_in_bar"])
        li = plan.get("b_lead_in") or {}
        if r is None and li.get("status"):  # cue spostato dalla guardia dell'ingresso di B (dj_set.entry_guard): già verificato
            f["entry"] = p["entry_factor"].get(li["status"], p["entry_factor"]["unknown"])
            if f["entry"] < 1.0:
                why.append(f"B entra in stato '{li['status']}' (battuta {plan['b_in_bar']}, cue spostato dalla guardia)")
        elif r is None:
            f["entry"] = p["unknown_cue_factor"]
            why.append(f"ingresso di B a battuta {plan['b_in_bar']} non nella scheda")
        elif r.get("manual"):
            f["entry"] = 1.0 if r["manual"].get("ok") else p["entry_factor"]["mid_line"]
        else:
            f["entry"] = p["entry_factor"].get(r.get("status"), p["entry_factor"]["unknown"])
            if f["entry"] < 1.0:
                why.append(f"B entra in stato '{r.get('status')}' (battuta {r['bar']})")
    value = _prod(f.values())
    return {"value": round(value, 4), "factors": {k: round(v, 4) for k, v in f.items()}, "overlap": overlap, "why": why}


# ------------------------------------------------------------------- hook
class ReliabilityHook:
    """Hook del runner (``SetHooks.reliability``): ``adjust_pair`` dopo i filtri della coppia, ``report`` nel set.json.

    ``mode``: ``"report"`` misura e registra senza cambiare costi né ammissibilità (confronto); ``"prefer"`` applica
    rischio e soglie. ``history``: set precedenti (``[{"order": [...], "pairs": [[a, b], ...]}]``) per l'opzione di
    novità (pesi ``w_pair_repeat``/``w_track_repeat``, 0 di default).
    """

    def __init__(self, cards: Mapping[str, Mapping[str, Any]], mode: str = "prefer", params: Optional[Mapping[str, Any]] = None,
                 history: Sequence[Mapping[str, Any]] = (), cards_dir: Optional[str] = None):
        if mode not in ("report", "prefer"):
            raise ValueError(f"mode {mode!r}")
        self.cards = dict(cards)
        self.mode = mode
        self.overrides = dict(params or {})
        self.p = values(params)
        self.history = list(history)
        self.cards_dir = cards_dir
        self.hist_pairs = {tuple(x) for h in self.history for x in h.get("pairs") or []}
        self.hist_tracks = set(self.history[-1].get("order") or []) if self.history else set()
        self.tracks: dict[str, dict[str, Any]] = {}
        self.pairs: dict[tuple[str, str], dict[str, Any]] = {}
        self.stale: dict[str, str] = {}

    # ---- per brano
    def track(self, slug: str, analysis: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        if slug in self.tracks:
            return self.tracks[slug]
        card = self.cards.get(slug)
        if card is None:
            row = {"slug": slug, "card": False, "planning": None, "avoid": False, "why": "nessuna scheda: rischio non misurato"}
        else:
            tr = track_reliability(card)
            row = {"slug": slug, "card": True, "value": tr["value"], "planning": tr["planning"], "components": tr["components"],
                   "open_flags": [x["id"] for x in tr["open_flags"]], "audio_sha256": (card.get("audio") or {}).get("sha256"),
                   "input_key": card.get("input_key")}
            if analysis is not None and not card_matches(card, analysis):
                self.stale[slug] = "la scheda descrive un'altra griglia: rifare la scheda"
                row["stale"] = self.stale[slug]
            row["avoid"] = bool(tr["planning"] < self.p["track_min"])
            row["why"] = (f"affidabilità di pianificazione {tr['planning']:.2f} < {self.p['track_min']:.2f}: "
                          + ", ".join(x["id"] for x in tr["open_flags"] if x.get("category") != "key")) if row["avoid"] else None
        self.tracks[slug] = row
        return row

    def card_for(self, slug: str) -> Optional[Mapping[str, Any]]:
        return None if slug in self.stale else self.cards.get(slug)

    # ---- per coppia
    def adjust_pair(self, pc: dict[str, Any], a: str, b: str, analyses: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        ta, tb = self.track(a, analyses.get(a)), self.track(b, analyses.get(b))
        ca, cb = self.card_for(a), self.card_for(b)
        cands = pc["candidates"]
        adm0 = [c for c in cands if c["admissible"]]
        best0 = adm0[0] if adm0 else None
        stretch = ((pc.get("compatibility") or {}).get("tempo") or {}).get("stretch_b_pct")
        if stretch is None:
            try:
                from musicmix import dj_set as ds

                stretch = 100.0 * ds.folded_tempo(analyses[a]["tempo"]["bpm"], analyses[b]["tempo"]["bpm"])["stretch"]
            except Exception:  # pragma: no cover - analisi senza tempo
                stretch = None
        avoid_track = [t["slug"] for t in (ta, tb) if t.get("avoid")]
        n_low = 0
        for c in cands:
            r = cue_reliability(ca, cb, c["plan"], c["decision"], self.overrides, stretch_pct=stretch)
            nov = 0.0
            if self.history:
                if (a, b) in self.hist_pairs:
                    nov += float(self.p["w_pair_repeat"])
                if b in self.hist_tracks:
                    nov += float(self.p["w_track_repeat"])
            r["novelty_cost"] = nov
            c["reliability"] = r
            if self.mode != "prefer":
                continue
            c["cost"] = float(c["cost"]) + float(self.p["w_risk"]) * (1.0 - r["value"]) + nov
            if c["admissible"] and (avoid_track or r["value"] < self.p["pair_min"]):
                c["admissible"] = False
                n_low += 1
                why = (f"affidabilità: brano evitato ({', '.join(avoid_track)})" if avoid_track else
                       f"affidabilità {r['value']:.2f} < {self.p['pair_min']:.2f}: " + "; ".join(r["why"][:3]))
                c.setdefault("notes", []).append(why)
        if self.mode == "prefer":
            cands.sort(key=lambda c: (not c["admissible"], c["cost"], -c["plan"]["score"]))
        adm1 = [c for c in cands if c["admissible"]]
        best1 = adm1[0] if adm1 else None

        def sig(c):
            return None if c is None else [int(c["plan"]["a_out_bar"]), int(c["plan"]["b_in_bar"]), c["decision"]["type_key"]]

        vals = [c["reliability"]["value"] for c in adm0] or [0.0]
        self.pairs[(a, b)] = {"a": a, "b": b, "admissible_before": len(adm0), "admissible_after": len(adm1),
                              "made_inadmissible": n_low, "avoided_tracks": avoid_track,
                              "best_before": sig(best0), "best_after": sig(best1),
                              "best_before_reliability": best0["reliability"]["value"] if best0 is not None else None,
                              "best_after_reliability": best1["reliability"]["value"] if best1 is not None else None,
                              "max_reliability": round(max(vals), 4), "best_before_why": best0["reliability"]["why"] if best0 is not None else None}
        return pc

    # ---- report nel set.json
    def report(self, ctx: Mapping[str, Any]) -> dict[str, Any]:
        order, chosen = list(ctx["order"]), list(ctx["chosen"])
        trans = []
        for t, c in enumerate(chosen):
            r = c.get("reliability") or {}
            trans.append({"index": t + 1, "a": order[t], "b": order[t + 1], "a_out_bar": int(c["plan"]["a_out_bar"]),
                          "b_in_bar": int(c["plan"]["b_in_bar"]), "type": c["decision"]["type_key"], "rule": c["decision"].get("rule"),
                          "cost": float(c["cost"]), "reliability": r.get("value"), "factors": r.get("factors"), "why": r.get("why"),
                          "novelty_cost": r.get("novelty_cost")})
        pairs = list(self.pairs.values())
        lost = [x for x in pairs if x["admissible_before"] > 0 and x["admissible_after"] == 0]
        changed = [x for x in pairs if x["admissible_after"] > 0 and x["best_before"] != x["best_after"]]
        rel = [x["reliability"] for x in trans if x["reliability"] is not None]
        return {"version": RELIABILITY_VERSION, "mode": self.mode, "cards_dir": self.cards_dir, "params": documented(self.overrides),
                "history": {"n_sets": len(self.history), "pairs": len(self.hist_pairs), "tracks_last_set": sorted(self.hist_tracks)},
                "tracks": dict(sorted(self.tracks.items())), "tracks_avoided": sorted(s for s, x in self.tracks.items() if x.get("avoid")),
                "tracks_without_card": sorted(s for s, x in self.tracks.items() if not x.get("card")), "stale_cards": dict(self.stale),
                "transitions": trans,
                "set_reliability": {"mean": round(sum(rel) / len(rel), 4) if rel else None, "min": min(rel) if rel else None,
                                    "product": round(_prod(rel), 4) if rel else None},
                "pairs": {"n": len(pairs), "made_unplannable": len(lost), "best_changed": len(changed),
                          "unplannable": [{"a": x["a"], "b": x["b"], "avoided_tracks": x["avoided_tracks"],
                                           "best_before": x["best_before"], "best_before_reliability": x["best_before_reliability"],
                                           "why": x["best_before_why"]} for x in lost],
                          "changed": changed}}


# ------------------------------------------------------- coverage / variety
def set_summary(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Ordine, coppie e tecniche di un set.json (render o solo piano) in una forma comune."""
    if doc.get("transitions"):
        tr = [{"a": t["a"], "b": t["b"], "technique": t.get("type"), "a_out_bar": (t.get("plan") or {}).get("a_out_bar"),
               "b_in_bar": (t.get("plan") or {}).get("b_in_bar")} for t in doc["transitions"]]
    elif (doc.get("event_plans") or {}).get("transitions"):
        tr = []
        for x in doc["event_plans"]["transitions"]:
            ep = x["event_plan"]
            tr.append({"a": x["a"], "b": x["b"], "technique": (ep.get("choice") or {}).get("type_key"),
                       "a_out_bar": (ep.get("a_event") or {}).get("exit_bar"), "b_in_bar": (ep.get("b_entry") or {}).get("downbeat_bar")})
    elif (doc.get("reliability") or {}).get("transitions"):
        tr = [{"a": x["a"], "b": x["b"], "technique": x["type"], "a_out_bar": x["a_out_bar"], "b_in_bar": x["b_in_bar"]}
              for x in doc["reliability"]["transitions"]]
    else:
        tr = []
    order = [e["slug"] for e in doc["tracklist"]] if doc.get("tracklist") else list((doc.get("ordering") or {}).get("order") or [])
    if not order and tr:
        order = [tr[0]["a"]] + [x["b"] for x in tr]
    return {"order": order, "pairs": [[x["a"], x["b"]] for x in tr], "transitions": tr}


def coverage(sets: Sequence[Mapping[str, Any]], library: Iterable[str]) -> dict[str, Any]:
    """Copertura della libreria: brani usati sui disponibili per set e cumulata nei set successivi."""
    lib = sorted(set(library))
    seen: set[str] = set()
    rows = []
    usage: Counter = Counter()
    for k, s in enumerate(sets):
        used = [x for x in s["order"] if x in lib]
        usage.update(used)
        new = sorted(set(used) - seen)
        seen |= set(used)
        rows.append({"set": s.get("name", k + 1), "n_used": len(set(used)), "share": round(len(set(used)) / len(lib), 4) if lib else None,
                     "new_tracks": new, "cumulative_used": len(seen),
                     "cumulative_share": round(len(seen) / len(lib), 4) if lib else None})
    return {"library_size": len(lib), "per_set": rows, "never_used": [x for x in lib if x not in seen],
            "usage_counts": {x: usage.get(x, 0) for x in lib}}


def _entropy(counts: Mapping[str, int]) -> Optional[float]:
    n = sum(counts.values())
    if n == 0 or len(counts) < 2:
        return 0.0 if n else None
    h = -sum((c / n) * math.log(c / n) for c in counts.values() if c)
    return round(h / math.log(len(counts)), 4)


def variety(sets: Sequence[Mapping[str, Any]], techniques_known: Optional[Iterable[str]] = None) -> dict[str, Any]:
    """Varietà dentro e tra i set: tecniche (conteggi, entropia normalizzata), tecnica uguale due volte di fila, coppie e
    transizioni (coppia + tecnica) ripetute rispetto ai set precedenti, Jaccard dei brani e delle coppie tra set consecutivi."""
    rows = []
    prev_pairs: set = set()
    prev_trans: set = set()
    prev_tracks: Optional[set] = None
    prev_pairs_last: Optional[set] = None
    all_tech: Counter = Counter()
    for k, s in enumerate(sets):
        tech = Counter(x["technique"] for x in s["transitions"] if x.get("technique"))
        all_tech.update(tech)
        seq = [x.get("technique") for x in s["transitions"]]
        runs = sum(1 for i in range(1, len(seq)) if seq[i] and seq[i] == seq[i - 1])
        pairs = {tuple(p) for p in s["pairs"]}
        trans = {(x["a"], x["b"], x.get("technique")) for x in s["transitions"]}
        tracks = set(s["order"])
        row = {"set": s.get("name", k + 1), "n_transitions": len(s["transitions"]), "techniques": dict(tech),
               "n_distinct_techniques": len(tech), "technique_entropy": _entropy(tech), "same_technique_twice_in_a_row": runs,
               "pairs_repeated_from_previous_sets": len(pairs & prev_pairs),
               "transitions_repeated_from_previous_sets": len(trans & prev_trans)}
        if prev_tracks is not None:
            row["tracks_jaccard_vs_previous"] = round(len(tracks & prev_tracks) / len(tracks | prev_tracks), 4) if tracks | prev_tracks else None
            row["pairs_jaccard_vs_previous"] = round(len(pairs & prev_pairs_last) / len(pairs | prev_pairs_last), 4) if pairs | prev_pairs_last else None
            row["same_order_as_previous"] = s["order"] == list(sets[k - 1]["order"])
        rows.append(row)
        prev_pairs |= pairs
        prev_trans |= trans
        prev_tracks, prev_pairs_last = tracks, pairs
    known = sorted(set(techniques_known or []) | set(all_tech))
    return {"per_set": rows, "techniques_overall": dict(all_tech), "technique_entropy_overall": _entropy(all_tech),
            "techniques_never_used": [t for t in known if t not in all_tech]}


def compare_plans(base: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    """Che cosa cambia tra due piani (set.json): brani, coppie, cue e tecniche."""
    sa, sb = set_summary(base), set_summary(new)
    pa = {tuple(p) for p in sa["pairs"]}
    pb = {tuple(p) for p in sb["pairs"]}
    ta = {(x["a"], x["b"]): x for x in sa["transitions"]}
    rows = []
    for x in sb["transitions"]:
        y = ta.get((x["a"], x["b"]))
        rows.append({"a": x["a"], "b": x["b"], "in_base": y is not None,
                     "cue": [x["a_out_bar"], x["b_in_bar"]], "base_cue": [y["a_out_bar"], y["b_in_bar"]] if y else None,
                     "technique": x["technique"], "base_technique": y["technique"] if y else None})
    return {"same_order": sa["order"] == sb["order"], "tracks_removed": sorted(set(sa["order"]) - set(sb["order"])),
            "tracks_added": sorted(set(sb["order"]) - set(sa["order"])),
            "pairs_removed": sorted([list(p) for p in pa - pb]), "pairs_added": sorted([list(p) for p in pb - pa]),
            "n_same_pairs": len(pa & pb), "n_same_cue": sum(1 for r in rows if r["in_base"] and r["cue"] == r["base_cue"]),
            "n_same_technique": sum(1 for r in rows if r["in_base"] and r["technique"] == r["base_technique"]), "rows": rows}
