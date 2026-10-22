"""Piano per eventi di una transizione A -> B (P0.5, schema ``musicmix.event_plan.1``).

Specifica: ``conoscenza_mix/_meta/PIANO_EVENTI_SPEC_2026-10-21.md``; implementazione e limiti:
``conoscenza_mix/_meta/PIANO_EVENTI_IMPL_2026-10-21.md``.

Per un candidato di :func:`musicmix.dj_set.candidates_for_pair` (piano di cue + decisione) il modulo
descrive, con campi espliciti e verificabili:

* l'**evento di A** in corso al punto d'uscita (sezione allin1, o etichetta ``dj_form`` se allin1 manca),
  se è completato o saltato, il verso (``dj_phrase.exit_point``) e gli eventi futuri di A non suonati;
* il **lead-in di B** (pickup, filtro, eco, build, break...) e il campione di B in cui B diventa udibile
  (stima prima del render);
* l'**evento obiettivo di B** (ritornello, drop/ingresso della cassa, inizio strofa, prima voce) e il
  **ritardo** tra B udibile (e lo switch) e il suo arrivo;
* le **alternative** della stessa coppia per asse (arrivo, uscita, cue, durata, tecnica), con costo e
  motivo di scarto, più la continuazione di A come solo riferimento;
* il **trace** dei moduli eseguiti con le versioni.

Tutto è descrittivo: nessun campo entra nei costi di default (regole 1–2 della specifica: un evento
saltato può essere la scelta giusta, il ritardo non ha un valore «buono»). L'unico uso nella selezione è
l'opzione **sperimentale e spenta** ``dj_set.CUE_PARAMS['w_event_arrival']`` (:func:`arrival_summary`):
a parità di costo tecnico (entro il peso) preferisce il candidato in cui A completa la sua sezione e
l'evento obiettivo di B arriva entro ``arrival_max_bars`` dallo switch. Etichetta [H]; contesto nei libri:
B&B 2003 cap. 13 'Even tighter' [P] («mix straight after a chorus ... the big opening moment in record 2
happens immediately after the chorus in record one», unità EPUB 33), Butler 2006 p. 246 [A] (il ritorno
della cassa, «dropping the beat», anche «when a new record enters the mix») e p. 247 [P] (Mystik: la
cassa tolta per quattro battiti e riportata «on beat»). Nessuna di queste pagine dà una soglia in battute:
1–2 battute è una nostra ipotesi da verificare con confronti a coppie (P1.1).

Funzioni pure: nessun I/O, nessun modello. Le informazioni di ``dj_phrase`` (uscita/ingresso) arrivano
già calcolate dal chiamante (``exit_info``/``entry_info`` o le funzioni ``exit_fn``/``entry_fn``).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from musicmix import dj_select as djs

EVENT_PLAN_SCHEMA = "musicmix.event_plan.1"
DJ_EVENTS_VERSION = "musicmix.dj_events.1"
SR = 48000  # campioni della sorgente (stessa convenzione di dj_phrase / dj_render)

#: parametri con origine
PARAMS: dict[str, tuple[Any, str]] = {
    "target_horizon_bars": (16, "[H] gli eventi di B si cercano nelle 16 battute dopo il downbeat d'ingresso (una sezione tipica; "
                                "B&B cap. 10: griglia di 8 battute, frasi da 8/16)"),
    "salient_sections": (("chorus", "verse", "bridge"), "[H] inizi di sezione allin1 che contano come evento di B"),
    "target_priority": (("chorus", "drop", "verse", "bridge", "first_vocal", "section_start"),
                        "[H] a parità di battuta vince il primo tipo; il bersaglio è l'evento più vicino all'ingresso"),
    "vocal_bar_active": (djs.VOCAL_BAR_ACTIVE, "[H] dj_select: battuta vocale se coperta >= 50 %"),
    "arrival_max_bars": (2.0, "[H] opzione sperimentale: evento obiettivo di B entro 2 battute dallo switch (0 = sullo switch). "
                              "Contesto [P] B&B cap. 13 'Even tighter'; [A]/[P] Butler pp. 246-247; nessuna soglia nei libri"),
    "max_skipped": (8, "[H] eventi futuri di A elencati (gli altri solo contati)"),
    "alt_arrival": (4, "[H] alternative con la stessa uscita di A e un altro ingresso di B"),
    "alt_exit": (3, "[H] alternative con lo stesso ingresso di B e un'altra uscita di A"),
    "alt_cue": (5, "[H] altre coppie di cue migliori per costo"),
    "max_table": (400, "[H] righe della tabella di tutti i candidati (= max_per_transition di dj_set.choose_cues)"),
}

LEAD_IN_KINDS = ("none", "pickup", "fill", "build", "filter_sweep", "echo_tail", "loop", "break", "blend", "backspin")
#: tecniche creative di dj_policy/dj_creative -> tipo di lead-in
_CREATIVE_LEAD = {"build_into_chorus": "build", "halving_build": "build", "break_before_entry": "break",
                  "loop_extend": "loop", "acapella_over": "blend", "vocal_swap": "blend", "backspin": "backspin"}


def _p(name: str, overrides: Optional[Mapping[str, Any]] = None) -> Any:
    if overrides and name in overrides:
        return overrides[name]
    return PARAMS[name][0]


def _sample(t_s: float) -> int:
    return int(round(float(t_s) * SR))


def _bar_s(track: Mapping[str, Any]) -> float:
    return djs.BEATS_PER_BAR * float(track["tempo"]["period_s"])


def _bar_time(track: Mapping[str, Any], bar: float) -> float:
    """Tempo (s) della battuta ``bar`` (frazionaria) sulla griglia di dj_select; estrapola oltre la tabella."""
    rows = track["phrases"]
    n = len(rows)
    if n == 0:
        return float(bar) * _bar_s(track)
    k = int(max(0, min(n - 1, int(bar))))
    return float(rows[k]["time_s"]) + (float(bar) - k) * _bar_s(track)


def section_spans(track: Mapping[str, Any]) -> list[tuple[int, int, str]]:
    from musicmix import dj_set as ds  # lazy: dj_set imports this module lazily too

    return ds.section_spans(track)


def _span_at(spans: Sequence[tuple[int, int, str]], bar: int) -> Optional[tuple[int, int, str]]:
    return next((s for s in spans if s[0] <= bar < s[1]), None)


def _form_label(track: Mapping[str, Any], bar: int) -> Optional[str]:
    labels = (track.get("form") or {}).get("labels") or []
    return labels[bar] if 0 <= bar < len(labels) else None


# ------------------------------------------------------------------ A side
def a_event(A: Mapping[str, Any], a_out_bar: int, a_out_s: float, *, exit_info: Optional[Mapping[str, Any]] = None,
            tail_bars: int = 0, params: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Evento di A in corso al punto d'uscita (battuta ``a_out_bar - 1``) e che cosa ne resta.

    ``fate``: ``cut_mid_line`` se ``dj_phrase.exit_point`` dice che un verso attraversa il downbeat (priorità:
    il verso troncato si sente anche a confine di sezione), altrimenti ``completed`` se l'uscita cade su un
    confine di sezione allin1, ``skipped_rest`` se cade dentro una sezione, ``unknown`` senza sezioni.
    ``tail_bars`` (blend): battute in cui A continua sotto B dopo l'uscita.
    """
    ja = int(a_out_bar)
    rows = A["phrases"]
    spans = section_spans(A)
    span = _span_at(spans, ja - 1) if ja > 0 else None
    kind, src = "unknown", None
    if ja <= 0:
        kind, src = "start", "dj_select"
    elif span is not None and span[2] not in ("unlabeled", None):
        kind, src = span[2], "allin1"
    else:
        fl = _form_label(A, ja - 1)
        if fl is not None:
            kind, src = fl, "dj_form"  # 'unknown' when dj_form has no core (dj_form.2)
    at_boundary = ja >= len(rows) or bool(rows[ja]["section_start"]) if rows else False
    ex = dict(exit_info) if exit_info else None
    line_status = ex.get("status") if ex else None
    if line_status == "mid_line":
        fate = "cut_mid_line"
    elif span is None and not at_boundary:
        fate = "unknown"
    elif at_boundary:
        fate = "completed"
    else:
        fate = "skipped_rest"
    future = []
    if span is not None and not at_boundary:
        future.append({"kind": span[2], "span_bars": [ja, span[1]], "part": "rest"})
    future += [{"kind": l, "span_bars": [a, b]} for a, b, l in spans if a >= ja]
    counts: dict[str, int] = {}
    for f in future:
        if f.get("part") != "rest":
            counts[f["kind"]] = counts.get(f["kind"], 0) + 1
    if ex:
        line_end = {"status": line_status, "p_line_end": ex.get("p_line_end"), "last_word": ex.get("last_word"),
                    "next_word": ex.get("next_word"), "tail_end_sample": ex.get("tail_end_sample"),
                    "a_pickup_sample": ex.get("a_pickup_sample"), "closure_diag": ex.get("krims_closure"),
                    "dry_cut_truncates": list(ex.get("dry_cut_truncates") or [])[:6],
                    "reasons": list(ex.get("reasons") or [])[:3], "source": "dj_phrase.exit_point",
                    "note": "closure_diag (Krims pp. 133, 194) è solo diagnostica: non entra nel costo"}
    else:
        line_end = {"status": "unavailable", "source": None}
    fa = A.get("form")
    if fa and fa.get("labels"):
        lab = fa["labels"]
        form_at_exit = {"before": lab[ja - 1] if 0 < ja <= len(lab) else None, "at": lab[ja] if 0 <= ja < len(lab) else None,
                        "core_status": fa.get("core_status", "found"), "status": "measured", "source": "dj_form"}
    else:
        form_at_exit = {"status": "unavailable", "source": None}
    mx = int(_p("max_skipped", params))
    return {"kind": kind, "label_source": src, "span_bars": [span[0], span[1]] if span else None,
            "exit_bar": ja, "exit_sample": _sample(a_out_s), "on_section_boundary": bool(at_boundary), "fate": fate,
            "skipped": future[:mx], "skipped_counts": counts, "skipped_total": len(future),
            "continues_under_b_bars": int(tail_bars), "line_end": line_end, "form_at_exit": form_at_exit}


# ------------------------------------------------------------------ B side
def b_events(B: Mapping[str, Any], b_in_bar: int, b_in_s: float, *, entry_info: Optional[Mapping[str, Any]] = None,
             params: Optional[Mapping[str, Any]] = None) -> list[dict[str, Any]]:
    """Eventi di B dal downbeat d'ingresso fino a ``target_horizon_bars`` dopo (battute di B, campioni di B).

    Tipi: inizi di sezione allin1 (``chorus``/``verse``/``bridge``, gli altri come ``section_start``), ``drop``
    (``dj_form``: ritorno della cassa o suo primo ingresso), ``first_vocal`` (prima parola di un verso da
    ``dj_phrase.entry_point`` se c'è, altrimenti inizio di una regione vocale delle battute di dj_select).
    Eventi sulla stessa battuta sono fusi (tipo per priorità, evidenze unite).
    """
    jb = int(b_in_bar)
    H = int(_p("target_horizon_bars", params))
    sal = tuple(_p("salient_sections", params))
    rows = B["phrases"]
    ev: list[dict[str, Any]] = []
    for a, _b, lab in section_spans(B):
        if jb <= a <= jb + H and a < len(rows) and rows[a]["section_start"]:
            ev.append({"kind": lab if lab in sal else "section_start", "bar": float(a), "evidence": [f"allin1:{lab}@{a}"],
                       "status": "measured"})
    fb = B.get("form") or {}
    drops = set(int(x) for x in fb.get("drops", []) or [])
    kick = ((fb.get("active") or {}).get("kick")) or []
    for k in sorted(drops):
        if jb <= k <= jb + H:
            ev.append({"kind": "drop", "bar": float(k), "evidence": [f"dj_form:drop@{k}"], "status": "measured"})
    if kick:
        first_kick = next((i for i, x in enumerate(kick) if x), None)
        if first_kick is not None and jb <= first_kick <= jb + H and first_kick not in drops and first_kick > 0:
            ev.append({"kind": "drop", "bar": float(first_kick), "evidence": [f"dj_form:kick_in@{first_kick}"], "status": "measured"})
    # first vocal
    thr = float(_p("vocal_bar_active", params))
    voc = [float(x) for x in (B.get("bars") or {}).get("vocal", [])]
    bar_s = _bar_s(B)
    st = (entry_info or {}).get("status")
    got = False
    if entry_info and st == "pickup":
        ps = int(entry_info.get("pickup_samples") or 0)
        ev.append({"kind": "first_vocal", "bar": float(jb) - ps / (SR * bar_s), "evidence": [
            "dj_phrase:pickup «" + " ".join(entry_info.get("pickup_words") or []) + "»"], "status": "measured"})
        got = True
    elif entry_info and st == "on_or_after_downbeat" and entry_info.get("first_word_rel") is not None:
        rel = int(entry_info["first_word_rel"])
        ev.append({"kind": "first_vocal", "bar": float(jb) + rel / (SR * bar_s),
                   "evidence": [f"dj_phrase:first_word «{entry_info.get('first_word')}»"], "status": "measured"})
        got = True
    if not got and voc:
        for k in range(jb, min(len(voc), jb + H + 1)):
            if voc[k] >= thr and (k == 0 or voc[k - 1] < thr):
                ev.append({"kind": "first_vocal", "bar": float(k), "evidence": [f"dj_select:vocal_onset@{k}"], "status": "estimated"})
                break
    # merge events on the same bar (rounded to 1/16 of a bar)
    prio = list(_p("target_priority", params))
    merged: dict[float, dict[str, Any]] = {}
    for e in ev:
        key = round(e["bar"] * 16) / 16
        if key in merged:
            m = merged[key]
            m["evidence"] += e["evidence"]
            if prio.index(e["kind"]) < prio.index(m["kind"]):
                m["kind"] = e["kind"]
            if e["status"] == "measured":
                m["status"] = "measured"
        else:
            merged[key] = dict(e, evidence=list(e["evidence"]))
    out = []
    for key in sorted(merged):
        e = merged[key]
        t = _bar_time(B, e["bar"])
        out.append({**e, "bar": round(e["bar"], 4), "sample": _sample(t), "bars_after_downbeat": round(e["bar"] - jb, 4)})
    return out


def pick_target(events: Sequence[Mapping[str, Any]], params: Optional[Mapping[str, Any]] = None,
                min_bar: Optional[float] = None) -> dict[str, Any]:
    """Evento obiettivo di B: il primo evento saliente (ritornello, drop, strofa, bridge, prima voce) dopo
    l'ingresso; solo se non ce n'è nessuno, il primo inizio di sezione; altrimenti ``unavailable``.

    ``min_bar``: battuta (frazionaria) in cui B diventa udibile; gli eventi prima non si sentono (es. il pickup di B
    con un'uscita a effetto che fa entrare B sul downbeat) e non possono essere il bersaglio."""
    prio = list(_p("target_priority", params))
    if min_bar is not None:
        events = [e for e in events if e["bar"] >= float(min_bar) - 1e-6]
    salient = [e for e in events if e["kind"] != "section_start"]
    pool = salient or list(events)
    if not pool:
        return {"kind": "unknown", "bar": None, "sample": None, "evidence": [], "status": "unavailable"}
    e = min(pool, key=lambda x: (x["bar"], prio.index(x["kind"])))
    return {"kind": e["kind"], "bar": e["bar"], "sample": e["sample"], "evidence": list(e["evidence"]), "status": e["status"]}


def _decision_kind(decision: Mapping[str, Any]) -> str:
    return str(decision.get("type"))


def lead_in(decision: Mapping[str, Any], plan: Mapping[str, Any], A: Mapping[str, Any], B: Mapping[str, Any],
            entry_info: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Tipo e durata del lead-in di B (in battute di A per gli effetti su A, di B per pickup e blend)."""
    t = _decision_kind(decision)
    r = dict(decision.get("render") or {})
    pr = dict(r.get("params") or {})
    bar_a, bar_b = _bar_s(A), _bar_s(B)
    per_a = float(A["tempo"]["period_s"])
    out: dict[str, Any] = {"kind": "none", "bars": 0.0, "pickup_words": [], "pickup_samples": 0, "side": None}
    if t == "cut":
        ps = plan.get("phrase_switch") or {}
        if ps and float(ps.get("pickup_s") or 0) > 0:
            out.update(kind="pickup", bars=float(ps["pickup_s"]) / bar_b, pickup_words=list(ps.get("pickup_words") or []),
                       pickup_samples=_sample(ps["pickup_s"]), side="b")
    elif t == "blend":
        out.update(kind="blend", bars=float(decision.get("bars") or 0), side="both")
    elif t in ("filter_out", "filter_into_chorus"):
        out.update(kind="filter_sweep", bars=float(r.get("sweep_bars", 0)), side="a")
    elif t == "echo_out":
        out.update(kind="echo_tail", bars=float(r.get("entry_bars", 0)), side="a")
    elif t == "backspin":
        out.update(kind="backspin", bars=float(r.get("spin_beats", 2)) * per_a / bar_a, side="a")
    elif t in _CREATIVE_LEAD:
        kind = _CREATIVE_LEAD[t]
        if t in ("build_into_chorus", "halving_build"):
            bars = float(pr.get("build_bars", 2 if t == "build_into_chorus" else 4))
        elif t == "break_before_entry":
            bars = float(pr.get("beats", 1)) / djs.BEATS_PER_BAR
        else:
            bars = float(pr.get("bars", decision.get("bars") or 0))
        out.update(kind=kind, bars=bars, side="a" if kind in ("build", "break", "backspin") else "both")
        if pr.get("gap_beats"):
            out["gap_beats"] = float(pr["gap_beats"])
    out["bars"] = round(float(out["bars"]), 4)
    if entry_info is not None:
        # B's own pickup (dj_phrase.entry_point): heard only when the render plays it (phrase cut)
        pw = list(entry_info.get("pickup_words") or [])
        out["b_line_status"] = entry_info.get("status")
        if pw and out["kind"] != "pickup":
            out["b_pickup_not_played"] = pw
    return out


def _audible_bar(decision: Mapping[str, Any], plan: Mapping[str, Any], A: Mapping[str, Any], B: Mapping[str, Any], jb: int) -> float:
    li = lead_in(decision, plan, A, B)
    return float(jb) - (float(li["bars"]) if li["kind"] == "pickup" else 0.0)


def _switch_bar_b(decision: Mapping[str, Any], b_in_bar: int) -> float:
    """Battuta di B in cui B resta solo: fine del blend, altrimenti il downbeat d'ingresso."""
    return float(b_in_bar) + (float(decision.get("bars") or 0) if _decision_kind(decision) == "blend" else 0.0)


def arrival_summary(A: Mapping[str, Any], B: Mapping[str, Any], cand: Mapping[str, Any], *,
                    decision: Optional[Mapping[str, Any]] = None, exit_info: Optional[Mapping[str, Any]] = None,
                    entry_info: Optional[Mapping[str, Any]] = None, params: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Riassunto leggero (per l'opzione sperimentale e per le alternative): destino dell'evento di A, evento
    obiettivo di B, ritardo dallo switch e se il candidato è «preferito» dall'opzione [H]."""
    plan = cand["plan"]
    dec = decision or cand["decision"]
    ja, jb = int(plan["a_out_bar"]), int(plan["b_in_bar"])
    ae = a_event(A, ja, plan["a_out_s"], exit_info=exit_info, tail_bars=int(dec.get("bars") or 0) if dec.get("type") == "blend" else 0,
                 params=params)
    tgt = pick_target(b_events(B, jb, plan["b_in_s"], entry_info=entry_info, params=params), params,
                      min_bar=_audible_bar(dec, plan, A, B, jb))
    sw = _switch_bar_b(dec, jb)
    fs = None if tgt["bar"] is None else round(float(tgt["bar"]) - sw, 4)
    mx = float(_p("arrival_max_bars", params))
    pref = bool(ae["fate"] == "completed" and fs is not None and 0.0 <= fs <= mx)
    return {"a_kind": ae["kind"], "a_fate": ae["fate"], "a_line": ae["line_end"]["status"], "b_target": tgt["kind"],
            "b_target_bar": tgt["bar"], "delay_bars": None if tgt["bar"] is None else round(float(tgt["bar"]) - jb, 4),
            "from_switch_bars": fs, "preferred": pref}


def _alt_row(A, B, c, chosen_cost: float, vary: str, exit_fn, entry_fn, params) -> dict[str, Any]:
    pl = c["plan"]
    s = arrival_summary(A, B, c, exit_info=exit_fn(pl["a_out_s"]) if exit_fn else None,
                        entry_info=entry_fn(pl["b_in_s"]) if entry_fn else None, params=params)
    d = float(c["cost"]) - float(chosen_cost)
    if not c.get("admissible", True):
        why = "non ammissibile: " + "; ".join(c.get("notes") or ["filtri del runner"])
    elif d > 1e-9:
        why = f"costo +{d:.3f} rispetto alla scelta"
    elif d < -1e-9:
        why = (f"costo minore ({d:+.3f}) ma scartato dalla scelta lungo il set (durata del brano, spazio con le transizioni "
               "vicine o sezioni intere: dj_set.choose_cues)")
    else:
        why = "stesso costo: vince l'ordine di ranking (punteggio del cue, finestre di default, cue più presto)"
    return {"vary": vary, "a_out_bar": int(pl["a_out_bar"]), "b_in_bar": int(pl["b_in_bar"]),
            "technique": c["decision"].get("type_key", c["decision"].get("type")), "rule": c["decision"].get("rule"),
            "cost": round(float(c["cost"]), 4), "delta_cost": round(d, 4), "admissible": bool(c.get("admissible", True)),
            "a_fate": s["a_fate"], "a_kind": s["a_kind"], "a_line": s["a_line"], "b_target": s["b_target"],
            "delay_bars": s["delay_bars"], "from_switch_bars": s["from_switch_bars"],
            "preferred_by_event_option": s["preferred"], "why_not": why}


def enumerate_alternatives(A: Mapping[str, Any], B: Mapping[str, Any], cand: Mapping[str, Any], decision: Mapping[str, Any], *,
                           pair_candidates: Optional[Sequence[Mapping[str, Any]]] = None,
                           policy: Optional[Sequence[Mapping[str, Any]]] = None,
                           order_params: Optional[Mapping[str, Any]] = None,
                           exit_fn: Optional[Callable[[float], Mapping[str, Any]]] = None,
                           entry_fn: Optional[Callable[[float], Mapping[str, Any]]] = None,
                           params: Optional[Mapping[str, Any]] = None) -> tuple[list[dict[str, Any]], list[list[Any]]]:
    """Alternative della stessa coppia per asse (regola 4 della specifica) e tabella compatta di tutti i candidati."""
    from musicmix import dj_set as ds

    plan = cand["plan"]
    ja, jb = int(plan["a_out_bar"]), int(plan["b_in_bar"])
    cost0 = float(cand["cost"])
    alts: list[dict[str, Any]] = []
    table: list[list[Any]] = []
    if pair_candidates:
        others = [c for c in pair_candidates if not (int(c["plan"]["a_out_bar"]) == ja and int(c["plan"]["b_in_bar"]) == jb)]
        seen: set[tuple[int, int]] = set()

        def take(rows, n, vary):
            k = 0
            for c in rows:
                key = (int(c["plan"]["a_out_bar"]), int(c["plan"]["b_in_bar"]))
                if key in seen:
                    continue
                seen.add(key)
                alts.append(_alt_row(A, B, c, cost0, vary, exit_fn, entry_fn, params))
                k += 1
                if k >= n:
                    break

        take([c for c in others if int(c["plan"]["a_out_bar"]) == ja], int(_p("alt_arrival", params)), "arrival")
        take([c for c in others if int(c["plan"]["b_in_bar"]) == jb], int(_p("alt_exit", params)), "exit")
        take(others, int(_p("alt_cue", params)), "cue")
        for c in list(pair_candidates)[: int(_p("max_table", params))]:
            pl = c["plan"]
            s = arrival_summary(A, B, c, exit_info=exit_fn(pl["a_out_s"]) if exit_fn else None,
                                entry_info=entry_fn(pl["b_in_s"]) if entry_fn else None, params=params)
            table.append([int(pl["a_out_bar"]), int(pl["b_in_bar"]), c["decision"].get("type_key"), round(float(c["cost"]), 4),
                          bool(c.get("admissible", True)), s["a_fate"], s["b_target"], s["delay_bars"], s["from_switch_bars"],
                          s["preferred"]])
    # duration / technique on the same cue (tree fallbacks, the tree decision itself when the chosen one is creative)
    tree = cand["decision"]
    same_cue = []
    if decision.get("creative") or decision.get("type_key") != tree.get("type_key"):
        same_cue.append((tree, "decisione dell'albero sullo stesso cue: sostituita dalla tecnica scelta"
                         + (f" ({decision.get('rule')})" if decision.get("rule") else "")))
    for f in tree.get("fallbacks") or []:
        same_cue.append((f, f"alternativa dell'albero per il QA (stesso cue): l'albero applica {tree.get('rule')} prima"))
    has_duration = False
    for d, why in same_cue:
        if d.get("type_key") == decision.get("type_key"):
            continue
        vary = "duration" if (d.get("type") == decision.get("type") == "blend") else "technique"
        has_duration |= vary == "duration"
        try:
            cost = ds.candidate_cost(plan, d, order_params, cand.get("context"))
        except Exception:  # pragma: no cover - defensive: creative decisions have no tree cost
            cost = None
        alts.append({"vary": vary, "technique": d.get("type_key", d.get("type")), "bars": d.get("bars"), "rule": d.get("rule"),
                     "cost": None if cost is None else round(float(cost), 4),
                     "delta_cost": None if cost is None else round(float(cost) - cost0, 4), "why_not": why})
    if not has_duration:
        alts.append({"vary": "duration", "status": "not_applicable",
                     "note": "nessuna durata alternativa enumerata per questa tecnica (l'albero varia solo i blend; la policy "
                             "propone una durata per tecnica)"})
    for p in policy or []:
        if p.get("technique") == decision.get("type_key") and p.get("rule") == decision.get("rule"):
            continue
        alts.append({"vary": "technique", "technique": p.get("technique"), "rule": p.get("rule"), "score": p.get("score"),
                     "duration_bars": p.get("duration_bars"), "cost": None, "source": "dj_policy.propose",
                     "why_not": p.get("why_not") or "proposta della policy non scelta (nessun costo nel Viterbi: ordine per punteggio)"})
    alts.append({"vary": "continue_A", "note": "solo riferimento di confronto («meglio il cambio o restare?»), mai clip da votare "
                                               "1-10 (AGENTS.md)"})
    return alts, table


TABLE_COLUMNS = ("a_out_bar", "b_in_bar", "technique", "cost", "admissible", "a_fate", "b_target", "delay_bars",
                 "from_switch_bars", "preferred_by_event_option")


def module_versions(extra: Optional[Mapping[str, Optional[str]]] = None) -> dict[str, Optional[str]]:
    from musicmix import dj_form, dj_set

    v = {"dj_events": DJ_EVENTS_VERSION, "dj_select": djs.SCHEMA_VERSION, "dj_set": dj_set.SCHEMA_VERSION,
         "dj_form": dj_form.FORM_VERSION}
    v.update(dict(extra or {}))
    return v


def event_plan(A: Mapping[str, Any], B: Mapping[str, Any], cand: Mapping[str, Any], *,
               decision: Optional[Mapping[str, Any]] = None, exit_info: Optional[Mapping[str, Any]] = None,
               entry_info: Optional[Mapping[str, Any]] = None,
               pair_candidates: Optional[Sequence[Mapping[str, Any]]] = None,
               policy: Optional[Sequence[Mapping[str, Any]]] = None, policy_rule: Optional[str] = None,
               order_params: Optional[Mapping[str, Any]] = None,
               exit_fn: Optional[Callable[[float], Mapping[str, Any]]] = None,
               entry_fn: Optional[Callable[[float], Mapping[str, Any]]] = None,
               executed: Iterable[str] = (), versions: Optional[Mapping[str, Optional[str]]] = None,
               cue_windows: str = "default", profile: Optional[str] = None,
               sources: Optional[Mapping[str, Any]] = None, params: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Documento ``musicmix.event_plan.1`` per un candidato (vedi la specifica). Non cambia costi né scelte.

    ``cand``: candidato di :func:`musicmix.dj_set.candidates_for_pair` (``plan``, ``context``, ``decision``, ``cost``);
    ``decision``: la tecnica scelta (default la decisione dell'albero; può essere una tecnica creativa della policy);
    ``exit_info``/``entry_info``: :func:`musicmix.dj_phrase.exit_point`/``entry_point`` al cue (o ``exit_fn``/``entry_fn``
    sul tempo in secondi); ``pair_candidates``: tutti i candidati della coppia per le alternative.
    """
    plan = cand["plan"]
    dec = dict(decision or cand["decision"])
    ja, jb = int(plan["a_out_bar"]), int(plan["b_in_bar"])
    if exit_info is None and exit_fn is not None:
        exit_info = exit_fn(plan["a_out_s"])
    if entry_info is None and entry_fn is not None:
        entry_info = entry_fn(plan["b_in_s"])
    pair = (cand.get("context") or {})
    blend_bars = int(dec.get("bars") or 0) if dec.get("type") == "blend" else 0
    ae = a_event(A, ja, plan["a_out_s"], exit_info=exit_info, tail_bars=blend_bars, params=params)
    li = lead_in(dec, plan, A, B, entry_info)
    b_in_sample = _sample(plan["b_in_s"])
    audible = b_in_sample - int(li.get("pickup_samples") or 0)
    events = b_events(B, jb, plan["b_in_s"], entry_info=entry_info, params=params)
    min_bar = float(jb) - (float(li["bars"]) if li["kind"] == "pickup" else 0.0)
    for e in events:
        e["audible"] = bool(e["bar"] >= min_bar - 1e-6)  # B's pickup before an effect exit is not heard
    tgt = pick_target(events, params, min_bar=min_bar)
    kick = ((B.get("form") or {}).get("active") or {}).get("kick") or []
    b_entry = {"audible_sample": audible, "audible_status": "estimated",
               "audible_note": "stima prima del render (campione sorgente di B); la verifica sul time-map renderizzato è del QA (P0.6)",
               "lead_in": li, "downbeat_bar": jb, "downbeat_sample": b_in_sample,
               "kick_at_entry": (bool(kick[jb]) if 0 <= jb < len(kick) else None),
               "section_at_entry": pair.get("b_section"), "section_start": pair.get("b_section_start")}
    bar_b = _bar_s(B)
    sw = _switch_bar_b(dec, jb)
    a_until = _sample(plan["a_out_s"] + blend_bars * _bar_s(A))
    delay: dict[str, Any] = {"status": "unavailable" if tgt["sample"] is None else tgt["status"],
                             "a_audible_until_sample": a_until,
                             "a_audible_until_status": "estimated" + (" (coda d'eco esclusa: durata dal render)" if dec.get("type") == "echo_out" else ""),
                             "switch_bar_b": sw,
                             "note": "descrizione, non una soglia (specifica, regola 2): il ritardo di Huron pp. 314-316 non ha un valore 'buono'"}
    if tgt["sample"] is not None:
        delay.update(bars=round((tgt["sample"] - audible) / (SR * bar_b), 4), s=round((tgt["sample"] - audible) / SR, 4),
                     from_downbeat_bars=round(float(tgt["bar"]) - jb, 4), from_switch_bars=round(float(tgt["bar"]) - sw, 4))
    compat = {}
    if "stretch_abs" in pair:
        compat = {"a_period_s": float(A["tempo"]["period_s"]), "b_period_s": float(B["tempo"]["period_s"]),
                  "stretch_b_pct": round(100.0 * float(pair["stretch_abs"]), 3), "octave_factor": pair.get("octave_factor", 1.0)}
    choice = {"technique": dec.get("type"), "type_key": dec.get("type_key", dec.get("type")), "label": dec.get("label"),
              "bars": dec.get("bars"), "render": dec.get("render"),
              "tree_rule": cand["decision"].get("rule"), "policy_rule": policy_rule if dec.get("creative") else None,
              "cost": round(float(cand["cost"]), 6), "creative": bool(dec.get("creative"))}
    alts, table = enumerate_alternatives(A, B, cand, dec, pair_candidates=pair_candidates, policy=policy,
                                         order_params=order_params, exit_fn=exit_fn, entry_fn=entry_fn, params=params)
    pref = arrival_summary(A, B, cand, decision=dec, exit_info=exit_info, entry_info=entry_info, params=params)
    ex = ["dj_select.plan_pair", "dj_set.transition_context", "dj_set.decide_transition", "dj_events.event_plan"]
    if exit_info is not None:
        ex.append("dj_phrase.exit_point")
    if entry_info is not None:
        ex.append("dj_phrase.entry_point")
    if A.get("form") and B.get("form"):
        ex.append("dj_form.overlap_profile")
    if policy is not None:
        ex.append("dj_policy.propose")
    for e in executed:
        if e not in ex:
            ex.append(e)
    return {
        "schema": EVENT_PLAN_SCHEMA, "a": A.get("slug"), "b": B.get("slug"), "grid": compat,
        "a_event": ae, "b_entry": b_entry, "b_target": tgt, "b_events": events, "delay": delay,
        "choice": choice, "event_option": {**pref, "arrival_max_bars": float(_p("arrival_max_bars", params)),
                                           "label": "[H] sperimentale: A completato e obiettivo di B entro arrival_max_bars dallo switch"},
        "alternatives": alts,
        "candidates_table": {"columns": list(TABLE_COLUMNS), "rows": table},
        "trace": {"modules": module_versions(versions), "executed": ex, "cue_windows": cue_windows, "profile": profile,
                  "sources": dict(sources or {}), "params": {k: _p(k, params) for k in PARAMS}},
    }


__all__ = ["EVENT_PLAN_SCHEMA", "DJ_EVENTS_VERSION", "PARAMS", "LEAD_IN_KINDS", "TABLE_COLUMNS", "a_event", "b_events",
           "pick_target", "lead_in", "arrival_summary", "enumerate_alternatives", "event_plan", "module_versions"]
