"""Policy delle tecniche creative: dato il contesto di una transizione, 2–4 proposte motivate.

Estende l'albero di :mod:`musicmix.dj_set` (che sceglie fra cut, blend, echo out,
filter out, filter verso il ritornello) **senza modificarlo**: qui si propongono le
tecniche di :mod:`musicmix.dj_creative` adatte allo stesso contesto, ciascuna con
parametri di render, durata in battute, motivazione e fonte, così che un critico
automatico (``dj_critic``) o l'ascolto scelgano fra render alternativi dello stesso
punto di cambio. ``dj_set`` potrà importare :func:`propose` dopo.

Ingresso: il contesto di :func:`musicmix.dj_set.transition_context` più alcune
chiavi facoltative (:func:`creative_context` le calcola dalle analisi di
``dj_select``):

* ``key_a``, ``key_b`` (nomi, es. "B minor"), ``key_shift`` (da
  :func:`musicmix.dj_creative.auto_key_shift`, None se nessuna trasposizione
  di ±1 rende la coppia compatibile);
* ``has_stems`` (stem vocali Demucs di A e B disponibili);
* ``lyric_links`` (da :func:`musicmix.dj_creative.find_lyric_links` per questi cue);
* ``recent_techniques`` (tecniche usate nelle ultime transizioni del set).

Regole: tabella :data:`POLICY_TABLE` (contesto → tecniche, durata, fonte). Le
fonti sono pratica DJ [P] (manuali e documentazione ufficiale dei software, vedi
``conoscenza_mix/_meta/TECNICHE_CREATIVE_2026-10-21.md``), regole di sistemi
pubblicati [S], risultati empirici [E] e i sei giudizi dell'utente [U]
(galleria v1, usati come priorità dichiarate, non come validazione); tutto il
resto è ipotesi [H]. Il ``prior`` serve solo a ordinare le proposte: **non è
una stima di qualità** e non va usato come voto.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

POLICY_VERSION = "musicmix.dj_policy.1"  # grid_guard (21/10/2026, come R1_grid_low); profilo libri (2026-10-21) opzionale: PROFILE_BOOKS, default invariati; + halving_build, break_before_entry

# Parametri a priori, con origine.
PARAMS: dict[str, tuple[Any, str]] = {
    "max_stretch": (0.08, "[S] MosaikBox ±8% (beatmatch); [E] Ishizaki 2009 disagio oltre ±10%"),
    "long_stretch": (0.05, "[E] Kim 2020 §6.1: 86,1% degli stretch entro 5%"),
    "vocal_bar": (0.5, "[H] dj_select.VOCAL_BAR_ACTIVE"),
    "instrumental_share": (0.25, "[H] dj_set: parte strumentale se quota vocale <= 0,25"),
    "exit_vocal_max": (0.1, "[H] dj_select.CUT_GUARD_MAX_COVER (ultimo battito di A libero)"),
    "energy_drop_db": (-6.0, "[H] stessa soglia di dj_set R5"),
    "energy_rise_db": (3.0, "[H] B sensibilmente più energico di A"),
    "lyric_min_score": (0.55, "[H] legame lessicale/rima con ASR affidabile (match x sqrt(conf))"),
    "chorus_labels": (("chorus",), "[H] etichette allin1 considerate ritornello/hook"),
    "repeat_penalty": (0.15, "[T] Williams 2025 (cita Hawkins 2013, debole): varietà delle tecniche"),
    # ---- profilo libri (2026-10-21; default neutri, vedi PROFILE_BOOKS e conoscenza_mix/libri/)
    "fx_marked": (("echo_freeze", "reverb_wash", "backspin", "tape_stop", "halving_build", "break_before_entry"),
                  "[P] Brewster & Broughton 2003 (B&B) cap. 29 'How to add FX': effetti marcati da usare con moderazione"),
    "fx_budget_penalty": (0.0, "[P] B&B cap. 29: 'Effects should be your secret weapon, not your constant companion'. Penalità a "
                               "ogni effetto marcato se un effetto marcato è già stato usato nelle ultime transizioni (oltre a "
                               "repeat_penalty per la stessa tecnica); 0 = spento"),
    "backspin_context_bonus": (0.0, "[P] B&B cap. 12: spinback 'at the end of a phrase in record 1 and just before an exciting start in "
                                    "record 2'; 'an escape hatch if you can't get two records to mix properly'. Bonus al backspin "
                                    "quando A esce a confine di sezione e B entra a inizio sezione non più calmo di A; 0 = spento"),
    "backspin_min_energy_db": (-1.0, "[H] 'exciting start' di B: energia di B >= energia di A - 1 dB"),
    "drop_entry_build": (False, "[A] Butler 2006 pp. 246-248 ('dropping the beat': effetti che salgono, poi 'you turn it off and drop "
                                "the beat ... and it's clean'; DJ Shiva, Mystik) e pp. 91-92 (withholding the beat) [pagine corrette 21/10/2026]. Se vero, la preparazione "
                                "build_into_chorus si propone anche quando B entra sul ritorno della propria cassa (drop calcolato da "
                                "musicmix.dj_form), non solo sul ritornello"),
    "tape_stop_punchy": (False, "[P] B&B cap. 12 'power-off wind-down': 'have a really punchy tune waiting in the wings, and as the "
                                "outgoing record dies, slam it in'. Se vero, con tempo lontano il tape stop si propone anche "
                                "quando B è più energico di A (oltre al caso [H] B più calmo)"),
    "halving_build": (False, "[A] Margulis 2014 p. 23 e p. 44 (unità ripetuta che si restringe = accelerazione percepita a tempo "
                             "costante); [T] Huron 2006 pp. 323-326 (accelerazione degli eventi come indicatore di climax), p. 319 "
                             "(grandi aumenti di arousal vanno preparati). Se vero, nel contesto C2 si propone anche "
                             "dj_creative.halving_build (4 battute; 8 se B è molto più energico e A ne ha abbastanza)"),
    "entry_break": (False, "[A] Huron 2006 pp. 248-249 (il fill rende certo il 'quando' del downbeat ipermetrico) e pp. 314-316 "
                           "(ritardare un evento prevedibile aumenta la tensione; p. 368: non troppo a lungo); [P] Katz 2012 p. 106 "
                           "(drop out sulla punchline). Se vero, quando A esce a confine di sezione e B entra a inizio sezione si "
                           "propone dj_creative.break_before_entry: beat_mute se A canta sull'ultimo battito (stem), altrimenti "
                           "fill di 1 battito, stop di 1 battito con tempo lontano"),
    "grid_guard": (True, "[H] come R1_grid_low dell'albero di dj_set: con la griglia di A o B a confidenza bassa niente tecniche "
                         "che dipendono dalla griglia (loop_extend, halving_build, stutter e pausa del build, break_before_entry, "
                         "acapella_over/vocal_swap con beatmatch). Caso: T13 rich_flex -> God's Plan, QA -316 ms (21/10/2026)"),
    "grid_low_levels": (("low",), "[H] stessi livelli di dj_set PARAMS grid_low_levels (tempo.confidence di dj_select)"),
    "build_gap_beats": (0, "[T] Huron 2006 pp. 314-316 (stop prima del drop), p. 368 (ritardo troppo lungo = tendenza persa); "
                           "[H] battiti di pausa prima di B nei build (0 = come prima; profilo libri 1, mai oltre 2)"),
}

#: Profilo "libri": sostituzioni dei PARAMS (fonti nei PARAMS, livelli di evidenza verificati in
#: PROFILE_BOOKS_EVIDENCE). Valori [H].
PROFILE_BOOKS: dict[str, Any] = {
    "drop_entry_build": True,
    "fx_budget_penalty": 0.10,
    "backspin_context_bonus": 0.20,
    "tape_stop_punchy": True,
    "halving_build": True,
    "entry_break": True,
    "build_gap_beats": 1,
}

#: Livello di evidenza delle voci di PROFILE_BOOKS (audit dei nove libri, 21/10/2026). ``source_level`` = che cosa offre
#: la fonte ([E] sperimentale, [T] teoria, [A] analisi, [P] pratica, [H] nostra ipotesi); ``rule_level`` = stato del
#: valore usato (pesi, bonus e durate sono [H]: nessuna voce è validata su ascolti). ``verified`` = pagine STAMPATE
#: rilette nel testo per pagina (tools/libri.py; B&B è un EPUB: capitolo).
PROFILE_BOOKS_EVIDENCE: dict[str, dict[str, str]] = {
    "drop_entry_build": {"source_level": "A+P", "rule_level": "H",
                         "verified": "Butler 2006 pp. 246-248 ('dropping the beat', DJ Shiva/Mystik), pp. 91-92 (withholding)",
                         "note": "il 'drop' è misurato da dj_form [H]; techno/EDM, non rap"},
    "fx_budget_penalty": {"source_level": "P", "rule_level": "H", "verified": "B&B 2003 cap. 29 ('secret weapon')",
                          "note": "penalità 0,10 nostra"},
    "backspin_context_bonus": {"source_level": "P", "rule_level": "H", "verified": "B&B cap. 12 ('escape hatch', fine frase di A)",
                               "note": "bonus 0,20 nostro"},
    "tape_stop_punchy": {"source_level": "P", "rule_level": "H", "verified": "B&B cap. 12 ('punchy tune waiting in the wings')",
                         "note": ""},
    "halving_build": {"source_level": "A+T", "rule_level": "H",
                      "verified": "Margulis 2014 p. 23, p. 44 (unità che si restringe = accelerazione); Huron 2006 p. 319, pp. 323-326",
                      "note": "Margulis analizza Rameau, non mix DJ; durata 4/8 battute nostra"},
    "entry_break": {"source_level": "A+T+P", "rule_level": "H",
                    "verified": "Huron 2006 pp. 248-249 (fill), pp. 314-316 (ritardo), p. 368; Katz 2012 p. 106; Krims 2000 pp. 137-139",
                    "note": "un battito: valore nostro"},
    "build_gap_beats": {"source_level": "T", "rule_level": "H", "verified": "Huron 2006 pp. 314-316, p. 368",
                        "note": "1 battito, mai oltre 2: valori nostri"},
}

# Prior a priori delle famiglie (solo ordinamento; [U] = derivato dai giudizi v1 come priorità).
PRIOR: dict[str, tuple[float, str]] = {
    "lyric_link": (0.80, "[H] legame testuale raro e riconoscibile; eco [U] 9/10"),
    "build_into_chorus": (0.75, "[U] drop secco sul ritornello 3/10 → serve preparazione; [E] Solberg & Dibben 2019 (EDM)"),
    "acapella_over": (0.70, "[P] acapella su strumentale (stems); [E] Davies 2014: una voce alla volta"),
    "vocal_swap": (0.65, "[P] vocal swap con stems; [S] bass swap Vande Veire 2018"),
    "loop_extend": (0.65, "[P] loop per allungare l'uscita; [U] blend 8 battute 8/10"),
    "echo_freeze": (0.60, "[U] echo out 9/10 e 7/10 (famiglia eco)"),
    "note_bridge": (0.50, "[H] ponte armonico per tempi lontani"),
    "reverb_wash": (0.50, "[P] reverb out; [H]"),
    "backspin": (0.45, "[P] backspin hip-hop; [H] effetto marcato, da usare con parsimonia"),
    "tape_stop": (0.40, "[P] brake/tape stop; [H] effetto marcato, da usare con parsimonia"),
    "halving_build": (0.70, "[A] Margulis 2014 pp. 23, 44; [T] Huron 2006 pp. 323-326; [H] valore: non ancora ascoltato, solo profilo libri"),
    "break_before_entry": (0.55, "[A] Huron 2006 pp. 248-249, 314-316; [P] Katz 2012 p. 106; [H] valore: non ancora ascoltato, solo profilo libri"),
}

POLICY_TABLE: list[dict[str, str]] = [
    {"id": "C1_lyric", "context": "parola finale di A e parola iniziale di B legate (uguale, lemma, rima, Soundex) con ASR affidabile",
     "techniques": "lyric_link (eco della parola di A, K-esima ripetizione sulla parola di B)", "duration": "dalla parola di A alla parola di B (1–3 battute)",
     "source": "[P] word play nel DJing hip-hop; [P] B&B 2003 cap. 23 'How to choose by meaning' / 'Telling tales' (legare i brani col testo); [A] Williams 2013 pp. 1-2 (il prestito come estetica hip-hop); [H] allineamento dell'eco"},
    {"id": "C2_chorus", "context": "B entra su un ritornello/hook (profilo libri: anche sul ritorno della cassa di B, drop di dj_form)", "techniques": "build_into_chorus (HPF + rullo + riser in tonalità di B, stutter facoltativo)",
     "duration": "2 battute (4 se B è molto più energico di A o il tempo è compatibile)", "source": "[U] drop secco 3/10; [E] Solberg & Dibben 2019; [P] build-up; [P] B&B cap. 9 ('Slow fade-in, then crash out') e cap. 28 ('cut the bass for a few bars during a buildup and then return it'); [A] Butler 2006 pp. 246-247 (bassi tolti per 4 battiti alla fine di 16 battute, poi 'dropping the beat')"},
    {"id": "C3_verse_handoff", "context": "A finisce un verso sul cue (voce prima, ultimo battito libero), B inizia a cantare sul cue, tempo entro ±8%, tonalità compatibile o a ±1 semitono, stem disponibili",
     "techniques": "acapella_over (4 battute; 8 se stretch <= 5% e B canta 8 battute) e vocal_swap (4 + 2 battute)",
     "duration": "4–8 battute", "source": "[P] acapella/vocal swap con stems; [P] B&B cap. 31 ('Play an acappella of one track over a dub of another' solo con tonalità compatibili) e cap. 28 ('Blend a vocal ... from one track over the kicks and bass-line of another'); [A] Butler p. 244 (mai due voci predominanti insieme); [E] Brungart 2001 / Davies 2014: una voce alla volta; [E] Kim 2020: trasposizioni reali quasi solo di 1 semitono"},
    {"id": "C4_extend", "context": "tempo entro ±8%, tonalità compatibile o a ±1, e A non ha abbastanza battute strumentali dopo il cue (voce che riprende o brano che finisce) oppure voci in entrambi",
     "techniques": "loop_extend (loop di 1–2 battute strumentali di A sotto l'ingresso di B)", "duration": "8 battute se stretch <= 5%, altrimenti 4",
     "source": "[P] loop per estendere l'outro; [P] Webber 2008 cap. 18 pp. 248-252 (estensione del break: loop di 2 battute, poi 1 o 4); [P] B&B cap. 13 ('bridge record', 'bonus beats') e cap. 33 (back-cue = quick mix di Flash); [A] Katz 2012 pp. 56-57; [S] Vande Veire 2018 bass swap; [E] Kim 2020 multipli di 8 battute"},
    {"id": "C5_far_tempo", "context": "tempo oltre ±8% (niente beatmatch)",
     "techniques": "echo_freeze; note_bridge con 1 battuta di solo pad se la tonalità è compatibile; backspin se A ha voce o energia alta sull'uscita; tape_stop se B è più calmo",
     "duration": "1 battuta di effetto (+1–2 di coda)", "source": "[U] echo 9/10; [P] spinback/brake: B&B cap. 12 (spinback a fine frase di A prima di un ingresso forte di B; power-off prima di un brano 'punchy'); [P] B&B cap. 29 (moderazione degli effetti); [H] pad ponte"},
    {"id": "C6_energy_drop", "context": "energia di B molto più bassa di A (< −6 dB relativi)", "techniques": "reverb_wash o tape_stop",
     "duration": "1 battuta di wash + 2 di coda / 2 battiti di stop", "source": "[P] reverb out / brake per chiudere; [P] B&B cap. 24 ('Fresh starts': dopo un culmine si riparte da un tempo diverso); [H]"},
    {"id": "C7_default", "context": "nessuna delle precedenti o proposte insufficienti", "techniques": "echo_freeze, reverb_wash",
     "duration": "1 + 1 battute", "source": "[U] famiglia eco 9/10; [H]"},
    {"id": "C2b_halving", "context": "[profilo libri, halving_build] come C2 (B entra su ritornello o drop), almeno 4 battute di A prima del cue",
     "techniques": "halving_build (loop di A che si dimezza fino al downbeat di B, passa-alto che sale, pausa facoltativa)",
     "duration": "4 battute (8 se B è molto più energico)",
     "source": "[A] Margulis 2014 p. 23, p. 44 (restringere l'unità ripetuta = accelerazione), pp. 130-132 (traiettoria); [T] Huron 2006 "
               "pp. 319, 323-326; pp. 314-316 (pausa prima del downbeat)"},
    {"id": "C8_break", "context": "[profilo libri, entry_break] A esce a confine di sezione, B entra a inizio sezione (downbeat ipermetrico certo)",
     "techniques": "break_before_entry: beat_mute (A canta sull'ultimo battito, stem), fill (1 battito), stop (1 battito, tempo lontano)",
     "duration": "1 battito",
     "source": "[A] Huron 2006 pp. 248-249 (fill), pp. 314-316 e p. 368 (ritardo breve); [A] Krims 2000 pp. 137-139 (stop); [P] Katz 2012 "
               "p. 106 (drop out sulla punchline); [A] Butler 2006 pp. 91-92, 188-190 (rimozioni anacrusiche)"},
]


def _p(name: str, overrides: Optional[Mapping[str, Any]] = None) -> Any:
    if overrides and name in overrides:
        return overrides[name]
    return PARAMS[name][0]


def creative_context(A: Mapping[str, Any], B: Mapping[str, Any], plan: Mapping[str, Any], pair: Mapping[str, Any], *,
                     has_stems: bool = True, lyric_links: Optional[Sequence[Mapping[str, Any]]] = None,
                     recent_techniques: Sequence[str] = ()) -> dict[str, Any]:
    """Contesto di :func:`musicmix.dj_set.transition_context` + chiavi per le tecniche creative."""
    from musicmix import dj_creative as cr
    from musicmix import dj_set as ds

    ctx = dict(ds.transition_context(A, B, plan, pair))
    ka, kb = (A.get("key") or {}).get("key"), (B.get("key") or {}).get("key")
    ctx.update(key_a=ka, key_b=kb, key_shift=cr.auto_key_shift(ka, kb, 1), has_stems=bool(has_stems),
               lyric_links=list(lyric_links or []), recent_techniques=list(recent_techniques),
               bar_a_s=4.0 * float(A["tempo"]["period_s"]), bar_b_s=4.0 * float(B["tempo"]["period_s"]))
    ja = int(plan["a_out_bar"])
    va = A["bars"]["vocal"]
    ctx["a_vocal_bars_before"] = [float(x) for x in va[max(0, ja - 4) : ja]]
    return ctx


def _bars_after_vocal(bars: Sequence[float], thr: float) -> int:
    """Battute consecutive con voce dall'inizio della lista."""
    k = 0
    for x in bars:
        if x < thr:
            break
        k += 1
    return k


def _proposal(tech: str, params: dict, bars: float, why: list[str], rule: str, bar_s: float, risks: Sequence[str] = ()) -> dict:
    prior, src = PRIOR[tech]
    return {"technique": tech, "params": params, "duration_bars": bars, "duration_s_estimate": bars * bar_s,
            "why": why, "rule": rule, "rule_source": next(r["source"] for r in POLICY_TABLE if r["id"] == rule),
            "prior": prior, "prior_source": src, "risks": list(risks)}


def propose(ctx: Mapping[str, Any], *, max_proposals: int = 4, min_proposals: int = 2,
            params: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Proposte ordinate (2–4) di tecniche creative per il contesto ``ctx``.

    Restituisce ``{proposals, considered, trace, version}``: ``proposals`` con
    ``technique``, ``params`` (argomenti per :func:`musicmix.dj_creative.render_creative`,
    esclusi audio, griglie e cue), ``duration_bars``, ``why``, ``rule`` e fonte,
    ``prior`` (solo ordinamento), ``risks``; ``considered`` elenca anche le tecniche
    scartate con il motivo.
    """
    thr = _p("vocal_bar", params)
    inst = _p("instrumental_share", params)
    s = float(ctx.get("stretch_abs", 1.0))
    tempo_ok = s <= _p("max_stretch", params)
    long_ok = s <= _p("long_stretch", params)
    rel = ctx.get("key_relation", "unknown")
    key_ok = rel in ("same", "adjacent", "relative") and bool(ctx.get("key_confident", False))
    shift = ctx.get("key_shift")
    key_fixable = key_ok or (shift is not None and abs(int(shift)) <= 1 and bool(ctx.get("key_confident", False)))
    shift_b = 0 if key_ok or shift is None else int(shift)
    stems = bool(ctx.get("has_stems", False))
    bar_a = float(ctx.get("bar_a_s", 2.0))
    av_before = list(ctx.get("a_vocal_bars_before") or [])
    av_after = list(ctx.get("a_vocal_bars") or [])
    bv_after = list(ctx.get("b_vocal_bars") or [])
    a_exit_clear = float(ctx.get("a_vocal_last_beat", 1.0)) <= _p("exit_vocal_max", params)
    a_verse_end = a_exit_clear and len(av_before) >= 2 and max(av_before[-2:]) >= thr
    b_sings_at_entry = bool(bv_after) and bv_after[0] >= thr
    b_sing_bars = _bars_after_vocal(bv_after, thr)
    a_inst_after = 0
    for x in av_after:
        if x > inst:
            break
        a_inst_after += 1
    a_left = int(ctx.get("a_bars_after", 0))
    a_before = int(ctx.get("a_bars_before", 0))
    chorus = ctx.get("b_section") in _p("chorus_labels", params)
    drop_entry = bool(_p("drop_entry_build", params)) and bool(ctx.get("b_entry_drop", False)) and not chorus
    de = float(ctx.get("energy_b_db_rel", 0.0)) - float(ctx.get("energy_a_db_rel", 0.0))
    last_bar_inst = bool(av_before) and av_before[-1] <= inst
    links = [l for l in ctx.get("lyric_links") or [] if float(l.get("score", 0.0)) >= _p("lyric_min_score", params)]
    # stessa misura di R1_grid_low (dj_set): livello tempo.confidence di A e B
    low_levels = tuple(_p("grid_low_levels", params))
    grid_low = bool(_p("grid_guard", params)) and (ctx.get("tempo_confidence_a") in low_levels
                                                    or ctx.get("tempo_confidence_b") in low_levels)
    grid_why = (f"griglia a confidenza bassa (A {ctx.get('tempo_confidence_a')}, B {ctx.get('tempo_confidence_b')}): "
                "tecnica legata alla griglia esclusa (come R1_grid_low)")
    out: list[dict] = []
    considered: list[dict] = []
    trace = [{"inputs": {"stretch": s, "tempo_ok": tempo_ok, "key_relation": rel, "key_ok": key_ok, "key_shift": shift,
                         "stems": stems, "a_verse_end": a_verse_end, "b_sings_at_entry": b_sings_at_entry,
                         "b_sing_bars": b_sing_bars, "a_instrumental_bars_after_cue": a_inst_after, "a_bars_after": a_left,
                         "b_chorus": chorus, "energy_b_minus_a_db": de, "lyric_links": len(links), "grid_low": grid_low}}]

    def skip(tech, why):
        considered.append({"technique": tech, "proposed": False, "why": why})

    # C1 lyric link
    if links:
        lk = links[0]
        sr = float(ctx.get("sample_rate", 48000))
        span_s = (lk["b_word"]["start_sample"] / sr - lk["b_cue_s"]) + (lk["a_cue_s"] - lk["a_word"]["start_sample"] / sr)
        out.append(_proposal("lyric_link", {"a_word": lk["a_word"], "b_word": lk["b_word"], "match": lk["match"],
                                            "cue_out_a_s": lk["a_cue_s"], "cue_in_b_s": lk["b_cue_s"]},
                             max(1.0, span_s / bar_a),
                             [f"'{lk['a_word'].get('text')}' → '{lk['b_word'].get('text')}' ({lk['match']['type']}, score {lk['score']:.2f})"],
                             "C1_lyric", bar_a, ["ASR non allineato al campione; legame percepibile solo se la parola è intelligibile"]))
    else:
        skip("lyric_link", "nessun legame di parole sopra la soglia per questi cue")
    # C2 chorus entry (o, con il profilo libri, ingresso sul ritorno della cassa di B: Butler pp. 246-248)
    if (chorus or drop_entry) and a_before >= 2:
        bb = 4 if (de >= _p("energy_rise_db", params) and a_before >= 4) else 2
        what = f"B entra su '{ctx.get('b_section')}'" if chorus else "B entra sul ritorno della sua cassa (drop, Butler pp. 246-248)"
        gb = min(2, int(_p("build_gap_beats", params) or 0))  # [T] Huron pp. 314-316, p. 368: at most 2 beats
        stutter = 1 if bb >= 2 and not a_exit_clear and not grid_low else 0
        if grid_low:
            gb = 0  # la pausa cade su un battito della griglia
        bpar = {"build_bars": bb, "snare": "auto", "riser": True, "key_b": ctx.get("key_b"), "stutter_bars": stutter}
        if gb:
            bpar["gap_beats"] = gb
        bwhy = [f"{what}: build di {bb} battute invece del drop secco",
                "stutter dell'ultima battuta perché A canta sull'uscita" if stutter else "uscita di A libera" if a_exit_clear
                else "A canta sull'uscita, ma senza stutter: griglia a confidenza bassa"]
        out.append(_proposal("build_into_chorus", bpar, bb, bwhy, "C2_chorus", bar_a,
                             ["riser e rullo possono suonare EDM in un contesto rap"]
                             + (["griglia a confidenza bassa: niente stutter né pausa; il rullo segue comunque la griglia"] if grid_low else [])))
        if _p("halving_build", params) and a_before >= 4 and grid_low:
            skip("halving_build", grid_why)
        elif _p("halving_build", params) and a_before >= 4:
            hb = 8 if (de >= _p("energy_rise_db", params) and a_before >= 8) else 4
            gb = min(2, int(_p("build_gap_beats", params) or 0))
            out.append(_proposal("halving_build", {"build_bars": hb, "start_unit": 2.0 if hb == 8 else 1.0, "min_unit": 0.25,
                                                   "source": "auto", "trajectory": "hpf_rise", "gap_beats": gb},
                                 hb, [f"{what}: loop di A che si dimezza per {hb} battute (accelerazione percepita, Margulis pp. 23, 44)"]
                                 + ([f"pausa di {gb} battito prima di B (Huron pp. 314-316)"] if gb else []),
                                 "C2b_halving", bar_a, ["loop brevi ripetuti: rischio 'disco rotto' se A canta (sorgente strumentale con lo stem)"]))
    else:
        skip("build_into_chorus", "B non entra su un ritornello né su un drop" if not (chorus or drop_entry)
             else "meno di 2 battute di A prima del cue")
    # C8 break before B's hypermetric downbeat (profilo libri)
    a_sings_on = bool(av_after) and av_after[0] > inst  # A's words continue past the cue: any exit there cuts them
    if grid_low and _p("entry_break", params):
        skip("break_before_entry", grid_why)
    elif _p("entry_break", params) and bool(ctx.get("a_exit_on_boundary", False)) and bool(ctx.get("b_section_start", False)) \
            and a_before >= 1 and not a_sings_on:
        if stems and not a_exit_clear:
            mode, why_b = "beat_mute", "A chiude la frase cantando: lo strumentale tace sull'ultimo battito, la punchline resta (Katz p. 106)"
        elif not tempo_ok:
            mode, why_b = "stop", "tempo lontano: un battito di stop prima dell'uno di B (Huron pp. 314-316)"
        else:
            mode, why_b = "fill", "fill di rullante sull'ultimo battito: rende certo il downbeat di B (Huron pp. 248-249)"
        out.append(_proposal("break_before_entry", {"mode": mode, "beats": 1}, 0.25, [why_b], "C8_break", bar_a,
                             ["effetto marcato: da non ripetere nel set"]))
    else:
        skip("break_before_entry", "A continua a cantare dopo il cue (la punchline deve finire prima dell'uno: Katz p. 106)"
             if a_sings_on else "profilo libri spento o A/B non su confini di sezione")
    # C3 verse handoff (stems)
    if grid_low:
        skip("acapella_over", grid_why + " (sovrapposizione beatmatch)")
        skip("vocal_swap", grid_why + " (sovrapposizione beatmatch)")
    elif tempo_ok and key_fixable and stems and a_verse_end and b_sings_at_entry and a_left >= 4:
        n = 8 if (long_ok and b_sing_bars >= 8 and a_left >= 9) else 4
        out.append(_proposal("acapella_over", {"bars": n, "key_shift_b": shift_b}, n + 0.25,
                             [f"A chiude il verso sul cue, B canta da subito ({b_sing_bars} battute): voce di B sullo strumentale di A per {n} battute",
                              f"trasposizione di B {shift_b:+d} semitoni" if shift_b else f"tonalità {rel}"],
                             "C3_verse_handoff", bar_a, ["artefatti di separazione Demucs nello strumentale di A"]))
        if a_left >= 6:
            out.append(_proposal("vocal_swap", {"bars": 4, "out_bars": 2, "key_shift_b": shift_b}, 6,
                                 ["voce a B a fine verso di A, base di A per 4 battute, poi bass swap"], "C3_verse_handoff", bar_a,
                                 ["due strumentali sovrapposti per 6 battute: richiede tonalità compatibile"]))
    else:
        why = [w for w, c in (("tempo oltre ±8%", not tempo_ok), ("tonalità non compatibile nemmeno a ±1", not key_fixable),
                              ("stem mancanti", not stems), ("A non chiude un verso sul cue", not a_verse_end),
                              ("B non canta sul cue", not b_sings_at_entry), ("meno di 4 battute di A dopo il cue", a_left < 4)) if c]
        skip("acapella_over", "; ".join(why))
        skip("vocal_swap", "; ".join(why))
    # C4 loop extend
    both_vocal = bool(bv_after[:4]) and bool(av_after[:4]) and any(x >= thr and y >= thr for x, y in zip(av_after[:4], bv_after[:4]))
    need_loop = a_inst_after < 4 or a_left < 8 or both_vocal
    if grid_low:
        skip("loop_extend", grid_why)
    elif tempo_ok and key_fixable and need_loop and a_before >= 2 and (stems or last_bar_inst):
        n = 8 if long_ok else 4
        lb = 1 if last_bar_inst else 2
        out.append(_proposal("loop_extend", {"loop_bars": lb, "bars": n, "key_shift_b": shift_b, "source": "auto"}, n,
                             [f"A ha solo {a_inst_after} battute strumentali dopo il cue ({a_left} battute in tutto)" if not both_vocal
                              else "voci in entrambi: il loop strumentale di A toglie lo scontro", f"loop di {lb} battute, blend di {n}"],
                             "C4_extend", bar_a, ["un loop di 1 battuta ripetuto 8 volte può risultare statico"]))
    else:
        skip("loop_extend", "non serve (A ha battute strumentali) o tempo/tonalità/stem non adatti")
    # C5 / C6 / C7 effects without overlap
    eff: list[dict] = []
    if not tempo_ok:
        eff.append(_proposal("echo_freeze", {"hold_bars": 1, "tail_bars": 1}, 2, [f"tempo {100 * s:.1f}% oltre ±8%: uscita con freeze, B pulito"], "C5_far_tempo", bar_a))
        if key_ok:
            eff.append(_proposal("note_bridge", {"gap_bars": 1, "key_a": ctx.get("key_a"), "key_b": ctx.get("key_b")}, 4,
                                 ["tempo lontano ma tonalità compatibile: una battuta di pad nella tonalità comune azzera il groove"], "C5_far_tempo", bar_a))
        bs_ctx = (bool(ctx.get("a_exit_on_boundary", False)) and bool(ctx.get("b_section_start", False))
                  and de >= _p("backspin_min_energy_db", params))
        if not a_exit_clear or de >= 0 or (bs_ctx and _p("backspin_context_bonus", params)):
            bp = _proposal("backspin", {"spin_beats": 2}, 0.5, ["A canta o spinge sull'uscita: spinback di 2 battiti e B sull'uno"], "C5_far_tempo", bar_a,
                           ["effetto marcato: da non ripetere nel set"])
            if bs_ctx and _p("backspin_context_bonus", params):
                bp["prior"] += float(_p("backspin_context_bonus", params))
                bp["why"].append("fine frase/sezione di A e ingresso forte di B a inizio sezione (B&B cap. 12)")
            eff.append(bp)
        if de < 0:
            eff.append(_proposal("tape_stop", {"stop_beats": 2}, 0.5, ["B più calmo: brake di A in 2 battiti"], "C5_far_tempo", bar_a))
        elif _p("tape_stop_punchy", params) and de >= _p("energy_rise_db", params):
            eff.append(_proposal("tape_stop", {"stop_beats": 2}, 0.5, ["B molto più energico: A si spegne e B entra di colpo (power-off, B&B cap. 12)"],
                                 "C5_far_tempo", bar_a, ["effetto marcato: da non ripetere nel set"]))
    if de < _p("energy_drop_db", params):
        eff.append(_proposal("reverb_wash", {"wash_bars": 1, "tail_bars": 2}, 3, [f"energia di B {de:+.1f} dB rispetto ad A: lavaggio di riverbero"], "C6_energy_drop", bar_a))
        eff.append(_proposal("tape_stop", {"stop_beats": 2}, 0.5, ["calo di energia: brake"], "C6_energy_drop", bar_a))
    if key_ok and tempo_ok and not any(p["technique"] == "note_bridge" for p in eff):
        eff.append(_proposal("note_bridge", {"gap_bars": 0, "key_a": ctx.get("key_a"), "key_b": ctx.get("key_b")}, 3,
                             ["tonalità compatibile: pad nella tonalità comune sotto la giunta"], "C7_default", bar_a))
    eff.append(_proposal("echo_freeze", {"hold_bars": 1, "tail_bars": 1}, 2, ["uscita con effetto, B pulito sul suo downbeat"], "C7_default", bar_a))
    eff.append(_proposal("reverb_wash", {"wash_bars": 1, "tail_bars": 2}, 3, ["uscita con riverbero, B pulito"], "C7_default", bar_a))
    seen = {p["technique"] for p in out}
    for p in eff:
        if p["technique"] not in seen:
            out.append(p)
            seen.add(p["technique"])
    # variety across the set [T] and ranking
    recent = list(ctx.get("recent_techniques") or [])
    marked = tuple(_p("fx_marked", params))
    fx_used = any(t in marked for t in recent)
    for p in out:
        p["score"] = p["prior"] - (_p("repeat_penalty", params) if p["technique"] in recent else 0.0)
        if fx_used and p["technique"] in marked:  # [P] B&B cap. 29: moderazione degli effetti marcati
            p["score"] -= float(_p("fx_budget_penalty", params))
        if p["rule"] in ("C1_lyric", "C2_chorus", "C2b_halving", "C3_verse_handoff", "C4_extend"):
            p["score"] += 0.1  # context-specific rules before generic exits [H]
    out.sort(key=lambda p: -p["score"])
    # keep families diverse: at most one of the "effect exits" before the context rules are exhausted
    chosen: list[dict] = []
    for p in out:
        if len(chosen) >= max_proposals:
            break
        chosen.append(p)
    if len(chosen) < min_proposals:
        chosen = out[:min_proposals]
    for p in out:
        considered.append({"technique": p["technique"], "proposed": p in chosen, "score": p["score"], "rule": p["rule"]})
    return {"version": POLICY_VERSION, "proposals": chosen, "considered": considered, "trace": trace,
            "note": "prior/score order the proposals only; they are not quality estimates (render and listen, or let dj_critic compare)"}


def table_markdown() -> str:
    """Tabella contesto → tecnica in Markdown (per la documentazione)."""
    rows = ["| Regola | Contesto | Tecniche | Durata | Fonte |", "|---|---|---|---|---|"]
    for r in POLICY_TABLE:
        rows.append(f"| {r['id']} | {r['context']} | {r['techniques']} | {r['duration']} | {r['source']} |")
    return "\n".join(rows)
