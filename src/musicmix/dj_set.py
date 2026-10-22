"""Set DJ continuo: tipo di transizione, ordine dei brani, cue lungo il set, giunte.

Livelli di Williams et al. 2025 [T] (transizione / traccia / mix), in tre parti
pure (nessun I/O audio, nessun modello), usate da ``tools/run_dj_set_v1.py``:

1. **Albero di decisione del tipo di transizione** (:func:`decide_transition`):
   regole esplicite in ordine fisso, la prima che si applica decide; ogni regola
   ha un identificativo, una condizione leggibile e una fonte (``TREE_RULES``).
   I parametri sono in ``TREE_PARAMS`` con la loro origine:

   * ``[E]`` risultato empirico di un paper, ``[S]`` regola di un sistema
     pubblicato, ``[P]`` pratica DJ, ``[T]`` teoria/rassegna;
   * ``[U]`` *dai giudizi utente v1* (``giudizi_cloud/dj_gallery_v1``, 6 voti
     di un solo ascoltatore: filter out 9, echo out 9, blend 8 battute 8, echo
     out 7, 7, drop diretto sul ritornello 3). Usati come priorità di
     progetto dichiarate, **non** come validazione: nessuna soglia è stata
     stimata da quei voti;
   * ``[H]`` ipotesi di progetto, da verificare con ascolto.

   Ingresso: il contesto di una coppia con i suoi cue (:func:`transition_context`,
   dalle analisi di :mod:`musicmix.dj_select`); uscita: tipo (``blend`` N
   battute, ``filter_out``, ``echo_out``, ``cut`` sul downbeat,
   ``filter_into_chorus`` = filter su A di 1–2 battute e ingresso di B sul
   ritornello al posto del drop secco), parametri di render, alternative
   ordinate per il QA e la traccia delle condizioni valutate. Il drop secco sul
   ritornello non è mai prodotto.

2. **Ordine del set** (:func:`order_set`): cammino minimo esatto (Held–Karp
   su sottoinsiemi, :func:`held_karp_path`) di lunghezza L tra 8 e 15 brani;
   costo di arco = tempo (scala log2 con equivalenza d'ottava 70↔140, Bittner
   2017 [S]; forte oltre ±10%, Ishizaki 2009 [E]; vietato oltre
   ``tempo_hard_max``), tonalità (Camelot: vincolo duro per i blend, già
   nell'albero; costo morbido per echo/cut/filter, Kell & Tzanetakis 2013 [E]),
   timbro (distanza tra MFCC medi standardizzati, Kell & Tzanetakis 2013 [E],
   Bittner 2017 [S]), qualità del miglior cue e tipo scelto dall'albero; costo
   di nodo = scarto dall'arco di energia richiesto (salita verso un picco e
   chiusura, parametrizzabile: Cliff 2000 [S] come interfaccia, forma [H]).

3. **Cue lungo il set** (:func:`choose_cues`): Viterbi esatto sui piani di
   :func:`musicmix.dj_select.plan_pair` di ogni coppia consecutiva, con
   vincolo sulla durata di ciascun brano nel set (``play_min_s``–
   ``play_max_s``, dall'ingresso alla successiva uscita), spazio per le giunte
   e una piccola penalità per lo stesso tipo ripetuto due volte di fila
   (Williams 2025 [T], che cita Hawkins 2013, tesi non letta: evidenza debole).

Più alcune funzioni di geometria delle clip (:func:`clip_geometry`) per
giuntare le clip di ``dj_render``/``dj_effects`` con l'audio intero dei brani.

4. **Profilo "libri" (2026-10-21)** (:data:`PROFILE_BOOKS`, :func:`profile`):
   regole tratte da cinque libri sul mestiere DJ (note in ``conoscenza_mix/libri/``),
   tutte **spente nei default** (parametri con valore neutro) e accese solo
   passando il profilo. Etichette delle fonti nei commenti: ``[P]`` pratica
   dichiarata da DJ esperti o manuali (Brewster & Broughton 2003 = "B&B",
   Webber 2008), ``[A]`` analisi accademica/etnografica (Katz 2012, Butler
   2006, Williams 2013), ``[H]`` ipotesi/valore scelto qui. Nessun libro misura
   l'effetto di queste regole: sono pratica, non evidenza.

   * taglio "sull'uno" dopo una sezione intera nell'hip-hop (B&B cap. 11, 13),
     taglio invece di effetto quando le tonalità cozzano (B&B cap. 15, 31);
   * blend brevi (max 8 battute) nell'hip-hop (B&B cap. 9);
   * backspin per tempi lontani a fine frase di A e su un ingresso forte di B
     (B&B cap. 12);
   * durata di ogni brano a **sezioni intere** (almeno strofa + ritornello
     completi, uscita a fine sezione, meglio subito dopo un ritornello:
     B&B cap. 13; Webber p. 225; Butler p. 222; Katz pp. 56–57, 75, 138);
   * arco di energia a più picchi con l'ultimo più alto (Butler p. 251;
     B&B cap. 24), leggera tendenza ad accelerare (Butler pp. 251–252);
   * moderazione degli effetti marcati (B&B cap. 29);
   * categorie formali di Butler calcolate sull'audio (:mod:`musicmix.dj_form`, se le
     analisi hanno ``form``): niente uscita durante un buildup di A (pp. 222–224, 242),
     uscita nel breakdown e ingresso sul "drop" di B favoriti (pp. 221–222, 245–247),
     blend accorciati o più costosi quando le tessiture sono piene o "busy" (p. 245).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from musicmix import dj_form
from musicmix import dj_select as djs

SCHEMA_VERSION = "musicmix.dj_set.1"


@dataclass(frozen=True)
class Param:
    value: Any
    source: str


# --------------------------------------------------------------- parameters
TREE_PARAMS: dict[str, Param] = {
    "blend_max_stretch": Param(djs.TEMPO_MAX_STRETCH, "[S] MosaikBox 2024 ±8%; coerente con Ishizaki 2009 [E] (disagio oltre ±10%)"),
    "long_blend_max_stretch": Param(djs.TEMPO_LONG_BLEND_STRETCH, "[E] Kim 2020 §6.1: 86,1% degli stretch entro 5%"),
    "blend_key_relations": Param(("same", "adjacent", "relative"), "[S] Vande Veire 2018 §4.2.1; [E] Kim 2020 §6.2 (key lock)"),
    "key_confident": Param(djs.KEY_CONFIDENT, "[H] dj_select.KEY_CONFIDENT"),
    "grid_low_levels": Param(("low",), "[H] griglia non affidabile: niente beatmatch"),
    "vocal_bar_active": Param(djs.VOCAL_BAR_ACTIVE, "[H] dj_select: battuta vocale se coperta >= 50%"),
    "clash_tolerated_bars": Param(djs.CLASH_TOLERATED_BARS, "[S] Vande Veire 2018 §4.2.4: clash da 2 battute vocali sovrapposte"),
    "instrumental_max_share": Param(0.25, "[H] parte 'strumentale' se la quota vocale media delle battute è <= 0,25"),
    "exit_vocal_max_last_beat": Param(djs.CUT_GUARD_MAX_COVER, "[H] dj_select.CUT_GUARD_MAX_COVER: uscita secca solo se l'ultimo battito di A è libero"),
    "chorus_labels": Param(("chorus",), "[H] etichette allin1 considerate ritornello"),
    "chorus_prep_sweep_bars": Param(2, "[U] drop diretto sul ritornello 3/10; [E] Solberg & Dibben 2019 (EDM: drop dopo preparazione); test proposto in REVISIONE_SET §7: 1–2 battute"),
    "filter_sweep_bars": Param(2, "[U] filter out 9/10 (galleria v1: sweep di 2 battute)"),
    "echo_delay_beats": Param(1.0, "[U] echo out 9, 7, 7 (galleria v1: delay di 1 battito)"),
    "echo_entry_bars": Param(1, "[U] galleria v1: eco agganciato una battuta prima, B pulito sul cue"),
    "energy_drop_db": Param(-6.0, "[H] stessa soglia di dj_select (calo di energia relativo)"),
    "blend_bars_instrumental": Param(16, "[E] Kim 2020 §6.3 (picchi a 32/64 beat) solo con stretch <= 5% e 16 battute strumentali"),
    "blend_bars_default": Param(8, "[U] blend 8 battute 8/10; [E] Kim 2020 (32 beat = 8 battute)"),
    "blend_bars_short": Param(4, "[H] blend breve con una sola voce (Vande Veire tollera 1 battuta)"),
    # ---- profilo libri (default neutri: comportamento v1 invariato)
    "hiphop_cut": Param(False, "[P] B&B 2003 cap. 11 'How to cut' (p. 68): i cut 'are key for breakbeat styles, like hip hop'; "
                               "cap. 13 'Even tighter' (p. 76 sgg.): nell'hip-hop si esce subito dopo un ritornello, di solito con un cut; "
                               "[A] Katz 2012 pp. 56-57 (quick mix di Flash), p. 138 (quick-mixing = segue di frammenti brevi). "
                               "Se vero: con due voci (R6) si taglia sull'uno quando A ha chiuso una strofa/ritornello a confine "
                               "di sezione e B parte a inizio sezione senza voce sul primo battito, invece di filter/echo"),
    "cut_after_labels": Param(("verse", "chorus"), "[P] B&B cap. 13 'How to find mix points': 'always let a singer finish their verse or chorus'"),
    "key_clash_cut": Param(False, "[P] B&B cap. 15 'How to mix tunes that clash': 'Cut them, don't blend them'; cap. 31 (p. 168): "
                                  "con tonalità diverse 'most DJs ... do some kind of cut'; [P] Webber 2008 p. 238. "
                                  "Se vero: R4 sceglie il cut sull'uno quando ammissibile, altrimenti l'uscita con effetto"),
    "backspin_far_tempo": Param(False, "[P] B&B cap. 12 'How to do stops and spinbacks' (p. 64 sgg.): spinback 'an escape hatch if you "
                                       "can't get two records to mix properly', da fare 'at the end of a phrase in record 1 and just before "
                                       "an exciting start in record 2'; tipico di 'ragga, drum & bass, and hip hop DJs'. Se vero: R2 "
                                       "(tempo oltre ±8%, cut non ammesso) usa il backspin quando A esce a confine di sezione e B entra "
                                       "a inizio sezione non più calmo di A"),
    "backspin_spin_beats": Param(2, "[H] durata dello spin (dj_creative.backspin: 1|2|4 battiti)"),
    "backspin_min_energy_db": Param(-1.0, "[H] 'exciting start' di B: energia di B >= energia di A - 1 dB"),
    "dense_blend_layers": Param(None, "[A] Butler 2006 p. 245 [corretto 21/10/2026: era p. 244] (DJ Vixen Swift: 'if it's too busy, I don't think it sounds good'; due "
                                      "superfici ritmiche attive = 'muddled', 'busy') e p. 129 (le parti sottili all'inizio/fine dei brani "
                                      "'decrease the likelihood of a clash'); p. 232 [corretto: era p. 231] (Adam Jay: tolgo elementi a fine brano 'to make room in "
                                      "the mix for sounds from the next record'). Se impostato (strati medi di A + B sulle 8 battute della "
                                      "sovrapposizione, da musicmix.dj_form) e superato, o se entrambi sono 'busy' su >= meta' delle battute, "
                                      "il blend si accorcia a blend_bars_short (R7c_dense). None = spento; valore [H]"),
}

#: tipi d'uscita "marcati" (effetti udibili come tali): da usare con moderazione [P] B&B cap. 29
#: ("Effects should be your secret weapon, not your constant companion").
FX_MARKED = ("echo_out", "backspin")

# Costo a priori del tipo (0 = preferito) usato solo per ordinare cue e coppie.
TYPE_COST: dict[str, Param] = {
    "filter_out": Param(0.10, "[U] filter out 9/10"),
    "echo_out": Param(0.20, "[U] echo out 9, 7, 7 (media 7,7)"),
    "blend8": Param(0.20, "[U] blend 8 battute 8/10"),
    "blend16": Param(0.30, "[H] non ancora ascoltato"),
    "blend4": Param(0.30, "[H] non ancora ascoltato"),
    "filter_into_chorus": Param(0.30, "[H] non ancora ascoltato (sostituisce il drop secco, [U] 3/10)"),
    "cut": Param(0.50, "[H] taglio sul downbeat non votato in v1"),
    "backspin": Param(0.40, "[P] B&B cap. 12 (escape hatch); [H] valore: effetto marcato non votato, prodotto solo dal profilo libri"),
}

ORDER_PARAMS: dict[str, Param] = {
    "len_min": Param(8, "richiesta: set di 8–15 brani"),
    "len_max": Param(15, "richiesta: set di 8–15 brani"),
    "len_target": Param(10, "[H] 10 brani × 90 s ≈ 15 min"),
    "max_exact": Param(15, "Held–Karp esatto fino a 15 brani candidati (2^n · n² operazioni); oltre si preseleziona"),
    "tempo_hard_max": Param(0.20, "[H] salto di tempo (ripiegato d'ottava) vietato oltre il 20%; Kim 2020 [E]: 98,6% degli stretch entro 20%"),
    "tempo_ref": Param(0.08, "[S] MosaikBox ±8%: costo (s/0,08)^2"),
    "tempo_ishizaki": Param(0.10, "[E] Ishizaki 2009: disagio oltre ±10%"),
    "tempo_ishizaki_step": Param(1.0, "[H] gradino di costo oltre ±10%"),
    "w_tempo": Param(1.0, "[H]"),
    "key_cost": Param({"same": 0.0, "relative": 0.25, "adjacent": 0.25, "diagonal": 0.75, "clash": 1.0, "unknown": 0.75},
                      "[S] Vande Veire 2018 / dj_select (stessa scala delle penalità)"),
    "w_key_blend": Param(1.0, "[H] tonalità piena nei blend (già vincolo duro dell'albero)"),
    "w_key_soft": Param(0.4, "[E] Kell & Tzanetakis 2013: tonalità non criterio primario; [E] Bittner 2017: key clash 0%"),
    "w_timbre": Param(0.6, "[E] Kell & Tzanetakis 2013 (ordine reale timbricamente più coerente); [S] Bittner 2017"),
    "w_cue": Param(0.5, "[H] qualità del miglior cue (dj_select.cue_score)"),
    "cue_ref": Param(5.0, "[H] cue_score di riferimento: costo = max(0, ref - score) / ref"),
    "w_type": Param(1.0, "[H] peso del costo a priori del tipo (TYPE_COST)"),
    "w_arc": Param(2.0, "[H] peso dell'arco di energia"),
    "w_smooth": Param(0.3, "[H] salto di energia tra brani consecutivi (alternanza locale ammessa: Neto 2024)"),
    "arc_start": Param(0.35, "[H] energia relativa del primo brano"),
    "arc_peak": Param(0.95, "[H] energia al picco"),
    "arc_peak_pos": Param(0.7, "[H] posizione del picco (frazione del set)"),
    "arc_end": Param(0.5, "[H] energia della chiusura"),
    # ---- profilo libri (default neutri)
    "arc_shape": Param("single", "'single' = salita a un picco e chiusura (v1, [H]); 'waves' = più picchi, l'ultimo più alto: "
                                 "[A] Butler 2006 p. 251 [corretto: era p. 250] ('More commonly, ... DJs create several major climaxes within a set, with the "
                                 "last one usually being the most dramatic'; DJ Shiva: 'pick them up; let them down; pick them up "
                                 "higher'); [P] B&B 2003 cap. 24 'How to pace the night' (set 'full of undulations, peaks, and "
                                 "troughs'; 'The workout'); [P] Webber 2008 p. 243 ('goes up from low to high then it goes back down')"),
    "arc_waves_knots": Param(((0.0, 0.3), (0.4, 0.75), (0.55, 0.5), (0.82, 0.95), (1.0, 0.55)),
                             "[H] nodi (posizione, energia) dell'arco 'waves': partenza bassa (B&B cap. 21: warm-up), primo picco, "
                             "valle ('fresh starts', B&B cap. 24), picco finale più alto (Butler p. 251), chiusura più bassa"),
    "arc_waves_min_len": Param(6, "[H] sotto i 6 brani l'arco a onde degenera: si usa 'single'"),
    "w_tempo_down": Param(0.0, "[A] Butler 2006 pp. 251-252: i DJ alzano il tempo nel set ma di poco ('just a few beats per minute "
                               "really makes a difference'); [P] B&B cap. 21 (warm-up: 'gently increase the tempo'). Costo per un calo "
                               "di tempo (ripiegato d'ottava) = w * calo / 8%; 0 = spento"),
    "type_cost_override": Param({}, "costi a priori dei tipi sostituiti dal profilo (vedi PROFILE_BOOKS)"),
    "w_exit_midsection": Param(0.0, "[P] B&B cap. 10 ('If you cut one off halfway ... it will throw your dancers off') e cap. 13 ('always "
                                    "let a singer finish their verse or chorus'); [P] Webber p. 225: uscita di A non a inizio sezione"),
    "w_exit_after_verse": Param(0.0, "[P] B&B cap. 13 'Even tighter': nell'hip-hop si esce 'immediately after the chorus'; un'uscita "
                                     "dopo una strofa (prima del ritornello) costa un po' di più [H valore]"),
    "w_entry_midsection": Param(0.0, "[P] B&B cap. 10 ('Cue up your incoming record so its first beat is the start of a phrase'); "
                                     "[A] Butler p. 243 [corretto: era p. 242] (allineare le sezioni): ingresso di B non a inizio sezione"),
    "w_phrase4": Param(0.0, "[P] B&B cap. 10 ('Dance music is built on an eight-bar grid'); [A] Butler p. 206 [corretto: era p. 205] (sequenze da 8 e 16 "
                            "battute): cue di B o di A solo su frase da 4 (non da 8/16)"),
    # ---- categorie formali di Butler (musicmix.dj_form; servono A['form'] e B['form'], altrimenti 0)
    "w_exit_in_build": Param(0.0, "[A] Butler 2006 pp. 222-225: il buildup porta al core (p. 225 [corretto]: 'one does not have a core followed by a "
                                  "buildup'); [P] Mystik in Butler p. 243 [corretto: era p. 242] ('You'll end up having two records clash — the builds won't be "
                                  "the same'): costo se A esce mentre sta costruendo (battuta prima dell'uscita = build e A prosegue in "
                                  "build/core)"),
    "w_exit_breakdown_bonus": Param(0.0, "[P] DJ Shiva in Butler 2006 p. 223 [corretto: era pp. 221-222] (il primo breakdown corto: 'That's usually where you mix "
                                         "out or you cut the other record out'); [P] B&B cap. 13 ('mix from the breaks'): bonus (costo "
                                         "negativo) se l'uscita di A cade all'inizio o dentro un suo breakdown"),
    "w_entry_drop_bonus": Param(0.0, "[A] Butler 2006 p. 3 e pp. 246-248 ('dropping the beat': il ritorno della cassa è il momento piu' "
                                     "forte; 'it almost always drops abruptly'): bonus (costo negativo) se B entra esattamente sul ritorno "
                                     "della sua cassa (drop di musicmix.dj_form)"),
    "w_dense_overlap": Param(0.0, "[A] Butler 2006 p. 245 ('busy', 'muddled') e p. 244 (mai due voci predominanti insieme) [pagine corrette 21/10/2026: erano 244 e 243]; p. 129: costo "
                                  "dei blend con tessitura piena: w * (max(0, strati A+B - dense_ref_layers) / 2 + quota di battute "
                                  "'busy' in entrambi)"),
    "w_cut_full_both": Param(0.0, "[P] B&B 2003 cap. 11 'When to cut' (p. 68): 'the more continuous melody there is on both sides of the "
                                  "join, the less likely it is that a cut will sound good'; [A] Butler 2006 p. 129 (parti sottili a inizio/"
                                  "fine brano). Costo di un cut se A (battuta prima del cut) e B (prima battuta) hanno entrambi la "
                                  "tessitura piena del loro core (musicmix.dj_form: n_layers >= core_level). [H] proxy: strati, non "
                                  "melodia; 0 = spento (anche nel profilo libri: proxy non validato)"),
    "dense_ref_layers": Param(6.0, "[H] somma media degli strati attivi di A e B (su 5+5) oltre la quale il blend costa di piu'"),
    # ---- generi lontani (libri, 21/10/2026; proposta B1)
    "far_timbre_pct": Param(None, "[H] ipotesi ingegneristica (proposta B1), NON una soglia dei libri. Contesto: Huron 2006 pp. 207-208 [E] "
                                  "(Perrott & Gjerdingen: lo stile si riconosce in 250 ms - 1 s, il timbro porta l'informazione di stile); "
                                  "p. 211 [E] (Krumhansl & Kessler: modulazione a tonalità VICINE 'firmly established' entro tre accordi, "
                                  "'a few seconds', con un residuo); il passaggio fra generi lontani è una congettura dell'autore ('I doubt "
                                  "that the switch would take more than a few seconds') e p. 213 chiede più ricerca. Verifica 21/10/2026 "
                                  "(revisione dei nove libri, ROOT_CHECKS.md): né il minimo di 3 s né il 90° percentile timbrico sono "
                                  "provati; la distanza timbrica del pool non è una misura calibrata di genere o familiarità. Coppia "
                                  "'lontana' se la distanza supera questo percentile del pool (None = spento): blend sotto "
                                  "far_min_blend_s non ammessi e coda d'eco fermata sul downbeat di B (far_echo_tail_stop). Con una "
                                  "matrice degenere (dispersione < far_timbre_min_spread) ci si astiene"),
    "far_timbre_min_spread": Param(1e-6, "[H] dispersione minima (IQR delle distanze fuori diagonale, unità di timbre_distances) sotto la quale "
                                         "il percentile non separa nulla: astensione (nessuna coppia 'lontana'). Controesempio dell'audit: "
                                         "matrice tutta 0 -> soglia 0 -> una coppia a distanza 0 classificata lontana"),
    "far_min_blend_s": Param(3.0, "[H] durata minima di un blend tra coppie 'lontane' (si spegne con None). Huron p. 211 riguarda tonalità "
                                  "vicine (tre accordi), non generi lontani né la durata di una sovrapposizione DJ"),
    "far_echo_tail_stop": Param(True, "[H] con coppie 'lontane' la coda d'eco di A si ferma sul downbeat di B (False = spento)"),
}

CUE_PARAMS: dict[str, Param] = {
    "play_min_s": Param(60.0, "richiesta: 60–120 s per brano"),
    "play_max_s": Param(120.0, "richiesta: 60–120 s per brano"),
    "play_target_s": Param(90.0, "[H] centro dell'intervallo"),
    "w_dur": Param(1.0, "[H] costo ((d - target)/target)^2"),
    "w_repeat": Param(0.15, "[T] Williams 2025 (cita Hawkins 2013, debole): varietà dei tipi"),
    "min_gap_s": Param(8.0, "[H] B (fine transizione) -> prossima uscita: spazio per la coda d'eco e la giunta"),
    "relax_steps": Param(3, "[H] allargamenti successivi dei limiti di durata se il set è infeasible"),
    # ---- profilo libri (default neutri)
    "dur_mode": Param("seconds", "'seconds' = solo durata in secondi (v1); 'sections' = ogni brano suona sezioni intere: almeno una "
                                 "strofa e un ritornello completi (se il brano li ha), cioè il 'core' di Butler p. 222 ('the song', "
                                 "'the meat of the song'); [P] B&B cap. 13 (lasciar finire strofa/ritornello; uscire subito dopo il "
                                 "ritornello nell'hip-hop); [P] Webber p. 225 (una strofa o un ritornello = 2-4 frasi: non interrompere "
                                 "una frase); [A] Katz pp. 56-57, 75, 138 (il DJ hip-hop suona frammenti scelti, non brani interi). "
                                 "I limiti in secondi restano come vincolo duro"),
    "core_labels": Param(("verse", "chorus"), "[H] etichette allin1 contate come sezioni 'core' (strofa/ritornello)"),
    "target_core_sections": Param(2, "[H] obiettivo: un ciclo strofa + ritornello (Katz p. 138: frammenti brevi; B&B cap. 13 'tighter mixing')"),
    "w_sections": Param(1.0, "[H] costo ((n_core - target) / target)^2 in modo 'sections'"),
    "w_fx_run": Param(0.0, "[P] B&B cap. 29 'How to add FX': 'moderation ... Effects should be your secret weapon, not your constant "
                           "companion': costo aggiuntivo se due transizioni consecutive sono entrambe effetti marcati (FX_MARKED)"),
    # ---- piano per eventi (P0.5, 21/10/2026): opzione SPERIMENTALE, spenta (0)
    "w_event_arrival": Param(0.0, "[H] sperimentale (P0.5): bonus (costo negativo) per un candidato in cui l'evento di A è completato "
                                  "(uscita a confine di sezione, nessun verso troncato) e l'evento obiettivo di B (ritornello, drop, "
                                  "strofa, prima voce) arriva entro event_arrival_max_bars dallo switch (musicmix.dj_events."
                                  "arrival_summary); 'a parità di costo tecnico': cambia solo scelte con differenza di costo < peso. "
                                  "Contesto, non soglia: [P] B&B cap. 13 'Even tighter' ('mix straight after a chorus ... the big "
                                  "opening moment in record 2 happens immediately after the chorus in record one'); [A] Butler 2006 "
                                  "p. 246 (la cassa che torna, anche 'when a new record enters the mix'), [P] p. 247 (Mystik). 0 = spento"),
    "event_arrival_max_bars": Param(2.0, "[H] sperimentale: 'entro 1-2 battute' dallo switch (0 = evento sullo switch); nessun libro dà "
                                         "un numero"),
}


def values(defaults: Mapping[str, Param], overrides: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Valori dei parametri (``Param.value``) con eventuali sostituzioni."""
    out = {k: p.value for k, p in defaults.items()}
    for k, v in (overrides or {}).items():
        if k not in out:
            raise KeyError(f"unknown parameter {k!r}")
        out[k] = v
    return out


def documented(defaults: Mapping[str, Param], overrides: Optional[Mapping[str, Any]] = None) -> dict[str, dict]:
    """``{nome: {value, source}}`` per i JSON delle run."""
    vals = values(defaults, overrides)
    return {k: {"value": vals[k], "source": p.source + (" (sostituito)" if overrides and k in overrides else "")}
            for k, p in defaults.items()}


#: Profilo "libri" (2026-10-21): sostituzioni dei parametri per i tre livelli. Ogni voce ha la
#: fonte nel ``Param`` corrispondente e il livello di evidenza verificato in PROFILE_BOOKS_EVIDENCE;
#: qui solo i valori. [H] = valore scelto qui (tutti i pesi e le soglie).
PROFILE_BOOKS: dict[str, dict[str, Any]] = {
    "tree": {
        "hiphop_cut": True,              # [P] B&B cap. 11, 13; [A] Katz pp. 56-57, 138
        "key_clash_cut": True,           # [P] B&B cap. 15, 31; Webber p. 238
        "backspin_far_tempo": True,      # [P] B&B cap. 12
        "blend_bars_instrumental": 8,    # [P] B&B cap. 9: 'Quick blend ... works well for hip hop'; hip hop 'use cuts more than blends'
        "dense_blend_layers": 7.0,       # [A] Butler p. 245, p. 129; [H] soglia (su 10 strati possibili)
    },
    "order": {
        "arc_shape": "waves",            # [A] Butler p. 251; [P] B&B cap. 24
        "w_tempo_down": 0.2,             # [A] Butler pp. 251-252 (accelerazione lieve); [H] peso piccolo
        "type_cost_override": {"cut": 0.20},  # [P] B&B cap. 11 (cut 'key' nell'hip-hop); [H] valore: tra filter (0,10) ed echo/blend8 (0,20) votati [U]
        "w_exit_midsection": 0.5,        # [P] B&B cap. 10, 13; Webber p. 225; [H] peso
        "w_exit_after_verse": 0.15,      # [P] B&B cap. 13 'Even tighter'; [H] peso
        "w_entry_midsection": 0.3,       # [P] B&B cap. 10; [H] peso
        "w_phrase4": 0.15,               # [P] B&B cap. 10; [A] Butler p. 206; [H] peso
        "w_exit_in_build": 0.3,          # [A] Butler pp. 222-225, p. 243; [H] peso
        "w_exit_breakdown_bonus": 0.1,   # [P] Shiva in Butler p. 223; B&B cap. 13; [H] peso
        "w_entry_drop_bonus": 0.1,       # [A] Butler p. 3, pp. 246-248; [H] peso
        "w_dense_overlap": 0.2,          # [A] Butler p. 245, p. 244, p. 129; [H] peso
        "far_timbre_pct": 90.0,          # [H] 90° percentile (proposta B1); Huron pp. 207-211 è solo contesto, non prova la soglia
    },
    "cues": {
        "dur_mode": "sections",          # [P] B&B cap. 13; Webber p. 225; [A] Butler p. 222; Katz pp. 75, 138
        "w_dur": 0.3,                    # [H] i secondi restano solo come centratura debole
        "w_fx_run": 0.3,                 # [P] B&B cap. 29 (moderazione)
    },
}


#: Livello di evidenza delle voci di PROFILE_BOOKS (audit dei nove libri, 21/10/2026): ``source_level`` = che cosa offre
#: la fonte ([E] sperimentale, [T] teoria, [A] analisi, [P] pratica/manuale/intervista, [H] nostra ipotesi);
#: ``rule_level`` = stato del valore scelto qui (il peso o la soglia sono sempre [H]: nessuna voce è validata su
#: ascolti). ``verified`` = pagine STAMPATE rilette nel testo per pagina (tools/libri.py; B&B è un EPUB: capitolo).
#: Correzioni di pagina di Butler 2006 trovate in questa verifica (le citazioni nei Param erano spesso 1 pagina prima):
#: 'busy' p. 245 (non 244), due voci p. 244 (non 243), Mystik p. 243 (non 242), Adam Jay p. 232 (non 231),
#: Shiva p. 223 (non 221-222), climax p. 251 (non 250), 'one does not have a core followed by a buildup' p. 225,
#: sequenze di 8/16 battute p. 206 (non 205), breakdown 'at least four measures' pp. 91-92, drop pp. 246-248.
PROFILE_BOOKS_EVIDENCE: dict[str, dict[str, str]] = {
    "tree.hiphop_cut": {"source_level": "P+A", "rule_level": "H",
                        "verified": "B&B 2003 cap. 11 ('key for breakbeat styles'), cap. 13 'Even tighter'; Katz 2012 pp. 56-57, 138",
                        "note": "le condizioni dell'albero (confine di sezione, primo battito libero) sono nostre"},
    "tree.key_clash_cut": {"source_level": "P", "rule_level": "H",
                           "verified": "B&B cap. 15 ('Cut them, don't blend them'), cap. 31; Webber 2008 p. 238 (rischio di clash)",
                           "note": "limite Williams 2013 p. 10: un frammento leggermente stonato può essere una citazione voluta; "
                                   "qui si tratta di due brani interi, non di un campione"},
    "tree.backspin_far_tempo": {"source_level": "P", "rule_level": "H", "verified": "B&B cap. 12 ('escape hatch')", "note": ""},
    "tree.blend_bars_instrumental": {"source_level": "P", "rule_level": "H", "verified": "B&B cap. 9 ('Quick blend')",
                                     "note": "8 battute è un valore nostro"},
    "tree.dense_blend_layers": {"source_level": "A+P", "rule_level": "H",
                                "verified": "Butler 2006 p. 245 ('too busy'), p. 244 (due voci), p. 129 (meno strati = meno clash), p. 232",
                                "note": "soglia 7 su 10 strati nostra; 'busy' misura debole (dj_form)"},
    "order.arc_shape": {"source_level": "A+P", "rule_level": "H",
                        "verified": "Butler 2006 p. 251 (più climax, l'ultimo il più forte); B&B cap. 24 ('undulations'); Webber p. 243",
                        "note": "nodi dell'arco nostri"},
    "order.w_tempo_down": {"source_level": "A+P", "rule_level": "H", "verified": "Butler 2006 pp. 251-252; B&B cap. 21", "note": ""},
    "order.type_cost_override": {"source_level": "P", "rule_level": "H", "verified": "B&B cap. 11", "note": "valore 0,20 nostro"},
    "order.w_exit_midsection": {"source_level": "P", "rule_level": "H",
                                "verified": "B&B cap. 10 ('halfway'), cap. 13 ('let a singer finish'); Webber p. 225", "note": ""},
    "order.w_exit_after_verse": {"source_level": "P", "rule_level": "H", "verified": "B&B cap. 13 'Even tighter'", "note": ""},
    "order.w_entry_midsection": {"source_level": "P+A", "rule_level": "H",
                                 "verified": "B&B cap. 10 ('first beat is the start of a phrase'); Butler p. 243 (sezioni allineate)",
                                 "note": ""},
    "order.w_phrase4": {"source_level": "P+A", "rule_level": "H", "verified": "B&B cap. 10 ('eight-bar grid'); Butler p. 206",
                        "note": "EDM/house; nel rap le frasi da 4 sono normali (Krims p. 53)"},
    "order.w_exit_in_build": {"source_level": "A+P", "rule_level": "H",
                              "verified": "Butler 2006 pp. 222-225 (funzioni formali; p. 225 'one does not have a core followed by a "
                                          "buildup'), p. 243 (Mystik)", "note": "il 'build' qui è misurato da dj_form [H]"},
    "order.w_exit_breakdown_bonus": {"source_level": "P", "rule_level": "H",
                                     "verified": "Butler 2006 p. 223 (DJ Shiva: 'mix out or you cut'); B&B cap. 13 (mix from breaks)",
                                     "note": ""},
    "order.w_entry_drop_bonus": {"source_level": "A+P", "rule_level": "H", "verified": "Butler 2006 p. 3, pp. 246-248", "note": ""},
    "order.w_dense_overlap": {"source_level": "A+P", "rule_level": "H", "verified": "Butler 2006 p. 245, p. 244, p. 129", "note": ""},
    "order.far_timbre_pct": {"source_level": "E+T", "rule_level": "H",
                             "verified": "Huron 2006 pp. 207-208 (stile in 250 ms - 1 s), p. 211 (tre accordi per tonalità vicine; "
                                         "generi lontani = congettura), p. 213",
                             "note": "né 3 s né il 90° percentile sono nei libri; distanza timbrica non calibrata come genere; "
                                     "astensione con matrice degenere; opzioni far_min_blend_s/far_echo_tail_stop"},
    "cues.dur_mode": {"source_level": "P+A", "rule_level": "H",
                      "verified": "B&B cap. 13; Webber p. 225; Butler pp. 222-223 ('the meat of the song'); Katz p. 75, p. 138",
                      "note": ""},
    "cues.w_dur": {"source_level": "H", "rule_level": "H", "verified": "(nessuna fonte: centratura debole)", "note": ""},
    "cues.w_fx_run": {"source_level": "P", "rule_level": "H", "verified": "B&B cap. 29 ('secret weapon')", "note": ""},
}


def profile(name: Optional[str]) -> dict[str, dict[str, Any]]:
    """Sostituzioni ``{"tree", "order", "cues"}`` del profilo ``name`` (``None``/"v1" = nessuna)."""
    if name in (None, "", "v1", "default"):
        return {"tree": {}, "order": {}, "cues": {}}
    if name == "books":
        return {k: dict(v) for k, v in PROFILE_BOOKS.items()}
    raise KeyError(f"unknown profile {name!r}")


TREE_RULES: list[dict[str, str]] = [
    {"id": "R1_grid_low", "if": "griglia di A o B a confidenza bassa",
     "then": "echo_out (nessun beatmatch richiesto)", "source": "[H]"},
    {"id": "R2_tempo_out", "if": "stretch ripiegato d'ottava > blend_max_stretch (8%)",
     "then": "cut sul downbeat se l'uscita di A è libera dalla voce, a confine di sezione, e B parte strumentale a inizio sezione (non ritornello); altrimenti echo_out",
     "source": "[S] MosaikBox ±8%; [E] Ishizaki 2009; [T] Williams 2025 (cut, echo out); [U] echo out 9/7/7"},
    {"id": "R3_chorus_entry", "if": "B entra su un ritornello",
     "then": "mai drop secco: filter su A per 1–2 battute poi B sul downbeat del ritornello (filter_into_chorus); echo_out se A ha voce nell'ultimo battito",
     "source": "[U] drop diretto 3/10; [E] Solberg & Dibben 2019; [E] Brungart 2001; [S] Bittner 2017 (vocalness)"},
    {"id": "R4_key_no_blend", "if": "tonalità non in {stessa, ±1, relativa} o stima poco confidente",
     "then": "niente sovrapposizione: filter_out (echo_out se A ha voce nell'ultimo battito)",
     "source": "[S] Vande Veire 2018 §4.2.1 (vincolo duro per i blend)"},
    {"id": "R5_energy_drop", "if": "energia di B all'ingresso < energia di A in uscita − 6 dB (relativi)",
     "then": "echo_out (la coda copre il calo)", "source": "[H]"},
    {"id": "R6_both_vocal", "if": "voce in entrambi già nel blend più corto (clash > 1 battuta su 4)",
     "then": "una voce alla volta: echo_out se A ha voce nell'ultimo battito, altrimenti filter_out a fine frase; mai blend",
     "source": "[E] Brungart 2001 (mascheramento informativo, parlato); [S] Vande Veire 2018; [S] MosaikBox"},
    {"id": "R7_instrumental", "if": "A e B strumentali sulle battute della sovrapposizione (quota vocale <= 0,25, clash <= 1)",
     "then": "blend 16 battute se stretch <= 5% e 16 battute strumentali per entrambi, altrimenti blend 8",
     "source": "[E] Kim 2020 §6.3; [U] blend 8 battute 8/10"},
    {"id": "R8_one_voice", "if": "al massimo una voce alla volta (clash <= 1 battuta)",
     "then": "blend 8 se stretch <= 5% e 8 battute senza clash, altrimenti blend 4",
     "source": "[S] Vande Veire 2018 (1 battuta tollerata); [H]"},
    {"id": "R9_default", "if": "nessuna delle precedenti (battute insufficienti)",
     "then": "filter_out", "source": "[U] filter out 9/10"},
    # ---- rami del profilo libri (attivi solo con i parametri corrispondenti)
    {"id": "R2b_backspin", "if": "[profilo libri, backspin_far_tempo] R2 senza cut ammissibile, A esce a confine di sezione, B entra a "
                                 "inizio sezione (non ritornello) con energia >= A - 1 dB",
     "then": "backspin di 2 battiti su A, B pulito sul suo downbeat", "source": "[P] Brewster & Broughton 2003 cap. 12 (spinback)"},
    {"id": "R7c_dense", "if": "[profilo libri, dense_blend_layers] blend previsto ma tessiture piene in A e B sulla sovrapposizione "
                              "(strati medi A+B >= soglia, o entrambi 'busy' su >= meta' delle battute; musicmix.dj_form)",
     "then": "blend accorciato a blend_bars_short (4 battute)",
     "source": "[A] Butler 2006 p. 244 ('too busy'), p. 129, p. 231 (sezioni sottili per mixare)"},
    {"id": "R4b_key_cut", "if": "[profilo libri, key_clash_cut] tonalità incompatibile e cut ammissibile",
     "then": "cut sull'uno (le due tonalità non suonano mai insieme)",
     "source": "[P] Brewster & Broughton 2003 cap. 15 ('Cut them, don't blend them'), cap. 31; Webber 2008 p. 238"},
    {"id": "R6b_hiphop_cut", "if": "[profilo libri, hiphop_cut] voci in entrambi, A ha appena chiuso una strofa/ritornello a confine di "
                                   "sezione con l'ultimo battito libero, B parte a inizio sezione senza voce sul primo battito",
     "then": "cut sull'uno (quick mix) invece di filter/echo",
     "source": "[P] Brewster & Broughton 2003 cap. 11, 13; [A] Katz 2012 pp. 56-57, 138"},
]


# ------------------------------------------------------------------ context
def _bars(track: Mapping[str, Any], key: str, start: int, n: int) -> list[float]:
    arr = track["bars"].get(key) or []
    return [float(x) for x in arr[max(0, start):max(0, start) + n]]


def transition_context(A: Mapping[str, Any], B: Mapping[str, Any], plan: Mapping[str, Any],
                       pair: Mapping[str, Any], events: bool = False) -> dict[str, Any]:
    """Ingressi dell'albero per un piano di :func:`musicmix.dj_select.plan_pair`.

    Convenzione "start" di dj_select: A esce dal downbeat ``a_out_bar`` (per un
    blend continua per N battute da lì), B entra dal downbeat ``b_in_bar``.

    ``events`` (opzione, default False = contesto invariato): aggiunge ``event_context`` con l'evento di A
    all'uscita e gli eventi di B dopo l'ingresso (:mod:`musicmix.dj_events`, P0.5). Dato additivo: l'albero
    e i costi non lo leggono.
    """
    ja, jb = int(plan["a_out_bar"]), int(plan["b_in_bar"])
    rows_a, rows_b = A["phrases"], B["phrases"]
    last_beat = _bars(A, "vocal_last_beat", ja - 1, 1)
    last_bar = _bars(A, "vocal", ja - 1, 1)
    t, k = pair["tempo"], pair["key"]
    return {
        "a": A.get("slug"), "b": B.get("slug"),
        "a_out_bar": ja, "b_in_bar": jb, "a_out_s": float(plan["a_out_s"]), "b_in_s": float(plan["b_in_s"]),
        "bpm_a": float(A["tempo"]["bpm"]), "bpm_b": float(B["tempo"]["bpm"]),
        "stretch_abs": abs(float(t["stretch_b_pct"])) / 100.0, "octave_factor": t.get("octave_factor", 1.0),
        "tempo_confidence_a": A["tempo"]["confidence"], "tempo_confidence_b": B["tempo"]["confidence"],
        "key_relation": k["relation"], "key_confident": bool(k.get("confident")),
        "camelot_a": k.get("camelot_a"), "camelot_b": k.get("camelot_b"),
        "a_vocal_last_beat": last_beat[0] if last_beat else 0.0,
        "a_vocal_last_bar": last_bar[0] if last_bar else 0.0,
        "a_vocal_bars": _bars(A, "vocal", ja, 16),
        "b_vocal_bars": _bars(B, "vocal", jb, 16),
        "a_bars_before": ja, "a_bars_after": len(rows_a) - ja, "b_bars_after": len(rows_b) - jb,
        "a_exit_on_boundary": bool(rows_a[ja]["section_start"]) if ja < len(rows_a) else False,
        "a_exit_section": rows_a[ja - 1]["section_label"] if 0 < ja <= len(rows_a) else "start",
        "b_section": rows_b[jb]["section_label"] if jb < len(rows_b) else "unlabeled",
        "b_section_start": bool(rows_b[jb]["section_start"]) if jb < len(rows_b) else False,
        "a_phrase_level": int(plan.get("a_out_phrase_level", 0)), "b_phrase_level": int(plan.get("b_in_phrase_level", 0)),
        "energy_a_db_rel": float(plan.get("energy_a_db_rel", 0.0)), "energy_b_db_rel": float(plan.get("energy_b_db_rel", 0.0)),
        **form_context(A, B, ja, jb),
        **({"event_context": event_context(A, B, plan)} if events else {}),
    }


def event_context(A: Mapping[str, Any], B: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    """Evento di A all'uscita ed eventi di B dopo l'ingresso (solo struttura: allin1, dj_form, battute vocali).
    Il piano completo, con dj_phrase, alternative e trace, è :func:`musicmix.dj_events.event_plan`."""
    from musicmix import dj_events as dje  # lazy: dj_events uses this module

    ae = dje.a_event(A, int(plan["a_out_bar"]), float(plan["a_out_s"]))
    ev = dje.b_events(B, int(plan["b_in_bar"]), float(plan["b_in_s"]))
    return {"a_kind": ae["kind"], "a_span_bars": ae["span_bars"], "a_fate": ae["fate"], "a_skipped_counts": ae["skipped_counts"],
            "b_events": [{k: e[k] for k in ("kind", "bar", "bars_after_downbeat")} for e in ev],
            "b_target": dje.pick_target(ev)["kind"]}


#: battute della finestra di sovrapposizione usata per le feature formali [H] (= blend di default)
FORM_WINDOW_BARS = 8


def form_context(A: Mapping[str, Any], B: Mapping[str, Any], ja: int, jb: int) -> dict[str, Any]:
    """Chiavi formali di Butler per l'albero e i costi (:mod:`musicmix.dj_form`).

    Servono ``A["form"]`` e ``B["form"]`` (documenti di :func:`musicmix.dj_form.analyse`
    sulla stessa griglia di battute di ``dj_select``); senza, ``form_available`` è falso
    e tutte le regole/costi formali restano spenti.
    """
    fa, fb = A.get("form"), B.get("form")
    if not fa or not fb:
        return {"form_available": False}
    prof = dj_form.overlap_profile(fa, fb, ja, jb, FORM_WINDOW_BARS)
    a_bd = any(a <= ja < b or ja == a for a, b in fa.get("breakdowns", []))
    la, lb = fa.get("n_layers") or [], fb.get("n_layers") or []
    # a track without a measurable core (dj_form.2: labels "unknown") has no "full texture of its core" [H]
    a_core, b_core = fa.get("core_status", "found") != "no_core", fb.get("core_status", "found") != "no_core"
    a_full = bool(a_core and 0 < ja <= len(la) and fa.get("core_level") is not None and la[ja - 1] >= fa["core_level"])
    b_full = bool(b_core and 0 <= jb < len(lb) and fb.get("core_level") is not None and lb[jb] >= fb["core_level"])
    return {"form_available": True, "a_core_known": a_core, "b_core_known": b_core, "a_full_before": a_full, "b_full_at": b_full, "a_form_before": prof["a_form_before"], "a_form_at": prof["a_form_at"],
            "b_form_at": prof["b_form_at"], "b_entry_drop": prof["b_entry_drop"], "a_exit_in_breakdown": bool(a_bd),
            "overlap_layers": prof["sum_layers"], "overlap_both_busy": prof["both_busy_share"],
            "overlap_both_vocals": prof["both_vocals_share"], "a_layers_out": prof["a_layers"], "b_layers_in": prof["b_layers"]}


# --------------------------------------------------------------------- tree
def _clash(ctx: Mapping[str, Any], n: int, thr: float) -> Optional[int]:
    a, b = ctx["a_vocal_bars"][:n], ctx["b_vocal_bars"][:n]
    if len(a) < n or len(b) < n:
        return None
    return int(sum(1 for x, y in zip(a, b) if x >= thr and y >= thr))


def _share(bars: Sequence[float], n: int) -> Optional[float]:
    return float(np.mean(bars[:n])) if len(bars) >= n else None


def _type_key(d: Mapping[str, Any]) -> str:
    return f"blend{d['bars']}" if d["type"] == "blend" else d["type"]


def _mk(kind: str, p: Mapping[str, Any], bars: int = 0, sweep: Optional[int] = None) -> dict[str, Any]:
    if kind == "blend":
        return {"type": "blend", "bars": int(bars), "render": {"bars": int(bars), "cue_convention": "start"},
                "label": f"blend {bars} battute"}
    if kind in ("filter_out", "filter_into_chorus"):
        s = int(sweep or p["filter_sweep_bars"])
        return {"type": kind, "bars": 0, "render": {"sweep_bars": s},
                "label": ("filter su A " + f"{s} batt. + ingresso sul ritornello") if kind == "filter_into_chorus" else f"filter out {s} batt."}
    if kind == "echo_out":
        return {"type": "echo_out", "bars": 0,
                "render": {"delay_beats": p["echo_delay_beats"], "entry_bars": p["echo_entry_bars"]}, "label": "echo out"}
    if kind == "cut":
        return {"type": "cut", "bars": 0, "render": {}, "label": "taglio sul downbeat"}
    if kind == "backspin":
        sb = int(p.get("backspin_spin_beats", 2))
        return {"type": "backspin", "bars": 0, "render": {"spin_beats": sb}, "label": f"backspin {sb} battiti"}
    raise ValueError(kind)


def decide_transition(ctx: Mapping[str, Any], params: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Albero di decisione (``TREE_RULES``) su un contesto di :func:`transition_context`.

    Restituisce ``{type, bars, render, label, type_key, rule, rule_source,
    trace, fallbacks}``; ``fallbacks`` sono le alternative ordinate (stesso cue)
    da provare se il QA fallisce. Non produce mai il drop secco.
    """
    p = values(TREE_PARAMS, params)
    thr = p["vocal_bar_active"]
    tol = p["clash_tolerated_bars"]
    inst = p["instrumental_max_share"]
    s = float(ctx["stretch_abs"])
    trace: list[dict[str, Any]] = []

    def check(rule: str, cond: str, ok: bool) -> bool:
        trace.append({"rule": rule, "condition": cond, "result": bool(ok)})
        return bool(ok)

    a_voc_exit = ctx["a_vocal_last_beat"] > p["exit_vocal_max_last_beat"]
    chorus = ctx["b_section"] in p["chorus_labels"]
    cut_ok = (not a_voc_exit and ctx["a_vocal_last_bar"] < thr and ctx["a_exit_on_boundary"]
              and ctx["b_section_start"] and not chorus
              and bool(ctx["b_vocal_bars"]) and ctx["b_vocal_bars"][0] < thr)
    sweep_ok = ctx["a_bars_before"] >= p["filter_sweep_bars"]
    echo_ok = ctx["a_bars_before"] >= 1
    # [P] B&B cap. 13: si taglia dopo che il cantante ha finito strofa o ritornello (A esce a confine di sezione)
    hiphop_cut_ok = cut_ok and ctx.get("a_exit_section") in tuple(p["cut_after_labels"])
    # [P] B&B cap. 12: spinback a fine frase di A, subito prima di un ingresso forte di B (non sul ritornello: [U] drop 3/10)
    backspin_ok = (bool(ctx["a_exit_on_boundary"]) and bool(ctx["b_section_start"]) and not chorus and echo_ok
                   and ctx["energy_b_db_rel"] - ctx["energy_a_db_rel"] >= p["backspin_min_energy_db"])
    trace.append({"rule": "inputs", "condition": (
        f"stretch {100 * s:.1f}%, tonalità {ctx['key_relation']} (confidente {ctx['key_confident']}), "
        f"voce A ultimo battito {ctx['a_vocal_last_beat']:.2f}, B entra su '{ctx['b_section']}'"
        f"{' (inizio sezione)' if ctx['b_section_start'] else ''}, cut ammissibile {cut_ok}"), "result": True})

    def effect_exit() -> dict[str, Any]:
        # una voce alla volta: uscita con effetto, echo se A canta sull'uscita
        if a_voc_exit or not sweep_ok:
            return _mk("echo_out", p)
        return _mk("filter_out", p)

    d: Optional[dict[str, Any]] = None
    rule = None
    if check("R1_grid_low", "griglia A o B a confidenza bassa",
             ctx["tempo_confidence_a"] in p["grid_low_levels"] or ctx["tempo_confidence_b"] in p["grid_low_levels"]):
        d, rule = _mk("echo_out", p), "R1_grid_low"
    elif check("R2_tempo_out", f"stretch {100 * s:.1f}% > {100 * p['blend_max_stretch']:.0f}%", s > p["blend_max_stretch"]):
        if cut_ok:
            d, rule = _mk("cut", p), "R2_tempo_out"
        elif p["backspin_far_tempo"] and check("R2b_backspin", "A esce a confine di sezione, B a inizio sezione non più calmo",
                                               backspin_ok):
            d, rule = _mk("backspin", p), "R2b_backspin"
        else:
            d, rule = _mk("echo_out", p), "R2_tempo_out"
    elif check("R3_chorus_entry", f"B entra su '{ctx['b_section']}' in {tuple(p['chorus_labels'])}", chorus):
        if a_voc_exit or ctx["a_bars_before"] < 1:
            d = _mk("echo_out", p)
        else:
            d = _mk("filter_into_chorus", p, sweep=min(p["chorus_prep_sweep_bars"], ctx["a_bars_before"]))
        rule = "R3_chorus_entry"
    elif check("R4_key_no_blend", f"tonalità {ctx['key_relation']} non in {tuple(p['blend_key_relations'])} o non confidente",
               ctx["key_relation"] not in p["blend_key_relations"] or not ctx["key_confident"]):
        if p["key_clash_cut"] and check("R4b_key_cut", "cut ammissibile (le tonalità non suonano mai insieme)", cut_ok):
            d, rule = _mk("cut", p), "R4b_key_cut"
        else:
            d, rule = effect_exit(), "R4_key_no_blend"
    elif check("R5_energy_drop", f"energia B − A = {ctx['energy_b_db_rel'] - ctx['energy_a_db_rel']:+.1f} dB < {p['energy_drop_db']:+.0f}",
               ctx["energy_b_db_rel"] - ctx["energy_a_db_rel"] < p["energy_drop_db"]):
        d, rule = _mk("echo_out", p), "R5_energy_drop"
    else:
        c4 = _clash(ctx, p["blend_bars_short"], thr)
        if check("R6_both_vocal", f"clash su {p['blend_bars_short']} battute = {c4} > {tol} (o battute insufficienti)",
                 c4 is None or c4 > tol):
            if p["hiphop_cut"] and check("R6b_hiphop_cut", f"A chiude '{ctx.get('a_exit_section')}' a confine di sezione, "
                                         "B parte a inizio sezione senza voce sul primo battito", hiphop_cut_ok):
                d, rule = _mk("cut", p), "R6b_hiphop_cut"
            else:
                d, rule = effect_exit(), "R6_both_vocal"
        else:
            n8, n16 = p["blend_bars_default"], p["blend_bars_instrumental"]
            c8, c16 = _clash(ctx, n8, thr), _clash(ctx, n16, thr)
            sa8, sb8 = _share(ctx["a_vocal_bars"], n8), _share(ctx["b_vocal_bars"], n8)
            sa16, sb16 = _share(ctx["a_vocal_bars"], n16), _share(ctx["b_vocal_bars"], n16)
            instr8 = c8 is not None and c8 <= tol and sa8 is not None and sb8 is not None and sa8 <= inst and sb8 <= inst
            if check("R7_instrumental", f"quote vocali su {n8} battute A {sa8}, B {sb8} <= {inst}, clash {c8}", instr8):
                long_ok = (s <= p["long_blend_max_stretch"] and c16 is not None and c16 <= tol
                           and sa16 is not None and sb16 is not None and sa16 <= inst and sb16 <= inst)
                trace.append({"rule": "R7_instrumental", "condition": f"16 battute strumentali e stretch <= {100 * p['long_blend_max_stretch']:.0f}%",
                              "result": bool(long_ok)})
                d, rule = _mk("blend", p, bars=n16 if long_ok else n8), "R7_instrumental"
            elif check("R8_one_voice", f"clash su {n8} battute = {c8} <= {tol} e stretch <= {100 * p['long_blend_max_stretch']:.0f}%",
                       c8 is not None and c8 <= tol and s <= p["long_blend_max_stretch"]):
                d, rule = _mk("blend", p, bars=n8), "R8_one_voice"
            elif check("R8_one_voice", f"clash su {p['blend_bars_short']} battute = {c4} <= {tol}", True):
                d, rule = _mk("blend", p, bars=p["blend_bars_short"]), "R8_one_voice"
    if d is None:
        d, rule = _mk("filter_out", p), "R9_default"
    # [A] Butler p. 244 / p. 129: tessiture piene in entrambi -> blend corto (profilo libri)
    if (d["type"] == "blend" and p["dense_blend_layers"] is not None and ctx.get("form_available")
            and d["bars"] > p["blend_bars_short"]):
        ol, bb = ctx.get("overlap_layers"), ctx.get("overlap_both_busy")
        dense = (ol is not None and ol >= float(p["dense_blend_layers"])) or (bb is not None and bb >= 0.5)
        if check("R7c_dense", f"strati A+B {ol if ol is None else round(ol, 2)} >= {p['dense_blend_layers']} o 'busy' in "
                              f"entrambi {bb if bb is None else round(bb, 2)} >= 0,5", dense):
            d, rule = _mk("blend", p, bars=p["blend_bars_short"]), "R7c_dense"

    # ---- alternatives on the same cue (QA fallback order) [H]
    fb: list[dict[str, Any]] = []
    if d["type"] == "blend":
        n = d["bars"] // 2
        while n >= p["blend_bars_short"]:
            c = _clash(ctx, n, thr)
            if c is not None and c <= tol:
                fb.append(_mk("blend", p, bars=n))
            n //= 2
        if sweep_ok:
            fb.append(_mk("filter_out", p))
        if echo_ok:
            fb.append(_mk("echo_out", p))
    elif d["type"] in ("filter_out", "filter_into_chorus"):
        if echo_ok:
            fb.append(_mk("echo_out", p))
        if cut_ok:
            fb.append(_mk("cut", p))
    elif d["type"] == "echo_out":
        if sweep_ok and not chorus:
            fb.append(_mk("filter_out", p))
        elif sweep_ok and chorus:
            fb.append(_mk("filter_into_chorus", p, sweep=min(p["chorus_prep_sweep_bars"], ctx["a_bars_before"])))
        if cut_ok:
            fb.append(_mk("cut", p))
    elif d["type"] == "cut":
        if echo_ok:
            fb.append(_mk("echo_out", p))
        if sweep_ok and not chorus:
            fb.append(_mk("filter_out", p))
    elif d["type"] == "backspin":
        if echo_ok:
            fb.append(_mk("echo_out", p))
        if sweep_ok:
            fb.append(_mk("filter_out", p))
    if (d["type"] == "echo_out" and p["backspin_far_tempo"] and s > p["blend_max_stretch"] and backspin_ok
            and not any(f["type"] == "backspin" for f in fb)):
        fb.append(_mk("backspin", p))
    src = next(r["source"] for r in TREE_RULES if r["id"] == rule)
    for x in fb:
        x.update(type_key=_type_key(x), rule=f"{rule}:alternativa", rule_source=src, trace=trace, fallbacks=[])
    return {**d, "type_key": _type_key(d), "rule": rule, "rule_source": src, "trace": trace, "fallbacks": fb,
            "cut_admissible": bool(cut_ok)}


def type_cost(decision: Mapping[str, Any], override: Optional[Mapping[str, float]] = None) -> float:
    """Costo a priori del tipo (``TYPE_COST``), con eventuali sostituzioni del profilo."""
    k = decision["type_key"]
    if override and k in override:
        return float(override[k])
    return float(TYPE_COST[k].value)


# ------------------------------------------------------------ pair features
def folded_tempo(bpm_a: float, bpm_b: float) -> dict[str, float]:
    """Distanza di tempo in scala log2 con equivalenza d'ottava (70 ↔ 140).

    ``log2_dist`` = min_k |log2(bpm_b / bpm_a) - k|; ``stretch`` = 2^dist - 1
    (frazione), ``octave`` = 2^k scelto. Stessa piegatura di
    :func:`musicmix.dj_select.tempo_compatibility` (fattori 1/2, 1, 2).
    """
    r = math.log2(bpm_b / bpm_a)
    k = round(r)
    d = abs(r - k)
    return {"log2_dist": d, "stretch": 2.0 ** d - 1.0, "octave": 2.0 ** k}


def tempo_cost(bpm_a: float, bpm_b: float, params: Optional[Mapping[str, Any]] = None) -> float:
    p = values(ORDER_PARAMS, params)
    s = folded_tempo(bpm_a, bpm_b)["stretch"]
    if s > p["tempo_hard_max"]:
        return math.inf
    c = (s / p["tempo_ref"]) ** 2
    if s > p["tempo_ishizaki"]:
        c += p["tempo_ishizaki_step"]
    return float(c)


def zscore_rows(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=float)
    sd = m.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return (m - m.mean(axis=0)) / sd


def timbre_distances(vectors: Mapping[str, Sequence[float]], slugs: Sequence[str]) -> np.ndarray:
    """Distanza euclidea / sqrt(dim) tra vettori timbrici standardizzati sul pool."""
    m = zscore_rows(np.array([vectors[s] for s in slugs], dtype=float))
    diff = m[:, None, :] - m[None, :, :]
    return np.sqrt((diff ** 2).sum(-1) / m.shape[1])


def energy_scores(raw: Mapping[str, float], slugs: Optional[Sequence[str]] = None) -> dict[str, float]:
    """Energia relativa in [0, 1] = rango percentile nel pool (pareggi mediati)."""
    slugs = list(slugs or raw)
    x = np.array([raw[s] for s in slugs], dtype=float)
    if len(x) == 1:
        return {slugs[0]: 0.5}
    order = x.argsort(kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    for v in np.unique(x):  # ties
        sel = x == v
        ranks[sel] = ranks[sel].mean()
    return {s: float(r / (len(x) - 1)) for s, r in zip(slugs, ranks)}


def arc_target(pos: int, length: int, params: Optional[Mapping[str, Any]] = None) -> float:
    """Energia richiesta alla posizione ``pos`` (0..L-1).

    ``arc_shape='single'`` (default, v1): salita lineare al picco, poi chiusura.
    ``arc_shape='waves'`` (profilo libri): interpolazione lineare tra i nodi
    ``arc_waves_knots`` (più picchi, l'ultimo più alto: Butler 2006 p. 250 [A];
    Brewster & Broughton 2003 cap. 24 [P]); sotto ``arc_waves_min_len`` brani si
    torna a 'single'.
    """
    p = values(ORDER_PARAMS, params)
    if length <= 1:
        return float(p["arc_peak"])
    u = pos / (length - 1)
    if p["arc_shape"] == "waves" and length >= int(p["arc_waves_min_len"]):
        knots = [(float(x), float(y)) for x, y in p["arc_waves_knots"]]
        xs, ys = [k[0] for k in knots], [k[1] for k in knots]
        return float(np.interp(u, xs, ys))
    if p["arc_shape"] not in ("single", "waves"):
        raise ValueError(f"unknown arc_shape {p['arc_shape']!r}")
    pk = float(p["arc_peak_pos"])
    if u <= pk:
        return float(p["arc_start"] + (p["arc_peak"] - p["arc_start"]) * (u / pk if pk > 0 else 1.0))
    return float(p["arc_peak"] + (p["arc_end"] - p["arc_peak"]) * ((u - pk) / (1 - pk) if pk < 1 else 1.0))


# ------------------------------------------------------------- candidates
def placement_cost(plan: Mapping[str, Any], ctx: Optional[Mapping[str, Any]],
                   params: Optional[Mapping[str, Any]] = None) -> dict[str, float]:
    """Costi di collocazione del cue (profilo libri; tutti 0 con i default).

    * uscita di A non a inizio sezione: ``w_exit_midsection`` [P] B&B cap. 10, 13; Webber p. 225;
    * uscita subito dopo una strofa (il ritornello di quel ciclo non è suonato): ``w_exit_after_verse``
      [P] B&B cap. 13 'Even tighter' (nell'hip-hop si esce subito dopo il ritornello);
    * ingresso di B non a inizio sezione: ``w_entry_midsection`` [P] B&B cap. 10; [A] Butler p. 242;
    * cue su frase da sole 4 battute: ``w_phrase4`` per lato [P] B&B cap. 10; [A] Butler p. 205.
    """
    p = values(ORDER_PARAMS, params)
    out = {"exit_midsection": 0.0, "exit_after_verse": 0.0, "entry_midsection": 0.0, "phrase4": 0.0}
    if ctx is not None:
        if not ctx.get("a_exit_on_boundary", True):
            out["exit_midsection"] = float(p["w_exit_midsection"])
        elif ctx.get("a_exit_section") == "verse":
            out["exit_after_verse"] = float(p["w_exit_after_verse"])
        if not ctx.get("b_section_start", True):
            out["entry_midsection"] = float(p["w_entry_midsection"])
    n4 = sum(1 for k in ("a_out_phrase_level", "b_in_phrase_level") if 0 < int(plan.get(k, 16) or 16) < 8)
    out["phrase4"] = float(p["w_phrase4"]) * n4
    # categorie di Butler (musicmix.dj_form): solo se le forme ci sono
    out.update(exit_in_build=0.0, exit_breakdown=0.0, entry_drop=0.0)
    if ctx is not None and ctx.get("form_available"):
        if ctx.get("a_form_before") == "build" and ctx.get("a_form_at") in ("build", "core"):
            out["exit_in_build"] = float(p["w_exit_in_build"])
        if ctx.get("a_exit_in_breakdown"):
            out["exit_breakdown"] = -float(p["w_exit_breakdown_bonus"])
        if ctx.get("b_entry_drop"):
            out["entry_drop"] = -float(p["w_entry_drop_bonus"])
    return out


def dense_overlap_cost(decision: Mapping[str, Any], ctx: Optional[Mapping[str, Any]],
                       params: Optional[Mapping[str, Any]] = None) -> float:
    """Costo di un blend con tessiture piene (Butler p. 244, p. 129); 0 per gli altri tipi o senza forme."""
    p = values(ORDER_PARAMS, params)
    if not p["w_dense_overlap"] or decision.get("type") != "blend" or ctx is None or not ctx.get("form_available"):
        return 0.0
    ol, bb = ctx.get("overlap_layers"), ctx.get("overlap_both_busy")
    extra = max(0.0, (ol or 0.0) - float(p["dense_ref_layers"])) / 2.0 + float(bb or 0.0)
    return float(p["w_dense_overlap"] * extra)


def candidate_cost(plan: Mapping[str, Any], decision: Mapping[str, Any],
                   params: Optional[Mapping[str, Any]] = None, ctx: Optional[Mapping[str, Any]] = None) -> float:
    """Costo di un candidato: qualità del cue + costo a priori del tipo (+ collocazione, profilo libri)."""
    p = values(ORDER_PARAMS, params)
    cue = max(0.0, p["cue_ref"] - float(plan["cue_score"])) / p["cue_ref"]
    place = sum(placement_cost(plan, ctx, params).values())
    dense = dense_overlap_cost(decision, ctx, params)
    full = 0.0
    if p["w_cut_full_both"] and decision.get("type") == "cut" and ctx is not None and ctx.get("a_full_before") and ctx.get("b_full_at"):
        full = float(p["w_cut_full_both"])  # [P] B&B cap. 11: continuous melody on both sides of a cut
    return float(p["w_cue"] * cue + p["w_type"] * type_cost(decision, p["type_cost_override"]) + place + dense + full)


#: tipi di transizione con sovrapposizione di A e B per piu' battute: con cue_windows="technique" restano nelle
#: finestre di default di dj_select (OUT >= 35 %, IN <= 50 % e 16 battute residue) [H]
OVERLAP_TYPES = ("blend",)


def candidates_for_pair(A: Mapping[str, Any], B: Mapping[str, Any], tree_params: Optional[Mapping[str, Any]] = None,
                        order_params: Optional[Mapping[str, Any]] = None, cue_windows: Optional[str] = None,
                        event_plans: bool = False) -> dict[str, Any]:
    """Tutti i piani di ``dj_select.plan_pair`` per A -> B con decisione dell'albero e costo.

    ``cue_windows`` (opzione, default None = "default", invariato): "technique" genera i cue con
    ``dj_select.CUE_WINDOWS['technique']`` (B può entrare fino al 75 % del brano) e scarta i candidati fuori
    dalle finestre di default la cui decisione è una sovrapposizione (:data:`OVERLAP_TYPES`): gli ingressi
    tardivi restano solo per cut, filter/echo out, filter verso il ritornello e backspin. [H] audit P0
    21/10/2026 (cue per ruolo e tecnica); nessuna varietà forzata: i costi non cambiano.

    ``event_plans`` (opzione, default False): ogni contesto riceve ``event_context`` e ogni candidato
    ``event_summary`` (:func:`musicmix.dj_events.arrival_summary`, solo struttura); costi e ordine invariati.
    """
    name = cue_windows or "default"
    if name not in djs.CUE_WINDOWS:
        raise KeyError(f"unknown cue_windows {cue_windows!r}")
    r = djs.plan_pair(A, B, top_k=10 ** 9, windows=None if name == "default" else djs.CUE_WINDOWS[name])
    pair = r["compatibility"]
    cands, dropped = [], 0
    for plan in r["plans"]:
        ctx = transition_context(A, B, plan, pair, events=event_plans)
        dec = decide_transition(ctx, tree_params)
        if name != "default" and not plan.get("windows_default_ok", True) and dec["type"] in OVERLAP_TYPES:
            dropped += 1
            continue
        c = {"plan": plan, "context": ctx, "decision": dec, "cost": candidate_cost(plan, dec, order_params, ctx),
             "admissible": True, "notes": []}
        if event_plans:
            from musicmix import dj_events as dje

            c["event_summary"] = dje.arrival_summary(A, B, c)
        if name != "default":
            c["cue_windows"] = "default" if plan.get("windows_default_ok", True) else name
        cands.append(c)
    # on equal cost and score a cue of the default windows comes first (the extra cues never win a tie by being earlier)
    cands.sort(key=lambda c: (c["cost"], -c["plan"]["score"], c.get("cue_windows", "default") != "default",
                              c["plan"]["a_out_s"], c["plan"]["b_in_s"]))
    out = {"a": A["slug"], "b": B["slug"], "compatibility": pair, "candidates": cands, "cue_windows": name}
    if name != "default":
        out["dropped_overlap_outside_default"] = dropped
    return out


#: Ingresso di B (21/10/2026, conoscenza_mix/_meta/INGRESSO_B_2026-10-21.md): versione della guardia.
ENTRY_GUARD_VERSION = "musicmix.dj_set.entry_guard.1"


def entry_guard(pc: Mapping[str, Any], A: Mapping[str, Any], B: Mapping[str, Any], entry_test: Callable[..., Any],
                tree_params: Optional[Mapping[str, Any]] = None, order_params: Optional[Mapping[str, Any]] = None,
                exempt_types: Sequence[str] = OVERLAP_TYPES, relax_if_empty: bool = False) -> dict[str, Any]:
    """Guardia dell'ingresso di B sui candidati di :func:`candidates_for_pair` (in place, restituisce ``pc``).

    Per ogni candidato con un ingresso pulito di B (tutti i tipi tranne :data:`OVERLAP_TYPES`: nel blend B sale col
    fader e una parola all'inizio della dissolvenza non è troncata) ``entry_test(cue)`` (es.
    :func:`musicmix.dj_phrase.entry_clear_test`) dice se B può entrare lì con la prima frase intera, eventualmente con
    un lead-in (``plan['b_lead_in']``). Se no, :func:`musicmix.dj_select.relocate_entry` cerca lo stesso cue 1-2 battute
    prima o un altro cue di B con la stessa uscita di A: il candidato viene sostituito (decisione e costo ricalcolati,
    nota) o, se nessun ingresso è pulito, diventa non ammissibile. Se nessun candidato resta ammissibile la coppia resta
    **senza candidati ammissibili** (costo d'arco infinito in :func:`edge_cost`: l'ordinatore la evita); nessun ripiego
    porta all'ascolto un ingresso che le analisi sconsigliano (orchestratore, 21/10/2026). ``relax_if_empty=True``
    (opzione esplicita, spenta) ripristina l'ammissibilità di prima con la nota ``entry_guard_relaxed``.
    """
    pair = pc["compatibility"]
    before = [c["admissible"] for c in pc["candidates"]]
    stats = {"clear": 0, "moved": 0, "inadmissible": 0, "exempt": 0, "version": ENTRY_GUARD_VERSION}
    seen = {(c["plan"]["a_out_bar"], c["plan"]["b_in_bar"], c["decision"]["type"]) for c in pc["candidates"]}
    for c in pc["candidates"]:
        if c["decision"]["type"] in exempt_types:
            stats["exempt"] += 1
            continue
        r = djs.relocate_entry(A, B, c["plan"], pair, entry_test)
        if r["status"] == "clear":
            c["plan"]["b_lead_in"] = _lead_in_summary(r["entry"])
            stats["clear"] += 1
            continue
        if r["status"] == "moved":
            plan = r["plan"]
            ctx = transition_context(A, B, plan, pair)
            dec = decide_transition(ctx, tree_params)
            key = (plan["a_out_bar"], plan["b_in_bar"], dec["type"])
            if key not in seen and dec["type"] not in exempt_types:
                seen.add(key)
                plan["b_lead_in"] = _lead_in_summary(r["entry"])
                c.update(plan=plan, context=ctx, decision=dec, cost=candidate_cost(plan, dec, order_params, ctx))
                c["notes"].append(f"ingresso di B spostato dalla battuta {r['from_bar']} alla {r['to_bar']} ({r['kind']}): "
                                  + "; ".join(r["tried"][0].get("reasons") or []))
                c["entry_guard"] = {"status": "moved", "from_bar": r["from_bar"], "to_bar": r["to_bar"], "kind": r["kind"]}
                stats["moved"] += 1
                continue
        c["admissible"] = False
        c["entry_guard"] = {"status": "inadmissible", "reasons": (r.get("entry") or {}).get("reasons")}
        c["notes"].append("ingresso di B a metà frase e nessun cue alternativo pulito (entry_guard)")
        stats["inadmissible"] += 1
    if not any(c["admissible"] for c in pc["candidates"]):
        stats["no_admissible_entry"] = True
    if relax_if_empty and not any(c["admissible"] for c in pc["candidates"]):
        for c, a in zip(pc["candidates"], before):
            c["admissible"] = a
        stats["relaxed"] = True
        pc.setdefault("notes", []).append("entry_guard_relaxed: nessun ingresso pulito di B, candidati lasciati ammissibili")
    pc["candidates"].sort(key=lambda c: (not c["admissible"], c["cost"], -c["plan"]["score"]))
    pc["entry_guard"] = stats
    return pc


def _lead_in_summary(e: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("version", "status", "lead_in_samples", "audible_sample", "downbeat_sample", "evidence", "lead_in_words", "first_word",
            "phrase_words", "acoustic", "reasons")
    return {k: e.get(k) for k in keys if k in e}


#: [H] durata minima di un blend tra coppie 'lontane' (default di ORDER_PARAMS['far_min_blend_s']). Huron 2006 p. 211 [E]
#: riguarda modulazioni a tonalità vicine (tre accordi, 'a few seconds'): non prova un minimo di 3 s per un blend DJ.
SCHEMA_SWITCH_S = 3.0


def far_timbre_calibration(timbre: np.ndarray, pct: Optional[float],
                           min_spread: Optional[float] = None) -> dict[str, Any]:
    """Soglia 'coppia lontana' (percentile ``pct`` delle distanze fuori diagonale) con il suo stato.

    ``status``: ``"off"`` (``pct`` None), ``"no_pairs"``, ``"degenerate"`` (IQR delle distanze < ``min_spread``
    o soglia <= 0: il percentile non separa nulla, per esempio una matrice tutta 0 -> ci si astiene) o ``"ok"``.
    ``threshold`` è None salvo con ``"ok"``; ``median`` serve ad :func:`apply_far_genre` (una coppia non può
    essere 'lontana' se non supera nemmeno la mediana del pool). [H] ipotesi ingegneristica, vedi
    ORDER_PARAMS['far_timbre_pct']."""
    spread_min = float(ORDER_PARAMS["far_timbre_min_spread"].value if min_spread is None else min_spread)
    if pct is None:
        return {"status": "off", "threshold": None, "pct": None}
    D = np.asarray(timbre, dtype=float)
    off = D[~np.eye(len(D), dtype=bool)] if D.ndim == 2 and len(D) > 1 else np.zeros(0)
    if not off.size:
        return {"status": "no_pairs", "threshold": None, "pct": float(pct)}
    thr = float(np.percentile(off, float(pct)))
    q25, med, q75 = (float(x) for x in np.percentile(off, [25, 50, 75]))
    status = "ok" if (q75 - q25) >= spread_min and thr > 0 else "degenerate"
    return {"status": status, "threshold": thr if status == "ok" else None, "raw_threshold": thr, "pct": float(pct),
            "median": med, "iqr": q75 - q25, "min_spread": spread_min, "n_pairs": int(off.size)}


def far_timbre_threshold(timbre: np.ndarray, pct: Optional[float]) -> Optional[float]:
    """Distanza timbrica oltre la quale una coppia è 'lontana' (percentile ``pct`` delle coppie del pool); None se spento
    o se la matrice è degenere (:func:`far_timbre_calibration`: astensione invece di una soglia 0)."""
    return far_timbre_calibration(timbre, pct)["threshold"]


def apply_far_genre(pc: Mapping[str, Any], timbre_d: float, threshold: Optional[float | Mapping[str, Any]], bpm_a: float, *,
                    min_blend_s: Optional[float] = SCHEMA_SWITCH_S, echo_tail_stop: bool = True) -> dict[str, Any]:
    """Proposta B1, **ipotesi [H]** (Huron 2006 pp. 207-211 è contesto, non prova la soglia né i 3 s): per una coppia
    timbricamente lontana niente sovrapposizione breve e ambigua: un blend più corto di ``min_blend_s`` diventa non
    ammissibile (None = regola spenta) e, con ``echo_tail_stop``, l'echo out ferma la sua coda sul downbeat di B
    (``render['tail_stop']``, raccordo netto). Cut, filter e backspin sono già netti.

    ``threshold``: float (soglia) o il dizionario di :func:`far_timbre_calibration`. 'Lontana' = ``timbre_d`` >=
    soglia **e** soglia > 0 (una soglia <= 0 viene da una matrice degenere: astensione) **e**, con la calibrazione,
    ``timbre_d`` > mediana del pool. Restituisce una copia di ``pc``; soglia None = nessun cambiamento."""
    import copy

    out = copy.deepcopy(dict(pc))
    cal = dict(threshold) if isinstance(threshold, Mapping) else {"threshold": threshold}
    thr, med = cal.get("threshold"), cal.get("median")
    far = (thr is not None and float(thr) > 0 and float(timbre_d) >= float(thr)
           and (med is None or float(timbre_d) > float(med)))
    status = cal.get("status") or ("off" if thr is None else ("degenerate" if float(thr) <= 0 else "ok"))
    out["far_genre"] = {"far": bool(far), "timbre_d": float(timbre_d), "threshold": thr, "status": status,
                        "evidence": "[H] ipotesi ingegneristica (B1); Huron pp. 207-213 non prova 3 s né il 90° percentile",
                        "min_blend_s": min_blend_s, "echo_tail_stop": bool(echo_tail_stop)}
    if not far:
        return out
    bar_s = 240.0 / float(bpm_a)
    for c in out["candidates"]:
        d = c["decision"]
        if d["type"] == "blend" and min_blend_s is not None and d["bars"] * bar_s < float(min_blend_s):
            c["admissible"] = False
            c.setdefault("notes", []).append(f"generi lontani [H]: blend di {d['bars'] * bar_s:.1f} s < {float(min_blend_s):g} s")
        elif d["type"] == "echo_out" and echo_tail_stop:
            d["render"] = dict(d["render"], tail_stop=True)
            c.setdefault("notes", []).append("generi lontani [H]: coda d'eco fermata sul downbeat di B (raccordo netto)")
    out["candidates"].sort(key=lambda c: (not c["admissible"], c["cost"], -c["plan"]["score"]))
    return out


def edge_cost(pc: Mapping[str, Any], bpm_a: float, bpm_b: float, timbre_d: float, e_a: float, e_b: float,
              params: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Costo dell'arco A -> B: tempo + tonalità + timbro + miglior candidato + salto di energia."""
    p = values(ORDER_PARAMS, params)
    adm = [c for c in pc["candidates"] if c["admissible"]]
    tc = tempo_cost(bpm_a, bpm_b, params)
    comps: dict[str, Any] = {"tempo": tc, "stretch_pct": 100 * folded_tempo(bpm_a, bpm_b)["stretch"]}
    if not adm or not math.isfinite(tc):
        return {**comps, "total": math.inf, "feasible": False,
                "why": "nessun cue ammissibile" if not adm else f"salto di tempo > {100 * p['tempo_hard_max']:.0f}%"}
    best = adm[0]
    rel = pc["compatibility"]["key"]["relation"]
    kc = p["key_cost"][rel] * (p["w_key_blend"] if best["decision"]["type"] == "blend" else p["w_key_soft"])
    comps.update(key=kc, key_relation=rel, timbre=p["w_timbre"] * timbre_d, candidate=best["cost"],
                 smooth=p["w_smooth"] * abs(e_b - e_a), best_type=best["decision"]["type_key"],
                 best_a_out_bar=best["plan"]["a_out_bar"], best_b_in_bar=best["plan"]["b_in_bar"])
    comps["tempo_down"] = tempo_down_cost(bpm_a, bpm_b, params)
    total = p["w_tempo"] * tc + kc + comps["timbre"] + comps["candidate"] + comps["smooth"] + comps["tempo_down"]
    return {**comps, "total": float(total), "feasible": True}


def tempo_down_cost(bpm_a: float, bpm_b: float, params: Optional[Mapping[str, Any]] = None) -> float:
    """Costo di un calo di tempo A -> B (ripiegato d'ottava): ``w_tempo_down`` * calo / 8%.

    [A] Butler 2006 pp. 250-252 (accelerazione lieve nel set); [P] B&B cap. 21 (warm-up). 0 di default.
    """
    p = values(ORDER_PARAMS, params)
    if not p["w_tempo_down"]:
        return 0.0
    r = math.log2(bpm_b / bpm_a)
    r -= round(r)
    return float(p["w_tempo_down"] * max(0.0, -r) / math.log2(1.0 + p["tempo_ref"]))


# --------------------------------------------------------------- Held–Karp
def held_karp_path(edge: np.ndarray, node: np.ndarray, length: int, start: Optional[int] = None,
                   end: Optional[int] = None) -> tuple[float, list[int]]:
    """Cammino hamiltoniano minimo su un sottoinsieme di ``length`` nodi (esatto).

    ``edge[i, j]``: costo dell'arco i -> j (``inf`` = vietato); ``node[pos, j]``:
    costo del nodo j alla posizione ``pos`` (0..length-1). Programmazione
    dinamica di Held–Karp sugli insiemi di nodi visitati (stato = insieme,
    ultimo nodo; la posizione è la cardinalità dell'insieme). Restituisce
    ``(costo, cammino)``; ``(inf, [])`` se nessun cammino è ammissibile.
    """
    E = np.asarray(edge, dtype=float)
    n = E.shape[0]
    N = np.asarray(node, dtype=float)
    if not 1 <= length <= n:
        raise ValueError("length must be in 1..n")
    if N.shape != (length, n):
        raise ValueError("node must have shape (length, n)")
    if n > 20:
        raise ValueError("held_karp_path: n > 20 not supported (memory)")
    full = 1 << n
    dp = np.full((full, n), np.inf)
    parent = np.full((full, n), -1, dtype=np.int8)
    for j in range(n):
        if start is None or j == start:
            dp[1 << j, j] = N[0, j]
    masks = np.arange(full)
    pop = np.zeros(full, dtype=np.int64)
    for j in range(n):
        pop += (masks >> j) & 1
    for k in range(1, length):
        level = masks[pop == k]
        sub = dp[level]  # (M, n)
        if not np.isfinite(sub).any():
            break
        for j in range(n):
            if end is not None and k == length - 1 and j != end:
                continue
            if start is not None and j == start:
                continue
            free = level[(level >> j) & 1 == 0]
            if free.size == 0:
                continue
            cand = dp[free] + E[:, j][None, :]  # (M', n) over last node
            last = np.argmin(cand, axis=1)
            val = cand[np.arange(free.size), last] + N[k, j]
            nm = free | (1 << j)
            better = val < dp[nm, j]
            dp[nm[better], j] = val[better]
            parent[nm[better], j] = last[better]
    final = masks[pop == length]
    if final.size == 0:
        return math.inf, []
    vals = dp[final]
    if end is not None:
        mask_vals = np.full_like(vals, np.inf)
        mask_vals[:, end] = vals[:, end]
        vals = mask_vals
    flat = int(np.argmin(vals))
    mi, j = divmod(flat, n)
    best = float(vals[mi, j])
    if not math.isfinite(best):
        return math.inf, []
    mask = int(final[mi])
    path = [j]
    while True:
        pj = int(parent[mask, j])
        if pj < 0:
            break
        mask ^= 1 << j
        j = pj
        path.append(j)
    path.reverse()
    return best, path


def preselect(slugs: Sequence[str], edge: np.ndarray, keep: int, k_best: int = 3) -> list[int]:
    """Indici dei ``keep`` brani con gli archi migliori (media dei ``k_best`` archi
    finiti più economici, entranti e uscenti) [H]: usato solo se i candidati
    superano ``max_exact``."""
    n = len(slugs)
    score = []
    for i in range(n):
        vals = np.concatenate([edge[i, :], edge[:, i]])
        vals = np.sort(vals[np.isfinite(vals)])
        score.append(float(vals[:k_best].mean()) if vals.size else math.inf)
    order = sorted(range(n), key=lambda i: (score[i], slugs[i]))
    return sorted(order[:keep])


def order_set(slugs: Sequence[str], edge: np.ndarray, energy: Mapping[str, float],
              params: Optional[Mapping[str, Any]] = None, start: Optional[str] = None,
              end: Optional[str] = None) -> dict[str, Any]:
    """Ordine di L brani (len_min..len_max) a costo minimo; L = il più vicino a
    ``len_target`` tra quelli ammissibili (se nessuno da len_min in su, il più
    lungo possibile). Restituisce l'ordine scelto e il costo per ogni L."""
    p = values(ORDER_PARAMS, params)
    slugs = list(slugs)
    E = np.asarray(edge, dtype=float).copy()
    np.fill_diagonal(E, np.inf)
    idx = list(range(len(slugs)))
    pre = None
    if len(slugs) > p["max_exact"]:
        idx = preselect(slugs, E, p["max_exact"])
        for must in (start, end):
            if must is not None and slugs.index(must) not in idx:
                idx[-1] = slugs.index(must)
                idx = sorted(set(idx))
        pre = [slugs[i] for i in idx]
    sub = [slugs[i] for i in idx]
    Es = E[np.ix_(idx, idx)]
    n = len(sub)
    e = np.array([energy[s] for s in sub])
    s_i = sub.index(start) if start is not None else None
    e_i = sub.index(end) if end is not None else None
    results = {}
    for L in range(1, min(n, p["len_max"]) + 1):
        node = np.array([[p["w_arc"] * (e[j] - arc_target(pos, L, params)) ** 2 for j in range(n)] for pos in range(L)])
        cost, path = held_karp_path(Es, node, L, s_i, e_i)
        if math.isfinite(cost):
            results[L] = {"cost": cost, "order": [sub[j] for j in path], "cost_per_transition": cost / max(L - 1, 1)}
    feasible = sorted(results)
    if not feasible:
        return {"order": [], "length": 0, "cost": math.inf, "by_length": {}, "preselected": pre,
                "note": "nessun cammino ammissibile"}
    in_range = [L for L in feasible if L >= p["len_min"]]
    if in_range:
        L = min(in_range, key=lambda x: (abs(x - p["len_target"]), -x))
        note = f"lunghezza {L} (obiettivo {p['len_target']}, ammissibili {in_range[0]}..{in_range[-1]})"
    else:
        L = feasible[-1]
        note = f"solo {L} brani concatenabili (< {p['len_min']}): set più lungo possibile"
    return {"order": results[L]["order"], "length": L, "cost": results[L]["cost"],
            "by_length": {str(k): v for k, v in results.items()}, "preselected": pre, "note": note}


def path_breakdown(order: Sequence[str], edges: Mapping[tuple[str, str], Mapping[str, Any]],
                   energy: Mapping[str, float], params: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Componenti del costo di un ordine (per il JSON delle motivazioni)."""
    p = values(ORDER_PARAMS, params)
    L = len(order)
    nodes = [{"slug": s, "position": i, "energy": energy[s], "target": arc_target(i, L, params),
              "arc_cost": p["w_arc"] * (energy[s] - arc_target(i, L, params)) ** 2} for i, s in enumerate(order)]
    arcs = [dict(edges[(a, b)], a=a, b=b) for a, b in zip(order, order[1:])]
    total = sum(x["arc_cost"] for x in nodes) + sum(x["total"] for x in arcs)
    return {"nodes": nodes, "edges": arcs, "total": total}


# ----------------------------------------------------------- cues (Viterbi)
def overlap_b_s(cand: Mapping[str, Any], B: Mapping[str, Any]) -> float:
    """Secondi di B (sorgente) suonati dentro la transizione (blend: N battute)."""
    d = cand["decision"]
    if d["type"] != "blend":
        return 0.0
    return float(d["bars"] * djs.BEATS_PER_BAR * B["tempo"]["period_s"])


def lead_a_s(cand: Mapping[str, Any], A: Mapping[str, Any]) -> float:
    """Secondi di A prima di ``a_out`` in cui la transizione è già iniziata."""
    d = cand["decision"]
    bar = djs.BEATS_PER_BAR * A["tempo"]["period_s"]
    if d["type"] in ("filter_out", "filter_into_chorus"):
        return float(d["render"]["sweep_bars"] * bar)
    if d["type"] == "echo_out":
        return float(d["render"]["entry_bars"] * bar)
    if d["type"] == "backspin":
        return float(d["render"]["spin_beats"] * A["tempo"]["period_s"])
    return 0.0


def tail_a_s(cand: Mapping[str, Any], A: Mapping[str, Any]) -> float:
    """Secondi di A suonati dopo ``a_out`` (blend: N battute)."""
    d = cand["decision"]
    if d["type"] != "blend":
        return 0.0
    return float(d["bars"] * djs.BEATS_PER_BAR * A["tempo"]["period_s"])


# ------------------------------------------------- whole sections per track
SECTION_REQUIREMENTS = ("full", "one", "none")


def section_spans(track: Mapping[str, Any]) -> list[tuple[int, int, str]]:
    """Sezioni del brano come ``(battuta_inizio, battuta_fine_esclusa, etichetta)`` dalla tabella delle frasi."""
    rows = track["phrases"]
    n = len(rows)
    if n == 0:
        return []
    starts = [int(r["bar_index"]) for r in rows if r["section_start"]]
    if not starts or starts[0] != int(rows[0]["bar_index"]):
        starts = [int(rows[0]["bar_index"])] + starts
    end = int(rows[-1]["bar_index"]) + 1
    lab = {int(r["bar_index"]): r["section_label"] for r in rows}
    return [(a, b, lab.get(a, "unlabeled")) for a, b in zip(starts, starts[1:] + [end]) if b > a]


def sections_played(track: Mapping[str, Any], start_bar: int, end_bar: int, cp: Mapping[str, Any]) -> dict[str, Any]:
    """Sezioni **complete** suonate tra ``start_bar`` (ingresso) e ``end_bar`` (uscita, esclusa)."""
    core = tuple(cp["core_labels"])
    done = [(a, b, l) for a, b, l in section_spans(track) if a >= start_bar and b <= end_bar]
    labels = [l for _, _, l in done]
    return {"complete": labels, "n_core": sum(1 for l in labels if l in core),
            "has_verse": "verse" in labels, "has_chorus": "chorus" in labels,
            "bars": [int(start_bar), int(end_bar)]}


def _track_has(track: Mapping[str, Any], label: str) -> bool:
    return any(l == label for _, _, l in section_spans(track))


def section_ok(track: Mapping[str, Any], played: Mapping[str, Any], req: str, cp: Mapping[str, Any]) -> bool:
    """Requisito di sezioni intere: 'full' = una strofa e un ritornello completi (se il brano li ha) e almeno una
    sezione core; 'one' = almeno una sezione core completa; 'none' = nessun requisito."""
    if req == "none":
        return True
    has_core = any(l in tuple(cp["core_labels"]) for _, _, l in section_spans(track))
    if not has_core:
        return True
    if played["n_core"] < 1:
        return False
    if req == "one":
        return True
    if req != "full":
        raise ValueError(req)
    return ((played["has_verse"] or not _track_has(track, "verse"))
            and (played["has_chorus"] or not _track_has(track, "chorus")))


def section_cost(played: Mapping[str, Any], cp: Mapping[str, Any]) -> float:
    t = float(cp["target_core_sections"])
    return float(cp["w_sections"] * ((played["n_core"] - t) / t) ** 2)


def _section_matrix(track: Mapping[str, Any], jb: np.ndarray, ja: np.ndarray, req: str,
                    cp: Mapping[str, Any]) -> np.ndarray:
    """Costo di sezione (inf se il requisito fallisce) per ogni (ingresso jb[i], uscita ja[j]) dello stesso brano."""
    spans = section_spans(track)
    core = tuple(cp["core_labels"])
    jb = np.asarray(jb, dtype=int)[:, None]
    ja = np.asarray(ja, dtype=int)[None, :]
    n_core = np.zeros((jb.shape[0], ja.shape[1]))
    has_v = np.zeros_like(n_core, dtype=bool)
    has_c = np.zeros_like(n_core, dtype=bool)
    for a, b, l in spans:
        m = (a >= jb) & (b <= ja)
        if l in core:
            n_core += m
        if l == "verse":
            has_v |= m
        if l == "chorus":
            has_c |= m
    t = float(cp["target_core_sections"])
    cost = cp["w_sections"] * ((n_core - t) / t) ** 2
    if req == "none" or not any(l in core for _, _, l in spans):
        return cost
    ok = n_core >= 1
    if req == "full":
        if any(l == "verse" for _, _, l in spans):
            ok &= has_v
        if any(l == "chorus" for _, _, l in spans):
            ok &= has_c
    return np.where(ok, cost, np.inf)


def _bar_at(track: Mapping[str, Any], t_s: float) -> int:
    rows = track["phrases"]
    if not rows:
        return 0
    k = int(np.argmin([abs(float(r["time_s"]) - t_s) for r in rows]))
    return int(rows[k]["bar_index"])


def first_start(A: Mapping[str, Any], a_out_s: float, cp: Mapping[str, Any], a_out_bar: Optional[int] = None,
                req: str = "none") -> dict[str, Any]:
    """Inizio del primo brano: dall'inizio se la durata rientra, altrimenti un
    downbeat di inizio frase (>= 8) con durata più vicina all'obiettivo [H].

    In modo 'sections' (profilo libri) l'inizio è l'inizio del brano (intro:
    Butler pp. 247-249, B&B cap. 21) o un inizio di sezione, scelto tra quelli
    che rispettano il requisito di sezioni intere ``req`` e i limiti in secondi.
    """
    if cp.get("dur_mode", "seconds") == "sections":
        jo = int(a_out_bar) if a_out_bar is not None else _bar_at(A, a_out_s)
        starts = [(0.0, None)] + [(float(r["time_s"]), int(r["bar_index"])) for r in A["phrases"]
                                   if r["section_start"] and r["time_s"] < a_out_s and int(r["bar_index"]) > 0]
        best = None
        for t0, bar in starts:
            d = a_out_s - t0
            pl = sections_played(A, 0 if bar is None else bar, jo, cp)
            if not (cp["play_min_s"] <= d <= cp["play_max_s"]) or not section_ok(A, pl, req, cp):
                continue
            c = section_cost(pl, cp) + cp["w_dur"] * ((d - cp["play_target_s"]) / cp["play_target_s"]) ** 2
            if best is None or c < best[0]:
                best = (c, t0, bar, d, pl)
        if best is not None:
            _, t0, bar, d, pl = best
            return {"start_s": t0, "bar": bar, "play_s": d, "from_track_start": bar is None, "sections": pl}
    if a_out_s <= cp["play_max_s"]:
        return {"start_s": 0.0, "bar": None, "play_s": a_out_s, "from_track_start": True}
    rows = [r for r in A["phrases"] if r["phrase_level"] >= 8 and r["time_s"] < a_out_s]
    best = min(rows, key=lambda r: abs((a_out_s - r["time_s"]) - cp["play_target_s"])) if rows else None
    if best is None:
        return {"start_s": 0.0, "bar": None, "play_s": a_out_s, "from_track_start": True}
    return {"start_s": float(best["time_s"]), "bar": int(best["bar_index"]), "play_s": a_out_s - float(best["time_s"]),
            "from_track_start": False}


def last_end(B: Mapping[str, Any], b_in_s: float, cp: Mapping[str, Any], b_in_bar: Optional[int] = None,
             req: str = "none") -> dict[str, Any]:
    """Fine dell'ultimo brano: fino alla fine se la durata rientra, altrimenti un
    downbeat di inizio frase (>= 8) con durata più vicina all'obiettivo [H].

    In modo 'sections' la fine è un confine di sezione (o la fine del brano) dopo
    sezioni intere che rispettano ``req`` [P] B&B cap. 13; Webber p. 225.
    """
    dur = float(B["duration_s"])
    if cp.get("dur_mode", "seconds") == "sections":
        jb = int(b_in_bar) if b_in_bar is not None else _bar_at(B, b_in_s)
        n_end = section_spans(B)[-1][1] if section_spans(B) else 0
        ends = [(float(r["time_s"]), int(r["bar_index"])) for r in B["phrases"]
                if r["section_start"] and r["time_s"] > b_in_s + 1.0] + [(dur, None)]
        best = None
        for t1, bar in ends:
            d = t1 - b_in_s
            pl = sections_played(B, jb, n_end if bar is None else bar, cp)
            if not (cp["play_min_s"] <= d <= cp["play_max_s"]) or not section_ok(B, pl, req, cp):
                continue
            c = section_cost(pl, cp) + cp["w_dur"] * ((d - cp["play_target_s"]) / cp["play_target_s"]) ** 2
            if best is None or c < best[0]:
                best = (c, t1, bar, d, pl)
        if best is not None:
            _, t1, bar, d, pl = best
            return {"end_s": t1, "bar": bar, "play_s": d, "to_track_end": bar is None, "sections": pl}
    if dur - b_in_s <= cp["play_max_s"]:
        return {"end_s": dur, "bar": None, "play_s": dur - b_in_s, "to_track_end": True}
    rows = [r for r in B["phrases"] if r["phrase_level"] >= 8 and r["time_s"] > b_in_s + 1.0]
    best = min(rows, key=lambda r: abs((r["time_s"] - b_in_s) - cp["play_target_s"])) if rows else None
    if best is None:
        return {"end_s": dur, "bar": None, "play_s": dur - b_in_s, "to_track_end": True}
    return {"end_s": float(best["time_s"]), "bar": int(best["bar_index"]), "play_s": float(best["time_s"]) - b_in_s,
            "to_track_end": False}


def _dur_cost(d: float, lo: float, hi: float, cp: Mapping[str, Any]) -> float:
    if d < lo or d > hi:
        return math.inf
    return float(cp["w_dur"] * ((d - cp["play_target_s"]) / cp["play_target_s"]) ** 2)


def _end_cost(fs_or_le: Mapping[str, Any], lo: float, hi: float, cp: Mapping[str, Any], req: str, track) -> float:
    c = _dur_cost(fs_or_le["play_s"], lo, hi, cp)
    if cp.get("dur_mode", "seconds") == "sections" and math.isfinite(c):
        pl = fs_or_le.get("sections")
        if pl is None:
            return math.inf if req != "none" else c
        if not section_ok(track, pl, req, cp):
            return math.inf
        c += section_cost(pl, cp)
    return c


def choose_cues(order: Sequence[str], analyses: Mapping[str, Mapping[str, Any]],
                pair_cands: Mapping[tuple[str, str], Mapping[str, Any]],
                params: Optional[Mapping[str, Any]] = None, max_per_transition: int = 400,
                event_fn: Optional[Callable[..., Mapping[str, Any]]] = None) -> dict[str, Any]:
    """Viterbi esatto sui candidati ammissibili di ogni transizione del set.

    Costo = somma dei costi dei candidati + durata di ogni brano nel set
    (vincolo duro ``play_min_s``..``play_max_s``, costo quadratico verso
    ``play_target_s``; primo e ultimo brano con :func:`first_start` /
    :func:`last_end`) + ``w_repeat`` per lo stesso tipo due volte di fila
    (+ ``w_fx_run`` se entrambi effetti marcati) + spazio minimo ``min_gap_s``
    tra la fine di una transizione e l'inizio della successiva. In modo
    ``dur_mode='sections'`` ogni brano deve suonare sezioni intere (requisito
    'full' = strofa e ritornello completi) con costo ``w_sections`` verso
    ``target_core_sections``. Se nessuna sequenza è ammissibile si rilassa prima
    il requisito di sezioni ('full' -> 'one' -> 'none'), poi i limiti di durata
    (``relax_steps`` volte, ×0,75 / ×1,33); tutto è registrato.

    Opzione sperimentale (spenta, ``w_event_arrival`` = 0): con un peso > 0 i candidati «preferiti» dal piano
    per eventi (A completato, obiettivo di B entro ``event_arrival_max_bars`` dallo switch) costano ``w`` in meno
    nel Viterbi. ``event_fn(A, B, cand) -> {"preferred": bool, ...}`` (default
    :func:`musicmix.dj_events.arrival_summary` sulla sola struttura; il runner può passare una funzione che usa
    ``dj_phrase``). Il riassunto è registrato in ``cand["event_pref"]`` solo con l'opzione accesa.
    """
    cp = values(CUE_PARAMS, params)
    if cp["dur_mode"] not in ("seconds", "sections"):
        raise ValueError(f"unknown dur_mode {cp['dur_mode']!r}")
    T = len(order) - 1
    if T < 1:
        raise ValueError("a set needs at least two tracks")
    lists = []
    for t in range(T):
        pc = pair_cands[(order[t], order[t + 1])]
        lists.append([c for c in pc["candidates"] if c["admissible"]][:max_per_transition])
        if not lists[-1]:
            raise ValueError(f"no admissible candidate for {order[t]} -> {order[t + 1]}")
    bonus = None
    pref_info = None
    w_ev = float(cp.get("w_event_arrival") or 0.0)
    if w_ev > 0:
        from musicmix import dj_events as dje

        fn = event_fn or (lambda A_, B_, c_: dje.arrival_summary(A_, B_, c_, params={"arrival_max_bars": cp["event_arrival_max_bars"]}))
        bonus, n_pref = [], []
        for t, lst in enumerate(lists):
            A_, B_ = analyses[order[t]], analyses[order[t + 1]]
            arr = np.zeros(len(lst))
            for k, c in enumerate(lst):
                s = dict(fn(A_, B_, c))
                c["event_pref"] = s
                if s.get("preferred"):
                    arr[k] = -w_ev
            bonus.append(arr)
            n_pref.append(int((arr < 0).sum()))
        pref_info = {"w_event_arrival": w_ev, "event_arrival_max_bars": cp["event_arrival_max_bars"],
                     "preferred_per_transition": n_pref, "label": "[H] sperimentale (P0.5)"}
    lo, hi = cp["play_min_s"], cp["play_max_s"]
    relax_log = []
    reqs = list(SECTION_REQUIREMENTS) if cp["dur_mode"] == "sections" else ["none"]
    attempts = [(r, lo, hi) for r in reqs]
    for step in range(1, int(cp["relax_steps"]) + 1):
        attempts.append(("none", lo * 0.75 ** step, hi * 1.33 ** step))
    for req, l0, h0 in attempts:
        res = _viterbi(order, analyses, lists, cp, l0, h0, req, bonus=bonus)
        if res is not None:
            if pref_info is not None:
                res["event_pref"] = dict(pref_info, chosen_preferred=[bool(c.get("event_pref", {}).get("preferred")) for c in res["chosen"]])
            res["play_bounds_s"] = [l0, h0]
            res["relaxations"] = relax_log
            res["section_requirement"] = req
            res["dur_mode"] = cp["dur_mode"]
            return res
        relax_log.append({"bounds_s": [l0, h0], "section_requirement": req, "result": "infeasible"})
    raise ValueError(f"no feasible cue sequence even with play bounds {attempts[-1][1]:.0f}-{attempts[-1][2]:.0f} s")


def _viterbi(order, analyses, lists, cp, lo, hi, req="none", bonus=None):
    T = len(lists)
    sections = cp.get("dur_mode", "seconds") == "sections"
    A0 = analyses[order[0]]
    # transition 0: first track duration
    node0 = []
    for k, c in enumerate(lists[0]):
        fs = first_start(A0, c["plan"]["a_out_s"], cp, c["plan"].get("a_out_bar"), req)
        node0.append(c["cost"] + (bonus[0][k] if bonus is not None else 0.0) + _end_cost(fs, lo, hi, cp, req, A0))
    score = np.array(node0)
    back: list[np.ndarray] = []
    fx = set(FX_MARKED)
    for t in range(1, T):
        prev, cur = lists[t - 1], lists[t]
        Bt = analyses[order[t]]  # track entered by t-1 and exited by t
        b_in = np.array([c["plan"]["b_in_s"] for c in prev])
        ovl = np.array([overlap_b_s(c, Bt) for c in prev])
        tprev = [c["decision"]["type"] for c in prev]
        a_out = np.array([c["plan"]["a_out_s"] for c in cur])
        lead = np.array([lead_a_s(c, Bt) for c in cur])
        tcur = [c["decision"]["type"] for c in cur]
        node = np.array([c["cost"] for c in cur])
        if bonus is not None:
            node = node + bonus[t]
        if t == T - 1:
            Bl = analyses[order[t + 1]]
            node = node + np.array([_end_cost(last_end(Bl, c["plan"]["b_in_s"], cp, c["plan"].get("b_in_bar"), req),
                                              lo, hi, cp, req, Bl) for c in cur])
        d = a_out[None, :] - b_in[:, None]
        dur = np.where((d >= lo) & (d <= hi), cp["w_dur"] * ((d - cp["play_target_s"]) / cp["play_target_s"]) ** 2, np.inf)
        if sections:
            jb = np.array([int(c["plan"]["b_in_bar"]) for c in prev])
            ja = np.array([int(c["plan"]["a_out_bar"]) for c in cur])
            dur = dur + _section_matrix(Bt, jb, ja, req, cp)
        gap = (a_out[None, :] - lead[None, :]) - (b_in[:, None] + ovl[:, None])
        dur = np.where(gap >= cp["min_gap_s"], dur, np.inf)
        rep = np.array([[(cp["w_repeat"] if x == y else 0.0) + (cp["w_fx_run"] if (x in fx and y in fx) else 0.0)
                         for y in tcur] for x in tprev])
        tot = score[:, None] + dur + rep
        arg = np.argmin(tot, axis=0)
        score = tot[arg, np.arange(len(cur))] + node
        back.append(arg)
    if T == 1:
        Bl = analyses[order[1]]
        score = score + np.array([_end_cost(last_end(Bl, c["plan"]["b_in_s"], cp, c["plan"].get("b_in_bar"), req),
                                            lo, hi, cp, req, Bl) for c in lists[0]])
    j = int(np.argmin(score))
    best = float(score[j])
    if not math.isfinite(best):
        return None
    picks = [j]
    for t in range(T - 1, 0, -1):
        j = int(back[t - 1][j])
        picks.append(j)
    picks.reverse()
    chosen = [lists[t][k] for t, k in enumerate(picks)]
    fs = first_start(analyses[order[0]], chosen[0]["plan"]["a_out_s"], cp, chosen[0]["plan"].get("a_out_bar"), req)
    le = last_end(analyses[order[-1]], chosen[-1]["plan"]["b_in_s"], cp, chosen[-1]["plan"].get("b_in_bar"), req)
    plays = [fs["play_s"]] + [chosen[t]["plan"]["a_out_s"] - chosen[t - 1]["plan"]["b_in_s"] for t in range(1, T)] + [le["play_s"]]
    played = [fs.get("sections")]
    for t in range(1, T):
        played.append(sections_played(analyses[order[t]], int(chosen[t - 1]["plan"]["b_in_bar"]),
                                      int(chosen[t]["plan"]["a_out_bar"]), cp))
    played.append(le.get("sections"))
    return {"cost": best, "chosen": chosen, "picks": picks, "first": fs, "last": le, "play_s": plays,
            "sections_played": played}


def alternatives(t: int, order: Sequence[str], analyses: Mapping[str, Mapping[str, Any]],
                 cands: Sequence[Mapping[str, Any]], prev_b_in_s: Optional[float], next_a_out: Optional[tuple[float, float]],
                 params: Optional[Mapping[str, Any]] = None, exclude: Iterable[int] = (), limit: int = 6) -> list[int]:
    """Indici dei candidati della transizione ``t`` che rispettano i vincoli con i
    vicini già fissati (ingresso precedente in A, uscita successiva di B con il
    suo anticipo ``lead``), ordinati per costo: alternative di cue per il QA."""
    cp = values(CUE_PARAMS, params)
    A, B = analyses[order[t]], analyses[order[t + 1]]
    ex = set(exclude)
    out = []
    for i, c in enumerate(cands):
        if i in ex or not c["admissible"]:
            continue
        a_out, b_in = c["plan"]["a_out_s"], c["plan"]["b_in_s"]
        if prev_b_in_s is not None:
            d = a_out - prev_b_in_s
            if not cp["play_min_s"] <= d <= cp["play_max_s"]:
                continue
        if next_a_out is not None:
            nxt, nlead = next_a_out
            d = nxt - b_in
            if not cp["play_min_s"] <= d <= cp["play_max_s"]:
                continue
            if (nxt - nlead) - (b_in + overlap_b_s(c, B)) < cp["min_gap_s"]:
                continue
        out.append(i)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------- clip geometry
def clip_geometry(meta: Mapping[str, Any], n: int) -> dict[str, Any]:
    """Regioni utili per giuntare una clip di dj_render/dj_effects (campioni della clip).

    * ``a_offset``: campione sorgente di A = campione clip + ``a_offset``, valido
      in ``[0, a_pure_end)`` (A solo, non filtrato: prima dell'inizio della transizione);
    * ``b_pure_start``: da qui in poi suona solo B (dopo la fine della
      transizione e dell'eventuale coda d'eco);
    * ``b_rate``: secondi sorgente di B per secondo d'uscita dopo la
      transizione (1 senza stretch); ``b_anchor``: (campione clip, campione
      sorgente B) da cui vale la retta.
    """
    ov, sw = meta["overlap"], meta.get("switch", {})
    a_off = int(ov["a_source_start_sample"]) - int(ov["output_start_sample"])
    start, end = int(ov["output_start_sample"]), int(ov["output_end_sample"])
    tail_end = end
    eff = meta.get("effect") or {}
    if eff.get("type") == "echo_out" and eff.get("repeats"):
        last = eff["repeats"][-1]
        tail_end = max(tail_end, int(last["output_sample"]) + int(eff.get("delay_samples", 0)))
    st = meta.get("stretch") or {}
    if st.get("applied"):
        rate = float(st["tempo_ratio_b"])
        anchor = (end, int(ov["b_source_end_sample"]))
    else:
        rate = 1.0
        anchor = (int(sw["output_sample"]), int(sw["b_source_sample"]))
    return {"a_offset": a_off, "a_pure_end": start, "transition_start": start, "switch": int(sw.get("output_sample", start)),
            "transition_end": end, "b_pure_start": min(n, tail_end), "b_rate": rate, "b_anchor": anchor, "n": n}


def clip_pos_of_b(geo: Mapping[str, Any], b_src_sample: float) -> float:
    c0, s0 = geo["b_anchor"]
    return c0 + (b_src_sample - s0) / geo["b_rate"]


def b_src_of_clip(geo: Mapping[str, Any], clip_sample: float) -> float:
    c0, s0 = geo["b_anchor"]
    return s0 + (clip_sample - c0) * geo["b_rate"]


__all__ = [n for n in dir() if not n.startswith("_") and n not in (
    "annotations", "dataclass", "math", "np", "Any", "Callable", "Iterable", "Mapping", "Optional", "Sequence", "djs")]
