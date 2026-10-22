# P0 "misure e regole" dopo l'audit dei nove libri (21/10/2026)

Tranche del backlog dell'audit ([`libri/audit_2026-10-21/BACKLOG.json`](../libri/audit_2026-10-21/BACKLOG.json)): P0.2
(misure no-core e onset con confini incerti), P0.3 (separare evidenza e ipotesi), P0.4 (cue per ruolo e tecnica), spec di
P0.5 (piano per eventi). Render, QA, effetti e critico (P0.1, P0.6) sono di un'altra tranche. Stato per item:
[`STATO_BACKLOG_CLOUD.md`](../libri/audit_2026-10-21/STATO_BACKLOG_CLOUD.md). Misure: `runs/p0_misure_regole/`
(`tools/p0_measure_rules.py`, solo lettura dei run congelati). Nessun ascolto: tutto ciò che segue è integrità delle misure e
tracciabilità delle fonti, non qualità musicale.

Legenda dei livelli (AGENTS.md): [E] risultato sperimentale, [T] teoria, [A] analisi/algoritmo di un autore, [P] pratica
(manuali, interviste), [H] ipotesi nostra. Pagine **stampate**, rilette sul testo per pagina (`tools/libri.py`; scarto
fisica−stampata: Ohriner +42, Huron +15, Margulis +15, Butler +14, Krims +14, Katz +17, Webber +15, Williams +24; B&B è un
EPUB, citato per capitolo). Nel repo solo citazioni brevi.

## 1. Tre controesempi (ROOT_COUNTEREXAMPLES.json): corretti, con test

| Controesempio | Prima | Ora | Test | Effetto misurato su dati reali |
|---|---|---|---|---|
| `dj_form.form_segments` senza core/cassa | 8 battute `outro` | `unknown` + `core_status = "no_core"`, `no_core_reason` (`no_kick` / `kick_without_full_texture`); intro/build/outro solo rispetto a un core; `dj_form.2` | `test_dj_form.py::test_no_core_is_unknown_not_outro`, `..._with_kick_but_thin_texture`, `..._core_found_keeps_the_previous_labels` | 1 forma su 36 in cache (`gods_plan`, set v2/v3): 58 battute tutte `outro`, cassa sopra la soglia 0,30 in 3 battute isolate (tolte dalla maggioranza mobile; 808 o cassa debole) → ora `no_core`/`no_kick`. Nessuna scelta cambia: le etichette `outro` non entravano nei costi; `dj_set.form_context` ora dà `a_full_before`/`b_full_at` falsi senza core (peso `w_cut_full_both` = 0) |
| `dj_phrase.with_stem_onsets`: parola 900–1000 ms, onset 1040 ms | onset assegnato alla parola, snapped 100 % | finestra `[on − 60 ms, off)`: l'onset oltre la fine non è mai assegnato; la sillaba resta stimata e diventa `uncertain_beyond_word`, non conta come snapped (`beyond_word="legacy"` riproduce il vecchio comportamento) | `test_dj_phrase_books.py::test_stem_onset_past_word_end_is_not_snapped`, `::test_onset_past_word_goes_to_the_next_word` | 36 brani di `runs/dj_phrase_v1` (`runs/p0_misure_regole/onsets.json`): 82 sillabe su 16.179 (0,5 %) erano agganciate oltre la fine della parola; ora 89 incerte, 165 sillabe spostate (l'onset passa alla parola successiva); snapped totali 11.896 → 11.898, mediana 0,741 invariata; fase vocale mediana +0,033 battiti invariata. La conclusione di F5 non cambia (limite = segmentazione sillabica) |
| `dj_set.far_timbre_threshold` / `apply_far_genre` con matrice tutta 0 | soglia 0, coppia a distanza 0 «lontana», coda d'eco fermata | `far_timbre_calibration`: stato `degenerate` (IQR < `far_timbre_min_spread` o soglia ≤ 0) → nessuna soglia, astensione; una coppia è lontana solo se ≥ soglia, soglia > 0 e sopra la mediana del pool; opzioni `far_min_blend_s` (3 s, None spegne) e `far_echo_tail_stop` (True) | `test_p0_misure_regole.py::test_identical_timbre_matrix_abstains_instead_of_far`, `::test_mostly_identical_pool_needs_more_than_the_median`, `::test_far_genre_regular_pool_unchanged_and_options_switch_off` | Pool reali: invariato (il 90° percentile è > 0 e > mediana; le decisioni con soglia float e con calibrazione coincidono, test) |

`tools/book_pipeline_probes.py` ora stampa per ogni caso l'esito attuale, il comportamento precedente quando è ancora
riproducibile e `fixed: true` (test `test_book_pipeline_probes_report_the_fixes`).

## 2. Evidenza e ipotesi operative (P0.3): verifiche nei libri

| Affermazione della revisione | Verifica nel testo | Esito | Cambiamento |
|---|---|---|---|
| Margulis 1/4/8 (p. 8) sono secondi, non copie | Fig. 1.2, p. 8: «repeating unit lengths of 1, 4, and 8 s» | **confermata** | la nostra scheda (cap. 1) era già corretta; nel codice nessuna regola usava 1/4/8 come copie. Nota esplicita in `BOOK_RULES['loop_focal_repeats']` e `BOOK_RULES_EVIDENCE` |
| Lidov (Margulis p. 51) dà categorie, non un massimo | p. 51: formativa/focale/testurale, «three-or-fourfold», «beyond the three-or-four mark»; «a single repetition is enough to establish a pattern» | **confermata** | `loop_focal_repeats`, `loop_establish_repeats`: fonte [T], soglia [H]. Annotata la scheda Margulis cap. 2 |
| «Oltre 2 copie vocali» non è una fonte | Margulis p. 16: speech-to-song con ripetizione «excessive and temporally regular», ~85 % degli ascoltatori; nessun numero | **confermata** | `loop_vocal_max_repeats` ora `[H]`, solo diagnostico; `loop_repetition_check` restituisce `findings` con livello, `diagnostic_only`, `include_hypotheses=False` toglie i limiti senza numero nelle fonti; contratto `ok` invariato per i chiamanti (`dj_creative.halving_build`) |
| Huron pp. 207–213 non prova 3 s né il 90° percentile | p. 207–208: stile in 250 ms, quasi al massimo a 1 s (timbro); p. 211: tonalità **vicine** «firmly established» entro tre accordi, «a few seconds»; schemi lontani: «I doubt that the switch would take more than a few seconds» (congettura); p. 213: «more research is needed» | **confermata** | `far_timbre_pct`, `far_min_blend_s`, `far_echo_tail_stop`, `schema_switch_seconds`: [H] con il contesto [E]/[T] nella fonte; opzioni per spegnerli; `PROFILE_BOOKS` annotato |
| Krims p. 32: «musical closure» = chiusura del discorso interpretativo | p. 32: «the problem of musical closure ... the only commonly accepted references of musical features are ... other musical features»; «closure of interpretive systems» | **confermata** | `krims_closure` di `exit_point` non usa p. 32 (fonti: p. 133 e p. 194, un brano ciascuna, rilette) ed è **già** fuori dal punteggio: resta diagnostica, con `KRIMS_CLOSURE_NOTE` e commento nel codice; nessuna scheda la usava come regola di cue (la scheda Krims cap. 1 la descriveva già come problema metodologico) |
| Williams p. 10: frammenti stonati/irregolari come segnale di citazione | p. 10: il prestito si segnala se il frammento «doesn't quite fit», «slightly out of tune» (i «de-tuned layers» di Krims) o di durata non 4/8 battute | **confermata** | annotato il limite dove premiamo l'uniformità: `BOOK_RULES['four_bar_groups']` (avviso «non multiplo di 4» = diagnostico, può essere voluto), `PROFILE_BOOKS_EVIDENCE['tree.key_clash_cut']` |
| Huron p. 286 («disco rotto», norma di tre) | p. 286: una registrazione dal vivo di Schickele, risate alla 4a ripetizione, «consistent with a norm of three repetitions»; Margulis pp. 173–174 (Johnstone) | **parziale**: è un'osservazione su un brano comico, non un esperimento | `loop_comic_repeats`: [E] → [A] fonte, [H] soglia |

**Livelli per ogni regola.** Nuove tabelle con `source_level`, `rule_level`, pagine verificate e nota:
`dj_phrase.BOOK_RULES_EVIDENCE` (23 regole), `dj_set.PROFILE_BOOKS_EVIDENCE` (20 voci di tree/order/cues),
`dj_policy.PROFILE_BOOKS_EVIDENCE` (7 voci). Test: `test_book_rules_have_verified_evidence_levels`,
`test_profile_books_have_verified_evidence_levels` (ogni chiave coperta, livelli validi). Tutti i pesi e le soglie dei profili
sono [H]; le fonti sono per lo più [P]/[A], qualche [E] di altri domini.

**Pagine di Butler corrette.** Rileggendo le frasi citate, diverse pagine di Butler 2006 nel codice e nelle schede erano una
pagina prima di quella stampata. Corrette nei commenti/fonti di `dj_set`, `dj_form`, `dj_policy` (annotando «corretto: era
p. X»), segnalate nelle schede:

| Citazione | Pagina giusta | Prima |
|---|---|---|
| breakdown «typically lasting for at least four measures» | pp. 91–92 | p. 91 |
| cambi che durano «at a minimum, four measures» | p. 189 | p. 188 |
| «sudden, dramatic drop to a thin texture» | p. 224 | p. 223 |
| «one does not have a core followed by a buildup» | p. 225 | pp. 222–224 |
| DJ Shiva: «that's usually where you mix out or you cut» | p. 223 | pp. 221–222 |
| Adam Jay: «to make room in the mix» | p. 232 | p. 231 |
| Mystik: «the builds won't be the same» | p. 243 | p. 242 |
| due voci insieme | p. 244 | p. 243 |
| «if it's too busy» | p. 245 | p. 244 |
| dropping the beat / «turn it off and drop the beat ... it's clean» | pp. 246–248 | pp. 245–247 |
| più climax, l'ultimo il più forte | p. 251 | p. 250 |
| sequenze di 8 e 16 battute | p. 206 | p. 205 |

Corrette o confermate anche: Butler p. 3 (Pullen toglie la cassa, il pubblico ne attende il ritorno), p. 129 («decreases the
likelihood of a clash»), pp. 222–223 («the meat of the song»), pp. 251–252 (pochi BPM). Tutte le altre pagine di Ohriner,
Huron, Margulis, Krims, Katz, Webber e i passi di B&B usati nei profili sono stati ritrovati (elenco nello script di verifica
riportato in `BOOK_RULES_EVIDENCE`/`PROFILE_BOOKS_EVIDENCE`).

## 3. Cue per ruolo e tecnica (P0.4)

**Misura** sul catalogo del ponte (`runs/dj_set_v3_bridge/dj_plan/tracks`, 21 brani, 420 coppie ordinate;
`runs/p0_misure_regole/cues.json`). Downbeat di inizio frase di livello ≥ 4: 514.

| Soglia (`dj_select.cue_candidates`) | Esclusi | Di cui «plausibili» [H] |
|---|---|---|
| OUT ≥ 35 % della durata | 186 | 55 su 183 uscite plausibili (30 %): fini di strofa/ritornello prima del 35 % |
| OUT con < 4 battute residue | 19 | 5 |
| IN ≤ 50 % | 250 | 77 su 188 ingressi plausibili (41 %), di cui 49 ritornelli |
| IN con < 16 battute residue | 86 | 16; **mai vincolante da sola** su questo catalogo (tutti oltre il 50 %) |

«Plausibile» è un proxy [H] senza annotazioni di DJ (non esistono per questi brani): IN = inizio di sezione allin1
«chorus»/«verse»; OUT = inizio di sezione dopo «chorus»/«verse». Non è recall/precisione su cue annotati: quella misura
richiede annotazioni held-out (resta aperta). Coppie di cue: 77.263 ammesse su 250.322 (69 % escluse). Il primo ritornello è
un cue IN in 20 brani su 21 (escluso solo in `18k_no_kizzy`, al 52 %). Molti ingressi esclusi lasciano a B più di 60 s
(la durata minima richiesta), quindi la soglia del 50 % è più stretta del vincolo di ruolo.

**Opzione implementata** (default invariato): `dj_select.CUE_WINDOWS["technique"]` e
`dj_set.candidates_for_pair(..., cue_windows="technique")`. Limiti di ruolo invece delle frazioni: B entra fino al 75 % se
gli restano ≥ 60 s e ≥ 16 battute; A può uscire da 60 s in poi (anche subito dopo un ritornello presto: B&B cap. 13 'Even
tighter' [P], «mix straight after a chorus ... very effective with breakbeat genres (hip hop ...)»; Katz p. 138 [A],
quick-mixing di frammenti brevi). 60 s = `CUE_PARAMS['play_min_s']` (richiesta dell'utente 60–120 s); `choose_cues`
controlla comunque la durata reale. Le finestre «technique» **aggiungono** cue, non ne tolgono; i candidati fuori dalle
finestre di default che l'albero farebbe come blend vengono scartati (una sovrapposizione resta nelle finestre di default);
a parità di costo vince il cue delle finestre di default (nessuna preferenza nascosta per uscite anticipate). Nessuna varietà
forzata: costi invariati.

Effetto (profilo libri, senza forme, prima dei filtri del runner): cue IN 264 → 381, OUT 309 → 357, candidati 77.263 →
128.588 (+51.325: 48.343 echo out, 2.493 backspin, 283 filter out, 203 filter verso il ritornello, 3 cut; 65 blend scartati).
Recuperati 45 dei 77 ingressi plausibili esclusi (22 ritornelli) e 22 delle 55 uscite plausibili (9 dopo un ritornello). Il
miglior candidato per coppia cambia in **1 coppia su 420** (`goosebumps → antidote`: filter out con B sulla strofa al 50,2 %,
costo 0,15 → 0,12). Quindi le finestre non sono il collo di bottiglia della scelta per coppia; l'effetto sul set
(`choose_cues`, vincoli di durata, `max_per_transition = 400`) va misurato con un run del runner, non fatto qui. Test:
`test_default_windows_unchanged`, `test_technique_windows_only_add_cues_and_flag_them`,
`test_candidates_for_pair_technique_windows_never_add_blends`.

## 4. Piano per eventi (P0.5)

Specifica dei dati e mappa sulle strutture esistenti in [`PIANO_EVENTI_SPEC_2026-10-21.md`](PIANO_EVENTI_SPEC_2026-10-21.md);
non implementata.

## 5. Test

Mirati (`test_dj_form`, `test_dj_phrase_books`, `test_dj_set`, `test_dj_select`, `test_dj_policy`, `test_dj_books`,
`test_dj_set_v3_bridge`, `test_p0_misure_regole`): 117 passati. Suite completa (`python -m pytest -q`, 10 min): **642
passati, 3 falliti**, tutti in `test_dj_qa.py`/`test_dj_creative.py` (stato `not_applicable` invece di `pass` nel QA: modifiche
in corso dell'altra tranche su `dj_qa`/`dj_creative`, file fuori da questo perimetro).

## 6. Che cosa resta

- Recall/precisione dei cue su brani held-out con cue annotati (servono annotazioni; il proxy qui non le sostituisce).
- Run del set con `cue_windows="technique"` (runner v2/v3: oggi non espone l'opzione) e confronto cieco con il default.
- Onset su audio annotato (P1.2): la correzione elimina un errore di contratto, non misura l'errore degli onset.
- `dj_form` su rap: la soglia della cassa (0,30 del 95° percentile) manca 808 lunghi (`gods_plan`); serve una misura di
  cassa/basso per l'hip-hop prima di usare intro/outro nei costi.
- Piano per eventi (P0.5) da implementare; QA su time-map (P0.6) dell'altra tranche.
