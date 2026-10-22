# Proposte di patch da Ohriner, Krims, Huron, Margulis (moduli diversi da dj_phrase)

Data: 21/10/2026. Nessuna di queste patch è applicata: `dj_creative.py` e `dj_policy.py` non vanno
toccati da questo lavoro, gli altri moduli hanno altri responsabili. Ogni proposta indica funzione,
regola, parametro (default che lascia invariato il comportamento attuale), fonte con pagina ed
etichetta ([E] sperimentale, [T] teoria, [A] analisi/pratica, [H] nostra ipotesi), e il test che
serve prima di attivarla. Gli strumenti già pronti in `dj_phrase` sono indicati con `P.`.

Regola generale (Huron p. 99 [T]; AGENTS.md): una regola di questi libri non diventa default
finché non è verificata su dati (MCFlow per il flow, ascolti per le transizioni).

---

## A. dj_select / dj_set — scelta del punto di uscita e d'ingresso

### A1. Fill o stop naturale nell'ultima battuta della frase di A
- **Dove**: `dj_select.cue_candidates(track, role="out")` (punteggio del candidato) o `dj_set.candidates_for_pair`.
- **Regola**: se nell'ultima battuta prima del cue-out di A c'è un fill (aumento degli attacchi percussivi
  nel 4° beat rispetto alla mediana delle battute precedenti della frase) o una battuta di stop
  (energia percussiva < −12 dB rispetto alla frase), aggiungere un bonus: il «quando» del downbeat
  successivo è reso certo (Huron pp. 247–250 [A]; Krims p. 98 [A]: l'upbeat nell'ultima battuta del gruppo annuncia il cambio).
- **Parametro**: `fill_bonus = 0.0` (proposto 0.15 dopo ascolto); soglia attacchi `fill_onset_ratio = 1.5` [H].
- **Dati**: stem `drums` di Demucs già presente; onset per battuta come in `dj_phrase.StemActivity`.
- **Test**: confronto in ascolto 2AFC tra uscite con e senza fill sulla stessa coppia (clip 20 s).

### A2. Fine verso «chiusa» vs fine verso con pickup (BC 3)
- **Dove**: `dj_phrase.exit_point` produce già `end_pos16`, `end_on_beat4`, `next_onset_gap_s`; in
  `dj_select.plan_transition` usare il campo per preferire, a parità di altro, una fine verso
  chiusa (ultima sillaba entro BC 3 e silenzio sul resto di BC 3) su una con pickup verso il
  verso successivo di A (attacchi su BC 3 dopo pausa su BC 2).
- **Fonte**: Krims p. 194 [A, un brano]; Krims p. 133 [A] (un verso senza upbeat chiude un'unità).
- **Parametro**: `closed_line_bonus = 0.0` [H].
- **Test prima di attivarla**: su MCFlow misurare se le fini verso seguite da pausa su BC 3 coincidono
  più spesso con fine di distico/strofa (`**break` ≥ 4) che le altre. `next_onset` nel modello di fine
  verso dà solo +0,001 di AUC (held-out), quindi è un indizio di *chiusura*, non di *fine verso*.

### A3. Ingresso con l'hook per brani noti
- **Dove**: `dj_set.candidates_for_pair` (scelta di `jb`), `dj_set.section_cost`.
- **Regola**: se B è marcato come noto al pubblico, preferire l'ingresso sul primo hook/ritornello
  (etichetta `chorus`) e lasciarlo arrivare pulito sul downbeat ipermetrico, senza sovrapposizione
  vocale (Huron p. 224 [E]: il primo suono distintivo basta al riconoscimento; Margulis p. 82 [E]:
  il ritornello è il frammento più ricordato; Margulis p. 153 [osservazione]: anticipazione condivisa).
- **Parametro**: `known_track_hook_bonus = 0.0`; richiede un campo `known` nel catalogo (oggi assente).

### A4. Introduzione «a Barcarolle»
- **Dove**: `dj_set.decide_transition`, regola R3_chorus_entry.
- **Regola**: prima dell'hook di B far sentire 2–4 battute di un elemento di B che lo anticipa
  (riff/strumentale), poi l'hook sul downbeat ipermetrico (Huron p. 321 [A]; Margulis p. 123 [storia]:
  versione semplice prima della ricca).
- **Parametro**: `pre_hook_bars = 0` (proposto 4) [H].

## B. dj_set — durata e tipo di raccordo, ordine

### B1. Cambio di schema tra generi lontani
- **Dove**: `dj_set.decide_transition` e `dj_set.edge_cost`.
- **Regola**: se la distanza timbrica A–B (z-score degli embedding, `timbre_distances`) supera il
  90° percentile del pool, vietare blend sovrapposti più corti di `P.BOOK_RULES["schema_switch_seconds"]`
  (3 s) e preferire un raccordo netto (stop/filtro/eco) a una sovrapposizione ambigua.
- **Fonte**: Huron pp. 207–208 [E] (stile riconosciuto in 0,25–1 s), pp. 210–211 [E] (~3 accordi per
  cambiare schema, residuo), p. 212 [T] (segnalare il cambio), p. 213 [T] (firewall).
- **Correzione 21/10/2026**: p. 211 misura modulazioni a tonalità vicine; per generi lontani l'autore congettura
  («I doubt ...») e p. 213 chiede altra ricerca. Il 90° percentile e i 3 s sono [H]; con matrice timbrica degenere
  ci si astiene (`dj_set.far_timbre_calibration`).
- **Parametro**: `far_genre_percentile = None` (spento); `far_genre_min_blend_s = 3.0`.

### B2. Pump-up preferito al pump-down
- **Dove**: `dj_creative.auto_key_shift` (solo tramite chiamante: `dj_set`/`dj_policy`), o `dj_set.edge_cost`.
- **Regola**: a parità di distanza Camelot, preferire lo shift di B che porta la tonalità **in su**
  di 1–2 semitoni rispetto ad A; uno spostamento in giù è raro nel pop e usato come effetto comico.
- **Fonte**: Huron p. 285 [A/osservazione: centinaia di pump-up, nessun pump-down noto]. [H] che valga in un mix.
- **Parametro**: `key_down_cost = 0.0` (proposto 0.05).

### B3. Peso dell'arco globale
- **Dove**: `dj_set.arc_target`, `dj_set.order_set` (peso dell'energia rispetto agli archi).
- **Regola**: non aumentare il peso dell'arco globale sopra quello delle singole transizioni; il
  pubblico è poco sensibile all'ordine a larga scala (Huron p. 236 [E]; Margulis pp. 128–129 [E]).
- **Azione**: nessun cambio di default; documentare la fonte accanto al parametro dell'arco.

### B4. Varietà del tipo di transizione (U di Wundt)
- **Dove**: `dj_policy.PARAMS["repeat_penalty"]` (oggi 0,15, fonte Williams 2025).
- **Regola**: aggiungere come fonte Margulis pp. 95–98 [E] (gradimento a U rovesciata con la
  ripetizione) e Huron p. 367 [T] (set troppo prevedibile → calo di attenzione); variare il tipo di
  transizione ma **non** il metro (Huron pp. 349–350 [T]).
- **Parametro**: invariato; solo fonte.

### B5. Brani lontani dallo stile del set
- **Dove**: `dj_set.candidates_for_pair` (ingresso di B).
- **Regola**: per un B con distanza di stile alta dal resto del set, preferire un ingresso su una
  sezione ripetitiva (intro/loop strumentale) che dia tempo di formare aspettative.
- **Fonte**: Huron p. 367 [T]; Margulis p. 23 [T] (la ripetizione aiuta la segmentazione).
- **Parametro**: `unfamiliar_intro_bonus = 0.0`.

## C. dj_creative / dj_policy — effetti (proposte, file non modificati)

### C1. Build per dimezzamento del loop
- **Dove**: nuova variante di `dj_creative.loop_extend` (`schedule="halving"`) o di `build_into_chorus`.
- **Regola**: il loop d'uscita di A si accorcia 2 → 1 → ½ → ¼ battute (ognuna ≥ 2 ripetizioni) fino al
  downbeat d'ingresso di B. Pianificazione pronta: `P.build_halving_schedule(total_bars, start_unit,
  min_unit, repeats_per_stage)` (testata).
- **Fonte**: Margulis p. 23 e p. 44 [A] (unità che si restringe = accelerazione percepita a tempo costante);
  Huron pp. 323–326 [T] (accelerazione degli eventi come indicatore di climax).
- **Nota**: `build_into_chorus` dimezza già la suddivisione del rullo (`ROLL_SCHEDULES`): è lo stesso
  principio sugli attacchi; il dimezzamento del loop lo applica al materiale di A.
- **Test**: confronto in ascolto con `build_into_chorus` standard su 4 coppie.

### C2. Loop d'estensione con traiettoria (Sokolov)
- **Dove**: `dj_creative.loop_extend`.
- **Regola**: ogni ripetizione del loop cambia un parametro (filtro passa-alto che sale di un'ottava,
  o uno strato che entra/esce) e l'intensità massima si concentra sull'ultima iterazione, che coincide
  con l'ingresso di B. Oggi il loop è identico a ogni ripetizione.
- **Fonte**: Margulis pp. 130–132 [A]; Margulis p. 129 [T] (la ripetizione troppo evidente è sgradita).
- **Parametro**: `trajectory = "none"` (default attuale), alternative `"hpf_rise"`, `"layer_add"`.
- **Vincolo collegato**: `P.loop_repetition_check(loop_bars, repeats, new_layer_entering)` avvisa già
  quando un loop ≤ 2 battute è ripetuto > 3 volte senza nuovo strato (Huron p. 286 [E, un caso];
  Margulis pp. 173–174).

### C3. Stop di una battuta prima del drop
- **Dove**: `dj_creative.build_into_chorus(gap_beats=...)` esiste già; `dj_policy.propose` dovrebbe
  proporre `gap_beats` ∈ {1, 2} (mai > 4 beat) quando B entra su un downbeat ipermetrico certo.
- **Fonte**: Huron pp. 314–316 [T/E] (ritardare un evento prevedibile aumenta la tensione; rallentare prima
  dei downbeat ipermetrici), p. 335 [A] (pausa nella formula prevedibile), p. 368 [T] (ritardo troppo
  lungo → si perde la tendenza); memoria a breve termine 3–5 s (p. 228 [E]).
- **Parametro**: `drop_gap_beats_max = 2` [H].

### C4. Diradare A quando entra la frase chiave di B
- **Dove**: `dj_creative.acapella_over`, `vocal_swap` (curve dei bassi/medi di A).
- **Regola**: quando la prima frase di B è un hook o una parola tonica sul downbeat dopo un upbeat lungo,
  anticipare il bass swap e abbassare i medi di A di 3–6 dB per la durata della frase.
- **Fonte**: Krims p. 196 [A, pratica di produzione]; Krims pp. 194–195 [A] (upbeat lungo enfatizza il downbeat).
- **Parametro**: `key_phrase_duck_db = 0.0`.

### C5. Coincidenza accenti voce–cassa nell'acapella
- **Dove**: `dj_policy.propose`, contesto C3_verse_handoff (acapella_over/vocal_swap).
- **Regola**: calcolare la quota di accenti vocali di B (`P.ohriner_accents` sulle sillabe Whisper)
  che coincidono con gli attacchi di cassa/rullante di A, confrontata con gli spostamenti di 1–4
  sedicesimi; segnalare come rischio i valori estremi (z > 2 o < −2): né la convergenza massima né
  quella minima sono «giuste», ma gli estremi cambiano il carattere del flow.
- **Fonte**: Ohriner pp. 164–175 [A] (flow convergente/divergente con il beat), p. 180 [E, N = 8].
- **Parametro**: diagnostica (`accent_kick_z`), nessun peso.

### C6. Fade lungo solo senza chiusura netta
- **Dove**: `dj_set.decide_transition` (tipo `fade`).
- **Regola**: un fade lungo di A «rimanda la chiusura all'infinito»: sceglierlo solo se A non ha una
  fine frase chiara nel punto d'uscita (`exit_point.status` ≠ clean) e mai sovrapposto a un drop di B.
- **Fonte**: Huron p. 318 [T].

## D. dj_critic — giudice automatico

### D1. Sorpresa sul «quando» separata dal «cosa»
- **Dove**: `dj_critic.surprisal_seam_stats` / `seam_feature_block`.
- **Regola**: aggiungere un canale di sorpresa metrica (scarto degli onset di B dalla griglia attesa
  di A; deviazione dal downbeat ipermetrico con `P.hypermetric_weight`) separato dalla sorpresa di
  contenuto (MusicGen NLL). In una buona transizione DJ la sorpresa metrica è ≈ 0; la sorpresa di
  contenuto va premiata solo se quella metrica è bassa.
- **Fonte**: Huron p. 116 [T] (cosa vs quando), pp. 270, 303–304 [T] (sorpresa piacevole solo in uno
  schema forte; la sincope presuppone il metro), pp. 349–350 [T] (contrametrico = sgradito).

### D2. Incertezza ≠ sorpresa realizzata ≠ accuratezza
- **Dove**: `dj_critic.MusicGenSurprisal` (output).
- **Regola**: registrare l'entropia predittiva prima del seam separatamente dal −log p realizzato.
- **Fonte**: Huron p. 55 [T].

### D3. Finestre del critico
- **Dove**: `dj_critic.SEAM_SIDE_S`, `AESTHETIC_WINDOW_S`.
- **Regola**: oltre alle finestre attuali, un blocco di 1 s attorno al seam (riconoscimento di stile,
  Huron pp. 207–208 [E]) e uno di 3–5 s (memoria a breve termine, p. 228 [E]); costanti in
  `P.BOOK_RULES["style_schema_seconds"]` e `["stm_seconds"]`.

### D4. Drop-out come evento
- **Dove**: `dj_qa` / `dj_critic.technical_from_qa`.
- **Regola**: un beat mancante nel seam non è «assenza neutra» ma un evento (Huron p. 52 [E citata]):
  se non è voluto (non c'è `gap_beats` nel rendering) segnalarlo come difetto.

## E. Valutazione umana (pagina d'ascolto Artifact)

### E1. Nessuna descrizione prima del voto
- **Regola**: nella pagina d'ascolto mostrare tecnica e motivazione solo **dopo** il voto.
- **Fonte**: Margulis p. 15 [E] (le note esplicative riducono il piacere, Margulis 2010).

### E2. Registrare ascolti ripetuti
- **Regola**: salvare quante volte la clip è stata riascoltata prima del voto; i voti sulla stessa clip
  ripetuti nel tempo non sono indipendenti (picco di tensione anticipato e attenuato con la familiarità).
- **Fonte**: Margulis p. 101 [E]; Huron pp. 133–134 [E] (familiarità inconscia).

### E3. Punto in cui si nota il cambio
- **Regola**: opzionale, un tasto «ho sentito il cambio» durante l'ascolto: misura continua oltre al voto finale.
- **Fonte**: Huron p. 51 [T] (il giudizio dopo uno stop mescola qualità e chiusura).

## F. dj_phrase — estensioni future (non fatte qui per mancanza di validazione)

- **Pendenza F0 nelle ultime sillabe** come indizio di fine verso (Huron pp. 77, 88 [E]; oggi c'è solo
  `f0_final_st`): serve F0 per sillaba dallo stem vocale; da validare su un corpus con audio (MCFlow
  non ha audio).
- **Rilevatore di inspirazione** prima del verso successivo (Ohriner pp. 43–44: fine frase = sillaba
  prima di un respiro [convenzione del corpus]).
- **Cambio di groove come confine di sezione** (Ohriner pp. 143–144 [A]): `P.groove_segments` sulle
  parole Whisper → candidato di cue; da confrontare con i confini SSM/LinkSeg.
- **F5. Onset sillabici dallo stem vocale** (priorità alta per le misure di Ohriner): `runs/dj_phrase_v2/compare.json`
  mostra che con i tempi delle parole Whisper (sillabe distribuite uniformemente nella parola) nPVI
  (mediana 48,5 contro 24,5 su MCFlow), grooviness (0,05 contro 0,51 di Ohriner Tab. 5.2) e fase vocale
  (Ohriner pp. 190–196: ritardo tipico ~0,11 beat; noi 0 con 5 % di disallineamento spiegato) sono
  dominate dal rumore di temporizzazione. Patch in `dj_phrase.ohriner_syllables_from_model` e
  `vocal_phase_alignment`: agganciare ogni sillaba al più vicino onset dello stem vocale (flusso
  spettrale, finestra ±60 ms) prima di quantizzare; verificare che la grooviness torni nell'intervallo
  di MCFlow sui brani con ASR affidabile.
- **Fase ipermetrica dalle fini verso**: verificato che non funziona (quota 0,28 ≈ caso 0,25 su MCFlow);
  serve la fase dalle sezioni strumentali (dj_select), non dalla voce.
