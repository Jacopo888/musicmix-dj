# Schede brano persistenti e affidabilità del pianificatore (21/10/2026)

Due pilastri dell'autonomia (AGENTS.md, «Autonomia»; piano in [`AUTONOMIA_VALUTAZIONE_2026-10-21.md`](AUTONOMIA_VALUTAZIONE_2026-10-21.md),
strade 6 e 9): ogni brano ha una **scheda preparata una volta sola** e il pianificatore può **scegliere le situazioni che sa
risolvere** invece di forzare ogni coppia; copertura della libreria e varietà dei set sono misurate a parte.
Nessun ascolto e nessun render: tutto ciò che segue è integrità e tracciabilità delle analisi, non qualità musicale.
Livelli: [S] sistema/strumento, [E] risultato sperimentale, [P] pratica, [H] ipotesi o soglia nostra, [M] misura nostra.

## 1. Codice

| Parte | Dove | Default |
|---|---|---|
| Scheda `musicmix.track-card.1` (pura, nessun I/O) | `src/musicmix/track_cards.py` (`build_card`, `grid_section`, `key_section`, `voice_section`, `entries_section`, `exits_section`, `apply_corrections`, `THRESHOLDS`, `FLAGS`) | — |
| Costruzione incrementale + correzioni | `tools/build_track_cards.py build [--only] [--force] [--compute-phrase]`, `correct --slug --kind ...` → `track_cards/<slug>.json`, `track_cards/index.json`, `track_cards/corrections.json` | rifà solo le schede con ingressi cambiati |
| Ponte | `tools/run_bridge_pool.py` → `run_plan`: dopo `dj_plan_pool` (exit 0) lancia `build_track_cards.py build --compute-phrase` (log `runs/cloud_session/pool_bridge_v1_track_cards.log`) | automatico |
| Affidabilità, copertura, varietà | `src/musicmix/reliability.py` (`track_reliability`, `cue_reliability`, `ReliabilityHook`, `coverage`, `variety`, `compare_plans`, `PARAMS`) | — |
| Hook del runner | `SetHooks.reliability` in `tools/run_dj_set_v2_books.py` (dopo i filtri e le correzioni della coppia; `set.json["reliability"]`) | `None` = invariato |
| Opzioni del runner v3 | `tools/run_dj_set_v3_bridge.py set --reliability {off,report,prefer} [--cards DIR] [--reliability-params JSON] [--novelty set.json ...] [--no-manual-exclude]` | `off` |
| Confronto dei piani | `tools/plan_reliability_report.py runs/plan_reliability_v1` → `compare.json` | — |
| Test | `tests/test_track_cards_reliability.py` (12 passati); suite completa: 778 passati, 1 saltato (17:45) | — |

## 2. Schema della scheda (`track_cards/<slug>.json`)

- **Identità**: `audio.sha256` = `track_id` del catalogo (SHA-256 dell'audio canonico, rimisurato: `audio.verified`);
  `inputs` = percorso e SHA-256 di ogni ingresso (voce di catalogo, analisi DJ, modello di versi, attività dello stem, JSON
  vocals, spettrogramma allin1 e stem per dimensione+mtime, lingua); `input_key` = hash di ingressi, soglie, gravità,
  correzioni e versione (`track_cards.1`). Una scheda si rifà solo se `input_key` cambia.
- **`grid`**: BPM, provider, confidenza di `dj_select`, `beat_f`/`downbeat_f`, residuo, `agreement` con gli altri provider
  (rapporto di tempo, coincidenza di battiti e downbeat), voti d'ottava, `tempo_octave_suspect`,
  `downbeat_phase_disputed_by`, `beat_phase_disputed_by`.
- **`key`**: tonalità, Camelot, livello e confidenza, stime dei quattro stimatori e quanti concordano.
- **`structure`**: sezioni allin1 in battute; forma `dj_form.3` (segmenti, battute per etichetta, drop, breakdown, core) dagli
  spettrogrammi per stem di allin1 (per i V12 anche `a.npy`/`b.npy` verificati per durata), altrimenti dal mix.
- **`voice`**: lingua (`track_filter`), statistiche dell'ASR (modello, lingua, confidenza mediana, quota < 0,5), modello di
  versi `dj_phrase` scelto (prima nella lingua del brano: per `18k_no_kizzy` quello italiano di `runs/asr_language_v1`, non
  quello `small.en` del set v3), versi con `p_end`, regioni vocali, quota di battute con voce.
- **`entries`**: tutti i cue IN di `dj_select` (finestre default e «technique», campo `window`) con lo stato di
  `dj_phrase.b_lead_in` (= guardia `entry_clear_test` di `dj_set.entry_guard`): `instrumental`/`on_downbeat`/`lead_in` = `ok`,
  `mid_line` = B entrerebbe a metà frase; lead-in in secondi e prima parola.
- **`exits`**: tutti i cue OUT con lo stato di `dj_phrase.exit_point` (`clean`, `tail`, `a_pickup`, `instrumental`,
  `gap_not_line_end`, `mid_line`, ...), `p_line_end`, `line_end`, `clean_for_cut` (= `exit_clear_test`), parole troncate.
- **`usable`**: battute degli ingressi puliti, delle uscite a fine verso, per il cut, a metà verso (dopo le correzioni).
- **`instrumental_regions`**: corse di ≥ 2 battute senza voce (`bars.vocal` < 0,5).
- **`stems`**: stem vocale verificato (`musicmix.stems.vocal_stem_record`: modello, checkpoint, `input_sha256`, controlli)
  o l'errore; stem Demucs a 4 sorgenti con provenienza e verifica d'identità.
- **`stretch`**: limite consigliato per i blend (±8 % [S] MosaikBox; ±5 % per i blend lunghi [E] Kim 2020 §6.1; ±5 % [H]
  se la voce copre > 75 % delle battute), `overlap_recommended` falso se la griglia è incerta [H].
- **`uncertain`**: casi incerti con `id`, gravità, categoria (`grid`, `key`, `phrase`, `track`), `fix` (`listen` = verifica
  umana a campione, `data` = analisi da fare/rifare, `avoid` = il pianificatore li aggira, `none`), descrizione, motivo,
  evidenza, azione, `resolved_by`.
- **`corrections`**: correzioni applicate, scartate (`stale`: legate a un altro audio, battuta che non è un cue) e in attesa di
  rianalisi; **`reliability`**: vedi §4.

### Correzioni manuali (valgono in ogni set)

`track_cards/corrections.json` (`musicmix.track-card-corrections.1`), una lista per brano; ogni voce ha `id`, data, autore,
`kind`, `audio_sha256` (la correzione vale solo per quell'audio), `flag`/`bar`/`value`, `why`. Tipi: `exclude`/`include`,
`verify_flag` (un caso incerto verificato all'ascolto non pesa più), `block_entry`/`approve_entry`, `block_exit`/`approve_exit`,
`bpm_octave`/`downbeat_offset_beats`/`key` (registrati, aggiungono `correction_pending_reanalysis` finché l'analisi non è
rifatta), `note`. Si aggiungono con `tools/build_track_cards.py correct ...`, che ricostruisce solo quella scheda; il runner
legge le schede a ogni set. Oggi il file è vuoto: nessuna correzione è stata inventata.

## 3. Regole dei casi incerti (tutte [H], `track_cards.THRESHOLDS`/`FLAGS`)

| Caso | Condizione | Gravità | Risolve |
|---|---|---|---|
| `tempo_octave_suspect` | un provider all'ottava e BPM fuori da 70–160 (es. `antidote` 65,6 vs 131,1) | 0,7 (ottava sbagliata = ogni confine di battuta sbagliato) | ascolto |
| `grid_confidence_low` | confidenza `low` di `dj_select` | 0,35 | ascolto |
| `grid_fit_low` | `beat_f` o `downbeat_f` < 0,70 | 0,15 | ascolto |
| `grid_residual_high` | residuo mediano > 20 ms (metà della tolleranza di fase del veto del set) | 0,2 | ascolto |
| `downbeat_phase_disputed` | coincidenza dei downbeat con un altro provider < 0,5 | 0,1 | ascolto |
| `beat_phase_disputed` | stesso tempo, coincidenza dei battiti < 0,5 | 0,15 | ascolto |
| `key_confidence_low`, `key_estimators_disagree` | livello ≠ `high`; < 2 stimatori d'accordo | 0,3; 0,1 (solo nei blend) | il pianificatore |
| `phrase_model_missing`, `asr_unreliable`, `asr_language_mismatch`, `stem_activity_missing` | analisi assente o nella lingua sbagliata | 0,3; 0,3; 0,3; 0,1 | dati |
| `asr_low_confidence` | > 20 % delle parole con confidenza < 0,5 | 0,1 | ascolto |
| `no_clean_entries`, `few_clean_entries`, `no_line_end_exits` | nessun / < 2 ingressi puliti; nessuna uscita a fine verso | 0,5; 0,2; 0,2 | ascolto |
| `vocal_stem_missing`, `analysis_missing`, `audio_sha_mismatch` | stem o analisi assenti, audio diverso | 0,5; 1; 1 | dati |
| `language_not_allowed`, `language_uncertain` | `track_filter` | 1; 0,1 | — |
| `form_no_core`, `form_missing` | `dj_form` senza core / non calcolata | 0,05 | il pianificatore / dati |

Le gravità non sono tarate su ascolti né sui brani di sotto: sono un ordine di rischio dichiarato. L'unico ritocco dopo la
prima esecuzione: `tempo_octave_suspect` 0,5 → 0,7, con il motivo scritto (un'ottava sbagliata sposta ogni confine di battuta),
non per ottenere un esito.

## 4. Affidabilità (`musicmix.reliability`, tutte [H])

- **Brano**: prodotto di `1 − gravità` dei casi aperti per categoria; `planning` esclude la tonalità (conta solo quando A e B
  suonano insieme). Un brano con `planning` < `track_min` = 0,35 non entra nel set (tutti i suoi archi vietati).
- **Candidato** (coppia + cue + tecnica): griglie e versi di A e B; stato del verso di A all'uscita (`exit_factor`: fine verso 1,
  `gap_not_line_end` 0,85, `no_gap` 0,75, `mid_line` 0,6, ignoto 0,8; cut senza fine verso pulita × 0,5); ingresso di B
  (`instrumental`/`on_downbeat` 1, `lead_in` 0,9, `mid_line` 0,5; cue spostato dalla guardia: lo stato del lead-in verificato;
  cue assente dalla scheda 0,8); con sovrapposizione (blend) anche tonalità di A e B, sovrapposizione sconsigliata × 0,6,
  stretching oltre il limite più stretto delle due schede × 0,6 (un solo fattore).
- **Opzione `prefer`**: costo del candidato + `w_risk` (0,6) × (1 − affidabilità); sotto `pair_min` = 0,30 il candidato non è
  ammissibile. **`report`**: stesse misure, costi e scelte invariati. **Novità** (spenta: pesi 0; `--novelty` usa
  `NOVELTY_ON` = 0,3 per una coppia già suonata, 0,15 per entrare in un brano del set precedente).
- Limiti: affidabilità ≠ qualità; i fattori sono moltiplicati come se i rischi fossero indipendenti (non lo sono: griglia e
  versi dipendono dalla stessa analisi); la scheda descrive le analisi, non le verifica.

## 5. Schede dei 22 brani del ponte (comprende i 7 V12), tutti en/it

`tools/build_track_cards.py build` → 22 schede in 20 s la prima volta, poi 1–2 s (nessuna rifatta se gli ingressi non
cambiano), 816 KB. Catalogo `runs/pool_bridge_v1/catalog_ready.json` (22 brani, 7 con origine V12) + `configs/music_pool_v12.json`
(stessi 7); 21 inglese, 1 italiano, nessuno escluso dalla lingua; audio rimisurato = `track_id` per tutti. Affidabilità
complessiva / di pianificazione [M dalle regole H]; IN = ingressi puliti / cue IN; OUT = uscite a fine verso / cue OUT.

| Brano | BPM | Camelot | lingua | affid. / pianif. | IN | OUT | casi incerti aperti (gravità, chi li risolve) |
|---|---|---|---|---|---|---|---|
| 18k_no_kizzy | 125,0 | 4A | it | 0,90 / 0,90 | 5/19 | 5/16 | asr_low_confidence (0,1, ascolto): 24 % di parole < 0,5 con l'ASR italiano; la scheda usa i versi italiani, il set v3 quelli `small.en` |
| 5_percent_tint | 140,0 | 10A | en | 1,00 / 1,00 | 14/21 | 5/20 | — |
| 6ix9ine_kooda | 91,0 | 5A | en | 1,00 / 1,00 | 5/7 | 1/8 | — (una sola uscita a fine verso) |
| a_lot | 146,0 | 5A | en | 1,00 / 1,00 | 21/29 | 21/29 | — |
| after_hours | 109,0 | 5A | en | 0,90 / 0,90 | 14/33 | 15/35 | asr_low_confidence (0,1) |
| antidote | 65,6 | 2B | en | 0,14 / 0,22 | 11/16 | 7/14 | **tempo_octave_suspect** (0,7, ascolto), grid_residual_high 35 ms (0,2), downbeat_phase_disputed (madmom, 0,1), key_confidence_low + disagree (0,3 + 0,1, aggirabili) |
| butterfly_effect | 70,5 | 12A | en | 0,90 / 0,90 | 4/9 | 5/9 | downbeat_phase_disputed (madmom 2×, 0,1) |
| creature | 142,0 | 10A | en | 1,00 / 1,00 | 15/23 | 8/21 | — |
| gods_plan | 77,2 | 9B | en | 0,63 / 1,00 | 12/13 | 9/11 | key_confidence_low + disagree (aggirabili) |
| goosebumps | 130,0 | 9A | en | 0,90 / 0,90 | 17/28 | 14/27 | downbeat_phase_disputed (Beat This! a metà tempo, 0,1) |
| heartless | 85,0 | 2A | en | 0,81 / 0,81 | 7/13 | 2/11 | downbeat_phase_disputed (0,1), asr_low_confidence (0,1) |
| jimmy_cooks | 83,0 | 4A | en | 0,48 / 0,77 | 9/16 | 9/15 | grid_fit_low 0,57 (0,15), downbeat_phase_disputed (0,1), key low + disagree |
| knife_talk | 73,0 | 3A | en | 1,00 / 1,00 | 12/17 | 9/14 | — |
| life_is_good | 71,0 | 9A | en | 0,59 / 0,59 | 0/14 | 0/16 | phrase_model_missing (0,3, dati: non è nel catalogo congelato del set v3), grid_fit_low 0,63 (0,15) |
| mafia | 110,0 | 5A | en | 0,77 / 0,77 | 10/23 | 6/21 | beat_phase_disputed (Beat This! coincidenza dei battiti 0, 0,15), downbeat_phase_disputed (0,1) |
| nonstop | 77,5 | 11A | en | 1,00 / 1,00 | 13/18 | 8/15 | — |
| reminder | 80,0 | 5A | en | 0,90 / 0,90 | 4/13 | 4/13 | stem_activity_missing (0,1, dati); versi V12 di `runs/dj_phrase_v1`; stem rifatto da un'altra tranche (`vocals_redo`) e verificato: alla prima costruzione era `vocal_stem_missing` (0,45) e la scheda si è rifatta da sola |
| rich_flex | 76,6 | 2B | en | 0,35 / 0,55 | 10/15 | 8/14 | **grid_confidence_low** (0,35, ascolto; QA del set v3: griglia di B −316 ms in T13), grid_fit_low 0,59 (0,15), key low + disagree |
| sicko_mode | 77,5 | 2A | en | 0,63 / 1,00 | 15/22 | 16/22 | key_confidence_low + disagree (aggirabili) |
| stargazing | 75,0 | 6A | en | 0,54 / 0,85 | 11/18 | 10/18 | grid_fit_low 0,62 (0,15), key low + disagree |
| superhero | 116,7 | 4A | en | 1,00 / 1,00 | 8/15 | 2/14 | — (due sole uscite a fine verso) |
| too_many_nights | 88,0 | 6A | en | 0,81 / 0,81 | 7/13 | 6/10 | stem_activity_missing (0,1, dati), asr_low_confidence (0,1); stem `vocals_redo` verificato (prima 0,41) |

**Intervento umano richiesto** (`index.json`): coda d'**ascolto** = 2 brani (`antidote`: ottava e downbeat; `rich_flex`: griglia);
coda **dati** = 3 (`life_is_good` versi; `reminder` e `too_many_nights` attività dello stem, ora che lo stem rifatto c'è); i casi di tonalità incerta (7 brani) li aggira il
pianificatore (niente blend), senza ascolto. Il ponte con `--compute-phrase` chiude da solo il caso di `life_is_good`: provato in
una cartella di prova (14 s, nessuna rete neurale), affidabilità 0,59 → 0,85; nella scheda vera il caso resta segnalato
perché questa tranche usa solo analisi già presenti.

## 6. Piano con e senza affidabilità (`runs/plan_reliability_v1/`)

Run di sola pianificazione (`set --plan-only --event-plan`, catalogo congelato del set v3, guardia dell'ingresso di B accesa
come oggi), ~2 min ciascuna; confronto in `compare.json` (`tools/plan_reliability_report.py`). Le tecniche confrontate sono
quelle che il set proverebbe per prime (policy dei libri o albero). [M] = misura su questo pool, non qualità.

| Variante | Comando (oltre a `--plan-only --event-plan --out runs/plan_reliability_v1/<nome>`) | costo d'ordine | affidabilità media / minima delle 13 transizioni | che cosa cambia rispetto a `off` |
|---|---|---|---|---|
| `off` | — | 24,468 | (0,657 / 0,372 misurata da `report`) | — |
| `report` | `--reliability report` | 24,468 | 0,657 / 0,372 | nulla (13/13 coppie, cue e tecniche): l'hook in misura non cambia scelte |
| `prefer` | `--reliability prefer` (w_risk 0,6) | 26,339 | 0,657 / 0,372 | **nulla**: il miglior candidato cambia in 105/306 coppie, ma l'ordine e il Viterbi dei cue restano gli stessi |
| `prefer_w1.5` | `--reliability-params '{"w_risk": 1.5}'` | 25,240 | 0,688 / 0,459 | via `after_hours`, dentro `superhero`; 5 coppie nuove, 7/13 cue uguali |
| `prefer_w3` | `'{"w_risk": 3.0}'` | 27,794 | 0,765 / 0,486 | via `mafia`, dentro `superhero`; 4 coppie nuove, 5/13 cue uguali; uscite di A a metà verso 3 → 1 |
| `off_noexclude` | `--no-manual-exclude` | 21,481 | — | senza l'esclusione a mano entra **`antidote`** (e `jimmy_cooks`) |
| `prefer_noexclude` | `--reliability prefer --no-manual-exclude` | 26,155 | 0,657 / 0,372 | **`antidote` evitato dalla scheda** (pianificazione 0,22 < 0,35: 36 coppie non pianificabili); stesso set di `prefer` |
| `repeat_2` | come `prefer` | 26,339 | 0,657 / 0,372 | identico a `prefer` (il pianificatore è deterministico) |
| `novelty_2` | `prefer --novelty prefer/set.json` | 28,685 | 0,713 / 0,497 | 13/13 coppie nuove; `superhero` al posto di `goosebumps` |
| `novelty_3` | `prefer --novelty prefer/set.json novelty_2/set.json` | 26,525 | 0,715 / 0,486 | dentro `heartless`, `jimmy_cooks`, `goosebumps`; 1 coppia ripetuta da `novelty_2` |

**Perché `prefer` (w 0,6) non cambia nulla** [M]: le alternative più affidabili esistono (es. T2 MAFIA → 18K, uscita di A a metà
verso, affidabilità 0,37: nel piano di `prefer` 8 alternative con A che completa la sezione costano 0,07–0,76 in meno) ma sono già scartate dal Viterbi
del set per durata dei brani, spazio tra transizioni e sezioni intere (`why_not` del piano per eventi). Il rischio aggiunto
(≤ 0,6 × 0,63 = 0,38) non supera questi vincoli. Nessuna coppia scende sotto `pair_min` (minima 0,37), nessun brano del pool
(dopo l'esclusione a mano di `antidote` e gli stem mancanti) sotto `track_min`. Non abbiamo alzato i pesi per «far succedere
qualcosa»: le varianti 1,5 e 3 sono una misura di sensibilità, non una nuova taratura.

**Che cosa evitano i pesi più alti** [M] (`compare.json`, `changed_transitions`, con l'affidabilità della scelta di `off`):

- MAFIA → 18K (0,37: A a metà verso a battuta 86, B con lead-in su un cue spostato) e After Hours → MAFIA (0,53: A su
  `gap_not_line_end`) spariscono in entrambe; con w 3 MAFIA esce dal set (griglia 0,77: Beat This! fuori fase sui battiti).
- Creature → BUTTERFLY EFFECT: da battuta 80 (A a metà verso, 0,49) a battuta 44 (0,81) con w 3.
- 5% TINT → Creature: B a battuta 8 invece di 43 (0,90 → 1,00) con w 3; Sicko Mode → Rich Flex: B a battuta 0 (0,50 → 0,55).
- Rich Flex → God's Plan (0,50, griglia di Rich Flex a bassa confidenza) sparisce con w 3: Rich Flex chiude il set.
- Il prezzo: con w 1,5 entra un blend (MAFIA → Superhero, 0,46: un blend porta tonalità e stretching nel rischio).

**Copertura e varietà** [M] (pool del set: 18 brani; schede: 22):

| Sequenza di set | brani usati per set | copertura cumulata del pool | coppie ripetute dal set precedente | tecnica uguale due volte di fila | entropia delle tecniche |
|---|---|---|---|---|---|
| set v3 congelato → `prefer` → `repeat_2` | 14, 14, 14 | 14/18 (0,78) fermo; mai usati `6ix9ine_kooda`, `heartless`, `jimmy_cooks`, `superhero` | 13/13 (Jaccard 1,0) | 1, 4, 4 | 0,88, 0,81, 0,81 |
| set v3 → `prefer` → `novelty_2` → `novelty_3` | 14, 14, 14, 14 | 14 → 14 → 15 → **17/18 (0,94)**; 17/22 schede; mai usato `6ix9ine_kooda` (91 BPM) | 13, 0, 5 (Jaccard coppie 1,0, 0,0, 0,04) | 1, 4, 5, 0 | 0,88, 0,81, 0,79, **0,96** |

Senza un'opzione di novità il pianificatore rifà lo stesso set (stesso ordine, cue e tecniche): la varietà va chiesta
esplicitamente. Con la novità [H] la copertura sale e le coppie cambiano, ma l'affidabilità media resta più alta di `prefer`
(0,71 contro 0,66) perché la novità agisce dopo il rischio. `backspin` compare solo in `novelty_2`; `cut` in nessun piano.

## 7. Limiti e cose aperte

- Le schede descrivono le analisi; i casi «ascolto» sono la coda della verifica umana a campione (AGENTS.md, Autonomia), non
  un voto per ogni brano. Nessuna correzione è ancora stata raccolta.
- Pesi e soglie [H] non validati; l'opzione `prefer` è spenta di default. La misura giusta è l'accordo con le correzioni e
  con i passaggi di riferimento (`tools/decision_agreement.py`), non ancora disponibile.
- La tabella degli ingressi/uscite copre i cue di `dj_select` (livello di frase ≥ 4); i cue spostati dalla guardia usano lo
  stato registrato nel piano.
- I modelli di versi di `reminder`/`too_many_nights` sono quelli V12 (`runs/dj_phrase_v1`, ASR del vecchio JSON vocals) senza
  attività dello stem; il catalogo congelato del set v3 li esclude ancora (stem assenti al momento del congelamento).
- Con il render (`set` senza `--plan-only`) il rapporto è scritto prima dei tentativi: le transizioni accettate dopo QA/veto
  possono differire (vedi `transitions[].attempts`).
