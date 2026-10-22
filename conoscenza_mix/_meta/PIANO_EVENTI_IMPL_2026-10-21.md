# Piano per eventi (P0.5) e cassa 808 in `dj_form`: implementazione (21/10/2026)

Implementa la specifica [`PIANO_EVENTI_SPEC_2026-10-21.md`](PIANO_EVENTI_SPEC_2026-10-21.md) (schema `musicmix.event_plan.1`)
e chiude il punto aperto «soglia della cassa sull'hip-hop (808)» di [`P0_MISURE_REGOLE_2026-10-21.md`](P0_MISURE_REGOLE_2026-10-21.md)
§6. Riga P0 della revisione: «Piano per eventi e integrazione reale» ([REVISIONE_PIPELINE](../libri/audit_2026-10-21/REVISIONE_PIPELINE.md)).
Nessun ascolto e nessun render audio: tutto ciò che segue è descrizione, tracciabilità e integrità delle misure, non qualità
musicale. Livelli: [P] pratica, [A] analisi di un autore, [T] teoria, [H] ipotesi nostra, [M] misura nostra.

## 1. Che cosa è implementato

| Parte | Dove | Default |
|---|---|---|
| Documento `musicmix.event_plan.1` per un candidato (evento di A, lead-in e evento obiettivo di B, ritardo, scelta, alternative per asse, tabella di tutti i candidati, trace con versioni) | `src/musicmix/dj_events.py` (`event_plan`, `a_event`, `b_events`, `pick_target`, `lead_in`, `arrival_summary`, `enumerate_alternatives`) | puro, nessun I/O; non entra nei costi |
| Contesto additivo | `dj_set.transition_context(..., events=False)` → `event_context`; `dj_set.candidates_for_pair(..., event_plans=False)` → `event_summary` per candidato | spento: contesto, costi e ordine identici (test) |
| Opzione **sperimentale [H]** | `dj_set.CUE_PARAMS['w_event_arrival']` (0 = spento), `event_arrival_max_bars` (2); `dj_set.choose_cues(..., event_fn=None)` | spenta |
| Runner | `tools/run_dj_set_v3_bridge.py set --plan-only --event-plan [--event-pref] [--out ...] [--form-params ...]`; hook `SetHooks.after_cues/event_fn/cue_overrides/form_params` in `tools/run_dj_set_v2_books.py` | senza flag il runner fa ciò che faceva |
| Cassa 808 | `dj_form.3`: `layers.kick_peak`, `kick_sustain` (diagnostica), `PARAMS['kick_peak_on'] = 0.50`, `active.kick_peak_only` | **acceso** (`kick_peak_on=None` = regola di dj_form.2) |

### Contenuto del piano (campi principali)

- `a_event`: sezione allin1 che contiene la battuta prima dell'uscita (`kind`, `span_bars`; se allin1 manca, l'etichetta di
  `dj_form`, `unknown` senza core), `fate` = `cut_mid_line` se `dj_phrase.exit_point` dice `mid_line` (priorità: il verso
  troncato si sente anche a confine di sezione), altrimenti `completed` (uscita su un confine) o `skipped_rest`;
  `skipped` = resto della sezione + sezioni future non suonate, con conteggi (B&B cap. 13); `line_end` dall'`exit_point`
  (stato, p, ultima parola, coda, pickup di A, `closure_diag` = Krims, **solo diagnostica**); `form_at_exit` da `dj_form`.
- `b_entry`: `audible_sample` **stimato prima del render** (campione sorgente di B: downbeat meno il pickup solo se la
  tecnica lo suona, cioè il cut di `dj_phrase`), `lead_in` (`pickup`, `blend`, `filter_sweep`, `echo_tail`, `build`,
  `break`, `backspin`, `loop`; durata in battute; `gap_beats` dei build), `b_pickup_not_played` quando B ha un pickup che
  la tecnica non suona, `kick_at_entry`.
- `b_events` / `b_target`: eventi di B nelle 16 battute dopo l'ingresso [H]: inizi di sezione allin1 (`chorus`, `verse`,
  `bridge`; gli altri come `section_start`), `drop` di `dj_form` (ritorno della cassa o suo primo ingresso), `first_vocal`
  (pickup/prima parola da `dj_phrase.entry_point`, altrimenti inizio di una regione vocale delle battute). Stessa battuta →
  un evento con le evidenze unite (priorità chorus > drop > verse > bridge > first_vocal). Il **bersaglio** è il primo
  evento saliente **udibile** (un pickup di B non suonato non può esserlo); `status` measured/estimated/unavailable.
- `delay`: `bars`/`s` da B udibile, `from_downbeat_bars`, `from_switch_bars` (switch = fine del blend, altrimenti il
  downbeat d'ingresso), `a_audible_until_sample` (blend: + N battute; coda d'eco esclusa, dal render). È una **descrizione**:
  nessun valore «buono» (specifica, regola 2; Huron pp. 314–316 non dà soglie).
- `alternatives`: `arrival` (stessa uscita, altri ingressi), `exit`, `cue`, `duration` (blend più corti dell'albero o
  `not_applicable`), `technique` (decisione dell'albero e fallback con il loro costo, proposte della policy con il motivo per
  cui non sono usate), `continue_A` (solo riferimento, mai clip da votare 1–10); ogni riga ha costo, Δ costo, destino di A,
  bersaglio e ritardo, e `why_not`. `candidates_table` conserva **tutti** i candidati della coppia (fino a 400) con le stesse
  colonne, anche i non ammissibili (regola 4).
- `trace`: versioni (`dj_events.1`, `dj_select.1`, `dj_set.1`, `dj_form.3`, `dj_phrase.1`, `dj_policy.1`; `dj_qa`/`dj_critic`
  quando la transizione è renderizzata), moduli eseguiti, finestre dei cue, profilo, sha256 delle analisi.

### Opzione sperimentale [H] e fonti (pagine rilette il 21/10/2026 con `tools/libri.py`)

Con `w_event_arrival > 0` un candidato costa `w` in meno nel Viterbi se **A completa la sua sezione** (`fate = completed`) e
**l'evento obiettivo di B arriva entro `event_arrival_max_bars` (2) dallo switch** (0 = sullo switch). Con `w = 0,05` cambia
solo scelte la cui differenza di costo è < 0,05 («a parità di costo tecnico»). Contesto, non soglie:

- [P] Brewster & Broughton 2003, cap. 13, 'Even tighter' (EPUB, unità 33): «mix straight after a chorus. You time it so that
  the big opening moment in record 2 happens immediately after the chorus in record one», «very effective with breakbeat
  genres (hip hop ...)», di solito con un cut.
- [A] Butler 2006 p. 246: la cassa che torna, «dropping the beat», anche «when a new record enters the mix»; [P] p. 247
  (Mystik: la cassa tolta per quattro battiti e riportata «on beat»).
- Nessuno dei libri dà un numero di battute: «entro 1–2 battute» è nostro e va stimato con confronti a coppie (P1.1).

## 2. Run di sola pianificazione sul catalogo congelato

`runs/dj_set_v3_events_plan/set.json` (24 s, nessun render): `/opt/musicmix/venv/bin/python tools/run_dj_set_v3_bridge.py
set --plan-only --event-plan --out runs/dj_set_v3_events_plan`. Ingressi letti dalla run congelata `runs/dj_set_v3_bridge`
(catalogo, analisi, modelli `dj_phrase`), forme in `.cache/dj_set_v3_events_plan/form`. Confronti nella stessa run:
`compare_kick_legacy/` (forme con la regola della cassa di dj_form.2) e `compare_event_pref/` (opzione sperimentale accesa).
Tecnica «pianificata» = quella che il set prova per prima (policy dei libri se è una regola di contesto giuntabile,
altrimenti l'albero), come `attempts_v3` (test). La run congelata ha accettato il primo tentativo in tutte le 13
transizioni, quindi la tecnica pianificata coincide con quella del set quando gli ingressi coincidono.

**Riproduzione del set congelato.** Con la regola della cassa di dj_form.2 (`compare_kick_legacy`): stesso ordine, stesso
costo d'ordine (24,068), **13/13 cue e 13/13 tecniche** uguali al set v3. Con dj_form.3: stesso ordine e 13/13 cue, 12/13
tecniche (T13, sotto).

| # | A → B (cue A/B) | evento di A: sezione, destino, verso | lead-in di B | evento obiettivo di B | ritardo da B udibile / dallo switch (battute) | tecnica pianificata | [H] preferito | alternative di arrivo (stessa uscita, Δ costo) |
|---|---|---|---|---|---|---|---|---|
| 1 | After Hours → MAFIA (64/54) | strofa [56,64) completata; `gap_not_line_end` («darkest») | eco 1 b. | strofa @54 | 0 / 0 | echo out (R6) | sì | B19 ritornello −0,05; B26 strofa −0,05 |
| 2 | MAFIA → No Kizzy (86/42) | bridge [70,86) **verso troncato** (`mid_line`) | build 2 b. + 1 battito di pausa | drop @42 (= strofa) | 0 / 0 | build verso ritornello/drop (C2) | no | B8 strofa −0,15; B23 strofa −0,15 |
| 3 | No Kizzy → Goosebumps (82/10) | ritornello [66,82) completato; `clean` | filtro 2 b.; **pickup di B non suonato** | strofa @10 | 0 / 0 | filter out (R4) | sì | B24 strofa 0; B48 strofa +0,05 |
| 4 | Goosebumps → 5% TINT (48/0) | ritornello [41,48) completato; `a_pickup` | blend 4 b. | strofa @8 | 8 / 4 | blend 4 (R8) | no | B20 ritornello (eco) −0,32; B52 ritornello −0,27 |
| 5 | 5% TINT → Creature (40/48) | ritornello [20,40) completato; `a_pickup` | build 2 b. + 1 battito | drop @48 (= strofa) | 0 / 0 | build (C2) | sì | B8 ritornello −0,05; B15 ritornello +0,05 |
| 6 | Creature → BUTTERFLY EFFECT (88/20) | ritornello [80,88) **verso troncato** | halving build 4 b. + 1 battito | ritornello @20 | 0 / 0 | halving build (C2b) | no | B0/B4/B12 strofa −0,05 |
| 7 | BUTTERFLY EFFECT → Knife Talk (40/24) | ritornello [36,40) **verso troncato** | build 2 b. + 1 battito | drop @24 (= strofa) | 0 / 0 | build (C2) | no | B10 drop 0; B28 drop +0,05 |
| 8 | Knife Talk → STARGAZING (44/18) | ritornello [40,46): **resto saltato**; coda protetta («bang,») | eco 1 b. | strofa @18 | 0 / 0 | echo out (R4) | no | B2 drop −0,10; B0 prima voce −0,02 |
| 9 | STARGAZING → A Lot (50/48) | break [42,50) completato; `clean` | filtro 2 b.; **pickup di B non suonato** | strofa @48 | 0 / 0 | filter out (R4) | sì | B16/B24/B32 strofa 0 |
| 10 | A Lot → Nonstop (88/13) | ritornello [80,88) completato; `a_pickup` | eco 1 b. | strofa @13 | 0 / 0 | echo out (R4) | sì | B8 drop −0,10; B19 drop −0,10 |
| 11 | Nonstop → Sicko Mode (35/31) | inst [31,35) completato; `gap_not_line_end` | eco 1 b.; **pickup di B non suonato** («So I said») | strofa @31 | 0 / 0 | echo out (R4) | sì | B8 ritornello 0; B20 strofa 0 |
| 12 | Sicko Mode → Rich Flex (60/25) | ritornello [49,60) completato; `a_pickup` | build 2 b. + 1 battito; **pickup di B non suonato** | drop @25 (= strofa) | 0 / 0 | build (C2) | sì | B17 drop −0,05; B8 ritornello +0,05 |
| 13 | Rich Flex → God's Plan (48/27) | ritornello [40,48) completato; coda protetta («Man,») | halving build 8 b. + 1 battito | drop @27 (808 dopo 12 battute senza, = strofa) | 0 / 0 | **halving build (C2b)**; set congelato: echo out (R1) | sì | B10 drop −0,05; B5 drop +0,05 |

Note: «Δ costo» negativo = alternativa più economica per la coppia ma scartata dal Viterbi per durata del brano, spazio con
le transizioni vicine o sezioni intere (`why_not`). Ogni piano ha anche 3 alternative di uscita, 5 di cue, le tecniche
dell'albero/policy sullo stesso cue e la tabella completa dei candidati.

Osservazioni [M] (da ascoltare, non giudizi):

- In 12 transizioni su 13 l'evento obiettivo di B arriva sullo switch (ritardo 0): il cue di `dj_select` cade quasi sempre
  su un inizio di sezione. Unica eccezione T4 (blend dall'intro di B: strofa 8 battute dopo l'ingresso, 4 dopo la fine del
  blend). Il ritardo, quindi, oggi non varia nel set: per i confronti P1.1 servono le alternative `arrival` (es. T4 B20, T1 B19).
- A è a metà verso in T2, T6, T7 (come nel set congelato) e in T8 esce a metà ritornello (resto saltato). Sono i 4 casi
  «non preferiti» oltre a T4.
- **Nuovo**: in T3, T9, T11, T12 B ha un pickup (il verso comincia prima del suo downbeat) che filter/eco/build non suonano:
  B entra con la prima frase troncata. Solo il cut di `dj_phrase` suona il pickup. Da controllare all'ascolto e nel QA (P0.6).
- **Opzione sperimentale**: 8/13 transizioni sono già «preferite»; con `w = 0,05` **nessuna scelta cambia** (controfattuale
  in `set.json` → `event_pref_counterfactual`, confermato da `compare_event_pref/`: 13/13 cue uguali). Sui dati sintetici
  cambia la scelta come atteso (test). In una versione intermedia della regola dell'808 (finestra su tutta la battuta)
  l'opzione cambiava T7/T8 per una parità esatta del costo del cammino: l'effetto dipende da pareggi, non da differenze
  grandi, come voluto.

## 3. Cassa 808 nell'hip-hop (`dj_form.3`)

**Problema** (P0.2, caso `gods_plan`): la cassa di una battuta era la media, sui quattro battiti, del picco di flusso
d'ampiezza 30/40–120 Hz (drums + bass) nella finestra −40/+60 ms, normalizzata al 95° percentile; attiva se ≥ 0,30 [H].
Un 808 lungo e sincopato (uno o due colpi per battuta, spesso fuori dal battito, coda tonale con glissando; Demucs lo mette
nello stem bass) dà una media 0,2–0,3 anche quando è il ritmo del brano: in `gods_plan` media 0,19–0,30 e massimo 1,0–1,2
nelle battute con l'808 [M].

**Regola** [H]: la battuta ha la cassa anche se il **massimo** dell'attacco basso nella battuta è ≥ `kick_peak_on` = 0,50
dello stesso riferimento. Finestra: dal downbeat − 40 ms all'ultimo battito − 40 ms. Un colpo isolato sull'ultimo battito è
un'anacrusi verso la battuta successiva — [A] Butler 2006 p. 189: le variazioni «almost always occur just before a metrical
or hypermetrical downbeat» — e un attacco pochi ms prima del downbeat successivo appartiene a quel downbeat. Senza questa
finestra la prima versione spostava di una battuta i drop di `18k_no_kizzy` (42→41), `knife_talk` (24→23) e `rich_flex`
(25 → nessuno) e cambiava tre tecniche del set (T2, T7, T12): corretto prima di questa nota, test
`test_lone_808_on_the_last_beat_is_an_anacrusis_not_a_kick_bar`. Il glissando non crea attacchi perché la banda somma
l'ampiezza di tutti i bin. `kick_sustain` (mediana/massimo dell'ampiezza bassa nella battuta) è registrato come
diagnostica, non usato. Limite: una nota di basso tenuta nuova conta come cassa; Butler descrive la cassa della techno,
l'808 è un'estensione nostra. Le forme in cache senza `kick_peak` (prima di .3) mantengono la regola vecchia.

**Effetto sulle 18 forme del catalogo del ponte** (`runs/dj_set_v3_events_plan/kick_808_effect.json`,
`tools/form_kick_808_effect.py`; stessi livelli per battuta, regola vecchia vs nuova) [M]:

| Misura | dj_form.2 | dj_form.3 |
|---|---|---|
| etichette cambiate | — | **473 / 1.665 battute (28 %) in 15 / 18 brani** |
| battute con cassa solo dalla nuova regola | — | 341 |
| brani senza core | 1 (`gods_plan`, `kick_without_full_texture` su questa griglia) | 0: `gods_plan` intro 0–5, core 5–15, breakdown 15–27, core 27–40, outro 40–58 |
| drop | 251 | 166 |
| breakdown | 45 | 17 |

Molti «drop» e breakdown di dj_form.2 erano lo sfarfallio della cassa tra battute (es. `butterfly_effect`: 19 drop, uno ogni
due battute → 0). Non tutti spariscono: in `goosebumps` i drop passano da 38 a 52 (schema dell'808 su due battute; la cassa
per battuta non smussata resta instabile) → il «drop» nell'hip-hop resta una misura debole [aperto].

**Effetto sul set v3** (stesso catalogo congelato, sola pianificazione): ordine invariato (costo 24,068 → 24,468 per i costi
di collocazione formali), 13/13 cue invariati, **una tecnica cambia, spiegata**: T13 Rich Flex → God's Plan passa da echo out
(albero, R1 griglia a bassa confidenza) a **halving build 8 battute (C2b)**, perché B entra a battuta 27, che ora è il ritorno
dell'808 di `gods_plan` dopo 12 battute senza (`b_entry_drop`, regola C2/C2b della policy dei libri, Butler pp. 246–248).
Attenzione: la policy non guarda la confidenza della griglia (R1 dell'albero sì) e la griglia di `rich_flex` è a bassa
confidenza (QA del set congelato: −316 ms); nel set con render il QA/veto deciderebbe. Nessun'altra scelta cambia.

## 4. Test

- `tests/test_dj_events.py` (8): schema completo e serializzabile; fixture (a) uscita a fine ritornello e ritornello
  successivo saltato, (b) pickup di 2 battiti (lead-in, campione udibile, pickup non suonato con l'eco), (c) ritornello 8
  battute dopo l'ingresso (ritardo 8), (d) brano senza core → `unknown`/`no_core`; piani e contesto non cambiano costi né
  scelte; l'opzione sperimentale cambia la scelta come atteso (B entra sulla strofa invece che sull'intro a parità di costo).
- `tests/test_dj_form.py` (+4): 808 lungo con glissando rilevato (la regola vecchia no), cassa in quattro invariata, cache
  senza `kick_peak` → regola vecchia, colpo isolato sull'ultimo battito = anacrusi.
- `tests/test_dj_set_v3_bridge.py` (+1): la tecnica pianificata coincide con il primo tentativo del render.
- Mirati (`test_dj_events`, `test_dj_form`, `test_dj_set`, `test_dj_select`, `test_dj_policy`, `test_dj_books`,
  `test_dj_books_integration`, `test_p0_misure_regole`, `test_dj_phrase_books`, `test_dj_set_v3_bridge`): 148 passati.
  Suite completa (`python -m pytest -q`, 12 min): **673 passati, 0 falliti**.

## 5. Limiti

- Il piano descrive, non valuta: `delay` e `fate` non hanno valori buoni; l'opzione [H] è spenta e non è validata.
- `audible_sample` è stimato prima del render; la verifica sul time-map renderizzato (P0.6) non è collegata qui. Con il
  render (`set --event-plan` senza `--plan-only`) il piano della transizione accettata va in `transitions[].event_plan`:
  questo percorso **non è stato eseguito** (niente render pesanti in questa tranche).
- Eventi di B: allin1, `dj_form` e la voce; nessun «hook» né ingresso di uno strumento oltre la cassa. Bersaglio = primo
  evento saliente udibile: una scelta [H] (alternativa possibile: l'evento più forte entro l'orizzonte).
- `dj_phrase` usa l'ASR `small.en` anche sul brano italiano (`18k_no_kizzy`): i suoi stati di verso sono poco credibili.
- La regola dell'808 è una soglia [H] su 18 brani senza annotazioni di cassa; i drop per battuta restano instabili sugli
  schemi di due battute.
