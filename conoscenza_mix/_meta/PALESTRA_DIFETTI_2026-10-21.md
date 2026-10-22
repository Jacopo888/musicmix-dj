# Palestra automatica degli errori (defect gym v1) — 21/10/2026

Requisito dell'utente (AGENTS.md, "Autonomia"): il sistema non deve aver bisogno dei suoi voti per funzionare; l'utente
non deve mai ascoltare errori che la macchina dovrebbe eliminare da sola. Primo pilastro del piano
[`AUTONOMIA_VALUTAZIONE_2026-10-21.md`](AUTONOMIA_VALUTAZIONE_2026-10-21.md) (strada 3): transizioni di riferimento,
varianti con difetti controllati, QA e critico devono rifiutare le varianti e non i riferimenti. Il gate diventa un
controllo obbligatorio per ogni aggiornamento della pipeline.

Codice: `src/musicmix/defect_gym.py` (operatori, segnali, classificazione, aggregazione, gate),
`tools/defect_gym.py` (`refs | run | report`, ricette di render), `configs/defect_gym_gate.json` (soglie [H]),
`tools/human_effort_report.py` (sforzo umano). Test: `tests/test_defect_gym.py` (rapido, suite normale),
`tests/test_defect_gym_gate.py` (`-m gym`). Run: `runs/defect_gym_v1/` (`refs.json`, `results.jsonl`,
`report.json`, `REPORT.md`), `runs/human_effort_v1/`.

Etichette: **[E]** misura nostra su questi dati; **[P]** pratica DJ dai libri (pagina citata); **[H]** ipotesi o
soglia nostra; **[U]** osservazione dell'utente.

## 1. Metodo

**Riferimenti** (`refs.json`, regole scritte nel file; solo brani en/it ammessi da `musicmix.track_filter`):

| origine | cosa | perché |
|---|---|---|
| `user_vote` | clip con ultimo voto >= 6 in `giudizi_cloud/` (galleria v1: 5; galleria cieca del set v3: 6), rifatte con la ricetta originale e verificate al bit sul WAV o sulla clip tagliata del manifest | transizioni giudicate accettabili dall'ascoltatore [U]; un voto >= 6 non garantisce l'assenza di difetti tecnici |
| `set_v3_qa_pass` | transizioni finali del set v3 con QA `pass` salvato (dj_qa.4) e nessun veto | richiesta dell'orchestratore |
| `set_v3_lead_in` | le stesse transizioni rifatte col runner corrente (`B_ENTRY` acceso) | riferimenti per "pickup tolto": tenute solo se c'è un lead-in di B e QA/veto le accettano |
| `natural` | cut di un brano con se stesso nello stesso punto (continuazione senza giunta), su un inizio frase interno a una sezione, senza parole a cavallo del downbeat e con un respiro >= 1/4 di battito | pulito per costruzione: ogni rifiuto è un falso allarme |
| `natural_blend` | blend 4 battute tra due inizi frase diversi dello stesso brano | sovrapposizione con stessa tonalità e tempo, per i difetti di blend (doppio basso, tonalità) |
| `natural_section` | come `natural` ma su un inizio sezione | solo falsi allarmi: la dinamica propria del brano resta |

Esclusi: le clip `verifica_t01` (voti 6 e 8) perché B è in tedesco (`track_filter`), e il blend di un brano con se
stesso nello stesso punto (somma coerente, +6 dB: non è un modello di transizione; scartato dopo la prima prova).

**Difetti** (`defect_gym.DEFECTS`; base e operatore nel codice). Due famiglie, metadati sempre coerenti con l'audio:

- *a monte* (render rifatto dal renderer vero con l'ingresso alterato, quindi time-map, a_map/b_map e parole sono
  quelli che il renderer scrive): parola mozzata in A (cue di A spostato di k battute intere finché una parola ASR
  attraversa l'uscita secca di A di >= 120 ms per lato [H]); parola mozzata in B (stesso, con le parole raffinate di
  `dj_phrase` all'ingresso udibile di B; lead-in spento, altrimenti la guardia lo ripara); pickup di B tolto (stesso
  render senza il lead-in); sfasamento ½ battito, ¼ battito, 30 ms (griglia di B spostata prima del render: il
  renderer allinea la griglia sbagliata e dichiara un residuo ~0, come con un errore del tracker); B 1 battuta in
  ritardo/anticipo (cue di B ±1 battuta).
- *nei deck* (errore di render, non dichiarato salvo il guadagno di B): basso di A aperto 2 battute dopo lo switch;
  B +3/+6 dB; buco di ½ e 1 battito dallo switch; clic di 1 ms a −20/−6 dBFS; B trasposto ±1 semitono (Rubber Band
  R3, solo con sovrapposizione: nel cut un cambio di tonalità è ammesso, B&B cap. 15 [P]); coda d'eco dell'ultimo
  battito di A con ritardo 0,37 battiti (`dj_effects.echo_out` accetta solo 0,5 o 1).
- Combinazioni (sui naturali): ¼ battito + 3 dB, parola in A + buco ½ battito, B in ritardo + eco fuori tempo,
  30 ms + clic −20 dBFS, doppio basso + tonalità.

**Valutatori**: `dj_qa.evaluate_render` con la configurazione del runner v3 (`QA_CFG`, controllo del buco acceso) e
gli stessi ingressi (`qa_v3`: deck, voci, parole di A e parole raffinate di B), critico v3 `fast` con il blocco v4
`technical` tramite `run_dj_set_v3_bridge.critic` (regole di veto del runner).

**Rilevamento appaiato**: una variante è *rilevata* se compare un segnale di rifiuto (QA `fail` o veto del critico)
assente nel riferimento. `warned` = solo avvisi nuovi; `unavailable` = nessun segnale e controllo atteso non misurato
o inesistente; `masked` = controllo atteso già `fail` nel riferimento (escluso dal denominatore, dichiarato);
`error` = valutazione fallita (nel denominatore). Nessuno di questi è un pass.

## 2. Risultati [E] (run del 21–22/10, `runs/defect_gym_v1/REPORT.md`)

31 riferimenti valutati (11 voto utente, 3 set v3 QA pass, 5 set v3 con lead-in, 6 naturali, 3 naturali di inizio
sezione, 3 blend naturali); 469 righe variante. Tutti i riferimenti rifatti dalle ricette sono **identici al bit** alla
clip salvata (14/14 verificabili); i deck ricostruiscono la clip con errore 0. Codice della pipeline invariato dalla
partenza della run (dj_qa.6 con `b_entry`, runner con `B_ENTRY`).

Rilevamento (pipeline = QA fail o veto del critico del runner; IC95 di Wilson in `REPORT.md`):

| difetto | n | rilevate | warn+ | chi lo vede | nota |
|---|---|---|---|---|---|
| parola mozzata in A | 14 | 100% | 100% | QA `verse_cut` 14/14 | 5 mascherate (riferimento già `verse_cut` fail) |
| parola mozzata in B | 18 | 100% | 100% | QA `b_entry` 18/18 | 4 mascherate |
| pickup di B tolto | 5 | 80% | 80% | QA `b_entry` 4/5 | mancata: lead-in di 33 ms (li_t05), dose sotto la tolleranza di 50 ms |
| sfasamento ½ battito | 19 | **32%** | 84% | QA `rhythm` 4, `b_entry` 2 | 4 mascherate; vedi §4.1 |
| sfasamento ¼ battito | 19 | **37%** | 89% | QA `rhythm` 5 | |
| sfasamento 30 ms | 19 | 16% | 68% | QA `rhythm` 3 | |
| B 1 battuta in ritardo | 21 | **43%** | 48% | `b_entry` 5, loudness 3 (per caso) | nessun controllo di frase: 11 non misurabili |
| B 1 battuta in anticipo | 23 | 43% | 43% | `b_entry` 9 (per caso) | 13 non misurabili |
| basso duplicato | 23 | 83% | 83% | QA `double_bass` 19 | mancate su tecniche creative/echo (§4.4) |
| B +6 dB | 23 | 100% | 100% | QA `loudness_jump` 23, critico 20, v4 23 | |
| B +3 dB | 23 | 57% | 87% | QA 13 | spesso solo `warn` (2–3 dB) |
| buco 1 battito | 23 | **9%** | 87% | QA `dropout` = solo `warn` | §4.3 |
| buco ½ battito | 23 | 13% | 35% | | sotto la soglia di durata del controllo |
| clic −6 / −20 dBFS | 23 | 9% / 0% | 9% / 0% | nessun controllo | §4.5 |
| tonalità ±1 semitono (blend) | 3 | 100% | 100% | critico `chroma_dist` | pochi riferimenti con sovrapposizione |
| eco fuori tempo | 23 | 4% | 4% | nessun controllo | §4.6 |
| combinazioni | 3–9 | 40–100% | | | ¼ battito + 3 dB 100%; parola A + buco 100%; ritardo + eco 44%; 30 ms + clic 40%; doppio basso + tonalità 100% |

Falsi allarmi (rifiuto del riferimento): **naturali 1/6 (17%)** — `nat_nonstop_bar12_cut`, una continuazione senza
giunta, è `fail` per `rhythm` ("B beat grid +21 ms": rumore di Beat This! sulla sola clip contro la soglia di 15 ms
[H]); naturali di inizio sezione 2/3 (ritmo −54 ms su una continuazione; +5,7 dB della dinamica propria del brano,
strofa che entra); **voto utente 5/11 (45%)**: `verse_cut` (dj_qa.5, parola sotto l'uscita a effetto) su g05 libri
(voto 7), g05 default e g10 default (voto 6) e `b_entry` (dj_qa.6) su g09 libri e default (voto 6); set v3 QA pass
3/3 (gli stessi controlli nuovi); blend naturali 3/3 (sezioni diverse dello stesso brano: ritmo, voci, `chroma_dist`).

## 2b. Sforzo umano [E] (`runs/human_effort_v1/`)

| run | azioni | dettaglio | stima |
|---|---|---|---|
| set v3 ponte (13 transizioni) | 54 | 30 voti chiesti (24 dati), 17 giudizi con nota/tag (`punto_frase_sbagliato` 4, `volume_salta` 3, `battute_sfasate` 1), 5 transizioni da rivedere (QA fail al ricontrollo .5), 1 esclusione in attesa d'ascolto (antidote), 1 griglia `low` (rich_flex) | ~540 azioni per 10 set |
| pool ponte (32 richieste) | 20 | 11 download falliti sul PC, 6 in attesa, 1 durata diversa dall'attesa (a_lot), 2 griglie da verificare (rich_flex, antidote) | ~62 azioni per 100 brani (34 download falliti, 19 in attesa) |

Le note dell'utente sul set v3 coincidono con difetti della palestra ("parola tagliata in B", "B tagliato nel secondo
sbagliato", `punto_frase_sbagliato`, `volume_salta`): il fuori frase e lo sfasamento di un ingresso pulito sono anche
i due difetti che QA e critico oggi lasciano passare.

## 3. Gate

`configs/defect_gym_gate.json`, dichiarato prima della run completa: rilevamento >= 0,9 per parola mozzata in A e
in B, sfasamento ½ battito, +6 dB, doppio basso; >= 0,8 per ¼ battito, buco di 1 battito, pickup tolto, tonalità
+1, B in ritardo di una battuta; falsi allarmi <= 0,10 sui naturali e <= 0,34 sui voti utente. Tutte [H]. Lancio:

    /opt/musicmix/venv/bin/python tools/defect_gym.py refs run report          # run completa (ore)
    /opt/musicmix/venv/bin/python -m pytest -m gym tests/test_defect_gym_gate.py # profilo gate in una cartella temporanea
    MUSICMIX_GYM_RUN=runs/defect_gym_v1 /opt/musicmix/venv/bin/python -m pytest -m gym tests/test_defect_gym_gate.py

Stato attuale: **FAIL**. Passano parola mozzata in A e in B, +6 dB, pickup tolto (0,80, n = 5), tonalità (n = 3);
falliscono sfasamento ½ e ¼ battito, basso duplicato (0,83), buco di 1 battito, B in ritardo di una battuta, falsi
allarmi sui naturali (0,17) e sui voti utente (0,45). Il test `-m gym` è quindi rosso finché le correzioni del §4 non
entrano in `dj_qa`/critico: è il lavoro prioritario successivo, non un difetto della palestra.

Emendamento dichiarato (dopo la prima run): i riferimenti `set_v3_lead_in` erano tenuti solo se accettati da QA e veto
(n = 2 < min_n); ora basta il lead-in dichiarato, il rifiuto del riferimento è contato come falso allarme e vale la
regola appaiata. Nessuna soglia del gate è stata cambiata dopo aver visto i risultati.

## 4. Difetti che QA e critico non rilevano oggi (proposte, non implementate: moduli di altri agenti)

1. **Sfasamento di B negli ingressi puliti** (echo/filter/build: 13 dei 19 riferimenti). Il QA misura lo scarto della
   griglia di B ("mix-only: B beat grid +186…+419 ms from A's grid") ma lo dichiara solo `warn`; il veto di fase del
   critico è spento perché il render dichiara fase 0 (la griglia sbagliata è quella che il render ha usato). Esempi:
   `uv_g1_g01` ½ battito → solo warn (+186,6 ms); `v3_t06` → warn (+419 ms); `uv_g1_g02` blend 8 → nessun segnale
   (Beat This! non accoppia i downbeat). Proposta: `fail` quando lo scarto mix-only supera ¼ di battito (o 40 ms) e la
   misura è coerente su >= 4 battiti di B, anche senza sovrapposizione; nel critico non sopprimere il veto di fase
   quando la misura sulla clip supera ¼ di battito *e* contraddice la fase dichiarata (la dichiarazione vale solo
   quanto la griglia d'analisi).
2. **Fuori frase** (B ±1 battuta): nessun controllo legge la posizione di frase di B. Proposta: `phrase_entry` nel QA
   o nel planner: il campione di B all'ingresso udibile deve cadere su un inizio frase della tabella `phrases` di
   dj_select (o di dj_phrase) e l'uscita di A su un confine di frase; `warn` a ±1 battuta, `fail` se la frase di B
   inizia dopo l'ingresso con voce.
3. **Buco non dichiarato**: il controllo D4 esiste ma è `warn`. Proposta: `fail` da 1 battito non dichiarato (le pause
   volute sono già esentate da `intended_gap`).
4. **Basso duplicato** non misurato nelle tecniche creative senza sovrapposizione (`qa_inputs` toglie il deck B) e
   mancato su `uv_v3_g05_default` (echo). Proposta: misurare il doppio basso sul mix (banda < 200 Hz: energia di A
   dopo lo switch prevista dai metadati) quando manca un deck.
5. **Clic**: nessun controllo. Proposta: picco della derivata seconda / energia > 5 kHz nei 5 ms attorno a switch e
   bordi dei segmenti, rispetto alla distribuzione locale degli attacchi.
6. **Eco fuori tempo**: nessun controllo. Proposta: autocorrelazione dell'inviluppo della coda dopo lo switch; il
   ritardo dominante deve essere 1/2, 1 battito (o un multiplo) entro 10 ms.
7. **Falsi allarmi**: soglia `rhythm` di 15 ms contro rumore di Beat This! di 21–54 ms sulle continuazioni senza giunta
   (nat_nonstop, nsec_superhero): usare l'IC della retta o una soglia per la via mix-only; `loudness_jump` scatta sulla
   dinamica propria del brano all'inizio sezione (+5,7 dB, gods_plan); `verse_cut`/`b_entry` respingono 5 clip votate
   6–7: da decidere con l'utente se una parola sotto la coda d'eco è accettabile (severità `warn` per l'uscita a
   effetto) — non tarare sui singoli voti.

## 5. Limiti

- **Riconoscere un peggioramento non equivale a riconoscere una bella transizione.** Il gate dice che QA e critico
  vedono i difetti di mestiere elencati; non dice nulla sul resto (scelta del brano, energia, gusto).
- I difetti sono simulati: un difetto reale diverso (es. un'eco mal tagliata dal renderer vero) può sfuggire.
- I naturali sono un brano con se stesso: puliti per costruzione, ma non transizioni tra due brani; sul rap denso
  pochi downbeat interni sono liberi da parole (6 brani su 19 hanno un inizio frase utilizzabile).
- I riferimenti `user_vote` sono di un solo ascoltatore e alcuni non ciechi; non sono usati per tarare nulla.
- Le soglie del gate sono [H]; nessuna è stimata su voti.

## 6. v2 (22/10/2026): `dj_qa.7` chiude i buchi del §4 e la palestra viene rifatta

Codice: `src/musicmix/dj_qa.py` (`musicmix.dj_qa.7`, changelog nel docstring), `src/musicmix/dj_critic_books.py`
(`musicmix.dj_critic_books.3`), `src/musicmix/dj_creative.py` (`qa_inputs`), `src/musicmix/defect_gym.py`
(`musicmix.defect_gym.2`, dose minima del basso duplicato), `tools/defect_gym.py` (misure compatte nelle righe, metà di
sviluppo/verifica, tabella prima/dopo), `tools/calibrate_qa_noise.py` (nuovo), `tools/recheck_set_v3_qa.py` (`--out`,
`--previous`). Test: `tests/test_dj_qa_v7.py` (nuovo), `tests/test_dj_qa.py`, `tests/test_dj_books_integration.py`
(aggiornati alla regola nuova). Run: `runs/defect_gym_v2/` (`calibration.json`, `calibration_points.jsonl`, `split.json`,
`results.jsonl`, `report.json`, `REPORT.md`), `runs/dj_set_v3_bridge/qa_v7_recheck.json`.

### 6.1 Calibrazione sul rumore delle giunte naturali [E]

`tools/calibrate_qa_noise.py`: 196 clip di giunte naturali (un brano che continua sé stesso: cut, echo out 1 battuta,
filter out 2 battute dallo stesso inizio frase; blend di 4 battute tra due inizi frase dello stesso brano) su **14 brani
disgiunti** dai riferimenti naturali della palestra (i 7 brani `nat_*`/`nsec_*`/`nbl_*` esclusi del tutto); nessun voto
letto. Esclusi dal calcolo delle soglie i brani la cui scheda chiede di ascoltare la griglia con gravità >= 0,3
(`rich_flex`: griglia `low` e cambio di beat, una continuazione lì misura +213 ms; `antidote`: ottava sospetta): restano
168 clip. Regole scritte nel tool (`RULES`) prima della run della palestra; la prima (1,5 × p95, minimo 40 ms) è stata
sostituita dopo aver visto **solo** i punti di calibrazione, perché il p95 (23 ms) non copriva la coda (−40 ms costanti su
`creature`, tracciato a mezzo tempo).

| misura | rumore naturale | soglia `dj_qa.7` |
|---|---|---|
| fase di B a un ingresso pulito (`beat_this_entry`, 75 misure affidabili su 144) | p50 6,5 ms, p95 22,9 ms, max 39,7 ms | **60 ms** (1,5 × max) [E]; 1/4 di battito va da 103 a 213 ms nel catalogo |
| stessa misura nelle sovrapposizioni (`beat_this_overlap_lines`, 15 affidabili su 24 blend) | max 23 ms | 60 ms (stessa regola) |
| xcorr del flusso tra i deck nei blend | max 6,7 ms | 15 ms dell'utente, invariata |
| Beat This! deck-vs-deck nei blend (accoppiamento dei downbeat) | mediana dell'errore p95 94 ms, max 254 ms | **solo `warn`**: non distingue un blend corretto da uno sfasato |
| silenzio dal lato di B (cut, dopo l'ingresso pulito, sovrapposizione) | max 0,12 battiti | 0,5 battiti (invariata) |
| silenzio dentro l'effetto d'uscita (eco, filtro) | max 1,06 battiti | 1,5 battiti [H] (>= max + 0,25) |
| clic (energia 20,5–23,5 kHz alla giunta / p99,9 locale) | max 4,45 | fail 6,7 (1,5 × max), warn 4,4 |
| eco: progressioni fuori griglia di copie di A | 4 `fail` + 2 su 168 (B = A ripete il proprio materiale) | resta **warn** |

### 6.2 Cosa cambia per ciascun buco del §4

1. **Sfasamento negli ingressi puliti.** Nuova misura `beat_this_entry` (`dj_qa._rhythm_entry`): rette di battito di Beat
   This! sul deck A prima dell'uscita e sul deck B dopo il suo downbeat (sulla clip se mancano i deck), lette ciascuna
   nel *proprio* battito dichiarato dal render; l'errore è la fase di B allo switch meno quella di A (il bias del
   tracker su quella clip). Conta solo se **affidabile**: >= 6 battiti per lato, rette diritte (rms <= 20 ms, al più 1/4
   di battiti fuori retta), consenso tra le due metà di ciascuna finestra (<= 25 ms), un secondo rivelatore indipendente
   (fase del flusso d'onset al battito dichiarato) concorde entro 0,2 battiti, Beat This! non fuori griglia nell'analisi
   di un brano la cui griglia è confermata da allin1/madmom. Allora sopra 60 ms è `fail`, **qualunque fase dichiari il
   render**; altrimenti il ritmo è `unavailable` (warn, mai pass). Sostituisce la via mix-only per tutti gli ingressi
   senza sovrapposizione ritmica (cut compreso). Il critico v4 (`dj_critic_books.3`) mette il veto `qa_entry_phase_ms`
   quando questa misura è `fail` (la fase dichiarata non la sopprime più); il veto del runner v3 non è stato toccato
   (file di un altro agente): nella pipeline il rifiuto arriva dal QA. L'ambiguità di fase del tracker (warn) ora
   richiede anche che Beat This! sia già fuori griglia nell'analisi di un brano (all'ottava diversa una coincidenza di
   0,5 non è "fuori griglia"): se Beat This! concorda con le griglie, un errore di mezzo battito sulla clip è vero.
2. **Falsi allarmi del ritmo sulle continuazioni.** La soglia di 15 ms nel cut era sotto il rumore (21–54 ms): ora 60 ms
   con le regole di affidabilità sopra; nei blend la xcorr resta a 15 ms (rumore 6,7 ms), l'accoppiamento deck-vs-deck di
   Beat This! dà al più `warn`, e si aggiunge la stessa misura a rette (`beat_this_overlap_lines`).
3. **B fuori frase.** Nuovo controllo richiesto `phrase` (`dj_qa.check_phrase`): il cue di B (`cue_in_b_s`) deve cadere su
   un inizio frase della tabella `phrases` dell'analisi usata dal render (`phrase_level` >= 4 o inizio sezione: la stessa
   definizione dei cue del pianificatore; in mancanza le `entries` della scheda brano con lo stesso SHA-256). Fuori
   griglia = `fail` [P] B&B cap. 10, Butler p. 185. `self_supervision.cue_naturalness` (solo dalla cache, mai calcolato) è
   una seconda opinione: se giudica il cue naturale quanto l'inizio frase vicino (p >= 0,5) il risultato scende a `warn`
   (griglia di frase forse sbagliata lì); senza cache è `unavailable` e non pesa. Uno spostamento dichiarato
   (`meta['b_entry']['relocated_bars']`) è `warn`. Posizione dell'uscita di A solo riportata.
4. **Buco non dichiarato.** `dropout` acceso di default e `fail` (era `warn`): lato B (cut ±1 s, da 0,1 s prima
   dell'ingresso pulito, sovrapposizione ±1 s) da 0,5 battiti; dentro l'effetto d'uscita da 1,5 battiti oltre la pausa
   dichiarata (prima una pausa dichiarata esentava anche un buco più lungo).
5. **Basso duplicato nelle tecniche creative.** `dj_creative.qa_inputs` passa sempre il deck B (halving build, break). Le
   tre "mancate" della v1 su `uv_v3_g05_*` e `uv_g1_g01` non erano errori del QA: la banda < 200 Hz di A dopo il cue è
   16–45 dB sotto quella di B (A senza basso lì), quindi la variante non conteneva un doppio basso. Emendamento della
   palestra (registrato in `configs/defect_gym_gate.json`, dichiarato prima della run v2): variante non generata se la
   dose iniettata è più di 12 dB sotto B [H].
6. **Clic ed eco fuori tempo.** `click`: rapporto tra l'energia 20,5–23,5 kHz nei ±5 ms di ogni giunta dichiarata (switch,
   inizio/fine transizione, fine del dry, ingresso udibile di B, bordi di `a_map`/`b_map`) e il p99,9 della stessa banda
   nel secondo attorno (le sorgenti codificate non hanno quasi nulla lassù) [H sul metodo, E sulla soglia]. `echo_tempo`:
   filtro adattato con i tre transienti più forti dell'ultimo battito udibile di A; >= 3 copie consecutive in
   progressione aritmetica che partono entro un battito dall'uscita devono avere passo multiplo di 1/4 o 1/3 di battito
   entro 10 ms (proposta §4.6); `warn` perché sulle continuazioni naturali compaiono progressioni spurie.
7. **Falsi allarmi su clip votate >= 6.** Analisi solo sulla metà di sviluppo (§6.3). `uv_v3_g05_default_echo_out`:
   «me» (confidenza ASR 0,998) finisce 88 ms dopo lo stop del dry; lo stem sorgente di A resta a −16…−19 dB per altri
   ~100 ms e poi cade a −31 dB: la parola è davvero troncata (l'eco ne ripete solo l'inizio). `uv_v3_g05_books_build_into_chorus`:
   lo stutter riparte a 11,929 s dentro «hear» (fine ASR 11,99 s, confidenza 0,53) mentre lo stem sorgente di A resta
   continuo (−14…−17 dB per 0,3 s, «hear me»): la voce è interrotta a metà. Nessun errore di misura dimostrato: le
   soglie non sono state toccate. Sulla metà di verifica (`uv_v3_g09_*`, `uv_v3_g10_default`) non si è guardato nulla
   prima del report; il «Pin a…» di g09 coincide con la diagnosi indipendente di INGRESSO_B (T9: frase di B iniziata prima).

### 6.3 Metà di sviluppo e di verifica

`runs/defect_gym_v2/split.json`, dichiarato prima di guardare i dati per le correzioni di misura: gruppi = coppia non
ordinata {A, B} (brano per i naturali), dal più grande, ciascuno nella metà che rende più simili i conteggi per origine
(sha256 come spareggio). Sviluppo: 25 riferimenti (6 voto utente: g01, g04 ×2, g05 creature, `uv_v3_g05` ×2); verifica:
24 (5 voto utente: g02, g07, g09 ×2, g10). Il gate resta calcolato su tutti i riferimenti con le soglie del 21/10.

### 6.4 Risultati della palestra v2 [E] (`runs/defect_gym_v2/REPORT.md`)

Stessi 31 riferimenti valutati e stesse 469 righe variante della v1 (ricette riusate da `refs.json`, render rifatti,
nessun WAV salvato), tutte valutate con `musicmix.dj_qa.7` (384 righe valutate; le altre non generabili, come in v1).

| difetto (pipeline) | v1 | v2 | soglia | nota |
|---|---|---|---|---|
| parola mozzata in A | 100% (14) | 100% (14) | 0,9 | |
| parola mozzata in B | 100% (18) | 100% (22) | 0,9 | |
| sfasamento ½ battito | 32% (19) | **39% (23)** | 0,9 | fail: 9 rilevate, 4 non misurabili, 7 solo avviso "non misurabile", 3 mancate (blend) |
| sfasamento ¼ battito | 37% (19) | **48% (23)** | 0,8 | fail: 11 rilevate, 4 non misurabili, 6 solo avviso, 2 mancate |
| sfasamento 30 ms | 16% | 9% | — | sotto il rumore (60 ms): atteso |
| B 1 battuta in ritardo | 43% (21) | **81% (21)** | 0,8 | pass (`phrase`) |
| B 1 battuta in anticipo | 43% | 91% | — | |
| basso duplicato | 83% (23) | **95% (20)** | 0,9 | pass (3 varianti senza dose non generate) |
| buco 1 battito | 9% | **100%** | 0,8 | pass |
| buco ½ battito | 13% | 9% | — | sotto la soglia di 0,5 battiti dopo le rampe |
| +6 dB / +3 dB | 100% / 57% | 100% / 57% | 0,9 / — | invariati |
| pickup tolto | 80% | 80% | 0,8 | invariato (li_t05, lead-in 33 ms) |
| tonalità ±1 (blend) | 100% | 100% | 0,8 | |
| clic −6 / −20 dBFS | 9% / 0% | **78% / 35%** | — | |
| eco fuori tempo | 4% | 0% rifiuti, 13% avvisi | — | il controllo è `warn` per scelta (§6.1) |

Falsi allarmi (rifiuto del riferimento):

| gruppo | n | v1 | v2 | chi |
|---|---|---|---|---|
| naturali | 6 | 17% | **0%** | — (soglia 0,10: pass) |
| naturali di inizio sezione | 3 | 67% | 33% | `nsec_gods_plan` loudness +5,7 dB (dinamica del brano) |
| blend naturali | 3 | 100% | 67% | non più ritmo: `vocal_clash` e `chroma_dist` (due sezioni dello stesso brano con voci sovrapposte) |
| voto utente | 11 | 45% | **45%** | invariato: `verse_cut` g05 ×2, g10; `b_entry` g09 ×2 (soglia 0,34: fail) |
| set v3 QA pass / con lead-in | 3 / 5 | 100% / 60% | 100% / 40% | stessi controlli `verse_cut`/`b_entry` |
| sviluppo (tutti) / verifica (tutti) | 17 / 14 | — | 41% / 43% | voto utente: sviluppo 2/6 (33%), verifica 3/5 (60%) |

Gate (`configs/defect_gym_gate.json`, soglie del 21/10 invariate; `MUSICMIX_GYM_RUN=runs/defect_gym_v2 pytest -m gym
tests/test_defect_gym_gate.py`): **FAIL** su tre regole su dodici: sfasamento ½ battito (0,39), ¼ battito (0,48), falsi
allarmi sui voti utente (0,45). Passano parola in A e B, +6 dB, basso duplicato, buco di 1 battito, pickup, tonalità, B in
ritardo di una battuta, falsi allarmi sui naturali. Per metà: sviluppo ½/¼ battito 58%/67%, verifica 18%/27% (i naturali
di `sicko_mode`, `stargazing`, `5_percent_tint` e i blend stanno nella verifica).

Perché lo sfasamento resta sotto soglia (righe in `results.jsonl`, campo `details.rhythm`): dove la misura è affidabile
il rilevamento è completo (tutte le echo/filter/build con griglie stabili: offset misurati 86–386 ms contro 1–25 ms del
riferimento), salvo `v3_t12` ½ battito (B tracciato al doppio: modulo mezzo battito, invisibile per costruzione). Le
mancate sono misure dichiarate **non affidabili**: Beat This! su `sicko_mode`, `stargazing`, `butterfly_effect`,
`5_percent_tint` e `a_lot` (lato A di g09) dà rette storte o un battito diverso da quello dichiarato del 6–10%, oppure i
due rivelatori non concordano; nei blend la misura a rette non arriva a una decisione e l'accoppiamento deck-vs-deck di
Beat This! è declassato a `warn` perché sui blend naturali sbaglia fino a 250 ms. Abbassare le richieste di affidabilità
porterebbe falsi allarmi (il rumore non affidabile arriva a 400 ms sulle continuazioni naturali): il passo successivo è
un secondo tracker indipendente (madmom/allin1 sulla clip, già usati nell'analisi) o il downbeat (Beat This!) per i lati
tracciati al doppio, non una soglia più bassa.

Perché restano i falsi allarmi sui voti utente: nessun errore di misura dimostrato sulla metà di sviluppo (§6.2, punto 7):
le parole sono davvero troncate. La regola del gate (al più 1/3 delle clip votate >= 6 rifiutata) resta; decidere se una
parola troncata sotto un'eco o uno stutter sia accettabile è una scelta dell'utente (ascolto delle 5 clip con le misure),
non una soglia da spostare qui.

### 6.5 Ricontrollo del set v3 con `dj_qa.7` (`runs/dj_set_v3_bridge/qa_v7_recheck.json`)

`tools/recheck_set_v3_qa.py --out runs/dj_set_v3_bridge/qa_v7_recheck.json --previous runs/dj_set_v3_bridge/qa_v5_recheck.json`:
13/13 PCM identici alla run congelata (non toccata). Verdetti: salvato (.4) 4 pass / 9 warn; ricontrollo precedente (.5)
1 pass / 7 warn / 5 fail; oggi (.7) 1 pass / 4 warn / **8 fail**. Cambiano 5 verdetti su 13 rispetto al .5: T1, T3, T11
diventano `fail` per `b_entry` (controllo del .6 sullo stem di B, il runner del ricontrollo spegne le parole raffinate
come nella run congelata); T4 (blend) `warn` → `pass` (l'accoppiamento deck-vs-deck di Beat This! non è più un motivo da
solo); T6 `pass` → `warn` (fase dell'ingresso non misurabile: Beat This! fuori retta sul lato A). Novità del .7 nei motivi
anche dove il verdetto non cambia: T13 fase +167 ms misurata in modo affidabile (A = `rich_flex`, griglia `low`); T10
eco a 0,424 battiti (avviso). Critico v4 blocco technical: 10 veto / 3 ok salvati → 12 ok / 1 veto (T13, `qa_entry_phase_ms`).

### 6.6 Limiti v2

- Gate FAIL su tre regole; nessuna soglia del gate è stata cambiata. Emendamenti solo della palestra (dose del basso
  duplicato) e del report (metà di sviluppo/verifica), registrati in `configs/defect_gym_gate.json`.
- La regola di calibrazione della fase è stata riscritta una volta dopo aver visto i soli punti di calibrazione (§6.1).
- `phrase` si fida della tabella di frase di `dj_select` (ancorata alle sezioni allin1); una tabella sbagliata dà falsi
  rifiuti. Lo spostamento di 1–2 battute di INGRESSO_B (R-B2, «battuta prima») è proprio un ingresso fuori frase: oggi il
  render non lo dichiara (`meta['b_entry']['relocated_bars']` va scritto dal runner, file di un altro agente), quindi nei
  set futuri quei cue risulteranno `fail` e verranno ripianificati. Va deciso con l'utente quale regola prevale.
  **Aggiornamento 22/10 (set v4b):** il runner v3 ora dichiara lo spostamento (`declare_entry_relocation`), quindi R-B2 dà
  `warn` dichiarato (T9 del set v4); quale regola prevalga resta una scelta dell'utente
  ([TENSIONE_RISOLUZIONE](TENSIONE_RISOLUZIONE_2026-10-22.md) §6c).
- `click` misura sopra 20,5 kHz: con sorgenti a banda piena o a 44,1 kHz la statistica perde senso (oltre 0,5·sr è
  `unavailable`); i bordi dei loop di `dj_creative` (rampe di 2 ms) arrivano a ~3, sotto la soglia.
- `echo_tempo` resta un avviso; il filtro adattato usa i transienti dell'ultimo battito di A e non vede un'eco della
  sola voce.
- Riconoscere un peggioramento non equivale a riconoscere una bella transizione (invariato).

Test (22/10/2026): mirati `tests/test_dj_qa.py`, `test_dj_qa_v7.py`, `test_defect_gym.py`, `test_dj_critic_books.py`,
`test_dj_books_integration.py`, `test_dj_qa_time_map.py`, `test_dj_creative.py` passati; suite completa **802 passati,
1 saltato** (18 min 51 s); `-m gym` su `runs/defect_gym_v2`: fallito come atteso (gate FAIL sulle tre regole sopra).
