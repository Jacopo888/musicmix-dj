# Carica di A, risoluzione di B, base continua, tonalità e testo (22/10/2026)

Richiesta: i criteri ricorrenti nei voti ciechi dell'utente (torneo dei cue `giudizi_cloud/b_entry_fix_v1`, galleria del set v3
`giudizi_cloud/dj_set_v3_bridge`) diventano punteggi **spiegabili** nel pianificatore, con pesi fissati prima di guardare i
voti e un controllo esterno dichiarato. Codice `src/musicmix/dj_tension.py` (`musicmix.dj_tension.1`), runner
`tools/run_dj_tension_v1.py` (pre-registrazione e controllo esterno), hook `SetHooks.tension` in
`tools/run_dj_set_v2_books.py`, opzione `--tension {off,report,prefer}` di `tools/run_dj_set_v3_bridge.py`, set
`tools/run_dj_set_v4.py` (run `runs/dj_set_v4/`). Test `tests/test_dj_tension.py`.

Legenda: **[E]** risultato sperimentale di una fonte; **[T]** teoria; **[A]** analisi di un autore; **[P]** pratica DJ;
**[H]** scelta/soglia nostra; **[M]** misura nostra; **[user]** osservazione dell'utente nei voti ciechi. I voti sono
**specifica** (che cosa cercare) e **controllo esterno** (§5): nessun peso è stimato su di essi (AGENTS.md).

## 1. Criteri dell'utente [user] e loro traduzione

| criterio nelle note | punteggio | che cosa NON fa |
|---|---|---|
| «A deve caricare (alzare la tensione)»; «A tagliata al minutaggio sbagliato / campionata all'inizio, manca la carica» | `a_charge(track, t, played_s)` | non equipara carica a volume: l'energia è una delle quattro componenti |
| «B deve risolvere»; 5% TINT→Creature «un 'ouuuu' all'inizio e poi parte la rappata hard» (entrata su strofa) | `b_resolution(track, t_udibile, t_downbeat, A)` | non usa «ritornello = risoluzione» (smentito in [SUPERVISIONE_DAI_BRANI](SUPERVISIONE_DAI_BRANI_2026-10-21.md) §6) |
| «la base non deve interrompersi tra A e B» | `continuity(decisione)` (pianificata) e `continuity_measured(audio, ...)` (sul render) | non dice che un blend è sempre meglio: è una preferenza dichiarata, pesata |
| «tonalità discordante» anche senza sovrapposizione (autotune) | `harmonic_clash(A, tA, B, tB, decisione, shift_b)` | non vieta: aggiunge un costo, maggiore con la sovrapposizione |
| testi/suoni affini tra fine di A e inizio di B (Butterfly Effect→Knife Talk, 'k'/'k') | `lyric_affinity(parole_fine_A, parole_inizio_B)` | nessuna semantica (non disponibile) |

## 2. Definizioni (tutti i punteggi in [0, 1], con componenti, pesi e motivi; `None` = astensione)

**Carica di A all'uscita** `a_charge` = media pesata di quattro componenti × (0,4 + 0,6 × minutaggio) [H]:
- `model` 0,30: `self_supervision.tension_at_exit` (contesto **prima** di t, DSP + forma + croma) come percentile del
  downbeat nel brano; per i 18 brani del dataset dal fold che non ha visto l'artista (`runs/insong_v1/cue_scores`), per
  `reminder`, `too_many_nights`, `life_is_good` dal modello su tutti i brani (non out-of-fold, dichiarato). [M] AUC 0,69–0,75
  contro strofa/intro su artisti tenuti fuori.
- `energy` 0,20: pendenza dell'energia del mix nelle ultime 4 battute (dB/battuta, logistica 1,5) e livello delle ultime 2
  sulla mediana del brano (logistica 3 dB). [T] Huron pdf p. 337 (climax = tensione accresciuta); la sola energia non è
  tensione (AGENTS.md), per questo pesa 0,2.
- `density` 0,15: pendenza del flusso spettrale (densità d'attacchi) nelle ultime 4 battute e parole per battuta nelle
  ultime 2 contro le 4 prima. [A] Butler pdf p. 238 (il buildup aumenta l'intensità ispessendo la tessitura e riempiendo
  posizioni ritmiche); Krims pdf p. 64 (stili «effusive»); Ohriner pdf p. 91 (accento come intensificazione, Roeder).
- `form` 0,35: posizione nella forma di A (sezioni allin1 + etichette `dj_form`), tabella [H] `FORM_TABLE`: fine della
  sezione che precede un ritornello 1,0 ([T] Huron pdf p. 26: la risposta di tensione è legata al periodo immediatamente
  prima dell'esito atteso: B prende il posto dell'arrivo); fine di build/bridge/strumentale 0,95 (Butler p. 238); a 1–2
  battute dal ritornello 0,8; fine del ritornello completato 0,75 ([P] B&B pdf p. 33 «mix straight after a chorus»); fine
  strofa 0,55; dentro il ritornello 0,5; dentro altra sezione 0,4; metà strofa 0,35; prime 4 battute di strofa 0,15; outro
  0,1; intro 0,05.
- `minutage`: credito di posizione 0 prima del 15 % del brano e 1 dopo il 40 %; credito del materiale suonato nel set 0 sotto
  20 s, 1 da 45 s; vale il minore [H] ([user] «A campionata all'inizio»).

**Risoluzione all'ingresso di B** `b_resolution` = media pesata [H] di segnali d'attacco nel punto in cui B diventa udibile
(lead-in compreso):
- `vocal_attack` 0,30: prima parola di B entro una battuta dal downbeat (pickup compreso), dopo un respiro >= 1 battito
  (regola R-B1 di [INGRESSO_B](INGRESSO_B_2026-10-21.md)) e con salto dello stem vocale nella prima battuta (logistica 6 dB);
  nessuna voce entro 2 battute = 0,25.
- `energy_jump` 0,15: energia (3 dB) e densità d'attacchi (0,15) della prima battuta di B contro le 2 precedenti **dello
  stesso B** ([T] Huron pdf pp. 39–40, valenza contrastiva).
- `kick_bass` 0,15: banda 20–150 Hz (cassa/basso/808) che sale nella prima battuta, drop di `dj_form` ([A] Butler pdf
  p. 246: la cassa che torna).
- `vocal_foreground` 0,15: livello dello stem vocale nelle prime 2 battute (rispetto al suo 95° percentile).
- `register_contrast` 0,15: distanza dei profili per banda (livello tolto) fra l'ultima battuta di A e la prima di B.
- `model` 0,10: `resolution_at_entry` (proxy «ritornello = arrivo», [M] debole e contrario ai voti: peso basso, dichiarato).

**Continuità della base** `continuity`: tabella [H] per tecnica (`CONTINUITY`): blend 1,0; loop/acapella/mashup su base
continua 0,9–0,95; filter out 0,6; build senza pausa 0,6; halving build senza pausa 0,55; cut 0,55; lyric link 0,5; echo out
0,3 («il beat di A si ferma, resta la coda»); qualsiasi build con `gap_beats` > 0 0,25; backspin 0,15; break `stop` 0.
[P] B&B pdf p. 34 («Mixing out of a break prevents the energy from dropping… bringing in another rising star»).
`continuity_measured`: sul render, in [−2, +1] battiti dallo switch, valore = 1 − buco/2 battiti [H]. Versione .1 (set v4): corsa
più lunga della banda 30–200 Hz sotto −15 dB rispetto alla mediana delle 2 battute prima e dopo. **Versione .2** (22/10, set v4b,
`CONTINUITY_MEASURED_VERSION`): un frame ha la base viva se la banda bassa **o** la banda delle percussioni (5–14 kHz,
charleston/rullante: il passa-alto di un filter out arriva a 1,2 kHz e non la tocca) è entro 15 dB dal riferimento **del proprio
lato** (A prima dello switch, B dopo: la base di B entra sul downbeat); nel lead-in della sola voce di B vale il riferimento più
basso dei due. La sola banda bassa resta riportata (`gap_beats_low`, `gap_beats_low_v1`) come diagnostica (§6c).

**Scontro tonale** `harmonic_clash` = esposizione della tecnica × presenza tonale × (0,6 × termine Camelot + 0,4 × termine
del croma) [H]. Camelot dopo l'eventuale trasposizione di B (+1 semitono = +7 posizioni): ≤ 1 passo 0, 2 passi 0,5, ≥ 3
passi 1, × confidenza delle chiavi (high 1 / medium 0,7 / low 0,4). Croma: 1 − coseno fra le ultime 2 battute di A e le
prime 2 di B (scala 0,3). Presenza tonale: media geometrica del picco del croma (max/somma oltre 0,10, piena a 0,25) ai due
lati. Esposizione: sovrapposizioni 1; echo 0,7; filter 0,6; cut 0,45; build 0,35; break 0,25; backspin 0,2. [P] B&B pdf
pp. 38–39 (chiavi che cozzano; «Cut them, don't blend them») è contraddetto dall'utente per la voce intonata: [user] prevale
e il cut ha esposizione 0,45, non 0.

**Affinità di testo** `lyric_affinity`: ultime 3 parole di A (finite ≥ 0,15 s prima dell'uscita, entro 2 battute), prime 4
di B dal punto udibile; legame = rima fonetica CMUdict di `dj_phrase` (`rhyme_key`, 0,75) o `dj_mashup.sound_affinity`
(uguale 1, lemma 0,85, rima 0,7, Soundex 0,5, vocalizzo 0,4, allitterazione 0,35) o assonanza della vocale tonica (0,3);
× radice della confidenza ASR minima (≥ 0,3), bonus 1,1 se è l'ultima parola di A, 1,05 se la prima di B.

**Costo nel pianificatore** (`transition_cost`, [H], scala dei costi di `dj_set`): 0,40 (1 − carica) + 0,40 (1 −
risoluzione) + 0,30 (1 − continuità) + 0,40 scontro − 0,15 affinità; un punteggio assente vale 0,5 (affinità assente 0).

## 3. Integrazione nel pianificatore

- `SetHooks.tension` (`run_dj_set_v2_books.build_set`): dopo filtri, guardia dell'ingresso di B, correzioni e affidabilità/
  novità, `TensionHook.adjust_pair` aggiunge il costo **solo ai candidati già ammissibili** (QA, `b_entry`, affidabilità
  decidono l'ammissibilità; la tensione sceglie tra ammessi); l'ordine (Held–Karp) e il Viterbi dei cue usano i costi nuovi.
  `report` nel `set.json["tension"]`.
- Tecnica: sullo stesso cue i tentativi sono riordinati per (0,3 × continuità − 0,4 × scontro), a parità l'ordine della
  policy/albero (`TensionHook.order_attempts`); con `--tension prefer` ogni build/halving della policy con pausa
  (`gap_beats` = 1 nel profilo libri) è preceduto dalla stessa tecnica **senza pausa** (`run_dj_set_v3_bridge._no_gap`).
- Per ogni transizione accettata `set.json` registra `tension` (carica e componenti, forma, minutaggio con i secondi di A
  suonati, risoluzione e componenti, continuità pianificata e misurata sul render, scontro tonale, affinità e legame,
  costo), `reliability` del candidato e `event_plan` (`musicmix.event_plan.1`).
- Default: `--tension off` (comportamento invariato). Il set v4 usa `prefer`.

## 4. Pre-registrazione

`runs/dj_tension_v1/PREREGISTRATION.json` (22/10/2026 00:17 UTC, modulo sha256 `df79839e…`): pesi, soglie, sette ipotesi
(H1–H7) e metriche (AUC Mann–Whitney con IC bootstrap per torneo/gruppo, 2000 repliche, seed 20261022; Spearman con p di
permutazione; test del segno). Scritta prima di aprire i file dei voti; **non del tutto cieca**: conoscevo i criteri e gli
esempi riassunti dall'orchestratore e i risultati aggregati di §6 di SUPERVISIONE_DAI_BRANI (carica 0,76, «arrivo» 0,35).
Nessun emendamento: pesi e soglie non sono stati toccati dopo il controllo (`AMENDMENTS.json` non esiste). Unica modifica
del test dopo l'esecuzione: un'asserzione di `tests/test_dj_tension.py` che descriveva male l'astensione (il modulo, già
pre-registrato, vale 0,5 per uno scontro non misurabile).

## 5. Controllo esterno [M] (un solo ascoltatore, pochi voti: diagnostica, non validazione)

`runs/dj_tension_v1/external.json`. Torneo: 61 varianti votate in 11 tornei, 21 accettate; 3 tornei hanno accettate e
rifiutate, 4 hanno una «migliore»; 5 tornei con almeno un'accettata contro 6 senza. Galleria v3: 19 clip votate (voti
2–7) in 10 gruppi; 5 gruppi hanno due versioni con continuità pianificata diversa.

| punteggio | torneo AUC accettate/rifiutate [IC95] (21/40) | AUC media dentro i tornei (n=3) | rango percentile della migliore (n=4) | tornei con/senza accettate (5/6) | galleria Spearman per clip (n=19), p | per gruppo (n=10), p | segno dentro il gruppo |
|---|---|---|---|---|---|---|---|
| **preferenza** (−costo) | 0,56 [0,32–0,79] | 0,80 | 0,66 | 0,63 | 0,44, p=0,07 | 0,55, p=0,11 | 1/5 |
| carica di A | **0,77 [0,57–0,91]** | 0,53 | 0,59 | 0,80 | 0,39, p=0,09 | 0,41, p=0,24 | — |
| risoluzione di B | 0,43 [0,24–0,62] | 0,67 | 0,71 | 0,40 | −0,08, p=0,74 | −0,07, p=0,87 | — |
| continuità pianificata | 0,39 [0,12–0,68] | 0,50 | 0,50 | 0,30 | −0,11, p=0,66 | −0,09, p=0,81 | **1/5** |
| − scontro tonale | 0,59 [0,31–0,88] | 0,00 | 0,54 | 0,70 | **0,49, p=0,03** | 0,61, p=0,06 | 3/5 |
| affinità di testo | **0,69 [0,55–0,86]** | 0,73 | 0,36 | 0,87 | **0,53, p=0,02** | 0,55, p=0,14 | — |
| continuità misurata (secondaria) | 0,46 [0,19–0,78] | 0,63 | 0,60 | 0,57 | 0,44, p=0,06 | 0,50, p=0,14 | 3/4 |

Esito delle ipotesi pre-registrate: **H1** (preferenza, accettate > rifiutate) direzione giusta ma IC che include 0,5; **H2**
sì (0,80, solo 3 tornei); **H3** sì (0,66, 4 tornei); **H4** debole (0,63); **H5** sì per la carica (0,77, IC sopra 0,5),
**no per la risoluzione** (0,43: migliore dello 0,35 di `resolution_at_entry` ma non sopra il caso); **H6** direzione giusta
(0,44 per clip, 0,55 per gruppo; p 0,07–0,11); **H7 no**: dentro i gruppi la versione con la continuità pianificata più alta
ha il voto più alto 1 volta su 5 (in 3 gruppi l'utente ha preferito il build con pausa di 1 battito all'echo out, che la
tabella dà 0,25 contro 0,30; in un altro l'echo out al filter out, 0,30 contro 0,60).

Lettura [M]:
- **La carica di A** regge su un secondo insieme di regole (0,77, come il solo modello 0,76): soprattutto **tra** tornei
  (quale regione di A), poco dentro un torneo (0,53).
- **L'affinità di testo** è il segnale nuovo più coerente coi voti (torneo 0,69, galleria 0,53); i legami trovati sono
  soprattutto allitterazioni e assonanze («try»→«type», «tryna»→«throw», «can't»→«Catch», «shit»→«said»), proprio il tipo
  citato dall'utente. Con 16 varianti su 61 con un legame non è una prova.
- **La continuità pianificata non concorda** con i voti: la tabella [H] ordina echo (0,30) sopra build con pausa (0,25) e
  l'utente il contrario. La continuità **misurata** sul render va un po' meglio (galleria 0,44, segno 3/4). Non abbiamo
  cambiato la tabella (sarebbe adattarla ai voti): il set v4 la usa com'è perché la preferenza è dichiarata dall'utente, e lo
  dichiariamo come limite.
- **La risoluzione di B per segnali d'attacco** non è ancora misurata bene: dentro i tornei va nella direzione giusta (0,67,
  migliore 0,71) ma tra tornei no. Il punto «B risolve» resta il più debole.
- Lo scontro tonale ha p = 0,03 sulla galleria ma AUC 0 dentro i 3 tornei (cue diversi della stessa coppia: lo scontro
  cambia poco e a caso).

## 6. Set v4 (`runs/dj_set_v4/`, `deliverables/set_dj_v4.mp3`)

Comando: `tools/run_dj_set_v4.py prepare set export gym` (default: `--reliability prefer --cards track_cards --novelty
runs/dj_set_v3_bridge/set.json --tension prefer`, piano per eventi acceso). Catalogo `runs/dj_set_v4/catalog_frozen.json` = copia
di `runs/pool_bridge_v1/catalog_ready.json` (22 brani en/it) con `vocals_path` risolto: override di
`runs/pool_bridge_v1/vocals_overrides.json` per `reminder`/`too_many_nights` (Demucs + ASR rifatti), ASR italiano
(`runs/asr_language_v1`) per `18k_no_kizzy` (il valore del catalogo resta in `vocals_path_catalog`). Analisi DJ e modelli
`dj_phrase` copiati dalla run v3 congelata per i 17 brani con ingressi identici, rifatti per 18K, `reminder`,
`too_many_nights`, `life_is_good`. Pool del set: 21 brani (`antidote` escluso a mano e dalla scheda). 268/420 coppie
ammissibili.

Esito [M]: set chiuso, 14 brani, 17:09, −13,95 LUFS, true peak −1,18 dBTP; **13/13 transizioni accettate da QA `dj_qa.6` e
veto tecnico** (6 pass, 7 warn, 0 fail; 19 tentativi, 6 fail e 2 veti scartati), nessuna ripianificazione d'arco, nessuna
coppia ripetuta dal set v3, affidabilità media 0,68 (minima 0,43). QA usato marcato in `set.json["qa_used"]`
(`dj_qa.6`, `needs_recheck: true`: `dj_qa.7` in preparazione da un altro agente).

| T | A → B | tecnica | carica A (forma) | risoluzione B | continuità pian./misurata | scontro | testo | QA |
|---|---|---|---|---|---|---|---|---|
| 1 | Reminder → God's Plan | build senza pausa | 0,65 (fine ritornello) | 0,60 | 0,60 / 0,00 | 0,07 | — | pass |
| 2 | God's Plan → Rich Flex | build (senza pausa nella policy) | 0,81 (fine sezione prima del ritornello) | 0,62 | 0,60 / 0,06 | 0,00 | 'still'→'slap' 0,38 | pass |
| 3 | Rich Flex → Sicko Mode | filter out | 0,58 (fine ritornello) | 0,47 | 0,60 / 0,50 | 0,05 | — | warn (griglia di B +181 ms nel mix) |
| 4 | Sicko Mode → Knife Talk | filter out (build senza pausa: parola di A tagliata) | 0,48 | 0,60 | 0,60 / 0,00 | 0,00 | — | pass |
| 5 | Knife Talk → Life Is Good | filter out | 0,44 (dentro il ritornello) | 0,64 | 0,60 / 0,00 | 0,04 | — | warn |
| 6 | Life Is Good → Creature | filter out | 0,57 | 0,55 | 0,60 / 0,00 | 0,00 | 'know'→'throw' 0,73 | warn (buco 0,46 s non dichiarato) |
| 7 | Creature → 5% TINT | blend 4 | 0,50 | 0,50 | 1,00 / 1,00 | 0,05 | — | warn |
| 8 | 5% TINT → Butterfly Effect | filtro + ingresso sul ritornello | 0,54 | 0,49 | 0,60 / 0,94 | 0,07 | 'tryna'→'through' 0,33 | pass |
| 9 | Butterfly Effect → A Lot | filter out | 0,66 (prima del ritornello) | 0,49 | 0,60 / 0,06 | 0,10 | — | pass |
| 10 | A Lot → Stargazing | build senza pausa | 0,52 (fine strofa) | 0,59 | 0,60 / 0,94 | 0,00 | 'niggas'→'night' 0,29 | pass |
| 11 | Stargazing → Nonstop | filter out | 0,70 (fine break) | 0,45 | 0,60 / 0,00 | 0,06 | — | warn (buco 0,75 s) |
| 12 | Nonstop → Jimmy Cooks | build senza pausa | 0,71 (fine strumentale) | 0,64 | 0,60 / 1,00 | 0,01 | — | warn |
| 13 | Jimmy Cooks → Heartless | echo out (3 tentativi scartati) | 0,63 | 0,69 | 0,30 / 0,00 | 0,03 | 'niggas'→'need' 0,31 | warn |

Che cosa ha cambiato la tensione [M] (confronto con la stessa pianificazione in `report`, `runs/dj_set_v4/plans/tension_report`):
ordine identico, cue diversi in 2 transizioni su 13 (T6 B a battuta 48 invece di 43; T7 A a battuta 88 invece di 80), il
miglior candidato cambia in 219/420 coppie ma il Viterbi e l'ordine (durate, spazi, sezioni intere) dominano; medie pianificate
quasi uguali (carica 0,597 → 0,598, risoluzione 0,548 → 0,550, affinità 0,076 → 0,133). L'effetto grande è sulla **tecnica**:
nessun build con pausa (nel set v3 erano 5 su 13: 4 build e 1 halving build), continuità pianificata media delle transizioni accettate 0,61 contro 0,42
dell'albero, un solo echo out (T13, dopo tre tentativi falliti). La continuità **misurata** (0,35 in media) segna come buco
anche il filter out (il passa-alto toglie la banda 30–200 Hz di A prima dello switch): la misura va corretta prima di usarla
come criterio.

Palestra dei difetti (`tools/defect_gym.py` non modificato, ricette `set_v3` sui globali ricollegati alla run v4; profilo
gate; `runs/dj_set_v4/defect_gym/`): vedi §6b.

### 6b. Palestra dei difetti sul set v4 [M]

`runs/dj_set_v4/defect_gym/REPORT.md` (`tools/run_dj_set_v4.py gym`, 33 min; QA `dj_qa.6` e critico importati prima delle
modifiche di `dj_qa.7` di un altro agente): 13 riferimenti = transizioni del set, 120 varianti con i difetti del gate.
**Gate FAIL**, come quello della palestra principale (`runs/defect_gym_v1/REPORT.md`): i limiti sono del QA/critico attuali,
non del set. Riferimenti: 12/12 valutati non segnalati (0 falsi allarmi); T9 non rifatta (StopIteration: il cue di B spostato
dalla guardia d'ingresso non è tra i piani di `dj_select.plan_pair`, che la ricetta `set_v3` della palestra usa). Rilevamento
(pipeline = QA fail o veto): parola di B tagliata 100 % (11), salto +6 dB 100 % (12), B in ritardo di una battuta 90 % (10),
parola di A tagliata 82 % (11), pickup tolto 80 % (5), doppio basso 58 % (12), sfasamento ½ e ¼ di battito 33 % (12),
buco di 1 battito 0 % (12; 83 % solo come avviso), scontro tonale +1 semitono non generabile (11/12 transizioni senza
sovrapposizione). Quindi il set v4 **non è verificato** contro sfasamento, buchi non dichiarati e doppio basso: va ricontrollato
con `dj_qa.7` (`set.json["qa_used"]["needs_recheck"] = true`).

### 6c. Set v4b: ricontrollo con `dj_qa.7` (`runs/dj_set_v4b/`, `deliverables/set_dj_v4b.mp3`) [M]

Comando: `tools/run_dj_set_v4b.py prepare set remeasure recheck export gym`. `prepare` copia da `runs/dj_set_v4` (congelata,
non toccata) catalogo, analisi DJ, modelli `dj_phrase` e cache delle attività (68 file, sha256 identici,
`copied_inputs.json`). `set` rifà la stessa pianificazione (stessi default del v4) e, per ogni transizione, prova **per primo**
il tentativo accettato nel v4 (stesso candidato, tecnica e motivo; `first_attempts.json`), valutato con `dj_qa.7`
(`QAConfig(dropout_check=True)`) e il veto del runner/critico v4 (`dj_critic_books.3`); se fallisse seguirebbero gli altri
tentativi e la ripianificazione senza ripieghi del runner.

Due correzioni prima della run:
1. **Spostamento del cue di B dichiarato.** `run_dj_set_v3_bridge.declare_entry_relocation` (chiamata da `render_v3`) scrive
   `meta['b_entry']['relocation']` (da/a battuta, tipo, `musicmix.dj_set.entry_guard.1`) e, per lo spostamento «stesso cue
   1–2 battute prima» (`kind = lead_bars`, R-B2 di [INGRESSO_B](INGRESSO_B_2026-10-21.md)), `relocated_bars`; `dj_qa.check_phrase`
   lo dà `warn` dichiarato. Un `cue_candidate` è già un inizio frase: solo provenienza. PCM invariato. La palestra
   (`tools/defect_gym.py::_set_plan`) ricostruisce il piano spostato da `set.json["transitions"][i]["entry_relocation"]`
   (prima: StopIteration su T9).
2. **Continuità misurata .2** (§2).

Esito: stesso ordine, **13/13 transizioni tenute al primo tentativo, PCM identico al v4** (sha256 di ogni clip e del set.wav),
nessuna ripianificazione. Verdetti `dj_qa.6` (v4) → `dj_qa.7`: 6 pass / 7 warn → **3 pass / 10 warn / 0 fail**, nessun veto.
Cambiano T1 e T8 (pass → warn: fase d'ingresso non misurabile in modo affidabile, i due rivelatori discordano) e T9 (pass →
warn: `phrase`, B entra 1 battuta prima del suo inizio frase, spostamento dichiarato). **Senza la correzione 1 T9 sarebbe
`fail`** (la seconda opinione `cue_naturalness` dà 0,17 al cue contro 0,58 all'inizio frase) e verrebbe ripianificata. Gli avvisi
`dropout` del v4 su T6 e T11 spariscono (in `dj_qa.7` il buco dentro l'effetto d'uscita conta da 1,5 battiti). Nessuna
fase misurata in modo affidabile sopra 60 ms: le avvertenze di ritmo sono «non misurabile», quindi **lo sfasamento non è
escluso** su 10 transizioni (limite noto della palestra v2, §6.4 di [PALESTRA_DIFETTI](PALESTRA_DIFETTI_2026-10-21.md)).

Continuità misurata (stesse clip): .1 media 0,35 → **.2 media 0,94**. I filter out (T3–T6, T9) passano da 0–0,5 a 0,94–1,0
(banda bassa assente 1–2 battiti, percussioni presenti); T1 da 0 a 1,0 (build con lead-in di B: la base di A resta sotto la voce);
restano bassi **T11** filter out 0,56 (0,88 battiti senza basso né percussioni: lo stesso buco che il v4 segnava come avviso
`dropout` 0,75 s) e **T13** echo out 0,81 (coda d'eco nella banda alta).

Export: set.wav identico al v4 (−13,95 LUFS, −1,18 dBTP). Il `set.mp3` del runner, decodificato, arriva a **−0,23 dBTP**
(sovraelongazione dell'encoder: vale anche per `deliverables/set_dj_v4.mp3`, identico). `deliverable_mp3` limita il WAV con tetto
abbassato (−2,64 dBTP al terzo passo) e ricodifica da pipe: `deliverables/set_dj_v4b.mp3` decodificato **−1,14 dBTP, −14,24 LUFS**
(`set.json["output"]["deliverable_mp3"]`).

Palestra dei difetti (`runs/dj_set_v4b/defect_gym/REPORT.md`, profilo gate, 46 min, `dj_qa.7` su 103 righe valutate): 13
riferimenti = transizioni del set, **tutti ricostruiti** (T9 compresa) e **0 falsi allarmi**; 130 varianti. **Gate FAIL** su 5
regole: sfasamento ½ battito 46 % e ¼ 54 % (4+4 non misurabili: Beat This! non affidabile su questi brani), parola di A tagliata
83 % (T1 mancata), basso duplicato 89 % (T7 blend mancato), pickup tolto 67 % (T6, T9 mancate); passano parola di B 100 %, +6 dB
100 %, buco di 1 battito 100 % (v4: 0 %), B in ritardo di una battuta 82 % (v4: 90 % con `dj_qa.6`; T11 mancata, T13 solo avviso);
scontro tonale non generabile (12/13 senza sovrapposizione). Quindi il set v4b è **verificato contro buchi, parole di B, salti di
livello e fuori frase**, **non** contro lo sfasamento dove la fase non è misurabile.

## 7. Limiti

- Un solo ascoltatore, 61 varianti e 19 clip, molti confronti: i p non sono corretti per i confronti multipli; le IC sono
  larghe. La pre-registrazione non è cieca rispetto ai risultati aggregati già noti (§4).
- Tutti i pesi e le soglie sono [H]; le fonti motivano la direzione delle componenti, non i numeri.
- `a_charge`/`b_resolution` usano sezioni allin1, `dj_form`, parole Whisper raffinate e lo stem Demucs, con i loro errori.
  Per `reminder`, `too_many_nights` e `life_is_good` il modello di carica non è out-of-fold.
- `continuity` per tecnica è una tabella. La misura sul render .2 guarda energia per banda, non riconosce la batteria: la coda
  d'eco o le sibilanti nella banda 5–14 kHz tengono viva la base per qualche frame (T13 echo out 0,81 anche se «il beat di A si
  ferma»); un passa-alto su A non è più un buco (la .1 lo contava, §6c).
- `harmonic_clash`: tonalità stimate (7 brani con confidenza bassa) e croma librosa sul mix (la voce rap è poco tonale: la
  presenza tonale è spesso bassa).
- `lyric_affinity`: ortografia/CMUdict inglese; per l'italiano (18K) valgono solo uguaglianza, Soundex e allitterazione;
  nessuna semantica.
- Integrità tecnica (QA, veto, palestra) non è qualità musicale: il set v4 non è stato ascoltato; il QA usato (`dj_qa.6`)
  non passa la palestra (§6b).
- La tensione cambia poco ordine e cue (dominano i vincoli del Viterbi); agisce soprattutto sulla tecnica.

## Test

`tests/test_dj_tension.py` (12: forma, carica con rampa e minutaggio, attacco dopo respiro e astensione, tabella e buco della
continuità, Camelot, scontro con e senza sovrapposizione, affinità di testo, costo e astensione, hook prefer/report, ordine dei
tentativi, variante senza pausa del runner). Mirati (tension, b_entry, v3 bridge, schede/affidabilità, dj_set): 58 passati.
Suite completa del 22/10 01:15: 760 passati, 1 saltato, 34 falliti — tutti per `NameError: check_click` in `src/musicmix/dj_qa.py`
(modifiche di `dj_qa.7` in corso da parte di un altro agente, file non toccato qui); la suite dello stesso agente alle 00:57, con
questi file già presenti, era 794 passati, 1 saltato.
