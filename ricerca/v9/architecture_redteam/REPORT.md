> Copia integrata. Leggere prima [le correzioni dell’orchestratore](../REVIEW_CORRECTIONS.md); gli originali restano nella cartella dello swarm.

# Report V9 — Perche V8 fallisce sul WHEN e come ricostruire il feedback autonomo

**Autore:** architecture_redteam
**Data:** 2026-10-19
**Perimetro:** audit dell'architettura V8 (WHEN + feedback autonomo) e proposta V9 evidence-based. Nessuna affermazione SOTA. Nessun giudizio d'ascolto: tutte le osservazioni audio sono derivate da codice o da fonti testuali.
**Deliverable collegati:** `output/sources.json`, `output/claims.json`, `output/RECOMMENDATIONS.json`.

---

## 1. Sintesi esecutiva

V8 non fallisce per mancanza di modelli di embedding o di feature. Fallisce per tre difetti architetturali misurabili nel codice:

1. **La policy e obbligata a transire.** `context/src/pool_experiment.py:32-53` seleziona una sola istanza per ogni coppia `(track_a, cue_out)` con `source_cap=1`, e `context/src/expectation_pipeline.py:34-55` costruisce sempre un confronto forzato A/B. Non esiste l'azione "resta in A" ne tantomeno "astieniti". Il sistema non puo quindi rispondere correttamente alla domanda WHEN, perche non puo mai dire "non ancora" o "non qui".
2. **Il criterio ottimizzato non e cio che viene ascoltato.** `context/src/directional.py:36` valuta un intervallo di fade di `0.04 s`, mentre `context/src/musical_preview.py:20-21` rende `0.012 s` con rampe sequenziali senza overlap. Lo score di ricerca non descrive il render.
3. **Confidenza non calibrata.** `confidence: null` resta corretto con quattro osservazioni. Il test Audiobox V8 su un solo item corrotto e i nuovi controesempi non validano il setup come gate del raccordo; non dimostrano debolezza generale del modello.

Su questa base la letteratura esterna consultata dice tre cose convergenti: (a) una transizione e un **intervallo** con inizio, punto di switch e fine, non un punto (Zehren et al. 2022, `t1/t2/t3`; CUE-DETR 2024; Cliff 2000); (b) i cue point sono **ambigui e soggettivi** (accordo tra annotatori che scende a ~62-65% con tre annotatori richiesti; AP massima 0.60 e F1 0.46 di CUE-DETR); (c) un sistema DJ completo deve **accoppiare** sequenziamento, traiettoria di tempo, durata dell'overlap, time-stretch, fase e guadagno (Cliff 2000).

La proposta separa eligibilita, ranking e decisione stay/transition/defer. Prima di un benchmark pertinente, i nuovi critici musicali sono diagnostici. L'accettazione autonoma musicale richiede un test congelato esterno con rischio/copertura misurati; si possono usare etichette e preferenze esperte gia disponibili, senza nuovi voti utente a ogni ciclo.

---

## 2. Cosa fa davvero V8 (audit del codice)

### 2.1 WHEN e modellato come coppia di cue puntuali
`context/src/directional.py` calcola uno score per una coppia di cue:
`0.82 * mean(harmony, timbre, rms_level, rms_trend, tempo) + 0.18 * vocal` (`context/src/directional.py:968-969`).
La finestra di contesto e `_WINDOWS=(4,8)` beat (`:33`), il fade target e 40 ms (`:36`), la fase massima 80 ms (`:37`), i candidati massimi 4096 (`:32`). La `vocal_score` e una funzione a gradini `1.0 / 0.97 / 0.82 / 0.70` (`:968`). La priorita di confine preferisce `phrase_boundary_supported` / `acoustic_boundary_supported`, ma **non verifica lo stato funzionale della traccia A** (dove siamo nella forma) ne la durata dell'evento. Questo e il default del modulo, non la configurazione V8: il run reale ha griglia completa1.611.642 coppie e cap200.000 per coppia, senza troncamento.

### 2.2 La selezione e greedy, non pianificata
`context/src/pool_search.py:288-299` (`_select_globally`) ordina per `-_proposal_score` e usa Dinic solo per la feasibility dei ruoli/coppie. Non c'e pianificazione di sequenza, ne scelta della durata, ne decisione stay/leave, ne eligibilita di frase. Il flow serve a rispettare cap, non a pianificare il WHEN.

### 2.3 Il trial e forzato
`context/src/pool_experiment.py:32-53`: `maximum=4, spacing_seconds=8, source_cap=1, incoming_cap=3`, raggruppamento per `(track_a, cue_out)`. `context/src/expectation_pipeline.py:34-55`: `maximum=3, spacing_seconds=12.0, incoming_separation_seconds=8.0`, sempre stesso outgoing. Il design rende impossibile osservare la scelta "resta in A".

### 2.4 Guard vocale e rampe hanno durate diverse
`context/src/musical_preview.py:17` `HANDOFF_VERSION='anchor_handoff_v1'`; `render_handoff` usa `context_seconds=10.0` e `fade_seconds=0.012` (`:20-21`), rampe sequenziali, nessun overlap, nessuna correzione di tempo/pitch/loudness. Il modulo legacy `context/src/blend_preview.py` e `equal_power_overlap_v1` con `TEMPO_NORMALIZATION='half_or_double_only'`. La ricerca ragiona a 40 ms; il render produce 12 ms. Le rampe da12ms sono contenute negli intervalli conservativi da40ms: non e prova di un taglio fuori dalla regione ammessa.

### 2.5 La motion analysis contiene un riferimento futuro
`A_natural_after` usa il seguito originale come riferimento offline esplicito, legittimo quando l'intero file e disponibile. Non sarebbe causale in uno scenario online senza futuro; non e automaticamente leakage offline. Il reranking resta sperimentale e non calibrato.

### 2.6 Il critic non e un validatore
`context/src/aesthetic_critic.py` e un `audiobox-seam-diagnostic-v1` su 10 s centrati sulla giunzione, con assi separati. Non e un critic di transizione sequenziale validato. `CRITIC_VALIDATION_V8.md` mostra che, pur con clipping duro al 5,48%, gli assi si muovono pochissimo; un solo item, nessun test-retest.

### 2.7 Cosa dicono i dati umani V8
`context/HUMAN_FEEDBACK.md` contiene 4 osservazioni: 2 `neither`, 1 `tie`, 1 preferenza per `directional_pool_baseline_v8`; `confidence: null`; nessun vincitore. Il JSON grezzo (`context/feedback_Pool_V8_3bdeea23988e22af.json`) registra anche le etichette `motion` (unclear/interrupts). Questo non e un set di valutazione: e un diagnostico tenuto fuori dal training.

**Conclusione dell'audit:** mancano scelta stay, contesto funzionale ed eventi condivisi. Non abbiamo ablation che attribuiscano causalmente tutti i fallimenti a queste limitazioni.

---

## 3. Evidenza esterna (nuove fonti, con ancore)

Le fonti sono dettagliate in `output/sources.json`. Qui la sintesi per tema, distinguendo cosa e stato letto in full text e cosa no.

### 3.1 La transizione e un intervallo, non un punto
- **Zehren et al. 2022** (S02, full text): definisce `t1` (inizio), `t2` (switch), `t3` (fine). La valutazione e soggettiva: la precisione media scende da ~90% con almeno un annotatore a ~70% con due e ~62-65% con tre. Dataset EDM.
- **CUE-DETR 2024** (S03, full text): 21.461 cue point, 4.710 tracce; AP massima 0.60 phrase-aligned, F1 0.46; il paper discute esplicitamente l'ambiguita e le differenze tra stili DJ.
- **Cliff 2000** (S06, full text): cross-fade con marker inizio/fine, beat-matching, time-stretching, bridge tracks per grandi salti di tempo.

Implicazione: il cue e una **decisione incerta**, non una verita. Un sistema che ne seleziona uno solo senza incertezza non e allineato allo stato dell'arte.

### 3.2 Lunghezze di transizione e struttura frase
- **Kim et al. 2020** (S01, full text): 20.765 transizioni; istogramma con picchi ogni 32 beat (frase); 86,1% dei cambi di tempo <5%, 94,5% <10%; accordo cue 40,4% entro una misura e 73,6% entro otto.
- **Raveform 2026** (S04, full text): 4.902 mix, 73.505 tracce suonate, 53.780 transizioni disponibili; vocabolario intro/buildup/drop/breakdown/cooldown/outro; lunghezza modale 30 s / 16 misure; BPM modale 128.

Implicazione: esiste un forte prior di durata in beat/frasi, ma e **EDM-condizionato**. Va reso esplicito e testabile, non usato come costante universale.

### 3.3 Timbre e selezione
- **Kell & Tzanetakis 2013** (S05, full text): su 114 Essential Mix, il timbro mostra il divario maggiore e piu significativo (Welch t-test p<0.0001); l'euristica Rank/NN e **peggiore** della sequenza DJ reale in inversione.

Implicazione: non ridurre il ranking a similarita di embedding tra tracce; il timbro va misurato **alla giunzione**.

### 3.4 Reversing e dataset sintetici
- **Schwarz & Fourer 2021** (S15, **abstract only**, full text bloccato): stima congiunta di posizione, time-scaling, cue e curve di fade; UnmixDB sintetico beat-sincrono. Da trattare con cautela perche il full text non e stato letto.

### 3.5 Flow rap
- **Adams 2009** (S13, full text): posizione metrica di accenti e rime, rapporto sillabe/beat, corrispondenza tra unita sintattiche e misure (`All Caps`: 17 unita su 20 misure).
- **Condit-Schultz 2017 MCFlow** (S12, **abstract only**, pagina/PDF bloccati): 124 brani, 374 versi, 5.803 misure, 86 rapper; il corpus codifica posizione metrica di sillabe e rime, densita di rima, lunghezza di frase.

Implicazione: il "rap piatto" di V8 non e un fatto; la prosodia rap ha struttura. Servono feature dedicate, non solo RMS.

### 3.6 Incertezza, astensione e stop sequenziale
- **Geifman & El-Yaniv 2017** (S09, full text): risk-coverage e SGR con garanzia di rischio sotto i.i.d.
- **Angelopoulos & Bates 2022** (S10, letture selezionate): coverage **marginale** sotto exchangeability; coverage condizionale non ottenibile in generale; drift richiede varianti pesate.
- **Wald 1945** (S16, **abstract only**): decisione accept/reject/continue; SPRT.

Implicazione: si puo costruire un checkpoint di astensione onesto, ma solo per sottoproblemi ben definiti e con dati di calibrazione scambiabili. **Non** si puo usare la conformal per validare il gusto musicale.

### 3.7 Sistemi DJ recenti
- **DJ-AI 2025** (S11, full text): ordinamento playlist + MERT + MusicGen; test umano con 20 partecipanti e 70,5% di preferenza, 10 coppie, nessun codice pubblico; l'autore del PDF non e compilato (metadata ACM incompleto).
- **Hein 2025** (S14, **abstract only**): CNN che predice volume/EQ per una transizione, **addestrata su un solo esempio**; codice pubblico, ma evidenza di learnability, non validazione.

Implicazione: il rendering generativo e rischioso per la voce e non riproducibile; l'idea trasferibile e l'**accoppiamento** di sequencing e mixing, non il modello generativo.

---

## 4. Architettura V9 proposta

La proposta e una ristrutturazione della decisione, non un nuovo modello di embedding.

### 4.1 Tre azioni di prima classe
```
        +-----------+      eligibile      +-------------------+
        |  LISTEN_A | ------------------->|  DECIDE            |
        +-----------+                     |  stay_A            |
              ^                           |  transition(t2,plan)|
              |                           |  abstain           |
              |                           +-------------------+
              |                                   |
              |  VERIFY (gate tecnico + umano)    v
              +---------------------------  RENDER(plan)
```
- `stay_A`: estende il playback di A; e l'azione corretta quando nessun candidato supera soglia o l'incertezza e alta.
- `transition(t2, plan)`: il piano contiene `start_A, end_A, start_B, end_B`, durata in beat, curva di fade, tempo factor, phase shift.
- `abstain`: nessuna proposta; il sistema dichiara l'incertezza.

Mappa su Wald (S16): accept/reject/continue. Mappa su Geifman (S09): coverage/rischio.

### 4.2 Separazione eligibilita / ranking / decisione
- **Eligibilita** (vincoli duri): il cue in A e un confine di segmento supportato con confidenza >= soglia; B entra in un segmento adatto; nessuna parola tagliata; fase entro 80 ms; tempo dentro i limiti. Se un vincolo non e verificabile, lo stato e `unknown`, non "ok".
- **Ranking** (utilita dell'evento): `PositionProvider(A, cue) -> score/uncertainty` (intra-mixability) e `CompatibilityProvider(A, B, plan) -> score/uncertainty` (extra-mixability), separati come in M-DJCUE (S07). Niente singolo score di coppia.
- **Decisione**: soglia di rischio calibrata; sopra soglia -> `abstain`.

### 4.3 Contratti dei provider
```
StructureProvider(track) -> segments[{start,end,function,confidence}]
VocalProvider(track, window) -> {activity, word_boundaries, confidence}
PositionProvider(trackA, cue) -> {score, uncertainty}
CompatibilityProvider(trackA, trackB, plan) -> {score, uncertainty}
TrajectoryProvider(context) -> {trend, uncertainty}
Renderer(plan) -> audio + render_report
Critic(render_report, plan) -> technical_defects[]
```
Gia esistono `Feature`, `Candidate`, `ScoreReport` e i protocolli in `context/src/contracts.py`: la V9 estende quei contratti, non li sostituisce.

### 4.4 Stato e causalita
Stato: `LISTEN_A` -> `ELIGIBLE_WINDOW(B)` -> `PREPARE_TRANSITION` -> `TRANSITION_ACTIVE` -> `VERIFY` / `ABSTAIN`. Ogni feature online deve essere calcolata solo dal passato; `A_natural_after` va rimosso dagli input online (C16) e mantenuto solo come target offline, se serve.

### 4.5 Uniformare search e render
Il `TransitionPlan` e l'unica fonte di verita: `directional.py` produce e valuta un piano; `musical_preview.render_handoff` consuma lo stesso piano. Le costanti 0.04/0.012 vanno eliminate in favore di campi del piano.

---

## 5. Feedback autonomo: cosa si puo automatizzare e cosa no

**Automatizzabile oggi (solo difetti tecnici, solo rigetto):**
- clip (campioni >= |1.0| prima del limiter),
- click/discontinuita (energia ad alta frequenza alla giunzione),
- disallineamento di battuta oltre tolleranza,
- word-cut stimato con alta confidenza (con la cautela di `SINGING_ALIGNMENT_V8.md`: token probability != word-boundary confidence),
- silenzio/livello anomalo.

Questi segnali **possono solo rifiutare**, mai premiare. Devono avere precisione misurata su difetti costruiti ad hoc.

**Da non usare ancora come accettazione validata; sperimentabili in diagnostica:**
- qualita musicale della transizione,
- appropriatezza del punto di switch,
- preferenza tra due transizioni.

Questo e un gate di deployment precedente alla validazione: i critici possono lavorare subito in diagnostica; per usarli nell'accettazione musicale serve prima un benchmark congelato pertinente, anche con etichette esperte esterne gia disponibili. Un tale insieme di preferenze sul nostro raccordo non e ancora pronto. Conformal non garantisce gusto o robustezza sotto drift arbitrario.

**Distinzione V8:** gli otto agenti sono strumenti di ricerca, non un ground truth percettivo. Il vero auto-feedback V8 comprende controlli tecnici e Audiobox eseguito sull'audio. Va ampliato e metavalutato sul compito della transizione, conservando le diagnostiche utili.

---

## 6. Rischi e limiti di questo report

- **Nessuna verifica audio.** Non ho ascoltato nulla; le affermazioni su qualita audio sono ipotesi derivate dal codice.
- **S02/S03/S04/S05/S06/S08/S09 sono EDM-centrici.** Il trasferimento a rap/pop e incerto; lo dichiaro in ogni sorgente.
- **S12, S14, S15, S16 letti solo come abstract/metadata** perche full text bloccato (bot/login/paywall). Non uso i loro numeri come prove forti.
- **Il caso rap e supportato qualitativamente** (S13 full text, S12 abstract) ma non esistono soglie frase validate per rap. Le proposte R05/R08 sono ad alta incertezza.
- **Rischio di sovra-ingegnerizzare l'incertezza:** se il sistema si astiene troppo, l'utente non riceve alcuna proposta. Va bilanciato con un fallback beat-only conservativo ma non validato, senza garanzia di fraseggio o qualita musicale.

---

## 7. Piano d'azione minimo (ordine)

1. **R02 + R01 (P0):** unificare `TransitionPlan` search/render; introdurre le tre azioni e il trial `stay_A`. Senza questo, ogni altra metrica e inquinata.
2. **R04 (P0):** calibrazione e rubricacoverage/rischio; rimuovere `confidence: null`.
3. **R03 (P0):** struttura funzionale di A come gate di eligibilita.
4. **R07 (P1):** controlli tecnici e critici diagnostici; benchmark esterno congelato prima di usare il premio per accettazione musicale. Nessun nuovo voto utente richiesto a ogni ciclo.
5. **R06 + R05 (P1):** separare Position/Compatibility; feature prosodiche/vocali.
6. **R08 + R09 (P1-P2):** cue come distribuzione; rimozione del riferimento futuro.
7. **R10 (P2):** protocollo pre-registrato (prima di raccogliere qualsiasi dato).

---

## 8. Conclusione

V8 non e "quasi pronto". E un sistema che non puo esprimere WHEN perche non ha un'azione di astensione, non ha un intervallo di transizione coerente tra scoring e render, e non ha incertezza. Le fonti esterne confermano che la transizione e un evento incerto e soggettivo, che la durata segue prior frase-condizionati, e che sequenziamento e mixing vanno decisi insieme. La V9 proposta non aggiunge un modello piu grande: aggiunge le tre azioni, la separazione eligibilita/ranking/decisione, e un feedback autonomo che sa dire "non lo so". Il guadagno principale non e un punteggio migliore, ma la possibilita di misurare onestamente cio che il sistema non sa.

Nessuna affermazione SOTA. Accettazione musicale non validata finche manca un benchmark pertinente; il runtime puo poi operare autonomamente usando evidenza esterna verificata.
