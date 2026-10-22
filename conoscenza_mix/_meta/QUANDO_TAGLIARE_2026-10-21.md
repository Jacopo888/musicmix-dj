# Quando tagliare: fine verso, pickup e confini nel rap (2026-10-21)

Obiettivo dell'utente (obiettivo 2): "migliorare il progetto science-based (psicologia ecc.) per capire
ancora meglio QUANDO tagliare, avendo un risultato". Problema misurato prima di questo lavoro: la guardia
vocale di `dj_select` (`cut_vocal_clear`: voce di A su al massimo il 10 % dell'ultimo battito prima del
downbeat) guarda solo se lo stem è acceso; nel rap lo stem è quasi sempre acceso, quindi quasi nessun cue
è "pulito"; i versi iniziano spesso con un pickup prima del downbeat e il renderer fa entrare B
esattamente sul downbeat; nessun modulo rilevava la fine delle barre/versi.

Legenda (come `REVISIONE_SET_TRANSIZIONI_2026-10-20.md`): **[E]** risultato empirico misurato (N
dichiarato); **[T]** teoria o quadro concettuale; **[S]** regola di un sistema; **[H]** ipotesi di questo
lavoro; **[M]** misura nostra su dati (corpus o pool), con il metodo dichiarato. "Letto": **full** (testo
completo), **abstract** (PubMed/Crossref/editore), **metadati** (Crossref verificato), **corpus** (dati
grezzi elaborati qui). Tutti i DOI sono stati verificati su Crossref il 21/10/2026
(`quando_tagliare_2026-10-21/texts/crossref_check_2026-10-21.txt`).

---

## 0. Sintesi

1. **[M, corpus MCFlow] La fine di frase nel rap è prevedibile da posizione metrica + pausa + rima +
   allungamento, ma la pausa da sola no.** Su 54 586 confini tra parole (124 brani, 7 928 frasi
   prosodiche = `**break` >= 3) un modello naive Bayes addestrato su metà dei brani e testato sull'altra
   metà ha AUC 0,85 con posizione+pausa, 0,93 aggiungendo la rima annotata, 0,94 con l'allungamento,
   0,95 con la punteggiatura; con un rilevatore di rima automatico (lo stesso usato sull'audio, applicato
   all'IPA del corpus) 0,92. Il 19 % delle pause tra parole **non** è fine frase e il 41 % delle fini
   frase **non** ha pausa: "fine frase = pausa" è falso nel rap (AGENTS.md).
2. **[M, corpus MCFlow] Il downbeat è raramente un punto pulito.** Delle 2 276 stanghette di battuta
   dentro le strofe, solo il 33 % cade tra la fine di una frase e l'inizio della successiva senza pickup;
   il 54 % cade subito dopo l'inizio di una frase partita entro un battito prima (pickup) e il 13 % a
   metà frase. Il 42,7 % delle frasi inizia entro un battito prima di un downbeat e un altro 13,3 % tra
   uno e due battiti prima; solo il 15,2 % inizia sul downbeat. Pickup entro un battito: mediana 2
   sedicesimi (mezzo battito).
3. **[M, corpus MCFlow] Le frasi finiscono tardi nella battuta**: il 54 % delle ultime sillabe di frase
   inizia tra il terzo sedicesimo del battito 3 e il secondo del battito 4 (sedicesimi 11–14); con pausa
   dopo la parola, p(fine frase) = 0,92 sui battiti 3–4 contro 0,58 sul battito 1. Le fini di iper-frase
   (`**break` = 4) cadono nel 59 % dei casi nell'ultima battuta di un gruppo di 4.
4. **[E] Psicologia dei confini**: il confine di frase musicale produce una risposta cerebrale (CPS,
   ~500–600 ms dopo la fine del confine; Knösche 2005), modulata da durata della pausa, durata
   dell'ultima nota e funzione armonica (Neuhaus 2006); il confine percepito coincide con un errore di
   predizione transitorio (Zacks 2007) e i modelli di aspettativa appresi (IDyOM, Pearce 2010) predicono
   i confini percepiti quasi quanto le regole Gestalt. **[H]** Un cambio di brano che coincide con un
   confine di verso già atteso fonde due confini invece di crearne uno nuovo in mezzo a una frase.
5. **[M, audio, 31 brani con trascrizione affidabile: 7 V12 + 24 CC] Risultato**: cue di uscita validi
   (downbeat di inizio frase di dj_select) da 13 su 406 con la vecchia guardia (3,2 %; 15 con la guardia
   della galleria v2) a 138 con `dj_phrase` (34 %); 0 parole troncate dalle nuove uscite (parole
   raffinate); livello dello stem negli ultimi 60 ms prima del taglio −32/−35 dB contro −3/−5 dB di un
   taglio secco su un downbeat qualsiasi. Ingressi: 100 su 352 hanno un pickup che l'ingresso sul
   downbeat perde sempre; la nuova regola ne conserva 68. Dettaglio nel §6.

---

## 1. Rap: flow, verso, rima

| Fonte | Tipo, letto | Cosa sostiene | Uso qui |
|---|---|---|---|
| Condit-Schultz, N. (2017). *MCFlow: A Digital Corpus of Rap Transcriptions*. Empirical Musicology Review 11(2):124–147. doi:10.18061/emr.v11i2.4961 (CC BY-NC 4.0). Corpus: github.com/Computational-Cognitive-Musicology-Lab/MCFlow @ `2f743913` | [E] corpus, **full + corpus** | 124 brani Billboard 1979–2014, trascrizioni Humdrum con ritmo, accento, `**break` (ToBI: 3 frase, 4 iper-frase), rima, IPA, testo con punteggiatura sintattica; frasi tipiche ~5 sillabe in 2 battiti o ~10 in 4; due addensamenti: inizio ~battito 1 → fine ~battito 4, e inizio ~3 → fine ~4 (p. 141–142); entropia delle posizioni di frase cresce dagli anni '80 | Tutte le tabelle di `dj_phrase` (§4) sono stimate qui; nessuna soglia sul pool. Il repository non contiene LICENSE: il corpus **non** è copiato nel repo, solo statistiche derivate (`configs/dj_phrase_mcflow_v1.json`) |
| Condit-Schultz (2016), dissertazione OSU (già in `02_fraseggio_rap_voce`) | [E] corpus, full | frasi prosodiche da contorno, declinazione, pause/allungamento; pickup prima del beat 1 normali | conferma pickup come inizio normale |
| Adams, K. (2009). *On the Metrical Techniques of Flow in Rap Music*. Music Theory Online 15(5). doi:10.30535/mto.15.5.1 | [T] analisi, **full** (HTML) | Flow = tecniche metriche (posizione delle rime, degli accenti, corrispondenza unità sintattiche/battute, sillabe per battito) + articolative; nel rap dei primi anni '80 le rime cadono "on or near successive fourth beats" (¶4); l'enjambment è "a syntactic connection that overrides a metrical boundary" (¶4); in "All Caps" solo 11 battute su 20 chiudono un'unità sintattica (¶24) | rima sul battito 4 come indizio, mai come regola; sintassi e metro possono divergere |
| Adams (2020), MTO 26.2 (già in libreria) | [T], full | il default di due battute non basta; sintassi può prevalere sulla rima | canali separati |
| Ohriner, M. (2019). *Flow: The Rhythmic Voice in Rap Music*. Oxford UP. doi:10.1093/oso/9780190670412.001.0001 | [T]+[E] corpus, **abstract dei capitoli** (Crossref) | definizioni di accento e rima calibrate sul rap (cap. 3); "vocal groove" = durate tra accenti di 2 o 3 sedicesimi che sommano a una battuta (cap. 4); corpus di strofe (cap. 2) | conferma l'unità "barra" e la rima come canale; numeri non letti, non usati |
| Ohriner (2017). *Metric Ambiguity and Flow in Rap Music*. EMR 11(2):153–179. doi:10.18061/emr.v11i2.4896 | [E] corpus-assisted, metadati | ambiguità metrica del flow | limite: allineamento flow/beat non garantito |
| Krims, A. (2000). *Rap Music and the Poetics of Identity*. Cambridge UP (ISBN 9780521634472; recensioni doi:10.30535/mto.7.4.2, 10.1093/ml/82.2.352) | [T], **recensioni + citazione in Adams 2009** | categorie di flow "sung", "percussion-effusive", "speech-effusive" | il modello non assume un solo tipo di flow |
| Kautny, O. (2015). *Lyrics and flow in rap music*. In J. A. Williams (ed.), Cambridge Companion to Hip-Hop, 101–117. doi:10.1017/cco9781139775298.011 | [T], metadati | rassegna su testo e flow | solo contesto |
| Komaniecki, R. (2017). *Analyzing Collaborative Flow in Rap Music*. MTO 23(4). doi:10.30535/mto.23.4.8 | [T], metadati | interazione di più rapper nella stessa strofa | rilevante per l'ipotesi "due voci in sovrapposizione"; non usato |
| Duinker (2020) (già in libreria) | [T], full | regole di segmentazione: rima, sintassi, respiro, somiglianza ritmica | stessa architettura a canali |

## 2. Chiusura, aspettativa, raggruppamento

| Fonte | Tipo, letto | Cosa sostiene | Uso qui |
|---|---|---|---|
| Lerdahl, F. & Jackendoff, R. (1983/1996). *A Generative Theory of Tonal Music*. MIT Press. doi:10.7551/mitpress/12513.001.0001 | [T], metadati + sintesi in Pearce 2010 | regole di preferenza del raggruppamento: GPR 2a (pausa), 2b (attacco dopo un IOI relativamente lungo), 3 (cambiamenti), 5 (simmetria: gruppi di pari lunghezza), 6 (parallelismo) | pausa e IOI come indizi (2a/2b), parallelismo delle rime (6), gruppi di 4 battute (5) |
| Narmour, E. (1990). *The Analysis and Cognition of Basic Melodic Structures: The Implication-Realization Model*. U. Chicago Press (recensioni doi:10.2307/897927, 10.2307/40285609) | [T], recensioni | la chiusura si realizza con pause, durate lunghe, accento metrico, risoluzione; l'implicazione continua altrimenti | durata lunga e pausa sono indizi di chiusura, non la chiusura |
| Huron, D. (2006). *Sweet Anticipation* (ITPRA). MIT Press. doi:10.7551/mitpress/6575.001.0001 | [T], descrizione e recensione (già in libreria) | risposte di immaginazione, tensione, predizione, reazione, valutazione; la chiusura è l'assenza di aspettativa di continuazione | tagliare dopo la fine di verso lascia A in uno stato di bassa aspettativa di continuazione [H] |
| Pearce, M. T., Müllensiefen, D. & Wiggins, G. A. (2010). *The role of expectation and probabilistic learning in auditory boundary perception: a model comparison*. Perception 39(10):1365–1389. doi:10.1068/p6507 | [E], **full** (copia OA Goldsmiths in `quando_tagliare_2026-10-21/originals/`) | 25 ascoltatori musicisti, 15 melodie; confine previsto dove l'informazione (IC) della nota seguente ha un picco; F1 medio: Grouper 0,83, LBDM 0,78, IDyOM 0,64 (non supervisionato), GPR 2a 0,58; modello ibrido (regressione logistica) F1 0,74 | abbiamo addestrato un modello IC (n-gram interpolato, ordini 0–3) sugli IOI tra parole di MCFlow: AUC 0,87 da solo per la fine frase, **nessun guadagno** sopra posizione+pausa+rima+allungamento (0,945 → 0,945). Tenuto come canale diagnostico |
| Pearce, M. T. (2018). *Statistical learning and probabilistic prediction in music cognition: mechanisms of stylistic enculturation*. Ann. N.Y. Acad. Sci. 1423:378–395. doi:10.1111/nyas.13654 | [T]+rassegna di simulazioni, abstract | aspettativa, emozione, memoria, segmentazione spiegate dalla predizione probabilistica appresa | il canale "sorpresa" va appreso dallo stile (rap), non preso da musica tonale |
| Hansen et al. (2021) (già in libreria) | [E], full | entropia predittiva influenza segmentazione e completezza | idem |

## 3. Prosodia di fine frase, segmentazione di eventi, confini nel cervello

| Fonte | Tipo, letto | Cosa sostiene | Uso qui |
|---|---|---|---|
| Wightman, C. W., Shattuck-Hufnagel, S., Ostendorf, M. & Price, P. J. (1992). *Segmental durations in the vicinity of prosodic phrase boundaries*. JASA 91(3):1707–1717. doi:10.1121/1.402450 | [E], abstract (PubMed) | l'allungamento pre-confine è ristretto alla rima della sillaba prima del confine; almeno 4 livelli di confine distinguibili dall'allungamento normalizzato | canale "allungamento finale": durata dell'ultima sillaba / mediana delle sillabe del brano; su MCFlow p(fine) = 0,65 se > 2× contro 0,07 se <= 1× |
| Steinhauer, K., Alter, K. & Friederici, A. D. (1999). *Brain potentials indicate immediate use of prosodic cues in natural speech processing*. Nat. Neurosci. 2:191–196. doi:10.1038/5757 | [E], abstract | CPS ai confini di frase intonativa; un falso confine prosodico inganna il parser | un taglio a metà frase crea un confine "falso" [H] |
| Knösche, T. R. et al. (2005). *Perception of phrase structure in music*. Hum. Brain Mapp. 24(4):259–273. doi:10.1002/hbm.20088 | [E], abstract | EEG: onda positiva ~500–600 ms dopo la fine di un confine di frase musicale (centro-parietale); MEG 400–700 ms; generatori limbici; interpretata come spostamento dell'attenzione da una frase alla successiva | [H] lasciare il confine di A completo (incluse le ~0,5 s dopo) prima che B si imponga |
| Neuhaus, C., Knösche, T. R. & Friederici, A. D. (2006). J. Cogn. Neurosci. 18(3):472–493. doi:10.1162/jocn.2006.18.3.472 | [E], abstract | la CPS musicale dipende da durata della pausa, durata dell'ultima nota prima della pausa, funzione armonica; musicisti e non musicisti differiscono; la CPS riflette l'elaborazione della frase, non della sola pausa | conferma: pausa e allungamento contano entrambi, la pausa non basta |
| Silva, S. et al. (2014). *Musical phrase boundaries, wrap-up and the closure positive shift*. Brain Res. 1585:99–107. doi:10.1016/j.brainres.2014.08.025 | [E], abstract | CPS alla fine della pausa non modulata dalla buona forma della frase; i potenziali all'inizio della pausa sì (integrazione "wrap-up") | il momento critico è l'inizio del confine (fine dell'ultima parola), non solo il silenzio |
| Krumhansl, C. L. & Jusczyk, P. W. (1990). *Infants' perception of phrase structure in music*. Psychol. Sci. 1:70–73. doi:10.1111/j.1467-9280.1990.tb00070.x | [E], metadati | i lattanti preferiscono pause inserite ai confini di frase rispetto a pause dentro le frasi (minuetti di Mozart) | un'interruzione (il cambio) è più accettabile a fine frase [H trasferita] |
| Zacks, J. M. et al. (2007). *Event perception: a mind-brain perspective*. Psychol. Bull. 133(2):273–293. doi:10.1037/0033-2909.133.2.273; Kurby & Zacks (2008) TICS 12:72–79. doi:10.1016/j.tics.2007.11.004 | [T]+[E] rassegna, abstract | un confine di evento è percepito quando c'è un errore di predizione transitorio; i confini regolano attenzione e memoria; segmentazione gerarchica | il cambio di brano è un confine di evento di livello alto; allinearlo a un confine di livello basso (fine verso) produce una gerarchia coerente [H] |
| Sridharan, D. et al. (2007). *Neural dynamics of event segmentation in music*. Neuron 55(3):521–532. doi:10.1016/j.neuron.2007.07.003 | [E], abstract | nelle transizioni tra movimenti di sinfonie: picco di attività nella transizione (anche senza stimolo), prima rete ventrale (salienza) poi dorsale (attenzione/memoria di lavoro) | il passaggio tra brani è elaborato come confine saliente: va reso coerente con la struttura, non nascosto [H] |

## 4. Cosa è stato implementato (`src/musicmix/dj_phrase.py`)

Modello esplicito di linee/barre per brano, canali separati, ogni costante in `PARAMS` con l'origine:

- **(a) parole**: timestamp faster-whisper. Pool V12: parole esistenti (`small.en`, inglese). Pool CC:
  ritrascrizione degli stem Demucs con il modello **multilingue ufficiale** `Systran/faster-whisper-small`
  (revisione `536b0662742c02347bc0e980a01041f333bce120`, rilevamento lingua; `tools/dj_phrase_asr.py`,
  stessi parametri di `vocal_boundaries.transcribe_audio`): le parole CC precedenti venivano da `small.en`
  forzato in inglese e sono allucinazioni ("I'm sorry, I'm sorry..." su brani tedeschi/russi). Raffinatura:
  Whisper incolla le parole (gap 0 per l'85 % delle parole del pool) e sbaglia di 100–200 ms, quindi la
  pausa dopo la parola *i* è il silenzio più lungo dello stem tra il centro di *i* e il centro di *i+1*,
  con attività **relativa** (entro 12 dB dal 90° percentile locale su 2 s: con una soglia globale l'85 %
  dei frame risulta attivo per riverbero/doppie/ad-lib) [H, calibrata sulla frequenza di pause notate di
  MCFlow, 10,4 %, non sui giudizi]. Parole senza supporto acustico scartate; trascrizioni inutilizzabili
  (lingua incerta, parole > 1 s, una parola > 50 % del testo) → il modello si astiene.
- **(b) rime**: CMUdict (cmusphinx @ `74790861`, BSD-2, sha256 `81917843…`) dalla vocale accentata alla
  fine; altre lingue: approssimazione ortografica (gruppi vocalici = sillabe; accento per regola di lingua:
  fr ultima, es/it/pt penultima, de prima; lettere mute finali del francese) [H]. Rima = stessa coda,
  assonanza o stessa parola 1–2 battute prima/dopo entro ±1 battito (posizioni parallele). Lo **stesso
  rilevatore** applicato all'IPA di MCFlow ha precisione 0,34 e richiamo 0,85 rispetto alle rime annotate e
  porta l'AUC da 0,85 a 0,92: la sua tabella (`rhyme_gap_detected`) è quella usata.
- **(c) prosodia**: pausa acustica in sedicesimi, allungamento finale, f0 YIN di `vocal_completion`
  (abbassamento finale in semitoni: **solo diagnostico**, nessuna stima su corpus), IC degli IOI (diagnostico).
- **(d) griglia**: battito della barra su cui inizia l'ultima sillaba (sillabe uniformi nella parola,
  agganciate all'onset spettrale più vicino entro 60 ms [H]).
- **probabilità di fine verso**: naive Bayes con le tabelle MCFlow (posizione×pausa, rima|pausa,
  allungamento, punteggiatura Whisper), canali dall'audio con esponente 0,5 [H]. Soglia 0,5 (MAP).
- **(e) pickup**: per un downbeat D di B, inizio del verso che attraversa D se parte entro 2 battiti prima
  ([E] MCFlow: 56 % delle frasi), in campioni; B a metà verso è rifiutato.
- **uscite di A** (`exit_point`): ultimo fine verso raggiungibile; stati `clean` (A libera sul downbeat),
  `a_pickup` (A ricomincia prima del downbeat: lo switch avviene prima del pickup di A, il verso nuovo di
  A non parte), `tail` (l'ultima parola sfora il downbeat fino a mezzo battito: coda protetta con lo stem
  vocale), `mid_line`, `gap_not_line_end`; controllo acustico indipendente dall'ASR (nessun frame entro
  20 dB dal percentile locale a ±40 ms dallo switch).
- **switch** (`phrase_switch`): punto più tardo ammesso da entrambe le finestre, sull'ottavo; B entra dal
  pickup con il suo downbeat sul downbeat di A. Render (`render_phrase_transition`): rimescola i deck di
  `dj_render` (cut) o `dj_effects.filter_out` senza modificarli; modo `stem` per il cut: tra lo switch e il
  downbeat suona lo strumentale di A (mix − stem vocale) con la sola voce di B (il pickup), poi B intero
  sull'uno [H].
- **opzione in `dj_select`**: `plan_pair(..., phrase={"a": model_a, "b": model_b, ...})` restituisce i piani
  "cut_phrase"; default invariato (i 31 test esistenti passano); `dj_phrase.exit_clear_test` si collega a
  `relocate_cut(is_clear=...)`.

## 5. Ipotesi e limiti

- Le tabelle vengono da **trascrizioni simboliche di 124 brani Billboard inglesi** (1979–2014): non da
  audio, non da trap recente, non da francese/tedesco/russo. Il trasferimento è [H]; i canali dall'audio
  sono ridotti (esponente 0,5) ma l'esponente non è stimato.
- "Fine verso" = confine prosodico `**break` >= 3 di MCFlow, non chiusura percepita né preferenza DJ.
  Nessuna fonte dice che tagliare a fine verso sia *preferito* in un DJ set: è l'ipotesi che i dati
  psicologici rendono plausibile (confini allineati, nessun falso confine) e che va ascoltata.
- Whisper può sbagliare parole e confini; la misura di troncamento è riportata sia con le parole raffinate
  (quelle usate dalla guardia) sia con le parole grezze, più un controllo puramente acustico (livello
  dello stem negli ultimi 60 ms prima del taglio). Le parole deboli sotto la soglia relativa possono
  sfuggire.
- Il modo `stem` dipende dalla separazione Demucs (residui della voce di A nello strumentale).
- La sorpresa MusicGen di `dj_critic` non è usata per la decisione; se disponibile è riportata come
  controllo (§6).

## 6. Risultato misurato (pool V12 + pool CC, `runs/dj_phrase_v1`)

Run `runs/dj_phrase_v1` (`tools/run_dj_phrase_v1.py analyze compare render`; `compare.json`,
`tracks/<slug>.json`, `render_manifest.json`, `clips/`). Brani: 7 V12 (parole `small.en`) + 31 CC
completi con ASR multilingue (24 con trascrizione affidabile; 7 dove il modello **si astiene**:
trascrizione inutilizzabile, p. es. lingua non riconosciuta o una parola ripetuta; 11 CC completi senza
ASR perché il lavoro sul pool CC è stato fermato). Cue = downbeat di inizio frase di `dj_select`
(`cue_candidates`), nessuna soglia adattata ai giudizi.

| (31 brani affidabili) | vecchia (`cut_vocal_clear`) | galleria v2 | nuova (`dj_phrase`) |
|---|---|---|---|
| uscite valide / candidati | 13 / 406 (3,2 %) | 15 / 406 (3,7 %) | **138 / 406 (34 %)** |
| di cui V12 | 4 / 113 | 4 / 113 | 30 / 113 |
| di cui CC | 9 / 293 | 11 / 293 | 108 / 293 |
| parole troncate dalle uscite valide (raffinate / ASR grezze) | 0 / 0 | 0 / 0 | **0** / 50 (§5: estensioni Whisper sopra il silenzio) |
| livello stem negli ultimi 60 ms (mediana per brano, dB re p90) | −58/−60 | – | −32/−35 |
| switch anticipato (pickup di A, ≤ 2 battiti) / coda protetta | – | – | 70 / 42 |

Un taglio secco su ciascun downbeat candidato tronca una parola di A nel 66 % dei casi (268/406 con
le parole raffinate, 242 con le grezze); il livello dello stem in quel punto è −3/−5 dB.

Ingressi di B (352 candidati): 203 (58 %) troncano una parola se B entra sul downbeat; 100 (28 %) hanno
un pickup che l'ingresso sul downbeat perde sempre. Nuova regola: 116 ingressi validi, **68 con pickup
conservato**, 0 parole troncate (39 con le parole grezze).

Coppie ordinate (stesso pool): V12 36/42 con almeno un'uscita+ingresso ammesso (25 con pickup) contro
18/42 con almeno un cut pulito per la vecchia guardia; CC 505/930 (384 con pickup) contro 240/930. Piano
scelto da ciascuna politica: il piano migliore di `dj_select` (cut in 39/42 coppie V12) tronca una parola
di A in 22/42 coppie V12 e 294/930 CC e con la guardia vecchia diventa `echo_out` in 40/42 e 810/930;
il piano migliore di `dj_phrase` esiste per 41/42 e 520/930 coppie, con 0 parole troncate e pickup di B
conservato in 11 e 297.

**Controllo di distribuzione** (versi dall'audio contro frasi MCFlow): parole per verso 6,9 (V12) e 7,6
(CC) contro 6,9; inizio del verso entro 2 battiti prima di un downbeat 59 % (V12) e 52 % (CC) contro 56 %;
ma "sul downbeat" 7 % contro 15 % e "1–2 battiti prima" 31 % contro 13 %: gli onset raffinati o la fase
della griglia anticipano. In `creature` gli inizi si addensano sul battito 2 (possibile errore di fase
della battuta della griglia, ipotesi da verificare). Fine verso sui battiti 3–4: 53 % V12, 68 % CC,
contro 71 % in MCFlow.

**Clip** (9 coppie, 36 clip da 20 s, scelte prima dello stop del pool CC: 3 CC in russo/tedesco e 6 V12;
3 coppie hanno tempi incompatibili e sono solo cut): per ogni coppia `phrase_cut` (modo `stem`),
`phrase_filter_out`, lo stesso cue reso con `dj_render` sul downbeat (`same_cue_downbeat_cut`) e il
piano di default di `dj_select` reso come cut. QA `dj_qa` con le stesse parole udibili per tutte:
`verse_cut` fallisce 6/9 (default), 5/9 (stesso cue sul downbeat), 2/9 (`phrase_cut`, entrambe uscite con
coda protetta: la regola ±150 ms di `dj_qa` le segnala per costruzione), 0/9 (`phrase_filter_out`,
esente). Parole troncate misurate sul render (raffinate): A 0/9 e B 0/9 per le versioni phrase, contro
A 4/9 e B 3/9 per lo stesso cue sul downbeat. I fail residui sono di ritmo (griglia), non di confine.
Nessun ascolto umano: integrità non è qualità (AGENTS.md).

**Sorpresa musicale**: il canale IC degli IOI (addestrato su MCFlow) è calcolato per ogni confine
(`gaps[].ioi_ic_next`) ma non entra nella decisione; la sorpresa MusicGen di `dj_critic` era ancora in
costruzione e non è stata usata.

## 7. Righe da aggiungere (non modificati qui)

INDEX.md, in coda:

```
## Quando tagliare: fine verso e pickup (2026-10-21)

Fonti (MCFlow come corpus, Adams 2009, Ohriner 2019, Pearce 2010, GTTM, Narmour, Wightman 1992, Knösche 2005, Neuhaus 2006, Silva 2014, Zacks 2007, Sridharan 2007) e risultati in [`_meta/QUANDO_TAGLIARE_2026-10-21.md`](_meta/QUANDO_TAGLIARE_2026-10-21.md); modulo `src/musicmix/dj_phrase.py`, tabelle `configs/dj_phrase_mcflow_v1.json`, run `runs/dj_phrase_v1/`.
```

CHANGELOG.md, in testa:

```
## 2026-10-21 — quando tagliare: fine verso e pickup (`dj_phrase`, run `runs/dj_phrase_v1/`)

- Nuovo `src/musicmix/dj_phrase.py` + `tests/test_dj_phrase.py` (15 test): parole Whisper raffinate sullo stem, rime (CMUdict / ortografia), prosodia, griglia, probabilità di fine verso con tabelle stimate su MCFlow (`tools/mcflow_phrase_stats.py`, `configs/dj_phrase_mcflow_v1.json`; AUC su metà dei brani 0,85 → 0,94), uscite di A a fine verso con coda protetta, ingressi di B dal pickup, render che rimescola i deck di `dj_render`/`dj_effects` (modo `stem`).
- `dj_select.plan_pair(..., phrase=...)` opzionale (default invariato) e `dj_phrase.exit_clear_test` per `relocate_cut`. ASR multilingue `Systran/faster-whisper-small` per il pool CC (`tools/dj_phrase_asr.py`, lingua rilevata sul parlato).
- Risultato su 31 brani: uscite valide 13 → 138 su 406, 0 parole troncate; 68 pickup conservati su 100. Nota `_meta/QUANDO_TAGLIARE_2026-10-21.md`.
```
