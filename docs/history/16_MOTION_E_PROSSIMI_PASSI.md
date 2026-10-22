# Motion, arco e prossimi passi

**Audit:** 18 ottobre 2026 · **Scope:** ranking direzionale V4 e primo
contratto sperimentale `musicmix.motion`.

**Aggiornamento V6 dopo il chiarimento dell'utente:** “rap piatto” indica un ingresso che non continua né risolve la suspense precedente. Un calo del proxy non dimostra la risoluzione dell'aspettativa. Il modulo corrente descrive dunque `tension_proxy_decrease/increase/stable`, senza emettere `resolution_likely` dal solo delta e dalla somiglianza cromatica. Le etichette precedenti citate più sotto appartengono al contratto sperimentale V5 e sono state ritirate prima di qualsiasi integrazione nel ranking. La ricerca dedicata con effort max è in `ricerca/MOTION_SUSPENSE_RICERCA.md` quando completata.

## Cosa sta succedendo oggi

Il ranker direzionale confronta contesti A→B di 4 o 8 beat.  Il termine
`_trend_score` confronta soltanto le due pendenze RMS:

```text
exp(-abs(trend_A - trend_B) / 0.35)
```

Questa è una misura di somiglianza tra pendenze, non una misura di tensione,
risoluzione o movimento musicale.  Due segmenti entrambi piatti ottengono un
valore alto; anche due segmenti entrambi in crescita ottengono un valore alto.
Non viene osservato l’arco prima/dopo il giunto, né viene verificato se una
crescita iniziata in A continui in B o se debba risolversi.

Il punteggio V4 usa inoltre la media dei valori disponibili di
`harmony_context`, `timbre_context`, `rms_level`, `rms_trend` e
`tempo_compatibility`, poi aggiunge la sicurezza vocale. `vertical` e
`creative` restano correttamente unknown.  Timbro e armonia sono spesso vicini
a 1 nei candidati selezionati, quindi possono rendere equivalenti molti
segmenti; RMS è energia osservata, non tensione percettiva.

Le selezioni V4 confermano il problema. Nel forward downbeat-only i quattro
candidati sono a 213,09→182,20 s, 192,10→2,74 s, 78,10→19,14 s e
138,10→120,10 s; nel reverse sono 177,44→192,10 s, 13,68→0,04 s,
21,86→9,08 s e 117,32→138,10 s. I contesti sono quasi sempre di 8 beat e
non c’è una feature che dica se l’energia osservata sta chiudendo un arco. Il
controllo all-beat ha aumentato la varietà dei cue, ma non ha risolto la
semantica: due selezioni per direzione restano `word_gap_only` e le altre sono
boundary acustici o phrase-supported.

Un esempio concreto è il reverse downbeat candidato 21,86→9,08 s: la pendenza
RMS di A è circa −0,056, quella di B circa +0,418, con score 0,735. Il ranker
lo vede come forte mismatch di trend, ma non può stabilire se sia un crescendo
che continua, una risoluzione intenzionale o un cambio brusco. Al contrario,
contesti quasi piatti possono ricevere score alto senza essere più interessanti.

## Evidenza dai paper già verificati

- Spotify, Sec. 4.1.1 e Algoritmo 1, p. 445, confronta regioni beat-sincrone
  `[i:i+n]` e conserva separati timbro, chroma, loudness, vocalness,
  sezioni e drop. Questo motiva finestre di contesto, non un nuovo peso RMS.
- AutoMashUpper, Sec. II.B, pp. 1727–1728, usa gruppi di quattro beat da
  downbeat e durate di 2/4/8/16 downbeat; Sec. III.A–D, pp. 1728–1732,
  separa armonia, ritmo, bande spettrali e tempo. È una base per misurare
  movimento a più scale, non una prova che RMS rappresenti tensione.
- Lee et al., Eq. (1)–(3), p. 401, e Sec. 7, pp. 401–403, distinguono
  continuità orizzontale e compatibilità verticale. L’arco A→B va quindi
  valutato separatamente dalla somiglianza locale e sottoposto ad ablation.
- All-In-One, Sec. 3–4, mostra beat/downbeat più affidabili dei confini di
  struttura (F1 circa 0,958/0,915 contro circa 0,660). Le sezioni euristiche
  non sono quindi una prova di frase o di tensione risolta.
- Herremans & Chew, *Tension ribbons: Quantifying and visualising tonal
  tension* (TENOR 2016), §§1, 2.2–2.4, 3, pp. 1–3, descrive cloud diameter,
  momentum e tensile strain su eventi simbolici MusicXML. È un riferimento
  per un futuro proxy melodico/armonico, ma richiede input simbolico e non
  autorizza a chiamare RMS tensione né fornisce pesi calibrati per questi WAV:
  [paper](https://www.tenor-conference.org/proceedings/2016/02_Herremans_tenor2016.pdf).
- Vande Veire & De Bie, *From raw audio to a seamless mix: creating an
  automated DJ system for Drum and Bass* (2018), Sec. 1.1.1, pp. 1–2, descrive
  la progressione rilassato→build-up→climax→riposo e l’uso di cue strutturali;
  il sistema è però specifico per Drum and Bass e non calibra il nostro
  motion score: [paper](https://link.springer.com/article/10.1186/s13636-018-0134-8).

## Contratto del primo esperimento

`musicmix.motion.analyze_motion(outgoing, incoming, windows_beats=(8,16,32))`
riceve sequenze cronologiche di `MotionBeat`. A contiene la coda prima del
cue, B la testa dopo il cue. Ogni beat può fornire `rms`, `novelty`, chroma a
12 bin, `confidence` e, solo se esiste un provider documentato,
`tension_proxy` in `[0,1]`.

Il risultato conserva una `MotionScaleReport` per ogni scala:

- pendenza RMS prima e dopo, differenza di livello RMS al confine e medie di
  novelty; sono descriptor osservati e non sono chiamati tensione;
- cosine chroma al confine senza rotazione di tonalità;
- media e delta del `tension_proxy`, se completo su entrambi i lati;
- confidenza, campi mancanti, note e classificazione deterministica.

Senza `tension_proxy`, la classificazione può essere solo
`energy_rise`, `energy_fall`, `energy_continuation` oppure
`unknown_tension`; non produce `resolution_likely` o `escalation_likely` per
inferenza da RMS. Con proxy misurato, le etichette restano dichiaratamente
`resolution_likely`, `escalation_likely` o `tension_continuation`, con
confidenza e soglia esposte, solo quando il proxy è completo su tutti i beat
della finestra; un proxy parziale resta unknown. Finestre non disponibili a
8/16/32 beat restano
`insufficient_window`.

## Prossimo esperimento concreto

1. Per ogni candidato già generato, esportare profili beat-sincroni di 8, 16 e
   32 beat per A-coda e B-testa. Conservare i dati grezzi e non usare ancora
   il report per cambiare il ranking.
2. Confrontare quattro ablation sullo stesso renderer e sugli stessi cue:
   baseline V4; baseline + arco RMS/novelty; baseline + chroma/novelty;
   baseline + `tension_proxy` quando disponibile. Misurare continuità,
   interesse, ripetizione e risoluzione percepita separatamente.
3. Annotare almeno 20–40 coppie con etichette `continua`, `risolve`,
   `cresce`, `contrasto`, `unknown`, senza dedurre l’etichetta da RMS. Usare
   intervalli per brano/ascoltatore e mantenere i controlli random e
   downbeat/all-beat.
4. Solo se l’arco migliora l’ascolto senza aumentare boundary o word-cut,
   introdurre un reranking separato con hard constraint per `unknown_tension`.
   Non sommare il motion score al punteggio locale prima dell’ablation.

Questo modulo non modifica `directional.py`, il renderer o la pipeline. È un
contratto di osservazione e un punto di misura per decidere se il problema
percepito è davvero tensione/movimento o soltanto ripetizione di cue simili.

## Audit V5 sui 18 cue V4

Lo script `tools/analyze_motion_profiles.py` ha analizzato i 4 cue
forward/reverse downbeat-only e i 5 cue forward/reverse all-beat già selezionati
da V4. Il risultato è [runs/motion_audit_v5.json](../runs/motion_audit_v5.json):
18 profili prodotti con le finestre mancanti esplicitamente gestite: 8 beat
validi 18/18 (`energy_rise` 9, `energy_fall` 9), 16 beat validi 18/18
(`energy_rise` 10, `energy_fall` 8), 32 beat validi 13/18 (5
`insufficient_window`; gli altri: `energy_rise` 5, `energy_fall` 7,
`energy_continuation` 1). Sono salvati hash dei due WAV e delle structure JSON,
campioni e provenienza dei descriptor. Le confidence
BeatThis restano metadata `beat_activation` non calibrati nella provenance e
non entrano nella confidence del motion report. Non sono stati
prodotti WAV, non è stato applicato reranking e nessun `tension_proxy` è stato
inventato; per questo ogni report contiene `missing=["tension_proxy"]` e non
può dichiarare risoluzione o escalation.

I profili sono estratti sulla timeline audio all’ancora
`original_structure_cue_samples`, perché le finestre 8/16/32 richiedono
intervalli beat reali e non è corretto interpolare descriptor a un cue vocale
spostato di pochi millisecondi. Ogni record conserva anche
`actual_candidate_cues` e `shift_samples`; il report non chiama l’ancora
originale “actual handoff” e marca esplicitamente che lo shift non è stato
applicato ai beat profiles.

Sul profilo di 8 beat, il forward downbeat 138,10→120,04 s riceve score locale
0,907 con harmony 0,970, timbre 0,999 e `rms_trend` 0,955, ma il motion audit
misura `energy_rise` con delta log-RMS +0,180. Il forward downbeat 78,10→19,14
s riceve score 0,905 con harmony 0,948 e timbre 1,000, ma misura
`energy_fall` con delta −0,087. Sono esempi di somiglianza locale che non
determina l’arco.

Il reverse downbeat 21,86→9,08 s è ancora più istruttivo: l’uscita ha pendenza
RMS −0,056, l’ingresso +0,418, il delta di livello è −0,968 e il chroma al
confine è 0,892; il punteggio resta 0,735 e il cue passa i vincoli vocali
acustici. Questo può essere un contrasto intenzionale o una risoluzione, ma il
dataset attuale non contiene la misura necessaria per scegliere fra le due.
L’all-beat audit amplia la copertura (delta 8-beat fino a +2,356/−2,280), non
trasforma questi numeri in tensione musicale. Il prossimo test deve quindi
annotare gli archi e validare un proxy di tensione separatamente prima di
usarlo per reranking.
