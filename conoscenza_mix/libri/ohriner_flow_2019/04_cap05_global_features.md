# Ohriner 2019 — cap. 5 «Features of Flow in the Genre and the Artist» (pp. 104–132)

Letto: testo completo + `Scripts/AnalysisScripts/Corpus Feature Generation.R`, `RhymeEntropyFunction.r`, `CentralPillarFunctions.r`, `DerivedData/CorpusGlobalFeatures.txt` (commit `faaec231`).

## Tesi

- Le «global features» descrivono uno stile (analisi di stile) e permettono di collocare un artista (analisi critica, Meyer 1973) (pp. 104, 124, 131–132) [T].
- Velocità sillabica costante negli anni; il tempo dei brani cala e la saturazione cresce (p. 107) [E].
- Le fini di frase si concentrano sul **beat 4** («central pillar», Griffiths 2003); le frasi iniziano vicino al battere e le differenze di lunghezza si assorbono «starting later, not ending earlier» (pp. 108–110) [E].
- Black Thought: complessità di rima e di groove + coerenza/comprensibilità della consegna (pp. 124–132) [A+E].

## Definizioni (come nel codice)

| Feature | Definizione | Pagina |
|---|---|---|
| sillabe/s | 1 / durata media delle sillabe che non chiudono un verso o un respiro | p. 106 |
| saturazione | sedicesimi occupati per battuta, media | p. 107 |
| variazione di saturazione | media di \|Δ\| tra battute consecutive | p. 108 |
| posizione di inizio/fine frase | densità circolare (von Mises, κ = 25) delle posizioni; «pilastro» = picco | pp. 108–110 |
| mescolanza delle rime | trasposizioni di un insertion sort della sequenza delle classi di rima, per battuta | p. 111 |
| entropia degli IRI | entropia scalata degli intervalli tra tutte le coppie di istanze di una classe, IRI ≤ 8 beat; anche mod 4 (sotto il beat) | pp. 112–115 |
| IAI medio | media degli IAI limit-3 (tra 2 e 3): misura della sincope | p. 116 |
| grooviness | quota della strofa in groove ≥ 2 battute per l'ascoltatore adattivo | p. 117 |
| groove adherence | quota dei groove ≥ 2 battute a 1 swap per mezza battuta coperta dallo stesso groove a sforzo 0 | pp. 117–119 |
| tipicità del groove | contributo alle prime 4 dimensioni di una correspondence analysis delle quote di classe | pp. 119–124 |
| dupleness | quota della strofa in <2222_2222> a sforzo 0 | Tab. 5.2 |

## Dati e numeri [E, corpus di 75 strofe salvo diversa indicazione]

- Velocità sillabica media **5,09 sill/s** (p. 106); nessuna differenza per regione né per anno (p. 107).
- Sillabe per parola da 1,10 (Kurtis Blow «The Breaks») a 1,54 (The Roots «What They Do»), t(160) = −4,84 (p. 105).
- Variazione di saturazione: di solito 1–2 sillabe tra battute; Wiz Khalifa «Black and Yellow» ~3, Kurtis Blow «The Breaks» < 1 (p. 108).
- Pilastro: solo 9/75 strofe nella prima metà della battuta, due terzi su o dopo la posizione 12 (p. 110). Kanye West «Stronger» (2007, 0:50–0:61) mette il pilastro dopo la posizione 2, in competizione con la cassa, come il campione dei Daft Punk (p. 110).
- Mescolanza delle rime: quasi metà delle strofe = 0 (p. 111). Tre rime consecutive a 3 beat solo in 3 dei 70 brani; quattro in nessuno (n. 14 p. 113).
- Duplo a 1mod2 (sincope di spostamento): 2 % del corpus; 12 strofe interamente duple; in 15 solo 1 IAI su 10 vale 3 (p. 116).
- Tab. 5.2 (p. 130), medie (DS) del corpus: saturazione 11,1 (1,45) posizioni; variazione 2,07 (0,73); **durata delle frasi 2,37 s (0,43)**; varianza 2,87; istanze per classe di rima 3,62; sillabe per istanza 1,65; mescolanza 0,15; duplo 0,45 (0,34); sill/s 5,09 (0,71); entropia IRI 0,44 (0,15); IRI mod 4 0,23 (0,16); grooviness 0,51 (0,34); adherence 0,67 (0,27); tempo 87,7 BPM (15,6). Black Thought (30 strofe): saturazione 11,79, variazione 1,70, frasi 2,13 s, istanze 5,07, sillabe per istanza 2,31.
- Tipicità (Tab. 5.1 p. 123): più atipiche «The Breaks» (92 % <332_332>), Grandmaster Flash «The Message» (100 % <323_2222>), Blu «Ambition» (61 % <333232>); più tipiche quelle ~0,7 duple + un po' di <332_2222>.

## Esempi

Treacherous Three «Feel the Heartbeat» (1981) vs Slaughterhouse «Y'all Ready Know» (2015) per la mescolanza (Es. 5.9); Kanye «Stronger» vs Jean Grae «My Crew» (2003, 1:09–1:20) per l'entropia degli IRI; Lil' Kim «Lighters Up» (<332_332> a livello di beat, IRI interi: alta entropia mod 16, bassa mod 4); Rick Ross «Aston Martin Music» (2010, 1:18–1:40) vs Wiz Khalifa (2011, 0:36–1:00) per la grooviness (0,8 vs 1); The Coup «The Magic Clap» (2012) vs Treacherous Three per l'adherence (stesso groove, 0,25 vs ~1).

## Regole operative

1. Le frasi rap si allungano **iniziando prima**, non finendo dopo: la fine è ancorata al beat 4 (p. 110) [E] → per MusicMix la fine verso è più prevedibile dell'inizio; l'inizio (pickup) varia.
2. Il pilastro è una proprietà del brano/strofa: ogni brano ha la sua posizione tipica di fine frase [E] → adattamento per brano (implementato come termine opzionale in `analyze_track`, verificato su MCFlow held-out).
3. «Virtuosità» secondo Ohriner = sillabe/parola, sillabe/s, saturazione, variazione, lunghezza delle classi di rima, mescolanza, avversione al duplo (p. 131) [A].
4. Comprensibilità: oltre una certa velocità servono i testi scritti (p. 131) [A, osservazione personale].

## Limiti

- Corpus piccolo (75), t-test multipli (Bonferroni su 15); il «pilastro» dipende dalla segmentazione per respiri.

## Implicazioni per MusicMix

- Riferimenti di distribuzione per i nostri brani: durata delle frasi 2,37 ± 0,43 s, saturazione 11,1 ± 1,45, 5,09 ± 0,71 sill/s (confronti in `runs/dj_phrase_v2`).
- Il pilastro per brano entra come termine opzionale della probabilità di fine verso (`PARAMS["pillar_weight"]`, default 0 = v1).
- Contrasto di saturazione/velocità tra uscita di A e ingresso di B = «flip» del flow (diagnostico).
