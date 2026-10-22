# Ohriner 2019 — cap. 4 «From Accent to Vocal Groove» (pp. 71–103)

Letto: testo completo + codice ufficiale (`GetL3Durations.R`, `GetVerseGrooves.r`, `SegmentVerseGrooves.r`, `groove dsegs (prime form).r`, `GetGrooveComplexity.r`, `GrooveSwapDistance.r`, `GroovinessFeatureFunctions.R`, commit `faaec231`).

## Tesi

- Il **vocal groove** — il patterning ripetuto degli intervalli tra accenti (IAI) e il suo rapporto col metro — è il tratto più emblematico del flow (p. 71) [T].
- Definizione provvisoria: la coerenza con cui ogni posizione della battuta è o non è accentata da una battuta all'altra; «confidence of prediction on the part of the listener grows as each correct prediction etches the rhythmic structure of the verse deeper into the grooves of memory» (p. 72) [T, collegato a Huron].
- Il groove vocale sta nella sola voce dell'MC (diversamente da Keil, Danielsen, Nketia), è virtuale (Deleuze/Danielsen: l'ascoltatore perdona perturbazioni) e la microtemporizzazione non ne è un costituente (Madison et al. 2011: lo swing non correla con la «grooviness» percepita) (pp. 74–83) [T].
- Ripetizione (Lidov 1979): **formativa** (richiesta dalla forma: ritornelli di 1–2 battute ripetute per riempire 4–8; distici di rime), **focale** (oltre le attese del genere, attira l'attenzione sulla ripetizione stessa), **testurale** (dopo ~5 ripetizioni, n. 4 p. 78, l'attenzione passa a «un'altra voce») (pp. 77–79). Il flow, per default, è l'«altra voce» rispetto allo strumentale testurale (p. 78) [T].
- Sincope: spostamento dell'accento su posizione debole (Temperley 1999) o poliritmo (3+3+3+3+4) (pp. 80–82); Witek 2017: i «buchi» della sincope invitano il corpo a riempirli (p. 82) [T].

## Metodi e regole formali

1. **Sette classi di groove** (p. 86): con IAI di 2 o 3 sedicesimi (nessun accento consecutivo; nessun IAI doppio di un altro, MWFR3 di GTTM) che sommano a 16: <2222_2222> (duplo), <332_332>, <332_2222>, <323_2222>, <333322>, <333232>, <3223222>. Rotazioni: 16 per le classi senza ripetizione interna, 8 per <332_332>, 2 per il duplo (n. 13).
2. Proprietà delle classi (pp. 86–89, Tab. 4.1): contorno ritmico, periodicità interna (la più lunga: <333322>, 12 unità di 3-durate), istogramma degli intervalli, entropia scalata degli IAI (0,65 duplo … 0,90 <332_2222>); <3223222> è l'unico «odd rhythm» (non biseca la battuta) e «maximally even», ma poco usato.
3. **Complessità metrica** di una rotazione (pp. 89–93, Tab. 4.2): somma sui 4 beat della complessità del tipo d'accento nel beat, stimata da −log della frequenza nel corpus: |1010| 0 (3 176 beat), |0010| 0,744 (456), |1001| 0,721 (554), |0100| 0,941 (231), |0101| 1 (161). «Lose Yourself» <332_332>@4 = 2,93; «8 Mile» @5 = 3,32; minimo con almeno una 3-durata 0,94 (<332_2222> da 2mod4). Gli accenti su 1mod4 sono solo il 6 % del corpus (p. 89) [E].
4. **Accenti limit-3** (p. 94, codice): le posizioni accentate valgono 1, le vicine 0; in ogni beat la prima e la terza posizione ancora indeterminate diventano 1 (silenzi su 0mod2 «sentiti» come accenti perché lo strumentale li riempie); le durate 4 diventano 2+2.
5. **Ascolto fuzzy** (pp. 94–98): sforzo = distanza DTW (Giorgino 2009, passo «symmetric2») tra le posizioni del groove ripetuto e gli accenti reali; spostare un accento di una posizione costa 2; tasso di sforzo = sforzo / lunghezza; lunghezza corretta l′ = l·(1 − r^¼) (p. 97). Si cercano tutte le istanze di ogni stringa di groove ripetuta 1–4 volte che iniziano e finiscono su accenti con tasso ≤ 0,25.
6. **Segmentazione** (p. 98): a 4 tassi di sforzo — 0 (ascoltatore *adattivo*), 1 swap per battuta (0,0625), 1 per mezza battuta (0,125), 1 per beat (0,25) (ascoltatore *persistente*) — si sceglie in ogni sedicesimo il groove con l′ massima; corse uguali = segmenti.

## Dati e numeri

- Tab. 4.3 (p. 101) quota di battute per classe (75 strofe, 0 swap): duplo 0,44; <332_2222> 0,27; <323_2222> 0,10; <332_332> 0,07; <333322> 0,05; <3223222> 0,04; <333232> 0,03. Con più tolleranza il duplo sale a 0,54 [E].
- 7 brani interamente duple senza swap, 19 (un quarto) a 1 swap per beat (p. 100). Rotazioni pari:dispari del duplo = 13:1 (p. 101). Le rotazioni più comuni mettono |1001| sul beat 3 (accenti su 8 e 11) (p. 101) [E].
- «Renegade» (Jay-Z ft. Eminem, 2001, 1:18–1:31): solo 13 di 24 mezze battute hanno [323]; sentire tutto come [323] costa sforzo 4, come duplo 36; >400 groove candidati all'inizio di m. 5 (pp. 95–98).
- Una strofa rap mantiene di solito un groove per 2–3 battute; eccezione: Eminem «We Made You», <323_2222>@8 per quasi 10 battute (pp. 94–95).

## Esempi

- Eminem «Lose Yourself» (2002, 0:54–1:40) vs demo (2014, 0:21–1:04): nella versione pubblicata a m. 9 cambiano le posizioni accentate (4, 6, 8, 11, 14 → 2, 4, 7, 10, 12, 15) e 13 posizioni su 16 diventano «sempre/mai/tutte-tranne-una/una sola volta» accentate: il groove marca la metà della strofa e il cambio narrativo («snap back to reality») (pp. 71–74, 98–100).
- Eminem «8 Mile» (4:05–4:13): <332_332>@5 (p. 84). Rick Ross «Aston Martin Music» (2010, 1:20–1:33) <332_2222>@8; Eminem «Go to Sleep» (2013, 0:14–0:26) @10; «The Way I Am» (2000, 2:43–2:54) @3, tra le più complesse; il beat preceduto da un accento 3mod4 perde rilievo, e il primo beat «non preempted» è il primo del ritornello (pp. 92–94).
- Eminem «Drug Ballad» (2000, 0:32–0:41): IAI con 4 e 5 → accenti limit-3 (pp. 94–95).

## Limiti

- Ciclo di 16 posizioni assunto (il rap a 12 o 24 unità esiste, p. 86); la scelta del tactus (rullante su 4 e 12) è una convenzione (cap. 1).
- Il modello fuzzy è dell'autore [A]; le soglie (tasso ≤ 0,25, segmenti ≥ 32 unità) sono scelte, non stimate su ascoltatori.

## Verifica nostra (port Python in `dj_phrase`, `tools/ohriner_port_check.py`)

Vedi `runs/dj_phrase_v2/ohriner_port_check.json` (valori riportati nel README della cartella libri e nel report): accenti su 55 935 sillabe non corrette a mano del corpus dell'autore — accordo **98,9 %**; complessità di «Lose Yourself»/«8 Mile» riprodotte esattamente (2,93 / 3,32). Le istanze e i segmenti di groove coincidono solo in parte con i file derivati dell'autore (i file `corpus_grooves.txt` risultano calcolati da una versione precedente degli accenti: stesso span con sforzo 0 nell'autore e 2 da noi per un accento spostato), vedi il report per le percentuali.

Numeri (run completa, 275 strofe, flowBook@faaec23): accenti 98,9 % su 55 935 sillabe non corrette a mano (94,2 % includendo le 2 809 correzioni manuali dell'autore, che il codice non può riprodurre); etichetta di groove per posizione uguale a quella dell'autore nell'80,1 % delle posizioni (84,2 % con tolleranza di sforzo 0,125); global features per strofa contro quelle pubblicate dall'autore: saturazione r = 1,00, sillabe/s r = 0,999, variazione di saturazione r = 0,98, grooviness r = 0,95, dupleness r = 0,89 (media 0,43 contro 0,35), adherence r = 0,67 (media 0,70 contro 0,66). Dupleness e adherence dipendono dai segmenti di groove, dove pesano gli accenti della versione precedente.

## Implicazioni per MusicMix

- `dj_phrase.groove_instances`/`groove_segments`/`ohriner_global_features` riproducono il modello: il groove dell'uscita di A e dell'ingresso di B è confrontabile con `groove_swap_distance` (0 = stesso groove; quanti «swap» servono per passare dall'uno all'altro).
- Il cambio di groove marca sezioni (Lose Yourself m. 9; Go to Sleep m. 17 con lo sparo sul battere e l'ingresso di rullante e hi-hat, cap. 6): **un cambio di groove nel flow è un indizio di confine di sezione** [A] — candidato per i cue (proposta).
- Ripetizione testurale dopo ~5 iterazioni (Lidov, n. 4 p. 78) → base per i limiti di `loop_repetition_check` (vedi Margulis).
