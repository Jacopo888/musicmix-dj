# Ohriner 2019 — Introduzione, convenzioni e cap. 1 «Flow in Rap Music: Sources of Confusion and a Strategy for Clarity» (pp. xxiii–xli, 3–30)

Fonte: M. Ohriner, *Flow: The Rhythmic Voice in Rap Music*, Oxford UP 2019, doi:10.1093/oso/9780190670412.001.0001. Letto: testo completo (copia privata dell'utente, non nel repo). Codice e dati ufficiali del libro: github.com/mohriner/flowBook @ `faaec23176acf7af339fdda5b119f4aa586bbbda` (clone locale in `../flowBook`, **non** copiato: il repository non dichiara una licenza; usiamo solo statistiche derivate e una reimplementazione in Python).

Legenda: **[T]** teoria/quadro concettuale; **[E]** evidenza empirica con N; **[A]** analisi o algoritmo del singolo autore; **[H]** nostra ipotesi.

## Tesi

- Il flow è «l'interazione tra i costituenti derivati (rima, accento, groove) e il metro» (p. 28, Es. 1.19) [T]. Costituenti primari: parole, fraseggio, ritmo (meno controversi); derivati: rima, accento, groove (p. 28).
- Invece di definire il flow da ciò che gli MC *dicono*, Ohriner osserva cosa *cambia* quando un MC annuncia nel testo che il flow è cambiato («flipping the flow», pp. 16–28) [A].
- La «legge del flow» prescrittiva (le sillabe devono stare sul beat; Edwards 2009 p. 63, Chuck D in Alim 2006 p. 96) non descrive la pratica: conta che sillabe *accentate o in rima* stiano in *posizioni metriche simili* tra un verso e l'altro (p. 13, analisi di Ice-T «I Ain't New Ta This», 1:14–1:20) [A].

## Metodi e convenzioni (pp. xxxvii–xl, 8–9)

- Griglia di 16 posizioni per battuta, numerate 0–15; 0mod4 = sul beat (0, 4, 8, 12); 0mod2 = posizioni pari (beat e metà beat) più forti; 1mod2 = dispari (deboli) (pp. 9, xxxviii).
- Quantizzazione interpretativa delle sillabe (atto interpretativo, p. 12); tactus scelto in modo che il rullante cada su 4 e 12 e la cassa su 0 e 8 («boom-bap», p. 18).
- dseg (segmento di durate, parentesi quadre), position class set (parentesi tonde), classe di groove (parentesi angolari, rotazioni equivalenti) (p. xxxviii).
- Trascrizioni come «righello» per battuta, cerchi grandi = accenti (algoritmo cap. 3), triangoli = correzioni manuali (p. xxxix).

## Dati e numeri

- Una forma di «flow» compare in circa il 20 % delle ~55 000 trascrizioni OHHLA (2016); ~10 000 occorrenze su 14 milioni di parole, 8 volte più frequente che nell'American National Corpus (pp. 5–6, n. 7) [E, corpus testuale].
- Classified «Still Got It»: posizioni 0mod2 occupate 23 volte contro 17 per 1mod2 (p. 9) [A, un esempio].

## Esempi analizzati (con minutaggi)

| Brano | Minutaggio | Cosa mostra |
|---|---|---|
| Classified, «Still Got It» (2009) | 0:40–0:52 | ambiguità lessicale come «flow» (p. 7) |
| Ice-T, «I Ain't New Ta This» (1993) | 1:14–1:20 | «sloppy» rima con «copy» ma su posizione 14 invece di 12: conta la posizione parallela della rima (pp. 12–14) |
| Eminem, «Business» (2002) | 0:42–0:52 | vocali aperte ogni ~3 sillabe, ciclo [655]: patterning proprio, non subordinato al beat (pp. 14–16) |
| OutKast, «Flip Flop Rock» (2003) | 1:42–2:02 | flip = più sillabe nelle posizioni prima vuote (ISI più brevi) (p. 17) |
| A$AP Rocky, «Purple Swag: Chapter 2» (2011) | 0:54–1:03 | flip = cambio di cardinalità C24 → C32 (p. 17) |
| Eminem, «Take from Me» (2011) | 2:03–2:16 | dieci sillabe in poco più di un secondo (p. 18) |
| Lil' Wayne, «Weezy Baby» (2005) | 0:37–0:50 | «più veloce» = più rime per unità di tempo (IRI più brevi), non più sillabe (p. 19) |
| Twista, «Say What?» (1992) | 1:16–1:21 | rime fisse su 6 e 10; flip = C24 e rima anticipata su 8 (pp. 19–20) |
| Fat Joe, «Breathe and Stop» (2006) | 2:18–2:30 | flip momentaneo: la rima cade un beat prima e poi torna (p. 20) |
| Flobots, «Airplane Mode» (2010) | 3:14–3:40 | IRI di 3 beat che «derivano» rispetto alla battuta (dissonanza di raggruppamento) e poi si riallineano (pp. 21–22) |
| T.I., «Sorry» (2012) | 2:11–2:26 | IRI di 1,5 beat instabili (p. 22) |
| KRS-One, «Don't Get So High» (2008) | 1:05–1:17 | flip = disgregazione degli IRI di 4 beat senza un nuovo schema (pp. 23–24) |
| Krizz Kaliko ft. Tech N9ne, «Strange» (2012) | 4:08–4:31 | flip = nuove classi di rima e nuovo groove di IAI ([2233222] → [3322222], rotazione a velocità doppia) (pp. 24–28) |

## Regole operative (con pagina)

1. **Parametri del flow** da rappresentare (p. 24, p. 28, Tab. 1.1 p. 30): durata sillabica (ISI), cardinalità dello spazio metrico (C16/C24/C32), posizione metrica della rima, IRI (inter-rhyme intervals), variabilità degli IRI (in beat e sotto il beat), IAI (inter-accent intervals) e il loro patterning (groove). [A]
2. **Default del rap**: IRI di 4 beat e rime vicino alla posizione 12 (beat 4) (p. 19) [A, osservazione generale poi quantificata al cap. 5].
3. **Un flip del flow** = cambio brusco di uno di questi parametri (p. 29) [T]. Per MusicMix: un cambio di brano che cambia densità, posizione delle rime o groove è percepibile come «flip» (vedi `dj_phrase.flow_compatibility`).
4. **Swing** nel rap è comune (n. 24 p. 26) [A]: la griglia a sedicesimi dritti è un'approssimazione.
5. **Tabella 1.1 (p. 30)**: elenco delle «global features» (sillabe/parola, sillabe/s, lunghezza delle frasi e sua variazione, proporzione di posizioni occupate e sua variazione, posizione di fine frase; numero di istanze per classe di rima, sillabe per istanza, sovrapposizione delle classi, complessità dello schema, variabilità degli IRI sopra e sotto il beat; tempo, patterning degli IAI, esattezza, tipicità, uniformità, sincope, accenti 1mod2). Sono riprodotte in `dj_phrase` (vedi `03_cap04_vocal_groove.md` e `04_cap05_global_features.md`).

## Limiti

- Analisi fondata su trascrizioni simboliche quantizzate a mano; la quantizzazione è un atto interpretativo (p. 12; «On Reproducible Research» p. xxxiv: «the most uncertain aspect is the validity of onset quantization»).
- Gli esempi di flip sono scelti tra versi che *annunciano* il cambio: selezione mirata, non campione [A].
- Corpus statunitense; l'autore non trova differenze regionali nel ritmo (p. xxx) e non tratta il rap non anglofono (p. xxx): per il pool francese/tedesco/russo di MusicMix il trasferimento è [H].

## Implicazioni per MusicMix

- Le misure da estrarre dall'audio sono quelle della Tab. 1.1; l'accento va calcolato sul metro (non dal volume) e la posizione delle rime conta più dell'allineamento al beat (p. 13).
- «Flow continuity» tra uscita di A e ingresso di B va descritta con gli stessi parametri del flip (densità, posizione della rima, IRI, groove); un contrasto non è un errore ma un «flip» (diagnostico).
