# Ohriner 2019 — cap. 2 «A Corpus for Rap Music Analysis» (pp. 31–46)

Letto: testo completo; verificati i file `SourceData/verseMetadata.txt`, `DerivedData/FlowBookCorpus.txt` del repository ufficiale (commit `faaec231`).

## Tesi

- Analisi di stile (norme del repertorio) e analisi critica (scelte del singolo artista contro quelle norme) sono inseparabili (Meyer 1973, p. 31–32) [T]. Un'eccezione (es. quattro rime a 3 beat di distanza in «Airplane Mode») è significativa solo rispetto a un corpus rappresentativo (pp. 31–33, n. 1).
- Rappresentare la musica in forma digitale obbliga a scegliere le caratteristiche prima dell'analisi (Marsden 2016, Selfridge-Field 1997, p. 34) [T].

## Metodi e dati

- **Campionamento** (pp. 37–39): 6 liste giornalistiche di «best emcees» (BET, Complex, Source, Metacritic ≥ 84, XXL Freshman, «slept-on»): 225 MC; per ciascuno il brano più «hot» (Echo Nest); sottocampione casuale di **75 artisti** stratificato per cronologia e regione; strofa prima/centrale/ultima scelta a caso [E: corpus N = 75 strofe]. Corpora aggiuntivi: Eminem, Black Thought, Talib Kweli (Parte II).
- Popolazione stimata: ~400 000 strofe (50 000 «masters» Discogs × 8 brani × 2 strofe, p. 36); trascrizione a mano a ~4 strofe/ora (50 anni per tutto) [stima dell'autore].
- L'allineamento forzato (Penn P2FA) fallisce sul rap nella finestra di centinaia di ms (strumentale + fonetica diversa): «manual annotation... is the only viable means» (p. 36) [A, prova informale]. **Per MusicMix**: le nostre parole Whisper sono più rumorose di una trascrizione manuale; misure fini (sedicesimi) vanno trattate come incerte.
- **Rappresentazione** (pp. 40–46): sillabe dal CMUPD sillabato (con un'appendice di ~1 000 parole AAVE/neologismi, p. 41); durata come coppia (cardinalità, numero di unità); onset tramite battuta e beat della prima sillaba, i pickup sono nella battuta −1 (p. 41); beat index continuo.
- **Fraseggio** (pp. 43–46): due segmentazioni — *phrase endings* = sillaba prima di un respiro (empirica, dall'audio); *line endings* = tanti segmenti grammaticalmente chiusi quante sono le battute. Ohriner preferisce i respiri: «emcees breathe when they believe their breaths will be unobtrusive, that is, when their listeners will not expect the completion of an ongoing idea» (p. 45).

## Esempi

- Bubba Sparxxx, «Deliverance» (2003, 1:05–1:15): ogni battuta = un verso che inizia sul/poco prima del battere e finisce con una rima sul beat 4 dove l'MC respira (p. 43–44) — caso in cui respiro e sintassi coincidono.
- The Roots ft. Erykah Badu, «You Got Me» (1999, 0:29–0:39): tre trascrizioni (Anthology of Rap, OHHLA, Ohriner per respiri) che segmentano diversamente (pp. 44–45).
- Lil' Wayne, «Weezy Baby» (2005, 0:37–0:44): la più difficile da quantizzare; le sillabe arrivano tutte dopo la griglia, alcune molto dopo (pp. 42–43, Es. 2.4).

## Regole operative

1. Fine frase = sillaba prima di un respiro; Krims (2000 pp. 100–102) invece fa iniziare ogni verso con la prima parola della battuta; Bradley/DuBois: un verso = una battuta (pp. 43–44). **Tre definizioni diverse di «verso»**: MusicMix usa i confini prosodici di MCFlow (`**break` ≥ 3), più vicini ai respiri di Ohriner che alle battute. [A]
2. Pickup: sillabe prima del primo battere annotate nella battuta −1 (p. 41) [convenzione].
3. Una pausa di respiro è un indizio di fine frase *dal punto di vista dell'MC*; non coincide con ogni pausa (AGENTS.md: fine frase ≠ pausa ASR) — Ohriner stesso tiene due segmentazioni.

## Limiti

- Liste «best of» compilate nel 2011–2013, canone con bias (donne, MC non neri, latinos: p. 38); ultime generazioni (SoundCloud, trap) sottorappresentate (p. 39 n. 6).
- Solo rap statunitense; nessun brano non inglese.

## Implicazioni per MusicMix

- Le statistiche di Ohriner sono su 75 strofe *quantizzate a mano*: possono servire da riferimento di distribuzione (come MCFlow), non da soglie per il nostro audio.
- Il respiro è un indizio di fine frase osservabile nello stem vocale (silenzio + inspirazione); oggi `dj_phrase` usa solo il silenzio. Proposta [H]: rilevatore di inspirazione (rumore a banda larga, ~200–500 ms, prima di un attacco) come canale aggiuntivo — non implementato (nessuna stima su corpus disponibile).
