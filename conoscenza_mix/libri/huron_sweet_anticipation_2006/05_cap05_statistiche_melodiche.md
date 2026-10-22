# Huron 2006 — cap. 5 «Statistical Properties of Music» (pp. 73–89)

## Tesi

Cinque regolarità melodiche robuste e le aspettative corrispondenti (p. 74) [E su corpora + esperimenti]:

1. **Prossimità**: intervalli piccoli predominano in 10 culture (Fig. 5.1 p. 74); gli ascoltatori elaborano più in fretta gli intervalli piccoli (Deutsch 1978; Aarden) (p. 75).
2. **Step declination**: gli intervalli grandi salgono più spesso, i piccoli scendono più spesso (Vos & Troost 1989, Fig. 5.2 p. 76); analogo alla **declinazione dell'intonazione nel parlato** (la frase sale presto e poi scende lentamente) (pp. 76–77).
3. **Step inertia**: nei corpora solo i gradi discendenti tendono a proseguire (70 % vs 51/49 % per gli ascendenti, Tab. 5.1 p. 78), ma i musicisti si aspettano inerzia in entrambe le direzioni (von Hippel; Aarden); i non musicisti no (pp. 77–80). Euristica «sbagliata» ma con la stessa accuratezza (~62 %) (p. 79).
4. **Regressione melodica**: dopo un salto la melodia torna verso la media; analisi su centinaia di melodie: la «post-skip reversal» non spiega **nulla** del residuo dopo la regressione alla media (solo Palestrina, poco) (Fig. 5.5 pp. 81–84). Gli ascoltatori musicisti invece si aspettano la reversal (euristica giusta nel ~70 % dei casi, più economica) (p. 85).
5. **Arco melodico**: su >6.000 canti popolari (Essen, ~10.000 frasi) il 40 % delle frasi è convesso, 4 volte più dei concavi; frasi ascendenti seguite da discendenti (arco su due frasi) ma non viceversa (p. 86); media di 6.364 frasi di 7 note ad arco (Fig. 5.6); dip centrale («McDonald's effect») per frasi ≥12 note (Fig. 5.7 p. 87). Aarden: gli ascoltatori sono più rapidi sugli intervalli **discendenti nella seconda metà della frase**, non sugli ascendenti nella prima (p. 88).

## Regole operative

1. **La fine di frase attesa è discendente** (seconda metà della frase, p. 88) [E, musicisti] — e nel parlato la frase declina (p. 77).
2. Gli ascoltatori usano euristiche semplici e quasi ottimali, non la statistica «vera» (pp. 79, 85) [E].
3. Le aspettative melodiche dettagliate sono più nette nei musicisti che nei non musicisti (pp. 79–80, 85) [E].

## Limiti

- Corpora notati, soprattutto melodie vocali popolari e classiche; nessun dato su rap/EDM; le misure degli esperimenti citati (N) non sono riportate nel capitolo.

## Implicazioni per MusicMix

- **Fine verso nel rap** [H, testabile]: la **declinazione della F0** nella seconda metà della linea è un indizio indipendente dalle pause ASR; con lo stem vocale (Demucs) e un tracker di pitch si può misurare la pendenza della F0 nelle ultime sillabe prima di un candidato fine-verso. AGENTS.md vieta di equiparare fine frase a pausa: questa è una seconda evidenza da combinare, non un sostituto. Proposta in `PROPOSTE_*` (non implementata: richiede F0 per sillaba).
- **Coppie di frasi** [H]: frase ascendente → aspettativa di una discendente (arco su due frasi, p. 86): uscire da A dopo la frase «di risposta» (discendente) è più chiuso che dopo la frase «di domanda».
- **Pubblico** [H]: in un club la maggior parte degli ascoltatori non sono musicisti → le aspettative fini (melodiche) contano meno di quelle metriche e di sezione.
