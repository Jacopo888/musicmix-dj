# Beat This! — nota di implementazione

Letto il testo completo: [Foscarin, Schlüter e Widmer, 2024](https://arxiv.org/html/2407.21658v1), in particolare §3.1–3.3, §4.2–4.4 e Tabella 2. Il repository ufficiale collega il checkpoint `small0` alla variante piccola.

Il modello produce eventi a 50 Hz. Il postprocessing seleziona picchi e riconduce i downbeat al beat più vicino; i timestamp non hanno precisione campionaria. La loss tollera piccoli scarti nelle annotazioni. Nel §4.2 gli autori spiegano che attivazioni molto alte sono intenzionali: **non sono confidenze calibrate della correttezza temporale**. Per questo non usiamo un valore vicino a uno come prova di allineamento.

La variante piccola riduce i parametri; la valutazione riporta anche limiti nelle metriche di continuità. Questi risultati non certificano i nostri due brani.

Scelte locali, ancora da validare: contesti direzionali di quattro/otto beat; limite di spostamento dei cue di 80 ms; controllo dello scarto relativo; esclusione delle attivazioni sotto 0,20. Sono policy del progetto, non soglie raccomandate dal paper. Le sezioni spettrali restano euristiche e i confini vocali provengono da un altro modulo.
