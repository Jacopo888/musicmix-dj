# Krims 2000 — cap. 3 «The musical poetics of a "revolutionary" identity» (pp. 93–122)

Letto: testo completo (OCR; le figure 3.2–3.4, grafici del layering, sono in parte illeggibili nell'OCR).

## Tesi

- L'identità culturale è un processo relazionale e conteso (Clifford 1988; Stokes 1994): la musica non marca soltanto uno spazio sociale, lo trasforma (pp. 93–95) [T].
- In Ice Cube «The Nigga Ya Love to Hate» (*AmeriKKKa's Most Wanted*, 1990, prod. The Bomb Squad) i processi delle tracce musicali — densità della tessitura, materiale intonato vs percussivo, conflitto duplo/terzina — accompagnano e trasformano il senso del testo (educazione vs intrattenimento; rifiuto della danza televisiva) (pp. 102–122) [A].

## Metodi

- **Layering graph** (pp. 97–99): 88 battute in colonne; righe per *configurazioni* (combinazioni stabili di tracce), *upbeat* (combinazioni di una battuta che precedono un punto di articolazione formale: inizio di strofe, ritornelli), *adjunct* (tracce sovrapposte di una battuta), *refrain*. BC (beat class) 1–4 con suddivisioni x, y, z (16 per battuta) (p. 99).
- Parametri seguiti: densità di attacchi per cella e quantità di materiale intonato vs non intonato (p. 105).

## Dati (dall'analisi del singolo brano) [A]

- Il brano si divide in **gruppi di 4 battute** marcati da cesure ritmiche e semantiche del rap; gli eventi musicali iniziano all'inizio di un gruppo o **nell'ultima battuta di un gruppo** (upbeat) (p. 98).
- Strofe: 1–24, 33–56, 65–86; ritornelli di 7 battute (25–31, 57–63).
- Densità: configurazione 1 (7 attacchi di percussione, basso synth con 3 attacchi a fine battuta) → refrain 1 (10 attacchi, 8 di basso) → refrain 2 (ostinato di ottoni con 13 attacchi) → **climax di densità in config. 1′ (m. 65)** → config. 3 (solo basso, 14 attacchi di percussione) → config. 4 (sola percussione, 9 attacchi) → ritorno a config. 1 (m. 81) (pp. 110–111).
- Nella strofa finale le configurazioni cambiano con proporzione **4:4:4:4:8** versi; dalla m. 79 un adjunct per verso: un ciclo regolare di cambi di strato ogni 4 battute (p. 118).
- Conflitto duplo/terzina: basso con terzine nella prima metà della battuta (1, 1z, 2y) e duple nella seconda; la terzina isolata del cimbalo all'upbeat 3 (m. 44–45) coincide con «Tom, Dick, and Hank»; da m. 80 le terzine saturano gli adjunct; l'ultimo verso è isolatamente duplo (pp. 112–116).
- n. 24 p. 121: il rapporto tra ritmo rappato e musica diventa progressivamente più obliquo e complesso nel corso del brano (anche in un brano dei Public Enemy, Walser 1995).

## Regole operative

1. **Struttura a 4 battute con «upbeat»**: i cambi di tessitura cadono sull'inizio di un gruppo di 4 o sono annunciati da una battuta di upbeat nell'ultima battuta del gruppo precedente (p. 98) [A, un brano; coerente con Huron sui fill, pp. 247–250].
2. **Arco di densità** dentro un brano: crescita graduale degli attacchi e del materiale intonato fino a un climax, poi diradamento verso la sola percussione (pp. 110–111) [A] — un brano rap ha un suo arco di tessitura misurabile come densità di attacchi per battuta, indipendente dal volume.
3. Un'anomalia locale di ritmo (terzina in contesto duplo o viceversa) marca un momento semantico importante (pp. 114–116) [A].

## Limiti

- Un solo brano, ascolto dell'autore; nessuna misura statistica.

## Implicazioni per MusicMix

- **Cue d'ingresso/uscita** [H, coerente con la pratica DJ già implementata]: cercare l'ingresso di B all'inizio di un gruppo di 4 battute; una battuta di «upbeat»/fill alla fine del gruppo precedente è un annuncio naturale della transizione (proposta per `dj_select`: rilevare la battuta di upbeat come aumento di attacchi percussivi o cambio di strato nell'ultima battuta del gruppo).
- **Arco del set**: la densità di attacchi e la quantità di materiale intonato per battuta sono descrittori dell'energia della tessitura alternativi al loudness (AGENTS.md: tensione ≠ volume).
