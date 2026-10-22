# Nota di sintesi — Butler, *Unlocking the Groove* (Indiana UP 2006)

Schede per capitolo: [`BUTLER_UNLOCKING_THE_GROOVE_2006/`](BUTLER_UNLOCKING_THE_GROOVE_2006/README.md). Natura: analisi musicale [A] (trascrizioni di 9 brani techno/trance) + etnografia (12 interviste, Indiana 2001–02).

## Tesi
La dance in 4/4 sembra semplice ma offre esperienze di tempo multiple; la forma nasce da **strati** che entrano ed escono, dalla **cassa** come strato decisivo e dal lavoro del DJ, che con dischi-"tool" costruisce un set unitario tenuto insieme dall'**intensità**.

## Le regole più utili (con pagina)
1. Forma prototipo **intro → buildup → core → breakdown → (buildup) → core più intenso → outro**; mai intro→core diretto né core→buildup (pp. 220–224) [P riportata + A].
2. **Breakdown** = cassa assente per almeno 4 battute (p. 91); il **ritorno della cassa** ("dropping the beat") è il momento più forte (p. 3; pp. 245–247).
3. Rimozioni brevi della cassa (1–2 battiti, 1–2 battute) **subito prima** di un downbeat ipermetrico: Mystik toglie il basso per 4 battiti alla fine di 16 battute (pp. 188, 192, 246).
4. Allineare battute, **ipermisure e sezioni** dei due dischi; "drop the 1 beat of the next record on the 15" è un errore (pp. 54, 184, 234).
5. Mixare dove la **tessitura è sottile** (intro/outro di soli beat): meno scontri di ritmo e altezza (pp. 129, 231).
6. Mai **due voci predominanti** insieme (p. 243); mai due superfici **"busy"** (p. 244); non coprire l'elemento dominante di un brano (pp. 244–245). *[Correzione 21/10/2026, riletto il testo per pagina: due voci p. 244, "busy" p. 245; altre pagine di Butler spostate di una in `../_meta/P0_MISURE_REGOLE_2026-10-21.md` §2.]*
7. DJ Shiva: si esce nel **primo breakdown** corto (pp. 221–222); Frankie Bones salta intro e outro lunghi (p. 225 n. 22).
8. Intro senza cassa sopra un outro con cassa richiede allineamento perfetto; meglio usare la **prima cassa** come guida (p. 233).
9. Arco del set: più **picchi**, l'ultimo il più alto; tempo che varia **poco** (pp. 250–252); classici verso la fine (p. 252).
10. Sfasamento progressivo tra due tracce = beatmatch fallito (p. 154).

## Implementazione in MusicMix
- **`src/musicmix/dj_form.py`** (nuovo): strati per battuta (cassa, basso, medi, alti, voce; stem Demucs a 4 tracce se presenti, altrimenti bande del mix meno la voce), densità di attacchi, etichette intro/build/core/groove/breakdown/outro, drop e rimozioni brevi, profilo di sovrapposizione A/B. Test: `tests/test_dj_form.py`.
- **`dj_set`** (profilo `books`): `form_context` nel contesto della transizione; costo per uscita durante un build di A, bonus per uscita nel breakdown e per ingresso sul drop di B, costo dei blend con tessiture piene/"busy" e regola **R7c_dense** (blend accorciato).
- **`dj_policy`** (profilo `books`): preparazione (build) anche quando B entra sul drop.
- Già prima: arco a onde, `w_tempo_down`, durata a sezioni (core ≈ strofa+ritornello) [H].

## Limiti
Techno/trance anni '90; le soglie delle feature sono [H]. Su 4 brani hip-hop/R&B la cassa stimata dalle bande del mix concorda con quella dello stem drums solo in 2 casi su 4 (vedi `conoscenza_mix/libri/README.md`, sezione verifica): per questo il runner usa gli stem a 4 tracce quando può.
