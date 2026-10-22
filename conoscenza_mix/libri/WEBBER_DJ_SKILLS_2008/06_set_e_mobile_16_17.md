# Webber 2008 — Capp. 16 "Set Building Skills" (pp. 240–244) e 17 "Mobile DJ Skills" (pp. 245–247)

## Cap. 16 — Tesi
Un set è "a group of records, spun one after the other … to create a complete musical presentation", da 30 minuti a 14 ore (p. 240). Va costruito sul pubblico, ma deve anche piacere al DJ.

## Qualità di un set (pp. 240–241) [P]
1. **Mixing skill**: mix "flawless, interesting, and creative"; scelta di blend, trucchi, scratch, effetti.
2. **Track selection**: equilibrio tra brani popolari/accessibili e brani nuovi "that go well together".
3. **Programming**: "Does your set go anywhere? Does the audience feel like the set progresses smoothly toward some goal? Are they exhausted 20 minutes into the set…?" (p. 241).
4. **Originality**: suono, trucchi, selezione riconoscibili.

## Fattori che influenzano il set (p. 241)
Umore ed energia del DJ, ora, dimensione del locale e della folla, tipo ed età del pubblico, luogo, tema della serata, **bravura del DJ precedente**, clima e attualità. Equilibrio tra "educare" e fare il "human jukebox" (p. 242).

## Pianificato vs. "on the fly" (pp. 242–243)
- Il set pianificato non risponde alla sala; quello improvvisato rischia accostamenti sbagliati ma permette il "serendipitous, perfect mix".
- **Chunking** (p. 243): conoscere blocchi di **2–3 brani** che stanno bene insieme e disporli secondo la serata.
- Mother Earth (p. 243): i DJ bravi "don't start their set off with the most popular thing, it goes up from low to high then it goes back down", e passano da old school a new school e ritorno.
- Demo CD: senza "train wrecks"; editare i propri errori è barare finché non si è dimostrato di saper suonare dal vivo (pp. 243–244).
- Bad sets e "the zone" (p. 244): imparare dagli errori; non pensare troppo ai dettagli tecnici.

## Cap. 17 — Mobile DJ (pp. 245–247), intervista a Chris Roman [P]
- Servizio: lasciare l'ego alla porta (p. 245). Il beat-mix "may not be required or may even be discouraged" (p. 246).
- **Apertura** (p. 247): "Don't play the hot stuff too early … Play the 'almost classics' they know for the first 20–30 minutes"; quando il **30–50 %** del pubblico si muove, mettere il brano più caldo, così lo si può ripetere a fine serata.
- Sempre attrezzatura di riserva (p. 247).

## Regole operative (con pagina)
| # | Regola | Pagina | Etichetta |
|---|---|---|---|
| P1 | Il set deve andare da qualche parte, senza esaurire il pubblico presto | p. 241 | [P] |
| P2 | Arco: partire basso, salire, poi ridiscendere; non iniziare col brano più popolare | p. 243 | [P] (testimonianza) |
| P3 | Lavorare per blocchi di 2–3 brani già provati insieme | p. 243 | [P] |
| P4 | Apertura con brani noti ma non i più caldi; il picco dopo che la pista si è riempita | p. 247 | [P] (contesto mobile) |
| P5 | Equilibrio tra brani noti e brani nuovi | pp. 240, 242 | [P] |

## Implicazioni per MusicMix
- P1–P2: l'arco "single" di `dj_set` (sale fino al ~70 % e poi scende) è coerente con Mother Earth; l'arco "waves" (profilo `books`) aggiunge una flessione intermedia come in B&B cap. 23. Nessuna delle due forme è validata su ascolti.
- P3 (chunking): **proposta** per `dj_set.order_set`: premiare coppie già giudicate buone dall'utente (diagnostica separata, non usarle come validazione).
- P4: in assenza di un pubblico reale, "brani noti" può essere approssimato solo con metadati di popolarità, oggi assenti: lasciato come ignoto.

## Limiti
Indicazioni qualitative; nessuna durata per brano, nessuna misura di energia. P2 è una frase di una promoter, non una regola dell'autore.
