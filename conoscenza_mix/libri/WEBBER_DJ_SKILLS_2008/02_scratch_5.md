# Webber 2008 — Cap. 5 "The Scratch DJ Revolution" (pp. 74–97)

## Tesi
"Scratch DJ" è un termine diffuso ma riduttivo (Flash: lo scratch è solo il "right hook" del pugile, p. 74). DJ Babu: "Before you learn how to scratch, you must first learn how to mix" (p. 74). La battle (routine di circa **6 minuti**: mix, scratch, beat juggling) è un mondo a sé, con pubblico fatto soprattutto di DJ (pp. 75–76); molte battle diventano "rather tedious" perché alcuni sono più musicali di altri (p. 76).

## Tecniche, esempi, regole (con pagina)
- DMC (dal 1986 lo scratch entra nella gara con DJ Cheese) e ITF con categorie separate: scratch, beat juggling, team, "X-perimental" con effetti/CD/laptop (pp. 75–76).
- Formati dei crew: un DJ fa il beat, uno il basso, uno una figura contrappuntistica, uno l'assolo (p. 77).
- **QBert** (pp. 78–86): il suo coach (Yoga Frog) gli dice "You're scratching too much … pause more … lay off here … do some more phrasing" (p. 80); "I'm always going for it, which is bad" (p. 80). Pratica: lento → veloce; elenchi di scratch combinati come rudimenti (pp. 80–81); hamster mode per lo scratch, normale per il juggle (p. 83).
- **Kid Koala** (pp. 87–97): il DJ come narratore; con una band impara che "there's definitely a higher order of what's good for the song. We're not just here to show off" (p. 93). Un selector gli spiega che "the selection and order you put them in will affect the mood in the room" e quanto tempo dedica a imparare le transizioni ("holding the mixes in line") (pp. 96–97).

## Regole operative ricavate
| # | Regola | Pagina | Etichetta |
|---|---|---|---|
| S1 | Gli effetti di manipolazione vanno dosati: pause, spazio, fraseggio; non "always going for it" | p. 80 | [P] (autocritica di QBert) |
| S2 | Ciò che serve al brano viene prima dell'esibizione tecnica | p. 93 | [P] |
| S3 | Selezione e ordine determinano l'umore della sala | pp. 96–97 | [P] aneddoto |

## Implicazioni per MusicMix
- Sostiene un **budget di effetti marcati** nel set (già implementato come `w_fx_run` in `dj_set` e `fx_budget_penalty` in `dj_policy`, profilo `books`): uno scratch/backspin/eco ripetuto a ogni cambio è il "going for it" di QBert.
- Lo scratch vero e proprio è fuori portata della pipeline offline (richiede manipolazione del vinile e materiale apposito); resta un'ipotesi per `dj_creative` solo come "drum tip" o "scratch cut" (cap. 19).

## Limiti
Interviste; nessuna regola misurabile sul mix tra brani.
