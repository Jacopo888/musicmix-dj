# Butler 2006 — Cap. 5 "Multimeasure Patterning" (pp. 179–200)

## Tesi
Tra la battuta e la forma del brano ci sono raggruppamenti di più battute (2, 4, 8, 16, 32) che il produttore crea e il DJ "fraseggia". Nella techno non sono *frasi* tonali (mancano armonia direzionale e cadenze, p. 182–183) ma pattern multi-battuta che in parte funzionano da **ipermetro**.

## Strumenti analitici [A]
- Suoni **rhythmic** (loop), **articulative** (brevi, intermittenti, ai confini delle sezioni: campioni "sound-byte", colpi), **atmospheric** (sfondi senza ritmo chiaro) (p. 179).
- Grafico tessiturale: una riga per strato, 16 battute per riga, "EFX" quando un cambio timbrico è ritmicamente significativo (p. 179–180).

## Come si creano i gruppi di battute (pp. 180–188) [A]
1. **Loop lunghi** (2–4 battute; nella trance bassi di 16–32 battute con un accordo ogni 4) — spesso un loop di 1 battuta con una variazione nella **quarta** (p. 181).
2. **Aggiunta/rimozione di strati**: il mezzo più comune nella techno; i cambi più forti riguardano **basso o cassa** (p. 181). L'aggiunta è più forte della rimozione (p. 188).
3. **Suoni articolativi** ai confini (p. 181).
- Gli strati restano dentro o fuori per **almeno 4 battute**; cambi brevi (1–2 battiti o 1–2 battute, quasi sempre rimozione della cassa) cadono **subito prima di un downbeat metrico o ipermetrico** e segnano la forma meglio dei cambi graduali (p. 188). Es. Jeff Mills toglie tutto per 2 battiti prima della battuta 25, che apre il gruppo di 16 più denso (pp. 188–189); in "Cubik" tagli di 2 o 4 battiti lasciano in evidenza un solo strato appena entrato (p. 189).
- I loop **intonati** (soprattutto il **basso** melodico con una progressione) rendono l'ipermetro molto più chiaro di quelli solo percussivi (pp. 190–191).
- Un tratto senza ipermetro sembra più lungo di uno di pari durata con ipermetro (p. 193).

## Il DJ e il fraseggio (pp. 183–184, 192–193) [P riportata]
- DJ Shiva: un buon DJ fa attenzione alla "composition, like how you phrase things"; "you layer the outro beats and the intro beats of the next song, and then the breaks all fall in the same spots. So it locks"; "I hate it when people drop the 1 beat of the next record on the 15 over here" (pp. 183–184).
- Mystik: si toglie il basso "in fours": alla fine dell'ultimo dei quattro gruppi di quattro battute (cioè a fine 16) "I'll drop it out maybe for four beats, and you bring it back in — and that's perfect timing" (p. 192–193).
- I DJ contano "in four" a ogni livello (battito, battuta, gruppo) (p. 192).
- Un tratto senza cassa a lungo (breakdown) può culminare in un **accelerando** di effetti che porta al ritorno del groove (Azzido Da Bass "Dooms Night", mm. 113–117) (p. 199).

## Turning the beat around e ipermetro (pp. 194–197)
- Il TBA prima di un downbeat ipermetrico **nega** l'accento atteso: è più disturbante di uno che avviene sul downbeat o subito dopo (p. 196–197).

## Regole operative ricavate
| # | Regola | Pagina | Etichetta |
|---|---|---|---|
| M1 | Allineare le sezioni dei due brani in modo che i loro confini (break, ingressi) cadano negli stessi punti; mai l'uno di B sulla battuta 15 di A | p. 184 | [P] riportata |
| M2 | Rimozione breve della cassa (≈1 battuta) alla fine di un gruppo di 16 per marcare il nuovo inizio | pp. 188, 192 | [P]+[A] |
| M3 | Gli ingressi di strati (soprattutto basso e cassa) avvengono su confini di 4/8/16 battute | pp. 181, 188 | [A] |
| M4 | Il basso melodico porta l'ipermetro: è lo strato che rende chiaro dove si è nella frase | p. 191 | [A] |
| M5 | Un "fill"/rimozione breve va **prima** del downbeat, non dopo | p. 188 | [A] |

## Feature computabili (in `dj_form.py`)
- `layer_change` per battuta: ingresso/uscita di uno stem (drums, bass, other, vocals) → confini di sezione candidati (M3).
- `kick_dropout_short`: assenza di cassa per 1–2 battiti/1–2 battute seguita da ritorno, con posizione rispetto alla griglia di 4/16 → "fill" anacrusico (M2, M5).
- `bass_entry`: prima battuta con basso dopo almeno 4 battute senza basso (M3, M4).

## Limiti
Pochi brani techno; le interviste sono di DJ locali. "Perfect timing" e "locks" sono giudizi pratici, non misure. Nella musica con canto (pop/hip-hop) le frasi vocali creano confini propri che Butler non tratta.
