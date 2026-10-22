# B&B 2003 — Capp. 20–25 ("The dancefloor")

## Cap. 20 "How to play out" (p. 110) [P]
- La preparazione conta più di tutto: portare "different moods, styles, and tempos"; non pianificare troppo: "Fight the temptation to prepare a whole set in your head. Be ready to respond to the crowd."
- Il pubblico vuole ballare, non giudica la tecnica.

## Cap. 21 "How to warm up" [P]
- Warm-up = "guiding an evening through many moods, from an empty dancefloor to peak hour and back again".
- Musica da warm-up: **melodica, familiare, tempo più lento**; brani "interessanti", vecchi classici.
- "If you think of your set graphically, it should have a **steady upward curve**"; "Increase the tempo slowly … Great DJing is about teasing and holding back."
- Fasi "From empty to full in two hours": 1) sei solo, alza gradualmente il tempo; 2) pionieri; 3) la tua entrata ("let's go" record); 4) coinvolgi chi sta ai bordi; 5) pronta al decollo; 6) "Bingo": il brano che porta da tiepido a caldo.
- Anthony Pappa: "No going over 125 bpm before midnight" (aneddoto).

## Cap. 22 "How to read a crowd" [P]
- La pista come organismo unico: energia totale da massimizzare; "party starters" e "party poopers".
- Indizi: chi lascia la pista, chi arriva, chi impazzisce, chi guarda la consolle. "Gently probe the crowd with a range of different records … faster or slower … big breakdowns or tracks that stay in a groove?"
- "Don't let them lead you, let them guide you"; "Ring the changes": cambiare stile e vedere la reazione.
- Fattori: dimensione (pubblico piccolo più sensibile), droghe/alcol, posa, **attention span** ("need a snare roll every two minutes"), conoscenza musicale.

## Cap. 23 "How to choose the next record" (p. 128) [P]
- "This is the single most essential skill of DJing … putting records in the right order is much more important" della tecnica.
- **Per mood**: livello di energia e sentimento di ogni disco; categorie "warm-up", "peak hour", "cooldown" e sottocategorie.
- **Per caratteristiche**: tempo ("you'll need to follow one record with another of **similar tempo**" salvo cambi voluti; 4–5 categorie personali di tempo), stile/strumentazione, **struttura** (ma "don't select a record purely because it mixes easily"), associazioni musicali (stesso ritmo/basso), mix collaudati (con moderazione).
- **Per significato**: testi e temi; "Telling tales" (D'Acquisto, Siano: brani in sequenza che "parlano"); Larry Levan: se cambia argomento, "a slight pause, a sound effect … to let you know it's a new paragraph".
- **Direzione**: ogni disco "perfect for now" e "pointing in the right direction"; pensare 5 dischi avanti; "It's a journey", "It's a story … a beginning, a middle, and an end".
- "Never ever plan a set" rigidamente; sì a strategia e forma generale. "On the diagonal": candidati successivi separati in borsa.

## Cap. 24 "How to pace the night" [P]
- "Your job as a DJ is to encourage and control the energy in the room"; forma "in mood and, crucially, in **tempo**": "it should be full of **undulations, peaks, and troughs**".
- Forme di set: **Up, up, and away** (sempre in salita, house moderna); **Roller coaster** (sale fino al picco e crolla a ritmo reggae, si ripete); **The workout** (lenta, sale, tiene il picco con qualche respiro, scende alla fine — set lunghi); **Riding the waves** (Mancuso: ondulato).
- "Good DJing is about keeping a constant groove while throwing in plenty of changes - balancing the expected with the surprising"; alternare "sprints and jogs".
- **Suspense**: "You make a crowd go mad by building up tension and then releasing it"; tecniche: serie di dischi in salita, "hints of a big song long before you play the whole thing", ripetitivo/ipnotico, togliere qualcosa (basso, volume, il brano).
- **Big-tune syndrome**: razionare i brani forti, costruire verso di loro.
- **Punteggiatura**: stop, ripartenze, cambi, pause; "**Fresh starts**": dopo un culmine, rallentare a un tempo totalmente diverso per ripartire. "Cleansing the floor".
- Notte intera: dividerla in sezioni (warm-up, picco, chiusura).

## Cap. 25 "How to come on after someone" [P]
- Lasciare finire l'ultimo disco del DJ precedente; **abbassare il volume** per avere "headroom" ("Give yourself somewhere to go").
- Entrata: mix senza giunte oppure qualcosa di riconoscibile/"noticeable" che annuncia il nuovo DJ.
- Dopo un DJ più veloce: "Few tracks sound good beyond **+4**"; meglio pausa e ripartenza, oppure rallentare l'ultimo brano del collega "by maybe **3 percent** without the crowd really noticing".

## Regole operative (sintesi)
| Regola | Cap. |
|---|---|
| Ordine e scelta del prossimo disco > tecnica | 23 |
| Brani successivi di tempo simile, salvo cambi voluti | 23 |
| Arco: warm-up in salita lenta (tempo ed energia), poi ondulazioni con picchi e valli; razionare i brani forti | 21, 24 |
| Dopo un culmine si può ripartire da un tempo diverso ("fresh start") | 24 |
| Accelerare lentamente; oltre +4 % i brani suonano male; −3 % passa inosservato | 21, 25 |
| Tenere margine di volume (headroom) | 25, 27 |
| Leggere la pista provando dischi diversi e osservando chi arriva/se ne va | 22 |

## Per MusicMix
- Implementato: arco `waves` (picchi e valli, l'ultimo più alto) e `w_tempo_down` (lieve preferenza per tempi in salita) nel profilo libri di dj_set.
- "Leggere la pista" non è computabile senza pubblico: la sostituzione onesta è l'**ascolto dell'utente** (giudizi) e la varietà esplorativa; il giudice deve trattare i voti come la "reazione della pista".
- "Telling tales"/significato: possibile feature lessicale tra testi consecutivi (ASR) — oggi solo `lyric_link` a livello di parola; proposta in PROPOSTE.
- Limite B&B: house/techno; i numeri (±3 %, +4 %) riguardano il pitch senza key lock.
