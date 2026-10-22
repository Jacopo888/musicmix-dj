# Krims 2000 — cap. 2 «A genre system for rap music» (pp. 46–92)

Letto: testo completo (OCR).

## Tesi

- Lo stile non è una proprietà oggettiva ma un fatto discorsivo e negoziato (p. 46) [T]. I generi formano un **sistema relazionale** (Jameson 1990: la «realtà» è ripartita tra i generi e dalle loro relazioni) (pp. 80–81) [T].
- Tre parametri definiscono il genere: stile delle tracce musicali, stile di MCing (flow), argomenti (p. 55) [T].
- Il genere system nordamericano è stabile dal ~1994 (declino del gangsta classico) (p. 47).

## Tre stili di flow (pp. 48–52) — direzioni su spettri, non categorie

| Stile | Tratti (Krims) | Esempi citati |
|---|---|---|
| **sung** | ripetizione ritmica, accenti sul beat (soprattutto forti), **pause regolari sul beat, su BC 3 = inizio del quarto beat**, distici rigidi, rime finali prevedibili sul quarto beat seguite da cesura | Too $hort, Grandmaster Flash «New York, New York», Beastie Boys (rime finali sul 4° beat urlate in unisono, distici regolari), Mack 10 «Backyard Boogie» |
| **percussion-effusive** | attacchi fuori beat, consegna «sharply-attacked and crisp», punti di staccato seguiti da brevi cesure che suddividono le unità; *non necessariamente veloce* né irregolare; marca lo «stile personale» trasversale ai generi | B-Real (Cypress Hill), Ice Cube (*The Predator* 1992), Method Man, Treach, KRS-One, Ice-T; all'estremo: Bone Thugs-N-Harmony, Crucial Conflict, Mystikal, OutKast — «not-New York» |
| **speech-effusive** | enunciazione vicina al parlato, poco senso di pulsazione, poliritmi irregolari e complessi, profusione di sillabe, **molte sillabe in rima consecutive**, sintassi e rime sfalsate rispetto alle battute; spesso **una cesura dentro il primo verso di una nuova unità di 4 battute** (fine della frase precedente o frase tronca) | Big Pun (1998), Nas, AZ, Jay-Z, Wu-Tang (Raekwon, Ghostface, RZA), Ras Kass, Twista, Bone Thugs |

- «Effusive» = effusione di schemi ritmici e poliritmi che traboccano metro, distico e raggruppamenti di 2/4, non necessariamente di sillabe (pp. 50–51).
- Gli stili coesistono (Treach li unisce) e cambiano dentro una strofa: «**rhythmic acceleration**» = da sung a speech-effusive con aumento della densità di attacchi e varietà degli intervalli tra rime (Aceyalone «Mic Check» 1995; RZA in «Wu-Gambinos») (p. 52) [A].
- Il sung e il percussion-effusive coesistevano prima; lo speech-effusive è la forma «contemporanea» (anni '90) (p. 52) [osservazione storica].

## Raggruppamenti di 4 battute (pp. 52–53)

«As in much popular (and other) music, measure groupings of four are often a norm in rap»: lunghezza di strofe e ritornelli, layering, cesure, pattern armonici, upbeat e downbeat di batteria tendono a cicli di 4 battute; raggruppamenti di 16 meno pervasivi; eccezioni: un distico aggiunto a fine strofa, gruppi dispari (p. 53) [T/osservazione].

## Layering e timbro (pp. 53–54)

Layering = costruzione della tessitura sovrapponendo tracce in loop e manipolandone la densità; generalizza il «break beat». Il timbro separa gli strati (Mobb Deep «Drop a Gem On 'Em» 1996). Il **«hip-hop sublime»** (pp. 73–74): strati densi, tutti in metro, ma dissonanti e perfino non intonati tra loro (frazioni di semitono), timbri inconciliabili (vinile, live processato) → «hardness», minaccia + piacere (Burke, Jameson) [T].

## I generi (pp. 55–80)

| Genere | Tracce | Flow | Argomenti |
|---|---|---|---|
| **party** | sezione ritmica in primo piano, layering minimo, **tempi più veloci**, armonie consonanti (Will Smith «Summertime» 1991) | il più vicino al **sung**: rime finali, poche rime interne, gruppi di 2 e 4 battute; **adatto a memorizzazione e sing-along; i DJ nei club lo apprezzano per l'entusiasmo partecipativo** | festa, umorismo, sesso, celebrazione dell'hip-hop |
| **mack** | R&B, strumenti dal vivo, layering poco marcato, ritornelli cantati (spesso da donne) | **resolutamente sung** (p. 82) | seduzione, ricchezza |
| **jazz/bohemian** | campioni jazz/soul, eclettismo ironico | tra sung ed effusive; Q-Tip: rime finali + attacchi fuori beat → «meta-poliritmo», regolare solo nel breve periodo | «conscious», didattico |
| **reality** (gangsta, knowledge) | «durezza»: basso dominante non focalizzato, dissonanze, campioni deformati/rumore di vinile; G-funk (West Coast) come eccezione «licenziata» dalla geografia | **effusive** (prima percussion-, poi speech-effusive); West Coast più lento e sung («laid-back») | strada, lamenti, conoscenza |
| **don** (dal ~1993, ibrido gangsta + mack) | anche dance e R&B lussuoso, ritornelli R&B cantati | speech-effusive, smussato nei singoli radiofonici | boss criminale, fantasia, ricchezza |

- Gli album mescolano generi ma hanno un genere prevalente (p. 87); ciò che un album esclude definisce la sua posizione (A Tribe Called Quest *Beats, Rhymes and Life* 1996: party + jazz/bohemian) (pp. 87–88).
- Queen Latifah 1998 (*Order in the Court*): passaggio da sung a speech-effusive per collocarsi nel nuovo sistema (pp. 88–89).
- Il «dance-oriented» = armonia triadica, pochi strati, poca variabilità del layering e tempo più veloce (n. 59 p. 85).

## Regole operative

1. Tre direzioni di flow con tratti osservabili (tabella sopra) [T] → operazionalizzate in `dj_phrase.krims_style` con descrittori di Ohriner [H]; **verificate su MCFlow** (124 brani): i 5 brani dei Beastie Boys (esempio di «sung» di Krims, p. 50) hanno tutti punteggio «sung» sopra la mediana; lo speech-effusive cresce con l'anno di uscita (ρ = +0,32), come sostiene Krims (p. 49).
2. Indicatore speech-effusive concreto: **fine verso nella prima battuta di un gruppo di 4** (p. 50) → descrittore `first_bar_caesura`.
3. Indicatore sung concreto: pause regolari all'inizio del quarto beat e rime finali sul quarto beat (p. 50).
4. Party/sung = più adatto a ballo e partecipazione nei club (pp. 56–57) [T/osservazione]; tempi party più veloci (p. 56).
5. I raggruppamenti di 4 battute (p. 53) sono la norma: cue e loop in multipli di 4 battute [T] (già regola DJ in `dj_select`).

## Limiti

- Sistema dei generi datato (1994–1999, Nord America); nessuna misura; i nomi degli stili sono di Krims, non dei fan (p. 48).

## Implicazioni per MusicMix

- **Selezione/ordine del set** [H]: lo stile dominante di A e B (Krims) come descrittore di contrasto; un passaggio sung → speech-effusive è un «rhythmic acceleration» che Krims descrive *dentro* una strofa: in un set può costruire energia (diagnostica, da ascoltare).
- **Tipo di transizione** [H]: brani party/sung (rime finali regolari, distici) offrono fini verso prevedibili → tagli puliti; brani speech-effusive con cesure dentro la prima battuta del gruppo di 4 → evitare di far partire B sul battere di un gruppo dove A è a metà frase (già coperto da `exit_point`).
- **Coerenza timbrica** («hip-hop sublime» vs R&B liscio) è un asse separato dal flow: non stimato qui.
