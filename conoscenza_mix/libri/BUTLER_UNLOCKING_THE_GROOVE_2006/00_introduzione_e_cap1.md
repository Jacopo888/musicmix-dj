# Butler 2006 — Introduzione (pp. 3–31) e cap. 1 "The History and Creation of EDM" (pp. 32–75)

Tipo di fonte: **analisi musicale [A]** (trascrizioni e analisi di 9 brani techno/EDM completi, App. C) + **ricerca sul campo** (12 interviste a DJ/produttori/fan dell'Indiana 2001–02, più Carl Craig, 808 State, Underworld nel 2004). Non ci sono esperimenti percettivi.

## Introduzione — tesi e metodo
- Scena d'apertura (Detroit, DEMF 2001, DJ Stacey Pullen): il DJ toglie la cassa, la folla si gira verso di lui "expectantly"; dopo una, due, tre, quattro battute la cassa rientra "with an instantaneous flick of the wrist" e la pista esplode (p. 3). "Because the beat commands such stature, removing it and bringing it back is one of the most powerful things a DJ can do" (p. 3). [osservazione etnografica]
- Il set come "single musical tapestry that is constantly transforming" anziché brani separati (p. 3).
- Tesi del libro: la dance in 4/4 apparentemente semplice offre esperienze di tempo ricche e multiple ("unlocking the groove", p. 4–5).
- Metodo: analisi dei brani **non mixati** (non si sa dove inizia e finisce un brano in un mix) + interviste e osservazione dei set (pp. 19–20). Jimmi Journey: memorizzare i dischi "not only their beats and their measures and their times, but all the instruments involved", per trovare subito un disco che vada col disco lasciato dal DJ precedente (p. 20).
- Ambito: analisi dettagliate soprattutto **techno**; affermazioni generali su "EDM" (p. 24). Hip-hop citato solo per somiglianze di produzione (p. 6 n. 2).

## Cap. 1 — Storia e ruoli (pp. 32–75)
- Storia (pp. 33–46): disco → garage/house (Chicago) → techno (Detroit) → UK; dal 1990 proliferazione di generi; jungle = break di funk accelerati con bassi a mezzo tempo di influenza reggae (p. 46).
- **Il DJ** (pp. 47–58):
  - Il set è "a unity": flusso continuo e distinzioni tra brani **minimizzate** (p. 48); DJing come atto compositivo: sovrapporre parti di brani diversi (es. basso di un disco con una battuta di voce di un altro), collage con cut avanti e indietro (p. 48).
  - **Programming**: "playing the right music at the right time … as well as choosing tracks that work well together"; per esempio un brano famoso "at a climactic point in the evening" (p. 49). Nei generi più "compositivi" (dagli anni '90) si sentono meno brani riconoscibili (p. 49).
  - Mix sequenziale: la fine di A si sovrappone all'inizio di B; durata variabile, dalla metà di A + metà di B fino ai due dischi interi insieme; oppure saltare avanti e indietro (p. 52). Nota: il **cutting** avanti e indietro "is more characteristic of hip hop"; nella EDM è una variazione temporanea (p. 52 n. 49).
  - **EQ**: usi timbrici (adattare alla sala, far fondere due dischi, richiamare l'attenzione, "tweaking") e **testurali**: togliere una parte, "Usually this is the bass drum"; se due linee di synth creano "too much pitch dissonance, one of them could be excised from the mix" (pp. 52–53).
  - Continuità: niente silenzi **e stesso tempo** da un disco all'altro; possibile perché i DJ restano in un genere con tempi vicini e il pitch ±8 % (p. 54).
  - **Allineamento delle periodicità ampie**: "Another component of mixing, rarely mentioned … but equally important, is the alignment of larger periodicities: measures, hypermeasures, and formal sections" (p. 54); tecnica: slip-cue.
  - Battiti non allineati "stand out like a sore thumb" ("like shoes in a dryer"); il **train wreck** "can clear the floor instantly" (p. 55).
  - Effetti (delay, reverb, echo, pan, flange) e sampler (pp. 55–56). Hawtin, *Decks, EFX & 909*: 3 piatti, effetti, 909, **38 dischi in circa un'ora** (p. 56).
  - Brani su 12": 4–10 minuti, tipicamente circa 6 (p. 56).
- **Il produttore** (pp. 58–69): i produttori fanno brani "that work well in this context" sapendo che il DJ ne attenuerà le differenze (p. 58); sequencer, drum machine 808/909, 303, MPC (pp. 60–68).
- **Ruoli ibridi** (pp. 69–71): remix (spesso più suonato dell'originale), live PA.
- **Il ballerino** (pp. 71–74): "performing audience"; i DJ bravi non suonano set prestabiliti, il comportamento della pista cambia "what music will be played, when it will be played, and how" (p. 71); "vibe" collettivo (p. 71).

## Regole operative ricavate
| # | Regola | Pagina | Etichetta |
|---|---|---|---|
| U1 | Togliere e restituire la cassa è uno dei gesti più forti del DJ; il rientro è un picco di energia | p. 3 (e cap. 2 p. 91) | [A]/osservazione |
| U2 | Mixare allineando battute, ipermisure e sezioni, non solo i battiti | p. 54 | [A]+[P] |
| U3 | Con EQ si toglie di solito la cassa di un disco; se due linee melodiche dissonano si toglie una delle due | p. 53 | [P] riportata |
| U4 | Il cut avanti e indietro è tipico dell'hip-hop; nella EDM è un'eccezione | p. 52 n. 49 | [A] |
| U5 | Un brano molto noto va al punto culminante della serata | p. 49 | [P] riportata |

## Implicazioni per MusicMix
- U2 conferma il requisito "cue sul downbeat a inizio frase" (AGENTS.md) e i vincoli di `dj_select`.
- U3 corrisponde a regole già in `dj_set` (key_clash_cut, bass swap) e suggerisce in `dj_render` un **taglio di banda sulla linea melodica in conflitto** (proposta).
- U4 conferma la scelta del profilo `books` di favorire il cut nell'hip-hop (B&B cap. 11, Katz pp. 56–57).

## Limiti
Repertorio techno anni '90; campo limitato all'Indiana. Le affermazioni sul DJ derivano da interviste, non da misure.
