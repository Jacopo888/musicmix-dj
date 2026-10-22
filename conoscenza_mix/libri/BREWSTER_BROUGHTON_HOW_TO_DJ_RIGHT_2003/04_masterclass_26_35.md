# B&B 2003 — Capp. 26–35 ("The masterclass")

## Cap. 26 "How to build a collection" (p. 146) [P]
- Vecchi classici ben piazzati "bring a dancefloor to its knees"; "Great records don't stop being great because they're old".

## Cap. 27 "How to improve your sound" (p. 152) [P]
- **Abbassare per suonare più forte**: meno gain, meno master, meno bassi → meno distorsione.
- **Livelli**: gain del canale tarato sul passaggio più denso del brano, che "occasionally touches but never stays in the red" (~0 dB); idem in uscita.
- **Gain structure** (tecnici del Fabric): errore tipico = alzare l'inizio quieto di un brano e non riabbassare quando cresce → "They build, build, build, till … pushing the system"; senza margine non si costruisce la dinamica del set. I sistemi grandi hanno compressori di protezione: oltre il rosso il suono peggiora.
- EQ della sala: meglio togliere ciò che è troppo e alzare il volume che aggiungere ciò che manca.

## Cap. 28 "How to EQ" [P]
- Pulire i blend: bassi tolti e **scambio delle linee di basso**; togliere gli alti al brano più brillante.
- Creativo: togliere medi+bassi (suono "tinny, anxious" → desiderio che torni il basso) o medi+alti (cupo, "in agguato"); **"Slam in the bass"**: tagliare i bassi per qualche battuta durante una salita e rimetterli "bamm" al momento giusto (es. quando rientra il basso); alzare gli alti durante il taglio dei bassi.
- Tom Middleton: con kick e basso la gente resta piantata; togliendo il basso e aggiungendo alti "people just put their hands in the air".
- Prudenza: un brano ben masterizzato va suonato senza EQ; "Too much brash Eqing can become tiresome"; non si fa un'acappella col solo EQ.

## Cap. 29 "How to add FX" (p. 160) [P]
- Effetti per evidenziare/esagerare punti (salite, picchi). **"Moderation, please … Effects should be your secret weapon, not your constant companion."**
- Catalogo: delay, **echo** ("staple of reggae and other booming bouncy music"), reverb, phasing ("Good for buildups"), flanging, chorus, tremolo, distortion, panning, **filters** (passa-basso lascia passare il basso e taglia medi e alti).

## Cap. 30 "How to hot-mix" [P]
- Campionatori/loop: "You need a 20-second capacity to sample eight bars of house; more for slower music." **Live re-edit** (estendere intro, break e breakdown), **sound-bed** (loop di basso+percussioni in un canale su cui mixare altre parti/brani), "drop a sample in" (un'acappella o un frammento), FX sul loop per creare progressione (buildup prima di ricadere nel brano).

## Cap. 31 "How to mix harmonically" (p. 168) [P]
- Compatibili con una tonalità: stessa; relativa minore; quarta (sottodominante); quinta (dominante). **Camelot**: "same or next to the key code … (not including diagonally)", es. 6A ↔ 5A, 7A, 6B.
- **Cambio di tonalità**: salire di un passo = "renewed energy"; "For the biggest key change effect, **cut** to a record one step to the right."
- Con tonalità diverse: "keep melodies and basslines apart … do some kind of cut or try to exactly match the end of one melody with the start of the new one".
- Usi armonici: blend impercettibili, melodia/voce di un brano sul basso di un altro, **acappella su dub**, bootleg 2-in-1.
- Pitch senza master tempo: ±3 % cambia la tonalità in modo udibile.
- Molti brani dance sono in minore (anche cap. 45).

## Cap. 32 "How to do deck effects" (p. 176) [P]
- Tre deck: due beat + un'acappella; **repeat tricks** con due copie sfasate (ripetere l'ultima battuta/mezza battuta/battito: "tease before a payoff", "repeat the last part of a verse"); **phasing** manuale con due copie quasi in fase ("great in a big buildup").

## Cap. 33 "How to back-cue" (p. 178) [P]
- "Back-cueing, looping or 'the Quick Mix,' as … Grandmaster Flash called it, is **the foundation of hip hop**."
- Esercizio: due copie, break di almeno 8 battute; "After **eight bars**, cut on-beat from 1 to 2" e viceversa; poi ogni **4**, poi ogni 2 battute.
- Flash (citato): "if I take the **most climactic part** of these records and just string 'em together and play 'em on time, back-to-back-to-back, I'm going to have the crowd totally excited."
- Segni sull'etichetta (orologio), cueing a vista.

## Cap. 34 "How to scratch" [P]
- "Can you make it funky? The best turntablists can actually make people dance"; gli originatori (Flash, Theodore, DXT) "know how to rock a dancefloor".
- Materiale ideale: suoni lunghi e continui (vocali "oooh", sirene, note lunghe); dischi conosciuti ("because you know the records, so you know how the DJ is deconstructing it", Mark Rae).
- Baby, tear, uso del crossfader col "cut-in point".

## Cap. 35 "How to beat-juggle" (p. 196) [P]
- Beat-chop (kick e rullante singoli), juggling con due copie, **strobing**, **fills** (raddoppiare i colpi tirando indietro una copia di mezzo battito; "tripling").

## Regole operative (sintesi)
| Regola | Cap. |
|---|---|
| Livello di riferimento sul passaggio più denso; non spingere i livelli brano dopo brano (margine) | 27 |
| Bass swap; "slam in the bass" dopo qualche battuta senza bassi | 28 |
| Effetti con moderazione | 29 |
| Loop/sound-bed per estendere e collegare; acappella sopra | 30 |
| Camelot ±1 o relativa; per cambiare tonalità: cut di un passo | 31 |
| Repeat trick: ripetere l'ultima battuta della strofa come teaser | 32 |
| Quick mix: cut sull'uno ogni 8 (poi 4) battute tra le parti "più culminanti" | 33 |
| Scratch e trucchi devono restare ballabili ("make it funky") | 34 |

## Per MusicMix
- Implementato: moderazione effetti (`FX_MARKED`, `w_fx_run` in dj_set; `fx_budget_penalty` in dj_policy).
- Proposte (PROPOSTE): gain structure/headroom nel section-gain del renderer; "slam in the bass" come variante del blend (bassi di B fuori per le ultime 2–4 battute prima di un ingresso forte, poi tutti insieme sul downbeat); "repeat trick" sull'ultima battuta della strofa di A prima del cut; key change "+1 Camelot" premiato quando l'arco sale.
