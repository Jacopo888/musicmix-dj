# Webber 2008 — Capp. 18 "Extending Breaks" (pp. 248–252) e 19 "Scratch DJ Skills" (pp. 253–276)

## Cap. 18 — Tesi
Estendere un break con due copie (Herc, perfezionato da Flash) serviva ai b-boy e agli MC; è la base del beat juggling (p. 248). Sul CD si può fare con il loop: "more reliable but potentially less exciting" (p. 248).

## Procedura passo-passo [P]
1. Scegliere inizio e fine della sezione; iniziare da un intro ritmico di **2–8 battute prima della voce** (es. "Low Down" di Boz Scaggs, usato da Flash) (p. 248).
2. **Marcare l'etichetta** (clock theory): a ore 12 o puntato verso la puntina; essere coerenti (pp. 248–250).
3. **Backspin di due battute** (pp. 250–251): suonare A contando 8 battiti; poco prima del nono fermare il disco e riportarlo all'inizio contando i giri.
4. **Backspin silenzioso**: crossfader dall'altra parte durante il ritorno; due battute ≈ **tre giri** (p. 251).
5. **Estensione** (pp. 251–252): A suona 8 battiti; poco prima del nono "slam the crossfader" su B rilasciato sul battito; riavvolgere A mentre suona B; ripetere.
6. Errori tipici: troppo indietro = pausa prima del downbeat; troppo poco o crossfader lento = downbeat tagliato. "**extending breaks is pointless if it doesn't groove. Accuracy and flow are of supreme importance**" (p. 252). Poi provare 1 o 4 battute.

## Cap. 19 — Tecniche (pp. 253–276) [P]
- Lo scratch "can accentuate a drum pattern, add rhythm to a vocal, give additional weight to a lyric" (p. 253).
- Senza fader: baby scratch (quarti/ottavi), drag (metà/interi), scribble (sedicesimi), tear, lazer, swipe, uzi (pp. 255–258).
- Campioni corti e "tip" (attacco) (pp. 259–260). **Drum tip come fill**: l'attacco di una cassa (o rullante) usato "as a fill leading into a downbeat"; usato anche estendendo i break perché il downbeat del break è di solito una cassa (Flash) (p. 260).
- Con il mixer: fader scratch, stab, **cutting** (il campione suona a velocità normale: utile con campioni vocali perché si capisce cosa si dice) (pp. 261–263); cutting di batteria (kick, kick-snare) (pp. 263–264).
- **Fades**: chiudere lentamente il fader su una figura ripetuta, "a fitting end to a scratch routine" (p. 265).
- **Echoes** manuali: ripetere il campione in tempo con il fader che a ogni ripetizione si ferma un po' più in basso (pp. 265–266).
- Transformer (p. 267), chirp (p. 268), hamster (p. 268), flare (pp. 269–272), crab (pp. 274–276).

## Regole operative (con pagina)
| # | Regola | Pagina | Etichetta |
|---|---|---|---|
| B1 | Il loop/estensione deve rientrare esattamente sul downbeat: né pausa né downbeat tagliato | p. 252 | [P] |
| B2 | Estensioni di 1, 2 o 4 battute; per l'intro ritmico prima della voce 2–8 battute | pp. 248, 252 | [P] |
| B3 | Un fill di cassa (drum tip) può preparare il downbeat d'ingresso | p. 260 | [P] |
| B4 | Eco "in tempo" con ripetizioni a livello decrescente | pp. 265–266 | [P] |
| B5 | Lo scratch/cutting di un frammento vocale va lasciato a velocità normale se deve essere capito | p. 262 | [P] |

## Implicazioni per MusicMix
- B1: è il criterio di correttezza per un loop di estensione e per la ripresa dopo un backspin (in `dj_creative.backspin` il B parte sul downbeat dopo `spin_beats`).
- B2: `backspin_spin_beats` (1/2/4) nel profilo `books` resta un parametro [H]; il libro parla di backspin *per ripetere* due battute, non di backspin come transizione (quella è in B&B cap. 12).
- B3, B4: proposte per `dj_creative` (fill di cassa prima del downbeat di B; echo out con decadimento a gradini in battiti).

## Limiti
Didattica per vinile; i "giri" dipendono da 33⅓ giri e tempo. Nessuna indicazione su quando usare queste tecniche in un set.
