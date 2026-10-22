# Ohriner 2019 — cap. 3 «From Rhythm to Accent, from Sound to Rhyme» (pp. 47–70)

Letto: testo completo + codice ufficiale `Scripts/DataScrubbingScripts/AppendAccent.r`, `AppendCMU.Accent.r` (commit `faaec231`).

## Tesi

- Nessuna teoria dell'accento esistente basta per il rap (pp. 47–56) [T]: quelle musicali (GTTM: accento fenomenico, strutturale, metrico, p. 48; Roeder 1995) presuppongono altezze discrete e durate varie; nel rap le durate sono uniformi (2/3 delle sillabe durano un sedicesimo, p. 51) e beat metrico e fenomenico coincidono quasi sempre (boom-bap, p. 51); quelle fonologiche (Hayes, Selkirk) non sono formalizzate oltre la parola, non trattano l'AAVE (p. 55) e il rap ridistribuisce gli accenti per la rima (Black Thought, «Now or Never», p. 56).
- Serve quindi un algoritmo esplicito, basato su posizione metrica, durata e accento lessicale (pp. 56–59) [A].
- La rima favorisce la memoria (dell'MC e di chi rappa insieme) e, via Huron 2006, la ricompensa predittiva: «peaks in felt competency are most likely to align with rhyming words» e rafforzano la segmentazione (p. 64) [T].

## Dati e numeri

- Rap: le durate di un quarto di beat sono **due terzi** della distribuzione (corpus, n = 13 973 eventi < 4 beat) contro durate varie in Beethoven op. 18 n. 1 (n = 4 467) (pp. 50–51, Es. 3.3) [E].
- Es. 3.5 (p. 53): gli accenti primari CMU cadono soprattutto su 0mod4 e 2mod4, i non-accenti su 1mod2; 6,6 % delle sillabe non sta su un sedicesimo (scartate) [E].
- Algoritmo «giusto» nel **96 %** dei casi rispetto all'ascolto dell'autore; errori < 4 % delle sillabe; più di metà degli errori riguarda monosillabi; nel corpus sono stati corretti a mano 208 su 2 301 funzionali contro 44 su 5 382 non funzionali (rapporto 12,5) (pp. 59–62, n. 11) [A].
- Confronto con Adams 2009 e Condit-Schultz 2016 su Kurtis Blow «Basketball» (1984, 0:20–0:39): ogni sillaba non accentata dall'algoritmo è non accentata in tutte e tre le analisi; ogni accento dell'algoritmo è accentato in Condit-Schultz (p. 62) [A, una strofa].
- Hirjee & Brown 2010: rhyme score = log-verosimiglianza di un appaiamento di vocali/consonanti finali in fine verso rispetto al caso; soglia 1,5; falsi positivi 5 %, falsi negativi 15 % (pp. 66–68) [E, citato].
- Circa metà delle classi di rima del corpus sono distici (cap. 4 n. 3, p. 77) [E].

## Algoritmo d'accento (pp. 57–59) — come implementato dal codice ufficiale

Booleane per sillaba: `b` sulla griglia dell'ottavo (beatIndex mod 0,5 = 0 → posizioni 0mod2), `l` lunga (> 0,34 beat, più di una terzina di crome), `w` accento primario di parola polisillabica (CMU = 1), `n` non-accento di polisillaba, `m` monosillabo.

1. Accento se `b` e (`w` o `m` o (`n` e `l`)), oppure se non `b` e (`w` o `l`).
2. Tolto l'accento a ogni sillaba entro un sedicesimo (|Δ| ≤ 0,25 beat) da un accento di parola o da un monosillabo lungo fuori dall'ottavo (escluse queste stesse sillabe). (Nel testo: «removing accents whose positions are adjacent to word accents or long monosyllables on 1mod2 positions», p. 59.)
3. Cinque sillabe su posizioni adiacenti con accenti solo alla prima e all'ultima → accento alla sillaba centrale.
4. Gli accenti secondari CMU diventano non-accento se adiacenti a un primario, altrimenti accento primario (p. 57 n. 8).

## Rima (pp. 63–70)

- La rima non è formalizzabile come l'accento: dipende da insiemi di equivalenza tra consonanti (Brogan 2012), dall'accento, dalla pronuncia piegata alla rima, dalle rime multisillabiche non contigue e dalla funzione cognitiva (Caplan 2014) (pp. 64–67) [T].
- Limiti temporali: una rima ha un limite inferiore (le due sillabe di «Downtown» non sono rima) e superiore (Black Thought «The OtherSide», mm. 5–16: la rima a distanza di più battute è percepita? dipende dall'attenzione, p. 66) [T]. Al cap. 5 il limite superiore operativo è **8 beat** (codice `RhymeEntropyFunction.r`, `upperLimit = 8`: «my own speculation»).
- Rappresentazione: (classe di rima, indice della sillaba nella classe) (p. 68); il **momento della rima** è l'ultima sillaba accentata (in almeno un'istanza) della rima multisillabica (p. 70) [A].
- Esempi: The Roots «The Fire» (2010, 1:56–2:01) vs «Lighthouse» (2011, 2:20–2:27): stessi accenti, ma le rime interne di «Lighthouse» articolano il distico (p. 63); Treacherous Three «Feel the Heartbeat» (1981, 0:34–0:43): rima monosillabica su posizione 11 ogni 2 battute, IRI sempre 4 beat; Salt-N-Pepa «Whatta Man» (1993, 0:36–0:55): classi mono- e multisillabiche sovrapposte, IRI da < 1 beat a > 2 battute (pp. 68–69).

## Regole operative

1. Accento = funzione di posizione 0mod2, durata > 0,34 beat e accento lessicale; non volume (p. 57) [A]. Implementato in `dj_phrase.ohriner_accents` (port fedele del codice).
2. Il momento percettivo di una rima multisillabica = ultima sillaba accentata (p. 70) [A]. `dj_phrase` cerca la rima sulla sillaba accentata finale della parola: coerente.
3. Finestra percettiva delle rime: ≤ 8 beat (cap. 5, codice) [A, speculazione dichiarata]. `dj_phrase` cerca rime a 1–2 battute (4–8 beat) ±1 beat: compatibile.

## Limiti

- L'algoritmo è tarato sull'ascolto dell'autore (96 %); il confronto esterno è su una strofa.
- Gli accenti dipendono da una quantizzazione manuale; con onset Whisper (errore 100–200 ms ≈ 1 sedicesimo a 90 BPM) la distinzione 0mod2/1mod2 è fragile.

## Verifica nostra (MCFlow, 124 brani; `tools/mcflow_flow_stats.py`)

- Algoritmo di Ohriner con CMUdict su 64 554 sillabe sulla griglia: accordo con l'accento annotato a mano in MCFlow **88,5 %** (precisione 0,90, richiamo 0,91); la regola «de-accenta le parole funzionali» peggiora (87,2 %) → non attiva di default. (Riferimento dell'autore: 96 % contro il proprio ascolto.)
- Ulteriore verifica sul corpus dell'autore (stesse sillabe, suo campo `accent`): vedi `03_cap04_vocal_groove.md` §Verifica.

## Implicazioni per MusicMix

- Canale «accento dell'ultima sillaba» come indizio di fine verso: **testato su MCFlow held-out, nessun guadagno** (AUC 0,918 → 0,915): non usato.
- Accenti → IAI → groove (cap. 4) sono invece la base per descrivere il flow di uscita/ingresso.
