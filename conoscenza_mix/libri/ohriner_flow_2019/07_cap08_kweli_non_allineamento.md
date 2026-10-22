# Ohriner 2019 — cap. 8 «Flow and Free Rhythm in Talib Kweli» e Appendice 2 (pp. 182–204, 209–210)

Letto: testo completo + `Scripts/AnalysisScripts/NonAlignmentFunctions.r` (commit `faaec231`).

## Tesi

- «Off-beat» non è binario: il non allineamento tra sillabe e beat si scompone in **sfasamento (phase shift)**, **swing**, **cambio di tempo (tempo shift)** e **decelerazione** di fine frase (allungamento prosodico del parlato) (pp. 188–200) [A].
- Un flow è davvero «off-beat» solo se il non allineamento non è spiegato da questi aggiustamenti (p. 200) [A].
- Il non allineamento è anche un mezzo di **rottura** (Rose 1994, Chang 2006: l'hip-hop valorizza flow e rottura): Kweli passa da un flow spiegabile musicalmente a uno «parlato» nelle sezioni estreme della strofa (pp. 203–204) [T/A].

## Metodi (pp. 184–188, App. 2)

- Corpus non quantizzato: 30 strofe di Kweli (~6 500 sillabe) e 13 strofe del corpus generale con a cappella utilizzabili (~2 220 sillabe) (pp. 183, 187).
- Onset della sillaba = istante in cui la vocale raggiunge metà della sua ampiezza massima (p-center pratico, Villing et al. 2011); allineamento forzato P2FA corretto a mano; differenza mediana tra due correzioni dello stesso brano 5 ms (vocali), 11 ms (consonanti) (p. 187).
- Beat dagli attacchi di cassa/rullante annotati in due punti lontani; allineamento a cappella–mix con due sillabe; a cappella e mix possono differire fino a 10 ms/s di durata (deriva da conversioni di formato) (p. 188; App. 2: 5 ms/s).
- **Spirale**: un giro = un beat, «minuti» = sessantesimi di beat (p. 185).
- Non allineamento medio = media |onset continuo − posizione quantizzata| in sessantesimi di beat; aggiustamenti ottimi per ricerca su griglia: tempo (moltiplicativo, attorno al centro), fase (additiva), swing (sposta solo le posizioni 1mod2: BUR di Benadon 2006); in caso di pari merito si sceglie l'aggiustamento più neutro (codice `OptimizeAlignment`).

## Dati e numeri

- JND tra onset: 25–40 ms (Levitin et al. 2000); in sequenze isocrone fino a 6 ms o 2,5 % del beat (Friberg & Sundberg 1995) (p. 185 n. 2) [E citato].
- «Get By» (2002, 0:21–0:27): ogni sillaba in ritardo, metà tra 6 e 8 «minuti»; fase ottima −6,5 min; BUR 1,34:1; con fase+swing+tempo il non allineamento scende da 6,65 a < 1 min/sillaba (tempo +1 %) (pp. 190–196).
- Kweli vs genere: fase M = −10,65 vs −6,84 min (t(17) = 3,09); in ms −0,12 vs −0,08 beat; tempo 0,99 vs 1,00 (t(45) = −3,04); saturazione 12,52 vs 11,08 (p. 192 n. 9, p. 196 n. 18, p. 197 n. 19) [E].
- **Swing**: due terzi delle battute del genere non migliorano con l'aggiustamento di swing (60 % in Kweli); swing raro nel rap (p. 195) [E].
- **Decelerazione di fine frase**: pendenza media positiva delle durate delle ultime sillabe (≥ 4 sedicesimi finali) — genere 0,11 (DS 0,2) > Kweli 0,05 (DS 0,11), t(64) = 2,40 (pp. 199–200 n. 24) [E]: l'allungamento finale del parlato è presente nel rap.
- Non allineamento non aggiustato: Kweli 11,43 vs genere 7,92 min (t(350) = 7,87); quota spiegata dagli aggiustamenti uguale nei due corpora (pp. 201–203) [E].
- Tab. 8.1 (p. 203): sezioni di «Get By» — hook con non allineamento ottimo 1,28–1,55 e 81–86 % spiegato; «Flow 4» (0:58–1:06) 4,69 → 3,39, solo 28 % spiegato = vero «off-beat».
- BUR del jazz: da ~1:1 a tempi veloci a 3,5:1 a tempi lenti (Benadon 2006; Friberg & Sundström 2002) (pp. 194–195).

## Regole operative

1. Misurare l'allineamento di una strofa: fase, swing (BUR) e tempo ottimi per battuta; ciò che resta è «parlato» (p. 196) [A].
2. **Allungamento di fine frase** misurabile come pendenza delle durate delle ultime sillabe (p. 199) [E]: canale coerente con l'«allungamento finale» già usato in `dj_phrase` (Wightman 1992; tabella MCFlow).
3. Uno sfasamento costante di 6–15 «minuti» (≈ 0,1–0,25 beat, 30–100 ms a 90 BPM) è normale e stilistico (pp. 190–192) [E].

## Limiti

- 13 strofe del genere con a cappella; il p-center è convenzionale.

## Implicazioni per MusicMix

- **Le nostre parole Whisper e la griglia del beat** hanno errori dello stesso ordine dello sfasamento stilistico (100–200 ms per Whisper, vedi QUANDO_TAGLIARE): prima di quantizzare le sillabe conviene stimare la **fase ottima per brano** (mediana dello scarto tra onset vocali e griglia) e sottrarla. Implementato come diagnostica `dj_phrase.vocal_phase_alignment` (sfasamento, swing, tempo per battuta con ricerca su griglia sugli onset dello stem) [A port semplificato].
- Lo sfasamento di A e di B è uno dei motivi per cui un taglio «sul battere» può troncare: il pickup di B parte prima o dopo la griglia di qualche decina di ms (si somma al margine di 40 ms dello switch).
- Rottura: passare da un flow «musicale» a uno «parlato» è un gesto espressivo; in un mix un ingresso di B su una sezione «off-beat» suonerà come errore se non preparato [H].

## Verifica sui nostri brani (21/10/2026)

`runs/dj_phrase_v2/compare.json`, 29 brani con ASR affidabile: con gli onset delle parole Whisper la
fase ottima mediana è 0 e la ricerca di fase+swing spiega solo il 5 % del disallineamento (BUR mediano
1,05). Il test sintetico recupera ritardi di 0,05 beat e swing 2:1, quindi il limite è la precisione
degli onset Whisper, non il metodo: la misura resta diagnostica finché gli onset non vengono dallo stem
vocale (proposta F5 in `PROPOSTE_OHRINER_KRIMS_HURON_MARGULIS.md`).
