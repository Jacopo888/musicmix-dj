# Huron 2006 — cap. 10 «Expectation in Time» (pp. 175–202)

Capitolo centrale per MusicMix (il *quando*).

## Tesi e dati

- **Tactus ottimale**: periodo 0,6–0,75 s (80–100 BPM): massima precisione nel tapping e tempo di tapping spontaneo; chi batte più veloce o più lento deriva verso questo valore (Woodrow, Fraisse, Povel & Essens) (pp. 175–176) [E].
- **Attenzione periodica** (Jones): l'attenzione è massima nei punti metrici forti; giudizi d'altezza più accurati quando il tono cade sul downbeat atteso e peggiori se anticipato o ritardato (Jones et al. 2002, Fig. 10.1–10.2 pp. 176–177) [E].
- L'**assenza** di un evento atteso provoca attività mentale (telefono che squilla a vuoto) (p. 178) [T].
- **Gerarchia metrica = probabilità di attacco** (canti infantili portoricani, Fig. 10.3) e **ipermetro**: nell'op. 54 n. 1 di Haydn i quattro downbeat di un gruppo di 4 battute seguono il profilo forte–debole–medio–debole (Fig. 10.4 pp. 179–180); battute raggruppate in periodi di 8 e doppi periodi di 16, spesso coincidenti con le frasi (p. 179) [E/A].
- Palmer & Krumhansl: probe-tone metrico, i toni sui beat più forti «stanno meglio» (p. 179); posizioni metriche percepite in modo **categoriale** come i gradi (p. 179) [E].
- Dipendenze a lungo raggio (m-dependency, Simpson): in Bach preludio in Do picchi a 8 e 16 note (Fig. 10.6) (pp. 181–183) [E].
- **Piacere del downbeat**: evento più prevedibile → prediction effect misattribuito; i **downbeat ipermetrici** (inizio di periodo/doppio periodo) suonano particolarmente «belli» (p. 184) [T]. Sincope, emiola e downbeat mancanti piacciono per altro meccanismo (contrastive valence, cap. 14) (p. 185).
- **Schemi ritmici di genere** (siciliano, Fig. 10.8–10.9; samba, reggae, backbeat rock…) appresi per esposizione (pp. 185–187) [E/T].
- Ritmi **non periodici ma prevedibili** (rimbalzo, aksak 3+2+2, valzer viennese con 2° beat anticipato e 3° ritardato, Moelants) → la base della previsione temporale è la **prevedibilità, non la periodicità** (pp. 187–188, 199) [E/T].
- **Ritmo del parlato**: sillabe non periodiche ma prevedibili; Patel & Daniele (300 temi) e Huron & Ollen (~10 volte tanti): melodie strumentali inglesi alternano lunghe/brevi più di quelle francesi (indice di variabilità delle durate successive, nPVI) (pp. 188–189) [E].
- **Categorie ritmiche** (Desain, Honing & Sadakata): i ritmi di tre note sono percepiti per categorie la cui estensione riflette la frequenza nel repertorio (regola di Bayes) (Fig. 10.13–10.15 pp. 191–194) [E].
- **Default binario**: nel repertorio classico 66 % metri binari/quaternari; EEG (Brochard et al.): in una sequenza isocrona gli eventi dispari sono accentati soggettivamente (pp. 195–196) [E].
- **Cosa + quando in poesia**: metro regolare + schema di rime rendono prevedibile **quando** cade e **come suona** l'ultima sillaba del verso → elaborazione facilitata e piacere (Frost, pp. 196–197) [T]. Ridondanza voluta in comunicazioni rumorose (alfabeto ICAO, versi di Shakespeare) (p. 197).
- **Raggruppamento ≠ metro**: le durate lunghe chiudono un gruppo, gli accenti dinamici lo aprono (Woodrow; Lerdahl & Jackendoff) (p. 197); un gruppo è tenuto in memoria a breve come unità; i click inseriti migrano percettivamente verso i **confini di gruppo** (Gregory); pattern di **1–2 s** come «atomi» (p. 198) [E].
- Il cervello predice **attacchi**, non intervalli inter-onset (pp. 199–200) [T]; due ritmi sono più simili se si aggiungono note nelle posizioni metriche probabili o se ne tolgono dalle improbabili (p. 200) [osservazione].
- La **sincope** è attribuita alla nota che precede il downbeat vuoto (Fig. 10.16 pp. 200–201) [T].
- Asimmetria anticipo/ritardo e «delay» rimandati al cap. 15 (p. 202).

## Regole operative

1. Ingressi e cambi vanno sui **downbeat ipermetrici** (inizio di gruppi di 4/8/16 battute): massima attenzione e massimo premio di previsione (pp. 176–180, 184) [E + T].
2. Un evento leggermente fuori dal punto atteso viene percepito peggio (p. 177) [E] → precisione di beatmatching/allineamento.
3. Fine di gruppo = **durata lunga**; inizio = accento (p. 197) [E].
4. Rima + metro rendono prevedibile la fine del verso (p. 196) [T].
5. Le unità ritmiche percettive durano ~1–2 s (p. 198) [E].

## Limiti

- Esperimenti su ascoltatori occidentali; dati di corpus su musica classica e folk; nessun dato su EDM/rap; l'apprendimento statistico del ritmo è indiretto (detto dall'autore, p. 201).

## Implicazioni per MusicMix

- **Cue e ingresso** [E→H]: conferma la regola DJ «cue sul downbeat a inizio frase» e la preferenza per confini a 8/16 battute (già in `dj_select`/`dj_phrase`); l'ingresso di B deve cadere esattamente sul downbeat (errore di allineamento percepibile).
- **Fine verso nel rap** [H, implementato come descrittore]: il verso rap è un caso di «cosa + quando» (rima sulla posizione metrica attesa; Ohriner cap. 3 e Krims cap. 2/4). Le feature `end_beat4`/`end_R`/pillar di `dj_phrase` misurano proprio la prevedibilità del *quando* della fine verso.
- **Durata lunga a fine gruppo** [E]: coerente con l'accento «l» di Ohriner (sillaba > 0,34 beat) e con la durata dell'ultima sillaba come indizio di fine verso già usato.
- **nPVI** (Patel & Daniele) [H]: indice di variabilità delle durate successive delle sillabe come descrittore di flow (alternanza lungo/breve vs isocronia «a macchina da cucire»), calcolabile sulle parole Whisper → implementato in `dj_phrase.syllable_npvi` (diagnostica, non usato nella selezione).
- **Tempo** [H]: 80–100 BPM è il tactus più preciso; i brani rap a ~85–95 BPM sono nella zona ottimale; con brani house a 120–128 BPM il tactus percepito può raddoppiare/dimezzare → le coppie a tempo «metà/doppio» (85 ↔ 170) sono compatibili sul tactus.
