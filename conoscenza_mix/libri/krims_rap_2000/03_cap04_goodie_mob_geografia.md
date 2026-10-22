# Krims 2000 — cap. 4 «Rap geography and soul food» (pp. 123–151)

Letto: testo completo (OCR; le figure 4.2–4.5, griglie di attacchi in sedicesimi, sono illeggibili nell'OCR: si riportano solo le conclusioni dell'autore).

## Tesi

- La geografia del rap è costruita per negazione rispetto al centro New York: chi viene da fuori «marca» la propria località (nomi dei gruppi, testi, copertine), New York resta «non marcata» (pp. 123–124) [T].
- In Goodie MoB «Soul Food» (album *Soul Food*, 1995, prod. Organized Noise) suono, flow, testo e video costruiscono insieme un'identità «Southern» comunitaria e una tensione rurale/urbano (pp. 124–151) [A].
- Lo stile musicale non è separabile dal significato culturale: l'arrivo dell'effusive nella seconda parte del brano è insieme convenzione di genere e segno di urbanità (p. 145) [T].

## Metodi

- Griglie di attacchi per battuta (BC 0–3 × sedicesimi 0–3, una riga per verso) come nel cap. 2; per il flow veloce di Khujo suddivisione in trentaduesimi (p. 134).
- Analisi di testo completo (Fig. 4.1, pp. 130–132), video, interviste, stampa (Rap Pages, The Source).

## Dati ed esempi [A, un brano]

- **Tessitura**: lento per il rap, dominato dai bassi, quasi **senza layering** (pochi cambi di strato) → indice di stile «Southern» nel sistema dei generi (p. 126); senza chitarre e basso funk.
- **Ogni verso = una battuta**; la prima sillaba cade sul battere o è la prima dopo il battere (p. 129).
- **Strofa 1 (T-Mo), Fig. 4.2**: rime in distici che arrivano su **BC 3** (inizio del quarto beat), una sola eccezione un sedicesimo prima («crew»); cesure deboli regolari su BC 1 e BC 2 (o un sedicesimo prima) → stile **sung** (pp. 132–133). Tratti elencati: rime finali nella stessa posizione metrica, ripetizione ritmica, coincidenza tra raggruppamento ritmico e sintattico, versi raggruppati in unità pari (p. 133).
- La pausa alla fine del quarto verso (nessun upbeat verso il verso seguente) chiude l'unità di due distici e la separa dal resto (p. 133) — **assenza di pickup = chiusura**.
- **Fine della strofa di Khujo, Fig. 4.3**: cesure imprevedibili, raffiche di sillabe in posizioni metriche variabili, nessuna ripetizione, nessuna rima fino all'ultimo distico → **speech-effusive** (p. 135).
- Andamento generale del brano: da sung verso speech-effusive man mano che si alternano gli MC; **Cee-Lo inverte il processo**: strofa che parte complessa e chiude con un distico sung prima del ritornello (Fig. 4.4, pp. 135–136). Quindi un **distico sung** prepara l'arrivo del ritornello cantato.
- **Ritornello**: cantato all'unisono/omofonico, armonie strette, poco melismatico, stile gospel/soul più antico; 4 battute (pp. 127, 138).
- **Eventi di layering** (rari, quindi marcati): dopo il primo ritornello tutto si ferma tranne il synth tenuto sull'entrata di Khujo («Sunday morning where you eating at?»), poi rientro pieno (p. 137); il **secondo ritornello è troncato dopo 1 battuta su 4** e tutte le tracce spariscono sotto «Hold up Cee…» di Big Gipp: il momento formale più forte del brano secondo le reazioni degli ascoltatori a cui l'autore lo fa sentire (pp. 138–139) [osservazione informale, N non dato].
- Spiegazione dell'effetto: non solo la durata attesa (4 battute, proporzionata alle strofe), ma la rottura di una relazione: il ritornello «liscio» compensava l'MCing sempre più effusive (pp. 138–139) [A].

## Regole operative

1. **Rime finali sul quarto beat (BC 3) + cesure regolari = sung** (pp. 132–133) [A] → conferma il descrittore `end_beat4` e `end_R` di `dj_phrase.krims_style` (sung).
2. **Un verso che non ha upbeat verso il successivo chiude un'unità formale** (p. 133) [A] → una fine di verso senza pickup è un candidato di uscita più forte; un verso con pickup prosegue.
3. **Il distico sung prima del ritornello** è un segnale di arrivo (p. 135) [A, un brano].
4. **Un ritornello stabilito a 4 battute crea l'attesa della sua durata**; troncarlo dopo 1 battuta, con stop totale delle tracce, è l'evento più forte (pp. 138–139) [A, reazioni informali] → coerente con Huron (aspettativa schematica sulla durata delle sezioni) [H].
5. **Layering raro = layering marcato**: in un brano con poca variazione di tessitura, un solo stop pesa più che in un brano stratificato (p. 137) [A].
6. Progressione sung → effusive dentro il brano (e dentro le strofe) come tratto di genere dell'«hardcore» (p. 149) [T/osservazione].

## Limiti

- Un solo brano e il suo video; nessuna misura; letture culturali (rurale/urbano, «double consciousness») non trasferibili a regole numeriche.

## Implicazioni per MusicMix

- **Punto di uscita da A** [H]: preferire una fine verso **senza pickup** (la voce si ferma prima del battere successivo) a fine di un gruppo pari di versi; misurabile con `next_onset_gap` (già nel config `dj_phrase_flow_v1.json`); su MCFlow il guadagno di questo termine per prevedere la fine verso è marginale (vedi `ohriner_flow_2019/02_cap03`), quindi resta un indicatore di *chiusura*, non di fine verso.
- **Stop/break come raccordo** [H]: uno stop totale delle tracce (solo voce o un pad) è il raccordo più marcato; vale di più se il brano A ha poca variazione di tessitura. Da usare con parsimonia (una volta per set) e solo su un cambio di sezione.
- **Durata delle sezioni** [H]: non troncare una sezione stabilita (es. ritornello di 4/8 battute) a meno di volerlo come effetto; la troncatura è percepita.
- **Coppie** [H]: un passaggio da un brano sung a uno speech-effusive replica la progressione di genere (p. 149); il passaggio opposto è un «ritorno» — da usare in fase di chiusura o per far arrivare un ritornello cantato.
