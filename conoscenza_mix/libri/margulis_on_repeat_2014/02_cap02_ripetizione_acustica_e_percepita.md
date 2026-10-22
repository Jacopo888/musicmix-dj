# Margulis 2014 — cap. 2 «From Acoustic to Perceived Repetition» (pp. 26–54)

## Tesi

- Ripetizione acustica ≠ ripetizione percepita: il **contesto** (ciò che precede/segue, autore, storia, posizione metrica) cambia il senso e persino il **suono** di un passaggio identico (Borges/Menard; campionamento) (pp. 26–33) [T].
- L'**hip hop** nasce dai DJ che ripetevano i **break** per far continuare a ballare tra un brano e l'altro; poi il break campionato diventa la base di un nuovo brano (p. 28) [osservazione storica].
- Ricontestualizzazione (Hanninen 2003): ogni ripetizione è insieme uguale e diversa (p. 32) [T].
- Una ripetizione **non è stasi**: secondo GTTM produce sempre tensione o rilassamento (anticipazione prolungazionale vs ripetizione che intensifica) (pp. 36–37) [T].
- Il silenzio, a parità di durata, è percepito più lungo/corto, teso/rilassato a seconda che la chiusura tonale sia stata raggiunta (Margulis 2007a) (p. 33) [E].
- La ripetizione rende salienti le poche cose che cambiano (posizione metrica) e porta in primo piano l'**ipermetro**; come il silenzio riduce l'informazione nuova, ma **preserva il metro**, mentre il silenzio spesso lo interrompe (p. 41) [T/A].
- Si riconosce come ripetizione solo ciò che è già un'**unità** («thing»): motivi brevi dentro gesti più lunghi che proseguono diversamente **non vengono mai riconosciuti** (0 identificazioni in 4 brani × 4 ascolti, Margulis 2012) (pp. 38–40) [E]; una battuta con funzione di **fine** e la stessa con funzione di **inizio** non sono percepite come ripetizione (Haydn, Sonata Hob. XVI/26) (pp. 42–43) [E].
- **Dimensione dell'unità ripetuta** (Rameau, Tambourin; Feld 2005): unità piccole = impressione ravvicinata e intensa; unità grandi = paesaggio, distanza; restringere l'unità da 2 a 1 battuta produce **accelerazione percepita**, ritornare a 8 battute «allarga» (p. 44) [A].
- Ripetizione **prossimale** (immediata) vs **distale** («ritorno»; Meyer 1973) (p. 44).
- Ripetizione «museumatica» (riff brevi in loop) vs «discorsiva» (sezioni) (Middleton 2006); nei loop densi l'ascoltatore costruisce il proprio percorso d'attenzione (Garcia 2005) (p. 50) [T].
- **Lidov 2004** (pp. 51–52) [T]:
  - ripetizione **formativa** (convenzionale, definisce le unità, non attira attenzione) vs **focale** (attira attenzione sul fatto stesso di ripetere);
  - diventa focale se **attraversa un confine di segmento più grande** o se è **tripla/quadrupla** (oltre modello + copia);
  - **oltre 3–4 volte** diventa **testurale**: «annulla la propria pretesa di attenzione» e sposta il fuoco su un'altra voce o su un aspetto che cambia.
- Meyer: la ripetizione aumenta la tensione, ma prolungata porta a **saturazione** (p. 51); la battuta delle barzellette americane arriva alla terza ripetizione (p. 51); il piacere ha un picco dopo un numero moderato di ripetizioni (rimando al cap. 5).
- Praeger (1882) esclude da ogni critica le **musiche ritmiche/da ballo**, dove la musica sostiene il movimento (p. 52).
- Debussy: 60–70 % delle frasi ripetute subito (DeVoto) (p. 52).
- La ripetizione aiuta la **separazione delle sorgenti**: un suono non rilevabile in una miscela diventa identificabile se ricorre in miscele diverse (McDermott et al. 2011); REPET separa voce e accompagnamento ripetitivo (pp. 53–54) [E].

## Regole operative

1. **1–2 ripetizioni** = formative; **3–4** = focali (attirano attenzione); **>4** = testurali (sfondo, l'attenzione va altrove) (Lidov, pp. 51–52) [T]. *[Precisazione 21/10/2026: Lidov propone categorie teoriche («three-or-fourfold», «beyond the three-or-four mark»), non un massimo empirico di loop; ogni soglia derivata è [H].]*
2. Restringere l'unità ripetuta = accelerazione percepita; allargarla = distensione (p. 44) [A].
3. La ripetizione preserva il metro; il silenzio lo può rompere (p. 41) [T].
4. Lo stesso materiale con funzione diversa (fine vs inizio) non è sentito come ripetizione (p. 43) [E].

## Limiti

- Classificazioni teoriche (Lidov, Middleton) non misurate; esempi classici.

## Implicazioni per MusicMix

- **`loop_repetition_check`** in `dj_phrase` [T→implementato]: soglie `loop_establish_repeats = 2`, `loop_focal_repeats = 4`, oltre le quali il loop deve diventare testurale (con un nuovo strato che entra: `loop_with_new_layer_max_bars = 8`), fondate su Lidov via Margulis p. 51 e su Huron p. 286.
- **Loop d'uscita di A** [H]: un loop di frase **vocale** che attraversa un confine di sezione è «focale» (attira attenzione) → va fatto solo se voluto; un loop di batteria/strumentale ripetuto >4 volte diventa sfondo ed è adatto sotto l'ingresso di B (l'attenzione si sposta sulla voce nuova): è esattamente il ruolo che Lidov assegna alla ripetizione testurale.
- **Build** [A→H]: dimezzare l'unità di loop (8→4→2→1) = accelerazione percepita senza cambiare BPM.
- **Uscita/ingresso** [E→H]: se B entra con lo stesso materiale di A (es. stesso campione), la funzione diversa (inizio vs fine) lo rende «nuovo»; il riuso di un break comune è nella storia stessa dell'hip hop (p. 28).
- **Separazione** [E→H]: la ripetizione rende più facile separare una voce che entra sopra un accompagnamento ripetitivo: la voce di B sopra un loop di A è più intelligibile se il loop è stabile.
