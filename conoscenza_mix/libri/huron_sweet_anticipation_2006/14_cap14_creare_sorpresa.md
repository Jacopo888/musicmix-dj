# Huron 2006 — cap. 14 «Creating Surprise» (pp. 269–304)

## Tesi

- Quattro sorprese: **schematica, dinamica, veridica, cosciente** (pp. 269–270) [T]. La familiarità estingue la sorpresa veridica ma non quella schematica.
- Le sorprese funzionano meglio se **preparate** da un contesto forte (abiti, luogo, stile stabilito); senza contesto non c'è aspettativa da violare (pp. 270–271) [T].
- **Schematiche**: cadenza d'inganno; accordi di mediante cromatica (6 per tonalità) → qualia «sorpresa luminosa» (maggiori) o «sorpresa oscura» (minori) (Tab. 14.1–14.2, pp. 271–275) [E informale con musicisti e non].
- **Veridiche**: Schickele cambia una nota del tema di Beethoven (5ª, 2° mov.) in una più probabile (≈400 volte) e chiude la frase → risata; opposto psicologico della cadenza d'inganno (pp. 275–277) [A + calcolo].
- **Dinamiche**: Haydn «Sorpresa» (accordo ff al 2° beat alla seconda esposizione), Beethoven «Ode to Joy» (4ª frase entra un quarto prima del downbeat → descritta come «thrilling») (pp. 278–279) [A, sondaggio informale].
- **Garden path** (Temperley): l'inizio induce il metro sbagliato e poi va riletto (Beethoven op. 14 n. 2) (pp. 279–281).
- **Frisson** (Sloboda 1991, 83 risposte): brividi correlati a **cambi armonici improvvisi** e **cambi bruschi di dinamica** (lacrime: appoggiature, sequenze); Schoenberg «Verklärte Nacht» m. 229 (tre sorprese insieme: accordo, ff dopo pp, downbeat dopo metro eroso); Pink Floyd «The Final Cut» ~2:50 (passaggio da pp a ff, Panksepp 1995); il frisson **resiste a ≥50 ascolti** e può crescere con la familiarità; anche il subito piano può darlo (pp. 281–283) [E].
- **Risata** (Schickele, >600 risate in 4 CD dal vivo): nove dispositivi, tutti violazioni d'aspettativa — suoni incongrui, **generi mescolati** (alto→basso fa ridere più di basso→alto), tonalità che deriva (tritono; «pump up» comune nel pop, «pump down» rarissimo), metro zoppo, ritardi implausibili (appoggiatura tenuta 4 battute: risata entro 1 s dal downbeat atteso), **ripetizione eccessiva** (un passaggio di 8 note ripetuto 12 volte: **il pubblico ride durante la 4ª ripetizione, coerente con una norma di 3 ripetizioni prima di cambiare**), incompetenza, citazione incongrua, citazione sbagliata (pp. 283–287) [E/A, registrazioni dal vivo].
- La risata richiede violazioni molto grandi + segnali di intento giocoso + contesto sociale (pp. 287–288) [T].
- **Awe**: rara; ipotesi: forte, cori, molti strumenti, accordi tenuti, spazio riverberante (The Matrix, Holst «Neptune») (pp. 288–290) [speculativo].
- **Weirdness**: violazioni senza valutazione positiva (note «sbagliate» di Prokofiev, ripetute subito per mostrarne l'intenzionalità) (pp. 290–292).
- **Ascolti ripetuti**: l'umorismo si consuma e diventa «strano»; il frisson resta; l'awe diventa frisson (pp. 292–294) [osservazione + T].
- **Sincope** = sorpresa istituzionalizzata: l'attacco atteso alla posizione metrica più forte successiva non arriva → tensione (attesa); tipi: onset, dinamica, armonica, agogica, tonale, mista; la sincope **non distrugge mai il metro** (pp. 294–303). Rag di Joplin: ~1 sincope per battuta, raramente 3; 1890–1940 la quantità di sincopi raddoppia ma i pattern restano (Huron & Ommen) (pp. 301–303) [E].

## Regole operative

1. Una sorpresa piacevole richiede un **contesto stabilito** (schema forte) e un esito valutato bene (pp. 270, 304) [T].
2. **Massimo ~3 ripetizioni identiche** di un pattern prima che la quarta sia percepita come anomala (in un contesto «classico») (p. 286) [E singolo caso, dal vivo].
3. Brividi: cambio armonico improvviso + cambio di dinamica, idealmente coincidenti con un downbeat (pp. 282–283) [E].
4. Un mix di generi alto→basso è comico più del contrario (p. 287) [osservazione].
5. La sincope presuppone e preserva il metro (p. 303) [T].

## Limiti

- Esempi soprattutto classici; dati di frisson da autoselezione (questionari); il limite di 3 ripetizioni viene da **un solo** passaggio umoristico e da un contesto in cui la ripetizione era già marcata (non da musica dance).

## Implicazioni per MusicMix

- **Loop di estensione** [E→H]: ripetere un frammento **vocale** riconoscibile più di 3 volte rischia l'effetto «disco rotto» (p. 286); per uno strumentale di groove la soglia è più alta (la musica dance ripete per definizione: Margulis distingue ripetizione testuale/di sfondo). In `dj_phrase.BOOK_RULES` `loop_vocal_max_repeats = 2` è più prudente; aggiunta la fonte Huron p. 286 per il valore 3 come limite superiore.
- **Transizioni con cambio di tonalità** [H]: un «pump up» (salto di tono verso l'alto) è un cliché pop riconosciuto; un salto verso il basso è raro e può sembrare comico/strano → nel blend tra brani in tonalità diverse preferire il passaggio che sale (Camelot +1/+2 semitoni) a quello che scende, a parità di distanza [H da validare all'ascolto].
- **Frisson nel drop** [H]: il momento di ingresso pieno di B può dare brividi se coincide con un cambio armonico e un aumento di densità/intensità dopo una riduzione (break), sul downbeat ipermetrico; AGENTS.md: l'intensità non è il volume del master → ottenerla con arrangiamento (rientro di basso/kick), loudness pareggiata.
- **Cambi di genere** [H]: un salto «alto→basso» (es. brano serio → brano party) può essere letto come comico; usarlo solo se voluto.
- **Giudice automatico** [H]: la sorpresa positiva richiede contesto stabile: il giudice dovrebbe premiare violazioni di *contenuto* solo se il *quando* è rispettato (sincope sì, perdita del metro no).
