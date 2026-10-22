# Katz 2012 — Cap. 8 "Falling Barriers: 2002–2011" (pp. 214–248) e "Conclusion: Full Circle" (pp. 249–253)

## Tesi
- Il digitale (CDJ 2001, sistemi a vinile di controllo come Serato/Traktor/Torq) è "the single biggest development in the history of the hip-hop DJ" (p. 214) e abbassa le barriere (costi, peso, accesso; più donne, accademie, videogiochi) [A].
- I DJ hip-hop rifiutano i CDJ (jog wheel piccola, latenza, "play themselves") ma adottano il DVS perché conserva il **vinile** come interfaccia (pp. 215–230).

## Passaggi operativi chiave
- **Durata dei brani nel set dei club DJ (p. 228)**: "Many club DJs don't play whole songs before mixing in a new tune—**two verses and chorus, or even just a chorus**—so it's crucial to be able to switch quickly and seamlessly between songs." [A, osservazione dell'autore sulla pratica dei club DJ hip-hop].
- **Quick mixing in crescita** (p. 228): col DVS si risparmiano "three or four seconds per song just cueing" (DJ Revolution); DJ A-Minor: i ballerini apprezzano "the quicker mixing style that delivers more songs per hour" — "good for the ADD crowd" [aneddoto P].
- **"The 1"** (DJ Hapa, Scratch DJ Academy, p. 232): il primo battito di una frase di 4 o 8 battute; "in order to mix records it is crucial to be able to identify 'the 1'"; prima si conta, poi lo si sente [P].
- Waveform colorate (rosso bassi, verde medi, blu alti; kick e rullanti evidenziati) → beatmatch visivo; rischio che il DJ guardi lo schermo invece della pista (pp. 223, 226) [A].
- Modalità ABS/REL del DVS; loop e cue con un tasto (pp. 222–223).
- Lezione DJ 101 (pp. 232–233): scratch su semibrevi, minime, semiminime contando "forward two three four, back two three four"; assolo improvvisato di **4 battute**; compito: sentire "the 1" in ogni brano.
- Mashup in DJ Hero (p. 238): il gioco consiste nel manipolare mashup di due brani popolari.
- Conclusione (pp. 249–252): festa al parco 2010 con Serato; Rich Medina in club mette "It's Just Begun" (1972) e la pista si riempie; la famiglia DJ–MC–ballerini si ricompone; temi: tecnologia, innovazione, competizione, identità.

## Regole operative
| Regola | Tipo | Pagina |
|---|---|---|
| Nel club hip-hop un brano dura tipicamente **due strofe + ritornello, o anche solo un ritornello** | [A] | 228 |
| Il pubblico di club apprezza mix rapidi (più brani all'ora) | [P] (aneddoto) | 228 |
| Il mix parte da "the 1", primo battito di una frase di 4/8 battute | [P] | 232 |
| Un classico riconoscibile riempie la pista all'istante (Medina, "It's Just Begun") | aneddoto | 250 |

## Limiti
- Osservazioni qualitative; nessuna misura di durata o di gradimento.

## Per MusicMix
- **dj_set modo `sections`**: requisito "full" = almeno una strofa e un ritornello completi; l'obiettivo `target_core_sections = 2` corrisponde a "un ciclo"; Katz p. 228 ammette "two verses and chorus" (3 sezioni core) o "just a chorus" (1): questi sono i limiti sensati del range (implementato come costo quadratico verso 2, requisito minimo rilassabile a "one").
- **Cue**: "the 1" di frase 4/8 = vincolo già presente (downbeat di inizio frase) e `w_phrase4` (preferenza per frasi da 8+).
