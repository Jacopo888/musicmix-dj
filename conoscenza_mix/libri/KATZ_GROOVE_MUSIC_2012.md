# Nota di sintesi — Katz, *Groove Music: The Art and Culture of the Hip-Hop DJ* (Oxford UP 2012)

Schede per capitolo: [`KATZ_GROOVE_MUSIC_2012/`](KATZ_GROOVE_MUSIC_2012/README.md). Natura: storia culturale e analisi [A] basata su interviste a DJ; le regole operative sono pratica riportata [P], non esperimenti.

## Tesi del libro
Il DJ hip-hop è un esecutore: dal break di Herc al quick mix di Flash, allo scratch, al turntablism e alle battle, il giradischi diventa strumento. La tensione costante è tra la tecnica del giradischi e i bisogni di chi balla (pp. 75, 138, 177).

## Le regole più utili (con pagina)
1. Giunta pulita su **beat, battuta e frase**; se la pista perde il beat anche per pochi secondi l'energia cala (Flash su Herc, p. 52) [P].
2. Il materiale da estendere è il **break** (2–4 battute di sola percussione), agganciato all'**uno** (pp. 15, 24) [A].
3. Il DJ hip-hop suona **le parti migliori**, non i brani interi (pp. 16, 20); nel club oggi un brano dura **due strofe e un ritornello, o anche solo un ritornello** (p. 228) [A].
4. Il mix parte da **"the 1"**, primo battito di una frase di 4/8 battute (p. 232) [P].
5. Sotto un MC il DJ tiene il beat stabile; il pubblico vuole anche voci e canzoni riconoscibili (pp. 73, 75) [P/A].
6. Togliere la base sulla **punchline** (fine della 4ª battuta) è un gesto idiomatico (p. 106) [P aneddoto].
7. Curva del crossfader: dolce per le dissolvenze, ripida (2,5 mm) per i tagli (p. 133) [P].
8. Prima far riconoscere il brano, poi manipolarlo; ripetere lo stesso espediente ne toglie l'effetto (pp. 161, 168, 173) [P].
9. Brano-àncora ricorrente e cambio su materiale con riff/basso simile per una giunta "magica" (Adventures, pp. 80–82) [A].
10. Fine: diradare la tessitura e chiudere con stop/rallentamento (Shadow, p. 196) [A].

## Implicazioni per MusicMix (stato)
- Implementato (profilo `books` di `dj_set`): durata a **sezioni intere** (`dur_mode="sections"`, Katz p. 228), cut sull'uno dopo una sezione nell'hip-hop (`hiphop_cut`, con B&B), costo per frasi da sole 4 battute (`w_phrase4`), budget degli effetti (`w_fx_run`).
- Proposto: beat-mute sulla punchline (`dj_creative`), coda di chiusura con tape stop, somiglianza di riff/basso in `dj_select` (vedi `PROPOSTE_KATZ_BB_WEBBER_BUTLER_WILLIAMS.md`).

## Limiti
Storia dell'hip-hop USA; le interviste descrivono gusti e pratiche, non misurano la qualità percepita delle transizioni.
