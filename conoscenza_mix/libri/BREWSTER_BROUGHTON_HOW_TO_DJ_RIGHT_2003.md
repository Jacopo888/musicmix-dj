# Nota di sintesi — Brewster & Broughton, *How to DJ Right* (Grove Press 2003)

Schede per capitolo: [`BREWSTER_BROUGHTON_HOW_TO_DJ_RIGHT_2003/`](BREWSTER_BROUGHTON_HOW_TO_DJ_RIGHT_2003/README.md). Natura: manuale di pratica [P] di due autori/DJ; l'ebook non ha numeri di pagina (si cita "cap. N" e la pagina d'inizio quando il libro la riporta).

## Tesi
"Where you mix is more important than how you mix" (cap. 13): scelta e ordine dei dischi e punti di mix contano più della tecnica.

## Le regole più utili
1. Ogni azione su un confine di **4/8 battute**; mai tagliare a metà frase; "always let a singer finish their verse or chorus" (capp. 10, 13).
2. **Hip-hop: più cut che blend**; blend brevi ("quick blend") (capp. 9, 11).
3. **Cut sull'uno**, tra l'ultimo battito della battuta di A e il primo di B; funziona tra parti percussive, raramente tra melodie piene (cap. 11, p. 68).
4. **Bass swap**: B senza bassi → scambio dei bassi su un confine di frase → A fuori (capp. 9, 28).
5. **Tonalità incompatibili**: "Cut them, don't blend them", o mix su parti solo percussive (cap. 15, p. 88; cap. 31).
6. **Spinback** a fine frase di A prima di un ingresso forte di B, "escape hatch" quando il mix non riesce (cap. 12).
7. Nell'hip-hop uscire **subito dopo un ritornello**, di solito con un cut (cap. 13, "Even tighter").
8. Uscire dal **break** a 2/3–3/4 del brano come da un outro (cap. 13).
9. Arco della serata con **picchi e valli**; accelerare lentamente (oltre +4 % suona male) (capp. 21, 24, 25).
10. Effetti con **moderazione** ("secret weapon, not your constant companion", cap. 29); margine di volume (cap. 27).

## Implicazioni per MusicMix (stato)
- Implementato (profilo `books`): `hiphop_cut`, `key_clash_cut`, `backspin_far_tempo`, blend hip-hop ≤ 8 battute, costi di collocazione (`w_exit_midsection`, `w_exit_after_verse`, `w_entry_midsection`, `w_phrase4`), arco a onde, `w_tempo_down`, `w_fx_run`; in `dj_policy`: backspin a fine frase, tape stop prima di un brano "punchy", budget degli effetti.
- Proposto: "slam in the bass" e headroom in `dj_render`; repeat trick, loop senza sbavature in `dj_creative`; matrice percussione/melodia per i cut in `dj_select`.

## Limiti
Pratica di club house/techno/garage con note sull'hip-hop; nessuna misura.
