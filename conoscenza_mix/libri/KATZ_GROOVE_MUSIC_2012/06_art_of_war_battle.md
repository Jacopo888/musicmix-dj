# Katz 2012 — Cap. 6 "The Art of War—The DJ Battle: 1991–1996" (pp. 153–178)

## Tesi
- Le battle (head-to-head 1–2 min; showcase DMC **6 min**, p. 154) sono composizioni **preparate**, non improvvisate; l'originalità è la virtù suprema, il "biting" il peggior vizio (pp. 154–155) [A].
- Elementi comuni: scratch, beat juggling, **wordplay** (parole prese dai dischi per vantarsi o insultare), **body tricks** (pp. 155–158).
- I "battle record" (dal 1992) e le compilation di break (Ultimate Beats and Breaks 1986–91) rendono accessibili i break ma staccano i DJ dai brani interi (pp. 158–162).
- La battle è diventata insulare: "too far from the dance element" (DJ Quest, p. 177); Apollo: i DJ dovrebbero saper anche **riempire una pista** (Triple Threat DJs, p. 177).

## Tecniche / procedure [P]
- **Arco di una routine** (Steve Dee, p. 168): "I would play the song, get you familiar with the song, double up a couple of times, to let you see where I'm leading it, and then escalate up and up and up. I get you to that head-nod stage, and then I come down." Dettaglio: giocare con kick o rullante, spezzettare una parola, combinare drumplay e wordplay, cambiare tempo (rallentare, o mettere a 45 giri e poi riportare al "head nod").
- **Usare brani che il pubblico conosce** (Rob Swift, p. 161): così il pubblico capisce come li stai trasformando; Qbert: "It's like jazz—you know the song and then they're flipping it" (p. 161).
- DJ X (racconto composito, pp. 166–174): apre facendo suonare il motivo di 4 note **due volte intatto** perché il pubblico lo riconosca, poi tema e variazioni crescenti (p. 168); in finale ripete la prima nota alzando il volume finché il pubblico riconosce il pezzo e applaude (p. 172).
- **Tones**: melodie suonate col pitch su un tono tenuto; il Technics 1200 consente **±8 %** di pitch (p. 169).
- Effetto di saturazione: dopo un'ora di insulti il pubblico "needs something stronger" (p. 173) → la stessa tecnica ripetuta perde effetto.
- DJ Quest: "only listen to pure scratching for about fifteen minutes at a stretch" (p. 177).

## Regole operative
| Regola | Tipo | Pagina |
|---|---|---|
| Arco "familiarizza → raddoppia → sali → head-nod → scendi" | [P] (Steve Dee) | 168 |
| Prima far riconoscere il brano (lasciarlo suonare), poi manipolarlo | [P] | 161, 168, 172 |
| Le manipolazioni funzionano su brani noti | [P] | 161 |
| Ripetere lo stesso espediente ne riduce l'effetto (varietà) | [A]/aneddoto | 173 |
| Pitch dei giradischi ±8 % (limite fisico storico del beatmatch) | [P] | 169 |
| Troppa tecnica allontana dalla pista: equilibrio tra tecnica e ballabilità | [A] | 177 |

## Limiti
- Le battle non sono set da ballo; le durate (2/6 min) sono per routine dimostrative.

## Per MusicMix
- **Ingresso di B**: lasciare suonare l'ingresso riconoscibile di B (il riff/l'hook) prima di qualunque effetto; non coprire con effetti l'attacco del brano nuovo — coerente con `effect_exit` (effetti su A, B pulito).
- **Varietà**: la penalità di ripetizione (`w_repeat`, `w_fx_run`, `repeat_penalty`) ha qui un appoggio aneddotico.
- Arco di Steve Dee come forma interna di una tecnica lunga (loop_extend/juggle): ingresso pulito → raddoppio → crescita → rientro.
- Il limite ±8 % coincide con `TEMPO_MAX_STRETCH` (MosaikBox) già in uso.
