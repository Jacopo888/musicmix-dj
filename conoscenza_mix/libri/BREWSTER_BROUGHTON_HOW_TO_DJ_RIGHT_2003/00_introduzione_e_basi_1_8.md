# B&B 2003 — Introduzione e capp. 1–8 ("The bedroom")

## Introduzione "How to DJ right" [P]
- Il DJ "track[s] down greatness in music and squeeze[s] it together"; esempio: "pick the funkiest two bars of a musician's entire life and loop them as a little intro" (intro).
- **Gerarchia del mestiere**: la tecnica conta poco rispetto a *quali dischi* e *in che ordine*: "Learning to mix is only your apprenticeship. It's maybe **5 percent of the craft**, at most"; "If your music is boring or badly chosen, the best beatmatching in the world won't save you"; alcuni grandi DJ erano "pretty ropey mixers" ma il disco successivo era perfetto per il momento.
- Il vero lavoro: conoscere la musica e "learn about people … what makes them laugh and smile and dance".

## Cap. 1–4 (uso del libro, comprare musica e attrezzatura, montaggio) [P]
- Musica: "Is it great music? Will it work on a dancefloor? Does it fit my style?"; "killer, not filler" (cap. 2).
- Attrezzatura (cap. 3, p. 28): giradischi direct-drive con coppia e pitch stabile; Technics SL1200; Vestax con pitch fino a ±50%; mixer per hip-hop con crossfader da scratch, "hamster switch" e curva regolabile; CDJ Pioneer/Denon.
- Montaggio (cap. 4): peso della testina 2–5 g; anti-skating a zero per lo scratch.

## Cap. 5 "How to get started" [P]
- Ascoltare separando gli strumenti; usare l'EQ per isolarli.
- **Cassa**: "In most house records, it hits on every beat … in most funk and hip-hop records it hits 'on the one' (i.e., 1, -, -, -)". Rullante sui controtempi, hi-hat.
- Voce: "a special kind of melody" con forza emotiva e narrativa; "They come in verses and choruses".

## Cap. 6 "How to cue" (p. 44) [P]
- Ascoltare due dischi insieme (uno in cuffia), split-cue; slip-cue: tenere il disco appena prima del primo battito, piccolo colpo per partenza a piena velocità.
- Juan Atkins: prima del mix "the guy who could leave the least amount of space in between each record was top guy".

## Cap. 7 "How to fade" (p. 48) [P]
- **Curve del crossfader**: *beatmix curve* (passaggio graduale, "no drop in sound level"), *scratch curve* (on/off, "cut-in point"), *dipped curve* (abbassa al centro per compensare la somma di due brani: "old-fashioned").
- Primo set senza beatmatch: "Do quick fades where the beats get tangled up and slow ones where you can (try fading over breakdowns or quiet bits). Some records work best with a gap and a clean beginning."

## Cap. 8 "How to beatmatch" (p. 50) [P]
- Avviare il secondo disco **in battuta** col primo; capire quale corre di più ("which record finishes four bars (or eight bars) first"; "b-boom" del kick doppio).
- Regolazione per bisezione del pitch (+1 → +3 → +2 …).
- Distinguere **deriva di tempo** (serve pitch) da **sfasamento costante** (basta un tocco).
- Dischi con batteria dal vivo oscillano di tempo.
- Due copie dello stesso disco quasi in fase → *phasing* (cap. 32).
- BPM: contare i battiti in 15 s × 4.

## Per MusicMix
- Il primato della **selezione e dell'ordine** sul mixer (5 % del mestiere) dà priorità a `order_set`/`choose_cues` rispetto al raffinamento del render [P].
- *Dipped curve* e somma di livelli: nei blend la somma dei due deck va compensata (già fatto con equal-power/loudness in dj_render).
- "Quick fades where the beats get tangled": se il beatmatch non è possibile, la dissolvenza deve essere breve (coerente con echo/filter out brevi e cut).
- Distinguere deriva e sfasamento: è ciò che dj_qa misura (drift/phase); per il giudice un errore di fase costante è meno grave di una deriva.
