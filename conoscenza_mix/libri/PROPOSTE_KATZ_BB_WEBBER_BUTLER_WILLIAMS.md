# Proposte di patch dai cinque libri (per i moduli che non modifico)

Da integrare a cura dell'orchestratore. Ogni proposta: **modulo → funzione**, regola, parametro (default neutro, valore del profilo libri), fonte con pagina, etichetta, e un test. Nessuna è validata da ascolti: sono pratica [P] o analisi [A] tradotte in ipotesi [H]. Già implementato da me: `dj_form` (nuovo), profilo `books` in `dj_set` e `dj_policy` (vedi `MAPPA_PIPELINE_KATZ_BB_WEBBER_BUTLER_WILLIAMS.md`).

---

## dj_render

### R-1 "Slam in the bass" (bass swap ritardato e brusco)
- **Funzione**: `render_transition` (blend), parte del bass swap.
- **Regola**: nel blend B entra **senza bassi** (`low` a −∞) per `bass_hold_bars` battute dopo il suo ingresso; poi il basso di B entra di colpo sul downbeat di una frase mentre quello di A esce (swap già a 20 ms). Oggi il low di B entra con il fader.
- **Parametro**: `DJRenderConfig.bass_hold_bars: int = 0` (0 = comportamento attuale); profilo libri `2` per blend ≥ 8 battute.
- **Fonte**: B&B cap. 28 "How to EQ" ("cut the bass for a few bars during a buildup and then return it", "slam in the bass"); cap. 9 (bass swap); Webber p. 42 (Oakenfold: "filter the bass out ... then bring in the bass of another record"); Butler p. 246 (basso tolto 4 battiti a fine 16, poi drop) [P].
- **Test**: con `bass_hold_bars=2` l'energia < 200 Hz della traccia B nell'uscita è ≤ −40 dB per 2 battute dall'ingresso; QA basso invariato (un solo basso).

### R-2 Incontro del tempo "a metà"
- **Funzione**: `choose_tempo_fold` / stretch.
- **Regola**: se `|rate − 1| > tempo_meet_threshold`, accelerare A e rallentare B (o viceversa) di metà ciascuno durante il blend, invece di portare solo B al tempo di A; mai oltre ±4 % per lato.
- **Parametro**: `DJRenderConfig.tempo_meet_threshold: float | None = None`; profilo libri `0.04`.
- **Fonte**: Webber p. 238 ("Does it sound better to meet somewhere in between?"; rallentare "may deflate the energy", accelerare "frantic"); B&B cap. 21 (oltre +4 % suona male) [P].
- **Test**: con 90→96 BPM, A finisce il blend a 93 e B parte a 93; nessuno stretch > 4 %.

### R-3 Taglio di banda della linea in conflitto
- **Funzione**: `render_transition` (blend) — nuova opzione `mid_duck_b_bars`.
- **Regola**: se la tonalità è "adjacent/relative" ma non "same", durante le prime N battute del blend la banda media di B (250–2500 Hz) resta a −12 dB (solo ritmo e basso di B), così le due linee melodiche non suonano piene insieme.
- **Parametro**: `mid_duck_b_bars: int = 0`; profilo libri `4` se relazione ≠ same.
- **Fonte**: Butler p. 53 ("if two synthesizer lines create too much pitch dissonance, one of them could be excised"); B&B cap. 15 ("a natty bit of filtering"); Webber p. 181 (kill di banda) [P].
- **Test**: energia media 250–2500 Hz di B nelle prime 4 battute ≤ −12 dB rispetto alle battute successive.

### R-4 Togliere la cassa di A per agganciarla al breakdown di B
- **Funzione**: `render_transition` (blend), usa `dj_form`.
- **Regola**: se B entra in un breakdown (`dj_form` label `breakdown`) e A ha cassa, togliere la banda < 120 Hz di A per le battute di sovrapposizione (o fino al drop di B).
- **Parametro**: `kick_follow_breakdown: bool = False`.
- **Fonte**: Butler p. 242–243 ("cut out the bass drum in one record to match a breakdown occurring in the other record") [P riportata].
- **Test**: con forme sintetiche, la banda bassa di A è attenuata solo dove B è in breakdown.

## dj_effects

### E-1 Echo out con decadimento a gradini in battiti
- **Funzione**: `echo_out`.
- **Regola**: opzione di eco "manuale": ripetizioni allineate al battito con perdita fissa di `step_db` per ripetizione (invece del feedback continuo), massimo 4 ripetizioni.
- **Parametro**: `DJEffectsConfig.echo_step_db: float | None = None`; profilo libri `-6.0`.
- **Fonte**: Webber pp. 265–266 ("Each time you repeat the sample, you stop the up fader at a slightly lower point than the time before") [P].
- **Test**: picchi delle ripetizioni a intervalli di un battito con decremento −6 dB ± 0,5.

### E-2 Filter out che lascia la sola cassa
- **Funzione**: `filter_out`.
- **Regola**: variante "kick only": HPF fino a 1,2 kHz sulle parti non-cassa ma la banda < 120 Hz resta piena fino al cue (A diventa solo cassa, poi entra B): rende la giunta un cut tra parti percussive.
- **Parametro**: `keep_kick: bool = False`.
- **Fonte**: Webber p. 181 ("cancel out everything but the kick and bass"); B&B cap. 11 (cut tra parti percussive) [P].
- **Test**: energia < 120 Hz di A invariata fino al cue, > 1 kHz attenuata.

## dj_creative

### C-1 Beat-mute sulla punchline
- **Funzione**: nuova `beat_mute(audio_a, audio_b, ..., mute_beats=1)` (o opzione di `loop_extend`).
- **Regola**: togliere lo strumentale di A (mix − voce) sull'ultimo battito (o sugli ultimi 2) della frase di 4 battute che precede il cue, lasciando la voce; B entra pulito sull'uno.
- **Parametro**: `mute_beats ∈ {1, 2}`; richiede stem e `dj_phrase` (fine di frase rap).
- **Fonte**: Katz p. 106 (Jazzy Jeff: "his punchline was on the fourth bar and to drop out"); Butler pp. 91, 188 (rimozioni anacrusiche prima del downbeat) [P/A].
- **Test**: RMS dello strumentale di A nell'ultimo battito ≤ −40 dB; voce invariata; nessun silenzio sul downbeat di B.

### C-2 Repeat trick (teaser)
- **Funzione**: nuova `repeat_tease(..., bars=1, repeats=2)`.
- **Regola**: ripetere l'ultima battuta della strofa di A (loop) 1–2 volte prima del cut su B, solo se la battuta ha voce che finisce entro l'ultimo battito.
- **Fonte**: B&B cap. 32 "How to do deck effects" (repeat trick); Katz p. 51 (Flash, ripetere battute esatte) [P].
- **Test**: due ripetizioni identiche (correlazione > 0,99) allineate alla griglia; cut sull'uno dopo.

### C-3 Drum tip prima del downbeat di B
- **Funzione**: opzione di `build_into_chorus` o nuova `drum_fill`.
- **Regola**: inserire l'attacco della cassa di B (estratto dallo stem drums o da `extract_snare`) sui sedicesimi dell'ultimo battito prima dell'ingresso.
- **Fonte**: Webber p. 260 ("use the tip ... of a drum sample as a fill leading into a downbeat") [P].
- **Test**: 2–4 attacchi nell'ultimo battito, nessun attacco dopo il downbeat oltre a B.

### C-4 Coda di chiusura del set
- **Funzione**: `tape_stop` usato come chiusura (non transizione).
- **Regola**: ultimo brano: a fine sezione, power-off di 2 battiti o dissolvenza di 2 battute (scelta del runner).
- **Fonte**: B&B cap. 12 (power-off per chiudere il set); Katz p. 196 (Shadow: coda con stop); Butler p. 249 (stop secco o abbassare) [P/A].
- **Test**: ultimi campioni a −60 dBFS; nessun clic (fade 10 ms).

### C-5 "Acapella bridge" [H]
- **Funzione**: nuova `acapella_bridge` (variante di `acapella_over`).
- **Regola**: voce di B (stem) su strumentale di A per 1–2 frasi di 4 battute **cicliche** (la frase vocale ripetuta torna al punto di partenza), solo con tonalità compatibile o trasposizione ≤ 1 semitono; eco leggera sulla voce entrante per distinguerla; poi bass swap.
- **Fonte**: Williams p. 122 (acapella su break di un altro disco, mixtape), pp. 134–136 (eco sulla voce citata), p. 155 (frase ciclica), pp. 149–155 (mezzo tono); B&B cap. 31 [A/P].
- **Test**: frase ripetuta ≤ 8 battute; stretch ≤ 5 %; QA clash = 0 (A senza voce).

## dj_select

### S-1 Matrice percussione/melodia per i cut
- **Funzione**: `plan_transition` / `cue_candidates` (ruolo out e in).
- **Regola**: per un cut, premiare uscite di A e ingressi di B dove la parte è percussiva (`dj_form`: pochi strati, cassa sì, medi/alti bassi); penalizzare "melodia piena → melodia piena" e "percussione → melodia piena".
- **Parametro**: `w_cut_texture: float = 0.0`.
- **Fonte**: B&B cap. 11 (p. 68: cut che funzionano e non funzionano) [P]; Butler p. 129 [A].
- **Test**: con due piani uguali, quello con A in `breakdown`/`intro` e B in `intro` ha punteggio maggiore.

### S-2 Somiglianza di basso/riff per giunte "magiche"
- **Funzione**: `pair_compatibility` (nuova componente).
- **Regola**: chroma del basso (stem bass, 40–250 Hz) su 2 battute ai cue: alta somiglianza = bonus d'ordine.
- **Fonte**: Katz pp. 80–82 (Flash cambia su materiale con riff/basso simile; "Good Times" come àncora) [A].
- **Test**: due brani con lo stesso riff sintetico → somiglianza > 0,9.

### S-3 Compatibilità dei pattern ritmici
- **Funzione**: `pair_compatibility`.
- **Regola**: correlazione dei profili di attacco per sedicesimo (cassa/rullante/hi-hat da `dj_form`) sulle battute di sovrapposizione; sincopi opposte → niente blend lungo.
- **Fonte**: B&B cap. 14 (p. 84: "a fight in a drum factory"); Butler pp. 243–244 (mesh) [P].
- **Test**: four-on-the-floor vs breakbeat sintetici → correlazione bassa.

## dj_critic (giudice) e dj_qa

### Q-1 "Busy" e strati in sovrapposizione
- **Funzione**: `dj_critic` (nuovo asse) o `dj_qa` (warn).
- **Regola**: usare `dj_form.overlap_profile` sulle battute della transizione: `both_busy_share ≥ 0,5` o `sum_layers ≥ 8` → avviso "tessitura piena"; `both_vocals_share > 0` → già coperto dal clash vocale.
- **Fonte**: Butler p. 244 ("too busy", "muddled"), p. 243 [P riportata].
- **Test**: due forme sintetiche piene → warn; una piena e una sottile → nessun warn.

### Q-2 Backbeat allineato
- **Funzione**: `dj_qa.evaluate_render` (allineamento).
- **Regola**: oltre al downbeat, misurare lo scarto tra gli attacchi di rullante di A e B sui battiti 2 e 4 nelle battute sovrapposte (banda 1–5 kHz dello stem drums); warn oltre 15 ms.
- **Fonte**: Webber p. 129 (Radar: si allineano i rullanti, più udibili della cassa in cuffia), pp. 226, 237 [P].
- **Test**: B sfasato di 20 ms solo sul rullante → warn.

### Q-3 Downbeat dopo l'ingresso della cassa
- **Funzione**: `dj_qa` (fase della griglia).
- **Regola**: se l'intro di B è senza cassa, confrontare la fase del downbeat stimata nell'intro con quella delle prime 4 battute con cassa (`dj_form`): disaccordo > 1 battito → warn "turning the beat around".
- **Fonte**: Butler pp. 127, 142 (la cassa sposta l'uno percepito), p. 233 (meglio la prima cassa come guida) [A].
- **Test**: griglia sintetica con intro sfasato di un ottavo → warn.

### Q-4 Momento musicale vs pulizia
- **Funzione**: `dj_critic` (combinazione dei punteggi).
- **Regola**: non usare mai l'integrità tecnica (QA pass) come proxy di qualità; tenere separati gli assi tecnici dal giudizio di ascolto.
- **Fonte**: B&B cap. 50; Katz p. 177; AGENTS.md (non equiparare integrità tecnica a qualità) [P].
