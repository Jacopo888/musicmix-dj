# Mappa: Ohriner, Krims, Huron, Margulis → processi della pipeline MusicMix

Data: 21/10/2026. Fonti: copie private (non nel repo), schede per capitolo in
`ohriner_flow_2019/`, `krims_rap_2000/`, `huron_sweet_anticipation_2006/`, `margulis_on_repeat_2014/`.
Pagine = numerazione stampata del libro.

Etichette: **[E]** risultato sperimentale (con N quando noto); **[T]** teoria/argomento dell'autore;
**[A]** analisi di brani o pratica descritta; **[H]** nostra ipotesi ingegneristica.
Colonna «Stato»: *impl.* = in `src/musicmix/dj_phrase.py` (questa iterazione o v1);
*prop.* = patch descritta in `PROPOSTE_OHRINER_KRIMS_HURON_MARGULIS.md`; *diag.* = solo diagnostica;
*no* = verificato e scartato.

Priorità (AGENTS.md, 20/10/2026): prima il mestiere DJ. I quattro libri **confermano** quasi tutte le
regole di mestiere già in uso (cue sul downbeat ipermetrico, uscita a fine frase, beatmatching) e
aggiungono soprattutto: (1) dove cade la fine del verso rap; (2) come preparare un ingresso
(fill/stop/build); (3) limiti della ripetizione nei loop. I moduli «psicologici» restano diagnostica.

---

## 1. Selezione delle coppie (compatibilità A→B)

| Cosa dice il libro | Pagina, tipo | Cosa cambiare | Stato |
|---|---|---|---|
| Lo stile si riconosce in 0,25–1 s, soprattutto dal timbro | Huron pp. 207–208 [E] | distanza timbrica (embedding MuQ/CLAP) come «distanza di schema» tra A e B, diagnostica | prop. (dj_set) |
| Passare a uno schema nuovo richiede ~3 accordi / pochi secondi e lascia residuo | Huron pp. 210–211 [E] *(correzione 21/10/2026: p. 211 riguarda modulazioni a tonalità VICINE; il passaggio fra generi lontani è una congettura dell'autore; il minimo di 3 s e il 90° percentile del timbro sono [H] nostri, opzioni `far_min_blend_s`/`far_echo_tail_stop`)* | tra brani lontani di genere: blend ≥ 3 s o raccordo netto (stop/filtro), mai sovrapposizione ambigua breve | prop. (dj_set/dj_policy); costante `BOOK_RULES["schema_switch_seconds"]` |
| Generi simili interferiscono, generi lontani sono separati («firewall») | Huron p. 213 [T] | segnalare il cambio quando la distanza è grande | prop. |
| Il flow segnala epoca, regione, genere | Krims p. 44 [T] | stile di flow (Krims) come coordinata di contrasto della coppia | impl. diag. (`krims_style`, `flow_compatibility`) |
| Tre stili: sung / percussion-effusive / speech-effusive | Krims pp. 48–51 [T] | punteggi z rispetto a MCFlow; verificati: Beastie Boys 5/5 «sung» sopra mediana; speech-effusive cresce con l'anno | impl. diag. |
| Cesura nella 1ª battuta del gruppo di 4 = speech-effusive | Krims p. 50 [T] | testato su MCFlow: r = 0,03 con lo speech-score, ρ = −0,11 con l'anno → **non** usato nello stile | no (solo descrittore `first_bar_caesura`) |
| Rap: durate sillabiche quasi isocrone | (nPVI di Patel & Daniele, citato da Huron pp. 188–189) [E] | nPVI sillabico sulle parole Whisper: MCFlow 24,5 ± 7,9 vs ~57 del parlato inglese; approssimazione audio r = 0,998 con gli onset annotati | impl. diag. (`syllable_npvi`) |
| Prevedibilità piacevole; brani noti hanno vantaggio veridico | Huron pp. 141, 241–242 [T+E] | bilanciare noti/nuovi; dato non disponibile oggi | prop. (registro) |
| Coppie già ascoltate in sequenza creano attesa del secondo | Margulis p. 116 [E] | caratteristica di coppia «sequenza nota» | prop. (dato mancante) |
| Pump-up (salto di tono in su) comune nel pop; pump-down praticamente assenti, usati come effetto comico (Schickele) | Huron p. 285 [A/osservazione] | a parità di distanza Camelot preferire lo shift che sale | prop. [H da ascoltare] |

## 2. Uscita da A (fine verso, pickup, anacrusi) e ingresso in B

| Cosa dice il libro | Pagina, tipo | Cosa cambiare | Stato |
|---|---|---|---|
| Le frasi rap si allungano iniziando prima; la fine è ancorata al beat 4 | Ohriner p. 110 [E, 75 strofe] | la fine verso è più prevedibile dell'inizio: cercare la fine, poi far partire il pickup di B prima del downbeat | impl. (v1) |
| Ogni brano ha il suo «pilastro» di fine frase | Ohriner p. 110 [E] | termine per brano `pillar_adjust` (log densità brano/corpus); MCFlow held-out AUC 0,918 → 0,925, F1 0,685 → 0,697 | impl. (`PARAMS["pillar_weight"]`, default 0, v2 = 1) |
| Accento = posizione 0mod2 + durata > 0,34 beat + accento lessicale; non il volume | Ohriner p. 57 [A] | port fedele; accordo con le annotazioni 98,9 % | impl. (v1, verificato v2) |
| Rima finale sul beat 4 (BC 3) + cesure regolari = sung | Krims pp. 132–133 [A] | `exit_point` riporta `end_pos16` e `end_on_beat4` | impl. diag. |
| Cesura su BC 3 chiude il verso; cesura su BC 2 lascia BC 3 a un upbeat che spinge nel successivo | Krims p. 194 [A, un brano] | distinguere fine «chiusa» da fine con pickup; test su MCFlow da fare | prop. (`pickup_bc3`) |
| Un verso senza upbeat verso il successivo chiude un'unità formale | Krims p. 133 [A] | uscita preferita: fine verso senza pickup a fine gruppo pari; `next_onset_gap` su MCFlow marginale per la *fine*, resta indicatore di *chiusura* | diag. |
| La fine frase è la parte più prevedibile, subito dopo l'incertezza è massima («reset»): 5,5 bit dentro, 5,0 sulle ultime due note, 6,2 attraverso il confine | Huron pp. 154–157 [E] | giustifica il cambio subito dopo la chiusura; mai uscire *prima* dell'ultima sillaba | impl. (regola v1; costanti `phrase_reset_bits`) |
| Discontinuità tra frasi meno salienti che dentro la frase | Huron p. 124 [osservazione]; p. 157 (2,9 vs 2,0 st) [E] | cambi di tonalità/registro meno udibili sul confine: conferma cue a fine frase; non vale nei blend sovrapposti | impl. (regola v1) |
| Fine di gruppo = durata lunga; inizio = accento | Huron p. 197 [E] | allungamento finale già canale v1; coerente con accento «l» di Ohriner | impl. (v1) |
| Fine di frase attesa discendente (F0) | Huron pp. 77, 88 [E] | pendenza F0 nelle ultime sillabe come secondo indizio | prop. (`f0_final_st` già calcolato; manca la pendenza) |
| Sfasamento costante voce/beat di 6–15/60 di beat è stilistico | Ohriner pp. 190–192 [E] | stimare fase e swing per brano prima di quantizzare | impl. diag. (`vocal_phase_alignment`) |
| Allungamento progressivo delle anacrusi crea attesa del battere | Ohriner p. 176 [A]; Krims pp. 194–195 [A] | un upbeat lungo di A prepara un downbeat forte di B | prop. |
| Ingresso di B sul downbeat ipermetrico (inizio gruppo 4/8/16) | Huron pp. 176–180, 184 [E+T]; Krims p. 98 [A] | già regola DJ; peso ipermetrico 1 / 0,25 / 0,5 / 0,25 | impl. (`hypermetric_weight`) |
| Il primo suono distintivo basta a riconoscere un brano noto; il ritornello è il frammento più ricordato | Huron p. 224 [E]; Margulis p. 82 [E] | per brani noti, entrare con l'hook pulito sul downbeat | prop. (dj_set entrate) |
| Un incipit simile all'hook ne costruisce l'attesa (Barcarolle) | Huron p. 321 [A] | far entrare prima un elemento di B che ricorda l'hook | prop. |

## 3. Tipo di transizione, durata, effetti

| Cosa dice il libro | Pagina, tipo | Cosa cambiare | Stato |
|---|---|---|---|
| Fill (anche un colpo sul 4° beat) o battuta di stop prima del downbeat ipermetrico rende certo il «quando» | Huron pp. 247–250 [A]; Krims p. 98 [A] | rilevare fill naturale nell'ultima battuta della frase di A e preferirlo come uscita; raccordo «stop 1 battuta» | prop. (dj_select/dj_policy) |
| Ritardare un evento molto prevedibile aumenta la tensione; se il ritardo è troppo lungo si perde la tendenza | Huron pp. 307, 314–316, 368 [T] | stop/break prima del drop ≤ 1–2 battute (entro 3–5 s di memoria a breve termine) | prop. (dj_policy) |
| Grandi cambi di arousal vanno preparati in anticipo | Huron p. 319 [osservazione] | verso B più energico: build 8–16 battute | prop. |
| Indicatori di climax: accelerazione eventi, salita, più strati, spettro più brillante | Huron pp. 323–326 [T] | build con roll, riser, filtro che si apre, strati; mai alzare il master (tensione ≠ volume) | impl. in dj_creative (v1, riferimento) |
| Restringere l'unità ripetuta = accelerazione percepita | Margulis pp. 23, 44 [A] | build per dimezzamento 8→4→2→1 battute (→ ½) | impl. (`build_halving_schedule`); prop. uso in dj_creative |
| 1–2 ripetizioni formative, 3–4 focali, > 4 testurali | Margulis pp. 51–52 (Lidov) [T] *(correzione 21/10/2026: categorie teoriche, non un massimo di loop; le soglie di `loop_repetition_check` sono [H] e diagnostiche; 1/4/8 di p. 8 sono secondi)* | limiti di `loop_repetition_check` | impl. (v1) |
| Più di 3 ripetizioni identiche → comico/«disco rotto» | Huron p. 286 [E, un caso]; Margulis pp. 173–174 [E+stima] | avviso per loop ≤ 2 battute ripetuti > 3 volte senza nuovo strato | impl. (`loop_comic_repeats = 3`) |
| Ripetere variando, risparmiare l'intensità per l'ultima iterazione (Sokolov) | Margulis pp. 130–132 [A] | loop d'estensione con traiettoria (filtro/strati) e massimo sull'ingresso di B | prop. (dj_creative) |
| Intro/outro per ripetizione crescente/calante portano dentro/fuori il momento sociale | Margulis p. 151 [A] | sovrapporre su sezioni ripetitive senza voce | impl. (regola DJ v1) |
| Diradare gli strati nel momento vocale culminante | Krims p. 196 [A] | quando entra una frase chiave di B, togliere strati di A (bass swap anticipato) | prop. (dj_policy) |
| Stop totale delle tracce = evento più forte; layering raro = layering marcato | Krims pp. 137–139 [A] | stop come raccordo, una volta per set, su cambio di sezione | prop. |
| Un parametro nuovo alla volta: se cambia il timbro, tieni convenzionale il ritmo | Huron pp. 267–268 [A] | tra generi diversi mantenere beatmatching e downbeat comune | impl. (regola DJ) |
| Mai perdere il metro: contrametrico = esperienza più sgradita | Huron pp. 349–350 [T] | nessun blend con griglie in conflitto | impl. (regola DJ) |
| Fade lungo «rimanda la chiusura all'infinito» | Huron p. 318 [T] | fade lungo solo se A non ha chiusura netta; mai sopra il drop di B | prop. |

## 4. Ordine del set e arco di tensione

| Cosa dice il libro | Pagina, tipo | Cosa cambiare | Stato |
|---|---|---|---|
| L'ordine a larga scala conta poco per l'ascolto non cosciente | Huron p. 236 [E]; Margulis pp. 128–129 [E] | peso dell'arco globale secondario rispetto alla qualità delle singole transizioni | prop. (dj_set, nessun cambio di default) |
| Gradimento a U rovesciata con le ripetizioni (Wundt) | Margulis pp. 95–98 [E] | distanziare ritorni dello stesso tipo di transizione/hook | prop. |
| Un set troppo prevedibile riduce l'attenzione | Huron p. 367 [T] | variare il tipo di transizione, non il metro | prop. |
| Arco di densità di attacchi indipendente dal volume | Krims pp. 110–111 [A]; Ohriner pp. 142–147 [A] | descrittori di energia ritmica vocale (saturazione, complessità groove) per l'arco | impl. diag. (`flow_window`, `ohriner_global_features`) |
| Brani lontani dalle convenzioni richiedono più ripetizione | Huron p. 367 [T]; Margulis p. 23 [T] | introdurli su sezioni ripetitive (loop/intro) | prop. |

## 5. Giudice automatico (sorpresa, aspettativa, ripetizione)

| Cosa dice il libro | Pagina, tipo | Cosa cambiare | Stato |
|---|---|---|---|
| Separare sorpresa sul «cosa» da quella sul «quando» | Huron p. 116 [T] | due canali nel critico: sorpresa metrica (≈ 0 in un buon cambio) e di contenuto | prop. (dj_critic) |
| Certezza (entropia) ≠ accuratezza («sicuri e sbagliati») | Huron p. 55 [T] | registrare entropia e sorpresa realizzata separatamente | prop. |
| Sorpresa piacevole solo dentro uno schema forte e con esito buono | Huron pp. 22, 36, 270, 304 [T] | non premiare la sorpresa in sé; premiare violazioni di contenuto solo se il metro è rispettato | prop. |
| Stile riconosciuto in 0,25–1 s; memoria a breve termine 3–5 s (≤ 12 s) | Huron pp. 207–208, 228 [E] | finestre del critico: 1 s per lo stile, 3–5 s per la continuità attorno al seam | prop.; costanti `stm_seconds`, `style_schema_seconds` |
| Un silenzio dove era atteso un beat è un evento (MMN) | Huron p. 52 [E citata] | drop-out nel seam è un effetto, non assenza neutra | prop. |

## 6. Valutazione (ascolti umani)

| Cosa dice il libro | Pagina, tipo | Cosa cambiare | Stato |
|---|---|---|---|
| Note esplicative prima dell'ascolto riducono il piacere | Margulis (2010) p. 15 [E] | nelle pagine d'ascolto nessuna descrizione della transizione prima del voto | prop. (pagina Artifact) |
| Il piacere da familiarità opera fuori dalla coscienza; giudizi espliciti attenti lo contrastano | Huron pp. 133–134 [E] | limite dichiarato: il voto attento sottostima il pubblico distratto | diag. |
| Con la familiarità il picco di tensione è anticipato e attenuato | Margulis p. 101 [E] | voti ripetuti sulla stessa clip non sono indipendenti | prop. (registrare n. ascolti) |
| Voto dopo stop ≈ probe-tone: mescola qualità e chiusura | Huron p. 51 [T] | registrare anche il punto in cui l'ascoltatore nota il cambio | prop. |
| Intuizioni analitiche vanno verificate con dati | Huron p. 99 [T] | ogni regola di questi libri resta [A]/[T] finché non c'è un test (fatto per pillar, Krims, nPVI, prima cesura) | impl. (metodo) |

---

## Verifiche numeriche di questa iterazione (runs/dj_phrase_v2/)

- `ohriner_port_check.json`: accordo degli accenti con le annotazioni di Ohriner 98,9 % (esclusi i toggle manuali; 94,2 % inclusi).
- `mcflow_books_eval.json`: pillar held-out AUC 0,918 → 0,925, F1 0,685 → 0,697; nPVI MCFlow 24,5 ± 7,9; nPVI onset vs durate r = 0,947, vs griglia uniforme r = 0,998; `first_bar_caesura` media 0,66, nessuna relazione con lo stile (r = 0,03) → scartata; `hypermeter_phase_share` 0,28 ≈ caso (0,25) → la fase ipermetrica non si stima dalle sole fini verso.
- `ohriner_port_check.json`: global features contro l'autore — saturazione r = 1,00, sillabe/s r = 0,999, grooviness r = 0,95, dupleness r = 0,89, adherence r = 0,67.
- `compare.json` (36 brani di `runs/dj_phrase_v1`, 2 saltati per input mancanti; 29 con ASR affidabile): pilastro w = 1 → uscite accettate 129 → 131 su 486 cue (+2, nessuna persa); versi 1485 → 1562; distanza TV delle posizioni di fine verso da MCFlow 0,192 → 0,208 (nessun avvicinamento) → **default resta 0**.
- Limite principale emerso: sulle parole Whisper le misure fini di Ohriner **non reggono** — nPVI mediana 48,5 (MCFlow 24,5), grooviness 0,05 (Ohriner 0,51), fase vocale stimata 0 con il 5 % di disallineamento spiegato; le misure a grana di battito (sillabe/s 4,7, posizione di fine verso, saturazione) restano usabili. Servono onset dallo stem vocale (proposta F5).
