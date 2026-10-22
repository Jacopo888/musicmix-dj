# Integrazione nel codice delle proposte dei libri (21/10/2026)

Fonti delle proposte: [`PROPOSTE_KATZ_BB_WEBBER_BUTLER_WILLIAMS.md`](PROPOSTE_KATZ_BB_WEBBER_BUTLER_WILLIAMS.md),
[`PROPOSTE_OHRINER_KRIMS_HURON_MARGULIS.md`](PROPOSTE_OHRINER_KRIMS_HURON_MARGULIS.md), mappe `MAPPA_PIPELINE_*.md`.
Ogni pagina citata qui è stata riletta nel testo del libro (`tools/libri.py`, numerazione stampata).
Le proposte per il giudice automatico sono in [`PROPOSTE_CRITICO_DA_INTEGRARE.md`](PROPOSTE_CRITICO_DA_INTEGRARE.md)
(`dj_critic.py` è in modifica da un altro agente).

**Stati.** *default* = attiva senza opzioni (solo dove non cambia il suono né le scelte); *profilo libri* = spenta nei
default, accesa da `dj_policy.PROFILE_BOOKS` / `dj_set.PROFILE_BOOKS` (come le regole libri già presenti);
*opzione* = spenta anche nel profilo, da accendere a mano (manca un ascolto o la misura è debole); *rinviata* /
*scartata* con il motivo. Nessuna regola è validata da ascolti: sono pratica [P], analisi [A], teoria [T] o
risultati sperimentali [E] di altri domini tradotti in ipotesi [H] (AGENTS.md: un articolo citato non valida
una nostra soglia).

## 1. Proposte integrate

| Proposta | Stato | File / funzione | Test | Fonte (verificata) |
|---|---|---|---|---|
| **F5** onset sillabici dallo stem vocale (p-centre) per nPVI, grooviness, fase vocale | opzione (misurata, §3) | `dj_phrase.stem_syllable_onsets`, `with_stem_onsets` (campi `syl_on`/`syl_src`, usati da `model_syllables`), `vocal_phase_alignment(..., onsets_s=)`, `STEM_ONSET_PARAMS`; run `tools/run_dj_phrase_v3_stem_onsets.py` → `runs/dj_phrase_v3_stem_onsets/compare.json` | `test_stem_onsets_are_half_amplitude_pcentres`, `test_with_stem_onsets_snaps_uniform_syllables_and_finds_the_lag` | Ohriner 2019 p. 187 [A] ("the moment its vowel reaches half of its maximum amplitude"), p. 186 (a cappella); pp. 190-196 (fase) |
| **C1** build per dimezzamento del loop (2→1→½→¼ battute fino al downbeat di B), passa-alto che sale, pausa facoltativa | profilo libri (`halving_build`, regola `C2b_halving`) | `dj_creative.halving_stages`, `halving_build`; `dj_policy` (`PARAMS["halving_build"]`) | `test_halving_stages_bar_aligned`, `test_halving_build_repeats_shrinking_units`, `test_halving_build_gap_is_silent`, `test_policy_halving_build_and_break_only_with_books_profile` | Margulis 2014 p. 23 [A] (unità da 2 battute, poi 1, ½, due note: "the cumulative effect is of acceleration"), p. 44; pp. 130-132 (traiettoria); Huron 2006 pp. 323-326 [T], p. 319 |
| **A1/C3** fill o stop di 1 battito prima dell'ingresso; beat-mute sulla punchline | profilo libri (`entry_break`, regola `C8_break`) | `dj_creative.break_before_entry(mode="stop"\|"beat_mute"\|"fill", beats=1\|2\|4)`; `dj_policy` sceglie beat_mute se A canta sull'ultimo battito (stem), stop con tempo lontano, altrimenti fill; nessun break se A canta anche nella battuta dopo il cue | `test_break_stop_silences_last_beat`, `test_break_beat_mute_keeps_the_voice`, `test_break_fill_adds_snare_in_last_beats`, test policy | Huron 2006 pp. 248-249 [A] (il fill "dramatically increases listener certainty about the when"), pp. 314-316 [T] (ritardo → tensione), p. 368 (ritardo troppo lungo: "the effect will be lost"); Krims 2000 pp. 137-139 [A]; Katz 2012 p. 106 [P]; Butler 2006 p. 91; Webber 2008 p. 260 (drum tip come fill) |
| **C3** pausa prima del drop nei build (`gap_beats`) | profilo libri (`build_gap_beats = 1`, mai > 2) | `dj_policy` passa `gap_beats` a `build_into_chorus` e `halving_build` | test policy | Huron pp. 314-316, p. 368 |
| **R-1/R-4** "slam in the bass": bassi di A fuori N battiti prima dello swap, basso di B di colpo sullo switch (un solo basso alla volta) | opzione (`DJRenderConfig.bass_gap_beats`, 0) | `dj_render._render_blend`, metadati `eq.books.bass_gap`, deck `gains.low_a` | `test_bass_gap_slams_b_bass_on_the_switch` | B&B 2003 cap. 28 [P] ("cut the bass for a few bars during a buildup ... and then return it, bamm"); Butler pp. 242-243, 246 [P riportata] |
| **R-3** medi di B attenuati nelle prime battute del blend (due linee che cozzano) | opzione (`mid_duck_b_bars`, `mid_duck_b_db = -12`) | `dj_render._render_blend`, `gains.mid_b` | `test_mid_duck_b_then_back_to_0db` | Butler p. 53 [P] ("one of them could be excised from the mix"); Webber p. 181; B&B cap. 15 |
| **E-1** eco manuale: ripetizioni uguali a un battito, -6 dB ciascuna, max 4, niente filtro | opzione (`DJEffectsConfig.echo_manual_repeats`, `echo_step_db`) | `dj_effects.echo_out` | `test_manual_echo_steps_down_6db_per_beat` | Webber 2008 pp. 265-266 [P] ("you stop the up fader at a slightly lower point than the time before") |
| **E-2** filter out che lascia cassa e basso | opzione (`filter_keep_low_hz`) | `dj_effects.filter_out` (`effect.keep_low`) | `test_filter_out_keep_kick_keeps_low_band` | Webber p. 181 [P] ("cancel out everything but the kick and bass"); B&B cap. 11 |
| **B1** generi lontani: niente sovrapposizione breve e ambigua; coda d'eco fermata sul downbeat di B | profilo libri (`far_timbre_pct = 90`) | `dj_set.far_timbre_threshold`, `apply_far_genre` (render `tail_stop`), `dj_effects` `tail_under_b_db=None`; collegato in `tools/run_dj_set_v2_books.py` | `test_far_genre_threshold_and_clean_switch`, `test_echo_tail_can_stop_on_b` | Huron 2006 pp. 207-208 [E] (schema attivato nei primi 250 ms), pp. 210-211 [E] (nuovo schema in ~3 accordi, "a few seconds", con residuo), p. 213 [T] *Correzione 21/10/2026: le soglie (90° percentile, 3 s) sono [H]; Huron p. 211 riguarda tonalità vicine. Astensione con matrice degenere; opzioni `far_min_blend_s`, `far_echo_tail_stop` (`_meta/P0_MISURE_REGOLE_2026-10-21.md`)* |
| **S-1** cut tra tessiture piene su entrambi i lati | opzione (`ORDER_PARAMS["w_cut_full_both"]`, 0 anche nel profilo) | `dj_set.form_context` (`a_full_before`, `b_full_at`), `candidate_cost` | `test_cut_full_both_sides_costs_only_when_enabled` | B&B cap. 11 "When to cut" [P] ("the more continuous melody there is on both sides of the join, the less likely it is that a cut will sound good"); Butler p. 129. Proxy [H]: strati di `dj_form`, non melodia |
| **D4** buco non voluto nella giunta | opzione (`QAConfig.dropout_check`) | `dj_qa.check_dropout`, `intended_gap` | `test_qa_dropout_warns_only_when_not_declared` | Huron 2006 p. 52 [E citata] (mismatch negativity per l'evento atteso mancante) |
| **E1** pagina d'ascolto senza descrizione tecnica prima del voto | opzione (`--blind`, pagine DJ a versioni) | `tools/build_listening_artifact.py`: badge "versione N", tecnica/warn QA/motivazioni rivelati dopo il giudizio salvato, preferenza con etichette neutre | `test_dj_review_blind_hides_technique_until_vote` | Margulis 2014 p. 15 [E] ("When Program Notes Don't Help", 2010) |
| QA delle tecniche senza sovrapposizione nuove | default (solo per `halving_build`/`break_before_entry`) | `dj_creative.qa_inputs`: deck B escluso, ritmo sul mix allo switch | verificato su `runs/books_integration_check` (T4) | — |
| **E2** ascolti ripetuti prima del voto | default (solo un campo in più nei giudizi) | stessa pagina: `plays_before_save`, `blind` nei documenti `judgments` | stesso test | Margulis p. 101 [E]; Huron pp. 133-134 [E] |
| **B&B cap. 29** budget degli effetti | già presente (profilo libri), esteso | `dj_policy.PARAMS["fx_marked"]` include `halving_build` e `break_before_entry` (penalità solo se `fx_budget_penalty` > 0) | `test_dj_books.py` (esistente) | B&B cap. 29 [P] ("Effects should be your secret weapon, not your constant companion") |
| **B&B cap. 12** backspin a fine frase per tempi lontani | già presente (profilo libri: `dj_set` R2b, `dj_policy` C5) | invariato | `test_dj_books.py` | B&B cap. 12 [P] |

Versioni: `dj_render.3`, `dj_effects.3`, `dj_creative.2`, `dj_qa.4` (con i default il PCM e le scelte non cambiano).

## 2. Proposte non integrate

| Proposta | Stato | Motivo |
|---|---|---|
| R-2 incontro del tempo a metà (Webber p. 238) | rinviata | richiede lo stretch anche di A dentro la transizione (oggi A non è mai stirato e la guardia delle parole lavora in campioni di A); Webber la pone come domanda da verificare all'ascolto ("Does it sound better to meet somewhere in between?") |
| A1 rilevare un fill **naturale** nell'ultima battuta di A (bonus di selezione) | rinviata | le analisi non hanno gli attacchi per battito dentro la battuta; gli stem drums esistono solo per 4 brani. Il fill **costruito** è in `break_before_entry` |
| A2 fine verso "chiusa" vs pickup (Krims p. 194) | rinviata | la proposta stessa chiede prima il test su MCFlow; analisi di un solo brano; `next_onset` +0,001 AUC |
| A3/A4 ingresso con l'hook per brani noti, Barcarolle | scartata / rinviata | manca un campo "noto al pubblico" nel catalogo; A4 è [H] |
| B2 pump-up preferito (Huron p. 285) | rinviata | osservazione sul pop, nessuna misura nel mix; da provare all'ascolto |
| B3 peso dell'arco, B4 varietà | solo fonte | nessun cambio di parametro (le fonti sono nelle mappe) |
| B5 brani lontani su sezioni ripetitive, C4 diradare A sulla frase chiave, C5 accenti voce-cassa, C6 fade lungo | rinviate | C5 richiede onset vocali affidabili (F5 non basta, §3); C6 non applicabile (l'albero non produce fade lunghi) |
| C2 loop con traiettoria | parziale | la traiettoria a filtro è in `halving_build` (`trajectory="hpf_rise"`); in `loop_extend` no |
| C-2 repeat tease, C-3 drum tip di B, C-4 coda di chiusura, C-5 acapella bridge | rinviate | C-3 coperto in parte dal `fill`; C-4: il montaggio del set ha già una dissolvenza finale; C-5 è [H] |
| S-2 somiglianza di basso, S-3 pattern ritmici | rinviate | servono stem bass/drums e chroma per battuta, non presenti nelle analisi |
| Q-1, Q-2, Q-3, D1-D3, Q-4 | al critico | `PROPOSTE_CRITICO_DA_INTEGRARE.md` |
| F0 finale, respiro, cambio di groove come confine | rinviate | servono F0 per sillaba e un corpus con audio |

## 3. Effetto misurato

### F5 (onset dallo stem) — `runs/dj_phrase_v3_stem_onsets/compare.json`
Stessi 36 modelli di `runs/dj_phrase_v1` (pilastro w = 1 come v2), 29 con ASR affidabile (10 in inglese, 19 in altre
lingue); 2 brani senza stem sulla macchina di analisi. Mediane, Whisper → stem (sillabe agganciate: 74 %, spostamento mediano +18 ms,
cioè la vocale arriva dopo l'inizio della parola Whisper):

| Misura | Whisper | Stem (F5) | Riferimento |
|---|---|---|---|
| fase vocale (battiti, + = voce in ritardo) | 0,000 | **+0,033** (IQR stem grezzo 0,017–0,067; ~22 ms) | Ohriner pp. 190-196: voce in ritardo, ~0,11 battiti |
| disallineamento dalla griglia (1/60 di battito) | 3,59 | 3,60 (onset grezzi dello stem 3,54) | 3,75 = onset senza aggancio alla griglia |
| quota spiegata da fase + swing | 0,046 | 0,095 | — |
| nPVI in tempo continuo | 48,5 | 58,6 | MCFlow 24,5 ± 7,9 (onset quantizzati) |
| nPVI sulle posizioni quantizzate (confrontabile con MCFlow) | 37,5 | 38,7 | MCFlow 24,5 |
| grooviness / adherence (port di Ohriner), brani **inglesi** | 0,069 / 0,57 | **0,118 / 0,72** | Ohriner Tab. 5.2: 0,51 / 0,67 |
| grooviness / adherence, altre lingue | 0,000 / 0,40 | 0,000 / 0,45 | — |

Lettura: gli onset dello stem si addensano attorno alle sedicesime più di quelli Whisper (istogramma della posizione nella
sedicesima su `heartless`, `mafia`, `after_hours`) e danno per la prima volta un segno della fase coerente con Ohriner
(voce in ritardo), ma **non** rendono affidabili nPVI e grooviness: il nPVI quantizzato non si muove, quello continuo
sale (il jitter reale degli onset entra nel rapporto) e la grooviness resta lontana dal corpus di Ohriner. Il limite non è
più solo la temporizzazione ma il conteggio e la segmentazione delle sillabe (CMUdict/euristica sulle parole Whisper;
nelle lingue diverse dall'inglese peggio) e la quantizzazione. Decisione: F5 resta **opzione** (`with_stem_onsets`, default
invariato); la fase vocale dallo stem si può usare come diagnostica; C5 (accenti voce-cassa) resta rinviata.

### Opzioni di render e tecniche nuove su audio reale — `runs/books_integration_check/report.json`
Tre transizioni di `runs/dj_set_v2_books` (stessi brani, cue e griglie del set), `tools/check_books_integration_real.py`,
QA `dj_qa` con Beat This! e controllo `dropout` acceso. Clip in `runs/books_integration_check/clips/` (non ascoltate).

| Clip | QA | Note |
|---|---|---|
| T1 heartless → coehn, blend 4 default | warn | griglia di B −43 ms allo switch (come nel set) |
| T1 blend 4 + `bass_gap_beats=2` + `mid_duck_b_bars=2` | **fail** (ritmo) | stessa mappa temporale del default (cambiano solo i guadagni EQ); il fail viene da Beat This! sul deck A **dopo** l'EQ (`dj_render` non passa al QA un `a_raw`): artefatto di misura, da sistemare nel QA prima di accendere l'opzione |
| T1 `halving_build` 4 battute + 1 battito di pausa | warn | griglia di B +52 ms; pausa riconosciuta come voluta (`dropout` pass, 0,65 s) |
| T1 `break_before_entry` beat_mute | **fail** (verse_cut) | A continua a cantare oltre l'uno ("my life"): il QA ha ragione. Corretto `dj_policy`: niente break se A canta nella battuta dopo il cue (Katz p. 106: la punchline deve finire prima dell'uno) |
| T4 scelta → porrazzo, blend 4 default / con EQ dei libri | pass / pass | salto di loudness identico (+0,04 dB) |
| T4 `break_before_entry` fill | **fail** (verse_cut) | parola "you" a cavallo dell'uscita (stessa causa: qualunque uscita in quel punto la taglia). Il primo tentativo falliva anche sul ritmo: B è muto prima dell'uno e il confronto deck-contro-deck non ha coppie → `dj_creative.qa_inputs` ora lascia fuori il deck B per le due tecniche nuove (ritmo misurato sul mix allo switch) |
| T5 porrazzo → k_os, `halving_build` | pass | pausa voluta 0,62 s |
| T5 `break_before_entry` stop | pass | stop di 0,61 s dichiarato |

Nessun doppio basso, nessun salto di loudness oltre 1 dB. Le opzioni EQ di `dj_render` restano spente (artefatto del QA
su T1, nessun ascolto); le due tecniche nuove restano nel solo profilo libri, con la guardia sulla voce di A.

## 4. Come accenderle

```python
from musicmix import dj_policy as pol, dj_set as ds, dj_render as dj, dj_effects as fx, dj_qa
pol.propose(ctx, params=pol.PROFILE_BOOKS)          # halving_build, break_before_entry, gap_beats = 1
ds.profile("books")                                  # far_timbre_pct = 90 (B1)
dj.DJRenderConfig(cue_convention="start", bass_gap_beats=2, mid_duck_b_bars=2)
fx.DJEffectsConfig(echo_manual_repeats=4, filter_keep_low_hz=120.0, tail_under_b_db=None)
dj_qa.QAConfig(dropout_check=True)
# python tools/build_listening_artifact.py <review_manifest.json> --out OUT --blind
```
