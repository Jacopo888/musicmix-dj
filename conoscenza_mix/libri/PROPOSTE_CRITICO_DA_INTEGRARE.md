# Proposte dei libri per il giudice automatico (`dj_critic.py`) — da integrare

Data: 21/10/2026. `dj_critic.py` è in modifica da un altro agente: qui solo le proposte, con fonte verificata
nel testo del libro (pagina stampata), stato degli strumenti già pronti negli altri moduli e test richiesto.
Etichette: [E] sperimentale, [T] teoria, [A] analisi/pratica descritta, [P] pratica DJ, [H] nostra ipotesi.
Regola generale (Huron 2006 p. 99 [T]; AGENTS.md): nessuna di queste diventa un peso del giudice senza un
confronto con gli ascolti tenuti separati; l'integrità tecnica non è qualità musicale.

| # | Proposta | Fonte (verificata) | Strumento già pronto | Cosa fare nel critico | Test prima di usarla |
|---|---|---|---|---|---|
| D1 | Sorpresa sul **quando** separata da quella sul **cosa**: in un buon cambio la sorpresa metrica è ≈ 0; la sorpresa di contenuto si premia solo se il metro è rispettato | Huron pp. 116 (cosa/quando), 270, 303–304 (la sincope presuppone il metro), 349–350 (contrametrico sgradito) [T] | `dj_qa` misura già l'errore di griglia (Beat This!); `dj_phrase.hypermetric_weight` (Huron pp. 179–180) | canale `metric_surprise` = scarto degli onset di B dalla griglia attesa di A + peso ipermetrico del downbeat d'ingresso; `content_surprise` (MusicGen NLL) usato solo se `metric_surprise` è basso | correlazione con i giudizi separati; la sorpresa metrica deve essere ≈ 0 sulle clip votate ≥ 8 |
| D2 | Incertezza (entropia predittiva prima del seam) ≠ sorpresa realizzata (−log p) ≠ accuratezza | Huron p. 55 [T] ("sicuri e sbagliati") | `MusicGenSurprisal` calcola già le probabilità | registrare entropia e −log p in campi separati, non sommarli | solo registrazione (diagnostica) |
| D3 | Finestre di 1 s (riconoscimento dello stile) e 3–5 s (memoria a breve termine) attorno al seam | Huron pp. 207–208 [E] (genere in 250 ms, quasi al massimo a 1 s), p. 228 [E] (STM 3–5 s, fino a 10–12) | costanti `dj_phrase.BOOK_RULES["style_schema_seconds"]`, `["stm_seconds"]` | aggiungere i due blocchi accanto a `SEAM_SIDE_S`/`AESTHETIC_WINDOW_S` | confronto della stabilità dei punteggi tra finestre sulle stesse clip |
| D4 | Un battito mancante nella giunta è un evento, non un'assenza neutra | Huron p. 52 [E citata] (mismatch negativity) | **fatto in `dj_qa`**: `QAConfig(dropout_check=True)` → controllo `dropout` (warn se il render non dichiara la pausa: `gap_beats`, `break_before_entry`, `tape_stop`, `backspin`) | `technical_from_qa` può leggere `checks["dropout"]` quando presente (assente con i default) | — |
| Q-1 | Tessiture piene in sovrapposizione ("busy", "muddled") | Butler p. 244 [P riportata] ("if it's too busy, I don't think it sounds good"), p. 243 *(pagine corrette 21/10/2026: p. 245 e p. 244)* | `dj_form.overlap_profile` (`both_busy_share`, `sum_layers`); `dj_set` già lo usa per accorciare i blend (R7c_dense) | asse "tessitura" solo diagnostico; la misura "busy" è debole (correlazione 0,0–0,5 tra rappresentazioni, `libri/README.md` §A) | prima rendere stabile la misura (onset dagli stem drums) |
| Q-2 | Backbeat (2 e 4) allineato, non solo il downbeat | Webber p. 129 (Radar: si allineano i rullanti), pp. 226, 237 [P] | nessuno (serve lo stem drums, presente solo per 4 brani) | scarto rullante A/B sui battiti 2 e 4 nella sovrapposizione, warn oltre 15 ms | B sfasato di 20 ms solo sul rullante → warn (sintetico) |
| Q-3 | Downbeat percepito dopo l'ingresso della cassa ("turning the beat around") | Butler pp. 127, 142, 233 [A] | `dj_form` (cassa per battuta) | confronto della fase dell'uno nell'intro senza cassa con le prime 4 battute con cassa | intro sintetico sfasato di un ottavo → warn |
| Q-4 | Momento musicale vs pulizia tecnica: assi separati | B&B cap. 50; Katz p. 177 [P]; AGENTS.md | — | non usare mai `pass` del QA come proxy di qualità; due blocchi separati nell'output | — |
| E-crit | Una transizione con descrizione letta prima dell'ascolto viene giudicata diversamente | Margulis 2014 p. 15 [E] ("When Program Notes Don't Help", 2010) | **fatto nella pagina d'ascolto**: `--blind` | se il critico viene tarato sui giudizi, usare solo quelli raccolti alla cieca (campo `blind` nei giudizi) | — |
| E2 | Voti ripetuti sulla stessa clip non sono indipendenti (familiarità) | Margulis p. 101 [E]; Huron pp. 133–134 [E] | **fatto nella pagina d'ascolto**: `plays_before_save` | pesare o separare i giudizi con molti ascolti ripetuti nella calibrazione | — |

## Collegamento con le tecniche nuove

- `dj_creative.halving_build` e `break_before_entry` (21/10/2026) producono silenzi o loop **voluti**: il critico
  deve leggere `metadata["creative"]["technique"]` e `params.gap_beats` prima di penalizzare un calo di energia o
  una ripetizione (stessa logica di `dj_qa.intended_gap`).
- `dj_phrase.loop_repetition_check` (Huron p. 286; Margulis pp. 173–174) è già nei metadati di `halving_build`
  (`creative.repetition_checks`): il critico può riportarne gli avvisi invece di ricalcolarli.

## Nota per il QA (emersa dalla verifica su audio reale, 21/10/2026)

`dj_render` restituisce `decks["a"]` già equalizzato e nessun `a_raw`: `dj_qa.check_rhythm` traccia quindi il deck A
dopo l'EQ. Con l'opzione `bass_gap_beats` (bassi di A tolti 2 battiti prima dello swap) Beat This! ha dato un fail di
ritmo su heartless → coehn con la **stessa** mappa temporale del render di default (warn). Prima di accendere le opzioni
EQ dei libri conviene passare al QA il deck A grezzo (`a_raw`); cambia però i verdetti dei blend esistenti, quindi va
deciso insieme al critico. Dettagli: `runs/books_integration_check/report.json`.

## Stato dopo il giudice v4 (21/10/2026, sera)

Implementazione: `src/musicmix/dj_critic_books.py` (modulo separato), collegato a `dj_critic.score_transition(s)` con
`version="v4"` (campo `v4` additivo; con il default `"v3"` l'output è identico). Nota completa, risultati e limiti:
[`../_meta/AUTO_CRITICO_V4_2026-10-21.md`](../_meta/AUTO_CRITICO_V4_2026-10-21.md); run `runs/dj_critic_v4/`.
Test: `tests/test_dj_critic_books.py`, `tests/test_calibrate_dj_critic.py`.

| # | Stato | File / funzione | Test |
|---|---|---|---|
| D1 | **fatto** (blocco `musical`, solo diagnostica): `metric_surprise` separata da `content_surprise`, riportata solo con il metro rispettato. **Limite misurato**: Beat This! sulla sola clip sbaglia di 45–243 ms su blend allineati a < 1 ms (rap), quindi l'atteso "≈ 0 sulle clip votate ≥ 8" non è verificato; per i blend preferire la griglia del render (rinviato alla galleria cieca) | `metric_surprise`, `hypermetric_entry`, `content_surprise` | `test_metric_surprise_zero_on_grid_and_high_when_b_is_off_grid`, `test_hypermetric_entry_profile`, `test_content_surprise_is_gated_by_the_metre`, `test_tempo_break_is_not_a_metric_surprise` |
| D2 | **fatto** (diagnostica): entropia prima e −log p dopo in campi separati | `MusicGenEntropy`, `entropy_by_frame`, `predictive_uncertainty` | `test_entropy_by_frame_follows_the_delay_pattern_and_uncertainty_fields_are_separate` |
| D3 | **fatto**: finestre 1/3/5 s (`WINDOWS_S` da `BOOK_RULES`) e stabilità tra finestre | `window_profile`, `window_stability`; `runs/dj_critic_v4/window_stability.json` | `test_window_profile_and_stability` |
| D4 | **fatto**: `checks["dropout"]` letto quando presente (avviso nel blocco `technical` solo se la pausa non è dichiarata) | `qa_dropout` | `test_qa_dropout_and_declared_break_are_not_penalised` |
| Q-1 | **fatto, solo diagnostica** (misura debole come in `dj_form`) | `texture_overlap` | (via `tools/check_v4_decks_real.py` su audio reale) |
| Q-2 | **fatto**: warn se la mediana *con segno* dello scarto supera 15 ms; solo con stem drums (il proxy sul mix dei deck non concorda con gli stem su audio reale: diagnostica); altrimenti `None` | `backbeat_alignment` | `test_backbeat_warns_when_only_the_snare_of_b_is_20ms_late`, `test_backbeat_is_none_without_separate_decks` |
| Q-3 | **fatto**: warn se l'intro senza cassa è sfasata ≥ 0,25 battiti e si sente da sola | `downbeat_after_kick` | `test_downbeat_after_kick_warns_for_an_intro_an_eighth_off`, `test_downbeat_after_kick_passes_for_an_aligned_intro_or_when_a_still_plays` |
| Q-4 | **fatto**: blocchi `technical` e `musical` (`diagnostic_only`), `pass` del QA mai usato come qualità | `score_v4` | `test_qa_dropout_and_declared_break_are_not_penalised`, `test_api_v4_is_additive_to_v3` |
| E-crit, E2 | **fatto (strumento)**: calibrazione solo su `blind=true`, un voto per clip, ascolti ripetuti separati/pesati, selezione e verifica divise per gruppo | `tools/calibrate_dj_critic.py` | `test_calibration_uses_only_blind_votes_and_separates_repeated_plays`, `test_calibration_without_blind_votes_has_empty_primary` |
| Tecniche nuove | **fatto**: pausa dichiarata esclusa dal salto di loudness, flag di fase non applicato a `tape_stop`/`backspin`, `repetition_checks` riportati | `declared_intent` | `test_qa_dropout_and_declared_break_are_not_penalised`, `test_repetition_checks_are_reported_not_recomputed` |
| Nota QA (`a_raw`) | **fatto, opzione spenta**: `decks["a_raw"]` in `dj_render`, `QAConfig(rhythm_on_raw_a)`; effetto misurato in `runs/dj_critic_v4/qa_raw_a.json` | `dj_render._render_blend`, `dj_qa.check_rhythm` | `test_render_exposes_a_raw_and_default_qa_is_unchanged` |
