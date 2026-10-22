# Giudice automatico v4: canali dei libri, QA su A grezzo, calibrazione cieca

Data: 2026-10-21 (sera). Segue [`AUTO_CRITICO_2026-10-21.md`](AUTO_CRITICO_2026-10-21.md) (v1–v3) e le proposte
[`../libri/PROPOSTE_CRITICO_DA_INTEGRARE.md`](../libri/PROPOSTE_CRITICO_DA_INTEGRARE.md).
Codice: `src/musicmix/dj_critic_books.py` (canali), `src/musicmix/dj_critic.py` (opzione `version="v4"`, additiva),
`src/musicmix/dj_render.py` (deck `a_raw`, additivo), `src/musicmix/dj_qa.py` (`QAConfig.rhythm_on_raw_a`, spenta),
`tools/run_dj_critic_v4.py`, `tools/measure_qa_raw_a.py`, `tools/check_v4_decks_real.py`, `tools/calibrate_dj_critic.py`.
Test: `tests/test_dj_critic_books.py`, `tests/test_calibrate_dj_critic.py`. Run: `runs/dj_critic_v4/`
(`PREREGISTRATION.json`, `scores_v4.json`, `validation_v4.json`, `window_stability.json`, `qa_raw_a.json`,
`decks_real.json`, `REPORT.md`).

Etichette: **[E]** risultato sperimentale; **[T]** teoria; **[A]** analisi/pratica descritta; **[P]** pratica DJ;
**[H]** ipotesi nostra; **[U]** giudizio dell'utente. Le pagine sono quelle stampate, rilette nel testo con
`tools/libri.py` in questa sessione; qui solo citazioni brevi.

## 0. Sintesi

- Il v4 **non** è un giudice del gusto. Separa in due blocchi nominati ciò che il v3 mescolava: `technical`
  (veto e integrità: il veto v3 più avvisi nuovi) e `musical` (diagnostica: sorpresa metrica, sorpresa di contenuto,
  incertezza, finestre, tessitura), che non è mai usato come veto (Q-4).
- L'API v3 (`dj_critic.score_transition(..., mode='fast')`) restituisce esattamente quanto prima; con
  `version="v4"` aggiunge il campo `v4` (test `test_api_v4_is_additive_to_v3`).
- Risultati della validazione, del QA su A grezzo e dei deck reali: §4.

## 1. Canali e fonti

| # | Canale | Blocco | Fonte (verificata) | Implementazione | Ingresso necessario (altrimenti `None`) |
|---|---|---|---|---|---|
| D1 | `metric_surprise` = max(min(1, scarto/80 ms), 1 − peso ipermetrico dello switch) | musical | Huron 2006 p. 116 [T] ("the independence of the what and when of surprise"); pp. 303–304 [T] ("Syncopation only challenges metric perceptions; it never annihilates meter"); Fig. 10.4 pp. 179–180 [A] (forte–debole–medio–debole) | `metric_surprise`, `hypermetric_entry` (usa `dj_phrase.hypermetric_weight`): retta di Beat This! sul lato A-solo estrapolata oltre la giunta, battiti di B sul lato B-solo; posizione dello switch rispetto al cue pianificato | beat tracker; per il peso ipermetrico `switch.a_source_sample`, `stretch.a_bar_s`, cue di A |
| D1 | `content_surprise` (picco z della NLL di MusicGen) | musical | Huron pp. 303–304 (la sincope presuppone il metro); pp. 24–25 (la sorpresa può piacere: mai veto) | `content_surprise`: valore riportato solo se `metric_ok`, altrimenti `gated` con il motivo; il grezzo resta in `peak_z` | NLL di MusicGen |
| D2 | entropia predittiva prima della giunta, −log p dopo, differenza ("sicuri e sbagliati") | musical | Huron p. 55 [T] ("One can be confidently wrong as well as confidently right") | `MusicGenEntropy.frame_nll_entropy`, `entropy_by_frame`, `predictive_uncertainty` (campi separati, z rispetto ai tratti A-solo/B-solo) | MusicGen (venv-np1) |
| D3 | finestre di 1, 3 e 5 s ai due lati: loudness, timbro (MFCC), profilo tonale, eccesso di NLL | musical | Huron pp. 207–208 [E] (Perrott & Gjerdingen: "the bulk of the psychological work is done in the first 250 milliseconds"; ~1 s); p. 228 [E] (memoria a breve termine "between three to five seconds", fino a 10–12) | `window_profile`, `window_stability`; costanti `WINDOWS_S` da `dj_phrase.BOOK_RULES` | — |
| D4 | `checks["dropout"]` del QA | technical (warn) | Huron p. 52 [E citata] (onda N2 per lo stimolo deviante) | `qa_dropout`; un silenzio dichiarato non è un avviso | QA con `dropout_check=True` |
| Q-1 | tessitura "busy" in sovrapposizione | musical | Butler 2006 p. 244 [P riportata] ("if it's too busy, I don't think it sounds good") | `texture_overlap` (`dj_form.bar_layers` per deck) | deck separati |
| Q-2 | backbeat: rullanti di A e B sui battiti 2 e 4 | technical (warn > 15 ms) | Webber 2008 p. 129 [P] ("you start listening for the snare, and matching the snare up"); soglia 15 ms [H] come `dj_qa` | `backbeat_alignment`: attacco più ripido della banda 180–3000 Hz entro ±70 ms in ciascun deck, scarto B − A; contrasto sulla cassa ai battiti 1 e 3 | deck separati o stem drums (preferiti) |
| Q-3 | downbeat dopo l'ingresso della cassa ("turning the beat around") | technical (warn) | Butler 2006 pp. 127, 142 [A] | `downbeat_after_kick`: fase della pulsazione nell'intro senza cassa contro le prime 4 battute con cassa (`dj_form.bar_layers` per la cassa); warn da 0,25 battiti e solo se l'intro si sente da sola ≥ 1 battuta [H] | deck B, oppure la parte B-solo della clip con Beat This! |
| Q-4 | due blocchi separati; `pass` del QA non è qualità | struttura | B&B cap. 50; Katz p. 177 [P]; AGENTS.md | `score_v4` → `technical` / `musical` (`diagnostic_only: true`) | — |
| — | tecniche volute (pause, loop, build, tape stop, backspin) | technical | Margulis 2014 p. 23 [A]; Huron pp. 248–249, 314–316; `dj_qa.intended_gap` | `declared_intent`: la pausa dichiarata è esclusa dal salto di loudness, il flag di fase non si applica a `tape_stop`/`backspin`, gli avvisi `creative.repetition_checks` sono riportati e non ricalcolati | metadati `creative` |

Il veto tecnico v4 = flag v3 di fase (> 40 ms), loudness (> 3 dB) e tonalità (> 1, solo con sovrapposizione),
come raccomandato nel §6 della nota v3; `vad_at_cut` resta un flag **indicativo** (Silero non informativo nella
dose-risposta). Le soglie nuove sono [H] o vengono dal QA; nessuna è stimata sui giudizi.

## 2. QA: ritmo di A misurato prima dell'EQ

`dj_render` restituiva solo il deck A equalizzato, e `dj_qa.check_rhythm` tracciava quindi A dopo l'EQ (con
`bass_gap_beats` un fail di Beat This! a parità di mappa temporale, `runs/books_integration_check`). Ora
`dj_render` aggiunge `decks["a_raw"]` (prima di EQ e fader, dopo l'eventuale ducking della voce) con il marcatore
`a_raw_opt_in`; `QAConfig(rhythm_on_raw_a=False)` (default) lo ignora, quindi PCM e verdetti di default sono
invariati (test `test_render_exposes_a_raw_and_default_qa_is_unchanged`); con `True` il ritmo si misura su A grezzo
e il report lo dichiara (`a_deck_measured`). I deck grezzi passati da `dj_creative`/`dj_effects` (senza marcatore)
si usano come prima in entrambi i casi. Effetto misurato: §4.3. Il default resta all'orchestratore.

## 3. Calibrazione futura (E-crit, E2)

`tools/calibrate_dj_critic.py` legge una cartella di giudizi `musicmix.dj-review.1` (file singoli o "Copia JSON"),
calcola o legge i punteggi v3/v4 delle clip del manifest e:
- usa nell'analisi principale **solo** i giudizi `blind: true` (Margulis 2014 p. 15 [E]: "people enjoy music more
  when they aren't given explicit information about it"); i non ciechi solo come diagnostica separata;
- tiene un voto per clip (l'ultimo salvato) e separa i giudizi con `plays_before_save` oltre `--max-plays`
  (Margulis p. 101 [E]; Huron pp. 133–134 [E], effetto di esposizione), con un'analisi secondaria pesata;
- divide i gruppi in selezione/verifica (sha256 del `group_id`): una scelta (`--choose`) si fa solo sulla selezione
  e si verifica sull'altra metà; di default nessun peso è stimato;
- riporta Spearman (IC bootstrap, p di permutazione), AUC ≥ 7 contro ≤ 4 e accuratezza a coppie (IC bootstrap), e
  dichiara quando i voti sono pochi.

## 4. Risultati

Numeri completi in `runs/dj_critic_v4/REPORT.md` e `validation_v4.json`. Regole e confronti dichiarati in
`PREREGISTRATION.json` **prima** di calcolare i canali sugli item giudicati, con quattro emendamenti registrati
(due item provati durante il debug; `vad_at_cut` tolto dal veto come già scritto nella regola; regola del backbeat
corretta su un riferimento noto, non sui voti; D1 v4.1 **post-hoc** per misurabilità).

### 4.1 Validazione v3 contro v4 sugli stessi giudizi [U]

Giudizi: S1 = 6 voti 1–10 della galleria v1 (non alla cieca; **selezione**: già usati per la v3b e per formulare D1);
V10 = 8 commenti verbali codificati da noi [H] e V12 main = 2 clip "molto insufficienti" (**verifica**); un solo
ascoltatore. IC 95% bootstrap sugli item.

| predittore | S1 Spearman | S1 AUC ≥7 vs ≤4 (5 vs 1) | V10 Spearman [verifica] | AUC S1≥7 vs V12 main (5 vs 2) [verifica] | AUC ≥7 vs tutti i negativi (5 vs 9) [H] |
|---|---|---|---|---|---|
| v3 ensemble | 0,44 [−1,00; 0,95] | 0,40 | 0,03 [−0,78; 0,90] | 0,30 [0,00; 0,80] | 0,56 [0,24; 0,89] |
| v3 numero di flag (neg.) | 0,54 [0,32; 1,00] | 0,60 | 0,35 [−0,33; 0,84] | 1,00 | 0,72 [0,48; 0,91] |
| **v4 veto tecnico** (neg.) | 0,43 [−0,46; 1,00] | 0,90 [0,70; 1,00] | 0,35 [−0,33; 0,84] | 0,95 [0,80; 1,00] | 0,88 [0,69; 1,00] |
| v4 sorpresa metrica (v4.0 pre-registrata) | n = 2 | — | n = 4 | — | 1 vs 5 |
| v4.1 sorpresa metrica (post-hoc) | 0,36 [−0,83; 1,00] | 0,90 | 0,51 [−0,46; 1,00] (n 6) | 0,40 [0,00; 0,80] | 0,67 [0,35; 0,97] (5 vs 7) |
| v4 "sicuri e sbagliati" 1 s (neg.) | −0,62 | 0,60 | 0,33 | 1,00 | 0,80 [0,49; 1,00] |
| v4 sorpresa di contenuto grezza (neg.) | −0,62 | 0,40 | 0,55 | 0,60 | 0,44 |

Accuratezza a coppie: v3 ensemble 0,62 (S1, 13 coppie) e 0,56 (V10, 18); flag v3 = veto v4: 0,65 e 0,67
[0,30; 0,94]. Lettura:
- **Nessuna differenza è conclusiva**: gli IC si sovrappongono quasi del tutto e S1 ha un solo voto ≤ 4.
- Il veto v4 differisce dai flag v3 solo perché `vad_at_cut` non è più un veto (scelta già motivata dalla
  dose-risposta v3, non da questi voti): AUC di sensibilità 0,88 contro 0,72, Spearman S1 0,43 contro 0,54.
- La sorpresa di contenuto è quasi sempre "non valutabile" (il metro non risulta rispettato: §4.2), quindi D1 nella
  forma proposta non è verificabile con questi strumenti; la versione grezza non ordina come l'ascoltatore su S1.
- V8 pool_04 (preferenza A): A meglio di B per sorpresa metrica (0,30 contro 1,00), ensemble v3 (0,41 contro 0,31);
  pari per i veto.

### 4.2 D1: la misura del "quando" sulla sola clip non è affidabile sul nostro rap [E nostro, n piccolo]

- Atteso dichiarato: sorpresa metrica ≤ 0,25 sulle 3 clip votate ≥ 8. **Falso**: v4.0 misurabile solo su 1
  (0,30, scarto 24 ms); v4.1 1,00 / 0,90 / 0,54 (scarti 83, 72, 43 ms).
- Sui 5 blend della galleria v2 rifatti con i deck (`decks_real.json`) l'allineamento dei downbeat dichiarato dal
  render è 0,09–0,29 ms, mentre Beat This! sulla sola clip dà 45–243 ms: il tracker cambia livello metrico o fase
  (mezzo battito) sul lato B. Lo stesso spiega i **veto di fase v3 su blend allineati** (g02, g03, g06 della
  galleria v2; g02 blend 8 della galleria v1, votato 8, ha `beat_phase_ms` 255 ms): falsi positivi del veto v3.
- Conseguenza: per i blend la griglia del render (`render_declared_downbeat_error_ms`, ora riportato in D1) è più
  affidabile della misura sulla clip; la misura solo-clip resta utile per cut ed effetti, con questa riserva. Huron
  (pp. 303–304) resta la ragione per separare *quando* e *cosa*; è lo strumento di misura a non bastare.

### 4.3 QA su A grezzo (`rhythm_on_raw_a=True`)

13 blend rifatti (5 galleria v2 con PCM identico allo sha256 salvato; 2 set v1b e 2 set v2 libri, PCM non identico
perché la configurazione del set è cambiata dopo quelle run; 4 della verifica dei libri, di cui 2 coincidono con il
set v2): **3 verdetti cambierebbero** (2 render distinti), tutti in peggio: set v1b t01 pass → warn (deriva non
significativa di −37 ms); heartless → coehn t01 (set v2 e verifica dei libri) warn → fail (downbeat di B a ~365 ms
anche su A grezzo). Con A grezzo il render con le opzioni EQ dei libri dà **la stessa** misura del default (365 ms,
fail per entrambi): l'opzione rende il ritmo indipendente dall'EQ, ma il fail su heartless → coehn non era solo un
artefatto dell'EQ come ipotizzato. Su galleria v2 g02 l'errore mediano dei downbeat scende da 262 a 8 ms (stesso
warn). Verdetto di default rifatto = salvato in 8/9 casi (g03: QA cambiato da allora). Il default resta spento.

### 4.4 Deck separati su audio reale (Q-1, Q-2, Q-3)

- Q-2 con gli stem drums HT-Demucs (5 blend, stessa mappa temporale del mix): 2 misurabili, entrambi `pass`
  (mediana con segno +5,2 e −2,6 ms), 3 con meno di 3 rullanti accoppiati (`unreliable`). Gli scarti per colpo sono
  sparsi (IQR 32–71 ms) a media ~0: nel rap i colpi vicini al 2 e al 4 non sono sempre lo stesso rullante; per questo
  la regola usa la mediana con segno (emendamento) e il proxy "mix del deck" (che su g02 dava +32 ms) **non** è
  considerato affidabile: senza stem drums il canale è `None` e il proxy resta diagnostica.
- Q-3: nessun intro senza cassa esposto nei blend provati (`not_applicable`/`pass`); verificato solo sui sintetici.
- Q-1: `both_busy_share` = 0 su tutti i 5 blend: la misura "busy" di `dj_form` non discrimina (come già notato).

### 4.5 D2, D3

- D2 (entropia, 66 clip): campi separati calcolati; nessuna soglia. "Sicuri e sbagliati" a 1 s ha AUC 0,80 sulla
  sensibilità ma Spearman −0,62 su S1: nessun segnale coerente.
- D3 (66 clip): Spearman tra finestre 1~3 s 0,60 (loudness) – 0,91 (NLL), 3~5 s 0,88–0,97, 1~5 s 0,64–0,88:
  stabili secondo la regola dichiarata (1~3 ≥ 0,5 e 3~5 ≥ 0,7). La finestra di 1 s è la meno stabile per loudness
  e tonalità (misura rumorosa su 1 s), più stabile per il timbro (0,83), coerente con Huron pp. 207–208.

## 5. Cosa resta rinviato e perché

- **Calibrazione vera**: richiede la prossima galleria **cieca** (`--blind`) con abbastanza voti; i giudizi esistenti
  sono pochi, di un ascoltatore, non ciechi e già visti. `tools/calibrate_dj_critic.py` è pronto.
- **D1 con la griglia del render** come misura principale per i blend (invece della sola clip): da dichiarare e
  provare sulla galleria cieca, non sugli stessi voti.
- **Q-2 senza stem drums**: serve Demucs anche per gli altri brani (oggi solo 5 brani della galleria + 2 del ponte).
- **Q-1**: resta diagnostica finché la misura di densità non è stabile tra rappresentazioni.
- **Sorpresa di contenuto come segnale**: valutabile quasi mai (metro non verificato); nessun uso come peso.
- Default di `rhythm_on_raw_a`: decisione dell'orchestratore (§4.3).

Non si promette che il giudice misuri il gusto: il blocco `technical` è un controllo di integrità con falsi positivi
noti (§4.2), il blocco `musical` è diagnostica.
