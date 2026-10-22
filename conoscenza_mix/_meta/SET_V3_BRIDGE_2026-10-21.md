# Set DJ v3 "ponte" e galleria A/B cieca (21/10/2026)

Codice: `tools/run_dj_set_v3_bridge.py` (fasi `prepare set gallery page`), strato sottile sopra
`tools/run_dj_set_v2_books.py` (ora `build_set(..., hooks=SetHooks(...))`, default invariati). Run: `runs/dj_set_v3_bridge/`
(`set.json`, `set.wav`, `set.mp3`, `tracklist.txt/csv`, `transitions/`, `gallery/`), copie in `deliverables/`.
Test: `tests/test_dj_set_v3_bridge.py`, `tests/test_dj_phrase.py::test_render_phrase_cut_overlap_source_samples_follow_the_pickup`.

Etichette: **[fonte]** regola dei libri o di un paper (libro e pagina nel campo `books_rules`/`rule_source` di `set.json`);
**[H]** soglia o scelta ingegneristica nostra; **[M]** misura nostra. Il QA e il critico verificano integrità tecnica, non gusto:
il set **non è stato ascoltato**.

## 1. Pool

- Catalogo congelato: `catalog_frozen.json` = `runs/pool_bridge_v1/catalog_ready.json` alle 14:50 UTC (21 brani, sha256
  `bcd98f28…`); la prima copia a 18 brani è in `catalog_frozen_18.json` (sha256 `1218c2f3…`; ricongelato su indicazione
  dell'orchestratore prima di qualsiasi render). Analisi DJ rifatte con `dj_plan_pool.analyse_track` su copie congelate dei
  provider (`dj_plan/providers`), modelli di versi `dj_phrase` in `phrase/`.
- Esclusi: `reminder`, `too_many_nights` (stem Demucs assenti sulla macchina di analisi: il canonico è stato ripristinato, gli stem no);
  `antidote` (tempo ambiguo d'ottava: allin1/Beat This! 65,58 BPM, madmom 131,14 BPM; con la griglia a metà tempo il
  downbeat può cadere sul terzo battito: escluso finché non è verificato all'ascolto). Restano 18 brani; Held–Karp ne
  sceglie 14 (`len_min` 12, `len_target` 14 [H]; `max_exact` 15 → preselezione). Fuori dall'ordine: `superhero`, `heartless`,
  `jimmy_cooks`, `6ix9ine_kooda` (91 BPM).
- Limite: l'ASR del ponte è `faster-whisper-small.en` anche per il brano italiano `18k_no_kizzy`; il filtro di qualità
  dell'ASR lo giudica "affidabile", quindi i suoi fine verso sono poco credibili [limite noto].

## 2. Cosa decide il set (in ordine)

1. Ordine e cue: profilo `books` di `dj_set` (B&B, Butler, Katz, Webber; pesi [H]) come v2.
2. Guardia del cut: `dj_phrase.exit_clear_test` (fine verso con p >= 0,5 stimata su MCFlow [M], finestra senza parole,
   stem silenzioso) al posto della guardia stem/ASR della galleria v2 quando l'ASR è affidabile; per la coppia
   `phrase_switch` (pickup di B intero, coda dell'ultima parola di A protetta) e render con
   `dj_phrase.render_phrase_transition` (pickup con lo stem). Nel set finale **nessun cut** è stato scelto (l'albero e i costi
   preferiscono echo/filter/build), quindi questa parte è attiva solo come guardia.
3. Tecnica: se la prima proposta di `dj_policy.propose(..., PROFILE_BOOKS)` sul cue scelto è una regola di contesto
   (C2 ritornello/drop, C2b halving build, C8 break) si rende quella con `dj_creative`, altrimenti vale l'albero `books`.
   `loop_extend`, `acapella_over` e `vocal_swap` (C3/C4, B stirato) **non sono giuntabili nel set**: i metadati di
   `dj_creative` non danno il campione sorgente di B dopo lo stretch (`overlap.b_source_end_sample` assente,
   `b_map.src_start = None`) e `dj_set.clip_geometry` va in errore (seconda esecuzione); nelle esecuzioni precedenti erano
   comunque bocciate dal QA del ritmo (downbeat a 258 ms, 0 downbeat appaiati). Da correggere in `dj_creative` [aperto].
4. Veto (scarta e prova il candidato successivo, registrato in `set.json` → `transitions[].vetoes`, `attempts[].veto`):
   QA `dj_qa.4` fail (incluso doppio basso e parola tagliata; controllo del buco D4 acceso) e flag del critico v3 `fast`:
   salto di loudness > 3 dB (non con una pausa dichiarata dal render), distanza tonale > 1 solo con due brani sovrapposti
   (blend), fase dei battiti > 40 ms **solo se il render non dichiara un residuo < 5 ms**. Il critico non ordina le
   alternative ammissibili. `vad_at_cut` (Silero) è solo diagnostica (non informativo sul rap, `runs/dj_critic_v1/REPORT.md` §10).
5. Montaggio come v1b/v2; set normalizzato a -14 LUFS integrati, true peak <= -1 dBTP (limiter a blocchi).

### Correzione del veto di fase (orchestratore, giudice v4)
Prima esecuzione (`set_run1.json`, `set_run1.log`): 32 tentativi vetati, **26 solo per la fase** (QA non fail), l'ultima
transizione `rich_flex → gods_plan` tenuta nonostante 10/10 veti di fase (59–289 ms). `runs/dj_critic_v4/REPORT.md` mostra
che Beat This! sulla clip mixata sbaglia di 45–243 ms anche su blend con allineamento dichiarato di 0,09–0,29 ms. Nella
versione finale il flag di fase è diagnostica quando il render dichiara la fase: blend → `downbeat_alignment.max_abs_error_ms`
(0,37 ms in T4); ingresso pulito senza sovrapposizione (echo, filter, build, halving) → downbeat di B posto per costruzione
sul downbeat di A (residuo 0 rispetto alle griglie delle analisi) [H: estensione nostra della regola dell'orchestratore,
che parlava di beatmatch; le griglie stesse possono essere sbagliate, vedi QA di T13]. Il blocco `v4` del critico
(`technical`/`musical`) è registrato in `transitions[].critic.v4` come sola diagnostica; il QA con `rhythm_on_raw_a=True`
è registrato in `transitions[].qa_rhythm_on_raw_a` (stessi verdetti del QA di default in tutte le 13 transizioni).

## 3. Risultato (set finale, 14 brani, 17:29, -14,02 LUFS, true peak -1,18 dBTP)

| # | A → B | BPM | Camelot | tecnica | regola (fonte) | QA | flag del critico (diagnostica) |
|---|---|---|---|---|---|---|---|
| 1 | After Hours → MAFIA | 109,0 → 110,0 | 5A → 5A | echo out, coda fermata sul downbeat di B (B1, Huron pp. 207-211) | R6 due voci [E Brungart 2001; S Vande Veire] | warn (ritmo non misurabile) | fase 265 ms (dich. 0) |
| 2 | MAFIA → No Kizzy | 110,0 → 125,0 | 5A → 4A | build verso il ritornello, 2 battute + 1 battito di pausa | C2 (B&B cap. 9, 28; Butler p. 246; Huron pp. 314-316) | warn (possibile ambiguità di mezzo battito, griglie confermate da madmom) | tonale 1,02 (senza sovrapposizione) |
| 3 | No Kizzy → Goosebumps | 125,0 → 130,0 | 4A → 9A | filter out 2 battute | R4 tonalità: niente blend (Vande Veire §4.2.1; B&B cap. 15) | warn (ritmo non misurabile) | fase 380 ms (dich. 0) |
| 4 | Goosebumps → 5% TINT | 130,0 → 140,0 (stretch -7,1 %) | 9A → 10A | blend 4 battute | R8 una voce alla volta (Vande Veire) | warn (ambiguità di mezzo battito) | fase 449 ms (dich. 0,37) |
| 5 | 5% TINT → Creature | 140,0 → 142,0 | 10A → 10A | build verso il ritornello | C2 | pass | — |
| 6 | Creature → BUTTERFLY EFFECT | 142,0 → 70,5 | 10A → 12A | halving build 4 battute (+1 battito di pausa) | C2b (Margulis 2014 pp. 23, 44; Huron pp. 314-316, 323-326) | pass | — |
| 7 | BUTTERFLY EFFECT → Knife Talk | 70,5 → 73,0 | 12A → 3A | build verso il ritornello | C2 | warn (ritmo non misurabile) | fase 96 ms (dich. 0) |
| 8 | Knife Talk → STARGAZING | 73,0 → 75,0 | 3A → 6A | echo out | R4 | warn (buco di 0,86 s non dichiarato) | tonale 1,34 |
| 9 | STARGAZING → A Lot | 75,0 → 146,0 | 6A → 5A | filter out 2 battute | R4 | warn (griglia di B -70 ms; buco 0,75 s) | fase 226 ms (dich. 0) |
| 10 | A Lot → Nonstop | 146,0 → 77,5 | 5A → 11A | echo out | R4 | pass | tonale 1,35 |
| 11 | Nonstop → Sicko Mode | 77,5 → 77,5 | 11A → 2A | echo out | R4 | warn (buco 0,89 s) | tonale 1,10 |
| 12 | Sicko Mode → Rich Flex | 77,5 → 76,6 | 2A → 2B | build verso il ritornello | C2 | pass | — |
| 13 | Rich Flex → God's Plan | 76,6 → 77,2 | 2B → 9B | echo out | R1 griglia a bassa confidenza [H] | warn (**griglia di B -316 ms dal downbeat di A**) | fase 192 ms, tonale 1,45 |

QA finale: 4 pass, 9 warn, 0 fail; 13 tentativi, 0 veti nella versione finale. Il punto di uscita (parola, stato del verso)
è in `transitions[].phrase`: in T2, T6, T7 A è **a metà verso** sul cue (`mid_line`): il build usa lo stutter
dell'ultima battuta e il QA non controlla la parola tagliata quando A esce con un effetto → rischio da ascoltare [aperto].
T13 resta il punto tecnicamente più dubbio (griglia di rich_flex a bassa confidenza, QA warn di 316 ms).

## 4. Galleria A/B cieca

`runs/dj_set_v3_bridge/gallery/review_manifest.json` (schema `musicmix.dj-review.1`), chiave in `gallery/blind_key.json`
(seed 20261021, **da non pubblicare**), pagina generata con `build_listening_artifact.py --blind` (30 clip, 18,5 MB).
Per le prime 12 transizioni: (A) albero di `dj_set` senza profilo con la guardia v2 (come `dj_set_v1b`), (B) la tecnica del
set; se coincidono, la prima alternativa dei libri sullo stesso cue ammessa da QA e veto (fallback dell'albero `books`).
Stesso cue (stessa uscita e ingresso), stesso QA e stesso veto per entrambe. 11 coppie vere, 1 gruppo con una sola clip
(T8: l'unica alternativa, filter out, fallisce il QA). In 5 coppie (T3, T4, T9, T10, T11) B è un'alternativa dell'albero
(echo ↔ filter, blend → filter), non una regola di contesto; in T1 la differenza è solo la coda d'eco fermata sul downbeat di
B (B1, Huron pp. 207-211). Sezione finale: 7 clip di `runs/books_integration_check` (blend default vs EQ dei libri, halving
build, break stop; escluse le due "break" con parola tagliata).
Durate secondo AGENTS.md (tutte `clip_rule_ok`): 10 s prima dell'inizio della transizione + transizione + 10 s dopo.

## 5. Limiti

- Integrità tecnica ≠ qualità: nessuna clip è stata ascoltata; le regole dei libri sono pratica/analisi tradotte in [H].
- Il veto di fase è stato ammorbidito dopo il giudice v4: le griglie delle analisi diventano l'unica garanzia di fase negli
  ingressi senza sovrapposizione; i warn del QA sul ritmo (T2, T4, T9, T13) vanno ascoltati.
- Nessun cut nel set; le tecniche con B stirato di `dj_creative` non sono ancora giuntabili.
- 4 brani esclusi per dati (stem assenti, tempo ambiguo); ASR inglese su un brano italiano.
- Il confronto A/B è su un solo ascoltatore e 11 coppie: anche un risultato netto sarebbe un'indicazione, non una
  validazione delle regole dei libri (AGENTS.md: niente soglie adattate al singolo esempio).
