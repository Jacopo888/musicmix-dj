# Biblioteca: libri (copie private, fuori dal repo)

## Audit integrale dei nove libri — 21 ottobre 2026

La revisione più recente è [qui](audit_2026-10-21/REVISIONE_PIPELINE.md), con [copertura](audit_2026-10-21/COVERAGE.json), fonti e priorità. Le schede sotto restano storiche: una regola citata non è automaticamente validata. Corpus Windows in cartelle sorelle `../libri` e `../.book_audit_private_20261021`; i percorsi Linux sotto descrivono la precedente macchina Linux.

I PDF/EPUB e i testi estratti stanno in `../libri/` e **non** vanno committati né copiati qui. In questa cartella ci sono solo schede per capitolo, note di sintesi, mappe verso la pipeline e proposte di patch, con citazioni brevi e numeri di pagina.

Ogni gruppo di libri è stato letto da un agente diverso: ciascuno aggiunge **la propria sezione** qui sotto (non sovrascrivere le altre).

---

## Sezione A — mestiere DJ: Katz, Brewster & Broughton, Webber, Butler, Williams (21/10/2026)

### Indice
| Libro | Schede per capitolo | Nota di sintesi |
|---|---|---|
| Katz, *Groove Music* (OUP 2012) | [`KATZ_GROOVE_MUSIC_2012/`](KATZ_GROOVE_MUSIC_2012/README.md) | [`KATZ_GROOVE_MUSIC_2012.md`](KATZ_GROOVE_MUSIC_2012.md) |
| Brewster & Broughton, *How to DJ Right* (Grove 2003) | [`BREWSTER_BROUGHTON_HOW_TO_DJ_RIGHT_2003/`](BREWSTER_BROUGHTON_HOW_TO_DJ_RIGHT_2003/README.md) | [`BREWSTER_BROUGHTON_HOW_TO_DJ_RIGHT_2003.md`](BREWSTER_BROUGHTON_HOW_TO_DJ_RIGHT_2003.md) |
| Webber, *DJ Skills* (Focal 2008) | [`WEBBER_DJ_SKILLS_2008/`](WEBBER_DJ_SKILLS_2008/README.md) | [`WEBBER_DJ_SKILLS_2008.md`](WEBBER_DJ_SKILLS_2008.md) |
| Butler, *Unlocking the Groove* (Indiana UP 2006) | [`BUTLER_UNLOCKING_THE_GROOVE_2006/`](BUTLER_UNLOCKING_THE_GROOVE_2006/README.md) | [`BUTLER_UNLOCKING_THE_GROOVE_2006.md`](BUTLER_UNLOCKING_THE_GROOVE_2006.md) |
| Williams, *Rhymin' and Stealin'* (Michigan UP 2013) | [`WILLIAMS_RHYMIN_AND_STEALIN_2013/`](WILLIAMS_RHYMIN_AND_STEALIN_2013/README.md) | [`WILLIAMS_RHYMIN_AND_STEALIN_2013.md`](WILLIAMS_RHYMIN_AND_STEALIN_2013.md) |

- Mappa processo per processo: [`MAPPA_PIPELINE_KATZ_BB_WEBBER_BUTLER_WILLIAMS.md`](MAPPA_PIPELINE_KATZ_BB_WEBBER_BUTLER_WILLIAMS.md).
- Proposte di patch per `dj_render`, `dj_effects`, `dj_creative`, `dj_select`, `dj_critic`/`dj_qa`: [`PROPOSTE_KATZ_BB_WEBBER_BUTLER_WILLIAMS.md`](PROPOSTE_KATZ_BB_WEBBER_BUTLER_WILLIAMS.md).

### Implementazione collegata
- `src/musicmix/dj_set.py`: profilo `books` (`profile("books")`, `PROFILE_BOOKS`), default invariati; regole R2b_backspin, R4b_key_cut, R6b_hiphop_cut, R7c_dense; costi di collocazione e formali; arco a onde; durata a sezioni; budget degli effetti.
- `src/musicmix/dj_policy.py`: `PROFILE_BOOKS` (backspin a fine frase, tape stop prima di un brano forte, budget degli effetti, build prima di un drop).
- `src/musicmix/dj_form.py` (nuovo): categorie di Butler come feature per battuta (strati, cassa, intro/build/core/groove/breakdown/outro, drop, rimozioni brevi, sovrapposizione A/B), dagli spettrogrammi per stem di All-In-One, dagli stem Demucs o dalle bande del mix.
- Test: `tests/test_dj_books.py`, `tests/test_dj_form.py`.
- Run: `tools/run_dj_set_v2_books.py` → `runs/dj_set_v2_books/` (confronto con `runs/dj_set_v1b` in `set.json["compare_v1b"]`).

### Verifica delle feature di Butler (21/10/2026, diagnostica, non validazione)
Su 4 brani con stem Demucs a 4 tracce già presenti (`heartless`, `after_hours`, `5_percent_tint`, `mafia`), confronto per battuta tra tre rappresentazioni (stem wav, spettrogrammi allin1, bande del mix − voce):
- **cassa** (attacco basso di drums + bass sui battiti): correlazione bande↔stem 0,98–1,00; spettrogramma↔stem 0,71–0,98. Con la sola cassa dello stem drums la correlazione bande↔stem scendeva a 0,05–0,36 su due brani: nell'hip-hop l'808 sta nel basso.
- **basso** e **voce**: 0,93–1,00 in tutti i casi.
- **etichette formali** spettrogramma↔stem uguali su 100 %, 94 %, 100 %, 44 % delle battute (`heartless` instabile).
- **densità di attacchi** ("busy"): correlazione 0,0–0,5 tra rappresentazioni; soglia alzata a 6 attacchi/battito e uso ridotto.
- Le etichette non sono state confrontate con un ascolto: corrispondenza con le categorie di Butler = [H].

### Set `runs/dj_set_v2_books` vs `runs/dj_set_v1b` (21/10/2026, non ascoltato)
- Pool: quello di v1b (23 brani) meno `reminder` e `too_many_nights`, che non hanno l'audio canonico o `a_vocals.wav` sulla macchina di analisi e che v1b non aveva scelto. Il ponte non aggiunge brani già analizzati.
- 10 brani, 13:24 (v1b 15:46). Ordine: 9 brani in comune; entra `k_os_rhymes`, esce `doctor_frost`.
- Tipi di transizione: 2 blend4, 6 echo out, 1 filter out (v1b: 1 blend4, 1 blend8, 5 echo out, 2 filter out). R7c_dense ha accorciato a 4 battute un blend con tessiture piene.
- QA finale: 6 pass, 3 warn, 0 fail (v1b uguale), ma nessun fail già al primo tentativo (v1b ne aveva 6, poi corretti). Uno dei warn è uno sfasamento della griglia di +310 ms su un echo out (k_os→dk_music).
- Durata per brano: media 80 s (v1b 95 s), effetto della durata a sezioni.
- Limite: `overlap_layers` vale 7–10 su quasi tutte le coppie, quindi la soglia di 7 strati discrimina poco. È un parametro [H] da ricalibrare.

### Righe da aggiungere a `conoscenza_mix/INDEX.md` (non modificato da questo agente)
```
- libri/ — cinque libri sul mestiere DJ (Katz 2012, Brewster & Broughton 2003, Webber 2008, Butler 2006, Williams 2013): schede per capitolo, note di sintesi, MAPPA_PIPELINE_KATZ_BB_WEBBER_BUTLER_WILLIAMS.md (processo per processo, con pagina), PROPOSTE_KATZ_BB_WEBBER_BUTLER_WILLIAMS.md (patch per render/effects/creative/select/critic). Implementato: profilo "books" di dj_set e dj_policy, modulo dj_form (categorie di Butler sull'audio).
```

### Righe da aggiungere a `conoscenza_mix/CHANGELOG.md` (non modificato da questo agente)
```
## 2026-10-21 — libri sul mestiere DJ (Katz, B&B, Webber, Butler, Williams)
- Lettura capitolo per capitolo; schede in conoscenza_mix/libri/<LIBRO>/, note di sintesi, mappa verso la pipeline e proposte.
- dj_set/dj_policy: profilo "books" (default invariati): cut sull'uno nell'hip-hop e con tonalità che cozzano (B&B capp. 11, 13, 15), backspin a fine frase per tempi lontani (B&B cap. 12), blend hip-hop ≤ 8 battute (B&B cap. 9), durata a sezioni intere (Katz p. 228; Butler p. 222; B&B cap. 13), arco a più picchi (Butler p. 250; B&B cap. 24), budget degli effetti (B&B cap. 29; Webber p. 184), costi di collocazione (B&B cap. 10; Webber p. 225).
- dj_form (nuovo): strati/cassa/intro-build-core-breakdown-outro/drop per battuta (Butler capp. 2, 5, 6); dj_set li usa per evitare uscite durante un build, favorire uscite nel breakdown e ingressi sul drop, accorciare i blend con tessiture piene (Butler pp. 221–247). Verifica tra rappresentazioni su 4 brani (vedi libri/README.md); soglie [H].
- runs/dj_set_v2_books: set rifatto col profilo "books" sullo stesso pool di dj_set_v1b; confronto in set.json.
```

---

## Sezione B — flow rap, aspettativa, ripetizione: Ohriner, Krims, Huron, Margulis (21/10/2026)

### Indice
| Libro | Schede per capitolo |
|---|---|
| Ohriner, *Flow: The Rhythmic Voice in Rap Music* (OUP 2019) | [`ohriner_flow_2019/`](ohriner_flow_2019/) — 8 schede (intro+cap. 1 … cap. 8) |
| Krims, *Rap Music and the Poetics of Identity* (CUP 2000) | [`krims_rap_2000/`](krims_rap_2000/) — 5 schede (intro+cap. 1 … cap. 5 e postfazione) |
| Huron, *Sweet Anticipation* (MIT 2006) | [`huron_sweet_anticipation_2006/`](huron_sweet_anticipation_2006/) — 17 schede, una per capitolo |
| Margulis, *On Repeat* (OUP 2014) | [`margulis_on_repeat_2014/`](margulis_on_repeat_2014/) — 8 schede, una per capitolo |

- Mappa processo per processo: [`MAPPA_PIPELINE_OHRINER_KRIMS_HURON_MARGULIS.md`](MAPPA_PIPELINE_OHRINER_KRIMS_HURON_MARGULIS.md).
- Proposte di patch (dj_select, dj_set, dj_creative, dj_policy, dj_critic, pagina d'ascolto): [`PROPOSTE_OHRINER_KRIMS_HURON_MARGULIS.md`](PROPOSTE_OHRINER_KRIMS_HURON_MARGULIS.md).
- Repo dell'autore usati solo per statistiche derivate (nessuna licenza: niente copie): `mohriner/flowBook@faaec23`, MCFlow.

### Implementazione collegata (`src/musicmix/dj_phrase.py`; default invariati)
- `pillar_adjust`, `reanalyze_with_pillar`, `PARAMS["pillar_weight"]` (0; v2 usa 1): pilastro di fine frase per brano (Ohriner p. 110).
- `syllable_npvi`, `hypermeter_phase`, `first_bar_caesura`, `hypermetric_weight`, `vocal_phase_alignment`, `build_halving_schedule`.
- `exit_point`: nuovi campi `end_pos16`, `end_on_beat4`, `krims_closure` (closed_bc3 / closed_early / pickup; Krims pp. 133, 194).
- `loop_repetition_check`: avviso «disco rotto» oltre 3 ripetizioni di loop ≤ 2 battute senza nuovo strato (Huron p. 286; Margulis pp. 173–174).
- `plan_pair_phrase(..., flow_weight=0, with_flow=False)`: continuità di flow nel piano (diagnostica).
- `BOOK_RULES`: costanti con fonte ed etichetta (hypermeter 1/0,25/0,5/0,25; STM 3–5 s; stile 0,25–1 s; cambio di schema 3 s; reset di frase 5,5/5,0/6,2 bit; nPVI parlato 57).
- Test: `tests/test_dj_phrase_books.py` (21 test). Run: `tools/run_dj_phrase_v2.py`, `tools/mcflow_flow_stats.py`, `tools/ohriner_port_check.py` → `runs/dj_phrase_v2/`.

### Verifiche (21/10/2026)
- Port di Ohriner sul suo corpus (275 strofe): accenti 98,9 % (55 935 sillabe); saturazione r = 1,00, sillabe/s r = 0,999, grooviness r = 0,95, dupleness r = 0,89, adherence r = 0,67.
- MCFlow held-out (124 brani): fine verso AUC 0,918 → 0,925 e F1 0,685 → 0,697 con il pilastro; ultimo accento e onset successivo non aiutano.
- Krims su MCFlow: Beastie Boys 5/5 «sung» sopra la mediana; speech-effusive cresce con l'anno (ρ = +0,31). `first_bar_caesura` (Krims p. 50) **non** validata (r = 0,03; ρ = −0,11): solo descrittore.
- nPVI MCFlow 24,5 ± 7,9 (parlato inglese ~57): il rap è molto più isocrono del parlato; approssimazione a sillabe uniformi r = 0,998.
- Brani di `runs/dj_phrase_v1` (36, 29 con ASR affidabile): pilastro → uscite accettate 129 → 131 su 486 (0 perse), nessun avvicinamento alla distribuzione MCFlow (TV 0,19 → 0,21); `flow_weight` = 1 cambia il piano migliore in 156/540 coppie. Sulle parole Whisper nPVI (48,5), grooviness (0,05) e fase vocale (spiegato 5 %) non sono affidabili → servono onset dallo stem vocale (proposta F5).

### Righe da aggiungere a `conoscenza_mix/INDEX.md` (non modificato)
- `libri/MAPPA_PIPELINE_OHRINER_KRIMS_HURON_MARGULIS.md` — cosa dicono Ohriner, Krims, Huron e Margulis per ogni processo (selezione, uscita/ingresso, raccordo, set, giudice, valutazione), con pagina.
- `libri/PROPOSTE_OHRINER_KRIMS_HURON_MARGULIS.md` — patch proposte per dj_select, dj_set, dj_creative, dj_policy, dj_critic e pagina d'ascolto.
- `libri/ohriner_flow_2019/`, `libri/krims_rap_2000/`, `libri/huron_sweet_anticipation_2006/`, `libri/margulis_on_repeat_2014/` — schede per capitolo.

### Righe da aggiungere a `conoscenza_mix/CHANGELOG.md` (non modificato)
## 2026-10-21 — libri su flow, aspettativa e ripetizione (Ohriner, Krims, Huron, Margulis)
- Schede per capitolo dei quattro libri, mappa verso la pipeline e proposte di patch.
- `dj_phrase`: pilastro di fine frase per brano (opzionale, held-out MCFlow AUC 0,918 → 0,925), nPVI, fase ipermetrica, cesura nella prima battuta (non validata), peso ipermetrico, allineamento voce–griglia con swing, build per dimezzamento, `krims_closure` in `exit_point`, avviso di loop «disco rotto», continuità di flow opzionale nel piano. Default invariati.
- `runs/dj_phrase_v2/`: verifiche del port di Ohriner, MCFlow e confronto sui brani di v1.

---

## Sezione C — integrazione delle proposte nel codice (21/10/2026)

- Registro proposta → stato → file/funzione → test → fonte: [`INTEGRAZIONE_PROPOSTE_2026-10-21.md`](INTEGRAZIONE_PROPOSTE_2026-10-21.md).
- Proposte per il giudice automatico, da integrare quando `dj_critic.py` è libero: [`PROPOSTE_CRITICO_DA_INTEGRARE.md`](PROPOSTE_CRITICO_DA_INTEGRARE.md).
- Codice: `dj_creative.halving_build` e `break_before_entry` (profilo libri via `dj_policy`), opzioni EQ di `dj_render`
  (`bass_gap_beats`, `mid_duck_b_bars`), di `dj_effects` (eco manuale, filter con cassa e basso, coda ferma), generi
  lontani in `dj_set` (`far_timbre_pct`, profilo libri), controllo `dropout` in `dj_qa`, onset dallo stem in `dj_phrase`
  (F5), pagina d'ascolto `--blind` e `plays_before_save`. Default invariati. Test: `tests/test_dj_books_integration.py`.
- Run: `runs/dj_phrase_v3_stem_onsets/` (F5), `runs/books_integration_check/` (clip reali con le opzioni nuove, QA tecnico).
