# 02 — Design Document: MusicMix intelligente e modulare

**Versione:** 0.1 (proposta di architettura, 18 ottobre 2026)  
**Stato:** design da validare con esperimenti, non affermazione di efficacia.  
**Tecnologia:** Python offline, PyTorch/librosa/FFmpeg per adapter, C++ opzionale per DSP.  
**Fonti:** link al paper direttamente in ogni sezione; l'indice completo è [`../ricerca/PAPER_LINKS.txt`](../ricerca/PAPER_LINKS.txt).

## 1. Problema, obiettivi, non-obiettivi

### 1.1 Problema

Dato un insieme di N brani, individuare *automaticamente* transizioni direzionali A→B tra posizioni arbitrarie, per esempio ritornello A → strofa B, parola A → parola B (con uguale parola o fonetica/intonazione affine), riff A → riff B oppure cambio di armonia/ritmo coerente. I due frammenti non devono necessariamente sovrapporsi: è fondamentale distinguere **successione** e **sovrapposizione**.

### 1.2 Requisiti funzionali

| ID | Requisito |
|---|---|
| FR-01 | Ingestione N file audio leciti, conversione a timeline canonica, metadati e digest. |
| FR-02 | Generazione di cue arbitrari lungo l'intera traccia (non obbligatoriamente intro/outro). |
| FR-03 | Modalità candidati: `structure`, `harmony_rhythm`, `word`, `phoneme`, `melody_riff`, `hybrid`. |
| FR-04 | Ogni candidato A→B mantiene direzione, intervalli, tecnica prevista, confidenze e spiegazione dello score. |
| FR-05 | Ranking distinguendo continuità, compatibilità verticale, orizzontale, originalità e costo di trasformazione. |
| FR-06 | Fase I genera preview *uniformi* e uno studio di ascolto cieco riproducibile. |
| FR-07 | Fase II implementa più strategie di rendering, senza cambiare i cue nei confronti controllati. |
| FR-08 | Mix N-brani tramite pianificatore su grafo con vincoli su copertura, durata, ripetizione e coerenza. |
| FR-09 | Backend degli estrattori intercambiabile; ogni output è cacheabile/versionato. |
| FR-10 | Supporto a musica senza voce e dati mancanti: fallback e confidenza, non inventare lyrics o pitch. |

### 1.3 Non-obiettivi della prima versione

Streaming live, latenza real-time, generazione ex novo di musica, clonazione di voci, perfezione del source separation, trascrizione armonica completa, automazione di diritti/licenze. Non addestrare da zero un foundation model.

**Fonti di impostazione:** [Bittner et al., playlist sequencing (2017)](https://archives.ismir.net/ismir2017/paper/000086.pdf); [Lee et al., vertical/horizontal mashability (2015)](https://www.ismir2015.uma.es/articles/302_Paper.pdf); [AutoMashup (2025)](https://arxiv.org/abs/2508.06516).

## 2. Principio fondamentale: due decisioni misurabili separatamente

- **Decisione 1 — SELEZIONE:** scegliere `(track_a, t_a, track_b, t_b, mode, alignment_plan)` e rankare le proposte. Il sistema può lavorare anche solo su feature/indici senza DSP avanzato.
- **Decisione 2 — RENDERING:** dato esattamente lo stesso candidato, trasformarlo in audio mediante `render_policy` e parametri dichiarati.

Non ottimizzare insieme i due blocchi nel primo esperimento: se si cambia il punto e il crossfade contemporaneamente, il listening test non può attribuire il miglioramento a una causa. Il *renderer neutro* di Step I è intenzionalmente semplice, ma deve essere sufficientemente corretto da non introdurre click, clipping o grossi disallineamenti.

**Paper:** [Automatic Playlist Sequencing and Transitions (2017)](https://archives.ismir.net/ismir2017/paper/000086.pdf); [Automatic DJ Transitions... (2022)](https://arxiv.org/abs/2110.06525).

## 3. Architettura: moduli isolati e DAG

```text
Input audio + manifest/licenze
     |
 [M00: decode & provenance] ---> traccia canonica e timeline
     |
     +----[M01: source separator, OPZIONALE]------ stems + confidence
     |                       |
     +----[M02: rhythm & structure]-------------- beat, downbeat, sections, cue
     +----[M03: vocals & lyrics]----------------- words, phonemes, F0, confidence
     +----[M04: instrumentation & motifs]-------- notes, riffs, onsets
     +----[M05: audio descriptors]--------------- chroma, energy, spectral
                                  |
                         [M06: candidate engines]
                         /    |        |         \
                    beat  word/phone  riff     structure
                         \    |        |         /
                         [M07: feature / constraints]
                                  |
                        [M08: ranker + explanation]
                              /             \
               [M09: neutral preview]   [M11: N-song planner]
                          |                    |
               [M10: listening eval]       mix plan
                          |                    |
                    [GATE A]                  |
                          \                   /
                             [M12: FX renderer]  <-- solo STEP II
                                     |
                               output + QA

[M13: observability/provenance] legge eventi/metriche da tutti i moduli,
                             non decide i candidati.
```

**Regola di dipendenza:** `M12` non viene importato da `M08`. Ogni modulo espone `input_schema`, `output_schema`, `version`, `quality_report` e `errors`; il coordinatore passa solo ID/riferimenti a file grandi, non tensori nascosti in memoria globale. I backend (Demucs, Basic Pitch, ecc.) sono adapter sostituibili.

**Paper guida:** [All-in-One (2023)](https://arxiv.org/abs/2307.16425); [AutoMashup (2025)](https://arxiv.org/abs/2508.06516); [CrossSong Puzzle (2015)](https://staff.aist.go.jp/m.goto/PAPER/SMC2015smith.pdf).

## 4. Moduli: confini, interfacce, diagnostica, paper

### M00 — Ingest e timeline canonica

- **Input:** file audio originali + manifest; **output:** `TrackAsset(id, sha256, path_wav, sample_rate, channels, duration_samples, decoder, decode_offset, permission_tag)`.
- **Responsabilità:** decodifica unica (FFmpeg), WAV PCM 48 kHz stereo per rendering; downmix/resample indipendente per inference dei singoli modelli; allineamento campioni ↔ secondi ↔ beat.
- **Vincolo:** offset delle decodifiche MP3 possono differire; ogni adapter produce timestamp riferiti alla timeline canonica, mai direttamente al suo audio resampled.
- **Test:** click sintetico noto al campione t=1.000s che rientra dallo stesso punto nei risultati di ogni adapter; round-trip timestamps.
- **Guasti diagnosticabili:** offset costante di tutti gli eventi; file corrotto; rate sbagliato; differenze canale.
- **Fonti:** [All-in-One paper (2023)](https://arxiv.org/abs/2307.16425) e [implementazione/documentazione del decoder](https://github.com/mir-aidj/all-in-one).

### M01 — Separazione sorgenti (optional provider)

- **Input:** `TrackAsset`; **output:** `StemRef` per vocals/drums/bass/other e diagnostica qualità/confidence (se stimabile). Il mix originale rimane fonte primaria per preview finale.
- **Provider:** `demucs_ht` prima, `bs_roformer` come confronto; implementare `SeparatorProtocol`. Non dare per scontato stem perfetto o accordato.
- **Test:** durate, offset, somme stems vs mix (errore normalizzato), ascolto su casi rumorosi; procedura di fallback `vocals_unavailable`.
- **Guasti:** leak percussioni nello stem voce, sibilanti degradate, costo GPU; loggare modello, checkpoint, versione, licenza.
- **Fonti:** [Hybrid Transformers for Music Source Separation (2022)](https://arxiv.org/abs/2211.08553); [BS-RoFormer (2023)](https://arxiv.org/abs/2309.02612); [AutoMashup (2025)](https://arxiv.org/abs/2508.06516).

### M02 — Analisi ritmo, struttura e cue

- **Input:** mix/stems; **output:** griglia beat/downbeat con confidenza, sezioni `[start,end,label]`, probabilità di confini, candidate anchor downbeat/phrase/drop.
- **Provider:** All-In-One come extractor principale da valutare; baseline librosa per onset/tempo, spectral recurrence, cue detection.
- **Test:** allineamento downbeat annotato, boundary F1 entro tolleranza fissata, coverage lungo l'intero brano; plot beat/section in audit.
- **Guasti:** half/double tempo, BPM variabile, sezioni ambiguamente etichettate; nei casi a bassa confidenza consentire cue locali a onset e parole.
- **Fonti:** [All-In-One (2023)](https://arxiv.org/abs/2307.16425); [Analyzing Song Structure with Spectral Clustering (2014)](https://brianmcfee.net/papers/ismir2014_spectral.pdf); [Automatic Detection of Cue Points (2020/23)](https://arxiv.org/abs/2007.08411).

### M03 — Voce, parole, fonemi e intonazione

- **Input:** vocals stem o mix, lyrics se disponibili; **output:** `WordSpan` (testo, forma normalizzata/lemma, language, start/end, phonemes, alignment_confidence), `PitchContour` (F0 Hz, voicing_prob, frame times), `VocalEvents`.
- **Provider:** aligner specifico per canto quando il testo è disponibile; ASR solo per proposte/assistenza, **non** come verità a livello di sillaba. CREPE/altro pitch extractor sui segmenti validi. Dizionario fonetico dipendente da lingua.
- **Test:** 2+ annotatori su sottoinsieme di parole; errore onset/offset e Word Error Rate del cantato; precisione della stima F0 con note sintetiche; visibilità di silenzi/non voiced.
- **Guasti:** voce corale, melismi, vibrato, autotune, sillaba trascinata, lingue diverse, lyric alignment falso. Se `alignment_confidence` insufficiente il generatore word non emette match automatici.
- **Fonti:** [Phoneme Level Lyrics Alignment and Text-Informed Singing Voice Separation (2021)](https://schufo.github.io/files/2021_TASLP3091817.pdf); [Real-Time Lyrics Alignment Using Chroma and Phonetic Features (2024)](https://arxiv.org/abs/2401.09200); [CREPE (2018)](https://arxiv.org/abs/1802.06182); [WhisperX (complementare, parlato)](https://arxiv.org/abs/2303.00747).

### M04 — Note, riff, timbro e motivi

- **Input:** stem strumentali quando presenti, altrimenti mix/HPSS; **output:** pitch events, intervalli relativi, onset, pattern ritmici e fingerprint corti per riff.
- **Algoritmo:** trascrizione note su stem dove possibile; estrarre anche chroma/CQT + curve di salienza, evitando di forzare trascrizione polifonica dove non affidabile. Match invarianti entro limiti a shift di tonalità, tempo e ottave.
- **Test:** riff annotati e negativi perturbati (sfasamento accenti, note sostituite); ablation senza trascrizione note.
- **Guasti:** strumenti polifonici, basso armonico, envelope percussivo, errori di onset, falsi match di riff generici.
- **Fonti:** [Basic Pitch (2022)](https://arxiv.org/abs/2203.09893); [Neural Loop Combiner (2020)](https://jbls.fun/documents/chen2020-ismir-neural_loop_combiner.pdf).

### M05 — Descrittori trasversali e FeatureStore

- **Output per finestra:** chroma/HPCP o Tonal Interval Vectors, spectral balance, RMS/LUFS, onset density, beat strength, timbral embeddings **diagnostici**, vocal activity. Feature riferite alla timeline canonica.
- **Caching:** chiave=(sha256 audio, input window, extractor_id/version/config, model checkpoint hash). Cache separata per audio, waveform e feature.
- **Qualità:** `feature_available`, `confidence`, `coverage`, `warnings`, non imputare zero per una feature non calcolata.
- **Fonti:** [AutoMashUpper (2014)](https://repositorio.inesctec.pt/bitstreams/97188d38-9bf1-46f9-99a3-08d3b656ca4f/download); [Leveraging Compatibility and Diversity (2023)](https://link.springer.com/article/10.1007/s00779-022-01702-z); [Spotify (2017)](https://archives.ismir.net/ismir2017/paper/000086.pdf).

### M06 — Generazione candidati a più scale

- **Input:** eventi, descrittori, indici per brano; **output:** `TransitionCandidate` con posizioni precise, ruolo di eventuali stem, motivazione generativa e parametri di allineamento PROPOSTI (non ancora eseguiti).
- **Generatori separati:** `DownbeatCandidateGenerator`, `WordPhonemeCandidateGenerator`, `MotifCandidateGenerator`, `StructuralCandidateGenerator`, `ComplementarityCandidateGenerator`.
- **Granularità:** finestre macro da 2/4/8/16 battute e micro-ancore a parola/fonema/note; valori configurabili, solo ipotesi iniziali.
- **Costi:** prefiltrare con feature economiche; ANN/index retrieval per shortlist; rerank costoso solo top-K; directional A→B distinto da B→A.
- **Filtri:** confidenza, range BPM/pitch consentiti, contesto musicale minimo, politica riuso, vincoli di posizione **nessun** vincolo fisso ultimo/inizio.
- **Test:** recall dei cue annotati; contatori pre/post filtro e motivo di ogni esclusione; shortlisting non deve rimuovere sistematicamente parole rare.
- **Fonti:** [Spotify (2017)](https://archives.ismir.net/ismir2017/paper/000086.pdf); [CrossSong (2015)](https://staff.aist.go.jp/m.goto/PAPER/SMC2015smith.pdf); [Cue Points (2020/23)](https://arxiv.org/abs/2007.08411).

### M07 — Calcolo features di coppia, vincoli e allineamenti

- **Input:** candidato grezzo + feature dei due brani; **output:** `PairFeatures` normalizzate, `constraints`, `missing`, `proposed_transform` e `confidence`.
- **Vocale:** match esatto/lemma/fonema; per stesse parole confrontare F0 in semitoni relativi alla mediana e derivate di pitch (non solo Hz assoluti); DTW con penalità di warp; ritmo sillabico; presenza vocale simultanea.
- **Strumentale:** corrispondenza sequenza intervalli relativi, distanza di onset; evitare match basati soltanto sulla nota tonica.
- **Musicale:** chroma/TIV, tempo/phase/downbeat, densità/spectral balance, energia, contesto dei segmenti prima/dopo il punto.
- **Vincoli hard:** input invalido, cue fuori durata, delta tempo > limite, sovrapposizione incompatibile non correggibile. I limiti di pitch/tempo devono essere configurabili e registrati.
- **Fonti:** [Lee (2015)](https://www.ismir2015.uma.es/articles/302_Paper.pdf); [AutoMashUpper (2014)](https://repositorio.inesctec.pt/bitstreams/97188d38-9bf1-46f9-99a3-08d3b656ca4f/download); [Phoneme Lyrics (2021)](https://schufo.github.io/files/2021_TASLP3091817.pdf); [CREPE (2018)](https://arxiv.org/abs/1802.06182).

### M08 — Ranker interpretabile con decomposizione

**Principio:** evitare un singolo embedding cosine. Scoring multilivello:

\[
S(c)=\alpha\, C_{horizontal} + \beta\, C_{vertical} + \gamma\, C_{continuity} + \delta\, I_{creative} - \lambda\, P_{transform} - \mu\, P_{artifact}.
\]

`horizontal` = qualità del **prima→dopo**; `vertical` = compatibilità della **sovrapposizione** quando prevista; `continuity` = riconoscibilità di una parola/riff/ritmo; `creative` = diversità/complementarità e valore percepito (initial proxy, validare); `transform` = costo di time/pitch; `artifact` = rischio da stem/ASR/DSP. Le feature indisponibili vengono escluse dalla normalizzazione esplicitando coverage; **non** equivalgono a score 0. Pesi in config per modalità; versionati.

**Output:** punteggio globale, score per dimensione, penalties, hard fail, confidenza, `why`, `why_not`, competitor appena sotto top-K. Impostare una shortlist diversificata: un massimo di transizioni quasi identiche per la stessa frase/parola/cue.

**Modello appreso (opzionale, DOPO benchmark):** confrontare logistic regression / gradient boosting pairwise / COCOLA come feature; hard negative con errori realistici e verifica di possibili false etichette. Split per canzone/artista quando possibile, non per singola clip.

**Test:** no-lyrics, no-F0, no-harmony, no-structure, no-stems, no-novelty; prove di ranking direzionale; fixture dove un match testuale perfetto è armonicamente disastroso.

**Fonti:** [Lee (2015)](https://www.ismir2015.uma.es/articles/302_Paper.pdf); [COCOLA (2025)](https://arxiv.org/abs/2404.16969); [Modeling Stem Compatibility (2021)](https://arxiv.org/abs/2103.14208); [Neural Loop Combiner (2020)](https://jbls.fun/documents/chen2020-ismir-neural_loop_combiner.pdf); [Diversity (2023)](https://link.springer.com/article/10.1007/s00779-022-01702-z); [AutoMashup (2025)](https://arxiv.org/abs/2508.06516).

### M09 — Renderer neutro per validare la selezione (Step I)

- **Input:** `TransitionCandidate` selezionato + master audio; **output:** preview A→B con contesto sufficiente per giudicare il passaggio.
- **Policy fissata a priori:** finestra d'ascolto identica per tutti, fade equal-power breve o cut su onset/downbeat, normalization costante; alignment leggero e uniforme solo se necessario, mai policy diversa favorita per un algoritmo.
- **Proprietà:** evitare click/overs/clipping; calcolare qualità audio di base, ma **non** usare modelli generativi o trick di correzione per nascondere cue sbagliati.
- **Test:** a stesso cue il renderer produce output deterministico; benchmark audio di click e livelli.
- **Fonti:** [Spotify (2017)](https://archives.ismir.net/ismir2017/paper/000086.pdf); [DJGAN (2022)](https://arxiv.org/abs/2110.06525) come precedente sperimentale di ascolti, non come obbligo GAN.

### M10 — Esperimenti e giudizi umani (Step I)

- **Input:** transizioni anonime, baseline e random; **output:** preferenze pairwise, dimensioni distinte (musicalità, continuità, creatività), confidenza e report.
- **Protocollo:** confronti ciechi, ordine randomizzato e bilanciato, stessa anteprima e loudness, coppie di brani stratificate; pilot test prima della raccolta vera. Analizzare clustering per coppia di brani/ascoltatore, non assumere indipendenza totale.
- **Metodologia:** gate A passa solo se il metodo supera in modo credibile almeno una baseline musicale forte, non semplicemente random.
- **Fonti:** [Automatic DJ Transitions with Differentiable Audio Effects and GANs (2022)](https://arxiv.org/abs/2110.06525); [Spotify (2017)](https://archives.ismir.net/ismir2017/paper/000086.pdf); [Lee (2015)](https://www.ismir2015.uma.es/articles/302_Paper.pdf).

### M11 — Pianificatore N canzoni

- **Input:** grafo di `TransitionCandidate` top-K, segment graph per brano, vincoli utente; **output:** `MixPlan` ordinato con ingresso/uscita e durata di ogni tratto.
- **Stato:** `(song_id, entry_time, exit_time, used_song_bitset, cumulative_duration, structural_context)`; arco direzionale con costo; beam search con diversificazione e constraints; eventuale A* / CP-SAT per varianti.
- **Importante:** il piano deve includere durata riprodotta del tratto dentro ogni canzone, non solo il punto di switch. Vietare stato impossibile in cui l'uscita precede l'ingresso o brani diventano clip microscopiche indesiderate.
- **Test:** nessuna canzone richiesta esclusa, nessuna ripetizione salvo policy, durata corretta, limite di densità transizioni, path riproducibile. Prima di N brani, verificare A→B su 2 canzoni.
- **Fonti:** [CrossSong (2015)](https://staff.aist.go.jp/m.goto/PAPER/SMC2015smith.pdf); [Spotify (2017)](https://archives.ismir.net/ismir2017/paper/000086.pdf); [Leveraging Compatibility and Diversity (2023)](https://link.springer.com/article/10.1007/s00779-022-01702-z).

### M12 — Renderer musicale avanzato (Step II)

- **Input:** `MixPlan`, audio master e stems, parametri, `RenderPolicy`; **output:** master stereo, loudness/true peak report, clip di audit per transizione.
- **Strategie:** (1) cut su accento, (2) beat-aligned crossfade, (3) word-to-word splice con micro alignment, (4) stem takeover con passaggio voce/accompagnamento, (5) riff overlay controllato. Non tutte le strategie sono adatte a tutti i candidati.
- **DSP:** warp tempo che preservi transienti, pitch shift entro limite, equal-power/per-band fade, EQ, automazione volume, limiter trasparente; confrontare Rubber Band/C++ solo quando il profilo indica un collo di bottiglia.
- **Prove:** *cue fissi* tra render baseline e avanzato, ascolti ciechi separati; log di trasformazioni, qualità stem, clipping e artefatti.
- **Fonti:** [DJGAN (2022)](https://arxiv.org/abs/2110.06525); [Hybrid Transformers (2022)](https://arxiv.org/abs/2211.08553); [BS-RoFormer (2023)](https://arxiv.org/abs/2309.02612); [Lee (2015)](https://www.ismir2015.uma.es/articles/302_Paper.pdf).

### M13 — Diagnostica, versioni e cause/effetti

- **Input:** eventi strutturati di ogni fase; **output:** `diagnostics.jsonl`, dashboard/offline report, artefatti indipendenti per fase.
- **Lineage:** `source_hash -> separator_version -> extractor_version -> generated_candidates -> ranked_candidates -> render_policy -> listener_trial`. Memorizzare eventi con `correlation_id` di candidato e traccia.
- **Error taxonomy:** decoding offset, stem leakage, beat/downbeat phase, wrong structure, lyric alignment, pitch voicing, missed retrieval, bad normalization, weighting bias, render artifact, invalid experimental comparison.
- **Causal debugging:** analizzare la *prima* fase difettosa, rieseguire controfattuali fissando i suoi output e modificandone solo uno a valle; golden fixtures e snapshot di feature. Sono esperimenti causali controllati sul software, non prova di causalità psicologica.
- **Fonti:** [Spotify (2017)](https://archives.ismir.net/ismir2017/paper/000086.pdf) (analisi errori beat); [All-In-One (2023)](https://arxiv.org/abs/2307.16425) (task correlati); [AutoMashup (2025)](https://arxiv.org/abs/2508.06516).

## 5. Algoritmo specifico: stesso vocabolo, intonazione simile

**Ipotesi di ricerca originale da testare; non presentarla come metodologia già validata su mix inter-song.**

1. Per A e B ottenere parole e timestamps affidabili. `match` esatto → lemma → vicinanza fonetica, con regole lingue-specifiche.
2. Estrarre segmenti vocali con pre/post-contesto e F0(t) dove `voicing_prob` è alto.
3. Convertire la F0 in semitoni e normalizzare rispetto a una baseline locale robusta (la stessa parola può essere cantata in registri differenti). Salvare anche versione assoluta per compatibilità armonica.
4. Allineare due curve mediante DTW **con vincolo sul warping**; confrontare profilo di intervalli, attacchi sillabici, accenti e durata della parola.
5. Controllare l'armonia nello spazio circostante: stessa parola non basta se una fine frase produce un'attesa risolta diversamente o stona.
6. Generare più ipotesi di switch: su attacco fonema, alla fine della parola, tra sillabe (solo se supportato), micro-sovrapposizione, proseguimento del vocal di A sul backing B.
7. Restituire `phonetic_match`, `pitch_contour_similarity`, `syllabic_timing`, `context_harmony`, `prosody_confidence` e `transform_cost`, senza collassarli prematuramente.

**Caso negativo intenzionale:** due brani dicono «amore» con F0 affine, ma il beat di B entra sul levare errato e l'accordo è dissonante: la compatibilità finale deve essere bassa. **Caso positivo non banale:** parole diverse ma stessa sequenza fonetica/melodica e allineamento ritmico.

**Fonti:** [Phoneme Lyrics 2021](https://schufo.github.io/files/2021_TASLP3091817.pdf); [CREPE 2018](https://arxiv.org/abs/1802.06182); [Lee 2015](https://www.ismir2015.uma.es/articles/302_Paper.pdf).

## 6. Scoring e ottimizzazione globale

### 6.1 Pipeline di ricerca

`index → fast candidate retrieval → cheap hard constraints → pairwise features → interpretable score → diversity-aware rerank → preview`.

Le stime globali di BPM/key possono essere usate per il pruning iniziale ma il giudizio finale deve essere **locale**, attorno al punto e al contesto di esecuzione. Conservare top-K da ogni generatore per non perdere casi vocali rari quando molti candidati ritmici dominano.

### 6.2 N canzoni

Definire `within-track edge` per riproduzione continuativa di un segmento e `cross-track edge` per una transizione. Non ridurre il problema a un grafo di *soli brani* con un arco per coppia: perderemmo posizione/contesto e vincoli temporali. Il planner può pre-selezionare top-E transizioni per coppia di brani e usare beam search con stato di copertura.

### 6.3 Diversità

Rerank top-K con penalità per stesso cue/stessa parola/stessa modalità/stessa zona della canzone. Un'ottima classifica deve offrire alternative *davvero diverse* e annotabili, non dieci copie dello stesso cut con spostamenti di 100 ms.

**Fonti:** [CrossSong (2015)](https://staff.aist.go.jp/m.goto/PAPER/SMC2015smith.pdf); [Spotify (2017)](https://archives.ismir.net/ismir2017/paper/000086.pdf); [Diversity (2023)](https://link.springer.com/article/10.1007/s00779-022-01702-z).

## 7. Contratti, storage e versioning

**Canonical time:** `t_samples` intero sul PCM 48 kHz per export, `t_seconds` come derived display; beat indices e timestamp dei modelli sono solo viste. Griglie di analisi 16/22.05/44.1 kHz sono interne agli adapter. Eventi con `start_sample <= end_sample`, sample indices non negativi.

**Persistenza proposta:** file audio su filesystem, metadata SQLite, JSON/Parquet per feature/events, cache content-addressed, `experiments/*.jsonl` per listening test; nessuna dipendenza da servizi esterni obbligatoria.

**Schema candidato:** `candidate_id`, `track_a`, `track_b`, `cue_out_sample`, `cue_in_sample`, `mode`, `context`, `feature_values`, `availability`, `confidence`, `proposed_transform`, `score_breakdown`, `reason_codes`, `stage_versions`. Dettagli in [`03_CONTRATTI_DATI.md`](03_CONTRATTI_DATI.md) e [`../schemas/transition_record.schema.json`](../schemas/transition_record.schema.json).

**Caching e collisioni:** contenuti hash SHA-256 + parametri rilevanti; aggiornare i modelli invalida solo le feature loro dipendenti. Non cambiare il formato del candidato senza aumentare `schema_version`.

## 8. Validazione e falsificabilità

**H0 primario:** la preferenza ascoltatori del ranker proposto **non** supera la migliore baseline musicale a policy di rendering fissa. L'ipotesi alternativa è una preferenza positiva. Non usare solo correlazioni con COCOLA o «precisione ASR» per dichiarare successo.

**Baselines:** `R0` cue casuali **validi**; `R1` Spotify-like (beat/chroma/timbre, range libero lungo traccia); `R2` AutoMashUpper-like (harmony/rhythm/spectral); `R3` ranker vocal/melody+structural.

**Esperimenti:** selezione punti con renderer neutro; rendering con cue fissi; ablation per feature; casi ad alta confidenza, basse confidence e differenti generi. Report primario: preferenza pairwise e intervallo bootstrap stratificato per coppia di brani; un punteggio singolo non è sufficiente. Protocollo completo: [`04_PROTOCOLLO_VALUTAZIONE.md`](04_PROTOCOLLO_VALUTAZIONE.md).

**Fonti:** [DJGAN (2022)](https://arxiv.org/abs/2110.06525); [Spotify (2017)](https://archives.ismir.net/ismir2017/paper/000086.pdf); [COCOLA (2025)](https://arxiv.org/abs/2404.16969).

## 9. Threats to validity / rischi

- **Dataset leakage:** split per canzone e idealmente per artista/raccolta, non per frammento.
- **Studio umano piccolo:** un pilot orienta lo sviluppo ma non produce una verità generale. Registrare partecipanti, intervalli, generi, hearing context.
- **Bias di render:** nei confronti Step I non usare renderer diverso per candidati diversi.
- **Bias di distorsione:** separazione e pitch-shift potrebbero migliorare compatibilità ma peggiorare timbro; misurare distintamente.
- **Riconoscimento parole:** ASR parlato può fallire sul cantato; lyric verification necessaria.
- **Metriche proxy:** COCOLA misura un task di coerenza stem, non «creatività dello switch»; score appresi rischiano shortcut.
- **Versioni/licenze:** checkpoint e codice esterni possono essere bloccati o dipendere da GPU/CUDA obsoleti; prevedere adapter/fallback.
- **Diritti:** l'autorizzazione ad analizzare file non implica autorizzazione a distribuire remix, stems o lyrics.

## 10. Decisioni lasciate esplicitamente aperte

| Decisione aperta | Esperimento discriminante |
|---|---|
| Demucs vs BS-RoFormer | Costo GPU / leak vocali / effetto sulla qualità di word matching; non solo SDR. |
| Audio transcription note vs chroma/CQT | Ablation `notes_off` su riff annotati + robustezza su audio polifonico. |
| DTW su F0 assoluta o relativa | Confronto manuale su stessa parola in tonalità e registri diversi. |
| Ranker lineare vs rete contrastiva | Confronto su split song-held-out, calibratura e interpretabilità. |
| Cut vs short crossfade per anteprima | Test di controllo su cue identici; fissare una policy dopo pilot. |
| Piano globale ottimizzato vs greedy | Valutazione N brani con vincoli reali, tracciamento combinatoria. |

**Tracciabilità completa:** [`06_TRACCIABILITA_RICERCA.md`](06_TRACCIABILITA_RICERCA.md) e [`../ricerca/MATRICE_PAPER_MODULI.csv`](../ricerca/MATRICE_PAPER_MODULI.csv).
