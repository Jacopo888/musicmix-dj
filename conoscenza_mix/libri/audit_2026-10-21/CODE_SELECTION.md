# Audit code: selection, cue, fraseggio e set

> Nota storica recuperata sul pin a8cf580c. Diverse lacune sono superate da 6f03b33 (dj_form, runner books, nuovi onset). Per lo stato corrente e le priorità corrette usare REVISIONE_PIPELINE.md e ROOT_RESUME.md; non trattare i test qui sotto come eseguiti sul nuovo pin.

Audit read-only del checkout `a8cf580c` (21-10-2026). Le righe sono riferite a
questo snapshot. `[E]/[S]/[P]/[T]/[U]/[H]` conservano il significato dichiarato
nel codice; un'origine bibliografica non equivale a una validazione del codice.

## Mappa effettiva della pipeline

| Stadio | Implementato | Entrypoint/default | Validazione effettiva |
|---|---|---|---|
| Analisi pool | `tools/dj_plan_pool.py:123-195` carica grid/provider, sezioni, Demucs occupancy, energia e chiavi; `dj_select` è puro e non fa inferenza (`src/musicmix/dj_select.py:1-9`). | `tools/dj_plan_pool.py:198-221` scrive tutte le coppie con `djs.plan_pair(...)`; nessuna guardia `dj_phrase`, nessun albero `dj_set`, nessun `dj_policy`. | Le regole sono dichiarate a priori e non fitted (`tools/dj_plan_pool.py:239-251`). Non è stata eseguita una nuova analisi audio in questo audit. |
| Pair/cue | Grid costante, 4/4, livelli 16/8/4, gate tempo/key, vocal bars e cue windows in `dj_select.py:65-88,153-213,374-411,625-644`. `plan_pair` usa la fraseologia solo se il chiamante passa esplicitamente `phrase` (`dj_select.py:838-859`). | Default = cue downbeat e `cut_vocal_guard=False` (`dj_select.py:701-708,750-753`). Le finestre 35%/50% e soglie sono ipotesi. | Test sintetici di grid/key/cue/guard; nessun giudizio musicale o audio-pool per la qualità del piano. |
| Frase/pickup | `dj_phrase` raffina ASR con stem, calcola confine MCFlow, ingressi/uscite e switch comune (`dj_phrase.py:496-595,847-899`). `plan_pair_phrase` mantiene solo piani `cut_phrase` verificati (`dj_phrase.py:1399-1429`). | `tools/run_dj_phrase_v1.py` è harness di confronto/render (`:302-354,471-565`), non il planner del set. Confronta anche il default non protetto. | MCFlow AUC è sul corpus simbolico, esplicitamente non sul pool audio (`dj_phrase.py:26-44`). Il runner registra hash di ASR/stem/CMUdict/MCFlow (`run_dj_phrase_v1.py:231-237`), ma non produce una valutazione umana. |
| Set | Decision tree, Held–Karp e Viterbi sono implementati (`dj_set.py:382-522,785-828,1056-1166`); il runner applica una guardia stem+ASR semplice (`run_dj_set_v1.py:217-221,500-529`). | `run_dj_set_v1.py:486-529` non chiama `ds.profile("books")`, non passa `tree_params/order_params`, e non carica `dj_phrase`/`dj_policy`. Le alternative QA sono sullo stesso cue o altri cue (`:603-608`). | Il runner stesso annota che il QA tecnico non è qualità musicale e il set non è stato ascoltato (`run_dj_set_v1.py:745-756`). Se tutti i tentativi falliscono, conserva comunque il meno grave (`:647-673`). |
| Libri/creative | `PROFILE_BOOKS` è concreto ma opzionale (`dj_set.py:233-265`); le regole flow sono diagnostiche (`dj_phrase.py:902-906`); `dj_policy` ordina proposte e dichiara che `dj_set` potrà importarlo (`dj_policy.py:1-8`). | Nessun default dell'entrypoint attiva libri, flow o policy. In questo snapshot le note libro di supporto non sono ancora nel catalogo `conoscenza_mix/libri`; le etichette nel codice sono quindi da trattare come provenance dichiarata, non come prova riproducibile. | `test_dj_books.py` verifica solo che i parametri cambino il comportamento delle funzioni, non l'effetto percettivo o la correttezza delle fonti. |

## Cose realmente verificate

Con `PYTHONPATH=src python -m pytest -q tests/test_dj_select.py tests/test_dj_phrase.py tests/test_dj_set.py tests/test_dj_policy.py tests/test_dj_books.py`: **78 passed, 1 skipped, 2 failed** in 17.97 s. Lo skip è CMUdict non staged. I due fallimenti sono entrambi i test di rendering phrase (`tests/test_dj_phrase.py:231-290`), bloccati in `dj_render.integrated_lufs` dalla dipendenza `pyloudnorm` assente; non sono un verdetto sul rendering. Non ho eseguito GPU, rete, download, pool audio o render del set.

Le prove disponibili sono dunque unit test su dati sintetici, controllo di schema/provenance e statistiche MCFlow. I giudizi utente v1 sono usati come priorità [U], non come stima o validazione (`dj_set.py:11-18,133-142`); non vanno ricontati come test del planner.

## Otto gap concreti e test da aggiungere

1. **Stack di produzione incompleto.** `dj_plan_pool` produce piani `dj_select` senza phrase/set/policy (`dj_plan_pool.py:217-221`), mentre il runner phrase usa la phrase guard solo nel confronto/render (`run_dj_phrase_v1.py:330-356,484-565`). Test: fixture con ASR/stem in cui il top default taglia pickup o fine verso; eseguire ogni entrypoint e asserire quale piano e guardia finiscono nell'artefatto.

2. **Profilo libri mai cablato.** `PROFILE_BOOKS` modifica tree/order/cue (`dj_set.py:235-256`), ma il set runner passa solo override CLI di durata/lunghezza (`run_dj_set_v1.py:486-529`). Test: stessa fixture con `profile("books")` e default; verificare nel JSON parametri, regole R2/R4/R6, arco e requisiti di sezione, poi un ascolto cieco separato. Non attribuire il risultato ai libri prima di questo confronto.

3. **Guardia phrase non è una proprietà end-to-end del renderer.** `phrase_switch` decide con finestre, soglie e stato acustico (`dj_phrase.py:861-899`), ma la verifica delle parole troncate è fatta dal runner, non dalla funzione (`run_dj_phrase_v1.py:201-204,225-229`). Test property-based: per ogni switch accettato, confrontare parole raw/refined, tail e deck PCM prima/dopo il render; nessuna parola deve violare il margine di 30 ms.

4. **Modello di frase con copertura linguistica e timing limitati.** Le sillabe sono distribuite uniformemente nella parola e l'ultimo onset è snap a flux entro 60 ms (`dj_phrase.py:512-530`); CMUdict o fallback ortografico (`dj_phrase.py:246-316`), gate ASR euristico (`:598-617`). Test stratificato per inglese/francese/spagnolo, ASR affidabile/inaffidabile e parole abutting; annotare confine umano, pickup, parola troncata e abstention. L'AUC MCFlow non basta.

5. **Grid e sezioni hanno assunzioni dure.** `BEATS_PER_BAR=4`, grid costante e reset/merge sezioni sono fissi (`dj_select.py:65-77,153-213,374-410`); `dj_plan_pool` salva infatti una “constant-tempo grid” (`dj_plan_pool.py:171-185`). Test: 3/4, tempo drift, poliritmo/double-time, boundary off-grid e provider discordanti; misurare errore di downbeat/phrase level e propagazione al cue.

6. **Provenance/confidenza delle chiavi è incompleta.** Essentia entra con `strength` ma `combine_key_estimates` usa solo `prob`, `window_stability` e accordo (`dj_plan_pool.py:105-120`; `dj_select.py:472-503`); Krumhansl è sempre aggiunto come fallback audio e i provider non portano nel track JSON revision/hash del modello (`dj_plan_pool.py:168-194`). Test: fixture con stime discordanti e provider mancanti; asserire confidence/source, loggare revision/hash e bloccare il blend quando la confidence è davvero insufficiente.

7. **Fallback e QA possono accettare un cambio non valido.** Le alternative dell'albero sono riusate senza una nuova verifica phrase (`dj_set.py:480-522`); il runner sceglie un candidato finché QA non fallisce e, se tutti falliscono, conserva il meno grave (`run_dj_set_v1.py:641-653`). Test: monkeypatch QA fail e candidate fallback; l'artefatto deve essere `fail/blocked` oppure richiedere revisione, mai promuovere automaticamente un fail a transizione finale.

8. **Progressione del set non è garantita dal contratto dichiarato.** `order_set` può restituire meno di 8 brani (`dj_set.py:785-828`) e oltre `max_exact` elimina i nodi con pochi archi medi migliori (`:771-782`), perdendo ponti unici; `choose_cues` rilassa fino a `0.75^step/1.33^step` (`:1056-1100`) e i fallback `first_start/last_end` possono restare fuori dai limiti quando non esiste un phrase start (`:967-999,1002-1035`). Test: grafi sintetici con bridge necessario, pool sotto 8, durata impossibile e sezioni incomplete; assert espliciti su `length`, limiti reali, note di relax e stato infeasible.

Priorità per la prossima validazione: (1) cablare e congelare lo stack scelto, (2) riparare la dipendenza/render test e verificare PCM/word guard, (3) fixture avversarie di grid/key/set, (4) solo dopo confronto umano pairwise su cut/blend/echo/filter e phrase-aware vs default. Le regole libro possono essere confrontate in quel protocollo, ma il codice attuale non dimostra ancora che migliorino “quando cambiare”.
