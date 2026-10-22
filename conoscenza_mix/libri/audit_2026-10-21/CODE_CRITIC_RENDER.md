# Audit codice critic/render/effects/creative/QA (2026-10-21)

> Nota storica recuperata dal primo checkout a8cf580c. Non descrive lo stato corrente: l'API criticv3 e la sua gestione delle feature mancanti sono già implementate. Le proposte di primo ensemble/astensione sotto sono superate. Usare REVISIONE_PIPELINE.md, CURRENT_TEST_RESULTS.json e RENDER_TRIAGE.md per la revisione attuale.

Audit read-only dello snapshot corrente. Ho letto `AGENTS.md`, l'indice e i riassunti pertinenti di `conoscenza_mix`. Le evidenze dei libri/pagine non sono state ancora esaminate in questa audit; le osservazioni sotto sono quindi solo codice e test esistenti. Non ho modificato il codice, né usato GPU, rete o download.

## Cosa è realmente cablato

- L'entrypoint operativo `tools/run_dj_creative_v1.py` seleziona coppie/cue tramite `dj_select`/policy, applica le tecniche di `dj_creative`/`dj_render`, esegue `dj_qa`, scarta i `fail` e scrive manifest di review. Il percorso di retry per la fase dei deck è alle righe 360-400: usa di nuovo lo stesso tracker QA dopo un phase nudge.
- `tools/build_listening_artifact.py` impacchetta i manifest e genera la galleria HTML/MP3. Non invoca `dj_critic` e non esegue una verifica percettiva o una verifica PCM dopo l'encoding.
- `tools/run_dj_critic.py` è un percorso separato di benchmark/API per i critic; il suo piano include stadi `muq`, `musicgen`, `emb`, `audiobox`, `qa`, `craft`, `train`, `score`, `validate` e report. Non è agganciato al ciclo creativo della galleria.
- `src/musicmix/dj_render.py`, `dj_effects.py` e `dj_creative.py` sono primitive di rendering/effetti e utility creative; non scelgono da sole il pair/cue. Le forme di fader/EQ, i livelli, i durate e molti guard sono marcati nel codice come ipotesi `[H]`/`[P]`.

## Stato della validazione eseguita

Comando eseguito senza cache bytecode:

```text
$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -p no:cacheprovider tests/test_dj_critic.py tests/test_dj_render.py tests/test_dj_effects.py tests/test_dj_creative.py tests/test_dj_qa.py tests/test_build_listening_artifact.py tests/test_dj_books.py -q
```

Risultato esatto: `31 passed, 5 skipped in 17.88s` (exit 0). I test sono utili per contratti e regressioni deterministiche; `tests/test_dj_critic.py` dichiara esplicitamente audio generato e nessun modello preaddestrato. Non dimostrano validità percettiva su rap reale, calibrazione con ascoltatori, indipendenza dell'evaluator o correttezza di un servizio HTTP di seek. `test_dj_books.py` verifica policy/default e contratti, non un percorso end-to-end book-to-audio.

## Top 8 gap concreti e validazioni proposte

### 1. Il critic aggrega score non calibrati sul target percettivo

`src/musicmix/dj_critic.py:109-118` fissa una combinazione con pesi uguali tra aesthetic, surprisal, seam e technical; `:667-684` fa la media dei critic disponibili senza una soglia minima di copertura. `MusicGenSurprisal` (`:411-481`) è teacher forcing su MusicGen-small/EnCodec, mentre `LinearSeamModel` (`:553-604`) è un logistic model allenato sui dati passati. Il benchmark di craft è costruito con difetti sintetici; score alti su quei difetti non sono una misura di “buona transizione” naturale. La mappa technical pass/warn/fail (`:607-645`) e i pesi non risultano calibrati contro giudizi umani. Se mancano critic, il risultato resta utilizzabile con `missing`, senza abstain.

Validazione: dataset reale di transizioni rap con rating pairwise e controlli di continuazione; split per track/pair con GroupKFold annidato; fit di calibrazione e pesi solo sul train; holdout umano separato; curva precision/coverage e abstention quando manca un critic o un controllo tecnico. Riportare anche l'ablazione di ciascun critic.

### 2. L'evaluator QA non è indipendente dal renderer che corregge

Nel live path `tools/run_dj_creative_v1.py:360-400`, una prima QA fallita sul ritmo determina `deck_phase_error`, poi il render viene nudged e riesaminato con lo stesso tracker/criterio. Il commento a `:391` ammette che il controllo del render conservato usa lo stesso tracker e “is not independent”. Questo può ottimizzare una metrica condivisa senza mostrare che l'errore percepito è migliorato.

Validazione: congelare un evaluator indipendente (tracker/feature extraction e soglie fissati prima del render), usare un secondo set di transizioni e rating umani ciechi; il phase nudge deve essere deciso su train/dev e valutato una sola volta su holdout. Confrontare anche contro un nudge casuale e il baseline senza nudge.

### 3. `pass` può significare “non misurato”

`src/musicmix/dj_qa.py:936-1001` costruisce il verdetto da `fail`/`warn`; i check `skipped` non entrano in `reasons`, e `passed = verdict != "fail"`. In particolare `check_rhythm` (`:588-621`) può essere skipped se tracker/deck non sono disponibili e `check_vocal_clash` (`:867-881`) / `check_verse_cut` (`:884-930`) saltano se mancano stem o parole. Il render quindi può essere pubblicato come pass con copertura QA parziale.

Validazione: matrice di casi con ogni combinazione di tracker/stem/words/grid mancanti; misurare coverage per check e impedire `pass` se un check obbligatorio è skipped (oppure produrre un verdetto esplicito `abstain`). Verificare sul manifest che il consumer non confonda `warn` e `pass`.

### 4. Provenienza e range sorgente non proteggono dal padding silenzioso

`dj_render.py:693-804` e `dj_effects.py:283-341` accettano array e un `provenance` libero; il JSON di grid caricato da `dj_render.py:174-189` non lega grid, audio e stem a hash/ID obbligatori. `_slice` in `dj_render.py:1167-1189` e `dj_effects.py:109-160` limita l'intervallo e riempie fuori range con zeri. Così un cue/offset errato può produrre un WAV apparentemente valido invece di abstain; l'offset nel provenance non è una prova che il PCM usato sia quello dichiarato.

Validazione: fixture con hash audio/grid/stem e cue volutamente fuori range o su track sbagliata; atteso errore/abstain, oppure metadata obbligatorio `source_padding_samples` e `source_hash`. Dopo ogni render, ricostruire e confrontare la finestra PCM usata con l'hash/offset dichiarati. Verificare anche mismatch di sample rate e track ID.

### 5. Cue, downbeat e semantica di sezione sono assunti dal chiamante

I parametri di snap in `dj_render.py:754-755` e le tolleranze QA (`dj_qa.py:57-82`) sono soglie `[H]`; `dj_effects.py:657-661` definisce `cue_in_is_section_start` come “garanzia del chiamante”, e il metadata a `:696-703` dice esplicitamente “not verified acoustically here”. Anche il tracking dei beat può degradare a check skipped/warn. Un label di sezione errato può quindi far passare `drop_in`/guard vocali con una premessa falsa.

Validazione: annotazioni indipendenti di downbeat, section start e vocal phrase su un holdout; misurare errore di snap e falsi pass/fail per tecnica. Separare “asserted” da “verified” nel verdetto e abstain se il contratto di cue non è verificato.

### 6. La review umana è contaminata e manca un vero controllo “resta A”

`tools/build_listening_artifact.py:979-1035` mostra nella galleria tecnica, badge `warn QA`, ragioni QA e label per versione; `run_dj_creative_v1.py:631-640` documenta che “continuing A is not offered as an item to rate”. Il listener può quindi usare la spiegazione tecnica come scorciatoia e non può confrontare direttamente il cambiamento con il controllo di non intervento. Un “standard reference” di transizione non sostituisce sempre il vero stay control.

Validazione: manifest randomizzato/blindato con ID opachi, tecnica e QA rivelati solo dopo il rating; includere per ogni pair un controllo continuation-A con loudness/inizio/fine comparabili; misurare inter-rater agreement, position/order effect e scelta “change vs stay”.

### 7. Le coordinate B sotto stretch non corrispondono sempre al PCM renderizzato

`dj_qa.py:1018-1085` converte parole/stem B con una posizione lineare basata su `tempo_ratio_b` (`:1044-1045`), mentre `dj_render.py:891-1139` usa mapping piecewise per griglia/barre e mantiene B stretchato attraverso l'overlap. In `dj_creative.py:863-891`, `map_to_clip` salta i segmenti con `rate != 1`; ciò può omettere parole/attività vocali proprio nella parte stretchata. I controlli source-vs-PCM possono quindi validare la coordinate sbagliata.

Validazione: usare una time-map per campione o per segmento salvata dal renderer e testare parole/stem su ogni boundary di stretch; confrontare marker rilevati dal PCM con marker proiettati; includere casi di rate piecewise, non solo ratio costante.

### 8. Le tecniche creative e fallback restano euristiche senza abstention percettivo

`dj_creative.py:1-56,77-125` dichiara forme, durate e livelli come ipotesi; `:899-909` usa `mute_guard` euristico; `:1280-1294` usa root 9 e crea comunque un riser quando la key non è nota (`key_known=false`). La parte lyric (`:1650-1782`) assegna pesi hand-coded a `rhyme_key`/Soundex (`1.0/.85/.7/.5`). Queste feature possono essere utili per proporre candidati, ma non sono evidence di qualità percettiva e non hanno un limite di confidenza che blocchi un candidato fragile.

Validazione: benchmark umano stratificato per tecnica, con key/ASR/lyrics corrotti e casi di “no safe candidate”; calibrare i pesi su train e misurare correlazione/choice accuracy su holdout. Se key o lyric alignment è sconosciuto, emettere candidato esplicito `abstain`/“non usare riser” invece del fallback pitchato; confrontare il fallback con un controllo senza effetto.

## Lettura complessiva

La pipeline attuale è una catena riproducibile di proposte, render e controlli tecnici, con alcuni componenti preaddestrati usati come feature/score (`Audiobox`, `MusicGen`, embedding) e molte decisioni calibrate a mano. Non è ancora una pipeline di giudizio percettivo indipendente: il critic è separato dal live entrypoint, la copertura mancante non produce abstain, alcune coordinate/provenance non sono vincolate al PCM, e la galleria espone informazioni che dovrebbero restare nascoste durante l'ascolto. Le 31 prove passate confermano contratti software dello snapshot; non supportano claim di generalizzazione percettiva. Nessuna evidenza derivata dai libri è stata incorporata in questo audit.
