# Pool e auto-feedback V8 — 19 ottobre 2026

Sono pronti quattro confronti A/B (otto WAV distinti da 20 secondi) su Reminder, Too Many Nights, 5% TINT e Creature. Ogni confronto conserva esattamente gli stessi dieci secondi uscenti; cambia l'ingresso. Tutti i brani compaiono una volta come sorgente uscente. Due confronti misurano il reranking sperimentale della continuazione, due esplorano un altro brano quando la policy non cambia il vincitore: questi ultimi non sono presentati come miglioramenti del ranking.

[Ascolto locale](http://127.0.0.1:8769/test/3bdeea23988e22af/index.html) · [Timestamp](../runs/pool_experiment_v8_final/TIMESTAMPS.md) · [Auto-feedback](../runs/pool_experiment_v8_final/AUTO_FEEDBACK.md) · [Ricerca e limiti delle fonti](../ricerca/v8/INDEX.md).

Il player permette di spostarsi nella registrazione, usa Range/HEAD e include −5/+5 secondi, salto al raccordo a 10 secondi e sorgenti con timestamp. Il server salva i giudizi in `../giudizi json`, con nome `giudizi_Pool_4_brani_V8_tutti_beat_3bdeea23988e22af.json` e revisioni. La chiave A/B privata non viene servita. I timestamp espliciti possono naturalmente rivelare l'identità dei brani; resta nascosta l'assegnazione degli algoritmi.

## Orchestrazione realmente eseguita

Otto task DeepSeek V4.1 Flash Free, variant high, con agente build di OpenCode 1.18.31: psicologia, MIR, ricerca nel pool, novelty/struttura, rendering, auto-feedback, allineamento del cantato e validazione del critico. Un server condiviso, prima sei e poi due task; cartelle separate, guard memoria e controllo quota. Le otto sessioni riuscite risultano archiviate e cancellate e il costo riportato dal provider è zero. I cinque tentativi iniziali con credenziale scaduta sono registrati separatamente; il server è stato riavviato usando la chiave pubblica del provider gratuito, senza cambiare credenziali globali. Il server OpenCode è stato fermato dopo il completamento; il server degli ascolti resta disponibile.

Registri: `runs/swarm_v8/swarm.json`, `swarm_followups.json`, cartelle `agents/`. Gli agenti hanno ricercato e scritto codice; l'integrazione, l'inferenza locale e la verifica dei WAV sono dell'orchestratore. I rapporti e i JSON delle fonti distinguono testo completo, lettura parziale e metadati. Le 50 voci bibliografiche includono repository e ripetizioni tra temi, non 50 pubblicazioni nuove lette integralmente.

## Pool, ricerca e diversità

Per le due nuove registrazioni (file dell'utente) metadati e hash sono conservati in `local_audio/pool_v8` e nel catalogo `runs/pool_v8/catalog.json`. Il titolo Travis è 5% TINT. Sono stati eseguiti Beat This!, Demucs htdemucs e faster-whisper small.en anche sui due nuovi brani; nessun forced aligner specifico per il canto è stato eseguito.

La ricerca copre tutte le 12 direzioni ordinate: 1.611.642 coppie della griglia completa dei beat, senza troncamento. Il prefiltro vocale esatto ne evita 1.597.302 e ne valuta 14.340. Una verifica reale separata su tutte le direzioni della griglia downbeat conserva esattamente i candidati ammessi e i punteggi del percorso senza prefiltro (`runs/swarm_v8/prefilter_real_equivalence.json`). La cache delle traiettorie evita di estrarre nuovamente ogni contesto; gli otto risultati finali coincidono con l'estrattore completo.

Dopo deduplicazione e storico giudicato restavano 1.438 geometrie; includendo le due proposte V7 già mostrate restano 1.181: 312 con evidenza euristica di continuazione e 869 sconosciute. Nessuna è certificata come risoluzione musicale. La selezione usa 42 gruppi di uscita, 67 nodi di ricerca senza troncamento, distanza minima di 8 secondi sui cue e un confronto per brano uscente. Non riutilizza gli audio V7. All'interno di ciascun confronto il contesto uscente uguale è intenzionale.

Lo storico congelato della run contiene 25 geometrie (23 giudicate più due proposte V7). Per le prossime run il registro `runs/published_listening_tests.json` aggiunge anche gli otto audio V8: 33 geometrie totali, senza trasformare proposte non giudicate in preferenze umane. `tools/update_audition_history.py` rigenera entrambi i registri.

## Cosa valuta l'auto-feedback

Il primo livello analizza il WAV renderizzato: clipping, picchi, discontinuità, DC, silenzio al raccordo e variazioni RMS. Può respingere un risultato con clipping e cercare una sostituzione entro un limite esplicito di iterazioni. Qui tutti gli otto audio passano il controllo tecnico duro al primo giro, con tre variazioni RMS descrittive oltre 6 dB: non vengono automaticamente chiamate errori musicali.

Il secondo livello esegue realmente Audiobox Aesthetics locale, con pesi ufficiali safetensors verificati, su dieci secondi centrati sul raccordo (secondi 5–15 del WAV). Conserva separati Content Enjoyment, Content Usefulness, Production Complexity e Production Quality. Non modifica la selezione e non produce un voto musicale unico. Gli agenti testuali DeepSeek non hanno ascoltato gli audio.

È stato aggiunto un controllo con due secondi artificialmente saturati attorno al raccordo: Production Quality scende di circa 0,154 mentre Content Enjoyment sale di circa 0,015. Questo singolo esempio non calibra il modello e mostra perché non si deve ottimizzare ciecamente un asse. File: `aesthetic_diagnostics.json`; misure tecniche in `quality_feedback.json`.

## Verifica e limiti

Suite finale: **196 test e 3 subtest passati**. Il primo comando senza percorso raccoglieva anche le copie archiviate degli agenti e falliva per omonimia; `testpaths = ["tests"]` in `pyproject.toml` corregge la raccolta. La suite completa del progetto passa con `python -m pytest -q`.

QA audio superata: otto WAV unici, campioni interni corrispondenti alle sorgenti canoniche, contesto uscente identico per coppia, copie servite identiche e traiettorie coincidenti con l'estrattore completo. QA trasporto superata: 24 richieste Range, otto HEAD, otto player navigabili e otto didascalie sorgente; chiave privata restituisce 404. La successiva correzione delle didascalie è documentata separatamente in `listening/ui_revision.json`; gli audio e il manifest originale restano quelli della run.

Lo stress sui timestamp non trova sovrapposizioni con le parole ASR a 0/20 ms di espansione ipotetica, ma sei degli otto raccordi diventano sensibili a 50 ms (anche a 100/200 ms). Non dimostra che sei parole siano tagliate: indica che non possiamo garantire confini vocali corretti con questi timestamp. Le probabilità ASR non sono confidenza calibrata del confine.

Il criterio motion resta euristico, con peso 0,20 e sconosciuto neutro; usa il seguito naturale di A come riferimento offline. Novelty/ricorrenza non diventano un provider di frase o arrivo. Non è stata validata la capacità di riconoscere suspense risolta. Il renderer resta un passaggio sequenziale con rampe di 12 ms e guadagno 0,9, senza correzione tempo/pitch. Il modulo opzionale di sovrapposizione ha guard vocali/tempo e test, ma non è usato in questi confronti.

I giudizi storici V3/V4 sono conservati; nessun ascolto umano V8, miglioramento percettivo o risultato SOTA è dichiarato. I prossimi blocchi sostanziali sono: confini del canto verificati su annotazioni, riconoscimento di frasi/arrivi, calibrazione indipendente del critico con controlli di diverso tipo e confronti umani separati da quelli usati per scegliere pesi. Il pool di quattro brani è un pilot, non un benchmark generalizzabile.

## Riproduzione

Dalla cartella del progetto, nell'ambiente già predisposto:

```powershell
$env:PYTHONPATH='src'
python -m pytest -q
python tools/update_audition_history.py
python tools/run_pool_experiment.py --output runs/nuova_prova --cue-policy beats
python tools/audit_pool_experiment.py runs/nuova_prova
python tools/score_mix_aesthetics.py runs/nuova_prova --output runs/nuova_prova/aesthetic_diagnostics.json --counterfactual
```

L'output deve essere nuovo. Il default usa lo storico anche delle proposte pubblicate, quindi una nuova run non riproduce la selezione V8 ormai esclusa: per riprodurre le condizioni di questa selezione usare `--history runs/pool_experiment_v8_final/inputs/history.json`. Il manifest finale conserva catalogo, hash, storia e provenienza della ricerca riutilizzata dalla run all-beat precedente. I pesi Audiobox sono necessari solo per la diagnostica neurale; non vengono scaricati implicitamente dal critico.
