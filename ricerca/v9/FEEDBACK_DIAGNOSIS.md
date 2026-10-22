# Diagnosi V8 dopo i giudizi salvati

Import verificato tramite test_id, run_id, chiave A/B e hash dei WAV. Quattro osservazioni, otto audio; nessuna stima di generalizzazione e nessun nuovo peso o soglia ricavato da questi casi.

| Prova | Audio | Sistema | Esito del confronto | Motion utente | Parola tagliata | Motion euristico | Audiobox CE | Audiobox PQ |
|---:|---|---|---|---|---|---|---:|---:|
| 1 | A | expectation_pool_v8 | neither | unclear | False | continuation | 7.305 | 8.231 |
| 1 | B | directional_pool_baseline_v8 | neither | unclear | False | unknown | 7.225 | 8.248 |
| 2 | A | directional_pool_baseline_v8 | tie | unclear | False | unknown | 7.369 | 8.054 |
| 2 | B | expectation_pool_v8 | tie | interrupts | False | continuation | 7.001 | 7.514 |
| 3 | A | pool_alternative_v8 | neither | interrupts | True | unknown | 7.210 | 8.195 |
| 3 | B | directional_pool_baseline_v8 | neither | interrupts | True | unknown | 7.529 | 8.344 |
| 4 | A | directional_pool_baseline_v8 | A | unclear | False | continuation | 7.366 | 8.229 |
| 4 | B | pool_alternative_v8 | A | interrupts | False | unknown | 7.294 | 8.322 |

## Evidenza e interpretazione

Due risposte “neither” rifiutano entrambi gli audio; “tie” indica solo parità relativa. La preferenza A nella prova 4 favorisce la baseline rispetto all’alternativa e non certifica una qualità assoluta sufficiente. Nei due confronti specifici di reranking la policy nuova non ottiene una vittoria: prova 1 neither, prova 2 tie.

Quattro audio sono annotati “interrupts”; nessuno “continues” o “resolves”. La policy assegna “continuation” anche all’audio B della prova 2, annotato come interruzione. Questo è un controesempio concreto alla capacità della feature attuale di rappresentare l’esperienza riferita, senza pretendere una matrice di confusione statisticamente valida su quattro prove.

Tutti gli otto audio superavano il controllo tecnico duro. Esso non ha quindi escluso i due audio con parole tagliate né i quattro con interruzione del motion. Il controllo misurava clipping/integrità del render: questi esiti mostrano un limite di copertura, non un errore di calcolo della metrica tecnica.

L’audio B della prova 3 ha il valore CE più alto fra gli otto (circa 7,529) e PQ circa 8,344, ma è dentro un confronto rifiutato con parola tagliata e motion interrotto. Nella prova 4 il PQ è maggiore nell’audio non preferito. Sono esempi sufficienti a vietare qui l’interpretazione “PQ alto = passaggio musicalmente riuscito”, non una calibrazione generale del modello o una prova che l’asse PQ debba misurare suspense.

Le due segnalazioni di word-cut nella prova 3 condividono lo stesso contesto uscente. È una pista per controllare il confine comune, non una localizzazione confermata del difetto. Le perturbazioni dei timestamp ASR nel file originale restano ipotetiche e non possono diventare automaticamente nuove etichette di parole tagliate.

## Cambio di problema da modellare

La ricerca corrente seleziona abbinamenti ammessi su beat e gap, poi cerca una differenza di vincitore con l’aggiunta motion. Non possiede un’azione esplicita “resta in A”, una stima validata del completamento della frase o una rappresentazione del periodo di preparazione e dell’arrivo atteso. L’esplorazione del pool e la deduplicazione risolvono copertura/riuso, non questa mancanza. Questa diagnosi deriva dal codice e dai giudizi; le proposte di architettura saranno collegate alle nuove fonti nel dossier.

## Riproducibilità

Dati per audio e hash degli input in [feedback_audit.json](../../runs/swarm_v9_research/feedback_audit.json). Import normalizzato nella cartella abituale `giudizi json/feedback_Pool_V8_3bdeea23988e22af.json`; export originale conservato. Storico successivo: 31 geometrie giudicate, 33 includendo le due proposte V7 non ancora normalizzate come giudizi. Nessuna proposta non giudicata viene usata come preferenza umana.
