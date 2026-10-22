# Modelli preaddestrati e selezione dei punti — V10

19 ottobre 2026. CUE-DETR, LinkSeg e SongEval eseguiti realmente; stato e artefatti sono registrati in `runs/pretrained_v10/ROOT_STATUS.json`. Nessun modello è stato addestrato da zero.

## Riutilizzo effettivo

**CUE-DETR.** Il wrapper usa la rete DETR/ResNet50, il checkpoint pubblico e il preprocessing dell'implementazione ETH-DISCO. Sorgente fissata al commit `d0462856ed2f59a1fb65267cfbe87340a65ad1bb`; codice MIT conservato nel checkout. I41.501.638 parametri sono stati caricati con zero chiavi mancanti, inattese o di forma diversa. Il flag dell'inizializzazione ImageNet viene disabilitato per evitare un download ridondante: il backbone timm resta identico e riceve tutti i pesi del checkpoint. Non è stato sostituito con un altro ResNet.

L'inferenza usa WAV canonici, mono22.050Hz secondo librosa, mel128bande/FFT2048, colormap e finestre originali, batch1 e due thread CPU. Conserva punteggi grezzi, normalizzati, picchi originali e duplicazioni. Il parametro `radius=16` opera sugli indici delle predizioni ordinate nell'implementazione pubblica: non viene chiamato distanza di16barre. La soglia0,9 è la soglia normalizzata originale, non una probabilità calibrata.

[Codice ufficiale e checkpoint](https://github.com/ETH-DISCO/cue-detr) · [Wrapper locale](../tools/run_cue_detr.py) · [Prova di caricamento](../runs/pretrained_v10/cue_strict_load_smoke.json).

**All-In-One.** L'adattatore invoca l'API ufficiale e mappa sezioni/eventi nel contratto `structure.v2`. Sul percorso Windows i quattro JSON iniziali dichiarano indisponibilità e zero sezioni. Su Linux/WSL sono stati installati Torch 2.9.1 e NATTEN 0.14.6 compilato per CPU, scaricato il checkpoint ufficiale harmonix-fold0 e completati i quattro stem HTDemucs di Reminder. Il primo tentativo ha rilevato una incompatibilità NumPy nel decoder madmom: la correzione esplicita `dtype=object` conserva lo stesso argmax delle probabilità HMM ed è verificata con uno smoke del decoder di 20 beat.

Il tentativo successivo riusa stem e spettrogrammi, ma il log termina con `KeyboardInterrupt` nel forward NATTEN1DAV: **nessuna inferenza strutturale completa viene dichiarata**. Un successivo controllo osserva anche un runtime WSL appena riavviato, senza attribuirgli la causa dell'interruzione. LinkSeg è il provider funzionale effettivamente eseguito sull'intero pool. Gli artefatti dei tentativi sono conservati.

L'adattatore ora lega input e cache tramite SHA256, impedisce collisioni fra i numerosi file `a.wav` del pool, verifica gli input staged e registra gli hash degli stem. Test mirati coprono anche la corruzione di uno staged input. Nessuna sezione euristica viene presentata come risultato neurale.

[Codice ufficiale](https://github.com/mir-aidj/all-in-one) · [Adattatore](../src/musicmix/pretrained_structure.py) · [Esito del tentativo Windows](../runs/pretrained_v10/manifest.json).

[Tentativo Linux iniziale](../runs/pretrained_v10/structure_allinone_pilot/manifest.json) · [Correzione del decoder](../runs/pretrained_v10/madmom_numpy_compat.json) · [Stato del retry](../runs/pretrained_v10/structure_allinone_retry/INTERRUPTED.json) · [Log del forward](../runs/pretrained_v10/structure_allinone_retry.log).

**SongEval.** Il critico ha eseguito cinque inferenze reali usando il `Generator` ufficiale e MuQ. Testa da 100.792.276 byte, backbone da 1.333.825.096 byte; hash e licenze sono registrati. Cinque finestre di 20 s, CPU con un thread, 174,788 s complessivi. I punteggi restano diagnostici e non entrano nella selezione del pilot.

| Caso | Coherence grezza |
|---|---:|
| V8 trial 3B già rifiutato | 4,353675 |
| Ordine dei segmenti invertito | 4,291658 |
| Seguito naturale di A | 4,375175 |
| Controllo all-beat mai presentato 1 | 3,886315 |
| Controllo all-beat mai presentato 2 | 3,750836 |

Nel caso già rifiutato la coerenza è quasi uguale al seguito naturale: questa diagnostica non giustifica una soglia di accettazione automatica. Un solo raccordo rifiutato non misura la capacità generale del modello. Non sono stati adattati pesi o soglie ai giudizi. [Risultati effettivi, cinque assi e hash audio](../runs/pretrained_v10/critic_song_eval_diagnostics.json).

[Inferenza ufficiale](https://github.com/ASLP-lab/SongEval/blob/main/eval.py) · [MuQ ufficiale](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter) · [Asset verificati](../runs/pretrained_v10/critic_song_eval_assets.json) · [Driver locale](../tools/run_pretrained_music_critic.py).

**Mosaikbox.** Il tentativo di esecuzione originale è bloccato da dipendenze/servizi e formati di analisi mancanti. Una prova ricostruita con proxy locali delle feature è stata esclusa e spostata nell'archivio: non viene usata come baseline Mosaikbox. Non esiste più un suo entrypoint di produzione in `tools/`.

[Audit e limiti](../runs/pretrained_v10/mosaik_baseline/REPORT.md).

**LinkSeg.** Eseguiti codice e checkpoint ufficiali fissati al commit `38dc0dabed14daa5fbef7a61ea3cb046517e74a0`, senza encoder sostitutivo. Checkpoint di 1.497.912 byte, 321.741 parametri, strict load senza chiavi mancanti o inattese. Runtime Linux/WSL CPU con un thread, DGL 1.1.1 e Torch 2.9.1. Si riutilizzano i beat Beat This! con le conversioni temporali del codice originale. Le sezioni sono riportate sulla timeline canonica a 48 kHz.

| Brano | Sezioni previste |
|---|---:|
| Reminder | 16 |
| Too Many Nights | 19 |
| 5% TINT | 10 |
| Creature | 15 |
| Heartless | 13 |
| MAFIA | 13 |

Gli 80 confini interni vengono usati come opportunità alternative di cambio. Le etichette funzionali non diventano automaticamente prove di risoluzione della suspense. Sono verificati hash dei sorgenti e checkpoint, identità audio/beat e protezione dalla sovrascrittura. Il solo modulo di esportazione JAMS non utilizzato è sostituito da un modulo vuoto per evitare dipendenze del formato: rete, feature e decoder ufficiali restano quelli originali.

[Repository ufficiale](https://github.com/morgan76/LinkSeg) · [Inferenze sui sei brani](../runs/pretrained_v10/linkseg_pool6/manifest.json) · [Runner](../tools/run_linkseg_official_pilot.py) · [Confini normalizzati](../runs/pretrained_v10/linkseg_registry_pool6.json).

## Cosa cambia nella selezione

Il modello propone opportunità nel singolo brano. Un adattatore leggero controlla se esse supportano **entrambi i punti effettivamente renderizzati** di un candidato A→B, non solo le ancore nominali sulla griglia. Hash, sample rate e durata devono coincidere con i WAV canonici; un cue appartenente a un'altra edizione del brano viene rifiutato.

La prova riutilizza il pool completo di candidati già ammessi dalla ricerca V8 e rimuove le regioni già presentate. Confronta tutti i beat con il sottoinsieme supportato dal modello. Punteggi delle coppie e renderer restano invariati; entrambi i bracci usano la medesima nuova selezione di diversità. Non è una riproduzione delle quattro prove di ascolto V8.

La tolleranza primaria di mezzo beat è fissata prima dei risultati; un quarto e un beat sono analisi di sensibilità. I punti troppo distanti o senza inferenza non ricevono supporto inventato. Zero candidati del modello produce un esito esplicito, senza promuovere al suo posto la baseline.

Il primo confronto sulla griglia congelata di quattro brani ha dato **zero candidati supportati**, su 22 ancora inediti dopo il filtro dello storico. CUE-DETR ha realmente proposto 20 punti (7 Reminder, 5 Too Many Nights, 2 5% TINT, 6 Creature), ma un'opportunità nel singolo brano non equivale a una coppia utilizzabile.

La seconda prova usa i timestamp del modello direttamente, invece di cercare soltanto candidati vicini sulla griglia preesistente. I beat originali restano una misura indipendente del tempo: non sono sostituiti con i cue. Il contesto audio viene traslato al punto effettivo, restano i controlli vocali e la differenza di fase dei due confini effettivi deve rientrare nella politica esistente di 80 ms. Questo controllo non certifica allineamento delle frasi o risoluzione della suspense. Il protocollo distingue questa prova successiva dalla prima.

Sul pool originale, con il controllo di fase applicato, restano **7 candidati ammessi e zero inediti** secondo lo storico. Non vengono rigenerati ascolti già presentati. [Audit del filtro storico](../runs/pretrained_v10/direct4_phase_history_audit.json).

Il pool V10 aggiunge **Heartless di The Weeknd** e **MAFIA di Travis Scott**, dalle rispettive pubblicazioni audio ufficiali. Identità, versioni e hash sono nel [registro delle sorgenti](../runs/pool_v10/INGEST_REPORT.md); Beat This!, Demucs e Whisper già disponibili vengono riutilizzati. Lo stato pronto del catalogo richiede entrambi i provider, non il solo WAV.

## Risultati sul pool di sei brani

Catalogo verificato: **6/6 pronti**, tutte le **30 direzioni** ricercate. CUE-DETR ha prodotto 33 cue: 7/5/2/6 sui quattro brani iniziali, 4 su Heartless e 9 su MAFIA. La baseline su tutti i beat ammette 31.942 candidati, di cui 5.702 non esclusi dallo storico. Il filtro storico opera su entrambi i brani, anche quando ricompaiono in ruoli invertiti.

| Opportunità preaddestrate | Ammessi prima dello storico | Inediti e supportati | Selezionati per diversità |
|---|---:|---:|---:|
| CUE-DETR, tempi originali | 16 | 6 | 2 |
| LinkSeg, confini interni | 36 | 7 | 2 |

I limiti di diversità restano identici: distanza di 8 s e massimo due impieghi di un brano nello stesso ruolo. Non vengono rilassati per riempire quattro posti. I controlli sono i medesimi quattro render in entrambi gli esperimenti: la galleria li mostra una volta sola. Il confronto è diagnostico fra selezioni, non un test A/B con contesto uscente uguale.

| Modello | Uscita A | Ingresso B |
|---|---|---|
| CUE-DETR | Creature 196,500 s | Heartless 0,511 s |
| CUE-DETR | MAFIA 74,211 s | Heartless 11,726 s |
| LinkSeg | 5% TINT 68,860 s | Creature 162,313 s |
| LinkSeg | Heartless 11,459 s | MAFIA 0,464 s |

Ogni render contiene 10 s prima e 10 s dopo il raccordo, con lo stesso renderer di base. I controlli tecnici e lo stress dei confini ASR sono salvati separatamente: non certificano parole complete né preferenza musicale. Nessun nuovo voto utente è richiesto per completare il pilot.

Sui due render CUE-DETR non emergono flag tecnici. Sui due render LinkSeg il controllo segnala rispettivamente un salto RMS di circa +11,89 dB e +11,58 dB; il secondo ha anche un bordo quasi silenzioso di almeno 50 ms. Questi sono limiti concreti dei render grezzi: non vengono nascosti né corretti soltanto nel braccio del modello, perché cambierebbe il confronto. Le etichette dei flag distinguono osservazioni di livello e possibili artefatti da giudizi musicali.

[Manifest CUE-DETR](../runs/pretrained_v10_pool6/manifest.json) · [Timestamp e audio CUE-DETR](../runs/pretrained_v10_pool6/TIMESTAMPS.md) · [Manifest LinkSeg](../runs/pretrained_v10_linkseg6/manifest.json) · [Timestamp e audio LinkSeg](../runs/pretrained_v10_linkseg6/TIMESTAMPS.md).

La [galleria locale unica](http://127.0.0.1:8771/test/review/index.html) contiene **8 audio distinti**: quattro proposte dei modelli e quattro controlli condivisi, senza duplicati. Include seek ±5 s, filtri, timestamp e osservazioni tecniche. [Manifest della galleria](../runs/pretrained_v10_review/manifest.json) e [QA HTTP](../runs/pretrained_v10_review/qa_http.json): otto HEAD, otto Range 206 e otto Range non validi 416 verificati; copia audio identica alle sorgenti. Riavvio: `python tools/serve_pretrained_review.py --serve --port 8771`.

Lo storico successivo è `runs/audition_history_v10.json`: 41 geometrie, di cui 8 nuove presentazioni, non nuovi voti. Il vecchio input resta invariato ed è anche archiviato in `runs/pretrained_v10/history_before_v10.json`. Le prossime ricerche dei runner del pool usano lo snapshot V10 quando presente. La suite completa passa **219 test e 3 subtest**.

[Protocollo fissato prima dei risultati](../runs/pretrained_v10/PILOT_PROTOCOL.md) · [Integrazione della selezione](../src/musicmix/learned_opportunities.py) · [Runner del confronto](../tools/run_pretrained_pool_pilot.py).

## Riproduzione

Con runtime e asset isolati già preparati, dalla cartella del progetto:

```powershell
python tools/run_cue_detr.py --out runs/nuova_inferenza/cue_detr.json --batch-size 1 --threads 2
python tools/prepare_cue_registry.py --input runs/nuova_inferenza/cue_detr.json --output runs/nuova_inferenza/cue_registry.json
python tools/run_pretrained_pool_pilot.py --cues runs/nuova_inferenza/cue_registry.json --output runs/nuovo_pilot
```

Usare nuovi percorsi di output per nuove prove e preservare i risultati congelati. I runtime dei modelli sono opzionali; non fanno parte delle dipendenze obbligatorie della CLI di base. La prima installazione richiede gli asset ufficiali e le versioni registrate nei manifest, non il solo comando del wrapper.

Per riprodurre le selezioni V10 dai risultati di inferenza già verificati occorre lo **storico precedente** archiviato: quello V10 include ora anche queste presentazioni e le escluderebbe correttamente. Esempio con due nuove cartelle di output:

```powershell
python tools/run_pretrained_pool_pilot.py --catalog runs/pool_v10/catalog_ready.json --cues runs/pretrained_v10/cue_registry_pool6.json --frozen-run runs/pretrained_v10_pool6_search --provided-run runs/pretrained_v10_direct6_search --history runs/pretrained_v10/history_before_v10.json --output runs/replay_cue_v10
python tools/run_pretrained_pool_pilot.py --catalog runs/pool_v10/catalog_ready.json --cues runs/pretrained_v10/linkseg_registry_pool6.json --frozen-run runs/pretrained_v10_pool6_search --provided-run runs/pretrained_v10_linkseg6_search --history runs/pretrained_v10/history_before_v10.json --output runs/replay_linkseg_v10
```

## Limiti di interpretazione

Copertura dei cue, assenza di clipping e punteggio SongEval non equivalgono a preferenza musicale. CUE-DETR proviene soprattutto dal dominio EDM; SongEval valuta estetica di canzoni e non è stato certificato per raccordi cross-song brevi. I confini ASR restano stime: non garantiscono parole integre.

Il progresso da misurare qui è l'esecuzione fedele dei modelli esistenti, l'effetto osservabile sui candidati e la sensibilità del critico. La comprensione affidabile della suspense e il miglioramento percettivo richiedono ancora il benchmark descritto nella V9. Nessun voto utente è necessario per completare questo pilot diagnostico.
