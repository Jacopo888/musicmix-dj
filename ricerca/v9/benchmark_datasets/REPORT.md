# Dati e benchmark per WHEN e feedback autonomo

Ricerca avviata da DeepSeek e completata dall’orchestratore dopo esaurimento quota. Le copie recuperate e le sessioni sono conservate; le fonti effettivamente usate sono elencate con scope in `sources.json`. Non sono stati scaricati corpora audio grandi o checkpoint in V9.

## Quale etichetta serve davvero

Tre famiglie di dati rispondono a domande diverse: annotazioni di struttura/voce indicano cosa avviene e quando; esempi di mix professionali indicano una scelta riuscita nel proprio contesto, con possibili errori di allineamento; preferenze confrontano risultati rispetto a una popolazione/rubrica. Non usare una delle tre come ground truth automatica delle altre.

| Risorsa | Uso concreto | Limite che conta qui |
|---|---|---|
| D01 Raveform | Esempi professionali, durate e funzioni dei segmenti | EDM, endpoints derivati da allineamenti |
| D02 EDM-CUE | Benchmark del provider di opportunità di cue | Cue singolo, annotazioni incompiute, nessun audio |
| D03 M-DJCUE | Tipi IN/OUT e confronto fra annotatori | IN in intro per scelta di annotazione |
| D04 UnmixDB | Verifica clock, fade, allineamento e corruzioni | Mix generati, non gusto professionale |
| D05 JamendoLyrics | Timing di parole nel canto e copertura | Alignment e ASR sono task distinti |
| D06 signalign-bench | Errori separati per rap/melisma/voce coperta | Stesso corpus Jamendo, bucket di un annotatore |
| D07 SongFormBench | Confini e funzioni di sezioni | Non dice quando abbandonare una frase |
| D08 TenseMusic | Traiettorie di tensione | Classica, non preferenza cross-song |
| D09 SongEval | Coerenza e fraseggio su canzoni generate | Rating sull’intera canzone |
| D10 CMI-Pref | Confronti di musicalità e istruzione | Nessuna etichetta diretta di risoluzione DJ |
| D11 MusTBench | Verificare timestamp e ordine del giudice audio | Grounding, non approvazione estetica |
| D12 HumMusQA | Controllo comprensione e scorciatoie testuali | Domande, non quality rating del raccordo |

## Priorità dati: eventi professionali e coordinate verificabili

Raveform è la pista più concreta per non chiedere giudizi a ogni iterazione. Prima di usarlo occorre congelare una versione, identificare la registrazione originale, controllare il match e ricostruire i clock A/B/master. Lo stesso originale compare in più set: split casuali per transizione disperdono la medesima registrazione tra train e test e possono gonfiare la performance. Raggruppare almeno per traccia e controllare artista/versione; registrare la copertura del repertorio.

La sua parte strutturale è annotata da esperti, mentre gli intervalli di mix provengono da allineamenti. Conservare questa differenza nel target e pesare/escludere casi dubbi con una regola dichiarata prima della prova. Non trasformare ogni punto non scelto in negativo: DJ differenti e intenti diversi possono rendere valide più alternative.

EDM-CUE e M-DJCUE permettono un benchmark del sottocompito cue. Il vincolo IN-intro di M-DJCUE è una limitazione della copertura dell’annotazione: un sistema che entra a metà brano non va valutato come sbagliato perché manca l’etichetta. Le tolleranze pubblicate devono restare separate da quelle vocali. Versioni vinyl/YouTube possono avere offset e fattori di playback diversi; la conversione dei timestamp precede qualsiasi valutazione.

## Parole tagliate: valutare anche quelle perse dal modello

JamendoLyrics permette di misurare onset/offset rispetto a parole annotate e linee vocali. La scheda distingue il benchmark di allineamento dalla trascrizione. Un forced aligner con lyrics corrette e un sistema ASR senza lyrics non sono lo stesso esperimento.

La risorsa comunitaria signalign-bench aggiunge bucket di difficoltà agli stessi79 brani, non79 esempi indipendenti in più. Il protocollo conta parole non riconosciute come errori di copertura: una MAE bassa sulle sole parole recuperate può nascondere molti dropout. Per MusicMix servono almeno copertura, errore di onset e offset, quantili/tail e percentuale entro soglie diverse, riportati per rap e voce coperta. I numeri del maintainer non diventano prestazioni del nostro pipeline: nessun allineatore nuovo è stato eseguito qui.

Non esportare lyrics intere nel report: sono sufficienti ID, tempi, metadati e misure. La licenza del benchmark non cancella quella delle registrazioni.

## Perché servono controlli distinti

UnmixDB è utile per verificare che il renderer rispetti clock, fattore di tempo, intervalli e curve note. Il suo ground truth è esatto per costruzione; non dimostra che un DJ avrebbe scelto quei passaggi. TenseMusic valuta curve, SongFormBench confini/funzioni, MusTBench grounding e SongEval/CMI giudizi diversi. Il punteggio massimo su una di queste risorse non è un punteggio complessivo di MusicMix.

Il caso MusTBench dimostra anche il bisogno di fissare gli artefatti: tabella del paper e README riportano conteggi di tracce differenti. Una riproduzione deve dichiarare quale file/commit usa. I link a pesi o dati “forthcoming/TBD” non si contano come componenti già eseguibili.

## Protocollo automatico minimo proposto

1. Congelare sorgenti, hash, artisti e split prima di scegliere modello/soglie. Tenere i giudizi V8 come diagnostica storica, non come training set del nuovo critico.
2. Riprodurre il compito cue su dati pertinenti, con metrica/tolleranze originali e vincoli di copertura; valutare separatamente il trasferimento a rap/pop.
3. Validare il timing vocale su annotazioni esterne, includendo parole perse e code d’errore; mai sostituire probabilità token a accuratezza del confine.
4. Misurare qualità tecnica con difetti costruiti e musicalità con dati esterni di preferenza; mantenere i target distinti.
5. Valutare il planner con alternative “resta”, diversi ingressi e durate, misurando rischio/copertura e numero di cambi ammessi. Un sistema che si astiene sempre non supera il protocollo.
6. Dopo questi controlli, eseguire il ciclo in autonomia senza richieste di ascolto ricorrenti. Dichiarare gli ambiti non coperti anziché produrre una confidenza senza dati.

Il pool attuale di quattro brani serve al debug e a verifiche di formato; non basta per addestrare e validare una pretesa SOTA. Questo dossier non fornisce ancora un benchmark pubblico condiviso di “suspense risolta in un raccordo rap”: nelle fonti esaminate quel target resta da specificare e verificare.
