# Critici automatici del raccordo: modelli, prova necessaria e fattibilità

Il subagente ha completato ricerca, letture selezionate e registro di18fonti, poi ha esaurito la quota prima della sintesi. L’orchestratore ha verificato direttamente i lavori e rilasci prioritari e scritto questo rapporto, aggiungendo MusTBench. La lettura di ogni fonte è delimitata in `sources.json`; il root ha letto soprattutto sezioni/abstract dei nuovi PDF, schede ufficiali e codice di inferenza, non tutte le appendici di ogni lavoro. Nessuno di questi nuovi modelli è stato installato o eseguito in V9. Audiobox già eseguito appartiene alla V8.

## Risultato pratico

Esistono ora critici musicali aperti più pertinenti di un solo indicatore di clipping. Nessuna delle prove esaminate dimostra che un checkpoint possa essere usato direttamente come giudice validato di “A costruisce suspense e B la risolve” nelle nostre registrazioni. Questo è un gap di task e dominio, non una ragione per rinunciare all’autonomia: serve una metavalutazione automatica prima di ottimizzarne il reward.

## Candidati prioritari

| Modello | Che cosa predice | Risorsa verificata | Come lo userei qui |
|---|---|---|---|
| CMI-RM (S15) | Musicalità e aderenza composizionale a testo/lyrics/audio | Codice,configYAML,pesiHF; CPU previsto nell’adattatore | Primo pilot di critic indipendente con contesto esplicito, poi test di raccordo |
| SongEval (S13) | Cinque aspetti estetici di canzoni generate | Benchmark e lavoro con modelli su rating esperti | Asse coerenza/fraseggio da confrontare senza dedurne un confine esatto |
| MuseCritic (S17) | Critica testuale intermedia più cinque reward | Repo e checkpoint HF verificati, inferenza documentata CUDA | Candidato semantico più costoso, da verificare per sensibilità al raccordo |
| TuneJury (S16) | Preferenza appresa pairwise, output per-clip | Head2.8M più CLAP/MERT congelati; licenzaNC | Baseline di preferenza soprattutto strumentale, con gap vocale esplicito |
| Music Flamingo (S12) | Comprensione/QA musicale con contesto lungo | Schede ufficiali,8B, uso non commerciale | Analisi audio con timestamp da verificare, non reward già calibrato |
| Audiobox | Quattro assi di qualità/contenuto | Già eseguito localmente in V8 | Diagnostica di produzione, non accettazione del motion |

Le quantità di parametri della testa non sono la memoria di inferenza: CMI usa anche MuQ-MuLan, TuneJury anche encoder pesanti. Il README TuneJury indica circa2.2GB per il solo encoderCLAP e segnala accesso a MERT; i pesi principali sonoCC-BY-NC4.0. MuseCritic ha scheda pesiApache-2.0 e istruzioniCUDA: licenze e dipendenze upstream vanno controllate separatamente. La macchina espone IntelUHD620, nessuna NVIDIA rilevata; non considero le pipeline GPU da7–8B già praticabili localmente in modo leggero.

## CMI-RM: perché vale un pilot e perché non è pronto

S15 combina preferenze esterne e pseudolabel e distingue musicalità da instruction alignment. Nel codice ufficiale ispezionato il campionamento è24kHz e max_dur30s di default. La modalità final tronca prima dei chunk; standard aggrega finestre. Occorre quindi consegnare al modello un estratto che includa la giunzione e tutta la preparazione pertinente, non l’inizio della canzone lasciando fuori il cambio. La sua testa da circa30M addestrabili non include il costo del backbone congelato.

Il benchmark misura generazioni e istruzioni composte: la presenza di audio di riferimento è promettente, ma non equivale a un target di arrivo musicale. Il pilot deve confrontare ingresso coerente, ingresso traslato e ordine invertito con la stessa qualità tecnica. Non assegnare una soglia dal nostro trial preferito e non passare il punteggio del selettore nel prompt del giudice.

## MuseCritic e TuneJury: due idee diverse

S17 usa una critica intermedia e reward su coerenza, musicalità, memorabilità, chiarezza strutturale e naturalezza vocale. Il suo miglioramento sul benchmark proprio e il confronto out-of-domain sono rilevanti per scegliere un candidato da provare, non un’accuratezza attesa sui nostri otto tagli. La spiegazione testuale deve essere ancorata all’audio e non è prova per il solo fatto di essere convincente. Repo e scheda pesi sono stati verificati dopo la ricerca; non è stato scaricato il checkpoint.

S16 apprende una preferenza relativa da rating esterni e può funzionare senza feedback utente all’inferenza. La scheda dichiara musica strumentale: i word-cut e l’aspettativa di una voce rap sono un trasferimento non provato. La calibrazione per nuovi generatori descritta dal lavoro usa ulteriori coppie di calibrazione, quindi non è una garanzia senza dati sul nostro dominio. Un prompt vuoto è una baseline pertinente per input fuori dal formato originale, non una soluzione provata alla continuità.

## Un giudice può rispondere senza ascoltare abbastanza

S04/BASS valuta struttura e ragionamento, S07/HumMusQA comprensione con domande umane, S08 factual comprehension e S09 contributo delle modalità. I lavori esaminati mostrano che un buon risultato linguistico o una descrizione plausibile possono nascondere dipendenza da priors testuali. Il root ha letto direttamente abstract/introduzioni di S04,S08,S09 e la schedaS07; le ulteriori letture del subagente sono archiviate. Non generalizziamo numeri di questi benchmark al nostro compito.

S19 richiede invece di collocare cambi e regioni nel tempo, evidenziando errori e timestamp fuori durata anche in modelli capaci di descrivere la musica. È un controllo essenziale sul giudice. Hit@3s, tuttavia, è una metrica di quel task: non certifica che una sillaba non sia tagliata di50ms. Il rilascio benchmark/README non dimostra la disponibilità dei pesi del modello MusT.

## Metriche locali e riferimenti

S02/LoopGen valuta la probabilità di token vicino alla cucitura di un loop. È una pista interessante per valutare sequenze al raccordo, ma le sue prove riguardano looping e un modello generativo specifico. Unire due brani di artisti diversi è un trasferimento da dimostrare. S03 studia l’accordo fra metriche di boundary e ascolto su SALAMI, con10esperti: un risultato di metavalutazione utile ma limitato, non una nuova legge di qualità.

CLAP/FAD o una distanza globale non diventano automaticamente un critico del punto di cambio. FAD è soprattutto una misura di distribuzione; un embedding medio può diluire proprio la regione che ci interessa. Anche un buon reward per-clip deve superare una verifica di sensibilità al raccordo e una di stabilità alle trasformazioni irrilevanti.

## Metavalutazione proposta prima del loop autonomo

- **Vincoli tecnici:** clipping/click/silenzio con corruzione nota; non etichette inventate di “brutta musica”.
- **Ancoraggio:** il giudice localizza i cambi dentro la durata e cambia risposta se l’audio pertinente viene sostituito o l’ordine modificato.
- **Contesto:** confronto delle stesse giunzioni con contesto breve/lungo; registrare i casi in cui il giudizio cambia o il raccordo viene troncato dall’input.
- **Identità e produzione:** livello, sorgente e artista bilanciati; il critico non deve vincere soltanto riconoscendo mastering e cantante.
- **Trasferimento:** benchmark esterno con esempi cross-song appropriati e split per registrazione, separato dalla selezione dei pesi.
- **Policy:** misurare errore e copertura dell’accettazione, includendo restare in A e l’astensione; vietare il successo fittizio ottenuto rifiutando tutto.

L’output utile è un fascicolo per evento: evidenza strutturale/vocale, stato delle metriche, giudizi indipendenti, disaccordo e motivazione di accettazione/rinvio. Il target operativo resta non chiedere nuovi voti all’utente a ogni iterazione, mantenendo chiaro ciò che è stato misurato e ciò che è ancora sconosciuto.
