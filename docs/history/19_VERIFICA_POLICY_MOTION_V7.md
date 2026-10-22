# V7 — policy di verifica per aspettativa, continuazione, chiusura e interruzione

Documento di revisione critica per l'esperimento V7. Ownership: ricerca
motion/suspense. Non modifica codice né il ranking stabile. Documenta invece
il pilot reranking esplicitamente autorizzato, con peso congelato, confronto
contro baseline e output ascoltabile. La policy va applicata alla versione
effettiva di expectation.py, alla pipeline di selezione e al cache prodotto
da audio_features.

## Scopo e stato

V7 prova a separare tre evidenze:

- continuation: B sviluppa il moto o la struttura osservata prima della
  giunzione;
- closure_candidate: B presenta indizi indipendenti di arrivo/chiusura;
- interruption: A aveva aperto un'aspettativa, B devia o la abbandona e non
  presenta una chiusura sufficiente.

Queste sono evidenze osservabili, non labels percettive validate. In
particolare, closure_candidate non deve essere serializzato come resolution o
release senza una successiva validazione umana. Il caso annotato Reminder 213,090 → Too Many
Nights 182,200 resta holdout: non può determinare soglie, pesi, normalizzazioni,
ammissione di candidati o scelta di feature.

La policy è deliberatamente più severa della sola traiettoria. Un calo di
energia o di un proxy di tensione può accompagnare un release, ma può anche
essere silenzio, fade, abbandono dell'arco o cambio di ruolo. L'energia non è
una prova di chiusura.

## Revisione dell'implementazione presente

La revisione statica del 18-10-2026 riguarda expectation.py,
expectation_pipeline.py e cache_candidate_trajectories.py. Nel workspace il
punto di integrazione verificabile è la pipeline controllata; questa policy
non considera quindi verificato un modulo candidate_search separato.

### Rischio residuo: metric alignment non è un arrivo

In expectation.py, _metric_evidence calcola la distanza fra cue_sample e
beat_anchor_sample e assegna score 1.0 quando coincidono. Un cue scelto dalla
griglia dei downbeat è per costruzione vicino a un beat; questo certifica
allineamento temporale, non che B abbia chiuso una frase, risolto una tensione
o ristabilito un groove. Nella funzione _arrival_evidence questo valore viene
poi sommato alle altre contribuzioni. La versione corrente richiede inoltre
un provider esplicito di arrivo/struttura, quindi ha corretto il falso
positivo più grave; tuttavia metric alignment resta una condizione e una
contribuzione della closure.

Il campo va interpretato e serializzato come beat_alignment, non come
arrival/closure. La chiusura corrente può essere un candidato soltanto perché
è presente anche structural_arrival con provider; il beat da solo non basta.
L'arrivo deve derivare da un boundary o da un evento post-entry osservabile
rispetto a una frase/struttura dichiarata. Se il provider fornisce soltanto
beat, il risultato deve restare un candidato ritmico, non una risoluzione.

### Rischio alto: min_independent_domains non è applicato

ExpectationThresholds impone min_trend_domains almeno 2, ma non definisce né
applica min_independent_domains per closure. _arrival_evidence calcola
independent_domain_count soltanto dentro diagnostics; la decisione effettiva
usa invece metric_status supportato, una lista event_domains non vuota,
stability_status supportato e structural_status supportato. Quindi il
conteggio delle famiglie non è ancora un vincolo esplicito.

Inoltre DOMAIN_NAMES tratta energy, events, bass, timbre e tonal_proxy come
domini separati, mentre trajectory-v1 li ricava tutti dalla stessa analisi
FFT. bass_entry e event_entry possono soddisfare due contributi pur essendo
due letture dello stesso ingresso acustico; la stabilità calcolata sui domini
piatti può contare ancora la stessa famiglia. Prima di usare closure_candidate
occorre raggruppare le feature in famiglie indipendenti e verificare un
minimo esplicito. La raccomandazione di questa policy resta almeno due
famiglie, con una famiglia di frase/metro, eventi/ritmo o tonalità/stabilità.
Nel pilot attuale questa è una limitazione da riportare, non un motivo per
inventare confidence o per promuovere automaticamente il risultato al ranking
stabile. Il valore e le fonti devono comunque comparire nel manifest quando
si introducono feature indipendenti.

### Correzione applicata: eleggibilità della closure richiede A

La versione corrente calcola la closure evidence su B, ma applica poi
closure_eligible = closure.status == supported and A_supported prima di
assegnare closure_candidate. Un B con ingresso e stabilizzazione senza A
osservata può quindi conservare un'evidence di arrivo B, ma il label finale
resta unknown. Questo è il confine corretto fra arrival evidence e chiusura
dell'aspettativa di A.

Il report deve mostrare esplicitamente A_before_supported e non deve
promuovere un arrivo di B a “risolve” quando A non è osservabile. Il pilot può
comunque conservare la evidence B-side per audit, purché non venga letta come
label percettiva.

### Rischio medio-alto: aggregazione fra finestre

_summarize calcola per ogni dominio la media dei punteggi sulle finestre 8,
16 e 32 beat e usa quella media per decidere up/down/flat. Al contrario
_arrival_evidence esamina soltanto la finestra con il numero minimo di beat.
Una sola scala può quindi cambiare la direzione aggregata, mentre la closure
usa un'altra scala senza un conteggio di accordo fra finestre. Questo non è
un problema di causalità se le finestre sono tutte dentro l'orizzonte, ma è
un problema di interpretazione.

Nel pilot il report deve conservare il dettaglio per finestra e dichiarare
che la decisione è multiscala euristica. Prima di usare il segnale nel ranking
stabile occorre scegliere e congelare una policy di accordo (per esempio
finestra primaria più una finestra di conferma), senza scegliere la policy
guardando il caso holdout. Un trend presente in una sola finestra non va
descritto come consenso multiscala; una finestra assente o vuota deve restare
unknown, non zero.

### Rischio medio-alto: availability e confidence del provider

La traiettoria dichiara confidence null e confidence_status
unknown_unvalidated, scelta corretta per un baseline non calibrato. La
versione corrente richiede per la closure un provider non vuoto per
structural_arrival, ma non possiede ancora una calibrazione percettiva del
provider. Il valore chroma_local_context_similarity è un proxy di stabilità,
non una confidence; la soglia 0.62 non lo rende una cadenza. Nella
traiettoria, la confidence dei beat passata come mapping viene
intenzionalmente ignorata e beat_provider_confidence_used resta false.

Questa combinazione deve rimanere marcata come evidence unvalidated. Non va
inventata una confidence numerica, ma va impedito che supported venga letto
come affidabile: qui significa evidence disponibile secondo il contratto del
provider, non qualità musicale validata. Il cache deve conservare la
confidence del beat/downbeat, lo stato del provider e il motivo per cui non
viene usata; la copia di un downbeat non può aumentare la confidence.

### Rischio medio: cache e identità della struttura

cache_candidate_trajectories.py legge gli eventi beats dei file di struttura,
mentre i cue delle proposte provengono dai downbeats. Il profilo viene quindi
ancorato al beat più vicino, senza un downbeat esplicito con confidence e
provenance nel profilo. Questo può essere utile per una finestra ritmica, ma
non soddisfa da solo una policy di arrivo downbeat/frasale.

La profile_cache_key attuale contiene track_id, lato, cue originale e shift;
il report contiene un config hash, ma la chiave non contiene esplicitamente
hash del provider beat/downbeat, hash della struttura o versione del
provider. Per un cache riusato tra run questo permette risultati stale.
Aggiungere questi elementi alla chiave o vietare il riuso fra run con
provenance diversa.

Il report dichiara cache_policy all admitted proposals, ma registra soprattutto
proposals_in_search e candidate_records. Non espone ancora in modo separato
generated_count, admitted_count, cached_count e motivi per ogni proposta
saltata. Le proposte malformate vengono ignorate in alcuni rami del loop;
questa perdita deve diventare un rejection record. --limit è correttamente
marcato partial_smoke_limit e non va usato come prova di copertura completa.

Nel file candidate_trajectories_v7.json attualmente presente il cache completo
contiene 284 candidate records e 142 coppie direzionali, pari a 110 proposte
forward più 174 reverse, 39 profili unici, 39 miss e 813 hit; non risultano
mismatch di allineamento. Il file candidate_trajectories_v7_allbeat.json.gz
contiene 4512 records e 2256 coppie direzionali. Tutti i 39 profili
del cache downbeat hanno confidence_status unknown_unvalidated e
beat_provider_confidence_used false, e uno ha zero finestre locali. Questo
rende i run utili come cache descrittivi, ma non come validazione della
confidence. I conteggi del manifest devono riferirsi al run effettivamente
usato e non a un esempio precedente.

Auditando i 284 records con expectation.py corrente, si ottengono 198 label
unknown e 86 continuation, con 284 closure_arrival unknown e nessun
closure_candidate o interruption. La causa principale è la mancanza del
provider phrase_or_structural_arrival. Questo è il comportamento conservativo
atteso dal fix: gli unknown restano neutrali nel rerank; non sono prova che la
policy abbia già risolto l'aspetto percettivo.

### Causalità e riferimento naturale

La costruzione di A_natural_after come profilo incoming sulla stessa cue di A
è coerente con il controllo offline definito nella ricerca. È però il futuro
reale di A e non può entrare in uno score dichiarato online-causale. Il runner
corrente dichiara analysis_mode=offline_reference e specifica che il
contributo è soltanto la direzione A_natural_after; questo va mantenuto
separato da una futura modalità live.

### Limite documentato: il riferimento naturale entra nello score offline

Nella versione corrente analyze_expectation aggiunge
natural_reference_direction a continuation_contributions. Quando la
continuation è supported, la media di quelle contribuzioni diventa
continuation.score e quindi, con il peso .20 del runner, entra
nell'adjusted_score. È coerente con il pilot corrente perché la modalità è
esplicitamente offline_reference; non è un modello causale live e non va
presentato come tale. Se in futuro si riusa il punteggio al cue, la direzione
natural_reference va mantenuta nei diagnostics ma esclusa dallo score/rerank.

Il profilo naturale con oggetto presente ma zero finestre va inoltre marcato
come unavailable/unknown, non come reference disponibile: l'esistenza della
chiave non prova che il futuro osservativo sia stato calcolato.

### Runner controllato: vincoli e modalità verificati

La pipeline corrente mantiene lo stesso cue-out, richiede almeno 8 secondi di
separazione fra gli ingressi policy/baseline, usa novelty di 8 secondi su
entrambe le sorgenti, spacing di 12 secondi e massimo 3 prove per direzione.
Il peso è .20 e i tie/alternative vicine vengono saltati. Questi vincoli
rendono il confronto ascoltabile, ma non trasformano il delta in una metrica
validata.

Il default di tools/run_expectation_experiment.py punta ora a
candidate_trajectories_v7_allbeat.json.gz, il cache all-beat completo da
4512 records / 2256 coppie; il cache downbeat
candidate_trajectories_v7.json resta disponibile per l'audit separato. Il
file smoke non va usato come copertura completa. Il report serializza già
analysis_mode e natural_future_reference; salva evidence e missing, ma non
l'intero diagnostics per ogni candidato, quindi la provenienza dettagliata
resta da controllare nel cache e nei file di codice hashati.

### Copertura dei test attuali

I test di test_expectation.py e test_expectation_pipeline.py passano con
PYTHONPATH=src, ma non coprono:

- cue/beat alignment uguale a 1 con assenza di un vero arrival;
- due contributi della stessa famiglia FFT che soddisfano il conteggio;
- beat/downbeat confidence mancante o provider non verificato;
- A_before assente con evidence B-side supported e label finale unknown;
- cache downbeat 284 records / 142 coppie e all-beat 4512 records / 2256
  coppie con uguaglianza degli insiemi;
- key stale fra provider/config diversi;
- leakage del holdout e orizzonte B_after oltre la finestra dichiarata.

Questi casi sono gate per interpretare V7 come policy semantica o per
promuoverla al ranking stabile. Il pilot reranking può essere eseguito prima,
con peso congelato, baseline affiancata, labels raccolte senza adattamento e
tutti gli unknown conservati nel report.

## Regole non negoziabili

### 1. Causalità e finestre osservate

Ogni output deve riportare in modo verificabile:

- cue A e cue B in campioni e secondi, sample rate e identificativi SHA;
- lato (A_before, B_after) e intervallo di beat/campioni usato;
- orizzonte di osservazione dopo l'ingresso B;
- versione del provider beat/downbeat, configurazione e hash del cache;
- se il punteggio è online-causale oppure offline-diagnostico.

Il punteggio di un candidato deve usare soltanto il contesto previsto dalla
modalità dichiarata. In modalità causale, B_after non può leggere eventi
successivi all'orizzonte dichiarato e non può usare la continuazione futura di
A come se fosse già stata ascoltata. In modalità offline è ammesso salvare
A_natural_after come riferimento osservativo, ma deve restare un campo
diagnostico separato: non è ground truth, non è una label e non deve diventare
un requisito per cui B deve copiare il futuro di A.

Una cache che contiene l'intera traccia non prova causalità. La prova è la
finestra effettivamente indicizzata nel record del candidato. Una funzione
che calcola prima profili completi e poi ritaglia la finestra è accettabile
solo se il profilo non ha normalizzazioni o filtri che dipendono da campioni
futuri; altrimenti il record deve essere marcato offline.

### 2. Domini indipendenti

Il conteggio di evidenze deve avvenire per famiglie, non per il numero di
colonne. Sono famiglie distinte, quando realmente disponibili:

1. energia/intensità;
2. eventi/ritmo/roughness;
3. frase/metro/downbeat/novelty;
4. tonalità/attrazione/stabilità;
5. ruolo/timbro/vocale.

RMS, log-RMS e pendenza RMS sono una sola famiglia. Chroma similarity e
tonal-stability derivate dalla stessa stima sono una sola famiglia. Due
finestre della stessa feature non sono due domini indipendenti.

min_independent_domains deve essere esplicito nel manifest quando si parla di
famiglie indipendenti. La raccomandazione iniziale per una futura policy
semantica è almeno 2 famiglie, di cui almeno una fra frase/metro, eventi/ritmo
o tonalità/stabilità; è una policy prudenziale da verificare sugli ascolti,
non una metrica pubblicata. Nel pilot corrente il conteggio resta diagnostico
e la mancanza di una famiglia indipendente va marcata come limitazione, non
trasformata in confidence o usata per far fallire la raccolta di confronti.

La presenza di due famiglie non dimostra la validità percettiva: impedisce
solo che una singola misura venga promossa a semantica.

### 3. Closure separata da energia

Nel pilot closure_candidate può essere emessa soltanto se:

- esiste un segnale di arrivo strutturale, metrico/frasale, tonale o di
  evento ritmico con provenienza dichiarata;
- il provider è disponibile e la finestra è sufficiente; l'assenza di
  calibrazione va marcata unknown_unvalidated;
- il conteggio delle famiglie indipendenti è riportato, senza chiamarlo
  validazione;
- non è semplicemente energy_after < energy_before.

Un calo di RMS/log-RMS, un calo di tension_proxy, un silenzio, un fade o una
similarità cromatica locale non bastano individualmente. Anche
energy_fall + chroma_similarity non è una cadenza: la prima è intensità e la
seconda somiglianza tonale locale.

Per EDM/DnB un drop può essere un indizio di release solo quando il build-up,
la posizione metrica e il ristabilirsi/trasformarsi degli eventi sono
osservabili. In pop/rap la chiusura può essere frase vocale, risposta
ritmica, downbeat, stabilizzazione o pausa strutturale. Non si deve usare il
genere rap come penalità.

### 4. Continuation e flat → flat

Una continuità alta non può risultare da due pendenze vicine a zero e da una
sola somiglianza di livello. flat → flat deve essere neutral_flat o unknown
finché non esiste un'evidenza separata di frase, groove/ritmo, ruolo o
sviluppo. Se le feature di moto sono tutte piatte, una similarità
orizzontale può descrivere compatibilità statica, non continuazione
aspettativa.

La continuazione può essere alta anche con energia non identica, ma deve
mostrare sviluppo o struttura coerente. Non va confusa con “B ha lo stesso
RMS” né con “B è vicino in chroma”. Il record deve conservare i motivi per
cui la continuità è stata accettata e quali domini hanno contribuito.

### 5. Interruption e assenza di arrivo

interruption richiede almeno:

- evidenza osservabile di moto/aspettativa in A_before;
- divergenza o mancato sviluppo in B_after;
- assenza di closure sufficiente secondo le regole precedenti.

Un segmento calmo o piatto non è automaticamente interruzione: può essere una
risoluzione, un cambio intenzionale o un contrasto comprensibile. Se A non
mostra preparazione e B è piatto, il risultato deve essere unknown o neutral,
non un falso negativo.

### 6. Confidence: solo provenance, mai riempitivo

Una confidence numerica deve avere un provider e una definizione. Non è
ammesso impostare 1.0 perché un valore è finito, perché un array è non vuoto
o perché il fallback FFT ha prodotto un numero. Un descriptor calcolato non è
per questo validato.

Ogni confidence deve indicare almeno:

- provider e versione;
- input e qualità/completezza;
- eventuale calibrazione o, se assente, il nome heuristic_confidence;
- aggregazione fra beat e domini;
- reason code per missing, partial, unknown e fallback.

Se la confidence del provider non esiste, usare null e propagare unknown,
oppure un campo separato availability senza chiamarlo confidence. Non
sostituire i missing con zero e non usare min/mean come calibrazione
implicita senza documentarlo. Una confidence geometrica della finestra può
attestare disponibilità, non qualità musicale.

## Policy sul cache audio_features

Il cache deve essere costruito per tutti i candidati ammessi dal run, prima
di qualsiasi top-K o diversificazione. Il manifest deve permettere il
controllo:

- candidate_generated_count;
- candidate_admitted_count;
- feature_cache_candidate_count;
- insieme degli ID ammessi e insieme degli ID cachati;
- motivo di ogni rifiuto;
- hit/miss/invalid e versione del feature provider.

Il requisito è uguaglianza degli insiemi, non solo dei conteggi. Per il cache
downbeat corrente il controllo atteso è 284 records / 142 coppie direzionali;
per candidate_trajectories_v7_allbeat.json.gz è 4512 records / 2256 coppie. Il
manifest deve
distinguere proposte generate, ammesse e cachate e registrare ogni esclusione.
Non è accettabile chiamare “all candidates” una cache che ha saltato candidati
per ranking o errori silenziosi.

Ogni chiave di cache deve includere, almeno, SHA audio A/B, sample rate,
cue samples, direzione, beat/downbeat provider e versione, finestre,
orizzonte, config hash e modalità causale/offline. Un cache condiviso fra
cues o tracce con chiave incompleta può produrre leakage o descrittori
spostati di un beat.

### Copia dei downbeat

Un downbeat copiato da una struttura o da un provider deve conservare
sample-origin, indice del beat, confidence e versione del provider. Copiarlo
nel record candidato non aumenta la confidence e non crea un secondo dominio
indipendente. Se A e B hanno downbeat su timeline diverse, la copia deve
conservare l'identità della traccia; non riusare l'indice di A come indice
assoluto di B.

La policy deve distinguere:

- downbeat osservato dal provider;
- downbeat ereditato/copiato nel cache;
- boundary inferita per fallback;
- boundary mancante.

Il fallback non può essere serializzato come downbeat osservato senza un
reason code.

## Holdout e leakage

Il caso annotato holdout è la direzione Reminder 213.09 → Too Many Nights
182.2, dove l'utente descrive A crescente/suspense e B rap piatto che non
continua né risolve. La coppia di riferimento positiva disponibile è
Reminder 192.1 → Too Many Nights 16.42. Questi esempi non rappresentano
automaticamente l'intera direzione e non si deve trasformare il caso in una
regola sul rap. Per valutazioni di generalizzazione si può poi separare a
livello di brano/artista. Non si devono stimare sul holdout:

- soglie di continuation/closure/interruption;
- min_independent_domains;
- pesi o coefficienti;
- normalizzazioni per traccia;
- scelta fra finestre 4/8/16/32;
- mapping delle confidence;
- ammissione/rifiuto o top-K.

Se i profili di holdout sono calcolati nello stesso run per un audit, questo
è lecito solo come inferenza congelata. Il manifest deve distinguere
holdout_observed_for_report da used_for_fit = false. Una cache non è training
leakage di per sé, ma diventa leakage se viene usata per cambiare la policy
dopo aver visto il giudizio dell'utente.

Il caso deve restare rappresentato nel QA con l'aspettativa “A crescente →
B piatto non continua e non risolve”; non deve diventare una regola
“rap = negativo”.

## Output obbligatorio e stati ammessi

Ogni candidatura deve produrre, anche se rifiutata:

- candidate_id, track_id, cue, beat/downbeat e config hash;
- profilo A_before e B_after con orizzonte;
- valori per dominio, availability e provenance;
- elenco dei domini indipendenti conteggiati;
- continuation_evidence, closure_candidate,
  interruption_evidence;
- stato finale secondo il contratto corrente: continuation,
  closure_candidate, interruption oppure unknown;
- reason codes leggibili;
- indicazione esplicita che nessuno stato equivale a una label umana
  validata.

La parola resolution va riservata alle annotazioni o a un livello successivo
validato. Una funzione che restituisce release soltanto perché energia o
proxy cala viola questa policy.

## QA minimo prima di considerare V7 leggibile

Servono fixture o audit deterministici per almeno questi casi:

1. A crescente → B piatto: non closure_candidate; candidato a interruption
   solo se l'aspettativa e il mismatch sono misurati;
2. A crescente → drop su downbeat con eventi ristabiliti: possibile
   closure_candidate con almeno due domini;
3. A crescente → silenzio/fade: non closure automatica;
4. flat → flat: non continuation alta da soli livelli o chroma;
5. chroma simile + energy fall senza frase/ritmo: non closure;
6. tonalità o downbeat missing: closure/arrival unknown, non confidence
   inventata; la continuation resta ammessa solo se il profilo osservato la
   supporta senza usare il campo mancante;
7. candidato con B_after che supera l'orizzonte: rifiuto o marcatura
   non-causale;
8. duplicate audio/cue o cache stale: reason code e nessun doppio contributo;
9. cache downbeat con 284 records / 142 coppie e cache all-beat con 4512
   records / 2256 coppie: uguaglianza degli ID ammessi e cachati, senza
   conteggio nascosto;
10. holdout visto nel report ma con used_for_fit = false.

La QA deve controllare anche che una feature mancante non venga trasformata in
zero, che due colonne della stessa famiglia non soddisfino da sole
min_independent_domains e che la copia del downbeat non alteri sample rate,
track ID o provenance.

## Giudizio e prossimi passi

La policy consente l'esperimento V7 come pilot diagnostico: peso .20
congelato, baseline affiancata, massimo 3 confronti per direzione, spacing e
novelty già dichiarati nel runner. Il pilot può cambiare il vincitore del
confronto per raccogliere ascolti; non autorizza a chiamare il risultato
“risoluzione” né a sostituire il ranking stabile prima delle labels umane.

Ordine consigliato:

1. congelare run/config/provider e verificare l'insieme completo
   generated/admitted/cached;
2. auditare causalità, downbeat provenance e confidence;
3. eseguire le fixture flat→flat, energy-fall e build-up→drop;
4. mantenere il caso annotato 213,090 → 182,200 come holdout non adattato;
5. raccogliere labels continua, risolve, interrompe,
   contrasto_intenzionale, unknown insieme al pilot e confrontarle con la
   baseline; usare queste labels per decidere soltanto in seguito se
   promuovere la policy al ranking stabile.
