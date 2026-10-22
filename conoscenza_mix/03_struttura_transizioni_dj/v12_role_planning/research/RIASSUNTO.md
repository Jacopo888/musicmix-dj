# Riassunto ricerca: opportunità role-aware e piani di transizione

Fonti archiviate e metadati verificabili sono in `research/sources.json`; i PDF
originali sono in `research/originals/` e il testo estratto con ancore di pagina
in `research/texts/`. Questo riassunto distingue **evidenza misurata dagli
autori**, **teoria/linee guida**, **ipotesi ingegneristiche nostre** e **limiti**.
Non è una validazione del selettore né un giudizio musicale.

## 1. Cosa dicono le fonti primarie

### Kim et al. 2020 (`kim_2020`)
- **Definizioni operative (teoria/descrizione del dominio):** cue-out è il punto
  in cui il brano precedente inizia a sfumare e il successivo inizia a entrare;
  cue-in è quando il precedente è completamente uscito e suona solo il
  successivo. La *transition region* va dal cue-out del precedente al cue-in del
  successivo; il cue-mid è il punto medio.
- **Evidenza misurata:** gli autori analizzano 1.557 mix, 13.728 tracce e 20.765
  transizioni con allineamento mix-to-track a feature beat-sync (chroma/MFCC) e
  DTW a sottosequenza. Riportano anche disaccordo umano sui confini e statistiche
  su durata delle transizioni e accordo tra DJ.
- **Conseguenza per noi:** non basta un solo timestamp. Uscita, pickup/ingresso e
  arrivo sono momenti distinti; la regione di transizione può essere lunga e
  incerta. Non deduciamo un livello di fade o una qualità da un cue.

### M-DJCUE 2019 (`mdjcue_2019`)
- **Teoria/linee guida:** i cue point sono soggettivi e dipendono da gusto,
  stile del DJ e caratteristiche del brano. Le annotazioni sono state filtrate
  per accordo tra annotatori.
- **Regole dichiarate:** cue-in nell'intro fino al primo beat della sezione core;
  cue-out ovunque ma non prima del core; un cue-out può avere una durata se
  seguito da un fade-out. Il punto deve essere a inizio battuta, preciso a 0,1 s,
  e non interrompere elementi forti come basso o voce.
- **Limite:** è un extended abstract non referato, con 135 tracce EDM e label
  soggettive. Non è una ground truth universale né una prova che le nostre soglie
  siano corrette.

### Zehren et al. 2020 (`cuepoints_2020`)
- **Teoria/regole derivate da interviste a DJ:** Rule 1, alto contrasto di
  novelty (ritmo, loudness, timbro, armonia); Rule 2, lo switch point cade sul
  downbeat all'inizio di un periodo; Rule 3, la sezione dopo lo switch deve
  reggersi da sola nel mix.
- **Evidenza misurata:** gli autori valutano il loro detector rule-based su un
  dataset EDM annotato e con ascolto; riportano circa il 96% di punti giudicati
  di buona qualità. È un risultato del loro sistema e del loro protocollo, non
  del nostro codice.
- **Attenzione:** il paper distingue pre-switch, switch e post-switch. Il lavoro
  correlato cita la prevenzione dei tagli sulle voci tramite struttura di frase;
  non è una nuova misura di questo articolo.
- **Licenza:** non è dichiarata nel testo estratto; non assumere CC BY.

### CUE-DETR 2024 (`cuedetr_2024`)
- **Evidenza misurata:** EDM-CUE è riportato con 4.710 tracce e 21k cue point
  annotati da quattro DJ professionisti di club, oltre a tempo, beat e downbeat.
  Il modello è un DETR fine-tuned su Mel-spettrogrammi.
- **Phrasing:** i cue point spesso coincidono con confini di frase; dominano
  frasi di 16 e 8 bar, con sezioni irregolari. Gli autori aggiungono punti
  phrase-aligned per valutare le previsioni.
- **Limite dichiarato dagli autori:** annotazioni di altri tipi di DJ potrebbero
  concentrare più cue point attorno a voci o pickup invece che sul primo downbeat
  della frase.
- **Conseguenza per noi:** un downbeat o un cue di un DJ club non è automaticamente
  un “arrivo” universale; pickup e protezione delle parole restano ipotesi
  separati e vanno verificati sui dati reali.

### VandeVeire & De Bie 2018 (`vandeveire_2018`)
- **Solo metadati:** DOI `10.1186/s13636-018-0134-8`, licenza CC BY secondo
  OpenAlex. Il PDF del publisher ha restituito una pagina HTML di challenge
  Cloudflare/HTTP 401, quindi non è stato letto né usato per affermazioni.
  L'artefatto non valido è conservato per audit.

## 2. Sintesi: cosa è solido e cosa resta ipotesi

**Più solido (più fonti concordano o è misura diretta):**
- I cue point hanno un ruolo direzionale (uscita vs ingresso) e non sono
  intercambiabili.
- Lo switch/ingresso è tipicamente legato alla griglia di battute/downbeat, ma
  la griglia non dimostra da sola una chiusura di frase o una buona transizione.
- Le annotazioni umane sono soggettive e possono non coprire tutti i punti validi.
- Un cue-out può descrivere l'inizio dell'uscita, mentre l'arrivo pieno del nuovo
  brano è un evento successivo: la regione di transizione è un intervallo.

**Teoria/linee guida, non misure nostre:**
- “La sezione dopo lo switch deve reggersi da sola” e “non interrompere voce o
  basso” sono criteri progettuali utili, ma non sono implementati come detector
  di salienza nel nostro modulo.
- La protezione delle parole tramite intervalli ASR è una nostra scelta
  ingegneristica; nessuna fonte qui prova che la nostra soglia di 80 ms sia
  ottimale.

**Ipotesi ingegneristiche nostre (da validare sui sei brani reali):**
- Un pickup può iniziare a un onset di parola poco prima dell'arrivo strutturale;
  l'arrivo resta separato e il ritardo è esplicito.
- Un evento di completamento vocale nominato può sostenere l'uscita, ma non è
  equivalente a una chiusura di frase.
- Un piano utile deve stare in una finestra temporale plausibile (qui al massimo
  8 s tra cambio, pickup e arrivo); questa soglia è una scelta di prodotto, non
  un risultato sperimentale.
- Diversificare tracce e regioni tramite cap e spaziatura evita la ripetizione
  degli stessi punti, ma non misura la qualità musicale.

## 3. Come la ricerca è collegata al codice

| Fonte | Uso implementato | Cosa non viene affermato |
|---|---|---|
| Kim 2020 | `RoleOpportunity.pickup_start_sample`, `arrival_sample`, `CandidatePlan.entry_window_*`: uscita, pickup e arrivo distinti | Nessun livello di fade o qualità percepita |
| M-DJCUE 2019 | `completion_supported`, `phrase_closure_supported`, `direction_evidence`: label direzionale, non prova di chiusura | Nessuna corrispondenza con le label del dataset |
| Zehren 2020 | downbeat come contesto; `outgoing_completion_supported` richiede un evento nominato; nessun punteggio di qualità inventato | Non si afferma che i piani rispettino la salienza del paper |
| CUE-DETR 2024 | `_word_pickup`, `arrival_word_status`, guardia parole: pickup/parola separati dal downbeat generico | Non si afferma di riprodurre le metriche CUE-DETR |
| VandeVeire 2018 | Solo metadati e blocco di retrieval documentato | Nessun contenuto letto o usato |

## 4. Limiti e prossimi controlli

- Non sono disponibili i sei brani reali né i loro output struttura/ASR in questa
  cartella: il pilot usa un fixture sintetico etichettato e non dimostra qualità
  musicale.
- Le fonti sono prevalentemente EDM/club; non generalizzare a generi o stili di
  DJ diversi.
- Il modulo preserva gli anchor del provider e non inventa beat, ma non è stato
  valutato con ascolti né con giudizi indipendenti.
- La soglia di 8 s per la finestra di transizione e la guardia parole di 80 ms
  sono configurabili e vanno confrontate con dati reali, non presentate come
  validate.
- VandeVeire 2018 resta bloccato; un accesso istituzionale o un repository
  author-approved sarebbe necessario per una sintesi completa.
