# Revisione tecnica indipendente dell'orchestratore

Queste note distinguono lettura, codice ispezionato e proposte. Nessuno dei nuovi modelli di questa ricerca è stato eseguito sull'audio; l'inferenza Audiobox appartiene alla V8 già documentata. Le copie scaricate sono fonti accessibili dall'editore, dagli autori o da repository ufficiali. Un download o un'estrazione di testo non equivalgono a lettura integrale.

## Punti e intervalli

**Evidenza:** Zehren et al., *Automatic Detection of Cue Points for the Emulation of DJ Mixing*, introduzione, definiscono inizio del passaggio, cambio di predominanza e scomparsa della prima traccia. Fonte editore: https://direct.mit.edu/comj/article/46/3/67/117159/Automatic-Detection-of-Cue-Points-for-the . La pagina è risultata indicizzata, ma il recupero diretto dell'orchestratore ha restituito 403: questa nota si limita al contenuto visibile dell'introduzione e non dichiara lettura integrale.

**Inferenza progettuale:** il nostro record `(cue_out, cue_in)` con handoff quasi istantaneo non rappresenta questi tre momenti. Il cambio può richiedere preparazione, pickup e arrivo; un buon punto di entrata non specifica quando iniziare a far sentire B. Propongo un evento con intervalli, clock delle due tracce e punto di predominanza espliciti. Un cut rapido resta un caso valido, non deve essere sostituito obbligatoriamente da un crossfade lungo.

## CUE-DETR: opportunità di cue, non certificato di cambio

**Evidenza:** https://arxiv.org/html/2407.06823v1 , §§3.1–3.5, 4.2–4.5. Dataset EDM in 4/4 e tempo costante, 4.710 brani, 21.461 cue di quattro DJ. Il testo distingue cue originali da confini di frase aggiunti per la valutazione: le due misure non vanno confuse. I cue sono soggettivi e incompleti; un punto non annotato non è automaticamente sbagliato. I risultati non validano rap/pop o la compatibilità tra due brani, né misurano suspense risolta. Lettura: sezioni indicate del testo HTML e confronto con il codice ufficiale; PDF locale archiviato, non dichiarato integralmente letto pagina per pagina.

**Codice ispezionato:** `root_sources/cue_inference.py`, esempio ufficiale https://raw.githubusercontent.com/ETH-DISCO/cue-detr/main/cue_points.py . Prepara l'intera serie di immagini in batch; usa CPU quando CUDA manca. Il parametro `radius` descritto come barre viene passato a `scipy.signal.find_peaks(distance=...)` sulla sequenza dei punteggi ordinati, non convertito in tempo o barre; inoltre applica min–max ai punteggi prima della soglia. Una soglia 0,9 di questo esempio non va presentata come probabilità calibrata, né la distanza 16 come garanzia di 16 barre. Sono osservazioni statiche sull'esempio scaricato, non un esperimento sul checkpoint. Prima di integrare serve riprodurre preprocessing/postprocessing del lavoro, conservando versione e coordinate.

## Mosaikbox: compatibilità locale non basta

**Evidenza:** https://zenodo.org/records/14877463/files/000093.pdf , pp. 852–855, §§3.3.6, 3.4, 4 e 5, e limiti p.856. Gli autori riportano che massimizzare la sola mixability su tutti i beat produceva cambi mal collocati e restringono la ricerca ai confini di segmento. Nell'ascolto DnB, la modifica degli stem migliora il risultato rispetto alla base, mentre l'informazione contestuale delle lyrics non fornisce un vantaggio significativo. Valutazione circoscritta a 30 partecipanti e selezioni da 16 brani DnB: nessuna estensione automatica ai nostri quattro brani. Lettura locale delle pagine indicate.

**Inferenza progettuale:** la compatibilità va valutata dopo l'ammissibilità dell'evento strutturale. Non copiare i pesi del paper, i suoi 55 secondi minimi o i 16 downbeat di durata come legge musicale: sono scelte sperimentali. La correlazione tra compatibilità della coppia e qualità del passaggio può cambiare con il renderer e i criteri di scelta.

## Decisioni a più scale

**Evidenza:** Williams et al. 2025, https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.TIME.2025.20 , §§2–3 e 5. Il lavoro distingue azioni sul raccordo, selezione della traccia e andamento del set; osserva che la raccomandazione sequenziale generica non include i vincoli locali del mix. È un framework/review, non una dimostrazione sperimentale di un planner nuovo. Lettura del testo HTML nelle sezioni indicate; PDF archiviato.

**Proposta:** trattare `resta`, `prepara`, `passa` e `deferisci` come azioni distinte con contesto gerarchico. Il seguito naturale A→A deve essere un controllo e un'opzione praticabile, senza assumere che ogni salto a B debba essere meno probabile o peggiore. Non abbiamo trovato in questa ricerca una validazione diretta di uno specifico algoritmo di optimal stopping per questi brani: quella formulazione resta proposta nostra.

## Critico con istruzioni: CMI-RM

**Evidenza:** https://arxiv.org/html/2603.00610v2 , §§3–4 e appendici C/F selezionate. Musicalità e aderenza a istruzioni sono compiti distinti; il benchmark include dati con preferenze umane e pseudolabel. Le prestazioni su generazione musicale non equivalgono a rilevare un raccordo cross-song mal posto. Un modello con audio di riferimento è interessante come esperimento, ma non è un certificato di risoluzione. Lettura HTML delle sezioni indicate, non dell'intero PDF di 32 pagine.

**Verifica pratica:** codice di inferenza ufficiale e `config.yaml` https://github.com/Haiwen-Xia/CMI-RewardBench . L'adattatore supporta `device=cpu`, default audio 24 kHz/max30 s; modalità final tronca prima di codificare chunk, standard aggrega finestre. Il modello configura il backbone congelato `OpenMuQ/MuQ-MuLan-large`: i circa 30M addestrabili riportati dal lavoro non sono il costo totale dell'inferenza. Codice Apache-2.0, scheda pesi https://huggingface.co/HaiwenXia/CMI-RM indica CC-BY-NC-4.0. Nessun peso scaricato o inference eseguita in V9. Il tentativo `config.json` restituiva404: il file reale è YAML, ora archiviato.

**Controllo necessario:** ingresso ordinato e finestre centrate sul raccordo; evitare che max_dur elimini il cambio o che una media sul brano diluisca il difetto. Confronti A/B, inversione ordine, contesto ridotto e cambio della sola sorgente servono per scoprire scorciatoie, senza etichettare ogni perturbazione come musicalmente brutta.

## MusTBench: il testo plausibile non prova il timestamp

**Evidenza:** https://arxiv.org/html/2605.29300v2 , §§3–5 e Limitations. Misura localizzazione di eventi, descrizioni di cambi, ordine e regioni di arousal; i modelli mostrano difficoltà e talvolta timestamp fuori durata. Il miglioramento del modello MusT non risolve il giudizio di compatibilità cross-song: è un altro compito. La metrica Hit@3s è troppo grossolana per certificare un confine vocale di pochi millisecondi. Lettura delle sezioni indicate del testo HTML, non di tutte le appendici.

**Verifica del rilascio:** https://github.com/sony/MusTBench README scaricato: contiene inferenza Qwen2.5-Omni e benchmark, con audio da acquisire separatamente e link Hugging Face TBD. Non dedurre che i pesi MusT siano già accessibili soltanto perché esiste il repository. Il README indica875 tracce mentre la tabella1 della v2 indica517: congelare e ispezionare l'artefatto effettivo prima di usarne un conteggio; la divergenza non va normalizzata in silenzio.

## SongEval e modello linguistico musicale lungo

**Evidenza:** SongEval https://arxiv.org/html/2505.10793v1 , §§3–5 e Limitations: include coerenza, struttura e fraseggio vocale tra i giudizi su canzoni intere generate, con addestramento su valutazioni di esperti. È più vicino alla domanda rispetto al solo clipping; resta da verificare il trasferimento a un singolo raccordo tra registrazioni commerciali. Una correlazione riportata nel proprio benchmark non è una percentuale di riuscita dei nostri mix. Lettura delle sezioni indicate.

**Scheda ufficiale ispezionata:** https://huggingface.co/nvidia/music-flamingo-think-2601-hf ; audio in finestre30 s fino a20 min, circa8B parametri, uso non commerciale, Linux/A100/H100 indicati e test su A10080GB. La macchina locale espone Intel UHD620, nessuna GPU NVIDIA rilevata. Non è una dipendenza locale leggera; non eseguita. Capacità di ragionamento musicale e caption non bastano come validazione di giudice del raccordo.

## Libri e pratica: cosa è stato realmente letto

La pagina Grove Atlantic di *How to DJ Right* https://groveatlantic.com/book/how-to-dj-right/ contiene un estratto introduttivo. Non contiene tutta la trattazione tecnica di phrasing: non attribuiamo all'estratto le regole che non mostra. Il capitolo accessibile dal sito dell'autore/editore di *Rock the Dancefloor!* https://www.digitaldjtips.com/rock-the-dancefloor/five-basic-dj-transitions/ tratta cut, fade e beatmix su frase. Il tutorial professionale https://www.digitaldjtips.com/dropmix-like-a-pro-hip-hop-dj/ tratta cambio rapido su frase e il caso di voce prima del primo beat: contraddice l'idea di imporre sempre mix lunghi o attacco esattamente sulla prima sillaba al downbeat. Lettura delle sezioni tecniche web; nessun libro intero dichiarato letto o recuperato da copie non autorizzate.

## Auto-supervisione senza nuovi giudizi a ogni giro

**Fonte selezionata:** Huang et al., https://arxiv.org/html/1709.04384 , introduzione e sezioni Music Puzzle Games/Experiments selezionate. La discriminazione dell'ordine e della consecutività fornisce supervisioni ricavate dall'audio senza nuove etichette manuali; il test medley cross-song è un'estensione difficile, non l'identità del task di training. Questa fonte sostiene l'esperimento di apprendimento sequenziale, non che ogni coppia casuale sia un mix brutto.

**Proposta:** combinare esempi nativi, cue professionali disponibili e controlli sintetici con identità di traccia/produzione bilanciata. Tenere separati compito tecnico e musicale; test per brano/artista, controlli cross-song realmente valutati nei dataset esterni e astensione in caso di evidenza insufficiente. L'operazione può evitare domande continue all'utente; la misura della qualità rimane fondata su dati esterni con annotazioni e limiti noti. Nessuna falsa etichetta “risolve” viene ricavata da RMS, novelty o da una spiegazione testuale.

## Aggiornamento 2026 che cambia il piano dati: Raveform

**Fonte nuova verificata:** https://reference-global.com/article/10.5334/tismir.288?tab=article , §§1,2.2,3,4.1 e tabella3; sito/schemi https://mir-aidj.github.io/raveform/ . Il lavoro pubblicato il20aprile2026 distingue cue generico da intervallo di transizione, sottolinea la dipendenza dalle due tracce e fornisce allineamenti tra mix e originali: 4.902 mix, 56.873 tracce uniche, 1.423 tracce annotate strutturalmente e 53.780 transizioni disponibili secondo la tabella3. I punti di mix derivano dagli allineamenti, non sono tutti annotazioni manuali di qualità. Il lessico energetico è una convenzione EDM esplicitamente limitata al corpus.

**Conseguenza:** il limite sui corpora citato dal framework2025 non va ripetuto come fatto assoluto attuale: Raveform2026 colma una parte importante del vuoto, pur non fornendo un benchmark diretto della suspense risolta nei nostri brani. Prima scelta per un pilot di imitazione dei punti professionali e stima della durata; trasferimento a trap/pop da valutare.

**Accesso verificato:** API Hugging Face mostra un archivio `raveform.zip` da479.239.104byte; non è stato scaricato dall'orchestratore per questa verifica di fattibilità. La pagina del modello rimanda ad All-In-One con avviso v2 forthcoming; l'esistenza del collegamento non prova che il modello aggiornato sia già rilasciato. Il README ufficiale All-In-One richiede NATTEN, con compilazione da sorgente suWindows. Conservare questa distinzione nella scelta del provider.
