# Letteratura scientifica per un sistema di Automatic Music Mixing intelligente

**Revisione del 18 ottobre 2026** · Ricerca orientata all’implementazione in Python (con C++ per i colli di bottiglia DSP).

**Obiettivo A (prima fase):** trovare automaticamente due punti, in posizioni arbitrarie di due brani, la cui connessione risulti *musicalmente interessante e convincente a un ascoltatore umano*. La corrispondenza può essere vocale/fonetica, melodica, ritmica, armonica, timbrica o strutturale.

**Obiettivo B (seconda fase):** rendere il passaggio fluido tramite tempo, pitch, EQ, allineamento, crossfade, stem mixing e automazioni.

**Criterio di inclusione:** ho selezionato 21 lavori centrali, privilegiando il corpo del paper e sezioni di metodologia, esperimenti, tabelle e limitazioni. Questa è una ricognizione approfondita e non una garanzia bibliometrica di *tutti* i lavori pubblicati sul tema. Le risorse ausiliarie sono separate e NON etichettate come letture integrali.

## Risultati che cambiano il disegno del nostro sistema

1. **Compatibilità di sovrapposizione ≠ compatibilità di successione.** Lee et al. separano mashability verticale/orizzontale e mostrano che la continuità della voce può mascherare una brusca sostituzione dell’accompagnamento. Non impostare lo score come un’unica cosine similarity.
2. **Un ottimo taglio non è soltanto una parte “simile”.** Gli autori del 2015 considerano efficace la complementarità tra parti musicalmente statiche e dinamiche. Ottimizzare solo somiglianza tende a produrre effetti banali o noiosi.
3. **L’asimmetria è fondamentale:** 'voce di A + accompagnamento B' e 'voce di B + accompagnamento A' richiedono score diversi (AutoMashup 2025). Le coppie di clip A→B devono anche essere valutate rispetto all’inverso B→A.
4. **Modelli generici ≠ giudice musicale:** in un corpus piccolo di AutoMashup, correlazioni CLAP/MERT vs COCOLA sono quasi nulle. COCOLA r≈0,54 con MOS umano sul proprio compito: utile ma non oracle.
5. **Serve una pipeline specifica per la parola cantata:** allineamento parola/fonema + voce separata + profilo relativo F0 + DTW. L’ASR ordinario non è un allineatore affidabile delle sillabe cantate.
6. **Il problema è combinatorio:** filtrare candidati strutturali, generare micro-punti vocali addizionali e poi reranking. Non valutare ogni possibile millisecondo × millisecondo né enumerare tutti gli ordini delle canzoni.

## Paper analizzati nel testo integrale

Ogni scheda specifica metodo, ciò che implementeremmo, limiti e il link primario. Gli indicatori (precision, recall, SDR, F1, preferenza) appartengono a task e dataset diversi: **non** sono metriche direttamente comparabili fra lavori.
### 01. AutoMashUpper: Automatic Creation of Multi-Song Music Mashups (2014)

**Ambito:** Ricerca frammenti e funzione di compatibilità. **Lettura:** TESTO INTEGRALE (versione HTML).

**Paper:** https://www.researchgate.net/publication/265130656_AutoMashUpper_Automatic_Creation_of_Multi-Song_Music_Mashups

**PDF/repository/materiale:** https://repositorio.inesctec.pt/bitstreams/97188d38-9bf1-46f9-99a3-08d3b656ca4f/download

**Metodo ed evidenza:** Beat, downbeat, segmentazione per frasi; stima locale della mashability come combinazione di chroma armonico, pattern ritmico e bilanciamento spettrale. Confronta sezioni in posizioni arbitrarie di altre canzoni e cerca anche trasposizioni di tonalità. L’audio è allineato con stretching e pitch shift.

**Cosa riutilizziamo:** Implementare un baseline interpretabile S_harmony+S_rhythm+S_spectral e conservarne separatamente i contributi; confrontare ranking locali anziché key globale del brano.

**Limiti rispetto al nostro obiettivo:** Pensato principalmente per sovrapposizione di brani su una base, non per match fonetici parola-parola, e presenta valutazioni di piacevolezza su esempi specifici.

### 02. Automatic Mashup Creation by Considering Both Vertical and Horizontal Mashabilities (2015)

**Ambito:** Compatibilità verticale e orizzontale. **Lettura:** PDF INTEGRALE, incluse formule ed esperimenti.

**Paper:** https://www.ismir2015.uma.es/articles/302_Paper.pdf

**Metodo ed evidenza:** Separa esplicitamente continuità temporale tra sezioni (orizzontale) e sovrapponibilità voce/accompagnamento (verticale). Definisce peso di bilanciamento delle variazioni armoniche (testo/accordi statici complementari a passaggi dinamici), penalità delle parti quasi silenziose, quattro casi di passaggio. 104 brani multitraccia; listening test: 27 partecipanti per orizzontale e 24 per verticale. Il caso che mantiene la linea vocale mentre cambia la base è generalmente meglio giudicato del suo inverso.

**Cosa riutilizziamo:** Il nostro score non deve trattare alla pari vocali e background. Separare S_horizontal e S_vertical e valorizzare la continuità della voce, anche se il sottofondo cambia drasticamente.

**Limiti rispetto al nostro obiettivo:** Assume stem lead/background già disponibili, griglie ritmiche in battute, e segmenti quasi regolari; non tratta il cambio di parola preciso alla sillaba.

### 03. Modeling the Compatibility of Stem Tracks to Generate Music Mashups (2021)

**Ambito:** Rete neurale per stem compatibility. **Lettura:** HTML INTEGRALE, metodologie e valutazioni.

**Paper:** https://arxiv.org/html/2103.14208

**PDF/repository/materiale:** https://arxiv.org/pdf/2103.14208

**Metodo ed evidenza:** Crea esempi positivi unendo stem dalla stessa canzone, negativi da stem di brani discordanti in key/tempo, e una terza categoria di combinazioni esterne musicalmente allineate ma senza etichetta. Valuta PostMixNet vs PreMixNet e usa LSRO, che impedisce di chiamare “sbagliati” tutti i mashup mai ascoltati.

**Cosa riutilizziamo:** Per lo step 1, partire da un punteggio euristico; se passiamo a ML usare 3 classi positive/negative/unlabeled, hard negatives, e scoring dipendente dal ruolo dello stem. Misurare su ascoltatori.

**Limiti rispetto al nostro obiettivo:** I positivi dello stesso brano sono una proxy di buona compatibilità, non una vera annotazione di creatività. L’algoritmo rallenta se effettua beat/key alignment per troppi candidati.

### 04. COCOLA: Coherence-Oriented Contrastive Learning of Musical Audio Representations (2025)

**Ambito:** Coerenza imparata tra stem. **Lettura:** HTML INTEGRALE, training e tabelle esperimenti.

**Paper:** https://arxiv.org/html/2404.16969

**PDF/repository/materiale:** https://arxiv.org/pdf/2404.16969

**Metodo ed evidenza:** Formazione contrastiva su sottogruppi di stem 5 secondi della stessa finestra (positivi) rispetto a finestre non corrispondenti; backbone EfficientNet-B0 e rappresentazioni che incorporano componente armonica/percussiva. Discriminazione per due candidati ~93,87% nel setup riportato; su grandi ranking K=64 ~54,33%. Valutazione MOS con 41 ascoltatori: correlazione Pearson dello score circa r=0,54.

**Cosa riutilizziamo:** Usare score COCOLA come feature del ranking, non come etichetta universale di qualità. Cross-validare con ascoltatori e con funzione bilanciata diversi tipi di transizioni.

**Limiti rispetto al nostro obiettivo:** Studia coerenza di stem simultanei, non il cut A→B, la posizione di parola o la piacevolezza narrativa del cambio di canzone.

### 05. AutoMashup: Automatic Music Mashups Creation (2025)

**Ambito:** Pipeline moderna end-to-end. **Lettura:** HTML INTEGRALE e confronto sperimentale.

**Paper:** https://arxiv.org/html/2508.06516

**PDF/repository/materiale:** https://arxiv.org/pdf/2508.06516

**Metodo ed evidenza:** Integra separazione Demucs, struttura All-In-One, adattamento musicale e COCOLA per valutare accoppiamenti vocal/accompaniment; rileva asimmetria della direzione A-vocal/B-backing e viceversa. Su 21 brani, le correlazioni con COCOLA di CLAP/MERT sono vicine a zero (0,051/-0,018), dunque embedding generici non bastano.

**Cosa riutilizziamo:** Riutilizzare architettura di preprocessing; in scoring valutare entrambe le direzioni e non fare retrieval esclusivamente per cosine embedding generico.

**Limiti rispetto al nostro obiettivo:** Corpus esplorativo piccolo, valutazione non equivale a prova di qualità percepita delle transizioni orizzontali.

### 06. Neural Loop Combiner: Neural Network Models for Assessing the Compatibility of Loops (2020)

**Ambito:** Pairwise learning di compatibilità. **Lettura:** PDF INTEGRALE (incluse tabelle e figure).

**Paper:** https://jbls.fun/documents/chen2020-ismir-neural_loop_combiner.pdf

**PDF/repository/materiale:** https://github.com/mir-aidj/neural-loop-combiner

**Metodo ed evidenza:** Costruisce coppie positive da loop che co-occorrono e negativi via riordinamenti temporali e abbinamenti cross-song; confronta CNN su segnale sommato con rete siamese. Prove di ascolto mostrano utilità della forma dei negativi: temporizzazioni volutamente scorrette insegnano al modello incompatibilità realistiche.

**Cosa riutilizziamo:** Allenare un eventuale reranker su coppie di clip con negativi “difficili”: parole identiche ma note incompatibili, stessa armonia ma accenti fuori tempo, frammenti spostati di una battuta.

**Limiti rispetto al nostro obiettivo:** Focalizzato su loop corti e un dataset principalmente hip-hop; negativi ottenuti da altri brani possono accidentalmente essere ottimi, quindi vanno trattati con prudenza.

### 07. Leveraging Compatibility and Diversity in Computer-Aided Music Mashup Creation (2022/2023)

**Ambito:** Top-K diverso, search multi-obiettivo. **Lettura:** ARTICOLO INTEGRALE.

**Paper:** https://link.springer.com/article/10.1007/s00779-022-01702-z

**Metodo ed evidenza:** Compatibilità basata su tonal interval vectors (TIV) per armonia, ritmo e spectral balance; ricerca multi-modale evolutiva opt-aiNet per soluzioni compatibili ma diverse. Pesi sperimentali armonia/ritmo/spettrale ~0,4/0,4/0,2, modificabili.

**Cosa riutilizziamo:** Non proporre le top 5 tutte quasi identiche: introdurre diversità di meccanismo (vocale, melodico, ritmico), canzone e tempo nel ranking. Diversificazione MMR o beam search diversificata come baseline economica.

**Limiti rispetto al nostro obiettivo:** Risultati su loop e ascoltatori, non corrispondono automaticamente alla valutazione di storytelling, parola e passaggio macrostrutturale.

### 08. Automatic Detection of Cue Points for DJ Mixing (2020 preprint / 2023 journal)

**Ambito:** Punti interessanti nella canzone. **Lettura:** HTML INTEGRALE, dataset e misure.

**Paper:** https://arxiv.org/html/2007.08411

**PDF/repository/materiale:** https://github.com/MZehren/Automix

**Metodo ed evidenza:** Estrae variazioni di sette feature legate a kick, snare, hi-hat, armonico, percussivo, CQT e chroma; novelty detection con kernel a scacchiera per trovare switch point. Dataset 134 tracce annotate; valutazione oggettiva e soggettiva non sovrapponibili (circa 86% precision e 49% recall sugli eventi; circa 96% punti giudicati buoni individualmente).

**Cosa riutilizziamo:** Generare insieme candidati strutturali (downbeat/chorus/bar) e microcandidati vocali (parola, sillaba) invece di esaminare ogni millisecondo.

**Limiti rispetto al nostro obiettivo:** Indirizzato a House/Techno EDM regolari; non fornisce comparazione semantica tra canzoni.

### 09. From Raw Audio to a Seamless Mix: Creating an Automated DJ System for Drum and Bass (2018)

**Ambito:** Sistema DJ pratico e controlli. **Lettura:** ARTICOLO INTEGRALE.

**Paper:** https://link.springer.com/article/10.1186/s13636-018-0134-8

**Metodo ed evidenza:** Pipeline completa con beat/downbeat, stima delle sezioni, cue points, previsione vocale con SVM, tipologie di transizione musicali (double drop, rolling, relaxed). Approccio regolistico; 90,9% accuratezza per determinate annotazioni nel set test e F1 voce ~61,7%, a mostrare che evitare vocal clash è difficile.

**Cosa riutilizziamo:** Implementare il prototipo con 3-4 schemi di transizione e regole esplicite, distinguendo quando sovrapporre voci e quando no.

**Limiti rispetto al nostro obiettivo:** Fortemente tarato su drum & bass 4/4, non su pop, rap e generi con tempi e sezioni meno regolari.

### 10. A Computational Analysis of Real-World DJ Mixes using Mix-to-Track Subsequence Alignment (2020)

**Ambito:** Dataset e comportamenti di DJ. **Lettura:** HTML INTEGRALE, allineamento e statistiche.

**Paper:** https://arxiv.org/html/2008.10267

**PDF/repository/materiale:** https://arxiv.org/pdf/2008.10267

**Metodo ed evidenza:** Allinea frammenti originali dei brani alle registrazioni di set DJ mediante rappresentazioni beat-synchronous (chroma/MFCC), key shift e subsequence-DTW, ricostruendo punti di ingresso, uscita e durate. 1.557 mix, 13.728 tracce e 20.765 transizioni secondo il corpus del paper; la maggior parte degli adattamenti di tempo è contenuta.

**Cosa riutilizziamo:** Riutilizzare la pipeline DTW per estrarre positivi reali da DJ set e stimare distribuzione dei punti di passaggio; per le nostre word-matches costruire un benchmark annotato distinto.

**Limiti rispetto al nostro obiettivo:** Il fatto che un DJ abbia scelto un punto non dimostra che sia perfetto; vincoli legali su brani e mix, bias di genere e metodi di allineamento.

### 11. Automatic DJ Transitions with Differentiable Audio Effects and Generative Adversarial Networks (2022)

**Ambito:** Secondo step: audio rendering. **Lettura:** HTML INTEGRALE, architettura e test ascolto.

**Paper:** https://arxiv.org/html/2110.06525

**PDF/repository/materiale:** https://arxiv.org/pdf/2110.06525

**Metodo ed evidenza:** Modello generativo imita transizioni di DJ con parametri di fader e equalizzazione multibanda differenziabili; esperimento con 136 ascoltatori validi su musicalità, fluidità e preferenze; mix umani tendono a vincere.

**Cosa riutilizziamo:** Da usare per lo STEP 2: imparare curve di fader e EQ, o ottimizzarle una volta selezionati i cue points.

**Limiti rispetto al nostro obiettivo:** Assume cue points già fissati; non determina quali parole, note o ritornelli debbano essere collegati.

### 12. Analyzing Song Structure with Spectral Clustering (2014)

**Ambito:** Segmentazione strutturale / ricorrenze. **Lettura:** PDF INTEGRALE, metodi e grafici.

**Paper:** https://brianmcfee.net/papers/ismir2014_spectral.pdf

**PDF/repository/materiale:** https://librosa.org/doc/latest/generated/librosa.segment.recurrence_matrix.html

**Metodo ed evidenza:** Combina ricorrenza timbrico-armonica (chroma) con successione locale (MFCC), calcola matrice di affinità e autovettori del Laplaciano normalizzato per identificare segmenti a più scale. Utile per ritornelli ripetuti e sezioni formalmente simili.

**Cosa riutilizziamo:** Usare per ridurre i candidati ridondanti e riconoscere blocchi chorus/verse, e non soltanto discontinuità di energia.

**Limiti rispetto al nostro obiettivo:** Confine musicalmente significativo non implica sempre forte discontinuità; una canzone con struttura libera può segmentarsi male.

### 13. All-In-One Metrical and Functional Structure Analysis with Neighborhood Attentions on Demixed Audio (2023)

**Ambito:** Beat, downbeat, verse/chorus e confini. **Lettura:** HTML INTEGRALE e tabelle metriche.

**Paper:** https://arxiv.org/html/2307.16425

**PDF/repository/materiale:** https://github.com/mir-aidj/all-in-one

**Metodo ed evidenza:** Rete multi-task che usa stem separati per tracciare beat, downbeat, boundaries ed etichette functional sections. Su Harmonix Set, beat F1 0,958, downbeat F1 0,915, structure boundary F1 ~0,660, quindi beat solido ma ritornelli/non ritornelli più incerti.

**Cosa riutilizziamo:** Prototipo immediato: fare analisi sul brano e generare finestre candidate intorno a downbeat e sezione, poi aggiungere punti fonetici fini.

**Limiti rispetto al nostro obiettivo:** Confini non sempre precisi e transizioni non necessariamente coincidenti con l’inizio di una sezione.

### 14. A Real-Time Lyrics Alignment System Using Chroma and Phonetic Features for Classical Vocal Performance (2024)

**Ambito:** Allineare fonemi + note. **Lettura:** HTML INTEGRALE, metriche e ablation.

**Paper:** https://arxiv.org/html/2401.09200

**PDF/repository/materiale:** https://arxiv.org/pdf/2401.09200

**Metodo ed evidenza:** Fonde phonetic posteriorgrams e chroma tramite dynamic time warping per seguire l’esecuzione vocale in tempo reale; valutazione su 24 Lieder di Schubert con MAE fino a ~136 ms nel setup riportato.

**Cosa riutilizziamo:** Per parole uguali in brani diversi combinare allineamento fonetico e curva F0, con DTW vincolata sul tempo; è un principio adattabile, non una soluzione pronta per pop.

**Limiti rispetto al nostro obiettivo:** Richiede score/MIDI e un testo di riferimento nell’applicazione originale; dominio classico anziché brani commerciali arbitrari.

### 15. Phoneme Level Lyrics Alignment and Text-Informed Singing Voice Separation (2021)

**Ambito:** Allineamento cantato parola/fonema. **Lettura:** TESTO INTEGRALE disponibili PDF/HTML research copy.

**Paper:** https://schufo.github.io/files/2021_TASLP3091817.pdf

**PDF/repository/materiale:** https://github.com/schufo/lyrics-aligner

**Metodo ed evidenza:** Introduce attenzione monotona DTW differenziabile per allineare fonemi a registrazioni cantate e addestra congiuntamente separazione di voce condizionata dal testo; con trascrizione affidabile allinea segmenti vocali e migliora separazione.

**Cosa riutilizziamo:** Usare allineatore vocale specifico per canto e salvare intervalli temporali di ogni parola e sillaba. Su quelle finestre confrontare frequenza fondamentale relativa e articolazione.

**Limiti rispetto al nostro obiettivo:** Richiede lyrics corretti; il testo non può essere sempre dedotto senza errori con un normale ASR. Va gestita la confidenza e la sincronizzazione.

### 16. CREPE: A Convolutional Representation for Pitch Estimation (2018)

**Ambito:** Curva F0 della voce. **Lettura:** PAPER E METODOLOGIA + PROVE disponibili.

**Paper:** https://arxiv.org/pdf/1802.06182

**PDF/repository/materiale:** https://github.com/marl/crepe

**Metodo ed evidenza:** Pitch tracker monofonico CNN direttamente su waveform: crop 1024 sample a 16 kHz, 360 categorie cent e confidenza voicing; confronti con pYIN e SWIPE e robustezza al rumore.

**Cosa riutilizziamo:** Su voce separata salvare F0(t) e normalizzarla sottraendo la mediana in semitoni; confrontare la forma della stessa parola con DTW e penalità confidenza bassa.

**Limiti rispetto al nostro obiettivo:** Una sola F0 per frame: sovrapposizione vocale, riverbero e stem contaminato possono rendere la stima falsa; non misura significato né fonemi.

### 17. A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation (Basic Pitch) (2022)

**Ambito:** Note strumentali e vocali. **Lettura:** PDF INTEGRALE con esperimenti e tabelle.

**Paper:** https://arxiv.org/pdf/2203.09893

**PDF/repository/materiale:** https://github.com/spotify/basic-pitch

**Metodo ed evidenza:** Rete leggera multi-output stima onset, pitch e attivazione nota, ricostruisce note MIDI con post-processing di posteriorgram; fra le prove note F_no_offset 0,79 su GuitarSet e 0,52 su Molina canto, dipendenti dal dataset; supporta vari strumenti.

**Cosa riutilizziamo:** Usare su basso/other stem per motivi melodici e chord contours; distinguere matching esatto della nota dalla trasposizione relativa.

**Limiti rispetto al nostro obiettivo:** Trascrivere pieno mix polifonico dà molti errori; onset/offset non sempre attendibili, specialmente per note prolungate e strumenti composti.

### 18. Hybrid Transformers for Music Source Separation (2022)

**Ambito:** Separare sorgenti audio. **Lettura:** HTML INTEGRALE, architettura e tabella SDR.

**Paper:** https://arxiv.org/html/2211.08553

**PDF/repository/materiale:** https://github.com/facebookresearch/demucs

**Metodo ed evidenza:** Demucs ibrido unisce percorso waveform e percorso spectrogram con attenzione self/cross per separare basso, batteria, voce e altri. Benchmark SDR 9,20 dB con training supplementare, 7,52 dB senza extra training riportati nel paper.

**Cosa riutilizziamo:** Preprocessing ai fini del riconoscimento strumenti e voce. Usare due livelli di accuratezza: stem rapidi per scanning, separazione di qualità su candidati top-K.

**Limiti rispetto al nostro obiettivo:** Imperfezioni stem (bleed, artifact) possono creare false somiglianze e, per il rendering, glitch. Repository Meta originale archiviato.

### 19. Music Source Separation with Band-Split RoPE Transformer (BS-RoFormer) (2023)

**Ambito:** Alternativa per separazione più pulita. **Lettura:** HTML INTEGRALE, architettura + ablation.

**Paper:** https://arxiv.org/html/2309.02612

**PDF/repository/materiale:** https://arxiv.org/pdf/2309.02612

**Metodo ed evidenza:** Segmenta lo spettrogramma complesso in bande non uniformi, gerarchia Transformer temporale/interbanda con RoPE; maschere complesse per mantenere modulo e fase. Sul MUSDB18-HQ media SDR 9,80 dB per modello piccolo senza dati aggiuntivi; il grande modello aggiunge centinaia di brani di training privati.

**Cosa riutilizziamo:** Confrontare separazione Demucs e BS-RoFormer sulle 10–20 transizioni candidate problematiche per ridurre leakage sulla voce, più che inseguire un singolo SDR.

**Limiti rispetto al nostro obiettivo:** Modelli grandi richiedono GPU; le cifre SDR variano con training e split, e non misurano automaticamente qualità creativa della transizione.

### 20. CrossSong Puzzle: Generating and Unscrambling Music Mashups with Real-time Interactivity (2015)

**Ambito:** Gestire la combinatoria e i vincoli globali. **Lettura:** PDF INTEGRALE, pseudocodice + analisi dei limiti.

**Paper:** https://staff.aist.go.jp/m.goto/PAPER/SMC2015smith.pdf

**PDF/repository/materiale:** https://staff.aist.go.jp/jun.kato/CrossSong/

**Metodo ed evidenza:** Ricerca combinatoria di frammenti tra 8 brani, con precomputazione mashability a livello di sezione, riduzione griglia delle trasposizioni, ricerca casuale con tempo limite e allineamento RubberBand. Stima ~10^16 combinazioni ancora prima delle trasposizioni.

**Cosa riutilizziamo:** Evita enumerazione N!: preindicizza frammenti per sezione/role, filtra localmente, salva archi top-K e cerca mix globali con beam search diversificato.

**Limiti rispetto al nostro obiettivo:** Assume 4/4 e BPM costante, limita a sezioni strutturali; valutazione riguarda anche usabilità del gioco e non soltanto preferenze di mix.

### 21. Automatic Playlist Sequencing and Transitions (2017)

**Ambito:** Ordinamento N canzoni e scelta punti di transizione. **Lettura:** PDF INTEGRALE, pseudocodice algoritmo, esperimenti e analisi degli errori.

**Paper:** https://archives.ismir.net/ismir2017/paper/000086.pdf

**PDF/repository/materiale:** https://research.atspotify.com/publications/automatic-playlist-sequencing-and-transitions

**Metodo ed evidenza:** Separa ordinamento dei brani (shortest Hamiltonian path in un grafo di timbro, tonalità e tempo) e ottimizzazione della transizione di coppia. Per ogni coppia di downbeat calcola matrici di costo su timbro, chroma, loudness, vocalness, fine su confini/drops; somma costi normalizzati e sceglie l’argmin. Prova 48 coppie di brani con 4 curatori: circa 64,13% Good, 28,26% OK, 7,61% Bad; il beat tracking era una delle maggiori fonti di problemi. Per il sequenziamento il giudizio su coppie brusche scende da media 4,2 a 2,4 in 6 playlist.

**Cosa riutilizziamo:** È quasi il template di un baseline da implementare ora: feature matrices + argmin(t_A,t_B), aggiungendo i termini parola+F0+riff e consentendo di invertire la direzione. Per N canzoni sostituire il grafo per soli brani con grafo di frammenti e stato di utilizzo.

**Limiti rispetto al nostro obiettivo:** Il loro pruning costringe a uscire nell’ultimo 25% di A ed entrare nel primo 20% di B: va RIMOSSO per il nostro obiettivo di collegare ritornelli e parti arbitrarie. Penalizza sempre la presenza simultanea di vocali, mentre a noi interessa talvolta proprio creare un handoff tra parole.


## Proposta di architettura — FIRST PROOF OF CONCEPT

### Preprocessing (offline, una volta per canzone)

`audio -> (Demucs/BS-RoFormer) stems -> beat/downbeat, segmenti/etichette, chroma, key, onset, loudness, F0 voce, parole+fonemi, note/motivi -> cache indicizzata`

**Finestre su due scale:** (i) 2, 4, 8, 16 battute attorno a cue points, downbeat, chorus/verse; (ii) 0,2–3 s attorno a parole, fonemi, note e riff. Le durate sono proposte progettuali, non risultati sperimentali dei paper.

### Candidate generation e ranking

- **Candidati strutturali:** cerca coppie di finestre con vicinanza armonica, differenza BPM correggibile, allineamento di battuta/frase e densità complementare. Recupera per ogni segmento K vicini, ma con scelta della direzione e stem role.
- **Candidati vocali/semantici:** quando due lyric token hanno lo stesso lemma o una relazione sonora/semantica, usa timestamps di inizio/fine parola, allineamento sillabico, confidenza F0 e DTW della curva di intonazione normalizzata. *Importantissimo:* la stessa parola non garantisce la stessa scansione prosodica e due curve F0 possono essere simili pur cadendo sopra armonie incompatibili.
- **Candidati riff/note:** su stem monofonici/strumentali, trascrivi note, confronta intervalli di altezza relativi e tempi relativi, permettendo trasposizione controllata.
- **Primo ranking:** punteggi espliciti `horizontal`, `vertical`, `rhythm`, `harmony`, `vocal-word`, `melody`, `novelty`, `transform_cost`, `artifact_risk`, con regole per evitare vocal clash.
- **Reranking:** addestra in futuro un modello pairwise o ranker a preferenze, mantenendo hard negatives *non falsamente etichettati*. COCOLA e/o PreMixNet sono feature/esperimenti, non verità percettiva.
- **Mix N brani:** nodi=(song, in, out, style/stem choice), archi=(candidate transition, score, transform). Beam search con vincoli di durata, varietà e numero d’utilizzi per traccia. Top 5 diversificato per tipo di passaggio.

### Validazione: prova che la macchina sceglie meglio del caso

**Dataset pilota:** 20–40 canzoni (più generi), con licenza/uso legittimo e lyrics verificati manualmente per un sottoinsieme. Per verificare *solo la scelta dei punti* esportare rendering standardizzati, semplici, uguali per tutti i sistemi (es. stesso crossfade ed eventualmente lo stesso beat alignment). Non usare effetti audio sofisticati che nascondano punti sbagliati: quello è il secondo step.

Per ogni coppia produrre almeno tre alternative: (A) random su downbeat valido, (B) baseline armonica+ritmica tipo AutoMashUpper, (C) sistema proposto con feature vocali/strumentali/strutturali. Far ascoltare coppie cieche A/B, ordine randomizzato, agli stessi valutatori, idealmente almeno 20–30 per un primo studio esplorativo (non un campione statisticamente dimensionato). Raccogliere separate valutazioni di **(1) plausibilità/musicalità del collegamento, (2) interesse creativo, (3) continuità percepita**. Stima Win Rate delle preferenze + intervalli di confidenza per brano/valutatore, e analisi per tipo di transizione (vocale, riff, ritmica, armonica). Evitare leakage tra brani training/test e limitare l’esposizione ripetitiva della stessa transizione.

**Ablation:** rimuovi una famiglia di feature per volta: no lyrics; no F0; no harmony; no structural segmentation; no stem roles; no complementary density. Se la prestazione non peggiora, la feature potrebbe essere superflua. Distinguere la capacità di riconoscere *match* dalla capacità di preferire la *migliore transizione*.

**Criterio di successo proposto:** miglioramento credibile e ripetibile delle preferenze di ascolto rispetto al baseline musicale e al caso (non solo correlazione con una metrica). Un semplice superamento del random non basta.

### Secondo step — effetti/rendering

Sviluppare poi stem-selective crossfade, phase-coherent alignment, pitch/time correction, EQ multibanda, gestione transienti e clipping, gestione voce-vs-voce. Base research: DJGAN 2022; strumenti Demucs/BS-RoFormer, Rubber Band, loudness normalization. Qualità sonora giudicata con rating separato da creatività e scelta cue points.

## Ordine di lettura/implementazione consigliato

1. **Lee 2015** + **AutoMashUpper 2014**: baseline orizzontale/verticale.
2. **Cue Points 2020** + **All-In-One 2023**: trovare punti validi per passaggio.
3. **Phoneme Lyrics 2021** + **CREPE 2018**: la feature unica della parola con uguale intonazione.
4. **Huang 2021** + **COCOLA 2025**: imparare/rerankare compatibilità senza assumere che similitudine sia qualità.
5. **CrossSong 2015** + **Diversity 2023**: scalabilità per N brani e top-K creativi.
6. **DJ Transitions GAN 2022** + **BS-RoFormer 2023**: secondo step.

## Risorse aggiuntive e codici — NON letti come paper integrali

- **Psychoacoustic Approaches for Harmonic Music Mixing (2016)** — https://www.mdpi.com/2076-3417/6/5/123 — Modello di consonanza basato su roughness e pitch commonality, più informativo della sola stima di tonalità; letto parzialmente tramite testo indicizzato, NON verificato integralmente.
- **WhisperX: Time-Accurate Speech Transcription of Long-Form Audio (2023)** — https://arxiv.org/abs/2303.00747 — ASR/forced alignment parlato; non assumere accuratezza su canto; riferimento complementare, non analizzato qui integralmente.
- **MSAF: Music Structure Analysis Framework** — https://github.com/urinieto/msaf — Libreria di segmentazione struttura musicale; non è un paper aggiuntivo esaminato integralmente in questa ricognizione.
- **MIREX Lyrics-to-Audio Alignment task (2024)** — https://music-ir.org/mirex/wiki/2024%3ALyrics-to-Audio_Alignment — Specifiche, metriche e dataset della valutazione di parole in canzoni; risorsa per benchmark.
- **All-In-One implementation** — https://github.com/mir-aidj/all-in-one — Codice e inferenza struttura.
- **AutoMashup implementation** — https://github.com/ax-le/automashup — Pipeline codice moderna.
- **COCOLA implementation** — https://github.com/gladia-research-group/cocola — Controllare repository/versione, compatibilità e checkpoint prima di riuso.
- **Zehren annotated cue dataset** — https://github.com/MZehren/M-DJCUE — Cue point annotati.
- **Phoneme singing alignment & separation** — https://github.com/schufo/plla-tisvs — Codice di training e separazione.
- **Dataset AutoMashup** — https://zenodo.org/records/16921593 — Materiali associati (verificare licenze e formato).
- **MUSDB18** — https://sigsep.github.io/datasets/musdb.html — Dataset benchmark di 4 stem; controllare licenza.
- **MedleyDB** — https://medleydb.weebly.com/ — Dataset multitraccia e annotazioni musica.

## Avvertenze

- Le pipeline spesso richiedono audio e testi protetti da diritto d’autore. Analisi di copie lecite e pubblicazione/distribuzione di mashup sono questioni distinte; controllare licenze dei brani, dei dataset e dei modelli.
- Non trattare la metrica SDR della separazione come un punteggio di qualità della transizione, né F1 del beat come misura di creatività.
- Molti paper valutano EDM, stem da stessa canzone o loop brevi, non il **caso più ambizioso di parola→parola tra brani diversi**. Questa rimane una ipotesi progettuale da sottoporre a esperimento percettivo.
- I link a repository e checkpoint possono cambiare: verificarne lo stato prima di integrare codice, dipendenze, licenze.
