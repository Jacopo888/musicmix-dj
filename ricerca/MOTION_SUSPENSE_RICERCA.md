# Motion, suspense, aspettativa e compatibilità della transizione

Ricerca bibliografica e proposta di valutazione per il selettore automatico di
transizioni. Data della ricerca: 2026-10-18.

Questo documento è ricerca soltanto: non modifica codice, ranking, pesi o
dataset del progetto. Il caso Reminder (circa 213.09 s) → Too Many Nights
(circa 182.20 s) è trattato come esempio negativo e come holdout; non viene
usato per regolare una soglia o un peso.

## Risultato principale

Il termine più utile per il fenomeno descritto dall'utente è **musical tension /
suspense / expectation**, con una distinzione esplicita fra:

- energia, intensità e arousal;
- tensione o suspense diretta verso un evento futuro;
- sorpresa, cioè la reazione alla violazione o all'errore di previsione;
- continuazione, cioè la percezione che il nuovo segmento sviluppi il moto
  appena iniziato;
- risoluzione o release, cioè l'arrivo in una condizione stabile, chiusa o
  metrica/armonicamente conclusiva;
- interruzione o abbandono, cioè la caduta o deviazione del moto senza un
  arrivo percepibile.

La conclusione operativa è netta: una diminuzione temporale di
tension_proxy **non è sufficiente** per chiamare una transizione
resolution_likely. Può indicare un release, ma anche l'abbandono della
aspettativa, una pausa, un cambio di ruolo o una divergenza che lascia l'arco
aperto. La similarità cromatica (anche con soglia 0.45) misura somiglianza
tonale locale; non dimostra una cadenza, la fine di una frase, una stabilità
metrica o il completamento dell'aspettativa. Il vecchio criterio
“proxy in calo + chroma similarity” deve quindi essere considerato un
indicatore di trend e non una prova di risoluzione.

Il caso Reminder → Too Many Nights illustra precisamente questa differenza:
un segmento A può costruire un aumento di aspettativa e il segmento B può
avere energia compatibile, tonalità vicina o un buon beat, ma entrare con un
rap piatto che non sviluppa il vettore precedente e non produce un arrivo.
“Rappato” è una descrizione del caso osservato; non è una regola per escludere
rap o strumentali. Un segmento rap può continuare o risolvere se conserva o
trasforma in modo leggibile ritmo, frase, ruolo, densità ed esito atteso.

## Terminologia: cosa misurano davvero i termini

| Costrutto | Domanda percettiva | Evidenza audio possibile | Cosa non implica |
|---|---|---|---|
| Energia / intensità | “Quanto è forte o pieno il suono?” | loudness/log-RMS, densità, roughness, onset rate, spettro | Non implica suspense o arrivo |
| Arousal | “Quanto è attivante?” | dinamica, tempo, densità, roughness, performance | Un evento sorprendente può essere molto attivante senza essere stato preparato |
| Tensione / suspense | “Sento instabilità, conflitto o attesa di qualcosa?” | trend di più feature, instabilità tonale, dissonanza, aspettativa, contesto | Non è una semplice discesa o salita di RMS |
| Sorpresa | “L’evento è diverso da ciò che mi aspettavo?” | errore rispetto a una previsione, novità, violazione di una continuazione | È soprattutto una risposta all’evento, non l’attesa che lo precede |
| Continuazione | “Il nuovo segmento porta avanti il moto appena avviato?” | direzione di trend, frase/metrica, ruolo ritmico, densità e sviluppo coerenti | Non richiede copia timbrica o energia identica |
| Risoluzione / release | “L’aspettativa trova un arrivo stabile o comprensibile?” | cadenza/attrazione, chiusura di frase, downbeat/arrivo, ritorno o stabilizzazione, drop strutturalmente preparato | Non equivale a “energia più bassa” |
| Interruzione / abbandono | “Il moto si spegne o devia senza chiusura?” | mismatch post-giunzione + arco aperto + assenza di evidenza di arrivo | Non è sempre un difetto: può essere contrasto intenzionale |

Lehne e Koelsch distinguono tensione e arousal: l'arousal può essere alto
senza tensione, mentre la tensione è legata a conflitto, instabilità,
dissonanza, incertezza e previsione di un evento futuro. La sorpresa tende a
seguire l'evento inatteso; la suspense è lo stato anticipatorio prima
dell'evento. In termini di progetto, il modello deve osservare almeno il
prima e il dopo della giunzione, invece di ricavare una semantica da una sola
pendenza.

## Fonti primarie e risultati verificati

### 1. Farbood (2012): modello temporale parametrico della tensione

**Riferimento.** Morwaread M. Farbood, “A Parametric, Temporal Model of
Musical Tension”, *Music Perception* 29(4), 387–428 (2012),
doi: [10.1525/mp.2012.29.4.387](https://doi.org/10.1525/mp.2012.29.4.387).
È stato letto il PDF dell'autrice, 42 pagine:
[PDF integrale](https://bpb-us-e1.wpmucdn.com/wp.nyu.edu/dist/f/11865/files/2020/08/Farbood_2012_Musical_Tension.pdf).

**Metodo e input.** Il modello combina armonia, altezza, aspettativa melodica,
dinamica, frequenza degli onset, tempo, metro e regolarità/sincopi ritmiche.
Usa una finestra attentiva corrente e una finestra di memoria precedente;
pendenze nella stessa direzione vengono amplificate e integrate nel tempo.
Gli esperimenti usano 207 file MIDI per il primo studio e dieci stimoli
classici nel secondo; il secondo studio ha 35 ascoltatori MIT (due esclusi),
slider continuo a 60 Hz e quattro ripetizioni. Gli stimoli sono Bach,
Vivaldi, Beethoven, Brahms, Schoenberg e materiale MIDI/WAV.

**Risultato utile.** Le finestre ottimali sono dell'ordine di secondi per molte
feature; in alcuni esempi le correlazioni fra modello e giudizi continui sono
alte (il documento riporta valori da circa .67 a .99 per gli estratti). La
struttura a memoria breve è una giustificazione per non limitarsi alla
differenza fra due finestre adiacenti.

**Cosa non dimostra.** Farbood modella una traiettoria percettiva di tensione,
aspettativa e salienza, ma non fornisce un classificatore “risolto/non
risolto”. Una pendenza in calo non dice se l'ascoltatore ha raggiunto una
cadenza o ha semplicemente perso l'aspettativa. Il modello non è addestrato
su audio pop/rap misto, non usa un dataset di transizioni DJ, e non ho trovato
codice o checkpoint pubblicati insieme all'articolo. La licenza del PDF
dell'autrice non è stata verificata come licenza di riuso del modello.

**Implicazione.** È una base metodologica per un indicatore chiamato
tension_trend_observable, non per resolution_likely. Risponde anche alla
domanda su Farbood: il modello incorpora variabili legate ad aspettativa,
salienza e direzione del cambiamento, ma la “departure” o la caduta della
tensione non è una prova di chiusura armonica o frasale.

### 2. Barchet, Rimmele e Pelofi (2024): TenseMusic da audio

**Riferimento.** A. V. Barchet, J. M. Rimmele, C. Pelofi, “TenseMusic: An
automatic prediction model for musical tension”, *PLOS ONE* 19(1),
e0296385 (2024), doi:
[10.1371/journal.pone.0296385](https://doi.org/10.1371/journal.pone.0296385).
È stato letto il testo HTML integrale:
[articolo open access](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0296385).
Il progetto pubblica codice e materiale su
[GitHub](https://github.com/vivienbarchet/TenseMusic) e
[OSF](https://osf.io/zspuw).

**Metodo e input.** Sei feature derivate dall'audio (loudness, altezza,
tensione tonale, roughness, tempo, frequenza degli onset) sono standardizzate,
smussate e campionate a 10 Hz. Il modello usa finestre attentive e di memoria
come Farbood; gli autori ottimizzano pesi e finestre con regressione
regolarizzata/GLMM. La raccolta ha 30 studenti (sei esclusi), 38 brani di
musica classica occidentale di 60–90 secondi, slider continuo e
cross-validation lasciando fuori un brano alla volta. Il giudizio richiesto
è la tensione percepita, non la qualità di una transizione.

**Fattibilità tecnica verificata.** Nella sezione *Introduction* gli autori
descrivono il modello come open-source e basato su Python, con il solo audio
come input. Le sezioni *Model generation*, *Automatic feature extraction* e
*Slope estimation* specificano notebook Python 3.9.7, librosa/MoSQITo/basic-pitch,
step di 250 ms, finestra attentiva di 3 s e memoria precedente; il codice di
estrazione e predizione è nel repository e gli artefatti di dati/statistica/
musica sono indicati su OSF. È quindi un candidato realistico per una prova
offline in Python, senza MATLAB e senza MIDI gold. Questo non rende la sua
uscita una decisione di risoluzione: pitch/tonal tension e onset possono
essere fragili su mix pop/rap e i pesi sono ottimizzati su classica.

**Risultati e limiti.** Il modello riportato ha media Spearman circa .59 in
validazione; l'articolo riporta anche valori intorno a .61–.68 per
configurazioni e scale temporali specifiche. Loudness è il contributo più
forte; roughness è rilevante; la tensione tonale e onset/tempo sono più
variabili. Gli autori avvertono che il modello non è universale e può fallire
su materiale percussivo o moderno. L'articolo è CC BY; il repository pubblico
è stato verificato, ma non ho trovato un file LICENSE che autorizzi
automaticamente il riuso del codice. Non c'è un checkpoint neurale.

**Implicazione.** È un buon confronto per le feature e le finestre, con costo
offline potenzialmente gestibile, ma il nome della sua uscita deve rimanere
“tensione osservabile”. In particolare, il fatto che loudness spieghi molto
del giudizio classico rafforza la separazione fra energy_* e
closure_evidence: RMS non va rinominato tensione.

### 3. Lehne e Koelsch (2015): teoria di tensione e suspense

**Riferimento.** Moritz Lehne e Stefan Koelsch, “Toward a general
psychological model of tension and suspense”, *Frontiers in Psychology* 6:79
(2015), doi:
[10.3389/fpsyg.2015.00079](https://doi.org/10.3389/fpsyg.2015.00079).
Testo HTML e PDF integrali consultati:
[articolo](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.00079/full)
e [PDF](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.00079/pdf).

**Contenuto.** La proposta descrive la tensione come stato affettivo
diretto verso un evento futuro e generato da conflitto, instabilità,
dissonanza o incertezza. L'arousal e la tensione sono correlati ma distinti:
un evento sorprendente può generare arousal senza essere stato anticipato con
suspense. Il modello concettuale passa da un evento iniziale a uno spazio di
esiti; l'incertezza e la posta emotiva degli esiti determinano la tensione,
che può generare un nuovo arco.

**Limiti.** È un modello teorico senza codice, checkpoint o dataset audio.
Gli autori dichiarano che la componente temporale, il calcolo delle
probabilità e la durata del ritardo restano incompleti. L'articolo è CC BY.

**Implicazione.** Fornisce il vocabolario per separare suspense_before,
surprise_after e resolution_after; non offre una regola numerica da
inserire nel ranking.

### 4. You et al. (2021): previsione contestuale e arco non chiuso

**Riferimento.** Shan You, Li Sun, Xiaomei Li e Yung Yang, “Contextual
prediction modulates musical tension: Evidence from behavioral and neural
responses”, *Brain and Cognition* 152:105771 (2021), doi:
[10.1016/j.bandc.2021.105771](https://doi.org/10.1016/j.bandc.2021.105771).
È stato letto il PDF integrale disponibile presso l'istituto:
[PDF](https://ir.psych.ac.cn/bitstream/311026/40203/1/Contextual%20prediction%20modulates%20musical%20tension_%20Evidence%20from%20behavioral%20and%20neural%20responses.pdf).
La pagina bibliografica è su
[PubMed](https://pubmed.ncbi.nlm.nih.gov/34217125/).

**Metodo e labels.** Trenta musicisti addestrati (uno escluso) ascoltano sei
sequenze pianistico-tonali di 14 accordi in quattro condizioni. Le
manipolazioni distinguono una conclusione tonica attesa da una conclusione
supertonica/incongruente e alta/bassa prevedibilità. Il disegno comprende
una mezza cadenza che apre l'aspettativa e una cadenza/accordo conclusivo che
può chiuderla. Sono raccolte tensione comportamentale e risposte EEG.

**Risultato utile.** Un accordo incongruente può mantenere alta la tensione
perché l'arco non si chiude, anche quando l'evento è già avvenuto. Il lavoro
separa tre forme discusse nella letteratura: tensione da sorpresa,
tensione da negazione della previsione e tensione da aspettativa/forward
directedness. La posizione della frontiera di frase determina se il nuovo
materiale continua un arco o ne avvia un altro.

**Trasferibilità.** Sono sequenze simboliche/pianistiche, tonalità occidentale
e ascoltatori esperti; non è un modello per mixed audio pop/rap. Non sono
pubblicati codice, checkpoint o dataset riutilizzabile; il paper Elsevier non
è verificato come open license. È però la fonte empirica più diretta per
separare continua, risolve e interrompe: una riduzione del proxy dopo
la giunzione non basta se la frase resta aperta.

### 5. Bigand e Parncutt (1999): stabilità locale e cadenza

**Riferimento.** Emmanuel Bigand e Richard Parncutt, “Perceiving musical
tension in long chord sequences”, *Psychological Research* 62, 237–254
(1999), doi:
[10.1007/s004260050053](https://doi.org/10.1007/s004260050053).
PDF integrale consultato:
[PDF](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/parncutt_richard/Pdfs/BiPa99_Tension.pdf).

**Metodo e risultato.** Musicisti e non musicisti valutano frammenti basati
su riduzioni armoniche del Preludio in Mi maggiore di Chopin. L'istruzione
operativa associa tensione alta a “serve continuazione” e tensione bassa a
“la sequenza potrebbe fermarsi naturalmente”. Le cadenze V–I producono
rilassamento sistematico; la tensione dipende in modo forte dalla cadenza
locale e da una finestra percettiva breve.

**Implicazione.** Una feature di arrivo deve avere evidenza di stabilità o
chiusura locale, non soltanto una pendenza negativa. Audio misto e rap
possono rendere la stima tonale incerta: il risultato corretto deve poter
essere unknown, non un falso positivo. Non sono stati trovati codice,
checkpoint o licenza di dataset.

### 6. Lerdahl e Krumhansl (2007): tensione tonale, attrazione e struttura

**Riferimento.** Fred Lerdahl e Carol L. Krumhansl, “Modeling Tonal Tension”,
*Music Perception* 24(4), 329–366 (2007), doi:
[10.1525/mp.2007.24.4.329](https://doi.org/10.1525/mp.2007.24.4.329).
È stato reperito il PDF degli autori:
[PDF](https://bussigel.com/lerdahl/pdf/Modeling%20Tonal%20Tension.pdf);
la trattazione concettuale e le sezioni sui quattro componenti sono state
consultate, mentre non è stata verificata ogni pagina del PDF in questa
passata.

Il modello combina struttura prolungazionale, pitch-space, tensione di
superficie e attrazione. Distingue attrazione/previsione da instabilità e
considera la segmentazione temporale e la cadenza come elementi di
partenza, tensing, ritorno e rilassamento. Esclude volutamente ritmo, tempo,
dinamica, testo e densità dal modello tonale.

È quindi una sorgente per feature di attrazione/stabilità, non una soluzione
completa per audio misto. Non ho verificato codice, checkpoint, dataset
riutilizzabile o licenza.

### 7. Herremans e Chew (2016): Tension Ribbons e spiral array

**Riferimento.** Dorien Herremans e Elaine Chew, “Tension Ribbons:
Quantifying and Visualising Tonal Tension”, *TENOR 2016*.
Preprint integrale consultato:
[PDF](https://www.dorienherremans.com/sites/default/files/paper_tenor_dh_preprint_small.pdf).

Il metodo calcola cloud diameter, cloud momentum e tensile strain nello
spiral array a partire da note/accordi simbolici. È collegato a giudizi di
forme di tensione raccolti da Farbood (2.661 risposte). Può fornire
tonal_stability o tonal_attraction se si dispone di una stima tonale
affidabile, ma non codifica da solo cadenza, frase, timbro, groove, ruolo
vocale o densità. L'input principale è MusicXML/simbolico, quindi il
trasferimento a mixed audio pop/rap è indiretto. La pagina del progetto
menzionata nel paper non è stata verificata come repository con licenza
riutilizzabile; non usare il metodo come checkpoint.

### 8. Granot e Eitan (2011): interazione di parametri dinamici

**Riferimento.** Roni Y. Granot e Zohar Eitan, “Musical Tension and the
Interaction of Dynamic Auditory Parameters”, *Music Perception* 28(3),
219–246 (2011), doi:
[10.1525/mp.2011.28.3.219](https://doi.org/10.1525/mp.2011.28.3.219).
È stato letto il full text della copia autoriale:
[copia](https://www.researchgate.net/publication/259730930_Musical_Tension_and_the_Interaction_of_Dynamic_Auditory_Parameters).

**Metodo.** 132 partecipanti, di cui 60 formati, ascoltano sequenze brevi
controllate che manipolano direzione/registro del pitch, loudness e tempo.
Il lavoro confronta una teoria basata su aspettativa e una basata su
intensità/arousal; analizza anche interazioni fra parametri.

**Risultati utili.** Le sequenze random o erratiche non generano
automaticamente alta tensione; un'intensificazione prevedibile può generare
più tensione di una serie che impedisce la previsione. Gli effetti di tempo,
registro e loudness non sono additivi e dipendono dal contesto. Un calo di
loudness o un cambio di tempo non può quindi essere interpretato da solo.

**Limiti.** Stimoli sintetici e brevi, senza gerarchia tonale/metrica completa;
validità ecologica limitata per pop/rap. Non sono stati trovati codice,
checkpoint o dataset con licenza riutilizzabile.

### 9. Farbood e Upham (2013): memoria lunga e contrasto fra frasi

**Riferimento.** Morwaread M. Farbood e Francis Upham, “Interpreting
expressive performance through listener judgments of musical tension”,
*Frontiers in Psychology* 4:998 (2013), doi:
[10.3389/fpsyg.2013.00998](https://doi.org/10.3389/fpsyg.2013.00998).
Testo integrale open access consultato:
[articolo](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2013.00998/full).

Il caso è un canto di Schubert con riduzione armonica. Il modello combina
dinamica, pitch, armonia, onset frequency e tempo con finestre attentive e
di memoria; la memoria armonica è molto più lunga (circa 20 s in alcune
analisi) delle finestre di dinamica/onset (circa 1–4 s). Il lavoro evidenzia
che il contrasto fra versi/frasi e la struttura contano. È CC BY, ma non
pubblica un checkpoint del modello; i risultati sono su un brano e su
materiale classico, quindi non vanno usati come soglie pop/rap.

È una ragione aggiuntiva per conservare contesti 4/8/16/32 beat e una
memoria strutturale più lunga, separando però tensione da chiusura.

### 10. Vande Veire e De Bie (2018): DJ automatico e drop

**Riferimento.** Arne Vande Veire e Tijl De Bie, “From raw audio to a
seamless mix: creating an automated DJ system for Drum and Bass”, *Journal
on Audio, Speech, and Music Processing* (2018), doi:
[10.1186/s13636-018-0134-8](https://doi.org/10.1186/s13636-018-0134-8).
Testo HTML integrale consultato:
[articolo](https://link.springer.com/article/10.1186/s13636-018-0134-8).

Il sistema usa beat/downbeat/struttura, cue points, crossfade, armonia,
stile, ritmo e presenza vocale su 160 tracce più 220 holdout; 18 fan DnB
valutano il sistema. Il genere viene esplicitamente modellato come
build-up → drop → segmento principale, con frasi di otto battute. Un
crossfade può lasciare il drop di B continuare l'energia di A, mentre una
transizione rilassata può interrompere intenzionalmente l'energia. L'energia
è stimata con RMS e non viene chiamata tensione psicologica.

Il codice è su [Bitbucket](https://bitbucket.org/ghentdatascience/dj) con
licenza AGPLv3 dichiarata nell'articolo. Il metodo è utile per l'idea
“build-up + arrivo ritmico/novità”, ma le assunzioni DnB (tempo, 4/4,
drop) non trasferiscono automaticamente a pop/rap. Un RMS basso o alto senza
evento strutturale non è una risoluzione.

### 11. Zehren, Alunno e Bientinesi (2022): cue point e anticipazione del drop

**Riferimento.** Martin Zehren, Carlo Alunno e Paolo Bientinesi, “Automatic
Detection of Cue Points for the Emulation of DJ Mixing”, *Computer Music
Journal* 46(3), 67–82 (2022), preprint:
[arXiv:2007.08411](https://arxiv.org/abs/2007.08411).
Sono stati letti il preprint integrale e i repository
[Automix](https://github.com/MZehren/Automix) e
[M-DJCUE](https://github.com/MZehren/M-DJCUE).

**Metodo/dataset.** Il sistema usa novelty e feature beat-sincrone di kick,
snare, hi-hat, loudness armonica/percussiva, CQT/PCP, con un kernel di
quattro battute. Il dataset contiene 134 brani EDM annotati da cinque
musicisti esperti; l'audio non è distribuito per copyright. Gli autori
riportano precision circa 86%, recall circa 49% e circa 95.8% di candidati
utilizzabili dopo valutazione soggettiva. Il codice è MIT; il dataset è GPL-3.0
ma non contiene l'audio originale.

Il lavoro osserva che il DJ può entrare alcune battute prima di una novità
per creare anticipazione. Questo supporta structural_arrival e
drop_preparation, non una conclusione semantica universale. È EDM, non
mixed pop/rap, e non etichetta continua/risolve/interrompe.

### 12. Kim et al. (2020): analisi di mix reali e struttura nativa

**Riferimento.** Injoon Kim e coautori, “A Computational Analysis of Real-World
DJ Mixes using Mix-To-Track Subsequence Alignment”, *ISMIR 2020*,
[arXiv:2008.10267](https://arxiv.org/abs/2008.10267).
Preprint integrale consultato; repository:
[djmix-analysis](https://github.com/mir-aidj/djmix-analysis).

Il lavoro allinea 1.557 mix, 13.728 tracce e 20.765 transizioni con
chroma/MFCC beat-sincroni e DTW invarianti a tempo/tonalità. Analizza cue-in,
cue-out, cue-mid, transizioni lunghe e contesto di 32 beat; trova picchi di
lunghezza ogni 32 beat, coerenti con frasi DJ. Il corpus è soprattutto
house/trance. Non fornisce labels di suspense, continuazione o risoluzione,
né una valutazione cross-track del tipo A naturale → B.

Il repository non mostra un file LICENSE verificabile nella copia consultata.
La fonte è utile per allineare e descrivere la continuazione nativa, non per
affermare che la continuazione originale sia la ground truth percettiva.

### 13. MERT (Li et al., 2023/ICLR 2024): embedding audio avanzato

**Riferimento.** Yizhi Li e coautori, “MERT: Acoustic Music Understanding
Model with Large-Scale Self-supervised Training”, arXiv:2306.00107,
[paper](https://arxiv.org/abs/2306.00107), codice ufficiale
[GitHub](https://github.com/yizhilll/MERT), checkpoint
[MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) e
[MERT-v1-330M](https://huggingface.co/m-a-p/MERT-v1-330M).
Sono stati consultati paper, codice e pagine dei checkpoint.

MERT è pretraining self-supervised su audio musicale, con modelli da 95M e
330M e valutazione su 14 task di music understanding. Il codice è Apache-2.0;
le pagine dei pesi riportano CC BY-NC 4.0. Non è addestrato per tensione,
transizioni o labels continua/risolve/interrompe; non esiste una garanzia che
un embedding vicino significhi continuazione o risoluzione. Il modello da
95M occupa circa 1.7 GB e quello da 330M circa 5.25 GB; la fattibilità CPU
per il progetto non è stata misurata.

È una possibile ablation avanzata su finestre beat, dopo un baseline
interpretabile. Non è una sostituzione della distinzione fra energia,
aspettativa e chiusura; la licenza non commerciale dei pesi va verificata
prima di qualunque distribuzione.

### 14. Huron (2006) e Meyer (1956): teoria dell'aspettativa

**Riferimenti.** David Huron, *Sweet Anticipation: Music and the Psychology of
Expectation*, MIT Press (2006), [pagina editore](https://mitpress.mit.edu/9780262083454/sweet-anticipation/).
Leonard B. Meyer, *Emotion and Meaning in Music*, University of Chicago Press
(1956), [record bibliografico](https://books.google.com/books?id=QXniAAAAMAAJ).

Per Huron è stata consultata la pagina editoriale e la descrizione del
modello ITPRA (Imagination, Tension, Prediction, Reaction, Appraisal); il
libro completo non era disponibile. Per Meyer sono stati letti due capitoli
in scansione:
[capitolo 1](https://acousticslab.org/RECA293/293Files/293Downloads/Meyer1956_EmotionMeaningMusicCh1.pdf)
e [capitolo 2](https://acousticslab.org/RECA293/293Files/293Downloads/Meyer1956_EmotionMeaningMusicCh2.pdf).
Le scansioni non sono state verificate come distribuzione ufficiale
open-license.

Meyer collega ambiguità e ritardo alla suspense, e considera comprensione
contestuale e chiusura necessarie perché l'accumulo abbia un rilascio
percepibile; la stessa deviazione può essere naturale o innaturale a seconda
dello stile e del contesto. Huron è utile per distinguere tensione
anticipatoria da reazione alla sorpresa. Nessuna delle due fonti fornisce
codice, checkpoint o una metrica pronta per audio pop/rap.

## La continuazione naturale di A come controllo osservativo

La proposta pratica di usare tre profili è ragionevole ma, allo stato della
ricerca, è una **ipotesi di progetto**, non una procedura validata da un
paper:

1. A_before: contesto che precede il punto di giunzione in A;
2. A_natural_after: ciò che la traccia A fa naturalmente dopo quel punto;
3. B_after: ciò che il candidato B fa dopo l'ingresso.

A_natural_after può essere usato offline per descrivere il vettore nativo di
A: direzione di energia, densità, frase, groove, ruolo vocale, tonalità,
novità e arrivo. Il confronto aiuta a vedere quando B abbandona un arco che A
stava rendendo leggibile. Non bisogna imporre che B copi A_after: una
transizione può risolvere con un colore diverso, continuare trasformando il
materiale o introdurre un contrasto intenzionale. Il futuro naturale di A
inoltre non è sempre ciò che l'ascoltatore aveva già previsto, soprattutto
quando il campione è breve o il brano cambia sezione.

Kim et al. supportano l'allineamento e la struttura nativa di tracce intere;
You et al. supportano la manipolazione sperimentale di continuazione e
chiusura. Nessuna delle fonti consultate valida direttamente un confronto
cross-track A_natural_after contro B_after. Occorre quindi raccogliere
labels e dichiarare il metodo come controllo osservativo, senza presentarlo
come ground truth o come validazione psicologica.

## Percorso implementativo immediato, CPU e interpretabile

L'obiettivo immediato è aggiungere un profilo diagnostico indipendente dal
ranking esistente. Non servono MIDI o accordi gold e non serve vietare
categorie vocali.

### Profilo a domini separati

Per finestre beat-sincrone già disponibili (4/8/16/32 beat, con downbeat e
confidenza) calcolare e conservare:

- energy_*: log-loudness/RMS, crest, dinamica e pendenza. Questo dominio
  descrive intensità, mai “tensione”;
- event_*: onset/event density, spectral flux, roughness, regolarità e
  variazioni di kick/snare/hi-hat quando stimabili;
- pitch_tonal_*: chroma/key stability, altezza/pitch contour, attrazione o
  strain tonale con confidence;
- phrase_structure_*: downbeat/phrase boundary, periodicità 4/8/16/32,
  novelty, pause/rest/fade e salienza di un cambio;
- role_timbre_*: cambiamenti timbrici, presenza vocale/rap, ruolo
  ritmico/strumentale e densità. La presenza rap è contesto, non una
  penalità globale;
- confidence_*: qualità e disponibilità di ogni dominio. Se la stima
  tonale è debole in un mix, il risultato deve essere unknown_tonal e non
  una finta cadenza.

Da Farbood/TenseMusic si può ricavare un profilo multifeature con finestra
attentiva e memoria, nominato tension_trend_observable. La sua pendenza è
una misura di moto percepibile, non una label di arrivo.

### Tre evidenze indipendenti

Per ogni transizione produrre risultati separati:

**continuation_evidence.** Quanto B_after preserva o sviluppa la direzione
di A_before: trend multifeature, posizione nella frase, ritmo/event density,
ruolo e progressione del segmento. La somiglianza deve essere distribuzionale
e direzionale, non “audio uguale”.

**closure_evidence.** Evidenza di arrivo: confine di frase/downbeat
coerente, stabilizzazione o attrazione tonale con confidence, ritorno di
motivo/ruolo, pausa o sustain con funzione strutturale, oppure
build-up → evento/drop metricamente allineato con ritmo/densità ristabiliti.
Chroma similarity da sola non entra come chiusura.

**interruption_evidence.** Divergenza dopo un build-up o una direzione
esplicita, assenza di closure_evidence, e mancata continuità di frase/ruolo/
ritmo. Va interpretata come candidata a interruzione, non come giudizio
estetico universale: un contrasto intenzionale può essere voluto.

Il sistema deve emettere unknown quando la confidence strutturale/tonale è
insufficiente. resolution_likely è ammessa solo quando closure_evidence
ha supporto indipendente; la caduta di tension_trend_observable può
accompagnarla, ma non sostituirla.

### Regola specifica per build-up e drop

Un drop è un candidato release quando il build-up è osservabile, la nuova
sezione arriva su una posizione metrica/frasale plausibile e il profilo
ritmico/event density viene ristabilito o trasformato in modo saliente. Una
semplice riduzione di RMS, un silenzio o un cambio timbrico non bastano.
Questa regola è ispirata alla letteratura EDM/DJ, ma deve essere attenuata
fuori da EDM: pop e rap possono risolvere tramite frase vocale, groove,
pausa, risposta ritmica o stabilizzazione, senza un drop EDM.

### A/B osservazionale

Per ogni cue offline registrare i profili A_before, A_natural_after e
B_after, normalizzati per traccia e tempo. Confrontare:

- direzione e durata dei trend;
- posizione dei downbeat e confini;
- densità/eventi e ritorno del groove;
- stabilità/attrazione tonale con confidence;
- ruolo vocale/timbrico e presenza di risposta;
- evidenza di arrivo rispetto all'assenza di arrivo.

Il test non deve chiedere “B è uguale ad A_after?”, ma “B continua,
risolve, interrompe o contrasta in modo comprensibile l'arco che A_before
aveva aperto?”. Il caso Reminder → Too Many Nights resta un holdout
indipendente.

## Percorso avanzato

Solo dopo il profilo interpretabile si può fare una ablation con embedding
audio congelato (per esempio MERT su finestre beat) e un classificatore o
ranker piccolo, con:

- split per brano/artista, non finestre casuali dello stesso brano;
- test separato per pop, rap, strumentale e cambi di ruolo;
- confronto feature ingegnerizzate, embedding e combinazione;
- misura della latenza/memoria su CPU e verifica della licenza CC BY-NC dei
  pesi MERT;
- modello calibrato soltanto su labels aggregate, mai sul singolo esempio.

MERT non contiene labels di tensione, risoluzione o DJ transition: un
guadagno nell'ablation sarebbe evidenza predittiva per questo dataset, non
prova che l'embedding “conosca” la suspense. I modelli TenseMusic/Farbood
restano diagnostici; Herremans/Chew e Lerdahl/Krumhansl restano sorgenti di
feature tonali con limiti audio.

## Labels e verifica con ascoltatori

La raccolta dovrebbe separare il pre e il post:

1. prima del cambio: buildup_before = none / rising / suspense;
2. dopo il cambio: continua / risolve / interrompe;
3. opzionali: contrasto_intenzionale, unknown, qualità globale e confidence;
4. energia/arousal in una domanda separata, così non diventa un sinonimo di
   tensione.

Le prove devono essere renderizzate con loudness e crossfade controllati,
randomizzate e ascoltate senza mostrare il nome dell'artista. Ogni cue deve
avere più candidati e più brani; includere il caso Reminder → Too Many
Nights, ma tenerlo fuori dal fit. Chiedere anche valutazioni A-alone e
A+B quando serve a distinguere ciò che l'ascoltatore porta avanti da ciò che
reagisce solo dopo l'ingresso.

Per un classificatore con consenso sufficiente si possono riportare macro-F1
o balanced accuracy per le tre labels; per il ranking, pairwise preference,
Recall@K o NDCG. Le unità di bootstrap e gli split devono raggruppare per
brano/cue/ascoltatore, evitando che finestre quasi identiche attraversino
train e test. Le soglie vanno fissate prima del test holdout. Il successo
specifico è ridurre i falsi “risolve” nel caso di abbandono e migliorare la
scelta di candidati continua/risolve, senza degradare i cambi intenzionali
e senza eliminare rap o strumentali.

## Raccomandazione per il ranking

Nella prima iterazione non cambiare i pesi del ranking. Esportare i tre
indicatori e usare l'audit per verificare:

- se continuation_evidence ordina meglio dei soli trend/chroma;
- se closure_evidence elimina falsi release da calo energetico;
- se interruption_evidence segnala il caso negativo senza penalizzare
  automaticamente il rap;
- se unknown evita decisioni arbitrarie quando chroma, downbeat o struttura
  sono poco affidabili.

Dopo labels indipendenti, confrontare baseline, baseline + traiettoria,
baseline + chiusura, baseline + entrambi e, separatamente, eventuale
embedding. Ogni modifica deve essere valutata su gruppi di brani e generi
tenuti fuori dal fit. Il caso dell'utente è un test qualitativo importante,
non una funzione obiettivo.

## Tabella di audit delle fonti

| Fonte | Full text / accesso | Input, labels, dataset | Codice, checkpoint, licenza verificata | Limite per questo progetto |
|---|---|---|---|---|
| Farbood 2012 | PDF autrice letto | MIDI/WAV classico; slider tensione; 207 file + 10 stimoli, 35 soggetti | Nessun codice/checkpoint verificato | Non classifica risoluzione; classica |
| Barchet et al. 2024 | HTML integrale letto | Audio; 6 feature; 38 brani classici, 30 soggetti, slider | Repo/GitHub e OSF pubblici; license file non verificato; nessun checkpoint | Tensione non transizione; pop/rap non validati |
| Lehne & Koelsch 2015 | HTML/PDF integrali | Modello teorico | Nessun codice/dataset; articolo CC BY | Teoria, senza algoritmo |
| You et al. 2021 | PDF integrale letto | Sequenze pianistiche di 14 accordi; 30 musicisti; EEG + ratings | Nessun codice/checkpoint/dataset; publisher non open verificato | Chiusura sperimentale tonale, non mixed audio |
| Bigand & Parncutt 1999 | PDF integrale letto | Riduzioni di Chopin; musicisti/non musicisti; tensione e continuazione | Nessun codice/checkpoint/licenza verificati | Cadence evidence simbolica/classica |
| Lerdahl & Krumhansl 2007 | PDF reperito; consultazione parziale | Modello tonale/giudizi su repertorio classico | Nessun codice/checkpoint/licenza verificati | Esclude ritmo, dinamica, timbro, testo |
| Herremans & Chew 2016 | Preprint integrale letto | Note/accordi, spiral array; 2.661 giudizi di forme | Pagina progetto citata, licenza non verificata | Feature tonali, non audio/closure completa |
| Granot & Eitan 2011 | Full text autoriale letto | 132 soggetti; pitch/loudness/tempo controllati | Nessun codice/checkpoint/dataset verificati | Stimoli sintetici brevi |
| Farbood & Upham 2013 | HTML integrale letto | Schubert; ratings di tensione; finestre memoria | CC BY; nessun checkpoint verificato | Un brano, classica |
| Vande Veire & De Bie 2018 | HTML integrale letto | 160 + 220 DnB; 18 fan; RMS/beat/struttura | Bitbucket AGPLv3 | Drop DnB, non tensione generale |
| Zehren et al. 2022 | arXiv integrale + repo | 134 EDM, 5 annotatori; cue/novelty | Automix MIT; M-DJCUE GPL-3.0; audio non incluso | Cue/drop, non continua/risolve/interrompe |
| Kim et al. 2020 | arXiv integrale + repo | 1.557 mix, 13.728 tracce, 20.765 transizioni | Repo senza license file verificato | Struttura nativa; nessuna label di suspense |
| MERT, Li et al. 2023/24 | Paper, codice e model cards letti | Pretraining audio; 95M/330M, 14 task | Codice Apache-2.0; pesi CC BY-NC 4.0 | Nessuna label di transizione; CPU da misurare |
| Huron 2006 | Pagina editore/metadata; libro completo non letto | ITPRA e teoria aspettativa | Nessun codice/checkpoint | Fonte teorica, accesso limitato |
| Meyer 1956 | Due capitoli in scansione; libro completo non letto | Teoria di aspettativa, suspense, cadenza | Nessun codice/checkpoint; licenza scansioni non verificata | Teoria storica, non modello audio |

## Sezioni e pagine consultate

Per rendere controllabile la profondità della lettura, questi sono i punti
verificati nei testi (quando il testo è numerato per sezioni, si riportano le
sezioni; quando la scansione è stabile, si riportano anche le pagine).

- **Farbood 2012:** abstract e introduzione; esperimenti 1 e 2; modello
  parametrico con attentional/memory window; risultati e tabella delle
  correlazioni. PDF, pp. 387–428.
- **Barchet et al. 2024:** abstract; *Introduction*; *Model generation*;
  *Automatic feature extraction*; *Slope estimation*; *Smoothing and tension
  prediction*; *Materials and methods*; *Model optimization*; risultati e
  limiti/discussione dell'articolo HTML.
- **Lehne & Koelsch 2015:** abstract; sezioni sul rapporto tensione/arousal,
  evento iniziale e spazio degli esiti, suspense e previsione; sezione sui
  limiti temporali del modello, HTML/PDF integrali.
- **You et al. 2021:** introduzione e tipologia di tensioni; ipotesi sulle
  mezze/full cadences; metodo e condizioni sperimentali; risultati EEG/
  ratings; discussione sull'arco non chiuso. PDF, pp. 1–2, 2–3 e 6–7.
- **Bigand & Parncutt 1999:** abstract; esperimenti 1 e 2; istruzioni
  “continuation/stop”; analisi delle cadenze e conclusioni, PDF integrale.
- **Lerdahl & Krumhansl 2007:** componenti di struttura prolungazionale,
  pitch-space, surface tension e attraction; segmentazione/cadenza; sezioni
  sperimentali del PDF autoriale reperito, con consultazione non pagina per
  pagina.
- **Herremans & Chew 2016:** spiral array; cloud diameter, cloud momentum e
  tensile strain; survey delle forme di tensione; preprint integrale.
- **Granot & Eitan 2011:** introduzione expectation/arousal; metodo con
  pitch/loudness/tempo; risultati sulle sequenze random e prevedibili;
  interazioni e limiti, copia autoriale integrale.
- **Farbood & Upham 2013:** modello attentional/memory; risultati per
  armonia, dinamica, onset e tempo; memoria lunga e contrasto fra frasi;
  discussione, HTML integrale.
- **Vande Veire & De Bie 2018:** modello DnB; segmentazione build-up/drop;
  selezione dei segmenti e transizioni; studio con fan e appendice/codice,
  HTML integrale.
- **Zehren et al. 2022:** definizione cue/drop; feature novelty beat-sincrone;
  dataset/annotazioni; protocollo e risultati precision/recall, preprint
  integrale.
- **Kim et al. 2020:** allineamento subsequence; cue-in/out/mid; estrazione
  beat/chroma/MFCC; analisi della lunghezza delle transizioni e struttura a
  32 beat, preprint integrale.
- **MERT:** abstract, architettura/pretraining, task di valutazione e model
  cards dei checkpoint; non ci sono sezioni su tensione o transizioni.
- **Huron 2006:** solo metadata/pagina editoriale e descrizione ITPRA; libro
  completo non accessibile.
- **Meyer 1956:** capitoli 1 e 2 nelle scansioni; in particolare le pagine
  del capitolo 1 sulla suspense/ritardo e quelle del capitolo 2 su
  aspettativa, contesto e cadenza; libro completo non accessibile.

## Tre priorità pratiche

1. **Separare le evidenze prima di ottimizzare.** Aggiungere un export
   diagnostico CPU con energy, event, phrase/structure, tonal confidence e
   role/timbre; calcolare tension_trend_observable in stile
   Farbood/TenseMusic, ma mantenere distinti continuation_evidence,
   closure_evidence e interruption_evidence. La priorità è eliminare il falso
   assioma “proxy in calo = risolto”.
2. **Usare il confronto A_before / A_natural_after / B_after come ipotesi
   osservativa.** Allineare i tre profili su beat e frasi, descrivere la
   traiettoria nativa di A e misurare se B sviluppa, chiude o abbandona
   quell'arco. Non imporre copia di A_after e non chiamare il controllo
   validazione psicologica finché non ci sono labels.
3. **Raccogliere labels cieche e validare fuori dal caso.** Per cue e brano
   raccogliere buildup_before e continua/risolve/interrompe, con energia/
   arousal separati e contrasto intenzionale opzionale. Tenere
   Reminder → Too Many Nights fuori dal fit; valutare per brano/artista e
   genere. Solo dopo questa verifica considerare MERT o un piccolo ranker
   addestrato su feature + embedding.

## Conclusione operativa

Il rilevatore deve cercare tre cose diverse: un moto che viene preparato,
un seguito che lo sviluppa e un arrivo che lo chiude. La prima può essere
stimata con finestre e trend multifeature; la seconda richiede confronto
direzionale fra A_before e B_after; la terza richiede evidenza di frase,
metro, stabilità/attrazione o evento strutturale. Quando manca la terza,
un calo del proxy è compatibile con interruzione e deve restare tale nel
report. Questo schema permette di trattare il caso Reminder → Too Many
Nights come negativo senza trasformarlo in un divieto sul rap e senza
confondere energia, sorpresa, suspense e risoluzione.
