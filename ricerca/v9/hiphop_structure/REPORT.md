> Copia integrata. Leggere prima [le correzioni dell’orchestratore](../REVIEW_CORRECTIONS.md); gli originali restano nella cartella dello swarm.

# Struttura rap/trap, frasi di flow e decisione di cambio traccia

Rapporto di ricerca per MusicMix, agente `hiphop_structure`. Data: 2026-10-19.
Perimetro: quando ha senso uscire da A e entrare in B in un pool rap/trap/pop (Reminder, Too Many Nights, 5% TINT, Creature), con render attuale 10 s A + 10 s B, rampe sequenziali da 12 ms, nessuna correzione di tempo/tonalita. Il report distingue evidenza nuova da cio che era gia in `context/`.

---

## 0. Risposta operativa breve

1. **Non esiste un "punto giusto" unico sul downbeat.** Le frasi di flow possono iniziare in levare (pickup), durare 2, 3, 4, 5+ beat, attraversare cicli in 3/4 o 5/4, e finire prima, su o dopo il beat 4. Un sistema che taglia sulla griglia dei downbeat taglia dentro le frasi per costruzione. (`S_mcflow_diss`, `S_ohriner2013`, `S_duinker2020`.)
2. **La chiusura di frase non e un evento di energia.** Non e dimostrata da allineamento al beat, silenzio, novelty spettrale o RMS calante. La chiusura e un'aspettativa realizzata: il flow ha costruito un bersaglio e lo raggiunge, oppure proietta chiaramente una continuazione. (`S_duinker2020` PR2; `S_adams2020`; `S_ohriner2019`.)
3. **Meglio astenersi che forzare.** L'accordo umano sui confini strutturali e buono ma non esatto (boundary F 0.69-0.82 con tolleranza 0.5-3 s) e crolla tra scale diverse (0.38-0.44); i migliori modelli audio arrivano a F 0.46 a +/-0.5 s. Un mixer autonomo deve produrre candidate ordinate con incertezza e astenersi nei conflitti. (`S_salami2011`, `S_ullrich2014`, `S_nieto2020`.)
4. **La decisione di cambio traccia e asimmetrica.** In A devo valutare se la frase conclude o proietta continuazione; in B devo valutare se l'ingresso e un inizio di frase, un pickup o un'anacrusi che si risolve dopo. Le due domande non vanno valutate con la stessa regola. (`S_duinker2020`, `S_ohriner2019`, `S_adams2020`.)

---

## 1. Metodo

- Lettura full-text di fonti primarie di teoria/corpus rap (MCFlow, Adams, Duinker, Ohriner, Wadsworth-Needle), di dataset/valutazione MIR (SALAMI, Ullrich, Nieto) e di letteratura su respiro (Nakano).
- Fonti pratiche specifiche hip-hop/open-format (Native Instruments, We Are Crossfader, Mixgraph, Songen trap arrangement) come supplemento, non come evidenza sperimentale.
- Copie legali sotto `sources/` con URL e SHA-256; nessun bypass di paywall o challenge. Le fonti non recuperate sono elencate come lead falliti e non contano come full-text.

---

## 2. Cosa definisce una frase nel flow rap (NEW)

### 2.1 Prosodia e declinazione di pitch

Condit-Schultz (`S_mcflow_diss`, capitoli 3.2.2-3.2.4) descrive l'unita prosodica prototipica come un salto verso una zona alta del registro seguito da una declinazione progressiva; le sottounita terminano con pitch relativamente alto e la frase completa con il pitch piu basso. Esistono anche "terminal shapes", contorni brevi e ripidi che delimitano le frasi. Ritmo e pause o allungamenti finali contribuiscono alla segmentazione. Questo e il meccanismo principale per cui la fine di frase non coincide necessariamente con un evento percussivo: e un contorno vocale.

### 2.2 La sintassi batte la rima

Adams (`S_adams2020`, sezioni [1.7]-[1.9] e [3.1]-[3.4]) mostra che unita sintattiche possono violare rima e metro e creare frasi che contengono rime e attraversano le battute. La rima, da sola, e uno dei criteri piu deboli: supporta un confine gia esistente ma non lo impone contro la sintassi. Duinker (`S_duinker2020`, SR3) arriva alla stessa conclusione, pur mostrando che in altri casi rima e cesura ritmica possono contraddire la sintassi. Conseguenza pratica: un taglio che spezza un'unita grammaticale e rischioso anche se cade su una rima o sul beat 4.

### 2.3 Motivo, armonia del beat, parametri multipli

Adams (`S_adams2020`, [2.1]-[2.24], [4.1]-[4.3]) propone tre parametri di frase in hip-hop: armonia del beat, sintassi lirica, motivo ritmico/melodico. Il beat puo definire la frase tramite lunghezza del loop, numero di armonie e direzionalita (loop di 1, 2 o 4 misure; chiusura armonica). Un beat con loop lungo e progressione forte puo prevalere sulle divisioni liriche; un beat ripetitivo cede alle liriche. L'ascoltatore puo entrare in entrainment con uno dei due layer. Ne segue che la "frase" non e un oggetto unico: e un compromesso tra parametri, e va trattata come ipotesi multipla.

### 2.4 Dentro il verso: tensione, sottosezioni, ambiguita

Wadsworth e Needle (`S_wadsworth2022`) modellano la tensione metrica in base a posizione, periodicita e prevalenza delle rime: rime piu rapide e in posizioni meno stabili sono piu tese; catene regolari sono piu rilassate. Cambi di tensione suggeriscono sottosezioni di verso e possono produrre ambiguita tra verso, chorus e categorie ibride. Questo e rilevante per scegliere un ingresso in B: non tutti i confini "verse/hook" sono netti, e alcune sezioni sono deliberatamente ibride.

---

## 3. Perche i downbeat fissi e i micro-gap tagliano le frasi (NEW)

### 3.1 Pickup e anacrusi

Il flow prototipico ha circa una frase per misura, ma la frase puo iniziare come pickup al beat 1 e finire sul beat 3, oppure iniziare dopo il beat 1 e finire sul beat 4 (`S_mcflow_diss`, capitolo 3.2.4). Un ingresso in levare e normale, non un errore. Duinker (`S_duinker2020`, capitolo 6.4.2) chiama "anticipation end rhymes" le rime finali che cadono prima del beat 4: l'anacrusi che segue e un esito della rima anticipata, non una dislocazione da correggere. Ohriner (`S_ohriner2013`) mostra densita di fine frase/linea vicino al beat 4 ma con coda significativa, e flussi anacrusici/pickup che convivono con il metro duplo.

### 3.2 Cicli in 3/4 e 5/4 e frasi cross-ritmiche

Sempre in Condit-Schultz (`S_mcflow_diss`, capitolo 3.2.4 e Figura 3.11) compaiono esempi di phrasing in 3/4 e 5/4 in OutKast, Biggie e Jay-Z. Ohriner (`S_ohriner2013`) studia un caso in cui il beat afforda sia un ciclo di quattro sia uno di tre beat, e analizza le rime a tre beat. Un rilevatore di frase vincolato a una griglia dupla perde queste strutture.

### 3.3 Micro-timing: 25-60 ms non sono rumore

Ohriner (`S_ohriner2019`, [2.1]-[2.5]) misura gli onset sillabici al p-center e stima la soglia di differenza per intervalli tra 25 e 40 ms; nel caso analizzato, uno scarto di 25-60 ms e percettivamente rilevante. La differenza tra flow "speech-rhythmic" e "music-rhythmic" e proprio il grado di quantizzazione a una griglia di 16 posizioni. Tagliare sulla griglia ignorando micro-timing e phase shift cancella informazione espressiva e puo spostare la cesura percepita.

### 3.4 Dissonanza metrica che non chiede risoluzione

Duinker (`S_duinker2020`, capitolo 6.4) propone quattro relazioni: exact alignment, nested alignment, grouping non-alignment, displacement non-alignment. I primi due dominano; gli altri sono piu rari. Punto critico: nella musica hip-hop la "dissonanza" metrica non e instabile e non richiede risoluzione. Un flow disallineato dal beat puo essere stabile e coerente. Quindi l'assenza di allineamento al downbeat non e prova di tensione irrisolta, e la presenza di allineamento non e prova di chiusura.

---

## 4. Respiro vs gap di parola

Il respiro e un marker spesso insufficiente e a volte contraddittorio. Duinker (`S_duinker2020`, SR4) nota che i punti di respiro coincidono spesso con altre cesure, ma che la loro udibilita e limitata da editing e overdub, e che un MC puo rappare attraverso diversi possibili confini prima di respirare. Nakano et al. (`S_nakano2008`) mostrano che un rilevatore HMM di respiro su canto non accompagnato raggiunge recall 97.5% ma precision 77.7%: molti falsi positivi. Se questo vale su voce isolata e cantata, su rap over beat e ancora meno affidabile. I gap tra parole ASR misurano la consegna vocale, non la chiusura musicale della frase: un segmento ASR include pause interne e non puo da solo contraddire uno stem vocale a bassa energia (questo era gia emerso nel contesto, `context/docs/history/13_STRUTTURA_E_FRASI.md`). Proposta: tre canali separati e deboli (respiro acustico, gap parola, completezza sintattica), mai un cancello duro.

---

## 5. Confini a piu livelli e incertezza (SALAMI, Ullrich, Nieto)

- **Accordo umano.** SALAMI (`S_salami2011`, Sezione 4, Tabelle 3-4): boundary F 0.69 a +/-0.5 s e 0.77 a +/-3 s tra due annotazioni large-scale; 0.73/0.82 tra small-scale; solo 0.38-0.44 tra scale/schemi diversi. "Boundary" dipende dalla scala e dal formato di annotazione.
- **Rilevazione automatica.** Ullrich et al. (`S_ullrich2014`): CNN su mel-spettrogrammi con F 0.46 a +/-0.5 s e 0.62 a +/-3 s; il contesto migliore e 16-32 s con smearing 1.5-6 s. Un baseline a confini regolari raggiunge gia circa 0.33 a +/-3 s. I confini audio automatici sono quindi grossolani, non verita.
- **Natura del problema.** Nieto et al. (`S_nieto2020`) sostengono che la struttura musicale e soggettiva, ambigua e gerarchica; piu annotazioni possono essere valide e sistemi application-dependent con output multipli sono preferibili a una singola ground truth.

Implicazione diretta: valutare i cue a piu tolleranze (almeno +/-0.5 s e +/-3 s) e a piu livelli (frase vs sezione), riportando l'incertezza invece di un punto secco.

---

## 6. Hook, verse e drop: fra teoria, pratica DJ e produzione trap

- La teoria rap vede i confini di verso/ritornello come ruoli formali, non come oggetti netti: Wadsworth-Needle (`S_wadsworth2022`) mostrano sezioni ibride e ambigue; Adams (`S_adams2020`, [1.13]) considera le frasi due misure come default che di per se non produce direzione/chiusura.
- La pratica DJ generalista (`S_ni2024`) definisce la frase come tipicamente 8-16 battute e raccomanda prep, conteggio, cue point ai confini di frase, beatmatch e allineamento mix-in/mix-out; il passaggio chorus-end -> verse-start e descritto come buon punto di taglio rapido.
- La pratica hip-hop/open-format (`S_crossfader2022`, `S_mixgraph`) insiste invece su transizioni brevi, cut, wordplay, acapella, hot cue verso hook/chorus/verse, con overlap tipicamente 4-8 battute o nullo; mescolare a lungo con due voci e descritto come confuso.
- La produzione trap (`S_songen2025`) propone spesso 8 battute di intro, 16 di hook, 16 di verse, 8 di bridge, con intro minimale che trattiene il beat pieno e hook che aggiunge layer. E un template, non un risultato misurato: va verificato per brano.

Sintesi: un prior di griglia 8/16 puo aiutare nei brani trap con struttura regolare, ma non va imposto. Hip-hop e open-format hanno intros spesso non DJ-friendly e frasi vocali che iniziano subito; la strategia corretta resta la rilevazione di frase con fallback di astensione.

---

## 7. Asimmetria A/B e contesto anticipatorio

Il contesto attuale usa 10 s per lato e una politica simmetrica di matching/continuazione. Le fonti suggeriscono tre asimmetrie:

1. **Chiusura vs ingresso in levare.** A puo concludere con una rima anticipata (prima del beat 4) e B entrare in pickup; la risoluzione arriva dopo. (`S_duinker2020`; `S_ohriner2019`.)
2. **Direzione vs allineamento.** La chiusura in A e un'aspettativa realizzata (sintassi, rima/cadenza, motivo, contorno terminale); l'ingresso in B e un punto di partenza metrico. Sono due domande separate. (`S_adams2020`; `S_duinker2020` PR2.)
3. **Continuazione di A vs arrivo in B.** Se A proietta continuazione, B deve consegnare un arrivo coerente entro le prime battute; se A conclude, B puo entrare anche solo con un pickup ben posto. (`S_ohriner2019`; `S_duinker2020`.)

---

## 8. Cosa NON dimostra chiusura

- Allineamento al beat o al downbeat: il flow e spesso volutamente non allineato e stabile (`S_duinker2020`, `S_ohriner2019`).
- Silenzio o caduta di RMS: possono essere breakdown, loop end o semplice spazio tra eventi; il respiro non e marker sufficiente (`S_nakano2008`, `S_duinker2020` SR4).
- Picco di novelty spettrale: e esattamente cio che un modello di confine generico rileva, ma con F 0.46-0.62 e su generi non rap (`S_ullrich2014`).
- Gap ASR tra parole: misura pause di consegna, non chiusura di frase; un segmento puo contenere pause interne (contesto, doc 13).
- Rima finale isolata: e il criterio piu debole se la sintassi continua (`S_adams2020` [1.7]-[1.9]).

---

## 9. Implicazioni concrete per MusicMix

Priorita P0:
- **R1 - Gate multi-evidanza con astensione.** Quattro sub-score (fine frase vocale, prior rima/cadenza, completezza sintattica, cesura/resto) + probabile astensione. Nessun cancello su singolo downbeat.
- **R2 - Finestre asimmetriche A/B e pickup-aware.** Uscita A valutata su conclusione/continuazione; ingresso B ammesso in una piccola finestra prima del downbeat, con offset registrato.
- **R5 - Test di chiusura con controlli negativi.** Separare "frase chiusa", "frase continua", "chiusa ma con attesa verso B", "sconosciuto"; vietare chiusura da sola energia.

Priorita P1:
- **R3 - Canali separati respiro/gap/sintassi come evidenza debole e calibrata.**
- **R4 - Output a due livelli: candidate di frase + contesto di sezione/hook**, senza etichettare euristicamente le sezioni come verse/hook.

Priorita P2:
- **R6 - Audit di trasferimento di genere.** Verificare se un prior 8/16 aiuta o peggiora per ciascuno dei quattro brani, con annotazioni multiple e casi di astensione.

---

## 10. Contraddizioni, limiti e lead falliti

Contraddizioni / tensioni tra fonti:
- La pratica DJ generalista dice "frase = 8-16 battute"; la ricerca rap mostra frasi di 1, 2, 3, 5+ beat e phrasing non standard. Non sono in conflitto necessariamente (scale diverse), ma un prior 8/16 va applicato solo al livello sezione, non a quello frase.
- Alcune guide trap presentano la struttura 8/16 come standard; Duinker e Ohriner mostrano hypermetri irregolari (5, 6, 7 misure) e non-alignment stabile. Il template non e una legge.
- Adams: syntax > rhyme; Duinker: rima e cesura possono contraddire la sintassi. Conclusione: nessun singolo parametro vince sempre; serve pesatura per brano/contesto e astensione.

Lead falliti (non usati come full-text):
- MCFlow article PDF (emusicology.org) bloccato da bot challenge; usata la dissertazione.
- Ohriner, Flow (OUP) capitolo 2: HTTP 403.
- Waller 2019: il download era un HTML di redirect, non PDF.
- Duinker 2021/2022 Current Musicology "Song Form and the Mainstreaming of Hip-Hop": errore SSL; manca un tassello su hook/verse come sequenza core.
- Gomez 2023 (CUNY) e Kalasinskaite 2025 (MDPI): HTTP 403.
- vibesdj.io: HTTP 429.

---

## 11. Tabella fonti (copie locali in `sources/`, SHA-256 abbreviato)

| ID | Fonte | Anno | Tipo | Lettura | SHA-256 (prefisso) |
|---|---|---|---:|---|---|
| S_mcflow_diss | Condit-Schultz, MCFlow (dissertazione OSU) | 2016 | primaria corpus | full text, sez. scelte | `7ce4d37c13415ed1` |
| S_mcflow_article | Condit-Schultz, MCFlow (EMR) | 2016 | primaria | solo metadati (bloccato) | `c28a50157ac30413` (challenge HTML) |
| S_adams2020 | Adams, Parameters of Phrase in Hip-Hop (MTO 26.2) | 2020 | primaria teoria | full text | `dae19fafdeb04503` |
| S_adams2009 | Adams, Metrical Techniques of Flow (MTO 15.5) | 2009 | primaria teoria | full text | `01e420c48dfd5ff8` |
| S_duinker2020 | Duinker, Diversification and Post-Regionalism (McGill) | 2020 | primaria dissertazione | sez. scelte (cap. 6) | `eef6b9631433d60f` |
| S_ohriner2013 | Ohriner, Metric Ambiguity and Flow (EMR) | 2016 | primaria corpus | full text | `5ffcbac4276ee5c5` |
| S_ohriner2019 | Ohriner, Lyric, Rhythm and Non-Alignment (MTO 25.1) | 2019 | primaria teoria | full text | `8c3d0b84f3704850` |
| S_wadsworth2022 | Wadsworth e Needle, Rhyme, Metrical Tension (Integral 35) | 2022 | primaria teoria | full text | `e6aabce87c42d1fb` |
| S_salami2011 | Smith et al., SALAMI annotations (ISMIR) | 2011 | primaria dataset | full text | `27940683db5e15c1` |
| S_ullrich2014 | Ullrich et al., Boundary Detection CNN (ISMIR) | 2014 | primaria MIR | full text | `fa4dd9fda2817ef7` |
| S_nieto2020 | Nieto et al., Audio-Based Music Structure Analysis (TISMIR) | 2020 | primaria review | sez. scelte | `c5e72aa5c561ca62` |
| S_nakano2008 | Nakano et al., Breath Sounds in Singing (ICMPC) | 2008 | primaria | full text | `b20b22febefe6cc1` |
| S_ni2024 | Native Instruments, Phrase Mixing | 2024 | ufficiale/practitioner | full text | `bc33d84fcb225aeb` |
| S_crossfader2022 | We Are Crossfader, How To Mix Hip Hop | 2022 | practitioner | full text | `ce937465d8211738` |
| S_mixgraph | Mixgraph, Hip Hop Mixing Guide | n.d. | practitioner | full text | `4e5481b56d44ed0f` |
| S_songen2025 | Songen, How to Arrange a Trap Beat | 2025 | practitioner | full text | `13ccdcf70150b368` |

### NEW vs gia in context

Gia in `context/` (non rivendicato come nuovo): renderer a rampe sequenziali senza correzione tempo/tonalita; policy orientata al matching/continuazione; Beat This! small0 con sigmoid non calibrato; sezioni euristiche e non verse/hook; comportamento di Whisper/segmenti ASR e pause; limiti di Audiobox come critic; il fatto che l'ASR da solo non basta a promuovere tagli interni (`context/docs/history/13_STRUTTURA_E_FRASI.md`, `context/research_v8/`).

Nuovo in questo report: evidenza primaria su prosodia/terminal shapes e definizione di frase nel flow (MCFlow); varianza di lunghezza e posizione metrica di 7,889 frasi; phrasing 3/4 e 5/4; sintassi > rima e parametri multipli (Adams 2020); cinque regole di segmentazione e quattro tipi di allineamento, con non-risoluzione della dissonanza metrica (Duinker); p-center e micro-timing 25-60 ms, speech- vs music-rhythmic (Ohriner 2019); tensione da rima e sottosezioni di verso (Wadsworth-Needle); numeri di accordo umano e di modelli audio su confini (SALAMI, Ullrich, Nieto); precision/recall del respiro cantato (Nakano); prassi DJ hip-hop vs generalista e template trap (NI, Crossfader, Mixgraph, Songen).

---

## 12. Limiti e cosa resta da misurare

- Nessuna fonte audio valida un modello di confine su join sequenziali tra canzoni; Ullrich nomina la transition detection come task correlato ma non la sperimenta.
- Nessuna fonte misura i quattro brani specifici. Ogni trasferimento e un'inferenza.
- Le evidenze rap piu forti sono simboliche/musicologiche (MCFlow, Duinker, Adams, Ohriner), non modelli audio.
- I numeri SALAMI/Ullrich sono confini strutturali generici, non fine di frase vocale rap.
- Il respiro e studiato su canto isolato, non su rap mixato.
- Restano da chiarire: come definire operativamente "conclusione" per il flow; quanti cue validi esistono davvero nei quattro brani; se un prior 8/16 danneggi i brani non trap del pool; quale tasso di astensione sia accettabile per un DJ automatico.

Direzione consigliata: implementare R1, R2 e R5, con R3 e R4 come supporto, e usare R6 per decidere quanto del template trap e trasferibile. Ogni raccomandazione ha esperimento, controlli di fallimento e test di accettazione in `output/RECOMMENDATIONS.json`.
