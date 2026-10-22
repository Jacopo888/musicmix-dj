> Copia integrata. Leggere prima [le correzioni dell’orchestratore](../REVIEW_CORRECTIONS.md); gli originali restano nella cartella dello swarm.

# Quando cambiare traccia: competenza pratica su frase, cue, drop, energia e mix rapido open-format

Data: 2026-10-19
Agente: `dj_practice_books`
Ambito: `runs/swarm_v9_research/agents/dj_practice_books`

## 1. Scopo e posizionamento rispetto al contesto

Questa ricerca riguarda la competenza dei DJ su **quando** effettuare il cambio traccia: frase mixing, scelta dei cue, drop swap, flusso di tensione/energia, mixaggio rapido hip-hop/open-format, pickup/anacrusi e sovrapposizione vocale.

Tutte le fonti di questo pacchetto sono **NUOVE** rispetto a `context/research_v8/`. Il V8 copre psicologia, MIR, auto-feedback, validazione di critic e allineamento del canto, ma **non** copre la craft pratica del DJ. Qui non si ripete il V8; si aggiunge la parte mancante: regole decisionali osservabili su mix-in start, handoff/dominanza, mix-out end e arrivo di frase.

Vincoli rispettati: nessun ascolto, nessuna etichetta audio, nessuna modifica a `context/`, solo scritture in `output/` e `sources/`, nessun training sui pesi a partire da `context/HUMAN_FEEDBACK.md` (che resta diagnostica held-out). Il sistema attuale (10 s A + 10 s B, rampe sequenziali da 12 ms, nessuna correzione di tempo/pitch, policy basata soprattutto su matching/continuazione) viene trattato come punto di partenza da correggere, non come base validata.

## 2. Fonti e profondità di lettura

19 fonti sostanziali. Il dettaglio completo e le URL sono in `output/sources.json`; i file salvati e gli hash sono in `sources/`.

| id | tipo | lettura |
|---|---|---|
| `ni_cue_points` | tutorial ufficiale Native Instruments | testo completo |
| `ni_phrase_mixing` | tutorial ufficiale Native Instruments | testo completo |
| `ni_transitions` | tutorial ufficiale Native Instruments | testo completo |
| `serato_cues_jordan` | blog ufficiale Serato, workflow pratico | testo completo |
| `serato_flips` | tutorial ufficiale Serato | testo completo |
| `serato_cue_support` | manuale ufficiale Serato | testo completo |
| `rekordbox_phrase_guide` | manuale ufficiale AlphaTheta (PDF, 8 pagine) | testo completo |
| `rekordbox_faq` | FAQ ufficiale AlphaTheta | sezioni selezionate |
| `djstudio_phrasing` | guida vendor/educator DJ.Studio | testo completo |
| `dtt_phrasing_2009` | tutorial practitioner DJ TechTools | testo completo |
| `dtt_phrasing_2014` | tutorial practitioner DJ TechTools | testo completo |
| `ddt_dropmix` | tutorial practitioner Digital DJ Tips (open-format) | testo completo |
| `ddt_five_transitions` | capitolo di libro *Rock the Dancefloor* (Phil Morse) | capitolo completo, legalmente libero |
| `beattiming` | guida educator BeatTiming | testo completo |
| `mik_next_song` | workflow ufficiale Mixed In Key | testo completo |
| `mik_power_block` | tutorial ufficiale Mixed In Key | testo completo |
| `mik_energy` | capitolo di libro *Beyond Beatmatching* (Mixed In Key) | capitolo completo, legalmente libero |
| `zehren2020` | paper accademico EDM-only | testo completo, 7 pagine |
| `vandeveire2018` | paper accademico open access DnB-only | sezioni selezionate, 21 pagine |

Due fonti di libro sono capitoli interi pubblicati legalmente dagli autori/editori (*Rock the Dancefloor*, *Beyond Beatmatching*). Non è stato letto alcun libro completo in modo non autorizzato.

## 3. Tassonomia pratica: intenti di transizione e opportunità di confine

### 3.1 Intenti

| intento | cosa deve fare il cambio | evidenza principale |
|---|---|---|
| **sustain / hold** | mantenere il groove e la sensazione, cambiare traccia senza cambiare direzione | Mixed In Key descrive esplicitamente "hold the groove"; Vande Veire ha transizioni a basso contrasto |
| **lift / arrival** | far arrivare il drop/hook di B in un punto di payoff; l'energia sale di poco | Mixed In Key raccomanda un passo di energia; Zehren richiede salienza; drop swap 16/32 battute |
| **contrast** | cambio marcato e intenzionale, non graduale | Mixed In Key elenca il contrasto come scelta legittima; Power Block usa salti rapidi |
| **reset / release** | togliere energia per far respirare; può usare un breakdown | Mixed In Key vieta breakdown-to-breakdown e high-to-intro come default, ma ammette il reset intenzionale |
| **quick turnover** | cambiare in fretta, anche cambiando BPM/genere, senza beatmix lungo | Digital DJ Tips dropmix; Mixed In Key Power Block 30-60 s; Native Instruments cita hip hop/DnB/open format |
| **continue / new section** | far iniziare a B una nuova sezione mentre A esce | Native Instruments: fine chorus verso verse; BeatTiming: B inizia quando A inizia l'outro |

### 3.2 Opportunità di confine

1. **Downbeat / inizio di frase**: punto canonico per far entrare B. Ricorre in Native Instruments, Serato, Digital DJ Tips, Mixed In Key, DJ TechTools.
2. **Drop / arrivo**: il drop di B deve coincidere con un cambio strutturale di A. Vande Veire usa 16 misure prima del drop per il double drop; BeatTiming usa 16 o 32 battute.
3. **Fine del payoff (chorus/drop)**: DJ TechTools 2009 formula "respect the chorus": non tagliare prima del payoff e mixa fuori dal chorus.
4. **Breakdown / uscita dal breakdown**: non è un punto automatico. Beattiming e Mixed In Key avvertono che due breakdown insieme possono spegnere la pista, mentre A in breakdown e B in drop può essere un contrasto forte.
5. **Outro / intro strumentale**: regione sicura per sovrapposizione di batteria; il capitolo *Rock the Dancefloor* ammette intro/outro senza batteria se si sovrappone una parte drums-only dell'altra traccia.
6. **Onset vocale / pickup**: cue sulla voce per evitare due voci insieme; drop-on-four quando la voce inizia prima del "one".
7. **Fill In**: in rekordbox è un cambio improvvisativo fino a 4 beat alla fine di Intro, Up e Chorus; si perde se la frase viene modificata.
8. **Loop / mix section**: Serato Flip e i loop servono a estendere intro corti o con voce; è un workaround di produzione, non una regola live generale.

### 3.3 Tre timestamp, non uno

Zehren separa:

- **mix-in start**: B è udibile ma A è ancora dominante;
- **handoff / switch**: B diventa prevalente su A;
- **mix-out end**: A diventa inaudibile.

Nei tagli queste tre fasi possono essere istantanee. Questa decomposizione è la base per non confondere "ho iniziato a mixare" con "ho consegnato la traccia" e con "ho finito di uscire". Il sistema attuale non distingue questi momenti.

## 4. Regole decisionali con provenienza

| regola | provenienza | nota critica |
|---|---|---|
| Inizia B all'inizio di una frase di A | NI, Mixed In Key, DJ TechTools, *Rock the Dancefloor* | regola più condivisa; non dice quante battute |
| Cue su downbeat, voce, drop/chorus, outro, mix-out | NI, Serato, Mixed In Key, rekordbox | strumenti ufficiali; Serato consente 8 cue salvati nel file |
| Se la voce inizia prima del "one", drop sul "four" | Digital DJ Tips | gestisce il pickup; richiede rilevazione dell'onset vocale |
| Evita due voci guida sovrapposte | NI, Serato, Vande Veire | Vande Veire: clash se overlap >= 2 misure, 1 tollerata; previene il 69.3% dei clash ma scarta il 16.5% di crossfade non clashing |
| Non tagliare prima del payoff; mixa fuori dal chorus | DJ TechTools 2009, Mixed In Key | argomento forte contro policy solo-matching |
| Non mischiare high intensity con intro, breakdown con breakdown, breakdown con high intensity | Mixed In Key | ammette il reset intenzionale; non è una regola assoluta |
| Allinea il drop di B a un cambio di A, con anticipo di 16 o 32 battute | Vande Veire (16 misure), BeatTiming (16/32) | Vande Veire è DnB-only e fissa 175 BPM |
| Non forzare lunghezze di frase uguali | BeatTiming, DJ TechTools 2009 | 16 vs 32 o 64 funziona iniziando più tardi la traccia corta |
| Una sezione dopo lo switch deve essere saliente e reggersi da sola | Zehren (Rule 3) | salienza implementata come soglia di energia armonica, dichiarata approssimativa |
| Considera il destino di B, non solo l'uscita di A | Vande Veire, BeatTiming | "in time but out of phrase" è l'errore tipico |

## 5. Pickup, anacrusi e voce

- **Anacrusi / pickup**: Zehren osserva che per la presenza di anacrusi il primo beat del brano non è necessariamente l'inizio del periodo; stima l'offset di fase tra otto candidati e sceglie quello con massima novelty. Digital DJ Tips fornisce l'equivalente pratico: drop sul "four" quando la voce inizia prima della battuta.
- **Overlap vocale**: la regola pratica è cue dove entra la voce e dissolvenza della voce precedente. La versione quantitativa di Vande Veire tollera una misura di sovrapposizione e considera clash due o più misure. Il rilevatore è un baseline; il paper stesso lo indica come migliorabile.
- **Conseguenza per MusicMix**: la mancanza di ascolto non autorizza a inventare giudizi vocali. Si possono usare cue di onset e stem/feature vocali esistenti, con incertezza esplicita e astensione quando il segnale è ambiguo. Non si può affermare "closure" dalla sola assenza di voce.

## 6. Stay vs leave e destino congiunto

Nessuna fonte singola fornisce un algoritmo "resta o esci". La lettura incrociata sostiene però una precondizione esplicita:

- se A è in una regione di basso valore (breakdown, intro, outro strumentale) e B ha un arrivo valido, uscire presto è plausibile;
- se A ha un payoff pendente (chorus/drop non ancora consumato), tagliare prima è sconsigliato;
- se A è in una fase di salita (riser/build), abbandonarla senza un arrivo equivalente in B produce la sensazione di aspettativa tradita;
- B deve avere un confine o un arrivo usabile: "joint destination availability", non solo similarità.

Questa è una **inferenza** (claim `C25`), non una citazione diretta. Va implementata come vincolo, non come punteggio implicito.

## 7. Mix rapido open-format e doppio tempo

- **Dropmix**: cambio brusco ma a tempo; può cambiare drasticamente tempo o genere senza beatmix di otto battute (`ddt_dropmix`).
- **Power Block**: nuova traccia ogni 30-60 secondi, stesso key, cue al centro del brano, salto dell'intro (`mik_power_block`).
- **Cut on the one**: Native Instruments attribuisce a hip hop, drum'n'bass, open format e urban DJ l'uso di mix più corti e del taglio sul downbeat.
- **Doppio tempo**: 85 BPM con 170 BPM è l'esempio pratico; Serato Flip crea sezioni half/double time. Va trattato come ramo esplicito, non come regola generale, perché può distruggere la continuità vocale se applicato a materiale rap/pop con hook lunghi.

Il sistema attuale ha un solo formato (10 s + 10 s): non rappresenta né il taglio né il blend lungo. Le raccomandazioni chiedono modalità esplicite e giustificabili.

## 8. Contraddizioni e controesempi

1. **Lunghezza di frase**: Native Instruments e DJ.Studio dicono 8-16 battute; DJ TechTools 2009 dice 8/16/32 count; DJ TechTools 2014 dice 16/32; Mixed In Key dice 32 beat; BeatTiming dice 8/16/32/64 e nega una regola universale. Conclusione: nessun valore fisso è giustificato.
2. **"Otto battute" di Native Instruments**: la stessa fonte, in altri articoli, usa 16/32 battute e 4/8 battute. La frase va trattata come un'ipotesi con confidenza, non come griglia.
3. **Breakdown**: non è automaticamente un punto di mix; alcune combinazioni sono deliberate, altre sono vietate.
4. **Energia**: Mixed In Key usa una scala 1-10 proprietaria e afferma essa stessa che è un aiuto alla scelta, non una garanzia. Non esiste validazione indipendente della scala.
5. **Modelli automatici**: Zehren è EDM-only, 99-147 BPM, periodicità di 4 battute, e dichiara che le sue regole sono sufficienti ma non necessarie. Vande Veire è DnB-only, 160-190 BPM, mix a 175 BPM fisso, frase multiplo di 8 misure. Nessuno testa rap/pop, stem simultanei o voce principale sovrapposta in stile rap.

## 9. Limiti, lead falliti e gap aperti

**Limiti**
- Nessun ascolto: le conclusioni sono testuali e strutturali, non percettive.
- Le fonti practitioner non hanno misure; le fonti accademiche sono limitate a EDM/DnB.
- `context/HUMAN_FEEDBACK.md` ha 4 osservazioni, in maggioranza "neither"/"tie", un solo "A", `confidence: null`: non permette alcuna generalizzazione.
- I cue vocali dipendono da rilevatori non validati sul materiale rap/pop del pool.

**Lead falliti**
- *How to DJ Right*: nessun testo completo accessibile legalmente; le anteprime non consentono di citare regole specifiche. Escluso dalle fonti, riportato come lead.
- *The Secret DJ* e *Last Night a DJ Saved My Life*: nessun estratto craft utile trovato legalmente.
- Vande Veire su Springer/imec: bloccato da `Client Challenge` HTML e da link `request-a-copy`; risolto con la copia repository UGent (`sources/vandeveire2018.pdf`).
- File falliti conservati come evidenza: `sources/vandeveire_html.raw` (3 KB, Client Challenge), `sources/ugent_page.raw`, `sources/vandeveire_imec.raw`.

**Gap aperti**
- Nessun dataset pubblico di transizioni rap/pop annotate per "quando cambiare".
- Nessun test dei metodi su stem simultanei o su voce parlata/rap con pickup lunghi.
- Nessuna metrica validata di "closure" o di qualità percepita per il pool di quattro brani.
- Nessuna etichetta vocale per *Reminder*, *Too Many Nights*, *5% TINT*, *Creature*.

## 10. Implicazioni per MusicMix

Il sistema deve smettere di scegliere il punto di cambio come effetto collaterale del matching. Le raccomandazioni operative sono in `output/RECOMMENDATIONS.json`; in sintesi:

1. rappresentare ogni transizione come tre timestamp con boundary tipizzati;
2. scegliere prima l'intento (hold, lift, contrast, reset, continue, quick turnover);
3. aggiungere un guard vocale/pickup con incertezza calibrata e astensione;
4. vincolare la transizione alla presenza di un arrivo valido in B;
5. bloccare i falsi positivi di closure (silenzio, novelty, RMS calante, solo beat alignment);
6. introdurre modalità esplicite cut / quick-mix / sustained blend;
7. esporre confidenza e astensione per traccia e per transizione;
8. costruire un critic offline autonomo che non usi la diagnostica umana per aggiornare i pesi.

Il criterio di accettazione trasversale è: nessuna affermazione di qualità musicale non dimostrata; ogni decisione mostra confidenza, motivo e possibilità di astensione.

## 11. File prodotto

- `output/sources.json`: 19 fonti con autori, anno, URL, tipo, profondità di lettura, sezioni lette, claim più forte, limiti, trasferibilità a rap/pop, licenza, codice/dati e risorse di calcolo.
- `output/claims.json`: 26 claim con `source_ids`, ancore e tipo (`evidence`/`inference`/`proposal`), esplicitamente NEW rispetto al contesto.
- `output/RECOMMENDATIONS.json`: 8 raccomandazioni prioritizzate con observation -> evidence -> mechanism -> experiment -> failure controls -> acceptance test.
- `sources/`: copie locali delle fonti con hash SHA256, più i file falliti dei lead non accessibili.
