# Revisione DJ: biblioteca, letteratura empirica e direzione del progetto

Data: 2026-10-20. Revisione in sola lettura: nessun file di codice è stato modificato e non è stato fatto alcun commit.
Legenda delle etichette di evidenza:
- **[E]** risultato empirico misurato dagli autori (corpus di mix reali, test d'ascolto con N dichiarato);
- **[S]** regola di progetto di un sistema pubblicato, valutata solo nel suo insieme;
- **[P]** pratica o linea guida dichiarata da DJ o annotatori, senza misura;
- **[H]** ipotesi di questa revisione, da verificare.

---

## 0. Sintesi in cinque righe

1. La letteratura empirica sui mix reali indica regole concrete di mestiere: transizioni lunghe multiple di 32 beat, tempo-stretch piccolo (86,1% delle tracce entro ±5%), quasi nessuna trasposizione, cue su downbeat a inizio frase, bass swap con EQ, niente sovrapposizione di due voci. Il renderer del progetto non applica nessuna di queste regole.
2. Le due transizioni V12 pubblicate sono cambi secchi (`handoff`, rampe di 12 ms, `time_stretch: none`). Secondo la misura interna `runs/dj_audit_v12/metrics.json`:
   - i cue cadono tra 140 e 1000 ms fuori da beat o downbeat;
   - B entra alla battuta 6 o 10 della sua frase di 16, cioè non a inizio frase;
   - il tempo differisce del 14–15%;
   - una coppia ha distanza Camelot 4–6;
   - una coppia ha un salto di loudness di 9,4 dB.

   Un DJ esperto scarterebbe queste coppie e questi punti prima di qualsiasi considerazione su tensione o sintassi.
3. Gli studi d'ascolto mostrano dove si perde qualità percepita. In DJtransGAN 2022 gli ascoltatori esperti attribuiscono i difetti alla scelta della coppia e dei cue, non alla curva di fade. In Vande Veire 2018 la selezione dei brani sposta il voto medio da 2,39 a 3,89.
4. Nessuno studio empirico trovato tratta le transizioni in mix hip-hop/rap. Questa è la vera lacuna da colmare con dati propri (§6, P2). La psicologia dell'aspettativa non la colma.
5. Direzione proposta, in quest'ordine:
   - **P0**: renderer DJ standard (beatmatch con stretch, cue su frase, EQ swap, loudness) e un gate di compatibilità della coppia;
   - **P1**: cue da struttura `allin1` o CUE-DETR e tipi di transizione per il rap;
   - **P2**: statistiche apprese da mix reali (`djmix-dataset`, `djmix-analysis`, `djmixer-estimation`).

   I moduli psicologici vanno congelati come diagnostica opzionale.

---

## 1. Inventario della biblioteca e squilibrio

### 1.1 Fonti per sezione (da `_meta/SOURCES_MANIFEST.json`, i `sources.json` dei bundle V12 e `texts/`)

| Area | Fonti distinte | Testo archiviato (byte estratti) | Natura |
|---|---|---:|---|
| 01 psicologia aspettativa | Farbood 2012; Lehne & Koelsch 2015; Hansen et al. 2021; Turrell et al. 2025; Grosz et al. 2025 (solo abstract); Kurby & Zacks 2008; Sridharan et al. 2007 (abstract); Adams 2020 (duplicato) | 612.380 | teoria/psicologia sperimentale su toni, EDM, classica |
| 02 fraseggio rap/voce | MCFlow (Condit-Schultz 2016/2017); Adams 2020; Duinker 2020/2021; YIN 2002; Silero VAD; torchcrepe | 2.437.900 | musicologia del flow e strumenti DSP |
| 03 struttura/transizioni DJ | Vande Veire & De Bie 2018; Kim et al. 2020; Schwarz & Fourer 2019; M-DJCUE 2019; Zehren et al. 2020; CUE-DETR 2024 | 270.865 | mix reali e sistemi auto-DJ |
| 04 modelli | CUE-DETR; Zehren 2020; Raveform 2026; LinkSeg (solo README) | 130.316 | cue/struttura |
| 05 renderer | MosaikBox (Sowula & Knees 2024); "Daito Manabe 2026" | 85.681 | auto-DJ con stem e diagnostica |
| 06 valutazione | TenseMusic 2024; SongEval 2025; Music Flamingo 2025; Beat This 2024; SSM-Net 2023; All-In-One 2023; Medley2K 2020 | 336.847 | modelli e metriche |
| 07 benchmark | M-DJCUE (134 JAMS parsati); scheda Raveform | 83.385 | dati |

**Squilibrio misurato.**
- Psicologia, cognizione e musicologia del rap (sezioni 01–02) valgono circa **3,05 MB di testo**, contro circa **0,57 MB** di letteratura sul mestiere del DJ (03 + 05 + parti di 04/07): un rapporto di circa **5:1**.
- Lo squilibrio si ripete nel codice. I moduli su aspettativa, voce e sintassi (`expectation`, `functional_event`, `syntactic_completion`, `vocal_completion`, `vocal_boundaries`, `word_guard`, `trajectory`, `motion`, `phrase_evidence`, `transition_intent`…) contano **8.495 righe su 19.550**. I renderer (`blend_preview`, `musical_preview`, `intent_render`, `preview`) contano **1.212 righe**, senza EQ, senza time-stretch e senza modello di fade DJ.
- L'INDEX dichiara come conclusione operativa "protezione del fraseggio vocale, trend di tensione". Non nomina beatmatching, EQ o compatibilità tonale.

### 1.2 Problemi di qualità nella biblioteca

- **Kim 2020 riassunto a metà.**
  - In `03/RIASSUNTO.md` e in `v12_role_planning` il paper compare solo come "finestre di hit 15/30/60 s" e come definizione di cue-out e cue-in.
  - La sezione 6 del paper, cioè l'analisi musicologica di tempo, trasposizione, durata delle transizioni e accordo sui cue, è nel testo locale (`03/texts/kim2020_djmix.txt`, pp. 5–6) ma non è mai riportata.
  - Il manifest attribuisce il paper a "Jongpil Kim et al." con titolo "Automatic DJ Mix Analysis from Online Mixes". Gli autori corretti sono **Taejun Kim, Minsuk Choi, Evan Sacks, Yi-Hsuan Yang, Juhan Nam**, il titolo è quello ISMIR 2020 citato sotto.
- **Vande Veire 2018 dichiarato "bloccato, solo metadati"** in `v12_role_planning/research/RIASSUNTO.md`. Il full text però è presente in `03/texts/vandeveire2018.txt`. Le sue regole (tre tipi di transizione, bass/treble swap, vocal clash, armonia) non sono riportate da nessuna parte.
- **"Schwaab et al. 2019"** nel manifest è in realtà **Diemo Schwarz e Dominique Fourer**, CMMR 2019.
- **"Daito Manabe 2026, DJ Mix Research Institute"** non è una pubblicazione peer-reviewed verificabile: l'URL canonico non c'è e l'istituto non è identificato. Va marcata come letteratura grigia o non verificata.
- **MosaikBox** è archiviato e riassunto solo come "modifica di stem". Le sue regole di mix concrete non sono riportate:
  - normalizzazione a −14 LUFS;
  - stretch massimo ±8%;
  - transizione di 16 downbeat;
  - bass swap esteso agli alti;
  - attenuazione della voce uscente se le voci si sovrappongono per più di 2 s.

### 1.3 Regole DJ documentate nella biblioteca ma non implementate

| Regola (fonte già in biblioteca) | Tipo | Stato nel codice |
|---|---|---|
| Durata della transizione multipla di 32 beat, con picchi a 32/64/96… (Kim 2020 §6.3) | [E] | `DEFAULT_OVERLAP_SECONDS = 1.0`; V12 renderizzata come `handoff` con fade di 12 ms |
| Tempo-stretch per il beatmatch: 86,1% delle tracce entro 5% (Kim 2020 §6.1); limite ±8% (MosaikBox) | [E]/[S] | `TIME_STRETCH_POLICY = "none"` (`intent_render.py`) |
| Key lock attivo, trasposizione rara (Kim 2020 §6.2); tonalità compatibili per quinta o relativa, ±1 semitono con pitch shift (Vande Veire §4.2.1) | [E]/[S] | nessun modulo di tonalità; nessun gate Camelot |
| Bass/treble swap al punto di switch, fade lineare del volume (Vande Veire §4.3; MosaikBox §3.4) | [S] | nessun filtro EQ in `src/` |
| Switch sul downbeat a inizio periodo di 4 battute (Zehren 2020 Rule 2); cue su frasi di 16/8 battute (CUE-DETR) | [P]/[E] | cue V12 fuori dal downbeat di 140–1000 ms; B a metà frase |
| IN nell'intro fino al primo beat della sezione core; OUT non prima del core (M-DJCUE) | [P] | non usato come vincolo |
| Vocal clash: scartare candidati con 2 o più battute vocali sovrapposte (Vande Veire §4.2.4); attenuare la voce uscente se la sovrapposizione supera 2 s (MosaikBox) | [S] | esiste solo una word guard di 80 ms per i tagli; nessuna attenuazione di stem nel render |
| Loudness di entrambi i deck allineata (−14 LUFS, MosaikBox) | [S] | solo match RMS locale limitato (±6/+3 dB); salto misurato di 9,4 dB |
| Tre tipi di transizione con macchina a stati: double drop, rolling, relaxed (Vande Veire Tab. 3) | [S] | non presente; il progetto usa le intenzioni "continuation/arrival" |

---

## 2. Letteratura empirica su mix reali e sistemi auto-DJ

Ogni voce è stata verificata sulla fonte originale in questa sessione. Tra parentesi quadre: cosa è stato letto davvero.

1. **Kim, Choi, Sacks, Yang, Nam (2020).** *A Computational Analysis of Real-World DJ Mixes using Mix-To-Track Subsequence Alignment.* ISMIR 2020.
   - URL: https://arxiv.org/abs/2008.10267
   - Codice: https://github.com/mir-aidj/djmix-analysis
   - Letto: full text locale, pp. 1–6.
   - Dataset: 1.557 mix da 1001Tracklists, 13.728 tracce, 20.765 transizioni, "mostly House and Trance", tempo medio 127 BPM.
   - Il dataset originale non è pubblico secondo Raveform 2026.
2. **Kim, Yang, Nam (2021).** *Reverse-Engineering the Transition Regions of Real-World DJ Mixes using Sub-band Analysis with Convex Optimization.* NIME 2021.
   - DOI: https://doi.org/10.21428/92fbeb44.4b2fc7b9
   - Codice: https://github.com/mir-aidj/transition-analysis
   - Letto: abstract e README; il full text pubpub ha restituito 403.
   - Il metodo stima le curve di guadagno per sotto-banda. Batte il crossfade lineare e l'analisi a banda singola nella ricostruzione; test d'ascolto con N = 14.
3. **Kim, Yang, Nam (2022).** *Joint Estimation of Fader and Equalizer Gains of DJ Mixers using Convex Optimization.* DAFx 2022.
   - PDF: https://mac.kaist.ac.kr/pubs/KimYangNam-dafx2022.pdf
   - Codice: https://github.com/mir-aidj/djmixer-estimation
   - Dataset: https://github.com/mir-aidj/djmix-dataset
   - Letto: full text.
   - Nuovo dataset MixesDB: **5.040 mix, 50.742 transizioni, 52.910 tracce uniche, 6.672 ore**.
   - Modello: fader più EQ a 3 bande (low shelf, mid peak con Q = 3, high shelf), con curve monotone (A scende, B sale).
4. **Kim, Kim, Kim, Nam (2026).** *Raveform.* TISMIR 9(1):131–143.
   - DOI: https://doi.org/10.5334/tismir.288
   - Letto: pagina dell'articolo e testo locale.
   - Dati: 4.902 mix, 56.873 tracce uniche, 1.423 annotate con struttura funzionale e metrica, 53.780 transizioni.
   - Segmento più frequente: 16 battute (circa 30 s a 128 BPM). Cambi a multipli di 4 o 8 battute, usati come linea guida di annotazione [P].
   - Non riporta statistiche su dove cadono i cue rispetto alle sezioni.
5. **Kim & Nam (2023).** *All-In-One Metrical and Functional Structure Analysis…* WASPAA 2023.
   - URL: https://arxiv.org/abs/2307.16425
   - Codice: https://github.com/mir-aidj/all-in-one (MIT)
   - Restituisce beat, downbeat, BPM e sezioni funzionali (intro/verse/chorus/bridge/outro…). Un adapter esiste già in `src/musicmix/pretrained_structure.py`.
   - Dello stesso gruppo: **DJ StructFreak**, ISMIR 2023 LBD (https://ismir2023program.ismir.net/lbd_328.html), un auto-DJ centrato sulla scelta del mix point tramite embedding di struttura; solo demo, senza valutazione.
6. **Vande Veire & De Bie (2018).** *From raw audio to a seamless mix: creating an automated DJ system for Drum and Bass.* EURASIP JASMP 2018:13.
   - DOI: https://doi.org/10.1186/s13636-018-0134-8
   - Codice: https://github.com/lenvdv/dnb-autodj-3
   - Letto: full text locale.
7. **Chen, Hsu, Liao, Martínez Ramírez, Mitsufuji, Yang (2022).** *Automatic DJ Transitions with Differentiable Audio Effects and Generative Adversarial Networks* (DJtransGAN). ICASSP 2022.
   - URL: https://arxiv.org/abs/2110.06525
   - Demo e codice: https://paulyuchen.com/djtransgan-icassp2022/
   - Letto: full text.
8. **Zehren, Alunno, Bientinesi.**
   - M-DJCUE, ISMIR 2019 LBD: https://github.com/MZehren/M-DJCUE
   - *Automatic Detection of Cue Points for the Emulation of DJ Mixing*, arXiv 2020 (https://arxiv.org/abs/2007.08411), poi Computer Music Journal 46(3), annata 2022, pubblicato 2023. Verificato da hpac.cs.umu.se; il testo CMJ non è stato letto.
   - Un report UMINF 24.08 del 2024, "Towards automatic DJ mixing", risulta solo dall'elenco del profilo; non letto.
9. **Argüello, Lanzendörfer, Wattenhofer (2024).** *Cue Point Estimation using Object Detection* (CUE-DETR). ISMIR 2024.
   - URL: https://arxiv.org/abs/2407.06823
   - Codice e pesi: https://github.com/ETH-DISCO/cue-detr (MIT), https://huggingface.co/disco-eth/cue-detr
   - Dataset EDM-CUE: 4.710 tracce, circa 21k cue di 4 DJ.
10. **Sowula & Knees (2024).** *MosaikBox: Improving Fully Automatic DJ Mixing Through Rule-based Stem Modification and Precise Beat-Grid Estimation.* ISMIR 2024. Letto: full text locale, §3.4 e §4.
11. **Schwarz & Fourer.**
    - *UnmixDB*, ISMIR 2018 LBD: https://hal.archives-ouvertes.fr/hal-02010431 ; Zenodo: https://zenodo.org/record/1422385
    - *Methods and Datasets for DJ-Mix Reverse Engineering*, CMMR 2019: https://hal.science/hal-02172427
    - Sono mix sintetici, quindi non adatti a studiare il gusto umano.
12. **Sonnleitner, Arzt, Widmer (2016).** *Landmark-Based Audio Fingerprinting for DJ Mix Monitoring.* ISMIR 2016, pp. 185–191 (https://ir.webis.de/anthology/2016.ismir_conference-2016.24).
    - Contiene il dataset **Mixotic**: 10 mix, 118 tracce CC, con confini annotati.
13. **Gebhardt, Davies, Seeber (2016).** *Psychoacoustic Approaches for Harmonic Music Mixing.* Applied Sciences 6(5):123 (https://doi.org/10.3390/app6050123).
    - Letto: abstract e record. La pagina MDPI ha restituito 403.
    - La roughness calcolata su pochi parziali predice consonanza e piacevolezza del mix meglio della sola regola "key match" del software DJ. La tabella comparativa non è stata letta.
14. Altri lavori, citati solo per completezza e non letti:
    - Bittner et al. (2017), *Automatic Playlist Sequencing and Transitions*, ISMIR 2017, pp. 442–448: solo metadati;
    - Faber et al. (2020), *Medley2K*: 2.000 medley e 7.712 transizioni, generi misti, in biblioteca;
    - Smith (2013), *Hip-Hop Turntablism, Creativity and Collaboration*: libro;
    - Katz (2012), *Groove Music*: libro.

    Il dataset **GiantSteps** (key/tempo, EDM) non è stato verificato in questa sessione.

---

## 3. Regole e numeri empirici

| Tema | Risultato | Tipo | Fonte |
|---|---|---|---|
| **Durata della transizione** | L'istogramma delle durate in beat ha **picchi ogni 32 beat** (8 battute, circa 15 s a 127 BPM), con code fino a oltre 200 beat. "DJs consider the repetitive structures… when they make transitions". | [E] | Kim 2020 §6.3, Fig. 6 |
| | Double drop: fade-in di 16 e fade-out di 32 misure. Rolling: 16 e 16. Relaxed: 16 e 16. | [S] | Vande Veire 2018 §4.2.3 |
| | 16 downbeat (8 prima e 8 dopo il punto di transizione) | [S] | MosaikBox 2024 §3.4 |
| | Regione fissata a 8 battute | [S] | DJtransGAN 2022 §3.1 |
| **Confine percepito** | Gli annotatori umani collocano il confine soprattutto al cue-in di B: è il più vicino nel 52% dei casi, contro 30% per il cue-out e 18% per il cue-mid. Il disaccordo umano sui confini è di circa 9 s di deviazione standard. | [E] | Kim 2020 Tab. 3, §5 |
| **Tempo-stretch** | **86,1%** delle tracce adattate entro il 5%, **94,5%** entro il 10%, **98,6%** entro il 20% (distribuzione a doppia esponenziale centrata su 0). | [E] | Kim 2020 §6.1 |
| | Limite di stretch ±8% | [S] | MosaikBox |
| | Coppie con differenza di al massimo 5 BPM, poi stretch e pitch-shift su B | [S] | DJtransGAN |
| | Stretch separato per componente armonica e percussiva (HPSS + phase vocoder/OLA) per ridurre gli artefatti | [S] | Vande Veire §4.3 |
| **Tonalità** | Solo **2,5%** di 24.202 tracce trasposte; di queste il **94,3%** di un solo semitono. In pratica il key lock è attivo. | [E] | Kim 2020 §6.2 |
| | Ammesse stessa tonalità, quinta su o giù, relativa; ±1 semitono con pitch shift | [S] | Vande Veire §4.2.1 |
| | Differenza di al massimo 2 semitoni | [S] | DJtransGAN |
| | La roughness predice la consonanza percepita del mix | [E] | Gebhardt 2016 (abstract) |
| **Cue rispetto alle frasi** | Accordo tra DJ diversi sulla stessa traccia: **23,6%** a deviazione zero, **40,4%** entro 1 misura, **73,6%** entro 8 misure, **86,2%** entro 16. | [E] | Kim 2020 §6.4 |
| | Switch sul downbeat del primo bar di un periodo di 4 battute; alta novelty; la sezione dopo lo switch deve reggersi da sola. Precisione 86% e recall 49% con tutte le regole su M-DJCUE. | [P]+[E] | Zehren 2020 |
| | Cue spesso su confini di frase, frasi dominanti di 16 e 8 battute; F1 massimo 0,46 | [E] | CUE-DETR 2024 |
| | Cue IN nell'intro fino al primo beat della sezione core; OUT non prima del core; 1.849 eventi su 134 tracce | [P] | M-DJCUE (parse locale) |
| **EQ e fader** | Durante il fade-in, B entra senza bassi e alti. Allo switch bassi e alti passano di colpo a B e vengono tolti ad A, poi A sfuma. Bassi e alti di una sola traccia per volta. | [S] | Vande Veire §4.3 |
| | Bass swap esteso agli alti; mid di A ridotti se i timbri sono diversi | [S] | MosaikBox |
| | Curve monotone di fader e EQ a 3 bande, stimabili su mix reali (5.040 mix MixesDB) | [E] (metodo) | Kim DAFx 2022 |
| | La GAN su EQ e fader non è significativamente diversa da crossfade lineare o regole; vince solo l'umano | [E] | DJtransGAN (136 partecipanti validi su 188) |
| **Voci** | Vocal clash se si sovrappongono almeno 2 misure vocali (1 è tollerata). Il detector SVM previene il 69,3% dei clash e scarta per errore il 16,5% dei non-clash. | [S]+[E] | Vande Veire §4.2.4 e §5 |
| | Se le voci si sovrappongono per più di 2 s, attenuare la voce uscente tramite stem HT-Demucs. La rimozione di stem "significantly enhances the perceived quality" (30 partecipanti). | [S]+[E] | MosaikBox |
| | Regole diverse per V→V, NV→V, V→NV, NV→NV | [S] | DJtransGAN (baseline "Rule") |
| **Loudness** | Entrambe le tracce normalizzate a −14 LUFS prima della transizione | [S] | MosaikBox |
| **Selezione della coppia** | Selezione per stile: voto medio 3,89 contro 2,39 (18 fan DnB). I commenti degli esperti attribuiscono la qualità a coppia e cue sbagliati più che al mixer. | [E] | Vande Veire Tab. 4; DJtransGAN §4 |
| **Tipi di transizione per genere** | DnB: double drop, rolling, relaxed, con matrice di Markov (es. dopo rolling: 0,8 rolling e 0,2 relaxed) | [S] | Vande Veire Tab. 3 |
| | **Hip-hop/rap: nessuno studio empirico trovato** su durata, tipo o posizione delle transizioni. I corpora esistenti sono House, Trance, EDM e DnB. | lacuna | ricerca di questa sessione |
| | Pratica hip-hop/open-format: transizioni brevi, cioè cut sull'"uno" del nuovo brano, blend di 4–8 battute su intro o strumentale, echo/filter out, spinback, drop sul ritornello. Tempo spesso gestito in half/double-time. | [H] (pratica, non misurata) | conoscenza pratica, non verificata |

**Cosa i numeri non dicono.**
- Tutte le statistiche [E] vengono da EDM o DnB a 4/4 molto regolare.
- Il rap del pool (Travis Scott, Drake, The Weeknd, Pop Smoke, Metro Boomin) ha tempi da 70 a 120 BPM, intro brevi e voce quasi continua. Durate di 32 beat e intro mixabili non sono garantite.
- Le regole di forma (cue su downbeat e frase, beatmatch, tonalità, niente doppia voce, niente salto di loudness) non dipendono dal genere. La durata e il tipo di transizione invece vanno ricalibrati sul rap [H].

---

## 4. Confronto con l'approccio attuale

### 4.1 Cosa la letteratura supporta di quanto fatto
- Distinguere cue-out, cue-in e cue-mid, cioè uscita e ingresso asimmetrici. Lo supportano Kim 2020 e M-DJCUE.
- Evitare parole tagliate e sovrapposizioni di voci. Lo fanno Vande Veire, MosaikBox e DJtransGAN. Il progetto però lo fa a livello di parola (80 ms), mentre i sistemi pubblicati lo fanno a livello di battuta o di secondi e con attenuazione degli stem.
- Usare modelli pre-addestrati ufficiali (Beat This, Demucs, `allin1`, CUE-DETR) con provenienza tracciata.
- Separare diagnostica tecnica e giudizio. Il controllo di loudness e clipping è coerente con MosaikBox.

### 4.2 Errori e lacune
1. **Il renderer non fa un mix DJ.**
   - Fa un cambio secco con rampe di 12 ms oppure un overlap di 1 s, senza stretch e senza EQ.
   - Nei mix reali la transizione dura tipicamente 8–16 battute o più, con beatmatch ed EQ swap.
   - Nessun lavoro sul quando o sul perché può compensare un renderer che non sincronizza i beat. Nella clip 2 il primo beat di B è sfasato del 44% di un beat rispetto alla griglia di A.
2. **Nessun gate di compatibilità della coppia.**
   - Coppie con tempo diverso del 15% e tonalità a distanza Camelot 4–6 sono state pubblicate. Nei mix reali solo circa il 5,5% delle tracce supera il 10% di stretch.
   - Gli esperti in DJtransGAN indicano proprio coppia e cue come causa principale della bassa qualità.
3. **I cue non sono vincolati a downbeat e frase.** Nelle release V12 i cue cadono a centinaia di ms dal downbeat e a metà frase per B. È la regola più robusta della letteratura (Zehren Rule 2; Kim §6.4; CUE-DETR).
4. **Clip d'ascolto di 20 s (10 + 10 s, regola dell'utente in `AGENTS.md`).** Una transizione DJ di 16 battute a 100 BPM dura circa 38 s e non entra nella clip. Una di 4 battute (circa 9,6 s) entra appena. Con la regola attuale si possono quindi ascoltare solo cut e blend brevi [H, conseguenza aritmetica]. Cambiare la durata delle clip è una decisione che spetta all'utente.
5. **La psicologia dell'aspettativa è stata usata come motore di selezione.**
   - Le fonti 01 sono su toni, classica ed EDM, e i loro stessi riassunti dicono che non forniscono soglie.
   - La validazione su 8.400 candidati dà un gate "vicino al livello di rumore" (`IMPLEMENTAZIONE_V11.md`, `INTEGRAZIONE.md`).
   - Il costo è alto: circa 8,5k righe di codice e circa 3 MB di testo. Il beneficio misurato è nullo.
6. **La biblioteca ha letto male le fonti DJ** (§1.2). I numeri utili di Kim 2020, Vande Veire e MosaikBox erano già in locale ma non sono mai entrati nei riassunti né nel codice.
7. **Manca un baseline DJ standard** contro cui confrontare. Nessuna release confronta il sistema con una semplice transizione "alla Vande Veire": beatmatch, frase, 8–16 battute, bass swap.

---

## 5. Direzione concreta, orientata al risultato

### P0: rendere ogni transizione tecnicamente da DJ (prima di qualsiasi altro lavoro)
1. **Beatgrid e downbeat** da Beat This (già nel pool V12) oppure da `allin1` (adapter già presente). Validare a orecchio su 7 brani con un click sovrapposto, prima di usarli.
2. **Tempo**:
   - portare B al tempo di A con stretch di qualità: `pyrubberband`/Rubber Band, oppure HPSS più `pytsmod` come in Vande Veire;
   - rampa di tempo durante la transizione, oppure stretch costante su B;
   - limite ±8% (MosaikBox), considerando le equivalenze half/double-time (70 ↔ 140) [H: tipico del trap];
   - oltre il limite: solo cut sull'"uno" o echo-out, mai blend.
3. **Gate di compatibilità**:
   - tempo come sopra;
   - tonalità entro Camelot ±1 o relativa (Vande Veire, Kim), con pitch shift opzionale di ±1 semitono;
   - in alternativa o in aggiunta, la roughness di Gebhardt 2016 sulla finestra di overlap.
4. **Cue**:
   - OUT di A e IN di B sempre sul downbeat del primo bar di un periodo di 4/8/16 battute;
   - IN preferibilmente all'inizio di una sezione `allin1` (intro, verse, chorus);
   - OUT non prima della fine del contenuto core di A (M-DJCUE).
5. **Mixer**:
   - volume e EQ a 3 bande (low shelf, mid peak, high shelf con i parametri di Kim DAFx 2022);
   - bass swap sul downbeat dello switch (Vande Veire, MosaikBox);
   - loudness dei deck a un target comune (−14 LUFS, MosaikBox).
6. **Voci**: se gli stem vocali di Demucs si sovrappongono per più di 2 s o più di 1 battuta, attenuare la voce di A oppure scegliere un tipo cut (MosaikBox, Vande Veire).
7. **Clip d'ascolto**: con la regola dei 20 s (10 + 10) testare prima cut e blend di 4 battute centrati sul cambio. Per i blend da 8–16 battute serve che l'utente decida un'eccezione alla durata delle clip (proposta: transizione intera più 4 battute per lato).
8. **Baseline**: questo renderer "Vande Veire/MosaikBox per rap" diventa il riferimento B0 in ogni A/B futuro.

### P1: tipi di transizione per rap e candidati di cue
- Usare tre o quattro template parametrizzati da durata in battute, curve di volume ed EQ, trattamento della voce:
  - **cut on the one**: A esce alla fine di una frase, B entra al downbeat di inizio sezione, eventualmente con echo-out o reverse su A;
  - **blend breve**: 4–8 battute su intro o strumentale di B o outro di A, con bass swap;
  - **blend lungo**: 16 battute, solo con BPM e tonalità compatibili e senza voci sovrapposte;
  - **drop sul ritornello**: B entra sul chorus.

  La scelta dipende da compatibilità, stato vocale e sezioni [H, da validare con ascolto].
- Candidati di cue: confini di sezione `allin1` (già integrato) più CUE-DETR (MIT, pesi su HuggingFace, addestrato su EDM: usarlo come proposta, non come oracolo).
- Ordinamento semplice e interpretabile: allineamento di frase, compatibilità BPM e tonalità, assenza di clash vocale, energia della sezione (Vande Veire, Zehren).

### P2: imparare dai mix reali, compreso il rap (la lacuna vera)
- Usare `djmix-dataset` (5.040 mix MixesDB, scaricabile con il pacchetto `djmix`), `djmix-analysis` (subsequence DTW per estrarre cue, durate e stretch) e `djmixer-estimation` (curve di fader ed EQ). Con questi strumenti si costruiscono **le statistiche che mancano**.
- Filtrare o raccogliere mix **hip-hop/rap/open-format**, ad esempio dai tag di genere MixesDB o 1001Tracklists, se presenti [H: non verificato]. Misurare:
  - la distribuzione delle durate in battute;
  - la quota di cut rispetto ai blend;
  - la posizione dei cue rispetto alle sezioni `allin1`;
  - lo stretch applicato;
  - la sovrapposizione delle voci.
- Questo darebbe numeri [E] per il rap al posto di soglie inventate.
- Fine-tuning di CUE-DETR o di un selettore sui cue estratti dai mix rap reali (Raveform e EDM-CUE mostrano un degrado cross-dataset: serve un dominio proprio).
- DJtransGAN o la stima di Kim come modello delle curve EQ/fader. È utile ma ha priorità minore: la GAN non batte il crossfade lineare in modo significativo.

### P3: cosa congelare
- `functional_event`, `syntactic_completion`, i comparatori di tensione e TenseMusic vanno spostati in diagnostica opzionale, fuori dal gate di selezione, finché non mostrano un vantaggio su B0 in A/B.
- `word_guard` e lo stato vocale restano utili come vincoli, insieme alla regola sugli stem vocali.
- Correggere i metadati della biblioteca: Kim 2020, Schwarz & Fourer, stato di Vande Veire, "Daito Manabe".

### Criterio di successo (pratico)
Ascolto A/B cieco, sulle stesse coppie, di B0 (P0) contro la release V12 e contro una transizione fatta a mano da un DJ in un software standard. In parallelo, metriche oggettive per ogni transizione:
- errore di fase dei beat (in ms);
- errore di tempo (in %);
- cue sul downbeat a inizio frase (sì/no);
- distanza Camelot;
- salto di LUFS;
- secondi di sovrapposizione vocale.

Queste metriche sono già in parte calcolate da `runs/dj_audit_v12/measure.py`, una misura interna non validata, basata su librosa e Krumhansl. Le transizioni che falliscono le metriche oggettive non vanno sottoposte all'ascolto.

---

## 6. Limiti di questa revisione
- Della NIME 2021 sono stati letti solo abstract e README; di Gebhardt 2016 e del CMJ di Zehren solo abstract e metadati; Bittner 2017 non è stato letto.
- Nessuna fonte empirica sul DJing hip-hop è stata trovata con le ricerche fatte. Esiste letteratura etnografica (Katz 2012, Smith 2013) che non è stata letta. Le regole hip-hop qui sono quindi marcate [H].
- I numeri della misura interna `dj_audit_v12` (BPM, tonalità, downbeat) derivano da stimatori automatici e possono contenere errori di half/double-time. Ad esempio, nella clip 2 il BPM di A risulta 70 e quello di B 120.
