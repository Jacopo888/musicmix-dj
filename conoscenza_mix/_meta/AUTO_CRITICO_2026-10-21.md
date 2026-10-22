# Auto-feedback dei mix: un critico automatico di transizioni e set

Data: 2026-10-21 (notte tra 20 e 21 ottobre). Obiettivo dell'utente: "trovare un modo per fare auto-feedback dei mix, senza il mio intervento; indaga a fondo".
Codice: `src/musicmix/dj_critic.py`, `tests/test_dj_critic.py`, `tools/run_dj_critic.py`. Run: `runs/dj_critic_v1/` (`REPORT.md`, `PREREGISTRATION.json`, `scores.json`, `validation.json`).

Etichette: **[E]** risultato empirico misurato dagli autori; **[S]** sistema o metrica pubblicata, valutata nel suo insieme; **[T]** teoria o rassegna; **[H]** ipotesi di progetto nostra; **[U]** giudizio dell'utente. "Letto" indica cosa è stato aperto davvero in questa sessione.

---

## 0. Sintesi

1. **Non esiste un critico validato per transizioni DJ**, e tanto meno per il rap. La ricerca non ha trovato dataset pubblici di voti umani su transizioni tra brani né metriche di "DJ mix quality" validate contro ascoltatori. Il lavoro più vicino (DJtransGAN 2022) usa un discriminatore solo per addestrare il generatore e valuta con ascolto umano. Williams et al. 2025 (già in biblioteca) dichiarano la qualità "highly subjective".
2. Esistono però **tre famiglie di misure preaddestrate** che rispondono a pezzi della domanda:
   - **predittori estetici per clip** (Audiobox Aesthetics 2025, SongEval 2025, MusicEval/AudioMOS 2025);
   - **modelli di aspettativa su audio** (surprisal / information content da modelli autoregressivi: Bjare et al. 2025; MusicGen usato come stimatore di sorpresa da Noguchi et al. 2026; "seam perplexity" di LoopGen 2025);
   - **embedding musicali** (MuQ 2025, MERT 2024, CLAP 2023) su cui addestrare discriminatori con negativi sintetici, come fa Neural Loop Combiner 2020 per la compatibilità di loop.
3. Nessuna di queste famiglie è stata validata sulle giunte tra brani. Il critico costruito qui le usa **come misure separate, ognuna con incertezza**, e le confronta onestamente con i pochi giudizi disponibili. La combinazione è stata dichiarata prima del confronto (`PREREGISTRATION.json`).
4. Risultati: §6 di questa nota e `runs/dj_critic_v1/REPORT.md`.

---

## 1. Domanda e vincoli

Domanda operativa: data una clip di transizione (o un set), produrre senza ascoltatore un punteggio utile a **scegliere o ottimizzare** la transizione, con un'incertezza dichiarata. Vincoli dal progetto:
- priorità al mestiere DJ (REVISIONE_DJ_2026-10-20): beatmatch, cue su frase, EQ/bass swap, tonalità, voci, loudness;
- integrità tecnica ≠ qualità musicale (AGENTS.md); i moduli psicologici restano diagnostica;
- giudizi umani: pochissimi e di un solo ascoltatore; non vanno usati per tarare e poi per validare (Cawley & Talbot 2010, già in biblioteca §5 della revisione set).

---

## 2. Fonti (verificate in questa sessione)

### 2.1 Predittori estetici appresi

**Tjandra, Wu, Guo, Hoffman, Ellis, Vyas, Shi, Chen, Le, Zacharov, Wood, Lee, Hsu (2025).** *Meta Audiobox Aesthetics: Unified Automatic Quality Assessment for Speech, Music, and Sound.* arXiv:2502.05139. Codice e pesi: https://github.com/facebookresearch/audiobox-aesthetics , HF `facebook/audiobox-aesthetics`.
- Letto: abstract e HTML (sezioni 2–3 e tabelle dei risultati).
- [S] Quattro assi: Production Quality (PQ, aspetti tecnici), Production Complexity (PC, numero di componenti), Content Enjoyment (CE), Content Usefulness (CU). Encoder tipo WavLM (12 layer, 768), 16 kHz mono, addestramento su crop casuali di 10 s, inferenza a finestre di 10 s. Circa 500 h, 97k clip (parlato, suoni, musica), 3 voti per clip, 158 annotatori selezionati.
- [E] Correlazione di Pearson per clip, musica naturale (AES-Natural, Tab. 7): PQ 0,887; PC 0,905; CE 0,750; CU 0,835. Su PAM-music (Tab. 4): 0,587 / 0,710 / 0,661 / 0,558. Gli autori notano prestazioni inferiori a PAM sulla musica "per dati non visti".
- Limiti per noi: misura la clip, non la giunta; PC non è una qualità da massimizzare; nessun dato su transizioni o rap.

**Yao, Ma, Xue, Chen, Hao, Jiang, Liu, Yuan, Xu, Xue, Liu, Xie (2025).** *SongEval: A Benchmark Dataset for Song Aesthetics Evaluation.* arXiv:2505.10793. Codice e testa: https://github.com/ASLP-lab/SongEval (revisione `848fb2ff…`, testa `ckpt/model.safetensors`, sha256 `0cb1eb8b…`), backbone MuQ-large-msd-iter.
- Letto: HTML (dataset, baseline, risultati).
- [E] 2.399 canzoni intere (circa 140 h, inglese e cinese, nove generi), voti 1–5 di annotatori con formazione musicale su cinque dimensioni: coerenza, memorabilità, naturalezza di respiro e fraseggio vocale, chiarezza della struttura, musicalità. Il numero di annotatori è riportato in modo incoerente (16 nell'abstract, 4 per brano nel §3.2).
- [E] Musicalità, livello utterance: baseline SSL su MuQ LCC 0,908, SRCC 0,893; UTMOS-based LCC 0,916. Per confronto, gli assi Audiobox correlano 0,61–0,65 (CE, CU, PQ) e 0,39 (PC) con la musicalità SongEval.
- Limiti per noi: addestrato su canzoni intere (in gran parte generate), non su finestre di 10 s né su giunte; dimensioni tra loro dipendenti (dichiarato dagli autori). Licenza CC-BY-NC-SA.

**MusicEval (2025)** arXiv:2501.10811 e **AudioMOS Challenge 2025** arXiv:2509.01336. Letto: risultati di ricerca e abstract. [E] 2.748 clip generate da 31 modelli, 13.740 voti di 14 esperti; baseline su CLAP. La sfida riporta che il CLAP score correla poco con la percezione umana. Non usato: dominio text-to-music, non transizioni.

### 2.2 Embedding musicali e distanze di distribuzione

- **Zhu et al. (2025).** *MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization.* arXiv:2501.01108; pesi `OpenMuQ/MuQ-large-msd-iter` (revisione `0562a578…`). Letto: abstract. [S] Predice token Mel-RVQ; stato dell'arte dichiarato su compiti di comprensione musicale. È il backbone di SongEval, quindi già nel progetto.
- **Li et al. (2024).** *MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training.* ICLR 2024, arXiv:2306.00107. Letto: abstract. [S] 95M–330M parametri, insegnanti RVQ-VAE e CQT, 14 compiti. Usato in v2: `m-a-p/MERT-v1-95M` @12af15fe (hidden state 6 e 12, 75 Hz); nel confronto per brano è l'embedding migliore per (c).
- **Wu, Chen, Zhang, Hui, Berg-Kirkpatrick, Dubnov (2023).** *Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation* (LAION-CLAP). ICASSP 2023, arXiv:2211.06687. Letto: abstract. Usato in v2: `laion/larger_clap_music` @a0b4534a (embedding audio di finestre prima/dopo/attraverso); da solo debole (AUC 0,57), utile insieme a MERT.
- **Kilgour et al. (2019)** *Fréchet Audio Distance* (arXiv:1812.08466) e **Gui, Gamper, Braun, Emmanouilidou (2024)** *Adapting Fréchet Audio Distance for Generative Music Evaluation*, ICASSP 2024, arXiv:2311.01616, toolkit `fadtk`. Letto: abstract e sintesi. [E] l'accordo di FAD con i giudizi dipende molto da embedding, dimensione del campione e insieme di riferimento; il FAD per brano può segnalare outlier. **Implica**: FAD confronta distribuzioni, non una singola transizione; usabile solo come diagnostica di un insieme di transizioni contro un insieme di riferimento (es. transizioni di DJ reali), che oggi non abbiamo [H].

### 2.3 Aspettativa e sorpresa

- **Huron (2006)** *Sweet Anticipation* [T] (già in biblioteca, solo recensioni): la sorpresa è uno dei meccanismi dell'esperienza musicale, non un difetto in sé.
- **Pearce (2018).** *Statistical learning and probabilistic prediction in music cognition: mechanisms of stylistic enculturation.* Annals of the New York Academy of Sciences 1423(1):378–395, DOI 10.1111/nyas.13654 (PMC6849749). Letto: abstract. [T]+[E rassegna] IDyOM (modello simbolico, variable-order Markov) simula aspettativa, segmentazione e altro come un unico processo predittivo.
- **Abdallah & Plumbley (2009).** *Information dynamics: patterns of expectation and surprise in the perception of music.* Connection Science 21(2–3):89–117. Letto: solo metadati. [T].
- **Bjare, Cantisani, Lattner, Widmer (2025).** *Estimating Musical Surprisal in Audio.* ICASSP 2025, arXiv:2501.07474, codice `SonyCSLParis/audioic`. Letto: HTML (metodo e risultati). [E] Transformer autoregressivo (GIVT, 12 layer) su latenti continui Music2Latent; information content (IC) = NLL del frame successivo. L'IC cala con la ripetizione su dati pop proprietari (−1,30, significativo), è più alto per i segmenti che compaiono più tardi, correla soprattutto con loudness (0,45–0,70) e flusso spettrale (0,38–0,56). Aggiunto all'inviluppo, migliora la predizione dell'EEG (Wilcoxon p = 0,025). **Non** riporta picchi di IC ai confini di segmento.
- **Bjare, Lattner, Widmer (2025).** *Estimating Musical Surprisal from Audio in Autoregressive Diffusion Model Noise Spaces.* ISMIR 2025, arXiv:2508.05306. Letto: abstract. [E] l'IC da modelli di diffusione autoregressivi eguaglia o supera GIVT nella sorpresa di altezza monofonica e nella **rilevazione di confini di segmento**.
- **Noguchi, Akama, Nakamura, Minamikawa, Polouliakh (2026).** arXiv:2603.03190. Letto: abstract e frammenti di ricerca. [E] usano sorpresa ed entropia **calcolate con MusicGen** come rappresentazione di aspettativa per pre-addestrare encoder EEG; notano che il pattern a ritardo dei 4 codebook complica la sorpresa congiunta.
- **Copet et al. (2023).** *Simple and Controllable Music Generation* (MusicGen). NeurIPS 2023, arXiv:2306.05284. Letto: §3.1 e dati (PDF). [S] EnCodec 32 kHz, 4 codebook RVQ da 2048, 50 Hz; pattern "delay"; modelli 300M/1,5B/3,3B; crop di 30 s; 20k h di musica licenziata; dropout della descrizione testuale 0,5 dopo il merge delle condizioni (quindi il modello ha visto molti esempi con condizione nulla).
- **LoopGen (2025).** *Training-Free Loopable Music Generation.* arXiv:2504.04466 v4. Letto: HTML. [S] propone la **seam perplexity** = perplessità di un modello generativo (MusicGen) sui token subito dopo la giunta di un loop. Il test d'ascolto (506 risposte, 18 sessioni) valida LoopGen contro una baseline, **non la metrica**: nessuna correlazione metrica–voto è riportata; la finestra W non è specificata; la metrica varia tra clip di 10 e 30 s senza differenze percepite.

**Implica.** La sorpresa di un modello audio autoregressivo è una misura plausibile di "quanto la giunta è inattesa per un modello addestrato su musica reale" [H], con precedente diretto (LoopGen) ma senza validazione percettiva della giunta. Va normalizzata contro la sorpresa naturale (la musica reale ha picchi di sorpresa ai cambi di sezione, Bjare 2025 ISMIR), quindi qui si usa uno z-score rispetto ai tratti A-solo e B-solo della stessa clip e un percentile rispetto a giunte naturali reali.

### 2.3b Misure del mestiere (v3)

- **Foscarin, Schlüter, Widmer (2024).** *Beat This! Accurate beat tracking without DBN postprocessing.* ISMIR 2024 (arXiv:2407.21658); checkpoint `beat_this-small0.ckpt` del progetto (lo stesso di `dj_qa`). Usato per la fase delle griglie A/B alla giunta [S].
- **ITU-R BS.1770-4 (2015)** loudness K-pesata, via `pyloudnorm` [S].
- **Krumhansl & Kessler (1982)** profili tonali (Psychological Review 89(4):334–368); qui distanza tra i vettori di correlazione con i 24 profili prima/dopo la giunta [S]. Non riletto in questa sessione.
- **Silero VAD** (Silero Team, pacchetto `silero_vad`, pesi inclusi): addestrato su parlato; sul rap con base non distingue la parola tagliata (dose non informativa) [S, limite di dominio].

### 2.4 Mixability e funzioni di costo (già in biblioteca, riletti per questo scopo)

- **Davies et al. (2014)** AutoMashUpper (TASLP): mashability da similarità armonica/ritmica e bilanciamento spettrale [S].
- **Bittner et al. (2017)** ISMIR: costo di transizione da timbro, chroma, loudness bassa, voce congiunta, fine su confine [S].
- **Gebhardt, Davies, Seeber (2016)**: roughness predice la consonanza del mix meglio del solo "key match" [E, abstract].
- **Chen, Smith, Yang (2020).** *Neural Loop Combiner: Neural Network Models for Assessing the Compatibility of Loops.* ISMIR 2020, arXiv:2008.02011. Letto: PDF (§4, §5). [E] Negativi sintetici "within-song" (reverse, shift, rearrange) e "between-song" (random, selected); CNN migliore in classificazione (accuratezza 0,57–0,63), Siamese migliore nel ranking; nel test d'ascolto (116 partecipanti, α = 0,778) **CNN + reverse** è preferito, anche alle combinazioni originali, e AutoMashUpper è il peggiore. Gli autori segnalano una **discrepanza tra metriche oggettive sui negativi sintetici e giudizio soggettivo**. È il precedente metodologico diretto del critico (c) e anche il suo avvertimento principale.

### 2.5 Discriminatori e valutazione di transizioni

- **Chen et al. (2022) DJtransGAN** (ICASSP 2022, già in biblioteca): discriminatore addestrato su mix reali per guidare un generatore di curve EQ/fader; valutazione con 136 ascoltatori validi (preferenza, smoothness, suoni in conflitto). Il discriminatore non è proposto come critico indipendente.
- **Rilevazione di splice nel parlato** (es. Moussa et al. 2023, SigPointer, arXiv:2307.05641): localizza giunte artificiali nel parlato. Dominio adiacente (forense), non musicale; nessuna fonte trovata su "edit detection" musicale con validazione percettiva.
- **DJ mix quality assessment**: nessuna fonte trovata con metrica validata. DJ-AI 2025 (già in biblioteca, evidenza debole) usa chroma/DTW e una descrizione errata di FAD.

---

## 3. Progetto del critico (cosa è implementato e perché)

| Critico | Misura | Fonte del metodo | Cosa può dire | Cosa non può dire |
|---|---|---|---|---|
| (a) estetico | Audiobox CE/CU/PQ e SongEval (5 assi) su finestre di 10 s: giunta (3 posizioni ±1 s) contro lati A-solo e B-solo della stessa clip, e contro le sorgenti originali alla stessa posizione ("restare su A", "B al suo cue"). Percentile del delta rispetto a giunte naturali. | Tjandra 2025 [S]; Yao 2025 [S]; riferimento interno [H] | se la finestra della giunta peggiora la "qualità di clip" rispetto alle sorgenti | se il cambio è musicalmente sensato |
| (b) sorpresa | NLL per frame (somma dei 4 codebook) di MusicGen-small non condizionato, teacher forcing; picco lisciato (0,25 s) tra 0,5 s prima dell'inizio e 2 s dopo la fine della transizione; z robusto rispetto ai tratti A-solo/B-solo; percentile rispetto a giunte naturali. | Copet 2023 [S]; Bjare 2025 [E]; LoopGen [S]; Noguchi 2026 [E] | se la giunta è più inattesa delle variazioni interne ai brani | se l'inatteso è piacevole (Huron: la sorpresa può essere voluta) |
| (c) giunta | Logistica su embedding MuQ (layer 6 e 12) prima/dopo/attraverso la giunta, più salto frame-a-frame e novità di Foote 2000; positivi = continuazioni reali attraverso confini di frase, salti puliti tra inizi di frase dello stesso brano, cut tra brani con tempo e tonalità compatibili; negativi = sfasamento 30–150 ms, salto ±4–8 dB, pitch shift incompatibile (±1, 2, 3, 4, 6 semitoni), taglio a metà parola (parole ASR), doppio basso, uscita fuori frase. Validazione per brano (fold di brani disgiunti). | Chen 2020 [E]; Foote 2000 [S]; negativi definiti dal mestiere DJ (REVISIONE_DJ) | se la giunta somiglia a un difetto noto | difetti non simulati; gusto |
| (d) tecnico | Controlli `dj_qa` (ritmo, doppio basso, loudness, true peak, clash vocale, parola tagliata) | regole [H]/[S] già documentate | integrità tecnica | qualità musicale |

Combinazione **dichiarata prima** del confronto con i giudizi: media a pesi uguali dei critici disponibili espressi come "bontà" in [0,1]; incertezza = propagazione delle sd per critico; disaccordo tra critici riportato a parte.

Scelte e cautele:
- Il critico (c) per gli item che usano i 7 brani V12 (tutti quelli con giudizi umani) è **riaddestrato senza quei brani**.
- Le clip storiche V8 e V10 (audio perso nel trasferimento) sono state **rigenerate bit per bit** con `musical_preview.render_handoff` (anchor_handoff_v1) e verificate sullo sha256 registrato nei manifest e nei giudizi: 16/16 identiche.
- I negativi con pitch shift passano dal phase vocoder di librosa; per non far imparare "artefatto = difetto", anche una parte dei positivi passa dal vocoder con 0 semitoni.
- I tagli tra brani sono senza time-stretch (solo coppie entro ±6% di tempo, solo cut): la deriva di tempo non è udibile dopo un cut netto.

---

## 4. Evidenza e ipotesi: cosa è dimostrato e cosa no

| Affermazione | Stato |
|---|---|
| Audiobox e SongEval predicono voti umani su clip e canzoni intere | [E] nei rispettivi domini (r 0,56–0,91) |
| … predicono la qualità di una transizione tra due brani | **non dimostrato**; qui testato su n minuscolo |
| La sorpresa di un modello audio autoregressivo traccia l'aspettativa umana | [E] parziale (EEG, Bjare 2025; Noguchi 2026), non su giunte |
| Picco di sorpresa alla giunta = transizione peggiore | [H]; la sorpresa può essere voluta (Huron [T]) |
| Un classificatore su difetti sintetici ordina le transizioni come un ascoltatore | [H]; Chen 2020 [E] mostra che metrica oggettiva e soggettiva possono divergere |
| Un punteggio combinato è una funzione obiettivo affidabile | **non dimostrato**; vedi §6 |

---

## 5. Limiti generali

- Un solo ascoltatore, n da 2 a 8 per insieme di validazione; le coppie non sono indipendenti.
- I giudizi V12 ("molto insufficienti") sono una dichiarazione globale riportata dall'orchestratore, non voti per clip.
- La codifica ordinale dei commenti verbali V10 è nostra [H].
- I critici preaddestrati non hanno visto rap in transizione; MusicGen è addestrato su musica licenziata (stock), MuQ su MSD, SongEval su canzoni in gran parte generate.
- CPU condivisa: i tempi di calcolo riportati sono pessimistici.

---

## 6. Risultati della run dj_critic_v1 (v1 → v3, 21/10 pomeriggio)

Numeri completi: `runs/dj_critic_v1/REPORT.md` §§5–12; modelli `models/v2_models.json`, `models/v3_models.json`
(v3a pre-specificata: `models/v3a_models_prespecified.json`); dose `dose.json`; validazione `validation_v3.json`
(v3a: `validation_v3a_prespecified.json`).

**Dati.** 986 giunte sintetiche pianificate; 919 eseguibili (i WAV canonici di `reminder` e `too_many_nights`
non sono più sulla macchina di analisi: 67 esempi e la base dose 6 esclusi). Embedding completi per MuQ, MERT-v1-95M, LAION-CLAP
(919/919), MusicGen sul sottoinsieme lungo (119/119), 136 varianti dose su 7 basi. I 50 `x_key` (cut tra
tonalità lontane) sono esclusi da addestramento e AUC (vedi §8, Brewster & Broughton cap. 15).

**(c) giunta su embedding.** CV per brano (5 fold), stessi 875 esempi: MERT full_pca32 0,68; MERT+CLAP
full_pca32 0,68 (scelto); MuQ full_pca32 0,64; MuQ+MERT+CLAP 0,65; CLAP 0,57. Con tutti i dati MuQ non aiuta
(la v1 su 123 esempi dava 0,63). AUC per difetto: tonalità 0,94, volume 0,85, doppio basso 0,66, fase 0,58,
fuori frase 0,54, parola tagliata 0,53; cross-track 0,37–0,65.

**(e) mestiere DSP, nuovo** (`craft_features`): |salto di loudness| (BS.1770, 3 s), |sfasamento| delle griglie
dei battiti A/B alla giunta (Beat This! small0, rette nei lati A-solo/B-solo; la prima versione solo DSP
con pettine sull'inviluppo d'attacco aveva errore mediano 54–109 ms sulle continuazioni naturali ed è stata
sostituita), distanza dei profili tonali Krumhansl-Kessler, voce Silero sul taglio, quota armonica HPSS ai due lati.

**Dose-risposta (criterio principale senza umani; ρ medio di Spearman bontà~intensità, atteso < 0):**

| critico | fase 0–200 ms | volume 0–9 dB | Camelot 0–6 | parola tagliata peggiore | esito |
|---|---|---|---|---|---|
| Beat This! fase (craft) | **−0,86** | −0,24 | −0,19 | 3/5 | tenuto |
| loudness (craft) | −0,12 | **−1,00** | −0,11 | 2/5 | tenuto |
| distanza tonale (craft) | +0,26 (piatta) | +0,65 (piatta) | **−0,44** | 2/5 | tenuto in v3b, escluso in v3a |
| giunta (c) MERT+CLAP | −0,33 | **−0,88** | +0,01 | 2/5 | tenuto |
| sorpresa MusicGen | −0,36 | −0,38 | +0,01 | 1/5 | tenuto, poi non aggiunto (peggiora la CV) |
| Audiobox Δ | +0,03 | +0,20 | **+0,34** | 4/6 | **escluso** (anti-correlato) |
| SongEval Δ | −0,29 | **+0,62** | −0,24 | 2/5 | **escluso** (premia il salto di volume) |
| voce Silero al taglio | +0,28 | 0,00 | +0,18 | 3/5 | escluso (non informativo) |
| ensemble v3 | −0,62 | −0,93 | −0,16 | 1/5 | — |

Nessun critico riconosce la parola tagliata (5–6 coppie, nessuno ≥ 75%). Nella famiglia tonalità tutte le
curve fanno un gradino 0 → 1 e poi restano piatte: il phase vocoder altera anche il livello (la loudness
scende da 0,59 a 0,16), quindi la dose tonale è in parte confusa con il volume [limite del banco].

**Ensemble v3 (solo dati sintetici).** Logistica con pesi ≥ 0 sulle bontà dei critici tenuti, CV per brano:
AUC OOF 0,69 (Brier 0,22) con loudness 0,24, fase 0,13, distanza tonale 0,11, giunta 0,54 (coefficienti
standardizzati); solo DSP 0,66; sola giunta 0,68. Calibrazione OOF ragionevole (0,18→0,03; 0,50→0,41; 0,82→0,81).

**Validazione sui giudizi umani** (unici giudizi disponibili: 6 voti galleria v1, 8 commenti V10 codificati,
2 V12 main; nessun giudizio nuovo nei database degli Artifact d'ascolto né in `giudizi_cloud/`). Accuratezza a
coppie aggregata (41 coppie, IC95 bootstrap sugli item): ensemble v3 **0,51** [0,22–0,85]; solo DSP 0,56
[0,26–0,87]; **numero di flag di veto 0,74** [0,55–0,91] (v3a, prima del flag tonale solo-blend: 0,82 [0,64–0,94]);
singole misure: fase Beat This! 0,82 [0,58–0,98], distanza tonale 0,78 [0,57–0,96], sorpresa 0,60, Audiobox 0,40;
v1 combinato 0,66. LOGO per coppia di brani (7 gruppi, scelta del critico sui gruppi restanti): S1 0,46, S2 0,56.
Con n così piccoli nessuna differenza è conclusiva; l'unico segnale coerente tra benchmark e giudizi è che
**i difetti di mestiere misurati direttamente (fase, tonalità, volume) e i flag** concordano con l'ascoltatore
più del punteggio appreso.

**Raccomandazione d'uso.**
- Affidabile come **veto tecnico**: flag `beat_phase_ms > 40`, `loud_jump_db > 3`, `chroma_dist > 1` (solo con
  sovrapposizione); più un flag = scartare o rifare la transizione. Monotonia dose-risposta forte (ρ −0,86/−1,00).
- Utile come **ordinamento grezzo** tra candidati tecnicamente puliti (score v3 ± sd), con bassa fiducia.
- **Non** affidabile per il gusto, l'opportunità del cambio, la parola tagliata, il fraseggio (fuori frase AUC 0,51)
  e il doppio basso; Audiobox/SongEval non vanno usati per scegliere transizioni (premiano il salto di volume).

## 6b. Libri (indicazione dell'utente 21/10): cosa entra nel giudice

Testo per pagina in `../libri/pages/` (privato); qui solo citazioni brevi. Pagine stampate; per
Brewster & Broughton (epub) il capitolo.

- **Brewster & Broughton, How to DJ Right (2003), cap. 15 "How to avoid key clashes"**: lo scontro di tonalità
  si sente quando i brani suonano insieme; rimedio: "Cut them, don't blend them", anche per cambiare tonalità
  di proposito. **Applicato**: i cut `x_key` non sono più negativi (esclusi da addestramento e AUC) e il flag
  tonale vale solo con sovrapposizione (`craft_flags(..., is_cut=True)`). Webber, *DJ Skills* p. 238 (due dischi
  in tonalità diverse "clash horribly" se sovrapposti) conferma.
- **B&B cap. 11 "How to cut"**: i cut funzionano da percussione a percussione o da melodia piena a percussione,
  raramente da melodia piena a melodia piena; il taglio va "in the tiny silence between the beats" a fine
  battuta. **Applicato come diagnostica**: `harm_ratio_pre/post` e `cut_melodic_risk` (HPSS); nessuna etichetta
  sintetica, quindi non entra nell'ensemble. **Proposto (v4)**: difetto sintetico "cut melodia→melodia" contro
  positivo "cut percussione→percussione" usando gli stem (pool CC).
- **B&B cap. 10 "How to match phrases"** e Butler, *Unlocking the Groove* p. 185 (DJ Shiva: allineare
  intro/outro perché "the breaks all fall in the same spots. So it locks"): conferma del difetto "fuori frase";
  B&B nota però che nei brani molto ripetitivi "no one will notice". Il giudice non lo riconosce (AUC 0,51):
  limite dichiarato.
- **Webber p. 244**: il pubblico reagisce ai "train wrecks" evidenti, mentre "slipped beat, or botched harmony"
  li notano soprattutto i DJ; **B&B, cap. "How to play out"**: "What sounds like a train wreck to you is often
  barely noticeable to the crowd". Gerarchia di gravità usata per la raccomandazione: fase grossolana e salti di
  volume = veto; errori piccoli = solo ordinamento. Non è stata usata per pesare l'ensemble (non verificabile).
- **Huron, Sweet Anticipation p. 116** (aspettativa del "cosa" e del "quando") e pp. 349–350: **applicato**
  come lettura separata in `scores_v3.json` (`huron_when_what`): il "quando" è la fase Beat This!, il "cosa" è la
  sorpresa MusicGen, letta come sorpresa di contenuto solo se la fase è entro 40 ms. pp. 24–25 (valenza
  contrastiva: la sorpresa può dare piacere) è il motivo per cui la sorpresa resta diagnostica e non veto.
- **Proposte dei lettori non applicate al giudice** (motivo): Margulis, ripetizione/"disco rotto" (serve un
  nuovo tipo di giunta sintetica: proposto per v4); Butler p. 244 "too busy" con `dj_form.overlap_profile`
  (richiede blend lunghi, assenti nel banco sintetico; più adatto a `dj_qa`); Huron p. 55 entropia predittiva
  separata dalla sorpresa (richiede di ricalcolare MusicGen); drop-out come evento (Huron p. 52; nessun
  esempio sintetico).

---

## 7. Righe proposte per INDEX.md e CHANGELOG.md (da integrare a cura dell'orchestratore)

Per `INDEX.md`, dopo "Implementazione del set DJ v1":

```
## Auto-feedback: critico automatico di transizioni (2026-10-21)

Quattro critici indipendenti con incertezza (Audiobox + SongEval relativi alle sorgenti, sorpresa di MusicGen alla giunta, classificatore auto-supervisionato su embedding MuQ con difetti sintetici, dj_qa) e validazione sui giudizi disponibili: [`_meta/AUTO_CRITICO_2026-10-21.md`](_meta/AUTO_CRITICO_2026-10-21.md) (`src/musicmix/dj_critic.py`, run `runs/dj_critic_v1/`).
```

Per `06_valutazione_autofeedback` (riga di fonti): Tjandra et al. 2025 (Audiobox Aesthetics, arXiv:2502.05139); Yao et al. 2025 (SongEval, arXiv:2505.10793); Bjare et al. 2025 ICASSP (arXiv:2501.07474) e ISMIR 2025 (arXiv:2508.05306); Copet et al. 2023 (MusicGen); LoopGen 2025 (arXiv:2504.04466); Chen, Smith, Yang 2020 (Neural Loop Combiner, arXiv:2008.02011); Gui et al. 2024 (FAD, arXiv:2311.01616); Zhu et al. 2025 (MuQ, arXiv:2501.01108); Pearce 2018 (DOI 10.1111/nyas.13654); Noguchi et al. 2026 (arXiv:2603.03190).

Per `CHANGELOG.md`:

```
- 2026-10-21: critico automatico dj_critic v1 (4 critici, combinazione dichiarata prima della validazione, ricostruzione bit-esatta di 16 clip storiche V8/V10); nota _meta/AUTO_CRITICO_2026-10-21.md; run runs/dj_critic_v1.
```

---

## 8. v4 (21/10/2026, sera): canali dei libri in un modulo separato

Il giudice v4 aggiunge i canali proposti dai libri (D1–D4, Q-1…Q-4, E-crit, E2) in `src/musicmix/dj_critic_books.py`,
collegati all'API con `dj_critic.score_transition(..., version="v4")` (campo `v4` additivo: con il default il v3 è
invariato), separa l'output in un blocco `technical` (veto/integrità) e un blocco `musical` (diagnostica, mai veto),
aggiunge il deck `a_raw` a `dj_render` e l'opzione `QAConfig.rhythm_on_raw_a` (spenta) e prepara la calibrazione
sui giudizi ciechi (`tools/calibrate_dj_critic.py`). Fonti, definizioni, risultati e limiti:
[`AUTO_CRITICO_V4_2026-10-21.md`](AUTO_CRITICO_V4_2026-10-21.md); run `runs/dj_critic_v4/`.
