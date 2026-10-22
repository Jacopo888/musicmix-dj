# 01 — Roadmap: dalla scoperta alla transizione

**Data:** 18 ottobre 2026 • **Dipendenze:** sequenziali sui gate, modulari sui componenti • **Priorità:** dimostrazione scientifica prima degli effetti audio.

## I due grandi step

**STEP I — SCELTA DEI PUNTI:** dimostrare che, dati due brani, il programma individua autonomamente collegamenti A(tₐ)→B(tᵦ), senza vincoli fissi di inizio/fine, che persone reali preferiscono a baseline credibili. Include un *renderer neutro minimale*, indispensabile per far ascoltare le transizioni, ma **non** ne ottimizza ancora gli effetti.

**STEP II — REALIZZAZIONE DEL MIX:** dati punti di ingresso/uscita *identici*, migliorare la resa con stem, adattamento tempo/pitch, EQ, fade, automazioni, limiter e controlli di artefatti.

## STEP I — Milestone e condizioni di accettazione

| Milestone | Deliverable | Esperimento / criterio di uscita | Paper guida |
|---|---|---|---|
| **M0 Fondazioni** | Repository, licenze, 20–40 brani legittimi e diversificati, metadata e tracciamento; conversione WAV coerente | Test di decodifica, sincronizzazione e integrità dei file | [All-in-One 2023](https://arxiv.org/abs/2307.16425), [DJ mixes 2020](https://arxiv.org/abs/2008.10267) |
| **M1 Baseline strutturale** | Beat/downbeat, chroma, RMS, segmentazione, cue, candidati downbeat→downbeat da posizioni arbitrarie | Su ogni coppia recupera top-K spiegabili; tutti i candidati riproducibili | [Spotify 2017](https://archives.ismir.net/ismir2017/paper/000086.pdf), [AutoMashUpper 2014](https://repositorio.inesctec.pt/bitstreams/97188d38-9bf1-46f9-99a3-08d3b656ca4f/download) |
| **M2 Primer qualitativo** | Preview standard e primo studio cieco di selezione dei punti | Baseline musicale vs random vincolato; verificare se test/protocollo discriminano qualità | [DJGAN 2022](https://arxiv.org/abs/2110.06525), [Spotify 2017](https://archives.ismir.net/ismir2017/paper/000086.pdf) |
| **M3 Matching vocale** | Stem vocali, lyrics allineate, parole/fonemi, pitch F0 e DTW; candidato parola→parola | Verifica manuale su subset delle parole; ablation `no_lyrics`, `no_F0` | [Phoneme Lyrics 2021](https://schufo.github.io/files/2021_TASLP3091817.pdf), [CREPE 2018](https://arxiv.org/pdf/1802.06182) |
| **M4 Matching strumentale** | Riff, note/intervalli, onset e motivi: candidato motivo→motivo | Esperimenti mirati riff simili vs negativi difficili; ablation `no_melody` | [Basic Pitch 2022](https://arxiv.org/pdf/2203.09893), [Neural Loop Combiner 2020](https://jbls.fun/documents/chen2020-ismir-neural_loop_combiner.pdf) |
| **M5 Ranker interpretabile** | Score separati per continuity, horizontal, vertical, novelty, costo trasformazioni, con breakdown | Confronto cieco random vs baseline Spotify-like vs ranker completo; error analysis | [Lee 2015](https://www.ismir2015.uma.es/articles/302_Paper.pdf), [AutoMashup 2025](https://arxiv.org/abs/2508.06516) |
| **M6 Gate scientifico A** | Dataset di giudizi, report statistico, top-K diversificato, ablation e freeze baseline | Evidenza replicabile di preferenza rispetto alla migliore baseline: intervallo di confidenza compatibile con un reale miglioramento; analisi per tipo di collegamento | [COCOLA 2025](https://arxiv.org/abs/2404.16969), [Diversity 2023](https://link.springer.com/article/10.1007/s00779-022-01702-z) |
| **M7 N brani** | Grafo di clip e transizioni, ricerca globale con limiti di riuso, vincoli narrativi, durata e copertura | Mix N-brani riproducibile; nessun brano saltato salvo policy dichiarata; niente tagli ciclici illegali | [CrossSong 2015](https://staff.aist.go.jp/m.goto/PAPER/SMC2015smith.pdf), [Spotify 2017](https://archives.ismir.net/ismir2017/paper/000086.pdf) |

**Gate A (non negoziabile):** la metrica non è «ha trovato la stessa parola» ma «gli ascoltatori preferiscono questa transizione a una baseline musicale». La stessa parola può essere un falso positivo armonico e ritmico. Non passare al perfezionamento del rendering per nascondere un problema di selezione.

## STEP II — Milestone e condizioni di accettazione

| Milestone | Deliverable | Esperimento / criterio di uscita | Paper guida |
|---|---|---|---|
| **M8 Renderer multistem** | Uscita/entrata selettiva per voce, basso, batteria e altro; crossfade a durate diverse | A cue invariati migliora pulizia percepita senza degradare la musicalità | [Hybrid Transformers 2022](https://arxiv.org/abs/2211.08553), [DJGAN 2022](https://arxiv.org/abs/2110.06525) |
| **M9 Correzioni DSP** | Time-stretch, pitch shift limitato, beat phase alignment, loudness, EQ, limiter | Artefatti misurati/ascoltati per modulo; superamento del renderer neutro a cue fissi | [DJGAN 2022](https://arxiv.org/abs/2110.06525), [BS-RoFormer 2023](https://arxiv.org/abs/2309.02612) |
| **M10 Render policy e QA** | Selezione tra cut, word handoff, fade, stem takeover in base al contesto; esportazione stereo | Blind paired test renderer avanzato vs neutro, **stessi** punti A/B | [Lee 2015](https://www.ismir2015.uma.es/articles/302_Paper.pdf), [DJGAN 2022](https://arxiv.org/abs/2110.06525) |
| **M11 Packaging** | CLI, caching incrementale, report ispezionabile, preset, gestione N brani | Rerun su un'altra macchina, output deterministico o seed registrato, test regressione | [CrossSong 2015](https://staff.aist.go.jp/m.goto/PAPER/SMC2015smith.pdf) |

## Metriche consigliate per i gate

**Retrieval intrinseco (diagnostico, NON successo finale):** precision@K di punti validi, copertura per modalità, recall di esempi umani se disponibili, % cue errati, errori di parola/fonema, deviazione downbeat, candidati scartati.  
**Qualità percepita (primaria):** win rate preferenza cieca rispetto a baseline musicale, intervallo di confidenza per coppia di brani, rate di transizioni non musicali, interesse creativo.  
**Rendering (solo Step II):** preferenza a cue fissi, percezione di artefatti, clic, overlap vocale indesiderato, clipping e loudness.  
**Efficienza:** tempo preprocessing per minuto, candidati/s, peak RAM/VRAM, dimensione cache.

## Piano sperimentale minimo

1. Pre-registrare soglie/metriche e congelare lo split prima di confrontare modelli.
2. Dataset pilota eterogeneo con diritti adeguati; annotazioni precise per un sottoinsieme di parole e riff.
3. Random **su cue validi** (non su millisecondi casuali), baseline strutturale musicale, ranker proposto.
4. Generare preview con **identica** politica di render per tutti i sistemi.
5. Bilanciare l'ordine degli ascolti, nascondere nomi sistemi, valutare per brano/coppia, non per render replicati.
6. Eseguire ablation una variabile alla volta e controllare distribuzione dei casi difficili.
7. Congelare la baseline prima di iniziare Step II. I renderer possono cambiare, i cue del confronto no.

## Rischi bloccanti

- I brani vocali con ASR inaffidabile causano *falsi match*: lyrics opzionali + confidence, mai requisito universale.
- Separazione voce introduce artefatti: valutare una modalità direttamente dal mix originale.
- Il ranker potrebbe imparare solo armonia e preferire transizioni banali: distinguere **compatibilità** da **interesse creativo**.
- Bias dei generi e degli ascoltatori: stratificare e documentare.
- Diritti su audio, lyrics e dataset: verificare prima di salvare/pubblicare asset.

## Cosa non serve ora

- Addestrare un LLM o modelli end-to-end.
- Effetti neural audio sofisticati.
- Interfaccia grafica o sistema live.
- Supporto completo N brani prima di validare le coppie A→B.
