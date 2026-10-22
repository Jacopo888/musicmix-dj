# 06 — Research-to-implementation: cosa prendere dai paper

Questo documento collega risultati letti nella revisione precedente a **moduli concreti** e **verifiche riproducibili**. Per i metodi descritti, le prestazioni pubblicate nei paper NON vanno trasferite automaticamente al task del nostro progetto.

| Paper | Ipotesi / tecnica trasferibile | Modulo | Esperimento che ne verifica l'utilità |
|---|---|---|---|
| [01 AutoMashUpper (2014)](https://repositorio.inesctec.pt/bitstreams/97188d38-9bf1-46f9-99a3-08d3b656ca4f/download) | Chroma+ritmo+spettro per mashability locale | M05–M08 | Baseline B2 su cue liberi |
| [02 Lee et al. (2015)](https://www.ismir2015.uma.es/articles/302_Paper.pdf) | Compatibilità verticale ≠ orizzontale e importanza continuità vocale | M07–M09/M12 | Classificare transition type, ablation separate |
| [03 Modeling Stem Compatibility (2021)](https://arxiv.org/abs/2103.14208) | Contrastive/classificazione su stems e non etichettati | M08 | Reranker ML vs score interpretabile, hard negatives |
| [04 COCOLA (2025)](https://arxiv.org/abs/2404.16969) | Feature di coerenza audio complementare | M08 | Aggiungere COCOLA solo tra candidate top-K e misurare preferenze |
| [05 AutoMashup (2025)](https://arxiv.org/abs/2508.06516) | Direzione A-vocals/B-backing, limiti general audio embeddings | M01/M08 | Swap direzionale e confronto CLAP vs task feature |
| [06 Neural Loop Combiner (2020)](https://jbls.fun/documents/chen2020-ismir-neural_loop_combiner.pdf) | Negativi realistici (timing errato e wrong pairs) | M07–M08 | A/B score su coppie corrette e perturbate |
| [07 Diversity in Mashups (2023)](https://link.springer.com/article/10.1007/s00779-022-01702-z) | Diversificazione del top-K, multi-obiettivo | M08/M11 | Qualità e diversity@K contro top-K non diversificato |
| [08 Cue Points (2020/23)](https://arxiv.org/abs/2007.08411) | Cue significativi e dataset etichettato | M02/M06 | Precision/Recall cue e retrieval oracle |
| [09 Raw Audio to Seamless Mix (2018)](https://link.springer.com/article/10.1186/s13636-018-0134-8) | Vincoli pratici DJ e pipeline end-to-end | M02/M09/M12 | Checklist rendering e short previews |
| [10 DJ Mix Alignment (2020)](https://arxiv.org/abs/2008.10267) | Ricostruire transizioni umane da DJ mixes | M10 | Costruire corpus di cue realistici, rispettando licenze |
| [11 DJGAN (2022)](https://arxiv.org/abs/2110.06525) | Policy di effetti e protocolli di ascolto | M10/M12 | Renderer neutro vs avanzato con cue fissi |
| [12 Spectral Clustering (2014)](https://brianmcfee.net/papers/ismir2014_spectral.pdf) | Matrice di ricorrenza per segmenti e ritorni | M02 | Versi/cori con boundary gold |
| [13 All-In-One (2023)](https://arxiv.org/abs/2307.16425) | Beat/downbeat/functional structure da stems | M02 | Ablation beats vs joint multi-task |
| [14 Lyrics Real-Time (2024)](https://arxiv.org/abs/2401.09200) | Combinare caratteristiche fonetiche e chroma per allineamento | M03 | Word alignment accuracy su canto |
| [15 Phoneme Lyrics (2021)](https://schufo.github.io/files/2021_TASLP3091817.pdf) | Attenzione/DTW monotona e allineamento fonema | M03/M07 | Inizio/fine parola su gold; controllo melisma |
| [16 CREPE (2018)](https://arxiv.org/abs/1802.06182) | Curva F0 robusta da stem voce | M03/M07 | Match parola con stessa vs diversa intonazione |
| [17 Basic Pitch (2022)](https://arxiv.org/abs/2203.09893) | Note e multipitch estimation | M04 | Retrieval di riff e confronto con chroma semplice |
| [18 Hybrid Transformers (2022)](https://arxiv.org/abs/2211.08553) | Source separation stems vocal/drum/bass/other | M01/M12 | Leak audio vs qualità di matching |
| [19 BS-RoFormer (2023)](https://arxiv.org/abs/2309.02612) | Alternativa separazione in bande | M01/M12 | Costo e migliore accuratezza su subset difficile |
| [20 CrossSong Puzzle (2015)](https://staff.aist.go.jp/m.goto/PAPER/SMC2015smith.pdf) | Evitare esplosione combinatoria | M06/M11 | Beam search vs greedy e vincoli |
| [21 Spotify (2017)](https://archives.ismir.net/ismir2017/paper/000086.pdf) | Pairwise cost matrix su punti beat e ordinamento brani | M06–M11 | Baseline B1 senza limite a intro/outro |

## Le tre lacune che dobbiamo colmare con nostri esperimenti

1. **Word-to-word cross-song transition:** lyrics alignment e pitch extraction esistono, ma il risultato percettivo di switch su una stessa parola fra due registrazioni diverse non è garantito da quei paper. È il test più caratteristico del progetto.
2. **Coerenza ≠ creatività:** stem coherence models e cosine similarity non dicono necessariamente se un taglio sia sorprendente e musicalmente significativo.
3. **Confronto fair della ricerca:** molti lavori misurano mix completo o stem overlap; noi dobbiamo isolare cue choice da rendering e usare un esperimento fattoriale/sequenziale.

## Riferimenti complementari (non trattati come letti integralmente)

- [Psychoacoustic Approaches for Harmonic Music Mixing](https://www.mdpi.com/2076-3417/6/5/123): possibile metrica di consonanza fine per M07.
- [WhisperX](https://arxiv.org/abs/2303.00747): ASR/forced alignment parlato, possibile assistenza ma va testato sul cantato.
- [MIREX Lyrics-to-Audio Alignment](https://music-ir.org/mirex/wiki/2024%3ALyrics-to-Audio_Alignment): metriche di allineamento.
- [MUSDB18](https://sigsep.github.io/datasets/musdb.html), [MedleyDB](https://medleydb.weebly.com/): dati multitraccia, soggetti a licenza e adattabilità del task.
