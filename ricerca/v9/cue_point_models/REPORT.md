> Copia integrata. Leggere prima [le correzioni dell’orchestratore](../REVIEW_CORRECTIONS.md); gli originali restano nella cartella dello swarm.

# Cue point, transizioni automatiche e timing mix-in/mix-out

**Agente:** `cue_point_models` (swarm_v9) · **Data:** 2026-10-19 · **Modello:** OpenCode / DeepSeek (nessuna delega)
**Ambito:** stima dei cue point, transizioni DJ automatiche, timing appreso mix-in/mix-out, reverse engineering/allineamento di DJ mix, dataset annotati di cue.
**Metodo:** ricerca profonda su famiglie di query multiple + citation chasing avanti/indietro; sono stati salvati e letti i full text dei lavori più rilevanti (copie in `sources/` con URL e SHA256). Nessun modello scaricato/eseguito, nessun audio ascoltato: nessun giudizio audio inventato.

> Nota di onestà: tutti i numeri di benchmark appartengono al dataset/compito dei rispettivi lavori e **non** sono prova sul pool di questo progetto (Reminder, Too Many Nights, 5% TINT, Creature). Le citazioni brevi sono parafrasate; i righi di ancoraggio sono in `claims.json`.

---

## 1. Sintesi esecutiva

1. La stima dei cue point è oggi una **task per singolo brano** (object detection / regole / alignment). Nessun lavoro rivisto risolve la decisione "restare in A o cambiare, e dove", per una **coppia** di brani, come obiettivo appreso.
2. I cue umani si concentrano sui **multipli di frase** (8/16 bar): 23,6% deviazione zero, 40,4% entro 1 misura, 73,6% entro 8 misure, 86,2% entro 16 misure (Kim 2020). Questo è il pattern da usare come **ipotesi**, non come legge, per il rap.
3. I dataset forti sono **EDM / 4-4**: EDM-CUE (4.710 tracce), M-DJCUE (134), Raveform (1.423 annotate). **Non esiste un dataset di cue/frasi per rap/hip-hop.** Questa è l'incertezza di trasferimento centrale.
4. La durata della transizione **non è una costante**: DnB 16-32 misure (VandeVeire 2018), DJtransGAN fissa 8 bar, Spotify un numero di beat. La qualità dipende dalla **coppia** (extra-mixability, M-DJCUE 2024).
5. Se si renderizza un join e si hanno i due deck + il master, una parte della qualità diventa **misurabile per componenti separate** (loudness, headroom, collisione/continuità spettrale, smoothness del gain, stabilità stereo) - **mai** un singolo scalare (Daito Manabe 2026).
6. Warning diretto contro i ranking di continuità: i lavori dichiarano la qualità soggettiva e **non** riducibile a un unico punteggio; embeddings generici **non** sono un oracolo di compatibilità (ripreso anche in V8).

---

## 2. Fonti verificate (15 voci; 13 primarie)

| id | tipo | profondità | nota chiave |
|---|---|---|---|
| cue_detr_2024 | preprint + code/weights | full_text | cue = object detection; EDM-CUE 4.710 tracce; F1 0.46; **singolo brano** |
| vandeveire_2018 | journal + code | full_text | DnB 160-190 BPM; cue 16 misure prima del drop; fade 16/32 misure; 91% annotazioni corrette; **17 s CPU per transizione** |
| zehren_2020_cue | journal + code | full_text | switch point rule-based, ~86% precision su 134 tracce EDM |
| mdjcue_2024 | LBD + dataset | full_text | linee guida annotazione; **intra- vs extra-mixability** |
| zehren2024_interpretability | journal (MDPI, CC BY) | full_text | EDM-150 (150 tracce, 3 esperti, 625 posizioni); precision 86-93%, recall 29-54%; feature: novelty energia, timbro, drum onset, armonia |
| kim2020_djmix_alignment | conference | full_text | 1.557 mix / 13.728 tracce / 20.765 transizioni; hit rate 0,58/0,80/0,93 a 15/30/60 s |
| raveform_tismir2026 | journal + dataset | full_text | 4.902 mix, 1.423 annotate; metriche beat/downbeat ±70 ms, CMLt ±17,5%, HR.5F ±0,5 s; **degrado cross-dataset** |
| bittner_2017_spotify | conference (practitioner) | full_text | 64,13% Good / 28,26% OK / 7,61% Bad; failure: beat disallineati 15,22%, mid-vocals 6,52% |
| djtransgan_2021 | conference + code | full_text | cue-in/out definiti; regione **8 bar fissi**; DDSP fader/EQ + GAN |
| cmmr2019_djmix | conference + dataset | full_text | UnmixDB; 5 passi (DTW, allineamento campione, track removal, curva di volume, cue); errore ~25 ms |
| schwarz_2018_cue | conference (practitioner) | full_text | euristiche strutturali; >50% entro 10 s, 2/3 entro 20 s; guardia loudness -9 dB |
| daito_manabe_2026 | preprint/practitioner | full_text | diagnostica per componenti su deck A/B/master; **niente scalare unico** |
| sowula_knees_2024_mosaikbox | conference | full_text | stem modification + beat-grid stretto; qualità soggettiva; r=0,83 |
| cuedetr_repo | official code | selected_sections | I/O concreto (`cue_points.py`, mp3 -> `_cue_points.txt`); pin torch 2.3.1/transformers 4.42.3 |
| (cmmr dataset) UnmixDB | dataset | metadata | 37 playlist x 12 varianti = 444 mix; Zenodo 1422385 |

---

## 3. Cosa è NUOVO rispetto al contesto V8

Già in contesto (V8): All-In-One, SongFormer, MuQ, MERT, Beat This!, DJtransGAN, Zehren 2020, AutoMashup, Bittner 2017, MSAF.

**Nuovo in questa ricerca:**
- **CUE-DETR (2024)** come modello cue-point con code+checkpoint e I/O concreto.
- **Raveform (TISMIR 2025-2026)**: il dataset EDM-in-DJ-mix più grande e il degrado cross-dataset misurato.
- **Zehren 2024, interpretability (Signals)**: analisi delle feature e della loro interazione su EDM-150.
- **Daito Manabe 2026**: framework interpretabile di valutazione per componenti (non scalare).
- **M-DJCUE 2024**: tassonomia label + **intra/extra-mixability**.
- **CMMR 2019 / UnmixDB**: metrica di allineamento **senza ground truth** (track removal, ~10 dB di calo RMS) e dataset open.
- **Schwarz 2018**: guardia loudness -9 dB e regione cue di 4 misure ancorata al beat.
- **VandeVeire 2018** letto per intero (in V8 era solo sintetizzato).

---

## 4. Cue point: cosa sono esattamente (input, label, tolleranze)

- **Input:** audio di un singolo brano (mp3 nel tool CUE-DETR); oppure mix + playlist di tracce note (reverse engineering).
- **Label:** posizione/i di cue in secondi. M-DJCUE impone **inizio di battuta**, precisione **1/10 s**; cue-in dentro l'intro fino al primo beat della sezione "core"; cue-out ovunque ma non prima del core; cue-out può avere durata (fade-out).
- **Tolleranze (per gli acceptance test):** Kim 2020 misura a 15/30/60 s; Raveform a ±70 ms (beat/downbeat), ±17,5% (CMLt), ±0,5 s (confini). Queste sono le unità corrette per giudicare "abbastanza vicino", non un metro astratto.
- **Struttura:** picchi a 32/64/128 beat (multipli di frase); 8 e 16 bar le frasi più rappresentate (EDM).

**Implicazione per il progetto:** una tolleranza realistica per un cue "buono" è **dell'ordine della misura musicale** (0,4-2 s a 128 BPM), non del millisecondo, salvo che il problema sia il taglio di una parola (lì conta la parola, non la battuta).

---

## 5. Cross-song vs single-track vs frase

La tabella sotto risponde direttamente alla richiesta di valutare **su quale compito** ogni modello è stato testato.

| Lavoro | Singolo brano | Coppia/cross-song | Mix sequenziale reale | Stem simultanei | Genere |
|---|---|---|---|---|---|
| CUE-DETR 2024 | **sì** | no | no | no | EDM 4/4 |
| VandeVeire 2018 | sì | sì (regole) | mix generati | no | DnB |
| Zehren 2020 / 2024 | sì | no | no | no | EDM |
| M-DJCUE 2024 | sì (label) | extra-mixability (concetto) | no | no | EDM |
| Kim 2020 | no | sì (mix noto -> tracce) | **sì** | no | dance |
| UnmixDB / CMMR 2019 | no | sì | **sì** (generati) | no | dance |
| Bittner 2017 | no | sì | prodotto | no | pop/quadruple |
| DJtransGAN 2021 | no | sì | sì (8 bar) | no | elettronica |
| Raveform 2026 | sì | no | metadati di mix | no | EDM |

**Conclusione:** il progetto deve costruire da sé il ponte **selezione + render cross-song**; può però riusare (a) i cue come **candidati** e (b) le metriche di allineamento/rimozione e i diagnostici per componenti come **verifica**.

---

## 6. Contraddizioni, prove negative e avvertenze

- **Recall bassa vs precision alta:** tutti i detector di switch/cue hanno precision alta ma recall 29-54% (Zehren 2024). Non aspettarsi copertura.
- **Embeddings non sono l'oracolo:** più lavori (e AutoMashup in V8) negano che similarità/chroma/embedding = compatibilità o "risoluzione". Confermato anche dalla natura soggettiva dichiarata in Sowula & Knees 2024.
- **Fallimenti umani ricorrenti:** "beats do not align" 15,22% e "mid-vocals" 6,52% (Bittner 2017). Il taglio a metà parola è un fallimento noto e concreto, non un'ipotesi.
- **Degrado cross-dominio:** il downbeat F1 di un modello EDM forte scende da ~0,915 (contesto pop/Harmonix) a ~0,753 (condizione cross-dataset EDM). Questo è il migliore indizio quantitativo di quanto ci si deve aspettare meno su materiale fuori dominio.
- **Sensibilità dell'allineamento:** Schwarz 2019 nota che un errore di velocità dello 0,1% desincronizza dopo pochi secondi; il progetto **non** fa correzione di tempo/tono, quindi i join con rate diverso vanno dichiarati a rischio.

---

## 7. Durata della transizione e dipendenza dai due brani

- **Durata:** DnB 16-32 misure; DJtransGAN fissa 8 bar; Spotify usa un numero di beat pesato verso i confini di sezione; Schwarz propone cue region di 4 misure. Non esiste una durata "giusta" universale: va **decisa per coppia** e allineata alla frase.
- **Coppia:** M-DJCUE separa *intra-mixability* (un cue funziona dentro il fraseggio del brano) da *extra-mixability* (quanto bene due brani stanno insieme). L'esito dipende quindi da **entrambi** i brani e dalla durata scelta, non dal solo punto di uscita.
- **Conseguenza pratica:** il "quando cambiare" va trattato come una superficie (cue_in x cue_out x fadeA x fadeB), non come una soglia. Vedi R4.

---

## 8. Matrice di fattibilità (Windows, CPU, nessun download qui)

| Approccio | Costo | Rischio Windows/CPU | Payoff | Priorità |
|---|---|---|---|---|
| Euristiche cue (novelty energia + drum onset + armonia + loudness) | basso | basso | candidati interpretabili | **1** |
| Harness UnmixDB-style (allineamento + track removal) | medio | basso (DSP/NumPy) | metrica onesta senza ground truth | **1** |
| Diagnostici per componenti (Daito-Manabe) | medio | basso (STFT) | valutazione multi-componente | **2** |
| CUE-DETR (checkpoint) | alto | alto (torch 2.3.1, mp3, GPU consigliata) | locator EDM | 3 |
| All-In-One / Raveform baseline | alto | alto (NATTEN/Demucs/madmom) | confini/structure | 3 |
| DJtransGAN training | molto alto | GPU | render appreso | 3 (ricerca) |

---

## 9. Miglior esperimento eseguibile adesso

**R1 + R2 combinati**, in due fasi brevi:

1. **Candidati (R1):** sui quattro brani, con Beat This! già integrato, generare cue candidati allineati all'inizio di battuta, con feature separate (novelty energia, drum onset, MFCC/PCP, flag loudness -9 dB) e **abstention** esplicita se il metro è incerto -> `cue_candidates.json`.
2. **Verifica (R2):** riprodurre la metrica di UnmixDB (allineamento + suppression ratio) su mix sintetici noti e poi sui render 10sA+10sB; riportare hit-rate a 15/30/60 s e il calo RMS.

**Controlli di fallimento:** niente scalare fuso; assumere mixing livello 1-2 (tracce quasi invariate); flaggare i casi distortion/bass dove la metrica peggiora (Schwarz 2019); nessuna affermazione di qualità dal solo RMS.

**Test di accettazione:** generazione deterministica e riproducibile di candidati con provenienza per-feature; harness che stampa errore di allineamento e suppression ratio per ogni join e dichiara "non applicabile" quando le assunzioni non valgono.

---

## 10. Requisito "aspettativa crescente in A -> continuazione o arrivo in B"

Non esiste in letteratura un modello validato che predica la "risoluzione dell'aspettativa" per una coppia arbitraria. Coerentemente con il divieto di scalari arbitrari (R6), il requisito va verificato come **vettore di condizioni separate**, ciascuna con incertezza:

1. la feature crescente (es. pendenza di energia/centroide spettrale) **non si inverte** attraverso il join;
2. se in A si esaurisce, in B c'è un **arrivo cadenzale/downbeat**;
3. nessun **taglio di parola** (riusare `word_guard`);
4. ancora **metrica** (downbeat) presente;
5. ogni condizione non calcolabile = **UNKNOWN** -> astensione, mai imputazione.

Nessuna somma, nessun punteggio unico, nessuna pretesa di gusto.

---

## 11. Lead falliti e lacune aperte

- **Non recuperati:** PDF UnmixDB (403, coperto da CMMR 2019); PDF diretto MDPI (403, coperto via mirror); "blind DJ mix transcription" di André et al. (visto solo come citazione secondaria in Daito Manabe 2026); dettagli completi AutoMashup (V8).
- **Lacuna centrale:** **nessun dataset cue/frasi per rap/hip-hop**; nessun modello decisionale cross-song appreso; nessun benchmark percettivo pubblico per la risoluzione A->B.
- **Rischio di over-claim:** tutte le durate/regole di frase sono EDM/DnB e vanno **ri-stimate** sul pool rap, non importate.

---

## 12. Riferimenti principali (URL)

- CUE-DETR https://arxiv.org/abs/2407.06823 · code https://github.com/ETH-DISCO/cue-detr · data/weights https://huggingface.co/disco-eth/cue-detr
- VandeVeire & De Bie 2018 https://doi.org/10.1186/s13636-018-0134-8
- Zehren et al. 2020 https://arxiv.org/abs/2007.08411 · Zehren et al. 2024 https://doi.org/10.3390/signals5040036
- Kim et al. 2020 https://arxiv.org/abs/2008.10267 · Raveform https://doi.org/10.5334/tismir.288
- Bittner et al. 2017 https://archives.ismir.net/ismir2017/paper/000086.pdf
- DJtransGAN https://arxiv.org/abs/2110.06525
- Schwarz & Fourer 2019 (UnmixDB) https://hal.science/hal-02172427 · UnmixDB https://zenodo.org/record/1422385
- Schwarz et al. 2018 https://abcdj.eu/download/a-heuristic-algorithm-for-dj-cue-point-estimation/

Dettagli, letture effettive, licenze, risorse e SHA256 in `sources.json`; affermazioni con ancore in `claims.json`; raccomandazioni con observation->evidence->mechanism->experiment->controls->acceptance in `RECOMMENDATIONS.json`.
