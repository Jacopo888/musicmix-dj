# Sintesi italiana — critic/adoption v12 (20 ottobre 2026)

Questa sintesi riguarda solo le fonti primarie scaricate e lette in questo turno. Le etichette **misurato**, **teoria** e **ipotesi ingegneristica** sono usate nel senso della biblioteca `conoscenza_mix/`: un numero pubblicato vale nel dataset e nel compito di quel lavoro, non nel pool MusicMix.

## 1. MusicFlamingo: audio-LLM musicale, non un critico di transizioni

**Misurato (dagli autori).** Music Flamingo è un modello audio-linguaggio da circa 8B parametri, costruito su Audio Flamingo 3 e Qwen2.5-7B, addestrato su MF-Skills (circa 2M canzoni complete, 2,1M didascalie, circa 0,9M coppie QA). Il model card dichiara fino a 20 minuti di audio e finestre di 30 s; i risultati dichiarati riguardano 10+ benchmark di comprensione e ragionamento musicale.

**Limite per il nostro compito.** Non è un valutatore di transizioni A→B: non produce un punteggio locale calibrato, non è addestrato su cue o su esiti di mix, e le risposte sono testo da interpretare. I pesi sono 8B in BF16, non stanno nel budget di questo turno e non sono stati scaricati. La licenza del modello è NVIDIA OneWay Noncommercial; il codice del repository è MIT. **Adozione: respinta come gate**; eventualmente utile solo come descrittore offline, non come reward.

## 2. Beat This!: timestamp affidabili, ma nessuna nozione di closure

**Misurato (dagli autori).** Beat This! è una rete di circa 20M parametri che lavora su 30 s di audio mono a 22,05 kHz e produce beat/downbeat senza DBN. Il paper riporta F1 superiore allo stato dell'arte su un insieme eterogeneo, ma anche metriche di continuità (CMLt/AMLt) peggiori e fallimenti su generi sottorappresentati. Il modello principale è circa 78 MB; il modello `small` circa 8,1 MB. Codice e pesi sono MIT.

**Limite per il nostro compito.** Un downbeat è un candidato temporale, non una risoluzione musicale. Il paper non contiene annotazioni di transizioni, né giudizi di frase o di arrivo. Il modello `small` supera comunque il limite di 5 MB di questo turno, quindi non è stato scaricato. **Adozione: fornitore di timestamp** (se il budget pesi verrà riaperto), non gate.

## 3. SSM-Net: il candidato più leggero per un critic di confine strutturale

**Misurato (dagli autori).** SSM-Net impara un encoder e dei kernel di novelty con due loss: una sull'auto-similarità (SSM) e una sulla novelty. L'input pubblicato è log-mel a 80 bande, finestra 92 ms, hop 23 ms, aggregazione a 0,1 s, patch di 40 frame (4 s) con hop 0,5 s. Il modello ha circa 500K parametri. Su RWC-Pop-AIST riporta HR3F = 0,713 e HR0.5F = 0,399; su SALAMI Pop (upper) HR3F = 0,631/0,624; su SALAMI-IA 0,520/0,511; su SALAMI-Two 0,521/0,530. Gli autori notano sovra-segmentazione con annotazioni upper e sotto-segmentazione con lower; l'hop di 0,5 s limita la precisione fine. Il paper è CC BY 4.0; il codice è LGPL-3.0; i due `state_dict` ufficiali nel repository sono 1.483.802 e 2.289.503 byte.

**Limite per il nostro compito.** Un confine strutturale non è una risoluzione: il modello non sa se B realizza l'aspettativa di A, né se una parola è tagliata. La licenza LGPL-3.0 e l'assenza di una licenza separata per i pesi vanno registrate prima dell'integrazione. **Adozione: feature di evidenza locale**, non reward scalare.

## 4. All-In-One: struttura funzionale e metrica in un solo modello piccolo, ma costoso da eseguire

**Misurato (dagli autori).** All-In-One predice insieme beat, downbeat, confini di segmento e etichette funzionali (intro/verse/chorus/...) da spettrogrammi di stem separati. Ha circa 300K parametri e risultati misurati sul Harmonix Set. Il repository è MIT; il checkpoint `harmonix-fold0` osservato con una HEAD sull'HF dell'autore è 1.400.571 byte (non scaricato). Il README ufficiale richiede NATTEN su Windows e una pipeline di source separation (demucs).

**Limite per il nostro compito.** Le etichette funzionali non sono una risoluzione e possono essere sbagliate su rap/trap; il costo di installazione (NATTEN, demucs, pesi di separazione) è maggiore del vantaggio. **Adozione: seconda scelta**, solo se il root accetta il costo di demix e compilazione.

## 5. Medley2K: annotazioni esterne di transizione, ma simboliche

**Misurato (dagli autori).** Medley2K contiene 2.000 medley e 7.712 transizioni etichettate, estratte da punteggi MIDI; la validazione dell'estrazione su 30 medley riporta precisione 90,70% e recall 57,07%. Gli esperimenti mostrano che addestrare sulla transizione, e non su musica arbitraria, migliora il modello generativo.

**Limite per il nostro compito.** È MIDI, non audio prodotto; l'etichetta è un punto di transizione, non un giudizio di qualità; il recall è basso. Non è stato scaricato. **Adozione: protocollo di valutazione esterna**, non modello.

## 6. Prova eseguita in questo turno (sintetica, onesta)

Poiché il turno vieta altri download, pesi e Torch/TF, non è stato eseguito SSM-Net con i suoi pesi. È stata invece eseguita una prova CPU di *feature plumbing* con `numpy`, `scipy`, `librosa` e `soundfile`: 80 coppie sintetiche (48 train, 32 test held-out per seed del generatore), clip da 20 s (10 s per lato), stesso tempo/key/fase di beat/RMS. Il membro positivo ha un evento di arrivo esplicito sul seam (crash + layer di lead); il negativo è lo stesso audio base senza quell'arrivo.

- Baseline di continuità RMS: AUC held-out **0,348**, paired accuracy **0,375**.
- Novelty tipo SSM-Net (front-end pubblicato, kernel checkerboard, nessun peso): AUC **1,000**, paired accuracy **1,000**.
- Delta AUC novelty − baseline: **0,652**, CI bootstrap 95% **[0,477, 0,808]** su 32 coppie.
- Onset al seam da solo: AUC **1,000** (il contrasto sintetico è troppo facile).
- Il punteggio combinato novelty+onset è saturo (AUC 1,000; accuratezza al threshold di train 1,000).

**Interpretazione.** La baseline RMS viene ingannata dal transient di arrivo; la novelty locale separa i due casi. Questo dimostra che il front-end e la misura funzionano e che un proxy di loudness non è sufficiente. **Non dimostra** che una transizione reale sia buona, che il rap/trap sia coperto, o che una novelty alta sia sempre preferibile: il contrasto è costruito e saturo. La validazione esterna con ascoltatori e cue reali resta il blocker principale.

## 7. Decisione di adozione

La via più concreta è **SSM-Net-style local boundary novelty come evidenza di candidato**, non come gate o reward aggregato. Va accompagnata da downbeat/beat (Beat This! o All-In-One), da `word_guard`/frase e da un'astensione esplicita. La validazione deve usare coppie esterne held-out (idealmente cue M-DJCUE e transizioni Medley2K, con audio e licenze verificate) e split per traccia/artista, non per singola transizione. SongEval e la qualità dell'intera canzone restano escluse come gate: non sono state validate su transizioni.
