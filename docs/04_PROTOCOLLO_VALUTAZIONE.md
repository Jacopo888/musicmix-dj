# 04 — Protocollo sperimentale: dimostrare qualità musicale

## Ipotesi pre-registrata

**H0:** il candidato selezionato dal nostro ranker non è preferito più frequentemente del candidato di una baseline musicale forte, dopo standardizzazione del renderer.  
**H1:** il candidato del nostro ranker è preferito più frequentemente.  
**Outcome primario:** probabilità di preferenza cieca vs miglior baseline (escludendo pari merito o trattandoli separatamente in analisi predefinita).

## Dataset e split

- **Pilot:** 20–40 canzoni di generi/lingue/tempi differenti, usate legittimamente, con sottogruppo di lyrics e riff verificati manualmente. Dimensioni *indicative*, non stima di potenza statistica.
- **Split:** non condividere stessa canzone tra train/validation/test di un eventuale reranker; idealmente separare anche artisti/versioni live/remastering.
- **Sampling:** coppie stratificate per compatibilità armonica/rhythmic, genere simile/diverso, presenza/assenza di voce, parole comuni; conservare coppie negative.
- **Fonte di dati DJ:** [Computational Analysis of Real-World DJ Mixes (2020)](https://arxiv.org/abs/2008.10267); dati studio [Automatic DJ Transitions (2022)](https://arxiv.org/abs/2110.06525).

## Algoritmi da confrontare

| Etichetta cieca | Sistema | Cosa misura |
|---|---|---|
| B0 | Casualità condizionata: scelta tra downbeat/phrase cue validi, rispettando gli stessi limiti BPM/key | Chance **non caricaturale** |
| B1 | Spotify-like cost matrix: beat, timbre, harmony, vocal activity, section boundary. **Niente pruning obbligatorio intro/outro** | Baseline tecnica forte |
| B2 | AutoMashUpper-like: armonia + ritmo + bilanciamento spettrale | Baseline musicale alternativa |
| P0 | Il nostro score `structure+compatibility` (senza word/pitch) | Beneficio della nostra implementazione |
| P1 | P0 + word/phoneme/F0 e riff | Contributo specifico della proposta |

Paper: [Spotify 2017](https://archives.ismir.net/ismir2017/paper/000086.pdf), [AutoMashUpper 2014](https://repositorio.inesctec.pt/bitstreams/97188d38-9bf1-46f9-99a3-08d3b656ca4f/download), [Lee 2015](https://www.ismir2015.uma.es/articles/302_Paper.pdf).

## Preview per Step I

- Stessa durata del contesto (ad es. 10 s prima + 10 s dopo, da fissare nel pilot); volume normalizzato e identico trattamento.
- Render minimalista: crossfade o cut identici per tutti i candidati **di un medesimo esperimento** e niente correzione creativa selettiva.
- Se una parola richiede micro-alignment, predefinire una policy comune deterministica; confrontare anche sub-studi word-only a condizioni corrette.
- Esportare clip stereo anonime + `manifest.csv` di decodifica, senza rivelare sistema al valutatore.

## Scheda ascolto

I valutatori ricevono A e B della stessa coppia di brani (ordine random). Domande distinte:

1. **Preferenza complessiva:** A / B / pari merito / entrambi inaccettabili.
2. **Musicalità del collegamento** 1–5: il passaggio ha senso musicalmente?
3. **Continuità percepita** 1–5: qualcosa del brano A sembra proseguire coerentemente in B?
4. **Interesse creativo** 1–5: la scelta è interessante/non banale?
5. **Disturbo tecnico** 1–5: click, pitch artificiale, risonanze, volume, voce sovrapposta involontariamente?
6. **Motivo libero (facoltativo)**: che cosa non funziona e quando?

Il punto 5 è una variabile di controllo per Step I e outcome primario secondario per Step II.

## Assegnazione, cecità e statistica

- Randomizzazione dell'ordine A/B, no label di algoritmo, limite a ripetizioni stesso brano per evitare familiarità.
- Pilot iniziale con 20–30 ascoltatori è utile per trovare problemi, **non** garantisce potenza adeguata; pianificare numerosità usando varianza osservata.
- Unit of clustering: coppia di brani e ascoltatore. Evitare di conteggiare decine di render della stessa coppia come osservazioni indipendenti.
- Report: stima di win rate, CI bootstrap per blocchi/pair o modello gerarchico se appropriato, analisi per modalità; dichiarare numero di annotazioni, variabilità e casi esclusi.
- Registrare multiple comparison e scoring deciso **prima** di guardare i risultati del test set.

## Ablation e diagnosi

| Esperimento | Domanda causale sul software |
|---|---|
| `no_lyrics` | Le parole contribuiscono davvero al ranking? |
| `no_F0` | L'intonazione porta valore oltre la semplice identità testuale? |
| `no_melody` | Note/riff aggiungono vantaggio? |
| `no_harmony` | Il vantaggio è solo matching di parola sopra armonie sbagliate? |
| `no_structure` | La priorità delle frasi/dowbeats porta un vero beneficio? |
| `no_separation` | Gli artefatti stem danneggiano più di quanto aiutino? |
| `no_diversity` | La qualità dei top-K aumenta a scapito della varietà? |

Ogni ablation conserva stesso split, set di input, seed, preview policy e metrics. In Step II si confrontano **cue identici** tra `neutral_renderer` e `advanced_renderer`.

## Gate A e Gate B

**Gate A (ricerca):** miglioramento credibile rispetto alla baseline musicale più forte, validato su più coppie/brani e spiegabile anche con le ablation. Se non è osservato: rivedere candidate generation, oracolo manuale e labels prima di usare più effetti.

**Gate B (effetti):** preferenza positiva per il rendering migliorato **agli stessi punti**; riduzione dei difetti udibili. Se la resa migliora ma la scelta cue non cambia, riportare questo come risultato del renderer, non del retrieval.

## Link metodologici

- [Automatic DJ Transitions with Differentiable Audio Effects and GANs (2022)](https://arxiv.org/abs/2110.06525)
- [Automatic Playlist Sequencing and Transitions (2017)](https://archives.ismir.net/ismir2017/paper/000086.pdf)
- [Automatic Mashup Creation by Considering Vertical and Horizontal Mashabilities (2015)](https://www.ismir2015.uma.es/articles/302_Paper.pdf)
- [COCOLA (2025)](https://arxiv.org/abs/2404.16969) — usa preferenze umane sul proprio task, non ground truth universale.
