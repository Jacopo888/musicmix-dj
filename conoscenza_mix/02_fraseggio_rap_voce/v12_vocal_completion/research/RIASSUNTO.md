# Riassunto della ricerca e dell'implementazione vocale

Data: 2026-10-20. Perimetro: descrittori misurati di frase/prosodia attorno a una cue,
non un classificatore di closure. Le fonti complete, URL, DOI, hash, licenza, sezioni
lette e tipo di estrazione sono in `research/sources.json`.

## Fonti primarie e cosa sostengono

**Condit-Schultz (2017), MCFlow: A Digital Corpus of Rap Transcriptions**, DOI
`10.18061/emr.v11i2.4961`. Evidenza misurata sul corpus: 124 brani, 374 versi,
5.803 misure; il paper descrive prosodia, sintassi, rime e fraseggio. Il punto
metodologico letto nel testo è che i confini prosodici sono codificati separatamente
dalla sintassi; un confine sintattico e uno prosodico non coincidono sempre. La
`break spine` ammette confini deboli, di sotto-frase, di frase e iper-frase. Il paper
cita ritmo, stress e contorni di altezza come indizi di confine; una pausa/rest è un
indizio, non una prova. Questi sono risultati descrittivi del corpus simbolico, non
prestazioni audio né una validazione delle nostre soglie.

**Duinker (2020), dissertation McGill**, repository open access. Evidenza dichiarata:
472 versi da 160 brani; teoria dei profili di flow e cinque regole di segmentazione
(rima, sintassi/soggetto, respiro, somiglianza ritmica e conflitti tra indizi). È
teoria musicologica basata su trascrizioni manuali, non un modello audio addestrato.
Conseguenza per il codice: i canali vanno tenuti separati e un singolo respiro o una
singola rima non bastano.

**Duinker (2021), Music Theory Spectrum**, DOI `10.1093/mts/mtab003`. Accesso:
solo abstract via Crossref, perché la pagina dell'editore ha restituito 403. L'abstract
sostiene che la segmentazione del flow coinvolge aspetti linguistici, sintattici,
articolatori e corporei, e che l'allineamento/disallineamento col beat produce varietà.
Non è stato letto il testo completo e non viene usato per soglie.

**de Cheveigné & Kawahara (2002), YIN**, DOI `10.1121/1.1458024`. L'algoritmo usa la
funzione differenza, la cumulative mean normalized difference `d'(tau)` e una soglia
assoluta; se nessun minimo scende sotto soglia si sceglie il minimo globale. Nel
codice la CMND è ora accumulata da `tau=1..max_lag` e divisa per il `tau` reale,
poi la ricerca è ristretta a `min_lag..max_lag`. La soglia del paper è 0.1. YIN è
un estimatore DSP, non un modello appreso e non una probabilità di voiced.

**Silero VAD 6.2.3**, GitHub `https://github.com/snakers4/silero-vad`, licenza MIT.
Pesi JIT ufficiali inclusi nel pacchetto, eseguiti in locale su CPU con un thread.
Il modello è un VAD appreso: dice speech/non-speech, non confine di frase rap né
closure. Nel run è un canale indipendente, non un oracolo.

**torchcrepe / CREPE** è stato valutato come possibile pitch appreso (pesi tiny/full
ufficiali, MIT), ma non è stato installato: `torchaudio` non era disponibile e YIN
bastava per i descrittori richiesti. Non viene presentato come eseguito.

## Implementazione

`src/musicmix/vocal_completion.py` espone:

- `extract_vocal_features(samples, sample_rate, canonical_start_sample=..., use_pretrained_vad=...)`
- `describe_vocal_cue(track, cue_sample, words=..., segments=...)`
- `analyze_vocal_completion(...)`
- `rank_phrase_end_candidates(...)`
- `analyze_staged_file(path, clock, cue_canonical_sample, ...)`
- `run_silero_vad(...)` con esito esplicito `available`/`reason` se il runtime manca.

Le coordinate pubbliche sono campioni canonici a 48 kHz. La cue canonica viene
convertita in locale con `cue_sample - canonical_start_sample`; cue fuori dallo
span canonico sollevano `VocalCompletionClockError`. Le finestre ai bordi sono
clampate senza cambiare la griglia dei campioni.

Canali separati per ogni cue:

- attività vocale a energia relativa con chiusura dei buchi di 120 ms;
- attacchi da flux spettrale normalizzato, distanza minima 90 ms;
- pitch YIN con CMND completa, chunk di 256 frame, continuità ottava e disponibilità
  del contorno; nessuna disponibilità viene inventata;
- terminal lengthening misurato come durata dall'ultimo attacco al rilascio,
  rapportata all'intervallo mediano tra attacchi nel contesto;
- respiro come flatness/HF/ZCR nella finestra non vocale, indice non calibrato;
- ASR parole/segmenti opzionali, con `asr_segment_is_not_prosodic_phrase: true`
  e nessuna inferenza semantica;
- VAD preaddestrato Silero come canale indipendente e confronto con il VAD DSP;
- uscita A (`outgoing_completion_descriptors`) e ingresso B
  (`incoming_pickup_descriptors`) tenuti asimmetrici.

Il punteggio composito è dichiarato `descriptor_heuristic_not_probability_not_closure`.
I pesi sono priors ingegneristici scritti nel JSON, non fittati su giudizi. Il modulo
non emette etichette `closure`, `resolved`, `chorus` o `phrase_boundary_certified`.

## Evidenza misurata nel run

Comando: `python tools/run_vocal_completion_pilot.py` con `PYTHONPATH=src`.
File elaborati: `inputs/vocal_01.wav`, `inputs/vocal_02.wav`, 77 cue ciascuno
(passo 0.25 s, bordi 0.5 s). Gli hash sono verificati contro `inputs/source_clocks.json`
e la sorgente canonica parte dal campione 3.360.000.

Silero VAD sul primo file trova due intervalli: 0.0-12.0 s e 13.7-20.0 s.
La cue 12.00 s è il primo candidato del ranking DSP, e cade esattamente sul confine
del primo intervallo (non dentro lo speech). Sul secondo file Silero trova più
intervalli; la cue 5.00 s del ranking DSP cade dentro un intervallo Silero, quindi
i due canali sono in conflitto. Il report del run conserva questo conflitto invece
di trasformarlo in una closure.

I punteggi non sono accuratezze: non ci sono annotazioni di confini di frase nel run.
Il ranking misura solo l'ordinamento dei descrittori. Le clip di ascolto non sono
state prodotte; un'eventuale clip futura deve restare di 20 s (10 s per lato).

## Incertezza residua

- Nessuna ground truth di frase/closure rap e nessun giudizio utente: impossibile
  riportare accuratezza, F1 o una probabilità calibrata.
- ASR non eseguito: il canale parole è assente e viene dichiarato tale. Un segmento
  ASR non viene mai promosso a frase prosodica.
- YIN può sbagliare ottava; la correzione a mediana mobile è euristica e viene
  esplicitata. Il confronto con `librosa.yin` è opzionale e nel run è stato saltato
  perché librosa non era installato.
- Silero VAD e VAD DSP possono divergere su code vocali, respiri o riverbero.
- Il respiro è un indice broadband non calibrato, non un evento certificato.
- Il flusso A/B è asimmetrico per costruzione, ma non è una decisione di mix.
