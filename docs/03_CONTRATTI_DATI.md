# 03 — Contratti dati e interfacce

## Filosofia

Ogni modulo consuma e restituisce **oggetti versionati**. Evitare feature nascoste in singoli notebook, offset non convertiti o punteggi senza giustificazione. Se un modulo fallisce, gli altri possono essere testati con fixture/golden data.

## Identificatori

- `run_id`: UUID o digest univoco per un'esecuzione; `track_id`: derivato dal contenuto + identificatore di libreria, mai dal solo titolo.
- `candidate_id`: digest deterministico di tracce, posizioni, modalità e versione generatore (non del punteggio, che può cambiare).
- `schema_version`: versione del contratto; `model_version`: versione del modello/pesi; `params_sha256`: hash delle opzioni.

## Timeline e precisione

- Timeline canonica: **48.000 samples/s**, PCM stereo per il rendering; tutti gli eventi persistiti in `start_sample/end_sample` su quella timeline, intervalli semiaperti `[start,end)`.
- `start_sample >= 0`, `end_sample <= track_length_samples`, `end_sample > start_sample` per segmenti non vuoti.
- Timecodes forniti da modelli a sample rate diversi devono essere ricondotti alla timeline canonica con resampling/offset documentati.
- Beat index non equivale a tempo assoluto: mantenere entrambi, più BPM locale e sua confidenza.

## Oggetti principali

### TrackAsset

`track_id, input_uri, source_sha256, canonical_audio_uri, sample_rate, channels, duration_samples, decoder_version, license_tag, metadata, ingest_warnings`.

### StemRef

`track_id, stem_type (vocals|drums|bass|other), audio_uri, separator_id, model_hash, start_offset_samples, quality_report`. L'assenza dello stem è un valore gestito, non una stringa URI fittizia.

### MusicalEvents

`beat`/`downbeat`: sample, confidence; `Section`: start, end, class, confidence; `WordSpan`: string, lemma, phoneme list, sample span, `confidence_alignment`; `PitchContour`: campioni F0 Hz con probabilità voiced; `Motif`: onset and pitch intervals, estensione temporale, confidence.

### TransitionCandidate

`candidate_id, track_a, track_b, cue_out_sample, cue_in_sample, mode, generator_id, context_before_after, feature_values, feature_confidence, proposed_alignment, flags, reason_codes`. **A→B distinto da B→A** e dal crossfade simultaneo. Non inserire la stima di qualità audio nell'identità del candidato.

### ScoreReport

`score_total, compatible, per_dimension, penalties, unavailable_features, passed_constraints, score_model_id, why, why_not`. Score normalizzato **solo tra le dimensioni valide**, con disponibilità esplicita; escludere candidati inaffidabili con policy dichiarata.

### RenderArtifact

`candidate_id, render_policy, render_version, peak, loudness, artifacts, sha256, path, audio_duration_samples, warnings`.

### ListeningJudgement

`study_id, participant_pseudonym, song_pair_id, trial_id, left_render_id, right_render_id, blinded_order_seed, answer, musicality, continuity, creative_interest, heard_on, flags`. Registrare risposte senza memorizzare identità personali superflue.

## Interfacce Python suggerite

```python
from typing import Protocol

class Extractor(Protocol):
    def extract(self, track_asset, context): ...  # produce artefatti e quality_report

class CandidateGenerator(Protocol):
    def generate(self, track_a, track_b, feature_store, context): ...

class PairFeatureComputer(Protocol):
    def compute(self, candidate, feature_store): ...

class Ranker(Protocol):
    def rank(self, candidates, features, policy): ...

class PreviewRenderer(Protocol):
    def render(self, candidate, policy): ...

class EvaluationStore(Protocol):
    def record_trial(self, trial_result): ...
```

Non sono implementazioni operative: definiscono le **porte** per mock/test e adapter sostituibili. Il sottoprogetto `src/musicmix/` include un ranker minimo realmente eseguibile su feature precalcolate.

## Eventi di log standard (JSONL)

```json
{"run_id":"run_001","candidate_id":"cand_001","module":"M03_lyrics", "stage":"extract", "status":"low_confidence", "error_code":"SUNG_WORD_ALIGNMENT", "model_version":"candidate","details":{"confidence":0.31}}
```

Ogni evento include almeno: `run_id`, `module`, `stage`, `status`, `timestamp_utc`, `duration_ms`, `input_hash`, `output_hash`, `warnings`. Campi di audio/testo sensibili non devono essere dumpati direttamente nel log.

## Schema verificabile

Vedi [`../schemas/transition_record.schema.json`](../schemas/transition_record.schema.json) e fixture [`../examples/candidate_example.json`](../examples/candidate_example.json). Gli schemi JSON sono contratti d'interfaccia: non pretendono di codificare tutta la semantica musicale.

## Link di ricerca

- [Bittner et al. (2017): cue matrix e feature](https://archives.ismir.net/ismir2017/paper/000086.pdf)
- [Kim & Nam (2023): beat/downbeat/sections](https://arxiv.org/abs/2307.16425)
- [Schulze-Forster et al. (2021): phoneme alignment](https://schufo.github.io/files/2021_TASLP3091817.pdf)
- [Lee et al. (2015): compatibilità orizzontale/verticale](https://www.ismir2015.uma.es/articles/302_Paper.pdf)
