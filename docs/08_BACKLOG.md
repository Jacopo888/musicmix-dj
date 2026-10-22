# 08 — Backlog atomico (task implementabili e verificabili)

Ogni ticket ha un modulo proprietario, output, test e paper di riferimento. **Ordine di priorità**: ricerca valida prima di effetti audio.

| Task | Modulo | Output richiesto | Definizione di completamento |
|---|---|---|---|
| T001 | M00 | ingest FFmpeg, WAV 48k, hash, metadata | Test offset e duration round-trip superati |
| T002 | M13 | run manifest e JSONL di log | Ogni artefatto ha ID/versione e lineage |
| T003 | M02 | beat/downbeat/section extractor adapter | Quality report e fixture con gold |
| T004 | M05 | chroma, RMS, onset, timbre per finestra | FeatureStore deterministico con confidence |
| T005 | M06 | candidate generator downbeat arbitrarie | Recall su golden cue, motivo per candidati scartati |
| T006 | M07 | pair features harmony/rhythm/context | Tests su note/tempi sintetici |
| T007 | M08 | baseline Spotify-like direzionale | Breakdown score e regression fixtures |
| T008 | M09 | preview stereo neutro | No clipping/click su fixture, stesso policy per tutti |
| T009 | M10 | ascolti pilot B0 vs B1 | Dataset giudizi anonimi ed esito riproducibile |
| T010 | M01 | separator provider + caching stems | Stem offset test e fallback |
| T011 | M03 | aligner cantato per lyrics note | Subset annotato, confidenza per parola |
| T012 | M03 | pitch F0 + voiced mask | Differenza fra Hz assoluti e semitoni relativi |
| T013 | M06 | word-to-word candidate generator | Stessa parola in punti arbitrari entrambe le tracce |
| T014 | M07/M08 | DTW intonazione / prosodia / musical context | Hard negatives stessa parola ma mismatch beat/harmony |
| T015 | M04 | riff generator con interval patterns | Recall e false positives su fixture |
| T016 | M08 | diversity-aware top-K e ablation | `no_lyrics`, `no_F0`, `no_melody`, `no_harmony` |
| T017 | M10 | studio cieco comparativo P1 vs B1/B2 | Gate A verificato e limiti riportati |
| T018 | M11 | N-song beam planner | Path valido con ingresso<uscita e copertura |
| T019 | M12 | multistem entry/exit rendering | Stessi cue per confronto con M09 |
| T020 | M12 | time stretch/pitch and limiter | Artifact/loudness report per transition |
| T021 | M10 | studio cue-fixed rendering | Gate B verificato separatamente da Gate A |
| T022 | M13 | diagnosis report per candidato | Scomposizione cause e golden replays |

### Dipendenze critiche

- T001→T003/T004→T005/T006→T007→T008→T009.
- T010→T011/T012→T013→T014→T017.
- T015 alimenta T016, ma non blocca baseline M1.
- T017 prima di T018/T019/T020, salvo prototipi isolati.

### Letture minime per il primo sprint tecnico

1. [Spotify 2017](https://archives.ismir.net/ismir2017/paper/000086.pdf) per T005/T007.
2. [All-In-One 2023](https://arxiv.org/abs/2307.16425) per T003.
3. [AutoMashUpper 2014](https://repositorio.inesctec.pt/bitstreams/97188d38-9bf1-46f9-99a3-08d3b656ca4f/download) per T004/T006.
4. [Lee 2015](https://www.ismir2015.uma.es/articles/302_Paper.pdf) per T007/T014.
5. [Phoneme Lyrics 2021](https://schufo.github.io/files/2021_TASLP3091817.pdf) e [CREPE 2018](https://arxiv.org/abs/1802.06182) per T011–T014.
