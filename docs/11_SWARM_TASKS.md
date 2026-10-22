# 11 — Task board dello swarm

**Aggiornato:** 18 ottobre 2026  ·  **Stato:** board di lavoro, non certificato

Questa board mantiene gli ID del backlog atomico e rende esplicita la
differenza fra codice già eseguibile, lavoro parziale e lavoro futuro. Un
task `PARZIALE` non soddisfa ancora la definizione di completamento di
`docs/08_BACKLOG.md`; `FUTURO` non è stato implementato. La suite automatica
verde copre i contratti della baseline, non la qualità musicale né il Gate A.

## Milestone

### Aggiornamento sprint struttura/voce

- **V4:** importate sei osservazioni umane V3 con mapping confermato; conservato il primo caso sufficiente senza training automatico. Corretti compatibilità word-gap/ASR e riuso cue. Top-K con diversità su entrambe le timeline, controllo casuale sui cue validi e posizioni A/B bilanciate. Quattro proposte downbeat e cinque beat per direzione; 94 test, QA su 18 WAV. Server locale con run/test ID e salvataggio diretto in `giudizi json`. Gate A ancora aperto; dettaglio in `15_DIVERSITA_E_ASCOLTI_V4.md`.

- **T003 parziale avanzato:** Beat This! small0 operativo sui due brani; sezioni spettrali euristiche. Manca un gold ritmico annotato.
- **T004/T006/T007 parziali avanzati:** contesti direzionali di quattro/otto beat, sequenze cromatiche e ranking con vincoli vocali. Non è una riproduzione certificata di una baseline pubblicata.
- **T009/T017 parziali, Gate A aperto:** preparazione di confronti A/B a renderer fisso; mancano giudizi e controllo random valido/baseline forte.
- **T010/T011/T014 parziali avanzati:** Demucs htdemucs e faster-whisper small.en eseguiti sulle tracce complete; protezione dei fade, gestione incerto e regressione del caso segnalato. Mancano forced alignment validato sul cantato e benchmark annotato.
- **T022 parziale avanzato:** snapshot evidenze, hash codice/parametri/audio, diagnosi e verifica automatica delle run. Risultati correnti in `13_STRUTTURA_E_FRASI.md`; il resto della board conserva lo storico e i requisiti ancora aperti.

| Milestone | Perimetro | Stato verificabile | Prossimo criterio di uscita |
|---|---|---|---|
| M0 — ingest e lineage | T001–T002 | **PARZIALE**: `audio.py`, ingest, manifest/eventi e hash sono usabili; mancano round-trip offset e QA completo degli artefatti | fixture audio con durata/offset verificati e lineage completo per ogni output |
| M1 — cue, retrieval e preview | T003–T008 | **PARZIALE ma demo eseguibile**: baseline onset/FFT, ricerca e renderer neutro sono collegati; test automatici correnti passano, ma non è una replica dei paper | provider strutturale oppure fallback dichiarato, regioni musicali, ranking direzionale e fixture percettive |
| M2 — pilot | T009 | **FUTURO** | dataset di giudizi anonimi con B0 e confronto controllato |
| M3 — matching vocale | T010–T014 | **FUTURO** | stems, lyrics/F0, DTW e gold subset |
| M4 — matching strumentale | T015 | **FUTURO** | riff/intervalli con recall e negativi difficili |
| M5 — ranker interpretabile | T007/T016 | **PARZIALE** | ranking completo, ablation e confronto percettivo |
| M6 — Gate A | T017 | **VALIDAZIONE UMANA DA ESEGUIRE** | studio cieco preregistrato con evidenza di preferenza |
| M7 — N brani | T018 | **FUTURO, dopo Gate A** | path multi-brano valido e copertura |
| M8 — renderer multistem | T019 | **FUTURO, dopo Gate A** | confronto con renderer neutro a cue fissi |
| M9 — correzioni DSP | T020 | **FUTURO** | tempo/pitch/loudness e report artefatti |
| M10 — render policy e Gate B | T021 | **FUTURO** | studio a cue fissi |
| M11 — packaging | CLI/cache/report | **PARZIALE** | installazione riproducibile, cache e QA esteso |

## Ownership corrente

| Area | Moduli/file di riferimento | Responsabilità attuale |
|---|---|---|
| Audio e analisi | `src/musicmix/audio.py`, `src/musicmix/analysis.py`, ingest | decoder PCM/FFmpeg, descriptor FFT, RMS, chroma snapshot, flux e cue `onset_grid`; fallback deterministico |
| Retrieval | `src/musicmix/search.py`, ranking | griglia coppie, feature interpretabili, cap/spacing e score con breakdown; `vertical`/`creative` restano unknown quando mancano dati |
| Coordinamento | CLI, pipeline, `src/musicmix/preview.py` | manifest, JSONL, lineage, report e `neutral_linear_v1`; integrazione end-to-end |
| Ricerca e umano | provider M02, stems, lyrics, F0, pilot | lavoro futuro o Gate A; nessun risultato umano va inferito dai test software |

## T001–T022

| ID | Modulo | Stato | Risultato corrente / prossimo passo |
|---|---|---|---|
| T001 | M00 | **PARZIALE — baseline** | WAV PCM, ingest FFmpeg 48 kHz, hash e metadata sono presenti. Aggiungere fixture round-trip per durata, sample zero ed eventuale encoder delay prima di dichiararlo completo. |
| T002 | M13 | **PARZIALE — baseline** | Manifest di run, eventi JSONL, ID/versioni e lineage di base sono presenti nella pipeline. Verificare che ogni preview/report/rejection sia risalibile a input e configurazione. |
| T003 | M02 | **PARZIALE — fallback** | `analysis.py` emette `onset_grid` da FFT/flux; non emette beat, downbeat o section. Implementare adapter All-In-One versionato con confidence e quality report, mantenendo il fallback esplicito. |
| T004 | M05 | **PARZIALE — fallback** | Sono disponibili chroma snapshot, RMS, onset e timbre deterministici. Manca un FeatureStore beat-sincrono con confidence, versioning e finestre coerenti con il provider strutturale. |
| T005 | M06 | **PARZIALE — baseline** | La ricerca genera coppie sulla griglia onset con cap e spacing. Servono candidati su finestre di n beat/downbeat, recall su cue golden e motivi di scarto; evitare pruning intro/outro implicito. |
| T006 | M07 | **PARZIALE — baseline** | Il pair score usa cosine chroma/timbre e rapporti RMS/onset/tempo su punti locali; non è ancora harmony/rhythm/context a regioni. Aggiungere fixture sintetiche per note, tempo, direzione A→B e B→A. |
| T007 | M08 | **PARZIALE — baseline** | Ranker interpretabile e breakdown esistono, con pesi correnti documentati come ipotesi. Mancano matrici Spotify-like standardizzate, vocalness/boundary e contesti specifici per direzione. |
| T008 | M09 | **DISPONIBILE — baseline** | `neutral_linear_v1` produce preview stereo con stessa policy e i test automatici correnti passano. Fixture di durata, headroom ed estremi del clip sono presenti; restano QA sui click interni e report di loudness prima del confronto umano. |
| T009 | M10 | **FUTURO** | Pilot B0 vs baseline musicale su cue validi e casuali, con renderer identico, randomizzazione e giudizi anonimi. Dipende da T003–T008 e da un corpus lecito. |
| T010 | M01 | **FUTURO** | Provider separator con cache, offset test e fallback esplicito. Nessuna verticalità di Lee va dichiarata finché non esistono stem/ruoli utilizzabili. |
| T011 | M03 | **FUTURO** | Aligner lyrics cantato con parola, fonemi, onset/offset e confidence su subset annotato; il link fornito per il paper 2021 è indisponibile, quindi serve una sorgente e un gold subset tracciati. |
| T012 | M03 | **FUTURO** | F0 CREPE o provider equivalente su voce separata, voiced mask e semitoni relativi. Misurare errore a 10/50 cent sul corpus del progetto; non trasferire i numeri CREPE automaticamente. |
| T013 | M06 | **FUTURO** | Candidate generator word-to-word, includendo occorrenze arbitrarie in entrambe le tracce e fallback senza lyrics. Richiede confidence T011 e policy per testi incompleti. |
| T014 | M07/M08 | **FUTURO** | DTW di intonazione/prosodia e contesto armonico/beat con hard negatives (stessa parola, mismatch musicale). Separare ablation da qualsiasi score aggregato. |
| T015 | M04 | **FUTURO** | Riff/interval-pattern generator con recall e false positive su fixture. Non blocca il primo baseline M1. |
| T016 | M08 | **FUTURO** | Top-K diversity-aware e ablation `no_lyrics`, `no_F0`, `no_melody`, `no_harmony`; preregistrare criteri e mantenere i breakdown. |
| T017 | M10 | **BLOCCATO DA VALIDAZIONE UMANA** | Studio cieco comparativo P1 vs B1/B2, con cue selezionati da ogni sistema, renderer uguale e intervalli di confidenza. È il **Gate A**: nessun agente può marcarlo superato dai soli test automatici. |
| T018 | M11 | **FUTURO — dopo Gate A** | Beam planner N-song con path valido, ingresso/uscita e copertura; usare transizioni già valutate e registrare perché un ramo viene scartato. |
| T019 | M12 | **FUTURO — dopo Gate A** | Rendering multistem con entry/exit sugli stessi cue usati nel pilot; confronto isolato con M09 e offset verificati. |
| T020 | M12 | **FUTURO — dopo Gate A** | Time-stretch/pitch/limiter con artifact report e loudness report. Non confondere il rendering trasformativo con la validazione del ranking. |
| T021 | M10 | **FUTURO — Gate B separato** | Studio cue-fixed per valutare il renderer a parità di coppia e cue. Gate B resta distinto da Gate A, che riguarda la scelta del candidato. |
| T022 | M13 | **PARZIALE — baseline** | Candidate records, rejection reasons, manifest e log esistono. Completare diagnosis report con scomposizione per feature, golden replays, causa-effetto e collegamento alla preview. |

## Prossimo sprint operativo

1. **T003:** definire l'interfaccia del provider (`beat`, `downbeat`,
   `boundary`, `label`, `confidence`, versione) e il report di qualità; se il
   provider non è disponibile, fallire con stato esplicito e usare B0.
2. **T005–T007:** generare finestre direzionali su tutto il brano, mantenere
   le matrici per componente e confrontare A→B con B→A; non chiamare questa
   baseline una riproduzione di Spotify finché mancano beat/downbeat/section,
   vocalness e standardizzazione delle matrici.
3. **T009:** costruire prima il pilot valid-vs-random con lo stesso renderer
   neutro; raccogliere recall/oracle e ascolti senza rivelare il metodo.
4. **T017:** consegnare al revisore umano protocollo, audio, random seed,
   breakdown e limiti. Il Gate A resta **non verificato** fino a quella
   decisione.

## Gate A umano

Gate A richiede una decisione esterna al codice: almeno un confronto cieco fra
B0 (onset-grid heuristic), una baseline musicale documentata e la proposta,
su cue validi e su un controllo casuale condizionato agli stessi vincoli. Ogni
condizione deve usare lo stesso renderer, la stessa policy di gain e la stessa
durata; il report deve conservare seed, coppie, direzione, ranking, giudizi,
intervalli di confidenza e ablation. Il risultato deve distinguere qualità
del cue, qualità del ranking e artefatti del rendering. Fino alla revisione e
alla firma del revisore umano, T017 e M4 restano aperti e il progetto non può
essere descritto come validato percettivamente.
