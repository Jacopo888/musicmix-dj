# 05 — Debugging: isolare cause, conseguenze e responsabilità

## Principio operativo

Per ogni transizione produrre un **pacchetto di evidenza**: waveform di origine, segmento annotato, feature estratte, candidati generati/scartati, breakdown del punteggio, audio preview, valutazioni. La prima fase che perde l'informazione necessaria è il sospetto principale. Non attribuire al ranker un cue che M02 non ha mai generato.

## Albero diagnostico

```text
L'utente dice: «Questo switch non funziona»
  |
  +-- Il collegamento sarebbe musicalmente valido con un taglio migliore?
  |     +-- SI --> M09/M12 (render, allineamento, stem, clipping)
  |     +-- NO --> È stato recuperato un punto valido tra i candidati?
  |                  +-- NO --> M00/M02/M03/M04/M05/M06 (input/estrazione/retrieval)
  |                  +-- SI --> M07/M08 (features / ranker / trasformabilità)
  |
  +-- Manca una parola/nota fondamentale? --> M03/M04 + confidenza e logs
  +-- Le scelte A→B e B→A sono simili senza motivo? --> M07/M08 (direzione)
  +-- Tutti i top-K sono quasi identici? --> M08 diversity / M06 candidate quota
```

## Failure taxonomy verificabile

| Sintomo | Modulo probabile | Test di isolamento | Intervento possibile |
|---|---|---|---|
| Tutti i tempi sono traslati 30 ms | M00 | impulso noto al sample canonico, confrontare decoder | conversione/offset unificati |
| Entrata di B sul levare | M02/M07 | confrontare downbeat annotato vs calcolato | cambiare modello o gate di confidenza |
| «Stessa parola» in realtà sono parole diverse | M03 | gold lyrics e ascolto vocal stem | aligner per canto / confidence |
| F0 alta su consonanti sorde | M03 | voiced mask + spettrogramma | maschere confidenza / pitch smoothing |
| Riff simili solo perché stessa tonica | M04 | sequenze intervalli e onset | fingerprint più discriminante |
| Un ottimo cue non compare fra i candidati | M06 | top-K coverage rispetto a gold | cambiare retrieval/quote e soglie |
| Lo score premia il match testuale stonato | M07/M08 | diff breakdown: lyrics vs harmony | gate armonico, calibratura |
| A→B e B→A trattati uguale | M07/M08 | swap del ruolo e contesto | features direzionali |
| Click udibile solo durante fade | M09/M12 | render a cue fissi, segnale sintetico | crossfade zero-cross/transient |
| Voce robotica dopo pitch shift | M12 | stem originale vs trasformato | limitare shift / modello DSP diverso |
| Test umano migliora solo con un renderer | M09/M10 | stesso cue, render A/B e test fattoriale | separare qualità di scelta/render |
| Top 5 praticamente identiche | M08 | distanza tra risultati stessi segmenti | diversify, MMR, quota per modalità |

## Logging obbligatorio per `candidate_id`

- `source_hash`, `source_time_in_samples`, `run_id`, `dependency_versions`.
- `cue_generation_reason`, `candidate_generation_stage`, `rejected_reasons`.
- `features`: `{value, confidence, producer_version}`.
- `score`: contributo numerico **per dimensione** + weight + penalty + version.
- `render`: cue, mix policy, time-stretch factor, pitch-shift semitoni, stem roles, peak e warnings.
- `evaluation`: blinded ID e trial associato.

## Esperimenti di causal debugging

1. **Golden extraction:** sostituisci M03 con lyrics e F0 annotati correttamente, lascia invariato il ranker. Se il ranking migliora, il problema nasce a monte.
2. **Oracle retrieval:** aggiungi manualmente un cue corretto tra i candidati M06 e controlla se il ranker lo recupera; separa recall vs ranking.
3. **Feature ablation:** fissa candidati e rimuovi solo F0 o armonia; non toccare audio e renderer.
4. **Render invariant:** stesso `candidate_id`, due render, stessi punti: un eventuale miglioramento è attribuibile al renderer.
5. **Score audit:** stampa i 10 candidati più alti e i 10 rifiutati per capire se la penalità di trasformazione è sproporzionata.

## Metriche waterfall, da registrare ad ogni run

`n_tracks_ingested → n_tracks_valid → n_events_extracted → n_candidates_generated → n_candidates_hard_valid → n_candidates_ranked → n_diverse_selected → n_previews_rendered → n_judged`

Ogni salto ha count per modulo, reason code e esempio audio. Salvare l'**oracolo manuale** su alcune coppie, altrimenti non sapremo se l'assenza di buone transizioni deriva da sorgenti incompatibili o dalla ricerca.

## Paper che motivano questa diagnostica

- [Bittner et al. 2017](https://archives.ismir.net/ismir2017/paper/000086.pdf): beat tracking e fallimenti di crossfade.
- [All-In-One 2023](https://arxiv.org/abs/2307.16425): segmenti / beat / downbeat correlati.
- [AutoMashup 2025](https://arxiv.org/abs/2508.06516): direzionalità e limiti degli embedding generici.
- [Schulze-Forster et al. 2021](https://schufo.github.io/files/2021_TASLP3091817.pdf): precisezza dell'allineamento di fonemi e lyrics.
