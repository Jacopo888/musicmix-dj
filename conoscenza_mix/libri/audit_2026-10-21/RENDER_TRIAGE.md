# Render triage — snapshot `6f03b33f1a94b997003fc1ccfb5e092eaa74e083`

Audit circoscritto ai tre fail di `tests/test_dj_render.py` riportati in
`CURRENT_TEST_RESULTS.json` e riprodotti isolatamente in
`RENDER_ISOLATED_TEST_RESULTS.json`. Non sono state modificate funzioni di
produzione, test o fixture; il probe ha usato solo segnali sintetici.

## Ambiente e percorso eseguito

- Windows: `rubberband.exe` non è nel `PATH`; `ffmpeg.exe` è presente e
  contiene il filtro `rubberband` (`librubberband`), build osservata
  `ffmpeg 9.0-full_build-www.gyan.dev 9.0`.
- Con `PYTHONDONTWRITEBYTECODE=1` e
  `PYTHONPATH=../.book_audit_private_20261021/test_runtime;src`,
  `stretch_backend_info("auto")` restituisce `backend="ffmpeg"`,
  motore `af_rubberband (R2 realtime)`, `keyframes=False`.
- In `src/musicmix/dj_render.py:940-950`, l'assenza di keyframe fa ricadere
  anche la configurazione predefinita `stretch_mode="grid"` su `fit`, cioè un
  rapporto costante. Nel fixture 120→126 BPM il rapporto calcolato è coerente:
  `tempo_ratio_b=0.95238095`, `time_ratio_b=1.05`, `b_bpm_effective=120`.

## Probe minimo del backend

Probe eseguito tramite `_run_stretch` su una sinusoide stereo sintetica a 82 Hz,
48 kHz, 4 s, `time_ratio=1.05`, senza keyframe. Output osservato: 4,2 s, picco
FFT a **78,095 Hz**, non 82 Hz (tolleranza del test ±0,5 Hz).

Questo localizza un comportamento riproducibile nel percorso FFmpeg/
`af_rubberband` attivo su questa macchina: la proprietà dichiarata nel modulo
come “pitch invariato” non è osservata dal fallback. Il probe non distingue se
la causa più profonda sia la build/librubberband o l'invocazione del filtro;
non è stato possibile confrontare il CLI Rubber Band R3 perché assente.

Il comando è costruito in `src/musicmix/dj_render.py:564-590`: il ramo FFmpeg
calcola `tempo = 1/time_ratio` (`:581`) e invoca
`rubberband=tempo=...:pitchq=quality:transients=crisp:channels=together`
(`:582-585`). Il render usa quel risultato direttamente in `:974-976` e lo
conserva come `decks["b_stretched_raw"]` in `:1167-1169`.

## Lettura dei tre fail

### `test_stretch_preserves_pitch`

È un fail confermato del backend attivo, non un semplice artefatto del test:
il test legge lo stesso `b_stretched_raw` (`tests/test_dj_render.py:130-134`)
e trova 78 Hz contro l'atteso 82 Hz (`:135-139`); il probe diretto riproduce
la stessa deviazione senza il renderer di transizione.

### `test_never_two_full_basses`

Non emerge un secondo difetto indipendente nel bass swap. Il renderer imposta
`low_b = clip(...)` e `low_a = 1-low_b` (`src/musicmix/dj_render.py:988-990`),
poi applica quei guadagni ai due deck (`:1067-1071`). Il controllo separato
dei gain complementari passa nella suite.

Il test fallito normalizza però la reference di B a 82 Hz
(`tests/test_dj_render.py:161-166`). Con l'output del fallback la fondamentale
di B è circa 78 Hz: nella finestra di reference a 82 Hz la misura è quasi zero
(circa `0.00184` nel render osservato), quindi i rapporti in dB diventano
instabili e `both_full` conta 68 finestre. Ricalcolando la stessa metrica a
78 Hz, la reference è circa `0.2277` e `both_full=0`. Questo fail è quindi una
conseguenza misurabile del fail di pitch; non prova che il renderer sommi due
bassi pieni.

### `test_downbeats_of_stretched_b_land_on_a_grid`

Il render conserva una misura interna coerente: `downbeat_alignment` riporta
`max_abs_error_ms=2.858`, `interdeck_max_abs_lag_ms=3.493` e 9 downbeat
confidenti. L'assert che fallisce usa invece il detector indipendente a banda
3 kHz (`tests/test_dj_render.py:105-116`). Nel run osservato alcuni click non
superano la soglia relativa: per gli expected 18, 20 e 24 s il nearest-neighbor
riusa eventi a 16/22 s, generando errori di circa ±2 s. La misura non mostra
uno spostamento uniforme della griglia; mostra eventi non rilevati o alterati
dal percorso di stretch, più una regola di matching che può associare lo stesso
onset a più expected.

La classificazione prudente è quindi: stesso percorso FFmpeg già dimostrato
problematico sul pitch; il fail del detector è compatibile con un effetto sui
transienti del fallback, ma da questi dati non si può separare completamente
quel comportamento dalla fragilità del detector nearest-neighbor. Non va
registrato come un terzo bug indipendente di allineamento.

## Conclusione operativa

La causa confermata è una sola: nel contesto Windows della run, il fallback
FFmpeg selezionato automaticamente non conserva la fondamentale durante il
time-stretch. Il fail del bass test dipende dalla reference a 82 Hz dopo tale
deviazione; il fail dei downbeat resta un'interazione backend/detector, mentre
la misura interna del renderer è entro 10 ms. La presenza di FFmpeg fa passare
`HAS_STRETCH` (`tests/test_dj_render.py:22-24`) anche quando il backend non ha
la stessa semantica/qualità del CLI Rubber Band, quindi i risultati vanno
etichettati per backend.

## Test concreto proposto

Aggiungere un test isolato, eseguito solo quando `stretch_backend_info("auto")`
seleziona FFmpeg, che generi la stessa sinusoide a 82 Hz e chiami
`_run_stretch(x, 48000, 1.05, None, backend)`. Misurare il picco FFT dopo aver
scartato il bordo e richiedere `82.0 ± 0.5 Hz`, oltre a verificare la durata
attesa `4.2 ± 1` campione. Il test deve fallire con il backend osservato e
separare immediatamente un problema di backend da fixture, bass swap o
allineamento.

Dopo una correzione o una sostituzione del backend, rieseguire i tre test
isolati. Per il test dei downbeat, rendere il matching uno-a-uno (un onset non
può essere riusato) e riportare separatamente “onset mancante” e “onset fuori
finestra”; mantenere comunque la soglia di ±10 ms. Per il bass test mantenere
82 Hz come contratto di pitch preservation, invece di adattarlo ai 78 Hz del
fallback.
