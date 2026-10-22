# Valutazione e auto-feedback

## Audit integrale dei nove libri — 21 ottobre 2026

criticv3, ensemble, dose-risposta e astensione esistono; non proporre il primo aggregatore. Rimane da misurare trasferimento da difetti sintetici a passaggi naturali e preferenza, separando precisione tecnica e funzione musicale. Split per artista/brano/coppia, tuning distinto, positivi reali, negativi sottili, calibrazione e coverage. bec3420 aggiunge --blind e conteggio play: attivarlo, controllare ID e controbilanciare ordine. Un benchmark umano iniziale valida il giudice; non implica un umano per ogni mix.

Dettagli, fonti/pagine e criteri di accettazione: [revisione completa](../libri/audit_2026-10-21/REVISIONE_PIPELINE.md).

## Evidenza misurata

**TenseMusic (Barchet, Rimmele e Pelofi, 2024)** è un modello audio-input con sei feature, trend e memoria, valutato su 38 estratti classici. È adatto come riferimento per una traiettoria di tensione, non come etichetta di risoluzione di un raccordo: un calo della curva può essere una rilassatezza locale senza un evento corretto in B.

### Feasibility smoke eseguito (20 ottobre 2026)

Ho verificato il repository ufficiale [vivienbarchet/TenseMusic](https://github.com/vivienbarchet/TenseMusic), revisione dell'albero `0e704c7bdb06fec2144477f7e35004931dd038ef`. I tre moduli ufficiali (`predict_on_audio1.py`, `Only_features_revised.py`, `tension_functions_revised.py`) passano il controllo di sintassi. Ho generato un WAV mono di 10,000 s / 44.100 Hz / 441.000 campioni (sha256 `81f20e9db52b508740e0acb40511d22ae0121fffcbb8e75bf8913c9f7bd5669d`) e provato l'import del modulo ufficiale di estrazione: l'esecuzione si ferma prima di leggere l'audio con `ModuleNotFoundError: No module named 'librosa'`. Nell'ambiente corrente mancano anche TensorFlow, Basic Pitch, MoSQITo e altri pacchetti richiesti dall'implementazione. **Non è stata eseguita una predizione TenseMusic e non è integrata nel pool.** Il risultato è quindi una fattibilità condizionata alla creazione dell'ambiente `environment.yml`, senza installazione pesante in questo turno; il report riproducibile è [RESULT.json](../../runs/science_v11/agents/psychology_library/tensemusic_smoke/RESULT.json).

Il PDF PLOS archiviato ha 18 pagine secondo `pypdf` (non 13). Qui “full text archived/extracted” indica conservazione ed estrazione del PDF; la lettura usata nel riassunto è limitata alle sezioni ancorate: pp. 1-5 (modello e feature), pp. 12-16 (risultati, limiti e conclusioni). Il paper dichiara 38 estratti classici e una correlazione media circa `r=.60`; non dimostra trasferimento a rap, voce o raccordi cross-song.

**SongEval (2025)** raccoglie valutazioni esperte su dimensioni estetiche di canzoni generate. La coerenza/fraseggio dell'intera canzone può informare una rubrica, ma non è una label di cue, di word cut o di transizione tra due stems. Il run V10 locale è diagnostico: non giustifica l'uso di SongEval come filtro di selezione.

Le metriche dei lavori di struttura (HR.5F, F1, ACC), le metriche di allineamento (ms, beat hit-rate), i controlli renderer (dB, headroom) e i giudizi percettivi rispondono a domande diverse. Non sono intercambiabili e non vanno sommati in un “quality score” senza validazione esterna.

## Protocollo proposto

1. **Tecnica:** clock, range/headroom, clipping, silenzio al seam, continuity del gain e allineamento.
2. **Metrica:** beat/downbeat e tolleranza dichiarata; nessun downbeat viene chiamato closure.
3. **Voce:** copertura, onset/offset parola, respiro e frase; includere parole perse, non solo quelle recuperate.
4. **Aspettativa:** trend in A, evento in B, sorpresa/incertezza e tipo di arrivo; `UNKNOWN` quando il modello non lo sa.
5. **Giudizio:** preferenze/held-out separati, congelati prima delle soglie; V8/V10 restano diagnostica e non training.

## Cosa è una buona astensione

Un sistema che non seleziona un join quando mancano dati vocali o il renderer è tecnicamente invalido non ha “fallito il gusto”: ha rispettato la misura. L'astensione deve però essere auditabile (quale componente mancava, quale soglia originale, quale fonte) e non può diventare un modo per ottenere hit-rate perfetto ignorando i casi difficili.
