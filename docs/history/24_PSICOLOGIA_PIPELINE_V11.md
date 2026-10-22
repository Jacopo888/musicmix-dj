# V11 — decisione del cambio, evidenze e controllo del render

**Verifica finale:** 267 test e 3 subtest superati. La run release contiene
sei punti di cambio, dodici handoff e sei riferimenti naturali. L'audit
`qa_independent_v3.json` passa senza difetti: sorgenti/PCM, guadagni, clocks,
entrambi i lati vocali e novità dei cue. Sono conservati 27 riscontri descrittivi
di contesto già ascoltato; nessuna ripetizione dell'intera transizione.
HTTP HEAD/Range/416 verificato sui 18 file. L'interazione nel browser non è
stata automatizzata: il relativo strumento locale non era disponibile.

L'audit iniziale `qa_independent.json` è conservato: aveva confuso durata
sorgente e output, riquantizzato il PCM naturale e trattato la sovrapposizione
di contesti come difetto. Le correzioni riguardano l'auditor; i WAV sono rimasti
identici. Lo stato completo è in `runs/psychology_v11_release/ROOT_STATUS.json`.

## Obiettivo e confini della misura

La decisione deve considerare se continuare A, preparare un cambio, eseguirlo
o rinviarlo. La funzione desiderata può essere continuità del movimento,
arrivo o contrasto deliberato. Una pausa vocale, un beat o la parola “chorus”
non dimostrano da soli una risoluzione. Il sistema attuale può raccogliere
evidenze di continuità; manca ancora una misura validata di completamento
dell'aspettativa tra due brani rap/trap.

## Implementazione effettiva

- `phrase_evidence.py` distingue parola attiva, copertura locale ASR,
  occupazione del segmento del decoder, clearance acustica e chiusura musicale.
  Un segmento ASR può contenere pause: non equivale a una parola continua.
  La chiusura di frase rimane ignota senza un provider pertinente.
- `transition_intent.py` produce azione, intento, ragioni, dati mancanti e
  copertura. I punteggi non sono probabilità di gradimento. Il confronto con
  la continuazione naturale di A non premia l'identità del file.
- `pooled_trajectory.py` riusa i descrittori dei beat reali, ricalcolando
  l'intervallo di bordo al campione effettivo del cue. Non sposta la griglia.
- `expectation.py` conserva i trend multiscala come ipotesi ingegneristiche.
  Il log-RMS non normalizza più la direzione sulla propria origine assoluta:
  un guadagno audio costante non deve cambiare quella misura di movimento.
- `build_vocal_opportunities_v11.py` unisce opportunità CUE-DETR, LinkSeg,
  estremi dei segmenti ASR e pause osservate fra parole, con provenienza e ruoli.
  Il vecchio adattatore di ricerca ignora i ruoli aggiuntivi; i controlli vocali
  direzionali e la verifica successiva restano necessari.
- `intent_render.py` avvolge i renderer esistenti: contesto minimo 16 s per
  lato, clocks sorgente/output espliciti, guadagno limitato, protezione del
  quasi silenzio, headroom e condizioni restrittive per la sovrapposizione.
  Il driver V11 usa 24 s per lato e non amplifica il lato più quieto.
- `transition_validation.py` valuta il WAV realmente generato al suo join.
  Clipping e identità errata sono difetti; salto di livello e frase ignota
  restano osservazioni distinte. SongEval non attribuisce approvazione.
- `audit_psychology_v11.py` confronta indipendentemente PCM, sorgenti, clocks,
  guadagni e storico. Controlla anche la continuazione naturale di A.
- `serve_psychology_review.py` espone un player italiano con seek e HTTP Range.

## Esperimento controllato

Il driver `tools/run_psychology_pool_v11.py` congela catalogo, storico, hash
dei moduli e manifest delle ricerche. Esclude le zone già proposte in V10
prima della selezione. Richiede contesto completo anche per la continuazione
naturale di A e per l'analisi precedente di B. B-before è conservato come
contesto: il valutatore di aspettativa attuale usa A-before, A-after e B-after.

Per ogni candidato selezionato produce tre ascolti da 48 s:

1. stesso cue con attenuazione controllata del lato più forte;
2. stesso cue con volume originale, per isolare l'effetto del renderer;
3. continuazione originale di A, con identico contesto precedente.

La terza variante non è una risposta corretta per definizione. I giudizi
precedenti, incluso audio_003 V10, non sono usati per regolare soglie o reward.
Le prove sintetiche verificano proprietà costruttive, non il gusto musicale.

## Riferimenti mantenuti e prossimo limite

La biblioteca è in `conoscenza_mix/INDEX.md`, con otto riassunti tematici,
originali, testi estratti e metadati. `AGENTS.md` ne richiede l'aggiornamento.
Le fonti su memoria/aspettativa, frase rap e transizioni DJ motivano componenti
separate; non validano le soglie del nostro scorer su questo pool.

Il passo successivo sostanziale è una misura temporale di aspettativa/evento
con validazione esterna pertinente. TenseMusic è un candidato da riutilizzare
fedelmente, con dominio e dipendenze da verificare. CUE-DETR e LinkSeg sono
già eseguiti, ma non sono modelli di risoluzione tra canzoni. SongEval resta
diagnostico dopo il controllo V10. Non è dimostrata qualità 10/10 né SOTA.

## Riproduzione

Usare una cartella output nuova per preservare ogni esperimento:

```powershell
python tools/run_psychology_pool_v11.py --history runs/audition_history_v10.json --extra-search runs/psychology_v11_vocal_search_fresh3 --output runs/psychology_v11_replay
python tools/audit_psychology_v11.py runs/psychology_v11_replay/manifest.json
python tools/serve_psychology_review.py --manifest runs/psychology_v11_release/review_manifest.json --port 8772 --serve
$env:PYTHONPATH='src'
python -m pytest -q
```

Per nuove proposte lo storico aggiornato è `runs/audition_history_v11.json`
(47 regioni, sei proposte V11 aggiunte senza inventare giudizi); il driver lo
preferisce automaticamente. Lo storico V10 nel comando serve a riprodurre
l'esperimento congelato. La galleria già generata resta nella cartella release.

I primi due tentativi sono conservati: `runs/psychology_v11` (189 valutati,
zero eseguibili) e `runs/psychology_v11_fresh` (1.263 valutati, zero eseguibili).
Hanno rivelato una conflazione tra occupazione del segmento ASR e parola
attiva, corretta esplicitando le due dimensioni. Non sono ascolti migliorati
né esperimenti di validazione percettiva.

Il primo tentativo `runs/psychology_v11_final` è incompleto e non pubblicato:
un percorso relativo nell'esportazione del player ne ha interrotto la scrittura.
La correzione è eseguita nella run separata `runs/psychology_v11_release`.
Quest'ultima contiene sei punti di cambio, dodici render e sei riferimenti A.
La copia esatta dei sette moduli eseguiti è conservata in `source/` e
verificata contro gli hash del manifest.

L'esclusione dello storico riguarda i cue entro otto secondi, in qualunque
ruolo. Il contesto da 48 s può includere porzioni già ascoltate: questa
sovrapposizione è una descrizione della copertura, non una certificazione di
novità di ogni campione e non un difetto del PCM.

La ricostruzione in `explanation_manifest.json` riproduce le sei decisioni
e salva quattro profili per candidato. Mostra anche un limite concreto:
“continuation” può derivare dall'accordo di due domini mentre altri divergono;
per esempio energia, timbro e movimento cromatico possono suggerire direzioni
diverse. Non è una misura validata della suspense percepita. La presenza di
sei WAV tecnicamente corretti non dimostra sei mix musicalmente riusciti.
