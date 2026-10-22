# Revisione esterna dell'andamento (GPT Astra, condivisa dall'utente) — 21/10/2026

Revisione sul commit `e0a0de25` di `claude/cloud-setup`, senza modifiche al codice. Riassunto e decisioni prese.

## Punti principali della revisione
- Obiettivo giudicato fattibile (set rap sulla libreria dell'utente, che impari le sue preferenze); nessun ostacolo
  fondamentale, ma distanza tra capacità dei moduli e qualità costante delle decisioni musicali.
- Le funzionalità sono cresciute più in fretta delle prove che il risultato piaccia: galleria cieca v3, 28 schede
  (24 con voto), 19 voti principali con mediana 5 e un solo voto ≥ 7; libri meglio 3 volte, default 2, 4 pari.
- Il problema ricorrente è il cue (note "B nel secondo sbagliato", "B non risolve bene A"): confrontare effetti
  sullo stesso cue non può correggere un cue sbagliato.
- Ostacoli: errori di integrazione (stem, coordinate dopo lo stretch, controlli non applicati agli effetti:
  risolvibili), percezione audio difficile (griglia rap, attacchi vocali, 808: migliorabile), scelta musicale e gusto
  (servono esempi e preferenze pertinenti), garanzie assolute (non promettibili).
- Problemi segnalati nel codice: riferimento fisso ad `a_vocals.wav` in più loader; ripieghi che portano all'ascolto
  ciò che le analisi sconsigliano (`best_fail` nel runner, `entry_guard_relaxed` in `dj_set.entry_guard`).
- Il risultato MCFlow viene da trascrizioni simboliche: il trasferimento su trap registrata con ASR va verificato.
- Proposta di ciclo: (1) passaggi di riferimento verificati su brani noti; (2) renderer sui punti verificati;
  (3) confronti di cue a effetto fisso, poi effetti sui cue migliori; (4) selezione con verifica su esempi nuovi;
  (5) valutazione di set interi. Traguardo: un piccolo set approvato dall'inizio alla fine con ogni correzione registrata.

## Decisioni (orchestratore)
- Ripieghi forzati tolti dal default (agente ingresso di B): pair senza candidati ammissibili → ripianificazione
  (prolungare A, altro ingresso, togliere l'arco, cambiare B); relaxation solo come opzione esplicita.
- Risolutore unico dello stem vocale con verifica della provenienza (agente lingua/ASR).
- Verifica dell'ingresso di B con tecnica fissa e cue variabili.
- Nuovi strumenti: correzioni strutturate nel voto (±battute per uscita di A e ingresso di B, pickup, sezione,
  brano B, tecnica), pagina di approvazione del set intero, torneo dei cue, applicazione delle correzioni,
  passaggi di riferimento (`giudizi_cloud/reference_transitions/`, separati dai dati di taratura), metrica di accordo.
- I mashup restano una linea esplorativa chiesta dall'utente, valutata con lo stesso protocollo.
