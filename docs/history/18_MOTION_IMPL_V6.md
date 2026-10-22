# V6 — aspettativa, continuazione e risoluzione

Il chiarimento dell'utente definisce il problema: A prepara suspense o un moto crescente, B entra piatto e non sviluppa né chiude quell'aspettativa. Il fatto che l'esempio sia rap non è una regola di genere. Il caso negativo e la precisazione sono conservati in `examples/feedback/v4_forward_rank1_motion_and_wordcut.json`.

## Ricerca dedicata

L'agente `motion_research_max`, modello gpt-5.6-luna con effort max, ha prodotto [la ricerca e l'audit delle fonti](../ricerca/MOTION_SUSPENSE_RICERCA.md). Distingue i testi letti integralmente da quelli consultati solo in parte e separa implementazioni disponibili da ipotesi del progetto.

Tre riferimenti operativi:

- [Farbood 2012](https://doi.org/10.1525/mp.2012.29.4.387): finestre attentive e memoria del contesto per seguire una traiettoria, oltre ai descrittori locali.
- [TenseMusic 2024](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0296385): modello audio con sei famiglie di feature e codice Python. La valutazione è principalmente su musica classica; non prova qualità del mashup o riconoscimento di risoluzione nei nostri brani.
- [Lehne e Koelsch 2015](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.00079/full): distinzione fra anticipazione/suspense, arousal e sorpresa. È un quadro teorico, non un classificatore pronto.

Il calo di energia o di un proxy di tensione può accompagnare una risoluzione oppure un'interruzione. La chiusura richiede evidenza aggiuntiva di frase, arrivo metrico/ritmico o stabilità armonica. Nel modulo sperimentale `motion.py` sono state ritirate le vecchie etichette `resolution_likely`/`escalation_likely` emesse dal solo delta: ora si descrive soltanto la variazione del proxy.

## Cambiamenti preparati per i prossimi esperimenti

- Misure audio multiscala in `trajectory.py`: energia, brillantezza, bassi, attacchi e movimento cromatico. Sono osservabili/proxy, non un modello validato di suspense.
- Opzione `--collect-motion`: ogni estratto può ricevere un'annotazione umana `continues`, `resolves`, `interrupts`, `unclear`, conservata nel JSON e nella normalizzazione. Nessuna label è assegnata dal volume o dal contenuto rap.
- I futuri pacchetti della pipeline escludono di default WAV ripetuti e A/B byte-identici. Le esclusioni sono salvate in `dedup.json`. Se non resta un confronto nuovo, il sistema lo dichiara. L'opzione esplicita `--allow-repeated-listening-audio` resta per disegni di confronto che richiedano un riferimento condiviso.

Le prove già ascoltate non sono state rimescolate. L'esclusione dei duplicati per hash non risolve da sola la ripetizione percettiva di sezioni simili o spostate di un beat: resta da valutare la diversità fra famiglie di segmenti.

`tools/analyze_trajectory.py` ha prodotto `runs/trajectory_audit_v6.json`: 18 selezioni V4 più il riferimento V3 giudicato sufficiente, marcato separatamente. Per ciascuna selezione sono misurati A-prima, B-dopo e la continuazione originale di A, con serie cronologiche per beat e finestre 8/16/32. Quando manca solo il contesto adiacente, restano le misure locali disponibili e il proxy cromatico contestuale è sconosciuto. Non vengono aggiunti campioni, beat o finestre fittizie. Altezza melodica, tensione percettiva e chiusura semantica restano sconosciute senza un provider verificato.

Verifica corrente: **117 test passati**. Nessun nuovo WAV e nessuna modifica ai pesi del ranking sono stati prodotti da questa ricerca/diagnostica.

## Decisione implementativa

Il passo seguente è confrontare tre evidenze separate: continuazione, chiusura e interruzione. Le nuove misure sono una base diagnostica; il ranking V4 resta il riferimento finché non esiste una verifica della nuova politica.

Il confronto con la continuazione originale di A (`A_before`, `A_natural_after`, `B_after`) è un'ipotesi osservativa del progetto. Non assume che B debba copiare il seguito originale e non prova che quella sia l'unica risoluzione. Il contesto mancante vicino agli estremi del brano deve restare esplicitamente indisponibile.

Prima di usare un nuovo criterio nel ranking: misurare sugli ascolti se distingue davvero risoluzione da semplice calo, tenere il caso negativo dell'utente fuori dall'adattamento delle soglie e confrontare la nuova scelta con il riferimento a renderer fisso. Il successivo confronto del renderer deve invece mantenere fissi i cue.
