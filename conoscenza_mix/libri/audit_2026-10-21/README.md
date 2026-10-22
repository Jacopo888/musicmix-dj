# Audit completo dei libri e della pipeline — 21 ottobre 2026

Risultato principale: [revisione della pipeline e priorità](REVISIONE_PIPELINE.md). Le proposte sono distinte dalle funzioni già implementate e dai risultati verificati. Nessuna modifica alla produzione in questo ramo.

| Copia fornita | Unità coperte |
|---|---:|
| Webber | 301/301 pagine PDF |
| Ohriner | 290/290 |
| Katz | 350/350 |
| Margulis | 225/225 |
| Krims | 236/236 |
| Williams | 284/284 |
| Huron | 477/477 |
| Butler | 364/364 |
| Brewster/Broughton | 75/75 blocchi EPUB, 67 elementi spine |

25 blocchi affidati a Luna5.6 xhigh; massimo tre lettori contemporanei. [Queue e report](READING_QUEUE.json), [ledger per pagina](readers/), [verifica copertura](COVERAGE.json), [fonti/hash](SOURCES.json), [decisioni del coordinatore](ROOT_CHECKS.md). Ogni pagina è registrata con argomento e stato di lettura; la verifica conta attestazioni, non misura comprensione. Il coordinatore ha riletto le selezioni complete e controllato figure, tabelle, note e interpretazioni. Non ogni proposta dei lettori è stata promossa.

Produzione: test eseguiti su6f03b33; delta successivo bec3420 ricontrollato in [LATEST_DELTA.md](LATEST_DELTA.md). Suite: **242pass/8skip/6fail**; tre dipendenze mancanti, tre asserzioni sullo stesso render. [Risultato completo](CURRENT_TEST_RESULTS.json), [esecuzione isolata](RENDER_ISOLATED_TEST_RESULTS.json), [triage](RENDER_TRIAGE.md). Non è una suite tutta verde né una valutazione percettiva.

Tre casi riprodotti dalle funzioni reali: [input e output](ROOT_COUNTEREXAMPLES.json). Per ripeterli, dalla radice del checkout:

```powershell
python -X utf8 tools/book_pipeline_probes.py
python -X utf8 tools/verify_book_read_coverage.py
```

Il secondo comando usa il manifest privato nella cartella sorella `.book_audit_private_20261021`; sorgenti in `../libri`. Il corpus integrale e i render non sono inclusi nell'audit Git. I lettori di altri checkout devono ricreare il corpus da copie locali corrispondenti agli hash di SOURCES.json, oppure indicare `--manifest`.

La ripresa ha recuperato16 coppie ledger/report da dati letterali della sessione precedente, senza eseguire i comandi registrati; [manifest di recupero](RECOVERY_MANIFEST.json). I vecchi audit CODE_SELECTION/CODE_CRITIC_RENDER sono conservati come storici e marcati superati. I risultati correnti sono nella revisione e nei file dedicati.

[BACKLOG.json](BACKLOG.json) contiene nove interventi proposti, dipendenze e criteri di accettazione. Il primo è qualificare il backend locale di stretch; seguono contratti delle misure, candidati e integrazione. Nessun download di canzoni, inferenza di nuovi modelli o ascolto di nuovi mix è stato eseguito da questo audit.
