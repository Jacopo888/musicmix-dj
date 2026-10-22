# Ricerca V8 — indice e revisione di integrazione

Ricerca eseguita da agenti OpenCode `opencode/deepseek-v4.1-flash-free`, variant `high`, il 19 ottobre 2026. Le profondità di lettura sono indicate per fonte: non tutti i testi sono stati letti integralmente. I registri in `runs/swarm_v8/swarm.json` e `swarm_followups.json` conservano modello, stato, costo dichiarato e cancellazione delle sessioni dopo archiviazione.

- [Psicologia, aspettative e struttura](PSYCHOLOGY_V8.md)
- [MIR, modelli recenti e fattibilità](MIR_SOTA_V8.md)
- [Auto-feedback e valutazione dell’audio](AUTO_FEEDBACK_RESEARCH_V8.md)
- [Validazione del critico e controfattuali](CRITIC_VALIDATION_V8.md)
- [Allineamento del cantato e parole tagliate](SINGING_ALIGNMENT_V8.md)

Gli omonimi `*_sources.json` riportano link, lettura effettiva e limiti. Il numero di voci include repository, modelli e paper ripetuti tra temi; non è un conteggio di nuove pubblicazioni uniche.

## Decisioni verificate dall’orchestratore

1. Il pool usa quattro brani reali e tutte le dodici direzioni. I vecchi guard vocali restano conservativi ma non garantiscono ancora confini corretti sul cantato.
2. I descrittori di novelty e continuazione restano euristici. Non diventano una prova di risoluzione solo perché un’altra pubblicazione usa la parola tensione.
3. Audiobox Aesthetics è stato effettivamente eseguito localmente dall’orchestratore DOPO la ricerca degli agenti: pesi ufficiali safetensors, hash verificato, finestra di dieci secondi centrata sul raccordo. I report dei singoli agenti che dichiarano di non aver eseguito modelli descrivono correttamente il loro lavoro, non l’integrazione successiva.
4. I quattro assi Audiobox non vengono fusi né usati come premio di continuità. Un singolo controllo corrotto non calibra soglie o significatività. COCOLA riguarda sovrapposizioni di stem, non è automaticamente trasferibile a una successione A→B; non è stato scaricato o eseguito.
5. SOFA/STARS e i provider neurali di struttura sono candidati per ulteriori prove, non dipendenze installate o sistemi già validati qui. Le indicazioni di licenza e hardware restano specifiche delle fonti consultate.
6. Sono presenti giudizi umani storici V3/V4. Essi non costituiscono validazione della V8 né un benchmark pubblico di stato dell’arte. Nessun miglioramento percettivo o primato SOTA viene dichiarato.

## Correzioni della revisione

La copia integrata MIR precisa che esistono i giudizi storici e limita le affermazioni SOTA ai benchmark riportati nei lavori esaminati. La copia integrata della validazione del critico corregge l’intestazione della tabella dei delta: i valori negativi erano `corrotto − policy`, non il contrario. Gli originali degli agenti restano nelle rispettive cartelle per tracciabilità.

Le suite complete eseguite nelle copie degli agenti mancavano della cartella `tools/`: quel limite d’isolamento non è una regressione del progetto. L’orchestratore esegue la suite nel progetto completo dopo l’integrazione.
