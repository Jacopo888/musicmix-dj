# Ricerca V9 — quando cambiare e auto-feedback

19 ottobre 2026. [Sintesi e ordine d'implementazione](../../docs/history/22_QUANDO_CAMBIARE_E_AUTOFEEDBACK_V9.md).

Otto filoni paralleli DeepSeek via server OpenCode condiviso: quattro completati dagli agenti, quattro con quota esaurita e sintesi completata separatamente dall'orchestratore. Gli originali e i messaggi di sessione sono conservati in `runs/swarm_v9_research/agents`. Ogni cartella integrata espone `PROVENANCE.json`, registro fonti, affermazioni e raccomandazioni.

| Filone | Dossier | Provenienza della sintesi |
|---|---|---|
| Manuali, fraseggio e pratica DJ | [Report](dj_practice_books/REPORT.md) | Agente completato, revisione root |
| Modelli di cue e timing | [Report](cue_point_models/REPORT.md) | Agente completato, revisione root |
| Psicologia di aspettativa e tensione | [Report](psychology_when/REPORT.md) | Root dopo quota agente |
| Struttura hip-hop, voce e frase | [Report](hiphop_structure/REPORT.md) | Agente completato, revisione root |
| Critici musicali appresi | [Report](automatic_critic_models/REPORT.md) | Root dopo quota agente; registro parziale recuperato |
| Auto-supervisione e ciclo di feedback | [Report](self_supervised_feedback/REPORT.md) | Root dopo quota agente |
| Dataset e benchmark | [Report](benchmark_datasets/REPORT.md) | Root dopo quota agente |
| Revisione critica dell'architettura | [Report](architecture_redteam/REPORT.md) | Agente completato, revisione root |

Leggere le [correzioni](REVIEW_CORRECTIONS.md) prima di applicare raccomandazioni dei singoli agenti. In particolare: M-DJCUE è del 2019; la correlazione 0,83 di Mosaikbox riguarda giudizi umani, non un critico automatico; Raveform aggiorna nel 2026 il quadro dei dataset; la griglia V8 effettiva non era troncata a 4096 punti. I dati sconosciuti restano null e il seguito originale di A è un riferimento offline dichiarato.

L'[inventario](../../runs/swarm_v9_research/bundle_inventory.json) conta 112 voci di fonte e 96 URL canonici distinti, **non** altrettante pubblicazioni indipendenti o letture integrali. Comprende codice e guide; la lettura di sezioni di tesi non equivale alla lettura dell'intera tesi. I file con estensione PDF ma intestazione non PDF sono registrati come recuperi falliti. La raccolta di un documento non prova che tutte le sue pagine siano state lette.

- [Diagnosi dei giudizi V8](FEEDBACK_DIAGNOSIS.md)
- [Note di verifica diretta](ROOT_REVIEW_NOTES.md)
- [Correzioni e limiti delle conclusioni](REVIEW_CORRECTIONS.md)
- [Specifica dei prossimi esperimenti, non implementata](V9_EXPERIMENT_SPEC.json)

Sono stati consultati paper aperti, repository ufficiali, copie istituzionali e capitoli/estratti autorizzati. Non sono stati scaricati nuovi pesi o corpus audio in V9. La ricerca non dimostra ancora un miglioramento percettivo o SOTA del progetto.
