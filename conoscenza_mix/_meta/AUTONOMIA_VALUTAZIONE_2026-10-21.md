# Autonomia della valutazione — 21/10/2026

Requisito dell'utente (da una proposta di GPT Astra condivisa dall'utente): nuovi brani e nuovi set non devono
richiedere il suo voto per funzionare; ascoltare solo lui non scala. Niente DJ esterni (strada 2 esclusa
dall'utente). La verifica umana resta periodica e su un campione, non a ogni set.

## Strade proposte e decisione
| # | Strada | Stato |
|---|---|---|
| 1 | Imparare dai set DJ che piacciono all'utente (allineamento mix–brano, distribuzioni di cue, sovrapposizioni, tecniche) | pipeline in costruzione, validata sui nostri set; servono 3–5 mix di riferimento indicati dall'utente |
| 2 | DJ pagato che insegna al sistema | esclusa dall'utente |
| 3 | Palestra automatica degli errori come gate obbligatorio | in costruzione (`defect_gym`) |
| 4 | Brani stessi come esempi di continuità (transizioni interne; COCOLA per la coerenza tra stem) | da fare dopo i mashup (calcolo pesante) |
| 5 | Squadra di giudici specializzati con incertezza (ritmo, voce, stem, evento previsto vs udibile; SongEval/MuseCritic solo se utili sulle giunte rap) | parziale (QA .5, critico v4 technical); da completare |
| 6 | Schede brano preparate una volta sola | implementato: `track_cards/`, [nota](SCHEDE_BRANO_AFFIDABILITA_2026-10-21.md) |
| 7 | Gusto dall'ascolto normale (salti, riascolti, con consenso) | da fare (segnali ambigui: affinano un profilo, non lo creano) |
| 8 | Domande di stile rare, con limite esplicito (anche zero) | da fare |
| 9 | Scegliere i problemi che sa risolvere (affidabilità), con copertura e varietà misurate a parte | implementato come opzione spenta (`reliability`, `--reliability prefer`), [nota](SCHEDE_BRANO_AFFIDABILITA_2026-10-21.md) |

Misura di successo aggiunta: intervento umano richiesto da 100 brani nuovi e 10 set nuovi (`tools/human_effort_report.py`).
La pagina di approvazione del set e il torneo dei cue restano strumenti di verifica periodica a campione, non un passaggio obbligatorio.
