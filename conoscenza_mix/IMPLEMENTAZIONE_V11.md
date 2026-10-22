# Evidenza → implementazione V11

Aggiornamento root: 2026-10-20. Questa tabella documenta il trasferimento
ingegneristico; non attribuisce ai paper la validazione delle nostre soglie.

| Fonte / sezione | Cosa sostiene | Componente effettiva | Cosa rimane ignoto |
|---|---|---|---|
| Farbood; Lehne/Koelsch; Hansen, 01 | Contesto temporale, aspettativa, incertezza e memoria | Profili A-before/A-after/B-before/B-after; decisioni stay/prepare/execute/defer | I trend acustici non identificano quale aspettativa semantica sia attiva |
| Turrell, 01 | Build-up/drop e tensione nel compito EDM studiato | Separazione tra continuità del movimento e arrivo | Nessuna certificazione di drop/risoluzione rap da loudness o label chorus |
| MCFlow; Adams; Duinker, 02 | Rap: sintassi, ritmo, motivo, respiro e metrica interagiscono | Parola attiva distinta da segmento ASR; frase e word guard in campi diversi | Chiusura sintattica/prosodica ignota nei sei mix; non è risolta da pause ASR |
| CUE-DETR; LinkSeg; lavori DJ, 03–04 | Proposte di cue e struttura funzionale | Output ufficiali già eseguiti su sei brani, riusati insieme a opportunità vocali | Supporto di opportunità diverso da qualità del raccordo tra due canzoni |
| TenseMusic, 06 | Sei feature, dinamica e memoria su estratti classici | Paper e sorgenti ufficiali archiviati; preflight documentato | Inference non eseguita: import fermo a librosa assente; altre dipendenze e trasferimento rap da verificare |
| SongEval, 06 | Dimensioni estetiche su canzoni generate | Run reale V10 conservata come diagnostica | Non è un reward validato per transizioni; escluso dalla selezione |
| M-DJCUE, 07 | Cue IN/OUT con annotatori e protocollo EDM | Albero ufficiale completo archiviato e 134 JAMS parseati: 1.849 eventi | Non contiene label di risoluzione A→B rap; nessun nuovo score percettivo validato |
| Protocollo ingegneristico, 00/05/06 | Distinguere misura tecnica e qualità musicale | Render da 48 s, clocks esatti, attenuazione limitata, audit PCM indipendente, controlli a cue identico e native-A | Un WAV corretto può restare musicalmente insoddisfacente |

## Risultato osservato

`runs/psychology_v11_release/manifest.json` congela 33.204 proposte distinte:
31.537 escluse dallo storico, 404 dal contesto richiesto, 1.263 valutate.
La policy rinvia 1.127 candidati e ammette 136 per evidenza di continuità e
clearance temporale delle parole. Ne seleziona sei diversificando zone e
direzioni; ciascuno ha due render allo stesso cue e una continuazione A.

**Tutti e sei mantengono la chiusura della frase come ignota.** “Execute”
significa che i vincoli della policy permettono il render diagnostico; non
significa risoluzione percepita, frase completa o qualità 10/10.

Il replay delle sei decisioni è verificato in `explanation_manifest.json`.
I profili mostrano conflitti tra alcuni domini anche quando due concordano:
la label di continuità del proxy non equivale a continuità psicologica globale.
Questa è una limitazione osservata, conservata senza aggiustare le soglie sui
sei casi selezionati.

I giudizi V8/V10 non sono usati per fit di soglie, ranking o reward. L'esempio
V10 audio_003 rimane una singola osservazione positiva, non una regola generale.

## Priorità sostanziale successiva

1. Provider di completamento prosodico/sintattico e di evento d'arrivo B con
   incertezza separata dalla clearance delle parole.
2. Riutilizzo fedele di modelli temporali esistenti e verifica del dominio;
   TenseMusic non va sostituito con un proxy RMS chiamandolo lo stesso modello.
3. Valutazione esterna su registrazioni/artist/versioni disgiunti. Non
   trasformare punti non annotati in negativi né continuazione A in positivo.
4. Solo dopo questa verifica, usare un critic musicale per auto-feedback.
   L'audit tecnico già implementato non soddisfa da solo quel requisito.
