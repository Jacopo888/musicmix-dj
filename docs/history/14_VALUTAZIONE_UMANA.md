# Valutazione umana delle transizioni

Questo documento riporta l'import dei due export A/B forniti dall'utente. La
relazione tra file di giudizi e run è stata confermata esplicitamente:

- `giudizi_musicmix.json` → `structured_forward_v3`;
- `giudizi_musicmix 2.json` → `structured_reverse_v3`.

L'importatore non assume questa associazione dai soli `pair_id`: i `pair_id`
(`rank_1`–`rank_3`) sono riutilizzati nei due run e non contengono né run né
direzione. Ogni import usa quindi due argomenti distinti, `--judgements` e
`--run`, e fa il join con il `listening_ab/key.json` selezionato tramite
`trial + pair_id`, richiedendo esattamente le label `A` e `B`.

## Evidenza importata

| Export | Run ID | SHA-256 export | SHA-256 key | Osservazioni |
|---|---|---|---|---:|
| `giudizi_musicmix.json` | `4f513476-e95c-496d-ab10-a0e5c287909f` | `e0596fa862066af10c6ec770819274328235f19d3efa1bf1ce1655ac096f6a15` | `fa2c1fc6e4803c52d410967e72cf995590e47391e8779bf84389d2af02da803c` | 3 |
| `giudizi_musicmix 2.json` | `977375e5-f79c-4b21-916b-3d6b046adf0a` | `56a3f026453b20f43a84285015bae54580e8bc1a5a8cebfd7de8b668e1583dd7` | `e0de9722d57235f91132f59ff24444b89fa540a4324a8de9f27e4c4c51101e0b` | 3 |

Il join confermato identifica `structured_directional_v2` come candidato A e
`onset_snapshot_v1` come candidato B in entrambi i run. Le sorgenti complete,
con i percorsi dei file e gli hash, sono conservate nei JSON normalizzati:

- `<workspace>/../giudizi json/feedback_forward_v3.json`;
- `<workspace>/../giudizi json/feedback_reverse_v3.json`.

## Osservazioni per trial

La colonna “taglio parola” è il campo esplicito compilato dall'ascoltatore;
`false` significa che non è stato segnalato un taglio in quel candidato.

| Run | Trial | Pair | Preferenza | A: taglio parola | B: taglio parola |
|---|---:|---|---|---:|---:|
| forward | 1 | `rank_1` | A | false | true |
| forward | 2 | `rank_2` | neither | false | false |
| forward | 3 | `rank_3` | A | false | false |
| reverse | 1 | `rank_1` | A | false | false |
| reverse | 2 | `rank_2` | A | false | true |
| reverse | 3 | `rank_3` | A | false | true |

Nel commento del forward trial 3 l'ascoltatore ha annotato: “una delle
migliori in assoluto la A, forse la prima sufficiente”. Il commento è
conservato nel JSON normalizzato e non viene trasformato in un punteggio.
È un'osservazione qualitativa su `forward_v3/rank_3/A`, non un'etichetta da
usare per addestramento automatico.

I JSON normalizzati registrano la stessa regola in `usage_policy`: queste
osservazioni sono descrittive e `training_eligible` è `false`.

## Portata e limiti

Queste sono sei osservazioni descrittive su due direzioni e tre coppie. Il
risultato può documentare cosa è stato preferito in ciascun ascolto e quali
candidati hanno ricevuto una segnalazione di parola tagliata; non sostiene un
vincitore generale, una superiorità del sistema in popolazione, una
significatività o un intervallo di confidenza. Tutti i campi `confidence` sono
`null` intenzionalmente: una singola osservazione di ascolto non produce una
confidenza statistica.

Nella confezione V3 le nuove proposte erano casualmente tutte nella posizione A. La versione successiva bilancia le posizioni A/B entro ciascun test e conserva identificativi run/test nell'export. Il primo pilot non separa un eventuale effetto della posizione A dalla preferenza per il sistema.

Prima della conferma utente il report deve restare neutro rispetto ai sistemi:
un export A/B da solo non consente di dedurre quale run o direzione abbia
generato le label. In questa versione la conferma è registrata nel campo
`mapping.confirmed` dei due output (`status: confirmed_by_user`).

## API riproducibile

Validazione e import sono in
[`src/musicmix/evaluation.py`](../src/musicmix/evaluation.py). Il comando
richiede sempre il file dei giudizi e il run scelto:

```powershell
$env:PYTHONPATH = "src"
python tools/import_listening_judgements.py `
  --judgements "<workspace>/../giudizi json\giudizi_musicmix.json" `
  --run runs/structured_forward_v3 `
  --output "<workspace>/../giudizi json\feedback_forward_v3.json" `
  --mapping-confirmed
```

Sono accettati sia `musicmix.listening.1` sia `musicmix.listening.2`. Per la
versione 2 vengono verificati `test_id` e `run_id` contro gli stessi campi nel
`key.json` selezionato (e contro il manifest quando presente); un mismatch
interrompe l'import. La versione 1 resta importabile solo fornendo
esplicitamente `--run`, perché i suoi `pair_id` non identificano da soli run o
direzione.

Per il secondo export si sostituiscono i due percorsi con `giudizi_musicmix 2.json`,
`structured_reverse_v3` e `feedback_reverse_v3.json`. Senza
`--mapping-confirmed`, l'import salva la preferenza come label ma lascia
`preferred_candidate` a `null`, così un'associazione non confermata non viene
presentata come risultato di sistema. Duplicati esatti vengono collassati;
duplicati confliggenti, schema errato, risposta fuori da `A/B/tie/neither`,
trial mancanti e key ambigue interrompono l'import.

La suite dedicata è `tests/test_evaluation.py` e copre questi rifiuti, la
deduplicazione e l'assenza intenzionale di confidenza statistica.

## Nuovi ascolti V4

Sono stati importati anche i due export V4 con schema
`musicmix.listening.2`, verificando `test_id` e `run_id` contro le rispettive
key. L'associazione è quindi:

- `giudizi_Reminder_verso_Too_Many_Nights_V4_0eff5afcebe403d2.json` →
  `structured_forward_v4`, run ID
  `326d4386-31f0-4551-852c-b89c8f266976`;
- `giudizi_Too_Many_Nights_verso_Reminder_V4_30391fd8f08c3189.json` →
  `structured_reverse_v4`, run ID
  `4493210e-a823-41ec-9a4e-37fbe4e3432c`.

Gli export hanno rispettivamente SHA-256
`66a45de7a0f33328d02a1b4b2bf95ab123efe7cf92bcacc28ea0e7063ec6bff5` e
`6bc0d8becdec368c961a3bf4d3e2f6c1725002020e6ce09842459788c489e91a`.
Le key usate nel join hanno SHA-256
`dde9bf96cf4674ef4f31db9b515358dbf5f0301e87b270a37070892a03f055cf` e
`11e617d0b1894d39648e96712636b475b3ce08472699a39564a79f00df280977`.
Gli output normalizzati sono
`<workspace>/../giudizi json/feedback_forward_v4.json` e
`<workspace>/../giudizi json/feedback_reverse_v4.json`.

| Run | Trial | Pair | A dal key | B dal key | Preferenza | Taglio A/B |
|---|---:|---|---|---|---|---|
| forward | 1 | `rank_1` | `structured_forward_v3` | `directional-v2` | A | false / true |
| forward | 2 | `rank_2` | `directional-v2` | `structured_forward_v3` | B | false / false |
| forward | 3 | `rank_3` | `structured_forward_v3` | `directional-v2` | A | false / false |
| forward | 4 | `random_1` | `uniform_valid_cue_pair` | `directional-v2` | neither | false / false |
| forward | 5 | `random_2` | `directional-v2` | `uniform_valid_cue_pair` | B | false / false |
| forward | 6 | `random_3` | `directional-v2` | `uniform_valid_cue_pair` | A | false / false |
| reverse | 1 | `rank_1` | `structured_reverse_v3` | `directional-v2` | A | false / false |
| reverse | 2 | `rank_2` | `directional-v2` | `structured_reverse_v3` | B | false / false |
| reverse | 3 | `rank_3` | `structured_reverse_v3` | `directional-v2` | B | false / false |
| reverse | 4 | `random_1` | `uniform_valid_cue_pair` | `directional-v2` | A | false / false |
| reverse | 5 | `random_2` | `directional-v2` | `uniform_valid_cue_pair` | neither | false / false |
| reverse | 6 | `random_3` | `directional-v2` | `uniform_valid_cue_pair` | A | false / false |

### Lettura qualitativa dei commenti

Le annotazioni restano osservazioni dell'ascoltatore, non etichette
automatiche e non prove sull'identità dei brani. Il termine “rap” nei commenti
descrive una qualità percepita del passaggio; l'identità di A/B e dei sistemi è
presa soltanto dalla key con timestamp e sorgente, senza inferirla dal testo.

- **Forward, `rank_1`:** l'ascoltatore giudica B pessima per movimento/momentum:
  a un crescendo seguirebbe un ingresso piatto percepito come rap piatto; B ha
  anche il solo taglio di parola segnalato in questa coppia.
- **Forward, `rank_2` e `rank_3`:** il giudizio tratta i due casi come
  ripetizioni del problema precedente; nel trial 2 sono “uguali”, nel trial 3
  la preferenza resta A.
- **Forward, `random_1`:** B è giudicata pessima; A conserva un errore di
  movimento, con sensazione di due segmenti incollati e di mood discordanti.
  Non è una valutazione positiva del random: la risposta è `neither`.
- **Forward, `random_2`:** A è associata allo stesso taglio; B è descritta
  come vicina alla sufficienza.
- **Forward, `random_3`:** A è quasi sufficiente; viene suggerita una breve
  sovrapposizione fra le canzoni prima del cambio come correzione futura.
- **Reverse, `rank_1`:** A è apprezzata perché mantiene lo stesso ritmo; B è
  giudicata negativa per il passaggio percepito da rap a instrumental.
- **Reverse, `rank_2` e `rank_3`:** il trial 3 richiama A come “la stessa di
  prima” e considera entrambe vicine alla sufficienza; la preferenza salvata è
  B. Il trial 2 non contiene commento.
- **Reverse, `random_1`:** A è indicata come la stessa del confronto 3; B
  riceve di nuovo la critica rap→instrumental. I trial `random_2` e
  `random_3` non hanno commento; rispettivamente `neither` e A.

Queste ripetizioni descrivono la familiarità dell'ascolto con alcuni estratti,
non dodici osservazioni indipendenti. Il campo `usage_policy` dei JSON resta
`descriptive_only: true` e `training_eligible: false`.

### Ripetizioni byte-identiche degli estratti

La diversità dei cue limita il riuso delle ancore nelle proposte top-K, ma non
impedisce che uno stesso WAV venga scelto sia nel gruppo `rank_*` sia nel
controllo `random_*`. Il controllo SHA-256 delle sorgenti nelle key V4 trova:

| Direzione | SHA-256 (prefisso) | Posizioni duplicate nella key |
|---|---|---|
| forward | `395cbad61964dc3b` | `rank_1/A`, `rank_2/A`, `random_2/A` |
| forward | `b16141bc8ec59686` | `rank_1/B`, `random_1/B` |
| forward | `98cab6f72f549009` | `rank_3/B`, `random_3/A` |
| reverse | `af3586f3d9f54b85` | `rank_1/B`, `random_1/B` |
| reverse | `1f847fc3a92fe47a` | `rank_2/A`, `random_2/A` |
| reverse | `05db1563e637eba3` | `rank_3/B`, `random_3/A` |

Il key forward contiene 12 entry ma 8 WAV distinti; il key reverse contiene
12 entry e 9 WAV distinti. Le collisioni sono riferimenti allo stesso file
audio, non nuove prove indipendenti. Il confronto con il pilot V3 precedente
mostra inoltre file byte-identici riutilizzati fra run V3/V4 (per esempio
forward V3 `rank_1/A` con hash `395cbad6…` e reverse V3 `rank_2/A` con hash
`1fa43b13…`). Anche il test beat V4 riusa alcuni file del pool V4. Questo è
documentato per evitare di contare un WAV ripetuto come un ascolto nuovo,
lasciando invariati ranking, cue e renderer.
