# Integrazione `functional-event-v1.2` nel modulo principale

Data: 2026-10-20. Stato: `unvalidated_engineering_comparator` (invariato).

## Cosa è stato integrato

Codice: [`src/musicmix/functional_event.py`](../../../src/musicmix/functional_event.py)
(sha256 `77b232cd…`), test: [`tests/test_functional_event.py`](../../../tests/test_functional_event.py)
(16 test). La revisione viene dall'audit dell'agente DeepSeek
(`runs/result_swarm_v12/agents/functional_event/research/BRANCHES.md`, sha256 `9aae064a…`;
`outputs/branch_audit.json`, sha256 `94c7026b…`). È stata portata **integralmente**:
nella revisione non c'erano parti da scartare. Questa nota aggiunge una validazione sugli audio del pool,
che l'agente non aveva potuto fare.

| Modifica | Effetto | Tipo di giustificazione |
|---|---|---|
| L'inversione saliente A/B ferma solo la *continuazione*; non blocca più un arrivo sostenuto (`conflict_entry_vs_continuation` non viene più emesso) | prima: attacco B + inversione → astensione | **Teoria**: Lehne & Koelsch 2015 (tensione orientata al futuro, aspettativa che si risolve) e Kurby & Zacks 2008 (confine di evento come errore di predizione): un arrivo può invertire il moto precedente. La regola concreta è un'**ipotesi** |
| `a_directed_expectation`: almeno 2 domini salienti e coerenti in `A_before` (soglia 0.18, crescenti o calanti) | gate dell'arrivo | **Ipotesi ingegneristica** (proxy non calibrato di aspettativa proiettata) |
| `directed_arrival` = A diretto + attacco locale B persistente → `post_entry_attack_stabilized`; senza A diretto → `unknown`, astenuto, `score=None` | un attacco locale da solo non certifica un arrivo tra brani | Ipotesi, coerente con la regola di progetto "conservare l'ignoto" |
| `local_entry_event` esplicito, `is_cross_song_resolution=False` | solo diagnostica | Vincolo di progetto |
| Una parola attiva al cue (`word_active_at_cue=True`) rende la completion `absent` | nessun effetto sulle etichette nella pipeline attuale, perché lì la completion non è mai `supported` | Adams 2020 (chiusura di frase ≠ gap/parola) + AGENTS.md |

`EvidencePipeline.assess`, `rank_key`, `eligible` e `SUPPORTED_EVENTS` restano invariati
(test di contratto JSON aggiunto). Nessuna soglia è stata modificata.

## Validazione sul pool V12 (misura, non giudizio)

8400 candidati: 200 per ciascuna delle 42 direzioni, ottenuti da 40 opportunità d'uscita e
40 d'ingresso per brano (`runs/opportunities_v12`, `passes_constraints`, ±10 s di contesto,
seed 12). I profili sono `PooledTrajectory` sui beat osservati, come in `EvidencePipeline.profile`.
Risultati: `runs/functional_event_v12_integration/`.

| Etichetta | v1.1 | v1.2 |
|---|---:|---:|
| preparation_not_realized | 2374 | 2374 |
| conflict_entry_vs_continuation | 1962 | 0 |
| post_entry_attack_stabilized | 1181 | 2295 |
| pickup_or_preparation | 1076 | 1076 |
| post_entry_attack_transient | 783 | 783 |
| unknown | 519 | 1367 |
| sustained_motion | 323 | 323 |
| continuation_without_local_event | 182 | 182 |
| evento fisico supportato (ammissibile) | 1686 | 2800 |

Transizioni: conflict→stabilized 1465, conflict→unknown 497, stabilized→unknown 351.

**Casi sospetti.**
1. Il gate "A diretto" passa nel 73% dei candidati reali. Su rumore i.i.d. sintetico
   (8 beat) passa nel 52%: la direzione su 8 beat è vicina al livello di rumore, quindi il gate
   distingue poco. Lo stesso proxy decideva il vecchio veto (`conflict` nel 23% dei casi), che
   era quindi altrettanto fragile.
2. Il gate non privilegia i domini core: 79 arrivi si reggono solo su timbre/tonal_proxy
   (per esempio Reminder 171.8 s → Too Many Nights 10.1 s, con energia e basso calanti).
   Al contrario, Creature 58.7 s (energia +0.83, basso +0.95, timbre −1.0, tonal −0.49)
   risulta "non diretto" per pareggio 2 contro 2.
3. Il punteggio, una volta superato il gate, dipende solo da B: lo stesso cue B
   (Too Many Nights 10.1 s, intro) vale 0.849 con qualunque A. In 105 dei 280 cue B
   l'etichetta alterna tra `stabilized` e `unknown` solo in base al rumore direzionale di A.

## Diagnostica holdout (separata, nessuna taratura)

Eseguita `tools/evaluate_v12_holdout.py` sulla macchina di analisi, con v1.1 e v1.2 nello stesso ambiente.
Le clip V10 non sono presenti sulla macchina di analisi: il controllo hash delle clip è stato saltato, mentre
geometria e tracce canoniche sono verificate. La v1.1 riproduce esattamente il report congelato.
La v1.2 ha metriche identiche: positivo al 3° posto, pairwise 0.667, positivo non ammesso,
1 negativo ammesso. La copertura sale da 3 a 5. Il positivo `audio_003` passa da conflict a
`unknown` (A non diretto); i rifiuti `audio_002` e `audio_006` passano da conflict a `stabilized`.
Con otto clip non c'è alcuna evidenza di miglioramento, né di peggioramento.

## Ancora aperto (ipotesi, non implementate)

- Un test di direzione che tenga conto del rumore (finestra più lunga o null model) per A,
  da progettare senza usare l'holdout.
- Richiedere un dominio core anche nel gate di A, per coerenza con `_entry_evidence`.
- Nessuna fonte valida "arrivo" per coppie rap arbitrarie: la v1.2 non è una misura di
  risoluzione psicologica.
