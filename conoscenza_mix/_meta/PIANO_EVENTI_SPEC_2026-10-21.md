# Piano per eventi: specifica dei dati (P0.5, 21/10/2026)

Stato: **implementata il 21/10/2026** (`src/musicmix/dj_events.py`, run `runs/dj_set_v3_events_plan/`; differenze e limiti in
[`PIANO_EVENTI_IMPL_2026-10-21.md`](PIANO_EVENTI_IMPL_2026-10-21.md)). Testo originale della specifica: Pronta per la tranche successiva del backlog dell'audit
(`libri/audit_2026-10-21/BACKLOG.json`, item P0.5, dipende da P0.2/P0.3/P0.4). Riferimenti:
[revisione della pipeline](../libri/audit_2026-10-21/REVISIONE_PIPELINE.md) («dove siamo nel brano A, quale evento vogliamo
far entrare di B, come e quando renderlo udibile»), [`P0_MISURE_REGOLE_2026-10-21.md`](P0_MISURE_REGOLE_2026-10-21.md).

## 1. Perché

Oggi un candidato di transizione è una coppia di downbeat (`dj_select.plan_transition`: `a_out_bar`, `b_in_bar`) più una
decisione dell'albero (`dj_set.decide_transition`: tipo, battute, render). Il contesto (`dj_set.transition_context`) descrive
le due posizioni, ma non dice **quale evento di A si salta o si completa**, **quale evento di B è l'obiettivo** né **quanto
dista** dal punto in cui B diventa udibile. Le pratiche dei libri lo chiedono esplicitamente, senza dare soglie:

- B&B 2003 cap. 11 e 13 [P]: collocare due arrangiamenti; pickup; «mix straight after a chorus ... the big opening moment in
  record 2 happens immediately after the chorus in record one» (cap. 13 'Even tighter'); elementi futuri di B conservati o
  saltati.
- Webber 2008 p. 225 [P]: beat sync e phrase sync sono cose distinte; non interrompere una frase.
- Krims 2000 p. 98, Fig. 3.2 [A] (citato dalla revisione, non riletto qui): configurazioni, upbeat, aggiunte e ritornelli come unità diverse; p. 133 e p. 194 [A] (riletti): un
  verso senza upbeat chiude l'unità, una cesura su BC 2 lascia BC 3 per l'upbeat (un brano per pagina).
- Butler 2006 pp. 222–225 [A] e p. 243 [P]: funzioni formali intro/build/core/breakdown/outro; «the builds won't be the same».
- Huron 2006 pp. 248–249, 314–316, 368 [A/T]: un fill rende certo il «quando»; ritardare un evento atteso aumenta la tensione,
  ma non troppo a lungo. Nessuna di queste pagine è una soglia di qualità per un mix (AGENTS.md).

Il piano per eventi rende questi elementi **campi espliciti e verificabili**, così il QA e l'ascolto possono controllare che
l'evento previsto sia quello udito, e i confronti possono variare un solo elemento alla volta (stessa coppia e tecnica,
arrivo o lead-in diversi).

## 2. Schema dei dati

Unità: campioni della sorgente (48 kHz) + indice di battuta della griglia di `dj_select` (stessa griglia di `dj_form` e dei
modelli `dj_phrase`). Ogni campo misurato ha `source` (modulo e versione) e `status` ∈ {`measured`, `estimated`,
`unavailable`, `not_applicable`}: un evento non misurabile resta `unavailable`, non viene inventato (P0.2: niente
«outro» senza core, niente onset assegnati oltre la parola).

```jsonc
// musicmix.event_plan.1 — un candidato di transizione A -> B
{
  "schema": "musicmix.event_plan.1",
  "a": "slug_A", "b": "slug_B",
  "grid": {"a_period_s": 0.5, "b_period_s": 0.49, "stretch_b_pct": 2.1},   // da dj_select.pair_compatibility

  // ---- A: l'evento in corso al punto d'uscita e che cosa ne resta
  "a_event": {
    "kind": "verse|chorus|hook|bridge|intro|outro|breakdown|build|core|instrumental|unknown",
    "label_source": "allin1|dj_form|dj_phrase",          // da quale analisi viene l'etichetta
    "span_bars": [24, 40],                                // sezione allin1 / segmento dj_form che contiene l'uscita
    "exit_bar": 40, "exit_sample": 3840000,               // = plan.a_out_bar / a_out_s
    "fate": "completed|skipped_rest|cut_mid_line|unknown",// completato fino al confine, resto saltato, verso troncato
    "skipped": [{"kind": "chorus", "span_bars": [40, 48]}],  // eventi futuri di A non suonati (B&B cap. 13)
    "line_end": {"status": "clean|tail|a_pickup|mid_line|instrumental|asr_unreliable",  // dj_phrase.exit_point
                 "p_line_end": 0.82, "last_word": "...", "tail_end_sample": null,
                 "a_pickup_sample": null, "closure_diag": "closed_bc3|closed_early|pickup|null"},  // krims_closure: SOLO diagnostica
    "form_at_exit": {"before": "core", "at": "outro", "core_status": "found|no_core"}  // dj_form.overlap_profile / form_segments
  },

  // ---- B: come entra e quale evento è l'obiettivo
  "b_entry": {
    "audible_sample": 1234000,       // primo campione udibile di B nel render (inizio blend/fill/pickup)
    "lead_in": {"kind": "none|pickup|fill|build|filter_sweep|echo_tail|loop|break",
                "bars": 1.0, "pickup_words": ["yeah"], "pickup_samples": 9600},   // dj_phrase.entry_point / dj_policy
    "downbeat_bar": 0, "downbeat_sample": 35640                                     // = plan.b_in_bar / b_in_s
  },
  "b_target": {
    "kind": "chorus|hook|verse|drop|section_start|first_vocal|instrumental|unknown",
    "bar": 8, "sample": 1180000,     // dove l'evento obiettivo di B arriva (può coincidere con il downbeat d'ingresso)
    "evidence": ["allin1:chorus@8", "dj_form:drop@8"],
    "status": "measured|estimated|unavailable"
  },
  "delay": {                         // distanza tra B udibile e l'evento obiettivo: il «ritardo» di Huron, NON una soglia
    "bars": 8.0, "s": 15.6,
    "a_audible_until_sample": 3870000  // fine udibile di A (coda d'eco, blend)
  },

  // ---- la scelta e le alternative valutate per la STESSA coppia
  "choice": {"technique": "blend|cut|echo_out|filter_out|filter_into_chorus|backspin|halving_build|break_before_entry|...",
             "bars": 8, "render": {}, "tree_rule": "R8_one_voice", "policy_rule": null, "cost": 0.34},
  "alternatives": [
    {"vary": "cue",       "a_out_bar": 32, "b_in_bar": 0, "cost": 0.40, "why_not": "..."},
    {"vary": "duration",  "bars": 4,  "cost": 0.37},
    {"vary": "technique", "technique": "echo_out", "cost": 0.52},
    {"vary": "continue_A", "note": "solo riferimento di confronto, mai clip da votare 1-10 (AGENTS.md)"}
  ],

  // ---- tracciabilità
  "trace": {"modules": {"dj_select": "musicmix.dj_select.1", "dj_set": "...", "dj_form": "musicmix.dj_form.2",
                        "dj_phrase": "musicmix.dj_phrase.1", "dj_policy": "...", "dj_critic": null},
            "executed": ["dj_select.plan_pair", "dj_set.decide_transition", "dj_phrase.exit_point", "dj_phrase.entry_point"],
            "cue_windows": "default|technique", "profile": "v1|books",
            "sources": {"a_sha256": "...", "b_sha256": "..."}}
}
```

Regole del contratto:

1. `a_event.fate` e `b_target` sono **descrizioni**, non giudizi: un `skipped_rest` può essere la scelta giusta (B&B cap. 13,
   uscire subito dopo il ritornello), un contrasto preparato può funzionare (revisione, Huron pp. 314–316).
2. `delay` non ha un valore «buono» di default. Eventuali preferenze si stimano con confronti a coppie sullo stesso contesto
   (P1.1), non dai libri.
3. `closure_diag` (Krims pp. 133, 194) resta diagnostica: non entra nel costo (Krims p. 32 riguarda la chiusura del discorso
   interpretativo, non una cadenza; verifica 21/10/2026).
4. Le alternative si enumerano **prima** del ranking e si conservano tutte nel trace, anche le non ammissibili con il motivo,
   così un ascolto può confrontare stessa coppia/tecnica con arrivi o lead-in diversi.

## 3. Mappa sulle strutture esistenti (che cosa c'è già, che cosa manca)

| Campo | Esiste già in | Manca |
|---|---|---|
| `a_event.span_bars`, `kind` (allin1) | `dj_select.phrase_table` (`section_label`, `bars_to_next_section`), `dj_set.section_spans` | span della sezione corrente nel contesto: oggi solo `a_exit_section` (etichetta della battuta prima) |
| `a_event.fate`, `skipped` | `dj_set.sections_played` (sezioni intere suonate fra ingresso e uscita) | calcolo dell'evento in corso **al** punto d'uscita e degli eventi saltati; `cut_mid_line` da `dj_phrase.exit_point.status = mid_line` |
| `a_event.line_end` | `dj_phrase.exit_point` (`status`, `p_line_end`, `last_word`, `tail_end_sample`, `a_pickup_sample`, `krims_closure`) | collegamento al planner del set: `run_dj_set_v3_bridge.py` lo usa come guardia, `dj_set.choose_cues` no |
| `a_event.form_at_exit` | `dj_set.form_context` (`a_form_before`, `a_form_at`, `a_core_known`) da `dj_form.overlap_profile` | — (pronto; `unknown` se `core_status = no_core`) |
| `b_entry.lead_in` | `dj_phrase.entry_point` (`pickup_samples`, `pickup_words`), `dj_set.lead_a_s` (sweep/eco/spin), `dj_policy.propose` (build, halving, break) | un unico campo con il tipo di lead-in e la durata; oggi è sparso fra decisione, policy e render |
| `b_entry.audible_sample` | metadati di `dj_render`/`dj_creative` (`regions`, `b_map`) dopo il render | stima **prima** del render (dal tipo e dalle battute) e verifica dopo (P0.6, time-map) |
| `b_target` | `transition_context.b_section`, `b_section_start`; `dj_form.drops`; `dj_select` (`SECTION_CORE`) | ricerca del primo evento obiettivo dopo `b_in_bar` (ritornello, drop, prima voce) e `status` |
| `delay` | — | differenza `b_target.sample - b_entry.audible_sample` (in battute di B) |
| `choice` | `dj_set.candidates_for_pair` (`decision`, `cost`), `dj_policy.propose` | — |
| `alternatives` | `dj_set.alternatives` (solo cue con vincoli di durata), `decision["fallbacks"]` dell'albero | enumerazione per asse (cue / durata / tecnica) con la stessa coppia; `continue_A` come riferimento |
| `trace` | `run_dj_set_v2_books.py` (`versions`, `params`, `provenance` delle forme), `cue_windows` | elenco dei moduli **eseguiti** per quel candidato (la revisione nota che phrase e critic v3 non sono invocati dal runner books) |

## 4. Implementazione proposta per la tranche successiva (non fatta qui)

1. `dj_set.event_plan(A, B, cand, phrase_a=None, phrase_b=None) -> dict` puro, che costruisce lo schema sopra da un candidato
   di `candidates_for_pair` + (facoltativo) modelli `dj_phrase`; nessun cambiamento di costo.
2. `candidates_for_pair(..., event_plans=False)`: con `True` allega `event_plan` a ogni candidato (memoria: solo campi scalari;
   con le finestre "technique" i candidati sono ~128 mila sul catalogo del ponte, quindi calcolo pigro o solo per i primi
   `max_per_transition = 400` di `choose_cues`).
3. Runner: `run_dj_set_v2_books.py`/`run_dj_set_v3_bridge.py` scrivono `event_plan` della transizione scelta e delle
   alternative nel `set.json`; il QA (P0.6) confronta `b_entry.audible_sample` e `b_target.sample` con il time-map renderizzato.
4. Test: fixture sintetiche con (a) uscita a fine ritornello e ritornello successivo saltato, (b) B con pickup di 2 battiti,
   (c) B il cui ritornello arriva 8 battute dopo l'ingresso, (d) brano senza core → `form_at_exit.core_status = no_core` e
   `kind = unknown`; un test che verifica che `event_plan` non cambia scelte né costi.
5. Ascolto (P1.1): stessa coppia e tecnica, `delay` diversi (arrivo diretto sul ritornello vs 8 battute prima), confronto a
   coppie cieco; nessuna soglia di `delay` prima di quei dati.
