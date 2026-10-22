# Ultimo delta audit — `6f03b33` → `bec3420`

Audit statico del diff di produzione, con tre smoke test di funzioni isolate estratte da `git show bec3420:<file>`. Non ho eseguito la pipeline, `check_books_integration_real` né la suite. I conteggi `242 pass / 8 skip / 6 fail` restano attribuiti a `6f03b33`, non a `bec3420`.

## Controesempi root

- **Ancora presente — P0, `dj_form.form_segments`.** `src/musicmix/dj_form.py` non cambia. Con otto battute senza attività di layer/cassa, `full.any()` è falso, `first_core=None`, `last_core=None`, ma il ciclo outro assegna tutte le etichette `outro`. Serve `unknown`/`unsupported` o una gestione esplicita dell'assenza di core; non è superato.
- **Ancora presente — P0, `dj_phrase.with_stem_onsets`.** Il delta tocca solo `vocal_phase_alignment`; il limite `searchsorted(..., max(off, tg[-1] + md[-1]))` resta. Test isolato su `off=1.000 s`, onset stem `1.040 s`: la seconda sillaba è marcata `stem` a 1.040 s. La finestra può oltrepassare la fine ASR; non è superato. La nuova `onsets_s=` è diagnostica raw e non corregge l'assegnazione.
- **Ancora presente — P0, `dj_set.apply_far_genre`.** `far_timbre_threshold` restituisce il percentile del pool senza escludere il caso degenerato. Matrice timbrica 3×3 tutta zero → soglia `0.0`; distanza `0.0` soddisfa `>=` e classifica `far`, aggiungendo `tail_stop=True` a `echo_out`. È ancora una misura relativa, non un detector di generi lontani; non è superato.

## Cosa cambia davvero

- `tools/build_listening_artifact.py` aggiunge `--blind`/`blind=True`: nel markup visibile usa “versione N”, nasconde badge tecnica, warning QA e metadati del piano e li rivela dopo un giudizio salvato; registra `plays_before_save` e ha `test_dj_review_blind_hides_technique_until_vote`. Questo **supera solo parzialmente** il gap UI: è opt-in, il default resta non cieco, il test è principalmente uno snapshot HTML e `clip-data` conserva `kind`/`label`/`qa_verdict` mentre l'id del clip resta visibile. Per un ascolto cieco reale va chiamato esplicitamente `--blind` e vanno verificati gli id/JSON non informativi.
- `tools/check_books_integration_real.py` è un nuovo helper per render locali e QA tecnico (`dropout_check=True`), con clip WAV e report locale. Non invoca `dj_phrase` né `dj_critic`; non c'è un risultato prodotto da questo commit. Inoltre il report non registra/verifica gli SHA delle sorgenti canoniche rispetto al `set.json`, quindi non autorizza ad ascoltare o citare eventuali risultati remoti: serve un run locale con sorgenti identificate e ascolto separato.
- `tools/run_dj_set_v2_books.py` continua a non importare/chiamare phrase o critic. La selezione dei tentativi continua a scegliere `best_fail` quando tutti falliscono e salva `kept_despite_fail=true`: il gap **all-fail kept flag è superato come tracciabilità**, ma il fallimento resta tenuto nel set. Se la raccomandazione richiede rifiutare ogni candidato `fail`, resta aperta; non c'è promozione nascosta.
- La QA usa ancora `dj_qa.b_source_position`, che per i render stretchati proietta con una retta basata su `tempo_ratio_b`; il `_PiecewiseMap` del renderer non viene usato per la proiezione di parole/stem. Il gap time-map QA lineare resta aperto.
- La guardia nuova in `dj_policy` evita `break_before_entry` quando A continua a cantare oltre il cue e `dj_creative` registra `a_vocal_across_downbeat`. È una correzione locale alla tecnica break, non la validazione generale di cue/ruolo richiesta dalla revisione.

## Raccomandazioni non superate

Restano aperte: gestione esplicita di form senza core e confini onset, separazione/calibrazione delle ipotesi libri, cue per ruolo/tecnica, integrazione reale di phrase+critic nel runner, QA piecewise e copertura per tecnica, policy che non tenga un candidato `fail`, validazione percettiva del critic, F5 con audio annotato e contesto del set. Le nuove tecniche creative, il percentile `far_timbre_pct=90`, il dropout check e `--blind` sono opzioni/profili con test tecnici; non costituiscono validazione musicale né cancellano le cautele di `REVISIONE_PIPELINE.md`.

File osservati nel delta: `src/musicmix/dj_creative.py`, `src/musicmix/dj_phrase.py`, `src/musicmix/dj_policy.py`, `src/musicmix/dj_qa.py`, `src/musicmix/dj_render.py`, `tools/build_listening_artifact.py`, `tools/check_books_integration_real.py`, `tools/run_dj_set_v2_books.py`, `tests/test_build_listening_artifact.py`, `tests/test_dj_books_integration.py` e le note `conoscenza_mix/libri/INTEGRAZIONE_PROPOSTE_2026-10-21.md` / `PROPOSTE_CRITICO_DA_INTEGRARE.md`.
