















































































































































## Objective
- Build and run an event-centric A→B comparator that fixes weak continuation logic in `src/musicmix/functional_event.py`, using `A_before`, `A_after`, `B_before`, `B_after`, vocal-completion evidence and section context.
- Deliver runnable code, tests, pilot outputs, archived primary research, and a concise `REPORT.md` with exact commands and measured results.

## Important Details
- Workspace: `<workspace>\runs\result_swarm_v12\agents\functional_event`; work only inside this folder.
- Python on Windows; project imports require `$env:PYTHONPATH='src'`. CPU threads 1; no global installs; download budget 20 MB.
- Playback stays 20 s (10 s per side); analysis may use longer context.
- Required semantic constraints: no `chorus = resolution`, no word-gap-as-phrase-closure, no global scalar false certainty, no native-A identity reward, preserve domain disagreement and abstention.
- Existing weak logic: accepted any two matching domains even if salient other domains reversed. New scorer uses weighted agreement, explicit reversal rejection, local `B_before → B_after` event contrast, persistence/stabilization, preparation vs attack distinction, and `unknown` with null score on conflict/incomplete evidence.
- Pilot beats are **observed spectral-flux onset events** from `musicmix.analysis.analyze_track`, explicitly not a validated beat grid; `PooledTrajectory` is reused with those observed beats only, never a synthesized grid.
- Research summaries must be Italian and distinguish measured evidence, theory, and engineering hypothesis.
- Primary sources archived: Hansen 2021, Lehne & Koelsch 2015, Kurby & Zacks 2008, Adams 2020, Sridharan 2007 (abstract only), Farbood 2012 (abstract only).
- New root-usable contract: `FunctionalEventScorer.score(candidate, profiles)`, `.rerank(baseline_score, profiles=...)`, `.rerank_pool(rows, profile_lookup)`; pool rows receive `event_baseline_score`, `event_adjusted_score`, `event_score`, `event_label`, `event_abstained`, `event_weight`, `event_missing`.
- Pilot labels/scores:
  - `B_actual`: `pickup_or_preparation`, `+0.38`
  - `B_event_muted`: `preparation_not_realized`, `-0.29956027667078877`
  - `B_event_transplanted`: `preparation_not_realized`, `-0.1793799281835562`
  - Synthetic `same_slope_no_entry_bump`: `post_entry_attack_transient`, `-0.40723460156859614`
  - Synthetic `same_slope_with_entry_bump`: `post_entry_attack_stabilized`, `0.7010960284334358`
  - Synthetic `two_matches_two_reversals`: `conflict_entry_vs_continuation`, no score
- Tests currently pass: `10 passed in 0.29s`.
- Research archive currently has a broken Kurby source: `https://pmc.ncbi.nlm.nih.gov/articles/PMC2263140/` returns a reCAPTCHA page (`Checking your browser - reCAPTCHA`) of 21,313 bytes. Replace with author PDF found at `https://www.gvsu.edu/cms4/asset/92386C7F-BF75-96ED-67D9FAA81701D122/kurby__zacks_2008.pdf`.
- `research/RIASSUNTO.md` currently says `858.565 byte`; this is stale after the last re-download and must match the final valid archive size.

## Work State
### Completed
- Created `src/musicmix/functional_event.py` with:
  - `VERSION = "functional-event-v1-event-centric"`
  - `TRANSFER_STATUS = "unvalidated_engineering_comparator"`
  - `FunctionalEventThresholds`, `DomainEventEvidence`, `FunctionalEventAssessment`, `FunctionalEventRerank`, `VocalObservation`, `SectionObservation`
  - `assess_functional_event(...)`, `rerank_functional_event(...)`, `FunctionalEventScorer`
  - Labels: `post_entry_attack_stabilized`, `post_entry_attack_transient`, `preparation_not_realized`, `pickup_or_preparation`, `sustained_motion`, `continuation_without_local_event`, `conflict_entry_vs_continuation`, `unknown`
- Fixed two-matching-domain weakness: continuation now requires weighted majority, `agreement_floor=0.55`, at least 2 matching domains, and rejects salient reversals via `contradiction_weight=0.30`.
- Local `B_before → B_after` contrast now affects label/score; `B_event_muted` changes from `pickup_or_preparation` to `preparation_not_realized`.
- Created `tests/test_functional_event.py` with 10 tests covering weak continuation, local event sensitivity, same-slope/different-event controls, pickup, word-gap rejection, chorus-not-resolution, abstention, rerank contract, and threshold validation.
- Created `tools/run_functional_event_pilot.py`; ran it successfully.
- Created staged audio:
  - `outputs/pilot/audio/a_staged_20s.wav`
  - `outputs/pilot/audio/b_actual_20s.wav`
  - `outputs/pilot/audio/b_event_muted_20s.wav`
  - `outputs/pilot/audio/b_event_transplanted_20s.wav`
- Created `outputs/pilot/results.json` and `outputs/pilot/pool_rerank_contract_demo.json`.
- Created `research/collect_research.py`, `research/originals/`, `research/texts/`, `research/sources.json`, `research/RIASSUNTO.md`.
- Fixed `rerank_pool` to select profile keys case-insensitively through `self._profile(...)`.
- Ran `python -m compileall -q ...` successfully.
- Verified commands:
  - `$env:PYTHONPATH='src'; python -m pytest tests\test_functional_event.py -q`
  - `$env:PYTHONPATH='src'; python tools\run_functional_event_pilot.py`
  - `python research\collect_research.py`

### Active
- Research archive needs repair for Kurby & Zacks 2008; the current `research/texts/kurby2008.txt` begins with the reCAPTCHA page, not the paper.
- Need to update `research/RIASSUNTO.md` byte count/hash references after repairing the archive.
- `REPORT.md` has not been created yet.

### Blocked
- PMC anti-bot blocks direct `https://pmc.ncbi.nlm.nih.gov/articles/PMC2263140/`; current local Kurby archive is invalid.
- Sridharan 2007 and Farbood 2012 are abstract/metadata only because publisher full text is paywalled.
- No validated beat/downbeat provider, separated-vocal completion evidence, functional section provider, human listening ratings, or real pool evidence; transfer remains explicitly unvalidated.

## Next Move
1. Patch `research/collect_research.py` so Kurby 2008 uses `https://www.gvsu.edu/cms4/asset/92386C7F-BF75-96ED-67D9FAA81701D122/kurby__zacks_2008.pdf`, re-run `python research\collect_research.py`, and verify `research/texts/kurby2008.txt` no longer contains `Checking your browser - reCAPTCHA`.
2. Update `research/RIASSUNTO.md` with the final archive byte count and any changed hashes; then write `REPORT.md` with exact commands, observed outputs, source URLs, theory-vs-measurement separation, and remaining blockers.
3. Re-run final verification commands (`pytest`, pilot) and ensure `REPORT.md` quotes the final observed outputs.

## Relevant Files
- `src/musicmix/functional_event.py`: main comparator, scorer, abstention, local B-event logic.
- `tests/test_functional_event.py`: 10 tests; all currently pass.
- `tools/run_functional_event_pilot.py`: actual local staged-audio pilot plus synthetic controls.
- `outputs/pilot/results.json`: measured pilot cases, threshold metadata, source clocks, limitations.
- `outputs/pilot/pool_rerank_contract_demo.json`: root-rerank contract demonstration.
- `outputs/pilot/audio/`: staged 20-second clips and diagnostic edits.
- `research/collect_research.py`: primary-paper downloader/extractor; Kurby URL must be replaced.
- `research/sources.json`: current manifest; Kurby entry is invalid until fixed.
- `research/RIASSUNTO.md`: Italian research summary; byte count is stale.
- `research/originals/`, `research/texts/`: archived bytes and extracted text.
- `REPORT.md`: required final concise report, not yet written.



























Implemented and verified.

- `REPORT.md` created with exact commands, observed outputs, source URLs, theory-vs-measurement separation, and blockers.
- Tests: `10 passed in 0.39s`.
- Pilot: `B_actual` → `pickup_or_preparation` `+0.38`; `B_event_muted` → `preparation_not_realized` `-0.29956027667078877`; `B_event_transplanted` → `preparation_not_realized` `-0.1793799281835562`.
- Research archive repaired: Kurby full text is now 189,412 bytes; total archive 858,603 bytes. `research/RIASSUNTO.md` corrected.

Remaining limitations: no human listening validation, no validated beat/vocal/section evidence; Sridharan 2007 and Farbood 2012 remain abstract-only.