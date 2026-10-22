













































































































## Objective
- Implement role-aware WHEN/WHERE transition opportunity extraction and candidate event plans for MusicMix, replacing the generic provided-cue adapter that ignores incoming/outgoing roles.
- Deliver runnable code, tests, pilot output, archived primary research, and a concise `REPORT.md` with exact commands, observed outputs, URLs, and blockers.

## Important Details
- Workspace: `<workspace>\runs\result_swarm_v12\agents\role_planning`; work only inside this folder.
- Owned paths: `src/musicmix/role_opportunities.py`, `tests/test_role_opportunities.py`, `tools/run_role_planning_pilot.py`, `outputs/`, `research/`.
- Environment: Python 3.11.9, numpy 2.4.6, pytest 9.1.1, pypdf 6.16.2. Use `PYTHONPATH=src` for project imports. CPU threads 1. No global installs; local venv or `pip --target runtime/`.
- `inputs/`, `tests/`, and `tools/` were initially empty. No real six-track catalog or actual structure/vocal JSON was provided; root supplies real data after integration.
- Synthetic fixtures must be explicitly labelled synthetic; do not claim measured musical quality, SOTA, 10/10, or invented performance.
- Playback must stay 20 s (10 s per side); analysis may use longer context. Do not treat RMS as tension, word gaps as phrase closure, or chorus labels as resolution.
- Existing generic adapter is `src/musicmix/directional.py` with `cue_policy="provided"`; it consumes `provided_cues` timestamps but ignores incoming/outgoing roles.
- Required semantics: preserve provider timestamps/canonical clock; never create fake beats; never shift anchors by more than 80 ms; distinguish outgoing completion from incoming pickup and actual arrival; record lookahead arrival delay explicitly.
- Plans must include `wait`/`prepare`/`change` events and an entry window; protect known words; diversify tracks/regions without fabricating quality scores.
- Research budget: 15 MB, official/public/publisher/author sources only; archive under `research/originals` and `research/texts`; create `sources.json` with URL, DOI, SHA256, license, actual sections read, and extraction vs reading notes. Research summaries must be in Italian and distinguish measured evidence, theory, and engineering hypothesis.
- Downloaded primary sources: `https://archives.ismir.net/ismir2020/paper/000352.pdf` (Kim 2020), `https://archives.ismir.net/ismir2019/latebreaking/000025.pdf` (M-DJCUE 2019), `https://arxiv.org/pdf/2007.08411` (cue points 2020), `https://arxiv.org/pdf/2407.06823` (CUE-DETR 2024).
- VandeVeire/De Bie 2018 DOI `10.1186/s13636-018-0134-8`; OpenAlex PDF URL `https://asmp-eurasipjournals.springeropen.com/track/pdf/10.1186/s13636-018-0134-8` was blocked by Cloudflare/401.
- Useful research notes found: Kim 2020 defines cue-out where previous track starts fading out, cue-in where previous track can no longer be heard, and transition region between them; M-DJCUE says cue-in lies in intro up to first beat of core and cue-out excludes intro; CUE-DETR relates cue points/phrase boundaries and notes vocal/pickup annotations from other DJ genres; CUE points 2020 Rule 3 requires the post-switch section to “stand on its own,” switch points are downbeat-aligned, and vocal cuts should be prevented.

## Work State
### Completed
- Explored project structure, `TASK.md`, `pyproject.toml`, `knowledge/` summaries, and relevant modules including `contracts.py`, `structure.py`, `vocal_boundaries.py`, `phrase_evidence.py`, `learned_opportunities.py`, `audio.py`, `analysis.py`, `search.py`, `ranking.py`, `directional.py`.
- Confirmed `src/musicmix` snapshot contains beat/structure/vocal modules but no role-aware opportunity module.
- Downloaded and archived 4 PDFs, total 2,447,588 bytes:
  - `research/originals/kim_2020.pdf` 261,906 bytes
  - `research/originals/mdjcue_2019.pdf` 71,439 bytes
  - `research/originals/cuepoints_2020.pdf` 416,701 bytes
  - `research/originals/cuedetr_2024.pdf` 1,694,504 bytes
- Created `research/extract_pdfs.py`; extracted page-anchored text:
  - `cuedetr_2024.pdf`: 8 pages, 36,298 chars
  - `cuepoints_2020.pdf`: 7 pages, 33,715 chars
  - `kim_2020.pdf`: 7 pages, 35,384 chars
  - `mdjcue_2019.pdf`: 2 pages, 8,399 chars
- Created `src/musicmix/role_opportunities.py` implementing `PlanConfig`, `RoleOpportunity`, `PlanEvent`, `CandidatePlan`, `TrackOpportunities`, `RolePlanResult`, `extract_track_opportunities`, `select_role_plans`, `generate_role_plans`, and `plan_catalog`.
- The module consumes track mappings with `structure` beats/downbeats, `vocals` words/coverage, `provided_opportunities`/`provided_cues`, and optional `vocal_completions`; it maps OUT/IN roles, treats neutral LinkSeg boundaries as both roles, derives incoming pickup from ASR word onsets, keeps arrival separate, and emits wait/prepare/change plans with entry windows.
- Created `tests/test_role_opportunities.py` with synthetic contract tests covering role separation, pickup vs arrival delay, exact anchor preservation, missing-grid handling, word protection, plan events, seconds conversion, ASR segment endpoints, named vocal completions, six-track API diversity, and consumption of a real `structure.analyze_structure(...).to_dict()` document.

### Active
- Code and tests have been written but not yet executed. Need run pytest and fix expected integration/API mistakes.
- Known likely code issues to review/fix:
  - `generate_role_plans` diagnostics uses `all(not track.beats or True for track in extracted)`, so `no_fake_beats` is always true; replace with meaningful `tracks_without_grid` / anchor-shift diagnostics.
  - `_coverage_unknown_at` calls `_sample_from_record(record, 1, ...)`, using sample rate 1; coverage interval parsing likely needs the actual track sample rate or supported fields.
  - `_normalise_words` treats a bare `end` field as seconds; canonical contract should prefer `end_sample`.
  - Verify `ProviderPrediction` constructor, `analyze_structure` provider interface, and `StructureAnalysis.to_dict()` fields match test assumptions.
  - Verify `select_role_plans` evidence tuple sort and dataclass JSON serializability.
- No `tools/run_role_planning_pilot.py`, `outputs/`, `research/sources.json`, `research/RIASSUNTO.md`, or `REPORT.md` yet.

### Blocked
- VandeVeire full PDF unavailable: Springer/imec links returned HTML/Cloudflare or HTTP 401; local `research/originals/vandeveire_2018.pdf` is a 3,038-byte HTML error page, not a valid PDF. `research/originals/imec_handle.html` and `research/originals/openalex_vandeveire.json` exist.
- No real catalog/input data; pilot must use a clearly labelled synthetic six-track fixture unless `--catalog` is supplied.

## Next Move
1. Run `$env:PYTHONPATH='src'; python -m pytest tests/test_role_opportunities.py -q` from the workspace and fix all failures; re-run until green.
2. Review/fix `role_opportunities.py` diagnostics and coverage parsing, especially the always-true `no_fake_beats` expression and `_coverage_unknown_at` sample-rate handling.
3. Create `tools/run_role_planning_pilot.py` with optional `--catalog`, synthetic six-track fallback labelled `synthetic_contract_fixture`, and JSON output to `outputs/role_planning_pilot.json`; run it with `$env:PYTHONPATH='src'; python tools/run_role_planning_pilot.py`.
4. Write `research/RIASSUNTO.md` in Italian, `research/sources.json` with SHA256/URL/DOI/license/sections read, and `REPORT.md` with exact commands, observed outputs, source URLs, and remaining blockers.

## Relevant Files
- `src/musicmix/role_opportunities.py`: newly created role-aware extraction/planning API; needs test-driven fixes.
- `tests/test_role_opportunities.py`: synthetic contract tests; not yet run.
- `tools/run_role_planning_pilot.py`: still to create; runnable pilot entry point.
- `outputs/`: still to create; store pilot JSON.
- `research/extract_pdfs.py`: PDF-to-text extraction helper used.
- `research/originals/`: archived PDFs plus failed VandeVeire HTML/OpenAlex metadata.
- `research/texts/`: extracted page-anchored paper text files.
- `research/sources.json`: still to create.
- `research/RIASSUNTO.md`: still to create; Italian research summary.
- `REPORT.md`: still to create; exact commands/outputs/blockers.
- `src/musicmix/directional.py`: existing provided-cue adapter whose role-blind behavior motivated the new module.
- `src/musicmix/structure.py`: real beat/downbeat document contract used by tests.
- `src/musicmix/vocal_boundaries.py` and `src/musicmix/phrase_evidence.py`: vocal/word/coverage semantics to consume.
- `knowledge/`: Italian research summaries and evidence rules.
- `TASK.md` and `pyproject.toml`: task constraints and pytest configuration.





































































































































