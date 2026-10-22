# External cue benchmark: runnable result

## What was built

A root-ready benchmark that reads the complete public **M-DJCUE** JAMS annotations, creates a deterministic grouped split, and evaluates a prediction JSON for `IN`/`OUT` cue timing, abstention, region coverage and annotator agreement.

Official dataset repository discovered from the papers: `MZehren/M-DJCUE`, revision `7fd692c793a4a4744ab61e409854302269b0974c`. The `MarcoStringa/M-DJCUE` URL in the task returned **HTTP 404** (verified through the GitHub API), so it is not used.

No audio was downloaded. No package was installed; Python 3.11.9 and the already-present `pypdf 6.16.2` were sufficient.

Artifacts:

- `tools/build_external_transition_benchmark.py` — reader, grouped split, evaluator, baselines, CLI.
- `outputs/mdjcue_benchmark.json` — 134 tracks, raw events, merged clusters, regions, split.
- `outputs/split.json` — group membership and leakage checks.
- `outputs/baseline_position_fraction_predictions.json`.
- `outputs/baseline_position_absolute_predictions.json`.
- `outputs/evaluation_position_fraction.json`.
- `outputs/evaluation_position_absolute.json`.
- `outputs/evaluation_cli_baseline.json` — same real baseline consumed through the public `evaluate` command.
- `outputs/baseline_summary.json`, `outputs/benchmark_summary.json`, `outputs/README.md`.
- `research/sources.json`, `research/SUMMARY.md`, `research/originals/`, `research/texts/`.
- `tests/test_external_benchmark.py`.

## Exact commands and observed results

```powershell
python tools\build_external_transition_benchmark.py all --out outputs --seed 20261020 --test-fraction 0.3
```

Observed data counts:

```text
tracks=134 events=1849 clusters=1082
IN:  events=1037 clusters=565 support>=2=311 support>=3=161
OUT: events=812  clusters=517 support>=2=224 support>=3=69
cluster support histogram: {1:547, 2:305, 3:230}
```

Observed split:

```text
119 grouped components; 36 test groups
train=94 tracks / 83 artists
test=40 tracks / 36 artists
artist_overlap=[]
version_family_overlap=[]
```

The split groups tracks connected by the same normalized artist or the same artist+base-title version family. It was reproduced in a separate output directory; test IDs and all content except the generation timestamp were identical.

```powershell
python tools\build_external_transition_benchmark.py evaluate --predictions outputs\baseline_position_fraction_predictions.json --out outputs\evaluation_cli_baseline.json
```

Observed metadata-only position baseline at 0,5 s tolerance on the frozen test split:

| Baseline | Gold | Pred in region | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train median duration fraction | 346 | 73 | 3 | 70 | 343 | 0,0411 | 0,0087 | 0,0143 |
| train median absolute time | 346 | 70 | 2 | 68 | 344 | 0,0286 | 0,0058 | 0,0096 |

The fraction baseline predicts `IN` at `0.1089 * duration` and `OUT` at `0.7748 * duration`, fitted only on the 94 training tracks. Per role, its `IN` precision/recall are `0.0571 / 0.0113`; `OUT` is `0.0263 / 0.0059`. On the higher-agreement subset (`support >= 2`, 167 gold clusters) it gets TP=2, FP=70, FN=165. This is a metadata/position baseline, not an audio algorithm and not evidence that WHENmix can detect cues.

Observed annotator agreement on all 134 tracks (same track/label, greedy one-to-one matching at 0,5 s):

```text
IN : mean pairwise F1 0.6211, 402 pairs
OUT: mean pairwise F1 0.4238, 400 pairs
all: mean pairwise F1 0.5227, 802 pairs
```

Observed test coverage:

```text
IN region : 40/40 tracks, 25.9% of total duration
OUT region: 40/40 tracks, 56.9% of total duration
unscored gap: mean 97.8 s, median 75.3 s
```

Region definitions are empirical proxies: `IN=[0, max(IN)+0.5 s]` and `OUT=[min(OUT)-0.5 s, duration]`. Predictions outside these regions are reported as out-of-region, never as false positives. Missing predictions are abstentions and lower recall. There are no inferred negatives at unannotated points.

```powershell
python -m pytest -q tests\test_external_benchmark.py
```

Observed: `3 passed in 0.15s`.

## Prediction input

The evaluator accepts flat rows (`predictions`) or per-track cues (`tracks`), with `time_s` or `time`, and `abstain: true` rows. Unknown track IDs are rejected. See `outputs/README.md` for the exact schemas.

## Sources and archive

- Dataset: https://github.com/MZehren/M-DJCUE
  - Revision: `7fd692c793a4a4744ab61e409854302269b0974c`
  - License: GPL-3.0 for repository annotations/metadata; audio copyrighted and not distributed.
  - Archived tarball `research/originals/mdjcue_repo_7fd692c.tar.gz`, SHA256 `6f6c8cccbe0707ecabe4bf3e74b449898c0c755555bcaded7effc3be107bba92`, 967052 bytes.
  - Combined sorted JAMS SHA256 `c1729ee5d2f8f07487e5699b1331829d5e0806c4271265658cca4e77afd92629`.
- ISMIR 2019 late-breaking abstract: https://archives.ismir.net/ismir2019/latebreaking/000025.pdf
  - CC BY 4.0 per paper; 2 pages; SHA256 `2501589c39348d9d7faf96acbe2d7f25a1f569ed70118252d1429a5080c05ad3`.
- arXiv 2007.08411: https://arxiv.org/abs/2007.08411
  - DOI `10.48550/arXiv.2007.08411`; 7 pages; PDF SHA256 `de51eed1cb28b2679c9a5b926153db44b5706fb0103474be45d1fc6b4883c2cd`.
  - HTML SHA256 `b5499b6f1740cebaf37080ff04bd4ef37002f434ae074a8493dec0a9687d9bf9`.
- GitHub API repo/commit metadata is in `research/originals/github_repo_M-DJCUE.json` and `github_commit_7fd692c.json`.
- `research/sources.json` records URLs, DOI, SHA256, license, actual sections read, and whether evidence came from extraction or direct reading.

Reported paper results are labeled as paper results: 86% precision and 49% recall with all rules at 0,5 s. They were not reproduced here.

## Remaining blockers and domain limits

- M-DJCUE is EDM 1987-2016, not rap/trap. Transfer to WHENmix is unproven.
- No audio is distributed or downloaded, so this benchmark cannot check sample alignment, loudness, vocal cuts or render artifacts.
- No beat/downbeat grid is shipped; a downbeat baseline is explicitly `not_available`.
- No A-to-B transition pairs or professional DJ-set alignment are shipped. Cue points are per-track and independent of the next track; no pair evidence was fabricated.
- Four vinyl tracks have annotations beyond the declared metadata duration. The reader uses `duration_s = max(declared duration, last annotated event)` and lists every case in `reading_issues`.
- The empirical label regions are coverage proxies, not proof that all unannotated times are acceptable or unacceptable.
- The baseline is weak by design. It validates the reader/evaluator plumbing and the difficulty of 0,5 s cue timing, not a WHENmix audio model.
- The benchmark does not render clips. The product playback rule remains 20 s (10 s per side); this artifact evaluates cue timing only.
