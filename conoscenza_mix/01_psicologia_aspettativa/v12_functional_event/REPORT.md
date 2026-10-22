# Functional-event comparator: implementation and pilot report

Date: 2026-10-20  
Version: `functional-event-v1-event-centric`  
Transfer status: `unvalidated_engineering_comparator`

## What changed

- Replaced “two matching domains means continuation” with weighted domain agreement, an explicit salient-reversal veto, and a required local `B_before -> B_after` event contrast.
- Added persistence/stabilization checks and a separate preparation/pickup path. A prepared B that does not realize its entry event becomes `preparation_not_realized` with a negative adjustment.
- Added abstention: missing profiles, conflicting entry/continuation evidence, or absent independent vocal/section evidence produce `unknown`/no score instead of a forced choice.
- Kept `confidence=None`; no threshold is presented as a calibrated probability.
- Added `FunctionalEventScorer.score/rerank/rerank_pool` for root use. Pool rows retain the baseline and expose `event_adjusted_score`, `event_label`, `event_abstained`, `event_weight`, `event_missing`.
- Section labels are boundary/change context only. `chorus` is not mapped to resolution, a word gap is not phrase completion, and no track-identity or waveform-identity reward is used.

## Commands and observed output

```powershell
$env:PYTHONPATH='src'; python -m pytest tests\test_functional_event.py -q
```

Observed: `10 passed in 0.39s`.

```powershell
$env:PYTHONPATH='src'; python tools\run_functional_event_pilot.py
```

Observed:

```json
{
  "event": "functional_event_pilot_complete",
  "results": "outputs\\pilot\\results.json",
  "actual_labels": {
    "B_actual": "pickup_or_preparation",
    "B_event_muted": "preparation_not_realized",
    "B_event_transplanted": "preparation_not_realized"
  },
  "actual_scores": {
    "B_actual": 0.38,
    "B_event_muted": -0.29956027667078877,
    "B_event_transplanted": -0.1793799281835562
  },
  "entry_domains": {
    "B_actual": ["bass"],
    "B_event_muted": ["bass"],
    "B_event_transplanted": ["timbre"]
  },
  "pool_rows": ["B_actual", "B_event_muted", "B_event_transplanted"]
}
```

```powershell
python -m compileall -q src\musicmix\functional_event.py tools\run_functional_event_pilot.py tests\test_functional_event.py
```

Observed: exit code 0, no output.

```powershell
python research\collect_research.py
```

Observed:

```json
{"event": "research_archived", "sources": 6, "bytes": 858603}
```

## Pilot design and measurements

Sources: local `inputs/real_01.wav` (A) and `inputs/real_02.wav` (B), each staged from sample 3,360,000 at 48 kHz for 20 s. Cue at 10 s; listening clips are 20 s with 10 s per side. SHA-256 and source clocks are in `outputs/pilot/results.json`.

Beats are **observed spectral-flux onsets** from the existing `musicmix.analysis.analyze_track` detector (77 for A, 76 for B), not a validated beat grid. `tempo_used_as_beat_grid` is `false`; `PooledTrajectory` consumes only those observed beats.

Actual local cases:

| Case | Label | Event adjustment | Entry evidence | Missing |
|---|---|---:|---|---|
| B actual | `pickup_or_preparation` | `+0.38` | bass supported; timbre negative | stabilized event, section, vocal completion, pickup |
| First 0.42 s muted (gain 0.25) | `preparation_not_realized` | `-0.29956027667078877` | bass positive; energy, events, timbre negative | stabilized event, section, vocal completion, pickup |
| First event replaced by a real later B onset | `preparation_not_realized` | `-0.1793799281835562` | timbre positive; bass negative | stabilized event, section, vocal completion, pickup |

The muted/transplanted variants keep `B_before` and the sample timeline unchanged; they are diagnostic perturbations, not new candidates. Audio files are `outputs/pilot/audio/*_20s.wav`.

Synthetic logic controls (no audio/listening claim):

| Control | Label | Event score |
|---|---|---:|
| attack stabilized | `post_entry_attack_stabilized` | `+0.9211105175107714` |
| two matching domains, two salient reversals | `conflict_entry_vs_continuation` | `null` (abstained) |
| same endpoint slope, no entry bump | `post_entry_attack_transient` | `-0.40723460156859614` |
| same endpoint slope, entry bump | `post_entry_attack_stabilized` | `+0.7010960284334358` |

The same-slope pair isolates local entry evidence from a two-slope comparison. The reversal case verifies that two matching domains no longer override strong contrary evidence.

Thresholds are versioned engineering hypotheses, not measurements: trend `0.18`, flat `0.08`, event `0.30`, persistence retention `0.50`, `min_domains=2`, agreement floor `0.55`, contradiction weight `0.30`, attack head `2` beats, preparation lead `3` beats, stabilization tail `3` beats.

## Research archive

`research/sources.json` contains byte counts, SHA-256, access time, license/status and read depth. `research/originals/` holds downloaded bytes; `research/texts/` holds extraction.

| ID | Source URL | Bytes | Depth |
|---|---|---:|---|
| Hansen 2021 | `https://www.marcus-pearce.com/assets/papers/HansenEtAl2021.pdf` | 356,156 | full text |
| Lehne 2015 | `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4324075/fullTextXML` | 142,908 | full text |
| Kurby 2008 | `https://pmc.ncbi.nlm.nih.gov/articles/PMC2263140/?report=classic` | 189,412 | full text |
| Adams 2020 | `https://mtosmt.org/issues/mto.20.26.2/mto.20.26.2.adams.html` | 159,246 | full text |
| Sridharan 2007 | `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:17678862&format=json&resultType=core` | 5,576 | abstract/metadata only |
| Farbood 2012 | `https://api.crossref.org/works/10.1525/mp.2012.29.4.387` | 5,305 | abstract/metadata only |

Total archived: 858,603 bytes. Kurby is an NIH author manuscript served through PMC; the collector retries the anti-bot interstitial and verifies the saved text is not a challenge page. Sridharan and Farbood remain abstract-only because publisher full text is paywalled.

Theory support used: event segmentation as transient prediction-error/boundary updating (Kurby & Zacks); multidimensional, future-oriented tension rather than a single slope (Lehne & Koelsch; Farbood abstract); hip-hop phrase direction/closure where a word gap or final rhyme alone is insufficient (Adams). Measured here: acoustic descriptors and comparator sensitivity on staged PCM. Not measured here: listener-perceived preparation/arrival/resolution, validated beat or phrase boundaries, vocal completion, or transfer to a real pool.

## Root integration

```python
from musicmix.functional_event import FunctionalEventScorer

scorer = FunctionalEventScorer(weight=0.2)
rows = scorer.rerank_pool(rows, profile_lookup)
```

`outputs/pilot/pool_rerank_contract_demo.json` shows the row shape. The baseline score is preserved; abstained candidates keep `event_score=None` and do not receive an adjustment.

## Remaining blockers and limits

- No human listening judgments or user ratings for these transitions.
- No validated beat/downbeat grid, separated vocals, or functional-section provider; vocal and section evidence remain `unknown`.
- Only two 20-second local clips and diagnostic edits, not a real candidate pool.
- Thresholds and labels are unvalidated and must not be described as SOTA, 10/10, or musical resolution.
- Sridharan 2007 and Farbood 2012 are abstract-only in this archive.
