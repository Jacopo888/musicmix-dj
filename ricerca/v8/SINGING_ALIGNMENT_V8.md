# SINGING_ALIGNMENT_V8 — sung/rapped word cuts: evidence and implementation plan

**Status:** research recommendation only. No source code, models, or audio were changed in this run.  
**Date:** 2026-10-19  
**Scope:** the unresolved case where a Demucs-separated vocal is transcribed by faster-whisper `small.en`, the existing `word-guard-v1` is applied, and the user still hears words cut at a transition.  
**Decision owner:** root. This document does not choose or install a production provider.

## 1. Recommendation in one page

1. **Do not treat Whisper/faster-whisper token probability as word-boundary confidence.** It is a decoding posterior for a token, not a calibrated probability that a sung word starts or ends at a particular sample.
2. **Do not broaden the current guard into “ASR gap = silence.”** A missing word, a repeated chorus, or a bad transcript can create an apparent gap inside a real sung word. The current project already treats missing ASR as unknown; preserve that rule.
3. **First fix the decision rule, not the model:** make whole-word preservation a hard constraint. If both sides of a transition cannot be placed outside a verified whole-word envelope plus a guard, reject or move the transition to a structural boundary. Never trim the word to make the candidate fit.
4. **Then trial a singing-specific forced aligner on short, trusted windows.** SOFA is the first learned candidate worth benchmarking: MIT code, singing-oriented, official README claims better singing performance/faster inference than MFA, an English community checkpoint exists, and ONNX export exists. Its English checkpoint is a community model of 406,079,728 bytes; code and model licences/provenance are not equivalent, and CPU behaviour on this laptop is unmeasured.
5. **Keep STARS as the stronger but heavier second experiment.** It is singing-specific and MIT-licensed code, with Chinese and Chinese+English checkpoints, but its documented runtime is Python 3.10 / PyTorch 2.4 / CUDA 12.8 and its public evaluation is not an English rap benchmark.
6. **Do not make WhisperX/wav2vec2 the primary fix.** WhisperX is valuable for speech word timestamps, but its English default aligner is a speech-trained wav2vec2 character CTC model; no primary study retrieved here validates it on singing or rap. Rousso et al. found MFA beat WhisperX and MMS on manually aligned speech even when only correctly recognised words were compared.
7. **Add a mismatch/collapse gate before trusting any lyric alignment.** Ariga et al. show that forced alignment collapses when the supplied lyrics are not what is actually sung (Japanese karaoke parodies), and that an edit-distance-based collapse metric detects this with AUC 0.963. That is evidence for detecting mismatch, not evidence that arbitrary repeated ASR lyrics can be aligned safely.
8. **Measure and report uncertainty; do not invent it.** Until a safety model is calibrated on local annotated transitions, emit `supported` / `unsafe` / `unknown` with reason codes. Do not emit a fabricated “boundary confidence” derived only from ASR probability.

## 2. Why the present system can still cut words

These are plausible, testable failure modes; they are not a claim that one has already been proven by listening.

| Failure mode | Consequence | Test |
|---|---|---|
| ASR word timestamps under- or over-shoot held vowels, consonant releases, melisma, or rap onsets | The “gap” after a word is still inside the performed word | Annotate word envelopes; shift ASR spans by ±20/50/100 ms and inspect accepted cuts |
| Low ASR probability is filtered, but the *span itself* is wrong | A high-probability token can still have an inaccurate boundary; a low-probability word can be real | Compare ASR span to manual span and to an independent aligner |
| Repeated choruses, omitted words, paraphrases, or wrong-order ASR text | A global or long-window alignment assigns a word to the wrong occurrence | Counterfactual tests with repeated/corrupted lyric hypotheses |
| The guard trusts an internal gap between two ASR spans | If either span is truncated, the gap can lie inside a word | Inject a deliberately truncated span; ensure abstention |
| Crossfade is confused with a cut point | A cue can be “outside” the word while the fade still overlaps it | Enforce the fade-interval formulas in §5, not just cue-point formulas |
| Demucs bleed/removal changes the apparent onset/offset | Acoustic support is measured on a different signal from the reference | Impulse/offset tests and stem-vs-mix envelope comparison |
| No suitable whole-word gap exists near the musical boundary | The current system may choose a nearby gap that is not actually safe | Reject or change candidate; do not force a boundary |

The current `word_guard.py` already has the right conservative instinct: it merges overlapping spans, uses the minimum confidence in an overlap cluster, and raises `NoSafeBoundaryError` when no qualifying internal gap exists. Its weak point is epistemic: it still trusts the supplied spans as the geometry of the words. A guard cannot repair a word span that is already wrong.

## 3. What the primary evidence supports

| Source | What was verified | Relevance and limit |
|---|---|---|
| [SOFA repository](https://github.com/qiuqiao/SOFA) and [LICENSE](https://github.com/qiuqiao/SOFA/blob/main/LICENSE) | MIT code; Python 3.8 environment; text `.lab` required; default dictionary is `opencpop-extension` (Chinese); `--mode force|match`; `--save_confidence`; TextGrid/HTK/CSV outputs; `match` selects a contiguous subsequence of the supplied phoneme sequence | Singing-oriented and practical to trial, but it needs a text/phoneme hypothesis. It does not discover lyrics or fix arbitrary ASR text. The default dictionary is not English. |
| [SOFA English model discussion #31](https://github.com/qiuqiao/SOFA/discussions/31) and [release `v1.0.0_en`](https://github.com/spicytigermeat/SOFA-Models/releases/tag/v1.0.0_en) | Community checkpoint `tgm_en_v100.ckpt`, 406,079,728 bytes; ARPABET + `tr`, `dr`, `ax`, `dx`; trained by `spicytigermeat`; author reports public and commercial use approved; known issues include post-vocalic `[r]`, limited high/femme-range data, and sensitivity to inexact dictionary phonemes | Author approval is not a formal model licence or a dataset licence. Verify before adopting. The author’s “trained on SOFA v1.0.3” note is provenance, not an independent quality evaluation. |
| [SOFA `infer.py`](https://github.com/qiuqiao/SOFA/blob/main/infer.py) | After inference, SOFA applies AP detection and `post_processing` before export | Raw model output is not what downstream receives; compare both if doing research. |
| [SOFA `post_processing.py`](https://github.com/qiuqiao/SOFA/blob/main/modules/utils/post_processing.py) | `MIN_SP_LENGTH = 0.1`, `SP_MERGE_LENGTH = 0.3`; fills small leading/trailing gaps and merges small AP/SP gaps | A post-processed SOFA gap is not an independently measured silence. Do not use those gaps as ground-truth word boundaries. |
| [STARS paper](https://arxiv.org/abs/2507.06670) | Unified singing transcription, phoneme/word alignment, note transcription, technique/style annotation; claims superior performance over existing annotation approaches; phoneme-audio and word-level outputs | Paper is 2025 and singing-specific, but the evidence is not an English rap validation. |
| [STARS repository](https://github.com/gwx314/STARS) | MIT code; official pretrained checkpoints include Chinese and Chinese+English; README tested with Python 3.10, PyTorch 2.4.0, CUDA 12.8; pure-vocal input recommended; metadata needs words/phonemes/`ph2words` | Strong singing candidate, but heavier integration, GPU-oriented instructions, and checkpoint licences/sizes were not verified here. It still needs the correct word/phoneme sequence as input. |
| [WhisperX paper](https://arxiv.org/abs/2303.00747) / [repository](https://github.com/m-bain/whisperX) | BSD-2-Clause code; speech word-level timestamps via VAD and forced phoneme alignment; English examples use wav2vec2 alignment | Speech benchmark and speech-trained aligner. Not evidence of singing/rap boundary validity. |
| [Rousso et al., Interspeech 2024](https://arxiv.org/abs/2406.19363) | MFA outperformed WhisperX and MMS on manually aligned TIMIT and Buckeye, even when comparisons were restricted to correctly recognised words | Directly warns against assuming ASR-derived boundaries are precise, but it is speech-only, not singing. |
| [Ariga et al., EUSIPCO 2023](https://doi.org/10.23919/EUSIPCO58844.2023.10290010) / [PDF](https://eurasip.org/Proceedings/Eusipco/Eusipco2023/pdfs/0000286.pdf) | Karaoke parody detection by measuring forced-alignment collapse; Japanese singing dataset; edit-distance-based metric; AUC 0.963 | Supports a mismatch/collapse gate before trusting an alignment. Read at abstract/metadata level here; do not overstate the exact metric or its transfer to English rap. |
| [Kelley 2025](https://arxiv.org/abs/2506.01256) | Ten-model ensembles; median boundary plus a 97.85% order-statistic confidence interval; gradient boundaries; TIMIT/Buckeye; intended to flag boundaries for review | Good methodology for *measuring* boundary uncertainty, but ensembling speech aligners is not a turnkey singing fix. |
| [Cheng, Nakano & Goto, DAFx 2025](https://www.dafx.de/paper-archive/2025/DAFx25_paper_15.pdf) | Lyrics-to-audio alignment with frame-wise phoneme labels derived from word annotations and masked cross-entropy; reported Jamendo MAE 0.216 s, median absolute error 0.041 s | Singing/lyrics-relevant, but the evaluation is not this project’s four songs and the reported mean error is large relative to a 20 ms guard. Read at abstract/archive level. |

### What is not established

- No retrieved primary source shows that a speech-trained CTC/wav2vec2 aligner is reliable on separated sung vocals generally, or on rap specifically.
- No retrieved source solves “ASR produced an incorrect or repeated lyric sequence” by alignment alone.
- No public checkpoint’s performance on these four English songs is known.
- SOFA and STARS CPU runtime and peak memory on this Windows/16 GB laptop were not measured.
- The SOFA English checkpoint has author permission but no formal model licence text verified.
- No listening test was run. This document makes no claim about what the user will hear.

## 4. Failure model and policy

### 4.1 Evidence hierarchy

Use the strongest available evidence, in this order:

1. **Manual word-boundary annotation** for the target phrase (or explicit user-provided corrected text plus verified boundaries).
2. **Independent acoustic support** from the separated vocal: a stable onset/offset envelope around the full performed word, not merely low RMS in an ASR gap.
3. **Agreement between two independent alignment methods** on the same short window (for example, SOFA and a second singing-capable aligner, or SOFA and a conservative acoustic onset estimate), with the disagreement interval stored.
4. **A single uncalibrated ASR/alignment span** — usable only as a proposal, never sufficient to authorise a cut.
5. **Missing evidence** — `unknown`; fail closed.

A low RMS interval is never by itself proof of a word boundary. A high ASR probability is never by itself proof of a boundary.

### 4.2 Hard rules

For every accepted transition:

- The full outgoing word, including release/decay, must end before the A fade begins, minus a guard.
- The full incoming word must start after the B fade ends, plus a guard.
- If no such whole-word placement exists within the allowed shift, the candidate is rejected or moved to a verified structural boundary. The word must not be shortened, faded through, or split.
- If the lyric hypothesis is missing, wrong, repeated, or unverified, the word-derived boundary is `unknown`; use a structural boundary or abstain.
- If the phrase is dense rap/rapid syllables or a melisma crosses the proposed gap, treat the gap as unsafe unless independently verified.
- Any transition whose safety depends on post-processed SOFA SP/AP gaps is `unknown`, not `supported`.
- The renderer must record whether the fade interval overlaps any known or uncertain word envelope. Overlap with a known word is a hard failure.

### 4.3 Reason codes

Use stable reasons so counterfactuals are auditable:

```text
word_envelope_verified
word_envelope_unknown
word_envelope_conflict
no_safe_internal_gap
gap_inside_untrusted_word
lyrics_hypothesis_missing
lyrics_hypothesis_repeated
lyrics_alignment_collapse
alignment_disagreement_exceeds_guard
asr_probability_not_boundary_confidence
fade_overlaps_word
fade_inside_guard
sample_identity_mismatch
timebase_ambiguous
manual_annotation_used
```

## 5. Identity, time, offset, and sample-mapping contract

### 5.1 Canonical timeline

- Canonical rate: **48,000 samples/s**.
- Persist intervals as **half-open integer sample intervals** `[start_sample, end_sample)`.
- `0 <= start_sample < end_sample <= duration_samples`.
- Store `duration_samples` and `duration_seconds = duration_samples / 48000`; never infer duration only from a provider timestamp.
- Store the SHA-256 of the exact canonical audio file used for alignment, plus the source SHA-256 and the stem SHA-256.
- Do not mix timestamps from different decodes in one document. If a stem was resampled or padded, record its mapping.

### 5.2 Current project path (for audit)

`src/musicmix/vocal_boundaries.py` currently:

- resamples Demucs output to 48 kHz, then trims or zero-pads to the source duration;
- reports “no intentional offset”;
- converts seconds with Python `round(seconds * sample_rate)` (banker's rounding), clamps to track bounds, and drops empty intervals.

Therefore a new adapter must not silently describe existing timestamps as “exact” or “offset-free.” It must record:

```json
{
  "timebase": {
    "canonical_sample_rate": 48000,
    "unit": "sample",
    "interval": "half-open",
    "source_time_origin": "canonical_file_start",
    "seconds_to_sample_rule": "python_round_half_even",
    "offset_samples": 0,
    "offset_provenance": "demucs_audiophile_resample_then_trim_or_pad; no intentional offset"
  }
}
```

If a future adapter uses a different rule, use a new `seconds_to_sample_rule` value (for example `fraction_round_half_up`) and do not compare its output to old samples without recording the rule. A one-sample difference is physically small but must remain deterministic and auditable.

### 5.3 Provider-to-canonical mapping

For each provider or alignment window, persist:

```json
{
  "provider": "sofa",
  "model_id": "tgm_en_v100",
  "model_sha256": "...",
  "window_id": "w_0001",
  "window_start_sample": 120000,
  "window_end_sample": 480000,
  "provider_input_sha256": "...",
  "provider_input_sample_rate": 48000,
  "provider_output_time_origin": "window_start",
  "provider_offset_samples": 0,
  "conversion_rule": "provider_samples_plus_window_start",
  "canonical_offset_samples": 120000
}
```

Rules:

- If a provider returns times relative to its input window, add the window start **once**.
- Do not resample timestamps; resample audio only. If the provider internally resamples, its times are on its own clock; convert them with the declared rule.
- For a stem that begins before/after the canonical track, persist `stem_start_offset_samples` and subtract/add it explicitly. Do not assume zero.
- Validate with an impulse at several known canonical samples after resampling, trim/pad, and stem separation. An impulse test is the contract test, not a listening impression.
- Crossfade semantics in the current preview: A is retained through `cue_out_sample` and B begins at `cue_in_sample`; the final `fade_samples` of A overlap the first `fade_samples` of B. Thus the formulas below concern the fade intervals, not merely the cue points.

### 5.4 Whole-word placement formulas

Let:

- `fade` = rendered fade in samples, on the 48 kHz timeline;
- `guard` = additional safety guard in samples;
- `word_a` = verified full envelope of the last outgoing word;
- `word_b` = verified full envelope of the first incoming word.

A-side acceptance:

```text
word_a.end_sample + guard <= cue_out_sample - fade
```

B-side acceptance:

```text
cue_in_sample + fade + guard <= word_b.start_sample
```

If either inequality fails, the candidate cannot claim whole-word preservation. Move the cue only within the allowed shift window; otherwise reject or use a verified structural boundary. Do not reduce `fade` below its configured minimum merely to make an unsafe candidate pass; if a shorter fade is a deliberate policy, evaluate it as a separate policy and record it.

For a hard cut (`fade = 0`), the same formulas apply with `fade = 0`; but the current neutral preview uses a non-zero fade by default, so the fade interval is the relevant one.

### 5.5 Alignment record

Proposed record (documentation only; no schema change was made here):

```json
{
  "word_id": "w_0007",
  "text": "word",
  "start_sample": 240000,
  "end_sample": 252000,
  "envelope": {
    "start_sample": 239520,
    "end_sample": 253440,
    "method": "manual_plus_acoustic_release",
    "guard_samples": 960
  },
  "phonemes": [
    {"phone": "W", "start_sample": 240000, "end_sample": 243840, "posterior": null}
  ],
  "alignment": {
    "method": "sofa",
    "model_sha256": "...",
    "window_id": "w_0001",
    "sequence_position": 7,
    "text_source": "asr_window_hypothesis"
  },
  "uncertainty": {
    "method": "cross_method_disagreement",
    "lower_start_sample": 239000,
    "upper_start_sample": 240100,
    "lower_end_sample": 251800,
    "upper_end_sample": 253900,
    "coverage_target": null,
    "calibration_status": "uncalibrated"
  },
  "evidence": {
    "asr_word_probability": 0.91,
    "asr_probability_semantics": "token decode posterior; not boundary confidence",
    "acoustic_envelope_support": "supported",
    "cross_method_agreement": "unknown",
    "manual_verified": false
  },
  "safety": {
    "status": "unknown",
    "reasons": ["cross_method_agreement_unknown"]
  }
}
```

## 6. Priority implementation recipe

### P0 — Make whole-word preservation a hard constraint (no new model)

This is the highest-value change and does not require a download.

1. Keep the current `word-guard-v1` as a **post-gate**, not as the authority on word geometry.
2. Before calling it, attach provenance to every word span: text source (`asr`, `manual`, `catalog`), window ID, model hash, and whether the span was independently verified.
3. Build a conservative **word envelope**:
   - start at the earlier of the ASR/manual start and a locally detected acoustic onset;
   - end at the later of the ASR/manual end and a locally detected release/decay end;
   - never shrink the envelope to fit a cue;
   - if onset or release is ambiguous or conflicts with the ASR span, mark `unknown`.
4. Apply the formulas in §5.4. If both sides cannot be placed, reject the candidate or move to a verified downbeat/phrase boundary. Do not “make it fit” by shortening a word.
5. Add a test that deliberately truncates one ASR word span by 50–150 ms. The expected result is `unknown`/`NoSafeBoundaryError`, not an accepted cut inside the word.
6. Add a test where the vocal stem is time-shifted by a known number of samples; the expected result is that the same physical word is found on the canonical timeline, or the record fails with `sample_identity_mismatch`.
7. Add a test where ASR inserts a false gap inside a sustained note. If the gap cannot be independently supported, the transition must abstain.
8. Record every decision and reason code in the alignment document; do not collapse `unknown` into `supported` for rendering convenience.

**Why first:** it addresses the observed symptom without assuming that a new model is correct, and it prevents the system from confidently cutting a word because the transcript geometry is wrong.

### P1 — Trial SOFA on short, trusted windows (root approval required)

SOFA is the best first learned experiment because it is singing-oriented and its code is MIT. It is not a lyric discovery system.

1. Obtain root approval for the 406,079,728-byte English community checkpoint and for its provenance/licence review; do not download it in this research task.
2. Create short windows (target: one phrase or roughly 3–10 s; tune only on a validation song) around the candidate transition. Long full-song windows increase the chance of matching the wrong chorus occurrence.
3. Supply each window with only its **verified or window-local** text hypothesis. Prefer manual/user-corrected text; otherwise use ASR text for that window, never a guessed full-song lyric.
4. Convert text to ARPABET using the checkpoint’s dictionary. Do not use the default Chinese `opencpop-extension` dictionary for English.
5. Run SOFA in `force` mode when the full window text is trusted; use `match` only for a genuinely partial contiguous hypothesis. `match` cannot repair an arbitrary repeated or reordered full lyric sequence.
6. Export word/phone intervals and, if using it, `confidence.csv`. Label those values as provider posteriors, not calibrated boundary probabilities.
7. Cross-check SOFA against:
   - the ASR/manual word envelope;
   - a second independent window or alignment run with a small time perturbation;
   - the acoustic release/onset evidence.
8. Accept a word boundary only when the uncertainty interval fits inside the available guard on both sides. If SOFA and the envelope disagree by more than the guard, mark `alignment_disagreement_exceeds_guard` and fail closed.
9. Compute an alignment-collapse diagnostic per window (guided by Ariga et al.): unusual AP/SP proportion, irregular path/phone durations, low frame support, or a high edit-distance-like mismatch signal. A high collapse score means the lyric hypothesis is suspect; do not “repair” it by moving words.
10. Benchmark CPU wall time, peak memory, and failure modes on one short phrase before considering all four songs. The README documents a GPU-oriented ONNX path; CPU feasibility is not established.

**SOFA-specific cautions:**

- The official post-processing fills gaps shorter than 0.1 s and merges AP/SP gaps shorter than 0.3 s. Do not use the edited gaps as silence evidence.
- The community English model author notes dictionary/phoneme exactness matters and reports known `[r]` and range limitations.
- The model was trained on a small set of listed voice providers; do not generalize to all singers or rap.
- The code is MIT, but the checkpoint’s formal licence is not established by the repository’s MIT file. Author permission is not a licence review.

### P2 — STARS only after SOFA or if GPU capacity appears

Use STARS when the project can afford a GPU-oriented stack and wants a singing-specific model with note/technique outputs. It expects explicit word/phoneme sequences and `ph2words`; it therefore inherits the same “wrong or repeated lyrics” problem. The bilingual checkpoint is the relevant one for English, but English-only accuracy and checkpoint size/licence still need verification.

Do not make STARS the first move on a Windows CPU/16 GB machine without a measured short-window benchmark.

### P3 — Add measured uncertainty, not a confidence cosmetic

After a candidate provider works, add one of these conservative uncertainty mechanisms:

- **Perturbation spread:** rerun the same short window with ±20/50/100 ms context shifts and slightly different ASR hypotheses; measure the spread of word boundaries. If the spread plus guard does not fit, abstain.
- **Two-method disagreement:** compare SOFA with a second singing-capable aligner or a verified acoustic envelope; store the interval, not just a point.
- **Calibrated safety model:** only if local manual labels exist. Fit a simple model (for example logistic regression) on features such as provider posterior, cross-method spread, acoustic support, word duration, and collapse score. Report calibration (reliability/ECE) and held-out coverage. Until then, `calibration_status` must be `uncalibrated`.

Kelley’s ensemble confidence intervals are a methodological precedent, but they were evaluated on speech (TIMIT/Buckeye). Do not present them as validated singing confidence.

## 7. Independent evaluation and counterfactual tests

### 7.1 Ground truth

- Use the four project songs or a legally usable subset.
- Annotate the full word envelopes around each target transition, not just the cue. Include consonant releases and sustained vowels.
- Have two annotators label a subset independently; adjudicate disagreements and retain the disagreement interval.
- Blind the annotator to provider/system identity and to the proposed cue where possible.
- Store annotations on the canonical 48 kHz timeline with the same half-open convention.

### 7.2 Primary metrics

- **Word-cut violation rate:** accepted transitions where the fade interval overlaps a manually verified word envelope. Target for claimed-safe transitions: zero on the evaluation set.
- **Boundary offset:** signed `predicted - manual` at word start and end; report median, MAE, p95, and worst case in milliseconds.
- **Word inclusion:** whether the complete outgoing/incoming word is present outside the fade, not merely whether the cue is near it.
- **Abstention rate:** proportion of candidates marked `unknown` or rejected. Report it with the violation rate; a system that abstains everywhere is safe but useless.
- **Unsafe-pass rate:** transitions marked `supported` that later fail manual review. This is the most important error to drive to zero.
- **Coverage by song:** do not aggregate away a single catastrophic song.
- **Provider collapse rate:** proportion of windows rejected for lyric mismatch/collapse.

Do not use “ASR confidence” as the primary outcome. It is an input feature at best.

### 7.3 Counterfactual test matrix

| Test | Manipulation | Expected behavior |
|---|---|---|
| Timestamp jitter | Shift provider words by ±10/20/50/100 ms | Safety status degrades before an unsafe cut is accepted; no silent “still safe” at large shifts |
| Truncated word | Remove 50–150 ms from one ASR span | `gap_inside_untrusted_word` or `no_safe_internal_gap`; no cut inside the word |
| False internal gap | Insert a spurious gap inside a sustained word | Abstain unless independent acoustic support exists |
| Wrong lyric | Replace a window’s text with a paraphrase or wrong words | Collapse/mismatch gate rejects or marks unknown; no confident alignment |
| Repeated lyric | Duplicate a chorus or repeat a word sequence | Window-local alignment only; no global reuse; ambiguous occurrence rejects |
| Reordered lyric | Swap two lines/words | Collapse gate flags mismatch; fail closed |
| Missing lyric | No text for a window | `lyrics_hypothesis_missing`; structural fallback or abstain |
| Stem offset | Delay/advance the vocal stem by 1–1000 samples | Same physical word recovered on canonical timeline, or explicit identity failure |
| Sample-rate path | 44.1/48/96 kHz source and re-encode | Impulse lands at the expected canonical sample; no cumulative drift |
| Fade sweep | 20/100/500 ms fades at the same cue | Word-overlap check uses the fade interval; a safe cue under one policy is not assumed safe under another |
| High ASR probability / bad boundary | Keep a high token probability but move the span | No supported status from probability alone |
| No separation | Use the mix instead of the stem | Evidence degrades to unknown; no fabricated silence |
| Rap/dense syllables | Fast syllable sequence crossing the proposed gap | Conservative abstention unless verified; no assumption that singing-trained models transfer |

### 7.4 Acceptance gates for a provider trial

- Gate S0: zero verified word-cut violations among transitions it marks `supported` on the held-out evaluation set.
- Gate S1: no `unsupported` result is relabelled as safe by a downstream heuristic.
- Gate S2: report boundary MAE/p95 and abstention together; do not report a single “accuracy” number without the abstention denominator.
- Gate S3: reproduce all sample/time contracts with impulse tests before listening tests.
- Gate S4: only after S0–S3, run a blinded listening study for musical quality; keep cue selection and renderer quality as separate outcomes.

These gates are deliberately conservative. They measure the symptom the project can verify (word-boundary violations) rather than an unverified claim about perceived audio.

## 8. What to measure rather than fabricate

| Do measure | Do not report as |
|---|---|
| ASR token probability (with model/version) | “word boundary confidence” |
| Manual boundary error in ms | “alignment confidence 0.9” |
| Cross-method boundary spread in samples | “probability the cut is safe” |
| Acoustic envelope support status | “silence” from an ASR gap |
| Collapse score / mismatch flags | “the lyrics are correct” |
| Abstention and coverage | “the system handles all words” |
| Provider CPU wall time / peak RSS | “CPU feasible” without a benchmark |
| Calibration error/coverage on local labels | “calibrated” from a single threshold |

If a numeric confidence is ever exposed, it must have a named calibration set, a target coverage, and a held-out reliability report. Until then, use categorical safety plus evidence fields.

## 9. Open decisions for root

1. Approve or reject the SOFA English community checkpoint after a licence/provenance review; the code’s MIT licence does not automatically cover the weights.
2. Decide whether correct/manual lyric text is available for the four songs. Without it, SOFA is a constrained experiment, not a solution.
3. Decide whether to accept a structural fallback (downbeat/phrase boundary) when no whole-word boundary is safe. This is preferable to cutting a word but changes musical placement.
4. Decide the maximum allowed cue shift and minimum fade for the first experiment. Keep them fixed across compared systems.
5. Decide whether the first evaluation is annotation-only or includes a blinded listening study. Do not conflate the two.
6. If STARS is considered, confirm GPU availability and checkpoint licence/size before integration.

## 10. Bottom line

The current system’s most likely unresolved defect is not that `word-guard-v1` lacks a threshold; it is that the guard treats potentially wrong ASR spans as word geometry. The safe next step is to make whole-word preservation a hard, auditable constraint and to fail closed when the evidence is insufficient. SOFA is a reasonable first singing-specific provider trial on short verified windows, with its English checkpoint and post-processing treated cautiously. WhisperX/wav2vec2 should not be promoted to primary singing alignment on current evidence. No provider should be trusted until local word-envelope annotations and counterfactual tests show zero unsafe cuts.
