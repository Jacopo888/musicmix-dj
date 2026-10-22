# Validating a local automatic music-transition critic (V8)

Status: research and frozen experiment proposal. Date: 2026-10-19.
Author: research subagent. No audio was listened to by this agent. No model,
checkpoint, or audio was downloaded, installed, or executed. Existing
`docs/`, `ricerca/`, `src/`, and `tests/` were not modified.

## 1. Bottom line

The current centered-10 s Audiobox diagnostic is **not validated as a
transition critic**, and the three supplied outputs show why it must not be
used as one. In the supplied corruption, 5.48% of samples were hard-clipped
(peak 1.0, 105,168 clipped samples), yet the four Audiobox axes moved by only
0.014 to 0.156 points on their predicted scale. That is a descriptive result,
not a significance test: there is one item, no repeated inference, no
confidence interval, no human anchor, and no test-retest. Some corruption
deltas (CE, PC) are smaller than the ordinary policy-vs-baseline differences,
so none of them can be called a calibrated perceptual effect.

No publicly available, validated **sequential transition** critic was found in
the sources consulted. COCOLA is the strongest runnable local candidate for
harmonic/rhythmic **simultaneous stem compatibility**, but its input role is
wrong for a transition: it is trained on two stem subsets of the **same 5 s
window**, with other windows/tracks as negatives. Feeding a pre-join window and
a post-join window is out of domain. COCOLA is CPU-feasible (57 MB checkpoint,
MIT code), but should be a diagnostic arm in a frozen experiment, not a gate or
reward.

The defensible position now is:

1. Keep the deterministic technical checks as **technical defect detectors**
   only. Validate them against injected defects with known sample locations.
2. Run a frozen, paired, multi-arm counterfactual manifest. Treat clipping,
   clicks, dropouts as known technical negatives; treat level/DC steps as known
   signal deviations, not aesthetic defects; treat tempo/key mismatches as known
   construction mismatches whose perceived badness still needs humans; treat
   selected-vs-alternative cross-song pairs and same-song controls as musical
   alternatives with no automatic ground truth.
3. Require artist/song-disjoint held-out human pairwise ratings before any
   reranking. Never optimize against one critic; report Audiobox axes
   separately and never collapse them into a single "transition score".

## 2. What the supplied V7 outputs actually show

`observed_audiobox_v7.json` (SHA-256
`f11f64cf3ae13e42d4fb9c290884987cc7b1758d50f3526d4348505adb82c0f7`)
contains three outputs from the same centered 10 s, 16 kHz excerpt: the V7
policy render, a V7 baseline render, and a 2 s hard-clipped corruption of the
policy render.

| Arm | CE | CU | PC | PQ | Technical evidence |
|---|---:|---:|---:|---:|---|
| policy | 7.3115 | 7.6104 | 6.9959 | 7.9777 | no clipping; peak 0.8976 |
| clipped corruption | 7.1696 | 7.5051 | 6.9821 | 7.8219 | 105,168 clipped samples; 5.4775%; peak 1.0 |
| baseline | 7.1411 | 7.5675 | 6.9287 | 7.9138 | no clipping; peak 0.8976 |

Paired deltas, corrupted minus policy (header corrected during root review):

| Axis | Delta | Relative |
|---|---:|---:|
| CE | -0.1419 | -1.94% |
| CU | -0.1053 | -1.38% |
| PC | -0.0137 | -0.20% |
| PQ | -0.1557 | -1.95% |

Interpretation that the evidence supports:

- The corruption is unambiguously a digital technical defect by construction.
- The Audiobox axes remain in a narrow band and do not isolate the seam. A
  centered 10 s window contains 5 s on each side; the local defect is diluted,
  and the model was trained for per-clip aesthetics, not seam detection.
- The policy-vs-baseline differences (CE +0.170, CU +0.043, PC +0.067,
  PQ +0.064) are of the same order as the corruption effect, so the corruption
  delta cannot be interpreted as a clean "quality drop" without a paired design
  and uncertainty estimates.
- `musical_continuity="not_validated_for_transitions"` in the JSON is correct.
  The result does not license a threshold, a pass/fail rule, or a reranker.

What it does **not** show: that Audiobox is statistically insensitive, that
0.14 points is imperceptible, or that any axis has a known sign for a seam.
Those claims require repeated runs, multiple items, and human calibration.

## 3. Primary-source findings

Read depth is stated exactly. "Full HTML, core sections" means the paper body
was fetched and read around the relevant methods/results, not merely the
abstract. No listed model was executed.

| Source | Exact URL | Read depth | Relevant finding | Limit for this project |
|---|---|---|---|---|
| COCOLA paper (Ciranni et al., ICASSP 2025) | https://arxiv.org/abs/2404.16969 ; HTML: https://arxiv.org/html/2404.16969v4 | Full HTML, method, training, score, MOS section | Pairwise contrastive model of harmonic/rhythmic coherence for stems/submixes; positives are disjoint stem subsets **within the same 5 s window**; negatives can be same track in different windows. Authors report Pearson r = 0.54 (p < 0.01) between COCOLA score and MOS from 41 participants (0-4) for shifted/stretched chunks. | Simultaneous stem compatibility, not temporal succession; unnormalized score; one accompaniment-style MOS study; not a technical-defect detector. |
| COCOLA official repository and checkpoint | https://github.com/gladia-research-group/cocola ; https://drive.google.com/file/d/1S-_OvnDwNFLNZD5BmI1Ouck_prutRVWZ/view?usp=share_link | README, requirements, model code, LICENSE, checkpoint page metadata | Code MIT. Checkpoint `checkpoint-epoch=87-val_loss=0.00.ckpt`, 57 MB, Google Drive. EfficientNet-B0, 16 kHz mono, 5 s (80,000 samples), HPS two-channel input; batch/CPU inference in plain PyTorch. | Checkpoint was **not downloaded or run**. No separate checkpoint license found; only repository MIT. No latency measured. |
| AutoMashup (Delabaere et al., GRETSI 2025) | https://arxiv.org/abs/2508.06516 ; HTML: https://arxiv.org/html/2508.06516v1 ; code: https://github.com/ax-le/automashup | Full HTML, core sections; repo README and GitHub license API | Uses COCOLA for **simultaneous vocal-vs-instrumental** compatibility; compatibility is asymmetric; CLAP/MERT zero-shot embeddings did not reproduce COCOLA; code BSD-3-Clause. | Not a transition critic; no listening validation in this paper (future work); 21 tracks / 420 samples; the claimed COCOLA-perception link is cited, not independently established. |
| DJtransGAN (Chen et al., ICASSP 2022) | https://arxiv.org/abs/2110.06525 ; HTML: https://arxiv.org/html/2110.06525v2 ; code: https://github.com/ChenPaulYu/DJtransGAN | Full HTML, method and subjective evaluation; repo README, inference/download code, LICENSE | GAN that **generates** a transition from paired tracks with cue points; evaluation is a listening test (Preference, Smoothness, Melodious), not an automatic critic. Pre-trained generators 133 MB, MIT code. | Generator, not a validator; discriminator realism score is not a validated perceptual score; EDM-only training; licensing of the in-house training mixes is not reproducible. |
| Huang et al., AAAI 2021, stem compatibility | https://arxiv.org/abs/2103.14208 ; DOI: https://doi.org/10.1609/aaai.v35i1.16092 | Full HTML, data generation, negative conditions, objective/subjective sections | Trains compatibility from original vs **unmatched key and/or tempo** stems; matched key/tempo treated as unlabeled; near-perfect objective classification, then subjective listening. | No public code/weights found; in-house MSS; simultaneous mashup, not sequential transition; near-perfect scores are expected because the negative construction is gross. |
| Aligning TTM evaluation with human preferences / MusicPrefs (Huang et al., 2025) | https://arxiv.org/abs/2503.16669 ; HTML: https://arxiv.org/html/2503.16669 | Full HTML, synthetic meta-evaluation and preference sections | Validates metrics by **controlled monotonic degradations** and rank correlation against the induced ordering; FAD was weak, MAD stronger; human preference data are pairwise. | TTM corpus metric, not a local transition critic; reference-based; supports the validation *method*, not a transition score. |
| Rubric-Based Optimization for TTM (Wang et al., 2026) | https://arxiv.org/abs/2610.03589 ; HTML: https://arxiv.org/html/2610.03589v2 | Full HTML, abstract, experiments, limitations | Single-metric preference optimization improves the targeted evaluator while degrading independent ones on MusicGen-small; rubric rewards help several metrics but ALM rubrics transfer only partially to tempo/instrumentation and not measurably to key; clips are short, limiting transitions/endings. | Text-to-music post-training, not transition validation; confirms the danger of optimizing one critic and the need for independent axes. |

### 3.1 COCOLA verdict, explicitly

- **CPU availability:** plausible and runnable in principle. The released model is
  EfficientNet-B0 with a linear projection, 5 s of 16 kHz mono audio, and the
  requirements pin ordinary PyTorch/Lightning/librosa. The 57 MB checkpoint is
  small enough for CPU. This is an inspection verdict, not an execution result.
- **License:** repository code is MIT (verified in `LICENSE`). The Google Drive
  checkpoint page states no separate license. Treat checkpoint licensing as
  unresolved until the authors clarify; do not redistribute it.
- **Input role:** `x` and `y` are each `[1, 16000*5]` mono 5 s audio, or 5 s
  submixes. Training positives are disjoint stem subsets from the **same
  window**; negatives are windows from the same or other tracks. Thus the
  model answers "do these simultaneous parts cohere?", not "does B follow A?".
- **Transition use:** feeding the last 5 s of A and the first 5 s of B creates a
  pair that resembles a **negative** training pair (different windows), not a
  positive. It may still be useful as a harmonic/percussive diagnostic, but it
  is out of domain and must be validated, not assumed.
- **Score:** bilinear similarity, higher = more coherent, unnormalized and not a
  probability. Do not compare absolute values across datasets or call a raw
  difference a MOS difference. Use within-context paired deltas.

### 3.2 AutoMashup and DJtransGAN

AutoMashup is useful as a pipeline and as evidence that COCOLA can rank
simultaneous vocal/instrumental compatibility. It is not evidence for
sequential seams, and its own paper defers perceptual evaluation. DJtransGAN
can generate a transition, but its discriminator is trained to distinguish
generated from real DJ mixes; that is a realism classifier, not a validated
transition-quality critic. Its inputs are cue-point pairs at 44.1 kHz and its
training domain is EDM, so it is a poor default critic for a general local
mixer.

### 3.3 Preference/rubric and counterfactual evidence

Wang et al. show the core failure mode: optimizing a single automatic judge
moves that judge and can move independent judges in the opposite direction.
Their rubric prompt does mention abrupt cuts, clipping, and incoherent
transitions, but the experiments are text-to-music post-training and the
authors state that short clips limit evaluation of transitions. A rubric model
can therefore be a **diagnostic prompt**, not a validated reward.

MusicPrefs supplies the right validation template: construct controlled,
monotonic degradations, freeze the ordering, and measure rank correlation with
that ordering before trusting a metric. Their context-length manipulation is
the closest analogue here: shorter-context generation creates jarring
transitions, so the degradation order is known before any critic is scored.
Apply that logic to seams. Do not use the critic's own rankings as training
labels (self-feedback), and do not select the critic that wins on the same items
used to tune it.

## 4. Frozen experiment manifest V8.1

This manifest is deliberately small enough to render and score locally, and
large enough to estimate paired effects. It must be written to a JSON file and
hashed before any scoring. No condition may be added, removed, or retuned after
the first critic output is seen.

### 4.1 Unit construction

- Audio: 48 kHz stereo source, 20.000 s excerpt, join at exactly 10.000 s
  (480,000 samples). Ten seconds on each side match the current centered
  diagnostic. Technical metrics use the full 20 s file; Audiobox/COCOLA use a
  10 s centered excerpt resampled to 16 kHz mono.
- All cross-song joins use a fixed 12 ms equal-power crossfade at a beat-aligned
  join, unless the condition explicitly changes the join. The fade length and
  alignment algorithm are frozen in the manifest.
- Loudness: source arms are gain-staged to the same integrated loudness before
  corruption; no limiter. Record the gain and the pre-corruption peak. If a
  source needs > 0.5 dB correction, exclude it rather than normalize differently.
- Every item records: `item_id`, `context_id`, `arm`, `condition_family`,
  `source_a_sha256`, `source_b_sha256`, `samples_sha256`, `sample_rate`,
  `join_sample`, `fade_samples`, `corruption_recipe`, `seed`, `known_truth`,
  `requires_human`, `critic_versions`, and `code_sha256`.

### 4.2 Materials and disjointness

- 8 contexts, each with a distinct artist for A and a distinct artist for B:
  16 songs, 16 artists, no song reused as both A and B.
- For each A, two cross-song alternatives:
  - `B_sel`: the current project-selected continuation, if one exists.
  - `B_alt`: a different song, different artist, matched to the same tempo
    (within 2%) and compatible key (same key or ±1 semitone after transposition)
    so that the comparison is musical/style preference, not a confounded
    technical or rhythmic mismatch.
- 8 natural contiguous controls: one 20 s excerpt from inside each A, no edit,
  no join. This is a no-seam continuity sanity control, **not** a transition.
- 8 same-song splice controls: two contiguous 10 s halves of `B_sel` joined at an
  internal phrase boundary. This is a real join with high musical continuity by
  construction, but it is still an edit and must be labeled as a control, not as
  a human-preferred transition.
- Split: 4 contexts (8 artists) are `dev`; 4 contexts (8 artists) are `locked_test`.
  No artist, song, or source recording appears in both. The locked test is opened
  once, after critic versions and thresholds are frozen.
- Check and record whether any source song appears in COCOLA's training corpora
  (MUSDB18-HQ, MoisesDB, Slakh2100, CocoChorales). If membership is unknown,
  report it as possible leakage; do not silently claim disjointness from
  COCOLA training.

### 4.3 Condition arms

Apply the technical and musical ladders to the 8 `B_sel` cross-song units. Apply
the alternative and control arms to both `B_sel` and `B_alt` where indicated.

| ID | Arm / recipe | Known ground truth | Requires human? |
|---|---|---|---|
| C0 | `natural_contiguous`: 20 s inside one song, no edit | No seam; natural within-song continuity | No for "no seam"; yes for any aesthetic claim |
| C1 | `same_song_splice`: contiguous halves of one song, real hard/12 ms join | Real join; musical identity continuity by construction | Yes for preference; it is a control, not a "good transition" label |
| C2 | `clean_join_sel`: A to `B_sel`, beat-aligned, 12 ms crossfade | None; no technical defect by construction | Yes: musical validity |
| C3 | `clean_join_alt`: A to `B_alt`, same join protocol | None; style/alternative preference | Yes |
| C4 | `clip_ladder`: gain +6/+12/+18 dB over 2 s centered at join, hard clip | **Technical defect by construction** (samples exceed full scale) | No for "clipped"; audibility still not assumed |
| C5 | `click_ladder`: one-sample impulse at join, sample replacement amplitude 0.2/0.5/0.9 | **Technical discontinuity by construction** | No for "injected click"; perceptual salience not assumed |
| C6 | `dropout_ladder`: replace 5/20/50 ms around join with digital silence | **Technical discontinuity by construction** | No for "dropout"; artistic silence is a separate control |
| C7 | `level_step`: +3/+6/+12 dB on the post-join half | Known signal-level step, **not necessarily a defect** | Yes for aesthetic verdict |
| C8 | `dc_step`: +0.02/+0.05/+0.10 full-scale DC on post-join half | Known DC deviation, **not necessarily audible** | Yes for aesthetic verdict |
| C9 | `tempo_ladder`: time-stretch B by 2/4/8% before join | Known tempo mismatch by construction | Yes for perceived badness |
| C10 | `phase_ladder`: shift B by 1/8, 1/4, 1/2 beat at the join | Known beat-phase mismatch by construction | Yes for perceived badness |
| C11 | `key_ladder`: transpose B by +1/+2/+6 semitones | Known pitch/key change by construction | Yes: dissonance depends on key relation and context |
| C12 | `key_compatible`: transpose B by +7 semitones (perfect fifth) or relative major/minor | Known harmonic alternative, not a mismatch | Yes |
| C13 | `policy_vs_baseline`: the two real V7 renders, if the same context is available | None; real system comparison | Yes |

Confound control: C7 and C8 must be rendered with enough headroom that the
intended step does not also clip; C4 is the only arm allowed to clip. Verify
this with the technical detector and reject any item that acquires an
unintended second defect. C9-C11 must be checked for time-stretch/pitch-shift
artifacts and for clipping before they enter the manifest.

Important distinctions:

- **Known ground-truth technical negatives:** C4 clipping, C5 injected click,
  C6 dropout. These can be scored as detection targets without human labels.
- **Known signal deviations that are not automatically defects:** C7 level step,
  C8 DC step. They are measurable, but a level change or a DC offset can be an
  artistic or editing choice. Technical flags are observations.
- **Known construction mismatches:** C9 tempo, C10 phase, C11 key. The mismatch
  is known from the recipe; whether listeners perceive it as bad is not known.
  C12 is a harmonic alternative, not a mismatch.
- **No automatic ground truth:** C0, C1, C2, C3, C13. C0 and C1 are continuity
  controls; C2/C3 are musical alternatives; C13 is a real-system comparison.
  Only humans can supply preference labels for these.

### 4.4 Counts and freeze

- 16 cross-song clean units (8 `B_sel` + 8 `B_alt`) and 16 controls (8 natural,
  8 same-song splices).
- 8 ladder families (C4-C11) x 3 levels x 8 `B_sel` contexts = 192 items.
- 1 key-compatible arm x 8 contexts = 8 items.
- Total: 232 items. This is a runnable paired set, not a powered perceptual
  benchmark. If human time is tight, rate a pre-specified subset (16 clean + 16
  controls + 3 levels of C4/C9/C10/C11 on all 8 contexts = 128 items); do not
  thin the ladder levels after seeing critic scores.
- Freeze: write `manifest_v8_1.json`, its SHA-256, the critic code hash, model
  revision/checkpoint SHA-256, and the analysis script hash. Lock `locked_test`
  until the dev analysis and all thresholds are written down.

### 4.5 Outputs to record per item

- Deterministic technical report: peak, clipped sample count/fraction, seam jump
  and availability, DC offset, RMS change, silence fractions; plus the injected
  defect's known sample indices and amplitude.
- Audiobox: all four axes separately (CE, CU, PC, PQ), model revision, device,
  threads, excerpt offsets. Never report a single average axis as "transition
  quality".
- COCOLA (if run): BOTH, HARMONIC, PERCUSSIVE scores separately, on the last 5 s
  of A and first 5 s of B; checkpoint hash; no cross-dataset absolute comparison.
- Any other critic: versioned score, input role, and whether it saw simultaneous
  stems, sequential audio, or the rendered mix.

### 4.6 Pre-registered checks (write these before looking at scores)

These are project decision rules, not perceptual constants.

1. **Technical detector sanity (no humans needed).** On C4/C5/C6, the relevant
   deterministic detector must flag at least 90% of injected items, with a 95%
   Wilson lower bound above 0.75, and must flag at most 1 of 16 clean/natural
   controls. If it fails, fix the detector before using it as a gate. This
   validates detection, not aesthetics.
2. **Natural-control sanity.** For every critic claimed to measure continuity,
   the paired score for C0 (natural contiguous) should not be below C4/C5/C6
   on the same context in a majority of contexts. A critic that systematically
   prefers clipped/clicked audio over natural audio is invalid for this use.
   Report the proportion and a paired bootstrap interval; do not call a single
   item a failure.
3. **Monotonic sensitivity.** For each ladder (C4, C5, C6, C7, C8, C9, C10,
   C11), compute the within-context Spearman/Kendall correlation between score
   and the known corruption level, then average across contexts. A usable
   diagnostic should have a positive rank correlation with a 95% bootstrap
   interval excluding zero for at least the technical ladders it claims to
   detect. Report each ladder separately; do not pool them into one number.
4. **Discrimination, not raw MOS.** For clean-vs-corrupted and selected-vs-
   alternative pairs, report balanced accuracy, AUC, and Cliff's delta with
   paired bootstrap intervals. A raw mean difference of 0.1 or 0.2 points is
   not evidence of calibration and must not be reported as "significant".
5. **Specificity.** COCOLA and Audiobox should not be treated as technical
   defect detectors merely because they move on C4/C5/C6. If their movement on
   technical corruptions is comparable to their movement on musical mismatch,
   label them non-specific and keep the deterministic checks as the only gate.
6. **Human gate.** No reranking or hard acceptance until a locked, artist/song-
   disjoint human study shows pairwise accuracy above chance with a 95%
   interval lower bound above 0.5. A reasonable project target is 0.65 balanced
   accuracy, but that is a product decision, not a scientific threshold.
   Report per-family results (technical, rhythmic, harmonic, stylistic) and
   per-song outliers, not only the aggregate.
7. **No self-training.** Never train or tune on critic-generated labels. Never
   select the critic by the same items used to report its final accuracy. Keep
   an untouched test set and independent evaluators.

### 4.7 Future human protocol (artist/song-disjoint)

- Raters: at least 5 per item, blinded to arm and system; no one who designed
  the critic or the manifest. Include both musicians/DJs and general listeners,
  reported separately.
- Task: forced choice between two 20 s renders of the **same context**, with
  options "A better", "B better", "no difference", plus separate 1-5 ratings for
  technical cleanliness, rhythmic continuity, harmonic compatibility, and
  structural fit. Do not ask for a single "MOS".
- Pairs: clean vs each corrupted ladder level; `B_sel` vs `B_alt`; same context
  only. Randomize left/right and arm order. Include catch trials (C0 vs a
  clipped version) and attention checks.
- Agreement: report Krippendorff's alpha or intraclass correlation; if agreement
  is low, do not average it away. Report rater-level and item-level results.
- Analysis: paired preference models with context and rater as grouping factors;
  bootstrap over contexts and raters; report uncertainty. Treat ties explicitly.
- Disjointness: locked test artists/songs never appear in dev, threshold tuning,
  critic training, or prompt development. Check COCOLA training overlap as above.

## 5. What to do now, in order

1. **Do not change the selector.** Keep the existing ranking and manifest. Run
   the technical checks on every render and store the full report, including
   missing evidence.
2. **Build the frozen manifest** above and hash it. Render the 232 items. Verify
   that each corruption is actually present at the expected sample indices and
   that the clean controls have no clipping/click flags.
3. **Score the three supplied V7 outputs** as exploratory items only. Record the
   paired deltas, but do not fit a threshold and do not call the clipping delta
   significant. Use them to document the current diagnostic's failure mode.
4. **Optionally run COCOLA on CPU** on the 5 s pre/post windows, after obtaining
   the 57 MB checkpoint and confirming its license with the authors. Record
   HARMONIC, PERCUSSIVE, and BOTH separately. Treat it as an out-of-domain
   diagnostic, never as a gate. Do not run it on the 20 s file and do not
   average the two sides into a "transition score".
5. **Do not install or run DJtransGAN as a critic.** If a transition generator
   is needed later, its discriminator can be studied as a realism baseline, but
   it is not validated for this task and its training domain is EDM.
6. **Collect the human data** on dev first. Only after the protocol and analysis
   are frozen, open the locked test. Then decide whether any learned critic
   beats the technical + simple musical-feature baseline.

## 6. Explicit prohibitions

- Do not call a raw Audiobox axis difference a calibrated MOS difference or a
  significant effect.
- Do not collapse CE/CU/PC/PQ into one transition score. PC is complexity, not a
  quality objective; CU is usefulness; none is seam validity.
- Do not treat "no technical flags" as "musically valid".
- Do not use a critic's own rankings as training labels or as the sole selection
  criterion.
- Do not tune on the locked test, and do not report only aggregate results that
  hide per-song failures.
- Do not claim COCOLA, Audiobox, DJtransGAN, or a rubric model was validated
  here. None was executed or human-calibrated in this task.

## 7. Limitations of this note

- No listening, no inference, no checkpoint download, no audio rendering.
- The 57 MB COCOLA and 133 MB DJtransGAN sizes are from the public download
  pages; no checksum was computed because the files were not downloaded.
- COCOLA's checkpoint license is not separately stated on the linked page.
- The manifest's counts and thresholds are engineering decisions, not
  statistically powered perceptual constants. The locked test is small.
- The source review is targeted, not exhaustive. "No validated sequential
  transition critic found" means not found in these sources, not proven absent.
- The observed Audiobox deltas come from a single supplied run; they are
  descriptive evidence of dilution/non-specificity, not a formal sensitivity
  analysis.
