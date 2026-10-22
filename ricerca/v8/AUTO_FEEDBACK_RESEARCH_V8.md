# Automatic feedback on rendered music mixes: evidence review (V8)

Status: research note plus CPU-only implementation contract. Date: 2026-10-19.

## Scope and non-claims

The question is whether a renderer can automatically reject technically defective
transitions and, later, rerank candidates by musical continuity. This note separates
three claims that are often conflated:

1. **Technical defect detection** (clipping, discontinuity, DC offset, silence, level
   jump) can be computed from the rendered waveform with fixed rules. It is auditable
   and cheap. It is implemented in `src/musicmix/mix_quality.py`.
2. **Perceptual audio quality** is a learned or reference-based judgement. Speech
   metrics and global audio embeddings are not automatically valid for music seams.
3. **Musical transition validity** (does the second track belong after the first at
   this point?) requires a validated pairwise preference critic and held-out human
   ratings. No such critic is available in this workspace.

The agent that wrote this note has no audio input. Nothing below is a listening test,
a human validation, or a state-of-the-art claim. Read-depth is stated per source; a
full read means the paper's HTML/PDF body was read, not merely the abstract.

## Primary sources actually consulted

| Source | URL | Read depth | What it supports | Limits for this project |
|---|---|---|---|---|
| Kilgour et al., *Fr\'echet Audio Distance* (2018) | https://arxiv.org/abs/1812.08466 | Abstract plus indexed PDF evidence | Reference-free metric comparing distributions of audio embeddings; reported correlation 0.52 with human ratings versus 0.39 for SDR in their setting | Global distribution metric, not a local transition critic; embedding and reference-set dependent |
| Gui et al., *Adapting Fr\'echet Audio Distance for Generative Music Evaluation* (2023) | https://arxiv.org/abs/2311.01616 | Full HTML | FAD has sample-size bias, embedding/reference bias, and can hide per-song failures; proposes per-song FAD for outliers | Confirms that a single global distance is unsafe as a reranking objective |
| Cideron et al., *MusicRL* (2024) | https://arxiv.org/abs/2402.04229 | Full HTML, core sections | Human-preference reward models and RL for music generation; preference data and reward models can guide generation | Not a transition validator; preference model is tied to its data and generation task |
| Gao et al., *Scaling Laws for Reward Model Overoptimization* (2022) | https://arxiv.org/abs/2210.10760 | Full HTML | Proxy rewards are overoptimized by best-of-n and RL; held-out reward eventually degrades | Warns against optimizing any learned critic without independent evaluation |
| MuQ (2025) | https://arxiv.org/abs/2501.01108 | Full HTML | Self-supervised music representations transfer to MIR tasks | Representation quality on MIR does not establish transition or mix-quality validity |
| CLAP (2022) | https://arxiv.org/abs/2206.04769 | Abstract | Joint audio-text embedding for zero-shot audio classification/retrieval | Text-audio alignment is not a pairwise musical-continuation score |
| DNSMOS P.835 (2021) | https://arxiv.org/abs/2110.01763 | Full HTML | No-reference speech quality via P.835 dimensions; trained on speech/noise-suppressor data | Speech quality, not music mixing or transition validity |
| ViSQOL v3 (2020) | https://arxiv.org/abs/2004.09584 | Abstract | Full-reference perceptual audio quality for speech and audio | Requires a reference; a transition has no single reference rendering |
| CDPAM (2021) | https://arxiv.org/abs/2102.05109 | Abstract | Learned full-reference perceptual audio similarity trained on human triplet judgements | Speech perturbations and full-reference setting; not validated for music seams |
| Vinay & Lerch (2022) | https://arxiv.org/abs/2209.00130 | Full HTML | Objective generative-audio metrics are insufficient proxies for perceptual quality | Direct caution against reporting a single automatic score as quality |
| Tjandra et al., *Meta Audiobox Aesthetics* (2025) | https://arxiv.org/abs/2502.05139 | Full HTML plus official repository README | No-reference, per-item prediction of four axes (Production Quality, Production Complexity, Content Enjoyment, Content Usefulness); evaluated on speech, sound, and music; reports comparable or superior correlation to MOS on several sets | Per-clip aesthetic axes, not transition validity; 16 kHz mono, 10 s windows; subjective axes are not seam metrics; repository inference downloads a checkpoint |
| AudioMOS Challenge 2025 | https://arxiv.org/abs/2509.01336 | Full HTML | First challenge for automatic subjective quality prediction for synthetic audio; Track 1 TTM musical quality/alignment, Track 2 Audiobox four axes, Track 3 speech at multiple rates; 24 teams; system-level SRCC primary | Evaluates generated clips/systems, not transitions or pairwise continuation; track 2 spans TTS/TTA/TTM; no team ranked first on all axes |
| Wang et al., *Rubric-Based Optimization for Text-to-Music Generation* (2026) | https://arxiv.org/abs/2610.03589 | Full HTML (v2), core sections and appendices tables | Single-metric preference optimization improves its own metric while degrading others; rubric rewards can improve several metrics; ALM rubric transfers poorly to measurable attributes; human listening test on best run | Text-to-music post-training, not mix transitions; ratings share evaluator limitations; short clips limit long-range form/transitions/endings |
| Official Audiobox Aesthetics repository | https://github.com/facebookresearch/audiobox-aesthetics | README and usage/API | Public per-clip CLI/Python inference returning CE/CU/PC/PQ; evaluation data and checkpoint links; CC-BY-4.0 code | Automatic checkpoint download; not a transition model; no project-local benchmark |

Earlier project research in `ricerca/` was read for context (mashup compatibility,
transition selection, tension/suspense). Those qualitative cases are treated as
holdouts, not as data for fitting thresholds.

## Synthesis

### Technical checks are legitimate now

The implemented checks are deterministic signal measurements: sample peak, clipping
count at a documented threshold, seam jump relative to the median nearby sample
derivative, DC offset, near-silent seam edges, and pre/post RMS change. Each has a
unit and an availability status. Missing context at track edges is reported, not
imputed. In digital silence the relative jump is explicitly `undefined_in_silence`
because dividing by a near-zero derivative would manufacture certainty.

These measurements can reject a render that clips or contains an obvious click at the
join. They cannot accept a render: zero flags only means no configured artifact was
detected. The report therefore sets `status="technical_only"`,
`musical_continuity="unknown_without_validated_critic"`, `musical_quality=None`, and
`confidence=None`.

### Global embeddings are not local continuity

FAD compares distributions of embeddings and is useful for corpus-level enhancement
evaluation. Gui et al. show that aggregate FAD can hide per-item failures and is
sensitive to sample size and reference choice. MuQ and CLAP provide representations,
not a validated relation "track B follows track A musically at sample s". Using cosine
distance between adjacent windows as a transition score would be an unvalidated proxy.
It is therefore excluded from the current reranker.

### Speech metrics do not transfer by assertion

DNSMOS, ViSQOL, and CDPAM were built and validated around speech/noise suppression or
full-reference similarity. Music transitions involve harmonic, rhythmic, and structural
continuation; none of these papers validates that use. A speech metric may correlate
with gross artifacts, but it must not be presented as a music-transition critic.

### Learned critics need independent evaluation

MusicRL demonstrates that human-preference reward models can guide music generation.
Gao et al. show that optimizing a proxy reward leads to overoptimization: the proxy
rises while held-out human preference eventually falls. Wang et al. (2026) provide a
music-specific demonstration: preference optimization built from any single automatic
evaluator (CLAP, SongEval, or Audiobox-Aesthetics) raises that evaluator while lowering
at least one independent evaluator on MusicGen-small; the targeted metric improves and
the others deteriorate. A renderer that reranks with a learned critic will exploit its
errors unless candidates are evaluated on frozen, held-out human ratings and the
critic's training data is disjoint from the test set. The same caution applies to any
automatic score used for selection.

### Per-clip aesthetic predictors are not seam critics

Audiobox Aesthetics predicts four per-clip axes from 16 kHz mono audio using 10 s
windows: Production Quality, Production Complexity, Content Enjoyment, and Content
Usefulness. It is reference-free and per-item, which is closer to a usable critic than
a corpus-level FAD. However, none of the four axes is defined as transition validity,
and the model was trained on isolated clips. The official repository exposes a
`audio-aes` CLI and a Python predictor returning `{"CE", "CU", "PC", "PQ"}` per input;
using it would download a checkpoint, so it was not executed here.

If root later implements an optional Audiobox critic, the safe use is diagnostic:
score the pre-join and post-join windows separately and record the axis deltas, rather
than treating the average of the two windows as a transition score. Even then, the
paper reports only moderate correlations on music (for example, PQ vs. human PQ on
PAM-music, r = 0.587) and the predictor is not validated on rendered seams. A drop in
Production Quality across a seam is not automatically a defect; it may reflect an
intentional arrangement change.

AudioMOS Challenge 2025 confirms both the maturity and the limits of this family of
metrics. Track 1 predicted overall musical quality and textual alignment for
text-to-music; Track 2 used the four Audiobox axes across speech, sound, and music;
Track 3 covered speech at multiple sampling rates. Twenty-four teams participated and
improved on the baselines, but no team ranked first on all four Audiobox axes, and the
primary metric was system-level rank correlation. These are clip- or system-level
judgements, not pairwise transition preferences, and a good system-level ranker can
still misrank two individual renders of the same transition.

Wang et al. also show that ALM rubric rewards transfer only partially to measurable
attributes (tempo, key, instrumentation) and can fail outright for key. This supports a
division of labor: use direct measurements where the property is measurable, and do not
expect a general aesthetic model to resolve a specific musical relation. Their reported
limitations explicitly note that short generated clips limit evaluation of long-range
form, transitions, and endings — the exact object this project needs.

## Recommendation: CPU now, learned critic later

### Phase 1 (current, CPU-only, no model downloads)

- Run `evaluate_mix_quality` on every candidate render, binding `track_id`,
  `samples_sha256`, sample rate, frame count, join sample, config hash, and module
  version.
- Hard-reject only documented technical defects (clipping above threshold, empty
  track). Treat discontinuity, seam silence, DC offset, and level change as heuristic
  observations or descriptive evidence, not as aesthetic verdicts.
- Preserve the full candidate manifest and the report for every rejected and accepted
  candidate. Never overwrite the pre-rejection ranking.
- Use technical flags as tie-breakers only after the selector score, and record that
  policy in the manifest.
- Collect human pairwise ratings on a held-out set of rendered seams, including
  intentional artistic silences and level changes, before setting any new threshold.

### Phase 2 (only after data exists)

- Define the target as a pairwise preference: given the same transition context, which
  render is preferred by human raters, and on which dimension (technical cleanliness,
  rhythmic continuity, harmonic compatibility, structural fit).
- Train or adapt a music encoder (MuQ/CLAP are candidates, not established solutions)
  on that pairwise task with artist/song-disjoint splits.
- Evaluate with held-out raters and songs, report calibration and per-song outliers,
  and compare against the Phase 1 rule baseline.
- Keep an independent evaluation set that is never used for model selection; monitor
  for reward overoptimization when the critic is used to rerank. Wang et al. show that
  single-metric optimization can trade one evaluator against another, so require
  improvement on independent evaluators, not only on the reward source.
- If an Audiobox Aesthetics critic is added, use it as a diagnostic on pre- and
  post-join windows with all four axes recorded separately; never collapse it into a
  single transition score, and validate any threshold against held-out human pairwise
  ratings of seams. Do not use AudioMOS-style system-level rank correlation as evidence
  that individual seam rankings are correct.
- Do not deploy the critic as a scalar "musical quality" until it beats the rule
  baseline on held-out human preference and its failure modes are documented.

## Practical loop

```
candidates -> render -> mix_quality report
           -> reject only technical defects
           -> keep manifest + report
           -> selector ranking unchanged
           -> (later) validated pairwise critic rerank
           -> held-out human evaluation
```

A candidate rejected for clipping can be re-rendered with gain staging. A candidate
flagged for discontinuity can be inspected or re-rendered with a longer crossfade, but
the flag is not proof that the music is wrong. A candidate with no flags is not proof
that the transition is musically valid.

## Open limitations

- No human listening validation was performed here.
- Thresholds in `mix_quality.py` are project heuristics, not perceptual constants.
- The seam jump ratio uses sample derivatives; it is sensitive to high-frequency
  content and to the guard/context lengths recorded in the report.
- Silence, DC offset, and level changes can be intentional; flags are observations.
- FAD, MuQ, CLAP, DNSMOS, ViSQOL, CDPAM, Audiobox Aesthetics, and the AudioMOS
  challenge systems were not executed. Their inclusion is evidence review, not
  benchmark evidence for this pipeline.
- Audiobox Aesthetics was read from the paper and repository documentation; no
  checkpoint was downloaded and no inference was run. Its reported correlations are
  the authors' results on their datasets, not measurements on this project's renders.
- The AudioMOS summary paper reports challenge outcomes and participant techniques; it
  does not establish that any submitted model predicts transition validity.
- The source list is not exhaustive; it is the set actually consulted within the task
  budget.
