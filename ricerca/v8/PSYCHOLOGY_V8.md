# PSYCHOLOGY_V8 — Musical expectation, tension, segmentation, groove and affect

Additional primary-literature review for the MusicMix transition selector.
Date of review: 2026-10-19. Author: research_psychology subagent (no code changed).

Companion to `ricerca/MOTION_SUSPENSE_RICERCA.md`. That document already covers
Farbood 2012, Barchet/TenseMusic 2024, Lehne & Koelsch 2015, You 2021, Bigand &
Parncutt 1999, Lerdahl & Krumhansl 2007, Herremans & Chew 2016, Granot & Eitan
2011, Farbood & Upham 2013, Vande Veire 2018, Zehren 2022, Kim 2020, MERT, Huron
2006, Meyer 1956. **This file does not repeat them.** It adds 10 sources (9
research/empirical + 1 recent break-routine study already central to the user's
case) and turns each into an implementable audio-feature or listening-test
consequence.

Nothing here is a validated audio predictor of "resolution". Psychology papers
measure human perception, not mix compatibility. The recommendations are
engineering hypotheses to be tested on labels, never thresholds fitted on a
single qualitative holdout.

---

## 0. Scope, terminology and the core distinction

Three questions must stay separate (they are answered by different evidence):

1. **Is A rising/preparing an event?** (anticipation / tension / suspense)
2. **Does B develop or transform that arc?** (continuation, groove/entrainment)
3. **Does B close it, abandon it, or deliberately contrast it?** (closure /
   interruption / intentional contrast)

A pure RMS fall, chroma similarity, or beat alignment can suggest *energy*
change. None of the papers below turns those into a **closure** decision. Event
boundaries are signalled by *transient prediction error* (Kurby & Zacks;
Sridharan; Burunat), not by "B is quieter" or "B is tonally near".

**Score/symbolic vs polyphonic-audio** (explicit, per paper):
- **Symbolic/score-only or controlled-MIDI**: Cheung 2024 (chord
  progressions), Mas-Herrero & Marco-Pallarés 2025 (melodies + IDyOM), Zhang
  2022 (score-derived phrases), Sioros 2014 (MIDI melodies).
- **Polyphonic-audio model**: Gu et al. 2026 (DEAM audio → valence/arousal).
- **Neuro/behavioural, not audio predictors**: Burunat 2024, Sridharan 2007,
  Kurby & Zacks 2008 (theory), Etani 2024 (review).
- **Behavioural with musical audio, no published model**: Turrell et al. 2025
  (EDM break routines).

The three existing tools (Beat This, Demucs, Whisper) are the *measurement*
layer. The papers below decide *which measurements are psychologically
meaningful*, and where a measurement must stay `unknown`.

---

## 1. Cheung, Harrison, Koelsch, Pearce, Friederici & Meyer (2024)
**Cognitive and sensory expectations independently shape musical expectancy and pleasure.**

- Journal: *Philosophical Transactions of the Royal Society B* 379(1895):
  20220420 (issue dated 2024; published online 2023-12-18).
- DOI: https://doi.org/10.1098/rstb.2022.0420
- Full text read: https://pmc.ncbi.nlm.nih.gov/articles/PMC10725761 (open, CC BY)
- **Read depth:** full HTML article; introduction, computational-model
  descriptions (SD, PP, TE, IDyOM), methods, validity and the main model-
  comparison results. Late results/tables read partially (tool output truncated).
- **Design / subjects:** Experiment 1 = 25 adults (13 musicians, 12
  non-musicians), continuous surprise ratings; Experiment 2 = 39 adults,
  continuous pleasantness. Stimuli: 30 isochronous chord progressions
  (30–38 chords each; 1009 chords analysed) sampled from **739 US Billboard
  pop songs** (McGill Billboard corpus); chords synthesised, transposed to C,
  with a background rhythm.
- **Results:** Sensory and cognitive expectations make **independent,
  non-overlapping** contributions to expectancy and pleasantness. Cognitive
  (IDyOM/tonal-hierarchy) expectations explained **more than twice** the
  variance in perceived surprise than sensory models. Arousal correlated with
  surprise; valence effect inconclusive.
- **Causal vs correlational:** correlational (Bayesian multilevel regression of
  ratings against model simulations). No intervention.
- **Limitations:** symbolic chord abstractions, not audio; Western tonal pop;
  no tension-to-resolution or transition labels; IDyOM trained on 1958–1991
  Billboard; small samples; chord background rhythm is artificial.
- **Implementable consequence:** a "surprise"/novelty term is legitimate as a
  *component* of tension, but it must come from a context model, not from raw
  spectral distance alone. For our mixed audio this means: treat tonal/novelty
  evidence as **low-confidence/`unknown`** unless key/chroma confidence is high;
  never rename energy into surprise. Keep sensory (timbre/spectral-flux) and
  cognitive (chroma-key/tonal-context) novelty as **separate diagnostics** in
  the trajectory profile so their independent contributions can be audited.

## 2. Turrell, Halpern, Bannister, Chai-Wi-Ting & Javadi (2025)
**Building the Anticipation: How variation in tension mediates emotions in music.**

- Journal: *Music Perception* 42(3): 256–268 (Feb 2025; online 2024-12-19).
- DOI: https://doi.org/10.1525/mp.2024.aa004
- Access used: abstract + repository metadata at
  https://kar.kent.ac.uk/106400/ (author-accepted manuscript listed; the
  full-text PDF itself was not opened in this pass, and the publisher version
  is paywalled).
- **Read depth:** abstract + item metadata only (truthful: full text not read).
- **Design / subjects:** 34 participants listened to EDM **break routines**
  (build-up → highly expected drop), continuously rating felt tension;
  peak-pleasurable emotions measured once per routine in valence–arousal space.
- **Results:** Tension **increased during build-up** and **decreased at the
  drop** when expectations were fulfilled. Average tension correlated with
  valence/arousal/emotional strength, **but** participants reported continued
  average tension after the break rather than a peak-pleasure signature — i.e.
  greater tension is not automatically greater pleasure.
- **Causal vs correlational:** correlational within-subject continuous ratings;
  no experimental manipulation of the drop.
- **Limitations:** EDM-specific; abstract-level evidence here; no audio model;
  does not label "resolved vs abandoned"; break routines are a stylised form.
- **Implementable consequence:** this is the closest published analogue of the
  user's "rising tension must resolve into the drop" intuition. It justifies a
  **build-up → metric arrival** closure template only where a build-up is
  *observable* (rising multifeature trend) and arrival is on a plausible
  downbeat/phrase. It does **not** justify treating a quiet entrance as a drop,
  and it does not transfer to rap entry automatically — rap can resolve through
  vocal phrase/groove instead of an EDM drop.

## 3. Burunat, Levitin & Toiviainen (2024)
**Breaking (musical) boundaries by investigating brain dynamics of event segmentation during real-life music-listening.**

- Journal: *PNAS* 121(36): e2319459121.
- DOI: https://doi.org/10.1073/pnas.2319459121
- Full text read: https://pmc.ncbi.nlm.nih.gov/articles/PMC11388323 (CC BY)
- **Read depth:** full HTML; significance, introduction, results (GLM, ICA,
  Granger), discussion, musicianship. Tail truncated but core read.
- **Design / subjects:** 36 listeners (18 musicians, 18 non-musicians); fMRI
  during listening to **three pieces of different genres**; fine-grained phrase
  boundaries from a real-time segmentation task (kernel-density agreement).
  RMS was regressed out of brain data (r = 0.05 with boundary regressor).
- **Results:** an **Early Auditory Network** peaks *before* the boundary
  (anticipation, posterior auditory, right-biased); a **Boundary Transition
  Network** engages during and after the boundary (middle/anterior auditory +
  Rolandic operculum), with fronto-parietal **deactivation**. Granger causality:
  early network → transition network. Musicianship changed *network
  integration*, not the basic sequence.
- **Causal vs correlational:** Granger "causality" (directed prediction) and
  fMRI correlation; not behavioural causal manipulation.
- **Limitations:** no audio feature model, no tension/closure labels, Western
  repertoire only, fMRI 2-s resolution; authors explicitly caution against
  over-generalising beyond the studied styles.
- **Implementable consequence:** a **pre-boundary anticipation window** and a
  **post-boundary integration window** should be measured as distinct states.
  Operationally: keep `A_before` (pre-junction, rising?) separate from
  `A_natural_after`/`B_after` (new event model, integration), and never infer
  closure from the boundary alone. This backs the existing A_before /
  A_natural_after / B_after design in `MOTION_SUSPENSE_RICERCA.md`.

## 4. Zhang, Sun, Wu & Yang (2022)
**Tension experience induced by tonal and melodic shift at music phrase boundaries.**

- Journal: *Scientific Reports* 12: 8304.
- DOI: https://doi.org/10.1038/s41598-022-11949-4
- Full text read: https://pmc.ncbi.nlm.nih.gov/articles/PMC9117266 (CC BY)
- **Read depth:** full HTML; abstract, introduction, behavioural results, ERP/
  ERSP results, discussion, limitations, methods.
- **Design / subjects:** 35 professional musicians (31 in EEG after exclusions),
  continuous felt-tension ratings + EEG while listening to two-phrase
  sequences: **repetition (RP)** vs **tonal shift (TS)** vs **melodic shift
  (MS)**, real classical-repertoire phrases.
- **Results:** TS and MS both produced **higher tension** than RP in the second
  phrase, with TS highest. TS → P300 + LPC, increased beta/gamma (tonal-shift
  detection + integration). MS → ERAN + LPC, decreased theta / increased alpha
  (attentional + working-memory load). Repetition (a strong prolongation)
  produced the **lowest** tension. This pattern is consistent with weak
  prolongation / prediction violation in event structure.
- **Causal vs correlational:** experimental manipulation of phrase relationship
  (causal for the tension-rating difference), EEG markers correlational.
- **Limitations:** expert musicians only, tonal Western music, score-derived
  controlled phrases, no audio model, small n; author-acknowledged first-phrase
  confound.
- **Implementable consequence:** **repetition is the strongest closure-like
  cue** (lowest tension) and **shift is a tension/continuation cue**, not
  closure. This is a direct warning against reading "chroma similar between A
  and B" as resolution: repetition/tonal-nearness lowers tension only when it is
  *structural repetition* inside a readable phrase. For audio, the testable
  proxy is phrase-level motif/role return, not a single chroma cosine.

## 5. Sridharan, Levitin, Chafe, Berger & Menon (2007)
**Neural dynamics of event segmentation in music: ventral and dorsal networks.**

- Journal: *Neuron* 55(3): 521–532.
- DOI: https://doi.org/10.1016/j.neuron.2007.07.003
- Access: https://pubmed.ncbi.nlm.nih.gov/17678862 and open author PDF
  https://cns.iisc.ac.in/sridhar/pi/pdfs/sridharan_etal_neuron_2007.pdf
- **Read depth:** abstract + partial open PDF (introduction, results fragments,
  figures/discussion references). Not every page verified.
- **Design / subjects:** musically untrained listeners, fMRI, entire symphonic
  works; 10-s windows around **movement transitions** (coarse boundaries);
  one transition per window; heart-rate variability also measured.
- **Results:** movement transitions produced peak activity **when there was no
  physical stimulus** (silence between movements). A right-lateralised
  **ventral fronto-temporal** network (salient-event detection) was followed in
  time by a **dorsal fronto-parietal** network (attention/WM update), with
  directed ventral→dorsal influence. RR-variability increased at transitions.
- **Causal vs correlational:** model-dependent/model-free fMRI + Granger-style
  directed analysis; not a behavioural intervention.
- **Limitations:** coarse boundaries (whole movements), one style (English
  Baroque), untrained listeners only, semiclassical acoustic setting, no audio
  features.
- **Implementable consequence:** **silence/void is a boundary event, not a
  closure guarantee**. A gap between sections can be an *event boundary* that
  opens a new model. Therefore a quiet B entrance is ambiguous by default and
  should require independent closure evidence before any `closure_candidate`
  label. Supports keeping `interruption`/`unknown` reachable in quiet
  entrances.

## 6. Kurby & Zacks (2008)
**Segmentation in the perception and memory of events (Event Segmentation Theory).**

- Journal: *Trends in Cognitive Sciences* 12(2): 72–79.
- DOI: https://doi.org/10.1016/j.tics.2007.11.004
- Full text read: https://pmc.ncbi.nlm.nih.gov/articles/PMC2263140 (author MS)
- **Read depth:** full HTML read (including Boxes 1–4, Figure 2 on musical
  movements, references).
- **Design:** theoretical review of primary behavioural, eye-tracking, pupil,
  fMRI and EEG work; includes Sridharan's music study as evidence.
- **Results (theory):** event boundaries arise as a **side effect of prediction
  error** when an "event model" (what-is-happening-now) is updated. Boundaries
  are detected at multiple timescales and are hierarchical; boundaries relate
  to working-memory updating and stronger long-term encoding. Segmentation
  "feeds" comprehension and adaptation; it is largely automatic (pre-task
  reading-time, pupil and neurophysiological measures).
- **Causal vs correlational:** review; the underlying claim is a mechanistic
  theory (prediction-error → model update), supported but not proven causally.
- **Limitations:** general cognition, not music-specific; no audio features, no
  crossfade/transition labels.
- **Implementable consequence:** gives the principled definition for our
  boundary/npvelty term: **event boundary = transient prediction-error /
  representation change**, measurable as a peak in beat-synchronous novelty and
  role/timbre change relative to a stable recent model. It is **not** a
  resolution. Provides the theoretical licence for a **hierarchical,
  multi-scale boundary read** (bar / 4-bar / phrase) rather than a single
  onset-based boundary.

## 7. Etani, Miura, Kawase, Fujii, Keller, Vuust & Kudo (2024)
**A review of psychological and neuroscientific research on musical groove.**

- Journal: *Neuroscience & Biobehavioral Reviews* 158: 105522.
- DOI: https://doi.org/10.1016/j.neubiorev.2023.105522 (open access)
- Access: https://pubmed.ncbi.nlm.nih.gov/38141692 and publisher HTML.
- **Read depth:** abstract, highlights and multiple full-text fragments
  (definition of groove, syncopation, sensorimotor prediction, reward,
  individual differences) from the open-access article. Review, not a single new
  experiment.
- **Results:** groove = pleasurable urge to move; strongest with **moderate
  syncopation** (inverted-U between metrical regularity and syncopation);
  motor/reward networks and temporal prediction are engaged. Groove is shaped
  by tempo, beat salience, event density, microtiming, harmonic complexity and
  low-frequency content; genre matters. Predictive-coding accounts explain why
  moderate precision-weighted prediction error is rewarding; open questions
  include the causal direction between urge-to-move and pleasure.
- **Causal vs correlational:** synthesis of correlational/experimental
  literature; no audio model.
- **Limitations:** review-level evidence; groove ≠ tension ≠ resolution; much
  evidence from controlled rhythms/drum patterns.
- **Implementable consequence:** gives a **continuation/entrainment** signal
  that is legitimate for rap: if B maintains a groove-consistent pulse and a
  *moderate* syncopation/event-density profile, its entrance can continue the
  motor arc even without tonal closure. A "flat" entrance reads as low
  entrainment, not as resolution. This is the evidence for treating rap as
  capable of continuation/resolution via rhythm, not as intrinsically bad.

## 8. Sioros, Miron, Davies, Gouyon & Madison (2014)
**Syncopation creates the sensation of groove in synthesised music examples.**

- Journal: *Frontiers in Psychology* 5: 1036.
- DOI: https://doi.org/10.3389/fpsyg.2014.01036
- Full text read: https://pmc.ncbi.nlm.nih.gov/articles/PMC4165312 (CC BY)
- **Read depth:** full HTML; abstract, introduction, both experiments
  (participants, stimuli, transforms, ratings, results), discussion.
- **Design / subjects:** Experiment 1 = 28 participants (within-subject, 12
  melodies × 5 transforms); Experiment 2 = 22 participants (6 transforms incl.
  density controls). Controlled **MIDI piano melodies**, syncopation generated
  algorithmically by shifting notes from strong to weak metrical positions.
- **Results:** moderate syncopation produced **significantly higher groove**
  than deadpan or maximum syncopation (inverted-U), and the effect was specific
  to syncopation (density-only transforms had weaker effects). Strong effect
  sizes for the sparse syncopation transforms (d ≈ 0.44–0.60).
- **Causal vs correlational:** causal experimental manipulation of the stimuli.
- **Limitations:** synthesised monophonic melodies, forced entrainment-free
  rating, no real polyphonic mix, groove ≠ resolution, no tension/closure
  labels, non-representative samples (mostly students/professionals).
- **Implementable consequence:** syncopation and event density are
  **audio-computable** (from Beat This beats + onset/velocity streams) and can
  serve as an **entrainment/continuation** feature. Because of the inverted-U,
  do not reward "more syncopation" monotonically; score **distance from a
  moderate reference profile**, with explicit `unknown` when beats are weak.

## 9. Mas-Herrero & Marco-Pallarés (2025)
**Predictive processes shape individual musical preferences.**

- Journal: *PNAS* 122(29): e2500494122.
- DOI: https://doi.org/10.1073/pnas.2500494122; PMID 40663615; PMC12304940.
- Access used: Europe PMC record (publisher page returned HTTP 403).
- **Read depth:** full abstract + metadata (via Europe PMC); body not read.
- **Design / subjects:** two **large independent samples**; a novel
  decision-making task in which participants rated preferences for
  **AI-generated melodies** varying orthogonally in surprise (information
  content) and uncertainty (entropy), scored with **IDyOM**. Computational
  models of the predictability/uncertainty interaction then predicted
  participants' genre preferences and pleasure for real compositions.
- **Results:** **inverted-U** between predictability and preference; **interaction
  of entropy and surprise**: small surprises preferred in low-entropy melodies,
  larger surprises in high-entropy music; the model predicted real-music
  pleasure and genre taste. Directionality of the entropy×IC interaction was
  **opposite** to some earlier IDyOM studies.
- **Causal vs correlational:** experimental manipulation of generated melodies
  plus correlational prediction of real-music preference.
- **Limitations:** symbolic melodies (IDyOM), simplified stimuli; populations
  not reported here in detail; pleasure/preference ≠ transition resolution;
  individual differences large.
- **Implementable consequence:** **there is no single optimum of surprise or
  novelty**. Support for "one rising curve must be continued" is
  listener- and context-dependent. Concretely: candidate selection across songs
  should **not** use a single novelty target; report novelty/entropy context
  and let the arc test be direction/continuation based. Also rationale for
  keeping individual-difference labels (optional) and for not threshold-fitting
  on one case.

## 10. Gu, Shao, Li & Fan (2026)
**Interpretable deep learning reveals distinct spectral and temporal drivers of perceived musical emotion.**

- Journal: *Scientific Reports* (published 2026-01-05).
- DOI: https://doi.org/10.1038/s41598-025-34238-2; PMID 41491208; PMC12859068.
- Access used: Europe PMC abstract + partial PMC article (intro/results fragments).
- **Read depth:** full abstract + metadata; article fragments read; not a
  page-by-page reading.
- **Design:** theory-guided hybrid CNN+Transformer trained on **DEAM**
  (**1,802 polyphonic music excerpts** with continuous valence–arousal ratings);
  interpretability via attention; ablation of theory constraints.
- **Results:** model reaches **CCC 0.67 (valence)** and **0.73 (arousal)**.
  Arousal is driven by **rhythmic regularity + spectral flux**; valence by
  **harmonic complexity + musical mode**. Theory-guided constraints improved
  performance.
- **Causal vs correlational:** supervised predictive modelling of human ratings;
  attention attributions are correlational, not causal.
- **Limitations:** DEAM repertoire/style skew, continuous self-report labels,
  not transition-specific, no closure/resolution label; audio model but not a
  mix-compatibility model.
- **Implementable consequence:** confirms that **arousal (energy-like) and
  valence (stability/tension-like) are distinct acoustic axes**. This supports
  the project's separation of `energy_*` from `closure_evidence`: spectral flux
  and rhythmic regularity feed *arousal only*, while harmonic complexity/mode
  feed *valence*. Do not collapse them into one "intensity" score. If a frozen
  audio model is added later, this is a concrete baseline target (CCC), not a
  claim of resolution prediction.

---

## Cross-paper synthesis

### What is supported
- Event boundaries are **prediction-error / model-update** events, hierarchical
  and multi-scale (Kurby & Zacks; Sridharan; Burunat). They are not closures.
- **Anticipation before the boundary** and **integration after it** are
  separable states (Burunat), matching the A_before / B_after split.
- **Repetition lowers tension** and tonal/melodic **shift raises tension**
  (Zhang); so chroma/tonal nearness is a *tension* cue, never proof of closure.
- **Moderate syncopation + event density** drive groove/entrainment
  (Sioros; Etani). Rap entrances can continue a motor arc through groove.
- **No universal predictability optimum** (Mas-Herrero); surprise is
  context-dependent and listener-dependent.
- **Arousal ≠ valence** acoustically (Gu); energy-like features must not be
  renamed tension or closure.

### What is NOT supported (echoes prior V7 finding)
- A mere RMS fall, a chroma cosine, or beat alignment does not establish
  resolution.
- A quiet/silent entrance is an ambiguous **event boundary**, not an arrival
  (Sridharan).
- No single threshold on novelty/surprise selects good candidates across songs
  (Mas-Herrero).
- None of these papers is a polyphonic-audio transition predictor; only Gu is a
  polyphonic-audio model at all, and only for valence/arousal.

### Symbolic/score-only vs polyphonic-audio (explicit)
| Source | Input type | Usable directly on mixed audio? |
|---|---|---|
| Cheung 2024 | Symbolic chords from pop corpus | No; needs key/chroma model |
| Mas-Herrero 2025 | Symbolic melodies + IDyOM | No |
| Zhang 2022 | Score-derived phrases | No |
| Sioros 2014 | MIDI melodies | No (but syncopation metric is transferable) |
| Turrell 2025 | EDM audio, ratings only | No model; template only |
| Burunat 2024 / Sridharan 2007 / Kurby 2008 | fMRI/theory | No; design guidance |
| Etani 2024 | Review | No; feature guidance |
| Gu 2026 | Polyphonic audio | Yes, but valence/arousal only |

---

## Prioritized, CPU-feasible recommendations (Beat This + Demucs + Whisper)

Framed as candidate selection **across the 4-song pool**
(Reminder → Too Many Nights, 5% TINT, Creature feat. Swae Lee), not as cues for
one fixed pair. No ranking weights changed; all items are diagnostics/audits.

**P1 — Separate arousal from valence/closure in the exported profile.**
Keep `energy_*` (log-RMS, flux, crest) strictly as arousal evidence (Gu). Add a
`harmony_complexity` / `mode` diagnostic from Demucs harmonic stems + chroma as
*valence* context only. Never let an energy drop enter `closure_evidence`.

**P2 — Measure anticipation vs integration as distinct windows (Burunat).**
For every candidate junction, export `A_before` (pre-junction trend + boundary
anticipation peak) and `B_after` (post-boundary integration). Closure requires
independent evidence in `B_after`, not just a change at the boundary.

**P3 — Entrainment/continuation from groove features (Sioros; Etani).**
From Beat This beats + onset/velocity: compute syncopation distance-from-moderate
and event-density continuity across the junction per beat-window. Use as
`continuation_evidence` support for rap entrances; keep `unknown` when beat
confidence is low. This is the concrete mechanism by which a rap entrance can
continue or resolve instead of being penalised.

**P4 — Closure via repetition/arrival, not chroma nearness (Zhang; Turrell).**
Test for *structural* return (motif/role/density return, metrical arrival) in
`B_after`, or a build-up→arrival when a build-up is observable. A chroma cosine
alone must stay non-decisive. Keep `interruption` and `unknown` reachable.

**P5 — Cross-song candidate matrix, not one-pair fitting.**
For each ordered pair (A→B) in the pool, export the three evidence objects and
the P1–P4 diagnostics with strict source hash/rate binding and explicit
`missing`. Rank candidates using direction/continuation and closure evidence;
never calibrate a threshold on Reminder→Too Many Nights. Treat that pair as a
qualitative holdout.

**P6 — Listening-test protocol (no human validation claimed here).**
Render randomised, loudness-matched transitions with more than one candidate per
cue and more than one song; hide artist names. Collect (a) pre: none/rising/
suspense; (b) post: continues / resolves / interrupts / intentional contrast /
unknown; (c) energy–arousal as a *separate* question. Split by song/artist.
Report macro-F1/balanced accuracy for the label set or pairwise preference/
Recall@K for ranking. This is the only route to any performance claim; none is
made now.

**P7 — Optional advanced ablation (after labels exist).**
A frozen polyphonic audio embedding or the Gu-style valence/arousal model may be
added as an ablation with song/artist-grouped splits and separate pop/rap/
instrumental tests. Gu's CCC (0.67/0.73) is a benchmark target for the emotion
axis only, not evidence about transition resolution. Verify any weight licence
before distribution and do not download weights in this task.

---

## Truthfulness and limitations of this review

- Read depths differ and are stated per paper; two entries (Turrell,
  Mas-Herrero) rely on abstracts/metadata, and Sridharan/Gu on partial text.
- All sources are Western-music or controlled-stimulus studies; none provides
  DJ-transition ground truth or a causal test that "B resolves A".
- Groove (Etani) and event segmentation (Kurby & Zacks) are reviews/theory;
  treat them as design guidance, not measurements.
- No human listening validation, no benchmark, and no SOTA performance is
  claimed. The recommendations are falsifiable engineering hypotheses to be
  evaluated on independent labels.
