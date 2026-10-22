# MIR_SOTA_V8 — Modern MIR for automatic DJ transition selection/rendering

Compiled: 2026-10-19, agent `research_mir_sota` (swarm_v8).
Scope: primary publications and official code/model documentation that can
realistically affect MusicMix M02 (structure), M03 (vocal/phrase boundaries),
M08 (compatibility / semantic resolution) and M09 (render). No models were
downloaded and no packages installed. Network was used only to read public
papers/repos/model cards and their official file listings.

This document is deliberately conservative. Where I read only the abstract,
README, tables or the repo file listing, I say so. Benchmarks quoted from a
paper belong to that paper's dataset and task and are **not** evidence about
Reminder/The Weeknd, Too Many Nights/Metro Boomin, 5% TINT/Travis Scott or
Creature/Pop Smoke.

---

## 1. Read method and honesty statement

| Source | Depth actually read | Where |
|---|---|---|
| All-In-One (Kim & Nam, WASPAA 2023) | Abstract + full README + full repo file doc; paper body previously read integrally in `ricerca/VERIFICA_PAPER_IMPLEMENTAZIONE.md` §4 | arXiv HTML, GitHub README |
| SongFormer (Hao et al., 2025) | Abstract + full arXiv HTML intro/tables (v1/v2/v3) + full README + HuggingFace file listing | arXiv, GitHub, HF |
| MuQ (Zhu et al., 2025) | Abstract + README + method snippets (Mel-RVQ, MuLan) + HF file listing. Paper body **not** fully read | arXiv, GitHub, HF |
| MERT (Li et al., ICLR 2024) | Abstract + results tables + HF/API metadata. Paper body **not** fully read | arXiv, HF |
| Beat This! (Foscarin et al., ISMIR 2024) | Abstract + full README + checkpoint listing; paper body previously read in `ricerca/BEAT_THIS_IMPLEMENTAZIONE.md` | arXiv, GitHub |
| DJtransGAN (Chen et al., ICASSP 2022) | Abstract + significant HTML excerpts (DDSP fader/EQ, data pipeline, MSAF+madmom cue stage). Paper body **not** fully read | arXiv HTML |
| Cue points for DJ mixing (Zehren et al., 2020) | Abstract + introduction excerpt. Paper body **not** fully read | arXiv PDF |
| AutoMashup (Delabaere et al., GRETSI 2025) | Abstract + arXiv HTML body (Allin1 pipeline, COCOLA selection, O(n²) limitation). Paper body **not** fully read | arXiv, HAL |
| Spotify Playlist Sequencing & Transitions (Bittner et al., ISMIR 2017) | Body previously read integrally in `ricerca/VERIFICA_PAPER_IMPLEMENTAZIONE.md` §1 | ISMIR archive |
| MSAF (Nieto & Bello, ISMIR 2016) | Full README + documented algorithm list | GitHub |

I did **not** run any of these models. All footprint numbers come from official
hosting pages (HuggingFace file listings) or official READMEs, not from local
measurement.

---

## 2. Verified source table

Legend for "Runnable": `code+ckpt` = official inference code with published
weights; `code` = official code without published weights in the listing I saw;
`lib` = library/toolkit.

| # | Source | Task | Dataset / metric (paper) | Runnable | Code license | Weights | Footprint (official listing) |
|---|---|---|---|---|---|---|---|
| S1 | **All-In-One** Kim & Nam 2023 | beat, downbeat, section boundary + label | Harmonix Set 8-fold; beat F1 0.958, downbeat F1 0.915, boundary HR.5F 0.660, label PWF 0.738 | code+ckpt (`pip install allin1`) | MIT | bundled checkpoint (model cards) | needs PyTorch + **NATTEN** + Demucs + madmom; ~300k-param head, stems at 44.1 kHz |
| S2 | **SongFormer** Hao et al. 2025 | music structure boundaries + functional labels | SongFormBench-HarmonixSet ACC 0.807 / HR.5F 0.703; RWC-Pop ACC 0.814 | code+ckpt | CC-BY-4.0 (code) | HF `ASLP-lab/SongFormer` | `model.safetensors` = 2.76 GB + `SongFormer.pt` head 104 MB; **also requires MuQ + MusicFM SSL backbones**; expects 24 kHz; GPU recommended |
| S3 | **MuQ** Zhu et al. 2025 | SSL music embedding; MuQ-MuLan music-text | MIR downstream (tagging/classification); MuLan zero-shot MagnaTagATune SOTA | code+ckpt (`pip install muq`) | MIT (code) | HF `OpenMuQ/*` | MuQ-large ≈300M params, `model.safetensors` 1.33 GB; MuLan ≈700M; weights **CC-BY-NC-4.0**; needs 24 kHz, fp32 |
| S4 | **MERT** Li et al. ICLR 2024 | SSL acoustic music embedding | 14 MIR tasks; MERT-330M overall SOTA aggregate | code+ckpt | (repo) | HF `m-a-p/MERT-v1-330M` | ≈330M params; repo/API listing ~1.26 GB, storage ~6.5 GB with variants; weights **CC-BY-NC-4.0** |
| S5 | **Beat This!** Foscarin et al. ISMIR 2024 | beat + downbeat (no DBN) | beats/downbeats F1 SOTA; worse on continuity metrics | code+ckpt (`pip install beat-this`) | MIT | same page | `small0` ≈8.1 MB, `final0` ≈78 MB; CPU-capable; 22 kHz mono front-end; already integrated as an optional provider in this repo |
| S6 | **DJtransGAN** Chen et al. ICASSP 2022 | learned DJ transition (DDSP EQ+fader, GAN) | listening test vs baselines; first data-driven transition generator | code (repos exist) | repo-dependent | training ckpts not confirmed in listing | uses MSAF (CQT structure) + madmom (beat/key) for cue generation |
| S7 | **Cue points for DJ mixing** Zehren et al. 2020 | "switch point" detection for EDM | own annotated set; ~96% judged good | code (`MZehren/Automix`) | repo-dependent | — | EDM-focused; heuristic + evaluated against manual annotation |
| S8 | **AutoMashup** Delabaere et al. GRETSI 2025 | mashup creation + song-pair selection | COCOLA as compatibility; finds *asymmetry*; CLAP/MERT **fail** to reproduce COCOLA coherence | code (`ax-le/automashup`) | repo-dependent | uses Allin1/Demucs/COCOLA | O(n²) pairwise; explicit negative result for generic embeddings |
| S9 | **Spotify sequencing** Bittner et al. ISMIR 2017 | playlist sequencing + pairwise transition cue search | 48 pairs, 4 pro curators; 64.13% Good / 28.26% OK / 7.61% Bad | (product, no ckpt) | — | — | beat-synchronous feature matrices T/C/ℓ/v, standardized cost Λ, beat-aligned stretch |
| S10 | **MSAF** Nieto & Bello ISMIR 2016 | music structure analysis toolkit | ISMIR-2016 systematic exploration | lib (`pip install .`) | MIT | none (non-neural) | pure Python + librosa/scikit-learn; CPU-only; Foote novelty / checkerboard / spectral clustering |

Links (primary, exact):
- S1 paper https://arxiv.org/abs/2307.16425 · repo https://github.com/mir-aidj/all-in-one · license https://raw.githubusercontent.com/mir-aidj/all-in-one/main/LICENSE
- S2 paper https://arxiv.org/abs/2510.02797 · repo https://github.com/ASLP-lab/SongFormer · model https://huggingface.co/ASLP-lab/SongFormer/tree/main
- S3 paper https://arxiv.org/abs/2501.01108 · repo https://github.com/tencent-ailab/MuQ · weights https://huggingface.co/OpenMuQ/MuQ-large-msd-iter/tree/main
- S4 paper https://arxiv.org/abs/2306.00107 · model https://huggingface.co/m-a-p/MERT-v1-330M
- S5 paper https://arxiv.org/abs/2407.21658 · repo https://github.com/CPJKU/beat_this
- S6 paper https://arxiv.org/abs/2110.06525
- S7 paper https://arxiv.org/abs/2007.08411 · repo https://github.com/MZehren/M-DJCUE (data) / `MZehren/Automix` (code)
- S8 paper https://arxiv.org/abs/2508.06516 · repo https://github.com/ax-le/automashup
- S9 paper https://archives.ismir.net/ismir2017/paper/000086.pdf
- S10 repo https://github.com/urinieto/msaf · paper https://ccrma.stanford.edu/~urinieto/MARL/publications/ISMIR2016-NietoBello.pdf

---

## 3. What changed since the V7 research baseline

1. **Reviewed structural models improve on All-In-One in their reported benchmarks.** SongFormer
   (S2, Sep–Oct 2025) reports strict boundary HR.5F 0.703 on
   SongFormBench-HarmonixSet vs All-In-One 0.596, and 0.651 on RWC-Pop vs
   LinkSeg 0.648 (tables read directly from the arXiv HTML and the GitHub
   README; identical numbers in both). This is real progress on *boundary*
   accuracy, which matters for "don't cut inside a phrase". But it is heavy
   (see §5) and it labels functional sections (intro/verse/chorus/...), which
   is a different claim from "this is a good perceptual transition cue".

2. **Generic audio embeddings are not a compatibility oracle.** AutoMashup
   (S8) explicitly reports that CLAP and MERT embeddings **fail to reproduce**
   the perceptual coherence measured by COCOLA, and that mashup compatibility
   is asymmetric (role-dependent vocals vs accompaniment). This is a direct
   primary-source warning against treating a chroma/embedding cosine as
   "resolution". MuQ/MERT (S3/S4) are strong *representation* backbones, but
   neither paper claims to predict whether an A→B transition resolves tension.

3. **Beat tracking without DBN is now practical and tiny.** Beat This! (S5)
   ships `small0` at ≈8.1 MB and runs on CPU; the repo already integrates it.
   This confirms beats/downbeats are a solved-ish *metrical* anchor, but the
   authors themselves note worse continuity metrics — so beats alone are not a
   phrase or a resolution signal.

4. **No model reviewed here is validated to output "musical expectation is resolved".** Across S1–S10
   the closest primitive is *functional section labels* (S1/S2) and
   *compatibility scores* (S6/S8). Neither is validated for our A/B pair. The
   user's requirement (rising tension continues OR resolves into B) remains an
   unsolved, unbenchmarked objective here.

---

## 4. Feasibility on this machine (Windows, CPU, no downloads now)

- **S5 Beat This! small0** — already the chosen M02 provider in
  `src/musicmix/structure.py`. ~8 MB, CPU, MIT. Keep as the beat/downbeat
  anchor. This is the only learned MIR model currently justified to run here.
- **S1 All-In-One** — inference exists on PyPI (`allin1`) and is CPU-capable,
  but it pulls **NATTEN** (on Windows must be built from source), Demucs and
  madmom, and processes via 4-stem separation. That is a large, fragile
  dependency surface. Treat as an *optional heavy backend*, gated behind an
  explicit extra and a weights-license record. It is the most realistic neural
  path to real section boundaries/labels without writing our own model.
- **S2 SongFormer** — strongest boundary accuracy, but ~2.76 GB annotation
  model **plus** MuQ and MusicFM backbones, 24 kHz input, GPU strongly
  recommended. Not CPU-feasible for a 15-minute/no-download workflow. Later.
- **S3/S4 MuQ/MERT** — 1.33 GB / ~1.3 GB, 24 kHz, CC-BY-NC-4.0. Usable later
  as embedding *features* for a trained ranker, not as standalone truth. The
  NC weight license must be recorded before any distribution claim.
- **S10 MSAF** — pure Python + librosa, CPU-only, MIT, no weights. The most
  feasible *optional structural backend* available **today** to replace
  "beats only" with real novelty/segment boundaries (see §6).

---

## 5. Why a defensible "SOTA" claim is not currently possible

A defensible SOTA claim for our task would require all of:

1. **A defined task with a public, matched test set.** For us that would be
   A→B transition-cue selection on a curated set, with annotations of
   "continuation vs abandoned tension" — which we do not have. Our four songs
   are qualitative holdouts (see `docs/14`, `docs/19`), never a threshold-fit
   set.
2. **A baseline and a metric that both parties share.** Boundary metrics
   (HR.5F/ACC) measure segmentation, not transition quality. Compatibility
   metrics (COCOLA) are a proxy that S8 shows generic embeddings do not match.
   We have neither a validated metric nor a baseline implementation of any
   paper here.
3. **A fair, source-bound evaluation.** Our repo binds outputs to exact source
   hashes/sample rates. Any SOTA claim must reproduce that binding and report
   explicit missing evidence instead of fabricated confidences.
4. **Independent perceptual validation.** Historical V3/V4 user judgements exist
   in this repo, but they do not validate V8, an automatic critic, or a public
   SOTA benchmark (`docs/history/14_VALUTAZIONE_UMANA.md`).

What we **can** benchmark now (measurable, honest):
- Determinism/contracts of `analyze_structure` and its Beat This! adapter
  (already covered by the existing test suite).
- *Recall/coverage* of candidate cue points on the four tracks: how many
  downbeats/section boundaries exist away from the extremes, per direction.
- *Agreement* between a heuristic boundary set and an optional provider
  boundary set (e.g. MSAF vs `heuristic_feature_sections`) as a diagnostic —
  explicitly not a quality score.
- The existing structured-run metrics (pairs evaluated, admitted proposals,
  render time) as reproducibility numbers.

None of these is a SOTA claim.

---

## 6. Structural segmentation provider to replace "beats only"

Requirement from the task: *at least investigate an actual structural
segmentation provider to replace relying only on beats; prefer a feasible
optional backend path.* Recommendation, in order:

### 6.1 Now (feasible, no weights): MSAF (S10) as an optional backend
- Pure Python + librosa/scikit-learn, MIT, CPU-only, no checkpoint download.
- Provides boundary algorithms (Foote novelty, checkerboard kernel, spectral
  clustering) and can be driven from the existing canonical timeline.
- **Honest framing:** MSAF boundaries are *heuristic/structural*, not learned
  functional labels. They must be emitted with a label such as
  `msaf_boundary` and `confidence=None`, in the same spirit as the existing
  `heuristic_feature_segment`. MSAF does **not** certify chorus/verse/phrase.
- Integration shape (do not implement now, just the contract): a
  `tools/analyze_structure.py` adapter that maps MSAF segment boundaries to
  `ProviderPrediction.sections`, leaving `beats`/`downbeats` to Beat This!.
  This upgrades M02 from "beats only + spectral heuristic" to "beats +
  toolkit structural boundaries", with explicit provenance.

### 6.2 Later (neural, heavy): All-In-One (S1) then SongFormer (S2)
- All-In-One gives learned boundaries **and** functional labels; feasible on
  CPU but with a heavy, Windows-fragile dependency stack (NATTEN/Demucs/
  madmom). Gate behind an optional extra and a model-registry entry with exact
  checkpoint hash and license.
- SongFormer is the current accuracy leader but not CPU-feasible here; record
  as "later / research".

### 6.3 Do not claim
- Do not rename `heuristic_feature_segment` to "chorus/verse"; do not present
  MSAF or All-In-One output as a *perceptually validated* transition cue. The
  papers measure segmentation, not transition pleasantness.

---

## 7. Recommendation summary for V8+

- **Implement now (small, safe):** document and, when a code task is
  authorized, add an optional MSAF-based structural backend alongside the
  existing Beat This! provider; keep `kind`/label provenance explicit and
  `confidence=None`.
- **Later:** All-In-One as an optional heavy backend; MuQ/MERT embeddings only
  as features for a *to-be-validated* ranker, never as a resolution oracle.
- **Never (without evidence):** assert SOTA, assert semantic resolution, or
  treat chroma/embedding/RMS similarity as continuity.

---

## 8. Limitations of this document

- Read depth is mixed and stated in §1; MuQ/MERT/DJtransGAN/Cue/AutoMashup were
  not read end-to-end.
- Footprint numbers are from official host listings, not local measurement.
- Licenses are recorded as seen on official pages (code vs weights separated
  where possible) but were not legally audited.
- No model was run; no transition was rendered; no human listened.
