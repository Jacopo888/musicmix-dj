# Paper notes and implementation mapping

Source: G. Peeters, *Self-Similarity-Based and Novelty-based loss for music
structure analysis*, ISMIR 2023, arXiv:2309.02243v1. Archived at
`research/originals/2309.02243v1.pdf`; extracted text at
`research/paper_text.txt`.

These notes separate **paper statements**, **official-code behavior**, and
**what this run actually did**.

## 1. Task and assumptions

The paper addresses music structure analysis (MSA), specifically segment
boundary detection. It uses the standard assumptions of novelty,
homogeneity, and repetition (Section 1). The system is trained to predict a
self-similarity matrix (SSM) and a novelty curve jointly. It is a
boundary detector; the paper notes the SSM can also support labeling, but
labeling is not evaluated (Section 4).

## 2. Input frontend (paper Section 2.1)

Paper statement:

- 80-band Mel spectrogram.
- 92 ms window, 23 ms hop.
- `log(1 + 100 * mel)` compression.
- Mean aggregation over time to a 0.1 s hop.
- Patches of 40 successive synced frames (nominal 4 s), hop 5 frames
  (nominal 0.5 s).

Official-code behavior (`source/ssmnet/utils.py`):

- `librosa.load(audio_file)` with defaults: 22050 Hz, mono.
- `librosa.feature.melspectrogram(y, sr, n_mels=80, fmax=8000)` with
  librosa defaults `n_fft=2048`, `hop_length=512`, i.e. 92.88 ms window and
  23.22 ms hop.
- `log(1 + 100 * mel)`, then per-Mel-band mean/std normalization.
- `f_reduce_time` uses `int(floor(0.1 / 0.02322)) = 4` STFT frames, so the
  actual synced hop is **0.09288 s**, not exactly 0.1 s.
- `f_patches` takes 20 synced frames on each side: 40 frames = **3.7152 s**
  patch duration, hop 5 synced frames = **0.464399 s**.
- Patch centers are the synced-frame centers, so the first/last centers are
  not at the audio edges.

Consequence for the novelty kernel: the code computes
`kernel_Ldemi = round(10 / 0.464399) = 22`, hence `M = 45`, and the learned
`conv_novelty` kernel is **45 x 45**, spanning about **20.9 s**. The paper
says the kernel is 41 x 41 (roughly 20 s). The 41-vs-45 difference is
expected from the actual synced hop and integer rounding; it is not a
checkpoint mismatch. The downloaded checkpoint indeed stores
`conv_novelty.weight` as `(3, 1, 45, 45)`.

## 3. SSM loss (paper Section 2.2)

For each patch, the encoder produces an embedding. Embeddings are
L2-normalized, and the estimated SSM is

```text
SSM_hat_ij = 1 - ||e_i - e_j||^2 / 4
```

which is a scaled cosine similarity in [0, 1]. The SSM loss is a weighted
binary cross-entropy against the annotation-derived binary SSM, with the
weight correcting the strong imbalance of positive/negative pairs.
Official implementation: `source/ssmnet/model.py`, `get_ssm`, and
`source/ssmnet/utils.py`, `f_weighted_bce_loss`.

## 4. Novelty loss (paper Section 2.3)

A set of learned kernels is convolved along the SSM diagonal, combined by a
1x1 convolution, and passed through a sigmoid. The novelty loss is BCE
between the predicted novelty and the smoothed annotation novelty. The
paper says the kernel convolution is "without bias"; the official code uses
`nn.Conv2d` defaults, and the checkpoint contains `conv_novelty.bias` and
`lin_novelty.bias`. This is an implementation detail of the released code.

Official implementation: `SsmNet.get_novelty` in
`source/ssmnet/model.py`; kernel initialization in
`f_checkerboard_kernel`.

## 5. Relative feature learning (paper Section 2.4)

Self-attention lets each patch embedding depend on the other patches of the
same track. This is the motivation for the Transformer encoder blocks.
`do_nb_attention=1` means one such block (the `nbatt1` checkpoint).

## 6. Architecture (paper Section 2.5)

Five convolution blocks:

| block | channels | kernel | pool |
|---|---:|---|---|
| 1 | 32 | 5x5 | 2x2 |
| 2 | 32 | 5x5 | 2x2 |
| 3 | 64 | 5x5 | 2x2 |
| 4 | 64 | 5x5 | 2x2 |
| 5 | 128 | 5x2 | 5x2 |

Each block is Conv2d -> PReLU -> MaxPool2d. The output is flattened to a
128-vector. Then N Transformer encoder layers (`d_model=128`, 8 heads,
feed-forward 128), tanh, and L2 normalization. The paper reports 268K-567K
parameters for N=0..3. The nbatt1 state_dict here has **367,911**
parameters, consistent with N=1.

## 7. Training (paper Section 2.6)

Joint loss `lambda * SSM_loss + (1-lambda) * novelty_loss`, Adam with
learning rate 0.001, early stopping on validation loss with patience 50 and
at most 500 epochs, mini-batch = 10 tracks. Training data: 693 Harmonix
tracks and 298 Isophonics tracks. Test data: RWC-Pop-AIST and SALAMI
subsets. The paper's best lambda is 0.5.

We did not retrain. We loaded the released nbatt1 checkpoint.

## 8. Postprocessing (paper Section 3.2)

The paper detects boundaries from the novelty curve with a peak-to-mean
ratio, a 20 s mean window, threshold 1.35, and minimum inter-distance 7 s.
The official config (`config_example.yaml`) matches: `peak_mean_Ldemi_sec:
10`, `peak_distance_sec: 7`, `peak_threshold: 1.35`. The runner calls the
official `ssmnet.utils.f_get_peaks` directly.

## 9. Paper-reported results (not results on our clips)

From Table 2, for the N=1 / lambda=0.5 / trained-checkerboard-init
configuration (the family of our checkpoint), the paper reports:

- RWC-Pop-AIST: HR3F = .677
- SA-Pop (Upper/An1): HR3F = .631
- SA-IA (Upper/An1): HR3F = .520
- SA-Two (Upper/An1): HR3F = .521

The best RWC-Pop configuration in the paper is N=3 with HR3F = .713. The
paper explicitly notes that HR0.5F is low (e.g. .399 for the best
RWC-Pop configuration) because the feature hop is large; it discusses HR3F
rather than HR0.5F. These are test-set results from the paper, not claims
about the two 20 s clips in this folder.

## 10. What this run actually does

- Uses the official frontend functions (`f_extract_feature`,
  `f_reduce_time`, `f_patches`) through `SsmNetDeploy.m_get_features`.
- Instantiates the official `SsmNet` class with `do_nb_attention=1` and the
  config kernel parameters.
- Loads the official nbatt1 state_dict with a strict key/shape check.
- Calls the official `get_novelty` (SSM + learned novelty kernels).
- Calls the official `f_get_peaks` for events.
- Exports the raw novelty curve, the official events, the diagnostic peak
  candidates, the SSM `.npy`, the official-style segment CSV, and JSON
  metadata.
- Does **not** evaluate against ground truth, does not retrain, and does not
  replace weights.

## 11. Limits relevant to the 20 s clips

The model was trained on full tracks. The learned novelty kernel spans about
20.9 s, and the official peak-to-mean window is 20 s. A 20 s clip is
therefore at or below the scale of the model's context, and the clip edges
carry little or no context.

A sharper version of the same issue: the 20 s clips produce 35x35 SSMs,
while the learned novelty kernel is 45x45. The official
`Conv2d(padding='same')` zero-pads the SSM, so a substantial part of the
kernel support is padding rather than audio. The 20 s peak-to-mean window is
also truncated at both ends of the 35-point curve. These are consequences of
applying the full-track model to a short clip, not implementation bugs.

The novelty values are sigmoid outputs but are not calibrated probabilities.
Boundary timing is quantized to the 0.4644 s patch hop. These facts limit
interpretation; they are not implementation blockers. The two clips produced
actual model outputs (Section 3 of the report), but no claim is made that
those events are validated structural boundaries.
