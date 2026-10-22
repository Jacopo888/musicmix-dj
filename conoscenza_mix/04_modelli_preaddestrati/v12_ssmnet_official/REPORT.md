# Official SSM-Net inference report

## Summary

- Implemented and ran the **official pretrained SSM-Net** from
  `geoffroypeeters/ssmnet_ISMIR2023` at pinned commit
  `95c937c9facd482eb4c5dbf18137661ca040b824`.
- Used the official smallest checkpoint `nbatt1_..._state_dict.pt`
  (1,483,802 bytes, SHA256 `7e4863926e734b6cfee761101e3e226c52cfbd89b50f4b6e2f6751fbf363a3a9`,
  N=1 Transformer block, 367,911 parameters).
- Loaded it with an **exact state_dict key/shape check** (31 keys, no
  missing, unexpected, or mismatched entries).
- Used the original frontend (`source/ssmnet/utils.py`), model
  (`source/ssmnet/model.py`), and peak post-processing
  (`source/ssmnet/utils.py:f_get_peaks`). No substitute or random weights.
- Produced actual local structural novelty curves and official local events
  for `inputs/real_01.wav` and `inputs/real_02.wav`.
- No blocker: both clips ran successfully.

**Scope:** these are structural boundary novelty outputs from the official
pretrained model. They are not a mix-level judgement, a ranking, or a
quality/SOTA claim.

## Provenance and licenses

| item | value |
|---|---|
| repo | https://github.com/geoffroypeeters/ssmnet_ISMIR2023 |
| pinned commit | `95c937c9facd482eb4c5dbf18137661ca040b824` (2023-10-20) |
| paper | arXiv:2309.02243v1, ISMIR 2023, CC BY 4.0 |
| checkpoint | `nbatt1_0411-132051_9_epoch=11-step=1200_state_dict.pt` |
| checkpoint size | 1,483,802 bytes |
| checkpoint SHA256 | `7e4863926e734b6cfee761101e3e226c52cfbd89b50f4b6e2f6751fbf363a3a9` |
| checkpoint git blob SHA | `437fa24d2382a46bf9ea5c507d4e40eb745262a5` |
| code license declared | LGPL-3.0 in `setup.py` |
| repo LICENSE file | absent at the pinned commit |
| GitHub API license | `null` |
| checkpoint license | **not stated separately; unknown** |
| source snapshot | `source/` (official files, hashes in `research/SOURCES.md`) |
| paper archive | `research/originals/2309.02243v1.pdf`, metadata XML, extracted text |

The checkpoint is redistributed only as the official artifact required to
run the official model. Do not assume it inherits the code's declared
license.

## What was run

Command (the runner inserts the three allowed dependency directories itself):

```powershell
python tools\run_ssmnet_official.py inputs\real_01.wav inputs\real_02.wav
```

Pipeline, exactly following the official code:

1. `SsmNetDeploy.m_get_features`: `librosa.load` (22050 Hz, mono),
   `f_extract_feature` (80 Mel bands, fmax 8000, `log(1+100*mel)`),
   `f_reduce_time` (nominal 0.1 s), `f_patches` (40 synced frames per patch,
   hop 5).
2. Instantiate official `SsmNet` with `do_nb_attention=1` and the kernel
   parameters from `config_example.yaml` (`kernel_nb=3`, `Ldemi=10 s`,
   `sigma=5 s`), then load the official state_dict strictly.
3. `SsmNet.get_novelty`: learned embeddings -> SSM -> learned 3-kernel
   novelty convolution -> sigmoid.
4. Official `f_get_peaks`: peak-to-mean ratio with a 20 s mean window,
   threshold 1.35, minimum inter-distance 7 s.
5. Export raw novelty, official events, diagnostic peak candidates, SSM,
   official-style segments, and JSON metadata.

The official `SsmNetDeploy.m_get_ssm_novelty` hardcodes the checkpoint path
under `source/ssmnet/weights_deploy/`. The runner repeats its two inference
lines with an explicit `weights/` path and adds the strict check; the model
math, frontend and peak routine are unchanged.

CPU threads were pinned to 1. No global installs and no Torch/TF download.
The first clip's run took 11.93 s including library import; the second took
0.98 s. This is far below the 8-minute budget.

### Implementation detail: 45x45 kernel, not 41x41

The paper says the learned novelty kernel is 41x41 (roughly 20 s). The
official frontend's actual synced hop is 0.09288 s (4 STFT frames), so the
patch hop is 0.464399 s. The code computes
`round(10 / 0.464399) = 22`, giving a **45x45** kernel spanning about
20.9 s. The checkpoint indeed stores `conv_novelty.weight` as
`(3, 1, 45, 45)`. This is expected from the official code and integer
rounding, not a state_dict mismatch.

## Actual results

Both inputs are 20.000 s, 48 kHz, stereo, 16-bit PCM. The official frontend
converts them to 22050 Hz mono for analysis.

| clip | patches | patch centers | official events | segment export |
|---|---:|---|---|---|
| `real_01` | 35 | 1.904-17.694 s | 2.833 s, 14.907 s | 0.000, 2.833, 14.907, 17.694 s |
| `real_02` | 35 | 1.904-17.694 s | 2.368 s, 12.585 s | 0.000, 2.368, 12.585, 17.694 s |

Official event detail:

| clip | event time | novelty score | peak-to-mean |
|---|---:|---:|---:|
| `real_01` | 2.832834 s | 0.202463 | 5.294 |
| `real_01` | 14.907211 s | 0.558224 | 4.744 |
| `real_02` | 2.368435 s | 0.160046 | 3.224 |
| `real_02` | 12.585215 s | 0.140218 | 2.011 |

Raw novelty ranges over the 35 patch centers:

- `real_01`: min 0.000567, max 0.558224, mean 0.114581.
- `real_02`: min 0.002787, max 0.345626, mean 0.070311.

Interpretation cautions:

- The official event selection uses **peak-to-mean ratio**, not a raw
  novelty threshold. A low raw novelty value does not mean "no event" under
  the official rule.
- The novelty score is the model's sigmoid output but is **not calibrated**
  as a probability.
- The official patch loop defines novelty only at patch centers on a
  **0.464399 s** grid. The patch centers span 1.904-17.694 s; the patches
  themselves support about 0.046-19.551 s.
- The official segment export adds 0 and the last patch-center time, so it
  ends at 17.694 s, **not** at the 20 s audio end.
- A 20 s clip is at or below the design scale of this model: the learned
  kernel spans ~20.9 s, the official peak-to-mean window is 20 s, and the
  paper's training segments have median durations around 15-18 s. The clip
  outputs are actual pretrained inference, but they should not be treated as
  validated structural boundaries.
- Each clip's SSM is only 35x35, smaller than the 45x45 learned kernel. The
  official `Conv2d(padding='same')` therefore zero-pads a large part of the
  kernel support. This is a direct consequence of running the full-track
  model on a 20 s clip.
- The 20 s peak-to-mean window is truncated at both ends of the 35-point
  curve. The early events (2.37-2.83 s) are especially close to that
  truncated edge and should be read with this in mind.

The raw curve is in `outputs/*_novelty.csv`; it can be re-thresholded by the
root without rerunning the model.

## Verification performed

- Checkpoint SHA256 checked against the pinned value at run time.
- Model state_dict key sets and tensor shapes compared explicitly before a
  `strict=True` load: 31 keys, 0 missing, 0 unexpected, 0 shape mismatches.
- Computed kernel shape 45x45 matches the checkpoint.
- Diagnostic peak-to-mean candidates above the official threshold were
  compared to `f_get_peaks` output and were identical.
- Official source file hashes recorded in `research/SOURCES.md` and
  `outputs/provenance.json`.
- `torch` loaded from the already-present base Python installation
  (2.14.0+cpu); `librosa`/`scipy`/`soundfile`/`yaml` from the allowed
  read-only runtime directory. No installation or model download other than
  the official checkpoint.
- A `padding='same'` warning with even kernel sizes is emitted by PyTorch for
  the official architecture. It is expected and does not alter the strict
  weight match.

## Output files

| path | content |
|---|---|
| `outputs/summary.json` | both clips, full metadata and event lists |
| `outputs/<clip>_run.json` | per-clip metadata, strict-load report, events |
| `outputs/<clip>_novelty.csv` | `patch_index,time_local_sec,time_canonical_sec,novelty_score` |
| `outputs/<clip>_peak_candidates.csv` | local maxima, peak-to-mean, official flag |
| `outputs/<clip>_segments.csv` | official-style segment export |
| `outputs/<clip>_ssm.npy` | model SSM, 35x35 float32 |
| `outputs/provenance.json` | pins, hashes, licenses, runtime paths |
| `outputs/real_01_canonical_offset92p5.json` | adapter demonstration for a clip starting at 92.5 s |
| `research/originals/` | paper PDF, arXiv metadata, extracted text |
| `research/SOURCES.md` | source pins, hashes, licenses, budget |
| `research/paper_notes.md` | paper claims vs official-code behavior |

## Canonical time-offset adapter for 7 full songs

`tools/time_offset_adapter.py` converts one run or a manifest of runs to a
single canonical full-song timeline. The mapping is:

```text
time_canonical_sec = time_local_sec + clip_start_sec
sample_index       = round(time_canonical_sec * source_sr)
```

Segments are emitted as half-open `[start, stop)`; events are instants.
The adapter maps the novelty curve, official events, diagnostic peak
candidates, segments, patch-center coverage, and patch support.

Single run:

```powershell
python tools\time_offset_adapter.py outputs\real_01_run.json `
  --clip-start-sec 92.5 --source-sr 48000 `
  --window-start-sec 95 --window-end-sec 105 `
  -o outputs\canonical.json
```

Seven songs: fill `tools/canonical_manifest.template.json` (one entry per
song with `song_id`, `run_json`, `clip_start_sec`, `source_sr`), then:

```powershell
python tools\time_offset_adapter.py --manifest seven_songs.json `
  --expect-songs 7 -o outputs\canonical_7songs.json
```

If SSM-Net is run directly on a full song, pass `clip_start_sec = 0` (or
omit it and ensure the run's `offset_sec` is 0). If it is run on a 20 s clip
cut from a song, pass the canonical song time of the clip's first sample.
The adapter was tested on a two-entry temporary manifest; the seven-entry
template is provided but was not run because the seven full songs are
outside this agent's allowed directory.

## Limitations and non-claims

- No ground-truth evaluation was performed; this is an inference/provenance
  task, not a benchmark.
- No claim of 10/10, SOTA, or model superiority. The paper-reported numbers
  are quoted separately in `research/paper_notes.md` and are not results on
  these clips.
- The outputs are structural boundary novelty only. They are not an
  assessment of a mix or of arrangement suitability.
- The checkpoint license is unknown; the repository has no LICENSE file,
  despite `setup.py` declaring LGPL-3.0.
- A 20 s input is short relative to the model's ~20.9 s novelty kernel and
  20 s post-processing window; edge context is limited.
- Boundary timing is quantized to 0.464399 s. The paper itself reports low
  HR0.5F for this reason.
- The exact nbatt1 training YAML was not released separately; the
  architecture/hyperparameters used are those implied by
  `config_example.yaml` (with `do_nb_attention=1`) and proven consistent by
  the strict state_dict match.

## Blockers

None for the staged 20 s clips. Full-song execution for the root's seven
songs requires those audio files and the root's manifest; the adapter and
runner are ready.

