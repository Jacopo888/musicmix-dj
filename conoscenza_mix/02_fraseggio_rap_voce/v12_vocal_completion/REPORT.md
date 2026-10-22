# Vocal completion: bounded frame-feature extraction

## Scope

Owned files only:

- `src/musicmix/vocal_completion.py`
- `tests/test_vocal_completion.py`

No model, data, runtime, or package-install changes. Optional Silero/Torch paths
were left untouched and were not downloaded or imported by these tests.

## Source change

`extract_vocal_features` no longer materialises the full-song frame matrix,
`float64` squared frame matrix, or full STFT power matrix.

- Frame count is derived from the sample clock (`_frame_count`).
- Each block reads at most `FEATURE_CHUNK_FRAMES = 512` frames via
  `_frame_block` / `_frame_descriptors`.
- RMS, ZCR, periodic-Hann STFT magnitude, flatness, and >2 kHz ratio are
  computed per block and written into scalar per-frame arrays.
- Spectral flux carries the previous block's final magnitude vector across each
  boundary, so the per-frame grid and flux definition are unchanged.
- YIN reuses the existing `_yin_pitch` chunked CMND implementation.
- No full spectral matrix is retained; peak working memory is one bounded
  block (512 x 2048 samples plus one 1025-bin STFT).

The STFT still uses periodic Hann (`get_window("hann", 2048, fftbins=True)`),
48 kHz, 2048-sample frames, 480-sample hop, `boundary=None`, `padded=False`.

## New tests

- `test_feature_chunk_size_is_result_invariant`
  Compares chunk sizes 7 vs 512 for RMS, ZCR, flatness, HF ratio, flux, f0,
  and voicing.
- `test_chunked_descriptors_match_reference_short_stft`
  Compares chunked descriptors against the original full-frame periodic-Hann
  STFT reference on a short tone.

Existing gain-invariance, known-pitch, YIN chunk-invariance, and source-clock
tests remain green.

## Commands

Run from this agent folder:

```powershell
$env:PYTHONPATH='src'; python -m pytest tests/test_vocal_completion.py -q
```

New tests only:

```powershell
$env:PYTHONPATH='src'; python -m pytest tests/test_vocal_completion.py::test_feature_chunk_size_is_result_invariant tests/test_vocal_completion.py::test_chunked_descriptors_match_reference_short_stft -q
```

Bounded-memory probe on a synthetic 60 s 48 kHz stem (no file I/O). Change
`n` to `6*60*sr` for the 6 min stem; cost grows with duration while retained
memory stays bounded by one chunk. The 6 min variant was not completed inside
this bounded handoff (YIN's per-frame Python loop dominates), so the short
probe is the one to run first:

```powershell
$env:PYTHONPATH='src'; python -c "import tracemalloc,numpy as np; from musicmix.vocal_completion import extract_vocal_features; sr=48000; n=60*sr; t=np.arange(n,dtype=np.float32)/sr; s=(0.2*np.sin(2*np.pi*180*t)).astype(np.float32); tracemalloc.start(); tr=extract_vocal_features(s,sr); cur,peak=tracemalloc.get_traced_memory(); print('frames',tr.frame_centers.size,'peak_MB',round(peak/1e6,1))"
```

## Result

`10 passed, 1 skipped` (the skip is the optional librosa reference when
librosa is unavailable). The 60 s probe above observed `frames 5996` and
`peak_MB 65.5`, consistent with bounded chunk working memory rather than a
duration-proportional spectral matrix. Root integration should replace its
local copy of `vocal_completion.py` with this module and keep the two new
tests.
