# Sources, pins, licenses, and hashes

## Official repository

- URL: https://github.com/geoffroypeeters/ssmnet_ISMIR2023
- Pinned commit: `95c937c9facd482eb4c5dbf18137661ca040b824`
- Commit date: 2023-10-20T15:17:10Z
- Commit message: `changed organization`
- Default branch at pin: `main`
- Downloaded at the pinned raw URLs, not from a moving branch.

We downloaded only the small source files and the smallest official
checkpoint. We did not clone or download git history, groundtruth pyjama
files, or the larger checkpoints.

## Paper

- arXiv: https://arxiv.org/abs/2309.02243
- Version archived: `2309.02243v1`, submitted 2023-09-05
- Title: *Self-Similarity-Based and Novelty-based loss for music structure analysis*
- Author: Geoffroy Peeters (LTCI, TÃ©lÃ©com-Paris, Institut Polytechnique de Paris)
- Venue: ISMIR 2023, Milan, Italy
- Paper license: CC BY 4.0 (stated in the paper's copyright block)
- PDF: https://arxiv.org/pdf/2309.02243v1
- Metadata: https://export.arxiv.org/api/query?id_list=2309.02243

## Checkpoint used

- File: `nbatt1_0411-132051_9_epoch=11-step=1200_state_dict.pt`
- Size: 1,483,802 bytes (~1.48 MB)
- SHA256: `7e4863926e734b6cfee761101e3e226c52cfbd89b50f4b6e2f6751fbf363a3a9`
- URL: https://raw.githubusercontent.com/geoffroypeeters/ssmnet_ISMIR2023/95c937c9facd482eb4c5dbf18137661ca040b824/ssmnet/weights_deploy/nbatt1_0411-132051_9_epoch=11-step=1200_state_dict.pt
- Contents: plain `OrderedDict` of tensors, 31 keys, 367,911 parameters.
- Git blob SHA (SHA-1 of the git object, not the file SHA256):
  `437fa24d2382a46bf9ea5c507d4e40eb745262a5`.
- `do_nb_attention=1` (one Transformer encoder block).
- `conv_novelty.weight` shape `(3, 1, 45, 45)`, matching the implemented
  feature grid (see `paper_notes.md` for the 41-vs-45 detail).
- We did not substitute, retrain, reinitialize, or randomize any weight.
  The runner loads this exact file and checks its SHA256 at run time.

## Source files and hashes

All files below are the official files at the pinned commit.

| path | bytes | SHA256 |
|---|---:|---|
| `source/README.md` | 3737 | `9ed218688bcb7b505fc6bdc52863bf72c155ec54a960f2bc45863970f343f64c` |
| `source/requirements.txt` | 214 | `184f85c72d775c7b19caf9e48d32c6a048fa3f2aaecd782580e0004640a81c9c` |
| `source/setup.py` | 794 | `1811916c39a4c9164eee19c5441da1c1c3a83a09f7903ff7f586d743d5341ff7` |
| `source/ssmnet/__init__.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `source/ssmnet/core.py` | 5933 | `7456c54dfba4f4f73907a73261b3c072c21846f1379fdbeb0a0e98324cab2b8f` |
| `source/ssmnet/model.py` | 5446 | `1fc4c55d83862aab2ac6d4c6322be7651f3b153f81da7dea26b32696e8ced425` |
| `source/ssmnet/utils.py` | 6321 | `3746ff7cf6dcf5d33bc4ecf8a1c3f5a33ef40efda3942d1028f0e58772ac9512` |
| `source/ssmnet/parser.py` | 766 | `cf5bb215db80393e2dea776806dc34c5c5a305b7c2dee7ea1736cd570b776017` |
| `source/ssmnet/example.py` | 1084 | `c136bae139ad83e294f0b02a25bd7276892b910452a4ac2a39c0b4b2e8ee0591` |
| `source/ssmnet/weights_deploy/config_example.yaml` | 650 | `72331df4a2a9b97482d9494dc4f9431e1a61114acc1fb3cd44c916446295659b` |

Git blob SHAs (SHA-1 of the git objects) for the key code files:
`ssmnet/core.py` = `707b13fe927552ebc75a1d0bea2b2a04063b74d6`,
`ssmnet/model.py` = `c0e428c70002c222fd7fa9f9f3d0850260d454c9`,
`ssmnet/utils.py` = `7dc98e00f9654bbaa5a8d84b0288c9004fbf97e4`.

Archived paper artifacts:

| path | bytes | SHA256 |
|---|---:|---|
| `research/originals/2309.02243v1.pdf` | 3,616,924 | `b1131b338c52914fc51eacdab9433fa9cd1e987ef89e69cd918afa311164f834` |
| `research/originals/arxiv_2309.02243_metadata.xml` | 2,313 | `9d60adb41f0e5db406cfcd77f95c01975b5e2c438f651a806fae7edb355d005a` |
| `research/paper_text.txt` | 40,291 | `e3f7c257e7895815544e4ee93a0d32b321faf6103b2c73cc42b1c6b98ca97e3f` |

## License findings (kept separate)

- The repository has **no standalone `LICENSE` file** at the pinned commit.
- The GitHub API reports `license: null`.
- `source/setup.py` declares `license='LGPL-3.0'`. We treat that as the
  code's declared license, but the absence of a LICENSE file means the
  repository's licensing is not unambiguous.
- **The checkpoint has no separate license statement.** Its license is
  unknown/unspecified. The checkpoint is redistributed here only as the
  official artifact required to run the official model; do not assume it
  inherits the code license.
- The paper itself is CC BY 4.0.

## Dependency provenance

No global installs, no Torch/TF download. The runner adds the three
explicit read-only directories from the task to `sys.path` and imports the
already-present libraries. Actual resolved paths at run time:

- Python 3.11.9
- torch 2.14.0+cpu, `<python site-packages>`
- numpy 2.4.6, base Python site-packages
- librosa 0.10.1, `...\agents\critic_research\runtime\librosa\__init__.py`
- scipy 1.17.1, `...\agents\critic_research\runtime\scipy\__init__.py`
- soundfile 0.14.0, `...\agents\critic_research\runtime\soundfile.py`
- PyYAML 6.0.3, `...\agents\critic_research\runtime\yaml\__init__.py`

`torch` was not present in the cached directories; the base Python
installation was used without installing or downloading anything. CPU
threads are pinned to 1 in the runner.

Paper text extraction used the already-present pypdf 6.16.2 (user
site-packages). It is not used by the inference runner.

## Network budget

Downloaded bytes, excluding tiny GitHub API JSON responses:

- checkpoint: 1,483,802
- official source files: 24,945
- arXiv metadata XML: 2,313
- arXiv PDF v1: 3,616,924

Total: **5,127,984 bytes (~5.13 MB decimal, ~4.89 MiB)**, below the 8 MB
budget. No model weights other than the nbatt1 state_dict were downloaded.

