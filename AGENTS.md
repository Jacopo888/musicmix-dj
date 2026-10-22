# MusicMix DJ: rules for research and implementation (humans and coding agents)

- Before changing selection, rendering or evaluation of transitions, read `conoscenza_mix/INDEX.md` and the
  summaries of the relevant sections; update the notes after every substantial iteration, linking each source
  to the code that implements it.
- Distinguish experimental results, theories, engineering hypotheses and listener observations. A cited article
  does not validate our thresholds, scores or a specific transition.
- Reuse official pretrained models and implementations when appropriate; record revisions, checkpoints, hashes,
  licences and domain limits. Never commit weights or audio.
- Do not equate tension with loudness, phrase end with an ASR pause, chorus with resolution, or technical
  integrity with musical quality. Keep the unknown explicit.
- Decide jointly whether to change, when to leave A, where to enter B and which transition to use; keep
  "continue A" as an explicit action, shown only as a comparison reference, never as a clip to rate.
- Keep source audio, judgements and frozen runs immutable. Previous judgements are held-out diagnostics: do not
  tune thresholds on the single example someone liked and then present it as an independent validation.
- Verify source identity (hashes), coordinates in samples, context, affected words and the produced PCM; make
  pickups and arrivals explicit.
- Listening clips last 20 seconds, 10 seconds per side (blends: 10 s before + whole transition + 10 s after).
  Keep seek/HEAD/Range support in the local player.
- Priority: DJ craft first (beatmatching, cue on the downbeat at a phrase start, EQ / bass swap, tempo and key
  compatibility, vocal management, loudness). Psychological modules stay diagnostic.
- Books and papers: cite book/paper and page for every derived rule; keep only notes and short quotations in
  the repository, never full texts.
- Autonomy: new tracks and new sets must work without a human vote each time; supervision comes from automatic
  checks on renders (defect gym), track cards, examples from the tracks themselves and reference DJ sets, with
  periodic human checks on samples.
- Do not promise 10/10 ratings or state-of-the-art results without a relevant evaluation.
- Use only music you have the rights to use.
