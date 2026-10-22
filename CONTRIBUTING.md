# Contributing to MusicMix DJ

Thanks for helping. Issues and pull requests are welcome in English or Italian.

## Hard rules

1. **No commercial audio.** Do not commit, attach or link audio, stems or renders of music you do not have the
   rights to redistribute. Tests use synthetic signals generated in code; examples use public-domain or
   Creative Commons material with the licence recorded.
2. **No protected texts.** No PDFs, EPUBs, scans or extracted full texts of books and papers, and no lyrics.
   Cite with DOI/URL; add bibliographic metadata to the relevant `SOURCES.json`.
3. **Short quotations only**, at most a few sentences (about 300 characters), always with book/paper and page
   (or section), and only when the exact wording matters. Prefer a paraphrase.
4. **Separate evidence from hypotheses.** Label rules as `[E]` experimental result, `[T]` theory, `[P]` DJ
   practice, `[H]` our engineering hypothesis. A cited paper does not validate one of our thresholds; a
   transition that one person liked is not a validation.
5. **No model weights or datasets** in the repository. Add them to `env/fetch_models.py` with the official URL,
   revision and SHA-256, and their licence to `THIRD_PARTY.md`.
6. **No secrets or personal data**: no tokens, private paths, e-mail addresses or listening judgements of other
   people without consent.

## Workflow

- Run `pytest -q -m "not data and not gym"` before opening a pull request; add tests with synthetic audio for new
  behaviour. Tests that need real analysed tracks get `@pytest.mark.data("relative/path", ...)` and are skipped
  when the data are missing.
- Keep coordinates in samples, record versions and hashes of inputs and outputs, and never modify frozen run
  outputs: write new runs instead.
- Before changing selection, rendering or evaluation, read `conoscenza_mix/INDEX.md` and update the relevant
  notes when the change is substantial.
- Listening clips are 20 seconds, 10 seconds on each side of the switch (for overlapping blends: 10 s before
  the transition, the whole transition, 10 s after).

By contributing you agree that your contribution is licensed under the Apache License 2.0.
