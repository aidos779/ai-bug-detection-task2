# Reproducibility and team workflow

## Two-minute audit

- [ ] `Dockerfile` pins the full image digest and Python 3.12.10; no floating
  runtime package installations occur.
- [ ] `docker build` runs all tests and succeeds; the README smoke command emits
  `summary.json`, `predictions.csv` and `latencies.csv`.
- [ ] Config records seed 42, alpha, decision threshold, warm-up/repetition counts,
  data/output paths and guardrails; summary records config/data/code hashes.
- [ ] Labels, snippet IDs and source projects are valid; smoke fixtures are marked
  synthetic; real runs have reviewed ground truth and no train/test leakage.
- [ ] `.gitignore` excludes environments, caches, outputs, raw/processed data,
  secrets and model binaries; author names, methodology and MIT license are present.

## Team Git workflow

The repository is initialized and published with the Task 2 deliverable. Each
author should use their own Git identity for subsequent work. Do not manufacture
commits or co-author lines to suggest contributions that did not occur.

```bash
git clone https://github.com/aidos779/ai-bug-detection-task2.git
cd ai-bug-detection-task2
git switch -c codex/annotation-protocol
# Edit the protocol or instrument and run the tests.
git add docs/data_protocol.md
git commit -m "Clarify label evidence for mutable-default defects"
git push -u origin codex/annotation-protocol
```

Open a pull request for the other author to review. Use another descriptive
branch for other changes. Before merging, review tests, dataset provenance,
protocol changes and tracked files. Changes to the registered experiment must
be documented before collecting or inspecting the new test results.

## Reproducing a real run

Archive the exact code revision, config, source/annotation manifest, data hashes,
container architecture and host details together with the result files. A
container fixes software dependencies; it does not make timing identical on
different hardware. Quality scores and seeded resampling are deterministic for
the same frozen data and implementation. Timing and RSS remain observations.
