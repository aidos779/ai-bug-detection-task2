# Instrument verification

Checks completed on 8 October 2026. These are engineering checks on handmade
smoke fixtures, not original research results on real software.

## Executed checks

- Local Python 3.12.14 on macOS arm64: all 9 unit tests passed.
- Reference Docker build: succeeded with pinned Python 3.12.10 / Debian bookworm;
  all 9 unit tests ran successfully during the build.
- Container smoke run: succeeded with `--cpus=1 --memory=512m`, persisted the JSON
  summary and both CSV outputs through the mounted results directory.
- Eight test rows produced eight paired records; 30 repetitions produced 240
  timing observations per method after five warm-up passes.
- Input leakage, invalid labels, duplicate ASTs, known confusion counts, NB
  hand-calculated posteriors, bootstrap pairing, smoke interpretation and CLI
  failure behaviour are covered by the tests.
- Methodology PDF: three A4 pages, rendered and visually checked for text/table
  overflow and pagination.

## Fixture observations from the reference container

| Method | TP/FP/TN/FN | Precision | Recall | F1 | FPR | p95 latency ms |
|---|---|---|---|---|---|---|
| Static AST rules | 4/0/4/0 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.030173 |
| Multinomial NB | 4/2/2/0 | 0.6667 | 1.0000 | 0.8000 | 0.5000 | 0.005752 |

Delta F1 = -0.20; fixture-only project-bootstrap interval =
[-0.3333, 0.0000].
All 2000 of the 2,000 draws retained both classes.
Process peak RSS was 21.44 MiB on Linux aarch64.
Timings and RSS are observations and are not exact reproducibility targets.

The NB false-positive guardrail **failed** on these fixtures (0.50 > 0.10).
This does not mean the instrument failed: it correctly reports an unfavourable
method outcome. It must not be hidden or presented as an AI success. Research
hypotheses remain untested; the program's smoke interpretation is
`pipeline_validated_only_no_research_conclusion` and `practical_success` is false.

## Frozen input and instrument hashes

- `train_sha256`: `b81b5778fcea0b4af6dab82aeb5d50bdbadec6af4cc259e31db6a38055f84247`
- `test_sha256`: `fc61aaa9933aff7f54c571ba68b661c2f17ece0ada7ef2cbb4ed50b399d5f5a7`
- `config_sha256`: `6d9ca2fe013ce9ced5db02f78fbab1ad716a8ea7c1207fe9da7167a06121e60d`
- `code_sha256`: `ff4c534d911449806fcada56428d11866b7950e757f766a57e75c859cf388837`

## Research-data readiness

The measurement instrument is ready. The real-code dataset and independent
annotation are planned for W7 and have not been collected. `configs/research.json`
points to the future ignored data files and refuses smoke fixtures and undersized
corpora. The input checks cannot replace source/label review or licensing checks.
