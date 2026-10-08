# AI Tools for Software Bug Detection

**Authors:** Baltabay Aidos and Bissen Yerkebulan  
**Course deliverable:** Task 2 - Methodology Passport Card and Research Repository  
**Status:** Runnable pilot instrument; real-code data collection is planned  
**License:** MIT

## Overview

This repository develops the comparative-effectiveness question from our Task 1
review, *The Effectiveness of AI Tools in Software Bug Detection*. It provides a
small, fully executable quantitative pilot for Python defect detection. A learned
multinomial Naive Bayes (NB) lexical classifier is compared with a transparent,
non-AI abstract-syntax-tree (AST) rule detector. The two target defect families are
division by zero and unintended state sharing through mutable default arguments.

The NB model is a modest supervised-learning baseline. It is not an LLM,
LineVul, BugLab, or a reproduction of any reviewed tool. Its small scope enables
the complete measurement pipeline to run without model downloads or an API key.
It cannot establish the effectiveness of all AI tools.

### Research Question

On the same project-separated, independently labelled Python snippets, does a
learned lexical classifier improve defect-detection F1 relative to fixed AST
rules, while meeting the registered resource and false-positive limits?

For population performance within this bounded setting, **H0:** F1(NB) -
F1(static) <= 0; **H1:** F1(NB) - F1(static) > 0. The primary statistic is the
absolute F1 difference. A gain of 0.05 is the separately registered practical
threshold. These are planned hypotheses; the included fixtures do not test them.

## System Requirements

- Recommended environment: Docker Engine or Docker Desktop with Linux containers.
- Image: Python **3.12.10**, Debian bookworm, immutable multi-architecture digest
  recorded in `Dockerfile`; use the same image architecture and host for comparisons.
- Suggested Docker allocation for this pilot: one CPU and 512 MiB RAM; no GPU.
- Offline execution after the base image is pulled. No external packages, network
  calls, source-code execution, notebook server, or model weights are required.
- The local alternative is Python 3.12 on macOS or Linux. The container is the
  reference environment; other interpreter patches are compatibility checks.
- Classification outputs are deterministic. Wall-clock timings and process memory
  are observations and will vary across runs and machines.

## Quickstart Guide

Clone, build, and run in **three commands** (a POSIX shell):

```bash
git clone https://github.com/aidos779/ai-bug-detection-task2.git ai-bug-detection-task2
docker build -t ai-bug-detection-task2:1.0 ai-bug-detection-task2
docker run --rm --cpus=1 --memory=512m -v "$PWD/ai-bug-detection-task2/results:/app/results" ai-bug-detection-task2:1.0
```

The image build runs the complete unit-test suite. The run trains NB, warms up
both methods, measures paired predictions and latency, and creates:

- `results/smoke/summary.json`: quality metrics, delta F1, project-bootstrap
  interval, guardrails, environment details, and SHA-256 provenance hashes.
- `results/smoke/predictions.csv`: one paired prediction record per test snippet.
- `results/smoke/latencies.csv`: every timed detector call in milliseconds.

If the checkout already exists, use only the last two commands and point the build
at that directory. Docker creates the mounted results directory if necessary.

### Local alternative and tests

From the repository root, no installation step is needed:

```bash
python3 -m unittest discover -s tests -v
python3 src/benchmark.py --config configs/smoke.json
```

Configuration paths are resolved against the repository root, independently of
the shell's current directory. Any invalid input or measurement failure returns
a nonzero exit code. `requirements.txt` documents the absence of third-party
packages; the Python patch and operating-system image are pinned in `Dockerfile`.

## Repository Structure

```text
ai-bug-detection-task2/
├── README.md
├── LICENSE
├── CITATION.cff
├── Dockerfile
├── requirements.txt
├── .gitignore
├── .dockerignore
├── configs/
│   ├── smoke.json
│   └── research.json
├── data/
│   └── sample/
│       ├── README.md
│       ├── train.jsonl
│       └── test.jsonl
├── src/
│   ├── __init__.py
│   └── benchmark.py
├── tests/
│   └── test_benchmark.py
├── docs/
│   ├── methodology_passport.md
│   ├── methodology_passport.pdf
│   ├── data_protocol.md
│   ├── reproducibility.md
│   ├── verification.md
│   └── references.bib
├── notebooks/
│   └── README.md
└── .github/workflows/
    └── verify.yml
```

`results/`, private/raw data, processed research corpora, caches, credentials,
and model binaries are ignored. Only the two tiny fixture files are versioned.
The PDF is the coursework document, not a dataset or a model binary.

## Planned Benchmarks

| Benchmark | Data | Purpose | Measurements | Interpretation |
|---|---|---|---|---|
| W6 smoke check | 8 training and 8 test fixtures; 4 disjoint test fixture groups | Verify instrument and paths | Confusion matrices, F1, latency, RSS, output hashes | Pipeline check only |
| W7 primary pilot | >=50 training and >=100 real reviewed test snippets; >=10 held-out source projects | Compare the two registered methods | F1 difference and paired project-bootstrap 95% interval | Bounded empirical evidence |
| W7 category analysis | Same held-out data, divided into two defect families | Identify where performance differs | Per-category precision, recall, F1 | Descriptive secondary analysis |
| Resource repeat | Same fixed data and host, 30 timed repetitions after 5 warm-up passes | Measure computational cost | p50/p95 latency, detector-only throughput, whole-process peak RSS | Hardware-specific observations |

Actual research results are **not collected yet**. The fixture ground truth has
four positives and four negatives. Its expected static-rule F1 is 1.00 and NB F1
is 0.80 (delta = -0.20). Those deliberately small, template-like examples only
verify measurement. They must not be reported as evidence that one tool is better
on real software. The JSON result explicitly says
`pipeline_validated_only_no_research_conclusion`.

## Experimental Design and Measurement

The independent variable is the detection method. Category is a stratification
factor rather than a randomized treatment. F1 is primary; precision, recall,
false-positive rate, latency, throughput and memory are dependent measures. Both
methods receive the identical held-out source text and ground-truth labels.

NB uses regex-token occurrence counts, training-only vocabulary, empirical class
priors, Laplace smoothing alpha=1.0 and threshold=0.5. Unseen tokens are ignored.
The static detector flags literal numeric zero divisors (`/`, `//`, `%`) and
list/dict/set literals in positional or keyword-only defaults. It does not reason
about reachability, variable values, mutability intent, or interprocedural flow.
Neither implementation sees test labels during prediction.

The instrument rejects overlapping train/test project IDs, snippet IDs, and
identical parsed ASTs. This detects exact structural duplicates, not all renamed
or semantically equivalent clones. Near-duplicate review is required before the
research dataset is frozen. Projects, preprocessing, thresholds, image, host,
single-process execution and repetition counts are kept fixed for each comparison.
Timing order is randomized with seed 42. Training time is reported separately.

F1 = 2TP / (2TP + FP + FN); precision = TP / (TP + FP); recall = TP / (TP + FN);
FPR = FP / (FP + TN). Undefined precision/recall/F1 are assigned 0; undefined FPR
is null and cannot pass its guardrail. The paired bootstrap samples whole test
projects and uses the same sampled rows for both methods. Single-class resamples
are excluded, with the retained draw count reported. The planned research run
uses 10,000 draws; smoke mode uses 2,000.

Registered guardrails are FPR <= 0.10 and p95 detector latency <= 100 ms for each
method, whole-benchmark peak resident memory <= 256 MiB, and zero measurement
errors. Peak RSS covers the shared process including training and resampling;
it is not a per-model memory comparison. Detector throughput excludes data
loading, training, output writing and bootstrap analysis.

An interval wholly above zero supports H1 within the registered scope. Practical
success additionally requires observed delta F1 >=0.05 and every guardrail to
pass. Other intervals leave superiority unestablished; they do not prove equal
performance. Smoke mode never issues a research conclusion, even with a positive
delta or passing guardrails.

## Running the Registered Research Configuration

Prepare `data/processed/train.jsonl` and `data/processed/test.jsonl` using
[`docs/data_protocol.md`](docs/data_protocol.md). Verify licensing, labels,
project separation, near-duplicate removal and the frozen data manifest before
measurement. The loader requires both classes, at least 50 training records,
100 test records, and 10 test projects in research mode. Each record must use
`source: real_reviewed`, contain `provenance` and `label_rationale`, and follow
the sample schema. These checks verify structure; they cannot establish that a
human review actually occurred or that labels are correct.

```bash
docker run --rm --cpus=1 --memory=512m -v "$PWD/data/processed:/app/data/processed:ro" -v "$PWD/results:/app/results" ai-bug-detection-task2:1.0 python src/benchmark.py --config configs/research.json
```

The two methods run offline on the same source snippets. This command intentionally
fails while the real research data is absent. Do not relabel the fixtures as real
data or tune the classifier on the final test set. Larger data, additional defect
families, thresholds or methods require a revised protocol before a new test.

## Reproducibility and Git Workflow

See [`docs/reproducibility.md`](docs/reproducibility.md) for the audit checklist and
team workflow, and [`docs/verification.md`](docs/verification.md) for the actual
local/container checks. Commit code, protocol and tiny sample data; keep outputs
outside version control. For a real run, archive the exact config, data hashes,
code commit, machine details and result files together.

## References

The literature context and bibliography continue Task 1, which reviewed ten
selected studies from 2018-2024. Relevant primary sources include
[BugLab](https://proceedings.neurips.cc/paper/2021/hash/ea96efc03b9a050d895110db8c4af057-Abstract.html)
and the [Real-Vul evaluation](https://arxiv.org/abs/2407.03093). The model follows
the multinomial count-and-smoothing formulation in the
[Naive Bayes documentation](https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes);
scikit-learn is not imported. The rules use the
[Python AST API](https://docs.python.org/3.12/library/ast.html).
Full Task 1 bibliography: [`docs/references.bib`](docs/references.bib).

## Citation and License

```bibtex
@software{baltabay_bissen_2026_bug_detection_pilot,
  author = {Baltabay, Aidos and Bissen, Yerkebulan},
  title = {AI Tools for Software Bug Detection: A Reproducible Pilot Benchmark},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/aidos779/ai-bug-detection-task2}
}
```

Our code, documentation and handmade smoke fixtures are released under the MIT
license. Third-party research data retains its original license and attribution.
