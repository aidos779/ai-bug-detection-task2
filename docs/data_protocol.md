# Data collection and annotation protocol

## Scope and sampling

The evaluation unit is one independently labelled Python function/snippet with
the context needed to judge one of two registered defect families: division by
zero or unintended state sharing through a mutable default argument. A negative
label means no confirmed defect in that registered family; it does not assert
that the entire function is correct.

For W7, collect a pilot of at least 50 training snippets and 100 test snippets
from at least 10 held-out source projects. Target 25 positive and 25 negative
training records, and 50 positive and 50 negative test records. Within the test
set, target 25 positives and 25 negatives per family. Freeze counts before
measurement and report any departures. The 50% test prevalence is a deliberate
case-control design, not an estimate of defect prevalence in deployed software.
Precision and F1 are interpreted at this selected class composition only.

Use traceable open-source bug-fix commits, regression tests and reviewed
counterexamples. Record source project, commit, source path, licensing/attribution,
and the label evidence in an accompanying local manifest. Do not invent commit
links, label unknown snippets as clean, or infer ground truth from either tested
detector. If insufficient verified data is available, remain in smoke mode.

## Ground truth

For each positive, retain a concrete failure explanation supported by a fix and/or
a regression test. A mutable default is not automatically a bug: verify that
shared state is unintended and observable across calls. For a negative, review
the relevant execution context and intended behaviour. Dormant/unreachable zero
division and intentionally shared defaults are useful counterexamples and may
expose false alarms in the limited static rules.

The two authors independently annotate candidate snippets, then resolve
disagreements before the test set is frozen. Record original labels,
disagreements and the resolution in the private annotation log. This is a
planned collection procedure; no real snippets have been reviewed yet.

## Separation and preprocessing

Assign whole source projects to train or test before fitting any model. A
vulnerable function and its patched counterpart must remain in the same split.
Remove exact AST duplicates and review renamed/template/near-duplicate clones
across projects. The program detects exact AST identity only. The manual review
is necessary because project separation alone does not prevent copied code.

Freeze a consistent context-extraction rule. Remove defect labels, bug-fix
messages, post-fix annotations and accidental label-correlated comments from
both splits. Keep original files and hashes in the private manifest. Do not
change extraction or hyperparameters after inspecting final test performance.
Alpha=1.0 and threshold=0.5 are registered; any later tuning uses training-only
validation and requires a new pre-test protocol.

## Required JSONL schema

Use all sample fields plus the real-data provenance fields below. Each record
must be syntactically valid Python, have a unique ID, and use label 0 or 1.

| Field | Required value or content |
|---|---|
| `id` | Unique snippet identifier; no labels encoded in IDs |
| `project` | Stable source-project ID, disjoint across splits |
| `code` | Frozen source text, including the agreed context |
| `label` | Integer 0 or 1 from independent review |
| `category` | `division_by_zero` or `mutable_default` |
| `source` | `real_reviewed` |
| `provenance` | Real repository/commit/path reference or a manifest key |
| `label_rationale` | Concrete reviewed reason, including evidence reference |

The loader checks required fields, labels, syntax, duplicate ASTs, overlap and
minimum sample size. Provenance strings are not proof of annotation quality;
the accompanying source and annotation manifests must be audited by the authors.

## Frozen experiment and analysis

Save SHA-256 hashes for train/test JSONL and the full config. Record the source
manifest revision, image digest, git commit and host specification. Use one
process, the same host/image, five warm-up passes and 30 timed repetitions, with
seed 42 determining call order and bootstrap draws. Compare the same held-out
snippets, retain all predictions, and do not exclude inconvenient false alarms.

Report confusion counts, precision/recall/F1/FPR, the absolute paired F1
difference, a 95% project-cluster percentile-bootstrap interval (10,000 draws),
per-family results and every guardrail. Resamples missing a class are omitted
and the valid draw count is reported. The finite, small-project bootstrap is
exploratory evidence rather than a guarantee of a calibrated significance test.

Latency excludes fitting/loading/resampling; model-fit time is a separate field.
RSS is the process-wide high-water mark, including analysis, and cannot support
a claim about which model independently uses less memory. For a future memory
comparison, run methods in separately isolated processes under a new protocol.

## Data storage

Store raw and reviewed real data outside Git (`data/raw/` and `data/processed/`
are ignored). Keep a small text manifest with source attribution, versions,
licensing and hashes; do not redistribute code without checking its source
terms. Use an appropriate data archive for the final reproducible study. The
included handmade fixtures and our code/documentation use MIT.
