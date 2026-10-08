# Smoke data

`train.jsonl` and `test.jsonl` each contain eight handmade JSONL records (one
record per physical line). Newlines within Python code are JSON-escaped. Each
split has four defect-positive and four defect-negative records, divided into
two defect categories and four fixture project IDs.

These are **synthetic, template-like pipeline fixtures**, not sampled software
projects. Project IDs represent test groups only. Passing project-separation and
AST-duplicate checks does not turn them into realistic held-out software.

| Field | Meaning |
|---|---|
| `id` | Unique snippet identifier |
| `project` | Source project/group, disjoint across splits |
| `code` | Python source string; parsed but never executed |
| `label` | Integer 1 for a known scoped defect, 0 for a reviewed negative |
| `category` | `division_by_zero` or `mutable_default` |
| `source` | `synthetic_smoke` for these fixtures |

Positive division examples have a literal zero divisor. Negative examples have
a nonzero constant. Positive mutable defaults are modified, so state is shared
across calls; negatives allocate a fresh list/dictionary when the default is
`None`. Real labels must be determined independently of detector output.
