"""Validate a Task 3 JSONL run and summarize its two distinct workloads."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from . import benchmark as b
else:
    import benchmark as b


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_run(path):
    """Return metadata and enriched call rows. Reject mixed, corrupt or partial runs."""
    with Path(path).open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    _require(len(records) >= 3, "Missing run header, calls or footer")
    meta, end = records[0], records[-1]
    _require(meta.get("record_type") == "run" and meta.get("schema_version") == 1,
             "Unsupported run header")
    _require(end.get("record_type") == "end", "Missing run footer")
    _require(bool(meta["run_id"]) and all(r["run_id"] == meta["run_id"] for r in records),
             "Mixed run identifiers")
    inputs = pd.DataFrame(meta["inputs"]).set_index("snippet_id", verify_integrity=True)
    _require(set(inputs["corpus"]) == {"fixture", "repository"}, "Expected both input corpora")
    for row in meta["inputs"]:
        _require(type(row["input_bytes"]) is int and row["input_bytes"] > 0,
                 "Invalid input size")
        if row["corpus"] == "fixture":
            _require(type(row["label"]) is int and row["label"] in (0, 1)
                     and row["category"] in ("division_by_zero", "mutable_default"),
                     "Invalid fixture label or category")
        else:
            _require(row["label"] is None and row["category"] is None,
                     "Repository functions must be unlabelled")
    phase_counts = {"warmup": meta["warmup_passes"], "benchmark": meta["repetitions"]}
    _require(all(type(n) is int and n > 0 for n in phase_counts.values()),
             "Invalid repetition counts")
    # No materialized Cartesian product. Unique valid cells plus the planned count
    # check that every (phase, repetition, method, input) occurs exactly once.
    seen = set()
    entered_benchmark = False
    for i, row in enumerate(records[1:-1], 2):
        phase, repetition = row["phase"], row["repetition"]
        _require(row["record_type"] == "call" and phase in phase_counts
                 and type(repetition) is int and 1 <= repetition <= phase_counts[phase]
                 and row["method"] in b.METHODS and row["snippet_id"] in inputs.index,
                 f"Unexpected call identity on line {i}")
        _require(not (entered_benchmark and phase == "warmup"), "Warm-up after benchmark")
        entered_benchmark |= phase == "benchmark"
        key = (phase, repetition, row["method"], row["snippet_id"])
        _require(key not in seen, f"Duplicate call {key}")
        seen.add(key)
        _require(type(row["elapsed_ns"]) is int and row["elapsed_ns"] >= 0,
                 f"Invalid elapsed time on line {i}")
        if row["status"] == "ok":
            score = row["score"]
            _require(row["elapsed_ns"] > 0 and type(score) in (int, float) and 0 <= score <= 1
                     and type(row["prediction"]) is int
                     and row["prediction"] == int(score >= meta["decision_threshold"])
                     and row["error"] is None, f"Inconsistent successful call on line {i}")
        else:
            _require(row["status"] == "error" and bool(row["error"])
                     and row["score"] is None and row["prediction"] is None,
                     f"Inconsistent failed call on line {i}")
    errors = sum(row["status"] == "error" for row in records[1:-1])
    _require(end["measurement_errors"] == errors and end["calls"] == len(seen)
             and end["status"] == ("error" if errors else "ok"),
             "Footer disagrees with observed attempts or measurement errors")
    expected = len(inputs) * len(b.METHODS) * sum(phase_counts.values())
    _require(len(seen) == expected,
             f"Incomplete run: {len(seen)}/{expected} attempts, "
             f"{errors} measurement errors ({errors / max(1, len(seen)):.2%} of observed attempts)")
    _require(math.isfinite(end["peak_rss_mib"]) and end["peak_rss_mib"] > 0,
             "Invalid process peak RSS")

    calls = pd.DataFrame(records[1:-1]).join(inputs, on="snippet_id", validate="many_to_one")
    successful = calls["status"].eq("ok")
    scores = calls[successful].groupby(["method", "snippet_id"])["score"]
    _require(not (scores.nunique() > 1).any(), "Scores varied across calls on the same input")
    calls["latency_ms"] = calls["elapsed_ns"] / 1_000_000
    calls["is_outlier"] = False
    # Flag spikes within each input/method pair to avoid mistaking large inputs for noise.
    kept = successful & calls["phase"].eq("benchmark")
    groups = calls[kept].groupby(["method", "snippet_id"])["latency_ms"]
    q1 = groups.transform("quantile", q=0.25)
    q3 = groups.transform("quantile", q=0.75)
    margin = 1.5 * (q3 - q1)
    calls.loc[kept, "is_outlier"] = ~calls.loc[kept, "latency_ms"].between(q1 - margin, q3 + margin)
    return {**meta, "end": end}, calls


def _block_intervals(matrix, draws, seed):
    """Resample whole repeat passes; each pass retains its complete mix of inputs."""
    rng = np.random.default_rng(seed)
    medians, rates = [], []
    # Use the same seed and ordered pass IDs for every method, preserving pairing.
    for start in range(0, draws, 128):
        indices = rng.integers(len(matrix), size=(min(128, draws - start), len(matrix)))
        sample = matrix[indices].reshape(len(indices), -1)
        medians.extend(np.median(sample, axis=1))
        rates.extend(1000 * sample.shape[1] / sample.sum(axis=1))
    return np.quantile([medians, rates], [0.025, 0.975], axis=1).T


def _performance(meta, group):
    ok = group[group["status"].eq("ok")]
    errors = len(group) - len(ok)
    values = ok["latency_ms"]
    row = {"attempts": len(group), "successes": len(ok), "measurement_errors": errors,
           "error_rate": errors / len(group), "zero_errors_pass": errors == 0,
           "outlier_count": int(ok["is_outlier"].sum()),
           "n_inputs": group["snippet_id"].nunique(), "n_passes": group["repetition"].nunique()}
    for q in (25, 50, 75, 95, 99):
        row[f"latency_p{q}_ms"] = values.quantile(q / 100)
    row.update(latency_mean_ms=values.mean(), latency_sd_ms=values.std(ddof=1),
               latency_iqr_ms=row["latency_p75_ms"] - row["latency_p25_ms"],
               throughput_snippets_per_s=1000 * len(ok) / values.sum() if len(ok) else np.nan,
               p95_within_100ms=bool(row["latency_p95_ms"] <= 100))
    # Failed attempts remain in the denominator of error_rate. Interval estimates
    # require complete passes; never bootstrap a selectively censored workload.
    intervals = np.full((2, 2), np.nan)
    if not errors:
        matrix = ok.pivot(index="repetition", columns="snippet_id", values="latency_ms")
        intervals = _block_intervals(matrix.to_numpy(), meta["bootstrap_draws"], meta["seed"])
        # These describe variability between observed passes. The overall rate above
        # is a ratio of totals rather than an unweighted mean of per-pass rates.
        pass_rates = 1000 * matrix.shape[1] / matrix.sum(axis=1)
        row["pass_throughput_mean_per_s"] = pass_rates.mean()
        row["pass_throughput_sd_per_s"] = pass_rates.std(ddof=1)
        for q in (50, 95, 99):
            row[f"pass_throughput_p{q}_per_s"] = pass_rates.quantile(q / 100)
    row["latency_p50_ci_low_ms"], row["latency_p50_ci_high_ms"] = intervals[0]
    row["throughput_ci_low_snippets_per_s"], row["throughput_ci_high_snippets_per_s"] = intervals[1]
    return row


def summarize(meta, calls):
    """Return corpus/method summaries, fixture category scores and the paired F1 contrast."""
    evaluation = calls[calls["phase"].eq("benchmark")]
    fixtures = evaluation[evaluation["corpus"].eq("fixture")]
    # One fixed prediction per input, independent of the number of timed repeats.
    unique = fixtures[fixtures["status"].eq("ok")].drop_duplicates(["snippet_id", "method"])
    quality, categories = {}, []
    for method, group in unique.groupby("method"):
        if len(group) != fixtures["snippet_id"].nunique():
            continue  # Missing predictions prevent a quality comparison on the fixed set.
        def metrics(frame):
            return b.classification_metrics(frame["label"].astype(int).tolist(),
                                            frame["prediction"].astype(int).tolist())
        overall = metrics(group)
        quality[method] = {**overall, "quality_n": len(group),
                           "fpr_within_0_10": overall["false_positive_rate"] is not None
                           and overall["false_positive_rate"] <= 0.10}
        for category, subset in group.groupby("category"):
            categories.append({"method": method, "category": category, "n": len(subset),
                               **metrics(subset)})

    delta = {"delta_f1": None, "lower": None, "upper": None, "valid_draws": 0,
             "requested_draws": meta["bootstrap_draws"]}
    if len(quality) == len(b.METHODS):
        rows = unique.drop_duplicates("snippet_id").set_index("snippet_id").sort_index()
        paired = unique.pivot(index="snippet_id", columns="method", values="prediction")
        predictions = {m: paired.loc[rows.index, m].astype(int).tolist() for m in b.METHODS}
        test = rows[["project", "label"]].astype({"label": int}).to_dict("records")
        delta = b.paired_project_bootstrap(test, predictions, meta["bootstrap_draws"], meta["seed"])
        delta["delta_f1"] = quality["multinomial_nb"]["f1"] - quality["static_rules"]["f1"]

    summary = []
    for (corpus, method), group in evaluation.groupby(["corpus", "method"], sort=False):
        row = {"corpus": corpus, "method": method, **_performance(meta, group)}
        if corpus == "fixture" and method in quality:
            row.update(quality[method])
        summary.append(row)
    return pd.DataFrame(summary), pd.DataFrame(categories), delta


def write_outputs(directory, summary, categories):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summary.to_csv(directory / "metrics_summary.csv", index=False)
    categories.to_csv(directory / "fixture_categories.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=b.ROOT / "data/raw/test_run.jsonl")
    parser.add_argument("--out", type=Path, default=b.ROOT / "data/processed")
    args = parser.parse_args()
    meta, calls = load_run(args.raw)
    summary, categories, delta = summarize(meta, calls)
    write_outputs(args.out, summary, categories)
    print(summary.to_string(index=False))
    print("Fixture F1 contrast:", json.dumps(delta))
    return int(meta["end"]["measurement_errors"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
