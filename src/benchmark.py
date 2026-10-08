"""Train a lexical classifier, compare it with static rules, and record metrics.

Source snippets are parsed as text and are never executed.
Only Python's standard library is used.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("static_rules", "multinomial_nb")
TOKEN_PATTERN = re.compile(r"\b\w+\b|//|[/%=]", re.UNICODE)


def tokens(code):
    return Counter(TOKEN_PATTERN.findall(code.lower()))


def code_hash(code):
    tree = ast.parse(code)
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def load_data(path):
    records, ids, hashes = [], set(), set()
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"id", "project", "code", "label", "category", "source"}
            if not required.issubset(row):
                raise ValueError(f"{path}:{number}: missing fields {required - row.keys()}")
            if type(row["label"]) is not int or row["label"] not in (0, 1):
                raise ValueError(f"{path}:{number}: label must be integer 0 or 1")
            if any(not isinstance(row[k], str) or not row[k].strip()
                   for k in required - {"label"}):
                raise ValueError(f"{path}:{number}: nonempty string fields required")
            if row["category"] not in ("division_by_zero", "mutable_default"):
                raise ValueError(f"{path}:{number}: unsupported defect category")
            digest = code_hash(row["code"])
            if row["id"] in ids or digest in hashes:
                raise ValueError(f"{path}:{number}: duplicate identifier or AST")
            ids.add(row["id"])
            hashes.add(digest)
            records.append(row)
    if not records or {r["label"] for r in records} != {0, 1}:
        raise ValueError("Each split must contain both classes")
    return records


def check_splits(train, test):
    for key in ("id", "project"):
        if {r[key] for r in train} & {r[key] for r in test}:
            raise ValueError(f"Train/test leakage: overlapping {key}")
    if {code_hash(r["code"]) for r in train} & {code_hash(r["code"]) for r in test}:
        raise ValueError("Train/test leakage: overlapping AST")


class MultinomialNB:
    """Count-based multinomial Naive Bayes with empirical class priors."""

    def __init__(self, alpha=1.0):
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.alpha = alpha

    def fit(self, records):
        self.vocabulary = set()
        counts = [Counter(), Counter()]
        classes = Counter()
        for row in records:
            features = tokens(row["code"])
            self.vocabulary.update(features)
            counts[row["label"]].update(features)
            classes[row["label"]] += 1
        if set(classes) != {0, 1}:
            raise ValueError("Training requires both classes")
        self.log_prior = [math.log(classes[c] / len(records)) for c in (0, 1)]
        self.log_likelihood = []
        for c in (0, 1):
            total = sum(counts[c].values()) + self.alpha * len(self.vocabulary)
            self.log_likelihood.append({
                word: math.log((counts[c][word] + self.alpha) / total)
                for word in self.vocabulary
            })
        return self

    def score(self, code):
        features = tokens(code)
        scores = [self.log_prior[c] + sum(
            count * self.log_likelihood[c][word]
            for word, count in features.items() if word in self.vocabulary
        ) for c in (0, 1)]
        difference = scores[0] - scores[1]
        if difference >= 0:
            value = math.exp(-difference)
            return value / (1 + value)
        return 1 / (1 + math.exp(difference))


def static_score(code):
    """Flag literal zero division or a mutable literal used as a default."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)):
                if node.right.value == 0:
                    return 1.0
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = node.args.defaults + [d for d in node.args.kw_defaults if d is not None]
            if any(isinstance(d, (ast.List, ast.Dict, ast.Set)) for d in defaults):
                return 1.0
    return 0.0


def classification_metrics(labels, predictions):
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Labels and predictions must have the same nonzero length")
    if any(type(x) is not int or x not in (0, 1) for x in labels + predictions):
        raise ValueError("Metrics require binary integer labels and predictions")
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None}


def percentile(values, probability):
    if not values or not 0 <= probability <= 1:
        raise ValueError("Invalid percentile arguments")
    ordered = sorted(values)
    point = (len(ordered) - 1) * probability
    left, right = math.floor(point), math.ceil(point)
    return ordered[left] + (ordered[right] - ordered[left]) * (point - left)


def paired_project_bootstrap(test, predictions, draws, seed):
    groups = defaultdict(list)
    for index, row in enumerate(test):
        groups[row["project"]].append(index)
    projects = sorted(groups)
    if len(projects) < 2:
        raise ValueError("Bootstrap requires at least two test projects")
    rng, deltas = random.Random(seed), []
    for _ in range(draws):
        indices = [i for project in rng.choices(projects, k=len(projects)) for i in groups[project]]
        labels = [test[i]["label"] for i in indices]
        # Single-class resamples do not identify the binary detection endpoint.
        if set(labels) != {0, 1}:
            continue
        metrics = {m: classification_metrics(labels, [predictions[m][i] for i in indices])
                   for m in METHODS}
        deltas.append(metrics["multinomial_nb"]["f1"] - metrics["static_rules"]["f1"])
    if len(deltas) < max(20, draws // 2):
        raise ValueError("Too few valid bootstrap draws; revise sampling or collect more data")
    return {"method": "paired project-cluster percentile bootstrap",
            "confidence_level": 0.95, "requested_draws": draws, "valid_draws": len(deltas),
            "lower": percentile(deltas, 0.025), "upper": percentile(deltas, 0.975)}


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config):
    if config["mode"] not in ("smoke", "research"):
        raise ValueError("mode must be smoke or research")
    for name in ("repetitions", "bootstrap_draws"):
        if type(config[name]) is not int or config[name] < (100 if name == "bootstrap_draws" else 1):
            raise ValueError(f"Invalid {name}")
    if type(config["warmup_passes"]) is not int or config["warmup_passes"] < 0:
        raise ValueError("Invalid warmup_passes")
    if not 0 < config["decision_threshold"] < 1:
        raise ValueError("decision_threshold must be between 0 and 1")
    if not math.isfinite(config["alpha"]) or config["alpha"] <= 0:
        raise ValueError("alpha must be finite and positive")
    g = config["guardrails"]
    if not 0 <= g["max_false_positive_rate"] <= 1:
        raise ValueError("Invalid false positive guardrail")
    for name in ("max_p95_latency_ms", "max_process_peak_rss_mib"):
        if not math.isfinite(g[name]) or g[name] <= 0:
            raise ValueError(f"Invalid {name}")
    if config["practical_f1_gain"] < 0:
        raise ValueError("practical_f1_gain cannot be negative")


def run(config_path):
    config_path = resolve(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    train_path, test_path = resolve(config["train_path"]), resolve(config["test_path"])
    train, test = load_data(train_path), load_data(test_path)
    check_splits(train, test)
    if config["mode"] == "research":
        if len(train) < 50 or len(test) < 100 or len({r["project"] for r in test}) < 10:
            raise ValueError("Research mode requires >=50 training / >=100 test snippets and >=10 test projects")
        if any(r["source"] != "real_reviewed" or not r.get("provenance")
               or not r.get("label_rationale") for r in train + test):
            raise ValueError("Research mode requires real_reviewed data, provenance, and label rationale")
    started = time.perf_counter_ns()
    model = MultinomialNB(config["alpha"]).fit(train)
    fit_ms = (time.perf_counter_ns() - started) / 1e6
    scorers = {"static_rules": static_score, "multinomial_nb": model.score}
    for _ in range(config["warmup_passes"]):
        for row in test:
            for scorer in scorers.values():
                scorer(row["code"])
    timings = {m: [] for m in METHODS}
    scores = {m: [None] * len(test) for m in METHODS}
    rng = random.Random(config["seed"])
    for repetition in range(config["repetitions"]):
        indices = list(range(len(test)))
        rng.shuffle(indices)
        for i in indices:
            order = list(METHODS)
            rng.shuffle(order)
            for method in order:
                t0 = time.perf_counter_ns()
                score = scorers[method](test[i]["code"])
                timings[method].append((time.perf_counter_ns() - t0) / 1e6)
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Invalid classifier score")
                if repetition and score != scores[method][i]:
                    raise ValueError("Nondeterministic predictions detected")
                scores[method][i] = score
    predictions = {m: [int(s >= config["decision_threshold"]) for s in scores[m]] for m in METHODS}
    labels = [r["label"] for r in test]
    metrics = {}
    for method in METHODS:
        metrics[method] = classification_metrics(labels, predictions[method])
        metrics[method].update({"p50_latency_ms": percentile(timings[method], 0.5),
                               "p95_latency_ms": percentile(timings[method], 0.95),
                               "throughput_snippets_per_second": len(timings[method]) / (sum(timings[method]) / 1000)})
    ci = paired_project_bootstrap(test, predictions, config["bootstrap_draws"], config["seed"])
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mib = rss / (1024 ** 2 if sys.platform == "darwin" else 1024)
    guards = {m: {"false_positive_rate": metrics[m]["false_positive_rate"] is not None and
                  metrics[m]["false_positive_rate"] <= config["guardrails"]["max_false_positive_rate"],
                  "p95_latency": metrics[m]["p95_latency_ms"] <= config["guardrails"]["max_p95_latency_ms"]}
              for m in METHODS}
    guards["process_peak_rss"] = rss_mib <= config["guardrails"]["max_process_peak_rss_mib"]
    guards["measurement_errors"] = True  # Any failed measurement aborts without a success result.
    all_pass = guards["process_peak_rss"] and all(all(guards[m].values()) for m in METHODS)
    delta = metrics["multinomial_nb"]["f1"] - metrics["static_rules"]["f1"]
    if config["mode"] == "smoke":
        interpretation = "pipeline_validated_only_no_research_conclusion"
    elif ci["lower"] > 0:
        interpretation = "evidence_supports_h1_within_registered_scope"
    else:
        interpretation = "superiority_not_established"
    by_category = {}
    for category in sorted({r["category"] for r in test}):
        ix = [i for i, r in enumerate(test) if r["category"] == category]
        by_category[category] = {m: classification_metrics([labels[i] for i in ix], [predictions[m][i] for i in ix])
                                 for m in METHODS}
    result = {"mode": config["mode"], "created_utc": datetime.now(timezone.utc).isoformat(),
              "train_rows": len(train), "test_rows": len(test),
              "test_projects": len({r["project"] for r in test}),
              "metrics": metrics, "delta_f1": delta, "delta_f1_ci95": ci,
              "per_category": by_category, "guardrails": guards,
              "all_guardrails_pass": all_pass, "interpretation": interpretation,
              "practical_success": config["mode"] == "research" and ci["lower"] > 0
                  and delta >= config["practical_f1_gain"] and all_pass,
              "model_fit_ms": fit_ms, "benchmark_process_peak_rss_mib": rss_mib,
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "machine": platform.machine(), "processor": platform.processor()},
              "hashes": {"train_sha256": file_sha256(train_path), "test_sha256": file_sha256(test_path),
                         "config_sha256": file_sha256(config_path), "code_sha256": file_sha256(__file__)},
              "config": config}
    out = resolve(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "project", "category", "label", "static_prediction", "nb_prediction", "nb_score"])
        for i, row in enumerate(test):
            writer.writerow([row["id"], row["project"], row["category"], row["label"],
                             predictions["static_rules"][i], predictions["multinomial_nb"][i], scores["multinomial_nb"][i]])
    with (out / "latencies.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "call_index", "latency_ms"])
        for m in METHODS:
            writer.writerows((m, i, value) for i, value in enumerate(timings[m]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.json")
    args = parser.parse_args()
    try:
        result = run(args.config)
    except (ValueError, KeyError, OSError, SyntaxError, TypeError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"mode": result["mode"], "delta_f1": result["delta_f1"],
                      "interpretation": result["interpretation"],
                      "output_dir": str(resolve(result["config"]["output_dir"]))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
