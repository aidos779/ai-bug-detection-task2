"""Collect one paired Task 3 run from the frozen Task 2 sources (stdlib only)."""
import argparse
import ast
import hashlib
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from . import benchmark as base
else:
    import benchmark as base

ROOT = base.ROOT
UPSTREAM = "374dea01bd5983d9081e94f8a8dccb47168ab94a"
# A later change to these files defines a different workload and needs a new snapshot.
FROZEN = {
    "Dockerfile": "20a97aa54efe665d3563b2912568de6182499e87915d7f2b569a2557e52b257e",
    "src/benchmark.py": "ff4c534d911449806fcada56428d11866b7950e757f766a57e75c859cf388837",
    "tests/test_benchmark.py": "59f08b93cc11c8cb84648fadc1b5f7a2a8a05c8c0b97804e2779193957920dd8",
    "configs/research.json": "b13b706bffbee32b6363ff294cef86b6594b7a003840c7d465f73b92f207e8fa",
    "configs/smoke.json": "6d9ca2fe013ce9ced5db02f78fbab1ad716a8ea7c1207fe9da7167a06121e60d",
    "data/sample/test.jsonl": "fc61aaa9933aff7f54c571ba68b661c2f17ece0ada7ef2cbb4ed50b399d5f5a7",
    "data/sample/train.jsonl": "b81b5778fcea0b4af6dab82aeb5d50bdbadec6af4cc259e31db6a38055f84247",
}
RAW_PATH = ROOT / "data/raw/test_run.jsonl"
MODEL_PATH = ROOT / "models/task3_nb.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def frozen_hashes():
    hashes = {name: base.file_sha256(ROOT / name) for name in FROZEN}
    if hashes != FROZEN:
        changed = [name for name in hashes if hashes[name] != FROZEN[name]]
        raise ValueError(f"Inputs differ from frozen Task 2 snapshot: {changed}")
    return dict(hashes, **{"src/collect_task3.py": base.file_sha256(__file__)})


def inputs_for_run(test):
    """Return actual input text and source metadata. Never execute the snippets."""
    records = []
    for line, row in enumerate(test, 1):
        code = row["code"]
        records.append(dict(snippet_id=row["id"], corpus="fixture", project=row["project"],
                            label=row["label"], category=row["category"], code=code,
                            source_path="data/sample/test.jsonl", source_start_line=line,
                            source_end_line=line, source_name=ast.parse(code).body[0].name))
    function_types = (ast.FunctionDef, ast.AsyncFunctionDef)
    for path in ("src/benchmark.py", "tests/test_benchmark.py"):
        text = (ROOT / path).read_text(encoding="utf-8")
        candidates = []
        for item in ast.parse(text).body:
            if isinstance(item, ast.ClassDef):
                items, prefix = item.body, item.name + "."
            else:
                items, prefix = [item], ""
            for node in items:
                if not isinstance(node, function_types):
                    continue
                name = prefix + node.name
                code = textwrap.dedent(ast.get_source_segment(text, node, padded=True))
                ast.parse(code)
                candidates.append(dict(snippet_id=f"{path}#{name}", corpus="repository",
                                       project="aidos779/ai-bug-detection-task2",
                                       category=None, label=None, code=code, source_path=path,
                                       source_start_line=node.lineno, source_end_line=node.end_lineno,
                                       source_name=name))
        # Six size ranks per file, including both extremes; tie-break by source order.
        candidates.sort(key=lambda r: (len(r["code"].encode("utf-8")), r["source_start_line"]))
        records.extend(candidates[i * (len(candidates) - 1) // 5] for i in range(6))
    if len({r["snippet_id"] for r in records}) != len(records):
        raise ValueError("Duplicate workload ID")
    for row in records:
        raw_bytes = row["code"].encode("utf-8")
        row.update(input_bytes=len(raw_bytes), input_loc=len(row["code"].splitlines()),
                   code_sha256=digest(raw_bytes))
    return records


def model_bytes(model, cfg):
    artifact = dict(format_version=1, algorithm="multinomial_nb", tokenizer=base.TOKEN_PATTERN.pattern,
                    lowercase=True, source="synthetic_smoke", upstream_commit=UPSTREAM,
                    training_sha256=FROZEN[cfg["train_path"]], alpha=model.alpha,
                    decision_threshold=cfg["decision_threshold"], classes=[0, 1],
                    log_prior=model.log_prior, log_likelihood=model.log_likelihood)
    return (json.dumps(artifact, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def load_model(path):
    """Read the portable JSON artifact and return (Task 2 scorer, decision threshold)."""
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if (artifact["format_version"] != 1 or artifact["algorithm"] != "multinomial_nb"
            or artifact["classes"] != [0, 1] or artifact["tokenizer"] != base.TOKEN_PATTERN.pattern
            or artifact["lowercase"] is not True):
        raise ValueError("Unsupported model format")
    model = base.MultinomialNB(artifact["alpha"])
    model.log_prior = artifact["log_prior"]
    model.log_likelihood = artifact["log_likelihood"]
    model.vocabulary = set(model.log_likelihood[0])
    threshold = artifact["decision_threshold"]
    if (len(model.log_prior) != 2 or len(model.log_likelihood) != 2
            or model.vocabulary != set(model.log_likelihood[1]) or not 0 < threshold < 1
            or not all(math.isfinite(x) for x in model.log_prior)
            or not all(math.isfinite(x) for c in model.log_likelihood for x in c.values())):
        raise ValueError("Invalid model parameters")
    return model, threshold


def read_limit(path):
    p = Path(path)
    return p.read_text(encoding="utf-8").strip() if p.is_file() else None


def probe_clock():
    # Back-to-back reads measure observable increments, not absolute timer accuracy.
    minimum = None
    for _ in range(50):
        reads = [time.perf_counter_ns() for _ in range(1001)]
        step = min((b - a for a, b in zip(reads, reads[1:]) if b > a), default=None)
        if step is not None and (minimum is None or step < minimum):
            minimum = step
    return {"adjacent_pairs": 50_000, "min_positive_increment_ns": minimum}


def environment():
    info = dict(python=platform.python_version(), system=platform.system(), release=platform.release(),
                machine=platform.machine(), cpu_model=platform.processor() or None,
                logical_cpus=os.cpu_count(), hash_seed=os.environ.get("PYTHONHASHSEED"),
                timer="time.perf_counter_ns", clock_resolution_s=time.get_clock_info("perf_counter").resolution)
    info["clock_probe"] = probe_clock()
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    info["dockerfile_base_image"] = next(line.split()[1] for line in dockerfile.splitlines()
                                         if line.startswith("FROM "))
    # The host can pass docker image inspect's immutable ID; a process cannot infer it from a tag.
    info["runtime_image_id_reported_by_launcher"] = os.environ.get("TASK3_IMAGE_ID")
    if sys.platform == "darwin":
        info["cpu_model"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                                    text=True).strip()
    if sys.platform.startswith("linux"):
        for line in (read_limit("/proc/cpuinfo") or "").splitlines():
            if line.startswith(("model name", "Hardware")) and ":" in line:
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
        info["cpu_max_v2"] = read_limit("/sys/fs/cgroup/cpu.max")
        info["memory_max_bytes_v2"] = read_limit("/sys/fs/cgroup/memory.max")
        info["cpu_quota_us_v1"] = read_limit("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        info["cpu_period_us_v1"] = read_limit("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        info["memory_limit_bytes_v1"] = read_limit("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if hasattr(os, "sched_getaffinity"):
            info["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    if hasattr(os, "sysconf"):
        info["os_visible_ram_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return info


def measure(inputs, scorers, cfg, run_id):
    rng = random.Random(cfg["seed"])
    events, scores = [], {}
    threshold = cfg["decision_threshold"]
    for phase, passes in (("warmup", cfg["warmup_passes"]), ("benchmark", cfg["repetitions"])):
        for repetition in range(1, passes + 1):
            order = inputs.copy()
            rng.shuffle(order)
            for row in order:
                code, snippet_id = row["code"], row["snippet_id"]
                methods = list(base.METHODS)
                rng.shuffle(methods)
                for method in methods:
                    scorer = scorers[method]
                    error = None
                    start = time.perf_counter_ns()
                    try:
                        score = scorer(code)
                    except Exception as exc:
                        elapsed = time.perf_counter_ns() - start
                        error = f"{type(exc).__name__}: {exc}"
                    else:
                        elapsed = time.perf_counter_ns() - start
                        if not math.isfinite(score) or not 0 <= score <= 1 or elapsed <= 0:
                            error = "Invalid score or timer interval"
                        elif score != scores.setdefault((method, snippet_id), score):
                            error = "Score changed between identical calls"
                    events.append(dict(record_type="call", run_id=run_id, phase=phase,
                                       repetition=repetition, method=method, snippet_id=snippet_id,
                                       elapsed_ns=elapsed, score=None if error else score,
                                       prediction=None if error else int(score >= threshold),
                                       status="error" if error else "ok", error=error))
                    if error:
                        return events
    return events


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            tmp.write(data)
            tmp.close()
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)


def collect(raw_path, model_path):
    started = utc()
    hashes = frozen_hashes()
    cfg = json.loads((ROOT / "configs/smoke.json").read_text())
    draws = json.loads((ROOT / "configs/research.json").read_text())["bootstrap_draws"]
    base.validate_config(cfg)
    train = base.load_data(ROOT / cfg["train_path"])
    test = base.load_data(ROOT / cfg["test_path"])
    base.check_splits(train, test)
    inputs = inputs_for_run(test)
    run_id = str(uuid.uuid4())
    env = environment()
    model = base.MultinomialNB(cfg["alpha"])
    start = time.perf_counter_ns()
    model.fit(train)
    fit_ms = (time.perf_counter_ns() - start) / 1_000_000
    snapshot = model_bytes(model, cfg)
    header = dict(record_type="run", schema_version=1, run_id=run_id, started_utc=started,
                  upstream_commit=UPSTREAM, seed=cfg["seed"], warmup_passes=cfg["warmup_passes"],
                  repetitions=cfg["repetitions"], alpha=cfg["alpha"], decision_threshold=cfg["decision_threshold"],
                  bootstrap_draws=draws, guardrails=cfg["guardrails"], practical_f1_gain=cfg["practical_f1_gain"],
                  model_fit_ms=fit_ms, environment=env, hashes=hashes, model_sha256=digest(snapshot),
                  inputs=[{k: v for k, v in row.items() if k != "code"} for row in inputs])
    events = measure(inputs, {"static_rules": base.static_score, "multinomial_nb": model.score}, cfg, run_id)
    errors = sum(e["status"] == "error" for e in events)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    end = dict(record_type="end", run_id=run_id, finished_utc=utc(), status="error" if errors else "ok",
               calls=len(events), measurement_errors=errors,
               peak_rss_mib=peak / (1024**2 if sys.platform == "darwin" else 1024))
    # RSS covers imports, input preparation, fit, warm-up, inference and buffered events,
    # through the last call. Serialization and downstream notebook analysis follow this sample.
    raw = "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                  for row in [header, *events, end]).encode("utf-8")
    if errors:
        failed = raw_path.with_name(f"{raw_path.stem}.failed-{run_id}.jsonl")
        atomic_write(failed, raw)
        raise RuntimeError(f"Measurement failed; diagnostic run at {failed}")
    atomic_write(model_path, snapshot)
    atomic_write(raw_path, raw)
    print(f"{run_id}: {len(events)} paired-method calls; log {raw_path}; model {model_path}")


def verify_model(raw_path, model_path):
    """Replay the serialized model on every distinct input; verify all archived NB calls."""
    model, threshold = load_model(model_path)
    with raw_path.open(encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if header["hashes"] != frozen_hashes() or header["upstream_commit"] != UPSTREAM:
            raise ValueError("Frozen source or collector SHA-256 differs from the run")
        if header["model_sha256"] != base.file_sha256(model_path):
            raise ValueError("Model SHA-256 differs from the run")
        inputs = inputs_for_run(base.load_data(ROOT / "data/sample/test.jsonl"))
        metadata = [{k: v for k, v in row.items() if k != "code"} for row in inputs]
        if metadata != header["inputs"] or threshold != header["decision_threshold"]:
            raise ValueError("Input provenance or decision threshold differs from the run")
        expected = {row["snippet_id"]: model.score(row["code"]) for row in inputs}
        counts = {key: 0 for key in expected}
        for line in stream:
            row = json.loads(line)
            if row["record_type"] != "call" or row["method"] != "multinomial_nb":
                continue
            score = expected[row["snippet_id"]]
            if (row["status"] != "ok" or not math.isclose(score, row["score"], rel_tol=1e-12)
                    or int(score >= threshold) != row["prediction"]):
                raise ValueError(f"Model replay mismatch: {row['snippet_id']}")
            counts[row["snippet_id"]] += 1
    if set(counts.values()) != {header["warmup_passes"] + header["repetitions"]}:
        raise ValueError("Incomplete model replay")
    print(f"Model replay verified for {len(expected)} inputs and {sum(counts.values())} NB calls.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--verify-model", action="store_true", help="check saved predictions without remeasuring")
    args = parser.parse_args()
    if args.raw.resolve() == args.model.resolve():
        parser.error("Raw log and model must use separate files")
    (verify_model if args.verify_model else collect)(args.raw, args.model)
