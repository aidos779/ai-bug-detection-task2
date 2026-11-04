"""Regression tests for parsing, units and the two sampling units in Task 3."""
from importlib.util import find_spec
import json
from pathlib import Path
import tempfile
import unittest

# The original Task 2 Docker image deliberately ships only the standard library.
if find_spec("numpy") is None or find_spec("pandas") is None:
    raise unittest.SkipTest("Task 3 analysis requires NumPy and pandas")

from src import process_data as p


def sample_log():
    """Known answers for tests only; never used in the published benchmark."""
    inputs = []
    for i in range(5):
        labelled = i < 4
        inputs.append(dict(snippet_id=str(i), corpus="fixture" if labelled else "repository",
                           project=str(i // 2), category="mutable_default" if labelled else None,
                           label=i % 2 if labelled else None, input_bytes=8))
    header = dict(record_type="run", schema_version=1, run_id="test", warmup_passes=1,
                  repetitions=6, seed=42, decision_threshold=0.5, bootstrap_draws=100,
                  inputs=inputs)
    records = [header]
    for phase, repeats in (("warmup", 1), ("benchmark", 6)):
        for repetition in range(1, repeats + 1):
            for row in inputs:
                for method in p.b.METHODS:
                    score = float(row["label"] or 0) if method == "static_rules" else 0.
                    records.append(dict(record_type="call", run_id="test", phase=phase,
                                        repetition=repetition, method=method,
                                        snippet_id=row["snippet_id"], score=score,
                                        prediction=int(score >= .5), status="ok", error=None,
                                        elapsed_ns=(99 if phase == "warmup" else 1) * 1_000_000))
    records.append(dict(record_type="end", run_id="test", status="ok", calls=len(records)-1,
                        measurement_errors=0, peak_rss_mib=10.))
    return records


def load(records):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "run.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        return p.load_run(path)


class Task3AnalysisTests(unittest.TestCase):
    def test_warmup_units_and_independent_quality(self):
        meta, calls = load(sample_log())
        summary, categories, delta = p.summarize(meta, calls)
        self.assertTrue((summary.latency_p50_ms == 1).all())
        self.assertTrue((summary.throughput_snippets_per_s == 1000).all())
        self.assertTrue((summary.latency_p50_ci_low_ms == 1).all())
        self.assertTrue((summary.throughput_ci_high_snippets_per_s == 1000).all())
        self.assertTrue((summary.query("corpus == 'fixture'").quality_n == 4).all())
        self.assertTrue(summary.query("corpus == 'repository'").f1.isna().all())
        self.assertTrue((categories.n == 4).all())
        self.assertEqual(delta["delta_f1"], -1.)
        self.assertEqual(delta["valid_draws"], 100)

    def test_complete_grid_and_footer(self):
        for mutation, message in [(lambda r: r.pop(2), "Footer"),
                                  (lambda r: r.__setitem__(2, r[1]), "Duplicate"),
                                  (lambda r: r.pop(), "Missing run footer")]:
            with self.subTest(message=message):
                rows = sample_log()
                mutation(rows)
                with self.assertRaisesRegex(ValueError, message):
                    load(rows)
        rows = sample_log()
        rows.pop(2)
        rows[-1]["calls"] -= 1
        with self.assertRaisesRegex(ValueError, "Incomplete run"):
            load(rows)

    def test_errors_and_outliers_remain_visible(self):
        rows = sample_log()
        rows[-2].update(status="error", error="test error", prediction=None, score=None)
        rows[-1].update(status="error", measurement_errors=1)
        rows[-3]["elapsed_ns"] *= 100
        meta, calls = load(rows)
        summary, _, _ = p.summarize(meta, calls)
        real = summary.query("corpus == 'repository'").set_index("method")
        self.assertEqual(real.at["multinomial_nb", "error_rate"], 1 / 6)
        self.assertFalse(real.at["multinomial_nb", "zero_errors_pass"])
        self.assertTrue(p.pd.isna(real.at["multinomial_nb", "latency_p50_ci_low_ms"]))
        self.assertEqual(real.at["static_rules", "outlier_count"], 1)
        self.assertGreater(real.at["static_rules", "latency_p99_ms"], 90)

    def test_block_bootstrap_resamples_passes(self):
        # Both inputs rise and fall together. Resampling individual calls would
        # invalidate the common disturbance shared by each pass.
        matrix = p.np.array([[1., 1.], [1., 1.], [100., 100.]])
        ci = p._block_intervals(matrix, 1000, 42)
        self.assertEqual(ci.tolist(), [[1., 100.], [10., 1000.]])
