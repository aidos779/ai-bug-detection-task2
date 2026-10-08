import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.benchmark import (ROOT, MultinomialNB, check_splits, classification_metrics,
                           load_data, paired_project_bootstrap, percentile, run, static_score)


class BenchmarkTests(unittest.TestCase):
    def test_confusion_and_undefined_precision(self):
        m = classification_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual((m['tp'], m['fp'], m['tn'], m['fn']), (1, 1, 1, 1))
        self.assertEqual(m['f1'], 0.5)
        self.assertEqual(m['false_positive_rate'], 0.5)
        self.assertEqual(classification_metrics([1, 0], [0, 0])['f1'], 0)
        self.assertIsNone(classification_metrics([1], [1])['false_positive_rate'])

    def test_static_rules_and_safe_counterexamples(self):
        self.assertEqual(static_score('def bad(x):\n return x / 0'), 1)
        self.assertEqual(static_score('def bad(x=[]):\n x.append(1)\n return x'), 1)
        self.assertEqual(static_score('def ok(x):\n return x / 2'), 0)
        self.assertEqual(static_score('def ok(x=None):\n return [] if x is None else x'), 0)
        self.assertEqual(static_score('def bad(*, x={}):\n x[1] = 2\n return x'), 1)

    def test_nb_hand_calculated_posteriors(self):
        records = [{'code': 'a a', 'label': 0}, {'code': 'b b', 'label': 1}]
        model = MultinomialNB(alpha=1).fit(records)
        self.assertAlmostEqual(model.score('a'), 0.25)
        self.assertAlmostEqual(model.score('b'), 0.75)
        self.assertAlmostEqual(model.score('unknown'), 0.5)
        self.assertAlmostEqual(model.score('b b'), 0.9)

    def test_leakage_detection(self):
        train = load_data(ROOT / 'data/sample/train.jsonl')
        test = load_data(ROOT / 'data/sample/test.jsonl')
        check_splits(train, test)
        with self.assertRaises(ValueError):
            check_splits(train, train)
        altered = [dict(row) for row in test]
        altered[0]['project'] = train[0]['project']
        with self.assertRaises(ValueError):
            check_splits(train, altered)
        altered = [dict(row) for row in test]
        altered[0]['code'] = train[0]['code'] + '\n# whitespace-normalized duplicate'
        with self.assertRaises(ValueError):
            check_splits(train, altered)

    def test_invalid_labels_and_duplicates(self):
        original = (ROOT / 'data/sample/test.jsonl').read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.jsonl'
            row = json.loads(original.splitlines()[0])
            row['label'] = 2
            path.write_text(json.dumps(row) + '\n')
            with self.assertRaises(ValueError):
                load_data(path)
            path.write_text(original + original.splitlines()[0] + '\n')
            with self.assertRaises(ValueError):
                load_data(path)

    def test_bootstrap_is_paired_and_reproducible(self):
        test = load_data(ROOT / 'data/sample/test.jsonl')
        labels = [row['label'] for row in test]
        predictions = {'static_rules': labels, 'multinomial_nb': labels}
        a = paired_project_bootstrap(test, predictions, 100, 42)
        b = paired_project_bootstrap(test, predictions, 100, 42)
        self.assertEqual(a, b)
        self.assertEqual((a['lower'], a['upper']), (0, 0))
        self.assertEqual(percentile([0, 10], 0.95), 9.5)

    def test_end_to_end_records_and_smoke_interpretation(self):
        config = json.loads((ROOT / 'configs/smoke.json').read_text())
        config['repetitions'] = 2
        config['bootstrap_draws'] = 100
        with tempfile.TemporaryDirectory() as directory:
            config['output_dir'] = directory
            config_path = Path(directory) / 'config.json'
            config_path.write_text(json.dumps(config))
            result = run(config_path)
            self.assertEqual(result['test_rows'], 8)
            self.assertEqual(result['metrics']['static_rules']['f1'], 1)
            self.assertEqual(result['interpretation'], 'pipeline_validated_only_no_research_conclusion')
            self.assertTrue((Path(directory) / 'summary.json').exists())
            self.assertEqual(len((Path(directory) / 'predictions.csv').read_text().splitlines()), 9)
            self.assertEqual(len((Path(directory) / 'latencies.csv').read_text().splitlines()), 33)
            self.assertIn('code_sha256', result['hashes'])

    def test_research_mode_rejects_smoke_data(self):
        config = json.loads((ROOT / 'configs/smoke.json').read_text())
        config['mode'] = 'research'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                run(path)

    def test_cli_failure_has_nonzero_status(self):
        process = subprocess.run([sys.executable, str(ROOT / 'src/benchmark.py'),
                                  '--config', 'does-not-exist.json'], capture_output=True, text=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('Benchmark failed', process.stderr)


if __name__ == '__main__':
    unittest.main()
