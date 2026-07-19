import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_symbolic_predictions.py"
SPEC = importlib.util.spec_from_file_location("evaluate_symbolic_predictions", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SymbolicEvaluationTests(unittest.TestCase):
    def test_symbolic_exact_does_not_require_rdkit_validity(self):
        benchmark = [
            {
                "id": "a",
                "paper_group": "paper-1",
                "difficulty": "hard",
                "ground_truth": {"smiles": "C1=CC=CC=C1.[Fe][?]"},
            }
        ]
        predictions = [{"id": "a", "prediction": "C1=CC=CC=C1. [Fe][?]"}]

        report, details = MODULE.evaluate(benchmark, predictions)

        self.assertEqual(report["counts"]["raw_exact"], 0)
        self.assertEqual(report["counts"]["whitespace_normalized_exact"], 1)
        self.assertTrue(details[0]["whitespace_normalized_exact"])


if __name__ == "__main__":
    unittest.main()
