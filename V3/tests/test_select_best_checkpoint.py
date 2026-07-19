import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "select_best_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("select_best_checkpoint", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CheckpointSelectionTests(unittest.TestCase):
    def test_near_tie_selects_earlier_checkpoint(self):
        rows = [
            {"checkpoint": "checkpoint-200", "step": 200, "macro_exact": 0.300, "min_valid_rate": 0.8},
            {"checkpoint": "checkpoint-400", "step": 400, "macro_exact": 0.304, "min_valid_rate": 0.8},
        ]

        winner = MODULE.select_checkpoint(rows, validity_floor=0.7, tie_tolerance=0.005)

        self.assertEqual(winner["checkpoint"], "checkpoint-200")

    def test_validity_floor_excludes_higher_scoring_checkpoint(self):
        rows = [
            {"checkpoint": "checkpoint-200", "step": 200, "macro_exact": 0.300, "min_valid_rate": 0.8},
            {"checkpoint": "checkpoint-400", "step": 400, "macro_exact": 0.350, "min_valid_rate": 0.6},
        ]

        winner = MODULE.select_checkpoint(rows, validity_floor=0.7, tie_tolerance=0.005)

        self.assertEqual(winner["checkpoint"], "checkpoint-200")


if __name__ == "__main__":
    unittest.main()
