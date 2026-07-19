import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "update_experiment_matrix.py"
SPEC = importlib.util.spec_from_file_location("update_experiment_matrix", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExperimentMatrixTests(unittest.TestCase):
    def test_completed_probe_rows_are_filled_without_changing_future_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            matrix = root / "matrix.csv"
            analysis = root / "analysis.json"
            matrix.write_text(
                "run_id,dev_core_exact,dev_region_exact,dev_valid,selected_checkpoint,status\n"
                "data_00_s1,,,,,planned\n"
                "final_s1,,,,,planned\n",
                encoding="utf-8",
            )
            analysis.write_text(
                json.dumps(
                    {
                        "runs": {
                            "data_00_s1": {
                                "checkpoint": "checkpoint-250",
                                "min_valid_rate": 0.7,
                                "panels": {
                                    "legacy_core_dev": {"canonical_exact": 0.31},
                                    "legacy_region_dev": {"canonical_exact": 0.32},
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            MODULE.update_matrix(matrix, analysis)

            with matrix.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["dev_core_exact"], "0.310000000")
            self.assertEqual(rows[0]["selected_checkpoint"], "checkpoint-250")
            self.assertEqual(rows[0]["status"], "completed")
            self.assertEqual(rows[1]["status"], "planned")


if __name__ == "__main__":
    unittest.main()
