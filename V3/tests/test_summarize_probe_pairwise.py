import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_probe_pairwise.py"
SPEC = importlib.util.spec_from_file_location("summarize_probe_pairwise", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ProbePairwiseSummaryTests(unittest.TestCase):
    def test_collect_extracts_run_panel_and_cluster_ci(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = {
                "n": 2,
                "baseline": "V3/eval_runs_probes/data_00_s1/checkpoint-250/legacy_core_dev/details.jsonl",
                "candidate": "V3/eval_runs_probes/data_11_s1/checkpoint-250/legacy_core_dev/details.jsonl",
                "canonical_exact": {
                    "independent_units": 2,
                    "resampling_unit": "structure_id",
                    "delta_mean": 0.1,
                    "ci95_low": -0.1,
                    "ci95_high": 0.2,
                    "probability_delta_gt_zero": 0.7,
                },
                "valid_smiles": {"delta_mean": 0.0},
                "selection_gate": {"pass": True},
            }
            (root / "comparison.json").write_text(json.dumps(payload), encoding="utf-8")

            rows = MODULE.collect(root)

            self.assertEqual(rows[0]["baseline_run"], "data_00_s1")
            self.assertEqual(rows[0]["candidate_run"], "data_11_s1")
            self.assertEqual(rows[0]["panel"], "legacy_core_dev")
            self.assertEqual(rows[0]["resampling_unit"], "structure_id")
            self.assertIn("[-0.100000, 0.200000]", MODULE.render_markdown(rows))


if __name__ == "__main__":
    unittest.main()
