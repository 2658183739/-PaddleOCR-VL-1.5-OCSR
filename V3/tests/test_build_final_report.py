import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_final_report.py"
SPEC = importlib.util.spec_from_file_location("build_final_report", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FinalReportTests(unittest.TestCase):
    def test_render_includes_design_boundary_checkpoints_and_runtime(self):
        panel = {"canonical_exact": 0.3, "valid_rate": 0.7}
        report = {
            "accuracy": {
                "canonical_exact_match_accuracy": 0.25,
                "valid_smiles_rate": 0.8,
            },
            "total": 1,
        }
        summary = {
            "dataset": {
                "base_rows": 10,
                "control_rows": 8,
                "wild_train_rows": 2,
                "wild_locked_rows": 1,
                "wild_locked_paper_groups": 1,
                "wild_symbolic_rows": 1,
                "scaffold_novel_rows": 1,
                "human_review_status": "owner_attested_complete",
            },
            "probe": {
                "conditions": {
                    key: {
                        "mean_macro_exact": 0.3,
                        "seed_range": 0.01,
                        "min_valid_rate": 0.7,
                    }
                    for key in ("00", "10", "01", "11")
                },
                "effects": {
                    "wild_main_effect": 0.0,
                    "augmentation_main_effect": 0.0,
                    "interaction": 0.0,
                },
                "winner": {"dataset_path": "train.jsonl"},
                "diagnostics": {
                    "augmentation_dose2": {"macro_exact": 0.3, "delta_vs_11_s1": 0.0},
                    "warmstart": {"continuation_minus_base15": 0.3},
                },
            },
            "probe_pairwise": {
                "comparisons": [
                    {
                        "baseline_run": "data_00_s1",
                        "candidate_run": "data_11_s1",
                        "panel": "legacy_core_dev",
                        "independent_units": 10,
                        "exact_delta": 0.01,
                        "exact_ci95_low": -0.01,
                        "exact_ci95_high": 0.03,
                    }
                ]
            },
            "final_checkpoint_selection": {
                "checkpoints": [
                    {
                        "checkpoint": "checkpoint-200",
                        "step": 200,
                        "macro_exact": 0.3,
                        "min_valid_rate": 0.7,
                    }
                ],
                "winner": {
                    "checkpoint": "checkpoint-200",
                    "macro_exact": 0.3,
                    "min_valid_rate": 0.7,
                },
            },
            "final_vs_hard_replay": {
                "winner": {"label": "final"},
                "candidate": {"macro_exact": 0.29},
                "baseline": {"macro_exact": 0.3},
                "candidate_macro_delta": -0.01,
                "minimum_improvement": 0.005,
            },
            "generation_policy_selection": {
                "winner": {"label": "beam4_return4"},
                "candidate": {"label": "beam4_chem_light", "macro_exact": 0.29},
                "baseline": {"label": "beam4_return4", "macro_exact": 0.3},
                "candidate_macro_delta": -0.01,
                "minimum_improvement": 0.005,
            },
            "generation_policy_beam_selection": {
                "winner": {"label": "beam4_return4"},
                "candidate": {"label": "beam4_return4", "macro_exact": 0.36},
                "baseline": {"label": "greedy", "macro_exact": 0.3},
                "candidate_macro_delta": 0.06,
                "minimum_improvement": 0.005,
            },
            "posttraining_pairwise": {
                "final_vs_hard_replay": [
                    {
                        "panel": "legacy_core_dev",
                        "independent_units": 10,
                        "exact_delta": -0.01,
                        "exact_ci95_low": -0.03,
                        "exact_ci95_high": 0.01,
                        "valid_delta": 0.0,
                    }
                ]
            },
            "locked": {
                "wild_strict": report,
                "wild_scaffold_novel": report,
                "wild_symbolic": {
                    "counts": {"total": 1},
                    "accuracy": {
                        "whitespace_normalized_exact_match_accuracy": 0.1,
                        "nonempty_prediction_rate": 0.9,
                    },
                },
                "private_photo": None,
            },
            "training_metrics": {
                "final_s1": {
                    "train_loss": 0.2,
                    "train_runtime": 60.0,
                    "train_samples_per_second": 8.0,
                    "train_steps_per_second": 0.25,
                }
            },
        }

        markdown = MODULE.render_markdown(summary)

        self.assertIn("运行顺序未完全随机化", markdown)
        self.assertIn("checkpoint-200", markdown)
        self.assertIn("runtime (min)", markdown)
        self.assertIn("training_artifacts/resume", markdown)
        self.assertIn("paired bootstrap", markdown)
        self.assertIn("至少 0.005", markdown)
        self.assertIn("`beam4_return4` macro exact=0.3600", markdown)
        self.assertIn("`greedy`=0.3000", markdown)
        self.assertIn("`beam4_chem_light` macro exact=0.2900", markdown)
        self.assertIn("manual_review_attestation.json", markdown)


if __name__ == "__main__":
    unittest.main()
