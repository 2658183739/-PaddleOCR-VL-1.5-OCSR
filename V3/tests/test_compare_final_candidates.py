import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_final_candidates.py"
SPEC = importlib.util.spec_from_file_location("compare_final_candidates", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def candidate(label, macro, valid=0.8, core=None, region=None):
    return {
        "label": label,
        "macro_exact": macro,
        "min_valid_rate": valid,
        "panels": {
            "legacy_core_dev": {"canonical_exact": macro if core is None else core},
            "legacy_region_dev": {"canonical_exact": macro if region is None else region},
        },
    }


class CandidateDecisionTests(unittest.TestCase):
    def test_near_tie_keeps_simpler_baseline(self):
        baseline = candidate("greedy", 0.30)
        beam = candidate("beam", 0.302)

        decision = MODULE.choose_winner(baseline, beam, 0.005, 0.005)

        self.assertEqual(decision["winner"]["label"], "greedy")
        self.assertFalse(decision["candidate_meaningful_improvement"])

    def test_meaningful_nonregressing_gain_selects_candidate(self):
        baseline = candidate("final", 0.30)
        replay = candidate("hard_replay", 0.307)

        decision = MODULE.choose_winner(baseline, replay, 0.005, 0.005)

        self.assertEqual(decision["winner"]["label"], "hard_replay")
        self.assertTrue(decision["candidate_meaningful_improvement"])


if __name__ == "__main__":
    unittest.main()
