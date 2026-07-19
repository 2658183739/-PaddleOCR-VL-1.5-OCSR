from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "V3" / "scripts"))

from build_v3_datasets import build_grouped_wild_split, canonicalize, paper_group_from_id
from compare_eval_runs import metric_summary
from evaluate_ocsr_predictions_detailed import canonicalize_smiles as eval_canonicalize
from infer_ocsr_transformers import canonicalize_smiles as infer_canonicalize
from infer_ocsr_transformers import select_shard_records
from run_sharded_inference import merge_shards
from summarize_checkpoint_eval_results import infer_metadata
from import_private_photo_data import requested_structure_split, split_for_structure


def wild_row(paper: str, page: int, molecule: int, smiles: str):
    identifier = f"{paper}_{page}_figure_0_mol_{molecule}.jpg"
    return {
        "id": identifier,
        "smiles": smiles,
        "symbols": ["C"] * max(1, len(smiles)),
        "hardcase_label": ["Blurry or Unclear Image"],
        "local_image_path": f"X:/{identifier}",
    }


class GroupedSplitTests(unittest.TestCase):
    def test_paper_group_parser(self):
        sample = "10.1002_anie.202400632_2_figure_0_mol_5.jpg"
        self.assertEqual(paper_group_from_id(sample), "10.1002_anie.202400632")

    def test_paper_groups_never_cross_train_and_test(self):
        rows = [
            wild_row("paper.a", 1, 0, "C"),
            wild_row("paper.a", 1, 1, "CC"),
            wild_row("paper.b", 1, 0, "CCC"),
            wild_row("paper.b", 1, 1, "CCCC"),
            wild_row("paper.c", 1, 0, "CCO"),
            wild_row("paper.c", 1, 1, "CCN"),
            wild_row("paper.d", 1, 0, "CCCl"),
            wild_row("paper.d", 1, 1, "CCBr"),
        ]
        train, locked, _, stats, split = build_grouped_wild_split(
            rows,
            legacy_dev_smiles=set(),
            eval_target=2,
            train_cap=20,
            eval_max_per_paper=1,
            eval_min_papers=2,
        )
        train_papers = {row["meta"]["paper_group"] for row in train}
        locked_papers = {row["paper_group"] for row in locked}
        self.assertFalse(train_papers & locked_papers)
        self.assertEqual(len(locked_papers), 2)
        self.assertEqual(len(locked), 2)
        self.assertEqual(stats["strict_eval_excess_held_out"], 2)
        split_by_paper = {}
        for row in split:
            split_by_paper.setdefault(row["paper_group"], set()).add(row["split"])
        for paper in locked_papers:
            self.assertNotIn("train", split_by_paper[paper])

    def test_locked_eval_canonical_molecules_are_unique(self):
        rows = [
            wild_row("paper.a", 1, 0, "C"),
            wild_row("paper.a", 1, 1, "C"),
            wild_row("paper.b", 1, 0, "CC"),
            wild_row("paper.b", 1, 1, "CCC"),
            wild_row("paper.c", 1, 0, "CCCC"),
            wild_row("paper.d", 1, 0, "CCO"),
        ]
        _, locked, _, _, _ = build_grouped_wild_split(
            rows,
            legacy_dev_smiles=set(),
            eval_target=4,
            train_cap=20,
            eval_max_per_paper=2,
            eval_min_papers=4,
        )
        canonical = [row["ground_truth"]["smiles"] for row in locked]
        self.assertEqual(len(canonical), len(set(canonical)))


class CanonicalizationTests(unittest.TestCase):
    def test_canonicalization_is_idempotent(self):
        for smiles in ["C(C)O", "p12p3p1p23", "O=C([O-])CO", "C[C@H](O)Cl"]:
            first = canonicalize(smiles)
            self.assertIsNotNone(first)
            self.assertEqual(canonicalize(first), first)

    def test_multifragment_labels_are_rejected(self):
        self.assertIsNone(canonicalize("CCO.Cl"))
        self.assertIsNone(eval_canonicalize("CCO.Cl"))
        self.assertIsNone(infer_canonicalize("CCO.Cl"))

    def test_symbolic_dummy_atoms_are_rejected_consistently(self):
        self.assertIsNone(canonicalize("[*]CC"))
        self.assertIsNone(eval_canonicalize("[*]CC"))
        self.assertIsNone(infer_canonicalize("[*]CC"))


class InferenceShardingTests(unittest.TestCase):
    def test_shards_are_disjoint_and_cover_all_records(self):
        records = list(range(13))
        shards = [select_shard_records(records, index, 4) for index in range(4)]
        flattened = [item for shard in shards for item in shard]
        self.assertEqual(sorted(flattened), records)
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_invalid_shard_is_rejected(self):
        with self.assertRaises(ValueError):
            select_shard_records([1, 2], 2, 2)

    def test_merge_restores_benchmark_order(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            benchmark = root / "benchmark.jsonl"
            first = root / "part-00.jsonl"
            second = root / "part-01.jsonl"
            output = root / "pred.jsonl"
            benchmark.write_text(
                "".join(json.dumps({"id": item}) + "\n" for item in ["a", "b", "c"]),
                encoding="utf-8",
            )
            first.write_text(
                json.dumps({"id": "a", "prediction": "A"})
                + "\n"
                + json.dumps({"id": "c", "prediction": "C"})
                + "\n",
                encoding="utf-8",
            )
            second.write_text(
                json.dumps({"id": "b", "prediction": "B"}) + "\n",
                encoding="utf-8",
            )
            merge_shards(benchmark, [first, second], output)
            merged = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["id"] for row in merged], ["a", "b", "c"])

    def test_v3_summary_metadata_uses_relative_directories(self):
        root = Path("/tmp/eval-root")
        report = root / "data_11_s1" / "checkpoint-250" / "legacy_core_dev" / "report.json"
        self.assertEqual(
            infer_metadata(report, root),
            ("data_11_s1", "checkpoint-250", "legacy_core_dev"),
        )


class ClusterBootstrapTests(unittest.TestCase):
    def test_metric_resamples_clusters_not_images(self):
        baseline = [
            {"id": "a1", "paper_group": "a", "canonical_exact_match": False},
            {"id": "a2", "paper_group": "a", "canonical_exact_match": True},
            {"id": "b1", "paper_group": "b", "canonical_exact_match": False},
        ]
        candidate = [
            {"id": "a1", "paper_group": "a", "canonical_exact_match": True},
            {"id": "a2", "paper_group": "a", "canonical_exact_match": True},
            {"id": "b1", "paper_group": "b", "canonical_exact_match": False},
        ]
        result = metric_summary(
            baseline,
            candidate,
            "canonical_exact_match",
            iterations=200,
            seed=7,
            cluster_field="paper_group",
        )
        self.assertEqual(result["independent_units"], 2)
        self.assertEqual(result["resampling_unit"], "paper_group")
        self.assertAlmostEqual(result["delta_mean"], 0.25)


class PrivateSplitTests(unittest.TestCase):
    def test_same_structure_always_has_same_split(self):
        first = split_for_structure("structure-42", 0.30)
        for _ in range(10):
            self.assertEqual(split_for_structure("structure-42", 0.30), first)

    def test_explicit_structure_split_overrides_hash(self):
        self.assertEqual(requested_structure_split("structure-42", 0.30, "eval"), "eval")
        self.assertEqual(requested_structure_split("structure-42", 0.30, "train"), "train")

    def test_invalid_explicit_structure_split_is_rejected(self):
        with self.assertRaises(ValueError):
            requested_structure_split("structure-42", 0.30, "validation")


if __name__ == "__main__":
    unittest.main()
