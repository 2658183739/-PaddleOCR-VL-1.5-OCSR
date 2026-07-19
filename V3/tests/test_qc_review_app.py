import csv
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from qc_review_app import ReviewStore  # noqa: E402


FIELDS = [
    "panel",
    "benchmark_role",
    "sample_id",
    "source",
    "difficulty",
    "image",
    "label",
    "automated_status",
    "reviewer_1",
    "reviewer_1_decision",
    "reviewer_1_reason",
    "reviewer_2",
    "reviewer_2_decision",
    "reviewer_2_reason",
    "final_decision",
    "review_time",
]


class ReviewStoreTest(unittest.TestCase):
    def test_reviewer_and_adjudicator_updates_are_persisted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = (
                root
                / "V3"
                / "data"
                / "eval"
                / "canonical_smiles_main_v1"
                / "images"
                / "decimer"
                / "sample.png"
            )
            image.parent.mkdir(parents=True)
            image.write_bytes(b"not-decoded-by-review-store")
            review_csv = root / "V3" / "qc" / "eval_manual_review.csv"
            review_csv.parent.mkdir(parents=True)
            with review_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(
                    {
                        "panel": "core_767",
                        "benchmark_role": "legacy_development",
                        "sample_id": "sample",
                        "source": "decimer",
                        "difficulty": "hard",
                        "image": "images/decimer/sample.png",
                        "label": "CCO",
                        "automated_status": "pass",
                        "reviewer_1_decision": "pending",
                        "reviewer_2_decision": "pending",
                        "final_decision": "pending",
                    }
                )

            reviewer = ReviewStore(review_csv, root, "1")
            self.assertEqual(reviewer.image_path(reviewer.rows[0]), str(image))
            reviewer.save(0, "pass", "ok", "R1")

            adjudicator = ReviewStore(review_csv, root, "adjudicator")
            adjudicator.save(0, "reject", "label_mismatch", "ADJ")

            with review_csv.open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["reviewer_1"], "R1")
            self.assertEqual(row["reviewer_1_decision"], "pass")
            self.assertEqual(row["reviewer_1_reason"], "ok")
            self.assertEqual(row["final_decision"], "reject")
            self.assertEqual(row["final_decision_reason"], "label_mismatch")
            self.assertIn("+00:00", row["review_time"])


if __name__ == "__main__":
    unittest.main()
