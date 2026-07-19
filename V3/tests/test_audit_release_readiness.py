import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_release_readiness.py"
SPEC = importlib.util.spec_from_file_location("audit_release_readiness", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReleaseReadinessTests(unittest.TestCase):
    def test_audit_manifest_counts_nested_metadata_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            rows = [
                {
                    "meta": {
                        "source": "a",
                        "license": "Apache-2.0",
                        "source_url_or_doc": "https://example.test/a",
                        "structure_id": "s1",
                    }
                },
                {"meta": {"source": "b"}},
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            result = MODULE.audit_manifest(path)

            self.assertEqual(result["rows"], 2)
            self.assertEqual(result["license_coverage"], 0.5)
            self.assertEqual(result["source_url_coverage"], 0.5)
            self.assertEqual(result["structure_id_coverage"], 0.5)

    def test_render_includes_manual_review_and_private_counts(self):
        audit = {
            "manifests": {},
            "project_files": {},
            "manual_review_rows": 1068,
            "manual_review_pending_rows": 1068,
            "manual_review_panel_counts": {"core_767": 767, "wild_strict_v3": 301},
            "manual_review_attested_complete": True,
            "private_photo_rows": 0,
            "release_blockers": ["pending review"],
        }

        markdown = MODULE.render_markdown(audit)

        self.assertIn("Pending final decisions: 1068", markdown)
        self.assertIn("Owner attestation: complete", markdown)
        self.assertIn("Private-photo rows: 0", markdown)


if __name__ == "__main__":
    unittest.main()
