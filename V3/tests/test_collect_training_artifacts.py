import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_training_artifacts.py"
SPEC = importlib.util.spec_from_file_location("collect_training_artifacts", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ResumeArtifactTests(unittest.TestCase):
    def test_copy_resume_checkpoint_copies_only_resume_files_with_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            (source / "lora_model_state.pdparams").write_bytes(b"adapter")
            (source / "optimizer.pdopt").write_bytes(b"optimizer")
            (source / "ignored.txt").write_text("skip", encoding="utf-8")

            copied = MODULE.copy_resume_checkpoint(source, target)

            self.assertEqual(
                {Path(row["path"]).name for row in copied},
                {"lora_model_state.pdparams", "optimizer.pdopt"},
            )
            self.assertTrue(all(len(row["sha256"]) == 64 for row in copied))
            self.assertFalse((target / "ignored.txt").exists())
            self.assertIn("optimizer.pdopt", MODULE.RESUME_NAMES)


if __name__ == "__main__":
    unittest.main()
