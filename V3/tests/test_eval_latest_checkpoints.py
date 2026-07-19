import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_latest_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("eval_latest_checkpoints", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RuntimeMetadataTests(unittest.TestCase):
    def test_sync_copies_missing_metadata_without_overwriting_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_dir = root / "base"
            runtime_dir = root / "runtime"
            base_dir.mkdir()
            runtime_dir.mkdir()

            for name in MODULE.REQUIRED_RUNTIME_NAMES:
                (base_dir / name).write_text(f"base:{name}", encoding="utf-8")
            (base_dir / "config.json").write_text("base-config", encoding="utf-8")
            (base_dir / "processor_config.json").write_text("processor", encoding="utf-8")
            (runtime_dir / "config.json").write_text("export-config", encoding="utf-8")

            copied = MODULE.sync_runtime_metadata(runtime_dir, base_dir)

            self.assertEqual((runtime_dir / "config.json").read_text(encoding="utf-8"), "export-config")
            self.assertEqual(
                (runtime_dir / "processor_config.json").read_text(encoding="utf-8"), "processor"
            )
            self.assertEqual(
                {path.name for path in copied},
                MODULE.REQUIRED_RUNTIME_NAMES | {"processor_config.json"},
            )

    def test_sync_rejects_missing_required_custom_code(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_dir = root / "base"
            runtime_dir = root / "runtime"
            base_dir.mkdir()
            runtime_dir.mkdir()

            with self.assertRaisesRegex(FileNotFoundError, "configuration_paddleocr_vl.py"):
                MODULE.sync_runtime_metadata(runtime_dir, base_dir)

    def test_build_phase_commands_forwards_positive_limit(self):
        phase_run = MODULE.PhaseRun(
            phase="smoke",
            phase_root=Path("phase"),
            checkpoint_dir=Path("phase/checkpoint-1"),
            lora_dir=Path("phase/checkpoint-1"),
            model_dir=Path("phase/checkpoint-1/export"),
            output_root=Path("eval/smoke/checkpoint-1"),
            benchmarks=(
                MODULE.BenchmarkSpec("dev", Path("V3/data/eval/dev/labels.jsonl"), 64),
            ),
        )

        commands = MODULE.build_phase_commands(
            project_root=Path("/project"),
            phase_run=phase_run,
            prompt_file=Path("/project/V3/configs/prompt.txt"),
            device="cuda",
            torch_dtype="bfloat16",
            min_pixels=1,
            max_pixels=2,
            limit=1,
        )

        self.assertIn("--limit", commands[0])
        self.assertEqual(commands[0][commands[0].index("--limit") + 1], "1")


if __name__ == "__main__":
    unittest.main()
