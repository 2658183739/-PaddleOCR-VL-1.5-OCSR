import argparse
import hashlib
import json
import shutil
from pathlib import Path


RUN_IDS = (
    "data_11_s1",
    "data_00_s1",
    "data_10_s1",
    "data_01_s1",
    "data_11_s2",
    "data_10_s2",
    "data_00_s2",
    "data_01_s2",
    "warmstart_control_s1",
    "aug_dose2_s1",
    "final_s1",
    "hard_replay_s1",
)

METADATA_NAMES = {
    "config.json",
    "generation_config.json",
    "lora_config.json",
    "model.safetensors.index.json",
    "optimizer_name_suffix.json",
    "scheduler_name_suffix.json",
    "train_results.json",
    "trainer_state.json",
    "training_args.bin",
}

RESUME_NAMES = {
    "config.json",
    "lora_config.json",
    "lora_model_state.pdparams",
    "optimizer.pdopt",
    "rng_state_0.pth",
    "saved_signal_0",
    "scheduler.pdparams",
    "trainer_state.json",
    "training_args.bin",
}


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def copy_metadata(source_root: Path, target_root: Path):
    copied = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.name not in METADATA_NAMES:
            continue
        relative = path.relative_to(source_root)
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(str(relative))
    return copied


def copy_resume_checkpoint(
    source_root: Path,
    target_root: Path,
    project_root: Path | None = None,
    artifact_root: Path | None = None,
):
    copied = []
    for name in sorted(RESUME_NAMES):
        source = source_root / name
        if not source.is_file():
            continue
        target = target_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(
            {
                "path": str(target.relative_to(artifact_root) if artifact_root else target),
                "source": str(source.relative_to(project_root) if project_root else source),
                "size_bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-dir", default="V3/evidence/training_artifacts"
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    manifest = {"runs": {}, "weights": [], "resume_artifacts": []}
    outputs_root = project_root / "V3" / "outputs"
    for run_id in RUN_IDS:
        run_root = outputs_root / run_id
        if not run_root.exists():
            manifest["runs"][run_id] = {"status": "missing", "metadata": []}
            continue
        copied = copy_metadata(run_root, output_dir / "outputs" / run_id)
        manifest["runs"][run_id] = {"status": "present", "metadata": copied}

        weight_paths = (
            list(run_root.rglob("*.safetensors"))
            + list(run_root.rglob("*.distcp"))
            + list(run_root.rglob("*.pdparams"))
        )
        for path in sorted(weight_paths):
            manifest["weights"].append(
                {
                    "path": str(path.relative_to(project_root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    final_model = project_root / "V3" / "models" / "final_best_export"
    if final_model.exists():
        copy_metadata(final_model, output_dir / "models" / "final_best_export")
        for path in sorted(final_model.rglob("*.safetensors")):
            manifest["weights"].append(
                {
                    "path": str(path.relative_to(project_root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    selection_path = project_root / "V3" / "evidence" / "final_checkpoint_selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        checkpoint_name = selection["winner"]["checkpoint"]
        source = outputs_root / "final_s1" / checkpoint_name
        manifest["resume_artifacts"].extend(
            copy_resume_checkpoint(
                source,
                output_dir / "resume" / "final_s1" / checkpoint_name,
                project_root=project_root,
                artifact_root=output_dir,
            )
        )

    hard_checkpoint = outputs_root / "hard_replay_s1" / "checkpoint-300"
    if hard_checkpoint.exists():
        manifest["resume_artifacts"].extend(
            copy_resume_checkpoint(
                hard_checkpoint,
                output_dir / "resume" / "hard_replay_s1" / "checkpoint-300",
                project_root=project_root,
                artifact_root=output_dir,
            )
        )

    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
