import argparse
import dataclasses
import re
import shutil
import subprocess
from pathlib import Path


DEFAULT_PHASES = [
    "probe_a_control",
    "probe_d_wild_only",
    "probe_e_aug_only",
    "probe_b_recommended",
    "probe_c_real_heavy",
    "probe_base15_recommended",
    "final_continue_a100",
    "hard_replay_a100",
]


@dataclasses.dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    jsonl_rel_path: Path
    max_new_tokens: int


@dataclasses.dataclass(frozen=True)
class PhaseRun:
    phase: str
    phase_root: Path
    checkpoint_dir: Path
    lora_dir: Path
    model_dir: Path
    output_root: Path
    benchmarks: tuple[BenchmarkSpec, ...]


DEFAULT_BENCHMARKS = (
    BenchmarkSpec(
        name="legacy_core_dev",
        jsonl_rel_path=Path("V3/data/eval/dev_legacy_core_strict/labels.jsonl"),
        max_new_tokens=256,
    ),
    BenchmarkSpec(
        name="legacy_region_dev",
        jsonl_rel_path=Path("V3/data/eval/dev_legacy_region_strict/labels.jsonl"),
        max_new_tokens=256,
    ),
)

RUNTIME_METADATA_NAMES = {
    "README.md",
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "configuration_paddleocr_vl.py",
    "generation_config.json",
    "image_processing_paddleocr_vl.py",
    "inference.yml",
    "lora_config.json",
    "model.safetensors.index.json",
    "model.safetensors",
    "modeling_paddleocr_vl.py",
    "peft_model.safetensors.index.json",
    "preprocessor_config.json",
    "processing_paddleocr_vl.py",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
}

REQUIRED_RUNTIME_NAMES = {
    "configuration_paddleocr_vl.py",
    "image_processing_paddleocr_vl.py",
    "modeling_paddleocr_vl.py",
    "processing_paddleocr_vl.py",
}


def list_numeric_checkpoints(phase_dir: Path) -> list[Path]:
    pattern = re.compile(r"^checkpoint-(\d+)$")
    ranked: list[tuple[int, Path]] = []
    for child in phase_dir.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if not match:
            continue
        ranked.append((int(match.group(1)), child))
    ranked.sort(key=lambda item: item[0])
    return [path for _, path in ranked]


def find_latest_numeric_checkpoint(phase_dir: Path) -> Path | None:
    checkpoints = list_numeric_checkpoints(phase_dir)
    return checkpoints[-1] if checkpoints else None


def sync_runtime_metadata(runtime_dir: Path, base_model_dir: Path) -> list[Path]:
    copied = []
    for name in sorted(RUNTIME_METADATA_NAMES):
        source = base_model_dir / name
        destination = runtime_dir / name
        if source.is_file() and not destination.exists():
            shutil.copy2(source, destination)
            copied.append(destination)

    missing = sorted(name for name in REQUIRED_RUNTIME_NAMES if not (runtime_dir / name).is_file())
    if missing:
        raise FileNotFoundError(
            "Runtime export is missing required custom model files: " + ", ".join(missing)
        )
    return copied


def build_phase_runs(
    project_root: Path,
    phases: list[str] | None = None,
    eval_root: Path | None = None,
    benchmarks: tuple[BenchmarkSpec, ...] = DEFAULT_BENCHMARKS,
    all_checkpoints: bool = False,
) -> list[PhaseRun]:
    selected_phases = phases or list(DEFAULT_PHASES)
    effective_eval_root = eval_root or (project_root / "V3" / "eval_runs_latest")

    runs = []
    for phase in selected_phases:
        phase_dir = project_root / "V3" / "outputs" / phase
        if not phase_dir.exists():
            continue
        checkpoints = list_numeric_checkpoints(phase_dir) if all_checkpoints else []
        if not all_checkpoints:
            latest = find_latest_numeric_checkpoint(phase_dir)
            checkpoints = [latest] if latest is not None else []
        if not checkpoints:
            continue
        for checkpoint_dir in checkpoints:
            lora_dir = checkpoint_dir if all_checkpoints else phase_dir
            runs.append(
                PhaseRun(
                    phase=phase,
                    phase_root=phase_dir,
                    checkpoint_dir=checkpoint_dir,
                    lora_dir=lora_dir,
                    model_dir=lora_dir / "export",
                    output_root=effective_eval_root / phase / checkpoint_dir.name,
                    benchmarks=benchmarks,
                )
            )
    return runs


def export_runtime_model_dir(
    lora_dir: Path,
    runtime_dir: Path,
    base_model_dir: Path,
    export_config: Path,
    export_device: str,
    dry_run: bool,
):
    if runtime_dir.exists() and not dry_run:
        shutil.rmtree(runtime_dir)
    command = [
        "paddleformers-cli",
        "export",
        str(export_config),
        f"model_name_or_path={base_model_dir}",
        f"output_dir={lora_dir}",
        f"device={export_device}",
    ]
    run_command(command, dry_run=dry_run)
    if dry_run:
        return
    copied = sync_runtime_metadata(runtime_dir=runtime_dir, base_model_dir=base_model_dir)
    if copied:
        print("[INFO] copied runtime metadata:", ", ".join(path.name for path in copied))
    if not (runtime_dir / "config.json").exists():
        raise FileNotFoundError(f"Checkpoint export did not produce config.json: {runtime_dir}")
    if not any(runtime_dir.glob("model*.safetensors")):
        raise FileNotFoundError(f"Checkpoint export did not produce model safetensors: {runtime_dir}")


def build_phase_commands(
    project_root: Path,
    phase_run: PhaseRun,
    prompt_file: Path,
    device: str,
    torch_dtype: str,
    min_pixels: int,
    max_pixels: int,
    workers: int = 1,
    python_bin: str = "python",
    limit: int = 0,
) -> list[list[str]]:
    commands: list[list[str]] = []

    infer_script = project_root / "V3" / "scripts" / "infer_ocsr_transformers.py"
    if workers > 1:
        infer_script = project_root / "V3" / "scripts" / "run_sharded_inference.py"
    eval_script = project_root / "V3" / "scripts" / "evaluate_ocsr_predictions_detailed.py"

    for benchmark in phase_run.benchmarks:
        benchmark_dir = phase_run.output_root / benchmark.name
        pred_jsonl = benchmark_dir / "pred.jsonl"
        report_json = benchmark_dir / "report.json"
        details_jsonl = benchmark_dir / "details.jsonl"
        bench_jsonl = project_root / benchmark.jsonl_rel_path

        infer_command = [
                python_bin,
                str(infer_script),
                "--model-dir",
                str(phase_run.model_dir),
                "--benchmark-jsonl",
                str(bench_jsonl),
                "--project-root",
                str(project_root),
                "--output-jsonl",
                str(pred_jsonl),
                "--prompt-file",
                str(prompt_file),
                "--device",
                device,
                "--torch-dtype",
                torch_dtype,
                "--max-new-tokens",
                str(benchmark.max_new_tokens),
                "--min-pixels",
                str(min_pixels),
                "--max-pixels",
                str(max_pixels),
            ]
        if workers > 1:
            infer_command.extend(["--workers", str(workers)])
        if limit > 0:
            infer_command.extend(["--limit", str(limit)])
        commands.append(infer_command)
        commands.append(
            [
                python_bin,
                str(eval_script),
                "--benchmark-jsonl",
                str(bench_jsonl),
                "--prediction-jsonl",
                str(pred_jsonl),
                "--report-json",
                str(report_json),
                "--details-jsonl",
                str(details_jsonl),
            ]
        )

    return commands


def run_command(command: list[str], dry_run: bool):
    print("[CMD]", " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def run_phase(
    project_root: Path,
    phase_run: PhaseRun,
    prompt_file: Path,
    device: str,
    torch_dtype: str,
    min_pixels: int,
    max_pixels: int,
    workers: int,
    dry_run: bool,
    python_bin: str,
    base_model_dir: Path,
    export_config: Path,
    export_device: str,
    limit: int,
):
    phase_run.model_dir.parent.mkdir(parents=True, exist_ok=True)
    phase_run.output_root.mkdir(parents=True, exist_ok=True)
    for benchmark in phase_run.benchmarks:
        (phase_run.output_root / benchmark.name).mkdir(parents=True, exist_ok=True)

    export_runtime_model_dir(
        lora_dir=phase_run.lora_dir,
        runtime_dir=phase_run.model_dir,
        base_model_dir=base_model_dir,
        export_config=export_config,
        export_device=export_device,
        dry_run=dry_run,
    )

    commands = build_phase_commands(
        project_root=project_root,
        phase_run=phase_run,
        prompt_file=prompt_file,
        device=device,
        torch_dtype=torch_dtype,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        workers=workers,
        python_bin=python_bin,
        limit=limit,
    )

    try:
        for command in commands:
            run_command(command, dry_run=dry_run)
    finally:
        if phase_run.model_dir.exists() and not dry_run:
            shutil.rmtree(phase_run.model_dir)


def summarize_results(
    project_root: Path,
    eval_root: Path,
    csv_out: Path,
    md_out: Path,
    python_bin: str,
    dry_run: bool,
):
    summary_script = project_root / "V3" / "scripts" / "summarize_checkpoint_eval_results.py"
    command = [
        python_bin,
        str(summary_script),
        "--eval-root",
        str(eval_root),
        "--csv-out",
        str(csv_out),
        "--md-out",
        str(md_out),
    ]
    run_command(command, dry_run=dry_run)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--phase", action="append", help="Limit to one or more output directory names.")
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--min-pixels", type=int, default=50176)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--eval-root", default="V3/eval_runs_latest")
    parser.add_argument("--summary-csv", default="V3/evidence/latest_checkpoint_eval_summary.csv")
    parser.add_argument("--summary-md", default="V3/evidence/latest_checkpoint_eval_summary.md")
    parser.add_argument("--prompt-file", default="V3/configs/prompt.txt")
    parser.add_argument("--base-model-dir", default="V3/models/v2_1_export")
    parser.add_argument("--export-config", default="V3/configs/export_selected.yaml")
    parser.add_argument("--export-device", default="cpu")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    eval_root = (project_root / args.eval_root).resolve()
    summary_csv = (project_root / args.summary_csv).resolve()
    summary_md = (project_root / args.summary_md).resolve()
    prompt_file = (project_root / args.prompt_file).resolve()
    base_model_dir = (project_root / args.base_model_dir).resolve()
    export_config = (project_root / args.export_config).resolve()

    runs = build_phase_runs(
        project_root=project_root,
        phases=args.phase or list(DEFAULT_PHASES),
        eval_root=eval_root,
        all_checkpoints=args.all_checkpoints,
    )
    if not runs:
        raise FileNotFoundError("No numeric checkpoint-* directories found for the requested phases.")

    for run in runs:
        print(f"[INFO] phase={run.phase} checkpoint={run.checkpoint_dir.name}")
        run_phase(
            project_root=project_root,
            phase_run=run,
            prompt_file=prompt_file,
            device=args.device,
            torch_dtype=args.torch_dtype,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            workers=args.workers,
            dry_run=args.dry_run,
            python_bin=args.python_bin,
            base_model_dir=base_model_dir,
            export_config=export_config,
            export_device=args.export_device,
            limit=args.limit,
        )

    summarize_results(
        project_root=project_root,
        eval_root=eval_root,
        csv_out=summary_csv,
        md_out=summary_md,
        python_bin=args.python_bin,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
