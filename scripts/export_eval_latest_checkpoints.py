import argparse
import dataclasses
import re
import shutil
import subprocess
from pathlib import Path


DEFAULT_PHASES = ["singleline_rw_lora"]


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
    model_dir: Path
    output_root: Path
    benchmarks: tuple[BenchmarkSpec, ...]


DEFAULT_BENCHMARKS = (
    BenchmarkSpec(
        name="canonical_main",
        jsonl_rel_path=Path("V2/data/eval/canonical_smiles_main_v1/annotations/labels.jsonl"),
        max_new_tokens=256,
    ),
    BenchmarkSpec(
        name="mixed_v1p1",
        jsonl_rel_path=Path("V2/data/eval/ocsr_realworld_mixed_eval_v1p1/annotations/labels.jsonl"),
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
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
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


def build_phase_runs(
    project_root: Path,
    phases: list[str] | None = None,
    eval_root: Path | None = None,
    benchmarks: tuple[BenchmarkSpec, ...] = DEFAULT_BENCHMARKS,
    all_checkpoints: bool = False,
) -> list[PhaseRun]:
    selected_phases = phases or list(DEFAULT_PHASES)
    effective_eval_root = eval_root or (project_root / "V2" / "eval_runs_latest")

    runs = []
    for phase in selected_phases:
        phase_dir = project_root / "V2" / "outputs" / phase
        if not phase_dir.exists():
            continue
        checkpoints = list_numeric_checkpoints(phase_dir) if all_checkpoints else []
        if not all_checkpoints:
            latest = find_latest_numeric_checkpoint(phase_dir)
            checkpoints = [latest] if latest is not None else []
        if not checkpoints:
            continue
        for checkpoint_dir in checkpoints:
            runs.append(
                PhaseRun(
                    phase=phase,
                    phase_root=phase_dir,
                    checkpoint_dir=checkpoint_dir,
                    model_dir=effective_eval_root / phase / f"{phase}_{checkpoint_dir.name}_runtime",
                    output_root=effective_eval_root / phase / checkpoint_dir.name,
                    benchmarks=benchmarks,
                )
            )
    return runs


def stage_runtime_model_dir(phase_root: Path, checkpoint_dir: Path, runtime_dir: Path, base_model_dir: Path | None = None):
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    for path in phase_root.iterdir():
        if not path.is_file():
            continue
        if path.name not in RUNTIME_METADATA_NAMES:
            continue
        shutil.copy2(path, runtime_dir / path.name)

    for path in checkpoint_dir.iterdir():
        if not path.is_file():
            continue
        shutil.copy2(path, runtime_dir / path.name)

    if base_model_dir is not None and base_model_dir.exists():
        for path in base_model_dir.iterdir():
            if not path.is_file():
                continue
            if path.name not in RUNTIME_METADATA_NAMES:
                continue
            target = runtime_dir / path.name
            if not target.exists():
                shutil.copy2(path, target)

    if not (runtime_dir / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json in phase root: {phase_root}")


def build_phase_commands(
    project_root: Path,
    phase_run: PhaseRun,
    prompt_file: Path,
    device: str,
    torch_dtype: str,
    min_pixels: int,
    max_pixels: int,
    python_bin: str = "python",
) -> list[list[str]]:
    commands: list[list[str]] = []

    infer_script = project_root / "V2" / "scripts" / "infer_ocsr_transformers.py"
    eval_script = project_root / "V2" / "scripts" / "evaluate_ocsr_predictions_detailed.py"

    for benchmark in phase_run.benchmarks:
        benchmark_dir = phase_run.output_root / benchmark.name
        pred_jsonl = benchmark_dir / "pred.jsonl"
        report_json = benchmark_dir / "report.json"
        details_jsonl = benchmark_dir / "details.jsonl"
        bench_jsonl = project_root / benchmark.jsonl_rel_path

        commands.append(
            [
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
        )
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
    dry_run: bool,
    python_bin: str,
    base_model_dir: Path,
):
    phase_run.model_dir.parent.mkdir(parents=True, exist_ok=True)
    phase_run.output_root.mkdir(parents=True, exist_ok=True)
    for benchmark in phase_run.benchmarks:
        (phase_run.output_root / benchmark.name).mkdir(parents=True, exist_ok=True)

    stage_runtime_model_dir(phase_run.phase_root, phase_run.checkpoint_dir, phase_run.model_dir, base_model_dir)

    commands = build_phase_commands(
        project_root=project_root,
        phase_run=phase_run,
        prompt_file=prompt_file,
        device=device,
        torch_dtype=torch_dtype,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        python_bin=python_bin,
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
    summary_script = project_root / "V2" / "scripts" / "summarize_checkpoint_eval_results.py"
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
    parser.add_argument("--phase", action="append", choices=DEFAULT_PHASES, help="Limit to one or more phases.")
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--min-pixels", type=int, default=50176)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument("--eval-root", default="V2/eval_runs_latest")
    parser.add_argument("--summary-csv", default="V2/reports/latest_checkpoint_eval_summary.csv")
    parser.add_argument("--summary-md", default="V2/reports/latest_checkpoint_eval_summary.md")
    parser.add_argument("--prompt-file", default="V2/configs/prompt.txt")
    parser.add_argument("--base-model-dir", default="models/PaddleOCR-VL-0.9B")
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
            dry_run=args.dry_run,
            python_bin=args.python_bin,
            base_model_dir=base_model_dir,
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
