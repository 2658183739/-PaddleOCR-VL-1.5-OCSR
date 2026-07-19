import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def merge_shards(benchmark_path: Path, part_paths: list[Path], output_path: Path):
    benchmark_rows = list(read_jsonl(benchmark_path))
    ordered_ids = [str(row["id"]) for row in benchmark_rows]
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("Benchmark IDs must be unique")

    predictions = {}
    for part_path in part_paths:
        for row in read_jsonl(part_path):
            row_id = str(row["id"])
            if row_id in predictions:
                raise ValueError(f"Duplicate prediction ID: {row_id}")
            predictions[row_id] = row

    missing = [row_id for row_id in ordered_ids if row_id not in predictions]
    extra = sorted(set(predictions) - set(ordered_ids))
    if missing or extra:
        raise ValueError(
            f"Shard merge mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row_id in ordered_ids:
            handle.write(json.dumps(predictions[row_id], ensure_ascii=False) + "\n")


def build_command(args, infer_script: Path, shard_index: int, part_path: Path):
    command = [
        args.python_bin,
        str(infer_script),
        "--model-dir",
        str(Path(args.model_dir).resolve()),
        "--benchmark-jsonl",
        str(Path(args.benchmark_jsonl).resolve()),
        "--project-root",
        str(Path(args.project_root).resolve()),
        "--output-jsonl",
        str(part_path),
        "--device",
        args.device,
        "--torch-dtype",
        args.torch_dtype,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--min-pixels",
        str(args.min_pixels),
        "--max-pixels",
        str(args.max_pixels),
        "--tta-preset",
        args.tta_preset,
        "--num-beams",
        str(args.num_beams),
        "--num-return-sequences",
        str(args.num_return_sequences),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--repetition-penalty",
        str(args.repetition_penalty),
        "--no-repeat-ngram-size",
        str(args.no_repeat_ngram_size),
        "--shard-index",
        str(shard_index),
        "--num-shards",
        str(args.workers),
    ]
    if args.prompt_file:
        command.extend(["--prompt-file", str(Path(args.prompt_file).resolve())])
    if args.attn_implementation:
        command.extend(["--attn-implementation", args.attn_implementation])
    if args.do_sample:
        command.append("--do-sample")
    if args.save_candidates:
        command.append("--save-candidates")
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--benchmark-jsonl", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--min-pixels", type=int, default=50176)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--tta-preset", choices=["none", "light"], default="none")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--save-candidates", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("workers must be at least 1")

    infer_script = Path(__file__).resolve().with_name("infer_ocsr_transformers.py")
    output_path = Path(args.output_jsonl).resolve()
    shard_root = output_path.parent / f"{output_path.stem}_shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    processes = []
    log_handles = []
    part_paths = []
    try:
        for shard_index in range(args.workers):
            part_path = shard_root / f"part-{shard_index:02d}.jsonl"
            log_path = shard_root / f"part-{shard_index:02d}.log"
            part_paths.append(part_path)
            log_handle = log_path.open("w", encoding="utf-8")
            log_handles.append(log_handle)
            command = build_command(args, infer_script, shard_index, part_path)
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
            )

        failures = []
        for shard_index, process in enumerate(processes):
            return_code = process.wait()
            if return_code != 0:
                failures.append((shard_index, return_code))
        if failures:
            raise RuntimeError(f"Inference shards failed: {failures}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for log_handle in log_handles:
            log_handle.close()

    merge_shards(Path(args.benchmark_jsonl).resolve(), part_paths, output_path)
    print(f"Merged {len(list(read_jsonl(output_path)))} predictions into {output_path}")


if __name__ == "__main__":
    main()
