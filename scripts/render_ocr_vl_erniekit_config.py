import argparse
import math
from pathlib import Path


AMP_WHITE_LIST = [
    "lookup_table",
    "lookup_table_v2",
    "flash_attn",
    "matmul",
    "matmul_v2",
    "fused_gemm_epilogue",
]

AMP_BLACK_LIST = [
    "reduce_sum",
    "softmax_with_cross_entropy",
    "c_softmax_with_cross_entropy",
    "elementwise_div",
    "sin",
    "cos",
]


def count_jsonl_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def yaml_path_text(path_value: str):
    return Path(path_value).resolve().as_posix()


def yaml_float_text(value: float, precision: int = 12):
    text = f"{float(value):.{precision}f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def str2bool(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def flag_was_passed(flag: str) -> bool:
    argv = __import__("sys").argv[1:]
    return any(token == flag or token.startswith(flag + "=") for token in argv)


def normalize_train_device(value: str):
    text = str(value).strip().lower()
    if text in {"gpu", "cuda"}:
        return "gpu"
    if text == "mlu":
        return "mlu"
    if text == "cpu":
        return "cpu"
    raise argparse.ArgumentTypeError(f"Invalid training device: {value}")


def render_yaml(args, train_count: int):
    max_steps = args.max_steps_override or max(
        1, math.ceil(train_count * args.epochs / args.gradient_accumulation_steps)
    )
    warmup_steps = args.warmup_steps_override or max(10, round(max_steps * 0.01))
    eval_steps = args.eval_steps_override or max(100, round(max_steps * 0.1))
    save_steps = args.save_steps_override or max(200, round(max_steps * 0.1))
    evaluation_strategy = args.evaluation_strategy
    if not args.do_eval and not args.evaluation_strategy_explicit:
        evaluation_strategy = "no"

    train_jsonl = yaml_path_text(args.train_jsonl)
    eval_jsonl = yaml_path_text(args.eval_jsonl)
    model_dir = yaml_path_text(args.model_dir)
    output_dir = yaml_path_text(args.output_dir)

    lines = [
        "### data",
        'train_dataset_type: "erniekit"',
        'eval_dataset_type: "erniekit"',
        f'train_dataset_path: "{train_jsonl}"',
        'train_dataset_prob: "1.0"',
        f'eval_dataset_path: "{eval_jsonl}"',
        'eval_dataset_prob: "1.0"',
        f"max_seq_len: {args.max_seq_len}",
        "num_samples_each_epoch: 6000000",
        "use_pic_id: False",
        "sft_replace_ids: True",
        "sft_image_normalize: True",
        "sft_image_rescale: True",
        'image_dtype: "float32"',
        f"variable_resolution: {args.variable_resolution}",
        "",
        "### model",
        f'model_name_or_path: "{model_dir}"',
        "fine_tuning: Full",
        "multimodal: True",
        f"use_flash_attention: {args.use_flash_attention}",
        f"use_sparse_flash_attn: {args.use_sparse_flash_attn}",
        "",
        "### finetuning",
        'stage: "OCR-VL-SFT"',
        f"seed: {args.seed}",
        f'device: "{args.device}"',
        "do_train: True",
        f"do_eval: {args.do_eval}",
        "distributed_dataloader: False",
        f"dataloader_num_workers: {args.dataloader_num_workers}",
        f"prefetch_factor: {args.prefetch_factor}",
        "batch_size: 1",
        f"packing_size: {args.packing_size}",
        f"packing: {args.packing}",
        "padding: False",
        f"num_train_epochs: {args.epochs}",
        f"max_steps: {max_steps}",
        "eval_batch_size: 1",
        f"eval_steps: {eval_steps}",
        f'evaluation_strategy: "{evaluation_strategy}"',
        f"save_steps: {save_steps}",
        f"save_total_limit: {args.save_total_limit}",
        "save_strategy: steps",
        f"logging_steps: {args.logging_steps}",
        "release_grads: True",
        f"gradient_accumulation_steps: {args.gradient_accumulation_steps}",
        f'logging_dir: "{output_dir}/tensorboard_logs"',
        f'output_dir: "{output_dir}"',
        'report_to: "none"',
        "disable_tqdm: True",
        "",
        "### train",
        f"warmup_steps: {warmup_steps}",
        f"learning_rate: {yaml_float_text(args.learning_rate)}",
        'lr_scheduler_type: "cosine"',
        f"min_lr: {yaml_float_text(args.min_lr)}",
        "layerwise_lr_decay_bound: 1.0",
        "from_scratch: 0",
        "",
        "### optimizer",
        f"weight_decay: {yaml_float_text(args.weight_decay)}",
        f"adam_epsilon: {yaml_float_text(args.adam_epsilon)}",
        f"adam_beta1: {yaml_float_text(args.adam_beta1)}",
        f"adam_beta2: {yaml_float_text(args.adam_beta2)}",
        "",
        "### performance",
        "tensor_parallel_degree: 1",
        "pipeline_parallel_degree: 1",
        "sharding_parallel_degree: 1",
        f"sharding: {args.sharding}",
        "sequence_parallel: False",
        'pipeline_parallel_config: "enable_delay_scale_loss enable_release_grads disable_partial_send_recv"',
        f"recompute: {args.recompute}",
        'recompute_granularity: "full"',
        "recompute_use_reentrant: True",
        f'compute_type: "{args.compute_type}"',
        f'fp16_opt_level: "{args.fp16_opt_level}"',
        "disable_ckpt_quant: True",
        "amp_custom_white_list:",
    ]

    for item in AMP_WHITE_LIST:
        lines.append(f"  - {item}")
    lines.append("amp_custom_black_list:")
    for item in AMP_BLACK_LIST:
        lines.append(f"  - {item}")

    lines.extend(
        [
            (f"max_pixels: {args.max_pixels}" if args.max_pixels > 0 else None),
            (f"min_pixels: {args.min_pixels}" if args.min_pixels > 0 else None),
            f"unified_checkpoint: {args.unified_checkpoint}",
            (
                f'unified_checkpoint_config: "{args.unified_checkpoint_config}"'
                if args.unified_checkpoint_config
                else None
            ),
            f"use_async_save: {args.use_async_save}",
            f"convert_from_hf: {args.convert_from_hf}",
            f"save_to_hf: {args.save_to_hf}",
            "",
        ]
    )
    return "\n".join(line for line in lines if line is not None), max_steps, warmup_steps, eval_steps, save_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", type=normalize_train_device, default="gpu")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--packing-size", type=int, default=8)
    parser.add_argument("--packing", type=str2bool, default=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--dataloader-num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--min-lr", type=float, default=5e-7)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--do-eval", type=str2bool, default=True)
    parser.add_argument("--evaluation-strategy", default="steps")
    parser.add_argument("--max-steps-override", type=int, default=0)
    parser.add_argument("--warmup-steps-override", type=int, default=0)
    parser.add_argument("--eval-steps-override", type=int, default=0)
    parser.add_argument("--save-steps-override", type=int, default=0)
    parser.add_argument("--use-flash-attention", type=str2bool, default=True)
    parser.add_argument("--use-sparse-flash-attn", type=str2bool, default=True)
    parser.add_argument("--recompute", type=str2bool, default=True)
    parser.add_argument("--variable-resolution", type=str2bool, default=True)
    parser.add_argument("--max-pixels", type=int, default=0)
    parser.add_argument("--min-pixels", type=int, default=0)
    parser.add_argument("--compute-type", default="bf16")
    parser.add_argument("--fp16-opt-level", default="O2")
    parser.add_argument("--sharding", default="stage1")
    parser.add_argument("--unified-checkpoint", type=str2bool, default=True)
    parser.add_argument("--unified-checkpoint-config", default="")
    parser.add_argument("--use-async-save", type=str2bool, default=False)
    parser.add_argument("--convert-from-hf", type=str2bool, default=True)
    parser.add_argument("--save-to-hf", type=str2bool, default=True)
    parser.add_argument("--epochs", type=int, default=0)
    args = parser.parse_args()

    if args.device == "mlu":
        if not flag_was_passed("--packing-size"):
            args.packing_size = 1
        if not flag_was_passed("--gradient-accumulation-steps"):
            args.gradient_accumulation_steps = 1
        if not flag_was_passed("--dataloader-num-workers"):
            args.dataloader_num_workers = 2
        if not flag_was_passed("--prefetch-factor"):
            args.prefetch_factor = 2
        if not flag_was_passed("--use-flash-attention"):
            args.use_flash_attention = False
        if not flag_was_passed("--use-sparse-flash-attn"):
            args.use_sparse_flash_attn = False
        if not flag_was_passed("--recompute"):
            args.recompute = False
        if not flag_was_passed("--save-to-hf"):
            args.save_to_hf = False
        if not flag_was_passed("--compute-type"):
            args.compute_type = "fp16"
        if not flag_was_passed("--fp16-opt-level"):
            args.fp16_opt_level = "O1"

    if args.epochs <= 0:
        args.epochs = 1 if args.preset == "quick" else 2
    args.evaluation_strategy_explicit = flag_was_passed("--evaluation-strategy")

    train_jsonl = Path(args.train_jsonl).resolve()
    train_count = count_jsonl_rows(train_jsonl)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config_path).resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_text, max_steps, warmup_steps, eval_steps, save_steps = render_yaml(args, train_count)
    config_path.write_text(yaml_text, encoding="utf-8")

    print("Preset:", args.preset)
    print("Train samples:", train_count)
    print("Epochs:", args.epochs)
    print("Max steps:", max_steps)
    print("Warmup steps:", warmup_steps)
    print("Eval steps:", eval_steps)
    print("Save steps:", save_steps)
    print("Config:", config_path)


if __name__ == "__main__":
    main()
