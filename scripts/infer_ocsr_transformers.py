import argparse
import json
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


DEFAULT_PROMPT = (
    "OCR: Output only the canonical SMILES string for the molecule shown in the image."
)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_prompt(prompt_file: Path | None, prompt_text: str | None):
    if prompt_text:
        return prompt_text.strip()
    if prompt_file and prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8").strip()
    return DEFAULT_PROMPT


def load_prompt_list(prompt_file: Path | None, prompt_text: str | None, prompt_list_file: Path | None):
    prompts = []
    if prompt_list_file and prompt_list_file.exists():
        for line in prompt_list_file.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text:
                prompts.append(text)
    else:
        prompts.append(load_prompt(prompt_file, prompt_text))
    return prompts


def cleanup_prediction(text: str, prompt: str):
    cleaned = text.strip()
    prefixes = [
        prompt,
        f"User:{prompt}",
        f"User: {prompt}",
        f"user:{prompt}",
        f"user: {prompt}",
        "assistant",
        "Assistant",
        "user",
        "User",
        "User:",
        "user:",
        "Assistant:",
        "assistant:",
        "OCR:",
    ]
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""
    for line in reversed(lines):
        if line.lower() not in {"assistant", "assistant:", "user", "user:"}:
            return line
    return lines[-1]


def canonicalize_smiles(smiles_text: str):
    import re
    from rdkit import Chem

    text = str(smiles_text or "").strip()
    if not text:
        return None
    text = re.sub(r"\s+", "", text)
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def build_tta_images(image: Image.Image, tta_preset: str):
    base = image.convert("RGB")
    variants = [("orig", base)]

    if tta_preset == "none":
        return variants

    gray = ImageOps.grayscale(base).convert("RGB")
    auto = ImageOps.autocontrast(gray)
    high = ImageEnhance.Contrast(auto).enhance(1.8)
    sharp = ImageEnhance.Sharpness(high).enhance(1.6)
    variants.extend(
        [
            ("gray_auto", auto),
            ("high_contrast", high),
            ("sharp_contrast", sharp),
        ]
    )
    return variants


def predict_once(model, processor, image: Image.Image, prompt: str, max_new_tokens: int):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return cleanup_prediction(decoded, prompt), decoded


def choose_best_candidate(candidates):
    valid = [item for item in candidates if item["canonical_prediction"]]
    if valid:
        frequency = {}
        for item in valid:
            key = item["canonical_prediction"]
            frequency[key] = frequency.get(key, 0) + 1

        ranked = sorted(
            valid,
            key=lambda item: (
                -frequency[item["canonical_prediction"]],
                len(item["prediction"]),
                item["prompt_index"],
                item["tta_index"],
            ),
        )
        best = dict(ranked[0])
        best["selection_reason"] = "valid_majority_vote"
        best["vote_count"] = frequency[best["canonical_prediction"]]
        return best

    non_empty = [item for item in candidates if item["prediction"]]
    if non_empty:
        best = dict(non_empty[0])
        best["selection_reason"] = "first_non_empty"
        best["vote_count"] = 1
        return best

    best = dict(candidates[0])
    best["selection_reason"] = "fallback_empty"
    best["vote_count"] = 1
    return best


def predict_with_ensemble(model, processor, image_path: Path, prompts, max_new_tokens: int, tta_preset: str):
    base_image = Image.open(image_path).convert("RGB")
    tta_images = build_tta_images(base_image, tta_preset)
    candidates = []

    for prompt_index, prompt in enumerate(prompts):
        for tta_index, (tta_name, image_variant) in enumerate(tta_images):
            prediction, raw_text = predict_once(
                model,
                processor,
                image_variant,
                prompt,
                max_new_tokens,
            )
            candidates.append(
                {
                    "prompt": prompt,
                    "prompt_index": prompt_index,
                    "tta_name": tta_name,
                    "tta_index": tta_index,
                    "prediction": prediction,
                    "canonical_prediction": canonicalize_smiles(prediction),
                    "raw_text": raw_text,
                }
            )

    best = choose_best_candidate(candidates)
    return best, candidates


def resolve_runtime_image_path(raw_path: str, project_root: Path | None, benchmark_path: Path | None):
    raw = Path(raw_path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        if project_root is not None:
            candidates.append(project_root / raw)
        if benchmark_path is not None:
            candidates.append(benchmark_path.parent / raw)
            candidates.append(benchmark_path.parent.parent / raw)
            candidates.append(benchmark_path.parent.parent.parent / raw)
        candidates.append(raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return raw.resolve() if raw.is_absolute() else raw


def get_record_image_ref(record: dict) -> str:
    if "image_path" in record and str(record.get("image_path", "")).strip():
        return str(record["image_path"])
    if "image" in record and str(record.get("image", "")).strip():
        return str(record["image"])
    raise KeyError("Record is missing both 'image_path' and 'image' fields")


def resolve_device(torch, requested_device: str):
    device = requested_device.strip().lower()
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        try:
            import torch_mlu  # noqa: F401
        except ImportError:
            pass
        if hasattr(torch, "mlu") and torch.mlu.is_available():
            return "mlu"
        return "cpu"

    if device == "mlu":
        try:
            import torch_mlu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Requested device=mlu, but torch_mlu is not installed in the inference environment."
            ) from exc
        if not hasattr(torch, "mlu") or not torch.mlu.is_available():
            raise RuntimeError("Requested device=mlu, but no MLU device is available.")
        return "mlu"

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device=cuda, but CUDA is not available.")
        return "cuda"

    if device == "cpu":
        return "cpu"

    raise ValueError(f"Unsupported device: {requested_device}")


def resolve_torch_dtype(torch, requested_dtype: str, device: str):
    name = requested_dtype.strip().lower()
    if name == "auto":
        if device == "cuda":
            return torch.bfloat16
        if device == "mlu":
            return torch.float16
        return torch.float32

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {requested_dtype}")
    return mapping[name]


def iter_model_safetensor_shards(model_dir: Path):
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        filenames = sorted(set(payload.get("weight_map", {}).values()))
        for name in filenames:
            path = model_dir / name
            if path.exists():
                yield path
        return

    for path in sorted(model_dir.glob("model*.safetensors")):
        if path.is_file():
            yield path


def load_model_from_unified_checkpoint(
    model_dir: Path,
    torch,
    trust_remote_code: bool,
    torch_dtype,
    attn_implementation: str,
):
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        model_dir,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation or None,
    )
    target_state = model.state_dict()

    shard_paths = list(iter_model_safetensor_shards(model_dir))
    if not shard_paths:
        raise FileNotFoundError(
            f"No model safetensors shards found under: {model_dir}"
        )

    missing_keys = []
    unexpected_keys = []
    for shard_path in shard_paths:
        state_dict = {}
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                target_tensor = target_state.get(key)
                if (
                    target_tensor is not None
                    and tensor.ndim == 2
                    and tuple(tensor.shape) != tuple(target_tensor.shape)
                    and tuple(tensor.shape[::-1]) == tuple(target_tensor.shape)
                ):
                    tensor = tensor.transpose(0, 1).contiguous()
                state_dict[key] = tensor
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing_keys = incompatible.missing_keys
        unexpected_keys.extend(incompatible.unexpected_keys)
        del state_dict

    if missing_keys:
        print(
            f"[infer_ocsr_transformers] WARNING: missing keys after unified checkpoint load: "
            f"{missing_keys[:20]}",
            flush=True,
        )
    if unexpected_keys:
        print(
            f"[infer_ocsr_transformers] WARNING: unexpected keys after unified checkpoint load: "
            f"{unexpected_keys[:20]}",
            flush=True,
        )

    return model


def load_causal_lm_model(model_dir: Path, model_kwargs, torch):
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_dir,
            **model_kwargs,
        )
    except (OSError, RuntimeError) as exc:
        message = str(exc)
        if (
            "does not contain the valid metadata" not in message
            and "size mismatch for weight" not in message
        ):
            raise

        print(
            "[infer_ocsr_transformers] Falling back to manual shard loading for "
            "an ERNIEKit unified checkpoint.",
            flush=True,
        )
        return load_model_from_unified_checkpoint(
            model_dir=model_dir,
            torch=torch,
            trust_remote_code=model_kwargs.get("trust_remote_code", False),
            torch_dtype=model_kwargs.get("torch_dtype"),
            attn_implementation=model_kwargs.get("attn_implementation", ""),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--benchmark-jsonl", default="")
    parser.add_argument("--image", default="")
    parser.add_argument("--project-root", default="")
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--prompt-text", default="")
    parser.add_argument("--prompt-list-file", default="")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tta-preset", choices=["none", "light"], default="none")
    parser.add_argument("--save-candidates", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--min-pixels", type=int, default=0)
    parser.add_argument("--max-pixels", type=int, default=0)
    args = parser.parse_args()

    prompt_file = Path(args.prompt_file).resolve() if args.prompt_file else None
    prompt_list_file = Path(args.prompt_list_file).resolve() if args.prompt_list_file else None
    project_root = Path(args.project_root).resolve() if args.project_root else None
    prompts = load_prompt_list(prompt_file, args.prompt_text, prompt_list_file)

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    device = resolve_device(torch, args.device)
    torch_dtype = resolve_torch_dtype(torch, args.torch_dtype, device)
    model_dir = Path(args.model_dir).resolve()

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = load_causal_lm_model(
        model_dir=model_dir,
        model_kwargs=model_kwargs,
        torch=torch,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    if hasattr(processor, "image_processor"):
        if args.min_pixels > 0 and hasattr(processor.image_processor, "min_pixels"):
            processor.image_processor.min_pixels = args.min_pixels
        if args.max_pixels > 0 and hasattr(processor.image_processor, "max_pixels"):
            processor.image_processor.max_pixels = args.max_pixels

    if args.image:
        image_path = Path(args.image).resolve()
        best, candidates = predict_with_ensemble(
            model,
            processor,
            image_path,
            prompts,
            args.max_new_tokens,
            args.tta_preset,
        )
        result = {
            "image_path": str(image_path),
            "prompt": best["prompt"],
            "prediction": best["prediction"],
            "canonical_prediction": best["canonical_prediction"],
            "raw_text": best["raw_text"],
            "selection_reason": best["selection_reason"],
            "vote_count": best["vote_count"],
        }
        if args.save_candidates:
            result["candidates"] = candidates
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if not args.benchmark_jsonl:
        raise ValueError("Either --image or --benchmark-jsonl is required.")

    benchmark_path = Path(args.benchmark_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve() if args.output_jsonl else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    records = list(read_jsonl(benchmark_path))
    if args.limit > 0:
        records = records[: args.limit]

    if output_path:
        handle = output_path.open("w", encoding="utf-8")
    else:
        handle = None

    try:
        for record in records:
            image_path = resolve_runtime_image_path(
                get_record_image_ref(record),
                project_root,
                benchmark_path,
            )
            best, candidates = predict_with_ensemble(
                model,
                processor,
                image_path,
                prompts,
                args.max_new_tokens,
                args.tta_preset,
            )
            row = {
                "id": record["id"],
                "image_path": str(image_path),
                "prompt": best["prompt"],
                "prediction": best["prediction"],
                "canonical_prediction": best["canonical_prediction"],
                "raw_text": best["raw_text"],
                "selection_reason": best["selection_reason"],
                "vote_count": best["vote_count"],
            }
            if args.save_candidates:
                row["candidates"] = candidates
            line = json.dumps(row, ensure_ascii=False)
            if handle:
                handle.write(line + "\n")
            else:
                print(line)
    finally:
        if handle:
            handle.close()


if __name__ == "__main__":
    main()
