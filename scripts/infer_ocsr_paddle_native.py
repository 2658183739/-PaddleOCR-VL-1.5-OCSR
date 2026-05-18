import argparse
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


DEFAULT_PROMPT = (
    "OCR: Output only the canonical SMILES string for the molecule shown in the image."
)
DEFAULT_PREFIX = "<think>\n\n</think>\n\n"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def str2bool(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_prompt(prompt_file: Path | None, prompt_text: str | None):
    if prompt_text:
        return prompt_text.strip()
    if prompt_file and prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8").strip()
    return DEFAULT_PROMPT


def load_prompt_list(
    prompt_file: Path | None,
    prompt_text: str | None,
    prompt_list_file: Path | None,
):
    prompts = []
    if prompt_list_file and prompt_list_file.exists():
        for line in prompt_list_file.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text:
                prompts.append(text)
    else:
        prompts.append(load_prompt(prompt_file, prompt_text))
    return prompts


def cleanup_prediction(text: str, prompt: str, prefix: str):
    cleaned = str(text or "").replace("\r\n", "\n").strip()

    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.S).strip()

    prefixes = [
        prefix,
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
    for maybe_prefix in prefixes:
        maybe_prefix = str(maybe_prefix or "").strip()
        if maybe_prefix and cleaned.startswith(maybe_prefix):
            cleaned = cleaned[len(maybe_prefix) :].strip()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""
    for line in reversed(lines):
        lowered = line.lower()
        if lowered in {"assistant", "assistant:", "user", "user:"}:
            continue
        if line.startswith("Assistant:"):
            return line[len("Assistant:") :].strip()
        if line.startswith("assistant:"):
            return line[len("assistant:") :].strip()
        return line
    return lines[-1]


def canonicalize_smiles(smiles_text: str):
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


def bootstrap_ernie_imports(ernie_dir: Path):
    ernie_root = str(ernie_dir.resolve())
    if ernie_root not in sys.path:
        sys.path.insert(0, ernie_root)


class NativePredictor:
    def __init__(self, args):
        self.args = args
        self.ernie_dir = Path(args.ernie_dir).resolve()
        bootstrap_ernie_imports(self.ernie_dir)

        if args.device == "mlu" and args.set_mlu_env:
            os.environ.setdefault("ACCELERATOR_BACKEND", "mlu")
            os.environ.setdefault("PADDLE_XCCL_BACKEND", "mlu")

        import paddle

        from data_processor.image_preprocessor.image_preprocessor_adaptive import (
            AdaptiveImageProcessor,
        )
        from data_processor.steps.end2end_processing import End2EndProcessor
        from ernie.configuration_paddleocr_vl import PaddleOCRVLConfig
        from ernie.modeling_paddleocr_vl import PaddleOCRVLForConditionalGeneration
        from ernie.tokenizer_vl import Ernie4_5_VLTokenizer
        from ernie.utils.mm_data_utils import MMSpecialTokensConfig
        from erniekit.hparams.preprocess_args import End2EndProcessorArguments

        self.paddle = paddle
        self.AdaptiveImageProcessor = AdaptiveImageProcessor
        self.End2EndProcessor = End2EndProcessor
        self.PaddleOCRVLConfig = PaddleOCRVLConfig
        self.PaddleOCRVLForConditionalGeneration = PaddleOCRVLForConditionalGeneration
        self.Ernie4_5_VLTokenizer = Ernie4_5_VLTokenizer
        self.MMSpecialTokensConfig = MMSpecialTokensConfig
        self.End2EndProcessorArguments = End2EndProcessorArguments

        self.black = [
            "reduce_sum",
            "c_softmax_with_cross_entropy",
            "elementwise_div",
            "sin",
            "cos",
            "sort",
            "multinomial",
        ]
        self.white = [
            "lookup_table",
            "lookup_table_v2",
            "flash_attn",
            "matmul",
            "matmul_v2",
            "fused_gemm_epilogue",
        ]

        self.model_dir = Path(args.model_dir).resolve()
        self.base_model_dir = Path(args.base_model_dir).resolve()
        self.vision_model_dir = (
            Path(args.vision_model_name_or_path).resolve()
            if args.vision_model_name_or_path
            else self.base_model_dir
        )
        self.tokenizer_dir = (
            Path(args.tokenizer_dir).resolve()
            if args.tokenizer_dir
            else self.base_model_dir
        )

        self.paddle.set_device(args.device)
        self.dtype = args.dtype

        self._init_tokenizer_and_processor()
        self._init_model()

    def _build_processor_args(self):
        adaptive_max_imgtoken_option = self.args.crop_tile_option or None
        adaptive_max_imgtoken_rate = self.args.crop_tile_rate or None
        explicit_max_pixels = self.args.max_pixels if self.args.max_pixels > 0 else None
        explicit_min_pixels = (
            self.args.min_pixels if self.args.min_pixels >= 0 else None
        )

        # In inference, an explicit pixel budget should take precedence over any
        # random adaptive image-token schedule. Otherwise the processor may
        # silently shrink `max_pixels` again and reject tall/wide molecules.
        if explicit_max_pixels is not None or explicit_min_pixels is not None:
            adaptive_max_imgtoken_option = None
            adaptive_max_imgtoken_rate = None

        # The base PaddleOCR-VL processor ships with a fairly high default
        # `min_pixels` (~147k). If we only lower `max_pixels` for inference,
        # that inherited lower bound can still reject long molecular diagrams.
        if explicit_max_pixels is not None and explicit_min_pixels is None:
            explicit_min_pixels = 0

        processor_args = self.End2EndProcessorArguments(
            tokenizer=str(self.tokenizer_dir),
            tokenizer_name=str(self.tokenizer_dir),
            use_pic_id=self.args.use_pic_id,
            variable_resolution=self.args.variable_resolution,
            adaptive_max_imgtoken_option=adaptive_max_imgtoken_option,
            adaptive_max_imgtoken_rate=adaptive_max_imgtoken_rate,
            max_pixels=explicit_max_pixels,
            min_pixels=explicit_min_pixels,
            chat_template=self.args.chat_template,
            image_dtype=self.args.image_dtype,
            sft_replace_ids=self.args.sft_replace_ids,
            sft_image_rescale=self.args.sft_image_rescale,
            sft_image_normalize=self.args.sft_image_normalize,
            load_args_from_api=True,
        )
        processor_args.batch_size = 1
        processor_args.max_seq_length = self.args.max_seq_length
        processor_args.serialize_output = False
        processor_args.data_filelist = None
        processor_args.corpus_name = "ocsr_native_infer"
        return processor_args

    def _init_tokenizer_and_processor(self):
        tokenizer = self.Ernie4_5_VLTokenizer.from_pretrained(
            str(self.tokenizer_dir),
            model_max_length=self.args.max_seq_length,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.ignored_index = -100
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        self.tokenizer = tokenizer

        self.image_preprocess = self.AdaptiveImageProcessor.from_pretrained(
            str(self.vision_model_dir)
        )

        processor_args = self._build_processor_args()
        self.processor = self.End2EndProcessor(
            processor_args,
            tokenizer=self.tokenizer,
            image_preprocess=self.image_preprocess,
        )
        self.processor.eval()
        self.processor.sft()

    def _init_model(self):
        config = self.PaddleOCRVLConfig.from_pretrained(
            str(self.model_dir),
        )
        config.use_cache = False
        config.max_sequence_length = self.args.max_seq_length
        config.seqlen = self.args.max_seq_length
        config.dtype = self.dtype
        config.tensor_parallel_degree = 1
        config.tensor_parallel_rank = 0
        config.vision_config.tensor_parallel_degree = 1
        config.vision_config.tensor_parallel_rank = 0
        config.pixel_hidden_size = config.vision_config.hidden_size
        config.im_patch_id = self.tokenizer.get_vocab()[
            self.MMSpecialTokensConfig.get_special_tokens_info()["image_placeholder"]
        ]
        config.max_text_id = config.im_patch_id
        config.use_flash_attn = self.args.use_flash_attn
        config.vision_config.use_flash_attention = self.args.use_flash_attn
        config.use_flash_attention = self.args.use_flash_attn
        config.use_flash_attn_with_mask = self.args.use_flash_attn
        config.use_sparse_flash_attn = False
        config.vision_config.use_sparse_flash_attn = False
        config.use_sparse_head_and_loss_fn = False
        config.use_fused_head_and_loss_fn = False
        config.use_mem_eff_attn = False
        config.recompute = False
        config.vision_config.recompute = False
        config.sequence_parallel = False

        self.paddle.set_default_dtype(self.dtype)
        model = self.PaddleOCRVLForConditionalGeneration.from_pretrained(
            str(self.model_dir),
            config=config,
            convert_from_hf=False,
        )
        self.config = model.config
        self.vision_config = config.vision_config

        if self.dtype != "float32":
            model = self.paddle.amp.decorate(models=model, level="O2", dtype=self.dtype)

        model.eval()
        self.model = model

    def _build_request(self, image_ref, prompt: str):
        return {
            "context": [
                {
                    "role": "user",
                    "utterance": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_ref,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            "top_p": self.args.top_p,
            "temperature": self.args.temperature,
            "penalty_score": self.args.penalty_score,
            "frequency_score": self.args.frequency_score,
            "presence_score": self.args.presence_score,
            "max_dec_len": self.args.max_dec_len,
            "stop_sequences": [],
            "top_k": self.args.top_k,
            "seed": self.args.seed,
            "prefix": self.args.prefix,
        }

    def _preprocess(self, request_data):
        generation_configs = {
            "max_length": request_data["max_dec_len"],
            "stop_sequences": request_data["stop_sequences"],
            "top_p": request_data["top_p"],
            "temperature": request_data["temperature"],
            "top_k": request_data["top_k"],
            "penalty_score": request_data["penalty_score"],
            "frequency_score": request_data["frequency_score"],
            "presence_score": request_data["presence_score"],
            "eos_token_id": self.tokenizer._convert_token_to_id(self.tokenizer.eos_token),
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        processed = self.processor.process(request_data)
        if not processed:
            raise ValueError(
                "Processor returned no samples. The image was likely rejected by "
                "adaptive resize/token budgeting; try disabling crop-tile "
                "adaptation or increasing --max-pixels."
            )
        one = processed[0]
        input_ids = one["input_ids"][None, :]
        token_type_ids = one["token_type_ids"][None, :]
        position_ids = one.get("position_ids", None)
        if position_ids is not None:
            position_ids = position_ids[None, :]

        if one.get("images", None) is not None:
            image_type_ids = one["image_type_ids"][None, :]
            images = one["images"]
            grid_thw = one.get("grid_thw", None)

            self.image_preprocess.image_mean_tensor = self.paddle.to_tensor(
                self.image_preprocess.image_mean, dtype="float32"
            ).reshape([1, 3, 1, 1])
            self.image_preprocess.image_std_tensor = self.paddle.to_tensor(
                self.image_preprocess.image_std, dtype="float32"
            ).reshape([1, 3, 1, 1])
            self.image_preprocess.rescale_factor = self.paddle.to_tensor(
                self.image_preprocess.rescale_factor, dtype="float32"
            )
            self.image_preprocess.image_mean_tensor = self.image_preprocess.image_mean_tensor.squeeze(
                [-2, -1]
            ).repeat_interleave(self.vision_config.patch_size**2, -1)
            self.image_preprocess.image_std_tensor = self.image_preprocess.image_std_tensor.squeeze(
                [-2, -1]
            ).repeat_interleave(self.vision_config.patch_size**2, -1)

            images = self.image_preprocess.rescale_factor * images.astype("float32")
            images = (images - self.image_preprocess.image_mean_tensor) / self.image_preprocess.image_std_tensor

            input_ids = self.paddle.to_tensor(input_ids, dtype=self.paddle.int64)
            image_type_ids = self.paddle.to_tensor(image_type_ids, dtype=self.paddle.int64)
            token_type_ids = self.paddle.to_tensor(token_type_ids, dtype=self.paddle.int64)
            images = self.paddle.to_tensor(images, dtype=self.dtype)
            if grid_thw is not None:
                grid_thw = self.paddle.to_tensor(grid_thw, dtype=self.paddle.int64)
        else:
            image_type_ids, images, grid_thw = None, None, None
            input_ids = self.paddle.to_tensor(input_ids, dtype=self.paddle.int64)
            token_type_ids = self.paddle.to_tensor(token_type_ids, dtype=self.paddle.int64)

        if position_ids is not None:
            position_ids = self.paddle.to_tensor(position_ids, dtype=self.paddle.int64)

        return (
            {
                "input_ids": input_ids,
                "image_type_ids": image_type_ids,
                "token_type_ids": token_type_ids,
                "images": images,
                "grid_thw": grid_thw,
                "position_ids": position_ids,
            },
            generation_configs,
        )

    def _infer(self, inputs, **kwargs):
        base_generate_kwargs = {
            "max_length": kwargs.get("max_length", self.args.max_dec_len),
            "top_p": kwargs.get("top_p", self.args.top_p),
            "temperature": kwargs.get("temperature", self.args.temperature),
            "top_k": kwargs.get("top_k", self.args.top_k),
            "penalty_score": kwargs.get("penalty_score", self.args.penalty_score),
            "frequency_score": kwargs.get("frequency_score", self.args.frequency_score),
            "presence_score": kwargs.get("presence_score", self.args.presence_score),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            # PaddleOCR-VL's cached decode path is fragile on our current
            # MLU/native route because RoPE deltas are resumed through a
            # model-local ContextVar. Disabling cache is slower but avoids the
            # broken second-step path and is acceptable for smoke/eval runs.
            "use_cache": kwargs.get("use_cache", False),
        }
        base_generate_kwargs = {
            key: value for key, value in base_generate_kwargs.items() if value is not None
        }

        with self.paddle.no_grad():
            with self.paddle.amp.auto_cast(
                True,
                custom_black_list=self.black,
                custom_white_list=self.white,
                level="O2",
                dtype=self.dtype,
            ):
                try:
                    # Prefer bypassing the PaddleOCR-VL wrapper so we can keep
                    # the same generation arguments as the official dynamic
                    # inference path.
                    base_generate = super(type(self.model), self.model).generate
                    out = base_generate(
                        input_ids=inputs["input_ids"],
                        token_type_ids=inputs["token_type_ids"],
                        image_type_ids=inputs["image_type_ids"],
                        images=inputs["images"],
                        grid_thw=inputs["grid_thw"],
                        position_ids=inputs["position_ids"],
                        **base_generate_kwargs,
                    )
                except Exception:
                    # Fallback to the OCR model's local wrapper, which expects
                    # a single positional `inputs` dictionary.
                    out = self.model.generate(
                        inputs,
                        max_new_tokens=base_generate_kwargs.get(
                            "max_length", self.args.max_dec_len
                        ),
                        use_cache=base_generate_kwargs.get("use_cache", False),
                    )
        return out

    def _postprocess(self, predictions, stop_sequences):
        results = []
        for text_tensor in predictions:
            text_str = self.tokenizer.decode(
                text_tensor.numpy().tolist(),
                skip_special_tokens=False,
            )
            for stop in stop_sequences:
                text_str = text_str.split(stop)[0]
            results.append(text_str)
        return results

    def predict_text(self, image_ref, prompt: str):
        request_data = self._build_request(image_ref, prompt)
        tokenized_source, generation_configs = self._preprocess(request_data)
        predictions = self._infer(tokenized_source, **generation_configs)
        decoded = self._postprocess(
            predictions[0],
            stop_sequences=generation_configs.get("stop_sequences", []),
        )[0]
        return decoded


def predict_once(predictor: NativePredictor, image_ref, prompt: str):
    raw_text = predictor.predict_text(image_ref, prompt)
    cleaned = cleanup_prediction(raw_text, prompt, predictor.args.prefix)
    return cleaned, raw_text


def materialize_variant_image(image_variant: Image.Image):
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    image_variant.save(tmp_path, format="PNG")
    return tmp_path


def predict_with_ensemble(
    predictor: NativePredictor,
    image_path: Path,
    prompts,
    tta_preset: str,
):
    base_image = Image.open(image_path).convert("RGB")
    tta_images = build_tta_images(base_image, tta_preset)
    candidates = []
    temp_files = []

    try:
        for prompt_index, prompt in enumerate(prompts):
            for tta_index, (tta_name, image_variant) in enumerate(tta_images):
                if tta_name == "orig":
                    image_ref = str(image_path)
                else:
                    tmp_path = materialize_variant_image(image_variant)
                    temp_files.append(tmp_path)
                    image_ref = str(tmp_path)

                prediction, raw_text = predict_once(
                    predictor,
                    image_ref,
                    prompt,
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
    finally:
        for tmp_path in temp_files:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    best = choose_best_candidate(candidates)
    return best, candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--vision-model-name-or-path", default="")
    parser.add_argument("--tokenizer-dir", default="")
    parser.add_argument("--ernie-dir", required=True)
    parser.add_argument("--benchmark-jsonl", default="")
    parser.add_argument("--image", default="")
    parser.add_argument("--project-root", default="")
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--prompt-text", default="")
    parser.add_argument("--prompt-list-file", default="")
    parser.add_argument("--max-dec-len", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tta-preset", choices=["none", "light"], default="none")
    parser.add_argument("--save-candidates", action="store_true")
    parser.add_argument("--device", default="mlu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--penalty-score", type=float, default=1.0)
    parser.add_argument("--frequency-score", type=float, default=0.0)
    parser.add_argument("--presence-score", type=float, default=0.0)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--rope-3d", type=int, default=1)
    parser.add_argument("--variable-resolution", type=str2bool, default=True)
    parser.add_argument("--use-flash-attn", type=str2bool, default=False)
    parser.add_argument("--crop-tile-option", default="")
    parser.add_argument("--crop-tile-rate", default="")
    parser.add_argument("--max-pixels", type=int, default=0)
    parser.add_argument("--min-pixels", type=int, default=-1)
    parser.add_argument("--chat-template", default="ernie_vl")
    parser.add_argument("--image-dtype", default="float32")
    parser.add_argument("--use-pic-id", type=str2bool, default=False)
    parser.add_argument("--sft-replace-ids", type=str2bool, default=True)
    parser.add_argument("--sft-image-rescale", type=str2bool, default=True)
    parser.add_argument("--sft-image-normalize", type=str2bool, default=True)
    parser.add_argument("--set-mlu-env", type=str2bool, default=True)
    args = parser.parse_args()

    prompt_file = Path(args.prompt_file).resolve() if args.prompt_file else None
    prompt_list_file = Path(args.prompt_list_file).resolve() if args.prompt_list_file else None
    project_root = Path(args.project_root).resolve() if args.project_root else None
    prompts = load_prompt_list(prompt_file, args.prompt_text, prompt_list_file)

    predictor = NativePredictor(args)

    if args.image:
        image_path = Path(args.image).resolve()
        best, candidates = predict_with_ensemble(
            predictor,
            image_path,
            prompts,
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

    handle = output_path.open("w", encoding="utf-8") if output_path else None
    try:
        for record in records:
            image_path = resolve_runtime_image_path(
                record["image_path"],
                project_root,
                benchmark_path,
            )
            best, candidates = predict_with_ensemble(
                predictor,
                image_path,
                prompts,
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
