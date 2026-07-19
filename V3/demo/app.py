#!/usr/bin/env python3
"""GPU-backed Gradio demo for the V3 OCSR model."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import gradio as gr


V3_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = V3_ROOT.parent
sys.path.insert(0, str(V3_ROOT / "scripts"))

from infer_ocsr_transformers import (  # noqa: E402
    build_generation_kwargs,
    load_generation_model,
    predict_with_ensemble,
    resolve_device,
    resolve_torch_dtype,
)


PROMPT = "OCR: Output only the canonical SMILES string for the molecule shown in the image."
MODEL_DIR = Path(
    os.environ.get("V3_MODEL_DIR", V3_ROOT / "models" / "v2_1_export")
).resolve()


class Runtime:
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.torch = None
        self.lock = threading.Lock()

    def load(self) -> None:
        if self.model is not None:
            return
        with self.lock:
            if self.model is not None:
                return
            import torch
            from transformers import AutoProcessor

            if not MODEL_DIR.exists():
                raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
            device = resolve_device(torch, "cuda")
            if not str(device).startswith("cuda"):
                raise RuntimeError("The V3 demo requires a CUDA GPU")
            dtype = resolve_torch_dtype(torch, "bfloat16", device)
            self.model = load_generation_model(
                str(MODEL_DIR),
                {
                    "trust_remote_code": True,
                    "torch_dtype": dtype,
                },
                torch,
            ).to(device).eval()
            self.processor = AutoProcessor.from_pretrained(
                str(MODEL_DIR), trust_remote_code=True
            )
            if hasattr(self.processor, "image_processor"):
                if hasattr(self.processor.image_processor, "min_pixels"):
                    self.processor.image_processor.min_pixels = 50176
                if hasattr(self.processor.image_processor, "max_pixels"):
                    self.processor.image_processor.max_pixels = 200704
            self.torch = torch

    def predict(
        self,
        image_path: str | None,
        beams: int,
        max_new_tokens: int,
        light_tta: bool,
    ):
        if not image_path:
            raise gr.Error("Select an image first")
        self.load()
        generation = build_generation_kwargs(
            max_new_tokens=int(max_new_tokens),
            num_beams=int(beams),
            num_return_sequences=1,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
        )
        with self.lock:
            with self.torch.inference_mode():
                best, candidates = predict_with_ensemble(
                    self.model,
                    self.processor,
                    Path(image_path),
                    [PROMPT],
                    generation,
                    "light" if light_tta else "none",
                )
        canonical = best.get("canonical_prediction")
        output = canonical or best.get("prediction", "")
        validity = "Valid RDKit SMILES" if canonical else "Invalid or non-canonical output"
        rows = [
            [
                index + 1,
                item.get("prediction", ""),
                item.get("canonical_prediction") or "",
                item.get("selection_reason", ""),
                item.get("vote_count", 0),
                item.get("smiles_structure_penalty", ""),
            ]
            for index, item in enumerate(candidates)
        ]
        return output, validity, best.get("selection_reason", ""), rows


runtime = Runtime()

with gr.Blocks(title="PaddleOCR-VL OCSR V3") as demo:
    gr.Markdown("# PaddleOCR-VL OCSR V3")
    with gr.Row():
        image = gr.Image(type="filepath", label="Molecular structure image", height=460)
        with gr.Column():
            smiles = gr.Code(label="Canonical SMILES", language=None, lines=6)
            validity = gr.Textbox(label="Validity", interactive=False)
            selection = gr.Textbox(label="Selection", interactive=False)
            with gr.Row():
                beams = gr.Slider(1, 4, value=1, step=1, label="Beams")
                max_tokens = gr.Slider(64, 512, value=256, step=32, label="Max tokens")
            light_tta = gr.Checkbox(value=False, label="Light TTA")
            with gr.Row():
                run = gr.Button("Run OCSR", variant="primary")
                clear = gr.ClearButton([image, smiles, validity, selection])
    candidates = gr.Dataframe(
        headers=["#", "Prediction", "Canonical", "Selection", "Votes", "Penalty"],
        datatype=["number", "str", "str", "str", "number", "number"],
        interactive=False,
        label="Candidates",
        wrap=True,
    )
    run.click(
        runtime.predict,
        inputs=[image, beams, max_tokens, light_tta],
        outputs=[smiles, validity, selection, candidates],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )
