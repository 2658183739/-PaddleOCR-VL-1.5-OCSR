#!/usr/bin/env bash
set -euo pipefail

PADDLEFORMERS_COMMIT="e51f911c23b41283ef6c62f8aa4a7e99291bcd11"
PADDLE_CU126_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu126/"
PYTORCH_CU118_INDEX="https://download.pytorch.org/whl/cu118"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --force-reinstall \
  paddlepaddle-gpu==3.3.1 \
  -i "$PADDLE_CU126_INDEX"

# PaddleFormers is pinned to the exact VL-SFT commit used by the experiments.
python -m pip install \
  "git+https://github.com/PaddlePaddle/PaddleFormers.git@${PADDLEFORMERS_COMMIT}"

# Torch is used only by the independent generation/evaluation processes.
python -m pip install torch==2.1.2 --index-url "$PYTORCH_CU118_INDEX"
python -m pip install \
  transformers==4.55.4 \
  numpy==1.26.4 \
  Pillow==10.3.0 \
  protobuf==6.33.4 \
  rdkit==2025.9.6 \
  safetensors==0.7.0 \
  einops==0.8.2 \
  binpacking==2.0.1 \
  importlib-metadata==9.0.0 \
  orjson==3.11.9 \
  paddlecodec==0.2.0

python - <<'PY'
import paddle
import torch

paddle.set_device("gpu")
print("paddle", paddle.__version__, paddle.version.cuda(), paddle.get_device())
print(paddle.randn([2, 2]))
print("torch", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print(torch.randn((2, 2), device="cuda"))
PY

python -m pip check || true
