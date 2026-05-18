# Cloud 4090 Paddle/PaddleFormers Setup

This setup is written for the current cloud root:

```text
/data/coding/data
```

All commands below assume the project root is exactly `/data/coding/data`.

## 1. Create a clean Python environment

### Option A: conda (preferred)

```bash
conda create -n paddleocr_v2 python=3.10 -y
conda activate paddleocr_v2
python -m pip install -U pip setuptools wheel
```

### Option B: venv fallback

```bash
python3.10 -m venv /data/coding/paddleocr_v2_env
source /data/coding/paddleocr_v2_env/bin/activate
python -m pip install -U pip setuptools wheel
```

## 2. Install PaddleFormers with GPU support

The 4090 host reports CUDA 12.4. Use the official cu126 wheels as the closest supported stable GPU line.

```bash
python -m pip install "paddleformers[paddlefleet]" --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

## 3. Optional NVIDIA acceleration packages

```bash
pip install triton==3.6.0
pip install use-triton-in-paddle==0.1.0
```

If these introduce issues, remove them rather than debugging first:

```bash
pip uninstall triton -y
pip uninstall use-triton-in-paddle -y
```

## 4. Verify the environment

```bash
cd /data/coding/data
nvidia-smi
python --version
python -c "import paddle; print('paddle=', paddle.__version__)"
python -c "import paddle; paddle.set_device('gpu'); print('device=', paddle.get_device())"
python -c "import paddleformers; print('paddleformers ok')"
paddleformers-cli --help | head
```

Expected:

- `paddle` imports successfully
- GPU device resolves to `gpu:0`
- `paddleformers-cli` is callable

## 5. Rebuild V2 local data products on the cloud host

```bash
cd /data/coding/data
bash V2/run_4090_preflight.sh
```

## 6. Start Phase 1 LoRA training

```bash
cd /data/coding/data
bash V2/run_4090_lora_phase1.sh
```

## 7. After training finishes

Inspect:

```text
/data/coding/data/V2/outputs/phase1_lora/train_phase1.log
/data/coding/data/V2/outputs/phase1_lora/
```

Then follow the evaluation checklist in:

```text
V2/runbooks/4090_lora_v1.md
```
