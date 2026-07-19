# V3 Demo

在已配置 CUDA、PyTorch、Transformers 和 Gradio 的 A100 环境中运行：

```bash
export V3_MODEL_DIR="$PWD/V3/models/final_selected_export"
export GRADIO_SERVER_PORT=7860
python V3/demo/app.py
```

如果最终模型尚未导出，可先用默认的 `V3/models/v2_1_export` 做功能验证。服务启动后访问 `http://<A100-host>:7860`。

Demo 复用 `V3/scripts/infer_ocsr_transformers.py` 的模型加载、候选生成和选择逻辑，不维护第二套推理实现。界面显示最终 canonical SMILES、RDKit validity、选择原因和完整候选表。
