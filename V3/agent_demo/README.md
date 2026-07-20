# MolTrace OCSR Agent

这是 PaddleOCR-VL OCSR V3 的可审计前后端工作台。它把识别拆成输入校验、图像质检、候选生成、RDKit 校验、一致性排序和证据汇总六步，并保留候选级结果，而不是把一次生成包装成不可解释的答案。

## 快速启动

只需要 Node.js 18+，无需安装 npm 依赖：

```bash
cd V3/agent_demo
npm start
```

打开 `http://127.0.0.1:8787`，点击“载入示例”可体验完整引导演示。演示结果使用咖啡因已知标签并在界面中明确标注，不作为模型在线推理证据；对任意上传图，在模型不可用时系统会拒绝伪造结果。

## 接入 V3 GPU 模型

```bash
set V3_MODEL_DIR=D:\models\final_best_export
set PYTHON_BIN=python
set V3_DEVICE=cuda
npm start
```

后端会固定调用 `../scripts/infer_ocsr_transformers.py`。Python 环境需安装 V3 模型所需的 `torch`、`transformers`、`Pillow`、`safetensors` 和 `rdkit`。生产服务建议把模型改成长驻进程；当前适配器以每请求独立 CLI 进程为边界，便于比赛材料复现和失败隔离。

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | 服务、模型与历史状态 |
| `GET` | `/api/model` | 版本锁定的模型和评测证据 |
| `POST` | `/api/agent/run` | 执行六步 OCSR Agent |
| `POST` | `/api/validate` | 无 RDKit 时的轻量词法预检 |
| `GET` | `/api/history` | 最近 20 条结果摘要 |
| `DELETE` | `/api/history` | 清空内存历史 |

默认不持久化上传原图。真实推理期间只在系统临时目录写入输入，并在子进程退出后删除；历史仅保存 SHA-256 前 12 位指纹、文件名、模式、结果和耗时。

## 测试

```bash
npm test
```

测试覆盖图片输入约束、SMILES 轻量检查、参数边界和引导演示的审计轨迹。模型精度不由 Web 单元测试证明，正式指标仍以冻结的 evaluation run 和 `V3/evidence/FINAL_RESULTS.json` 为准。
